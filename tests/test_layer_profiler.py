"""
test_layer_profiler.py — Kiểm tra LayerProfiler v2 (TV1 · Mức Layer)
=====================================================================
Mỗi test ứng với một lỗi đã sửa ở GĐ1:
    1. Đo thời gian đúng trên CPU (và GPU nếu có)
    2. Module gọi nhiều lần trong 1 forward không bị ghi đè
    3. Bộ nhớ layer không bị trộn với RAM tiến trình
    4. Output nhiều tensor được cộng đủ; ReLU inplace không bị đếm trùng
    5. calibrate_overhead đếm đúng số layer có hook

Chạy:
    pytest tests/test_layer_profiler.py -v
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.collector import (  # noqa: E402
    LayerProfiler,
    calibrate_overhead_details,
    estimate_timer_bias,
)

CUDA = torch.cuda.is_available()


# ─────────────────────────────────────────────────────────────────────────────
# Model nhỏ dùng để test
# ─────────────────────────────────────────────────────────────────────────────

class Sleep(nn.Module):
    """Layer ngủ đúng `ms` mili-giây — biết trước thời gian thật để kiểm tra."""

    def __init__(self, ms: float):
        super().__init__()
        self.ms = ms

    def forward(self, x):
        time.sleep(self.ms / 1000.0)
        return x + 0


class SharedBlock(nn.Module):
    """Gọi cùng một layer 2 lần trong 1 forward (giống module dùng chung trọng số)."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 16)

    def forward(self, x):
        return self.fc(self.fc(x))


class MultiOut(nn.Module):
    """Leaf layer trả về 3 tensor (giống head nhiều tầng của YOLO)."""

    def forward(self, x):
        return x * 1, x * 2, x * 3


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(0.1)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return self.fc(self.drop(self.flat(self.pool(x))))


def _profile(model, x, device="cpu", iteration=0):
    profiler = LayerProfiler(model, device=device)
    with torch.no_grad(), profiler.profile_context(iteration=iteration):
        model(x)
    return profiler


# ─────────────────────────────────────────────────────────────────────────────
# 1. Thời gian
# ─────────────────────────────────────────────────────────────────────────────

def test_cpu_timing_matches_known_sleep():
    model = nn.Sequential(Sleep(20), Sleep(5))
    df = _profile(model, torch.zeros(4)).to_dataframe()
    t = dict(zip(df["layer_name"], df["raw_time_ms"]))
    assert 19.0 <= t["0"] <= 40.0
    assert 4.5 <= t["1"] <= 20.0
    assert set(df["timing_backend"]) == {"cpu_perf_counter"}


def test_records_one_per_hooked_layer_per_iteration():
    model = SmallCNN().eval()
    profiler = LayerProfiler(model)
    profiler.attach_hooks()
    with torch.no_grad():
        for i in range(3):
            with profiler.profile_context(iteration=i):
                model(torch.randn(2, 3, 16, 16))
    df = profiler.to_dataframe()
    # Dropout bị bỏ qua mặc định, ReLU thì được đo (cần cho tỷ trọng Activation ở TN1)
    assert profiler.n_hooked == 6
    assert "ReLU" in set(df["layer_type"])
    assert "Dropout" not in set(df["layer_type"])
    assert df.groupby("iteration").size().tolist() == [6, 6, 6]
    assert set(profiler.get_iteration_totals()) == {0, 1, 2}


def test_timer_bias_is_subtracted():
    model = nn.Sequential(Sleep(5))
    profiler = LayerProfiler(model)
    profiler.set_overhead(1.0)
    with profiler.profile_context():
        model(torch.zeros(1))
    rec = profiler.get_records()[0]
    assert rec.forward_time_ms == pytest.approx(rec.raw_time_ms - 1.0, abs=1e-4)


def test_estimate_timer_bias_is_tiny_on_cpu():
    bias = estimate_timer_bias("cpu", n_layers=16, n_runs=10)
    assert 0.0 <= bias < 0.5  # dưới 0.5ms mỗi layer


# ─────────────────────────────────────────────────────────────────────────────
# 2. Module gọi nhiều lần
# ─────────────────────────────────────────────────────────────────────────────

def test_reentrant_module_gets_one_record_per_call():
    df = _profile(SharedBlock(), torch.randn(2, 16)).to_dataframe()
    fc = df[df["layer_name"] == "fc"]
    assert len(fc) == 2
    assert sorted(fc["call_index"]) == [0, 1]
    assert (fc["raw_time_ms"] > 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# 3 & 4. Bộ nhớ
# ─────────────────────────────────────────────────────────────────────────────

def test_cpu_memory_not_mixed_with_process_rss():
    model = SmallCNN().eval()
    df = _profile(model, torch.randn(2, 3, 16, 16)).to_dataframe()
    conv = df[df["layer_name"] == "conv"].iloc[0]
    expected_mb = 2 * 8 * 16 * 16 * 4 / 1024**2
    assert conv["activation_mb"] == pytest.approx(expected_mb, rel=1e-3)
    assert conv["gpu_mem_before_mb"] == 0.0
    assert conv["gpu_mem_delta_mb"] == pytest.approx(expected_mb, abs=1e-3)
    assert conv["process_rss_mb"] > 50  # RAM tiến trình nằm ở cột riêng


def test_inplace_relu_and_view_are_not_double_counted():
    df = _profile(SmallCNN().eval(), torch.randn(2, 3, 16, 16)).to_dataframe()
    act = dict(zip(df["layer_name"], df["activation_mb"]))
    assert act["relu"] == 0.0   # inplace: ghi đè lên output của bn
    assert act["flat"] == 0.0   # view: dùng chung vùng nhớ với input


def test_multi_output_layer_counts_all_tensors():
    x = torch.randn(4, 8)
    df = _profile(nn.Sequential(MultiOut()), x).to_dataframe()
    one_tensor_mb = x.numel() * x.element_size() / 1024**2
    assert df.iloc[0]["activation_mb"] == pytest.approx(3 * one_tensor_mb, rel=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# 5. calibrate_overhead
# ─────────────────────────────────────────────────────────────────────────────

def test_calibrate_counts_all_hooked_layers():
    details = calibrate_overhead_details(
        SmallCNN(), torch.randn(2, 3, 16, 16), n_warmup=2, n_measure=5
    )
    # v1 chỉ đếm layer có tham số (conv, bn, fc = 3) → sai. v2 đếm đủ 6 layer có hook.
    assert details["n_hooked"] == 6
    assert details["t_clean_ms"] > 0 and details["t_hooked_ms"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# GPU (tự bỏ qua nếu máy không có CUDA)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not CUDA, reason="Không có GPU CUDA")
def test_cuda_layer_time_matches_reference_event():
    layer = nn.Linear(4096, 4096).cuda().eval()
    x = torch.randn(256, 4096, device="cuda")
    model = nn.Sequential(layer)

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    profiler = LayerProfiler(model, device="cuda")
    refs = []
    with torch.no_grad():
        for _ in range(10):
            model(x)
        # Đo XEN KẼ tham chiếu (CUDA event trực tiếp, không hook) và profiler,
        # để xung nhịp GPU thay đổi không làm lệch một bên
        for i in range(20):
            torch.cuda.synchronize()
            start.record()
            layer(x)
            end.record()
            torch.cuda.synchronize()
            refs.append(start.elapsed_time(end))

            profiler.attach_hooks()
            with profiler.profile_context(iteration=i):
                model(x)
            profiler.remove_hooks()

    df = profiler.to_dataframe()
    measured = float(df["raw_time_ms"].median())
    reference = float(torch.tensor(refs).median())
    assert set(df["timing_backend"]) == {"cuda_event"}
    assert measured == pytest.approx(reference, rel=0.15)


@pytest.mark.skipif(not CUDA, reason="Không có GPU CUDA")
def test_cuda_sum_of_layers_not_more_than_total():
    gc.collect()                 # dọn tensor còn sót từ test trước
    torch.cuda.empty_cache()
    model = SmallCNN().cuda().eval()
    x = torch.randn(8, 3, 64, 64, device="cuda")
    profiler = LayerProfiler(model, device="cuda")
    profiler.attach_hooks()
    with torch.no_grad():
        for i in range(5):
            with profiler.profile_context(iteration=i):
                model(x)
    df = profiler.to_dataframe()
    totals = profiler.get_iteration_totals()
    for it, group in df.groupby("iteration"):
        assert group["raw_time_ms"].sum() <= totals[it] * 1.01
    # memory_allocated là bộ nhớ cả chương trình (bị ảnh hưởng khi Python dọn rác) →
    # kiểm tra activation_mb, chỉ tính output của chính layer
    conv = df[df["layer_name"] == "conv"].iloc[0]
    assert conv["activation_mb"] == pytest.approx(8 * 8 * 64 * 64 * 4 / 1024**2, rel=1e-3)
    assert set(df["timing_backend"]) == {"cuda_event"}
