"""Unit tests verifying scientific profiling rigor of the layer profiler (src/layer)."""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

from typing import Any, Callable, Optional

try:
    import pytest
except ImportError:
    class _MockApprox:
        def __init__(self, val: Any, rel: Optional[float] = None, abs: Optional[float] = None) -> None:
            self.val, self.rel, self.abs = val, rel, abs
        def __eq__(self, other: Any) -> bool:
            d = abs(self.val - other)
            return (self.abs is not None and d <= self.abs) or (self.rel is not None and self.val != 0 and d / abs(self.val) <= self.rel) or d < 1e-4

    class _MockPytest:
        approx = staticmethod(lambda v, rel=None, abs=None: _MockApprox(v, rel, abs))
        class mark:
            skipif = staticmethod(lambda cond, reason="": (lambda fn: fn))

    pytest = _MockPytest()  # type: ignore

try:
    import torch
    import torch.nn as nn
    HAS_TORCH, CUDA = True, torch.cuda.is_available()
except ImportError:
    HAS_TORCH, CUDA = False, False
    class _MockModule: pass
    class _MockNN: Module = _MockModule
    torch, nn = None, _MockNN()  # type: ignore

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from src.layer.overhead import calibrate_overhead_details, estimate_timer_bias  # noqa: E402
    from src.layer.plugin import LayerProfilerPlugin  # noqa: E402
except ImportError:
    LayerProfilerPlugin = None  # type: ignore
    calibrate_overhead_details = None  # type: ignore
    estimate_timer_bias = None  # type: ignore


if HAS_TORCH:
    class Sleep(nn.Module):
        def __init__(self, ms: float) -> None:
            super().__init__()
            self.ms = ms
            self.actual_ms = 0.0
        def forward(self, x: Any) -> Any:
            t0 = time.perf_counter()
            time.sleep(self.ms / 1000.0)
            self.actual_ms = (time.perf_counter() - t0) * 1000.0
            return x + 0

    class SharedBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(16, 16)
        def forward(self, x: Any) -> Any:
            return self.fc(self.fc(x))

    class MultiOut(nn.Module):
        def forward(self, x: Any) -> Any:
            return x * 1, x * 2, x * 3

    class SmallCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)
            self.bn = nn.BatchNorm2d(8)
            self.relu = nn.ReLU(inplace=True)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.flat = nn.Flatten()
            self.drop = nn.Dropout(0.1)
            self.fc = nn.Linear(8, 4)
        def forward(self, x: Any) -> Any:
            x = self.relu(self.bn(self.conv(x)))
            return self.fc(self.drop(self.flat(self.pool(x))))
else:
    Sleep = SharedBlock = MultiOut = SmallCNN = None  # type: ignore


def _profile(model: Any, x: Any, device: str = "cpu", n_iter: int = 1, bias: bool = False) -> Any:
    plugin = LayerProfilerPlugin(subtract_timer_bias=bias)
    plugin.attach(model, device)
    with torch.no_grad():
        for i in range(n_iter):
            plugin.start(i)
            model(x)
            plugin.stop()
    return plugin


def test_cpu_timing_matches_known_sleep() -> None:
    if not HAS_TORCH or LayerProfilerPlugin is None: return
    model = nn.Sequential(Sleep(20), Sleep(5))
    df = _profile(model, torch.zeros(4), n_iter=2).to_dataframe()   # vòng 0 là khởi động
    df = df[df["iteration"] == 1]
    t = dict(zip(df["layer_name"], df["raw_time_ms"]))
    for name, layer in zip(["0", "1"], model):
        assert t[name] >= layer.ms * 0.95 and t[name] >= layer.actual_ms and t[name] - layer.actual_ms < 2.0
    assert set(df["timing_backend"]) == {"cpu_perf_counter"}


def test_records_one_per_hooked_layer_per_iteration() -> None:
    if not HAS_TORCH or LayerProfilerPlugin is None: return
    plugin = _profile(SmallCNN().eval(), torch.randn(2, 3, 16, 16), n_iter=3)
    df = plugin.to_dataframe()
    assert plugin.n_hooked == 6
    assert "ReLU" in set(df["layer_type"])
    assert "Dropout" not in set(df["layer_type"])
    assert df.groupby("iteration").size().tolist() == [6, 6, 6]
    assert set(plugin.get_iteration_totals()) == {0, 1, 2}


def test_timer_bias_is_subtracted() -> None:
    if not HAS_TORCH or LayerProfilerPlugin is None: return
    model = nn.Sequential(Sleep(5))
    plugin = LayerProfilerPlugin()
    plugin.attach(model, "cpu")
    plugin.timer_bias_ms = 1.0
    plugin.start(0)
    model(torch.zeros(1))
    plugin.stop()
    rec = plugin.to_dataframe().iloc[0]
    assert rec["time_ms"] == pytest.approx(rec["raw_time_ms"] - 1.0, abs=1e-4)


def test_estimate_timer_bias_is_tiny_on_cpu() -> None:
    if not HAS_TORCH or estimate_timer_bias is None: return
    bias = estimate_timer_bias("cpu", n_layers=16, n_runs=10)
    assert 0.0 <= bias < 0.5


def test_reentrant_module_gets_one_record_per_call() -> None:
    if not HAS_TORCH or LayerProfilerPlugin is None: return
    df = _profile(SharedBlock(), torch.randn(2, 16)).to_dataframe()
    fc = df[df["layer_name"] == "fc"]
    assert len(fc) == 2
    assert sorted(fc["call_index"]) == [0, 1]
    assert (fc["raw_time_ms"] > 0).all()


def test_activation_memory_is_output_size() -> None:
    if not HAS_TORCH or LayerProfilerPlugin is None: return
    df = _profile(SmallCNN().eval(), torch.randn(2, 3, 16, 16)).to_dataframe()
    conv = df[df["layer_name"] == "conv"].iloc[0]
    assert conv["activation_mb"] == pytest.approx(2 * 8 * 16 * 16 * 4 / 1024**2, rel=1e-3)


def test_inplace_relu_and_view_are_not_double_counted() -> None:
    if not HAS_TORCH or LayerProfilerPlugin is None: return
    df = _profile(SmallCNN().eval(), torch.randn(2, 3, 16, 16)).to_dataframe()
    act = dict(zip(df["layer_name"], df["activation_mb"]))
    assert act["relu"] == 0.0
    assert act["flat"] == 0.0


def test_multi_output_layer_counts_all_tensors() -> None:
    if not HAS_TORCH or LayerProfilerPlugin is None: return
    x = torch.randn(4, 8)
    df = _profile(nn.Sequential(MultiOut()), x).to_dataframe()
    one_tensor_mb = x.numel() * x.element_size() / 1024**2
    assert df.iloc[0]["activation_mb"] == pytest.approx(3 * one_tensor_mb, rel=1e-3)


def test_calibrate_counts_all_hooked_layers() -> None:
    if not HAS_TORCH or calibrate_overhead_details is None: return
    details = calibrate_overhead_details(SmallCNN().eval(), torch.randn(2, 3, 16, 16), n_warmup=2, n_measure=5)
    assert details["n_hooked"] == 6
    assert details["t_clean_ms"] > 0 and details["t_hooked_ms"] > 0


@pytest.mark.skipif(not CUDA, reason="Không có GPU CUDA")
def test_cuda_layer_time_matches_reference_event() -> None:
    if not CUDA or not HAS_TORCH or LayerProfilerPlugin is None: return
    layer = nn.Linear(4096, 4096).cuda().eval()
    x = torch.randn(256, 4096, device="cuda")
    model = nn.Sequential(layer)

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    plugin = LayerProfilerPlugin()
    refs = []
    with torch.no_grad():
        for _ in range(10):
            model(x)
        plugin.attach(model, "cuda")
        for i in range(20):
            torch.cuda.synchronize()
            start.record()
            layer(x)          # hook không ghi ngoài start()/stop()
            end.record()
            torch.cuda.synchronize()
            refs.append(start.elapsed_time(end))
            plugin.start(i)
            model(x)
            plugin.stop()
        plugin.detach()

    df = plugin.to_dataframe()
    measured = float(df["raw_time_ms"].median())
    reference = float(torch.tensor(refs).median())
    assert len(df) == 20
    assert set(df["timing_backend"]) == {"cuda_event"}
    assert measured == pytest.approx(reference, rel=0.15)


@pytest.mark.skipif(not CUDA, reason="Không có GPU CUDA")
def test_cuda_sum_of_layers_not_more_than_total() -> None:
    if not CUDA or not HAS_TORCH or LayerProfilerPlugin is None: return
    gc.collect()
    torch.cuda.empty_cache()
    x = torch.randn(8, 3, 64, 64, device="cuda")
    plugin = _profile(SmallCNN().cuda().eval(), x, device="cuda", n_iter=5)
    df = plugin.to_dataframe()
    totals = plugin.get_iteration_totals()
    for it, group in df.groupby("iteration"):
        assert group["raw_time_ms"].sum() <= totals[it] * 1.01
    conv = df[df["layer_name"] == "conv"].iloc[0]
    assert conv["activation_mb"] == pytest.approx(8 * 8 * 64 * 64 * 4 / 1024**2, rel=1e-3)
    assert set(df["timing_backend"]) == {"cuda_event"}
