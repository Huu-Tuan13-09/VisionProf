"""
test_layer_flops_plugin.py — Kiểm tra FLOPs, plugin `layer` và phân tích mức Layer (TV1)
=========================================================================================
Chạy:
    pytest tests/test_layer_flops_plugin.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.flops import add_efficiency, analyze_layers, model_total_flops  # noqa: E402
from src.hw_specs import HardwareSpec, detect_hardware  # noqa: E402
from src.layer_analysis import compare_top_k, compare_type_share, top_k_layers, type_share  # noqa: E402
from src.layer_plugin import LayerPlugin  # noqa: E402

B, H, W = 2, 16, 16


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc(self.flat(self.pool(self.relu(self.bn(self.conv(x))))))


SPEC = HardwareSpec("test_hw", "Test", peak_gflops_fp32=100.0, peak_gflops_fp16=100.0,
                    bandwidth_gbs=10.0, source="test")


@pytest.fixture
def analysis():
    return analyze_layers(SmallCNN().eval(), torch.randn(B, 3, H, W)).set_index("layer_name")


# ── FLOPs ────────────────────────────────────────────────────────────────────

def test_conv_and_linear_flops_match_formula(analysis):
    conv_flops = 2 * B * 8 * H * W * (3 * 3 * 3)          # 2·B·Cout·H·W·(Cin·K·K)
    assert analysis.loc["conv", "flops"] == pytest.approx(conv_flops)
    assert analysis.loc["conv", "flops_source"] == "counter"
    assert analysis.loc["fc", "flops"] == pytest.approx(2 * B * 8 * 4)


def test_elementwise_layers_use_analytic_estimate(analysis):
    assert analysis.loc["bn", "flops_source"] == "analytic"
    assert analysis.loc["bn", "flops"] == pytest.approx(2 * B * 8 * H * W)
    assert analysis.loc["relu", "flops"] == pytest.approx(B * 8 * H * W)
    assert analysis.loc["flat", "flops_source"] == "none"


def test_bytes_and_intensity(analysis):
    conv = analysis.loc["conv"]
    assert conv["bytes_in"] == B * 3 * H * W * 4
    assert conv["bytes_out"] == B * 8 * H * W * 4
    assert conv["bytes_params"] == (8 * 3 * 3 * 3 + 8) * 4
    assert conv["arithmetic_intensity"] == pytest.approx(conv["flops"] / conv["bytes_total"])


def test_total_flops_counts_only_counter_by_default(analysis):
    df = analysis.reset_index()
    assert model_total_flops(df) == pytest.approx(analysis.loc["conv", "flops"] + analysis.loc["fc", "flops"])
    assert model_total_flops(df, "all") > model_total_flops(df)


def test_efficiency_and_bound(analysis):
    times = pd.DataFrame({
        "layer_name": ["conv", "bn"], "layer_type": ["Conv2d", "BatchNorm2d"],
        "call_index": [0, 0], "forward_time_ms": [0.01, 0.01],
    })
    eff = add_efficiency(times, analysis.reset_index(), SPEC).set_index("layer_name")
    conv = eff.loc["conv"]
    assert conv["achieved_gflops"] == pytest.approx(conv["gflops"] / 1e-5)
    expected_attainable = min(SPEC.peak_gflops_fp32, SPEC.bandwidth_gbs * conv["arithmetic_intensity"])
    assert conv["attainable_gflops"] == pytest.approx(expected_attainable)
    assert conv["bound"] == ("compute" if conv["arithmetic_intensity"] >= SPEC.ridge_point() else "memory")
    assert eff.loc["bn", "bound"] == "memory"


class AttnBlock(nn.Module):
    def __init__(self, e: int = 32, heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(e, heads, batch_first=True)
        self.norm = nn.LayerNorm(e)

    def forward(self, x):
        return self.norm(self.attn(x, x, x, need_weights=False)[0])


def test_multihead_attention_is_one_block_with_full_flops():
    b, l, e = 2, 10, 32
    model, x = AttnBlock(e).eval(), torch.randn(b, l, e)
    a = analyze_layers(model, x).set_index("layer_name")
    # Đo nguyên khối: không có dòng riêng cho attn.out_proj (tránh đếm trùng)
    assert "attn" in a.index and "attn.out_proj" not in a.index
    # Chiếu Q,K,V + output (8·B·L·E²) + QKᵀ và ×V (4·B·L·L·E)
    assert a.loc["attn", "flops"] == pytest.approx(8 * b * l * e * e + 4 * b * l * l * e)


def test_detect_cpu_spec_is_positive():
    spec = detect_hardware("cpu")
    assert spec.hw_id == "cpu_laptop" and spec.peak_gflops_fp32 > 0 and spec.bandwidth_gbs > 0


# ── Plugin ───────────────────────────────────────────────────────────────────

def test_plugin_end_to_end_writes_contract_files(tmp_path):
    model, x = SmallCNN().eval(), torch.randn(B, 3, H, W)
    plugin = LayerPlugin(hw_spec=SPEC)
    plugin.attach(model, "cpu")
    with torch.no_grad():
        for i in range(3):
            plugin.start(i)
            model(x)
            plugin.stop()
    paths = plugin.save(tmp_path)

    layers = pd.read_csv(paths["layers"])
    assert list(layers.columns[:7]) == [
        "iteration", "layer_name", "layer_type", "time_ms", "activation_mb", "flops", "params",
    ]
    assert len(layers) == 3 * 6
    assert layers.loc[layers["layer_name"] == "conv", "flops"].iloc[0] > 0

    summary = pd.read_csv(paths["summary"])
    assert {"efficiency_pct", "bound", "achieved_gflops"} <= set(summary.columns)

    meta = json.loads(Path(paths["meta"]).read_text(encoding="utf-8"))
    assert meta["n_hooked_layers"] == 6 and meta["n_iterations"] == 3
    assert meta["total_flops_counter"] > 0

    # Sau save() model không còn hook nào
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())


def test_plugin_skips_non_torch_model(tmp_path):
    plugin = LayerPlugin()
    plugin.attach(object(), "cpu")
    plugin.start(0)
    plugin.stop()
    assert plugin.save(tmp_path) == {}


# ── Phân tích ────────────────────────────────────────────────────────────────

def _fake_layers(times: dict) -> pd.DataFrame:
    rows = []
    for it in range(3):
        for order, (name, (ltype, t)) in enumerate(times.items()):
            rows.append({"iteration": it, "layer_name": name, "layer_type": ltype, "call_index": 0,
                         "exec_order": order, "time_ms": t, "activation_mb": 1.0})
    return pd.DataFrame(rows)


def test_top_k_and_type_share():
    df = _fake_layers({"a": ("Conv2d", 5.0), "b": ("Conv2d", 3.0), "c": ("ReLU", 2.0)})
    top = top_k_layers(df, k=2)
    assert top["layer_name"].tolist() == ["a", "b"] and top["rank"].tolist() == [1, 2]
    share = type_share(df).set_index("layer_type")
    assert share.loc["Conv2d", "pct"] == pytest.approx(80.0)
    assert share["pct"].sum() == pytest.approx(100.0)


def test_compare_cpu_gpu_shows_rank_change():
    cpu = _fake_layers({"conv": ("Conv2d", 10.0), "fc": ("Linear", 1.0)})
    gpu = _fake_layers({"conv": ("Conv2d", 0.1), "fc": ("Linear", 0.5)})
    cmp = compare_top_k(cpu, gpu, k=10).set_index("layer_name")
    assert cmp.loc["conv", "rank_cpu"] == 1 and cmp.loc["conv", "rank_gpu"] == 2
    assert cmp.loc["conv", "speedup"] == pytest.approx(100.0)
    share = compare_type_share(cpu, gpu).set_index("layer_type")
    assert share.loc["Linear", "pct_change"] > 0
