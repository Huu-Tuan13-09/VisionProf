"""
flops.py — FLOPs, lưu lượng bộ nhớ & hiệu suất từng layer (TV1 · Mức Layer)
============================================================================
Thành phần:
  - analyze_layers      : Chạy model 1 lần, đếm FLOPs + số byte đọc/ghi của từng leaf layer
  - add_efficiency      : Ghép với thời gian đo của LayerProfiler → GFLOP/s đạt được,
                          arithmetic intensity, % hiệu suất (Roofline), compute/memory-bound
  - run_model           : Gọi model với input dạng Tensor / tuple / dict

Nguồn FLOPs:
  - "counter" : torch.utils.flop_counter.FlopCounterMode (Conv, Linear, MatMul, Attention…)
  - "analytic": ước lượng theo số phần tử cho các layer FlopCounterMode không đếm
                (BatchNorm, LayerNorm, ReLU, GELU, Pooling, Softmax…)
  - "none"    : layer không tính toán (Flatten, Identity…)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode, sdpa_flop_count

from src.collector import LayerProfiler, _iter_tensors
from src.hw_specs import HardwareSpec

aten = torch.ops.aten


def _native_mha_flops(query, key, value, embed_dim, num_head, *args, out_shape=None, **kwargs) -> int:
    """
    nn.MultiheadAttention chạy đường tắt (fast path, batch_first) — query/key: (B, L, E) / (B, S, E).
    Chiếu Q (2BLE²) + chiếu K,V (4BSE²) + QKᵀ và ×V (4BLSE) + chiếu output (2BLE²).
    """
    b, l, e = query
    s = key[1]
    return 4 * b * l * e * e + 4 * b * s * e * e + 4 * b * l * s * e


def _cpu_sdpa_flops(query, key, value, *args, out_shape=None, **kwargs) -> int:
    """scaled_dot_product_attention trên CPU — dùng công thức chuẩn của PyTorch."""
    return sdpa_flop_count(query, key, value)


# Các phép attention gộp mà FlopCounterMode (PyTorch 2.x) chưa đếm khi chạy trên CPU
_CUSTOM_FLOP_FORMULAS = {
    op: fn for op, fn in [
        (getattr(aten, "_native_multi_head_attention", None), _native_mha_flops),
        (getattr(aten, "_scaled_dot_product_flash_attention_for_cpu", None), _cpu_sdpa_flops),
    ] if op is not None
}

# Số FLOPs trên mỗi phần tử output, dùng khi FlopCounterMode trả về 0
_ANALYTIC_FLOPS_PER_ELEMENT: Dict[str, float] = {
    "BatchNorm1d": 2, "BatchNorm2d": 2, "BatchNorm3d": 2,
    "GroupNorm": 5, "InstanceNorm2d": 5, "LayerNorm": 5, "RMSNorm": 4,
    "ReLU": 1, "ReLU6": 1, "LeakyReLU": 1, "Hardtanh": 1, "PReLU": 2,
    "GELU": 8, "SiLU": 4, "Sigmoid": 4, "Tanh": 4, "Hardswish": 4, "Hardsigmoid": 3, "Mish": 6,
    "Softmax": 3, "LogSoftmax": 4,
    "MaxPool2d": 1, "AvgPool2d": 1, "AdaptiveAvgPool2d": 1, "AdaptiveMaxPool2d": 1,
    "Upsample": 1,
}


@dataclass
class CallArgs:
    """Bộ tham số gọi model đã chụp lại: model(*args, **kwargs)."""
    args:   tuple = ()
    kwargs: dict  = field(default_factory=dict)


def run_model(model: nn.Module, inputs: Any) -> Any:
    """Gọi model với input dạng Tensor, tuple/list (*args), dict (**kwargs, vd model HF) hoặc CallArgs."""
    if isinstance(inputs, CallArgs):
        return model(*inputs.args, **inputs.kwargs)
    if isinstance(inputs, dict):
        return model(**inputs)
    if isinstance(inputs, (tuple, list)):
        return model(*inputs)
    return model(inputs)


def _tensor_bytes(obj: Any) -> int:
    return sum(t.numel() * t.element_size() for t in _iter_tensors(obj))


def _numel(obj: Any) -> int:
    return sum(t.numel() for t in _iter_tensors(obj))


def analyze_layers(
    model: nn.Module,
    inputs: Any,
    exclude_types: Optional[Tuple[type, ...]] = None,
    min_param_threshold: int = 0,
) -> pd.DataFrame:
    """
    Chạy model 1 lần (không đo thời gian) để lấy FLOPs và lưu lượng bộ nhớ của từng leaf layer.
    Chọn layer giống hệt LayerProfiler để ghép được theo layer_name.

    Returns: DataFrame, mỗi dòng 1 layer, giá trị tính cho MỘT lần gọi:
        layer_name, layer_type, n_calls, flops, flops_source,
        bytes_in, bytes_out, bytes_params, bytes_total, arithmetic_intensity
    """
    selector = LayerProfiler(model, exclude_types=exclude_types, min_param_threshold=min_param_threshold)
    targets = selector._select_targets()

    stats: Dict[str, Dict[str, Any]] = {}
    handles = []

    def make_hook(name: str, module: nn.Module):
        params_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
        params_bytes += sum(b.numel() * b.element_size() for b in module.buffers())

        def hook(mod, inputs, output):
            s = stats.setdefault(name, {
                "layer_type": type(mod).__name__, "n_calls": 0,
                "bytes_in": 0, "bytes_out": 0, "bytes_params": params_bytes, "numel_out": 0,
            })
            s["n_calls"] += 1
            s["bytes_in"] += _tensor_bytes(inputs)
            s["bytes_out"] += _tensor_bytes(output)
            s["numel_out"] += _numel(output)
        return hook

    for name, module in targets:
        handles.append(module.register_forward_hook(make_hook(name, module)))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), FlopCounterMode(display=False, custom_mapping=_CUSTOM_FLOP_FORMULAS) as counter:
            run_model(model, inputs)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    counts = counter.get_flop_counts()
    root = type(model).__name__

    rows: List[Dict[str, Any]] = []
    for name, _ in targets:
        s = stats.get(name)
        if s is None:          # layer không được gọi trong forward
            continue
        n = s["n_calls"]
        key = root if name == "<root>" else f"{root}.{name}"
        counted = float(sum(counts.get(key, {}).values()))
        if counted > 0:
            flops, source = counted / n, "counter"
        elif s["layer_type"] in _ANALYTIC_FLOPS_PER_ELEMENT:
            flops = _ANALYTIC_FLOPS_PER_ELEMENT[s["layer_type"]] * s["numel_out"] / n
            source = "analytic"
        else:
            flops, source = 0.0, "none"

        bytes_in, bytes_out = s["bytes_in"] / n, s["bytes_out"] / n
        bytes_total = bytes_in + bytes_out + s["bytes_params"]
        rows.append({
            "layer_name":           name,
            "layer_type":           s["layer_type"],
            "n_calls":              n,
            "flops":                flops,
            "flops_source":         source,
            "bytes_in":             bytes_in,
            "bytes_out":            bytes_out,
            "bytes_params":         s["bytes_params"],
            "bytes_total":          bytes_total,
            "arithmetic_intensity": flops / bytes_total if bytes_total > 0 else 0.0,
        })
    df = pd.DataFrame(rows)
    # Tổng FLOPs cả model, gồm cả phép tính nằm ngoài layer (vd matmul attention của BERT)
    df.attrs["global_flops"] = float(sum(counts.get("Global", {}).values()))
    return df


def model_total_flops(analysis: pd.DataFrame, source: str = "counter") -> float:
    """
    Tổng FLOPs của cả model cho 1 lần forward.
    Mặc định chỉ cộng nguồn "counter" (con số chuẩn hay được báo cáo trong các bài báo);
    dùng source="all" để cộng cả phần ước lượng.
    """
    df = analysis if source == "all" else analysis[analysis["flops_source"] == source]
    return float((df["flops"] * df["n_calls"]).sum())


def add_efficiency(
    layer_times: pd.DataFrame,
    analysis: pd.DataFrame,
    spec: HardwareSpec,
    precision: str = "fp32",
    time_col: str = "forward_time_ms",
) -> pd.DataFrame:
    """
    Ghép thời gian đo (DataFrame của LayerProfiler) với kết quả analyze_layers().

    Thời gian mỗi lần gọi = trung vị qua các iteration của (layer_name, call_index).

    Cột thêm vào:
        flops, gflops, bytes_total, arithmetic_intensity,
        achieved_gflops   : GFLOP/s đạt được
        achieved_gbs      : GB/s đọc/ghi đạt được
        attainable_gflops : min(peak, bandwidth × AI)
        efficiency_pct    : % so với attainable (layer không có FLOPs → % băng thông)
        bound             : "compute" hoặc "memory"
    """
    per_call = (
        layer_times.groupby(["layer_name", "layer_type", "call_index"], as_index=False)[time_col]
        .median()
        .rename(columns={time_col: "time_ms"})
    )
    cols = ["layer_name", "flops", "flops_source", "bytes_total", "arithmetic_intensity"]
    df = per_call.merge(analysis[cols], on="layer_name", how="left")

    peak = spec.peak_gflops(precision)
    bw = spec.bandwidth_gbs
    ridge = spec.ridge_point(precision)
    seconds = df["time_ms"] / 1000.0
    valid = seconds > 0

    df["gflops"] = df["flops"] / 1e9
    df["achieved_gflops"] = np.where(valid, df["gflops"] / seconds.where(valid, 1), np.nan)
    df["achieved_gbs"] = np.where(valid, df["bytes_total"] / 1e9 / seconds.where(valid, 1), np.nan)
    df["attainable_gflops"] = np.minimum(peak, bw * df["arithmetic_intensity"])
    has_flops = df["flops"] > 0
    df["efficiency_pct"] = np.where(
        has_flops,
        df["achieved_gflops"] / df["attainable_gflops"].where(has_flops, 1) * 100.0,
        df["achieved_gbs"] / bw * 100.0,
    )
    df["bound"] = np.where(df["arithmetic_intensity"] >= ridge, "compute", "memory")
    df["hw_id"] = spec.hw_id
    df["precision"] = precision
    return df
