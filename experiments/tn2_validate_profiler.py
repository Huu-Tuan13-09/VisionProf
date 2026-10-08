"""
TN2 — Kiểm chứng công cụ đo mức Layer (TV1 · trả lời CH2)
==========================================================
CH2: Công cụ của nhóm đo có đúng không, và làm model chậm đi bao nhiêu?

Đo 3 thứ cho mỗi model:
  1. Overhead       : model chậm đi bao nhiêu % khi gắn profiler (chạy xen kẽ có/không hook)
  2. Sai lệch       : so tổng thời gian theo loại layer với torch.profiler
                      (vd Σ Conv2d của mình ↔ aten::conv2d của torch.profiler)
  3. Thời gian bỏ sót: (thời gian forward không hook − Σ thời gian các layer) / thời gian forward
                      — phần các phép tính không nằm trong layer nào (x + y, torch.cat, matmul của attention…)
  4. Ngưỡng sàn     : thời gian bộ đo trả về khi KHÔNG có phép tính nào giữa 2 mốc, và % layer
                      nhanh dưới 3× ngưỡng này (số đo của chúng chủ yếu là sai số bộ đo).

Hạn chế đã biết (GPU): CUDA event cộng thêm vài µs đến vài chục µs cho mỗi khoảng đo
(nặng nhất trên Windows WDDM). Layer lớn (Conv, Linear, Attention) đo chính xác; layer rất
nhỏ ở batch 1 (BatchNorm, ReLU) bị đo dư → xem cột layers_below_3x_floor_pct.

Chạy:
    python experiments/tn2_validate_profiler.py --device cpu
    python experiments/tn2_validate_profiler.py --device cuda

    → results/{máy}/{model}/TN2/bs1_fp32/ : tn2_by_type.csv, tn2_summary.json (+ layers.csv của lần đo)
    → results/analysis/TN2/summary_{máy}.csv : bảng tổng hợp cho báo cáo
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd
import torch
from torch.profiler import ProfilerActivity, profile

from tv1_common import (
    RESULTS_DIR, hw_id_for, load_model, profile_layers, result_dir, sync,
)
from src.collector import calibrate_overhead_details, estimate_timer_bias
from src.flops import run_model
from src.layer_analysis import per_layer_time

logger = logging.getLogger("TN2")

# Loại layer (module) ↔ phép tính cấp cao nhất của nó trong torch.profiler
LAYER_TO_ATEN = {
    "Conv2d":            ["aten::conv2d"],
    "Linear":            ["aten::linear"],
    "BatchNorm2d":       ["aten::batch_norm"],
    "LayerNorm":         ["aten::layer_norm"],
    "ReLU":              ["aten::relu", "aten::relu_"],
    "Hardswish":         ["aten::hardswish", "aten::hardswish_"],
    "Hardsigmoid":       ["aten::hardsigmoid", "aten::hardsigmoid_"],
    "SiLU":              ["aten::silu", "aten::silu_"],
    "GELU":              ["aten::gelu"],
    "GELUActivation":    ["aten::gelu"],
    "MaxPool2d":         ["aten::max_pool2d"],
    "AdaptiveAvgPool2d": ["aten::adaptive_avg_pool2d"],
    "Embedding":         ["aten::embedding"],
}


def reference_by_type(model, inputs, device: str, n_iter: int) -> pd.DataFrame:
    """Thời gian trung bình mỗi lần forward của từng phép aten, đo bằng torch.profiler."""
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.startswith("cuda") else [])
    with torch.no_grad(), profile(activities=activities) as prof:
        for _ in range(n_iter):
            run_model(model, inputs)
        sync(device)

    rows = []
    for ev in prof.key_averages():
        if device.startswith("cuda"):
            t_us = getattr(ev, "device_time_total", None)
            if t_us is None:
                t_us = ev.cuda_time_total
        else:
            t_us = ev.cpu_time_total
        rows.append({"aten_op": ev.key, "ref_ms_per_iter": t_us / 1000.0 / n_iter, "ref_calls": ev.count / n_iter})
    return pd.DataFrame(rows)


def compare_with_reference(layers: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Σ thời gian theo loại layer (của mình) so với phép aten tương ứng (torch.profiler)."""
    ours = per_layer_time(layers).groupby("layer_type", as_index=False).agg(
        ours_ms=("time_ms", "sum"), ours_calls=("n_calls", "sum"))
    rows = []
    for _, r in ours.iterrows():
        ops = LAYER_TO_ATEN.get(r["layer_type"])
        if not ops:
            continue
        match = ref[ref["aten_op"].isin(ops)]
        if match.empty:
            continue
        ref_ms = float(match["ref_ms_per_iter"].sum())
        rows.append({
            "layer_type": r["layer_type"],
            "aten_ops": "+".join(ops),
            "ours_ms": r["ours_ms"],
            "ref_ms": ref_ms,
            "abs_error_ms": abs(r["ours_ms"] - ref_ms),
            "rel_error_pct": abs(r["ours_ms"] - ref_ms) / ref_ms * 100.0 if ref_ms > 0 else float("nan"),
            "ours_calls": r["ours_calls"],
            "ref_calls": float(match["ref_calls"].sum()),
        })
    return pd.DataFrame(rows)


def validate_model(name: str, device: str, batch_size: int, precision: str, n_iter: int) -> dict:
    out_dir = result_dir(device, name, "TN2", batch_size, precision)
    model, inputs = load_model(name, device, batch_size, precision)

    # 1. Overhead (xen kẽ có / không hook)
    first = inputs if isinstance(inputs, torch.Tensor) else None
    if first is not None:
        over = calibrate_overhead_details(model, first, n_warmup=10, n_measure=n_iter, device=device)
    else:  # model nhận dict (BERT): tự đo theo cùng cách
        over = _overhead_generic(model, inputs, device, n_iter)

    # 2. Đo bằng profiler của mình
    plugin = profile_layers(model, inputs, device, out_dir, n_warmup=10, n_iter=n_iter)
    layers = plugin.profiler.to_dataframe().rename(columns={"forward_time_ms": "time_ms"})

    # 3. Đo tham chiếu bằng torch.profiler
    ref = reference_by_type(model, inputs, device, n_iter)
    by_type = compare_with_reference(layers, ref)
    by_type.to_csv(out_dir / "tn2_by_type.csv", index=False)

    layer_sum = float(per_layer_time(layers)["time_ms"].sum())
    t_clean = over["t_clean_ms"]
    # Mốc để tính "bỏ sót": CPU → thời gian forward không hook;
    # GPU → thời gian GPU chạy liền mạch cả forward (không tính lúc GPU chờ CPU gửi lệnh)
    if device.startswith("cuda"):
        t_base = float(pd.Series(plugin.profiler.get_iteration_totals()).median())
    else:
        t_base = t_clean
    total_ref = float(by_type["ref_ms"].sum()) if not by_type.empty else float("nan")
    total_ours = float(by_type["ours_ms"].sum()) if not by_type.empty else float("nan")

    # 4. Ngưỡng sàn của bộ đo: thời gian đo được khi giữa 2 mốc KHÔNG có phép tính nào.
    #    Layer nhanh cỡ dưới ~3× ngưỡng này thì số đo chủ yếu là sai số của bộ đo
    #    (đặc biệt với CUDA event trên Windows WDDM).
    floor_ms = estimate_timer_bias(device, tiny_kernel=True)
    per_call_ms = layers.groupby(["layer_name", "call_index"])["raw_time_ms"].median()
    below_floor_pct = float((per_call_ms < 3 * floor_ms).mean() * 100.0)

    summary = {
        "model": name,
        "hw_id": hw_id_for(device),
        "n_hooked": over["n_hooked"],
        "t_clean_ms": t_clean,
        "t_hooked_ms": over["t_hooked_ms"],
        "overhead_pct": over["overhead_pct"],
        "overhead_per_layer_ms": over["per_layer_ms"],
        "layer_sum_ms": layer_sum,
        "t_base_ms": t_base,
        "unattributed_pct": max(0.0, (t_base - layer_sum) / t_base * 100.0) if t_base > 0 else float("nan"),
        "matched_ours_ms": total_ours,
        "matched_ref_ms": total_ref,
        "total_rel_error_pct": abs(total_ours - total_ref) / total_ref * 100.0 if total_ref > 0 else float("nan"),
        "timer_bias_ms": plugin.timer_bias_ms,
        "timer_floor_ms": floor_ms,
        "layers_below_3x_floor_pct": below_floor_pct,
    }
    with open(out_dir / "tn2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(
        f"[{name}] overhead={summary['overhead_pct']:.1f}% · sai lệch={summary['total_rel_error_pct']:.1f}% · "
        f"bỏ sót={summary['unattributed_pct']:.1f}%"
    )
    return summary


def _overhead_generic(model, inputs, device: str, n_iter: int) -> dict:
    """Như calibrate_overhead_details nhưng cho input dạng dict / tuple."""
    import time
    import numpy as np
    from src.collector import LayerProfiler

    profiler = LayerProfiler(model, device=device, gpu_queue_prefill=False)
    clean, hooked = [], []
    with torch.no_grad():
        for _ in range(10):
            run_model(model, inputs)
        for i in range(n_iter):
            sync(device)
            t0 = time.perf_counter()
            run_model(model, inputs)
            sync(device)
            clean.append((time.perf_counter() - t0) * 1000.0)

            profiler.attach_hooks()
            t0 = time.perf_counter()
            profiler.begin_iteration(i)
            run_model(model, inputs)
            profiler.end_iteration()
            hooked.append((time.perf_counter() - t0) * 1000.0)
            n_hooked = profiler.n_hooked
            profiler.remove_hooks()
    t_clean, t_hooked = float(np.median(clean)), float(np.median(hooked))
    diff = max(0.0, t_hooked - t_clean)
    return {"t_clean_ms": t_clean, "t_hooked_ms": t_hooked, "n_hooked": n_hooked,
            "overhead_pct": diff / t_clean * 100.0, "per_layer_ms": diff / max(n_hooked, 1)}


def main() -> None:
    p = argparse.ArgumentParser(description="TN2 — Kiểm chứng công cụ đo mức Layer")
    p.add_argument("--models", nargs="+", default=["resnet50", "bert_base"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    p.add_argument("--iters", type=int, default=50)
    args = p.parse_args()

    rows = []
    for name in args.models:
        try:
            rows.append(validate_model(name, args.device, args.batch, args.precision, args.iters))
        except Exception as e:
            logger.error(f"[{name}] lỗi: {e}")
    if rows:
        out = RESULTS_DIR / "analysis" / "TN2"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"summary_{hw_id_for(args.device)}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        logger.info(f"Tổng hợp TN2 → {path}")


if __name__ == "__main__":
    main()
