"""
TN1 — Phân rã thời gian theo layer (TV1 · trả lời CH1)
=======================================================
CH1: Thời gian tốn nhiều nhất ở loại layer nào? CPU và GPU khác nhau ra sao?

Bước 1 — ĐO (chạy trên từng máy):
    # Laptop (CPU)
    python experiments/tn1_layer_breakdown.py --device cpu
    # Colab (GPU T4)
    python experiments/tn1_layer_breakdown.py --device cuda

    → results/{máy}/{model}/TN1/bs1_fp32/layers.csv, layers_summary.csv, layer_meta.json

Bước 2 — PHÂN TÍCH (sau khi đã gom kết quả CPU và GPU về cùng thư mục results/):
    python experiments/tn1_layer_breakdown.py --analyze --cpu-hw cpu_laptop --gpu-hw colab_t4

    → results/analysis/TN1/{model}/ : top10_*.csv, type_share_*.csv,
                                     compare_top10.csv, compare_type_share.csv, report.md
"""

from __future__ import annotations

import argparse
import gc
import logging

import torch

from tv1_common import (
    RESULTS_DIR, TORCH_MODELS, load_model, profile_layers, result_dir,
)
from src.layer_analysis import (
    compare_top_k, compare_type_share, load_layers, to_markdown, top_k_layers, type_share,
)

logger = logging.getLogger("TN1")


def run_measure(models, device: str, batch_size: int, precision: str, n_warmup: int, n_iter: int) -> None:
    for name in models:
        out_dir = result_dir(device, name, "TN1", batch_size, precision)
        logger.info(f"── TN1 · {name} · {device} → {out_dir}")
        try:
            model, inputs = load_model(name, device, batch_size, precision)
            profile_layers(model, inputs, device, out_dir, n_warmup=n_warmup, n_iter=n_iter)
        except Exception as e:
            logger.error(f"[{name}] lỗi: {e}")
        finally:
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()


def run_analyze(models, cpu_hw: str, gpu_hw: str, batch_size: int, precision: str, k: int) -> None:
    cfg = f"bs{batch_size}_{precision}"
    for name in models:
        paths = {hw: RESULTS_DIR / hw / name / "TN1" / cfg / "layers.csv" for hw in (cpu_hw, gpu_hw)}
        available = {hw: p for hw, p in paths.items() if p.exists()}
        if not available:
            logger.warning(f"[{name}] chưa có kết quả TN1 — bỏ qua.")
            continue

        out = RESULTS_DIR / "analysis" / "TN1" / name
        out.mkdir(parents=True, exist_ok=True)
        layers = {hw: load_layers(p) for hw, p in available.items()}
        report = [f"# TN1 · {name} ({cfg})\n"]

        for hw, df in layers.items():
            top = top_k_layers(df, k)
            share = type_share(df)
            top.to_csv(out / f"top{k}_{hw}.csv", index=False)
            share.to_csv(out / f"type_share_{hw}.csv", index=False)
            report += [f"## Top-{k} layer chậm nhất · {hw}\n", to_markdown(top), "",
                       f"## Tỷ trọng theo loại layer · {hw}\n", to_markdown(share), ""]

        if len(layers) == 2:
            cmp_top = compare_top_k(layers[cpu_hw], layers[gpu_hw], k)
            cmp_share = compare_type_share(layers[cpu_hw], layers[gpu_hw])
            cmp_top.to_csv(out / f"compare_top{k}.csv", index=False)
            cmp_share.to_csv(out / "compare_type_share.csv", index=False)
            report += [f"## CPU ({cpu_hw}) và GPU ({gpu_hw}) · top-{k}\n", to_markdown(cmp_top), "",
                       "## CPU và GPU · tỷ trọng theo loại layer\n", to_markdown(cmp_share), ""]
        else:
            report.append(f"> Chỉ có kết quả của {list(layers)} — cần cả {cpu_hw} và {gpu_hw} để so sánh.\n")

        (out / "report.md").write_text("\n".join(report), encoding="utf-8")
        logger.info(f"[{name}] Phân tích → {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="TN1 — Phân rã thời gian theo layer")
    p.add_argument("--models", nargs="+", default=TORCH_MODELS)
    p.add_argument("--device", default="cpu", help="cpu hoặc cuda")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--analyze", action="store_true", help="Chỉ phân tích kết quả đã có")
    p.add_argument("--cpu-hw", default="cpu_laptop")
    p.add_argument("--gpu-hw", default="colab_t4")
    p.add_argument("--top", type=int, default=10)
    args = p.parse_args()

    if args.analyze:
        run_analyze(args.models, args.cpu_hw, args.gpu_hw, args.batch, args.precision, args.top)
    else:
        run_measure(args.models, args.device, args.batch, args.precision, args.warmup, args.iters)


if __name__ == "__main__":
    main()
