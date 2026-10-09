"""
demo_layer_profiler.py — Demo mức Layer cho buổi trình bày (TV1)
================================================================
Gắn profiler vào một model chỉ với vài dòng code, rồi in ra:
  - top-10 layer chậm nhất
  - tỷ trọng thời gian theo loại layer
  - tổng FLOPs và layer đạt hiệu suất cao / thấp nhất

Chạy:
    python experiments/demo_layer_profiler.py                      # ResNet-50 trên CPU
    python experiments/demo_layer_profiler.py --model vit_b_16 --device cuda
"""

from __future__ import annotations

import argparse
import logging

import torch

from tv1_common import TORCH_MODELS, load_model
from src.collector import LayerProfiler
from src.flops import add_efficiency, analyze_layers, model_total_flops, run_model
from src.hw_specs import detect_hardware
from src.layer_analysis import to_markdown, top_k_layers, type_share


def main() -> None:
    p = argparse.ArgumentParser(description="Demo LayerProfiler")
    p.add_argument("--model", default="resnet50", choices=TORCH_MODELS)
    p.add_argument("--device", default="cpu")
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()
    logging.getLogger("collector").setLevel(logging.WARNING)

    model, x = load_model(args.model, args.device)
    with torch.no_grad():
        for _ in range(5):                                  # khởi động
            run_model(model, x)

        # ── Phần tích hợp: chỉ cần chừng này dòng ────────────────────────────
        profiler = LayerProfiler(model, device=args.device)
        for i in range(args.iters):
            with profiler.profile_context(iteration=i):
                run_model(model, x)
        df = profiler.to_dataframe()
        # ─────────────────────────────────────────────────────────────────────

    layers = df.rename(columns={"forward_time_ms": "time_ms"})
    print(f"\n=== {args.model} · {args.device} · {df['layer_name'].nunique()} layer · {args.iters} vòng đo ===\n")
    print("Top-10 layer chậm nhất (ms / 1 lần forward):\n")
    print(to_markdown(top_k_layers(layers, 10)))
    print("\nTỷ trọng theo loại layer:\n")
    print(to_markdown(type_share(layers)))

    analysis = analyze_layers(model, x)
    eff = add_efficiency(df, analysis, detect_hardware(args.device))
    eff = eff[eff["flops_source"] == "counter"].sort_values("efficiency_pct")
    print(f"\nTổng FLOPs: {analysis.attrs['global_flops'] / 1e9:.2f} G "
          f"(Σ layer: {model_total_flops(analysis) / 1e9:.2f} G)")
    if not eff.empty:
        lo, hi = eff.iloc[0], eff.iloc[-1]
        print(f"Hiệu suất thấp nhất: {lo.layer_name} ({lo.layer_type}) {lo.efficiency_pct:.1f}% · {lo.bound}-bound")
        print(f"Hiệu suất cao nhất : {hi.layer_name} ({hi.layer_type}) {hi.efficiency_pct:.1f}% · {hi.bound}-bound")


if __name__ == "__main__":
    main()
