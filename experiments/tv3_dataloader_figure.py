"""
tv3_dataloader_figure.py — CH5 (2/2): nạp dữ liệu có làm GPU phải chờ?
=======================================================================
Đọc results/colab_t4/{model}/TN6/bs32_fp32_w{W}/dataloader_summary.json (scripts/run_tn6_dataloader.py)
và vẽ docs/tv3/figures/fig_ch5_dataloader_workers.png: thời gian một bước huấn luyện tách
data_load / forward / backward / optimizer theo num_workers, kèm % chờ dữ liệu và ảnh/giây.

Chạy:
    python experiments/tv3_dataloader_figure.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results" / "colab_t4"
OUT_PATH = PROJECT_ROOT / "docs" / "tv3" / "figures" / "fig_ch5_dataloader_workers.png"

SURFACE, INK, INK_2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
PHASES = [("data_load", "Chờ dữ liệu", "#c3c2b7"), ("forward", "Forward", "#2a78d6"),
          ("backward", "Backward", "#eb6834"), ("optimizer", "Optimizer", "#1baf7a")]
MODELS = [("mobilenet_v3_large", "MobileNetV3-L"), ("resnet50", "ResNet-50")]
WORKERS = [0, 2, 4]


def _load(model: str, workers: int) -> dict:
    path = RESULTS / model / "TN6" / f"bs32_fp32_w{workers}" / "dataloader_summary.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
    })
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=False)
    for ax, (model, label) in zip(axes, MODELS):
        runs = [_load(model, w) for w in WORKERS]
        x = np.arange(len(WORKERS))
        bottom = np.zeros(len(WORKERS))
        for key, name, color in PHASES:
            vals = np.array([r["median_ms"][key] for r in runs])
            ax.bar(x, vals, 0.55, bottom=bottom, color=color, label=name)
            bottom += vals
        top = bottom.max()
        for i, r in enumerate(runs):
            ax.text(i, bottom[i] + top * 0.02,
                    f"chờ {r['data_load_pct']:.1f}%\n{r['images_per_s']:.0f} ảnh/s",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"num_workers={w}" for w in WORKERS])
        ax.set_ylim(0, top * 1.22)
        ax.set_ylabel("Thời gian một bước, trung vị (ms)")
        ax.set_title(f"{label} (batch 32)")
    axes[0].legend(loc="upper right", ncol=2)
    fig.suptitle("CH5: GPU chờ dữ liệu bao lâu? DataLoader thật trên Colab T4, 2 vCPU",
                 fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
