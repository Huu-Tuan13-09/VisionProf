"""
tv1_figures.py — Vẽ biểu đồ cho báo cáo phần TV1 (CH1, CH2)
============================================================
Đọc kết quả trong results/ (TN1, TN2) và lưu PNG vào docs/tv1/figures/:

  fig1_type_share.png   CH1 · Tỷ trọng thời gian theo nhóm layer, CPU và GPU, 7 model
  fig2_gpu_speedup.png  CH1 · GPU nhanh hơn CPU bao nhiêu lần (tổng thời gian các layer)
  fig3_roofline.png     CH1 · Roofline: layer Conv / Linear / Attention trên CPU laptop và T4
  fig4_tn2_error.png    CH2 · Sai lệch so với torch.profiler theo loại layer (Colab T4)

Chạy:
    python experiments/tv1_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
OUT = PROJECT_ROOT / "docs" / "tv1" / "figures"
CPU_HW, GPU_HW = "cpu_laptop", "colab_t4"

MODELS = {  # tên thư mục → tên hiển thị
    "mobilenet_v3_large": "MobileNetV3",
    "resnet50": "ResNet-50",
    "yolov8n": "YOLOv8n",
    "vit_b_16": "ViT-B/16",
    "distilbert": "DistilBERT",
    "bert_base": "BERT-base",
    "mlp": "MLP",
}

# ── Bảng màu (reference palette, bản sáng) — thứ tự cố định, đã kiểm tra mù màu ──
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]

GROUPS = ["Conv", "Linear", "Attention", "Chuẩn hoá", "Kích hoạt", "Pooling", "Khác"]
GROUP_OF = {
    "Conv2d": "Conv",
    "Linear": "Linear", "NonDynamicallyQuantizableLinear": "Linear",
    "MultiheadAttention": "Attention",
    "BatchNorm2d": "Chuẩn hoá", "LayerNorm": "Chuẩn hoá",
    "ReLU": "Kích hoạt", "GELU": "Kích hoạt", "GELUActivation": "Kích hoạt", "SiLU": "Kích hoạt",
    "Hardswish": "Kích hoạt", "Hardsigmoid": "Kích hoạt", "Tanh": "Kích hoạt", "Sigmoid": "Kích hoạt",
    "MaxPool2d": "Pooling", "AdaptiveAvgPool2d": "Pooling", "AvgPool2d": "Pooling",
}
COLOR_OF = dict(zip(GROUPS, SERIES))


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": INK_2,
        "axes.edgecolor": AXIS, "axes.linewidth": 1.0,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0, "grid.linestyle": "-",
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "axes.titleweight": "bold", "axes.titlesize": 12,
        "axes.titlelocation": "left",
    })


def _ink_for(hex_color: str) -> str:
    """Chữ đặt trong ô màu: trắng hoặc đen tuỳ độ sáng của nền."""
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return INK if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.55 else "#ffffff"


def _type_share(hw: str, model: str) -> pd.Series:
    df = pd.read_csv(RESULTS / "analysis" / "TN1" / model / f"type_share_{hw}.csv")
    df["group"] = df["layer_type"].map(GROUP_OF).fillna("Khác")
    return df.groupby("group")["pct"].sum().reindex(GROUPS, fill_value=0.0)


def _total_ms(hw: str, model: str) -> float:
    return pd.read_csv(RESULTS / "analysis" / "TN1" / model / f"type_share_{hw}.csv")["time_ms"].sum()


# ─────────────────────────────────────────────────────────────────────────────

def fig1_type_share() -> Path:
    rows = []
    for key, label in MODELS.items():
        for hw, dev in ((CPU_HW, "CPU"), (GPU_HW, "GPU T4")):
            rows.append((f"{label} · {dev}", _type_share(hw, key)))

    fig, ax = plt.subplots(figsize=(10, 7.2))
    # Mỗi model 2 thanh (CPU, GPU) sát nhau, giữa các model có khoảng trống
    y = []
    for i in range(len(MODELS)):
        y += [i * 3.0, i * 3.0 + 1.0]
    y = np.array(y)
    left = np.zeros(len(rows))
    for g in GROUPS:
        vals = np.array([s[g] for _, s in rows])
        ax.barh(y, vals, left=left, height=0.8, color=COLOR_OF[g], label=g,
                edgecolor=SURFACE, linewidth=1.5)
        for yi, l, v in zip(y, left, vals):
            if v >= 12:  # chỉ ghi số khi đủ chỗ
                ax.text(l + v / 2, yi, f"{v:.0f}%", ha="center", va="center", fontsize=8.5,
                        color=_ink_for(COLOR_OF[g]))
        left += vals

    ax.set_yticks(y, [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% tổng thời gian các layer trong 1 lần forward (batch 1)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    fig.suptitle("Thời gian tập trung ở nhóm layer nào? CPU laptop và GPU Colab T4",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    ax.legend(ncol=7, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9, handlelength=1.2,
              columnspacing=1.0)
    fig.tight_layout()
    path = OUT / "fig1_type_share.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig2_gpu_speedup() -> Path:
    data = sorted(
        ((label, _total_ms(CPU_HW, key) / _total_ms(GPU_HW, key)) for key, label in MODELS.items()),
        key=lambda t: t[1],
    )
    labels, vals = zip(*data)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(labels, vals, height=0.55, color=SERIES[0])
    for i, v in enumerate(vals):
        ax.text(v + 0.15, i, f"×{v:.1f}", va="center", fontsize=9.5, color=INK)
    ax.axvline(1.0, color=MUTED, linewidth=1.0)
    ax.set_xlim(0, max(vals) * 1.15)
    ax.set_xlabel("Tổng thời gian layer trên CPU ÷ trên GPU T4 (batch 1, FP32) · đường dọc = ×1, không nhanh hơn")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("GPU T4 nhanh hơn CPU laptop bao nhiêu lần?")
    fig.tight_layout()
    path = OUT / "fig2_gpu_speedup.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig3_roofline() -> Path:
    picks = [("resnet50", "ResNet-50"), ("vit_b_16", "ViT-B/16"), ("bert_base", "BERT-base")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=False)
    for ax, (hw, title) in zip(axes, ((CPU_HW, "CPU laptop"), (GPU_HW, "GPU Colab T4"))):
        spec = json.load(open(RESULTS / hw / "resnet50" / "TN1" / "bs1_fp32" / "layer_meta.json",
                              encoding="utf-8"))["hardware"]
        peak, bw = spec["peak_gflops_fp32"], spec["bandwidth_gbs"]
        ai = np.logspace(-1, 3, 200)
        ax.plot(ai, np.minimum(peak, bw * ai), color=INK_2, linewidth=2)
        ridge = peak / bw
        ax.annotate(f"điểm gãy ≈ {ridge:.0f} FLOP/byte", (ridge, peak), xytext=(4, 6),
                    textcoords="offset points", fontsize=8.5, color=INK_2)

        for color, (key, label) in zip(SERIES, picks):
            s = pd.read_csv(RESULTS / hw / key / "TN1" / "bs1_fp32" / "layers_summary.csv")
            s = s[(s["flops_source"] == "counter") & (s["time_ms"] > 0)]
            ax.scatter(s["arithmetic_intensity"], s["achieved_gflops"], s=36, color=color,
                       edgecolor=SURFACE, linewidth=1.2, label=label, alpha=0.9, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Arithmetic intensity (FLOP / byte)")
        ax.set_title(f"{title} · trần {peak / 1000:.1f} TFLOP/s, {bw:.0f} GB/s", fontsize=11)
    axes[0].set_ylabel("Tốc độ tính đạt được (GFLOP/s)")
    axes[1].legend(loc="lower right", fontsize=9)
    fig.suptitle("Roofline: layer Conv / Linear / Attention đạt bao nhiêu % sức tối đa của máy?",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    path = OUT / "fig3_roofline.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig4_tn2_error() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, (key, label) in zip(axes, (("resnet50", "ResNet-50"), ("bert_base", "BERT-base"))):
        d = pd.read_csv(RESULTS / GPU_HW / key / "TN2" / "bs1_fp32" / "tn2_by_type.csv")
        d = d.sort_values("ref_ms", ascending=False).reset_index(drop=True)  # layer lớn ở trên
        ax.barh(d["layer_type"], d["rel_error_pct"], height=0.55, color=SERIES[0])
        for i, r in d.iterrows():
            ax.text(r["rel_error_pct"] + 1.5, i, f"{r['rel_error_pct']:.1f}%  ({r['ref_ms']:.3f} ms)",
                    va="center", fontsize=8.5, color=INK)
        ax.axvline(10, color=MUTED, linewidth=1.0)
        ax.invert_yaxis()
        ax.set_xlim(0, 130)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Sai lệch so với torch.profiler (%)")
    fig.suptitle("Layer lớn đo chính xác, layer rất nhỏ bị đo dư — GPU Colab T4, batch 1\n"
                 "(xếp từ layer tốn nhiều thời gian nhất xuống; trong ngoặc: thời gian thật / 1 lần forward;"
                 " đường dọc = mục tiêu 10%)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    path = OUT / "fig4_tn2_error.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    for fn in (fig1_type_share, fig2_gpu_speedup, fig3_roofline, fig4_tn2_error):
        print(f"✓ {fn()}")


if __name__ == "__main__":
    sys.exit(main())
