"""
tv3_figures.py — Sinh biểu đồ phân tích thực nghiệm cho Báo cáo phần TV3 (CH5, CH6)
==================================================================================
Đọc số liệu thực nghiệm gốc trong results/ (TN6, TN7) và render các biểu đồ chất lượng cao
vào thư mục docs/tv3/figures/:

  fig_ch5_training_breakdown.png     CH5 · Phân bổ thời gian Forward / Backward / Optimizer (Colab T4)
  fig_ch5_backward_forward_ratio.png CH5 · Tỷ số Backward / Forward so với đường chuẩn lý thuyết 2.0x
  fig_ch5_memory_train_vs_infer.png  CH5 · So sánh đỉnh VRAM Inference vs Training & Tỷ lệ phóng đại
  fig_ch6_pipeline_breakdown.png     CH6 · Phân rã quy trình thật (Pre / Model / Post) & Ngưỡng 30%
  fig_ch6_cpu_vs_gpu_pipeline.png    CH6 · Nghịch lý tăng tốc: So sánh cơ cấu Pipeline CPU vs T4

Chạy:
    python experiments/tv3_figures.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
OUT_DIR = PROJECT_ROOT / "docs" / "tv3" / "figures"

# ── Bảng màu chuẩn hoá đồng bộ toàn dự án ─────────────────────────────────────
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

COLOR_FWD = "#2a78d6"       # Blue
COLOR_BWD = "#eb6834"       # Coral / Orange
COLOR_OPT = "#1baf7a"       # Teal / Green
COLOR_PRE = "#e87ba4"       # Rose
COLOR_POST = "#4a3aa7"      # Indigo / Purple
COLOR_ACCENT = "#d9381e"    # Red reference line


def _setup_style() -> None:
    """Cài đặt style đồng bộ với tv1_figures và phong cách báo cáo khoa học."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "xtick.color": MUTED,
        "ytick.color": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.titlelocation": "left",
    })


def _safe_savefig(fig: plt.Figure, path: Path) -> None:
    """Lưu biểu đồ chất lượng cao (DPI 200)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  [OK] Saved figure: {path.relative_to(PROJECT_ROOT)}")
    except Exception as exc:
        print(f"  [Error] Could not save {path}: {exc}")
    finally:
        plt.close(fig)


# ── BIỂU ĐỒ 1: TN6 Training Breakdown ──────────────────────────────────────────
def plot_ch5_training_breakdown() -> None:
    """Vẽ biểu đồ phân rã thời gian Forward / Backward / Optimizer (Colab T4)."""
    models = ["resnet50", "mobilenet_v3_large", "distilbert", "mlp"]
    labels = ["ResNet-50", "MobileNetV3", "DistilBERT", "MLP"]

    fwd_times, bwd_times, opt_times = [], [], []
    for m in models:
        csv_p = RESULTS / "colab_t4" / m / "TN6" / "bs8_fp32" / "phases.csv"
        if csv_p.exists():
            df = pd.read_csv(csv_p)
            med = df.groupby("phase")["time_ms"].median()
            fwd_times.append(med.get("forward", 0.0))
            bwd_times.append(med.get("backward", 0.0))
            opt_times.append(med.get("optimizer", 0.0))
        else:
            fwd_times.append(0.0)
            bwd_times.append(0.0)
            opt_times.append(0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.2, 1]})

    # Subplot 1: Stacked Bar theo thời gian tuyệt đối (ms)
    x = np.arange(len(labels))
    w = 0.5
    p1 = ax1.bar(x, fwd_times, w, label="Forward pass", color=COLOR_FWD)
    p2 = ax1.bar(x, bwd_times, w, bottom=fwd_times, label="Backward pass", color=COLOR_BWD)
    bottom_opt = np.array(fwd_times) + np.array(bwd_times)
    p3 = ax1.bar(x, opt_times, w, bottom=bottom_opt, label="Optimizer step", color=COLOR_OPT)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontweight="bold")
    ax1.set_ylabel("Thời gian trung vị (ms)")
    ax1.set_title("a) Thời gian tuyệt đối từng bước huấn luyện (Colab T4, bs=8)")
    ax1.legend(loc="upper right")

    # In nhãn tổng số ms trên đỉnh cột
    for i, total in enumerate(bottom_opt + np.array(opt_times)):
        ax1.text(i, total + 1.5, f"{total:.1f} ms", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Subplot 2: 100% Stacked Bar theo tỷ lệ phần trăm (%)
    totals = bottom_opt + np.array(opt_times)
    fwd_pct = np.array(fwd_times) / totals * 100
    bwd_pct = np.array(bwd_times) / totals * 100
    opt_pct = np.array(opt_times) / totals * 100

    ax2.bar(x, fwd_pct, w, label="Forward (%)", color=COLOR_FWD)
    ax2.bar(x, bwd_pct, w, bottom=fwd_pct, label="Backward (%)", color=COLOR_BWD)
    ax2.bar(x, opt_pct, w, bottom=fwd_pct + bwd_pct, label="Optimizer (%)", color=COLOR_OPT)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontweight="bold")
    ax2.set_ylabel("Tỷ trọng thời gian (%)")
    ax2.set_ylim(0, 105)
    ax2.set_title("b) Cơ cấu tỷ trọng thời gian các pha (%)")

    # Chèn text % vào trong cột
    for i in range(len(labels)):
        ax2.text(i, fwd_pct[i] / 2, f"{fwd_pct[i]:.0f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        ax2.text(i, fwd_pct[i] + bwd_pct[i] / 2, f"{bwd_pct[i]:.0f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        if opt_pct[i] >= 6.0:
            ax2.text(i, fwd_pct[i] + bwd_pct[i] + opt_pct[i] / 2, f"{opt_pct[i]:.0f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=8)

    fig.tight_layout()
    _safe_savefig(fig, OUT_DIR / "fig_ch5_training_breakdown.png")


# ── BIỂU ĐỒ 2: TN6 Backward/Forward Ratio ───────────────────────────────────────
def plot_ch5_backward_forward_ratio() -> None:
    """Vẽ biểu đồ tỷ số Backward / Forward so với mốc lý thuyết 2.0x (CH5)."""
    models = ["resnet50", "mobilenet_v3_large", "distilbert", "mlp"]
    labels = ["ResNet-50", "MobileNetV3", "DistilBERT", "MLP"]
    ratios = []

    for m in models:
        csv_p = RESULTS / "colab_t4" / m / "TN6" / "bs8_fp32" / "phases.csv"
        if csv_p.exists():
            df = pd.read_csv(csv_p)
            med = df.groupby("phase")["time_ms"].median()
            fwd = med.get("forward", 1.0)
            bwd = med.get("backward", 0.0)
            ratios.append(bwd / fwd)
        else:
            ratios.append(0.0)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, ratios, width=0.45, color=COLOR_BWD, edgecolor=AXIS, linewidth=1)

    # Đường mốc lý thuyết 2.0x
    ax.axhline(2.0, color=COLOR_ACCENT, linestyle="--", linewidth=1.8, label="Lý thuyết: Backward = 2.0× Forward")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylabel("Hệ số Backward / Forward (lần)")
    ax.set_ylim(0, 2.7)
    ax.set_title("CH5: Tỷ số Backward / Forward trên Colab T4 (Kỳ vọng: ~2.0×)")
    ax.legend(loc="upper left")

    for bar, val in zip(bars, ratios):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.06,
            f"{val:.2f}×",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
            color=INK,
        )

    # Chú thích khoa học
    ax.text(
        0.5, -0.15,
        "• DistilBERT đạt 2.21× (chuẩn lý thuyết Transformer)\n"
        "• CNNs (ResNet-50 1.82×, MobileNetV3 1.65×) do cuDNN tối ưu hóa kernel tích chập backward",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color=INK_2,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=GRID, edgecolor="none", alpha=0.5),
    )

    fig.tight_layout()
    _safe_savefig(fig, OUT_DIR / "fig_ch5_backward_forward_ratio.png")


# ── BIỂU ĐỒ 3: TN6 Memory Train vs Inference ────────────────────────────────────
def plot_ch5_memory_train_vs_infer() -> None:
    """So sánh đỉnh cấp phát VRAM giữa Inference và Training (CH5)."""
    models = ["resnet50", "mobilenet_v3_large", "mlp"]
    labels = ["ResNet-50", "MobileNetV3", "MLP"]

    infer_mbs, train_mbs, ratios = [], [], []
    for m in models:
        json_p = RESULTS / "colab_t4" / m / "TN6" / "bs8_fp32" / "memory_ratio.json"
        if json_p.exists():
            with open(json_p, "r", encoding="utf-8") as f:
                data = json.load(f)
                infer_mbs.append(data.get("infer_peak_vram_mb", 0.0))
                train_mbs.append(data.get("train_peak_vram_mb", 0.0))
                ratios.append(data.get("memory_ratio", 0.0))
        else:
            infer_mbs.append(0.0)
            train_mbs.append(0.0)
            ratios.append(0.0)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(labels))
    w = 0.32

    b1 = ax.bar(x - w / 2, infer_mbs, w, label="Inference Peak VRAM", color=COLOR_FWD)
    b2 = ax.bar(x + w / 2, train_mbs, w, label="Training Peak VRAM", color=COLOR_BWD)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylabel("Peak VRAM (MB)")
    ax.set_title("CH5: Đỉnh cấp phát VRAM Inference vs Training (Colab T4, bs=8)")
    ax.set_ylim(0, max(train_mbs) * 1.25)
    ax.legend(loc="upper left")

    # Hiển thị số MB và tỷ số phóng đại
    for i in range(len(labels)):
        ax.text(x[i] - w / 2, infer_mbs[i] + 15, f"{infer_mbs[i]:.0f}M", ha="center", fontsize=9, fontweight="bold")
        ax.text(x[i] + w / 2, train_mbs[i] + 15, f"{train_mbs[i]:.0f}M", ha="center", fontsize=9, fontweight="bold")
        # Badge tỷ lệ phóng đại
        ax.annotate(
            f"Tăng {ratios[i]:.2f}×",
            xy=(x[i] + w / 2, train_mbs[i]),
            xytext=(x[i] + w / 2, train_mbs[i] + 90),
            ha="center",
            fontweight="bold",
            fontsize=9,
            color=COLOR_ACCENT,
            arrowprops=dict(arrowstyle="->", color=COLOR_ACCENT, lw=1.2),
        )

    fig.tight_layout()
    _safe_savefig(fig, OUT_DIR / "fig_ch5_memory_train_vs_infer.png")


# ── BIỂU ĐỒ 4: TN7 Pipeline Breakdown ──────────────────────────────────────────
def plot_ch6_pipeline_breakdown() -> None:
    """Biểu đồ phân rã quy trình ứng dụng thực tế & kiểm chứng giả thuyết CH6."""
    models_info = [
        ("yolov8n", "YOLOv8n (CV)", ["preprocess", "model_forward", "postprocess_nms"]),
        ("bert_base", "BERT-base (NLP)", ["tokenize", "model_forward", "decode_pooling"]),
        ("xgboost", "XGBoost (Tabular)", ["csv_parse", "predict", ""]),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 4.5))

    y_positions = np.arange(len(models_info))
    bar_height = 0.45

    # Định nghĩa cấu trúc dữ liệu %
    data_rows = []
    for model_name, label, phases in models_info:
        csv_p = RESULTS / "colab_t4" / model_name / "TN7" / "bs1_fp32" / "phases.csv"
        if csv_p.exists():
            df = pd.read_csv(csv_p)
            med = df.groupby("phase")["time_ms"].median().to_dict()
            total = sum(med.values())
            p_pre = med.get(phases[0], 0.0) / total * 100
            p_mod = med.get(phases[1], 0.0) / total * 100
            p_post = med.get(phases[2], 0.0) / total * 100 if phases[2] else 0.0
            data_rows.append((p_pre, p_mod, p_post, total))
        else:
            data_rows.append((0, 0, 0, 0))

    # Vẽ horizontal stacked bar
    for idx, (p_pre, p_mod, p_post, total) in enumerate(data_rows):
        y = y_positions[idx]
        b1 = ax.barh(y, p_pre, bar_height, color=COLOR_PRE, label="Tiền xử lý (Preprocess)" if idx == 0 else "")
        b2 = ax.barh(y, p_mod, bar_height, left=p_pre, color=COLOR_FWD, label="Model Forward" if idx == 0 else "")
        b3 = ax.barh(y, p_post, bar_height, left=p_pre + p_mod, color=COLOR_POST, label="Hậu xử lý (Postprocess/NMS)" if idx == 0 else "")

        # Text % vào trong thanh
        if p_pre > 5:
            ax.text(p_pre / 2, y, f"{p_pre:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        ax.text(p_pre + p_mod / 2, y, f"{p_mod:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        if p_post > 5:
            ax.text(p_pre + p_mod + p_post / 2, y, f"{p_post:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9)

        # Chú thích tổng thời gian bên cạnh
        ax.text(102, y, f"Tổng: {total:.1f}ms", va="center", fontsize=9, fontweight="bold", color=INK_2)

    # Đường ngưỡng 30% overhead ngoài model (CH6)
    ax.axvline(30.0, color=COLOR_ACCENT, linestyle="--", linewidth=1.5, label="Ngưỡng CH6: Overhead ngoài model > 30%")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([m[1] for m in models_info], fontweight="bold")
    ax.set_xlabel("Tỷ trọng thời gian trong toàn bộ pipeline (%)")
    ax.set_xlim(0, 115)
    ax.set_title("CH6: Phân rã thời gian Pipeline suy luận thực tế (Colab T4, bs=1)")
    ax.legend(loc="lower right")

    fig.tight_layout()
    _safe_savefig(fig, OUT_DIR / "fig_ch6_pipeline_breakdown.png")


# ── BIỂU ĐỒ 5: TN7 CPU vs GPU Pipeline Breakdown ──────────────────────────────
def plot_ch6_cpu_vs_gpu_pipeline() -> None:
    """So sánh cơ cấu Pipeline của YOLOv8n trên CPU vs GPU (Hiện tượng bộc lộ bottleneck)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

    # Đọc số liệu YOLOv8n
    # CPU:
    cpu_csv = RESULTS / "colab_cpu" / "yolov8n" / "TN7" / "bs1_fp32" / "phases.csv"
    df_cpu = pd.read_csv(cpu_csv)
    med_cpu = df_cpu.groupby("phase")["time_ms"].median()
    # T4:
    t4_csv = RESULTS / "colab_t4" / "yolov8n" / "TN7" / "bs1_fp32" / "phases.csv"
    df_t4 = pd.read_csv(t4_csv)
    med_t4 = df_t4.groupby("phase")["time_ms"].median()

    colors = [COLOR_PRE, COLOR_FWD, COLOR_POST]
    labels = ["Tiền xử lý (Letterbox)", "Model Forward", "Hậu xử lý (NMS)"]

    vals_cpu = [med_cpu["preprocess"], med_cpu["model_forward"], med_cpu["postprocess_nms"]]
    vals_t4 = [med_t4["preprocess"], med_t4["model_forward"], med_t4["postprocess_nms"]]

    # Donut CPU
    wedges1, texts1, autotexts1 = ax1.pie(
        vals_cpu,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.45, edgecolor="w", linewidth=2),
        pctdistance=0.75,
    )
    for at in autotexts1:
        at.set_fontweight("bold")
        at.set_fontsize(9)
    ax1.set_title(f"a) Colab CPU (Tổng trễ: {sum(vals_cpu):.1f} ms)\nModel chiếm tuyệt đối 94.0%")

    # Donut GPU T4
    wedges2, texts2, autotexts2 = ax2.pie(
        vals_t4,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.45, edgecolor="w", linewidth=2),
        pctdistance=0.75,
    )
    for at in autotexts2:
        at.set_fontweight("bold")
        at.set_fontsize(9)
    ax2.set_title(f"b) Colab Tesla T4 GPU (Tổng trễ: {sum(vals_t4):.1f} ms)\nNgoài model bộc lộ chiếm 41.0% (>30%)")

    fig.suptitle(
        "Nghịch lý chuyển dịch điểm nghẽn của YOLOv8n: Từ Compute-Bound sang Pipeline-Bound",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )

    fig.tight_layout()
    _safe_savefig(fig, OUT_DIR / "fig_ch6_cpu_vs_gpu_pipeline.png")


def main() -> None:
    """Hàm chạy chính sinh tất cả các biểu đồ cho TV3."""
    _setup_style()
    print("=== Đang sinh bộ biểu đồ khoa học cho Báo cáo TV3 ===")
    plot_ch5_training_breakdown()
    plot_ch5_backward_forward_ratio()
    plot_ch5_memory_train_vs_infer()
    plot_ch6_pipeline_breakdown()
    plot_ch6_cpu_vs_gpu_pipeline()
    print("=== Hoàn tất sinh biểu đồ TV3 vào docs/tv3/figures/ ===")


if __name__ == "__main__":
    main()
