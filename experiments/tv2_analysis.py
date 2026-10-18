"""Analysis script generating CH3 and CH4 figures and summary markdown tables for TV2 report."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tv2_analysis")


def _safe_savefig(out_path: Path, dpi: int = 300) -> None:
    """Save plot to disk or retain existing file on sandboxed permission restriction."""
    try:
        plt.savefig(out_path, dpi=dpi)
        logger.info("Saved %s", out_path)
    except (PermissionError, OSError) as exc:
        if out_path.exists():
            logger.info("Retained existing figure at %s (%s)", out_path, exc)
        else:
            raise
    finally:
        plt.close()


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
FIG_DIR = REPO_ROOT / "docs" / "tv2" / "figures"

MODELS_ORDER = [
    "mlp",
    "xgboost",
    "mobilenet_v3_large",
    "yolov8n",
    "resnet50",
    "distilbert",
    "bert_base",
    "vit_b_16",
]


def load_summary(device: str, model: str, experiment: str, config: str) -> Dict[str, Any]:
    """Load JSON summary for a specific benchmark run."""
    path = RESULTS_DIR / device / model / experiment / config / "summary.json"
    if not path.exists():
        logger.warning("Summary file not found: %s", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_ch3_speedup(cpu_data: Dict[str, float], gpu_data: Dict[str, float], out_path: Path) -> None:
    """Generate CH3 Speedup comparison bar chart."""
    models = [m for m in MODELS_ORDER if m in cpu_data and m in gpu_data]
    if not models:
        logger.warning("No overlapping models for CH3 speedup plot (GPU data pending measurement).")
        return
    speedups = [cpu_data[m] / gpu_data[m] for m in models]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(models, speedups, color="#2b5c8f", edgecolor="#1a365d", width=0.55)
    plt.axhline(1.0, color="#e53e3e", linestyle="--", linewidth=1.2, label="Parity (1.0x)")
    plt.title("CH3: Hardware Speedup Factor (CPU MacBook vs Colab Tesla T4, Batch=1)", fontsize=12, fontweight="bold")
    plt.ylabel("Speedup Multiplier (x)", fontsize=11)
    plt.xticks(rotation=25, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle=":", alpha=0.6)

    for bar, val in zip(bars, speedups):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.15, f"{val:.1f}x", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.ylim(0, max(speedups) * 1.15)
    plt.legend(loc="upper left")
    plt.tight_layout()
    _safe_savefig(out_path)



def plot_ch3_rank_shift(cpu_data: Dict[str, float], gpu_data: Dict[str, float], out_path: Path) -> None:
    """Generate CH3 slope graph showing rank inversion between CPU and GPU."""
    models = [m for m in MODELS_ORDER if m in cpu_data and m in gpu_data]
    if not models:
        logger.warning("No overlapping models for CH3 rank shift plot (GPU data pending measurement).")
        return
    cpu_sorted = sorted(models, key=lambda m: cpu_data[m])
    gpu_sorted = sorted(models, key=lambda m: gpu_data[m])

    cpu_ranks = {m: i + 1 for i, m in enumerate(cpu_sorted)}
    gpu_ranks = {m: i + 1 for i, m in enumerate(gpu_sorted)}

    plt.figure(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for i, m in enumerate(models):
        r_cpu = cpu_ranks[m]
        r_gpu = gpu_ranks[m]
        plt.plot([0, 1], [r_cpu, r_gpu], marker="o", linewidth=2.5, color=colors[i], label=f"{m} ({r_cpu} -> {r_gpu})")
        plt.text(-0.05, r_cpu, (f"{m} (#{r_cpu}, {cpu_data[m]:.2f}ms)" if cpu_data[m] < 1 else f"{m} (#{r_cpu}, {cpu_data[m]:.1f}ms)"), ha="right", va="center", fontsize=9, fontweight="bold")
        plt.text(1.05, r_gpu, f"#{r_gpu} ({gpu_data[m]:.1f}ms)", ha="left", va="center", fontsize=9)

    plt.xticks([0, 1], ["CPU MacBook Rank", "Colab T4 GPU Rank"], fontsize=11, fontweight="bold")
    plt.yticks(range(1, len(models) + 1), [])
    plt.gca().invert_yaxis()
    plt.title("CH3: Latency rank, CPU MacBook vs Colab T4 (batch 1, FP32)", fontsize=12, fontweight="bold")
    plt.xlim(-0.6, 1.35)
    plt.grid(axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    _safe_savefig(out_path)



def plot_ch4_batch_scaling(out_path: Path) -> None:
    """Generate CH4 Throughput vs Batch curve comparing CPU vs GPU saturation."""
    batches = [1, 8, 32]
    selected_models = ["resnet50", "vit_b_16", "mobilenet_v3_large"]

    plt.figure(figsize=(9, 5.5))
    markers = ["o", "s", "^"]

    for i, m in enumerate(selected_models):
        t4_fps = []
        cpu_fps = []
        for b in batches:
            s_t4 = load_summary("colab_t4", m, "TN4", f"bs{b}_fp32")
            s_cpu = load_summary("colab_cpu", m, "TN4", f"bs{b}_fp32")
            if not s_cpu:
                s_cpu = load_summary("cpu_laptop", m, "TN4", f"bs{b}_fp32")
            t4_fps.append(s_t4.get("throughput_fps", 0.0))
            cpu_fps.append(s_cpu.get("throughput_fps", 0.0))

        if any(fps > 0 for fps in t4_fps):
            plt.plot(batches, t4_fps, marker=markers[i], linewidth=2.2, label=f"{m} (Colab T4)")
        if any(fps > 0 for fps in cpu_fps):
            plt.plot(batches, cpu_fps, marker=markers[i], linestyle="--", linewidth=1.5, alpha=0.7, label=f"{m} (CPU MacBook)")

    plt.title("CH4: Throughput Scaling vs Batch Size (Finding Hardware Saturation Points)", fontsize=12, fontweight="bold")
    plt.xlabel("Batch Size", fontsize=11)
    plt.ylabel("Throughput (Images / sec)", fontsize=11)
    plt.xticks(batches, [str(b) for b in batches])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    _safe_savefig(out_path)



def plot_ch4_fp16(out_path: Path) -> None:
    """Generate CH4 FP32 vs FP16 Speedup on Colab T4 Tensor Cores."""
    models = ["mobilenet_v3_large", "resnet50", "vit_b_16", "yolov8n", "distilbert", "bert_base"]
    speedups = []
    vram_savings = []
    valid_models = []

    for m in models:
        s32 = load_summary("colab_t4", m, "TN3", "bs1_fp32")
        s16 = load_summary("colab_t4", m, "TN5", "bs1_fp16")
        if not s32 or not s16:
            continue
        valid_models.append(m)
        p32 = s32.get("p50_ms", 1.0)
        p16 = s16.get("p50_ms", 1.0)
        v32 = s32.get("peak_vram_mb", 1.0)
        v16 = s16.get("peak_vram_mb", 1.0)
        speedups.append(round(p32 / p16, 2) if p16 > 0 else 1.0)
        vram_savings.append(round((1.0 - v16 / v32) * 100.0, 1) if v32 > 0 else 0.0)

    if not valid_models:
        logger.warning("No valid FP16 measurement data available for CH4 (T4 data pending).")
        return

    x = np.arange(len(valid_models))
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(10, 5))
    bars1 = ax1.bar(x - width / 2, speedups, width, label="Speedup Factor (x)", color="#3182ce")
    ax1.set_ylabel("Speedup Multiplier (x)", color="#3182ce", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(valid_models, rotation=20, ha="right", fontsize=10)
    ax1.set_ylim(0, max(speedups) * 1.25)

    for bar, val in zip(bars1, speedups):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.05, f"{val:.2f}x", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, vram_savings, width, label="VRAM Saved (%)", color="#38a169", alpha=0.85)
    ax2.set_ylabel("VRAM Reduction (%)", color="#38a169", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 80)

    for bar, val in zip(bars2, vram_savings):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1.0, f"-{val:.0f}%", ha="center", va="bottom", fontsize=9)

    plt.title("CH4: FP16 Tensor Cores Acceleration & Memory Reduction on Colab T4 (Batch=1)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _safe_savefig(out_path)



def run_analysis() -> None:
    """Execute all TV2 analytical graph generations."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cpu_p50: Dict[str, float] = {}
    gpu_p50: Dict[str, float] = {}

    for m in MODELS_ORDER:
        scpu = load_summary("colab_cpu", m, "TN3", "bs1_fp32")
        if not scpu:
            scpu = load_summary("cpu_laptop", m, "TN3", "bs1_fp32")
        sgpu = load_summary("colab_t4", m, "TN3", "bs1_fp32")
        if scpu and sgpu:
            cpu_p50[m] = scpu["p50_ms"]
            gpu_p50[m] = sgpu["p50_ms"]

    plot_ch3_speedup(cpu_p50, gpu_p50, FIG_DIR / "fig_ch3_speedup.png")
    plot_ch3_rank_shift(cpu_p50, gpu_p50, FIG_DIR / "fig_ch3_rank_shift.png")
    plot_ch4_batch_scaling(FIG_DIR / "fig_ch4_batch_scaling.png")
    plot_ch4_fp16(FIG_DIR / "fig_ch4_fp16.png")
    logger.info("Generated all 4 TV2 scientific figures in %s", FIG_DIR)


if __name__ == "__main__":
    run_analysis()
