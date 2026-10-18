"""Compute diagnostic rules (RB-01 to RB-05) for VisionProf v2.0."""

from typing import List, Optional
from src.core.constants import RuleStatus
from src.core.types import RuleResult


def check_rb01_gpu_utilization(mean_gpu_util: float) -> RuleResult:
    """RB-01: Flag low GPU utilization during inference/training (<50% indicates under-utilization)."""
    if mean_gpu_util <= 0.0:
        return RuleResult("RB-01", RuleStatus.SKIPPED, "GPU Utilization", "No GPU metrics sampled.")
    if mean_gpu_util < 50.0:
        rec = "Gợi ý: Tăng kích thước batch size hoặc áp dụng CUDA Graphs (torch.cuda.make_graphed_callables) để loại bỏ kernel launch overhead."
        return RuleResult("RB-01", RuleStatus.WARN, "Low GPU Utilization", f"Mean GPU usage is only {mean_gpu_util:.1f}%.", recommendation=rec)
    return RuleResult("RB-01", RuleStatus.PASS, "Healthy GPU Compute", f"Mean GPU utilization is {mean_gpu_util:.1f}%.")


def check_rb02_tail_latency(p50_ms: float, p99_ms: float) -> RuleResult:
    """RB-02: Detect latency variance and tail spikes (tail ratio P99/P50 > 1.5)."""
    if p50_ms <= 0.0:
        return RuleResult("RB-02", RuleStatus.SKIPPED, "Tail Latency", "Invalid median latency.")
    ratio = p99_ms / p50_ms
    rec = "Gợi ý: Khóa tần số xung nhịp CPU/GPU, tăng số vòng warm-up, hoặc tắt GC ngầm (gc.disable()) trong vòng đo."
    if ratio > 2.0:
        return RuleResult("RB-02", RuleStatus.FAIL, "Severe Latency Jitter", f"P99/P50 ratio is {ratio:.2f} (>2.0).", recommendation=rec)
    if ratio > 1.5:
        return RuleResult("RB-02", RuleStatus.WARN, "High Latency Jitter", f"P99/P50 ratio is {ratio:.2f} (>1.5).", recommendation=rec)
    return RuleResult("RB-02", RuleStatus.PASS, "Stable Latency", f"P99/P50 ratio is {ratio:.2f}.")


def check_rb03_backward_bottleneck(forward_ms: float, backward_ms: float) -> RuleResult:
    """RB-03: Check if backward phase takes disproportionately longer than forward (>2.5x)."""
    if forward_ms <= 0.0 or backward_ms <= 0.0:
        return RuleResult("RB-03", RuleStatus.SKIPPED, "Backward Bottleneck", "Requires training phase metrics.")
    ratio = backward_ms / forward_ms
    if ratio > 2.5:
        rec = "Gợi ý: Cân nhắc dùng activation checkpointing (torch.utils.checkpoint) hoặc kiểm tra các phép toán autograd phức tạp."
        return RuleResult("RB-03", RuleStatus.WARN, "Heavy Backward Phase", f"Backward takes {ratio:.1f}x forward duration.", recommendation=rec)
    return RuleResult("RB-03", RuleStatus.PASS, "Normal Backward Balance", f"Backward/Forward ratio is {ratio:.1f}x.")


def check_rb04_compute_efficiency(total_gflops: float, latency_ms: float) -> RuleResult:
    """RB-04: Evaluate compute density in ms/GFLOP (Rule updated to ms/GFLOP standard)."""
    if total_gflops <= 0.0 or latency_ms <= 0.0:
        return RuleResult("RB-04", RuleStatus.SKIPPED, "Compute Density", "Requires layer profiler (layers.csv).")
    ms_per_gflop = latency_ms / total_gflops
    if ms_per_gflop > 2.0:
        rec = "Gợi ý: Tối ưu kích thước ma trận là bội số của 8 hoặc 16 để kích hoạt Tensor Cores trên GPU."
        return RuleResult("RB-04", RuleStatus.WARN, "Inefficient Compute Density", f"Consuming {ms_per_gflop:.2f} ms/GFLOP.", recommendation=rec)
    return RuleResult("RB-04", RuleStatus.PASS, "Optimal Compute Density", f"Efficient at {ms_per_gflop:.2f} ms/GFLOP.")


def check_rb05_optimizer_stall(opt_ms: float, total_step_ms: float) -> RuleResult:
    """RB-05: Check for optimizer stalls (>30% of total training step duration)."""
    if total_step_ms <= 0.0 or opt_ms <= 0.0:
        return RuleResult("RB-05", RuleStatus.SKIPPED, "Optimizer Stall", "No optimizer phase metrics available.")
    ratio = opt_ms / total_step_ms
    if ratio > 0.30:
        rec = "Gợi ý: Bật tham số fused=True trong AdamW (torch.optim.AdamW(..., fused=True)) để tăng tốc cập nhật trọng số trên GPU."
        return RuleResult("RB-05", RuleStatus.WARN, "Optimizer Stall", f"Optimizer consumes {ratio:.1%} of step.", recommendation=rec)
    return RuleResult("RB-05", RuleStatus.PASS, "Fast Optimizer Step", f"Optimizer consumes {ratio:.1%}.")
