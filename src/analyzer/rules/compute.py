"""Compute diagnostic rules (RB-01 to RB-05) for VisionProf v2.0."""

from typing import List, Optional
from src.core.constants import RuleStatus
from src.core.types import RuleResult


def check_rb01_gpu_utilization(mean_gpu_util: float) -> RuleResult:
    """RB-01: Flag low GPU utilization during inference/training (<50% indicates under-utilization)."""
    if mean_gpu_util <= 0.0:
        return RuleResult("RB-01", RuleStatus.SKIPPED, "GPU Utilization", "No GPU metrics sampled.")
    if mean_gpu_util < 50.0:
        return RuleResult("RB-01", RuleStatus.WARN, "Low GPU Utilization", f"Mean GPU usage is only {mean_gpu_util:.1f}%.")
    return RuleResult("RB-01", RuleStatus.PASS, "Healthy GPU Compute", f"Mean GPU utilization is {mean_gpu_util:.1f}%.")


def check_rb02_tail_latency(p50_ms: float, p99_ms: float) -> RuleResult:
    """RB-02: Detect latency variance and tail spikes (tail ratio P99/P50 > 1.5)."""
    if p50_ms <= 0.0:
        return RuleResult("RB-02", RuleStatus.SKIPPED, "Tail Latency", "Invalid median latency.")
    ratio = p99_ms / p50_ms
    if ratio > 2.0:
        return RuleResult("RB-02", RuleStatus.FAIL, "Severe Latency Jitter", f"P99/P50 ratio is {ratio:.2f} (>2.0).")
    if ratio > 1.5:
        return RuleResult("RB-02", RuleStatus.WARN, "High Latency Jitter", f"P99/P50 ratio is {ratio:.2f} (>1.5).")
    return RuleResult("RB-02", RuleStatus.PASS, "Stable Latency", f"P99/P50 ratio is {ratio:.2f}.")


def check_rb03_backward_bottleneck(forward_ms: float, backward_ms: float) -> RuleResult:
    """RB-03: Check if backward phase takes disproportionately longer than forward (>2.5x)."""
    if forward_ms <= 0.0 or backward_ms <= 0.0:
        return RuleResult("RB-03", RuleStatus.SKIPPED, "Backward Bottleneck", "Requires training phase metrics.")
    ratio = backward_ms / forward_ms
    if ratio > 2.5:
        return RuleResult("RB-03", RuleStatus.WARN, "Heavy Backward Phase", f"Backward takes {ratio:.1f}x forward duration.")
    return RuleResult("RB-03", RuleStatus.PASS, "Normal Backward Balance", f"Backward/Forward ratio is {ratio:.1f}x.")


def check_rb04_compute_efficiency(total_gflops: float, latency_ms: float) -> RuleResult:
    """RB-04: Evaluate compute density in ms/GFLOP (Rule updated to ms/GFLOP standard)."""
    if total_gflops <= 0.0 or latency_ms <= 0.0:
        return RuleResult("RB-04", RuleStatus.SKIPPED, "Compute Density", "Requires layer profiler (layers.csv).")
    ms_per_gflop = latency_ms / total_gflops
    if ms_per_gflop > 2.0:
        return RuleResult("RB-04", RuleStatus.WARN, "Inefficient Compute Density", f"Consuming {ms_per_gflop:.2f} ms/GFLOP.")
    return RuleResult("RB-04", RuleStatus.PASS, "Optimal Compute Density", f"Efficient at {ms_per_gflop:.2f} ms/GFLOP.")


def check_rb05_optimizer_stall(opt_ms: float, total_step_ms: float) -> RuleResult:
    """RB-05: Check for optimizer stalls (>30% of total training step duration)."""
    if total_step_ms <= 0.0 or opt_ms <= 0.0:
        return RuleResult("RB-05", RuleStatus.SKIPPED, "Optimizer Stall", "No optimizer phase metrics available.")
    ratio = opt_ms / total_step_ms
    if ratio > 0.30:
        return RuleResult("RB-05", RuleStatus.WARN, "Optimizer Stall", f"Optimizer consumes {ratio:.1%} of step.")
    return RuleResult("RB-05", RuleStatus.PASS, "Fast Optimizer Step", f"Optimizer consumes {ratio:.1%}.")
