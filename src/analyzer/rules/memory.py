"""Memory diagnostic rules (RA-01 to RA-05) for VisionProf v2.0."""

from typing import Any, Dict, List
from src.core.constants import RuleStatus
from src.core.types import RuleResult


def check_ra01_oom_risk(peak_vram_mb: float, total_vram_mb: float) -> RuleResult:
    """RA-01: Evaluate peak VRAM against hardware ceiling (threshold: >90% WARN, >95% FAIL)."""
    if total_vram_mb <= 0.0:
        return RuleResult("RA-01", RuleStatus.SKIPPED, "OOM Risk", "VRAM capacity unknown.")
    util_ratio = peak_vram_mb / total_vram_mb
    if util_ratio > 0.95:
        return RuleResult("RA-01", RuleStatus.FAIL, "Severe OOM Risk", f"Peak VRAM utilizes {util_ratio:.1%} of memory.")
    if util_ratio > 0.90:
        return RuleResult("RA-01", RuleStatus.WARN, "High VRAM Utilization", f"Peak VRAM utilizes {util_ratio:.1%} of memory.")
    return RuleResult("RA-01", RuleStatus.PASS, "Normal VRAM Usage", f"Peak VRAM is safe ({util_ratio:.1%}).")


def check_ra02_fragmentation(allocated_mb: float, reserved_mb: float) -> RuleResult:
    """RA-02: Evaluate GPU memory allocator fragmentation ratio (reserved - allocated)."""
    if reserved_mb <= 0.0:
        return RuleResult("RA-02", RuleStatus.SKIPPED, "Fragmentation", "No reserved GPU memory tracked.")
    frag_ratio = (reserved_mb - allocated_mb) / reserved_mb
    if frag_ratio > 0.40:
        return RuleResult("RA-02", RuleStatus.WARN, "High Fragmentation", f"Unallocated reserve is {frag_ratio:.1%}.")
    return RuleResult("RA-02", RuleStatus.PASS, "Healthy Allocator", f"Fragmentation is low ({frag_ratio:.1%}).")


def check_ra03_activation_hotspot(layer_activations: List[float]) -> RuleResult:
    """RA-03: Identify if a single leaf layer consumes >50% of cumulative activation memory."""
    if not layer_activations:
        return RuleResult("RA-03", RuleStatus.SKIPPED, "Activation Hotspot", "Requires layer profiler (layers.csv).")
    total_act = sum(layer_activations)
    if total_act <= 0.0:
        return RuleResult("RA-03", RuleStatus.PASS, "Negligible Activation", "Total activation memory is minimal.")
    max_act = max(layer_activations)
    ratio = max_act / total_act
    if ratio > 0.50:
        return RuleResult("RA-03", RuleStatus.WARN, "Activation Bottleneck", f"Single layer occupies {ratio:.1%} of activations.")
    return RuleResult("RA-03", RuleStatus.PASS, "Balanced Activations", f"Top layer occupies {ratio:.1%}.")


def check_ra04_memory_leak(rss_samples: List[float]) -> RuleResult:
    """RA-04: Detect monotonic growth in process RSS over benchmark passes."""
    if len(rss_samples) < 10:
        return RuleResult("RA-04", RuleStatus.SKIPPED, "Memory Leak", "Insufficient samples to establish trend.")
    first_half = sum(rss_samples[:5]) / 5.0
    last_half = sum(rss_samples[-5:]) / 5.0
    growth_mb = last_half - first_half
    if growth_mb > 150.0:
        return RuleResult("RA-04", RuleStatus.FAIL, "Suspected Memory Leak", f"RSS grew by {growth_mb:.1f} MB.")
    if growth_mb > 50.0:
        return RuleResult("RA-04", RuleStatus.WARN, "Mild Memory Growth", f"RSS grew by {growth_mb:.1f} MB.")
    return RuleResult("RA-04", RuleStatus.PASS, "Stable Process RSS", f"RSS variation is within margin ({growth_mb:.1f} MB).")


def check_ra05_host_device_transfer(transfer_time_ms: float, total_time_ms: float) -> RuleResult:
    """RA-05: Detect excessive CPU-GPU transfer overhead (>20% total iteration)."""
    if total_time_ms <= 0.0 or transfer_time_ms <= 0.0:
        return RuleResult("RA-05", RuleStatus.SKIPPED, "Transfer Overhead", "No transfer phase data available.")
    ratio = transfer_time_ms / total_time_ms
    if ratio > 0.20:
        return RuleResult("RA-05", RuleStatus.WARN, "Excessive Transfer", f"Transfers consume {ratio:.1%} iteration time.")
    return RuleResult("RA-05", RuleStatus.PASS, "Normal Transfer", f"Transfers consume {ratio:.1%}.")
