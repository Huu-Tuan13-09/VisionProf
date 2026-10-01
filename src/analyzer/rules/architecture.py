"""Architecture and layer structure diagnostic rules (RD-01 to RD-05) for VisionProf v2.0."""

from typing import List, Tuple
from src.core.constants import RuleStatus
from src.core.types import RuleResult


def check_rd01_param_concentration(layer_params: List[Tuple[str, int]]) -> RuleResult:
    """RD-01: Detect parameter concentration (>60% of total weights in a single layer)."""
    if not layer_params:
        return RuleResult("RD-01", RuleStatus.SKIPPED, "Param Concentration", "Requires layer profiler (layers.csv).")
    total_p = sum(p for _, p in layer_params)
    if total_p == 0:
        return RuleResult("RD-01", RuleStatus.PASS, "Parameterless Model", "No parameters recorded.")
    max_name, max_p = max(layer_params, key=lambda item: item[1])
    ratio = max_p / float(total_p)
    if ratio > 0.60:
        return RuleResult("RD-01", RuleStatus.WARN, "Heavy Parameter Layer", f"Layer '{max_name}' holds {ratio:.1%} of parameters.")
    return RuleResult("RD-01", RuleStatus.PASS, "Balanced Parameter Spread", f"Top layer holds {ratio:.1%} of parameters.")


def check_rd02_sub_microsecond_layers(layer_durations_ms: List[Tuple[str, float]]) -> RuleResult:
    """RD-02: Detect sub-microsecond layers (<0.001 ms) where kernel launch and overhead dominate."""
    if not layer_durations_ms:
        return RuleResult("RD-02", RuleStatus.SKIPPED, "Sub-microsecond Layers", "Requires layer profiler (layers.csv).")
    sub_count = sum(1 for _, d in layer_durations_ms if d < 0.001)
    ratio = sub_count / float(len(layer_durations_ms))
    if ratio > 0.30:
        return RuleResult("RD-02", RuleStatus.WARN, "High Sub-Microsecond Layer Ratio", f"{sub_count} layers ({ratio:.1%}) run in <1 microsecond; fusion recommended.")
    return RuleResult("RD-02", RuleStatus.PASS, "Acceptable Layer Granularity", f"{sub_count} sub-microsecond layers detected.")


def check_rd03_memory_compute_disparity(act_mb: float, gflops: float) -> RuleResult:
    """RD-03: Detect memory-bound layers with disproportionately high activation per GFLOP."""
    if gflops <= 0.0 or act_mb <= 0.0:
        return RuleResult("RD-03", RuleStatus.SKIPPED, "Memory-Compute Ratio", "Insufficient layer data.")
    ratio = act_mb / gflops
    if ratio > 50.0:
        return RuleResult("RD-03", RuleStatus.WARN, "High Memory-to-Compute Ratio", f"Ratio is {ratio:.1f} MB/GFLOP (Memory-bound).")
    return RuleResult("RD-03", RuleStatus.PASS, "Balanced Memory-Compute", f"Ratio is {ratio:.1f} MB/GFLOP.")


def check_rd04_layer_bottleneck(layer_times: List[Tuple[str, float]]) -> RuleResult:
    """RD-04: Flag single leaf layer consuming >35% of total model forward latency."""
    if not layer_times:
        return RuleResult("RD-04", RuleStatus.SKIPPED, "Layer Bottleneck", "Requires layer profiler (layers.csv).")
    total_time = sum(t for _, t in layer_times)
    if total_time <= 0.0:
        return RuleResult("RD-04", RuleStatus.PASS, "Zero Latency", "Model latency negligible.")
    top_name, top_time = max(layer_times, key=lambda x: x[1])
    ratio = top_time / total_time
    if ratio > 0.35:
        return RuleResult("RD-04", RuleStatus.WARN, "Single Layer Bottleneck", f"Layer '{top_name}' takes {ratio:.1%} of forward time.")
    return RuleResult("RD-04", RuleStatus.PASS, "Even Latency Distribution", f"Top layer takes {ratio:.1%}.")


def check_rd05_latency_flops_outliers(layer_stats: List[Tuple[str, float, float]]) -> RuleResult:
    """RD-05: Identify outliers where layer takes high latency despite negligible FLOPs."""
    if not layer_stats:
        return RuleResult("RD-05", RuleStatus.SKIPPED, "Latency-FLOPs Outliers", "Requires layer profiler (layers.csv).")
    outliers: List[str] = []
    total_time = sum(t for _, t, _ in layer_stats)
    for name, t, flops in layer_stats:
        time_pct = (t / total_time) * 100.0 if total_time > 0 else 0.0
        if time_pct > 15.0 and flops < 1_000_000:
            outliers.append(name)
    if outliers:
        return RuleResult("RD-05", RuleStatus.WARN, "Inefficient Non-Compute Layers", f"Outliers with high latency but low FLOPs: {', '.join(outliers[:3])}")
    return RuleResult("RD-05", RuleStatus.PASS, "Proportional Latency/FLOPs", "No anomalous low-FLOP bottlenecks detected.")
