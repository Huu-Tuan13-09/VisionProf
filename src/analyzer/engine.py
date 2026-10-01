"""DiagnosticEngine coordinating rule evaluations from standardized results directories."""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.analyzer.rules import (
    check_ra01_oom_risk,
    check_ra02_fragmentation,
    check_ra03_activation_hotspot,
    check_ra04_memory_leak,
    check_ra05_host_device_transfer,
    check_rb01_gpu_utilization,
    check_rb02_tail_latency,
    check_rb03_backward_bottleneck,
    check_rb04_compute_efficiency,
    check_rb05_optimizer_stall,
    check_rc01_io_ratio,
    check_rc02_worker_variance,
    check_rc03_gpu_starvation,
    check_rc04_batch_underutilization,
    check_rd01_param_concentration,
    check_rd02_sub_microsecond_layers,
    check_rd03_memory_compute_disparity,
    check_rd04_layer_bottleneck,
    check_rd05_latency_flops_outliers,
)
from src.core.constants import RuleStatus
from src.core.types import RuleResult

logger = logging.getLogger(__name__)


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None and str(val).strip() != "" else default
    except (ValueError, TypeError):
        return default


class DiagnosticEngine:
    """Loads standardized results artifacts and applies all 19 heuristics."""

    def __init__(self) -> None:
        pass

    def evaluate_directory(self, run_dir: Path) -> List[RuleResult]:
        """Execute full suite of 19 diagnostic rules against run directory artifacts."""
        results: List[RuleResult] = []

        summary_file = run_dir / "summary.json"
        summary_data: Dict[str, Any] = {}
        if summary_file.exists() and summary_file.stat().st_size > 0:
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    summary_data = json.load(f)
            except Exception:
                pass

        env_file = run_dir / "env.json"
        env_data: Dict[str, Any] = {}
        if env_file.exists() and env_file.stat().st_size > 0:
            try:
                with open(env_file, "r", encoding="utf-8") as f:
                    env_data = json.load(f)
            except Exception:
                pass

        config_file = run_dir / "config.json"
        config_data: Dict[str, Any] = {}
        if config_file.exists() and config_file.stat().st_size > 0:
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
            except Exception:
                pass

        # 1. Compute & Resource Metrics
        p50 = _safe_float(summary_data.get("p50_ms", 0.0))
        p99 = _safe_float(summary_data.get("p99_ms", 0.0))
        fps = _safe_float(summary_data.get("throughput_fps", 0.0))
        peak_vram = _safe_float(summary_data.get("peak_vram_mb", 0.0))
        reserved_vram = _safe_float(summary_data.get("reserved_vram_mb", 0.0))
        total_vram = _safe_float(env_data.get("gpu", {}).get("vram_total_gb", 0.0)) * 1024.0
        batch_size = int(config_data.get("batch_size", 1))

        results.append(check_ra01_oom_risk(peak_vram, total_vram))
        results.append(check_ra02_fragmentation(peak_vram, reserved_vram))
        results.append(check_rb02_tail_latency(p50, p99))
        results.append(check_rc04_batch_underutilization(batch_size, fps))

        # 2. Resource Sampling Data
        res_file = run_dir / "resources.csv"
        rss_samples: List[float] = []
        gpu_utils: List[float] = []
        if res_file.exists() and res_file.stat().st_size > 0:
            try:
                with open(res_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        rss_samples.append(_safe_float(row.get("rss_mb", 0.0)))
                        gpu_utils.append(_safe_float(row.get("gpu_util_pct", 0.0)))
            except Exception:
                pass

        results.append(check_ra04_memory_leak(rss_samples))
        mean_gpu = (sum(gpu_utils) / len(gpu_utils)) if gpu_utils else 0.0
        results.append(check_rb01_gpu_utilization(mean_gpu))

        # 3. Layer Breakdown Data (Graceful SKIPPED if missing or empty)
        layer_file = run_dir / "layers.csv"
        if layer_file.exists() and layer_file.stat().st_size > 0:
            results.extend(self._evaluate_layer_rules(layer_file, p50))
        else:
            logger.info("layers.csv missing or empty in %s; marking layer rules as SKIPPED.", run_dir)
            for rule_id, name in [
                ("RA-03", "Activation Hotspot"),
                ("RB-04", "Compute Efficiency ms/GFLOP"),
                ("RD-01", "Param Concentration"),
                ("RD-02", "Sub-microsecond Layers"),
                ("RD-03", "Memory-to-Compute Disparity"),
                ("RD-04", "Layer Bottleneck"),
                ("RD-05", "Latency-FLOPs Outliers"),
            ]:
                results.append(RuleResult(rule_id, RuleStatus.SKIPPED, name, "Requires layer profiler (layers.csv)."))

        # 4. Phase Breakdown Data (Graceful SKIPPED if missing or empty)
        phase_file = run_dir / "phases.csv"
        if phase_file.exists() and phase_file.stat().st_size > 0:
            results.extend(self._evaluate_phase_rules(phase_file))
        else:
            for rule_id, name in [
                ("RA-05", "Host-Device Transfer"),
                ("RB-03", "Backward Bottleneck"),
                ("RB-05", "Optimizer Stall"),
                ("RC-01", "I/O Ratio"),
                ("RC-02", "Worker Variance"),
                ("RC-03", "GPU Starvation"),
            ]:
                results.append(RuleResult(rule_id, RuleStatus.SKIPPED, name, "Requires phase profiler (phases.csv)."))

        return results

    def _evaluate_layer_rules(self, layer_file: Path, total_latency: float) -> List[RuleResult]:
        activations: List[float] = []
        durations: List[Tuple[str, float]] = []
        params: List[Tuple[str, int]] = []
        stats: List[Tuple[str, float, float]] = []
        total_flops = 0.0

        try:
            with open(layer_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get("layer_name", "")
                    t_ms = _safe_float(row.get("time_ms", 0.0))
                    act = _safe_float(row.get("activation_mb", 0.0))
                    flops = _safe_float(row.get("flops", 0.0))
                    p = int(_safe_float(row.get("params", 0)))

                    activations.append(act)
                    durations.append((name, t_ms))
                    params.append((name, p))
                    stats.append((name, t_ms, flops))
                    total_flops += flops
        except Exception:
            pass

        total_gflops = total_flops / 1e9
        return [
            check_ra03_activation_hotspot(activations),
            check_rb04_compute_efficiency(total_gflops, total_latency),
            check_rd01_param_concentration(params),
            check_rd02_sub_microsecond_layers(durations),
            check_rd03_memory_compute_disparity(sum(activations), total_gflops),
            check_rd04_layer_bottleneck(durations),
            check_rd05_latency_flops_outliers(stats),
        ]

    def _evaluate_phase_rules(self, phase_file: Path) -> List[RuleResult]:
        fwd_ms = 0.0
        bwd_ms = 0.0
        opt_ms = 0.0
        io_ms = 0.0
        total_ms = 0.0
        io_samples: List[float] = []

        try:
            with open(phase_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    phase = row.get("phase", "")
                    t_ms = _safe_float(row.get("time_ms", 0.0))
                    total_ms += t_ms
                    if "forward" in phase:
                        fwd_ms += t_ms
                    elif "backward" in phase:
                        bwd_ms += t_ms
                    elif "opt" in phase:
                        opt_ms += t_ms
                    elif "data" in phase or "dataloader" in phase:
                        io_ms += t_ms
                        io_samples.append(t_ms)
        except Exception:
            pass

        return [
            check_ra05_host_device_transfer(0.0, total_ms),
            check_rb03_backward_bottleneck(fwd_ms, bwd_ms),
            check_rb05_optimizer_stall(opt_ms, total_ms),
            check_rc01_io_ratio(io_ms, total_ms),
            check_rc02_worker_variance(io_samples),
            check_rc03_gpu_starvation(io_ms, fwd_ms + bwd_ms),
        ]
