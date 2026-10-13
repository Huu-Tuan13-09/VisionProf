"""Full-model latency benchmarking, percentiles (P50..P99), throughput and cold-start tracking."""

import contextlib
import csv
import gc
import json
import logging
import os
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.interfaces import ModelAdapter, ProfilerPlugin
from src.core.types import BenchmarkMetric, BenchmarkSummary

logger = logging.getLogger(__name__)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class ModelBenchmark:
    """Benchmark harness executing warmup, latency measurement passes, and metrics aggregation."""

    def __init__(self, warmup_iters: int = 15, measure_iters: int = 50) -> None:
        self.warmup_iters = warmup_iters
        self.measure_iters = measure_iters
        self._metrics: List[BenchmarkMetric] = []
        self._cold_start_ms: float = 0.0
        self._peak_vram_mb: float = 0.0
        self._reserved_vram_mb: float = 0.0
        self._rss_mb: float = 0.0
        self._params_m: float = 0.0

    def run_warmup(
        self,
        adapter: ModelAdapter,
        model: Any,
        batch_size: int,
        device: str,
        precision: str = "fp32",
        **kwargs: Any,
    ) -> None:
        """Execute warmup iterations to prime CUDA caches and JIT kernels."""
        inputs = adapter.make_input(batch_size=batch_size, device=device, precision=precision, **kwargs)
        is_cuda = "cuda" in device.lower() or "t4" in device.lower()
        infer_ctx = torch.inference_mode() if HAS_TORCH and getattr(adapter, "is_torch", True) else contextlib.nullcontext()

        logger.info("Executing %d warmup iterations...", self.warmup_iters)
        with infer_ctx:
            for i in range(self.warmup_iters):
                t0 = time.perf_counter()
                _ = adapter.forward(model, inputs)
                if is_cuda and HAS_TORCH and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                if i == 0:
                    self._cold_start_ms = round((t1 - t0) * 1000.0, 4)

    def measure(
        self,
        adapter: ModelAdapter,
        model: Any,
        batch_size: int,
        device: str,
        precision: str = "fp32",
        plugins: Optional[List[ProfilerPlugin]] = None,
        **kwargs: Any,
    ) -> List[BenchmarkMetric]:
        """Execute benchmark iterations with active plugins."""
        self._metrics.clear()
        active_plugins = plugins or []
        inputs = adapter.make_input(batch_size=batch_size, device=device, precision=precision, **kwargs)
        is_cuda = ("cuda" in device.lower() or "t4" in device.lower()) and HAS_TORCH and torch.cuda.is_available()

        gc.collect()
        if is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        if hasattr(model, "parameters"):
            try:
                self._params_m = round(sum(p.numel() for p in model.parameters()) / 1e6, 3)
            except Exception:
                self._params_m = 0.0
        else:
            self._params_m = 0.0

        infer_ctx = torch.inference_mode() if HAS_TORCH and getattr(adapter, "is_torch", True) else contextlib.nullcontext()
        logger.info("Executing %d measurement iterations on %s...", self.measure_iters, device)
        with infer_ctx:
            for i in range(self.measure_iters):
                for p in active_plugins:
                    p.start(iteration=i)

                t0 = time.perf_counter()
                _ = adapter.forward(model, inputs)
                if is_cuda:
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                for p in active_plugins:
                    p.stop()

                elapsed_ms = (t1 - t0) * 1000.0
                self._metrics.append(
                    BenchmarkMetric(iteration=i, latency_ms=round(elapsed_ms, 4), timestamp=time.time())
                )

        if is_cuda:
            self._peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
            self._reserved_vram_mb = round(torch.cuda.max_memory_reserved() / (1024 * 1024), 2)
        else:
            self._peak_vram_mb = 0.0
            self._reserved_vram_mb = 0.0

        if HAS_PSUTIL:
            try:
                self._rss_mb = round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2)
            except Exception:
                self._rss_mb = 0.0

        return list(self._metrics)

    def summarize(
        self,
        batch_size: int,
        peak_vram_mb: float = 0.0,
        rss_mb: float = 0.0,
        reserved_vram_mb: float = 0.0,
        params_m: Optional[float] = None,
    ) -> BenchmarkSummary:
        """Compute P50, P90, P95, P99, Throughput FPS, and tail ratio across 12 metrics."""
        latencies = [m.latency_ms for m in self._metrics]
        effective_peak_vram = peak_vram_mb if peak_vram_mb > 0 else self._peak_vram_mb
        effective_reserved = reserved_vram_mb if reserved_vram_mb > 0 else self._reserved_vram_mb
        effective_rss = rss_mb if rss_mb > 0 else self._rss_mb
        effective_params = params_m if params_m is not None else self._params_m

        if not latencies:
            return BenchmarkSummary(
                mean_ms=0.0,
                p50_ms=0.0,
                p90_ms=0.0,
                p95_ms=0.0,
                p99_ms=0.0,
                tail_ratio=1.0,
                throughput_fps=0.0,
                peak_vram_mb=effective_peak_vram,
                rss_mb=effective_rss,
                reserved_vram_mb=effective_reserved,
                std_ms=0.0,
                params_m=effective_params,
                cold_start_ms=self._cold_start_ms,
                extra={"cold_start_ms": self._cold_start_ms},
            )

        mean_ms = statistics.mean(latencies)
        std_ms = statistics.stdev(latencies) if len(latencies) >= 2 else 0.0

        if HAS_NUMPY:
            p50_ms, p90_ms, p95_ms, p99_ms = np.percentile(latencies, [50, 90, 95, 99])
        else:
            latencies_sorted = sorted(latencies)
            n = len(latencies_sorted)
            p50_ms = statistics.median(latencies_sorted)
            p90_ms = latencies_sorted[int(n * 0.90)]
            p95_ms = latencies_sorted[int(n * 0.95)]
            p99_ms = latencies_sorted[min(n - 1, int(n * 0.99))]

        tail_ratio = round(float(p99_ms / p50_ms), 3) if p50_ms > 0 else 1.0
        throughput_fps = round((batch_size * 1000.0) / mean_ms, 2) if mean_ms > 0 else 0.0

        return BenchmarkSummary(
            mean_ms=round(float(mean_ms), 4),
            p50_ms=round(float(p50_ms), 4),
            p90_ms=round(float(p90_ms), 4),
            p95_ms=round(float(p95_ms), 4),
            p99_ms=round(float(p99_ms), 4),
            tail_ratio=tail_ratio,
            throughput_fps=throughput_fps,
            peak_vram_mb=round(float(effective_peak_vram), 2),
            rss_mb=round(float(effective_rss), 2),
            reserved_vram_mb=round(float(effective_reserved), 2),
            std_ms=round(float(std_ms), 4),
            params_m=round(float(effective_params), 3),
            cold_start_ms=round(float(self._cold_start_ms), 4),
            extra={"cold_start_ms": round(float(self._cold_start_ms), 4)},
        )

    def save(self, out_dir: Path, summary: BenchmarkSummary) -> None:
        """Save benchmark.csv and summary.json."""
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_file = out_dir / "benchmark.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "latency_ms", "timestamp"])
            for m in self._metrics:
                writer.writerow([m.iteration, m.latency_ms, m.timestamp])

        json_file = out_dir / "summary.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2)

        logger.info("Saved benchmark results and summary to %s", out_dir)
