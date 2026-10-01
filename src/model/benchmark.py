"""Full-model latency benchmarking, percentiles (P50..P99), throughput and cold-start tracking."""

import csv
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.interfaces import ModelAdapter, ProfilerPlugin
from src.core.types import BenchmarkMetric, BenchmarkSummary

logger = logging.getLogger(__name__)

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

    def run_warmup(
        self,
        adapter: ModelAdapter,
        model: Any,
        batch_size: int,
        device: str,
        precision: str = "fp32",
    ) -> None:
        """Execute warmup iterations to prime CUDA caches and JIT kernels."""
        inputs = adapter.make_input(batch_size=batch_size, device=device, precision=precision)
        is_cuda = "cuda" in device.lower() or "t4" in device.lower()

        logger.info("Executing %d warmup iterations...", self.warmup_iters)
        for i in range(self.warmup_iters):
            t0 = time.perf_counter()
            _ = adapter.forward(model, inputs)
            if is_cuda and HAS_TORCH:
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
    ) -> List[BenchmarkMetric]:
        """Execute benchmark iterations with active plugins."""
        self._metrics.clear()
        active_plugins = plugins or []
        inputs = adapter.make_input(batch_size=batch_size, device=device, precision=precision)
        is_cuda = "cuda" in device.lower() or "t4" in device.lower()

        logger.info("Executing %d measurement iterations on %s...", self.measure_iters, device)
        for i in range(self.measure_iters):
            for p in active_plugins:
                p.start(iteration=i)

            t0 = time.perf_counter()
            _ = adapter.forward(model, inputs)
            if is_cuda and HAS_TORCH:
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            for p in active_plugins:
                p.stop()

            elapsed_ms = (t1 - t0) * 1000.0
            self._metrics.append(
                BenchmarkMetric(iteration=i, latency_ms=round(elapsed_ms, 4), timestamp=time.time())
            )

        return list(self._metrics)

    def summarize(
        self,
        batch_size: int,
        peak_vram_mb: float = 0.0,
        rss_mb: float = 0.0,
        reserved_vram_mb: float = 0.0,
    ) -> BenchmarkSummary:
        """Compute P50, P90, P95, P99, Throughput FPS, and tail ratio."""
        latencies = [m.latency_ms for m in self._metrics]
        if not latencies:
            return BenchmarkSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        latencies.sort()
        mean_ms = statistics.mean(latencies)
        p50_ms = statistics.median(latencies)
        p90_ms = latencies[int(len(latencies) * 0.90)]
        p95_ms = latencies[int(len(latencies) * 0.95)]
        p99_ms = latencies[min(len(latencies) - 1, int(len(latencies) * 0.99))]
        tail_ratio = round(p99_ms / p50_ms, 3) if p50_ms > 0 else 1.0
        throughput_fps = round((batch_size * 1000.0) / mean_ms, 2) if mean_ms > 0 else 0.0

        return BenchmarkSummary(
            mean_ms=round(mean_ms, 4),
            p50_ms=round(p50_ms, 4),
            p90_ms=round(p90_ms, 4),
            p95_ms=round(p95_ms, 4),
            p99_ms=round(p99_ms, 4),
            tail_ratio=tail_ratio,
            throughput_fps=throughput_fps,
            peak_vram_mb=peak_vram_mb,
            rss_mb=rss_mb,
            reserved_vram_mb=reserved_vram_mb,
            extra={"cold_start_ms": self._cold_start_ms},
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
