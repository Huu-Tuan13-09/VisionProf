"""Calibrates profiling overhead introduced by hook execution on leaf modules."""

import time
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)


class OverheadCalibrator:
    """Measures and calibrates CPU/GPU profiling hook overhead."""

    def __init__(self, iterations: int = 100) -> None:
        self.iterations = iterations
        self._mean_hook_overhead_ns: float = 0.0

    def calibrate(self) -> float:
        """Run benchmark iterations on empty pre/post hook callbacks."""
        timestamps: List[int] = []

        def dummy_hook() -> None:
            _ = time.perf_counter_ns()

        for _ in range(self.iterations):
            t0 = time.perf_counter_ns()
            dummy_hook()
            t1 = time.perf_counter_ns()
            timestamps.append(t1 - t0)

        # Drop first 10 warmup passes
        valid_samples = timestamps[10:] if len(timestamps) > 10 else timestamps
        self._mean_hook_overhead_ns = sum(valid_samples) / float(len(valid_samples))
        logger.info("Hook overhead calibrated: %.2f ns/hook", self._mean_hook_overhead_ns)
        return self._mean_hook_overhead_ns

    @property
    def hook_overhead_ms(self) -> float:
        """Return calibrated overhead in milliseconds."""
        return self._mean_hook_overhead_ns / 1_000_000.0

    def deduct_overhead(self, measured_ms: float, hook_count: int = 1) -> float:
        """Subtract expected hook overhead from raw measurement."""
        total_overhead = self.hook_overhead_ms * hook_count
        return max(0.0, measured_ms - total_overhead)
