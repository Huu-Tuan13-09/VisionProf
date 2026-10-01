"""High-resolution CPU timing utilities for layer and phase measurements."""

import time
from typing import Optional


class CPUTimer:
    """Nanosecond-resolution CPU timer using time.perf_counter_ns()."""

    def __init__(self) -> None:
        self._start_ns: Optional[int] = None
        self._elapsed_ms: float = 0.0

    def start(self) -> None:
        """Record the start timestamp in nanoseconds."""
        self._start_ns = time.perf_counter_ns()

    def stop(self) -> float:
        """Stop timing and return the elapsed time in milliseconds."""
        if self._start_ns is None:
            return 0.0
        elapsed_ns = time.perf_counter_ns() - self._start_ns
        self._elapsed_ms = elapsed_ns / 1_000_000.0
        self._start_ns = None
        return self._elapsed_ms

    @property
    def elapsed_ms(self) -> float:
        """Return the most recent elapsed duration in milliseconds."""
        return self._elapsed_ms
