"""Context manager and tracker for pipeline and training execution phases.

GPU sync policy: when device is 'cuda', torch.cuda.synchronize() is called
inside measure() immediately before recording t1 so the elapsed time reflects
true kernel completion, not just kernel launch time.
"""

import logging
import time
from contextlib import contextmanager
from typing import Dict, Generator, List, Optional

from src.core.constants import PhaseType
from src.core.types import PhaseRecord

logger = logging.getLogger(__name__)


def _sync_if_cuda(device: str) -> None:
    """Call torch.cuda.synchronize() when device is GPU, no-op otherwise."""
    if "cuda" in device or "t4" in device:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass


class PhaseTimer:
    """Multi-phase timer measuring execution breakdown (forward, backward, opt, data).

    Usage::

        timer = PhaseTimer(device="cuda")
        timer.set_iteration(0)

        with timer.measure("forward"):
            model(inputs)
        with timer.measure("backward"):
            loss.backward()
    """

    def __init__(self, device: str = "cpu") -> None:
        self._records: List[PhaseRecord] = []
        self._current_iter: int = 0
        self._active_start: Optional[float] = None
        self.device: str = device

    def set_iteration(self, iteration: int) -> None:
        """Set current active iteration index."""
        self._current_iter = iteration

    def set_device(self, device: str) -> None:
        """Update device so GPU sync policy is applied correctly."""
        self.device = device

    @contextmanager
    def measure(self, phase_name: str) -> Generator[None, None, None]:
        """Context manager to measure duration of a specific execution phase.

        On CUDA devices, synchronizes the GPU before recording the stop time
        so that async kernel execution is fully accounted for.
        """
        _sync_if_cuda(self.device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _sync_if_cuda(self.device)   # wait for GPU kernels to finish
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0
            self._records.append(
                PhaseRecord(
                    iteration=self._current_iter,
                    phase=phase_name,
                    time_ms=round(elapsed_ms, 4),
                )
            )
            logger.debug("Phase '%s' iter=%d: %.4f ms", phase_name, self._current_iter, elapsed_ms)

    def add_record(self, record: PhaseRecord) -> None:
        """Directly append a PhaseRecord."""
        self._records.append(record)

    def get_records(self) -> List[PhaseRecord]:
        """Return all recorded phase durations."""
        return list(self._records)

    def summary(self) -> Dict[str, float]:
        """Return mean duration in ms for each phase name across all iterations."""
        from collections import defaultdict
        totals: Dict[str, float] = defaultdict(float)
        counts: Dict[str, int] = defaultdict(int)
        for r in self._records:
            totals[r.phase] += r.time_ms
            counts[r.phase] += 1
        return {phase: round(totals[phase] / counts[phase], 4) for phase in totals}

    def clear(self) -> None:
        """Reset all recorded phase entries."""
        self._records.clear()
