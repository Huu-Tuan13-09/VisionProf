"""Context manager and tracker for pipeline and training execution phases."""

import time
from contextlib import contextmanager
from typing import Dict, Generator, List, Optional

from src.core.constants import PhaseType
from src.core.types import PhaseRecord


class PhaseTimer:
    """Multi-phase timer measuring execution breakdown (forward, backward, opt, data)."""

    def __init__(self) -> None:
        self._records: List[PhaseRecord] = []
        self._current_iter: int = 0
        self._active_start: Optional[float] = None

    def set_iteration(self, iteration: int) -> None:
        """Set current active iteration index."""
        self._current_iter = iteration

    @contextmanager
    def measure(self, phase_name: str) -> Generator[None, None, None]:
        """Context manager to measure duration of a specific execution phase."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0
            self._records.append(
                PhaseRecord(
                    iteration=self._current_iter,
                    phase=phase_name,
                    time_ms=round(elapsed_ms, 4),
                )
            )

    def add_record(self, record: PhaseRecord) -> None:
        """Directly append a PhaseRecord."""
        self._records.append(record)

    def get_records(self) -> List[PhaseRecord]:
        """Return all recorded phase durations."""
        return list(self._records)

    def clear(self) -> None:
        """Reset all recorded phase entries."""
        self._records.clear()
