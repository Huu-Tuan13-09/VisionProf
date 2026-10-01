"""PhaseProfilerPlugin implementing stage/phase level profiling to phases.csv."""

import csv
import logging
from pathlib import Path
from typing import Any, List

from src.core.interfaces import ProfilerPlugin
from src.core.types import PhaseRecord
from src.phase.timer import PhaseTimer

logger = logging.getLogger(__name__)


class PhaseProfilerPlugin(ProfilerPlugin):
    """Profiler plugin capturing stage durations across training and inference pipelines."""

    name: str = "phase"

    def __init__(self) -> None:
        self.timer = PhaseTimer()
        self.device: str = "cpu"
        self._current_iter: int = 0
        self._iter_start_time: float = 0.0
        self._recorded_in_current_iter: bool = False

    def attach(self, model: Any, device: str) -> None:
        """Initialize plugin state with target model and device."""
        self.device = device
        self.timer.clear()
        logger.info("Attached PhaseProfilerPlugin on %s", device)

    def start(self, iteration: int) -> None:
        """Signal start of iteration."""
        import time
        self._current_iter = iteration
        self.timer.set_iteration(iteration)
        self._iter_start_time = time.perf_counter()
        self._recorded_in_current_iter = False

    def stop(self) -> None:
        """Signal end of iteration, recording default forward phase if none recorded."""
        import time
        if not self._recorded_in_current_iter and self._iter_start_time > 0.0:
            elapsed_ms = (time.perf_counter() - self._iter_start_time) * 1000.0
            self.record_phase("forward", elapsed_ms)

    def record_phase(self, phase_name: str, duration_ms: float) -> None:
        """Directly append an explicit phase record."""
        self._recorded_in_current_iter = True
        self.timer.add_record(
            PhaseRecord(iteration=self._current_iter, phase=phase_name, time_ms=round(duration_ms, 4))
        )

    def save(self, out_dir: Path) -> None:
        """Write collected phase records to phases.csv."""
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_file = out_dir / "phases.csv"
        records = self.timer.get_records()

        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "phase", "time_ms"])
            for r in records:
                writer.writerow([r.iteration, r.phase, r.time_ms])

        logger.info("Saved %d phase records to %s", len(records), csv_file)
