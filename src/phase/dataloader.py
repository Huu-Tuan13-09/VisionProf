"""DataLoader sentinel measuring data retrieval and I/O stalls (filtering batch 0)."""

import logging
import time
from typing import Any, Generator, Iterable, Iterator, List, Optional

logger = logging.getLogger(__name__)


class DataLoaderSentinel:
    """Wraps an iterable DataLoader to measure I/O wait time between compute blocks."""

    def __init__(self, dataloader: Iterable[Any]) -> None:
        self.dataloader = dataloader
        self._io_durations_ms: List[float] = []

    def __iter__(self) -> Iterator[Any]:
        iterator = iter(self.dataloader)
        batch_idx = 0

        first_batch_ms = 0.0
        while True:
            t0 = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                if batch_idx == 1 and not self._io_durations_ms:
                    self._io_durations_ms.append(round(first_batch_ms, 4))
                    logger.debug("Single batch dataset detected; retaining batch 0 I/O metric.")
                break
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0

            # Discard batch 0 to ignore cold-start worker spawn overhead if multiple batches exist
            if batch_idx > 0:
                self._io_durations_ms.append(round(elapsed_ms, 4))
            else:
                first_batch_ms = elapsed_ms
                logger.debug("Cold-start batch 0 discarded from steady-state I/O metrics: %.2f ms", elapsed_ms)

            batch_idx += 1
            yield batch

    @property
    def io_durations_ms(self) -> List[float]:
        """Return list of steady-state I/O wait durations."""
        return list(self._io_durations_ms)

    def mean_io_ms(self) -> float:
        """Compute mean steady-state I/O duration in milliseconds."""
        if not self._io_durations_ms:
            return 0.0
        return sum(self._io_durations_ms) / float(len(self._io_durations_ms))
