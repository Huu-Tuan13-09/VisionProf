"""Deferred CUDA Events dispatcher and LIFO event stack for precise GPU timing."""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class DeferredCudaDispatcher:
    """
    Records CUDA events inside hooks without blocking the CPU, then synchronizes once per iteration.

    queue_prefill: before each forward the GPU is kept busy (torch.cuda._sleep) long enough for the
    CPU to enqueue every kernel. The GPU then runs back-to-back, so the time between two events is
    real GPU compute, not GPU idle time waiting for the CPU (same view as Nsight / torch.profiler).
    The prefill length adapts to the measured enqueue time of the previous iteration.
    """

    def __init__(self, enabled: bool = True, gpu_idx: int = 0, queue_prefill: bool = True) -> None:
        self.enabled: bool = bool(enabled and HAS_TORCH and torch.cuda.is_available())
        self.gpu_idx: int = gpu_idx
        self.queue_prefill: bool = queue_prefill and self.enabled
        self._event_pool: List[Any] = []
        self._event_cursor: int = 0
        self._stacks: Dict[str, List[Tuple[Any, Any]]] = {}
        self._completed: List[Tuple[str, int, Any, Any]] = []
        self._iter_events: Optional[Tuple[Any, Any]] = None
        self._iter_t0_ns: int = 0
        self._prefill_ms: float = 50.0
        self._cycles_per_ms: Optional[float] = None
        self._prefill_ok: Dict[int, bool] = {}

    def _next_events(self) -> Tuple[Any, Any]:
        """Fetch a pair of CUDA events from the pool (reused across iterations)."""
        if self._event_cursor + 2 > len(self._event_pool):
            self._event_pool.extend([torch.cuda.Event(enable_timing=True) for _ in range(64)])
        start_evt = self._event_pool[self._event_cursor]
        stop_evt = self._event_pool[self._event_cursor + 1]
        self._event_cursor += 2
        return start_evt, stop_evt

    def _gpu_cycles_per_ms(self) -> float:
        """Measure once how many GPU cycles torch.cuda._sleep spends per millisecond."""
        if self._cycles_per_ms is None:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda._sleep(1_000)
            start.record()
            torch.cuda._sleep(2_000_000)
            end.record()
            torch.cuda.synchronize(self.gpu_idx)
            self._cycles_per_ms = 2_000_000 / max(start.elapsed_time(end), 1e-3)
        return self._cycles_per_ms

    def begin_iteration(self) -> None:
        """Drain the GPU, prefill the queue, then record the whole-forward start event."""
        if not self.enabled:
            return
        self._stacks.clear()
        self._completed.clear()
        self._event_cursor = 0
        torch.cuda.synchronize(self.gpu_idx)
        if self.queue_prefill and hasattr(torch.cuda, "_sleep"):
            torch.cuda._sleep(int(self._prefill_ms * self._gpu_cycles_per_ms()))
        self._iter_events = self._next_events()
        self._iter_events[0].record()
        self._iter_t0_ns = time.perf_counter_ns()

    def record_start(self, layer_name: str) -> None:
        """Record the start CUDA event on the current stream."""
        if not self.enabled:
            return
        start_evt, stop_evt = self._next_events()
        self._stacks.setdefault(layer_name, []).append((start_evt, stop_evt))
        start_evt.record()

    def record_stop(self, layer_name: str, call_idx: int = 0) -> None:
        """Record the stop CUDA event on the current stream (LIFO pop)."""
        if not self.enabled:
            return
        stack = self._stacks.get(layer_name)
        if not stack:
            return
        start_evt, stop_evt = stack.pop()
        stop_evt.record()
        self._completed.append((layer_name, call_idx, start_evt, stop_evt))

    def end_iteration(self, iteration: int) -> Tuple[Dict[Tuple[str, int], float], float]:
        """
        Record the whole-forward stop event, synchronize ONCE and resolve every event.

        Returns: ({(layer_name, call_idx): ms}, whole-forward GPU time in ms).
        """
        if not self.enabled or self._iter_events is None:
            return {}, 0.0
        iter_start, iter_stop = self._iter_events
        iter_stop.record()
        if self.queue_prefill:
            # The CPU must finish enqueueing before the prefill drains, otherwise the GPU waits.
            enqueue_ms = (time.perf_counter_ns() - self._iter_t0_ns) / 1e6
            self._prefill_ok[iteration] = enqueue_ms < self._prefill_ms
            self._prefill_ms = min(1000.0, max(5.0, 1.5 * enqueue_ms + 2.0))
        torch.cuda.synchronize(self.gpu_idx)
        durations = {(name, idx): float(s.elapsed_time(e)) for name, idx, s, e in self._completed}
        total_ms = float(iter_start.elapsed_time(iter_stop))
        self._stacks.clear()
        self._completed.clear()
        self._iter_events = None
        return durations, total_ms

    def get_prefill_ok(self) -> Dict[int, bool]:
        """Return history of GPU prefill status per iteration."""
        return dict(self._prefill_ok)

    def reset(self) -> None:
        """Reset internal cursor and stacks."""
        self._stacks.clear()
        self._completed.clear()
        self._event_cursor = 0
        self._iter_events = None
        self._prefill_ok.clear()
