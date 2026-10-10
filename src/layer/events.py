"""Deferred CUDA Events dispatcher and LIFO event stack for precise GPU timing."""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class DeferredCudaDispatcher:
    """Dispatches and tracks CUDA timing events using a deferred synchronization model."""

    def __init__(self, enabled: bool = True, gpu_idx: int = 0, queue_prefill: bool = True) -> None:
        self.enabled: bool = bool(enabled and HAS_TORCH and torch.cuda.is_available())
        self.gpu_idx: int = gpu_idx
        self.queue_prefill: bool = queue_prefill and self.enabled
        self._event_pool: List[Any] = []
        self._event_cursor: int = 0
        self._stacks: Dict[str, List[Tuple[Any, Any]]] = {}
        self._completed: List[Tuple[str, Any, Any]] = []
        self._prefill_ms: float = 50.0
        self._prefill_ok: Dict[int, bool] = {}

    def _next_events(self) -> Tuple[Any, Any]:
        """Fetch a pair of CUDA events from pool, allocating in batches if necessary."""
        if self._event_cursor + 2 > len(self._event_pool):
            self._event_pool.extend([torch.cuda.Event(enable_timing=True) for _ in range(64)])
        start_evt = self._event_pool[self._event_cursor]
        stop_evt = self._event_pool[self._event_cursor + 1]
        self._event_cursor += 2
        return start_evt, stop_evt

    def record_start(self, layer_name: str) -> Optional[Any]:
        """Record the start CUDA event on the current stream."""
        if not self.enabled:
            return None
        start_evt, stop_evt = self._next_events()
        self._stacks.setdefault(layer_name, []).append((start_evt, stop_evt))
        start_evt.record()
        return start_evt

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

    def prefill_queue(self, iteration: int, enqueue_time_ms: float = 0.0) -> None:
        """Keep GPU busy during queue enqueue to avoid GPU pipeline starvation."""
        if not self.queue_prefill or not self.enabled:
            return
        cycles_per_ms = 1_000_000  # Conservative cycle approximation for GPU sleep
        try:
            if hasattr(torch.cuda, "_sleep"):
                torch.cuda._sleep(int(self._prefill_ms * cycles_per_ms))
        except Exception:
            pass
        if enqueue_time_ms > 0:
            self._prefill_ok[iteration] = enqueue_time_ms < self._prefill_ms
            self._prefill_ms = min(1000.0, max(5.0, 1.5 * enqueue_time_ms + 2.0))

    def synchronize_and_collect(self) -> Dict[Any, float]:
        """Synchronize CUDA stream once and calculate elapsed milliseconds per layer."""
        durations: Dict[Any, float] = {}
        if not self.enabled:
            return durations

        torch.cuda.synchronize(self.gpu_idx)

        for layer_name, call_idx, start_evt, stop_evt in self._completed:
            elapsed = float(start_evt.elapsed_time(stop_evt))
            # Store per-call duration by (layer_name, call_idx)
            durations[(layer_name, call_idx)] = round(elapsed, 5)
            # Also aggregate total duration by layer_name string
            durations[layer_name] = round(durations.get(layer_name, 0.0) + elapsed, 5)

        self._stacks.clear()
        self._completed.clear()
        self._event_cursor = 0
        return durations

    def get_prefill_ok(self) -> Dict[int, bool]:
        """Return history of GPU prefill status per iteration."""
        return dict(self._prefill_ok)

    def reset(self) -> None:
        """Reset internal cursor and stacks."""
        self._stacks.clear()
        self._completed.clear()
        self._event_cursor = 0
        self._prefill_ok.clear()
