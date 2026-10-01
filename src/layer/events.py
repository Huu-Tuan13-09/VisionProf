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

    def __init__(self, enabled: bool = True) -> None:
        self.enabled: bool = enabled and HAS_TORCH and torch.cuda.is_available()
        self._pending_stacks: Dict[str, List[Tuple[Any, Any]]] = {}
        self._completed_pairs: Dict[str, List[Tuple[Any, Any]]] = {}

    def record_start(self, layer_name: str) -> Optional[Any]:
        """Record the start CUDA event on the current stream."""
        if not self.enabled:
            return None
        start_event = torch.cuda.Event(enable_timing=True)
        stop_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        self._pending_stacks.setdefault(layer_name, []).append((start_event, stop_event))
        return start_event

    def record_stop(self, layer_name: str) -> None:
        """Record the stop CUDA event on the current stream (LIFO matching and pop)."""
        if not self.enabled:
            return
        stack = self._pending_stacks.get(layer_name)
        if not stack:
            return
        start_event, stop_event = stack.pop()
        stop_event.record()
        self._completed_pairs.setdefault(layer_name, []).append((start_event, stop_event))

    def synchronize_and_collect(self) -> Dict[str, float]:
        """Synchronize CUDA stream once and calculate elapsed milliseconds for all layers."""
        durations: Dict[str, float] = {}
        if not self.enabled:
            return durations

        torch.cuda.synchronize()

        for layer_name, event_pairs in self._completed_pairs.items():
            total_ms = 0.0
            for start_evt, stop_evt in event_pairs:
                elapsed = start_evt.elapsed_time(stop_evt)
                total_ms += elapsed
            durations[layer_name] = round(total_ms, 4)

        self._pending_stacks.clear()
        self._completed_pairs.clear()
        return durations

    def reset(self) -> None:
        """Clear all active and completed event pairs."""
        self._pending_stacks.clear()
        self._completed_pairs.clear()
