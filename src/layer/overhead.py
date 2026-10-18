"""Calibrates profiling overhead and intrinsic timer bias."""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _sync(device: str) -> None:
    if HAS_TORCH and ("cuda" in device.lower() or "t4" in device.lower()) and torch.cuda.is_available():
        torch.cuda.synchronize()


def estimate_timer_bias(
    device: str = "cpu",
    n_layers: int = 64,
    n_runs: int = 30,
    n_warmup: int = 5,
    tiny_kernel: bool = False,
) -> float:
    """
    Estimate the measurement error that sits INSIDE each layer's timed interval (ms per layer).

    Profiles a chain of layers that compute (almost) nothing with the real layer profiler: the
    time it reports for them is the profiler's own error.
      - tiny_kernel=False: chain of nn.Identity (no kernel) -> subtracted from CPU layer times.
      - tiny_kernel=True : chain of ReLU on a 1-element tensor -> each layer launches the smallest
        possible kernel; gives the real floor of CUDA event timing (incl. launch cost).
    """
    if not HAS_TORCH:
        return 0.0
    from src.layer.plugin import LayerProfilerPlugin  # lazy: plugin imports this module

    dev_str = "cuda" if ("cuda" in device.lower() or "t4" in device.lower()) and torch.cuda.is_available() else "cpu"
    make_layer = nn.ReLU if tiny_kernel else nn.Identity
    chain = nn.Sequential(*[make_layer() for _ in range(n_layers)]).to(dev_str)
    x = torch.zeros(1, device=dev_str)

    plugin = LayerProfilerPlugin(subtract_timer_bias=False, exclude_types=(), gpu_queue_prefill=False)
    plugin.attach(chain, dev_str)
    with torch.no_grad():
        for i in range(n_warmup + n_runs):
            plugin.start(i)
            chain(x)
            plugin.stop()
    plugin.detach()
    df = plugin.to_dataframe()
    measured = df.loc[df["iteration"] >= n_warmup, "raw_time_ms"] if not df.empty else []
    bias = float(measured.median()) if len(measured) else 0.0
    logger.info("Estimated timer bias on %s: %.2f us/layer", dev_str, bias * 1000.0)
    return bias


def calibrate_overhead_details(
    model: Any,
    inputs: Any,
    n_warmup: int = 5,
    n_measure: int = 20,
    device: str = "cpu",
    exclude_types: Optional[Tuple[type, ...]] = None,
) -> Dict[str, float]:
    """
    Measure how much the layer profiler slows the forward pass down (for TN2).

    Runs INTERLEAVED: no hooks -> with the real profiler -> no hooks -> ... so that warming up
    does not bias one side, then compares medians. The denominator is the number of modules
    that actually carry hooks. `inputs` may be a Tensor, tuple (*args) or dict (**kwargs).
    """
    if not HAS_TORCH or not isinstance(model, nn.Module):
        return {"t_clean_ms": 0.0, "t_hooked_ms": 0.0, "n_hooked": 0, "overhead_pct": 0.0, "per_layer_ms": 0.0}
    from src.layer.flops import run_model
    from src.layer.plugin import LayerProfilerPlugin  # lazy: plugin imports this module

    # Queue prefill off: we want the REAL cost of the hooks during a normal run
    plugin = LayerProfilerPlugin(subtract_timer_bias=False, exclude_types=exclude_types, gpu_queue_prefill=False)
    clean_times: List[float] = []
    hooked_times: List[float] = []
    n_hooked = 0
    with torch.no_grad():
        for _ in range(n_warmup):
            run_model(model, inputs)
        _sync(device)
        for i in range(n_measure):
            _sync(device)
            t0 = time.perf_counter()
            run_model(model, inputs)
            _sync(device)
            clean_times.append((time.perf_counter() - t0) * 1000.0)

            plugin.attach(model, device)
            t0 = time.perf_counter()
            plugin.start(i)
            run_model(model, inputs)
            plugin.stop()
            hooked_times.append((time.perf_counter() - t0) * 1000.0)
            n_hooked = plugin.n_hooked
            plugin.detach()

    t_clean, t_hooked = float(np.median(clean_times)), float(np.median(hooked_times))
    diff = max(0.0, t_hooked - t_clean)
    details = {
        "t_clean_ms": t_clean,
        "t_hooked_ms": t_hooked,
        "n_hooked": n_hooked,
        "overhead_pct": diff / t_clean * 100.0 if t_clean > 0 else 0.0,
        "per_layer_ms": diff / max(n_hooked, 1),
    }
    logger.info("[calibrate] no_hook=%.3fms | with_hook=%.3fms | n_hooked=%d | overhead=%.1f%%",
                t_clean, t_hooked, n_hooked, details["overhead_pct"])
    return details


class OverheadCalibrator:
    """Legacy helper for backward compatibility."""

    def __init__(self, iterations: int = 100) -> None:
        self.iterations = iterations

    def calibrate(self) -> float:
        return estimate_timer_bias("cpu") * 1_000_000.0
