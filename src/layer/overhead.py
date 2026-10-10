"""Calibrates profiling overhead and intrinsic timer bias."""

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def estimate_timer_bias(
    device: str = "cpu",
    n_layers: int = 64,
    n_runs: int = 30,
    n_warmup: int = 5,
    tiny_kernel: bool = False,
) -> float:
    """Estimate measurement overhead inside the timing interval (ms per layer)."""
    if not HAS_TORCH:
        return 0.0

    dev_str = "cuda" if "cuda" in device.lower() or "t4" in device.lower() else "cpu"
    if dev_str == "cuda" and not torch.cuda.is_available():
        dev_str = "cpu"

    make_layer = nn.ReLU if tiny_kernel else nn.Identity
    chain = nn.Sequential(*[make_layer() for _ in range(n_layers)]).to(dev_str)
    x = torch.zeros(1, device=dev_str)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            chain(x)
        if dev_str == "cuda":
            torch.cuda.synchronize()

        durations: List[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter_ns()
            chain(x)
            if dev_str == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter_ns()
            durations.append((t1 - t0) / (1_000_000.0 * n_layers))

    bias = float(sorted(durations)[len(durations) // 2]) if durations else 0.0
    logger.info("Estimated timer bias on %s: %.4f ms/layer", dev_str, bias)
    return bias


def calibrate_overhead_details(
    model: Any,
    input_tensor: Any,
    n_warmup: int = 5,
    n_measure: int = 20,
    device: str = "cpu",
) -> Dict[str, float]:
    """Measure how much attaching profiling hooks slows down the forward pass (for TN2)."""
    if not HAS_TORCH or not isinstance(model, nn.Module):
        return {"t_clean_ms": 0.0, "t_hooked_ms": 0.0, "n_hooked": 0, "overhead_pct": 0.0, "per_layer_ms": 0.0}

    dev = torch.device("cuda" if "cuda" in device.lower() or "t4" in device.lower() else "cpu")
    model = model.to(dev).eval()
    inp = input_tensor.to(dev)

    # Count leaf modules
    leaf_modules = [m for m in model.modules() if len(list(m.children())) == 0]
    n_hooked = len(leaf_modules)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(inp)
        if dev.type == "cuda":
            torch.cuda.synchronize()

        clean_times: List[float] = []
        for _ in range(n_measure):
            t0 = time.perf_counter_ns()
            model(inp)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter_ns()
            clean_times.append((t1 - t0) / 1_000_000.0)

        # Hooked run simulation
        hooks = []
        for mod in leaf_modules:
            hooks.append(mod.register_forward_pre_hook(lambda m, i: None))
            hooks.append(mod.register_forward_hook(lambda m, i, o: None))

        hooked_times: List[float] = []
        try:
            for _ in range(n_measure):
                t0 = time.perf_counter_ns()
                model(inp)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter_ns()
                hooked_times.append((t1 - t0) / 1_000_000.0)
        finally:
            for h in hooks:
                h.remove()

    t_clean = sorted(clean_times)[len(clean_times) // 2] if clean_times else 0.0
    t_hooked = sorted(hooked_times)[len(hooked_times) // 2] if hooked_times else 0.0
    overhead_pct = ((t_hooked - t_clean) / t_clean * 100.0) if t_clean > 0 else 0.0
    per_layer_ms = ((t_hooked - t_clean) / n_hooked) if n_hooked > 0 else 0.0

    return {
        "t_clean_ms": round(t_clean, 4),
        "t_hooked_ms": round(t_hooked, 4),
        "n_hooked": n_hooked,
        "overhead_pct": round(overhead_pct, 2),
        "per_layer_ms": round(per_layer_ms, 6),
    }


class OverheadCalibrator:
    """Legacy helper for backward compatibility."""

    def __init__(self, iterations: int = 100) -> None:
        self.iterations = iterations

    def calibrate(self) -> float:
        return estimate_timer_bias("cpu") * 1_000_000.0
