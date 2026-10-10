"""FLOPs estimation and theoretical computational load calculation per layer."""

from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import pandas as pd

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
    aten = getattr(torch.ops, "aten", None)
except ImportError:
    HAS_TORCH = False
    aten = None

try:
    from torch.utils.flop_counter import FlopCounterMode, sdpa_flop_count
    HAS_FLOP_COUNTER = True
except ImportError:
    HAS_FLOP_COUNTER = False
    FlopCounterMode = None  # type: ignore
    sdpa_flop_count = None


def _native_mha_flops(query: Any, key: Any, value: Any, embed_dim: Any, num_head: Any, *args: Any, **kwargs: Any) -> int:
    b, l, e = query
    s = key[1]
    return 4 * b * l * e * e + 4 * b * s * e * e + 4 * b * l * s * e


def _cpu_sdpa_flops(query: Any, key: Any, value: Any, *args: Any, **kwargs: Any) -> int:
    if HAS_TORCH and sdpa_flop_count is not None:
        return sdpa_flop_count(query, key, value)
    return 0


_CUSTOM_FLOP_FORMULAS: Dict[Any, Callable] = {}
if HAS_TORCH and aten is not None:
    for op, fn in [
        (getattr(aten, "_native_multi_head_attention", None), _native_mha_flops),
        (getattr(aten, "_scaled_dot_product_flash_attention_for_cpu", None), _cpu_sdpa_flops),
    ]:
        if op is not None:
            _CUSTOM_FLOP_FORMULAS[op] = fn

_ANALYTIC_FLOPS: Dict[str, float] = {
    "BatchNorm1d": 2.0, "BatchNorm2d": 2.0, "BatchNorm3d": 2.0,
    "LayerNorm": 5.0, "RMSNorm": 4.0, "ReLU": 1.0, "GELU": 8.0,
    "SiLU": 4.0, "Sigmoid": 4.0, "MaxPool2d": 1.0, "AvgPool2d": 1.0,
}


def _iter_tensors(obj: Any) -> Generator[Any, None, None]:
    if not HAS_TORCH:
        return
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            yield from _iter_tensors(item)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_tensors(v)


def _tensor_bytes(obj: Any) -> int:
    return sum(t.numel() * t.element_size() for t in _iter_tensors(obj))


def run_model(model: Any, inputs: Any) -> Any:
    """Forward-pass dispatch handling inputs as Tensor, tuple, or dict."""
    if isinstance(inputs, dict):
        return model(**inputs)
    if isinstance(inputs, (tuple, list)):
        return model(*inputs)
    return model(inputs)


class FlopsCalculator:
    """Calculates parameters and theoretical FLOPs for leaf modules."""

    @staticmethod
    def count_parameters(module: Any) -> int:
        """Count trainable and non-trainable parameters in a module."""
        if not HAS_TORCH or not isinstance(module, nn.Module):
            return 0
        return sum(p.numel() for p in module.parameters())

    @staticmethod
    def estimate_module_flops(module: Any, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> int:
        """Analytical FLOPs estimation for standard Conv2d, Linear, and normalization layers."""
        if not HAS_TORCH or not isinstance(module, nn.Module):
            return 0
        mod_type = type(module).__name__
        if mod_type == "Linear" and len(output_shape) >= 2 and hasattr(module, "in_features"):
            batch_elements = 1
            for dim in output_shape[:-1]:
                batch_elements *= dim
            return 2 * batch_elements * module.in_features * module.out_features
        if mod_type == "Conv2d" and len(output_shape) == 4 and len(input_shape) == 4:
            b, c_out, h_out, w_out = output_shape
            c_in = input_shape[1]
            k_h, k_w = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
            groups = getattr(module, "groups", 1)
            return 2 * b * c_out * h_out * w_out * (c_in // groups) * k_h * k_w
        if mod_type == "MultiheadAttention" and len(input_shape) >= 3:
            b, l, e = input_shape[:3]
            return 8 * b * l * e * e + 4 * b * l * l * e
        return 0


def select_targets(
    model: Any,
    exclude_types: Optional[Tuple[type, ...]] = None,
    atomic_types: Optional[Tuple[type, ...]] = None,
) -> List[Tuple[str, Any]]:
    """Select target layers: atomic modules (e.g. MultiheadAttention) or leaf modules, skipping children."""
    if not HAS_TORCH or not isinstance(model, nn.Module):
        return []
    exclude = exclude_types or (nn.Dropout, nn.Identity)
    atomic = atomic_types or ((nn.MultiheadAttention,) if HAS_TORCH else ())
    targets: List[Tuple[str, Any]] = []
    atomic_prefixes: List[str] = []
    for name, module in model.named_modules():
        if any(p == "" or name.startswith(p + ".") for p in atomic_prefixes):
            continue
        if isinstance(module, atomic):
            atomic_prefixes.append(name)
        elif len(list(module.children())) > 0:
            continue
        if isinstance(module, exclude):
            continue
        targets.append((name or "<root>", module))
    return targets


def analyze_layers(model: Any, inputs: Any, exclude_types: Optional[Tuple[type, ...]] = None) -> pd.DataFrame:
    """Execute single forward pass to capture FLOPs and memory traffic for each leaf layer."""
    if not HAS_TORCH or not isinstance(model, nn.Module):
        return pd.DataFrame()

    targets = select_targets(model, exclude_types=exclude_types)
    stats: Dict[str, Dict[str, Any]] = {}
    handles = []


    def _make_hook(name: str, mod: Any) -> Callable:
        p_bytes = sum(p.numel() * p.element_size() for p in mod.parameters())
        def _hook(m: Any, inp: Any, out: Any) -> None:
            s = stats.setdefault(name, {
                "layer_type": type(mod).__name__, "n_calls": 0,
                "bytes_in": 0, "bytes_out": 0, "bytes_params": p_bytes, "numel_out": 0,
                "input_shape": tuple(inp[0].shape) if inp and hasattr(inp[0], "shape") else (),
                "output_shape": tuple(out.shape) if hasattr(out, "shape") else (),
            })
            s["n_calls"] += 1
            s["bytes_in"] += _tensor_bytes(inp)
            s["bytes_out"] += _tensor_bytes(out)
            s["numel_out"] += sum(t.numel() for t in _iter_tensors(out))
        return _hook

    for name, module in targets:
        handles.append(module.register_forward_hook(_make_hook(name, module)))

    was_training = model.training
    model.eval()
    counts: Dict[str, Dict[str, Any]] = {}
    try:
        if HAS_FLOP_COUNTER and FlopCounterMode is not None:
            with torch.no_grad(), FlopCounterMode(display=False, custom_mapping=_CUSTOM_FLOP_FORMULAS) as counter:
                run_model(model, inputs)
            counts = counter.get_flop_counts()
        else:
            with torch.no_grad():
                run_model(model, inputs)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    root = type(model).__name__
    rows: List[Dict[str, Any]] = []

    for name, mod in targets:
        s = stats.get(name)
        if s is None:
            continue
        n = s["n_calls"]
        key = root if name == "<root>" else f"{root}.{name}"
        counted = float(sum(counts.get(key, {}).values()))
        if counted > 0:
            flops, source = counted / n, "counter"
        elif s["layer_type"] in _ANALYTIC_FLOPS:
            flops = _ANALYTIC_FLOPS[s["layer_type"]] * s["numel_out"] / n
            source = "analytic"
        else:
            est = float(FlopsCalculator.estimate_module_flops(mod, s.get("input_shape", ()), s.get("output_shape", ())))
            flops, source = (est, "analytic") if est > 0 else (0.0, "none")

        b_in, b_out = s["bytes_in"] / n, s["bytes_out"] / n
        b_total = b_in + b_out + s["bytes_params"]
        rows.append({
            "layer_name": name,
            "layer_type": s["layer_type"],
            "n_calls": n,
            "flops": flops,
            "flops_source": source,
            "bytes_in": b_in,
            "bytes_out": b_out,
            "bytes_params": s["bytes_params"],
            "bytes_total": b_total,
            "arithmetic_intensity": flops / b_total if b_total > 0 else 0.0,
        })

    df = pd.DataFrame(rows)
    df.attrs["global_flops"] = float(sum(counts.get("Global", {}).values()))
    return df


def model_total_flops(analysis: pd.DataFrame, source: str = "counter") -> float:
    """Sum total FLOPs across entire model forward pass."""
    if analysis.empty:
        return 0.0
    df = analysis if source == "all" else analysis[analysis["flops_source"] == source]
    return float((df["flops"] * df["n_calls"]).sum())
