"""Zero-Config LayerProfilerPlugin implementing layer-level latency, memory, and Roofline tracking."""

import csv
import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.core.interfaces import ProfilerPlugin
from src.core.types import LayerRecord
from src.layer.events import DeferredCudaDispatcher
from src.layer.flops import FlopsCalculator, analyze_layers, model_total_flops
from src.layer.hw_specs import HardwareSpec, detect_hardware
from src.layer.memory import ActivationMemoryTracker
from src.layer.overhead import estimate_timer_bias
from src.layer.roofline import add_efficiency
from src.layer.timer import CPUTimer

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder safely converting NumPy scalar and array types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        return obj.tolist() if isinstance(obj, np.ndarray) else super().default(obj)


class LayerProfilerPlugin(ProfilerPlugin):
    """Zero-Config layer-level profiler hooking leaf modules without manual model code edits."""

    name: str = "layer"

    def __init__(self, subtract_timer_bias: bool = True, hw_spec: Optional[HardwareSpec] = None) -> None:
        self.device: str = "cpu"
        self._enabled: bool = False
        self._is_cuda: bool = False
        self.subtract_timer_bias: bool = subtract_timer_bias
        self.hw_spec: Optional[HardwareSpec] = hw_spec
        self.precision: str = "fp32"
        self.timer_bias_ms: float = 0.0
        self._hooks: List[Any] = []
        self._records: List[LayerRecord] = []
        self._current_iter: int = 0
        self._cuda_dispatcher: Optional[DeferredCudaDispatcher] = None
        self._cpu_timers: Dict[str, CPUTimer] = {}
        self._call_counts: Dict[str, int] = {}
        self._raw_rows: List[Dict[str, Any]] = []
        self._model_ref: Optional[Any] = None
        self._captured_inputs: Optional[Any] = None
        self._capture_handle: Optional[Any] = None
        self._iter_totals: Dict[int, float] = {}

    def attach(self, model: Any, device: str) -> None:
        """Attach pre/post forward hooks automatically to all leaf modules."""
        self.device = device
        self.detach()

        if not HAS_TORCH or not isinstance(model, nn.Module):
            logger.info("Bypassed: non-PyTorch model (%s) - layer hooks skipped.", type(model).__name__)
            self._enabled = False
            return

        self._enabled = True
        self._model_ref = model
        self._is_cuda = "cuda" in device.lower() or "t4" in device.lower()
        if self.hw_spec is None:
            self.hw_spec = detect_hardware(self.device)

        if self.subtract_timer_bias and not self._is_cuda:
            self.timer_bias_ms = estimate_timer_bias(self.device)

        self._cuda_dispatcher = DeferredCudaDispatcher(enabled=self._is_cuda)

        # Hook to capture input on iteration 0 for FLOPs calculation
        def _capture(*hook_args: Any, **hook_kwargs: Any) -> None:
            if self._captured_inputs is not None:
                return
            if len(hook_args) >= 3:
                args, kwargs = hook_args[1], hook_args[2]
            elif len(hook_args) == 2:
                args, kwargs = hook_args[1], hook_kwargs
            else:
                args, kwargs = (), hook_kwargs

            if args:
                self._captured_inputs = args
            elif kwargs:
                self._captured_inputs = kwargs

        try:
            self._capture_handle = model.register_forward_pre_hook(_capture, with_kwargs=True)
        except TypeError:
            self._capture_handle = model.register_forward_pre_hook(_capture)

        leaf_count = 0
        for name, module in model.named_modules():
            if len(list(module.children())) == 0 and not isinstance(module, (nn.Dropout, nn.Identity)):
                self._hooks.append(module.register_forward_pre_hook(self._make_pre_hook(name)))
                self._hooks.append(module.register_forward_hook(self._make_post_hook(name, module)))
                leaf_count += 1

        logger.info("Zero-Config attached %d leaf module hooks on %s.", leaf_count, device)

    def _make_pre_hook(self, name: str) -> Callable:
        def _pre_hook(module: Any, inputs: Any) -> None:
            if not self._enabled:
                return
            if self._is_cuda and self._cuda_dispatcher:
                self._cuda_dispatcher.record_start(name)
            else:
                timer = self._cpu_timers.setdefault(name, CPUTimer(bias_ms=self.timer_bias_ms))
                timer.start()
        return _pre_hook

    def _make_post_hook(self, name: str, module: Any) -> Callable:
        def _post_hook(module_ref: Any, inputs: Any, output: Any) -> None:
            if not self._enabled:
                return
            out_shape, act_mb = ActivationMemoryTracker.calculate_output_meta(inputs, output)
            in_shape = inputs[0].shape if inputs and hasattr(inputs[0], "shape") else ()
            params = FlopsCalculator.count_parameters(module_ref)
            flops = FlopsCalculator.estimate_module_flops(module_ref, in_shape, tuple(out_shape))
            call_idx = self._call_counts.get(name, 0)
            self._call_counts[name] = call_idx + 1

            if self._is_cuda and self._cuda_dispatcher:
                self._cuda_dispatcher.record_stop(name, call_idx)
                self._raw_rows.append({
                    "iteration": self._current_iter, "layer_name": name,
                    "layer_type": type(module_ref).__name__, "call_index": call_idx,
                    "activation_mb": act_mb, "flops": flops, "params": params,
                    "in_shape": list(in_shape), "out_shape": out_shape,
                })
            else:
                timer = self._cpu_timers.get(name)
                time_ms = timer.stop() if timer else 0.0
                raw_ms = timer.raw_ms if timer else time_ms
                self._raw_rows.append({
                    "iteration": self._current_iter, "layer_name": name,
                    "layer_type": type(module_ref).__name__, "call_index": call_idx,
                    "time_ms": round(time_ms, 5), "raw_time_ms": round(raw_ms, 5),
                    "activation_mb": act_mb, "flops": flops, "params": params,
                    "in_shape": list(in_shape), "out_shape": out_shape,
                })
        return _post_hook

    def start(self, iteration: int) -> None:
        """Mark start of a new measurement iteration."""
        self._current_iter = iteration
        self._call_counts.clear()

    def stop(self) -> None:
        """Complete iteration and aggregate measurements."""
        if not self._enabled:
            return
        if self._is_cuda and self._cuda_dispatcher:
            cuda_durations = self._cuda_dispatcher.synchronize_and_collect()
            for r in self._raw_rows:
                if r["iteration"] == self._current_iter and "time_ms" not in r:
                    dur = cuda_durations.get((r["layer_name"], r["call_index"]), cuda_durations.get(r["layer_name"], 0.0))
                    r["time_ms"] = dur
                    r["raw_time_ms"] = dur

    def save(self, out_dir: Path) -> None:
        """Write collected layer records to layers.csv, layers_summary.csv, and layer_meta.json."""
        out_dir.mkdir(parents=True, exist_ok=True)
        if not self._raw_rows:
            return

        df_raw = pd.DataFrame(self._raw_rows)
        analysis_df = pd.DataFrame()
        if self._model_ref is not None and self._captured_inputs is not None:
            try:
                analysis_df = analyze_layers(self._model_ref, self._captured_inputs)
            except Exception as e:
                logger.warning("analyze_layers failed: %s", e)

        # Merge FLOPs if available
        if not analysis_df.empty:
            df_raw = df_raw.drop(columns=["flops"], errors="ignore")
            df_raw = df_raw.merge(analysis_df[["layer_name", "flops", "flops_source"]], on="layer_name", how="left")
            df_raw["flops"] = df_raw["flops"].fillna(0.0)
            df_raw["flops_source"] = df_raw["flops_source"].fillna("none")
        else:
            df_raw["flops_source"] = "analytical"

        # Export layers.csv with standard 7 columns in front
        cols_std = ["iteration", "layer_name", "layer_type", "time_ms", "activation_mb", "flops", "params",
                    "call_index", "raw_time_ms", "flops_source", "in_shape", "out_shape"]
        df_raw[[c for c in cols_std if c in df_raw.columns]].to_csv(out_dir / "layers.csv", index=False)

        # Export layers_summary.csv (Roofline)
        spec = self.hw_spec or detect_hardware(self.device)
        summary_df = add_efficiency(df_raw, analysis_df, spec, precision=self.precision, time_col="time_ms")
        if not summary_df.empty:
            summary_df.to_csv(out_dir / "layers_summary.csv", index=False)

        # Export layer_meta.json
        meta = {
            "plugin": self.name, "device": self.device, "precision": self.precision,
            "n_hooked_layers": int(df_raw["layer_name"].nunique()),
            "n_iterations": int(df_raw["iteration"].nunique()),
            "timer_bias_ms": self.timer_bias_ms,
            "total_flops_counter": model_total_flops(analysis_df) if not analysis_df.empty else 0.0,
            "hardware": spec.to_dict(),
        }
        with open(out_dir / "layer_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
        logger.info("Saved %d layer records and metadata to %s", len(df_raw), out_dir)

    def detach(self) -> None:
        """Remove all PyTorch hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        if self._capture_handle is not None:
            self._capture_handle.remove()
            self._capture_handle = None
        self._cpu_timers.clear()
        self._call_counts.clear()


@contextmanager
def profile_layers(model: Any, device: str = "colab_cpu") -> Generator[LayerProfilerPlugin, None, None]:
    """Context manager for profiling layer execution."""
    plugin = LayerProfilerPlugin()
    plugin.attach(model, device)
    try:
        yield plugin
    finally:
        plugin.detach()
