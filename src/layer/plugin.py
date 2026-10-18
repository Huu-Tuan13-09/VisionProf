"""Zero-Config LayerProfilerPlugin implementing layer-level latency, memory, and Roofline tracking."""

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil

from src.core.interfaces import ProfilerPlugin
from src.layer.events import DeferredCudaDispatcher
from src.layer.flops import analyze_layers, model_total_flops, select_targets
from src.layer.hw_specs import HardwareSpec, detect_hardware
from src.layer.memory import ActivationMemoryTracker
from src.layer.overhead import estimate_timer_bias
from src.layer.roofline import add_efficiency

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

_LAYER_CSV_COLS = [
    "iteration", "layer_name", "layer_type", "time_ms", "activation_mb", "flops", "params",
    "call_index", "exec_order", "raw_time_ms", "flops_source", "in_shape", "out_shape",
    "process_rss_mb", "timing_backend", "device",
]


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder safely converting NumPy scalar and array types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.bool_, bool)): return bool(obj)
        if isinstance(obj, (np.integer, int)): return int(obj)
        if isinstance(obj, (np.floating, float)): return float(obj)
        return obj.tolist() if isinstance(obj, np.ndarray) else super().default(obj)


class LayerProfilerPlugin(ProfilerPlugin):
    """
    Zero-Config layer-level profiler hooking leaf modules without manual model code edits.

    Hooks only record between start() and stop(); any other forward pass (warm-up, FLOPs
    counting, torch.profiler reference runs) is ignored. Everything costly (shapes, memory)
    happens OUTSIDE the timed interval: the pre-hook starts the clock on its last line and the
    post-hook stops it on its first line.
    """

    name: str = "layer"

    def __init__(
        self,
        subtract_timer_bias: bool = True,
        hw_spec: Optional[HardwareSpec] = None,
        exclude_types: Optional[Tuple[type, ...]] = None,
        gpu_queue_prefill: bool = True,
        precision: Optional[str] = None,
    ) -> None:
        self.device, self._enabled, self._is_cuda, self._active = "cpu", False, False, False
        self.subtract_timer_bias, self.hw_spec = subtract_timer_bias, hw_spec
        self.exclude_types, self.gpu_queue_prefill = exclude_types, gpu_queue_prefill
        self.precision, self.timer_bias_ms = precision or "fp32", 0.0
        self._fixed_precision = precision is not None
        self._hooks: List[Any] = []
        self._current_iter, self._exec_counter, self._iter_t0_ns = 0, 0, 0
        self._rss_mb = 0.0
        self._cuda_dispatcher: Optional[DeferredCudaDispatcher] = None
        self._cpu_stacks: Dict[str, List[int]] = {}
        self._call_counts: Dict[str, int] = {}
        self._params: Dict[str, int] = {}
        self._pending: List[Dict[str, Any]] = []
        self._raw_rows: List[Dict[str, Any]] = []
        self._model_ref = self._captured_inputs = self._capture_handle = None
        self._iter_totals: Dict[int, float] = {}

    @property
    def n_hooked(self) -> int:
        """Number of modules carrying a pre/post hook pair."""
        return len(self._hooks) // 2

    def attach(self, model: Any, device: str) -> None:
        """Attach pre/post forward hooks automatically to all leaf modules."""
        self.detach()
        self.device = str(device)
        if not HAS_TORCH or not isinstance(model, nn.Module):
            logger.info("Bypassed: non-PyTorch model (%s) - layer hooks skipped.", type(model).__name__)
            self._enabled = False
            return

        self._enabled, self._model_ref, self._captured_inputs = True, model, None
        self._is_cuda = "cuda" in self.device.lower() or "t4" in self.device.lower()
        if not self._fixed_precision:
            p = next(model.parameters(), None)
            self.precision = "fp16" if p is not None and p.dtype in (torch.float16, torch.bfloat16) else "fp32"
        if self.hw_spec is None:
            self.hw_spec = detect_hardware(self.device)
        if self.subtract_timer_bias and not self._is_cuda:
            self.timer_bias_ms = estimate_timer_bias(self.device)
        self._cuda_dispatcher = DeferredCudaDispatcher(enabled=self._is_cuda, queue_prefill=self.gpu_queue_prefill)

        # Capture the first measured input to count FLOPs at save() time, outside the measurement
        def _capture(module: Any, args: Any, kwargs: Any) -> None:
            # run_model() replays a tuple as *args and a dict as **kwargs
            if self._active and self._captured_inputs is None and not (args and kwargs):
                self._captured_inputs = kwargs or args or None
        self._capture_handle = model.register_forward_pre_hook(_capture, with_kwargs=True)

        for name, module in select_targets(model, exclude_types=self.exclude_types):
            self._params[name] = sum(p.numel() for p in module.parameters())
            self._hooks.append(module.register_forward_pre_hook(self._make_pre_hook(name)))
            self._hooks.append(module.register_forward_hook(self._make_post_hook(name)))
        logger.info("Zero-Config attached %d leaf module hooks on %s.", self.n_hooked, self.device)

    def _make_pre_hook(self, name: str) -> Callable:
        def _pre_hook(module: Any, inputs: Any) -> None:
            if not self._active:
                return
            in_shape = list(inputs[0].shape) if inputs and hasattr(inputs[0], "shape") else []
            self._pending.append({"layer_name": name, "in_shape": in_shape, "exec_order": self._exec_counter})
            self._exec_counter += 1
            if self._is_cuda and self._cuda_dispatcher and self._cuda_dispatcher.enabled:
                self._cuda_dispatcher.record_start(name)
            else:
                self._cpu_stacks.setdefault(name, []).append(time.perf_counter_ns())
        return _pre_hook

    def _make_post_hook(self, name: str) -> Callable:
        def _post_hook(module: Any, inputs: Any, output: Any) -> None:
            if not self._active:
                return
            call_idx = self._call_counts.get(name, 0)
            if self._is_cuda and self._cuda_dispatcher and self._cuda_dispatcher.enabled:
                self._cuda_dispatcher.record_stop(name, call_idx)
                raw_ms = None
            else:
                t_end = time.perf_counter_ns()
                stack = self._cpu_stacks.get(name)
                raw_ms = (t_end - stack.pop()) / 1e6 if stack else 0.0
            self._call_counts[name] = call_idx + 1
            out_shape, act_mb = ActivationMemoryTracker.calculate_output_meta(inputs, output)
            entry = next(e for e in reversed(self._pending) if e["layer_name"] == name and "call_index" not in e)
            entry.update({
                "iteration": self._current_iter, "layer_type": type(module).__name__,
                "call_index": call_idx, "raw_time_ms": raw_ms, "activation_mb": act_mb,
                "params": self._params.get(name, 0), "out_shape": out_shape,
            })
        return _post_hook

    def start(self, iteration: int) -> None:
        """Mark start of a new measurement iteration."""
        if not self._enabled:
            return
        self._current_iter, self._exec_counter = iteration, 0
        self._call_counts.clear()
        self._cpu_stacks.clear()
        self._pending.clear()
        # RAM of the whole process, once per iteration and kept apart from per-layer memory
        self._rss_mb = psutil.Process().memory_info().rss / (1024.0 * 1024.0)
        if self._is_cuda and self._cuda_dispatcher:
            self._cuda_dispatcher.begin_iteration()
        self._active = True
        self._iter_t0_ns = time.perf_counter_ns()

    def stop(self) -> None:
        """Complete the iteration: synchronize the GPU once and resolve every layer duration."""
        if not self._enabled or not self._active:
            return
        self._active = False
        if self._is_cuda and self._cuda_dispatcher and self._cuda_dispatcher.enabled:
            durations, total_ms = self._cuda_dispatcher.end_iteration(self._current_iter)
            backend = "cuda_event"
        else:
            total_ms = (time.perf_counter_ns() - self._iter_t0_ns) / 1e6
            durations, backend = {}, "cpu_perf_counter"
        bias = 0.0 if backend == "cuda_event" else self.timer_bias_ms
        for r in self._pending:
            if "call_index" not in r:
                continue
            raw = r["raw_time_ms"] if r["raw_time_ms"] is not None else durations.get((r["layer_name"], r["call_index"]), 0.0)
            r["raw_time_ms"], r["time_ms"] = round(raw, 5), round(max(0.0, raw - bias), 5)
            r["timing_backend"], r["device"] = backend, self.device
            r["process_rss_mb"] = round(self._rss_mb, 2)
            self._raw_rows.append(r)
        self._pending = []
        self._iter_totals[self._current_iter] = round(total_ms, 5)

    def save(self, out_dir: Path) -> None:
        """Write collected layer records to layers.csv, layers_summary.csv, and layer_meta.json."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not self._raw_rows:
            return

        df_raw = pd.DataFrame(self._raw_rows)
        analysis_df = pd.DataFrame()
        if self._model_ref is not None and self._captured_inputs is not None:
            try:
                analysis_df = analyze_layers(self._model_ref, self._captured_inputs, exclude_types=self.exclude_types)
            except Exception as e:
                logger.warning("analyze_layers failed: %s", e)

        if not analysis_df.empty:
            df_raw = df_raw.merge(analysis_df[["layer_name", "flops", "flops_source"]], on="layer_name", how="left")
            df_raw["flops"] = df_raw["flops"].fillna(0.0)
            df_raw["flops_source"] = df_raw["flops_source"].fillna("none")
        else:
            df_raw["flops"], df_raw["flops_source"] = 0.0, "none"

        df_raw[[c for c in _LAYER_CSV_COLS if c in df_raw.columns]].to_csv(out_dir / "layers.csv", index=False)

        spec = self.hw_spec or detect_hardware(self.device)
        summary_df = add_efficiency(df_raw, analysis_df, spec, precision=self.precision, time_col="time_ms")
        if not summary_df.empty:
            summary_df.to_csv(out_dir / "layers_summary.csv", index=False)

        prefill = self._cuda_dispatcher.get_prefill_ok() if self._cuda_dispatcher else {}
        has_analysis = not analysis_df.empty
        meta = {
            "plugin": self.name, "device": self.device, "precision": self.precision,
            "n_hooked_layers": int(df_raw["layer_name"].nunique()),
            "n_iterations": int(df_raw["iteration"].nunique()),
            "timer_bias_ms": self.timer_bias_ms,
            "total_flops_model": float(analysis_df.attrs.get("global_flops", 0.0)) if has_analysis else 0.0,
            "total_flops_counter": model_total_flops(analysis_df) if has_analysis else 0.0,
            "total_flops_all": model_total_flops(analysis_df, source="all") if has_analysis else 0.0,
            "iteration_total_ms": self._iter_totals,
            "gpu_prefill_ok_ratio": sum(prefill.values()) / len(prefill) if prefill else None,
            "hardware": spec.to_dict(),
        }
        with open(out_dir / "layer_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
        logger.info("Saved %d layer records and metadata to %s", len(df_raw), out_dir)

    @property
    def profiler(self) -> "LayerProfilerPlugin":
        return self

    def to_dataframe(self) -> pd.DataFrame:
        """Return captured raw rows (one per layer call per iteration) as a DataFrame."""
        return pd.DataFrame(self._raw_rows) if self._raw_rows else pd.DataFrame()

    def get_iteration_totals(self) -> Dict[int, float]:
        """Whole-forward time per iteration (GPU: back-to-back GPU time between two events)."""
        return self._iter_totals

    def detach(self) -> None:
        """Remove all PyTorch hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        if self._capture_handle is not None:
            self._capture_handle.remove()
            self._capture_handle = None
        self._active = False
        self._cpu_stacks.clear()
        self._call_counts.clear()
        self._params.clear()


@contextmanager
def profile_layers(model: Any, device: str = "colab_cpu") -> Generator[LayerProfilerPlugin, None, None]:
    """Context manager for profiling layer execution."""
    plugin = LayerProfilerPlugin()
    plugin.attach(model, device)
    try:
        yield plugin
    finally:
        plugin.detach()
