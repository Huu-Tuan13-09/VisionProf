"""Zero-Config LayerProfilerPlugin implementing layer-level latency and memory tracking."""

import csv
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from src.core.interfaces import ProfilerPlugin
from src.core.types import LayerRecord
from src.layer.events import DeferredCudaDispatcher
from src.layer.flops import FlopsCalculator
from src.layer.memory import ActivationMemoryTracker
from src.layer.timer import CPUTimer

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class LayerProfilerPlugin(ProfilerPlugin):
    """Zero-Config layer-level profiler hooking only leaf modules without manual code edits."""

    name: str = "layer"

    def __init__(self) -> None:
        self.device: str = "cpu"
        self._enabled: bool = False
        self._is_cuda: bool = False
        self._hooks: List[Any] = []
        self._records: List[LayerRecord] = []
        self._current_iter: int = 0
        self._cuda_dispatcher: Optional[DeferredCudaDispatcher] = None
        self._cpu_timers: Dict[str, CPUTimer] = {}
        self._temp_inputs: Dict[str, Any] = {}
        self._temp_outputs: Dict[str, Any] = {}

    def attach(self, model: Any, device: str) -> None:
        """Automatically traverse and register pre/post hooks to all leaf modules."""
        self.device = device
        self.detach()

        if not HAS_TORCH or not isinstance(model, nn.Module):
            logger.info("Bypassed: non-PyTorch model (%s) - layer hooks skipped.", type(model).__name__)
            self._enabled = False
            return

        self._enabled = True
        self._is_cuda = "cuda" in device.lower() or "t4" in device.lower()
        self._cuda_dispatcher = DeferredCudaDispatcher(enabled=self._is_cuda)

        leaf_count = 0
        for name, module in model.named_modules():
            # Leaf modules have no child modules
            if len(list(module.children())) == 0:
                h_pre = module.register_forward_pre_hook(self._make_pre_hook(name))
                h_post = module.register_forward_hook(self._make_post_hook(name, module))
                self._hooks.extend([h_pre, h_post])
                leaf_count += 1

        logger.info("Zero-Config attached %d leaf module hooks on %s.", leaf_count, device)

    def _make_pre_hook(self, name: str) -> Any:
        def pre_hook(module: Any, inputs: Any) -> None:
            if not self._enabled:
                return
            if self._is_cuda and self._cuda_dispatcher:
                self._cuda_dispatcher.record_start(name)
            else:
                timer = self._cpu_timers.setdefault(name, CPUTimer())
                timer.start()
        return pre_hook

    def _make_post_hook(self, name: str, module: Any) -> Any:
        def post_hook(module_ref: Any, inputs: Any, output: Any) -> None:
            if not self._enabled:
                return
            act_mb = ActivationMemoryTracker.calculate_output_mb(output)
            params = FlopsCalculator.count_parameters(module_ref)
            in_shape = inputs[0].shape if inputs and hasattr(inputs[0], "shape") else ()
            out_shape = output.shape if hasattr(output, "shape") else ()
            flops = FlopsCalculator.estimate_module_flops(module_ref, in_shape, out_shape)

            if self._is_cuda and self._cuda_dispatcher:
                self._cuda_dispatcher.record_stop(name)
                # Aggregate activation memory and FLOPs for repeated module invocations
                if name in self._temp_outputs:
                    prev_act, prev_flops, p, mod_type = self._temp_outputs[name]
                    self._temp_outputs[name] = (prev_act + act_mb, prev_flops + flops, p, mod_type)
                else:
                    self._temp_outputs[name] = (act_mb, flops, params, type(module_ref).__name__)
            else:
                timer = self._cpu_timers.get(name)
                elapsed_ms = timer.stop() if timer else 0.0
                record = LayerRecord(
                    iteration=self._current_iter,
                    layer_name=name,
                    layer_type=type(module_ref).__name__,
                    time_ms=round(elapsed_ms, 4),
                    activation_mb=act_mb,
                    flops=flops,
                    params=params,
                    device=self.device,
                )
                self._records.append(record)
        return post_hook

    def start(self, iteration: int) -> None:
        """Mark start of a new measurement iteration."""
        self._current_iter = iteration
        self._temp_inputs.clear()
        self._temp_outputs.clear()

    def stop(self) -> None:
        """Complete iteration and aggregate measurements (sync CUDA events if applicable)."""
        if not self._enabled:
            return
        if self._is_cuda and self._cuda_dispatcher:
            cuda_durations = self._cuda_dispatcher.synchronize_and_collect()
            for name, elapsed_ms in cuda_durations.items():
                if name in self._temp_outputs:
                    act_mb, flops, params, mod_type = self._temp_outputs[name]
                    record = LayerRecord(
                        iteration=self._current_iter,
                        layer_name=name,
                        layer_type=mod_type,
                        time_ms=elapsed_ms,
                        activation_mb=act_mb,
                        flops=flops,
                        params=params,
                        device=self.device,
                    )
                    self._records.append(record)

    def save(self, out_dir: Path) -> None:
        """Write collected layer records to layers.csv."""
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_file = out_dir / "layers.csv"
        headers = ["iteration", "layer_name", "layer_type", "time_ms", "activation_mb", "flops", "params", "device"]

        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for r in self._records:
                writer.writerow([
                    r.iteration, r.layer_name, r.layer_type, r.time_ms,
                    r.activation_mb, r.flops, r.params, r.device
                ])
        logger.info("Saved %d layer records to %s", len(self._records), csv_file)

    def detach(self) -> None:
        """Remove all active PyTorch forward hooks to prevent memory leaks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._cpu_timers.clear()
        self._temp_outputs.clear()


@contextmanager
def profile_layers(model: Any, device: str = "colab_cpu") -> Generator[LayerProfilerPlugin, None, None]:
    """Zero-config Context Manager to profile any model's layers seamlessly."""
    plugin = LayerProfilerPlugin()
    plugin.attach(model, device)
    try:
        yield plugin
    finally:
        plugin.detach()
