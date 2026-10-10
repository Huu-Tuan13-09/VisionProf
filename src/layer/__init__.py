"""Layer-level profiling module for VisionProf."""

from src.layer.analysis import compare_top_k, compare_type_share, load_layers, per_layer_time, to_markdown, top_k_layers, type_share
from src.layer.events import DeferredCudaDispatcher
from src.layer.flops import FlopsCalculator, analyze_layers, model_total_flops, run_model
from src.layer.hw_specs import HardwareSpec, calculate_roofline_efficiency, detect_hardware, get_hardware_specs
from src.layer.memory import ActivationMemoryTracker
from src.layer.overhead import OverheadCalibrator, calibrate_overhead_details, estimate_timer_bias
from src.layer.plugin import LayerProfilerPlugin, profile_layers
from src.layer.roofline import add_efficiency
from src.layer.timer import CPUTimer

__all__ = [
    "DeferredCudaDispatcher",
    "FlopsCalculator",
    "ActivationMemoryTracker",
    "OverheadCalibrator",
    "HardwareSpec",
    "detect_hardware",
    "get_hardware_specs",
    "calculate_roofline_efficiency",
    "LayerProfilerPlugin",
    "profile_layers",
    "CPUTimer",
    "analyze_layers",
    "model_total_flops",
    "run_model",
    "add_efficiency",
    "estimate_timer_bias",
    "calibrate_overhead_details",
    "load_layers",
    "per_layer_time",
    "top_k_layers",
    "type_share",
    "compare_top_k",
    "compare_type_share",
    "to_markdown",
]
