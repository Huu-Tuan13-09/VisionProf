"""Layer-level profiling module for VisionProf."""

from src.layer.events import DeferredCudaDispatcher
from src.layer.flops import FlopsCalculator
from src.layer.hw_specs import calculate_roofline_efficiency, get_hardware_specs
from src.layer.memory import ActivationMemoryTracker
from src.layer.overhead import OverheadCalibrator
from src.layer.plugin import LayerProfilerPlugin, profile_layers
from src.layer.timer import CPUTimer

__all__ = [
    "DeferredCudaDispatcher",
    "FlopsCalculator",
    "ActivationMemoryTracker",
    "OverheadCalibrator",
    "get_hardware_specs",
    "calculate_roofline_efficiency",
    "LayerProfilerPlugin",
    "profile_layers",
    "CPUTimer",
]
