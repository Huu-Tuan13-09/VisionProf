"""VisionProf: Deep Learning Performance Profiling and Diagnostic Framework."""

__version__ = "2.0.0"

from src.core.constants import DeviceType, ModelFamily, PhaseType, RuleStatus
from src.core.interfaces import ModelAdapter, ProfilerPlugin
from src.core.paths import build_result_path
from src.layer.plugin import LayerProfilerPlugin, profile_layers
from src.phase.plugin import PhaseProfilerPlugin

__all__ = [
    "__version__",
    "DeviceType",
    "ModelFamily",
    "PhaseType",
    "RuleStatus",
    "ModelAdapter",
    "ProfilerPlugin",
    "build_result_path",
    "LayerProfilerPlugin",
    "PhaseProfilerPlugin",
    "profile_layers",
]
