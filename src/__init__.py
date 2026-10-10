"""VisionProf: Deep Learning Performance Profiling and Diagnostic Framework."""

__version__ = "2.0.0"

from src.core.constants import DeviceType, ModelFamily, PhaseType, RuleStatus
from src.core.interfaces import ModelAdapter, ProfilerPlugin
from src.core.paths import build_result_path
try:
    from src.layer.plugin import LayerProfilerPlugin, profile_layers
except ImportError:
    LayerProfilerPlugin = profile_layers = None  # type: ignore

try:
    from src.phase.plugin import PhaseProfilerPlugin
except ImportError:
    PhaseProfilerPlugin = None  # type: ignore

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
