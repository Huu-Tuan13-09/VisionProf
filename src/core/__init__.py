"""Core module for VisionProf containing interfaces, types, constants and utilities."""

from src.core.constants import DeviceType, ModelFamily, PhaseType, RuleStatus
from src.core.interfaces import ModelAdapter, ProfilerPlugin
from src.core.paths import build_result_path, sanitize_name
from src.core.types import BenchmarkMetric, LayerRecord, PhaseRecord

__all__ = [
    "DeviceType",
    "ModelFamily",
    "PhaseType",
    "RuleStatus",
    "ModelAdapter",
    "ProfilerPlugin",
    "build_result_path",
    "sanitize_name",
    "LayerRecord",
    "PhaseRecord",
    "BenchmarkMetric",
]
