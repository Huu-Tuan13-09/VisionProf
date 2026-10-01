"""Core constants and enumerations for VisionProf."""

from enum import Enum
from typing import List


class DeviceType(str, Enum):
    """Standard hardware device targets."""
    COLAB_CPU = "colab_cpu"
    COLAB_T4 = "colab_t4"
    CPU_LAPTOP = "cpu_laptop"


class ModelFamily(str, Enum):
    """Supported model family categorizations."""
    CV = "cv"
    NLP = "nlp"
    TABULAR = "tabular"


class PhaseType(str, Enum):
    """Execution phases in training and inference pipelines."""
    PREPROCESS = "preprocess"
    DATALOADER = "dataloader"
    FORWARD = "forward"
    BACKWARD = "backward"
    OPTIMIZER = "optimizer"
    POSTPROCESS = "postprocess"


class RuleStatus(str, Enum):
    """Diagnostic rule validation outcomes."""
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"


class PrecisionType(str, Enum):
    """Numerical precisions for benchmarking."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


def get_supported_devices() -> List[str]:
    """Return a list of supported hardware device identifiers."""
    return [d.value for d in DeviceType]
