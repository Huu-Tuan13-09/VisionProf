"""Data contracts, records, and schema definitions for VisionProf."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional
from src.core.constants import RuleStatus


@dataclass
class LayerRecord:
    """Record representing layer-level profiling metrics."""
    iteration: int
    layer_name: str
    layer_type: str
    time_ms: float
    activation_mb: float = 0.0
    flops: int = 0
    params: int = 0
    device: str = "cpu"

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return asdict(self)


@dataclass
class PhaseRecord:
    """Record representing stage/phase level profiling metrics."""
    iteration: int
    phase: str
    time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return asdict(self)


@dataclass
class BenchmarkMetric:
    """Record for an individual benchmark iteration."""
    iteration: int
    latency_ms: float
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return asdict(self)


@dataclass
class ResourceRecord:
    """Record for system resource sampling at 100ms interval."""
    timestamp: float
    cpu_pct: float
    rss_mb: float
    gpu_util_pct: float = 0.0
    vram_mb: float = 0.0
    power_w: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return asdict(self)


@dataclass
class BenchmarkSummary:
    """Aggregated statistical summary of benchmark metrics."""
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    tail_ratio: float
    throughput_fps: float
    peak_vram_mb: float = 0.0
    rss_mb: float = 0.0
    reserved_vram_mb: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary."""
        return asdict(self)


@dataclass
class RuleResult:
    """Diagnostic heuristic evaluation result."""
    rule_id: str
    status: RuleStatus
    title: str
    message: str
    recommendation: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        data = asdict(self)
        data["status"] = self.status.value
        return data
