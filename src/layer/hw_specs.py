"""Hardware specification definitions and detection for Roofline calculations."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class HardwareSpec:
    """Theoretical peak computing capacity and bandwidth limits for a hardware target."""

    hw_id: str
    name: str
    peak_gflops_fp32: float
    peak_gflops_fp16: float
    bandwidth_gbs: float
    source: str = "datasheet"

    def peak_gflops(self, precision: str = "fp32") -> float:
        """Return peak compute capability in GFLOP/s based on target precision."""
        return self.peak_gflops_fp16 if precision.lower() == "fp16" else self.peak_gflops_fp32

    def ridge_point(self, precision: str = "fp32") -> float:
        """Arithmetic intensity (FLOP/byte) threshold where workload transitions to compute-bound."""
        return self.peak_gflops(precision) / self.bandwidth_gbs if self.bandwidth_gbs > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert specification into a JSON-serializable dictionary."""
        return asdict(self)


KNOWN_GPUS: Dict[str, HardwareSpec] = {
    "T4": HardwareSpec(
        hw_id="colab_t4",
        name="NVIDIA Tesla T4",
        peak_gflops_fp32=8_100.0,
        peak_gflops_fp16=65_000.0,
        bandwidth_gbs=320.0,
        source="datasheet",
    ),
}


def estimate_cpu_spec(
    flops_per_cycle_per_core: int = 32,
    bandwidth_gbs: float = 51.2,
    max_freq_ghz: Optional[float] = None,
    n_cores: Optional[int] = None,
) -> HardwareSpec:
    """Estimate peak theoretical capacity for local host CPU."""
    cores = n_cores or (psutil.cpu_count(logical=False) if HAS_PSUTIL else 4) or 4
    freq = max_freq_ghz
    if freq is None and HAS_PSUTIL:
        f = psutil.cpu_freq()
        freq = (f.max or f.current) / 1000.0 if f else 2.5
    freq = freq or 2.5
    peak = cores * freq * flops_per_cycle_per_core
    return HardwareSpec(
        hw_id="cpu_laptop",
        name=f"CPU {cores} cores @ {freq:.2f} GHz",
        peak_gflops_fp32=peak,
        peak_gflops_fp16=peak,
        bandwidth_gbs=bandwidth_gbs,
        source="estimate",
    )


def detect_hardware(device: str = "cpu", **cpu_overrides: Any) -> HardwareSpec:
    """Detect hardware specs from target device name or active PyTorch runtime."""
    dev_lower = device.strip().lower()

    if "t4" in dev_lower:
        return KNOWN_GPUS["T4"]

    if dev_lower == "colab_cpu":
        return HardwareSpec(
            hw_id="colab_cpu",
            name="Intel Xeon CPU @ 2.20GHz (Colab)",
            peak_gflops_fp32=140.0,
            peak_gflops_fp16=140.0,
            bandwidth_gbs=35.0,
            source="datasheet",
        )

    if dev_lower.startswith("cuda") and HAS_TORCH and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        for key, spec in KNOWN_GPUS.items():
            if key in gpu_name:
                return spec
        return KNOWN_GPUS["T4"]

    return estimate_cpu_spec(**cpu_overrides)


def get_hardware_specs(device: str) -> Dict[str, float]:
    """Backward-compatible helper returning dictionary of peaks."""
    spec = detect_hardware(device)
    return {
        "peak_fp32_tflops": spec.peak_gflops_fp32 / 1000.0,
        "peak_fp16_tflops": spec.peak_gflops_fp16 / 1000.0,
        "memory_bandwidth_gb_s": spec.bandwidth_gbs,
        "vram_gb": 16.0 if "t4" in spec.hw_id else 12.0,
    }
