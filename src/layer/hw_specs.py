"""Theoretical peak hardware specifications for Roofline and efficiency calculations."""

from typing import Dict


HARDWARE_PEAKS: Dict[str, Dict[str, float]] = {
    "colab_t4": {
        "peak_fp32_tflops": 8.1,
        "peak_fp16_tflops": 65.0,
        "memory_bandwidth_gb_s": 320.0,
        "vram_gb": 16.0,
    },
    "colab_cpu": {
        "peak_fp32_tflops": 0.14,
        "peak_fp16_tflops": 0.14,
        "memory_bandwidth_gb_s": 35.0,
        "vram_gb": 12.7,
    },
    "cpu_laptop": {
        "peak_fp32_tflops": 0.25,
        "peak_fp16_tflops": 0.25,
        "memory_bandwidth_gb_s": 45.0,
        "vram_gb": 16.0,
    },
}


def get_hardware_specs(device: str) -> Dict[str, float]:
    """Retrieve theoretical hardware peak specification dictionary for a given device."""
    dev_key = device.strip().lower().replace("-", "_")
    return HARDWARE_PEAKS.get(dev_key, HARDWARE_PEAKS["colab_cpu"])


def calculate_roofline_efficiency(
    gflops: float,
    time_ms: float,
    device: str,
    precision: str = "fp32",
) -> float:
    """Calculate percentage efficiency relative to theoretical hardware compute ceiling."""
    if time_ms <= 0.0 or gflops <= 0.0:
        return 0.0
    specs = get_hardware_specs(device)
    peak_tflops = specs.get(f"peak_{precision.lower()}_tflops", specs["peak_fp32_tflops"])
    # Actual TFLOPs achieved = (GFLOPs / 1000) / (time_ms / 1000) = GFLOPs / time_ms
    achieved_tflops = gflops / time_ms
    pct = (achieved_tflops / peak_tflops) * 100.0
    return round(min(100.0, pct), 2)
