"""
hw_specs.py — Thông số tối đa của phần cứng (TV1 · Mức Layer)
==============================================================
Dùng để tính "% hiệu suất" của từng layer theo mô hình Roofline đơn giản:

    attainable_gflops = min(peak_gflops, bandwidth_gbs × arithmetic_intensity)
    efficiency_pct    = achieved_gflops / attainable_gflops × 100

- Layer có arithmetic intensity (FLOPs/byte) thấp hơn ridge point → giới hạn bởi
  băng thông bộ nhớ (memory-bound), ngược lại → giới hạn bởi sức tính (compute-bound).

Thông số GPU lấy từ datasheet của NVIDIA. Thông số CPU laptop được ƯỚC LƯỢNG
(số core × xung nhịp × FLOPs/chu kỳ) vì mỗi thành viên dùng một laptop khác nhau —
có thể ghi đè bằng tham số khi gọi detect_hardware().
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import psutil
import torch


@dataclass
class HardwareSpec:
    """Thông số tối đa của một thiết bị."""
    hw_id:            str      # "colab_t4", "cpu_laptop", ...
    name:             str      # tên đầy đủ
    peak_gflops_fp32: float    # sức tính tối đa FP32 (GFLOP/s)
    peak_gflops_fp16: float    # sức tính tối đa FP16 (Tensor Core nếu có)
    bandwidth_gbs:    float    # băng thông bộ nhớ tối đa (GB/s)
    source:           str      # "datasheet" hoặc "estimate"

    def peak_gflops(self, precision: str = "fp32") -> float:
        return self.peak_gflops_fp16 if precision == "fp16" else self.peak_gflops_fp32

    def ridge_point(self, precision: str = "fp32") -> float:
        """Arithmetic intensity (FLOPs/byte) mà tại đó layer chuyển từ memory-bound sang compute-bound."""
        return self.peak_gflops(precision) / self.bandwidth_gbs

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# GPU: theo datasheet NVIDIA
KNOWN_GPUS: Dict[str, HardwareSpec] = {
    "T4": HardwareSpec(
        hw_id="colab_t4", name="NVIDIA Tesla T4",
        peak_gflops_fp32=8_100.0, peak_gflops_fp16=65_000.0,
        bandwidth_gbs=320.0, source="datasheet",
    ),
    # Chỉ dùng để thử code trên máy cá nhân, KHÔNG dùng cho số liệu báo cáo
    "4060 Laptop": HardwareSpec(
        hw_id="gpu_dev_rtx4060_laptop", name="NVIDIA GeForce RTX 4060 Laptop GPU",
        peak_gflops_fp32=11_600.0, peak_gflops_fp16=46_400.0,
        bandwidth_gbs=256.0, source="datasheet",
    ),
}


def estimate_cpu_spec(
    flops_per_cycle_per_core: int = 32,
    bandwidth_gbs: float = 51.2,
    max_freq_ghz: Optional[float] = None,
    n_cores: Optional[int] = None,
) -> HardwareSpec:
    """
    Ước lượng thông số CPU laptop.

    Args:
        flops_per_cycle_per_core: 32 với AVX2 + FMA (đa số laptop Intel/AMD hiện nay),
                                  64 nếu CPU có AVX-512.
        bandwidth_gbs           : 51.2 GB/s = DDR4-3200 kênh đôi;
                                  76.8 GB/s = DDR5-4800 kênh đôi.
        max_freq_ghz / n_cores  : ghi đè nếu psutil đọc sai.
    """
    cores = n_cores or psutil.cpu_count(logical=False) or 1
    freq = max_freq_ghz
    if freq is None:
        f = psutil.cpu_freq()
        freq = (f.max or f.current) / 1000.0 if f else 2.5
        freq = freq or 2.5
    peak = cores * freq * flops_per_cycle_per_core
    return HardwareSpec(
        hw_id="cpu_laptop",
        name=f"CPU {cores} cores @ {freq:.2f} GHz",
        peak_gflops_fp32=peak,
        peak_gflops_fp16=peak,          # CPU không có Tensor Core
        bandwidth_gbs=bandwidth_gbs,
        source="estimate",
    )


def detect_hardware(device: str = "cpu", **cpu_overrides: Any) -> HardwareSpec:
    """
    Trả về thông số của thiết bị đang chạy.
    GPU: tra bảng KNOWN_GPUS theo tên card. CPU: ước lượng (xem estimate_cpu_spec).
    """
    if device.startswith("cuda") and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(torch.device(device))
        for key, spec in KNOWN_GPUS.items():
            if key in gpu_name:
                return spec
        raise ValueError(
            f"Chưa có thông số cho GPU '{gpu_name}'. Thêm vào KNOWN_GPUS trong hw_specs.py."
        )
    return estimate_cpu_spec(**cpu_overrides)
