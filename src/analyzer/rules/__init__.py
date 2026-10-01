"""Registry and exports for all 19 heuristic diagnostic rules v2.0."""

from typing import List

from src.analyzer.rules.architecture import (
    check_rd01_param_concentration,
    check_rd02_sub_microsecond_layers,
    check_rd03_memory_compute_disparity,
    check_rd04_layer_bottleneck,
    check_rd05_latency_flops_outliers,
)
from src.analyzer.rules.compute import (
    check_rb01_gpu_utilization,
    check_rb02_tail_latency,
    check_rb03_backward_bottleneck,
    check_rb04_compute_efficiency,
    check_rb05_optimizer_stall,
)
from src.analyzer.rules.dataloader import (
    check_rc01_io_ratio,
    check_rc02_worker_variance,
    check_rc03_gpu_starvation,
    check_rc04_batch_underutilization,
)
from src.analyzer.rules.memory import (
    check_ra01_oom_risk,
    check_ra02_fragmentation,
    check_ra03_activation_hotspot,
    check_ra04_memory_leak,
    check_ra05_host_device_transfer,
)

__all__ = [
    "check_ra01_oom_risk",
    "check_ra02_fragmentation",
    "check_ra03_activation_hotspot",
    "check_ra04_memory_leak",
    "check_ra05_host_device_transfer",
    "check_rb01_gpu_utilization",
    "check_rb02_tail_latency",
    "check_rb03_backward_bottleneck",
    "check_rb04_compute_efficiency",
    "check_rb05_optimizer_stall",
    "check_rc01_io_ratio",
    "check_rc02_worker_variance",
    "check_rc03_gpu_starvation",
    "check_rc04_batch_underutilization",
    "check_rd01_param_concentration",
    "check_rd02_sub_microsecond_layers",
    "check_rd03_memory_compute_disparity",
    "check_rd04_layer_bottleneck",
    "check_rd05_latency_flops_outliers",
]
