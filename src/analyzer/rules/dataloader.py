"""DataLoader and I/O diagnostic rules (RC-01 to RC-04) for VisionProf v2.0."""

import statistics
from typing import List
from src.core.constants import RuleStatus
from src.core.types import RuleResult


def check_rc01_io_ratio(io_time_ms: float, total_step_ms: float) -> RuleResult:
    """RC-01: Flag excessive data fetching stall (>20% of iteration step)."""
    if total_step_ms <= 0.0 or io_time_ms <= 0.0:
        return RuleResult("RC-01", RuleStatus.SKIPPED, "I/O Stall Ratio", "No DataLoader timing collected.")
    ratio = io_time_ms / total_step_ms
    rec = "Gợi ý: Cấu hình DataLoader(..., num_workers=2, pin_memory=True, persistent_workers=True) để nạp trước dữ liệu song song."
    if ratio > 0.35:
        return RuleResult("RC-01", RuleStatus.FAIL, "Severe I/O Bottleneck", f"I/O takes {ratio:.1%} of step time.", recommendation=rec)
    if ratio > 0.20:
        return RuleResult("RC-01", RuleStatus.WARN, "Moderate I/O Bottleneck", f"I/O takes {ratio:.1%} of step time.", recommendation=rec)
    return RuleResult("RC-01", RuleStatus.PASS, "Efficient Data Pipeline", f"I/O takes {ratio:.1%} of step time.")


def check_rc02_worker_variance(io_samples: List[float]) -> RuleResult:
    """RC-02: Measure coefficient of variation in DataLoader retrieval latency (>0.40 indicates imbalance)."""
    if len(io_samples) < 5:
        return RuleResult("RC-02", RuleStatus.SKIPPED, "Worker Variance", "Too few I/O samples.")
    mean_val = statistics.mean(io_samples)
    if mean_val <= 0.0:
        return RuleResult("RC-02", RuleStatus.PASS, "Zero I/O Variance", "Data retrieval instantaneous.")
    cv = statistics.stdev(io_samples) / mean_val
    if cv > 0.50:
        rec = "Gợi ý: Kiểm tra tốc độ đọc I/O đĩa cứng hoặc tăng prefetch_factor để làm mịn độ trễ nạp dữ liệu giữa các worker."
        return RuleResult("RC-02", RuleStatus.WARN, "High Worker Latency Variance", f"I/O Coefficient of Variation: {cv:.2f}.", recommendation=rec)
    return RuleResult("RC-02", RuleStatus.PASS, "Consistent I/O Throughput", f"I/O Coefficient of Variation: {cv:.2f}.")


def check_rc03_gpu_starvation(io_time_ms: float, compute_time_ms: float) -> RuleResult:
    """RC-03: Detect GPU Starvation (I/O wait exceeds active GPU forward+backward compute)."""
    if compute_time_ms <= 0.0 or io_time_ms <= 0.0:
        return RuleResult("RC-03", RuleStatus.SKIPPED, "GPU Starvation", "Phase metrics unavailable.")
    if io_time_ms > compute_time_ms:
        diff_ms = io_time_ms - compute_time_ms
        rec = "Gợi ý: Thiết lập num_workers=2-4 và pin_memory=True, chuyển tiền xử lý sang GPU bằng torchvision/DALI để triệt tiêu starvation."
        return RuleResult("RC-03", RuleStatus.FAIL, "GPU Starvation Detected", f"GPU idle waiting for data ({diff_ms:.1f} ms longer than compute).", recommendation=rec)
    return RuleResult("RC-03", RuleStatus.PASS, "No GPU Starvation", "GPU active compute dominates I/O wait.")


def check_rc04_batch_underutilization(batch_size: int, throughput_fps: float) -> RuleResult:
    """RC-04: Evaluate if small batch size causes launch overhead dominance."""
    if batch_size == 1 and throughput_fps < 20.0:
        rec = "Gợi ý: Tăng batch_size (lên 8 hoặc 32) để bão hòa GPU Cores và triệt tiêu chi phí gửi kernel tuần tự."
        return RuleResult("RC-04", RuleStatus.WARN, "Small Batch Overhead", "Batch size 1 may suffer from launch overhead; consider batch scaling.", recommendation=rec)
    return RuleResult("RC-04", RuleStatus.PASS, "Adequate Batching", f"Batch {batch_size} throughput is satisfactory.")
