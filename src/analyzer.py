"""
analyzer.py — Module phân tích kết quả profiling & Rule Catalog
=================================================================
Thành phần:
  - RuleResult        : Dataclass chứa kết quả một quy tắc
  - RuleCatalog       : Bộ quy tắc R-A, R-B, R-C, R-D (Heuristics)
      R-A: Memory Analysis (OOM risk, fragmentation, allocation hotspots)
      R-B: Compute Analysis (layer speed imbalance, FLOPS bottleneck)
      R-C: Data I/O Analysis (DataLoader bottleneck)
      R-D: Architecture Analysis (parameter efficiency, depth imbalance)
  - ABComparisonEngine: So sánh thống kê hai phiên profiling

Lưu trữ: JSON + CSV (pandas)
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger("analyzer")


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder xử lý an toàn các kiểu dữ liệu numpy/scipy
    thường xuất hiện khi dùng pandas aggregation và scipy stats.
    """
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.integer,)):          # np.int32, np.int64, ...
            return int(obj)
        if isinstance(obj, (np.floating,)):         # np.float32, np.float64, ...
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# ─────────────────────────────────────────────────────────────────────────────
# Enums & Constants
# ─────────────────────────────────────────────────────────────────────────────

class Severity(str, Enum):
    """Mức độ nghiêm trọng của vấn đề phát hiện."""
    CRITICAL = "CRITICAL"   # Cần xử lý ngay, ảnh hưởng nặng
    HIGH     = "HIGH"       # Vấn đề quan trọng
    MEDIUM   = "MEDIUM"     # Cần chú ý
    LOW      = "LOW"        # Gợi ý cải thiện
    OK       = "OK"         # Không có vấn đề


class RuleCategory(str, Enum):
    """Danh mục quy tắc heuristic."""
    MEMORY    = "R-A"   # Memory Analysis
    COMPUTE   = "R-B"   # Compute Analysis
    IO        = "R-C"   # Data I/O Analysis
    ARCH      = "R-D"   # Architecture Analysis


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RuleResult:
    """Kết quả áp dụng một quy tắc heuristic."""
    rule_id:       str
    rule_name:     str
    category:      str          # RuleCategory value
    severity:      str          # Severity value
    passed:        bool
    message:       str
    details:       Dict[str, Any] = field(default_factory=dict)
    suggestions:   List[str]      = field(default_factory=list)
    affected_layers: List[str]    = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisReport:
    """Báo cáo phân tích đầy đủ cho một mô hình."""
    model_name:    str
    device:        str
    total_rules:   int
    passed:        int
    failed:        int
    critical:      int
    rules:         List[RuleResult] = field(default_factory=list)
    metadata:      Dict[str, Any]   = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "model_name":  self.model_name,
            "device":      self.device,
            "total_rules": self.total_rules,
            "passed":      self.passed,
            "failed":      self.failed,
            "critical":    self.critical,
            "metadata":    self.metadata,
            "rules":       [r.to_dict() for r in self.rules],
        }
        return d

    def save_json(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        logger.info(f"Analysis report -> {path}")
        return path

    def save_csv(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for r in self.rules:
            row = {k: v for k, v in r.to_dict().items()
                   if not isinstance(v, (list, dict))}
            row["suggestions"]     = " | ".join(r.suggestions)
            row["affected_layers"] = " | ".join(r.affected_layers)
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        logger.info(f"Analysis CSV → {path}")
        return path

    def print_summary(self) -> None:
        """In tóm tắt ra console."""
        SEP = "─" * 70
        print(f"\n{SEP}")
        print(f"📊 ANALYSIS REPORT: {self.model_name}  [{self.device}]")
        print(f"{SEP}")
        print(f"  Tổng quy tắc : {self.total_rules}")
        print(f"  ✅ Passed    : {self.passed}")
        print(f"  ❌ Failed    : {self.failed}")
        print(f"  🔴 Critical  : {self.critical}")
        print(f"{SEP}")
        for r in sorted(self.rules, key=lambda x: (x.passed, x.severity)):
            icon = "✅" if r.passed else ("🔴" if r.severity == Severity.CRITICAL else "⚠️")
            print(f"  {icon} [{r.rule_id}] {r.rule_name} — {r.severity}")
            print(f"      {r.message}")
            if r.suggestions:
                for s in r.suggestions:
                    print(f"      💡 {s}")
        print(f"{SEP}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers nội bộ
# ─────────────────────────────────────────────────────────────────────────────

def _load_df(source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """Nạp DataFrame từ path CSV hoặc trực tiếp từ DataFrame."""
    if isinstance(source, pd.DataFrame):
        return source.copy()
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    return pd.read_csv(path)


def _percentile(arr: np.ndarray, pct: float) -> float:
    """Tính percentile an toàn."""
    if len(arr) == 0:
        return 0.0
    return float(np.percentile(arr, pct))


def _iqr_outliers(arr: np.ndarray, k: float = 1.5) -> np.ndarray:
    """Trả về mask của outlier theo IQR method."""
    if len(arr) < 4:
        return np.zeros(len(arr), dtype=bool)
    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
    iqr = q3 - q1
    return (arr < q1 - k * iqr) | (arr > q3 + k * iqr)


# ─────────────────────────────────────────────────────────────────────────────
# Rule Catalog
# ─────────────────────────────────────────────────────────────────────────────

class RuleCatalog:
    """
    Bộ quy tắc heuristic phân tích bottleneck mô hình PyTorch.

    Nhóm R-A: Memory Analysis
        RA-01 OOM Risk Detection
        RA-02 Memory Fragmentation Risk
        RA-03 Memory Allocation Hotspot
        RA-04 Peak Memory Efficiency
        RA-05 Monotonic Memory Growth (Memory Leak heuristic)

    Nhóm R-B: Compute Analysis
        RB-01 Layer Speed Imbalance (Top-K slowest layers)
        RB-02 Compute Variance (Inconsistent timing)
        RB-03 Serial Bottleneck Layer
        RB-04 Forward Time per Parameter Efficiency
        RB-05 Compute Utilization Drop Detection

    Nhóm R-C: Data I/O Analysis
        RC-01 DataLoader I/O Bottleneck
        RC-02 DataLoader Variance (inconsistent load times)
        RC-03 Compute Starvation (compute too fast relative to I/O)
        RC-04 Zero-Worker Warning

    Nhóm R-D: Architecture Analysis
        RD-01 Parameter Concentration (one layer holds most params)
        RD-02 Dead Layer Detection (zero output variance heuristic)
        RD-03 Depth vs. Width Imbalance
        RD-04 Layer Type Distribution
        RD-05 Param Count vs. Time Efficiency
    """

    # ── Ngưỡng mặc định (có thể override khi khởi tạo) ───────────────────

    DEFAULT_THRESHOLDS = {
        # R-A
        "ra01_oom_risk_pct":         85.0,   # % GPU mem → cảnh báo OOM
        "ra01_oom_critical_pct":     95.0,   # % GPU mem → critical OOM
        "ra02_frag_threshold_mb":    500.0,  # MB mem delta bất thường
        "ra03_hotspot_top_k":        5,      # Top-K layer tiêu thụ nhiều mem nhất
        "ra03_hotspot_ratio":        0.5,    # Top-K layers chiếm > 50% tổng mem
        "ra04_efficiency_ratio":     0.7,    # Peak mem / total_params phải hợp lý
        "ra05_growth_window":        5,      # Số iteration để kiểm tra memory leak

        # R-B
        "rb01_top_k":                5,      # Top-K layer chậm nhất
        "rb01_slow_ratio":           0.5,    # Top-K layers chiếm > 50% tổng time
        "rb02_cv_threshold":         0.5,    # Coefficient of Variation > 50% → bất ổn
        "rb03_bottleneck_ratio":     0.3,    # 1 layer chiếm > 30% tổng time
        "rb04_ms_per_mparam":        5.0,    # ms/triệu param ngưỡng cao
        "rb05_util_drop_pct":        40.0,   # GPU util giảm > 40% → drop

        # R-C
        "rc01_io_bottleneck_ratio":  0.4,    # Load time > 40% tổng time
        "rc01_io_critical_ratio":    0.6,    # Load time > 60% tổng time → critical
        "rc02_cv_threshold":         0.8,    # CV load time > 80%
        "rc03_starvation_ratio":     0.1,    # Compute < 10% tổng time

        # R-D
        "rd01_param_concentration":  0.7,    # 1 layer nắm giữ > 70% tổng params
        "rd03_max_type_ratio":       0.8,    # 1 loại layer chiếm > 80%
        "rd05_ms_per_mparam_ratio":  3.0,    # Layer spend > 3x trung bình ms/mparam
    }

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}

    # ─────────────────────────────────────────────────────────────────────────
    # R-A: Memory Analysis
    # ─────────────────────────────────────────────────────────────────────────

    def ra01_oom_risk(
        self,
        df: pd.DataFrame,
        total_gpu_mem_mb: float,
    ) -> RuleResult:
        """
        RA-01: Phát hiện nguy cơ OOM (Out-of-Memory).
        Kiểm tra peak memory usage so với tổng dung lượng GPU.
        """
        RID = "RA-01"
        if total_gpu_mem_mb <= 0 or "gpu_mem_after_mb" not in df.columns:
            return RuleResult(
                rule_id="RA-01", rule_name="OOM Risk Detection",
                category=RuleCategory.MEMORY, severity=Severity.OK,
                passed=True, message="Không có dữ liệu GPU memory để kiểm tra.",
            )

        peak_mb  = float(df["gpu_mem_after_mb"].max())
        usage_pct = (peak_mb / total_gpu_mem_mb) * 100.0

        crit_pct = self.thresholds["ra01_oom_critical_pct"]
        warn_pct = self.thresholds["ra01_oom_risk_pct"]

        if usage_pct >= crit_pct:
            severity = Severity.CRITICAL
            passed   = False
            msg = (f"🔴 Nguy cơ OOM nghiêm trọng: Peak={peak_mb:.1f}MB "
                   f"({usage_pct:.1f}% / {total_gpu_mem_mb:.0f}MB tổng GPU).")
            suggestions = [
                "Giảm batch_size ngay lập tức.",
                "Bật torch.cuda.empty_cache() sau mỗi batch.",
                "Dùng gradient checkpointing (torch.utils.checkpoint).",
                "Xem xét mixed-precision (torch.autocast).",
            ]
        elif usage_pct >= warn_pct:
            severity = Severity.HIGH
            passed   = False
            msg = (f"⚠️ Cảnh báo OOM: Peak={peak_mb:.1f}MB "
                   f"({usage_pct:.1f}% GPU). Gần ngưỡng giới hạn.")
            suggestions = [
                "Giảm batch_size hoặc model size.",
                "Kích hoạt mixed-precision training.",
            ]
        else:
            severity = Severity.OK
            passed   = True
            msg = (f"✅ Memory OK: Peak={peak_mb:.1f}MB "
                   f"({usage_pct:.1f}% / {total_gpu_mem_mb:.0f}MB).")
            suggestions = []

        return RuleResult(
            rule_id="RA-01", rule_name="OOM Risk Detection",
            category=RuleCategory.MEMORY, severity=severity,
            passed=passed, message=msg,
            details={"peak_mb": peak_mb, "usage_pct": usage_pct,
                     "total_gpu_mb": total_gpu_mem_mb},
            suggestions=suggestions,
        )

    def ra02_memory_fragmentation(self, df: pd.DataFrame) -> RuleResult:
        """
        RA-02: Phát hiện nguy cơ phân mảnh bộ nhớ GPU.
        Kiểm tra các bước nhảy bất thường trong gpu_mem_delta_mb.
        """
        if "gpu_mem_delta_mb" not in df.columns:
            return RuleResult(
                rule_id="RA-02", rule_name="Memory Fragmentation Risk",
                category=RuleCategory.MEMORY, severity=Severity.OK,
                passed=True, message="Không có dữ liệu gpu_mem_delta.",
            )

        deltas   = df["gpu_mem_delta_mb"].abs().values
        thresh   = self.thresholds["ra02_frag_threshold_mb"]
        outliers = deltas > thresh
        n_out    = int(outliers.sum())

        if n_out > 0:
            affected = df.loc[outliers, "layer_name"].tolist()[:10]
            severity = Severity.HIGH if n_out > 5 else Severity.MEDIUM
            return RuleResult(
                rule_id="RA-02", rule_name="Memory Fragmentation Risk",
                category=RuleCategory.MEMORY, severity=severity,
                passed=False,
                message=(f"Phát hiện {n_out} layer có biến động memory > "
                         f"{thresh:.0f}MB — dấu hiệu phân mảnh bộ nhớ."),
                details={"n_anomaly_layers": n_out, "threshold_mb": thresh,
                         "max_delta_mb": float(deltas.max())},
                suggestions=[
                    "Gọi torch.cuda.empty_cache() thường xuyên hơn.",
                    "Xem xét dùng memory-efficient attention.",
                    "Tránh lưu intermediate tensors không cần thiết.",
                ],
                affected_layers=affected,
            )

        return RuleResult(
            rule_id="RA-02", rule_name="Memory Fragmentation Risk",
            category=RuleCategory.MEMORY, severity=Severity.OK,
            passed=True,
            message=f"✅ Không phát hiện biến động memory bất thường (ngưỡng {thresh:.0f}MB).",
            details={"max_delta_mb": float(deltas.max()) if len(deltas) > 0 else 0.0},
        )

    def ra03_memory_hotspot(self, df: pd.DataFrame) -> RuleResult:
        """
        RA-03: Xác định các layer tiêu thụ memory nhiều nhất (hotspot).
        Kiểm tra nếu Top-K layer chiếm quá nhiều tổng memory delta.
        """
        if "gpu_mem_delta_mb" not in df.columns or len(df) == 0:
            return RuleResult(
                rule_id="RA-03", rule_name="Memory Allocation Hotspot",
                category=RuleCategory.MEMORY, severity=Severity.OK,
                passed=True, message="Không đủ dữ liệu.",
            )

        k       = int(self.thresholds["ra03_hotspot_top_k"])
        ratio_t = self.thresholds["ra03_hotspot_ratio"]

        # Lấy trung bình theo layer
        agg = df.groupby("layer_name")["gpu_mem_delta_mb"].mean().abs()
        total_mem = float(agg.sum())

        if total_mem <= 0:
            return RuleResult(
                rule_id="RA-03", rule_name="Memory Allocation Hotspot",
                category=RuleCategory.MEMORY, severity=Severity.OK,
                passed=True, message="Tổng delta memory = 0, không phát hiện hotspot.",
            )

        top_k      = agg.nlargest(k)
        top_k_sum  = float(top_k.sum())
        top_k_ratio = top_k_sum / total_mem

        if top_k_ratio >= ratio_t:
            return RuleResult(
                rule_id="RA-03", rule_name="Memory Allocation Hotspot",
                category=RuleCategory.MEMORY, severity=Severity.HIGH,
                passed=False,
                message=(f"Top-{k} layers chiếm {top_k_ratio*100:.1f}% tổng memory allocation. "
                         f"Phát hiện memory hotspot."),
                details={"top_k": k, "top_k_ratio": round(top_k_ratio, 4),
                         "hotspot_layers": top_k.to_dict()},
                suggestions=[
                    "Tối ưu hoá các layer hotspot (giảm size intermediate tensor).",
                    "Cân nhắc gradient checkpointing cho các block hotspot.",
                ],
                affected_layers=list(top_k.index),
            )

        return RuleResult(
            rule_id="RA-03", rule_name="Memory Allocation Hotspot",
            category=RuleCategory.MEMORY, severity=Severity.OK,
            passed=True,
            message=f"✅ Memory phân phối hợp lý: Top-{k} chiếm {top_k_ratio*100:.1f}%.",
            details={"top_k_ratio": round(top_k_ratio, 4)},
        )

    def ra04_peak_memory_efficiency(
        self,
        df: pd.DataFrame,
        total_params: int,
    ) -> RuleResult:
        """
        RA-04: Đánh giá hiệu quả sử dụng peak memory so với tổng tham số.
        Ratio = peak_memory_MB / (total_params_M) — mức baseline tham khảo.
        """
        if "gpu_mem_after_mb" not in df.columns or total_params <= 0:
            return RuleResult(
                rule_id="RA-04", rule_name="Peak Memory Efficiency",
                category=RuleCategory.MEMORY, severity=Severity.OK,
                passed=True, message="Không đủ dữ liệu để đánh giá.",
            )

        peak_mb     = float(df["gpu_mem_after_mb"].max())
        params_m    = total_params / 1e6
        ratio       = peak_mb / params_m  # MB per million params

        # FP32: ~4 bytes/param → 4MB/Mparam là baseline lý thuyết
        # Thực tế (activations, optimizer state) thường 10-30x
        baseline    = 4.0   # MB/Mparam minimum
        warn_limit  = 50.0  # MB/Mparam

        if ratio > warn_limit:
            severity = Severity.MEDIUM
            passed   = False
            msg = (f"Memory/param ratio cao: {ratio:.1f}MB/Mparam "
                   f"(params={params_m:.1f}M, peak={peak_mb:.1f}MB).")
            suggestions = [
                "Kiểm tra xem có lưu activations thừa không.",
                "Dùng half-precision (fp16/bf16) để giảm bộ nhớ.",
            ]
        else:
            severity = Severity.OK
            passed   = True
            msg = (f"✅ Memory/param ratio hợp lý: {ratio:.1f}MB/Mparam.")
            suggestions = []

        return RuleResult(
            rule_id="RA-04", rule_name="Peak Memory Efficiency",
            category=RuleCategory.MEMORY, severity=severity,
            passed=passed, message=msg,
            details={"peak_mb": peak_mb, "total_params_M": round(params_m, 2),
                     "ratio_mb_per_mparam": round(ratio, 2)},
            suggestions=suggestions,
        )

    def ra05_memory_growth(self, df: pd.DataFrame) -> RuleResult:
        """
        RA-05: Phát hiện xu hướng tăng memory liên tục qua các iteration
               (heuristic phát hiện memory leak).
        """
        if "gpu_mem_after_mb" not in df.columns or "iteration" not in df.columns:
            return RuleResult(
                rule_id="RA-05", rule_name="Memory Leak Heuristic",
                category=RuleCategory.MEMORY, severity=Severity.OK,
                passed=True, message="Không đủ dữ liệu iteration.",
            )

        window = int(self.thresholds["ra05_growth_window"])
        # Lấy peak memory theo iteration
        iter_peak = df.groupby("iteration")["gpu_mem_after_mb"].max().sort_index()
        if len(iter_peak) < window + 1:
            return RuleResult(
                rule_id="RA-05", rule_name="Memory Leak Heuristic",
                category=RuleCategory.MEMORY, severity=Severity.OK,
                passed=True, message=f"Ít hơn {window+1} iterations, bỏ qua.",
            )

        # Kiểm tra xu hướng tuyến tính
        x   = np.arange(len(iter_peak))
        y   = iter_peak.values
        slope, intercept, r_value, p_value, _ = scipy_stats.linregress(x, y)

        trend_mb_per_iter = float(slope)
        r2 = float(r_value ** 2)

        # Đáng báo động nếu: slope > 1MB/iter VÀ R² > 0.85
        if trend_mb_per_iter > 1.0 and r2 > 0.85:
            return RuleResult(
                rule_id="RA-05", rule_name="Memory Leak Heuristic",
                category=RuleCategory.MEMORY, severity=Severity.HIGH,
                passed=False,
                message=(f"⚠️ Phát hiện tăng memory liên tục: "
                         f"+{trend_mb_per_iter:.2f}MB/iteration (R²={r2:.3f}). "
                         "Có thể là memory leak."),
                details={"slope_mb_per_iter": round(trend_mb_per_iter, 4),
                         "r_squared": round(r2, 4), "p_value": round(p_value, 4)},
                suggestions=[
                    "Kiểm tra vòng lặp training có giữ lại references không.",
                    "Gọi del tensor và torch.cuda.empty_cache() sau mỗi batch.",
                    "Dùng torch.no_grad() trong validation loop.",
                ],
            )

        return RuleResult(
            rule_id="RA-05", rule_name="Memory Leak Heuristic",
            category=RuleCategory.MEMORY, severity=Severity.OK,
            passed=True,
            message=f"✅ Không phát hiện xu hướng tăng memory: "
                    f"slope={trend_mb_per_iter:.4f}MB/iter, R²={r2:.3f}.",
            details={"slope_mb_per_iter": round(trend_mb_per_iter, 4),
                     "r_squared": round(r2, 4)},
        )

    # ─────────────────────────────────────────────────────────────────────────
    # R-B: Compute Analysis
    # ─────────────────────────────────────────────────────────────────────────

    def rb01_layer_speed_imbalance(self, df: pd.DataFrame) -> RuleResult:
        """
        RB-01: Top-K layers chậm nhất chiếm quá nhiều tổng thời gian.
        """
        if "forward_time_ms" not in df.columns or len(df) == 0:
            return RuleResult(
                rule_id="RB-01", rule_name="Layer Speed Imbalance",
                category=RuleCategory.COMPUTE, severity=Severity.OK,
                passed=True, message="Không đủ dữ liệu.",
            )

        k       = int(self.thresholds["rb01_top_k"])
        ratio_t = self.thresholds["rb01_slow_ratio"]

        agg        = df.groupby("layer_name")["forward_time_ms"].mean()
        total_time = float(agg.sum())
        if total_time <= 0:
            return RuleResult(
                rule_id="RB-01", rule_name="Layer Speed Imbalance",
                category=RuleCategory.COMPUTE, severity=Severity.OK,
                passed=True, message="Tổng forward time = 0.",
            )

        top_k      = agg.nlargest(k)
        top_k_sum  = float(top_k.sum())
        top_k_ratio = top_k_sum / total_time

        if top_k_ratio >= ratio_t:
            return RuleResult(
                rule_id="RB-01", rule_name="Layer Speed Imbalance",
                category=RuleCategory.COMPUTE, severity=Severity.HIGH,
                passed=False,
                message=(f"Top-{k} layers chậm nhất chiếm {top_k_ratio*100:.1f}% "
                         f"tổng forward time ({total_time:.2f}ms)."),
                details={"top_k": k, "ratio": round(top_k_ratio, 4),
                         "total_time_ms": round(total_time, 3),
                         "slowest_layers": top_k.round(4).to_dict()},
                suggestions=[
                    "Profile các layer nặng này bằng torch.profiler chi tiết.",
                    "Xem xét thay thế bằng layer nhẹ hơn (depthwise conv, ...).",
                    "Kiểm tra kích thước input/output của các layer này.",
                ],
                affected_layers=list(top_k.index),
            )

        return RuleResult(
            rule_id="RB-01", rule_name="Layer Speed Imbalance",
            category=RuleCategory.COMPUTE, severity=Severity.OK,
            passed=True,
            message=f"✅ Compute phân phối hợp lý: Top-{k} chiếm {top_k_ratio*100:.1f}%.",
            details={"top_k_ratio": round(top_k_ratio, 4), "top_k_layers": top_k.round(4).to_dict()},
        )

    def rb02_compute_variance(self, df: pd.DataFrame) -> RuleResult:
        """
        RB-02: Phương sai thời gian tính toán bất thường (timing noise).
        Đo Coefficient of Variation (CV) = std/mean của forward_time_ms.
        """
        if "forward_time_ms" not in df.columns or len(df) < 10:
            return RuleResult(
                rule_id="RB-02", rule_name="Compute Variance",
                category=RuleCategory.COMPUTE, severity=Severity.OK,
                passed=True, message="Không đủ dữ liệu để đo variance.",
            )

        cv_thresh = self.thresholds["rb02_cv_threshold"]
        mean_t    = float(df["forward_time_ms"].mean())
        std_t     = float(df["forward_time_ms"].std())
        cv        = std_t / mean_t if mean_t > 0 else 0.0

        # Layers có CV cao nhất
        layer_cv = (
            df.groupby("layer_name")["forward_time_ms"]
              .agg(["mean", "std"])
              .dropna()
        )
        layer_cv["cv"] = layer_cv["std"] / layer_cv["mean"].clip(lower=1e-9)
        unstable = layer_cv[layer_cv["cv"] > cv_thresh].sort_values("cv", ascending=False)

        if len(unstable) > 0:
            return RuleResult(
                rule_id="RB-02", rule_name="Compute Variance",
                category=RuleCategory.COMPUTE, severity=Severity.MEDIUM,
                passed=False,
                message=(f"{len(unstable)} layers có CV > {cv_thresh:.1f} "
                         f"(timing không ổn định). Overall CV={cv:.3f}."),
                details={"overall_cv": round(cv, 4), "n_unstable_layers": len(unstable),
                         "unstable_layers": unstable.head(5)["cv"].round(4).to_dict()},
                suggestions=[
                    "Tăng số iteration warmup.",
                    "Kiểm tra tác động của background processes (Windows).",
                    "Dùng torch.cuda.synchronize() trước/sau đo lường.",
                    "Chạy lại trên hệ thống ít tải hơn.",
                ],
                affected_layers=list(unstable.index[:10]),
            )

        return RuleResult(
            rule_id="RB-02", rule_name="Compute Variance",
            category=RuleCategory.COMPUTE, severity=Severity.OK,
            passed=True,
            message=f"✅ Timing ổn định: Overall CV={cv:.3f} (ngưỡng {cv_thresh}).",
            details={"overall_cv": round(cv, 4)},
        )

    def rb03_serial_bottleneck(self, df: pd.DataFrame) -> RuleResult:
        """
        RB-03: Một layer đơn lẻ chiếm quá nhiều tổng thời gian.
        """
        if "forward_time_ms" not in df.columns or len(df) == 0:
            return RuleResult(
                rule_id="RB-03", rule_name="Serial Bottleneck Layer",
                category=RuleCategory.COMPUTE, severity=Severity.OK,
                passed=True, message="Không đủ dữ liệu.",
            )

        ratio_t  = self.thresholds["rb03_bottleneck_ratio"]
        agg      = df.groupby("layer_name")["forward_time_ms"].mean()
        total_t  = float(agg.sum())
        if total_t <= 0:
            return RuleResult(
                rule_id="RB-03", rule_name="Serial Bottleneck Layer",
                category=RuleCategory.COMPUTE, severity=Severity.OK,
                passed=True, message="Tổng time = 0.",
            )

        max_layer = agg.idxmax()
        max_time  = float(agg.max())
        ratio     = max_time / total_t

        if ratio >= ratio_t:
            return RuleResult(
                rule_id="RB-03", rule_name="Serial Bottleneck Layer",
                category=RuleCategory.COMPUTE, severity=Severity.CRITICAL,
                passed=False,
                message=(f"Layer '{max_layer}' chiếm {ratio*100:.1f}% tổng forward time "
                         f"({max_time:.3f}ms / {total_t:.3f}ms). Bottleneck nghiêm trọng!"),
                details={"bottleneck_layer": max_layer, "layer_time_ms": round(max_time, 4),
                         "total_time_ms": round(total_t, 4), "ratio": round(ratio, 4)},
                suggestions=[
                    f"Tối ưu hoá hoặc thay thế layer '{max_layer}'.",
                    "Xem xét chia nhỏ layer này thành các sub-operations.",
                    "Kiểm tra kích thước tensor input/output của layer này.",
                    "Cân nhắc dùng torch.compile() cho layer này.",
                ],
                affected_layers=[max_layer],
            )

        return RuleResult(
            rule_id="RB-03", rule_name="Serial Bottleneck Layer",
            category=RuleCategory.COMPUTE, severity=Severity.OK,
            passed=True,
            message=f"✅ Không có layer đơn lẻ gây bottleneck (max={ratio*100:.1f}%).",
            details={"max_ratio": round(ratio, 4), "max_layer": max_layer},
        )

    def rb04_time_per_param_efficiency(self, df: pd.DataFrame) -> RuleResult:
        """
        RB-04: Đánh giá hiệu quả thời gian per triệu tham số (ms/Mparam).
        Layer nào có ms/Mparam cao bất thường thì kém hiệu quả.
        """
        if not {"forward_time_ms", "param_count"}.issubset(df.columns):
            return RuleResult(
                rule_id="RB-04", rule_name="Time per Parameter Efficiency",
                category=RuleCategory.COMPUTE, severity=Severity.OK,
                passed=True, message="Không đủ dữ liệu.",
            )

        thresh_ms = self.thresholds["rb04_ms_per_mparam"]
        agg = df.groupby("layer_name").agg(
            mean_time=("forward_time_ms", "mean"),
            total_params=("param_count",  "first"),
        )
        agg = agg[agg["total_params"] > 1000]  # Bỏ qua layer quá nhỏ
        if len(agg) == 0:
            return RuleResult(
                rule_id="RB-04", rule_name="Time per Parameter Efficiency",
                category=RuleCategory.COMPUTE, severity=Severity.OK,
                passed=True, message="Không có layer đủ params để đánh giá.",
            )

        agg["ms_per_mparam"] = agg["mean_time"] / (agg["total_params"] / 1e6).clip(lower=1e-9)
        inefficient = agg[agg["ms_per_mparam"] > thresh_ms].sort_values("ms_per_mparam", ascending=False)

        if len(inefficient) > 0:
            return RuleResult(
                rule_id="RB-04", rule_name="Time per Parameter Efficiency",
                category=RuleCategory.COMPUTE, severity=Severity.MEDIUM,
                passed=False,
                message=(f"{len(inefficient)} layers có ms/Mparam > {thresh_ms:.1f} "
                         f"(kém hiệu quả tính toán)."),
                details={"threshold": thresh_ms,
                         "inefficient": inefficient["ms_per_mparam"].round(4).head(5).to_dict()},
                suggestions=[
                    "Kiểm tra xem layer có đang chạy trên CPU không.",
                    "Xem xét fusing operations (conv+bn+relu) bằng torch.compile.",
                    "Các layer attention thường có ms/Mparam cao — dùng Flash Attention.",
                ],
                affected_layers=list(inefficient.index[:10]),
            )

        return RuleResult(
            rule_id="RB-04", rule_name="Time per Parameter Efficiency",
            category=RuleCategory.COMPUTE, severity=Severity.OK,
            passed=True,
            message=f"✅ ms/Mparam của tất cả layers hợp lý (ngưỡng {thresh_ms}).",
            details={"max_ms_per_mparam": round(float(agg["ms_per_mparam"].max()), 4)},
        )

    def rb05_gpu_utilization_drop(
        self,
        df: pd.DataFrame,
        device: str = "cpu",
    ) -> RuleResult:
        """
        RB-05: Phát hiện các thời điểm utilization giảm mạnh bất thường.
        Hiển thị nhãn 'CPU Utilization Drop' khi device='cpu',
        'GPU Utilization Drop' khi device='cuda'.
        """
        # Nhãn theo device
        is_gpu    = device.startswith("cuda")
        dev_label = "GPU" if is_gpu else "CPU"
        rule_name = f"{dev_label} Utilization Drop"

        if "gpu_util_pct" not in df.columns or len(df) < 5:
            return RuleResult(
                rule_id="RB-05", rule_name=rule_name,
                category=RuleCategory.COMPUTE, severity=Severity.OK,
                passed=True, message=f"Không du du lieu {dev_label} utilization.",
            )

        drop_thresh = self.thresholds["rb05_util_drop_pct"]
        utils       = df["gpu_util_pct"].values
        mean_util   = float(utils.mean())

        if mean_util < 5.0:
            return RuleResult(
                rule_id="RB-05", rule_name=rule_name,
                category=RuleCategory.COMPUTE, severity=Severity.OK,
                passed=True,
                message=(
                    f"{dev_label} util = {mean_util:.1f}% — "
                    + ("CPU-only, ket qua chinh xac tu psutil." if not is_gpu
                       else "Binh thuong cho inference."),
                ),
            )

        low_util_mask = utils < (mean_util - drop_thresh)
        n_drops       = int(low_util_mask.sum())
        layers_drop   = df.loc[low_util_mask, "layer_name"].tolist()[:10]

        if n_drops > len(df) * 0.1:  # > 10% so lan do
            suggestions = [
                "Kiem tra data pipeline co tao ra bubbles (idle time) khong.",
                f"Tang batch size de {dev_label} luon ban.",
            ]
            if is_gpu:
                suggestions.append("Prefetch data voi DataLoader(prefetch_factor=2).")
            else:
                suggestions.append("Kiem tra background process tren Windows (Task Manager).")

            return RuleResult(
                rule_id="RB-05", rule_name=rule_name,
                category=RuleCategory.COMPUTE, severity=Severity.MEDIUM,
                passed=False,
                message=(
                    f"{dev_label} util giam > {drop_thresh:.0f}% so voi mean ({mean_util:.1f}%) "
                    f"tai {n_drops} diem do ({n_drops/len(df)*100:.1f}%)."
                ),
                details={
                    "device":        device,
                    "mean_util_pct": round(mean_util, 2),
                    "n_drops":       n_drops,
                    "drop_threshold":drop_thresh,
                },
                suggestions=suggestions,
                affected_layers=layers_drop,
            )

        return RuleResult(
            rule_id="RB-05", rule_name=rule_name,
            category=RuleCategory.COMPUTE, severity=Severity.OK,
            passed=True,
            message=f"{dev_label} utilization on dinh: mean={mean_util:.1f}%.",
            details={"device": device, "mean_util_pct": round(mean_util, 2), "n_drops": n_drops},
        )

    # ─────────────────────────────────────────────────────────────────────────
    # R-C: Data I/O Analysis
    # ─────────────────────────────────────────────────────────────────────────

    def rc01_dataloader_bottleneck(
        self,
        io_ratio: float,
        mean_load_ms: float,
        mean_compute_ms: float,
    ) -> RuleResult:
        """
        RC-01: DataLoader I/O bottleneck — thời gian tải data chiếm quá nhiều.
        Args:
            io_ratio       : load_time / (load_time + compute_time).
            mean_load_ms   : Thời gian load trung bình mỗi batch (ms).
            mean_compute_ms: Thời gian compute trung bình mỗi batch (ms).
        """
        crit_t  = self.thresholds["rc01_io_critical_ratio"]
        warn_t  = self.thresholds["rc01_io_bottleneck_ratio"]

        if io_ratio >= crit_t:
            severity = Severity.CRITICAL
            passed   = False
            msg = (f"🔴 DataLoader là bottleneck nghiêm trọng: "
                   f"I/O={io_ratio*100:.1f}% thời gian "
                   f"(load={mean_load_ms:.1f}ms vs compute={mean_compute_ms:.1f}ms).")
            suggestions = [
                "Tăng num_workers (khuyến nghị: 4–8 workers trên Colab).",
                "Dùng pin_memory=True nếu có GPU.",
                "Bật persistent_workers=True để tránh re-spawn.",
                "Xem xét cache dataset vào RAM hoặc SSD.",
                "Dùng WebDataset hoặc LMDB cho dataset lớn.",
            ]
        elif io_ratio >= warn_t:
            severity = Severity.HIGH
            passed   = False
            msg = (f"⚠️ DataLoader chậm: I/O={io_ratio*100:.1f}% thời gian "
                   f"(load={mean_load_ms:.1f}ms vs compute={mean_compute_ms:.1f}ms).")
            suggestions = [
                "Tăng num_workers.",
                "Dùng pin_memory=True.",
                "Xem xét giảm augmentation phức tạp hoặc dùng GPU augmentation (Albumentations GPU).",
            ]
        else:
            severity = Severity.OK
            passed   = True
            msg = (f"✅ DataLoader OK: I/O={io_ratio*100:.1f}% "
                   f"(load={mean_load_ms:.1f}ms, compute={mean_compute_ms:.1f}ms).")
            suggestions = []

        return RuleResult(
            rule_id="RC-01", rule_name="DataLoader I/O Bottleneck",
            category=RuleCategory.IO, severity=severity,
            passed=passed, message=msg,
            details={"io_ratio": round(io_ratio, 4), "mean_load_ms": round(mean_load_ms, 3),
                     "mean_compute_ms": round(mean_compute_ms, 3)},
            suggestions=suggestions,
        )

    def rc02_dataloader_variance(
        self,
        load_times: List[float],
    ) -> RuleResult:
        """
        RC-02: Kiểm tra độ không ổn định của DataLoader (CV của load time).
        """
        cv_thresh = self.thresholds["rc02_cv_threshold"]
        if not load_times or len(load_times) < 5:
            return RuleResult(
                rule_id="RC-02", rule_name="DataLoader Timing Variance",
                category=RuleCategory.IO, severity=Severity.OK,
                passed=True, message="Không đủ dữ liệu (< 5 batches).",
            )

        arr  = np.array(load_times, dtype=float)
        mean = float(arr.mean())
        std  = float(arr.std())
        cv   = std / mean if mean > 0 else 0.0

        if cv > cv_thresh:
            return RuleResult(
                rule_id="RC-02", rule_name="DataLoader Timing Variance",
                category=RuleCategory.IO, severity=Severity.MEDIUM,
                passed=False,
                message=(f"DataLoader load time không ổn định: CV={cv:.3f} "
                         f"(mean={mean:.1f}ms, std={std:.1f}ms)."),
                details={"cv": round(cv, 4), "mean_ms": round(mean, 3), "std_ms": round(std, 3)},
                suggestions=[
                    "Kiểm tra ổ đĩa có bị phân mảnh hay I/O contention không.",
                    "Dùng SSD thay HDD cho dataset.",
                    "Kiểm tra network nếu data đang streaming từ GCS/S3.",
                ],
            )

        return RuleResult(
            rule_id="RC-02", rule_name="DataLoader Timing Variance",
            category=RuleCategory.IO, severity=Severity.OK,
            passed=True,
            message=f"✅ DataLoader ổn định: CV={cv:.3f}.",
            details={"cv": round(cv, 4)},
        )

    def rc03_compute_starvation(
        self,
        io_ratio: float,
    ) -> RuleResult:
        """
        RC-03: GPU đang bị 'đói' tính toán vì load quá nhanh (hiếm, nhưng cần check).
        """
        starv_t = self.thresholds["rc03_starvation_ratio"]
        compute_ratio = 1.0 - io_ratio

        if compute_ratio < starv_t:
            return RuleResult(
                rule_id="RC-03", rule_name="Compute Starvation",
                category=RuleCategory.IO, severity=Severity.MEDIUM,
                passed=False,
                message=(f"Compute time cực thấp ({compute_ratio*100:.1f}%) — "
                         "model quá nhỏ so với batch size hoặc data không được xử lý đúng."),
                details={"compute_ratio": round(compute_ratio, 4)},
                suggestions=[
                    "Kiểm tra model có đang chạy đúng trên GPU không.",
                    "Tăng model complexity hoặc batch size.",
                ],
            )

        return RuleResult(
            rule_id="RC-03", rule_name="Compute Starvation",
            category=RuleCategory.IO, severity=Severity.OK,
            passed=True,
            message=f"✅ Compute ratio hợp lý: {compute_ratio*100:.1f}%.",
        )

    def rc04_zero_worker_warning(self, worker_count: int) -> RuleResult:
        """
        RC-04: Cảnh báo nếu DataLoader dùng 0 workers (single-process loading).
        """
        if worker_count == 0:
            return RuleResult(
                rule_id="RC-04", rule_name="Zero Worker Warning",
                category=RuleCategory.IO, severity=Severity.MEDIUM,
                passed=False,
                message="DataLoader dùng num_workers=0 (single-process I/O). Có thể chậm hơn nhiều.",
                details={"worker_count": 0},
                suggestions=[
                    "Đặt num_workers=2 hoặc 4 để dùng multi-process loading.",
                    "Trên Windows: dùng num_workers=2 (tránh overhead spawn quá nhiều process).",
                    "Trên Colab: num_workers=4 là khuyến nghị tốt.",
                ],
            )

        return RuleResult(
            rule_id="RC-04", rule_name="Zero Worker Warning",
            category=RuleCategory.IO, severity=Severity.OK,
            passed=True,
            message=f"✅ DataLoader dùng {worker_count} workers.",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # R-D: Architecture Analysis
    # ─────────────────────────────────────────────────────────────────────────

    def rd01_parameter_concentration(self, df: pd.DataFrame) -> RuleResult:
        """
        RD-01: Một layer đơn lẻ chứa quá nhiều tham số (concentration > threshold).
        """
        if "param_count" not in df.columns or len(df) == 0:
            return RuleResult(
                rule_id="RD-01", rule_name="Parameter Concentration",
                category=RuleCategory.ARCH, severity=Severity.OK,
                passed=True, message="Không đủ dữ liệu.",
            )

        thresh = self.thresholds["rd01_param_concentration"]
        agg    = df.groupby("layer_name")["param_count"].first()
        total  = float(agg.sum())
        if total <= 0:
            return RuleResult(
                rule_id="RD-01", rule_name="Parameter Concentration",
                category=RuleCategory.ARCH, severity=Severity.OK,
                passed=True, message="Tổng params = 0.",
            )

        max_layer = agg.idxmax()
        max_ratio = float(agg.max()) / total

        if max_ratio >= thresh:
            return RuleResult(
                rule_id="RD-01", rule_name="Parameter Concentration",
                category=RuleCategory.ARCH, severity=Severity.HIGH,
                passed=False,
                message=(f"Layer '{max_layer}' chứa {max_ratio*100:.1f}% tổng params. "
                         "Thiết kế mất cân bằng."),
                details={"dominant_layer": max_layer,
                         "dominant_ratio": round(max_ratio, 4),
                         "total_params": int(total)},
                suggestions=[
                    "Xem xét chia nhỏ layer này.",
                    "Áp dụng weight sharing hoặc low-rank factorization.",
                    "Kiểm tra xem lớp embedding có quá lớn không.",
                ],
                affected_layers=[max_layer],
            )

        return RuleResult(
            rule_id="RD-01", rule_name="Parameter Concentration",
            category=RuleCategory.ARCH, severity=Severity.OK,
            passed=True,
            message=f"✅ Params phân phối hợp lý: max layer = {max_ratio*100:.1f}%.",
            details={"max_ratio": round(max_ratio, 4)},
        )

    def rd02_dead_layer_detection(self, df: pd.DataFrame) -> RuleResult:
        """
        RD-02: Phát hiện layer có thời gian = 0 (có thể là dead layer hoặc bị skip).
        """
        if "forward_time_ms" not in df.columns:
            return RuleResult(
                rule_id="RD-02", rule_name="Dead Layer Detection",
                category=RuleCategory.ARCH, severity=Severity.OK,
                passed=True, message="Không đủ dữ liệu.",
            )

        agg          = df.groupby("layer_name")["forward_time_ms"].mean()
        dead_layers  = agg[agg < 1e-6].index.tolist()

        if len(dead_layers) > 0:
            has_params = df[df["layer_name"].isin(dead_layers)]["param_count"].max()
            severity   = Severity.MEDIUM if has_params > 0 else Severity.LOW

            return RuleResult(
                rule_id="RD-02", rule_name="Dead Layer Detection",
                category=RuleCategory.ARCH, severity=severity,
                passed=False,
                message=(f"Phát hiện {len(dead_layers)} layer có forward_time ≈ 0ms "
                         "(có thể là dead layer, short-circuit, hoặc bị cắt bởi conditional)."),
                details={"n_dead_layers": len(dead_layers)},
                suggestions=[
                    "Kiểm tra các layer này có được kích hoạt đúng không.",
                    "Xem xét loại bỏ dead layers để giảm complexity.",
                ],
                affected_layers=dead_layers[:20],
            )

        return RuleResult(
            rule_id="RD-02", rule_name="Dead Layer Detection",
            category=RuleCategory.ARCH, severity=Severity.OK,
            passed=True,
            message=f"✅ Không phát hiện dead layer.",
        )

    def rd03_layer_type_distribution(self, df: pd.DataFrame) -> RuleResult:
        """
        RD-03: Phân tích phân phối loại layer. Cảnh báo nếu một loại chiếm áp đảo.
        """
        if "layer_type" not in df.columns or len(df) == 0:
            return RuleResult(
                rule_id="RD-03", rule_name="Layer Type Distribution",
                category=RuleCategory.ARCH, severity=Severity.OK,
                passed=True, message="Không đủ dữ liệu.",
            )

        thresh    = self.thresholds["rd03_max_type_ratio"]
        type_dist = df.groupby("layer_type")["layer_name"].nunique()
        total_n   = int(type_dist.sum())
        if total_n == 0:
            return RuleResult(
                rule_id="RD-03", rule_name="Layer Type Distribution",
                category=RuleCategory.ARCH, severity=Severity.OK,
                passed=True, message="Không có layer nào.",
            )

        type_ratios = type_dist / total_n
        dom_type    = type_ratios.idxmax()
        dom_ratio   = float(type_ratios.max())

        dist_info = type_dist.to_dict()

        if dom_ratio >= thresh:
            return RuleResult(
                rule_id="RD-03", rule_name="Layer Type Distribution",
                category=RuleCategory.ARCH, severity=Severity.LOW,
                passed=False,
                message=(f"Loại layer '{dom_type}' chiếm {dom_ratio*100:.1f}% "
                         f"({int(type_dist[dom_type])}/{total_n} layers). Kiến trúc đơn điệu?"),
                details={"dominant_type": dom_type, "dominant_ratio": round(dom_ratio, 4),
                         "distribution": dist_info},
                suggestions=[
                    "Xem xét thêm skip connections (ResNet) hoặc attention (Transformer).",
                    "Đây là cảnh báo thấp — có thể là kiến trúc intentional.",
                ],
            )

        return RuleResult(
            rule_id="RD-03", rule_name="Layer Type Distribution",
            category=RuleCategory.ARCH, severity=Severity.OK,
            passed=True,
            message=f"✅ Layer types đa dạng: {len(type_dist)} loại khác nhau.",
            details={"distribution": dist_info},
        )

    def rd04_depth_profile(self, df: pd.DataFrame) -> RuleResult:
        """
        RD-04: Phân tích độ sâu mô hình (số layer) vs. thời gian trung bình mỗi layer.
        """
        if "forward_time_ms" not in df.columns or len(df) == 0:
            return RuleResult(
                rule_id="RD-04", rule_name="Depth Profile",
                category=RuleCategory.ARCH, severity=Severity.OK,
                passed=True, message="Không đủ dữ liệu.",
            )

        n_layers    = df["layer_name"].nunique()
        total_time  = float(df.groupby("layer_name")["forward_time_ms"].mean().sum())
        avg_per_lay = total_time / n_layers if n_layers > 0 else 0.0
        total_params = int(df.groupby("layer_name")["param_count"].first().sum())

        return RuleResult(
            rule_id="RD-04", rule_name="Depth Profile",
            category=RuleCategory.ARCH, severity=Severity.OK,
            passed=True,
            message=(f"Mô hình có {n_layers} layer đo được. "
                     f"Avg time/layer={avg_per_lay:.3f}ms. "
                     f"Total params={total_params/1e6:.2f}M."),
            details={
                "n_layers":          n_layers,
                "total_time_ms":     round(total_time, 3),
                "avg_time_per_layer": round(avg_per_lay, 4),
                "total_params":      total_params,
            },
        )

    def rd05_param_time_outliers(self, df: pd.DataFrame) -> RuleResult:
        """
        RD-05: Phát hiện layer có ms/Mparam cao bất thường so với trung bình mô hình.
        """
        if not {"forward_time_ms", "param_count"}.issubset(df.columns):
            return RuleResult(
                rule_id="RD-05", rule_name="Param-Time Outliers",
                category=RuleCategory.ARCH, severity=Severity.OK,
                passed=True, message="Không đủ dữ liệu.",
            )

        ratio_mult = self.thresholds["rd05_ms_per_mparam_ratio"]
        agg = df.groupby("layer_name").agg(
            mean_time=("forward_time_ms", "mean"),
            total_params=("param_count",  "first"),
        )
        agg = agg[agg["total_params"] > 1000]
        if len(agg) < 3:
            return RuleResult(
                rule_id="RD-05", rule_name="Param-Time Outliers",
                category=RuleCategory.ARCH, severity=Severity.OK,
                passed=True, message="Không đủ layers (< 3) để so sánh.",
            )

        agg["ms_per_mparam"] = agg["mean_time"] / (agg["total_params"] / 1e6).clip(lower=1e-9)
        median_ratio = float(agg["ms_per_mparam"].median())
        outliers     = agg[agg["ms_per_mparam"] > median_ratio * ratio_mult]

        if len(outliers) > 0:
            return RuleResult(
                rule_id="RD-05", rule_name="Param-Time Outliers",
                category=RuleCategory.ARCH, severity=Severity.MEDIUM,
                passed=False,
                message=(f"{len(outliers)} layers có ms/Mparam > {ratio_mult}× median "
                         f"({median_ratio:.2f}ms/Mparam). Kém hiệu quả tương đối."),
                details={"median_ms_per_mparam": round(median_ratio, 4),
                         "multiplier": ratio_mult,
                         "outliers": outliers["ms_per_mparam"].round(4).head(5).to_dict()},
                suggestions=[
                    "Kiểm tra xem các layer này có đang chạy đúng device không.",
                    "Xem xét thay thế bằng layer hiệu quả hơn (ví dụ: depthwise conv).",
                ],
                affected_layers=list(outliers.index[:10]),
            )

        return RuleResult(
            rule_id="RD-05", rule_name="Param-Time Outliers",
            category=RuleCategory.ARCH, severity=Severity.OK,
            passed=True,
            message=f"✅ Không có layer nào có ms/Mparam bất thường.",
            details={"median_ms_per_mparam": round(median_ratio, 4)},
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API: run_all
    # ─────────────────────────────────────────────────────────────────────────

    def run_all(
        self,
        df: pd.DataFrame,
        model_name: str = "model",
        device:     str = "cpu",
        total_gpu_mem_mb: float = 0.0,
        total_params:     int   = 0,
        dataloader_stats: Optional[Dict[str, Any]] = None,
    ) -> AnalysisReport:
        """
        Chạy toàn bộ Rule Catalog trên một DataFrame profiling.

        Args:
            df               : DataFrame chứa layer records (output của collector).
            model_name       : Tên mô hình.
            device           : 'cpu' hoặc 'cuda'.
            total_gpu_mem_mb : Tổng GPU memory (MB) — 0 nếu CPU-only.
            total_params     : Tổng params của mô hình.
            dataloader_stats : Dict từ DataLoaderStats.to_dict() (tùy chọn).

        Returns:
            AnalysisReport đầy đủ.
        """
        results: List[RuleResult] = []

        # ── R-A: Memory ──────────────────────────────────────────────────────
        results.append(self.ra01_oom_risk(df, total_gpu_mem_mb))
        results.append(self.ra02_memory_fragmentation(df))
        results.append(self.ra03_memory_hotspot(df))
        results.append(self.ra04_peak_memory_efficiency(df, total_params))
        results.append(self.ra05_memory_growth(df))

        # ── R-B: Compute ─────────────────────────────────────────────────────
        results.append(self.rb01_layer_speed_imbalance(df))
        results.append(self.rb02_compute_variance(df))
        results.append(self.rb03_serial_bottleneck(df))
        results.append(self.rb04_time_per_param_efficiency(df))
        results.append(self.rb05_gpu_utilization_drop(df, device=device))

        # ── R-C: Data I/O ────────────────────────────────────────────────────
        if dataloader_stats:
            io_ratio     = dataloader_stats.get("io_bottleneck_ratio", 0.0)
            mean_load    = dataloader_stats.get("mean_load_time_ms",    0.0)
            mean_compute = dataloader_stats.get("mean_compute_time_ms", 0.0)
            load_times   = dataloader_stats.get("batch_load_times",     [])
            worker_count = dataloader_stats.get("worker_count",         0)

            results.append(self.rc01_dataloader_bottleneck(io_ratio, mean_load, mean_compute))
            results.append(self.rc02_dataloader_variance(load_times))
            results.append(self.rc03_compute_starvation(io_ratio))
            results.append(self.rc04_zero_worker_warning(worker_count))
        else:
            for rule_id, name in [
                ("RC-01", "DataLoader I/O Bottleneck"),
                ("RC-02", "DataLoader Timing Variance"),
                ("RC-03", "Compute Starvation"),
                ("RC-04", "Zero Worker Warning"),
            ]:
                results.append(RuleResult(
                    rule_id=rule_id, rule_name=name,
                    category=RuleCategory.IO, severity=Severity.OK,
                    passed=True,
                    message="Không có dữ liệu DataLoader — bỏ qua nhóm R-C.",
                ))

        # ── R-D: Architecture ────────────────────────────────────────────────
        results.append(self.rd01_parameter_concentration(df))
        results.append(self.rd02_dead_layer_detection(df))
        results.append(self.rd03_layer_type_distribution(df))
        results.append(self.rd04_depth_profile(df))
        results.append(self.rd05_param_time_outliers(df))

        # ── Tổng hợp ─────────────────────────────────────────────────────────
        passed_count   = sum(1 for r in results if r.passed)
        failed_count   = sum(1 for r in results if not r.passed)
        critical_count = sum(1 for r in results if r.severity == Severity.CRITICAL)

        return AnalysisReport(
            model_name=model_name,
            device=device,
            total_rules=len(results),
            passed=passed_count,
            failed=failed_count,
            critical=critical_count,
            rules=results,
            metadata={
                "total_params":     total_params,
                "total_gpu_mem_mb": total_gpu_mem_mb,
                "has_dataloader":   dataloader_stats is not None,
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# A/B Comparison Engine
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LayerDiff:
    """Kết quả so sánh một layer cụ thể giữa session A và B."""
    layer_name:          str
    mean_time_a_ms:      float
    mean_time_b_ms:      float
    delta_ms:            float    # B - A
    delta_pct:           float    # (B - A) / A * 100
    mem_delta_a_mb:      float
    mem_delta_b_mb:      float
    mem_delta_diff_mb:   float    # B - A (memory delta)
    is_regression:       bool     # True nếu B chậm hơn/dùng nhiều mem hơn A đáng kể
    is_improvement:      bool
    statistical_sig:     bool     # Có ý nghĩa thống kê không (Mann-Whitney p < 0.05)
    p_value:             float


@dataclass
class ABReport:
    """Báo cáo so sánh A/B đầy đủ."""
    session_a_name:  str
    session_b_name:  str
    total_layers:    int
    regressions:     int
    improvements:    int
    unchanged:       int
    overall_speedup: float   # mean_time_A / mean_time_B (> 1 = B nhanh hơn)
    layer_diffs:     List[LayerDiff] = field(default_factory=list)
    summary:         Dict[str, Any]  = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_a": self.session_a_name,
            "session_b": self.session_b_name,
            "total_layers":    self.total_layers,
            "regressions":     self.regressions,
            "improvements":    self.improvements,
            "unchanged":       self.unchanged,
            "overall_speedup": self.overall_speedup,
            "summary":         self.summary,
            "layer_diffs":     [asdict(d) for d in self.layer_diffs],
        }

    def save_json(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        logger.info(f"A/B Report -> {path}")
        return path

    def save_csv(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [asdict(d) for d in self.layer_diffs]
        pd.DataFrame(rows).to_csv(path, index=False)
        logger.info(f"A/B CSV → {path}")
        return path

    def print_summary(self) -> None:
        SEP = "─" * 70
        speedup_icon = "🚀" if self.overall_speedup > 1 else ("🐌" if self.overall_speedup < 1 else "➡️")
        print(f"\n{SEP}")
        print(f"⚡ A/B COMPARISON: {self.session_a_name}  vs  {self.session_b_name}")
        print(f"{SEP}")
        print(f"  Tổng layer so sánh : {self.total_layers}")
        print(f"  ✅ Cải thiện       : {self.improvements}")
        print(f"  ❌ Hồi quy         : {self.regressions}")
        print(f"  ➡️  Không đổi       : {self.unchanged}")
        print(f"  {speedup_icon} Speedup tổng thể  : {self.overall_speedup:.3f}x "
              f"({'B nhanh hơn' if self.overall_speedup > 1 else 'A nhanh hơn'})")
        print(f"{SEP}")

        # Top regressions
        regressions = sorted(
            [d for d in self.layer_diffs if d.is_regression],
            key=lambda x: x.delta_pct, reverse=True
        )[:5]
        if regressions:
            print("  Top hồi quy (B chậm hơn A):")
            for d in regressions:
                print(f"    ❌ {d.layer_name}: +{d.delta_pct:.1f}% "
                      f"({d.mean_time_a_ms:.3f}ms → {d.mean_time_b_ms:.3f}ms)")

        # Top improvements
        improvements = sorted(
            [d for d in self.layer_diffs if d.is_improvement],
            key=lambda x: x.delta_pct
        )[:5]
        if improvements:
            print("  Top cải thiện (B nhanh hơn A):")
            for d in improvements:
                print(f"    ✅ {d.layer_name}: {d.delta_pct:.1f}% "
                      f"({d.mean_time_a_ms:.3f}ms → {d.mean_time_b_ms:.3f}ms)")
        print(f"{SEP}\n")


class ABComparisonEngine:
    """
    So sánh thống kê hai phiên profiling A và B.

    Thuật toán:
        - Với mỗi layer chung giữa A và B:
            * Tính delta thời gian và phần trăm thay đổi.
            * Mann-Whitney U test kiểm định ý nghĩa thống kê (p < 0.05).
            * Phân loại: regression / improvement / unchanged.
        - Tổng hợp speedup tổng thể = mean_time_A / mean_time_B.

    Cách dùng:
        engine = ABComparisonEngine(
            df_a=df_session_a,
            df_b=df_session_b,
            name_a="ResNet50 Baseline",
            name_b="ResNet50 + torch.compile",
            regression_threshold_pct=5.0,
            improvement_threshold_pct=5.0,
        )
        report = engine.compare()
        report.print_summary()
        report.save_json("reports/ab_comparison.json")
    """

    def __init__(
        self,
        df_a: Union[str, Path, pd.DataFrame],
        df_b: Union[str, Path, pd.DataFrame],
        name_a: str = "Session A",
        name_b: str = "Session B",
        regression_threshold_pct:  float = 5.0,
        improvement_threshold_pct: float = 5.0,
        alpha: float = 0.05,
    ):
        """
        Args:
            df_a, df_b               : DataFrames hoặc đường dẫn CSV của session A và B.
            name_a, name_b           : Tên hiển thị.
            regression_threshold_pct : Ngưỡng % để coi là regression (B chậm hơn A).
            improvement_threshold_pct: Ngưỡng % để coi là improvement (B nhanh hơn A).
            alpha                    : Mức ý nghĩa thống kê cho Mann-Whitney test.
        """
        self.df_a   = _load_df(df_a)
        self.df_b   = _load_df(df_b)
        self.name_a = name_a
        self.name_b = name_b
        self.reg_pct = regression_threshold_pct
        self.imp_pct = improvement_threshold_pct
        self.alpha   = alpha

    def _aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tổng hợp DataFrame theo layer: mean + std + list of values."""
        agg = df.groupby("layer_name").agg(
            mean_time        = ("forward_time_ms",   "mean"),
            std_time         = ("forward_time_ms",   "std"),
            mean_mem_delta   = ("gpu_mem_delta_mb",  "mean"),
            param_count      = ("param_count",       "first"),
        ).fillna(0)
        # Lưu list raw values để test thống kê
        raw = df.groupby("layer_name")["forward_time_ms"].apply(list)
        agg["raw_times"] = raw
        return agg

    def _mann_whitney(
        self,
        times_a: List[float],
        times_b: List[float],
    ) -> Tuple[float, bool]:
        """
        Mann-Whitney U test (non-parametric, phù hợp với mẫu nhỏ).
        Returns: (p_value, is_significant)
        """
        if len(times_a) < 3 or len(times_b) < 3:
            return 1.0, False
        try:
            _, p = scipy_stats.mannwhitneyu(times_a, times_b, alternative="two-sided")
            return float(p), p < self.alpha
        except Exception:
            return 1.0, False

    def _aggregate_by_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fallback aggregation theo layer_type (dùng khi không có layer name chung).
        Mỗi 'layer' trong kết quả là một loại layer (Conv2d, Linear, ...).
        """
        if "layer_type" not in df.columns:
            # Nếu không có layer_type, dùng 1 nhóm duy nhất "ALL"
            df = df.copy()
            df["layer_type"] = "ALL"

        agg = df.groupby("layer_type").agg(
            mean_time       =("forward_time_ms",  "mean"),
            std_time        =("forward_time_ms",  "std"),
            mean_mem_delta  =("gpu_mem_delta_mb", "mean"),
            param_count     =("param_count",       "first"),
        ).fillna(0)
        raw = df.groupby("layer_type")["forward_time_ms"].apply(list)
        agg["raw_times"] = raw
        return agg

    def compare(self) -> ABReport:
        """
        Thực hiện so sánh A/B và trả về ABReport.

        Chế độ so sánh:
          - same_arch  : có layer name chung — so sánh từng layer (chính xác nhất).
          - cross_arch : không có layer chung — tự động fallback sang so sánh
                         theo layer_type (Conv2d, Linear, ...) rồi tổng thể.
        """
        if "forward_time_ms" not in self.df_a.columns or \
           "forward_time_ms" not in self.df_b.columns:
            raise ValueError("DataFrames phải chứa cột 'forward_time_ms'.")

        agg_a = self._aggregate(self.df_a)
        agg_b = self._aggregate(self.df_b)

        common_layers = set(agg_a.index) & set(agg_b.index)

        # ── Fallback: cross-architecture comparison theo layer_type ─────────────
        if not common_layers:
            logger.info(
                f"A/B '{self.name_a}' vs '{self.name_b}': không có layer name chung. "
                "Fallback sang so sánh theo layer_type (cross-architecture mode)."
            )
            return self._compare_cross_arch(agg_a, agg_b)

        # ── Same-architecture: so sánh từng layer ───────────────────────────
        layer_diffs:  List[LayerDiff] = []
        n_regression  = 0
        n_improvement = 0
        n_unchanged   = 0

        all_times_a = self.df_a["forward_time_ms"].values
        all_times_b = self.df_b["forward_time_ms"].values

        for layer in sorted(common_layers):
            row_a = agg_a.loc[layer]
            row_b = agg_b.loc[layer]

            mean_a = float(row_a["mean_time"])
            mean_b = float(row_b["mean_time"])

            delta_ms  = mean_b - mean_a
            delta_pct = (delta_ms / mean_a * 100.0) if mean_a > 0 else 0.0

            mem_a    = float(row_a.get("mean_mem_delta", 0))
            mem_b    = float(row_b.get("mean_mem_delta", 0))
            mem_diff = mem_b - mem_a

            times_a_raw = row_a.get("raw_times", [])
            times_b_raw = row_b.get("raw_times", [])
            p_val, is_sig = self._mann_whitney(times_a_raw, times_b_raw)

            is_regression  = delta_pct > self.reg_pct  and is_sig
            is_improvement = delta_pct < -self.imp_pct and is_sig

            if is_regression:
                n_regression += 1
            elif is_improvement:
                n_improvement += 1
            else:
                n_unchanged += 1

            layer_diffs.append(LayerDiff(
                layer_name        = layer,
                mean_time_a_ms    = round(mean_a, 4),
                mean_time_b_ms    = round(mean_b, 4),
                delta_ms          = round(delta_ms, 4),
                delta_pct         = round(delta_pct, 2),
                mem_delta_a_mb    = round(mem_a, 4),
                mem_delta_b_mb    = round(mem_b, 4),
                mem_delta_diff_mb = round(mem_diff, 4),
                is_regression     = is_regression,
                is_improvement    = is_improvement,
                statistical_sig   = is_sig,
                p_value           = round(p_val, 6),
            ))

        mean_total_a    = float(np.mean(all_times_a)) if len(all_times_a) > 0 else 1.0
        mean_total_b    = float(np.mean(all_times_b)) if len(all_times_b) > 0 else 1.0
        overall_speedup = mean_total_a / mean_total_b if mean_total_b > 0 else 1.0

        p_overall, sig_overall = self._mann_whitney(
            list(all_times_a), list(all_times_b)
        )

        summary = {
            "comparison_mode":  "same_arch",
            "common_layers":    len(common_layers),
            "only_in_a":        len(set(agg_a.index) - common_layers),
            "only_in_b":        len(set(agg_b.index) - common_layers),
            "overall_speedup":  round(overall_speedup, 4),
            "overall_p_value":  round(p_overall, 6),
            "overall_significant": sig_overall,
            "mean_time_a_ms":   round(mean_total_a, 4),
            "mean_time_b_ms":   round(mean_total_b, 4),
            "alpha":            self.alpha,
            "regression_threshold_pct":  self.reg_pct,
            "improvement_threshold_pct": self.imp_pct,
        }

        report = ABReport(
            session_a_name  = self.name_a,
            session_b_name  = self.name_b,
            total_layers    = len(common_layers),
            regressions     = n_regression,
            improvements    = n_improvement,
            unchanged       = n_unchanged,
            overall_speedup = round(overall_speedup, 4),
            layer_diffs     = layer_diffs,
            summary         = summary,
        )

        logger.info(
            f"A/B (same_arch) '{self.name_a}' vs '{self.name_b}': "
            f"speedup={overall_speedup:.3f}x | "
            f"improvement={n_improvement} | regression={n_regression} | "
            f"p={p_overall:.4f} ({'sig' if sig_overall else 'not sig'})"
        )
        return report

    def _compare_cross_arch(self, agg_a: pd.DataFrame, agg_b: pd.DataFrame) -> ABReport:
        """
        Fallback comparison khi 2 kiến trúc khác nhau hoàn toàn.
        So sánh theo nhóm layer_type (Conv2d, Linear, LayerNorm, ...).
        Thêm 1 nhóm đặc biệt '__total__' đại diện toàn bộ mô hình.
        """
        agg_type_a = self._aggregate_by_type(self.df_a)
        agg_type_b = self._aggregate_by_type(self.df_b)

        # Các loại layer mà ít nhất 1 trong 2 model có
        all_types = sorted(set(agg_type_a.index) | set(agg_type_b.index))

        layer_diffs:  List[LayerDiff] = []
        n_regression  = 0
        n_improvement = 0
        n_unchanged   = 0

        for ltype in all_types:
            has_a = ltype in agg_type_a.index
            has_b = ltype in agg_type_b.index

            mean_a = float(agg_type_a.loc[ltype, "mean_time"]) if has_a else 0.0
            mean_b = float(agg_type_b.loc[ltype, "mean_time"]) if has_b else 0.0
            mem_a  = float(agg_type_a.loc[ltype, "mean_mem_delta"]) if has_a else 0.0
            mem_b  = float(agg_type_b.loc[ltype, "mean_mem_delta"]) if has_b else 0.0

            delta_ms  = mean_b - mean_a
            delta_pct = (delta_ms / mean_a * 100.0) if mean_a > 0 else 0.0

            times_a_raw = list(agg_type_a.loc[ltype, "raw_times"]) if has_a else []
            times_b_raw = list(agg_type_b.loc[ltype, "raw_times"]) if has_b else []
            p_val, is_sig = self._mann_whitney(times_a_raw, times_b_raw)

            is_regression  = delta_pct > self.reg_pct  and is_sig
            is_improvement = delta_pct < -self.imp_pct and is_sig

            if is_regression:
                n_regression += 1
            elif is_improvement:
                n_improvement += 1
            else:
                n_unchanged += 1

            layer_diffs.append(LayerDiff(
                layer_name        = f"[type] {ltype}",
                mean_time_a_ms    = round(mean_a, 4),
                mean_time_b_ms    = round(mean_b, 4),
                delta_ms          = round(delta_ms, 4),
                delta_pct         = round(delta_pct, 2),
                mem_delta_a_mb    = round(mem_a, 4),
                mem_delta_b_mb    = round(mem_b, 4),
                mem_delta_diff_mb = round(mem_b - mem_a, 4),
                is_regression     = is_regression,
                is_improvement    = is_improvement,
                statistical_sig   = is_sig,
                p_value           = round(p_val, 6),
            ))

        # Nhóm tổng thể '__total__' — so sánh mean forward time toàn mô hình
        all_a = self.df_a["forward_time_ms"].values
        all_b = self.df_b["forward_time_ms"].values
        mean_total_a    = float(np.mean(all_a)) if len(all_a) > 0 else 1.0
        mean_total_b    = float(np.mean(all_b)) if len(all_b) > 0 else 1.0
        overall_speedup = mean_total_a / mean_total_b if mean_total_b > 0 else 1.0

        p_overall, sig_overall = self._mann_whitney(list(all_a), list(all_b))

        # Thêm dòng tổng thể
        delta_total_ms  = mean_total_b - mean_total_a
        delta_total_pct = (delta_total_ms / mean_total_a * 100.0) if mean_total_a > 0 else 0.0
        is_total_reg = delta_total_pct > self.reg_pct  and sig_overall
        is_total_imp = delta_total_pct < -self.imp_pct and sig_overall
        layer_diffs.append(LayerDiff(
            layer_name        = "[TOTAL]",
            mean_time_a_ms    = round(mean_total_a, 4),
            mean_time_b_ms    = round(mean_total_b, 4),
            delta_ms          = round(delta_total_ms, 4),
            delta_pct         = round(delta_total_pct, 2),
            mem_delta_a_mb    = 0.0,
            mem_delta_b_mb    = 0.0,
            mem_delta_diff_mb = 0.0,
            is_regression     = is_total_reg,
            is_improvement    = is_total_imp,
            statistical_sig   = sig_overall,
            p_value           = round(p_overall, 6),
        ))

        summary = {
            "comparison_mode":       "cross_arch",
            "note":                  "Layer names differ; compared by layer_type groups.",
            "layer_types_in_a":      len(agg_type_a),
            "layer_types_in_b":      len(agg_type_b),
            "common_types":          len(set(agg_type_a.index) & set(agg_type_b.index)),
            "overall_speedup":       round(overall_speedup, 4),
            "overall_p_value":       round(p_overall, 6),
            "overall_significant":   sig_overall,
            "mean_time_a_ms":        round(mean_total_a, 4),
            "mean_time_b_ms":        round(mean_total_b, 4),
            "alpha":                 self.alpha,
            "regression_threshold_pct":  self.reg_pct,
            "improvement_threshold_pct": self.imp_pct,
        }

        report = ABReport(
            session_a_name  = self.name_a,
            session_b_name  = self.name_b,
            total_layers    = len(layer_diffs),
            regressions     = n_regression,
            improvements    = n_improvement,
            unchanged       = n_unchanged,
            overall_speedup = round(overall_speedup, 4),
            layer_diffs     = layer_diffs,
            summary         = summary,
        )

        logger.info(
            f"A/B (cross_arch) '{self.name_a}' vs '{self.name_b}': "
            f"speedup={overall_speedup:.3f}x | types_compared={len(all_types)} | "
            f"p={p_overall:.4f} ({'sig' if sig_overall else 'not sig'})"
        )
        return report


# ─────────────────────────────────────────────────────────────────────────────
# Convenience API
# ─────────────────────────────────────────────────────────────────────────────

def analyze_from_csv(
    csv_path:         Union[str, Path],
    model_name:       str   = "model",
    device:           str   = "cpu",
    total_gpu_mem_mb: float = 0.0,
    total_params:     int   = 0,
    dataloader_json:  Optional[Union[str, Path]] = None,
    output_dir:       Union[str, Path] = "reports",
    thresholds:       Optional[Dict[str, float]] = None,
) -> AnalysisReport:
    """
    Convenience function: đọc CSV → chạy toàn bộ Rule Catalog → lưu báo cáo.

    Args:
        csv_path        : Đường dẫn CSV chứa layer records.
        model_name      : Tên mô hình.
        device          : 'cpu' hoặc 'cuda'.
        total_gpu_mem_mb: Tổng GPU memory MB.
        total_params    : Tổng params.
        dataloader_json : (Tùy chọn) Đường dẫn JSON của DataLoaderSentinel.save_report().
        output_dir      : Thư mục lưu báo cáo.
        thresholds      : Override thresholds tùy chỉnh.

    Returns:
        AnalysisReport
    """
    df = pd.read_csv(csv_path)

    # Nạp dataloader stats nếu có
    dl_stats = None
    if dataloader_json:
        dl_path = Path(dataloader_json)
        if dl_path.exists():
            with open(dl_path, "r", encoding="utf-8") as f:
                dl_data = json.load(f)
            dl_stats = dl_data.get("summary", {})
            # Thêm batch_load_times nếu có
            batch_detail = dl_data.get("batch_detail", [])
            dl_stats["batch_load_times"] = [b.get("load_time_ms", 0) for b in batch_detail]

    catalog = RuleCatalog(thresholds=thresholds)
    report  = catalog.run_all(
        df=df,
        model_name=model_name,
        device=device,
        total_gpu_mem_mb=total_gpu_mem_mb,
        total_params=total_params,
        dataloader_stats=dl_stats,
    )

    out = Path(output_dir) / model_name
    out.mkdir(parents=True, exist_ok=True)
    report.save_json(out / "analysis_report.json")
    report.save_csv(out  / "analysis_report.csv")
    report.print_summary()
    return report


def compare_ab_from_csv(
    csv_a:      Union[str, Path],
    csv_b:      Union[str, Path],
    name_a:     str   = "Session A",
    name_b:     str   = "Session B",
    output_dir: Union[str, Path] = "reports",
    reg_pct:    float = 5.0,
    imp_pct:    float = 5.0,
    alpha:      float = 0.05,
) -> ABReport:
    """
    Convenience function: đọc 2 CSV → so sánh A/B → lưu báo cáo.

    Returns:
        ABReport
    """
    engine = ABComparisonEngine(
        df_a=csv_a, df_b=csv_b,
        name_a=name_a, name_b=name_b,
        regression_threshold_pct=reg_pct,
        improvement_threshold_pct=imp_pct,
        alpha=alpha,
    )
    report = engine.compare()
    report.print_summary()

    out = Path(output_dir) / "ab_comparison"
    out.mkdir(parents=True, exist_ok=True)
    report.save_json(out / f"{name_a}_vs_{name_b}.json".replace(" ", "_"))
    report.save_csv(out  / f"{name_a}_vs_{name_b}.csv".replace(" ", "_"))
    return report
