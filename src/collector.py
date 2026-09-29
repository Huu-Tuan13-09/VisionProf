"""
collector.py — Module thu thập dữ liệu profiling tầng (layer-level)
=======================================================================
Thành phần:
  - LayerProfiler      : Hook-based profiler, ghi nhận thời gian + bộ nhớ từng layer
  - MultiModelProfiler : Điều phối profiling nhiều mô hình, lưu JSON/CSV
  - DataLoaderSentinel : Sentinel wrapper đo lường I/O vs Compute time
  - calibrate_overhead : Hiệu chỉnh độ trễ do hook gây ra

Tương thích: Windows CPU + Google Colab GPU T4
Lưu trữ: JSON + CSV (pandas) — không dùng SQLite
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
import threading
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
from tqdm import tqdm

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("collector")


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder an toàn cho numpy.bool_, int64, float64, ndarray."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# ── pynvml — fallback an toàn nếu không có GPU / pynvml ─────────────────────
_PYNVML_OK = False
_nvml_handle = None

try:
    import pynvml
    pynvml.nvmlInit()
    _PYNVML_OK = True
    logger.info("pynvml khởi tạo thành công.")
except Exception as _e:
    warnings.warn(
        f"pynvml không khả dụng ({_e}). "
        "Sẽ dùng torch.cuda.memory_stats() + psutil làm fallback.",
        RuntimeWarning,
        stacklevel=2,
    )


def _get_nvml_handle(device_index: int = 0) -> Optional[Any]:
    """Lấy NVML handle, trả về None nếu không thể."""
    global _nvml_handle
    if not _PYNVML_OK:
        return None
    try:
        if _nvml_handle is None:
            _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        return _nvml_handle
    except Exception:
        return None


def _gpu_memory_bytes(device_index: int = 0) -> Tuple[int, int]:
    """
    Trả về (used_bytes, total_bytes) GPU memory.
    Fallback: torch.cuda.memory_stats → psutil RAM
    """
    handle = _get_nvml_handle(device_index)
    if handle is not None:
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return int(info.used), int(info.total)
        except Exception:
            pass

    # Fallback 1: torch.cuda
    if torch.cuda.is_available():
        try:
            stats = torch.cuda.memory_stats(device_index)
            used  = stats.get("allocated_bytes.all.current", 0)
            total = torch.cuda.get_device_properties(device_index).total_memory
            return int(used), int(total)
        except Exception:
            pass

    # Fallback 2: psutil RAM (cho CPU-only)
    vm = psutil.virtual_memory()
    return int(vm.used), int(vm.total)


def _gpu_utilization_pct(device_index: int = 0) -> float:
    """Trả về % GPU utilization. Fallback: CPU percent."""
    handle = _get_nvml_handle(device_index)
    if handle is not None:
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(util.gpu)
        except Exception:
            pass
    return float(psutil.cpu_percent(interval=None))


def _extract_tensor_from_output(output: Any) -> Optional[torch.Tensor]:
    """
    Trích xuất tensor đầu tiên từ output của layer.
    Xử lý an toàn: Tensor, tuple, list, dict (YOLOv8, ViT, ...).
    """
    if output is None:
        return None
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            result = _extract_tensor_from_output(item)
            if result is not None:
                return result
        return None
    if isinstance(output, dict):
        for v in output.values():
            result = _extract_tensor_from_output(v)
            if result is not None:
                return result
        return None
    # Đối với các object đặc biệt (ultralytics Results, ...)
    if hasattr(output, "pred"):
        return _extract_tensor_from_output(output.pred)
    if hasattr(output, "logits"):
        return output.logits
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LayerRecord:
    """Bản ghi thống kê cho một layer tại một lần forward pass."""
    layer_name:        str
    layer_type:        str
    input_shape:       List[int]
    output_shape:      List[int]
    param_count:       int
    forward_time_ms:   float   = 0.0
    gpu_mem_before_mb: float   = 0.0
    gpu_mem_after_mb:  float   = 0.0
    gpu_mem_delta_mb:  float   = 0.0
    gpu_util_pct:      float   = 0.0
    iteration:         int     = 0
    device:            str     = "cpu"


@dataclass
class ProfileSession:
    """Tổng hợp kết quả của một lần profiling toàn bộ mô hình."""
    model_name:    str
    device:        str
    total_time_ms: float
    peak_mem_mb:   float
    records:       List[LayerRecord] = field(default_factory=list)
    metadata:      Dict[str, Any]    = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LayerProfiler
# ─────────────────────────────────────────────────────────────────────────────

class LayerProfiler:
    """
    Hook-based profiler — gắn vào từng layer của mô hình PyTorch.

    Cách dùng:
        profiler = LayerProfiler(model, device="cuda")
        with profiler.profile_context(iteration=0):
            output = model(input_tensor)
        records = profiler.get_records()
    """

    def __init__(
        self,
        model: nn.Module,
        device: Union[str, torch.device] = "cpu",
        gpu_device_index: int = 0,
        exclude_types: Optional[Tuple[type, ...]] = None,
        min_param_threshold: int = 0,
    ):
        """
        Args:
            model              : Mô hình PyTorch cần profile.
            device             : 'cpu' hoặc 'cuda'.
            gpu_device_index   : Chỉ số GPU (mặc định 0).
            exclude_types      : Tuple các loại layer bỏ qua (vd: (nn.ReLU,)).
            min_param_threshold: Chỉ profile layer có số params >= ngưỡng này.
        """
        self.model           = model
        self.device          = str(device)
        self.gpu_idx         = gpu_device_index
        self.exclude_types   = exclude_types or (nn.ReLU, nn.Dropout, nn.Identity)
        self.min_param_count = min_param_threshold
        self._records:     List[LayerRecord] = []
        self._hooks:       List[Any]         = []
        self._overhead_ms: float             = 0.0
        self._current_iter: int              = 0
        self._rss_start_mb: float            = 0.0   # RSS của process ở đầu mỗi iter (CPU)
        self._lock = threading.Lock()

    # ── Overhead calibration ────────────────────────────────────────────────

    def set_overhead(self, overhead_ms: float) -> None:
        """Đặt giá trị overhead đã hiệu chỉnh (từ calibrate_overhead)."""
        self._overhead_ms = overhead_ms

    # ── Hook helpers ────────────────────────────────────────────────────────

    def _make_hooks(self, name: str, module: nn.Module) -> Tuple[Callable, Callable]:
        """
        Tạo cặp pre-hook và post-hook cho một layer.

        Chiến lược đo memory theo device:
          - CUDA : dùng torch.cuda.memory_allocated() — cheap, không có syscall.
          - CPU  : tính activation memory trực tiếp từ output tensor
                   (nelement * element_size / MB). Không gọi psutil bên trong
                   hook để tránh overhead ~8ms/layer trên Windows.
                   RSS của process được đo ở cấp profile_context (một lần/iter),
                   sau đó chia đều cho số layers làm giá trị mem_before.
        """
        state: Dict[str, Any] = {}
        is_gpu = self.device.startswith("cuda")

        def pre_hook(mod: nn.Module, inputs: Any) -> None:
            state["t_start"] = time.perf_counter()
            if is_gpu:
                # Cheap CUDA call — không blocking
                state["mem_before_mb"] = torch.cuda.memory_allocated(self.gpu_idx) / 1024**2
                state["gpu_util"]      = _gpu_utilization_pct(self.gpu_idx)
            else:
                # CPU: dùng RSS đã đo ở profile_context, không gọi psutil ở đây
                state["mem_before_mb"] = self._rss_start_mb
                state["gpu_util"]      = 0.0
            # Ghi input shape an toàn
            inp_tensor = _extract_tensor_from_output(inputs)
            state["input_shape"] = list(inp_tensor.shape) if inp_tensor is not None else []

        def post_hook(mod: nn.Module, inputs: Any, output: Any) -> None:
            t_end   = time.perf_counter()
            elapsed = (t_end - state.get("t_start", t_end)) * 1000.0
            elapsed = max(0.0, elapsed - self._overhead_ms)

            out_tensor   = _extract_tensor_from_output(output)
            output_shape = list(out_tensor.shape) if out_tensor is not None else []
            param_count  = sum(p.numel() for p in mod.parameters())

            if is_gpu:
                mem_after = torch.cuda.memory_allocated(self.gpu_idx) / 1024**2
            else:
                # Tính activation memory từ output tensor — zero syscall overhead
                if out_tensor is not None:
                    act_mb    = out_tensor.nelement() * out_tensor.element_size() / 1024**2
                else:
                    act_mb    = 0.0
                mem_after = state.get("mem_before_mb", 0.0) + act_mb

            mem_before = state.get("mem_before_mb", 0.0)

            record = LayerRecord(
                layer_name        = name,
                layer_type        = type(mod).__name__,
                input_shape       = state.get("input_shape", []),
                output_shape      = output_shape,
                param_count       = param_count,
                forward_time_ms   = round(elapsed, 4),
                gpu_mem_before_mb = round(mem_before, 3),
                gpu_mem_after_mb  = round(mem_after, 3),
                gpu_mem_delta_mb  = round(mem_after - mem_before, 3),
                gpu_util_pct      = round(state.get("gpu_util", 0.0), 2),
                iteration         = self._current_iter,
                device            = self.device,
            )
            with self._lock:
                self._records.append(record)

        return pre_hook, post_hook

    # ── Hook management ─────────────────────────────────────────────────────

    def attach_hooks(self) -> None:
        """Gắn hooks vào tất cả các layer thoả mãn điều kiện."""
        self.remove_hooks()
        for name, module in self.model.named_modules():
            if isinstance(module, self.exclude_types):
                continue
            param_count = sum(p.numel() for p in module.parameters())
            if param_count < self.min_param_count:
                continue
            # Chỉ gắn vào leaf modules để tránh double-counting
            child_count = sum(1 for _ in module.children())
            if child_count > 0:
                continue
            pre_h, post_h = self._make_hooks(name, module)
            h_pre  = module.register_forward_pre_hook(pre_h)
            h_post = module.register_forward_hook(post_h)
            self._hooks.extend([h_pre, h_post])
        logger.info(f"Đã gắn {len(self._hooks) // 2} layer hooks.")

    def remove_hooks(self) -> None:
        """Gỡ bỏ tất cả hooks đã gắn."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ── Context manager ─────────────────────────────────────────────────────

    @contextmanager
    def profile_context(self, iteration: int = 0) -> Generator[None, None, None]:
        """
        Context manager để profile một lần forward pass.

        Ví dụ:
            with profiler.profile_context(iteration=i):
                output = model(x)
        """
        self._current_iter = iteration

        # Đo RSS của process một lần duy nhất trước khi attach hooks (CPU-safe)
        # Tránh gọi psutil bên trong từng layer hook
        if not self.device.startswith("cuda"):
            try:
                self._rss_start_mb = psutil.Process().memory_info().rss / 1024**2
            except Exception:
                self._rss_start_mb = 0.0

        self.attach_hooks()
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        try:
            yield
        finally:
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            self.remove_hooks()

            # Đo RSS cuối — dùng để báo cáo peak RSS trong DataLoaderSentinel
            if not self.device.startswith("cuda"):
                try:
                    rss_end = psutil.Process().memory_info().rss / 1024**2
                    logger.debug(
                        f"[profile_context iter={iteration}] "
                        f"Process RSS: {self._rss_start_mb:.1f} -> {rss_end:.1f} MB "
                        f"(delta={rss_end - self._rss_start_mb:+.1f} MB)"
                    )
                except Exception:
                    pass

    # ── Data access ─────────────────────────────────────────────────────────

    def get_records(self) -> List[LayerRecord]:
        """Trả về danh sách bản ghi."""
        return list(self._records)

    def to_dataframe(self) -> pd.DataFrame:
        """Chuyển đổi records thành DataFrame."""
        if not self._records:
            return pd.DataFrame()
        return pd.DataFrame([asdict(r) for r in self._records])

    def clear_records(self) -> None:
        """Xoá toàn bộ records."""
        with self._lock:
            self._records.clear()

    def save_csv(self, path: Union[str, Path]) -> Path:
        """Lưu records ra file CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        logger.info(f"Đã lưu {len(df)} records → {path}")
        return path

    def save_json(self, path: Union[str, Path]) -> Path:
        """Lưu records ra file JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(r) for r in self._records]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
        logger.info(f"Da luu {len(data)} records -> {path}")
        return path


# ─────────────────────────────────────────────────────────────────────────────
# 2. calibrate_overhead
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_overhead(
    model: nn.Module,
    input_tensor: torch.Tensor,
    n_warmup: int = 5,
    n_measure: int = 20,
    device: str = "cpu",
) -> float:
    """
    Đo overhead trung bình (ms) mà hooks gây ra cho mỗi layer.

    Thuật toán:
        - Chạy forward pass CÓ hooks và KHÔNG CÓ hooks.
        - Overhead = (T_with_hooks - T_without_hooks) / số_layer_đo.

    Returns:
        overhead_ms (float): Overhead trung bình mỗi layer (ms).
    """
    device_obj = torch.device(device)
    model = model.to(device_obj).eval()
    input_tensor = input_tensor.to(device_obj)

    def _run_timed(attach: bool, n_runs: int) -> float:
        profiler = LayerProfiler(model, device=device)
        if attach:
            profiler.attach_hooks()
        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _extract_tensor_from_output(model(input_tensor))
                if device == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000.0)
        if attach:
            profiler.remove_hooks()
        return float(np.median(times))

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _extract_tensor_from_output(model(input_tensor))

    t_no_hook  = _run_timed(attach=False, n_runs=n_measure)
    t_with_hook = _run_timed(attach=True,  n_runs=n_measure)

    # Tính số layer đo được
    n_layers = sum(
        1 for _, m in model.named_modules()
        if sum(1 for _ in m.children()) == 0 and sum(p.numel() for p in m.parameters()) > 0
    )
    n_layers = max(n_layers, 1)

    overhead = max(0.0, (t_with_hook - t_no_hook) / n_layers)
    logger.info(
        f"[calibrate] no_hook={t_no_hook:.3f}ms | "
        f"with_hook={t_with_hook:.3f}ms | "
        f"n_layers={n_layers} | overhead/layer={overhead:.4f}ms"
    )
    return overhead


# ─────────────────────────────────────────────────────────────────────────────
# 3. DataLoaderSentinel
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataLoaderStats:
    """Thống kê hiệu năng DataLoader."""
    total_batches:       int   = 0
    total_load_time_ms:  float = 0.0
    total_compute_time_ms: float = 0.0
    mean_load_time_ms:   float = 0.0
    mean_compute_time_ms: float = 0.0
    std_load_time_ms:    float = 0.0
    std_compute_time_ms: float = 0.0
    io_bottleneck_ratio: float = 0.0   # load_time / (load_time + compute_time)
    peak_ram_mb:         float = 0.0
    peak_gpu_mem_mb:     float = 0.0
    worker_count:        int   = 0
    batch_size:          int   = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DataLoaderSentinel:
    """
    Wrapper đo lường thời gian tải dữ liệu (I/O) vs tính toán (Compute).

    Cách dùng:
        sentinel = DataLoaderSentinel(dataloader, device="cuda")
        for batch in sentinel:
            output = model(batch[0])
            sentinel.mark_compute_end()
        stats = sentinel.get_stats()
        sentinel.save_report("reports/dataloader_stats.json")
    """

    def __init__(
        self,
        dataloader: Iterable,
        device: str = "cpu",
        gpu_device_index: int = 0,
        description: str = "DataLoader",
    ):
        self._loader      = dataloader
        self._device      = device
        self._gpu_idx     = gpu_device_index
        self._description = description

        self._load_times:    List[float] = []
        self._compute_times: List[float] = []
        self._peak_ram_mb:   float       = 0.0
        self._peak_gpu_mb:   float       = 0.0
        self._compute_start: Optional[float] = None
        self._prev_batch_end: Optional[float] = None

        # Đọc metadata từ DataLoader nếu có
        self._batch_size   = getattr(dataloader, "batch_size", 0) or 0
        self._worker_count = getattr(dataloader, "num_workers", 0) or 0

    def __iter__(self) -> Generator[Any, None, None]:
        self._prev_batch_end = None
        for batch in self._loader:
            t_batch_ready = time.perf_counter()

            # Load time = thời gian từ lúc kết thúc compute batch trước đến khi batch này sẵn sàng
            if self._prev_batch_end is not None:
                load_ms = (t_batch_ready - self._prev_batch_end) * 1000.0
                self._load_times.append(load_ms)

            # Theo dõi RAM / GPU mem
            ram_used  = psutil.virtual_memory().used / 1024**2
            self._peak_ram_mb = max(self._peak_ram_mb, ram_used)

            gpu_used, _ = _gpu_memory_bytes(self._gpu_idx)
            gpu_mb = gpu_used / 1024**2
            self._peak_gpu_mb = max(self._peak_gpu_mb, gpu_mb)

            # Đặt điểm bắt đầu compute
            self._compute_start   = t_batch_ready
            self._prev_batch_end  = None   # Sẽ cập nhật khi gọi mark_compute_end()
            yield batch

    def mark_compute_end(self) -> None:
        """Gọi SAU khi hoàn thành tính toán cho một batch (vd: sau loss.backward())."""
        t_now = time.perf_counter()
        if self._compute_start is not None:
            compute_ms = (t_now - self._compute_start) * 1000.0
            self._compute_times.append(compute_ms)
            self._prev_batch_end = t_now
            self._compute_start  = None

    def get_stats(self) -> DataLoaderStats:
        """Tính toán và trả về thống kê tổng hợp."""
        load_arr    = np.array(self._load_times,    dtype=float)
        compute_arr = np.array(self._compute_times, dtype=float)

        total_load    = float(load_arr.sum())    if len(load_arr)    > 0 else 0.0
        total_compute = float(compute_arr.sum()) if len(compute_arr) > 0 else 0.0
        total_time    = total_load + total_compute
        ratio         = total_load / total_time if total_time > 0 else 0.0

        return DataLoaderStats(
            total_batches         = len(self._compute_times),
            total_load_time_ms    = round(total_load, 3),
            total_compute_time_ms = round(total_compute, 3),
            mean_load_time_ms     = round(float(load_arr.mean()),    4) if len(load_arr)    > 0 else 0.0,
            mean_compute_time_ms  = round(float(compute_arr.mean()), 4) if len(compute_arr) > 0 else 0.0,
            std_load_time_ms      = round(float(load_arr.std()),     4) if len(load_arr)    > 0 else 0.0,
            std_compute_time_ms   = round(float(compute_arr.std()),  4) if len(compute_arr) > 0 else 0.0,
            io_bottleneck_ratio   = round(ratio, 4),
            peak_ram_mb           = round(self._peak_ram_mb,  2),
            peak_gpu_mem_mb       = round(self._peak_gpu_mb,  2),
            worker_count          = self._worker_count,
            batch_size            = self._batch_size,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Trả về DataFrame batch-by-batch."""
        n = min(len(self._load_times), len(self._compute_times))
        if n == 0:
            return pd.DataFrame()
        data = {
            "batch_index":       list(range(n)),
            "load_time_ms":      self._load_times[:n],
            "compute_time_ms":   self._compute_times[:n],
        }
        return pd.DataFrame(data)

    def save_report(self, path: Union[str, Path]) -> Path:
        """Lưu báo cáo DataLoader stats ra JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stats = self.get_stats()
        payload = {
            "description":   self._description,
            "summary":       stats.to_dict(),
            "batch_detail":  self.to_dataframe().to_dict(orient="records"),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
        logger.info(f"DataLoader report -> {path}")
        return path

    def save_csv(self, path: Union[str, Path]) -> Path:
        """Lưu batch-by-batch ra CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        logger.info(f"DataLoader CSV → {path}")
        return path


# ─────────────────────────────────────────────────────────────────────────────
# 4. MultiModelProfiler
# ─────────────────────────────────────────────────────────────────────────────

class MultiModelProfiler:
    """
    Điều phối profiling nhiều mô hình PyTorch trong một phiên thực nghiệm.

    Tính năng:
        - Profile N mô hình với M iteration mỗi mô hình.
        - Tự động hiệu chỉnh overhead (calibrate_overhead).
        - Lưu checkpoint sau mỗi mô hình (JSON + CSV).
        - Hỗ trợ warmup iterations.
        - Tổng hợp kết quả thành một DataFrame duy nhất.

    Cách dùng:
        mmp = MultiModelProfiler(
            models={"resnet": resnet50, "vit": vit_b16},
            input_factory=lambda name: torch.randn(1, 3, 224, 224),
            device="cuda",
            save_dir="data/profiles",
        )
        mmp.run(n_iterations=10, n_warmup=3)
        df = mmp.get_summary_dataframe()
    """

    def __init__(
        self,
        models:          Dict[str, nn.Module],
        input_factory:   Callable[[str], torch.Tensor],
        device:          str = "cpu",
        gpu_device_index: int = 0,
        save_dir:        Union[str, Path] = "data/profiles",
        calibrate:       bool = True,
        exclude_types:   Optional[Tuple[type, ...]] = None,
        min_param_threshold: int = 0,
    ):
        """
        Args:
            models           : Dict tên → nn.Module.
            input_factory    : Hàm nhận tên mô hình, trả về input tensor.
            device           : 'cpu' hoặc 'cuda'.
            gpu_device_index : Chỉ số GPU.
            save_dir         : Thư mục lưu kết quả.
            calibrate        : Có thực hiện hiệu chỉnh overhead không.
            exclude_types    : Layer types bỏ qua khi profile.
            min_param_threshold: Ngưỡng params tối thiểu.
        """
        self.models           = models
        self.input_factory    = input_factory
        self.device           = device
        self.gpu_idx          = gpu_device_index
        self.save_dir         = Path(save_dir)
        self.do_calibrate     = calibrate
        self.exclude_types    = exclude_types
        self.min_param_count  = min_param_threshold
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._sessions:  Dict[str, ProfileSession] = {}
        self._overheads: Dict[str, float]          = {}

    # ── Internals ────────────────────────────────────────────────────────────

    def _profile_single(
        self,
        model_name:   str,
        model:        nn.Module,
        n_iterations: int,
        n_warmup:     int,
    ) -> ProfileSession:
        """Profile một mô hình đơn lẻ."""
        device_obj = torch.device(self.device)
        model = model.to(device_obj).eval()
        input_tensor = self.input_factory(model_name).to(device_obj)

        # Calibrate overhead
        overhead_ms = 0.0
        if self.do_calibrate:
            logger.info(f"[{model_name}] Đang hiệu chỉnh overhead hook...")
            try:
                overhead_ms = calibrate_overhead(
                    model, input_tensor,
                    n_warmup=3, n_measure=10,
                    device=self.device,
                )
                self._overheads[model_name] = overhead_ms
            except Exception as e:
                logger.warning(f"[{model_name}] calibrate_overhead lỗi: {e}. Dùng overhead=0.")

        # Warmup
        logger.info(f"[{model_name}] Warmup {n_warmup} lần...")
        with torch.no_grad():
            for _ in range(n_warmup):
                _extract_tensor_from_output(model(input_tensor))
        if self.device == "cuda":
            torch.cuda.synchronize()

        # Profiling iterations
        profiler = LayerProfiler(
            model,
            device=self.device,
            gpu_device_index=self.gpu_idx,
            exclude_types=self.exclude_types,
            min_param_threshold=self.min_param_count,
        )
        profiler.set_overhead(overhead_ms)

        t_total_start = time.perf_counter()

        with torch.no_grad():
            for i in tqdm(range(n_iterations), desc=f"Profiling {model_name}", leave=False):
                with profiler.profile_context(iteration=i):
                    _extract_tensor_from_output(model(input_tensor))

        if self.device == "cuda":
            torch.cuda.synchronize()

        total_ms = (time.perf_counter() - t_total_start) * 1000.0

        # Peak memory
        gpu_used, _ = _gpu_memory_bytes(self.gpu_idx)
        peak_mb = gpu_used / 1024**2

        session = ProfileSession(
            model_name=model_name,
            device=self.device,
            total_time_ms=round(total_ms, 3),
            peak_mem_mb=round(peak_mb, 3),
            records=profiler.get_records(),
            metadata={
                "n_iterations":  n_iterations,
                "n_warmup":      n_warmup,
                "overhead_ms":   round(overhead_ms, 4),
                "param_count":   sum(p.numel() for p in model.parameters()),
            },
        )
        return session

    def _save_session(self, session: ProfileSession) -> None:
        """Lưu checkpoint của một session."""
        base = self.save_dir / session.model_name
        base.mkdir(parents=True, exist_ok=True)

        # Lưu JSON (metadata + summary)
        summary = {
            "model_name":    session.model_name,
            "device":        session.device,
            "total_time_ms": session.total_time_ms,
            "peak_mem_mb":   session.peak_mem_mb,
            "metadata":      session.metadata,
            "n_records":     len(session.records),
        }
        with open(base / "session_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)

        # Lưu CSV (records chi tiết)
        if session.records:
            df = pd.DataFrame([asdict(r) for r in session.records])
            df.to_csv(base / "layer_records.csv", index=False)

        logger.info(f"[{session.model_name}] Checkpoint đã lưu → {base}")

    # ── Public API ───────────────────────────────────────────────────────────

    def run(self, n_iterations: int = 10, n_warmup: int = 3) -> None:
        """
        Chạy profiling tuần tự cho tất cả mô hình.

        Args:
            n_iterations: Số lần forward pass có đo lường.
            n_warmup    : Số lần forward pass làm nóng máy (không đo).
        """
        total_models = len(self.models)
        logger.info(f"Bắt đầu profiling {total_models} mô hình | device={self.device}")

        for idx, (name, model) in enumerate(self.models.items(), 1):
            logger.info(f"── [{idx}/{total_models}] Mô hình: {name} ──")
            try:
                session = self._profile_single(name, model, n_iterations, n_warmup)
                self._sessions[name] = session
                self._save_session(session)
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"[{name}] Profiling thất bại: {e}", exc_info=True)

        # Lưu báo cáo tổng hợp
        self._save_combined_report()
        logger.info("✅ Hoàn tất profiling tất cả mô hình.")

    def _save_combined_report(self) -> None:
        """Lưu báo cáo tổng hợp tất cả session ra JSON + CSV."""
        all_records = []
        summaries   = []

        for name, session in self._sessions.items():
            for r in session.records:
                d = asdict(r)
                all_records.append(d)

            summaries.append({
                "model_name":    session.model_name,
                "device":        session.device,
                "total_time_ms": session.total_time_ms,
                "peak_mem_mb":   session.peak_mem_mb,
                **session.metadata,
            })

        # Lưu tổng hợp
        combined_path = self.save_dir / "combined_summary.json"
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump({"sessions": summaries, "overheads": self._overheads}, f,
                      indent=2, cls=_NumpyEncoder)
        logger.info(f"Bao cao tong hop -> {combined_path}")

        if all_records:
            df_all = pd.DataFrame(all_records)
            csv_path = self.save_dir / "all_records.csv"
            df_all.to_csv(csv_path, index=False)
            logger.info(f"Toàn bộ records → {csv_path}")

    def get_session(self, model_name: str) -> Optional[ProfileSession]:
        """Lấy ProfileSession của một mô hình cụ thể."""
        return self._sessions.get(model_name)

    def get_summary_dataframe(self) -> pd.DataFrame:
        """Tổng hợp tất cả records thành một DataFrame."""
        all_records = []
        for session in self._sessions.values():
            for r in session.records:
                d = asdict(r)
                d["session_total_ms"] = session.total_time_ms
                d["session_peak_mem_mb"] = session.peak_mem_mb
                all_records.append(d)
        if not all_records:
            return pd.DataFrame()
        return pd.DataFrame(all_records)

    def get_session_names(self) -> List[str]:
        """Trả về danh sách tên mô hình đã profile."""
        return list(self._sessions.keys())

    @classmethod
    def load_from_dir(cls, save_dir: Union[str, Path]) -> "MultiModelProfiler":
        """
        Nạp lại kết quả đã lưu từ thư mục (chỉ dùng để phân tích, không profile lại).

        Returns:
            Instance MultiModelProfiler với _sessions đã được nạp.
        """
        save_dir = Path(save_dir)
        # Tạo instance giả (không cần models thực)
        instance = cls.__new__(cls)
        instance.models           = {}
        instance.input_factory    = lambda _: torch.zeros(1)
        instance.device           = "cpu"
        instance.gpu_idx          = 0
        instance.save_dir         = save_dir
        instance.do_calibrate     = False
        instance.exclude_types    = None
        instance.min_param_count  = 0
        instance._sessions        = {}
        instance._overheads       = {}

        # Đọc combined summary
        combined_path = save_dir / "combined_summary.json"
        if combined_path.exists():
            with open(combined_path, "r", encoding="utf-8") as f:
                combined = json.load(f)
            instance._overheads = combined.get("overheads", {})

        # Đọc từng model folder
        for model_dir in save_dir.iterdir():
            if not model_dir.is_dir():
                continue
            csv_path  = model_dir / "layer_records.csv"
            json_path = model_dir / "session_summary.json"

            if not json_path.exists():
                continue

            with open(json_path, "r", encoding="utf-8") as f:
                summary = json.load(f)

            records = []
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    # Parse list columns
                    def _parse_list(val: Any) -> List[int]:
                        if isinstance(val, list):
                            return val
                        if isinstance(val, str):
                            try:
                                return json.loads(val)
                            except Exception:
                                return []
                        return []

                    records.append(LayerRecord(
                        layer_name        = str(row.get("layer_name", "")),
                        layer_type        = str(row.get("layer_type", "")),
                        input_shape       = _parse_list(row.get("input_shape", "[]")),
                        output_shape      = _parse_list(row.get("output_shape", "[]")),
                        param_count       = int(row.get("param_count", 0)),
                        forward_time_ms   = float(row.get("forward_time_ms", 0)),
                        gpu_mem_before_mb = float(row.get("gpu_mem_before_mb", 0)),
                        gpu_mem_after_mb  = float(row.get("gpu_mem_after_mb", 0)),
                        gpu_mem_delta_mb  = float(row.get("gpu_mem_delta_mb", 0)),
                        gpu_util_pct      = float(row.get("gpu_util_pct", 0)),
                        iteration         = int(row.get("iteration", 0)),
                        device            = str(row.get("device", "cpu")),
                    ))

            session = ProfileSession(
                model_name    = summary.get("model_name", model_dir.name),
                device        = summary.get("device", "cpu"),
                total_time_ms = summary.get("total_time_ms", 0.0),
                peak_mem_mb   = summary.get("peak_mem_mb", 0.0),
                records       = records,
                metadata      = summary.get("metadata", {}),
            )
            instance._sessions[session.model_name] = session
            logger.info(f"Nạp session: {session.model_name} ({len(records)} records)")

        return instance
