"""
collector.py — Module thu thập dữ liệu profiling tầng (layer-level)
=======================================================================
Thành phần:
  - LayerProfiler       : Hook-based profiler, ghi nhận thời gian + bộ nhớ từng layer
                          (CPU: perf_counter_ns · GPU: CUDA Events, đồng bộ 1 lần/forward)
  - MultiModelProfiler  : Điều phối profiling nhiều mô hình, lưu JSON/CSV
  - DataLoaderSentinel  : Sentinel wrapper đo lường I/O vs Compute time
  - calibrate_overhead  : Đo chi phí tổng của hook (để báo cáo overhead)
  - estimate_timer_bias : Đo sai số bên trong khoảng đo (để trừ vào thời gian layer)

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

_MB = 1024 ** 2


@dataclass
class LayerRecord:
    """
    Bản ghi thống kê cho một layer tại một lần forward pass.

    Các cột cũ giữ nguyên tên để analyzer.py / dashboard.py không phải sửa.
    Các cột mới (v2) nằm ở cuối.
    """
    layer_name:        str
    layer_type:        str
    input_shape:       List[int]
    output_shape:      List[int]
    param_count:       int
    forward_time_ms:   float   = 0.0   # thời gian đã trừ timer bias
    gpu_mem_before_mb: float   = 0.0   # CUDA: memory_allocated trước layer · CPU: 0
    gpu_mem_after_mb:  float   = 0.0   # CUDA: memory_allocated sau layer  · CPU: activation_mb
    gpu_mem_delta_mb:  float   = 0.0
    gpu_util_pct:      float   = 0.0   # không đo trong hook nữa (do ResourceSampler đảm nhận)
    iteration:         int     = 0
    device:            str     = "cpu"
    # ── Cột mới (v2) ──
    raw_time_ms:       float   = 0.0   # thời gian đo được, chưa trừ timer bias
    activation_mb:     float   = 0.0   # tổng output tensor mới của layer (bỏ qua inplace / view)
    process_rss_mb:    float   = 0.0   # RAM của cả tiến trình, đo 1 lần mỗi iteration
    call_index:        int     = 0     # lần gọi thứ mấy của module trong iteration
    exec_order:        int     = 0     # thứ tự layer bắt đầu chạy trong iteration
    timing_backend:    str     = "cpu_perf_counter"   # hoặc "cuda_event"


@dataclass
class ProfileSession:
    """Tổng hợp kết quả của một lần profiling toàn bộ mô hình."""
    model_name:    str
    device:        str
    total_time_ms: float
    peak_mem_mb:   float
    records:       List[LayerRecord] = field(default_factory=list)
    metadata:      Dict[str, Any]    = field(default_factory=dict)


def _iter_tensors(obj: Any) -> Generator[torch.Tensor, None, None]:
    """Duyệt đệ quy mọi tensor trong output (Tensor, tuple, list, dict)."""
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            yield from _iter_tensors(item)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_tensors(v)


def _storage_key(t: torch.Tensor) -> Optional[int]:
    """Địa chỉ vùng nhớ gốc của tensor — dùng để nhận ra output inplace / view."""
    try:
        return t.untyped_storage().data_ptr()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 1. LayerProfiler
# ─────────────────────────────────────────────────────────────────────────────

class LayerProfiler:
    """
    Hook-based profiler — đo thời gian & bộ nhớ của từng leaf layer.

    Cách đo thời gian:
      - CPU : time.perf_counter_ns() trong pre/post hook.
      - CUDA: torch.cuda.Event được ghi vào stream trong hook (không chặn CPU),
              chỉ đồng bộ 1 LẦN khi kết thúc forward (Deferred Synchronization).
    Module được gọi nhiều lần trong 1 forward được quản lý bằng stack riêng,
    nên lần gọi sau không ghi đè lần gọi trước.

    Hạn chế đã biết (GPU): mỗi cặp CUDA event cộng thêm một "ngưỡng sàn" vài µs đến
    vài chục µs (nặng nhất trên Windows WDDM). Layer lớn đo chính xác; layer rất nhỏ
    (BatchNorm, ReLU ở batch 1) bị đo dư. TN2 đo ngưỡng này bằng estimate_timer_bias().

    Cách dùng:
        profiler = LayerProfiler(model, device="cuda")
        with profiler.profile_context(iteration=0):
            output = model(input_tensor)
        df = profiler.to_dataframe()
    """

    def __init__(
        self,
        model: nn.Module,
        device: Union[str, torch.device] = "cpu",
        gpu_device_index: int = 0,
        exclude_types: Optional[Tuple[type, ...]] = None,
        min_param_threshold: int = 0,
        atomic_types: Optional[Tuple[type, ...]] = None,
        gpu_queue_prefill: bool = True,
        track_allocator: bool = False,
    ):
        """
        Args:
            model              : Mô hình PyTorch cần profile.
            device             : 'cpu' hoặc 'cuda'.
            gpu_device_index   : Chỉ số GPU (mặc định 0).
            exclude_types      : Tuple các loại layer bỏ qua. Mặc định bỏ
                                 nn.Dropout, nn.Identity (không tính toán khi inference).
            min_param_threshold: Chỉ profile layer có số params >= ngưỡng này.
            atomic_types       : Module được đo NGUYÊN KHỐI, không đi vào module con.
                                 Mặc định nn.MultiheadAttention: phần lớn tính toán bên trong
                                 (chiếu Q/K/V, nhân ma trận attention) là hàm, không phải layer
                                 con — nếu chỉ hook layer lá sẽ bỏ sót gần hết thời gian attention.
            gpu_queue_prefill  : (GPU) Cho GPU bận một khoảng trước mỗi forward để CPU kịp xếp
                                 hết lệnh vào hàng đợi. Khi đó GPU chạy liền mạch và khoảng giữa
                                 2 CUDA event là thời gian GPU tính thật, không lẫn thời gian
                                 GPU ngồi chờ CPU gửi lệnh (giống cách Nsight / torch.profiler đo).
            track_allocator    : (GPU) Ghi torch.cuda.memory_allocated() trước/sau mỗi layer vào
                                 gpu_mem_before/after_mb. Mặc định TẮT vì mỗi lần gọi tốn ~0.2ms;
                                 bộ nhớ layer đã có ở cột activation_mb.
        """
        self.model           = model
        self.device          = str(device)
        self.is_cuda         = self.device.startswith("cuda")
        self.gpu_idx         = gpu_device_index
        self.exclude_types   = exclude_types if exclude_types is not None else (nn.Dropout, nn.Identity)
        self.atomic_types    = atomic_types if atomic_types is not None else (nn.MultiheadAttention,)
        self.min_param_count = min_param_threshold
        self.gpu_queue_prefill = gpu_queue_prefill and self.is_cuda
        self.track_allocator   = track_allocator and self.is_cuda
        self._prefill_ms:      float = 50.0    # tự điều chỉnh theo thời gian CPU xếp lệnh
        self._cycles_per_ms:   Optional[float] = None
        self._prefill_ok:      Dict[int, bool] = {}
        self._records:      List[LayerRecord] = []
        self._hooks:        List[Any]         = []
        self._param_counts: Dict[str, int]    = {}
        self._overhead_ms:  float             = 0.0
        self._current_iter: int               = 0
        self._rss_mb:       float             = 0.0
        self._lock = threading.Lock()

        # Trạng thái trong một iteration
        self._stacks:       Dict[str, List[list]] = {}
        self._call_counts:  Dict[str, int]        = {}
        self._exec_counter: int                   = 0
        self._pending:      List[tuple]           = []   # CUDA: chờ resolve sau synchronize
        self._event_pool:   List[Any]             = []
        self._event_cursor: int                   = 0
        self._iter_t0_ns:   int                   = 0
        self._iter_events:  Optional[Tuple[Any, Any]] = None
        self._iter_totals:  Dict[int, float]      = {}

    # ── Timer bias ──────────────────────────────────────────────────────────

    def set_overhead(self, overhead_ms: float) -> None:
        """
        Đặt timer bias (ms) được trừ khỏi thời gian mỗi layer.
        Nên lấy giá trị từ estimate_timer_bias(), KHÔNG dùng calibrate_overhead()
        (con số đó là tổng chi phí hook, phần lớn nằm ngoài khoảng đo).
        """
        self._overhead_ms = max(0.0, float(overhead_ms))

    # ── Chọn layer & gắn hook ───────────────────────────────────────────────

    def _select_targets(self) -> List[Tuple[str, nn.Module]]:
        """
        Leaf module (không có module con) + module atomic_types (đo nguyên khối),
        trừ exclude_types và layer ít params. Module con của khối atomic bị bỏ qua
        để không đếm trùng thời gian.
        """
        targets = []
        atomic_prefixes: List[str] = []
        for name, module in self.model.named_modules():
            if any(p == "" or name.startswith(p + ".") for p in atomic_prefixes):
                continue   # nằm bên trong một khối atomic đã chọn
            is_atomic = isinstance(module, self.atomic_types)
            if is_atomic:
                atomic_prefixes.append(name)
            elif any(True for _ in module.children()):
                continue
            if isinstance(module, self.exclude_types):
                continue
            n_params = sum(p.numel() for p in module.parameters())
            if n_params < self.min_param_count:
                continue
            name = name or "<root>"
            self._param_counts[name] = n_params
            targets.append((name, module))
        return targets

    def attach_hooks(self) -> None:
        """Gắn hooks vào tất cả leaf layer thoả mãn điều kiện."""
        self.remove_hooks()
        for name, module in self._select_targets():
            pre_h, post_h = self._make_hooks(name, module)
            self._hooks.append(module.register_forward_pre_hook(pre_h))
            self._hooks.append(module.register_forward_hook(post_h))
        logger.info(f"Đã gắn {self.n_hooked} layer hooks.")

    def remove_hooks(self) -> None:
        """Gỡ bỏ tất cả hooks đã gắn."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @property
    def n_hooked(self) -> int:
        """Số module đang được gắn hook."""
        return len(self._hooks) // 2

    # ── Hooks ───────────────────────────────────────────────────────────────

    def _next_events(self) -> Tuple[Any, Any]:
        """Lấy 2 CUDA event từ pool (tái sử dụng giữa các iteration)."""
        if self._event_cursor + 2 > len(self._event_pool):
            self._event_pool.extend(torch.cuda.Event(enable_timing=True) for _ in range(64))
        ev_start = self._event_pool[self._event_cursor]
        ev_end   = self._event_pool[self._event_cursor + 1]
        self._event_cursor += 2
        return ev_start, ev_end

    def _output_meta(self, inputs: Any, output: Any) -> Tuple[List[int], int]:
        """
        Trả về (output_shape, số byte activation mới).
        Duyệt mọi tensor của output (vd YOLO trả về nhiều tensor); bỏ qua tensor
        dùng chung vùng nhớ với input (ReLU inplace, Flatten view) để không đếm trùng.
        Chỉ trả về số nguyên/list — không giữ tham chiếu tensor.
        """
        input_keys = {_storage_key(t) for t in _iter_tensors(inputs)}
        out_shape: List[int] = []
        seen: set = set()
        total_bytes = 0
        for t in _iter_tensors(output):
            if not out_shape:
                out_shape = list(t.shape)
            key = _storage_key(t)
            if key is not None and (key in input_keys or key in seen):
                continue
            if key is not None:
                seen.add(key)
            total_bytes += t.numel() * t.element_size()
        return out_shape, total_bytes

    def _make_hooks(self, name: str, module: nn.Module) -> Tuple[Callable, Callable]:
        """
        Tạo cặp pre-hook / post-hook cho một layer.
        Mọi việc tốn thời gian (đọc shape, đọc bộ nhớ) đều làm NGOÀI khoảng đo:
        pre-hook bắt đầu bấm giờ ở dòng cuối, post-hook dừng bấm giờ ở dòng đầu.
        """
        layer_type  = type(module).__name__
        param_count = self._param_counts.get(name, 0)
        is_cuda     = self.is_cuda
        gpu_idx     = self.gpu_idx
        track_alloc = self.track_allocator

        def pre_hook(mod: nn.Module, inputs: Any) -> None:
            inp = _extract_tensor_from_output(inputs)
            in_shape = list(inp.shape) if inp is not None else []
            order = self._exec_counter
            self._exec_counter += 1
            stack = self._stacks.setdefault(name, [])
            if is_cuda:
                mem_before = torch.cuda.memory_allocated(gpu_idx) / _MB if track_alloc else 0.0
                ev_start, ev_end = self._next_events()
                stack.append([in_shape, order, mem_before, ev_start, ev_end])
                ev_start.record()
            else:
                entry = [in_shape, order, 0.0, 0]
                stack.append(entry)
                entry[3] = time.perf_counter_ns()

        def post_hook(mod: nn.Module, inputs: Any, output: Any) -> None:
            if is_cuda:
                entry = self._stacks[name].pop()
                entry[4].record()
            else:
                t_end = time.perf_counter_ns()
                entry = self._stacks[name].pop()

            out_shape, act_bytes = self._output_meta(inputs, output)
            call_idx = self._call_counts.get(name, 0)
            self._call_counts[name] = call_idx + 1

            if is_cuda:
                in_shape, order, mem_before, ev_start, ev_end = entry
                mem_after = torch.cuda.memory_allocated(gpu_idx) / _MB if track_alloc else 0.0
                self._pending.append((
                    name, layer_type, in_shape, out_shape, param_count, order,
                    call_idx, mem_before, mem_after, act_bytes, ev_start, ev_end,
                ))
            else:
                in_shape, order, _, t_start = entry
                raw_ms = (t_end - t_start) / 1e6
                self._add_record(
                    name, layer_type, in_shape, out_shape, param_count, order,
                    call_idx, raw_ms, 0.0, 0.0, act_bytes,
                )

        return pre_hook, post_hook

    def _add_record(
        self, name: str, layer_type: str, in_shape: List[int], out_shape: List[int],
        param_count: int, order: int, call_idx: int, raw_ms: float,
        mem_before: float, mem_after: float, act_bytes: int,
    ) -> None:
        act_mb = act_bytes / _MB
        if not self.track_allocator:
            # Không trộn RAM tiến trình / bộ nhớ cả chương trình vào bộ nhớ layer
            # (xem process_rss_mb); giữ cột cũ với nghĩa "bộ nhớ output của layer"
            mem_before, mem_after = 0.0, act_mb
        record = LayerRecord(
            layer_name        = name,
            layer_type        = layer_type,
            input_shape       = in_shape,
            output_shape      = out_shape,
            param_count       = param_count,
            forward_time_ms   = round(max(0.0, raw_ms - self._overhead_ms), 5),
            gpu_mem_before_mb = round(mem_before, 3),
            gpu_mem_after_mb  = round(mem_after, 3),
            gpu_mem_delta_mb  = round(mem_after - mem_before, 3),
            gpu_util_pct      = 0.0,
            iteration         = self._current_iter,
            device            = self.device,
            raw_time_ms       = round(raw_ms, 5),
            activation_mb     = round(act_mb, 6),
            process_rss_mb    = round(self._rss_mb, 2),
            call_index        = call_idx,
            exec_order        = order,
            timing_backend    = "cuda_event" if self.is_cuda else "cpu_perf_counter",
        )
        with self._lock:
            self._records.append(record)

    # ── Vòng đo ─────────────────────────────────────────────────────────────

    def begin_iteration(self, iteration: int = 0) -> None:
        """Bắt đầu một lần forward được đo (tự gắn hook nếu chưa gắn)."""
        if not self._hooks:
            self.attach_hooks()
        self._current_iter = iteration
        self._stacks.clear()
        self._call_counts.clear()
        self._pending.clear()
        self._exec_counter = 0
        self._event_cursor = 0
        try:
            self._rss_mb = psutil.Process().memory_info().rss / _MB
        except Exception:
            self._rss_mb = 0.0
        if self.is_cuda:
            torch.cuda.synchronize(self.gpu_idx)
            if self.gpu_queue_prefill:
                torch.cuda._sleep(int(self._prefill_ms * self._gpu_cycles_per_ms()))
            ev_start, ev_end = self._next_events()
            self._iter_events = (ev_start, ev_end)
            ev_start.record()
        self._iter_t0_ns = time.perf_counter_ns()

    def _gpu_cycles_per_ms(self) -> float:
        """Số chu kỳ GPU trong 1ms — đo 1 lần để quy đổi thời gian 'lấp hàng đợi' sang chu kỳ."""
        if self._cycles_per_ms is None:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda._sleep(1_000)
            start.record()
            torch.cuda._sleep(2_000_000)
            end.record()
            torch.cuda.synchronize(self.gpu_idx)
            self._cycles_per_ms = 2_000_000 / max(start.elapsed_time(end), 1e-3)
        return self._cycles_per_ms

    def end_iteration(self) -> float:
        """
        Kết thúc lần forward: đồng bộ GPU một lần duy nhất, tính thời gian các layer.
        Returns: tổng thời gian forward của iteration (ms). Trên GPU khi bật gpu_queue_prefill,
        đây là thời gian GPU chạy liền mạch (không tính thời gian chờ CPU gửi lệnh).
        """
        if self.is_cuda and self._iter_events is not None:
            ev_start, ev_end = self._iter_events
            ev_end.record()
            if self.gpu_queue_prefill:
                # Thời gian CPU xếp lệnh phải ngắn hơn thời gian lấp hàng đợi thì GPU mới chạy
                # liền mạch; tự điều chỉnh cho iteration sau.
                enqueue_ms = (time.perf_counter_ns() - self._iter_t0_ns) / 1e6
                self._prefill_ok[self._current_iter] = enqueue_ms < self._prefill_ms
                self._prefill_ms = min(1000.0, max(5.0, 1.5 * enqueue_ms + 2.0))
            torch.cuda.synchronize(self.gpu_idx)
            total_ms = ev_start.elapsed_time(ev_end)
            for (name, layer_type, in_shape, out_shape, param_count, order, call_idx,
                 mem_before, mem_after, act_bytes, l_start, l_end) in self._pending:
                self._add_record(
                    name, layer_type, in_shape, out_shape, param_count, order,
                    call_idx, l_start.elapsed_time(l_end), mem_before, mem_after, act_bytes,
                )
            self._pending.clear()
            self._iter_events = None
        else:
            total_ms = (time.perf_counter_ns() - self._iter_t0_ns) / 1e6
        self._iter_totals[self._current_iter] = total_ms
        return total_ms

    @contextmanager
    def profile_context(self, iteration: int = 0) -> Generator[None, None, None]:
        """
        Context manager để profile một lần forward pass.

        Ví dụ:
            with profiler.profile_context(iteration=i):
                output = model(x)

        Nếu hook chưa được gắn trước, context tự gắn và tự gỡ khi kết thúc.
        """
        attached_here = not self._hooks
        self.begin_iteration(iteration)
        try:
            yield
        finally:
            self.end_iteration()
            if attached_here:
                self.remove_hooks()

    # ── Data access ─────────────────────────────────────────────────────────

    def get_records(self) -> List[LayerRecord]:
        """Trả về danh sách bản ghi."""
        return list(self._records)

    def get_iteration_totals(self) -> Dict[int, float]:
        """Tổng thời gian forward (ms) của từng iteration — dùng tính phần thời gian bỏ sót."""
        return dict(self._iter_totals)

    def get_prefill_ok(self) -> Dict[int, bool]:
        """(GPU) Iteration nào GPU chạy liền mạch (True) — False nghĩa là có lúc GPU chờ CPU."""
        return dict(self._prefill_ok)

    def to_dataframe(self) -> pd.DataFrame:
        """Chuyển đổi records thành DataFrame."""
        if not self._records:
            return pd.DataFrame()
        return pd.DataFrame([asdict(r) for r in self._records])

    def clear_records(self) -> None:
        """Xoá toàn bộ records."""
        with self._lock:
            self._records.clear()
        self._iter_totals.clear()
        self._prefill_ok.clear()

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
# 2. calibrate_overhead & estimate_timer_bias
# ─────────────────────────────────────────────────────────────────────────────

def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def calibrate_overhead_details(
    model: nn.Module,
    input_tensor: torch.Tensor,
    n_warmup: int = 5,
    n_measure: int = 20,
    device: str = "cpu",
    exclude_types: Optional[Tuple[type, ...]] = None,
    min_param_threshold: int = 0,
) -> Dict[str, float]:
    """
    Đo profiler làm model chậm đi bao nhiêu (dùng cho TN2).

    Chạy XEN KẼ: không hook → có hook → không hook → ... để máy nóng lên
    không làm lệch kết quả. Lấy trung vị của mỗi bên.

    Returns: dict gồm
        t_clean_ms      : thời gian forward không gắn hook
        t_hooked_ms     : thời gian forward có profiler (gồm cả bước resolve CUDA event)
        n_hooked        : số module thực sự được gắn hook
        overhead_pct    : (t_hooked − t_clean) / t_clean × 100
        per_layer_ms    : (t_hooked − t_clean) / n_hooked
    """
    device_obj = torch.device(device)
    model = model.to(device_obj).eval()
    input_tensor = input_tensor.to(device_obj)
    # Tắt "lấp hàng đợi": ở đây cần chi phí THẬT của hook khi chạy bình thường
    profiler = LayerProfiler(
        model, device=device,
        exclude_types=exclude_types, min_param_threshold=min_param_threshold,
        gpu_queue_prefill=False,
    )

    with torch.no_grad():
        for _ in range(n_warmup):
            model(input_tensor)
        _sync(device)

        clean_times, hooked_times = [], []
        for i in range(n_measure):
            # Không hook
            _sync(device)
            t0 = time.perf_counter()
            model(input_tensor)
            _sync(device)
            clean_times.append((time.perf_counter() - t0) * 1000.0)

            # Có hook
            profiler.attach_hooks()
            t0 = time.perf_counter()
            profiler.begin_iteration(i)
            model(input_tensor)
            profiler.end_iteration()
            hooked_times.append((time.perf_counter() - t0) * 1000.0)
            n_hooked = profiler.n_hooked
            profiler.remove_hooks()

    t_clean  = float(np.median(clean_times))
    t_hooked = float(np.median(hooked_times))
    n_hooked = max(n_hooked, 1)
    diff = max(0.0, t_hooked - t_clean)
    details = {
        "t_clean_ms":   t_clean,
        "t_hooked_ms":  t_hooked,
        "n_hooked":     n_hooked,
        "overhead_pct": diff / t_clean * 100.0 if t_clean > 0 else 0.0,
        "per_layer_ms": diff / n_hooked,
    }
    logger.info(
        f"[calibrate] no_hook={t_clean:.3f}ms | with_hook={t_hooked:.3f}ms | "
        f"n_hooked={n_hooked} | overhead={details['overhead_pct']:.1f}% "
        f"({details['per_layer_ms']:.4f}ms/layer)"
    )
    return details


def calibrate_overhead(
    model: nn.Module,
    input_tensor: torch.Tensor,
    n_warmup: int = 5,
    n_measure: int = 20,
    device: str = "cpu",
    exclude_types: Optional[Tuple[type, ...]] = None,
    min_param_threshold: int = 0,
) -> float:
    """
    Chi phí TỔNG của hook, chia đều cho mỗi layer được gắn hook (ms/layer).

    Sửa so với v1: mẫu số là số module THỰC SỰ được gắn hook (kể cả layer
    không có tham số như ReLU, Pooling), không chỉ các layer có tham số.

    Lưu ý: con số này dùng để BÁO CÁO overhead. Không trừ nó vào thời gian
    từng layer — dùng estimate_timer_bias() cho việc đó.
    """
    return calibrate_overhead_details(
        model, input_tensor, n_warmup=n_warmup, n_measure=n_measure, device=device,
        exclude_types=exclude_types, min_param_threshold=min_param_threshold,
    )["per_layer_ms"]


def estimate_timer_bias(
    device: str = "cpu",
    n_layers: int = 64,
    n_runs: int = 30,
    n_warmup: int = 5,
    tiny_kernel: bool = False,
) -> float:
    """
    Ước lượng sai số nằm BÊN TRONG khoảng đo của mỗi layer (ms).

    Cách làm: profile một chuỗi layer gần như không tính toán gì. Thời gian đo
    được trên các layer này chính là sai số của bộ đo.
      - tiny_kernel=False: chuỗi nn.Identity (không chạy phép tính nào) → dùng để TRỪ
        vào forward_time_ms trên CPU qua LayerProfiler.set_overhead().
      - tiny_kernel=True : chuỗi ReLU trên tensor 1 phần tử → mỗi layer gửi đúng 1 kernel
        nhỏ nhất xuống GPU; đo được "ngưỡng sàn" thật của CUDA event (gồm chi phí gửi kernel).
    """
    make = nn.ReLU if tiny_kernel else nn.Identity
    chain = nn.Sequential(*[make() for _ in range(n_layers)]).to(device)
    x = torch.zeros(1, device=device)
    profiler = LayerProfiler(chain, device=device, exclude_types=())
    profiler.attach_hooks()
    with torch.no_grad():
        for i in range(n_warmup + n_runs):
            profiler.begin_iteration(i)
            chain(x)
            profiler.end_iteration()
    profiler.remove_hooks()
    df = profiler.to_dataframe()
    bias = float(df.loc[df["iteration"] >= n_warmup, "raw_time_ms"].median())
    logger.info(f"[timer bias] device={device} | bias={bias * 1000:.2f}µs/layer")
    return bias


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

        # Calibrate: overhead tổng (để báo cáo) + timer bias (để trừ vào từng layer)
        overhead_ms   = 0.0
        timer_bias_ms = 0.0
        if self.do_calibrate:
            logger.info(f"[{model_name}] Đang hiệu chỉnh overhead hook...")
            try:
                overhead_ms = calibrate_overhead(
                    model, input_tensor,
                    n_warmup=3, n_measure=10,
                    device=self.device,
                    exclude_types=self.exclude_types,
                    min_param_threshold=self.min_param_count,
                )
                self._overheads[model_name] = overhead_ms
                # Trên GPU, khoảng giữa 2 CUDA event là thời gian kernel nên không trừ bias
                if not self.device.startswith("cuda"):
                    timer_bias_ms = estimate_timer_bias(self.device)
            except Exception as e:
                logger.warning(f"[{model_name}] calibrate_overhead lỗi: {e}. Dùng overhead=0.")

        # Warmup
        logger.info(f"[{model_name}] Warmup {n_warmup} lần...")
        with torch.no_grad():
            for _ in range(n_warmup):
                _extract_tensor_from_output(model(input_tensor))
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

        # Profiling iterations
        profiler = LayerProfiler(
            model,
            device=self.device,
            gpu_device_index=self.gpu_idx,
            exclude_types=self.exclude_types,
            min_param_threshold=self.min_param_count,
        )
        profiler.set_overhead(timer_bias_ms)
        profiler.attach_hooks()

        t_total_start = time.perf_counter()

        with torch.no_grad():
            for i in tqdm(range(n_iterations), desc=f"Profiling {model_name}", leave=False):
                with profiler.profile_context(iteration=i):
                    _extract_tensor_from_output(model(input_tensor))

        profiler.remove_hooks()
        if self.device.startswith("cuda"):
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
                "timer_bias_ms": round(timer_bias_ms, 5),
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
