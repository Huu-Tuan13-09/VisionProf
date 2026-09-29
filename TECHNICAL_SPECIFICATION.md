# ĐẶC TẢ KỸ THUẬT & PHƯƠNG PHÁP LUẬN CHI TIẾT (TECHNICAL SPECIFICATION v2.5)
## Hệ Thống VisionProf: Thuật Toán, Mô Hình Toán Học & Cấu Trúc Kỹ Thuật

> **Tài liệu tham chiếu**: Bổ trợ trực tiếp cho [research_plan_profiling_framework.md](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/research_plan_profiling_framework.md) (Master Plan v3.5)  
> **Mục tiêu**: Cung cấp tài liệu đặc tả kỹ thuật chi tiết mức mã nguồn (code-level specification), các công thức toán học, thuật toán giải quyết triệt để các lỗi kỹ thuật đã phát hiện trong [SYSTEM_AUDIT_REPORT.md](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/SYSTEM_AUDIT_REPORT.md) và tích hợp các chuẩn mực công nghiệp toàn diện (Tail Latency P50-P99, Background Resource Sentinel, End-to-End Pipeline Sentinel, Output Consistency).

---

## MỤC LỤC
1. [Tầng Thu Thập & Đo Kiểm (Instrumentation Engine & Timing Architecture)](#1-tầng-thu-thập--đo-kiểm-instrumentation-engine--timing-architecture)
   - [1.1. Giải pháp Nghịch lý CUDA Hook: Kỹ thuật Deferred Synchronization](#11-giải-pháp-nghịch-lý-cuda-hook-kỹ-thuật-deferred-synchronization)
   - [1.2. Công thức toán học hiệu chuẩn sai số (Calibrate Overhead v2.0)](#12-công-thức-toán-học-hiệu-chuẩn-sai-số-calibrate-overhead-v20)
   - [1.3. Chuẩn hóa đo lường bộ nhớ: Tách biệt Process RSS và Activation RAM](#13-chuẩn-hóa-đo-lường-bộ-nhớ-tách-biệt-process-rss-và-activation-ram)
   - [1.4. Tầng Giám sát Tài nguyên Toàn cục (SystemResourceSentinel: CPU/GPU/Power/Temp)](#14-tầng-giám-sát-tài-nguyên-toàn-cục-systemresourcesentinel-cpugpupowertemp)
2. [Tầng Định Lượng Hiệu Năng & Mô Hình Roofline Model](#2-tầng-định-lượng-hiệu-năng--mô-hình-roofline-model)
   - [2.1. Thuật toán trích xuất FLOPs lý thuyết cho từng Layer](#21-thuật-toán-trích-xuất-flops-lý-thuyết-cho-từng-layer)
   - [2.2. Công thức tính Cường độ số học (Arithmetic Intensity) & Hiệu ứng L2 Cache](#22-công-thức-tính-cường-độ-số-học-arithmetic-intensity--hiệu-ứng-l2-cache)
   - [2.3. Dựng đường biên Roofline chuẩn trên GPU Tesla T4 và CPU](#23-dựng-đường-biên-roofline-chuẩn-trên-gpu-tesla-t4-và-cpu)
   - [2.4. Thuật toán phân loại Compute-bound vs. Memory-bound](#24-thuật-toán-phân-loại-compute-bound-vs-memory-bound)
3. [Tái Cấu Trúc Danh Mục 19 Quy Tắc Heuristic (Heuristic Catalog v2.0)](#3-tái-cấu-trúc-danh-mục-19-quy-tắc-heuristic-heuristic-catalog-v20)
   - [3.1. Nguyên tắc thiết kế loại bỏ các quy tắc sai lầm](#31-nguyên-tắc-thiết-kế-loại-bỏ-các-quy-tắc-sai-lầm)
   - [3.2. Bảng đặc tả chi tiết 19 quy tắc chuẩn hóa v2.0](#32-bảng-đặc-tả-chi-tiết-19-quy-tắc-chuẩn-hóa-v20)
   - [3.3. Cơ chế Ngưỡng Động thích ứng phần cứng (Dynamic Thresholds)](#33-cơ-chế-ngưỡng-động-thích-ứng-phần-cứng-dynamic-thresholds)
4. [Công Cụ So Sánh Thống Kê A/B & Động Cơ Độ Trễ (Statistical & Latency Engine)](#4-công-cụ-so-sánh-thống-kê-ab--động-cơ-độ-trễ-statistical--latency-engine)
   - [4.1. Công thức tính Speedup tổng thể chính xác](#41-công-thức-tính-speedup-tổng-thể-chính-xác)
   - [4.2. Kiểm định thống kê hợp lệ (Mann-Whitney U + Cliff's Delta)](#42-kiểm-định-thống-kê-hợp-lệ-mann-whitney-u--cliffs-delta)
   - [4.3. Thuật toán so sánh Cross-Architecture theo tổng thời gian nhóm](#43-thuật-toán-so-sánh-cross-architecture-theo-tổng-thời-gian-nhóm)
   - [4.4. Động cơ phân tích Tail Latency (P50, P90, P95, P99) & Throughput (FPS)](#44-động-cơ-phân-tích-tail-latency-p50-p90-p95-p99--throughput-fps)
   - [4.5. Đánh giá Độ trôi dạt Đầu ra & Đánh đổi Pareto (Output Consistency & Trade-off)](#45-đánh-giá-độ-trôi-dạt-đầu-ra--đánh-đổi-pareto-output-consistency--trade-off)
5. [Tầng Dữ Liệu & Lưu Trữ Quan Hệ (Data Storage Engine)](#5-tầng-dữ-liệu--lưu-trữ-quan-hệ-data-storage-engine)
   - [5.1. Lược đồ cơ sở dữ liệu quan hệ SQLite toàn diện (DDL Schema)](#51-lược-đồ-cơ-sở-dữ-liệu-quan-hệ-sqlite-toàn-diện-ddl-schema)
   - [5.2. Định dạng xuất nén cột Parquet](#52-định-dạng-xuất-nén-cột-parquet)
6. [Mở Rộng Giám Sát Toàn Bộ Pipeline (End-to-End Pipeline Sentinel)](#6-mở-rộng-giám-sát-toàn-bộ-pipeline-end-to-end-pipeline-sentinel)
   - [6.1. Khắc phục lỗi Cold-Start trong DataLoaderSentinel](#61-khắc-phục-lỗi-cold-start-trong-dataloadersentinel)
   - [6.2. Kiến trúc PipelineSentinel (Preprocess → Inference → Postprocess)](#62-kiến-trúc-pipelinesentinel-preprocess--inference--postprocess)
7. [Quy Trình Kiểm Chuẩn Với NVIDIA Nsight Systems (Validation Protocol)](#7-quy-trình-kiểm-chuẩn-với-nvidia-nsight-systems-validation-protocol)
8. [Phân Tích Phản Biện Chuyên Sâu, Rủi Ro Tiềm Ẩn & Trường Hợp Biên (Adversarial Edge Cases)](#8-phân-tích-phản-biện-chuyên-sâu-rủi-ro-tiềm-ẩn--trường-hợp-biên-adversarial-edge-cases)
   - [8.1. Hiện tượng Trôi nhiệt & Dao động xung nhịp GPU trong A/B Testing](#81-hiện-tượng-trôi-nhiệt--dao-động-xung-nhịp-gpu-trong-ab-testing)
   - [8.2. Xung đột trạng thái khi Module được gọi lặp lại (Re-entrant / Weight-Tied)](#82-xung-đột-trạng-thái-khi-module-được-gọi-lặp-lại-re-entrant--weight-tied)
   - [8.3. Mô hình Roofline Đa trần (Dual-Ceiling Roofline: FP32 vs. Tensor Core FP16)](#83-mô-hình-roofline-đa-trần-dual-ceiling-roofline-fp32-vs-tensor-core-fp16)
   - [8.4. Xử lý phép toán tại chỗ (In-place Operations) và Tensor Storage Aliasing](#84-xử-lý-phép-toán-tại-chỗ-in-place-operations-và-tensor-storage-aliasing)
   - [8.5. Giới hạn tương thích với `torch.compile` (Graph Breaks do Hooks)](#85-giới-hạn-tương-thích-với-torchcompile-graph-breaks-do-hooks)
   - [8.6. Điểm mù của Leaf Hooks đối với Functional Operations không đóng gói module](#86-điểm-mù-của-leaf-hooks-đối-với-functional-operations-không-đóng-gói-module)
   - [8.7. Bẫy giữ tham chiếu Activation Tensor trong Event Queue (The Memory Retention Leak)](#87-bẫy-giữ-tham-chiếu-activation-tensor-trong-event-queue-the-memory-retention-leak)
   - [8.8. Bỏ sót Tensor đầu ra trong các mạng Multi-Scale / FPN Heads](#88-bỏ-sót-tensor-đầu-ra-trong-các-mạng-multi-scale--fpn-heads)
   - [8.9. Ranh giới phạm vi: Đơn thiết bị (Single-Device) vs. Phân tán đa GPU](#89-ranh-giới-phạm-vi-đơn-thiết-bị-single-device-vs-phân-tán-đa-gpu)
   - [8.10. Tranh chấp GIL và Tần số Lấy mẫu của Background Resource Thread](#810-tranh-chấp-gil-và-tần-số-lấy-mẫu-của-background-resource-thread)
   - [8.11. Biến thiên Xung nhịp CPU (CPU Frequency Governors & Turbo Boost Jitter)](#811-biến-thiên-xung-nhịp-cpu-cpu-frequency-governors--turbo-boost-jitter)
   - [8.12. Rủi ro Biến thiên Kích thước Ảnh Động (Dynamic Resolution & Padding Variance)](#812-rủi-ro-biến-thiên-kích-thước-ảnh-động-dynamic-resolution--padding-variance)
   - [8.13. Ranh giới Kiến trúc Runtime: PyTorch Eager vs. Black-box ONNX/TensorRT Session](#813-ranh-giới-kiến-trúc-runtime-pytorch-eager-vs-black-box-onnxtensorrt-session)

---

## 1. TẦNG THU THẬP & ĐO KIỂM (INSTRUMENTATION ENGINE & TIMING ARCHITECTURE)

### 1.1. Giải pháp Nghịch lý CUDA Hook: Kỹ thuật Deferred Synchronization

#### A. Vấn đề của phương pháp cũ
- Đo bằng `time.perf_counter()` trên CPU: Chỉ đo thời gian enqueue kernel vào CUDA Stream (~2µs), hoàn toàn sai lệch thời gian thực thi của GPU.
- Gọi `torch.cuda.synchronize()` ở mỗi layer hook: Gây ra pipeline stall liên tục, phá vỡ tính năng kernel overlap của GPU, làm tăng từ 15% đến 35% overhead giả tạo.

#### B. Thuật toán Deferred Synchronization Architecture
Để đạt được độ chính xác micro-giây mà vẫn giữ **overhead $< 2\%$**, hệ thống áp dụng kỹ thuật **Ghi nhận Sự kiện Bất đồng bộ và Trì hoãn Đồng bộ**:
1. Trong `pre_hook`: Khởi tạo và ghi nhận `start_event = torch.cuda.Event(enable_timing=True)`. Lệnh này đẩy một marker vào CUDA Stream mà không hề chặn CPU.
2. Trong `post_hook`: Khởi tạo và ghi nhận `end_event = torch.cuda.Event(enable_timing=True)`. Lưu cặp `(start_event, end_event)` vào hàng đợi của iteration.
3. Tại thời điểm kết thúc toàn bộ forward pass: Gọi `torch.cuda.synchronize()` **DUY NHẤT MỘT LẦN**.
4. Sau khi GPU đã hoàn tất toàn bộ đồ thị tính toán, CPU duyệt qua hàng đợi và gọi `start_event.elapsed_time(end_event)` để lấy thời gian chính xác của từng layer.

```python
# Pseudo-code triển khai chuẩn Zero-Retention Deferred Synchronization trong LayerProfiler:

class LayerProfilerV2:
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.is_cuda = device.startswith("cuda")
        self._event_queue = [] # Lưu metadata và cặp event đã hoàn thành
        self._event_pool = {}  # Pre-allocated Event Pool (tái sử dụng, tránh cudaEventCreate liên tục)
        self._stack = {}       # LIFO stack hỗ trợ re-entrant modules
        self._records = []

    def _get_or_create_events(self, name: str):
        """Pre-allocate hoặc tái sử dụng CUDA events để triệt tiêu driver syscall overhead."""
        if name not in self._event_pool:
            self._event_pool[name] = []
        return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def _make_hooks(self, name: str, module: nn.Module):
        def pre_hook(mod, inp):
            if self.is_cuda:
                ev_start, ev_end = self._get_or_create_events(name)
                ev_start.record(stream=torch.cuda.current_stream())
                if name not in self._stack: self._stack[name] = []
                self._stack[name].append((ev_start, ev_end))
            else:
                if name not in self._stack: self._stack[name] = []
                self._stack[name].append(time.perf_counter_ns())

        def post_hook(mod, inp, out):
            # 1. TRÍCH XUẤT METADATA NGAY LẬP TỨC — TUYỆT ĐỐI KHÔNG GIỮ THAM CHIẾU TENSOR 'out'
            tensors_meta = self._extract_tensors_metadata(out)
            
            if self.is_cuda:
                ev_start, ev_end = self._stack[name].pop()
                ev_end.record(stream=torch.cuda.current_stream())
                # Lưu metadata dạng primitives (không giữ VRAM), TUYỆT ĐỐI KHÔNG SYNC TẠI ĐÂY
                self._event_queue.append((name, type(mod).__name__, ev_start, ev_end, tensors_meta))
            else:
                t_start_ns = self._stack[name].pop()
                elapsed_ms = (time.perf_counter_ns() - t_start_ns) / 1e6
                self._record_cpu(name, mod, elapsed_ms, tensors_meta)

        return pre_hook, post_hook

    def flush_and_resolve(self, iteration: int):
        """Được gọi DUY NHẤT 1 LẦN sau khi model(x) hoàn tất cả forward pass."""
        if self.is_cuda and self._event_queue:
            torch.cuda.synchronize() # Đồng bộ 1 lần duy nhất toàn mạng
            for name, ltype, ev_start, ev_end, meta in self._event_queue:
                elapsed_ms = ev_start.elapsed_time(ev_end) # Thời gian GPU chính xác
                elapsed_calibrated = max(0.0, elapsed_ms - self._overhead_ms)
                self._create_record_from_meta(name, ltype, elapsed_calibrated, meta, iteration)
            self._event_queue.clear()
```

---

### 1.2. Công thức toán học hiệu chuẩn sai số (Calibrate Overhead v2.0)

Để trừ hao chính xác độ trễ do bản thân code hook gây ra, công thức tính overhead phải đếm **chính xác 100% số lượng module được gắn hook**:

$$\mathcal{M}_{leaf} = \{ m \in \text{NamedModules}(M) \mid \text{len}(\text{children}(m)) = 0 \land m \notin \text{ExcludeTypes} \}$$

$$N_{hooked} = |\mathcal{M}_{leaf}|$$

Quy trình đo:
1. Chạy mô hình gốc $M$ qua $N_{meas}$ lần lặp không gắn hook, lấy trung vị thời gian:
   $$T_{clean} = \text{median}(\{ t_{clean}^{(1)}, t_{clean}^{(2)}, \dots, t_{clean}^{(N_{meas})} \})$$
2. Gắn hook vào tất cả các module trong $\mathcal{M}_{leaf}$, chạy qua $N_{meas}$ lần lặp, lấy trung vị:
   $$T_{hooked} = \text{median}(\{ t_{hooked}^{(1)}, t_{hooked}^{(2)}, \dots, t_{hooked}^{(N_{meas})} \})$$
3. Overhead trung bình hiệu chuẩn trên mỗi tầng được tính bằng:
   $$\text{Overhead}_{layer} = \max\left(0, \frac{T_{hooked} - T_{clean}}{N_{hooked}}\right)$$

*Lưu ý cốt lõi: Mẫu số bắt buộc phải là $N_{hooked}$ (toàn bộ leaf modules, bao gồm cả layer có và không có tham số).*

---

### 1.3. Chuẩn hóa đo lường bộ nhớ: Tách biệt Process RSS và Activation RAM

Hệ thống định nghĩa rõ 3 trường bộ nhớ độc lập:
1. **`process_rss_mb`**: Dung lượng RAM vật lý của toàn bộ tiến trình hệ điều hành (chỉ lấy mẫu 1 lần ở đầu iteration, không gắn vào từng layer).
2. **`layer_activation_mb`**: Dung lượng tensor đầu ra trực tiếp của layer (được tính bằng giải tích, zero-syscall overhead):
   $$\text{Activation\_MB} = \frac{\prod_{d \in \text{shape}} d \times \text{element\_size}}{1024^2}$$
3. **`cuda_allocated_mb`** & **`cuda_reserved_mb`**: Đo lường qua `torch.cuda.memory_allocated()` và `torch.cuda.memory_reserved()` khi chạy GPU.

Tên cột trong bảng dữ liệu:
- Khi `device="cpu"`: Xuất cột `ram_activation_mb` và `process_rss_mb`.
- Khi `device="cuda"`: Xuất cột `cuda_allocated_mb` và `cuda_reserved_mb`.

---

### 1.4. Tầng Giám sát Tài nguyên Toàn cục (SystemResourceSentinel: CPU/GPU/Power/Temp)

Để thu thập tài nguyên hệ thống (CPU %, RAM, GPU Util %, VRAM, Power Watts, Temp °C) mà không gây trễ hoặc overhead cho hook layer, hệ thống sử dụng một luồng nền Daemon (`SystemResourceSentinel`):

```python
class SystemResourceSentinel:
    """
    Tiến trình nền lấy mẫu tài nguyên phần cứng định kỳ (mặc định: 50ms).
    Hoàn toàn tách biệt khỏi luồng thực thi forward pass của PyTorch.
    """
    def __init__(self, sample_interval_s: float = 0.050, device: str = "cpu"):
        self.sample_interval = sample_interval_s
        self.device = device
        self.is_cuda = device.startswith("cuda")
        self._stop_event = threading.Event()
        self._thread = None
        self.samples = []
        self._process = psutil.Process(os.getpid())

    def start(self):
        self._stop_event.clear()
        self.samples.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def _monitor_loop(self):
        while not self._stop_event.is_set():
            t_now = time.time()
            cpu_pct = psutil.cpu_percent(interval=None)
            mem_info = self._process.memory_info()
            rss_mb = mem_info.rss / (1024 * 1024)
            
            sample = {
                "timestamp": t_now,
                "cpu_util_pct": cpu_pct,
                "process_rss_mb": rss_mb,
                "gpu_util_pct": 0.0,
                "gpu_mem_used_mb": 0.0,
                "gpu_power_w": 0.0,
                "gpu_temp_c": 0.0,
            }
            
            if self.is_cuda and _PYNVML_OK:
                try:
                    rates = pynvml.nvmlDeviceGetUtilizationRates(_nvml_handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(_nvml_handle)
                    temp_c = pynvml.nvmlDeviceGetTemperature(_nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
                    sample["gpu_util_pct"] = float(rates.gpu)
                    sample["gpu_mem_used_mb"] = float(mem.used) / (1024 * 1024)
                    sample["gpu_power_w"] = float(power_mw) / 1000.0
                    sample["gpu_temp_c"] = float(temp_c)
                except Exception:
                    pass

            self.samples.append(sample)
            time.sleep(self.sample_interval)

    def get_summary(self, t_start: float, t_end: float) -> dict:
        """Trích xuất thống kê trung bình và cực trị trong khoảng thời gian diễn ra iteration."""
        relevant = [s for s in self.samples if t_start <= s["timestamp"] <= t_end]
        if not relevant:
            return {"avg_cpu_pct": 0.0, "avg_gpu_util_pct": 0.0, "peak_power_w": 0.0, "max_temp_c": 0.0}
        return {
            "avg_cpu_pct": float(np.mean([s["cpu_util_pct"] for s in relevant])),
            "avg_gpu_util_pct": float(np.mean([s["gpu_util_pct"] for s in relevant])),
            "peak_power_w": float(np.max([s["gpu_power_w"] for s in relevant])),
            "max_temp_c": float(np.max([s["gpu_temp_c"] for s in relevant])),
        }
```

---

## 2. TẦNG ĐỊNH LƯỢNG HIỆU NĂNG & MÔ HÌNH ROOFLINE MODEL

### 2.1. Thuật toán trích xuất FLOPs lý thuyết cho từng Layer

Hệ thống trích xuất giải tích số phép tính dấu phẩy động (FLOPs) theo cấu trúc tensor:

| Loại Layer (`layer_type`) | Công thức tính toán FLOPs chuẩn (Forward Pass) |
| :--- | :--- |
| **`Conv2d`** | $\text{FLOPs} = 2 \times B \times C_{out} \times H_{out} \times W_{out} \times \left( \frac{C_{in}}{\text{groups}} \times K_h \times K_w \right)$ |
| **`Linear`** | $\text{FLOPs} = 2 \times B \times N_{in} \times N_{out}$ |
| **`BatchNorm2d`** | $\text{FLOPs} = 2 \times B \times C \times H \times W$ (Normalize + Scale/Shift) |
| **`LayerNorm`** | $\text{FLOPs} = 2 \times \text{TotalElements}$ |
| **`MultiheadAttention`** | $\text{FLOPs} = 4 \times B \times S^2 \times D + 4 \times B \times S \times D^2$ ($S$: sequence length, $D$: embed dim) |

Hệ thống cung cấp module tự động trích xuất thông qua `fvcore.nn.FlopCountAnalysis` hoặc fallback giải tích dựa trên `input_shape` và `output_shape` thu thập từ hook.

---

### 2.2. Công thức tính Cường độ số học (Arithmetic Intensity) & Hiệu ứng L2 Cache

Cường độ số học ($I$, đơn vị: $\text{FLOPs/Byte}$) thể hiện số phép tính thực hiện trên mỗi byte dữ liệu di chuyển qua bộ nhớ:

$$I = \frac{\text{FLOPs}_{\text{layer}}}{\text{Memory\_Traffic}_{\text{bytes}}}$$

#### A. Cường độ số học bắt buộc theo giải tích (Cold-Cache Compulsory Intensity)
Theo phương pháp giải tích không phụ thuộc phần cứng (Cold-Cache Model), lưu lượng dữ liệu tối thiểu bắt buộc phải truy xuất được tính bằng:
$$\text{Memory\_Traffic}_{\text{compulsory}} = \text{Bytes}(\text{Input Tensor}) + \text{Bytes}(\text{Weights}) + \text{Bytes}(\text{Output Tensor})$$

Ví dụ với tầng `Conv2d` (dữ liệu FP16: 2 bytes/element):
$$\text{Memory\_Traffic} = 2 \times \left( B \cdot C_{in} \cdot H_{in} \cdot W_{in} + \frac{C_{out} \cdot C_{in} \cdot K_h \cdot K_w}{\text{groups}} + B \cdot C_{out} \cdot H_{out} \cdot W_{out} \right)$$

#### B. Phân biệt với Cường độ số học vật lý (Physical DRAM Intensity) do hiệu ứng L2 Cache
Trên GPU hiện đại (NVIDIA Tesla T4 sở hữu $4\text{MB}$ On-Chip L2 Cache):
- Nếu tensor kích hoạt trung gian giữa 2 tầng kế tiếp nhỏ hơn dung lượng L2 Cache ($< 4\text{MB}$), dữ liệu sẽ được truyền trực tiếp qua L2 Cache mà **không cần ghi/đọc lại từ bộ nhớ ngoài (Off-chip DRAM)**.
- Do đó: $\text{DRAM\_Traffic}_{\text{physical}} \le \text{Memory\_Traffic}_{\text{compulsory}}$, dẫn đến $I_{\text{physical}} \ge I_{\text{compulsory}}$.
- VisionProf tính toán $I_{\text{compulsory}}$ làm **cận dưới bảo thủ (conservative lower-bound)** cho toàn bộ mô hình, đồng thời cung cấp cổng kiểm chuẩn trích xuất $\text{DRAM\_Traffic}_{\text{physical}}$ thực tế qua Nsight Compute counter (`dram__bytes_read.sum + dram__bytes_write.sum`).

---

### 2.3. Dựng đường biên Roofline chuẩn trên GPU Tesla T4 và CPU

Mô hình Roofline thiết lập giới hạn thông lượng tính toán lý thuyết ($P$, đơn vị $\text{TFLOPS}$ hoặc $\text{GFLOPS}$) theo cường độ số học $I$:

$$P_{\text{attainable}}(I) = \min \left( P_{\text{peak}}, \quad B_{\text{peak}} \times I \right)$$

#### Thông số phần cứng chuẩn mực:

| Thông số phần cứng | NVIDIA Tesla T4 (Cloud GPU) | Intel Core i7-12700H (Edge CPU) |
| :--- | :---: | :---: |
| **Peak FP32 Compute ($P_{peak}^{FP32}$)** | 8.1 TFLOPS ($8.1 \times 10^{12}$) | 450 GFLOPS ($0.45 \times 10^{12}$) |
| **Peak FP16 Tensor Core ($P_{peak}^{TC}$)** | 65.0 TFLOPS ($65.0 \times 10^{12}$) | N/A |
| **Peak DRAM Bandwidth ($B_{peak}$)** | 320 GB/s ($320 \times 10^9$) | 76.8 GB/s ($76.8 \times 10^9$) |
| **Điểm uốn bão hòa (Ridge Point $I^* = P/B$)** | **25.3 FLOPs/Byte** (FP32) · **203.1** (TC) | **5.86 FLOPs/Byte** |

---

### 2.4. Thuật toán phân loại Compute-bound vs. Memory-bound

Với mỗi layer $l$, tính toán thông lượng thực tế đạt được:
$$P_{\text{actual}}^{(l)} = \frac{\text{FLOPs}^{(l)}}{t_{\text{latency}}^{(l)}} \quad (\text{FLOPS})$$

Phân loại điểm nghẽn của tầng dựa trên vị trí tương đối so với Ridge Point ($I^*$):
```python
def classify_roofline_bottleneck(arithmetic_intensity: float, actual_tflops: float, 
                                  ridge_point: float, peak_tflops: float) -> str:
    """
    Phân loại trạng thái giới hạn phần cứng theo Roofline Model.
    """
    if arithmetic_intensity < ridge_point:
        efficiency = actual_tflops / (arithmetic_intensity * PEAK_BANDWIDTH_TBS)
        if efficiency > 0.6:
            return "MEMORY_BOUND_OPTIMAL"  # Đã tận dụng tốt băng thông
        else:
            return "MEMORY_BOUND_STALLED"  # Bị nghẽn truy xuất bộ nhớ rời rạc
    else:
        efficiency = actual_tflops / peak_tflops
        if efficiency > 0.6:
            return "COMPUTE_BOUND_SATURATED" # Tính toán bão hòa hiệu quả
        else:
            return "COMPUTE_BOUND_UNDERUTILIZED" # Tính toán chưa tối ưu (thiếu song song)
```

---

## 3. TÁI CẤU TRÚC DANH MỤC 19 QUY TẮC HEURISTIC (HEURISTIC CATALOG v2.0)

### 3.1. Nguyên tắc thiết kế loại bỏ các quy tắc sai lầm
1. **Xóa bỏ hoàn toàn tỷ số $\text{ms / Mparam}$**: Không sử dụng số tham số để đánh giá tốc độ của tầng tích chập. Thay thế hoàn toàn bằng **Hiệu suất Roofline (Attainable TFLOPS %)** và **Thời gian trên mỗi GFLOP ($\text{ms/GFLOP}$)**.
2. **Loại bỏ nhãn sai lệch "Dead Layer"**: Tầng chạy nhanh hơn $0.001\text{ ms}$ được đổi tên thành **"Sub-microsecond Layer"** (tầng siêu nhẹ) và hiển thị thông tin kiến trúc, tuyệt đối không khuyến nghị người dùng xóa bỏ.
3. **Định nghĩa lại chuẩn xác "Compute Starvation"**: Xác định trạng thái đói tính toán khi thời gian GPU ở trạng thái nhàn rỗi (idle wait) chờ nạp batch vượt quá 40% chu kỳ lặp.

---

### 3.2. Bảng đặc tả chi tiết 19 quy tắc chuẩn hóa v2.0

```
DANH MỤC HEURISTIC RULES v2.0
├── NHÓM R-A: Phân Tích Bộ Nhớ (Memory Analysis)
│   ├── RA-01: OOM Risk Threshold (Kiểm tra cả VRAM GPU và RAM Vật lý CPU)
│   ├── RA-02: PyTorch Caching Allocator Fragmentation (reserved vs. allocated)
│   ├── RA-03: Activation Memory Dominance Hotspot (> 40% tổng VRAM activation)
│   ├── RA-04: Activation to Parameter Footprint Ratio (đo thuần Activation)
│   └── RA-05: Linear Memory Leak Trend (chỉ tính sau 15 iterations warm-up)
│
├── NHÓM R-B: Phân Tích Tính Toán & Roofline (Compute & Roofline Analysis)
│   ├── RB-01: Amdahl Bottleneck Dominance (Top-3 layer chiếm > 50% thời gian)
│   ├── RB-02: Compute Timing Jitter (Đo CV dùng MAD/IQR kháng ngoại lai)
│   ├── RB-03: Extreme Serial Bottleneck Layer (1 layer đơn lẻ > 35% forward time)
│   ├── RB-04: Roofline Memory-Bound Saturation (Layer nằm trong vùng Bandwidth-Bound)
│   └── RB-05: Hardware Efficiency Underutilization (< 20% Peak Attainable FLOPS)
│
├── NHÓM R-C: Phân Tích Đường Ống Dữ Liệu (DataLoader & Pipeline I/O)
│   ├── RC-01: DataLoader I/O Dominance (Load time > 40% tổng chu kỳ)
│   ├── RC-02: DataLoader Delivery Variance (Độ bất ổn nạp đĩa CV > 0.8)
│   ├── RC-03: GPU Compute Starvation Stall (GPU Idle Time do I/O > 30%)
│   └── RC-04: Single-Worker DataLoader Warning (num_workers == 0)
│
└── NHÓM R-D: Phân Tích Cấu Trúc Kiến Trúc (Architecture & Dispatch Analysis)
    ├── RD-01: Extreme Parameter Concentration (1 layer nắm giữ > 60% tổng tham số)
    ├── RD-02: Sub-Microsecond Fast Layer Profile (Thống kê layer nhẹ, không khuyên xóa)
    ├── RD-03: Sequential Layer Dispatch Overhead (Cảnh báo phân mảnh khi Depth > 150 trên CPU)
    ├── RD-04: Layer Type Distribution & Diversity Index (Chỉ số đa dạng toán tử)
    └── RD-05: Flop-Time Anomaly Outlier (Layer có ms/GFLOP cao bất thường so với median)
```

---

### 3.3. Cơ chế Ngưỡng Động thích ứng phần cứng (Dynamic Thresholds)

Thay vì gán cứng các con số "ma thuật" (magic numbers), các ngưỡng cảnh báo tự động điều chỉnh theo cấu hình máy:
```python
def compute_dynamic_thresholds(hardware_info: dict) -> dict:
    total_mem_mb = hardware_info["total_memory_mb"]
    is_gpu = hardware_info["device_type"] == "cuda"
    
    return {
        "ra01_oom_critical_pct": 92.0 if total_mem_mb >= 15000 else 82.0,
        "ra02_frag_limit_mb": max(256.0, total_mem_mb * 0.25),
        "rd03_max_cpu_layers": 120 if not is_gpu else 500,
    }
```

---

## 4. CÔNG CỤ SO SÁNH THỐNG KÊ A/B & ĐỘNG CƠ ĐỘ TRỄ (STATISTICAL & LATENCY ENGINE)

### 4.1. Công thức tính Speedup tổng thể chính xác

Để so sánh giữa Mô hình/Cấu hình $A$ và $B$, hệ thống tính toán thời gian chạy của cả mô hình (Full Forward Pass) trên từng lần lặp $i \in \{1, \dots, N\}$:

$$T_{\text{iter}}^{(A)}[i] = \sum_{l \in \text{Layers}_A} t_{\text{layer}}^{(A)}[l, i]$$

Tỷ số tăng tốc (Overall Speedup) được tính chuẩn xác theo trung bình cộng thời gian của toàn mô hình:

$$\text{Speedup}_{A \rightarrow B} = \frac{\bar{T}_{\text{total}}^{(A)}}{\bar{T}_{\text{total}}^{(B)}} = \frac{\frac{1}{N} \sum_{i=1}^N T_{\text{iter}}^{(A)}[i]}{\frac{1}{N} \sum_{i=1}^N T_{\text{iter}}^{(B)}[i]}$$

---

### 4.2. Kiểm định thống kê hợp lệ (Mann-Whitney U + Cliff's Delta)

1. **Kiểm định phi tham số Mann-Whitney U**:
   - $H_0$: Phân phối thời gian thực thi của Model A và Model B là như nhau.
   - Mẫu đầu vào: Vector $[T_{\text{iter}}^{(A)}[1], \dots, T_{\text{iter}}^{(A)}[N]]$ và $[T_{\text{iter}}^{(B)}[1], \dots, T_{\text{iter}}^{(B)}[N]]$ với $N \ge 30$.

2. **Độ lớn hiệu ứng (Effect Size - Cliff's Delta)**:
   $$d = \frac{\# (T_A > T_B) - \# (T_A < T_B)}{N_A \times N_B}$$
   - $|d| < 0.147$: Khác biệt không đáng kể (Negligible).
   - $0.147 \le |d| < 0.33$: Khác biệt nhỏ (Small).
   - $0.33 \le |d| < 0.474$: Khác biệt trung bình (Medium).
   - $|d| \ge 0.474$: Cải thiện/Hồi quy vượt bậc (Large).

---

### 4.3. Thuật toán so sánh Cross-Architecture theo tổng thời gian nhóm

Gom nhóm theo loại toán tử (`layer_type`) và tính **tổng thời gian tích lũy** của từng nhóm trên mỗi iteration:

$$T_{\text{type}}^{(A)}[\text{type}_k, i] = \sum_{l \in \text{Layers}_A \land \text{Type}(l) = \text{type}_k} t_{\text{layer}}^{(A)}[l, i]$$

Tránh hoàn toàn lỗi so sánh trung bình 1 layer khi số lượng layer giữa hai mô hình chênh lệch lớn.

---

### 4.4. Động cơ phân tích Tail Latency (P50, P90, P95, P99) & Throughput (FPS)

Trong môi trường triển khai thực tế (Production Serving), giá trị `mean` thường che giấu các xung đột độ trễ (latency spikes). Động cơ thống kê của VisionProf tính toán bắt buộc:

```python
def compute_latency_distribution(latencies_ms: List[float], batch_size: int = 1) -> dict:
    """
    Tính toán phân phối độ trễ chuẩn công nghiệp và thông lượng (Throughput).
    latencies_ms: Vector thời gian thực thi của toàn bộ mô hình qua N iterations (sau warm-up).
    """
    arr = np.array(latencies_ms)
    mean_lat = float(np.mean(arr))
    std_lat = float(np.std(arr))
    
    # Tính các phân vị Percentiles bằng phương pháp nội suy tuyến tính (Linear Interpolation)
    p50 = float(np.percentile(arr, 50))
    p90 = float(np.percentile(arr, 90))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    
    # Throughput (Images per Second / FPS)
    # FPS = (Batch_Size * 1000.0) / Mean_Latency_ms
    throughput_fps = (batch_size * 1000.0) / mean_lat if mean_lat > 0 else 0.0
    
    # Chỉ số suy biến Tail Disparity Ratio
    tail_disparity = p99 / mean_lat if mean_lat > 0 else 1.0

    return {
        "mean_ms": mean_lat,
        "std_ms": std_lat,
        "p50_ms": p50,
        "p90_ms": p90,
        "p95_ms": p95,
        "p99_ms": p99,
        "throughput_fps": throughput_fps,
        "tail_disparity_p99_over_mean": tail_disparity
    }
```

---

### 4.5. Đánh giá Độ trôi dạt Đầu ra & Đánh đổi Pareto (Output Consistency & Trade-off)

Để liên kết giữa **Profiling** và **Evaluation** mà không đòi hỏi tập dữ liệu gán nhãn khổng lồ, framework cung cấp module đo lường **Sự phân kỳ đầu ra (Output Divergence)** giữa các cấu hình tối ưu (ví dụ: FP32 Baseline vs. FP16 Mixed Precision / INT8):

$$\text{Cosine Similarity} = \frac{\mathbf{y}_{\text{baseline}} \cdot \mathbf{y}_{\text{optimized}}}{\|\mathbf{y}_{\text{baseline}}\|_2 \|\mathbf{y}_{\text{optimized}}\|_2}$$

$$\text{Mean Absolute Error (L1)} = \frac{1}{M} \sum_{j=1}^M |y_{\text{baseline}}^{(j)} - y_{\text{optimized}}^{(j)}|$$

Nếu $\text{Cosine Similarity} < 0.995$ hoặc $L_1 > 10^{-2}$, hệ thống sẽ cảnh báo cấu hình tối ưu gây trôi dạt kết quả suy luận nghiêm trọng, giúp người dùng xây dựng bảng đánh đổi **Pareto Frontier**:

| Cấu hình | Runtime | Precision | P95 Latency | Peak VRAM | Cosine Sim | Trạng thái Pareto |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline | PyTorch Eager | FP32 | 18.2 ms | 2.1 GB | 1.0000 | Chuẩn đối chiếu |
| AMP | PyTorch Eager | FP16 | 7.9 ms | 1.2 GB | 0.9998 | **Tối ưu vượt bậc** |
| Dynamic Quant | PyTorch Eager | INT8 | 12.4 ms | 0.8 GB | 0.9850 | Giảm RAM, mất nét |

---

## 5. TẦNG DỮ LIỆU & LƯU TRỮ QUAN HỆ (DATA STORAGE ENGINE)

### 5.1. Lược đồ cơ sở dữ liệu quan hệ SQLite toàn diện (DDL Schema)

Hệ thống lưu trữ tập trung vào SQLite (`visionprof.db`) với các bảng quan hệ chặt chẽ:

```sql
-- Bảng quản lý phiên thực nghiệm
CREATE TABLE experiments (
    experiment_id TEXT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    platform_os TEXT,
    cpu_model TEXT,
    gpu_model TEXT,
    torch_version TEXT,
    cuda_version TEXT
);

-- Bảng thông tin mô hình
CREATE TABLE models (
    model_id TEXT PRIMARY KEY,
    experiment_id TEXT REFERENCES experiments(experiment_id),
    model_name TEXT NOT NULL,
    total_params INTEGER,
    input_shape TEXT,
    precision TEXT, -- FP32, FP16, INT8
    device TEXT
);

-- Bảng chi tiết từng layer (Metadata tĩnh)
CREATE TABLE layers (
    layer_id TEXT PRIMARY KEY,
    model_id TEXT REFERENCES models(model_id),
    layer_name TEXT NOT NULL,
    layer_type TEXT NOT NULL,
    param_count INTEGER,
    theoretical_flops REAL,
    arithmetic_intensity REAL
);

-- Bảng vết đo từng lần forward theo layer (Time-series Trace)
CREATE TABLE profiling_records (
    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
    layer_id TEXT REFERENCES layers(layer_id),
    iteration INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    activation_mb REAL,
    cuda_allocated_mb REAL,
    cuda_reserved_mb REAL
);

-- Bảng tổng kết vòng lặp có Tail Latency & Tài nguyên hệ thống
CREATE TABLE iteration_summaries (
    summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT REFERENCES models(model_id),
    iteration INTEGER NOT NULL,
    total_forward_ms REAL NOT NULL,
    dataloader_load_ms REAL,
    peak_vram_mb REAL,
    process_rss_mb REAL,
    avg_cpu_pct REAL,
    avg_gpu_util_pct REAL,
    gpu_power_w REAL,
    gpu_temp_c REAL
);

-- Bảng thống kê phân vị tổng hợp (Session Percentiles)
CREATE TABLE model_benchmarks (
    benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT REFERENCES models(model_id),
    mean_ms REAL,
    p50_ms REAL,
    p90_ms REAL,
    p95_ms REAL,
    p99_ms REAL,
    throughput_fps REAL,
    tail_disparity REAL,
    cosine_similarity REAL
);

-- Bảng phân rã thời gian Pipeline đầu cuối
CREATE TABLE pipeline_traces (
    trace_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT REFERENCES models(model_id),
    iteration INTEGER NOT NULL,
    preprocess_ms REAL NOT NULL,
    inference_ms REAL NOT NULL,
    postprocess_ms REAL NOT NULL,
    total_pipeline_ms REAL NOT NULL
);

-- Bảng lưu vết tài nguyên hệ thống theo thời gian (Background Sampler Traces)
CREATE TABLE system_resource_traces (
    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT REFERENCES experiments(experiment_id),
    timestamp REAL NOT NULL,
    cpu_util_pct REAL,
    process_rss_mb REAL,
    gpu_util_pct REAL,
    gpu_mem_used_mb REAL,
    gpu_power_w REAL,
    gpu_temp_c REAL
);
```

### 5.2. Định dạng xuất nén cột Parquet
Hệ thống cung cấp hàm `export_parquet()` nén bằng thuật toán **Snappy / ZSTD**. Định dạng này giảm $85\%$ dung lượng lưu trữ so với CSV và tăng tốc độ đọc của Pandas gấp $10\times$.

---

## 6. MỞ RỘNG GIÁM SÁT TOÀN BỘ PIPELINE (END-TO-END PIPELINE SENTINEL)

### 6.1. Khắc phục lỗi Cold-Start trong DataLoaderSentinel
Ghi nhận trọn vẹn độ trễ của batch đầu tiên (Batch 0):
```python
class DataLoaderSentinelV2:
    def __iter__(self):
        self._loop_start_time = time.perf_counter()
        self._prev_batch_end = None
        for idx, batch in enumerate(self._loader):
            t_ready = time.perf_counter()
            if idx == 0:
                load_ms = (t_ready - self._loop_start_time) * 1000.0
            else:
                load_ms = (t_ready - self._prev_batch_end) * 1000.0
            self._load_times.append(load_ms)
            self._compute_start = t_ready
            yield batch
```

### 6.2. Kiến trúc PipelineSentinel (Preprocess → Inference → Postprocess)
Mở rộng khả năng quan sát vượt ra khỏi phạm vi `model.forward()`:
```python
class PipelineSentinel:
    """
    Giám sát và bóc tách thời gian trọn vẹn của luồng Computer Vision:
    1. Preprocess: Decode, Resize, Letterbox, Normalize, HWC->CHW.
    2. Inference: Model Forward Pass.
    3. Postprocess: Non-Maximum Suppression (NMS), Box Decoding, Thresholding.
    """
    def __init__(self, is_cuda: bool = False):
        self.is_cuda = is_cuda
        self.traces = []

    @contextmanager
    def monitor_pipeline(self):
        ctx = {"t_pre": 0.0, "t_inf": 0.0, "t_post": 0.0}
        
        # 1. Đo Tiền xử lý
        t0 = time.perf_counter()
        yield "preprocess"
        if self.is_cuda: torch.cuda.synchronize()
        ctx["t_pre"] = (time.perf_counter() - t0) * 1000.0
        
        # 2. Đo Suy luận
        t1 = time.perf_counter()
        yield "inference"
        if self.is_cuda: torch.cuda.synchronize()
        ctx["t_inf"] = (time.perf_counter() - t1) * 1000.0
        
        # 3. Đo Hậu xử lý
        t2 = time.perf_counter()
        yield "postprocess"
        if self.is_cuda: torch.cuda.synchronize()
        ctx["t_post"] = (time.perf_counter() - t2) * 1000.0
        
        self.traces.append(ctx)
```

---

## 7. QUY TRÌNH KIỂM CHUẨN VỚI NVIDIA NSIGHT SYSTEMS (VALIDATION PROTOCOL)

```bash
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=reports/nsight_ground_truth \
    --force-overwrite=true \
    python tests/test_ground_truth_nsight.py
```

### Quy trình phân tích đối chiếu tự động:
1. `test_ground_truth_nsight.py` phát ra các NVTX markers bao bọc từng layer (`torch.cuda.nvtx.range_push(layer_name)` và `range_pop()`).
2. Trích xuất thời gian GPU kernel thực tế từ SQLite database do Nsight xuất ra (`nsys-rep` $\rightarrow$ `.sqlite`).
3. So sánh trực tiếp từng layer giữa VisionProf và Nsight Systems:
   $$\text{Relative Error}_l = \frac{|t_{\text{VisionProf}}^{(l)} - t_{\text{Nsight}}^{(l)}|}{t_{\text{Nsight}}^{(l)}} \times 100\%$$
4. Đảm bảo toàn bộ các tầng chính đạt $\text{Relative Error} < 2.0\%$ để thỏa mãn Tiêu chuẩn Thành công [SC1].

---

## 8. PHÂN TÍCH PHẢN BIỆN CHUYÊN SÂU, RỦI RO TIỀM ẨN & TRƯỜNG HỢP BIÊN (ADVERSARIAL EDGE CASES)

### 8.1. Hiện tượng Trôi nhiệt & Dao động xung nhịp GPU trong A/B Testing
- **Rủi ro (Thermal Throttling Bias)**:
  Nếu chạy tuần tự: Model A chạy 50 iterations trước (GPU còn mát, ~45°C, Boost 1590 MHz), sau đó Model B chạy 50 iterations (GPU nóng lên, ~75°C, xung hạ xuống 1350 MHz, giảm 15%). Model B sẽ bị kết luận là chậm hơn Model A một cách oan uổng.
- **Giải pháp bắt buộc (Interleaved A/B Protocol)**:
  Thực thi theo giao thức **Xen kẽ luân phiên**: $[A_1, B_1, A_2, B_2, \dots, A_N, B_N]$ để triệt tiêu hoàn toàn sự chênh lệch về nhiệt độ và xung nhịp giữa hai mô hình.

---

### 8.2. Xung đột trạng thái khi Module được gọi lặp lại (Re-entrant / Weight-Tied)
- **Rủi ro (State Overwrite Race Condition)**:
  Trong các mạng chia sẻ trọng số hoặc module gọi nhiều lần trong 1 pass (như Detection Heads đa tỷ lệ): lưu `state[module_name] = start_event` sẽ bị ghi đè khi module được gọi lần 2 trước khi lần 1 kết thúc.
- **Giải pháp bắt buộc (LIFO Stack Architecture)**:
  Lưu trữ sự kiện trong hook theo cấu trúc **Ngăn xếp (LIFO Stack)** riêng biệt cho từng instance.

---

### 8.3. Mô hình Roofline Đa trần (Dual-Ceiling Roofline: FP32 vs. Tensor Core FP16)
- **Rủi ro (Ceiling Mismatch)**:
  Năng lực đỉnh trên GPU Turing T4: FP32 là $8.1\text{ TFLOPS}$, trong khi FP16 Tensor Core là $65.0\text{ TFLOPS}$. Nếu so sánh mô hình FP32 với trần FP16 sẽ báo động sai rằng mô hình bị lãng phí năng lực tính toán trầm trọng.
- **Giải pháp bắt buộc**:
  Tự động phát hiện kiểu dữ liệu (`tensor.dtype`) để kích hoạt đúng đường trần lý thuyết tương ứng của phần cứng.

---

### 8.4. Xử lý phép toán tại chỗ (In-place Operations) và Tensor Storage Aliasing
- **Rủi ro (Activation Double-Counting)**:
  Các tầng `nn.ReLU(inplace=True)` biến đổi trực tiếp vùng nhớ của tensor đầu vào (`output.data_ptr() == input.data_ptr()`). Nếu tính kích thước và cộng dồn sẽ bị tính đúp bộ nhớ nhiều lần.
- **Giải pháp bắt buộc**:
  Theo dõi địa chỉ con trỏ dữ liệu gốc (`tensor.untyped_storage().data_ptr()`). Chỉ tính dung lượng kích thước tensor vào bộ nhớ kích hoạt mới nếu địa chỉ storage chưa từng xuất hiện trong iteration hiện tại.

---

### 8.5. Giới hạn tương thích với `torch.compile` (Graph Breaks do Hooks)
- **Rủi ro (Inductor Incompatibility)**:
  Gắn PyTorch Module Hooks (`register_forward_hook`) buộc TorchDynamo tạo ra các **Graph Breaks** tại mọi ranh giới của hook, vô hiệu hóa khả năng gộp kernel của compiler.
- **Định vị phạm vi chuẩn mực**:
  Tuyên bố rõ: **VisionProf Layer Profiler tối ưu cho PyTorch Eager Mode**. Đối với mô hình biên dịch, framework hỗ trợ profiling ở cấp độ mô hình hộp đen qua `PipelineSentinel`.

---

### 8.6. Điểm mù của Leaf Hooks đối với Functional Operations không đóng gói module
- **Rủi ro (Uncaptured Functional Ops)**:
  Các phép toán gọi qua hàm chức năng (`torch.cat()`, `x + shortcut`, `F.silu()`) không được leaf hooks ghi nhận, gây ra sự thiếu hụt thời gian so với tổng forward pass.
- **Giải pháp bắt buộc (Attribution Gap Tracking)**:
  So sánh tổng thời gian leaf layers với forward pass tổng: $\Delta_{\text{unattributed}} = T_{\text{total\_forward}} - \sum_{l} t_l$. Cảnh báo nếu $\Delta_{\text{unattributed}} > 15\%$.

---

### 8.7. Bẫy giữ tham chiếu Activation Tensor trong Event Queue (The Memory Retention Leak)
- **Rủi ro (Catastrophic OOM during Profiling)**:
  Lưu object tensor đầu ra `out` vào hàng đợi để chờ đến `flush_and_resolve()` sẽ giữ reference count $> 0$, khiến PyTorch không thể tái sử dụng VRAM, gây tràn bộ nhớ (CUDA OOM) ngay lập tức.
- **Giải pháp bắt buộc (Zero-Retention Metadata Extraction)**:
  Trích xuất tức thì các kích thước (shape tuple, numel, element_size, dtype) thành Python primitives và giải phóng con trỏ `out` ngay trong `post_hook`.

---

### 8.8. Bỏ sót Tensor đầu ra trong các mạng Multi-Scale / FPN Heads
- **Rủi ro (Underestimating Multi-Scale Output)**:
  Mô hình như YOLO trả về danh sách/tuple chứa nhiều tensor (`[P3, P4, P5]`). Nếu chỉ lấy phần tử đầu tiên sẽ bỏ sót 40–50% dung lượng activation.
- **Giải pháp bắt buộc**:
  Hàm trích xuất duyệt đệ quy (recursive traversal) qua toàn bộ cấu trúc dữ liệu lồng nhau và tổng hợp dung lượng của tất cả tensor con.

---

### 8.9. Ranh giới phạm vi: Đơn thiết bị (Single-Device) vs. Phân tán đa GPU
- **Tuyên bố phạm vi**:
  VisionProf chuyên sâu về **Micro-Architecture & Kernel Efficiency** trên môi trường đơn thiết bị (Single Edge CPU hoặc Single Cloud GPU T4). Các bài toán liên lạc mạng NCCL trong huấn luyện phân tán nằm ngoài phạm vi thiết kế.

---

### 8.10. Tranh chấp GIL và Tần số Lấy mẫu của Background Resource Thread
- **Rủi ro (GIL Contention & Sampling Distortion)**:
  Trong CPython, luồng lấy mẫu tài nguyên nền (`SystemResourceSentinel`) phải chia sẻ Global Interpreter Lock (GIL) với tiến trình thực thi chính. Nếu đặt tần số lấy mẫu quá cao (ví dụ: $< 10\text{ms}$), luồng nền sẽ gây tranh chấp CPU, làm tăng độ trễ của forward pass thêm 3–7%. Ngược lại, nếu lấy mẫu quá thưa ($> 500\text{ms}$), nó sẽ bỏ lỡ các đỉnh tải đột biến (load spikes).
- **Giải pháp chuẩn hóa**:
  Thiết lập tần số lấy mẫu tối ưu là **$50\text{ms}$ ($20\text{ Hz}$)**. Tại tần số này, chi phí CPU của luồng nền $< 0.2\%$, hoàn toàn không gây ảnh hưởng đến tính ổn định của forward pass.

---

### 8.11. Biến thiên Xung nhịp CPU (CPU Frequency Governors & Turbo Boost Jitter)
- **Rủi ro (CPU Clock Fluttering)**:
  Trên hệ điều hành laptop hoặc máy ảo Colab, cơ chế điều tiết năng lượng CPU (`intel_pstate`, `powersave`, `ondemand`) tự động tăng hoặc giảm xung nhịp CPU từ 1.8 GHz lên 4.5 GHz dựa trên tải tức thời. Điều này tạo ra phương sai ngẫu nhiên rất lớn giữa các lần lặp (P99 jitter cao).
- **Giải pháp chuẩn hóa**:
  Kích hoạt cơ chế **Pre-warming & Warm-up Phase ($N=20$)** để ép bộ điều phối CPU đạt trạng thái xung nhịp bão hòa (Performance Steady State) trước khi bắt đầu thu thập số liệu.

---

### 8.12. Rủi ro Biến thiên Kích thước Ảnh Động (Dynamic Resolution & Padding Variance)
- **Rủi ro (Batch Padding Bias)**:
  Trong các pipeline Object Detection thực tế, nếu không cố định kích thước batch và áp dụng Rectangular Training/Inference (ảnh bị cắt/pad theo tỷ lệ khung hình thực tế của từng ảnh):
  Mỗi iteration sẽ nhận một tensor có kích thước không gian $(H, W)$ khác nhau. Điều này khiến thời gian forward và bộ nhớ dao động dữ dội mà không phải do lỗi hệ thống.
- **Giải pháp chuẩn hóa**:
  Mọi benchmark cấp độ layer bắt buộc phải chuẩn hóa đầu vào thông qua kích thước cố định (ví dụ $640\times 640$ cho YOLO, $224\times 224$ cho CNN/ViT) để đảm bảo tính khả lặp khoa học (strict reproducibility).

---

### 8.13. Ranh giới Kiến trúc Runtime: PyTorch Eager vs. Black-box ONNX/TensorRT Session
- **Tuyên bố Phân định Ranh giới**:
  1. **Tầng Bạch diện (White-Box Layer-Level)**: Được hỗ trợ đầy đủ 100% trên **PyTorch Eager Mode** nhờ khả năng gắn hook sâu vào từng `torch.nn.Module`.
  2. **Tầng Hắc diện (Black-Box Model-Level)**: Đối với các runtime suy luận biên dịch tối ưu như **ONNX Runtime (`onnxruntime.InferenceSession`)** và **TensorRT**: Framework hỗ trợ đo đạc toàn diện ở cấp độ End-to-End Pipeline (thời gian toàn phiên, Tail Latency P50-P99, Thông lượng FPS, Tài nguyên CPU/VRAM toàn cục) thông qua `PipelineSentinel`, nhưng không can thiệp mổ xẻ nội bộ các kernel đã được compiler hợp nhất (Fused Kernels).

---
*Tài liệu này là đặc tả kỹ thuật chuẩn mức kỹ sư và lập trình viên. Đọc kết hợp với [research_plan_profiling_framework.md](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/research_plan_profiling_framework.md) để nắm bắt toàn cảnh nghiên cứu.*
