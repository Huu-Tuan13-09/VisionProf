# 04. Tính Chuẩn Xác Khoa Học & Quy Tắc Đo Hiệu Năng (Scientific Rigor & Profiling Rules)

Tài liệu này tổng hợp toàn bộ các yêu cầu phương pháp luận và các lỗi kỹ thuật cốt tử đã được chỉ ra trong `SYSTEM_AUDIT_REPORT.md` và định hướng giải quyết chuẩn tắc trong `PROJECT_PLAN.md`.

---

## 1. Đo Thời Gian GPU Bằng Kỹ Thuật Deferred CUDA Events

### 1.1. Các Lỗi Nghiêm Cấm (Anti-Patterns)
- ❌ **CẤM**: Dùng `time.perf_counter()` trên CPU để đo thời gian các layer trên GPU (PyTorch gọi CUDA bất đồng bộ, chỉ đo ~2µs thời gian CPU enqueue).
- ❌ **CẤM**: Gọi `torch.cuda.synchronize()` bên trong hook của từng layer (gây GPU pipeline starvation, làm rỗng hàng đợi kernel và tăng overhead lên 200–500%).

### 1.2. Giải Pháp Bắt Buộc: Device-Aware Timing Dispatcher
- **Nếu chạy trên GPU (`device.type == 'cuda'`)**:
  - Dùng **Deferred CUDA Events** kết hợp LIFO Stack cho các submodules được gọi lặp lại.
  - Pre-hook: `start_evt.record(stream)`.
  - Post-hook: `end_evt.record(stream)`.
  - **Chỉ gọi `torch.cuda.synchronize()` duy nhất một lần ở cuối toàn bộ forward pass**, sau đó lặp qua danh sách sự kiện để tính `elapsed_time()`.
- **Nếu chạy trên CPU (`device.type == 'cpu'`)**:
  - Dùng `time.perf_counter_ns()` (độ chính xác nano-giây) và quy đổi sang mili-giây (`ms`).

### 1.3. Hiệu Chuẩn Overhead (`calibrate_overhead`)
- Đếm chính xác số leaf layer thực tế được gắn hook (không tính module cha/container tránh chia sai mẫu số).

---

## 2. Quy Chuẩn Khởi Động (Warm-Up) & Phân Phối Trễ (Tail Latency)

### 2.1. Số Lượng Vòng Lặp Khởi Động & Đo Chuẩn Xác
- **Inference**:
  - Khởi động (Warm-up): Tối thiểu **10–20 vòng** (không ghi số liệu) để ổn định cuDNN autotune, nạp cache L2/L3, kích hoạt GPU clock boost.
  - Đo chính thức: **50 vòng**.
- **Training**:
  - Khởi động: **5 vòng**.
  - Đo chính thức: **30 vòng**.

### 2.2. Phân Phối Độ Trễ Đầy Đủ (Tail Latency Metrics)
- ❌ Không chỉ đo `mean` và `std`.
- Bắt buộc tính toán và lưu trữ:
  - **P50 (Median)**: Đại diện cho hiệu năng ổn định thông thường.
  - **P90, P95, P99**: Đánh giá độ trễ đuôi (tail latency spikes).
  - **Tail Ratio**: $TDR = P99 / P50$ (đánh giá độ trồi sụt / jitter).
  - **Throughput**: $FPS = \frac{\text{Batch Size} \times 1000}{\text{Mean (ms)}}$.
  - **Cold-start**: Thời gian của vòng chạy đầu tiên (iteration 0).

---

## 3. Quản Lý Bộ Nhớ: Tách Biệt Process RSS vs Activation Memory

- ❌ **CẤM**: Gán bộ nhớ `process_rss` (lấy từ `psutil`) vào từng layer (gây false positive 100% cho rule `RA-04` trên CPU).
- **Quy chuẩn phân định**:
  - **GPU Memory**: Đo qua `torch.cuda.max_memory_allocated()` và `torch.cuda.memory_reserved()`.
  - **Activation Memory**: Ước tính từ shape và dtype tensor đầu ra của layer:
    $$\text{Activation MB} = \frac{\prod \text{dims} \times \text{element\_size\_bytes}}{1024^2}$$
  - **Host Memory (Process RSS)**: Chỉ đo ở cấp độ tiến trình toàn cục trong `ResourceSampler`.

---

## 4. Giám Sát Tài Nguyên: Cấm NVML Trong Hook

- ❌ **CẤM**: Gọi C-API của NVML (`pynvml`) hoặc `psutil` trong forward/backward hook (mất 5–10ms/lần gọi).
- **Giải pháp**: Xây dựng `ResourceSampler` chạy trên một **Background Daemon Thread** độc lập, lấy mẫu đều đặn mỗi 100ms và ghi ra `resources.csv`.

---

## 5. Sửa 4 Luật Chẩn Đoán Sai Lệch (Heuristics v2.0)

| Mã luật | Tên cũ | Sai lầm v1.0 | Sửa chuẩn trong v2.0 |
| :--- | :--- | :--- | :--- |
| **RB-04** | Time per Param | Dùng $\text{ms / Mparam} > 5.0$, phạt nhầm Conv2d đầu (ảnh to, param ít). | Đổi thành **$\text{ms / GFLOP}$** (đọc FLOPs từ `layers.csv`) hoặc Roofline % peak. |
| **RD-05** | Param-Time Outliers | Dùng tỷ số $\text{ms / Mparam}$, phạt nhầm BatchNorm, LayerNorm. | Phân tích theo mật độ tính toán FLOPs thay vì số lượng weights. |
| **RD-02** | Dead Layer | Báo động layer $< 0.001\text{ms}$ là dead layer và khuyên xóa. | Đổi thành **Sub-microsecond Layer**: Thông báo layer rất nhẹ hoặc identity/view operation. |
| **RC-03** | Compute Starvation | Giải thích ngược nghĩa và trùng lặp với RC-01. | Xác định GPU idled waiting for batch (GPU util $\approx 0\%$ khi nạp batch tiếp theo). |

---

## 6. Sửa Thuật Toán So Sánh A/B & DataLoader Sentinel

### 6.1. Speedup A/B Chuẩn Xác
- ❌ **CẤM**: Lấy trung bình cộng tỷ lệ từng layer: $\text{mean}(\text{time}_{A, i} / \text{time}_{B, i})$.
- **BẮT BUỘC**: Tính Speedup trên **tổng thời gian thực thi của toàn bộ mô hình trên từng iteration**:
  $$\text{Speedup} = \frac{\sum_{i} \text{TotalTime}_{A}^{(i)}}{\sum_{i} \text{TotalTime}_{B}^{(i)}}$$
- Áp dụng kiểm định phi tham số Mann-Whitney U trên 2 phân phối latency.

### 6.2. Sửa `DataLoaderSentinel`
- Khắc phục lỗi bỏ sót batch đầu tiên khi lặp qua generator của DataLoader.

---

## 7. Tiêu Chí Nghiệm Thu Thực Nghiệm (Validation Criteria)

- **Sai số so với `torch.profiler`**: Tổng thời gian layer của VisionProf phải lệch **$< 10\%$** so với `torch.profiler`.
- **Overhead của profiler**: Thời gian chạy có hook so với không hook làm chậm đi **$< 5\%$** trên GPU và **$< 2\%$** trên CPU.
- **Quy tắc đo khử nhiễu**: Cắm sạc laptop, bật high performance, cố định threads, so sánh A/B phải chạy **xen kẽ A-B-A-B** để triệt tiêu ảnh hưởng nhiệt độ (thermal throttling).
