# BÁO CÁO KIỂM TOÁN TOÀN DIỆN HỆ THỐNG VISIONPROF (v3.0)
## (Comprehensive System Audit, Research Plan Gap Analysis & Architectural Roadmap)

> **Dự án**: VisionProf (Cross-Platform PyTorch Layer Profiler & Automated Heuristic Bottleneck Analyzer)  
> **Phiên bản kiểm toán**: Nhánh `main` (commit hiện tại)  
> **Tài liệu đối chiếu**:
> - Đề cương nghiên cứu: [`research_plan_profiling_framework.md`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/research_plan_profiling_framework.md) (v3.5)
> - Đặc tả kỹ thuật chi tiết: [`TECHNICAL_SPECIFICATION.md`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/TECHNICAL_SPECIFICATION.md) (v2.5)
> - Tài liệu kỹ thuật: [`README.md`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/README.md)
> - Mã nguồn hệ thống: [`src/collector.py`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/src/collector.py), [`src/analyzer.py`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/src/analyzer.py), [`src/dashboard.py`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/src/dashboard.py)
> - Kịch bản kiểm thử: [`tests/test_modern_models_benchmark.py`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/tests/test_modern_models_benchmark.py), [`tests/test_training_bottleneck.py`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/tests/test_training_bottleneck.py)
> - Dữ liệu thực nghiệm: [`data/profiles/`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/data/profiles), [`reports/`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/reports)

---

## MỤC LỤC
1. [Tóm Tắt Điều Hành (Executive Summary)](#1-tóm-tắt-điều-hành-executive-summary)
2. [Độ Vênh Giữa Đề Cương, Đặc Tả và Mã Nguồn Thực Tế (Gaps Analysis)](#2-độ-vênh-giữa-đề-cương-đặc-tả-và-mã-nguồn-thực-tế-gaps-analysis)
   - [2.1. Vắng bóng hoàn toàn Roofline Model và Arithmetic Intensity trong Code](#21-vắng-bóng-hoàn-toàn-roofline-model-và-arithmetic-intensity-trong-code)
   - [2.2. Sự trôi dạt (Drift) của Bộ Quy tắc Heuristic: Từ chuẩn hệ thống sang 19 quy tắc tự chế](#22-sự-trôi-dạt-drift-của-bộ-quy-tắc-heuristic-từ-chuẩn-hệ-thống-sang-19-quy-tắc-tự-chế)
   - [2.3. Nghịch lý Synchronization trong đo lường GPU Hooks](#23-nghịch-lý-synchronization-trong-đo-lường-gpu-hooks)
   - [2.4. Hạ chuẩn tầng lưu trữ: Từ SQLite/Parquet xuống Flat JSON/CSV](#24-hạ-chuẩn-tầng-lưu-trữ-từ-sqliteparquet-xuống-flat-jsoncsv)
   - [2.5. Không có Ground Truth Validation với NVIDIA Nsight CLI](#25-không-có-ground-truth-validation-với-nvidia-nsight-cli)
   - [2.6. Bỏ trống 100% các nghiên cứu thực nghiệm Ablation (AB-1 đến AB-7)](#26-bỏ-trống-100-các-nghiên-cứu-thực-nghiệm-ablation-ab-1-đến-ab-7)
   - [2.7. Rút gọn Warm-up tùy tiện: Từ 20 iterations xuống còn 3 iterations](#27-rút-gọn-warm-up-tùy-tiện-từ-20-iterations-xuống-còn-3-iterations)
   - [2.8. Khuyết thiếu hoàn toàn Tail Latency (P50, P90, P95, P99) và Throughput trong Code](#28-khuyết-thiếu-hoàn-toàn-tail-latency-p50-p90-p95-p99-và-throughput-trong-code)
   - [2.9. Sai lệch trong Giám sát Tài nguyên: Gọi NVML trong Hook thay vì Background Thread](#29-sai-lệch-trong-giám-sát-tài-nguyên-gọi-nvml-trong-hook-thay-vì-background-thread)
   - [2.10. Vắng bóng Kịch bản Thực nghiệm End-to-End Pipeline CV (Preprocess vs. Postprocess)](#210-vắng-bóng-kịch-bản-thực-nghiệm-end-to-end-pipeline-cv-preprocess-vs-postprocess)
3. [Chi Tiết Các Lỗi Kỹ Thuật & Sai Lệch Phương Pháp Luận Trong Codebase](#3-chi-tiết-các-lỗi-kỹ-thuật--sai-lệch-phương-pháp-luận-trong-codebase)
   - [3.1. Đo thời gian GPU bị sai lệch hoàn toàn (Dùng CPU time.perf_counter)](#31-đo-thời-gian-gpu-bị-sai-lệch-hoàn-toàn-dùng-cpu-timeperf_counter)
   - [3.2. Thuật toán trừ hao sai số (calibrate_overhead) bị lỗi logic đếm tầng](#32-thuật-toán-trừ-hao-sai-số-calibrate_overhead-bị-lỗi-logic-đếm-tầng)
   - [3.3. Nhập nhằng bộ nhớ: Trộn lẫn Process RSS với Activation Memory trên CPU](#33-nhập-nhằng-bộ-nhớ-trộn-lẫn-process-rss-với-activation-memory-trên-cpu)
   - [3.4. Sai lệch toán học & thống kê nghiêm trọng trong A/B Comparison Engine](#34-sai-lệch-toán-học--thống-kê-nghiêm-trọng-trong-ab-comparison-engine)
   - [3.5. Lỗi định nghĩa và ngộ nhận lý thuyết trong các quy tắc Heuristics](#35-lỗi-định-nghĩa-và-ngộ-nhận-lý-thuyết-trong-các-quy-tắc-heuristics)
   - [3.6. Khiếm khuyết trong DataLoaderSentinel](#36-khiếm-khuyết-trong-dataloadersentinel)
4. [Kiểm Toán Chi Tiết Ma Trận 19 Quy Tắc Heuristic (Rule-by-Rule Audit Matrix)](#4-kiểm-toán-chi-tiết-ma-trận-19-quy-tắc-heuristic-rule-by-rule-audit-matrix)
5. [Đối Chiếu Với Chuẩn Mực Của Một AI Profiling Framework Toàn Diện](#5-đối-chiếu-với-chuẩn-mực-của-một-ai-profiling-framework-toàn-diện)
   - [5.1. Phân định Ba Tầng: Evaluation vs. Benchmarking vs. Profiling](#51-phân-định-ba-tầng-evaluation-vs-benchmarking-vs-profiling)
   - [5.2. Phạm vi Pipeline: Thiếu Preprocessing & Postprocessing (NMS)](#52-phạm-vi-pipeline-thiếu-preprocessing--postprocessing-nms)
   - [5.3. Phân phối độ trễ: Tầm quan trọng của Tail Latency (P50, P90, P95, P99)](#53-phân-phối-độ-trễ-tầm-quan-trọng-của-tail-latency-p50-p90-p95-p99)
   - [5.4. Đánh đổi Hiệu năng vs. Độ chính xác (Pareto Frontier)](#54-đánh-đổi-hiệu-năng-vs-độ-chính-xác-pareto-frontier)
   - [5.5. Giới hạn Runtime: PyTorch Eager vs. ONNX Runtime / TensorRT](#55-giới-hạn-runtime-pytorch-eager-vs-onnx-runtime--tensorrt)
   - [5.6. Tài nguyên Phần cứng Mở rộng (Power, Thermal Throttling, Edge Metrics)](#56-tài-nguyên-phần-cứng-mở-rộng-power-thermal-throttling-edge-metrics)
6. [Lộ Trình Cải Tiến & Nâng Cấp Khả Thi (Actionable Roadmap)](#6-lộ-trình-cải-tiến--nâng-cấp-khả-thi-actionable-roadmap)

---

## 1. TÓM TẮT ĐIỀU HÀNH (EXECUTIVE SUMMARY)

Hệ thống **VisionProf** được định vị trong đề cương nghiên cứu ([`research_plan_profiling_framework.md`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/research_plan_profiling_framework.md) v3.5) và tài liệu đặc tả ([`TECHNICAL_SPECIFICATION.md`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/TECHNICAL_SPECIFICATION.md) v2.5) như một hệ thống profiling tiên tiến kết hợp chẩn đoán tự động cho các mô hình thị giác máy tính:
1. Profiling đa nền tảng (CPU Laptop $\leftrightarrow$ GPU T4 Colab) với độ phân giải cấp tầng (layer-level).
2. Phân loại điểm nghẽn bằng mô hình Roofline (Compute vs. Memory Bound) và 19 quy tắc Heuristics v2.0.
3. Kỹ thuật Deferred CUDA Events triệt tiêu GPU bubble giữ overhead $< 2\%$.
4. Giám sát toàn diện từ End-to-End Pipeline đến Tài nguyên hệ thống toàn cục (CPU, RAM, GPU, Power, Temp).

Tuy nhiên, qua việc **kiểm toán chéo toàn diện giữa Tài liệu Thiết kế và Mã nguồn thực tế triển khai (Codebase)**, chúng tôi phát hiện:
- **Độ vênh thực thi (Implementation Gap) cực kỳ lớn**: Toàn bộ các thuật toán cốt lõi đã đặc tả chi tiết trong `TECHNICAL_SPECIFICATION.md` (Deferred CUDA Events, LIFO Stack, fvcore FLOPs, SQLite DDL, SystemResourceSentinel, sửa đổi Speedup và Heuristics) **CHƯA HỀ ĐƯỢC CHUYỂN ĐỔI VÀO MÃ NGUỒN `src/`**.
- Codebase hiện tại trong `src/collector.py`, `src/analyzer.py` vẫn là bản **v1.0 cũ** với đầy đủ các lỗi kỹ thuật nghiêm trọng (đo GPU bằng CPU clock, chia trung bình layer để tính speedup, lưu JSON/CSV phẳng).
- Nếu mang mã nguồn hiện tại đi chạy thực nghiệm, toàn bộ số liệu đo GPU latency, RAM activation và kết quả chẩn đoán A/B sẽ **hoàn toàn vô giá trị về mặt khoa học**.

---

## 2. ĐỘ VÊNH GIỮA ĐỀ CƯƠNG, ĐẶC TẢ VÀ MÃ NGUỒN THỰC TẾ (GAPS ANALYSIS)

### 2.1. Vắng bóng hoàn toàn Roofline Model và Arithmetic Intensity trong Code
- **Đặc tả & Đề cương**: Bắt buộc tính FLOPs giải tích, Cường độ số học (FLOPs/Byte), vẽ biểu đồ Roofline trên Streamlit để phân loại Compute-bound vs Memory-bound.
- **Thực tế trong Mã nguồn**: Không có bất kỳ dòng code nào trong `src/` tính toán FLOPs hoặc dựng Roofline. `LayerRecord` hoàn toàn thiếu vắng các trường này.
- **Hệ quả**: Giả thuyết H1 không thể kiểm chứng được nếu chạy trên code hiện tại.

### 2.2. Sự trôi dạt (Drift) của Bộ Quy tắc Heuristic: Từ chuẩn hệ thống sang 19 quy tắc tự chế
- **Đặc tả**: 19 quy tắc v2.0 loại bỏ hoàn toàn $\text{ms / Mparam}$, thay bằng $\text{ms / GFLOP}$ và hiệu suất Roofline; đổi Dead Layer thành Sub-microsecond Layer; sửa lỗi định nghĩa Compute Starvation.
- **Thực tế trong Mã nguồn**: Code [`src/analyzer.py`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/src/analyzer.py) vẫn chứa các quy tắc cũ sai lầm: `RB-04` và `RD-05` vẫn dùng $\text{ms / Mparam}$, `RD-02` vẫn coi layer chạy $< 0.001\text{ms}$ là dead layer và khuyên xóa bỏ.

### 2.3. Nghịch lý Synchronization trong đo lường GPU Hooks
- **Đặc tả**: Áp dụng kỹ thuật Deferred Synchronization: ghi nhận `torch.cuda.Event` bất đồng bộ trong hook, chỉ đồng bộ hóa `torch.cuda.synchronize()` một lần duy nhất ở cuối forward pass.
- **Thực tế trong Mã nguồn**: [`src/collector.py` Dòng 271](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/src/collector.py#L271) đo bằng `time.perf_counter()` trên CPU, chỉ đo thời gian enqueue lệnh (~2µs), hoàn toàn sai lệch thời gian thực tế của GPU.

### 2.4. Hạ chuẩn tầng lưu trữ: Từ SQLite/Parquet xuống Flat JSON/CSV
- **Đặc tả**: Lược đồ quan hệ SQLite (`visionprof.db`) chuẩn hóa 7 bảng quan hệ và xuất Parquet nén cột.
- **Thực tế trong Mã nguồn**: [`src/collector.py` Dòng 11](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/src/collector.py#L11) ghi rõ *"Lưu trữ: JSON + CSV (pandas) — không dùng SQLite"*.

### 2.5. Không có Ground Truth Validation với NVIDIA Nsight CLI
- **Đặc tả & Đề cương**: Viết kịch bản đối chiếu NVTX marker với trace SQLite từ `nsys profile` (sai số $< 2\%$).
- **Thực tế trong Mã nguồn**: Hoàn toàn vắng bóng script validation này trong `tests/`.

### 2.6. Bỏ trống 100% các nghiên cứu thực nghiệm Ablation (AB-1 đến AB-7)
- **Đặc tả & Đề cương**: 7 bài toán bóc tách thành phần bắt buộc cho bài báo MLSys.
- **Thực tế trong Mã nguồn**: Chưa có script thực thi nào được viết trong codebase.

### 2.7. Rút gọn Warm-up tùy tiện: Từ 20 iterations xuống còn 3 iterations
- **Đặc tả & Đề cương**: $N_{warmup} = 20$ để ổn định Caching Allocator và cuDNN autotune.
- **Thực tế trong Mã nguồn**: [`tests/test_modern_models_benchmark.py`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/tests/test_modern_models_benchmark.py#L80) đặt `N_WARMUP = 3`.

### 2.8. Khuyết thiếu hoàn toàn Tail Latency (P50, P90, P95, P99) và Throughput trong Code
- **Đặc tả v2.5**: Bắt buộc tính P50, P90, P95, P99, Throughput FPS, và Tail Disparity Ratio.
- **Thực tế trong Mã nguồn**: Schema `ProfileSession` và `collector.py` chỉ tính `mean` và `std`, che giấu hoàn toàn hiện tượng jitter và trễ đuôi (tail latency spikes).

### 2.9. Sai lệch trong Giám sát Tài nguyên: Gọi NVML trong Hook thay vì Background Thread
- **Đặc tả v2.5**: Tách thành `SystemResourceSentinel` chạy luồng nền 50ms đo CPU %, RAM, GPU Util %, Power, Temp.
- **Thực tế trong Mã nguồn**: Code cũ gọi hàm NVML C-API đồng bộ bên trong hook của từng layer, vừa gây trễ cho hook vừa vô nghĩa vì NVML lấy mẫu ở chu kỳ 100-1000ms.

### 2.10. Vắng bóng Kịch bản Thực nghiệm End-to-End Pipeline CV (Preprocess vs. Postprocess)
- **Đặc tả v2.5**: Thiết kế `PipelineSentinel` và Scenario 6 bóc tách Preprocess $\rightarrow$ Inference $\rightarrow$ Postprocess NMS.
- **Thực tế trong Mã nguồn**: Chưa có bất kỳ module nào triển khai `PipelineSentinel`, thư mục `tests/` chỉ chạy `model.forward()`.

---

## 3. CHI TIẾT CÁC LỖI KỸ THUẬT & SAI LỆCH PHƯƠNG PHÁP LUẬN TRONG CODEBASE

### 3.1. Đo thời gian GPU bị sai lệch hoàn toàn (Dùng CPU time.perf_counter)
- **Vị trí**: [`src/collector.py` (Dòng 256–274)](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/src/collector.py#L256-L274)
- **Hậu quả**: Tất cả số liệu đo GPU latency trên Colab T4 hoặc GPU cục bộ đều là thời gian CPU enqueue lệnh, hoàn toàn là số liệu rác.

### 3.2. Thuật toán trừ hao sai số (calibrate_overhead) bị lỗi logic đếm tầng
- **Vị trí**: [`src/collector.py` (Dòng 469–475)](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/src/collector.py#L469-L481)
- **Hậu quả**: Mẫu số chỉ đếm tầng có tham số, trong khi hook gắn vào toàn bộ leaf modules. Overhead bị thổi phồng gấp 2 lần, làm các tầng chạy nhanh bị trừ âm và ép về `0.0000 ms`.

### 3.3. Nhập nhằng bộ nhớ: Trộn lẫn Process RSS với Activation Memory trên CPU
- **Vị trí**: [`src/collector.py` (Dòng 264, 287)](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/src/collector.py#L264), [`src/analyzer.py` (Dòng 438)](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/src/analyzer.py#L438)
- **Hậu quả**: Toàn bộ các layer trên CPU đều bị gán dung lượng RAM ~500MB của cả tiến trình Python, dẫn đến Rule `RA-04` liên tục báo động sai 100% trên CPU.

### 3.4. Sai lệch toán học & thống kê nghiêm trọng trong A/B Comparison Engine
- **Vị trí**: [`src/analyzer.py` (Dòng 1582–1590, 1684–1691)](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/src/analyzer.py#L1582-L1600)
- **Hậu quả**: Lấy trung bình cộng của các layer đơn lẻ chia cho nhau. Một mô hình có 100 layer nhẹ có thể chậm hơn mô hình 10 layer nặng gấp đôi, nhưng VisionProf lại kết luận mô hình 100 layer nhanh hơn gấp 5 lần!

### 3.5. Lỗi định nghĩa và ngộ nhận lý thuyết trong các quy tắc Heuristics
1. `RD-02`: Đánh đồng layer chạy nhanh $< 0.001\text{ms}$ là dead layer và khuyên xóa bỏ.
2. `RB-04` & `RD-05`: Dùng tỷ số $\text{ms / Mparam}$ báo động sai hầu hết các tầng Conv2d đầu tiên.
3. `RC-03`: Giải thích ngược nghĩa Compute Starvation và trùng lặp điều kiện với `RC-01`.

### 3.6. Khiếm khuyết trong DataLoaderSentinel
- Bỏ sót Batch 0 (Cold-start drop) và lỗi Silent Fail nếu người dùng quên gọi `mark_compute_end()`.

---

## 4. KIỂM TOÁN CHI TIẾT MA TRẬN 19 QUY TẮC HEURISTIC (RULE-BY-RULE AUDIT MATRIX)

| Rule ID | Tên quy tắc | Nhóm | Hiện trạng logic trong Code | Đánh giá tính đúng đắn khoa học | Nguy cơ False Positive | Giải pháp đã đặc tả trong Spec v2.5 |
| :--- | :--- | :---: | :--- | :--- | :---: | :--- |
| **RA-01** | OOM Risk Detection | Memory | `peak_mb / total_gpu_mem_mb >= 85%` | Hợp lý trên GPU, bỏ qua RAM CPU | Thấp trên GPU, Vô hiệu trên CPU | Bổ sung kiểm tra `psutil.virtual_memory()`. |
| **RA-02** | Memory Fragmentation | Memory | `max(gpu_mem_delta_mb) > 500 MB` | Sai lý thuyết (đây là single alloc lớn) | Trung bình | Đổi thành `reserved - allocated` của PyTorch allocator. |
| **RA-03** | Allocation Hotspot | Memory | Top-5 layer chiếm $> 50\%$ tổng mem | Ngưỡng quá nhạy với mạng nơ-ron sâu | **Rất cao** | Nâng ngưỡng hoặc xét trên phân phối Pareto. |
| **RA-04** | Peak Memory Efficiency | Memory | `peak_mb / params_M > 50.0` | Sai nghiêm trọng trên CPU do trộn Process RSS | **100% trên CPU** | Tách riêng Activation Memory thuần túy khỏi Process RSS. |
| **RA-05** | Memory Leak Heuristic | Memory | Hồi quy tuyến tính slope $> 1.0$ & $R^2 > 0.85$ | Dễ nhầm với warm-up ở các iter đầu | Cao khi $N < 20$ | Bỏ qua 15 iterations đầu tiên trước khi tính hồi quy. |
| **RB-01** | Layer Speed Imbalance | Compute | Top-5 layer chiếm $> 50\%$ tổng thời gian | Định luật Amdahl/Pareto là tự nhiên | Cao | Đổi thành thông tin mô tả tỷ trọng (informative metric). |
| **RB-02** | Compute Variance | Compute | $CV = \text{std} / \text{mean} > 0.5$ | Dễ nhiễu bởi background task trên OS | Cao khi $N < 30$ | Tăng số iter và dùng MAD/IQR kháng ngoại lai. |
| **RB-03** | Serial Bottleneck | Compute | 1 layer đơn lẻ chiếm $> 30\%$ tổng thời gian | Hợp lý, nhận diện tốt bottleneck dị biệt | Thấp | Giữ nguyên, bổ sung điều kiện loại trừ nếu $L < 5$. |
| **RB-04** | Time per Param Efficiency | Compute | $\text{ms/Mparam} > 5.0$ | Sai cơ bản đối với CNN (bỏ qua ảnh lớn) | **Rất cao với Early CNN**| Thay bằng **Hiệu suất Roofline** hoặc ms / GFLOP. |
| **RB-05** | GPU Utilization Drop | Compute | Gọi NVML trong hook tìm điểm giảm $> 40\%$ | NVML không thể đo chính xác cấp micro-giây | Cao | Chuyển ra luồng nền `SystemResourceSentinel`. |
| **RC-01** | DataLoader I/O Bottleneck | I/O | $\text{load\_time} / \text{total\_time} > 40\%$ | Hợp lý, phát hiện chuẩn nghẽn nạp đĩa | Thấp | Giữ nguyên, sửa lỗi bỏ sót Batch 0. |
| **RC-02** | DataLoader Variance | I/O | $CV(\text{load\_time}) > 0.8$ | Hợp lý, phát hiện đĩa bị nghẽn ngẫu nhiên | Thấp | Giữ nguyên. |
| **RC-03** | Compute Starvation | I/O | $\text{compute\_ratio} < 0.1$ | Hiểu sai ngữ nghĩa và trùng lặp RC-01 | Cao (Trùng lặp) | Định nghĩa lại theo thời gian GPU rảnh rỗi chờ I/O. |
| **RC-04** | Zero-Worker Warning | I/O | Cảnh báo khi `num_workers == 0` | Rất thực tế cho người mới bắt đầu | Không | Giữ nguyên. |
| **RD-01** | Parameter Concentration | Arch | 1 layer giữ $> 70\%$ tổng tham số | Hợp lý cho tầng FC/Embedding quá khổ | Thấp | Giữ nguyên. |
| **RD-02** | Dead Layer Detection | Arch | $\text{forward\_time} < 0.001 \text{ ms}$ | Sai hoàn toàn: Layer nhanh bị coi là layer chết| **Rất cao** | Đổi tên thành "Sub-microsecond Layer", bỏ khuyên xóa. |
| **RD-03** | Layer Type Diversity | Arch | 1 loại layer chiếm $> 80\%$ tổng số | Phán đoán chủ quan | Trung bình | Đổi thành thông tin mô tả cấu trúc. |
| **RD-04** | Depth Profile | Arch | Thống kê số lượng layer và depth | Hợp lý (mang tính thông tin) | Không | Giữ nguyên. |
| **RD-05** | Param-Time Outliers | Arch | $\text{ms/Mparam} > 3 \times \text{median}$ | Mắc cùng sai lầm với RB-04 | **Rất cao** | Chuyển sang so sánh theo phân phối FLOPs. |

---

## 5. ĐỐI CHIẾU VỚI CHUẨN MỰC CỦA MỘT AI PROFILING FRAMEWORK TOÀN DIỆN

### 5.1. Phân định Ba Tầng: Evaluation vs. Benchmarking vs. Profiling
Một hệ thống quan sát AI chuẩn mực phải trả lời được cả 3 câu hỏi:
1. **Evaluation**: Độ chính xác (Top-1, mAP).
2. **Benchmarking**: Tốc độ tổng thể (Throughput, Tail Latency P95/P99).
3. **Profiling**: Nguyên nhân vi mô (Layer-level, Roofline, Operator breakdown).
VisionProf hiện tại tập trung rất mạnh vào **Profiling vi mô**, nhưng đã bỏ rơi hoàn toàn tầng **Evaluation**. Khắc phục bằng cách tích hợp module đo **Output Divergence (Cosine Sim & L1)** để xây dựng bảng đánh đổi **Pareto Frontier**.

### 5.2. Phạm vi Pipeline: Thiếu Preprocessing & Postprocessing (NMS)
Inference chỉ chiếm 30–50% thời gian thực tế của bài toán thị giác. Phần lớn thời gian bị nghẽn ở:
- **Preprocessing**: Đọc video, giải mã ảnh, Resize, Letterboxing, Chuẩn hóa màu, HWC $\rightarrow$ CHW.
- **Postprocessing**: Non-Maximum Suppression (NMS), Box Decoding.
Cần đưa `PipelineSentinel` từ bản đặc tả vào thực nghiệm chính thức.

### 5.3. Phân phối độ trễ: Tầm quan trọng của Tail Latency (P50, P90, P95, P99)
Trong Production Serving, P99 latency quyết định SLA hệ thống có bị vi phạm hay không. Việc bổ sung đầy đủ vector phân vị P50, P90, P95, P99 vào SQLite schema và báo cáo là yêu cầu bắt buộc.

### 5.4. Đánh đổi Hiệu năng vs. Độ chính xác (Pareto Frontier)
Khi chuyển đổi mô hình từ FP32 sang FP16 AMP hoặc INT8, người dùng bắt buộc phải biết: *"Mô hình nhanh hơn $2\times$ nhưng độ chính xác có bị giảm không?"*.

### 5.5. Giới hạn Runtime: PyTorch Eager vs. ONNX Runtime / TensorRT
Ranh giới thiết kế phải rõ ràng:
- **PyTorch Eager**: Hỗ trợ White-box Layer-level profiling bằng hooks.
- **ONNX Runtime / TensorRT**: Hỗ trợ Black-box Pipeline/Model-level profiling thông qua `PipelineSentinel`.

### 5.6. Tài nguyên Phần cứng Mở rộng (Power, Thermal Throttling, Edge Metrics)
Nhiệt độ (°C) và Công suất (Watts) là hai chỉ số sống còn trên Edge AI để phát hiện hiện tượng tự tụt xung nhịp bảo vệ phần cứng (Thermal Throttling). Được thu thập an toàn qua luồng nền `SystemResourceSentinel`.

---

## 6. LỘ TRÌNH CẢI TIẾN & NÂNG CẤP KHẢ THI (ACTIONABLE ROADMAP)

Lộ trình được phân kỳ thành 2 lộ trình rõ ràng:

```
┌────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 1: KHẮC PHỤC CỐT LÕI & CHUẨN HÓA ĐO LƯỜNG (Tuần 1 - Tuần 2)  │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Triển khai Deferred CUDA Events & LIFO Stack trong collector.py.    │
│ 2. Triển khai SystemResourceSentinel (CPU, RAM, GPU, Power, Temp).     │
│ 3. Sửa thuật toán calibrate_overhead (đếm đủ leaf modules).            │
│ 4. Tách biệt Process RSS và Activation RAM trên CPU.                  │
│ 5. Sửa công thức A/B Comparison Engine (tính speedup theo full model). │
│ 6. Bổ sung tính toán Tail Latency (P50, P90, P95, P99) và Throughput.   │
│ 7. Chuyển đổi tầng lưu trữ sang SQLite DDL Schema và xuất Parquet.    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 2: ROOFLINE MODEL & HEURISTIC ENGINE v2.0 (Tuần 3 - Tuần 4)  │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Tích hợp bộ tính FLOPs tự động (fvcore) và tính Arithmetic Intensity.│
│ 2. Dựng biểu đồ tương tác Roofline Model trên Streamlit Dashboard.     │
│ 3. Hiện thực hóa bộ 19 Quy tắc Heuristic v2.0 (loại bỏ ms/Mparam).      │
│ 4. Hiện thực hóa PipelineSentinel (Preprocess → Inference → Postprocess)│
│ 5. Bổ sung module đo Output Divergence (Cosine Similarity & L1).       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 3: BENCHMARK ĐA MÔ HÌNH & BÁO CÁO KẾT QUẢ (Tuần 5 - Tuần 6)  │
├────────────────────────────────────────────────────────────────────────┤
│ • [Đồ án Kỹ thuật]: Chạy 7 kịch bản trên 5 mô hình (CPU vs T4 Colab).   │
│ • [Đồ án Kỹ thuật]: Hoàn thiện Dashboard trực quan hóa toàn bộ báo cáo.│
│ • [Mở rộng MLSys]: Thực hiện 7 bài toán Ablation Studies (AB-1 đến AB-7)│
│ • [Mở rộng MLSys]: Chạy script kiểm chuẩn đối chiếu NVIDIA Nsight CLI.  │
│ • [Mở rộng MLSys]: Hoàn thiện bản thảo bài báo khoa học chuẩn MLSys.    │
└────────────────────────────────────────────────────────────────────────┘
```

---
*Báo cáo kiểm toán được cập nhật toàn diện đồng bộ với Master Research Plan v3.5 và Technical Specification v2.5.*
