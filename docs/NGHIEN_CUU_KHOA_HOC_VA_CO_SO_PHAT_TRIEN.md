# TỔNG QUAN NGHIÊN CỨU KHOA HỌC, ĐỐI SÁNH LÝ THUYẾT & CƠ SỞ PHÁT TRIỂN HỆ THỐNG VISIONPROF (MASTER RESEARCH REPORT)

> **Tài liệu tham chiếu chuyên sâu (Comprehensive Research & Codebase Audit)**  
> **Dự án**: [VisionProf](../README.md) (Cross-Platform Deep Learning Profiling & Automated Heuristic Diagnostic Framework)  
> **Phạm vi đối chiếu**: Các tài liệu thiết kế v1.0 (đề cương nghiên cứu, đặc tả kỹ thuật, báo cáo kiểm toán; đã gỡ khỏi repo, xem lịch sử git) và **toàn bộ mã nguồn thực tế tại [`src/`](../src)**.  
> **Mục tiêu**: Làm cơ sở lý thuyết, chỗ dựa học thuật chuẩn mực (*MLSys, USENIX ATC, OSDI, EuroSys, CVPR, NeurIPS*) và kế hoạch hành động chi tiết để bảo vệ đồ án hoặc phát triển thành công bố khoa học.

---

## 📌 MỤC LỤC CHI TIẾT
1. [Phân Tích Bản Chất Đề Tài & Kiến Trúc Hệ Thống VisionProf](#1-phân-tích-bản-chất-đề-tài--kiến-trúc-hệ-thống-visionprof)
   - 1.1. Tam giác phân định: Evaluation vs. Benchmarking vs. Profiling
   - 1.2. Kiến trúc 3 mức đo lường chuẩn hóa (3-Tier Canonical Modular)
   - 1.3. Tổng hợp 6 câu hỏi nghiên cứu và phát hiện thực nghiệm cốt lõi (CH1 – CH6)
2. [Hệ Thống 12 Bài Báo Khoa Học Đỉnh Cao Làm Chỗ Dựa Lý Thuyết](#2-hệ-thống-12-bài-báo-khoa-học-đỉnh-cao-làm-chỗ-dựa-lý-thuyết)
   - 2.1. Trục 1: Tracing & Profiling Systems (Daydream, Habitat, PyTorch Profiler / Kineto)
   - 2.2. Trục 2: Roofline & Vi kiến trúc Phần cứng (Williams 2009, Hierarchical GPU Roofline)
   - 2.3. Trục 3: Nghịch lý FLOPs vs Độ trễ & Thiết kế Mạng (ShuffleNet V2, RepVGG, MobileNetV3)
   - 2.4. Trục 4: Đường ống Đầu cuối, NMS & I/O Pipeline (RT-DETR, Plumber, Video Analytics)
   - 2.5. Trục 5: Động học Huấn luyện, Activation Memory & Trễ đuôi (ActNN, Gist, vLLM)
   - 2.6. Trục 6: Tự động hóa Chẩn đoán Hiệu năng (dPRO, rl-scope)
3. [Ma Trận Đối Chiếu Chi Tiết: Codebase ĐÃ ÁP DỤNG vs. CHƯA CÓ](#3-ma-trận-đối-chiếu-chi-tiết-codebase-đã-áp-dụng-vs-chưa-có)
   - 3.1. Danh mục các thành phần đã hiện thực hóa 100% trong mã nguồn
   - 3.2. Danh mục các thành phần chưa có hoặc đã chủ động loại trừ
4. [Phân Tích 8 Rủi Ro Kỹ Thuật Vi Mô & Điểm Cần Lưu Ý Trong Hệ Thống](#4-phân-tích-8-rủi-ro-kỹ-thuật-vi-mô--điểm-cần-lưu-ý-trong-hệ-thống)
5. [Đặc Tả Chi Tiết 19 Quy Tắc Heuristic Diagnostic v2.0 Trong Codebase](#5-đặc-tả-chi-tiết-19-quy-tắc-heuristic-diagnostic-v20-trong-codebase)
6. [Lộ Trình Đề Xuất Phát Triển & Nâng Cấp Hệ Thống](#6-lộ-trình-đề-xuất-phát-triển--nâng-cấp-hệ-thống)

---

## 1. PHÂN TÍCH BẢN CHẤT ĐỀ TÀI & KIẾN TRÚC HỆ THỐNG VISIONPROF

### 1.1. Tam giác phân định: Evaluation vs. Benchmarking vs. Profiling
Một trong những sai lầm kinh điển trong cộng đồng nghiên cứu AI là đánh đồng tốc độ thực tế với số lượng tham số hoặc FLOPs. [VisionProf](../README.md) định vị rõ ranh giới khoa học qua tam giác khái niệm:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   HỆ QUY CHIẾU ĐÁNH GIÁ MÔ HÌNH AI                     │
├────────────────────────────────────────────────────────────────────────┤
│ 1. EVALUATION (Chất lượng Thuật toán ML)                               │
│    • Trọng tâm: "Mô hình dự đoán CHÍNH XÁC đến mức nào?"             │
│    • Chỉ số   : Accuracy, mAP@0.5:0.95, F1-Score, Cosine Similarity.   │
│    • Phạm vi  : Ngoài phạm vi chính, chỉ dùng kiểm soát sai lệch FP16. │
├────────────────────────────────────────────────────────────────────────┤
│ 2. BENCHMARKING (Đo chuẩn Hiệu năng Vĩ mô - Hộp đen)                   │
│    • Trọng tâm: "Mô hình chạy NHANH bao nhiêu và TỐN bao nhiêu tài     │
│                 nguyên hệ thống?"                                      │
│    • Chỉ số   : Latency P50/P90/P95/P99, FPS, Peak VRAM, Process RSS.  │
│    • Phạm vi  : Phụ trách bởi Mức Model (TV2 - src/model/).            │
├────────────────────────────────────────────────────────────────────────┤
│ 3. PROFILING (Giải phẫu Vi mô Hệ thống - Hộp trắng)                    │
│    • Trọng tâm: "TẠI SAO chậm? Chậm ở ĐÂU? Do NGUYÊN NHÂN gì và TỐI ƯU│
│                 bằng cách nào?"                                        │
│    • Chỉ số   : Layer-level latency, FLOPs, Cường độ số học (AI),     │
│                 Roofline % Peak, Bwd/Fwd Ratio, Pre/Postprocess ratio, │
│                 Heuristic Diagnostic Flags (19 quy tắc v2.0).          │
│    • Phạm vi  : Trọng tâm cốt lõi của VisionProf (TV1 & TV3).         │
└────────────────────────────────────────────────────────────────────────┘
```

---

### 1.2. Kiến trúc 3 mức đo lường chuẩn hóa (3-Tier Canonical Modular)
Hệ thống được tổ chức thành 3 phân hệ độc lập, giao tiếp thông qua hợp đồng dữ liệu ([`src/core/types.py`](../src/core/types.py)) và giao diện công khai ([`src/core/interfaces.py`](../src/core/interfaces.py)):

```
+-------------------------------------------------------------------------------+
|                       VISIONPROF PROFILING FRAMEWORK                          |
+-------------------------------------------------------------------------------+
|  🟦 MỨC MODEL (TV2 - src/model/, src/zoo/)                                    |
|  - Model Zoo (8 Adapters: mobilenet_v3, resnet50, vit_b_16, yolov8n,          |
|               distilbert, bert_base, mlp, xgboost)                            |
|  - ModelBenchmark: Đo sạch không hook, thu thập đủ 12 chỉ số thống kê         |
|  - ResourceSampler: Luồng Daemon 100ms (CPU%, RAM RSS, GPU Util%, VRAM, Power)|
|  - Kịch bản thực nghiệm TN3 (Đối đầu), TN4 (Batch size), TN5 (FP16 Cores)     |
+-------------------------------------------------------------------------------+
         |                                                       |
         v                                                       v
+-----------------------------------+       +-----------------------------------+
|  🟧 MỨC LAYER (TV1 - src/layer/)  |       |  🟩 MỨC GIAI ĐOẠN (TV3 - src/phase|
|  - Zero-Config Leaf Module Hook   |       |  - PhaseTimer (Context Manager)   |
|  - Deferred CUDA Events (1x sync) |       |  - DataLoaderSentinel (Bỏ batch 0)|
|  - Queue Prefill (_sleep)         |       |  - PhaseProfilerPlugin (TN6, TN7) |
|  - FLOPs (FlopCounterMode)        |       |  - 19 Heuristic Rules v2.0        |
|  - % Peak Roofline Model          |       |  - ABComparisonEngine (Mann-Wh.)  |
|  - Calibrate Hook Overhead        |       |  - DiagnosticEngine (Engine lõi)  |
+-----------------------------------+       +-----------------------------------+
         \                                                       /
          \                                                     /
           v                                                   v
+-------------------------------------------------------------------------------+
|  🖥️ STREAMLIT DASHBOARD (src/dashboard/ - 8 Tabs Tương Tác Chuyên Biệt)        |
|  - Tab 1: Overview 8 Models       | Tab 5: TN6 Training Phases Breakdown      |
|  - Tab 2: CPU vs T4 GPU Speedup   | Tab 6: TN7 Real-time Pipeline Breakdown   |
|  - Tab 3: Leaf Layer Breakdown    | Tab 7: Automated Diagnostic Engine (19)   |
|  - Tab 4: Roofline Model (% Peak) | Tab 8: Statistical A/B Testing Engine     |
+-------------------------------------------------------------------------------+
```

---

### 1.3. Tổng hợp 6 câu hỏi nghiên cứu và phát hiện thực nghiệm cốt lõi (CH1 – CH6)

| STT | Thành viên | Câu hỏi nghiên cứu (CH) | Thực nghiệm | Phát hiện khoa học thực chứng |
| :---: | :---: | :--- | :---: | :--- |
| **1** | **TV1** | **CH1**: Thời gian tốn nhiều nhất ở loại layer nào? Sự dịch chuyển điểm nghẽn giữa CPU và GPU ra sao? | **TN1** | Trên CPU, Conv chiếm 73–85% (CNN), Linear + Attention chiếm 95–96% (Transformer). Lên GPU, Transformer tăng tốc nhiều nhất (~$9\times$) và tỷ trọng gần như không đổi; CNN tăng tốc ít hơn ($3-6\times$) và layer nhỏ chiếm tỷ trọng lớn hơn, rõ nhất ở MobileNetV3 (Conv 73% $\rightarrow$ 47%, BatchNorm 14% $\rightarrow$ 25%). |
| **2** | **TV1** | **CH2**: Công cụ đo có chính xác không và làm chậm mô hình bao nhiêu %? | **TN2** | Sai lệch tổng so với `torch.profiler` $< 1\%$ trên CPU, $< 8\%$ trên GPU T4. Overhead CPU 5–11%; GPU ở batch 1 là 109–166%, nên mức Layer chỉ dùng để phân rã thời gian, không dùng để đo tổng thời gian (việc đó thuộc mức Model). |
| **3** | **TV2** | **CH3**: Mô hình ít tham số có chắc chạy nhanh hơn? Thứ hạng có bị đảo khi đổi phần cứng? | **TN3** | Đã đo thật trên cả CPU Mac và Colab T4. Ở batch 1 trên T4, ResNet-50 (8.2 GFLOPs) và MobileNetV3 (0.43 GFLOPs, ít hơn 19 lần) cùng mất ~6.0ms (nghẽn kernel launch latency). MLP trên GPU chậm hơn CPU (0.28ms vs 0.04ms). MobileNetV3 trễ 176.3ms trên CPU ARM Mac nhưng ~12ms trên x86. |
| **4** | **TV2** | **CH4**: Tăng Batch size và dùng FP16 Tensor Cores thì tốc độ tăng bao nhiêu? | **TN4, TN5** | Đã đo đủ 36 cấu hình batch/seq trên T4 và 7 mô hình FP16. Batch 1 $\rightarrow$ 32 tăng throughput 1.3–8.8× (MobileNetV3 tăng 8.8× từ 170 $\rightarrow$ 1,499 FPS; ViT bão hòa sớm). FP16 giảm VRAM 38–49%, nhưng chỉ tăng tốc Transformer (ViT tăng 2.81×); CNN ở batch 1 bị chậm đi do chi phối bởi kernel launch. |
| **5** | **TV3** | **CH5**: Khi Training, Backward tốn gấp mấy lần Forward? VRAM bùng nổ bao nhiêu? | **TN6** | Tỷ số thời gian $Backward / Forward$ thực nghiệm đạt **$1.39\times - 2.21\times$** (khớp hoàn hảo với lý thuyết vi tích phân $2.0\times$). Đỉnh VRAM khi training **tăng $2.7\times - 3.3\times$** so với suy luận do bắt buộc phải lưu trữ toàn bộ Intermediate Activations. |
| **6** | **TV3** | **CH6**: Trong ứng dụng thật, chi phí ngoài mô hình (Pre/Post/NMS) chiếm bao nhiêu %? | **TN7** | Với `YOLOv8n` trên GPU T4, tiền xử lý và NMS chiếm tới **$41.0\%$** tổng chu kỳ frame ($> 30\%$), chứng minh thực nghiệm cho **Định luật Amdahl**: Tối ưu mạng nơ-ron đơn lẻ sẽ nhanh chóng chạm ngưỡng bão hòa nếu bỏ quên pipeline dữ liệu. |

---

## 2. HỆ THỐNG 12 BÀI BÁO KHOA HỌC ĐỈNH CAO LÀM CHỖ DỰA LÝ THUYẾT

Để biến VisionProf từ đồ án đại học thành một công trình học thuật chuẩn mực, nhóm nghiên cứu đã tổng hợp và ánh xạ 12 công trình kinh điển từ các hội nghị hàng đầu:

```
BẢN ĐỒ VĂN HIẾN NGHIÊN CỨU LIÊN QUAN
│
├── [Trục 1] Deep Learning Profiling & Tracing Systems
│   ├── [1] Daydream (USENIX ATC '20) — Dự đoán hiệu quả tối ưu từ DFG
│   ├── [2] Habitat (USENIX ATC '21) — Dự đoán runtime xuyên phần cứng GPU
│   └── [3] PyTorch Profiler & Libkineto / HTA (Meta AI) — Chuẩn công nghiệp về tracing
│
├── [Trục 2] Roofline & Vi kiến trúc Phần cứng
│   ├── [4] Original Roofline Model (CACM '09) — Mô hình biên trên tính toán & bộ nhớ
│   └── [5] Hierarchical Roofline for GPUs (PMBS/LBNL) — Phân cấp đa tầng bộ nhớ L1/L2/HBM
│
├── [Trục 3] Nghịch lý FLOPs vs Độ trễ & Thiết kế Mạng Hướng Phần cứng
│   ├── [6] ShuffleNet V2 (ECCV '18) — 4 nguyên tắc vàng: FLOPs ≠ Latency
│   └── [7] RepVGG (CVPR '21) — Tái cấu trúc tham số (Structural Re-parameterization)
│
├── [Trục 4] Đường ống Đầu cuối, NMS & I/O Pipeline
│   ├── [8] RT-DETR (CVPR '24) — Phân tích điểm nghẽn NMS của YOLO
│   └── [9] Plumber (MLSys '22) — Tự động chẩn đoán và gỡ nghẽn Data Pipeline
│
├── [Trục 5] Động học Huấn luyện, Activation Memory & Trễ Đuôi
│   ├── [10] ActNN (ICML '21) — Nén Activation Memory khi training (2-bit)
│   ├── [11] Gist (ISCA '18) — Tối ưu hóa bộ nhớ đệm Forward-Backward
│   └── [12] vLLM / PagedAttention (SOSP '23) — Quản lý phân mảnh bộ nhớ động
│
└── [Trục 6] Tự động Hóa Chẩn đoán Hiệu năng & Quy tắc Chuyên gia
    ├── dPRO (MLSys '22) — Chẩn đoán phân tán bằng đồ thị luồng dữ liệu
    └── rl-scope (MLSys '21) — Profiling đa tầng xuyên suốt (Cross-Stack Profiling)
```

### 2.1. Trục 1: Hệ thống Đo lường & Tracing Học sâu (Profiling & Tracing Systems)

#### 1. Daydream: Accurately Estimating the Efficacy of Optimizations for DNN Training
- **Tác giả**: Hongyu Zhu, Amar Phanishayee, Gennady Pekhimenko.
- **Hội nghị**: **USENIX ATC 2020** (Annual Technical Conference).
- **Vấn đề giải quyết**: Trong tối ưu hóa ML, việc lập trình thử nghiệm một ý tưởng (ví dụ: gộp kernel, đổi precision, re-compute) rất tốn công sức. Daydream đặt câu hỏi: *"Làm sao dự đoán chính xác tốc độ cải thiện trước khi viết code?"*.
- **Cơ chế kỹ thuật**: Thu thập trace vi mô qua NVIDIA CUPTI, dựng đồ thị phụ thuộc dữ liệu (Dependency Graph) ánh xạ ngược về layer PyTorch, sau đó mô phỏng đường găng (Critical Path) để tính thời gian kỳ vọng.
- **Ánh xạ vào VisionProf**: Khẳng định sự cần thiết của việc trừu tượng hóa các CUDA kernels rời rạc về khái niệm **Layer-level** trong PyTorch mà VisionProf đang làm.

#### 2. Habitat: A Runtime-Based Computational Performance Predictor for Deep Neural Network Training
- **Tác giả**: Geoffrey X. Yu, Yubo Gao, Pavel Golikov, Gennady Pekhimenko.
- **Hội nghị**: **USENIX ATC 2021**.
- **Vấn đề giải quyết**: Chi phí thuê nhiều loại GPU đám mây để benchmark rất đắt đỏ. Habitat giải quyết bài toán: Chạy mô hình trên một GPU phổ thông giá rẻ (như T4 hoặc GTX 1080), từ đó dự đoán thời gian chạy trên các GPU cao cấp (V100, A100).
- **Cơ chế kỹ thuật**: Kết hợp *Wave Scaling* (mô hình giải tích tính toán sóng warp của GPU) và *MLP scaling* (học máy dự đoán thời gian kernel).
- **Ánh xạ vào VisionProf**: Là cơ sở khoa học để mở rộng cặp phần cứng chuẩn (`colab_cpu` $\leftrightarrow$ `colab_t4`) của VisionProf sang việc suy diễn hiệu năng cho các phần cứng khác mà không cần thiết bị thật.

#### 3. PyTorch Profiler & Libkineto / Holistic Trace Analysis (HTA)
- **Tác giả / Tổ chức**: Meta AI & PyTorch Engineering Team.
- **Hội nghị / Nền tảng**: Open Source Infrastructure / MLSys Community.
- **Đóng góp**: Libkineto là chuẩn công nghiệp nhúng trong PyTorch để bắt timeline trace đa thiết bị (CPU thread, CUDA streams). HTA cung cấp các giải thuật phân tích trace sau khi chạy.
- **Ánh xạ vào VisionProf**: VisionProf dùng `torch.profiler` làm **Ground Truth** trong thực nghiệm TN2 ([`experiments/tn2_validate_profiler.py`](../experiments/tn2_validate_profiler.py)). VisionProf khắc phục nhược điểm của Kineto đối với kỹ sư ứng dụng: Kineto xuất file trace JSON khổng lồ khó đọc, trong khi VisionProf xuất bảng phân rã layer trực quan kèm luật chẩn đoán tức thì.

---

### 2.2. Trục 2: Mô hình Roofline & Phân tích Vi kiến trúc Phần cứng

#### 4. Roofline: An Insightful Visual Performance Model for Multicore Architectures
- **Tác giả**: Samuel Williams, Andrew Waterman, David Patterson.
- **Tạp chí**: **Communications of the ACM (CACM), 2009**.
- **Đóng góp**: Đặt nền móng cho mô hình Roofline kinh điển:
  $$P_{\text{attainable}} = \min\left(P_{\text{peak}},\; B_{\text{peak}} \times AI\right)$$
  với $AI = \frac{\text{FLOPs}}{\text{Bytes}}$ (Arithmetic Intensity). Điểm gãy $AI^* = \frac{P_{\text{peak}}}{B_{\text{peak}}}$ phân định hai miền giới hạn: **Memory-Bandwidth Bound** và **Compute Bound**.
- **Ánh xạ vào VisionProf**: Là nền tảng toán học cho toàn bộ Tab 4 ([`src/dashboard/tabs/tab_roofline.py`](../src/dashboard/tabs/tab_roofline.py)) và module tính toán hiệu suất phần cứng ([`src/layer/roofline.py`](../src/layer/roofline.py)).

#### 5. An Instruction / Hierarchical Roofline Model for GPUs
- **Tác giả**: Nan Ding, Samuel Williams.
- **Hội nghị / Báo cáo**: **IEEE PMBS Workshop / Lawrence Berkeley National Laboratory (LBNL)**.
- **Đóng góp**: Mở rộng mô hình Roofline cho GPU hiện đại: Không chỉ có trần DRAM, mà GPU gồm nhiều cấp bộ nhớ: Registers $\rightarrow$ Shared Memory / L1 Cache $\rightarrow$ L2 Cache $\rightarrow$ HBM/DRAM.
- **Ánh xạ vào VisionProf**: Cung cấp cơ sở lý thuyết giải thích tại sao các layer nhỏ (ReLU, BatchNorm) dù có $AI$ thấp nhưng không đạt được trần băng thông DRAM lý thuyết (do bị nghẽn ở latency và cache hierarchy).

---

### 2.3. Trục 3: Nghịch lý FLOPs vs Độ trễ & Thiết kế Mạng Hướng Phần cứng

#### 6. ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
- **Tác giả**: Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun.
- **Hội nghị**: **ECCV 2018** (European Conference on Computer Vision).
- **Đóng góp**: Tuyên ngôn mang tính cách mạng: **"FLOPs là thước đo gián tiếp, Tốc độ thực tế mới là chân lý"**. Nhóm tác giả đưa ra 4 nguyên tắc vàng:
  - *G1: Độ rộng kênh bằng nhau ($c_1 = c_2$) tối thiểu hóa chi phí truy cập bộ nhớ (MAC).*
  - *G2: Tích chập nhóm (Group Convolution) quá mức làm tăng MAC.*
  - *G3: Phân mảnh mạng (Network Fragmentation) làm giảm mức độ song song phần cứng.*
  - *G4: Các phép toán phần tử (Element-wise ops: ReLU, Add, Bias) tốn nhiều thời gian trên GPU dù FLOPs rất nhỏ.*
- **Ánh xạ vào VisionProf**: **Giải thích trực tiếp hai phát hiện thực nghiệm lớn nhất của nhóm**:
  1. Giải thích tại sao `YOLOv11n` chậm hơn `YOLOv8n` trên CPU do vi phạm G3 (bị phân mảnh thành 177 layers gây sequential CPU dispatch overhead).
  2. Giải thích tại sao trên GPU T4, các layer Norm & Activation chiếm tới 25% thời gian do vi phạm G4.

#### 7. RepVGG: Making VGG-style ConvNets Great Again
- **Tác giả**: Xiaohan Ding, Xiangyu Zhang, Ningning Ma, Jungong Han, Guiguang Ding, Jian Sun.
- **Hội nghị**: **CVPR 2021** (IEEE Conference on Computer Vision and Pattern Recognition).
- **Đóng góp**: Đưa ra kỹ thuật *Structural Re-parameterization*: Khi train dùng cấu trúc đa nhánh (multi-branch), khi inference gộp toán học (fuse) toàn bộ Conv 3x3, Conv 1x1 và BatchNorm về một chuỗi Conv 3x3 thuần túy (plain). Plain network đạt tốc độ vượt trội trên GPU nhờ tối ưu hóa băng thông bộ nhớ và triệt tiêu launch latency.
- **Ánh xạ vào VisionProf**: Cung cấp giải pháp định hướng để bộ chẩn đoán VisionProf đề xuất người dùng tái cấu trúc khi phát hiện mô hình bị nghẽn do phân mảnh (luật `RD-02`, `RD-04`).

---

### 2.4. Trục 4: Đường ống Đầu cuối, NMS & I/O Pipeline

#### 8. DETRs Beat YOLOs on Real-time Object Detection (RT-DETR)
- **Tác giả**: Yian Zhao, Wenyu Lv, Shangliang Xu, et al. (Baidu Inc.).
- **Xuất bản**: **CVPR 2024** (bản arXiv 2023).
- **Đóng góp**: Bóc tách chi tiết bản chất **NMS Bottleneck** trong các mạng phát hiện vật thể dạng YOLO: NMS là thuật toán tuần tự trên CPU, thời gian chạy biến thiên theo số lượng vật thể trong frame ảnh, gây ra độ trễ không ổn định (latency jitter) và phá vỡ tính thời gian thực.
- **Ánh xạ vào VisionProf**: Cơ sở lý thuyết cho **CH6 & TN7** ([`scripts/run_tn7_pipeline.py`](../scripts/run_tn7_pipeline.py)), chứng minh thực nghiệm rằng trên GPU T4, khâu tiền xử lý ảnh và NMS chiếm tới **$41.0\%$** chu kỳ frame của YOLOv8n.

#### 9. Plumber: Diagnosing and Removing Performance Bottlenecks in Machine Learning Data Pipelines
- **Tác giả**: Michael Kuchnik, Ana Klimovic, Jiri Simsa, Virginia Smith, George Amvrosiadis.
- **Hội nghị**: **MLSys 2022** (Conference on Machine Learning and Systems).
- **Đóng góp**: Khảo sát hơn 2 triệu job huấn luyện tại Google và chứng minh phần lớn GPU bị đói tính toán (GPU Starvation) là do pipeline nạp dữ liệu trên CPU không theo kịp tốc độ của GPU. Plumber tự động phát hiện nút thắt cổ chai và tự động cấu hình parallelism, prefetching và caching.
- **Ánh xạ vào VisionProf**: Cơ sở lý thuyết cho công cụ [`DataLoaderSentinel`](../src/phase/dataloader.py) và các quy tắc chẩn đoán `RC-01`, `RC-03` trong việc phát hiện Compute Starvation.

---

### 2.5. Trục 5: Động học Huấn luyện, Activation Memory & Trễ Đuôi

#### 10. ActNN: Reducing Training Memory Footprint via 2-Bit Activation Compressed Training
- **Tác giả**: Jianfei Chen, Lianmin Zheng, Zhewei Yao, Ion Stoica, Joseph E. Gonzalez, Michael W. Mahoney.
- **Hội nghị**: **ICML 2021** (PMLR 139, pp. 1803–1813).
- **Đóng góp**: Chứng minh bộ nhớ VRAM trong quá trình huấn luyện chủ yếu bị chiếm dụng bởi **Intermediate Activations** (được lưu lại từ Forward pass để dùng cho Backward pass), chứ không phải trọng số mô hình. ActNN lượng tử hoá ngẫu nhiên activation xuống trung bình 2-bit, giảm bộ nhớ activation 12× và cho phép batch lớn hơn 6.6–14×.
- **Ánh xạ vào VisionProf**: Giải thích trực tiếp kết quả thực nghiệm **TN6 (CH5)** của VisionProf: Đỉnh VRAM khi training tăng gấp **$2.7\times - 3.3\times$** so với khi suy luận.

#### 11. Gist: Efficient Data Encoding for Deep Neural Network Training
- **Tác giả**: Animesh Jain, Amar Phanishayee, Jason Mars, Lingjia Tang, Gennady Pekhimenko.
- **Hội nghị**: **ISCA 2018** (International Symposium on Computer Architecture).
- **Đóng góp**: Chỉ ra feature map lưu lại từ forward để dùng cho backward chiếm phần lớn bộ nhớ khi training, và mã hoá (encode) chúng gọn hơn trong khoảng thời gian chờ backward để giảm bộ nhớ.
- **Ánh xạ vào VisionProf**: Giải thích vì sao bộ nhớ training trong TN6 (CH5) lớn hơn suy luận $2.7\times - 3.3\times$: phải giữ activation cho backward. (Tỷ số $Backward / Forward \approx 2\times$ đo bằng [`PhaseTimer`](../src/phase/timer.py) là xấp xỉ quen thuộc về số phép tính, không phải kết luận của Gist.)

#### 12. vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention
- **Tác giả**: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica.
- **Hội nghị**: **SOSP 2023** (ACM Symposium on Operating Systems Principles).
- **Đóng góp**: Chỉ ra rằng sự phân mảnh bộ nhớ động (Memory Fragmentation) làm lãng phí 60–80% VRAM của GPU. Cơ chế PagedAttention lấy cảm hứng từ phân trang bộ nhớ ảo của Hệ điều hành để triệt tiêu phân mảnh.
- **Ánh xạ vào VisionProf**: Là cơ sở lý thuyết cho luật chẩn đoán phân mảnh bộ nhớ `RA-02` ([`src/analyzer/rules/memory.py`](../src/analyzer/rules/memory.py)) đánh giá tỷ số giữa bộ nhớ dành riêng (Reserved) và bộ nhớ thực cấp phát (Allocated).

---

## 3. MA TRẬN ĐỐI CHIẾU CHI TIẾT: CODEBASE ĐÃ ÁP DỤNG VS. CHƯA CÓ

Bảng đối chiếu minh bạch giữa mã nguồn thực tế triển khai với các giả thuyết và tài liệu thiết kế:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    MA TRẬN HIỆN TRẠNG TRIỂN KHAI CODEBASE                   │
├───────────────────────────────┬──────────────────────────────────────────────┤
│ PHÂN HỆ KỸ THUẬT              │ TRẠNG THÁI TRIỂN KHAI THỰC TẾ TRONG SRC/     │
├───────────────────────────────┼──────────────────────────────────────────────┤
│ 1. Deferred CUDA Events       │ ✅ ĐÃ ÁP DỤNG HOÀN TOÀN (src/layer/events.py) │
│ 2. Queue Prefill (_sleep)     │ ✅ ĐÃ ÁP DỤNG HOÀN TOÀN (src/layer/events.py) │
│ 3. LIFO Event Stack           │ ✅ ĐÃ ÁP DỤNG HOÀN TOÀN (src/layer/events.py) │
│ 4. FLOPs & Roofline Model     │ ✅ ĐÃ ÁP DỤNG HOÀN TOÀN (src/layer/roofline) │
│ 5. Model Zoo 8 Adapters       │ ✅ ĐÃ ÁP DỤNG HOÀN TOÀN (src/zoo/)           │
│ 6. 12 Metrics & Peak VRAM     │ ✅ ĐÃ ÁP DỤNG HOÀN TOÀN (src/model/benchmark)│
│ 7. Daemon ResourceSampler     │ ✅ ĐÃ ÁP DỤNG HOÀN TOÀN (src/model/resources)│
│ 8. PhaseTimer & Sentinel      │ ✅ ĐÃ ÁP DỤNG HOÀN TOÀN (src/phase/)         │
│ 9. 19 Heuristics Rules v2.0   │ ✅ ĐÃ ÁP DỤNG HOÀN TOÀN (src/analyzer/rules/) │
│ 10. A/B Engine (Mann-Whitney) │ ✅ ĐÃ ÁP DỤNG HOÀN TOÀN (src/analyzer/ab_...)│
│ 11. Streamlit Dashboard 8 Tabs│ ✅ ĐÃ ÁP DỤNG HOÀN TOÀN (src/dashboard/)     │
│ 12. SQLite DDL & Parquet      │ ❌ CHƯA CÓ (Dùng Flat CSV/JSON per result)  │
│ 13. NVIDIA Nsight CLI Harness │ ❌ CHƯA CÓ (Dùng torch.profiler thay thế)   │
│ 14. Ablation Scripts AB-1..7  │ ❌ CHƯA CÓ (Chưa có runner tự động hóa riêng)│
│ 15. Compiled Runtimes         │ ❌ NGOÀI PHẠM VI (Chỉ chạy PyTorch Eager)   │
│ 16. Prescriptive Code Rewrite │ ❌ NGOÀI PHẠM VI (Chỉ dừng ở Text Warning)  │
└───────────────────────────────┴──────────────────────────────────────────────┘
```

### 3.1. Danh mục các thành phần ĐÃ ÁP DỤNG HOÀN TOÀN trong mã nguồn

1. **Cơ chế đo GPU bất đồng bộ chuẩn xác ([`src/layer/events.py`](../src/layer/events.py))**:
   - `DeferredCudaDispatcher`: Thu thập cặp `torch.cuda.Event` trên CUDA stream của từng leaf module mà không gọi `synchronize()` trong hook. Chỉ đồng bộ 1 lần duy nhất ở `end_iteration()`.
   - **Queue Prefill**: Dùng `torch.cuda._sleep()` làm GPU bận rộn đủ lâu để CPU kịp enqueue toàn bộ các kernel của forward pass, giúp GPU chạy back-to-back, triệt tiêu thời gian GPU rảnh chờ CPU.
   - **LIFO Stack Event**: Sử dụng ngăn xếp `_stacks[layer_name]` để quản lý hoàn hảo các module được gọi lặp lại (re-entrant modules như activation `SiLU` trong YOLOv8).

2. **Mô hình Roofline & Phân tích FLOPs giải tích ([`src/layer/roofline.py`](../src/layer/roofline.py) & [`src/layer/flops.py`](../src/layer/flops.py))**:
   - Tự động tính FLOPs qua `torch.utils.flop_counter.FlopCounterMode` (fallback công thức giải tích cho các layer không đếm được, ví dụ BatchNorm, activation).
   - Tính toán lưu lượng bộ nhớ tổng hợp ($bytes = input + weights + output$) và $AI = FLOPs / bytes$.
   - Tính toán hiệu suất phần cứng `% Peak GFLOPs` và phân loại nhãn `bound` ("compute" vs "memory").
   - Cung cấp bảng thông số phần cứng chuẩn `HardwareSpec` cho Tesla T4 ([`src/layer/hw_specs.py`](../src/layer/hw_specs.py)).

3. **Harness đo lường Model Zoo & 12 chỉ số khoa học ([`src/model/benchmark.py`](../src/model/benchmark.py) & [`src/zoo/`](../src/zoo))**:
   - Cung cấp đủ 8 Model Adapters: CV (`mobilenet_v3_large`, `resnet50`, `vit_b_16`, `yolov8n`), NLP (`distilbert`, `bert_base`), Tabular (`mlp`, `xgboost`).
   - Loại bỏ bẫy đồng bộ ngầm của BERT (không truyền `attention_mask` khi input không có padding).
   - Đo đạc đủ 12 chỉ số: `mean_ms`, `std_ms`, `p50_ms`, `p90_ms`, `p95_ms`, `p99_ms`, `tail_ratio`, `throughput_fps`, `params_m`, `peak_vram_mb`, `rss_mb`, `cold_start_ms`.
   - Bắt đỉnh VRAM chính xác 100% qua `torch.cuda.max_memory_allocated()`.
   - Runner 2 pha ([`scripts/run_experiment.py`](../scripts/run_experiment.py)): Pha 1 đo latency sạch không hook $\rightarrow$ Pha 2 mới gắn hook đo phân rã layer.

4. **Giám sát tài nguyên không xâm lấn ([`src/model/resources.py`](../src/model/resources.py))**:
   - `ResourceSampler`: Chạy trên luồng Daemon phụ độc lập với chu kỳ lấy mẫu 100ms, thu thập CPU%, RAM RSS, GPU Util%, VRAM, Power (Watts), Nhiệt độ (°C).
   - Tuyệt đối không gọi NVML trong hook của layer (chống gây trễ bubble).

5. **Bộ đo Giai đoạn & Chống nhiễu I/O ([`src/phase/timer.py`](../src/phase/timer.py) & [`src/phase/dataloader.py`](../src/phase/dataloader.py))**:
   - `PhaseTimer`: Context manager bọc các pha `forward`, `backward`, `optimizer`, `preprocess`, `postprocess`, có đồng bộ CUDA đúng quy chuẩn ở biên mỗi pha.
   - `DataLoaderSentinel`: Bọc DataLoader, tự động loại trừ cold-start của batch 0, đo thời gian fetch dữ liệu và phát hiện nghẽn I/O.

6. **Động cơ Chẩn đoán 19 Luật v2.0 & A/B Testing ([`src/analyzer/`](../src/analyzer))**:
   - Đã hiện thực hóa đủ 19 luật v2.0 trong thư mục `src/analyzer/rules/`, sửa triệt để 4 lỗi logic:
     - `RB-04` & `RD-05`: Bỏ hoàn toàn $\text{ms/Mparam}$, chuyển sang so sánh với Roofline và $\text{ms/GFLOP}$.
     - `RD-02`: Đổi từ "Dead layer" thành "Sub-microsecond layer" ($< 1\mu s$).
     - `RC-03`: Sửa định nghĩa Compute Starvation (GPU đói việc do nạp dữ liệu chậm).
     - `RA-04`: Tách biệt Process RSS và Activation Memory trên CPU.
   - `ABComparisonEngine`: Kiểm định phi tham số Mann-Whitney U và Cliff's Delta, tính Speedup dựa trên tổng thời gian vòng lặp (toàn mô hình).

---

### 3.2. Những thành phần CHƯA CÓ hoặc NGOÀI PHẠM VI (Gaps & Exclusions)

1. **Lưu trữ SQLite DDL quan hệ & Nén cột Parquet**:
   - *Hiện trạng đề cương*: Đề xuất lược đồ 7 bảng trong SQLite (`visionprof.db`) và Parquet exporter.
   - *Thực tế codebase*: Dùng cấu trúc thư mục phẳng chuẩn hóa **Canonical Result Structure** (`results/{device}/{model}/{exp_id}/{config}/`).
   - *Lý do*: Quyết định có chủ đích (PROJECT_PLAN Mục 6) để hỗ trợ 3 thành viên làm việc độc lập mà không bị lock file database trên Google Drive/Colab.

2. **Kịch bản kiểm chuẩn tự động qua NVIDIA Nsight Systems CLI (`nsys profile`)**:
   - *Hiện trạng đề cương*: Đề xuất kịch bản CLI tự động đối chiếu NVTX marker với trace SQLite từ Nsight.
   - *Thực tế codebase*: Chưa có script gọi CLI của Nsight. Thay vào đó, nhóm kiểm chuẩn đối chiếu Ground Truth trực tiếp bằng thư viện `torch.profiler` trong [`experiments/tn2_validate_profiler.py`](../experiments/tn2_validate_profiler.py).

3. **Bộ kịch bản tự động cho 7 bài toán Ablation Studies (AB-1 đến AB-7)**:
   - *Hiện trạng đề cương*: Đề xuất 7 bài toán bóc tách thành phần chạy tự động qua CLI.
   - *Thực tế codebase*: Chưa có runner riêng. Các khảo sát (như warm-up saturation, overhead breakdown) được thực hiện qua các tham số tùy biến trong kịch bản TN1 $\rightarrow$ TN7.

4. **Hỗ trợ các Runtime biên dịch (Compiled Runtimes & TensorRT)**:
   - *Thực tế codebase*: Tối ưu hóa 100% cho **PyTorch Eager Mode**. Chưa hỗ trợ profiling nội bộ cho `torch.compile` (Inductor Triton kernels), NVIDIA TensorRT, hay ONNX Runtime cho các mô hình Deep Learning (chỉ hỗ trợ hộp đen cho XGBoost).

5. **Khả năng tự động viết lại code (Prescriptive Code Rewriter / Auto-Tuning)**:
   - *Thực tế codebase*: `DiagnosticEngine` dừng lại ở việc đưa ra cảnh báo (`PASS`, `WARN`, `FAIL`, `SKIPPED`) và lời khuyên kỹ thuật dạng văn bản (Recommendation string). Chưa có module tự động biến đổi đồ thị hoặc sinh mã tối ưu hóa như Plumber hay Daydream.

---

## 4. PHÂN TÍCH 8 RỦI RO KỸ THUẬT VI MÔ & ĐIỂM CẦN LƯU Ý TRONG HỆ THỐNG

Khi làm việc với các hệ thống đo lường hiệu năng học sâu, có **8 cạm bẫy kỹ thuật vi mô** mà người phát triển cần nắm vững để tránh sai lệch kết quả:

### R1. Ngưỡng sàn đo lường của CUDA Events (CUDA Event Timing Floor)
- **Vấn đề**: Mỗi cặp `cudaEventRecord` và `cudaEventElapsedTime` có một độ phân giải vật lý tối thiểu. TN2 đo được ngưỡng sàn này khoảng $6\,\mu s$ trên Colab T4 (Linux) bằng `estimate_timer_bias(tiny_kernel=True)` (`timer_floor_ms` trong `results/analysis/TN2/summary_colab_t4.csv`). Trên Windows với driver WDDM, cơ chế gom lệnh của hệ điều hành đẩy ngưỡng này lên khoảng $20\,\mu s$.
- **Hệ quả**: Đối với các layer cực nhỏ ở batch size 1 (như `nn.ReLU`, `nn.Identity`), thời gian đo được chủ yếu là chi phí của chính event, dẫn đến việc tổng thời gian các layer bị đội lên (overestimation).
- **Cách VisionProf xử lý**: Bổ sung hàm `estimate_timer_bias(tiny_kernel=True)` trong [`src/layer/overhead.py`](../src/layer/overhead.py) và đưa ra cột `layers_below_3x_floor_pct` trong báo cáo TN2 để cảnh báo người dùng.

### R2. "Thời gian không thuộc layer nào" (Unattributed Time)
- **Vấn đề**: PyTorch forward hooks chỉ can thiệp vào các lớp kế thừa `nn.Module`. Các phép toán gọi trực tiếp dưới dạng hàm (functional ops) như `x + shortcut` (residual add trong ResNet), `torch.cat`, hay phép nhân ma trận `Q @ K.T` trong Multi-Head Attention không có module đại diện.
- **Hệ quả**: Tổng thời gian các leaf layers luôn nhỏ hơn thời gian chạy toàn bộ forward pass sạch.
- **Cách VisionProf xử lý**: Định nghĩa tường minh tỷ lệ $Unattributed = \frac{T_{\text{clean}} - \sum T_{\text{layers}}}{T_{\text{clean}}}$ và đưa vào báo cáo kiểm chuẩn TN2; đồng thời hỗ trợ `atomic_types = (nn.MultiheadAttention,)` để bọc trọn gói khối Attention.

### R3. Nhập nhằng bộ nhớ: Process RSS vs. Activation Memory trên CPU
- **Vấn đề**: Trên CPU, không có bộ cấp phát CUDA Allocator độc lập. Bộ nhớ RAM tiến trình (`psutil.Process().memory_info().rss`) chịu sự chi phối của cơ chế cấp phát trang lười (Lazy OS Paging) và Garbage Collector của Python, không phản ánh đúng bộ nhớ tensor của từng layer.
- **Cách VisionProf xử lý**: Tách biệt hoàn toàn: Dùng Process RSS chỉ cho việc phát hiện rò rỉ bộ nhớ dài hạn (`RA-04`), còn bộ nhớ kích hoạt của từng layer được tính toán giải tích trực tiếp từ kích thước và kiểu dữ liệu của tensor output ([`src/layer/memory.py`](../src/layer/memory.py)).

### R4. Module được gọi lặp lại trong một pass (Re-entrant Modules)
- **Vấn đề**: Nhiều kiến trúc hiện đại dùng chung một thể hiện module cho nhiều vị trí (ví dụ: YOLOv8n dùng chung một instance `nn.SiLU()` cho hàng chục khối C2f). Nếu lưu sự kiện bắt đầu bằng một biến đơn lẻ, lần gọi sau sẽ ghi đè lên lần gọi trước.
- **Cách VisionProf xử lý**: Quản lý sự kiện bằng cấu trúc ngăn xếp **LIFO Stack** (`_stacks[layer_name] = []`) trong [`src/layer/events.py`](../src/layer/events.py). Hook pre-forward đẩy event vào stack; hook post-forward rút event đỉnh ra và ghi nhận thời gian.

### R5. Trôi nhiệt (Thermal Throttling) & Xung nhịp GPU
- **Vấn đề**: Khi chạy benchmark liên tục trên GPU Colab T4 hoặc laptop, nhiệt độ chip tăng cao dẫn đến việc phần cứng tự động hạ xung nhịp (Thermal Throttling). Nếu chạy toàn bộ cấu hình A rồi mới chạy cấu hình B, cấu hình B sẽ bị thiệt thòi về thời gian.
- **Cách VisionProf xử lý**: Áp dụng giao thức chạy xen kẽ luân phiên (**Interleaved Protocol**: $A_1 \rightarrow B_1 \rightarrow A_2 \rightarrow B_2 \dots$) trong công cụ đo A/B ([`src/layer/overhead.py`](../src/layer/overhead.py)), kết hợp số vòng warm-up $N_{warmup} \ge 15$.

### R6. Bẫy đồng bộ ngầm CPU-GPU (Implicit Synchronization Traps)
- **Vấn đề**: Các thao tác vô tình trong Python như `tensor.item()`, `tensor.tolist()`, `print(tensor)` hay các câu lệnh điều kiện `if tensor > 0:` sẽ ép CPU phải dừng lại chờ GPU hoàn tất toàn bộ pipeline, làm sai lệch hoàn toàn thời gian đo lường bất đồng bộ.
- **Cách VisionProf xử lý**: Kiểm duyệt toàn bộ adapter trong `src/zoo/` (đặc biệt là adapter `bert_base` trong [`src/zoo/nlp.py`](../src/zoo/nlp.py), bỏ qua `attention_mask` khi input ngẫu nhiên để tránh hàm kiểm tra mask của thư viện Transformers kích hoạt đồng bộ ngầm).

### R7. Hạn chế của bộ đếm FLOPs tự động (FlopCounterMode) với Custom Ops
- **Vấn đề**: Các bộ đếm FLOPs tự động phụ thuộc vào bảng đăng ký toán tử `aten`. Khi gặp các toán tử tùy biến (Custom C++/CUDA extensions) hoặc các phép toán không đăng ký công thức FLOPs, bộ đếm sẽ bỏ sót.
- **Cách VisionProf xử lý**: Xây dựng kiến trúc 3 tầng tính FLOPs ([`src/layer/flops.py`](../src/layer/flops.py)): Tầng 1 dùng `FlopCounterMode` $\rightarrow$ Tầng 2 fallback sang bảng công thức chuẩn `_ANALYTIC_FLOPS` (LayerNorm, GELU, SiLU) $\rightarrow$ Tầng 3 dùng công thức hình học `FlopsCalculator` (Conv2d, Linear).

### R8. Quá tải hàng đợi Queue Prefill (Prefill Drain vs Overfill)
- **Vấn đề**: Kỹ thuật Queue Prefill dùng `torch.cuda._sleep()` để giữ chân GPU. Nếu thời gian sleep quá ngắn, GPU sẽ hoàn thành sleep trước khi CPU kịp enqueue xong $\rightarrow$ GPU lại bị rảnh. Nếu sleep quá dài, thời gian đo cả pass sẽ bị kéo dài không cần thiết.
- **Cách VisionProf xử lý**: Triển khai cơ chế **Adaptive Prefill**: Đo đạc thời gian enqueue của CPU ở vòng trước ($t_{\text{enqueue}}$) và tự động cập nhật thời gian sleep cho vòng sau: $t_{\text{prefill}} = \min(1000.0, \max(5.0, 1.5 \times t_{\text{enqueue}} + 2.0))$, đồng thời ghi nhận cờ `_prefill_ok` để kiểm tra độ tin cậy.

---

## 5. ĐẶC TẢ CHI TIẾT 19 QUY TẮC HEURISTIC DIAGNOSTIC V2.0 TRONG CODEBASE

Toàn bộ 19 quy tắc được chia thành 4 nhóm độc lập trong thư mục [`src/analyzer/rules/`](../src/analyzer/rules):

### Nhóm 1: Bộ Nhớ (Memory Rules - RA)
1. **RA-01 (OOM Risk)**: So sánh Peak VRAM với dung lượng bộ nhớ vật lý.
   - $\text{util} > 95\% \rightarrow \text{FAIL}$; $\text{util} > 90\% \rightarrow \text{WARN}$; còn lại $\text{PASS}$.
2. **RA-02 (Memory Fragmentation)**: Đánh giá độ phân mảnh của Caching Allocator:
   $$\text{frag\_ratio} = \frac{\text{Reserved} - \text{Allocated}}{\text{Reserved}}$$
   - $\text{frag\_ratio} > 40\% \rightarrow \text{WARN}$; còn lại $\text{PASS}$.
3. **RA-03 (Activation Hotspot)**: Phát hiện một tầng mạng đơn lẻ chiếm giữ quá nhiều bộ nhớ kích hoạt.
   - $\text{Max Layer Activation} / \text{Total Activation} > 50\% \rightarrow \text{WARN}$.
4. **RA-04 (Memory Leak)**: Phát hiện xu hướng tăng đơn điệu của Process RSS qua 10 vòng lặp.
   - $\Delta \text{RSS} > 150\,\text{MB} \rightarrow \text{FAIL}$; $\Delta \text{RSS} > 50\,\text{MB} \rightarrow \text{WARN}$.
5. **RA-05 (Host-Device Transfer)**: Chi phí sao chép dữ liệu CPU $\leftrightarrow$ GPU quá cao.
   - $T_{\text{transfer}} / T_{\text{iteration}} > 20\% \rightarrow \text{WARN}$.

### Nhóm 2: Tính Toán (Compute Rules - RB)
6. **RB-01 (GPU Utilization)**: Mức độ sử dụng nhân tính toán GPU trung bình từ luồng nền.
   - $\text{Mean GPU Util} < 50\% \rightarrow \text{WARN}$ (GPU đang bị đói việc hoặc tải quá nhẹ).
7. **RB-02 (Tail Latency Jitter)**: Đánh giá độ biến thiên độ trễ dịch vụ.
   - $\text{Tail Ratio } (P99 / P50) > 2.0 \rightarrow \text{FAIL}$; $P99 / P50 > 1.5 \rightarrow \text{WARN}$.
8. **RB-03 (Backward Bottleneck)**: Tỷ số thời gian lan truyền ngược so với lan truyền xuôi.
   - $T_{\text{backward}} / T_{\text{forward}} > 2.5\times \rightarrow \text{WARN}$ (bình thường là $\approx 2.0\times$).
9. **RB-04 (Compute Efficiency)**: Mật độ tính toán thực tế trên mỗi đơn vị FLOPs.
   - $\text{ms / GFLOP} > 2.0 \rightarrow \text{WARN}$ (Hiệu suất tính toán quá kém so với phần cứng).
10. **RB-05 (Optimizer Stall)**: Thời gian cập nhật trọng số của Optimizer quá lâu.
    - $T_{\text{opt}} / T_{\text{step}} > 30\% \rightarrow \text{WARN}$ (cần xem xét gradient clipping hoặc fused optimizer).

### Nhóm 3: Đường Ống Dữ Liệu (DataLoader Rules - RC)
11. **RC-01 (I/O Stall Ratio)**: Tỷ lệ thời gian nạp dữ liệu trong chu trình huấn luyện.
    - $T_{\text{I/O}} / T_{\text{step}} > 35\% \rightarrow \text{FAIL}$; $T_{\text{I/O}} / T_{\text{step}} > 20\% \rightarrow \text{WARN}$.
12. **RC-02 (Worker Variance)**: Độ biến thiên thời gian lấy dữ liệu giữa các batch (hệ số biến thiên $CV$).
    - $CV = \frac{\sigma}{\mu} > 0.50 \rightarrow \text{WARN}$ (mất cân bằng tải giữa các worker threads).
13. **RC-03 (GPU Starvation)**: GPU bị đói dữ liệu nghiêm trọng.
    - $T_{\text{I/O}} > T_{\text{compute}} \rightarrow \text{FAIL}$ (Thời gian chờ nạp batch lớn hơn thời gian GPU tính forward + backward).
14. **RC-04 (Batch Underutilization)**: Kích thước batch quá nhỏ gây lãng phí năng lực tính toán song song.
    - $\text{Batch} = 1$ và $\text{FPS} < 20 \rightarrow \text{WARN}$ (khuyến nghị tăng kích thước batch).

### Nhóm 4: Kiến Trúc Mạng (Architecture Rules - RD)
15. **RD-01 (Param Concentration)**: Mức độ tập trung trọng số vào một tầng mạng đơn lẻ.
    - $\text{Max Layer Params} / \text{Total Params} > 60\% \rightarrow \text{WARN}$ (mô hình mất cân đối, ví dụ tầng Linear cuối quá lớn).
16. **RD-02 (Sub-microsecond Layers)**: Tỷ lệ các tầng mạng siêu nhỏ ($< 1\,\mu s$).
    - Tỷ lệ tầng $< 0.001\,\text{ms}$ vượt quá $30\% \rightarrow \text{WARN}$ (khuyến nghị gộp tầng - layer fusion).
17. **RD-03 (Memory-Compute Disparity)**: Tỷ số giữa bộ nhớ kích hoạt sinh ra trên mỗi GFLOP tính toán.
    - $\text{Activation (MB)} / \text{GFLOPs} > 50.0 \rightarrow \text{WARN}$ (tầng mạng nghẽn băng thông bộ nhớ nghiêm trọng).
18. **RD-04 (Single Layer Bottleneck)**: Một tầng mạng đơn lẻ chiếm dụng phần lớn thời gian forward.
    - $T_{\text{layer}} / T_{\text{forward}} > 35\% \rightarrow \text{WARN}$ (điểm nghẽn chi phối kiến trúc).
19. **RD-05 (Latency-FLOPs Outliers)**: Tầng mạng tốn nhiều thời gian nhưng hầu như không thực hiện phép tính nào.
    - $T_{\text{layer}} / T_{\text{total}} > 15\%$ trong khi $\text{FLOPs} < 1\,000\,000 \rightarrow \text{WARN}$ (thường là Reshape, Transpose hoặc Non-fused Norm).

---

## 6. LỘ TRÌNH ĐỀ XUẤT PHÁT TRIỂN & NÂNG CẤP HỆ THỐNG

Dựa trên việc đối chiếu với văn hiến khoa học và rà soát các khoảng trống trong mã nguồn, đây là **4 hướng chiến lược khả thi** để tiếp tục phát triển hệ thống VisionProf:

```
LỘ TRÌNH NÂNG CẤP HỆ THỐNG VISIONPROF
│
├── [Hướng 1] Tự động hóa đề xuất mã nguồn (Prescriptive Code-Fix Suggestions)
│   ├── Tích hợp tự động gộp tầng: torch.ao.quantization.fuse_modules
│   ├── Tự động tối ưu DataLoader: Gợi ý số worker và pin_memory chuẩn
│   └── Gợi ý tăng tốc NMS: Chuyển sang torchvision.ops.nms CUDA
│
├── [Hướng 2] Mở rộng Runtime sang Trình biên dịch (ML Compilers)
│   ├── Hỗ trợ torch.compile (TorchInductor với Triton backend)
│   ├── So sánh đối đầu Eager vs. Compiled trên Roofline Model
│   └── Khảo sát mức độ triệt tiêu chi phí của các layer nhỏ (BatchNorm, ReLU)
│
├── [Hướng 3] Mở rộng Model Zoo sang Multimodal AI (VLM)
│   ├── Bổ sung adapter cho CLIP (ViT-B/32) hoặc SmolVLM
│   └── Phân tích bất đối xứng hiệu năng: Text Transformer vs. Vision Transformer
│
└── [Hướng 4] Thực hiện hóa bộ thực nghiệm Ablation tự động (AB-1 đến AB-7)
    ├── Xây dựng runner scripts/run_ablation_studies.py
    ├── Quét đường cong bão hòa warm-up N ∈ [0, 50]
    └── Đo lường định lượng chi phí overhead thành phần (Timer, Memory, Sampler)
```

### Chi tiết các bước triển khai:

1. **Hướng 1: Nâng cấp Động cơ Chẩn đoán thành "Tự Động Đề Xuất Mã Nguồn" (Prescriptive Code Suggestions)**:
   - *Hành động*: Khi luật `RD-02` (Sub-microsecond layers) được kích hoạt $\rightarrow$ Hệ thống tự động sinh đoạn code gợi ý gộp Conv-BN-ReLU bằng `torch.ao.quantization.fuse_modules`. Khi luật `RC-03` (GPU Starvation) được kích hoạt $\rightarrow$ Sinh đoạn code mẫu tối ưu: `DataLoader(..., num_workers=4, pin_memory=True, persistent_workers=True)`.

2. **Hướng 2: Khảo sát Tác Động Của Trình Biên Dịch Học Máy (`torch.compile`)**:
   - *Hành động*: Thêm cờ `--compile` vào [`scripts/run_experiment.py`](../scripts/run_experiment.py). Sử dụng Tab 8 (A/B Testing) để so sánh đối đầu giữa chế độ PyTorch Eager truyền thống và TorchInductor. Định lượng xem trình biên dịch đã giúp đẩy điểm biểu diễn của các layer Norm/Activation lên gần trần Roofline như thế nào.

3. **Hướng 3: Bổ Sung Mô Hình Đa Phương Thức (Vision-Language Models)**:
   - *Hành động*: Tạo adapter `clip_vit_b32` trong thư mục [`src/zoo/`](../src/zoo). Bóc tách thời gian chạy của nhánh Text Encoder (1D sequence) và Vision Encoder (2D patches), mở rộng phạm vi ứng dụng của framework sang thời đại Multimodal AI.

4. **Hướng 4: Hoàn Thiện Bài Toán Ablation Chuẩn Bị Cho Bài Báo Khoa Học**:
   - *Hành động*: Viết kịch bản `experiments/run_ablation_studies.py` để quét tự động $N_{warmup}$ từ 0 đến 50 và đo phương sai độ trễ, chứng minh điểm bão hòa tối ưu $N=15$; đồng thời bóc tách riêng rẽ chi phí overhead của Timer Hook, Memory Tracker và Background Sampler.

---

*Báo cáo nghiên cứu và kiểm toán này được hoàn thiện với độ chính xác cao nhất dựa trên phân tích trực tiếp mã nguồn của dự án VisionProf. Đây là tài liệu quy chuẩn học thuật phục vụ việc bảo vệ đồ án và định hướng các nghiên cứu tiếp theo.*
