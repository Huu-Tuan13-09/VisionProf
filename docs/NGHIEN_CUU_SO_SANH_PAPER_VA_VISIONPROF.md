# BÁO CÁO NGHIÊN CỨU & SO SÁNH CHUYÊN SÂU
## ĐỐI CHIẾU DỰ ÁN VISIONPROF VỚI HAI BÀI BÁO KHOA HỌC: PROOF (ICPP 2024) VÀ DEEPCONTEXT (ARXIV 2024)

> **Tài liệu tham chiếu chính**:
> 1. **Paper 1 (PRoof)**: *"PRoof: A Comprehensive Hierarchical Profiling Framework for Deep Neural Networks with Roofline Analysis"* — Siyu Wu, Hailong Yang, Xin You, Ruihao Gong, Yi Liu, Zhongzhi Luan, Depei Qian (Beihang University & SenseTime), công bố tại **ICPP '24** (The 53rd International Conference on Parallel Processing, 08/2024). [`paper1.pdf`](file:///d:/2027/XLDL/VisionProf/paper1.pdf).
> 2. **Paper 2 (DeepContext)**: *"DeepContext: A Context-aware, Cross-platform, and Cross-framework Tool for Performance Profiling and Analysis of Deep Learning Workloads"* — Qidong Zhao, Hao Wu, Yueming Hao, Zilingfeng Ye, Jiajia Li, Xu Liu, Keren Zhou (North Carolina State University & George Mason University), công bố trên **arXiv** (11/2024). [`paper2.pdf`](file:///d:/2027/XLDL/VisionProf/paper2.pdf).
> 3. **Dự án VisionProf**: Hệ thống profiling 3 mức đo lường cho mô hình AI trên CPU và GPU. Tài liệu kiến trúc và báo cáo thực nghiệm tại [`PROJECT_PLAN.md`](file:///d:/2027/XLDL/VisionProf/PROJECT_PLAN.md), [`TECHNICAL_SPECIFICATION.md`](file:///d:/2027/XLDL/VisionProf/TECHNICAL_SPECIFICATION.md), [`BAO_CAO_TV1.md`](file:///d:/2027/XLDL/VisionProf/docs/tv1/BAO_CAO_TV1.md), [`BAO_CAO_TV2.md`](file:///d:/2027/XLDL/VisionProf/docs/tv2/BAO_CAO_TV2.md), và [`BAO_CAO_TV3.md`](file:///d:/2027/XLDL/VisionProf/docs/tv3/BAO_CAO_TV3.md).

---

## MỤC LỤC
1. [Tóm tắt Tinh hoa của 2 Bài báo Khoa học](#1-tóm-tắt-tinh-hoa-của-2-bài-báo-khoa-học)
2. [Định vị Dự án VisionProf: Điểm mạnh & Kết quả Đã đạt](#2-định-vị-dự-án-visionprof-điểm-mạnh--kết-quả-đã-đạt)
3. [Ma trận So sánh Toàn diện (Comparative Matrix)](#3-ma-trận-so-sánh-toàn-diện-comparative-matrix)
4. [Phân tích Chi tiết Những Khoảng Trống (Gaps) VisionProf Còn Thiếu](#4-phân-tích-chi-tiết-những-khoảng-trống-gaps-visionprof-còn-thiếu)
   - 4.1. [Tầng Chặn & Khả Năng Thu Thập Ngữ Cảnh Đầy Đủ (Interception & Call-Path Stack)](#41-tầng-chặn--khả-năng-thu-thập-ngữ-cảnh-đầy-đủ-interception--call-path-stack)
   - 4.2. [Biên Dịch Đồ Thị, Gom Tầng (Operator Fusion) & Ánh Xạ Ngược (Layer Backtracking)](#42-biên-dịch-đồ-thị-gom-tầng-operator-fusion--ánh-xạ-ngược-layer-backtracking)
   - 4.3. [Mô Hình Hóa Bộ Nhớ SRAM/DRAM & Tính Toán FLOP Chuẩn Xác](#43-mô-hình-hóa-bộ-nhớ-sramdram--tính-toán-flop-chuẩn-xác)
   - 4.4. [Cấu Trúc Cây Ngữ Cảnh Gọi Hàm (Calling Context Tree - CCT)](#44-cấu-trúc-cây-ngữ-cảnh-gọi-hàm-calling-context-tree---cct)
   - 4.5. [Liên Kết Tự Động Chiều Ngược (Backward-to-Forward Attribution)](#45-liên-kết-tự-động-chiều-ngược-backward-to-forward-attribution)
   - 4.6. [Khả Năng Đa Nền Tảng Phần Cứng & Đa Khung Khái Niệm (Cross-Platform / Cross-Framework)](#46-khả-năng-đa-nền-tảng-phần-cứng--đa-khung-khái-niệm-cross-platform--cross-framework)
   - 4.7. [Cơ Chế Gom Mẫu Trực Tuyến Chống Bùng Nổ Dữ Liệu (Online Aggregation vs. Trace Bloat)](#47-cơ-chế-gom-mẫu-trực-tuyến-chống-bùng-nổ-dữ-liệu-online-aggregation-vs-trace-bloat)
   - 4.8. [Độ Sâu Của Bộ Luật Chẩn Đoán & Đề Xuất Tối Ưu Hóa Tự Động (Actionable Optimizations)](#48-độ-sâu-của-bộ-luật-chẩn-đoán--đề-xuất-tối-ưu-hóa-tự-động-actionable-optimizations)
   - 4.9. [Trực Quan Hóa Tương Tác & Tích Hợp Môi Trường Phát Triển (IDE Integration)](#49-trực-quan-hóa-tương-tác--tích-hợp-môi-trường-phát-triển-ide-integration)
5. [Lộ trình Đề xuất Nâng cấp VisionProf (Actionable Roadmap)](#5-lộ-trình-đề-xuất-nâng-cấp-visionprof-actionable-roadmap)
6. [Kết Luận](#6-kết-luận)

---

## 1. TÓM TẮT TINH HOA CỦA 2 BÀI BÁO KHOA HỌC

### 1.1. Paper 1: PRoof (ICPP 2024) — Hierarchical Profiling Framework with Roofline Analysis
Bài báo giải quyết hai bài toán hóc búa bậc nhất khi tối ưu hóa mô hình AI trong sản xuất (production inference runtimes):
1. **Khoảng cách ngữ nghĩa giữa Runtime tối ưu và Thiết kế mô hình gốc (Semantic Gap)**:
   Khi mô hình được đưa vào các runtime hiệu năng cao (TensorRT, OpenVINO, ONNX Runtime), đồ thị tính toán bị biến đổi hoàn toàn: nhiều tầng được gom lại (**Operator Fusion**, ví dụ Conv + Add + ReLU gộp thành 1 kernel duy nhất), các tầng chuyển đổi layout/datatype (Reorder, Transpose) được tự động chèn vào, và các backend hạ xuống hàng loạt CUDA kernel cụ thể. Các profiler phần cứng (như NVIDIA Nsight Compute - NCU) chỉ báo cáo tên kernel vi mô (mutilated mangled name) hoàn toàn vô nghĩa với kỹ sư thiết kế mô hình; ngược lại profiler của runtime chỉ đo độ trễ tầng backend mà không có chỉ số phần cứng.
   - **Giải pháp của PRoof**: Xây dựng **Layer Mapping / Backtracking** thông qua hai cấu trúc đồ thị trung gian: *Analyze Representation* (đồ thị ONNX gốc) và *Optimized Analyze Representation* (chứa nút ảo `_FusedOp`). Dùng giải thuật đối sánh I/O tensor (`get_subgraph_ops_by_io`) và bí danh tensor (`set_tensor_alias`) để liên kết 2 chiều giữa: **Python Source Code $\longleftrightarrow$ ONNX Nodes $\longleftrightarrow$ Backend Layers (TensorRT) $\longleftrightarrow$ Low-level CUDA Kernels**.
2. **Chi phí đo phần cứng quá lớn và thiếu tính khả chuyển**:
   Các công cụ dùng phần cứng hardware performance counters (như NCU) đòi hỏi cơ chế Kernel Replay (chạy đi chạy lại kernel hàng chục lần để đọc thanh ghi phần cứng), gây overhead khổng lồ (vài trăm đến hàng ngàn giây cho 1 mô hình), đồng thời không hỗ trợ các thiết bị biên không có tool như Raspberry Pi, Mobile NPU.
   - **Giải pháp của PRoof**: Xây dựng **Mô hình giải tích (Analytical Model)** dự đoán khối lượng FLOPs và lượng truy cập bộ nhớ DRAM (bytes) dựa trên shape inference:
     $$\text{Memory Access} = \sum_{p \in \text{params}} \text{Size}_p + \text{batch\_size} \times \left( \sum_{i \in \text{input}} \text{Size}_i + \sum_{o \in \text{output}} \text{Size}_o \right)$$
     *Phát hiện mấu chốt*: Với các tầng đã gom cụm (`_FusedOp`), dữ liệu trung gian giữa các toán tử con nằm hoàn toàn trên bộ nhớ đệm (on-chip SRAM/registers), không hề đọc/ghi qua DRAM. Do đó, chỉ tính kích thước tensor biên (vào/ra) của toàn cụm. Tốc độ đo của PRoof chỉ mất vài giây (chỉ đo latency thực của backend runtime rồi chia cho FLOP/Bytes giải tích), sai số dưới $2\text{--}8\%$ so với NCU.
   - **Case Studies đắt giá**: 
     - Tối ưu lại kiến trúc ShuffleNetV2 trên GPU A100: phát hiện thao tác Shuffle (Transpose/Data-copy) gây nghẽn băng thông bộ nhớ trầm trọng. Nhóm tác giả thiết kế lại khối cơ bản: bỏ Shuffle, tăng số kênh Conv (tăng FLOPs) nhưng loại bỏ hoàn toàn nghẽn bộ nhớ $\rightarrow$ tăng tốc throughput **1.64x**!
     - Tinh chỉnh tần số xung nhịp linh hoạt (DVFS) trên Jetson Orin NX: cân bằng xung nhịp GPU core và bộ nhớ EMC để tối ưu throughput dưới ngưỡng công suất 15W.

---

### 1.2. Paper 2: DeepContext (arXiv 11/2024) — Context-aware, Cross-platform Profiling
Bài báo tập trung vào vấn đề đứt gãy ngữ cảnh (Context Gap) và bùng nổ dữ liệu (Profile Bloat) trong các hệ thống phân tán và môi trường hỗn hợp:
1. **Đứt gãy toàn bộ chuỗi ngữ cảnh phần mềm**:
   Nsight Systems thấy C++ và GPU nhưng không hiểu Python và Operator AI; PyTorch profiler thấy Python/Operator nhưng không phân tích được C++ và mã máy GPU; JAX profiler không chạy được trên PyTorch; công cụ của NVIDIA không chạy được trên AMD ROCm.
   - **Giải pháp**: Tầng đệm **DLMonitor** (`libdlmonitor.so` tiêm qua `LD_PRELOAD`). Sử dụng API chính quy không cần sửa mã nguồn: PyTorch `aten::addGlobalCallback`, JAX binary instrumentation tại compilation pass, `CPython PyFrame` lấy call stack Python, `libunwind` lấy call stack C++, CUPTI (NVIDIA) và RocTracer (AMD) bắt GPU runtime/kernel. Ghép thành một chuỗi gọi thống nhất từ Python $\rightarrow$ Operator $\rightarrow$ C++ Library $\rightarrow$ GPU Kernel.
2. **Mất liên kết giữa Forward và Backward**:
   Trong PyTorch, tiến trình lan truyền ngược chạy trên các luồng CPU độc lập (backward worker threads), hoàn toàn tách rời ngữ cảnh Python nơi sinh ra forward operator. Khi GPU chạy một kernel như `indexing_backward_kernel`, kỹ sư không thể biết nó sinh ra từ dòng code Python nào.
   - **Giải pháp**: DeepContext lưu `sequence_id` và call path của forward pass; khi luồng backward kích hoạt với cùng `sequence_id`, hệ thống tự động ánh xạ ngược về dòng lệnh Python gốc của forward pass.
3. **Giải quyết vấn đề Bùng nổ Dữ liệu (Trace Bloat)**:
   Các profiler truyền thống ghi lại vết (trace) từng sự kiện, tạo ra file vài chục GB, gây tràn RAM (OOM) khi xuất dữ liệu. DeepContext sử dụng **Calling Context Tree (CCT)** và **Gom cụm trực tuyến (Online Streaming Aggregation)**: cộng dồn min/max/mean/stddev trực tiếp trên các nút cây ngữ cảnh, giúp giảm kích thước bộ nhớ từ **1.29x – 27x** so với PyTorch Profiler.
4. **Bộ Phân tích Chẩn đoán Tự động (Rule-Based Performance Analyzer)**:
   Cung cấp 5 luật phân tích mẫu cực kỳ thực chiến:
   - *Hotspot Identification*: Điểm nóng thời gian.
   - *Kernel Fusion Analysis*: Phát hiện các cụm kernel siêu nhỏ gọi liên tiếp (như softmax + copy + loss), gợi ý gom cụm bằng `torch.compile`.
   - *Forward/Backward Anomaly*: Phát hiện pha backward chậm gấp $>2\times$ so với forward (điển hình: `aten::index` tuần tự hóa các luồng truy cập trùng địa chỉ $\rightarrow$ đề xuất thay bằng `aten::index_select` dùng atomic operations, giúp tăng tốc **1.66x**).
   - *Fine-grained Stall Analysis*: Phân tích nguyên nhân dừng warp GPU bằng lấy mẫu lệnh (instruction sampling).
   - *CPU Latency / Thread Imbalance*: Phát hiện nghẽn CPU do đặt số luồng `DataLoader` (ví dụ 16 workers trên máy 6 cores gây context switch kéo dài 10s $\rightarrow$ giảm xuống 8 workers tiết kiệm 7s).
5. **Giao diện IDE Trực quan (VSCode Extension)**:
   Hiển thị biểu đồ ngọn lửa (Flame Graph) tương tác 2 chiều (Top-down và Bottom-up) tích hợp trực tiếp vào VSCode; nhấn chuột vào một khung (frame) là tự động mở file và nhảy đến đúng dòng mã nguồn Python gây lỗi.

---

## 2. ĐỊNH VỊ DỰ ÁN VISIONPROF: ĐIỂM MẠNH & KẾT QUẢ ĐÃ ĐẠT

Nhìn nhận một cách khách quan, dự án **VisionProf** của bạn đã hoàn thành xuất sắc các yêu cầu của một đề tài đo lường hiệu năng chuyên sâu ở cấp độ đại học/sau đại học:
- **Kiến trúc phân rã 3 mức đo lường hoàn chỉnh (3-Tier Profiling)**:
  - **Mức Layer (TV1)**: Zero-config leaf forward hook, áp dụng thành công kỹ thuật **Deferred CUDA Events** (chỉ đồng bộ 1 lần cuối forward pass) giúp triệt tiêu hiện tượng đo sai thời gian GPU do CPU không đồng bộ mà không phá vỡ pipeline hàng đợi GPU. Tích hợp tính toán FLOPs và dựng biểu đồ Roofline phân định rõ vùng Memory-bound vs Compute-bound.
  - **Mức Model (TV2)**: Bộ harness đo đạc 12 chỉ số thống kê phân phối chuẩn mực (P50, P90, P95, P99, Tail ratio, Peak VRAM qua allocator, RSS). Đã đo thật trên CPU laptop cho 8 mô hình đa lĩnh vực (CV, NLP, Tabular); so sánh với GPU Colab T4 (câu hỏi đảo thứ hạng CH3) đang chờ số đo T4 thật.
  - **Mức Giai đoạn (TV3)**: Bóc tách chu trình huấn luyện (DataLoader $\rightarrow$ Forward $\rightarrow$ Backward $\rightarrow$ Optimizer) và Pipeline suy luận (Preprocess $\rightarrow$ Forward $\rightarrow$ Postprocess/NMS). Thiết kế **19 Luật Chẩn đoán Heuristic v2.0** (sửa 4 lỗi logic v1.0) và Động cơ so sánh A/B có kiểm định thống kê Mann-Whitney U & Cliff's Delta.
  - **Dashboard trực quan**: Streamlit đa tab hỗ trợ phân tích tương tác và chẩn đoán tự động.

Tuy nhiên, khi đối chiếu với hai công trình nghiên cứu hàng đầu công bố năm 2024 (**PRoof** tại hội nghị hàng đầu ICPP '24 và **DeepContext** trên arXiv/SC/CGO ecosystem), **VisionProf bộc lộ nhiều khoảng trống kỹ thuật mang tính nền tảng (Foundational Architectural Gaps)**.

---

## 3. MA TRẬN SO SÁNH TOÀN DIỆN (COMPARATIVE MATRIX)

Dưới đây là bảng đối chiếu chi tiết giữa **VisionProf**, **PRoof**, và **DeepContext** trên 10 trục tiêu chí kỹ thuật:

| # | Tiêu chí Kỹ thuật | VisionProf (Hiện tại) | PRoof (ICPP 2024) | DeepContext (arXiv 2024) | Đánh giá Mức độ Thiếu của VisionProf |
| :-: | :--- | :--- | :--- | :--- | :--- |
| **1** | **Cơ chế Thu thập (Instrumentation)** | PyTorch Forward Hooks (`register_forward_hook`) & Context Managers (`PhaseTimer`). | Phân tích đồ thị tĩnh ONNX + Runtime Built-in Profilers + Vendor NCU. | Tầng shim **DLMonitor** (`LD_PRELOAD`), ATen Callbacks, JAX Binary Hook, `libunwind`. | 🔴 **Rất Lớn**: Hook mức PyTorch bỏ sót các phép toán dạng hàm (`torch.cat`, `x+y`, Attention matmul), gây 4–10% thời gian vô danh. |
| **2** | **Hỗ trợ Mô hình Biên Dịch & Runtime Tối Ưu** | ❌ Chỉ hỗ trợ PyTorch Eager Mode. Hỏng hoàn toàn khi dùng `torch.compile`, TensorRT, ONNX Runtime. | ✅ **Cốt lõi**: Hỗ trợ toàn diện TensorRT, OpenVINO, ONNX Runtime thông qua giải thuật **Layer Mapping & Backtracking**. | ✅ Hỗ trợ JAX JIT/XLA compilation; hiển thị ánh xạ giữa tầng gốc và cụm kernel đã gom. | 🔴 **Rất Lớn**: VisionProf không thể profile các mô hình đã compile/tối ưu trong production — nơi các tầng bị gom cụm (fused). |
| **3** | **Mô hình Hóa Bộ Nhớ (Memory Modeling)** | Giả định tĩnh: Đọc/ghi toàn bộ tensor từ RAM (DRAM Traffic). Bỏ qua hoàn toàn bộ nhớ đệm cache/SRAM. | ✅ **Phân rã On-chip SRAM vs DRAM**: Khi gom tầng (`_FusedOp`), tensor trung gian nằm trên SRAM, chỉ tính I/O biên ra DRAM. | Đo lường thực tế qua hardware counters (CUPTI / RocTracer) gán về cây ngữ cảnh. | 🟡 **Đáng Kể**: VisionProf tính sai băng thông bộ nhớ của các tầng nối tiếp/gom cụm (dẫn đến % hiệu suất Roofline bị ảo > 100%). |
| **4** | **Cấu trúc Ngữ cảnh Gọi (Calling Context)** | Bảng phẳng các leaf layers (`layers.csv`) hoặc các pha tuần tự (`phases.csv`). | Cấu trúc 4 tầng phân cấp: PyTorch Code $\rightarrow$ ONNX Nodes $\rightarrow$ Backend Layers $\rightarrow$ CUDA Kernels. | ✅ **Calling Context Tree (CCT)** đầy đủ từ Python $\rightarrow$ Framework $\rightarrow$ C++ $\rightarrow$ GPU Kernel $\rightarrow$ PC Instruction. | 🔴 **Rất Lớn**: VisionProf không có cấu trúc cây (tree), không biết layer được gọi từ đâu trong call stack phức tạp, không liên kết được xuống CUDA kernel. |
| **5** | **Đo Lường Huấn Luyện & Chiều Ngược (Backward)** | TV1 chỉ đo Forward. TV3 đo Backward như một hộp đen duy nhất (`with timer.phase("backward")`). | Tập trung vào Inference. | ✅ **Forward-Backward Association via Sequence ID**: Ánh xạ từng kernel backward rời rạc trên GPU về đúng dòng code Python forward. | 🔴 **Rất Lớn**: VisionProf hoàn toàn "mù" bên trong pha Backward (không biết layer nào gây nghẽn backward, ví dụ như lỗi atomic serialize của `aten::index`). |
| **6** | **Quy mô Dữ liệu & Quản lý Bộ nhớ Profiler** | Ghi log thô ra CSV từng iteration (`layers.csv`, `phases.csv`). File tăng tuyến tính theo thời gian chạy. | Tổng hợp theo từng layer backend. | ✅ **Gom cụm trực tuyến (Online Streaming Aggregation)** vào CCT; tiết kiệm bộ nhớ tới 27x, triệt tiêu nguy cơ OOM khi profile dài. | 🟡 **Đáng Kể**: VisionProf sẽ bị phình to dung lượng ổ đĩa và RAM khi profile quy mô lớn (hàng ngàn iteration). |
| **7** | **Khả Năng Đa Nền Tảng (Cross-Platform / Cross-HW)** | PyTorch trên CPU (laptop/Colab) và NVIDIA GPU (Tesla T4). | TensorRT, OpenVINO, ONNX Runtime trên NVIDIA GPU, Desktop GPU, x86 CPU, ARM CPU, Mobile NPU (Intel Ultra). | PyTorch & JAX trên NVIDIA GPU (CUPTI), AMD GPU (RocTracer / ROCm), x86 & ARM CPUs. | 🟡 **Đáng Kể**: VisionProf bị khóa cứng vào PyTorch và NVIDIA CUDA; không hỗ trợ AMD GPU, JAX, hay các NPU chuyên dụng. |
| **8** | **Độ Sâu Của Bộ Luật Chẩn Đoán (Diagnostics)** | 19 Luật Heuristic v2.0 dựa trên ngưỡng tĩnh (Threshold-based) phân tích trên file CSV. | Phân tích Roofline chuyên sâu, tái thiết kế khối mạng (ShuffleNet) và chỉnh tần số phần cứng (DVFS). | ✅ **5 Analysis Clients chuyên sâu**: Phát hiện kernel fusion, bất thường Backward/Forward, Instruction Stall analysis, Thread Imbalance. | 🟡 **Đáng Kể**: Luật của VisionProf chỉ dừng ở mức cảnh báo triệu chứng (warning), chưa đưa ra đề xuất mã nguồn cụ thể (Actionable Advice) như DeepContext. |
| **9** | **Khắc phục Sai số Phần cứng Chuyên sâu** | Sửa lỗi đồng bộ CUDA Event, lấp hàng đợi (`_sleep`), phát hiện đồng bộ ngầm (`attention_mask`). | Phát hiện và sửa lỗi tính sai FLOP Tensor Cores trong NVIDIA NCU (sửa số FLOP cố định 512 của lệnh HMMA/IMMA). | Sửa lỗi serialization của PyTorch `aten::index` trên GPU. | 🟢 **Ngang Tầm**: VisionProf đã làm rất tốt phần Deferred Sync và lấp hàng đợi, nhưng chưa đi sâu vào vi mã lệnh phần cứng như 2 paper. |
| **10** | **Giao diện Người Dùng & Tích Hợp (UX / GUI)** | Web Dashboard độc lập bằng Streamlit (Router `app.py`, 8 Tabs phân tích, biểu đồ Plotly). | PRoof DataViewer (Giao diện web trực quan hóa Roofline và layer latency). | ✅ **VSCode Extension WebView**: Biểu đồ ngọn lửa (Flame Graph) tương tác 2 chiều, **Click-to-Source Navigation** trực tiếp trong IDE. | 🟡 **Đáng Kể**: Streamlit của VisionProf tách rời khỏi IDE lập trình; thiếu Flame Graph và không thể nhảy trực tiếp đến dòng code gây lỗi. |

---

## 4. PHÂN TÍCH CHI TIẾT NHỮNG KHOẢNG TRỐNG (GAPS) VISIONPROF CÒN THIẾU

---

### 4.1. Tầng Chặn & Khả Năng Thu Thập Ngữ Cảnh Đầy Đủ (Interception & Call-Path Stack)

#### Hiện trạng của VisionProf:
- VisionProf (phần TV1) sử dụng `register_forward_pre_hook` và `register_forward_hook` gắn trên các `nn.Module` lá (leaf modules).
- **Hạn chế cố hữu đã được TV1 chỉ ra trong báo cáo**:
  1. *Phép tính dạng hàm bị bỏ sót hoàn toàn*: Các phép toán phổ biến như `x + shortcut` (residual connection trong ResNet), `torch.cat` (ghép tensor trong YOLO/ShuffleNet), hoặc phép nhân ma trận attention `torch.matmul` không thông qua `nn.Module`. Chúng tạo ra khoảng **"thời gian không thuộc layer nào" (unattributed time) từ 4% đến 10%**.
  2. *Không thấy mã máy C++ / CUDA Kernel*: VisionProf chỉ đo được ở mức đối tượng Python, không biết bên dưới module đó gọi bao nhiêu kernel GPU, kernel tên gì, hay thư viện cuDNN nào được thực thi.

#### Giải pháp đột phá từ DeepContext:
- DeepContext không phụ thuộc vào `nn.Module` hook. Thay vào đó, nó xây dựng tầng shim **DLMonitor** chèn qua `LD_PRELOAD`:
  - Trong PyTorch, nó móc vào giao diện cấp thấp `aten::addGlobalCallback`. Mọi toán tử ATen (kể cả phép cộng `aten::add`, ghép `aten::cat`, hay phép nhân ma trận `aten::matmul`) đều kích hoạt callback này.
  - Sử dụng `CPython` API để đọc con trỏ khung stack Python (`PyFrameObject`), dùng `libunwind` để mở cuộn stack C++, và dùng CUPTI/RocTracer để bắt kernel GPU.
  - **Kết quả**: DeepContext thu thập được chuỗi ngữ cảnh 4 tầng liền mạch:
    $$\text{Python Line (model.py:45)} \rightarrow \text{ATen Operator (aten::conv2d)} \rightarrow \text{C++ cuDNN Wrapper} \rightarrow \text{GPU Kernel (sm75\_xmma\_fprop)}$$
    Triệt tiêu hoàn toàn hiện tượng 10% unattributed time mà VisionProf đang mắc phải.

---

### 4.2. Biên Dịch Đồ Thị, Gom Tầng (Operator Fusion) & Ánh Xạ Ngược (Layer Backtracking)

#### Hiện trạng của VisionProf:
- VisionProf chỉ hoạt động trên PyTorch **Eager Mode**.
- Trong [BAO_CAO_TV1.md (Chương 7)](file:///d:/2027/XLDL/VisionProf/docs/tv1/BAO_CAO_TV1.md), TV1 thừa nhận:
  > *"Không áp dụng cho `torch.compile`, ONNX Runtime, TensorRT: Mô hình đã biên dịch gộp layer, hook không còn ý nghĩa. Đành dùng mức Model (đo hộp đen) cho các trường hợp này."*
- Đây là một điểm yếu rất lớn, vì trong thực tế triển khai sản xuất, hầu như không ai dùng Eager mode mà đều chuyển qua `torch.compile` (Inductor backend), ONNX Runtime, hoặc TensorRT. Khi đó, VisionProf hoàn toàn mất khả năng phân tích chi tiết.

#### Giải pháp đột phá từ PRoof:
- PRoof biến bài toán "Gom tầng trong Runtime" thành trọng tâm nghiên cứu:
  1. **Analyze Representation**: Đọc file ONNX tĩnh, phân tích shape của từng tensor, tạo danh sách các toán tử chuẩn (`Op defines`).
  2. **Optimized Analyze Representation**: Đại diện cho đồ thị sau khi runtime (như TensorRT) đã tối ưu. PRoof định nghĩa toán tử ảo `_FusedOp`. Một `_FusedOp` chứa một đồ thị con gồm nhiều layer gốc (ví dụ `[Conv, Add, ReLU]`).
  3. **Thuật toán Layer Mapping**:
     - Runtime TensorRT chèn các tầng định dạng dữ liệu như `reorder_1` ($t_2 \rightarrow t_{2\_r}$).
     - PRoof gọi API `set_tensor_alias(original=t2, alias=t2_r)`.
     - Tìm kiếm đồ thị con dựa trên input/output: `get_subgraph_ops_by_io(inputs=[t1, t2_r], outputs=[t3])` $\rightarrow$ trả về danh sách các toán tử con `[conv_2, add_3, relu_4]`.
     - Gói thành `set_fused_op(name="fused_op_2", ops=[conv_2, add_3, relu_4])`.
  - **Ý nghĩa**: PRoof ánh xạ được 2 chiều: từ kernel runtime bị biến đổi quay ngược trở lại chính xác từng dòng code và từng module trong thiết kế ban đầu của kỹ sư.

---

### 4.3. Mô Hình Hóa Bộ Nhớ SRAM/DRAM & Tính Toán FLOP Chuẩn Xác

#### Hiện trạng của VisionProf:
- VisionProf tính toán lưu lượng bộ nhớ (Memory Traffic) của một layer theo công thức:
  $$\text{Bytes} = \text{Size}_{\text{input}} + \text{Size}_{\text{weight}} + \text{Size}_{\text{output}}$$
- **Sai số khoa học**: Công thức này giả định rằng *mỗi layer đều phải nạp input từ RAM/DRAM và ghi output về RAM/DRAM*. 
- Trong [BAO_CAO_TV1.md (Chương 7)](file:///d:/2027/XLDL/VisionProf/docs/tv1/BAO_CAO_TV1.md), TV1 nhận định:
  > *"Lưu lượng bộ nhớ tính theo mô hình 'đọc hết từ RAM' làm các layer nhỏ có dữ liệu nằm trong cache đạt hiệu suất Roofline > 100%."*

#### Giải pháp đột phá từ PRoof:
- PRoof phân định ranh giới bộ nhớ rất rõ ràng dựa trên cơ chế phần cứng:
  - Các mạng nơ-ron sâu có luồng dữ liệu tĩnh (static control flow).
  - Khi hai hoặc nhiều toán tử được thực thi liên tiếp (đặc biệt khi được compiler gộp lại), **output của toán tử trước được giữ lại trên SRAM/Register trên chip** để làm input cho toán tử sau, hoàn toàn không đẩy ra DRAM.
  - Khi tính toán cho `_FusedOp`, PRoof **loại bỏ toàn bộ dung lượng của các tensor trung gian**:
    $$\text{Memory}_{\text{fused}} = \sum \text{Params} + \text{batch\_size} \times \left( \text{Input}_{\text{subgraph}} + \text{Output}_{\text{subgraph}} \right)$$
  - Điều này giải thích tại sao PRoof đạt sai số bộ nhớ chỉ **1.28% – 1.37%** so với công cụ phần cứng chuyên dụng NVIDIA NCU trên A100.
- **Sửa lỗi tính FLOP của Tensor Cores**: PRoof còn phát hiện ra NVIDIA NCU tính sai số FLOP trên Tensor Core (NCU mặc định nhân cố định 512 cho mọi lệnh HMMA/IMMA của kiến trúc Volta cũ). PRoof đã dùng kỹ thuật dịch ngược vi kiến trúc (micro-benchmarking) để hiệu chỉnh số FLOP chuẩn xác cho từng dòng GPU (Ampere/Ada Lovelace).

---

### 4.4. Cấu Trúc Cây Ngữ Cảnh Gọi Hàm (Calling Context Tree - CCT)

#### Hiện trạng của VisionProf:
- Dữ liệu thu thập của VisionProf được lưu dưới dạng danh sách phẳng (flat table) theo thời gian trong file `layers.csv`.
- Mặc dù TV1 đã giải quyết được vấn đề "Module dùng chung" (như SiLU trong YOLOv8n) bằng cơ chế `call_index`, nhưng VisionProf hoàn toàn không ghi nhận được **ngữ cảnh cha - con (Caller - Callee)**. Nếu một hàm tiện ích được gọi từ 5 vị trí khác nhau trong pipeline, VisionProf không thể phân rã thời gian cho từng luồng gọi.

#### Giải pháp đột phá từ DeepContext:
- DeepContext tổ chức toàn bộ dữ liệu profiling dưới dạng **Calling Context Tree (CCT)**:
  - Mỗi nút (node) trong cây đại diện cho một hàm/toán tử tại một vị trí mã nguồn cụ thể (được đồng nhất bởi đường dẫn file, số dòng, địa chỉ con trỏ lệnh PC).
  - Thuộc tính của một nút gồm có: số lần gọi (`count`), tổng thời gian (`gpu_time`, `cpu_time`), phân phối phương sai, đỉnh bộ nhớ, và số lượng warp stall.
  - Khi một sự kiện kết thúc ở lá, số liệu được lan truyền ngược (propagate) lên nút gốc của cây.
  - **Lợi ích**: Cho phép hiển thị biểu đồ ngọn lửa (Flame Graph) cả hai chiều:
    - **Top-down view**: Giúp đi từ hàm `main()` xuống sâu dần để tìm hàm nào tiêu tốn thời gian nhất.
    - **Bottom-up view**: Gom tất cả các lần gọi của cùng một kernel GPU (ví dụ `nchwToNhwcKernel`) từ khắp nơi trong chương trình lại để thấy tổng thiệt hại và truy vết ngược lên các nơi gọi nó.

---

### 4.5. Liên Kết Tự Động Chiều Ngược (Backward-to-Forward Attribution)

#### Hiện trạng của VisionProf:
- TV1 chỉ gắn hook vào chiều `forward` khi suy luận (`eval`). TV1 chưa cài đặt `register_full_backward_hook` cho từng layer.
- TV3 đo chiều `backward` trong thực nghiệm TN6, nhưng chỉ đo ở mức **Giai đoạn gộp (Coarse Phase)**:
  ```python
  with timer.phase("backward"):
      loss.backward()
  ```
- Kết quả trong `phases.csv` chỉ cho biết: "Toàn bộ pha backward mất 12ms (gấp đôi forward)", nhưng **hoàn toàn không thể trả lời câu hỏi**: *Layer nào trong số 100 layer đang làm chậm pha backward? Toán tử nào bị nghẽn trong backward?*

#### Giải pháp đột phá từ DeepContext:
- Trong PyTorch, cơ chế Autograd sinh ra các luồng CPU riêng biệt (`torch::autograd::Engine::thread_main`) để thực thi backward graph. Tại thời điểm này, call stack Python đã bị hủy, chỉ còn mã C++ thuần túy.
- DeepContext giải quyết bằng cơ chế **Sequence ID Caching**:
  1. Trong pha Forward: DLMonitor bắt từng toán tử, gán một `sequence_id` duy nhất và lưu cache đường dẫn gọi Python vào biến `thread-local`.
  2. Trong pha Backward: Luồng autograd kế thừa `sequence_id` từ đồ thị tính toán. DeepContext dùng ID này tra cứu lại đường dẫn gọi Forward ban đầu.
  3. **Case Study thực tế trong bài báo**: Trên mô hình gợi ý DLRM-small, pha backward chạy chậm bất thường ($39.9\%$ tổng thời gian trong khi forward chỉ mất $0.8\%$). Nhờ liên kết forward-backward, DeepContext phát hiện thủ phạm là `indexing_backward_kernel` xuất phát từ lệnh `embedding_table[idx_lookup]` (toán tử `aten::index`). Thao tác này trên PyTorch bị khóa đồng bộ tuần tự (serialization) do tính tất định (determinism). Đổi sang `aten::index_select` dùng atomic add đã giúp **tăng tốc mô hình 1.66x**!
  - *VisionProf hiện tại hoàn toàn bất lực trước những lỗi hiệu năng kiểu này.*

---

### 4.6. Khả Năng Đa Nền Tảng Phần Cứng & Đa Khung Khái Niệm (Cross-Platform / Cross-Framework)

#### Hiện trạng của VisionProf:
- VisionProf được thiết kế cho hệ sinh thái **PyTorch thuần túy** (chỉ có thêm adapter dạng hộp đen cho XGBoost qua CPU/GPU API của thư viện xgboost).
- Về phần cứng: Chỉ hỗ trợ kiến trúc x86 CPU và GPU NVIDIA CUDA (qua `torch.cuda.Event`).
- Khi chạy trên Windows laptop có cạc đồ họa (WDDM driver), TV1 phát hiện ngưỡng sàn CUDA Event bị vọt lên $21\mu s$ (gấp 3 lần Linux) làm sai lệch số đo, do đó dự án phải co cụm phạm vi về Google Colab Linux.

#### Mở rộng từ PRoof & DeepContext:
- **PRoof**:
  - Chạy trên 7 nền tảng phần cứng cực kỳ đa dạng: NVIDIA A100 (Data Center GPU), RTX 4090 (Desktop GPU), Jetson Orin NX (Edge GPU), Jetson Xavier NX, Intel Xeon Gold (Data Center CPU), Raspberry Pi 4B (Edge CPU ARM), và **Intel Meteor Lake NPU 3720** (Mobile NPU).
  - Chạy trên 3 runtime công nghiệp: TensorRT, OpenVINO, ONNX Runtime.
- **DeepContext**:
  - Hỗ trợ cả **PyTorch và JAX** (thông qua binary instrumentation trong trình biên dịch XLA).
  - Hỗ trợ song song cả **NVIDIA GPUs** (qua CUPTI) và **AMD GPUs** (qua RocTracer trên nền tảng ROCm).
  - Hỗ trợ cả x86 và ARM CPUs (dùng PAPI và Linux Perf events).
  - *Case Study AMD vs NVIDIA*: DeepContext chỉ ra rằng cùng toán tử `aten::instance_norm` nhưng chạy trên AMD MI250 lại bị chậm bất thường do PyTorch dùng chung template CUDA của NVIDIA (vốn tối ưu cho warp size 32, trong khi AMD có warp size 64) $\rightarrow$ cung cấp giải pháp chỉnh số thread per CTA để tăng tốc.

---

### 4.7. Cơ Chế Gom Mẫu Trực Tuyến Chống Bùng Nổ Dữ Liệu (Online Aggregation vs. Trace Bloat)

#### Hiện trạng của VisionProf:
- Mỗi vòng lặp (iteration) đo đạc, VisionProf ghi từng dòng thông tin của từng layer vào bộ nhớ RAM (`self.records`) rồi ghi ra đĩa `layers.csv`.
- Với 50 vòng đo trên ResNet-50 (126 layer $\times$ 50 = 6,300 dòng), kích thước file còn nhỏ. Nhưng nếu profile quá trình huấn luyện thật trong 1,000–10,000 steps, lượng dữ liệu CSV sẽ lên tới hàng triệu dòng, làm cạn kiệt RAM và khiến bước vẽ biểu đồ Streamlit bị treo (OOM hoặc lag).

#### Giải pháp đột phá từ DeepContext:
- DeepContext không lưu vết phẳng theo thời gian (Trace-based).
- Thay vào đó, nó áp dụng cơ chế **Online Streaming Aggregation**:
  - Khi một sự kiện kernel/operator kết thúc, số liệu ngay lập tức được cộng dồn vào nút tương ứng trên Calling Context Tree bằng công thức trực tuyến (Welford's algorithm để tính mean và variance).
  - Không cần lưu chi tiết từng millisecond của từng lần gọi trong quá khứ.
  - Bộ nhớ RAM tiêu thụ của DeepContext duy trì ổn định $1.00\times \text{--} 2.44\times$, trong khi PyTorch Profiler tiêu tốn từ $1.29\times$ đến $27.28\times$ RAM và thường xuyên bị crash Out-Of-Memory khi xuất file trace lớn.

---

### 4.8. Độ Sâu Của Bộ Luật Chẩn Đoán & Đề Xuất Tối Ưu Hóa Tự Động (Actionable Optimizations)

#### Hiện trạng của VisionProf:
- Hệ thống **19 Luật Heuristic v2.0** của TV3 trong [`analyzer.py`](file:///d:/2027/XLDL/VisionProf/src/analyzer.py) là một điểm sáng lớn (đã chia 4 nhóm RA, RB, RC, RD; sửa các lỗi RB-04, RD-02, RD-05, RC-03).
- **Hạn chế**:
  - Các luật của VisionProf chỉ dừng lại ở việc **Cảnh báo triệu chứng (Symptom Detection)**: ví dụ *"Cảnh báo: Tỷ lệ nạp dữ liệu > 20%"*, *"Cảnh báo: Layer chiếm > 30% thời gian"*, *"Cảnh báo: Tỷ số P99/P50 > 2.0"*.
  - Người dùng đọc xong cảnh báo vẫn phải **tự mày mò code** để tìm cách sửa, vì profiler không chỉ ra được dòng mã nguồn cụ thể hay kỹ thuật biến đổi code tương ứng.

#### Đột phá từ DeepContext & PRoof (Actionable Optimization Guidance):
Cả hai bài báo đều tiến thêm một bước dài: từ chẩn đoán đi thẳng đến **giải pháp tái cấu trúc mã nguồn (Code Refactoring Action)**:
1. **Chẩn đoán Kernel Fusion**: DeepContext phát hiện một hàm gọi chuỗi nhiều kernel con kích thước micro-giây (như softmax, copy, nll_loss) $\rightarrow$ Đưa ra khuyến nghị cụ thể: *"Bọc đoạn code này bằng `@torch.compile` hoặc viết Triton kernel gộp"*. Giúp giảm thời gian từ 30.5s xuống 23.9s trên Transformer-Big.
2. **Chẩn đoán Memory Format Layout**: DeepContext phát hiện kernel `cudnn::nchwToNhwcKernel` chiếm 15.4% GPU time trong U-Net $\rightarrow$ Chỉ ra nguyên nhân: PyTorch mặc định dùng `channels_first` nhưng cuDNN tối ưu trên `channels_last`, dẫn đến việc GPU liên tục hoán đổi định dạng $\rightarrow$ Khuyến nghị: Chuyển tensor đầu vào sang `to(memory_format=torch.channels_last)` ngay từ đầu pipeline $\rightarrow$ Giúp U-Net giảm từ 54s xuống 42s.
3. **Chẩn đoán Phân bổ Luồng CPU (Overscheduling)**: DeepContext phát hiện hàm nạp dữ liệu chiếm 69% CPU time vì cấu hình 16 worker threads trên máy chỉ có 6 core vật lý, gây quá tải bộ điều phối hệ điều hành $\rightarrow$ Khuyến nghị giảm xuống 8 workers $\rightarrow$ Giảm độ trễ từ 54s xuống 47s.
4. **Tái cấu trúc Kiến trúc Mô hình (Model Redesign)**: PRoof dùng biểu đồ Roofline chỉ ra phép Shuffle trong ShuffleNetV2 làm GPU A100 bị nghẽn băng thông nghiêm trọng $\rightarrow$ Đề xuất bỏ Shuffle, tăng kênh Conv $\rightarrow$ Tăng throughput 1.64x.

---

### 4.9. Trực Quan Hóa Tương Tác & Tích Hợp Môi Trường Phát Triển (IDE Integration)

#### Hiện trạng của VisionProf:
- Giao diện của VisionProf xây dựng bằng **Streamlit** độc lập (`src/dashboard/`).
- **Ưu điểm**: Dễ dựng, có nhiều biểu đồ đẹp (Plotly, bảng số liệu), phù hợp cho việc thuyết trình báo cáo và demo tổng quan.
- **Hạn chế trong quy trình lập trình thực tế**:
  - Tách rời hoàn toàn khỏi môi trường viết code (VS Code / PyCharm).
  - Khi dashboard báo *"Layer Conv2d_3 bị nghẽn"*, kỹ sư phải tự mở VS Code lên, dùng công cụ tìm kiếm xem `Conv2d_3` nằm ở file nào, dòng bao nhiêu.
  - Thiếu hẳn dạng biểu đồ **Flame Graph** (biểu đồ ngọn lửa) — định dạng chuẩn mực số 1 thế giới hiện nay để hiển thị phân cấp thời gian thực thi (được cả Linux Perf, PyTorch, Nsight sử dụng).

#### Giải pháp đột phá từ DeepContext:
- DeepContext loại bỏ Web Dashboard rời rạc, chuyển toàn bộ sang **VSCode Extension** sử dụng công nghệ WebView kết hợp đồ họa WebGL:
  - **Flame Graph tương tác đa chiều**: Cho phép cuộn, phóng to (zoom in) từng nhánh gọi hàm, chuyển đổi tức thì giữa góc nhìn Top-down và Bottom-up.
  - **Click-to-Source Navigation**: Kỹ sư AI chỉ cần nhấn chuột trái vào bất kỳ ô nào trên Flame Graph (dù là một phép toán ATen hay một kernel GPU), VSCode sẽ **tự động mở file `.py` tương ứng và nhảy con trỏ chuột đến đúng dòng lệnh đó**. Điều này rút ngắn thời gian debug từ hàng chục phút xuống còn 1 giây.

---

## 5. LỘ TRÌNH ĐỀ XUẤT NÂNG CẤP VISIONPROF (ACTIONABLE ROADMAP)

Để đưa VisionProf từ một đồ án môn học vươn lên thành một nghiên cứu có đóng góp học thuật thực sự tiệm cận PRoof và DeepContext, dưới đây là lộ trình 4 giai đoạn nâng cấp khuyến nghị:

```
+---------------------------------------------------------------------------------------+
|                       LỘ TRÌNH NÂNG CẤP VISIONPROF v3.0                              |
+---------------------------------------------------------------------------------------+
|  GIAI ĐOẠN 1: NÂNG CẤP ĐỘ SÂU ĐO LƯỜNG & KHẮC PHỤC THỜI GIAN VÔ DANH (QUÝ 1)          |
|  - Chuyển từ nn.Module Hooks sang TorchDispatchMode / PyTorch Global Callbacks        |
|  - Thu thập toàn bộ phép tính dạng hàm (torch.cat, x+shortcut, attention matmul)      |
|  - Triệt tiêu 10% "unattributed time" của TV1                                         |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|  GIAI ĐOẠN 2: HỖ TRỢ BIÊN DỊCH MÔ HÌNH & ÁNH XẠ GOM TẦNG (QUÝ 2 - HỌC TẬP PROOF)     |
|  - Hỗ trợ mô hình torch.compile (Inductor) và xuất ONNX -> TensorRT                   |
|  - Cài đặt cấu trúc đồ thị _FusedOp và giải thuật ánh xạ ngược (Layer Backtracking)   |
|  - Mô hình hóa bộ nhớ SRAM vs DRAM: loại bỏ trung gian tensor khi tính Roofline       |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|  GIAI ĐOẠN 3: CÂY NGỮ CẢNH CCT & LIÊN KẾT BACKWARD CHI TIẾT (QUÝ 3 - HỌC DEEPCONTEXT) |
|  - Xây dựng Calling Context Tree (CCT) kết hợp gom mẫu trực tuyến (Online Aggregation)|
|  - Cài đặt Sequence ID để liên kết kernel Backward ngược về mã nguồn Python Forward   |
|  - Mở rộng bộ luật: Chẩn đoán Format (channels_last), Kernel Fusion, và Overscheduling|
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|  GIAI ĐOẠN 4: TRỰC QUAN HÓA FLAME GRAPH & TÍCH HỢP IDE (QUÝ 4)                        |
|  - Tích hợp biểu đồ ngọn lửa (Flame Graph) vào Streamlit Dashboard hiện tại           |
|  - Phát triển VSCode Plugin hỗ trợ Click-to-Source Navigation                         |
|  - Thử nghiệm mở rộng sang phần cứng AMD ROCm và Jetson Edge                          |
+---------------------------------------------------------------------------------------+
```

### Chi tiết các module cần viết mới hoặc tái cấu trúc trong VisionProf:

1. **`src/core/dispatch_tracer.py` (Thay thế cho Hook lá hạn chế của TV1)**:
   - Tận dụng `torch.utils._python_dispatch.TorchDispatchMode` trong PyTorch 2.x để chặn mọi phép toán ATen mà không cần sửa code mô hình. Bắt trọn 100% thời gian thực thi, kể cả các phép tính dạng hàm.
2. **`src/layer/fused_mapping.py` (Áp dụng ý tưởng từ PRoof)**:
   - Đọc đồ thị ONNX hoặc đồ thị FX của `torch.compile`.
   - Nhận diện các cụm bị compiler gộp lại (ví dụ Conv + BatchNorm + ReLU gom thành một Triton kernel).
   - Tự động trừ khử lưu lượng DRAM đối với các tensor nội bộ trong cụm, giúp đường biểu diễn Roofline phản ánh đúng thực tế 100%.
3. **`src/phase/backward_tracer.py` (Áp dụng ý tưởng từ DeepContext)**:
   - Đăng ký hook trên tensor gradient hoặc can thiệp vào engine autograd để ghi nhận thời gian backward của từng layer đơn lẻ, thay vì chỉ đo 1 cục `loss.backward()` như hiện nay.
4. **Nâng cấp `src/analyzer/rules/` (Thêm 4 luật đề xuất tối ưu hóa hành động)**:
   - `RC-05 (DataLoader Thread Overscheduling)`: So sánh `num_workers` với `os.cpu_count()`. Nếu vượt quá số core vật lý và gây nghẽn CPU, tự động đề xuất giá trị tối ưu $\approx \text{cpu\_count} // 2$.
   - `RD-06 (Memory Format Sub-optimal)`: Kiểm tra nếu mô hình là CNN chạy trên GPU Tensor Core nhưng tensor chưa chuyển `channels_last`, phát cảnh báo đề xuất `.to(memory_format=torch.channels_last)`.
   - `RD-07 (Fusion Opportunity)`: Quét các chuỗi layer liên tiếp có kích thước micro-giây (RD-02) và đề xuất áp dụng `torch.compile(mode="reduce-overhead")`.
5. **Nâng cấp `src/dashboard/tabs/tab_flamegraph.py`**:
   - Thêm tab biểu đồ ngọn lửa (dùng thư viện `plotly` hoặc nhúng Web Component d3-flame-graph) cho phép xem cấu trúc gọi hàm phân cấp.

---

## 6. KẾT LUẬN

Qua việc nghiên cứu chuyên sâu hai công trình khoa học tiêu biểu năm 2024:
- **PRoof (ICPP 2024)** đã chỉ ra con đường giải quyết triệt để **bài toán triển khai thực tế (Production Deployment)**: vượt qua rào cản của việc tối ưu hóa gom tầng (Operator Fusion) trong các runtime đóng gói (TensorRT, OpenVINO) bằng giải thuật ánh xạ ngược đồ thị (Layer Backtracking) và mô hình giải tích bộ nhớ SRAM/DRAM không cần phần cứng đắt đỏ.
- **DeepContext (arXiv 2024)** đã giải quyết mẫu mực **bài toán ngữ cảnh toàn diện (Full-stack Calling Context)**: kết nối từ mã nguồn Python cấp cao xuống tận mã máy GPU, hóa giải điểm mù của pha lan truyền ngược (Backward Attribution), nén dữ liệu đo lường trực tuyến bằng cây CCT và hỗ trợ tối ưu hóa tương tác trực tiếp trong IDE.

Đối chiếu lại, **VisionProf** hiện tại là một nền tảng vững chắc, sở hữu phương pháp luận đo lường GPU khoa học (Deferred CUDA Events), bao phủ trọn vẹn 3 mức từ Layer đến Model và Pipeline, cùng hệ thống 19 luật chẩn đoán v2.0 được thiết kế bài bản. 

Những điểm **còn thiếu** được phân tích trong báo cáo này không phải là sai sót, mà là **khoảng cách giữa một đồ án đo lường mô hình PyTorch đơn lẻ (Academic Prototype) và một Hệ thống Profiling công nghiệp hiện đại (Production-grade System Tool)**. Việc tiếp thu những kỹ thuật đột phá từ PRoof và DeepContext sẽ là kim chỉ nam hoàn hảo để nâng tầm dự án VisionProf trong các giai đoạn phát triển tiếp theo.
