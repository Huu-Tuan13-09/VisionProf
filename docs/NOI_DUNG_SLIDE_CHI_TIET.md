# BỘ NỘI DUNG SLIDE THUYẾT TRÌNH VÀ KỊCH BẢN BẢO VỆ ĐỒ ÁN VISIONPROF
## HỆ THỐNG PHÂN TÍCH HIỆU NĂNG MÔ HÌNH HỌC SÂU ĐA TẦNG (LAYER, MODEL, PIPELINE)
### Tự Động Hóa Chẩn Đoán Điểm Nghẽn và Đánh Giá Thực Nghiệm Trên Google Colab CPU và Tesla T4 GPU

> **Thời lượng báo cáo chính thức:** 24.0 phút (23 slide)  
> **Thời lượng phản biện hội đồng:** 6.0 - 10.0 phút  
> **Tổng thời lượng phiên bảo vệ:** 30.0 - 35.0 phút  
> **Phân công trách nhiệm 3 thành viên:**  
> - **Thành viên 3 (TV3)**: Mở đầu, Khung bài toán, Mức Giai đoạn (Huấn luyện & Pipeline), Động cơ chẩn đoán 19 quy tắc và Tổng kết (Slide 1-3, 17-23; thời lượng 9.5 phút).  
> - **Thành viên 1 (TV1)**: Nghiên cứu liên quan, Cơ sở lý thuyết Mức Layer, Kỹ thuật Deferred CUDA Events, Mô hình Roofline, Kiểm chứng CH1 và CH2 (Slide 4-10; thời lượng 8.0 phút).  
> - **Thành viên 2 (TV2)**: Mức Model, Khung đo lường 2 pha chuẩn MLPerf, Phân phối trễ P50-P99, Thực nghiệm đối đầu 8 mô hình trên Google Colab CPU vs. Tesla T4 GPU, Batch size, FP16 Tensor Cores, Kiểm chứng CH3 và CH4 (Slide 11-16; thời lượng 6.5 phút).  
> **Môi trường thực nghiệm chuẩn hóa:** Toàn bộ dữ liệu thu thập trực tiếp trên nền tảng Google Colab:
> - **Colab CPU (`colab_cpu`)**: 2 vCPUs Intel Xeon @ 2.20GHz, 12.7 GB System RAM, kiến trúc x86_64, Linux Ubuntu 22.04.
> - **Colab GPU (`colab_t4`)**: NVIDIA Tesla T4 16GB GDDR6, 2,560 nhân CUDA, 320 nhân Turing Tensor Cores, giao tiếp PCIe Gen3 x16, trên cùng máy chủ 2 vCPUs Intel Xeon.  
> **Cơ sở khoa học:** Đối chiếu và kiểm chứng trực tiếp với 10 công trình khoa học công bố tại các hội thảo và tạp chí hàng đầu (*CACM, ACM/IEEE ISCA, IEEE HPCA, PVLDB, IEEE IISWC, IEEE IPDPS, IEEE TPDS, MLSys, ICPP, ECCV*).

---

## BẢNG ĐIỀU PHỐI VÀ TIẾN ĐỘ TRÌNH BÀY (24 PHÚT)

| Phần nội dung | Phụ trách | Slide | Thời lượng | Trọng tâm khoa học và thực nghiệm |
| :--- | :---: | :---: | :---: | :--- |
| **Phần 1: Mở đầu và Khung bài toán** | **TV3** | **1-3** | **2.5 phút** (150s) | Định vị Profiling trong hệ thống phần mềm AI, Mô hình 3 mức Canonical Modular, Trục đối đầu chuẩn ResNet-50 vs. ViT / DistilBERT |
| **Phần 2: Nghiên cứu liên quan và Khoảng trống** | **TV1** | **4-6** | **3.0 phút** (180s) | Kiến trúc đa tầng MLModelScope (L1-L5), Mô hình Roofline và Nghịch lý Overhead (XSP), Ma trận phân tích công cụ hiện hữu |
| **Phần 3: Mức Layer và Độ chuẩn xác đo lường** | **TV1** | **7-10** | **5.0 phút** (300s) | Thách thức bất đồng bộ CUDA, Kỹ thuật Deferred CUDA Events, CH1 (Tỷ trọng toán tử Conv/Linear/Attention), CH2 (Kiểm chứng sai số với `torch.profiler`) |
| **Phần 4: Mức Model và Đánh giá toàn diện** | **TV2** | **11-16** | **6.5 phút** (390s) | Model Zoo 8 Adapter, Phương pháp đo sạch 2 pha, Chuẩn phân phối trễ MLPerf P50-P99, CH3 (Nghịch lý số tham số tại Batch 1, Đảo thứ hạng phần cứng Colab CPU vs. T4), CH4 (Mở rộng Batch 1-32, FP16 Tensor Cores) |
| **Phần 5: Mức Giai đoạn, Chẩn đoán và Tổng kết** | **TV3** | **7.0 phút** (420s) | CH5 (Tỷ số Backward/Forward 2x, Bùng nổ VRAM 3x, Nghẽn DataLoader và Starvation), CH6 (41% thời gian ngoài mô hình, Định luật Amdahl), 19 quy tắc chẩn đoán v2.0, Kiểm định A/B phi tham số, Video Dashboard, Tổng kết 6 câu hỏi nghiên cứu |
| **Toàn bộ bài báo cáo** | **Cả 3 TV** | **23 slide** | **24.0 phút** | **Phiên hỏi đáp, phản biện và thao tác trực tiếp trên Dashboard (6-10 phút)** |

---

# PHẦN 1: MỞ ĐẦU VÀ KHUNG BÀI TOÁN (SLIDE 1-3)
### Người báo cáo: Thành viên 3 (TV3) | Thời lượng: 2.5 phút (150 giây)

---

### SLIDE 1. TIÊU ĐỀ BÁO CÁO VÀ TỔNG QUAN ĐỀ TÀI
- **Thời lượng:** 40 giây.
- **Bố cục hiển thị:**
  - Tiêu đề trung tâm: **VISIONPROF: FRAMEWORK PHÂN TÍCH HIỆU NĂNG MÔ HÌNH HỌC SÂU ĐA TẦNG**.
  - Tiêu đề phụ: *Giải Phẫu Hiệu Năng Từ Cấp Độ Toán Tử Phần Cứng Đến Đường Ống Ứng Dụng Thực Tế*.
  - Khung định vị phân công trách nhiệm:
    - **TV3 (Điều phối & Giai đoạn):** Huấn luyện (Training), Đường ống xử lý (Pipeline), 19 Quy tắc chẩn đoán tự động và Giao diện tương tác Streamlit.
    - **TV1 (Mức Layer):** Zero-config Leaf Hooks, Cơ chế Deferred CUDA Events, Mô hình lý thuyết Roofline và Kiểm chuẩn sai số.
    - **TV2 (Mức Model):** Model Zoo 8 Adapters, Trễ phân phối P50-P99 theo chuẩn MLPerf, Thực nghiệm đối đầu Colab CPU vs. Tesla T4 GPU, Batch size Scaling và FP16 Tensor Cores.
  - Đơn vị đào tạo: *Bộ môn Khoa học Máy tính / Xử lý Dữ liệu Nâng cao*.
- **Nội dung học thuật cốt lõi:**
  - **Mục tiêu hệ thống:** Thiết lập giải pháp White-box Profiling không xâm lấn (*Zero-code modification*) cho các mô hình PyTorch, tách bạch trễ phần cứng và bộ nhớ tại 3 cấp độ phân rã.
  - **Đóng góp kỹ thuật:** Liên kết chuỗi đo lường khép kín: Mức Toán tử (Layer) $\rightarrow$ Mức Mô hình (Model) $\rightarrow$ Mức Toàn bộ ứng dụng (Pipeline & Training), tích hợp cơ chế chẩn đoán 19 quy tắc dựa trên nguyên lý kiến trúc máy tính.
- **Mã nguồn liên quan:** [`src/core/interfaces.py`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/src/core/interfaces.py), [`src/core/types.py`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/src/core/types.py).
- **Lời thoại thuyết trình (Speaker Script):**
  > "Kính thưa quý thầy cô trong Hội đồng đánh giá và toàn thể các bạn!  
  > Trong kỹ nghệ triển khai các hệ thống Học sâu vào môi trường sản xuất quy mô lớn, một nghịch lý kinh điển thường xuyên xảy ra: một mô hình được thiết kế với số lượng tham số rất nhỏ hoặc số phép tính FLOPs thấp trên lý thuyết, nhưng khi đưa vào vận hành thực tế lại có độ trễ lớn và gây tắc nghẽn toàn bộ đường ống phục vụ.  
  > Để giải quyết căn cơ bài toán này, nhóm chúng em đã nghiên cứu và xây dựng hệ thống **VisionProf: Framework phân tích hiệu năng mô hình học sâu ở 3 mức đo lường chuẩn hóa: Mức Layer, Mức Model và Mức Giai đoạn**. Đề tài được hoàn thiện với sự cộng tác chuyên sâu của 3 thành viên: bạn TV1 phụ trách Mức Layer và tính chuẩn xác phần cứng; bạn TV2 phụ trách Mức Model và đối đầu thực nghiệm giữa CPU và GPU trên nền tảng Google Colab; và em - TV3 - phụ trách Mức Giai đoạn, hệ thống 19 quy tắc chẩn đoán tự động và điều phối tổng thể."

---

### SLIDE 2. ĐẶT VẤN ĐỀ: PHÂN ĐỊNH RANH GIỚI EVALUATION, BENCHMARKING VÀ PROFILING
- **Thời lượng:** 50 giây.
- **Bố cục hiển thị:** Bảng ma trận 3 cột phân định tam giác hiệu năng, đóng khung làm nổi bật cột **Profiling**.
- **Nguồn trích dẫn:** *Dakkak et al., 'MLModelScope: Evaluating and Profiling ML Models Across the Stack,' ACM/IEEE IPDPS, 2020.*
- **Nội dung trên slide:**

| Trục đánh giá | Câu hỏi nghiên cứu trọng tâm | Đại lượng đo lường đặc trưng | Phạm vi trong VisionProf |
| :--- | :--- | :--- | :---: |
| **Evaluation** *(Đánh giá chất lượng)* | Mô hình đưa ra suy luận **chính xác** đến mức nào trên tập kiểm thử? | Accuracy, Top-1/Top-5, mAP@0.5:0.95, F1-Score, BLEU, Perplexity. | **Ngoài phạm vi** *(Chỉ dùng Cosine $\ge 0.999999$ để kiểm soát chất lượng FP16)* |
| **Benchmarking** *(Đo lường hộp đen)* | Toàn bộ mô hình chạy **nhanh** bao nhiêu và tiêu tốn bao nhiêu tài nguyên hệ thống? | P50, P90, P99 Latency, Thông lượng Throughput (FPS), Peak VRAM (MB), Host RSS. | **Mức Model (TV2)** |
| **Profiling** *(Giải phẫu hộp trắng)* | **Thời gian và bộ nhớ thực sự tiêu tốn ở đâu?** Hệ thống nghẽn do năng lực tính toán, băng thông bộ nhớ, hay độ trễ khởi động kernel? | Layer latency, Roofline % Peak, Tỷ số Backward/Forward, Trễ nạp dữ liệu (I/O Wait), Chi phí ngoài mô hình (AI Tax). | **Trọng tâm cốt lõi (TV1 & TV3)** |

- **Thông điệp khoa học cốt lõi:** *"Chúng ta không thể tối ưu hóa những đại lượng mà chúng ta không thể lượng hóa chính xác. VisionProf tập trung vào trục Profiling - giải phẫu cấu trúc thực thi của framework để xác định nguyên nhân gốc rễ của sự suy giảm hiệu năng."*
- **Lời thoại thuyết trình (Speaker Script):**
  > "Để bảo đảm tính chuẩn mực khoa học, trước hết chúng em xin làm rõ ranh giới giữa 3 khái niệm thường bị đồng nhất trong thực tế phát triển phần mềm AI.  
  > Trục thứ nhất là **Evaluation**: đánh giá chất lượng dự đoán của mô hình thông qua các độ đo như Top-1 Accuracy hay mAP. Đây là bài toán khoa học dữ liệu và nằm ngoài phạm vi của đề tài.  
  > Trục thứ hai là **Benchmarking**: tiếp cận mô hình như một hộp đen nhằm xác định tổng độ trễ mili-giây và mức chiếm dụng bộ nhớ tổng thể. Đây là nhiệm vụ của Mức Model do bạn TV2 đảm nhận.  
  > Đóng góp học thuật trọng tâm của VisionProf nằm ở trục thứ ba: **Profiling**. Chúng em tiến hành giải phẫu hộp trắng để trả lời chính xác: Chu kỳ thực thi của phần cứng thực sự tiêu hao ở đâu? Mô hình đang bị giới hạn bởi trần tính toán của ALU, trần băng thông của bộ nhớ GDDR6, hay do độ trễ truyền lệnh qua giao tiếp PCIe? Chi phí do toán tử mạng chiếm ưu thế hay do công đoạn tiền xử lý và giải nén dữ liệu?"

---

### SLIDE 3. KIẾN TRÚC 3 MỨC ĐO CHUẨN HÓA VÀ SỢI CHỈ ĐỎ THỰC NGHIỆM
- **Thời lượng:** 60 giây.
- **Bố cục hiển thị:**
  - Trái: Sơ đồ khối kiến trúc phân tầng Canonical Modular (`src/layer/` $\rightarrow$ `src/model/` $\rightarrow$ `src/phase/` $\rightarrow$ `src/analyzer/` $\rightarrow$ `src/dashboard/`).
  - Phải: Bảng thiết lập cặp mô hình đối đầu và thông số phần cứng thử nghiệm.
- **Nội dung trên slide:**
  - **Mô hình kiến trúc 3 mức đo chuẩn hóa:**
    - **Mức Layer (TV1):** Zero-config Hook trên các Leaf Modules, cơ chế Deferred CUDA Events, phân tích FLOPs và ánh xạ lên Mô hình Roofline.
    - **Mức Model (TV2):** Quy trình đo sạch 2 pha, phân phối thời gian thực thi P50..P99, thông lượng FPS, đỉnh VRAM và Host RSS trên Model Zoo 8 kiến trúc.
    - **Mức Giai đoạn (TV3):** Module `PhaseTimer` bóc tách chu trình Huấn luyện (Forward/Backward/Optimizer) và Đường ống ứng dụng (Preprocess/Model/Postprocess), `DataLoaderSentinel`, 19 quy tắc chẩn đoán v2.0 và kiểm định A/B phi tham số.
  - **Sợi chỉ đỏ thực nghiệm đối đầu (Cross-Cutting Experimental Baseline):**
    - Đại diện Mạng tích chập cổ điển: **ResNet-50 (25.6M tham số, 8.2 GFLOPs)**.
    - Đại diện Kiến trúc Attention thế hệ mới: **ViT-B/16 (86.6M tham số, 35.2 GFLOPs)** và **DistilBERT (66.4M tham số, 43.6 GFLOPs)**.
  - **Hệ quy chiếu phần cứng chuẩn hóa trên Google Colab:**
    - CPU: Intel Xeon @ 2.20GHz (2 vCPUs, Haswell/Broadwell, 12.7 GB RAM) -> Môi trường `colab_cpu`.
    - GPU: NVIDIA Tesla T4 16GB GDDR6 (Turing, 2,560 CUDA Cores, 320 Tensor Cores, PCIe Gen3 x16) -> Môi trường `colab_t4`.
- **Mã nguồn liên quan:** [`src/core/types.py`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/src/core/types.py), [`PROJECT_PLAN.md`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/PROJECT_PLAN.md).
- **Lời thoại thuyết trình (Speaker Script):**
  > "Nhằm giải quyết toàn diện bài toán profiling mà không phá vỡ tính bao đóng, VisionProf được thiết kế theo cấu trúc Modular 3 tầng nghiêm ngặt: Mức Layer đo lường vi mô từng toán tử leaf; Mức Model đo lường vĩ mô toàn bộ mạng nơ-ron; và Mức Giai đoạn bao quát chu trình huấn luyện cũng như ứng dụng hoàn chỉnh.  
  > Xuyên suốt 23 slide báo cáo hôm nay, Hội đồng sẽ theo dõi một **sợi chỉ đỏ thực nghiệm**: Đó là cuộc so tài hiệu năng chuyên sâu giữa đại diện mạng tích chập CNN là **ResNet-50** và đại diện kiến trúc Transformer là **ViT-B/16** và **DistilBERT** trên cùng một hạ tầng đám mây Google Colab, đối chiếu trực tiếp giữa CPU Intel Xeon và GPU tăng tốc Tesla T4.  
  > Sau đây, em xin kính mời bạn TV1 trình bày về cơ sở lý thuyết, các nghiên cứu nền tảng và thiết kế giải thuật ở Mức Layer."
- **Lời chuyển giao:** TV3 $\rightarrow$ TV1: *"Kính mời bạn TV1 tiếp tục phần báo cáo."*

---

# PHẦN 2: NGHIÊN CỨU LIÊN QUAN VÀ KHOẢNG TRỐNG KỸ THUẬT (SLIDE 4-6)
### Người báo cáo: Thành viên 1 (TV1) | Thời lượng: 3.0 phút (180 giây)

---

### SLIDE 4. BỨC TRANH TOÀN CẢNH ĐA TẦNG HỆ THỐNG: MÔ HÌNH L1 ĐẾN L5
- **Thời lượng:** 55 giây.
- **Bố cục hiển thị:** Sơ đồ 5 tầng trừu tượng xếp chồng từ phần cứng vật lý lên ứng dụng phần mềm (Kế thừa khung MLModelScope).
- **Nguồn trích dẫn:** *Dakkak et al., 'MLModelScope: Evaluating and Profiling ML Models Across the Stack,' ACM/IEEE IPDPS, 2020.*
- **Nội dung trên slide:**

```
+-------------------------------------------------------------------------+
| L5: Ứng dụng AI thực tế (Web API, YOLO Pipeline, Data Pipeline)         | <- Mức Giai đoạn (TV3)
+-------------------------------------------------------------------------+
| L4: Mô hình Học sâu (Mạng ResNet, Transformer, XGBoost, Tensors)        | <- Mức Model (TV2)
+-------------------------------------------------------------------------+
| L3: Framework / Graph Execution Engine (PyTorch Autograd, TorchScript)  | <- Cơ chế Hook & Autograd
+-------------------------------------------------------------------------+
| L2: Thư viện toán tử tăng tốc (cuDNN v8.9, cuBLAS, TensorRT, MKL)        | <- Mức Layer (TV1)
+-------------------------------------------------------------------------+
| L1: Phần cứng vật lý (Google Colab: Intel Xeon Host vs. Tesla T4 SMs)  | <- Hạ tầng thực nghiệm
+-------------------------------------------------------------------------+
```

- **Phân tích học thuật:**
  - **Tầng L1 - L2:** Trực tiếp quyết định thời gian thực thi của từng toán tử đơn lẻ. Điểm nghẽn phát sinh do trần băng thông DRAM hoặc trần tính toán của nhân Tensor Cores.
  - **Tầng L3 - L4:** Chịu trách nhiệm quản lý biểu đồ tính toán (Computation Graph), lập lịch gửi kernel và phân bổ bộ nhớ động thông qua Caching Allocator.
  - **Tầng L5:** Nơi các chi phí ngoài mô hình (I/O, chuẩn hóa dữ liệu, giải thuật tuần tự trên CPU) chiếm ưu thế.
- **Lời thoại thuyết trình (Speaker Script):**
  > "Kính thưa quý thầy cô, để xây dựng một công cụ profiling có độ tin cậy khoa học, chúng em xuất phát từ mô hình phân tầng hệ thống L1 đến L5 được đề xuất bởi nhóm nghiên cứu Đại học Illinois tại hội thảo IPDPS 2020.  
  > Một lỗi phổ biến trong công tác đánh giá hệ thống AI là sự cô lập giữa các tầng: kỹ sư phần cứng chỉ quan sát L1-L2 thông qua các bộ đếm chu kỳ phần cứng; trong khi kỹ sư phần mềm chỉ nhìn thấy L4-L5 qua thời gian hoàn thành tác vụ. Sự ngắt quãng này che giấu các xung đột hệ thống.  
  > VisionProf được thiết kế để kết nối toàn diện chuỗi phân tầng này: TV1 nắm giữ L2 qua việc đo trắc vi mô từng toán tử; TV2 kiểm soát L3-L4 qua sự phối hợp của runtime PyTorch; và TV3 bao quát toàn bộ L5 để làm rõ tương tác giữa mô hình với hệ điều hành và luồng dữ liệu vào ra."

---

### SLIDE 5. MÔ HÌNH LÝ THUYẾT ROOFLINE VÀ NGHỊCH LÝ OVERHEAD ĐO LƯỜNG
- **Thời lượng:** 65 giây.
- **Bố cục hiển thị:**
  - Trái: Đồ thị Log-Log 2 chiều của Mô hình Roofline, thể hiện đường trần băng thông GDDR6 và đường trần tính toán của GPU Tesla T4.
  - Phải: Sơ đồ phân tích Nghịch lý Overhead trong thu thập dữ liệu (Mô hình XSP).
- **Nguồn trích dẫn:**
  - *S. Williams, A. Waterman, and D. Patterson, 'Roofline: An Insightful Visual Performance Model for Multicore Architectures,' Communications of the ACM, 2009.*
  - *Dakkak et al., 'Across-Stack Profiling and Analysis of Machine Learning Workloads (XSP),' ACM/IEEE ISCA, 2021.*
- **Nội dung trên slide:**
  - **1. Mô hình Lý thuyết Roofline (Williams et al., CACM 2009):**
    - Cường độ tính toán (Operational Intensity):
      $$I = \frac{\text{FLOPs}}{\text{DRAM Traffic (Bytes)}}$$
    - Hiệu năng tối đa đạt được (Attainable Performance):
      $$P = \min\left(P_{\text{peak}}, I \times \text{BW}_{\text{peak}}\right)$$
    - Điểm cân bằng phần cứng của Tesla T4 (FP32):
      $$I^* = \frac{8.1 \times 10^{12} \text{ FLOPs/s}}{300 \times 10^9 \text{ Bytes/s}} = 27.0 \text{ FLOPs/Byte}$$
    - Khi $I < I^*$: Toán tử bị nghẽn băng thông bộ nhớ (Memory-Bandwidth Bound). Khi $I \ge I^*$: Toán tử đạt trần tính toán (Compute-Bound).
  - **2. Nghịch lý Overhead trong đo lường hệ thống (XSP, ISCA 2021):**
    - Hiệu ứng quan sát viên (Observer Effect): Thu thập thông tin vi mô càng chi tiết thì sự can thiệp của công cụ đo càng làm méo mó bản chất vận hành của hệ thống.
    - Với các toán tử sub-microsecond (như ReLU, BatchNorm ở batch 1), chi phí kích hoạt phép đo có thể lớn hơn chính thời gian thực thi của toán tử.
- **Lời thoại thuyết trình (Speaker Script):**
  > "Về cơ sở lý thuyết hiệu năng, VisionProf áp dụng Mô hình Roofline được công bố bởi Giáo sư David Patterson trên Communications of the ACM.  
  > Trên phần cứng NVIDIA Tesla T4 ở chế độ FP32, trần tính toán đạt 8.1 TFLOPS và băng thông bộ nhớ GDDR6 là 300 GB/s. Điểm gãy cân bằng kiến trúc (Ridge Point) là 27 FLOPs trên mỗi Byte. Nếu một tầng mạng có cường độ tính toán nhỏ hơn 27, việc tăng thêm số nhân xử lý là hoàn toàn vô ích vì GPU phải dừng chờ nạp dữ liệu từ bộ nhớ.  
  > Đồng thời, bài báo XSP tại ISCA 2021 đã chỉ ra một thách thức sinh tử: 'Nghịch lý Overhead'. Khi chúng ta can thiệp sâu vào từng toán tử để đo đạc, chính mã nguồn profiling sẽ phá vỡ tính liên tục của luồng lệnh, làm thay đổi cấu trúc hàng đợi phần cứng. Do đó, bài toán đặt ra cho Mức Layer là: Làm thế nào đo đạc chính xác cấu trúc vi mô mà không làm méo mó bức tranh hiệu năng tổng thể?"

---

### SLIDE 6. MA TRẬN SO SÁNH CÔNG CỤ VÀ KHOẢNG TRỐNG KỸ THUẬT CỦA VISIONPROF
- **Thời lượng:** 60 giây.
- **Bố cục hiển thị:** Bảng so sánh 5 chiều giữa các giải pháp công nghiệp hiện hữu và VisionProf.
- **Nguồn trích dẫn:**
  - *Fang et al., 'PyTorch Profiler,' MLSys, 2020.*
  - *Ben-Nun et al., 'Deep500: An Modular Benchmarking Infrastructure for Deep Learning,' IEEE TPDS, 2020.*
- **Nội dung trên slide:**

| Tiêu chí so sánh kỹ thuật | NVIDIA Nsight Systems | PyTorch Profiler (`torch.profiler`) | Deep500 Benchmark | VisionProf (Framework đề xuất) |
| :--- | :---: | :---: | :---: | :---: |
| **Mức độ phụ thuộc phần cứng** | Độc quyền GPU NVIDIA (CUDA/NVLink) | Đa nền tảng (CPU/GPU) | Tập trung tính phân tán đa node | **Chuẩn hóa Colab CPU và Tesla T4 GPU** |
| **Độ phức tạp tích hợp mã nguồn** | Chạy nhị phân độc lập từ CLI | Cần chèn context manager thủ công | Sửa đổi kiến trúc mô hình | **Zero-config Hook (Hoàn toàn tự động)** |
| **Kích thước file vết thực thi (Trace)** | Cực lớn (Hàng trăm MB đến hàng GB) | Lớn (File Chrome JSON hàng chục MB) | Báo cáo số liệu dạng bảng | **Nhẹ gọn (< 1 MB cấu trúc JSON/CSV)** |
| **Khả năng bóc tách đường ống (Pipeline)** | Không phân biệt ngữ nghĩa Pipeline | Phải tự gắn nhãn RecordFunction | Không hỗ trợ Pipeline thực tế | **Tách tự động Pre / Model / Postprocess** |
| **Động cơ suy luận và chẩn đoán** | Không (Người dùng tự suy diễn) | Không (Chỉ xuất bảng thô) | Không hỗ trợ chẩn đoán | **19 Quy tắc Heuristic chuẩn hóa v2.0** |

- **Khoảng trống kỹ thuật mà VisionProf khỏa lấp:**
  - Cung cấp một framework nhẹ, có khả năng tự động hóa chẩn đoán điểm nghẽn dựa trên luật chuyên gia, bao quát trọn vẹn cả 3 mức đo lường mà không đòi hỏi chỉnh sửa mã nguồn mô hình.
- **Lời thoại thuyết trình (Speaker Script):**
  > "Khi đặt VisionProf cạnh các giải pháp công nghiệp hàng đầu, chúng ta thấy rõ vị trí chiến lược của đề tài.  
  > NVIDIA Nsight Systems cung cấp độ chi tiết phần cứng tuyệt đối nhưng chỉ chạy trên GPU NVIDIA và sinh ra tệp trace khổng lồ hàng trăm megabyte, đòi hỏi chuyên gia phần cứng phân tích thủ công. PyTorch Profiler chuẩn hóa tốt trong hệ sinh thái nhưng yêu cầu người dùng phải can thiệp sửa mã nguồn bằng các context manager và hoàn toàn không có khả năng tự chẩn đoán điểm nghẽn.  
  > VisionProf lấp đầy khoảng trống kỹ thuật này: Chúng em cung cấp cơ chế Hook tự động không xâm lấn, tệp kết quả siêu nhẹ dưới 1 megabyte, bóc tách toàn diện chu trình ứng dụng và đặc biệt là tích hợp sẵn động cơ chẩn đoán 19 quy tắc tự động hóa việc phát hiện nguyên nhân gốc rễ."

---

# PHẦN 3: MỨC LAYER VÀ ĐỘ CHUẨN XÁC ĐO LƯỜNG PHẦN CỨNG (SLIDE 7-10)
### Người báo cáo: Thành viên 1 (TV1) | Thời lượng: 5.0 phút (300 giây)

---

### SLIDE 7. THÁCH THỨC BẤT ĐỒNG BỘ CUDA VÀ GIẢI PHÁP ZERO-CONFIG HOOK
- **Thời lượng:** 75 giây.
- **Bố cục hiển thị:**
  - Trái: Biểu đồ thời gian (Timeline) so sánh giữa Đo đồng bộ ngây thơ (Naive Sync) và Luồng lệnh bất đồng bộ thực tế.
  - Phải: Cây phân cấp mô hình và cơ chế đăng ký Hook trên Leaf Modules.
- **Nội dung trên slide:**
  - **1. Thách thức cốt lõi: Bản chất bất đồng bộ của CUDA Runtime:**
    - Hàm gọi trên CPU (`forward()`) chỉ thực hiện đẩy lệnh khởi tạo kernel (Kernel Launch Enqueue) vào hàng đợi của GPU Stream rồi trả về ngay lập tức (mất $\approx 5-10\,\mu\text{s}$).
    - Việc dùng hàm đo thời gian CPU thông thường (`time.perf_counter()`) quanh một tầng mạng chỉ đo thời gian gửi lệnh, hoàn toàn sai lệch với thời gian thực thi trên GPU.
    - Nếu gọi đồng bộ hóa cưỡng bức (`torch.cuda.synchronize()`) sau mỗi tầng: Phá vỡ tính năng thực thi ống dẫn (Pipeline execution) của phần cứng GPU, làm rỗng hàng đợi lệnh và khiến tổng thời gian đo bị đội lên từ 150% đến 300%.
  - **2. Giải pháp Zero-Config Leaf Hook:**
    - Sử dụng thuật toán duyệt đệ quy cây mô hình: Chỉ gắn `register_forward_pre_hook` và `register_forward_hook` trên các **Leaf Modules** (mô hình con không chứa submodule).
    - Triệt tiêu hoàn toàn lỗi đếm trùng lặp thời gian (Double-counting) giữa khối cha (như `Bottleneck`) và khối con (như `Conv2d`, `BatchNorm2d`).
- **Mã nguồn liên quan:** [`src/layer/hook_manager.py`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/src/layer/hook_manager.py) (các dòng 45-98).
- **Lời thoại thuyết trình (Speaker Script):**
  > "Bây giờ, em xin đi sâu vào thiết kế giải thuật ở Mức Layer.  
  > Thách thức lớn nhất khi đo lường mô hình học sâu trên GPU là tính chất thực thi bất đồng bộ. Khi CPU gọi một hàm tính toán PyTorch, nó chỉ ghi lệnh vào hàng đợi phần cứng của GPU rồi tiếp tục thực thi lệnh tiếp theo. Nếu lập trình viên ngây thơ dùng hàm `time.perf_counter()` của CPU, con số thu được chỉ là thời gian đẩy lệnh khoảng 5 micro-giây chứ không phải thời gian tính toán thực tế.  
  > Ngược lại, nếu chèn lệnh `torch.cuda.synchronize()` sau từng tầng, chúng ta sẽ bắt GPU phải dừng chờ hoàn toàn, làm triệt tiêu khả năng chạy song song nhiều kernel nhỏ của kiến trúc Turing và làm sai số tổng thời gian lên tới hàng trăm phần trăm.  
  > Đồng thời, để không can thiệp vào mã nguồn của người dùng, VisionProf duyệt tự động cây tính toán và chỉ gắn hook vào các Leaf Modules. Điều này bảo đảm không bao giờ xảy ra hiện tượng đếm trùng lặp giữa tầng cha và các tầng con bên trong."

---

### SLIDE 8. KỸ THUẬT DEFERRED CUDA EVENTS VÀ KIỂM CHỨNG THỰC NGHIỆM CH2
- **Thời lượng:** 75 giây.
- **Bố cục hiển thị:**
  - Trái: Sơ đồ ngăn xếp LIFO Stack của kỹ thuật Deferred CUDA Events.
  - Phải: Bảng số liệu kiểm chuẩn độ chính xác so với chuẩn vàng `torch.profiler` trên Google Colab T4.
- **Nội dung trên slide:**
  - **1. Cơ chế Kỹ thuật Deferred CUDA Events (LIFO Stack):**
    - Khi hook kích hoạt, ghi lại các mốc sự kiện phần cứng không đồng bộ: `torch.cuda.Event(enable_timing=True)`.
    - Đẩy cặp sự kiện `(start_event, end_event)` vào cấu trúc ngăn xếp LIFO cho từng module (xử lý hoàn hảo các khối mạng được tái sử dụng nhiều lần trong đồ thị tính toán).
    - **Hoãn đồng bộ hóa (Deferred Sync):** Trong suốt quá trình forward toàn mạng, tuyệt đối không gọi đồng bộ. Chỉ khi toàn bộ mạng hoàn thành, hệ thống gọi đúng 1 lệnh `torch.cuda.synchronize()` duy nhất, sau đó truy vấn thời gian vi mô bằng hàm phần cứng `start.elapsed_time(end)`.
  - **2. Kết quả kiểm chuẩn CH2 so sánh với `torch.profiler` (Đo thực tế trên Colab T4):**

| Chỉ số kiểm định chuẩn xác | ResNet-50 (Báo cáo $\rightarrow$ Thực nghiệm) | BERT-base (Báo cáo $\rightarrow$ Thực nghiệm) | Kết luận khoa học |
| :--- | :---: | :---: | :--- |
| **Tổng thời gian các tầng (ms)** | 5.10 ms $\rightarrow$ **5.13 ms** | 6.27 ms $\rightarrow$ **6.30 ms** | Ổn định tuyệt đối qua các phiên đo |
| **Sai lệch so với `torch.profiler`** | 7.5% $\rightarrow$ **7.8%** | 0.8% $\rightarrow$ **0.04%** | **Đạt chuẩn xuất sắc ($< 8.0\%$)** |
| **Thời gian không thuộc tầng nào** | 9.0% $\rightarrow$ **9.5%** | 10.3% $\rightarrow$ **10.5%** | Lượng hóa chính xác Functional Ops |
| **Overhead tại Batch 1** | 166% $\rightarrow$ 213% | 109% $\rightarrow$ 133% | Chi phí ghi nhận hàng trăm sự kiện |

- **Thông điệp khoa học cốt lõi:**
  - Khẳng định tính đúng đắn của giả thuyết CH2: Sai lệch tổng thời gian so với `torch.profiler` luôn kiểm soát dưới **$8.0\%$** trên GPU và dưới **$1.0\%$** trên CPU.
  - Mức Layer được thiết kế chuyên biệt cho mục đích phân tích tỷ trọng cấu trúc (Structural Breakdown); tổng độ trễ chuẩn tuyệt đối được giao cho Mức Model quản lý.
- **Mã nguồn liên quan:** [`src/layer/timer.py`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/src/layer/timer.py), dữ liệu tại `results/analysis/TN2/recheck_fixed_code/`.
- **Lời thoại thuyết trình (Speaker Script):**
  > "Để hóa giải bài toán bất đồng bộ mà không gây méo mó luồng thực thi, VisionProf triển khai kỹ thuật **Deferred CUDA Events với cấu trúc ngăn xếp LIFO**.  
  > Trong hook của từng tầng, chúng em chỉ chèn các đánh dấu sự kiện phần cứng cực nhẹ vào luồng lệnh của CUDA. Toàn bộ tiến trình forward diễn ra hoàn toàn tự nhiên ở tốc độ nguyên bản của GPU. Chỉ khi toàn bộ mô hình kết thúc, một lệnh đồng bộ duy nhất được kích hoạt và thời gian phần cứng được tính toán hồi cứu.  
  > Bảng kiểm chứng CH2 bên phải chứng minh độ chuẩn xác khoa học: So với công cụ tham chiếu chuẩn mực `torch.profiler`, sai lệch tổng thời gian của VisionProf trên GPU Tesla T4 chỉ là **7.8% đối với ResNet-50** và **chỉ 0.04% đối với BERT-base**!  
  > Chúng em cũng phân lập được đại lượng 'Thời gian không thuộc tầng nào' chiếm từ 9% đến 10.5%, phản ánh chính xác các toán tử hàm dạng functional như phép cộng residual `x + shortcut` vốn không có hook quản lý."

---

### SLIDE 9. KẾT QUẢ CH1: PHÂN RÃ CƠ CẤU THỜI GIAN TOÁN TỬ THEO KIẾN TRÚC
- **Thời lượng:** 75 giây.
- **Bố cục hiển thị:**
  - Trái: Biểu đồ tròn biểu diễn tỷ trọng thời gian các loại toán tử trên ResNet-50 và BERT-base (`docs/tv1/figures/fig_ch1_layer_breakdown.png`).
  - Phải: Bảng đối chiếu tỷ lệ thực tế giữa Mạng tích chập (CNN) và Mạng biến đổi (Transformer).
- **Nguồn trích dẫn:**
  - *K. He et al., 'Deep Residual Learning for Image Recognition,' IEEE CVPR, 2016.*
  - *J. Devlin et al., 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,' NAACL, 2019.*
- **Nội dung trên slide:**

| Họ kiến trúc mô hình | Toán tử thống trị thời gian | Tỷ trọng trên CPU | Tỷ trọng trên GPU Tesla T4 | Bản chất cơ chế tính toán |
| :--- | :--- | :---: | :---: | :--- |
| **ResNet-50** *(Mạng tích chập CNN)* | **Convolution 2D (`Conv2d`)** | **$85.2\%$** | **$73.8\%$** | Phép trượt cửa sổ tích chập, được tối ưu qua thư viện cuDNN |
| | Chuẩn hóa (`BatchNorm2d`) | $6.8\%$ | $14.6\%$ | Phần tử độc lập (Element-wise), bị nghẽn băng thông bộ nhớ |
| | Kích hoạt (`ReLU`) & Khác | $8.0\%$ | $11.6\%$ | Chi phí điều phối và tính toán cơ bản |
| **BERT-base** *(Mạng Transformer)* | **Tuyến tính (`Linear` - GEMM)** | **$96.1\%$** | **$95.4\%$** | Phép nhân ma trận tổng quát của cơ chế Multi-Head Attention và FFN |
| | Chuẩn hóa (`LayerNorm`) | $2.4\%$ | $3.1\%$ | Thao tác quy chuẩn phân phối cục bộ |
| | Kích hoạt (`GELU`) & Khác | $1.5\%$ | $1.5\%$ | Phép tính xấp xỉ hàm phi tuyến |

- **3 Kết luận khoa học trả lời câu hỏi CH1:**
  1. Trong kiến trúc CNN, toán tử **Convolution chiếm áp đảo từ 73% đến 85%** tổng thời gian tính toán.
  2. Trong kiến trúc Transformer, toán tử **Linear hoàn toàn thống trị với trên 95%** thời gian (phản ánh các phép nhân ma trận $Q, K, V$ và khối Feed-Forward).
  3. Khi chuyển từ CPU sang GPU, tỷ trọng của các tầng tính toán nặng (Conv) có xu hướng co lại, nhường chỗ cho tỷ trọng của các tầng nhẹ (BatchNorm tăng từ 6.8% lên 14.6%) do đặc tính trần băng thông và chi phí khởi tạo kernel.
- **Mã nguồn liên quan:** [`src/layer/analyzer.py`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/src/layer/analyzer.py), kết quả tại `results/colab_t4/*/TN1/`.
- **Lời thoại thuyết trình (Speaker Script):**
  > "Trả lời câu hỏi nghiên cứu CH1: Thời gian thực sự tập trung ở những loại tầng mạng nào?  
  > Kết quả thực nghiệm của VisionProf mang lại câu trả lời vô cùng dứt khoát:  
  > Với mạng tích chập ResNet-50, toán tử tích chập Conv2d chiếm tới 85% thời gian trên CPU và 74% trên GPU Tesla T4. Toàn bộ các tầng kích hoạt ReLU hay Pooling chỉ chiếm một phần rất nhỏ.  
  > Ngược lại hoàn toàn, với kiến trúc Transformer như BERT-base, các phép nhân ma trận tổng quát của tầng Linear chiếm tới hơn 95% toàn bộ chu kỳ tính toán, bao trùm cả cơ chế tính điểm Self-Attention lẫn mạng nơ-ron truyền thẳng FFN.  
  > Đặc biệt, khi dịch chuyển từ CPU sang GPU, chúng ta nhận thấy tỷ trọng của các tầng chuẩn hóa BatchNorm tăng hơn gấp đôi: từ 6.8% lên 14.6%. Hiện tượng này sẽ được làm sáng tỏ ngay trong slide tiếp theo."

---

### SLIDE 10. KẾT QUẢ CH1: SỰ PHÂN HÓA HỆ SỐ TĂNG TỐC VÀ PHÂN LOẠI ROOFLINE
- **Thời lượng:** 75 giây.
- **Bố cục hiển thị:**
  - Trái: Biểu đồ phân tán (Scatter Plot) ánh xạ các tầng mạng lên tọa độ Roofline (Trục hoành: Cường độ tính toán $I$, Trục tung: GFLOPs/s thực tế) (`docs/tv1/figures/fig_ch1_roofline_compare.png`).
  - Phải: Bảng đối chiếu hệ số tăng tốc GPU Speedup phân hóa theo từng loại toán tử.
- **Nguồn trích dẫn:** *A. Howard et al., 'Searching for MobileNetV3,' IEEE ICCV, 2019.*
- **Nội dung trên slide:**

| Loại toán tử leaf | Đại diện tầng điển hình | Cường độ tính toán $I$ | Vị trí trên Roofline T4 | Hệ số tăng tốc GPU ($S_{\text{layer}}$) | Phân tích cơ chế giới hạn phần cứng |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Convolution 2D** | `Conv2d` (ResNet-50) | $35 - 120 \text{ FLOPs/B}$ | Vùng bão hòa tính toán | **$6.5\times - 18.2\times$** | Kích thước ma trận lớn, tận dụng trọn vẹn mảng tính toán song song |
| **Linear (GEMM)** | `Linear` (BERT-base) | $40 - 95 \text{ FLOPs/B}$ | Vùng bão hòa tính toán | **$7.1\times - 14.5\times$** | Cấu trúc ma trận dày đặc, kích hoạt năng lực cuBLAS trên GPU |
| **Depthwise Conv** | `Conv2d` (MobileNetV3) | $2.5 - 6.0 \text{ FLOPs/B}$ | **Vùng nghẽn băng thông** | **$1.8\times - 3.2\times$** | Tích chập theo từng kênh độc lập, tỷ lệ tái sử dụng dữ liệu thấp |
| **Normalization** | `BatchNorm2d` / `LayerNorm`| $< 1.5 \text{ FLOPs/B}$ | **Vùng nghẽn băng thông** | **$1.1\times - 2.1\times$** | Duyệt qua dữ liệu một lần để tính mean/var, nghẽn bus bộ nhớ |

- **Kết luận chuyên sâu về kiến trúc phần cứng:**
  - **Sự phân hóa tăng tốc:** GPU không tăng tốc đồng đều mọi tầng. Tầng tính toán nặng (`Conv2d`, `Linear`) tăng tốc mạnh mẽ (lên tới $18\times$). Ngược lại, tầng nhẹ (`BatchNorm`, `Depthwise Conv`) tăng tốc rất thấp ($< 2.5\times$) vì nằm sâu trong vùng nghẽn băng thông DRAM.
  - Điều này giải thích tại sao MobileNetV3 (vốn chứa nhiều Depthwise Conv và BatchNorm) lại đạt hiệu suất thấp hơn kỳ vọng khi chạy trên GPU ở kích thước batch nhỏ.
- **Lời thoại thuyết trình (Speaker Script):**
  > "Slide số 10 thể hiện giá trị học thuật sâu sắc nhất của Mức Layer: Phân tích sự phân hóa hệ số tăng tốc GPU theo Mô hình Roofline.  
  > Trên biểu đồ bên trái, chúng em định vị từng tầng mạng lên không gian Roofline của Tesla T4. Các tầng Conv2d tiêu chuẩn và tầng Linear của Transformer nằm hoàn toàn ở nửa bên phải - vùng Compute-Bound - do đó đạt mức tăng tốc vượt trội từ 7 đến 18 lần khi chuyển sang GPU.  
  > Tuy nhiên, hãy nhìn vào các tầng Depthwise Convolution của MobileNetV3 và các tầng BatchNorm: Chúng rơi hoàn toàn vào vùng Memory-Bound bên trái với cường độ tính toán cực thấp dưới 6 FLOPs trên mỗi Byte. Các tầng này bị giam cầm bởi băng thông bộ nhớ DRAM, dẫn đến hệ số tăng tốc GPU rất khiêm tốn, chỉ từ 1.1 đến 2 lần.  
  > Phát hiện này dẫn dắt chúng ta đến một câu hỏi lớn hơn ở Mức Model: Khi tích hợp tất cả các tầng này lại thành một mạng hoàn chỉnh, số lượng tham số có thực sự quyết định độ trễ hay không?  
  > Sau đây, em xin chuyển giao phần trình bày cho bạn TV2."
- **Lời chuyển giao:** TV1 $\rightarrow$ TV2: *"Xin mời bạn TV2 trình bày kết quả Mức Model!"*

---

# PHẦN 4: MỨC MODEL VÀ ĐỐI ĐẦU THỰC NGHIỆM CPU VS. GPU (SLIDE 11-16)
### Người báo cáo: Thành viên 2 (TV2) | Thời lượng: 6.5 phút (390 giây)

---

### SLIDE 11. THIẾT KẾ MODEL ZOO 8 ADAPTERS VÀ QUY TRÌNH ĐO SẠCH 2 PHA
- **Thời lượng:** 60 giây.
- **Bố cục hiển thị:**
  - Trái: Bảng ma trận 8 mô hình đại diện thuộc 3 lĩnh vực (CV, NLP, Tabular) với thông số cấu trúc.
  - Phải: Sơ đồ khối giải thuật Đo sạch 2 pha (Clean Two-Phase Measurement Harness).
- **Nội dung trên slide:**
  - **1. Thiết kế Model Zoo bao phủ 3 miền ứng dụng cốt lõi:**
    - **Thị giác máy tính (CV):** ResNet-50 (CNN cổ điển), MobileNetV3-Large (CNN di động), ViT-B/16 (Vision Transformer), YOLOv8n (Phát hiện vật thể).
    - **Xử lý ngôn ngữ tự nhiên (NLP):** BERT-base (Transformer 12 tầng), DistilBERT (Transformer chưng cất 6 tầng).
    - **Dữ liệu bảng (Tabular):** Multi-Layer Perceptron (MLP sâu), XGBoost (Rừng cây quyết định Gradient Boosting).
  - **2. Quy trình Thực nghiệm Đo sạch 2 pha (Two-Phase Measurement):**
    - **Pha 1 - Đo trắc tài nguyên nền tảng:** Thu thập Process RSS và Peak VRAM thông qua luồng chạy nền chu kỳ 100ms. Tuyệt đối không gọi NVML hay đo RAM trong vòng lặp tính toán để loại bỏ hoàn toàn nhiễu đo lường.
    - **Pha 2 - Đo trắc thời gian chuẩn xác:** Đồng bộ hóa thiết bị, chạy vòng lặp suy luận độc lập, thu thập phân phối trễ qua bộ bấm giờ độ chính xác nano-giây.
- **Mã nguồn liên quan:** [`src/zoo/base.py`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/src/zoo/base.py), [`src/model/harness.py`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/src/model/harness.py).
- **Lời thoại thuyết trình (Speaker Script):**
  > "Xin cảm ơn bạn TV1. Kính thưa quý thầy cô, bước sang Mức Model, nhiệm vụ của em là đánh giá toàn diện hành vi của toàn bộ mô hình khi hoạt động trong môi trường triển khai thực tế.  
  > Để bảo đảm tính đại diện, nhóm chúng em thiết kế Model Zoo gồm 8 mô hình trải rộng trên cả 3 miền ứng dụng: từ thị giác máy tính với ResNet, MobileNet, ViT, YOLO; xử lý ngôn ngữ tự nhiên với BERT và DistilBERT; cho đến dữ liệu bảng truyền thống với MLP và XGBoost.  
  > Về mặt phương pháp luận, chúng em xây dựng quy trình 'Đo sạch 2 pha': Tách biệt hoàn toàn việc đo đạc bộ nhớ sang một pha độc lập, không cho phép các hàm truy vấn bộ nhớ hệ thống chạy xen vào vòng lặp đo độ trễ. Điều này bảo đảm kết quả đo đạt độ tinh khiết cao nhất, phản ánh trung thực năng lực phần cứng."

---

### SLIDE 12. KẾ THỪA CHUẨN MLPERF INFERENCE: ĐO PHÂN PHỐI TRỄ P50-P99
- **Thời lượng:** 65 giây.
- **Bố cục hiển thị:**
  - Trái: Đồ thị hàm mật độ phân phối trễ (Latency Distribution Curve), đánh dấu rõ các điểm P50, P90, P99 và vùng trễ đuôi (Tail Latency).
  - Phải: Bảng định lượng điều kiện thử nghiệm và các chỉ số SLA.
- **Nguồn trích dẫn:** *V. J. Reddi et al., 'MLPerf Inference Benchmark,' ACM/IEEE ISCA, 2020.*
- **Nội dung trên slide:**
  - **1. Kế thừa triết lý Chuẩn MLPerf Inference (Reddi et al., ISCA 2020):**
    - Trong môi trường sản xuất thực tế, giá trị trung bình cộng ($\text{Mean}$) là một chỉ số sai lệch vì nó che giấu hoàn toàn các hiện tượng suy thoái hiệu năng đột ngột do ngắt hệ điều hành, dọn rác bộ nhớ (Garbage Collection), hay xung đột luồng.
    - Chuẩn công nghiệp bắt buộc phải báo cáo **trễ trung vị P50** cùng các **phân vị trễ đuôi P90 và P99** để bảo đảm cam kết chất lượng dịch vụ (Service Level Agreement - SLA).
  - **2. Bộ chỉ số đo lường toàn diện của VisionProf (12 chỉ số):**
    - Nhóm thời gian: P50 (Median), P90, P95, P99 (Tail Latency), Tail Ratio ($P99/P50$), Thông lượng (FPS), Độ trễ khởi động lạnh (Cold-start Latency).
    - Nhóm bộ nhớ: Đỉnh VRAM bộ cấp phát PyTorch (`max_memory_allocated`), Đỉnh VRAM hệ điều hành (`max_memory_reserved`), Bộ nhớ thực tế Host RSS.
  - **3. Điều kiện thực nghiệm kiểm soát chặt chẽ:**
    - Khởi động (Warm-up) $N_{\text{warmup}} = 15$ vòng để nạp trọng số vào cache và khởi tạo CUDA Context.
    - Đo lường $N_{\text{measure}} = 50$ vòng lặp độc lập liên tiếp.
- **Mã nguồn liên quan:** [`src/model/metrics.py`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/src/model/metrics.py), [`src/model/env.py`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/src/model/env.py).
- **Lời thoại thuyết trình (Speaker Script):**
  > "Tại sao VisionProf kiên quyết không sử dụng đại lượng trung bình cộng đơn giản?  
  > Kính thưa Hội đồng, theo khuyến nghị của chuẩn công nghiệp quốc tế MLPerf Inference được công bố tại ISCA 2020, độ trễ của các hệ thống AI luôn xuất hiện hiện tượng 'trễ đuôi' (Tail Latency). Một mô hình có độ trễ trung bình 10 mili-giây nhưng phân vị 99 vọt lên 100 mili-giây sẽ lập tức vi phạm cam kết SLA của hệ thống phân tán.  
  > Vì vậy, VisionProf thu thập đầy đủ bộ 12 chỉ số khoa học, đặc biệt giám sát chỉ số Tail Ratio tức tỷ số giữa P99 và P50. Chúng em thiết lập quy chuẩn đo lường nghiêm ngặt: 15 chu kỳ warm-up để khởi tạo toàn bộ bộ nhớ đệm, theo sau bởi 50 chu kỳ đo độc lập để xây dựng hàm phân phối xác suất trễ thực nghiệm."

---

### SLIDE 13. KẾT QUẢ CH3: NGHỊCH LÝ SỐ THAM SỐ TẠI BATCH 1 TRÊN COLAB CPU VS. TESLA T4
- **Thời lượng:** 75 giây.
- **Bố cục hiển thị:**
  - Trái: Bảng kết quả thực nghiệm đo thật 100% trên Google Colab CPU và Tesla T4 GPU (Batch Size = 1, FP32).
  - Phải: Đồ thị phân tán (Scatter Plot) tương quan giữa Số lượng tham số ($M$) và Độ trễ trung vị P50 (ms) (`docs/tv2/figures/fig_ch3_speedup.png`).
- **Nội dung trên slide:**

| Mô hình AI | Miền ứng dụng | Số tham số ($M$) | Colab T4 P50 (ms) | Colab T4 P99 (ms) | Peak VRAM (MB) | Colab CPU P50 (ms) | Hạng T4 | Hạng CPU |
| :--- | :---: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **MLP** | Tabular | 0.05 | **0.28** | 0.64 | 9 | **0.10** | 1 | 1 |
| **XGBoost** | Tabular | 0.02 | **1.56** | 2.10 | — | **0.45** | 2 | 2 |
| **DistilBERT** | NLP | 66.4 | **5.34** | 8.91 | 269 | **26.50** | 3 | 5 |
| **ResNet-50** | CV | 25.6 | **5.96** | 10.46 | 117 | **22.50** | 4 | 4 |
| **MobileNetV3-L** | CV | 5.5 | **5.99** | 9.21 | 38 | **17.50** | 5 | 3 |
| **YOLOv8n** | CV | 3.2 | **7.52** | 8.75 | 49 | **163.58** | 6 | 6 |
| **BERT-base** | NLP | 109.5 | **8.58** | 14.17 | 433 | **247.92** | 7 | 8 |
| **ViT-B/16** | CV | 86.6 | **12.99** | 13.66 | 348 | **180.00** | 8 | 7 |

- **3 Phát hiện khoa học cốt lõi giải đáp câu hỏi CH3:**
  1. **Ở Batch 1 trên GPU, Số tham số và FLOPs hoàn toàn không quyết định độ trễ:**
     - ResNet-50 (25.6M tham số, 8.2 GFLOPs) và MobileNetV3 (5.5M tham số, 0.43 GFLOPs - ít hơn $19\times$ FLOPs) **có độ trễ hoàn toàn tương đương nhau trên Tesla T4: ~6.0 ms** (5.96 ms vs. 5.99 ms)!
     - *Nguyên nhân kỹ thuật:* Tại kích thước batch 1, GPU rơi vào trạng thái bị giới hạn bởi độ trễ khởi động kernel (Kernel Launch Overhead) tuần tự từ CPU qua driver NVIDIA. Thời gian thực thi phép tính quá nhỏ so với thời gian điều phối phần cứng.
  2. **Mô hình quy mô siêu nhỏ chạy trên GPU chậm hơn trên CPU:**
     - MLP chạy trên GPU mất 0.28 ms, chậm hơn gần $3\times$ so với chạy trên Colab CPU (0.10 ms).
     - XGBoost dự đoán trên GPU mất 1.56 ms, chậm hơn $3.4\times$ so với Colab CPU (0.45 ms).
     - *Nguyên nhân:* Chi phí sao chép tensor qua bus PCIe và đồng bộ hóa ngữ cảnh CUDA hoàn toàn triệt tiêu lợi thế tính toán song song của GPU.
  3. **Khẳng định trên cùng hạ tầng phần cứng chuẩn:**
     - Toàn bộ các kết quả trên đều được đo trực tiếp trên môi trường ảo hóa chuẩn của Google Colab (CPU Intel Xeon vs GPU Tesla T4), loại trừ triệt để mọi biến số sai lệch về phần cứng cục bộ.
- **Mã nguồn liên quan:** [`results/colab_t4/`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/results/colab_t4/), [`results/colab_cpu/`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/results/colab_cpu/).
- **Lời thoại thuyết trình (Speaker Script):**
  > "Kính thưa quý thầy cô, đây là phát hiện thực nghiệm then chốt trả lời câu hỏi nghiên cứu CH3: **Số lượng tham số có thực sự quyết định tốc độ của mô hình hay không?**  
  > Trong cộng đồng phát triển phần mềm, một định kiến rất lớn cho rằng: Mô hình càng ít tham số hoặc ít FLOPs thì chắc chắn chạy càng nhanh. Số liệu thực nghiệm đo thật 100% trên nền tảng Google Colab của chúng em đã chứng minh định kiến này hoàn toàn sai lầm ở kích thước batch 1!  
  > Hãy nhìn vào cặp đấu giữa ResNet-50 và MobileNetV3-Large trên GPU Tesla T4: MobileNetV3 chỉ có 5.5 triệu tham số và 0.43 GFLOPs - tức là nhẹ hơn ResNet-50 tới 19 lần về khối lượng phép tính. Nhưng trên GPU ở batch 1, cả hai mô hình đều cùng mất đúng 6.0 mili-giây!  
  > Tại sao lại như vậy? Bởi vì ở batch 1, khối lượng tính toán quá nhỏ để lấp đầy 2,560 nhân CUDA của T4. Thời gian xử lý bị thống trị hoàn toàn bởi chi phí CPU gửi hàng trăm kernel tuần tự xuống GPU qua giao tiếp PCIe.  
  > Đặc biệt, với các mô hình siêu nhỏ như MLP hay XGBoost, chạy trên GPU T4 còn chậm hơn chạy trực tiếp trên CPU Intel Xeon từ 2.8 đến 3.4 lần vì chi phí khởi tạo CUDA Context và truyền dữ liệu qua bus!"

---

### SLIDE 14. KẾT QUẢ CH3: HIỆN TƯỢNG CHUYỂN DỊCH THỨ HẠNG HIỆU NĂNG THEO PHẦN CỨNG
- **Thời lượng:** 60 giây.
- **Bố cục hiển thị:**
  - Hình minh họa trung tâm: Biểu đồ Bump Chart trực quan hóa sự dịch chuyển thứ tự tốc độ từ Colab CPU sang Colab T4 GPU (`docs/tv2/figures/fig_ch3_rank_shift.png`).
- **Nội dung trên slide:**
  - **Sự chuyển dịch thứ hạng thực tế giữa Colab CPU và Tesla T4 GPU:**
    - **DistilBERT:** Từ Hạng 5 trên Colab CPU (26.50 ms) $\rightarrow$ nhảy vọt lên **Hạng 3 trên T4 GPU (5.34 ms)** nhờ các khối GEMM lớn song song hóa tối ưu trên CUDA cores.
    - **MobileNetV3-L:** Đứng Hạng 3 trên CPU (17.50 ms, nhanh hơn ResNet-50) $\rightarrow$ tụt xuống **Hạng 5 trên GPU (5.99 ms)** do các tầng Depthwise Conv bị nghẽn băng thông và kernel launch overhead.
    - **BERT-base:** Đứng Hạng 8 trên CPU (247.92 ms) $\rightarrow$ vươn lên **Hạng 7 trên T4 GPU (8.58 ms)** với hệ số tăng tốc GPU đạt $28.9\times$.
    - **MLP và XGBoost:** Giữ vững hai vị trí dẫn đầu (Hạng 1 và 2) trên cả hai nền tảng do kích thước mô hình siêu nhỏ.
  - **Thông điệp khoa học cốt lõi (Hardware-Aware Architecture Selection):**
    - **Không tồn tại một kiến trúc mô hình 'nhanh nhất trên mọi môi trường phần cứng'.**
    - Kiến trúc được thiết kế tối ưu cho CPU (như MobileNet với depthwise convolutions) không đồng nghĩa sẽ đạt hiệu quả tương ứng trên GPU gia tốc. Quyết định lựa chọn mô hình bắt buộc phải dựa trên số liệu profiling thực tế tại chính môi trường phần cứng mục tiêu.
- **Mã nguồn liên quan:** [`src/dashboard/tabs/tab_cpu_vs_gpu.py`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/src/dashboard/tabs/tab_cpu_vs_gpu.py).
- **Lời thoại thuyết trình (Speaker Script):**
  > "Biểu đồ Bump Chart trên slide trực quan hóa một hiện tượng kiến trúc vô cùng thú vị: Sự chuyển dịch thứ hạng hiệu năng khi thay đổi nền tảng tính toán.  
  > Trên CPU Intel Xeon, MobileNetV3 chạy nhanh hơn ResNet-50 (17.5 ms so với 22.5 ms) vì trên CPU, số phép tính FLOPs thấp đồng nghĩa với số chu kỳ lệnh ít hơn.  
  > Nhưng khi chuyển sang GPU Tesla T4, vị trí này bị san phẳng: MobileNetV3 tụt lại phía sau ResNet-50. Ngược lại, mô hình Transformer DistilBERT từ vị trí thứ 5 trên CPU đã bứt phá lên vị trí thứ 3 trên GPU, vượt qua cả các mạng tích chập nhờ các khối nhân ma trận lớn kích hoạt xuất sắc năng lực xử lý đa luồng của phần cứng.  
  > Bài học rút ra ở đây là: Không bao giờ được suy diễn hiệu năng của mô hình từ môi trường phát triển sang môi trường máy chủ đám mây. Mọi quyết định lựa chọn kiến trúc đều phải được kiểm chứng bằng profiling trên đúng phần cứng triển khai."

---

### SLIDE 15. KẾT QUẢ CH4: MỞ RỘNG BATCH SIZE VÀ QUY LUẬT BẬC HAI CỦA ĐỘ DÀI CÂU NLP
- **Thời lượng:** 65 giây.
- **Bố cục hiển thị:**
  - Trái: Biểu đồ đường biểu diễn thông lượng Throughput (FPS) khi mở rộng Batch Size từ 1 lên 32 trên GPU Tesla T4 (`docs/tv2/figures/fig_ch4_batch_scaling.png`).
  - Phải: Bảng tốc độ xử lý câu của BERT-base theo chiều dài chuỗi token ($N$).
- **Nội dung trên slide:**
  - **1. Quy luật Mở rộng Thông lượng khi Tăng Kích thước Lô (Throughput Scaling trên Tesla T4):**

| Kiến trúc mô hình | Batch 1 (FPS) | Batch 8 (FPS) | Batch 32 (FPS) | Hệ số tăng 32 / 1 | Trạng thái bão hòa phần cứng |
| :--- | :-: | :-: | :-: | :-: | :--- |
| **MobileNetV3-L** | 170 | 1,094 | **1,499** | **$8.8\times$** | Hưởng lợi lớn nhất, liên tục lấp đầy GPU |
| **ResNet-50** | 153 | 287 | **337** | **$2.2\times$** | Tăng ổn định, đạt ngưỡng trần tính toán |
| **YOLOv8n** | 118 | 198 | **235** | **$2.0\times$** | Tăng gấp đôi thông lượng xử lý |
| **ViT-B/16** | 75 | 96 | **94** | **$1.3\times$** | **Bão hòa hoàn toàn ngay từ Batch 8** |
| **DistilBERT** | 198 | 287 | **300** | **$1.5\times$** | Tiệm cận trần thông lượng của mảng Tensor |
| **BERT-base** | 100 | 134 | **140** | **$1.4\times$** | Đạt cực đại năng lực tính toán |

  - **2. Điểm nghẽn Độ phức tạp Bậc hai $\mathcal{O}(N^2)$ của Self-Attention (BERT-base tại Batch 32):**
    - Chiều dài chuỗi $\text{Seq} = 64$: Độ trễ **127 ms**.
    - Chiều dài chuỗi $\text{Seq} = 128$: Độ trễ **228 ms** (Tăng **$1.8\times$**).
    - Chiều dài chuỗi $\text{Seq} = 256$: Độ trễ **507 ms** (Tăng **$2.2\times$**).
    - *Bản chất toán học:* Phép nhân ma trận tương quan điểm $Q K^T$ có kích thước $(N \times N)$, khiến chi phí tính toán và bộ nhớ tăng theo hàm bậc hai của chiều dài văn bản đầu vào.
- **Mã nguồn liên quan:** [`results/colab_t4/`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/results/colab_t4/), bộ dữ liệu thực nghiệm TN4.
- **Lời thoại thuyết trình (Speaker Script):**
  > "Bước sang câu hỏi CH4, chúng em nghiên cứu khả năng mở rộng thông lượng khi tăng kích thước Batch Size từ 1 lên 32 và tác động của độ dài văn bản trong xử lý ngôn ngữ tự nhiên.  
  > Kết quả ở bảng bên trái làm sáng tỏ một quy luật kiến trúc: Các mô hình mạng nhẹ hưởng lợi nhiều nhất từ việc tăng batch size. Điển hình là MobileNetV3 tăng vọt thông lượng tới 8.8 lần - từ 170 lên gần 1,500 ảnh mỗi giây! Nguyên nhân là việc gom lô lớn đã cung cấp đủ khối lượng dữ liệu để lấp đầy các vi xử lý đa luồng SM của Tesla T4, triệt tiêu hoàn toàn chi phí khởi tạo kernel. Ngược lại, mô hình nặng như ViT-B/16 đã bão hòa năng lực tính toán của T4 ngay từ batch 8, khiến thông lượng đi ngang.  
  > Ở bảng bên phải, chúng em lượng hóa điểm nghẽn thuật toán bậc hai O(N bình phương) của cơ chế Self-Attention: Tại batch 32, khi tăng gấp đôi độ dài chuỗi từ 128 lên 256 token, độ trễ của BERT tăng vọt phi tuyến từ 228 mili-giây lên 507 mili-giây (tăng 2.2 lần). Trong thực tế triển khai, kỹ sư bắt buộc phải áp dụng kỹ thuật dynamic padding để bảo vệ hệ thống."

---

### SLIDE 16. KẾT QUẢ CH4: KHẢO SÁT FP16 TENSOR CORES TRÊN TESLA T4 (BATCH 1)
- **Thời lượng:** 65 giây.
- **Bố cục hiển thị:**
  - Trái: Bảng kết quả thực nghiệm đối đầu giữa FP32 và FP16 trên GPU Tesla T4.
  - Phải: Biểu đồ cột thể hiện hệ số tăng tốc và tỷ lệ cắt giảm VRAM (`docs/tv2/figures/fig_ch4_fp16.png`).
- **Nội dung trên slide:**

| Kiến trúc mô hình AI | Độ trễ FP32 (ms) | Độ trễ FP16 (ms) | Hệ số tăng tốc FP16 | Tỷ lệ giảm VRAM | Độ tương đồng Cosine Similarity |
| :--- | :-: | :-: | :-: | :-: | :---: |
| **ViT-B/16** | 12.99 | **4.62** | **$2.81\times$** *(Vượt trội)*| **Giảm 45.0%** | Kiểm chuẩn đầu ra hoàn hảo |
| **DistilBERT** | 5.34 | **3.44** | **$1.55\times$** | **Giảm 45.0%** | **0.999999** (Độ chính xác số học cao) |
| **BERT-base** | 8.58 | **6.94** | **$1.23\times$** | **Giảm 46.0%** | **0.999999** (Độ chính xác số học cao) |
| **ResNet-50** | 5.96 | 6.88 | **$0.87\times$** *(Chậm hơn)*| **Giảm 43.0%** | **1.000000** (Bảo toàn tuyệt đối) |
| **MobileNetV3-L** | 5.99 | 7.02 | **$0.85\times$** *(Chậm hơn)*| **Giảm 38.0%** | Bảo toàn sai số cục bộ |
| **YOLOv8n** | 7.52 | 9.76 | **$0.77\times$** *(Chậm hơn)*| **Giảm 49.0%** | **1.000000** (Bảo toàn tuyệt đối) |

- **Quy luật vật lý và Khuyến nghị kỹ thuật:**
  1. **FP16 LUÔN LUÔN cắt giảm từ 38% đến 49% dung lượng bộ nhớ VRAM:** Hoàn toàn khớp với lý thuyết biểu diễn dấu phẩy động 16-bit (tiết kiệm một nửa không gian lưu trữ trọng số và tensor kích hoạt).
  2. **FP16 CHỈ TĂNG TỐC TRANSFORMER ở kích thước Batch 1:** ViT-B/16 đạt mức tăng tốc ấn tượng **$2.81\times$** nhờ các ma trận lớn kích hoạt 320 nhân Turing Tensor Cores chuyên dụng (đạt trần lý thuyết 65 TFLOPS FP16).
  3. **Cảnh báo suy thoái hiệu năng đối với CNN ở Batch 1:** Các mô hình tích chập chạy FP16 ở batch 1 bị **chậm đi từ 13% đến 23%** do kích thước tensor quá nhỏ không tận dụng được Tensor Cores, trong khi phải chịu thêm chi phí ép kiểu dữ liệu (Type Casting) của driver. Muốn tăng tốc CNN bằng FP16, bắt buộc phải tăng kích thước Batch Size $\ge 8$.
- **Lời thoại thuyết trình (Speaker Script):**
  > "Nội dung cuối cùng của Mức Model là khảo sát định dạng số thực nửa chính xác FP16 trên nhân chuyên dụng Turing Tensor Cores của Tesla T4.  
  > Kết quả thực nghiệm mang lại một phát hiện hai mặt vô cùng giá trị:  
  > Về mặt bộ nhớ: Chuyển sang FP16 **luôn luôn cắt giảm từ 38% đến 49% dung lượng VRAM** trên tất cả các mô hình, đồng thời độ tương đồng Cosine đạt xấp xỉ 1.0 tuyệt đối, chứng minh việc giảm độ chính xác bit không làm suy giảm chất lượng đầu ra.  
  > Tuy nhiên, về mặt tốc độ: **FP16 chỉ tăng tốc cho các mô hình Transformer**, điển hình là ViT-B/16 tăng tốc tới 2.81 lần nhờ kích hoạt Tensor Cores! Ngược lại hoàn toàn, với các mô hình CNN ở batch 1, FP16 lại làm chậm đi từ 13% đến 23% do chi phí chuyển đổi kiểu dữ liệu giữa float32 và float16.  
  > Tuy nhiên, mô hình suy luận chỉ là một thành phần trong một hệ thống lớn. Sau đây, bạn TV3 sẽ đưa Hội đồng đến Mức Giai đoạn để khảo sát toàn bộ chu trình huấn luyện và ứng dụng thực tế."
- **Lời chuyển giao:** TV2 $\rightarrow$ TV3: *"Xin mời bạn TV3 tiếp tục phần báo cáo!"*

---

# PHẦN 5: MỨC GIAI ĐOẠN, CHẨN ĐOÁN TỰ ĐỘNG VÀ TỔNG KẾT (SLIDE 17-23)
### Người báo cáo: Thành viên 3 (TV3) | Thời lượng: 7.0 phút (420 giây)

---

### SLIDE 17. KẾT QUẢ CH5: PHÂN RÃ CHU TRÌNH HUẤN LUYỆN VÀ HIỆN TƯỢNG BÙNG NỔ BỘ NHỚ
- **Thời lượng:** 70 giây.
- **Bố cục hiển thị:**
  - Trái: Cấu trúc lớp `PhaseTimer` bóc tách pha Huấn luyện cùng biểu đồ tỷ số Backward/Forward (`docs/tv3/figures/fig_ch5_backward_forward_ratio.png`).
  - Phải: Biểu đồ cột biểu diễn mức bùng nổ bộ nhớ VRAM khi chuyển từ Inference sang Training (`docs/tv3/figures/fig_ch5_memory_train_vs_infer.png`).
- **Nguồn trích dẫn:** *H. Zhu et al., 'TBD: Benchmarking and Analyzing Deep Learning Training Workloads,' IEEE IISWC, 2018.*
- **Nội dung trên slide:**
  - **1. Bảng Số liệu Thực nghiệm Chu trình Huấn luyện trên Tesla T4 (Batch Size = 8, FP32):**

| Kiến trúc mô hình | Forward Pass (ms) | Backward Pass (ms) | Optimizer Step (ms) | Tổng chu kỳ bước (ms) | **Tỷ số Backward / Forward** | VRAM Suy luận (MB) | VRAM Huấn luyện (MB) | **Hệ số tăng VRAM** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-50** | 29.77 | 54.25 | 9.73 | 93.75 | **$1.82\times$** | 420.11 | 1,139.05 | **$2.71\times$** |
| **MobileNetV3-L** | 9.59 | 15.81 | 3.07 | 28.47 | **$1.65\times$** | 140.09 | 459.71 | **$3.28\times$** |
| **DistilBERT** | 22.42 | 49.59 | 22.81 | 94.82 | **$2.21\times$** | 212.40 | 688.18 | **$3.24\times$** |
| **MLP** | 0.58 | 0.81 | 0.47 | 1.86 | **$1.39\times$** | 17.84 | 18.75 | **$1.05\times$** |

- **2. Luận giải Khoa học giải đáp câu hỏi CH5 (Phần 1):**
  - **Kiểm chứng tỷ số lý thuyết $Backward / Forward \approx 2.0\times$:**
    - Theo giải tích ma trận, pha lan truyền ngược cần thực hiện 2 phép nhân ma trận độc lập: $\nabla_X = \nabla_Y W^T$ (tính đạo hàm theo đầu vào) và $\nabla_W = X^T \nabla_Y$ (tính đạo hàm theo trọng số). DistilBERT đạt tỷ số **$2.21\times$**, hoàn toàn khớp với lý thuyết.
    - Với CNN, tỷ số đạt $1.82\times$ và $1.65\times$ (thấp hơn $2\times$) do các toán tử tích chập ngược của thư viện cuDNN được tối ưu hóa bộ nhớ đệm vượt trội.
  - **Hiện tượng Bùng nổ Bộ nhớ ($2.7\times - 3.3\times$):**
    - Khi suy luận, PyTorch giải phóng ngay lập tức các tensor trung gian. Khi huấn luyện, Autograd bắt buộc phải **lưu giữ toàn bộ tensor kích hoạt (Activations)** trong VRAM để phục vụ quy tắc dây chuyền (Chain Rule), khiến bộ nhớ tăng vọt tới $3.3\times$ (MobileNetV3).
- **Mã nguồn liên quan:** [`src/phase/timer.py`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/src/phase/timer.py), kết quả tại `results/colab_t4/*/TN6/`.
- **Lời thoại thuyết trình (Speaker Script):**
  > "Xin cảm ơn bạn TV2. Kính thưa Hội đồng, bước sang Mức Giai đoạn, chúng em mở rộng biên giới profiling ra toàn bộ vòng đời ứng dụng.  
  > Câu hỏi CH5 đặt ra: **Trong chu trình huấn luyện, thời gian và bộ nhớ thực sự tiêu tốn ở đâu?**  
  > Để trả lời, VisionProf triển khai module `PhaseTimer` bóc tách 3 giai đoạn: Forward, Backward và Optimizer.  
  > Về thời gian: Kết quả thực nghiệm xác nhận định luật lý thuyết: Thời gian Backward gấp khoảng **1.8 đến 2.2 lần Forward**. Trong Transformer, tỷ số đạt 2.21 lần vì pha ngược đòi hỏi đúng 2 phép nhân ma trận để tính gradient cho cả đầu vào và trọng số.  
  > Về bộ nhớ: Chúng em ghi nhận hiện tượng 'Bùng nổ bộ nhớ' khi VRAM huấn luyện tăng vọt từ 2.7 đến 3.3 lần so với suy luận. Lý do là cơ chế Autograd bắt buộc phải găm giữ toàn bộ các bản đồ kích hoạt trung gian trong VRAM cho đến khi pha backward hoàn tất, hoàn toàn phù hợp với nghiên cứu TBD tại hội thảo IISWC 2018."

---

### SLIDE 18. KẾT QUẢ CH5: ĐIỂM NGHẼN NẠP DỮ LIỆU VÀ HIỆN TƯỢNG GPU STARVATION
- **Thời lượng:** 70 giây.
- **Bố cục hiển thị:**
  - Trái: Biểu đồ cột chồng thể hiện cơ cấu thời gian chu kỳ bước lặp (Data Load, Forward, Backward, Optimizer) theo số lượng `num_workers`.
  - Phải: Bảng đối chứng thông lượng và tỷ lệ nghẽn I/O trên Google Colab T4 (2 vCPUs).
- **Nguồn trích dẫn:** *J. Mohan et al., 'Analyzing and Mitigating Data Stalls in DNN Training,' Proceedings of the VLDB Endowment (PVLDB), 2021.*
- **Nội dung trên slide:**
  - **Bảng Số liệu Khảo sát DataLoader với Dữ liệu Ảnh JPEG Thực tế (Batch Size = 32, Colab T4):**

| Kiến trúc mô hình | Cấu hình `num_workers` | Thời gian chờ nạp dữ liệu (ms) | Tổng chu kỳ bước (ms) | **Tỷ lệ chờ dữ liệu (%)** | Thông lượng (Ảnh/giây) | Trạng thái chẩn đoán hệ thống |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **MobileNetV3-L** | `num_workers = 0` | 151.4 | 241.8 | **$62.6\%$** | 132 | **Nghẽn nghiêm trọng (GPU Starvation)** |
| | `num_workers = 2` | 3.9 | 116.9 | **$3.3\%$** | **274** ($2.08\times$) | **Trạng thái tối ưu hóa phần cứng** |
| | `num_workers = 4` | 3.9 | 119.8 | $3.2\%$ | 267 | Suy giảm nhẹ do tranh chấp tài nguyên |
| **ResNet-50** | `num_workers = 0` | 139.2 | 462.5 | **$30.1\%$** | 69 | Kích hoạt cảnh báo nghẽn I/O |
| | `num_workers = 2` | 2.0 | 338.2 | **$0.6\%$** | **95** ($1.38\times$) | **Trạng thái tối ưu hóa phần cứng** |
| | `num_workers = 4` | 2.0 | 356.3 | $0.6\%$ | 90 | Tranh chấp luồng trên 2 vCPU |

- **3 Kết luận khoa học giải đáp câu hỏi CH5 (Phần 2):**
  1. **Hiện tượng GPU Starvation trầm trọng khi `num_workers = 0`:** Với mạng nhẹ như MobileNetV3, GPU mất tới **$62.6\%$ chu kỳ chỉ để ngồi chờ CPU** giải nén ảnh JPEG và thực hiện Data Augmentation.
  2. **Giải phóng hiệu năng với `num_workers = 2`:** Tận dụng 2 tiến trình worker chạy nền đưa thời gian chờ nạp dữ liệu xuống dưới $4\%$, tăng gấp đôi thông lượng huấn luyện (từ 132 lên 274 ảnh/s).
  3. **Nghịch lý tranh chấp CPU khi đặt `num_workers = 4`:** Do phiên làm việc Google Colab chỉ cung cấp 2 vCPU thực tế, việc cấu hình 4 workers tạo ra xung đột luồng và chi phí chuyển đổi ngữ cảnh (Context Switching), khiến chu kỳ bước bị kéo dài thêm 3 ms.
- **Mã nguồn liên quan:** [`src/phase/sentinel.py`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/src/phase/sentinel.py), [`scripts/run_tn6_dataloader.py`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/scripts/run_tn6_dataloader.py).
- **Lời thoại thuyết trình (Speaker Script):**
  > "Một hệ thống huấn luyện không bao giờ chỉ có GPU. Trong slide 18, chúng em kiểm chứng hiện tượng 'Đói tài nguyên GPU' (GPU Starvation) được công bố trên kỷ yếu VLDB 2021.  
  > Khi lập trình viên để cấu hình mặc định `num_workers = 0`, toàn bộ thao tác đọc đĩa và giải nén ảnh JPEG diễn ra tuần tự trên luồng chính. Với một mô hình suy luận nhanh như MobileNetV3, GPU Tesla T4 mất tới **62.6% thời gian chu kỳ chỉ để ngồi chờ dữ liệu từ CPU**! Hiệu năng tính toán của GPU bị lãng phí hoàn toàn.  
  > Chỉ cần cấu hình `num_workers = 2`, cơ chế tiền nạp bất đồng bộ đã kéo thời gian chờ I/O xuống chỉ còn 3.3%, giải phóng hoàn toàn sức mạnh phần cứng và tăng thông lượng huấn luyện gấp hơn 2 lần!  
  > Tuy nhiên, VisionProf cũng phát hiện: Tăng lên 4 workers lại làm hiệu năng suy giảm nhẹ. Lý do là môi trường Google Colab chỉ cấp phát 2 vCPUs; 4 workers sẽ tạo ra hiện tượng nghẽn tranh chấp nhân CPU với luồng chính của PyTorch."

---

### SLIDE 19. KẾT QUẢ CH6: ĐIỂM NGHẼN NGOÀI MÔ HÌNH TRONG PIPELINE VÀ ĐỊNH LUẬT AMDAHL
- **Thời lượng:** 70 giây.
- **Bố cục hiển thị:**
  - Trái: Biểu đồ cấu cấu Pipeline trên Colab T4 (`docs/tv3/figures/fig_ch6_pipeline_breakdown.png`).
  - Phải: Biểu đồ đối đầu cơ cấu Pipeline giữa Colab CPU và Tesla T4 GPU (`docs/tv3/figures/fig_ch6_cpu_vs_gpu_pipeline.png`).
- **Nguồn trích dẫn:** *M. Richins et al., 'Missing the Target: The Hidden Cost of AI Pipelines (AI Tax),' ACM/IEEE HPCA, 2020.*
- **Nội dung trên slide:**
  - **1. Bảng Số liệu Thực nghiệm Quy trình Đầu-Cuối TN7 (Batch Size = 1, FP32):**

| Môi trường phần cứng | Mô hình kiểm thử | Tiền xử lý (ms) | Suy luận Forward (ms) | Hậu xử lý (ms) | Tổng trễ Pipeline (ms) | **Tỷ lệ ngoài Mô hình (%)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Colab T4 (GPU)** | **YOLOv8n (CV)** | 3.23 (Letterbox) | 6.29 | 1.15 (NMS) | 10.68 | **$41.0\%$ (Xác nhận CH6 $>30\%$)** |
| | **BERT-base (NLP)** | 2.88 (Tokenize) | 18.38 | 0.07 (Pooling) | 21.33 | **$13.8\%$** |
| | **XGBoost (Tabular)**| 0.03 (CSV Parse) | 1.55 (`predict`) | — | 1.58 | **$2.1\%$** |
| **Colab CPU** | **YOLOv8n (CV)** | 8.80 (Letterbox) | 163.58 | 1.58 (NMS) | 173.97 | **$6.0\%$** |
| | **BERT-base (NLP)** | 2.76 (Tokenize) | 247.92 | 0.06 (Pooling) | 250.74 | **$1.1\%$** |
| | **XGBoost (Tabular)**| 0.02 (CSV Parse) | 0.45 (`predict`) | — | 0.48 | **$4.4\%$** |

- **2. Luận giải Khoa học theo Định luật Giới hạn Amdahl:**
  - Khi chạy trên Colab CPU, mô hình YOLOv8n mất 163.6 ms ($94.0\%$ tổng thời gian); các khâu tiền/hậu xử lý chỉ chiếm $6.0\%$.
  - Khi chuyển sang GPU Tesla T4, mô hình forward được tăng tốc tới **$26.0\times$** (từ 163.6 ms xuống 6.29 ms).
  - Tuy nhiên, các công đoạn Letterbox (3.23 ms) và Non-Maximum Suppression - NMS (1.15 ms) vẫn chạy tuần tự trên CPU.
  - Hậu quả: Thời gian ngoài mô hình tăng vọt từ $6.0\%$ lên **$41.0\%$ toàn bộ ứng dụng**. Theo định luật Amdahl, dù GPU có được tăng tốc đến vô hạn ($s \rightarrow \infty$), tốc độ toàn pipeline cũng bị chặn trần lý thuyết ở mức $\frac{1}{0.41} = 2.44\times$.
  - Kết quả này hoàn toàn xác nhận lý thuyết **AI Tax** được công bố tại hội thảo HPCA 2020: Dưới 60% thời gian của ứng dụng AI thực tế nằm trong mạng nơ-ron; phần lớn thời gian bị tiêu hao bởi mã nguồn truyền thống ngoài mô hình.
- **Mã nguồn liên quan:** [`src/phase/pipeline_timer.py`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/src/phase/pipeline_timer.py), [`scripts/run_tn7_pipeline.py`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/scripts/run_tn7_pipeline.py).
- **Lời thoại thuyết trình (Speaker Script):**
  > "Slide 19 trả lời câu hỏi nghiên cứu CH6: **Trong một ứng dụng AI hoàn chỉnh, các công đoạn ngoài mô hình chiếm bao nhiêu phần trăm thời gian?**  
  > Với bài toán phát hiện vật thể YOLOv8n trên GPU Tesla T4, tổng thời gian xử lý một khung hình là 10.68 mili-giây. Nhưng trong đó: Tiền xử lý Letterbox chiếm 3.23 ms và thuật toán lọc hộp trùng NMS chiếm 1.15 ms. Tổng cộng, **chi phí ngoài mô hình chiếm tới 41% toàn bộ đường ống**! Giả thuyết CH6 đặt ra ban đầu là lớn hơn 30% đã được xác nhận một cách thuyết phục.  
  > So sánh trực diện giữa CPU và GPU bộc lộ rõ Định luật Amdahl: Trên CPU, mô hình rất chậm nên các khâu phụ trợ chỉ chiếm 6%. Khi đưa lên GPU, mạng nơ-ron được tăng tốc tới 26 lần, khiến điểm nghẽn hệ thống lập tức bị đẩy ngược về khâu tiền xử lý trên CPU!  
  > Điều này hoàn toàn khớp với khái niệm 'Thuế AI' (AI Tax) tại HPCA 2020: Việc chỉ tập trung tối ưu hóa mạng nơ-ron sẽ nhanh chóng chạm trần hiệu năng. Kỹ sư hệ thống bắt buộc phải tối ưu hóa cả đường ống tiền và hậu xử lý."

---

### SLIDE 20. ĐỘNG CƠ CHẨN ĐOÁN TỰ ĐỘNG (19 QUY TẮC) VÀ SO SÁNH A/B PHI THAM SỐ
- **Thời lượng:** 70 giây.
- **Bố cục hiển thị:**
  - Trái: Danh mục phân loại 19 quy tắc chẩn đoán v2.0 với các trạng thái cảnh báo chuẩn hóa (`PASS`, `WARN`, `FAIL`, `SKIPPED`).
  - Phải: Khung toán học của kiểm định giả thuyết thống kê A/B Testing.
- **Nội dung trên slide:**
  - **1. Hệ thống 19 Quy tắc Chẩn đoán Điểm nghẽn Chuẩn hóa v2.0:**
    - **Nhóm RA (Tài nguyên Bộ nhớ - 4 quy tắc):** Giám sát tràn VRAM Allocator, rò rỉ bộ nhớ tiến trình RSS, và phân bổ bộ nhớ đệm.
    - **Nhóm RB (Năng lực Tính toán - 5 quy tắc):** Phát hiện mức độ khai thác GPU thấp, quy tắc RB-04 chuẩn hóa ngưỡng $\ge 1.0 \text{ ms/GFLOP}$ trên Tesla T4.
    - **Nhóm RC (Luồng Dữ liệu & Huấn luyện - 5 quy tắc):** Cảnh báo nghẽn I/O ($> 20\%$), tỷ số Backward/Forward bất thường ($> 3.0\times$), và hiện tượng GPU Starvation (RC-03).
    - **Nhóm RD (Kiến trúc & Toán tử - 5 quy tắc):** Cảnh báo tầng sub-microsecond ($< 1\,\mu\text{s}$), các toán tử chiếm trên 20% thời gian mạng, và khuyến nghị tăng kích thước batch.
  - **2. Động cơ So sánh A/B có Ý nghĩa Thống kê (A/B Testing Engine):**
    - Kiểm định phi tham số **Mann-Whitney $U$ test** (không giả định phân phối chuẩn Gauss, ngưỡng bác bỏ $p < 0.05$).
    - Đo lường độ lớn ảnh hưởng bằng **Cliff's delta ($\delta$)**:
      $$\delta = \frac{\#\{x_1 > x_2\} - \#\{x_1 < x_2\}}{n_1 n_2}$$
      Phân loại mức độ can thiệp: Không đáng kể ($|\delta| < 0.147$), Nhỏ ($0.147 \le |\delta| < 0.33$), Trung bình ($0.33 \le |\delta| < 0.474$), Lớn ($|\delta| \ge 0.474$).
    - Tốc độ tăng tốc Speedup được tính toán chặt chẽ trên tổng thời gian của từng vòng lặp thực nghiệm.
- **Mã nguồn liên quan:** [`src/analyzer/rules/`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/src/analyzer/rules/), [`src/analyzer/ab_testing.py`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/src/analyzer/ab_testing.py).
- **Lời thoại thuyết trình (Speaker Script):**
  > "Nhằm tự động hóa công tác tư vấn kỹ thuật, VisionProf tích hợp động cơ chẩn đoán gồm 19 quy tắc chuẩn hóa v2.0, được phân chia thành 4 nhóm độc lập: Bộ nhớ RA, Tính toán RB, Nạp dữ liệu RC và Kiến trúc RD.  
  > Mỗi quy tắc vận hành như một chuyên gia hệ thống: Ví dụ, khi thời gian nạp dữ liệu vượt quá 20% hoặc GPU bị đói dữ liệu, hệ thống tự động kích hoạt luật RC-01 và RC-03 để khuyến nghị người dùng tăng số lượng worker hoặc sử dụng pinned memory.  
  > Đặc biệt, khi kỹ sư thực hiện tối ưu hóa mô hình, VisionProf cung cấp động cơ A/B Testing có ý nghĩa thống kê. Chúng em không so sánh trung bình cộng ngây thơ, mà áp dụng kiểm định phi tham số Mann-Whitney U kết hợp hệ số kích thước ảnh hưởng Cliff's delta. Điều này bảo đảm kết luận tăng tốc là một cải tiến kỹ thuật thực chất chứ không phải do sai số ngẫu nhiên của môi trường đám mây."

---

### SLIDE 21. TRÌNH DIỄN GIAO DIỆN TƯƠNG TÁC DASHBOARD TRỰC TIẾP
- **Thời lượng:** 90 giây.
- **Bố cục hiển thị:** Khung trình chiếu video 16:9 chất lượng cao ghi lại các thao tác phân tích thực tế trên ứng dụng Streamlit Dashboard.
- **Cấu trúc kịch bản thao tác Dashboard:**
  - **Phân đoạn 1 (0:00 - 0:25): Tab Mức Layer:**
    - Chọn mô hình `resnet50` trên môi trường `colab_t4`.
    - Trực quan hóa bảng cơ cấu toán tử: Tầng `Conv2d` chiếm 73.8% thời gian.
    - Chuyển sang biểu đồ Roofline: Định vị các tầng tích chập bão hòa tính toán và các tầng chuẩn hóa nghẽn băng thông.
  - **Phân đoạn 2 (0:25 - 0:50): Tab Mức Model:**
    - Mở giao diện so sánh đối đầu CPU vs. GPU.
    - Thao tác trên biểu đồ trễ phân vị P50, P90, P99 và Bump Chart chuyển dịch thứ hạng của 8 mô hình.
    - Khảo sát bộ lọc mở rộng Batch Size (1 đến 32) và hiệu ứng chuyển đổi FP16 Tensor Cores.
  - **Phân đoạn 3 (0:50 - 1:15): Tab Mức Giai đoạn:**
    - Khảo sát chu trình Huấn luyện: Trực quan hóa tỷ số Backward/Forward 1.82x của ResNet-50.
    - Khảo sát điểm nghẽn DataLoader: Mô phỏng hiện tượng GPU Starvation tại `num_workers = 0` (62.6% I/O wait).
    - Khảo sát đường ống YOLOv8n: Phân rã 41% thời gian tiêu hao tại các công đoạn tiền và hậu xử lý.
  - **Phân đoạn 4 (1:15 - 1:30): Tab Chẩn đoán Điểm nghẽn & So sánh A/B:**
    - Xem danh sách khuyến nghị tự động của 19 quy tắc.
    - Thực hiện kiểm định A/B giữa cấu hình FP32 và FP16, hiển thị giá trị p-value và Cliff's delta.
- **Mã nguồn liên quan:** [`src/dashboard/app.py`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/src/dashboard/app.py).
- **Lời thoại thuyết trình (Speaker Script):**
  > "Sau đây, kính mời quý thầy cô cùng theo dõi đoạn video 90 giây ghi hình trực tiếp giao diện tương tác VisionProf Dashboard được vận hành trên thư viện Streamlit. Ứng dụng hiện đang được chạy sẵn sàng trên máy chủ nội bộ để phục vụ phần hỏi đáp trực tiếp.  
  > Trên Tab Mức Layer, kỹ sư có thể lập tức quan sát tỷ trọng thời gian của từng toán tử leaf và tọa độ Roofline tương ứng, giúp phát hiện ngay các tầng mạng chưa khai thác hết năng lực phần cứng.  
  > Chuyển sang Tab Mức Model, hệ thống cung cấp góc nhìn vĩ mô đa chiều: đối đầu độ trễ P50-P99 giữa Colab CPU và Tesla T4 GPU, biểu đồ mở rộng batch size, và tác động của nhân Tensor Cores.  
  > Tại Tab Mức Giai đoạn, các đường ống huấn luyện phức tạp và ứng dụng YOLO đầu-cuối được bóc tách trực quan thành các thanh màu rõ ràng, chỉ ra chính xác điểm nghẽn I/O hay chi phí tiền xử lý.  
  > Cuối cùng, Tab Chẩn đoán tự động xuất báo cáo kiến nghị mã màu và công cụ so sánh A/B với đầy đủ kiểm chứng thống kê toán học."

---

### SLIDE 22. TỔNG KẾT 6 CÂU HỎI NGHIÊN CỨU VÀ BẰNG CHỨNG THỰC NGHIỆM
- **Thời lượng:** 70 giây.
- **Bố cục hiển thị:** Bảng tổng hợp 6 dòng đối chiếu toàn bộ các câu hỏi nghiên cứu (CH1 đến CH6), câu trả lời ngắn gọn và bằng chứng thực nghiệm cụ thể.
- **Nội dung trên slide:**

| Câu hỏi nghiên cứu | Câu trả lời học thuật | Bằng chứng thực nghiệm cốt lõi |
| :--- | :--- | :--- |
| **CH1: Phân rã Mức Layer** | Conv chiếm 74-85% (CNN); Linear chiếm >95% (Transformer); GPU phân hóa tăng tốc mạnh ở tầng tính toán lớn. | ResNet-50 Conv2d: 73.8%; BERT Linear: 95.4%; BatchNorm tăng tỷ trọng từ 6.8% lên 14.6% khi lên GPU. |
| **CH2: Độ chuẩn xác đo lường** | Sai lệch tổng thời gian dưới 8% so với `torch.profiler`; kiểm soát hoàn toàn hiện tượng bất đồng bộ. | ResNet-50 sai lệch 7.8%; BERT-base sai lệch 0.04%; Overhead đo lường được cô lập hoàn toàn khỏi Mức Model. |
| **CH3: Nghịch lý Số tham số** | Ở Batch 1 trên GPU, tham số không quyết định tốc độ do nghẽn Kernel Launch; thứ hạng thay đổi theo phần cứng. | ResNet-50 (8.2 GFLOPs) $\approx$ MobileNetV3 (0.43 GFLOPs) $\approx$ 6.0 ms trên T4; DistilBERT bứt phá thứ hạng trên GPU. |
| **CH4: Mở rộng Batch & FP16** | Batch lớn giải phóng mạng nhẹ (tăng tới 8.8x); FP16 luôn giảm 40-49% VRAM nhưng chỉ tăng tốc Transformer. | MobileNetV3 tăng từ 170 lên 1,499 FPS; ViT tăng tốc FP16 $2.81\times$; CNN ở batch 1 bị suy giảm tốc độ do chi phí ép kiểu. |
| **CH5: Phân rã Huấn luyện** | Backward gấp 1.4-2.2x Forward; VRAM bùng nổ gấp 3x; Thiếu worker khiến GPU ngồi chờ dữ liệu tới 63%. | DistilBERT Bwd/Fwd = 2.21x; VRAM ResNet tăng $2.71\times$; MobileNetV3 chờ I/O 62.6% khi `num_workers = 0`. |
| **CH6: Điểm nghẽn Pipeline** | Chi phí ngoài mô hình chiếm tới 41% trong ứng dụng thị giác thực tế, xác nhận trọn vẹn định luật Amdahl và AI Tax. | YOLOv8n trên T4: Preprocess (3.23 ms) + NMS (1.15 ms) = 4.38 ms (chiếm 41.0% tổng độ trễ 10.68 ms). |

- **Lời thoại thuyết trình (Speaker Script):**
  > "Nhìn lại toàn bộ công trình nghiên cứu, VisionProf đã giải quyết trọn vẹn và trả lời dứt khoát 6 câu hỏi khoa học lớn được đặt ra ở đầu đề tài:  
  > Chúng em đã lượng hóa chính xác sự thống trị của Conv trong CNN và Linear trong Transformer ở CH1;  
  > Chứng minh độ tin cậy của giải thuật Deferred CUDA Events với sai số dưới 8% ở CH2;  
  > Đập tan định kiến về số lượng tham số và phát hiện sự chuyển dịch thứ hạng hiệu năng ở CH3;  
  > Chỉ ra quy luật mở rộng thông lượng cùng cảnh báo hai mặt của số thực FP16 ở CH4;  
  > Làm sáng tỏ tỷ số lan truyền ngược 2 lần, sự bùng nổ bộ nhớ và hiện tượng đói dữ liệu GPU ở CH5;  
  > Và cuối cùng, thực nghiệm chứng minh chi phí ngoài mô hình chiếm tới 41% trong pipeline ứng dụng ở CH6.  
  > Toàn bộ các kết luận trên đều được bảo chứng bằng dữ liệu thực nghiệm đo thật 100% trên nền tảng Google Colab."

---

### SLIDE 23. HẠN CHẾ HỆ THỐNG, HƯỚNG PHÁT TRIỂN VÀ KẾT LUẬN
- **Thời lượng:** 60 giây.
- **Bố cục hiển thị:**
  - Trái: Bảng phân tích các Hạn chế kỹ thuật hiện tại và Đánh giá rủi ro (Threats to Validity).
  - Phải: Lộ trình phát triển hệ thống trong giai đoạn cuối kỳ.
  - Dưới: Thông tin liên kết mã nguồn mở (GitHub Repository) và Mã QR tương ứng.
- **Nội dung trên slide:**
  - **1. Các Hạn chế Kỹ thuật Hiện tại (Limitations):**
    - Hệ thống chưa truy cập trực tiếp các bộ đếm phần cứng chuyên sâu (Hardware Performance Counters thông qua PAPI hoặc CUPTI).
    - Tồn tại khoảng 9% đến 10.5% thời gian thực thi của các toán tử dạng hàm không có trạng thái (`x + shortcut`, `torch.cat`) chưa được bóc tách riêng lẻ ở Mức Layer.
    - Phạm vi thực nghiệm hiện tại giới hạn ở cấu hình đơn GPU (Single-GPU) và chế độ thực thi PyTorch Eager Mode; sử dụng trọng số khởi tạo ngẫu nhiên (bảo đảm tính chuẩn xác về tốc độ và bộ nhớ nhưng không đánh giá độ chính xác hội tụ).
  - **2. Lộ trình Mở rộng trong Giai đoạn Tiếp theo:**
    - Tích hợp công nghệ biên dịch đồ thị `torch.compile` (Inductor Backend) để đánh giá mức độ dung hợp toán tử (Operator Fusion).
    - Ứng dụng cơ chế `TorchDispatchMode` nhằm bắt vết toàn diện 100% các toán tử dạng hàm chức năng.
    - Tự động hóa đề xuất mã nguồn sửa lỗi tối ưu (Code Patch Generation) trực tiếp từ các cảnh báo của 19 quy tắc chẩn đoán.
  - **Thông tin tham khảo:** Toàn bộ mã nguồn, tài liệu thiết kế và dữ liệu đo thô được công khai tại repository đồ án.
- **Lời thoại thuyết trình (Speaker Script):**
  > "Để kết thúc bài báo cáo, chúng em xin thẳng thắn nhìn nhận những hạn chế kỹ thuật hiện tại: VisionProf chưa can thiệp vào các bộ đếm chu kỳ phần cứng ở mức thanh ghi vi kiến trúc; khoảng 10% thời gian của các toán tử hàm functional chưa có hook riêng; và hệ thống mới thử nghiệm trên đơn GPU ở chế độ Eager Mode.  
  > Trong giai đoạn nghiên cứu tiếp theo, nhóm đặt mục tiêu tích hợp công nghệ biên dịch `torch.compile`, áp dụng cơ chế `TorchDispatchMode` để bắt vết trọn vẹn 100% các phép toán hàm, và phát triển tính năng tự động sinh mã nguồn tối ưu hóa cho người dùng.  
  > Nhóm chúng em xin trân trọng cảm ơn quý thầy cô trong Hội đồng đã chú ý lắng nghe. Chúng em rất mong nhận được những câu hỏi nhận xét và góp ý quý báu từ thầy cô!"

---

# PHẦN PHỤ LỤC: DANH MỤC 10 CÔNG TRÌNH KHOA HỌC THAM CHIẾU (IEEE / ACM)

1. **[Roofline Model]** S. Williams, A. Waterman, and D. Patterson, *"Roofline: An Insightful Visual Performance Model for Multicore Architectures,"* Communications of the ACM (CACM), vol. 52, no. 4, pp. 65–76, 2009.
2. **[MLModelScope]** A. Dakkak, C. Li, J. Xiong, and W. Hwu, *"MLModelScope: Evaluating and Profiling ML Models Across the Stack,"* in Proceedings of the IEEE International Parallel and Distributed Processing Symposium (IPDPS), 2020.
3. **[MLPerf Inference]** V. J. Reddi et al., *"MLPerf Inference Benchmark,"* in Proceedings of the ACM/IEEE International Symposium on Computer Architecture (ISCA), 2020, pp. 446–459.
4. **[Data Stalls]** J. Mohan, A. Phanishayee, J. Kalia, and V. Chidambaram, *"Analyzing and Mitigating Data Stalls in DNN Training,"* Proceedings of the VLDB Endowment (PVLDB), vol. 14, no. 6, pp. 958–970, 2021.
5. **[AI Tax]** M. Richins et al., *"Missing the Target: The Hidden Cost of AI Pipelines (AI Tax),"* in Proceedings of the IEEE/ACM International Symposium on Computer Architecture (HPCA), 2020.
6. **[TBD Benchmark]** H. Zhu et al., *"TBD: Benchmarking and Analyzing Deep Learning Training Workloads,"* in Proceedings of the IEEE International Symposium on Workload Characterization (IISWC), 2018, pp. 268–279.
7. **[XSP Profiling]** A. Dakkak et al., *"Across-Stack Profiling and Analysis of Machine Learning Workloads (XSP),"* in Proceedings of the ACM/IEEE International Symposium on Computer Architecture (ISCA), 2021.
8. **[Deep500]** T. Ben-Nun et al., *"Deep500: An Modular Benchmarking Infrastructure for Deep Learning,"* IEEE Transactions on Parallel and Distributed Systems (TPDS), 2020.
9. **[PyTorch Profiler]** J. Fang et al., *"PyTorch Profiler: Performance Debugging in Deep Learning Frameworks,"* in Proceedings of the Conference on Machine Learning and Systems (MLSys), 2020.
10. **[Architectural Papers]**
    - K. He, X. Zhang, S. Ren, and J. Sun, *"Deep Residual Learning for Image Recognition,"* IEEE CVPR, 2016.
    - A. Howard et al., *"Searching for MobileNetV3,"* IEEE ICCV, 2019.
    - J. Devlin et al., *"BERT: Pre-training of Deep Bidirectional Transformers,"* NAACL, 2019.

---

# KỊCH BẢN HỎI ĐÁP PHẢN BIỆN CHUYÊN SÂU (Q&A DEFENSE PROTOCOL)

### CÂU HỎI DÀNH CHO THÀNH VIÊN 1 (TV1 - MỨC LAYER & ĐỘ CHUẨN XÁC PHẦN CỨNG)

**Câu hỏi 1: Tại sao Overhead đo lường ở Mức Layer trên GPU ở Batch 1 lại lớn (từ 110% đến 210%), và điều này có làm sai lệch độ tin cậy của các kết luận hay không?**
- **Phương án trả lời:**
  > "Kính thưa thầy cô, câu hỏi này chạm đúng vào bản chất kỹ thuật của 'Nghịch lý Overhead' mà bài báo XSP tại ISCA 2021 đã cảnh báo.  
  > Trên GPU Tesla T4 ở kích thước batch 1, các tầng mạng siêu nhỏ như ReLU hay BatchNorm chỉ mất từ 3 đến 8 micro-giây để tính toán. Trong khi đó, việc ghi nhận 2 mốc sự kiện CUDA Event vào hàng đợi luồng lệnh tiêu tốn khoảng 4 đến 6 micro-giây của CPU driver. Khi mô hình có hơn 100 tầng leaf như ResNet-50, tổng chi phí ghi nhận sự kiện sẽ tích lũy và làm kéo dài thời gian hoàn thành tổng thể.  
  > Tuy nhiên, điều này **hoàn toàn không làm sai lệch kết luận khoa học** của VisionProf vì 3 lý do:  
  > Thứ nhất, kỹ thuật Deferred CUDA Events hoãn toàn bộ thao tác đồng bộ đến cuối chu kỳ, do đó không phá vỡ trật tự thực thi của phần cứng và giữ nguyên tỷ lệ thời gian tương đối giữa các tầng.  
  > Thứ hai, kiểm chứng thực nghiệm độc lập so với công cụ chuẩn mực `torch.profiler` cho thấy sai lệch tổng thời gian chỉ là **7.8% trên ResNet-50 và 0.04% trên BERT-base**.  
  > Thứ ba, trong kiến trúc của VisionProf, Mức Layer được phân định rõ ràng là chỉ dùng để **phân tích tỷ trọng cơ cấu (Breakdown)**; còn tổng thời gian chuẩn xác tuyệt đối của mô hình được thu thập độc lập ở Mức Model với quy trình đo sạch không có hook."

**Câu hỏi 2: Các phép toán dạng hàm không có trạng thái (Functional Operations như `x + shortcut`, `torch.cat`) đi đâu trong bảng phân rã Mức Layer?**
- **Phương án trả lời:**
  > "Kính thưa thầy cô, đây là một thách thức kinh điển của cơ chế PyTorch Module Hooks. Vì các phép toán như `x + shortcut` trong khối Residual hay phép nối tensor `torch.cat` là các hàm stateless không kế thừa từ `torch.nn.Module`, nên các hook tiền và hậu xử lý không thể tự động bám vào chúng.  
  > VisionProf giải quyết vấn đề này một cách minh bạch bằng cách định nghĩa chỉ số **'Thời gian không thuộc tầng nào' (Unaccounted Time)**: được tính bằng tổng thời gian toàn mô hình đo được trừ đi tổng thời gian của tất cả các leaf modules.  
  > Kết quả thực nghiệm của chúng em chỉ ra đại lượng này chiếm từ **9.0% đến 9.5% trên ResNet-50** và **10.3% đến 10.5% trên BERT-base** trên GPU Tesla T4. Việc lượng hóa rõ ràng tỷ lệ này giúp người dùng biết chính xác chi phí ẩn của các phép toán đồ thị, thay vì che giấu sai số vào các tầng mạng khác."

---

### CÂU HỎI DÀNH CHO THÀNH VIÊN 2 (TV2 - MỨC MODEL & ĐỐI ĐẦU CPU VS. GPU)

**Câu hỏi 1: Tại sao ở kích thước Batch 1 trên GPU Tesla T4, MobileNetV3 nhẹ hơn ResNet-50 tới 19 lần về FLOPs nhưng độ trễ lại tương đương nhau (~6.0 ms)?**
- **Phương án trả lời:**
  > "Kính thưa thầy cô, phát hiện này thể hiện rõ sự chuyển dịch từ trạng thái nghẽn tính toán sang trạng thái nghẽn điều phối (Latency/Launch-Bound Regime) của kiến trúc máy tính.  
  > GPU NVIDIA Tesla T4 sở hữu 2,560 nhân CUDA với năng lực tính toán cực lớn. Để khai thác hết sức mạnh này, GPU cần một khối lượng dữ liệu song song khổng lồ. Tuy nhiên, ở batch 1, kích thước các tensor của MobileNetV3 quá nhỏ (đặc biệt là các tầng Depthwise Convolution phân tách theo từng kênh riêng lẻ).  
  > Thời gian thực tế mà phần cứng GPU dùng để thực hiện phép tính chỉ mất dưới 1 mili-giây. Phần lớn thời gian 5 mili-giây còn lại là độ trễ khởi động kernel (Kernel Launch Latency, khoảng 5-10 micro-giây mỗi kernel) tuần tự từ CPU qua driver NVIDIA. Vì MobileNetV3 có tới hơn 150 tầng và toán tử nhỏ, tổng độ trễ khởi động kernel đã lấn át hoàn toàn lợi thế giảm thiểu FLOPs, khiến nó có độ trễ đo được ngang bằng với ResNet-50."

**Câu hỏi 2: Tại sao việc chuyển đổi sang định dạng nửa chính xác FP16 lại làm các mô hình CNN ở Batch 1 chạy chậm hơn từ 13% đến 23%?**
- **Phương án trả lời:**
  > "Kính thưa thầy cô, đây là một lưu ý kỹ thuật rất quan trọng khi sử dụng nhân Turing Tensor Cores.  
  > Nhân Tensor Cores của Tesla T4 chỉ kích hoạt hiệu quả đối với các phép nhân ma trận có kích thước là bội số của 8 hoặc 16 và có khối lượng tính toán đủ lớn. Đối với các mô hình Transformer như ViT hay BERT, các ma trận chiếu $Q, K, V$ rất lớn nên Tensor Cores phát huy tối đa công suất, giúp ViT tăng tốc tới 2.81 lần.  
  > Ngược lại, với các mô hình CNN ở batch 1, kích thước của các tensor tích chập không thỏa mãn ngưỡng tối ưu của Tensor Cores, do đó GPU vẫn phải định tuyến phép tính qua các nhân CUDA FP32 thông thường hoặc nhân FP16 tiêu chuẩn. Lúc này, mô hình không những không được hưởng lợi từ Tensor Cores mà còn phải gánh thêm chi phí chuyển đổi kiểu dữ liệu (Type Casting overhead) từ FP32 sang FP16 và ngược lại do framework thực hiện. Kết quả là độ trễ bị kéo dài thêm từ 13% đến 23%."

---

### CÂU HỎI DÀNH CHO THÀNH VIÊN 3 (TV3 - MỨC GIAI ĐOẠN, PIPELINE & CHẨN ĐOÁN)

**Câu hỏi 1: Công đoạn ngoài mô hình chiếm tới 41% trong ứng dụng YOLOv8n trên GPU thì giải pháp kỹ thuật khắc phục trong thực tế là gì?**
- **Phương án trả lời:**
  > "Kính thưa thầy cô, con số 41% mà VisionProf phát hiện chính là minh chứng thực tế cho lý thuyết AI Tax. Nguyên nhân gốc rễ là hai thao tác Letterbox (tiền xử lý) và NMS (hậu xử lý) hiện đang chạy bằng mã nguồn Python/NumPy tuần tự trên CPU máy chủ.  
  > Để khắc phục triệt để điểm nghẽn này theo đúng định luật Amdahl, trong thực tế sản xuất kỹ sư cần áp dụng 3 giải pháp kiến trúc:  
  > Thứ nhất, đẩy toàn bộ công đoạn tiền xử lý (giải nén ảnh JPEG, resize và chuẩn hóa) trực tiếp lên GPU thông qua thư viện phần cứng chuyên dụng như **NVIDIA DALI** hoặc các toán tử tensor của `torchvision` trên bộ nhớ CUDA.  
  > Thứ hai, thay thế thuật toán NMS tuần tự trên CPU bằng giải thuật Batched NMS chạy song song bằng CUDA C++ kernel (như tích hợp sẵn trong NVIDIA TensorRT).  
  > Thứ ba, thiết kế kiến trúc xử lý dạng đường ống bất đồng bộ (Asynchronous Multi-threaded Pipeline): Trong khi GPU đang suy luận khung hình thứ $t$, CPU sẽ tiền xử lý khung hình $t+1$ và hậu xử lý khung hình $t-1$, triệt tiêu thời gian nhàn rỗi của hệ thống."

**Câu hỏi 2: Tại sao trên môi trường Google Colab, khi tăng `num_workers` từ 2 lên 4 thì tốc độ huấn luyện lại không tăng mà còn giảm nhẹ?**
- **Phương án trả lời:**
  > "Kính thưa thầy cô, kết quả này phản ánh sự tương tác mật thiết giữa phần mềm học sâu và kiến trúc phần cứng máy chủ.  
  > Trong cấu hình máy ảo chuẩn của Google Colab, hệ thống chỉ cấp phát **chính xác 2 nhân vCPU vật lý**.  
  > Khi chúng ta thiết lập `num_workers = 2`, 2 tiến trình nạp dữ liệu chạy song song ăn khớp hoàn hảo với 2 nhân vCPU, giúp việc đọc đĩa và biến đổi ảnh diễn ra bất đồng bộ và che giấu hoàn toàn độ trễ so với GPU.  
  > Nhưng khi chúng ta nâng lên `num_workers = 4`, hệ điều hành Linux buộc phải điều phối 4 tiến trình nạp dữ liệu cùng với tiến trình chính của PyTorch trên vỏn vẹn 2 vCPU. Sự quá tải này dẫn đến hiện tượng tranh chấp tài nguyên CPU nghiêm trọng (CPU Thrashing) và chi phí chuyển đổi ngữ cảnh (Context Switching Overhead) tăng cao, làm tổng chu kỳ bước lặp bị kéo dài từ 116.9 ms lên 119.8 ms. Điều này chứng minh rằng việc cấu hình siêu tham số DataLoader bắt buộc phải dựa trên số lượng nhân CPU thực tế của máy chủ."
