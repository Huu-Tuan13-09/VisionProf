# Dàn ý slide trình bày hoàn chỉnh (24 phút · 22 slide · 3 thành viên)

> 📌 **Nguyên tắc cốt lõi về số liệu:** Tài liệu này là **DÀN Ý CẤU TRÚC VÀ LUẬN ĐIỂM THUYẾT TRÌNH**, tập trung vào: **Bố cục hiển thị (Visual Layout)**, **Luận điểm khoa học**, **Cơ chế kỹ thuật** và **Thông điệp chốt**.  
> Mọi kết quả đo lường cụ thể được chuẩn hóa dưới dạng các vị trí giữ chỗ `[Số liệu...]` để cập nhật số liệu thực tế sau, tuyệt đối không phụ thuộc vào số liệu cứng ở bước này.

---

## ⏱️ Tổng quan phân bổ thời gian & Phân vai

| Phần | Người phụ trách | Slide | Thời lượng | Trọng tâm khoa học & Luận điểm |
| :--- | :---: | :---: | :---: | :--- |
| **Phần 1: Mở đầu & Khung bài toán** | 🟩 **TV3** | **1–3** | **3.0'** | Định vị Profiling, 3 mức đo, 2 mô hình xuyên suốt |
| **Phần 2: Mức Layer (TV1)** | 🟧 **TV1** | **4–9** | **7.0'** | Thách thức đo GPU, Roofline, CH1, CH2 (Tự kiểm chứng TN2) |
| **Phần 3: Mức Model (TV2)** | 🟦 **TV2** | **10–15** | **7.0'** | Đo sạch 2 pha, Chuẩn MLPerf P50–P99, CH3, CH4 |
| **Phần 4: Mức Giai đoạn & Tổng kết** | 🟩 **TV3** | **16–22** | **7.0'** | Training CH5, Pipeline Amdahl CH6, Chẩn đoán, Demo, Kết luận |
| **Tổng cộng** | **3 thành viên** | **22** | **24.0'** | **Dành riêng 5–10' sau bài cho phần Q&A và Live Demo trực tiếp** |

---

## 🟩 TV3: Phần 1. Mở đầu & Khung bài toán (Slide 1–3 · 3.0')

### Slide 1. Tiêu đề & Giới thiệu đề tài
- **Bố cục trực quan:** Tiêu đề trung tâm, bên dưới chia 3 khối tác giả kèm mã màu nhận diện (🟩 TV3, 🟧 TV1, 🟦 TV2), thông tin môn học và giảng viên hướng dẫn.
- **Nội dung hiển thị:**
  - *Đề tài:* VisionProf: Nền tảng Phân tích và Chẩn đoán Hiệu năng Mô hình Học sâu ở 3 Mức (Layer, Model, Giai đoạn).
  - *Đặc tính hệ thống:* Bộ công cụ profiling nhẹ cho PyTorch, bọc mô hình không cần sửa đổi mã nguồn (Zero-code modification).
- **Thông điệp chốt:** VisionProf cung cấp góc nhìn toàn diện từ vi mô phép tính đến vĩ mô pipeline ứng dụng.

### Slide 2. Đặt vấn đề: Evaluation, Benchmarking hay Profiling?
- **Bố cục trực quan:** Bảng so sánh 3 cột phân định khái niệm, cột thứ 3 (Profiling) được đóng khung viền nổi bật.
- **Nội dung hiển thị:**
  - *Evaluation (Đánh giá thuật toán):* Mô hình dự đoán đúng bao nhiêu? (Accuracy, mAP, F1) $\rightarrow$ *Ngoài phạm vi đề tài*.
  - *Benchmarking (Đo hiệu năng đen):* Mô hình chạy nhanh bao nhiêu? (Latency trung bình, Throughput tổng).
  - *Profiling (Mổ xẻ tài nguyên):* **Thời gian và bộ nhớ thực sự đi đâu?** Nghẽn do tính toán hay băng thông phần cứng?
- **Thông điệp chốt:** Đề tài tập trung 100% vào **Profiling**; giải thích cặn kẽ hành vi tiêu hao tài nguyên phần cứng.

### Slide 3. Khung 3 mức đo & Sợi chỉ đỏ dẫn dắt (The Red Thread)
- **Bố cục trực quan:** Sơ đồ phân tầng 3 mức đo xếp chồng, đặt cạnh biểu tượng của 2 mô hình đối sánh xuyên suốt.
- **Nội dung hiển thị:**
  - 3 Mức khảo sát: 🟧 Mức Layer (TV1) $\rightarrow$ 🟦 Mức Model (TV2) $\rightarrow$ 🟩 Mức Giai đoạn (TV3), trả lời 6 câu hỏi CH1–CH6.
  - Sợi chỉ đỏ dẫn dắt: Theo dõi hành trình đối đầu giữa **ResNet-50** (đại diện CNN) và **DistilBERT / ViT** (đại diện Transformer) qua cả 3 tầng.
- **Thông điệp chốt & Lời chuyển giao:** *"Để hiểu bản chất vật lý từ những phép tính vi mô nhất bên trong phần cứng, xin mời TV1 bắt đầu với Mức Layer."*

---

## 🟧 TV1: Phần 2. Mức Layer — Bản chất phần cứng & Kiểm chứng (Slide 4–9 · 7.0')

### Slide 4. Thách thức kỹ thuật: Đo GPU không dễ
- **Bố cục trực quan:** Sơ đồ dòng thời gian (Timeline): So sánh cơ chế CPU gửi lệnh bất đồng bộ với GPU thực thi; chỉ ra lỗi sai số của đồng hồ CPU.
- **Nội dung hiển thị:**
  - *Bản chất bất đồng bộ:* CPU chỉ xếp lệnh vào hàng đợi rồi chạy tiếp $\rightarrow$ dùng đồng hồ CPU chỉ đo được thời gian phát lệnh (Launch latency).
  - *Giải pháp Deferred CUDA Events:* Chèn Event đánh dấu trước/sau từng layer; chỉ đồng bộ `cudaStreamSynchronize` duy nhất 1 lần ở cuối vòng; dùng ngăn xếp LIFO cho layer tái sử dụng.
- **Thông điệp chốt:** Kỹ thuật Deferred CUDA Events bảo toàn luồng chạy song song của GPU, phản ánh đúng thời gian tính toán thực tế.

### Slide 5. Nền tảng học thuật: Roofline Model & Nghịch lý Overhead
- **Bố cục trực quan:** Chia 2 cột: Cột trái nhúng biểu đồ Roofline (`01_roofline_fig1.png`); Cột phải nhúng biểu đồ Overhead theo tầng đo từ paper XSP (`04_xsp_fig2_leveled_overhead.png`).
- **Nội dung hiển thị:**
  - *Roofline Model (Williams et al., CACM 2009):* Phân định ranh giới giữa nghẽn tính toán (Compute-Bound) và nghẽn băng thông bộ nhớ (Memory-Bound).
  - *Nghịch lý Overhead (Li et al., XSP, IPDPS 2020):* Đo càng chi tiết ở mức sâu thì công cụ đo càng làm phình to thời gian chạy thực tế.
- **Thông điệp chốt:** Mức Layer được thiết kế chuyên biệt để **phân rã tỷ lệ %**, việc chốt tổng thời gian sạch được giao cho Mức Model.

### Slide 6. CH1: Thời gian tập trung ở loại Layer nào?
- **Bố cục trực quan:** Biểu đồ cơ cấu thời gian theo nhóm layer (`fig1_type_share.png`) kèm bảng tỷ trọng thành phần.
- **Nội dung hiển thị:**
  - *CNN (ResNet-50):* Phép tính tích chập (Conv) chiếm tỷ trọng thời gian áp đảo `[Điền % Conv CPU vs T4]`.
  - *Transformer (ViT, BERT):* Nhân ma trận (Linear) và Attention chiếm tỷ trọng gần như tuyệt đối `[Điền % Linear+Attention]`.
  - *Mô hình nhẹ (MobileNetV3):* Trên GPU, tỷ trọng Conv giảm nhường chỗ cho BatchNorm và hàm kích hoạt tăng vọt `[Điền bảng % thay đổi]`.
- **Thông điệp chốt:** Các phép tính lớn được tăng tốc mạnh trên GPU khiến các thao tác phụ trợ nhỏ (element-wise) bộc lộ thành điểm nghẽn tỷ trọng.

### Slide 7. CH1: GPU giúp được bao nhiêu, và vì sao?
- **Bố cục trực quan:** Ghép biểu đồ cột Speedup GPU vs CPU (`fig2_gpu_speedup.png`) và Biểu đồ phân bố điểm làm việc trên Roofline (`fig3_roofline.png`).
- **Nội dung hiển thị:**
  - *So sánh gia tốc:* Transformer đạt mức tăng tốc vượt trội so với CNN nhờ cấu trúc nhân ma trận GEMM thân thiện với GPU `[Điền hệ số Speedup từng họ]`.
  - *Khai thác trần phần cứng:* Biểu đồ Roofline chỉ ra ở Batch Size = 1, các layer lớn vẫn chưa khai thác hết công suất đỉnh của GPU `[Điền % công suất Roofline]`.
- **Thông điệp chốt:** Kiến trúc Transformer tận dụng năng lực tính toán song song của GPU hiệu quả hơn nhiều so với kiến trúc tuần tự của CNN.

### Slide 8. CH2: Công cụ đo có chính xác không? (Tự kiểm chứng TN2)
- **Bố cục trực quan:** Biểu đồ đối chiếu sai số (`fig4_tn2_error.png`) kết hợp Bảng ma trận kiểm chứng độc lập so với `torch.profiler`.
- **Nội dung hiển thị:**
  - *Sai lệch thời gian:* Đạt độ chính xác khoa học cao so với công cụ chuẩn trên cả CPU và GPU `[Điền bảng sai số % CPU vs T4]`.
  - *Đo lường Overhead:* Đo đạc minh bạch độ phồng thời gian của công cụ đo ở batch 1 `[Điền bảng % overhead CPU vs T4]`.
  - *Phát hiện kỹ thuật:* Phân tích giới hạn sàn phần cứng của CUDA Event (~6 µs) gây đo dư ở các layer siêu ngắn (Sub-microsecond).
- **Thông điệp chốt:** VisionProf chứng minh tính đúng đắn khoa học thông qua tự kiểm chứng và công khai chi phí can thiệp phần cứng.

### Slide 9. Tổng kết Mức Layer & Khuyến nghị kỹ thuật
- **Bố cục trực quan:** 3 khối hộp tóm tắt khuyến nghị có icon trực quan, đóng khung vai trò của Mức Layer.
- **Nội dung hiển thị:**
  - 1. Muốn tối ưu CNN phải nhắm vào Conv; tối ưu Transformer phải nhắm vào Linear và Attention.
  - 2. Không lấy tổng thời gian từ Mức Layer do ảnh hưởng của overhead ghi nhận sự kiện (Events).
  - 3. Chuyển giao nhiệm vụ chốt độ trễ toàn thể cho Mức Model.
- **Thông điệp chốt & Lời chuyển giao:** *"Sau khi đã hiểu bản chất vi mô bên trong, xin mời TV2 trình bày bức tranh tổng thể ở Mức Model."*

---

## 🟦 TV2: Phần 3. Mức Model — Phân vị trễ & Sức mạnh phần cứng (Slide 10–15 · 7.0')

### Slide 10. Kiến trúc Mức Model & Kỹ thuật đo sạch 2 pha
- **Bố cục trực quan:** Sơ đồ khối kiến trúc `ModelAdapter` chuẩn hóa 8 mô hình, kết hợp sơ đồ luồng quy trình đo 2 pha độc lập.
- **Nội dung hiển thị:**
  - *Nguyên tắc đo sạch 2 pha (Clean 2-Phase Profiling):*
    - Pha 1: Vòng lặp đo Latency thuần túy (tắt toàn bộ profiler/hook) $\rightarrow$ triệt tiêu hoàn toàn độ phồng trễ.
    - Pha 2: Bật profiler thu thập thông số bộ nhớ và tài nguyên hệ thống.
  - *Thu thập VRAM chuẩn xác:* Đọc trực tiếp từ bộ cấp phát của GPU qua `torch.cuda.max_memory_allocated()`.
- **Thông điệp chốt:** Tách biệt đo tốc độ và đo tài nguyên là nguyên tắc sống còn để thu được độ trễ sạch.

### Slide 11. Chuẩn đo lường công nghiệp & Thiết lập thực nghiệm
- **Bố cục trực quan:** Đồ thị hàm mật độ phân phối độ trễ (Latency Distribution) làm nổi bật các mốc phân vị P50, P90, P99 và dải trễ đuôi (Tail Latency).
- **Nội dung hiển thị:**
  - *Chuẩn công nghiệp MLPerf (Reddi et al., ISCA 2020):* Bác bỏ trung bình cộng đơn giản; bắt buộc dùng $P_{90}$ và $P_{99}$ để đánh giá độ ổn định dịch vụ.
  - *Quy chuẩn thực nghiệm:* Khởi động ấm (Warm-up) 10 vòng, đo lặp 50 vòng thống kê, tự động kết xuất môi trường phần cứng `env.json`.
  - *Lưu ý môi trường CPU:* Phân định rạch ròi 2 cấu hình máy CPU (laptop x86 của TV1 vs MacBook chip Apple ARM64 của TV2), không so sánh chéo trực tiếp số đo giữa 2 máy.
- **Thông điệp chốt:** Đánh giá triển khai thực tế bắt buộc phải kiểm soát độ trễ đuôi $P_{99}$ chống hiện tượng nghẽn ngẫu ứng.

### Slide 12. CH3: Nghịch lý tham số & Hiện tượng Memory-Bound (Đo thật CPU)
- **Bố cục trực quan:** Biểu đồ đối đầu: Trục tung là Số lượng tham số, Trục hoành là Độ trễ thực tế trên CPU.
- **Nội dung hiển thị:**
  - *Nghịch lý đo thật:* Mô hình ít tham số nhất (MobileNetV3) lại có độ trễ lớn hơn nhiều lần so với mô hình nhiều tham số hơn (ResNet-50, DistilBERT) `[Điền số đo P50 và FPS CPU]`.
  - *Bản chất phần cứng:* Depthwise Separable Conv có cường độ số học thấp, biến MobileNet thành tác vụ nghẽn băng thông bộ nhớ RAM (Memory-Bound). Trên chip ARM64, kernel PyTorch cho Depthwise Conv tối ưu chưa tốt bằng x86, làm độ trễ tăng vọt. Ngược lại, Standard Conv và Attention tận dụng tốt bộ đệm cache và tập lệnh SIMD.
- **Thông điệp chốt:** Số lượng tham số **không** tỷ lệ thuận với tốc độ; độ tương thích giữa kiến trúc và phân cấp bộ nhớ mới là yếu tố quyết định.

### Slide 13. CH3: Thứ hạng đảo chiều trên GPU Colab T4
- **Bố cục trực quan:** Biểu đồ đường dốc thay đổi thứ hạng (Rank-Shift Slope Chart) nối từ phía CPU sang phía GPU T4.
- **Nội dung hiển thị:**
  - *Sự đảo chiều:* Băng thông VRAM cực lớn của GPU giải tỏa nút thắt bộ nhớ; các phép nhân ma trận lớn bứt phá về hiệu năng `[Điền bảng thứ hạng trễ trên T4]`.
  - *Nguyên tắc lựa chọn:* Khẳng định nguyên lý "Hardware-aware Selection" — không có mô hình nào tối ưu trên mọi loại phần cứng.
- **Thông điệp chốt:** Lựa chọn kiến trúc mô hình bắt buộc phải căn cứ vào phần cứng triển khai thực tế.

### Slide 14. CH4: Khả năng mở rộng Batch Size & Chiều dài chuỗi NLP
- **Bố cục trực quan:** Đồ thị đường biểu diễn Throughput (FPS) theo Batch Size và Đồ thị trễ Self-Attention theo chiều dài token.
- **Nội dung hiển thị:**
  - *Mở rộng Batch Size:* Thông lượng tăng nhanh ở các batch đầu nhưng đạt điểm bão hòa sớm; GPU cần batch đủ lớn để bù đắp chi phí khởi tạo `[Điền điểm bão hòa Throughput]`.
  - *Chiều dài chuỗi NLP:* Độ trễ bùng nổ theo bình phương chiều dài chuỗi $O(N^2)$ của cơ chế Attention, gây suy giảm mạnh thông lượng ở batch lớn `[Điền hệ số suy giảm FPS]`.
- **Thông điệp chốt:** Dynamic Batching là giải pháp cốt lõi cho GPU; cần giới hạn chặt chẽ chiều dài chuỗi đầu vào trong NLP.

### Slide 15. CH4: Sức mạnh Tensor Cores FP16 & Bảo toàn độ chính xác
- **Bố cục trực quan:** Biểu đồ cột nhóm so sánh FP32 vs FP16 trên 2 trục: Hệ số tăng tốc độ trễ và Tỷ lệ tiết kiệm dung lượng VRAM; Hộp chỉ số tương đồng đầu ra.
- **Nội dung hiển thị:**
  - *Gia tốc phần cứng:* Bật nửa độ chính xác FP16 kích hoạt nhân Tensor Cores giúp giảm độ trễ đáng kể `[Điền hệ số tăng tốc FP16]`.
  - *Tối ưu bộ nhớ:* Cắt giảm mạnh mẽ dung lượng VRAM đỉnh `[Điền % VRAM tiết kiệm]`.
  - *Kiểm chứng sai số:* Chỉ số Cosine Similarity giữa đầu ra FP32 và FP16 đạt mức tiệm cận 1.0 `[Điền Cosine Similarity]`.
- **Thông điệp chốt & Lời chuyển giao:** *"Mô hình AI chỉ là một mắt xích. Xin mời TV3 đưa chúng ta đến bức tranh toàn cảnh của Huấn luyện và Pipeline thực tế."*

---

## 🟩 TV3: Phần 4. Mức Giai đoạn, Chẩn đoán & Tổng kết (Slide 16–22 · 7.0')

### Slide 16. Kỹ thuật Mức Giai đoạn: Context Manager & PhaseTimer
- **Bố cục trực quan:** Sơ đồ phân rã các giai đoạn thực thi kết hợp đoạn mã mẫu bọc Context Manager (`with timer.phase():`) có đánh dấu đồng bộ luồng CUDA.
- **Nội dung hiển thị:**
  - *Cơ chế `PhaseTimer`:* Phân đoạn rạch ròi các công đoạn nối tiếp mà không xâm lấn kiến trúc mô hình; đồng bộ chính xác tại các biên thời gian.
  - *Hai phạm vi khảo sát:* Chu trình Huấn luyện (Forward $\rightarrow$ Loss $\rightarrow$ Backward $\rightarrow$ Step) và Pipeline Triển khai (Tiền xử lý $\rightarrow$ Model Forward $\rightarrow$ Hậu xử lý).
- **Thông điệp chốt:** Phân rã giai đoạn là phương pháp duy nhất định vị được chi phí nằm trong lõi mô hình hay ở các tác vụ bao quanh.

### Slide 17. CH5: Phân rã chu trình Huấn luyện & Bùng nổ VRAM
- **Bố cục trực quan:** Biểu đồ thanh ngang tỷ số Backward/Forward và Biểu đồ cột chồng VRAM so sánh Training vs Suy luận.
- **Nội dung hiển thị:**
  - *Tỷ số thời gian Backward / Forward:* Đối chiếu số đo thực tế với lý thuyết giải tích lan truyền ngược ($2\times FLOPs$) và phân tích tối ưu cuDNN cho CNN `[Điền tỷ số Bwd/Fwd]`.
  - *Bùng nổ bộ nhớ:* VRAM tăng vọt nhiều lần khi chuyển sang huấn luyện `[Điền tỷ lệ tăng VRAM]`.
  - *Nguyên nhân kỹ thuật:* Cơ chế đạo hàm tự động bắt buộc lưu giữ toàn bộ các bản đồ kích hoạt trung gian (Activation maps) trên bộ nhớ (Zhu et al., IISWC 2018).
- **Thông điệp chốt:** Định lượng nguyên nhân gây sập bộ nhớ (OOM) khi huấn luyện dù mô hình suy luận hoàn toàn bình thường.

### Slide 18. CH6: Điểm nghẽn ngoài mô hình & Định luật Amdahl
- **Bố cục trực quan:** Biểu đồ Donut phân bổ các khâu của Pipeline (YOLOv8) kết hợp Biểu đồ thanh so sánh tương quan tỷ trọng trên CPU vs GPU.
- **Nội dung hiển thị:**
  - *Xác nhận điểm nghẽn ngoài mô hình:* Trên GPU, tổng thời gian của tiền xử lý ảnh và hậu xử lý NMS chiếm tỷ trọng áp đảo `[Điền % ngoài mô hình trên T4]`.
  - *Nghịch lý Amdahl:* Trên CPU, mô hình chạy chậm che lấp tiền xử lý. Khi đưa lên GPU, mô hình được tăng tốc mạnh khiến khâu tiền xử lý ảnh trên CPU lộ ra thành nút thắt cổ chai lớn nhất.
  - *Đối chứng học thuật:* Khớp với luận điểm của nghiên cứu AI Tax (Richins et al., HPCA 2020: Phần lớn thời gian ứng dụng thật nằm ngoài AI).
- **Thông điệp chốt:** Tối ưu hóa mô hình AI trên GPU đã chạm trần; trọng tâm cải tiến phải chuyển sang khâu xử lý dữ liệu đầu vào.

### Slide 19. Động cơ Chẩn đoán tự động (19 Luật v2.0) & So sánh A/B
- **Bố cục trực quan:** Chia 2 nửa: Nửa trái hiển thị Danh mục luật chẩn đoán heuristic (PASS / WARNING / CRITICAL); Nửa phải hiển thị Đồ thị phân phối kiểm định A/B Testing.
- **Nội dung hiển thị:**
  - *19 Luật Heuristic v2.0:* Tự động quét 4 nhóm bất thường (Tài nguyên RA, Độ trễ RB, Dữ liệu RC, Cấu trúc RD); chuẩn hóa đơn vị ms/GFLOP và phân định Process RSS vs Tensor Memory.
  - *Kiểm định thống kê A/B:* Đánh giá cải tiến dựa trên tổng thời gian chu trình; sử dụng kiểm định phi tham số **Mann-Whitney U** ($p\text{-value}$) và thước đo **Cliff's Delta** để khẳng định ý nghĩa khoa học.
- **Thông điệp chốt:** Tự động hóa quá trình phát hiện điểm nghẽn và cung cấp khuyến nghị kỹ thuật có cơ sở toán học tin cậy.

### Slide 20. Video Walkthrough Dashboard (90 giây)
- **Bố cục trực quan:** Khung phát video 16:9 chất lượng cao nhúng giữa slide kèm 4 thẻ định danh 4 tab tính năng trọng tâm.
- **Nội dung thuyết minh theo Video:**
  - *Phân đoạn 1 (Tab Layer):* Trực quan hóa tỷ trọng Layer và Biểu đồ tương tác Roofline trên GPU.
  - *Phân đoạn 2 (Tab Model):* Đối đầu hiệu năng các mô hình và công cụ khảo sát Batch scaling.
  - *Phân đoạn 3 (Tab Pipeline):* Biểu đồ Donut bóc tách chi phí tiền/hậu xử lý của Pipeline ứng dụng.
  - *Phân đoạn 4 (Tab Chẩn đoán & A/B):* Danh mục vi phạm luật tự động và bảng so sánh A/B.
  - *(Chế độ phòng ngự: Streamlit Dashboard localhost:8501 chạy sẵn sàng phục vụ kiểm tra trực tiếp trong Q&A).*
- **Thông điệp chốt:** Thể hiện sản phẩm phần mềm hoàn thiện, vận hành trơn tru và trực quan hóa dữ liệu mạnh mẽ.

### Slide 21. Bảng tổng kết giải đáp 6 Câu hỏi nghiên cứu (CH1–CH6)
- **Bố cục trực quan:** Bảng ma trận 6 dòng tương ứng 6 câu hỏi, gồm 3 cột: Mã câu hỏi, Luận điểm khẳng định và Cơ chế kỹ thuật chứng minh.
- **Nội dung hiển thị:**
  - **CH1 (Layer):** Tỷ trọng tập trung ở Conv (CNN) và Linear/Attention (Transformer); batch 1 chưa khai thác hết trần Roofline `[Điền tóm tắt %]`.
  - **CH2 (Kiểm chứng):** Đạt độ chính xác cao so với `torch.profiler`; làm rõ ngưỡng sàn event `[Điền tóm tắt sai số]`.
  - **CH3 (Model):** Tham số không quyết định tốc độ; Memory-bound trên CPU; đảo chiều thứ hạng trên GPU `[Điền tóm tắt trễ]`.
  - **CH4 (Scaling):** Batch bão hòa sớm; Attention trễ bậc hai theo độ dài chuỗi; FP16 tăng tốc và giảm VRAM `[Điền tóm tắt tăng tốc]`.
  - **CH5 (Training):** Tỷ số Backward/Forward khớp lý thuyết; VRAM bùng nổ do lưu trữ Activation maps `[Điền tóm tắt VRAM]`.
  - **CH6 (Pipeline):** Lên GPU bộc lộ chi phí ngoài mô hình chiếm tỷ trọng chi phối theo Định luật Amdahl `[Điền tóm tắt % ngoài model]`.
- **Thông điệp chốt:** Khẳng định đề tài đã trả lời trọn vẹn và thuyết phục toàn bộ 6 câu hỏi nghiên cứu ban đầu.

### Slide 22. Hạn chế, Hướng phát triển & Mời Q&A
- **Bố cục trực quan:** Cột trái trình bày Hạn chế & Hướng phát triển; Cột phải hiển thị Mã QR dẫn tới GitHub Repository và Lời cảm ơn Hội đồng.
- **Nội dung hiển thị:**
  - *Hạn chế hiện tại:* Chưa phân tích sâu vào kernel GPU bằng CUPTI/hardware counters; giới hạn khảo sát trên đơn GPU; áp dụng cho PyTorch Eager mode.
  - *Hướng phát triển:* Mở rộng sang đồ thị tĩnh `torch.compile`, hỗ trợ profiling phân tán đa GPU (DDP).
  - *Lời kết:* Chân thành cảm ơn Quý Thầy/Cô và các bạn; Nhóm sẵn sàng giải đáp câu hỏi và thực hiện live demo theo yêu cầu.
- **Thông điệp chốt:** Kết thúc bài trình bày với phong thái tự tin, khiêm tốn học thuật và làm chủ hoàn toàn công nghệ.

---

## 🎯 Bộ câu hỏi phản biện dự kiến (Defense Prep - Trả lời theo cơ chế)

- 🟧 **TV1 (Mức Layer):**
  - *Câu hỏi:* Vì sao overhead đo lường trên GPU ở batch 1 lớn mà nhóm vẫn khẳng định số liệu là đáng tin cậy?
  - *Trả lời dựa trên cơ chế:* Sai lệch thời gian đo độc lập của từng layer so với `torch.profiler` vẫn nằm trong giới hạn kiểm soát ($<8\%$); bản chất overhead nằm ở chi phí ghi event giữa các layer, hoàn toàn không làm chậm thời gian thực thi của kernel GPU. Vì vậy nhóm chỉ dùng Mức Layer để xác định tỷ trọng %, còn độ trễ tổng thể sạch được đo riêng ở Mức Model.
- 🟦 **TV2 (Mức Model):**
  - *Câu hỏi:* Tại sao MobileNetV3 được thiết kế cho thiết bị nhẹ mà lại chạy chậm hơn ResNet-50 trên CPU Laptop?
  - *Trả lời dựa trên cơ chế:* MobileNetV3 dùng Depthwise Separable Conv có cường độ số học (FLOP/Byte) rất thấp. Phép toán này bị giới hạn bởi băng thông truy xuất bộ nhớ RAM (Memory-Bound) thay vì năng lực tính toán của CPU. Khi bộ nhớ đệm cache CPU không đủ lớn hoặc trên kiến trúc ARM kernel chưa tối ưu sâu, việc đọc/ghi liên tục gây nghẽn cổ chai.
- 🟩 **TV3 (Mức Giai đoạn):**
  - *Câu hỏi:* Nếu khâu tiền xử lý của Pipeline chiếm tỷ trọng lớn trên GPU, giải pháp khắc phục triệt để trong sản xuất là gì?
  - *Trả lời dựa trên cơ chế:* Cần chuyển toàn bộ quy trình tiền xử lý ảnh (giải mã, resize, chuẩn hóa) lên thực thi song song trực tiếp trên GPU (sử dụng NVIDIA DALI hoặc Torchvision GPU tensors), triệt tiêu chi phí truyền dữ liệu và xử lý tuần tự trên CPU.

---

## ✅ Checklist chuẩn bị thuyết trình

- [ ] **TV2:** Cập nhật các bảng số liệu thực tế vào các vị trí `[Điền số đo...]` sau khi hoàn tất lượt đo chính thức.
- [ ] **TV3:** Xuất video clip 90 giây quay màn hình Dashboard Streamlit để nhúng trực tiếp vào Slide 20.
- [ ] **Cả nhóm:** Đồng bộ slide theo mã màu nhận diện (🟩 TV3 · 🟧 TV1 · 🟦 TV2) và tổng duyệt khớp đúng thời lượng 22–24 phút.
