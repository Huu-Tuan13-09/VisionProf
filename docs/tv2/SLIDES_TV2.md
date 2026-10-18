# BỘ SLIDE THUYẾT TRÌNH & KỊCH BẢN DEMO — THÀNH VIÊN 2 (TV2)
## Mức Model: Đánh Giá Toàn Bộ Mô Hình & Đối Đầu Hiệu Năng CPU vs GPU (CH3 & CH4)

---

## 📑 PHẦN 1: NỘI DUNG SLIDE & LỜI THOẠI (SPEAKER NOTES)

### Slide 1: Đặt Vấn Đề & Nghịch Lý Tham Số
- **Tiêu đề:** 8 Mô Hình Trên CPU và GPU: Mô Hình Nào Nhanh Nhất, và Vì Sao Thứ Hạng Thay Đổi?
- **Nội dung chính:**
  - Định kiến phổ biến: *"Mô hình càng ít tham số thì chạy càng nhanh; GPU luôn tăng tốc đều cho mọi mô hình"*.
  - Thực tế thực nghiệm: **MobileNetV3 (5.5M)** chạy chậm hơn **DistilBERT (66.4M)** trên GPU!
  - Câu hỏi nghiên cứu:
    - **CH3:** Số lượng tham số có quyết định tốc độ? Chuyển sang GPU thì thứ hạng có đảo không?
    - **CH4:** Tăng Batch Size và dùng FP16 Tensor Cores thì tốc độ tăng bao nhiêu?
- **Lời thoại gợi ý (Speaker Notes):**
  > *"Kính thưa thầy cô và các bạn, khi lựa chọn mô hình AI để triển khai, các kỹ sư thường tin rằng mô hình càng ít tham số sẽ chạy càng nhanh, và nếu có GPU thì mô hình nào cũng sẽ được tăng tốc như nhau. Tuy nhiên, qua hệ thống VisionProf ở mức Model, chúng em đã phát hiện ra một nghịch lý thú vị: trên GPU Tesla T4, DistilBERT với 66 triệu tham số lại chạy nhanh hơn MobileNetV3 chỉ có 5.5 triệu tham số! Hôm nay, em xin đại diện nhóm trình bày câu trả lời khoa học cho hiện tượng đảo thứ hạng này cùng các phát hiện về tối ưu phần cứng."*

---

### Slide 2: Kiến Trúc Mức Model & Giải Pháp Kỹ Thuật (TV2)
- **Tiêu đề:** Thiết Kế Mức Model: Bộ Đo Benchmark & Giám Sát Khoa Học
- **Nội dung chính:**
  - **Model Zoo Chuẩn 8 Mô hình:** CV (MobileNetV3, ResNet-50, ViT, YOLOv8), NLP (BERT, DistilBERT), Tabular (MLP, XGBoost).
  - **Tách 2 pha độc lập:** Pha 1 đo latency sạch (không gắn hook) chống phồng trễ -> Pha 2 mới gắn profiler.
  - **Bắt đỉnh Peak VRAM chuẩn xác:** Trực tiếp từ `torch.cuda.max_memory_allocated()` của GPU allocator thay vì chỉ dựa vào polling 100ms.
  - **Thu thập đủ 12 chỉ số khoa học:** P50, P90, P95, P99, Throughput FPS, Tail Ratio, RSS RAM, Peak VRAM, Params, Cold-start.
- **Lời thoại gợi ý (Speaker Notes):**
  > *"Để đảm bảo tính chuẩn xác khoa học, ở mức Model, chúng em áp dụng nguyên tắc đo sạch 2 pha: tách biệt hoàn toàn vòng lặp đo latency khỏi các profiler hook nhằm tránh làm phồng thời gian đo từ 100% đến 160% trên GPU. Đồng thời, chúng em thu thập đủ 12 chỉ số, đặc biệt là trung vị P50 và trễ đuôi P99 thay vì chỉ lấy trung bình đơn giản, và bắt đỉnh VRAM chính xác đến từng kilobyte trực tiếp từ phần cứng."*

---

### Slide 3: Kết Quả CH3 — Nghịch Lý Tham Số & Hiện Tượng Thắt Cổ Chai Bộ Nhớ
- **Tiêu đề:** Kết Quả CH3: Nghịch Lý Tham Số & Bản Chất Phần Cứng
- **Nội dung chính:**
  - **Phát hiện đo thật trên CPU Laptop:**
    - MobileNetV3-Large (5.5M tham số) trễ tới **176.3 ms** (5.7 FPS), chậm nhất nhóm Deep Learning.
    - ResNet-50 (25.6M tham số) đạt **22.5 ms** (44.2 FPS) — nhanh hơn MobileNet gần **$8\times$** dù tham số gấp 5 lần!
    - DistilBERT (66.4M tham số) đạt **18.6 ms** (53.3 FPS) — nhanh thứ 3 toàn đoàn (chỉ sau MLP và XGBoost).
  - **Phát hiện đo thật trên Google Colab T4 (Batch 1, FP32):**
    - ResNet-50 (8.2 GFLOPs) và MobileNetV3 (0.43 GFLOPs, ít hơn 19 lần) **cùng mất ~6.0 ms**!
    - MLP trên T4 mất 0.28 ms, chậm hơn trên Mac (0.04 ms) tới **$7\times$** do chi phí PCIe và khởi tạo CUDA.
  - **Lý giải khoa học:** Ở batch 1 trên GPU, thời gian bị chi phối bởi chi phí gửi lệnh kernel (Kernel Launch Latency) chứ không phải FLOPs. Trên CPU ARM Mac, MobileNetV3 chậm vì thiếu vectorization cho Depthwise Conv (trên x86 chỉ ~12 ms).
- **Lời thoại gợi ý (Speaker Notes):**
  > *"Kính thưa thầy cô, qua các phép đo chuẩn xác không gắn hook của VisionProf, chúng em phát hiện nghịch lý: trên CPU Mac, MobileNetV3 dù chỉ 5.5 triệu tham số nhưng mất tới 176ms do thiếu vectorization trên ARM. Đặc biệt, khi lên GPU Tesla T4 ở batch 1, MobileNetV3 có FLOPs ít hơn ResNet-50 tới 19 lần nhưng cả hai đều cùng mất khoảng 6.0ms! Nguyên nhân là GPU bị nghẽn bởi chi phí gửi kernel tuần tự từ CPU sang GPU. Đồng thời, mô hình siêu nhẹ như MLP chạy trên GPU còn chậm hơn CPU tới 7 lần do chi phí truyền bus PCIe."*

---

### Slide 4: Kết Quả CH4 — Mở Rộng Batch Size & Chiều Dài Chuỗi NLP
- **Tiêu đề:** Kết Quả CH4: Điểm Bão Hòa Throughput & Tác Động Độ Dài Câu (Seq Len)
- **Nội dung chính:**
  - **Mở rộng Batch Size trên Colab Tesla T4 (Batch 1 $\rightarrow$ 32):**
    - MobileNetV3 tăng throughput vượt bậc **$8.8\times$** (từ 170 lên 1,499 FPS) do batch 1 chưa lấp đầy GPU.
    - ResNet-50 tăng $2.2\times$ (153 $\rightarrow$ 337 FPS), YOLOv8n tăng $2.0\times$ (118 $\rightarrow$ 235 FPS).
    - Transformer nặng (ViT-B/16) bão hòa sớm từ batch 8 (75 $\rightarrow$ 96 $\rightarrow$ 94 FPS).
  - **Độ dài câu NLP (Seq 64 / 128 / 256):**
    - Tại batch 32 trên T4, BERT-base mất 127ms (seq 64) $\rightarrow$ 228ms (128) $\rightarrow$ 507ms (256): mỗi lần gấp đôi độ dài, trễ tăng $1.8\times$ rồi $2.2\times$ do tính toán Self-Attention $O(N^2)$.
- **Lời thoại gợi ý (Speaker Notes):**
  > *"Tiếp theo là câu hỏi CH4 về mở rộng kích thước lô và độ dài chuỗi NLP. Trên GPU T4, mô hình nhẹ hưởng lợi nhiều nhất khi tăng batch: MobileNetV3 tăng thông lượng tới 8.8 lần nhờ lấp đầy 40 SMs của T4; trong khi ViT bão hòa ngay từ batch 8. Đối với mô hình ngôn ngữ BERT, độ trễ tăng theo hàm bậc hai của chiều dài câu N bình phương: tăng từ 64 lên 256 token khiến độ trễ tại batch 32 tăng vọt từ 127ms lên 507ms."*

---

### Slide 5: Kết Quả CH4 — Sức Mạnh Tensor Cores FP16
- **Tiêu đề:** Kết Quả CH4 (Phần 2): Tăng Tốc FP16 Tensor Cores & Tiết Kiệm VRAM
- **Nội dung chính:**
  - **Bảng số liệu đo thật trên Colab T4 (Batch 1):**
    - **ViT-B/16:** Trễ giảm từ 12.99ms $\rightarrow$ 4.62ms (**Tăng tốc $2.81\times$**, VRAM giảm 45%).
    - **DistilBERT:** 5.34ms $\rightarrow$ 3.44ms (**Tăng tốc $1.55\times$**, VRAM giảm 45%).
    - **BERT-base:** 8.58ms $\rightarrow$ 6.94ms (**Tăng tốc $1.23\times$**, VRAM giảm 46%).
    - **CNNs (ResNet-50 0.87×, MobileNetV3 0.85×, YOLOv8n 0.77×):** Chậm hơn khi bật FP16 ở batch 1!
  - **Luận giải cơ chế:**
    - FP16 **luôn giảm 38% đến 49% VRAM** cho toàn bộ mô hình có tham số lớn.
    - FP16 **chỉ tăng tốc Transformer** (tận dụng Tensor Cores tính toán GEMM lớn). Với CNN ở batch 1, tensor quá nhỏ khiến overhead chuyển kiểu và kernel launch làm tăng độ trễ.
- **Lời thoại gợi ý (Speaker Notes):**
  > *"Tại phần 2 của CH4, chúng em khảo sát số thực nửa chính xác FP16 trên Turing Tensor Cores của T4. Kết quả đo được: FP16 luôn cắt giảm từ 38% đến 49% VRAM cho mọi mô hình. Tuy nhiên, về tốc độ, FP16 chỉ tăng tốc vượt trội cho các mô hình Transformer — điển hình là ViT-B/16 tăng tốc tới 2.81 lần! Ngược lại, với CNN ở batch 1, FP16 lại chậm hơn từ 13% đến 23% vì khối tính toán quá nhỏ không đủ bù đắp chi phí chuyển kiểu dữ liệu. Muốn tận dụng FP16 cho CNN, bắt buộc phải tăng kích thước lô."*

---

### Slide 6: Kết Luận & Khuyến Nghị Kỹ Thuật (Takeaways)
- **Tiêu đề:** Khuyến Nghị Triển Khai Thực Tế Từ Góc Nhìn Mức Model
- **Nội dung chính:**
  - 1. **Đừng chọn mô hình chỉ dựa vào số tham số/FLOPs:** Phải đo đạc thực tế trên phần cứng đích (Hardware-aware Profiling).
  - 2. **MobileNet cho Edge CPU, Transformer cho Cloud GPU:** Tối ưu hóa lựa chọn kiến trúc theo đặc tính bộ nhớ và năng lực tính toán.
  - 3. **Khai thác FP16 đúng cách:** Luôn bật FP16 để tiết kiệm 50% VRAM; nhưng với CNN cần kết hợp Batch Size $\ge 8$ để Tensor Cores phát huy tác dụng.
  - 4. **Batching là chìa khóa của Throughput GPU:** Đưa batch size lên 8–32 giúp mô hình nhẹ tăng thông lượng tới gần $9\times$.
- **Lời thoại gợi ý (Speaker Notes):**
  > *"Tóm lại, thông điệp cốt lõi từ mức Model của VisionProf là: không có mô hình nào nhanh nhất cho mọi nền tảng. Khi triển khai trên Edge CPU, MobileNet là lựa chọn tối ưu; nhưng trên Cloud GPU, các mô hình Transformer lại phát huy sức mạnh vượt trội. Và trên GPU, hãy luôn bật FP16 cùng cơ chế dynamic batching để khai thác tối đa năng lực phần cứng. Em xin cảm ơn thầy cô và các bạn đã lắng nghe!"*

---

## 🎬 PHẦN 2: KỊCH BẢN DEMO (1–2 PHÚT)

### Mục tiêu Demo
Minh họa trực quan khả năng đo lường qua CLI và giao diện tương tác Streamlit Dashboard giải thích sinh động hai câu hỏi CH3 và CH4.

### Bước 1: Demo Chạy Đo Lường Qua CLI (30 giây)
- **Thao tác:** Mở terminal, chạy lệnh:
  ```bash
  python scripts/run_experiment.py --model resnet50 --device cpu_laptop --batch-size 1 --precision fp32 --experiment TN3 --profilers layer
  ```
- **Lời dẫn (Voiceover):**
  > *"Trước tiên, em xin demo công cụ dòng lệnh của TV2. Chỉ với một câu lệnh đơn giản, hệ thống tự động khởi tạo model adapter, thực hiện warm-up, đo lường 50 vòng sạch để lấy P50 và bắt đỉnh VRAM từ GPU allocator, sau đó mới kích hoạt layer profiler. Kết quả được lưu chuẩn hóa tại thư mục results với đầy đủ 12 metrics."*

### Bước 2: Demo Dashboard Tab 1 — Tổng Quan Mô Hình (30 giây)
- **Thao tác:** Chuyển sang trình duyệt tại `http://localhost:8501`, chọn **Tab 1: Model Overview**.
- **Tương tác:** Chọn bộ lọc Device (`cpu_laptop`, `colab_t4`), trỏ chuột vào biểu đồ Scatter Log-scale *Params vs P50 Latency*.
- **Lời dẫn (Voiceover):**
  > *"Trên Dashboard Tab 1 của TV2, người dùng có thể lọc theo thiết bị và nhóm mô hình. Biểu đồ Scatter này thể hiện rõ đánh đổi giữa số tham số và độ trễ. Chúng ta có thể thấy rõ các điểm mô hình Transformer nằm ở vùng độ trễ thấp trên GPU dù có lượng tham số lớn."*

### Bước 3: Demo Dashboard Tab 2 — So Sánh CPU vs GPU (40 giây)
- **Thao tác:** Bấm sang **Tab 2: CPU vs GPU Acceleration**.
- **Tương tác:** Cuộn qua biểu đồ Bar Chart Speedup, bảng Đảo thứ hạng, và bảng FP16 Tensor Cores.
- **Lời dẫn (Voiceover):**
  > *"Chuyển sang Tab 2, đây là nơi trả lời trực diện hai câu hỏi CH3 và CH4. Biểu đồ cột thể hiện mức tăng tốc: Transformer đạt trên 9 lần, trong khi MobileNet chỉ đạt chưa tới 2 lần. Phía dưới là bảng FP16 minh chứng rõ mức tiết kiệm 46% VRAM và độ tương đồng Cosine 0.9998 hoàn hảo."*
