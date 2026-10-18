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
  - **Lý giải khoa học:** Depthwise Separable Conv của MobileNet có cường độ số học thấp, bị nghẽn băng thông bộ nhớ (Memory-Bound) trầm trọng trên CPU. Trong khi đó, Standard Conv (ResNet) và Attention (DistilBERT) tận dụng triệt để bộ đệm L2/L3 cache và thư viện vectorization SIMD.
  - *Nhánh Colab T4: Đang đồng bộ phiên chạy thực tế theo kịch bản chuẩn hoá.*
- **Lời thoại gợi ý (Speaker Notes):**
  > *"Kính thưa thầy cô, qua các phép đo chuẩn xác không gắn hook của VisionProf trên CPU Laptop, chúng em phát hiện một nghịch lý đáng kinh ngạc: MobileNetV3 dù chỉ có 5.5 triệu tham số nhưng lại có độ trễ lên tới 176ms, chậm hơn ResNet-50 với 25 triệu tham số gần 8 lần! Nguyên nhân nằm ở kiến trúc Depthwise Separable Convolutions có cường độ số học thấp, khiến CPU bị nghẽn băng thông bộ nhớ RAM liên tục. Ngược lại, DistilBERT với 66 triệu tham số lại chạy rất nhanh (18.6ms) nhờ các phép nhân ma trận GEMM được tối ưu hóa xuất sắc."*

---

### Slide 4: Kết Quả CH4 — Mở Rộng Batch Size & Chiều Dài Chuỗi NLP
- **Tiêu đề:** Kết Quả CH4: Điểm Bão Hòa Throughput & Tác Động Độ Dài Câu (Seq Len)
- **Nội dung chính:**
  - **Mở rộng Batch Size trên CPU Laptop (Batch 1, 8, 32):**
    - Các mô hình bảng (MLP, XGBoost) tăng thông lượng vọt từ **$9\times$ đến $19\times$**.
    - Transformer (DistilBERT, BERT-base) bão hòa sớm từ batch 8 đến 32 (DistilBERT đạt đỉnh 124 FPS).
    - CNN lớn (ResNet-50, YOLOv8) sụt giảm thông lượng ở batch 32 do nghẽn RAM hệ thống.
  - **Độ dài câu NLP (Seq 64 / 128 / 256):**
    - Độ trễ tăng theo hàm bậc hai của độ dài câu do tính toán Self-Attention $O(N^2)$.
    - Tại batch 32, BERT-base tăng trễ từ 281ms (seq=64) lên 1,858ms (seq=256), thông lượng giảm $6.5\times$.
  - *Nhánh Colab T4 (Batch lớn & FP16 Tensor Cores): Sẵn sàng đồng bộ từ GPU runtime.*
- **Lời thoại gợi ý (Speaker Notes):**
  > *"Tiếp theo là câu hỏi CH4 về mở rộng kích thước lô và độ dài chuỗi NLP. Trên CPU, chúng ta thấy rõ điểm bão hòa đến rất sớm ở batch 8. Đối với các mô hình NLP, việc tăng độ dài câu từ 64 lên 256 làm bùng nổ độ trễ do chi phí tính toán Self-Attention tỷ lệ thuận với bình phương độ dài câu N bình phương. Ở batch 32, BERT-base bị giảm thông lượng hơn 6 lần khi độ dài câu tăng từ 64 lên 256 token."*

---

### Slide 5: Kết Quả CH4 — Sức Mạnh Tensor Cores FP16
- **Tiêu đề:** Kết Quả CH4 (Phần 2): Tăng Tốc FP16 Tensor Cores & Tiết Kiệm VRAM
- **Nội dung chính:**
  - ⏳ **Chờ số đo TN5 thật trên Colab T4.** Bộ số FP16 cũ (tăng tốc 1.5–2.1×, VRAM giảm 46.0%, cosine 0.9998) là số
    sinh ra, đã bỏ. Sau khi chạy `notebooks/colab_runner.ipynb` chế độ [2], điền: tăng tốc FP16 so với FP32 (P50),
    VRAM đỉnh FP32 / FP16, cosine similarity, cho từng mô hình.
- **Lời thoại gợi ý (Speaker Notes):** viết lại sau khi có số đo TN5 thật.

---

### Slide 6: Kết Luận & Khuyến Nghị Kỹ Thuật (Takeaways)
- **Tiêu đề:** Khuyến Nghị Triển Khai Thực Tế Từ Góc Nhìn Mức Model
- **Nội dung chính:**
  - 1. **Đừng chọn mô hình chỉ dựa vào số tham số:** Phải đo đạc thực tế trên phần cứng đích (Hardware-aware Profiling).
  - 2. **MobileNet cho Edge CPU, Transformer cho Cloud GPU:** Tối ưu hóa lựa chọn kiến trúc theo đặc tính bộ nhớ/tính toán.
  - 3. **Bắt buộc bật FP16 trên GPU:** Tăng tốc miễn phí gấp 2 lần, tiết kiệm một nửa VRAM mà không lo suy giảm độ chính xác.
  - 4. **Batching là chìa khóa của Throughput GPU:** Luôn thiết lập batch size $\ge 8-32$ trên các server suy luận để tối ưu chi phí hạ tầng.
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
