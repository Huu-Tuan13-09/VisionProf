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

### Slide 3: Kết Quả CH3 — Hiện Tượng Đảo Thứ Hạng (Rank Inversion)
- **Tiêu đề:** Kết Quả CH3: Nghịch Lý Tham Số & Hiện Tượng Đảo Thứ Hạng
- **Nội dung chính:**
  - Bảng so sánh Speedup:
    - **Hưởng lợi lớn nhất (8.7x – 9.3x):** BERT-base (9.3x), DistilBERT (8.8x), ViT-B/16 (8.7x).
    - **Hưởng lợi ít nhất (1.9x – 2.0x):** MobileNetV3-Large (1.95x), MLP (2.0x).
  - Hiện tượng đảo thứ hạng:
    - Trên CPU: MobileNetV3 đứng #3 (8.5ms), DistilBERT đứng #6 (31.2ms).
    - Trên GPU T4: DistilBERT vươn lên #3 (3.56ms), đẩy MobileNetV3 xuống #4 (4.38ms)!
- **Lời thoại gợi ý (Speaker Notes):**
  > *"Trên slide là biểu đồ minh chứng trực tiếp cho hiện tượng đảo thứ hạng giữa CPU và GPU. Trên CPU, MobileNetV3 là ngôi sao sáng với độ trễ chỉ 8.5ms, nhanh gấp gần 4 lần DistilBERT. Nhưng khi đưa lên GPU Tesla T4, DistilBERT tăng tốc tới 8.8 lần và vượt mặt MobileNetV3. Lý do là vì MobileNet sử dụng Depthwise Separable Convolutions có cường độ số học thấp, bị nghẽn băng thông bộ nhớ (Memory-Bound). Trong khi đó, các phép tính ma trận Self-Attention của Transformer lại là Compute-Bound, khớp hoàn hảo với hàng nghìn nhân tính toán song song của GPU."*

---

### Slide 4: Kết Quả CH4 — Mở Rộng Batch Size & Điểm Bão Hòa
- **Tiêu đề:** Kết Quả CH4 (Phần 1): Đường Cong Thông Lượng Theo Batch Size
- **Nội dung chính:**
  - Thử nghiệm trên Batch = [1, 8, 32]:
    - **GPU Tesla T4:** Thông lượng tăng từ **6.5x đến 7.5x** khi tăng batch từ 1 lên 32 (ResNet-50 tăng từ 177 FPS lên 1,320 FPS).
    - **CPU Laptop:** Thông lượng chỉ tăng từ **3.2x đến 3.5x** và nhanh chóng chạm trần bão hòa ở batch 8.
  - Kết luận: GPU cần batch lớn để lấp đầy tài nguyên tính toán (Amortized Overhead); CPU không được hưởng lợi nhiều do nghẽn băng thông RAM hệ thống.
- **Lời thoại gợi ý (Speaker Notes):**
  > *"Tiếp theo là câu hỏi CH4 về mở rộng kích thước lô. Trên GPU, chạy ở Batch 1 là một sự lãng phí tài nguyên lớn vì GPU bị 'đói dữ liệu'. Khi tăng batch lên 32, thông lượng của ResNet-50 nhảy vọt gấp 7.5 lần, đạt hơn 1.300 ảnh/giây. Trong khi đó, trên CPU, việc tăng batch chỉ giúp tăng tốc khoảng 3 lần rồi bão hòa ngay do giới hạn băng thông bộ nhớ RAM của máy chủ."*

---

### Slide 5: Kết Quả CH4 — Sức Mạnh Tensor Cores FP16
- **Tiêu đề:** Kết Quả CH4 (Phần 2): Tăng Tốc FP16 Tensor Cores & Tiết Kiệm VRAM
- **Nội dung chính:**
  - **Tăng tốc độ trễ:** Trung bình giảm trễ từ **1.5x đến 2.1x** (ResNet-50 tăng tốc 2.11x, BERT-base tăng tốc 2.07x).
  - **Tiết kiệm bộ nhớ VRAM:** Giảm đỉnh bộ nhớ **46.0%** trên toàn bộ các mô hình.
  - **Độ chính xác khoa học:** Cosine Similarity giữa đầu ra FP32 và FP16 đạt **0.9998** (> 0.999).
- **Lời thoại gợi ý (Speaker Notes):**
  > *"Điểm sáng tiếp theo của thực nghiệm CH4 là sức mạnh của nhân chuyên dụng Tensor Cores trên GPU Tesla T4 khi chạy ở độ chính xác nửa FP16. Không chỉ giúp mô hình chạy nhanh gấp đôi (2.1x), FP16 còn cắt giảm tới 46% dung lượng bộ nhớ VRAM, trong khi đầu ra dự đoán giữ nguyên độ tương đồng cosine tới 0.9998 so với FP32 gốc."*

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
