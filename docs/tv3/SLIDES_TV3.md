# BỘ SLIDE THUYẾT TRÌNH & KỊCH BẢN DEMO — THÀNH VIÊN 3 (TV3)
## Mức Giai Đoạn: Phân Rã Huấn Luyện, Pipeline Thực Tế & Chẩn Đoán Điểm Nghẽn Tự Động (CH5 & CH6)

---

## 📑 NỘI DUNG SLIDE & LỜI THOẠI THUYẾT TRÌNH (SPEAKER NOTES)

### Slide 1: Đặt Vấn Đề — Training & Pipeline: Điểm Nghẽn Thật Sự Nằm Ở Đâu?
- **Tiêu đề:** Phân Rã Giai Đoạn Huấn Luyện & Quy Trình Thực Tế (CH5 & CH6)
- **Nội dung trên slide:**
  - Định vị nghiên cứu: **Evaluation** (đúng/sai) vs **Benchmarking** (nhanh/chậm) vs **Profiling** (tại sao chậm, chậm ở khâu nào?).
  - Hai câu hỏi lớn trong triển khai thực tế:
    - **CH5 (Training):** Lan truyền ngược (Backward) tốn thời gian gấp mấy lần Forward? VRAM bùng nổ bao nhiêu lần? Nạp dữ liệu có làm nghẽn GPU?
    - **CH6 (Pipeline):** Trong ứng dụng thật, các công đoạn ngoài mô hình (tiền xử lý, NMS, tokenize) chiếm bao nhiêu %? Có vượt quá 30%?
  - Trọng tâm phụ trách của TV3: Mức Giai đoạn (`src/phase/`), Bộ 19 luật chẩn đoán v2.0 (`src/analyzer/`), và Dashboard điều phối (`src/dashboard/`).
- **Lời thoại gợi ý (Speaker Notes):**
  > *"Kính thưa quý thầy cô và các bạn! Sau khi TV1 đã mổ xẻ chi tiết từng layer và TV2 đánh giá toàn bộ mô hình khi suy luận, phần trình bày của em — Thành viên 3 — sẽ đưa chúng ta đến bức tranh toàn cảnh của hệ thống AI trong thực tế: Đó là chu trình Huấn luyện (Training) và Quy trình ứng dụng hoàn chỉnh (End-to-End Pipeline). Trong sản xuất, một mô hình có thể suy luận cực nhanh trong phòng thí nghiệm nhưng lại trở nên chậm chạp khi đưa vào ứng dụng vì bị nghẽn ở khâu đọc ảnh hoặc giải mã kết quả. Em sẽ lần lượt giải đáp định lượng hai câu hỏi CH5 và CH6, đồng thời giới thiệu Động cơ chẩn đoán tự động của VisionProf."*

---

### Slide 2: Kiến Trúc Mức Giai Đoạn & Hệ Thống Chẩn Đoán Heuristic v2.0
- **Tiêu đề:** Thiết Kế Kỹ Thuật: Context Manager, I/O Sentinel & 19 Luật Heuristic
- **Nội dung trên slide:**
  - **`PhaseTimer`**: Sử dụng Context Manager Pattern (`with timer.phase("name"):`), đồng bộ `torch.cuda.synchronize()` 2 đầu đo GPU chuẩn xác.
  - **`DataLoaderSentinel`**: Bọc `DataLoader`, đo thời gian I/O và tự động loại bỏ nhiễu khởi động worker ở batch 0.
  - **Hệ thống 19 Luật Chẩn đoán Heuristic v2.0**:
    - **Nhóm RA (5 luật):** Tài nguyên phần cứng (OOM risk, rò rỉ RAM).
    - **Nhóm RB (5 luật):** Độ trễ & ổn định (Tail latency, tỷ lệ Bwd/Fwd, overhead ngoài model).
    - **Nhóm RC (4 luật):** Luồng dữ liệu (DataLoader ratio, GPU Starvation).
    - **Nhóm RD (5 luật):** Cấu trúc đồ thị (Layer chiếm $>30\%$, sub-microsecond).
  - **Sửa 4 lỗi logic v1.0:** Đổi ms/Mparam $\rightarrow$ ms/GFLOP (RB-04, RD-05); Sub-microsecond thay vì "Dead layer" (RD-02); Chẩn đoán GPU Starvation thực tế (RC-03); Tách riêng RAM RSS và Tensor Memory (RA-04).
- **Lời thoại gợi ý (Speaker Notes):**
  > *"Để đo lường các giai đoạn huấn luyện tuần tự mà không can thiệp sâu vào kiến trúc model, em xây dựng PhaseTimer dựa trên Context Manager và cơ chế đồng bộ CUDA Streams chuẩn xác. Đồng thời, công cụ DataLoaderSentinel giúp đo độ trễ I/O thực tế mà không bị sai lệch bởi vòng lặp khởi tạo batch 0. Đặc biệt, bộ 19 luật chẩn đoán heuristic v2.0 đã khắc phục triệt để 4 lỗi logic nghiêm trọng của phiên bản cũ, chuyển đổi sang đơn vị chuẩn mili-giây trên GFLOP và phân định rạch ròi giữa RAM tiến trình với bộ nhớ tensor."*

---

### Slide 3: Kết Quả CH5 — Training Phase Breakdown: Backward & Bùng Nổ VRAM
- **Tiêu đề:** Kết Quả CH5: Tỷ Lệ Backward/Forward & Bùng Nổ Bộ Nhớ Khi Training
- **Hình ảnh minh họa:** `fig_ch5_training_breakdown.png` + `fig_ch5_backward_forward_ratio.png` + `fig_ch5_memory_train_vs_infer.png`
- **Nội dung trên slide:**
  - **Tỷ số Backward / Forward:**
    - **DistilBERT (Transformer):** Đạt **$2.21\times$** (khớp hoàn hảo lý thuyết giải tích Backprop $2\times FLOPs$).
    - **CNNs (ResNet-50 1.82×, MobileNetV3 1.65×):** Thấp hơn 2× nhờ thư viện NVIDIA cuDNN tối ưu hóa kernel tích chập ngược cực mạnh.
  - **Bùng nổ bộ nhớ VRAM (Train vs Infer):**
    - **ResNet-50:** Tăng từ 420 MB $\rightarrow$ 1,139 MB (**tăng $2.71\times$**).
    - **MobileNetV3:** Tăng từ 140 MB $\rightarrow$ 460 MB (**tăng $3.28\times$**).
    - *Nguyên nhân:* Bắt buộc lưu toàn bộ activations trung gian để tính đạo hàm + bộ nhớ 2 trạng thái động lượng ($m_t, v_t$) của AdamW.
  - **DataLoader Starvation:** Khi `num_workers = 0`, GPU bị đói dữ liệu làm hiệu suất sụt giảm nghiêm trọng (kích hoạt cảnh báo RC-03).
- **Lời thoại gợi ý (Speaker Notes):**
  > *"Chuyển sang kết quả thực nghiệm CH5 trên Colab T4. Về thời gian: lý thuyết chỉ ra rằng lan truyền ngược tốn gấp đôi phép tính forward. Số liệu thực tế của chúng em khẳng định điều này: DistilBERT đạt tỷ số 2.21 lần. Ở các mô hình tích chập như ResNet-50 hay MobileNet, tỷ số này rơi vào khoảng 1.65 đến 1.82 lần do cuDNN tối ưu hóa bộ nhớ đệm cho phép tích chập ngược rất tốt. Về bộ nhớ: VRAM khi training tăng vọt gấp 2.7 đến 3.3 lần! Đây là bằng chứng định lượng vì sao một batch size chạy mượt mà khi suy luận lại ngay lập tức gây sập bộ nhớ OOM khi chuyển sang huấn luyện."*

---

### Slide 4: Kết Quả CH6 — Pipeline Breakdown: Nghịch Lý Ngoài Model
- **Tiêu đề:** Kết Quả CH6: Phân Rã Quy Trình Thật & Nghịch Lý Định Luật Amdahl
- **Hình ảnh minh họa:** `fig_ch6_pipeline_breakdown.png` + `fig_ch6_cpu_vs_gpu_pipeline.png`
- **Nội dung trên slide:**
  - **Xác nhận giả thuyết CH6 (Overhead ngoài model $> 30\%$):**
    - **YOLOv8n trên Colab T4:** Tiền xử lý (3.23ms, $30.3\%$) + Hậu xử lý NMS (1.15ms, $10.7\%$) $\rightarrow$ **Tổng chi phí ngoài model = $41.0\%$ ($> 30\%$)**!
    - **BERT-base:** Tokenize chỉ chiếm $13.5\%$ (nhờ HuggingFace Rust tokenizer tối ưu).
  - **Nghịch lý chuyển dịch điểm nghẽn (Amdahl's Law):**
    - **Trên CPU:** Model Forward chiếm **$94.0\%$** (163.6ms), ngoài model chỉ chiếm $6.0\%$ (Compute-Bound).
    - **Lên GPU T4:** Model tăng tốc **$26\times$** (còn 6.29ms) $\rightarrow$ Chi phí ngoài model bộc lộ chiếm **$41.0\%$**!
  - **Bài học kiến trúc:** Tối ưu hóa mô hình AI trên GPU đã chạm trần; điểm nghẽn mới nằm ở khâu tiền xử lý ảnh trên CPU.
- **Lời thoại gợi ý (Speaker Notes):**
  > *"Tại câu hỏi CH6, chúng em khảo sát một quy trình phát hiện vật thể YOLOv8n hoàn chỉnh. Kết quả kiểm chứng rực rỡ: trên GPU T4, tổng thời gian ngoài mô hình chiếm tới 41% — vượt xa ngưỡng kỳ vọng 30%! Nhìn vào biểu đồ so sánh CPU và GPU, chúng ta thấy một hiện tượng cực kỳ thú vị theo Định luật Amdahl: Trên CPU, mô hình chạy rất chậm mất 163ms nên chiếm tới 94%, che lấp hoàn toàn tiền xử lý. Nhưng khi đưa lên GPU T4, mô hình chạy vèo một cái chỉ mất 6ms, khiến cho khâu tiền xử lý resize ảnh và NMS trên CPU bỗng nhiên trở thành nút thắt cổ chai chiếm gần nửa thời gian toàn ứng dụng!"*

---

### Slide 5: Hệ Thống Chẩn Đoán Tự Động & Động Cơ A/B Testing
- **Tiêu đề:** Tự Động Hóa Chẩn Đoán & So Sánh A/B Chuẩn Thống Kê (Tab 7 & Tab 8)
- **Nội dung trên slide:**
  - **Tab 7 (Diagnostic Engine):**
    - Tự động quét và đánh giá 19 luật heuristic theo mức độ nghiêm trọng: `CRITICAL`, `WARNING`, `PASS`.
    - Phát hiện tức thì: I/O Starvation khi nạp dữ liệu, rủi ro OOM bộ nhớ, trễ đuôi bất thường ($P99/P50$).
  - **Tab 8 (A/B Testing Engine):**
    - Tính toán hệ số Speedup chuẩn xác dựa trên tổng thời gian vòng lặp mô hình ($\text{Mean}_A / \text{Mean}_B$).
    - Kiểm định thống kê phi tham số **Mann-Whitney U** ($p\text{-value}$) và thước đo độ lớn ảnh hưởng **Cliff's Delta** để khẳng định sự cải thiện là có ý nghĩa khoa học.
- **Lời thoại gợi ý (Speaker Notes):**
  > *"Để giúp người dùng không bị 'ngợp' trước hàng trăm chỉ số số học, Tab 7 của Dashboard tích hợp Động cơ chẩn đoán tự động. Hệ thống sẽ quét qua 19 luật heuristic và chỉ ra ngay: hệ thống của bạn có bị nghẽn nạp dữ liệu không, có nguy cơ tràn RAM không. Đồng thời, Tab 8 cung cấp công cụ A/B Testing chuẩn mực, tính toán tốc độ tăng tốc và kiểm định ý nghĩa thống kê bằng thuật toán Mann-Whitney U, giúp các kỹ sư tự tin ra quyết định triển khai."*

---

### Slide 6: Demo Trực Tiếp Dashboard Streamlit & Kết Luận Toàn Đồ Án
- **Tiêu đề:** Demo Thực Tế & Tổng Kết Giá Trị Đồ Án VisionProf
- **Nội dung trên slide:**
  - **Demo trực tiếp trên `http://localhost:8501`**:
    - *Tab 5:* Trực quan hóa các pha Training và kiểm chứng tỷ số $Backward / Forward$.
    - *Tab 6:* Biểu đồ Donut phân bổ Pipeline và thanh đo vi phạm ngưỡng $30\%$.
    - *Tab 7:* Danh mục cảnh báo tự động của Diagnostic Engine.
    - *Tab 8:* So sánh đối đầu CPU vs GPU và FP32 vs FP16.
  - **Thông điệp kết luận:**
    - Hiệu năng AI là một hàm phức hợp giữa **Layer (TV1)**, **Model (TV2)** và **Pipeline (TV3)**.
    - Không thể tối ưu mù quáng; phải đo đúng, đo đủ và chẩn đoán trúng điểm nghẽn.
- **Lời thoại gợi ý (Speaker Notes):**
  > *"Sau đây, em xin phép chuyển sang màn hình Demo trực tiếp giao diện Streamlit Dashboard của VisionProf... (thực hiện demo 1-2 phút qua các Tab 5, 6, 7, 8)... Tóm lại, đồ án của nhóm chúng em đã xây dựng thành công một framework profiling toàn diện, giải thích tường minh từ cấp độ phép tính phần cứng cho tới quy trình ứng dụng thực tế. Em xin chân thành cảm ơn thầy cô và các bạn đã lắng nghe!"*
