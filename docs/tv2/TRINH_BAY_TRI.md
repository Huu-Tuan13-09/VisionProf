# Hướng dẫn trình bày · Trương Công Trí

> **Phần:** Mở đầu và mức Model (CH3, CH4) · **Màu slide:** 🟦 xanh dương · **Slide:** 1–4, 15–21 (11 slide) · **Thời lượng:** khoảng 9.8 phút
> Lời thoại lấy từ ghi chú người nói trong bộ slide; số liệu đã đối chiếu `results/`. Không cần đọc thuộc: nhìn cột "Ý chính" là đủ nhắc.

## Vị trí trong buổi trình bày

- Bạn mở đầu buổi trình bày (slide 1–4, gồm slide 6 câu hỏi nghiên cứu) rồi chuyển cho Quân.
- Quân chuyển lại cho bạn ở cuối slide 14: *"Đo tổng thời gian sạch của cả mô hình là phần của bạn Trí."*
- Bạn kết thúc ở slide 21 bằng câu chuyển cho Tuân.
- Thứ tự cả nhóm: Trí (1–4) → Quân (5–14) → Trí (15–21) → Tuân (22–31); slide 32 là phụ lục tài liệu tham khảo.

## Tóm tắt các slide

| Slide | Tiêu đề | Thời gian | Ý chính |
| :-: | :--- | :-: | :--- |
| 1 | VisionProf | 40 giây | Chào, giới thiệu đề tài, nói thứ tự trình bày. |
| 2 | Evaluation, Benchmarking hay Profiling? | 50 giây | Evaluation ngoài phạm vi · Benchmarking là mức Model · Profiling là trọng tâm. |
| 3 | 3 mức đo, 6 câu hỏi nghiên cứu | 60 giây | 3 mức ↔ 3 người ↔ CH1–CH6; 8 mô hình; GPU T4 trên Colab; sợi chỉ đỏ ResNet-50 so với ViT, BERT. |
| 4 | 6 câu hỏi nghiên cứu và thí nghiệm trả lời | 40 giây | CH1–CH2 mức Layer (TN1, TN2) · CH3–CH4 mức Model (TN3–TN5) · CH5–CH6 mức Giai đoạn (TN6, TN7). |
| 15 | 8 mô hình, một cách đo chung, tách 2 pha | 60 giây | ModelAdapter 3 hàm; pha 1 đo sạch không hook (15 vòng khởi động + 50 vòng đo); pha 2 mới gắn hook. |
| 16 | Báo cáo phân vị, không dùng trung bình | 60 giây | Báo P50–P99 theo MLPerf; ResNet-50 P99 = 1.76 × P50; ViT ổn định; 2 máy CPU khác nhau. |
| 17 | Ít tham số có chắc nhanh hơn? | 75 giây | T4: ResNet-50 ≈ MobileNetV3 ≈ 6 ms dù FLOPs chênh 19×; MLP GPU 0.28 ms, CPU 0.04 ms; MobileNetV3 176 ms trên chip Apple. |
| 18 | Thứ hạng tốc độ thay đổi theo phần cứng | 60 giây | MobileNetV3 8→5, YOLOv8n 7→6, BERT 5→7, ViT 6→8; không có mô hình nhanh nhất mọi nơi. |
| 19 | Tăng batch và độ dài câu | 65 giây | MobileNetV3 ×8.8 (170 → 1 499 ảnh/s); ViT bão hoà từ batch 8; BERT 127 → 228 → 507 ms theo độ dài câu. |
| 20 | FP16 trên T4: luôn giảm bộ nhớ, không phải lúc nào cũng nhanh hơn | 65 giây | ViT 2.81×, DistilBERT 1.55×, BERT 1.23×; CNN chậm hơn (0.77–0.87×); VRAM giảm 38–49%. |
| 21 | Tab CPU và T4: FP16 có nhanh hơn không | 15 giây | Bảng FP16 trên dashboard. Kết thúc bằng câu chuyển cho Tuân. |

## Lời thoại từng slide

### Slide 1. VisionProf
*khoảng 40 giây · Ý chính: Chào, giới thiệu đề tài, nói thứ tự trình bày.*

Kính chào thầy và các bạn. Nhóm em gồm 3 thành viên, xin trình bày đề tài VisionProf: phân tích hiệu năng mô hình học sâu ở 3 mức là Layer, Model và Giai đoạn.

Nói ngắn gọn, VisionProf là một công cụ nhẹ cho PyTorch. Chỉ cần gắn vào mô hình có sẵn, không phải sửa mã nguồn mô hình, là đo được thời gian và bộ nhớ đi vào đâu.

Bài trình bày khoảng 25 phút: em mở đầu, bạn Quân trình bày các nghiên cứu liên quan và mức Layer, em quay lại với mức Model, sau đó bạn Tuân trình bày mức Giai đoạn, phần chẩn đoán và tổng kết.

### Slide 2. Evaluation, Benchmarking hay Profiling?
*khoảng 50 giây · Ý chính: Evaluation ngoài phạm vi · Benchmarking là mức Model · Profiling là trọng tâm.*

Khi nói "đánh giá mô hình", thường có 3 việc khác nhau.

Evaluation hỏi mô hình dự đoán đúng bao nhiêu: accuracy, mAP, F1. Phần này nằm ngoài đề tài; nhóm dùng trọng số khởi tạo ngẫu nhiên vì chỉ quan tâm tốc độ và bộ nhớ.

Benchmarking hỏi mô hình chạy nhanh bao nhiêu: độ trễ, thông lượng. Đây là một phần của đề tài, nằm ở mức Model.

Profiling đi xa hơn: thời gian và bộ nhớ thực sự đi vào đâu. Layer nào tốn nhất, nó nghẽn vì tính toán hay vì băng thông bộ nhớ, và các bước ngoài mô hình như tiền xử lý, nạp dữ liệu chiếm bao nhiêu.

Câu chốt: benchmark cho biết bao lâu, profiling cho biết ở đâu và vì sao. Đề tài của nhóm tập trung vào profiling.

### Slide 3. 3 mức đo, 6 câu hỏi nghiên cứu
*khoảng 60 giây · Ý chính: 3 mức ↔ 3 người ↔ CH1–CH6; 8 mô hình; GPU T4 trên Colab; sợi chỉ đỏ ResNet-50 so với ViT, BERT.*

VisionProf đo ở 3 mức, mỗi thành viên phụ trách một mức và trả lời 2 câu hỏi nghiên cứu.

Mức Layer do bạn Quân làm: thời gian, FLOPs, bộ nhớ của từng layer, vẽ Roofline, và tự kiểm chứng công cụ đo có đúng không. Trả lời CH1 và CH2.

Mức Model do em làm: độ trễ của cả mô hình theo phân vị P50 đến P99, thông lượng, VRAM, và ảnh hưởng của batch, độ dài câu, FP16. Trả lời CH3 và CH4.

Mức Giai đoạn do bạn Tuân làm: tách một bước training thành forward, backward, optimizer, nạp dữ liệu; và đo pipeline thật gồm tiền xử lý, hậu xử lý. Trả lời CH5 và CH6.

Nhóm đo 8 mô hình thuộc 3 loại dữ liệu. GPU là Tesla T4 trên Google Colab. Số đo CPU lấy từ laptop của nhóm và Colab CPU; slide nào dùng máy nào sẽ ghi rõ ở chân slide.

Sợi chỉ đỏ xuyên suốt là so sánh CNN, đại diện là ResNet-50, với Transformer là ViT và BERT.

> ➡️ **Chuyển: Cụ thể 6 câu hỏi như sau.**

### Slide 4. 6 câu hỏi nghiên cứu và thí nghiệm trả lời
*khoảng 40 giây · Ý chính: CH1–CH2 mức Layer (TN1, TN2) · CH3–CH4 mức Model (TN3–TN5) · CH5–CH6 mức Giai đoạn (TN6, TN7).*

Cụ thể, 6 câu hỏi nghiên cứu là thế này. Mức Layer trả lời CH1: thời gian tập trung ở loại layer nào, và CH2: công cụ đo có đúng không. Mức Model trả lời CH3: ít tham số có chắc nhanh hơn, và CH4: tăng batch, độ dài câu, FP16 thì thay đổi ra sao. Mức Giai đoạn trả lời CH5: training tốn ở đâu, GPU có phải chờ dữ liệu không, và CH6: trong ứng dụng thật, phần ngoài mô hình chiếm bao nhiêu.

Mỗi câu hỏi có một thí nghiệm riêng, đánh số TN1 đến TN7; các slide sau sẽ gọi tắt như vậy.

> ➡️ **Chuyển: Trước khi vào thiết kế, bạn Quân sẽ trình bày các nghiên cứu đi trước và khoảng trống nhóm muốn lấp.**

### Slide 15. 8 mô hình, một cách đo chung, tách 2 pha
*khoảng 60 giây · Ý chính: ModelAdapter 3 hàm; pha 1 đo sạch không hook (15 vòng khởi động + 50 vòng đo); pha 2 mới gắn hook.*

Cảm ơn bạn Quân. Em trình bày mức Model: đo cả mô hình chạy nhanh bao nhiêu, tốn bao nhiêu bộ nhớ.

Bên trái là 8 mô hình. Mỗi mô hình được bọc trong một ModelAdapter có 3 hàm chung: load để nạp mô hình, make_input để tạo đầu vào đúng dạng, forward để chạy. Nhờ vậy cùng một đoạn code đo được cả ảnh, văn bản lẫn dữ liệu bảng.

Bên phải là điểm quan trọng nhất: đo tách 2 pha. Pha 1 đo độ trễ khi không gắn hook nào: khởi động 15 vòng, đo 50 vòng, đồng bộ GPU sau mỗi vòng để lấy đúng thời gian cả mô hình. VRAM đỉnh lấy từ bộ cấp phát của PyTorch; RAM và CPU được lấy mẫu ở một luồng nền mỗi 100 mili giây, không chen vào vòng đo.

Pha 2 mới gắn bộ đo Layer của bạn Quân và bộ đo Giai đoạn của bạn Tuân, chỉ để phân rã tỷ lệ.

Nhờ tách 2 pha, overhead 110 đến 210% của hook mà bạn Quân vừa nói không lọt vào độ trễ nhóm báo cáo.

### Slide 16. Báo cáo phân vị, không dùng trung bình
*khoảng 60 giây · Ý chính: Báo P50–P99 theo MLPerf; ResNet-50 P99 = 1.76 × P50; ViT ổn định; 2 máy CPU khác nhau.*

Nhóm không báo cáo độ trễ trung bình mà báo cáo theo phân vị: P50 là trung vị, P90, P95, P99 là phần đuôi. Lý do theo chuẩn MLPerf Inference của Reddi và cộng sự, ISCA 2020: người dùng cảm nhận những lần chậm, nên chuẩn công nghiệp lấy phân vị 90 hoặc 99.

Biểu đồ bên trái là số đo thật trên T4: thanh đậm là P50, thanh nhạt là P99. Với ResNet-50, P50 là 5.96 mili giây nhưng P99 là 10.46, gấp 1.76 lần. Nếu chỉ báo trung bình thì sẽ không thấy phần đuôi này. Ngược lại ViT rất ổn định, P99 chỉ hơn P50 khoảng 5%.

Mỗi lần chạy, công cụ tự lưu thông tin máy và phiên bản thư viện vào env.json để có thể đối chiếu lại.

Một lưu ý quan trọng: "CPU laptop" trong đề tài là 2 máy khác nhau. Thí nghiệm mức Layer chạy trên laptop x86 của bạn Quân; thí nghiệm mức Model chạy trên MacBook chip Apple của em. Nhóm không so số CPU giữa 2 máy này.

### Slide 17. Ít tham số có chắc nhanh hơn?
*khoảng 75 giây · Ý chính: T4: ResNet-50 ≈ MobileNetV3 ≈ 6 ms dù FLOPs chênh 19×; MLP GPU 0.28 ms, CPU 0.04 ms; MobileNetV3 176 ms trên chip Apple.*

Câu hỏi CH3: mô hình ít tham số có chắc chạy nhanh hơn không. Bảng bên trái là độ trễ trung vị ở batch 1, FP32, sắp theo T4.

Phát hiện thứ nhất, ở 2 dòng tô màu: trên T4, MobileNetV3 có số phép tính ít hơn ResNet-50 tới 19 lần, 0.43 so với 8.2 GFLOPs, nhưng hai mô hình cùng mất khoảng 6 mili giây. Ở batch 1, mỗi layer quá nhỏ để lấp đầy GPU, nên thời gian chủ yếu là chi phí CPU gửi từng kernel xuống GPU. MobileNetV3 có nhiều layer nhỏ nên không tận dụng được lợi thế ít phép tính. Điều này khớp với kết quả CH1 của bạn Quân: trên GPU, layer nhỏ chiếm tỷ trọng lớn hơn.

Phát hiện thứ hai: mô hình rất nhỏ chạy trên GPU còn chậm hơn CPU. MLP mất 0.28 mili giây trên T4 nhưng chỉ 0.04 trên CPU, vì chi phí gửi lệnh và chép dữ liệu sang GPU lớn hơn chính phép tính.

Phát hiện thứ ba, phải nói cho đúng: MobileNetV3 trên MacBook chip Apple chậm bất thường, 176 mili giây, trong khi trên laptop x86 chỉ khoảng 12. Nghĩa là tốc độ phụ thuộc cả mức tối ưu của thư viện cho từng phần cứng, không chỉ ở kiến trúc mô hình.

### Slide 18. Thứ hạng tốc độ thay đổi theo phần cứng
*khoảng 60 giây · Ý chính: MobileNetV3 8→5, YOLOv8n 7→6, BERT 5→7, ViT 6→8; không có mô hình nhanh nhất mọi nơi.*

Hình này xếp hạng 8 mô hình theo độ trễ, bên trái trên MacBook, bên phải trên T4. Hạng 1 là nhanh nhất.

Thứ hạng có thay đổi: MobileNetV3 từ hạng 8 lên hạng 5, YOLOv8n từ 7 lên 6; ngược lại BERT-base tụt từ 5 xuống 7 và ViT từ 6 xuống 8. DistilBERT và ResNet-50 giữ nguyên hạng 3 và 4.

Nhưng em nói rõ: phần lớn sự thay đổi đến từ việc MobileNetV3 chạy chậm bất thường trên chip Apple như slide trước. Vì vậy nhóm không gọi đây là "GPU đảo ngược thứ hạng", mà rút ra bài học thực tế hơn: không có mô hình nhanh nhất ở mọi nơi. Chọn mô hình phải đo trên đúng phần cứng sẽ triển khai.

### Slide 19. Tăng batch và độ dài câu
*khoảng 65 giây · Ý chính: MobileNetV3 ×8.8 (170 → 1 499 ảnh/s); ViT bão hoà từ batch 8; BERT 127 → 228 → 507 ms theo độ dài câu.*

Câu hỏi CH4, phần thứ nhất: tăng batch thì thông lượng tăng bao nhiêu. Bảng bên trái là số mẫu xử lý mỗi giây trên T4.

Mô hình nhẹ hưởng lợi nhiều nhất: MobileNetV3 tăng 8.8 lần, từ 170 lên gần 1 500 ảnh mỗi giây, vì ở batch 1 GPU gần như ngồi không, gộp nhiều ảnh mới lấp đầy được GPU. Đây cũng là cách khắc phục hiện tượng ở CH3. Ngược lại ViT gần như bão hoà ngay từ batch 8, tăng thêm batch không nhanh hơn; BERT và DistilBERT cũng chỉ tăng 1.4 đến 1.5 lần.

Phần thứ hai, với mô hình văn bản: BERT-base ở batch 32 mất 127 mili giây với câu 64 token, 228 với 128 token, và 507 với 256 token. Mỗi lần gấp đôi độ dài, độ trễ tăng 1.8 rồi 2.2 lần, tức nhanh hơn tuyến tính, vì phần Attention tăng theo bình phương độ dài. DistilBERT cũng tương tự.

Kết luận: mô hình nhẹ cần batch lớn mới tận dụng GPU, mô hình nặng bão hoà sớm, và với NLP cần giới hạn độ dài câu.

### Slide 20. FP16 trên T4: luôn giảm bộ nhớ, không phải lúc nào cũng nhanh hơn
*khoảng 65 giây · Ý chính: ViT 2.81×, DistilBERT 1.55×, BERT 1.23×; CNN chậm hơn (0.77–0.87×); VRAM giảm 38–49%.*

Phần thứ hai của CH4: chuyển sang FP16 trên T4, batch 1.

Về bộ nhớ, kết quả nhất quán: FP16 luôn giảm VRAM, từ 38 đến 49% với mọi mô hình.

Về tốc độ thì có hai mặt. FP16 chỉ nhanh hơn với Transformer: ViT nhanh hơn 2.81 lần, DistilBERT 1.55, BERT 1.23, vì phép nhân ma trận lớn dùng được Tensor Core của T4. Còn CNN ở batch 1 lại chậm hơn: ResNet-50 chỉ còn 0.87 lần, YOLOv8n 0.77 lần. Lý do giống CH3: ở batch 1 thời gian bị chi phối bởi gửi kernel và các layer nhỏ, Tensor Core không có việc, còn FP16 thêm thao tác chuyển kiểu.

Về độ chính xác số học: nhóm so output FP16 với FP32 cùng trọng số, cùng đầu vào. Cosine đạt 0.999999 trở lên. Riêng ViT và MobileNetV3 không tính được vì nhóm dùng trọng số khởi tạo ngẫu nhiên: lớp cuối của ViT được khởi tạo bằng 0 nên output luôn bằng 0. Đây không phải lỗi của FP16.

Bảng này cũng xem được trực tiếp trên dashboard.

### Slide 21. Tab CPU và T4: FP16 có nhanh hơn không
*khoảng 15 giây · Ý chính: Bảng FP16 trên dashboard. Kết thúc bằng câu chuyển cho Tuân.*

Trên dashboard, tab CPU và T4 có mục FP16: mỗi dòng là một mô hình với độ trễ FP32, FP16, mức tăng tốc và VRAM tiết kiệm. Ta thấy ngay ViT nhanh hơn 2.81 lần, còn ResNet-50 và YOLOv8n chậm hơn ở batch 1.

> ➡️ **Chuyển: Mô hình chỉ là một mắt xích. Bạn Tuân sẽ đưa chúng ta đến training và pipeline thật.**

## Câu hỏi có thể gặp

**H: Sao MobileNetV3 trên CPU chậm hơn ResNet-50 tới 8 lần?**

Đ: Chỉ xảy ra trên MacBook chip Apple (176 ms); trên laptop x86 khoảng 12 ms. Đây là đặc thù kernel PyTorch trên CPU ARM, không phải tính chất chung của CPU. Trên T4, hai mô hình bằng nhau (~6 ms).

**H: Sao FP16 làm CNN chậm hơn?**

Đ: Ở batch 1, thời gian chủ yếu là gửi kernel và các layer nhỏ, Tensor Core không có việc; FP16 còn thêm thao tác đổi kiểu. Transformer có phép nhân ma trận lớn nên được lợi.

**H: Sao báo P50, P99 mà không dùng trung bình?**

Đ: Theo MLPerf Inference (ISCA 2020): người dùng cảm nhận phần đuôi chậm. Ví dụ ResNet-50 trên T4 có P99 gấp 1.76 lần P50; trung bình che mất điều này.

**H: Có phải tất cả đều đo trên Colab?**

Đ: GPU thì đúng, tất cả là Tesla T4 trên Colab. CPU ở TN3, TN4 đo trên MacBook của em; chân slide đã ghi rõ. Vì 2 máy CPU khác nhau nên nhóm không so số CPU chéo giữa các thí nghiệm.

**H: Cosine của ViT và MobileNetV3 sao "không xác định"?**

Đ: Nhóm dùng trọng số khởi tạo ngẫu nhiên: lớp cuối của ViT khởi tạo bằng 0 nên output luôn 0; MobileNetV3 cho output cỡ 1e-10 nên FP16 tràn dưới thành 0. Không phải lỗi FP16; các mô hình khác đạt cosine ≥ 0.999999.

**H: Sao độ trễ tăng nhanh hơn tuyến tính theo độ dài câu?**

Đ: Self-attention tính ma trận độ dài × độ dài nên tăng theo bình phương: BERT batch 32 từ 127 → 228 → 507 ms khi độ dài 64 → 128 → 256.

**H: Đo 2 pha để làm gì?**

Đ: Pha 1 đo độ trễ không gắn hook nên số sạch; pha 2 mới gắn bộ đo Layer và Giai đoạn để phân rã. Overhead của hook (slide 13) không lọt vào độ trễ báo cáo.

## Trước giờ trình bày

- [ ] Đọc to một lượt, bấm giờ.
- [ ] Thuộc các con số trong cột "Ý chính".
- [ ] Tập câu chuyển phần với người nói trước và sau.
