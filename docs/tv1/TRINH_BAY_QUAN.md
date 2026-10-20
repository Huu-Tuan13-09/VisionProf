# Hướng dẫn trình bày · Võ Trần Anh Quân

> **Phần:** Nghiên cứu liên quan và mức Layer (CH1, CH2) · **Màu slide:** 🟧 cam · **Slide:** 4–13 (10 slide) · **Thời lượng:** khoảng 8.2 phút
> Lời thoại lấy từ ghi chú người nói trong bộ slide; số liệu đã đối chiếu `results/`. Không cần đọc thuộc: nhìn cột "Ý chính" là đủ nhắc.

## Vị trí trong buổi trình bày

- Bạn Trí chuyển sang bằng câu: *"Trước khi vào thiết kế, bạn Quân sẽ trình bày các nghiên cứu đi trước…"*
- Bạn kết thúc ở slide 13 bằng câu chuyển cho Trí.
- Thứ tự cả nhóm: Trí (1–3) → Quân (4–13) → Trí (14–20) → Tuân (21–30); slide 31 là phụ lục tài liệu tham khảo.

## Tóm tắt các slide

| Slide | Tiêu đề | Thời gian | Ý chính |
| :-: | :--- | :-: | :--- |
| 4 | Hiệu năng phụ thuộc cả chồng phần mềm và phần cứng | 60 giây | MLModelScope L1–L5: nhóm đo L1 (Model, Giai đoạn) và L2–L3 (Layer), không đo L4–L5. |
| 5 | Roofline: nghẽn do tính toán hay do bộ nhớ? | 30 giây | Hiệu năng = min(đỉnh tính toán, băng thông × FLOP/byte); điểm gãy ~19 (CPU), ~25 (T4). |
| 6 | XSP: đo càng sâu, công cụ đo càng làm chậm | 30 giây | Đo càng sâu overhead càng lớn → nhóm tự đo overhead (TN2), tổng thời gian lấy ở mức Model. |
| 7 | So sánh và khoảng trống nhóm lấp | 60 giây | Bảng 6 công trình; khoảng trống: gộp 3 mức trong một công cụ nhẹ; hạn chế: không đo kernel GPU. |
| 8 | Đo thời gian trên GPU không đơn giản | 75 giây | CPU chỉ gửi lệnh; 4 kỹ thuật: CUDA Event, đồng bộ 1 lần, lấp hàng đợi, ngăn xếp. |
| 9 | Thời gian tập trung ở loại layer nào? | 75 giây | CNN: Conv 73–85%; Transformer: 95–96%; MobileNetV3 trên T4: Conv 47%, chuẩn hoá 25%, kích hoạt 17%. |
| 10 | GPU nhanh hơn CPU bao nhiêu lần? | 40 giây | Transformer ×8.8–9.5; CNN ×3.0–5.8; MLP ×1.0. |
| 11 | Vì sao layer lớn chưa nhanh hơn nữa? | 35 giây | Layer lớn chỉ đạt 30–55% sức máy; depthwise ~15%; ít FLOPs ≠ nhanh (ShuffleNet V2, G4). |
| 12 | Công cụ đo có đúng không? | 75 giây | Sai lệch CPU 0.35% / 0.27%, T4 7.5% / 0.8% (đo lại 7.8% / 0.04%); overhead T4 110–210%; ngoài layer 4–10%. |
| 13 | Tab Layer: thời gian đi vào loại layer nào | 15 giây | Tab Layer: MobileNetV3 8.48 → 2.81 ms (×3.0). Kết thúc bằng câu chuyển cho Trí. |

## Lời thoại từng slide

### Slide 4. Hiệu năng phụ thuộc cả chồng phần mềm và phần cứng
*khoảng 60 giây · Ý chính: MLModelScope L1–L5: nhóm đo L1 (Model, Giai đoạn) và L2–L3 (Layer), không đo L4–L5.*

Em xin trình bày các nghiên cứu liên quan. Bức tranh chung đến từ MLModelScope của Dakkak và cộng sự, năm 2020: khi chạy một mô hình học sâu, hiệu năng phụ thuộc cả chồng từ mô hình, framework, layer, thư viện như cuDNN, xuống tới phần cứng. Họ gọi là 5 mức, L1 đến L5.

Vấn đề là các công cụ trước đây thường chỉ nhìn một mức. Nhìn ở mức mô hình thì không biết layer nào chậm; nhìn ở mức kernel thì không biết kernel đó thuộc layer nào.

VisionProf đặt mình vào bức tranh này như sau: mức Model và Giai đoạn của nhóm ứng với L1; mức Layer ứng với L2, L3. Nhóm không đo L4, L5 tức kernel GPU và hardware counter. Đây là hạn chế, em sẽ nói rõ ở slide so sánh.

### Slide 5. Roofline: nghẽn do tính toán hay do bộ nhớ?
*khoảng 30 giây · Ý chính: Hiệu năng = min(đỉnh tính toán, băng thông × FLOP/byte); điểm gãy ~19 (CPU), ~25 (T4).*

Nền tảng đầu tiên là mô hình Roofline của Williams, Waterman và Patterson, năm 2009.

Ý tưởng rất gọn: hiệu năng đạt được bằng giá trị nhỏ hơn giữa đỉnh tính toán của máy và băng thông bộ nhớ nhân với cường độ số học, tức số FLOP trên mỗi byte đọc ghi.

Trên hình, điểm nằm dưới phần mái dốc là bị giới hạn bởi băng thông bộ nhớ; điểm dưới phần mái bằng là bị giới hạn bởi tính toán. Điểm gãy trên laptop của em khoảng 19 FLOP mỗi byte, trên T4 khoảng 25.

Nhóm dùng đúng công thức này cho từng layer, để biết mỗi layer đạt bao nhiêu phần trăm sức tối đa của máy.

### Slide 6. XSP: đo càng sâu, công cụ đo càng làm chậm
*khoảng 30 giây · Ý chính: Đo càng sâu overhead càng lớn → nhóm tự đo overhead (TN2), tổng thời gian lấy ở mức Model.*

Công trình thứ hai là XSP của Li và cộng sự, IPDPS 2020. XSP ghép 3 mức là mô hình, layer và kernel GPU thành một dòng thời gian, và phân tích 65 mô hình.

Điểm quan trọng nhất họ chỉ ra, như trên hình: bật đo ở mức càng sâu thì overhead của chính công cụ đo càng lớn, nên số đo ở mức trên bị sai. XSP giải bằng cách đo lần lượt từng mức, mỗi mức một lượt chạy riêng.

Nhóm gặp đúng vấn đề này và giải theo cách tương tự: tự đo overhead và sai lệch của công cụ so với torch.profiler trong thí nghiệm TN2; mức Layer chỉ dùng để phân rã tỷ lệ, còn tổng thời gian sạch lấy ở mức Model, đo không gắn hook.

### Slide 7. So sánh và khoảng trống nhóm lấp
*khoảng 60 giây · Ý chính: Bảng 6 công trình; khoảng trống: gộp 3 mức trong một công cụ nhẹ; hạn chế: không đo kernel GPU.*

Bảng này so sánh 6 công trình với VisionProf. Lưu ý dấu gạch nghĩa là nội dung đó không phải trọng tâm của paper, không có nghĩa paper hoàn toàn không nhắc tới.

XSP và PRoof mạnh ở mức Layer và kernel, có Roofline. MLPerf chuẩn hoá cách đo độ trễ mức Model theo phân vị. AI Tax chỉ ra rằng trong ứng dụng thật, dưới 60% độ trễ là phần AI, còn lại là các bước xung quanh. Data Stalls chỉ ra training thường phải chờ dữ liệu.

Khoảng trống nhóm muốn lấp: mỗi công trình tập trung một mức, hoặc là hệ thống lớn chạy trên cụm máy, cần hardware counter. VisionProf gộp cả 3 mức vào một công cụ nhẹ cho PyTorch, đo cả suy luận, training và pipeline thật, trên laptop và GPU T4, và có thí nghiệm tự kiểm chứng độ chính xác.

Em cũng nói trước hạn chế: nhóm không đo kernel GPU và không dùng hardware counter, đó là thế mạnh của XSP và PRoof.

> ➡️ **Chuyển: Bắt đầu từ mức sâu nhất, mức Layer.**

### Slide 8. Đo thời gian trên GPU không đơn giản
*khoảng 75 giây · Ý chính: CPU chỉ gửi lệnh; 4 kỹ thuật: CUDA Event, đồng bộ 1 lần, lấp hàng đợi, ngăn xếp.*

Thách thức đầu tiên của mức Layer là đo trên GPU. CPU không đợi GPU: nó chỉ xếp lệnh vào hàng đợi rồi chạy tiếp. Như hình bên trái, CPU gửi lệnh cho 3 layer rất nhanh, còn GPU chạy chúng sau đó. Nếu dùng đồng hồ CPU, ta chỉ đo được thời gian gửi lệnh, gần như bằng 0 và sai hoàn toàn.

Cách dễ nhất là đồng bộ sau mỗi layer. Đo từng layer thì đúng, nhưng GPU phải dừng chờ CPU sau mỗi layer, nên tổng thời gian bị đội lên.

VisionProf làm 4 việc. Một: ghi CUDA Event ngay trong hook trước và sau mỗi layer; event được ghi trên chính dòng lệnh GPU nên đo đúng thời gian GPU chạy. Hai: chỉ đồng bộ một lần ở cuối mỗi vòng rồi mới đọc tất cả event. Ba: trước khi đo, cho GPU một việc nhỏ để hàng đợi luôn đầy, GPU chạy liền mạch thay vì chờ CPU gửi lệnh. Bốn: dùng ngăn xếp để xử lý layer được gọi nhiều lần trong một vòng.

Người dùng chỉ cần khoảng 5 dòng code như bên dưới, không phải sửa mô hình.

### Slide 9. Thời gian tập trung ở loại layer nào?
*khoảng 75 giây · Ý chính: CNN: Conv 73–85%; Transformer: 95–96%; MobileNetV3 trên T4: Conv 47%, chuẩn hoá 25%, kích hoạt 17%.*

Câu hỏi CH1: thời gian tập trung ở loại layer nào. Hình bên trái là tỷ trọng thời gian theo nhóm layer của 7 mô hình, mỗi mô hình 2 thanh: CPU laptop và GPU T4, batch 1.

Trên CPU, câu trả lời khá rõ: mô hình CNN dồn 73 đến 85% thời gian vào Conv; Transformer dồn 95 đến 96% vào Linear và Attention.

Điều thú vị là khi lên GPU. Với ResNet-50 và Transformer, tỷ trọng gần như không đổi. Nhưng với mô hình nhẹ MobileNetV3, Conv giảm từ 73% xuống 47%, trong khi chuẩn hoá tăng từ 14 lên 25%, kích hoạt từ 5 lên 17%.

Lý do: GPU tăng tốc mạnh các phép tính lớn, còn các layer nhỏ như BatchNorm, hàm kích hoạt thì hầu như không nhanh hơn được bao nhiêu, vì thời gian của chúng chủ yếu là chi phí khởi chạy. Nên trên GPU, layer nhỏ chiếm tỷ trọng lớn hơn.

Lưu ý: CPU ở đây là laptop x86 của em.

### Slide 10. GPU nhanh hơn CPU bao nhiêu lần?
*khoảng 40 giây · Ý chính: Transformer ×8.8–9.5; CNN ×3.0–5.8; MLP ×1.0.*

Hình này lấy tổng thời gian các layer trên CPU chia cho trên T4, tức GPU nhanh hơn bao nhiêu lần.

Transformer được tăng tốc mạnh nhất: ViT 9.5 lần, BERT 9.0, DistilBERT 8.8. Lý do như slide trước: 94 đến 96% thời gian của chúng là nhân ma trận lớn, đúng loại việc GPU làm tốt nhất.

CNN được tăng tốc ít hơn: ResNet-50 5.8, YOLOv8n 5.5, và thấp nhất là MobileNetV3 chỉ 3.0 lần, vì nó có nhiều layer nhỏ.

MLP gần như không nhanh hơn, vì mô hình quá nhỏ để tận dụng GPU.

### Slide 11. Vì sao layer lớn chưa nhanh hơn nữa?
*khoảng 35 giây · Ý chính: Layer lớn chỉ đạt 30–55% sức máy; depthwise ~15%; ít FLOPs ≠ nhanh (ShuffleNet V2, G4).*

Vậy vì sao các layer lớn chưa nhanh hơn nữa? Đây là Roofline của các layer Conv, Linear và Attention, bên trái là CPU, bên phải là T4.

Phần lớn các layer nằm ở vùng giới hạn bởi tính toán, nhưng chỉ đạt khoảng 30 đến 55% sức tối đa của máy. Lý do là batch 1: các phép nhân ma trận còn nhỏ nên phần cứng chưa được dùng hết.

Depthwise conv của MobileNetV3 còn thấp hơn, chỉ khoảng 15%, vì có rất ít phép tính trên mỗi byte dữ liệu.

Điều này khớp với nguyên tắc G4 trong bài ShuffleNet V2 của Ma và cộng sự, ECCV 2018: phép toán ít FLOPs vẫn tốn thời gian thật trên GPU. Nên chỉ đếm FLOPs thì không đủ để đoán tốc độ.

### Slide 12. Công cụ đo có đúng không?
*khoảng 75 giây · Ý chính: Sai lệch CPU 0.35% / 0.27%, T4 7.5% / 0.8% (đo lại 7.8% / 0.04%); overhead T4 110–210%; ngoài layer 4–10%.*

Câu hỏi CH2: công cụ đo của nhóm có đúng không. Nhóm so với torch.profiler, công cụ chính thức của PyTorch, trên ResNet-50 và BERT-base.

Sai lệch tổng thời gian: trên CPU chỉ 0.35% và 0.27%; trên T4 là 7.5% và 0.8%. Hôm 18/10 nhóm đo lại trên T4 bằng code hiện tại, kết quả gần như y hệt: 7.8% và 0.04%.

Có hai điểm phải nói thẳng. Thứ nhất, overhead trên GPU ở batch 1 lớn: gắn hook làm mô hình chạy chậm hơn khoảng 110 đến 210% tuỳ phiên Colab. Đó là chi phí ghi event và gọi hook giữa các layer. Vì vậy nhóm chỉ dùng mức Layer để phân rã tỷ lệ, còn tổng thời gian sạch thì lấy ở mức Model, đo không gắn hook. Đây đúng là bài học của XSP ở slide 6.

Thứ hai, khoảng 4 đến 10% thời gian không thuộc layer nào: đó là các phép toán dạng hàm như phép cộng shortcut hay torch.cat, không phải module nên không có hook. Nhóm đo và báo cáo con số này thay vì bỏ qua.

Ngoài ra, layer chỉ chạy vài micro giây như ReLU, BatchNorm ở batch 1 bị đo dư vì CUDA Event có ngưỡng sàn khoảng 6 micro giây.

Tổng kết mức Layer: đủ chính xác để phân rã, và biết rõ giới hạn của mình.

Trước khi chuyển phần, em cho xem kết quả này trên dashboard của nhóm.

### Slide 13. Tab Layer: thời gian đi vào loại layer nào
*khoảng 15 giây · Ý chính: Tab Layer: MobileNetV3 8.48 → 2.81 ms (×3.0). Kết thúc bằng câu chuyển cho Trí.*

Các kết quả mức Layer xem được trực tiếp trên dashboard của nhóm, chạy bằng một lệnh streamlit. Chọn một mô hình, ví dụ MobileNetV3: dashboard cho ngay tổng thời gian các layer trên CPU là 8.48 mili giây, trên T4 là 2.81, GPU nhanh hơn 3 lần.

Biểu đồ bên dưới chính là kết quả CH1 em vừa trình bày: thanh trên là CPU, thanh dưới là T4; Conv giảm từ 73 xuống 47%. Kéo xuống nữa có top-10 layer chậm nhất và Roofline.

> ➡️ **Chuyển: Đo tổng thời gian sạch của cả mô hình là phần của bạn Trí.**

## Câu hỏi có thể gặp

**H: Overhead trên GPU ở batch 1 tới 110–210%, sao vẫn tin số đo?**

Đ: Overhead làm cả mô hình chạy chậm hơn, nhưng sai lệch **tổng thời gian các layer** so với `torch.profiler` vẫn dưới 8% (đo lại 18/10 vẫn vậy). Overhead nằm ở chi phí ghi event và gọi hook giữa các layer. Vì vậy mức Layer chỉ dùng để phân rã tỷ lệ; tổng thời gian sạch lấy ở mức Model, đo không gắn hook.

**H: Phép toán dạng hàm như `x + shortcut`, `torch.cat` đi đâu?**

Đ: Không phải module nên không có hook. Nhóm đo thành chỉ số "thời gian không thuộc layer nào": 3.9–9.2% trên CPU, 9–10% trên T4 (ResNet-50, BERT-base). Cuối kỳ sẽ bắt bằng `TorchDispatchMode`.

**H: Sao không dùng luôn `torch.profiler`?**

Đ: `torch.profiler` dùng làm chuẩn để kiểm chứng (TN2). Nhóm cần thêm FLOPs, Roofline, tổng hợp theo layer và gắn được vào training, pipeline với file kết quả gọn, nên viết bộ đo riêng nhẹ hơn.

**H: Chỉ đồng bộ một lần cuối vòng thì làm sao biết thời gian từng layer?**

Đ: CUDA Event được ghi ngay trên dòng lệnh GPU, mốc thời gian do GPU ghi khi chạy tới đó. Đồng bộ cuối vòng chỉ để chờ GPU chạy xong rồi đọc `elapsed_time` giữa các cặp event.

**H: Layer chỉ chạy vài micro giây có đo đúng không?**

Đ: Không chính xác tuyệt đối: CUDA Event có ngưỡng sàn khoảng 6 µs nên ReLU, BatchNorm ở batch 1 bị đo dư. Nhóm chỉ dùng số này để so sánh tương đối và nói rõ là hạn chế.

**H: Đỉnh Roofline của máy lấy từ đâu?**

Đ: T4 theo datasheet: 8.1 TFLOPS FP32, 320 GB/s, điểm gãy ~25 FLOP/byte. CPU ước lượng theo số nhân × xung nhịp × 32 FLOP/chu kỳ, băng thông 51.2 GB/s, điểm gãy ~19.

**H: Số CPU ở mức Layer đo trên máy nào?**

Đ: Laptop x86 của em (TN1, TN2). Nhóm không so số này với số CPU của TN3, TN4 vì đó là MacBook chip Apple.

## Trước giờ trình bày

- [ ] Đọc to một lượt, bấm giờ.
- [ ] Thuộc các con số trong cột "Ý chính".
- [ ] Tập câu chuyển phần với người nói trước và sau.
