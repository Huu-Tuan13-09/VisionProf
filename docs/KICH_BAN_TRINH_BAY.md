# Kịch bản trình bày giữa kỳ (lời thoại từng slide)

> Lời thoại khớp với bộ slide HTML (24 slide, cùng nội dung với phần ghi chú của người nói trong slide). Mọi con số đã đối chiếu với `results/` và `docs/KET_QUA_COLAB_GIUA_KY.md`.
> Dàn ý, thời lượng và câu hỏi phản biện: `docs/DAN_Y_SLIDE.md`. Thời lượng: TV3 9.5 phút · TV1 8 phút · TV2 6.5 phút, tổng khoảng 24 phút.

## 🟩 TV3 · Mở đầu (slide 1–3)

### Slide 1. VisionProf
*TV3 · khoảng 40 giây.*

Kính chào thầy và các bạn. Nhóm em gồm 3 thành viên, xin trình bày đề tài VisionProf: phân tích hiệu năng mô hình học sâu ở 3 mức là Layer, Model và Giai đoạn.

Nói ngắn gọn, VisionProf là một công cụ nhẹ cho PyTorch. Chỉ cần gắn vào mô hình có sẵn, không phải sửa mã nguồn mô hình, là đo được thời gian và bộ nhớ đi vào đâu.

Bài trình bày khoảng 24 phút: em mở đầu, bạn TV1 trình bày các nghiên cứu liên quan và mức Layer, bạn TV2 trình bày mức Model, sau đó em quay lại với mức Giai đoạn, phần chẩn đoán, demo và tổng kết.

### Slide 2. Evaluation, Benchmarking hay Profiling?
*TV3 · khoảng 50 giây.*

Khi nói "đánh giá mô hình", thường có 3 việc khác nhau.

Evaluation hỏi mô hình dự đoán đúng bao nhiêu: accuracy, mAP, F1. Phần này nằm ngoài đề tài; nhóm dùng trọng số khởi tạo ngẫu nhiên vì chỉ quan tâm tốc độ và bộ nhớ.

Benchmarking hỏi mô hình chạy nhanh bao nhiêu: độ trễ, thông lượng. Đây là một phần của đề tài, nằm ở mức Model.

Profiling đi xa hơn: thời gian và bộ nhớ thực sự đi vào đâu. Layer nào tốn nhất, nó nghẽn vì tính toán hay vì băng thông bộ nhớ, và các bước ngoài mô hình như tiền xử lý, nạp dữ liệu chiếm bao nhiêu.

Câu chốt: benchmark cho biết bao lâu, profiling cho biết ở đâu và vì sao. Đề tài của nhóm tập trung vào profiling.

### Slide 3. 3 mức đo, 6 câu hỏi nghiên cứu
*TV3 · khoảng 60 giây.*

VisionProf đo ở 3 mức, mỗi thành viên phụ trách một mức và trả lời 2 câu hỏi nghiên cứu.

Mức Layer do TV1 làm: thời gian, FLOPs, bộ nhớ của từng layer, vẽ Roofline, và tự kiểm chứng công cụ đo có đúng không. Trả lời CH1 và CH2.

Mức Model do TV2 làm: độ trễ của cả mô hình theo phân vị P50 đến P99, thông lượng, VRAM, và ảnh hưởng của batch, độ dài câu, FP16. Trả lời CH3 và CH4.

Mức Giai đoạn do em làm: tách một bước training thành forward, backward, optimizer, nạp dữ liệu; và đo pipeline thật gồm tiền xử lý, hậu xử lý. Trả lời CH5 và CH6.

Nhóm đo 8 mô hình thuộc 3 loại dữ liệu, trên 3 loại phần cứng. Lưu ý: hai máy CPU là hai máy khác nhau, laptop x86 của TV1 và MacBook chip Apple của TV2, nên nhóm không so số CPU giữa hai máy.

Sợi chỉ đỏ xuyên suốt là so sánh CNN, đại diện là ResNet-50, với Transformer là ViT và BERT.

Chuyển: Trước khi vào thiết kế, bạn TV1 sẽ trình bày các nghiên cứu đi trước và khoảng trống nhóm muốn lấp.

## 🟧 TV1 · Nghiên cứu liên quan (slide 4–6)

### Slide 4. Hiệu năng phụ thuộc cả chồng phần mềm và phần cứng
*TV1 · khoảng 60 giây.*

Em xin trình bày các nghiên cứu liên quan. Bức tranh chung đến từ MLModelScope của Dakkak và cộng sự, năm 2020: khi chạy một mô hình học sâu, hiệu năng phụ thuộc cả chồng từ mô hình, framework, layer, thư viện như cuDNN, xuống tới phần cứng. Họ gọi là 5 mức, L1 đến L5.

Vấn đề là các công cụ trước đây thường chỉ nhìn một mức. Nhìn ở mức mô hình thì không biết layer nào chậm; nhìn ở mức kernel thì không biết kernel đó thuộc layer nào.

VisionProf đặt mình vào bức tranh này như sau: mức Model và Giai đoạn của nhóm ứng với L1; mức Layer ứng với L2, L3. Nhóm không đo L4, L5 tức kernel GPU và hardware counter. Đây là hạn chế, em sẽ nói rõ ở slide so sánh.

### Slide 5. Roofline và overhead của chính công cụ đo
*TV1 · khoảng 60 giây.*

Hai công trình là nền móng cho mức Layer của nhóm.

Bên trái là mô hình Roofline của Williams và cộng sự, năm 2009. Hiệu năng đạt được bằng giá trị nhỏ hơn giữa đỉnh tính toán của máy và băng thông bộ nhớ nhân với cường độ số học, tức số FLOP trên mỗi byte đọc ghi. Điểm nằm dưới phần mái dốc là bị giới hạn bởi bộ nhớ, dưới phần mái bằng là bị giới hạn bởi tính toán. Nhóm dùng đúng công thức này cho từng layer.

Bên phải là XSP của Li và cộng sự, IPDPS 2020. Điểm quan trọng nhất họ chỉ ra: đo càng sâu thì overhead của chính công cụ đo càng lớn, nên số đo dễ bị sai. XSP giải quyết bằng cách đo lần lượt từng mức.

Nhóm gặp đúng vấn đề này và giải theo cách khác: đo thẳng overhead và sai lệch của công cụ so với torch.profiler trong thí nghiệm TN2; mức Layer chỉ dùng để phân rã tỷ lệ, còn tổng thời gian sạch thì lấy ở mức Model, đo không gắn hook.

### Slide 6. So sánh và khoảng trống nhóm lấp
*TV1 · khoảng 60 giây.*

Bảng này so sánh 6 công trình với VisionProf. Lưu ý dấu gạch nghĩa là nội dung đó không phải trọng tâm của paper, không có nghĩa paper hoàn toàn không nhắc tới.

XSP và PRoof mạnh ở mức Layer và kernel, có Roofline. MLPerf chuẩn hoá cách đo độ trễ mức Model theo phân vị. AI Tax chỉ ra rằng trong ứng dụng thật, dưới 60% độ trễ là phần AI, còn lại là các bước xung quanh. Data Stalls chỉ ra training thường phải chờ dữ liệu.

Khoảng trống nhóm muốn lấp: mỗi công trình tập trung một mức, hoặc là hệ thống lớn chạy trên cụm máy, cần hardware counter. VisionProf gộp cả 3 mức vào một công cụ nhẹ cho PyTorch, đo cả suy luận, training và pipeline thật, trên laptop và GPU T4, và có thí nghiệm tự kiểm chứng độ chính xác.

Em cũng nói trước hạn chế: nhóm không đo kernel GPU và không dùng hardware counter, đó là thế mạnh của XSP và PRoof.

Chuyển: Bắt đầu từ mức sâu nhất, mức Layer.

## 🟧 TV1 · Mức Layer (slide 7–10)

### Slide 7. Đo thời gian trên GPU không đơn giản
*TV1 · khoảng 75 giây.*

Thách thức đầu tiên của mức Layer là đo trên GPU. CPU không đợi GPU: nó chỉ xếp lệnh vào hàng đợi rồi chạy tiếp. Như hình bên trái, CPU gửi lệnh cho 3 layer rất nhanh, còn GPU chạy chúng sau đó. Nếu dùng đồng hồ CPU, ta chỉ đo được thời gian gửi lệnh, gần như bằng 0 và sai hoàn toàn.

Cách dễ nhất là đồng bộ sau mỗi layer. Đo từng layer thì đúng, nhưng GPU phải dừng chờ CPU sau mỗi layer, nên tổng thời gian bị đội lên.

VisionProf làm 4 việc. Một: ghi CUDA Event ngay trong hook trước và sau mỗi layer; event được ghi trên chính dòng lệnh GPU nên đo đúng thời gian GPU chạy. Hai: chỉ đồng bộ một lần ở cuối mỗi vòng rồi mới đọc tất cả event. Ba: trước khi đo, cho GPU một việc nhỏ để hàng đợi luôn đầy, GPU chạy liền mạch thay vì chờ CPU gửi lệnh. Bốn: dùng ngăn xếp để xử lý layer được gọi nhiều lần trong một vòng.

Người dùng chỉ cần khoảng 5 dòng code như bên dưới, không phải sửa mô hình.

### Slide 8. Thời gian tập trung ở loại layer nào?
*TV1 · khoảng 75 giây.*

Câu hỏi CH1: thời gian tập trung ở loại layer nào. Hình bên trái là tỷ trọng thời gian theo nhóm layer của 7 mô hình, mỗi mô hình 2 thanh: CPU laptop và GPU T4, batch 1.

Trên CPU, câu trả lời khá rõ: mô hình CNN dồn 73 đến 85% thời gian vào Conv; Transformer dồn 95 đến 96% vào Linear và Attention.

Điều thú vị là khi lên GPU. Với ResNet-50 và Transformer, tỷ trọng gần như không đổi. Nhưng với mô hình nhẹ MobileNetV3, Conv giảm từ 73% xuống 47%, trong khi chuẩn hoá tăng từ 14 lên 25%, kích hoạt từ 5 lên 17%.

Lý do: GPU tăng tốc mạnh các phép tính lớn, còn các layer nhỏ như BatchNorm, hàm kích hoạt thì hầu như không nhanh hơn được bao nhiêu, vì thời gian của chúng chủ yếu là chi phí khởi chạy. Nên trên GPU, layer nhỏ chiếm tỷ trọng lớn hơn.

Lưu ý: CPU ở đây là laptop x86 của em.

### Slide 9. GPU giúp được bao nhiêu, và vì sao?
*TV1 · khoảng 75 giây.*

Hình bên trái: lấy tổng thời gian các layer trên CPU chia cho trên T4. Transformer được tăng tốc mạnh nhất: ViT 9.5 lần, BERT 9.0, DistilBERT 8.8. CNN được tăng tốc ít hơn: ResNet-50 5.8, YOLOv8n 5.5, MobileNetV3 chỉ 3.0. MLP gần như không nhanh hơn, vì mô hình quá nhỏ.

Vì sao? Như slide trước, Transformer dành 94 đến 96% thời gian cho nhân ma trận lớn, đúng loại việc GPU làm tốt nhất.

Hình bên phải là Roofline của các layer Conv, Linear, Attention. Phần lớn chúng nằm ở vùng giới hạn bởi tính toán, nhưng chỉ đạt khoảng 30 đến 55% sức tối đa của máy. Lý do là batch 1: các phép nhân ma trận còn nhỏ nên phần cứng chưa được dùng hết. Depthwise conv của MobileNetV3 còn thấp hơn, chỉ khoảng 15%, vì có rất ít phép tính trên mỗi byte dữ liệu.

Điều này khớp với nguyên tắc G4 trong bài ShuffleNet V2 của Ma và cộng sự, ECCV 2018: phép toán từng phần tử có ít FLOPs nhưng vẫn tốn thời gian thật trên GPU. Nên đếm FLOPs không đủ để đoán tốc độ.

### Slide 10. Công cụ đo có đúng không?
*TV1 · khoảng 75 giây.*

Câu hỏi CH2: công cụ đo của nhóm có đúng không. Nhóm so với torch.profiler, công cụ chính thức của PyTorch, trên ResNet-50 và BERT-base.

Sai lệch tổng thời gian: trên CPU chỉ 0.35% và 0.27%; trên T4 là 7.5% và 0.8%. Hôm 18/10 nhóm đo lại trên T4 bằng code hiện tại, kết quả gần như y hệt: 7.8% và 0.04%.

Có hai điểm phải nói thẳng. Thứ nhất, overhead trên GPU ở batch 1 lớn: gắn hook làm mô hình chạy chậm hơn khoảng 110 đến 210% tuỳ phiên Colab. Đó là chi phí ghi event và gọi hook giữa các layer. Vì vậy nhóm chỉ dùng mức Layer để phân rã tỷ lệ, còn tổng thời gian sạch thì lấy ở mức Model, đo không gắn hook. Đây đúng là bài học của XSP ở slide 5.

Thứ hai, khoảng 4 đến 10% thời gian không thuộc layer nào: đó là các phép toán dạng hàm như phép cộng shortcut hay torch.cat, không phải module nên không có hook. Nhóm đo và báo cáo con số này thay vì bỏ qua.

Ngoài ra, layer chỉ chạy vài micro giây như ReLU, BatchNorm ở batch 1 bị đo dư vì CUDA Event có ngưỡng sàn khoảng 6 micro giây.

Tổng kết mức Layer: đủ chính xác để phân rã, và biết rõ giới hạn của mình.

Chuyển: Đo tổng thời gian sạch của cả mô hình là phần của bạn TV2.

## 🟦 TV2 · Mức Model (slide 11–16)

### Slide 11. 8 mô hình, một cách đo chung, tách 2 pha
*TV2 · khoảng 60 giây.*

Cảm ơn bạn TV1. Em trình bày mức Model: đo cả mô hình chạy nhanh bao nhiêu, tốn bao nhiêu bộ nhớ.

Bên trái là 8 mô hình. Mỗi mô hình được bọc trong một ModelAdapter có 3 hàm chung: load để nạp mô hình, make_input để tạo đầu vào đúng dạng, forward để chạy. Nhờ vậy cùng một đoạn code đo được cả ảnh, văn bản lẫn dữ liệu bảng.

Bên phải là điểm quan trọng nhất: đo tách 2 pha. Pha 1 đo độ trễ khi không gắn hook nào: khởi động 15 vòng, đo 50 vòng, đồng bộ GPU sau mỗi vòng để lấy đúng thời gian cả mô hình. VRAM đỉnh lấy từ bộ cấp phát của PyTorch; RAM và CPU được lấy mẫu ở một luồng nền mỗi 100 mili giây, không chen vào vòng đo.

Pha 2 mới gắn bộ đo Layer của TV1 và bộ đo Giai đoạn của TV3, chỉ để phân rã tỷ lệ.

Nhờ tách 2 pha, overhead 110 đến 210% của hook mà bạn TV1 vừa nói không lọt vào độ trễ nhóm báo cáo.

### Slide 12. Báo cáo phân vị, không dùng trung bình
*TV2 · khoảng 60 giây.*

Nhóm không báo cáo độ trễ trung bình mà báo cáo theo phân vị: P50 là trung vị, P90, P95, P99 là phần đuôi. Lý do theo chuẩn MLPerf Inference của Reddi và cộng sự, ISCA 2020: người dùng cảm nhận những lần chậm, nên chuẩn công nghiệp lấy phân vị 90 hoặc 99.

Biểu đồ bên trái là số đo thật trên T4: thanh đậm là P50, thanh nhạt là P99. Với ResNet-50, P50 là 5.96 mili giây nhưng P99 là 10.46, gấp 1.76 lần. Nếu chỉ báo trung bình thì sẽ không thấy phần đuôi này. Ngược lại ViT rất ổn định, P99 chỉ hơn P50 khoảng 5%.

Mỗi lần chạy, công cụ tự lưu thông tin máy và phiên bản thư viện vào env.json để có thể đối chiếu lại.

Một lưu ý quan trọng: "CPU laptop" trong đề tài là 2 máy khác nhau. Thí nghiệm mức Layer chạy trên laptop x86 của TV1; thí nghiệm mức Model chạy trên MacBook chip Apple của em. Nhóm không so số CPU giữa 2 máy này.

### Slide 13. Ít tham số có chắc nhanh hơn?
*TV2 · khoảng 75 giây.*

Câu hỏi CH3: mô hình ít tham số có chắc chạy nhanh hơn không. Bảng bên trái là độ trễ trung vị ở batch 1, FP32, sắp theo T4.

Phát hiện thứ nhất, ở 2 dòng tô màu: trên T4, MobileNetV3 có số phép tính ít hơn ResNet-50 tới 19 lần, 0.43 so với 8.2 GFLOPs, nhưng hai mô hình cùng mất khoảng 6 mili giây. Ở batch 1, mỗi layer quá nhỏ để lấp đầy GPU, nên thời gian chủ yếu là chi phí CPU gửi từng kernel xuống GPU. MobileNetV3 có nhiều layer nhỏ nên không tận dụng được lợi thế ít phép tính. Điều này khớp với kết quả CH1 của bạn TV1: trên GPU, layer nhỏ chiếm tỷ trọng lớn hơn.

Phát hiện thứ hai: mô hình rất nhỏ chạy trên GPU còn chậm hơn CPU. MLP mất 0.28 mili giây trên T4 nhưng chỉ 0.04 trên CPU, vì chi phí gửi lệnh và chép dữ liệu sang GPU lớn hơn chính phép tính.

Phát hiện thứ ba, phải nói cho đúng: MobileNetV3 trên MacBook chip Apple chậm bất thường, 176 mili giây, trong khi trên laptop x86 của TV1 chỉ khoảng 12. Nghĩa là tốc độ phụ thuộc cả mức tối ưu của thư viện cho từng phần cứng, không chỉ ở kiến trúc mô hình.

### Slide 14. Thứ hạng tốc độ thay đổi theo phần cứng
*TV2 · khoảng 60 giây.*

Hình này xếp hạng 8 mô hình theo độ trễ, bên trái trên MacBook, bên phải trên T4. Hạng 1 là nhanh nhất.

Thứ hạng có thay đổi: MobileNetV3 từ hạng 8 lên hạng 5, YOLOv8n từ 7 lên 6; ngược lại BERT-base tụt từ 5 xuống 7 và ViT từ 6 xuống 8. DistilBERT và ResNet-50 giữ nguyên hạng 3 và 4.

Nhưng em nói rõ: phần lớn sự thay đổi đến từ việc MobileNetV3 chạy chậm bất thường trên chip Apple như slide trước. Vì vậy nhóm không gọi đây là "GPU đảo ngược thứ hạng", mà rút ra bài học thực tế hơn: không có mô hình nhanh nhất ở mọi nơi. Chọn mô hình phải đo trên đúng phần cứng sẽ triển khai.

### Slide 15. Tăng batch và độ dài câu
*TV2 · khoảng 65 giây.*

Câu hỏi CH4, phần thứ nhất: tăng batch thì thông lượng tăng bao nhiêu. Bảng bên trái là số mẫu xử lý mỗi giây trên T4.

Mô hình nhẹ hưởng lợi nhiều nhất: MobileNetV3 tăng 8.8 lần, từ 170 lên gần 1 500 ảnh mỗi giây, vì ở batch 1 GPU gần như ngồi không, gộp nhiều ảnh mới lấp đầy được GPU. Đây cũng là cách khắc phục hiện tượng ở CH3. Ngược lại ViT gần như bão hoà ngay từ batch 8, tăng thêm batch không nhanh hơn; BERT và DistilBERT cũng chỉ tăng 1.4 đến 1.5 lần.

Phần thứ hai, với mô hình văn bản: BERT-base ở batch 32 mất 127 mili giây với câu 64 token, 228 với 128 token, và 507 với 256 token. Mỗi lần gấp đôi độ dài, độ trễ tăng 1.8 rồi 2.2 lần, tức nhanh hơn tuyến tính, vì phần Attention tăng theo bình phương độ dài. DistilBERT cũng tương tự.

Kết luận: mô hình nhẹ cần batch lớn mới tận dụng GPU, mô hình nặng bão hoà sớm, và với NLP cần giới hạn độ dài câu.

### Slide 16. FP16 trên T4: luôn giảm bộ nhớ, không phải lúc nào cũng nhanh hơn
*TV2 · khoảng 65 giây.*

Phần thứ hai của CH4: chuyển sang FP16 trên T4, batch 1.

Về bộ nhớ, kết quả nhất quán: FP16 luôn giảm VRAM, từ 38 đến 49% với mọi mô hình.

Về tốc độ thì có hai mặt. FP16 chỉ nhanh hơn với Transformer: ViT nhanh hơn 2.81 lần, DistilBERT 1.55, BERT 1.23, vì phép nhân ma trận lớn dùng được Tensor Core của T4. Còn CNN ở batch 1 lại chậm hơn: ResNet-50 chỉ còn 0.87 lần, YOLOv8n 0.77 lần. Lý do giống CH3: ở batch 1 thời gian bị chi phối bởi gửi kernel và các layer nhỏ, Tensor Core không có việc, còn FP16 thêm thao tác chuyển kiểu.

Về độ chính xác số học: nhóm so output FP16 với FP32 cùng trọng số, cùng đầu vào. Cosine đạt 0.999999 trở lên. Riêng ViT và MobileNetV3 không tính được vì nhóm dùng trọng số khởi tạo ngẫu nhiên: lớp cuối của ViT được khởi tạo bằng 0 nên output luôn bằng 0. Đây không phải lỗi của FP16.

Chuyển: Mô hình chỉ là một mắt xích. Bạn TV3 sẽ đưa chúng ta đến training và pipeline thật.

## 🟩 TV3 · Mức Giai đoạn, chẩn đoán và tổng kết (slide 17–23)

### Slide 17. Training tốn thời gian và bộ nhớ ở đâu?
*TV3 · khoảng 60 giây.*

Cảm ơn bạn TV2. Em chuyển sang mức Giai đoạn, bắt đầu với training. Câu hỏi CH5: một bước training tốn thời gian và bộ nhớ ở đâu.

Công cụ là PhaseTimer: chỉ cần bọc từng giai đoạn trong một khối with timer.measure, như đoạn code bên dưới. Ở biên mỗi giai đoạn, PhaseTimer đồng bộ GPU để đo đúng thời gian, và không phải sửa mô hình.

Kết quả trên T4, batch 8: backward bằng 1.4 đến 2.2 lần forward. DistilBERT 2.21 lần, gần đúng mức lý thuyết 2 lần, vì mỗi phép nhân ma trận khi lan truyền ngược phải tính gradient cho cả đầu vào lẫn trọng số. ResNet-50 1.82, MobileNetV3 1.65, MLP 1.39.

Về bộ nhớ: VRAM khi training gấp 2.7 lần suy luận với ResNet-50 và 3.3 lần với MobileNetV3. Lý do là khi training phải giữ lại activation của mọi layer để dùng cho backward. Điều này khớp với nghiên cứu TBD của Zhu và cộng sự, IISWC 2018: feature map chiếm 62 đến 89% bộ nhớ khi training, không phải trọng số. DistilBERT nhóm chưa đo bộ nhớ, sẽ bổ sung cuối kỳ.

### Slide 18. Nạp dữ liệu có làm GPU phải chờ?
*TV3 · khoảng 70 giây.*

Phần thứ hai của CH5: trong training thật, GPU có phải ngồi chờ dữ liệu không. Nhóm dùng DataLoader thật: mỗi ảnh là file JPEG được giải mã và augmentation trên CPU, giống quy trình ImageNet, batch 32, trên Colab có 2 vCPU.

Biểu đồ tách mỗi bước thành 4 phần: phần xám là chờ dữ liệu, rồi forward, backward, optimizer.

Với num_workers bằng 0, tức nạp dữ liệu ngay trên tiến trình chính, MobileNetV3 mất 62.6% thời gian chỉ để chờ dữ liệu. GPU tính rất nhanh nhưng phải đợi CPU giải mã ảnh. ResNet-50 nặng hơn nên tỷ lệ chờ thấp hơn, 30.1%.

Chỉ cần 2 worker chạy song song, thời gian chờ giảm còn 3.3%, tốc độ huấn luyện MobileNetV3 tăng từ 132 lên 274 ảnh mỗi giây, gấp 2.1 lần.

Thêm lên 4 worker lại không nhanh hơn, vì Colab chỉ có 2 vCPU: worker thừa chỉ tranh CPU với tiến trình chính.

Kết quả này khớp với nghiên cứu Data Stalls của Mohan và cộng sự, PVLDB 2021: training thường bị nghẽn ở khâu nạp dữ liệu chứ không phải ở GPU.

### Slide 19. Điểm nghẽn ngoài mô hình và định luật Amdahl
*TV3 · khoảng 70 giây.*

Câu hỏi CH6: trong ứng dụng thật, các bước ngoài mô hình chiếm bao nhiêu. Nhóm đo cả pipeline: tiền xử lý, chạy mô hình, hậu xử lý.

Dòng tô màu là YOLOv8n trên T4: tổng 10.68 mili giây cho một khung hình, trong đó tiền xử lý 3.23 và NMS 1.15. Cộng lại, 41% thời gian nằm ngoài mô hình.

So với chạy trên Colab CPU: mô hình mất 163.6 mili giây, chiếm 94%, phần ngoài mô hình chỉ 6%. Lên GPU, mô hình nhanh hơn khoảng 26 lần, nhưng tiền xử lý và NMS vẫn chạy trên CPU, nên chúng lộ ra thành điểm nghẽn.

Đây chính là định luật Amdahl: dù mô hình có nhanh vô hạn, pipeline YOLOv8n trên T4 cũng chỉ nhanh hơn tối đa khoảng 2.4 lần, vì 4.38 mili giây tiền và hậu xử lý vẫn còn đó.

Kết quả khớp với nghiên cứu AI Tax của Richins và cộng sự, HPCA 2020: trong ứng dụng thật, dưới 60% độ trễ là phần AI. Nhóm đo được mô hình chiếm 59%.

Với BERT-base trên T4, tokenize chiếm 14%; XGBoost gần như toàn bộ là dự đoán.

Bài học: tăng tốc riêng mô hình nhanh chóng hết tác dụng; phải tối ưu cả tiền và hậu xử lý.

### Slide 20. Từ số đo đến gợi ý sửa: 19 luật chẩn đoán và A/B
*TV3 · khoảng 60 giây.*

Số đo chỉ có ích khi người dùng biết phải làm gì. VisionProf có bộ 19 luật chẩn đoán, chia 4 nhóm: RA về bộ nhớ, RB về tính toán, RC về nạp dữ liệu, RD về kiến trúc. Mỗi luật trả về một trong 4 trạng thái: PASS là ổn, WARN là cần chú ý, FAIL là có vấn đề, SKIPPED là thiếu dữ liệu để đánh giá.

Ví dụ luật RC-01: nếu thời gian chờ dữ liệu vượt 20% một bước training thì báo FAIL. Thí nghiệm nạp dữ liệu vừa rồi với num_workers bằng 0 có 62.6%, vượt ngưỡng 3 lần. Mỗi luật kèm theo gợi ý sửa code cụ thể, ở đây là cấu hình DataLoader với 2 worker và pin_memory. Gợi ý hiện ngay trên tab Chẩn đoán của dashboard.

Bên phải là phần so sánh A/B, ví dụ trước và sau khi tối ưu. Nhóm dùng kiểm định Mann-Whitney U vì độ trễ không có phân phối chuẩn, và Cliff's delta để đo độ lớn khác biệt. Speedup được tính trên tổng thời gian mỗi vòng; phiên bản đầu từng lấy trung bình speedup của từng layer và bị các layer siêu nhỏ làm lệch, nhóm đã sửa.

### Slide 21. Dashboard VisionProf trong 90 giây
*TV3 · 90 giây, để video tự chạy, nói kèm theo các mốc.*

0:00, tab Layer: chọn ResNet-50, xem tỷ trọng thời gian theo loại layer và biểu đồ Roofline của bạn TV1.

0:25, tab Model: so CPU với GPU, đổi batch 1, 8, 32 để thấy thông lượng tăng.

0:50, tab Training và Pipeline: tách forward, backward, optimizer; và pipeline YOLOv8n với 41% ngoài mô hình.

1:15, tab Chẩn đoán và A/B: các luật PASS, WARN, FAIL kèm gợi ý sửa code; so sánh hai lần chạy có kiểm định.

Dashboard chạy sẵn trên máy để thầy có thể yêu cầu demo trực tiếp trong phần hỏi đáp.

### Slide 22. 6 câu hỏi, 6 câu trả lời từ số đo
*TV3 · khoảng 70 giây.*

Tổng kết 6 câu hỏi nghiên cứu.

CH1: thời gian tập trung ở Conv với CNN, ở Linear và Attention với Transformer; lên GPU, layer nhỏ chiếm tỷ trọng lớn hơn.

CH2: công cụ đo đủ chính xác để phân rã, sai lệch tổng dưới 1% trên CPU và dưới 8% trên T4; nhưng overhead trên GPU lớn nên tổng thời gian lấy từ mức Model.

CH3: ít tham số chưa chắc nhanh hơn; trên T4 ở batch 1, ResNet-50 và MobileNetV3 cùng khoảng 6 mili giây, và thứ hạng thay đổi theo phần cứng.

CH4: tăng batch từ 1 lên 32 tăng thông lượng 1.3 đến 8.8 lần tuỳ mô hình; FP16 luôn giảm VRAM nhưng chỉ tăng tốc Transformer.

CH5: backward bằng 1.4 đến 2.2 lần forward, VRAM training gấp khoảng 3 lần; thiếu worker thì GPU chờ dữ liệu tới 63%.

CH6: với YOLOv8n trên T4, 41% thời gian nằm ngoài mô hình.

Nếu gom lại thành một câu: trên GPU, thứ làm chậm thường không phải phép tính lớn, mà là những thứ nhỏ xung quanh nó. Đó là lý do cần profiling ở cả 3 mức.

### Slide 23. Hạn chế và hướng phát triển
*TV3 · khoảng 50 giây.*

Nhóm nói rõ các hạn chế. Một: không đo đến kernel GPU và không dùng hardware counter; đó là thế mạnh của XSP và PRoof mà bạn TV1 đã nêu. Hai: khoảng 4 đến 10% thời gian thuộc các phép toán dạng hàm, không gán được cho layer nào. Ba: chỉ có một GPU là T4, chạy PyTorch eager, và hai máy CPU khác nhau nên không so số CPU chéo. Bốn: mô hình dùng trọng số khởi tạo ngẫu nhiên, đủ cho đo tốc độ và bộ nhớ, nhưng không dùng để nói về độ chính xác.

Hướng phát triển cho cuối kỳ: so sánh torch.compile với eager; bắt các phép toán dạng hàm bằng TorchDispatchMode để giảm phần thời gian không thuộc layer nào; làm thí nghiệm ablation về số vòng khởi động và overhead; đo bổ sung VRAM training của DistilBERT và thêm phần cứng.

Mã nguồn và dữ liệu đo nằm trong repo của nhóm. Em xin cảm ơn thầy và các bạn đã lắng nghe, nhóm sẵn sàng nhận câu hỏi.

## Phụ lục

### Slide 24. Tài liệu tham khảo
*Slide phụ lục, không trình bày. Mở khi hội đồng hỏi về nguồn.*
