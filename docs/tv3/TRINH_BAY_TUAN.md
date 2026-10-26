# Hướng dẫn trình bày · Phạm Hữu Tuân

> **Phần:** Mức Giai đoạn (CH5, CH6), chẩn đoán, A/B và tổng kết · **Màu slide:** 🟩 xanh lá · **Slide:** 22–31 (10 slide) · **Thời lượng:** khoảng 7.3 phút
> Lời thoại lấy từ ghi chú người nói trong bộ slide; số liệu đã đối chiếu `results/`. Không cần đọc thuộc: nhìn cột "Ý chính" là đủ nhắc.

## Vị trí trong buổi trình bày

- Trí chuyển sang ở cuối slide 21: *"Bạn Tuân sẽ đưa chúng ta đến training và pipeline thật."*
- Bạn kết thúc buổi trình bày (slide 31) và điều phối phần hỏi đáp.
- Chạy sẵn dashboard trước giờ trình bày: `docs/HUONG_DAN_CHAY_DASHBOARD.md`.
- Thứ tự cả nhóm: Trí (1–4) → Quân (5–14) → Trí (15–21) → Tuân (22–31); slide 32 là phụ lục tài liệu tham khảo.

## Tóm tắt các slide

| Slide | Tiêu đề | Thời gian | Ý chính |
| :-: | :--- | :-: | :--- |
| 22 | Training tốn thời gian và bộ nhớ ở đâu? | 60 giây | Backward / forward 1.39–2.21×; VRAM training ×2.7–3.3; TBD: feature map 62–89% bộ nhớ. |
| 23 | Nạp dữ liệu có làm GPU phải chờ? | 70 giây | 0 worker: chờ dữ liệu 62.6% (MobileNetV3), 30.1% (ResNet-50); 2 worker: 3.3%, 132 → 274 ảnh/s. |
| 24 | Tab Training: GPU chờ dữ liệu bao lâu | 15 giây | Bảng DataLoader trên dashboard. |
| 25 | Điểm nghẽn ngoài mô hình và định luật Amdahl | 70 giây | YOLOv8n trên T4: 3.23 + 1.15 ms ngoài mô hình = 41%; trên CPU 6%; Amdahl tối đa 2.4×; AI Tax < 60%. |
| 26 | Tab Pipeline: thời gian ngoài mô hình | 15 giây | Cột tím 41% vượt đường ngưỡng 30%. |
| 27 | Từ số đo đến gợi ý sửa: 19 luật chẩn đoán và A/B | 60 giây | 19 luật: RA 5, RB 5, RC 4, RD 5; PASS / WARN / FAIL / SKIPPED; RC-01 > 20%; A/B: Mann-Whitney U, Cliff's delta. |
| 28 | Tab Chẩn đoán: từ số đo đến gợi ý sửa | 15 giây | MobileNetV3 trên T4: 8 PASS, 4 WARN, 1 FAIL; RB-04 = 13.49 ms/GFLOP > ngưỡng 2.0. |
| 29 | Tab A/B: so sánh hai cấu hình có kiểm định | 15 giây | ViT FP32 so với FP16: 2.62×, Cliff's delta 1.0, p ≈ 7·10⁻¹⁸. |
| 30 | 6 câu hỏi, 6 câu trả lời từ số đo | 70 giây | 6 câu trả lời; thông điệp: trên GPU, thứ làm chậm là phần nhỏ xung quanh phép tính lớn. |
| 31 | Hạn chế và hướng phát triển | 50 giây | 4 hạn chế, 4 hướng cuối kỳ; cảm ơn và mời câu hỏi. |

## Lời thoại từng slide

### Slide 22. Training tốn thời gian và bộ nhớ ở đâu?
*khoảng 60 giây · Ý chính: Backward / forward 1.39–2.21×; VRAM training ×2.7–3.3; TBD: feature map 62–89% bộ nhớ.*

Cảm ơn bạn Trí. Em chuyển sang mức Giai đoạn, bắt đầu với training. Câu hỏi CH5: một bước training tốn thời gian và bộ nhớ ở đâu.

Công cụ là PhaseTimer: chỉ cần bọc từng giai đoạn trong một khối with timer.measure, như đoạn code bên dưới. Ở biên mỗi giai đoạn, PhaseTimer đồng bộ GPU để đo đúng thời gian, và không phải sửa mô hình.

Kết quả trên T4, batch 8: backward bằng 1.4 đến 2.2 lần forward. DistilBERT 2.21 lần, gần đúng mức lý thuyết 2 lần, vì mỗi phép nhân ma trận khi lan truyền ngược phải tính gradient cho cả đầu vào lẫn trọng số. ResNet-50 1.82, MobileNetV3 1.65, MLP 1.39.

Về bộ nhớ: VRAM khi training gấp 2.7 lần suy luận với ResNet-50 và 3.3 lần với MobileNetV3. Lý do là khi training phải giữ lại activation của mọi layer để dùng cho backward. Điều này khớp với nghiên cứu TBD của Zhu và cộng sự, IISWC 2018: feature map chiếm 62 đến 89% bộ nhớ khi training, không phải trọng số. DistilBERT nhóm chưa đo bộ nhớ, sẽ bổ sung cuối kỳ.

### Slide 23. Nạp dữ liệu có làm GPU phải chờ?
*khoảng 70 giây · Ý chính: 0 worker: chờ dữ liệu 62.6% (MobileNetV3), 30.1% (ResNet-50); 2 worker: 3.3%, 132 → 274 ảnh/s.*

Phần thứ hai của CH5: trong training thật, GPU có phải ngồi chờ dữ liệu không. Nhóm dùng DataLoader thật: mỗi ảnh là file JPEG được giải mã và augmentation trên CPU, giống quy trình ImageNet, batch 32, trên Colab có 2 vCPU.

Biểu đồ tách mỗi bước thành 4 phần: phần xám là chờ dữ liệu, rồi forward, backward, optimizer.

Với num_workers bằng 0, tức nạp dữ liệu ngay trên tiến trình chính, MobileNetV3 mất 62.6% thời gian chỉ để chờ dữ liệu. GPU tính rất nhanh nhưng phải đợi CPU giải mã ảnh. ResNet-50 nặng hơn nên tỷ lệ chờ thấp hơn, 30.1%.

Chỉ cần 2 worker chạy song song, thời gian chờ giảm còn 3.3%, tốc độ huấn luyện MobileNetV3 tăng từ 132 lên 274 ảnh mỗi giây, gấp 2.1 lần.

Thêm lên 4 worker lại không nhanh hơn, vì Colab chỉ có 2 vCPU: worker thừa chỉ tranh CPU với tiến trình chính.

Kết quả này khớp với nghiên cứu Data Stalls của Mohan và cộng sự, PVLDB 2021: training thường bị nghẽn ở khâu nạp dữ liệu chứ không phải ở GPU.

### Slide 24. Tab Training: GPU chờ dữ liệu bao lâu
*khoảng 15 giây · Ý chính: Bảng DataLoader trên dashboard.*

Tab Training có thêm tab con DataLoader và Workers. Bảng này là thí nghiệm nạp dữ liệu em vừa trình bày: không có worker thì 62.6% thời gian mỗi bước là chờ dữ liệu; 2 worker chỉ còn 3.3% và tốc độ tăng từ 132 lên 274 ảnh mỗi giây; 4 worker không nhanh hơn vì Colab chỉ có 2 vCPU.

### Slide 25. Điểm nghẽn ngoài mô hình và định luật Amdahl
*khoảng 70 giây · Ý chính: YOLOv8n trên T4: 3.23 + 1.15 ms ngoài mô hình = 41%; trên CPU 6%; Amdahl tối đa 2.4×; AI Tax < 60%.*

Câu hỏi CH6: trong ứng dụng thật, các bước ngoài mô hình chiếm bao nhiêu. Nhóm đo cả pipeline: tiền xử lý, chạy mô hình, hậu xử lý.

Dòng tô màu là YOLOv8n trên T4: tổng 10.68 mili giây cho một khung hình, trong đó tiền xử lý 3.23 và NMS 1.15. Cộng lại, 41% thời gian nằm ngoài mô hình.

So với chạy trên Colab CPU: mô hình mất 163.6 mili giây, chiếm 94%, phần ngoài mô hình chỉ 6%. Lên GPU, mô hình nhanh hơn khoảng 26 lần, nhưng tiền xử lý và NMS vẫn chạy trên CPU, nên chúng lộ ra thành điểm nghẽn.

Đây chính là định luật Amdahl: dù mô hình có nhanh vô hạn, pipeline YOLOv8n trên T4 cũng chỉ nhanh hơn tối đa khoảng 2.4 lần, vì 4.38 mili giây tiền và hậu xử lý vẫn còn đó.

Kết quả khớp với nghiên cứu AI Tax của Richins và cộng sự, HPCA 2020: trong ứng dụng thật, dưới 60% độ trễ là phần AI. Nhóm đo được mô hình chiếm 59%.

Với BERT-base trên T4, tokenize chiếm 14%; XGBoost gần như toàn bộ là dự đoán.

Bài học: tăng tốc riêng mô hình nhanh chóng hết tác dụng; phải tối ưu cả tiền và hậu xử lý.

### Slide 26. Tab Pipeline: thời gian ngoài mô hình
*khoảng 15 giây · Ý chính: Cột tím 41% vượt đường ngưỡng 30%.*

Tab Pipeline cho thấy phần trăm thời gian nằm ngoài mô hình. Cột tím là YOLOv8n trên T4: 41%, vượt đường vàng là ngưỡng 30% của giả thuyết CH6. Cùng pipeline đó chạy trên CPU thì phần ngoài mô hình chỉ khoảng 5 đến 6%, vì mô hình chậm hơn nhiều. Đây là hiệu ứng Amdahl em vừa nói.

### Slide 27. Từ số đo đến gợi ý sửa: 19 luật chẩn đoán và A/B
*khoảng 60 giây · Ý chính: 19 luật: RA 5, RB 5, RC 4, RD 5; PASS / WARN / FAIL / SKIPPED; RC-01 > 20%; A/B: Mann-Whitney U, Cliff's delta.*

Số đo chỉ có ích khi người dùng biết phải làm gì. VisionProf có bộ 19 luật chẩn đoán, chia 4 nhóm: RA về bộ nhớ, RB về tính toán, RC về nạp dữ liệu, RD về kiến trúc. Mỗi luật trả về một trong 4 trạng thái: PASS là ổn, WARN là cần chú ý, FAIL là có vấn đề, SKIPPED là thiếu dữ liệu để đánh giá.

Ví dụ luật RC-01: nếu thời gian chờ dữ liệu vượt 20% một bước training thì báo FAIL. Thí nghiệm nạp dữ liệu vừa rồi với num_workers bằng 0 có 62.6%, vượt ngưỡng 3 lần. Mỗi luật kèm theo gợi ý sửa code cụ thể, ở đây là cấu hình DataLoader với 2 worker và pin_memory. Gợi ý hiện ngay trên tab Chẩn đoán của dashboard.

Bên phải là phần so sánh A/B, ví dụ trước và sau khi tối ưu. Nhóm dùng kiểm định Mann-Whitney U vì độ trễ không có phân phối chuẩn, và Cliff's delta để đo độ lớn khác biệt. Speedup được tính trên tổng thời gian mỗi vòng; phiên bản đầu từng lấy trung bình speedup của từng layer và bị các layer siêu nhỏ làm lệch, nhóm đã sửa.

### Slide 28. Tab Chẩn đoán: từ số đo đến gợi ý sửa
*khoảng 15 giây · Ý chính: MobileNetV3 trên T4: 8 PASS, 4 WARN, 1 FAIL; RB-04 = 13.49 ms/GFLOP > ngưỡng 2.0.*

Tab Chẩn đoán chạy 19 luật trên một lần đo bất kỳ. Ví dụ MobileNetV3 trên T4: 8 luật PASS, 4 WARN, 1 FAIL. Mỗi luật có triệu chứng và hành động đề xuất. Luật RB-04 báo 13.49 mili giây cho mỗi GFLOP, vượt ngưỡng 2.0: đúng với kết luận CH3 là MobileNetV3 ít phép tính nhưng không tận dụng được GPU ở batch 1.

### Slide 29. Tab A/B: so sánh hai cấu hình có kiểm định
*khoảng 15 giây · Ý chính: ViT FP32 so với FP16: 2.62×, Cliff's delta 1.0, p ≈ 7·10⁻¹⁸.*

Tab cuối là so sánh A/B. Chọn A là ViT FP32, B là ViT FP16 trên T4. Kết quả nhanh hơn 2.62 lần; Cliff's delta bằng 1.0 nghĩa là mọi lần chạy của B đều nhanh hơn A; p-value cỡ 10 mũ trừ 18 nên khác biệt chắc chắn có ý nghĩa.

Nếu được hỏi vì sao 2.62 khác 2.81 ở slide FP16: 2.81 là tỷ số trung vị P50, còn 2.62 là tỷ số độ trễ trung bình của 50 vòng đo.

Dashboard đang chạy sẵn trên máy, thầy có thể yêu cầu xem trực tiếp bất kỳ tab nào trong phần hỏi đáp.

### Slide 30. 6 câu hỏi, 6 câu trả lời từ số đo
*khoảng 70 giây · Ý chính: 6 câu trả lời; thông điệp: trên GPU, thứ làm chậm là phần nhỏ xung quanh phép tính lớn.*

Tổng kết 6 câu hỏi nghiên cứu.

CH1: thời gian tập trung ở Conv với CNN, ở Linear và Attention với Transformer; lên GPU, layer nhỏ chiếm tỷ trọng lớn hơn.

CH2: công cụ đo đủ chính xác để phân rã, sai lệch tổng dưới 1% trên CPU và dưới 8% trên T4; nhưng overhead trên GPU lớn nên tổng thời gian lấy từ mức Model.

CH3: ít tham số chưa chắc nhanh hơn; trên T4 ở batch 1, ResNet-50 và MobileNetV3 cùng khoảng 6 mili giây, và thứ hạng thay đổi theo phần cứng.

CH4: tăng batch từ 1 lên 32 tăng thông lượng 1.3 đến 8.8 lần tuỳ mô hình; FP16 luôn giảm VRAM nhưng chỉ tăng tốc Transformer.

CH5: backward bằng 1.4 đến 2.2 lần forward, VRAM training gấp khoảng 3 lần; thiếu worker thì GPU chờ dữ liệu tới 63%.

CH6: với YOLOv8n trên T4, 41% thời gian nằm ngoài mô hình.

Nếu gom lại thành một câu: trên GPU, thứ làm chậm thường không phải phép tính lớn, mà là những thứ nhỏ xung quanh nó. Đó là lý do cần profiling ở cả 3 mức.

### Slide 31. Hạn chế và hướng phát triển
*khoảng 50 giây · Ý chính: 4 hạn chế, 4 hướng cuối kỳ; cảm ơn và mời câu hỏi.*

Nhóm nói rõ các hạn chế. Một: không đo đến kernel GPU và không dùng hardware counter; đó là thế mạnh của XSP và PRoof mà bạn Quân đã nêu. Hai: khoảng 4 đến 10% thời gian thuộc các phép toán dạng hàm, không gán được cho layer nào. Ba: chỉ có một GPU là T4, chạy PyTorch eager, và hai máy CPU khác nhau nên không so số CPU chéo. Bốn: mô hình dùng trọng số khởi tạo ngẫu nhiên, đủ cho đo tốc độ và bộ nhớ, nhưng không dùng để nói về độ chính xác.

Hướng phát triển cho cuối kỳ: so sánh torch.compile với eager; bắt các phép toán dạng hàm bằng TorchDispatchMode để giảm phần thời gian không thuộc layer nào; làm thí nghiệm ablation về số vòng khởi động và overhead; đo bổ sung VRAM training của DistilBERT và thêm phần cứng.

Mã nguồn và dữ liệu đo nằm trong repo của nhóm. Em xin cảm ơn thầy và các bạn đã lắng nghe, nhóm sẵn sàng nhận câu hỏi.

## Câu hỏi có thể gặp

**H: Tiền xử lý và NMS chiếm 41% trên GPU thì khắc phục thế nào?**

Đ: Đưa giải mã, resize, chuẩn hoá lên GPU (NVIDIA DALI, torchvision trên tensor GPU) và chạy NMS trên GPU. Theo Amdahl, nếu không làm vậy thì pipeline YOLOv8n trên T4 nhanh hơn tối đa khoảng 2.4 lần dù mô hình nhanh đến đâu.

**H: Sao 4 worker không nhanh hơn 2?**

Đ: Colab chỉ có 2 vCPU; worker thừa tranh CPU với tiến trình chính nên forward còn chậm đi một chút.

**H: Sao backward của CNN chưa tới 2 lần forward?**

Đ: Lý thuyết khoảng 2 lần vì mỗi phép nhân phải tính gradient cho cả đầu vào lẫn trọng số. CNN thấp hơn (1.65–1.82×) vì cuDNN tối ưu kernel tích chập ngược và lớp đầu không cần gradient theo đầu vào; MLP rất nhỏ nên chi phí cố định chiếm nhiều (1.39×).

**H: Ngưỡng 20% của luật RC-01 ở đâu ra?**

Đ: Là ngưỡng heuristic, chọn theo tinh thần Data Stalls (PVLDB 2021). Ngưỡng có thể chỉnh; điều quan trọng là luật chỉ ra đúng chỗ nghẽn và gợi ý cách sửa. Thí nghiệm num_workers = 0 có 62.6%, vượt ngưỡng 3 lần.

**H: Cliff's delta bằng 1.0 nghĩa là gì?**

Đ: Mọi lần chạy của B đều nhanh hơn mọi lần chạy của A, hai phân phối tách hẳn. Kèm p ≈ 7·10⁻¹⁸ (Mann-Whitney U) nên khác biệt chắc chắn có ý nghĩa.

**H: Sao slide A/B ghi 2.62× mà slide FP16 ghi 2.81×?**

Đ: 2.81 là tỷ số trung vị P50; 2.62 là tỷ số độ trễ trung bình của 50 vòng đo.

**H: VRAM training của DistilBERT đâu?**

Đ: Chưa đo ở giữa kỳ; đã đưa vào hướng phát triển cuối kỳ.

## Trước giờ trình bày

- [ ] Đọc to một lượt, bấm giờ.
- [ ] Thuộc các con số trong cột "Ý chính".
- [ ] Tập câu chuyển phần với người nói trước và sau.
