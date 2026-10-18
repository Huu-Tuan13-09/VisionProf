# Dàn ý slide trình bày (24 phút · 27 slide · 3 thành viên)

> Cấu trúc theo bài báo: Đặt vấn đề → Nghiên cứu liên quan → Thiết kế → Thực nghiệm → Kết quả → Kết luận.
> Mọi con số dưới đây lấy từ `results/` và báo cáo của từng người. Ô **[CHỜ SỐ ĐO]** là chỗ chưa có số liệu thật.
> Paper: `docs/research/TONG_HOP_NGHIEN_CUU.md` · Hình paper: `docs/research/figures/`.

## Quy tắc chung

- Mỗi câu hỏi nghiên cứu (CH): **1 bảng + 1 hình + 3 câu kết luận**, đúng như kế hoạch Mục 8.5.
- Hình lấy từ paper phải có **dòng nguồn ngay dưới hình**, ví dụ "Nguồn: Li et al., XSP, IPDPS 2020".
- Mỗi màu một người: 🟩 TV3 · 🟧 TV1 · 🟦 TV2. Góc slide ghi tên người trình bày.
- Mỗi người kết thúc bằng **một câu chuyển** sang người tiếp theo (ghi ở cuối mỗi phần).

## Tổng quan thời gian

| Phần | Người | Slide | Thời gian |
| :--- | :-: | :-: | :-: |
| A. Mở đầu, đặt vấn đề | 🟩 TV3 | 1–3 | 2' |
| B. Nghiên cứu liên quan | 🟧 TV1 | 4–7 | 4' |
| C. Kiến trúc & thiết lập thực nghiệm | 🟦 TV2 | 8–9 | 1.5' |
| D. Mức Layer: CH1, CH2 | 🟧 TV1 | 10–14 | 4.5' |
| E. Mức Model: CH3, CH4 | 🟦 TV2 | 15–19 | 4.5' |
| F. Mức Giai đoạn: CH5, CH6 + chẩn đoán | 🟩 TV3 | 20–24 | 4' |
| G. Demo, kết luận, tài liệu tham khảo | 🟩 TV3 | 25–27 | 3.5' |
| **Tổng** | TV1 8.5' · TV2 6' · TV3 9.5' | **27** | **24'** |

---

## 🟩 TV3: Phần A. Mở đầu (2')

### Slide 1. Tiêu đề
- "VisionProf: Phân tích hiệu năng mô hình học sâu ở 3 mức: Layer, Model, Giai đoạn". Tên 3 thành viên, môn học, giảng viên.

### Slide 2. Đặt vấn đề: Evaluation, Benchmarking hay Profiling?
- Bảng 3 cột: **Evaluation** (mô hình đúng bao nhiêu: accuracy) · **Benchmarking** (chạy nhanh bao nhiêu: latency, throughput) · **Profiling** (**thời gian và bộ nhớ đi đâu**).
- Câu chốt: đề tài là **profiling**, nên nhóm **không** đo accuracy, mAP.
- *Lý do có slide này:* tránh bị nhận xét lệch đề như nhóm trước (họ đo cả accuracy, F1).

### Slide 3. Câu hỏi nghiên cứu & 3 mức đo
- Hình 3 mức (Layer / Model / Giai đoạn) + 6 câu hỏi CH1–CH6, mỗi mức 2 câu, ghi rõ ai phụ trách.
- 8 mô hình (CV, NLP, Tabular) × 2 máy (laptop CPU, Colab T4).
- **Chuyển:** "Trước khi vào thiết kế, TV1 trình bày các nghiên cứu đi trước và khoảng trống nhóm muốn lấp."

---

## 🟧 TV1: Phần B. Nghiên cứu liên quan (4')

### Slide 4. Profiling học sâu: nhìn ở nhiều mức
- Hình: `docs/research/figures/06_mlmodelscope_fig1_L1_L5.png`. Đánh dấu L1 (Model + pipeline) và L2–L3 (Layer) là phần nhóm đo.
- Ý chính: hiệu năng phụ thuộc cả chồng phần mềm, phần cứng (model → framework → layer → thư viện → phần cứng); mỗi công cụ thường chỉ nhìn một tầng.
- Nguồn: Dakkak et al., MLModelScope, arXiv:2002.08295.

### Slide 5. Mức Layer: hiệu suất và cái giá của việc đo
- Hình trái: `01_roofline_fig1.png`, hiệu năng đạt được = min(đỉnh tính toán, băng thông × FLOP/byte).
- Hình phải: `04_xsp_fig2_leveled_overhead.png`, **đo càng sâu thì overhead của chính công cụ đo càng lớn**; XSP đo lần lượt từng mức để khỏi sai.
- Câu chốt: "Vì vậy nhóm dùng Roofline cho từng layer, và **tự kiểm chứng công cụ đo** ở TN2."
- Nguồn: Williams et al., CACM 2009 · Li et al., XSP, IPDPS 2020.

### Slide 6. Điểm nghẽn thường nằm ngoài mô hình
- Hình trái: `11_aitax_fig6_latency_breakdown.png`, ứng dụng thật: **dưới 60%** độ trễ là AI.
- Hình phải: `13_datastalls_fig1_pipeline.png`, khi training, GPU thường **phải chờ dữ liệu**.
- Câu chốt: "Vì vậy nhóm đo thêm mức Giai đoạn: training (TN6) và pipeline thật (TN7)."
- Nguồn: Richins et al., HPCA 2020 · Mohan et al., PVLDB 2021.

### Slide 7. So sánh & khoảng trống
- Bảng ở `docs/research/TONG_HOP_NGHIEN_CUU.md` Mục 3 (XSP, PRoof, MLPerf, MLModelScope, AI Tax, Data Stalls, **VisionProf**).
- Khoảng trống: gom **cả 3 mức** vào một công cụ nhẹ cho PyTorch; đo cả **suy luận, training, pipeline**; **CPU laptop với GPU T4**; **tự kiểm chứng** độ chính xác.
- Nói luôn hạn chế: không đo kernel GPU, không dùng hardware counter (đó là thế mạnh của XSP, PRoof).
- **Chuyển:** "TV2 sẽ trình bày kiến trúc chung và cách nhóm thiết lập thực nghiệm."

---

## 🟦 TV2: Phần C. Kiến trúc & thiết lập thực nghiệm (1.5')

### Slide 8. Kiến trúc VisionProf
- Sơ đồ: `ModelAdapter` (8 mô hình) → `run_experiment.py` → 3 plugin `layer` / model / `phase` → `results/{máy}/{model}/{TN}/` → Dashboard 8 tab + bộ chẩn đoán 19 luật.
- Ý chính: gắn bộ đo **không sửa mã mô hình**; đo latency sạch trước, gắn hook layer ở lượt riêng sau.

### Slide 9. Thiết lập thực nghiệm
- Bảng 8 mô hình (họ, tham số, input) · 2 máy (laptop CPU, Colab T4).
- Quy tắc đo: khởi động 10 vòng, đo 50 vòng; đồng bộ GPU trước khi bấm giờ; báo **P50 / P90 / P99** thay vì chỉ trung bình; tự lưu `env.json` (máy, phiên bản thư viện).
- 7 thực nghiệm TN1–TN7 ứng với 6 câu hỏi.
- **Chuyển:** "Bắt đầu từ mức sâu nhất: TV1 với mức Layer."

---

## 🟧 TV1: Phần D. Mức Layer: CH1, CH2 (4.5')

### Slide 10. Mức Layer làm gì?
- 5 dòng code gắn profiler (`attach` / `start` / `stop` / `save`), không sửa mô hình.
- Đo cho từng layer: thời gian, bộ nhớ output, FLOPs, % hiệu suất (Roofline).

### Slide 11. Đo đúng trên GPU không dễ
- GPU chạy **bất đồng bộ**: đo bằng đồng hồ CPU chỉ thấy thời gian *gửi lệnh*.
- Cách làm: CUDA Event ghi trong hook, **chỉ đồng bộ 1 lần** cuối vòng; "lấp hàng đợi" để GPU chạy liền mạch; module dùng lại nhiều lần thì quản lý bằng ngăn xếp.

### Slide 12. CH1: Thời gian tập trung ở loại layer nào?
- Hình: `docs/tv1/figures/fig1_type_share.png`.
- Bảng: ResNet-50: Conv 85% (CPU) / 83% (T4) · MobileNetV3: Conv 73% → 47%, BatchNorm 14% → 25% · BERT-base: Linear 95% · ViT-B/16: Linear + Attention 96%.
- Kết luận: CPU dồn vào Conv (CNN 73–85%) và Linear + Attention (Transformer 95–96%).

### Slide 13. CH1: GPU giúp được bao nhiêu, và vì sao?
- Hình: `fig2_gpu_speedup.png` + `fig3_roofline.png`.
- Transformer tăng tốc nhiều nhất (~9×), CNN 3–6×; các layer lớn ở batch 1 chỉ đạt khoảng 30–55% sức tối đa của máy.
- Giải thích: phép toán element-wise (BatchNorm, ReLU) ít FLOPs nhưng vẫn tốn thời gian trên GPU, nên tỷ trọng tăng ở mô hình nhẹ (ShuffleNet V2, Ma et al., ECCV 2018, nguyên tắc G4).

### Slide 14. CH2: Công cụ đo có đúng không?
- Hình: `fig4_tn2_error.png`. Bảng TN2:

  | | ResNet-50 CPU | BERT CPU | ResNet-50 T4 | BERT T4 |
  | :--- | :-: | :-: | :-: | :-: |
  | Sai lệch so với `torch.profiler` | 0.35% | 0.27% | 7.5% | 0.8% |
  | Overhead | 11.3% | 5.2% | 166% | 109% |

- Kết luận: đủ chính xác (< 1% CPU, < 8% GPU); overhead trên GPU ở batch 1 lớn, nên mức Layer **chỉ dùng để phân rã**, tổng thời gian để mức Model đo (đúng nhận xét của XSP ở Slide 5).
- **Chuyển:** "Đo tổng thời gian cả mô hình là phần của TV2."

---

## 🟦 TV2: Phần E. Mức Model: CH3, CH4 (4.5')

> ⚠️ Cả phần này **chờ số đo T4 thật** (`docs/tv2/YEU_CAU_DO_LAI_TN3_TN5.md`). Không dùng số trong
> `docs/tv2/BAO_CAO_TV2.md` hiện tại. Số **laptop CPU** đã đo thật, dùng được ngay.

### Slide 15. Mức Model đo gì?
- 12 chỉ số: P50 / P90 / P95 / P99, tỷ lệ P99/P50, throughput, cold-start, VRAM đỉnh, RAM, CPU% / GPU%.
- Vì sao báo phân vị: MLPerf Inference dùng độ trễ **phân vị 90** (single-stream) và ràng buộc **phân vị 99** (server). Nguồn: Reddi et al., ISCA 2020; hình `10_mlperf_fig4_scenarios.png` nếu còn chỗ.

### Slide 16. CH3: Bảng 8 mô hình × 2 máy
- Bảng P50, throughput, VRAM. Cột laptop CPU: lấy từ `results/cpu_laptop/*/TN3/`. Cột T4: **[CHỜ SỐ ĐO]**.

### Slide 17. CH3: Ít tham số có chắc nhanh hơn? Thứ hạng có đảo?
- Hình: `docs/tv2/figures/fig_ch3_rank_shift.png` (vẽ lại sau khi có số T4). Kết luận **[CHỜ SỐ ĐO]**.
- Nếu số thật không cho thấy đảo thứ hạng thì ghi đúng như đo được: đó vẫn là kết luận hợp lệ.

### Slide 18. CH4: Tăng batch
- Hình: `fig_ch4_batch_scaling.png`, throughput theo batch 1 / 8 / 32 trên CPU và T4; NLP thêm độ dài câu 64 / 128 / 256. **[CHỜ SỐ ĐO]**

### Slide 19. CH4: FP16 trên T4
- Hình: `fig_ch4_fp16.png`, tăng tốc, VRAM giảm, cosine similarity FP32 so với FP16. **[CHỜ SỐ ĐO]**
- **Chuyển:** "Mô hình chỉ là một phần. TV3 sẽ cho thấy training và quy trình thật tốn thời gian ở đâu."

---

## 🟩 TV3: Phần F. Mức Giai đoạn: CH5, CH6 (4')

### Slide 20. Mức Giai đoạn đo gì?
- `with timer.phase("backward"):` có đồng bộ GPU ở biên mỗi giai đoạn; `DataLoaderSentinel` đo thời gian chờ dữ liệu.

### Slide 21. CH5: Backward tốn gấp mấy lần forward? Bộ nhớ tăng bao nhiêu?
- Hình: `docs/tv3/figures/fig_ch5_backward_forward_ratio.png` + `fig_ch5_memory_train_vs_infer.png`.
- Backward / forward trên T4: DistilBERT 2.21× · ResNet-50 1.82× · MobileNetV3 1.65× · MLP 1.39×.
- VRAM training / suy luận: ResNet-50 420 → 1.139 MB (2.71×) · MobileNetV3 140 → 460 MB (3.28×).
- Giải thích bằng paper: feature map chiếm 62–89% bộ nhớ khi training (TBD, Zhu et al., IISWC 2018).

### Slide 22. CH5: Nạp dữ liệu có làm GPU phải chờ?
- ⚠️ **Chưa có số liệu.** TN6 hiện chỉ chạy `num_workers = 2` và chưa đo giai đoạn nạp dữ liệu. Cần chạy
  `scripts/run_tn6_training.py --workers 0`, `2`, `4` và ghi thời gian chờ dữ liệu, rồi mới làm slide này.
  Nếu không kịp, **bỏ slide** và bỏ câu "num_workers=0 gây nghẽn" khỏi slide TV3.
- Paper đối chiếu: Data Stalls (Mohan et al., PVLDB 2021).

### Slide 23. CH6: Ngoài mô hình chiếm bao nhiêu %?
- Hình: `fig_ch6_pipeline_breakdown.png` + `fig_ch6_cpu_vs_gpu_pipeline.png`.
- YOLOv8n trên T4: tiền xử lý 3.23 ms (30.3%) + NMS 1.15 ms (10.7%) = **41% ngoài mô hình**. Trên Colab CPU: mô hình chiếm 94%.
- Lên GPU, mô hình nhanh hơn ~26× (163.6 → 6.3 ms), nên phần ngoài mô hình lộ ra (định luật Amdahl).
- BERT-base trên T4: tokenize chiếm 13.5%.
- Đối chiếu: AI Tax đo dưới 60% là AI; nhóm đo 59% là mô hình. Hai kết quả cùng chiều.

### Slide 24. Chẩn đoán tự động & so sánh A/B
- 19 luật chia 4 nhóm (bộ nhớ, tính toán, nạp dữ liệu, kiến trúc), trạng thái PASS / WARN / FAIL / SKIPPED.
- A/B: Mann-Whitney U + Cliff's delta, speedup tính trên tổng thời gian mỗi vòng.

---

## 🟩 TV3: Phần G. Demo & kết luận (3.5')

### Slide 25. Demo dashboard (2')
- `streamlit run src/dashboard/app.py`. Kịch bản: tab Layer (ViT trên T4) → Roofline → Training → Pipeline YOLO → Chẩn đoán.
- Dự phòng: quay sẵn video demo nếu máy hoặc mạng lỗi.

### Slide 26. Kết luận, hạn chế, hướng phát triển
- 6 câu trả lời ngắn cho CH1–CH6 (một dòng mỗi câu; CH3, CH4 điền sau khi có số T4).
- Hạn chế: không đo kernel GPU / hardware counter; overhead mức Layer lớn trên GPU ở batch 1; 1 GPU; chỉ PyTorch eager.
- Hướng phát triển: `torch.compile`, đo kernel bằng `torch.profiler`, mở rộng mô hình.

### Slide 27. Tài liệu tham khảo
- Danh sách IEEE trong `docs/research/TONG_HOP_NGHIEN_CUU.md`, thêm ShuffleNet V2 (Ma et al., ECCV 2018).
- Mời câu hỏi.

---

## Câu hỏi dễ bị hỏi (mỗi người chuẩn bị phần mình)

- 🟧 **TV1:** Vì sao overhead trên GPU lớn mà vẫn tin số đo được? (Sai lệch so với `torch.profiler` vẫn nhỏ; overhead nằm ngoài khoảng đo của layer.) · Khác gì `torch.profiler`? (Gộp theo layer của mô hình, có FLOPs và Roofline, không cần đọc file trace.)
- 🟦 **TV2:** Vì sao đo trên Colab mà không dùng GPU riêng? (Để lặp lại được.) · P99 ổn định đến đâu với 50 vòng đo?
- 🟩 **TV3:** Vì sao CNN có backward / forward thấp hơn 2×? · 41% ngoài mô hình có còn đúng khi batch lớn không?

## Việc phải xong trước khi dựng slide

| Việc | Ai | Chặn slide |
| :--- | :-: | :-: |
| Đo thật TN3–TN5 trên T4, viết lại CH3, CH4 | 🟦 TV2 | 16–19, 26 |
| Chạy TN6 với `num_workers` 0 / 2 / 4, đo giai đoạn nạp dữ liệu (hoặc bỏ Slide 22) | 🟩 TV3 | 22 |
| Tải PDF PRoof bằng trình duyệt nếu muốn có hình PRoof ở Slide 7 | 🟧 TV1 | không bắt buộc |
| Chọn công cụ làm slide + mẫu chung (màu theo người) | Cả nhóm | tất cả |
