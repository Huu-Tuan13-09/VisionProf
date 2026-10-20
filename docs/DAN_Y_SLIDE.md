# Dàn ý slide trình bày hoàn chỉnh (24 phút · 23 slide · 3 thành viên)

> Khung 22 slide của TV2, thêm phần **Nghiên cứu liên quan** (paper) theo nhận xét của thầy, và điền **số đo thật**.
> 📖 **Hướng dẫn trình bày từng người:** [`tv1/TRINH_BAY_QUAN.md`](tv1/TRINH_BAY_QUAN.md) · [`tv2/TRINH_BAY_TRI.md`](tv2/TRINH_BAY_TRI.md) · [`tv3/TRINH_BAY_TUAN.md`](tv3/TRINH_BAY_TUAN.md).
> Nguồn số liệu: `docs/KET_QUA_COLAB_GIUA_KY.md` (Colab T4, 18/10), báo cáo từng thành viên, `results/`.
> Paper và hình: `docs/research/TONG_HOP_NGHIEN_CUU.md`, `docs/research/figures/`. Hình lấy từ paper phải **ghi nguồn ngay dưới hình**.
> Quy tắc mỗi câu hỏi nghiên cứu: **1 bảng + 1 hình + 3 câu kết luận**.

---

## ⏱️ Tổng quan phân bổ thời gian & Phân vai (cập nhật 20/10)

> Bộ slide HTML 31 trang: mỗi người 10 slide; thời gian Trí > Quân > Tuân, chênh khoảng 1 phút. Mỗi người tự trình bày ảnh chụp dashboard của phần mình. Số slide dưới đây theo bộ HTML; các mục chi tiết bên dưới vẫn giữ đánh số dàn ý cũ.

| Phần | Người trình bày | Slide (HTML) | Thời lượng | Trọng tâm |
| :--- | :---: | :---: | :---: | :--- |
| **Mở đầu & Khung bài toán** | 🟦 **Trương Công Trí** | **1–3** | 2.5' | Định vị Profiling, 3 mức đo, sợi chỉ đỏ |
| **Nghiên cứu liên quan** | 🟧 **Võ Trần Anh Quân** | **4–7** | 3.0' | MLModelScope, Roofline, XSP, bảng so sánh |
| **Mức Layer + dashboard** | 🟧 **Võ Trần Anh Quân** | **8–13** | 5.2' | Đo GPU, CH1, CH2, tab Layer |
| **Mức Model + dashboard** | 🟦 **Trương Công Trí** | **14–20** | 6.7' | Đo sạch 2 pha, P50–P99, CH3, CH4, tab FP16 |
| **Giai đoạn, chẩn đoán, tổng kết + dashboard** | 🟩 **Phạm Hữu Tuân** | **21–30** | 7.3' | CH5, CH6, 19 luật, A/B, 4 tab dashboard, kết luận |
| **Tổng cộng** | Trí 9.2' · Quân 8.2' · Tuân 7.3' | **30 + phụ lục** | **~25'** | Q&A: mở dashboard thật (`docs/HUONG_DAN_CHAY_DASHBOARD.md`) |

--- | :---: | :---: | :---: | :--- |
| **Phần 1: Mở đầu & Khung bài toán** | 🟩 **TV3** | **1–3** | **2.5'** | Định vị Profiling, 3 mức đo, 2 mô hình xuyên suốt |
| **Phần 2: Nghiên cứu liên quan** | 🟧 **TV1** | **4–6** | **3.0'** | Các mức profiling, Roofline & overhead, so sánh & khoảng trống |
| **Phần 3: Mức Layer** | 🟧 **TV1** | **7–10** | **5.0'** | Đo GPU, CH1, CH2 (tự kiểm chứng TN2) |
| **Phần 4: Mức Model** | 🟦 **TV2** | **11–16** | **6.5'** | Đo sạch 2 pha, P50–P99, CH3, CH4 |
| **Phần 5: Mức Giai đoạn & Tổng kết** | 🟩 **TV3** | **17–23** | **7.0'** | CH5 (training + nạp dữ liệu), CH6, chẩn đoán, demo, kết luận |
| **Tổng cộng** | TV1 8' · TV2 6.5' · TV3 9.5' | **23** | **24.0'** | Dành 5–10' sau bài cho Q&A và demo trực tiếp |

---

## 🟩 TV3: Phần 1. Mở đầu & Khung bài toán (Slide 1–3 · 2.5')

### Slide 1. Tiêu đề & Giới thiệu đề tài
- **Bố cục:** Tiêu đề giữa; 3 khối tác giả theo màu (🟩 TV3, 🟧 TV1, 🟦 TV2), môn học, giảng viên.
- **Nội dung:** VisionProf: phân tích hiệu năng mô hình học sâu ở 3 mức (Layer, Model, Giai đoạn); công cụ nhẹ cho PyTorch, gắn vào mô hình không sửa mã nguồn.

### Slide 2. Đặt vấn đề: Evaluation, Benchmarking hay Profiling?
- **Bố cục:** Bảng 3 cột, cột Profiling đóng khung nổi bật.
- **Nội dung:**
  - *Evaluation:* mô hình dự đoán đúng bao nhiêu (Accuracy, mAP, F1) → **ngoài phạm vi đề tài**.
  - *Benchmarking:* mô hình chạy nhanh bao nhiêu (latency, throughput).
  - *Profiling:* **thời gian và bộ nhớ thực sự đi đâu**, nghẽn do tính toán hay do băng thông bộ nhớ.
- **Thông điệp chốt:** Đề tài tập trung vào **Profiling**.

### Slide 3. Khung 3 mức đo & Sợi chỉ đỏ
- **Bố cục:** Sơ đồ 3 mức xếp chồng, cạnh 2 mô hình theo dõi xuyên suốt.
- **Nội dung:** 🟧 Layer (TV1) → 🟦 Model (TV2) → 🟩 Giai đoạn (TV3), trả lời CH1–CH6; 8 mô hình (CV, NLP, Tabular) trên laptop CPU và Colab T4. Sợi chỉ đỏ: **ResNet-50** (CNN) đối đầu **ViT / DistilBERT** (Transformer).
- **Lời chuyển:** *"Trước khi vào thiết kế, TV1 trình bày các nghiên cứu đi trước và khoảng trống nhóm muốn lấp."*

---

## 🟧 TV1: Phần 2. Nghiên cứu liên quan (Slide 4–6 · 3.0')

### Slide 4. Profiling học sâu: nhìn ở nhiều mức
- **Bố cục:** Hình `docs/research/figures/06_mlmodelscope_fig1_L1_L5.png`, đánh dấu phần nhóm đo: L1 (Model + pipeline), L2–L3 (Layer).
- **Nội dung:** Hiệu năng phụ thuộc cả chồng phần mềm, phần cứng (model → framework → layer → thư viện → phần cứng). Các công cụ trước thường chỉ nhìn một tầng.
- **Nguồn:** Dakkak et al., MLModelScope, arXiv:2002.08295.

### Slide 5. Roofline & nghịch lý overhead
- **Bố cục:** Trái `01_roofline_fig1.png`; phải `04_xsp_fig2_leveled_overhead.png`.
- **Nội dung:**
  - *Roofline (Williams et al., CACM 2009):* hiệu năng đạt được = min(đỉnh tính toán, băng thông × FLOP/byte) → phân định compute-bound và memory-bound.
  - *XSP (Li et al., IPDPS 2020):* đo càng sâu thì overhead của chính công cụ đo càng lớn; XSP đo lần lượt từng mức để khỏi sai.
- **Thông điệp chốt:** Nhóm dùng Roofline cho từng layer và **tự kiểm chứng công cụ đo** (TN2); mức Layer dùng để phân rã tỷ lệ, tổng thời gian sạch giao cho mức Model.

### Slide 6. So sánh & khoảng trống
- **Bố cục:** Bảng so sánh ở `docs/research/TONG_HOP_NGHIEN_CUU.md` Mục 3 (XSP, PRoof, MLPerf, MLModelScope, AI Tax, Data Stalls, **VisionProf**).
- **Nội dung:**
  - Mỗi công trình đo **một** mức hoặc là hệ thống lớn (cụm máy, hardware counter). AI Tax: dưới 60% độ trễ ứng dụng thật là AI. Data Stalls: training thường phải chờ dữ liệu.
  - **Khoảng trống:** gom cả 3 mức vào một công cụ nhẹ cho PyTorch; đo suy luận, training và pipeline thật; CPU laptop và GPU T4; tự kiểm chứng độ chính xác.
  - Nói luôn hạn chế: không đo kernel GPU, không dùng hardware counter (thế mạnh của XSP, PRoof).
- **Lời chuyển:** *"Bắt đầu từ mức sâu nhất: mức Layer."*

---

## 🟧 TV1: Phần 3. Mức Layer (Slide 7–10 · 5.0')

### Slide 7. Thách thức: đo GPU không dễ
- **Bố cục:** Dòng thời gian CPU gửi lệnh bất đồng bộ / GPU thực thi; chỉ ra sai số khi dùng đồng hồ CPU.
- **Nội dung:** CPU chỉ xếp lệnh vào hàng đợi rồi chạy tiếp → đồng hồ CPU chỉ đo được thời gian gửi lệnh. Giải pháp: CUDA Event ghi trong hook, **đồng bộ 1 lần cuối vòng**, lấp hàng đợi để GPU chạy liền mạch, ngăn xếp cho layer dùng lại nhiều lần. Gắn vào mô hình bằng 5 dòng code, không sửa mô hình.

### Slide 8. CH1: Thời gian tập trung ở loại layer nào?
- **Bố cục:** `docs/tv1/figures/fig1_type_share.png` + bảng tỷ trọng.
- **Nội dung:**

  | | CPU laptop (x86) | Colab T4 |
  | :--- | :-: | :-: |
  | ResNet-50: Conv | 85% | 83% |
  | MobileNetV3: Conv / BatchNorm | 73% / 14% | 47% / 25% |
  | BERT-base: Linear | 95% | 95% |
  | ViT-B/16: Linear + Attention | 96% | 94% |

- **Thông điệp chốt:** CPU dồn vào Conv (CNN 73–85%) và Linear + Attention (Transformer 95–96%). Lên GPU, phép tính lớn được tăng tốc nên layer nhỏ (BatchNorm, activation) chiếm tỷ trọng lớn hơn, rõ nhất ở mô hình nhẹ.

### Slide 9. CH1: GPU giúp được bao nhiêu, và vì sao?
- **Bố cục:** `fig2_gpu_speedup.png` + `fig3_roofline.png`.
- **Nội dung:**
  - Tăng tốc T4 so với CPU (tổng thời gian các layer): ViT 9.5×, BERT-base 9.0×, DistilBERT 8.8× · ResNet-50 5.8×, YOLOv8n 5.5×, MobileNetV3 3.0× · MLP 1.0×.
  - Layer lớn ở batch 1 chỉ đạt khoảng 30–55% sức tối đa của máy (Roofline).
  - Phép toán element-wise ít FLOPs nhưng vẫn tốn thời gian trên GPU (ShuffleNet V2, Ma et al., ECCV 2018, nguyên tắc G4).
- **Thông điệp chốt:** Transformer (nhân ma trận lớn) tận dụng GPU tốt hơn CNN, nhất là CNN nhẹ.

### Slide 10. CH2: Công cụ đo có đúng không? + tổng kết mức Layer
- **Bố cục:** `fig4_tn2_error.png` + bảng.
- **Nội dung:**

  | | ResNet-50 CPU | BERT CPU | ResNet-50 T4 | BERT T4 |
  | :--- | :-: | :-: | :-: | :-: |
  | Sai lệch tổng so với `torch.profiler` | 0.35% | 0.27% | 7.5% | 0.8% |
  | Overhead | 11% | 5% | 166% | 109% |
  | Thời gian không thuộc layer nào | 3.9% | 9.2% | 9.0% | 10.3% |

  - Đo lại trên T4 bằng code hiện tại (18/10): sai lệch 7.8% và 0.04%, cùng kết quả.
  - Ngưỡng sàn CUDA event khoảng 6 µs: layer chỉ chạy vài µs (ReLU, BatchNorm ở batch 1) bị đo dư; chỉ dùng để so sánh tương đối.
- **Thông điệp chốt:** Đủ chính xác để phân rã (< 1% CPU, < 8% GPU). Overhead GPU ở batch 1 lớn (khoảng 110–210% tuỳ phiên) → **không** lấy tổng thời gian từ mức Layer; việc đó của mức Model.
- **Lời chuyển:** *"Đo tổng thời gian sạch của cả mô hình là phần của TV2."*

---

## 🟦 TV2: Phần 4. Mức Model (Slide 11–16 · 6.5')

### Slide 11. Kiến trúc mức Model & đo sạch 2 pha
- **Bố cục:** Sơ đồ `ModelAdapter` (8 mô hình) + luồng đo 2 pha.
- **Nội dung:** Pha 1 đo latency **không gắn hook**; pha 2 mới gắn bộ đo layer và tài nguyên. VRAM đỉnh lấy từ `torch.cuda.max_memory_allocated()`.
- **Thông điệp chốt:** Tách đo tốc độ khỏi đo chi tiết là điều kiện để có độ trễ sạch (khớp Slide 5, 10).

### Slide 12. Chuẩn đo & thiết lập thực nghiệm
- **Bố cục:** Phân phối độ trễ, đánh dấu P50, P90, P99.
- **Nội dung:**
  - MLPerf Inference (Reddi et al., ISCA 2020) dùng độ trễ **phân vị 90 / 99**, không dùng trung bình.
  - Khởi động rồi đo 50 vòng; tự lưu `env.json`.
  - ⚠️ "Laptop CPU" là **2 máy khác nhau**: laptop x86 của TV1 (TN1, TN2) và MacBook chip Apple (arm64) của TV2 (TN3, TN4). Không so số CPU giữa 2 máy.

### Slide 13. CH3: Ít tham số có chắc nhanh hơn? (batch 1, FP32)
- **Bố cục:** Bảng 8 mô hình × (T4, CPU Mac) + `fig_ch3_speedup.png`.
- **Nội dung:**

  | Mô hình | Tham số | T4 P50 | CPU Mac P50 |
  | :--- | :-: | :-: | :-: |
  | DistilBERT | 66 M | 5.3 ms | 18.7 ms |
  | ResNet-50 | 26 M | 6.0 ms | 22.5 ms |
  | MobileNetV3-L | 5.5 M | 6.0 ms | 176.3 ms |
  | YOLOv8n | 3.2 M | 7.5 ms | 61.8 ms |
  | BERT-base | 110 M | 8.6 ms | 38.0 ms |
  | ViT-B/16 | 87 M | 13.0 ms | 45.8 ms |
  | MLP / XGBoost | < 0.1 M | 0.28 / 1.56 ms | 0.04 / 0.13 ms |

- **Thông điệp chốt:**
  1. **Trên T4 ở batch 1, tham số và FLOPs không quyết định độ trễ:** ResNet-50 (8.2 GFLOPs) và MobileNetV3 (0.43 GFLOPs, ít hơn 19 lần) cùng khoảng 6 ms vì GPU bị giới hạn bởi chi phí gửi kernel.
  2. **Mô hình rất nhỏ chạy GPU còn chậm hơn CPU** (MLP 0.28 ms so với 0.04 ms).
  3. MobileNetV3 chậm bất thường trên CPU ARM của Mac (176 ms; trên laptop x86 của TV1 khoảng 12 ms) → kết quả phụ thuộc cả phần cứng lẫn mức tối ưu thư viện cho phần cứng đó.

### Slide 14. CH3: Thứ hạng thay đổi theo phần cứng
- **Bố cục:** `fig_ch3_rank_shift.png` (nhãn trái: CPU Mac).
- **Nội dung:** Thứ hạng CPU Mac → T4: MobileNetV3 8 → 5, YOLOv8n 7 → 6, BERT-base 5 → 7, ViT 6 → 8; DistilBERT, ResNet-50 giữ hạng 3, 4.
- **Thông điệp chốt:** Không có mô hình "nhanh nhất mọi nơi"; chọn mô hình phải đo trên phần cứng triển khai thật. Lưu ý phần lớn thay đổi đến từ MobileNetV3 trên Mac (Slide 13).

### Slide 15. CH4: Tăng batch & độ dài câu
- **Bố cục:** `fig_ch4_batch_scaling.png` + bảng NLP theo độ dài câu.
- **Nội dung:**
  - Throughput T4, batch 1 → 32: MobileNetV3 **8.8×** (170 → 1 499 ảnh/s), ResNet-50 2.2×, YOLOv8n 2.0×, DistilBERT 1.5×, BERT 1.4×, ViT 1.3× (bão hoà từ batch 8).
  - BERT-base batch 32: 127 ms (seq 64) → 228 ms (128) → 507 ms (256): mỗi lần gấp đôi độ dài, độ trễ tăng 1.8× rồi 2.2× (Attention tăng theo bình phương).
- **Thông điệp chốt:** Mô hình nhẹ cần batch lớn mới lấp đầy GPU; mô hình nặng bão hoà sớm. Với NLP cần giới hạn độ dài câu.

### Slide 16. CH4: FP16 trên T4 (batch 1)
- **Bố cục:** `fig_ch4_fp16.png`.
- **Nội dung:**

  | | ViT-B/16 | DistilBERT | BERT-base | ResNet-50 | MobileNetV3 | YOLOv8n |
  | :--- | :-: | :-: | :-: | :-: | :-: | :-: |
  | Tăng tốc FP16 | **2.81×** | 1.55× | 1.23× | **0.87×** | 0.85× | **0.77×** |
  | VRAM giảm | 45% | 45% | 46% | 43% | 38% | 49% |

  - Cosine FP16 / FP32 (cùng trọng số, cùng input): ≥ 0.999999 với ResNet-50, YOLOv8n, DistilBERT, BERT, MLP. ViT, MobileNetV3: không xác định vì trọng số khởi tạo ngẫu nhiên cho output bằng 0.
- **Thông điệp chốt:** FP16 **luôn giảm VRAM** nhưng **chỉ tăng tốc Transformer**; CNN ở batch 1 còn chậm hơn vì bị giới hạn bởi gửi kernel, không phải tính toán.
- **Lời chuyển:** *"Mô hình chỉ là một mắt xích. TV3 đưa chúng ta đến training và pipeline thật."*

---

## 🟩 TV3: Phần 5. Mức Giai đoạn, Chẩn đoán & Tổng kết (Slide 17–23 · 7.0')

### Slide 17. CH5 (1/2): Training tốn thời gian và bộ nhớ ở đâu?
- **Bố cục:** Code mẫu `with timer.measure("backward"):` + `fig_ch5_backward_forward_ratio.png` + `fig_ch5_memory_train_vs_infer.png`.
- **Nội dung:**
  - `PhaseTimer` đồng bộ GPU ở biên mỗi giai đoạn; không sửa mô hình.
  - Backward / forward trên T4: DistilBERT 2.21× · ResNet-50 1.82× · MobileNetV3 1.65× · MLP 1.39×.
  - VRAM training / suy luận: ResNet-50 420 → 1 139 MB (2.7×) · MobileNetV3 140 → 460 MB (3.3×).
  - Feature map chiếm 62–89% bộ nhớ khi training (TBD, Zhu et al., IISWC 2018).
- **Thông điệp chốt:** Backward khoảng 1.4–2.2× forward (gần mức lý thuyết 2×); training tốn bộ nhớ gấp 3 lần vì phải giữ activation.

### Slide 18. CH5 (2/2): Nạp dữ liệu có làm GPU phải chờ?
- **Bố cục:** `docs/tv3/figures/fig_ch5_dataloader_workers.png` (cột chồng chờ dữ liệu / forward / backward / optimizer theo `num_workers`, vẽ bằng `experiments/tv3_dataloader_figure.py`).
- **Nội dung:** DataLoader thật (ảnh JPEG giải mã + augmentation trên CPU), batch 32, Colab 2 vCPU:

  | | `num_workers=0` | `num_workers=2` | `num_workers=4` |
  | :--- | :-: | :-: | :-: |
  | MobileNetV3: % chờ dữ liệu / ảnh/s | **62.6%** / 132 | 3.3% / **274** | 3.2% / 267 |
  | ResNet-50: % chờ dữ liệu / ảnh/s | **30.1%** / 69 | 0.6% / **95** | 0.6% / 90 |

- **Thông điệp chốt:** Không có worker, GPU ngồi chờ dữ liệu tới 63% thời gian; 2 worker tăng gấp đôi tốc độ huấn luyện MobileNetV3. 4 worker không nhanh hơn vì Colab chỉ có 2 vCPU. Khớp Data Stalls (Mohan et al., PVLDB 2021).

### Slide 19. CH6: Điểm nghẽn ngoài mô hình & định luật Amdahl
- **Bố cục:** `fig_ch6_pipeline_breakdown.png` + `fig_ch6_cpu_vs_gpu_pipeline.png`.
- **Nội dung:**
  - YOLOv8n trên T4: tiền xử lý 3.23 ms (30.3%) + NMS 1.15 ms (10.7%) = **41% ngoài mô hình**. Trên Colab CPU: mô hình chiếm 94%.
  - Lên GPU, mô hình nhanh hơn khoảng 26× (163.6 → 6.3 ms) nên phần ngoài mô hình lộ ra. BERT-base trên T4: tokenize 13.5%.
  - Khớp AI Tax (Richins et al., HPCA 2020): dưới 60% độ trễ ứng dụng thật là AI; nhóm đo được 59% là mô hình.
- **Thông điệp chốt:** Tăng tốc riêng mô hình nhanh chóng hết tác dụng; phải tối ưu cả tiền / hậu xử lý.

### Slide 20. Chẩn đoán tự động (19 luật) & so sánh A/B
- **Bố cục:** Trái: danh mục luật với trạng thái **PASS / WARN / FAIL / SKIPPED**; phải: kiểm định A/B.
- **Nội dung:** 4 nhóm luật: bộ nhớ RA (5), tính toán RB (5), nạp dữ liệu RC (4), kiến trúc RD (5); RB-04 dùng ms/GFLOP (ngưỡng 2.0); mỗi luật kèm gợi ý sửa code trên tab Chẩn đoán (ví dụ RC-01: chờ dữ liệu > 20% → `DataLoader(num_workers=2, pin_memory=True)`). A/B bằng Mann-Whitney U (p < 0.05) và Cliff's delta, speedup tính trên tổng thời gian mỗi vòng.

### Slide 21. Demo dashboard (6 slide ảnh chụp, khoảng 90 giây)
- **Bố cục:** Mỗi slide 1 ảnh chụp dashboard to + 3 ý cần chỉ vào: tab Layer, FP16 (CPU và T4), DataLoader, Pipeline, Chẩn đoán (mở RB-04), A/B (ViT FP32 so với FP16). Trong bộ slide HTML là trang 23–28.
- **Nội dung:** Dashboard `streamlit run src/dashboard/app.py` chạy sẵn để mở trực tiếp khi thầy hỏi; cách chạy và tab ứng với từng slide: `docs/HUONG_DAN_CHAY_DASHBOARD.md`.

### Slide 22. Tổng kết 6 câu hỏi nghiên cứu
- **Bố cục:** Bảng 6 dòng: câu hỏi · câu trả lời · bằng chứng.
- **Nội dung:**
  - **CH1:** Conv (CNN 73–85%), Linear + Attention (Transformer 95–96%); lên GPU layer nhỏ chiếm tỷ trọng lớn hơn (MobileNetV3 BatchNorm 14% → 25%).
  - **CH2:** Sai lệch < 1% CPU, < 8% GPU so với `torch.profiler`; overhead GPU batch 1 lớn → mức Layer chỉ dùng để phân rã.
  - **CH3:** Trên T4 ở batch 1, tham số / FLOPs không quyết định độ trễ (ResNet-50 ≈ MobileNetV3 ≈ 6 ms); thứ hạng thay đổi theo phần cứng.
  - **CH4:** Batch 1 → 32 tăng 1.3–8.8× tuỳ mô hình; FP16 giảm VRAM 38–49% nhưng chỉ tăng tốc Transformer (đến 2.8×).
  - **CH5:** Backward 1.4–2.2× forward; VRAM training gấp 2.7–3.3×; thiếu worker thì GPU chờ dữ liệu tới 63%.
  - **CH6:** YOLOv8n trên T4: 41% thời gian nằm ngoài mô hình.

### Slide 23. Hạn chế, hướng phát triển & Q&A
- **Nội dung:**
  - *Hạn chế:* không đo kernel GPU / hardware counter; khoảng 4–10% thời gian không thuộc layer nào (phép toán dạng hàm); 1 GPU; PyTorch eager; mô hình dùng trọng số khởi tạo ngẫu nhiên (đủ cho đo tốc độ, không dùng cho độ chính xác).
  - *Hướng phát triển (cuối kỳ):* `torch.compile`, bắt toán tử bằng `TorchDispatchMode`, thí nghiệm ablation, đo VRAM training của DistilBERT. (Gợi ý sửa code cho 19 luật đã xong 18/10.)
  - Tài liệu tham khảo (IEEE) ở slide phụ lục; mã QR tới repo.

---

## 🎯 Câu hỏi phản biện dự kiến

- 🟧 **TV1:**
  - *Overhead GPU batch 1 lớn, sao vẫn tin số đo?* → Sai lệch **tổng** so với `torch.profiler` < 8% (đo lại 18/10 vẫn vậy). Overhead nằm ở chi phí ghi event và gọi hook giữa các layer, nên mức Layer chỉ dùng để phân rã tỷ lệ; tổng thời gian lấy từ mức Model. Riêng layer siêu nhỏ (ReLU, BatchNorm ở batch 1) sai số theo loại lớn hơn nhiều vì ngưỡng sàn event ~6 µs.
  - *Phép toán dạng hàm (`x + shortcut`, `torch.cat`) đi đâu?* → Đo thành chỉ số "thời gian không thuộc layer nào": 3.9–9.2% (CPU), 9–10% (T4) trên ResNet-50 và BERT-base.
- 🟦 **TV2:**
  - *Sao MobileNetV3 trên CPU chậm hơn ResNet-50 8 lần?* → Chỉ xảy ra trên MacBook chip Apple (176 ms); trên laptop x86 khoảng 12 ms. Đây là đặc thù kernel PyTorch trên CPU ARM, không phải tính chất chung của CPU. Trên T4, hai mô hình bằng nhau (~6 ms) vì batch 1 bị giới hạn bởi gửi kernel.
  - *Sao FP16 làm CNN chậm hơn?* → Ở batch 1, thời gian chủ yếu là gửi kernel và các layer nhỏ, Tensor Core không có việc; FP16 thêm thao tác chuyển kiểu. Transformer có phép nhân ma trận lớn nên hưởng lợi.
- 🟩 **TV3:**
  - *Tiền xử lý chiếm 41% trên GPU thì khắc phục thế nào?* → Đưa giải mã / resize / chuẩn hoá lên GPU (NVIDIA DALI, torchvision trên tensor GPU), chạy NMS trên GPU.
  - *Sao 4 worker không nhanh hơn 2?* → Colab chỉ có 2 vCPU; thêm worker chỉ tranh CPU với tiến trình chính.

---

## ✅ Checklist trước buổi trình bày

- [ ] **TV1:** Slide 4–10.
- [ ] **TV2:** Slide 11–16. ✅ Đã đổi nhãn hình thành "CPU MacBook"; ✅ đã viết lại CH3, CH4 trong báo cáo.
- [ ] **TV3:** Slide 1–3, 17–23; ✅ hình Slide 18 đã có; ✅ ảnh chụp dashboard cho phần demo đã có; chạy thử dashboard trên máy trình chiếu.
- [ ] **Cả nhóm:** Mẫu slide chung theo màu (🟩 TV3 · 🟧 TV1 · 🟦 TV2); tổng duyệt đúng 24 phút.
