# 🔬 PROJECT PLAN: AI Model Profiling Framework

> **Đề tài:** Xây dựng framework đo và phân tích hiệu năng (profiling) cho các mô hình AI (Computer Vision · NLP · Machine Learning cổ điển) trên CPU và GPU
> **Loại:** Đồ án môn học · **Nhóm:** 3 người (TV1, TV2, TV3) · **Code nền:** VisionProf v1.0 (`src/collector.py`, `src/analyzer.py`, `src/dashboard.py`)

> [!TIP]
> **Đọc nhanh**
> - **Bài toán:** Model AI chạy nhanh/chậm thế nào, tốn bao nhiêu tài nguyên trên từng máy, và **chậm ở đâu**?
> - **Phạm vi:** 8 model (CV, NLP, ML) · 4 loại máy · inference là chính, training là phụ
> - **Chia việc:** mỗi người một **mức đo**, tự làm trọn từ code → chạy → phân tích → trình bày
> - **Làm song song**, chỉ cần chốt chung format dữ liệu ở đầu (Mục 8)

## 📑 Mục lục

1. [Bài toán & Mục tiêu](#1-bài-toán--mục-tiêu)
2. [Hiện trạng code](#2-hiện-trạng-code)
3. [Phạm vi: Model & Dữ liệu](#3-phạm-vi-model--dữ-liệu)
4. [Phạm vi: Máy đo & Training/Inference](#4-phạm-vi-máy-đo--traininginference)
5. [3 mức đo & Bộ Metrics](#5-3-mức-đo--bộ-metrics)
6. [Câu hỏi nghiên cứu & Thực nghiệm](#6-câu-hỏi-nghiên-cứu--thực-nghiệm)
7. [Phân công chi tiết](#7-phân-công-chi-tiết)
8. [Phối hợp chung](#8-phối-hợp-chung)
9. [Sản phẩm bàn giao](#9-sản-phẩm-bàn-giao)
10. [Rủi ro & Ngoài phạm vi](#10-rủi-ro--ngoài-phạm-vi)

---

## 1. Bài toán & Mục tiêu

### 1.1. Phát biểu bài toán

> **"Cho một mô hình AI, nó chạy nhanh/chậm thế nào, tốn bao nhiêu tài nguyên trên từng loại máy, và *thời gian/bộ nhớ bị tiêu tốn ở đâu* (layer nào, giai đoạn nào)?"**

Framework cần làm được 4 việc:
- **Đo** hiệu năng ở 3 mức chi tiết: toàn model · từng giai đoạn · từng layer.
- **Chuẩn hoá** kết quả để so sánh được giữa các model và giữa các máy.
- **Chẩn đoán** tự động điểm nghẽn (bottleneck) bằng bộ luật heuristic.
- **Trực quan hoá** kết quả trên dashboard.

### 1.2. Phân biệt 3 khái niệm (dùng khi trình bày)

| Khái niệm | Trả lời câu hỏi | Ví dụ chỉ số | Project có làm? |
| :--- | :--- | :--- | :---: |
| **Evaluation** | Model đoán *đúng* không? | Accuracy, mAP, F1 | ❌ (chỉ kiểm tra sai lệch output khi dùng FP16) |
| **Benchmarking** | Model chạy *nhanh* và *tốn* bao nhiêu? | Latency P50/P95, FPS, Peak Memory | ✅ |
| **Profiling** | *Tại sao* chậm, chậm *ở đâu*? | Thời gian từng layer, forward/backward, FLOPs | ✅ **(trọng tâm)** |

### 1.3. Mục tiêu cần đạt

| # | Mục tiêu | Tiêu chí đạt |
| :-: | :--- | :--- |
| G1 | Hỗ trợ 3 họ model | Profile được 8 model (CV, NLP, ML) bằng **1 lệnh** |
| G2 | Chạy trên nhiều máy | Cùng một cấu hình chạy được trên CPU laptop, GPU local, Colab, Kaggle |
| G3 | Đo đúng | Sai lệch so với `torch.profiler` **< 10%**; profiler làm model chậm đi **< 5%** |
| G4 | Phân tích có chiều sâu | Trả lời được 6 câu hỏi nghiên cứu (Mục 6) bằng số liệu |
| G5 | Trình bày được | Dashboard + báo cáo + slide + demo trực tiếp |

---

## 2. Hiện trạng code

| Thành phần | File | Trạng thái | Ai sửa |
| :--- | :--- | :--- | :-: |
| `LayerProfiler`: hook đo từng layer | `collector.py` | ⚠️ Đúng trên CPU, **sai trên GPU** (dùng đồng hồ CPU) | TV1 |
| `calibrate_overhead` | `collector.py` | ⚠️ Đếm sai số layer có gắn hook | TV1 |
| `DataLoaderSentinel`: đo thời gian nạp dữ liệu | `collector.py` | ⚠️ Dùng được, nhưng bỏ sót batch đầu tiên | TV3 |
| `RuleCatalog`: 19 luật chẩn đoán | `analyzer.py` | ⚠️ 4 luật sai logic: RB-04, RD-02, RD-05, RC-03 | TV3 |
| `ABComparisonEngine`: so sánh 2 model | `analyzer.py` | ⚠️ Speedup tính sai (lấy trung bình các layer) | TV3 |
| Dashboard 5 tab (Streamlit) | `dashboard.py` | ✅ Dùng được, cần thêm tab | Cả 3 |
| Benchmark 5 model CV trên CPU | `tests/` | ✅ Kết quả cũ trong `data/profiles/` | — |

**Còn thiếu:** NLP và ML cổ điển · đo training (forward / backward / optimizer) · P50/P95/P99 và throughput · FLOPs · ghi thông tin máy để so sánh giữa các máy · kiểm chứng độ chính xác của công cụ đo.

---

## 3. Phạm vi: Model & Dữ liệu

### 3.1. 8 model chính

| Họ | Model | Params | Input | Vì sao chọn | Nguồn |
| :--- | :--- | :-: | :--- | :--- | :--- |
| **CV** | MobileNetV3-Large | 5.5M | `3×224×224` | CNN nhẹ cho thiết bị yếu | `torchvision` |
| | ResNet-50 | 25.6M | `3×224×224` | CNN kinh điển, làm mốc so sánh | `torchvision` |
| | ViT-B/16 | 86.6M | `3×224×224` | Transformer cho ảnh | `torchvision` |
| | YOLOv8n | 3.2M | `3×640×640` | Nhận diện vật thể, có tiền/hậu xử lý (NMS) | `ultralytics` |
| **NLP** | DistilBERT | 66M | 128 token | BERT "rút gọn" (6 layer), so sánh cặp với BERT | `transformers` |
| | BERT-base | 110M | 128 token | Transformer NLP chuẩn (12 layer) | `transformers` |
| **ML** | MLP | ~0.1M | 54 đặc trưng | Model nhỏ: chi phí framework lớn hơn phần tính toán | Tự viết |
| | XGBoost | — | Covertype | Mô hình cây, không phải neural net, chạy được CPU & GPU | `xgboost` |

**Bonus (nếu dư sức):**
- YOLOv11n: tái hiện "YOLO Paradox", tức ít tham số hơn YOLOv8n nhưng chạy chậm hơn trên CPU.
- ConvNeXt-Tiny: đã có số liệu CPU cũ.
- GPT-2: đo độ trễ sinh từng token.

### 3.2. Dữ liệu

| Họ | Để đo tốc độ | Để đo training / quy trình thật |
| :--- | :--- | :--- |
| CV | Tensor ngẫu nhiên, kích thước cố định | CIFAR-10 (resize 224) · 100 ảnh COCO cho YOLO |
| NLP | Token ngẫu nhiên, độ dài cố định | SST-2 (qua thư viện HF `datasets`) |
| ML | Covertype (`sklearn.datasets.fetch_covtype`) | Covertype (chia train/test) |

> 📌 **Không cần train tới hội tụ.** Chỉ chạy vài chục vòng để đo hiệu năng.

---

## 4. Phạm vi: Máy đo & Training/Inference

### 4.1. 4 loại máy

| Máy | Thiết bị | Ai dùng | Vai trò |
| :--- | :--- | :--- | :--- |
| **Laptop CPU** | Intel/AMD, mỗi người 1 máy | Cả 3 | Đại diện máy cá nhân / thiết bị biên |
| **GPU local** | NVIDIA RTX | **Chỉ TV1** | GPU phổ thông, không giới hạn thời gian dùng |
| **Colab** | Tesla T4 16GB, có Tensor Core | Cả 3 (dùng chung) | **GPU chính của cả nhóm** |
| **Kaggle** | Tesla P100 16GB, **không** có Tensor Core | Cả 3 (dùng chung) | So với T4: FP16 còn lợi không khi GPU không có Tensor Core? |

> ⚠️ Kaggle có gói 2×T4 nhưng nhóm chỉ dùng **1 GPU**.

### 4.2. Inference vs Training

| | **Inference (trọng tâm, ~70%)** | **Training (phụ, ~30%)** |
| :--- | :--- | :--- |
| Chế độ | `model.eval()` + `torch.inference_mode()` | `model.train()`, optimizer AdamW |
| Đo gì | Đủ cả 3 mức: Model, Giai đoạn, Layer | Mức Model + Giai đoạn (nạp dữ liệu → forward → backward → optimizer). **Không đo từng layer.** |
| Số vòng | Khởi động 10 + đo 50 | Khởi động 5 + đo 30 |
| Model | Cả 8 | ResNet-50, MobileNetV3, DistilBERT, MLP, XGBoost `fit` |

---

## 5. 3 mức đo & Bộ Metrics

Ba mức giống **3 độ zoom khi nhìn cùng một model**. Đây **không phải thứ tự làm**: 3 mức được làm song song, mỗi người phụ trách một mức.

| Mức | Nhìn vào đâu | Áp dụng cho | Kỹ thuật đo | Ai |
| :--- | :--- | :--- | :--- | :-: |
| 🟧 **Layer** | Bên trong model | Model PyTorch (CV, NLP, MLP) | Forward hook · CPU: `perf_counter_ns` · GPU: `torch.cuda.Event`, đồng bộ 1 lần cuối | **TV1** |
| 🟦 **Model** | Toàn bộ model | Cả 8 model (kể cả XGBoost) | Bấm giờ bao ngoài + đồng bộ GPU; luồng nền lấy mẫu tài nguyên | **TV2** |
| 🟩 **Giai đoạn** | Từng bước xử lý | Cả 8 model | Context manager đánh dấu từng giai đoạn | **TV3** |

### Bảng Metrics

| Nhóm | Metric | Đơn vị | Mức | Công cụ |
| :--- | :--- | :-: | :-: | :--- |
| ⏱️ **Thời gian** | Mean / Std latency | ms | Model | `time.perf_counter` |
| | **P50 / P90 / P95 / P99** latency | ms | Model | `numpy.percentile` |
| | Tail ratio = P99 / P50 | × | Model | Độ ổn định (có bị giật không) |
| | Throughput | mẫu/s (ảnh, câu, dòng) | Model | `batch × 1000 / mean_ms` |
| | Cold-start (lần chạy đầu) | ms | Model | Vòng đầu tiên |
| | Thời gian từng giai đoạn | ms, % | Giai đoạn | Phase timer |
| | Thời gian từng layer / loại layer | ms, % | Layer | Hook + CUDA Events |
| 💾 **Bộ nhớ** | Peak VRAM | MB | Model | `torch.cuda.max_memory_allocated` |
| | RAM của chương trình (RSS) | MB | Model | `psutil` |
| | Bộ nhớ training / inference | × | Model | So TN6 với TN3 |
| | Bộ nhớ output từng layer | MB | Layer | `numel × element_size` |
| 🧮 **Tính toán** | Params | M | Model | `sum(p.numel())` |
| | FLOPs toàn model & từng layer | GFLOPs | Model, Layer | `torch.utils.flop_counter.FlopCounterMode` |
| | Hiệu suất đạt được | % so với sức tối đa của máy | Layer | FLOPs / thời gian, so với thông số phần cứng |
| 🖥️ **Hệ thống** | CPU %, GPU %, VRAM đang dùng | %, MB | Model | `psutil`, `pynvml` (luồng nền 100ms) |
| | Công suất GPU *(nếu đọc được)* | W | Model | `pynvml` |
| 📦 **Dữ liệu** | Tỷ lệ I/O = nạp dữ liệu / tổng thời gian | % | Giai đoạn | `DataLoaderSentinel` |
| ✅ **Output** | Cosine similarity FP32 so với FP16 | — | Model | So output của 2 cấu hình |
| 🔧 **Tự kiểm** | Profiler làm chậm bao nhiêu (overhead) | % | Layer | Chạy có hook so với không hook |
| | Thời gian không thuộc layer nào | % | Layer | Tổng thời gian − Σ các layer (các phép như `torch.cat`, `+`) |

**Riêng XGBoost:** thời gian `fit`, độ trễ `predict` (1 dòng và 10.000 dòng), số dòng/giây, RAM đỉnh, tốc độ theo `n_jobs` = 1 / 2 / 4 / tất cả, và **CPU so với GPU**.

---

## 6. Câu hỏi nghiên cứu & Thực nghiệm

**Câu hỏi (CH)** là điều báo cáo phải trả lời. **Thực nghiệm (TN)** là việc chạy máy để lấy số liệu trả lời câu hỏi đó.

### 6.1. 6 câu hỏi

| Ai | Câu hỏi | Thực nghiệm | Kỳ vọng |
| :-: | :--- | :--- | :--- |
| **TV1** | **CH1.** Thời gian tốn nhiều nhất ở **loại layer nào**? CPU và GPU khác nhau ra sao? | **TN1** | Trên CPU, Conv/Linear chiếm phần lớn. Trên GPU, tỷ trọng các layer nhỏ (Norm, Activation) tăng lên |
| | **CH2.** Công cụ của nhóm **đo có đúng không**, và làm model chậm đi bao nhiêu? | **TN2** | Sai lệch < 10% so với `torch.profiler`, làm chậm < 5% |
| **TV2** | **CH3.** Model ít tham số có chắc **chạy nhanh hơn**? Đổi máy thì **thứ hạng có đảo** không? | **TN3** | Có đảo. Model nhiều layer nhỏ (MobileNet, YOLO) ít hưởng lợi từ GPU, còn ViT/BERT hưởng lợi nhiều nhất |
| | **CH4.** **Tăng batch** hoặc dùng **FP16** thì nhanh thêm bao nhiêu? T4 khác P100 ra sao? | **TN4, TN5** | GPU tăng tốc mạnh khi batch lớn, CPU gần như không. FP16 nhanh rõ trên T4, ít lợi trên P100 |
| **TV3** | **CH5.** Khi **training**, backward tốn gấp mấy lần forward? Bộ nhớ tăng bao nhiêu? Nạp dữ liệu có làm GPU phải chờ? | **TN6** | Backward ≈ 2× forward; bộ nhớ gấp 2–4× inference; `num_workers=0` gây nghẽn |
| | **CH6.** Trong ứng dụng thật, **ngoài model** (tiền xử lý, NMS, tokenize) chiếm bao nhiêu %? | **TN7** | Với YOLO batch 1 trên GPU, tiền xử lý + hậu xử lý chiếm > 30% |

### 6.2. 7 thực nghiệm

| TN | Tên | Thay đổi gì | Model | Kết quả chính |
| :-: | :--- | :--- | :--- | :--- |
| **TN1** | Phân rã theo layer | CPU so với GPU | CV + NLP + MLP | Top-10 layer chậm nhất, tỷ trọng theo loại layer |
| **TN2** | Kiểm chứng công cụ | Có hook / không hook; so với `torch.profiler` | ResNet-50, BERT-base | % overhead, % sai lệch |
| **TN3** | So sánh cơ bản | 4 loại máy, batch 1 | Cả 8 | Bảng latency, throughput, bộ nhớ của 8 model × 4 máy |
| **TN4** | Tăng batch | batch 1 / 8 / 32 (NLP thêm độ dài 64 / 128 / 256) | Cả 8 | Đường throughput theo batch, điểm bão hoà |
| **TN5** | Độ chính xác số | FP32 so với FP16 trên GPU | CV + NLP | Mức tăng tốc, bộ nhớ giảm, cosine similarity |
| **TN6** | Một bước training | Loại máy; `num_workers` = 0 / 2 / 4 | 5 model ở Mục 4.2 | Biểu đồ cột: nạp dữ liệu / forward / backward / optimizer |
| **TN7** | Quy trình thật | Loại máy | YOLO (đọc ảnh → resize → model → NMS), BERT (tokenize → model), XGBoost (đọc CSV → predict) | % thời gian từng giai đoạn |

### 6.3. Chạy trên máy nào

| TN | Ai | Laptop CPU | GPU local | Colab T4 | Kaggle P100 |
| :-: | :-: | :-: | :-: | :-: | :-: |
| TN1 | TV1 | ✅ laptop TV1 | ✅ | ✅ | ◻️ |
| TN2 | TV1 | ✅ laptop TV1 | ✅ | ◻️ | ◻️ |
| TN3 | TV2 | ✅ laptop TV2 | ✅ *(TV1 chạy giúp 1 lệnh)* | ✅ | ✅ |
| TN4 | TV2 | ✅ laptop TV2 | ◻️ | ✅ | ◻️ |
| TN5 | TV2 | — | ◻️ | ✅ | ✅ |
| TN6 | TV3 | ✅ laptop TV3 (model nhỏ) | ◻️ | ✅ | ◻️ |
| TN7 | TV3 | ✅ laptop TV3 | ◻️ | ✅ | ◻️ |

✅ bắt buộc · ◻️ tuỳ chọn

### 6.4. Quy tắc đo chung

1. **Khởi động** 10 vòng (training: 5) và không tính các vòng này, rồi **đo** 50 vòng (training: 30).
2. **GPU:** gọi `torch.cuda.synchronize()` trước khi bấm giờ; bật `cudnn.benchmark = True`.
3. **Laptop:** cắm sạc, bật chế độ hiệu năng cao, tắt app nặng, cố định `torch.set_num_threads()`.
4. **So 2 model:** chạy **xen kẽ** A-B-A-B để máy nóng lên không làm lệch kết quả.
5. Mỗi thực nghiệm chỉ dùng **1 laptop** cho toàn bộ số liệu CPU.
6. Cố định seed và kích thước input. Mỗi lần chạy **tự lưu thông tin máy** (CPU, GPU, RAM, phiên bản thư viện).

---

## 7. Phân công chi tiết

**Nguyên tắc:** mỗi người **sở hữu trọn một mức đo**: tự viết công cụ → tự chạy → tự phân tích → tự làm tab dashboard → tự viết chương báo cáo → tự trình bày. Không ai phải trình bày kết quả đo bằng công cụ của người khác.

| | 👤 **TV1: Mức Layer** | 👤 **TV2: Mức Model** | 👤 **TV3: Mức Giai đoạn + Tổng kết** |
| :--- | :--- | :--- | :--- |
| **Trả lời** | CH1, CH2 | CH3, CH4 | CH5, CH6 |
| **Chạy** | TN1, TN2 | TN3, TN4, TN5 | TN6, TN7 |
| **File sở hữu** | `collector.py`, `flops.py`, `hw_specs.py` | `zoo/`, `benchmark.py`, `resources.py`, `run_experiment.py`, `notebooks/` | `phases.py`, `analyzer.py`, `dashboard/` |
| **Máy** | Laptop TV1 (CPU + GPU) · Colab | Laptop TV2 · Colab · Kaggle | Laptop TV3 · Colab |
| **Tab dashboard** | Layer · Hiệu suất phần cứng | Tổng quan · So sánh phần cứng | Chẩn đoán · A/B · DataLoader · Training & Pipeline |

### 👤 TV1: Mức Layer & Độ chính xác *(người có GPU)*

> **Trình bày:** "Framework mổ xẻ từng layer thế nào, và đo có đúng không" + demo gắn profiler vào model.

- **Code**
  - [ ] Sửa đo GPU: dùng `torch.cuda.Event`, chỉ đồng bộ 1 lần sau khi chạy xong model; dùng stack cho module được gọi nhiều lần
  - [ ] Sửa `calibrate_overhead` (đếm đúng số layer có hook)
  - [ ] Tách bộ nhớ output của từng layer khỏi RAM của cả chương trình
  - [ ] Tính FLOPs từng layer bằng `FlopCounterMode`; lập bảng thông số tối đa của 4 máy để tính % hiệu suất
  - [ ] Đóng gói thành plugin `layer` (Mục 8.1)
- **Chạy & phân tích:** TN1, TN2 → CH1, CH2
- **Việc phụ:** chạy giúp TV2 lệnh TN3 trên GPU local (chỉ chạy, không phân tích)
- **Báo cáo:** kiến thức nền (hook, CUDA Events, FLOPs) · công cụ liên quan (`torch.profiler`, Nsight) · thiết kế mức Layer · kết quả CH1–CH2 · kiểm chứng độ chính xác

### 👤 TV2: Mức Model & So sánh phần cứng

> **Trình bày:** "8 model × 4 loại máy: model nào nhanh nhất, trên máy nào".

- **Code**
  - [ ] Nạp 8 model qua một cách gọi chung (kèm letterbox + NMS cho YOLO, tokenizer cho BERT)
  - [ ] Đo toàn model: P50/P90/P95/P99, throughput, bộ nhớ đỉnh
  - [ ] Luồng nền theo dõi CPU %, RAM, GPU %, VRAM (mỗi 100ms) + tự ghi thông tin máy
  - [ ] Script chạy chung: `run_experiment.py --model resnet50 --exp TN3 --device cuda --profilers layer,phase`, có cờ `--skip-existing`
  - [ ] Notebook cho Colab / Kaggle: clone → cài → chạy → nén `results/`
- **Chạy & phân tích:** TN3, TN4, TN5 → CH3, CH4
- **Báo cáo:** kiến trúc tổng · thiết kế mức Model · thiết lập thực nghiệm (model, máy, quy tắc đo) · kết quả CH3–CH4
- *Bonus:* YOLOv11n, ConvNeXt-Tiny, GPT-2

### 👤 TV3: Mức Giai đoạn · Training, Pipeline, Chẩn đoán & Tổng kết

> **Trình bày:** mở đầu + "training và quy trình thật tốn thời gian ở đâu" + demo dashboard + kết luận.

- **Code**
  - [ ] Bộ đo giai đoạn `with t.phase("backward"):` (có đồng bộ GPU), đóng gói thành plugin `phase`
  - [ ] Chuyển `DataLoaderSentinel` sang `phases.py`, sửa lỗi bỏ sót batch đầu tiên
  - [ ] Sửa so sánh A/B (tính speedup trên tổng thời gian model mỗi vòng)
  - [ ] Sửa 4 luật chẩn đoán: RB-04, RD-05 (đổi ms/Mparam → ms/GFLOP), RD-02 (bỏ nhãn "dead layer"), RC-03
  - [ ] Tách dashboard thành khung `app.py` + thư mục `tabs/` (mỗi người 1 file tab)
- **Chạy & phân tích:** TN6, TN7 → CH5, CH6
- **Tổng kết**
  - [ ] Ghép báo cáo + slide (nội dung từng phần do chủ phần đó viết)
  - [ ] Viết **README tổng**: giới thiệu, cài đặt, cách chạy, tóm tắt kết quả (mỗi người gửi đoạn hướng dẫn phần code của mình)
- **Báo cáo:** giới thiệu & bài toán · thiết kế mức Giai đoạn & heuristic · kết quả CH5–CH6 · kết luận
- *Bonus:* đo backward từng layer (`register_full_backward_hook`)

---

## 8. Phối hợp chung

### 8.1. Chốt trước khi code (cả nhóm, 1 lần)

**① Cách gọi model:** TV2 viết, TV1 và TV3 dùng.

```python
class ModelAdapter:
    name: str          # "resnet50", "bert_base", "xgboost"...
    family: str        # "cv" | "nlp" | "ml"
    is_torch: bool     # False với XGBoost → bỏ qua mức Layer

    def load(self, device, precision="fp32"): ...
    def make_input(self, batch_size, **kw): ...        # seq_len cho NLP, img_size cho CV
    def forward(self, model, inputs): ...
    def train_step(self, model, batch, optimizer): ... # TV3 dùng cho TN6
    def preprocess(self, raw) / postprocess(self, out): ...  # TV3 dùng cho TN7
```

**② Cách gắn bộ đo:** TV1 viết plugin `layer`, TV3 viết plugin `phase`. Script của TV2 gọi cả hai.

```python
class ProfilerPlugin:
    name: str                       # "layer" (TV1) | "phase" (TV3)
    def attach(self, model, device): ...
    def start(self, iteration): ...
    def stop(self): ...
    def save(self, out_dir): ...    # ghi layers.csv (TV1) hoặc phases.csv (TV3)
```

**③ Nơi lưu kết quả:** mỗi file chỉ có 1 người ghi.

```
results/{máy}/{model}/{TN}/{cấu_hình}/
├── env.json          # TV2 · thông tin máy + phiên bản thư viện
├── config.json       # TV2 · model, batch, độ dài câu, FP32/FP16, thiết bị
├── benchmark.csv     # TV2 · vòng, latency_ms
├── summary.json      # TV2 · mean, p50, p90, p95, p99, throughput, bộ nhớ đỉnh
├── resources.csv     # TV2 · thời điểm, cpu %, ram, gpu %, vram, công suất
├── layers.csv        # TV1 · vòng, tên layer, loại, thời gian, bộ nhớ, flops, params
└── phases.csv        # TV3 · vòng, giai đoạn, thời gian
```

Tên máy: `cpu_laptop`, `gpu_local`, `colab_t4`, `kaggle_p100` · thời gian tính bằng **ms** · bộ nhớ tính bằng **MB**.

### 8.2. Kiến trúc & thư mục

```mermaid
graph LR
    CFG["⚙️ configs/*.yaml"] --> RUN["▶️ run_experiment.py (TV2)"]
    RUN --> ZOO["📦 Model Zoo (TV2)"]
    RUN --> P1["🟦 Benchmark + Resources (TV2)"]
    RUN --> P2["🟩 Phase plugin (TV3)"]
    RUN --> P3["🟧 Layer plugin (TV1)"]
    P1 & P2 & P3 --> RES["💾 results/"]
    RES --> ANA["📊 Analyzer (TV3)"] --> DASH["🖥️ Dashboard (cả 3)"]
```

```
VisionProf/
├── configs/                  # TV2 · file YAML cấu hình model + thực nghiệm
├── src/
│   ├── collector.py          # TV1 · LayerProfiler, calibrate_overhead
│   ├── flops.py, hw_specs.py # TV1 · FLOPs, thông số tối đa của 4 máy
│   ├── zoo/                  # TV2 · cv.py, nlp.py, tabular.py
│   ├── benchmark.py          # TV2 · đo toàn model
│   ├── resources.py          # TV2 · theo dõi tài nguyên + thông tin máy
│   ├── phases.py             # TV3 · PhaseTimer + DataLoaderSentinel
│   ├── analyzer.py           # TV3 · luật chẩn đoán + so sánh A/B
│   └── dashboard/            # TV3 làm app.py · tabs/ mỗi người 1 file
├── scripts/run_experiment.py # TV2
├── notebooks/                # TV2 · Colab, Kaggle
└── results/                  # kết quả theo từng máy
```

### 8.3. Thứ tự làm việc

| Bước | Việc | Ai | Điều kiện xong |
| :-: | :--- | :-: | :--- |
| 1 | Chốt Mục 8.1 | Cả nhóm | Commit vào repo |
| 2 | Code phần của mình. TV1 test tạm trên ResNet-50, TV3 test tạm trên dữ liệu cũ, nên không ai chờ TV2 | Song song | Mỗi phần chạy được riêng |
| 3 | Ghép thử | Cả nhóm | 1 model CV + 1 NLP + 1 ML chạy được trên cả 4 máy, với cả 2 plugin |
| 4 | Chạy thực nghiệm & phân tích phần mình | Song song | Đủ số liệu các ô ✅ ở Mục 6.3 |
| 5 | Ghép báo cáo, slide, README → tổng duyệt | TV3 + cả nhóm | Demo chạy trơn tru |

### 8.4. Quy tắc nhóm

- Mỗi người làm trên **branch riêng**, merge qua Pull Request, **review chéo** (TV1 → TV2 → TV3 → TV1).
- **Chỉ sửa file của mình.** Muốn đổi Mục 8.1 thì báo cả nhóm.
- Mỗi câu hỏi trong báo cáo có đủ **1 bảng + 1 biểu đồ + 3–5 câu kết luận**, do chủ phần tự làm.
- Colab / Kaggle giới hạn giờ dùng GPU, nên ai chạy phần nấy bằng **tài khoản riêng**.
- Riêng GPU local: TV2 gửi **đúng 1 lệnh**, TV1 chạy rồi push `results/gpu_local/`, không phân tích hộ.

---

## 9. Sản phẩm bàn giao

| Sản phẩm | Nội dung | Ai |
| :--- | :--- | :-: |
| **Source code** | Framework + script chạy + notebook Colab/Kaggle | Cả 3 |
| **Dữ liệu kết quả** | `results/` của 4 máy, kèm thông tin máy | Cả 3 |
| **Dashboard** | Streamlit, mỗi người 1–2 tab | Cả 3, TV3 làm khung |
| **README** | Giới thiệu, cài đặt, cách chạy, tóm tắt kết quả | TV3 (mỗi người gửi phần mình) |
| **Báo cáo** (~20–25 trang) | Theo dàn ý bên dưới | Mỗi người phần mình, TV3 ghép |
| **Slide + demo** | Mỗi người trình bày phần mình | Cả 3, TV3 làm khung |

**Dàn ý báo cáo**

| Chương | Nội dung | Ai viết |
| :-: | :--- | :-: |
| 1 | Giới thiệu & bài toán (Evaluation vs Benchmarking vs Profiling) | TV3 |
| 2 | Kiến thức nền: hook PyTorch, CUDA chạy bất đồng bộ & Events, FLOPs, tail latency | TV1 |
| 3 | Công cụ liên quan: `torch.profiler`, Nsight, TensorBoard Profiler, và framework này khác gì | TV1 |
| 4 | Thiết kế framework: kiến trúc tổng (TV2) + từng mức đo (mỗi người mức của mình) | Cả 3 |
| 5 | Thiết lập thực nghiệm: model, máy, quy tắc đo | TV2 |
| 6 | Kết quả & thảo luận: CH1–CH6, mỗi câu hỏi một mục | Mỗi người câu của mình |
| 7 | Hạn chế & hướng phát triển | Mỗi người phần mình |
| 8 | Kết luận | TV3 |

---

## 10. Rủi ro & Ngoài phạm vi

| Rủi ro | Mức | Cách xử lý |
| :--- | :-: | :--- |
| Colab / Kaggle ngắt giữa chừng | Cao | Lưu kết quả sau **mỗi** lần chạy; cờ `--skip-existing` để chạy tiếp |
| Số đo trên laptop dao động (nhiệt, pin, app chạy nền) | Cao | Cắm sạc, bật chế độ hiệu năng cao, khởi động 10 vòng, dùng P50 thay vì chỉ dùng trung bình, chạy xen kẽ A-B |
| Hook không bắt được phép tính dạng hàm (`torch.cat`, `x + y`) | TB | Báo cáo phần "thời gian không thuộc layer nào" như một metric, không coi là lỗi |
| 3 phần ghép vào không khớp | TB | Chốt Mục 8.1 **trước khi code** |
| Máy tính có GPU của TV1 bị hỏng | TB | Các TN bắt buộc của TV2, TV3 đã chạy trên Colab. TV1 chuyển TN1, TN2 sang Colab; chỉ mất cột GPU local ở TN3 |
| Khác phiên bản thư viện giữa các máy | TB | Ghim phiên bản trong `requirements.txt`, lưu phiên bản vào `env.json` |
| Không làm kịp hết | TB | Làm phần bắt buộc trước; nếu thiếu sức thì bỏ Kaggle hoặc TN5, TN7; bonus để cuối |

**Không làm** (để giữ phạm vi đồ án môn học):
- ❌ Kiểm chứng bằng NVIDIA Nsight (dùng `torch.profiler` thay thế)
- ❌ Lưu SQLite / Parquet (dùng CSV + JSON là đủ)
- ❌ Các bài ablation và mục tiêu nộp bài báo MLSys trong tài liệu cũ
- ❌ Training trên nhiều GPU · `torch.compile` · TensorRT
- ❌ LLM lớn (> 1B tham số) · đo accuracy / mAP

---

<p align="center"><i>Tài liệu định hướng chính của dự án. Các file cũ (<code>research_plan_profiling_framework.md</code>, <code>TECHNICAL_SPECIFICATION.md</code>, <code>SYSTEM_AUDIT_REPORT.md</code>) chỉ dùng để tham khảo chi tiết kỹ thuật.</i></p>
