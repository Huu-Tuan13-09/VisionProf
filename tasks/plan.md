# Kế hoạch Hoàn thiện Nhiệm vụ Thành viên 2 (TV2: Mức Model & So sánh CPU/GPU)

## 1. Tổng quan Dự án & Vị trí của TV2
Dự án **VisionProf** là framework đo và phân tích hiệu năng mô hình AI đa nền tảng (CV, NLP, Tabular) trên CPU Laptop và Google Colab Tesla T4. 
TV2 phụ trách **Mức Model (Toàn bộ mô hình)**, sở hữu:
- **Model Zoo 8 mô hình chuẩn:** MobileNetV3-Large, ResNet-50, ViT-B/16, YOLOv8n, DistilBERT, BERT-base, MLP, XGBoost. (Bonus: YOLOv11n, ConvNeXt-Tiny, GPT-2).
- **Bộ đo Benchmark & Giám sát Tài nguyên:** Thu thập đủ 12 metrics (P50..P99, Throughput, Cold-start, Peak VRAM, RAM RSS, CPU/GPU %, Power, Cosine similarity, XGBoost latency).
- **Thực nghiệm TN3, TN4, TN5:** Trả lời 2 câu hỏi nghiên cứu **CH3** (Mô hình ít tham số có chạy nhanh hơn không? Thứ hạng đảo nghịch CPU vs GPU) và **CH4** (Đường cong Throughput bão hòa theo Batch & Tăng tốc FP16 Tensor Cores).
- **Dashboard:** Tab 1 (Tổng quan) và Tab 2 (So sánh CPU/GPU).
- **Tài liệu bàn giao:** Chương 4.1-4.2, Chương 5, Chương 6.3-6.4, Chương 7, Slide và Kịch bản Demo.

---

## 2. Các Quyết định Kiến trúc & Quy chuẩn Bắt buộc (Golden Rules & Architecture)

1. **Tuân thủ Tuyệt đối Quy chuẩn Mã nguồn (`AGENTS.md` & `agent-rules/`):**
   - **Giới hạn kích thước file:** Tối đa 300 dòng code/file (Hard limit), cảnh báo nếu > 250 dòng.
   - **Đơn trách nhiệm (SRP):** Tách module chuyên biệt (ví dụ: tách benchmark XGBoost sang `src/model/tabular_bench.py` thay vì nhồi nhét vào `benchmark.py`).
   - **Type hints & Logging:** 100% type hints cho hàm public; cấm dùng lệnh bare `print()`, bắt buộc dùng `logging`.
   - **Rule Checker:** Mọi task code đều phải pass `python3 scripts/verify_rules.py` (exit code 0).

2. **Quy trình Git & Khởi tạo Nhánh làm việc (`agent-rules/03_git_workflow_and_collaboration.md`):**
   - **Vấn đề phân tách nhánh:** Nhánh `tv1-layer` chứa engine đo Layer của TV1 nhưng thiếu `src/zoo/`, `src/model/`, `scripts/` (vốn nằm trên `main`). Ngược lại `main` thiếu code của TV1.
   - **Giải pháp (Phase 0):** Tạo nhánh `feat/tv2-model-benchmark` từ `main`, merge `tv1-layer` vào, giải quyết 2 xung đột nhỏ (`requirements.txt` và `tests/test_layer_profiler.py`). Nhờ đó, TV2 có đầy đủ cả hạ tầng Model Zoo lẫn bộ đo Layer của TV1.

3. **Tách biệt Đo Benchmark Sạch khỏi Profiler Hooks (Khắc phục lỗi phồng trễ):**
   - *Nguyên nhân:* Kết quả TN2 của TV1 chứng minh hook làm chậm mô hình 5–11% trên CPU và **109–166% trên GPU T4 ở batch 1**.
   - *Quyết định:* Trong `scripts/run_experiment.py`, vòng lặp đo latency (P50..P99) của TV2 được chạy ở **Lượt 1 (Sạch hoàn toàn)**. Sau đó mới gắn `LayerPlugin` ở **Lượt 2** để trích xuất `layers.csv`.

4. **Đo Peak VRAM Chính xác (Loại bỏ điểm mù 100ms của Background Thread):**
   - *Nguyên nhân:* Forward pass ở batch 1 chỉ mất 5–20ms, đỉnh bộ nhớ xuất hiện và giải phóng ngay. Luồng `ResourceSampler` lấy mẫu mỗi 100ms sẽ **bỏ sót hoàn toàn đỉnh VRAM**!
   - *Quyết định:* Bắt buộc gọi `torch.cuda.reset_peak_memory_stats()` trước khi chạy và `torch.cuda.max_memory_allocated() / 1024**2` ngay sau vòng lặp để ghi nhận giá trị phần cứng chuẩn xác 100%.

5. **Chuẩn hóa Đường dẫn & Chống lỗi Case-Sensitivity trên Linux (Google Colab):**
   - *Nguyên nhân:* Trên Linux (Colab ext4), thư mục `TN1` và `tn1` là hai đường dẫn khác nhau. TV1 ghi hoa (`TN1`), trong khi `build_result_path` trên `main` đang ép thường (`tn1`).
   - *Quyết định:* Thống nhất chuẩn tên thực nghiệm viết HOA: `TN1`, `TN2`, `TN3`, `TN4`, `TN5`, `TN6`, `TN7` đúng theo [PROJECT_PLAN.md Mục 8.1 ③](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/PROJECT_PLAN.md#L325).

6. **Tuân thủ Tuyệt đối Hợp đồng Bàn giao với TV1 (Mục 8.1 & Tin nhắn TV1):**
   - Export alias `get_adapter(name)` bên cạnh `get_model_adapter(name)`.
   - Nạp `mobilenet_v3_large` (5.5M params), không dùng small.
   - Nạp pure `nn.Module` cho YOLOv8 (`yolo.model`).
   - Mọi `make_input` đều nhận `**kwargs` để nuốt `seq_len` mà không ném lỗi.
   - Với BERT/DistilBERT: `make_input` chỉ trả về `{"input_ids": ids}`, **loại bỏ `attention_mask`** khi không có padding để ngăn lỗi đồng bộ ngầm CPU-GPU (làm sai số GPU tới 87%).

7. **Cung cấp API phục vụ TV3 (Điểm chạm Mục 8.1 ① & 8.3 ②):**
   - Cung cấp `train_step(self, model, batch, optimizer)` trong các adapter PyTorch cho 5 model: ResNet-50, MobileNetV3-Large, DistilBERT, MLP, XGBoost (`fit`).
   - Cung cấp `preprocess()` và `postprocess()` thật: YOLOv8 (resize 640x640 + NMS) và BERT (tokenizer + decode) phục vụ TN7.

---

## 3. Danh sách Giai đoạn & Nhiệm vụ Chi tiết (Detailed Tasks)

### 🟢 Phase 0: Đồng bộ Git & Hợp nhất Mã nguồn (Pre-requisite Setup)

#### Task 0.1: Tạo nhánh làm việc và Hợp nhất `main` với `tv1-layer`
- **Mô tả:** Khởi tạo nhánh `feat/tv2-model-benchmark` từ `main`, thực hiện merge nhánh `tv1-layer` vào để có đầy đủ cả hạ tầng module trên `main` (`src/core/`, `src/zoo/`, `src/model/`, `scripts/`) và engine của TV1 (`src/layer_plugin.py`, `src/collector.py`, `src/flops.py`).
- **Tiêu chí chấp nhận:**
  - [ ] Giải quyết xung đột `requirements.txt`: gộp đủ `transformers>=4.40.0`, `matplotlib>=3.8.0`, `pyngrok>=7.0.0`, `streamlit>=1.46.0`.
  - [ ] Giải quyết xung đột `tests/test_layer_profiler.py`: giữ lại trọn vẹn bộ 11 tests chuẩn của TV1.
  - [ ] Cả `src/zoo/` và `src/layer_plugin.py` đều cùng tồn tại trong working tree.
- **Kiểm chứng:**
  - [ ] `git status` sạch sẽ trên nhánh mới.
  - [ ] Chạy `pytest tests/test_layer_profiler.py tests/test_layer_flops_plugin.py -q` pass 22 tests.
- **Phụ thuộc:** Không.
- **Tập tin liên quan:** `requirements.txt`, `tests/test_layer_profiler.py`.
- **Độ lớn:** S (2 files).

---

### 🟢 Phase 1: Chuẩn hóa Model Zoo & Hợp đồng Giao tiếp (Foundation)

#### Task 1.1: Chuẩn hóa Model Adapters Computer Vision (`src/zoo/cv.py`)
- **Mô tả:** Đổi `mobilenet_v3_small` thành `mobilenet_v3_large` (5.5M params). Đảm bảo `YOLOv8nAdapter.load()` trả về pure `nn.Module` (`yolo.model`). Chuẩn hóa `make_input` nhận `**kwargs` (chấp nhận `seq_len` không lỗi) và giữ kích thước ảnh `224x224` (CV), `640x640` (YOLO). Cài đặt `train_step()` cho `ResNet-50` & `MobileNetV3`, và `preprocess()`/`postprocess()` (NMS) cho `YOLOv8n`.
- **Tiêu chí chấp nhận:**
  - [ ] `MobileNetV3Adapter` nạp đúng `tvm.mobilenet_v3_large(weights=None)` (~5.5M params).
  - [ ] `YOLOv8nAdapter.load()` trả về đối tượng `nn.Module` (`yolo.model`), không trả wrapper `YOLO`.
  - [ ] Tất cả adapter CV nhận tham số `seq_len=...` qua `**kwargs` mà không ném `TypeError`.
  - [ ] `train_step()` và `preprocess/postprocess` được cài đặt đầy đủ type hints.
  - [ ] File `src/zoo/cv.py` $\le 300$ dòng code, không chứa lệnh bare `print()`.
- **Kiểm chứng:**
  - [ ] Unit test: `pytest tests/test_model_zoo.py -k "cv" -v`
  - [ ] Rule test: `python3 scripts/verify_rules.py`
  - [ ] Manual check: `python -c "from src.zoo.cv import MobileNetV3Adapter, YOLOv8nAdapter; m1=MobileNetV3Adapter().load('cpu'); inp1=MobileNetV3Adapter().make_input(1, seq_len=128); print(type(m1), inp1.shape)"`
- **Phụ thuộc:** Task 0.1.
- **Tập tin liên quan:** `src/zoo/cv.py`, `src/zoo/base.py`, `tests/test_model_zoo.py`.
- **Độ lớn:** S (2 files).

#### Task 1.2: Chuẩn hóa Model Adapters NLP & Xử lý bẫy Đồng bộ Ngầm (`src/zoo/nlp.py`)
- **Mô tả:** Cập nhật `DistilBERTAdapter` và `BERTBaseAdapter`. Trong `make_input`, **chỉ trả về `{"input_ids": ids}`**, không kèm `attention_mask` khi input ngẫu nhiên không có padding để ngăn transformers đồng bộ ngầm CPU-GPU. Bổ sung `**kwargs` cho `make_input`. Bổ sung `train_step()` cho `distilbert` và `preprocess/postprocess` (tokenize) cho `bert_base`.
- **Tiêu chí chấp nhận:**
  - [ ] `make_input` của `bert_base` và `distilbert` trả về dictionary chỉ chứa key `"input_ids"`.
  - [ ] `load()` trả về mô hình sẵn sàng cho inference trên CPU và CUDA/T4.
  - [ ] Thêm `train_step` và `preprocess/postprocess`. File $\le 300$ dòng code.
- **Kiểm chứng:**
  - [ ] Rule test: `python3 scripts/verify_rules.py`
  - [ ] Manual check: `python -c "from src.zoo.nlp import BERTBaseAdapter; inp=BERTBaseAdapter().make_input(1, seq_len=128); assert list(inp.keys()) == ['input_ids']"`
- **Phụ thuộc:** Task 1.1.
- **Tập tin liên quan:** `src/zoo/nlp.py`.
- **Độ lớn:** S (1 file).

#### Task 1.3: Chuẩn hóa Model Adapters Tabular & Tạo Cây Thực tế (`src/zoo/tabular.py`)
- **Mô tả:** Kiểm tra `MLPAdapter` (Covertype 54 features, 7 classes) và `XGBoostAdapter` (`is_torch=False`, Covertype data, `device="cuda"` trên Colab và `"cpu"` trên laptop). Bổ sung dữ liệu giả lập tối thiểu 1.000 mẫu để khởi tạo cây XGBoost đạt độ sâu 6 thực tế (tránh cây rỗng chạy 0.00ms). Bổ sung `train_step()` cho `MLPAdapter` và `fit()` cho `XGBoost`.
- **Tiêu chí chấp nhận:**
  - [ ] `MLPAdapter.make_input` nhận `seq_len` qua `**kwargs` không lỗi, trả về tensor `(B, 54)`.
  - [ ] `XGBoostAdapter.load()` nạp mô hình đã fit với dữ liệu giả lập đủ sâu, chạy ổn định trên CPU/GPU.
  - [ ] File $\le 300$ dòng code, type hints đầy đủ.
- **Kiểm chứng:**
  - [ ] Unit test: `pytest tests/test_model_zoo.py -k "tabular" -v`
  - [ ] Rule test: `python3 scripts/verify_rules.py`
- **Phụ thuộc:** Task 1.1, 1.2.
- **Tập tin liên quan:** `src/zoo/tabular.py`.
- **Độ lớn:** S (1 file).

#### Task 1.4: Cập nhật Registry và Export Giao thức Chung (`src/zoo/__init__.py`)
- **Mô tả:** Đăng ký đầy đủ 8 tên model chuẩn: `mobilenet_v3_large`, `resnet50`, `vit_b_16`, `yolov8n`, `distilbert`, `bert_base`, `mlp`, `xgboost` (kèm alias `mobilenet_v3`). Export hàm `get_adapter(name)` (alias cho `get_model_adapter`) đúng theo yêu cầu của TV1.
- **Tiêu chí chấp nhận:**
  - [ ] `from src.zoo import get_adapter` hoạt động thành công.
  - [ ] `get_adapter("mobilenet_v3_large")` và `get_adapter("mobilenet_v3")` đều trả về `MobileNetV3Adapter`.
  - [ ] Đủ 8 mô hình trong `list_available_models()`.
- **Kiểm chứng:**
  - [ ] Unit test: `pytest tests/test_model_zoo.py -v`
  - [ ] Kiểm tra tương thích với TV1: `python experiments/demo_layer_profiler.py --model resnet50 --iters 5` và `python experiments/demo_layer_profiler.py --model bert_base --iters 5` in ra Top-10 mà không rơi vào `_fallback_model`.
  - [ ] Rule test: `python3 scripts/verify_rules.py`
- **Phụ thuộc:** Task 1.1, 1.2, 1.3.
- **Tập tin liên quan:** `src/zoo/__init__.py`, `tests/test_model_zoo.py`.
- **Độ lớn:** S (2 files).

### 🛑 Checkpoint 1: Model Zoo Hoàn thiện & Khớp Nối TV1
- [ ] Mọi adapter trong Model Zoo khởi tạo và sinh input đúng kích thước quy định.
- [ ] Không còn bẫy ngầm `attention_mask` ở BERT.
- [ ] Script demo của TV1 `experiments/demo_layer_profiler.py` nhận diện thành công Zoo của TV2.
- [ ] Toàn bộ unit tests trong `tests/test_model_zoo.py` pass 100%.
- [ ] Lệnh `python3 scripts/verify_rules.py` exit code 0.

---

### 🟢 Phase 2: Hoàn thiện Bộ đo Benchmark, Giám sát Tài nguyên & Runner (Core Harness)

#### Task 2.1: Hoàn thiện `ModelBenchmark` Đủ 12 Metrics (`src/model/benchmark.py`)
- **Mô tả:** Nâng cấp `ModelBenchmark` và `BenchmarkSummary`:
  - Bổ sung `std_ms` (`statistics.stdev`).
  - Dùng `numpy.percentile` để tính chuẩn xác P50, P90, P95, P99.
  - Đếm `params_m` (`sum(p.numel() for p in model.parameters()) / 1e6`).
  - **Bắt đúng đỉnh VRAM:** Dùng `torch.cuda.reset_peak_memory_stats()` và `torch.cuda.max_memory_allocated() / 1024**2`.
  - Đo `rss_mb` qua `psutil.Process().memory_info().rss / 1024**2`.
  - Tính đúng `throughput_fps` ($\frac{\text{batch} \times 1000}{\text{mean\_ms}}$).
- **Tiêu chí chấp nhận:**
  - [ ] `summary.json` chứa đủ các trường: `mean_ms`, `std_ms`, `p50_ms`, `p90_ms`, `p95_ms`, `p99_ms`, `tail_ratio`, `throughput_fps`, `params_m`, `peak_vram_mb`, `rss_mb`, `cold_start_ms`.
  - [ ] Peak VRAM không bị rỗng/bằng 0 khi chạy trên GPU có CUDA.
  - [ ] File `src/model/benchmark.py` $\le 300$ dòng code.
- **Kiểm chứng:**
  - [ ] Viết unit test mới `tests/test_benchmark_metrics.py` kiểm tra các phép tính percentiles, tail ratio, throughput, params.
  - [ ] Rule test: `python3 scripts/verify_rules.py`
- **Phụ thuộc:** Checkpoint 1.
- **Tập tin liên quan:** `src/model/benchmark.py`, `src/core/types.py`, `tests/test_benchmark_metrics.py`.
- **Độ lớn:** M (3 files).

#### Task 2.2: Nâng cấp `ResourceSampler` & Tích hợp Đo Công suất GPU (`src/model/resources.py`)
- **Mô tả:** Đảm bảo luồng nền lấy mẫu mỗi 100ms đọc đủ CPU %, RAM (MB), GPU %, VRAM (MB) và Công suất tiêu thụ GPU (W) qua `pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0`. Tích hợp tính `cosine_similarity` giữa output FP32 và FP16 (từ `src/model/metrics.py`) vào summary khi chạy FP16.
- **Tiêu chí chấp nhận:**
  - [ ] `resources.csv` có cột `power_w` khi chạy trên môi trường có NVIDIA GPU.
  - [ ] Fallback an toàn (ghi `0.0` hoặc `null`) khi chạy trên CPU Laptop hoặc khi pynvml không đọc được power.
  - [ ] File `src/model/resources.py` $\le 300$ dòng code.
- **Kiểm chứng:**
  - [ ] Manual check: chạy 1 lượt sampler 1 giây trên local và kiểm tra cấu trúc `resources.csv`.
  - [ ] Rule test: `python3 scripts/verify_rules.py`
- **Phụ thuộc:** Task 2.1.
- **Tập tin liên quan:** `src/model/resources.py`, `src/model/metrics.py`.
- **Độ lớn:** S (2 files).

#### Task 2.3: Mô-đun Benchmark Chuyên biệt cho XGBoost (`src/model/tabular_bench.py`)
- **Mô tả:** Tạo file độc lập `src/model/tabular_bench.py` ($\le 250$ dòng) để thực hiện Metric 12 theo Mục 5.2: đo độ trễ khi dự đoán 1 dòng đơn lẻ vs 10.000 dòng theo lô; đo tốc độ theo số luồng CPU `n_jobs = [1, 2, 4, -1]`; so sánh CPU vs GPU (nếu có CUDA).
- **Tiêu chí chấp nhận:**
  - [ ] Hàm `benchmark_xgboost_latency(model, x_sample)` xuất ra bảng kết quả độ trễ theo kích thước batch và số luồng.
  - [ ] Kết quả được ghi nhận vào `summary.json` của mô hình `xgboost`.
  - [ ] File tuân thủ SRP, $\le 300$ dòng code, 100% type hints.
- **Kiểm chứng:**
  - [ ] Chạy benchmark XGBoost trên tập Covertype giả lập và in kết quả log ra màn hình.
  - [ ] Rule test: `python3 scripts/verify_rules.py`
- **Phụ thuộc:** Task 2.1.
- **Tập tin liên quan:** `src/model/tabular_bench.py` (tạo mới).
- **Độ lớn:** S (1 file).

#### Task 2.4: Tái cấu trúc `scripts/run_experiment.py` — Tách Lượt Đo Sạch & Ghép `LayerPlugin`
- **Mô tả:** Tái cấu trúc luồng chạy trong `scripts/run_experiment.py`:
  1. **Lượt 1 (Model Benchmark sạch):** Chạy warmup ➔ Chạy 50 vòng đo latency không gắn bất kỳ hook nào ➔ Ghi `benchmark.csv`, `summary.json`, `resources.csv`.
  2. **Lượt 2 (Profiler Plugins):** Nếu cờ `--profilers layer` được bật, khởi tạo `from src.layer_plugin import LayerPlugin`, gọi `attach(model, device)`, chạy đo forward bằng `run_model` ➔ Ghi `layers.csv`, `layers_summary.csv`, `layer_meta.json` ➔ `detach()`.
  3. Ghi `env.json` (chuẩn hóa tên thiết bị: `cpu_laptop` hoặc `colab_t4`) và `config.json`.
  4. Chuẩn hóa tên thư mục thực nghiệm chữ HOA: `TN1`, `TN2`, `TN3`, `TN4`, `TN5` trong `build_result_path` để tránh lỗi Case-Sensitivity trên Linux.
  5. Hỗ trợ cờ `--skip-existing`.
- **Tiêu chí chấp nhận:**
  - [ ] Latency của model ở Lượt 1 không bị nhiễm bẩn bởi hook overhead của Lượt 2.
  - [ ] Lệnh `python scripts/run_experiment.py --model resnet50 --device cpu_laptop --experiment TN3 --profilers layer` tạo ra đầy đủ cả 5 file của TV2 và 3 file của TV1.
  - [ ] Không ghi đè hoặc làm hỏng dữ liệu của nhau.
  - [ ] File `scripts/run_experiment.py` $\le 300$ dòng code.
- **Kiểm chứng:**
  - [ ] Chạy lệnh CLI mẫu trên CPU Laptop và kiểm tra danh mục file trong thư mục kết quả.
  - [ ] Chạy regression test: `pytest tests/test_layer_profiler.py tests/test_layer_flops_plugin.py -q` (22 tests vẫn pass).
  - [ ] Rule test: `python3 scripts/verify_rules.py`
- **Phụ thuộc:** Task 2.1, 2.2, 2.3.
- **Tập tin liên quan:** `scripts/run_experiment.py`, `src/core/paths.py`.
- **Độ lớn:** M (2 files).

### 🛑 Checkpoint 2: Tích hợp Toàn diện (Milestone M1 Đạt chuẩn)
- [ ] Chạy thử 1 model CV, 1 NLP, 1 ML qua `scripts/run_experiment.py` trên CPU hoàn tất không lỗi.
- [ ] Thư mục kết quả `results/cpu_laptop/...` sinh đủ: `env.json`, `config.json`, `benchmark.csv`, `summary.json`, `resources.csv` và `layers.csv`.
- [ ] Đạt chuẩn giao tiếp với TV1, pass 22 tests của TV1, pass `verify_rules.py`.

---

### 🟢 Phase 3: Thực thi Thực nghiệm diện rộng (TN3, TN4, TN5) (Evaluation)

#### Task 3.1: Hoàn thiện Google Colab Runner (`notebooks/colab_runner.ipynb`)
- **Mô tả:** Hoàn thiện notebook Colab để chạy tự động trên Tesla T4 GPU: clone repo, cài `requirements.txt`, thiết lập biến môi trường `VISIONPROF_RESULTS`, mount Google Drive, chạy vòng lặp TN3, TN4, TN5 có cờ `--skip-existing`, tự động sao lưu `results/` về Google Drive.
- **Tiêu chí chấp nhận:**
  - [ ] Notebook chạy từ đầu đến cuối không bị dừng do lỗi thiếu thư viện hay đường dẫn.
  - [ ] Tự động phát hiện GPU T4 và nạp đúng `device="colab_t4"`.
- **Kiểm chứng:**
  - [ ] Mở và chạy thử nghiệm khô (dry-run) các cell trên Colab.
- **Phụ thuộc:** Checkpoint 2.
- **Tập tin liên quan:** `notebooks/colab_runner.ipynb`.
- **Độ lớn:** S (1 file).

#### Task 3.2: Chạy Thực nghiệm TN3 (Baseline 8 Mô hình x CPU Laptop & Colab T4)
- **Mô tả:** Chạy 8 mô hình (MobileNetV3-Large, ResNet-50, ViT-B/16, YOLOv8n, DistilBERT, BERT-base, MLP, XGBoost) ở cấu hình `batch_size=1`, `precision=fp32`.
  - Trên **Laptop CPU:** `python scripts/run_experiment.py --model <M> --device cpu_laptop --batch-size 1 --precision fp32 --experiment TN3`
  - Trên **Colab T4:** Chạy tương ứng với `--device colab_t4`.
- **Tiêu chí chấp nhận:**
  - [ ] Sinh đủ 16 thư mục kết quả: `results/{cpu_laptop,colab_t4}/{model}/TN3/bs1_fp32/`.
  - [ ] Đầy đủ `summary.json`, `benchmark.csv`, `resources.csv`, `env.json`, `config.json`.
- **Kiểm chứng:**
  - [ ] Script kiểm tra tính toàn vẹn: xác nhận 16 thư mục đều có file JSON/CSV hợp lệ và không rỗng.
- **Phụ thuộc:** Task 3.1.
- **Tập tin liên quan:** `results/cpu_laptop/`, `results/colab_t4/`.
- **Độ lớn:** L (Data generation).

#### Task 3.3: Chạy Thực nghiệm TN4 (Batch Scaling & Token Length)
- **Mô tả:** Đánh giá ảnh hưởng của việc tăng kích thước lô dữ liệu:
  - CV & Tabular: `batch_size = [1, 8, 32]` trên CPU và T4.
  - NLP (BERT, DistilBERT): `batch_size = [1, 8, 32]` kết hợp `seq_len = [64, 128, 256]`.
- **Tiêu chí chấp nhận:**
  - [ ] Thu thập đủ dữ liệu Throughput (ảnh/s hoặc tokens/s) qua các mức batch size để vẽ đường cong bão hòa phần cứng.
- **Kiểm chứng:**
  - [ ] Kiểm tra các thư mục `results/{hw}/{model}/TN4/bs{B}_fp32/`.
- **Phụ thuộc:** Task 3.2.
- **Tập tin liên quan:** `results/`.
- **Độ lớn:** L (Data generation).

#### Task 3.4: Chạy Thực nghiệm TN5 (So sánh FP32 vs FP16 trên Colab T4)
- **Mô tả:** Chạy trên Colab T4 cho các mô hình CV và NLP hỗ trợ FP16. So sánh tốc độ tăng tốc của Tensor Cores, % VRAM tiết kiệm được, và tính **Cosine similarity** giữa output FP32 và FP16.
- **Tiêu chí chấp nhận:**
  - [ ] Dữ liệu TN5 được lưu tại `results/colab_t4/{model}/TN5/bs1_fp16/`.
  - [ ] Có chỉ số Cosine Similarity > 0.999 chứng minh độ ổn định số học.
- **Kiểm chứng:**
  - [ ] Kiểm tra trường `cosine_similarity` trong `summary.json` của TN5.
- **Phụ thuộc:** Task 3.2.
- **Tập tin liên quan:** `results/colab_t4/`.
- **Độ lớn:** M (Data generation).

### 🛑 Checkpoint 3: Hoàn tất Thu thập Dữ liệu Thực nghiệm
- [ ] Toàn bộ các ô tích xanh (✅) thuộc quyền sở hữu của TV2 ở Mục 6.3 [PROJECT_PLAN.md](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/PROJECT_PLAN.md#L217) đều đã có dữ liệu thật trong `results/`.
- [ ] Không có file kết quả nào bị rỗng, hỏng định dạng hoặc bị thiếu trường.

---

## 🟢 Phase 4: Phân tích Dữ liệu & Hoàn thiện Dashboard (Visualization & Insights)

#### Task 4.1: Xây dựng Bộ script Phân tích & Trích xuất Biểu đồ cho CH3 & CH4
- **Mô tả:** Viết script `experiments/tv2_analysis.py` ($\le 300$ dòng) tự động đọc dữ liệu từ `results/`:
  - Phân tích **CH3:** Bảng tổng hợp Latency P50, Throughput, Peak VRAM của 8 models; tính hệ số Speedup (CPU vs GPU); trích xuất hiện tượng đảo thứ hạng (Rank Shift).
  - Phân tích **CH4:** Vẽ biểu đồ đường cong Throughput theo Batch Size (tìm điểm bão hòa); bảng so sánh FP16 vs FP32 (speedup, bộ nhớ, cosine similarity).
  - Xuất các file ảnh PNG phục vụ báo cáo vào `docs/tv2/figures/`.
- **Tiêu chí chấp nhận:**
  - [ ] Tự động tạo các hình: `fig_ch3_speedup.png`, `fig_ch3_rank_shift.png`, `fig_ch4_batch_scaling.png`, `fig_ch4_fp16.png`.
  - [ ] File tuân thủ $\le 300$ dòng, 100% type hints, dùng logging.
- **Kiểm chứng:**
  - [ ] Chạy `python experiments/tv2_analysis.py` và kiểm tra thư mục `docs/tv2/figures/`.
  - [ ] Rule test: `python3 scripts/verify_rules.py`
- **Phụ thuộc:** Checkpoint 3.
- **Tập tin liên quan:** `experiments/tv2_analysis.py` (tạo mới), `docs/tv2/figures/`.
- **Độ lớn:** M (2 files).

#### Task 4.2: Hoàn thiện Tab Dashboard 1: Tổng quan Mô hình (`src/dashboard/tabs/tab_overview.py`)
- **Mô tả:** Nâng cấp Tab 1 thành giao diện phân tích vĩ mô chuyên nghiệp:
  - Bộ lọc tương tác chọn Thiết bị (`cpu_laptop`, `colab_t4`) và Loại mô hình (CV, NLP, ML).
  - Bảng tổng hợp KPI 8 models với đầy đủ P50, Throughput, Peak VRAM, Params (M), Cold-start.
  - Biểu đồ Scatter tương tác Plotly: Trục X (Số tham số), Trục Y (Latency P50) chứng minh trực quan: *không phải cứ ít tham số là chạy nhanh*.
- **Tiêu chí chấp nhận:**
  - [ ] Tab 1 hiển thị đầy đủ số liệu thật từ `results/`, không bị lỗi thiếu cột.
  - [ ] File $\le 300$ dòng code.
- **Kiểm chứng:**
  - [ ] Chạy `streamlit run src/dashboard/app.py` và kiểm tra Tab 1.
- **Phụ thuộc:** Task 4.1.
- **Tập tin liên quan:** `src/dashboard/tabs/tab_overview.py`.
- **Độ lớn:** S (1 file).

#### Task 4.3: Hoàn thiện Tab Dashboard 2: So sánh CPU vs GPU (`src/dashboard/tabs/tab_cpu_vs_gpu.py`)
- **Mô tả:** Nâng cấp Tab 2 tập trung trả lời CH3 và CH4:
  - Bảng đối đầu CPU vs GPU với cột `Speedup = Latency_CPU / Latency_GPU`.
  - Biểu đồ Bar chart so sánh mức tăng tốc: làm nổi bật các mô hình hưởng lợi lớn nhất (ViT, BERT ~9x) vs ít nhất (MobileNet ~3x, MLP ~1x).
  - Biểu đồ Throughput theo Batch size (TN4) và bảng so sánh FP16 (TN5).
- **Tiêu chí chấp nhận:**
  - [ ] Tab 2 hiển thị rõ nét hiện tượng đảo thứ hạng và điểm bão hòa throughput.
  - [ ] File $\le 300$ dòng code.
- **Kiểm chứng:**
  - [ ] Chạy `streamlit run src/dashboard/app.py` và kiểm tra Tab 2.
- **Phụ thuộc:** Task 4.1.
- **Tập tin liên quan:** `src/dashboard/tabs/tab_cpu_vs_gpu.py`.
- **Độ lớn:** S (1 file).

### 🛑 Checkpoint 4: Trực quan hóa & Dashboard Hoàn tất
- [ ] Cả 2 Tab Dashboard của TV2 hoạt động mượt mà, trực quan, không có lỗi runtime.
- [ ] Biểu đồ thể hiện sâu sắc 2 luận điểm khoa học CH3 và CH4.
- [ ] Pass `python3 scripts/verify_rules.py`.

---

## 🟢 Phase 5: Bàn giao Báo cáo, Slide & Demo (Deliverables & Polish)

#### Task 5.1: Soạn thảo Nội dung Báo cáo thuộc phần TV2 (`docs/tv2/BAO_CAO_TV2.md`)
- **Mô tả:** Viết chi tiết các chương do TV2 phụ trách theo đúng dàn ý Mục 9 và quy tắc Mục 8.5 (mỗi câu hỏi có **1 bảng + 1 biểu đồ + 3–5 câu kết luận**):
  - **Chương 4:** Mục 4.1 Kiến trúc tổng thể framework + Mục 4.2 Thiết kế mức Model.
  - **Chương 5:** Thiết lập thực nghiệm (Mô tả 8 model, môi trường máy, quy tắc đo).
  - **Chương 6:** Mục 6.3 Kết quả CH3 + Mục 6.4 Kết quả CH4.
  - **Chương 7:** Hạn chế và hướng phát triển của mức Model.
- **Tiêu chí chấp nhận:**
  - [ ] Nội dung chi tiết, số liệu trích xuất chính xác từ `results/`, có viện dẫn hình ảnh.
- **Kiểm chứng:**
  - [ ] Review đối chiếu từng mục với [PROJECT_PLAN.md](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/PROJECT_PLAN.md).
- **Phụ thuộc:** Checkpoint 4.
- **Tập tin liên quan:** `docs/tv2/BAO_CAO_TV2.md` (tạo mới).
- **Độ lớn:** M (1 file).

#### Task 5.2: Soạn thảo Slide Thuyết trình & Kịch bản Demo (`docs/tv2/SLIDES_TV2.md`)
- **Mô tả:** Soạn 5–6 slide tập trung vào chủ đề: *"8 mô hình trên CPU và GPU: mô hình nào nhanh nhất, và vì sao thứ hạng thay đổi?"*. Kèm lời nói gợi ý (speaker notes) và kịch bản demo 1–2 phút trên Dashboard / CLI.
- **Tiêu chí chấp nhận:**
  - [ ] File slide có cấu trúc rõ ràng, sẵn sàng để TV3 ghép vào Slide chung của nhóm.
  - [ ] Có kịch bản demo từng bước rõ ràng.
- **Kiểm chứng:**
  - [ ] Đọc thử nghiệm kịch bản demo và chạy lệnh mẫu.
- **Phụ thuộc:** Task 5.1.
- **Tập tin liên quan:** `docs/tv2/SLIDES_TV2.md` (tạo mới).
- **Độ lớn:** S (1 file).

#### Task 5.3: Soạn thảo Hướng dẫn Sử dụng (`docs/tv2/README_TV2.md`)
- **Mô tả:** Viết đoạn tài liệu README hướng dẫn cách dùng `ModelBenchmark`, `ResourceSampler`, cách chạy TN3, TN4, TN5 để bàn giao cho TV3 tích hợp vào `README.md` chính.
- **Tiêu chí chấp nhận:**
  - [ ] Hướng dẫn ngắn gọn, có code mẫu chạy được ngay.
- **Kiểm chứng:**
  - [ ] Chạy thử các câu lệnh mẫu trong README.
- **Phụ thuộc:** Task 5.1.
- **Tập tin liên quan:** `docs/tv2/README_TV2.md` (tạo mới).
- **Độ lớn:** XS (1 file).

### 🛑 Checkpoint 5: Nghiệm thu Toàn diện Sản phẩm TV2
- [ ] Code Model Zoo, Benchmark, Runner hoàn thiện 100%.
- [ ] Dữ liệu TN3, TN4, TN5 đầy đủ trong `results/`.
- [ ] 2 Tab Dashboard hiển thị ấn tượng.
- [ ] Báo cáo, Slide, README đầy đủ, sẵn sàng cho buổi tổng duyệt nhóm.
- [ ] Toàn bộ test và `scripts/verify_rules.py` pass 100%.

---

## 4. Bảng Quản trị Rủi ro & Biện pháp Phòng ngừa (Risk Matrix)

| Rủi ro tiềm ẩn | Mức độ | Biện pháp phòng ngừa chủ động |
| :--- | :---: | :--- |
| **Phân tách nhánh Git (tv1-layer vs main)** | **Nghiêm trọng (High)** | Thêm Phase 0: Tạo nhánh `feat/tv2-model-benchmark` từ `main`, merge `tv1-layer` vào trước khi code. |
| **Bỏ sót đỉnh Peak VRAM do luồng 100ms** | **Nghiêm trọng (High)** | Bắt buộc dùng `torch.cuda.max_memory_allocated()` của PyTorch allocator thay vì chỉ dựa vào `ResourceSampler`. |
| **Lỗi Case-Sensitivity trên Colab Linux (`TN1` vs `tn1`)** | **Cao (High)** | Thống nhất chuẩn tên thực nghiệm viết HOA trong `build_result_path`: `TN1`..`TN5`. |
| **Đo Latency bị nhiễm bẩn bởi hook overhead** | **Nghiêm trọng (High)** | Tách `run_experiment.py` thành 2 pha độc lập: Pha 1 đo sạch (P50, throughput), Pha 2 mới gắn `LayerPlugin`. |
| **Colab ngắt kết nối / Hết quota GPU giữa chừng** | **Cao (High)** | Lưu kết quả tức thời sau mỗi model; sử dụng cờ `--skip-existing`; mount Google Drive trực tiếp. |
| **Xung đột tên hàm / kiểu dữ liệu với TV1** | **Cao (High)** | Export alias `get_adapter = get_model_adapter`; trả về pure `nn.Module`; hỗ trợ `**kwargs` cho `seq_len`. |
| **Lỗi đồng bộ ngầm khi đo BERT trên GPU** | **Cao (High)** | Tuyệt đối không truyền `attention_mask` khi input không padding. |
| **Dữ liệu giả lập XGBoost quá nông (cây rỗng)** | **Trung bình (Med)** | Khởi tạo giả lập tối thiểu 1.000 mẫu để cây đạt độ sâu 6 đúng chuẩn Covertype. |
| **Dao động nhiệt độ / throttling trên Laptop CPU** | **Trung bình (Med)** | Cắm sạc laptop, bật chế độ hiệu năng cao, cố định số thread CPU, đo lấy trung vị P50 thay vì trung bình. |
| **Lỗi cài đặt XGBoost GPU trên Colab** | **Thấp (Low)** | Thiết lập cấu hình `tree_method="hist"`, `device="cuda"`; có khối try-except fallback an toàn. |
| **Vi phạm quy chuẩn độ dài file > 300 dòng** | **Trung bình (Med)** | Tách nhỏ file theo SRP (`tabular_bench.py`), kiểm tra liên tục qua `scripts/verify_rules.py`. |
