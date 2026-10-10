# Danh sách Công việc (TODO Checklist) — TV2 (Mức Model)

## Quy chuẩn bắt buộc trước khi commit/merge:
- [ ] File mới/sửa đổi $\le 300$ dòng code (khuyến nghị $\le 250$ dòng).
- [ ] 100% type hints cho hàm/phương thức public trong `src/`.
- [ ] Tuyệt đối không dùng lệnh bare `print()`, dùng `logging`.
- [ ] Lệnh `python3 scripts/verify_rules.py` trả về exit code 0.

---

## Phase 0: Đồng bộ Git & Hợp nhất Mã nguồn (Pre-requisite Setup)
- [ ] **Task 0.1:** Tạo nhánh làm việc và Hợp nhất `main` với `tv1-layer`
  - [ ] Checkout sang `main`, kéo code mới nhất (`git checkout main && git pull`).
  - [ ] Tạo nhánh mới: `git checkout -b feat/tv2-model-benchmark`.
  - [ ] Merge nhánh `tv1-layer` vào (`git merge tv1-layer`).
  - [ ] Giải quyết xung đột trong `requirements.txt` (giữ đủ các gói của cả 2 nhánh).
  - [ ] Giải quyết xung đột trong `tests/test_layer_profiler.py` (chọn bản 11 test chuẩn của TV1).
  - [ ] Xác nhận pass 22 unit test của TV1: `pytest tests/test_layer_profiler.py tests/test_layer_flops_plugin.py -q`.
- [ ] **Checkpoint 0:** Nhánh `feat/tv2-model-benchmark` có đầy đủ cả `src/zoo/` (từ `main`) và `src/layer_plugin.py` (từ `tv1-layer`), test xanh 100%.

---

## Phase 1: Chuẩn hóa Model Zoo & Hợp đồng Bàn giao (Foundation)
- [ ] **Task 1.1:** Chuẩn hóa Model Adapters Computer Vision (`src/zoo/cv.py`)
  - [ ] Đổi `mobilenet_v3_small` thành `mobilenet_v3_large` (5.5M params).
  - [ ] Đảm bảo `YOLOv8nAdapter.load()` trả về pure `nn.Module` (`yolo.model`).
  - [ ] Thêm `**kwargs` vào `make_input` để nhận `seq_len` không bị lỗi.
  - [ ] Giữ đúng kích thước input: `(B, 3, 224, 224)` cho CV, `(B, 3, 640, 640)` cho YOLO.
  - [ ] Cài đặt `train_step()` cho `resnet50` và `mobilenet_v3_large` (phục vụ TV3 TN6).
  - [ ] Cài đặt `preprocess()` và `postprocess()` (NMS) cho `yolov8n` (phục vụ TV3 TN7).
  - [ ] Kiểm tra file $\le 300$ dòng code và pass `scripts/verify_rules.py`.
- [ ] **Task 1.2:** Chuẩn hóa Model Adapters NLP & Xử lý bẫy Đồng bộ Ngầm (`src/zoo/nlp.py`)
  - [ ] Loại bỏ `attention_mask` khỏi dictionary trả về của `make_input` trong `BERTBaseAdapter` và `DistilBERTAdapter` (chỉ trả `{"input_ids": ids}`).
  - [ ] Thêm `**kwargs` vào `make_input`.
  - [ ] Cài đặt `train_step()` cho `distilbert` (phục vụ TV3 TN6) và `preprocess/postprocess` cho `bert_base` (phục vụ TV3 TN7).
  - [ ] Kiểm tra file $\le 300$ dòng code và pass `scripts/verify_rules.py`.
- [ ] **Task 1.3:** Chuẩn hóa Model Adapters Tabular & Tạo Cây Thực tế (`src/zoo/tabular.py`, `src/zoo/base.py`)
  - [ ] Kiểm tra `MLPAdapter` (54 đặc trưng, 7 lớp).
  - [ ] Cài đặt `train_step()` cho `MLPAdapter` (phục vụ TV3 TN6).
  - [ ] Đảm bảo `XGBoostAdapter` khởi tạo tối thiểu 1.000 mẫu giả lập để cây đạt độ sâu 6 chuẩn Covertype.
  - [ ] Hỗ trợ `device="cuda"` và `"cpu"` trên tập Covertype, hỗ trợ `fit` (phục vụ TV3 TN6).
  - [ ] Kiểm tra file $\le 300$ dòng code và pass `scripts/verify_rules.py`.
- [ ] **Task 1.4:** Cập nhật Registry và Export Giao thức Chung (`src/zoo/__init__.py`)
  - [ ] Đăng ký đủ 8 model: `mobilenet_v3_large`, `resnet50`, `vit_b_16`, `yolov8n`, `distilbert`, `bert_base`, `mlp`, `xgboost` (kèm alias `mobilenet_v3`).
  - [ ] Export alias `get_adapter = get_model_adapter`.
  - [ ] Cập nhật unit test `tests/test_model_zoo.py`.
  - [ ] Kiểm tra tương thích với script TV1: `demo_layer_profiler.py` không rơi vào `_fallback_model`.
  - [ ] Chạy `python3 scripts/verify_rules.py` pass 100%.
- [ ] **Checkpoint 1:** Kiểm tra Model Zoo qua `tests/test_model_zoo.py`, demo script TV1 chạy mượt mà, `verify_rules.py` pass.

---

## Phase 2: Hoàn thiện Bộ đo Benchmark, Giám sát Tài nguyên & Runner (Core Harness)
- [ ] **Task 2.1:** Hoàn thiện `ModelBenchmark` Đủ 12 Metrics (`src/model/benchmark.py`)
  - [ ] Tính thêm `std_ms`.
  - [ ] Dùng `numpy.percentile` tính P50, P90, P95, P99.
  - [ ] Tính `params_m` (`sum(p.numel()) / 1e6`).
  - [ ] Bắt đúng đỉnh VRAM bằng `torch.cuda.reset_peak_memory_stats()` và `torch.cuda.max_memory_allocated() / 1024**2`.
  - [ ] Đo `rss_mb` bằng `psutil.Process().memory_info().rss / 1024**2`.
  - [ ] Viết unit test `tests/test_benchmark_metrics.py`.
  - [ ] Kiểm tra file $\le 300$ dòng code và pass `scripts/verify_rules.py`.
- [ ] **Task 2.2:** Nâng cấp `ResourceSampler` & Tích hợp Đo Công suất GPU (`src/model/resources.py`)
  - [ ] Đọc thêm Công suất GPU (`power_w`) qua `pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0`.
  - [ ] Tích hợp tính `cosine_similarity` giữa output FP32 và FP16.
  - [ ] Kiểm tra file $\le 300$ dòng code và pass `scripts/verify_rules.py`.
- [ ] **Task 2.3:** Mô-đun Benchmark Chuyên biệt cho XGBoost (`src/model/tabular_bench.py`)
  - [ ] Tạo file mới `src/model/tabular_bench.py` ($\le 250$ dòng code).
  - [ ] Đo độ trễ dự đoán 1 dòng vs 10.000 dòng.
  - [ ] Đo mở rộng số luồng CPU (`n_jobs = [1, 2, 4, -1]`).
  - [ ] Ghi kết quả vào `summary.json` của `xgboost`.
  - [ ] Kiểm tra pass `scripts/verify_rules.py`.
- [ ] **Task 2.4:** Tái cấu trúc `scripts/run_experiment.py`
  - [ ] Tách làm 2 phase: Pha 1 đo Benchmark sạch ➔ Pha 2 mới gắn `LayerPlugin`.
  - [ ] Import đúng `from src.layer_plugin import LayerPlugin`.
  - [ ] Chuẩn hóa tên thiết bị: `cpu_laptop`, `colab_t4` (fallback `colab_cpu`).
  - [ ] Chuẩn hóa tên thư mục thực nghiệm chữ HOA: `TN1`..`TN5` trong `src/core/paths.py`.
  - [ ] Lưu kết quả đúng chuẩn thư mục Mục 8.1 ③ và hỗ trợ cờ `--skip-existing`.
  - [ ] Kiểm tra file $\le 300$ dòng code và pass `scripts/verify_rules.py`.
- [ ] **Checkpoint 2:** Chạy thử nghiệm CLI 1 model trên CPU Laptop thành công, pass 22 unit test của TV1, pass `verify_rules.py`.

---

## Phase 3: Thực thi Thực nghiệm diện rộng (TN3, TN4, TN5) (Evaluation)
- [ ] **Task 3.1:** Hoàn thiện Google Colab Runner (`notebooks/colab_runner.ipynb`)
  - [ ] Thiết lập cài đặt môi trường, mount Google Drive, đồng bộ kết quả.
  - [ ] Thêm lệnh chạy hàng loạt cho 8 model với `--skip-existing`.
- [ ] **Task 3.2:** Chạy Thực nghiệm TN3 (Baseline 8 Mô hình x CPU Laptop & Colab T4)
  - [ ] Chạy 8 model trên CPU Laptop (bs=1, fp32).
  - [ ] Chạy 8 model trên Colab T4 (bs=1, fp32).
  - [ ] Kiểm tra đủ 16 thư mục kết quả trong `results/`.
- [ ] **Task 3.3:** Chạy Thực nghiệm TN4 (Batch Scaling & Token Length)
  - [ ] Đo CV & Tabular ở `batch_size = [1, 8, 32]`.
  - [ ] Đo NLP ở `batch_size = [1, 8, 32]` và `seq_len = [64, 128, 256]`.
  - [ ] Lưu đầy đủ kết quả vào `results/`.
- [ ] **Task 3.4:** Chạy Thực nghiệm TN5 (So sánh FP32 vs FP16 trên Colab T4)
  - [ ] Chạy các mô hình CV và NLP hỗ trợ FP16 trên Colab T4.
  - [ ] Tính Cosine Similarity và kiểm tra mức tiết kiệm VRAM.
- [ ] **Checkpoint 3:** Kiểm tra tính toàn vẹn của dữ liệu trong `results/` (đầy đủ các ô ở Mục 6.3).

---

## Phase 4: Phân tích Dữ liệu & Hoàn thiện Dashboard (Visualization & Insights)
- [ ] **Task 4.1:** Xây dựng Script Phân tích & Trích xuất Biểu đồ (`experiments/tv2_analysis.py`)
  - [ ] Phân tích CH3: bảng Speedup, bảng thay đổi thứ hạng (Rank Shift).
  - [ ] Phân tích CH4: biểu đồ Throughput vs Batch Size, bảng so sánh FP16.
  - [ ] Xuất ảnh PNG vào `docs/tv2/figures/`.
  - [ ] Kiểm tra file $\le 300$ dòng code và pass `scripts/verify_rules.py`.
- [ ] **Task 4.2:** Hoàn thiện Tab Dashboard 1: Tổng quan Mô hình (`src/dashboard/tabs/tab_overview.py`)
  - [ ] Bổ sung bộ lọc Thiết bị và Loại mô hình.
  - [ ] Bảng tổng hợp KPI 8 models với đầy đủ 12 metrics.
  - [ ] Biểu đồ Scatter Params vs Latency P50.
  - [ ] Kiểm tra file $\le 300$ dòng code.
- [ ] **Task 4.3:** Hoàn thiện Tab Dashboard 2: So sánh CPU vs GPU (`src/dashboard/tabs/tab_cpu_vs_gpu.py`)
  - [ ] Biểu đồ Bar chart Speedup và biểu đồ thể hiện hiện tượng đảo thứ hạng.
  - [ ] Biểu đồ đường cong Throughput theo Batch size và bảng FP16.
  - [ ] Kiểm tra file $\le 300$ dòng code.
- [ ] **Checkpoint 4:** Kiểm tra 2 Tab Dashboard trên Streamlit với dữ liệu thật chạy trơn tru, pass `verify_rules.py`.

---

## Phase 5: Bàn giao Báo cáo, Slide & Demo (Deliverables & Polish)
- [ ] **Task 5.1:** Soạn thảo Nội dung Báo cáo TV2 (`docs/tv2/BAO_CAO_TV2.md`)
  - [ ] Chương 4: Mục 4.1 Kiến trúc tổng + Mục 4.2 Thiết kế mức Model.
  - [ ] Chương 5: Thiết lập thực nghiệm (8 models, cấu hình máy, quy tắc đo).
  - [ ] Chương 6: Mục 6.3 Kết quả CH3 + Mục 6.4 Kết quả CH4 (đủ 1 bảng + 1 biểu đồ + 3-5 câu kết luận mỗi mục).
  - [ ] Chương 7: Hạn chế và hướng phát triển của mức Model.
- [ ] **Task 5.2:** Soạn thảo Slide Thuyết trình & Kịch bản Demo (`docs/tv2/SLIDES_TV2.md`)
  - [ ] Soạn 5-6 slide mức Model kèm lời thoại gợi ý (speaker notes).
  - [ ] Viết kịch bản demo 1-2 phút trên CLI / Dashboard.
- [ ] **Task 5.3:** Soạn thảo Đoạn README (`docs/tv2/README_TV2.md`)
  - [ ] Hướng dẫn cài đặt, chạy ModelBenchmark và các lệnh thực nghiệm.
- [ ] **Checkpoint 5:** Tổng duyệt nghiệm thu toàn bộ sản phẩm của TV2 trước khi ghép chung với nhóm, pass 100% `scripts/verify_rules.py`.
