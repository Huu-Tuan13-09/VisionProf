# Danh sách Công việc (TODO Checklist) — TV2 (Mức Model)

## Quy chuẩn bắt buộc trước khi commit/merge:
- [x] File mới/sửa đổi $\le 300$ dòng code (khuyến nghị $\le 250$ dòng).
- [x] 100% type hints cho hàm/phương thức public trong `src/`.
- [x] Tuyệt đối không dùng lệnh bare `print()`, dùng `logging`.
- [x] Lệnh `python3 scripts/verify_rules.py` trả về exit code 0.

---

## Phase 0: Đồng bộ Git & Hợp nhất Mã nguồn (ĐÃ HOÀN THÀNH 100% TRÊN MAIN)
- [x] **Task 0.1:** Hợp nhất mã nguồn TV1 (`feature/integrate-tv1-modular`) vào `main`
  - [x] Checkout sang `main`, kéo code mới nhất (`git checkout main && git pull`).
  - [x] Đã hoàn tất tích hợp TV1 theo chuẩn Modular v2.0 (`src/layer/`) và merge sạch vào `main`.
  - [x] Đã gộp và đồng bộ toàn bộ dependencies trong `requirements.txt`.
  - [x] Đã pass toàn bộ 38 unit tests và kiểm tra `python3 scripts/verify_rules.py` đạt 100%.
- [x] **Checkpoint 0:** Nhánh `main` đã có đầy đủ `src/zoo/`, `src/model/`, `src/layer/`, `src/phase/`, `src/analyzer/`, `src/dashboard/`. TV2 chỉ cần tạo nhánh làm việc mới: `git checkout -b feature/tv2-model-benchmark`.

---

## Phase 1: Chuẩn hóa Model Zoo & Hợp đồng Bàn giao (Foundation)
- [x] **Task 1.1:** Chuẩn hóa Model Adapters Computer Vision (`src/zoo/cv.py`)
  - [x] Đổi `mobilenet_v3_small` thành `mobilenet_v3_large` (5.5M params).
  - [x] Đảm bảo `YOLOv8nAdapter.load()` trả về pure `nn.Module` (`yolo.model`).
  - [x] Thêm `**kwargs` vào `make_input` để nhận `seq_len` không bị lỗi.
  - [x] Giữ đúng kích thước input: `(B, 3, 224, 224)` cho CV, `(B, 3, 640, 640)` cho YOLO.
  - [x] Cài đặt `train_step()` cho `resnet50` và `mobilenet_v3_large` (phục vụ TV3 TN6).
  - [x] Cài đặt `preprocess()` và `postprocess()` (NMS) cho `yolov8n` (phục vụ TV3 TN7).
  - [x] Kiểm tra file $\le 300$ dòng code và pass `scripts/verify_rules.py`.
- [x] **Task 1.2:** Chuẩn hóa Model Adapters NLP & Xử lý bẫy Đồng bộ Ngầm (`src/zoo/nlp.py`)
  - [x] Loại bỏ `attention_mask` khỏi dictionary trả về của `make_input` trong `BERTBaseAdapter` và `DistilBERTAdapter` (chỉ trả `{"input_ids": ids}`).
  - [x] Thêm `**kwargs` vào `make_input`.
  - [x] Cài đặt `train_step()` cho `distilbert` (phục vụ TV3 TN6) và `preprocess/postprocess` cho `bert_base` (phục vụ TV3 TN7).
  - [x] Kiểm tra file $\le 300$ dòng code và pass `scripts/verify_rules.py`.
- [x] **Task 1.3:** Chuẩn hóa Model Adapters Tabular & Tạo Cây Thực tế (`src/zoo/tabular.py`, `src/zoo/base.py`)
  - [x] Kiểm tra `MLPAdapter` (54 đặc trưng, 7 lớp).
  - [x] Cài đặt `train_step()` cho `MLPAdapter` (phục vụ TV3 TN6).
  - [x] Đảm bảo `XGBoostAdapter` khởi tạo tối thiểu 1.000 mẫu giả lập để cây đạt độ sâu 6 chuẩn Covertype.
  - [x] Hỗ trợ `device="cuda"` và `"cpu"` trên tập Covertype, hỗ trợ `fit` (phục vụ TV3 TN6).
  - [x] Kiểm tra file $\le 300$ dòng code và pass `scripts/verify_rules.py`.
- [x] **Task 1.4:** Cập nhật Registry và Export Giao thức Chung (`src/zoo/__init__.py`)
  - [x] Đăng ký đủ 8 model: `mobilenet_v3_large`, `resnet50`, `vit_b_16`, `yolov8n`, `distilbert`, `bert_base`, `mlp`, `xgboost` (kèm alias `mobilenet_v3`).
  - [x] Export alias `get_adapter = get_model_adapter`.
  - [x] Cập nhật unit test `tests/test_model_zoo.py`.
  - [x] Kiểm tra tương thích với script TV1: `demo_layer_profiler.py` không rơi vào `_fallback_model`.
  - [x] Chạy `python3 scripts/verify_rules.py` pass 100%.
- [x] **Checkpoint 1:** Kiểm tra Model Zoo qua `tests/test_model_zoo.py`, demo script TV1 chạy mượt mà, `verify_rules.py` pass.

---

## Phase 2: Hoàn thiện Bộ đo Benchmark, Giám sát Tài nguyên & Runner (Core Harness)
- [x] **Task 2.1:** Hoàn thiện `ModelBenchmark` Đủ 12 Metrics (`src/model/benchmark.py`)
  - [x] Tính thêm `std_ms`.
  - [x] Dùng `numpy.percentile` tính P50, P90, P95, P99.
  - [x] Tính `params_m` (`sum(p.numel()) / 1e6`).
  - [x] Bắt đúng đỉnh VRAM bằng `torch.cuda.reset_peak_memory_stats()` và `torch.cuda.max_memory_allocated() / 1024**2`.
  - [x] Đo `rss_mb` bằng `psutil.Process().memory_info().rss / 1024**2`.
  - [x] Viết unit test `tests/test_benchmark_metrics.py`.
  - [x] Kiểm tra file $\le 300$ dòng code và pass `scripts/verify_rules.py`.
- [x] **Task 2.2:** Nâng cấp `ResourceSampler` & Tích hợp Đo Công suất GPU (`src/model/resources.py`)
  - [x] Đọc thêm Công suất GPU (`power_w`) qua `pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0`.
  - [x] Tích hợp tính `cosine_similarity` giữa output FP32 và FP16.
  - [x] Kiểm tra file $\le 300$ dòng code và pass `scripts/verify_rules.py`.
- [x] **Task 2.3:** Mô-đun Benchmark Chuyên biệt cho XGBoost (`src/model/tabular_bench.py`)
  - [x] Tạo file mới `src/model/tabular_bench.py` ($\le 250$ dòng code).
  - [x] Đo độ trễ dự đoán 1 dòng vs 10.000 dòng.
  - [x] Đo mở rộng số luồng CPU (`n_jobs = [1, 2, 4, -1]`).
  - [x] Ghi kết quả vào `summary.json` của `xgboost`.
  - [x] Kiểm tra pass `scripts/verify_rules.py`.
- [x] **Task 2.4:** Tái cấu trúc `scripts/run_experiment.py`
  - [x] Tách làm 2 phase: Pha 1 đo Benchmark sạch ➔ Pha 2 mới gắn `LayerPlugin`.
  - [x] Import đúng `from src.layer.plugin import LayerProfilerPlugin` (hoặc `from src.layer import LayerProfilerPlugin`).
  - [x] Chuẩn hóa tên thiết bị: `cpu_laptop`, `colab_t4` (fallback `colab_cpu`).
  - [x] Chuẩn hóa tên thư mục thực nghiệm chữ HOA: `TN1`..`TN5` trong `src/core/paths.py`.
  - [x] Lưu kết quả đúng chuẩn thư mục Mục 8.1 ③ và hỗ trợ cờ `--skip-existing`.
  - [x] Kiểm tra file $\le 300$ dòng code và pass `scripts/verify_rules.py`.
- [x] **Checkpoint 2:** Chạy thử nghiệm CLI 1 model trên CPU Laptop thành công, pass toàn bộ unit tests, pass `verify_rules.py`.

---

## Phase 3: Thực thi Thực nghiệm diện rộng (TN3, TN4, TN5) (Evaluation)
- [x] **Task 3.1:** Hoàn thiện Google Colab Runner (`notebooks/colab_runner.ipynb`)
  - [x] Thiết lập cài đặt môi trường, mount Google Drive, đồng bộ kết quả.
  - [x] Thêm lệnh chạy hàng loạt cho 8 model với `--skip-existing`.
- [x] **Task 3.2:** Chạy Thực nghiệm TN3 (Baseline 8 Mô hình x CPU Laptop & Colab T4)
  - [x] Chạy 8 model trên CPU Laptop (bs=1, fp32).
  - [x] Chạy 8 model trên Colab T4 (bs=1, fp32).
  - [x] Kiểm tra đủ 16 thư mục kết quả trong `results/`.
- [x] **Task 3.3:** Chạy Thực nghiệm TN4 (Batch Scaling & Token Length)
  - [x] Đo CV & Tabular ở `batch_size = [1, 8, 32]`.
  - [x] Đo NLP ở `batch_size = [1, 8, 32]` và `seq_len = [64, 128, 256]`.
  - [x] Lưu đầy đủ kết quả vào `results/`.
- [x] **Task 3.4:** Chạy Thực nghiệm TN5 (So sánh FP32 vs FP16 trên Colab T4)
  - [x] Chạy các mô hình CV và NLP hỗ trợ FP16 trên Colab T4.
  - [x] Tính Cosine Similarity và kiểm tra mức tiết kiệm VRAM.
- [x] **Checkpoint 3:** Kiểm tra tính toàn vẹn của dữ liệu trong `results/` (đầy đủ các ô ở Mục 6.3).

---

## Phase 4: Phân tích Dữ liệu & Hoàn thiện Dashboard (Visualization & Insights)
- [x] **Task 4.1:** Xây dựng Script Phân tích & Trích xuất Biểu đồ (`experiments/tv2_analysis.py`)
  - [x] Phân tích CH3: bảng Speedup, bảng thay đổi thứ hạng (Rank Shift).
  - [x] Phân tích CH4: biểu đồ Throughput vs Batch Size, bảng so sánh FP16.
  - [x] Xuất ảnh PNG vào `docs/tv2/figures/`.
  - [x] Kiểm tra file $\le 300$ dòng code và pass `scripts/verify_rules.py`.
- [x] **Task 4.2:** Hoàn thiện Tab Dashboard 1: Tổng quan Mô hình (`src/dashboard/tabs/tab_overview.py`)
  - [x] Bổ sung bộ lọc Thiết bị và Loại mô hình.
  - [x] Bảng tổng hợp KPI 8 models với đầy đủ 12 metrics.
  - [x] Biểu đồ Scatter Params vs Latency P50.
  - [x] Kiểm tra file $\le 300$ dòng code.
- [x] **Task 4.3:** Hoàn thiện Tab Dashboard 2: So sánh CPU vs GPU (`src/dashboard/tabs/tab_cpu_vs_gpu.py`)
  - [x] Biểu đồ Bar chart Speedup và biểu đồ thể hiện hiện tượng đảo thứ hạng.
  - [x] Biểu đồ đường cong Throughput theo Batch size và bảng FP16.
  - [x] Kiểm tra file $\le 300$ dòng code.
- [x] **Checkpoint 4:** Kiểm tra 2 Tab Dashboard trên Streamlit với dữ liệu thật chạy trơn tru, pass `verify_rules.py`.

---

## Phase 5: Bàn giao Báo cáo, Slide & Demo (Deliverables & Polish)
- [x] **Task 5.1:** Soạn thảo Nội dung Báo cáo TV2 (`docs/tv2/BAO_CAO_TV2.md`)
  - [x] Chương 4: Mục 4.1 Kiến trúc tổng + Mục 4.2 Thiết kế mức Model.
  - [x] Chương 5: Thiết lập thực nghiệm (8 models, cấu hình máy, quy tắc đo).
  - [x] Chương 6: Mục 6.3 Kết quả CH3 + Mục 6.4 Kết quả CH4 (đủ 1 bảng + 1 biểu đồ + 3-5 câu kết luận mỗi mục).
  - [x] Chương 7: Hạn chế và hướng phát triển của mức Model.
- [x] **Task 5.2:** Soạn thảo Slide Thuyết trình & Kịch bản Demo (`docs/tv2/SLIDES_TV2.md`)
  - [x] Soạn 5-6 slide mức Model kèm lời thoại gợi ý (speaker notes).
  - [x] Viết kịch bản demo 1-2 phút trên CLI / Dashboard.
- [x] **Task 5.3:** Soạn thảo Đoạn README (`docs/tv2/README_TV2.md`)
  - [x] Hướng dẫn cài đặt, chạy ModelBenchmark và các lệnh thực nghiệm.
- [x] **Checkpoint 5:** Tổng duyệt nghiệm thu toàn bộ sản phẩm của TV2 trước khi ghép chung với nhóm, pass 100% `scripts/verify_rules.py`.
