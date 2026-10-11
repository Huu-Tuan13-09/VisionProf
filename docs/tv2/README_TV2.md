# HƯỚNG DẪN SỬ DỤNG MỨC MODEL (TV2) — VISIONPROF

Tài liệu hướng dẫn sử dụng các thành phần thuộc quyền sở hữu của **Thành viên 2 (Mức Model)**: Model Zoo, bộ đo `ModelBenchmark`, bộ giám sát `ResourceSampler`, và các công cụ thực thi TN3, TN4, TN5.

---

## 1. Cài Đặt Môi Trường
```bash
# Kích hoạt virtualenv và cài đặt dependencies
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Sử Dụng Model Zoo (`src/zoo`)
Model Zoo cung cấp giao thức chung `ModelAdapter` cho 8 mô hình đa lĩnh vực:

```python
from src.zoo import get_adapter

# 1. Khởi tạo adapter cho mô hình bất kỳ
adapter = get_adapter("resnet50")  # hoặc mobilenet_v3_large, vit_b_16, bert_base, mlp, xgboost

# 2. Nạp mô hình vào thiết bị (cpu hoặc cuda/colab_t4)
model = adapter.load(device="cpu", precision="fp32")

# 3. Tạo mock batched input chuẩn hóa (hỗ trợ **kwargs)
inputs = adapter.make_input(batch_size=8, device="cpu", precision="fp32")

# 4. Chạy forward inference
outputs = adapter.forward(model, inputs)
```

---

## 3. Chạy Đo Benchmark Toàn Bộ Mô Hình (`ModelBenchmark`)
`ModelBenchmark` tự động thực hiện warm-up, đo lường phân vị độ trễ (P50..P99), bắt đỉnh VRAM từ GPU Allocator, và thu thập 12 metrics:

```python
from pathlib import Path
from src.zoo import get_adapter
from src.model.benchmark import ModelBenchmark

adapter = get_adapter("mobilenet_v3_large")
model = adapter.load("cpu")

bench = ModelBenchmark(warmup_iters=10, measure_iters=50)

# Khởi động warm-up
bench.run_warmup(adapter, model, batch_size=1, device="cpu")

# Đo lường sạch không gắn hook
metrics = bench.measure(adapter, model, batch_size=1, device="cpu")

# Tổng hợp 12 metrics và lưu kết quả
summary = bench.summarize(batch_size=1)
bench.save(Path("results/my_run"), summary)
```

---

## 4. Chạy Thực Nghiệm Qua Dòng Lệnh (CLI Runner)
Sử dụng `scripts/run_experiment.py` để chạy tự động theo chuẩn cấu trúc `results/`:

```bash
# 1. Chạy Baseline TN3 (Batch 1, FP32 trên CPU)
python scripts/run_experiment.py --model resnet50 --device cpu_laptop --batch-size 1 --precision fp32 --experiment TN3

# 2. Chạy Mở rộng Batch Size TN4 (Batch 32)
python scripts/run_experiment.py --model vit_b_16 --device cpu_laptop --batch-size 32 --precision fp32 --experiment TN4

# 3. Chạy So sánh FP16 Tensor Cores TN5 (trên GPU Colab T4)
python scripts/run_experiment.py --model bert_base --device colab_t4 --batch-size 1 --precision fp16 --experiment TN5
```

---

## 5. Chạy Phân Tích & Mở Dashboard
```bash
# Tự động xuất 4 biểu đồ báo cáo CH3 và CH4 vào docs/tv2/figures/
python experiments/tv2_analysis.py

# Khởi chạy Streamlit Dashboard xem Tab 1 & Tab 2
streamlit run src/dashboard/app.py
```
