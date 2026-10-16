# HƯỚNG DẪN KỸ THUẬT MỨC GIAI ĐOẠN (TV3) — VISIONPROF

Tài liệu hướng dẫn sử dụng và tra cứu kỹ thuật các thành phần thuộc phạm vi trách nhiệm của **Thành viên 3 (Mức Giai đoạn, Chẩn đoán & Dashboard)**:
- Bộ đo giai đoạn: `PhaseTimer`, `DataLoaderSentinel`, `PhaseProfilerPlugin` (`src/phase/`)
- Động cơ chẩn đoán & so sánh A/B: `DiagnosticEngine`, `ABComparisonEngine` (`src/analyzer/`)
- Scripts thực nghiệm: `scripts/run_tn6_training.py`, `scripts/run_tn7_pipeline.py`
- Giao diện Dashboard: Tabs 5, 6, 7, 8 và Router `src/dashboard/app.py`

---

## 1. Cài Đặt & Khởi Tạo Môi Trường
```bash
# Kích hoạt virtualenv (PowerShell trên Windows)
.\.venv\Scripts\activate

# Hoặc trên Linux / Google Colab:
source .venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt
```

---

## 2. Sử Dụng Bộ Đo Giai Đoạn (`src/phase/`)

### 2.1. Đo lường tùy biến bằng `PhaseTimer` (Context Manager Pattern)
`PhaseTimer` hỗ trợ đo các khối lệnh logic phức tạp, tự động đồng bộ CUDA stream chính xác khi chạy trên GPU:

```python
import torch
from src.phase.timer import PhaseTimer

timer = PhaseTimer(device="cuda")  # hoặc "cpu"

for epoch in range(num_epochs):
    timer.start_iteration(epoch)
    
    with timer.phase("forward"):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
    with timer.phase("backward"):
        loss.backward()
        
    with timer.phase("optimizer"):
        optimizer.step()
        optimizer.zero_grad()
        
    timer.end_iteration()

# Xuất kết quả ra pandas DataFrame
df_phases = timer.to_dataframe()
# In thống kê trung vị các phase (ms)
print(df_phases.groupby("phase")["time_ms"].median())
```

### 2.2. Giám sát I/O và nghẽn luồng với `DataLoaderSentinel`
Tự động đo thời gian lấy batch dữ liệu và loại trừ nhiễu khởi tạo worker ở batch 0:

```python
from torch.utils.data import DataLoader
from src.phase.dataloader import DataLoaderSentinel

loader = DataLoader(dataset, batch_size=32, num_workers=2)
sentinel = DataLoaderSentinel(loader)

for batch in sentinel:
    # Huấn luyện mô hình...
    pass

print(f"Thời gian I/O trung bình (đã loại batch 0): {sentinel.mean_io_ms():.2f} ms")
```

### 2.3. Tích hợp Plugin trong quy trình chuẩn (`PhaseProfilerPlugin`)
```python
from pathlib import Path
from src.phase.plugin import PhaseProfilerPlugin

plugin = PhaseProfilerPlugin()
plugin.attach(model, device="cuda")

for i in range(30):
    plugin.start(i)
    # Đo đạc và ghi nhận pha...
    plugin.record_phase("forward", fwd_ms)
    plugin.record_phase("backward", bwd_ms)
    plugin.stop()

# Tự động xuất file phases.csv theo schema chuẩn
plugin.save(Path("results/my_run"))
```

---

## 3. Chạy Thực Nghiệm TN6 & TN7

### 3.1. Kịch bản TN6 — Training Phase Breakdown (CH5)
Đo phân rã Forward / Backward / Optimizer và tỷ số bộ nhớ Train/Infer trên 5 mô hình (ResNet-50, MobileNetV3, DistilBERT, MLP, XGBoost):

```bash
# Chạy toàn bộ 5 mô hình trên GPU Tesla T4 (Google Colab)
python scripts/run_tn6_training.py --device colab_t4 --batch-size 8

# Chạy trên CPU để so sánh
python scripts/run_tn6_training.py --device colab_cpu --batch-size 8

# Chạy thử nghiệm nhanh cục bộ trên 1 mô hình
python scripts/run_tn6_training.py --device cpu_laptop --model resnet50 --batch-size 4
```

### 3.2. Kịch bản TN7 — Pipeline Breakdown (CH6)
Đo phân rã quy trình ứng dụng thực tế (Tiền xử lý $\rightarrow$ Model Inference $\rightarrow$ Hậu xử lý NMS/Tokenize):

```bash
# Chạy YOLOv8n, BERT-base, XGBoost trên Colab T4
python scripts/run_tn7_pipeline.py --device colab_t4 --batch-size 1

# Chạy trên Colab CPU để phân tích sự chuyển dịch điểm nghẽn
python scripts/run_tn7_pipeline.py --device colab_cpu --batch-size 1
```

### 3.3. Tái tạo bộ biểu đồ báo cáo khoa học
```bash
python experiments/tv3_figures.py
# Các file ảnh PNG sẽ được lưu tự động vào docs/tv3/figures/
```

---

## 4. Hệ Thống 19 Luật Chẩn Đoán Heuristic v2.0 (`src/analyzer/`)

Động cơ `DiagnosticEngine` tự động đánh giá các tệp kết quả (`summary.json`, `phases.csv`, `layers.csv`, `resources.csv`) tại thư mục thực nghiệm:

```python
from pathlib import Path
from src.analyzer.engine import DiagnosticEngine

engine = DiagnosticEngine()
results = engine.evaluate_directory(Path("results/colab_t4/resnet50/TN6/bs8_fp32"))

for r in results:
    if r.status.value in ["FAIL", "WARN"]:
        print(f"[{r.status.value}] {r.rule_id} - {r.rule_name}: {r.message}")
        print(f"  Khuyến nghị: {r.recommendation}")
```

### Bảng tra cứu 19 luật Heuristic:

| Mã luật | Tên luật | Mức độ cảnh báo | Điều kiện kích hoạt & Ý nghĩa |
| :---: | :--- | :---: | :--- |
| **RA-01** | Peak VRAM Limit | CRITICAL / WARN | Đỉnh VRAM vượt $>90\%$ (hoặc $>75\%$) dung lượng card đồ họa. Nguy cơ OOM. |
| **RA-02** | Low GPU Utilization | WARN | Độ chiếm dụng GPU trung bình $< 30\%$ trong lúc đo. |
| **RA-03** | Layer Activation Spikes | WARN | Tồn tại layer có tensor kích hoạt đột biến chiếm $>40\%$ tổng activation. |
| **RA-04** | Process RAM RSS Leak | WARN | RAM hệ thống tăng liên tục qua các iteration (tách biệt khỏi bộ nhớ tensor). |
| **RA-05** | Memory Bandwidth Bound | INFO | Tỷ lệ nạp/xả bộ nhớ cao trong khi compute utilization thấp. |
| **RB-01** | High Cold-Start Latency | INFO | Vòng forward đầu tiên chậm hơn $> 3\times$ so với trung vị P50. |
| **RB-02** | High Tail Latency (Jitter) | WARN | Tỷ số trễ đuôi $P99 / P50 > 2.0$. Hệ thống không ổn định cho SLA production. |
| **RB-03** | Heavy Backward Ratio | WARN | Thời gian Backward $> 2.5\times$ Forward (bất thường so với chuẩn $2.0\times$). |
| **RB-04** | Inefficient Compute Density | WARN | **(Đã sửa v2.0)**: Thời gian trên mỗi GFLOP vượt ngưỡng bất thường. |
| **RB-05** | High Pipeline Overhead | WARN | Thời gian ngoài mô hình (Pre/Post) chiếm $> 30\%$ toàn bộ pipeline. |
| **RC-01** | DataLoader I/O Bottleneck | WARN | Thời gian nạp dữ liệu chiếm $> 20\%$ tổng chu kỳ bước lặp. |
| **RC-02** | High I/O Duration Variance | INFO | Độ phân tán thời gian nạp batch cao, cảnh báo đọc đĩa không đều. |
| **RC-03** | GPU Starvation Detected | CRITICAL | **(Đã sửa v2.0)**: I/O chiếm $>35\%$ và GPU Util $<30\%$, GPU bị bỏ đói dữ liệu. |
| **RC-04** | Inefficient Worker Count | INFO | Khuyến nghị điều chỉnh `num_workers` khi I/O nghẽn trên CPU đa nhân. |
| **RD-01** | Single Layer Dominance | WARN | Có 1 layer duy nhất chiếm $> 30\%$ tổng thời gian mô hình. |
| **RD-02** | Sub-microsecond Layer | INFO | **(Đã sửa v2.0)**: Layer thực thi $< 1\mu s$, cảnh báo chi phí gọi hàm/event. |
| **RD-03** | Unattributed Overhead | INFO | Thời gian các phép toán dạng hàm (`torch.cat`, cộng residual) $> 20\%$. |
| **RD-04** | High Memory-Bound Ratio | WARN | Số lượng layer giới hạn bởi bộ nhớ (Arithmetic Intensity thấp) $> 50\%$. |
| **RD-05** | Flops-Normalized Discrepancy | INFO | **(Đã sửa v2.0)**: Sai lệch mật độ tính toán $ms / \text{GFLOP}$ giữa các khối. |

---

## 5. Động Cơ So Sánh A/B Testing (`ABComparisonEngine`)
Tính toán tốc độ tăng tốc và kiểm định ý nghĩa thống kê:

```python
from src.analyzer.ab_testing import ABComparisonEngine

baseline_ms = [29.5, 30.1, 29.8, 30.2, 29.7]   # Ví dụ FP32
candidate_ms = [14.8, 15.0, 14.9, 15.1, 14.7]  # Ví dụ FP16

result = ABComparisonEngine.compare(baseline_ms, candidate_ms)

print(f"Speedup: {result['speedup']:.2f}x")
print(f"p-value (Mann-Whitney U): {result['p_value']:.4e}")
print(f"Cliff's Delta: {result['cliffs_delta']:.3f} ({result['effect_size']})")
```

---

## 6. Khởi Chạy Dashboard Streamlit
Khởi chạy giao diện trực quan hóa toàn bộ dự án tại local:

```bash
# Khởi chạy bằng Streamlit CLI
streamlit run src/dashboard/app.py

# Hoặc qua python module:
python -m streamlit run src/dashboard/app.py
```
- Truy cập trình duyệt: `http://localhost:8501`
- Điều hướng các Tab TV3:
  - **Tab 5**: Phân rã giai đoạn Training TN6 (CH5)
  - **Tab 6**: Phân rã quy trình Pipeline TN7 (CH6)
  - **Tab 7**: Chẩn đoán tự động Diagnostic Engine (19 Rules)
  - **Tab 8**: So sánh đối đầu A/B Testing & Speedup
