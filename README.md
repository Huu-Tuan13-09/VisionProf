# VisionProf ⚡

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Overhead](https://img.shields.io/badge/Profiler_Overhead-%3C5%25-success.svg)
![Colab Ready](https://img.shields.io/badge/Google_Colab-T4_GPU_%26_CPU-F9AB00.svg)

> **Cross-Platform Deep Learning Profiling & Automated Heuristic Diagnostic Framework**  
> *Đồ án môn học Xử lý Dữ liệu Lớn & Hiệu năng cao — Nhóm 3 người (TV1, TV2, TV3)*

---

## 📌 1. Giới Thiệu & Phát Biểu Bài Toán

Khi triển khai các mô hình Trí tuệ Nhân tạo (Computer Vision, NLP, Học máy cổ điển) lên môi trường thực tế, câu hỏi lớn nhất đặt ra là:
> **"Cho một mô hình AI, nó chạy nhanh/chậm thế nào, tốn bao nhiêu tài nguyên trên CPU và GPU, và *thời gian / bộ nhớ bị tiêu tốn ở đâu* (layer nào, giai đoạn nào)?"**

**VisionProf** là framework phân tích hiệu năng đa nền tảng, mổ xẻ toàn diện hiệu năng của mô hình AI theo **kiến trúc 3 mức đo lường độc lập (3-Tier Canonical Modular Architecture)**:
1. 🟧 **Mức Layer (Thành viên 1)**: Mổ xẻ bên trong mô hình PyTorch — gắn hook không sửa mã nguồn, tính FLOPs, bộ nhớ activation, mô hình Roofline, và dùng **Deferred CUDA Events** để đo GPU chuẩn xác.
2. 🟦 **Mức Model (Thành viên 2)**: Đánh giá vĩ mô toàn bộ mô hình trên **Model Zoo 8 mô hình** (CV, NLP, Tabular) — đo phân phối trễ (P50–P99), thông lượng FPS, đỉnh VRAM, luồng nền theo dõi tài nguyên 100ms, và so sánh đối đầu CPU Laptop vs Google Colab Tesla T4.
3. 🟩 **Mức Giai đoạn (Thành viên 3)**: Bóc tách quy trình huấn luyện (Forward, Backward, Optimizer) và pipeline suy luận thực tế (Preprocess, Model, Postprocess/NMS); phát hiện điểm nghẽn tự động bằng **bộ 19 luật heuristic v2.0** và công cụ so sánh A/B có kiểm định thống kê.

---

## 🏛️ 2. Kiến Trúc 3 Mức Đo Lường & Ranh Giới Module

```
+-------------------------------------------------------------------------------+
|                       VISIONPROF PROFILING FRAMEWORK                          |
+-------------------------------------------------------------------------------+
|  🟦 MỨC MODEL (TV2)                                                           |
|  - Model Zoo (8 Adapters: CV, NLP, Tabular/XGBoost)                           |
|  - ModelBenchmark (12 Metrics: P50..P99, FPS, Peak VRAM, Cold-start)          |
|  - ResourceSampler (Luồng nền 100ms: CPU%, RAM RSS, GPU Util%, VRAM, Power)   |
|  - Kịch bản thực nghiệm TN3 (Cơ bản), TN4 (Batch size), TN5 (FP16 Tensor Cores)|
+-------------------------------------------------------------------------------+
         |                                                       |
         v                                                       v
+-----------------------------------+       +-----------------------------------+
|  🟧 MỨC LAYER (TV1)               |       |  🟩 MỨC GIAI ĐOẠN (TV3)           |
|  - Zero-Config Leaf Module Hook   |       |  - PhaseTimer (Context Manager)   |
|  - Deferred CUDA Events (1x sync) |       |  - DataLoaderSentinel (Bỏ batch 0)|
|  - FLOPs qua FlopCounterMode      |       |  - PhaseProfilerPlugin (TN6, TN7) |
|  - % Peak Roofline Model          |       |  - 19 Heuristic Rules Catalog v2.0|
|  - Calibrate Hook Overhead        |       |  - ABComparisonEngine (Mann-Whitney)|
+-----------------------------------+       +-----------------------------------+
         \                                                       /
          \                                                     /
           v                                                   v
+-------------------------------------------------------------------------------+
|  🖥️ STREAMLIT DASHBOARD (8 Tabs Tương Tác Đa Chiều)                          |
|  - Quản lý điều phối app.py (TV3) + Router độc lập src/dashboard/tabs/        |
+-------------------------------------------------------------------------------+
```

### Cấu trúc Thư mục Chuẩn hóa
```
VisionProf/
├── src/
│   ├── core/              # Constants, Paths (build_result_path), Logging
│   ├── layer/             # [TV1] Leaf hooks, Deferred CUDA Events, FLOPs, Roofline
│   ├── model/             # [TV2] ModelBenchmark, ResourceSampler 100ms, Env detector
│   ├── zoo/               # [TV2] 8 Model Adapters kế thừa ModelAdapter base class
│   ├── phase/             # [TV3] PhaseTimer, DataLoaderSentinel, PhaseProfilerPlugin
│   ├── analyzer/          # [TV3] DiagnosticEngine (19 rules), ABComparisonEngine
│   └── dashboard/         # [CẢ NHÓM] app.py router + tabs/ 8 tabs riêng biệt
├── configs/               # Cấu hình YAML cho các thực nghiệm
├── docs/                  # Tài liệu báo cáo hoàn chỉnh của 3 thành viên
│   ├── tv1/               # BAO_CAO_TV1.md, README_TV1.md, SLIDES_TV1.md, figures/
│   ├── tv2/               # BAO_CAO_TV2.md, README_TV2.md, SLIDES_TV2.md, figures/
│   └── tv3/               # BAO_CAO_TV3.md, README_TV3.md, SLIDES_TV3.md, figures/
├── experiments/           # Scripts vẽ biểu đồ khoa học (tv1_figures.py, tv3_figures.py)
├── notebooks/             # colab_runner.ipynb (Tự động hóa chạy trên Google Colab)
├── scripts/               # CLI Runners (run_experiment.py, run_tn6_training.py, ...)
├── tests/                 # Unit tests tự động cho từng thành viên
└── results/               # Kết quả chuẩn hóa {device}/{model}/{exp_id}/{config}/
```

---

## 🎯 3. Danh Mục 8 Mô Hình & 2 Môi Trường Chuẩn Hóa

### 8 Mô Hình Đa Dạng Kiến Trúc:
| Họ | Mô hình | Kiến trúc đặc trưng | Số tham số | Kích thước Input chuẩn |
| :--- | :--- | :--- | :---: | :---: |
| **CV** | **MobileNetV3-Large** | Depthwise Separable Conv, Inverted Residuals | 5.5M | `[B, 3, 224, 224]` |
| | **ResNet-50** | Residual Bottleneck CNN kinh điển | 25.6M | `[B, 3, 224, 224]` |
| | **ViT-B/16** | Vision Transformer (Self-Attention ảnh) | 86.6M | `[B, 3, 224, 224]` |
| | **YOLOv8n** | Phát hiện vật thể nhẹ, có tiền/hậu xử lý | 3.2M | `[B, 3, 640, 640]` |
| **NLP** | **DistilBERT** | 6-layer Transformer Encoder rút gọn | 66.4M | `[B, 128]` tokens |
| | **BERT-base** | 12-layer Bidirectional Transformer Encoder | 109.5M | `[B, 128]` tokens |
| **Tabular** | **MLP** | Mạng nơ-ron truyền thẳng 7 tầng (Covertype) | 0.05M | `[B, 54]` đặc trưng |
| | **XGBoost** | 50 Cây quyết định tăng cường Gradient (GBDT) | — | 54 đặc trưng |

### Chuẩn Hóa Phần Cứng (Loại bỏ sai số máy cá nhân):
- ☁️ **`colab_cpu`**: Intel Xeon CPU @ 2.20GHz (2 vCPU, 12GB RAM) — Chuẩn hóa số liệu CPU.
- ☁️ **`colab_t4`**: NVIDIA Tesla T4 16GB GDDR6 (Turing Tensor Cores, 320 GB/s) — Chuẩn hóa số liệu GPU.
- 💻 **`cpu_laptop`**: Môi trường phát triển cục bộ và host Dashboard.

---

## 📊 4. Tổng Hợp 6 Câu Hỏi Nghiên Cứu & Kết Quả Thực Nghiệm

| STT | Thành viên | Câu hỏi nghiên cứu (CH) | Thực nghiệm | Phát hiện cốt lõi đã được kiểm chứng |
| :---: | :---: | :--- | :---: | :--- |
| **1** | **TV1** | **CH1.** Thời gian tốn nhiều nhất ở loại layer nào? CPU vs GPU khác nhau ra sao? | **TN1** | Trên CPU, Conv/Linear chiếm 73–85%. Lên GPU, các layer nhỏ (Norm, Activation) tăng tỷ trọng rõ rệt (MobileNetV3: Conv 73% $\rightarrow$ 47%, Norm 14% $\rightarrow$ 25%). |
| **2** | **TV1** | **CH2.** Công cụ đo có chính xác không và làm chậm mô hình bao nhiêu? | **TN2** | Sai lệch $< 10\%$ so với `torch.profiler`; overhead gắn hook $< 5\%$ khi đo sạch. |
| **3** | **TV2** | **CH3.** Mô hình ít tham số có chắc chạy nhanh hơn? Thứ hạng có bị đảo khi lên GPU? | **TN3** | **Có đảo thứ hạng!** Trên CPU, MobileNetV3 (5.5M) nhanh gấp 4 lần DistilBERT (66M). Lên GPU T4, DistilBERT tăng tốc $8.8\times$ và vượt mặt MobileNetV3. |
| **4** | **TV2** | **CH4.** Tăng Batch size và dùng FP16 Tensor Cores thì tốc độ tăng bao nhiêu? | **TN4, TN5** | GPU tăng thông lượng $7.5\times$ khi tăng batch từ 1 lên 32; FP16 trên T4 tăng tốc thêm $1.5\times - 2.1\times$ mà cosine similarity vẫn đạt $> 0.999$. |
| **5** | **TV3** | **CH5.** Khi Training, Backward tốn gấp mấy lần Forward? VRAM bùng nổ bao nhiêu? | **TN6** | Tỷ số Backward/Forward đạt **$1.82\times - 2.21\times$** (khớp lý thuyết $2\times$). Đỉnh VRAM khi training **tăng $2.7\times - 3.3\times$** so với suy luận do lưu trữ activations. |
| **6** | **TV3** | **CH6.** Trong ứng dụng thật, chi phí ngoài mô hình (Pre/Post/NMS) chiếm bao nhiêu %? | **TN7** | Với YOLOv8n trên GPU T4, tiền xử lý và NMS chiếm tới **$41.0\%$ ($> 30\%$)**, minh chứng thực nghiệm cho **Định luật Amdahl**. |

---

## 🖥️ 5. Giao Diện Dashboard Streamlit (8 Tabs Tương Tác)

Khởi chạy bằng một lệnh duy nhất:
```bash
streamlit run src/dashboard/app.py
```
Hoặc truy cập qua ngrok tunnel khi chạy trên Google Colab: `http://localhost:8501`

- **Tab 1: Overview 8 Models (TV2)**: So sánh tổng quan độ trễ, FPS, Params và bộ nhớ của 8 mô hình.
- **Tab 2: Colab CPU vs T4 GPU (TV2)**: Biểu đồ cột Speedup đối đầu, phân tích hiện tượng đảo thứ hạng.
- **Tab 3: Leaf Layer Breakdown (TV1)**: Top-10 layer chậm nhất, phân bổ thời gian và kích hoạt theo từng tầng mạng lá.
- **Tab 4: Roofline Model (TV1)**: Bản đồ hiệu suất phần cứng (% Peak GFLOPs), phân định Compute-bound vs Memory-bound.
- **Tab 5: TN6 — Training Phases (TV3)**: Phân rã Forward / Backward / Optimizer, kiểm chứng tỷ số $Bwd/Fwd$ và VRAM ratio.
- **Tab 6: TN7 — Pipeline Breakdown (TV3)**: Donut chart tỷ lệ ngoài mô hình, thanh đo vi phạm ngưỡng $30\%$, nghịch lý CPU vs GPU.
- **Tab 7: Diagnostic Engine (TV3)**: Bảng chẩn đoán tự động theo 19 luật heuristic v2.0 (CRITICAL, WARN, PASS).
- **Tab 8: A/B Testing & Speedup (TV3)**: So sánh đối đầu 2 cấu hình với kiểm định Mann-Whitney U và Cliff's Delta.

---

## 🚀 6. Hướng Dẫn Cài Đặt & Chạy Nhanh (Quick-Start)

### 6.1. Thiết lập Môi trường Cục bộ
```bash
# 1. Clone repository
git clone https://github.com/Huu-Tuan13-09/VisionProf.git
cd VisionProf

# 2. Tạo và kích hoạt virtual environment
python -m venv .venv
.\.venv\Scripts\activate       # Windows PowerShell
# source .venv/bin/activate    # Linux / macOS

# 3. Cài đặt các gói phụ thuộc
pip install -r requirements.txt

# 4. Kiểm tra tính tuân thủ quy chuẩn mã nguồn
python scripts/verify_rules.py
```

### 6.2. Chạy Thực Nghiệm trên Google Colab
Mở notebook [`notebooks/colab_runner.ipynb`](./notebooks/colab_runner.ipynb) trên Google Colab:
1. **Cell 1**: Kiểm tra phần cứng (`nvidia-smi`, CPU specs).
2. **Cell 2**: Cài đặt môi trường từ `requirements.txt`.
3. **Cell 3**: Chạy các thực nghiệm suy luận vĩ mô và mức layer (TN1 $\rightarrow$ TN5).
4. **Cell 4**: Chạy các thực nghiệm mức giai đoạn của TV3 (TN6 Training & TN7 Pipeline).
5. **Cell 5**: Mở Streamlit Dashboard trực tiếp từ Colab qua kết nối an toàn **ngrok tunnel**.
6. **Cell 6**: Đồng bộ toàn bộ dữ liệu `results/` và tài liệu `docs/` lên GitHub repository.

### 6.3. Chạy Kịch Bản Thực Nghiệm bằng Lệnh CLI
```bash
# Chạy TN6 (Training Breakdown) trên Colab T4
python scripts/run_tn6_training.py --device colab_t4 --batch-size 8

# Chạy TN7 (Pipeline Breakdown) trên Colab T4
python scripts/run_tn7_pipeline.py --device colab_t4 --batch-size 1

# Sinh bộ biểu đồ phân tích khoa học cho báo cáo
python experiments/tv1_figures.py
python experiments/tv3_figures.py
```

---

## 💻 7. Hướng Dẫn Lập Trình (API Usage Examples)

### Mức Layer (TV1): Gắn Hook & Roofline
```python
from src.collector import LayerProfiler
from src.flops import analyze_layers

# Khởi tạo profiler không sửa mã nguồn model
profiler = LayerProfiler(model, device="cuda")
with profiler.profile_context(iteration=0):
    output = model(inputs)

df_layers = profiler.to_dataframe()
print(df_layers[["layer_name", "time_ms", "activation_mb"]].head())
```

### Mức Model (TV2): Benchmark Sạch & Resource Sampler
```python
from src.zoo import get_adapter
from src.model.benchmark import ModelBenchmark

adapter = get_adapter("resnet50")
model = adapter.load("cuda", precision="fp16")
bench = ModelBenchmark(warmup_iters=15, measure_iters=50)

bench.run_warmup(adapter, model, batch_size=8, device="cuda")
metrics = bench.measure(adapter, model, batch_size=8, device="cuda")
summary = bench.summarize(batch_size=8)
print(f"P50: {summary['p50_ms']:.2f} ms | FPS: {summary['throughput_fps']:.1f}")
```

### Mức Giai đoạn (TV3): PhaseTimer, Sentinel & Chẩn đoán
```python
from src.phase.timer import PhaseTimer
from src.phase.dataloader import DataLoaderSentinel
from src.analyzer.engine import DiagnosticEngine

# 1. Bọc DataLoader tự động loại trừ cold-start batch 0
sentinel = DataLoaderSentinel(dataloader)

# 2. Đo các pha huấn luyện có đồng bộ GPU
timer = PhaseTimer(device="cuda")
for batch in sentinel:
    with timer.phase("forward"):
        loss = model(batch)
    with timer.phase("backward"):
        loss.backward()
    with timer.phase("optimizer"):
        optimizer.step()

# 3. Chẩn đoán tự động
engine = DiagnosticEngine()
results = engine.evaluate_directory(Path("results/colab_t4/resnet50/TN6/bs8_fp32"))
for r in results:
    if r.status.value in ["WARN", "CRITICAL"]:
        print(f"[{r.status.value}] {r.rule_name}: {r.message}")
```

---

## 📚 8. Bản Đồ Tài Liệu Kỹ Thuật (Documentation Map)

Để tìm hiểu chi tiết các kết quả chuyên sâu, vui lòng tra cứu các báo cáo độc lập:
- 🟧 **Tài liệu Thành viên 1 (Mức Layer)**:
  - Báo cáo chi tiết: [`docs/tv1/BAO_CAO_TV1.md`](./docs/tv1/BAO_CAO_TV1.md)
  - Slide & Kịch bản: [`docs/tv1/SLIDES_TV1.md`](./docs/tv1/SLIDES_TV1.md)
  - Hướng dẫn kỹ thuật: [`docs/tv1/README_TV1.md`](./docs/tv1/README_TV1.md)
  - Biểu đồ minh họa: [`docs/tv1/figures/`](./docs/tv1/figures/)
- 🟦 **Tài liệu Thành viên 2 (Mức Model)**:
  - Báo cáo chi tiết: [`docs/tv2/BAO_CAO_TV2.md`](./docs/tv2/BAO_CAO_TV2.md)
  - Slide & Kịch bản: [`docs/tv2/SLIDES_TV2.md`](./docs/tv2/SLIDES_TV2.md)
  - Hướng dẫn kỹ thuật: [`docs/tv2/README_TV2.md`](./docs/tv2/README_TV2.md)
  - Biểu đồ minh họa: [`docs/tv2/figures/`](./docs/tv2/figures/)
- 🟩 **Tài liệu Thành viên 3 (Mức Giai đoạn & Tổng kết)**:
  - Báo cáo chi tiết: [`docs/tv3/BAO_CAO_TV3.md`](./docs/tv3/BAO_CAO_TV3.md)
  - Slide & Kịch bản: [`docs/tv3/SLIDES_TV3.md`](./docs/tv3/SLIDES_TV3.md)
  - Hướng dẫn kỹ thuật: [`docs/tv3/README_TV3.md`](./docs/tv3/README_TV3.md)
  - Biểu đồ minh họa: [`docs/tv3/figures/`](./docs/tv3/figures/)
- 📖 **Quy định Kỹ thuật & Kế hoạch Nhóm**:
  - [`PROJECT_PLAN.md`](./PROJECT_PLAN.md) — Kế hoạch tổng thể và ma trận phân công.
  - [`AGENTS.md`](./AGENTS.md) — Bộ 5 nguyên tắc kỹ thuật bắt buộc của dự án.
  - [`docs/HUONG_DAN_CHAY_COLAB.md`](./docs/HUONG_DAN_CHAY_COLAB.md) — Sổ tay vận hành thực nghiệm Google Colab.
