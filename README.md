# VisionProf ⚡

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
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
│   ├── tv1/               # BAO_CAO_TV1.md, README_TV1.md, figures/
│   ├── tv2/               # BAO_CAO_TV2.md, README_TV2.md, figures/
│   └── tv3/               # BAO_CAO_TV3.md, README_TV3.md, figures/
├── experiments/           # Script thực nghiệm & vẽ hình của từng TV (tn1_*, tn2_*, tv1_*, tv2_*, tv3_*)
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

### Các máy đo (tên thư mục trong `results/`):
- 💻 **`cpu_laptop`**: Laptop của thành viên — số liệu CPU của TN1–TN4 và một phần TN6, TN7.
- ☁️ **`colab_cpu`**: Google Colab CPU (Intel Xeon 2 vCPU, 12GB RAM) — số liệu CPU của TN6, TN7.
- ☁️ **`colab_t4`**: Google Colab GPU NVIDIA Tesla T4 16GB (Turing Tensor Cores, 320 GB/s) — số liệu GPU.

Mỗi lần chạy tự lưu thông tin máy và phiên bản thư viện vào `env.json` cạnh kết quả.

---

## 📊 4. Tổng Hợp 6 Câu Hỏi Nghiên Cứu & Kết Quả Thực Nghiệm

| STT | Thành viên | Câu hỏi nghiên cứu (CH) | Thực nghiệm | Phát hiện cốt lõi đã được kiểm chứng |
| :---: | :---: | :--- | :---: | :--- |
| **1** | **TV1** | **CH1.** Thời gian tốn nhiều nhất ở loại layer nào? CPU vs GPU khác nhau ra sao? | **TN1** | Trên CPU, Conv chiếm 73–85% (CNN), Linear + Attention chiếm 95–96% (Transformer). Lên GPU, Transformer tăng tốc nhiều nhất (~$9\times$) và tỷ trọng gần như không đổi; CNN tăng tốc ít hơn ($3-6\times$) và layer nhỏ chiếm tỷ trọng lớn hơn, rõ nhất ở MobileNetV3 (BatchNorm 14% $\rightarrow$ 25%). |
| **2** | **TV1** | **CH2.** Công cụ đo có chính xác không và làm chậm mô hình bao nhiêu? | **TN2** | Sai lệch tổng so với `torch.profiler` $< 1\%$ trên CPU, $< 8\%$ trên GPU T4. Overhead CPU 5–11%; GPU ở batch 1 là 109–166% $\rightarrow$ mức Layer dùng để phân rã thời gian, không dùng để đo tổng thời gian. |
| **3** | **TV2** | **CH3.** Mô hình ít tham số có chắc chạy nhanh hơn? Thứ hạng có bị đảo khi lên GPU? | **TN3** | Đã đo sạch 100% trên cả CPU Mac và Colab T4 (8 mô hình). Ở batch 1 trên T4, ResNet-50 (8.2 GFLOPs) và MobileNetV3 (0.43 GFLOPs) cùng mất ~6.0ms (nghẽn chi phí gửi kernel). Mô hình nhỏ (MLP) trên GPU chậm hơn CPU (0.28ms vs 0.04ms). MobileNetV3 trễ 176.3ms trên CPU ARM Mac nhưng ~12ms trên x86. |
| **4** | **TV2** | **CH4.** Tăng Batch size và dùng FP16 Tensor Cores thì tốc độ tăng bao nhiêu? | **TN4, TN5** | Đã đo đủ 36 cấu hình batch/seq trên T4 và 7 mô hình FP16. Batch 1 $\rightarrow$ 32 tăng throughput 1.3–8.8× (MobileNetV3 tăng 8.8× từ 170 $\rightarrow$ 1,499 FPS; ViT bão hòa sớm). FP16 giảm VRAM 38–49%, nhưng chỉ tăng tốc Transformer (ViT tăng 2.81×); CNN ở batch 1 bị chậm đi do chi phối bởi kernel launch. |
| **5** | **TV3** | **CH5.** Khi Training, Backward tốn gấp mấy lần Forward? VRAM bùng nổ bao nhiêu? | **TN6** | Tỷ số Backward/Forward đạt **$1.39\times - 2.21\times$** (khớp lý thuyết $2\times$). Đỉnh VRAM khi training **tăng $2.7\times - 3.3\times$** so với suy luận do lưu trữ activations. |
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
- **Tab 7: Diagnostic Engine (TV3)**: Bảng chẩn đoán tự động theo 19 luật heuristic v2.0 (PASS, WARN, FAIL, SKIPPED).
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
# Script chạy chung (TV2): đo mức Model, kèm plugin mức Layer và mức Giai đoạn
python scripts/run_experiment.py --model resnet50 --experiment TN3 --device cpu_laptop --profilers layer,phase

# TN1 (phân rã theo layer) và TN2 (kiểm chứng công cụ đo) của TV1
python experiments/tn1_layer_breakdown.py --device cpu      # hoặc --device cuda trên Colab
python experiments/tn2_validate_profiler.py --device cpu
python experiments/tn1_layer_breakdown.py --analyze         # gộp CPU + GPU → results/analysis/TN1/

# Chạy TN6 (Training Breakdown) trên Colab T4
python scripts/run_tn6_training.py --device colab_t4 --batch-size 8

# Chạy TN7 (Pipeline Breakdown) trên Colab T4
python scripts/run_tn7_pipeline.py --device colab_t4 --batch-size 1

# Sinh bộ biểu đồ phân tích khoa học cho báo cáo
python experiments/tv1_figures.py
python experiments/tv3_figures.py
```

### 6.4. Kiểm thử
```bash
pytest -q                        # toàn bộ unit test (test GPU tự bỏ qua nếu máy không có CUDA)
python scripts/verify_rules.py   # quy chuẩn mã nguồn: ≤ 300 dòng/file, type hints, không print() trong module lõi
```

---

## 💻 7. Hướng Dẫn Lập Trình (API Usage Examples)

### Mức Layer (TV1): Gắn Hook & Roofline
```python
from src.layer import add_efficiency, analyze_layers, detect_hardware
from src.layer.plugin import LayerProfilerPlugin

# Gắn hook vào mọi leaf layer, không sửa mã nguồn model
profiler = LayerProfilerPlugin()
profiler.attach(model, device="cuda")
with torch.no_grad():
    for i in range(50):
        profiler.start(i)          # chỉ ghi giữa start() và stop()
        model(inputs)
        profiler.stop()            # GPU: đồng bộ 1 lần cuối vòng
profiler.save(out_dir)             # → layers.csv, layers_summary.csv, layer_meta.json
profiler.detach()

df_layers = profiler.to_dataframe()
print(df_layers[["layer_name", "time_ms", "activation_mb"]].head())

# FLOPs và % hiệu suất theo Roofline
eff = add_efficiency(df_layers, analyze_layers(model, inputs), detect_hardware("cuda"), time_col="time_ms")
```

### Mức Model (TV2): Benchmark Sạch & Resource Sampler
```python
from src.zoo import get_adapter
from src.model.benchmark import ModelBenchmark

adapter = get_adapter("resnet50")
model = adapter.load("cuda", precision="fp16")
bench = ModelBenchmark(warmup_iters=15, measure_iters=50)

bench.run_warmup(adapter, model, batch_size=8, device="cuda", precision="fp16")
metrics = bench.measure(adapter, model, batch_size=8, device="cuda", precision="fp16")
summary = bench.summarize(batch_size=8)
print(f"P50: {summary.p50_ms:.2f} ms | FPS: {summary.throughput_fps:.1f}")
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
    if r.status.value in ["WARN", "FAIL"]:
        print(f"[{r.status.value}] {r.rule_id} {r.title}: {r.message}")
```

---

## 📚 8. Bản Đồ Tài Liệu Kỹ Thuật (Documentation Map)

Để tìm hiểu chi tiết các kết quả chuyên sâu, vui lòng tra cứu các báo cáo độc lập:
- 🟧 **Tài liệu Thành viên 1 (Mức Layer)**:
  - Báo cáo chi tiết: [`docs/tv1/BAO_CAO_TV1.md`](./docs/tv1/BAO_CAO_TV1.md)
  - Hướng dẫn kỹ thuật: [`docs/tv1/README_TV1.md`](./docs/tv1/README_TV1.md)
  - Biểu đồ minh họa: [`docs/tv1/figures/`](./docs/tv1/figures/)
- 🟦 **Tài liệu Thành viên 2 (Mức Model)**:
  - Báo cáo chi tiết: [`docs/tv2/BAO_CAO_TV2.md`](./docs/tv2/BAO_CAO_TV2.md)
  - Hướng dẫn kỹ thuật: [`docs/tv2/README_TV2.md`](./docs/tv2/README_TV2.md)
  - Biểu đồ minh họa: [`docs/tv2/figures/`](./docs/tv2/figures/)
- 🟩 **Tài liệu Thành viên 3 (Mức Giai đoạn & Tổng kết)**:
  - Báo cáo chi tiết: [`docs/tv3/BAO_CAO_TV3.md`](./docs/tv3/BAO_CAO_TV3.md)
  - Hướng dẫn kỹ thuật: [`docs/tv3/README_TV3.md`](./docs/tv3/README_TV3.md)
  - Biểu đồ minh họa: [`docs/tv3/figures/`](./docs/tv3/figures/)
- 🎤 **Trình bày giữa kỳ**:
  - [`docs/DAN_Y_SLIDE.md`](./docs/DAN_Y_SLIDE.md) — Dàn ý 23 slide, phân vai, câu hỏi phản biện.
  - [`docs/KICH_BAN_TRINH_BAY.md`](./docs/KICH_BAN_TRINH_BAY.md) — Lời thoại từng slide (đã đối chiếu số đo).
  - [`docs/KET_QUA_COLAB_GIUA_KY.md`](./docs/KET_QUA_COLAB_GIUA_KY.md) — Bảng số đo Colab T4 dùng cho slide.
- 📖 **Quy định Kỹ thuật & Kế hoạch Nhóm**:
  - [`PROJECT_PLAN.md`](./PROJECT_PLAN.md) — Kế hoạch tổng thể và ma trận phân công.
  - [`AGENTS.md`](./AGENTS.md) — Bộ 5 nguyên tắc kỹ thuật bắt buộc của dự án.
  - [`docs/HUONG_DAN_CHAY_COLAB.md`](./docs/HUONG_DAN_CHAY_COLAB.md) — Sổ tay vận hành thực nghiệm Google Colab.
