# Mức Layer (TV1) — đoạn README để ghép vào README tổng

## 🔬 Mức Layer: đo thời gian, bộ nhớ và hiệu suất từng layer

Gắn profiler vào bất kỳ mô hình PyTorch nào mà không sửa mã mô hình:

```python
from src.layer.plugin import LayerProfilerPlugin

profiler = LayerProfilerPlugin()
profiler.attach(model, device="cuda")               # hoặc "cpu"
with torch.no_grad():
    for i in range(50):
        profiler.start(i)
        model(x)
        profiler.stop()                              # chỉ ghi giữa start() và stop()
profiler.detach()
df = profiler.to_dataframe()                         # 1 dòng / layer / lần gọi / vòng đo
```

- **CPU:** đo bằng `time.perf_counter_ns()`. **GPU:** đo bằng CUDA event, chỉ đồng bộ 1 lần mỗi lần
  forward, có "lấp hàng đợi" để đo đúng thời gian GPU tính.
- Đo được module dùng chung (gọi nhiều lần trong 1 forward); đo `nn.MultiheadAttention` nguyên khối.
- Bộ nhớ: `activation_mb` = output của chính layer (không đếm trùng ReLU inplace / view);
  `process_rss_mb` = RAM của cả chương trình, để ở cột riêng.

### FLOPs và % hiệu suất (Roofline)

```python
from src.layer import add_efficiency, analyze_layers, detect_hardware

analysis = analyze_layers(model, x)                  # FLOPs, bytes, arithmetic intensity / layer
eff = add_efficiency(df, analysis, detect_hardware("cuda"), time_col="time_ms")
# → achieved_gflops, efficiency_pct, bound ("compute" / "memory") cho từng layer
```

### Dùng như plugin trong script chạy chung

`scripts/run_experiment.py --profilers layer` đã gọi sẵn plugin theo hợp đồng chung:

```python
plugin = LayerProfilerPlugin()
plugin.attach(model, device)
for i in range(n_iter):
    plugin.start(i); model(x); plugin.stop()
plugin.save(out_dir)        # → layers.csv, layers_summary.csv, layer_meta.json
plugin.detach()
```

Mô hình không phải PyTorch (vd XGBoost) → plugin tự bỏ qua.

### Chạy thực nghiệm TN1, TN2

| Lệnh | Việc |
| :--- | :--- |
| `python experiments/tn1_layer_breakdown.py --device cpu` | TN1 trên laptop: phân rã thời gian theo layer, 7 mô hình |
| `python experiments/tn1_layer_breakdown.py --device cuda` | TN1 trên GPU (Colab T4) |
| `python experiments/tn1_layer_breakdown.py --analyze` | So sánh CPU–GPU → `results/analysis/TN1/<model>/report.md` |
| `python experiments/tn2_validate_profiler.py --device cpu\|cuda` | TN2: sai lệch so với `torch.profiler`, overhead, thời gian bỏ sót |
| `python experiments/tv1_figures.py` | Vẽ hình cho báo cáo → `docs/tv1/figures/` |
| `python experiments/demo_layer_profiler.py --model resnet50` | Demo nhanh: top-10 layer, tỷ trọng, FLOPs |
| `streamlit run src/dashboard/app.py` | Dashboard chung; tab "Layer" và "Roofline" là phần của TV1 |

Tuỳ chọn hay dùng: `--models resnet50 vit_b_16` (chỉ chạy vài mô hình), `--iters 50`, `--warmup 10`.
Đổi thư mục kết quả: đặt biến môi trường `VISIONPROF_RESULTS` (vd thư mục Google Drive khi chạy Colab).

### Chạy trên Google Colab

```python
from google.colab import drive; drive.mount('/content/drive')
import os; os.environ["VISIONPROF_RESULTS"] = "/content/drive/MyDrive/VisionProf/results"
!git clone https://github.com/Huu-Tuan13-09/VisionProf.git
%cd VisionProf
!pip install -q transformers ultralytics
!python experiments/tn1_layer_breakdown.py --device cuda
!python experiments/tn2_validate_profiler.py --device cuda
```

### Kiểm thử

```bash
pytest tests/test_layer_profiler.py tests/test_layer_flops_roofline.py tests/test_layer_scientific_accuracy.py -q   # 29 test (2 test GPU cần CUDA)
```

### Lưu ý

- Trên GPU, layer chỉ chạy vài µs (BatchNorm, ReLU ở batch 1) bị đo dư vì ngưỡng sàn của CUDA event
  (~6 µs trên T4, ~20 µs trên Windows/WDDM). Số liệu GPU nên lấy từ Linux (Colab).
- Bật mức Layer làm mô hình chậm đi (CPU 5–11%, GPU ở batch 1: 109–166%) → **không** dùng nó để đo
  tổng thời gian; việc đó thuộc mức Model.
- Không truyền `attention_mask` cho BERT khi đo tốc độ nếu input không có padding: `transformers` sẽ
  đồng bộ CPU–GPU ngầm và làm sai số đo GPU.
