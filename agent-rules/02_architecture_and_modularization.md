# 02. Kiến Trúc Hệ Thống & Ranh Giới Module (Architecture & Modularization)

Tài liệu này xác định ranh giới trách nhiệm giữa các module, hợp đồng giao tiếp (Interface Contracts) và cấu trúc dữ liệu đầu ra chuẩn của **VisionProf**.

> [!IMPORTANT]
> **Cấu trúc thư mục chuẩn hóa (Flat Modular Architecture)**:  
> Cấu trúc cây thư mục vật lý đã được **chốt chính thức** theo mô hình Flat Modular tại [`PROJECT_PLAN.md`](../PROJECT_PLAN.md) (Mục 8.2).  
> Bộ quy tắc tập trung vào các **nguyên lý cốt lõi bất biến**: Ranh giới trách nhiệm giữa 3 mức đo (TV1: Layer, TV2: Model, TV3: Phase/Stage), Interface Contracts (`ModelAdapter`, `ProfilerPlugin`) và Data Schema (`results/`).

---

## 1. Phân Tách Trách Nhiệm 3 Mức Đo (3-Tier Separation of Concerns)

Hệ thống được tổ chức thành 3 mức đo lường độc lập, giao tiếp lỏng qua Public Interface:

| Mức đo lường | Module phụ trách | Trách nhiệm kỹ thuật cốt lõi | Dữ liệu đầu ra |
| :--- | :--- | :--- | :--- |
| 🟧 **Mức Layer** | `src/layer/`<br>(events, timer, memory, flops, overhead, hw_specs, plugin) | Zero-Config hook tự động vào leaf modules; Deferred CUDA Events; CPU timer nano-giây; tính FLOPs qua `FlopCounterMode`; % peak Roofline; calibrate hook overhead. | `layers.csv` |
| 🟦 **Mức Model** | `src/model/`<br>`src/zoo/`<br>`configs/`<br>`scripts/run_experiment.py` | Đo đạc toàn model (P50/P90/P95/P99, FPS, cold-start, peak VRAM); background thread lấy mẫu 100ms CPU/RAM/VRAM/Power; quản lý Model Zoo 8 models (kèm XGBoost bypass). | `summary.json`<br>`benchmark.csv`<br>`resources.csv`<br>`env.json` |
| 🟩 **Mức Giai đoạn** | `src/phase/`<br>`src/analyzer/`<br>`src/dashboard/` | Đo thời gian theo pha (preprocess, forward, backward, optimizer, postprocess); giám sát I/O DataLoader (loại trừ cold-start batch 0); bộ 19 luật chẩn đoán v2.0; so sánh A/B. | `phases.csv` |

**Nguyên tắc ranh giới bất biến**:
- Các module không can thiệp xuyên chéo logic nội bộ của nhau.
- Mọi tương tác liên module **bắt buộc** đi qua Public Interface (`ModelAdapter`, `ProfilerPlugin`) và Data Schema (`results/`).

---

## 2. Hợp Đồng Giao Tiếp Bất Biến (Interface Contracts)

### 2.1. Model Interface (`ModelAdapter`)
Nằm tại `src/zoo/base.py`, chuẩn hóa cách vận hành cho cả 8 models (CV, NLP, ML):

```python
from abc import ABC, abstractmethod
from typing import Any

class ModelAdapter(ABC):
    """Abstract class chuẩn hóa cho toàn bộ mô hình (CV, NLP, ML)."""
    name: str          # "resnet50", "bert_base", "xgboost", ...
    family: str        # "cv" | "nlp" | "ml"
    is_torch: bool     # False với XGBoost -> bỏ qua hook mức layer

    @abstractmethod
    def load(self, device: str, precision: str = "fp32") -> Any: ...

    @abstractmethod
    def make_input(self, batch_size: int, device: str = "cpu", **kwargs) -> Any: ...

    @abstractmethod
    def forward(self, model: Any, inputs: Any) -> Any: ...

    def train_step(self, model: Any, batch: Any, optimizer: Any) -> float:
        """Đo một bước training (forward + loss + backward + optimizer)."""
        raise NotImplementedError

    def preprocess(self, raw_data: Any) -> Any: ...   # Tiền xử lý dữ liệu thô
    def postprocess(self, model_output: Any) -> Any: ... # Hậu xử lý (NMS, tokenize)
```

### 2.2. Plugin Đo Lường (`ProfilerPlugin`)
Chuẩn hóa cơ chế cắm ghép công cụ đo lường độc lập vào pipeline:

```python
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

class ProfilerPlugin(ABC):
    """Interface plugin chuẩn hóa cho các bộ đo (layer, phase,...)."""
    name: str                       # "layer" | "phase"

    @abstractmethod
    def attach(self, model: Any, device: str) -> None: ...

    @abstractmethod
    def start(self, iteration: int) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def save(self, out_dir: Path) -> None: ...  # Ghi layers.csv hoặc phases.csv
```

---

## 3. Cấu Trúc Kết Quả Đầu Ra (Results Schema)

Đường dẫn lưu trữ: `results/{máy}/{model}/{TN}/{cấu_hình}/` (sinh bởi `build_result_path()`)
- Thiết bị (`máy`): `colab_cpu` | `colab_t4` (chuẩn hoá Google Colab, kèm fallback `cpu_laptop`)
- Đơn vị chuẩn: Thời gian = `ms`, Bộ nhớ = `MB`, Khối lượng tính toán = `GFLOPs`, Tốc độ = `FPS` / `samples/s`.

| File | Mục đích & Nội dung chính |
| :--- | :--- |
| `env.json` | Phần cứng và runtime: CPU name, GPU name, RAM GB, VRAM GB, OS, versions |
| `config.json` | Cấu hình chạy: model, batch_size, precision, warmup_iters, measure_iters, device |
| `benchmark.csv` | Dữ liệu latency chi tiết: `iteration,latency_ms` |
| `summary.json` | Thống kê phân phối trễ: `mean_ms`, `p50_ms`, `p90_ms`, `p95_ms`, `p99_ms`, `tail_ratio`, `throughput_fps`, `peak_vram_mb`, `rss_mb` |
| `resources.csv` | Lấy mẫu tài nguyên (mỗi 100ms): `timestamp,cpu_pct,rss_mb,gpu_util_pct,vram_mb,power_w` |
| `layers.csv` | Bóc tách layer: `iteration,layer_name,layer_type,time_ms,activation_mb,flops,params` |
| `phases.csv` | Bóc tách phase: `iteration,phase,time_ms` (preprocess, forward, backward, optimizer, postprocess) |

---

## 4. Nguyên Tắc Tích Hợp Hệ Thống (Integration Principles)

1. **Ghép nối lỏng (Loose Coupling)**: Các profiler plugins bật/tắt độc lập thông qua cấu hình `--profilers layer,phase` mà không làm thay đổi luồng thực thi của runner.
2. **Giao tiếp qua file**: Analyzer và Dashboard đọc dữ liệu đầu vào trực tiếp từ schema `results/`, tách biệt hoàn toàn pha thu thập và pha trực quan hóa.
3. **Tương thích ngược**: Khi chia nhỏ code cũ, giữ re-export tại `src/` để đảm bảo test suite không bị gãy trong quá trình refactor.
