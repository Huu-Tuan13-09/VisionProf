# 01. Quy Chuẩn Code & Tổ Chức File (Coding Standards & File Organization)

Tài liệu này quy định các tiêu chuẩn bắt buộc về chất lượng mã nguồn, phong cách lập trình và tổ chức file trong toàn bộ dự án **VisionProf**.

---

## 1. Giới Hạn Kích Thước File & Đơn Trách Nhiệm (SRP)

Codebase v1.0 từng gặp vấn đề với các "monolithic god files" (`analyzer.py` ~1800 dòng, `collector.py` ~900 dòng, `dashboard.py` ~900 dòng). Để đảm bảo tính mô-đun hóa cao, dễ bảo trì và mở rộng, toàn bộ code mới phải tuân theo các quy định cứng:

### 1.1. Giới Hạn Độ Dài (Line Length Limits)
- **Tối đa lý tưởng**: $\le$ **250 dòng** code/file (tính cả docstrings và imports).
- **Hard Limit**: **300 dòng** code/file.
- **Kích hoạt Refactor**: Khi một file chạm mốc 250 dòng, phải chủ động phân tách các class phụ hoặc logic trợ giúp sang file/module riêng. Tuyệt đối không để file vượt quá 300 dòng.

### 1.2. Nguyên Tắc Mỗi File Một Mục Đích (Single Purpose)
- Mỗi file module chỉ giải quyết một bài toán nghiệp vụ đơn nhất:
  - `flops.py`: Chỉ tính FLOPs (không kèm logic benchmark hay timer).
  - `phases.py`: Chỉ quản lý context manager đo phase forward/backward/pre/post.
  - `resources.py`: Chỉ quản lý thread đo thông số phần cứng nền (CPU, RAM, GPU Util).
- Tên file rõ nghĩa, dùng `snake_case` danh từ: `flop_counter.py`, `layer_profiler.py`, `resource_sampler.py`.

---

## 2. Nguyên Tắc Thiết Kế Mã Nguồn & Lean Code

1. **Thiết kế hướng Interface**: Mọi module đo lường và adapter mở rộng bằng cách kế thừa Abstract Base Class (`ModelAdapter`, `ProfilerPlugin`), không sửa đổi mã nguồn lõi hiện có.
2. **Lean Code & YAGNI**:
   - Không trừu tượng hóa khi chỉ có 1 trường hợp sử dụng. Áp dụng quy tắc "lặp lại 3 lần mới trừu tượng hóa".
   - **Ưu tiên Standard Library & Native Features**:
     - Dùng `dataclasses` thay vì các framework ORM nặng nề.
     - Dùng `pathlib.Path` thay vì nối chuỗi `os.path.join`.
     - Dùng `torch.utils.flop_counter.FlopCounterMode` của PyTorch 2.x thay vì kéo thêm dependency ngoài.
3. **Loại bỏ Dead Code**: Không comment code cũ để lại "phòng hờ". Xóa sạch mã thừa.

---

## 3. Tiêu Chuẩn Viết Code Python (Python 3.10+)

### 3.1. Type Annotations (Bắt Buộc 100%)
Mọi hàm, method và thuộc tính public đều phải có type hint đầy đủ:

```python
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class BenchmarkResult:
    model_name: str
    latency_p50_ms: float
    latency_p95_ms: float
    peak_vram_mb: float
    throughput_fps: float

def execute_benchmark(
    adapter: "ModelAdapter", 
    iterations: int = 50,
    warmup: int = 20
) -> BenchmarkResult:
    ...
```

### 3.2. Không Dùng Magic Numbers & Magic Strings
Khai báo hằng số (`UPPER_SNAKE_CASE`) hoặc `Enum`:

```python
from enum import Enum

class HardwareEnvironment(str, Enum):
    CPU_LAPTOP = "cpu_laptop"
    COLAB_T4 = "colab_t4"

DEFAULT_WARMUP_ITERATIONS = 20
DEFAULT_MEASURE_ITERATIONS = 50
GPU_MEMORY_DANGER_THRESHOLD = 0.85
```

### 3.3. Quy Chuẩn Docstring & Chú Thích
Sử dụng Google Style Docstring cho mọi public class / function:

```python
def measure_layer_latency(
    layer: torch.nn.Module, 
    inputs: torch.Tensor
) -> float:
    """Đo thời gian thực thi của một leaf layer bằng Deferred CUDA Events.

    Args:
        layer: PyTorch leaf module cần đo lường.
        inputs: Input tensor cho layer.

    Returns:
        Thời gian trễ tính bằng mili-giây (ms).

    Raises:
        RuntimeError: Nếu phát hiện thiết bị GPU bị ngắt kết nối hoặc OOM.
    """
    ...
```

### 3.4. Logging & Error Handling
- **CẤM**: Dùng lệnh `print()` để debug trong các module đo lường và phân tích (`src/`).
- **BẮT BUỘC**: Sử dụng thư viện `logging`:
```python
import logging

logger = logging.getLogger(__name__)
logger.info(f"Đã hoàn thành warm-up {warmup_iters} iterations.")
```
- **Xử lý ngoại lệ**: Không viết bare `except:` hoặc `except Exception: pass`. Bắt đúng loại exception cụ thể (`torch.cuda.OutOfMemoryError`, `FileNotFoundError`) kèm ghi log nguyên nhân rõ ràng.
