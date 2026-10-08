"""
tv1_common.py — Hàm dùng chung cho thực nghiệm của TV1 (TN1, TN2)
=================================================================
- load_model   : nạp model + input. Ưu tiên model zoo của TV2 (src/zoo) khi đã có;
                 tạm thời dựng model từ cấu trúc (không tải trọng số — đo tốc độ
                 không phụ thuộc giá trị trọng số).
- hw_id_for    : tên máy theo quy ước Mục 8.1 ③ (cpu_laptop, colab_t4, …)
- profile_layers: warm-up + đo N vòng bằng LayerPlugin, lưu vào results/
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Tuple

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.flops import run_model  # noqa: E402
from src.hw_specs import detect_hardware  # noqa: E402
from src.layer_plugin import LayerPlugin  # noqa: E402

# 7 model PyTorch của nhóm (XGBoost không có layer nên không nằm ở mức Layer)
TORCH_MODELS = ["mobilenet_v3_large", "resnet50", "vit_b_16", "yolov8n", "distilbert", "bert_base", "mlp"]
# Đổi được bằng biến môi trường, vd khi chạy thử không muốn ghi vào results/ thật
RESULTS_DIR = Path(os.environ.get("VISIONPROF_RESULTS", PROJECT_ROOT / "results"))


class TabularMLP(nn.Module):
    """MLP cho dữ liệu bảng Covertype (54 đặc trưng, 7 lớp)."""

    def __init__(self, n_features: int = 54, n_classes: int = 7, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def _fallback_model(name: str, batch_size: int, seq_len: int) -> Tuple[nn.Module, Any]:
    """Dựng model tạm khi chưa có model zoo của TV2."""
    import torchvision.models as tvm

    if name == "mobilenet_v3_large":
        return tvm.mobilenet_v3_large(weights=None), torch.randn(batch_size, 3, 224, 224)
    if name == "resnet50":
        return tvm.resnet50(weights=None), torch.randn(batch_size, 3, 224, 224)
    if name == "vit_b_16":
        return tvm.vit_b_16(weights=None), torch.randn(batch_size, 3, 224, 224)
    if name == "mlp":
        return TabularMLP(), torch.randn(batch_size, 54)
    if name == "yolov8n":
        from ultralytics import YOLO
        return YOLO("yolov8n.yaml").model, torch.randn(batch_size, 3, 640, 640)
    if name in ("bert_base", "distilbert"):
        from transformers import BertConfig, BertModel, DistilBertConfig, DistilBertModel
        model = BertModel(BertConfig()) if name == "bert_base" else DistilBertModel(DistilBertConfig())
        ids = torch.randint(0, 30522, (batch_size, seq_len))
        return model, {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    raise ValueError(f"Model không hỗ trợ: {name}. Chọn trong {TORCH_MODELS}")


def _to_device(obj: Any, device: str, dtype: torch.dtype) -> Any:
    if isinstance(obj, torch.Tensor):
        obj = obj.to(device)
        return obj.to(dtype) if obj.is_floating_point() else obj
    if isinstance(obj, dict):
        return {k: _to_device(v, device, dtype) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return type(obj)(_to_device(v, device, dtype) for v in obj)
    return obj


def load_model(
    name: str, device: str, batch_size: int = 1, precision: str = "fp32", seq_len: int = 128,
) -> Tuple[nn.Module, Any]:
    """Trả về (model ở chế độ eval trên device, input tương ứng)."""
    try:  # Model zoo của TV2 (PROJECT_PLAN Mục 8.1 ①)
        from src.zoo import get_adapter  # type: ignore
        adapter = get_adapter(name)
        model = adapter.load(device, precision=precision)
        inputs = adapter.make_input(batch_size, seq_len=seq_len)
        return model.eval(), _to_device(inputs, device, next(model.parameters()).dtype)
    except ImportError:
        pass

    model, inputs = _fallback_model(name, batch_size, seq_len)
    dtype = torch.float16 if precision == "fp16" else torch.float32
    model = model.to(device=device, dtype=dtype).eval()
    return model, _to_device(inputs, device, dtype)


def hw_id_for(device: str) -> str:
    """Tên máy dùng làm thư mục kết quả."""
    return "cpu_laptop" if not device.startswith("cuda") else detect_hardware(device).hw_id


def result_dir(device: str, model: str, exp: str, batch_size: int, precision: str) -> Path:
    """results/{máy}/{model}/{TN}/{cấu_hình}/ — đúng quy ước Mục 8.1 ③."""
    return RESULTS_DIR / hw_id_for(device) / model / exp / f"bs{batch_size}_{precision}"


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def profile_layers(
    model: nn.Module, inputs: Any, device: str, out_dir: Path,
    n_warmup: int = 10, n_iter: int = 50,
) -> LayerPlugin:
    """Warm-up rồi đo n_iter vòng bằng LayerPlugin, lưu kết quả vào out_dir."""
    with torch.no_grad():
        for _ in range(n_warmup):
            run_model(model, inputs)
        sync(device)

        plugin = LayerPlugin()
        plugin.attach(model, device)
        for i in range(n_iter):
            plugin.start(i)
            run_model(model, inputs)
            plugin.stop()
    plugin.save(out_dir)
    return plugin
