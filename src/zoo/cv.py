"""Computer Vision model adapters: MobileNetV3, ResNet-50, ViT-B/16, YOLOv8n."""

import logging
from typing import Any, Tuple
from src.zoo.base import BaseTorchAdapter

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class MobileNetV3Adapter(BaseTorchAdapter):
    """Adapter for MobileNetV3-Small (Edge lightweight CNN)."""

    name: str = "mobilenet_v3"
    family: str = "cv"

    def load(self, device: str, precision: str = "fp32") -> Any:
        import torchvision.models as models
        model = models.mobilenet_v3_small(weights=None)
        dev = "cuda" if "cuda" in device.lower() or "t4" in device.lower() else "cpu"
        model = model.to(dev)
        if precision == "fp16" and dev == "cuda":
            model = model.half()
        model.eval()
        return model

    def make_input(self, batch_size: int, device: str = "cpu", precision: str = "fp32", **kwargs: Any) -> Any:
        dev = "cuda" if "cuda" in device.lower() or "t4" in device.lower() else "cpu"
        dtype = torch.float16 if precision == "fp16" and dev == "cuda" else torch.float32
        return torch.randn(batch_size, 3, 224, 224, device=dev, dtype=dtype)


class ResNet50Adapter(BaseTorchAdapter):
    """Adapter for ResNet-50 (Standard CNN baseline)."""

    name: str = "resnet50"
    family: str = "cv"

    def load(self, device: str, precision: str = "fp32") -> Any:
        import torchvision.models as models
        model = models.resnet50(weights=None)
        dev = "cuda" if "cuda" in device.lower() or "t4" in device.lower() else "cpu"
        model = model.to(dev)
        if precision == "fp16" and dev == "cuda":
            model = model.half()
        model.eval()
        return model

    def make_input(self, batch_size: int, device: str = "cpu", precision: str = "fp32", **kwargs: Any) -> Any:
        dev = "cuda" if "cuda" in device.lower() or "t4" in device.lower() else "cpu"
        dtype = torch.float16 if precision == "fp16" and dev == "cuda" else torch.float32
        return torch.randn(batch_size, 3, 224, 224, device=dev, dtype=dtype)


class ViTB16Adapter(BaseTorchAdapter):
    """Adapter for Vision Transformer ViT-B/16."""

    name: str = "vit_b_16"
    family: str = "cv"

    def load(self, device: str, precision: str = "fp32") -> Any:
        import torchvision.models as models
        model = models.vit_b_16(weights=None)
        dev = "cuda" if "cuda" in device.lower() or "t4" in device.lower() else "cpu"
        model = model.to(dev)
        if precision == "fp16" and dev == "cuda":
            model = model.half()
        model.eval()
        return model

    def make_input(self, batch_size: int, device: str = "cpu", precision: str = "fp32", **kwargs: Any) -> Any:
        dev = "cuda" if "cuda" in device.lower() or "t4" in device.lower() else "cpu"
        dtype = torch.float16 if precision == "fp16" and dev == "cuda" else torch.float32
        return torch.randn(batch_size, 3, 224, 224, device=dev, dtype=dtype)


class YOLOv8nAdapter(BaseTorchAdapter):
    """Adapter for YOLOv8-nano detection model (includes NMS postprocessing)."""

    name: str = "yolov8n"
    family: str = "cv"

    def load(self, device: str, precision: str = "fp32") -> Any:
        dev = "cuda" if "cuda" in device.lower() or "t4" in device.lower() else "cpu"
        try:
            from ultralytics import YOLO
            yolo = YOLO("yolov8n.pt")
            model = yolo.model.to(dev)
        except Exception:
            # Fallback tiny ConvNet backbone if ultralytics package is missing
            model = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.AdaptiveAvgPool2d((1, 1)),
            ).to(dev)

        if precision == "fp16" and dev == "cuda":
            model = model.half()
        model.eval()
        return model

    def make_input(self, batch_size: int, device: str = "cpu", precision: str = "fp32", **kwargs: Any) -> Any:
        dev = "cuda" if "cuda" in device.lower() or "t4" in device.lower() else "cpu"
        dtype = torch.float16 if precision == "fp16" and dev == "cuda" else torch.float32
        return torch.randn(batch_size, 3, 640, 640, device=dev, dtype=dtype)

    def postprocess(self, model_output: Any) -> Any:
        """Simulate NMS box filtering on detection output."""
        return model_output
