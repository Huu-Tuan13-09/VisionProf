"""Model Zoo registry and factory methods for VisionProf."""

from typing import Dict, List, Type

from src.core.interfaces import ModelAdapter
from src.zoo.cv import MobileNetV3Adapter, ResNet50Adapter, ViTB16Adapter, YOLOv8nAdapter
from src.zoo.nlp import BERTBaseAdapter, DistilBERTAdapter
from src.zoo.tabular import MLPAdapter, XGBoostAdapter

_REGISTRY: Dict[str, Type[ModelAdapter]] = {
    "mobilenet_v3": MobileNetV3Adapter,
    "resnet50": ResNet50Adapter,
    "vit_b_16": ViTB16Adapter,
    "yolov8n": YOLOv8nAdapter,
    "distilbert": DistilBERTAdapter,
    "bert_base": BERTBaseAdapter,
    "mlp": MLPAdapter,
    "xgboost": XGBoostAdapter,
}


def get_model_adapter(name: str) -> ModelAdapter:
    """Instantiate a model adapter by registered model identifier name."""
    clean_name = name.strip().lower().replace("-", "_")
    if clean_name not in _REGISTRY:
        raise KeyError(f"Unknown model '{clean_name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[clean_name]()


def list_available_models() -> List[str]:
    """Return list of all registered model identifiers."""
    return list(_REGISTRY.keys())


__all__ = [
    "get_model_adapter",
    "list_available_models",
    "MobileNetV3Adapter",
    "ResNet50Adapter",
    "ViTB16Adapter",
    "YOLOv8nAdapter",
    "DistilBERTAdapter",
    "BERTBaseAdapter",
    "MLPAdapter",
    "XGBoostAdapter",
]
