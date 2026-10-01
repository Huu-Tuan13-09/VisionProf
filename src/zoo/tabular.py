"""Tabular model adapters: Multi-Layer Perceptron (MLP) and XGBoost (Covertype)."""

import logging
from typing import Any, Tuple
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from src.zoo.base import BaseNonTorchAdapter, BaseTorchAdapter

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class MLPAdapter(BaseTorchAdapter):
    """Adapter for PyTorch Tabular Multi-Layer Perceptron (Covertype 54 features)."""

    name: str = "mlp"
    family: str = "tabular"

    def load(self, device: str, precision: str = "fp32") -> Any:
        dev = "cuda" if "cuda" in device.lower() or "t4" in device.lower() else "cpu"
        model = nn.Sequential(
            nn.Linear(54, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 7),
        ).to(dev)
        if precision == "fp16" and dev == "cuda":
            model = model.half()
        model.eval()
        return model

    def make_input(self, batch_size: int, device: str = "cpu", precision: str = "fp32", **kwargs: Any) -> Any:
        dev = "cuda" if "cuda" in device.lower() or "t4" in device.lower() else "cpu"
        dtype = torch.float16 if precision == "fp16" and dev == "cuda" else torch.float32
        return torch.randn(batch_size, 54, device=dev, dtype=dtype)


class XGBoostAdapter(BaseNonTorchAdapter):
    """Adapter for XGBoost Gradient Boosted Trees (non-PyTorch model)."""

    name: str = "xgboost"
    family: str = "tabular"
    is_torch: bool = False

    def load(self, device: str, precision: str = "fp32") -> Any:
        try:
            import xgboost as xgb
            tree_method = "hist"
            device_target = "cuda" if "cuda" in device.lower() or "t4" in device.lower() else "cpu"
            model = xgb.XGBClassifier(
                n_estimators=50,
                max_depth=6,
                tree_method=tree_method,
                device=device_target,
            )
            # Fit on a tiny dummy matrix to initialize trees
            dummy_x = np.random.randn(20, 54).astype(np.float32)
            dummy_y = np.random.randint(0, 7, size=(20,))
            model.fit(dummy_x, dummy_y)
            return model
        except Exception as exc:
            logger.warning("XGBoost not available or GPU mode unsupported: %s. Using Mock.", exc)
            return _MockXGBoostModel()

    def make_input(self, batch_size: int, device: str = "cpu", precision: str = "fp32", **kwargs: Any) -> Any:
        return np.random.randn(batch_size, 54).astype(np.float32)

    def forward(self, model: Any, inputs: Any) -> Any:
        return model.predict(inputs)


class _MockXGBoostModel:
    """Mock fallback for environments without xgboost installed."""
    def predict(self, x: Any) -> Any:
        return np.zeros(len(x), dtype=int)
