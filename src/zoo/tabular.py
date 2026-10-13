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
            import os
            import sys
            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
            device_target = "cuda" if "cuda" in device.lower() or "t4" in device.lower() else "cpu"
            if device_target == "cpu" or sys.platform == "darwin":
                os.environ.setdefault("OMP_NUM_THREADS", "1")
            import xgboost as xgb
            tree_method = "hist"
            kwargs: dict[str, Any] = {
                "n_estimators": 50,
                "max_depth": 6,
                "tree_method": tree_method,
                "device": device_target,
            }
            if device_target == "cpu" or sys.platform == "darwin":
                kwargs["n_jobs"] = 1
            model = xgb.XGBClassifier(**kwargs)
            # Fit on 1,000 synthetic samples to initialize trees to realistic depth 6
            dummy_x = np.random.randn(1000, 54).astype(np.float32)
            dummy_y = np.arange(1000) % 7
            model.fit(dummy_x, dummy_y)
            return model
        except Exception as exc:
            logger.warning("XGBoost not available or GPU mode unsupported: %s. Using Mock.", exc)
            mock = _MockXGBoostModel()
            dummy_x = np.random.randn(1000, 54).astype(np.float32)
            dummy_y = np.arange(1000) % 7
            mock.fit(dummy_x, dummy_y)
            return mock

    def make_input(self, batch_size: int, device: str = "cpu", precision: str = "fp32", **kwargs: Any) -> Any:
        return np.random.randn(batch_size, 54).astype(np.float32)

    def forward(self, model: Any, inputs: Any) -> Any:
        return model.predict(inputs)

    def fit(self, model: Any, x: Any, y: Any) -> Any:
        """Fit XGBoost model on tabular data and labels."""
        model.fit(x, y)
        return model

    def train_step(self, model: Any, batch: Any, optimizer: Any = None, timer: Any = None) -> float:
        """Execute one training fit step for tabular XGBoost model."""
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            x, y = batch
        else:
            x = batch
            num_classes = len(getattr(model, "classes_", []))
            if num_classes > 1 and len(x) < num_classes:
                repeat_count = int(np.ceil(num_classes / max(1, len(x))))
                x = np.repeat(x, repeat_count, axis=0)[:num_classes]
                y = np.arange(num_classes)
            elif num_classes > 1:
                y = np.arange(len(x)) % num_classes
            else:
                y = np.zeros(len(x), dtype=int)
        if timer is not None:
            with timer.measure("forward"):
                model.fit(x, y)
        else:
            model.fit(x, y)
        return 0.0


class _MockXGBoostModel:
    """Mock fallback for environments without xgboost installed."""

    def fit(self, x: Any, y: Any) -> "_MockXGBoostModel":
        return self

    def predict(self, x: Any) -> Any:
        return np.zeros(len(x), dtype=int)
