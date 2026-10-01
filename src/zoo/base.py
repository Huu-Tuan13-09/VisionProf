"""Base adapter classes for PyTorch and non-PyTorch models."""

from abc import abstractmethod
from typing import Any, Dict
from src.core.interfaces import ModelAdapter


class BaseTorchAdapter(ModelAdapter):
    """Base class for all PyTorch neural network adapters."""

    is_torch: bool = True

    def forward(self, model: Any, inputs: Any) -> Any:
        """Execute default forward pass through nn.Module."""
        if isinstance(inputs, dict):
            return model(**inputs)
        if isinstance(inputs, (list, tuple)):
            return model(*inputs)
        return model(inputs)


class BaseNonTorchAdapter(ModelAdapter):
    """Base class for non-PyTorch models (e.g. XGBoost, scikit-learn)."""

    is_torch: bool = False

    def train_step(self, model: Any, batch: Any, optimizer: Any) -> float:
        """Non-PyTorch models do not support standard PyTorch train_step."""
        raise NotImplementedError("Standard PyTorch train_step is not supported for non-torch models.")
