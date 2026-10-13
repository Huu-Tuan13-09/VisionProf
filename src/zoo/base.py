"""Base adapter classes for PyTorch and non-PyTorch models."""

from abc import abstractmethod
from typing import Any, Dict, Optional
from src.core.interfaces import ModelAdapter

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


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

    def train_step(self, model: Any, batch: Any, optimizer: Any, timer: Optional[Any] = None) -> float:
        """Execute standard PyTorch train step: forward, loss, backward, optimizer."""
        model.train()
        optimizer.zero_grad()

        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            inputs, targets = batch
        else:
            inputs = batch
            targets = None

        if timer is not None:
            with timer.measure("forward"):
                outputs = self.forward(model, inputs)
                loss = self._compute_step_loss(outputs, targets)
            with timer.measure("backward"):
                loss.backward()
            with timer.measure("optimizer"):
                optimizer.step()
        else:
            outputs = self.forward(model, inputs)
            loss = self._compute_step_loss(outputs, targets)
            loss.backward()
            optimizer.step()

        return float(loss.item())

    def _compute_step_loss(self, outputs: Any, targets: Optional[Any]) -> Any:
        """Compute scalar loss for training step."""
        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state.sum()
        if isinstance(outputs, (tuple, list)):
            return outputs[0].sum()
        if isinstance(outputs, dict):
            first_val = next(iter(outputs.values()))
            return first_val.sum()
        if targets is not None and HAS_TORCH and isinstance(outputs, torch.Tensor) and isinstance(targets, torch.Tensor):
            criterion = nn.CrossEntropyLoss() if outputs.ndim > 1 and targets.ndim == 1 else nn.MSELoss()
            return criterion(outputs, targets)
        if hasattr(outputs, "sum"):
            return outputs.sum()
        return outputs


class BaseNonTorchAdapter(ModelAdapter):
    """Base class for non-PyTorch models (e.g. XGBoost, scikit-learn)."""

    is_torch: bool = False

    def train_step(self, model: Any, batch: Any, optimizer: Any = None, timer: Any = None) -> float:
        """Non-PyTorch models do not support standard PyTorch train_step."""
        raise NotImplementedError("Standard PyTorch train_step is not supported for non-torch models.")
