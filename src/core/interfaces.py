"""Abstract base class contracts and plugin interfaces for VisionProf."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional


class ModelAdapter(ABC):
    """Standardized abstract adapter interface for any model (CV, NLP, Tabular)."""

    name: str
    family: str
    is_torch: bool = True

    @abstractmethod
    def load(self, device: str, precision: str = "fp32") -> Any:
        """Load and return the underlying model weights on specified device and precision."""
        ...

    @abstractmethod
    def make_input(self, batch_size: int, device: str = "cpu", precision: str = "fp32", **kwargs: Any) -> Any:
        """Create mock synthetic or real batched inputs for the model."""
        ...

    @abstractmethod
    def forward(self, model: Any, inputs: Any) -> Any:
        """Execute one forward inference pass."""
        ...

    def train_step(self, model: Any, batch: Any, optimizer: Any) -> float:
        """Execute one training step (forward + loss + backward + optimizer step)."""
        raise NotImplementedError("train_step is optional and not implemented for this adapter.")

    def preprocess(self, raw_data: Any) -> Any:
        """Preprocess raw input data into model-compatible tensors."""
        return raw_data

    def postprocess(self, model_output: Any) -> Any:
        """Postprocess raw model predictions (e.g. NMS, argmax, decoding)."""
        return model_output


class ProfilerPlugin(ABC):
    """Abstract plugin interface for measurement probes (layer profiler, phase profiler)."""

    name: str

    @abstractmethod
    def attach(self, model: Any, device: str) -> None:
        """Attach probes, hooks, or interceptors to model or runtime."""
        ...

    @abstractmethod
    def start(self, iteration: int) -> None:
        """Begin recording for a given measurement iteration."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop recording and collect measurements for the active iteration."""
        ...

    @abstractmethod
    def save(self, out_dir: Path) -> None:
        """Persist collected measurement records to destination directory."""
        ...

    def detach(self) -> None:
        """Optional teardown hook to remove probes and restore original state."""
        pass
