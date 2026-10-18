"""Model evaluation metrics: Cosine similarity for FP32 vs FP16, and Tabular metrics."""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def primary_output_tensor(output: Any) -> Any:
    """Pick the tensor that represents a model's prediction (Tensor, tuple/list, dict or HF ModelOutput)."""
    if not HAS_TORCH:
        return None
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict) or hasattr(output, "keys"):
        for key in ("logits", "last_hidden_state"):
            if key in output and isinstance(output[key], torch.Tensor):
                return output[key]
        output = list(output.values())
    if isinstance(output, (tuple, list)):
        for item in output:
            tensor = primary_output_tensor(item)
            if tensor is not None:
                return tensor
    return None


def calculate_cosine_similarity(tensor_a: Any, tensor_b: Any) -> float:
    """
    Cosine similarity between two model outputs (e.g. FP32 vs FP16 of the same weights and input).

    Accepts Tensors, tuples/lists, dicts and HuggingFace ModelOutput; returns NaN when no
    comparable tensor is found, so a failed comparison is never reported as a perfect match.
    """
    a, b = primary_output_tensor(tensor_a), primary_output_tensor(tensor_b)
    if a is None or b is None or a.shape != b.shape:
        return float("nan")

    flat_a = a.detach().float().flatten()
    flat_b = b.detach().float().flatten()
    norm_a = torch.norm(flat_a)
    norm_b = torch.norm(flat_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return float("nan")

    sim = torch.dot(flat_a, flat_b) / (norm_a * norm_b)
    return round(float(sim.item()), 6)


def calculate_accuracy(y_true: Any, y_pred: Any) -> float:
    """Calculate basic top-1 accuracy for predictions."""
    correct = 0
    total = len(y_true)
    if total == 0:
        return 0.0

    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            correct += 1

    return round(correct / float(total), 4)
