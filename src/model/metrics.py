"""Model evaluation metrics: Cosine similarity for FP32 vs FP16, and Tabular metrics."""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def calculate_cosine_similarity(tensor_a: Any, tensor_b: Any) -> float:
    """Calculate cosine similarity between two PyTorch tensors (e.g., FP32 vs FP16 outputs)."""
    if not HAS_TORCH:
        return 1.0
    if not isinstance(tensor_a, torch.Tensor) or not isinstance(tensor_b, torch.Tensor):
        return 1.0

    flat_a = tensor_a.detach().float().flatten()
    flat_b = tensor_b.detach().float().flatten()

    norm_a = torch.norm(flat_a)
    norm_b = torch.norm(flat_b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0

    dot = torch.dot(flat_a, flat_b)
    sim = dot / (norm_a * norm_b)
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
