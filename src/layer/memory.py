"""Activation memory calculation utility based on output tensor shapes."""

from typing import Any, Tuple

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class ActivationMemoryTracker:
    """Calculates activation memory footprint in MB directly from output tensors."""

    @staticmethod
    def calculate_tensor_mb(tensor_obj: Any) -> float:
        """Calculate memory consumed by a single PyTorch tensor in megabytes."""
        if not HAS_TORCH or not isinstance(tensor_obj, torch.Tensor):
            return 0.0
        bytes_count = tensor_obj.nelement() * tensor_obj.element_size()
        return round(bytes_count / (1024.0 * 1024.0), 4)

    @classmethod
    def calculate_output_mb(cls, output: Any) -> float:
        """Calculate total activation memory from tensor, tuple, list, or dict of tensors."""
        if not HAS_TORCH:
            return 0.0

        if isinstance(output, torch.Tensor):
            return cls.calculate_tensor_mb(output)

        total_mb = 0.0
        if isinstance(output, (list, tuple)):
            for item in output:
                total_mb += cls.calculate_output_mb(item)
        elif isinstance(output, dict):
            for val in output.values():
                total_mb += cls.calculate_output_mb(val)

        return round(total_mb, 4)
