"""Module for counting floating point operations (FLOPs) of PyTorch layers."""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class FlopsCalculator:
    """Estimates FLOPs for PyTorch leaf modules (Conv2d, Linear, etc.)."""

    @staticmethod
    def estimate_module_flops(module: Any, input_shape: Any, output_shape: Any) -> int:
        """Estimate FLOPs for standard leaf operations when exact counter is unavailable."""
        if not HAS_TORCH or not isinstance(module, nn.Module):
            return 0

        # Conv2d: 2 * Cin * Cout * Kh * Kw * Hout * Wout / groups
        if isinstance(module, nn.Conv2d):
            if len(output_shape) == 4 and len(input_shape) == 4:
                b, c_out, h_out, w_out = output_shape
                k_h, k_w = module.kernel_size
                c_in = module.in_channels
                groups = module.groups
                flops_per_instance = 2 * (c_in // groups) * k_h * k_w * c_out * h_out * w_out
                return int(flops_per_instance)

        # Linear: 2 * In_features * Out_features
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            batch_size = output_shape[0] if len(output_shape) > 0 else 1
            return int(2 * in_features * out_features * batch_size)

        return 0

    @staticmethod
    def count_parameters(module: Any) -> int:
        """Count trainable and non-trainable parameters in a module."""
        if not HAS_TORCH or not isinstance(module, nn.Module):
            return 0
        return sum(p.numel() for p in module.parameters())
