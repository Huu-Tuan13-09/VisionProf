"""Activation memory calculation utility based on output tensor shapes."""

from typing import Any, Generator, List, Optional, Set, Tuple

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _iter_tensors(obj: Any) -> Generator[Any, None, None]:
    """Recursively yield all torch.Tensor instances contained in inputs or output."""
    if not HAS_TORCH:
        return
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            yield from _iter_tensors(item)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_tensors(v)


def _storage_key(t: Any) -> Optional[int]:
    """Retrieve raw memory storage data pointer of a tensor to recognize views/inplace ops."""
    try:
        return t.untyped_storage().data_ptr()
    except Exception:
        return None


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

    @classmethod
    def calculate_output_meta(cls, inputs: Any, output: Any) -> Tuple[List[int], float]:
        """Return (primary_output_shape, new_activation_mb) ignoring inplace ops and tensor views."""
        if not HAS_TORCH:
            return [], 0.0

        input_keys: Set[int] = {_storage_key(t) for t in _iter_tensors(inputs) if _storage_key(t) is not None}
        out_shape: List[int] = []
        seen_keys: Set[int] = set()
        total_bytes = 0

        for t in _iter_tensors(output):
            if not out_shape and hasattr(t, "shape"):
                out_shape = list(t.shape)
            key = _storage_key(t)
            if key is not None and (key in input_keys or key in seen_keys):
                continue
            if key is not None:
                seen_keys.add(key)
            total_bytes += t.numel() * t.element_size()

        act_mb = round(total_bytes / (1024.0 * 1024.0), 4)
        return out_shape, act_mb
