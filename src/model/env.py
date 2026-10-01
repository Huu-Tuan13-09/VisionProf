"""System and hardware environment metadata collector."""

import json
import logging
import platform
import sys
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def collect_env_metadata(target_device: str = "colab_cpu") -> Dict[str, Any]:
    """Collect hardware, operating system, and runtime library metadata."""
    data: Dict[str, Any] = {
        "target_device": target_device,
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": sys.version.split()[0],
        },
        "cpu": {
            "processor": platform.processor(),
            "physical_cores": psutil.cpu_count(logical=False) if HAS_PSUTIL else 0,
            "logical_cores": psutil.cpu_count(logical=True) if HAS_PSUTIL else 0,
        },
        "memory": {
            "total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2) if HAS_PSUTIL else 0.0,
        },
        "gpu": {
            "available": False,
            "device_name": "None",
            "vram_total_gb": 0.0,
            "cuda_version": "None",
        },
        "packages": {},
    }

    # GPU Metadata with safe fallback
    if HAS_TORCH and torch.cuda.is_available():
        data["gpu"]["available"] = True
        data["gpu"]["device_name"] = torch.cuda.get_device_name(0)
        data["gpu"]["cuda_version"] = torch.version.cuda or "unknown"
        props = torch.cuda.get_device_properties(0)
        data["gpu"]["vram_total_gb"] = round(props.total_memory / (1024 ** 3), 2)

    # Installed library versions
    for pkg in ["torch", "torchvision", "transformers", "xgboost", "psutil", "numpy", "pandas"]:
        try:
            mod = __import__(pkg)
            data["packages"][pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            data["packages"][pkg] = "not_installed"

    return data


def save_env_metadata(out_path: Path, target_device: str = "colab_cpu") -> Path:
    """Collect and persist env.json to the target destination path."""
    metadata = collect_env_metadata(target_device)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved environment metadata to %s", out_path)
    return out_path
