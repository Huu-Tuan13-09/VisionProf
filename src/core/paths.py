"""Standardized file path and directory generation utilities for VisionProf."""

from pathlib import Path
from typing import Dict, Optional, Union


def sanitize_name(name: str) -> str:
    """Normalize names of devices, models, and experiments into safe directory tokens."""
    token = name.strip().replace("-", "_").replace(" ", "_").replace("/", "_")
    if token.upper().startswith("TN") and token[2:].isdigit():
        return token.upper()
    return token.lower()


def build_result_path(
    base_dir: Union[str, Path],
    device: str,
    model: str,
    experiment: str,
    batch_size: int,
    precision: str = "fp32",
    seq_len: Optional[int] = None,
) -> Path:
    """Generate standardized results path: results/{device}/{model}/{experiment}/bs{batch_size}_{precision}/"""
    dev = sanitize_name(device)
    mdl = sanitize_name(model)
    exp = sanitize_name(experiment)
    prec = sanitize_name(precision)
    if seq_len is not None and seq_len != 128:
        config_dir = f"bs{batch_size}_seq{seq_len}_{prec}"
    else:
        config_dir = f"bs{batch_size}_{prec}"

    target_path = Path(base_dir) / dev / mdl / exp / config_dir
    target_path.mkdir(parents=True, exist_ok=True)
    return target_path


def get_expected_result_files(run_dir: Union[str, Path]) -> Dict[str, Path]:
    """Return dictionary of expected standard artifact paths within a run directory."""
    dir_path = Path(run_dir)
    return {
        "env": dir_path / "env.json",
        "config": dir_path / "config.json",
        "benchmark": dir_path / "benchmark.csv",
        "summary": dir_path / "summary.json",
        "resources": dir_path / "resources.csv",
        "layers": dir_path / "layers.csv",
        "layers_summary": dir_path / "layers_summary.csv",
        "layer_meta": dir_path / "layer_meta.json",
        "phases": dir_path / "phases.csv",
    }
