"""Standardized file path and directory generation utilities for VisionProf."""

from pathlib import Path
from typing import Dict, Union


def sanitize_name(name: str) -> str:
    """Normalize names of devices, models, and experiments into safe directory tokens."""
    return name.strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")


def build_result_path(
    base_dir: Union[str, Path],
    device: str,
    model: str,
    experiment: str,
    batch_size: int,
    precision: str = "fp32",
) -> Path:
    """Generate standardized results path: results/{device}/{model}/{experiment}/bs{batch_size}_{precision}/"""
    dev = sanitize_name(device)
    mdl = sanitize_name(model)
    exp = sanitize_name(experiment)
    prec = sanitize_name(precision)
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
        "phases": dir_path / "phases.csv",
    }
