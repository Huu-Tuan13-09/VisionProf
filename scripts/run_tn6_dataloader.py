"""TN6 (phần nạp dữ liệu) – CH5: nạp dữ liệu có làm GPU phải chờ không?

Training CV models with a real ``torch.utils.data.DataLoader``: every sample is a
JPEG decoded and augmented on the CPU (RandomResizedCrop + flip + normalize, the
usual ImageNet recipe). The run is repeated for several ``num_workers`` values and
each training step is split into data_load / forward / backward / optimizer.

data_load = waiting for the next batch from the DataLoader (measured through
DataLoaderSentinel) + copying it to the device.

Output per run (one row = one phase of one iteration)::

    results/{device}/{model}/TN6/bs{batch}_{precision}_w{workers}/dataloader_phases.csv
    results/{device}/{model}/TN6/bs{batch}_{precision}_w{workers}/dataloader_summary.json

The file is not named ``phases.csv`` so the TN6 dashboard tab keeps reading only the
forward / backward / optimizer runs of ``run_tn6_training.py``.

Usage::

    python scripts/run_tn6_dataloader.py --device colab_t4
    python scripts/run_tn6_dataloader.py --device colab_t4 --model resnet50 --workers 0 2 4
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.core.paths import sanitize_name
from src.model.env import save_env_metadata
from src.phase.dataloader import DataLoaderSentinel
from src.zoo import get_model_adapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("tn6_dataloader")

CV_MODELS: List[str] = ["resnet50", "mobilenet_v3_large"]
N_DISTINCT_IMAGES = 256      # distinct JPEGs kept in memory; samples cycle over them
SOURCE_SIZE = 320            # JPEG side before RandomResizedCrop to 224


class SyntheticJpegDataset(Dataset):
    """Random-content JPEGs decoded and augmented on the CPU for every sample."""

    def __init__(self, n_samples: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.n_samples = n_samples
        self.jpegs: List[bytes] = []
        for _ in range(N_DISTINCT_IMAGES):
            pixels = rng.integers(0, 256, (SOURCE_SIZE, SOURCE_SIZE, 3), dtype=np.uint8)
            buf = io.BytesIO()
            Image.fromarray(pixels).save(buf, format="JPEG", quality=90)
            self.jpegs.append(buf.getvalue())
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Any:
        image = Image.open(io.BytesIO(self.jpegs[idx % N_DISTINCT_IMAGES])).convert("RGB")
        return self.transform(image), idx % 1000


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def run_once(model_name: str, device_id: str, batch_size: int, workers: int,
             warmup: int, measure: int, results_base: Path) -> Dict[str, float]:
    """Train ``warmup + measure`` steps with a real DataLoader and save the phase breakdown."""
    device = "cuda" if "t4" in device_id or "cuda" in device_id else "cpu"
    run_dir = (results_base / sanitize_name(device_id) / sanitize_name(model_name) / "TN6"
               / f"bs{batch_size}_fp32_w{workers}")
    run_dir.mkdir(parents=True, exist_ok=True)
    save_env_metadata(run_dir / "env.json", target_device=device_id)

    adapter = get_model_adapter(model_name)
    model = adapter.load(device=device, precision="fp32")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    dataset = SyntheticJpegDataset(n_samples=(warmup + measure + 1) * batch_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=(device == "cuda"), drop_last=True)
    sentinel = DataLoaderSentinel(loader)
    batches = iter(sentinel)

    rows: List[Dict[str, Any]] = []
    for step in range(warmup + measure):
        _sync(device)
        t0 = time.perf_counter()
        images, labels = next(batches)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        _sync(device)
        t1 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        out = adapter.forward(model, images)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        loss = criterion(logits, labels)
        _sync(device)
        t2 = time.perf_counter()
        loss.backward()
        _sync(device)
        t3 = time.perf_counter()
        optimizer.step()
        _sync(device)
        t4 = time.perf_counter()
        if step >= warmup:
            it = step - warmup
            for phase, a, b in (("data_load", t0, t1), ("forward", t1, t2),
                                ("backward", t2, t3), ("optimizer", t3, t4)):
                rows.append({"iteration": it, "phase": phase, "time_ms": round((b - a) * 1000.0, 4)})

    with open(run_dir / "dataloader_phases.csv", "w", encoding="utf-8") as f:
        f.write("iteration,phase,time_ms\n")
        f.writelines(f"{r['iteration']},{r['phase']},{r['time_ms']}\n" for r in rows)

    med = {p: float(np.median([r["time_ms"] for r in rows if r["phase"] == p]))
           for p in ("data_load", "forward", "backward", "optimizer")}
    step_ms = sum(med.values())
    summary = {
        "model": model_name, "device": device_id, "batch_size": batch_size, "num_workers": workers,
        "cpu_count": os.cpu_count(), "warmup": warmup, "measure": measure,
        "median_ms": {k: round(v, 3) for k, v in med.items()},
        "step_ms": round(step_ms, 3),
        "data_load_pct": round(med["data_load"] / step_ms * 100.0, 2),
        "images_per_s": round(batch_size * 1000.0 / step_ms, 1),
        "sentinel_mean_io_ms": round(sentinel.mean_io_ms(), 3),
    }
    with open(run_dir / "dataloader_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("%s w=%d: step %.1f ms, data_load %.1f%%, %.1f img/s",
                model_name, workers, step_ms, summary["data_load_pct"], summary["images_per_s"])

    del model, optimizer, loader, batches
    if device == "cuda":
        torch.cuda.empty_cache()
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description="TN6: nạp dữ liệu với DataLoader thật (CH5)")
    p.add_argument("--device", default="colab_t4", choices=["cpu_laptop", "colab_cpu", "colab_t4"])
    p.add_argument("--model", default="all", help="resnet50, mobilenet_v3_large hoặc all")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, nargs="+", default=[0, 2, 4])
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--measure", type=int, default=30)
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()

    if "t4" in args.device and not torch.cuda.is_available():
        logger.error("Chọn colab_t4 nhưng không có GPU CUDA. Đổi runtime sang T4 GPU.")
        return 1
    models = CV_MODELS if args.model == "all" else [args.model]
    for model_name in models:
        for workers in args.workers:
            run_once(model_name, args.device, args.batch_size, workers,
                     args.warmup, args.measure, REPO_ROOT / args.results_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
