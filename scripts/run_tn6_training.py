"""TN6 – Training-phase breakdown (CH5): nạp dữ liệu/forward/backward/optimizer.

Runs the PhaseProfilerPlugin across 5 training models (ResNet-50, MobileNetV3,
DistilBERT, MLP, XGBoost) on both cpu_laptop and colab_t4/colab_cpu.

Output per run (one row = one iteration)::

    results/{device}/{model}/TN6/bs{batch}_{precision}/phases.csv

Usage::

    python scripts/run_tn6_training.py --device cpu_laptop
    python scripts/run_tn6_training.py --device colab_t4
    python scripts/run_tn6_training.py --device colab_cpu --model resnet50 --workers 0
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.paths import build_result_path
from src.model.env import save_env_metadata
from src.phase.plugin import PhaseProfilerPlugin
from src.zoo import get_model_adapter

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tn6")

# ── Constants ────────────────────────────────────────────────────────────────
TN6_MODELS: List[str] = [
    "resnet50",
    "mobilenet_v3_large",
    "distilbert",
    "mlp",
    "xgboost",
]
WARMUP_ITERS: int = 5
MEASURE_ITERS: int = 30


# ── Helpers ──────────────────────────────────────────────────────────────────

def _resolve_device(device: str) -> str:
    """Map canonical device name to PyTorch device string."""
    return "cuda" if ("cuda" in device.lower() or "t4" in device.lower()) else "cpu"


def _sync_if_cuda(device: str) -> None:
    """Synchronise CUDA stream when running on GPU."""
    if HAS_TORCH and ("cuda" in device or "t4" in device):
        if torch.cuda.is_available():
            torch.cuda.synchronize()


# ── XGBoost training step ────────────────────────────────────────────────────

def _run_xgboost_training(
    adapter: Any,
    model: Any,
    run_dir: Path,
    device: str,
    batch_size: int,
) -> None:
    """Measure XGBoost fit time using PhaseTimer and write phases.csv."""
    plugin = PhaseProfilerPlugin()
    plugin.attach(model, device=device)

    if not HAS_NUMPY:
        logger.warning("numpy not available; skipping XGBoost training measurement.")
        return

    x = np.random.randn(max(batch_size, 100), 54).astype(np.float32)
    y = np.arange(len(x)) % 7

    # Warmup
    for _ in range(2):
        adapter.train_step(model, (x, y), optimizer=None)

    # Measure
    for i in range(MEASURE_ITERS):
        plugin.start(i)
        t0 = time.perf_counter()
        adapter.train_step(model, (x, y), optimizer=None)
        elapsed = (time.perf_counter() - t0) * 1000.0
        plugin.record_phase("fit", elapsed)
        plugin.stop()

    plugin.save(run_dir)
    logger.info("XGBoost TN6 saved to %s", run_dir)


# ── PyTorch training loop ────────────────────────────────────────────────────

def _run_torch_training(
    adapter: Any,
    model: Any,
    run_dir: Path,
    device: str,
    batch_size: int,
    precision: str,
    num_workers: int,
    seq_len: int,
) -> None:
    """Run training loop with PhaseProfilerPlugin for PyTorch models."""
    if not HAS_TORCH:
        logger.error("PyTorch not available; cannot run PyTorch training.")
        return

    torch_device = _resolve_device(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    plugin = PhaseProfilerPlugin()
    plugin.attach(model, device=device)

    # Build a simple synthetic DataLoader
    inp_sample = adapter.make_input(
        batch_size=batch_size, device=device, precision=precision, seq_len=seq_len
    )
    # Create a flat list of (batch, label) tuples to simulate DataLoader
    if isinstance(inp_sample, dict):
        dummy_inputs = [inp_sample] * (WARMUP_ITERS + MEASURE_ITERS)
        dummy_targets = None
    else:
        dummy_targets = torch.zeros(batch_size, dtype=torch.long, device=torch_device)
        dummy_inputs = [inp_sample] * (WARMUP_ITERS + MEASURE_ITERS)

    # Warmup phase (silent, no record)
    model.train()
    for wi in range(WARMUP_ITERS):
        optimizer.zero_grad()
        with torch.enable_grad():
            out = adapter.forward(model, dummy_inputs[wi])
            if hasattr(out, "loss") and out.loss is not None:
                loss = out.loss
            elif hasattr(out, "last_hidden_state"):
                loss = out.last_hidden_state.sum()
            elif isinstance(out, (tuple, list)):
                loss = out[0].sum()
            else:
                loss = out.sum() if hasattr(out, "sum") else torch.tensor(0.0)
        loss.backward()
        optimizer.step()
    _sync_if_cuda(device)

    # Measurement phase: record forward / backward / optimizer sub-steps
    for i in range(MEASURE_ITERS):
        batch = dummy_inputs[WARMUP_ITERS + i]
        plugin.start(i)
        optimizer.zero_grad()

        # ── forward ───────────────────────────────────────────────────────────
        _sync_if_cuda(device)
        t0 = time.perf_counter()
        with torch.enable_grad():
            out = adapter.forward(model, batch)
            if hasattr(out, "loss") and out.loss is not None:
                loss = out.loss
            elif hasattr(out, "last_hidden_state"):
                loss = out.last_hidden_state.sum()
            elif isinstance(out, (tuple, list)):
                loss = out[0].sum()
            else:
                loss = out.sum() if hasattr(out, "sum") else torch.tensor(0.0)
        _sync_if_cuda(device)
        fwd_ms = (time.perf_counter() - t0) * 1000.0
        plugin.record_phase("forward", fwd_ms)

        # ── backward ──────────────────────────────────────────────────────────
        _sync_if_cuda(device)
        t1 = time.perf_counter()
        loss.backward()
        _sync_if_cuda(device)
        bwd_ms = (time.perf_counter() - t1) * 1000.0
        plugin.record_phase("backward", bwd_ms)

        # ── optimizer ─────────────────────────────────────────────────────────
        _sync_if_cuda(device)
        t2 = time.perf_counter()
        optimizer.step()
        _sync_if_cuda(device)
        opt_ms = (time.perf_counter() - t2) * 1000.0
        plugin.record_phase("optimizer", opt_ms)

        plugin.stop()

    plugin.save(run_dir)
    logger.info("TN6 PyTorch training phases saved to %s", run_dir)


# ── Memory snapshot ──────────────────────────────────────────────────────────

def _measure_memory_ratio(
    adapter: Any,
    device: str,
    batch_size: int,
    precision: str,
    seq_len: int,
) -> dict:
    """Measure peak VRAM for inference vs training (GPU only)."""
    torch_device = _resolve_device(device)
    if not HAS_TORCH or torch_device != "cuda" or not torch.cuda.is_available():
        return {}

    model_infer = adapter.load(device=device, precision=precision)
    inputs = adapter.make_input(
        batch_size=batch_size, device=device, precision=precision, seq_len=seq_len
    )
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        adapter.forward(model_infer, inputs)
    torch.cuda.synchronize()
    infer_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    model_train = adapter.load(device=device, precision=precision)
    model_train.train()
    optimizer = torch.optim.AdamW(model_train.parameters(), lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    with torch.enable_grad():
        out = adapter.forward(model_train, inputs)
        loss = out.sum() if hasattr(out, "sum") else torch.tensor(0.0)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    train_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    del model_infer, model_train
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "infer_peak_vram_mb": round(infer_mb, 2),
        "train_peak_vram_mb": round(train_mb, 2),
        "memory_ratio": round(train_mb / infer_mb, 3) if infer_mb > 0 else None,
    }


# ── Per-model runner ─────────────────────────────────────────────────────────

def run_model(
    model_name: str,
    device: str,
    batch_size: int,
    precision: str,
    num_workers: int,
    seq_len: int,
    results_base: Path,
    skip_existing: bool,
) -> None:
    """Run TN6 training-phase profiling for a single model."""
    adapter = get_model_adapter(model_name)
    run_dir = build_result_path(
        base_dir=results_base,
        device=device,
        model=adapter.name,
        experiment="TN6",
        batch_size=batch_size,
        precision=precision,
    )

    phases_csv = run_dir / "phases.csv"
    if skip_existing and phases_csv.exists():
        logger.info("Skipping %s on %s — phases.csv already exists.", model_name, device)
        return

    save_env_metadata(run_dir / "env.json", target_device=device)
    config = {
        "model": model_name, "device": device, "batch_size": batch_size,
        "precision": precision, "num_workers": num_workers,
        "warmup": WARMUP_ITERS, "measure": MEASURE_ITERS, "experiment": "TN6",
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    logger.info("=== TN6: %s on %s (bs=%d, prec=%s) ===", model_name, device, batch_size, precision)
    model = adapter.load(device=device, precision=precision)

    if model_name == "xgboost":
        _run_xgboost_training(adapter, model, run_dir, device, batch_size)
    else:
        _run_torch_training(
            adapter, model, run_dir, device, batch_size,
            precision, num_workers, seq_len,
        )

    # Optional VRAM memory ratio snapshot
    memory_info = _measure_memory_ratio(adapter, device, batch_size, precision, seq_len)
    if memory_info:
        with open(run_dir / "memory_ratio.json", "w", encoding="utf-8") as fh:
            json.dump(memory_info, fh, indent=2)
        logger.info("Memory ratio (train/infer): %.3fx", memory_info.get("memory_ratio", 0))

    del model
    gc.collect()
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for TN6."""
    parser = argparse.ArgumentParser(
        description="TN6: Training-phase breakdown (CH5) for VisionProf"
    )
    parser.add_argument("--device", type=str, default="cpu_laptop",
                        choices=["cpu_laptop", "colab_cpu", "colab_t4"],
                        help="Target hardware device")
    parser.add_argument("--model", type=str, default="all",
                        help="Model name or 'all' to run all 5 TN6 models")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for training steps")
    parser.add_argument("--precision", type=str, default="fp32",
                        choices=["fp32", "fp16"], help="Numerical precision")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Sequence length for NLP models")
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader num_workers (0=no multiprocessing)")
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERS,
                        help="Warmup iterations (default 5)")
    parser.add_argument("--measure", type=int, default=MEASURE_ITERS,
                        help="Measurement iterations (default 30)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip model if phases.csv already exists")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Base output directory")
    return parser.parse_args()


def main() -> int:
    """Entry point: run TN6 for one or all configured models."""
    args = parse_args()
    results_base = REPO_ROOT / args.results_dir

    # Override module-level constants when user passes CLI flags
    global WARMUP_ITERS, MEASURE_ITERS  # noqa: PLW0603
    WARMUP_ITERS = args.warmup
    MEASURE_ITERS = args.measure

    models_to_run = TN6_MODELS if args.model == "all" else [args.model]

    for model_name in models_to_run:
        try:
            run_model(
                model_name=model_name,
                device=args.device,
                batch_size=args.batch_size,
                precision=args.precision,
                num_workers=args.workers,
                seq_len=args.seq_len,
                results_base=results_base,
                skip_existing=args.skip_existing,
            )
        except Exception as exc:
            logger.error("TN6 failed for model '%s': %s", model_name, exc, exc_info=True)

    logger.info("TN6 complete. Results in: %s", results_base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
