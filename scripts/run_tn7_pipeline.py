"""TN7 – Real-world pipeline breakdown (CH6): preprocess / model / postprocess.

Profiles the end-to-end pipeline of 3 real-world model families:
  • YOLO   : read tensor → preprocess (resize) → model → postprocess (NMS)
  • BERT   : raw text → tokenize → model → decode
  • XGBoost: load CSV row → predict

Output per run::

    results/{device}/{model}/TN7/bs{batch}_{precision}/phases.csv

Usage::

    python scripts/run_tn7_pipeline.py --device cpu_laptop
    python scripts/run_tn7_pipeline.py --device colab_t4
    python scripts/run_tn7_pipeline.py --device colab_cpu --model yolov8n
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.paths import build_result_path
from src.model.env import save_env_metadata
from src.phase.plugin import PhaseProfilerPlugin
from src.zoo import get_model_adapter

try:
    import torch
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
logger = logging.getLogger("tn7")

# ── Constants ────────────────────────────────────────────────────────────────
TN7_MODELS: List[str] = ["yolov8n", "bert_base", "xgboost"]
WARMUP_ITERS: int = 10
MEASURE_ITERS: int = 50


# ── Helpers ──────────────────────────────────────────────────────────────────

def _resolve_torch_device(device: str) -> str:
    return "cuda" if ("cuda" in device.lower() or "t4" in device.lower()) else "cpu"


def _sync_if_cuda(device: str) -> None:
    if HAS_TORCH and ("cuda" in device or "t4" in device):
        if torch.cuda.is_available():
            torch.cuda.synchronize()


def _timed(fn: Any, *args: Any, device: str = "cpu") -> float:
    """Run callable and return wall-clock duration in ms (GPU-synced)."""
    _sync_if_cuda(device)
    t0 = time.perf_counter()
    fn(*args)
    _sync_if_cuda(device)
    return (time.perf_counter() - t0) * 1000.0


# ── YOLO pipeline ─────────────────────────────────────────────────────────────

def _run_yolo_pipeline(
    device: str,
    batch_size: int,
    precision: str,
    run_dir: Path,
    skip_existing: bool,
) -> None:
    """Measure YOLO full pipeline: preprocess / forward / postprocess."""
    if not HAS_TORCH:
        logger.warning("PyTorch unavailable; skipping YOLO TN7.")
        return

    adapter = get_model_adapter("yolov8n")
    phases_csv = run_dir / "phases.csv"
    if skip_existing and phases_csv.exists():
        logger.info("Skipping yolov8n TN7 — phases.csv exists.")
        return

    model = adapter.load(device=device, precision=precision)
    torch_dev = _resolve_torch_device(device)

    plugin = PhaseProfilerPlugin()
    plugin.attach(model, device=device)

    # Raw synthetic "image" — simulates reading from disk/resize pipeline
    raw_image = torch.randint(0, 256, (batch_size, 3, 720, 1280), dtype=torch.float32)

    # Warmup
    for _ in range(WARMUP_ITERS):
        inp = adapter.preprocess(raw_image)
        with torch.inference_mode():
            out = adapter.forward(model, inp)
        adapter.postprocess(out)
    _sync_if_cuda(device)

    # Measure
    for i in range(MEASURE_ITERS):
        plugin.start(i)

        pre_ms = _timed(adapter.preprocess, raw_image, device=device)
        plugin.record_phase("preprocess", pre_ms)

        inp = adapter.preprocess(raw_image)
        with torch.inference_mode():
            fwd_ms = _timed(adapter.forward, model, inp, device=device)
        plugin.record_phase("model_forward", fwd_ms)

        out = adapter.forward(model, inp)
        post_ms = _timed(adapter.postprocess, out, device=device)
        plugin.record_phase("postprocess_nms", post_ms)

        plugin.stop()

    plugin.save(run_dir)
    logger.info("YOLO TN7 pipeline saved to %s", run_dir)
    del model
    gc.collect()
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── BERT pipeline ─────────────────────────────────────────────────────────────

_SAMPLE_TEXTS: List[str] = [
    "The quick brown fox jumps over the lazy dog.",
    "Profiling AI models reveals where computation time is actually spent.",
    "GPU acceleration benefits depend heavily on model architecture and batch size.",
    "Data preprocessing can be a significant bottleneck in real-world pipelines.",
]


def _run_bert_pipeline(
    device: str,
    batch_size: int,
    precision: str,
    seq_len: int,
    run_dir: Path,
    skip_existing: bool,
) -> None:
    """Measure BERT full pipeline: tokenize / forward / decode."""
    if not HAS_TORCH:
        logger.warning("PyTorch unavailable; skipping BERT TN7.")
        return

    adapter = get_model_adapter("bert_base")
    phases_csv = run_dir / "phases.csv"
    if skip_existing and phases_csv.exists():
        logger.info("Skipping bert_base TN7 — phases.csv exists.")
        return

    model = adapter.load(device=device, precision=precision)

    plugin = PhaseProfilerPlugin()
    plugin.attach(model, device=device)

    # Build a batch of raw text
    raw_batch = (_SAMPLE_TEXTS * ((batch_size // len(_SAMPLE_TEXTS)) + 1))[:batch_size]

    # Warmup
    for _ in range(WARMUP_ITERS):
        inp = adapter.preprocess(raw_batch)
        torch_dev = _resolve_torch_device(device)
        if isinstance(inp, dict):
            inp = {k: v.to(torch_dev) for k, v in inp.items() if HAS_TORCH and hasattr(v, "to")}
        with torch.inference_mode():
            out = adapter.forward(model, inp)
        adapter.postprocess(out)
    _sync_if_cuda(device)

    # Measure
    torch_dev = _resolve_torch_device(device)
    for i in range(MEASURE_ITERS):
        plugin.start(i)

        # Tokenize (preprocess)
        pre_ms = _timed(adapter.preprocess, raw_batch, device=device)
        plugin.record_phase("tokenize", pre_ms)

        inp = adapter.preprocess(raw_batch)
        if isinstance(inp, dict):
            inp = {k: v.to(torch_dev) for k, v in inp.items() if HAS_TORCH and hasattr(v, "to")}

        with torch.inference_mode():
            fwd_ms = _timed(adapter.forward, model, inp, device=device)
        plugin.record_phase("model_forward", fwd_ms)

        out = adapter.forward(model, inp)
        post_ms = _timed(adapter.postprocess, out, device=device)
        plugin.record_phase("decode_pooling", post_ms)

        plugin.stop()

    plugin.save(run_dir)
    logger.info("BERT TN7 pipeline saved to %s", run_dir)
    del model
    gc.collect()
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── XGBoost pipeline ──────────────────────────────────────────────────────────

def _run_xgboost_pipeline(
    device: str,
    batch_size: int,
    run_dir: Path,
    skip_existing: bool,
) -> None:
    """Measure XGBoost pipeline: load CSV → parse features → predict."""
    if not HAS_NUMPY:
        logger.warning("numpy unavailable; skipping XGBoost TN7.")
        return

    adapter = get_model_adapter("xgboost")
    phases_csv = run_dir / "phases.csv"
    if skip_existing and phases_csv.exists():
        logger.info("Skipping xgboost TN7 — phases.csv exists.")
        return

    model = adapter.load(device=device)

    plugin = PhaseProfilerPlugin()
    plugin.attach(model, device=device)

    # Simulate raw "CSV rows" as a structured array (serialise → deserialise)
    raw_rows = np.random.randn(max(batch_size, 1), 54).astype(np.float32)
    raw_str = "\n".join(",".join(f"{v:.6f}" for v in row) for row in raw_rows)

    def _parse_csv(text: str) -> np.ndarray:
        """Simulate CSV parsing into numpy feature matrix."""
        lines = text.strip().split("\n")
        return np.array([[float(v) for v in ln.split(",")] for ln in lines], dtype=np.float32)

    # Warmup
    for _ in range(WARMUP_ITERS):
        feats = _parse_csv(raw_str)
        adapter.forward(model, feats)

    # Measure
    for i in range(MEASURE_ITERS):
        plugin.start(i)

        parse_ms = _timed(_parse_csv, raw_str, device=device)
        plugin.record_phase("csv_parse", parse_ms)

        feats = _parse_csv(raw_str)
        pred_ms = _timed(adapter.forward, model, feats, device=device)
        plugin.record_phase("predict", pred_ms)

        plugin.stop()

    plugin.save(run_dir)
    logger.info("XGBoost TN7 pipeline saved to %s", run_dir)


# ── Per-model dispatcher ──────────────────────────────────────────────────────

def run_model(
    model_name: str,
    device: str,
    batch_size: int,
    precision: str,
    seq_len: int,
    results_base: Path,
    skip_existing: bool,
) -> None:
    """Dispatch to the correct per-model TN7 pipeline runner."""
    adapter = get_model_adapter(model_name)
    run_dir = build_result_path(
        base_dir=results_base,
        device=device,
        model=adapter.name,
        experiment="TN7",
        batch_size=batch_size,
        precision=precision,
    )

    save_env_metadata(run_dir / "env.json", target_device=device)
    config = {
        "model": model_name, "device": device, "batch_size": batch_size,
        "precision": precision, "seq_len": seq_len,
        "warmup": WARMUP_ITERS, "measure": MEASURE_ITERS, "experiment": "TN7",
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    logger.info("=== TN7 pipeline: %s on %s ===", model_name, device)

    if model_name == "yolov8n":
        _run_yolo_pipeline(device, batch_size, precision, run_dir, skip_existing)
    elif model_name in ("bert_base", "bert-base"):
        _run_bert_pipeline(device, batch_size, precision, seq_len, run_dir, skip_existing)
    elif model_name == "xgboost":
        _run_xgboost_pipeline(device, batch_size, run_dir, skip_existing)
    else:
        logger.warning("Model '%s' not in TN7 scope; skipping.", model_name)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for TN7."""
    parser = argparse.ArgumentParser(
        description="TN7: Real-world pipeline breakdown (CH6) for VisionProf"
    )
    parser.add_argument("--device", type=str, default="cpu_laptop",
                        choices=["cpu_laptop", "colab_cpu", "colab_t4"],
                        help="Target hardware device")
    parser.add_argument("--model", type=str, default="all",
                        help="Model name or 'all' (yolov8n, bert_base, xgboost)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (use 1 for real-world single-item scenario)")
    parser.add_argument("--precision", type=str, default="fp32",
                        choices=["fp32", "fp16"], help="Numerical precision")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Sequence length for BERT model")
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERS,
                        help="Warmup iterations (default 10)")
    parser.add_argument("--measure", type=int, default=MEASURE_ITERS,
                        help="Measurement iterations (default 50)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip model if phases.csv already exists")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Base output directory")
    return parser.parse_args()


def main() -> int:
    """Entry point: run TN7 for configured models."""
    args = parse_args()
    results_base = REPO_ROOT / args.results_dir

    global WARMUP_ITERS, MEASURE_ITERS  # noqa: PLW0603
    WARMUP_ITERS = args.warmup
    MEASURE_ITERS = args.measure

    models_to_run = TN7_MODELS if args.model == "all" else [args.model]

    for model_name in models_to_run:
        try:
            run_model(
                model_name=model_name,
                device=args.device,
                batch_size=args.batch_size,
                precision=args.precision,
                seq_len=args.seq_len,
                results_base=results_base,
                skip_existing=args.skip_existing,
            )
        except Exception as exc:
            logger.error("TN7 failed for model '%s': %s", model_name, exc, exc_info=True)

    logger.info("TN7 complete. Results in: %s", results_base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
