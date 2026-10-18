"""CLI runner executing benchmark experiments according to configuration arguments."""

import argparse
import contextlib
import gc
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.paths import build_result_path
from src.layer.plugin import LayerProfilerPlugin
from src.model.benchmark import ModelBenchmark
from src.model.env import save_env_metadata
from src.model.metrics import calculate_cosine_similarity
from src.model.resources import ResourceSampler
from src.model.tabular_bench import benchmark_xgboost_latency
from src.phase.plugin import PhaseProfilerPlugin
from src.zoo import get_model_adapter

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("runner")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="VisionProf Benchmark Experiment Runner")
    parser.add_argument("--model", type=str, default="resnet50", help="Model name (e.g. resnet50, bert_base, xgboost)")
    parser.add_argument("--device", type=str, default="colab_cpu", help="Target device (colab_cpu, colab_t4, cpu_laptop)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16"], help="Precision")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length for NLP models")
    parser.add_argument("--experiment", type=str, default="TN3", help="Experiment identifier (TN1..TN7)")
    parser.add_argument("--profilers", type=str, default="layer", help="Comma-separated profilers (layer,phase)")
    parser.add_argument("--warmup", type=int, default=15, help="Number of warmup iterations")
    parser.add_argument("--measure", type=int, default=50, help="Number of measurement iterations")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if summary.json already exists")
    parser.add_argument("--results-dir", type=str, default="results", help="Base directory for results")
    return parser.parse_args()


def main() -> int:
    """Main execution orchestrator running clean benchmark and profiler passes."""
    args = parse_args()
    results_base = REPO_ROOT / args.results_dir

    # 1. Load Model Adapter and canonicalize model name
    adapter = get_model_adapter(args.model)
    canonical_model = adapter.name

    run_dir = build_result_path(
        base_dir=results_base,
        device=args.device,
        model=canonical_model,
        experiment=args.experiment,
        batch_size=args.batch_size,
        precision=args.precision,
        seq_len=args.seq_len if adapter.family == "nlp" else None,
    )

    summary_file = run_dir / "summary.json"
    if args.skip_existing and summary_file.exists():
        logger.info("Found existing summary at %s; skipping execution (--skip-existing).", summary_file)
        return 0

    logger.info("Starting experiment: Model=%s (%s), Device=%s, Batch=%d, Precision=%s",
                args.model, canonical_model, args.device, args.batch_size, args.precision)

    # 2. Save Environment Metadata and Run Config
    save_env_metadata(run_dir / "env.json", target_device=args.device)
    config_dict = vars(args)
    config_dict["canonical_model"] = canonical_model
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    model = adapter.load(device=args.device, precision=args.precision)

    # 3. Phase 1: Clean Benchmark Latency & Resource Monitoring (No Hooks Attached)
    sampler = ResourceSampler(interval_s=0.1)
    sampler.start()

    bench = ModelBenchmark(warmup_iters=args.warmup, measure_iters=args.measure)
    try:
        bench.run_warmup(
            adapter,
            model,
            batch_size=args.batch_size,
            device=args.device,
            precision=args.precision,
            seq_len=args.seq_len,
        )
        bench.measure(
            adapter,
            model,
            batch_size=args.batch_size,
            device=args.device,
            precision=args.precision,
            plugins=None,  # Pure clean measurement without hook overhead
            seq_len=args.seq_len,
        )
    finally:
        resource_samples = sampler.stop()
        sampler.save(run_dir / "resources.csv")

    summary = bench.summarize(batch_size=args.batch_size)

    # 4. Specialized Metrics: XGBoost Tabular Benchmark & FP16 Cosine Similarity
    if args.model == "xgboost":
        logger.info("Running specialized XGBoost tabular benchmark...")
        x_sample = adapter.make_input(batch_size=100, device=args.device)
        xgb_extra = benchmark_xgboost_latency(model, x_sample)
        summary.extra.update(xgb_extra)

    if args.precision == "fp16" and adapter.is_torch:
        try:
            logger.info("Evaluating FP32 vs FP16 numerical cosine similarity...")
            fp32_model = adapter.load(device=args.device, precision="fp32")
            inp_fp32 = adapter.make_input(batch_size=args.batch_size, device=args.device, precision="fp32", seq_len=args.seq_len)
            inp_fp16 = adapter.make_input(batch_size=args.batch_size, device=args.device, precision="fp16", seq_len=args.seq_len)
            out_fp32 = adapter.forward(fp32_model, inp_fp32)
            out_fp16 = adapter.forward(model, inp_fp16)
            cos_sim = calculate_cosine_similarity(out_fp32, out_fp16)
            summary.extra["cosine_similarity"] = cos_sim
        except Exception as exc:
            logger.warning("Could not calculate cosine similarity: %s", exc)

    bench.save(run_dir, summary)

    # 5. Phase 2: Profiler Plugins (Layer and Phase Breakdown)
    profiler_names = [p.strip().lower() for p in args.profilers.split(",") if p.strip()]

    infer_ctx = torch.inference_mode() if HAS_TORCH else contextlib.nullcontext()

    if "layer" in profiler_names and adapter.is_torch:
        logger.info("Running Phase 2: LayerProfilerPlugin collection...")
        layer_plugin = LayerProfilerPlugin()
        layer_plugin.attach(model, device=args.device)
        test_inp = adapter.make_input(batch_size=args.batch_size, device=args.device, precision=args.precision, seq_len=args.seq_len)
        with infer_ctx:
            for i in range(min(5, args.measure)):
                layer_plugin.start(iteration=i)
                _ = adapter.forward(model, test_inp)
                if "cuda" in args.device.lower() and HAS_TORCH and torch.cuda.is_available():
                    torch.cuda.synchronize()
                layer_plugin.stop()
        layer_plugin.save(run_dir)
        layer_plugin.detach()

    if "phase" in profiler_names and adapter.is_torch:
        logger.info("Running Phase 2: PhaseProfilerPlugin collection...")
        phase_plugin = PhaseProfilerPlugin()
        phase_plugin.attach(model, device=args.device)
        test_inp = adapter.make_input(batch_size=args.batch_size, device=args.device, precision=args.precision, seq_len=args.seq_len)
        with infer_ctx:
            for i in range(min(5, args.measure)):
                phase_plugin.start(iteration=i)
                _ = adapter.forward(model, test_inp)
                phase_plugin.stop()
        phase_plugin.save(run_dir)
        phase_plugin.detach()

    gc.collect()

    logger.info("Experiment successfully completed. Results written to %s", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
