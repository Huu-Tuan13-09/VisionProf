"""CLI runner executing benchmark experiments according to configuration arguments."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.interfaces import ProfilerPlugin
from src.core.paths import build_result_path
from src.layer.plugin import LayerProfilerPlugin
from src.model.benchmark import ModelBenchmark
from src.model.env import save_env_metadata
from src.model.resources import ResourceSampler
from src.phase.plugin import PhaseProfilerPlugin
from src.zoo import get_model_adapter

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
    parser.add_argument("--experiment", type=str, default="tn3", help="Experiment identifier (tn1..tn7)")
    parser.add_argument("--profilers", type=str, default="layer,phase", help="Comma-separated profilers (layer,phase)")
    parser.add_argument("--warmup", type=int, default=15, help="Number of warmup iterations")
    parser.add_argument("--measure", type=int, default=50, help="Number of measurement iterations")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if summary.json already exists")
    parser.add_argument("--results-dir", type=str, default="results", help="Base directory for results")
    return parser.parse_args()


def main() -> int:
    """Main execution orchestrator."""
    args = parse_args()
    results_base = REPO_ROOT / args.results_dir

    run_dir = build_result_path(
        base_dir=results_base,
        device=args.device,
        model=args.model,
        experiment=args.experiment,
        batch_size=args.batch_size,
        precision=args.precision,
    )

    summary_file = run_dir / "summary.json"
    if args.skip_existing and summary_file.exists():
        logger.info("Found existing summary at %s; skipping execution (--skip-existing).", summary_file)
        return 0

    logger.info("Starting experiment: Model=%s, Device=%s, Batch=%d, Precision=%s",
                args.model, args.device, args.batch_size, args.precision)

    # 1. Save Environment Metadata and Run Config
    save_env_metadata(run_dir / "env.json", target_device=args.device)
    config_dict = vars(args)
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    # 2. Load Model Adapter
    adapter = get_model_adapter(args.model)
    model = adapter.load(device=args.device, precision=args.precision)

    # 3. Setup Profiler Plugins
    plugins: List[ProfilerPlugin] = []
    profiler_names = [p.strip().lower() for p in args.profilers.split(",") if p.strip()]

    if "layer" in profiler_names:
        layer_plugin = LayerProfilerPlugin()
        layer_plugin.attach(model, device=args.device)
        plugins.append(layer_plugin)

    if "phase" in profiler_names:
        phase_plugin = PhaseProfilerPlugin()
        phase_plugin.attach(model, device=args.device)
        plugins.append(phase_plugin)

    # 4. Background Resource Sampler
    sampler = ResourceSampler(interval_s=0.1)
    sampler.start()

    # 5. Benchmark Execution
    bench = ModelBenchmark(warmup_iters=args.warmup, measure_iters=args.measure)
    try:
        bench.run_warmup(
            adapter,
            model,
            batch_size=args.batch_size,
            device=args.device,
            precision=args.precision,
        )
        bench.measure(
            adapter,
            model,
            batch_size=args.batch_size,
            device=args.device,
            precision=args.precision,
            plugins=plugins,
        )
    finally:
        # 6. Teardown and Save
        resource_samples = sampler.stop()
        sampler.save(run_dir / "resources.csv")
        for p in plugins:
            p.save(run_dir)
            p.detach()

    # Compute Summary
    peak_vram = max((r.vram_mb for r in resource_samples), default=0.0)
    rss_mb = max((r.rss_mb for r in resource_samples), default=0.0)
    summary = bench.summarize(
        batch_size=args.batch_size,
        peak_vram_mb=peak_vram,
        rss_mb=rss_mb,
    )
    bench.save(run_dir, summary)
    logger.info("Experiment successfully completed. Results written to %s", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
