import tempfile
from pathlib import Path
import pytest
from src.core.types import BenchmarkMetric, BenchmarkSummary
from src.model.benchmark import ModelBenchmark
from src.model.env import collect_env_metadata, save_env_metadata
from src.model.metrics import calculate_accuracy, calculate_cosine_similarity
from src.model.resources import ResourceSampler
from src.model.tabular_bench import benchmark_xgboost_latency
from src.zoo.tabular import MLPAdapter, XGBoostAdapter


def test_benchmark_metrics_computation():
    """Verify that summarize computes all 12 metrics accurately."""
    bench = ModelBenchmark(warmup_iters=5, measure_iters=10)
    # Mock measurement metrics with predetermined values
    bench._metrics = [
        BenchmarkMetric(iteration=i, latency_ms=float(i + 1), timestamp=float(i))
        for i in range(10)
    ]
    bench._cold_start_ms = 42.5
    bench._peak_vram_mb = 128.0
    bench._rss_mb = 64.0
    bench._params_m = 0.55

    summary = bench.summarize(batch_size=2)
    data = summary.to_dict()

    # Check that all 12 fields are present
    required_fields = [
        "mean_ms",
        "std_ms",
        "p50_ms",
        "p90_ms",
        "p95_ms",
        "p99_ms",
        "tail_ratio",
        "throughput_fps",
        "params_m",
        "peak_vram_mb",
        "rss_mb",
        "cold_start_ms",
    ]
    for field in required_fields:
        assert field in data, f"Field '{field}' missing from BenchmarkSummary dict!"

    assert summary.mean_ms == 5.5
    assert summary.p50_ms == 5.5
    assert summary.cold_start_ms == 42.5
    assert summary.params_m == 0.55
    assert summary.peak_vram_mb == 128.0
    assert summary.rss_mb == 64.0
    assert summary.tail_ratio > 1.0
    # throughput = (2 * 1000) / 5.5 = 363.64
    assert summary.throughput_fps == pytest.approx(363.64, rel=1e-2)


def test_benchmark_empty_metrics():
    """Verify safe fallback when no iterations were measured."""
    bench = ModelBenchmark()
    summary = bench.summarize(batch_size=1)
    assert summary.mean_ms == 0.0
    assert summary.p50_ms == 0.0
    assert summary.throughput_fps == 0.0


def test_benchmark_forwards_kwargs_to_adapter():
    """Verify that run_warmup and measure pass kwargs (e.g. seq_len) to adapter."""
    adapter = MLPAdapter()
    model = adapter.load("cpu")
    bench = ModelBenchmark(warmup_iters=2, measure_iters=3)
    bench.run_warmup(adapter, model, batch_size=2, device="cpu", seq_len=64)
    metrics = bench.measure(adapter, model, batch_size=2, device="cpu", seq_len=64)
    assert len(metrics) == 3


def test_xgboost_latency_benchmarking():
    """Verify XGBoost specialized latency benchmarking."""
    adapter = XGBoostAdapter()
    model = adapter.load("cpu")
    x_sample = adapter.make_input(batch_size=50, device="cpu")
    results = benchmark_xgboost_latency(model, x_sample, measure_iters=5)
    assert "xgboost_single_row_ms" in results
    assert "xgboost_batch_10k_ms" in results
    assert "xgboost_n_jobs_latency_ms" in results


def test_metrics_cosine_similarity_and_accuracy():
    """Verify cosine similarity and accuracy calculations."""
    try:
        import torch
        t1 = torch.tensor([1.0, 2.0, 3.0])
        t2 = torch.tensor([1.0, 2.0, 3.0])
        t3 = torch.tensor([-1.0, -2.0, -3.0])
        assert calculate_cosine_similarity(t1, t2) == pytest.approx(1.0, rel=1e-3)
        assert calculate_cosine_similarity(t1, t3) == pytest.approx(-1.0, rel=1e-3)
    except ImportError:
        pass

    assert calculate_accuracy([0, 1, 1, 0], [0, 1, 0, 0]) == 0.75
    assert calculate_accuracy([], []) == 0.0


def test_resource_sampler_lifecycle():
    """Verify ResourceSampler background thread lifecycle and save."""
    sampler = ResourceSampler(interval_s=0.05)
    sampler.start()
    import time
    time.sleep(0.12)
    records = sampler.stop()
    assert len(records) >= 1

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "resources.csv"
        saved_path = sampler.save(csv_path)
        assert saved_path.exists()
        assert saved_path.stat().st_size > 0


def test_env_metadata_save_and_load():
    """Verify collection and persistence of environment metadata."""
    meta = collect_env_metadata(target_device="cpu_laptop")
    assert "os" in meta
    assert "cpu" in meta
    assert meta["target_device"] == "cpu_laptop"

    with tempfile.TemporaryDirectory() as tmpdir:
        out_json = Path(tmpdir) / "env.json"
        saved = save_env_metadata(out_json, target_device="cpu_laptop")
        assert saved.exists()


def test_build_result_path_seq_len():
    """Verify build_result_path handles sequence lengths correctly without collision."""
    from src.core.paths import build_result_path

    with tempfile.TemporaryDirectory() as tmpdir:
        p_def = build_result_path(tmpdir, "cpu_laptop", "bert_base", "TN4", 8, "fp32", seq_len=128)
        assert p_def.name == "bs8_fp32"

        p_none = build_result_path(tmpdir, "cpu_laptop", "resnet50", "TN4", 8, "fp32")
        assert p_none.name == "bs8_fp32"

        p_64 = build_result_path(tmpdir, "cpu_laptop", "bert_base", "TN4", 8, "fp32", seq_len=64)
        assert p_64.name == "bs8_seq64_fp32"

        p_256 = build_result_path(tmpdir, "cpu_laptop", "bert_base", "TN4", 8, "fp32", seq_len=256)
        assert p_256.name == "bs8_seq256_fp32"

        assert p_64 != p_def
        assert p_256 != p_def


