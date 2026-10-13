"""Specialized benchmark routines for tabular models (XGBoost)."""

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def benchmark_xgboost_latency(
    model: Any,
    x_sample: Any,
    measure_iters: int = 20,
) -> Dict[str, Any]:
    """Benchmark XGBoost latency for single row vs 10,000 batch and across CPU thread counts."""
    results: Dict[str, Any] = {
        "xgboost_single_row_ms": 0.0,
        "xgboost_batch_10k_ms": 0.0,
        "xgboost_batch_10k_fps": 0.0,
        "xgboost_n_jobs_latency_ms": {},
    }

    if not HAS_NUMPY or x_sample is None:
        logger.warning("Numpy or x_sample not available for XGBoost specialized benchmark.")
        return results

    try:
        # 1. Single row inference (Batch = 1)
        x_single = x_sample[:1]
        for _ in range(5):  # Warmup
            _ = model.predict(x_single)

        single_times: List[float] = []
        for _ in range(measure_iters):
            t0 = time.perf_counter()
            _ = model.predict(x_single)
            t1 = time.perf_counter()
            single_times.append((t1 - t0) * 1000.0)

        single_median = float(np.median(single_times))
        results["xgboost_single_row_ms"] = round(single_median, 4)

        # 2. Large batch inference (10,000 rows)
        repeat_factor = int(np.ceil(10000 / len(x_sample)))
        x_10k = np.repeat(x_sample, repeat_factor, axis=0)[:10000]

        for _ in range(3):  # Warmup
            _ = model.predict(x_10k)

        batch_times: List[float] = []
        for _ in range(max(5, measure_iters // 2)):
            t0 = time.perf_counter()
            _ = model.predict(x_10k)
            t1 = time.perf_counter()
            batch_times.append((t1 - t0) * 1000.0)

        batch_median = float(np.median(batch_times))
        results["xgboost_batch_10k_ms"] = round(batch_median, 4)
        if batch_median > 0:
            results["xgboost_batch_10k_fps"] = round((10000.0 * 1000.0) / batch_median, 2)

        # 3. Thread scaling across n_jobs = [1, 2, 4, -1]
        n_jobs_list = [1, 2, 4, -1]
        n_jobs_latencies: Dict[str, float] = {}

        for nj in n_jobs_list:
            if hasattr(model, "set_params"):
                try:
                    model.set_params(n_jobs=nj)
                except Exception:
                    pass

            t_list: List[float] = []
            for _ in range(5):
                t0 = time.perf_counter()
                _ = model.predict(x_10k)
                t1 = time.perf_counter()
                t_list.append((t1 - t0) * 1000.0)

            n_jobs_latencies[str(nj)] = round(float(np.median(t_list)), 4)

        results["xgboost_n_jobs_latency_ms"] = n_jobs_latencies

    except Exception as exc:
        logger.error("Error during specialized XGBoost benchmarking: %s", exc)

    return results
