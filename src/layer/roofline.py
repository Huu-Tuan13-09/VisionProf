"""Roofline modeling and computational efficiency evaluation per layer."""

import numpy as np
import pandas as pd

from src.layer.hw_specs import HardwareSpec


def add_efficiency(
    layer_times: pd.DataFrame,
    analysis: pd.DataFrame,
    spec: HardwareSpec,
    precision: str = "fp32",
    time_col: str = "forward_time_ms",
) -> pd.DataFrame:
    """Merge empirical layer durations with theoretical analysis to calculate Roofline efficiency metrics."""
    if layer_times.empty:
        return pd.DataFrame()

    per_call = (
        layer_times.groupby(["layer_name", "layer_type", "call_index"], as_index=False)[time_col]
        .median()
        .rename(columns={time_col: "time_ms"})
    )

    if analysis.empty:
        df = per_call.copy()
        df["flops"] = 0.0
        df["flops_source"] = "none"
        df["bytes_total"] = 0.0
        df["arithmetic_intensity"] = 0.0
    else:
        cols = ["layer_name", "flops", "flops_source", "bytes_total", "arithmetic_intensity"]
        df = per_call.merge(analysis[cols], on="layer_name", how="left")

    peak = spec.peak_gflops(precision)
    bw = spec.bandwidth_gbs
    ridge = spec.ridge_point(precision)
    seconds = df["time_ms"] / 1000.0
    valid = seconds > 0

    df["gflops"] = df["flops"] / 1e9
    df["achieved_gflops"] = np.where(valid, df["gflops"] / seconds.where(valid, 1), np.nan)
    df["achieved_gbs"] = np.where(valid, df["bytes_total"] / 1e9 / seconds.where(valid, 1), np.nan)
    df["attainable_gflops"] = np.minimum(peak, bw * df["arithmetic_intensity"])
    has_flops = df["flops"] > 0
    attainable_safe = df["attainable_gflops"].replace(0, np.nan)
    bw_safe = max(bw, 1e-6)
    df["efficiency_pct"] = np.where(
        has_flops,
        np.where(attainable_safe.notna() & (attainable_safe > 0), df["achieved_gflops"] / attainable_safe * 100.0, np.nan),
        df["achieved_gbs"] / bw_safe * 100.0,
    )
    df["bound"] = np.where(df["arithmetic_intensity"] >= ridge, "compute", "memory")
    df["hw_id"] = spec.hw_id
    df["precision"] = precision
    return df
