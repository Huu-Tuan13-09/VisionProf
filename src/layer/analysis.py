"""Layer-level performance analysis, ranking, and aggregation utilities."""

from pathlib import Path
from typing import List, Union

import pandas as pd


def load_layers(path: Union[str, Path]) -> pd.DataFrame:
    """Read layers.csv file safely into DataFrame."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "layer_name" in df.columns:
        df["layer_name"] = df["layer_name"].astype(str)
    if "layer_type" in df.columns:
        df["layer_type"] = df["layer_type"].astype(str)
    return df



def per_layer_time(layers: pd.DataFrame, time_col: str = "time_ms") -> pd.DataFrame:
    """Calculate median forward duration per (layer, call_index), aggregated across iterations."""
    if layers.empty or time_col not in layers.columns:
        return pd.DataFrame()

    has_call_idx = "call_index" in layers.columns
    has_act = "activation_mb" in layers.columns
    has_order = "exec_order" in layers.columns

    group_keys = ["layer_name", "layer_type"]
    if has_call_idx:
        group_keys.append("call_index")

    agg_dict: dict = {time_col: "median"}
    if has_act:
        agg_dict["activation_mb"] = "median"
    if has_order:
        agg_dict["exec_order"] = "min"

    per_call = layers.groupby(group_keys, as_index=False).agg(**{k: (k, fn) for k, fn in agg_dict.items()})

    named_aggs = {"time_ms": (time_col, "sum")}
    if has_act:
        named_aggs["activation_mb"] = ("activation_mb", "sum")
    if has_call_idx:
        named_aggs["n_calls"] = ("call_index", "count")
    if has_order:
        named_aggs["exec_order"] = ("exec_order", "min")

    df = per_call.groupby(["layer_name", "layer_type"], as_index=False).agg(**named_aggs)
    total = df["time_ms"].sum()
    df["pct"] = (df["time_ms"] / total * 100.0) if total > 0 else 0.0
    return df



def top_k_layers(layers: pd.DataFrame, k: int = 10, time_col: str = "time_ms") -> pd.DataFrame:
    """Extract top-K slowest layers ranked by execution time."""
    if layers.empty:
        return pd.DataFrame()
    df = per_layer_time(layers, time_col=time_col).sort_values("time_ms", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    cols = [c for c in ["rank", "layer_name", "layer_type", "time_ms", "pct", "activation_mb", "n_calls"] if c in df.columns]
    return df.head(k)[cols]


def type_share(layers: pd.DataFrame, time_col: str = "time_ms") -> pd.DataFrame:
    """Calculate cumulative latency share aggregated by layer type."""
    if layers.empty:
        return pd.DataFrame()
    df = per_layer_time(layers, time_col=time_col)
    out = (
        df.groupby("layer_type", as_index=False)
        .agg(time_ms=("time_ms", "sum"), n_layers=("layer_name", "count"))
        .sort_values("time_ms", ascending=False)
    )
    total = out["time_ms"].sum()
    out["pct"] = (out["time_ms"] / total * 100.0) if total > 0 else 0.0
    return out.reset_index(drop=True)


def compare_top_k(cpu_layers: pd.DataFrame, gpu_layers: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Compare top slow layers between CPU and GPU configurations side-by-side."""
    def _ranked(layers: pd.DataFrame, tag: str) -> pd.DataFrame:
        df = per_layer_time(layers).sort_values("time_ms", ascending=False).reset_index(drop=True)
        df[f"rank_{tag}"] = range(1, len(df) + 1)
        df["layer_name"] = df["layer_name"].astype(str)
        df["layer_type"] = df["layer_type"].astype(str)
        return df.rename(columns={"time_ms": f"time_ms_{tag}", "pct": f"pct_{tag}"})[
            ["layer_name", "layer_type", f"time_ms_{tag}", f"pct_{tag}", f"rank_{tag}"]
        ]


    merged = _ranked(cpu_layers, "cpu").merge(
        _ranked(gpu_layers, "gpu"), on=["layer_name", "layer_type"], how="outer"
    )
    merged = merged[(merged["rank_cpu"] <= k) | (merged["rank_gpu"] <= k)]
    merged["speedup"] = merged["time_ms_cpu"] / merged["time_ms_gpu"]
    merged["rank_change"] = merged["rank_cpu"] - merged["rank_gpu"]
    return merged.sort_values("rank_cpu").reset_index(drop=True)


def compare_type_share(cpu_layers: pd.DataFrame, gpu_layers: pd.DataFrame) -> pd.DataFrame:
    """Compare layer type cumulative latency shares between CPU and GPU."""
    cpu_s = type_share(cpu_layers).rename(columns={"pct": "pct_cpu", "time_ms": "time_ms_cpu", "n_layers": "n_layers_cpu"})
    gpu_s = type_share(gpu_layers).rename(columns={"pct": "pct_gpu", "time_ms": "time_ms_gpu", "n_layers": "n_layers_gpu"})
    merged = cpu_s.merge(gpu_s, on="layer_type", how="outer").fillna(0.0)
    merged["pct_change"] = merged["pct_gpu"] - merged["pct_cpu"]
    return merged.sort_values("time_ms_cpu", ascending=False).reset_index(drop=True)


def to_markdown(df: pd.DataFrame, float_fmt: str = "%.3f") -> str:
    """Format DataFrame as a Markdown table string."""
    if df.empty:
        return ""
    return df.to_markdown(index=False, floatfmt=float_fmt)
