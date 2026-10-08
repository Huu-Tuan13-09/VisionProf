"""
layer_analysis.py — Phân tích kết quả mức Layer (TV1 · CH1)
============================================================
Đọc layers.csv do LayerPlugin ghi ra và tạo các bảng dùng cho báo cáo / dashboard:
  - per_layer_time      : thời gian mỗi layer (trung vị qua các iteration)
  - top_k_layers        : top-k layer chậm nhất
  - type_share          : tỷ trọng thời gian theo loại layer
  - compare_top_k       : bảng top-k CPU và GPU đặt cạnh nhau (kèm hạng trên mỗi máy)
  - compare_type_share  : tỷ trọng theo loại layer, CPU và GPU đặt cạnh nhau
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import pandas as pd


def load_layers(path: Union[str, Path]) -> pd.DataFrame:
    """Đọc 1 file layers.csv."""
    return pd.read_csv(path)


def per_layer_time(layers: pd.DataFrame, time_col: str = "time_ms") -> pd.DataFrame:
    """
    Thời gian của mỗi layer trong 1 lần forward.
    Lấy trung vị qua các iteration cho từng (layer, lần gọi), rồi cộng các lần gọi
    của cùng một layer (module dùng chung được gọi nhiều lần).
    """
    per_call = (
        layers.groupby(["layer_name", "layer_type", "call_index"], as_index=False)
        .agg(time_ms=(time_col, "median"), activation_mb=("activation_mb", "median"),
             exec_order=("exec_order", "min"))
    )
    df = (
        per_call.groupby(["layer_name", "layer_type"], as_index=False)
        .agg(time_ms=("time_ms", "sum"), activation_mb=("activation_mb", "sum"),
             n_calls=("call_index", "count"), exec_order=("exec_order", "min"))
    )
    total = df["time_ms"].sum()
    df["pct"] = df["time_ms"] / total * 100.0 if total > 0 else 0.0
    return df.sort_values("exec_order").reset_index(drop=True)


def top_k_layers(layers: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Top-k layer chậm nhất, kèm hạng và % tổng thời gian."""
    df = per_layer_time(layers).sort_values("time_ms", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df.head(k)[["rank", "layer_name", "layer_type", "time_ms", "pct", "activation_mb", "n_calls"]]


def type_share(layers: pd.DataFrame) -> pd.DataFrame:
    """Tỷ trọng thời gian theo loại layer (Conv2d, BatchNorm2d, ReLU, Linear…)."""
    df = per_layer_time(layers)
    out = (
        df.groupby("layer_type", as_index=False)
        .agg(time_ms=("time_ms", "sum"), n_layers=("layer_name", "count"))
        .sort_values("time_ms", ascending=False)
    )
    total = out["time_ms"].sum()
    out["pct"] = out["time_ms"] / total * 100.0 if total > 0 else 0.0
    return out.reset_index(drop=True)


def compare_top_k(cpu_layers: pd.DataFrame, gpu_layers: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """
    Bảng gộp CPU và GPU: các layer nằm trong top-k của ít nhất 1 máy,
    kèm thời gian và hạng trên từng máy — thấy ngay layer nào "đổi hạng".
    """
    def ranked(layers: pd.DataFrame, tag: str) -> pd.DataFrame:
        df = per_layer_time(layers).sort_values("time_ms", ascending=False).reset_index(drop=True)
        df[f"rank_{tag}"] = range(1, len(df) + 1)
        return df.rename(columns={"time_ms": f"time_ms_{tag}", "pct": f"pct_{tag}"})[
            ["layer_name", "layer_type", f"time_ms_{tag}", f"pct_{tag}", f"rank_{tag}"]
        ]

    merged = ranked(cpu_layers, "cpu").merge(
        ranked(gpu_layers, "gpu"), on=["layer_name", "layer_type"], how="outer"
    )
    merged = merged[(merged["rank_cpu"] <= k) | (merged["rank_gpu"] <= k)]
    merged["speedup"] = merged["time_ms_cpu"] / merged["time_ms_gpu"]
    merged["rank_change"] = merged["rank_cpu"] - merged["rank_gpu"]
    return merged.sort_values("rank_cpu").reset_index(drop=True)


def compare_type_share(cpu_layers: pd.DataFrame, gpu_layers: pd.DataFrame) -> pd.DataFrame:
    """Tỷ trọng theo loại layer trên CPU và GPU đặt cạnh nhau."""
    cpu = type_share(cpu_layers).rename(columns={"time_ms": "time_ms_cpu", "pct": "pct_cpu"})
    gpu = type_share(gpu_layers).rename(columns={"time_ms": "time_ms_gpu", "pct": "pct_gpu"})
    df = cpu.merge(gpu.drop(columns="n_layers"), on="layer_type", how="outer").fillna(0.0)
    df["pct_change"] = df["pct_gpu"] - df["pct_cpu"]
    return df.sort_values("pct_cpu", ascending=False).reset_index(drop=True)


def to_markdown(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Bảng Markdown để dán vào báo cáo (không cần thư viện tabulate)."""
    cols: List[str] = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(" --- " for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(format(v, floatfmt) if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
