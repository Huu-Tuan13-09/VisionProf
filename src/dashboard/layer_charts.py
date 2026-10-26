"""Biểu đồ layer và Roofline dùng chung cho hai trang phân tích."""

from typing import Any, Dict, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

CPU_HW, GPU_HW = "cpu_laptop", "colab_t4"
DEVICE_LABELS = {CPU_HW: "CPU laptop", GPU_HW: "GPU Colab T4"}

_PALETTE = {
    "light": ["#B4480F", "#2456A6", "#1E7A4C", "#9C740D", "#AB3270", "#54717C", "#7757A6"],
    "dark":  ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"],
}
_ROOF_INK = {"light": "#52514e", "dark": "#c3c2b7"}
GROUPS = ["Conv", "Linear", "Attention", "Chuẩn hoá", "Kích hoạt", "Pooling", "Khác"]
GROUP_OF = {
    "Conv2d": "Conv", "Linear": "Linear", "NonDynamicallyQuantizableLinear": "Linear",
    "MultiheadAttention": "Attention", "BatchNorm2d": "Chuẩn hoá", "LayerNorm": "Chuẩn hoá",
    "ReLU": "Kích hoạt", "GELU": "Kích hoạt", "GELUActivation": "Kích hoạt", "SiLU": "Kích hoạt",
    "Hardswish": "Kích hoạt", "Hardsigmoid": "Kích hoạt", "Tanh": "Kích hoạt", "Sigmoid": "Kích hoạt",
    "MaxPool2d": "Pooling", "AdaptiveAvgPool2d": "Pooling", "AvgPool2d": "Pooling",
}


def _mode() -> str:
    try:
        return "dark" if st is not None and st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def _colors() -> Dict[str, str]:
    return dict(zip(GROUPS, _PALETTE[_mode()]))


def _fig_type_share(share: pd.DataFrame) -> Optional[Any]:
    if go is None:
        return None
    df = share.copy()
    df["group"] = df["layer_type"].map(GROUP_OF).fillna("Khác")
    colors = _colors()
    fig = go.Figure()
    for g in GROUPS:
        sub = df[df["group"] == g]
        if sub.empty:
            continue
        cpu, gpu = sub["pct_cpu"].sum(), sub["pct_gpu"].sum()
        types = ", ".join(sub["layer_type"])
        fig.add_bar(
            y=[DEVICE_LABELS[CPU_HW], DEVICE_LABELS[GPU_HW]], x=[cpu, gpu], name=g, orientation="h",
            marker=dict(color=colors[g], line=dict(width=0)), width=0.5,
            text=[f"{v:.0f}%" if v >= 8 else "" for v in (cpu, gpu)], textposition="inside",
            hovertemplate=f"<b>{g}</b> ({types})<br>%{{y}}: %{{x:.1f}}%<extra></extra>",
        )
    fig.update_layout(
        barmode="stack", bargap=0.35, height=260, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[0, 100], title="% thời gian các layer", ticksuffix="%"),
        yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=1.15, x=0),
    )
    return fig


def make_roofline_chart(summary: pd.DataFrame, hardware: Dict[str, Any]) -> Optional[Any]:
    """Vẽ Roofline từ bảng layer đã phân tích, giữ nguyên dữ liệu và đỉnh phần cứng."""
    if go is None:
        return None
    df = summary[(summary["flops_source"] == "counter") & (summary["time_ms"] > 0)].copy()
    df["group"] = df["layer_type"].map(GROUP_OF).fillna("Khác")
    peak, bw = hardware.get("peak_gflops_fp32", 8100.0), hardware.get("bandwidth_gbs", 320.0)
    ridge = peak / bw if bw > 0 else 1.0
    lo = max(min(df["arithmetic_intensity"].min(), 1.0) / 2, 0.05) if not df.empty else 0.1
    hi = max(df["arithmetic_intensity"].max() * 2, ridge * 4) if not df.empty else 1000
    colors = _colors()
    fig = go.Figure()
    fig.add_scatter(
        x=[lo, ridge, hi], y=[bw * lo, peak, peak], mode="lines", name="Trần Roofline",
        line=dict(color=_ROOF_INK[_mode()], width=2), hoverinfo="skip",
    )
    for g in GROUPS:
        sub = df[df["group"] == g]
        if sub.empty:
            continue
        fig.add_scatter(
            x=sub["arithmetic_intensity"], y=sub["achieved_gflops"], mode="markers", name=g,
            marker=dict(size=10, color=colors[g], line=dict(width=2, color="rgba(0,0,0,0)")),
            customdata=sub[["layer_name", "layer_type", "time_ms", "efficiency_pct", "bound"]],
            hovertemplate=("<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                           "AI: %{x:.1f} FLOP/byte · %{y:.0f} GFLOP/s<br>"
                           "%{customdata[2]:.3f} ms · hiệu suất %{customdata[3]:.0f}% · %{customdata[4]}-bound<extra></extra>"),
        )
    fig.update_layout(
        height=460, margin=dict(l=20, r=20, t=100, b=25),
        xaxis=dict(type="log", dtick=1, exponentformat="none", title="Cường độ tính toán (FLOP / byte)"),
        yaxis=dict(type="log", dtick=1, exponentformat="none", title="Tốc độ tính đạt được (GFLOP/s)"),
        legend=dict(orientation="h", y=1.04, x=0),
        title=dict(text=f"{hardware.get('name', 'Device')} · trần {peak / 1000:.1f} TFLOP/s, {bw:.0f} GB/s · điểm gãy ≈ {ridge:.0f} FLOP/byte", font=dict(size=13)),
    )
    return fig


def _fig_timeline(layers: pd.DataFrame) -> Optional[Any]:
    if go is None:
        return None
    order_col = "exec_order" if "exec_order" in layers.columns else "call_index"
    per = (layers.groupby(["layer_name", "layer_type", "call_index"], as_index=False)
           .agg(time_ms=("time_ms", "median"), exec_order=(order_col, "min"))
           .sort_values("exec_order"))
    per["group"] = per["layer_type"].map(GROUP_OF).fillna("Khác")
    colors = _colors()
    fig = go.Figure()
    for g in GROUPS:
        sub = per[per["group"] == g]
        if sub.empty:
            continue
        fig.add_bar(
            x=sub["exec_order"], y=sub["time_ms"], name=g, marker=dict(color=colors[g], line=dict(width=0)),
            customdata=sub[["layer_name", "layer_type"]],
            hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>%{y:.3f} ms<extra></extra>",
        )
    fig.update_layout(
        barmode="overlay", bargap=0.15, height=320, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="Thứ tự layer chạy trong 1 lần forward"), yaxis=dict(title="ms (trung vị)"),
        legend=dict(orientation="h", y=1.15, x=0),
    )
    return fig


