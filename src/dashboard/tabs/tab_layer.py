"""Tab 3: Leaf layer breakdown, Roofline modeling, and hotspot analysis [Owned by TV1]."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.dashboard.utils import page_header, show_chart

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

try:
    import streamlit as st
except ImportError:
    st = None

CPU_HW, GPU_HW = "cpu_laptop", "colab_t4"
CONFIG = "bs1_fp32"
MODEL_LABELS = {
    "mobilenet_v3_large": "MobileNetV3-Large", "resnet50": "ResNet-50", "yolov8n": "YOLOv8n",
    "vit_b_16": "ViT-B/16", "distilbert": "DistilBERT", "bert_base": "BERT-base", "mlp": "MLP",
}
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


def _cache_data(func: Any) -> Any:
    if st is not None and hasattr(st, "cache_data"):
        return st.cache_data(show_spinner=False)(func)
    return func


def _mode() -> str:
    try:
        return "dark" if st is not None and st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def _colors() -> Dict[str, str]:
    return dict(zip(GROUPS, _PALETTE[_mode()]))


@_cache_data
def _read_csv(path: str) -> Optional[pd.DataFrame]:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


@_cache_data
def _read_json(path: str) -> Optional[dict]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _find_tn1_path(results_dir: Path, *parts: str) -> Path:
    p = results_dir.joinpath(*parts)
    if not p.exists():
        lowered_parts = [part.lower() if part in ("TN1", "TN2") else part for part in parts]
        alt = results_dir.joinpath(*lowered_parts)
        if alt.exists():
            return alt
    return p


def _available_models(results_dir: Path) -> List[str]:
    base = _find_tn1_path(results_dir, "analysis", "TN1")
    return [m for m in MODEL_LABELS if (base / m).exists()]


def _tn1(results_dir: Path, hw: str, model: str, name: str) -> Any:
    path = _find_tn1_path(results_dir, hw, model, "TN1", CONFIG, name)
    return _read_json(str(path)) if name.endswith(".json") else _read_csv(str(path))


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


def _fig_roofline(summary: pd.DataFrame, hardware: dict) -> Optional[Any]:
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
        height=420, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(type="log", title="Cường độ tính toán (FLOP / byte)"),
        yaxis=dict(type="log", title="Tốc độ tính đạt được (GFLOP/s)"),
        legend=dict(orientation="h", y=1.12, x=0),
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


def render_layer_tab(results_dir: Path) -> None:
    """Render Tab 3: Detailed leaf layer breakdown, Roofline curve, and CPU vs GPU comparison."""
    if st is None:
        return
    page_header("Phân tích layer", "layer")
    models = _available_models(results_dir)
    if not models:
        st.info("Chưa có kết quả TN1. Chạy `python experiments/tn1_layer_breakdown.py --analyze` trước.")
        return

    c1, c2 = st.columns([2, 1])
    model = c1.selectbox("Mô hình", models, format_func=MODEL_LABELS.get, key="layer_tab_model")
    hw = c2.radio("Thiết bị cho biểu đồ chi tiết", [GPU_HW, CPU_HW], format_func=DEVICE_LABELS.get, horizontal=True, key="layer_tab_hw")

    share_path = _find_tn1_path(results_dir, "analysis", "TN1", model, "compare_type_share.csv")
    top_path = _find_tn1_path(results_dir, "analysis", "TN1", model, "compare_top10.csv")
    share, top = _read_csv(str(share_path)), _read_csv(str(top_path))
    meta_cpu, meta_gpu = _tn1(results_dir, CPU_HW, model, "layer_meta.json"), _tn1(results_dir, GPU_HW, model, "layer_meta.json")

    if share is not None and "time_ms_cpu" in share.columns:
        t_cpu, t_gpu = share["time_ms_cpu"].sum(), share["time_ms_gpu"].sum()
        k = st.columns(4)
        k[0].metric("Tổng thời gian layer · CPU", f"{t_cpu:.2f} ms")
        k[1].metric("Tổng thời gian layer · GPU T4", f"{t_gpu:.2f} ms")
        k[2].metric("Tăng tốc CPU / T4", f"×{t_cpu / t_gpu:.1f}" if t_gpu > 0 else "—")
        meta = meta_gpu or meta_cpu
        if meta and meta.get("total_flops_counter"):
            flops = meta["total_flops_counter"]
            k[3].metric("FLOPs / 1 lần forward", f"{flops / 1e9:.2f} G" if flops >= 1e8 else f"{flops / 1e6:.2f} M")

        st.markdown("##### Tỷ trọng thời gian theo loại layer")
        fig_share = _fig_type_share(share)
        if fig_share is not None:
            show_chart(fig_share, key="layer_tab_share")
        with st.expander("Xem bảng số · tỷ trọng layer"):
            st.dataframe(share.round(3), width="stretch", hide_index=True, column_config={
                "layer_type": "Loại layer", "time_ms_cpu": "CPU (ms)", "time_ms_gpu": "T4 (ms)",
                "pct_cpu": "CPU (%)", "pct_gpu": "T4 (%)", "speedup": "Tăng tốc (×)",
            })

    if top is not None:
        st.markdown("##### Top-10 layer chậm · CPU và T4")
        cols = [c for c in ["layer_name", "layer_type", "time_ms_cpu", "rank_cpu", "time_ms_gpu", "rank_gpu", "speedup"] if c in top.columns]
        st.dataframe(top[cols].round(3), width="stretch", hide_index=True, column_config={
            "layer_name": "Layer", "layer_type": "Loại", "time_ms_cpu": "CPU (ms)",
            "time_ms_gpu": "T4 (ms)", "rank_cpu": "Hạng CPU", "rank_gpu": "Hạng T4", "speedup": "Tăng tốc (×)",
        })

    layers = _tn1(results_dir, hw, model, "layers.csv")
    summary = _tn1(results_dir, hw, model, "layers_summary.csv")
    meta = meta_gpu if hw == GPU_HW else meta_cpu
    if layers is not None and not layers.empty:
        st.markdown(f"##### Thời gian từng layer theo thứ tự chạy · {DEVICE_LABELS.get(hw, hw)}")
        fig_time = _fig_timeline(layers)
        if fig_time is not None:
            show_chart(fig_time, key="layer_tab_timeline")
    if summary is not None and meta and meta.get("hardware"):
        st.markdown(f"##### Roofline · {DEVICE_LABELS.get(hw, hw)}")
        fig_roof = _fig_roofline(summary, meta["hardware"])
        if fig_roof is not None:
            show_chart(fig_roof, key="layer_tab_roof")


def render(results_dir: Path) -> None:
    """Entrypoint compatible with router in src/dashboard/app.py."""
    render_layer_tab(results_dir)
