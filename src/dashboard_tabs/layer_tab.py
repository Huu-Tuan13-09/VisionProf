"""
layer_tab.py — Tab dashboard "Layer" (TV1 · CH1, CH2)
=====================================================
Ghép vào khung dashboard chung (TV3):

    from src.dashboard_tabs.layer_tab import render_layer_tab
    with tab_layer:
        render_layer_tab()

Xem trước riêng tab này:
    streamlit run src/dashboard_tabs/layer_tab.py

Đọc kết quả có sẵn trong results/ (TN1, TN2) — không chạy lại thực nghiệm.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

RESULTS_DIR = _PROJECT_ROOT / "results"
CPU_HW, GPU_HW = "cpu_laptop", "colab_t4"
CONFIG = "bs1_fp32"
MODEL_LABELS = {
    "mobilenet_v3_large": "MobileNetV3-Large", "resnet50": "ResNet-50", "yolov8n": "YOLOv8n",
    "vit_b_16": "ViT-B/16", "distilbert": "DistilBERT", "bert_base": "BERT-base", "mlp": "MLP",
}
DEVICE_LABELS = {CPU_HW: "CPU laptop", GPU_HW: "GPU Colab T4"}

# Bảng màu (reference palette) — thứ tự cố định, đã kiểm tra mù màu; bản sáng / tối
_PALETTE = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"],
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
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def _colors() -> Dict[str, str]:
    return dict(zip(GROUPS, _PALETTE[_mode()]))


# ── Đọc dữ liệu ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _read_csv(path: str) -> Optional[pd.DataFrame]:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def _read_json(path: str) -> Optional[dict]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _available_models(results_dir: Path) -> list:
    base = results_dir / "analysis" / "TN1"
    return [m for m in MODEL_LABELS if (base / m).exists()]


def _tn1(results_dir: Path, hw: str, model: str, name: str):
    path = results_dir / hw / model / "TN1" / CONFIG / name
    return _read_json(str(path)) if name.endswith(".json") else _read_csv(str(path))


# ── Biểu đồ ─────────────────────────────────────────────────────────────────

def _fig_type_share(share: pd.DataFrame) -> go.Figure:
    share = share.copy()
    share["group"] = share["layer_type"].map(GROUP_OF).fillna("Khác")
    colors = _colors()
    fig = go.Figure()
    for g in GROUPS:
        sub = share[share["group"] == g]
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


def _fig_roofline(summary: pd.DataFrame, hardware: dict) -> go.Figure:
    df = summary[(summary["flops_source"] == "counter") & (summary["time_ms"] > 0)].copy()
    df["group"] = df["layer_type"].map(GROUP_OF).fillna("Khác")
    peak, bw = hardware["peak_gflops_fp32"], hardware["bandwidth_gbs"]
    ridge = peak / bw
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
                           "%{customdata[2]:.3f} ms · hiệu suất %{customdata[3]:.0f}% · %{customdata[4]}-bound"
                           "<extra></extra>"),
        )
    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(type="log", title="Arithmetic intensity (FLOP / byte)"),
        yaxis=dict(type="log", title="Tốc độ tính đạt được (GFLOP/s)"),
        legend=dict(orientation="h", y=1.12, x=0),
        title=dict(text=f"{hardware['name']} · trần {peak / 1000:.1f} TFLOP/s, {bw:.0f} GB/s · "
                        f"điểm gãy ≈ {ridge:.0f} FLOP/byte", font=dict(size=13)),
    )
    return fig


def _fig_timeline(layers: pd.DataFrame) -> go.Figure:
    per = (layers.groupby(["layer_name", "layer_type", "call_index"], as_index=False)
           .agg(time_ms=("time_ms", "median"), exec_order=("exec_order", "min"))
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


# ── Tab ─────────────────────────────────────────────────────────────────────

def render_layer_tab(results_dir: Path = RESULTS_DIR) -> None:
    """Vẽ tab "Layer" — gọi bên trong một `st.tabs(...)` của dashboard chung."""
    results_dir = Path(results_dir)
    models = _available_models(results_dir)
    if not models:
        st.info("Chưa có kết quả TN1. Chạy `python experiments/tn1_layer_breakdown.py --analyze` trước.")
        return

    c1, c2 = st.columns([2, 1])
    model = c1.selectbox("Mô hình", models, format_func=MODEL_LABELS.get, key="layer_tab_model")
    hw = c2.radio("Máy (cho biểu đồ chi tiết)", [GPU_HW, CPU_HW], format_func=DEVICE_LABELS.get,
                  horizontal=True, key="layer_tab_hw")

    share = _read_csv(str(results_dir / "analysis" / "TN1" / model / "compare_type_share.csv"))
    top = _read_csv(str(results_dir / "analysis" / "TN1" / model / "compare_top10.csv"))
    meta_cpu, meta_gpu = _tn1(results_dir, CPU_HW, model, "layer_meta.json"), _tn1(results_dir, GPU_HW, model, "layer_meta.json")

    # ── Chỉ số chính ──
    if share is not None:
        t_cpu, t_gpu = share["time_ms_cpu"].sum(), share["time_ms_gpu"].sum()
        k = st.columns(4)
        k[0].metric("Tổng thời gian layer · CPU", f"{t_cpu:.2f} ms")
        k[1].metric("Tổng thời gian layer · GPU T4", f"{t_gpu:.2f} ms")
        k[2].metric("GPU nhanh hơn", f"×{t_cpu / t_gpu:.1f}" if t_gpu > 0 else "—")
        meta = meta_gpu or meta_cpu
        if meta and meta.get("total_flops_model"):
            flops = meta["total_flops_model"]
            k[3].metric("FLOPs / 1 lần forward",
                        f"{flops / 1e9:.2f} G" if flops >= 1e8 else f"{flops / 1e6:.2f} M",
                        help=f"{meta['n_hooked_layers']} layer được đo")

        st.markdown("##### Thời gian tập trung ở nhóm layer nào? (CH1)")
        st.plotly_chart(_fig_type_share(share), width="stretch", key="layer_tab_share")
        with st.expander("Bảng tỷ trọng theo loại layer"):
            st.dataframe(share.round(3), width="stretch", hide_index=True)
    else:
        st.warning("Cần kết quả của cả CPU và GPU để so sánh — chạy `--analyze` sau khi gom đủ số liệu.")

    # ── Top-10 ──
    if top is not None:
        st.markdown("##### Top-10 layer chậm nhất — CPU và GPU đặt cạnh nhau")
        cols = ["layer_name", "layer_type", "time_ms_cpu", "rank_cpu", "time_ms_gpu", "rank_gpu", "speedup"]
        st.dataframe(
            top[cols].rename(columns={
                "layer_name": "Layer", "layer_type": "Loại", "time_ms_cpu": "CPU (ms)", "rank_cpu": "Hạng CPU",
                "time_ms_gpu": "GPU (ms)", "rank_gpu": "Hạng GPU", "speedup": "Nhanh hơn (×)",
            }).round(3),
            width="stretch", hide_index=True,
        )

    # ── Chi tiết theo máy ──
    layers = _tn1(results_dir, hw, model, "layers.csv")
    summary = _tn1(results_dir, hw, model, "layers_summary.csv")
    meta = meta_gpu if hw == GPU_HW else meta_cpu
    if layers is not None:
        st.markdown(f"##### Thời gian từng layer theo thứ tự chạy · {DEVICE_LABELS[hw]}")
        st.plotly_chart(_fig_timeline(layers), width="stretch", key="layer_tab_timeline")
    if summary is not None and meta and meta.get("hardware"):
        st.markdown(f"##### Roofline · {DEVICE_LABELS[hw]} — vì sao layer lớn chưa nhanh hơn?")
        st.plotly_chart(_fig_roofline(summary, meta["hardware"]), width="stretch", key="layer_tab_roof")
        st.caption("Chỉ vẽ layer có FLOPs đếm được (Conv, Linear, Attention). Layer nặng về bộ nhớ "
                   "(BatchNorm, ReLU…) không vẽ vì dữ liệu nhỏ nằm trong cache khiến % hiệu suất vượt 100%.")

    # ── CH2 ──
    st.markdown("##### Công cụ đo có đúng không? (CH2 · TN2)")
    rows = []
    for h in (CPU_HW, GPU_HW):
        d = _read_csv(str(results_dir / "analysis" / "TN2" / f"summary_{h}.csv"))
        if d is not None:
            d = d.assign(Máy=DEVICE_LABELS[h])
            rows.append(d)
    if rows:
        tn2 = pd.concat(rows)
        show = {
            "Máy": "Máy", "model": "Mô hình", "total_rel_error_pct": "Sai lệch so với torch.profiler (%)",
            "overhead_pct": "Overhead (%)", "unattributed_pct": "Không thuộc layer nào (%)",
            "layers_below_3x_floor_pct": "Layer dưới 3× ngưỡng sàn (%)",
        }
        st.dataframe(tn2[[c for c in show if c in tn2]].rename(columns=show).round(2),
                     width="stretch", hide_index=True)
    else:
        st.info("Chưa có kết quả TN2. Chạy `python experiments/tn2_validate_profiler.py`.")


if __name__ == "__main__":
    st.set_page_config(page_title="Layer · VisionProf", page_icon="🔬", layout="wide")
    st.title("🔬 Mức Layer")
    render_layer_tab()
