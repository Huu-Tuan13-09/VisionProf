"""Tab 6: Tỷ trọng, overhead và thời gian các pipeline TN7."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.dashboard.utils import page_header, show_chart

try:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

from src.dashboard.utils import DEVICE_COLORS, PHASE_COLORS as _PHASE_COLORS

_MODEL_LABELS: Dict[str, str] = {
    "yolov8n":  "YOLO v8-nano (CV)",
    "bert_base": "BERT-base (NLP)",
    "xgboost":  "XGBoost (Tabular)",
}
_OVERHEAD_PHASES = {
    "preprocess", "tokenize", "csv_parse", "postprocess_nms", "decode_pooling"
}


def _load_tn7_phases(results_dir: Path) -> pd.DataFrame:
    """Scan **/TN7/**/phases.csv and aggregate per (device, model, phase)."""
    rows: List[Dict[str, Any]] = []
    for csv_path in results_dir.glob("**/TN7/**/phases.csv"):
        parts = csv_path.relative_to(results_dir).parts
        if len(parts) < 4:
            continue
        device, model = parts[0], parts[1]
        try:
            df_raw = pd.read_csv(csv_path)
        except Exception:
            continue
        if df_raw.empty or "phase" not in df_raw.columns:
            continue
        for phase, grp in df_raw.groupby("phase"):
            rows.append({
                "device": device,
                "model": model,
                "model_label": _MODEL_LABELS.get(model, model),
                "phase": phase,
                "mean_ms": round(grp["time_ms"].mean(), 4),
                "p50_ms":  round(grp["time_ms"].median(), 4),
                "p95_ms":  round(grp["time_ms"].quantile(0.95), 4),
                "n_iter":  len(grp),
                "is_overhead": phase in _OVERHEAD_PHASES,
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _compute_overhead_pct(df: pd.DataFrame) -> pd.DataFrame:
    """Compute % time outside model (overhead) per (device, model)."""
    total = df.groupby(["device", "model"])["mean_ms"].sum().rename("total_ms")
    overhead = df[df["is_overhead"]].groupby(["device", "model"])["mean_ms"].sum().rename("overhead_ms")
    merged = pd.concat([total, overhead], axis=1).fillna(0).reset_index()
    merged["overhead_pct"] = (merged["overhead_ms"] / merged["total_ms"] * 100).round(1)
    merged["model_label"] = merged["model"].map(_MODEL_LABELS).fillna(merged["model"])
    return merged


def _render_pie_breakdown(df: pd.DataFrame) -> None:
    """Pie/donut chart per model showing phase proportion."""
    st.subheader("Tỷ trọng thời gian theo giai đoạn")

    devices = sorted(df["device"].unique())
    sel_dev = st.radio("Thiết bị:", devices, index=devices.index("colab_t4") if "colab_t4" in devices else 0, horizontal=True, key="tn7_dev_pie")
    models = sorted(df[df["device"] == sel_dev]["model"].unique())
    if not models:
        st.info(f"Không có dữ liệu TN7 cho {sel_dev}.")
        return

    cols = st.columns(min(len(models), 3))
    for idx, model in enumerate(models[:3]):
        sub = df[(df["device"] == sel_dev) & (df["model"] == model)]
        if sub.empty:
            continue
        total = sub["mean_ms"].sum()
        with cols[idx]:
            fig = go.Figure(go.Pie(
                labels=sub["phase"].tolist(),
                values=sub["mean_ms"].tolist(),
                hole=0.45,
                marker_colors=[_PHASE_COLORS.get(p, "#94A3B8") for p in sub["phase"]],
                textinfo="percent", textposition="inside",
                hovertemplate="%{label}<br>%{value:.2f} ms<br>%{percent}<extra></extra>",
            ))
            fig.update_layout(
                title=f"{_MODEL_LABELS.get(model, model)}<br><sup>Tổng: {total:.1f} ms</sup>",
                height=360, uniformtext=dict(minsize=11, mode="hide"),
                margin=dict(t=65, b=65, l=15, r=15),
                legend=dict(orientation="h", y=-.08, x=0, font=dict(size=10)),
                template="plotly_white",
            )
            show_chart(fig)


def _render_overhead_bar(df: pd.DataFrame) -> None:
    """Grouped bar: overhead% across models and devices."""
    st.subheader("Overhead ngoài mô hình")
    df_pct = _compute_overhead_pct(df)
    if df_pct.empty:
        return

    fig = px.bar(
        df_pct.sort_values("overhead_pct", ascending=False),
        x="model_label", y="overhead_pct", color="device",
        barmode="group",
        text="overhead_pct",
        title="Tiền xử lý và hậu xử lý trong tổng thời gian",
        labels={"overhead_pct": "Overhead (%)", "model_label": "Mô hình", "device": "Thiết bị"},
        color_discrete_map=DEVICE_COLORS,
        template="plotly_white",
    )
    fig.add_hline(y=30, line_dash="dot", line_color="#F59E0B",
                  annotation_text="Mốc 30% (CH6)", annotation_position="top right")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(height=400, yaxis=dict(range=[0, 110]))
    show_chart(fig)


def _render_stacked_absolute(df: pd.DataFrame) -> None:
    """Stacked absolute ms bar: model × device."""
    st.subheader("Thời gian mỗi giai đoạn")

    devices = sorted(df["device"].unique())
    sel_dev = st.radio("Thiết bị:", devices, index=devices.index("colab_t4") if "colab_t4" in devices else 0, horizontal=True, key="tn7_dev_abs")
    sub = df[df["device"] == sel_dev]
    if sub.empty:
        st.info(f"Không có dữ liệu cho {sel_dev}.")
        return

    pivot = sub.pivot_table(
        index="model_label", columns="phase", values="mean_ms", aggfunc="mean"
    ).fillna(0)

    fig = go.Figure()
    for phase in pivot.columns:
        color = _PHASE_COLORS.get(phase, "#94A3B8")
        fig.add_trace(go.Bar(
            name=phase, x=pivot.index.tolist(), y=pivot[phase].tolist(),
            marker_color=color,
            text=[f"{v:.2f}" for v in pivot[phase]],
            textposition="inside",
        ))
    fig.update_layout(
        barmode="stack",
        title=f"Thời gian theo giai đoạn · {sel_dev}",
        xaxis_title="Mô hình",
        yaxis_title="Thời gian trung bình (ms)",
        template="plotly_white",
        legend_title="Giai đoạn",
        height=420,
    )
    show_chart(fig)


def _render_cpu_vs_gpu_overhead(df: pd.DataFrame) -> None:
    """Side-by-side comparison: how overhead % differs CPU vs GPU."""
    st.subheader("Overhead trên CPU và T4")
    df_pct = _compute_overhead_pct(df)
    if df_pct.empty or df_pct["device"].nunique() < 2:
        st.info("Cần dữ liệu từ ít nhất 2 thiết bị để so sánh.")
        return

    fig = px.bar(
        df_pct.sort_values(["model_label", "device"]),
        x="model_label", y="overhead_pct", color="device",
        barmode="group",
        facet_col=None,
        text="overhead_pct",
        title="Overhead · CPU và T4",
        labels={"overhead_pct": "Overhead (%)", "model_label": "Mô hình", "device": "Thiết bị"},
        color_discrete_map=DEVICE_COLORS,
        template="plotly_white",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.add_hline(y=30, line_dash="dot", line_color="gray",
                  annotation_text="30%")
    fig.update_layout(height=400, yaxis=dict(range=[0, 110]))
    show_chart(fig)


def _render_raw_table(df: pd.DataFrame) -> None:
    """Raw mean/P50/P95 table for all phases."""
    disp = df[["device", "model_label", "phase", "mean_ms", "p50_ms", "p95_ms", "n_iter"]].copy()
    disp.columns = ["Thiết bị", "Mô hình", "Giai đoạn", "Trung bình (ms)", "P50 (ms)", "P95 (ms)", "Số iteration"]
    st.dataframe(disp.sort_values(["Thiết bị", "Mô hình", "Giai đoạn"]), width="stretch", hide_index=True)


def render(results_dir: Path) -> None:
    """Render Tab 6: TN7 Real-World Pipeline Phase Breakdown — answers CH6."""
    if not HAS_DEPS:
        st.error("Thiếu thư viện: pandas, plotly, streamlit.")
        return

    page_header("Pipeline TN7", "phase")
    df = _load_tn7_phases(results_dir)
    if df.empty:
        st.info(
            "Chưa có dữ liệu TN7. Chạy: "
            "`python scripts/run_tn7_pipeline.py --device colab_t4`"
        )
        return

    df_pct = _compute_overhead_pct(df)
    if not df_pct.empty:
        yolo_t4 = df_pct[(df_pct["model"] == "yolov8n") & (df_pct["device"] == "colab_t4")]
        bert_t4 = df_pct[(df_pct["model"] == "bert_base") & (df_pct["device"] == "colab_t4")]
        xgb_t4  = df_pct[(df_pct["model"] == "xgboost")  & (df_pct["device"] == "colab_t4")]
        c1, c2, c3 = st.columns(3)
        c1.metric("Overhead YOLO · T4",
                  f"{yolo_t4['overhead_pct'].values[0]:.1f}%" if not yolo_t4.empty else "—")
        c2.metric("Overhead BERT · T4",
                  f"{bert_t4['overhead_pct'].values[0]:.1f}%" if not bert_t4.empty else "—")
        c3.metric("Overhead XGBoost · T4",
                  f"{xgb_t4['overhead_pct'].values[0]:.1f}%" if not xgb_t4.empty else "—")

    tab_pie, tab_bar, tab_abs, tab_cmp = st.tabs([
        "Tỷ trọng (%)",
        "Overhead % Bar",
        "Thời gian (ms)",
        "CPU và T4",
    ])
    with tab_pie:
        _render_pie_breakdown(df)
    with tab_bar:
        _render_overhead_bar(df)
    with tab_abs:
        _render_stacked_absolute(df)
    with tab_cmp:
        _render_cpu_vs_gpu_overhead(df)
    with st.expander("Xem bảng số · giai đoạn pipeline"):
        _render_raw_table(df)
