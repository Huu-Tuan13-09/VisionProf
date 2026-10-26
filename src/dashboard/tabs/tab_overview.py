"""Tab 1: Overview and macro comparison across 8 models [Owned by TV2]."""

from pathlib import Path
from typing import Any, Dict, List

from src.dashboard.utils import page_header, show_chart

try:
    import pandas as pd
    import plotly.express as px
    import streamlit as st
    from src.dashboard.utils import load_all_run_summaries, make_bar_chart
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

FAMILY_MAP: Dict[str, str] = {
    "mobilenet_v3_large": "CV",
    "resnet50": "CV",
    "vit_b_16": "CV",
    "yolov8n": "CV",
    "distilbert": "NLP",
    "bert_base": "NLP",
    "mlp": "Tabular",
    "xgboost": "Tabular",
}


def render(results_dir: Path) -> None:
    """Render Tab 1: Comprehensive overview and macro metrics for 8 models."""
    if not HAS_DEPS:
        return

    page_header("Tổng quan mô hình")

    summaries = load_all_run_summaries(results_dir)
    if not summaries:
        st.info("Chưa có dữ liệu trong `results/`. Hãy chạy `scripts/run_experiment.py` trước.")
        return

    df = pd.DataFrame(summaries)
    df["family"] = df["model"].map(lambda m: FAMILY_MAP.get(m, "Other"))

    # Interactive Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        devices = sorted(list(df["device"].dropna().unique()))
        selected_devices = st.multiselect("Thiết bị", options=devices, default=devices)
    with col_f2:
        families = ["CV", "NLP", "Tabular"]
        selected_families = st.multiselect("Nhóm mô hình", options=families, default=families)
    with col_f3:
        experiments = sorted(list(df["experiment"].dropna().unique()))
        selected_exps = st.multiselect("Thực nghiệm", options=experiments, default=["TN3"] if "TN3" in experiments else experiments)

    filtered_df = df[
        df["device"].isin(selected_devices)
        & df["family"].isin(selected_families)
        & df["experiment"].isin(selected_exps)
    ]

    if filtered_df.empty:
        st.warning("Không có dữ liệu phù hợp với bộ lọc hiện tại.")
        return

    # Metric Table Display
    columns_12 = [
        "device", "model", "family", "config",
        "p50_ms", "mean_ms", "std_ms", "p90_ms", "p99_ms",
        "tail_ratio", "throughput_fps", "params_m",
        "peak_vram_mb", "rss_mb", "cold_start_ms",
    ]
    avail_cols = [c for c in columns_12 if c in filtered_df.columns]
    k1, k2, k3 = st.columns(3)
    k1.metric("Mô hình", filtered_df["model"].nunique())
    k2.metric("Thiết bị", filtered_df["device"].nunique())
    k3.metric("Lượt đo", len(filtered_df))
    with st.expander("Xem bảng số", expanded=False):
        st.dataframe(filtered_df[avail_cols].sort_values(by=["device", "p50_ms"]),
                     width="stretch", hide_index=True, column_config={
                         "device": "Thiết bị", "model": "Mô hình", "family": "Nhóm", "config": "Cấu hình",
                         "p50_ms": "P50 (ms)", "mean_ms": "Trung bình (ms)", "std_ms": "Độ lệch chuẩn (ms)",
                         "p90_ms": "P90 (ms)", "p99_ms": "P99 (ms)", "tail_ratio": "P99 / P50",
                         "throughput_fps": "Thông lượng (FPS)", "params_m": "Tham số (triệu)",
                         "peak_vram_mb": "Peak VRAM (MB)", "rss_mb": "Process RSS (MB)", "cold_start_ms": "Cold-start (ms)",
                     })


    # Macro Visualizations
    st.subheader("Tham số và độ trễ")
    if "params_m" in filtered_df.columns and "p50_ms" in filtered_df.columns:
        fig_scatter = px.scatter(
            filtered_df,
            x="params_m",
            y="p50_ms",
            color="device",
            size=filtered_df["throughput_fps"].clip(lower=1.0),
            hover_name="model",
            hover_data=["family", "throughput_fps", "tail_ratio", "peak_vram_mb"],
            log_x=True,
            log_y=True,
            title="Tham số và P50 · kích thước điểm theo FPS",
            labels={"params_m": "Tham số (triệu, thang log)", "p50_ms": "P50 (ms, thang log)",
                    "device": "Thiết bị", "family": "Nhóm", "throughput_fps": "Thông lượng (FPS)",
                    "tail_ratio": "P99 / P50", "peak_vram_mb": "Peak VRAM (MB)"},
            template="plotly_white",
        )
        show_chart(fig_scatter)

    c1, c2 = st.columns(2)
    with c1:
        if "p50_ms" in filtered_df.columns:
            fig_lat = make_bar_chart(filtered_df, x_col="model", y_col="p50_ms", color_col="device", title="Độ trễ P50 (ms)")
            if fig_lat:
                show_chart(fig_lat)
    with c2:
        if "throughput_fps" in filtered_df.columns:
            fig_fps = make_bar_chart(filtered_df, x_col="model", y_col="throughput_fps", color_col="device", title="Thông lượng (FPS)")
            if fig_fps:
                show_chart(fig_fps)
