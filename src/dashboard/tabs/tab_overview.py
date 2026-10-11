"""Tab 1: Overview and macro comparison across 8 models [Owned by TV2]."""

from pathlib import Path
from typing import Any, Dict, List

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

    st.header("📊 Model Overview & Macro Comparison")
    st.markdown(
        "Khảo sát toàn cảnh 12 chỉ số hiệu năng (Latency P50..P99, Thông lượng FPS, Peak VRAM, Process RSS, Đỉnh khởi động Cold-start) "
        "giữa 8 mô hình đa lĩnh vực (CV, NLP, Tabular) trên CPU Laptop và Google Colab Tesla T4."
    )

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
        selected_devices = st.multiselect("💻 Thiết bị (Hardware)", options=devices, default=devices)
    with col_f2:
        families = ["CV", "NLP", "Tabular"]
        selected_families = st.multiselect("🧠 Nhóm mô hình (Family)", options=families, default=families)
    with col_f3:
        experiments = sorted(list(df["experiment"].dropna().unique()))
        selected_exps = st.multiselect("🔬 Thực nghiệm (Experiment)", options=experiments, default=["TN3"] if "TN3" in experiments else experiments)

    filtered_df = df[
        df["device"].isin(selected_devices)
        & df["family"].isin(selected_families)
        & df["experiment"].isin(selected_exps)
    ]

    if filtered_df.empty:
        st.warning("Không có dữ liệu phù hợp với bộ lọc hiện tại.")
        return

    # Metric Table Display
    st.subheader("📋 Bảng Tổng Hợp 12 Chỉ Số Hiệu Năng")
    columns_12 = [
        "device", "model", "family", "config",
        "p50_ms", "mean_ms", "std_ms", "p90_ms", "p99_ms",
        "tail_ratio", "throughput_fps", "params_m",
        "peak_vram_mb", "rss_mb", "cold_start_ms",
    ]
    avail_cols = [c for c in columns_12 if c in filtered_df.columns]
    st.dataframe(filtered_df[avail_cols].sort_values(by=["device", "p50_ms"]), use_container_width=True)

    # Macro Visualizations
    st.subheader("📈 Phân Tích Đánh Đổi: Tham Số vs Độ Trễ (CH3)")
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
            title="Đánh đổi: Số tham số (M) vs Latency P50 (ms) [Kích thước điểm: FPS]",
            labels={"params_m": "Tham số (Triệu - Log scale)", "p50_ms": "P50 Latency (ms - Log scale)"},
            template="plotly_white",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        if "p50_ms" in filtered_df.columns:
            fig_lat = make_bar_chart(filtered_df, x_col="model", y_col="p50_ms", color_col="device", title="Median Latency P50 (ms)")
            if fig_lat:
                st.plotly_chart(fig_lat, use_container_width=True)
    with c2:
        if "throughput_fps" in filtered_df.columns:
            fig_fps = make_bar_chart(filtered_df, x_col="model", y_col="throughput_fps", color_col="device", title="Thông lượng Throughput (FPS)")
            if fig_fps:
                st.plotly_chart(fig_fps, use_container_width=True)
