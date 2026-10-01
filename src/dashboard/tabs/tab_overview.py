"""Tab 1: Overview and macro comparison across 8 models [Owned by TV2]."""

from pathlib import Path
from typing import Any

try:
    import pandas as pd
    import streamlit as st
    from src.dashboard.utils import load_all_run_summaries, make_bar_chart
except ImportError:
    pass


def render(results_dir: Path) -> None:
    """Render Tab 1: Overview of 8 benchmarked models."""
    st.header("📊 Model Overview & Macro Comparison")
    st.markdown("So sánh hiệu năng vĩ mô (P50 Latency, Throughput FPS, Peak VRAM) giữa 8 mô hình đa lĩnh vực.")

    summaries = load_all_run_summaries(results_dir)
    if not summaries:
        st.info("Chưa có dữ liệu trong `results/`. Hãy chạy `scripts/run_experiment.py` trước.")
        return

    df = pd.DataFrame(summaries)
    st.subheader("Bảng Tổng Hợp Chỉ Số")
    display_cols = [c for c in ["device", "model", "experiment", "config", "p50_ms", "throughput_fps", "peak_vram_mb"] if c in df.columns]
    st.dataframe(df[display_cols], use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if "p50_ms" in df.columns and "model" in df.columns:
            fig_lat = make_bar_chart(df, x_col="model", y_col="p50_ms", color_col="device", title="Median Latency P50 (ms)")
            if fig_lat:
                st.plotly_chart(fig_lat, use_container_width=True)

    with col2:
        if "throughput_fps" in df.columns and "model" in df.columns:
            fig_fps = make_bar_chart(df, x_col="model", y_col="throughput_fps", color_col="device", title="Throughput (FPS)")
            if fig_fps:
                st.plotly_chart(fig_fps, use_container_width=True)
