"""Tab 3: Leaf layer breakdown and latency hotspots [Owned by TV1]."""

from pathlib import Path
from typing import Any

try:
    import pandas as pd
    import streamlit as st
    from src.dashboard.utils import make_bar_chart
except ImportError:
    pass


def render(results_dir: Path) -> None:
    """Render Tab 3: Detailed breakdown of individual leaf layers."""
    st.header("🟧 Leaf Layer Breakdown & Hotspot Identification")
    st.markdown("Phân tích thời gian thực thi, activation memory và FLOPs của từng layer lá.")

    layer_files = list(results_dir.glob("**/layers.csv"))
    if not layer_files:
        st.info("Chưa có file `layers.csv`. Hãy chạy thực nghiệm với `--profilers layer`.")
        return

    if not selected_file or selected_file.stat().st_size == 0:
        st.warning("File layers.csv rỗng.")
        return

    try:
        df = pd.read_csv(selected_file)
    except Exception:
        st.error("Không thể đọc file layers.csv.")
        return

    if df.empty or "layer_name" not in df.columns:
        st.warning("Dữ liệu layer rỗng hoặc không có cột layer_name.")
        return

    st.subheader(f"Dữ Liệu Layer Profiling ({len(df)} bản ghi)")
    st.dataframe(df.head(20), use_container_width=True)

    # Top slow layers
    top_layers = df.groupby("layer_name")["time_ms"].mean().reset_index().sort_values(by="time_ms", ascending=False).head(10)
    fig_top = make_bar_chart(top_layers, x_col="layer_name", y_col="time_ms", title="Top 10 Layer Chậm Nhất (ms)")
    if fig_top:
        st.plotly_chart(fig_top, use_container_width=True)
