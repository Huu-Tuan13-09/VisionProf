"""Tab 2: CPU vs GPU comparative performance [Owned by TV2]."""

from pathlib import Path
from typing import Any

try:
    import pandas as pd
    import streamlit as st
    from src.dashboard.utils import load_all_run_summaries, make_bar_chart
except ImportError:
    pass


def render(results_dir: Path) -> None:
    """Render Tab 2: Colab CPU vs Colab T4 GPU comparisons."""
    st.header("⚡ Colab CPU vs Colab T4 GPU Acceleration")
    st.markdown("Đo lường hệ số gia tốc phần cứng (Speedup), thông lượng và mức tiêu thụ tài nguyên.")

    summaries = load_all_run_summaries(results_dir)
    if not summaries:
        st.info("Chưa tìm thấy dữ liệu để so sánh CPU vs GPU.")
        return

    df = pd.DataFrame(summaries)
    if "device" not in df.columns or "model" not in df.columns:
        st.warning("Dữ liệu thiếu trường device hoặc model.")
        return

    # Pivot to compare P50 across devices
    try:
        pivot_df = df.pivot_table(index="model", columns="device", values="p50_ms", aggfunc="mean").reset_index()
        if "colab_cpu" in pivot_df.columns and "colab_t4" in pivot_df.columns:
            pivot_df["Speedup_GPU_vs_CPU"] = (pivot_df["colab_cpu"] / pivot_df["colab_t4"]).round(2)
        st.subheader("Hệ Số Tăng Tốc (Speedup Multiplier)")
        st.dataframe(pivot_df, use_container_width=True)
    except Exception as exc:
        st.error(f"Lỗi tính toán pivot table: {exc}")
