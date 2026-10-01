"""Tab 5: Training pipeline and stage breakdown [Owned by TV3]."""

from pathlib import Path
from typing import Any

try:
    import pandas as pd
    import streamlit as st
    from src.dashboard.utils import make_bar_chart
except ImportError:
    pass


def render(results_dir: Path) -> None:
    """Render Tab 5: Training step breakdown and E2E pipeline analysis."""
    st.header("🟩 Training Step & Pipeline Phase Breakdown")
    st.markdown("Phân bổ thời gian giữa Forward pass, Backward pass, Optimizer step, và DataLoader.")

    phase_files = list(results_dir.glob("**/phases.csv"))
    if not phase_files:
        st.info("Chưa có file `phases.csv`. Hãy chạy thực nghiệm với `--profilers phase`.")
        return

    if not selected_file or selected_file.stat().st_size == 0:
        st.warning("File phases.csv rỗng.")
        return

    try:
        df = pd.read_csv(selected_file)
    except Exception:
        st.error("Không thể đọc file phases.csv.")
        return

    if df.empty or "phase" not in df.columns:
        st.warning("Dữ liệu phase rỗng hoặc không có cột phase.")
        return

    st.subheader("Dữ Liệu Giai Đoạn (Phases)")
    phase_summary = df.groupby("phase")["time_ms"].mean().reset_index()

    fig_phases = make_bar_chart(phase_summary, x_col="phase", y_col="time_ms", title="Thời Gian Trung Bình Theo Giai Đoạn (ms)")
    if fig_phases:
        st.plotly_chart(fig_phases, use_container_width=True)
