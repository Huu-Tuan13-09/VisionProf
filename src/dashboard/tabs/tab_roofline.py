"""Tab 4: Roofline model efficiency analysis [Owned by TV1]."""

from pathlib import Path
from typing import Any

try:
    import pandas as pd
    import streamlit as st
    from src.layer.hw_specs import calculate_roofline_efficiency, get_hardware_specs
except ImportError:
    pass


def render(results_dir: Path) -> None:
    """Render Tab 4: Roofline hardware efficiency evaluation."""
    st.header("📈 Roofline Model & Hardware Peak Efficiency")
    st.markdown("Đánh giá mức độ tiệm cận công suất tính toán đỉnh (% Peak TFLOPs) của Colab T4 và Xeon CPU.")

    layer_files = list(results_dir.glob("**/layers.csv"))
    if not layer_files:
        st.info("Cần dữ liệu `layers.csv` để tính toán Roofline efficiency.")
        return

    if not selected_file or selected_file.stat().st_size == 0:
        st.warning("File layers.csv rỗng.")
        return

    try:
        df = pd.read_csv(selected_file)
    except Exception:
        st.error("Không thể đọc file layers.csv.")
        return

    if df.empty or "time_ms" not in df.columns:
        st.warning("Dữ liệu layer rỗng hoặc không có cột time_ms.")
        return

    num_iters = max(1, int(df["iteration"].nunique())) if "iteration" in df.columns else 1
    avg_flops = (df["flops"].sum() / num_iters) / 1e9 if "flops" in df.columns else 0.0
    avg_time_ms = df["time_ms"].sum() / num_iters

    path_str = str(selected_file)
    if "colab_t4" in path_str:
        device = "colab_t4"
    elif "cpu_laptop" in path_str:
        device = "cpu_laptop"
    else:
        device = "colab_cpu"

    eff_pct = calculate_roofline_efficiency(avg_flops, avg_time_ms, device)

    col1, col2, col3 = st.columns(3)
    col1.metric("GFLOPs / Iteration", f"{avg_flops:.2f} GFLOPs")
    col2.metric("Thời Gian Layers / Pass", f"{avg_time_ms:.2f} ms")
    col3.metric("Hiệu Suất vs Peak", f"{eff_pct:.2f}%")
