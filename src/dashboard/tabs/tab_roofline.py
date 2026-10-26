"""Tab 4: Roofline model efficiency analysis [Owned by TV1]."""

import json
from pathlib import Path

from src.dashboard.layer_charts import make_roofline_chart
from src.dashboard.utils import page_header, show_chart

try:
    import pandas as pd
    import streamlit as st
    from src.layer.hw_specs import calculate_roofline_efficiency
except ImportError:
    pass


def render(results_dir: Path) -> None:
    """Render Tab 4: Roofline hardware efficiency evaluation."""
    page_header("Roofline", "layer")
    st.caption("Hiệu suất tính toán so với đỉnh phần cứng.")

    layer_files = sorted(results_dir.glob("**/layers.csv"))
    if not layer_files:
        st.info("Cần dữ liệu `layers.csv` để tính toán Roofline efficiency.")
        return

    preferred = results_dir / "colab_t4/resnet50/TN1/bs1_fp32/layers.csv"
    default_index = layer_files.index(preferred) if preferred in layer_files else 0
    selected_file = st.selectbox("Chọn lượt đo", layer_files, index=default_index,
                                 format_func=lambda p: str(p.parent.relative_to(results_dir)))
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
    col1.metric("GFLOPs / iteration", f"{avg_flops:.2f} GFLOPs")
    col2.metric("Thời gian layer / pass", f"{avg_time_ms:.2f} ms")
    col3.metric("Hiệu suất so với đỉnh", f"{eff_pct:.2f}%")

    summary_path = selected_file.with_name("layers_summary.csv")
    meta_path = selected_file.with_name("layer_meta.json")
    if summary_path.exists() and meta_path.exists():
        try:
            summary = pd.read_csv(summary_path)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            st.warning(f"Không đọc được dữ liệu đồ thị Roofline: {exc}")
            return
        if meta.get("hardware"):
            st.subheader("Layer trên đường Roofline")
            st.caption("Mỗi điểm là một layer. Di chuột để xem FLOPs, thời gian và hiệu suất; "
                       "đường trần biểu diễn giới hạn phần cứng.")
            fig = make_roofline_chart(summary, meta["hardware"])
            if fig is not None:
                show_chart(fig, key="roofline_detail")
            st.caption("Thẻ số dùng trung bình mỗi pass; đồ thị dùng bảng layers_summary.csv đã lưu.")
    else:
        st.info("Lượt đo này chưa có layers_summary.csv hoặc layer_meta.json để vẽ đường Roofline.")
