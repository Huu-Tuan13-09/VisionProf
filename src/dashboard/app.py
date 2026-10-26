"""Điểm vào và điều hướng gốc Streamlit của VisionProf."""

import sys
from functools import partial
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.dashboard.tabs import (
    render_ab_test, render_cpu_vs_gpu, render_diagnostics, render_layer,
    render_overview, render_pipeline, render_roofline, render_training,
)
from src.dashboard.utils import configure_dashboard

PAGE_GROUPS = {
    "Mức Model": [
        ("1. Tổng quan", "dashboard", "overview", render_overview),
        ("2. CPU và T4", "speed", "cpu-t4", render_cpu_vs_gpu),
    ],
    "Mức Layer": [
        ("3. Phân tích layer", "layers", "layers", render_layer),
        ("4. Roofline", "show_chart", "roofline", render_roofline),
    ],
    "Mức Giai đoạn": [
        ("5. Training TN6", "model_training", "training", render_training),
        ("6. Pipeline TN7", "account_tree", "pipeline", render_pipeline),
    ],
    "Chẩn đoán": [
        ("7. Chẩn đoán", "troubleshoot", "diagnostics", render_diagnostics),
        ("8. So sánh A/B", "compare_arrows", "comparison", render_ab_test),
    ],
}


def main() -> None:
    """Hiển thị tám trang theo mức đo, giữ thứ tự dùng trong slide."""
    st.set_page_config(page_title="VisionProf", layout="wide", page_icon=":material/monitoring:")
    configure_dashboard()
    st.logo(str(Path(__file__).with_name("brand.svg")), size="large")
    pages = {
        group: [st.Page(partial(render, PROJECT_ROOT / "results"), title=title,
                        icon=f":material/{icon}:", url_path=url, default=url == "overview")
                for title, icon, url, render in items]
        for group, items in PAGE_GROUPS.items()
    }
    page = st.navigation(pages, expanded=True)
    st.sidebar.markdown('''<div class="vp-sidebar-note">PHÂN TÍCH HIỆU NĂNG<br>
Từ tổng quan đến nút thắt
<div class="vp-legend"><span style="border-color:#2456A6">Model</span>
<span style="border-color:#B4480F">Layer</span>
<span style="border-color:#1E7A4C">Giai đoạn</span></div></div>''', unsafe_allow_html=True)
    page.run()


if __name__ == "__main__":
    main()
