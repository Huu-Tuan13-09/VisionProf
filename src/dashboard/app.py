"""Điểm vào dashboard Streamlit của VisionProf."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.dashboard.tabs import (
    render_ab_test,
    render_cpu_vs_gpu,
    render_diagnostics,
    render_layer,
    render_overview,
    render_pipeline,
    render_roofline,
    render_training,
)
from src.dashboard.utils import configure_dashboard

PAGES = {
    "1. Tổng quan": render_overview,
    "2. CPU và T4": render_cpu_vs_gpu,
    "3. Phân tích layer": render_layer,
    "4. Roofline": render_roofline,
    "5. Training TN6": render_training,
    "6. Pipeline TN7": render_pipeline,
    "7. Chẩn đoán": render_diagnostics,
    "8. So sánh A/B": render_ab_test,
}


def main() -> None:
    """Hiển thị thanh điều hướng và mục đang chọn."""
    st.set_page_config(page_title="VisionProf", layout="wide", page_icon="📊")
    configure_dashboard()
    st.sidebar.title("VisionProf")
    st.sidebar.caption("Hiệu năng mô hình · Từ tổng quan đến nút thắt")
    choice = st.sidebar.radio("Nội dung", list(PAGES), key="dashboard_page")
    PAGES[choice](PROJECT_ROOT / "results")


if __name__ == "__main__":
    main()
