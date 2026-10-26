"""Đọc kết quả và thống nhất cách trình bày dashboard."""

import json
import logging
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

logger = logging.getLogger(__name__)
LAYER_COLOR = "#B4480F"
MODEL_COLOR = "#2456A6"
PHASE_COLOR = "#1E7A4C"
CHART_COLORS = [MODEL_COLOR, LAYER_COLOR, PHASE_COLOR, "#7757A6", "#AB3270", "#54717C", "#9C740D"]
DEVICE_COLORS = {"cpu_laptop": "#64748B", "colab_cpu": LAYER_COLOR, "colab_t4": MODEL_COLOR}
PHASE_COLORS = {
    "forward": MODEL_COLOR, "model_forward": MODEL_COLOR, "predict": MODEL_COLOR,
    "backward": LAYER_COLOR, "optimizer": PHASE_COLOR,
    "dataloader": "#9C740D", "data_load": "#9C740D", "fit": "#7757A6",
    "preprocess": LAYER_COLOR, "tokenize": LAYER_COLOR, "csv_parse": LAYER_COLOR,
    "postprocess_nms": "#AB3270", "decode_pooling": PHASE_COLOR,
}
CHART_LABELS = {
    "model": "Mô hình", "device": "Thiết bị", "model_label": "Mô hình",
    "p50_ms": "P50 (ms)", "throughput_fps": "Thông lượng (FPS)",
    "latency_ms": "Độ trễ (ms)", "Variant": "Cấu hình",
}


def configure_dashboard() -> None:
    """Áp dụng khoảng cách, thẻ số chính và bảng màu chung."""
    px.defaults.color_discrete_sequence = CHART_COLORS
    px.defaults.color_discrete_map = DEVICE_COLORS
    st.markdown("""
<style>
.block-container {max-width: 1440px; padding-top: 4rem; padding-bottom: 2rem;}
[data-testid="stSidebar"] {border-right: 1px solid #DCE3EB;}
[data-testid="stSidebar"] .block-container {padding-top: 2rem;}
[data-testid="stMetric"] {border: 1px solid #DCE3EB; border-radius: 12px;
    padding: 16px 18px; background: var(--secondary-background-color, #F3F6FA);}
[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums; font-size: 1.65rem;}
[data-testid="stMetricLabel"] {font-size: .85rem;}
h1 {font-size: 2rem !important; letter-spacing: -.035em;}
h2 {font-size: 1.35rem !important;}
h3 {font-size: 1.1rem !important;}
[data-testid="stTabs"] [role="tab"] {padding-left: 12px; padding-right: 12px;}
.vp-level {font-size: .78rem; font-weight: 650; letter-spacing: .06em;
    text-transform: uppercase; margin-bottom: .25rem;}
</style>
""", unsafe_allow_html=True)


def page_header(title: str, level: str = "model") -> None:
    """Hiển thị tiêu đề ngắn cùng màu nhận diện mức đo."""
    labels = {"layer": ("Mức layer", LAYER_COLOR), "model": ("Mức model", MODEL_COLOR),
              "phase": ("Mức giai đoạn", PHASE_COLOR)}
    label, color = labels[level]
    st.markdown(f'<div class="vp-level" style="color:{color}">{escape(label)}</div>', unsafe_allow_html=True)
    st.title(title)


def show_chart(fig: Any, key: Optional[str] = None) -> None:
    """Hiển thị biểu đồ rõ chữ trên nền sáng, giữ nguyên dữ liệu."""
    fig.update_layout(
        template="plotly_white", font=dict(family="Arial, sans-serif", size=13, color="#253449"),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        hoverlabel=dict(bgcolor="#FFFFFF", font_color="#253449"),
        title_font_size=16,
    )
    st.plotly_chart(fig, width="stretch", key=key, theme=None, config={"displaylogo": False})


def load_all_run_summaries(results_dir: Path) -> List[Dict[str, Any]]:
    """Đọc summary kèm thông tin thiết bị, mô hình và cấu hình từ đường dẫn."""
    summaries: List[Dict[str, Any]] = []
    if not results_dir.exists():
        return summaries
    for summary_path in results_dir.glob("**/summary.json"):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            rel_parts = summary_path.relative_to(results_dir).parts
            if len(rel_parts) >= 4:
                data["device"], data["model"], data["experiment"], data["config"] = rel_parts[:4]
            data["path"] = str(summary_path.parent)
            summaries.append(data)
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Không đọc được summary %s: %s", summary_path, exc)
    return summaries


def make_bar_chart(
    df: Any, x_col: str, y_col: str, color_col: Optional[str] = None, title: str = "",
) -> Any:
    """Tạo biểu đồ thanh với màu thiết bị nhất quán."""
    fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=title, barmode="group",
                 labels=CHART_LABELS, color_discrete_map=DEVICE_COLORS)
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig


def make_box_plot(df: Any, x_col: str, y_col: str, title: str = "") -> Any:
    """Tạo biểu đồ phân phối độ trễ, giữ đầy đủ các điểm đo."""
    fig = px.box(df, x=x_col, y=y_col, title=title, points="all", labels=CHART_LABELS)
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig
