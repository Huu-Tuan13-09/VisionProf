"""Đọc kết quả và thống nhất cách trình bày dashboard."""

import json
import logging
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from src.dashboard.styles import apply_styles

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
    template = go.layout.Template(pio.templates["plotly_white"])
    template.layout.update(
        font=dict(family="Arial, sans-serif", size=13, color="#253449"),
        colorway=CHART_COLORS, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        hoverlabel=dict(bgcolor="#FFFFFF", font_color="#253449"),
        title=dict(font=dict(size=16), x=.02, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0,
                    title_text="", font=dict(size=12), tracegroupgap=8),
        xaxis=dict(gridcolor="#EDF1F6", linecolor="#DCE3ED", zeroline=False,
                   automargin=True, title_standoff=16),
        yaxis=dict(gridcolor="#EDF1F6", linecolor="#DCE3ED", zeroline=False,
                   automargin=True, title_standoff=12),
    )
    pio.templates["visionprof"] = template
    px.defaults.template = "visionprof"
    apply_styles()


def page_header(title: str, level: str = "model") -> None:
    """Hiển thị tiêu đề ngắn cùng màu nhận diện mức đo."""
    labels = {"layer": ("Mức layer", LAYER_COLOR), "model": ("Mức model", MODEL_COLOR),
              "phase": ("Mức giai đoạn", PHASE_COLOR)}
    label, color = labels[level]
    descriptions = {
        "Tổng quan mô hình": "Đọc nhanh P50, thông lượng và quy mô mô hình trên từng thiết bị.",
        "CPU và T4": "So sánh phần cứng, batch size và FP16 từ các lượt đo đã lưu.",
        "Phân tích layer": "Xác định layer chiếm thời gian và đối chiếu CPU với GPU T4.",
        "Training TN6": "Theo dõi thời gian training, VRAM và chi phí nạp dữ liệu.",
        "Pipeline TN7": "Tách thời gian mô hình khỏi tiền xử lý và hậu xử lý.",
        "Chẩn đoán": "Đọc kết quả 19 luật và tìm điểm cần tối ưu cho từng lượt đo.",
    }
    st.markdown(f'<style>:root {{ --vp-accent: {color}; }}</style>'
                f'<div class="vp-level" style="color:{color}">{escape(label)}</div>',
                unsafe_allow_html=True)
    st.title(title)
    if title in descriptions:
        st.caption(descriptions[title])


def show_chart(fig: Any, key: Optional[str] = None) -> None:
    """Hiển thị biểu đồ rõ chữ trên nền sáng, giữ nguyên dữ liệu."""
    fig.update_layout(template="visionprof", font=dict(family="Arial, sans-serif", size=13),
                      paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF")
    st.plotly_chart(fig, width="stretch", key=key, theme=None,
                    config={"displaylogo": False, "displayModeBar": "hover",
                            "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


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
