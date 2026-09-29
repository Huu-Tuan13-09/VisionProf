"""
dashboard.py — Giao diện Streamlit + Plotly cho hệ thống profiling
====================================================================
Chạy bằng lệnh:
    streamlit run src/dashboard.py

Tính năng:
    Tab 1 - Tổng quan     : So sánh nhiều mô hình (bảng + bar chart)
    Tab 2 - Layer Analysis : Top-K slowest, memory heatmap theo iteration
    Tab 3 - Rule Catalog   : Kết quả 19 quy tắc R-A→R-D, severity pie
    Tab 4 - A/B Comparison : Waterfall chart delta, scatter speedup
    Tab 5 - DataLoader     : Load vs Compute timeline, bottleneck gauge
"""

from __future__ import annotations

# ── sys.path setup — chạy từ bất kỳ thư mục nào ────────────────────────────
import sys
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Standard imports ─────────────────────────────────────────────────────────
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── src imports ───────────────────────────────────────────────────────────────
from src.analyzer import (
    RuleCatalog,
    ABComparisonEngine,
    AnalysisReport,
    Severity,
    RuleCategory,
    analyze_from_csv,
    compare_ab_from_csv,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config — phải gọi đầu tiên
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "PyTorch Layer Profiler Dashboard",
    page_icon   = "⚡",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS tùy chỉnh
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Font chữ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .main-header h1 { color: #e94560; margin: 0; font-size: 1.8rem; font-weight: 700; }
    .main-header p  { color: #a0aec0; margin: 0.3rem 0 0 0; font-size: 0.9rem; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2a3a, #243447);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-val  { font-size: 1.8rem; font-weight: 700; color: #63b3ed; }
    .metric-label{ font-size: 0.8rem; color: #718096; margin-top: 4px; }

    /* Severity badges */
    .badge-critical { background:#e53e3e; color:white; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-weight:600; }
    .badge-high     { background:#dd6b20; color:white; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-weight:600; }
    .badge-medium   { background:#d69e2e; color:white; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-weight:600; }
    .badge-low      { background:#38a169; color:white; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-weight:600; }
    .badge-ok       { background:#3182ce; color:white; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-weight:600; }

    /* Divider */
    .section-divider { border-top: 1px solid rgba(255,255,255,0.08); margin: 1.5rem 0; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #0d1117; }
    [data-testid="stSidebar"] .stMarkdown { color: #8b949e; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants / Color palette
# ─────────────────────────────────────────────────────────────────────────────
SEVERITY_COLORS = {
    Severity.CRITICAL: "#e53e3e",
    Severity.HIGH:     "#dd6b20",
    Severity.MEDIUM:   "#d69e2e",
    Severity.LOW:      "#38a169",
    Severity.OK:       "#3182ce",
}
SEVERITY_ORDER = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.OK]

MODEL_PALETTE = px.colors.qualitative.Plotly

PLOTLY_DARK_TEMPLATE = "plotly_dark"

# ─────────────────────────────────────────────────────────────────────────────
# Cached loaders
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    """Nạp CSV với cache."""
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    """Nạp JSON với cache."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def run_rule_catalog(
    csv_path: str,
    model_name: str,
    device: str,
    total_gpu_mem_mb: float,
    dl_json_path: str,
) -> dict:
    """Chạy Rule Catalog và cache kết quả."""
    df = load_csv(csv_path)
    total_params = int(df.groupby("layer_name")["param_count"].first().sum()) if "param_count" in df.columns else 0

    dl_stats = None
    if dl_json_path and Path(dl_json_path).exists():
        raw = load_json(dl_json_path)
        dl_stats = raw.get("summary", {})
        batch_detail = raw.get("batch_detail", [])
        dl_stats["batch_load_times"] = [b.get("load_time_ms", 0) for b in batch_detail]

    catalog = RuleCatalog()
    report  = catalog.run_all(
        df=df,
        model_name=model_name,
        device=device,
        total_gpu_mem_mb=total_gpu_mem_mb,
        total_params=total_params,
        dataloader_stats=dl_stats,
    )
    return report.to_dict()


def _find_profile_dirs(base_dir: Path) -> list[Path]:
    """Tìm tất cả thư mục con có file layer_records.csv."""
    return sorted([
        d for d in base_dir.iterdir()
        if d.is_dir() and (d / "layer_records.csv").exists()
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────────────────────────────────────

def _chart_top_k_layers(df: pd.DataFrame, k: int = 15, color: str = "#63b3ed") -> go.Figure:
    """Bar chart: Top-K layers chậm nhất."""
    agg = (
        df.groupby("layer_name")["forward_time_ms"]
          .mean()
          .nlargest(k)
          .sort_values(ascending=True)
          .reset_index()
    )
    agg.columns = ["Layer", "Mean Time (ms)"]
    fig = px.bar(
        agg, x="Mean Time (ms)", y="Layer",
        orientation="h",
        template=PLOTLY_DARK_TEMPLATE,
        color="Mean Time (ms)",
        color_continuous_scale="Reds",
        title=f"Top-{k} Layers Chậm Nhất (ms)",
    )
    fig.update_layout(
        height=max(350, k * 30),
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def _chart_memory_by_layer(df: pd.DataFrame, k: int = 15) -> go.Figure:
    """Bar chart: Top-K layers tiêu thụ GPU memory."""
    if "gpu_mem_delta_mb" not in df.columns:
        return go.Figure().update_layout(title="Không có dữ liệu GPU memory")
    agg = (
        df.groupby("layer_name")["gpu_mem_delta_mb"]
          .mean()
          .abs()
          .nlargest(k)
          .sort_values(ascending=True)
          .reset_index()
    )
    agg.columns = ["Layer", "|Mem Delta| (MB)"]
    fig = px.bar(
        agg, x="|Mem Delta| (MB)", y="Layer",
        orientation="h",
        template=PLOTLY_DARK_TEMPLATE,
        color="|Mem Delta| (MB)",
        color_continuous_scale="Purples",
        title=f"Top-{k} Layers Tiêu Thụ Memory Nhiều Nhất",
    )
    fig.update_layout(
        height=max(350, k * 30),
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def _chart_memory_heatmap(df: pd.DataFrame, top_k: int = 20) -> go.Figure:
    """Heatmap: forward_time_ms theo (layer, iteration)."""
    if "iteration" not in df.columns or df["iteration"].nunique() < 2:
        return go.Figure().update_layout(title="Cần >= 2 iterations cho heatmap")

    top_layers = (
        df.groupby("layer_name")["forward_time_ms"]
          .mean()
          .nlargest(top_k)
          .index.tolist()
    )
    df_filt = df[df["layer_name"].isin(top_layers)]
    pivot   = df_filt.pivot_table(
        index="layer_name", columns="iteration", values="forward_time_ms", aggfunc="mean"
    ).fillna(0)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f"Iter {c}" for c in pivot.columns],
        y=pivot.index.tolist(),
        colorscale="YlOrRd",
        colorbar=dict(title="ms"),
    ))
    fig.update_layout(
        title=f"Heatmap Forward Time theo Iteration (Top-{top_k} Layers)",
        template=PLOTLY_DARK_TEMPLATE,
        height=max(400, top_k * 28),
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(title="Iteration"),
        yaxis=dict(title="Layer"),
    )
    return fig


def _chart_layer_type_pie(df: pd.DataFrame) -> go.Figure:
    """Pie chart: phân phối loại layer."""
    if "layer_type" not in df.columns:
        return go.Figure()
    counts = df.groupby("layer_type")["layer_name"].nunique().sort_values(ascending=False)
    fig = px.pie(
        values=counts.values,
        names=counts.index,
        title="Phân Phối Loại Layer",
        template=PLOTLY_DARK_TEMPLATE,
        color_discrete_sequence=px.colors.qualitative.Pastel,
        hole=0.4,
    )
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def _chart_scatter_time_vs_params(df: pd.DataFrame) -> go.Figure:
    """Scatter: forward_time_ms vs param_count."""
    if not {"forward_time_ms", "param_count", "layer_type"}.issubset(df.columns):
        return go.Figure()
    agg = df.groupby(["layer_name", "layer_type"]).agg(
        mean_time  = ("forward_time_ms", "mean"),
        param_count= ("param_count",     "first"),
    ).reset_index()
    agg = agg[agg["param_count"] > 0]
    fig = px.scatter(
        agg,
        x="param_count", y="mean_time",
        color="layer_type",
        hover_name="layer_name",
        log_x=True,
        size="param_count",
        size_max=30,
        labels={"param_count": "Params (log)", "mean_time": "Mean Time (ms)"},
        title="Forward Time vs. Số Tham Số",
        template=PLOTLY_DARK_TEMPLATE,
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def _chart_model_comparison(df_combined: pd.DataFrame) -> go.Figure:
    """Bar chart nhóm: so sánh mean_time_ms giữa các mô hình."""
    if "model_name" not in df_combined.columns or "forward_time_ms" not in df_combined.columns:
        return go.Figure()
    agg = df_combined.groupby("model_name").agg(
        total_time  = ("forward_time_ms", "sum"),
        mean_time   = ("forward_time_ms", "mean"),
        peak_mem    = ("gpu_mem_after_mb","max"),
        n_layers    = ("layer_name",      "nunique"),
    ).reset_index()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Tổng Forward Time (ms)", "Peak GPU Memory (MB)"),
    )
    colors = MODEL_PALETTE[:len(agg)]

    fig.add_trace(
        go.Bar(x=agg["model_name"], y=agg["total_time"], marker_color=colors,
               name="Total Time", text=agg["total_time"].round(1), textposition="outside"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=agg["model_name"], y=agg["peak_mem"], marker_color=colors,
               name="Peak Mem", showlegend=False,
               text=agg["peak_mem"].round(1), textposition="outside"),
        row=1, col=2,
    )
    fig.update_layout(
        template=PLOTLY_DARK_TEMPLATE,
        title="So Sánh Hiệu Năng Các Mô Hình",
        height=420,
        showlegend=False,
        margin=dict(l=10, r=10, t=70, b=10),
    )
    return fig


def _chart_rule_results(rules: list[dict]) -> go.Figure:
    """Grouped bar: số lượng quy tắc theo severity."""
    severity_counts: dict[str, int] = {s.value: 0 for s in SEVERITY_ORDER}
    for r in rules:
        sev = r.get("severity", "OK")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    colors = [SEVERITY_COLORS.get(Severity(s), "#888") for s in severity_counts.keys()]
    fig = go.Figure(data=[
        go.Bar(
            x=list(severity_counts.keys()),
            y=list(severity_counts.values()),
            marker_color=colors,
            text=list(severity_counts.values()),
            textposition="outside",
        )
    ])
    fig.update_layout(
        template=PLOTLY_DARK_TEMPLATE,
        title="Phân Phối Kết Quả Rule Catalog theo Severity",
        xaxis_title="Severity", yaxis_title="Số Quy Tắc",
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def _chart_rule_by_category(rules: list[dict]) -> go.Figure:
    """Stacked bar: kết quả quy tắc theo danh mục."""
    cat_data: dict[str, dict[str, int]] = {}
    cat_labels = {
        RuleCategory.MEMORY.value: "R-A Memory",
        RuleCategory.COMPUTE.value:"R-B Compute",
        RuleCategory.IO.value:     "R-C I/O",
        RuleCategory.ARCH.value:   "R-D Arch",
    }
    for r in rules:
        cat = r.get("category", "?")
        sev = r.get("severity", "OK")
        if cat not in cat_data:
            cat_data[cat] = {"CRITICAL":0,"HIGH":0,"MEDIUM":0,"LOW":0,"OK":0}
        cat_data[cat][sev] = cat_data[cat].get(sev, 0) + 1

    cats    = [cat_labels.get(c, c) for c in cat_data.keys()]
    traces  = []
    for sev in ["CRITICAL","HIGH","MEDIUM","LOW","OK"]:
        vals = [cat_data[c].get(sev, 0) for c in cat_data.keys()]
        traces.append(go.Bar(
            name=sev, x=cats, y=vals,
            marker_color=SEVERITY_COLORS.get(Severity(sev), "#888"),
        ))
    fig = go.Figure(data=traces)
    fig.update_layout(
        barmode="stack",
        template=PLOTLY_DARK_TEMPLATE,
        title="Rule Catalog — Kết Quả theo Danh Mục",
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _chart_ab_waterfall(diffs: list[dict]) -> go.Figure:
    """Waterfall chart: delta_ms của từng layer."""
    top_n = 20
    # Lấy top regressions + improvements
    regressions  = sorted([d for d in diffs if d.get("is_regression")],
                          key=lambda x: x["delta_pct"], reverse=True)[:top_n // 2]
    improvements = sorted([d for d in diffs if d.get("is_improvement")],
                          key=lambda x: x["delta_pct"])[:top_n // 2]
    selected     = improvements + regressions

    if not selected:
        return go.Figure().update_layout(title="Không có thay đổi đáng kể")

    names  = [d["layer_name"] for d in selected]
    deltas = [d["delta_pct"]  for d in selected]
    colors = ["#e53e3e" if v > 0 else "#38a169" for v in deltas]

    fig = go.Figure(go.Bar(
        x=names, y=deltas,
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in deltas],
        textposition="outside",
    ))
    fig.add_hline(y=0, line_width=1, line_color="white", line_dash="dot")
    fig.update_layout(
        template=PLOTLY_DARK_TEMPLATE,
        title="A/B Delta Time % (đỏ=chậm hơn, xanh=nhanh hơn)",
        xaxis_title="Layer",
        yaxis_title="Δ Time (%)",
        height=420,
        margin=dict(l=10, r=10, t=50, b=40),
        xaxis=dict(tickangle=-45),
    )
    return fig


def _chart_ab_scatter(diffs: list[dict]) -> go.Figure:
    """Scatter: mean_time_a vs mean_time_b với đường y=x."""
    times_a = [d["mean_time_a_ms"] for d in diffs]
    times_b = [d["mean_time_b_ms"] for d in diffs]
    names   = [d["layer_name"] for d in diffs]
    pcts    = [d["delta_pct"]   for d in diffs]
    colors  = ["#e53e3e" if d.get("is_regression") else
               ("#38a169" if d.get("is_improvement") else "#718096")
               for d in diffs]

    fig = go.Figure()
    # Đường y = x (không thay đổi)
    mx = max(max(times_a), max(times_b)) * 1.05 if times_a else 1
    fig.add_trace(go.Scatter(
        x=[0, mx], y=[0, mx], mode="lines",
        line=dict(color="gray", dash="dash", width=1),
        name="y = x (không đổi)",
    ))
    fig.add_trace(go.Scatter(
        x=times_a, y=times_b,
        mode="markers",
        marker=dict(color=colors, size=8, opacity=0.85),
        text=[f"{n}<br>Δ={p:+.1f}%" for n, p in zip(names, pcts)],
        hovertemplate="%{text}<br>A=%{x:.3f}ms B=%{y:.3f}ms<extra></extra>",
        name="Layers",
    ))
    fig.update_layout(
        template=PLOTLY_DARK_TEMPLATE,
        title="A/B Scatter: Session A vs Session B (ms)",
        xaxis_title="Session A (ms)",
        yaxis_title="Session B (ms)",
        height=450,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def _chart_dataloader_timeline(batch_df: pd.DataFrame) -> go.Figure:
    """Stacked bar: load_time vs compute_time theo batch."""
    if batch_df.empty or "load_time_ms" not in batch_df.columns:
        return go.Figure().update_layout(title="Không có dữ liệu DataLoader chi tiết")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=batch_df["batch_index"],
        y=batch_df["load_time_ms"],
        name="Load Time (I/O)",
        marker_color="#e94560",
    ))
    if "compute_time_ms" in batch_df.columns:
        fig.add_trace(go.Bar(
            x=batch_df["batch_index"],
            y=batch_df["compute_time_ms"],
            name="Compute Time (GPU)",
            marker_color="#63b3ed",
        ))
    fig.update_layout(
        barmode="stack",
        template=PLOTLY_DARK_TEMPLATE,
        title="DataLoader: Load vs Compute Time mỗi Batch",
        xaxis_title="Batch Index",
        yaxis_title="Time (ms)",
        height=380,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def _chart_io_gauge(io_ratio: float) -> go.Figure:
    """Gauge chart: tỷ lệ I/O bottleneck."""
    color = "#38a169" if io_ratio < 0.4 else ("#dd6b20" if io_ratio < 0.6 else "#e53e3e")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=io_ratio * 100,
        title={"text": "I/O Bottleneck Ratio (%)", "font": {"size": 14}},
        delta={"reference": 40},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": color},
            "steps": [
                {"range": [0,  40], "color": "#1a2535"},
                {"range": [40, 60], "color": "#2d3748"},
                {"range": [60, 100],"color": "#4a2020"},
            ],
            "threshold": {
                "line":  {"color": "#e53e3e", "width": 4},
                "thickness": 0.75,
                "value":     60,
            },
        },
        number={"suffix": "%", "font": {"size": 28}},
    ))
    fig.update_layout(
        template=PLOTLY_DARK_TEMPLATE,
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """Render sidebar và trả về config dict."""
    with st.sidebar:
        st.markdown("## ⚡ PyTorch Profiler")
        st.markdown("---")

        st.markdown("### 📁 Thư Mục Dữ Liệu")
        base_dir_str = st.text_input(
            "Profiles directory",
            value=str(_PROJECT_ROOT / "data" / "profiles"),
            help="Thư mục chứa các sub-folder mô hình (output của MultiModelProfiler)",
        )
        base_dir = Path(base_dir_str)

        st.markdown("### 🎛️ Cài Đặt Phân Tích")
        total_gpu_mem = st.number_input(
            "GPU Memory tổng (MB)",
            min_value=0.0,
            value=15360.0,
            step=1024.0,
            help="T4=15360MB, A100=40960MB, RTX3090=24576MB. Nhập 0 nếu CPU-only.",
        )
        device = st.selectbox("Device", ["cpu", "cuda"], index=0)

        st.markdown("### 📊 A/B Comparison")
        model_dirs: list[Path] = []
        if base_dir.exists():
            model_dirs = _find_profile_dirs(base_dir)

        model_names = [d.name for d in model_dirs]
        sel_a = st.selectbox("Session A", model_names, index=0 if model_names else None)
        sel_b = st.selectbox(
            "Session B",
            model_names,
            index=min(1, len(model_names) - 1) if len(model_names) > 1 else 0,
        )
        reg_pct = st.slider("Regression threshold (%)", 1.0, 20.0, 5.0, 1.0)
        imp_pct = st.slider("Improvement threshold (%)", 1.0, 20.0, 5.0, 1.0)

        st.markdown("### 🔍 DataLoader JSON")
        dl_json = st.text_input(
            "DataLoader report JSON path",
            value="",
            help="Đường dẫn tới file JSON từ DataLoaderSentinel.save_report()",
        )

        st.markdown("---")
        st.caption(f"Project root: `{_PROJECT_ROOT}`")

    return {
        "base_dir":       base_dir,
        "model_dirs":     model_dirs,
        "model_names":    model_names,
        "total_gpu_mem":  total_gpu_mem,
        "device":         device,
        "sel_a":          sel_a,
        "sel_b":          sel_b,
        "reg_pct":        reg_pct,
        "imp_pct":        imp_pct,
        "dl_json":        dl_json,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tab renderers
# ─────────────────────────────────────────────────────────────────────────────

def tab_overview(cfg: dict) -> None:
    """Tab 1: Tổng quan so sánh các mô hình."""
    st.subheader("📊 Tổng Quan — So Sánh Mô Hình")

    if not cfg["model_dirs"]:
        st.info("⚙️ Chưa có dữ liệu. Chạy `MultiModelProfiler.run()` để tạo dữ liệu.")
        _render_quickstart()
        return

    # Đọc combined summary
    combined_path = cfg["base_dir"] / "combined_summary.json"
    if combined_path.exists():
        combined = load_json(str(combined_path))
        sessions = combined.get("sessions", [])
        if sessions:
            df_sess = pd.DataFrame(sessions)
            st.dataframe(
                df_sess.style.highlight_max(
                    subset=[c for c in ["total_time_ms","peak_mem_mb"] if c in df_sess.columns],
                    color="#4a2020",
                ).highlight_min(
                    subset=[c for c in ["total_time_ms","peak_mem_mb"] if c in df_sess.columns],
                    color="#1a3a1a",
                ),
                use_container_width=True,
                key="tab1_sessions_df",
            )

    # Ghép tất cả CSV
    all_dfs = []
    for d in cfg["model_dirs"]:
        csv_p = d / "layer_records.csv"
        if csv_p.exists():
            df_m = load_csv(str(csv_p))
            df_m["model_name"] = d.name
            all_dfs.append(df_m)

    if not all_dfs:
        st.warning("Không đọc được dữ liệu layer records.")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)

    # Chart so sánh
    st.plotly_chart(_chart_model_comparison(df_all), use_container_width=True, key="tab1_model_cmp")

    # Scatter time vs params
    col1, col2 = st.columns(2)
    with col1:
        # Model chọn để xem chi tiết
        sel_model = st.selectbox("Chọn mô hình để xem chi tiết:", cfg["model_names"])
        if sel_model:
            df_sel = df_all[df_all["model_name"] == sel_model]
            st.plotly_chart(_chart_scatter_time_vs_params(df_sel), use_container_width=True, key="tab1_scatter_sel")

    with col2:
        if sel_model:
            df_sel = df_all[df_all["model_name"] == sel_model]
            st.plotly_chart(_chart_layer_type_pie(df_sel), use_container_width=True, key="tab1_pie_sel")

    # Bảng tổng hợp per model
    st.markdown("#### 📋 Bảng Tổng Hợp Hiệu Năng")
    summary_rows = []
    for d in cfg["model_dirs"]:
        csv_p = d / "layer_records.csv"
        json_p = d / "session_summary.json"
        if not csv_p.exists():
            continue
        df_m = load_csv(str(csv_p))
        sm   = load_json(str(json_p)) if json_p.exists() else {}
        summary_rows.append({
            "Mô hình":           d.name,
            "Số layers đo":      df_m["layer_name"].nunique() if "layer_name" in df_m.columns else 0,
            "Total time (ms)":   round(df_m["forward_time_ms"].sum(), 2) if "forward_time_ms" in df_m.columns else 0,
            "Mean/layer (ms)":   round(df_m["forward_time_ms"].mean(), 4) if "forward_time_ms" in df_m.columns else 0,
            "Peak mem (MB)":     round(sm.get("peak_mem_mb", 0), 2),
            "Total params (M)":  round(df_m["param_count"].iloc[0:1].sum() / 1e6, 2) if "param_count" in df_m.columns else 0,
            "Iterations":        sm.get("metadata", {}).get("n_iterations", "?"),
        })
    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, key="tab1_summary_df")


def tab_layer_analysis(cfg: dict) -> None:
    """Tab 2: Phân tích chi tiết từng layer."""
    st.subheader("🔬 Phân Tích Layer Chi Tiết")

    if not cfg["model_dirs"]:
        st.info("Chưa có dữ liệu profiling.")
        return

    sel_model = st.selectbox("Chọn mô hình:", cfg["model_names"], key="tab2_model")
    csv_path  = cfg["base_dir"] / sel_model / "layer_records.csv"
    if not csv_path.exists():
        st.error(f"Không tìm thấy: {csv_path}")
        return

    df = load_csv(str(csv_path))
    top_k = st.slider("Top-K layers hiển thị:", 5, 50, 15, key="tab2_topk")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(_chart_top_k_layers(df, k=top_k), use_container_width=True, key="tab2_topk_layers")
    with col2:
        st.plotly_chart(_chart_memory_by_layer(df, k=top_k), use_container_width=True, key="tab2_mem_by_layer")

    st.plotly_chart(_chart_memory_heatmap(df, top_k=min(top_k, 25)), use_container_width=True, key="tab2_mem_heatmap")
    st.plotly_chart(_chart_scatter_time_vs_params(df), use_container_width=True, key="tab2_scatter_time")

    # Bảng raw data
    with st.expander("📋 Raw Layer Records"):
        agg = df.groupby(["layer_name", "layer_type"]).agg(
            mean_time_ms = ("forward_time_ms",   "mean"),
            std_time_ms  = ("forward_time_ms",   "std"),
            param_count  = ("param_count",        "first"),
            mean_mem_delta = ("gpu_mem_delta_mb", "mean"),
            n_samples    = ("forward_time_ms",   "count"),
        ).reset_index().round(4)
        st.dataframe(
            agg.sort_values("mean_time_ms", ascending=False),
            use_container_width=True,
            height=400,
            key="tab2_raw_df",
        )


def tab_rule_catalog(cfg: dict) -> None:
    """Tab 3: Kết quả Rule Catalog."""
    st.subheader("📏 Rule Catalog — Kết Quả Phân Tích Heuristic")

    if not cfg["model_dirs"]:
        st.info("Chưa có dữ liệu.")
        return

    sel_model = st.selectbox("Chọn mô hình để phân tích:", cfg["model_names"], key="tab3_model")
    csv_path  = cfg["base_dir"] / sel_model / "layer_records.csv"
    if not csv_path.exists():
        st.error(f"Không tìm thấy: {csv_path}")
        return

    with st.spinner("Đang chạy Rule Catalog..."):
        report_dict = run_rule_catalog(
            csv_path        = str(csv_path),
            model_name      = sel_model,
            device          = cfg["device"],
            total_gpu_mem_mb= cfg["total_gpu_mem"],
            dl_json_path    = cfg["dl_json"],
        )

    rules = report_dict.get("rules", [])

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-val">{report_dict["total_rules"]}</div>'
            f'<div class="metric-label">Tổng Quy Tắc</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-val" style="color:#38a169">{report_dict["passed"]}</div>'
            f'<div class="metric-label">Passed ✅</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-val" style="color:#dd6b20">{report_dict["failed"]}</div>'
            f'<div class="metric-label">Failed ❌</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-val" style="color:#e53e3e">{report_dict["critical"]}</div>'
            f'<div class="metric-label">Critical 🔴</div></div>', unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(_chart_rule_results(rules), use_container_width=True, key="tab3_rule_results")
    with col2:
        st.plotly_chart(_chart_rule_by_category(rules), use_container_width=True, key="tab3_rule_cat")

    # Bảng chi tiết quy tắc
    st.markdown("#### 📋 Chi Tiết Từng Quy Tắc")

    # Filter severity
    sev_filter = st.multiselect(
        "Lọc theo Severity:",
        [s.value for s in SEVERITY_ORDER],
        default=[Severity.CRITICAL.value, Severity.HIGH.value, Severity.MEDIUM.value],
        key="tab3_sev_filter",
    )

    for r in rules:
        if r["severity"] not in sev_filter and sev_filter:
            continue
        sev   = r["severity"]
        color = SEVERITY_COLORS.get(Severity(sev), "#888")
        icon  = "✅" if r["passed"] else ("🔴" if sev == "CRITICAL" else "⚠️")

        with st.expander(
            f"{icon} [{r['rule_id']}] {r['rule_name']} — {sev}",
            expanded=(sev in ["CRITICAL", "HIGH"]),
        ):
            st.markdown(f"**Thông điệp:** {r['message']}")
            if r.get("suggestions"):
                st.markdown("**💡 Gợi ý:**")
                for s in r["suggestions"]:
                    st.markdown(f"- {s}")
            if r.get("affected_layers"):
                st.markdown(f"**Layers liên quan:** `{'`, `'.join(r['affected_layers'][:5])}`")
            if r.get("details"):
                st.json(r["details"])


def tab_ab_comparison(cfg: dict) -> None:
    """Tab 4: A/B Comparison."""
    st.subheader("⚡ A/B Comparison Engine")

    if len(cfg["model_dirs"]) < 2:
        st.info("Cần ít nhất 2 model profiles để so sánh. Chọn Session A và B ở sidebar.")
        return

    sel_a, sel_b = cfg["sel_a"], cfg["sel_b"]
    if not sel_a or not sel_b:
        st.warning("Vui lòng chọn Session A và Session B trong sidebar.")
        return
    if sel_a == sel_b:
        st.warning("Session A và B phải khác nhau.")
        return

    csv_a = cfg["base_dir"] / sel_a / "layer_records.csv"
    csv_b = cfg["base_dir"] / sel_b / "layer_records.csv"

    if not csv_a.exists() or not csv_b.exists():
        st.error(f"Không tìm thấy CSV của session A ({sel_a}) hoặc B ({sel_b}).")
        return

    with st.spinner("Đang so sánh A/B..."):
        try:
            engine = ABComparisonEngine(
                df_a=str(csv_a), df_b=str(csv_b),
                name_a=sel_a, name_b=sel_b,
                regression_threshold_pct=cfg["reg_pct"],
                improvement_threshold_pct=cfg["imp_pct"],
            )
            report = engine.compare()
        except Exception as e:
            st.error(f"Lỗi A/B Compare: {e}")
            return

    # Metrics
    speedup   = report.overall_speedup
    spd_color = "#38a169" if speedup > 1.05 else ("#e53e3e" if speedup < 0.95 else "#d69e2e")
    spd_label = f"{sel_b} nhanh hơn {sel_a}" if speedup > 1 else f"{sel_a} nhanh hơn {sel_b}"

    col1, col2, col3, col4, col5 = st.columns(5)
    metrics = [
        (f"{speedup:.3f}×", "Overall Speedup", spd_color),
        (str(report.total_layers), "Layer So Sánh", "#63b3ed"),
        (str(report.improvements), "Cải Thiện ✅", "#38a169"),
        (str(report.regressions),  "Hồi Quy ❌",   "#e53e3e"),
        (str(report.unchanged),    "Không Đổi ➡️", "#718096"),
    ]
    for col, (val, label, color) in zip([col1, col2, col3, col4, col5], metrics):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-val" style="color:{color}">{val}</div>'
                f'<div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True,
            )

    sig = report.summary.get("overall_significant", False)
    p   = report.summary.get("overall_p_value", 1.0)
    st.caption(
        f"📊 Ý nghĩa thống kê (Mann-Whitney): p={p:.4f} — "
        f"{'✅ Có ý nghĩa thống kê (p < 0.05)' if sig else '⚠️ Chưa có ý nghĩa thống kê'}. "
        f"{spd_label}."
    )

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # Charts
    diffs = [d.__dict__ if hasattr(d, "__dict__") else d
             for d in report.layer_diffs]
    # Convert LayerDiff dataclasses to dicts
    import dataclasses
    diffs_dict = [dataclasses.asdict(d) if dataclasses.is_dataclass(d) else d
                  for d in report.layer_diffs]

    col1, col2 = st.columns([3, 2])
    with col1:
        st.plotly_chart(_chart_ab_waterfall(diffs_dict), use_container_width=True, key="tab4_ab_waterfall")
    with col2:
        st.plotly_chart(_chart_ab_scatter(diffs_dict), use_container_width=True, key="tab4_ab_scatter")

    # Bảng chi tiết
    with st.expander("📋 Chi Tiết Từng Layer"):
        df_diff = pd.DataFrame(diffs_dict).round(4)
        df_diff = df_diff.sort_values("delta_pct", ascending=False)
        st.dataframe(df_diff, use_container_width=True, height=400, key="tab4_diff_df")


def tab_dataloader(cfg: dict) -> None:
    """Tab 5: DataLoader Analysis."""
    st.subheader("🔄 DataLoader I/O Analysis")

    dl_json = cfg.get("dl_json", "").strip()

    if not dl_json:
        st.info(
            "Chưa có dữ liệu DataLoader. Nhập đường dẫn JSON ở sidebar, "
            "hoặc chạy `DataLoaderSentinel.save_report(path)` trước."
        )
        _render_dataloader_usage()
        return

    dl_path = Path(dl_json)
    if not dl_path.exists():
        st.error(f"Không tìm thấy: {dl_path}")
        return

    data    = load_json(str(dl_path))
    summary = data.get("summary", {})
    batches = data.get("batch_detail", [])
    batch_df = pd.DataFrame(batches)

    # Metrics
    io_ratio = float(summary.get("io_bottleneck_ratio", 0))
    col1, col2, col3, col4 = st.columns(4)
    cols_info = [
        ("io_bottleneck_ratio", "I/O Ratio", f"{io_ratio*100:.1f}%"),
        ("mean_load_time_ms",   "Mean Load (ms)",  f"{summary.get('mean_load_time_ms', 0):.2f}"),
        ("mean_compute_time_ms","Mean Compute (ms)",f"{summary.get('mean_compute_time_ms', 0):.2f}"),
        ("worker_count",        "Workers",          str(summary.get("worker_count", 0))),
    ]
    for col, (_, label, val) in zip([col1, col2, col3, col4], cols_info):
        with col:
            st.metric(label, val)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(_chart_dataloader_timeline(batch_df), use_container_width=True, key="tab5_dl_timeline")
    with col2:
        st.plotly_chart(_chart_io_gauge(io_ratio), use_container_width=True, key="tab5_io_gauge")

    # Rule C tự động
    st.markdown("#### 📏 Kết Quả Rule R-C cho DataLoader Này")
    catalog = RuleCatalog()
    load_times = [b.get("load_time_ms", 0) for b in batches]
    rules_rc = [
        catalog.rc01_dataloader_bottleneck(
            io_ratio,
            float(summary.get("mean_load_time_ms", 0)),
            float(summary.get("mean_compute_time_ms", 0)),
        ),
        catalog.rc02_dataloader_variance(load_times),
        catalog.rc03_compute_starvation(io_ratio),
        catalog.rc04_zero_worker_warning(int(summary.get("worker_count", 0))),
    ]
    for r in rules_rc:
        icon = "✅" if r.passed else ("🔴" if r.severity == Severity.CRITICAL else "⚠️")
        st.markdown(f"**{icon} [{r.rule_id}] {r.rule_name}** — {r.severity}")
        st.markdown(f"&nbsp;&nbsp;{r.message}")
        if r.suggestions:
            for s in r.suggestions:
                st.markdown(f"&nbsp;&nbsp;💡 {s}")

    with st.expander("📋 Summary JSON"):
        st.json(summary)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: quickstart / usage guides
# ─────────────────────────────────────────────────────────────────────────────

def _render_quickstart() -> None:
    st.markdown("""
    #### 🚀 Hướng Dẫn Nhanh
    ```python
    # 1. Chạy benchmark (tạo data)
    python tests/test_modern_models_benchmark.py

    # 2. Chạy test bottleneck
    python tests/test_training_bottleneck.py

    # 3. Mở dashboard này
    streamlit run src/dashboard.py
    ```
    """)


def _render_dataloader_usage() -> None:
    st.markdown("""
    #### 📦 Cách dùng DataLoaderSentinel
    ```python
    from src.collector import DataLoaderSentinel

    sentinel = DataLoaderSentinel(train_loader, device="cuda")
    for batch in sentinel:
        images, labels = batch
        output = model(images)
        loss   = criterion(output, labels)
        loss.backward()
        optimizer.step()
        sentinel.mark_compute_end()  # <-- gọi sau mỗi backward()

    sentinel.save_report("reports/dataloader_stats.json")
    ```
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>⚡ PyTorch Layer Profiler Dashboard</h1>
        <p>Phân tích bottleneck, Rule Catalog R-A→R-D và A/B Comparison cho các mô hình Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)

    cfg = render_sidebar()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Tổng Quan",
        "🔬 Layer Analysis",
        "📏 Rule Catalog",
        "⚡ A/B Compare",
        "🔄 DataLoader",
    ])

    with tab1:
        tab_overview(cfg)
    with tab2:
        tab_layer_analysis(cfg)
    with tab3:
        tab_rule_catalog(cfg)
    with tab4:
        tab_ab_comparison(cfg)
    with tab5:
        tab_dataloader(cfg)


if __name__ == "__main__":
    main()
