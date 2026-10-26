"""Tab 3: Leaf layer breakdown, Roofline modeling, and hotspot analysis [Owned by TV1]."""

import json
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd

from src.dashboard.utils import page_header, show_chart

import streamlit as st

from src.dashboard.layer_charts import (
    CPU_HW, GPU_HW, DEVICE_LABELS, _fig_type_share, _fig_timeline,
    make_roofline_chart as _fig_roofline,
)

CONFIG = "bs1_fp32"
MODEL_LABELS = {
    "mobilenet_v3_large": "MobileNetV3-Large", "resnet50": "ResNet-50", "yolov8n": "YOLOv8n",
    "vit_b_16": "ViT-B/16", "distilbert": "DistilBERT", "bert_base": "BERT-base", "mlp": "MLP",
}


def _cache_data(func: Any) -> Any:
    if st is not None and hasattr(st, "cache_data"):
        return st.cache_data(show_spinner=False)(func)
    return func


@_cache_data
def _read_csv(path: str) -> Optional[pd.DataFrame]:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


@_cache_data
def _read_json(path: str) -> Optional[dict]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _find_tn1_path(results_dir: Path, *parts: str) -> Path:
    p = results_dir.joinpath(*parts)
    if not p.exists():
        lowered_parts = [part.lower() if part in ("TN1", "TN2") else part for part in parts]
        alt = results_dir.joinpath(*lowered_parts)
        if alt.exists():
            return alt
    return p


def _available_models(results_dir: Path) -> List[str]:
    base = _find_tn1_path(results_dir, "analysis", "TN1")
    return [m for m in MODEL_LABELS if (base / m).exists()]


def _tn1(results_dir: Path, hw: str, model: str, name: str) -> Any:
    path = _find_tn1_path(results_dir, hw, model, "TN1", CONFIG, name)
    return _read_json(str(path)) if name.endswith(".json") else _read_csv(str(path))


def render_layer_tab(results_dir: Path) -> None:
    """Render Tab 3: Detailed leaf layer breakdown, Roofline curve, and CPU vs GPU comparison."""
    if st is None:
        return
    page_header("Phân tích layer", "layer")
    models = _available_models(results_dir)
    if not models:
        st.info("Chưa có kết quả TN1. Chạy `python experiments/tn1_layer_breakdown.py --analyze` trước.")
        return

    c1, c2 = st.columns([2, 1])
    model = c1.selectbox("Mô hình", models, format_func=MODEL_LABELS.get, key="layer_tab_model")
    hw = c2.radio("Thiết bị cho biểu đồ chi tiết", [GPU_HW, CPU_HW], format_func=DEVICE_LABELS.get, horizontal=True, key="layer_tab_hw")

    share_path = _find_tn1_path(results_dir, "analysis", "TN1", model, "compare_type_share.csv")
    top_path = _find_tn1_path(results_dir, "analysis", "TN1", model, "compare_top10.csv")
    share, top = _read_csv(str(share_path)), _read_csv(str(top_path))
    meta_cpu, meta_gpu = _tn1(results_dir, CPU_HW, model, "layer_meta.json"), _tn1(results_dir, GPU_HW, model, "layer_meta.json")

    if share is not None and "time_ms_cpu" in share.columns:
        t_cpu, t_gpu = share["time_ms_cpu"].sum(), share["time_ms_gpu"].sum()
        k = st.columns(4)
        k[0].metric("Tổng thời gian layer · CPU", f"{t_cpu:.2f} ms")
        k[1].metric("Tổng thời gian layer · GPU T4", f"{t_gpu:.2f} ms")
        k[2].metric("Tăng tốc CPU / T4", f"×{t_cpu / t_gpu:.1f}" if t_gpu > 0 else "—")
        meta = meta_gpu or meta_cpu
        if meta and meta.get("total_flops_counter"):
            flops = meta["total_flops_counter"]
            k[3].metric("FLOPs / 1 lần forward", f"{flops / 1e9:.2f} G" if flops >= 1e8 else f"{flops / 1e6:.2f} M")

        st.markdown("##### Tỷ trọng thời gian theo loại layer")
        fig_share = _fig_type_share(share)
        if fig_share is not None:
            show_chart(fig_share, key="layer_tab_share")
        with st.expander("Xem bảng số · tỷ trọng layer"):
            st.dataframe(share.round(3), width="stretch", hide_index=True, column_config={
                "layer_type": "Loại layer", "time_ms_cpu": "CPU (ms)", "time_ms_gpu": "T4 (ms)",
                "pct_cpu": "CPU (%)", "pct_gpu": "T4 (%)", "speedup": "Tăng tốc (×)",
            })

    if top is not None:
        st.markdown("##### Top-10 layer chậm · CPU và T4")
        cols = [c for c in ["layer_name", "layer_type", "time_ms_cpu", "rank_cpu", "time_ms_gpu", "rank_gpu", "speedup"] if c in top.columns]
        st.dataframe(top[cols].round(3), width="stretch", hide_index=True, column_config={
            "layer_name": "Layer", "layer_type": "Loại", "time_ms_cpu": "CPU (ms)",
            "time_ms_gpu": "T4 (ms)", "rank_cpu": "Hạng CPU", "rank_gpu": "Hạng T4", "speedup": "Tăng tốc (×)",
        })

    layers = _tn1(results_dir, hw, model, "layers.csv")
    summary = _tn1(results_dir, hw, model, "layers_summary.csv")
    meta = meta_gpu if hw == GPU_HW else meta_cpu
    if layers is not None and not layers.empty:
        st.markdown(f"##### Thời gian từng layer theo thứ tự chạy · {DEVICE_LABELS.get(hw, hw)}")
        fig_time = _fig_timeline(layers)
        if fig_time is not None:
            show_chart(fig_time, key="layer_tab_timeline")
    if summary is not None and meta and meta.get("hardware"):
        st.markdown(f"##### Roofline · {DEVICE_LABELS.get(hw, hw)}")
        fig_roof = _fig_roofline(summary, meta["hardware"])
        if fig_roof is not None:
            show_chart(fig_roof, key="layer_tab_roof")


def render(results_dir: Path) -> None:
    """Entrypoint compatible with router in src/dashboard/app.py."""
    render_layer_tab(results_dir)
