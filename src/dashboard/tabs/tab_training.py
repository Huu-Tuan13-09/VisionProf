"""Tab 5: Giai đoạn training, VRAM và DataLoader của TN6."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.utils import (
    DEVICE_COLORS as _DEVICE_COLORS, PHASE_COLORS as _PHASE_COLORS, page_header, show_chart,
)


def _load_tn6_phases(results_dir: Path) -> pd.DataFrame:
    """Scan results/**TN6**/phases.csv and dataloader_phases.csv and return aggregated DataFrame."""
    rows: List[Dict[str, Any]] = []
    seen = set()
    for pat in ("**/TN6/**/phases.csv", "**/TN6/**/dataloader_phases.csv"):
        for csv_path in results_dir.glob(pat):
            if csv_path in seen: continue
            seen.add(csv_path)
            parts = csv_path.relative_to(results_dir).parts
            if len(parts) < 4: continue
            device, model, config = parts[0], parts[1], parts[3]
            try:
                df_raw = pd.read_csv(csv_path)
            except Exception: continue
            if df_raw.empty or "phase" not in df_raw.columns: continue
            for phase, grp in df_raw.groupby("phase"):
                rows.append({
                    "device": device, "model": model, "config": config, "phase": phase,
                    "mean_ms": round(grp["time_ms"].mean(), 3),
                    "p50_ms": round(grp["time_ms"].median(), 3),
                    "p95_ms": round(grp["time_ms"].quantile(0.95), 3),
                    "std_ms": round(grp["time_ms"].std(), 3),
                    "n_iter": len(grp),
                })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _load_memory_ratio(results_dir: Path) -> pd.DataFrame:
    """Load memory_ratio.json files from TN6 runs."""
    rows: List[Dict[str, Any]] = []
    for jf in results_dir.glob("**/TN6/**/memory_ratio.json"):
        parts = jf.relative_to(results_dir).parts
        if len(parts) < 4: continue
        device, model = parts[0], parts[1]
        try:
            with open(jf, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            rows.append({
                "device": device, "model": model,
                "infer_vram_mb": data.get("infer_peak_vram_mb", 0),
                "train_vram_mb": data.get("train_peak_vram_mb", 0),
                "memory_ratio": data.get("memory_ratio", None),
            })
        except Exception: continue
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _render_stacked_breakdown(df: pd.DataFrame) -> None:
    """Stacked bar: each model coloured by phase — shows total and breakdown."""
    st.subheader("Thời gian mỗi bước training")
    st.caption("Thời gian trung bình theo giai đoạn; mỗi màu ứng với một giai đoạn.")

    devices = sorted(df["device"].unique())
    sel_dev = st.radio("Thiết bị:", devices, index=devices.index("colab_t4") if "colab_t4" in devices else 0, horizontal=True, key="tn6_dev_stacked")
    sub = df[df["device"] == sel_dev].copy()
    if sub.empty:
        st.info("Không có dữ liệu TN6 cho thiết bị này.")
        return

    pivot = sub.pivot_table(index="model", columns="phase", values="mean_ms", aggfunc="mean").fillna(0)
    pivot["total_ms"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total_ms", ascending=False).drop(columns="total_ms")

    fig = go.Figure()
    for phase in pivot.columns:
        color = _PHASE_COLORS.get(phase, "#94A3B8")
        fig.add_trace(go.Bar(
            name=phase, x=pivot.index.tolist(), y=pivot[phase].tolist(),
            marker_color=color,
            text=[f"{v:.1f}" for v in pivot[phase]],
            textposition="inside",
        ))
    fig.update_layout(
        barmode="stack", title=f"Thời gian theo giai đoạn · {sel_dev}",
        xaxis_title="Mô hình", yaxis_title="Thời gian trung bình (ms)",
        template="plotly_white", legend_title="Giai đoạn", height=420,
    )
    show_chart(fig)


def _render_bwd_fwd_ratio(df: pd.DataFrame) -> None:
    """Bar chart: backward/forward ratio to answer CH5."""
    st.subheader("Tỷ lệ backward / forward")

    pivot = df.pivot_table(
        index=["device", "model"], columns="phase", values="mean_ms", aggfunc="mean"
    ).reset_index()

    if "forward" not in pivot.columns or "backward" not in pivot.columns:
        st.info("Chưa đủ dữ liệu backward/forward để tính tỷ lệ.")
        return

    pivot["bwd_fwd_ratio"] = (pivot["backward"] / pivot["forward"]).round(2)

    col1, col2 = st.columns(2)
    for dev, col in zip(sorted(df["device"].unique()), [col1, col2]):
        sub = pivot[pivot["device"] == dev].dropna(subset=["bwd_fwd_ratio"])
        with col:
            if sub.empty:
                st.info(f"Không có dữ liệu cho {dev}.")
                continue
            color = _DEVICE_COLORS.get(dev, "#94A3B8")
            fig = px.bar(
                sub.sort_values("bwd_fwd_ratio", ascending=False),
                x="model", y="bwd_fwd_ratio",
                title=f"Backward / forward · {dev}",
                color_discrete_sequence=[color],
                text="bwd_fwd_ratio",
                labels={"bwd_fwd_ratio": "Tỷ lệ (×)", "model": "Mô hình"},
                template="plotly_white",
            )
            fig.add_hline(y=2.0, line_dash="dot", line_color="gray",
                          annotation_text="Kỳ vọng 2×", annotation_position="top right")
            fig.update_layout(height=350)
            show_chart(fig)


def _render_memory_ratio(df_mem: pd.DataFrame) -> None:
    """Show train/infer VRAM ratio from memory_ratio.json."""
    st.subheader("VRAM training và inference")
    if df_mem.empty:
        st.info("Chưa có file `memory_ratio.json` (chỉ xuất hiện khi chạy trên GPU CUDA).")
        return

    fig = px.bar(
        df_mem.dropna(subset=["memory_ratio"]).sort_values("memory_ratio", ascending=False),
        x="model", y="memory_ratio", color="device",
        barmode="group",
        text="memory_ratio",
        title="Tỷ lệ peak VRAM · training / inference",
        labels={"memory_ratio": "Train / infer (×)", "model": "Mô hình"},
        color_discrete_map=_DEVICE_COLORS,
        template="plotly_white",
    )
    fig.add_hline(y=2.0, line_dash="dash", line_color="#F59E0B",
                  annotation_text="Mốc tham chiếu ×2")
    fig.update_layout(height=380)
    show_chart(fig)


def _render_latency_dist(results_dir: Path) -> None:
    """Box plot: iteration-level time_ms distribution per phase."""
    st.subheader("Phân phối thời gian theo giai đoạn")

    phase_files = sorted(list(results_dir.glob("**/TN6/**/phases.csv")) + list(results_dir.glob("**/TN6/**/dataloader_phases.csv")))
    if not phase_files:
        return

    file_opts = {str(p.relative_to(results_dir)): p for p in phase_files}
    sel = st.selectbox("Chọn lượt đo", list(file_opts.keys()),
                       key="tn6_box_sel")
    if not sel:
        return
    try:
        df_raw = pd.read_csv(file_opts[sel])
    except Exception:
        st.error("Không thể đọc file.")
        return

    if df_raw.empty:
        return

    fig = px.box(
        df_raw, x="phase", y="time_ms", color="phase",
        points="all", title=f"Phân phối thời gian · {sel}",
        labels={"time_ms": "Thời gian (ms)", "phase": "Giai đoạn"},
        color_discrete_map=_PHASE_COLORS, template="plotly_white",
    )
    fig.update_layout(height=380, showlegend=False)
    show_chart(fig)


def _render_dataloader_workers(results_dir: Path) -> None:
    """Render DataLoader num_workers sweep analysis (CH5)."""
    st.subheader("DataLoader và thời gian chờ dữ liệu")
    st.caption("Dữ liệu đo thật với ảnh JPEG + Transforms trên Colab T4 (2 vCPU, Batch 32).")
    dl_rows = []
    for p in results_dir.glob("**/TN6/**/dataloader_summary.json"):
        try:
            with open(p, "r", encoding="utf-8") as f: d = json.load(f)
            dl_rows.append({
                "model": d.get("model"), "num_workers": d.get("num_workers"),
                "data_load_ms": round(d.get("median_ms", {}).get("data_load", 0), 1),
                "step_ms": round(d.get("step_ms", 0), 1),
                "data_load_%": f"{d.get('data_load_pct', 0):.1f}%",
                "throughput_img_s": round(d.get("images_per_s", 0), 1),
            })
        except Exception: continue
    if not dl_rows:
        st.info("Chưa có kết quả dataloader_summary.json.")
        return
    st.dataframe(pd.DataFrame(dl_rows).sort_values(by=["model", "num_workers"]),
                 width="stretch", hide_index=True, column_config={
                     "model": "Mô hình", "num_workers": "Workers", "data_load_ms": "Nạp dữ liệu (ms)",
                     "step_ms": "Bước training (ms)", "data_load_%": "Thời gian chờ (%)",
                     "throughput_img_s": "Thông lượng (ảnh/s)",
                 })


def render(results_dir: Path) -> None:
    """Render Tab 5: TN6 Training Phase Breakdown — answers CH5."""
    page_header("Training TN6", "phase")

    df = _load_tn6_phases(results_dir)
    if df.empty:
        st.info("Chưa có dữ liệu TN6. Chạy: `python scripts/run_tn6_training.py --device colab_t4`")
        return

    df_std = df[~df["config"].str.contains("_w")].copy() if "config" in df.columns else df

    t4 = df_std[df_std["device"] == "colab_t4"]
    if not t4.empty:
        t4_piv = t4.pivot_table(index="model", columns="phase", values="mean_ms", aggfunc="mean")
        if "forward" in t4_piv.columns and "backward" in t4_piv.columns:
            ratios = (t4_piv["backward"] / t4_piv["forward"]).dropna()
            col_k1, col_k2, col_k3 = st.columns(3)
            col_k1.metric("Backward / forward TB · T4", f"{ratios.mean():.2f}×")
            col_k2.metric("Mô hình có tỷ lệ cao nhất", ratios.idxmax(), delta=f"{ratios.max():.2f}×", delta_color="off", delta_arrow="off")
            col_k3.metric("Mô hình training chuẩn", f"{t4['model'].nunique()}")

    tab_bkd, tab_mem, tab_dl, tab_dist = st.tabs([
        "Thời gian giai đoạn", "VRAM train / infer", "DataLoader & Workers", "Phân phối độ trễ",
    ])
    with tab_bkd:
        _render_stacked_breakdown(df_std)
        _render_bwd_fwd_ratio(df_std)
    with tab_mem:
        _render_memory_ratio(_load_memory_ratio(results_dir))
    with tab_dl:
        _render_dataloader_workers(results_dir)
    with tab_dist:
        _render_latency_dist(results_dir)
