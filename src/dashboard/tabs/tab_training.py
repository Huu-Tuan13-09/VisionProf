"""Tab 5a — TN6: Training Phase Breakdown (CH5) [Owned by TV3].

Visualises forward / backward / optimizer step durations across 5 training
models on colab_t4 and colab_cpu.  Answers CH5:
  - Backward ≈ 2× forward?
  - Memory train vs inference?
  - DataLoader starvation at num_workers=0?
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

# ── Phase colour palette (stable across all charts) ──────────────────────────
_PHASE_COLORS: Dict[str, str] = {
    "forward":   "#3B82F6",   # blue
    "backward":  "#EF4444",   # red
    "optimizer": "#10B981",   # green
    "dataloader": "#F59E0B",  # amber
    "fit":       "#8B5CF6",   # violet (XGBoost)
}
_DEVICE_COLORS: Dict[str, str] = {
    "colab_t4":  "#7C3AED",
    "colab_cpu": "#0EA5E9",
    "cpu_laptop": "#64748B",
}


# ── Data loading helpers ─────────────────────────────────────────────────────

def _load_tn6_phases(results_dir: Path) -> pd.DataFrame:
    """Scan results/**TN6**/phases.csv and return aggregated DataFrame."""
    rows: List[Dict[str, Any]] = []
    for csv_path in results_dir.glob("**/TN6/**/phases.csv"):
        parts = csv_path.relative_to(results_dir).parts
        if len(parts) < 4:
            continue
        device, model = parts[0], parts[1]
        try:
            df_raw = pd.read_csv(csv_path)
        except Exception:
            continue
        if df_raw.empty or "phase" not in df_raw.columns:
            continue
        for phase, grp in df_raw.groupby("phase"):
            rows.append({
                "device": device,
                "model": model,
                "phase": phase,
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
        if len(parts) < 4:
            continue
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
        except Exception:
            continue
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Section renderers ────────────────────────────────────────────────────────

def _render_stacked_breakdown(df: pd.DataFrame) -> None:
    """Stacked bar: each model coloured by phase — shows total and breakdown."""
    st.subheader("📊 Phân Rã Bước Training (Stacked Bar)")
    st.caption("Mỗi thanh = tổng một iteration. Màu thể hiện giai đoạn.")

    devices = sorted(df["device"].unique())
    sel_dev = st.radio("Thiết bị:", devices, horizontal=True, key="tn6_dev_stacked")
    sub = df[df["device"] == sel_dev].copy()
    if sub.empty:
        st.info("Không có dữ liệu TN6 cho thiết bị này.")
        return

    # Pivot: model × phase → mean_ms
    pivot = sub.pivot_table(index="model", columns="phase", values="mean_ms", aggfunc="mean").fillna(0)
    pivot["total_ms"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total_ms", ascending=False).drop(columns="total_ms")

    fig = go.Figure()
    for phase in pivot.columns:
        color = _PHASE_COLORS.get(phase, "#94A3B8")
        fig.add_trace(go.Bar(
            name=phase,
            x=pivot.index.tolist(),
            y=pivot[phase].tolist(),
            marker_color=color,
            text=[f"{v:.1f}" for v in pivot[phase]],
            textposition="inside",
        ))
    fig.update_layout(
        barmode="stack",
        title=f"Training Step Breakdown — {sel_dev}",
        xaxis_title="Model",
        yaxis_title="Thời gian trung bình (ms)",
        template="plotly_white",
        legend_title="Phase",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_bwd_fwd_ratio(df: pd.DataFrame) -> None:
    """Bar chart: backward/forward ratio to answer CH5."""
    st.subheader("🔄 Tỷ Lệ Backward / Forward (CH5)")
    st.caption("Kỳ vọng: backward ≈ 2× forward. GPU backward = accumulate gradients → ≥ 2×.")

    # Pivot so we can compute ratio
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
                title=f"Backward/Forward Ratio — {dev}",
                color_discrete_sequence=[color],
                text="bwd_fwd_ratio",
                labels={"bwd_fwd_ratio": "Ratio (×)", "model": "Model"},
                template="plotly_white",
            )
            fig.add_hline(y=2.0, line_dash="dot", line_color="gray",
                          annotation_text="Kỳ vọng 2×", annotation_position="top right")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)


def _render_memory_ratio(df_mem: pd.DataFrame) -> None:
    """Show train/infer VRAM ratio from memory_ratio.json."""
    st.subheader("💾 Bộ Nhớ Training vs Inference (CH5)")
    if df_mem.empty:
        st.info("Chưa có file `memory_ratio.json` (chỉ xuất hiện khi chạy trên GPU CUDA).")
        return

    st.caption("Kỳ vọng: bộ nhớ training gấp 2–4× inference (activations + gradients).")
    fig = px.bar(
        df_mem.dropna(subset=["memory_ratio"]).sort_values("memory_ratio", ascending=False),
        x="model", y="memory_ratio", color="device",
        barmode="group",
        text="memory_ratio",
        title="Memory Ratio: Train / Inference Peak VRAM",
        labels={"memory_ratio": "Train/Infer (×)", "model": "Model"},
        color_discrete_map=_DEVICE_COLORS,
        template="plotly_white",
    )
    fig.add_hline(y=2.0, line_dash="dash", line_color="#F59E0B",
                  annotation_text="×2 baseline")
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)


def _render_latency_dist(results_dir: Path) -> None:
    """Box plot: iteration-level time_ms distribution per phase."""
    st.subheader("📦 Phân Phối Thời Gian Theo Giai Đoạn (Box Plot)")

    phase_files = list(results_dir.glob("**/TN6/**/phases.csv"))
    if not phase_files:
        return

    file_opts = {str(p.relative_to(results_dir)): p for p in phase_files}
    sel = st.selectbox("Chọn file phases.csv:", list(file_opts.keys()),
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
        points="all",
        title=f"Phân Phối Thời Gian Theo Phase — {sel}",
        labels={"time_ms": "Thời gian (ms)", "phase": "Giai đoạn"},
        color_discrete_map=_PHASE_COLORS,
        template="plotly_white",
    )
    fig.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# ── Main render function ─────────────────────────────────────────────────────

def render(results_dir: Path) -> None:
    """Render Tab 5: TN6 Training Phase Breakdown — answers CH5."""
    if not HAS_DEPS:
        st.error("Thiếu thư viện: pandas, plotly, streamlit.")
        return

    st.header("🟩 TN6 — Training Phase Breakdown (CH5)")
    st.markdown(
        "**CH5**: Khi training, backward tốn gấp mấy lần forward? "
        "Bộ nhớ tăng bao nhiêu? Nạp dữ liệu có làm GPU phải chờ?"
    )

    df = _load_tn6_phases(results_dir)
    if df.empty:
        st.info(
            "Chưa có dữ liệu TN6. Chạy: "
            "`python scripts/run_tn6_training.py --device colab_t4`"
        )
        return

    # ── KPI metrics row ───────────────────────────────────────────────────────
    st.markdown("---")
    t4 = df[df["device"] == "colab_t4"]
    if not t4.empty:
        t4_piv = t4.pivot_table(index="model", columns="phase", values="mean_ms", aggfunc="mean")
        if "forward" in t4_piv.columns and "backward" in t4_piv.columns:
            ratios = (t4_piv["backward"] / t4_piv["forward"]).dropna()
            avg_ratio = ratios.mean()
            max_model = ratios.idxmax()
            col_k1, col_k2, col_k3 = st.columns(3)
            col_k1.metric("⚡ Backward/Forward Trung Bình (T4)", f"{avg_ratio:.2f}×")
            col_k2.metric("🔝 Model Cao Nhất", max_model,
                          f"{ratios[max_model]:.2f}×")
            col_k3.metric("🔬 Số Model Đo Được", f"{t4['model'].nunique()}")
    st.markdown("---")

    tab_bkd, tab_mem, tab_dist = st.tabs([
        "📊 Phân Rã Giai Đoạn",
        "💾 Bộ Nhớ Train/Infer",
        "📦 Phân Phối Latency",
    ])
    with tab_bkd:
        _render_stacked_breakdown(df)
        st.divider()
        _render_bwd_fwd_ratio(df)
    with tab_mem:
        df_mem = _load_memory_ratio(results_dir)
        _render_memory_ratio(df_mem)
    with tab_dist:
        _render_latency_dist(results_dir)
