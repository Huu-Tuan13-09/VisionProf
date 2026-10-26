"""Tab 2: CPU vs GPU comparative performance and hardware acceleration [Owned by TV2]."""

from pathlib import Path

from src.dashboard.utils import page_header, show_chart

try:
    import pandas as pd
    import plotly.express as px
    import streamlit as st
    from src.dashboard.utils import load_all_run_summaries
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


def render(results_dir: Path) -> None:
    """Render Tab 2: CPU Laptop vs Colab T4 GPU comparisons (CH3 & CH4)."""
    if not HAS_DEPS:
        return

    page_header("CPU và T4")

    summaries = load_all_run_summaries(results_dir)
    if not summaries:
        st.info("Chưa tìm thấy dữ liệu để so sánh CPU vs GPU trong `results/`.")
        return

    df = pd.DataFrame(summaries)
    if "device" not in df.columns or "model" not in df.columns:
        st.warning("Dữ liệu thiếu trường bắt buộc 'device' hoặc 'model'.")
        return

    tab_cpu, tab_batch, tab_fp16 = st.tabs(["CPU và T4", "Batch size", "FP16"])

    with tab_cpu:
        st.subheader("Tăng tốc phần cứng · TN3")
        tn3_df = df[df["experiment"] == "TN3"]
        if tn3_df.empty:
            tn3_df = df[df["config"] == "bs1_fp32"]

        if not tn3_df.empty:
            try:
                pivot_lat = tn3_df.pivot_table(index="model", columns="device", values="p50_ms", aggfunc="mean").reset_index()
                cpu_col = "cpu_laptop" if "cpu_laptop" in pivot_lat.columns else ("colab_cpu" if "colab_cpu" in pivot_lat.columns else None)
                gpu_col = "colab_t4" if "colab_t4" in pivot_lat.columns else None

                if cpu_col and gpu_col:
                    pivot_lat["Speedup_T4_vs_CPU"] = (pivot_lat[cpu_col] / pivot_lat[gpu_col]).round(2)
                    pivot_lat = pivot_lat.sort_values(by="Speedup_T4_vs_CPU", ascending=False)

                    with st.expander("Xem bảng số · CPU và T4"):
                        st.dataframe(pivot_lat, width="stretch", hide_index=True)

                    fig_speedup = px.bar(
                        pivot_lat,
                        x="model",
                        y="Speedup_T4_vs_CPU",
                        text="Speedup_T4_vs_CPU",
                        title="Tăng tốc T4 so với CPU · batch 1, FP32",
                        labels={"Speedup_T4_vs_CPU": "Tăng tốc (×)", "model": "Mô hình"},
                        color="Speedup_T4_vs_CPU",
                        color_continuous_scale="Blues",
                        template="plotly_white",
                    )
                    fig_speedup.add_hline(y=1.0, line_dash="dash", line_color="#64748B", annotation_text="Ngang nhau (1×)")
                    fig_speedup.update_traces(texttemplate="%{text}x", textposition="outside")
                    show_chart(fig_speedup)

            except Exception as exc:
                st.error(f"Lỗi tính toán bảng so sánh TN3: {exc}")

    with tab_batch:
        st.subheader("Thông lượng theo batch size · TN4")
        tn4_df = df[df["experiment"] == "TN4"].copy()
        if not tn4_df.empty:
            try:
                # Extract batch number from config (e.g. bs8_fp32 -> 8, bs8_seq256_fp32 -> 8)
                tn4_df["batch_size"] = tn4_df["config"].str.extract(r"bs(\d+)_").astype(float)
                seq_suffix = tn4_df["config"].str.extract(r"(_seq\d+)_")[0].fillna("")
                tn4_df["model_label"] = tn4_df["model"] + seq_suffix
                fig_batch = px.line(
                    tn4_df.sort_values(by="batch_size"),
                    x="batch_size",
                    y="throughput_fps",
                    color="model_label",
                    line_dash="device",
                    markers=True,
                    title="Thông lượng theo batch size",
                    labels={"batch_size": "Batch size", "throughput_fps": "Thông lượng (FPS)",
                            "model_label": "Mô hình", "device": "Thiết bị"},
                    template="plotly_white",
                )
                show_chart(fig_batch)
            except Exception as exc:
                st.error(f"Lỗi vẽ biểu đồ TN4: {exc}")

    with tab_fp16:
        st.subheader("FP16 trên T4 · TN5")
        try:
            t4_data = df[df["device"] == "colab_t4"]
            t4_fp32 = (
                t4_data[(t4_data["experiment"] == "TN3") & (t4_data["config"] == "bs1_fp32")]
                .drop_duplicates(subset=["model"])
                .set_index("model")
            )
            t4_fp16 = (
                t4_data[(t4_data["experiment"] == "TN5") & (t4_data["config"] == "bs1_fp16")]
                .drop_duplicates(subset=["model"])
                .set_index("model")
            )

            fp16_rows = []
            for m in t4_fp16.index:
                if m in t4_fp32.index:
                    p32 = float(t4_fp32.loc[m, "p50_ms"])
                    p16 = float(t4_fp16.loc[m, "p50_ms"])
                    v32 = float(t4_fp32.loc[m, "peak_vram_mb"])
                    v16 = float(t4_fp16.loc[m, "peak_vram_mb"])
                    speedup_fp16 = round(p32 / p16, 2) if p16 > 0 else 1.0
                    vram_saved = round((1.0 - v16 / v32) * 100.0, 1) if v32 > 0 else 0.0

                    extra = t4_fp16.loc[m, "extra"] if "extra" in t4_fp16.columns else {}
                    cos_val = extra.get("cosine_similarity", None) if isinstance(extra, dict) else None
                    cos_str = f"{cos_val:.6f}" if isinstance(cos_val, (float, int)) and cos_val > 0 else "Không xác định*"

                    fp16_rows.append({
                        "model": m,
                        "FP32_p50_ms": round(p32, 2),
                        "FP16_p50_ms": round(p16, 2),
                        "Speedup_FP16": f"{speedup_fp16}x",
                        "VRAM_FP32_mb": round(v32, 1),
                        "VRAM_FP16_mb": round(v16, 1),
                        "VRAM_Saved_%": f"-{vram_saved:.0f}%",
                        "Cosine_Sim": cos_str,
                    })

            if fp16_rows:
                df_fp16 = pd.DataFrame(fp16_rows)
                st.dataframe(df_fp16, width="stretch", hide_index=True, column_config={
                    "model": "Mô hình", "FP32_p50_ms": "P50 FP32 (ms)", "FP16_p50_ms": "P50 FP16 (ms)",
                    "Speedup_FP16": "Tăng tốc (×)", "VRAM_FP32_mb": "VRAM FP32 (MB)",
                    "VRAM_FP16_mb": "VRAM FP16 (MB)", "VRAM_Saved_%": "Giảm VRAM (%)", "Cosine_Sim": "Cosine",
                })
                st.caption("Cosine không xác định khi output xấp xỉ 0.")
        except Exception as exc:
            st.error(f"Lỗi tính toán bảng TN5 (FP16): {exc}")
