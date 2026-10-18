"""Tab 2: CPU vs GPU comparative performance and hardware acceleration [Owned by TV2]."""

from pathlib import Path
from typing import Any, Dict, List

try:
    import pandas as pd
    import plotly.express as px
    import streamlit as st
    from src.dashboard.utils import load_all_run_summaries, make_bar_chart
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


def render(results_dir: Path) -> None:
    """Render Tab 2: CPU Laptop vs Colab T4 GPU comparisons (CH3 & CH4)."""
    if not HAS_DEPS:
        return

    st.header("⚡ So Sánh Phần Cứng: CPU Laptop vs Colab T4 GPU (CH3 & CH4)")
    st.markdown(
        "Đối đầu hiệu năng giữa CPU Laptop và Google Colab Tesla T4 GPU. Phân tích hiện tượng "
        "**đảo thứ hạng mô hình (CH3)**, **đường cong bão hòa batch size (CH4)** và **tăng tốc Tensor Cores FP16**."
    )

    summaries = load_all_run_summaries(results_dir)
    if not summaries:
        st.info("Chưa tìm thấy dữ liệu để so sánh CPU vs GPU trong `results/`.")
        return

    df = pd.DataFrame(summaries)
    if "device" not in df.columns or "model" not in df.columns:
        st.warning("Dữ liệu thiếu trường bắt buộc 'device' hoặc 'model'.")
        return

    # Section 1: CH3 - Đối Đầu Cơ Bản & Hệ Số Tăng Tốc (TN3)
    st.subheader("1. Đối Đầu Cơ Bản & Hệ Số Tăng Tốc (TN3: Batch=1, FP32)")
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

                st.dataframe(pivot_lat, use_container_width=True)

                fig_speedup = px.bar(
                    pivot_lat,
                    x="model",
                    y="Speedup_T4_vs_CPU",
                    text="Speedup_T4_vs_CPU",
                    title="CH3: Hệ số tăng tốc phần cứng (Speedup = Latency_CPU / Latency_GPU)",
                    labels={"Speedup_T4_vs_CPU": "Hệ số tăng tốc (Lần)", "model": "Mô hình"},
                    color="Speedup_T4_vs_CPU",
                    color_continuous_scale="Blues",
                    template="plotly_white",
                )
                fig_speedup.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Parity (1.0x)")
                fig_speedup.update_traces(texttemplate="%{text}x", textposition="outside")
                st.plotly_chart(fig_speedup, use_container_width=True)

                st.info(
                    "💡 **Nhận xét hiện tượng Đảo Thứ Hạng & Nghịch lý Tham số (CH3):**\n"
                    "- **Ở batch 1 trên T4, FLOPs/Tham số không quyết định độ trễ:** ResNet-50 (8.2 GFLOPs) và MobileNetV3 (0.43 GFLOPs, ít hơn 19 lần) "
                    "cùng mất ~6.0 ms do GPU bị giới hạn bởi chi phí gửi lệnh kernel (Kernel Launch Latency).\n"
                    "- **Mô hình siêu nhẹ chạy GPU chậm hơn CPU:** MLP trên GPU mất 0.28 ms trong khi trên CPU Mac chỉ mất 0.04 ms (chậm hơn 7 lần) "
                    "do chi phí truyền dữ liệu bus PCIe vượt quá thời gian tính toán.\n"
                    "- **MobileNetV3 trên CPU ARM:** Độ trễ 176.3 ms do kernel PyTorch thiếu tối ưu vector hóa cho depthwise convolution trên ARM (trên CPU x86 chỉ ~12 ms)."
                )
        except Exception as exc:
            st.error(f"Lỗi tính toán bảng so sánh TN3: {exc}")

    # Section 2: CH4 - Đường Cong Bão Hòa Khi Tăng Batch (TN4)
    st.subheader("2. Mở Rộng Kích Thước Lô & Điểm Bão Hòa (TN4: Batch 1, 8, 32)")
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
                title="CH4: Thông Lượng (Throughput FPS) theo Batch Size giữa CPU và GPU",
                labels={"batch_size": "Batch Size", "throughput_fps": "Throughput (FPS)"},
                template="plotly_white",
            )
            st.plotly_chart(fig_batch, use_container_width=True)
            st.caption("Ghi chú: Nét liền đại diện cho Colab T4, nét đứt đại diện cho CPU Laptop.")
        except Exception as exc:
            st.error(f"Lỗi vẽ biểu đồ TN4: {exc}")

    # Section 3: CH4 - Tăng Tốc Tensor Cores FP32 vs FP16 (TN5)
    st.subheader("3. Tăng Tốc Số Học FP16 Tensor Cores (TN5: Colab T4)")
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
            st.dataframe(df_fp16, use_container_width=True)
            st.info(
                "💡 **Kết luận TN5 (FP16 Tensor Cores trên T4):**\n"
                "- **Tiết kiệm VRAM:** FP16 cắt giảm 38%–49% Peak VRAM cho toàn bộ các mô hình có tham số lớn.\n"
                "- **Tăng tốc chọn lọc:** FP16 tăng tốc vượt trội cho Transformer (ViT-B/16 tăng **2.81x**, DistilBERT tăng 1.55x) "
                "do chứa các ma trận GEMM lớn lấp đầy Tensor Cores. Với CNN ở batch 1 (ResNet 0.87x, YOLO 0.77x), "
                "mô hình bị chậm hơn vì khối tính toán nhỏ không bù đắp được chi phí gửi kernel và chuyển đổi kiểu dữ liệu.\n"
                "- *Cosine similarity của ViT-B/16 và MobileNetV3 không xác định do khởi tạo ngẫu nhiên cho output xấp xỉ 0."
            )
    except Exception as exc:
        st.error(f"Lỗi tính toán bảng TN5 (FP16): {exc}")
