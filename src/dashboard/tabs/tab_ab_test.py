"""Tab 7: A/B comparison and statistical hypothesis testing [Owned by TV3]."""

from pathlib import Path
from typing import Any

try:
    import pandas as pd
    import streamlit as st
    from src.analyzer.ab_testing import ABComparisonEngine
    from src.dashboard.utils import make_box_plot
except ImportError:
    pass


def render(results_dir: Path) -> None:
    """Render Tab 7: Rigorous A/B comparison with Mann-Whitney U test."""
    st.header("🧪 A/B Comparison & Statistical Significance")
    st.markdown("So sánh hai cấu hình (ví dụ: FP32 vs FP16, Batch 1 vs Batch 32) với kiểm định Mann-Whitney U & Cliff's Delta.")

    bench_files = list(results_dir.glob("**/benchmark.csv"))
    if len(bench_files) < 2:
        st.info("Cần ít nhất 2 file `benchmark.csv` trong `results/` để thực hiện kiểm định A/B.")
        return

    col1, col2 = st.columns(2)
    with col1:
        run_a = st.selectbox("Chọn Cấu hình A (Baseline):", bench_files, key="run_a", format_func=lambda p: str(p.parent.relative_to(results_dir)))
    with col2:
        run_b = st.selectbox("Chọn Cấu hình B (Optimized):", bench_files, key="run_b", format_func=lambda p: str(p.parent.relative_to(results_dir)))

    if run_a and run_b:
        if run_a.stat().st_size == 0 or run_b.stat().st_size == 0:
            st.warning("Một trong hai file benchmark được chọn bị rỗng.")
            return

        try:
            df_a = pd.read_csv(run_a)
            df_b = pd.read_csv(run_b)
            if "latency_ms" not in df_a.columns or "latency_ms" not in df_b.columns:
                st.warning("File benchmark thiếu cột 'latency_ms'.")
                return

            lats_a = df_a["latency_ms"].dropna().tolist()
            lats_b = df_b["latency_ms"].dropna().tolist()
            if not lats_a or not lats_b:
                st.warning("Không có dữ liệu latency hợp lệ để so sánh.")
                return

            res = ABComparisonEngine.compare_runs(lats_a, lats_b)

            c1, c2, c3 = st.columns(3)
            c1.metric("Overall Speedup", f"{res['speedup']}x")
            c2.metric("Cliff's Delta", f"{res['cliffs_delta']}")
            c3.metric("p-value", f"{res['p_value']:.4e}")

            # Combined box plot
            df_a["Variant"] = "A (Baseline)"
            df_b["Variant"] = "B (Optimized)"
            combined = pd.concat([df_a, df_b])
            fig_box = make_box_plot(combined, x_col="Variant", y_col="latency_ms", title="Phân Phối Độ Trễ A vs B (ms)")
            if fig_box:
                st.plotly_chart(fig_box, use_container_width=True)
        except Exception as exc:
            st.error(f"Lỗi khi xử lý dữ liệu A/B test: {exc}")
