"""Tab 6: Heuristic diagnostic rules v2.0 assessment report [Owned by TV3]."""

from pathlib import Path
from typing import Any

try:
    import pandas as pd
    import streamlit as st
    from src.analyzer.engine import DiagnosticEngine
except ImportError:
    pass


def render(results_dir: Path) -> None:
    """Render Tab 6: 19 heuristic rules diagnostic report."""
    st.header("🩺 Diagnostic Engine v2.0 (19 Heuristic Rules)")
    st.markdown("Hệ thống cảnh báo tự động phát hiện nghẽn cổ chai bộ nhớ, tính toán, và kiến trúc.")

    subdirs = [p for p in results_dir.glob("*/*/*/*") if p.is_dir()]
    if not subdirs:
        st.info("Chưa có thư mục kết quả trong `results/`.")
        return

    selected_dir = st.selectbox("Chọn thư mục thực nghiệm để chẩn đoán:", subdirs, format_func=lambda p: str(p.relative_to(results_dir)))
    if not selected_dir:
        return

    engine = DiagnosticEngine()
    results = engine.evaluate_directory(selected_dir)

    records = [r.to_dict() for r in results]
    df = pd.DataFrame(records)

    # Status summary
    pass_cnt = sum(1 for r in results if r.status.value == "PASS")
    warn_cnt = sum(1 for r in results if r.status.value == "WARN")
    fail_cnt = sum(1 for r in results if r.status.value == "FAIL")
    skip_cnt = sum(1 for r in results if r.status.value == "SKIPPED")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PASS", pass_cnt)
    c2.metric("WARN", warn_cnt)
    c3.metric("FAIL", fail_cnt)
    c4.metric("SKIPPED", skip_cnt)

    st.subheader("Chi Tiết Từng Quy Tắc")
    st.dataframe(df[["rule_id", "status", "title", "message"]], use_container_width=True)
