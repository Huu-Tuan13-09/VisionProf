"""Tab 7: Kết quả và khuyến nghị từ 19 luật chẩn đoán."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.analyzer.engine import DiagnosticEngine
from src.dashboard.tabs.diagnostic_text import RULE_LABELS, STATUS_LABELS, translate_message
from src.dashboard.utils import page_header


def render(results_dir: Path) -> None:
    """Chọn lượt đo, xem một luật và tra bảng đầy đủ của engine."""
    page_header("Chẩn đoán", "phase")
    subdirs = [p for p in results_dir.glob("*/*/*/*") if p.is_dir()]
    if not subdirs:
        st.info("Chưa có lượt đo để chẩn đoán.")
        return

    selected_dir = st.selectbox("Chọn lượt đo", subdirs,
                                format_func=lambda p: str(p.relative_to(results_dir)), key="diagnostic_run")
    if not selected_dir:
        return
    results = DiagnosticEngine().evaluate_directory(selected_dir)
    if not results:
        st.info("Lượt đo này chưa có kết quả chẩn đoán.")
        return

    for col, status in zip(st.columns(4), STATUS_LABELS):
        count = sum(r.status.value == status for r in results)
        col.metric(f"{STATUS_LABELS[status]} · {status}", count)

    rules = {r.rule_id: r for r in results}
    rule_ids = sorted(rules)
    selected_rule = st.selectbox(
        "Xem luật", rule_ids, index=rule_ids.index("RB-04") if "RB-04" in rule_ids else 0,
        format_func=lambda rid: f"{rid} · {RULE_LABELS.get(rid, rid)}", key="diagnostic_rule",
    )
    rule = rules[selected_rule]
    with st.container(border=True):
        st.subheader(f"{rule.rule_id} · {RULE_LABELS.get(rule.rule_id, rule.title)}")
        status = rule.status.value
        message = f"**{STATUS_LABELS[status]} ({status})** · {translate_message(rule.message)}"
        presenter = {"PASS": st.success, "WARN": st.warning, "FAIL": st.error, "SKIPPED": st.info}[status]
        presenter(message)
        if rule.recommendation:
            st.markdown(f"**Đề xuất:** {rule.recommendation}")

    actionable = [r for r in results if r.status.value in ("WARN", "FAIL") and r.recommendation]
    if actionable:
        st.subheader("Các điểm cần chú ý")
        for r in actionable:
            with st.expander(f"{r.rule_id} · {RULE_LABELS.get(r.rule_id, r.title)} · {STATUS_LABELS[r.status.value]}"):
                st.write(translate_message(r.message))
                st.markdown(f"**Đề xuất:** {r.recommendation}")

    with st.expander("Xem bảng số · toàn bộ 19 luật"):
        records = [{"Mã luật": r.rule_id, "Trạng thái": f"{STATUS_LABELS[r.status.value]} ({r.status.value})",
                    "Nội dung": RULE_LABELS.get(r.rule_id, r.title), "Kết quả": translate_message(r.message),
                    "Đề xuất": r.recommendation} for r in results]
        st.dataframe(pd.DataFrame(records), width="stretch", hide_index=True)
