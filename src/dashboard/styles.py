"""Kiểu trình bày thống nhất cho dashboard, không cần thư viện ngoài."""

import streamlit as st

CSS = """
<style>
:root {--vp-accent: #2456A6; --vp-border: #E2E8F0; --vp-muted: #607089;}
.block-container {max-width: 1400px; padding: 2.8rem 2.6rem 3rem;}
[data-testid="stSidebar"] {background: #FFFFFF; border-right: 1px solid var(--vp-border);}
[data-testid="stSidebarNav"] {padding-top: 1rem;}
[data-testid="stSidebarNav"] a {border-radius: 8px; margin: 3px 0; padding: .55rem .75rem;}
[data-testid="stSidebarNav"] a[aria-current="page"] {
    background: #EDF3FC; border-left: 3px solid var(--vp-accent); font-weight: 650;}
[data-testid="stSidebarNav"] a:hover {background: #F0F4FA;}
[data-testid="stSidebarNavSeparator"] {border-color: var(--vp-border);}
[data-testid="stMetric"] {border: 1px solid var(--vp-border);
    border-top: 3px solid var(--vp-accent); border-radius: 12px;
    padding: 18px 20px; background: #FFFFFF; min-height: 132px;
    box-shadow: 0 3px 12px rgba(27, 48, 79, .035);}
[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;
    font-size: 1.85rem; font-weight: 650; letter-spacing: -.035em;}
[data-testid="stMetricLabel"] {color: var(--vp-muted); font-size: .84rem;}
h1 {font-size: 2rem !important; font-weight: 700 !important;
    letter-spacing: -.035em; padding-top: .1rem !important; padding-bottom: .4rem !important;}
h2 {font-size: 1.25rem !important; letter-spacing: -.02em;}
h3 {font-size: 1.1rem !important;}
h5 {font-size: 1rem !important;}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {color: var(--vp-muted) !important;}
[data-testid="stPlotlyChart"] {background: #FFFFFF; border: 1px solid var(--vp-border);
    border-radius: 12px; padding: 0; overflow: hidden; box-shadow: 0 3px 12px rgba(27,48,79,.025);}
[data-testid="stDataFrame"], [data-testid="stExpander"] {border-radius: 10px;}
[data-testid="stTabs"] [role="tablist"] {gap: 8px; border-bottom: 1px solid var(--vp-border);}
[data-testid="stTabs"] [role="tab"] {padding: .6rem 1rem; font-weight: 550;}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--vp-accent); background: #FFFFFF; border-radius: 8px 8px 0 0;}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {background-color: var(--vp-accent);}
[data-testid="stTabs"] [role="tabpanel"] {padding-top: 1.2rem;}
[data-baseweb="select"] > div {background: #FFFFFF; border-color: #DCE3ED; border-radius: 8px;}
.vp-level {font-size: .74rem; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; margin-bottom: .15rem; display: flex; align-items: center; gap: 8px;}
.vp-level::before {content: ''; width: 7px; height: 7px; border-radius: 50%; background: currentColor;}
.vp-sidebar-note {border-top: 1px solid var(--vp-border); padding-top: 1rem;
    margin-top: 1.2rem; font-size: .8rem; line-height: 1.8; color: var(--vp-muted);}
.vp-legend {display: flex; gap: 14px; margin-top: 8px; font-size: .75rem;}
.vp-legend span {border-bottom: 2px solid; padding-bottom: 2px;}
@media (max-width: 900px) {.block-container {padding: 2.5rem 1rem 2rem;}}
</style>
"""


def apply_styles() -> None:
    """Áp dụng bố cục, điều hướng và thẻ số cho mọi trang."""
    st.markdown(CSS, unsafe_allow_html=True)
