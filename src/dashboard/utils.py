"""Plotly visualization helpers and data loading utilities for Streamlit dashboard."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def load_all_run_summaries(results_dir: Path) -> List[Dict[str, Any]]:
    """Scan results directory and return parsed summaries along with path metadata."""
    summaries: List[Dict[str, Any]] = []
    if not results_dir.exists():
        return summaries

    for summary_path in results_dir.glob("**/summary.json"):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            rel_parts = summary_path.relative_to(results_dir).parts
            if len(rel_parts) >= 4:
                data["device"] = rel_parts[0]
                data["model"] = rel_parts[1]
                data["experiment"] = rel_parts[2]
                data["config"] = rel_parts[3]
            data["path"] = str(summary_path.parent)
            summaries.append(data)
        except Exception:
            continue

    return summaries


def make_bar_chart(
    df: Any,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    title: str = "",
) -> Any:
    """Create a standardized Plotly bar chart."""
    if not HAS_PLOTLY or not HAS_PANDAS:
        return None
    fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=title, barmode="group")
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
    return fig


def make_box_plot(df: Any, x_col: str, y_col: str, title: str = "") -> Any:
    """Create a standardized latency distribution box plot."""
    if not HAS_PLOTLY or not HAS_PANDAS:
        return None
    fig = px.box(df, x=x_col, y=y_col, title=title, points="all")
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
    return fig
