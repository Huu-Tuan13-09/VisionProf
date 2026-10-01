"""Main Streamlit application entrypoint for VisionProf dashboard."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import streamlit as st
    from src.dashboard.tabs import (
        render_ab_test,
        render_cpu_vs_gpu,
        render_diagnostics,
        render_layer,
        render_overview,
        render_roofline,
        render_training,
    )
except ImportError:
    pass


def main() -> None:
    """Run Streamlit dashboard application."""
    st.set_page_config(page_title="VisionProf Dashboard", layout="wide", page_icon="⚡")
    st.sidebar.title("⚡ VisionProf")
    st.sidebar.markdown("**Deep Learning Profiling & Diagnostic Framework**")

    results_dir = PROJECT_ROOT / "results"
    st.sidebar.markdown(f"**Results Directory**: `{results_dir}`")

    tab_names = [
        "1. Overview (8 Models)",
        "2. Colab CPU vs T4 GPU",
        "3. Leaf Layer Breakdown",
        "4. Roofline Model",
        "5. Training & Pipeline",
        "6. Diagnostic Engine (19 Rules)",
        "7. A/B Testing & Speedup",
    ]

    choice = st.sidebar.radio("Navigation", tab_names)

    if choice == tab_names[0]:
        render_overview(results_dir)
    elif choice == tab_names[1]:
        render_cpu_vs_gpu(results_dir)
    elif choice == tab_names[2]:
        render_layer(results_dir)
    elif choice == tab_names[3]:
        render_roofline(results_dir)
    elif choice == tab_names[4]:
        render_training(results_dir)
    elif choice == tab_names[5]:
        render_diagnostics(results_dir)
    elif choice == tab_names[6]:
        render_ab_test(results_dir)


if __name__ == "__main__":
    main()
