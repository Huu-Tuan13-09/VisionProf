"""Dashboard tabs package containing view modules for VisionProf."""

from src.dashboard.tabs.tab_ab_test import render as render_ab_test
from src.dashboard.tabs.tab_cpu_vs_gpu import render as render_cpu_vs_gpu
from src.dashboard.tabs.tab_diagnostics import render as render_diagnostics
from src.dashboard.tabs.tab_layer import render as render_layer
from src.dashboard.tabs.tab_overview import render as render_overview
from src.dashboard.tabs.tab_roofline import render as render_roofline
from src.dashboard.tabs.tab_training import render as render_training

__all__ = [
    "render_overview",
    "render_cpu_vs_gpu",
    "render_layer",
    "render_roofline",
    "render_training",
    "render_diagnostics",
    "render_ab_test",
]
