"""Analyzer module for 19 heuristic rules and A/B statistical testing."""

import importlib.util
import sys
from pathlib import Path

from src.analyzer.ab_testing import ABComparisonEngine
from src.analyzer.engine import DiagnosticEngine

__all__ = [
    "DiagnosticEngine",
    "ABComparisonEngine",
    "RuleCatalog",
    "AnalysisReport",
    "Severity",
    "analyze_from_csv",
    "compare_ab_from_csv",
]

# Backward compatibility bridge with legacy src/analyzer.py
RuleCatalog = None  # type: ignore
AnalysisReport = None  # type: ignore
Severity = None  # type: ignore
analyze_from_csv = None  # type: ignore
compare_ab_from_csv = None  # type: ignore

_legacy_path = Path(__file__).resolve().parent.parent / "analyzer.py"
if _legacy_path.exists():
    _spec = importlib.util.spec_from_file_location("src.analyzer_legacy", _legacy_path)
    if _spec and _spec.loader:
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules[_spec.name] = _mod
        try:
            _spec.loader.exec_module(_mod)
            RuleCatalog = getattr(_mod, "RuleCatalog", None)
            AnalysisReport = getattr(_mod, "AnalysisReport", None)
            Severity = getattr(_mod, "Severity", None)
            analyze_from_csv = getattr(_mod, "analyze_from_csv", None)
            compare_ab_from_csv = getattr(_mod, "compare_ab_from_csv", None)
        except Exception:
            pass

