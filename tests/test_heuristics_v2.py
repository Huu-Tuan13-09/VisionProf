"""Unit tests for TV3 19 Heuristics v2.0 and A/B Testing Engine."""

import tempfile
from pathlib import Path

from src.analyzer.ab_testing import ABComparisonEngine
from src.analyzer.engine import DiagnosticEngine
from src.core.constants import RuleStatus


def test_missing_layers_csv_returns_skipped():
    """Verify missing layers.csv marks layer rules as SKIPPED instead of crashing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        run_dir = Path(tmp_dir)
        engine = DiagnosticEngine()
        results = engine.evaluate_directory(run_dir)

        # Ensure all 19 rules produce an evaluation result
        assert len(results) == 19

        # Layer-dependent rules must be marked SKIPPED
        skipped_rule_ids = {r.rule_id for r in results if r.status == RuleStatus.SKIPPED}
        assert "RA-03" in skipped_rule_ids
        assert "RB-04" in skipped_rule_ids
        assert "RD-01" in skipped_rule_ids


def test_ab_speedup_calculation():
    """Verify Speedup A/B is computed from overall model latencies."""
    baseline = [100.0, 100.0, 100.0]
    optimized = [50.0, 50.0, 50.0]

    speedup = ABComparisonEngine.compute_speedup(baseline, optimized)
    assert speedup == 2.0

    delta = ABComparisonEngine.compute_cliffs_delta(baseline, optimized)
    assert delta == 1.0  # Perfect positive difference
