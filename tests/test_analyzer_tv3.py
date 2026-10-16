"""Unit tests for TV3 Analyzer and Integration (Part B).

Covers: DiagnosticEngine (19 rules), ABComparisonEngine, PhaseTimer+ModelZoo integration
"""

import csv
import json
import tempfile
from pathlib import Path

import pytest

from src.analyzer.ab_testing import ABComparisonEngine
from src.analyzer.engine import DiagnosticEngine
from src.core.constants import RuleStatus
from src.phase.timer import PhaseTimer


# ─── DiagnosticEngine Tests ─────────────────────────────────────────────────────

class TestDiagnosticEngine:
    """Tests for DiagnosticEngine evaluate_directory() with various artifact combinations."""

    def test_empty_dir_returns_19_rules(self) -> None:
        """All 19 rules must be returned even when no artifacts exist."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            engine = DiagnosticEngine()
            results = engine.evaluate_directory(Path(tmp_dir))
        assert len(results) == 19

    def test_missing_layers_csv_skipped(self) -> None:
        """Layer-dependent rules must be SKIPPED when layers.csv is absent."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            engine = DiagnosticEngine()
            results = engine.evaluate_directory(Path(tmp_dir))
        skipped = {r.rule_id for r in results if r.status == RuleStatus.SKIPPED}
        for rule_id in ("RA-03", "RB-04", "RD-01", "RD-02", "RD-03", "RD-04", "RD-05"):
            assert rule_id in skipped, f"{rule_id} should be SKIPPED"

    def test_missing_phases_csv_skipped(self) -> None:
        """Phase-dependent rules must be SKIPPED when phases.csv is absent."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            engine = DiagnosticEngine()
            results = engine.evaluate_directory(Path(tmp_dir))
        skipped = {r.rule_id for r in results if r.status == RuleStatus.SKIPPED}
        for rule_id in ("RB-03", "RB-05", "RC-01", "RC-02", "RC-03"):
            assert rule_id in skipped, f"{rule_id} should be SKIPPED"

    def test_high_tail_ratio_triggers_rb02_fail(self) -> None:
        """With P99/P50 > 2.0, RB-02 should FAIL."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary = {"p50_ms": 10.0, "p99_ms": 30.0, "throughput_fps": 100.0}
            (Path(tmp_dir) / "summary.json").write_text(json.dumps(summary))
            engine = DiagnosticEngine()
            results = engine.evaluate_directory(Path(tmp_dir))
        rb02 = next(r for r in results if r.rule_id == "RB-02")
        assert rb02.status == RuleStatus.FAIL

    def test_phases_csv_heavy_backward_triggers_rb03_warn(self) -> None:
        """phases.csv with backward > 2.5x forward should trigger RB-03 WARN."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            with open(run_dir / "phases.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["iteration", "phase", "time_ms"])
                writer.writerow([0, "forward", 10.0])
                writer.writerow([0, "backward", 35.0])   # ratio 3.5 > 2.5
                writer.writerow([0, "optimizer", 5.0])
            engine = DiagnosticEngine()
            results = engine.evaluate_directory(run_dir)
        rb03 = next(r for r in results if r.rule_id == "RB-03")
        assert rb03.status == RuleStatus.WARN

    def test_phases_csv_low_io_passes_rc01(self) -> None:
        """Low I/O ratio (<20%) should pass RC-01."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            with open(run_dir / "phases.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["iteration", "phase", "time_ms"])
                writer.writerow([0, "dataloader", 5.0])    # 5/105 ≈ 4.7% < 20%
                writer.writerow([0, "forward", 90.0])
                writer.writerow([0, "backward", 10.0])
            engine = DiagnosticEngine()
            results = engine.evaluate_directory(run_dir)
        rc01 = next(r for r in results if r.rule_id == "RC-01")
        assert rc01.status == RuleStatus.PASS


# ─── ABComparisonEngine Tests ────────────────────────────────────────────────────

class TestABComparisonEngine:
    """Tests for A/B comparison speedup, Cliff's delta, and Mann-Whitney U."""

    def test_speedup_2x(self) -> None:
        """Baseline 2x slower yields speedup = 2.0."""
        baseline = [100.0] * 10
        optimized = [50.0] * 10
        speedup = ABComparisonEngine.compute_speedup(baseline, optimized)
        assert speedup == pytest.approx(2.0)

    def test_speedup_equal_configs(self) -> None:
        """Equal configurations must yield speedup = 1.0."""
        latencies = [80.0] * 10
        speedup = ABComparisonEngine.compute_speedup(latencies, latencies)
        assert speedup == pytest.approx(1.0)

    def test_speedup_empty_list_returns_one(self) -> None:
        """Empty lists must not crash — return 1.0."""
        assert ABComparisonEngine.compute_speedup([], []) == 1.0

    def test_cliffs_delta_perfect_separation(self) -> None:
        """All A > all B gives Cliff's delta = 1.0."""
        a = [100.0, 110.0, 120.0]
        b = [10.0, 20.0, 30.0]
        delta = ABComparisonEngine.compute_cliffs_delta(a, b)
        assert delta == pytest.approx(1.0)

    def test_cliffs_delta_equal_samples(self) -> None:
        """Equal samples give delta = 0.0."""
        data = [50.0, 60.0, 70.0]
        delta = ABComparisonEngine.compute_cliffs_delta(data, data)
        assert delta == pytest.approx(0.0)

    def test_compare_runs_returns_required_keys(self) -> None:
        """compare_runs() must return all required result keys."""
        a = [100.0] * 20
        b = [50.0] * 20
        result = ABComparisonEngine.compare_runs(a, b)
        required_keys = {
            "speedup", "cliffs_delta", "mann_whitney_u",
            "p_value", "statistically_significant",
            "sample_count_a", "sample_count_b",
        }
        assert required_keys.issubset(result.keys())
        assert result["speedup"] == pytest.approx(2.0)
        assert result["statistically_significant"] is True

    def test_compare_runs_few_samples_no_crash(self) -> None:
        """compare_runs() with < 5 samples must not crash."""
        result = ABComparisonEngine.compare_runs([10.0, 20.0], [5.0, 8.0])
        assert "speedup" in result
        assert result["p_value"] == pytest.approx(1.0)


# ─── Integration: PhaseTimer + ModelZoo ─────────────────────────────────────────

class TestPhaseTimerWithModelZoo:
    """Integration tests: PhaseTimer correctly times ModelZoo adapters on CPU."""

    def test_phase_timer_with_mlp_adapter(self) -> None:
        """PhaseTimer measure() works correctly around MLPAdapter forward."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")

        from src.zoo.tabular import MLPAdapter
        adapter = MLPAdapter()
        model = adapter.load(device="cpu")
        inputs = adapter.make_input(batch_size=4, device="cpu")

        timer = PhaseTimer(device="cpu")
        timer.set_iteration(0)
        model.eval()
        with torch.inference_mode():
            with timer.measure("forward"):
                _ = adapter.forward(model, inputs)

        records = timer.get_records()
        assert len(records) == 1
        assert records[0].phase == "forward"
        assert records[0].time_ms > 0.0

    def test_phase_timer_full_training_step_mlp(self) -> None:
        """PhaseTimer captures forward/backward/optimizer phases during MLP train_step."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")

        from src.zoo.tabular import MLPAdapter
        adapter = MLPAdapter()
        model = adapter.load(device="cpu")
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        inputs = adapter.make_input(batch_size=8, device="cpu")
        targets = torch.randint(0, 7, (8,))

        timer = PhaseTimer(device="cpu")
        timer.set_iteration(0)

        adapter.train_step(model, (inputs, targets), optimizer, timer=timer)

        records = timer.get_records()
        phase_names = {r.phase for r in records}
        assert "forward" in phase_names
        assert "backward" in phase_names
        assert "optimizer" in phase_names
        for r in records:
            assert r.time_ms > 0.0
