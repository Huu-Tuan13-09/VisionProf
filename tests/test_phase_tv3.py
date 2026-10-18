"""Unit tests for TV3 Phase Profiler core components (Part A).

Covers: PhaseTimer, DataLoaderSentinel, PhaseProfilerPlugin
"""

import csv
import tempfile
import time
from pathlib import Path

import pytest

from src.phase.dataloader import DataLoaderSentinel
from src.phase.plugin import PhaseProfilerPlugin
from src.phase.timer import PhaseTimer
from src.core.types import PhaseRecord


# ─── PhaseTimer Tests ──────────────────────────────────────────────────────────

class TestPhaseTimer:
    """Tests for PhaseTimer context manager and summary logic."""

    def test_measure_records_positive_duration(self) -> None:
        """measure() must record a positive elapsed time."""
        timer = PhaseTimer(device="cpu")
        timer.set_iteration(0)
        with timer.measure("forward"):
            time.sleep(0.005)
        records = timer.get_records()
        assert len(records) == 1
        assert records[0].phase == "forward"
        assert records[0].time_ms > 0.0
        assert records[0].iteration == 0

    def test_measure_multiple_phases(self) -> None:
        """All phase names are recorded in order."""
        timer = PhaseTimer(device="cpu")
        timer.set_iteration(1)
        phases = ["dataloader", "forward", "backward", "optimizer"]
        for p in phases:
            with timer.measure(p):
                pass
        records = timer.get_records()
        assert len(records) == 4
        assert [r.phase for r in records] == phases

    def test_set_iteration_updates_records(self) -> None:
        """Records carry the correct iteration index."""
        timer = PhaseTimer(device="cpu")
        for i in range(3):
            timer.set_iteration(i)
            with timer.measure("forward"):
                pass
        records = timer.get_records()
        assert [r.iteration for r in records] == [0, 1, 2]

    def test_summary_returns_mean_per_phase(self) -> None:
        """summary() computes mean time_ms per phase across iterations."""
        timer = PhaseTimer(device="cpu")
        timer.add_record(PhaseRecord(iteration=0, phase="forward", time_ms=10.0))
        timer.add_record(PhaseRecord(iteration=1, phase="forward", time_ms=20.0))
        timer.add_record(PhaseRecord(iteration=0, phase="backward", time_ms=30.0))
        s = timer.summary()
        assert s["forward"] == pytest.approx(15.0)
        assert s["backward"] == pytest.approx(30.0)

    def test_clear_resets_all_records(self) -> None:
        """clear() empties all accumulated records."""
        timer = PhaseTimer(device="cpu")
        timer.set_iteration(0)
        with timer.measure("forward"):
            pass
        assert len(timer.get_records()) == 1
        timer.clear()
        assert len(timer.get_records()) == 0

    def test_set_device_changes_sync_device(self) -> None:
        """set_device() updates internal device without error on cpu."""
        timer = PhaseTimer(device="cpu")
        timer.set_device("cpu")
        with timer.measure("test"):
            pass
        assert len(timer.get_records()) == 1

    def test_add_record_direct_append(self) -> None:
        """add_record() accepts explicit PhaseRecord."""
        timer = PhaseTimer()
        record = PhaseRecord(iteration=5, phase="optimizer", time_ms=2.5)
        timer.add_record(record)
        records = timer.get_records()
        assert len(records) == 1
        assert records[0].time_ms == 2.5


# ─── DataLoaderSentinel Tests ───────────────────────────────────────────────────

class TestDataLoaderSentinel:
    """Tests for DataLoaderSentinel batch-0 discard logic."""

    def test_discards_batch_zero_keeps_rest(self) -> None:
        """Batch 0 must not appear in io_durations_ms; batches 1+ are kept."""
        mock_data = [10, 20, 30, 40]
        sentinel = DataLoaderSentinel(mock_data)
        collected = list(sentinel)
        assert collected == mock_data
        assert len(sentinel.io_durations_ms) == 3

    def test_single_batch_retained(self) -> None:
        """Single-batch dataset retains batch-0 as fallback."""
        sentinel = DataLoaderSentinel([42])
        collected = list(sentinel)
        assert collected == [42]
        assert len(sentinel.io_durations_ms) == 1

    def test_empty_dataloader(self) -> None:
        """Empty dataloader yields nothing and records nothing."""
        sentinel = DataLoaderSentinel([])
        collected = list(sentinel)
        assert collected == []
        assert sentinel.io_durations_ms == []
        assert sentinel.mean_io_ms() == 0.0

    def test_mean_io_ms_correctness(self) -> None:
        """mean_io_ms() returns non-negative mean of steady-state durations."""
        sentinel = DataLoaderSentinel([1, 2, 3])
        _ = list(sentinel)
        assert len(sentinel.io_durations_ms) == 2
        assert sentinel.mean_io_ms() >= 0.0

    def test_yields_all_batches_in_order(self) -> None:
        """Iterator must yield every element from source in original order."""
        source = list(range(10))
        sentinel = DataLoaderSentinel(source)
        result = list(sentinel)
        assert result == source


# ─── PhaseProfilerPlugin Tests ──────────────────────────────────────────────────

class TestPhaseProfilerPlugin:
    """Tests for PhaseProfilerPlugin lifecycle and CSV output."""

    def test_attach_resets_state(self) -> None:
        """attach() should clear timer and set device correctly."""
        plugin = PhaseProfilerPlugin()
        plugin.attach(None, device="cpu")
        assert plugin.device == "cpu"
        assert plugin.timer.get_records() == []

    def test_start_stop_default_forward_phase(self) -> None:
        """stop() records a 'forward' phase if no explicit phase was recorded."""
        plugin = PhaseProfilerPlugin()
        plugin.attach(None, device="cpu")
        plugin.start(0)
        time.sleep(0.002)
        plugin.stop()
        records = plugin.timer.get_records()
        assert len(records) == 1
        assert records[0].phase == "forward"
        assert records[0].time_ms > 0.0

    def test_explicit_record_phase_prevents_default(self) -> None:
        """If record_phase() is called, stop() must NOT add a duplicate forward."""
        plugin = PhaseProfilerPlugin()
        plugin.attach(None, device="cpu")
        plugin.start(0)
        plugin.record_phase("forward", 12.5)
        plugin.record_phase("backward", 24.0)
        plugin.stop()
        records = plugin.timer.get_records()
        phases = [r.phase for r in records]
        assert phases.count("forward") == 1
        assert "backward" in phases

    def test_save_creates_phases_csv(self) -> None:
        """save() must write a valid phases.csv with header and data rows."""
        plugin = PhaseProfilerPlugin()
        plugin.attach(None, device="cpu")
        plugin.start(0)
        plugin.record_phase("forward", 10.0)
        plugin.record_phase("backward", 20.0)
        plugin.stop()

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            plugin.save(out_dir)
            csv_path = out_dir / "phases.csv"
            assert csv_path.exists()
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 2
            assert rows[0]["phase"] == "forward"
            assert float(rows[0]["time_ms"]) == pytest.approx(10.0)

    def test_get_summary_returns_means(self) -> None:
        """get_summary() aggregates phase means correctly."""
        plugin = PhaseProfilerPlugin()
        plugin.attach(None, device="cpu")
        plugin.timer.add_record(PhaseRecord(iteration=0, phase="forward", time_ms=10.0))
        plugin.timer.add_record(PhaseRecord(iteration=1, phase="forward", time_ms=30.0))
        summary = plugin.get_summary()
        assert summary["forward"] == pytest.approx(20.0)

    def test_multiple_iterations(self) -> None:
        """Plugin accumulates records across multiple iterations."""
        plugin = PhaseProfilerPlugin()
        plugin.attach(None, device="cpu")
        for i in range(5):
            plugin.start(i)
            plugin.record_phase("forward", float(i * 10))
            plugin.stop()
        records = plugin.timer.get_records()
        assert len(records) == 5
        assert [r.iteration for r in records] == list(range(5))
