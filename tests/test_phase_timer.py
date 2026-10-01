"""Unit tests for TV3 Phase Profiler and DataLoader Sentinel."""

import tempfile
import time
from pathlib import Path

from src.phase.dataloader import DataLoaderSentinel
from src.phase.plugin import PhaseProfilerPlugin
from src.phase.timer import PhaseTimer


def test_phase_timer_context_manager():
    """Verify PhaseTimer records phase durations accurately."""
    timer = PhaseTimer()
    timer.set_iteration(0)

    with timer.measure("forward"):
        time.sleep(0.01)

    records = timer.get_records()
    assert len(records) == 1
    assert records[0].phase == "forward"
    assert records[0].time_ms > 0.0


def test_dataloader_sentinel_discards_batch_zero():
    """Verify DataLoaderSentinel discards cold start batch 0 from steady state metrics."""
    mock_batches = [1, 2, 3, 4]
    sentinel = DataLoaderSentinel(mock_batches)

    collected = []
    for b in sentinel:
        collected.append(b)

    assert collected == mock_batches
    # 4 batches yielded, but batch 0 is discarded, so 3 steady-state I/O metrics recorded
    assert len(sentinel.io_durations_ms) == 3


def test_phase_plugin_save():
    """Test saving phase records to phases.csv."""
    plugin = PhaseProfilerPlugin()
    plugin.attach(None, device="cpu")
    plugin.start(0)
    plugin.record_phase("forward", 15.2)
    plugin.record_phase("backward", 28.4)
    plugin.stop()

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)
        plugin.save(out_dir)
        assert (out_dir / "phases.csv").exists()
