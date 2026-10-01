"""Unit tests for TV1 Layer Profiler components."""

import tempfile
from pathlib import Path
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.layer.flops import FlopsCalculator
from src.layer.memory import ActivationMemoryTracker
from src.layer.overhead import OverheadCalibrator
from src.layer.plugin import LayerProfilerPlugin, profile_layers


def test_zero_config_attachment():
    """Verify that LayerProfilerPlugin attaches only to leaf modules and cleans up."""
    if not HAS_TORCH:
        return
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
    )
    plugin = LayerProfilerPlugin()
    plugin.attach(model, device="cpu")

    # Should attach 3 leaf modules: Conv2d, BatchNorm2d, ReLU (total 6 hook handles)
    assert len(plugin._hooks) == 6

    # Test forward pass with plugin
    plugin.start(iteration=0)
    x = torch.randn(2, 3, 32, 32)
    _ = model(x)
    plugin.stop()

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)
        plugin.save(out_dir)
        assert (out_dir / "layers.csv").exists()

    plugin.detach()
    assert len(plugin._hooks) == 0


def test_non_torch_bypass():
    """Verify that non-PyTorch objects are bypassed safely without exceptions."""
    non_torch_model = object()
    plugin = LayerProfilerPlugin()
    plugin.attach(non_torch_model, device="cpu")
    assert not plugin._enabled


def test_activation_and_flops():
    """Test activation memory and FLOPs calculation for Conv2d."""
    if not HAS_TORCH:
        return
    conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
    inp = torch.randn(1, 3, 10, 10)
    out = conv(inp)

    act_mb = ActivationMemoryTracker.calculate_output_mb(out)
    assert act_mb > 0.0

    flops = FlopsCalculator.estimate_module_flops(conv, inp.shape, out.shape)
    assert flops > 0
