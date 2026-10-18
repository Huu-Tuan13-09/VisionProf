"""Unit tests for TV1 Layer Profiler components."""

import tempfile
from pathlib import Path
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from src.layer.flops import FlopsCalculator
    from src.layer.memory import ActivationMemoryTracker
    from src.layer.overhead import OverheadCalibrator
    from src.layer.plugin import LayerProfilerPlugin, profile_layers
except ImportError:
    FlopsCalculator = ActivationMemoryTracker = OverheadCalibrator = LayerProfilerPlugin = profile_layers = None  # type: ignore


def test_zero_config_attachment():
    """Verify that LayerProfilerPlugin attaches only to leaf modules and cleans up."""
    if not HAS_TORCH or LayerProfilerPlugin is None:
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
    if LayerProfilerPlugin is None:
        return
    non_torch_model = object()
    plugin = LayerProfilerPlugin()
    plugin.attach(non_torch_model, device="cpu")
    assert not plugin._enabled


def test_activation_and_flops():
    """Test activation memory and FLOPs calculation for Conv2d."""
    if not HAS_TORCH or ActivationMemoryTracker is None:
        return
    conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
    inp = torch.randn(1, 3, 10, 10)
    out = conv(inp)

    act_mb = ActivationMemoryTracker.calculate_output_mb(out)
    assert act_mb > 0.0

    flops = FlopsCalculator.estimate_module_flops(conv, inp.shape, out.shape)
    assert flops > 0


class _Sleep(nn.Module if HAS_TORCH else object):
    """Layer ngủ `ms` mili-giây và tự ghi lại thời gian ngủ thật."""

    def __init__(self, ms: float) -> None:
        super().__init__()
        self.ms, self.actual_ms = ms, 0.0

    def forward(self, x):
        import time
        t0 = time.perf_counter()
        time.sleep(self.ms / 1000.0)
        self.actual_ms = (time.perf_counter() - t0) * 1000.0
        return x + 0


def _run(plugin, model, x, n_iter=3):
    with torch.no_grad():
        for i in range(n_iter):
            plugin.start(i)
            model(x)
            plugin.stop()


def test_forward_outside_start_stop_is_not_recorded():
    """Forward ngoài start()/stop() (warm-up, đếm FLOPs, torch.profiler) không được ghi thêm."""
    if not HAS_TORCH or LayerProfilerPlugin is None:
        return
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2)).eval()
    x = torch.randn(4, 8)
    plugin = LayerProfilerPlugin(subtract_timer_bias=False)
    plugin.attach(model, device="cpu")
    _run(plugin, model, x)
    with torch.no_grad():
        for _ in range(5):
            model(x)
    with tempfile.TemporaryDirectory() as tmp_dir:
        plugin.save(Path(tmp_dir))   # save() chạy thêm 1 forward để đếm FLOPs
    df = plugin.to_dataframe()
    assert df.groupby("iteration").size().tolist() == [3, 3, 3]
    assert "forward_time_ms" not in df.columns   # tránh trùng cột time_ms khi đổi tên


def test_cpu_layer_time_matches_sleep_duration():
    """Khoảng đo bao trọn layer và chỉ dư chi phí nhỏ của hook (không gồm việc tính shape/bộ nhớ)."""
    if not HAS_TORCH or LayerProfilerPlugin is None:
        return
    model = nn.Sequential(_Sleep(20), _Sleep(5))
    plugin = LayerProfilerPlugin(subtract_timer_bias=False)
    plugin.attach(model, device="cpu")
    _run(plugin, model, torch.zeros(4), n_iter=2)   # vòng 0 là khởi động, chỉ xét vòng 1
    df = plugin.to_dataframe()
    t = dict(zip(df.loc[df["iteration"] == 1, "layer_name"], df.loc[df["iteration"] == 1, "raw_time_ms"]))
    for name, layer in zip(["0", "1"], model):
        assert t[name] >= layer.actual_ms
        assert t[name] - layer.actual_ms < 2.0


def test_iteration_total_covers_layers_and_csv_has_exec_order():
    """Tổng 1 vòng là thời gian cả forward (≥ tổng các layer); layers.csv có exec_order."""
    if not HAS_TORCH or LayerProfilerPlugin is None:
        return
    model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.Flatten(), nn.Linear(8 * 14 * 14, 4)).eval()
    plugin = LayerProfilerPlugin(subtract_timer_bias=False)
    plugin.attach(model, device="cpu")
    _run(plugin, model, torch.randn(1, 3, 16, 16))
    df = plugin.to_dataframe()
    per_iter = df.groupby("iteration")["time_ms"].sum()
    for it, total in plugin.get_iteration_totals().items():
        assert total >= per_iter[it]
    with tempfile.TemporaryDirectory() as tmp_dir:
        plugin.save(Path(tmp_dir))
        import pandas as pd
        saved = pd.read_csv(Path(tmp_dir) / "layers.csv")
    assert saved.groupby("iteration")["exec_order"].apply(list).tolist()[0] == [0, 1, 2, 3]


def test_overhead_counts_only_hooked_layers():
    """Overhead dùng bộ đo thật; mẫu số là số layer THỰC SỰ có hook (Dropout bị bỏ qua)."""
    if not HAS_TORCH or LayerProfilerPlugin is None:
        return
    from src.layer.overhead import calibrate_overhead_details
    model = nn.Sequential(nn.Linear(8, 8), nn.Dropout(0.1), nn.ReLU(), nn.Linear(8, 2)).eval()
    details = calibrate_overhead_details(model, {"input": torch.randn(2, 8)}, n_warmup=1, n_measure=3)
    assert details["n_hooked"] == 3
    assert details["t_hooked_ms"] > 0 and details["t_clean_ms"] > 0
