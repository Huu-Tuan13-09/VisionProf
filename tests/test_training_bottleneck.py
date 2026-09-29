"""
test_training_bottleneck.py — Kịch bản giả lập DataLoader bottleneck & Training Loop
======================================================================================
Mục tiêu:
    1. Tạo dataset giả lập với độ trễ I/O có kiểm soát (sleep_ms)
    2. Đo lường DataLoaderSentinel với nhiều cấu hình num_workers
    3. Chạy LayerProfiler trong training loop (forward + backward)
    4. Phân tích tự động với Rule Catalog R-A, R-B, R-C

Chạy:
    python tests/test_training_bottleneck.py
    # Hoặc pytest:
    pytest tests/test_training_bottleneck.py -v -s
"""

from __future__ import annotations

# ── sys.path setup — chạy từ bất kỳ thư mục nào ────────────────────────────
import sys
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Standard imports ─────────────────────────────────────────────────────────
import json
import logging
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ── src imports ───────────────────────────────────────────────────────────────
from src.collector import (
    LayerProfiler,
    DataLoaderSentinel,
    calibrate_overhead,
    _extract_tensor_from_output,
)
from src.analyzer import (
    RuleCatalog,
    Severity,
    analyze_from_csv,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_bottleneck")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
RESULTS_DIR = _PROJECT_ROOT / "data" / "profiles" / "bottleneck_test"
REPORTS_DIR = _PROJECT_ROOT / "reports"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
N_CLASSES  = 10
IMG_SIZE   = 224
N_BATCHES  = 30    # Số batch mỗi epoch (nhỏ để test nhanh)
N_EPOCHS   = 2     # Số epoch training
N_PROFILE_ITER = 5 # Số lần profile forward pass


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset giả lập với độ trễ I/O có kiểm soát
# ─────────────────────────────────────────────────────────────────────────────

class SlowDataset(Dataset):
    """
    Dataset giả lập có thể điều chỉnh độ trễ I/O.

    Mục đích:
        - sleep_ms=0   : Dataset nhanh (không có bottleneck)
        - sleep_ms=20  : Giả lập đọc từ HDD chậm
        - sleep_ms=100 : Giả lập bottleneck nghiêm trọng (network storage)
    """

    def __init__(
        self,
        n_samples: int      = 256,
        img_size:  int      = 224,
        n_classes: int      = 10,
        sleep_ms:  float    = 0.0,
        noise_std: float    = 5.0,   # Thêm noise vào sleep (ms) để tạo variance
    ):
        self.n_samples = n_samples
        self.img_size  = img_size
        self.n_classes = n_classes
        self.sleep_ms  = sleep_ms
        self.noise_std = noise_std

        # Pre-generate data trong RAM để tập trung đo I/O simulation
        self.data   = torch.randn(n_samples, 3, img_size, img_size)
        self.labels = torch.randint(0, n_classes, (n_samples,))

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        # Giả lập độ trễ đọc file
        if self.sleep_ms > 0:
            actual_sleep = self.sleep_ms + np.random.normal(0, self.noise_std)
            actual_sleep = max(0.0, actual_sleep)
            time.sleep(actual_sleep / 1000.0)
        return self.data[idx], self.labels[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Mô hình đơn giản để test
# ─────────────────────────────────────────────────────────────────────────────

class SimpleConvNet(nn.Module):
    """
    Mô hình CNN nhỏ để test training bottleneck.
    Đủ phức tạp để profiling có ý nghĩa, đủ nhỏ để test nhanh.
    """

    def __init__(self, n_classes: int = 10, img_size: int = 224):
        super().__init__()
        # Block 1
        self.conv1 = nn.Conv2d(3,  32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)   # 112x112

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)   # 56x56

        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))  # 4x4

        # Classifier
        self.flatten = nn.Flatten()
        self.fc1  = nn.Linear(128 * 4 * 4, 256)
        self.relu4= nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.3)
        self.fc2  = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.drop(self.relu4(self.fc1(x)))
        return self.fc2(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Kịch bản đo DataLoaderSentinel
# ─────────────────────────────────────────────────────────────────────────────

def run_dataloader_benchmark(
    sleep_ms:    float = 0.0,
    num_workers: int   = 0,
    description: str   = "fast_loader",
) -> dict:
    """
    Đo hiệu năng DataLoader với DataLoaderSentinel.

    Args:
        sleep_ms    : Độ trễ I/O giả lập mỗi sample (ms).
        num_workers : Số worker processes của DataLoader.
        description : Tên mô tả kịch bản.

    Returns:
        dict chứa DataLoaderStats.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"[DataLoader] Kịch bản: {description}")
    logger.info(f"  sleep_ms={sleep_ms}ms | num_workers={num_workers}")

    dataset = SlowDataset(
        n_samples=N_BATCHES * BATCH_SIZE,
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        sleep_ms=sleep_ms,
    )
    # Windows: dùng num_workers=0 khi trong __main__ không dùng spawn-safe
    safe_workers = num_workers  # Được wrap trong if __name__ == "__main__"

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=safe_workers,
        pin_memory=(DEVICE == "cuda"),
    )

    model     = SimpleConvNet(N_CLASSES, IMG_SIZE).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    sentinel = DataLoaderSentinel(
        dataloader=loader,
        device=DEVICE,
        description=description,
    )

    logger.info(f"  Đang chạy {N_BATCHES} batches...")
    for images, labels in sentinel:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        output = model(images)
        loss   = criterion(output, labels)
        loss.backward()
        optimizer.step()

        sentinel.mark_compute_end()  # Đánh dấu kết thúc compute

    stats = sentinel.get_stats()

    # Lưu báo cáo
    out_path = RESULTS_DIR / f"dataloader_{description}.json"
    sentinel.save_report(str(out_path))
    sentinel.save_csv(str(RESULTS_DIR / f"dataloader_{description}.csv"))

    logger.info(f"  → I/O ratio:    {stats.io_bottleneck_ratio*100:.1f}%")
    logger.info(f"  → Mean load:    {stats.mean_load_time_ms:.2f} ms")
    logger.info(f"  → Mean compute: {stats.mean_compute_time_ms:.2f} ms")
    logger.info(f"  → Peak RAM:     {stats.peak_ram_mb:.1f} MB")

    return stats.to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Kịch bản training loop với LayerProfiler
# ─────────────────────────────────────────────────────────────────────────────

def run_training_profiling() -> str:
    """
    Chạy training loop có profiling layer-level.

    Returns:
        Đường dẫn CSV chứa layer records.
    """
    logger.info(f"\n{'='*60}")
    logger.info("[Training] Chạy profiling trong training loop...")

    model     = SimpleConvNet(N_CLASSES, IMG_SIZE).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Calibrate overhead
    dummy_input = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    logger.info("[Training] Calibrating hook overhead...")
    try:
        overhead_ms = calibrate_overhead(
            model, dummy_input,
            n_warmup=3, n_measure=5, device=DEVICE,
        )
    except Exception as e:
        logger.warning(f"calibrate_overhead lỗi: {e}. Dùng overhead=0.")
        overhead_ms = 0.0

    # Tạo dataset và loader (không có delay cho training profiling)
    dataset = SlowDataset(n_samples=N_PROFILE_ITER * BATCH_SIZE, sleep_ms=0.0)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    profiler = LayerProfiler(model, device=DEVICE)
    profiler.set_overhead(overhead_ms)

    model.train()
    all_records = []
    iteration   = 0

    logger.info(f"[Training] Profiling {N_PROFILE_ITER} iterations...")
    for images, labels in loader:
        if iteration >= N_PROFILE_ITER:
            break

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()

        # Profile forward pass
        with profiler.profile_context(iteration=iteration):
            output = model(images)

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        # Thu thập records của iteration này
        iter_records = [r for r in profiler.get_records() if r.iteration == iteration]
        all_records.extend(iter_records)

        logger.info(
            f"  Iter {iteration}: loss={loss.item():.4f} | "
            f"n_records={len(iter_records)}"
        )
        iteration += 1

    # Lưu CSV
    csv_path = RESULTS_DIR / "training_layer_records.csv"
    df       = profiler.to_dataframe()
    df.to_csv(str(csv_path), index=False)
    logger.info(f"[Training] Đã lưu {len(df)} records → {csv_path}")

    return str(csv_path)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Chạy Rule Catalog phân tích
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(csv_path: str, dl_json_path: str = "") -> None:
    """Phân tích kết quả profiling bằng Rule Catalog."""
    logger.info(f"\n{'='*60}")
    logger.info("[Analysis] Chạy Rule Catalog...")

    df = pd.read_csv(csv_path)
    total_params = int(df.groupby("layer_name")["param_count"].first().sum()) \
                   if "param_count" in df.columns else 0

    dl_stats = None
    if dl_json_path and Path(dl_json_path).exists():
        with open(dl_json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        dl_stats = raw.get("summary", {})
        batch_detail = raw.get("batch_detail", [])
        dl_stats["batch_load_times"] = [b.get("load_time_ms", 0) for b in batch_detail]

    # Lấy tổng GPU memory nếu có
    total_gpu_mem = 0.0
    if DEVICE == "cuda":
        try:
            total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**2
        except Exception:
            pass

    catalog = RuleCatalog()
    report  = catalog.run_all(
        df=df,
        model_name="SimpleConvNet",
        device=DEVICE,
        total_gpu_mem_mb=total_gpu_mem,
        total_params=total_params,
        dataloader_stats=dl_stats,
    )

    report.print_summary()

    # Lưu báo cáo
    out_dir = REPORTS_DIR / "bottleneck_test"
    report.save_json(str(out_dir / "analysis_report.json"))
    report.save_csv(str(out_dir / "analysis_report.csv"))


# ─────────────────────────────────────────────────────────────────────────────
# 6. So sánh A/B: fast_loader vs slow_loader
# ─────────────────────────────────────────────────────────────────────────────

def run_ab_dataloader_comparison() -> None:
    """So sánh hai kịch bản DataLoader bằng A/B Engine."""
    from src.analyzer import ABComparisonEngine

    logger.info(f"\n{'='*60}")
    logger.info("[A/B] So sánh DataLoader fast vs slow...")

    csv_fast = RESULTS_DIR / "dataloader_fast_loader.csv"
    csv_slow = RESULTS_DIR / "dataloader_slow_loader.csv"

    if not csv_fast.exists() or not csv_slow.exists():
        logger.warning("[A/B] Chưa có dữ liệu DataLoader. Bỏ qua A/B comparison.")
        return

    # Đọc batch-level data cho từng loader
    df_fast = pd.read_csv(str(csv_fast))
    df_slow = pd.read_csv(str(csv_slow))

    if df_fast.empty or df_slow.empty:
        logger.warning("[A/B] DataLoader CSV trống. Bỏ qua.")
        return

    # Chuẩn hóa để A/B compare hoạt động: cần cột forward_time_ms, layer_name
    # Ở đây ta dùng load_time_ms và compute_time_ms làm "layers"
    def _reshape_for_ab(df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Reshape batch-level DF sang format layer_records cho A/B."""
        rows = []
        for _, row in df.iterrows():
            rows.append({
                "layer_name":      "DataLoading",
                "layer_type":      "IO",
                "forward_time_ms": row.get("load_time_ms", 0),
                "gpu_mem_delta_mb":0,
                "param_count":     0,
                "iteration":       row.get("batch_index", 0),
                "device":          DEVICE,
            })
            rows.append({
                "layer_name":      "Compute",
                "layer_type":      "Compute",
                "forward_time_ms": row.get("compute_time_ms", 0),
                "gpu_mem_delta_mb":0,
                "param_count":     0,
                "iteration":       row.get("batch_index", 0),
                "device":          DEVICE,
            })
        return pd.DataFrame(rows)

    df_a = _reshape_for_ab(df_fast, "fast")
    df_b = _reshape_for_ab(df_slow, "slow")

    try:
        engine = ABComparisonEngine(
            df_a=df_a, df_b=df_b,
            name_a="Fast Loader (0ms)",
            name_b="Slow Loader (20ms)",
            regression_threshold_pct=10.0,
            improvement_threshold_pct=10.0,
        )
        report = engine.compare()
        report.print_summary()

        out_dir = REPORTS_DIR / "bottleneck_test"
        out_dir.mkdir(parents=True, exist_ok=True)
        report.save_json(str(out_dir / "ab_fast_vs_slow.json"))
        report.save_csv(str(out_dir  / "ab_fast_vs_slow.csv"))
    except Exception as e:
        logger.warning(f"[A/B] Lỗi: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Pytest fixtures và test functions
# ─────────────────────────────────────────────────────────────────────────────

def test_slow_dataset_delay():
    """Pytest: Kiểm tra SlowDataset tạo đúng độ trễ."""
    sleep_ms = 10.0
    ds = SlowDataset(n_samples=3, sleep_ms=sleep_ms, noise_std=0.0)
    t0 = time.perf_counter()
    _ = ds[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms >= sleep_ms * 0.5, \
        f"Expected delay >= {sleep_ms*0.5:.1f}ms, got {elapsed_ms:.1f}ms"
    logger.info(f"[test] SlowDataset delay: {elapsed_ms:.1f}ms (expected ~{sleep_ms}ms)")


def test_dataloader_sentinel_basic():
    """Pytest: Kiểm tra DataLoaderSentinel ghi nhận được load/compute time."""
    ds     = SlowDataset(n_samples=BATCH_SIZE * 5, sleep_ms=0.0)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=0)
    model  = SimpleConvNet(N_CLASSES, IMG_SIZE)
    model.eval()

    sentinel = DataLoaderSentinel(loader, device="cpu", description="test")
    batch_count = 0

    with torch.no_grad():
        for images, labels in sentinel:
            _ = model(images)
            sentinel.mark_compute_end()
            batch_count += 1

    stats = sentinel.get_stats()
    assert stats.total_batches == batch_count, \
        f"Expected {batch_count} batches, got {stats.total_batches}"
    assert stats.mean_compute_time_ms >= 0, "compute time phải >= 0"
    assert 0.0 <= stats.io_bottleneck_ratio <= 1.0, "ratio phải trong [0,1]"
    logger.info(f"[test] Sentinel OK: {stats.total_batches} batches, ratio={stats.io_bottleneck_ratio:.3f}")


def test_layer_profiler_records():
    """Pytest: Kiểm tra LayerProfiler ghi đủ records."""
    model     = SimpleConvNet(N_CLASSES, IMG_SIZE)
    profiler  = LayerProfiler(model, device="cpu")
    dummy_in  = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)

    with profiler.profile_context(iteration=0):
        with torch.no_grad():
            _ = model(dummy_in)

    records = profiler.get_records()
    assert len(records) > 0, "Phải ghi được ít nhất 1 record"

    # Kiểm tra các trường bắt buộc
    for r in records:
        assert r.layer_name != "", "layer_name không được trống"
        assert r.forward_time_ms >= 0, "forward_time_ms phải >= 0"

    logger.info(f"[test] LayerProfiler OK: {len(records)} records")


def test_rule_catalog_smoke():
    """Pytest: Smoke test Rule Catalog chạy không lỗi trên dữ liệu ngẫu nhiên."""
    # Tạo DataFrame giả
    n = 50
    df = pd.DataFrame({
        "layer_name":        [f"layer_{i}" for i in range(n)],
        "layer_type":        np.random.choice(["Conv2d","Linear","BatchNorm2d"], n),
        "forward_time_ms":   np.random.exponential(1.0, n),
        "gpu_mem_before_mb": np.random.uniform(100, 200, n),
        "gpu_mem_after_mb":  np.random.uniform(100, 220, n),
        "gpu_mem_delta_mb":  np.random.uniform(-5, 20, n),
        "param_count":       np.random.randint(100, 1_000_000, n),
        "gpu_util_pct":      np.random.uniform(50, 95, n),
        "iteration":         np.repeat(np.arange(5), 10),
        "device":            "cpu",
    })

    catalog = RuleCatalog()
    report  = catalog.run_all(
        df=df,
        model_name="smoke_test",
        device="cpu",
        total_gpu_mem_mb=0.0,
        total_params=int(df["param_count"].sum()),
    )

    assert report.total_rules > 0, "Phải có ít nhất 1 quy tắc"
    assert report.passed + report.failed == report.total_rules
    logger.info(f"[test] RuleCatalog smoke OK: {report.total_rules} rules, "
                f"{report.passed} passed, {report.failed} failed")


# ─────────────────────────────────────────────────────────────────────────────
# Main — chạy toàn bộ kịch bản
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import platform
    print(f"\n{'='*60}")
    print(f" PyTorch Layer Profiler — DataLoader Bottleneck Test")
    print(f"{'='*60}")
    print(f" Device   : {DEVICE}")
    print(f" OS       : {platform.system()} {platform.release()}")
    print(f" PyTorch  : {torch.__version__}")
    print(f" Batch    : {BATCH_SIZE} | Batches: {N_BATCHES} | Epochs: {N_EPOCHS}")
    print(f" Save dir : {RESULTS_DIR}")
    print(f"{'='*60}\n")

    # ── Chạy unit tests trước ────────────────────────────────────────────────
    print("[1/6] Chạy unit tests smoke...")
    try:
        test_slow_dataset_delay()
        print("  ✅ test_slow_dataset_delay")
    except AssertionError as e:
        print(f"  ⚠️ test_slow_dataset_delay: {e}")

    try:
        test_dataloader_sentinel_basic()
        print("  ✅ test_dataloader_sentinel_basic")
    except AssertionError as e:
        print(f"  ❌ test_dataloader_sentinel_basic: {e}")

    try:
        test_layer_profiler_records()
        print("  ✅ test_layer_profiler_records")
    except AssertionError as e:
        print(f"  ❌ test_layer_profiler_records: {e}")

    try:
        test_rule_catalog_smoke()
        print("  ✅ test_rule_catalog_smoke")
    except AssertionError as e:
        print(f"  ❌ test_rule_catalog_smoke: {e}")

    # ── Kịch bản DataLoader nhanh (baseline) ─────────────────────────────────
    print("\n[2/6] DataLoader nhanh (không có I/O delay)...")
    stats_fast = run_dataloader_benchmark(
        sleep_ms=0.0,
        num_workers=0,
        description="fast_loader",
    )

    # ── Kịch bản DataLoader chậm (bottleneck giả lập) ────────────────────────
    print("\n[3/6] DataLoader chậm (sleep=20ms/sample)...")
    stats_slow = run_dataloader_benchmark(
        sleep_ms=20.0,
        num_workers=0,
        description="slow_loader",
    )

    # ── Kịch bản DataLoader rất chậm (critical bottleneck) ───────────────────
    print("\n[4/6] DataLoader rất chậm (sleep=80ms/sample — critical)...")
    stats_critical = run_dataloader_benchmark(
        sleep_ms=80.0,
        num_workers=0,
        description="critical_loader",
    )

    # ── In bảng so sánh DataLoader ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(" So Sánh DataLoader Benchmarks")
    print(f"{'='*60}")
    print(f"{'Kịch bản':<20} {'Load(ms)':>10} {'Compute(ms)':>12} {'I/O Ratio':>10} {'Kết luận':>15}")
    print("-" * 70)
    scenarios = [
        ("fast_loader",     stats_fast),
        ("slow_loader",     stats_slow),
        ("critical_loader", stats_critical),
    ]
    for name, s in scenarios:
        ratio = s.get("io_bottleneck_ratio", 0)
        verdict = "OK" if ratio < 0.4 else ("Cảnh báo" if ratio < 0.6 else "CRITICAL")
        print(
            f"{name:<20} {s.get('mean_load_time_ms',0):>10.2f} "
            f"{s.get('mean_compute_time_ms',0):>12.2f} "
            f"{ratio*100:>9.1f}% {verdict:>15}"
        )

    # ── Training loop profiling ───────────────────────────────────────────────
    print(f"\n[5/6] Profiling training loop...")
    csv_path = run_training_profiling()

    # ── Rule Catalog analysis ─────────────────────────────────────────────────
    print(f"\n[6/6] Chạy Rule Catalog analysis...")
    dl_json_path = str(RESULTS_DIR / "dataloader_slow_loader.json")
    run_analysis(csv_path, dl_json_path=dl_json_path)

    # ── A/B comparison ────────────────────────────────────────────────────────
    run_ab_dataloader_comparison()

    print(f"\n{'='*60}")
    print(" HOÀN TẤT! Kết quả lưu tại:")
    print(f"   Profiles : {RESULTS_DIR}")
    print(f"   Reports  : {REPORTS_DIR / 'bottleneck_test'}")
    print(f"\n Mở dashboard:")
    print(f"   streamlit run src/dashboard.py")
    print(f"{'='*60}\n")
