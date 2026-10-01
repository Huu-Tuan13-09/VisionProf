"""Background daemon sampler for CPU, RAM, GPU utilization and VRAM (100ms interval)."""

import csv
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, List, Optional

from src.core.types import ResourceRecord

logger = logging.getLogger(__name__)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class ResourceSampler:
    """Samples hardware resources asynchronously at fixed 100ms intervals."""

    def __init__(self, interval_s: float = 0.1) -> None:
        self.interval_s: float = interval_s
        self._records: List[ResourceRecord] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._process = psutil.Process(os.getpid()) if HAS_PSUTIL else None
        self._nvml_handle: Optional[Any] = None
        self._lock = threading.Lock()
        self._init_nvml()

    def _init_nvml(self) -> None:
        """Safely initialize NVML handle if hardware and drivers exist."""
        if HAS_PYNVML:
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                logger.info("NVML initialized successfully for resource sampling.")
            except Exception as exc:
                logger.debug("NVML init failed, falling back to torch.cuda metrics: %s", exc)
                self._nvml_handle = None

    def _shutdown_nvml(self) -> None:
        """Safely terminate NVML session."""
        if self._nvml_handle is not None and HAS_PYNVML:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_handle = None

    def start(self) -> None:
        """Spawn background daemon sampler thread."""
        with self._lock:
            self._records.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self._thread.start()

    def _sampling_loop(self) -> None:
        while not self._stop_event.is_set():
            now = time.time()
            cpu_pct = self._process.cpu_percent(interval=None) if self._process else 0.0
            rss_mb = (self._process.memory_info().rss / (1024 * 1024)) if self._process else 0.0

            gpu_util = 0.0
            vram_mb = 0.0
            power_w = 0.0

            if self._nvml_handle is not None:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                    gpu_util = float(util.gpu)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                    vram_mb = round(mem.used / (1024 * 1024), 2)
                    power_w = round(pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000.0, 2)
                except Exception:
                    pass
            elif HAS_TORCH and torch.cuda.is_available():
                vram_mb = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2)

            rec = ResourceRecord(
                timestamp=now,
                cpu_pct=round(cpu_pct, 1),
                rss_mb=round(rss_mb, 1),
                gpu_util_pct=gpu_util,
                vram_mb=vram_mb,
                power_w=power_w,
            )
            with self._lock:
                self._records.append(rec)
            time.sleep(self.interval_s)

    def stop(self) -> List[ResourceRecord]:
        """Signal sampling thread to stop and wait for completion."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._shutdown_nvml()
        with self._lock:
            return list(self._records)

    def save(self, out_path: Path) -> Path:
        """Write collected resource timeline samples to CSV file."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            records_snapshot = list(self._records)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "cpu_pct", "rss_mb", "gpu_util_pct", "vram_mb", "power_w"])
            for r in records_snapshot:
                writer.writerow([r.timestamp, r.cpu_pct, r.rss_mb, r.gpu_util_pct, r.vram_mb, r.power_w])
        logger.info("Saved %d resource samples to %s", len(records_snapshot), out_path)
        return out_path
