"""
layer_plugin.py — Plugin "layer" theo hợp đồng chung (PROJECT_PLAN Mục 8.1 ②)
==============================================================================
Script run_experiment.py (TV2) gọi plugin này qua cờ `--profilers layer`:

    plugin = LayerPlugin()
    plugin.attach(model, device)
    for i in range(n_iterations):          # sau khi đã warm-up
        plugin.start(i)
        model(inputs)
        plugin.stop()
    plugin.save(out_dir)                    # → layers.csv, layers_summary.csv, layer_meta.json

File ghi ra (TV1 sở hữu):
  - layers.csv          : 1 dòng / layer / lần gọi / iteration
                          (iteration, layer_name, layer_type, time_ms, activation_mb, flops, params, …)
  - layers_summary.csv  : 1 dòng / layer / lần gọi — trung vị thời gian, GFLOP/s, % hiệu suất, bound
  - layer_meta.json     : số layer có hook, timer bias, tổng FLOPs, thông số phần cứng, tổng thời gian mỗi iteration
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn

from src.collector import LayerProfiler, _NumpyEncoder, estimate_timer_bias
from src.flops import CallArgs, add_efficiency, analyze_layers, model_total_flops
from src.hw_specs import HardwareSpec, detect_hardware

logger = logging.getLogger("layer_plugin")


class LayerPlugin:
    """Plugin đo mức Layer: thời gian, bộ nhớ, FLOPs, % hiệu suất từng layer."""

    name = "layer"

    def __init__(
        self,
        subtract_timer_bias: bool = True,
        exclude_types: Optional[Tuple[type, ...]] = None,
        hw_spec: Optional[HardwareSpec] = None,
        precision: Optional[str] = None,
    ):
        """
        Args:
            subtract_timer_bias: Trên CPU, trừ sai số bên trong khoảng đo (estimate_timer_bias).
                                 Trên GPU không trừ (khoảng giữa 2 CUDA event là thời gian kernel).
            exclude_types      : Layer bỏ qua (mặc định nn.Dropout, nn.Identity).
            hw_spec            : Thông số phần cứng; mặc định tự nhận diện.
            precision          : "fp32" / "fp16"; mặc định đoán theo dtype tham số của model.
        """
        self.subtract_timer_bias = subtract_timer_bias
        self.exclude_types = exclude_types
        self.hw_spec = hw_spec
        self.precision = precision
        self.model: Optional[nn.Module] = None
        self.device = "cpu"
        self.profiler: Optional[LayerProfiler] = None
        self.timer_bias_ms = 0.0
        self._captured: Optional[CallArgs] = None
        self._capture_handle = None
        self.enabled = False

    # ── Hợp đồng chung ──────────────────────────────────────────────────────

    def attach(self, model: Any, device: str) -> None:
        """Gắn hook vào model. Model không phải PyTorch (vd XGBoost) → plugin tự tắt."""
        if not isinstance(model, nn.Module):
            logger.warning("[layer] Model không phải nn.Module → bỏ qua mức Layer.")
            self.enabled = False
            return
        self.enabled = True
        self.model, self.device = model, str(device)
        if self.precision is None:
            p = next(model.parameters(), None)
            self.precision = "fp16" if p is not None and p.dtype in (torch.float16, torch.bfloat16) else "fp32"
        if self.hw_spec is None:
            self.hw_spec = detect_hardware(self.device)

        if self.subtract_timer_bias and not self.device.startswith("cuda"):
            self.timer_bias_ms = estimate_timer_bias(self.device)

        self.profiler = LayerProfiler(model, device=self.device, exclude_types=self.exclude_types)
        self.profiler.set_overhead(self.timer_bias_ms)
        self.profiler.attach_hooks()

        # Chụp input của lần gọi đầu tiên để đếm FLOPs lúc save() (ngoài thời gian đo)
        def capture(mod, args, kwargs):
            if self._captured is None:
                self._captured = CallArgs(tuple(args), dict(kwargs))
        self._capture_handle = model.register_forward_pre_hook(capture, with_kwargs=True)

    def start(self, iteration: int) -> None:
        if self.enabled:
            self.profiler.begin_iteration(iteration)

    def stop(self) -> None:
        if self.enabled:
            self.profiler.end_iteration()

    def save(self, out_dir: Union[str, Path]) -> Dict[str, Path]:
        """Ghi layers.csv, layers_summary.csv, layer_meta.json vào out_dir."""
        if not self.enabled:
            return {}
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.detach()

        raw = self.profiler.to_dataframe()
        if raw.empty:
            logger.warning("[layer] Chưa có bản ghi nào — start()/stop() đã được gọi chưa?")
            return {}

        analysis = self._analyze()
        layers = self._format_layers(raw, analysis)
        summary = add_efficiency(raw, analysis, self.hw_spec, self.precision) if not analysis.empty else pd.DataFrame()

        paths = {
            "layers": out_dir / "layers.csv",
            "summary": out_dir / "layers_summary.csv",
            "meta": out_dir / "layer_meta.json",
        }
        layers.to_csv(paths["layers"], index=False)
        summary.to_csv(paths["summary"], index=False)
        with open(paths["meta"], "w", encoding="utf-8") as f:
            json.dump(self._meta(raw, analysis), f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
        logger.info(f"[layer] Đã lưu {len(layers)} dòng → {out_dir}")
        return paths

    # ── Nội bộ ──────────────────────────────────────────────────────────────

    def detach(self) -> None:
        """Gỡ toàn bộ hook (gọi tự động trong save())."""
        if self.profiler is not None:
            self.profiler.remove_hooks()
        if self._capture_handle is not None:
            self._capture_handle.remove()
            self._capture_handle = None

    def _analyze(self) -> pd.DataFrame:
        if self._captured is None:
            logger.warning("[layer] Không chụp được input → bỏ qua FLOPs.")
            return pd.DataFrame()
        try:
            return analyze_layers(self.model, self._captured, exclude_types=self.exclude_types)
        except Exception as e:  # FLOPs là phần phụ, không để làm hỏng cả lần chạy
            logger.warning(f"[layer] Đếm FLOPs lỗi: {e}")
            return pd.DataFrame()
        finally:
            self._captured = None   # nhả tham chiếu tensor input

    @staticmethod
    def _format_layers(raw: pd.DataFrame, analysis: pd.DataFrame) -> pd.DataFrame:
        """Đúng thứ tự cột theo hợp đồng Mục 8.1 ③, các cột phụ để phía sau."""
        df = raw.rename(columns={"forward_time_ms": "time_ms", "param_count": "params"})
        if not analysis.empty:
            df = df.merge(analysis[["layer_name", "flops", "flops_source"]], on="layer_name", how="left")
        else:
            df["flops"], df["flops_source"] = float("nan"), "none"
        first = ["iteration", "layer_name", "layer_type", "time_ms", "activation_mb", "flops", "params"]
        extra = ["call_index", "exec_order", "raw_time_ms", "flops_source", "input_shape", "output_shape",
                 "process_rss_mb", "gpu_mem_before_mb", "gpu_mem_after_mb", "timing_backend", "device"]
        return df[first + [c for c in extra if c in df.columns]]

    def _meta(self, raw: pd.DataFrame, analysis: pd.DataFrame) -> Dict[str, Any]:
        totals = self.profiler.get_iteration_totals()
        per_iter_sum = raw.groupby("iteration")["forward_time_ms"].sum()
        return {
            "plugin": self.name,
            "device": self.device,
            "precision": self.precision,
            "n_hooked_layers": int(raw.drop_duplicates("layer_name").shape[0]),
            "n_iterations": int(raw["iteration"].nunique()),
            "timer_bias_ms": self.timer_bias_ms,
            "total_flops_model": analysis.attrs.get("global_flops") if not analysis.empty else None,
            "total_flops_counter": model_total_flops(analysis) if not analysis.empty else None,
            "total_flops_all": model_total_flops(analysis, "all") if not analysis.empty else None,
            "iteration_total_ms": totals,
            # GPU: tỷ lệ iteration GPU chạy liền mạch (nên = 1.0; thấp hơn → số đo lẫn thời gian chờ CPU)
            "gpu_prefill_ok_ratio": (
                sum(prefill.values()) / len(prefill) if (prefill := self.profiler.get_prefill_ok()) else None
            ),
            "iteration_layer_sum_ms": per_iter_sum.to_dict(),
            "hardware": self.hw_spec.to_dict() if self.hw_spec else None,
        }
