# Kế Hoạch Nghiên Cứu (Bản Cập Nhật)
## Profiling Framework cho Vision AI Models — Cross-Platform CPU vs. GPU

> **Phiên bản**: v2.0 — Đã tích hợp thông tin hardware & định hướng thực tế  
> **Hardware scope**: CPU-only Laptop (Windows) ↔ Google Colab (Tesla T4 16GB)  
> **Định hướng**: 80% Systems Profiling · 20% Automated Diagnostic Heuristics  
> **Model targets**: YOLOv8, YOLOv11, ConvNeXt, Vision Transformer, MobileNetV3

---

## ⚡ Thay Đổi Quan Trọng So Với v1.0

Việc làm rõ hai biến số đã **đóng cửa hoàn toàn PA-A và PA-C** từ v1.0, đồng thời
tạo ra một hướng hoàn toàn mới — không phải LLM, mà là **Vision Model Profiling
trên cross-platform edge–cloud**. Đây thực ra là gap **ít cạnh tranh hơn** và
**dễ publish hơn** so với LLM profiling vì:

- LLM profiling đang rất crowded (vLLM, Meta, Google đều làm).
- Cross-platform Vision profiling trên edge CPU vs. cloud GPU với
  automated heuristic layer gần như chưa có paper hệ thống.
- T4 là GPU phổ biến nhất trong giáo dục/research ở Đông Nam Á
  → demographic angle thú vị cho MLSys/EuroSys.

---

## 1. THU HẸP PHẠM VI — Research Question Duy Nhất

### 1.1 Research Question Chính (Đã Xác Định)

> **"Có thể xây dựng một profiling framework portable, overhead thấp (<3%),
> với độ phân giải per-layer, tự động phân loại bottleneck và đưa ra
> khuyến nghị kỹ thuật khả thi — hoạt động nhất quán trên cả CPU-only
> và CUDA GPU mà không cần cấu hình thủ công?"**

### 1.2 Phân Rã Thành Sub-Questions

| # | Sub-question | Phần framework giải quyết |
|---|-------------|--------------------------|
| **SQ1** | Bottleneck phân bổ theo layer như thế nào, và có thay đổi khi chuyển từ CPU → GPU không? | Layer Profiler + A/B Comparison Engine |
| **SQ2** | Profiling overhead ảnh hưởng thế nào đến kết quả đo lường? Và làm thế nào để tối thiểu hóa? | Overhead Calibration Module |
| **SQ3** | Heuristic rule nào đủ tin cậy để tự động "bắt bệnh" từ profiling data? | Automated Heuristic Analyzer |
| **SQ4** | Giữa YOLOv8/v11, ConvNeXt, ViT, MobileNetV3 — kiến trúc nào có bottleneck profile khác biệt nhất và tại sao? | Cross-Architecture Benchmark Study |

### 1.3 Hypothesis

```
H1 (Bottleneck shift): Conv2d-heavy architectures (YOLOv8/v11, MobileNetV3) 
    là memory-bandwidth-bound trên GPU T4 nhưng compute-bound trên CPU,
    trong khi ViT (Attention layers) là compute-bound trên cả hai 
    nhưng với compute pattern khác nhau (sequential vs. parallel).

H2 (Overhead portability): Framework đạt được <3% throughput overhead 
    trên cả CPU và GPU bằng cách dùng cùng hook API nhưng backend 
    timing khác nhau (time.perf_counter trên CPU, CUDA Events trên GPU).

H3 (Heuristic accuracy): Tập hợp ≤15 deterministic rules có thể 
    phát hiện đúng ≥85% bottleneck cases được xác nhận bởi Nsight/VTune
    mà không cần model ML phức tạp.

H4 (Cross-platform insight): Ranking kiến trúc theo throughput trên CPU 
    KHÁC với ranking trên GPU T4 — có ít nhất 1 "inversion" — cho thấy
    benchmark CPU không thể predict GPU performance.
```

### 1.4 Success Criteria

| Tiêu chí | Mức tối thiểu (acceptable) | Mức tốt (strong paper) |
|---------|--------------------------|----------------------|
| Profiling overhead | < 5% throughput degradation | < 2% |
| Timing accuracy vs. ground truth | < 5% error | < 2% error |
| Heuristic detection rate | ≥ 75% correct diagnosis | ≥ 90% |
| Model coverage | ≥ 3 architectures | ≥ 5 architectures |
| Platform coverage | CPU + 1 GPU | CPU + ≥ 2 GPU types |
| False positive rate (heuristics) | < 20% | < 10% |

---

## 2. RÀ SOÁT TÀI LIỆU (Related Work & Gap Analysis)

### 2.1 Danh Sách Tài Liệu Phải Đọc

#### Tier 1 — Tools & Frameworks Trực Tiếp Cạnh Tranh

| Công cụ / Paper | Cần phân tích gì |
|----------------|-----------------|
| **torch.profiler** (PyTorch ≥ 1.9) | Granularity có đến layer không? Overhead? CPU/GPU unified? |
| **TensorBoard Profiler Plugin** | Visualization quality, trace format, Chrome Trace |
| **NVIDIA Nsight Systems** | Kernel-level ground truth — dùng làm baseline validation |
| **NVIDIA Nsight Compute** | Roofline model output, memory bandwidth measurement |
| **Intel VTune Profiler** | CPU profiling depth — so sánh với approach của ta |
| **ONNX Runtime Profiling** | Cross-platform profiling — tương tự goal nhưng khác level |
| **PyTorch Benchmark Utils** (`torch.utils.benchmark`) | Timing accuracy, warm-up strategy |

#### Tier 2 — Papers Về Methodology

| Paper | Venue | Relevance |
|-------|-------|----------|
| **Roofline: An Insightful Visual Performance Model** (Williams'09) | CACM | Compute vs. memory bound analysis — cite để justify H1 |
| **EfficientNet** (Tan & Le, ICML'19) | ICML | FLOP ≠ latency — empirical evidence cần đối chiếu |
| **Benchmarking Neural Network Training Algorithms** (MLCommons) | MLSys'23 | Benchmark methodology best practices |
| **MLPerf Inference Benchmark** | MLSys | Standard benchmark design — compare methodology |
| **DeepView** (MLSys'21) | MLSys 2021 | Layer-wise profiling cho DNN — closest related work, đọc kỹ |
| **Habitat** (OSDI'21) | OSDI 2021 | Cross-GPU performance prediction — methodology overlap |
| **nn-Meter** (MobiSys'21) | MobiSys 2021 | Edge device profiling — đọc để differentiate từ edge angle |
| **BOLT** (MLSys'22) | MLSys 2022 | Mobile neural network profiling |

#### Tier 3 — Vision Architecture Papers (Cần Hiểu Để Interpret Findings)

| Paper | Cần đọc gì |
|-------|-----------|
| **YOLOv8/v11** (Ultralytics) | Architecture detail, bottleneck layer design |
| **ConvNeXt** (Liu et al., CVPR'22) | Depthwise conv pattern, layer distribution |
| **ViT** (Dosovitskiy et al., ICLR'21) | Attention complexity, memory pattern |
| **MobileNetV3** (Howard et al., ICCV'19) | Depthwise separable conv, squeeze-excitation |

### 2.2 Gap Analysis — Điều Không Ai Làm Được Hiện Tại

#### Gap #1 — Không Có Công Cụ Cross-Platform Tự Động Chuyển Backend *(Core Gap)*
- **Hiện trạng**: PyTorch Profiler chạy được cả CPU/GPU nhưng API không thống nhất
  (CPU dùng `with_record_shapes`, GPU cần `use_cuda=True`). Kết quả trace ở hai
  platform không so sánh được trực tiếp vì format khác nhau, unit khác nhau.
- **Gap**: Không có framework nào tự động detect device, normalize output format,
  và produce comparable A/B report giữa CPU và GPU run của cùng một model.
- **Claim**: "Chúng tôi là framework đầu tiên produce device-normalized, directly
  comparable profiling reports across CPU và CUDA với zero configuration."

#### Gap #2 — Heuristic Diagnosis Layer Chưa Được Formalize
- **Hiện trạng**: Tất cả tools (Nsight, PyTorch Profiler) dump raw data.
  Người dùng phải tự interpret. Không có tool nào có rule-based diagnosis.
- **Gap**: Không có paper nào formalize "profiling heuristic ruleset" cho
  vision models và validate accuracy của rule đó.
- **Claim**: "Chúng tôi propose và validate tập hợp N diagnostic rules đạt
  ≥X% accuracy so với manual expert diagnosis."

#### Gap #3 — Cross-Architecture Benchmark Trên T4-Class GPU Chưa Có
- **Hiện trạng**: MLPerf dùng A100/H100. Benchmark papers thường dùng V100+.
  T4 là GPU phổ biến nhất trong academia/Colab nhưng không được cover kỹ.
- **Gap**: Không có paper systematic study bottleneck profile của YOLO/ViT/ConvNeXt
  trên T4 vs. CPU-only, với per-layer granularity.
- **Claim**: "Empirical study đầu tiên về bottleneck profile của 5 vision
  architectures trên T4 GPU vs. CPU, revealing 3 unexpected performance inversion."

#### Gap #4 — Overhead Measurement Methodology Chưa Transparent
- **Hiện trạng**: Hầu hết papers không report profiling overhead của chính tool,
  hoặc report ở điều kiện tốt nhất (batch size nhỏ, single run).
- **Gap**: Không có standardized way để measure và report profiler overhead
  across platforms.

### 2.3 Positioning Template Cho Related Work Section

```
2.1 General-Purpose DL Profilers
    → PyTorch Profiler, TensorBoard: "General purpose, không có
      automated diagnosis, không tự normalize CPU/GPU output"
    → Nsight Systems/Compute: "Requires CUDA, không hoạt động trên
      CPU-only environments, steep learning curve, no heuristics"
    → Positioning: "Chúng tôi complement Nsight (dùng làm ground truth)
      thay vì replace, focus vào accessibility và automation"

2.2 Edge/Mobile Profiling Tools  
    → nn-Meter, BOLT: "Focus trên mobile hardware (ARM), không
      support CUDA path, không có diagnostic layer"
    → Positioning: "Chúng tôi target x86 CPU + NVIDIA GPU — different
      deployment scenario"

2.3 Performance Modeling Approaches
    → Habitat, Paleo: "Predictive (analytical model), không đo thực"
    → Positioning: "Chúng tôi là measurement-based, không prediction —
      complementary approach"

2.4 Benchmark Studies  
    → MLPerf: "Protocol-focused, không có tool contribution, 
      enterprise hardware only"
    → Positioning: "Chúng tôi target accessible hardware (Colab T4)
      — democratization angle"
```

---

## 3. THIẾT KẾ ĐÓNG GÓP KHOA HỌC (Contribution Design)

### 3.1 Phân Loại Đóng Góp

```
CONTRIBUTION MAP
│
├── [C1] SYSTEM CONTRIBUTION — Framework & Tool
│   ├── Portable profiling API (CPU/CUDA auto-detect)
│   ├── Layer-wise instrumentation via PyTorch Hooks
│   ├── Device-normalized metric format
│   └── Interactive dashboard (A/B comparison view)
│
├── [C2] METHODOLOGICAL CONTRIBUTION
│   ├── Overhead calibration protocol (làm thế nào để đo overhead
│   │   của chính profiler một cách chính xác)
│   └── Diagnostic heuristic ruleset (formalized, với validation)
│
└── [C3] EMPIRICAL CONTRIBUTION — Findings
    ├── Cross-architecture bottleneck profiles (5 models × 2 platforms)
    ├── Bottleneck inversion evidence (CPU rank ≠ GPU rank)
    ├── Quantified DataLoader overhead (num_workers impact)
    └── Conv2d vs. Attention compute pattern difference on T4
```

### 3.2 Concrete Claims Template

#### System Claims:
```
[SC1] "VisionProf reduces profiling overhead to X.X% throughput degradation 
       (vs. Y.Y% for PyTorch Profiler in full-trace mode) while providing 
       per-layer latency breakdown with <Z ms timing error on both 
       CPU and T4 GPU."

[SC2] "VisionProf's device-agnostic API requires ≤7 lines of code to 
       integrate into any PyTorch training/inference script, with zero 
       platform-specific configuration."

[SC3] "Our heuristic analyzer correctly identifies N out of M manually 
       diagnosed bottleneck cases (X% accuracy), including DataLoader 
       stalls, VRAM overflow risk, and compute-bound layer groups."
```

#### Empirical Claims:
```
[EC1] "We find that bottleneck ranking by layer type changes significantly 
       between CPU and T4 GPU: Conv2d is memory-bandwidth-bound on T4 
       but compute-bound on CPU, while self-attention shows the opposite 
       pattern — a finding not predictable from FLOP analysis alone."

[EC2] "YOLOv8 and YOLOv11 show X% of total inference latency concentrated 
       in the final N detection head layers on CPU, but only Y% on T4 — 
       suggesting architecture-specific optimization opportunities differ 
       by deployment target."

[EC3] "DataLoader with num_workers=0 introduces Z% artificial latency 
       inflation that is systematically misattributed to model forward 
       pass by existing profiling tools."

[EC4] "ViT-B/16 achieves higher throughput than YOLOv8-M on T4 GPU 
       but 2.3× lower throughput on CPU — a performance inversion 
       invisible in FLOP-based analysis."
```

### 3.3 Novelty Checklist

- [x] Cross-platform automated profiling → **falsifiable** (overhead số cụ thể)
- [x] Heuristic ruleset → **reproducible** (rules có thể public, validated by others)
- [x] Bottleneck inversion finding → **surprising & actionable**
- [x] T4-focused benchmark → **niche but real gap** (democratization angle)
- [ ] Cần verify: Không có paper nào 2023–2025 đã làm cross-platform Vision profiling với heuristics

---

## 4. THIẾT KẾ KIẾN TRÚC KỸ THUẬT (Technical Design)

### 4.1 Kiến Trúc Framework Tổng Thể

```
┌─────────────────────────────────────────────────────────────────────┐
│                        VisionProf Framework                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              LAYER 0: DEVICE DETECTION & ROUTING            │    │
│  │                                                              │    │
│  │   DeviceProbe.detect() → "cpu" | "cuda:0"                   │    │
│  │   ├── CPU path: time.perf_counter, psutil, no CUDA API      │    │
│  │   └── CUDA path: torch.cuda.Event, pynvml, CUDA streams     │    │
│  └───────────────────┬─────────────────────────────────────────┘    │
│                      │                                               │
│  ┌───────────────────▼─────────────────────────────────────────┐    │
│  │         LAYER 1: INSTRUMENTATION ENGINE                      │    │
│  │                                                              │    │
│  │  ┌──────────────────┐    ┌──────────────────────────────┐   │    │
│  │  │  PyTorch Hooks   │    │     Timing Backend           │   │    │
│  │  │                  │    │                              │   │    │
│  │  │ register_forward │    │  CPU: perf_counter_ns()      │   │    │
│  │  │ _pre_hook()      │    │  GPU: cuda.Event.elapsed()   │   │    │
│  │  │ register_forward │    │                              │   │    │
│  │  │ _hook()          │    │  Both: Warm-up (N=20 iters)  │   │    │
│  │  └──────────────────┘    └──────────────────────────────┘   │    │
│  │                                                              │    │
│  │  ┌──────────────────┐    ┌──────────────────────────────┐   │    │
│  │  │ Memory Tracker   │    │  DataLoader Sentinel         │   │    │
│  │  │                  │    │                              │   │    │
│  │  │ CPU: psutil      │    │  Wraps __iter__ to measure   │   │    │
│  │  │ GPU: pynvml      │    │  fetch time per batch        │   │    │
│  │  │ (async polling)  │    │                              │   │    │
│  │  └──────────────────┘    └──────────────────────────────┘   │    │
│  └───────────────────┬─────────────────────────────────────────┘    │
│                      │                                               │
│  ┌───────────────────▼─────────────────────────────────────────┐    │
│  │         LAYER 2: DATA COLLECTOR & NORMALIZER                 │    │
│  │                                                              │    │
│  │  Raw Trace → ProfilingRecord (normalized schema):            │    │
│  │  {layer_name, layer_type, latency_ms, memory_delta_mb,      │    │
│  │   device, flops_theoretical, arithmetic_intensity,           │    │
│  │   memory_bw_utilization, run_id, timestamp}                  │    │
│  │                                                              │    │
│  │  Storage: SQLite (local) / Parquet (export)                  │    │
│  └───────────────────┬─────────────────────────────────────────┘    │
│                      │                                               │
│  ┌───────────────────▼─────────────────────────────────────────┐    │
│  │         LAYER 3: ANALYSIS ENGINE                             │    │
│  │                                                              │    │
│  │  ┌─────────────────────┐  ┌──────────────────────────────┐  │    │
│  │  │  Bottleneck         │  │  A/B Comparison Engine       │  │    │
│  │  │  Classifier         │  │                              │  │    │
│  │  │                     │  │  CPU_run ↔ GPU_run           │  │    │
│  │  │  Roofline-based:    │  │  Model_A ↔ Model_B           │  │    │
│  │  │  compute-bound vs.  │  │                              │  │    │
│  │  │  memory-bound       │  │  Delta report, inversion     │  │    │
│  │  │  per layer          │  │  detection, ranking change   │  │    │
│  │  └─────────────────────┘  └──────────────────────────────┘  │    │
│  │                                                              │    │
│  │  ┌─────────────────────────────────────────────────────┐    │    │
│  │  │            Automated Heuristic Analyzer              │    │    │
│  │  │                                                      │    │    │
│  │  │  Rule Engine → Diagnosis → Recommendation           │    │    │
│  │  │  (see Section 4.3 for full ruleset)                  │    │    │
│  │  └─────────────────────────────────────────────────────┘    │    │
│  └───────────────────┬─────────────────────────────────────────┘    │
│                      │                                               │
│  ┌───────────────────▼─────────────────────────────────────────┐    │
│  │         LAYER 4: DASHBOARD / VISUALIZATION                   │    │
│  │  (Streamlit hoặc Plotly Dash — chạy local & Colab)           │    │
│  │                                                              │    │
│  │  • Waterfall chart: layer latency breakdown                  │    │
│  │  • Roofline plot: compute vs. memory bound per layer         │    │
│  │  • Side-by-side: CPU vs. GPU comparison                      │    │
│  │  • Heuristic report card: diagnosis + recommendation          │    │
│  │  • VRAM timeline (GPU mode only)                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Hook Implementation Chi Tiết

```python
# Core hook pattern — hoạt động trên cả CPU và CUDA
class LayerProfiler:
    def __init__(self, model, device):
        self.device = device
        self.records = {}
        self._hooks = []
        self._register_hooks(model)

    def _register_hooks(self, model):
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # leaf modules only
                pre = module.register_forward_pre_hook(
                    self._make_pre_hook(name))
                post = module.register_forward_hook(
                    self._make_post_hook(name))
                self._hooks.extend([pre, post])

    def _make_pre_hook(self, name):
        def hook(module, input):
            if self.device == "cpu":
                self.records[name] = {"start": time.perf_counter_ns()}
            else:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                self.records[name] = {"start_event": event}
        return hook

    def _make_post_hook(self, name):
        def hook(module, input, output):
            if self.device == "cpu":
                elapsed = (time.perf_counter_ns() - 
                          self.records[name]["start"]) / 1e6  # ms
            else:
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()
                torch.cuda.synchronize()
                elapsed = self.records[name]["start_event"].elapsed_time(
                    end_event)
            self.records[name]["latency_ms"] = elapsed
            self.records[name]["output_shape"] = (
                output.shape if hasattr(output, 'shape') else None)
        return hook

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
```

**Lưu ý quan trọng về hook:**
- Chỉ hook **leaf modules** (tránh double-counting nested modules)
- Trên GPU: **phải** `torch.cuda.synchronize()` trước khi `elapsed_time()`
- Warm-up: Chạy ít nhất 20 iterations trước khi bắt đầu record để tránh JIT/CUDA init bias
- Memory: Đo `torch.cuda.memory_allocated()` trước và sau forward để tính delta per-layer

### 4.3 Automated Heuristic Analyzer — Ruleset Đề Xuất

> **Nguyên tắc thiết kế**: Mỗi rule phải có threshold cụ thể, justification lý thuyết,
> và recommendation actionable. Rules phải được validate (Section 5).

```
RULE CATALOG v1.0

═══ GROUP A: DataLoader Bottleneck ═══

[R-A1] DataLoader Stall Detection
  Condition: dataloader_fetch_time > 0.3 × total_iter_time
  Severity: HIGH
  Diagnosis: "DataLoader là bottleneck — GPU/CPU đang chờ data"
  Recommendation: "Tăng num_workers=4 (hoặc os.cpu_count()//2),
                  bật pin_memory=True nếu dùng GPU"

[R-A2] num_workers=0 Warning  
  Condition: num_workers == 0 AND device == "cuda"
  Severity: MEDIUM
  Diagnosis: "Single-threaded data loading on GPU workload"
  Recommendation: "Đặt num_workers≥2, thử prefetch_factor=2"

═══ GROUP B: Compute Bottleneck ═══

[R-B1] Layer-Level Compute Saturation (GPU)
  Condition: SM_utilization > 90% AND memory_bw_util < 50%
             (sustained > 80% of forward pass time)
  Severity: INFO
  Diagnosis: "Model là compute-bound — GPU đang được dùng hiệu quả"
  Recommendation: "Cân nhắc Mixed Precision (AMP/FP16) để tăng throughput
                  thêm ~1.5–2× với compute-bound workload"

[R-B2] Conv2d Dominance
  Condition: sum(latency[Conv2d layers]) > 0.6 × total_forward_latency
  Severity: INFO
  Diagnosis: "Conv2d chiếm >60% latency — kiến trúc conv-heavy"
  Recommendation: "Nếu inference edge/CPU: cân nhắc MobileNetV3 hoặc
                  EfficientNet-Lite; nếu GPU: kiểm tra kernel fusion"

[R-B3] Attention Head Overhead (ViT)
  Condition: layer_type == MultiheadAttention AND
             latency > 2× median_layer_latency
  Severity: MEDIUM
  Diagnosis: "Attention layer là bottleneck đáng kể"
  Recommendation: "Thử Flash Attention 2 (nếu CUDA >= 8.0), hoặc
                  giảm sequence length / số head nếu accuracy cho phép"

═══ GROUP C: Memory Bottleneck ═══

[R-C1] VRAM Overflow Risk
  Condition: vram_used > 0.85 × vram_total
  Severity: CRITICAL
  Diagnosis: "VRAM sắp đầy — nguy cơ OOM error"
  Recommendation: "Giảm batch size, bật gradient_checkpointing,
                  dùng AMP (FP16 giảm ~50% VRAM)"

[R-C2] Memory-Bandwidth Saturation (GPU)
  Condition: memory_bw_utilization > 80% AND SM_util < 50%
  Severity: HIGH  
  Diagnosis: "Model là memory-bandwidth-bound — compute đang chờ data"
  Recommendation: "Layer fusion có thể giảm memory traffic;
                  quantization (INT8) có thể giảm BW requirement"

[R-C3] Activation Memory Spike
  Condition: max(memory_delta_per_layer) > 0.3 × total_vram_used
  Severity: MEDIUM
  Diagnosis: "Một số layer tạo activation memory rất lớn"
  Recommendation: "Xem xét gradient checkpointing tại layer này;
                  kiểm tra nếu có tensor được giữ không cần thiết"

═══ GROUP D: Platform-Specific ═══

[R-D1] FP32 on GPU (no AMP)
  Condition: device == "cuda" AND dtype == torch.float32
             AND throughput < expected_fp16_throughput × 0.6
  Severity: MEDIUM
  Diagnosis: "Chạy FP32 trên GPU — bỏ lỡ Tensor Core acceleration"
  Recommendation: "Bật torch.autocast('cuda') — T4 có Tensor Cores
                  tối ưu cho FP16, kỳ vọng speedup 1.5–2×"

[R-D2] CPU Inference với Model Lớn
  Condition: device == "cpu" AND 
             total_params > 50e6 AND
             avg_latency_per_batch > 500ms
  Severity: MEDIUM
  Diagnosis: "Model có thể quá lớn cho CPU inference real-time"
  Recommendation: "Cân nhắc ONNX Runtime với OpenVINO backend,
                  hoặc chuyển sang kiến trúc mobile-optimized"

[R-D3] Kernel Launch Overhead (GPU)
  Condition: device == "cuda" AND
             num_small_ops_per_forward > 500 AND
             median_kernel_time < 0.1ms
  Severity: LOW
  Diagnosis: "Nhiều kernel nhỏ — kernel launch overhead đáng kể"
  Recommendation: "Dùng torch.compile() (PyTorch 2.0+) để fuse kernels;
                  hoặc TorchScript cho production"
```

### 4.4 Xử Lý Profiling Overhead Bias

| Chiến lược | Cách implement | Expected overhead reduction |
|-----------|---------------|---------------------------|
| **Selective sampling** | Profile 1/N iterations (N=10 mặc định) | ~90% overhead reduction |
| **Async memory polling** | Thread riêng poll NVML mỗi 100ms | Tách overhead khỏi forward pass |
| **Leaf-only hooks** | Không hook container modules | ~40% hook overhead reduction |
| **No-copy tensor inspection** | `output.shape` thay vì `output.clone()` | Tránh memory allocation |
| **Overhead calibration run** | Chạy baseline (no hooks) + profiled, report delta | Transparency cho reviewer |

```python
# Overhead calibration protocol
def calibrate_overhead(model, sample_input, n_runs=100):
    """Measure profiler's own overhead. Must be reported in paper."""
    # Baseline (no profiling)
    times_baseline = []
    for _ in range(n_runs):
        t0 = time.perf_counter_ns()
        model(sample_input)
        times_baseline.append(time.perf_counter_ns() - t0)
    
    # With profiling
    profiler = LayerProfiler(model, device)
    times_profiled = []
    for _ in range(n_runs):
        t0 = time.perf_counter_ns()
        model(sample_input)
        times_profiled.append(time.perf_counter_ns() - t0)
    profiler.remove_hooks()
    
    overhead_pct = (np.median(times_profiled) - np.median(times_baseline)) / \
                   np.median(times_baseline) * 100
    return overhead_pct  # Report này PHẢI có trong paper
```

---

## 5. THIẾT KẾ THỰC NGHIỆM (Experimental Design)

### 5.1 Bộ Model Đánh Giá

| Model | Loại Architecture | Lý do chọn | Input size |
|-------|------------------|-----------|-----------|
| **YOLOv8-S** | CNN (Conv-heavy, multi-scale) | State-of-art detection, phổ biến nhất | 640×640 |
| **YOLOv11-S** | CNN (C2PSA, Attention hybrid) | Phiên bản mới nhất, hybrid architecture | 640×640 |
| **ConvNeXt-T** | Pure CNN (modernized) | ConvNet tối ưu, đối lập với ViT | 224×224 |
| **ViT-B/16** | Pure Attention | Benchmark Attention pattern | 224×224 |
| **MobileNetV3-Large** | Lightweight CNN (Depthwise Sep) | Edge deployment target | 224×224 |

> **Lý do 5 model này**: Cover full spectrum từ lightweight edge → heavy attention,
> đảm bảo H1 và H4 có thể được test. YOLO thêm vào đặc biệt vì detection head
> tạo ra multi-scale feature bottleneck không có ở classification models.

### 5.2 Workload Scenarios

```
Scenario 1: Single Image Inference (batch_size=1)
  → Đo latency thực tế kịch bản production edge deployment

Scenario 2: Batch Inference (batch_size=8, 16, 32)  
  → Đo throughput, memory scaling, batching efficiency
  → Chú ý: T4 có 16GB — ConvNeXt và ViT với bs=32 sẽ gần limit

Scenario 3: Warm vs. Cold Start
  → Đo CUDA init overhead, JIT compilation effect
  → Phân biệt "first run" vs. "steady state" latency

Scenario 4: DataLoader Integration
  → Dùng ImageFolder dataset với num_workers={0, 2, 4}
  → Đây là scenario để trigger Rule R-A1, R-A2

Scenario 5: FP32 vs. FP16 (AMP) — GPU only
  → Validate Rule R-D1, đo speedup thực tế trên T4
```

### 5.3 Baselines Để So Sánh

| Baseline | Platform | Điểm yếu cần demonstrate |
|---------|---------|--------------------------|
| `torch.profiler.profile()` — full trace | CPU + GPU | Overhead cao (~15-30% ở trace mode), không tự diagnosis |
| `time.perf_counter()` naive timing | CPU + GPU | Không sync CUDA → undercount GPU latency, không per-layer |
| NVIDIA Nsight Systems | GPU only | Không chạy trên CPU, cần CUDA toolkit, không portable |
| No profiling (baseline throughput) | Both | Ground truth cho overhead measurement |

### 5.4 Metrics Đánh Giá Bản Thân Framework

```
ACCURACY METRICS:
├── Timing accuracy: |profiler_ms - nsight_ms| / nsight_ms < 2%
│   (Nsight = ground truth cho GPU; perf_counter_ns = ground truth CPU)
├── Memory accuracy: |profiler_mb - cuda.memory_stats()_mb| < 1%
├── Heuristic precision: TP / (TP + FP) for each rule
├── Heuristic recall: TP / (TP + FN) for each rule
└── Attribution accuracy: sum(layer_latencies) ≈ total_forward_latency
    (Giải thích overhead nếu sum > total)

OVERHEAD METRICS:
├── Throughput degradation % (mức độ chậm hơn so với no-hook)
├── Latency overhead at p50, p95, p99
├── Peak memory overhead (framework's own memory)
└── Overhead vs. sample_rate curve (ablation A2)

USABILITY METRICS:
├── Integration LoC: ≤10 lines target
├── Setup time: ≤2 minutes từ pip install đến first report
└── Dashboard load time: ≤5s cho report 1000 layers
```

### 5.5 Ablation Study Questions

| ID | Câu hỏi | Cách thực hiện |
|----|---------|----------------|
| **AB-1** | Overhead breakdown: hook vs. memory polling vs. DB write? | Tắt từng component riêng lẻ, đo overhead |
| **AB-2** | Sample rate vs. accuracy tradeoff? | sample_rate = {1,5,10,20,50}, đo timing error |
| **AB-3** | Leaf-only hook vs. all-module hook: overhead và accuracy? | Toggle `leaf_only=True/False` |
| **AB-4** | Warm-up iterations needed? (bao nhiêu là đủ?) | N_warmup = {0,5,10,20,50}, đo variance |
| **AB-5** | Heuristic rule contribution: accuracy nếu bỏ từng group? | Remove Group A/B/C/D rules, đo overall accuracy |
| **AB-6** | FP16 vs. FP32: framework accuracy thay đổi không? | Profile same model ở hai precision mode |
| **AB-7** | Batch size ảnh hưởng đến bottleneck classification? | bs=1 vs. bs=16: compute-bound classification thay đổi? |

---

## 6. RỦI RO & PHƯƠNG ÁN DỰ PHÒNG (Risk Assessment)

### 6.1 Risk Register

| ID | Rủi ro | Xác suất | Mức độ | Fallback |
|----|--------|---------|--------|---------|
| **R1** | Paper concurrent về vision profiling xuất hiện trên ArXiv | Thấp | Cao | Narrow sang T4-specific + heuristic layer (hai gaps độc lập) |
| **R2** | T4 trên Colab bị preempt / timeout trong long experiments | Cao | Trung bình | Checkpoint từng model run; dùng Colab Pro hoặc GCP free tier |
| **R3** | PyTorch Hooks không đủ granularity cho một số custom ops | Trung bình | Trung bình | Thêm `torch.jit.trace` + manual timing cho edge cases |
| **R4** | Heuristic accuracy thấp (<75%) sau validation | Thấp-Trung | Cao | Thu hẹp ruleset về rules độ tin cậy cao; re-frame là "initial study" |
| **R5** | NVML không available hoặc inaccurate trên Colab | Trung bình | Trung bình | Fallback sang `torch.cuda.memory_stats()` + `nvidia-smi` parsing |
| **R6** | Overhead > 5% — không đạt target | Thấp | Cao | Tăng sample_rate mặc định; present trade-off curve |
| **R7** | Tất cả 5 model có bottleneck profile giống nhau (H4 sai) | Thấp | Cao | Vẫn là finding: "bottleneck profiles converge — architecture choice không quan trọng bằng nghĩ" |

### 6.2 Colab-Specific Risk Mitigation

```python
# Anti-preemption pattern cho Colab experiments
import pickle, os

def run_with_checkpoint(model_name, experiment_fn, checkpoint_dir="./checkpoints"):
    ckpt_path = f"{checkpoint_dir}/{model_name}_results.pkl"
    if os.path.exists(ckpt_path):
        print(f"[SKIP] {model_name}: checkpoint found, loading...")
        with open(ckpt_path, 'rb') as f:
            return pickle.load(f)
    
    results = experiment_fn()
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(ckpt_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"[SAVED] {model_name}: results checkpointed")
    return results
```

### 6.3 Hardware Scope Limitations — Phải Acknowledge trong Paper

> Reviewer sẽ hỏi về generalizability. Cần honest về limitation:

```
Limitations to acknowledge:
1. "Experiments conducted on T4 (Turing arch, 8.1 TFLOPS FP16). 
    Results may not generalize to Ampere (A100) or Hopper (H100) 
    due to different SM count, cache hierarchy, and Tensor Core generations."

2. "CPU experiments on single laptop — results reflect x86 
    Alder/Raptor Lake behavior. ARM (Apple Silicon, Snapdragon) 
    not covered in this work."

3. "Multi-GPU distributed profiling out of scope."

→ Framing: "Scope is intentionally focused on accessible hardware 
   (democratization goal), not enterprise-class systems."
```

---

## 7. ĐỊNH HƯỚNG PUBLICATION

### 7.1 Venue Phù Hợp Nhất

#### 🏆 Primary Target: **MLSys** (Proceedings of Machine Learning and Systems)

**Lý do phù hợp:**
- MLSys explicitly welcomes "tools and systems that enable ML research"
- Vision model profiling + heuristic diagnostic = strong system + empirical combo
- Cross-platform angle (edge CPU + cloud GPU) fits MLSys's growing interest in
  deployment-oriented systems
- T4/Colab = "accessible hardware" narrative phù hợp community MLSys

**Điều MLSys reviewer kỳ vọng:**
- Artifact: Framework phải open-source và reproducible
- Evaluation: ≥ 3 models, clear baseline, số liệu overhead cụ thể
- Finding: Non-obvious insight từ empirical study
- Generality: Framework không chỉ work cho 1 model

#### 🥈 Secondary Target: **EuroSys** hoặc **USENIX ATC**

**Lý do:**
- EuroSys accepts systems tools với strong empirical evaluation
- ATC accepts "measuring and understanding" papers nếu methodology mới
- Lower bar cho phần optimization contribution (20% của đề tài)

#### 🥉 Workshop Track (Test Water / Short Paper):

| Workshop | Venue | Notes |
|---------|-------|-------|
| MLSys Workshop on Systems for ML | Co-located MLSys | Test reception của community |
| NeurIPS Workshop on Efficient NLP (nếu expand) | NeurIPS | Nếu add LLM experiments sau |
| CVPR Workshop on Efficient CV | CVPR | Phù hợp hơn nếu nhấn vision angle |

### 7.2 Positioning Statement Cho Paper

```
"We present VisionProf, a portable, low-overhead profiling framework 
for vision neural networks that operates consistently across CPU-only 
and CUDA-accelerated environments without manual configuration. 

Unlike existing tools that require platform-specific setup and produce 
non-comparable outputs across hardware, VisionProf provides:
(1) unified per-layer profiling with <X% overhead,
(2) an automated diagnostic engine with N validated heuristic rules, 
(3) a comparative benchmark study revealing performance inversions 
    across 5 architectures on CPU vs. T4 GPU.

Our findings show that [key finding], challenging the conventional 
assumption that [conventional wisdom], and providing actionable 
guidance for practitioners deploying vision models on accessible hardware."
```

### 7.3 Pre-Submission Checklist

```
□ Framework open-sourced trên GitHub với README + example notebook
□ Colab demo notebook chạy được end-to-end trong <30 phút
□ Overhead measurement reported under worst-case (not best-case) conditions
□ Baseline comparison honest: acknowledge when Nsight gives more detail
□ Heuristic rules có validation section riêng với confusion matrix
□ Limitations section thẳng thắn về T4-only GPU scope
□ Reproducibility: random seed, PyTorch version, CUDA version documented
□ Artifact evaluation badge (MLSys yêu cầu)
```

---

## 8. BƯỚC TIẾP THEO CỤ THỂ (Actionable Next Steps)

### Phase 1 — Foundation (Ưu tiên cao nhất)

1. **Literature Scan** (1–2 ngày):
   - Đọc **DeepView** (MLSys'21) và **nn-Meter** (MobiSys'21) — hai paper gần nhất với hướng này
   - Search "vision model profiling cross-platform" trên Semantic Scholar, Paperswithcode
   - Verify Gap #1, #2, #3 chưa bị đóng bởi paper 2024–2025

2. **Feasibility Micro-Experiment** (2–3 ngày):
   - Implement `LayerProfiler` cơ bản (code ở Section 4.2)
   - Chạy trên YOLOv8-S ở cả CPU và T4, đo overhead thực tế
   - Nếu overhead > 10% → cần redesign trước khi commit toàn bộ

3. **Heuristic Rule Draft** (1 ngày):
   - Finalize ruleset từ Section 4.3
   - Xác định ground truth source để validate rules (Nsight Compute metric nào?)

### Phase 2 — Core Development

4. Implement đầy đủ 4 layers của framework
5. Chạy thực nghiệm 5 model × 2 platform × 5 scenarios
6. Validate heuristic rules (AB-5)
7. Build dashboard (Streamlit recommended cho Colab compatibility)

### Phase 3 — Paper Writing

8. Write empirical findings section trước (findings drive narrative)
9. Frame technical design section sau findings để show "tool enables discovery"
10. Ablation study → Appendix nếu bị cut vì page limit

---

*Kế hoạch này phản ánh đầy đủ hai ràng buộc đã xác định: (1) CPU laptop + T4 single GPU, (2) 80% systems profiling + 20% heuristic diagnostic. Phương Án A (LLM) và PA-C (Unified Metric) từ v1.0 đã được loại bỏ hoàn toàn.*
