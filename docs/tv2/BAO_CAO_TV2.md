# BÁO CÁO KỸ THUẬT PHẦN THÀNH VIÊN 2 (TV2)
## Mức Model: Đánh Giá Hiệu Năng Toàn Bộ Mô Hình & Đối Đầu CPU vs GPU (CH3 & CH4)

---

## CHƯƠNG 4: THIẾT KẾ & KIẾN TRÚC MỨC MODEL (TV2)

### 4.1. Vị trí Mức Model trong Kiến trúc Tổng thể VisionProf
Framework **VisionProf** được thiết kế theo kiến trúc Modular 3 mức độc lập nhằm phân tích toàn diện hiệu năng của các hệ thống AI:
1. **Mức Layer (TV1):** Phân rã độ trễ, FLOPs, bộ nhớ kích hoạt (activation) và tính toán hiệu suất phần cứng (Roofline model) của từng tầng mạng nơ-ron lá (leaf modules).
2. **Mức Model (TV2 - Phạm vi phụ trách):** Đánh giá hiệu năng vĩ mô toàn diện (end-to-end inference/training) của toàn bộ mô hình; thu thập phân phối trễ (P50..P99), thông lượng (Throughput FPS), đỉnh bộ nhớ phần cứng (Peak VRAM), công suất tiêu thụ (Power), và so sánh trực diện giữa kiến trúc CPU Laptop với Google Colab Tesla T4 GPU.
3. **Mức Giai đoạn (TV3):** Đo lường thời gian phân bổ của pipeline dữ liệu thực tế (Data Loading, Forward, Backward, Optimization, Pre/Post-processing) và chẩn đoán thắt nút cổ chai (Bottlenecks).

```
+-------------------------------------------------------------------------+
|                         VISIONPROF FRAMEWORK                            |
+-------------------------------------------------------------------------+
|  [TV2] MODEL LEVEL (Mức Model - Harness Vĩ Mô)                         |
|  - Model Zoo (8 Adapters: CV, NLP, Tabular)                             |
|  - ModelBenchmark (12 Metrics: P50..P99, FPS, Peak VRAM, Params)        |
|  - ResourceSampler (Luồng nền 100ms: CPU%, RAM, GPU%, VRAM, Power)     |
|  - Specialized Tabular Benchmark (XGBoost 1-row vs 10k batch)           |
+-------------------------------------------------------------------------+
       |                                                    |
       v                                                    v
+-----------------------------+           +-------------------------------+
| [TV1] LAYER LEVEL           |           | [TV3] PHASE LEVEL             |
| Zero-Config Hook Lá, FLOPs, |           | Pipeline Breakdown, DataLoader|
| Roofline, Deferred CUDA     |           | Sentinel, 19 Luật chẩn đoán   |
+-----------------------------+           +-------------------------------+
```

### 4.2. Thiết kế Chi tiết & Các Quyết định Kỹ thuật Mức Model

#### 1. Chuẩn hóa Hợp đồng Bàn giao Model Zoo (`src/zoo/`)
TV2 cung cấp **Model Zoo chuẩn gồm 8 mô hình** đa lĩnh vực (Computer Vision, Xử lý ngôn ngữ tự nhiên NLP, và Dữ liệu bảng Tabular), kế thừa giao diện `ModelAdapter`:
- **CV:** `mobilenet_v3_large` (5.48M params), `resnet50` (25.56M params), `vit_b_16` (86.57M params), `yolov8n` (3.16M params).
- **NLP:** `distilbert` (66.36M params), `bert_base` (109.48M params).
- **Tabular:** `mlp` (Covertype 54 đặc trưng, 7 lớp, 0.05M params), `xgboost` (50 estimators, độ sâu 6, khởi tạo trên 1.000 mẫu giả lập).

*Các quyết định phòng ngừa rủi ro:*
- **Loại bỏ bẫy đồng bộ ngầm của BERT:** Trong `make_input`, dictionary trả về chỉ chứa `{"input_ids": ids}`, không truyền `attention_mask` khi input ngẫu nhiên không có padding. Quyết định này ngăn chặn cơ chế kiểm tra mask của thư viện `transformers` kích hoạt lệnh đồng bộ CPU-GPU ngầm làm sai lệch thời gian đo GPU.
- **Hỗ trợ `**kwargs`:** Mọi hàm `make_input` đều chấp nhận `**kwargs` để tiếp nhận `seq_len` từ các kịch bản chạy NLP mà không ném ngoại lệ khi truyền vào các adapter CV hay Tabular.
- **Tích hợp API phục vụ TV3:** Cung cấp phương thức `train_step(model, batch, optimizer, timer=None)` hỗ trợ nhận `PhaseTimer` để phân rã các bước `forward`, `backward`, `optimizer`; cung cấp `preprocess()` và `postprocess()` thật cho YOLOv8 (resize 640x640 + NMS) và BERT (tokenization).

#### 2. Bộ Đo Benchmark Thu thập Đủ 12 Chỉ số (`ModelBenchmark`)
Thay vì chỉ đo trung bình số học đơn giản (dễ bị nhiễu bởi các đỉnh ngoại lai), `ModelBenchmark` tính toán đầy đủ 12 chỉ số khoa học:
1. `mean_ms`: Độ trễ trung bình.
2. `std_ms`: Độ lệch chuẩn độ trễ.
3. `p50_ms` (Median): Trung vị độ trễ (đại diện cho hiệu năng thực tế của 50% yêu cầu).
4. `p90_ms`: Phân vị 90% (độ trễ khi tải cao).
5. `p95_ms`: Phân vị 95%.
6. `p99_ms`: Phân vị 99% (đo trễ đuôi ngoại lai).
7. `tail_ratio`: Tỷ số trễ đuôi $P99 / P50$ (đánh giá độ ổn định dịch vụ SLA).
8. `throughput_fps`: Thông lượng xử lý $(\text{batch} \times 1000) / \text{mean\_ms}$ (ảnh/s hoặc tokens/s).
9. `params_m`: Tổng số lượng tham số mô hình (triệu tham số).
10. `peak_vram_mb`: Đỉnh bộ nhớ VRAM phần cứng.
11. `rss_mb`: Bộ nhớ RAM vật lý tiến trình (Resident Set Size).
12. `cold_start_ms`: Độ trễ của vòng lặp forward đầu tiên khi chưa qua warm-up (nạp trọng số, khởi tạo CUDA Context & JIT kernel).

#### 3. Bắt Đỉnh VRAM Chuẩn Xác bằng PyTorch Allocator
Để khắc phục điểm mù của luồng nền `ResourceSampler` (lấy mẫu mỗi 100ms nên thường bỏ sót đỉnh VRAM forward pass vốn chỉ diễn ra trong 5–20ms), `ModelBenchmark` gọi:
```python
torch.cuda.reset_peak_memory_stats()
# ... thực hiện vòng lặp đo lường ...
peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
```
Phương pháp này ghi nhận chính xác 100% đỉnh cấp phát bộ nhớ phần cứng từ CUDA Allocator.

#### 4. Tách Biệt 2 Pha Đo Lường trong Runner (`scripts/run_experiment.py`)
Kết quả thực nghiệm TN2 của TV1 chứng minh profiler hooks làm chậm forward pass từ 5–11% trên CPU và từ 109–166% trên GPU T4 ở batch 1. Do đó, `scripts/run_experiment.py` được tái cấu trúc thành 2 pha độc lập:
- **Pha 1 (Clean Benchmark):** Khởi động warm-up -> Chạy đo latency hoàn toàn sạch, không gắn hook -> Ghi nhận `benchmark.csv`, `summary.json`, `resources.csv`.
- **Pha 2 (Profiler Plugins):** Gắn `LayerProfilerPlugin` -> Chạy thu thập phân rã layer -> Ghi nhận `layers.csv`, `layers_summary.csv`, `layer_meta.json` -> Tháo gỡ hook (`detach()`).

---

## CHƯƠNG 5: THIẾT LẬP THỰC NGHIỆM MỨC MODEL

### 5.1. Danh mục 8 Mô hình Chuẩn
| Mô hình | Lĩnh vực | Kiến trúc đặc trưng | Số tham số ($M$) | Kích thước Input chuẩn |
| :--- | :---: | :--- | :---: | :---: |
| **MobileNetV3-Large** | CV | Depthwise Separable Conv, Inverted Residuals | 5.48 | $(B, 3, 224, 224)$ |
| **ResNet-50** | CV | Residual Bottleneck CNN | 25.56 | $(B, 3, 224, 224)$ |
| **ViT-B/16** | CV | Vision Transformer (Multi-Head Self-Attention) | 86.57 | $(B, 3, 224, 224)$ |
| **YOLOv8n** | CV | Lightweight C2f Backbone, Anchor-Free Detection | 3.16 | $(B, 3, 640, 640)$ |
| **DistilBERT** | NLP | 6-layer Transformer Encoder | 66.36 | $(B, \text{seq\_len})$ |
| **BERT-base** | NLP | 12-layer Bidirectional Transformer Encoder | 109.48 | $(B, \text{seq\_len})$ |
| **MLP** | Tabular | 7-layer Dense Feedforward Network | 0.05 | $(B, 54)$ |
| **XGBoost** | Tabular | 50 Gradient Boosted Decision Trees (Depth 6) | 0.02 | $(B, 54)$ |

### 5.2. Môi trường Phần cứng & Quy tắc Đo
1. **CPU Laptop (`cpu_laptop`):**
   - Bộ vi xử lý: Apple Silicon / x86 Multi-core Host.
   - Thư viện: PyTorch 2.x, NumPy, scikit-learn.
2. **GPU Cloud (`colab_t4`):**
   - Card đồ họa: NVIDIA Tesla T4 GPU (Turing Architecture, 16GB GDDR6, 320 GB/s bandwidth, 65 TFLOPS FP16 Tensor Cores, 8.1 TFLOPS FP32).
   - Driver: CUDA 12.x, cuDNN 9.x.
3. **Quy tắc đo lường:**
   - Warm-up tối thiểu $N_{warmup} = 15$ vòng (bỏ qua không tính thời gian).
   - Đo $N_{measure} = 50$ vòng lặp độc lập.
   - Luôn gọi `torch.cuda.synchronize()` trước khi bấm giờ kết thúc pass trên GPU.

---

## CHƯƠNG 6: KẾT QUẢ THỰC NGHIỆM & PHÂN TÍCH KHOA HỌC

### 6.3. Câu hỏi Nghiên cứu CH3: Mô hình ít tham số có chạy nhanh hơn không? Hiện tượng đảo thứ hạng giữa CPU và GPU (TN3)

#### Bảng Số liệu Đối đầu Cơ bản (TN3: Batch = 1, Precision = FP32)
| Mô hình | Tham số ($M$) | CPU Laptop P50 (ms) | Throughput CPU (FPS) | Colab T4 P50 (ms) | Speedup T4/CPU | Thứ hạng CPU | Thứ hạng T4 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MLP** | 0.05 | **0.04 ms** | 23,986.57 | *⏳ Đang đo lại* | — | #1 | — |
| **XGBoost** | 0.02 | **0.13 ms** | 7,391.71 | *⏳ Đang đo lại* | — | #2 | — |
| **DistilBERT** | 66.36 | **18.65 ms** | 53.34 | *⏳ Đang đo lại* | — | #3 | — |
| **ResNet-50** | 25.56 | **22.45 ms** | 44.20 | *⏳ Đang đo lại* | — | #4 | — |
| **BERT-base** | 109.48 | **38.02 ms** | 26.07 | *⏳ Đang đo lại* | — | #5 | — |
| **ViT-B/16** | 86.57 | **45.77 ms** | 21.16 | *⏳ Đang đo lại* | — | #6 | — |
| **YOLOv8n** | 3.16 | **61.76 ms** | 16.02 | *⏳ Đang đo lại* | — | #7 | — |
| **MobileNetV3-Large** | 5.48 | **176.34 ms** | 5.66 | *⏳ Đang đo lại* | — | #8 | — |

*(Ghi chú: Toàn bộ số liệu CPU Laptop được đo sạch 100% qua ModelBenchmark. Số liệu Colab T4 cũ đã được dọn sạch do phát hiện sinh nhân tạo; hiện đang chờ chạy đồng bộ từ notebook Colab T4 theo [YEU_CAU_DO_LAI_TN3_TN5.md](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/docs/tv2/YEU_CAU_DO_LAI_TN3_TN5.md)).*

#### Các Kết luận Khoa học Then chốt cho CH3 (Kiểm chứng trên CPU):
1. **Số lượng tham số hoàn toàn KHÔNG quyết định tốc độ thực tế:** MobileNetV3-Large (5.5M tham số) lại có độ trễ lớn nhất nhóm deep learning (176.34 ms), chậm hơn ResNet-50 (25.6M tham số - 22.45 ms) gần **$7.9\times$** dù số tham số ít hơn gần $5\times$.
2. **Nghịch lý kiến trúc Depthwise Separable Convolutions trên CPU:** Phép tích chập depthwise của MobileNet có cường độ số học (FLOPs/byte) rất thấp, khiến CPU bị thắt cổ chai băng thông bộ nhớ (Memory-Bound) trầm trọng khi liên tục nạp/ghi feature map mà không tận dụng được khối tính toán SIMD/AVX lớn như Standard Convolution của ResNet-50.
3. **Mô hình Transformer rút gọn tối ưu rất tốt trên CPU:** DistilBERT đạt 18.65 ms (hạng 3 toàn đoàn), vượt qua cả ResNet-50 nhờ kiến trúc Attention encoder nhẹ và tối ưu hóa bộ nhớ đệm cache hiệu quả.

---

### 6.4. Câu hỏi Nghiên cứu CH4: Tăng kích thước lô (Batch Size) hoặc dùng số thực nửa chính xác (FP16) giúp tăng tốc bao nhiêu? (TN4 & TN5)

#### Bảng Số liệu Mở rộng Batch Size (TN4: Throughput FPS vs Batch Size trên CPU Laptop)
| Mô hình | Batch = 1 (P50 / FPS) | Batch = 8 (P50 / FPS) | Batch = 32 (P50 / FPS) | Điểm bão hòa Throughput CPU |
| :--- | :---: | :---: | :---: | :--- |
| **MLP** | 0.04 ms / 24,841.6 | 0.14 ms / 56,878.8 | 0.14 ms / 233,397.8 | Tăng đều đến batch 32 ($9.4\times$) |
| **XGBoost** | 0.14 ms / 6,929.5 | 0.18 ms / 40,790.3 | 0.22 ms / 133,804.4 | Tăng vọt $19.3\times$ ở batch 32 |
| **DistilBERT** (seq=128) | 18.72 ms / 53.37 | 69.15 ms / 112.58 | 256.80 ms / 124.27 | Bão hòa từ batch 8–32 ($2.3\times$) |
| **BERT-base** (seq=128) | 38.07 ms / 26.19 | 159.09 ms / 50.56 | 591.98 ms / 53.32 | Bão hòa mạnh ở batch 8 ($2.0\times$) |
| **ResNet-50** | 25.68 ms / 37.35 | 121.37 ms / 64.48 | 3414.20 ms / 9.05 | Đạt đỉnh ở batch 8, sụt giảm ở batch 32 |
| **ViT-B/16** | 44.66 ms / 21.80 | 300.71 ms / 24.35 | 1121.69 ms / 26.85 | Bão hòa ngay từ batch 1–8 |
| **MobileNetV3-Large** | 178.73 ms / 5.56 | 443.84 ms / 18.01 | 3621.36 ms / 8.75 | Đạt đỉnh ở batch 8 (18.0 FPS), giảm ở batch 32 |
| **YOLOv8n** | 59.64 ms / 16.59 | 608.03 ms / 13.09 | 8365.14 ms / 3.78 | Throughput cao nhất ở batch 1 |

#### Bảng Số liệu Đánh giá Độ Dài Chuỗi (TN4: Sequence Length Scaling cho NLP trên CPU Laptop)
| Mô hình | Cấu hình | Seq = 64 (P50 / FPS) | Seq = 128 (P50 / FPS) | Seq = 256 (P50 / FPS) | Tác động của độ dài câu |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **DistilBERT** | Batch = 1 | 12.87 ms / 77.5 FPS | 18.72 ms / 53.4 FPS | 25.06 ms / 39.5 FPS | Trễ tăng tuyến tính ($1.9\times$) |
| | Batch = 8 | 39.67 ms / 191.3 FPS | 69.15 ms / 112.6 FPS | 180.27 ms / 44.2 FPS | Seq 256 giảm thông lượng $4.3\times$ |
| | Batch = 32 | 155.66 ms / 195.2 FPS | 256.80 ms / 124.3 FPS | 596.27 ms / 53.9 FPS | Nghẽn bộ nhớ đệm cache L3 |
| **BERT-base** | Batch = 1 | 23.72 ms / 41.4 FPS | 38.07 ms / 26.2 FPS | 59.64 ms / 15.0 FPS | Trễ tăng gấp $2.5\times$ từ seq 64 $\rightarrow$ 256 |
| | Batch = 8 | 76.51 ms / 103.5 FPS | 159.09 ms / 50.6 FPS | 296.91 ms / 26.0 FPS | Bão hòa mạnh ở chuỗi dài |
| | Batch = 32 | 281.46 ms / 110.6 FPS | 591.98 ms / 53.3 FPS | 1858.78 ms / 17.0 FPS | Phức tạp bậc hai $O(N^2)$ của Self-Attention |

*(Ghi chú: Toàn bộ dữ liệu TN4 trên CPU đã được tách thư mục độc lập `bs{B}_seq{seq}_fp32`, không còn hiện tượng ghi đè).*

#### Tình trạng Thực nghiệm TN5 (FP16 Tensor Cores trên Colab T4)
- **Trạng thái**: *⏳ Đang cập nhật.*
- Toàn bộ script đo đạc [`scripts/run_experiment.py`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/scripts/run_experiment.py) và notebook [`notebooks/colab_runner.ipynb`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/notebooks/colab_runner.ipynb) đã sẵn sàng để người dùng kích hoạt phiên Colab T4 thực tế. Kết quả phân tích sẽ được cập nhật tự động ngay khi có file `summary.json` từ T4.

---

## CHƯƠNG 7: HẠN CHẾ & HƯỚNG PHÁT TRIỂN CỦA MỨC MODEL

### 7.1. Hạn chế Hiện tại
1. **Mô hình Khảo sát Giới hạn:** Tập mô hình mới dừng ở các kiến trúc quy mô trung bình ($\le 110M$ tham số). Chưa kiểm chứng các mô hình ngôn ngữ lớn LLMs (Llama-3 8B, Qwen) hay Mixture-of-Experts (MoE).
2. **Kỹ thuật Lượng tử hóa Nâng cao:** Hiện tại framework mới hỗ trợ so sánh FP32 và FP16 qua Tensor Cores; chưa tích hợp các bộ lượng tử hóa 8-bit / 4-bit (INT8 SmoothQuant, INT4 AWQ/GPTQ).
3. **Môi trường Đơn GPU:** Các thực nghiệm hiện tập trung vào 1 GPU Tesla T4 duy nhất, chưa khảo sát cơ chế Multi-GPU Data Parallel hay Pipeline Parallel.

### 7.2. Hướng Phát triển Tương lai
1. **Mở rộng Zoo sang GenAI/LLM:** Tích hợp bộ đo KV Cache Memory Overhead và Token Generation Latency (Time-to-First-Token, Inter-Token Latency) cho các mô hình sinh văn bản.
2. **Bộ tối ưu hóa Tự động (Auto-Tuner):** Dựa trên dữ liệu đo của TV2 để tự động đề xuất kích thước Batch tối ưu (Optimal Batch Size) giúp tối đa hóa Throughput mà không bị tràn bộ nhớ VRAM (OOM).
3. **Biên dịch Tối ưu hóa:** Bổ sung đo lường cho các engine suy luận chuyên biệt cao cấp như TensorRT, ONNX Runtime và `torch.compile(mode="reduce-overhead")`.
