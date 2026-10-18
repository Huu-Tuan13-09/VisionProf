# Kết quả đo Colab T4 cho giữa kỳ (18/10/2026)

> Chạy bằng `notebooks/colab_midterm_runs.ipynb` trên Colab **Tesla T4, 2 vCPU**. 57 lần chạy, đã kiểm tra:
> không có số liệu sinh ra, đủ TN3 8/8, TN4 36/36, TN5 7/7, TN6 6/6. Số liệu gốc nằm trong `results/colab_t4/`.
> Cột "CPU Mac" là MacBook chip Apple (arm64) của TV2 (`results/cpu_laptop/*/TN3, TN4`), **không phải** laptop x86 của TV1.
> Dùng bảng này để điền các ô `[Điền ...]` trong `docs/DAN_Y_SLIDE.md`.

## CH2: kiểm chứng lại code mức Layer đã sửa (TV1 · Slide 8)

| | ResNet-50 (báo cáo → code đã sửa) | BERT-base (báo cáo → code đã sửa) |
| :--- | :-: | :-: |
| Tổng thời gian các layer | 5.10 → 5.13 ms | 6.27 → 6.30 ms |
| Sai lệch so với `torch.profiler` | 7.5% → 7.8% | 0.8% → 0.04% |
| Thời gian không thuộc layer nào | 9.0% → 9.5% | 10.3% → 10.5% |
| Overhead (thay đổi giữa các phiên Colab) | 166% → 213% | 109% → 133% |

Kết luận: code hiện tại cho **cùng kết quả** với số trong báo cáo. Overhead GPU ở batch 1 dao động giữa các phiên;
nên nói "khoảng 110–210%". Số gốc: `results/analysis/TN2/recheck_fixed_code/`.

## CH3: batch 1, FP32 (TV2 · Slide 12–13)

| Mô hình | Tham số (M) | T4 P50 (ms) | T4 P99 (ms) | VRAM T4 (MB) | CPU Mac P50 (ms) | Hạng T4 | Hạng Mac |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| MLP | 0.05 | 0.28 | 0.64 | 9 | 0.04 | 1 | 1 |
| XGBoost | — | 1.56 | 2.10 | — | 0.13 | 2 | 2 |
| DistilBERT | 66.4 | 5.34 | 8.91 | 269 | 18.65 | 3 | 3 |
| ResNet-50 | 25.6 | 5.96 | 10.46 | 117 | 22.45 | 4 | 4 |
| MobileNetV3-L | 5.5 | 5.99 | 9.21 | 38 | 176.34 | 5 | 8 |
| YOLOv8n | 3.2 | 7.52 | 8.75 | 49 | 61.76 | 6 | 7 |
| BERT-base | 109.5 | 8.58 | 14.17 | 433 | 38.02 | 7 | 5 |
| ViT-B/16 | 86.6 | 12.99 | 13.66 | 348 | 45.77 | 8 | 6 |

Kết luận đo được:
1. **Ở batch 1 trên T4, số tham số / FLOPs không quyết định độ trễ:** ResNet-50 (8.2 GFLOPs) và MobileNetV3 (0.43 GFLOPs, ít hơn 19 lần)
   cùng khoảng 6 ms. GPU đang bị giới hạn bởi chi phí gửi kernel, khớp với CH1 (layer nhỏ chiếm tỷ trọng lớn hơn trên GPU).
2. **Mô hình rất nhỏ chạy GPU còn chậm hơn CPU** (MLP 0.28 ms trên T4 so với 0.04 ms trên Mac).
3. Thứ hạng có thay đổi giữa Mac và T4 (MobileNetV3 8 → 5, BERT-base 5 → 7), nhưng phần lớn do MobileNetV3 chạy
   bất thường chậm trên CPU ARM của Mac (176 ms; trên laptop x86 của TV1 khoảng 12 ms). Nên trình bày đúng như vậy,
   không gọi là "đảo thứ hạng do GPU" như kỳ vọng ban đầu.

## CH4: batch và độ dài câu (TV2 · Slide 14)

Throughput trên T4 (ảnh hoặc câu / giây, NLP seq = 128):

| Mô hình | Batch 1 | Batch 8 | Batch 32 | Tăng 32/1 |
| :--- | :-: | :-: | :-: | :-: |
| MobileNetV3-L | 170 | 1 094 | 1 499 | **8.8×** |
| ResNet-50 | 153 | 287 | 337 | 2.2× |
| YOLOv8n | 118 | 198 | 235 | 2.0× |
| ViT-B/16 | 75 | 96 | 94 | 1.3× (bão hoà từ batch 8) |
| DistilBERT | 198 | 287 | 300 | 1.5× |
| BERT-base | 100 | 134 | 140 | 1.4× |

- Mô hình nhẹ hưởng lợi nhiều nhất khi tăng batch (MobileNetV3 8.8×) vì batch 1 chưa lấp đầy GPU; mô hình nặng
  (ViT, BERT) gần bão hoà ngay từ batch 8.
- Độ dài câu: BERT-base batch 32 mất 127 ms (seq 64) → 228 ms (128) → 507 ms (256): **mỗi lần gấp đôi độ dài, độ trễ
  tăng 1.8× rồi 2.2×**, nhanh hơn tuyến tính (phần Attention tăng theo bình phương). DistilBERT tương tự: 59 → 107 → 248 ms.

## CH4: FP16 so với FP32 trên T4, batch 1 (TV2 · Slide 15)

| Mô hình | FP32 (ms) | FP16 (ms) | Tăng tốc | VRAM giảm | Cosine FP16 / FP32 |
| :--- | :-: | :-: | :-: | :-: | :-: |
| ViT-B/16 | 12.99 | 4.62 | **2.81×** | 45% | không xác định* |
| DistilBERT | 5.34 | 3.44 | 1.55× | 45% | 0.999999 |
| BERT-base | 8.58 | 6.94 | 1.23× | 46% | 0.999999 |
| MLP | 0.28 | 0.28 | 0.99× | 1% | 1.000000 |
| ResNet-50 | 5.96 | 6.88 | **0.87×** | 43% | 1.000000 |
| MobileNetV3-L | 5.99 | 7.02 | 0.85× | 38% | không xác định* |
| YOLOv8n | 7.52 | 9.76 | **0.77×** | 49% | 1.000000 |

- **FP16 chỉ nhanh hơn với Transformer** (nhân ma trận lớn, dùng được Tensor Core). Với CNN ở batch 1, FP16 còn
  **chậm hơn** vì thời gian bị chi phối bởi gửi kernel, không phải tính toán. VRAM giảm 38–49% với mọi mô hình có trọng số lớn.
- *Cosine không xác định vì mô hình dùng trọng số khởi tạo ngẫu nhiên: torchvision khởi tạo lớp cuối của ViT bằng 0
  (output luôn 0), MobileNetV3 cho output cỡ 1e-10 nên FP16 tràn dưới thành 0. Không phải lỗi FP16.

## CH5: nạp dữ liệu có làm GPU phải chờ? (TV3 · thêm vào Slide 17)

DataLoader thật: ảnh JPEG giải mã + augmentation trên CPU, batch 32, Colab 2 vCPU (`scripts/run_tn6_dataloader.py`).

| Mô hình | `num_workers` | Chờ dữ liệu (ms) | Một bước (ms) | % chờ dữ liệu | Ảnh / giây |
| :--- | :-: | :-: | :-: | :-: | :-: |
| MobileNetV3-L | 0 | 151.4 | 241.8 | **62.6%** | 132 |
| MobileNetV3-L | 2 | 3.9 | 116.9 | 3.3% | **274** |
| MobileNetV3-L | 4 | 3.9 | 119.8 | 3.2% | 267 |
| ResNet-50 | 0 | 139.2 | 462.5 | **30.1%** | 69 |
| ResNet-50 | 2 | 2.0 | 338.2 | 0.6% | **95** |
| ResNet-50 | 4 | 2.0 | 356.3 | 0.6% | 90 |

- Không có worker phụ, GPU ngồi chờ dữ liệu **63%** thời gian với mô hình nhẹ, 30% với mô hình nặng. 2 worker đưa
  con số này về dưới 4% và tăng gấp đôi tốc độ huấn luyện MobileNetV3.
- 4 worker **không nhanh hơn** 2 vì Colab chỉ có 2 vCPU; forward còn chậm đi một chút vì worker tranh CPU với tiến
  trình chính. Khớp với nghiên cứu Data Stalls (Mohan et al., PVLDB 2021).

## Còn lại để TV2 / TV3 tự sửa

- Nhãn "CPU Laptop" trong `docs/tv2/figures/*.png` thực chất là MacBook của TV2: đổi nhãn (hoặc tên thư mục máy).
- `docs/tv2/BAO_CAO_TV2.md`, `SLIDES_TV2.md`, dòng CH3, CH4 trong `README.md`: viết lại theo các bảng trên.
- `docs/tv3/`: thêm kết quả nạp dữ liệu vào phần CH5.
