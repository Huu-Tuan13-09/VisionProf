# Yêu cầu: đo lại TN3–TN5 trên Colab T4 và cập nhật CH3–CH4

> Gửi: TV2 (và agent code của TV2) · Từ: TV1 · Mức ưu tiên: **cao, đang chặn việc làm slide chung**
>
> Đọc kèm: `PROJECT_PLAN.md` Mục 5.2, 6, 8.1 · `agent-rules/04_profiling_and_scientific_rigor.md`

## 1. Vấn đề

Toàn bộ số liệu **Colab T4 của TN3, TN4, TN5** (39 thư mục trong `results/colab_t4/*/TN{3,4,5}/`) không phải
số đo. Chúng được tạo bởi `scripts/populate_tv2_benchmarks.py` từ các hằng số gõ sẵn
(`T4_BASE_P50`, `CPU_BASE_P50`, `T4_VRAM_MB`). Bằng chứng:

| Dấu hiệu | Ở đâu |
| :--- | :--- |
| P90 = P50 × 1.05, P95 = P50 × 1.08, P99 = P50 × 1.15 **chính xác** ở cả 39 thư mục | `summary.json` |
| Độ trễ từng vòng là dãy tăng đều (5.537, 5.5935, 5.65…); timestamp bắt đầu `1728000000.0` (10/2024), cách đều 0.01 s | `benchmark.csv` |
| Không có thông tin máy, phiên bản thư viện (bản đo thật trên laptop thì có) | `env.json` |
| Số trong báo cáo trùng hằng số của script, **kể cả số CPU**: MobileNetV3 CPU ghi 8.52 ms, nhưng `results/cpu_laptop/mobilenet_v3_large/TN3/bs1_fp32/summary.json` đo thật là 176.34 ms | `docs/tv2/BAO_CAO_TV2.md` (CH3, CH4) |
| FP16 giảm VRAM "−46.0%" giống hệt nhau ở mọi model | `docs/tv2/BAO_CAO_TV2.md` (CH4) |

Số liệu **CPU laptop** của TN3, TN4 là đo thật. README tổng đang ghi CH3–CH4 là "⏳ Đang cập nhật".

Nếu nộp như hiện tại, kết luận CH3–CH4 không dựa trên phép đo nào, và cả nhóm có thể bị xem là gian lận số liệu.
Code đo của TV2 (`src/model/`, `src/zoo/`, `scripts/run_experiment.py`) chạy tốt (đã ghép thử với
ResNet-50, DistilBERT, XGBoost), nên chỉ cần chạy thật và viết lại phần kết quả.

## 2. Lỗi phụ cần sửa trước khi đo: TN4 ghi đè độ dài câu

`notebooks/colab_runner.ipynb` (cell 5, `run_eval_t4_tn4`) chạy BERT/DistilBERT với `--seq-len 64/128/256`,
nhưng `src/core/paths.py::build_result_path` đặt tên thư mục là `bs{B}_{precision}`, **không có seq**.
3 lần chạy ghi vào cùng một thư mục, chỉ còn lại kết quả seq=256. TN3 (bs=1) lại đo seq=128, nên hai bảng
không so được với nhau. Số liệu `results/cpu_laptop/*/TN4` của BERT/DistilBERT cũng bị lỗi này.

Cách sửa gợi ý (không làm vỡ đường dẫn mà TV1, TV3 đang đọc): giữ `bs{B}_{precision}` khi `seq_len == 128`
(mặc định), và chỉ thêm hậu tố khi khác, ví dụ `bs8_seq256_fp32`. Nhớ cập nhật chỗ đọc trong
`experiments/tv2_analysis.py` và các tab dashboard của TV2 nếu cần.

## 3. Việc cần làm

1. `git pull` nhánh `main`, tạo nhánh mới, ví dụ `fix/tv2-real-t4-measurements`.
2. **Xoá số liệu sinh ra và script tạo ra nó**, để không ai dùng nhầm:
   ```bash
   git rm scripts/populate_tv2_benchmarks.py
   git rm -r results/colab_t4/*/TN3 results/colab_t4/*/TN4 results/colab_t4/*/TN5
   ```
3. Sửa lỗi ở Mục 2, chạy `pytest -q` và `python scripts/verify_rules.py` (phải pass).
4. **Đo thật trên Colab T4:** mở `notebooks/colab_runner.ipynb`, chọn runtime **T4 GPU**, chạy cell 5 và chọn
   chế độ **[2] GPU T4 chuyên sâu** (TN3 8 model, TN4 batch 1/8/32, TN5 FP16). Trong lúc đo không chạy việc
   khác trên runtime đó. Kết quả lưu sau mỗi lần chạy; nếu Colab ngắt thì chạy tiếp, lệnh nào lỗi thì
   chạy lại riêng lệnh đó.
5. Chạy lại TN4 của BERT/DistilBERT trên **laptop CPU** sau khi sửa Mục 2 (để có đủ seq 64/128/256).
6. **Tự kiểm tra**: đoạn dưới đây phải in ra `0 thư mục`:
   ```python
   import csv, glob, json
   bad = []
   for f in sorted(glob.glob("results/*/*/TN[345]/*/summary.json")):
       d = f.rsplit("/", 1)[0]
       s = json.load(open(f)); env = json.load(open(d + "/env.json"))
       p50 = s.get("p50_ms") or 0
       formula = p50 > 0 and abs(s["p90_ms"] - round(p50 * 1.05, 4)) < 1e-3 and abs(s["p99_ms"] - round(p50 * 1.15, 4)) < 1e-3
       ts = [float(r["timestamp"]) for r in csv.DictReader(open(d + "/benchmark.csv")) if r.get("timestamp")]
       if formula or "os" not in env or (ts and ts[0] == 1728000000.0):
           bad.append(d)
   print(f"{len(bad)} thư mục có dấu hiệu số liệu sinh ra", *bad[:5], sep="\n  ")
   ```
   Trên dữ liệu hiện tại, đoạn này in ra 39 và không báo nhầm thư mục CPU đo thật nào.
7. Vẽ lại hình: `python experiments/tv2_analysis.py` → `docs/tv2/figures/`.
8. **Viết lại kết quả theo số đo mới** trong `docs/tv2/BAO_CAO_TV2.md` (CH3, CH4), `docs/tv2/SLIDES_TV2.md`,
   `docs/tv2/README_TV2.md`, và 2 dòng CH3, CH4 trong bảng Mục 4 của `README.md`. Mọi con số phải lấy từ
   `results/`; số CPU lấy từ `results/cpu_laptop/`, không lấy từ hằng số cũ. Nếu kết quả thật khác kỳ vọng
   (ví dụ không có đảo thứ hạng), cứ ghi đúng như đo được: đó vẫn là một kết luận hợp lệ.
9. Mở Pull Request vào `main`, nhờ TV1 review (theo vòng review chéo ở Mục 8.3).

## 4. Điều kiện xong

- [ ] Không còn `scripts/populate_tv2_benchmarks.py`; đoạn kiểm tra ở bước 6 in `0 thư mục`.
- [ ] `results/colab_t4/` có đủ TN3 (8 model), TN4 (8 model × batch 1/8/32, NLP thêm seq 64/128/256), TN5 (7 model FP16), mỗi thư mục có `env.json` ghi máy Colab T4 thật.
- [ ] TN4 của BERT/DistilBERT không còn ghi đè giữa các độ dài câu, cả trên T4 lẫn laptop.
- [ ] Mọi số trong báo cáo, slide, README của TV2 khớp với `results/`.
- [ ] `pytest -q` và `python scripts/verify_rules.py` pass.

## 5. Lưu ý cho agent

- **Không** tạo, nội suy hay "điền" số liệu dưới bất kỳ hình thức nào. Thiếu số đo thì để trống và ghi rõ là thiếu.
- Không sửa file của TV1 (`src/layer/`, `experiments/tn1_*`, `tn2_*`, `tv1_*`, `docs/tv1/`) hay TV3 (`src/phase/`,
  `src/analyzer/`, `docs/tv3/`). Plugin `layer` chạy ở lượt đo riêng sau benchmark, nên không làm lệch P50/P99.
- Không dùng `torch.compile`, TensorRT hay thay đổi model ngoài phạm vi Mục 10 của kế hoạch.
- Commit chỉ ghi tên TV2, không thêm dòng đồng tác giả là AI.
