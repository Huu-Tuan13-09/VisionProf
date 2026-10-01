# AI Agent Instructions for VisionProf

> **QUAN TRỌNG**: Dự án này có bộ quy tắc kỹ thuật chuyên sâu đặt tại thư mục [`agent-rules/`](file:///Users/congtri/IT/Dai_Hoc/Xu_ly_du_lieu/VisionProf/agent-rules).  
> Mọi AI Agent tham gia đọc hiểu, viết code, sửa lỗi hoặc tái cấu trúc trong kho lưu trữ này **BẮT BUỘC** phải tuân thủ nghiêm ngặt các quy định dưới đây.

---

## ⚡ 5 Nguyên Tắc Bắt Buộc (Golden Rules)

1. **Giới hạn kích thước file & Đơn trách nhiệm (SRP)**: 
   - **Tối đa 300 dòng code/file** (Hard limit). Khuyến nghị $\le 250$ dòng.
   - Mỗi file một mục đích duy nhất. Tuyệt đối không tạo monolithic file.
   - Chi tiết: [`01_coding_standards.md`](./agent-rules/01_coding_standards.md)

2. **Kiến trúc 3 mức đo & Hợp đồng giao tiếp (Cấu trúc Canonical Modular)**:
   - Cấu trúc thư mục tuân thủ mô hình Modular chốt trong [`PROJECT_PLAN.md`](./PROJECT_PLAN.md):
     - 🟧 **Mức Layer (TV1)**: `src/layer/` (Zero-Config hook lá, Deferred CUDA Events, FLOPs, activation memory, overhead, Roofline).
     - 🟦 **Mức Model (TV2)**: `src/model/`, `src/zoo/`, `configs/`, `scripts/` (Trễ P50-P99, FPS, Peak VRAM, Process RSS, background sampler 100ms, 8 model adapters).
     - 🟩 **Mức Giai đoạn (TV3)**: `src/phase/`, `src/analyzer/`, `src/dashboard/` (PhaseTimer, DataLoaderSentinel, 19 luật chẩn đoán v2.0, A/B comparison, Streamlit tabs).
   - Giao tiếp liên module **bắt buộc** qua Public Interfaces (`ModelAdapter`, `ProfilerPlugin`) và Data Schema (`results/` sinh bởi `build_result_path`).
   - Chi tiết: [`02_architecture_and_modularization.md`](./agent-rules/02_architecture_and_modularization.md)

3. **Quy trình Git an toàn & Cách ly phạm vi (Safe Git Workflow)**:
   - **CẤM push trực tiếp lên `main`**. Làm việc trên branch riêng: `<type>/<scope>-<short-description>`.
   - Áp dụng **Zero-Overlap Policy**: Chỉ sửa file thuộc phạm vi module liên quan. Rebase `main` trước khi tạo PR.
   - Chi tiết: [`03_git_workflow_and_collaboration.md`](./agent-rules/03_git_workflow_and_collaboration.md)

4. **Tính chuẩn xác khoa học (Scientific Rigor)**:
   - CẤM đo GPU bằng clock CPU. Bắt buộc dùng **Deferred CUDA Events** (sync 1 lần cuối pass, LIFO stack cho module tái sử dụng).
   - Đo phân phối trễ: **Median (P50), P90, P95, P99, Throughput (FPS), Tail Ratio (P99/P50)**.
   - Warm-up: Inference $N_{warmup} \ge 10-20$ (đo 50); Training $N_{warmup} \ge 5$ (đo 30).
   - Tách biệt Process RSS và Activation Memory trên CPU (chống false positive RA-04).
   - CẤM gọi NVML trong hook của layer (chuyển sang luồng nền `ResourceSampler` 100ms).
   - Sửa 4 heuristics: RB-04, RD-05 (sang ms/GFLOP), RD-02 (Sub-microsecond layer), RC-03 (GPU starvation).
   - Thiết bị: Chuẩn hoá đo trên **Google Colab CPU (`colab_cpu`)** và **Colab T4 (`colab_t4`)** (kèm fallback `cpu_laptop`). Kết nối qua remote ngrok tunnel hoặc sync local. Ngoài phạm vi: Kaggle, GPU cá nhân.
   - Chi tiết: [`04_profiling_and_scientific_rigor.md`](./agent-rules/04_profiling_and_scientific_rigor.md)

5. **Kiểm thử tự động & Tự rà soát**:
   - Luôn viết unit test kèm theo trong `tests/`.
   - BẮT BUỘC chạy `python3 scripts/verify_rules.py` thành công (exit code 0) trước khi báo cáo hoàn thành.

---

## ✅ Checklist Tự Rà Soát Trước Khi Phản Hồi (Agent Self-Checklist)

Trước khi phản hồi kết quả cho người dùng, AI Agent phải tự kiểm tra:
- [ ] File mới hoặc sửa đổi có $\le 300$ dòng không? (Nếu có $\rightarrow$ Phải tách module).
- [ ] Code có type hints 100% cho các public functions/methods không?
- [ ] Có còn lệnh `print()` debug nào trong các module `src/` không? (Đổi sang `logging`).
- [ ] Đo thời gian GPU có đảm bảo dùng Deferred CUDA Events không?
- [ ] Có gán nhầm Process RSS vào layer activation memory không?
- [ ] Thuật toán Speedup A/B có tính đúng theo tổng thời gian toàn mô hình mỗi iteration không?
- [ ] Đã chạy lệnh `python3 scripts/verify_rules.py` trả về exit code 0 chưa?
- [ ] Định dạng phản hồi: Súc tích, minh bạch mã nguồn bằng clickable link `[filename](file:///path/to/file)`.

---

## 🎯 Ma Trận Tối Ưu Ngữ Cảnh (Context Routing Matrix)

Để **tối ưu context window**, AI Agent chỉ cần đọc duy nhất tài liệu liên quan đến tác vụ đang thực hiện:

| Khi bạn nhận nhiệm vụ về... | Hãy đọc duy nhất file này | Nội dung tra cứu chính |
| :--- | :--- | :--- |
| **Viết code mới, refactor class, chia file** | [`01_coding_standards.md`](./agent-rules/01_coding_standards.md) | Giới hạn 300 dòng, Type hints, Logging, Hằng số Enum, Clean code. |
| **Cấu trúc module, thêm Model, sửa Schema** | [`02_architecture_and_modularization.md`](./agent-rules/02_architecture_and_modularization.md) | Interface `ModelAdapter`, `ProfilerPlugin`, schema `results/`. |
| **Tạo branch, commit, rebase, mở PR** | [`03_git_workflow_and_collaboration.md`](./agent-rules/03_git_workflow_and_collaboration.md) | Tên branch, Conventional commits, cách ly file chống conflict, rebase. |
| **Đo GPU/CPU, CUDA Events, Memory, Rules** | [`04_profiling_and_scientific_rigor.md`](./agent-rules/04_profiling_and_scientific_rigor.md) | Deferred CUDA Events, P50-P99, cấm NVML trong hook, sửa 4 heuristics. |

---

## 🛠️ Công Cụ Kiểm Tra Tuân Thủ Tự Động (Linter)

Bất kỳ khi nào viết hoặc sửa code, chạy lệnh kiểm tra bắt buộc:
```bash
python3 scripts/verify_rules.py
```
Nếu lệnh trả về mã lỗi (`exit code 1`), AI Agent **bắt buộc phải sửa lại code** trước khi phản hồi người dùng.
