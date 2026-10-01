# 03. Quy Trình Git & Phối Hợp Kỹ Thuật (Git Workflow & Collaboration)

Tài liệu này định nghĩa quy trình Git an toàn tuyệt đối, triệt tiêu nguy cơ xung đột mã nguồn (Merge Conflicts) và bảo vệ tính toàn vẹn của nhánh chính (`main`).

---

## 1. Mô Hình Phân Nhánh (Branching Model)

Dự án áp dụng mô hình **Trunk-Based Development kết hợp Short-lived Feature Branches**.

### 1.1. Nguyên Tắc Cốt Lõi
1. **Khóa nhánh `main` đối với Direct Push**:
   - Tuyệt đối không commit hoặc push trực tiếp lên `main`.
   - Mọi thay đổi đều phải thực hiện trên một feature branch riêng biệt và tích hợp qua Pull Request (PR).
2. **Tuổi thọ nhánh ngắn (Short-lived)**:
   - Một nhánh tính năng chỉ tồn tại trong thời gian ngắn (tối đa 1–2 ngày) cho một phạm vi hẹp. Hoàn thành là merge ngay để tránh phân kỳ mã nguồn (code drift).

### 1.2. Quy Ước Đặt Tên Nhánh (Branch Naming)
Cú pháp chuẩn: `<type>/<scope>-<short-description>`
- `feature/layer-cuda-events`
- `feature/zoo-model-adapters`
- `feature/phase-timer`
- `fix/collector-overhead-count`
- `fix/analyzer-speedup-calc`
- `refactor/split-dashboard-views`
- `docs/update-spec-contracts`

---

## 2. Quy Chuẩn Commit (Conventional Commits)

Mọi commit phải tuân thủ chuẩn Conventional Commits:
```
<type>(<scope>): <mô tả ngắn gọn>

[Tùy chọn: mô tả chi tiết lý do và thay đổi]
```

- **`<type>` hợp lệ**:
  - `feat`: Tính năng mới
  - `fix`: Sửa lỗi
  - `refactor`: Tái cấu trúc mã nguồn không đổi hành vi
  - `perf`: Cải thiện hiệu năng profiling
  - `test`: Thêm hoặc cập nhật unit test
  - `docs`: Cập nhật tài liệu kỹ thuật, quy tắc
  - `chore`: Bảo trì build, cấu hình, `.gitignore`

---

## 3. Chiến Lược Triệt Tiêu Xung Đột (Conflict Prevention)

### 3.1. Cách Ly Phạm Vi Thay Đổi (Zero-Overlap File Policy)
Xung đột chỉ xảy ra khi hai thay đổi đồng thời sửa cùng một dòng mã trong cùng một file. Do đó:
- Tuân thủ nghiêm ngặt **Ranh giới module** tại [`02_architecture_and_modularization.md`](./02_architecture_and_modularization.md).
- Mỗi nhánh tính năng chỉ được phép tác động lên các file thuộc phạm vi module liên quan.
- Khi cần thay đổi Interface chung (`ModelAdapter`, `ProfilerPlugin`, schema `results/`), thay đổi đó phải được merge độc lập trước khi các nhánh phụ thuộc tiếp tục phát triển.

### 3.2. Quy Trình Đồng Bộ Thường Xuyên (Frequent Rebase)
Trước khi bắt đầu code và trước khi tạo PR, luôn đồng bộ nhánh tính năng với `main` mới nhất:

```bash
git checkout main
git pull origin main
git checkout feature/my-feature
git rebase main
# (Nếu có conflict: giải quyết xong -> git rebase --continue)
```

### 3.3. Bảo Vệ Dữ Liệu: Tuyệt Đối Không Commit File Rác & Checkpoints
Bắt buộc có trong `.gitignore`:
- Môi trường ảo: `.venv/`, `venv/`, `env/`
- Bytecode Python: `__pycache__/`, `*.pyc`
- Trọng số mô hình tải về: `*.pt`, `*.pth`, `*.onnx`, `*.bin`, `weights/`
- Dữ liệu tải về tự động: `data/coco/`, `data/cifar/`, `data/covtype/`
- Kết quả tạm thời: `results/temp/`, `*.log`
*(Chỉ commit file cấu hình chuẩn trong `configs/` và kết quả thực nghiệm chính thức)*

---

## 4. Tiêu Chuẩn Review & Tích Hợp (Merge Standards)

Trước khi merge bất kỳ thay đổi nào vào `main`:

1. **Kiểm tra tự động**:
   - Chạy `python3 scripts/verify_rules.py` $\rightarrow$ Kết quả exit code 0.
   - Chạy unit tests liên quan trong thư mục `tests/`.
2. **Review Checklist**:
   - [ ] Mã nguồn có tuân thủ giới hạn $\le 300$ dòng/file không?
   - [ ] Có bare `print()` trong module đo lường/phân tích không?
   - [ ] Có vi phạm các nguyên tắc profiling khoa học (CUDA Events, warmup, RSS vs Activation) không?
   - [ ] Có phá vỡ interface contracts hoặc kết quả schema không?
3. **Chiến lược Merge**: Dùng **Squash and Merge** hoặc **Rebase and Merge** để giữ lịch sử commit trên nhánh `main` tuyến tính và sạch sẽ.
