# Chạy dashboard khi trình bày

Slide 14, 21, 24, 26, 28, 29 là ảnh chụp dashboard. Khi thầy muốn xem trực tiếp, mở dashboard trên máy đang trình chiếu.
Dashboard chỉ đọc số liệu có sẵn trong `results/`, không chạy mô hình, nên không cần GPU hay torch.

## 1. Cài một lần

```bash
git pull
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install streamlit pandas plotly scipy
```

Nếu đã cài đủ `requirements.txt` thì bỏ qua bước này.

## 2. Chạy trước buổi trình bày

```bash
streamlit run src/dashboard/app.py
```

Mở **http://localhost:8501** trong trình duyệt, để sẵn ở một tab. Cổng đang bận thì thêm `--server.port 8502`.
Trên WSL, mở link này bằng trình duyệt của Windows như bình thường.

## 3. Tab nào ứng với slide nào

| Slide | Tab trên thanh bên trái | Thao tác để giống ảnh trên slide |
| :-: | :--- | :--- |
| 14 | 3. Leaf Layer Breakdown | Mô hình mặc định MobileNetV3-Large |
| 21 | 2. Colab CPU vs T4 GPU | Kéo xuống mục 3 (FP16) |
| 24 | 5. TN6 — Training Phases | Tab con "DataLoader & Workers" |
| 26 | 6. TN7 — Pipeline Breakdown | Tab con "Overhead % Bar" |
| 28 | 7. Diagnostic Engine | Chọn `colab_t4/mobilenet_v3_large/TN3/bs1_fp32`, mở luật RB-04 |
| 29 | 8. A/B Testing & Speedup | A = `colab_t4/vit_b_16/TN3/bs1_fp32`, B = `colab_t4/vit_b_16/TN5/bs1_fp16` |

Ô chọn có tìm kiếm: bấm vào rồi gõ một phần tên, ví dụ `mobilenet_v3_large/TN3`.

## 4. Muốn thầy mở trên máy khác

- **Cùng Wi-Fi:** terminal in ra dòng `Network URL: http://<ip>:8501`, gửi link đó cho thầy.
- **Khác mạng:** cần một đường hầm công khai, ví dụ `ngrok http 8501` (notebook Colab của nhóm đã có sẵn ô ngrok).

## 5. Kiểm tra trước giờ trình bày

- [ ] `git pull` để có số liệu mới nhất trong `results/`.
- [ ] Mở thử đủ 6 tab ở bảng trên, không tab nào báo lỗi.
- [ ] Để sẵn dashboard ở một tab trình duyệt, bên cạnh tab slide.
