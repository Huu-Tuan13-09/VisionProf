# Chạy dashboard khi trình bày

Dashboard đọc kết quả đã đo trong `results/`, không chạy mô hình hay đo lại GPU.
Số thứ tự 1–8 trên thanh bên được giữ cố định để khớp với slide.

## 1. Chuẩn bị môi trường

Nếu chưa có môi trường, cài một lần:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install streamlit pandas plotly scipy
```

Nếu môi trường hiện tại đã có các thư viện này thì bỏ qua bước trên.

## 2. Mở dashboard

Chạy từ thư mục gốc VisionProf:

```bash
.venv/bin/streamlit run src/dashboard/app.py --server.headless true --server.address 127.0.0.1
```

Mở **http://127.0.0.1:8501** bằng trình duyệt. Trên WSL, dùng trình duyệt Windows như bình thường.
Nếu cổng bận, thêm `--server.port 8502` rồi mở cổng tương ứng.

**Lần đầu Streamlit hỏi email:** bấm **Enter** để bỏ qua, hoặc chạy với `--server.headless true` như trên.
Dừng dashboard bằng **Ctrl+C** trong terminal đang chạy.

Theme mặc định là nền sáng. Màu cam `#B4480F` nhận diện mức layer, xanh dương `#2456A6` mức model,
xanh lá `#1E7A4C` mức giai đoạn. Màu thiết bị và giai đoạn được dùng nhất quán giữa các biểu đồ.

## 3. Tám mục trên thanh bên

| Mục | Nội dung và thao tác chính |
| :--- | :--- |
| 1. Tổng quan | Lọc thiết bị, nhóm mô hình, thực nghiệm; xem P50, FPS và đánh đổi tham số / độ trễ. |
| 2. CPU và T4 | Ba tab con: **CPU và T4**, **Batch size**, **FP16**. Bảng FP16 hiện trực tiếp, có P50 FP32/FP16, tăng tốc, VRAM và cosine. |
| 3. Phân tích layer | Chọn mô hình và thiết bị; xem tổng thời gian CPU, T4, tăng tốc, FLOPs, tỷ trọng loại layer, top-10, thứ tự thực thi và Roofline. |
| 4. Roofline | Chọn lượt đo layer; xem GFLOPs, thời gian và hiệu suất so với đỉnh phần cứng. |
| 5. Training TN6 | Thời gian giai đoạn, VRAM train / infer, **DataLoader & Workers**, phân phối độ trễ. |
| 6. Pipeline TN7 | Tỷ trọng (%), **Overhead % Bar**, thời gian (ms), CPU và T4. |
| 7. Chẩn đoán | Chọn lượt đo, chọn luật; **RB-04** được mở mặc định. Xem trạng thái, kết quả và đề xuất. |
| 8. So sánh A/B | Chọn A và B; xem speedup, Cliff's Delta, p-value và phân phối độ trễ. |

Bảng trùng biểu đồ nằm trong expander **Xem bảng số**. Bảng FP16, top-10 layer và DataLoader vẫn hiện trực tiếp.
Các ô chọn có tìm kiếm: bấm rồi gõ một phần tên lượt đo, ví dụ `mobilenet_v3_large/TN3`.

## 4. Mở đúng phần trên slide

| Slide | Mục | Thao tác |
| :-: | :--- | :--- |
| 14 | 3. Phân tích layer | Chọn MobileNetV3-Large; CPU, T4 và tăng tốc nằm ngay dưới bộ chọn. |
| 21 | 2. CPU và T4 | Mở tab con **FP16**. |
| 24 | 5. Training TN6 | Mở tab con **DataLoader & Workers**. |
| 26 | 6. Pipeline TN7 | Mở tab con **Overhead % Bar**. |
| 28 | 7. Chẩn đoán | Chọn `colab_t4/mobilenet_v3_large/TN3/bs1_fp32`, chọn luật **RB-04**. |
| 29 | 8. So sánh A/B | A = `colab_t4/vit_b_16/TN3/bs1_fp32`, B = `colab_t4/vit_b_16/TN5/bs1_fp16`. |

## 5. Kiểm tra trước buổi trình bày

- Mở đủ 8 mục, kiểm tra dữ liệu và các phần dùng trên slide.
- Kiểm tra lại A và B đã chọn đúng hai lượt đo; speedup dùng kết quả từ engine A/B hiện có.
- Để sẵn dashboard bên cạnh slide trong trình duyệt.
- Dashboard chỉ mở tại `127.0.0.1` trên máy trình bày.

## 6. Điều hướng và cách đọc giao diện

Thanh bên dùng điều hướng gốc `st.navigation` / `st.Page`, có icon và chia thành
**Mức Model**, **Mức Layer**, **Mức Giai đoạn**, **Chẩn đoán**. Bấm tên mục để chuyển trang;
mục đang mở có nền nổi bật và vạch màu. Số thứ tự 1–8 vẫn khớp với slide.
Mỗi trang có đường dẫn riêng (ví dụ `/layers`, `/roofline`, `/diagnostics`), có thể đánh dấu
trang trên trình duyệt. Giữ nguyên bộ lọc khi chuyển trang không được bảo đảm;
kiểm tra lại lựa chọn trước khi trình bày.

- Biểu đồ **Tham số và P50** dùng điểm 14 px, kèm nhãn mô hình; màu và ký hiệu
  phân biệt thiết bị. FPS nằm trong tooltip. Kích thước điểm không còn biểu diễn FPS.
  Hai trục vẫn dùng thang log và số liệu giữ nguyên.
- **Roofline** mặc định mở `colab_t4/resnet50/TN1/bs1_fp32` nếu có.
  Đồ thị dùng chung cách vẽ với mục 3, đọc `layers_summary.csv` và `layer_meta.json`.
  Thẻ số vẫn dùng trung bình mỗi pass từ `layers.csv` như trước.
- **Chẩn đoán** mặc định chọn lượt đo MobileNetV3-Large TN3 trên T4;
  **So sánh A/B** mặc định chọn ViT-B/16 FP32 TN3 và FP16 TN5 nếu có dữ liệu.
- Thẻ số và biểu đồ dùng màu nhận diện của mức đo. Plotly thống nhất font,
  lưới nhạt và legend; nút tải ảnh hiện khi di chuột lên biểu đồ.
- `.streamlit/config.toml` đặt `client.toolbarMode = "minimal"` để thu gọn toolbar
  và ẩn nút Deploy. Khởi động lại Streamlit nếu vừa đổi cấu hình/theme.
