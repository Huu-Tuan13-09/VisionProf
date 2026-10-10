# Slide phần TV1 — Mức Layer & Độ chính xác

> 6 slide + demo, khoảng 6–7 phút. Mỗi slide gồm **nội dung trên slide**, **hình** và **lời nói gợi ý**.
> Hình nằm trong `docs/tv1/figures/`.

---

## Slide 1 — Mức Layer làm gì?

**Trên slide**
- Câu hỏi: *Mô hình chậm ở **layer nào**, và **vì sao**?*
- Gắn "đồng hồ" vào từng layer của mô hình PyTorch — không sửa mã mô hình
- Đo: thời gian · bộ nhớ output · FLOPs · % hiệu suất so với sức tối đa của máy
- Chạy được trên CPU và GPU với cùng một API

**Lời nói gợi ý**
> Mức Model của TV2 cho biết mô hình chạy bao lâu. Mức Layer mổ xẻ bên trong: trong 30 ms của
> ResNet-50, layer nào tốn thời gian, và layer đó chậm vì nặng thật hay vì chưa dùng hết sức máy.

---

## Slide 2 — Đo đúng trên GPU không dễ

**Trên slide**
- GPU chạy **bất đồng bộ**: CPU chỉ *gửi lệnh* rồi chạy tiếp
  → đồng hồ CPU chỉ đo được lúc gửi lệnh (lỗi của bản v1.0)
- Cách làm: **CUDA event** quanh mỗi layer, **đồng bộ 1 lần** sau cả lần forward
- **Lấp hàng đợi GPU** để GPU chạy liền mạch → đo đúng thời gian GPU tính
- Các lỗi khác đã sửa: module gọi nhiều lần · bộ nhớ layer bị trộn RAM chương trình ·
  attention của ViT bị bỏ sót (thiếu 1/3 FLOPs)

**Lời nói gợi ý**
> Bản cũ đo GPU bằng đồng hồ CPU nên toàn bộ số liệu GPU là sai. Chúng em chuyển sang CUDA event và
> chỉ đồng bộ một lần, để công cụ đo không làm hỏng việc CPU và GPU chạy song song.

---

## Slide 3 — CH1: Thời gian tập trung ở đâu?

**Hình:** `fig1_type_share.png`

**Trên slide**
- CNN: Conv chiếm 73–85% trên CPU
- Transformer: Linear + Attention chiếm 95–96%
- Lên GPU: **MobileNetV3** Conv 73% → 47%, BatchNorm 14% → 25%, kích hoạt 5% → 17%

**Lời nói gợi ý**
> Trên GPU, các phép tính lớn được tăng tốc rất mạnh nên các layer nhỏ như BatchNorm, ReLU lại chiếm
> tỷ trọng lớn hơn — thấy rõ nhất ở mô hình nhẹ MobileNetV3. Với Transformer thì tỷ trọng gần như
> không đổi vì gần như mọi thứ là nhân ma trận.

---

## Slide 4 — CH1: GPU giúp được bao nhiêu, và vì sao?

**Hình:** `fig2_gpu_speedup.png` (trái) + `fig3_roofline.png` (phải, có thể chỉ dùng nửa T4)

**Trên slide**
- Transformer nhanh hơn **×9–9.5**, CNN ×3–6, MLP ×1.0
- ResNet-50: `maxpool` chậm nhất trên CPU (hạng 1) → hạng 59 trên GPU (nhanh hơn ×74)
- Roofline: Conv / Linear / Attention mới đạt **~30–55%** sức tối đa ở batch 1

**Lời nói gợi ý**
> Biểu đồ Roofline cho thấy các layer lớn đều thuộc vùng giới hạn bởi sức tính, nhưng mới dùng khoảng
> một nửa sức máy — với batch 1, phép nhân ma trận còn quá nhỏ để lấp đầy phần cứng.

---

## Slide 5 — CH2: Công cụ đo có đúng không?

**Hình:** `fig4_tn2_error.png`

**Trên slide**

| | CPU laptop | GPU T4 |
| :--- | :-: | :-: |
| Sai lệch so với `torch.profiler` | **0.3%** | **0.8–7.5%** |
| Overhead | 5–11% | 109–166% |

- Đạt mục tiêu sai lệch < 10%
- Layer rất nhỏ trên GPU bị đo dư (ngưỡng sàn CUDA event ~6 µs)
- Phát hiện: `attention_mask` của BERT gây đồng bộ ngầm → sai lệch 87% → sửa còn 0.8%

**Lời nói gợi ý**
> Chúng em kiểm chứng bằng `torch.profiler`. Sai lệch tổng đều dưới 10%; layer lớn rất chính xác,
> layer chỉ chạy vài micro-giây thì bị đo dư — đây là giới hạn của CUDA event mà báo cáo ghi rõ.
> Overhead trên GPU lớn nên mức Layer chỉ dùng để phân rã thời gian, còn tổng thời gian thì lấy từ
> mức Model.

---

## Slide 6 — Hạn chế & hướng phát triển

**Trên slide**
- Layer rất nhỏ trên GPU đo chưa chính xác → chế độ đo bằng CUPTI
- Overhead GPU ở batch 1 lớn → chỉ đo một phần số vòng
- 4–10% thời gian không thuộc layer nào (phép tính dạng hàm)
- Mới đo batch 1, FP32, chiều forward

---

## Demo (1–2 phút)

**Chuẩn bị trước:** mở terminal ở thư mục dự án, kích hoạt `.venv`, mở sẵn tab dashboard.

1. Mở `experiments/demo_layer_profiler.py`, chỉ vào khối **"Phần tích hợp: chỉ cần chừng này dòng"**.
2. Chạy:
   ```bash
   python experiments/demo_layer_profiler.py --model resnet50 --iters 10
   ```
   → in top-10 layer, tỷ trọng theo loại layer, tổng FLOPs (8.18 G) sau vài giây.
3. Chuyển sang dashboard:
   ```bash
   streamlit run src/dashboard_tabs/layer_tab.py
   ```
   → chọn **ViT-B/16**, chỉ vào chỉ số "GPU nhanh hơn ×9.5" và biểu đồ tỷ trọng; rê chuột lên một
   điểm trên Roofline để xem tên layer và % hiệu suất.

**Phương án dự phòng:** nếu máy chạy chậm, chụp sẵn ảnh màn hình kết quả demo và tab dashboard.
