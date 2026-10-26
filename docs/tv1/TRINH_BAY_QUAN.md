# Phần trình bày của Võ Trần Anh Quân · slide 5–14

> **Phần:** Nghiên cứu liên quan và mức Layer (CH1, CH2) · **Màu slide:** 🟧 cam · **10 slide, khoảng 9 phút**
> Đọc phần A trước để nắm cả mạch, rồi mới đọc phần B cho từng slide. Lời thoại đầy đủ nằm trong ghi chú người nói của bộ slide; chữ trên slide đã đủ để nhìn vào mà nói.

---

## A. Hiểu tổng thể trước

### A1. VisionProf là gì, phần của bạn nằm ở đâu

VisionProf là công cụ nhẹ cho PyTorch: gắn vào mô hình có sẵn, **không sửa mã mô hình**, đo được thời gian và bộ nhớ đi vào đâu. Nhóm đo ở 3 mức, mỗi người một mức:

| Mức | Người | Hỏi gì | Câu hỏi | Thí nghiệm |
| :-- | :-- | :-- | :-: | :-: |
| **Layer** | **Quân** | Bên trong mô hình, layer nào tốn thời gian, vì sao | **CH1, CH2** | **TN1, TN2** |
| Model | Trí | Cả mô hình chạy nhanh bao nhiêu, tốn bao nhiêu bộ nhớ | CH3, CH4 | TN3–TN5 |
| Giai đoạn | Tuân | Training và pipeline thật tốn ở bước nào | CH5, CH6 | TN6, TN7 |

- **CH1:** Thời gian tập trung ở loại layer nào? CPU và GPU khác nhau ra sao? → trả lời bằng **TN1**.
- **CH2:** Công cụ đo có chính xác không, làm mô hình chậm đi bao nhiêu? → trả lời bằng **TN2**.
- **Máy đo phần của bạn:** CPU là **laptop x86 của bạn**, GPU là **Tesla T4 trên Colab**. Mọi số đo đều ở **batch 1, FP32**.
- **Mô hình:** 7 mô hình PyTorch (ResNet-50, MobileNetV3-Large, ViT-B/16, YOLOv8n, BERT-base, DistilBERT, MLP). XGBoost không có layer PyTorch nên không đo ở mức Layer.

### A2. Mạch câu chuyện của 10 slide

```
Người đi trước làm gì?          Nhóm làm gì khác?         Mức Layer của nhóm
┌──────────────────────┐       ┌────────────────┐       ┌───────────────────────────────────────────┐
│ 5 MLModelScope       │       │ 8 Bảng so sánh │       │ 9  Đo trên GPU thế nào (kỹ thuật)          │
│   "có 5 tầng"        │──────▶│   khoảng trống │──────▶│ 10 CH1: thời gian dồn vào layer nào        │
│ 6 Roofline           │       │   + hạn chế    │       │ 11 CH1: GPU nhanh hơn CPU bao nhiêu        │
│   "nghẽn ở đâu"      │       └────────────────┘       │ 12 CH1: vì sao chưa nhanh hơn (Roofline)   │
│ 7 XSP                │                                │ 13 CH2: công cụ đo có đúng không           │
│   "đo sâu thì chậm"  │                                │ 14 Dashboard → chuyển cho Trí              │
└──────────────────────┘                                └───────────────────────────────────────────┘
```

Mỗi bài báo ở slide 5–7 đều có một "móc" nối sang phần sau:

| Bài báo | Bài học | Nhóm dùng ở đâu |
| :-- | :-- | :-- |
| MLModelScope | Nhìn một tầng thì đoán sai chỗ chậm | Lý do có 3 mức đo |
| Roofline | Biết layer nghẽn do tính hay do bộ nhớ | **Slide 12** |
| XSP | Đo càng sâu thì công cụ đo càng làm chậm | **Slide 13** (overhead) và cách Trí đo tổng thời gian không gắn hook |

### A3. Từ khoá cần hiểu (giải thích dễ)

| Từ | Nghĩa |
| :-- | :-- |
| **Layer** (leaf layer) | Khối nhỏ nhất trong mô hình PyTorch: Conv2d, BatchNorm, ReLU, Linear, Attention… |
| **Forward hook** | Hàm PyTorch tự gọi ngay trước và sau khi một layer chạy. Nhóm gắn bộ đo vào đây nên không phải sửa mô hình. |
| **Chạy bất đồng bộ** | CPU chỉ "gửi lệnh" cho GPU rồi làm tiếp, không đợi GPU tính xong. Như gửi đơn hàng: gửi xong không có nghĩa là hàng đã giao. |
| **CUDA Event** | Mốc thời gian do **chính GPU** ghi khi chạy tới điểm đó. Hai mốc trừ nhau ra thời gian GPU thật sự tính. |
| **Đồng bộ** (synchronize) | Bắt CPU dừng chờ GPU làm xong. Cần để đọc kết quả, nhưng làm nhiều thì GPU bị ngắt quãng. |
| **Overhead** | Phần chậm thêm do chính công cụ đo gây ra. |
| **FLOPs** | Số phép tính (nhân, cộng số thực) một layer phải làm. |
| **FLOP/byte** | Số phép tính làm được trên mỗi byte đọc từ bộ nhớ. Cao: "tính nhiều, đọc ít". Thấp: "đọc nhiều, tính ít". |
| **Nghẽn bộ nhớ / nghẽn tính toán** | Chậm vì chờ đọc dữ liệu (memory-bound), hoặc chậm vì máy tính hết sức (compute-bound). |
| **% sức máy** | Tốc độ tính layer đạt được ÷ đỉnh sức tính của máy. |
| **Speedup** | Thời gian trên CPU ÷ thời gian trên GPU. Ví dụ ×3.0 là GPU nhanh gấp 3. |
| **Batch 1** | Mỗi lần chạy chỉ đưa 1 ảnh hoặc 1 câu vào. Giống lúc dùng thật, nhưng GPU không được dùng hết sức. |
| **torch.profiler** | Công cụ đo chính thức của PyTorch. Nhóm dùng làm "thước chuẩn" để kiểm tra công cụ của mình. |
| **Kernel GPU, cuDNN** | Kernel là đoạn chương trình chạy thật trên GPU. cuDNN là thư viện của NVIDIA mà Conv, BatchNorm gọi xuống. Nhóm **không đo** tới mức này. |
| **Depthwise conv** | Conv tách theo kênh, mỗi kênh một bộ lọc riêng, nên rất ít phép tính. MobileNetV3 dùng nhiều loại này. |

### A4. Số phải thuộc

| Slide | Số |
| :-: | :-- |
| 6 | Điểm gãy: laptop CPU **~19**, T4 **~25** FLOP/byte (T4: 8.1 TFLOPS ÷ 320 GB/s) |
| 7 | XSP, ResNet-50, batch 256, V100: **275.1 ms**; overhead **157 ms** (đo thêm layer), **215.2 ms** (đo thêm kernel); 65 mô hình |
| 10 | CPU: Conv **73–85%** ở CNN; Linear + Attention **95–96%** ở Transformer. MobileNetV3: Conv **73 → 47%**, chuẩn hoá **14 → 25%**, kích hoạt **5 → 17%** |
| 11 | Transformer **×8.8–9.5**; CNN **×3.0–5.8**; MLP **×1.0** |
| 12 | Layer lớn đạt **30–55%** sức máy; depthwise **~15%** |
| 13 | Sai lệch: CPU **0.35% / 0.27%**, T4 **7.5% / 0.8%** (đo lại 18/10: **7.8% / 0.04%**). Overhead T4 **110–210%**. Ngoài layer **4–10%**. Sàn CUDA Event **~6 µs** |
| 14 | MobileNetV3: CPU **8.48 ms**, T4 **2.81 ms**, **×3.0** |

### A5. Vị trí trong buổi trình bày

- Trí trình bày slide 1–4 rồi mời bạn: *"Trước khi vào thiết kế, bạn Quân sẽ trình bày các nghiên cứu đi trước…"*
- Bạn trình bày slide 5–14, kết thúc bằng câu chuyển cho Trí.
- Thứ tự cả nhóm: Trí (1–4) → **Quân (5–14)** → Trí (15–21) → Tuân (22–31). Slide 32 là phụ lục tài liệu tham khảo.

---

## B. Từng slide

### Slide 5 · MLModelScope: chạy một mô hình đi qua 5 tầng · ~60 giây

**Slide nói gì:** khi chạy một mô hình, thời gian đi qua 5 tầng, từ mô hình xuống tới phần cứng.

**Đọc hình bên trái (Hình 1 của bài báo):**

| Tầng | Trong hình có | Nói dễ hiểu |
| :-- | :-- | :-- |
| L1 Model | Pre-process → Model Inference → Post-process | Cả mô hình, gồm bước chuẩn bị đầu vào và xử lý đầu ra |
| L2 Framework | Conv, BN, ReLU, Concat, Softmax | Từng layer trong PyTorch, TensorFlow… |
| L3 Layer | Malloc, CUDNN, Transpose, Free | Các lời gọi thư viện **bên trong** một layer |
| L4 Library | Malloc, Kernel, Free | Các kernel chạy thật trên GPU |
| L5 Hardware | L$ Access, DRAM Access, SP Flops | Phần cứng: đọc cache, đọc bộ nhớ, phép tính |

**Hiểu sâu:** "chậm" có thể do bất kỳ tầng nào. Đứng ở L1 chỉ biết mô hình mất 10 ms, không biết layer nào tốn. Đứng ở L4 thấy hàng nghìn kernel, nhưng không biết kernel nào thuộc layer nào. Vì vậy cần đo nhiều tầng và ghép lại.

> ⚠️ **Chỗ cần thống nhất trên slide.** Ô "VisionProf áp dụng" trên link hiện ghi **"L2-L3 → mức Layer · L4–L5 chưa đo"**, còn ghi chú người nói ghi "L2 → mức Layer, L3–L5 chưa đo".
> Theo đúng hình gốc, bộ đo của nhóm gắn hook vào từng layer PyTorch (Conv, BN, ReLU), tức là **L2**. **L3** là các lời gọi cuDNN, Malloc **bên trong** layer, nhóm không đo.
> Tên "L3 Layer" dễ làm tưởng là mức Layer của nhóm, nhưng nội dung của L3 không phải thứ nhóm đo. Nên sửa slide về **"L2 → mức Layer · L3–L5 chưa đo"**.
> Nếu vẫn giữ "L2-L3" mà thầy hỏi "nhóm có đo cuDNN không?", câu trả lời thật là **không**.

### Slide 6 · Roofline: nghẽn do tính toán hay do bộ nhớ? · ~60 giây

**Slide nói gì:** một hình duy nhất cho biết một phép tính bị chặn bởi sức tính của máy hay bởi tốc độ đọc bộ nhớ.

**Đọc hình:**
- Trục ngang là FLOP/byte, trục dọc là tốc độ tính đạt được (GFLOP/s).
- Đường "mái nhà" có 2 đoạn:
  - **mái dốc** bên trái: bộ nhớ không cấp dữ liệu kịp;
  - **mái bằng** bên phải: máy đã tính hết sức (Opteron X2 là 16 GFLOP/s).
- **Điểm gãy** là chỗ hai mái gặp nhau.

**Hiểu sâu, có ví dụ:**
- Công thức: **tốc độ đạt được = min(đỉnh sức tính, băng thông × FLOP/byte)**.
- Với T4: đỉnh 8.1 TFLOPS, băng thông 320 GB/s. Điểm gãy = 8 100 ÷ 320 ≈ **25** FLOP/byte.
  - Một layer chỉ có 10 FLOP/byte thì nhiều nhất đạt 320 × 10 = 3.2 TFLOPS, chưa tới một nửa sức T4. Layer đó **nghẽn bộ nhớ**, mua GPU tính mạnh hơn cũng không giúp.
  - Một layer có 100 FLOP/byte thì chạm mái bằng, tức **nghẽn tính toán**.
- Hình bên phải của bài báo so Opteron X4 với X2. X4 có mái bằng cao hơn, nhưng ở vùng ít FLOP/byte hai đường gần trùng nhau, nên phép tính đang chờ bộ nhớ gần như không nhanh hơn. Ý này **không còn ghi trên slide** nhưng có trong ghi chú, bạn có thể nói miệng.

**Nối sang:** slide 12 đặt từng layer của nhóm lên Roofline này.

### Slide 7 · XSP: đo càng sâu, công cụ đo càng làm chậm · ~60 giây

**Slide nói gì:** đo càng chi tiết thì chính công cụ đo càng chen thêm thời gian, nên không được lấy tổng thời gian từ lượt đo chi tiết.

**Đọc hình** (ResNet-50, batch 256, GPU V100):

| Dòng | Đo gì | Kết quả |
| :-- | :-- | :-- |
| M | Chỉ đo cả mô hình | Suy luận 275.1 ms |
| M/L | Đo thêm từng layer | Công cụ chen thêm 157 ms (ô đen "Profiling Overhead") |
| M/L/G | Đo thêm kernel GPU | Phần chen thêm tăng lên 215.2 ms |

Ô **xanh** là thời gian thật của mô hình, ô **đen** là thời gian do công cụ đo sinh ra.

**Hiểu sâu:**
- Mỗi lần gắn đo vào một layer là thêm một lần gọi hàm và ghi mốc thời gian. Mô hình có hàng trăm layer thì phần thêm này cộng dồn lại.
- Cách của XSP: **mỗi mức lấy số ở lượt chạy của đúng mức đó**, rồi ghép lại. Tổng thời gian lấy từ lượt M; tỷ lệ giữa các layer lấy từ lượt M/L.

**Nhóm áp dụng y hệt:**
- Trí đo tổng thời gian ở lượt **không gắn hook**.
- Mức Layer của bạn chỉ dùng để **chia tỷ lệ** giữa các layer.
- Overhead của nhóm được tự đo ở **TN2, slide 13**.

### Slide 8 · So sánh và khoảng trống nhóm lấp · ~60 giây

**Slide nói gì:** mỗi công trình trước mạnh ở một mức. VisionProf gộp 3 mức vào một công cụ nhẹ, đổi lại không đo sâu tới kernel.

**Đọc bảng:** "Có" là trọng tâm của bài báo đó. "–" là **không phải trọng tâm**, không có nghĩa là bài hoàn toàn không nhắc tới. Câu này có ghi ở chân slide, nên nói rõ khi trình bày.

| Công trình | Mạnh ở |
| :-- | :-- |
| XSP, PRoof | Thời gian từng layer, kernel GPU, Roofline |
| MLPerf | Chuẩn cách đo độ trễ mức Model theo phân vị P50–P99 |
| MLModelScope | Đo lặp lại được trên nhiều framework, nhiều phần cứng |
| AI Tax | Trong ứng dụng thật, phần AI chỉ chiếm dưới 60% độ trễ, còn lại là các bước xung quanh |
| Data Stalls | Training thường phải chờ nạp dữ liệu |

- **Khoảng trống:** chưa có công cụ nhẹ nào cho PyTorch gộp cả Layer, Model, Giai đoạn, chạy được trên laptop và Colab T4, lại có thí nghiệm tự kiểm chứng (TN2).
- **Hạn chế, nói trước để thầy không bắt lỗi:** không đo kernel GPU, không dùng hardware counter (bộ đếm phần cứng như số lần đọc cache).

**Câu chuyển:** *"Bắt đầu từ mức sâu nhất, mức Layer."*

### Slide 9 · Đo thời gian trên GPU · ~75 giây

**Slide nói gì:** đo thời gian trên GPU khó vì CPU không chờ GPU. Nhóm giải bằng 4 kỹ thuật.

**Đọc sơ đồ bên trái:** dòng CPU có 3 ô "gửi" ngắn, dòng GPU có 3 ô Layer dài, chạy sau đó. Đồng hồ CPU chỉ đo được mấy ô "gửi", gần như bằng 0.

**Hai cách sai:**
1. Dùng đồng hồ CPU: chỉ đo thời gian gửi lệnh, sai hoàn toàn.
2. Đồng bộ sau mỗi layer: số từng layer đúng, nhưng GPU phải dừng chờ sau mỗi layer nên tổng thời gian bị đội lên.

**4 kỹ thuật của nhóm:**

| # | Kỹ thuật | Để làm gì |
| :-: | :-- | :-- |
| 1 | CUDA Event trong hook trước và sau mỗi layer | GPU tự ghi mốc khi chạy tới, nên đo đúng thời gian GPU tính |
| 2 | Đồng bộ **một lần** ở cuối vòng | Không chặn GPU giữa các layer, chạy xong mới đọc hết các mốc |
| 3 | Lấp hàng đợi GPU trước khi đo | Hàng đợi luôn có việc, GPU chạy liền mạch, không ngồi chờ CPU gửi lệnh |
| 4 | Ngăn xếp | Một layer được gọi nhiều lần trong một vòng thì mỗi lần vào, ra được ghép đúng cặp |

**Khung code:** chỉ khoảng 5 dòng: `attach` → `start` / `stop` quanh mỗi vòng → `save`. Ý cần nói: **không sửa mã mô hình**.

### Slide 10 · CH1: thời gian tập trung ở loại layer nào? · ~75 giây

**Slide nói gì:** trên CPU thời gian dồn vào phép tính lớn. Lên GPU, layer nhỏ chiếm phần lớn hơn, rõ nhất ở MobileNetV3.

**Đọc hình:** mỗi mô hình có 2 thanh ngang, thanh CPU và thanh T4. Mỗi màu là một loại layer, độ dài là % thời gian.

**Bảng trên slide:**

| Mô hình · loại layer | CPU | T4 |
| :-- | :-: | :-: |
| ResNet-50 · Conv | 85% | 83% |
| MobileNetV3 · Conv | 73% | **47%** |
| MobileNetV3 · Chuẩn hoá | 14% | **25%** |
| MobileNetV3 · Kích hoạt | 5% | **17%** |
| BERT-base · Linear | 95% | 95% |
| ViT-B/16 · Linear + Attention | 96% | 94% |

**Vì sao lên GPU thì layer nhỏ chiếm nhiều hơn?**
- GPU tăng tốc rất mạnh phép tính lớn như Conv lớn, nhân ma trận.
- Layer nhỏ như BatchNorm, ReLU có rất ít việc. Thời gian của chúng chủ yếu là **chi phí khởi chạy** mỗi lần gọi GPU, nên gần như không nhanh hơn.
- Phần lớn co lại mạnh, phần nhỏ đứng yên, nên tỷ trọng của phần nhỏ tăng lên.
- MobileNetV3 vốn nhiều layer nhỏ nên thấy rõ nhất. ResNet-50 và Transformer có layer lớn chiếm áp đảo nên tỷ trọng gần như không đổi.

**Chân slide:** CPU là laptop x86, batch 1, FP32.

### Slide 11 · CH1: GPU nhanh hơn CPU bao nhiêu lần? · ~40 giây

**Slide nói gì:** mô hình càng nhiều phép nhân ma trận lớn thì GPU càng giúp nhiều.

**Cách tính:** tổng thời gian các layer trên CPU laptop ÷ trên T4.

| Nhóm | Speedup | Lý do |
| :-- | :-- | :-- |
| Transformer: ViT 9.5, BERT 9.0, DistilBERT 8.8 | **×8.8–9.5** | 94–96% thời gian là nhân ma trận lớn, đúng việc GPU làm giỏi nhất |
| CNN: ResNet-50 5.8, YOLOv8n 5.5, MobileNetV3 3.0 | **×3.0–5.8** | Có nhiều layer nhỏ; MobileNetV3 nhiều nhất nên thấp nhất |
| MLP | **×1.0** | Mô hình quá nhỏ, chi phí gọi GPU lấn hết lợi ích |

**Nối với slide 10:** slide 10 nói layer nhỏ không nhanh lên trên GPU. Mô hình nào nhiều layer nhỏ thì speedup thấp.

### Slide 12 · CH1: vì sao layer lớn chưa nhanh hơn nữa? · ~35 giây

**Slide nói gì:** đây là Roofline của slide 6 áp lên layer thật của nhóm. Layer lớn chỉ dùng được 30–55% sức máy.

**Đọc hình:** bên trái CPU, bên phải T4. Mỗi chấm là một layer Conv, Linear hoặc Attention có FLOPs đếm được. Đường mái là giới hạn của máy. Chấm càng xa mái thì càng phí sức máy.

**Ba ý trên slide:**
1. **Chỉ 30–55% sức máy:** phần lớn layer nằm bên phải điểm gãy, tức về lý thuyết là nghẽn tính toán, nhưng vẫn dưới mái xa. Lý do là batch 1, ma trận còn nhỏ, GPU chưa có đủ việc để chạy hết các nhân.
2. **Depthwise conv ~15%:** rất ít phép tính trên mỗi byte, nằm bên trái điểm gãy, tức nghẽn bộ nhớ.
3. **Ít FLOPs không có nghĩa là nhanh:** khớp với nguyên tắc **G4** trong bài ShuffleNet V2 (Ma và cộng sự, ECCV 2018). G4 nói các phép toán theo từng phần tử như ReLU, cộng tensor có FLOPs rất nhỏ nhưng vẫn tốn thời gian thật, vì phải đọc ghi bộ nhớ nhiều. Nên chỉ đếm FLOPs thì không đoán được tốc độ.

### Slide 13 · CH2: công cụ đo có đúng không? · ~75 giây

**Slide nói gì:** so với torch.profiler, công cụ của nhóm đủ chính xác để chia tỷ lệ giữa các layer. Nhóm cũng nói rõ 3 giới hạn.

**Bảng trên slide** (TN2, batch 1, FP32):

| So với torch.profiler | ResNet-50 CPU | BERT CPU | ResNet-50 T4 | BERT T4 |
| :-- | :-: | :-: | :-: | :-: |
| Sai lệch tổng thời gian | 0.35% | 0.27% | 7.5% | 0.8% |
| Đo lại bằng code hiện tại (18/10) | – | – | 7.8% | 0.04% |
| Overhead khi gắn hook | 11% | 5% | 166% | 109% |
| Thời gian không thuộc layer nào | 3.9% | 9.2% | 9.0% | 10.3% |

**Cách hiểu từng dòng:**
- **Sai lệch tổng thời gian:** tổng thời gian các layer theo VisionProf so với theo torch.profiler. CPU dưới 1%, T4 dưới 8%, đo lại vẫn ra gần như vậy. **Kết luận: đủ chính xác để phân rã tỷ lệ.**
- **Overhead:** gắn hook làm mô hình chậm thêm bao nhiêu. Trên CPU nhỏ (5–11%). Trên T4 ở batch 1 rất lớn, khoảng 110–210% tuỳ phiên Colab.
  - Lý do: mỗi layer trên GPU chỉ chạy vài chục micro giây, nên chi phí gọi hook và ghi event tính ra phần trăm thành lớn.
  - Vì vậy nhóm **không lấy tổng thời gian từ lượt gắn hook**. Đúng bài học XSP ở slide 7.
- **Không thuộc layer nào (4–10%):** phép cộng shortcut `x + y`, `torch.cat`… là hàm, không phải layer nên không có hook. Nhóm đo và báo con số này, không giấu đi.
- **Sàn ~6 µs:** CUDA Event không đo chính xác việc ngắn hơn khoảng 6 micro giây, nên ReLU, BatchNorm ở batch 1 bị đo dư. Chỉ dùng số này để so sánh tương đối.

**Câu chốt:** *"Mức Layer đủ chính xác để phân rã, và biết rõ giới hạn của mình."*

### Slide 14 · Dashboard, tab Layer · ~15 giây

**Slide nói gì:** kết quả CH1 xem trực tiếp được trên dashboard của nhóm.

**3 điểm đánh số trên ảnh:**
1. Chọn mô hình MobileNetV3: tổng thời gian CPU **8.48 ms**, T4 **2.81 ms**, nhanh hơn **×3.0**.
2. Thanh trên là CPU, dưới là T4: Conv **73% → 47%**, đúng kết quả slide 10.
3. Kéo xuống có top-10 layer chậm nhất và Roofline.

**Nếu thầy muốn xem thật:** chạy `streamlit run src/dashboard/app.py` → mở tab **"3. Leaf Layer Breakdown"**. Chi tiết trong [`../HUONG_DAN_CHAY_DASHBOARD.md`](../HUONG_DAN_CHAY_DASHBOARD.md). Lần đầu chạy Streamlit có thể hỏi email, bấm Enter để bỏ qua.

**Câu chuyển cho Trí:** *"Đo tổng thời gian sạch của cả mô hình là phần của bạn Trí."*

---

## C. Câu hỏi có thể gặp

**H: Overhead trên GPU tới 110–210%, sao vẫn tin số đo?**
Đ:
- Overhead làm cả mô hình chạy chậm hơn, nhưng **sai lệch tổng thời gian các layer** so với torch.profiler vẫn dưới 8%. Đo lại ngày 18/10 vẫn ra như vậy.
- Mức Layer chỉ dùng để chia tỷ lệ. Tổng thời gian sạch lấy ở mức Model, đo không gắn hook.

**H: Phép cộng shortcut, `torch.cat` đi đâu?**
Đ: Chúng không phải module nên không có hook. Nhóm đo thành chỉ số "thời gian không thuộc layer nào": 3.9–9.2% trên CPU, 9–10% trên T4. Cuối kỳ sẽ bắt bằng `TorchDispatchMode`.

**H: Sao không dùng luôn torch.profiler?**
Đ: torch.profiler là thước chuẩn để kiểm chứng (TN2). Nhóm cần thêm FLOPs, Roofline, tổng hợp theo layer, và gắn được vào training, pipeline với file kết quả gọn, nên viết bộ đo riêng nhẹ hơn.

**H: Chỉ đồng bộ một lần cuối vòng thì làm sao biết thời gian từng layer?**
Đ: CUDA Event do GPU ghi khi chạy tới, nên mốc thời gian đã nằm sẵn trên GPU. Đồng bộ cuối vòng chỉ để chờ GPU xong rồi đọc khoảng cách giữa từng cặp mốc.

**H: Layer chỉ chạy vài micro giây có đo đúng không?**
Đ: Không tuyệt đối. Sàn CUDA Event khoảng 6 µs nên ReLU, BatchNorm ở batch 1 bị đo dư. Nhóm chỉ dùng số này để so sánh tương đối và ghi rõ là hạn chế.

**H: Đỉnh Roofline của máy lấy từ đâu?**
Đ:
- T4 theo thông số của NVIDIA: 8.1 TFLOPS FP32, 320 GB/s, điểm gãy ~25.
- CPU ước lượng theo số nhân × xung nhịp × 32 phép tính mỗi chu kỳ, băng thông 51.2 GB/s, điểm gãy ~19.

**H: Số CPU ở mức Layer đo trên máy nào?**
Đ: Laptop x86 của em (TN1, TN2). Nhóm không so số này với CPU ở TN3, TN4 vì đó là MacBook chip Apple.

**H: Nhóm đo ở tầng nào trong 5 tầng MLModelScope?**
Đ:
- L1 (cả mô hình, tiền và hậu xử lý) ứng với mức Model và mức Giai đoạn.
- L2 (từng layer PyTorch) ứng với mức Layer.
- Lời gọi cuDNN, kernel GPU và bộ đếm phần cứng thì chưa đo, đây là hạn chế.

**H: Sao đo ở batch 1?**
Đ: Batch 1 giống lúc dùng thật, ví dụ xử lý từng ảnh một, và là trường hợp khó nhất cho GPU. Ảnh hưởng của batch lớn hơn là phần CH4 của Trí.

## D. Trước giờ trình bày

- [ ] Đọc phần A một lượt, nói lại được mạch 5 → 14 không cần nhìn.
- [ ] Thuộc bảng A4.
- [ ] Tập to một lượt với slide, bấm giờ khoảng 9 phút.
- [ ] Mở sẵn dashboard ở tab "3. Leaf Layer Breakdown".
- [ ] Tập câu nhận lời từ Trí (cuối slide 4) và câu chuyển cho Trí (cuối slide 14).
