# BÁO CÁO RÀ SOÁT MÃ NGUỒN, PHÂN TÍCH PHẢN BIỆN CHUYÊN SÂU & PHÂN ĐỊNH LỘ TRÌNH TRIỂN KHAI DỰ ÁN VISIONPROF

> **Tài liệu quy chuẩn hành động (Actionable Roadmap & Defense Strategy)**  
> **Dự án**: [VisionProf](../README.md) — Cross-Platform AI Profiling & Automated Heuristic Diagnostic Framework  
> **Cơ sở đối chiếu**: [`NGHIEN_CUU_KHOA_HOC_VA_CO_SO_PHAT_TRIEN.md`](NGHIEN_CUU_KHOA_HOC_VA_CO_SO_PHAT_TRIEN.md), [`NGHIEN_CUU_SO_SANH_PAPER_VA_VISIONPROF.md`](NGHIEN_CUU_SO_SANH_PAPER_VA_VISIONPROF.md), [`PROJECT_PLAN.md`](../PROJECT_PLAN.md), [`DAN_Y_SLIDE.md`](DAN_Y_SLIDE.md) và toàn bộ mã nguồn thực tế tại [`src/`](../src).  
> **Mục tiêu cốt lõi**: **Rà soát thực tế mã nguồn**, **phân tích phản biện kỹ thuật sâu sắc**, phân định rạch ròi giữa **những việc BẮT BUỘC hoàn thiện để báo cáo bảo vệ** và **những bài toán ĐẶT VÀO PHƯƠNG HƯỚNG PHÁT TRIỂN TIẾP THEO (Future Work)**.

> **Cập nhật 18/10/2026 (sau lượt đo Colab T4):** Vấn đề 1 đã xong: TN3, TN4, TN5 đo thật trên T4, đã có số
> (`KET_QUA_COLAB_GIUA_KY.md`); TN6 có thêm phần nạp dữ liệu với DataLoader thật (`scripts/run_tn6_dataloader.py`).
> Nhóm quyết định **giữa kỳ chỉ làm số đo + slide**; các việc P1 (gợi ý code trong chẩn đoán, ablation) chuyển sang **cuối kỳ**.
> Các câu trả lời mẫu ở Mục 5 đã sửa theo số đo thật.

---

## 📌 MỤC LỤC
1. [Tổng Quan Kiểm Toán Mã Nguồn & Rà Soát Thực Tế Codebase](#1-tổng-quan-kiểm-toán-mã-nguồn--rà-soát-thực-tế-codebase)
2. [Phân Tích & Phản Biện Chuyên Sâu Từng Vấn Đề Kỹ Thuật](#2-phân-tích--phản-biện-chuyên-sâu-từng-vấn-đề-kỹ-thuật)
3. [Phân Định Rạch Ròi: Hoàn Thiện Để Báo Cáo vs. Phương Hướng Phát Triển](#3-phân-định-rạch-ròi-hoàn-thiện-để-báo-cáo-vs-phương-hướng-phát-triển)
4. [Kế Hoạch Hành Động Cho Từng Thành Viên (TV1, TV2, TV3)](#4-kế-hoạch-hành-động-cho-từng-thành-viên-tv1-tv2-tv3)
5. [Kịch Bản Bảo Vệ & Bộ Câu Hỏi Phản Biện Sát Sườn Của Hội Đồng](#5-kịch-bản-bảo-vệ--bộ-câu-hỏi-phản-biện-sát-sườn-của-hội-đồng)

---

## 1. TỔNG QUAN KIỂM TOÁN MÃ NGUỒN & RÀ SOÁT THỰC TẾ CODEBASE

Sau khi thực hiện rà soát trực tiếp toàn bộ cây mã nguồn, dữ liệu thực nghiệm và hệ thống kiểm thử tự động, kết quả kiểm toán hiện trạng như sau:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        KẾT QUẢ KIỂM TOÁN THỰC TẾ CODEBASE VISIONPROF                  │
├───────────────────────────────┬────────────────────────────────────────────────────────┤
│ 1. Đơn vị kiểm thử (Unit Test)│ ✅ 84 PASSED / 84 tests trong môi trường ảo (.venv)   │
│ 2. Quy tắc mã nguồn (Linter)  │ ✅ PASS 100% (python3 scripts/verify_rules.py)         │
│ 3. Dữ liệu CPU Laptop         │ ✅ ĐẦY ĐỦ số đo thật: TN1, TN2, TN3, TN4, TN6, TN7     │
│ 4. Dữ liệu Colab T4           │ ✅ ĐÃ ĐO THẬT TN3, TN4, TN5 + TN6 nạp dữ liệu (18/10)  │
│ 5. Lỗi ghi đè độ dài câu NLP  │ ✅ ĐÃ SỬA XONG trong src/core/paths.py & runner        │
│ 6. Hạ tầng đo GPU bất đồng bộ │ ✅ HOÀN THIỆN: Deferred CUDA Events + Queue Prefill   │
│ 7. Động cơ 19 Luật chẩn đoán  │ ✅ HOÀN THIỆN logic v2.0 (nhưng chưa có code mẫu)      │
│ 8. Dashboard Web Streamlit    │ ✅ HOÀN THIỆN 8 Tabs tương tác Plotly                  │
└───────────────────────────────┴────────────────────────────────────────────────────────┘
```

### 1.1. Những gì đã hoàn thành xuất sắc trong mã nguồn
1. **Kiến trúc phân tầng chuẩn mực ([`src/core/`](../src/core/))**:
   - Tách bạch giao diện trừu tượng ([`src/core/interfaces.py`](../src/core/interfaces.py)) gồm `ModelAdapter` và `ProfilerPlugin`.
   - Chuẩn hóa cấu trúc đường dẫn sinh dữ liệu ([`src/core/paths.py`](../src/core/paths.py)) với `build_result_path`, bảo đảm cách ly 100% giữa các thực nghiệm mà không xung đột file.
2. **Kỹ thuật đo GPU bất đồng bộ khoa học ([`src/layer/events.py`](../src/layer/events.py))**:
   - `DeferredCudaDispatcher`: Chỉ đồng bộ `torch.cuda.synchronize()` duy nhất một lần ở cuối iteration, giữ cho GPU enqueue kernel liên tục.
   - `Adaptive Queue Prefill`: Dùng `torch.cuda._sleep()` giữ chân GPU ở đầu pass để CPU nạp toàn bộ danh sách kernel vào hàng đợi.
   - `LIFO Event Stack`: Quản lý sự kiện bằng ngăn xếp, đo chuẩn xác các module được gọi lặp lại (như `SiLU` trong YOLOv8n).
3. **Mô hình Roofline & Phân tích FLOPs giải tích ([`src/layer/roofline.py`](../src/layer/roofline.py))**:
   - Tự động phân loại `Compute-bound` vs `Memory-bound` và tính `% Peak Performance` dựa trên thông số phần cứng [`HardwareSpec`](../src/layer/hw_specs.py).
   - Bộ đếm FLOPs đa tầng kết hợp `FlopCounterMode` và bảng công thức giải tích cho Activation/Norm.
4. **Harness đo lường Model Zoo & Giám sát tài nguyên ([`src/model/`](../src/model/))**:
   - Hỗ trợ 8 model adapters ([`src/zoo/`](../src/zoo/)) phủ đủ CV, NLP, Tabular và XGBoost.
   - Đo sạch không hook (Phase 1), thu thập đủ 12 chỉ số thống kê (P50–P99, FPS, Peak VRAM).
   - `ResourceSampler` chạy luồng Daemon 100ms, không gọi NVML trong hook của layer.
5. **Bộ đo Giai đoạn & Chẩn đoán 19 Luật ([`src/phase/`](../src/phase/) & [`src/analyzer/`](../src/analyzer/))**:
   - [`PhaseTimer`](../src/phase/timer.py) phân rã chính xác các pha Training và Pipeline.
   - [`DataLoaderSentinel`](../src/phase/dataloader.py) tự loại trừ batch 0 cold-start.
   - 19 luật v2.0 đã sửa triệt để 4 lỗi logic cũ (`RB-04`, `RD-02`, `RD-05`, `RC-03`).
   - [`ABComparisonEngine`](../src/analyzer/ab_testing.py) kiểm định Mann-Whitney U và Cliff's Delta chuẩn thống kê.

---

## 2. PHÂN TÍCH & PHẢN BIỆN CHUYÊN SÂU TỪNG VẤN ĐỀ KỸ THUẬT

Dưới đây là phân tích và phản biện kỹ lưỡng 10 vấn đề kỹ thuật được rút ra từ việc đối chiếu codebase với các tài liệu nghiên cứu:

---

### Vấn đề 1: Tử huyệt dữ liệu TN3, TN4, TN5 trên Colab T4 của TV2
* **Hiện trạng thực tế**: Thư mục `results/colab_t4/` hiện tại **HOÀN TOÀN TRỐNG** cho các thực nghiệm TN3, TN4, TN5 (toàn bộ 39 thư mục dữ liệu giả lập sinh từ script cũ đã bị xoá theo yêu cầu tại [`YEU_CAU_DO_LAI_TN3_TN5.md`](tv2/YEU_CAU_DO_LAI_TN3_TN5.md)). Trong khi đó, `results/cpu_laptop/` đã có đầy đủ số đo thật.
* **Phản biện kỹ thuật**: 
  - Nếu mang báo cáo đi nộp hoặc trình bày slide mà chưa chạy đo thật trên Colab T4, các câu hỏi nghiên cứu **CH3** (Đảo thứ hạng CPU vs GPU) và **CH4** (Mở rộng Batch size, Tensor Cores FP16) sẽ không có bất kỳ bằng chứng thực nghiệm GPU nào hỗ trợ.
  - Hội đồng chuyên môn chỉ cần kiểm tra file `summary.json` và `benchmark.csv` là phát hiện ngay việc thiếu hụt số liệu GPU.
* **Mức độ nghiêm trọng**: **P0**. ✅ **Đã xong 18/10** (57 lần chạy trên Tesla T4, xem `KET_QUA_COLAB_GIUA_KY.md`).

---

### Vấn đề 2: "Thời gian vô danh" (Unattributed Time ~4–10%)
* **Hiện trạng thực tế**: TV1 dùng `register_forward_hook` trên các `nn.Module` lá ([`src/layer/plugin.py`](../src/layer/plugin.py)). Các phép toán dạng hàm (*functional operations*) như `x + shortcut` (phép cộng phần tử trong ResNet), `torch.cat` (ghép tensor trong YOLO/ShuffleNet) hay phép nhân ma trận Attention `Q @ K.T` không thông qua `nn.Module`.
* **Phản biện kỹ thuật**:
  - TN2 đã đo lường và ghi nhận tỷ lệ thời gian vô danh này: $Unattributed = \frac{T_{\text{clean}} - \sum T_{\text{layers}}}{T_{\text{clean}}} \approx 4\% \text{--} 10\%$.
  - Đối chiếu với bài báo **DeepContext (arXiv 2024)**: DeepContext giải quyết triệt để vấn đề này nhờ chèn tầng đệm `DLMonitor` thông qua `aten::addGlobalCallback` của PyTorch C++ engine, bắt trọn 100% mọi toán tử ATen cấp thấp.
  - Tuy nhiên, trong phạm vi đồ án hiện tại, việc viết wrapper C++ qua `LD_PRELOAD` là quá sức và không cần thiết. Việc TV1 chủ động định nghĩa và đo lường tỷ lệ Unattributed Time trong [`experiments/tn2_validate_profiler.py`](../experiments/tn2_validate_profiler.py) đã là một luận cứ khoa học rất trung thực và vững chắc.
* **Mức độ nghiêm trọng**: **P2 (Hạn chế lý thuyết đã kiểm soát - Đưa vào Future Work)**.

---

### Vấn đề 3: Sự bất lực trước Runtime biên dịch (`torch.compile`, TensorRT)
* **Hiện trạng thực tế**: VisionProf chỉ hoạt động trên **PyTorch Eager Mode**. Khi người dùng bật `torch.compile` (TorchInductor với Triton backend) hoặc xuất sang TensorRT / ONNX Runtime, toàn bộ đồ thị tính toán bị gom cụm (**Operator Fusion**), khiến hook của `nn.Module` không còn được kích hoạt.
* **Phản biện kỹ thuật**:
  - Đối chiếu với bài báo **PRoof (ICPP 2024)**: PRoof giải quyết bài toán này bằng cơ chế **Layer Backtracking**: xây dựng cấu trúc đồ thị ảo `_FusedOp`, đối sánh I/O tensor (`get_subgraph_ops_by_io`) để ánh xạ từ kernel đã compile quay ngược lại layer ban đầu trong mã nguồn Python.
  - Việc đòi hỏi VisionProf phải hỗ trợ mô hình đã biên dịch ngay trong đồ án hiện tại là không khả thi về mặt thời gian (đây là công trình nghiên cứu của cả một nhóm tiến sĩ tại Đại học Bắc Hàng & SenseTime trong nhiều năm).
* **Mức độ nghiêm trọng**: **P2 (Đặt vào Phương hướng phát triển tiếp theo)**.

---

### Vấn đề 4: Sai số mô hình hóa bộ nhớ (SRAM/Cache on-chip vs. DRAM)
* **Hiện trạng thực tế**: VisionProf tính lưu lượng bộ nhớ theo mô hình tĩnh: $\text{Bytes} = \text{Input} + \text{Weight} + \text{Output}$ ([`src/layer/roofline.py`](../src/layer/roofline.py)).
* **Phản biện kỹ thuật**:
  - Công thức này giả định mọi tensor đều phải nạp từ DRAM và ghi ra DRAM. Trong thực tế phần cứng GPU, dữ liệu giữa các layer liên tiếp (nhất là các layer nhỏ như BatchNorm, ReLU) được giữ lại trên **SRAM / L1 / L2 Cache** trên chip.
  - Do đó, lưu lượng DRAM thực tế nhỏ hơn lý thuyết, dẫn đến việc tính toán hiệu suất Roofline của một số layer nhỏ bị ảo (> 100% trần băng thông).
  - PRoof khắc phục bằng cách trừ khử toàn bộ tensor trung gian trong cụm `_FusedOp`. Đối với VisionProf hiện tại, chỉ cần nêu rõ giả định này trong phần phân tích lý thuyết Roofline là hoàn toàn thuyết phục.
* **Mức độ nghiêm trọng**: **P2 (Đặt vào Phương hướng phát triển tiếp theo)**.

---

### Vấn đề 5: Cấu trúc dữ liệu phẳng & Nguy cơ bùng nổ dữ liệu (Trace Bloat)
* **Hiện trạng thực tế**: Dữ liệu đo được ghi thô ra từng dòng trong file CSV phẳng (`layers.csv`, `phases.csv`).
* **Phản biện kỹ thuật**:
  - Nếu chỉ chạy 50 vòng đo benchmark, file CSV chỉ vài nghìn dòng (rất nhẹ, mở nhanh).
  - Nhưng nếu profile quá trình huấn luyện thật kéo dài hàng chục epoch (hàng trăm nghìn bước), file CSV sẽ phình to hàng trăm MB, gây cạn kiệt RAM và treo Dashboard.
  - Đối chiếu với **DeepContext (arXiv 2024)**: Họ dùng cấu trúc cây **Calling Context Tree (CCT)** và thuật toán **Online Streaming Aggregation** (Welford) để cộng dồn trực tiếp $P_{50}, P_{99}, \mu, \sigma$ vào nút cây, giúp tiết kiệm bộ nhớ tới $27\times$.
  - Với mục tiêu đồ án hiện tại (chạy benchmark có kiểm soát số vòng), định dạng CSV phẳng là đủ tốt và dễ xử lý bằng `pandas`.
* **Mức độ nghiêm trọng**: **P3 (Đặt vào Phương hướng phát triển tiếp theo)**.

---

### Vấn đề 6: Điểm mù chiều ngược (Backward Pass Blackbox)
* **Hiện trạng thực tế**: TV3 đo chiều backward thông qua context manager:
  ```python
  with timer.phase("backward"):
      loss.backward()
  ```
* **Phản biện kỹ thuật**:
  - Kết quả chỉ cho biết toàn bộ pha backward mất bao nhiêu ms (ví dụ 12ms, gấp đôi forward), nhưng **hoàn toàn mù bên trong**: không biết layer nào hay toán tử nào đang gây nghẽn backward.
  - Đối chiếu với **DeepContext**: Họ giải quyết bằng cách dùng `Sequence ID` để liên kết các autograd worker threads trong C++ ngược về call stack Python ở forward pass.
  - Trong đồ án hiện tại, mục tiêu của TV3 là kiểm chứng lý thuyết $Backward / Forward \approx 2.0\times$ (CH5). Phép đo ở mức PhaseTimer đã trả lời câu hỏi này ($1.39\times$ với MLP đến $2.21\times$ với DistilBERT). Việc phân rã sâu từng layer trong backward là bài toán nâng cao cho tương lai.
* **Mức độ nghiêm trọng**: **P2 (Đặt vào Phương hướng phát triển tiếp theo)**.

---

### Vấn đề 7: Động cơ chẩn đoán thiếu "Gợi ý hành động sửa code" (Actionable Recommendations)
* **Hiện trạng thực tế**: Hệ thống 19 luật v2.0 trong [`src/analyzer/rules/`](../src/analyzer/rules/) trả về trạng thái (`PASS`, `WARN`, `FAIL`, `SKIPPED`) và chuỗi `message` mô tả triệu chứng, nhưng trường `recommendation` của dataclass [`RuleResult`](../src/core/types.py#L88) thường bị để trống hoặc chỉ có lời khuyên chung chung.
* **Phản biện kỹ thuật**:
  - Một hệ thống chẩn đoán xuất sắc không chỉ "bắt bệnh" mà phải "kê đơn". Nếu người dùng thấy cảnh báo `RD-02: 30% layers < 1us`, họ không biết phải sửa thế nào.
  - Nếu hệ thống tự động in ra đoạn code mẫu (`code_snippet`): `torch.ao.quantization.fuse_modules(...)` hay cấu hình chuẩn `DataLoader(..., num_workers=4, pin_memory=True)`, giá trị thực tiễn của công cụ sẽ tăng vọt.
  - **Khả thi**: Việc này cực kỳ dễ làm trong mã nguồn hiện tại, chỉ cần bổ sung text string vào hàm kiểm tra rule mà không làm vỡ schema dữ liệu.
* **Mức độ nghiêm trọng**: **P1 (Rất quan trọng — Nên hoàn thiện ngay trước báo cáo)**.

---

### Vấn đề 8: Thiếu kịch bản tự động hóa Ablation Study mini (Bảo chứng phương pháp đo)
* **Hiện trạng thực tế**: Trong [`PROJECT_PLAN.md`](../PROJECT_PLAN.md) đề xuất 7 bài toán bóc tách thành phần (AB-1 đến AB-7). Tuy nhiên, codebase hiện tại chưa có script tự động chạy trọn gói các bài toán này.
* **Phản biện kỹ thuật**:
  - Khi bảo vệ, hội đồng rất hay hỏi: *"Làm sao các em biết warm-up 15 vòng là đủ?"* hoặc *"Bộ theo dõi tài nguyên ResourceSampler có làm chậm mô hình không?"*.
  - Nếu có một script `experiments/run_ablation_mini.py` quét $N_{warmup} \in [0, 50]$ chứng minh phương sai bão hòa ở $N=15$, và đo overhead của Layer hook so với ResourceSampler, nhóm sẽ có bằng chứng đanh thép không thể bị bẻ gãy.
* **Mức độ nghiêm trọng**: **P1 (Rất quan trọng — Nên hoàn thiện ngay trước báo cáo)**.

---

### Vấn đề 9: Khóa chặt vào PyTorch và NVIDIA CUDA Linux (Platform Lock-in)
* **Hiện trạng thực tế**: VisionProf gắn chặt với PyTorch API và NVIDIA CUDA Event.
* **Phản biện kỹ thuật**:
  - PRoof hỗ trợ TensorRT, OpenVINO, ONNX Runtime trên cả GPU, CPU x86/ARM và NPU di động; DeepContext hỗ trợ cả PyTorch, JAX, NVIDIA và AMD ROCm.
  - Tuy nhiên, việc hỗ trợ đa nền tảng đòi hỏi hạ tầng phần cứng thật (phải có máy chạy card AMD, chip Apple Silicon, chip Jetson). Đây là bài toán kỹ thuật hệ thống mở rộng, hoàn toàn nằm ngoài phạm vi một đồ án tốt nghiệp đại học.
* **Mức độ nghiêm trọng**: **P3 (Đặt vào Phương hướng phát triển tiếp theo)**.

---

### Vấn đề 10: Trải nghiệm tách rời IDE lập trình (No IDE Integration)
* **Hiện trạng thực tế**: Giao diện người dùng sử dụng Streamlit Web App độc lập ([`src/dashboard/app.py`](../src/dashboard/app.py)).
* **Phản biện kỹ thuật**:
  - DeepContext xây dựng tiện ích mở rộng VSCode Extension với tính năng **Click-to-Source Navigation** (nhấn chuột vào biểu đồ ngọn lửa là con trỏ nhảy ngay đến dòng code Python tương ứng).
  - Đối với bài toán báo cáo và trình diễn trước hội đồng, Web Dashboard trên Streamlit là hình thức demo trực quan, sống động và dễ theo dõi nhất. Việc phát triển extension cho IDE thuộc về khía cạnh sản phẩm thương mại.
* **Mức độ nghiêm trọng**: **P3 (Đặt vào Phương hướng phát triển tiếp theo)**.

---

## 3. PHÂN ĐỊNH RẠCH RÒI: HOÀN THIỆN ĐỂ BÁO CÁO VS. PHƯƠNG HƯỚNG PHÁT TRIỂN

Dựa trên phân tích phản biện ở trên, dưới đây là bảng phân định rạch ròi hai nhóm nhiệm vụ:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│              MA TRẬN PHÂN ĐỊNH PHẠM VI DỰ ÁN: HIỆN TẠI VS. TƯƠNG LAI                   │
├────────────────────────────────────────────────────────┬───────────────────────────────┤
│ NHÓM A: BẮT BUỘC HOÀN THIỆN ĐỂ BÁO CÁO (CURRENT SCOPE)  │ NHÓM B: ĐẶT VÀO HƯỚNG PHÁT    │
│                                                        │ TRIỂN TIẾP THEO (FUTURE WORK) │
├────────────────────────────────────────────────────────┼───────────────────────────────┤
│ 1. [P0] Chạy đo thật Colab T4 cho TN3, TN4, TN5 (TV2)  │ 1. Chuyển sang TorchDispatch/ │
│    - Xóa script populate, đo thật trên T4 qua notebook │    ATen Callbacks cấp thấp.   │
│    - Cập nhật số liệu & biểu đồ CH3, CH4 chuẩn xác.    │ 2. Hỗ trợ mô hình biên dịch   │
│ 2. [P0] Sửa dứt điểm phân biệt seq_len NLP             │    (torch.compile, TensorRT). │
│    - Đã có code trong paths.py, cần verify khi đo thật │ 3. Cài đặt Layer Backtracking │
│ 3. [P1] Bổ sung Code mẫu vào 3 Luật Chẩn đoán chính   │    và toán tử ảo _FusedOp.    │
│    - Thêm actionable snippets cho RD-02, RC-03, RD-06  │ 4. Mô hình hóa SRAM vs DRAM   │
│ 4. [P1] Viết kịch bản tự động Ablation Study mini      │    (loại trừ cache traffic).  │
│    - Quét bão hòa N_warmup và bóc tách overhead.       │ 5. Calling Context Tree (CCT) │
│ 5. [P0] Chuẩn bị Luận cứ bảo vệ về các hạn chế đã biết │    và Online Aggregation.     │
│    - Giải trình khoa học 10% Unattributed Time (TN2).  │ 6. Phân rã Backward chi tiết  │
│    - Giải trình lý do chọn Eager mode thay compile.    │    thông qua Sequence ID.     │
│ 6. [P0] Hoàn thiện Video Demo 90 giây Dashboard        │ 7. Đa nền tảng: AMD ROCm,     │
│    - Nhúng vào Slide 20 phục vụ phương án demo an toàn │    Apple Metal, NPU Edge.     │
│ 7. [P0] Kiểm tra Linter & Tests (verify_rules & pytest)│ 8. Tích hợp VSCode Extension  │
│    - Bảo đảm exit code 0 trước khi đóng gói báo cáo.   │    Click-to-Source Navigation.│
└────────────────────────────────────────────────────────┴───────────────────────────────┘
```

---

## 4. KẾ HOẠCH HÀNH ĐỘNG CHO TỪNG THÀNH VIÊN (TV1, TV2, TV3)

### 🟧 4.1. Nhiệm vụ Thành viên 1 (TV1 - Mức Layer & Nền tảng Đo lường)
* **Việc cần hoàn thiện ngay để báo cáo**:
  1. **Xây dựng kịch bản Ablation Study mini** (`experiments/run_ablation_mini.py`):
     - Đo phương sai độ trễ khi $N_{warmup} \in [0, 5, 10, 15, 20, 30]$ trên ResNet-50 $\rightarrow$ Chứng minh tại sao chọn $N_{warmup}=15$.
     - Đo bóc tách Overhead: Chạy sạch vs Bật Layer Hook vs Bật kèm ResourceSampler 100ms $\rightarrow$ Đo overhead thật (TN2: 5–11% trên CPU) và giải trình nguyên nhân Overhead ở batch 1 trên GPU (ngưỡng sàn CUDA event ~6 µs).
  2. **Hoàn thiện slide Mức Layer (Slide 4–9)**:
     - Làm rõ luận cứ khoa học về 4–10% "Thời gian vô danh" (*Unattributed time* do các phép toán dạng hàm `x + shortcut`, `torch.cat` không thông qua `nn.Module`).
* **Nội dung nhận lãnh trong phần "Hướng phát triển tương lai"**:
  - Trình bày giải pháp chuyển đổi từ hook `nn.Module` sang `TorchDispatchMode` trong PyTorch 2.x để triệt tiêu hoàn toàn thời gian vô danh.
  - Trình bày giải thuật **Layer Backtracking & `_FusedOp`** (học tập từ PRoof ICPP 2024) để profile các mô hình đã biên dịch bằng `torch.compile`.

---

### 🟦 4.2. Nhiệm vụ Thành viên 2 (TV2 - Mức Model & Đo chuẩn Vĩ mô)
* **Việc cần hoàn thiện ngay để báo cáo (Đang là nút thắt tiến độ)**:
  1. **Chạy đo thật trên Google Colab T4 cho TN3, TN4, TN5**:
     - Mở [`notebooks/colab_runner.ipynb`](../notebooks/colab_runner.ipynb), chọn runtime GPU T4, chạy Cell 5 chế độ **[2] GPU T4 chuyên sâu**.
     - Đảm bảo đủ 8 model cho TN3, 8 model $\times$ batch (1, 8, 32) cho TN4, và 7 model FP16 cho TN5.
     - Kiểm tra kết quả: File `env.json` phải ghi nhận card GPU Tesla T4 thật, timestamp không cách đều giả tạo.
  2. **Cập nhật lại toàn bộ biểu đồ và báo cáo TV2**:
     - Chạy `python experiments/tv2_analysis.py` để sinh lại hình ảnh trong `docs/tv2/figures/`.
     - Viết lại phần kết quả cho **CH3** (Hiện tượng đảo thứ hạng) và **CH4** (Batch scaling & Tensor Cores FP16) trong [`docs/tv2/BAO_CAO_TV2.md`](tv2/BAO_CAO_TV2.md) và Slide 10–15.
* **Nội dung nhận lãnh trong phần "Hướng phát triển tương lai"**:
  - Trình bày định hướng mở rộng Model Zoo sang mô hình Đa phương thức (**Multimodal AI - CLIP/SmolVLM**).
  - Trình bày về khả năng đo đạc đa nền tảng trên GPU AMD (ROCm) và các dòng chip Apple Silicon.

---

### 🟩 4.3. Nhiệm vụ Thành viên 3 (TV3 - Mức Giai đoạn, Chẩn đoán & Tổng kết)
* **Việc cần hoàn thiện ngay để báo cáo**:
  1. **Nâng cấp Động cơ Chẩn đoán với Code mẫu gợi ý (Actionable Code Suggestions)**:
     - Trong [`src/analyzer/rules/architecture.py`](../src/analyzer/rules/architecture.py), cập nhật luật `RD-02` (Sub-microsecond layers) bổ sung gợi ý gộp tầng:
       ```python
       recommendation = "Gợi ý: Áp dụng torch.ao.quantization.fuse_modules(model, [['conv', 'bn', 'relu']]) để triệt tiêu overhead."
       ```
     - Trong [`src/analyzer/rules/dataloader.py`](../src/analyzer/rules/dataloader.py), cập nhật luật `RC-03` (GPU Starvation) bổ sung cấu hình DataLoader tối ưu:
       ```python
       recommendation = "Gợi ý: Cấu hình DataLoader(..., num_workers=4, pin_memory=True, persistent_workers=True)."
       ```
     - Thêm trường `recommendation` hiển thị rõ ràng trên Tab 7 của Dashboard ([`src/dashboard/tabs/tab_diagnostics.py`](../src/dashboard/tabs/tab_diagnostics.py)).
  2. **Chuẩn bị Demo và Slide báo cáo chung**:
     - Quay video chất lượng cao 90 giây walkthrough 4 Tab trọng tâm của Streamlit Dashboard, nhúng trực tiếp vào Slide 20 (theo dàn ý [`DAN_Y_SLIDE.md`](DAN_Y_SLIDE.md)).
     - Tổng duyệt bài thuyết trình 24 phút, bảo đảm phân phối thời gian nhịp nhàng giữa 3 thành viên.
* **Nội dung nhận lãnh trong phần "Hướng phát triển tương lai"**:
  - Trình bày về **Calling Context Tree (CCT)** và giải pháp **Backward Attribution via Sequence ID** (học tập từ DeepContext) để bóc tách chi tiết từng kernel trong pha Backward.
  - Trình bày về ý tưởng phát triển **VSCode Extension** tích hợp Flame Graph Click-to-Source.

---

## 5. KỊCH BẢN BẢO VỆ & BỘ CÂU HỎI PHẢN BIỆN SÁT SƯỜN CỦA HỘI ĐỒNG

Dưới đây là 5 câu hỏi "hóc búa" nhất mà Hội đồng chấm đồ án chắc chắn sẽ xoáy sâu, kèm theo câu trả lời mẫu mực, chuẩn xác về mặt khoa học:

### Câu hỏi 1 (Dành cho TV1): *"Tại sao Overhead trên GPU ở batch 1 lên tới khoảng 110–210% mà nhóm vẫn khẳng định công cụ đo lường của mình là chính xác?"*
> **Trả lời chuẩn mực**:  
> *"Thưa Hội đồng, đây chính là hiện tượng **Nghịch lý Overhead** kinh điển trong đo lường hệ thống (đã được công bố tại hội nghị quốc tế IEEE IPDPS 2020 bởi Li et al.). Trên GPU, mỗi cặp CUDA Event có một chi phí vật lý cố định từ $6\,\mu s$ đến $10\,\mu s$ (ngưỡng sàn timer). Khi chạy ở batch size 1, các layer rất nhỏ như ReLU hay BatchNorm chỉ tốn $2\text{--}3\,\mu s$, nên chi phí ghi nhận event vô tình lớn hơn chính thời gian tính toán của layer đó.  
> Tuy nhiên, kiểm chứng TN2 của nhóm so sánh trực tiếp với `torch.profiler` cho thấy **sai lệch tổng luôn dưới 8% trên GPU và dưới 1% trên CPU** (đo lại bằng code hiện tại ngày 18/10 vẫn vậy). Riêng các layer siêu nhỏ như ReLU, sai số theo loại layer lớn hơn nhiều vì ngưỡng sàn event, nên nhóm chỉ dùng chúng để so sánh tương đối. Đặc biệt, cơ chế **Deferred CUDA Events** của nhóm chỉ đồng bộ duy nhất một lần ở cuối pass, hoàn toàn không làm ngắt quãng hàng đợi của GPU. Nhóm đã phân định kiến trúc rất rạch ròi: **Mức Layer chỉ dùng để phân rã tỷ trọng %, còn tổng thời gian sạch tuyệt đối của mô hình được giao cho Mức Model đo đạc độc lập không gắn hook**."*

---

### Câu hỏi 2 (Dành cho TV1): *"Nhóm dùng forward hook trên nn.Module, vậy các phép toán dạng hàm như phép cộng residual trong ResNet hay phép nhân ma trận Attention biến đi đâu?"*
> **Trả lời chuẩn mực**:  
> *"Thưa Hội đồng, các phép toán dạng hàm (*functional operations*) như `x + shortcut` hay `torch.cat` không kế thừa từ `nn.Module` nên PyTorch hook không thể can thiệp. Nhóm hoàn toàn nhận thức được giới hạn này và đã đưa ra chỉ số đo lường tường minh trong TN2 gọi là **Unattributed Time (Thời gian vô danh)**:
> $$Unattributed = \frac{T_{\text{clean}} - \sum T_{\text{layers}}}{T_{\text{clean}}}$$
> Thực nghiệm TN2 trên ResNet-50 và BERT-base cho thấy phần thời gian vô danh này dao động từ **$4\%$ đến $10\%$** (CPU: 3.9% và 9.2%; T4: 9.0% và 10.3%). Điều này chứng minh các layer `nn.Module` chính vẫn chi phối áp đảo trên $90\%$ thời gian thực thi của mạng. Việc triệt tiêu hoàn toàn $10\%$ này đòi hỏi phải can thiệp vào tầng phân phối toán tử `TorchDispatchMode` của PyTorch 2.x hoặc thư viện C++ `libunwind`, và nhóm đã xếp đây là mục tiêu ưu tiên số 1 trong phương hướng phát triển tiếp theo."*

---

### Câu hỏi 3 (Dành cho TV2): *"Tại sao MobileNetV3 được thiết kế cho thiết bị nhẹ mà khi chạy trên CPU lại chậm hơn ResNet-50 tới gần 8 lần?"*
> **Trả lời theo số đo thật**:  
> *"Thưa Hội đồng, con số 176 ms đó chỉ xuất hiện trên MacBook chip Apple (CPU ARM). Trên laptop x86 của nhóm, MobileNetV3 chỉ mất khoảng 12 ms. Vì vậy đây là đặc thù của kernel PyTorch cho depthwise convolution trên CPU ARM, không phải tính chất chung của CPU; nhóm trình bày nó như một ví dụ cho việc **kết quả phụ thuộc cả phần cứng lẫn mức tối ưu thư viện**.  
> Trên GPU T4 ở batch 1, ResNet-50 (8.2 GFLOPs) và MobileNetV3 (0.43 GFLOPs, ít hơn 19 lần) cùng mất khoảng 6 ms: GPU đang bị giới hạn bởi chi phí gửi kernel chứ không phải số phép tính. Đúng tinh thần ShuffleNet V2 (ECCV 2018): FLOPs không tỷ lệ thuận với tốc độ thực tế. Khi tăng batch lên 32, MobileNetV3 mới bứt lên (throughput tăng 8.8×, ResNet-50 chỉ 2.2×)."*

---

### Câu hỏi 4 (Dành cho TV3): *"Khâu tiền xử lý ảnh và NMS của YOLOv8n chiếm tới 41% chu kỳ frame trên GPU T4. Ý nghĩa thực tiễn của con số này là gì?"*
> **Trả lời chuẩn mực**:  
> *"Thưa Hội đồng, con số $41.0\%$ trong thực nghiệm TN7 là minh chứng thực nghiệm sinh động nhất cho **Định luật Amdahl** và khái niệm **'Thuế AI' (AI Tax)** trong các ứng dụng thị giác máy tính thực tế.  
> Khi chạy trên CPU, mô hình suy luận chiếm tới $94\%$ thời gian, che lấp toàn bộ các khâu tiền/hậu xử lý. Nhưng khi đưa lên GPU T4, mạng nơ-ron được tăng tốc tới 26 lần (từ 163.6 ms xuống 6.3 ms), khiến khâu tiền xử lý (resize, chuẩn hóa ảnh bằng OpenCV trên CPU) và thuật toán NMS tuần tự lộ rõ thành **nút thắt cổ chai chi phối toàn bộ hệ thống**.  
> Kết quả này cảnh báo các kỹ sư AI rằng: Nếu chỉ tập trung tối ưu mạng nơ-ron mà bỏ quên đường ống dữ liệu, hiệu năng ứng dụng sẽ nhanh chóng chạm trần. Giải pháp kỹ thuật khắc phục là phải chuyển toàn bộ pipeline tiền xử lý và NMS sang thực thi trực tiếp trên GPU thông qua NVIDIA DALI hoặc Torchvision GPU Ops."*

---

### Câu hỏi 5 (Dành cho cả nhóm): *"Tại sao công cụ của nhóm không hỗ trợ torch.compile hay TensorRT — vốn là chuẩn mực triển khai AI hiện nay?"*
> **Trả lời chuẩn mực**:  
> *"Thưa Hội đồng, nhóm định vị VisionProf là một công cụ **Profiling trong giai đoạn Nghiên cứu & Thiết kế Kiến trúc (Development & Architectural Profiling)**. Ở giai đoạn này, nhà nghiên cứu cần nhìn thấy cấu trúc giải phẫu vi mô của từng tầng mạng, từng tham số và từng hàm kích hoạt cụ thể.  
> Khi mô hình được đưa vào các runtime biên dịch tối ưu như `torch.compile` (Inductor backend) hay TensorRT, đồ thị tính toán bị biến đổi hoàn toàn: nhiều tầng bị gom cụm (**Operator Fusion**), các tầng định dạng dữ liệu tự động được chèn vào. Khi đó, các công cụ đo thông thường sẽ mất hoàn toàn dấu vết của mã nguồn Python ban đầu.  
> Như bài báo đỉnh cao **PRoof công bố tại hội nghị ICPP tháng 8/2024** đã chỉ ra, việc đo lường các runtime biên dịch đòi hỏi phải xây dựng một giải thuật ánh xạ ngược đồ thị vô cùng phức tạp (**Layer Backtracking** qua nút ảo `_FusedOp`). Nhóm đã nghiên cứu rất kỹ lý thuyết của PRoof và xác định đây là trọng tâm nâng cấp lớn nhất của VisionProf trong giai đoạn tiếp theo."*

---

## 6. KẾT LUẬN & BẢNG TIẾN ĐỘ THỰC HIỆN TRƯỚC NGÀY BÁO CÁO

Để bảo đảm đồ án đạt điểm tối đa và hoàn thiện mọi tài liệu, toàn bộ các công việc cấp thiết thuộc **Nhóm A** cần được thực hiện theo đúng mốc thời gian:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                      LỊCH TRÌNH 3 BƯỚC ĐÓNG GÓI BÁO CÁO VISIONPROF                     │
├───────────────┬─────────────────────────────────────────────────────────┬──────────────┤
│ MỐC THỜI GIAN │ NỘI DUNG CÔNG VIỆC BẮT BUỘC                             │ PHỤ TRÁCH    │
├───────────────┼─────────────────────────────────────────────────────────┼──────────────┤
│ **BƯỚC 1**    │ • Chạy notebook Colab T4 thu thập số đo thật TN3, TN4,  │ 🟦 **TV2**   │
│ *(Ưu tiên 1)* │   TN5 cho 8 mô hình.                                    │              │
│               │ • Chạy tv2_analysis.py cập nhật biểu đồ CH3, CH4.       │              │
├───────────────┼─────────────────────────────────────────────────────────┼──────────────┤
│ **BƯỚC 2**    │ • Bổ sung Code gợi ý sửa lỗi vào 3 luật (RD-02, RC-03). │ 🟩 **TV3**   │
│ *(Ưu tiên 2)* │ • Viết script run_ablation_mini.py đo bão hòa warm-up.  │ 🟧 **TV1**   │
│               │ • Quay video 90s Demo Dashboard chèn vào Slide 20.      │ 🟩 **TV3**   │
├───────────────┼─────────────────────────────────────────────────────────┼──────────────┤
│ **BƯỚC 3**    │ • Chạy scripts/verify_rules.py & pytest (exit code 0).  │ 👥 **Cả nhóm│
│ *(Đóng gói)*  │ • Khớp nối dàn ý slide DAN_Y_SLIDE.md và tổng duyệt.    │              │
└───────────────┴─────────────────────────────────────────────────────────┴──────────────┘
```

Tài liệu này là kim chỉ nam phân định rõ ràng giữa **hiện thực triển khai** và **tầm nhìn khoa học**. Bằng việc hoàn thiện dứt điểm các số đo thực nghiệm còn thiếu và bảo vệ vững chắc các luận điểm khoa học, nhóm VisionProf hoàn toàn tự tin đạt kết quả xuất sắc nhất trước Hội đồng đánh giá.
