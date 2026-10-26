# Tổng hợp nghiên cứu liên quan

> Dùng cho phần **"Nghiên cứu liên quan"** của bài trình bày (TV1, khoảng 4 phút) và Chương 3 của báo cáo.
> Mọi con số dưới đây đã được đối chiếu với nội dung paper; trích dẫn đã kiểm tra qua Crossref / arXiv.
> Bản đầy đủ hơn (12 paper theo 5 trục, rủi ro kỹ thuật, 19 luật chẩn đoán, lộ trình) của TV2:
> [`../NGHIEN_CUU_KHOA_HOC_VA_CO_SO_PHAT_TRIEN.md`](../NGHIEN_CUU_KHOA_HOC_VA_CO_SO_PHAT_TRIEN.md).
> Hình trong `figures/` được cắt từ PDF gốc. Khi đưa lên slide, **ghi nguồn ngay dưới hình** (dòng "Nguồn" ở từng mục).

## 1. Bức tranh chung

Các nghiên cứu về profiling mô hình học sâu chia theo đúng 3 mức mà VisionProf đo:

| Mức | Câu hỏi | Paper | Liên hệ VisionProf |
| :--- | :--- | :--- | :--- |
| Nền tảng | Một phép tính đang bị giới hạn bởi tính toán hay bởi băng thông bộ nhớ? | [1] Roofline | Cách tính % hiệu suất ở mức Layer |
| 🟧 Layer | Thời gian đi vào layer / kernel nào, và đo như vậy có đúng không? | [2] XSP · [3] PRoof · [4] Hierarchical Roofline | TV1 · CH1, CH2 |
| 🟦 Model | Đo độ trễ, thông lượng của cả mô hình thế nào cho công bằng, lặp lại được? | [5] MLPerf Inference · [6] MLModelScope | TV2 · CH3, CH4 |
| 🟩 Giai đoạn | Ngoài mô hình (tiền / hậu xử lý, nạp dữ liệu, training) tốn bao nhiêu? | [7] AI Tax · [8] Data Stalls · [9] TBD | TV3 · CH5, CH6 |

**Khoảng trống VisionProf lấp:** các công trình trên thường đo **một** mức, hoặc là hệ thống lớn (cụm máy,
nhiều framework, hardware counter). VisionProf gom **cả 3 mức** vào một công cụ nhẹ cho PyTorch, không sửa
mã mô hình; đo cả **suy luận lẫn training và pipeline thật**; so **CPU laptop với GPU T4**; và **tự kiểm chứng
độ chính xác của công cụ đo** so với `torch.profiler` (TN2).

## 2. Từng paper

### [1] Roofline — nền tảng của phân tích hiệu suất
- **Trích dẫn:** S. Williams, A. Waterman, D. Patterson, "Roofline: An Insightful Visual Performance Model for
  Multicore Architectures," *Communications of the ACM*, 52(4), pp. 65–76, 2009.
- **Ý chính:** hiệu năng đạt được = min(đỉnh tính toán, băng thông bộ nhớ × cường độ số học FLOP/byte).
  Vẽ trên đồ thị log–log, điểm nằm dưới "mái dốc" là memory-bound, dưới "mái bằng" là compute-bound.
- **VisionProf:** `src/layer/roofline.py` dùng đúng công thức này cho từng layer, với thông số CPU laptop và T4.
- **Hình:** `figures/01_roofline_fig1.png` · Nguồn: Williams et al., CACM 2009, Fig. 1.

### [2] XSP — profiling xuyên tầng, xử lý overhead của chính công cụ đo
- **Trích dẫn:** C. Li, A. Dakkak, J. Xiong, W. Wei, L. Xu, W. Hwu, "XSP: Across-Stack Profiling and Analysis of
  Machine Learning Models on GPUs," *IEEE IPDPS*, 2020. arXiv:1908.06869.
- **Ý chính:** các công cụ cũ chỉ nhìn một tầng (model, framework, thư viện, phần cứng). XSP ghép chúng bằng
  distributed tracing thành một timeline model → layer → GPU kernel; phân tích 65 mô hình.
- **Điểm quan trọng nhất cho nhóm:** XSP thừa nhận **đo càng sâu thì overhead càng lớn** và đưa ra cách đo
  "leveled and iterative" (đo lần lượt từng mức) để số đo mỗi mức không bị overhead của mức sâu hơn làm sai.
- **VisionProf:** cùng vấn đề, cách giải khác: TN2 đo thẳng overhead (CPU 5–11%, GPU 109–166%) và sai lệch so với
  `torch.profiler` (< 1% CPU, < 8% GPU), rồi **tách** đo mức Layer khỏi lượt benchmark mức Model.
- **Hình:** `03_xsp_fig1_levels.png` (3 mức model / layer / kernel), `04_xsp_fig2_leveled_overhead.png`
  (overhead xuất hiện khi đo sâu hơn), `05_xsp_fig9_layer_roofline.png` · Nguồn: Li et al., IPDPS 2020.

### [3] PRoof — Roofline phân cấp cho từng layer
- **Trích dẫn:** S. Wu, H. Yang, X. You, R. Gong, Y. Liu, Z. Luan, D. Qian, "PRoof: A Comprehensive Hierarchical
  Profiling Framework for Deep Neural Networks with Roofline Analysis," *ICPP '24*, pp. 822–832, 2024.
  DOI: 10.1145/3673038.3673116 (truy cập mở, CC BY 4.0).
- **Ý chính:** gắn hiệu năng phần cứng ngược về layer của mô hình gốc (kể cả khi backend đã gộp nhiều layer),
  dùng mô hình giải tích để ước lượng FLOP và lượng truy cập bộ nhớ thay vì hardware counter tốn kém, và vẽ
  Roofline theo nhiều cấp, trên nhiều nền tảng.
- **VisionProf:** cũng đếm FLOP bằng phân tích (`FlopCounterMode`) thay vì hardware counter, và vẽ Roofline từng
  layer. Khác: PRoof tập trung suy luận qua backend tối ưu; VisionProf đo trực tiếp trên PyTorch, thêm
  training và pipeline.
- **Hình:** chưa có (ACM chặn tải tự động). Xem Mục 5.
- Nhóm khác đã trình bày kỹ PRoof và MLModelScope: nhóm mình chỉ nhắc ngắn và **so sánh**, không kể lại.

### [4] Hierarchical Roofline cho ứng dụng học sâu
- **Trích dẫn:** C. Yang, Y. Wang, S. Farrell, T. Kurth, S. Williams, "Hierarchical Roofline Performance Analysis
  for Deep Learning Applications," arXiv:2009.05257, 2020.
- **Ý chính:** Roofline theo nhiều tầng bộ nhớ (L1, L2, HBM) và có trần riêng cho Tensor Core / FP16; áp dụng cho
  DeepCAM trên TensorFlow và PyTorch, tách forward, backward và optimizer.
- **VisionProf:** liên hệ trực tiếp với CH4 (FP16 trên Tensor Core của T4) và với cách tách forward / backward /
  optimizer ở TN6.
- **Hình:** `02_hier_roofline_fig3.png` · Nguồn: Yang et al., arXiv:2009.05257, Fig. 3.

### [5] MLPerf Inference — đo mức Model sao cho công bằng
- **Trích dẫn:** V. J. Reddi et al., "MLPerf Inference Benchmark," *ACM/IEEE ISCA*, 2020. arXiv:1911.02549.
- **Ý chính:** chuẩn benchmark suy luận với 4 kịch bản (single-stream, multistream, server, offline). Kịch bản
  single-stream lấy **độ trễ phân vị 90**; server phải đảm bảo ràng buộc độ trễ ở **phân vị 99**, không dùng trung bình.
- **VisionProf:** là lý do mức Model báo P50 / P90 / P95 / P99 và tỷ lệ P99/P50 thay vì chỉ trung bình (TV2).
- **Hình:** `09_mlperf_fig2_stack.png` (quá nhiều lựa chọn ở mỗi tầng phần mềm, phần cứng),
  `10_mlperf_fig4_scenarios.png` (4 kịch bản) · Nguồn: Reddi et al., ISCA 2020.

### [6] MLModelScope — đánh giá mô hình ở 5 mức trừu tượng
- **Trích dẫn:** A. Dakkak, C. Li, J. Xiong, W. Hwu, "MLModelScope: A Distributed Platform for Model Evaluation and
  Benchmarking at Scale," arXiv:2002.08295, 2020.
- **Ý chính:** nền tảng phân tán, độc lập framework / phần cứng; mô tả việc chạy một mô hình qua 5 mức
  L1 Model → L2 Framework → L3 Layer → L4 Library → L5 Hardware; chỉ ra thay đổi nhỏ trong pipeline (thứ tự
  kênh màu, cách giải mã ảnh) làm đổi kết quả.
- **VisionProf:** 3 mức của nhóm tương ứng L1 (Model + Giai đoạn) và L2 (các layer Conv, BN, ReLU trong framework).
  Nhóm không đo L3–L5 (lời gọi cuDNN trong mỗi layer, kernel GPU, hardware counter): đó là hạn chế cần nói rõ.
- **Hình:** `06_mlmodelscope_fig1_L1_L5.png` (**hình nên dùng làm nền cho slide "bức tranh chung"**),
  `08_mlmodelscope_fig8_layer_timeline.png` · Nguồn: Dakkak et al., arXiv:2002.08295.

### [7] AI Tax — phần ngoài mô hình mới là điểm nghẽn
- **Trích dẫn:** D. Richins et al., "Missing the Forest for the Trees: End-to-End AI Application Performance in
  Edge Data Centers," *IEEE HPCA*, pp. 515–528, 2020. Bản mở rộng: "AI Tax: The Hidden Cost of AI Data Center
  Applications," arXiv:2007.10571.
- **Ý chính:** với ứng dụng nhận diện khuôn mặt, **dưới 60%** độ trễ đầu–cuối nằm ở các bước AI; hơn 1/3 thời gian
  của một khung hình nằm ở broker. Theo định luật Amdahl, tăng tốc riêng phần AI nhanh chóng hết tác dụng.
- **VisionProf:** chính là CH6 / TN7. YOLOv8n trên T4: tiền xử lý + NMS chiếm khoảng 41% thời gian pipeline.
- **Hình:** `11_aitax_fig6_latency_breakdown.png`, `12_aitax_fig9_amdahl.png` · Nguồn: Richins et al., arXiv:2007.10571.

### [8] Data Stalls — training phải chờ dữ liệu
- **Trích dẫn:** J. Mohan, A. Phanishayee, A. Raniwala, V. Chidambaram, "Analyzing and Mitigating Data Stalls in DNN
  Training," *PVLDB*, 14(5), pp. 771–784, 2021. arXiv:2007.06775.
- **Ý chính:** phân tích 9 mô hình: trong nhiều trường hợp thời gian training **chủ yếu là chờ dữ liệu** (đọc từ
  ổ đĩa và tiền xử lý). Công cụ DS-Analyzer đo stall; thư viện CoorDL giảm thời gian training tới 5× trên một máy.
- **VisionProf:** là cơ sở của `DataLoaderSentinel` và thí nghiệm `num_workers` = 0 / 2 / 4 trong TN6 (TV3).
- **Hình:** `13_datastalls_fig1_pipeline.png`, `14_datastalls_fig3_fetch_stalls.png` · Nguồn: Mohan et al., PVLDB 2021.

### [9] TBD — training tốn bộ nhớ vì feature map
- **Trích dẫn:** H. Zhu, M. Akrout, B. Zheng, A. Pelegris, A. Phanishayee, B. Schroeder, G. Pekhimenko, "TBD:
  Benchmarking and Analyzing Deep Neural Network Training," *IEEE IISWC*, 2018. arXiv:1803.06905.
- **Ý chính:** khi training, **feature map** (output trung gian của các layer) chiếm phần lớn bộ nhớ GPU: từ 62%
  (Deep Speech 2) tới 89% (ResNet-50), không phải trọng số; bộ nhớ tăng gần tuyến tính theo batch.
- **VisionProf:** giải thích kết quả CH5: bộ nhớ training gấp khoảng 2.7–3.3 lần suy luận vì phải giữ activation.
- **Hình:** `15_tbd_fig9_memory_breakdown.png` · Nguồn: Zhu et al., IISWC 2018, Fig. 9.

## 3. Bảng so sánh (đưa lên slide cuối của phần nghiên cứu)

| | XSP [2] | PRoof [3] | MLPerf Inf. [5] | MLModelScope [6] | AI Tax [7] | Data Stalls [8] | **VisionProf** |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Thời gian từng layer | ✅ | ✅ | — | ✅ | — | — | ✅ |
| Roofline | ✅ | ✅ | — | — | — | — | ✅ |
| Phân phối độ trễ mức Model (P50–P99) | — | — | ✅ | — | — | — | ✅ |
| Pipeline thật (tiền / hậu xử lý) | ✅ | — | — | ✅ | ✅ | — | ✅ |
| Training (forward / backward / nạp dữ liệu) | — | — | — | — | — | ✅ | ✅ |
| Xử lý overhead / kiểm chứng công cụ đo | ✅ | — | — | — | — | — | ✅ (TN2) |
| Kernel GPU, hardware counter | ✅ | ✅ | — | ✅ | — | — | ❌ |
| Quy mô | 65 mô hình | nhiều mô hình và nền tảng | chuẩn công nghiệp | nhiều framework, phân tán | data center thật | cụm máy Microsoft | 8 mô hình, laptop + Colab T4 |

✅ có · — không phải trọng tâm của paper · ❌ VisionProf không làm (hạn chế).
Ô "—" nghĩa là paper không lấy nội dung đó làm trọng tâm, **không** có nghĩa là hoàn toàn không đề cập. Khi bị hỏi,
trả lời theo hướng đó.

## 4. Gợi ý 4 slide cho phần nghiên cứu (TV1, ~4 phút)

1. **"Profiling học sâu: nhìn ở nhiều mức"**: hình `06_mlmodelscope_fig1_L1_L5.png`, đánh dấu 3 mức nhóm đo.
   Một câu: "Mỗi nghiên cứu thường chỉ nhìn một mức."
2. **Mức Layer: đo sâu thì công cụ đo làm sai số đo**: `04_xsp_fig2_leveled_overhead.png` + `01_roofline_fig1.png`.
   Câu chốt: "Vì vậy nhóm kiểm chứng công cụ đo ở TN2."
3. **Ngoài mô hình mới là điểm nghẽn**: `11_aitax_fig6_latency_breakdown.png` + `13_datastalls_fig1_pipeline.png`.
   Câu chốt: "Vì vậy nhóm đo cả mức Giai đoạn (TN6, TN7)."
4. **Bảng so sánh Mục 3 + khoảng trống** → chuyển sang "Kiến trúc 3 mức của VisionProf" (TV2).

Mỗi người dùng lại paper của mức mình khi trình bày kết quả, ví dụ: TV2 nhắc MLPerf khi giải thích vì sao báo P99;
TV3 nhắc AI Tax ở CH6 ("paper đo dưới 60% là AI, nhóm đo được 41% ngoài mô hình") và TBD ở CH5.

## 5. Còn thiếu

- **Hình của PRoof:** ACM chặn tải tự động. Mở <https://dl.acm.org/doi/pdf/10.1145/3673038.3673116> bằng trình duyệt,
  lưu file vào thư mục `docs/research/` với tên `proof.pdf`. **Không commit file PDF**; chỉ cần để đó cho TV1 cắt hình.

## Tài liệu tham khảo (định dạng IEEE)

[1] S. Williams, A. Waterman, and D. Patterson, "Roofline: An insightful visual performance model for multicore architectures," *Commun. ACM*, vol. 52, no. 4, pp. 65–76, 2009.

[2] C. Li, A. Dakkak, J. Xiong, W. Wei, L. Xu, and W. Hwu, "XSP: Across-stack profiling and analysis of machine learning models on GPUs," in *Proc. IEEE IPDPS*, 2020.

[3] S. Wu, H. Yang, X. You, R. Gong, Y. Liu, Z. Luan, and D. Qian, "PRoof: A comprehensive hierarchical profiling framework for deep neural networks with roofline analysis," in *Proc. 53rd Int. Conf. Parallel Processing (ICPP)*, 2024, pp. 822–832.

[4] C. Yang, Y. Wang, S. Farrell, T. Kurth, and S. Williams, "Hierarchical roofline performance analysis for deep learning applications," arXiv:2009.05257, 2020.

[5] V. J. Reddi *et al.*, "MLPerf inference benchmark," in *Proc. ACM/IEEE ISCA*, 2020.

[6] A. Dakkak, C. Li, J. Xiong, and W. Hwu, "MLModelScope: A distributed platform for model evaluation and benchmarking at scale," arXiv:2002.08295, 2020.

[7] D. Richins *et al.*, "Missing the forest for the trees: End-to-end AI application performance in edge data centers," in *Proc. IEEE HPCA*, 2020, pp. 515–528.

[8] J. Mohan, A. Phanishayee, A. Raniwala, and V. Chidambaram, "Analyzing and mitigating data stalls in DNN training," *Proc. VLDB Endow.*, vol. 14, no. 5, pp. 771–784, 2021.

[9] H. Zhu, M. Akrout, B. Zheng, A. Pelegris, A. Phanishayee, B. Schroeder, and G. Pekhimenko, "TBD: Benchmarking and analyzing deep neural network training," in *Proc. IEEE IISWC*, 2018.
