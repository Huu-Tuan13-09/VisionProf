"""Nhãn và thông báo tiếng Việt cho kết quả chẩn đoán, giữ nguyên trị số."""

import re

RULE_LABELS = {
    "RA-01": "Nguy cơ hết VRAM", "RA-02": "Phân mảnh bộ nhớ GPU",
    "RA-03": "Điểm nóng activation", "RA-04": "Biến động Process RSS",
    "RA-05": "Truyền dữ liệu CPU–GPU", "RB-01": "Mức sử dụng GPU",
    "RB-02": "Độ ổn định độ trễ", "RB-03": "Tỷ lệ backward / forward",
    "RB-04": "Hiệu quả tính toán · ms/GFLOP", "RB-05": "Thời gian optimizer",
    "RC-01": "Thời gian chờ I/O", "RC-02": "Biến động DataLoader",
    "RC-03": "GPU chờ dữ liệu", "RC-04": "Hiệu quả batch size",
    "RD-01": "Phân bố tham số", "RD-02": "Layer dưới 1 µs",
    "RD-03": "Tỷ lệ bộ nhớ / tính toán", "RD-04": "Nút thắt tại layer",
    "RD-05": "Độ trễ và FLOPs bất thường",
}
STATUS_LABELS = {"PASS": "Đạt", "WARN": "Cảnh báo", "FAIL": "Có vấn đề", "SKIPPED": "Thiếu dữ liệu"}

_MESSAGES = {
    "VRAM capacity unknown.": "Chưa biết dung lượng VRAM.",
    "No reserved GPU memory tracked.": "Chưa có số đo bộ nhớ GPU dành trước.",
    "Requires layer profiler (layers.csv).": "Cần dữ liệu layer (layers.csv).",
    "Requires phase profiler (phases.csv).": "Cần dữ liệu giai đoạn (phases.csv).",
    "Total activation memory is minimal.": "Tổng bộ nhớ activation rất nhỏ.",
    "Insufficient samples to establish trend.": "Chưa đủ mẫu để xác định xu hướng.",
    "No transfer phase data available.": "Chưa có dữ liệu thời gian truyền CPU–GPU.",
    "No GPU metrics sampled.": "Chưa có mẫu đo GPU.",
    "Invalid median latency.": "Giá trị độ trễ P50 không hợp lệ.",
    "Requires training phase metrics.": "Cần dữ liệu giai đoạn training.",
    "No optimizer phase metrics available.": "Chưa có dữ liệu giai đoạn optimizer.",
    "No DataLoader timing collected.": "Chưa có số đo thời gian DataLoader.",
    "Too few I/O samples.": "Chưa đủ mẫu I/O.",
    "Data retrieval instantaneous.": "Thời gian lấy dữ liệu xấp xỉ 0.",
    "Phase metrics unavailable.": "Chưa có dữ liệu giai đoạn.",
    "GPU active compute dominates I/O wait.": "Thời gian GPU tính toán lớn hơn thời gian chờ I/O.",
    "Batch size 1 may suffer from launch overhead; consider batch scaling.":
        "Batch size 1 có thể bị ảnh hưởng bởi chi phí gửi kernel; cân nhắc tăng batch size.",
    "No parameters recorded.": "Không ghi nhận tham số.",
    "Insufficient layer data.": "Chưa đủ dữ liệu layer.",
    "Model latency negligible.": "Độ trễ mô hình rất nhỏ.",
    "No anomalous low-FLOP bottlenecks detected.": "Không phát hiện nút thắt bất thường ở layer có ít FLOPs.",
}
_PATTERNS = [
    (r"Peak VRAM utilizes (.+) of memory\.", r"Peak VRAM chiếm \1 bộ nhớ."),
    (r"Peak VRAM is safe \((.+)\)\.", r"Peak VRAM trong mức an toàn (\1)."),
    (r"Unallocated reserve is (.+)\.", r"Bộ nhớ dành trước chưa được cấp phát: \1."),
    (r"Fragmentation is low \((.+)\)\.", r"Phân mảnh thấp (\1)."),
    (r"Single layer occupies (.+) of activations\.", r"Một layer chiếm \1 bộ nhớ activation."),
    (r"Top layer occupies (.+)\.", r"Layer lớn nhất chiếm \1 bộ nhớ activation."),
    (r"RSS grew by (.+) MB\.", r"Process RSS tăng \1 MB."),
    (r"RSS variation is within margin \((.+) MB\)\.", r"Biến động Process RSS trong giới hạn (\1 MB)."),
    (r"Transfers consume (.+) iteration time\.", r"Truyền dữ liệu chiếm \1 thời gian iteration."),
    (r"Transfers consume (.+)\.", r"Truyền dữ liệu chiếm \1."),
    (r"Mean GPU usage is only (.+)\.", r"Mức sử dụng GPU trung bình chỉ đạt \1."),
    (r"Mean GPU utilization is (.+)\.", r"Mức sử dụng GPU trung bình: \1."),
    (r"P99/P50 ratio is (.+)\.", r"Tỷ lệ P99/P50: \1."),
    (r"Backward takes (.+)x forward duration\.", r"Backward tốn \1× thời gian forward."),
    (r"Backward/Forward ratio is (.+)x\.", r"Tỷ lệ backward / forward: \1×."),
    (r"Consuming (.+) ms/GFLOP\.", r"Chi phí tính toán: \1 ms/GFLOP."),
    (r"Efficient at (.+) ms/GFLOP\.", r"Tính toán hiệu quả: \1 ms/GFLOP."),
    (r"Optimizer consumes (.+) of step\.", r"Optimizer chiếm \1 thời gian bước training."),
    (r"Optimizer consumes (.+)\.", r"Optimizer chiếm \1."),
    (r"I/O takes (.+) of step time\.", r"I/O chiếm \1 thời gian bước training."),
    (r"I/O Coefficient of Variation: (.+)\.", r"Hệ số biến thiên I/O: \1."),
    (r"GPU idle waiting for data \((.+) ms longer than compute\)\.",
     r"GPU chờ dữ liệu (lâu hơn thời gian tính toán \1 ms)."),
    (r"Batch (.+) throughput is satisfactory\.", r"Thông lượng với batch \1 đạt yêu cầu."),
    (r"Layer '(.+)' holds (.+) of parameters\.", r"Layer '\1' chứa \2 tham số."),
    (r"Top layer holds (.+) of parameters\.", r"Layer lớn nhất chứa \1 tham số."),
    (r"(.+) layers \((.+)\) run in <1 microsecond; fusion recommended\.",
     r"\1 layer (\2) chạy dưới 1 µs; nên áp dụng fusion."),
    (r"(.+) sub-microsecond layers detected\.", r"Phát hiện \1 layer chạy dưới 1 µs."),
    (r"Ratio is (.+) MB/GFLOP \(Memory-bound\)\.", r"Tỷ lệ: \1 MB/GFLOP (memory-bound)."),
    (r"Ratio is (.+) MB/GFLOP\.", r"Tỷ lệ: \1 MB/GFLOP."),
    (r"Layer '(.+)' takes (.+) of forward time\.", r"Layer '\1' chiếm \2 thời gian forward."),
    (r"Top layer takes (.+)\.", r"Layer chậm nhất chiếm \1."),
    (r"Outliers with high latency but low FLOPs: (.+)", r"Layer có độ trễ cao nhưng ít FLOPs: \1"),
]


def translate_message(message: str) -> str:
    """Dịch thông báo của engine, giữ nguyên giá trị và tên layer."""
    if message in _MESSAGES:
        return _MESSAGES[message]
    for pattern, replacement in _PATTERNS:
        if re.fullmatch(pattern, message):
            return re.sub(pattern, replacement, message)
    return message
