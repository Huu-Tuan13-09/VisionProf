# TN1 · yolov8n (bs1_fp32)

## Top-10 layer chậm nhất · cpu_laptop

| rank | layer_name | layer_type | time_ms | pct | activation_mb | n_calls |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | model.0.act | SiLU | 2.297 | 7.506 | 52.979 | 57 |
| 2 | model.9.m | MaxPool2d | 1.684 | 5.504 | 0.586 | 3 |
| 3 | model.22.cv3.0.1.conv | Conv2d | 1.172 | 3.829 | 1.953 | 1 |
| 4 | model.22.cv3.0.0.conv | Conv2d | 1.158 | 3.783 | 1.953 | 1 |
| 5 | model.22.cv2.0.0.conv | Conv2d | 0.992 | 3.240 | 1.562 | 1 |
| 6 | model.22.cv2.0.1.conv | Conv2d | 0.890 | 2.908 | 1.562 | 1 |
| 7 | model.1.conv | Conv2d | 0.743 | 2.427 | 3.125 | 1 |
| 8 | model.15.cv1.conv | Conv2d | 0.649 | 2.120 | 1.562 | 1 |
| 9 | model.3.conv | Conv2d | 0.642 | 2.097 | 1.562 | 1 |
| 10 | model.22.cv3.1.0.conv | Conv2d | 0.608 | 1.987 | 0.488 | 1 |

## Tỷ trọng theo loại layer · cpu_laptop

| layer_type | time_ms | n_layers | pct |
| --- | --- | --- | --- |
| Conv2d | 23.908 | 64 | 78.117 |
| SiLU | 2.297 | 1 | 7.506 |
| BatchNorm2d | 1.944 | 57 | 6.352 |
| MaxPool2d | 1.684 | 1 | 5.504 |
| Upsample | 0.463 | 2 | 1.514 |
| Concat | 0.308 | 4 | 1.008 |

## Top-10 layer chậm nhất · colab_t4

| rank | layer_name | layer_type | time_ms | pct | activation_mb | n_calls |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | model.0.act | SiLU | 0.401 | 7.248 | 52.979 | 57 |
| 2 | model.22.cv3.0.1.conv | Conv2d | 0.287 | 5.186 | 1.953 | 1 |
| 3 | model.22.cv3.0.0.conv | Conv2d | 0.240 | 4.333 | 1.953 | 1 |
| 4 | model.22.cv3.1.0.conv | Conv2d | 0.190 | 3.440 | 0.488 | 1 |
| 5 | model.22.cv2.1.0.conv | Conv2d | 0.186 | 3.370 | 0.391 | 1 |
| 6 | model.22.cv2.0.0.conv | Conv2d | 0.170 | 3.074 | 1.562 | 1 |
| 7 | model.22.cv2.0.1.conv | Conv2d | 0.169 | 3.064 | 1.562 | 1 |
| 8 | model.22.cv2.2.0.conv | Conv2d | 0.154 | 2.778 | 0.098 | 1 |
| 9 | model.22.cv3.2.0.conv | Conv2d | 0.147 | 2.650 | 0.122 | 1 |
| 10 | model.7.conv | Conv2d | 0.115 | 2.083 | 0.391 | 1 |

## Tỷ trọng theo loại layer · colab_t4

| layer_type | time_ms | n_layers | pct |
| --- | --- | --- | --- |
| Conv2d | 4.351 | 64 | 78.690 |
| BatchNorm2d | 0.642 | 57 | 11.601 |
| SiLU | 0.401 | 1 | 7.248 |
| Concat | 0.073 | 4 | 1.328 |
| MaxPool2d | 0.036 | 1 | 0.652 |
| Upsample | 0.027 | 2 | 0.481 |

## CPU (cpu_laptop) và GPU (colab_t4) · top-10

| layer_name | layer_type | time_ms_cpu | pct_cpu | rank_cpu | time_ms_gpu | pct_gpu | rank_gpu | speedup | rank_change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model.0.act | SiLU | 2.297 | 7.506 | 1 | 0.401 | 7.248 | 1 | 5.732 | 0 |
| model.9.m | MaxPool2d | 1.684 | 5.504 | 2 | 0.036 | 0.652 | 52 | 46.724 | -50 |
| model.22.cv3.0.1.conv | Conv2d | 1.172 | 3.829 | 3 | 0.287 | 5.186 | 2 | 4.087 | 1 |
| model.22.cv3.0.0.conv | Conv2d | 1.158 | 3.783 | 4 | 0.240 | 4.333 | 3 | 4.832 | 1 |
| model.22.cv2.0.0.conv | Conv2d | 0.992 | 3.240 | 5 | 0.170 | 3.074 | 6 | 5.833 | -1 |
| model.22.cv2.0.1.conv | Conv2d | 0.890 | 2.908 | 6 | 0.169 | 3.064 | 7 | 5.253 | -1 |
| model.1.conv | Conv2d | 0.743 | 2.427 | 7 | 0.107 | 1.943 | 12 | 6.914 | -5 |
| model.15.cv1.conv | Conv2d | 0.649 | 2.120 | 8 | 0.040 | 0.721 | 45 | 16.268 | -37 |
| model.3.conv | Conv2d | 0.642 | 2.097 | 9 | 0.090 | 1.630 | 19 | 7.123 | -10 |
| model.22.cv3.1.0.conv | Conv2d | 0.608 | 1.987 | 10 | 0.190 | 3.440 | 4 | 3.197 | 6 |
| model.7.conv | Conv2d | 0.544 | 1.776 | 13 | 0.115 | 2.083 | 10 | 4.719 | 3 |
| model.22.cv2.1.0.conv | Conv2d | 0.520 | 1.699 | 14 | 0.186 | 3.370 | 5 | 2.791 | 9 |
| model.22.cv3.2.0.conv | Conv2d | 0.355 | 1.161 | 21 | 0.147 | 2.650 | 9 | 2.425 | 12 |
| model.22.cv2.2.0.conv | Conv2d | 0.310 | 1.014 | 34 | 0.154 | 2.778 | 8 | 2.021 | 26 |

## CPU và GPU · tỷ trọng theo loại layer

| layer_type | time_ms_cpu | n_layers | pct_cpu | time_ms_gpu | pct_gpu | pct_change |
| --- | --- | --- | --- | --- | --- | --- |
| Conv2d | 23.908 | 64 | 78.117 | 4.351 | 78.690 | 0.573 |
| SiLU | 2.297 | 1 | 7.506 | 0.401 | 7.248 | -0.258 |
| BatchNorm2d | 1.944 | 57 | 6.352 | 0.642 | 11.601 | 5.249 |
| MaxPool2d | 1.684 | 1 | 5.504 | 0.036 | 0.652 | -4.852 |
| Upsample | 0.463 | 2 | 1.514 | 0.027 | 0.481 | -1.033 |
| Concat | 0.308 | 4 | 1.008 | 0.073 | 1.328 | 0.321 |
