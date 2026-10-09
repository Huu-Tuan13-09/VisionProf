# TN1 · mobilenet_v3_large (bs1_fp32)

## Top-10 layer chậm nhất · cpu_laptop

| rank | layer_name | layer_type | time_ms | pct | activation_mb | n_calls |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | features.2.block.0.0 | Conv2d | 0.283 | 3.339 | 3.062 | 1 |
| 2 | classifier.0 | Linear | 0.215 | 2.535 | 0.005 | 1 |
| 3 | classifier.3 | Linear | 0.195 | 2.300 | 0.004 | 1 |
| 4 | features.3.block.1.0 | Conv2d | 0.190 | 2.238 | 0.861 | 1 |
| 5 | features.3.block.0.0 | Conv2d | 0.176 | 2.080 | 0.861 | 1 |
| 6 | features.0.0 | Conv2d | 0.169 | 1.995 | 0.766 | 1 |
| 7 | features.4.block.0.0 | Conv2d | 0.165 | 1.947 | 0.861 | 1 |
| 8 | features.12.block.3.0 | Conv2d | 0.162 | 1.912 | 0.084 | 1 |
| 9 | features.3.block.2.0 | Conv2d | 0.162 | 1.910 | 0.287 | 1 |
| 10 | features.13.block.0.0 | Conv2d | 0.157 | 1.854 | 0.502 | 1 |

## Tỷ trọng theo loại layer · cpu_laptop

| layer_type | time_ms | n_layers | pct |
| --- | --- | --- | --- |
| Conv2d | 6.185 | 62 | 72.895 |
| BatchNorm2d | 1.227 | 46 | 14.463 |
| Linear | 0.410 | 2 | 4.834 |
| AdaptiveAvgPool2d | 0.252 | 9 | 2.966 |
| Hardswish | 0.200 | 21 | 2.351 |
| ReLU | 0.169 | 19 | 1.992 |
| Hardsigmoid | 0.042 | 8 | 0.498 |

## Top-10 layer chậm nhất · colab_t4

| rank | layer_name | layer_type | time_ms | pct | activation_mb | n_calls |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | features.14.block.2.fc1 | Conv2d | 0.086 | 3.063 | 0.001 | 1 |
| 2 | features.15.block.2.fc1 | Conv2d | 0.085 | 3.041 | 0.001 | 1 |
| 3 | features.14.block.2.fc2 | Conv2d | 0.074 | 2.631 | 0.004 | 1 |
| 4 | features.15.block.2.fc2 | Conv2d | 0.074 | 2.626 | 0.004 | 1 |
| 5 | classifier.0 | Linear | 0.069 | 2.464 | 0.005 | 1 |
| 6 | features.15.block.3.0 | Conv2d | 0.067 | 2.403 | 0.030 | 1 |
| 7 | features.14.block.3.0 | Conv2d | 0.067 | 2.402 | 0.030 | 1 |
| 8 | features.15.block.0.0 | Conv2d | 0.067 | 2.373 | 0.179 | 1 |
| 9 | features.16.0 | Conv2d | 0.066 | 2.339 | 0.179 | 1 |
| 10 | features.15.block.3.1 | BatchNorm2d | 0.063 | 2.257 | 0.030 | 1 |

## Tỷ trọng theo loại layer · colab_t4

| layer_type | time_ms | n_layers | pct |
| --- | --- | --- | --- |
| Conv2d | 1.328 | 62 | 47.277 |
| BatchNorm2d | 0.705 | 46 | 25.122 |
| Hardswish | 0.241 | 21 | 8.585 |
| AdaptiveAvgPool2d | 0.167 | 9 | 5.942 |
| ReLU | 0.150 | 19 | 5.340 |
| Linear | 0.130 | 2 | 4.631 |
| Hardsigmoid | 0.087 | 8 | 3.102 |

## CPU (cpu_laptop) và GPU (colab_t4) · top-10

| layer_name | layer_type | time_ms_cpu | pct_cpu | rank_cpu | time_ms_gpu | pct_gpu | rank_gpu | speedup | rank_change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| features.2.block.0.0 | Conv2d | 0.283 | 3.339 | 1 | 0.020 | 0.721 | 37 | 13.997 | -36 |
| classifier.0 | Linear | 0.215 | 2.535 | 2 | 0.069 | 2.464 | 5 | 3.108 | -3 |
| classifier.3 | Linear | 0.195 | 2.300 | 3 | 0.061 | 2.167 | 13 | 3.207 | -10 |
| features.3.block.1.0 | Conv2d | 0.190 | 2.238 | 4 | 0.016 | 0.557 | 43 | 12.146 | -39 |
| features.3.block.0.0 | Conv2d | 0.176 | 2.080 | 5 | 0.013 | 0.473 | 52 | 13.291 | -47 |
| features.0.0 | Conv2d | 0.169 | 1.995 | 6 | 0.016 | 0.556 | 44 | 10.853 | -38 |
| features.4.block.0.0 | Conv2d | 0.165 | 1.947 | 7 | 0.012 | 0.441 | 53 | 13.344 | -46 |
| features.12.block.3.0 | Conv2d | 0.162 | 1.912 | 8 | 0.018 | 0.656 | 40 | 8.801 | -32 |
| features.3.block.2.0 | Conv2d | 0.162 | 1.910 | 9 | 0.012 | 0.411 | 56 | 14.030 | -47 |
| features.13.block.0.0 | Conv2d | 0.157 | 1.854 | 10 | 0.020 | 0.708 | 39 | 7.913 | -29 |
| features.15.block.3.0 | Conv2d | 0.135 | 1.592 | 15 | 0.067 | 2.403 | 6 | 2.002 | 9 |
| features.14.block.3.0 | Conv2d | 0.133 | 1.572 | 17 | 0.067 | 2.402 | 7 | 1.978 | 10 |
| features.16.0 | Conv2d | 0.091 | 1.075 | 42 | 0.066 | 2.339 | 9 | 1.389 | 33 |
| features.15.block.0.0 | Conv2d | 0.090 | 1.064 | 44 | 0.067 | 2.373 | 8 | 1.355 | 36 |
| features.14.block.2.fc2 | Conv2d | 0.086 | 1.014 | 45 | 0.074 | 2.631 | 3 | 1.164 | 42 |
| features.15.block.2.fc2 | Conv2d | 0.082 | 0.971 | 46 | 0.074 | 2.626 | 4 | 1.117 | 42 |
| features.14.block.2.fc1 | Conv2d | 0.055 | 0.643 | 48 | 0.086 | 3.063 | 1 | 0.634 | 47 |
| features.15.block.2.fc1 | Conv2d | 0.052 | 0.610 | 50 | 0.085 | 3.041 | 2 | 0.606 | 48 |
| features.15.block.3.1 | BatchNorm2d | 0.024 | 0.279 | 99 | 0.063 | 2.257 | 10 | 0.374 | 89 |

## CPU và GPU · tỷ trọng theo loại layer

| layer_type | time_ms_cpu | n_layers | pct_cpu | time_ms_gpu | pct_gpu | pct_change |
| --- | --- | --- | --- | --- | --- | --- |
| Conv2d | 6.185 | 62 | 72.895 | 1.328 | 47.277 | -25.618 |
| BatchNorm2d | 1.227 | 46 | 14.463 | 0.705 | 25.122 | 10.660 |
| Linear | 0.410 | 2 | 4.834 | 0.130 | 4.631 | -0.204 |
| AdaptiveAvgPool2d | 0.252 | 9 | 2.966 | 0.167 | 5.942 | 2.976 |
| Hardswish | 0.200 | 21 | 2.351 | 0.241 | 8.585 | 6.234 |
| ReLU | 0.169 | 19 | 1.992 | 0.150 | 5.340 | 3.348 |
| Hardsigmoid | 0.042 | 8 | 0.498 | 0.087 | 3.102 | 2.604 |
