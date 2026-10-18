# TN1 · resnet50 (bs1_fp32)

## Top-10 layer chậm nhất · cpu_laptop

| rank | layer_name | layer_type | time_ms | pct | activation_mb | n_calls |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | maxpool | MaxPool2d | 1.671 | 5.550 | 0.766 | 1 |
| 2 | layer4.0.conv2 | Conv2d | 1.170 | 3.884 | 0.096 | 1 |
| 3 | layer4.0.downsample.0 | Conv2d | 1.119 | 3.717 | 0.383 | 1 |
| 4 | layer4.1.conv2 | Conv2d | 1.104 | 3.666 | 0.096 | 1 |
| 5 | layer4.2.conv2 | Conv2d | 1.067 | 3.543 | 0.096 | 1 |
| 6 | conv1 | Conv2d | 0.660 | 2.190 | 3.062 | 1 |
| 7 | layer3.0.downsample.0 | Conv2d | 0.633 | 2.102 | 0.766 | 1 |
| 8 | layer3.0.conv2 | Conv2d | 0.607 | 2.014 | 0.191 | 1 |
| 9 | layer2.0.downsample.0 | Conv2d | 0.605 | 2.008 | 1.531 | 1 |
| 10 | layer2.0.conv2 | Conv2d | 0.580 | 1.925 | 0.383 | 1 |

## Tỷ trọng theo loại layer · cpu_laptop

| layer_type | time_ms | n_layers | pct |
| --- | --- | --- | --- |
| Conv2d | 25.546 | 53 | 84.817 |
| BatchNorm2d | 1.983 | 53 | 6.584 |
| MaxPool2d | 1.671 | 1 | 5.550 |
| ReLU | 0.538 | 17 | 1.786 |
| Linear | 0.310 | 1 | 1.029 |
| AdaptiveAvgPool2d | 0.071 | 1 | 0.235 |

## Top-10 layer chậm nhất · colab_t4

| rank | layer_name | layer_type | time_ms | pct | activation_mb | n_calls |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | layer4.0.conv2 | Conv2d | 0.308 | 5.975 | 0.096 | 1 |
| 2 | layer4.2.conv2 | Conv2d | 0.299 | 5.804 | 0.096 | 1 |
| 3 | layer4.1.conv2 | Conv2d | 0.299 | 5.798 | 0.096 | 1 |
| 4 | layer3.0.conv2 | Conv2d | 0.182 | 3.538 | 0.191 | 1 |
| 5 | layer3.3.conv2 | Conv2d | 0.179 | 3.477 | 0.191 | 1 |
| 6 | layer3.5.conv2 | Conv2d | 0.179 | 3.472 | 0.191 | 1 |
| 7 | layer3.2.conv2 | Conv2d | 0.179 | 3.472 | 0.191 | 1 |
| 8 | layer3.1.conv2 | Conv2d | 0.179 | 3.469 | 0.191 | 1 |
| 9 | layer3.4.conv2 | Conv2d | 0.179 | 3.467 | 0.191 | 1 |
| 10 | layer4.0.downsample.0 | Conv2d | 0.117 | 2.272 | 0.383 | 1 |

## Tỷ trọng theo loại layer · colab_t4

| layer_type | time_ms | n_layers | pct |
| --- | --- | --- | --- |
| Conv2d | 4.283 | 53 | 83.117 |
| BatchNorm2d | 0.547 | 53 | 10.620 |
| ReLU | 0.253 | 17 | 4.901 |
| Linear | 0.038 | 1 | 0.737 |
| MaxPool2d | 0.023 | 1 | 0.437 |
| AdaptiveAvgPool2d | 0.010 | 1 | 0.188 |

## CPU (cpu_laptop) và GPU (colab_t4) · top-10

| layer_name | layer_type | time_ms_cpu | pct_cpu | rank_cpu | time_ms_gpu | pct_gpu | rank_gpu | speedup | rank_change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| maxpool | MaxPool2d | 1.671 | 5.550 | 1 | 0.023 | 0.437 | 59 | 74.190 | -58 |
| layer4.0.conv2 | Conv2d | 1.170 | 3.884 | 2 | 0.308 | 5.975 | 1 | 3.800 | 1 |
| layer4.0.downsample.0 | Conv2d | 1.119 | 3.717 | 3 | 0.117 | 2.272 | 10 | 9.563 | -7 |
| layer4.1.conv2 | Conv2d | 1.104 | 3.666 | 4 | 0.299 | 5.798 | 3 | 3.696 | 1 |
| layer4.2.conv2 | Conv2d | 1.067 | 3.543 | 5 | 0.299 | 5.804 | 2 | 3.568 | 3 |
| conv1 | Conv2d | 0.660 | 2.190 | 6 | 0.090 | 1.749 | 16 | 7.319 | -10 |
| layer3.0.downsample.0 | Conv2d | 0.633 | 2.102 | 7 | 0.105 | 2.033 | 13 | 6.043 | -6 |
| layer3.0.conv2 | Conv2d | 0.607 | 2.014 | 8 | 0.182 | 3.538 | 4 | 3.328 | 4 |
| layer2.0.downsample.0 | Conv2d | 0.605 | 2.008 | 9 | 0.083 | 1.620 | 20 | 7.246 | -11 |
| layer2.0.conv2 | Conv2d | 0.580 | 1.925 | 10 | 0.112 | 2.179 | 11 | 5.163 | -1 |
| layer3.1.conv2 | Conv2d | 0.570 | 1.893 | 11 | 0.179 | 3.469 | 8 | 3.189 | 3 |
| layer3.2.conv2 | Conv2d | 0.558 | 1.853 | 12 | 0.179 | 3.472 | 7 | 3.121 | 5 |
| layer3.5.conv2 | Conv2d | 0.554 | 1.839 | 13 | 0.179 | 3.472 | 6 | 3.095 | 7 |
| layer3.4.conv2 | Conv2d | 0.545 | 1.811 | 18 | 0.179 | 3.467 | 9 | 3.053 | 9 |
| layer3.3.conv2 | Conv2d | 0.538 | 1.786 | 21 | 0.179 | 3.477 | 5 | 3.002 | 16 |

## CPU và GPU · tỷ trọng theo loại layer

| layer_type | time_ms_cpu | n_layers | pct_cpu | time_ms_gpu | pct_gpu | pct_change |
| --- | --- | --- | --- | --- | --- | --- |
| Conv2d | 25.546 | 53 | 84.817 | 4.283 | 83.117 | -1.699 |
| BatchNorm2d | 1.983 | 53 | 6.584 | 0.547 | 10.620 | 4.036 |
| MaxPool2d | 1.671 | 1 | 5.550 | 0.023 | 0.437 | -5.112 |
| ReLU | 0.538 | 17 | 1.786 | 0.253 | 4.901 | 3.116 |
| Linear | 0.310 | 1 | 1.029 | 0.038 | 0.737 | -0.293 |
| AdaptiveAvgPool2d | 0.071 | 1 | 0.235 | 0.010 | 0.188 | -0.047 |
