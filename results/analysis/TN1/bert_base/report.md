# TN1 · bert_base (bs1_fp32)

## Top-10 layer chậm nhất · cpu_laptop

| rank | layer_name | layer_type | time_ms | pct | activation_mb | n_calls |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | encoder.layer.6.intermediate.dense | Linear | 1.575 | 2.797 | 1.500 | 1 |
| 2 | encoder.layer.4.intermediate.dense | Linear | 1.546 | 2.745 | 1.500 | 1 |
| 3 | encoder.layer.5.intermediate.dense | Linear | 1.540 | 2.735 | 1.500 | 1 |
| 4 | encoder.layer.3.intermediate.dense | Linear | 1.539 | 2.732 | 1.500 | 1 |
| 5 | encoder.layer.8.intermediate.dense | Linear | 1.531 | 2.719 | 1.500 | 1 |
| 6 | encoder.layer.1.intermediate.dense | Linear | 1.530 | 2.717 | 1.500 | 1 |
| 7 | encoder.layer.7.intermediate.dense | Linear | 1.529 | 2.715 | 1.500 | 1 |
| 8 | encoder.layer.10.intermediate.dense | Linear | 1.528 | 2.713 | 1.500 | 1 |
| 9 | encoder.layer.11.intermediate.dense | Linear | 1.520 | 2.698 | 1.500 | 1 |
| 10 | encoder.layer.0.intermediate.dense | Linear | 1.518 | 2.695 | 1.500 | 1 |

## Tỷ trọng theo loại layer · cpu_laptop

| layer_type | time_ms | n_layers | pct |
| --- | --- | --- | --- |
| Linear | 53.776 | 73 | 95.471 |
| GELUActivation | 1.564 | 12 | 2.777 |
| LayerNorm | 0.876 | 25 | 1.555 |
| Embedding | 0.094 | 3 | 0.167 |
| Tanh | 0.017 | 1 | 0.031 |

## Top-10 layer chậm nhất · colab_t4

| rank | layer_name | layer_type | time_ms | pct | activation_mb | n_calls |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | encoder.layer.10.output.dense | Linear | 0.155 | 2.484 | 0.375 | 1 |
| 2 | encoder.layer.9.output.dense | Linear | 0.155 | 2.484 | 0.375 | 1 |
| 3 | encoder.layer.7.output.dense | Linear | 0.155 | 2.481 | 0.375 | 1 |
| 4 | encoder.layer.6.output.dense | Linear | 0.155 | 2.480 | 0.375 | 1 |
| 5 | encoder.layer.3.output.dense | Linear | 0.155 | 2.477 | 0.375 | 1 |
| 6 | encoder.layer.5.output.dense | Linear | 0.154 | 2.468 | 0.375 | 1 |
| 7 | encoder.layer.0.output.dense | Linear | 0.154 | 2.467 | 0.375 | 1 |
| 8 | encoder.layer.8.output.dense | Linear | 0.154 | 2.467 | 0.375 | 1 |
| 9 | encoder.layer.1.output.dense | Linear | 0.154 | 2.465 | 0.375 | 1 |
| 10 | encoder.layer.4.output.dense | Linear | 0.153 | 2.459 | 0.375 | 1 |

## Tỷ trọng theo loại layer · colab_t4

| layer_type | time_ms | n_layers | pct |
| --- | --- | --- | --- |
| Linear | 5.938 | 73 | 95.146 |
| LayerNorm | 0.200 | 25 | 3.198 |
| GELUActivation | 0.082 | 12 | 1.318 |
| Embedding | 0.017 | 3 | 0.269 |
| Tanh | 0.004 | 1 | 0.069 |

## CPU (cpu_laptop) và GPU (colab_t4) · top-10

| layer_name | layer_type | time_ms_cpu | pct_cpu | rank_cpu | time_ms_gpu | pct_gpu | rank_gpu | speedup | rank_change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| encoder.layer.6.intermediate.dense | Linear | 1.575 | 2.797 | 1 | 0.126 | 2.021 | 14 | 12.491 | -13 |
| encoder.layer.4.intermediate.dense | Linear | 1.546 | 2.745 | 2 | 0.123 | 1.976 | 23 | 12.539 | -21 |
| encoder.layer.5.intermediate.dense | Linear | 1.540 | 2.735 | 3 | 0.125 | 1.997 | 20 | 12.359 | -17 |
| encoder.layer.3.intermediate.dense | Linear | 1.539 | 2.732 | 4 | 0.126 | 2.024 | 13 | 12.183 | -9 |
| encoder.layer.8.intermediate.dense | Linear | 1.531 | 2.719 | 5 | 0.125 | 2.000 | 19 | 12.270 | -14 |
| encoder.layer.1.intermediate.dense | Linear | 1.530 | 2.717 | 6 | 0.125 | 2.002 | 15 | 12.249 | -9 |
| encoder.layer.7.intermediate.dense | Linear | 1.529 | 2.715 | 7 | 0.123 | 1.971 | 24 | 12.432 | -17 |
| encoder.layer.10.intermediate.dense | Linear | 1.528 | 2.713 | 8 | 0.125 | 2.002 | 16 | 12.230 | -8 |
| encoder.layer.11.intermediate.dense | Linear | 1.520 | 2.698 | 9 | 0.125 | 2.002 | 18 | 12.163 | -9 |
| encoder.layer.0.intermediate.dense | Linear | 1.518 | 2.695 | 10 | 0.124 | 1.986 | 22 | 12.248 | -12 |
| encoder.layer.8.output.dense | Linear | 1.477 | 2.622 | 13 | 0.154 | 2.467 | 8 | 9.592 | 5 |
| encoder.layer.0.output.dense | Linear | 1.467 | 2.605 | 14 | 0.154 | 2.467 | 7 | 9.529 | 7 |
| encoder.layer.6.output.dense | Linear | 1.466 | 2.602 | 15 | 0.155 | 2.480 | 4 | 9.471 | 11 |
| encoder.layer.7.output.dense | Linear | 1.465 | 2.602 | 16 | 0.155 | 2.481 | 3 | 9.463 | 13 |
| encoder.layer.5.output.dense | Linear | 1.464 | 2.599 | 17 | 0.154 | 2.468 | 6 | 9.502 | 11 |
| encoder.layer.10.output.dense | Linear | 1.463 | 2.597 | 18 | 0.155 | 2.484 | 1 | 9.436 | 17 |
| encoder.layer.3.output.dense | Linear | 1.460 | 2.592 | 19 | 0.155 | 2.477 | 5 | 9.445 | 14 |
| encoder.layer.4.output.dense | Linear | 1.459 | 2.591 | 20 | 0.153 | 2.459 | 10 | 9.509 | 10 |
| encoder.layer.9.output.dense | Linear | 1.452 | 2.579 | 22 | 0.155 | 2.484 | 2 | 9.371 | 20 |
| encoder.layer.1.output.dense | Linear | 1.450 | 2.573 | 24 | 0.154 | 2.465 | 9 | 9.422 | 15 |

## CPU và GPU · tỷ trọng theo loại layer

| layer_type | time_ms_cpu | n_layers | pct_cpu | time_ms_gpu | pct_gpu | pct_change |
| --- | --- | --- | --- | --- | --- | --- |
| Linear | 53.776 | 73 | 95.471 | 5.938 | 95.146 | -0.326 |
| GELUActivation | 1.564 | 12 | 2.777 | 0.082 | 1.318 | -1.459 |
| LayerNorm | 0.876 | 25 | 1.555 | 0.200 | 3.198 | 1.644 |
| Embedding | 0.094 | 3 | 0.167 | 0.017 | 0.269 | 0.102 |
| Tanh | 0.017 | 1 | 0.031 | 0.004 | 0.069 | 0.038 |
