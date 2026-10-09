# TN1 · mlp (bs1_fp32)

## Top-10 layer chậm nhất · cpu_laptop

| rank | layer_name | layer_type | time_ms | pct | activation_mb | n_calls |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | net.2 | Linear | 0.011 | 35.158 | 0.001 | 1 |
| 2 | net.0 | Linear | 0.008 | 26.866 | 0.001 | 1 |
| 3 | net.4 | Linear | 0.006 | 19.071 | 0.000 | 1 |
| 4 | net.1 | ReLU | 0.003 | 10.282 | 0.001 | 1 |
| 5 | net.3 | ReLU | 0.003 | 8.624 | 0.001 | 1 |

## Tỷ trọng theo loại layer · cpu_laptop

| layer_type | time_ms | n_layers | pct |
| --- | --- | --- | --- |
| Linear | 0.024 | 3 | 81.095 |
| ReLU | 0.006 | 2 | 18.905 |

## Top-10 layer chậm nhất · colab_t4

| rank | layer_name | layer_type | time_ms | pct | activation_mb | n_calls |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | net.0 | Linear | 0.007 | 23.263 | 0.001 | 1 |
| 2 | net.2 | Linear | 0.007 | 23.215 | 0.001 | 1 |
| 3 | net.4 | Linear | 0.006 | 19.620 | 0.000 | 1 |
| 4 | net.3 | ReLU | 0.005 | 17.079 | 0.001 | 1 |
| 5 | net.1 | ReLU | 0.005 | 16.824 | 0.001 | 1 |

## Tỷ trọng theo loại layer · colab_t4

| layer_type | time_ms | n_layers | pct |
| --- | --- | --- | --- |
| Linear | 0.021 | 3 | 66.097 |
| ReLU | 0.011 | 2 | 33.903 |

## CPU (cpu_laptop) và GPU (colab_t4) · top-10

| layer_name | layer_type | time_ms_cpu | pct_cpu | rank_cpu | time_ms_gpu | pct_gpu | rank_gpu | speedup | rank_change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| net.2 | Linear | 0.011 | 35.158 | 1 | 0.007 | 23.215 | 2 | 1.459 | -1 |
| net.0 | Linear | 0.008 | 26.866 | 2 | 0.007 | 23.263 | 1 | 1.113 | 1 |
| net.4 | Linear | 0.006 | 19.071 | 3 | 0.006 | 19.620 | 3 | 0.936 | 0 |
| net.1 | ReLU | 0.003 | 10.282 | 4 | 0.005 | 16.824 | 5 | 0.589 | -1 |
| net.3 | ReLU | 0.003 | 8.624 | 5 | 0.005 | 17.079 | 4 | 0.486 | 1 |

## CPU và GPU · tỷ trọng theo loại layer

| layer_type | time_ms_cpu | n_layers | pct_cpu | time_ms_gpu | pct_gpu | pct_change |
| --- | --- | --- | --- | --- | --- | --- |
| Linear | 0.024 | 3 | 81.095 | 0.021 | 66.097 | -14.998 |
| ReLU | 0.006 | 2 | 18.905 | 0.011 | 33.903 | 14.998 |
