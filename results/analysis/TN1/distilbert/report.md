# TN1 · distilbert (bs1_fp32)

## Top-10 layer chậm nhất · cpu_laptop

| rank | layer_name | layer_type | time_ms | pct | activation_mb | n_calls |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | transformer.layer.5.ffn.lin1 | Linear | 1.559 | 5.530 | 1.500 | 1 |
| 2 | transformer.layer.4.ffn.lin1 | Linear | 1.554 | 5.511 | 1.500 | 1 |
| 3 | transformer.layer.3.ffn.lin1 | Linear | 1.545 | 5.480 | 1.500 | 1 |
| 4 | transformer.layer.2.ffn.lin1 | Linear | 1.543 | 5.471 | 1.500 | 1 |
| 5 | transformer.layer.0.ffn.lin1 | Linear | 1.534 | 5.442 | 1.500 | 1 |
| 6 | transformer.layer.1.ffn.lin1 | Linear | 1.528 | 5.421 | 1.500 | 1 |
| 7 | transformer.layer.4.ffn.lin2 | Linear | 1.463 | 5.188 | 0.375 | 1 |
| 8 | transformer.layer.1.ffn.lin2 | Linear | 1.463 | 5.187 | 0.375 | 1 |
| 9 | transformer.layer.2.ffn.lin2 | Linear | 1.460 | 5.179 | 0.375 | 1 |
| 10 | transformer.layer.5.ffn.lin2 | Linear | 1.460 | 5.178 | 0.375 | 1 |

## Tỷ trọng theo loại layer · cpu_laptop

| layer_type | time_ms | n_layers | pct |
| --- | --- | --- | --- |
| Linear | 26.864 | 36 | 95.276 |
| GELUActivation | 0.786 | 6 | 2.788 |
| LayerNorm | 0.465 | 13 | 1.650 |
| Embedding | 0.081 | 2 | 0.287 |

## Top-10 layer chậm nhất · colab_t4

| rank | layer_name | layer_type | time_ms | pct | activation_mb | n_calls |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | transformer.layer.4.ffn.lin2 | Linear | 0.159 | 4.949 | 0.375 | 1 |
| 2 | transformer.layer.5.ffn.lin2 | Linear | 0.159 | 4.941 | 0.375 | 1 |
| 3 | transformer.layer.2.ffn.lin2 | Linear | 0.158 | 4.908 | 0.375 | 1 |
| 4 | transformer.layer.1.ffn.lin2 | Linear | 0.157 | 4.896 | 0.375 | 1 |
| 5 | transformer.layer.0.ffn.lin2 | Linear | 0.157 | 4.882 | 0.375 | 1 |
| 6 | transformer.layer.3.ffn.lin2 | Linear | 0.156 | 4.849 | 0.375 | 1 |
| 7 | transformer.layer.4.ffn.lin1 | Linear | 0.129 | 4.013 | 1.500 | 1 |
| 8 | transformer.layer.5.ffn.lin1 | Linear | 0.128 | 3.997 | 1.500 | 1 |
| 9 | transformer.layer.2.ffn.lin1 | Linear | 0.127 | 3.955 | 1.500 | 1 |
| 10 | transformer.layer.1.ffn.lin1 | Linear | 0.127 | 3.951 | 1.500 | 1 |

## Tỷ trọng theo loại layer · colab_t4

| layer_type | time_ms | n_layers | pct |
| --- | --- | --- | --- |
| Linear | 3.052 | 36 | 94.974 |
| LayerNorm | 0.107 | 13 | 3.316 |
| GELUActivation | 0.043 | 6 | 1.326 |
| Embedding | 0.012 | 2 | 0.384 |

## CPU (cpu_laptop) và GPU (colab_t4) · top-10

| layer_name | layer_type | time_ms_cpu | pct_cpu | rank_cpu | time_ms_gpu | pct_gpu | rank_gpu | speedup | rank_change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| transformer.layer.5.ffn.lin1 | Linear | 1.559 | 5.530 | 1 | 0.128 | 3.997 | 8 | 12.140 | -7 |
| transformer.layer.4.ffn.lin1 | Linear | 1.554 | 5.511 | 2 | 0.129 | 4.013 | 7 | 12.051 | -5 |
| transformer.layer.3.ffn.lin1 | Linear | 1.545 | 5.480 | 3 | 0.127 | 3.945 | 11 | 12.188 | -8 |
| transformer.layer.2.ffn.lin1 | Linear | 1.543 | 5.471 | 4 | 0.127 | 3.955 | 9 | 12.136 | -5 |
| transformer.layer.0.ffn.lin1 | Linear | 1.534 | 5.442 | 5 | 0.126 | 3.927 | 12 | 12.160 | -7 |
| transformer.layer.1.ffn.lin1 | Linear | 1.528 | 5.421 | 6 | 0.127 | 3.951 | 10 | 12.036 | -4 |
| transformer.layer.4.ffn.lin2 | Linear | 1.463 | 5.188 | 7 | 0.159 | 4.949 | 1 | 9.198 | 6 |
| transformer.layer.1.ffn.lin2 | Linear | 1.463 | 5.187 | 8 | 0.157 | 4.896 | 4 | 9.295 | 4 |
| transformer.layer.2.ffn.lin2 | Linear | 1.460 | 5.179 | 9 | 0.158 | 4.908 | 3 | 9.258 | 6 |
| transformer.layer.5.ffn.lin2 | Linear | 1.460 | 5.178 | 10 | 0.159 | 4.941 | 2 | 9.194 | 8 |
| transformer.layer.0.ffn.lin2 | Linear | 1.455 | 5.161 | 11 | 0.157 | 4.882 | 5 | 9.276 | 6 |
| transformer.layer.3.ffn.lin2 | Linear | 1.450 | 5.143 | 12 | 0.156 | 4.849 | 6 | 9.307 | 6 |

## CPU và GPU · tỷ trọng theo loại layer

| layer_type | time_ms_cpu | n_layers | pct_cpu | time_ms_gpu | pct_gpu | pct_change |
| --- | --- | --- | --- | --- | --- | --- |
| Linear | 26.864 | 36 | 95.276 | 3.052 | 94.974 | -0.301 |
| GELUActivation | 0.786 | 6 | 2.788 | 0.043 | 1.326 | -1.462 |
| LayerNorm | 0.465 | 13 | 1.650 | 0.107 | 3.316 | 1.666 |
| Embedding | 0.081 | 2 | 0.287 | 0.012 | 0.384 | 0.097 |
