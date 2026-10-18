# TN1 · vit_b_16 (bs1_fp32)

## Top-10 layer chậm nhất · cpu_laptop

| rank | layer_name | layer_type | time_ms | pct | activation_mb | n_calls |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | encoder.layers.encoder_layer_2.self_attention | MultiheadAttention | 2.895 | 3.213 | 0.577 | 1 |
| 2 | encoder.layers.encoder_layer_8.self_attention | MultiheadAttention | 2.885 | 3.202 | 0.577 | 1 |
| 3 | encoder.layers.encoder_layer_4.self_attention | MultiheadAttention | 2.883 | 3.200 | 0.577 | 1 |
| 4 | encoder.layers.encoder_layer_7.self_attention | MultiheadAttention | 2.857 | 3.171 | 0.577 | 1 |
| 5 | encoder.layers.encoder_layer_3.self_attention | MultiheadAttention | 2.852 | 3.165 | 0.577 | 1 |
| 6 | encoder.layers.encoder_layer_11.self_attention | MultiheadAttention | 2.834 | 3.145 | 0.577 | 1 |
| 7 | encoder.layers.encoder_layer_1.self_attention | MultiheadAttention | 2.825 | 3.136 | 0.577 | 1 |
| 8 | encoder.layers.encoder_layer_10.self_attention | MultiheadAttention | 2.823 | 3.133 | 0.577 | 1 |
| 9 | encoder.layers.encoder_layer_6.self_attention | MultiheadAttention | 2.812 | 3.121 | 0.577 | 1 |
| 10 | encoder.layers.encoder_layer_0.self_attention | MultiheadAttention | 2.805 | 3.113 | 0.577 | 1 |

## Tỷ trọng theo loại layer · cpu_laptop

| layer_type | time_ms | n_layers | pct |
| --- | --- | --- | --- |
| Linear | 52.224 | 25 | 57.959 |
| MultiheadAttention | 34.067 | 12 | 37.808 |
| GELU | 1.826 | 12 | 2.027 |
| LayerNorm | 1.128 | 25 | 1.252 |
| Conv2d | 0.859 | 1 | 0.954 |

## Top-10 layer chậm nhất · colab_t4

| rank | layer_name | layer_type | time_ms | pct | activation_mb | n_calls |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | encoder.layers.encoder_layer_4.self_attention | MultiheadAttention | 0.304 | 3.187 | 0.577 | 1 |
| 2 | encoder.layers.encoder_layer_11.self_attention | MultiheadAttention | 0.304 | 3.187 | 0.577 | 1 |
| 3 | encoder.layers.encoder_layer_5.self_attention | MultiheadAttention | 0.303 | 3.182 | 0.577 | 1 |
| 4 | encoder.layers.encoder_layer_6.self_attention | MultiheadAttention | 0.303 | 3.178 | 0.577 | 1 |
| 5 | encoder.layers.encoder_layer_3.self_attention | MultiheadAttention | 0.303 | 3.178 | 0.577 | 1 |
| 6 | encoder.layers.encoder_layer_2.self_attention | MultiheadAttention | 0.302 | 3.174 | 0.577 | 1 |
| 7 | encoder.layers.encoder_layer_0.self_attention | MultiheadAttention | 0.302 | 3.174 | 0.577 | 1 |
| 8 | encoder.layers.encoder_layer_9.self_attention | MultiheadAttention | 0.302 | 3.173 | 0.577 | 1 |
| 9 | encoder.layers.encoder_layer_1.self_attention | MultiheadAttention | 0.302 | 3.170 | 0.577 | 1 |
| 10 | encoder.layers.encoder_layer_7.self_attention | MultiheadAttention | 0.301 | 3.163 | 0.577 | 1 |

## Tỷ trọng theo loại layer · colab_t4

| layer_type | time_ms | n_layers | pct |
| --- | --- | --- | --- |
| Linear | 5.315 | 25 | 55.794 |
| MultiheadAttention | 3.628 | 12 | 38.088 |
| LayerNorm | 0.232 | 25 | 2.440 |
| Conv2d | 0.178 | 1 | 1.873 |
| GELU | 0.172 | 12 | 1.804 |

## CPU (cpu_laptop) và GPU (colab_t4) · top-10

| layer_name | layer_type | time_ms_cpu | pct_cpu | rank_cpu | time_ms_gpu | pct_gpu | rank_gpu | speedup | rank_change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| encoder.layers.encoder_layer_2.self_attention | MultiheadAttention | 2.895 | 3.213 | 1 | 0.302 | 3.174 | 6 | 9.575 | -5 |
| encoder.layers.encoder_layer_8.self_attention | MultiheadAttention | 2.885 | 3.202 | 2 | 0.301 | 3.160 | 12 | 9.582 | -10 |
| encoder.layers.encoder_layer_4.self_attention | MultiheadAttention | 2.883 | 3.200 | 3 | 0.304 | 3.187 | 1 | 9.497 | 2 |
| encoder.layers.encoder_layer_7.self_attention | MultiheadAttention | 2.857 | 3.171 | 4 | 0.301 | 3.163 | 10 | 9.482 | -6 |
| encoder.layers.encoder_layer_3.self_attention | MultiheadAttention | 2.852 | 3.165 | 5 | 0.303 | 3.178 | 5 | 9.419 | 0 |
| encoder.layers.encoder_layer_11.self_attention | MultiheadAttention | 2.834 | 3.145 | 6 | 0.304 | 3.187 | 2 | 9.336 | 4 |
| encoder.layers.encoder_layer_1.self_attention | MultiheadAttention | 2.825 | 3.136 | 7 | 0.302 | 3.170 | 9 | 9.357 | -2 |
| encoder.layers.encoder_layer_10.self_attention | MultiheadAttention | 2.823 | 3.133 | 8 | 0.301 | 3.162 | 11 | 9.371 | -3 |
| encoder.layers.encoder_layer_6.self_attention | MultiheadAttention | 2.812 | 3.121 | 9 | 0.303 | 3.178 | 4 | 9.290 | 5 |
| encoder.layers.encoder_layer_0.self_attention | MultiheadAttention | 2.805 | 3.113 | 10 | 0.302 | 3.174 | 7 | 9.278 | 3 |
| encoder.layers.encoder_layer_5.self_attention | MultiheadAttention | 2.804 | 3.111 | 11 | 0.303 | 3.182 | 3 | 9.250 | 8 |
| encoder.layers.encoder_layer_9.self_attention | MultiheadAttention | 2.792 | 3.099 | 12 | 0.302 | 3.173 | 8 | 9.239 | 4 |

## CPU và GPU · tỷ trọng theo loại layer

| layer_type | time_ms_cpu | n_layers | pct_cpu | time_ms_gpu | pct_gpu | pct_change |
| --- | --- | --- | --- | --- | --- | --- |
| Linear | 52.224 | 25 | 57.959 | 5.315 | 55.794 | -2.165 |
| MultiheadAttention | 34.067 | 12 | 37.808 | 3.628 | 38.088 | 0.280 |
| GELU | 1.826 | 12 | 2.027 | 0.172 | 1.804 | -0.223 |
| LayerNorm | 1.128 | 25 | 1.252 | 0.232 | 2.440 | 1.189 |
| Conv2d | 0.859 | 1 | 0.954 | 0.178 | 1.873 | 0.919 |
