# TN1 · vit_b_16 (bs1_fp32)

## Top-10 layer chậm nhất · cpu_laptop

|   rank | layer_name                                     | layer_type         |            time_ms |                pct |   activation_mb |   n_calls |
|-------:|:-----------------------------------------------|:-------------------|-------------------:|-------------------:|----------------:|----------:|
|      1 | encoder.layers.encoder_layer_3.self_attention  | MultiheadAttention | 1.714795           | 3.62940969113206   |          0.5771 |         1 |
|      2 | encoder.layers.encoder_layer_2.self_attention  | MultiheadAttention | 1.7049400000000001 | 3.6085513188449316 |          0.5771 |         1 |
|      3 | encoder.layers.encoder_layer_1.self_attention  | MultiheadAttention | 1.67515            | 3.545499983438178  |          0.5771 |         1 |
|      4 | encoder.layers.encoder_layer_9.self_attention  | MultiheadAttention | 1.6744599999999998 | 3.5440395798990485 |          0.5771 |         1 |
|      5 | encoder.layers.encoder_layer_4.self_attention  | MultiheadAttention | 1.67167            | 3.5381344699364825 |          0.5771 |         1 |
|      6 | encoder.layers.encoder_layer_7.self_attention  | MultiheadAttention | 1.652875           | 3.4983543474467234 |          0.5771 |         1 |
|      7 | encoder.layers.encoder_layer_5.self_attention  | MultiheadAttention | 1.65163            | 3.4957192714956857 |          0.5771 |         1 |
|      8 | encoder.layers.encoder_layer_11.self_attention | MultiheadAttention | 1.645              | 3.481686698358835  |          0.5771 |         1 |
|      9 | encoder.layers.encoder_layer_6.self_attention  | MultiheadAttention | 1.644545           | 3.480723678633757  |          0.5771 |         1 |
|     10 | encoder.layers.encoder_layer_0.self_attention  | MultiheadAttention | 1.635045           | 3.4606166733848767 |          0.5771 |         1 |

## Tỷ trọng theo loại layer · cpu_laptop

| layer_type         |             time_ms |   n_layers |                 pct |
|:-------------------|--------------------:|-----------:|--------------------:|
| Linear             | 21.71717            |         25 | 45.96497380850913   |
| MultiheadAttention | 19.92665            |         12 | 42.1752901202748    |
| GELU               |  3.2237750000000003 |         12 |  6.823206404864286  |
| LayerNorm          |  1.999805           |         25 |  4.232641013867166  |
| Conv2d             |  0.379815           |          1 |  0.8038886524845962 |

## Top-10 layer chậm nhất · colab_t4

|   rank | layer_name                                     | layer_type         |             time_ms |                pct |   activation_mb |   n_calls |
|-------:|:-----------------------------------------------|:-------------------|--------------------:|-------------------:|----------------:|----------:|
|      1 | encoder.layers.encoder_layer_4.self_attention  | MultiheadAttention | 0.3036              | 3.187142253517429  |        0.577148 |         1 |
|      2 | encoder.layers.encoder_layer_11.self_attention | MultiheadAttention | 0.30355             | 3.186617361841949  |        0.577148 |         1 |
|      3 | encoder.layers.encoder_layer_5.self_attention  | MultiheadAttention | 0.3031              | 3.181893336762624  |        0.577148 |         1 |
|      4 | encoder.layers.encoder_layer_6.self_attention  | MultiheadAttention | 0.30274999999999996 | 3.1782190950342613 |        0.577148 |         1 |
|      5 | encoder.layers.encoder_layer_3.self_attention  | MultiheadAttention | 0.302735            | 3.178061627531617  |        0.577148 |         1 |
|      6 | encoder.layers.encoder_layer_2.self_attention  | MultiheadAttention | 0.302385            | 3.174387385803254  |        0.577148 |         1 |
|      7 | encoder.layers.encoder_layer_0.self_attention  | MultiheadAttention | 0.30235500000000004 | 3.1740724507979663 |        0.577148 |         1 |
|      8 | encoder.layers.encoder_layer_9.self_attention  | MultiheadAttention | 0.30221             | 3.1725502649390718 |        0.577148 |         1 |
|      9 | encoder.layers.encoder_layer_1.self_attention  | MultiheadAttention | 0.301935            | 3.1696633607239297 |        0.577148 |         1 |
|     10 | encoder.layers.encoder_layer_7.self_attention  | MultiheadAttention | 0.301295            | 3.1629447472777796 |        0.577148 |         1 |

## Tỷ trọng theo loại layer · colab_t4

| layer_type         |            time_ms |   n_layers |                 pct |
|:-------------------|-------------------:|-----------:|--------------------:|
| Linear             | 5.3148550000000006 |         25 | 55.79446291771537   |
| MultiheadAttention | 3.628195           |         12 | 38.08818705039748   |
| LayerNorm          | 0.232465           |         25 |  2.4403788668113617 |
| Conv2d             | 0.178435           |          1 |  1.8731809222871634 |
| GELU               | 0.171825           |         12 |  1.8037902427886447 |

## CPU (cpu_laptop) và GPU (colab_t4) · top-10

| layer_name                                     | layer_type         |        time_ms_cpu |            pct_cpu |   rank_cpu |         time_ms_gpu |            pct_gpu |   rank_gpu |            speedup |   rank_change |
|:-----------------------------------------------|:-------------------|-------------------:|-------------------:|-----------:|--------------------:|-------------------:|-----------:|-------------------:|--------------:|
| encoder.layers.encoder_layer_3.self_attention  | MultiheadAttention | 1.714795           | 3.62940969113206   |          1 | 0.302735            | 3.178061627531617  |          5 | 5.664343402645879  |            -4 |
| encoder.layers.encoder_layer_2.self_attention  | MultiheadAttention | 1.7049400000000001 | 3.6085513188449316 |          2 | 0.302385            | 3.174387385803254  |          6 | 5.63830877854391   |            -4 |
| encoder.layers.encoder_layer_1.self_attention  | MultiheadAttention | 1.67515            | 3.545499983438178  |          3 | 0.301935            | 3.1696633607239297 |          9 | 5.548048421017768  |            -6 |
| encoder.layers.encoder_layer_9.self_attention  | MultiheadAttention | 1.6744599999999998 | 3.5440395798990485 |          4 | 0.30221             | 3.1725502649390718 |          8 | 5.540716720161477  |            -4 |
| encoder.layers.encoder_layer_4.self_attention  | MultiheadAttention | 1.67167            | 3.5381344699364825 |          5 | 0.3036              | 3.187142253517429  |          1 | 5.506159420289856  |             4 |
| encoder.layers.encoder_layer_7.self_attention  | MultiheadAttention | 1.652875           | 3.4983543474467234 |          6 | 0.301295            | 3.1629447472777796 |         10 | 5.485902520785277  |            -4 |
| encoder.layers.encoder_layer_5.self_attention  | MultiheadAttention | 1.65163            | 3.4957192714956857 |          7 | 0.3031              | 3.181893336762624  |          3 | 5.449125701088749  |             4 |
| encoder.layers.encoder_layer_11.self_attention | MultiheadAttention | 1.645              | 3.481686698358835  |          8 | 0.30355             | 3.186617361841949  |          2 | 5.419206061604349  |             6 |
| encoder.layers.encoder_layer_6.self_attention  | MultiheadAttention | 1.644545           | 3.480723678633757  |          9 | 0.30274999999999996 | 3.1782190950342613 |          4 | 5.432023121387283  |             5 |
| encoder.layers.encoder_layer_0.self_attention  | MultiheadAttention | 1.635045           | 3.4606166733848767 |         10 | 0.30235500000000004 | 3.1740724507979663 |          7 | 5.4076995584660414 |             3 |

## CPU và GPU · tỷ trọng theo loại layer

| layer_type         |         time_ms_cpu |   n_layers_cpu |             pct_cpu |        time_ms_gpu |   n_layers_gpu |             pct_gpu |          pct_change |
|:-------------------|--------------------:|---------------:|--------------------:|-------------------:|---------------:|--------------------:|--------------------:|
| Linear             | 21.71717            |             25 | 45.96497380850913   | 5.3148550000000006 |             25 | 55.79446291771537   |  9.829489109206243  |
| MultiheadAttention | 19.92665            |             12 | 42.1752901202748    | 3.628195           |             12 | 38.08818705039748   | -4.0871030698773225 |
| GELU               |  3.2237750000000003 |             12 |  6.823206404864286  | 0.171825           |             12 |  1.8037902427886447 | -5.0194161620756415 |
| LayerNorm          |  1.999805           |             25 |  4.232641013867166  | 0.232465           |             25 |  2.4403788668113617 | -1.7922621470558044 |
| Conv2d             |  0.379815           |              1 |  0.8038886524845962 | 0.178435           |              1 |  1.8731809222871634 |  1.0692922698025673 |
