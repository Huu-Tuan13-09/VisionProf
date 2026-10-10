# TN1 · vit_b_16 (bs1_fp32)

## Top-10 layer chậm nhất · cpu_laptop

|   rank | layer_name                                     | layer_type         |            time_ms |                pct |   activation_mb |   n_calls |
|-------:|:-----------------------------------------------|:-------------------|-------------------:|-------------------:|----------------:|----------:|
|      1 | encoder.layers.encoder_layer_2.self_attention  | MultiheadAttention | 2.89525            | 3.213195716108983  |        0.577148 |         1 |
|      2 | encoder.layers.encoder_layer_8.self_attention  | MultiheadAttention | 2.8847500000000004 | 3.2015426446923034 |        0.577148 |         1 |
|      3 | encoder.layers.encoder_layer_4.self_attention  | MultiheadAttention | 2.88335            | 3.1999889018367456 |        0.577148 |         1 |
|      4 | encoder.layers.encoder_layer_7.self_attention  | MultiheadAttention | 2.8569500000000003 | 3.170689750846235  |        0.577148 |         1 |
|      5 | encoder.layers.encoder_layer_3.self_attention  | MultiheadAttention | 2.8516             | 3.1647522335053537 |        0.577148 |         1 |
|      6 | encoder.layers.encoder_layer_11.self_attention | MultiheadAttention | 2.83405            | 3.145274956994617  |        0.577148 |         1 |
|      7 | encoder.layers.encoder_layer_1.self_attention  | MultiheadAttention | 2.8253000000000004 | 3.135564064147384  |        0.577148 |         1 |
|      8 | encoder.layers.encoder_layer_10.self_attention | MultiheadAttention | 2.8226             | 3.132567560068808  |        0.577148 |         1 |
|      9 | encoder.layers.encoder_layer_6.self_attention  | MultiheadAttention | 2.8124000000000002 | 3.121247433549747  |        0.577148 |         1 |
|     10 | encoder.layers.encoder_layer_0.self_attention  | MultiheadAttention | 2.80535            | 3.1134232284556895 |        0.577148 |         1 |

## Tỷ trọng theo loại layer · cpu_laptop

| layer_type         |   time_ms |   n_layers |                 pct |
|:-------------------|----------:|-----------:|--------------------:|
| Linear             |  52.2244  |         25 | 57.95949170412298   |
| MultiheadAttention |  34.06725 |         12 | 37.80839021142002   |
| GELU               |   1.826   |         12 |  2.026524610177016  |
| LayerNorm          |   1.1279  |         25 |  1.2517618334165697 |
| Conv2d             |   0.85945 |          1 |  0.9538316408634373 |

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

| layer_name                                     | layer_type         |        time_ms_cpu |            pct_cpu |   rank_cpu |         time_ms_gpu |            pct_gpu |   rank_gpu |           speedup |   rank_change |
|:-----------------------------------------------|:-------------------|-------------------:|-------------------:|-----------:|--------------------:|-------------------:|-----------:|------------------:|--------------:|
| encoder.layers.encoder_layer_2.self_attention  | MultiheadAttention | 2.89525            | 3.213195716108983  |          1 | 0.302385            | 3.174387385803254  |          6 | 9.574714354217305 |            -5 |
| encoder.layers.encoder_layer_8.self_attention  | MultiheadAttention | 2.8847500000000004 | 3.2015426446923034 |          2 | 0.30106             | 3.1604777564030213 |         12 | 9.581977014548597 |           -10 |
| encoder.layers.encoder_layer_4.self_attention  | MultiheadAttention | 2.88335            | 3.1999889018367456 |          3 | 0.3036              | 3.187142253517429  |          1 | 9.497200263504611 |             2 |
| encoder.layers.encoder_layer_7.self_attention  | MultiheadAttention | 2.8569500000000003 | 3.170689750846235  |          4 | 0.301295            | 3.1629447472777796 |         10 | 9.482235018835363 |            -6 |
| encoder.layers.encoder_layer_3.self_attention  | MultiheadAttention | 2.8516             | 3.1647522335053537 |          5 | 0.302735            | 3.178061627531617  |          5 | 9.419459263051845 |             0 |
| encoder.layers.encoder_layer_11.self_attention | MultiheadAttention | 2.83405            | 3.145274956994617  |          6 | 0.30355             | 3.186617361841949  |          2 | 9.336353154340307 |             4 |
| encoder.layers.encoder_layer_1.self_attention  | MultiheadAttention | 2.8253000000000004 | 3.135564064147384  |          7 | 0.301935            | 3.1696633607239297 |          9 | 9.357312004239324 |            -2 |
| encoder.layers.encoder_layer_10.self_attention | MultiheadAttention | 2.8226             | 3.132567560068808  |          8 | 0.30122000000000004 | 3.1621574097645593 |         11 | 9.370559723789919 |            -3 |
| encoder.layers.encoder_layer_6.self_attention  | MultiheadAttention | 2.8124000000000002 | 3.121247433549747  |          9 | 0.30274999999999996 | 3.1782190950342613 |          4 | 9.289512799339391 |             5 |
| encoder.layers.encoder_layer_0.self_attention  | MultiheadAttention | 2.80535            | 3.1134232284556895 |         10 | 0.30235500000000004 | 3.1740724507979663 |          7 | 9.278331762332355 |             3 |
| encoder.layers.encoder_layer_5.self_attention  | MultiheadAttention | 2.8036             | 3.111481049886243  |         11 | 0.3031              | 3.181893336762624  |          3 | 9.249752556911911 |             8 |
| encoder.layers.encoder_layer_9.self_attention  | MultiheadAttention | 2.79205            | 3.0986626713278946 |         12 | 0.30221             | 3.1725502649390718 |          8 | 9.238774362198473 |             4 |

## CPU và GPU · tỷ trọng theo loại layer

| layer_type         |   time_ms_cpu |   n_layers_cpu |             pct_cpu |        time_ms_gpu |   n_layers_gpu |             pct_gpu |           pct_change |
|:-------------------|--------------:|---------------:|--------------------:|-------------------:|---------------:|--------------------:|---------------------:|
| Linear             |      52.2244  |             25 | 57.95949170412298   | 5.3148550000000006 |             25 | 55.79446291771537   | -2.1650287864076105  |
| MultiheadAttention |      34.06725 |             12 | 37.80839021142002   | 3.628195           |             12 | 38.08818705039748   |  0.279796838977461   |
| GELU               |       1.826   |             12 |  2.026524610177016  | 0.171825           |             12 |  1.8037902427886447 | -0.22273436738837105 |
| LayerNorm          |       1.1279  |             25 |  1.2517618334165697 | 0.232465           |             25 |  2.4403788668113617 |  1.188617033394792   |
| Conv2d             |       0.85945 |              1 |  0.9538316408634373 | 0.178435           |              1 |  1.8731809222871634 |  0.9193492814237261  |
