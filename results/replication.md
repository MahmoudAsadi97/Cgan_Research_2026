# Re-training on other hardware

Runs of `runs_cpu/` (cpu, torch 2.11.0+cu130, 1 thread(s)), evaluated on the held-out test households with the protocol of `summary.md`.  The published runs were trained on a CUDA device; the CPU and CUDA generators draw different random numbers, so every run here is a new, independent run of its configuration.  Accuracies in percent.

## Published seeds and the re-trained runs

SD and range over the published seeds; the last column is the distance of the re-trained run to the published mean in published SDs.

### `final` [best]

| metric             | published, 5 seeds   | published range   |   re-trained, seed 0 | distance, seed 0 (SD)   |
|:-------------------|:---------------------|:------------------|---------------------:|:------------------------|
| Accuracy           | 87.87 ± 0.21         | 87.61 to 88.12    |              87.97   | +0.5                    |
| Worst label        | 74.64 ± 0.57         | 74.11 to 75.40    |              75.26   | +1.1                    |
| Mean profile       | 86.73 ± 0.21         | 86.37 to 86.90    |              87.12   | +1.9                    |
| Std profile        | 79.64 ± 0.38         | 79.20 to 80.19    |              79.58   | -0.2                    |
| KP mean            | 92.79 ± 0.20         | 92.46 to 92.97    |              93.03   | +1.2                    |
| KP std             | 90.08 ± 0.37         | 89.74 to 90.63    |              89.94   | -0.4                    |
| Hourly 1-KS        | 93.47 ± 0.05         | 93.43 to 93.54    |              93.58   | +2.4                    |
| KP 1-KS            | 93.84 ± 0.09         | 93.71 to 93.94    |              93.8    | -0.4                    |
| FID (5 KP)         | 0.853 ± 0.034        | 0.807 to 0.897    |               0.819  | -1.0                    |
| rFD                | 0.117 ± 0.0048       | 0.109 to 0.121    |               0.116  | -0.2                    |
| Corr. MAE          | 0.0706 ± 0.00098     | 0.0691 to 0.0719  |               0.0706 | -0.1                    |
| C2ST AUC           | 0.609 ± 0.0056       | 0.602 to 0.615    |               0.615  | +1.2                    |
| Coverage           | 0.808 ± 0.0021       | 0.807 to 0.812    |               0.807  | -0.8                    |
| DCR ratio          | 1.01 ± 0.0031        | 1 to 1.01         |               1      | -0.9                    |
| Neg. share (no PV) | 0 ± 0                | 0 to 0            |               0      | –                       |

### `final` [last]

| metric             | published, 5 seeds   | published range   |   re-trained, seed 0 | distance, seed 0 (SD)   |
|:-------------------|:---------------------|:------------------|---------------------:|:------------------------|
| Accuracy           | 87.61 ± 0.23         | 87.34 to 87.86    |              88.12   | +2.3                    |
| Worst label        | 70.84 ± 1.07         | 69.19 to 71.98    |              72.33   | +1.4                    |
| Mean profile       | 86.39 ± 0.22         | 86.16 to 86.70    |              86.93   | +2.5                    |
| Std profile        | 78.71 ± 0.50         | 77.94 to 79.19    |              79.68   | +1.9                    |
| KP mean            | 93.22 ± 0.18         | 93.00 to 93.42    |              93.57   | +1.9                    |
| KP std             | 89.74 ± 0.31         | 89.41 to 90.11    |              90.22   | +1.6                    |
| Hourly 1-KS        | 93.56 ± 0.02         | 93.53 to 93.59    |              93.67   | +5.0                    |
| KP 1-KS            | 94.16 ± 0.08         | 94.06 to 94.24    |              94.26   | +1.4                    |
| FID (5 KP)         | 0.809 ± 0.034        | 0.78 to 0.852     |               0.723  | -2.5                    |
| rFD                | 0.129 ± 0.0037       | 0.125 to 0.134    |               0.125  | -1.2                    |
| Corr. MAE          | 0.0717 ± 0.0012      | 0.0703 to 0.0736  |               0.0715 | -0.1                    |
| C2ST AUC           | 0.587 ± 0.0017       | 0.585 to 0.589    |               0.587  | +0.3                    |
| Coverage           | 0.817 ± 0.002        | 0.814 to 0.819    |               0.816  | -0.7                    |
| DCR ratio          | 0.999 ± 0.0041       | 0.992 to 1        |               1      | +1.0                    |
| Neg. share (no PV) | 0 ± 0                | 0 to 0            |               0      | –                       |

### `improved` [best] (published values from the rounded cells of `ablation.csv`)

| metric             | published, 5 seeds   |   re-trained, seed 0 | distance, seed 0 (SD)   |
|:-------------------|:---------------------|---------------------:|:------------------------|
| Accuracy           | 87.50 ± 0.20         |              87.67   | +0.9                    |
| Worst label        | 76.00 ± 1.50         |              77.16   | +0.8                    |
| Mean profile       | 86.50 ± 0.20         |              87      | +2.5                    |
| Std profile        | 79.40 ± 0.60         |              80.01   | +1.0                    |
| KP mean            | 91.80 ± 0.30         |              91.43   | -1.2                    |
| KP std             | 89.80 ± 0.50         |              89.35   | -0.9                    |
| Hourly 1-KS        | 93.40 ± 0.10         |              93.43   | +0.3                    |
| KP 1-KS            | 93.20 ± 0.20         |              93.12   | -0.4                    |
| FID (5 KP)         | 1.03 ± 0.075         |               1.08   | +0.7                    |
| rFD                | 0.117 ± 0.0051       |               0.112  | -1.0                    |
| Corr. MAE          | 0.0705 ± 0.00098     |               0.0696 | -0.9                    |
| C2ST AUC           | 0.645 ± 0.011        |               0.646  | +0.1                    |
| Coverage           | 0.791 ± 0.0043       |               0.792  | +0.2                    |
| DCR ratio          | 1.03 ± 0.0058        |               1.03   | +0.7                    |
| Neg. share (no PV) | 0 ± 0                |               0      | –                       |
