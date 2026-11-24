# Re-training on other hardware

Runs of `runs_cpu/` (cpu, torch 2.11.0+cu130, 1 thread(s)), evaluated on the held-out test households with the protocol of `summary.md`.  The published runs were trained on a CUDA device; the CPU and CUDA generators draw different random numbers, so every run here is a new, independent run of its configuration.  Accuracies in percent.

## Published seeds and the re-trained runs

SD and range over the published seeds; the last column is the distance of the re-trained run to the published mean in published SDs.

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
