| model                             |   Seeds | Accuracy   | Mean profile   | Std profile   | KP mean    | KP std     | Hourly 1-KS   | KP 1-KS    | Worst label   | Labels >= 85%   | FID (5 KP)   | KL near-peak    | KL near-base   | rFD             | Corr. MAE        | C2ST AUC        | Precision      | Recall          | Coverage        | DCR ratio     | Neg. share (no PV)   |
|:----------------------------------|--------:|:-----------|:---------------|:--------------|:-----------|:-----------|:--------------|:-----------|:--------------|:----------------|:-------------|:----------------|:---------------|:----------------|:-----------------|:----------------|:---------------|:----------------|:----------------|:--------------|:---------------------|
| Independent hours                 |       5 | 78.4 ± 0.1 | 86.2 ± 0.1     | 78.8 ± 0.2    | 67.6 ± 0.1 | 62.9 ± 0.1 | 93.6 ± 0.0    | 73.1 ± 0.0 | 59.9 ± 0.4    | 0.9 ± 0.9       | 11.7 ± 0.078 | 0.877 ± 0.0055  | 1.53 ± 0.0039  | 0.481 ± 0.0023  | 0.247 ± 9.8e-05  | 0.989 ± 0.00025 | 0.293 ± 0.0017 | 0.808 ± 0.002   | 0.177 ± 0.0011  | 1.9 ± 0.0019  | 1.91e-07 ± 2.6e-07   |
| Multivariate normal               |       5 | 74.7 ± 0.1 | 86.2 ± 0.2     | 79.1 ± 0.0    | 60.6 ± 0.1 | 59.0 ± 0.0 | 77.3 ± 0.0    | 70.3 ± 0.1 | 52.8 ± 0.4    | 6.2 ± 0.0       | 37.8 ± 0.12  | 0.638 ± 0.0048  | 1.39 ± 0.0069  | 0.126 ± 0.00034 | 0.0714 ± 0.00093 | 0.986 ± 0.00011 | 0.357 ± 0.0015 | 0.849 ± 0.00023 | 0.189 ± 0.00066 | 2.12 ± 0.0025 | 0.234 ± 0.00029      |
| Gaussian copula                   |       5 | 85.7 ± 0.2 | 86.3 ± 0.3     | 78.8 ± 0.4    | 86.0 ± 0.1 | 87.1 ± 0.3 | 93.6 ± 0.0    | 88.8 ± 0.1 | 67.4 ± 0.8    | 70.3 ± 1.6      | 2.04 ± 0.038 | 0.113 ± 0.0016  | 0.142 ± 0.0013 | 0.141 ± 0.0035  | 0.0971 ± 0.0015  | 0.804 ± 0.00081 | 0.764 ± 0.0016 | 0.895 ± 0.0013  | 0.619 ± 0.0012  | 1.07 ± 0.0011 | 1.82e-06 ± 2e-06     |
| Gaussian mixture (160 components) |       5 | 85.8 ± 0.1 | 85.7 ± 0.4     | 76.4 ± 0.1    | 88.8 ± 0.1 | 89.1 ± 0.1 | 93.2 ± 0.1    | 91.4 ± 0.1 | 68.4 ± 0.9    | 66.9 ± 3.2      | 1.43 ± 0.043 | 0.0605 ± 0.0011 | 0.137 ± 0.0018 | 0.147 ± 0.0022  | 0.0759 ± 0.00082 | 0.681 ± 0.0011  | 0.792 ± 0.0017 | 0.887 ± 0.0009  | 0.773 ± 0.00093 | 1.02 ± 0.0014 | 0 ± 0                |

Classical generators fitted per label on the training households and scored on the held-out test households with the protocol of `results/summary.md` (as many generated days per label as the test households have; mean ± SD over 5 seeds).  A seed fixes the sampling noise and, for the mixture, the initialisation of the fit; the other fits are deterministic, so their SD is sampling noise only.  The mixture uses the per-hour value transform and the no-PV floor of the final model.

Number of mixture components, selected on the validation households (seed 0; highest validation accuracy, the smaller mixture if two are within 0.1 point):

|   components |   validation accuracy (%) |   worst label (%) | selected   |
|-------------:|--------------------------:|------------------:|:-----------|
|            1 |                     83.05 |             74.89 |            |
|            2 |                     84.17 |             75.76 |            |
|            5 |                     84.95 |             70.35 |            |
|           10 |                     85.3  |             73.12 |            |
|           20 |                     85.67 |             73.86 |            |
|           40 |                     86.17 |             73.95 |            |
|           80 |                     86.73 |             75.77 |            |
|          160 |                     87.18 |             78.04 | yes        |

`final` (validation-selected checkpoints, 5 seeds, `all_runs.csv`) against every generator.  Values are fractions; + every seed of the final model is better than every seed of the generator, - every seed is worse, = the seeds overlap:

| metric                   | better        | final             | Independent hours   | Multivariate normal   | Gaussian copula   | Gaussian mixture (160 components)   |
|:-------------------------|:--------------|:------------------|:--------------------|:----------------------|:------------------|:------------------------------------|
| Accuracy                 | higher        | 0.8787 ± 0.0021   | 0.784 +             | 0.7469 +              | 0.8571 +          | 0.8585 +                            |
| Mean profile             | higher        | 0.8673 ± 0.0021   | 0.8617 +            | 0.8617 =              | 0.8626 =          | 0.8574 +                            |
| Std profile              | higher        | 0.7964 ± 0.0038   | 0.788 +             | 0.7905 +              | 0.7878 =          | 0.7639 +                            |
| KP mean                  | higher        | 0.9279 ± 0.002    | 0.676 +             | 0.606 +               | 0.8603 +          | 0.8883 +                            |
| KP std                   | higher        | 0.9008 ± 0.0037   | 0.6286 +            | 0.5896 +              | 0.8714 +          | 0.891 +                             |
| Hourly 1-KS              | higher        | 0.9347 ± 0.00046  | 0.9364 -            | 0.773 +               | 0.9365 -          | 0.9319 +                            |
| KP 1-KS                  | higher        | 0.9384 ± 0.00091  | 0.7312 +            | 0.7026 +              | 0.8876 +          | 0.914 +                             |
| Worst label              | higher        | 0.7464 ± 0.0057   | 0.5994 +            | 0.5278 +              | 0.6745 +          | 0.6836 +                            |
| Labels >= 85%            | higher        | 0.825 ± 0.03      | 0.009375 +          | 0.0625 +              | 0.7031 +          | 0.6687 +                            |
| FID (5 KP)               | lower         | 0.8534 ± 0.034    | 11.68 +             | 37.84 +               | 2.039 +           | 1.435 +                             |
| KL near-peak             | lower         | 0.05611 ± 0.0017  | 0.8775 +            | 0.6384 +              | 0.1135 +          | 0.06048 +                           |
| KL near-base             | lower         | 0.0797 ± 0.0021   | 1.53 +              | 1.39 +                | 0.1423 +          | 0.1366 +                            |
| rFD                      | lower         | 0.117 ± 0.0048    | 0.4805 +            | 0.1259 +              | 0.1409 +          | 0.1466 +                            |
| Corr. MAE                | lower         | 0.07063 ± 0.00098 | 0.2471 +            | 0.07135 =             | 0.09713 +         | 0.07595 +                           |
| C2ST AUC                 | closer to 0.5 | 0.6086 ± 0.0056   | 0.989 +             | 0.9856 +              | 0.8038 +          | 0.6814 +                            |
| Precision                | higher        | 0.8227 ± 0.0023   | 0.2929 +            | 0.3571 +              | 0.7637 +          | 0.7915 +                            |
| Recall                   | higher        | 0.878 ± 0.0015    | 0.8077 +            | 0.8485 +              | 0.8952 -          | 0.8873 -                            |
| Coverage                 | higher        | 0.8083 ± 0.0021   | 0.1765 +            | 0.1888 +              | 0.6187 +          | 0.7733 +                            |
| DCR ratio                | closer to 1.0 | 1.007 ± 0.0031    | 1.9 +               | 2.12 +                | 1.07 +            | 1.022 +                             |
| Neg. share (no PV)       | lower         | 0 ± 0             | 1.908e-07 =         | 0.2335 +              | 1.822e-06 +       | 0 =                                 |
| paper.mean_diff_lt1      | higher        | 0.9925 ± 0.0036   | 0.6006 +            | 0.6038 +              | 0.9356 +          | 0.9738 +                            |
| paper.std_diff_lt1       | higher        | 0.9994 ± 0.0014   | 0.6831 +            | 0.5112 +              | 0.9969 =          | 0.9975 =                            |
| paper.kl_lt0.3           | higher        | 1 ± 0             | 0.4756 +            | 0.3737 +              | 0.9831 +          | 0.9931 =                            |
| paper.fid_lt5            | higher        | 1 ± 0             | 0.9262 +            | 0.6975 +              | 0.9975 =          | 0.9975 =                            |
| paper.mean_kl            | lower         | 0.03721 ± 0.001   | 0.602 +             | 0.7171 +              | 0.09119 +         | 0.05619 +                           |
| paper.mean_fid           | lower         | 0.1433 ± 0.007    | 2.242 +             | 7.344 +               | 0.3602 +          | 0.2536 +                            |
| paper_code.mean_diff_lt1 | higher        | 0.9925 ± 0.0028   | 0.7844 +            | 0.5106 +              | 0.96 +            | 0.985 +                             |
| paper_code.std_diff_lt1  | higher        | 1 ± 0             | 0.825 +             | 0.7637 +              | 0.9963 =          | 1 =                                 |
| paper_code.fid_lt5       | higher        | 1 ± 0             | 0.9481 +            | 0.7525 +              | 1 =               | 1 =                                 |
| paper_code.mean_fid      | lower         | 0.136 ± 0.007     | 1.438 +             | 5.568 +               | 0.2827 +          | 0.1591 +                            |
