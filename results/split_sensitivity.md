| set                         |   split of the study (seed 2024) |   mean of 20 other splits |   median |   SD over splits | range          | other splits below it   |   household SE, split of the study |
|:----------------------------|---------------------------------:|--------------------------:|---------:|-----------------:|:---------------|:------------------------|-----------------------------------:|
| real: training households   |                            87.95 |                     88.44 |    88.66 |             1.18 | 84.63 to 90.06 | 4 of 20                 |                               0.78 |
| real: validation households |                            86.06 |                     86.7  |    86.72 |             1.1  | 84.43 to 88.45 | 5 of 20                 |                               0.98 |
| Gaussian copula             |                            85.78 |                     86.28 |    86.54 |             1.12 | 82.65 to 87.76 | 4 of 20                 |                               0.77 |

Statistical accuracy (%) on the test households of the split of the study and of 20 other random splits of the households (same fractions and stratification; a new split changes the training, validation and test households).  Real days of the training and of the validation households of each split, and a Gaussian copula fitted on its training households.  The last column is the half-sampling standard error of `uncertainty.md` for the split of the study, which keeps the training households fixed.

Differences within a split (percentage points):

| difference                                              |   split of the study (seed 2024) |   mean of 20 other splits |   SD over splits | range          |
|:--------------------------------------------------------|---------------------------------:|--------------------------:|-----------------:|:---------------|
| real: validation households - real: training households |                            -1.89 |                     -1.75 |             1.34 | -3.98 to +0.92 |
| Gaussian copula - real: training households             |                            -2.17 |                     -2.16 |             0.18 | -2.46 to -1.67 |

Terms of the accuracy and worst label, mean ± SD over the other splits:

|                             | accuracy   | worst label   | mean profile   | std profile   | KP mean    | KP std     | hourly 1-KS   | KP 1-KS    |
|:----------------------------|:-----------|:--------------|:---------------|:--------------|:-----------|:-----------|:--------------|:-----------|
| real: training households   | 88.4 ± 1.2 | 76.1 ± 6.6    | 87.5 ± 1.3     | 80.9 ± 2.5    | 93.3 ± 0.9 | 89.7 ± 1.6 | 93.2 ± 0.4    | 94.5 ± 0.4 |
| real: validation households | 86.7 ± 1.1 | 73.2 ± 6.8    | 84.6 ± 1.8     | 79.7 ± 2.1    | 91.6 ± 1.0 | 88.5 ± 1.4 | 91.6 ± 0.7    | 93.3 ± 0.4 |
| Gaussian copula             | 86.3 ± 1.1 | 73.1 ± 6.5    | 87.4 ± 1.3     | 80.7 ± 2.6    | 86.1 ± 0.7 | 85.9 ± 1.6 | 93.1 ± 0.4    | 88.9 ± 0.3 |


Trained model `final` (validation-selected checkpoint) on the test households of the split it was trained on, with the benchmark and the copula of the same split.  Accuracies in percent and differences in percentage points; then the classifier test (0.5 ideal), coverage and DCR ratio (about 1 expected) of the same sets:

| split                           | seeds   | `final`      | real: training households   | Gaussian copula   | `final` - training households   | `final` - Gaussian copula   |
|:--------------------------------|:--------|:-------------|:----------------------------|:------------------|:--------------------------------|:----------------------------|
| of the study (seed 2024)        | 1       | 87.97        | 87.95                       | 85.78             | +0.02                           | +2.19                       |
| seed 0                          | 1       | 88.89        | 88.81                       | 86.35             | +0.08                           | +2.54                       |
| seed 1                          | 1       | 87.71        | 88.11                       | 86.44             | -0.39                           | +1.28                       |
| seed 2                          | 1       | 88.18        | 88.75                       | 86.56             | -0.57                           | +1.62                       |
| seed 3                          | 1       | 88.30        | 88.80                       | 86.65             | -0.50                           | +1.66                       |
| mean ± SD of the 4 other splits |         | 88.27 ± 0.48 | 88.62 ± 0.34                | 86.50 ± 0.13      | -0.35 ± 0.29                    | +1.77 ± 0.54                |

| split                           | seeds   | C2ST AUC, `final`   | C2ST AUC, real: training households   | C2ST AUC, Gaussian copula   | coverage, `final`   | coverage, real: training households   | coverage, Gaussian copula   | DCR ratio, `final`   | DCR ratio, Gaussian copula   |
|:--------------------------------|:--------|:--------------------|:--------------------------------------|:----------------------------|:--------------------|:--------------------------------------|:----------------------------|:---------------------|:-----------------------------|
| of the study (seed 2024)        | 1       | 0.615               | 0.486                                 | 0.804                       | 0.807               | 0.824                                 | 0.618                       | 1.004                | 1.070                        |
| seed 0                          | 1       | 0.597               | 0.501                                 | 0.806                       | 0.829               | 0.837                                 | 0.646                       | 0.960                | 1.019                        |
| seed 1                          | 1       | 0.607               | 0.493                                 | 0.804                       | 0.825               | 0.835                                 | 0.647                       | 1.004                | 1.073                        |
| seed 2                          | 1       | 0.610               | 0.490                                 | 0.800                       | 0.808               | 0.827                                 | 0.629                       | 1.004                | 1.065                        |
| seed 3                          | 1       | 0.619               | 0.497                                 | 0.796                       | 0.805               | 0.822                                 | 0.637                       | 0.995                | 1.069                        |
| mean ± SD of the 4 other splits |         | 0.608 ± 0.009       | 0.495 ± 0.005                         | 0.802 ± 0.004               | 0.817 ± 0.012       | 0.830 ± 0.007                         | 0.640 ± 0.008               | 0.991 ± 0.021        | 1.057 ± 0.025                |

Terms of the accuracy and worst label over these 4 other splits, mean ± SD:

|                           | accuracy   | worst label   | mean profile   | std profile   | KP mean    | KP std     | hourly 1-KS   | KP 1-KS    |
|:--------------------------|:-----------|:--------------|:---------------|:--------------|:-----------|:-----------|:--------------|:-----------|
| `final`                   | 88.3 ± 0.5 | 79.6 ± 3.1    | 87.3 ± 1.0     | 82.2 ± 0.6    | 92.0 ± 0.5 | 88.8 ± 0.5 | 92.6 ± 0.3    | 93.7 ± 0.4 |
| real: training households | 88.6 ± 0.3 | 78.1 ± 5.4    | 87.2 ± 1.0     | 82.2 ± 0.6    | 92.9 ± 0.5 | 90.2 ± 0.6 | 92.9 ± 0.2    | 94.2 ± 0.4 |
| Gaussian copula           | 86.5 ± 0.1 | 74.5 ± 6.4    | 87.1 ± 0.8     | 82.0 ± 0.7    | 85.8 ± 0.4 | 86.6 ± 0.5 | 92.8 ± 0.2    | 88.7 ± 0.2 |


Trained model `improved` (validation-selected checkpoint) on the test households of the split it was trained on, with the benchmark and the copula of the same split.  Accuracies in percent and differences in percentage points; then the classifier test (0.5 ideal), coverage and DCR ratio (about 1 expected) of the same sets:

| split                           | seeds   | `improved`   | real: training households   | Gaussian copula   | `improved` - training households   | `improved` - Gaussian copula   |
|:--------------------------------|:--------|:-------------|:----------------------------|:------------------|:-----------------------------------|:-------------------------------|
| of the study (seed 2024)        | 1       | 87.67        | 87.95                       | 85.78             | -0.28                              | +1.89                          |
| seed 0                          | 1       | 88.92        | 88.81                       | 86.35             | +0.11                              | +2.57                          |
| seed 1                          | 1       | 87.69        | 88.11                       | 86.44             | -0.42                              | +1.25                          |
| seed 2                          | 1       | 87.70        | 88.75                       | 86.56             | -1.05                              | +1.14                          |
| seed 3                          | 1       | 87.50        | 88.80                       | 86.65             | -1.31                              | +0.85                          |
| seed 4                          | 1       | 87.01        | 86.67                       | 84.68             | +0.34                              | +2.33                          |
| seed 5                          | 1       | 88.87        | 89.81                       | 87.48             | -0.94                              | +1.40                          |
| seed 6                          | 1       | 87.43        | 88.99                       | 86.79             | -1.56                              | +0.64                          |
| seed 7                          | 1       | 85.25        | 84.63                       | 82.65             | +0.62                              | +2.60                          |
| mean ± SD of the 8 other splits |         | 87.55 ± 1.15 | 88.07 ± 1.66                | 85.95 ± 1.55      | -0.53 ± 0.81                       | +1.60 ± 0.79                   |

| split                           | seeds   | C2ST AUC, `improved`   | C2ST AUC, real: training households   | C2ST AUC, Gaussian copula   | coverage, `improved`   | coverage, real: training households   | coverage, Gaussian copula   | DCR ratio, `improved`   | DCR ratio, Gaussian copula   |
|:--------------------------------|:--------|:-----------------------|:--------------------------------------|:----------------------------|:-----------------------|:--------------------------------------|:----------------------------|:------------------------|:-----------------------------|
| of the study (seed 2024)        | 1       | 0.646                  | 0.486                                 | 0.804                       | 0.792                  | 0.824                                 | 0.618                       | 1.034                   | 1.070                        |
| seed 0                          | 1       | 0.639                  | 0.501                                 | 0.806                       | 0.806                  | 0.837                                 | 0.646                       | 0.990                   | 1.019                        |
| seed 1                          | 1       | 0.676                  | 0.493                                 | 0.804                       | 0.794                  | 0.835                                 | 0.647                       | 1.038                   | 1.073                        |
| seed 2                          | 1       | 0.638                  | 0.490                                 | 0.800                       | 0.795                  | 0.827                                 | 0.629                       | 1.030                   | 1.065                        |
| seed 3                          | 1       | 0.649                  | 0.497                                 | 0.796                       | 0.787                  | 0.822                                 | 0.637                       | 1.027                   | 1.069                        |
| seed 4                          | 1       | 0.668                  | 0.511                                 | 0.809                       | 0.788                  | 0.830                                 | 0.623                       | 1.071                   | 1.104                        |
| seed 5                          | 1       | 0.660                  | 0.496                                 | 0.801                       | 0.801                  | 0.838                                 | 0.642                       | 1.027                   | 1.070                        |
| seed 6                          | 1       | 0.677                  | 0.497                                 | 0.802                       | 0.794                  | 0.830                                 | 0.640                       | 1.066                   | 1.097                        |
| seed 7                          | 1       | 0.658                  | 0.511                                 | 0.809                       | 0.792                  | 0.826                                 | 0.635                       | 1.026                   | 1.067                        |
| mean ± SD of the 8 other splits |         | 0.658 ± 0.015          | 0.499 ± 0.008                         | 0.803 ± 0.004               | 0.795 ± 0.006          | 0.830 ± 0.006                         | 0.638 ± 0.008               | 1.034 ± 0.025           | 1.071 ± 0.026                |

Terms of the accuracy and worst label over these 8 other splits, mean ± SD:

|                           | accuracy   | worst label   | mean profile   | std profile   | KP mean    | KP std     | hourly 1-KS   | KP 1-KS    |
|:--------------------------|:-----------|:--------------|:---------------|:--------------|:-----------|:-----------|:--------------|:-----------|
| `improved`                | 87.5 ± 1.2 | 75.7 ± 7.2    | 87.1 ± 1.2     | 80.6 ± 2.5    | 91.1 ± 0.8 | 88.1 ± 1.6 | 92.5 ± 0.6    | 93.2 ± 0.4 |
| real: training households | 88.1 ± 1.7 | 73.4 ± 9.3    | 87.0 ± 1.4     | 80.2 ± 3.8    | 93.0 ± 1.2 | 89.6 ± 2.2 | 93.0 ± 0.5    | 94.4 ± 0.5 |
| Gaussian copula           | 85.9 ± 1.5 | 70.5 ± 9.2    | 86.9 ± 1.4     | 80.1 ± 3.8    | 86.0 ± 0.6 | 85.8 ± 2.3 | 92.9 ± 0.6    | 88.9 ± 0.3 |


`final` minus `improved` on the splits with runs of both (validation-selected checkpoints, each on the test households of its split; accuracy and worst label in percentage points):

| split                           | accuracy (points)   | worst label (points)   | C2ST AUC       | coverage       | DCR ratio      |
|:--------------------------------|:--------------------|:-----------------------|:---------------|:---------------|:---------------|
| of the study (seed 2024)        | +0.30               | -1.90                  | -0.031         | +0.015         | -0.030         |
| seed 0                          | -0.03               | -4.70                  | -0.042         | +0.023         | -0.030         |
| seed 1                          | +0.03               | -1.10                  | -0.069         | +0.031         | -0.034         |
| seed 2                          | +0.48               | +1.46                  | -0.028         | +0.013         | -0.026         |
| seed 3                          | +0.81               | +6.27                  | -0.031         | +0.017         | -0.033         |
| mean ± SD of the 4 other splits | +0.32 ± 0.40        | +0.48 ± 4.61           | -0.043 ± 0.019 | +0.021 ± 0.007 | -0.031 ± 0.004 |
