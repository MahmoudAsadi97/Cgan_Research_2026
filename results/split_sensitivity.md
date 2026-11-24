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
