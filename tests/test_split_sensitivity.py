"""Scores over other household splits, on the small synthetic archive."""
import numpy as np
import pandas as pd
import pytest

import split_sensitivity as S


def test_one_split_scores_the_three_sets(archive):
    rows = S.one_split((3, str(archive)))
    assert [r["set"] for r in rows] == list(S.SETS)
    assert all(r["split_seed"] == 3 and 0 <= r["accuracy"] <= 1 for r in rows)
    assert rows == S.one_split((3, str(archive)))                       # a split seed fixes the result
    assert rows[0]["accuracy"] != S.one_split((4, str(archive)))[0]["accuracy"]


def test_household_se_is_half_the_sd_of_the_half_differences(tmp_path):
    a = pd.DataFrame({"real[train]": [0.90, 0.80, 0.70, 0.85], "baseline[copula]/seed0": [0.80, 0.82, 0.78, 0.80],
                      "baseline[copula]/seed1": [0.70, 0.90, 0.80, 0.60], "final[best]/seed0": [0.9, 0.9, 0.9, 0.9]})
    b = pd.DataFrame({"real[train]": [0.80, 0.85, 0.75, 0.80], "baseline[copula]/seed0": [0.81, 0.80, 0.80, 0.79],
                      "baseline[copula]/seed1": [0.75, 0.80, 0.85, 0.70], "final[best]/seed0": [0.8, 0.8, 0.8, 0.8]})
    pd.concat({"A": a, "B": b}, axis=1).to_csv(tmp_path / "uncertainty_replicates.csv", index=False)
    se = S.household_se(tmp_path / "uncertainty_replicates.csv")
    assert set(se) == {"real[train]", "baseline[copula]"}
    assert se["real[train]"] == pytest.approx(np.std([0.10, -0.05, -0.05, 0.05], ddof=1) / 2)
    copula = (a.iloc[:, 1:3] - b.iloc[:, 1:3]).mean(axis=1)             # the seeds are averaged before the SD
    assert se["baseline[copula]"] == pytest.approx(copula.std(ddof=1) / 2)
    assert S.household_se(tmp_path / "missing.csv") == {}
