"""Scores over other household splits, on the small synthetic archive."""
import numpy as np
import pandas as pd
import pytest

import split_sensitivity as S
import train as T


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


def test_a_trained_run_is_scored_on_the_test_households_of_its_split(archive, tmp_path, monkeypatch):
    # the classifier test needs five households per label, more than the small archive has
    monkeypatch.setattr("metrics.c2st_auc", lambda *args, **kwargs: float("nan"))
    small = {"epochs": 2, "val_every": 1, "val_max_per_label": 40, "batch_size": 128, "hidden_size": 32}
    for name, split_seed in (("improved", S.STUDY_SEED), ("improved_split1", 1), ("improved_split2", 2),
                             ("final_split1", 1)):
        config = name.split("_")[0]
        T.train({**T.DEFAULTS, **T.CONFIGS[config], **small, "config": config, "seed": 0,
                 "data": str(archive), "split_seed": split_seed}, tmp_path / "runs" / name / "seed0")
    models = [str(tmp_path / "runs" / name) for name in ("improved_split1", "improved_split2", "improved",
                                                         "final_split1")]
    assert S.one_run(S.finished(models)[0])["split_seed"] == 1
    S.main(["--splits", "3", "--workers", "1", "--out", str(tmp_path), "--data", str(archive), "--trained", *models])
    df = pd.read_csv(tmp_path / "split_sensitivity.csv")
    runs = df[df.set == "improved[best]/seed0"].set_index("split_seed")
    assert sorted(runs.index) == [1, 2, S.STUDY_SEED] and runs[["accuracy", "profile.coverage"]].notna().all().all()
    real = df[df.set == "real[train]"].set_index("split_seed")
    assert (runs["test days"] == real.loc[runs.index, "test days"]).all()      # the test households of its own split
    sets = df[df.set.isin(S.SETS)]
    assert len(df) == 4 * len(S.SETS) + 4 and sets[sets.split_seed == 0][list(S.PROFILE)].isna().all().all()
    assert df[df.set == "final[best]/seed0"].split_seed.tolist() == [1]
    assert sets[sets.split_seed != 0]["profile.coverage"].notna().all()     # the splits with a trained run
    text = (tmp_path / "split_sensitivity.md").read_text(encoding="utf-8")
    assert "Trained model `improved`" in text and text.count("mean ± SD of the 2 other splits") == 2
    assert "Trained model `final`" in text
    for folder, depth in (("other", 2), ("again", 3)):                  # another depth; the same run once more
        (tmp_path / "runs" / folder / "seed0").mkdir(parents=True)
        for name in ("config.json", "train_summary.json"):
            cfg = (tmp_path / "runs" / "improved" / "seed0" / name).read_text()
            (tmp_path / "runs" / folder / "seed0" / name).write_text(cfg.replace('"depth": 3', f'"depth": {depth}'))
    with pytest.raises(SystemExit, match="differ in more than the split and the seed"):
        S.finished([*models, str(tmp_path / "runs" / "other")])
    with pytest.raises(SystemExit, match=f"more than one seed0 of improved on the split {S.STUDY_SEED}"):
        S.finished([*models, str(tmp_path / "runs" / "again")])
    with pytest.raises(SystemExit, match="not among the scored splits"):
        S.main(["--splits", "1", "--workers", "1", "--out", str(tmp_path), "--data", str(archive),
                "--trained", *models])
