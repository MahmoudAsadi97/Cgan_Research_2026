"""Statistical accuracy over other random splits of the households.

    python split_sensitivity.py                 # the published split and 20 other splits

Every number of the study is conditional on one random split of the households
(``split_seed`` 2024).  This script draws the split again with other seeds and
scores, on the test households of every split, what needs no trained network:

    real: training households     real days of the training households of the split
    real: validation households   real days of its validation households
    Gaussian copula               ``baselines.GaussianCopula`` fitted on its training households

It shows whether the published split is an ordinary one, how the spread over
splits compares with the household standard error of ``uncertainty.py`` (which
keeps the training households fixed and resamples the test households), and
whether the difference between two sets is the same in every split.

Output: ``results/split_sensitivity.md`` and ``.csv`` (one row per split and set).
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import GaussianCopula
from data_prep import load_profiles
from evaluate import EVAL_SEED, evaluate, real_reference, summarise

PUBLISHED_SEED = 2024
TERMS = {"accuracy": "accuracy", "accuracy.min_label": "worst label", "mean_profile": "mean profile",
         "std_profile": "std profile", "kpi_mean": "KP mean", "kpi_std": "KP std", "hourly_dist": "hourly 1-KS",
         "kpi_dist": "KP 1-KS"}
SETS = {"real[train]": "real: training households", "real[val]": "real: validation households",
        "baseline[copula]": "Gaussian copula"}


def one_split(job) -> list[dict]:
    """The three sets scored on the test households of the split with seed ``job[0]``."""
    seed, data = job
    splits = load_profiles(data, split_seed=seed)
    test = splits["test"]
    sets = {"real[train]": real_reference(splits, "train", "test")[0],
            "real[val]": real_reference(splits, "val", "test")[0],
            "baseline[copula]": GaussianCopula().fit(splits["train"]).sample_kwh(test.label, EVAL_SEED)}
    rows = []
    for name, gen in sets.items():
        s = summarise(evaluate(gen, test, splits["train"], profile=False))
        rows.append({"split_seed": seed, "set": name, "test households": len(np.unique(test.meter)),
                     "test days": len(test), **{k: s[k] for k in TERMS}})
    return rows


def household_se(replicates: Path) -> dict[str, float]:
    """Half-sampling standard error of the accuracy of every set stored by ``uncertainty.py``."""
    if not replicates.exists():
        return {}
    rep = pd.read_csv(replicates, header=[0, 1])
    d = rep["A"] - rep["B"]
    out = {}
    for name in SETS:
        cols = [c for c in d.columns if c == name or c.startswith(name + "/")]
        if cols:
            out[name] = float(d[cols].mean(axis=1).std(ddof=1) / 2)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", type=int, default=20, help="number of other splits (seeds 0, 1, ...)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results")
    ap.add_argument("--data", default="data/processed/fluvius_hourly.npz")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [PUBLISHED_SEED, *range(args.splits)]
    with ProcessPoolExecutor(args.workers) as ex:
        rows = [r for part in ex.map(one_split, [(s, args.data) for s in seeds]) for r in part]
    df = pd.DataFrame(rows)
    df.to_csv(out / "split_sensitivity.csv", index=False)

    se = household_se(out / "uncertainty_replicates.csv")
    acc = 100 * df.pivot(index="split_seed", columns="set", values="accuracy")
    other = acc.drop(index=PUBLISHED_SEED)
    table = [{"set": title, f"published split (seed {PUBLISHED_SEED})": f"{acc.loc[PUBLISHED_SEED, name]:.2f}",
              f"mean of {len(other)} other splits": f"{other[name].mean():.2f}",
              "median": f"{other[name].median():.2f}", "SD over splits": f"{other[name].std(ddof=1):.2f}",
              "range": f"{other[name].min():.2f} to {other[name].max():.2f}",
              "other splits below the published one": f"{int((other[name] < acc.loc[PUBLISHED_SEED, name]).sum())} "
                                                      f"of {len(other)}",
              "household SE, published split": f"{100 * se[name]:.2f}" if name in se else "–"}
             for name, title in SETS.items()]
    reference = "real[train]"
    diffs = [{"difference": f"{title} - {SETS[reference]}",
              f"published split (seed {PUBLISHED_SEED})": f"{(acc[name] - acc[reference]).loc[PUBLISHED_SEED]:+.2f}",
              f"mean of {len(other)} other splits": f"{(other[name] - other[reference]).mean():+.2f}",
              "SD over splits": f"{(other[name] - other[reference]).std(ddof=1):.2f}",
              "range": f"{(other[name] - other[reference]).min():+.2f} to {(other[name] - other[reference]).max():+.2f}"}
             for name, title in SETS.items() if name != reference]
    terms = (df[df.split_seed != PUBLISHED_SEED].groupby("set", sort=False)[list(TERMS)].agg(["mean", "std"]))
    detail = pd.DataFrame({TERMS[k]: [f"{100 * terms.loc[n, (k, 'mean')]:.1f} ± {100 * terms.loc[n, (k, 'std')]:.1f}"
                                      for n in SETS] for k in TERMS}, index=list(SETS.values()))
    text = (pd.DataFrame(table).to_markdown(index=False) + "\n\n"
            f"Statistical accuracy (%) on the test households of the published split and of {args.splits} other "
            "random splits of the households (same fractions and stratification; a new split changes the training, "
            "validation and test households).  Real days of the training and of the validation households of each "
            "split, and a Gaussian copula fitted on its training households.  The last column is the half-sampling "
            "standard error of `uncertainty.md` for the published split, which keeps the training households "
            "fixed.\n\nDifferences within a split (percentage points):\n\n"
            + pd.DataFrame(diffs).to_markdown(index=False)
            + "\n\nTerms of the accuracy and worst label, mean ± SD over the other splits:\n\n"
            + detail.to_markdown() + "\n")
    (out / "split_sensitivity.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
