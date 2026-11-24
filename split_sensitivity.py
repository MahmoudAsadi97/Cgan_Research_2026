"""Statistical accuracy over other random splits of the households.

    python split_sensitivity.py                                      # the split of the study and 20 other splits
    python run_jobs.py jobs/splits.txt --runs runs_splits            # `improved` trained on eight of them
    python split_sensitivity.py --trained runs_splits/* runs/improved

Every number of the study is conditional on one random split of the households
(``split_seed`` 2024).  This script draws the split again with other seeds and
scores, on the test households of every split, what needs no trained network:

    real: training households     real days of the training households of the split
    real: validation households   real days of its validation households
    Gaussian copula               ``baselines.GaussianCopula`` fitted on its training households

It shows whether the split of the study is an ordinary one, how the spread over
splits compares with the household standard error of ``uncertainty.py`` (which
keeps the training households fixed and resamples the test households), and
whether the difference between two sets is the same in every split.

``--trained`` adds trained models.  A run is trained and selected on the
training and validation households of the split in its configuration
(``split_seed``), so every finished run of the given model folders is scored
on the test households of its own split (``report.evaluate_run``,
validation-selected checkpoint) and compared with the benchmark and the copula
of that split.  The runs must be one configuration.

Output: ``results/split_sensitivity.md`` and ``.csv`` (one row per split and set).
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import GaussianCopula
from data_prep import load_profiles
from evaluate import EVAL_SEED, evaluate, real_reference, summarise

STUDY_SEED = 2024
TERMS = {"accuracy": "accuracy", "accuracy.min_label": "worst label", "mean_profile": "mean profile",
         "std_profile": "std profile", "kpi_mean": "KP mean", "kpi_std": "KP std", "hourly_dist": "hourly 1-KS",
         "kpi_dist": "KP 1-KS"}
SETS = {"real[train]": "real: training households", "real[val]": "real: validation households",
        "baseline[copula]": "Gaussian copula"}
PROFILE = {"profile.c2st_auc": "C2ST AUC", "profile.coverage": "coverage", "profile.dcr_ratio": "DCR ratio"}


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


def one_run(run_dir) -> dict:
    """A trained run scored on the test households of the split it was trained on."""
    from report import evaluate_run

    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    s = evaluate_run(run_dir, "best", {})
    return {"split_seed": cfg["split_seed"], "set": f"{cfg['config']}[best]/{run_dir.name}",
            **{k: s[k] for k in (*TERMS, *PROFILE)}}


def finished(models: list[str]) -> list[Path]:
    """Finished runs of the model folders; they must differ in the split and the seed only."""
    runs = [d for m in models for d in sorted(Path(m).glob("seed*")) if (d / "train_summary.json").exists()]
    cfgs = {json.dumps({k: v for k, v in json.loads((d / "config.json").read_text()).items()
                        if k not in ("split_seed", "seed", "data")}, sort_keys=True) for d in runs}
    if len(cfgs) > 1:
        raise SystemExit("--trained: the runs are not one configuration")
    return runs


def trained_tables(df: pd.DataFrame) -> str:
    """The trained model next to the benchmark and the copula of the same split, split by split."""
    model = df.set.str.extract(r"^(.+\[best\])/seed\d+$")[0]
    t = df[model.notna()]
    name = f"`{model.dropna().iloc[0].split('[')[0]}`"
    runs = t.groupby("split_seed")[[*TERMS, *PROFILE]].mean()
    acc = 100 * df[model.isna()].pivot(index="split_seed", columns="set", values="accuracy").loc[runs.index]
    cols = {name: 100 * runs.accuracy, SETS["real[train]"]: acc["real[train]"],
            SETS["baseline[copula]"]: acc["baseline[copula]"],
            f"{name} - training households": 100 * runs.accuracy - acc["real[train]"],
            f"{name} - Gaussian copula": 100 * runs.accuracy - acc["baseline[copula]"],
            **{title: runs[k] for k, title in PROFILE.items()}}
    table = pd.DataFrame(cols)
    spec = {c: "+.2f" if " - " in c else ".3f" if c in PROFILE.values() else ".2f" for c in table}
    other = table.drop(index=STUDY_SEED, errors="ignore")
    order = [s for s in (STUDY_SEED, *other.index) if s in table.index]
    body = table.loc[order].copy()
    for c in body:
        body[c] = body[c].map(lambda v, c=c: format(v, spec[c]))
    body.insert(0, "seeds", t.groupby("split_seed").size().loc[order])
    body.insert(0, "split", [f"of the study (seed {s})" if s == STUDY_SEED else f"seed {s}" for s in order])
    detail = ""
    if len(other) > 1:
        summary = {c: f"{other[c].mean():{spec[c]}} ± {other[c].std(ddof=1):{spec[c].lstrip('+')}}" for c in table}
        body = pd.concat([body, pd.DataFrame([{"split": f"mean ± SD of the {len(other)} other splits", "seeds": "",
                                               **summary}])])
        sets = {name: t[t.split_seed != STUDY_SEED].groupby("split_seed")[list(TERMS)].mean(),
                **{SETS[k]: df[(df.set == k) & df.split_seed.isin(other.index)].set_index("split_seed")[list(TERMS)]
                   for k in ("real[train]", "baseline[copula]")}}
        terms = pd.DataFrame({TERMS[k]: [f"{100 * v[k].mean():.1f} ± {100 * v[k].std(ddof=1):.1f}"
                                         for v in sets.values()] for k in TERMS}, index=list(sets))
        detail = (f"\n\nTerms of the accuracy and worst label over these {len(other)} other splits, mean ± SD:\n\n"
                  + terms.to_markdown())
    return (f"\n\nTrained model {name} (validation-selected checkpoint) on the test households of the split it "
            "was trained on, with the benchmark and the copula of the same split; accuracies in percent, "
            "differences in percentage points:\n\n" + body.to_markdown(index=False, disable_numparse=True)
            + detail + "\n")


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
    ap.add_argument("--trained", nargs="*", default=[], metavar="FOLDER",
                    help="model folders with trained runs (<folder>/seed<k>), e.g. runs_splits/* runs/improved")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [STUDY_SEED, *range(args.splits)]
    runs = finished(args.trained)
    jobs = [(s, args.data) for s in seeds]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as ex:
            parts, trained = list(ex.map(one_split, jobs)), list(ex.map(one_run, runs))
    else:
        parts, trained = [one_split(j) for j in jobs], [one_run(d) for d in runs]
    rows = [r for part in parts for r in part]
    size = {r["split_seed"]: (r["test households"], r["test days"]) for r in rows}
    for r in trained:
        if r["split_seed"] not in size:
            raise SystemExit(f"--trained: split {r['split_seed']} is not among the scored splits (--splits)")
        households, days = size[r["split_seed"]]
        rows.append({"split_seed": r.pop("split_seed"), "set": r.pop("set"), "test households": households,
                     "test days": days, **r})
    df = pd.DataFrame(rows)
    df.to_csv(out / "split_sensitivity.csv", index=False)

    se = household_se(out / "uncertainty_replicates.csv")
    acc = 100 * df.pivot(index="split_seed", columns="set", values="accuracy")
    other = acc.drop(index=STUDY_SEED)
    table = [{"set": title, f"split of the study (seed {STUDY_SEED})": f"{acc.loc[STUDY_SEED, name]:.2f}",
              f"mean of {len(other)} other splits": f"{other[name].mean():.2f}",
              "median": f"{other[name].median():.2f}", "SD over splits": f"{other[name].std(ddof=1):.2f}",
              "range": f"{other[name].min():.2f} to {other[name].max():.2f}",
              "other splits below it": f"{int((other[name] < acc.loc[STUDY_SEED, name]).sum())} "
                                                      f"of {len(other)}",
              "household SE, split of the study": f"{100 * se[name]:.2f}" if name in se else "–"}
             for name, title in SETS.items()]
    reference = "real[train]"
    diffs = []
    for name, title in SETS.items():
        if name == reference:
            continue
        d = other[name] - other[reference]
        study = acc.loc[STUDY_SEED, name] - acc.loc[STUDY_SEED, reference]
        diffs.append({"difference": f"{title} - {SETS[reference]}",
                      f"split of the study (seed {STUDY_SEED})": f"{study:+.2f}",
                      f"mean of {len(other)} other splits": f"{d.mean():+.2f}",
                      "SD over splits": f"{d.std(ddof=1):.2f}", "range": f"{d.min():+.2f} to {d.max():+.2f}"})
    terms = (df[df.split_seed != STUDY_SEED].groupby("set", sort=False)[list(TERMS)].agg(["mean", "std"]))
    detail = pd.DataFrame({TERMS[k]: [f"{100 * terms.loc[n, (k, 'mean')]:.1f} ± {100 * terms.loc[n, (k, 'std')]:.1f}"
                                      for n in SETS] for k in TERMS}, index=list(SETS.values()))
    text = (pd.DataFrame(table).to_markdown(index=False) + "\n\n"
            f"Statistical accuracy (%) on the test households of the split of the study and of {args.splits} other "
            "random splits of the households (same fractions and stratification; a new split changes the training, "
            "validation and test households).  Real days of the training and of the validation households of each "
            "split, and a Gaussian copula fitted on its training households.  The last column is the half-sampling "
            "standard error of `uncertainty.md` for the split of the study, which keeps the training households "
            "fixed.\n\nDifferences within a split (percentage points):\n\n"
            + pd.DataFrame(diffs).to_markdown(index=False)
            + "\n\nTerms of the accuracy and worst label, mean ± SD over the other splits:\n\n"
            + detail.to_markdown() + "\n")
    if trained:
        text += trained_tables(df)
    (out / "split_sensitivity.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
