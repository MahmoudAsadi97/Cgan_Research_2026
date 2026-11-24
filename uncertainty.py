"""Household-level uncertainty of the test accuracy (stratified half-sampling).

    python uncertainty.py --runs runs --final final --faithful faithful --reps 100

All rows share the same 354 test households, and which households ended up in
the test set is the main source of uncertainty of the headline accuracy (the
spread over training seeds is much smaller).  In each of ``reps`` replicates
the test households of every technology group are split at random into two
halves A and B; every generated set (each seed of each model, kept fixed) and
the two real-data benchmarks are scored with ``metrics.accuracy`` on the real
days of A and of B (the generated days aligned with those real days).  The
standard error of the full-sample accuracy is SD(acc_A - acc_B) / 2, and the
same replicates give paired standard errors for differences between models.
Halves are used instead of a with-replacement bootstrap because the accuracy
depends on the number of real days (fewer days give a lower score): a bootstrap
shifts it down by about 3 points, while the half-to-half difference does not
depend on that shift.

Output: ``results/uncertainty.md`` with, for the final model (seed-averaged),
accuracy, household SE, seed SD, total SE = sqrt(SE_house^2 + SD_seed^2 / n),
the 95 % interval and the one-sided 95 % lower bound against 85 %, and paired
differences to the GitHub model and the real-data benchmarks.  ``--compare``
adds paired differences to other trained models (e.g. ``abl_quantile``) and
``--baselines`` to the classical generators of ``baselines.py`` (run it first).

``--stored results`` reuses the half-sample scores of an earlier run of this
script (``uncertainty_replicates.csv``, with the accuracies of
``all_runs.csv``): sets that are stored are not sampled again, so rows can be
added without the checkpoints of the models that are already in the table, for
instance the classical generators on a machine without the trained models.
The household halves depend only on the replicate number, and the stored
scores of the real-data benchmarks must equal the recomputed ones.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from data_prep import decode_label, load_profiles
from evaluate import EVAL_SEED, real_reference
from metrics import accuracy

_STATE = {}


def _init(real_X, real_label, real_meter, sources):
    _STATE.update(X=real_X, label=real_label, meter=real_meter, sources=sources)


def _acc(rows: np.ndarray, gen: np.ndarray) -> float:
    X, label = _STATE["X"], _STATE["label"]
    accs = []
    for lab in np.unique(label[rows]):
        r = rows[label[rows] == lab]
        accs.append(accuracy(X[r], gen[r])["accuracy"])
    return float(np.mean(accs))


def _replicate(rep: int) -> dict:
    rng = np.random.default_rng(10_000 + rep)
    meter, label = _STATE["meter"], _STATE["label"]
    group = decode_label(label)[1]
    half_a = np.zeros(len(label), bool)
    for g in np.unique(group):
        houses = rng.permutation(np.unique(meter[group == g]))
        half_a |= np.isin(meter, houses[: len(houses) // 2]) & (group == g)
    a, b = np.flatnonzero(half_a), np.flatnonzero(~half_a)
    return {name: (_acc(a, gen), _acc(b, gen)) for name, gen in _STATE["sources"].items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--final", default="final")
    ap.add_argument("--faithful", default="faithful")
    ap.add_argument("--compare", nargs="*", default=[], metavar="MODEL[:CKPT]",
                    help="other trained models to compare with the final model, e.g. abl_quantile improved_big")
    ap.add_argument("--baselines", nargs="*", default=[], help="classical baselines, e.g. copula gmm")
    ap.add_argument("--baseline-seeds", type=int, default=5)
    ap.add_argument("--stored", default=None, metavar="FOLDER",
                    help="folder with uncertainty_replicates.csv and all_runs.csv of an earlier run; "
                         "sets stored there are not sampled again")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--data", default="data/processed/fluvius_hourly.npz")
    args = ap.parse_args(argv)
    from train import load_run

    runs = Path(args.runs)
    splits = load_profiles(args.data)
    test = splits["test"]
    sources, point, stored = {}, {}, {}
    if args.stored:
        rep = pd.read_csv(Path(args.stored) / "uncertainty_replicates.csv", header=[0, 1])
        if len(rep) < args.reps:
            raise SystemExit(f"{args.stored} holds {len(rep)} replicates, fewer than --reps {args.reps}")
        stored = {name: (rep["A"][name].to_numpy()[: args.reps], rep["B"][name].to_numpy()[: args.reps])
                  for name in rep["A"].columns}
        published = pd.read_csv(Path(args.stored) / "all_runs.csv").set_index(["model", "seed"])["accuracy"]
        for name in stored:                 # "final[best]/seed0" is the row ("final [best]", "seed0")
            if not name.startswith(("real[", "baseline[")):
                point[name] = float(published[name.split("/")[0].replace("[", " ["), name.split("/")[1]])
    compare = [(m, c or "best") for m, _, c in (spec.partition(":") for spec in args.compare)]
    for model, ckpts in ((args.final, ["best"]), (args.faithful, ["best", "last"]), *((m, [c]) for m, c in compare)):
        for run_dir in sorted((runs / model).glob("seed*")):
            if not (run_dir / "train_summary.json").exists():
                continue
            for ckpt in ckpts:
                name = f"{model}[{ckpt}]/{run_dir.name}"
                if name in stored:
                    continue
                wrapped, cfg = load_run(run_dir, ckpt)
                sources[name] = wrapped.sample_kwh(test.label, EVAL_SEED + cfg["seed"]).astype(np.float32)
                point[name] = json.loads((run_dir / f"eval_test_{ckpt}.json").read_text())["accuracy"]
    titles = {}
    if args.baselines:
        import baselines as B

        cache = runs / "baselines"
        for name in args.baselines:
            k = B.selected_components(cache) if name == "gmm" else None
            tag = f"{name}{k or ''}"                     # the mixture is stored with its number of components
            titles[tag] = B.TITLES[name] + (f" ({k} components)" if k else "")
            model = None
            for seed in range(args.baseline_seeds):
                key = f"baseline[{tag}]/seed{seed}"
                point[key] = json.loads((cache / name / f"seed{seed}_eval_test.json").read_text())["accuracy"]
                if key in stored:
                    continue
                if name == "gmm" or model is None:
                    model = B.build(name, splits["train"], seed, k, cache)
                sources[key] = model.sample_kwh(test.label, EVAL_SEED + seed).astype(np.float32)
    for src in ("val", "train"):
        sources[f"real[{src}]"] = real_reference(splits, src, "test")[0].astype(np.float32)
        point[f"real[{src}]"] = json.loads((runs / "real_reference" / f"eval_test_{src}.json").read_text())["accuracy"]
    print(f"{len(sources)} generated sets to score, {len(stored)} stored, {args.reps} replicates", flush=True)

    with ProcessPoolExecutor(args.workers, initializer=_init,
                             initargs=(test.X, test.label, test.meter, sources)) as ex:
        reps = list(ex.map(_replicate, range(args.reps)))
    a = pd.DataFrame([{k: v[0] for k, v in r.items()} for r in reps])
    b = pd.DataFrame([{k: v[1] for k, v in r.items()} for r in reps])
    for name, (stored_a, stored_b) in stored.items():
        if name in a:                                    # the benchmarks: same halves and data as the stored run
            if not (np.allclose(a[name], stored_a, atol=1e-9) and np.allclose(b[name], stored_b, atol=1e-9)):
                raise SystemExit(f"the stored half-sample scores of {name} differ from the recomputed ones")
        a[name], b[name] = stored_a, stored_b
    order = [*stored, *(c for c in a.columns if c not in stored)]
    a, b = a[order], b[order]
    d = a - b                                            # half-to-half differences, one column per set

    def group(prefix):
        return [c for c in a.columns if c.startswith(prefix)]

    def se(cols_x, cols_y=None):
        """SE of mean(x) - mean(y) over the given sets (paired through the replicates)."""
        diff = d[cols_x].mean(axis=1) - (d[cols_y].mean(axis=1) if cols_y else 0)
        return float(diff.std(ddof=1) / 2)

    final = group(f"{args.final}[best]")
    acc = float(np.mean([point[c] for c in final]))
    sd_seed = float(np.std([point[c] for c in final], ddof=1)) if len(final) > 1 else 0.0
    se_house = se(final)
    se_total = float(np.sqrt(se_house ** 2 + sd_seed ** 2 / len(final)))
    rows = [{"quantity": f"accuracy, {args.final} (mean of {len(final)} seeds)", "estimate": 100 * acc,
             "SE households": 100 * se_house, "SD seeds": 100 * sd_seed, "SE total": 100 * se_total,
             "95% CI": f"{100 * (acc - 1.96 * se_total):.1f} to {100 * (acc + 1.96 * se_total):.1f}",
             "one-sided 95% lower bound": 100 * (acc - 1.645 * se_total)}]
    for other, label in ((group(f"{args.faithful}[last]"), "GitHub cGAN [last]"),
                         (group(f"{args.faithful}[best]"), "GitHub cGAN [best]"),
                         *((group(f"{m}[{c}]"), f"{m} [{c}]") for m, c in compare),
                         *((group(f"baseline[{tag}]"), title) for tag, title in titles.items()),
                         (["real[val]"], "real: validation households"),
                         (["real[train]"], "real: training households")):
        if not other:
            continue
        diff = acc - float(np.mean([point[c] for c in other]))
        s = se(final, other)
        rows.append({"quantity": f"{args.final} - {label}", "estimate": 100 * diff, "SE households": 100 * s,
                     "SD seeds": np.nan, "SE total": np.nan,
                     "95% CI": f"{100 * (diff - 1.96 * s):.1f} to {100 * (diff + 1.96 * s):.1f}",
                     "one-sided 95% lower bound": 100 * (diff - 1.645 * s)})
    table = pd.DataFrame(rows).round(2)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    note = (f"\nStratified half-sampling of the test households, {args.reps} replicates; differences are paired "
            "(same household splits for all sets).  Accuracies in percent, differences in percentage points.  "
            "The final-model SE for differences uses the household term only (seed SD is shown separately).\n")
    (out / "uncertainty.md").write_text(table.to_markdown(index=False) + "\n" + note, encoding="utf-8")
    pd.concat({"A": a, "B": b}, axis=1).to_csv(out / "uncertainty_replicates.csv", index=False)
    print(table.to_markdown(index=False))


if __name__ == "__main__":
    main()
