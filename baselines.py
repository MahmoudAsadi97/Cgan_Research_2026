"""Classical generators as baselines, evaluated with the protocol of the cGAN models.

    python baselines.py                              # fit, select on validation, score on the test households
    python baselines.py --models copula gmm --seeds 3

Every generator is fitted per label on the days of the training households and
then treated like a trained cGAN: for every label it produces as many days as
the test households have, and ``evaluate.evaluate`` computes the same metrics
(``results/baselines.md`` and ``.csv``).  Fitting and sampling use NumPy,
SciPy and scikit-learn on the CPU, so the table can be repeated without a GPU.
With the seeds of the final model in ``results/all_runs.csv`` (``report.py``),
``baselines.md`` also compares the final model with every generator, metric by
metric.

    indep_hours   every hour drawn on its own from the days of the label: exact hourly
                  distributions, no relation between the hours of a day
    gaussian      multivariate normal with the mean and covariance of the label (kWh)
    copula        Gaussian copula: the empirical distribution of every hour of the label, joined
                  by the correlation matrix of the normal scores
    gmm           Gaussian mixture with full covariances per label, in the per-hour Gaussianised
                  space of the final model (``train.Scaler("quantile")``) and with its floor of
                  0 kWh for labels without PV.  The number of components is chosen on the
                  validation households with the rule used for the cGAN candidates: highest
                  validation accuracy, the smaller mixture if two are within 0.1 point

A seed fixes the sampling noise and, for ``gmm``, the initialisation of the fit.  The other fits
are deterministic, so their spread over seeds is sampling noise only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from data_prep import N_LABELS, decode_label, load_profiles
from evaluate import EVAL_SEED, evaluate, summarise

ROOT = Path(__file__).resolve().parent
GMM_COMPONENTS = [1, 2, 5, 10, 20, 40, 80, 160]
SELECTION_MARGIN = 0.001          # 0.1 percentage point, as for the cGAN candidates
TITLES = {"indep_hours": "Independent hours", "gaussian": "Multivariate normal", "copula": "Gaussian copula",
          "gmm": "Gaussian mixture"}


class Baseline:
    """A generator fitted per label; ``sample_kwh(labels, seed)`` as ``train.Wrapped``."""

    def fit(self, train) -> "Baseline":
        for lab in range(N_LABELS):
            self.fit_label(lab, np.asarray(train.X[train.label == lab], np.float64))
        return self

    def fit_label(self, lab: int, X: np.ndarray):
        raise NotImplementedError

    def sample_label(self, lab: int, n: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def sample_kwh(self, labels, seed) -> np.ndarray:
        """One day per entry of ``labels`` (any order).  Every label has its own random stream, so the
        days of a label do not depend on which other labels are requested."""
        labels = np.asarray(labels)
        out = np.empty((len(labels), 24))
        for lab in np.unique(labels):
            m = labels == lab
            out[m] = self.sample_label(int(lab), int(m.sum()), np.random.default_rng([int(seed), int(lab)]))
        return out


class IndependentHours(Baseline):
    """Every hour drawn independently from the training days of the label."""

    def __init__(self):
        self.X = {}

    def fit_label(self, lab, X):
        self.X[lab] = X

    def sample_label(self, lab, n, rng):
        X = self.X[lab]
        return np.column_stack([X[rng.integers(0, len(X), n), t] for t in range(X.shape[1])])


class Gaussian(Baseline):
    """Multivariate normal with the mean and covariance of the label, in kWh."""

    def __init__(self):
        self.mean, self.chol = {}, {}

    def fit_label(self, lab, X):
        self.mean[lab] = X.mean(0)
        self.chol[lab] = np.linalg.cholesky(np.cov(X, rowvar=False))

    def sample_label(self, lab, n, rng):
        return self.mean[lab] + rng.standard_normal((n, len(self.mean[lab]))) @ self.chol[lab].T


class GaussianCopula(Baseline):
    """Empirical distribution of every hour of the label, Gaussian dependence between the hours.

    Normal scores z = Phi^-1((rank - 0.5) / n) of the training days give the correlation matrix; a
    sample is a normal vector with that correlation, mapped back hour by hour through the empirical
    quantile function (linear interpolation between the sorted training values, so generated values
    stay inside the observed range of the hour in the label).
    """

    def __init__(self):
        self.sorted, self.chol = {}, {}

    def fit_label(self, lab, X):
        n = len(X)
        z = norm.ppf((rankdata(X, axis=0) - 0.5) / n)
        corr = np.corrcoef(z, rowvar=False)
        self.chol[lab] = np.linalg.cholesky(corr + 1e-9 * np.eye(len(corr)))
        self.sorted[lab] = np.sort(X, axis=0)

    def sample_label(self, lab, n, rng):
        xs = self.sorted[lab]
        u = norm.cdf(rng.standard_normal((n, xs.shape[1])) @ self.chol[lab].T)
        p = (np.arange(len(xs)) + 0.5) / len(xs)
        return np.column_stack([np.interp(u[:, t], p, xs[:, t]) for t in range(xs.shape[1])])


class GaussianMixture(Baseline):
    """Gaussian mixture per label in the Gaussianised space of the final model, with its no-PV floor."""

    def __init__(self, n_components: int, scaler, seed: int = 0, nonneg_without_pv: bool = True):
        self.k, self.scaler, self.seed, self.nonneg_without_pv = n_components, scaler, seed, nonneg_without_pv
        self.weights, self.means, self.chols = {}, {}, {}

    def fit_label(self, lab, X):
        from sklearn.mixture import GaussianMixture as SkGMM

        gm = SkGMM(self.k, covariance_type="full", max_iter=500, random_state=self.seed).fit(self.scaler.transform(X))
        self.weights[lab], self.means[lab] = gm.weights_, gm.means_
        self.chols[lab] = np.linalg.cholesky(gm.covariances_)

    def sample_label(self, lab, n, rng):
        comp = rng.choice(self.k, size=n, p=self.weights[lab])
        z = self.means[lab][comp] + np.einsum("nij,nj->ni", self.chols[lab][comp],
                                              rng.standard_normal((n, self.means[lab].shape[1])))
        x = self.scaler.inverse(z)
        if self.nonneg_without_pv and int(decode_label(lab)[1]) & 1 == 0:
            x = np.maximum(x, 0.0)
        return x

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        labels = sorted(self.weights)
        np.savez_compressed(path, labels=labels, **{name: np.stack([getattr(self, name)[lab] for lab in labels])
                                                    for name in ("weights", "means", "chols")})

    def load(self, path: Path) -> "GaussianMixture":
        z = np.load(path)
        for name in ("weights", "means", "chols"):
            setattr(self, name, {int(lab): z[name][i] for i, lab in enumerate(z["labels"])})
        return self


def build(name: str, train, seed: int = 0, n_components: int | None = None, cache: Path | None = None) -> Baseline:
    """The fitted baseline ``name``.  ``gmm`` needs the number of components; with ``cache`` (a folder) its
    fitted mixtures are stored there and read back when the code and the data preparation are unchanged."""
    if name == "indep_hours":
        return IndependentHours().fit(train)
    if name == "gaussian":
        return Gaussian().fit(train)
    if name == "copula":
        return GaussianCopula().fit(train)
    if name == "gmm":
        from train import Scaler

        model = GaussianMixture(n_components, Scaler("quantile").fit(train.X), seed)
        path = cache and Path(cache) / "gmm" / f"fit_k{n_components}_seed{seed}_{code_hash()}.npz"
        if path and path.exists():
            return model.load(path)
        model.fit(train)
        if path:
            model.save(path)
        return model
    raise ValueError(f"unknown baseline {name!r}")


def selected_components(cache: Path) -> int:
    """The number of mixture components chosen by the last ``baselines.py`` run in this folder."""
    return int(json.loads((Path(cache) / "gmm_selection" / "selected.json").read_text())["components"])


def select_components(scores: dict[int, float], margin: float = SELECTION_MARGIN) -> int:
    """Highest validation accuracy; the smallest mixture within ``margin`` of it."""
    best = max(scores.values())
    return min(k for k, v in scores.items() if v >= best - margin)


def code_hash() -> str:
    """Fingerprint of what determines the generated days and their scores: the generators, the value
    transform and the validation of ``train.py``, the data preparation and the evaluation code."""
    import inspect

    import train

    h = hashlib.sha1()
    for obj in (Baseline, IndependentHours, Gaussian, GaussianCopula, GaussianMixture, train.Scaler, train.Validator):
        h.update(inspect.getsource(obj).encode())
    for f in ("metrics.py", "evaluate.py", "data_prep.py"):
        h.update((ROOT / f).read_bytes())
    return h.hexdigest()[:12]


def comparison(final: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Every ranked metric: the final model (mean ± SD over its seeds) next to the mean of every generator
    in ``rows``, marked ``+`` where every seed of the final model is better than every seed of the
    generator, ``-`` where every seed is worse, and ``=`` where the seeds overlap."""
    from report import COLUMNS, DIRECTION, _badness

    out = []
    for k, direction in DIRECTION.items():
        if direction is None or k not in final or k not in rows or final[k].isna().all():
            continue
        row = {"metric": COLUMNS.get(k, k.replace("criteria.", "")),
               "better": f"closer to {direction[1]}" if isinstance(direction, tuple) else
               {+1: "higher", -1: "lower"}[direction],
               "final": f"{final[k].mean():.4g} ± {final[k].std():.2g}"}
        bad_final = _badness(final[k], direction)
        for model, g in rows.groupby("model", sort=False):
            bad = _badness(g[k], direction)
            mark = "+" if bad_final.max() < bad.min() else "-" if bad_final.min() > bad.max() else "="
            row[model] = f"{g[k].mean():.4g} {mark}"
        out.append(row)
    return pd.DataFrame(out)


def _cached(path: Path, fingerprint: dict, compute):
    """``compute()`` (a dict), cached in ``path`` together with ``fingerprint``."""
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get("_fingerprint") == fingerprint:
            return cached
    out = {**compute(), "_fingerprint": fingerprint}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    return out


def main(argv=None):
    from report import table
    from train import DEFAULTS, Validator

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(TITLES), choices=list(TITLES))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--components", nargs="+", type=int, default=GMM_COMPONENTS,
                    help="candidate numbers of mixture components (selected on the validation households)")
    ap.add_argument("--runs", default="runs", help="folder for the cached evaluations (runs/baselines)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--trained", default=None, metavar="CSV",
                    help="all_runs.csv with the seeds of the trained models (default: in --out)")
    ap.add_argument("--final", default="final", help="trained model compared with the generators")
    ap.add_argument("--data", default="data/processed/fluvius_hourly.npz")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cache, out = Path(args.runs) / "baselines", Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = load_profiles(args.data)
    train, test = splits["train"], splits["test"]
    code = code_hash()
    notes = []

    n_components = None
    if "gmm" in args.models:      # model selection on the validation households, seed 0 (no test data)
        validator = Validator(splits["val"], DEFAULTS["val_max_per_label"], DEFAULTS["val_seed"])
        sel = {}
        for k in args.components:
            r = _cached(cache / "gmm_selection" / f"k{k}.json", {"code": code, "k": k},
                        lambda: dict(zip(("val_accuracy", "val_min_label"),
                                         validator.score(build("gmm", train, 0, k, cache)))))
            sel[k] = r
            print(f"gmm, {k} components: validation accuracy {100 * r['val_accuracy']:.2f} %", flush=True)
        n_components = select_components({k: r["val_accuracy"] for k, r in sel.items()})
        (cache / "gmm_selection" / "selected.json").write_text(json.dumps({"components": n_components}))
        selection = pd.DataFrame({"components": list(sel),
                                  "validation accuracy (%)": [f"{100 * r['val_accuracy']:.2f}" for r in sel.values()],
                                  "worst label (%)": [f"{100 * r['val_min_label']:.2f}" for r in sel.values()],
                                  "selected": ["yes" if k == n_components else "" for k in sel]})
        notes.append("Number of mixture components, selected on the validation households (seed 0; highest "
                     "validation accuracy, the smaller mixture if two are within 0.1 point):\n\n"
                     + selection.to_markdown(index=False) + "\n")

    rows = []
    for name in args.models:
        model = None if name == "gmm" else build(name, train)
        for seed in range(args.seeds):
            fp = {"code": code, "eval_seed": EVAL_SEED, "seed": seed,
                  "components": n_components if name == "gmm" else None}

            def compute():
                m = build("gmm", train, seed, n_components, cache) if name == "gmm" else model
                df = evaluate(m.sample_kwh(test.label, EVAL_SEED + seed), test, train)
                (cache / name).mkdir(parents=True, exist_ok=True)
                df.to_csv(cache / name / f"seed{seed}_eval_test.csv", index=False)
                return summarise(df)

            summary = _cached(cache / name / f"seed{seed}_eval_test.json", fp, compute)
            title = TITLES[name] + (f" ({n_components} components)" if name == "gmm" else "")
            rows.append({"model": title, "seed": f"seed{seed}", **summary})
            print(f"{title} seed {seed}: accuracy {100 * summary['accuracy']:.2f} %", flush=True)

    scores = pd.DataFrame(rows).drop(columns="_fingerprint")
    scores.to_csv(out / "baselines.csv", index=False)
    trained = Path(args.trained) if args.trained else out / "all_runs.csv"
    if trained.exists():
        final = pd.read_csv(trained)
        final = final[final.model == f"{args.final} [best]"]
        if len(final):
            notes.append(f"`{args.final}` (validation-selected checkpoints, {len(final)} seeds, `{trained.name}`) "
                         "against every generator.  Values are fractions; + every seed of the final model is "
                         "better than every seed of the generator, - every seed is worse, = the seeds overlap:\n\n"
                         + comparison(final, scores).to_markdown(index=False) + "\n")
    t = table(rows)
    text = (t.to_markdown() + "\n\nClassical generators fitted per label on the training households and scored on the "
            f"held-out test households with the protocol of `results/summary.md` (as many generated days per label as "
            f"the test households have; mean ± SD over {args.seeds} seeds).  A seed fixes the sampling noise and, for "
            "the mixture, the initialisation of the fit; the other fits are deterministic, so their SD is sampling "
            "noise only.  The mixture uses the per-hour value transform and the no-PV floor of the final model.\n")
    (out / "baselines.md").write_text(text + "".join("\n" + n for n in notes), encoding="utf-8")
    print(t.to_markdown())


if __name__ == "__main__":
    main()
