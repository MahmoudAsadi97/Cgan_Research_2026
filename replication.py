"""Compare a re-training on other hardware with the published runs, and add its extra seeds.

    python run_jobs.py jobs/replication.txt --runs runs_cpu      # improved and final, seed 0
    python run_jobs.py jobs/extra_seeds.txt --runs runs_cpu      # seeds 1-2 of abl_shuffle and abl_quantile
    python replication.py --runs runs_cpu

The result tables come from runs on a CUDA device.  A seed fixes a run on one type
of device, but the CPU and CUDA generators draw different numbers, so the same
configuration and seed trained on the CPU is a new, independent run of that
configuration.  This script evaluates every finished run of ``--runs`` on the test
households (``report.evaluate_run``) and writes

    replication_runs.csv   all metrics of every re-trained run and checkpoint
    replication.md         * configurations with published seeds: the published mean, SD and range over
                             seeds next to every re-trained run, with its distance to the published mean
                             in published SDs;
                           * steps of the ablation ladder with seeds beyond the published ones: the
                             published and the new seeds together.

Published values are read from ``all_runs.csv`` and, for the steps of the ablation ladder, from
``ablation_runs.csv`` (``report.py --ablation``) or, where that file does not exist, from the rounded cells
of ``ablation.csv``.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from report import COLUMNS, STEPS, evaluate_run, finished_runs

METRICS = ["accuracy", "accuracy.min_label", "mean_profile", "std_profile", "kpi_mean", "kpi_std", "hourly_dist",
           "kpi_dist", "paper.fid", "profile.rfd", "profile.corr_mae", "profile.c2st_auc", "profile.coverage",
           "profile.dcr_ratio", "physical.neg_share"]
PERCENT = {"accuracy", "accuracy.min_label", "accuracy.share_labels_ge_85", "mean_profile", "std_profile", "kpi_mean",
           "kpi_std", "hourly_dist", "kpi_dist"}


def _cell(text):
    """``'87.5 ± 0.2'`` -> (87.5, 0.2); ``'66.1'`` -> (66.1, nan)."""
    mean, _, sd = str(text).partition("±")
    return float(mean), float(sd) if sd.strip() else np.nan


class Published:
    """Seeds of one configuration and checkpoint in the published tables: the values of every seed
    (``runs``: seeds x metrics) or, from a rounded table, the number of seeds with their mean and SD."""

    def __init__(self, runs: pd.DataFrame | None = None, n: int = 0, mean=None, sd=None):
        self.runs = runs
        self.n = len(runs) if runs is not None else n
        self.mean = runs.mean() if runs is not None else mean
        self.sd = runs.std(ddof=1) if runs is not None else sd

    def combined(self, metric: str, new: list[float]) -> tuple[float, float]:
        """Mean and SD of the published seeds together with ``new`` values."""
        if self.runs is not None:
            v = np.r_[self.runs[metric].to_numpy(), new]
            return float(v.mean()), float(v.std(ddof=1))
        m, s, n, new = self.mean[metric], self.sd[metric], self.n, np.asarray(new, float)
        total = n + len(new)
        mean = (n * m + new.sum()) / total
        squares = (0.0 if n < 2 else (n - 1) * s ** 2) + n * m ** 2 + (new ** 2).sum()
        return float(mean), float(np.sqrt(max(squares - total * mean ** 2, 0.0) / (total - 1)))


def read_published(results: Path) -> dict[tuple[str, str], Published]:
    """(model, checkpoint) -> published seeds."""
    out = {}
    runs = pd.read_csv(results / "all_runs.csv")
    runs = runs[runs.seed.astype(str).str.startswith("seed")]
    for name, g in runs.groupby("model", sort=False):
        model, ckpt = name[:-1].split(" [")
        out[(model, ckpt)] = Published(g.set_index("seed")[[m for m in METRICS if m in g]].astype(float))
    raw = results / "ablation_runs.csv"
    if raw.exists():
        for model, g in pd.read_csv(raw).groupby("model", sort=False):
            out.setdefault((model, "best"), Published(g.set_index("seed")[[m for m in METRICS if m in g]].astype(float)))
    elif (results / "ablation.csv").exists():
        table = pd.read_csv(results / "ablation.csv", index_col=0)
        for model, row in table.iterrows():
            if (model, "best") in out or model not in STEPS:
                continue
            cells = {m: _cell(row[COLUMNS[m]]) for m in METRICS if COLUMNS[m] in row}
            scale = {m: 0.01 if m in PERCENT else 1.0 for m in cells}
            out[(model, "best")] = Published(n=int(row["Seeds"]),
                                             mean=pd.Series({m: v[0] * scale[m] for m, v in cells.items()}),
                                             sd=pd.Series({m: v[1] * scale[m] for m, v in cells.items()}))
    return out


def fmt(metric: str, value: float, sd: float = np.nan, digits: int = 2) -> str:
    if not np.isfinite(value):
        return "–"
    if metric in PERCENT:
        return f"{100 * value:.{digits}f}" + (f" ± {100 * sd:.{digits}f}" if np.isfinite(sd) else "")
    return f"{value:.3g}" + (f" ± {sd:.2g}" if np.isfinite(sd) else "")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs_cpu", help="folder with the re-trained runs")
    ap.add_argument("--results", default="results", help="folder with the published tables")
    ap.add_argument("--out", default=None, help="output folder (default: --results)")
    ap.add_argument("--ckpts", nargs="+", default=["best", "last"])
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    runs, results = Path(args.runs), Path(args.results)
    out = Path(args.out or args.results)
    out.mkdir(parents=True, exist_ok=True)
    published = read_published(results)

    cache, rows, devices = {}, [], set()
    models = sorted(p.name for p in runs.iterdir() if p.is_dir() and any(p.glob("seed*/train_summary.json")))
    if not models:
        raise SystemExit(f"no finished run in {runs}")
    for model in models:
        for run_dir in finished_runs(runs, model):
            env_file = run_dir / "environment.json"
            env = json.loads(env_file.read_text())[0] if env_file.exists() else {}
            devices.add(f"{env.get('device', 'unknown device')}, torch {env.get('torch', '?')}, "
                        f"{env.get('threads', '?')} thread(s)")
            for ckpt in args.ckpts:
                summary = evaluate_run(run_dir, ckpt, cache)
                rows.append({"model": model, "ckpt": ckpt, "seed": int(run_dir.name[4:]),
                             "device": env.get("device", ""), **{k: v for k, v in summary.items() if k != "_fingerprint"}})
    new = pd.DataFrame(rows)
    new.to_csv(out / "replication_runs.csv", index=False)

    text = ["# Re-training on other hardware\n",
            f"Runs of `{runs.name}/` ({'; '.join(sorted(devices))}), evaluated on the held-out test households with "
            "the protocol of `summary.md`.  The published runs were trained on a CUDA device; the CPU and CUDA "
            "generators draw different random numbers, so every run here is a new, independent run of its "
            "configuration.  Accuracies in percent.\n"]

    # 1. configurations with published seeds
    first = True
    for (model, ckpt), g in new.groupby(["model", "ckpt"], sort=False):
        pub = published.get((model, ckpt))
        g = g[g.seed < (pub.n if pub else 0)]
        if pub is None or pub.n < 2 or g.empty:
            continue
        if first:
            text.append("## Published seeds and the re-trained runs\n\nSD and range over the published seeds; the last "
                        "column is the distance of the re-trained run to the published mean in published SDs.\n")
            first = False
        table = []
        for m in [m for m in METRICS if m in pub.mean.index]:
            row = {"metric": COLUMNS[m], f"published, {pub.n} seeds": fmt(m, pub.mean[m], pub.sd[m])}
            if pub.runs is not None:
                row["published range"] = f"{fmt(m, pub.runs[m].min())} to {fmt(m, pub.runs[m].max())}"
            for _, r in g.iterrows():
                row[f"re-trained, seed {r.seed}"] = fmt(m, r[m])
                z = (r[m] - pub.mean[m]) / pub.sd[m] if pub.sd[m] > 0 else np.nan
                row[f"distance, seed {r.seed} (SD)"] = f"{z:+.1f}" if np.isfinite(z) else "–"
            table.append(row)
        rounded = "" if pub.runs is not None else " (published values from the rounded cells of `ablation.csv`)"
        text.append(f"### `{model}` [{ckpt}]{rounded}\n\n" + pd.DataFrame(table).to_markdown(index=False) + "\n")

    # 2. steps with seeds beyond the published ones
    first = True
    for model, g in new[new.ckpt == "best"].groupby("model", sort=False):
        pub = published.get((model, "best"))
        extra = g[g.seed >= (pub.n if pub else 0)].sort_values("seed")
        if pub is None or extra.empty:
            continue
        if first:
            text.append("## Steps of the ablation ladder with additional seeds\n\nValidation-selected checkpoints.  The "
                        "published seeds were trained on CUDA, the additional seeds here.\n")
            first = False
        table = []
        for m in [m for m in METRICS if m in pub.mean.index]:
            row = {"metric": COLUMNS[m],
                   f"published, {pub.n} seed{'s' * (pub.n > 1)}": fmt(m, pub.mean[m], pub.sd[m], 1 if pub.runs is None else 2)}
            for _, r in extra.iterrows():
                row[f"seed {r.seed}"] = fmt(m, r[m])
            mean, sd = pub.combined(m, extra[m].tolist())
            row[f"all {pub.n + len(extra)} seeds"] = fmt(m, mean, sd)
            table.append(row)
        text.append(f"### `{model}`: {STEPS.get(model, '')}\n\n" + pd.DataFrame(table).to_markdown(index=False) + "\n")

    (out / "replication.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
