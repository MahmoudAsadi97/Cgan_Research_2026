"""Compare the results with the published figures of Li, Dong & Qiu (2025).

    python compare_paper.py --pdf Paper_1.pdf --reference-gen "cGAN_EB/cGAN Generation"

The paper reports its validation as heatmaps (Figs. 5-6: differences of mean and
standard deviation of the key parameters; Figs. 9-10: "KL divergence"; Figs.
11-12: FID), against the training data.  This script uses the definitions of the
published cGAN_EB code (``metrics.reference_key_parameters``,
``metrics.reference_kld``).  Their near-peak and near-base are the 2nd-highest and
2nd-lowest hour, while the generated days in the paper's Figs. 7-8 are the 3rd-highest
and 4th-lowest hour of the published days; ``--ranks 21 3`` repeats everything with
those ranks (the real days are not public, so the ranks of the heatmaps themselves
cannot be tested).  The script

1. extracts the heatmap images from the PDF (``pdfimages``, poppler) and
   digitises every cell: the rows of the grid follow from the colour bar, which
   spans the height of the heatmap; the column centres from the 32 rotated
   tick labels; the value of a cell is the centre of the band of the figure's
   own colour bar with the closest colour (median colour of the cell's centre),
   so JPEG and colour-profile shifts are calibrated out.  Cells at the ends of
   the colour scale are censored (true value at least that large); their share
   is reported;
2. computes the same quantities for every seed of the models of this repository
   with the same definitions, under the paper's protocol (generated vs the
   training households, 2,000 generated days per label) and against the held-out
   test households;
3. writes ``paper_comparison.md`` (mean and SD over seeds per criterion),
   ``paper_comparison.csv`` (every value per seed, benchmark and control),
   ``paper_digitised.csv`` (every digitised cell) and ``paper_cells_*.csv``
   (every cell per seed and benchmark).

The generated days published with the reference (``--reference-gen``) are the
days of the paper's Figs. 3-4; with the digitised differences they give the load
scale of its real data, which are not public.

The comparison is between different data sets (110 US households with PV and
EV charger levels vs 2,367 Flemish households with PV / heat pump / EV flags);
it shows whether the levels the paper reports as good are reached here, not a
ranking on common data (the GitHub model on our data is that ranking).
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from data_prep import N_LABELS, load_profiles
from metrics import reference_key_parameters, reference_kld

DIFF_ROWS = ["PeakLoad_mean", "PeakLoad_std", "BaseLoad_mean", "BaseLoad_std", "PeakDuration_mean",
             "PeakDuration_std", "RiseTime_mean", "RiseTime_std", "FallTime_mean", "FallTime_std"]
KEY_ROWS = ["PeakLoad", "BaseLoad", "PeakDuration", "RiseTime", "FallTime"]
LOAD_ROWS, TIME_ROWS = KEY_ROWS[:2], KEY_ROWS[2:]
TICKS = {(-1.5, 1.5): 7, (0.0, 1.0): 6, (0.0, 10.0): 6}     # tick labels of each colour bar
# Positions of the near-peak and near-base load in the sorted day: of the published code, of the paper's Figs. 7-8.
CODE_RANKS, FIGURE_RANKS = (22, 1), (21, 3)
SHUFFLES = 40                                               # shuffles of the control of the code's "KLD"
ranks = CODE_RANKS                                          # set by --ranks
# Heatmaps in the PDF: page, index of the image on that page, figure, quantity, day type, rows, colour-bar range.
HEATMAPS = [
    (9, 0, "Fig. 5", "difference", "Weekday", DIFF_ROWS, (-1.5, 1.5)),
    (9, 1, "Fig. 6", "difference", "Weekend", DIFF_ROWS, (-1.5, 1.5)),
    (11, 0, "Fig. 9", "KLD", "Weekday", KEY_ROWS, (0.0, 1.0)),
    (11, 1, "Fig. 10", "KLD", "Weekend", KEY_ROWS, (0.0, 1.0)),
    (12, 0, "Fig. 11", "FID", "Weekday", KEY_ROWS, (0.0, 10.0)),
    (12, 1, "Fig. 12", "FID", "Weekend", KEY_ROWS, (0.0, 10.0)),
]
CODE_KEYS = {"Peak_load": "PeakLoad", "Base_load": "BaseLoad", "Peak_duration": "PeakDuration",
             "Rise": "RiseTime", "Fall": "FallTime"}
# Groups of the reference, in the order of its labels (and of the columns of the heatmaps).
REF_GROUPS = ["No PV EV", "PV", "EV_lv1", "EV_lv2", "EV_both", "PV+EV_lv1", "PV+EV_lv2", "PV+EV_both"]
REF_SEASONS = ["Spring", "Summer", "Fall", "Winter"]
PROTOCOLS = {"paper": "paper protocol", "test": "held-out test households"}


# ---------------------------------------------------------------------------
# digitising
# ---------------------------------------------------------------------------
def extract_images(pdf: Path, page: int, folder: Path) -> list[Path]:
    prefix = folder / f"p{page}"
    subprocess.run(["pdfimages", "-png", "-f", str(page), "-l", str(page), str(pdf), str(prefix)], check=True)
    return sorted(folder.glob(f"p{page}-*.png"))


def _runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Start / end (exclusive) of runs of True of at least ``min_len``."""
    out, start = [], None
    for i, v in enumerate(np.r_[mask, False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    return out


def find_colorbar(img: np.ndarray, vrange: tuple[float, float]):
    """x0, x1 of the colour bar and the rows y0, y1 of vmax and vmin.

    The bar is the right-most block of columns that are off-white over most of their
    height.  Its ends can be (almost) white, so the value scale comes from the tick
    labels to its right: their vertical centres are evenly spaced from vmax (top) to
    vmin (bottom), and a straight-line fit gives the rows of vmax and vmin.
    """
    h, w, _ = img.shape
    s = img.astype(int).sum(-1)
    frac = (s < 3 * 253).mean(0)
    cols = np.flatnonzero(frac > 0.5)
    cols = cols[cols > w // 2]
    x1 = cols[-1] + 1
    x0 = x1 - 1
    while x0 - 1 in set(cols):
        x0 -= 1
    labels = (s[:, x1 + 3:] < 3 * 140).any(1)
    centres = np.array([(a + b - 1) / 2 for a, b in _runs(labels, 5)])
    if len(centres) != TICKS[vrange]:
        raise ValueError(f"found {len(centres)} colour-bar tick labels instead of {TICKS[vrange]}")
    vmin, vmax = vrange
    values = np.linspace(vmax, vmin, len(centres))
    slope, intercept = np.polyfit(values, centres, 1)
    return x0, x1, intercept + slope * vmax, intercept + slope * vmin


def column_centres(img: np.ndarray, y_bottom: int, x_max: int, n: int = 32) -> np.ndarray:
    """Centres of the ``n`` rotated tick labels below the heatmap (dark pixels)."""
    band = img[y_bottom + 5:, :x_max].astype(int)
    dark = (band.sum(-1) < 3 * 140).any(0)
    runs = _runs(dark, 3)
    # merge fragments closer than 4 px
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < 4:
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append(r)
    if len(merged) != n:
        raise ValueError(f"found {len(merged)} tick labels instead of {n}")
    return np.array([(a + b - 1) / 2 for a, b in merged])


def digitise(img: np.ndarray, n_rows: int, vrange: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Values (n_rows x 32) and a censoring mask (True: at the end of the colour scale)."""
    cx0, cx1, y_top, y_bottom = find_colorbar(img, vrange)
    cy0, cy1 = int(np.ceil(y_top)) + 1, int(np.floor(y_bottom)) - 1
    bar = np.median(img[cy0:cy1, cx0 + (cx1 - cx0) // 4:cx1 - (cx1 - cx0) // 4].astype(float), axis=1)
    vmin, vmax = vrange
    bar_vals = vmax - (np.arange(cy0, cy1) - y_top) / (y_bottom - y_top) * (vmax - vmin)
    xs = column_centres(img, int(y_bottom), cx0 - 20)
    step_x = np.median(np.diff(xs))
    step_y = (y_bottom - y_top) / n_rows
    cy0 = y_top
    values = np.empty((n_rows, len(xs)))
    for i in range(n_rows):
        yc = cy0 + (i + 0.5) * step_y
        for j, xc in enumerate(xs):
            patch = img[int(yc - 0.25 * step_y):int(yc + 0.25 * step_y),
                        int(xc - 0.25 * step_x):int(xc + 0.25 * step_x)].reshape(-1, 3).astype(float)
            colour = np.median(patch, axis=0)
            dist = ((bar - colour) ** 2).sum(1)
            values[i, j] = bar_vals[dist == dist.min()].mean()      # centre of the matching colour band
    tol = 0.02 * (vmax - vmin)
    censored = (values >= vmax - tol) | ((values <= vmin + tol) & (vmin < 0))
    return values, censored


def digitise_paper(pdf: Path) -> pd.DataFrame:
    from PIL import Image

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        cache = {}
        for page, k, fig, quantity, daytype, names, vrange in HEATMAPS:
            if page not in cache:
                cache[page] = extract_images(pdf, page, Path(tmp))
            img = np.asarray(Image.open(cache[page][k]).convert("RGB"))
            values, censored = digitise(img, len(names), vrange)
            for i, name in enumerate(names):
                for j in range(values.shape[1]):
                    rows.append({"figure": fig, "quantity": quantity, "daytype": daytype, "row": name,
                                 "column": j, "value": values[i, j], "censored": bool(censored[i, j])})
    return pd.DataFrame(rows)


def published_days(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    """The 2,000 generated days per label published with the reference (the days of the paper's
    Figs. 3-4) and their labels, in the label order of this repository."""
    days, labels = [], []
    for d, daytype in enumerate(("Weekday", "Weekend")):
        for g, group in enumerate(REF_GROUPS):
            for s, season in enumerate(REF_SEASONS):
                x = pd.read_csv(folder / f"{daytype} - {group} {season}_gen.csv").to_numpy(np.float64)
                days.append(x)
                labels.append(np.full(len(x), 32 * d + 4 * g + s))
    return np.concatenate(days), np.concatenate(labels)


def key_parameters(X: np.ndarray) -> dict[str, np.ndarray]:
    return reference_key_parameters(X, *ranks)


def paper_load_scale(days, labels, paper: pd.DataFrame) -> pd.DataFrame:
    """Near-peak load per label of the paper's real data: mean and SD, at the rank of its figures, of the
    published generated days minus the digitised differences of Figs. 5-6 (generated - real).  The daily
    energy is that of the generated days."""
    differences = paper[paper.quantity == "difference"].set_index(["daytype", "column", "row"]).value
    rows = []
    for lab in range(N_LABELS):
        x = days[labels == lab]
        near_peak = np.sort(x, axis=1)[:, FIGURE_RANKS[0]]
        key = {"daytype": "Weekday" if lab < 32 else "Weekend", "row": "PeakLoad", "column": lab % 32}
        delta = differences.loc[key["daytype"], key["column"]]
        rows.append({"quantity": "scale", **key, "value": near_peak.mean() - delta["PeakLoad_mean"],
                     "energy": x.sum(1).mean()})
        rows.append({"quantity": "scale_std", **key, "value": near_peak.std(ddof=1) - delta["PeakLoad_std"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# our values
# ---------------------------------------------------------------------------
def our_cells(real, gen, real_label, gen_label) -> pd.DataFrame:
    rows = []
    for lab in range(N_LABELS):
        kr = key_parameters(real[real_label == lab])
        kg = key_parameters(gen[gen_label == lab])
        kld = reference_kld(kr, kg)
        daytype = "Weekday" if lab < 32 else "Weekend"
        for code, title in CODE_KEYS.items():
            r, g = kr[code], kg[code]
            vals = {
                ("difference", f"{title}_mean"): g.mean() - r.mean(),
                ("difference", f"{title}_std"): g.std(ddof=1) - r.std(ddof=1),
                ("KLD", title): kld[code],
                ("FID", title): (r.mean() - g.mean()) ** 2 + (r.std(ddof=1) - g.std(ddof=1)) ** 2,
            }
            if title == "PeakLoad":
                vals[("scale", title)] = r.mean()
                vals[("scale_std", title)] = r.std(ddof=1)
            for (quantity, row), v in vals.items():
                rows.append({"quantity": quantity, "daytype": daytype, "row": row, "column": lab % 32,
                             "value": float(v)})
    return pd.DataFrame(rows)


def model_cells(run_dir: Path, ckpt: str, splits, protocol: str, n_per_label=2000, seed=0):
    """The cells of one run, with the generated days and their labels."""
    from evaluate import EVAL_SEED
    from train import load_run

    wrapped, cfg = load_run(run_dir, ckpt)
    if protocol == "paper":       # generated vs training households, 2,000 generated days per label (in-sample)
        real = splits["train"]
        gen_label = np.repeat(np.arange(N_LABELS), n_per_label)
    else:                         # generated vs held-out test households, as many days as the test split
        real = splits["test"]
        gen_label = real.label
    gen = wrapped.sample_kwh(gen_label, EVAL_SEED + cfg["seed"] + seed)
    return our_cells(real.X.astype(np.float64), gen, real.label, gen_label), gen, gen_label


def kld_control(days, labels, seed=0) -> dict:
    """The code's "KLD" of days against a shuffled copy of themselves: the two samples are identical,
    so a divergence would be 0."""
    rng = np.random.default_rng(seed)
    values = []
    for lab in range(N_LABELS):
        k = key_parameters(days[labels == lab])
        perm = rng.permutation(len(k["Peak_load"]))
        values += list(reference_kld(k, {name: v[perm] for name, v in k.items()}).values())
    values = pd.Series(values)
    return {"KLD: median": values.fillna(np.inf).median(), "KLD: share ≤ 0.3": (values <= 0.3).mean()}


def stand_in_cells(splits, flip: int, n_per_label=2000, seed=0) -> pd.DataFrame:
    """Days that are wrong by construction in place of generated days (paper protocol): training days
    of the label with bits of the label flipped (1-3: the season, 4: PV, 8: heat pump, 16: EV, 32: the day
    type) or, with ``flip`` 0, flat days: the mean of a training day of the label plus noise of SD
    0.001 kWh (on an exactly flat day every hour ties and the key parameters degenerate)."""
    train = splits["train"]
    X, label = train.X.astype(np.float64), train.label
    rng = np.random.default_rng(seed)
    gen_label = np.repeat(np.arange(N_LABELS), n_per_label)
    idx = np.concatenate([rng.permutation(np.flatnonzero(label == (lab ^ flip)))[:n_per_label]
                          for lab in range(N_LABELS)])
    gen = X[idx]
    if flip == 0:
        gen = gen.mean(1, keepdims=True).repeat(gen.shape[1], 1) + rng.normal(0, 1e-3, gen.shape)
    return our_cells(X, gen, label, gen_label)


def benchmark_cells(splits, protocol: str, source="val", n_per_label=2000, seed=0) -> pd.DataFrame:
    """Real days of ``source`` households in place of generated days, at most 2,000 per label with the
    paper's protocol (for ``source`` train a sample of the very days they are compared with): what a pool
    of that size scores with these definitions on this data."""
    from evaluate import real_reference

    if protocol == "paper":
        rng = np.random.default_rng(seed)
        pool = splits[source]
        idx = np.concatenate([rng.permutation(np.flatnonzero(pool.label == lab))[:n_per_label]
                              for lab in range(N_LABELS)])
        real = splits["train"]
        return our_cells(real.X.astype(np.float64), pool.X[idx].astype(np.float64), real.label, pool.label[idx])
    gen, _ = real_reference(splits, source, "test")
    test = splits["test"]
    return our_cells(test.X.astype(np.float64), gen.astype(np.float64), test.label, test.label)


def summarise(cells: pd.DataFrame, relative_bar: float, censored: pd.Series | None = None) -> dict:
    """Criteria quoted by the paper, over all 64 labels x 5 key parameters, and views for comparing across
    data sets: the time rows alone, the groups with the same label in both data sets (no technology,
    columns 0-3; PV only, columns 4-7) and the load differences relative to the near-peak of the label.
    A "KLD" that is undefined fails its level, as in ``evaluate.paper_criteria``, and ranks last in the
    median."""
    out = {}
    d = cells[cells.quantity == "difference"]
    for kind, unit_rows in (("load (kWh)", LOAD_ROWS), ("time (h)", TIME_ROWS)):
        for stat in ("mean", "std"):
            sel = d[d.row.isin([f"{r}_{stat}" for r in unit_rows])].value.abs()
            out[f"|Δ {stat}| {kind}: median"] = sel.median()
            out[f"|Δ {stat}| {kind}: share < 1"] = (sel < 1).mean()
    time = d[d.row.isin([f"{r}_mean" for r in TIME_ROWS])]
    out["|Δ mean| time (h): share of labels with all three < 1"] = (
        time.groupby(["daytype", "column"]).value.apply(lambda v: (v.abs() < 1).all()).mean())
    out["|Δ mean| time (h), no technology and PV only: share < 1"] = (time[time.column < 8].value.abs() < 1).mean()
    no_technology = time[time.column < 4].value.abs()
    out["|Δ mean| time (h), no technology: share < 1"] = (no_technology < 1).mean()
    out["|Δ mean| time (h), no technology: median"] = no_technology.median()
    near_peak = cells[cells.quantity == "scale"][["daytype", "column", "value"]].rename(columns={"value": "near_peak"})
    for row, name in (("PeakLoad", "near-peak"), ("BaseLoad", "near-base")):
        for stat in ("mean", "std"):
            sel = d[d.row == f"{row}_{stat}"].merge(near_peak, on=["daytype", "column"])
            out[f"|Δ {stat}| {name} / near-peak: share < scale-matched bar"] = (
                (sel.value.abs() / sel.near_peak) < relative_bar).mean()
    spread = cells[cells.quantity == "scale_std"].merge(near_peak, on=["daytype", "column"])
    out["near-peak: median of std / mean"] = (spread.value / spread.near_peak).median()
    out["near-peak, no technology: median of std / mean"] = (
        (spread.value / spread.near_peak)[spread.column < 4].median())
    k = cells[cells.quantity == "KLD"]
    failed = k[~(k.value <= 0.3)]
    # bit 0 of the group is PV here; the paper's PV groups are 1, 5, 6, 7, so this count holds for it only at 0
    pv_near_base = (failed.row == "BaseLoad") & ((failed.column // 4) % 2 == 1)
    out["KLD: median"] = k.value.fillna(np.inf).median()
    out["KLD: share ≤ 0.3"] = (k.value <= 0.3).mean()
    out["KLD: undefined values (of 320)"] = float(k.value.isna().sum())
    out["KLD: failures outside the near-base of PV labels"] = float((~pv_near_base).sum())
    f = cells[cells.quantity == "FID"]
    out["FID: median"] = f.value.median()
    out["FID: share < 5"] = (f.value < 5).mean()
    time_fid = f[f.row.isin(TIME_ROWS)].value
    out["FID, time rows: median"] = time_fid.median()
    out["FID, time rows: share < 5"] = (time_fid < 5).mean()
    out["share of cells at the end of the colour scale"] = censored.mean() if censored is not None else np.nan
    return out


def cell_text(values: list[float], share: bool) -> str:
    """Mean and SD over seeds; a single value (one seed, the paper, the real-data benchmarks) stands alone."""
    v = np.asarray(values, float)[np.isfinite(values)] * (100 if share else 1)
    if len(v) == 0:
        return "–"
    text = f"{v.mean():.1f}" if share else f"{v.mean():.3g}"
    if len(v) > 1:
        sd = 0.0 if np.ptp(v) == 0 else v.std(ddof=1)       # no rounding noise on a value fixed by the real data
        text += f" ± {sd:.1f}" if share else f" ± {sd:.2g}"
    return text + (" %" if share else "")


def main(argv=None):
    from report import DISPLAY, finished_runs, parse_spec

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--reference-gen", required=True, type=Path,
                    help="folder 'cGAN Generation' of the reference repository (published generated days)")
    ap.add_argument("--runs", default="runs", type=Path)
    ap.add_argument("--out", default="results", type=Path)
    ap.add_argument("--models", nargs="+",
                    default=["final:best", "faithful:last", "abl_shuffle:best", "abl_shuffle:last"])
    ap.add_argument("--data", default="data/processed/fluvius_hourly.npz")
    ap.add_argument("--ranks", nargs=2, type=int, default=CODE_RANKS, metavar=("PEAK", "BASE"),
                    help="positions of the near-peak and near-base in the sorted day; 21 3 for those of Figs. 7-8")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    global ranks
    ranks = tuple(args.ranks)

    long = []

    def record(values: dict, model: str, protocol: str, ckpt="", seed=""):
        long.extend({"model": model, "ckpt": ckpt, "seed": seed, "protocol": protocol, "metric": m, "value": v}
                    for m, v in values.items())

    paper = digitise_paper(args.pdf)
    paper.to_csv(args.out / "paper_digitised.csv", index=False)
    days, day_labels = published_days(args.reference_gen)
    scale = paper_load_scale(days, day_labels, paper)
    near_peak = scale[scale.quantity == "scale"]
    relative_bar = 1 / near_peak.value.median()        # the paper's 1 kWh relative to its median near-peak
    summary = summarise(pd.concat([paper, scale]), relative_bar, paper.censored)
    record(summary, "paper", "paper")
    columns = {"Li et al. (2025), digitised (in-sample, their data)": {"paper": [summary]}}

    splits = load_profiles(args.data)
    names = {**DISPLAY, "abl_shuffle": "GitHub cGAN + reshuffled batches"}
    controls = {}
    for spec in args.models:
        model, ckpt = parse_spec(spec)
        run_dirs = list(finished_runs(args.runs, model))
        if not run_dirs:
            raise SystemExit(f"no finished run of {model} in {args.runs}")
        column = {protocol: [] for protocol in PROTOCOLS}
        for run_dir in run_dirs:
            for protocol in PROTOCOLS:
                cells, gen, gen_label = model_cells(run_dir, ckpt, splits, protocol)
                cells.to_csv(args.out / f"paper_cells_{model}_{ckpt}_{run_dir.name}_{protocol}.csv", index=False)
                column[protocol].append(summarise(cells, relative_bar))
                record(column[protocol][-1], model, protocol, ckpt, run_dir.name)
                if protocol == "paper":
                    control = kld_control(gen, gen_label)
                    controls.setdefault(f"{model} [{ckpt}]", []).append(control)
                    record(control, model, "self", ckpt, run_dir.name)
                print(f"{model} [{ckpt}] {run_dir.name} {protocol} done", flush=True)
        columns[f"{names.get(model, model)} [{ckpt}], {len(run_dirs)} seed{'s' * (len(run_dirs) > 1)}"] = column
    for source, title in (("val", "Real days of validation households"), ("train", "Real days of training households")):
        column = {}
        for protocol in PROTOCOLS:
            cells = benchmark_cells(splits, protocol, source)
            cells.to_csv(args.out / f"paper_cells_real_{source}_{protocol}.csv", index=False)
            column[protocol] = [summarise(cells, relative_bar)]
            record(column[protocol][0], f"real_{source}", protocol)
        columns[title] = column
    for name, x, labels in (("published", days, day_labels),
                            ("real_train", splits["train"].X.astype(np.float64), splits["train"].label),
                            ("real_test", splits["test"].X.astype(np.float64), splits["test"].label)):
        controls[name] = [kld_control(x, labels, shuffle) for shuffle in range(SHUFFLES)]
        for shuffle, control in enumerate(controls[name]):
            record(control, name, "self", seed=f"shuffle{shuffle}")
    stand_ins = {}
    for name, flip in (("PV flag flipped", 4), ("heat-pump flag flipped", 8), ("EV flag flipped", 16),
                       ("other season 1", 1), ("other season 2", 2), ("other season 3", 3),
                       ("other day type", 32), ("flat days", 0)):
        stand_ins[name] = summarise(stand_in_cells(splits, flip), relative_bar)
        record(stand_ins[name], f"stand_in {name}", "paper")
    pd.DataFrame(long).to_csv(args.out / "paper_comparison.csv", index=False)

    rows = list(summary)
    table = {}
    for name, column in columns.items():
        for protocol, runs in column.items():
            title = name if name.startswith("Li") else f"{name}, {PROTOCOLS[protocol]}"
            table[title] = [cell_text([r[m] for r in runs], "share" in m) for m in rows]
    fmt = pd.DataFrame(table, index=[m.replace("|", "\\|") for m in rows])
    train = splits["train"]
    ours = pd.Series(key_parameters(train.X.astype(np.float64))["Peak_load"]).groupby(train.label).mean()
    same_rank = pd.Series(reference_key_parameters(train.X.astype(np.float64), *FIGURE_RANKS)["Peak_load"])
    same_rank = same_rank.groupby(train.label).mean()
    energy = pd.Series(train.X.astype(np.float64).sum(1)).groupby(train.label).mean()
    bar = "|Δ mean| near-peak / near-peak: share < scale-matched bar"
    base_std = "|Δ std| near-base / near-peak: share < scale-matched bar"
    time_share, time_std = "|Δ mean| time (h): share < 1", "|Δ std| time (h): share < 1"
    kld_median, kld_share = "KLD: median", "KLD: share ≤ 0.3"

    def spread(runs: list[dict]) -> str:
        r = pd.DataFrame(runs)
        return (f"{r[kld_median].min():.3f}-{r[kld_median].max():.3f}, {100 * r[kld_share].min():.1f}-"
                f"{100 * r[kld_share].max():.1f} % at or below 0.3 (100 % in {(r[kld_share] == 1).sum()} of {len(r)})")

    first = "{} [{}]".format(*parse_spec(args.models[0]))
    rank_text = {CODE_RANKS: "the 2nd-highest and 2nd-lowest hour, as the published code",
                 FIGURE_RANKS: "the 3rd-highest and 4th-lowest hour, which reproduce the paper's Figs. 7-8 from the "
                               "published days"}.get(ranks, f"the hours at positions {ranks} of the sorted day")
    pool = "Real days of validation households", "Real days of training households"
    note = (f"\nAll values with the definitions of the published cGAN_EB code (rise and fall: |argmax - argmin| of the "
            f"distance to the near-peak / near-base; its \"KLD\"; per-parameter 1-D FID), over 64 labels x 5 key "
            f"parameters, with near-peak and near-base at {rank_text}.  Paper protocol: 2,000 generated days per "
            f"label against all days of the training households (the code's \"KLD\" then pairs only the first 2,000 "
            f"of them, in household order).  Held-out: as many generated days as the test households have.  ± is the "
            f"SD over training seeds; ``paper_comparison.csv`` has every seed.  Noise drawn on the GPU, so the model "
            f"columns repeat exactly only with CUDA.\n\n"
            f"The paper's values are digitised from its heatmaps (colour scales -1.5..1.5, 0..1, 0..10) to about one "
            f"of the 256 colours of a scale (0.012, 0.004, 0.039); the digitised cells are not published.  Saturated "
            f"cells count at the end of the scale; all are |Δ mean| time cells above 1 h, so no median or share "
            f"changes.\n\n"
            f"The \"Real days\" columns replace the generated days by real days: of other households, except \"Real "
            f"days of training households, paper protocol\", a sample of the days it is compared with (with the paper "
            f"protocol at most 2,000 per label; 32 labels of the validation households have fewer, down to 1,159).  "
            f"They are benchmarks, not ceilings: the score depends on the size and composition of the pool (held-out, "
            f"scale-matched near-peak mean {100 * columns[pool[0]]['test'][0][bar]:.1f} % with the 356 validation "
            f"households, {100 * columns[pool[1]]['test'][0][bar]:.1f} % with the 1,657 training households; near-base "
            f"std {100 * columns[pool[0]]['test'][0][base_std]:.1f} % and "
            f"{100 * columns[pool[1]]['test'][0][base_std]:.1f} %).\n\n"
            f"Loads: the median near-peak of a label is {near_peak.value.median():.2f} kWh in the paper's real data "
            f"(third-highest hour of the published generated days minus the digitised difference) and "
            f"{same_rank.median():.2f} kWh at the same rank in the training households ({ours.median():.2f} kWh at the "
            f"near-peak of this table), the median daily energy "
            f"{near_peak.energy.median():.1f} kWh in the published generated days and {energy.median():.1f} kWh (net) "
            f"in the training households, so the 1 kWh level is lenient here.  The scale-matched bar is the paper's "
            f"1 kWh relative to its median near-peak ({100 * relative_bar:.1f} %, about "
            f"{relative_bar * ours.median():.3f} kWh here), applied per label to the four load rows.  The time "
            f"parameters do not depend on the load scale, but only the group without technology has the same label and "
            f"quantity in both data sets.\n\n"
            f"The code's \"KLD\" pairs real day i with generated day i and is no divergence.  Days against a shuffled "
            f"copy of themselves, identical samples, score (median, share): published days of the reference "
            f"{spread(controls['published'])}; all training days {spread(controls['real_train'])}; test days "
            f"{spread(controls['real_test'])}; {first} over its seeds {spread(controls[first])}.  Undefined values "
            f"fail the level and rank last in the median; they arise where negative near-base load makes every term "
            f"infinite.\n\n"
            f"Stand-ins that are wrong by construction, paper protocol (share at the KLD level; |Δ mean| time < 1 h; "
            f"|Δ std| time < 1 h): "
            + "; ".join(f"{name} {100 * s[kld_share]:.1f}, {100 * s[time_share]:.1f}, {100 * s[time_std]:.1f} %"
                        for name, s in stand_ins.items()) + ".\n")
    (args.out / "paper_comparison.md").write_text(fmt.to_markdown() + "\n" + note, encoding="utf-8")
    print(f"written to {args.out / 'paper_comparison.md'}")


if __name__ == "__main__":
    main()
