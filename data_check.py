"""Check that a local copy of the data is the one behind the published results.

    python data_check.py            # after data_prep.py

The checks need no trained model:

* the SHA-256 of ``raw.zip`` is that of the archive in the data release;
* the cleaning removes what the README states, and the split has its numbers of households and days;
* every meter is in the split stored in ``results/household_split.csv``, so the split does not depend on
  the random-number stream of a NumPy version;
* the days per label and split are those of ``results/table1_labels.csv``.

``household_table`` writes the stored split (``report.py`` calls it with the result tables).
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from data_prep import GROUP_NAMES, N_LABELS, load_profiles

ROOT = Path(__file__).resolve().parent
# raw.zip of the release ``data``; the reported results were computed from this archive.
RAW_SHA256 = "934cc871e9fe18dbe30530472816d711ece878ec7cd8725d9d8417bbdb389bd8"
SPLIT_NAMES = ["train", "val", "test"]
PUBLISHED = {"duplicate_meters": 33, "no_flow_meter_days": 866,
             "meters": {"train": 1657, "val": 356, "test": 354},
             "days": {"train": 602356, "val": 129540, "test": 128826}}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def household_table(path: str | Path = "data/processed/fluvius_hourly.npz", split_seed: int = 2024,
                    fractions=(0.7, 0.15, 0.15), splits=None) -> pd.DataFrame:
    """Split and number of valid days of every meter: ``EAN_ID``, the three indicators, the group, the split
    (``train`` / ``val`` / ``test``, or ``duplicate`` for the removed copies) and the days used."""
    z = np.load(path, allow_pickle=False)
    splits = splits or load_profiles(path, "net", split_seed, fractions)
    group = z["pv"] + 2 * z["hp"] + 4 * z["ev"]
    split = np.full(len(group), "duplicate", object)
    days = np.zeros(len(group), np.int64)
    for name in SPLIT_NAMES:
        meters, counts = np.unique(splits[name].meter, return_counts=True)
        split[meters], days[meters] = name, counts
    return pd.DataFrame({"EAN_ID": z["ean"], "pv": z["pv"], "hp": z["hp"], "ev": z["ev"],
                         "group": np.array(GROUP_NAMES)[group], "split": split, "days": days})


def check(data: Path, raw: Path | None = None, results: Path = ROOT / "results") -> dict[str, bool]:
    """Name of every check -> passed."""
    out = {}
    if raw is not None and raw.exists():
        out["raw.zip is the archive of the data release (SHA-256)"] = sha256_of(raw) == RAW_SHA256
    splits = load_profiles(data)
    info = splits["train"].info
    out["33 duplicated meters and 866 meter-days without flow removed"] = all(
        info[k] == PUBLISHED[k] for k in ("duplicate_meters", "no_flow_meter_days"))
    out["1,657 / 356 / 354 households and 602,356 / 129,540 / 128,826 days"] = all(
        info[k] == PUBLISHED[k] for k in ("meters", "days"))
    table = household_table(data, splits=splits)
    stored = pd.read_csv(results / "household_split.csv")
    out["every meter is in its published split (results/household_split.csv)"] = bool(
        (table.astype(str).to_numpy() == stored.astype(str).to_numpy()).all()) if table.shape == stored.shape else False
    labels = pd.read_csv(results / "table1_labels.csv")
    out["days per label and split as in results/table1_labels.csv"] = all(
        (np.bincount(splits[name].label, minlength=N_LABELS) == labels[f"days {name}"].to_numpy()).all()
        for name in SPLIT_NAMES)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw/raw.zip")
    ap.add_argument("--data", default="data/processed/fluvius_hourly.npz")
    args = ap.parse_args(argv)
    result = check(Path(args.data), Path(args.raw))
    for name, passed in result.items():
        print(f"{'ok    ' if passed else 'FAILED'}  {name}")
    return 0 if all(result.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
