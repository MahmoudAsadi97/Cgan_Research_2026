"""Checks of the prepared data against the published numbers.

They need ``data/processed/fluvius_hourly.npz`` (``python data_prep.py``) and are skipped without it.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_check import RAW_SHA256, check, household_table, sha256_of
from data_prep import GROUP_NAMES, N_LABELS, load_profiles

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "fluvius_hourly.npz"
RAW = ROOT / "data" / "raw" / "raw.zip"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="prepared data not found; run data_prep.py first")


@pytest.fixture(scope="module")
def splits():
    return load_profiles(DATA)


def test_counts_of_the_readme(splits):
    info = splits["train"].info
    assert info["duplicate_meters"] == 33
    assert info["no_flow_meter_days"] == 866
    assert info["meters"] == {"train": 1657, "val": 356, "test": 354}
    assert info["days"] == {"train": 602356, "val": 129540, "test": 128826}
    for split in splits.values():
        assert len(np.unique(split.label)) == N_LABELS
        assert split.X.shape[1] == 24 and np.isfinite(split.X).all()


def test_no_household_is_in_two_splits(splits):
    meters = [set(np.unique(s.meter)) for s in splits.values()]
    assert not (meters[0] & meters[1] or meters[0] & meters[2] or meters[1] & meters[2])


def test_split_is_the_published_one():
    published = pd.read_csv(ROOT / "results" / "household_split.csv")
    pd.testing.assert_frame_equal(household_table(DATA), published, check_dtype=False)
    assert published.groupby("split").size().to_dict() == {"duplicate": 33, "test": 354, "train": 1657, "val": 356}
    assert set(published.group) == set(GROUP_NAMES)


def test_label_table_is_the_published_one(splits):
    published = pd.read_csv(ROOT / "results" / "table1_labels.csv")
    for name in ("train", "val", "test"):
        days = np.bincount(splits[name].label, minlength=N_LABELS)
        assert (days == published[f"days {name}"].to_numpy()).all()


def test_labels_without_pv_hold_almost_no_negative_load(splits):
    """Three meters without the PV indicator register a little injection; all negative values are in training."""
    negative = {}
    for name, split in splits.items():
        no_pv = (split.label // 4) % 2 == 0
        negative[name] = int((split.X[no_pv] < 0).sum())
    assert negative == {"train": 5, "val": 0, "test": 0}


@pytest.mark.skipif(not RAW.exists(), reason="raw.zip not found")
def test_raw_archive_is_the_released_one():
    assert sha256_of(RAW) == RAW_SHA256


def test_data_check_passes():
    result = check(DATA)
    assert len(result) == 4 and all(result.values()), result
