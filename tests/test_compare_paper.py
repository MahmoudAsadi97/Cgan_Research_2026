import numpy as np
import pandas as pd
import pytest

from compare_paper import TICKS, cell_text, digitise, kld_control, our_cells, paper_load_scale, summarise


def synthetic_heatmap(values, vrange):
    """A heatmap with a colour bar and tick labels, laid out like the figures of the paper: 256 colours,
    the first at the top of the bar (vmax)."""
    vmin, vmax = vrange
    n_rows, n_cols = values.shape
    colours = np.stack([np.arange(256), 255 - np.arange(256), np.full(256, 128)], axis=1).astype(np.uint8)
    index = lambda v: np.clip(((vmax - v) / (vmax - vmin) * 256).astype(int), 0, 255)
    y0, step_y, x0, step_x = 20, 150, 10, 30
    y1 = y0 + n_rows * step_y
    bar0, bar1 = x0 + n_cols * step_x + 30, x0 + n_cols * step_x + 60
    img = np.full((y1 + 60, bar1 + 40, 3), 255, np.uint8)
    for i in range(n_rows):
        for j in range(n_cols):
            img[y0 + i * step_y:y0 + (i + 1) * step_y, x0 + j * step_x:x0 + (j + 1) * step_x] = (
                colours[index(values[i, j])])
    img[y0:y1, bar0:bar1] = colours[np.linspace(0, 256, y1 - y0, endpoint=False).astype(int)][:, None]
    for y in np.linspace(y0, y1, TICKS[vrange]):
        img[int(y) - 4:int(y) + 4, bar1 + 5:bar1 + 15] = 0              # colour-bar tick labels
    for j in range(n_cols):
        xc = x0 + j * step_x + step_x // 2
        img[y1 + 10:y1 + 30, xc - 3:xc + 3] = 0                         # column tick labels
    return img


@pytest.mark.parametrize("vrange", [(-1.5, 1.5), (0.0, 10.0)])
def test_digitise_recovers_the_cells(vrange):
    vmin, vmax = vrange
    rng = np.random.default_rng(0)
    truth = rng.uniform(vmin + 0.02 * (vmax - vmin), vmax - 0.02 * (vmax - vmin), (5, 32))
    values, censored = digitise(synthetic_heatmap(truth, vrange), 5, vrange)
    assert np.abs(values - truth).max() < 1.5 * (vmax - vmin) / 256
    assert not censored.any()


def test_digitise_refuses_a_wrong_number_of_tick_labels():
    img = synthetic_heatmap(np.zeros((5, 32)), (-1.5, 1.5))           # 7 tick labels
    with pytest.raises(ValueError, match="tick labels"):
        digitise(img, 5, (0.0, 10.0))                                 # this scale has 6


def test_digitise_refuses_a_wrong_number_of_columns():
    img = synthetic_heatmap(np.zeros((5, 31)), (-1.5, 1.5))           # the code expects 32 tick labels
    with pytest.raises(ValueError, match="instead of 32"):
        digitise(img, 5, (-1.5, 1.5))


def test_digitise_flags_cells_at_the_ends_of_the_scale():
    vrange = (-1.5, 1.5)
    truth = np.zeros((5, 32))
    truth[2, 7], truth[3, 9] = 4.0, -4.0                              # outside the colour scale
    values, censored = digitise(synthetic_heatmap(truth, vrange), 5, vrange)
    assert censored[2, 7] and censored[3, 9] and censored.sum() == 2
    assert values[2, 7] > 1.44 and values[3, 9] < -1.44


def test_our_cells_maps_labels_to_columns():
    rng = np.random.default_rng(0)
    label = np.repeat(np.arange(64), 30)
    real = rng.normal(5, 1, (len(label), 24))
    gen = real + (label[:, None] + 1) / 100                           # every label has its own offset
    cells = our_cells(real, gen, label, label)
    d = cells[(cells.quantity == "difference") & (cells.row == "PeakLoad_mean")]
    assert len(d) == 64
    assert d[(d.daytype == "Weekday") & (d.column == 31)].value.item() == pytest.approx(0.32)
    assert d[(d.daytype == "Weekend") & (d.column == 0)].value.item() == pytest.approx(0.33)
    assert d[(d.daytype == "Weekend") & (d.column == 31)].value.item() == pytest.approx(0.64)


def test_our_cells_uses_the_sample_sd_and_both_terms_of_the_fid():
    label = np.repeat(np.arange(64), 2)
    real = np.ones((128, 24))                                         # near-peak 1 on every day: mean 1, SD 0
    gen = np.where(np.arange(128)[:, None] % 2 == 0, 1.0, 3.0) * np.ones((1, 24))   # near-peak 1, 3: mean 2, SD 2 ** 0.5
    cells = our_cells(real, gen, label, label)
    pick = lambda quantity, row: cells[(cells.quantity == quantity) & (cells.row == row)].value
    assert pick("difference", "PeakLoad_std").to_numpy() == pytest.approx(2 ** 0.5)
    assert pick("FID", "PeakLoad").to_numpy() == pytest.approx(3.0)


def test_paper_load_scale_takes_the_difference_off_the_generated_days():
    labels = np.repeat(np.arange(64), 2)
    days = np.where(np.arange(128)[:, None] % 2 == 0, 4.0, 6.0) * np.ones((1, 24))    # near-peak mean 5, SD 2 ** 0.5
    days[:, 22:] += 1                                                 # the two highest hours are not the near-peak
    rows = [{"quantity": "difference", "daytype": daytype, "column": column, "row": row, "value": value}
            for daytype in ("Weekday", "Weekend") for column in range(32)
            for row, value in (("PeakLoad_mean", 1.0), ("PeakLoad_std", 0.5))]
    scale = paper_load_scale(days, labels, pd.DataFrame(rows))
    assert (scale[scale.quantity == "scale"].value == 4).all()        # generated - (generated - real)
    assert scale[scale.quantity == "scale_std"].value.to_numpy() == pytest.approx(2 ** 0.5 - 0.5)
    assert scale[scale.quantity == "scale"].energy.to_numpy() == pytest.approx(122)


def test_kld_control_of_identical_days_scores_zero():
    labels = np.repeat(np.arange(64), 3)
    control = kld_control(np.tile(np.arange(24.0), (len(labels), 1)), labels)
    assert control == {"KLD: median": 0.0, "KLD: share ≤ 0.3": 1.0}


def cells_frame(difference=0.5, kld=0.1, fid=1.0, near_peak=2.0):
    rows = []
    for daytype in ("Weekday", "Weekend"):
        for column in range(32):
            rows.append({"quantity": "scale", "daytype": daytype, "row": "PeakLoad", "column": column,
                         "value": near_peak})
            for row in ("PeakLoad", "BaseLoad", "PeakDuration", "RiseTime", "FallTime"):
                rows.append({"quantity": "KLD", "daytype": daytype, "row": row, "column": column, "value": kld})
                rows.append({"quantity": "FID", "daytype": daytype, "row": row, "column": column, "value": fid})
                for stat in ("mean", "std"):
                    rows.append({"quantity": "difference", "daytype": daytype, "row": f"{row}_{stat}",
                                 "column": column, "value": difference})
    return pd.DataFrame(rows)


def test_summarise_levels():
    s = summarise(cells_frame(), relative_bar=0.3)
    assert s["|Δ mean| time (h): share < 1"] == 1
    assert s["|Δ mean| time (h): share of labels with all three < 1"] == 1
    assert s["|Δ mean| near-peak / near-peak: share < scale-matched bar"] == 1      # 0.5 / 2 < 0.3
    assert summarise(cells_frame(), relative_bar=0.2)[
        "|Δ mean| near-peak / near-peak: share < scale-matched bar"] == 0
    assert s["FID, time rows: share < 5"] == 1


def test_summarise_undefined_kld_fails_its_level():
    cells = cells_frame()
    mask = (cells.quantity == "KLD") & (cells.row == "BaseLoad") & (cells.column.isin([4, 5]))    # group 1, PV
    cells.loc[mask, "value"] = np.nan
    other = (cells.quantity == "KLD") & (cells.row == "RiseTime") & (cells.column == 0)
    cells.loc[other, "value"] = 0.5
    s = summarise(cells, relative_bar=0.3)
    assert s["KLD: undefined values (of 320)"] == 4
    assert s["KLD: share ≤ 0.3"] == (320 - 6) / 320
    assert s["KLD: failures outside the near-base of PV labels"] == 2
    assert s["KLD: median"] == 0.1                                    # undefined values rank last, not out


def test_summarise_median_ranks_undefined_kld_last():
    cells = cells_frame()
    kld = cells.index[cells.quantity == "KLD"]
    cells.loc[kld[:300], "value"] = np.nan
    assert summarise(cells, relative_bar=0.3)["KLD: median"] == np.inf


def test_summarise_pv_mask_covers_every_pv_group():
    cells = cells_frame()
    kld = cells.quantity == "KLD"
    cells.loc[kld & (cells.row == "BaseLoad") & cells.column.isin([4, 13, 22, 31]), "value"] = np.nan    # groups 1, 3, 5, 7
    cells.loc[kld & (cells.row == "BaseLoad") & (cells.column == 8), "value"] = 0.5                      # group 2: no PV
    s = summarise(cells, relative_bar=0.3)
    assert s["KLD: undefined values (of 320)"] == 8
    assert s["KLD: failures outside the near-base of PV labels"] == 2


def test_summarise_undefined_kld_outside_pv_near_base_is_a_failure():
    cells = cells_frame()
    cells.loc[(cells.quantity == "KLD") & (cells.row == "RiseTime") & (cells.column == 0), "value"] = np.nan
    assert summarise(cells, relative_bar=0.3)["KLD: failures outside the near-base of PV labels"] == 2


def test_summarise_keeps_the_load_and_time_rows_apart():
    cells = cells_frame()
    cells.loc[(cells.quantity == "difference") & (cells.row == "PeakLoad_mean"), "value"] = 2.0
    cells.loc[(cells.quantity == "difference") & (cells.row == "RiseTime_std"), "value"] = 2.0
    s = summarise(cells, relative_bar=0.3)
    assert s["|Δ mean| load (kWh): share < 1"] == 0.5
    assert s["|Δ mean| time (h): share < 1"] == 1
    assert s["|Δ std| load (kWh): share < 1"] == 1
    assert s["|Δ std| time (h): share < 1"] == pytest.approx(2 / 3)
    assert s["|Δ mean| near-peak / near-peak: share < scale-matched bar"] == 0      # 2 / 2 > 0.3
    assert s["|Δ mean| near-base / near-peak: share < scale-matched bar"] == 1      # 0.5 / 2 < 0.3


def test_summarise_label_groups():
    cells = cells_frame()
    time_mean = (cells.quantity == "difference") & (cells.row == "RiseTime_mean")
    cells.loc[time_mean & (cells.column == 3), "value"] = 2.0
    cells.loc[time_mean & (cells.column == 7), "value"] = 2.0
    cells.loc[time_mean & (cells.column == 8), "value"] = 2.0
    s = summarise(cells, relative_bar=0.3)
    assert s["|Δ mean| time (h), no technology and PV only: share < 1"] == pytest.approx(44 / 48)
    assert s["|Δ mean| time (h), no technology: share < 1"] == pytest.approx(22 / 24)


def test_cell_text():
    assert cell_text([0.5, 0.7], share=True) == "60.0 ± 14.1 %"
    assert cell_text([0.1234], share=False) == "0.123"
    assert cell_text([np.nan], share=False) == "–"
    assert cell_text([0.1 + 0.2] * 5, share=False) == "0.3 ± 0"                          # no rounding noise
