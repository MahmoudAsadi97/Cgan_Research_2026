import numpy as np
import pytest
from scipy.stats import spearmanr

from baselines import Gaussian, GaussianCopula, GaussianMixture, IndependentHours, select_components
from metrics import _ks
from train import Scaler

NO_PV, PV = 0, 4                                  # labels of the groups "No PV HP EV" and "PV" (weekday, spring)


def skewed_days(n=6000, seed=0, shift=0.0):
    """Right-skewed days with a strong relation between neighbouring hours."""
    rng = np.random.default_rng(seed)
    z = np.empty((n, 24))
    z[:, 0] = rng.standard_normal(n)
    for t in range(1, 24):
        z[:, t] = 0.8 * z[:, t - 1] + 0.6 * rng.standard_normal(n)
    return np.exp(0.7 * z) * (1 + 0.5 * np.sin(np.arange(24) / 24 * 2 * np.pi)) + shift


def fitted(model, days=None):
    days = skewed_days() if days is None else days
    model.fit_label(NO_PV, days)
    return model, days


def test_copula_keeps_the_hourly_distributions_and_the_rank_correlation():
    model, real = fitted(GaussianCopula())
    gen = model.sample_kwh(np.full(6000, NO_PV), seed=1)
    assert max(_ks(real[:, t], gen[:, t]) for t in range(24)) < 0.04
    assert np.abs(spearmanr(real).statistic - spearmanr(gen).statistic).max() < 0.06
    assert gen.min() >= real.min() and gen.max() <= real.max()         # inside the observed range


def test_independent_hours_keep_the_distributions_and_drop_the_correlation():
    model, real = fitted(IndependentHours())
    gen = model.sample_kwh(np.full(6000, NO_PV), seed=1)
    assert max(_ks(real[:, t], gen[:, t]) for t in range(24)) < 0.04
    corr = np.corrcoef(gen, rowvar=False)
    assert np.abs(corr[np.triu_indices(24, 1)]).max() < 0.08
    assert np.corrcoef(real, rowvar=False)[0, 1] > 0.6


def test_gaussian_keeps_mean_and_covariance():
    model, real = fitted(Gaussian())
    gen = model.sample_kwh(np.full(40000, NO_PV), seed=1)
    assert np.abs(gen.mean(0) - real.mean(0)).max() < 0.03
    assert np.abs(np.cov(gen, rowvar=False) - np.cov(real, rowvar=False)).max() < 0.08
    assert (gen < 0).any()                                             # the normal ignores that load is positive


def test_mixture_finds_two_kinds_of_days_and_floors_labels_without_pv():
    low, high = skewed_days(3000, 0), skewed_days(3000, 1, shift=6.0)
    real = np.concatenate([low, high])
    scaler = Scaler("quantile").fit(np.concatenate([real, real - 2.0]))   # all labels pooled, as in training
    distance = {}
    for k in (1, 2):
        model = GaussianMixture(k, scaler, seed=0)
        model.fit_label(NO_PV, real)
        model.fit_label(PV, real - 2.0)
        gen = model.sample_kwh(np.r_[np.full(6000, NO_PV), np.full(6000, PV)], seed=3)
        distance[k] = _ks(real.sum(1), gen[:6000].sum(1))              # daily energy: two kinds of days
    assert distance[2] < 0.15 < distance[1]                            # one component cannot hold both
    assert 0.45 < (gen[:6000].sum(1) > real.sum(1).mean()).mean() < 0.6
    assert gen[:6000].min() >= 0 and gen[6000:].min() < 0              # floor at 0 kWh without PV only


def test_sampling_repeats_for_a_seed_and_does_not_depend_on_the_other_labels():
    model = GaussianCopula()
    model.fit_label(NO_PV, skewed_days(2000, 0))
    model.fit_label(PV, skewed_days(2000, 1) - 1.0)
    both = model.sample_kwh(np.array([PV, NO_PV, NO_PV, PV, NO_PV]), seed=7)
    assert np.array_equal(both, model.sample_kwh(np.array([PV, NO_PV, NO_PV, PV, NO_PV]), seed=7))
    assert np.array_equal(both[[1, 2, 4]], model.sample_kwh(np.full(3, NO_PV), seed=7))
    assert not np.array_equal(both, model.sample_kwh(np.array([PV, NO_PV, NO_PV, PV, NO_PV]), seed=8))


@pytest.mark.parametrize("scores,expected", [
    ({1: 0.80, 5: 0.850, 10: 0.8505, 20: 0.851}, 5),                   # 5 is within 0.1 point of the best
    ({1: 0.80, 5: 0.84, 10: 0.86, 20: 0.85}, 10),
    ({2: 0.90}, 2),
])
def test_select_components(scores, expected):
    assert select_components(scores) == expected
