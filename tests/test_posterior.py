"""Tests for the posterior statistics and the Feldman-Cousins helpers."""

import numpy as np
import pytest

from softpaws.comparison import feldman_cousins as fc
from softpaws.comparison import posterior as post


def _gaussian_chain(mean, sigma, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(mean, sigma, size=(n, len(mean)))


def test_box_and_log_gaussian():
    assert post.inside_box([0.5, 2.0], [(0.0, 1.0), (1.0, 3.0)])
    assert not post.inside_box([1.0, 2.0], [(0.0, 1.0), (1.0, 3.0)])
    assert post.log_gaussian_in_log([1.0, 2.0], [1.0, 2.0], 0.1) == 0.0
    assert post.log_gaussian_in_log([1.0], [np.exp(0.1)], 0.1) == pytest.approx(-0.5)
    assert post.log_gaussian_in_log([1.0], [0.0], 0.1) == -np.inf
    assert post.log_gaussian_in_log([1.0], [np.nan], 0.1) == -np.inf


def test_marginal_summary_and_intersection():
    chain = _gaussian_chain([0.5, 1.0], [0.1, 0.2])
    priors = {"a": (0.0, 1.0), "b": (0.0, 1.05)}
    summary = post.marginal_summary(chain, ("a", "b"), priors, best=[0.5, 1.0])
    assert summary["a"]["median"] == pytest.approx(0.5, abs=0.01)
    assert summary["a"]["ci68"][0] == pytest.approx(0.4, abs=0.01)
    assert summary["a"]["best_fit"] == 0.5
    assert not summary["a"]["rails_prior"]["high"]
    assert summary["b"]["rails_prior"]["high"]
    overlap = post.intersection([(0.0, 1.0), (0.5, 2.0)])
    assert overlap == {"low": 0.5, "high": 1.0, "empty": False}
    assert post.intersection([(0.0, 1.0), (1.5, 2.0)])["empty"]


def test_product_posterior_and_compatibility():
    a = _gaussian_chain([1.0, 0.5], [0.1, 0.1], seed=1)
    b = _gaussian_chain([1.0, 0.5], [0.1, 0.1], seed=2)
    product = post.product_posterior([a, b], ("x", "y"), n_grid=80)
    assert product["x"]["median"] == pytest.approx(1.0, abs=0.01)
    width = product["y"]["ci68"][1] - product["y"]["ci68"][0]
    assert width == pytest.approx(2 * 0.1 / np.sqrt(2), rel=0.15)
    pair = post.pairwise_compatibility(a, b)
    assert pair["dof"] == 2 and pair["chi2"] < 1.0 and pair["sigma"] < 1.0
    far = _gaussian_chain([2.0, 0.5], [0.1, 0.1], seed=3)
    assert post.pairwise_compatibility(a, far)["sigma"] > 5.0
    loo = post.leave_one_out_compatibility([a, b, far])
    assert loo[2]["sigma"] > loo[0]["sigma"]
    glob = post.global_compatibility([a, b], ("x", "y"))
    assert glob["dof"] == 2 and glob["sigma"] < 1.5
    assert glob["combined_mean"]["x"] == pytest.approx(1.0, abs=0.01)
    assert glob["combined_sigma"]["x"] == pytest.approx(0.1 / np.sqrt(2), rel=0.1)


def test_sample_posterior_recovers_a_gaussian():
    def log_prob(theta):
        return -0.5 * float(np.sum(((theta - np.array([1.0, -2.0])) / 0.3) ** 2))

    chain, best = post.sample_posterior(log_prob, [0.9, -1.9], [0.05, 0.05], 16, 600, seed=4)
    assert chain.shape[1] == 2 and chain.shape[0] == 16 * 400
    np.testing.assert_allclose(chain.mean(axis=0), [1.0, -2.0], atol=0.08)
    np.testing.assert_allclose(best, [1.0, -2.0], atol=0.15)


def test_feldman_cousins_helpers(tmp_path):
    grid = np.linspace(0.0, 1.0, 41)
    curve = 20.0 * (grid - 0.3) ** 2
    lo, hi = fc.profile_interval(grid, curve, 1.0)
    assert lo == pytest.approx(0.3 - np.sqrt(0.05), abs=0.03)
    assert hi == pytest.approx(0.3 + np.sqrt(0.05), abs=0.03)
    with pytest.raises(ValueError):
        fc.profile_interval(grid, curve + 100.0, 1.0)
    truths = np.linspace(0.0, 0.9, 4)
    q = np.array([[0.0, 1.0, 2.0, 3.0]] * 4) * (1 + truths[:, None])
    thresholds = fc.calibrated_thresholds(q, truths, grid)
    assert set(thresholds) == {68.27, 95.0}
    assert thresholds[95.0][0] == pytest.approx(np.percentile(q[0], 95.0))
    assert np.all(np.diff(thresholds[68.27]) >= -1e-12)

    rng = np.random.default_rng(0)
    q_toys = fc.toy_statistics(
        truths,
        lambda r: np.full(3, 5.0 + r),
        lambda r, toy: float(toy.sum() - 15.0 - 3 * r) ** 2 / 15.0,
        50,
        rng,
    )
    assert q_toys.shape == (4, 50) and np.all(q_toys >= 0.0)

    calls = []
    meta = {"r_true": truths, "n_toys": 50, "seed": 0, "version": 1}
    path = tmp_path / "toys.npz"
    first = fc.cached_toys(path, meta, lambda: calls.append(1) or q_toys)
    second = fc.cached_toys(path, meta, lambda: calls.append(1) or q_toys)
    assert len(calls) == 1 and np.array_equal(first, second)
    fc.cached_toys(path, {**meta, "version": 2}, lambda: calls.append(1) or q_toys)
    assert len(calls) == 2
