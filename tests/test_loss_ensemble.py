"""Tests for the loss-model ensemble and the kernel-scaling hook."""

import json
import pathlib

import numpy as np
import pytest

from softpaws.transport import coefficients, loss_ensemble, muon_range
from softpaws.transport.coefficients import (
    KernelScaling,
    kernel_scaling,
    scaled_kernel,
    set_kernel_scaling,
)

PAPER_SCRIPTS = pathlib.Path(__file__).parents[1] / "scripts" / "2026_muon_transport"
CACHE = PAPER_SCRIPTS / "output" / "69_ensemble.npz"
BASELINE = json.loads(
    (pathlib.Path(__file__).parent / "regression" / "baseline.json").read_text()
)


def test_scaling_hook_multiplies_the_accessors():
    e = np.array([1e4, 1e6, 1e8])
    b, d, moments = (
        coefficients.drift_coefficient(e),
        coefficients.diffusion_coefficient(e),
        coefficients.log_loss_moments(e),
    )
    scaling = KernelScaling(lambda x: 1.1 * np.ones_like(x), lambda x: 0.8 * np.ones_like(x))
    assert kernel_scaling() is None
    with scaled_kernel(scaling):
        assert kernel_scaling() is scaling
        np.testing.assert_allclose(coefficients.drift_coefficient(e), 1.1 * b)
        np.testing.assert_allclose(coefficients.diffusion_coefficient(e), 0.8 * d)
        first, second, third = coefficients.log_loss_moments(e)
        np.testing.assert_allclose(first, 1.1 * moments[0])
        np.testing.assert_allclose(second, 0.8 * moments[1])
        np.testing.assert_allclose(third, 0.8 * moments[2])
    assert kernel_scaling() is None
    np.testing.assert_allclose(coefficients.drift_coefficient(e), b)


def test_scaling_shortens_the_range():
    base = np.asarray(muon_range.stochastic_muon_range_km(1e6, 1e3)).item()
    more_loss = KernelScaling(lambda x: 1.2, lambda x: 1.2)
    with scaled_kernel(more_loss):
        scaled = np.asarray(muon_range.stochastic_muon_range_km(1e6, 1e3)).item()
    assert scaled < base
    set_kernel_scaling(None)


def test_variant_scaling_interpolates_and_prefactor():
    grid = np.logspace(2, 10, 5)
    ratios = {"v": np.stack([np.linspace(1.0, 1.4, 5), np.linspace(1.0, 2.0, 5)], axis=1)}
    scaling = loss_ensemble.variant_scaling(ratios, "v", grid)
    assert scaling.kappa_1(1e2) == pytest.approx(1.0)
    assert scaling.kappa_1(1e10) == pytest.approx(1.4)
    assert scaling.kappa_2(1e6) == pytest.approx(1.5)
    assert scaling.kappa_1(1e12) == pytest.approx(1.4)  # held at the end
    assert loss_ensemble.variant_scaling(ratios, None, grid) is None
    both = loss_ensemble.variant_scaling(ratios, "v", grid, prefactor=0.5)
    assert both.kappa_1(1e10) == pytest.approx(0.7)
    plain = loss_ensemble.variant_scaling(ratios, None, grid, prefactor=0.9)
    assert plain.kappa_1(1e5) == 0.9 and plain.kappa_2(1e5) == 0.9


def test_variant_ratios_and_implied_scales():
    grid = np.logspace(2, 10, 9)
    base = {ch: np.ones((9, 2)) for ch in ("bremsstrahlung", "pair production", "photonuclear")}
    ensemble = {f"base {ch}": v for ch, v in base.items()}
    ensemble["swap photo BB"] = 1.3 * np.ones((9, 2))
    variants = {"photo BB": ("photonuclear", "x")}
    baseline, ratios = loss_ensemble.variant_ratios(ensemble, variants)
    np.testing.assert_allclose(ratios["photo BB"], (3.0 - 1.0 + 1.3) / 3.0)
    scales = loss_ensemble.implied_scales(ratios, grid)
    assert scales["photo BB"] == pytest.approx(1.1)


@pytest.mark.skipif(not CACHE.exists(), reason="no cached ensemble")
def test_cached_ensemble_gives_the_informed_prior():
    """The spread of the implied scales is the N(1, 0.057) prior of Section IV A."""
    baseline, ratios = loss_ensemble.load_ensemble(CACHE)
    assert set(ratios) == set(loss_ensemble.VARIANTS)
    for values in ratios.values():
        assert values.shape == (loss_ensemble.E_GRID.size, loss_ensemble.N_MOMENTS)
    scales = np.array(list(loss_ensemble.implied_scales(ratios).values()))
    want = BASELINE["informed_transport_prior"]["prior_sigma"]
    assert np.std(scales) == pytest.approx(want, abs=0.015)
    assert abs(np.mean(scales) - 1.0) < 0.05
