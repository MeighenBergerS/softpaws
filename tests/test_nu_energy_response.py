"""Tests for the per-neutrino-energy response pieces.

Cover the Gil-Pelaez log-loss CDF against the density-inversion route, the
stochastic muon range against its deterministic CSDA counterpart, and the
neutral-current regeneration ladder against pure absorption.
"""

import numpy as np
import pytest

from softpaws.transport.attenuation import (
    regenerated_transmission,
    survival_probability,
)
from softpaws.transport.coefficients import diffusion_coefficient, drift_coefficient
from softpaws.transport.cross_section import bgr18_cross_section
from softpaws.transport.loss_distribution import (
    log_loss_cdf,
    loss_density,
    survival_from_density,
)
from softpaws.transport.soft_volume import (
    DEFAULT_MUON_THRESHOLD_GEV,
    muon_range_km,
    stochastic_muon_range_km,
)


def _coefficients(energy_gev: float) -> tuple[float, float]:
    energy = np.array([energy_gev])
    return (
        float(np.atleast_1d(drift_coefficient(energy))[0]),
        float(np.atleast_1d(diffusion_coefficient(energy))[0]),
    )


def test_log_loss_cdf_is_a_probability():
    b_mu, d_mu = _coefficients(8.0e5)
    cdf = log_loss_cdf(np.log(8.0e5 / 1.0e3), np.linspace(0.0, 80.0, 41), b_mu, d_mu)
    assert np.all(cdf >= 0.0)
    assert np.all(cdf <= 1.0)
    # Deeper propagation can only lose more energy, so survival is monotone.
    # The tolerance is the residual k-quadrature ripple, which sits at ~1e-4
    # where the CDF is still near one and is far below any physical effect.
    assert np.all(np.diff(cdf) <= 1.0e-4)


def test_log_loss_cdf_matches_density_inversion():
    b_mu, d_mu = _coefficients(8.0e5)
    w_star = np.log(8.0e5 / 1.0e3)
    w_grid = np.linspace(0.0, 60.0, 6001)
    depths = np.array([5.0, 10.0, 15.0, 20.0])
    reference = np.array([
        1.0 - float(survival_from_density(w_star, w_grid, loss_density(w_grid, d, b_mu, d_mu)))
        for d in depths
    ])
    assert np.allclose(log_loss_cdf(w_star, depths, b_mu, d_mu), reference, atol=5.0e-3)


def test_log_loss_cdf_starts_at_unity():
    b_mu, d_mu = _coefficients(1.0e6)
    # A muon that has propagated no distance has lost nothing. The residual is
    # the k-truncation of the inversion, not a physical deficit.
    for energy_gev in (1.0e6, 8.0e5):
        b_mu, d_mu = _coefficients(energy_gev)
        at_zero = log_loss_cdf(np.log(energy_gev / 1.0e3), 0.0, b_mu, d_mu)[0]
        assert at_zero == pytest.approx(1.0, abs=1.0e-3)


def test_stochastic_range_is_shorter_at_high_energy():
    energy = np.array([1.0e6, 1.0e7, 1.0e8])
    stochastic = stochastic_muon_range_km(energy)
    deterministic = muon_range_km(energy)
    # The loss law is right-skewed, so more muons fall short of the mean range
    # than overshoot it, and the shortfall grows with the number of e-folds.
    assert np.all(stochastic < deterministic)
    ratio = stochastic / deterministic
    assert np.all(np.diff(ratio) < 0.0)
    assert np.all(ratio > 0.8)


def test_stochastic_range_vanishes_below_threshold():
    below = stochastic_muon_range_km(np.array([0.5 * DEFAULT_MUON_THRESHOLD_GEV]))
    assert below[0] == 0.0


def test_regeneration_reduces_to_absorption_with_one_level():
    cross_section = bgr18_cross_section()
    columns = np.array([1.0e8, 1.0e9])
    _, weights = regenerated_transmission(1.0e8, columns, cross_section, n_levels=1)
    plain = survival_probability(1.0e8, columns, cross_section=cross_section)
    assert np.allclose(weights[0], plain)


def test_regeneration_only_adds_flux():
    cross_section = bgr18_cross_section()
    columns = np.array([1.0e8, 1.0e9, 5.0e9])
    energies, weights = regenerated_transmission(1.0e8, columns, cross_section)
    plain = survival_probability(1.0e8, columns, cross_section=cross_section)
    assert np.all(weights >= 0.0)
    # The top rung is the never-scattered population, i.e. pure absorption.
    assert np.allclose(weights[0], plain)
    # Everything below it is regenerated, so the total can only be larger.
    assert np.all(weights.sum(axis=0) >= plain)
    # The ladder descends in energy by a fixed fraction per rung.
    assert np.all(np.diff(energies) < 0.0)


def test_closed_form_matches_the_depth_integral():
    energy = np.array([1.0e4, 1.0e5, 1.0e6, 1.0e7, 1.0e8])
    closed = stochastic_muon_range_km(energy, method="closed")
    quadrature = stochastic_muon_range_km(energy, method="quadrature")
    # The renewal expansion is exact to well under a centimetre over four
    # decades; see docs/first_passage_range.md.
    assert np.allclose(closed, quadrature, atol=1.0e-2)


def test_closed_form_replaces_mean_y_by_mean_log():
    from scipy.special import polygamma

    from softpaws.transport.eigenvalue import two_moment_loss_spectrum

    energy = np.array([1.0e8])
    b_mu, d_mu = _coefficients(float(energy[0]))
    kappa, p = two_moment_loss_spectrum(b_mu, d_mu)
    first = kappa * polygamma(1, p + 1.0)
    # Phi'(0) = <-ln(1-y)> >= <y> = b_mu for any positive loss spectrum, so the
    # stochastic range is always the shorter one.
    assert first > b_mu
    ratio = stochastic_muon_range_km(energy)[0] / muon_range_km(energy)[0]
    assert b_mu / first < ratio < 1.0


def test_stochastic_range_rejects_unknown_method():
    with pytest.raises(ValueError, match="method must be"):
        stochastic_muon_range_km(np.array([1.0e6]), method="nope")


def test_flavour_transmission_reproduces_the_geometric_ladder():
    from softpaws.transport.attenuation import flavour_transmission

    cross_section = bgr18_cross_section()
    columns = np.array([1.0e8, 1.0e9, 3.0e9])
    _, ladder = regenerated_transmission(1.0e8, columns, cross_section)
    _, grid = flavour_transmission(
        1.0e8, columns, cross_section, flavour="mu", n_grid=200, decades=5.0
    )
    # Two independent discretizations of the same nu_mu cascade: a geometric
    # ladder stepping by (1 - <y>) exactly, and a log-energy grid that splits
    # the feed between neighbouring nodes. The ladder itself is checked against
    # a dense matrix exponential in test_regeneration_only_adds_flux.
    assert np.allclose(grid.sum(axis=0), ladder.sum(axis=0), rtol=2.0e-3)


def test_tau_stays_transparent_where_muon_does_not():
    from softpaws.transport.attenuation import flavour_transmission

    cross_section = bgr18_cross_section()
    columns = np.array([1.0e9, 5.0e9])
    _, mu = flavour_transmission(1.0e8, columns, cross_section, flavour="mu")
    _, tau = flavour_transmission(1.0e8, columns, cross_section, flavour="tau")
    # Charged current terminates a nu_mu but regenerates a nu_tau, so the tau
    # flavour survives the Earth where the muon flavour does not, and the gap
    # widens with column depth.
    assert np.all(tau.sum(axis=0) > mu.sum(axis=0))
    assert tau.sum(axis=0)[1] / mu.sum(axis=0)[1] > tau.sum(axis=0)[0] / mu.sum(axis=0)[0]


def test_flavour_transmission_rejects_unknown_flavour():
    from softpaws.transport.attenuation import flavour_transmission

    with pytest.raises(ValueError, match="flavour must be"):
        flavour_transmission(1.0e6, np.array([1.0e8]), flavour="electron")
