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
from softpaws.transport.coefficients import (
    diffusion_coefficient,
    drift_coefficient,
    log_loss_moments,
)
from softpaws.transport.cross_section import bgr18_cross_section
from softpaws.transport.loss_distribution import (
    log_loss_cdf,
    loss_density,
    survival_from_density,
)
from softpaws.transport.muon_range import (
    DEFAULT_MUON_THRESHOLD_GEV,
    muon_range_km,
    stochastic_muon_range_km,
    stochastic_muon_range_variance_km2,
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
    # Pinned to the family and to the frozen kernel: the depth integral builds
    # one kernel and holds it for the whole descent, so this checks the renewal
    # expansion and neither the choice of loss moments nor where they are read.
    energy = np.array([1.0e4, 1.0e5, 1.0e6, 1.0e7, 1.0e8])
    common = {"log_loss_source": "family", "kernel_evaluation": "frozen"}
    closed = stochastic_muon_range_km(energy, method="closed", **common)
    quadrature = stochastic_muon_range_km(energy, method="quadrature", **common)
    # The renewal expansion is exact to well under a centimetre over four
    # decades; see docs/first_passage_range.md.
    assert np.allclose(closed, quadrature, atol=1.0e-2)


def test_running_kernel_lengthens_the_range_by_more_at_longer_lever_arm():
    # Freezing the kernel at production charges the muon the loss rate of the
    # top of its descent for the whole of it, so it is short -- one-sidedly, and
    # by more the further the muon falls. Measured against PROPOSAL in
    # examples/39_range_moment_estimator.py: 4.0% rms frozen, 1.4% running.
    energy = np.array([1.0e4, 1.0e5, 1.0e6, 1.0e7, 1.0e8])
    running = stochastic_muon_range_km(energy)
    frozen = stochastic_muon_range_km(energy, kernel_evaluation="frozen")
    excess = running / frozen - 1.0
    assert np.all(excess >= 0.0)
    assert np.all(np.diff(excess) > 0.0)
    # Effective areas live at the top of this band, where it is worth ~11%.
    assert excess[-1] == pytest.approx(0.11, abs=0.02)


def test_running_and_frozen_agree_when_the_kernel_does_not_run():
    # The two differ only through the energy dependence of the kernel, so over a
    # lever arm short enough that it cannot run they have to coincide. The
    # ionization tail has to come off for this: it spans a decade below E_* and
    # runs whatever the radiative segment above does.
    energy, threshold = np.array([1.02e5]), 1.0e5
    common = {"threshold_gev": threshold, "include_ionization": False}
    assert stochastic_muon_range_km(energy, **common) == pytest.approx(
        stochastic_muon_range_km(energy, kernel_evaluation="frozen", **common), rel=1.0e-3
    )


def test_kernel_evaluation_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="kernel_evaluation must be"):
        stochastic_muon_range_km(np.array([1.0e6]), kernel_evaluation="nope")
    with pytest.raises(ValueError, match="needs kernel_evaluation='frozen'"):
        stochastic_muon_range_km(
            np.array([1.0e6]), method="quadrature", log_loss_source="family"
        )


def test_closed_form_replaces_mean_y_by_mean_log():
    energy = np.array([1.0e8])
    b_mu, _ = _coefficients(float(energy[0]))
    first = float(log_loss_moments(energy)[0][0])
    # Phi'(0) = <-ln(1-y)> >= <y> = b_mu for any positive loss spectrum, so the
    # stochastic range is always the shorter one.
    assert first > b_mu
    ratio = stochastic_muon_range_km(energy)[0] / muon_range_km(energy)[0]
    assert b_mu / first < ratio < 1.0


def test_stochastic_range_rejects_unknown_method():
    with pytest.raises(ValueError, match="method must be"):
        stochastic_muon_range_km(np.array([1.0e6]), method="nope")


def test_ionization_splice_deactivates_above_its_matching_energy():
    # A fitted threshold can land above E_*, at which point there is no ionizing
    # segment to splice: the muon stops counting while it is still radiative.
    # The two-regime range has to degrade to the radiative one there, not fail,
    # because the posterior samplers of examples 29 and 33 do sample it.
    energy = np.array([1.0e6, 1.0e8])
    high = stochastic_muon_range_km(energy, threshold_gev=3.0e4)
    radiative = stochastic_muon_range_km(energy, threshold_gev=3.0e4, include_ionization=False)
    assert np.allclose(high, radiative)


def test_ionization_splice_never_exceeds_the_csda_range():
    # Both treatments now carry the same loss terms, so the stochastic range can
    # only be the shorter one: fluctuations remove range, they never add it.
    # Purely radiative it can exceed R_CSDA near threshold, which is the defect
    # the splice removes.
    energy = np.logspace(3.5, 8.0, 20)
    spliced = stochastic_muon_range_km(energy)
    csda = muon_range_km(energy)
    assert np.all(spliced <= csda + 1.0e-9)
    radiative = stochastic_muon_range_km(energy, include_ionization=False)
    assert np.any(radiative > csda)


def test_ionization_splice_is_deterministic_below_the_matching_energy():
    # A muon born below E_* never enters the radiative regime the first-passage
    # expansion describes, so the CSDA range is the whole of its answer.
    energy = np.array([2.0e3, 5.0e3, 9.0e3])
    assert np.allclose(stochastic_muon_range_km(energy), muon_range_km(energy))


def test_ionization_splice_is_continuous_across_the_matching_energy():
    # The deterministic segment starts at the mean arrival energy and not at
    # E_*, which is what keeps the two regimes from double counting the
    # first-passage overshoot. Leaving it in opens a ~0.4 km step here.
    #
    # A residual step survives because the two regimes shed log energy at
    # different rates -- Phi'(0) above E_*, b_mu below it -- so the overshoot
    # stretch is costed at the radiative rate on one side and the CSDA rate on
    # the other. It is 0.13 km against a ~5 km range, inside the 3% that
    # ``match_energy_gev`` is a convention for either way.
    match = 1.0e4
    below = stochastic_muon_range_km(np.array([match * 0.999]), match_energy_gev=match)[0]
    above = stochastic_muon_range_km(np.array([match * 1.001]), match_energy_gev=match)[0]
    assert above - below == pytest.approx(0.0, abs=0.2)


def test_ionization_splice_shortens_the_range_at_every_energy():
    # Ionization adds a loss channel below E_*, so it can only remove range.
    energy = np.array([1.0e5, 1.0e6, 1.0e8])
    assert np.all(
        stochastic_muon_range_km(energy)
        < stochastic_muon_range_km(energy, include_ionization=False)
    )


def test_frozen_splice_changes_sign_where_running_does_not():
    # Under a frozen kernel two effects ran against each other in the spliced
    # decade: ionization shortened the range, and dropping the production-energy
    # loss rate for a muon that is by then at TeV energies lengthened it, with
    # the second winning at the top. Running the kernel down the trajectory
    # already carries the second, so only the shortening survives.
    high = np.array([1.0e8])
    frozen = {"kernel_evaluation": "frozen"}
    assert stochastic_muon_range_km(high, **frozen) > stochastic_muon_range_km(
        high, include_ionization=False, **frozen
    )


def test_ionization_splice_closed_form_matches_the_depth_integral():
    energy = np.array([1.0e5, 1.0e6, 1.0e7, 1.0e8])
    common = {"log_loss_source": "family", "kernel_evaluation": "frozen"}
    closed = stochastic_muon_range_km(energy, method="closed", **common)
    quadrature = stochastic_muon_range_km(energy, method="quadrature", **common)
    assert np.allclose(closed, quadrature, atol=1.0e-2)


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


def test_ionization_splice_stays_finite_for_an_invalid_loss_spectrum():
    # b_scale below ~0.287 drives d_mu / b_mu above 1, where the two-moment
    # kernel is no longer a loss spectrum and the range is meaningless. It still
    # has to come back finite: examples 29 and 33 sample that corner, and an
    # infinity there kills the chain instead of being rejected by the posterior.
    value = stochastic_muon_range_km(np.array([1.86e6]), 226.76, b_scale=0.2433)
    assert np.all(np.isfinite(value))


# ---------------------------------------------------------------------------
# Variance of the first-passage range. The reference numbers are the PROPOSAL
# Monte Carlo of examples/39_range_moment_estimator.py, 2000 muons per point,
# propagated from the production energy down to 100 TeV.
# ---------------------------------------------------------------------------

# log10(eps / GeV) -> sigma_R [km w.e.] measured, stopping at 1e5 GeV.
_MC_SIGMA_KM = {6.0: 2.330, 6.5: 3.030, 7.0: 3.583, 7.5: 4.063}


def test_range_variance_matches_the_monte_carlo():
    # Parameter-free: the closed form carries no freedom once the kernel is
    # fixed. The coefficients are frozen at the production energy here where the
    # Monte Carlo sees them run, which is most of the residual.
    for log10_energy, sigma_mc in _MC_SIGMA_KM.items():
        variance = stochastic_muon_range_variance_km2(10.0**log10_energy, 1.0e5)[0]
        assert np.sqrt(variance) == pytest.approx(sigma_mc, rel=0.07)


def test_range_variance_needs_the_overshoot_constant():
    # Without the constant the expansion is a large-w asymptote and runs high.
    # This is what makes the constant worth carrying rather than dropping.
    energy = 1.0e6
    phi_prime, phi_second, _ = (float(m[0]) for m in log_loss_moments(energy))
    leading_only = phi_second * np.log(energy / 1.0e5) / phi_prime**3
    with_constant = stochastic_muon_range_variance_km2(energy, 1.0e5)[0]
    assert leading_only > with_constant
    assert np.sqrt(leading_only) / _MC_SIGMA_KM[6.0] > 1.15
    assert np.sqrt(with_constant) / _MC_SIGMA_KM[6.0] == pytest.approx(1.0, abs=0.05)


def test_range_variance_beats_the_second_order_form():
    # The drift-diffusion transport carries d_mu = <y^2> where the range needs
    # <ln^2(1-y)>. The two differ by a factor of about four, so its spread is
    # low at every lever arm and gets worse with distance.
    energies = 10.0 ** np.array([6.0, 7.0, 7.5])
    b_mu = drift_coefficient(energies)
    d_mu = diffusion_coefficient(energies)
    drift = b_mu + 0.5 * d_mu
    w = np.log(energies / 1.0e5)
    second_order = np.sqrt(w * d_mu / drift**3)
    exact = np.sqrt(stochastic_muon_range_variance_km2(energies, 1.0e5))
    measured = np.array([_MC_SIGMA_KM[6.0], _MC_SIGMA_KM[7.0], _MC_SIGMA_KM[7.5]])
    assert np.all(second_order < 0.75 * measured)
    assert np.all(np.abs(exact / measured - 1.0) < 0.07)
    # And the deficit widens rather than closing.
    assert np.all(np.diff(second_order / measured) < 0.0)


def test_range_variance_grows_with_the_lever_arm():
    energy = np.logspace(5.5, 8.0, 12)
    variance = stochastic_muon_range_variance_km2(energy, 1.0e5)
    assert np.all(np.diff(variance) > 0.0)


def test_range_variance_scales_with_the_kernel():
    # Every moment is linear in the kernel normalization, so a rescaled kernel
    # scales the variance by 1 / b_scale^2 exactly.
    energy = np.array([1.0e6, 1.0e7])
    base = stochastic_muon_range_variance_km2(energy, 1.0e5)
    scaled = stochastic_muon_range_variance_km2(energy, 1.0e5, b_scale=2.0)
    assert np.allclose(scaled, base / 4.0)


def test_range_variance_vanishes_below_threshold():
    assert stochastic_muon_range_variance_km2(np.array([0.5e3]))[0] == 0.0
    # And it floors at zero rather than going negative where the expansion in
    # 1 / w stops applying, a few e-folds above threshold.
    assert stochastic_muon_range_variance_km2(np.array([1.2e3]))[0] == 0.0


def test_log_loss_moments_exceed_the_y_moments():
    # -ln(1-y) >= y for every positive loss spectrum, and the gap grows with the
    # order because the logarithm diverges where y saturates.
    energy = np.array([1.0e5, 1.0e6, 1.0e7])
    phi_prime, phi_second, phi_third = log_loss_moments(energy)
    assert np.all(phi_prime > drift_coefficient(energy))
    assert np.all(phi_second > diffusion_coefficient(energy))
    second_gap = phi_second / diffusion_coefficient(energy)
    first_gap = phi_prime / drift_coefficient(energy)
    assert np.all(second_gap > first_gap)
    assert np.all(phi_third > 0.0)
