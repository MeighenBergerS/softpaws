"""Tests for the tau-neutrino contribution to the through-going muon sample.

Cover the decay-length normalization against the paper-adjacent literature
value (``~5 km`` at 100 PeV), the tau decay spectrum's closed-form moments
against direct quadrature, that the tau/numu ratio grows with energy the way
``docs`` motivates it (a few percent at 1 PeV, tens of percent at 100 PeV), and
the closed-form single-event log-loss law ``tau_loss_density`` -- in
particular the total-mass identity that pins down its derivation
(``integral P(w) dw = tau_survival_before_decay``) in the limits where that
identity is easiest to check by hand.
"""

import numpy as np
import pytest

from softpaws.response.soft_volume import (
    SoftVolumeResponse,
    tau_induced_differential_rate,
    tau_induced_expected_counts,
    tau_induced_expected_counts_attenuated,
)
from softpaws.transport.attenuation import effective_solid_angle
from softpaws.transport.coefficients import diffusion_coefficient, drift_coefficient
from softpaws.transport.eigenvalue import spectral_index
from softpaws.transport.tau import (
    BR_TAU_TO_MU,
    MEAN_Z,
    decay_length_km,
    decay_spectrum,
    tau_loss_density,
    tau_survival_before_decay,
    tau_to_muon_ratio,
    z_moment,
    z_symbol,
)

GAMMA_IC = 2.38  # IceCube 9.5 yr diffuse-flux best fit
A_IC = spectral_index(GAMMA_IC)  # ~0.98


# ---------------------------------------------------------------------------
# Decay length
# ---------------------------------------------------------------------------


def test_decay_length_matches_reference_value_at_100_pev():
    # ell_tau ~ 5 km at E_tau = 100 PeV (order-of-magnitude literature value).
    assert decay_length_km(1.0e8) == pytest.approx(4.9, rel=0.05)


def test_decay_length_scales_linearly_with_energy():
    ratio = decay_length_km(2.0e8) / decay_length_km(1.0e8)
    assert ratio == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Decay spectrum and its Mellin moments
# ---------------------------------------------------------------------------


def test_decay_spectrum_normalized():
    z = np.linspace(0.0, 1.0, 20001)
    assert np.trapezoid(decay_spectrum(z), z) == pytest.approx(1.0, rel=1e-6)


def test_z_moment_at_zero_is_normalization():
    assert z_moment(0.0) == pytest.approx(1.0)


def test_z_moment_at_one_is_mean_z():
    assert z_moment(1.0) == pytest.approx(MEAN_Z)
    assert MEAN_Z == pytest.approx(0.3)


def test_z_moment_matches_quadrature():
    z = np.linspace(1e-6, 1.0, 20001)
    for a in (0.5, 0.98, 2.0):
        direct = np.trapezoid(z**a * decay_spectrum(z), z)
        assert z_moment(a) == pytest.approx(direct, rel=1e-4)


def test_z_symbol_preserves_complex_dtype():
    # tau_loss_density needs <z^s> at s = -i k; the real z_moment would drop Im.
    value = z_symbol(-1j * np.array([0.0, 1.0, 2.0]))
    assert np.iscomplexobj(value)
    assert value[0] == pytest.approx(1.0)  # s = 0 is the normalization


def test_z_symbol_matches_z_moment_on_the_real_line():
    a = np.array([0.0, 0.5, 0.98, 2.0])
    np.testing.assert_allclose(z_symbol(a), z_moment(a))


# ---------------------------------------------------------------------------
# Tau / numu ratio
# ---------------------------------------------------------------------------


def test_tau_ratio_is_percent_level_at_1_pev():
    ratio = tau_to_muon_ratio(A_IC, 1.0e6)
    assert 0.02 < ratio < 0.10


def test_tau_ratio_grows_with_energy():
    # ell_tau grows linearly with energy while Phi(A) is roughly flat, so the
    # bracket [1 + ell_tau Phi(A)] -- and hence the ratio -- rises with energy.
    low = tau_to_muon_ratio(A_IC, 1.0e6)
    high = tau_to_muon_ratio(A_IC, 1.0e8)
    assert high > 3.0 * low


def test_tau_ratio_bounded_by_branching_ratio_times_moment_at_low_energy():
    # As E -> 0 the decay length -> 0, so the bracket -> 1 and the ratio ->
    # B_tau_to_mu * <z^A>.
    ratio = tau_to_muon_ratio(A_IC, 1.0)
    assert ratio == pytest.approx(BR_TAU_TO_MU * z_moment(A_IC), rel=1e-2)


# ---------------------------------------------------------------------------
# Response-level wrapper
# ---------------------------------------------------------------------------


def test_tau_induced_rate_requires_exact_method():
    drift = SoftVolumeResponse(radius_km=0.62, method="drift")
    with pytest.raises(ValueError):
        tau_induced_differential_rate(drift, 1.0e6, phi0=0.63, gamma=GAMMA_IC)


def test_tau_induced_rate_is_small_fraction_of_numu():
    exact = SoftVolumeResponse(radius_km=0.62, method="exact", column_depth_km=1.95)
    energy = 1.0e6
    numu_rate = exact.differential_rate(energy, phi0=0.63, gamma=GAMMA_IC)
    tau_rate = tau_induced_differential_rate(exact, energy, phi0=0.63, gamma=GAMMA_IC)
    assert 0.0 < tau_rate[0] < 0.1 * numu_rate[0]


def test_tau_induced_expected_counts_positive_and_growing_fraction():
    exact = SoftVolumeResponse(radius_km=0.62, method="exact", column_depth_km=1.95)
    edges = np.array([5.0, 6.0, 7.0, 8.0])  # 100 TeV to 1 EeV
    livetime_s = 9.5 * 365.25 * 86400.0
    solid_angle_sr = 2.0 * np.pi

    numu_counts = exact.expected_counts(edges, 0.63, GAMMA_IC, livetime_s, solid_angle_sr)
    tau_counts = tau_induced_expected_counts(
        exact, edges, 0.63, GAMMA_IC, livetime_s, solid_angle_sr
    )
    assert np.all(tau_counts > 0.0)
    fraction = tau_counts / numu_counts
    # The tau fraction should increase from the lowest to the highest bin.
    assert fraction[-1] > fraction[0]


def test_tau_induced_expected_counts_attenuated_matches_hand_computed():
    exact = SoftVolumeResponse(radius_km=0.62, method="exact", column_depth_km=1.95)
    edges = np.array([5.0, 6.0])
    livetime_s = 3.0
    dec_min, dec_max = 0.0, 90.0
    n_subdivisions, n_dec = 32, 32

    counts = tau_induced_expected_counts_attenuated(
        exact, edges, 0.63, GAMMA_IC, livetime_s, dec_min, dec_max,
        n_subdivisions=n_subdivisions, n_dec=n_dec,
    )

    energy = np.logspace(edges[0], edges[1], n_subdivisions)
    rate = tau_induced_differential_rate(exact, energy, 0.63, GAMMA_IC)
    omega_eff = effective_solid_angle(energy, dec_min, dec_max, n_dec=n_dec)
    expected = np.trapezoid(rate * omega_eff, energy) * livetime_s
    assert counts[0] == pytest.approx(expected, rel=1e-9)


def test_tau_induced_expected_counts_attenuated_rejects_closed_form_column():
    exact = SoftVolumeResponse(
        radius_km=0.62, method="exact", column_depth_km=1.95, attenuation_column_g_cm2=1.0e5,
    )
    with pytest.raises(ValueError):
        tau_induced_expected_counts_attenuated(
            exact, np.array([5.0, 6.0]), 0.63, GAMMA_IC, 1.0, 0.0, 90.0
        )


# ---------------------------------------------------------------------------
# Single-event log-loss law (tau_loss_density)
# ---------------------------------------------------------------------------

E_1PEV = 1.0e6  # GeV, observed muon energy for these tests
B_MU = float(drift_coefficient(E_1PEV)[0])
D_MU = float(diffusion_coefficient(E_1PEV)[0])


def test_tau_survival_matches_total_mass_of_density():
    # Psi(0) = integral P(w) dw = tau_survival_before_decay, by construction
    # (see the module derivation): the fixed-hypothesis total-mass identity.
    ell_km = 3.0
    e_tau = 5.0e7  # 50 PeV: ell_tau ~ 2.4 km, comparable to ell_km -- the
    # regime where the truncation actually matters.
    w = np.linspace(1e-3, 15.0, 2000)
    density = tau_loss_density(w, ell_km, e_tau, B_MU, D_MU, n_k=2**12)
    mass = np.trapezoid(density, w)
    expected = tau_survival_before_decay(ell_km, e_tau)
    assert mass == pytest.approx(expected, rel=0.02)


def test_tau_loss_density_mass_to_one_as_tau_decays_immediately():
    # ell_tau -> 0 (tiny tau energy): the tau decays right at production, so
    # the whole column is available and the tau always decays before ell.
    ell_km = 3.0
    e_tau = 1.0
    w = np.linspace(1e-3, 20.0, 2000)
    density = tau_loss_density(w, ell_km, e_tau, B_MU, D_MU, n_k=2**12)
    mass = np.trapezoid(density, w)
    assert mass == pytest.approx(1.0, abs=1e-3)


def test_tau_loss_density_mass_to_zero_for_huge_tau_energy():
    # ell_tau >> ell (absurdly high tau energy): the tau essentially never
    # decays before reaching the detector, so no muon is produced this way.
    ell_km = 3.0
    e_tau = 1.0e14
    w = np.linspace(1e-3, 20.0, 2000)
    density = tau_loss_density(w, ell_km, e_tau, B_MU, D_MU, n_k=2**12)
    mass = np.trapezoid(density, w)
    assert mass == pytest.approx(0.0, abs=1e-3)


def test_tau_loss_density_nonnegative():
    ell_km = 3.0
    e_tau = 5.0e7
    w = np.linspace(1e-3, 15.0, 2000)
    density = tau_loss_density(w, ell_km, e_tau, B_MU, D_MU, n_k=2**12)
    assert np.all(density >= 0.0)


def test_tau_loss_density_diagonal_mode_shape():
    # e_tau_gev may broadcast against w_grid (the reconstruction use case,
    # where the tau-energy hypothesis is tied to the w grid point itself).
    ell_km = 3.0
    w = np.linspace(1e-3, 10.0, 500)
    e_tau_diag = E_1PEV * np.exp(w)
    density = tau_loss_density(w, ell_km, e_tau_diag, B_MU, D_MU)
    assert density.shape == w.shape
    assert np.all(np.isfinite(density))
