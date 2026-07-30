"""Tests for the exact collision eigenvalue and the exact soft volume.

These lock in the properties derived in ``docs/exact_soft_volume_notes.md``
(and ``docs/2026_softvolume.pdf``): the exactness identities ``Phi(1) = b_mu``
and ``Phi(2) = 2 b_mu - d_mu``, agreement of the closed form with a direct
quadrature, monotonic growth of ``Phi`` where the Fokker-Planck truncation turns
over, and the finite-column saturation factor that cures the drift-form
divergence at and beyond the cross-section pole.
"""

import numpy as np
import pytest

from softpaws.transport.coefficients import (
    diffusion_coefficient,
    drift_coefficient,
    loss_spectrum_y_grid,
    third_moment_coefficient,
)
from softpaws.transport.eigenvalue import (
    phi_drift,
    phi_eigenvalue,
    phi_eigenvalue_at_energy,
    phi_eigenvalue_derivative,
    phi_eigenvalue_quadrature,
    phi_eigenvalue_three_moment,
    phi_fokker_planck,
    phi_symbol,
    phi_symbol_three_moment,
    spectral_index,
    three_moment_loss_spectrum,
    two_moment_loss_spectrum,
)
from softpaws.transport.soft_volume import (
    saturation_factor,
    scale_breaking_saturation_factor,
    soft_volume_attenuated_exact,
    soft_volume_drift,
    soft_volume_exact,
)
from softpaws.transport.source import inelasticity_factor

E_1PEV = 1.0e6  # GeV
GAMMA_IC = 2.38
LAMBDA_IC = 0.4
B_1PEV = 0.35
D_1PEV = 0.0766


# ---------------------------------------------------------------------------
# Spectral index
# ---------------------------------------------------------------------------


def test_spectral_index_value():
    assert spectral_index(GAMMA_IC, LAMBDA_IC) == pytest.approx(0.98, abs=1e-9)


def test_spectral_index_allows_nonpositive():
    # Unlike spectral_penalty, the exact path does not reject A <= 0.
    assert spectral_index(1.2, LAMBDA_IC) == pytest.approx(-0.2, abs=1e-9)


# ---------------------------------------------------------------------------
# Exactness identities: Phi(1) = b_mu, Phi(2) = 2 b_mu - d_mu
# ---------------------------------------------------------------------------


def test_phi_identity_at_one():
    assert float(phi_eigenvalue(1.0, B_1PEV, D_1PEV)) == pytest.approx(B_1PEV, rel=1e-12)


def test_phi_identity_at_two():
    expected = 2.0 * B_1PEV - D_1PEV
    assert float(phi_eigenvalue(2.0, B_1PEV, D_1PEV)) == pytest.approx(expected, rel=1e-12)


def test_phi_vanishes_at_zero():
    # Phi(0) = 0 identically: a monochromatic source (a dark matter line, App.
    # I) has no continuum spectral index to attenuate against, so the s=0
    # line-of-sight propagator e^{-l Phi(0)} is trivially 1 for every l --
    # the identity examples/23_dm_lines.py's App. I treatment relies on.
    assert float(phi_eigenvalue(0.0, B_1PEV, D_1PEV)) == pytest.approx(0.0, abs=1e-12)


def test_phi_matches_fokker_planck_near_a_one():
    # At A = 0.98 the truncation error is ~0.2% (Part 10.2).
    a = 0.98
    exact = float(phi_eigenvalue(a, B_1PEV, D_1PEV))
    fp = float(phi_fokker_planck(a, B_1PEV, D_1PEV))
    assert exact == pytest.approx(fp, rel=3e-3)


def test_phi_grows_where_fokker_planck_turns_over():
    # The exact Phi grows monotonically; A' b_mu peaks (~A=5) then falls (Part 10.4).
    a = np.array([1.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    exact = phi_eigenvalue(a, B_1PEV, D_1PEV)
    assert np.all(np.diff(exact) > 0)
    fp = phi_fokker_planck(a, B_1PEV, D_1PEV)
    # Fokker-Planck turns over and even goes negative past A ~ 10.
    assert not np.all(np.diff(fp) > 0)


def test_phi_drift_is_leading_term():
    assert float(phi_drift(1.0, B_1PEV)) == pytest.approx(B_1PEV, rel=1e-12)


# ---------------------------------------------------------------------------
# Closed form vs direct quadrature of the same two-moment family
# ---------------------------------------------------------------------------


def test_closed_form_matches_quadrature():
    kappa, p = two_moment_loss_spectrum(B_1PEV, D_1PEV)

    def dgamma_dy(y):
        return kappa * (1.0 - y) ** p / y

    a = np.array([0.5, 0.98, 2.0, 3.0])
    closed = phi_eigenvalue(a, B_1PEV, D_1PEV)
    quad = phi_eigenvalue_quadrature(a, dgamma_dy)
    np.testing.assert_allclose(quad, closed, rtol=2e-3)


def test_quadrature_scalar_returns_scalar_shape():
    kappa, p = two_moment_loss_spectrum(B_1PEV, D_1PEV)
    phi = phi_eigenvalue_quadrature(1.0, lambda y: kappa * (1.0 - y) ** p / y)
    assert phi.shape == ()


def test_phi_at_energy_uses_table1_coefficients():
    a = 0.98
    phi = phi_eigenvalue_at_energy(a, E_1PEV)
    direct = phi_eigenvalue(a, drift_coefficient(E_1PEV), diffusion_coefficient(E_1PEV))
    np.testing.assert_allclose(phi, direct, rtol=1e-12)


# ---------------------------------------------------------------------------
# Saturation factor
# ---------------------------------------------------------------------------


def test_saturation_factor_limit_at_zero():
    # (1 - e^{-Phi x}) / Phi -> x as Phi -> 0.
    assert float(saturation_factor(0.0, 2.0)) == pytest.approx(2.0, rel=1e-9)
    assert float(saturation_factor(1e-15, 2.0)) == pytest.approx(2.0, rel=1e-6)


def test_saturation_factor_matches_table():
    # 1 - e^{-Phi x} at Phi = 0.35/km for the depths in Part 7.4.
    phi = 0.35
    x = np.array([1.0, 2.0, 3.0, 5.0, 10.0])
    frac = 1.0 - np.exp(-phi * x)
    np.testing.assert_allclose(frac, [0.30, 0.50, 0.65, 0.82, 0.97], atol=0.01)


def test_saturation_factor_finite_for_negative_phi():
    # For Phi < 0 it continues to (e^{|Phi| x} - 1) / |Phi|, still positive/finite.
    phi = -0.2
    x = 2.0
    val = float(saturation_factor(phi, x))
    assert val == pytest.approx((np.expm1(abs(phi) * x)) / abs(phi), rel=1e-9)
    assert np.isfinite(val) and val > 0


# ---------------------------------------------------------------------------
# Coupled attenuation R_nu(x, A) (Eq. 11)
# ---------------------------------------------------------------------------


def test_r_nu_reduces_to_plain_saturation_at_zero_inv_lambda():
    phi, x = 0.35, 2.5
    plain = float(saturation_factor(phi, x))
    coupled = float(saturation_factor(phi, x, inv_lambda_per_km=0.0))
    assert coupled == pytest.approx(plain, rel=1e-12)


def test_r_nu_matches_closed_form_away_from_pole():
    phi, x, inv_lambda = 0.35, 2.0, 0.05
    expected = (np.exp(-inv_lambda * x) - np.exp(-phi * x)) / (phi - inv_lambda)
    assert float(saturation_factor(phi, x, inv_lambda)) == pytest.approx(expected, rel=1e-12)


def test_r_nu_lhopital_matches_numerical_limit_at_pole():
    x = 3.0
    phi = 0.4
    inv_lambda_near = phi + 1e-7  # just off the removable singularity
    numerical = float(saturation_factor(phi, x, inv_lambda_near))
    at_pole = float(saturation_factor(phi, x, phi))
    assert at_pole == pytest.approx(numerical, rel=1e-5)
    assert at_pole == pytest.approx(x * np.exp(-phi * x), rel=1e-9)


def test_r_nu_transparent_earth_limit_as_phi_vanishes():
    # Phi -> 0: R_nu -> (1 - e^{-x/Lambda_nu}) / (1/Lambda_nu), the plain
    # neutrino-survival integral with no muon transport loss at all.
    x, inv_lambda = 4.0, 0.1
    expected = (1.0 - np.exp(-inv_lambda * x)) / inv_lambda
    assert float(saturation_factor(0.0, x, inv_lambda)) == pytest.approx(expected, rel=1e-9)


def test_r_nu_setting_inv_lambda_to_zero_recovers_unattenuated_range():
    # Eq. 11's own stated limit: 1/Lambda_nu -> 0 recovers the transparent-Earth
    # effective range identically (App. C.3).
    phi, x = 0.2, 5.0
    assert float(saturation_factor(phi, x, 0.0)) == pytest.approx(
        float(saturation_factor(phi, x)), rel=1e-12
    )


# ---------------------------------------------------------------------------
# Exact soft volume
# ---------------------------------------------------------------------------


def test_exact_infinite_column_reduces_to_range_over_phi():
    r = 0.62
    v = soft_volume_exact(r, E_1PEV, GAMMA_IC, include_inelasticity=False, source="table1")
    phi = float(phi_eigenvalue(spectral_index(GAMMA_IC, LAMBDA_IC), B_1PEV, D_1PEV))
    expected = np.pi * r**2 / phi
    np.testing.assert_allclose(v, expected, rtol=1e-9)


def test_exact_close_to_drift_near_a_one():
    # At A = 0.98, Phi(A) ~ b_mu A, so the exact (no-I) volume ~ the drift volume.
    r = 0.62
    v_exact = soft_volume_exact(r, E_1PEV, GAMMA_IC, include_inelasticity=False)
    v_drift = soft_volume_drift(r, E_1PEV, GAMMA_IC)
    np.testing.assert_allclose(v_exact, v_drift, rtol=0.02)


def test_exact_inelasticity_reduces_volume():
    r = 0.62
    with_i = soft_volume_exact(r, E_1PEV, GAMMA_IC, include_inelasticity=True)
    without_i = soft_volume_exact(r, E_1PEV, GAMMA_IC, include_inelasticity=False)
    assert with_i[0] < without_i[0]


def test_exact_finite_column_below_infinite():
    r = 0.62
    finite = soft_volume_exact(r, E_1PEV, GAMMA_IC, column_depth_km=1.95)
    infinite = soft_volume_exact(r, E_1PEV, GAMMA_IC, column_depth_km=None)
    assert finite[0] < infinite[0]


def test_exact_infinite_column_raises_past_pole():
    # lambda > gamma - 1 -> A < 0 -> Phi(A) < 0; infinite column diverges.
    with pytest.raises(ValueError):
        soft_volume_exact(0.62, E_1PEV, GAMMA_IC, lam=1.5, column_depth_km=None)


def test_exact_finite_column_survives_past_pole():
    # With a finite column the saturation factor keeps it finite and positive.
    v = soft_volume_exact(0.62, E_1PEV, GAMMA_IC, lam=1.5, column_depth_km=1.95)
    assert np.all(np.isfinite(v)) and v[0] > 0


# ---------------------------------------------------------------------------
# Coupled attenuated soft volume (Eq. 11)
# ---------------------------------------------------------------------------


def test_attenuated_exact_reduces_to_plain_exact_at_zero_inv_lambda():
    # 1/Lambda_nu -> 0 must recover soft_volume_exact's V_det and V_soft exactly
    # (App. C.3's stated limit).
    r = 0.62
    x = 1.95
    v_det_eff, v_soft = soft_volume_attenuated_exact(
        r, E_1PEV, GAMMA_IC, x, inv_lambda_nu_per_km=0.0, source="table1",
    )
    v_soft_plain = soft_volume_exact(r, E_1PEV, GAMMA_IC, column_depth_km=x, source="table1")
    a = spectral_index(GAMMA_IC, LAMBDA_IC)
    v_det_plain = float(inelasticity_factor(a)) * (4.0 / 3.0 * np.pi * r**3)
    np.testing.assert_allclose(v_soft, v_soft_plain, rtol=1e-9)
    np.testing.assert_allclose(v_det_eff, v_det_plain, rtol=1e-9)


def test_attenuated_exact_shrinks_with_stronger_absorption():
    # A shorter interaction length must reduce both terms.
    r = 0.62
    x = 100.0
    weak = soft_volume_attenuated_exact(r, E_1PEV, GAMMA_IC, x, inv_lambda_nu_per_km=1e-4)
    strong = soft_volume_attenuated_exact(r, E_1PEV, GAMMA_IC, x, inv_lambda_nu_per_km=1e-2)
    assert strong[0][0] < weak[0][0]  # V_det term
    assert strong[1][0] < weak[1][0]  # V_soft term


def test_attenuated_exact_positive_and_finite_past_the_pole():
    # A < 0 (Phi(A) < 0) plus attenuation must stay finite and positive, same
    # protection soft_volume_exact's saturation factor already provides.
    v_det_eff, v_soft = soft_volume_attenuated_exact(
        0.62, E_1PEV, GAMMA_IC, column_depth_km=50.0, inv_lambda_nu_per_km=1e-3, lam=1.5,
    )
    assert np.all(np.isfinite(v_det_eff)) and np.all(v_det_eff > 0)
    assert np.all(np.isfinite(v_soft)) and np.all(v_soft > 0)


# ---------------------------------------------------------------------------
# Eigenvalue derivative Phi'(A) (App. F, Eq. F4)
# ---------------------------------------------------------------------------


def test_phi_derivative_matches_finite_difference():
    a = 0.98
    da = 1e-4
    phi_plus = float(phi_eigenvalue(a + da, B_1PEV, D_1PEV))
    phi_minus = float(phi_eigenvalue(a - da, B_1PEV, D_1PEV))
    numeric = (phi_plus - phi_minus) / (2.0 * da)
    analytic = float(phi_eigenvalue_derivative(a, B_1PEV, D_1PEV))
    assert analytic == pytest.approx(numeric, rel=1e-5)


def test_phi_derivative_is_positive():
    # Bernstein-function property (App. B): Phi' > 0 everywhere.
    for a in (-0.5, 0.0, 0.5, 1.0, 2.0, 5.0):
        assert float(phi_eigenvalue_derivative(a, B_1PEV, D_1PEV)) > 0.0


def test_phi_derivative_drift_fallback():
    # d(A b_mu)/dA = b_mu when d_mu <= 0 (the same fallback phi_eigenvalue uses).
    assert float(phi_eigenvalue_derivative(1.5, B_1PEV, 0.0)) == pytest.approx(B_1PEV)


# ---------------------------------------------------------------------------
# Scale-breaking saturation factor (App. F, Eq. F4)
# ---------------------------------------------------------------------------


def test_scale_breaking_reduces_to_plain_at_zero_beta():
    phi, phi_prime, x = 0.35, 0.5, 2.0
    plain = float(saturation_factor(phi, x))
    corrected = float(scale_breaking_saturation_factor(phi, phi_prime, 0.0, x))
    assert corrected == pytest.approx(plain, rel=1e-9)


def test_scale_breaking_positive_beta_shrinks_range():
    # Phi' > 0 always, so beta > 0 steepens the running index with depth,
    # increasing losses and shrinking the effective range (Eq. F4's own
    # "the spectrum steepens with depth" statement).
    phi = float(phi_eigenvalue(0.98, B_1PEV, D_1PEV))
    phi_prime = float(phi_eigenvalue_derivative(0.98, B_1PEV, D_1PEV))
    x = 2.0
    plain = float(scale_breaking_saturation_factor(phi, phi_prime, 0.0, x))
    shrunk = float(scale_breaking_saturation_factor(phi, phi_prime, 0.05, x))
    grown = float(scale_breaking_saturation_factor(phi, phi_prime, -0.05, x))
    assert shrunk < plain < grown


def test_scale_breaking_matches_quadrature_by_hand():
    phi, phi_prime, beta, x = 0.35, 0.6, 0.03, 3.0
    ell = np.linspace(0.0, x, 4096)
    expected = np.trapezoid(np.exp(-(ell * phi + 0.5 * beta * ell**2 * phi * phi_prime)), ell)
    result = float(scale_breaking_saturation_factor(phi, phi_prime, beta, x, n_steps=4096))
    assert result == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# beta wired through soft_volume_exact (App. F)
# ---------------------------------------------------------------------------


def test_exact_beta_zero_matches_plain_bit_for_bit():
    r = 0.62
    plain = soft_volume_exact(r, E_1PEV, GAMMA_IC, column_depth_km=1.95)
    with_beta = soft_volume_exact(r, E_1PEV, GAMMA_IC, column_depth_km=1.95, beta=0.0)
    np.testing.assert_array_equal(plain, with_beta)


def test_exact_nonzero_beta_requires_finite_column():
    with pytest.raises(ValueError):
        soft_volume_exact(0.62, E_1PEV, GAMMA_IC, column_depth_km=None, beta=0.05)


def test_exact_positive_beta_shrinks_soft_volume():
    r = 0.62
    plain = soft_volume_exact(r, E_1PEV, GAMMA_IC, column_depth_km=1.95, beta=0.0)
    shrunk = soft_volume_exact(r, E_1PEV, GAMMA_IC, column_depth_km=1.95, beta=0.05)
    assert shrunk[0] < plain[0]


# ---------------------------------------------------------------------------
# Three-moment loss family
# ---------------------------------------------------------------------------

T_1PEV = float(third_moment_coefficient(E_1PEV)[0])
B_PROP_1PEV = float(drift_coefficient(E_1PEV)[0])
D_PROP_1PEV = float(diffusion_coefficient(E_1PEV)[0])


def test_three_moment_reproduces_its_moments():
    # kappa B(n + q, p + 1) must return the moments it was calibrated to. The
    # grid is log spaced at both ends: dGamma/dy ~ y^(q-1) with q < 0 is steeper
    # than 1/y, which a linear grid cannot resolve.
    kappa, q, p = three_moment_loss_spectrum(B_PROP_1PEV, D_PROP_1PEV, T_1PEV)
    y = loss_spectrum_y_grid(n_soft=200_000, n_hard=100_000, log10_y_min=-16.0)
    spectrum = float(kappa) * y ** (float(q) - 1.0) * (1.0 - y) ** float(p)
    for order, expected in ((1, B_PROP_1PEV), (2, D_PROP_1PEV), (3, T_1PEV)):
        assert np.trapezoid(y**order * spectrum, y) == pytest.approx(expected, rel=1e-3)


def test_three_moment_exactness_identities():
    # The binomial expansion of 1 - (1-y)^A terminates at A = 1, 2, 3, so those
    # three eigenvalues are fixed by the moments alone.
    for a, expected in (
        (1.0, B_PROP_1PEV),
        (2.0, 2.0 * B_PROP_1PEV - D_PROP_1PEV),
        (3.0, 3.0 * B_PROP_1PEV - 3.0 * D_PROP_1PEV + T_1PEV),
    ):
        phi = phi_eigenvalue_three_moment(a, B_PROP_1PEV, D_PROP_1PEV, T_1PEV)
        assert float(phi) == pytest.approx(expected, rel=1e-12)


def test_three_moment_soft_exponent_is_negative():
    # PROPOSAL's soft pile-up is steeper than 1/y (pair production), so the
    # calibrated q comes out below zero -- the two-moment family's q = 0 cannot
    # represent it.
    _, q, p = three_moment_loss_spectrum(B_PROP_1PEV, D_PROP_1PEV, T_1PEV)
    assert -1.0 < float(q) < 0.0
    # And the freed parameter lands on the hard end: p drops from ~ +2.1 to < 0,
    # so dGamma/dy no longer vanishes as y -> 1.
    assert -1.0 < float(p) < 0.0
    assert float(two_moment_loss_spectrum(B_PROP_1PEV, D_PROP_1PEV)[1]) > 2.0


def test_three_moment_reduces_to_two_moment_as_q_vanishes():
    kappa, p = two_moment_loss_spectrum(B_1PEV, D_1PEV)
    a = np.array([0.5, 1.7, 3.0, 6.0])
    np.testing.assert_allclose(
        phi_symbol_three_moment(a, float(kappa), 1.0e-12, float(p)),
        phi_symbol(a, kappa, p),
        rtol=1e-9,
    )


def test_three_moment_symbol_matches_quadrature():
    kappa, q, p = three_moment_loss_spectrum(B_PROP_1PEV, D_PROP_1PEV, T_1PEV)

    def dgamma_dy(y):
        return float(kappa) * y ** (float(q) - 1.0) * (1.0 - y) ** float(p)

    a = np.array([0.5, 2.0, 4.0])
    closed = phi_symbol_three_moment(a, float(kappa), float(q), float(p))
    # Log-spaced quadrature by hand: phi_eigenvalue_quadrature's linear grid
    # cannot resolve the y^(q-1) soft end.
    y = loss_spectrum_y_grid(n_soft=100_000, n_hard=50_000, log10_y_min=-16.0)
    quad = np.array(
        [np.trapezoid(dgamma_dy(y) * (1.0 - (1.0 - y) ** value), y) for value in a]
    )
    np.testing.assert_allclose(closed, quad, rtol=2e-3)


def test_three_moment_symbol_finite_for_large_imaginary_argument():
    # The inversion evaluates Phi(-i k) out to k ~ 1e3, where Gamma itself
    # overflows; the loggamma route must stay finite and grow.
    kappa, q, p = three_moment_loss_spectrum(B_PROP_1PEV, D_PROP_1PEV, T_1PEV)
    k = np.array([1.0, 100.0, 1000.0])
    phi = phi_symbol_three_moment(-1j * k, float(kappa), float(q), float(p))
    assert np.all(np.isfinite(phi))
    assert np.all(np.diff(phi.real) > 0.0)


def test_three_moment_rejects_inconsistent_moments():
    # t_mu below d_mu^2 / b_mu violates moment log-convexity and has no
    # positive-spectrum solution.
    with pytest.raises(ValueError, match="convergent domain|three-moment"):
        three_moment_loss_spectrum(0.38, 0.092, 0.0092)
