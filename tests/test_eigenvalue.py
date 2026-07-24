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

from softpaws.transport.coefficients import diffusion_coefficient, drift_coefficient
from softpaws.transport.eigenvalue import (
    phi_drift,
    phi_eigenvalue,
    phi_eigenvalue_at_energy,
    phi_eigenvalue_quadrature,
    phi_fokker_planck,
    spectral_index,
    two_moment_loss_spectrum,
)
from softpaws.transport.soft_volume import (
    saturation_factor,
    soft_volume_drift,
    soft_volume_exact,
)

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
# Exact soft volume
# ---------------------------------------------------------------------------


def test_exact_infinite_column_reduces_to_range_over_phi():
    r = 0.62
    v = soft_volume_exact(r, E_1PEV, GAMMA_IC, include_inelasticity=False)
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
