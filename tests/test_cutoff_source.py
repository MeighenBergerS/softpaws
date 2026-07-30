"""Tests for the exponential-cutoff source treatment (App. H).

Cover the real-space log-loss convolution (:mod:`softpaws.transport.
cutoff_source`) and its wiring into ``SoftVolumeResponse``: the
``e0 -> inf`` reduction to the plain (uncut) treatment, cutoff suppression at
high energy, and -- the whole point of App. H -- that it stays finite and
positive in the ``A < 0``, large-column regime where the plain treatment's
saturation factor blows up (``docs/2026_softvolume_vs_implementation.md``
S:2.3).
"""

import numpy as np
import pytest

from softpaws.response.soft_volume import SoftVolumeResponse
from softpaws.transport.coefficients import diffusion_coefficient, drift_coefficient
from softpaws.transport.cutoff_source import cutoff_soft_rate_density
from softpaws.transport.eigenvalue import phi_eigenvalue
from softpaws.transport.soft_volume import saturation_factor

GAMMA_IC = 2.38
LAMBDA_IC = 0.4
PHI0_IC = 0.63
E_1PEV = 1.0e6
X_MILD = 1.95
X_EXTREME = 1.0e4


# ---------------------------------------------------------------------------
# cutoff_soft_rate_density: the analytic identity E[e^{-A w(ell)}] = e^{-ell Phi(A)}
# ---------------------------------------------------------------------------


def test_cutoff_soft_rate_density_matches_analytic_limit_at_large_e0():
    # For a pure power-law weak-rate density and e0 -> inf, App. B's exact
    # moment identity E[e^{-A w(ell)}] = e^{-ell Phi(A)} collapses the whole
    # xi-integral to the closed-form saturation factor -- the same identity
    # SoftVolumeResponse.differential_rate_with_cutoff relies on to recover
    # the plain (uncut) treatment.
    b_mu = float(drift_coefficient(E_1PEV)[0])
    d_mu = float(diffusion_coefficient(E_1PEV)[0])
    a = 0.98
    x = X_MILD

    def weak_rate_density(eps):
        return eps ** (-1.0 - a)

    result = cutoff_soft_rate_density(
        1.0, 1.0e16, x, b_mu, d_mu, weak_rate_density, n_xi=8, n_w=256, n_k=2048,
    )
    phi_a = float(phi_eigenvalue(a, b_mu, d_mu))
    expected = weak_rate_density(np.array([1.0]))[0] * float(saturation_factor(phi_a, x))
    assert result[0] == pytest.approx(expected, rel=0.03)


# ---------------------------------------------------------------------------
# Response-layer wiring
# ---------------------------------------------------------------------------


def test_cutoff_requires_exact_method():
    resp = SoftVolumeResponse(radius_km=0.62, method="drift")
    with pytest.raises(ValueError):
        resp.differential_rate_with_cutoff(E_1PEV, PHI0_IC, GAMMA_IC, e0_cutoff_gev=1.0e8)


def test_cutoff_requires_finite_column():
    resp = SoftVolumeResponse(radius_km=0.62, method="exact", column_depth_km=None)
    with pytest.raises(ValueError):
        resp.differential_rate_with_cutoff(E_1PEV, PHI0_IC, GAMMA_IC, e0_cutoff_gev=1.0e8)


def test_cutoff_rejects_tabulated_cross_section():
    from softpaws.transport.cross_section import bgr18_cross_section

    resp = SoftVolumeResponse(
        radius_km=0.62, method="exact", column_depth_km=X_MILD,
        cross_section=bgr18_cross_section(),
    )
    with pytest.raises(ValueError):
        resp.differential_rate_with_cutoff(E_1PEV, PHI0_IC, GAMMA_IC, e0_cutoff_gev=1.0e8)


def test_cutoff_inside_population_matches_direct_formula():
    # The in-detector population needs no convolution: I(A) e^{-E/E0} V_det
    # times the uncut weak-rate density, evaluated directly at E.
    resp = SoftVolumeResponse(radius_km=0.62, method="exact", column_depth_km=X_MILD)
    e = np.array([E_1PEV])
    e0 = 1.0e7
    inside = resp.differential_rate_with_cutoff(
        e, PHI0_IC, GAMMA_IC, e0_cutoff_gev=e0, part="inside"
    )
    plain_inside = resp.differential_rate(e, PHI0_IC, GAMMA_IC, part="inside")
    expected = plain_inside * np.exp(-e / e0)
    np.testing.assert_allclose(inside, expected, rtol=1e-9)


def test_cutoff_total_is_inside_plus_soft():
    resp = SoftVolumeResponse(radius_km=0.62, method="exact", column_depth_km=X_MILD)
    e = np.array([E_1PEV])
    kwargs = dict(e0_cutoff_gev=1.0e8)
    total = resp.differential_rate_with_cutoff(e, PHI0_IC, GAMMA_IC, **kwargs)
    inside = resp.differential_rate_with_cutoff(e, PHI0_IC, GAMMA_IC, part="inside", **kwargs)
    soft = resp.differential_rate_with_cutoff(e, PHI0_IC, GAMMA_IC, part="soft", **kwargs)
    np.testing.assert_allclose(total, inside + soft, rtol=1e-9)


def test_cutoff_reduces_to_plain_treatment_as_e0_grows():
    # As E0 -> inf the cutoff source becomes the plain power law, and both
    # populations should converge to the uncut differential_rate.
    resp = SoftVolumeResponse(radius_km=0.62, method="exact", column_depth_km=X_MILD)
    e = np.array([1.0e5])
    plain = resp.differential_rate(e, PHI0_IC, GAMMA_IC, part="total")
    ratios = [
        float(
            resp.differential_rate_with_cutoff(e, PHI0_IC, GAMMA_IC, e0_cutoff_gev=e0)[0]
            / plain[0]
        )
        for e0 in (1.0e10, 1.0e13, 1.0e16)
    ]
    # Monotonically approaching 1 as E0 grows, and already close at the largest.
    assert ratios[-1] == pytest.approx(1.0, abs=0.03)
    assert ratios[0] < ratios[1] < ratios[2] or all(
        abs(r - 1.0) < 0.03 for r in ratios
    )


def test_cutoff_suppresses_flux_above_e0():
    resp = SoftVolumeResponse(radius_km=0.62, method="exact", column_depth_km=X_MILD)
    e0 = 1.0e7
    energy = np.array([1.0e5, 1.0e6, 1.0e7, 1.0e8])
    rate = resp.differential_rate_with_cutoff(energy, PHI0_IC, GAMMA_IC, e0_cutoff_gev=e0)
    plain = resp.differential_rate(energy, PHI0_IC, GAMMA_IC)
    ratio = rate / plain
    # Suppression grows monotonically once E approaches/exceeds E0.
    assert np.all(np.diff(ratio) < 0.0)
    assert ratio[0] > 0.9  # E << E0: negligible suppression
    assert ratio[-1] < 0.5  # E >= E0: strongly suppressed


def test_cutoff_stays_finite_past_the_pole_with_large_column():
    # The paper's own motivating pathology: A < 0 (lambda past the
    # cross-section pole) with a realistic Earth-crossing column blows the
    # plain saturation factor up to an astronomical (but finite, per the
    # saturation factor's own analytic continuation) value; the cutoff
    # treatment should instead give a modest, finite, positive, and
    # monotonically falling rate.
    resp = SoftVolumeResponse(radius_km=0.62, method="exact", column_depth_km=X_EXTREME)
    energy = np.logspace(5.0, 7.0, 5)
    rate = resp.differential_rate_with_cutoff(
        energy, PHI0_IC, GAMMA_IC, e0_cutoff_gev=1.0e7, lam=1.5,
    )
    assert np.all(np.isfinite(rate))
    assert np.all(rate > 0.0)
    assert np.all(np.diff(rate) < 0.0)


def test_expected_counts_with_cutoff_matches_hand_integrated_reference():
    resp = SoftVolumeResponse(radius_km=0.62, method="exact", column_depth_km=X_MILD)
    edges = np.array([5.0, 5.3])
    counts = resp.expected_counts_with_cutoff(
        edges, PHI0_IC, GAMMA_IC, e0_cutoff_gev=1.0e8, livetime_s=1.0e7,
        solid_angle_sr=2.0 * np.pi, n_subdivisions=24,
    )
    energy = np.logspace(5.0, 5.3, 24)
    rate = resp.differential_rate_with_cutoff(
        energy, PHI0_IC, GAMMA_IC, e0_cutoff_gev=1.0e8,
    )
    expected = np.trapezoid(rate, energy) * 1.0e7 * 2.0 * np.pi
    assert counts[0] == pytest.approx(expected, rel=1e-9)
