"""Tests for the drift-limit soft-volume event rate.

Cover the weak-source ingredients and the ``SoftVolumeResponse`` forward model:
its factorization into target volume times weak-rate density, and the
order-of-magnitude physics (an SM through-going rate at 100 PeV well below one
event for IceCube's exposure -- the KM3NeT tension of arXiv:2607.13143).
"""

import numpy as np
import pytest

from softpaws.response.soft_volume import (
    FLUX_PIVOT_GEV,
    GAMMA_EFF_BOUNDS,
    SoftVolumeResponse,
    _local_spectral_index,
    power_law_flux,
)
from softpaws.transport.source import (
    E0_CROSS_GEV,
    SIGMA0_CM2,
    cc_cross_section,
    nucleon_number_density,
)
from softpaws.utils.constants import AVOGADRO_PER_MOL, RHO_WATER_G_CM3

GAMMA_IC = 2.38
PHI0_IC = 0.63
E_1PEV = 1.0e6
E_100PEV = 1.0e8


# ---------------------------------------------------------------------------
# Weak-source ingredients
# ---------------------------------------------------------------------------


def test_cross_section_anchored_at_10pev():
    assert cc_cross_section(E0_CROSS_GEV)[0] == pytest.approx(SIGMA0_CM2, rel=1e-12)


def test_cross_section_power_law_scaling():
    # One decade up in energy scales by 10^lambda (default lambda = 0.4).
    ratio = cc_cross_section(1.0e8)[0] / cc_cross_section(1.0e7)[0]
    assert ratio == pytest.approx(10.0**0.4, rel=1e-12)


def test_nucleon_number_density_water():
    assert nucleon_number_density(RHO_WATER_G_CM3) == pytest.approx(
        RHO_WATER_G_CM3 * AVOGADRO_PER_MOL, rel=1e-12
    )


def test_power_law_flux_at_pivot():
    # At the pivot the flux is phi0 * 1e-18.
    assert power_law_flux(FLUX_PIVOT_GEV, PHI0_IC, GAMMA_IC)[0] == pytest.approx(
        PHI0_IC * 1e-18, rel=1e-12
    )


def test_power_law_flux_scaling():
    ratio = power_law_flux(1e6, PHI0_IC, GAMMA_IC)[0] / power_law_flux(1e5, PHI0_IC, GAMMA_IC)[0]
    assert ratio == pytest.approx(10.0 ** (-GAMMA_IC), rel=1e-12)


# ---------------------------------------------------------------------------
# SoftVolumeResponse
# ---------------------------------------------------------------------------


@pytest.fixture
def response():
    return SoftVolumeResponse(radius_km=0.62)  # IceCube-like, water


def test_v_det_is_about_one_km3(response):
    # 0.62 km sphere -> ~1 km^3 = 1e15 cm^3.
    assert response.v_det_cm3 == pytest.approx(1.0e15, rel=0.05)


def test_total_is_inside_plus_soft(response):
    e = np.array([E_1PEV, E_100PEV])
    total = response.differential_rate(e, PHI0_IC, GAMMA_IC, part="total")
    inside = response.differential_rate(e, PHI0_IC, GAMMA_IC, part="inside")
    soft = response.differential_rate(e, PHI0_IC, GAMMA_IC, part="soft")
    np.testing.assert_allclose(total, inside + soft, rtol=1e-12)


def test_rate_factorizes_into_volume_and_weak_density(response):
    e = E_1PEV
    rate = response.differential_rate(e, PHI0_IC, GAMMA_IC, part="soft")
    vol = response.target_volume_cm3(e, GAMMA_IC, part="soft")
    weak = response.weak_rate_density(e, PHI0_IC, GAMMA_IC)
    np.testing.assert_allclose(rate, vol * weak, rtol=1e-12)


def test_soft_to_inside_ratio_matches_volume_ratio(response):
    # At fixed energy the rate ratio is just the volume ratio.
    e = E_1PEV
    inside = response.differential_rate(e, PHI0_IC, GAMMA_IC, part="inside")[0]
    soft = response.differential_rate(e, PHI0_IC, GAMMA_IC, part="soft")[0]
    v_inside = response.target_volume_cm3(e, GAMMA_IC, part="inside")[0]
    v_soft = response.target_volume_cm3(e, GAMMA_IC, part="soft")[0]
    assert soft / inside == pytest.approx(v_soft / v_inside, rel=1e-12)


def test_invalid_part_raises(response):
    with pytest.raises(ValueError):
        response.target_volume_cm3(E_1PEV, GAMMA_IC, part="bogus")


def test_sm_through_going_at_100pev_below_one_event(response):
    # The KM3NeT tension: an SM diffuse flux yields far fewer than one
    # through-going event at 100 PeV over IceCube's 9.5 yr exposure.
    livetime_s = 9.5 * 365.25 * 86400.0
    edges = np.array([8.0, 9.0])  # 100 PeV - 1 EeV
    counts = response.expected_counts(
        edges, PHI0_IC, GAMMA_IC, livetime_s, solid_angle_sr=2 * np.pi, part="soft"
    )
    assert 1e-3 < counts[0] < 1.0


def test_expected_counts_scale_linearly_with_exposure(response):
    edges = np.array([5.0, 6.0])
    base = response.expected_counts(edges, PHI0_IC, GAMMA_IC, 1.0, 1.0)
    scaled = response.expected_counts(edges, PHI0_IC, GAMMA_IC, 10.0, 2.0)
    np.testing.assert_allclose(scaled, 20.0 * base, rtol=1e-12)


def test_expected_counts_scale_linearly_with_normalization(response):
    edges = np.array([5.0, 6.0])
    base = response.expected_counts(edges, 1.0, GAMMA_IC, 1.0, 1.0)
    scaled = response.expected_counts(edges, 3.0, GAMMA_IC, 1.0, 1.0)
    np.testing.assert_allclose(scaled, 3.0 * base, rtol=1e-12)


# ---------------------------------------------------------------------------
# Tabulated-flux path: expected_counts_from_flux
# ---------------------------------------------------------------------------


def test_local_spectral_index_recovers_a_power_law():
    energy = np.logspace(3.0, 8.0, 40)
    gamma_eff = _local_spectral_index(energy, energy**-2.7)
    np.testing.assert_allclose(gamma_eff, 2.7, rtol=1e-10)


def test_local_spectral_index_clipped_at_the_bounds():
    energy = np.logspace(3.0, 8.0, 40)
    # An exponential cutoff drives the local slope arbitrarily steep.
    gamma_eff = _local_spectral_index(energy, np.exp(-energy / 1.0e4))
    assert gamma_eff.max() == pytest.approx(GAMMA_EFF_BOUNDS[1])
    assert gamma_eff.min() >= GAMMA_EFF_BOUNDS[0]


def test_counts_from_power_law_flux_match_the_power_law_path(response):
    # Fed the same power law it is parametrized by, the tabulated-flux path must
    # reproduce expected_counts over the band's geometric solid angle.
    edges = np.array([5.0, 5.5, 6.0])
    dec_min, dec_max = 0.0, 90.0
    solid_angle_sr = 2.0 * np.pi * (np.sin(np.deg2rad(dec_max)) - np.sin(np.deg2rad(dec_min)))

    reference = response.expected_counts(
        edges, PHI0_IC, GAMMA_IC, 1.0e7, solid_angle_sr, n_subdivisions=64,
    )
    from_flux = response.expected_counts_from_flux(
        edges,
        lambda energy, _dec: power_law_flux(energy, PHI0_IC, GAMMA_IC),
        1.0e7,
        dec_min,
        dec_max,
        attenuate=False,
        n_subdivisions=64,
        n_dec=256,
    )
    np.testing.assert_allclose(from_flux, reference, rtol=1e-4)


def test_counts_from_flux_scale_linearly_with_the_flux(response):
    edges = np.array([5.0, 6.0])
    kwargs = dict(livetime_s=1.0e7, dec_min_deg=0.0, dec_max_deg=90.0, attenuate=False)
    base = response.expected_counts_from_flux(
        edges, lambda energy, _dec: power_law_flux(energy, 1.0, GAMMA_IC), **kwargs,
    )
    scaled = response.expected_counts_from_flux(
        edges, lambda energy, _dec: 4.0 * power_law_flux(energy, 1.0, GAMMA_IC), **kwargs,
    )
    np.testing.assert_allclose(scaled, 4.0 * base, rtol=1e-12)


def test_attenuation_suppresses_counts_from_flux(response):
    # The upgoing band is opaque at PeV energies, so switching attenuation on
    # must remove most of the rate.
    edges = np.array([6.0, 7.0])
    kwargs = dict(livetime_s=1.0e7, dec_min_deg=0.0, dec_max_deg=90.0)
    flux = lambda energy, _dec: power_law_flux(energy, PHI0_IC, GAMMA_IC)  # noqa: E731

    bare = response.expected_counts_from_flux(edges, flux, attenuate=False, **kwargs)
    attenuated = response.expected_counts_from_flux(edges, flux, attenuate=True, **kwargs)
    assert 0.0 < attenuated[0] < 0.5 * bare[0]


def test_counts_from_flux_see_the_declination_dependence(response):
    # A flux switched off over half the band gives fewer counts than one that is
    # on everywhere, and more than nothing.
    edges = np.array([5.0, 6.0])
    kwargs = dict(livetime_s=1.0e7, dec_min_deg=0.0, dec_max_deg=90.0, attenuate=False)

    def half_sky(energy, dec):
        return np.where(dec < 45.0, power_law_flux(energy, PHI0_IC, GAMMA_IC), 0.0)

    full = response.expected_counts_from_flux(
        edges, lambda energy, _dec: power_law_flux(energy, PHI0_IC, GAMMA_IC), **kwargs,
    )
    half = response.expected_counts_from_flux(edges, half_sky, **kwargs)
    assert 0.0 < half[0] < full[0]


def test_counts_from_flux_rejects_double_attenuation():
    response = SoftVolumeResponse(radius_km=0.62, attenuation_column_g_cm2=1.0e10)
    with pytest.raises(ValueError):
        response.expected_counts_from_flux(
            np.array([5.0, 6.0]),
            lambda energy, _dec: power_law_flux(energy, PHI0_IC, GAMMA_IC),
            1.0e7,
            0.0,
            90.0,
        )


# ---------------------------------------------------------------------------
# Exact-method SoftVolumeResponse
# ---------------------------------------------------------------------------


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        SoftVolumeResponse(radius_km=0.62, method="bogus")


def test_exact_response_still_factorizes():
    resp = SoftVolumeResponse(radius_km=0.62, method="exact", column_depth_km=1.95)
    e = E_1PEV
    rate = resp.differential_rate(e, PHI0_IC, GAMMA_IC, part="total")
    vol = resp.target_volume_cm3(e, GAMMA_IC, part="total")
    weak = resp.weak_rate_density(e, PHI0_IC, GAMMA_IC)
    np.testing.assert_allclose(rate, vol * weak, rtol=1e-12)


def test_exact_total_is_inside_plus_soft():
    resp = SoftVolumeResponse(radius_km=0.62, method="exact", column_depth_km=1.95)
    e = np.array([E_1PEV, E_100PEV])
    total = resp.differential_rate(e, PHI0_IC, GAMMA_IC, part="total")
    inside = resp.differential_rate(e, PHI0_IC, GAMMA_IC, part="inside")
    soft = resp.differential_rate(e, PHI0_IC, GAMMA_IC, part="soft")
    np.testing.assert_allclose(total, inside + soft, rtol=1e-12)


def test_exact_inside_carries_inelasticity():
    # In the exact master formula I(A) multiplies both populations, so the
    # effective inside volume is I(A) V_det < V_det.
    resp = SoftVolumeResponse(radius_km=0.62, method="exact", column_depth_km=1.95)
    v_inside = resp.target_volume_cm3(E_1PEV, GAMMA_IC, part="inside")[0]
    assert v_inside < resp.v_det_cm3


def test_finite_column_gives_fewer_counts_than_infinite():
    edges = np.array([5.0, 6.0])
    finite = SoftVolumeResponse(radius_km=0.62, method="exact", column_depth_km=1.95)
    infinite = SoftVolumeResponse(radius_km=0.62, method="exact", column_depth_km=None)
    c_finite = finite.expected_counts(edges, PHI0_IC, GAMMA_IC, 1.0, 1.0, part="soft")
    c_infinite = infinite.expected_counts(edges, PHI0_IC, GAMMA_IC, 1.0, 1.0, part="soft")
    assert c_finite[0] < c_infinite[0]


# ---------------------------------------------------------------------------
# Implied effective area
# ---------------------------------------------------------------------------


def test_effective_area_matches_volume_times_cross_section(response):
    e = np.array([E_1PEV, E_100PEV])
    aeff = response.effective_area_cm2(e, GAMMA_IC)
    vol = response.target_volume_cm3(e, GAMMA_IC)
    sigma = cc_cross_section(e)
    np.testing.assert_allclose(aeff, vol * response.n_nucleon_cm3 * sigma, rtol=1e-12)


def test_effective_area_positive(response):
    aeff = response.effective_area_cm2(np.array([E_1PEV, E_100PEV]), GAMMA_IC)
    assert np.all(aeff > 0.0)


def test_effective_area_part_matches_target_volume_part(response):
    e = E_1PEV
    aeff_soft = response.effective_area_cm2(e, GAMMA_IC, part="soft")
    aeff_inside = response.effective_area_cm2(e, GAMMA_IC, part="inside")
    aeff_total = response.effective_area_cm2(e, GAMMA_IC, part="total")
    assert aeff_total[0] == pytest.approx(aeff_soft[0] + aeff_inside[0], rel=1e-12)
