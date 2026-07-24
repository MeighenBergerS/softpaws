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
    SoftVolumeResponse,
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
