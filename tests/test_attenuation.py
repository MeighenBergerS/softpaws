"""Tests for the Earth neutrino-attenuation model.

Cover the chord geometry (horizon vs. nadir), the PREM density profile, the
total cross section, the survival-probability limits, and the consistency of the
closed-form representative column and the per-event effective solid angle.
"""

import numpy as np
import pytest

from softpaws.constants import (
    AVOGADRO_PER_MOL,
    CM_PER_KM,
    EARTH_RADIUS_KM,
    RHO_EARTH_MEAN_G_CM3,
    RHO_WATER_G_CM3,
)
from softpaws.transport.attenuation import (
    TOTAL_TO_CC_RATIO,
    earth_chord_length_km,
    effective_solid_angle,
    mean_density_column,
    neutrino_interaction_length_km,
    prem_column,
    prem_density,
    representative_column,
    survival_probability,
    total_cross_section,
)
from softpaws.transport.source import cc_cross_section

E_1PEV = 1.0e6  # GeV


# ---------------------------------------------------------------------------
# Chord geometry
# ---------------------------------------------------------------------------


def test_chord_vanishes_at_horizon():
    assert earth_chord_length_km(0.0)[0] == pytest.approx(0.0, abs=1e-9)


def test_chord_is_diameter_at_nadir():
    assert earth_chord_length_km(90.0)[0] == pytest.approx(2.0 * EARTH_RADIUS_KM)


def test_chord_zero_for_downgoing():
    assert earth_chord_length_km(-30.0)[0] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# PREM density
# ---------------------------------------------------------------------------


def test_prem_inner_core_central_density():
    # PREM central density is 13.0885 g/cm^3.
    assert prem_density(0.0)[0] == pytest.approx(13.0885, abs=1e-4)


def test_prem_density_monotonic_shell_ordering():
    # Core is denser than mantle, mantle denser than crust.
    core = prem_density(1000.0)[0]
    mantle = prem_density(5000.0)[0]
    crust = prem_density(6360.0)[0]
    assert core > mantle > crust


def test_prem_density_zero_outside_earth():
    assert prem_density(EARTH_RADIUS_KM + 100.0)[0] == pytest.approx(0.0)


def test_prem_diameter_column_matches_known_value():
    # The straight-through (nadir) PREM column is ~1.1e10 g/cm^2, well above the
    # mean-density estimate because the dense core dominates the path.
    column = prem_column(90.0)
    assert column == pytest.approx(1.1e10, rel=0.1)
    assert column > mean_density_column(90.0)[0]


# ---------------------------------------------------------------------------
# Cross section and survival
# ---------------------------------------------------------------------------


def test_total_cross_section_scales_cc():
    assert total_cross_section(E_1PEV)[0] == pytest.approx(
        TOTAL_TO_CC_RATIO * cc_cross_section(E_1PEV)[0]
    )


def test_survival_is_unity_for_zero_column():
    assert survival_probability(E_1PEV, 0.0)[0] == pytest.approx(1.0)


def test_survival_in_unit_interval_and_decreasing():
    energies = np.array([1e5, 1e6, 1e7, 1e8])
    d_nu = survival_probability(energies, mean_density_column(45.0)[0])
    assert np.all((d_nu >= 0.0) & (d_nu <= 1.0))
    # Higher energy -> larger cross section -> more absorption.
    assert np.all(np.diff(d_nu) < 0.0)


# ---------------------------------------------------------------------------
# Representative column and effective solid angle
# ---------------------------------------------------------------------------


def test_representative_column_full_hemisphere():
    # Solid-angle average over dec in [0, 90] reduces to rho_mean * R_Earth.
    expected = RHO_EARTH_MEAN_G_CM3 * EARTH_RADIUS_KM * 1.0e5  # km -> cm
    assert representative_column(0.0, 90.0) == pytest.approx(expected, rel=1e-12)


def test_representative_column_requires_ordered_band():
    with pytest.raises(ValueError):
        representative_column(90.0, 0.0)


def test_effective_solid_angle_recovers_geometric_at_low_energy():
    # With negligible cross section the effective solid angle -> geometric 2 pi.
    # At 1 GeV the extrapolated power-law cross section is tiny but nonzero, so
    # allow a percent-level residual absorption.
    omega = effective_solid_angle(np.array([1.0]), 0.0, 90.0)
    assert omega[0] == pytest.approx(2.0 * np.pi, rel=1e-2)


def test_effective_solid_angle_below_geometric_at_high_energy():
    geometric = 2.0 * np.pi
    omega = effective_solid_angle(np.array([1e7]), 0.0, 90.0)
    assert 0.0 < omega[0] < geometric


# ---------------------------------------------------------------------------
# Neutrino interaction length (Eq. 11 coupling)
# ---------------------------------------------------------------------------


def test_interaction_length_matches_total_cross_section():
    # Lambda_nu = 1 / (N_A sigma_tot rho), in km at the reference density.
    sigma = total_cross_section(E_1PEV)[0]
    expected_km = 1.0 / (AVOGADRO_PER_MOL * sigma * RHO_WATER_G_CM3 * CM_PER_KM)
    assert neutrino_interaction_length_km(E_1PEV)[0] == pytest.approx(expected_km, rel=1e-12)


def test_interaction_length_is_thousands_of_km_we_at_pev():
    # Palmisano et al. (arXiv:2607.13143) and the method paper quote
    # Lambda_nu ~ O(10^3) km.w.e.
    assert 1.0e2 < neutrino_interaction_length_km(E_1PEV)[0] < 1.0e5


def test_interaction_length_decreases_with_energy():
    # Larger cross section at higher energy -> shorter interaction length.
    lengths = neutrino_interaction_length_km(np.array([1e5, 1e6, 1e7, 1e8]))
    assert np.all(np.diff(lengths) < 0.0)


def test_interaction_length_scales_inversely_with_density():
    water = neutrino_interaction_length_km(E_1PEV, RHO_WATER_G_CM3)[0]
    dense = neutrino_interaction_length_km(E_1PEV, 2.0 * RHO_WATER_G_CM3)[0]
    assert dense == pytest.approx(water / 2.0, rel=1e-12)
