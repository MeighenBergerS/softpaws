"""Tests for the Earth-geometry helpers."""

import numpy as np
import pytest

from softpaws.transport import attenuation, earth
from softpaws.utils.constants import CM_PER_KM, EARTH_RADIUS_KM


def test_reexports_survive():
    """The PREM functions moved; the old import path still works."""
    assert attenuation.prem_column is earth.prem_column
    assert attenuation.earth_chord_length_km is earth.earth_chord_length_km


def test_zenith_grid():
    theta, w = earth.zenith_grid(90)
    assert theta.shape == (90,) and w.sum() == pytest.approx(1.0)
    assert theta[0] < 90.0 < theta[-1]
    cos = np.cos(np.deg2rad(theta))
    np.testing.assert_allclose(np.diff(cos), -2.0 / 90, atol=1e-12)
    theta_up, _ = earth.zenith_grid(10, cos_range=(-1.0, 0.0))
    assert np.all(theta_up > 90.0)
    theta_same, _ = earth.zenith_grid(10, cos_range=(0.0, -1.0))
    np.testing.assert_allclose(theta_same, theta_up)


def test_overburden():
    got = earth.overburden_km(np.array([1.0, 0.5, 0.0, -0.5]), 2.0)
    np.testing.assert_allclose(got, [2.0, 4.0, earth.MAX_UPSTREAM_KM, earth.MAX_UPSTREAM_KM])
    assert earth.overburden_km(1e-9, 2.0, cap_km=50.0) == 50.0


def test_neutrino_column_matches_the_old_formula():
    """The column is what examples 30 to 35 computed inline."""
    theta_deg, _ = earth.zenith_grid(30)
    cos_theta = np.cos(np.deg2rad(theta_deg))
    depth, rho = 3.184, 1.02
    got = earth.neutrino_column_g_cm2(cos_theta, depth, rho)
    with np.errstate(divide="ignore"):
        down = np.where(cos_theta > 0.0, depth / np.maximum(cos_theta, 1e-6), np.inf)
    water = np.where(np.isfinite(down), np.minimum(down, 100.0), 0.0) * CM_PER_KM * rho
    prem = np.array([earth.prem_column(t - 90.0) if t > 90.0 else 0.0 for t in theta_deg])
    np.testing.assert_allclose(got, np.where(theta_deg > 90.0, prem, water))
    assert float(earth.neutrino_column_g_cm2(0.0, depth, rho)) == 0.0


def test_prem_diameter_column():
    """Straight through the centre the PREM column is ~1.1e10 g cm^-2."""
    column = earth.prem_column(90.0)
    assert column == pytest.approx(1.10e10, rel=0.02)
    assert earth.prem_column(0.0) == 0.0
    assert float(earth.earth_chord_length_km(90.0)[0]) == pytest.approx(2 * EARTH_RADIUS_KM)
