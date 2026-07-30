"""Tests for softpaws.transport.cross_section and its table loader.

Covers the analytic power law, the tabulated BGR18 model shipped with the
package, and the energy-dependent ``lambda`` that a tabulated model feeds into
the soft volume.
"""

import numpy as np
import pytest

from softpaws.data.loader import load_cross_section_table
from softpaws.response.soft_volume import SoftVolumeResponse
from softpaws.transport.attenuation import survival_probability
from softpaws.transport.cross_section import (
    DEFAULT_TOTAL_TO_CC,
    PowerLawCrossSection,
    TabulatedCrossSection,
    bgr18_cross_section,
)
from softpaws.transport.source import DEFAULT_LAMBDA, E0_CROSS_GEV, SIGMA0_CM2, cc_cross_section

# ---------------------------------------------------------------------------
# Table loading
# ---------------------------------------------------------------------------


def test_load_cross_section_table_returns_sorted_columns():
    energy, sigma = load_cross_section_table("BGR18", "cc")
    assert energy.shape == sigma.shape
    assert np.all(np.diff(energy) > 0)
    assert np.all(sigma > 0)


def test_load_cross_section_table_rejects_unknown_channel():
    with pytest.raises(ValueError):
        load_cross_section_table("BGR18", "bogus")


def test_load_cross_section_table_reports_a_missing_model():
    with pytest.raises(FileNotFoundError):
        load_cross_section_table("NoSuchModel", "cc")


# ---------------------------------------------------------------------------
# PowerLawCrossSection: must reproduce the analytic function it replaces
# ---------------------------------------------------------------------------


def test_power_law_matches_the_analytic_cross_section():
    model = PowerLawCrossSection()
    energy = np.logspace(3.0, 8.0, 20)
    np.testing.assert_allclose(model.cc(energy), cc_cross_section(energy), rtol=1e-12)


def test_power_law_anchored_at_its_pivot():
    model = PowerLawCrossSection()
    assert model.cc(E0_CROSS_GEV)[0] == pytest.approx(SIGMA0_CM2, rel=1e-12)


def test_power_law_slope_is_constant():
    model = PowerLawCrossSection()
    slope = model.local_slope(np.logspace(2.0, 9.0, 15))
    np.testing.assert_allclose(slope, DEFAULT_LAMBDA, rtol=1e-12)


def test_power_law_total_is_the_scaled_cc():
    model = PowerLawCrossSection()
    energy = 1.0e6
    assert model.total(energy)[0] == pytest.approx(
        DEFAULT_TOTAL_TO_CC * model.cc(energy)[0], rel=1e-12
    )


# ---------------------------------------------------------------------------
# TabulatedCrossSection (BGR18)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bgr18() -> TabulatedCrossSection:
    return bgr18_cross_section()


def test_tabulated_reproduces_its_own_table(bgr18):
    # The smoothing spline trades a little accuracy on the value for a slope
    # that is not dominated by digitization noise; 2% is the tolerance that buys.
    energy, sigma = load_cross_section_table("BGR18", "cc")
    np.testing.assert_allclose(bgr18.cc(energy), sigma, rtol=0.02)


def test_tabulated_is_monotonically_rising(bgr18):
    sigma = bgr18.cc(np.logspace(3.0, 9.0, 40))
    assert np.all(np.diff(sigma) > 0)


def test_tabulated_slope_is_positive_and_falling(bgr18):
    # The cross section goes from roughly linear in the valence regime to a
    # slowly rising small-x one, so the slope decreases with energy.
    energy = np.logspace(3.0, 9.0, 25)
    slope = bgr18.local_slope(energy)
    assert np.all(slope > 0.0)
    assert slope[0] > slope[-1]
    assert slope[0] == pytest.approx(1.0, abs=0.15)
    assert slope[-1] == pytest.approx(0.35, abs=0.1)


def test_tabulated_slope_varies_smoothly_with_energy(bgr18):
    # lambda feeds A = gamma - lambda - 1, so it has to be a smooth function of
    # energy rather than the piecewise-constant staircase that differentiating a
    # linear interpolation would give. Sampled finely, it steps by <0.02 and its
    # residual up-and-down wiggle is small against the overall fall.
    slope = bgr18.local_slope(np.logspace(3.0, 9.0, 500))
    step = np.diff(slope)
    assert np.abs(step).max() < 0.02
    assert step[step > 0].sum() < 0.3 * -step[step < 0].sum()


def test_tabulated_nc_is_a_sensible_fraction_of_cc(bgr18):
    energy = np.logspace(3.0, 9.0, 20)
    ratio = bgr18.nc(energy) / bgr18.cc(energy)
    assert np.all((ratio > 0.35) & (ratio < 0.5))


def test_tabulated_total_is_cc_plus_nc(bgr18):
    energy = np.logspace(3.0, 9.0, 10)
    np.testing.assert_allclose(bgr18.total(energy), bgr18.cc(energy) + bgr18.nc(energy), rtol=1e-12)


def test_power_law_overshoots_the_table_at_low_energy(bgr18):
    # The motivating discrepancy: the power law is anchored at 10 PeV and
    # overshoots badly when extrapolated down, while agreeing near 1 PeV.
    power_law = PowerLawCrossSection()
    assert power_law.cc(1.0e3)[0] / bgr18.cc(1.0e3)[0] > 5.0
    assert power_law.cc(1.0e6)[0] / bgr18.cc(1.0e6)[0] == pytest.approx(1.0, abs=0.15)


def test_tabulated_extrapolates_as_a_power_law(bgr18):
    # Outside the table the curve continues with the boundary slope, so a decade
    # below the lower edge the ratio is set by that slope, not by a cubic runaway.
    lo = bgr18.energy_range_gev[0]
    slope = bgr18.local_slope(lo)[0]
    ratio = bgr18.cc(lo / 10.0)[0] / bgr18.cc(lo)[0]
    assert ratio == pytest.approx(10.0**-slope, rel=1e-6)


# ---------------------------------------------------------------------------
# Use inside the forward model
# ---------------------------------------------------------------------------


def test_response_defaults_to_the_power_law():
    response = SoftVolumeResponse(radius_km=0.62)
    energy = np.logspace(4.0, 7.0, 10)
    np.testing.assert_allclose(
        response.cc_cross_section_cm2(energy), cc_cross_section(energy), rtol=1e-12
    )
    assert response.spectral_slope(energy) == DEFAULT_LAMBDA


def test_response_uses_a_supplied_cross_section(bgr18):
    response = SoftVolumeResponse(radius_km=0.62, cross_section=bgr18)
    energy = np.logspace(4.0, 7.0, 10)
    np.testing.assert_allclose(response.cc_cross_section_cm2(energy), bgr18.cc(energy), rtol=1e-12)
    np.testing.assert_allclose(response.spectral_slope(energy), bgr18.local_slope(energy))


@pytest.mark.parametrize("method", ["drift", "diffusion", "exact"])
def test_target_volume_handles_an_energy_dependent_lambda(bgr18, method):
    # A tabulated cross section makes A = gamma - lambda(E) - 1 an array, which
    # every transport method has to broadcast through.
    response = SoftVolumeResponse(radius_km=0.62, method=method, cross_section=bgr18)
    energy = np.logspace(5.0, 7.0, 12)
    volume = response.target_volume_cm3(energy, gamma=2.38)
    assert volume.shape == energy.shape
    assert np.all(volume > 0)


def test_tabulated_lowers_the_effective_area_below_a_pev(bgr18):
    # The power law's overshoot at 10 TeV carries straight into A_eff, which is
    # what distorted the implied selection efficiency of example 20.
    power_law = SoftVolumeResponse(radius_km=0.62)
    tabulated = SoftVolumeResponse(radius_km=0.62, cross_section=bgr18)
    ratio = (
        power_law.threshold_effective_area_cm2(1.0e4)[0]
        / tabulated.threshold_effective_area_cm2(1.0e4)[0]
    )
    assert ratio > 2.0


def test_tabulated_cross_section_deepens_earth_attenuation(bgr18):
    # sigma_tot from the table exceeds 1.4 x the power law above ~1 PeV, and the
    # survival probability is exponential in it, so a 20% difference in the cross
    # section becomes an order of magnitude in D_nu on a full-diameter column.
    column_g_cm2 = 1.0e10
    power_law = survival_probability(1.0e7, column_g_cm2)[0]
    tabulated = survival_probability(1.0e7, column_g_cm2, cross_section=bgr18)[0]
    assert tabulated < 0.2 * power_law
