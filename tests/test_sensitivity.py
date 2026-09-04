"""Tests for the sensitivity converters and the one-number reach fit.

The three converters are arithmetic on an effective area, so the checks here
are the identities that arithmetic has to satisfy: linearity in the exposure,
the fixed ratio between a line and a single event, and agreement with the
point-source ceiling the declination module used to carry on its own.
"""

import numpy as np
import pytest

from softpaws.detectors import ARCA230, GVD, ICECUBE, PONE, TRIDENT
from softpaws.response import declination as dec
from softpaws.response import sensitivity as sens
from softpaws.response import site_models as sm

LOG10_E = np.linspace(4.0, 8.0, 17)
ENERGY = 10.0**LOG10_E

#: A rising response, close enough to a real one for the identities to bite.
AEFF = 1.0e4 * (ENERGY / 1.0e5) ** 0.5


def test_single_event_scales_with_exposure_and_solid_angle():
    """One event per decade is linear in the events and inverse in the exposure."""
    one = sens.single_event_sensitivity(AEFF, 1.0e8, LOG10_E)
    np.testing.assert_allclose(sens.single_event_sensitivity(AEFF, 2.0e8, LOG10_E), 0.5 * one)
    np.testing.assert_allclose(
        sens.single_event_sensitivity(AEFF, 1.0e8, LOG10_E, n_events=2.0), 2.0 * one
    )
    diffuse = sens.single_event_sensitivity(AEFF, 1.0e8, LOG10_E, solid_angle_sr=4.0 * np.pi)
    np.testing.assert_allclose(diffuse, one / (4.0 * np.pi))


def test_single_event_delivers_one_event_per_decade():
    """The count a decade of the returned flux delivers is the one asked for.

    A decade is ``ln(10)`` wide in ``ln(E)``, and the count over it is
    ``T A_eff E phi ln(10)`` with the integrand held at the node, which is the
    convention the differential sensitivity is quoted in.
    """
    livetime_s, n_events = 1.0e8, 1.0
    e2_flux = sens.single_event_sensitivity(AEFF, livetime_s, LOG10_E, n_events=n_events)
    flux = e2_flux / ENERGY**2
    count = livetime_s * AEFF * flux * ENERGY * np.log(10.0)
    np.testing.assert_allclose(count, n_events, rtol=1e-12)


def test_line_and_single_event_differ_by_a_constant():
    """A line limit is the events assumed times the width of a decade."""
    line = sens.line_sensitivity(AEFF, 1.0e8)
    single = sens.single_event_sensitivity(AEFF, 1.0e8, LOG10_E)
    np.testing.assert_allclose(
        ENERGY * line / single, sens.N_EVENTS_LIMIT * np.log(10.0), rtol=1e-12
    )


def test_power_law_matches_the_declination_ceiling():
    """The point-source ceiling is the same quadrature, and now the same code."""
    aeff = 1.0e4 * (10.0**dec.COMMON_LOG10_E / sens.PIVOT_ENERGY_GEV) ** 0.5
    np.testing.assert_allclose(
        dec.point_source_sensitivity(aeff, 1.0e8, 2.0),
        sens.power_law_sensitivity(aeff, 1.0e8, 2.0, dec.COMMON_LOG10_E,
                                   emin_gev=dec.DEFAULT_EMIN_GEV),
    )


def test_power_law_window_and_columns():
    """Throwing away the bottom of the band can only weaken the limit."""
    whole = sens.power_law_sensitivity(AEFF, 1.0e8, 2.0, LOG10_E)
    raised = sens.power_law_sensitivity(AEFF, 1.0e8, 2.0, LOG10_E, emin_gev=1.0e6)
    assert raised > whole
    stacked = sens.power_law_sensitivity(
        np.stack([AEFF, 2.0 * AEFF], axis=1), 1.0e8, 2.0, LOG10_E
    )
    np.testing.assert_allclose(stacked, [whole, 0.5 * whole])
    diffuse = sens.power_law_sensitivity(AEFF, 1.0e8, 2.0, LOG10_E, solid_angle_sr=2.0)
    np.testing.assert_allclose(diffuse, 0.5 * whole)


def test_published_effective_area_carries_its_sky():
    """Each shipped curve comes back on the grid asked for, with its convention."""
    log10_e = np.arange(5.0, 6.51, 0.5)
    for site in (ARCA230, PONE, TRIDENT):
        aeff, cos_range = sm.published_effective_area_cm2(site, log10_e)
        assert cos_range == (-1.0, 1.0)
        assert np.all(np.isfinite(aeff)) and np.all(aeff > 0.0)
        assert np.all(np.diff(aeff) > 0.0)
    assert sm.PUBLISHED_SKY["IceCube"] == (-1.0, 0.0)
    with pytest.raises(KeyError):
        sm.published_effective_area_cm2(GVD, log10_e)


def test_fit_light_reach_recovers_its_own_curve():
    """A curve the model made at one scanned point is fitted back to it."""
    log10_e = np.array([6.0])
    cos_theta, weights = np.array([-0.6, -0.2]), np.array([0.5, 0.5])
    fractions = np.array([0.8, 1.0])
    truth_km = ICECUBE.radius_km * (fractions[0] - 1.0) / np.log(
        dec.REACH_REFERENCE_GEV / dec.REACH_PIVOT_GEV
    )
    published = dec.directional_effective_area_cm2(
        ICECUBE, cos_theta, 1.0e3, reach_km=truth_km, log10_e=log10_e
    ) @ weights
    reach_km, residual, models = dec.fit_light_reach(
        ICECUBE, cos_theta, weights, published, 1.0e3, log10_e=log10_e, fractions=fractions
    )
    assert reach_km == pytest.approx(truth_km)
    assert residual[0] == pytest.approx(0.0, abs=1e-12)
    assert residual[1] > residual[0]
    np.testing.assert_allclose(models[0], published)


def test_fit_light_reach_needs_a_positive_node():
    """A curve with nothing to fit against is an error, not a silent answer."""
    with pytest.raises(ValueError):
        dec.fit_light_reach(
            ICECUBE, np.array([-0.5]), np.array([1.0]), np.array([np.nan]), 1.0e3,
            log10_e=np.array([6.0]), fractions=np.array([1.0]),
        )


# ---------------------------------------------------------------------------
# The atmospheric background and the limit it buys
# ---------------------------------------------------------------------------


#: One flat direction and one flat flux, so every identity below is arithmetic
#: the test can predict rather than a number the model has to produce.
class _FlatFlux:
    """Atmospheric flux of a fixed value, whatever the energy and direction."""

    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self, energy_gev, dec_deg):
        return np.full(np.broadcast(energy_gev, dec_deg).shape, self.value)


def test_table_declination_folds_both_verticals_together():
    """Overhead and nadir cross the same atmosphere, and the horizon the most."""
    from softpaws.fluxes import table_declination_deg

    np.testing.assert_allclose(
        table_declination_deg(np.array([-1.0, -0.5, 0.0, 0.5, 1.0])),
        [90.0, 30.0, 0.0, 30.0, 90.0],
    )


def test_point_source_bin_grows_with_its_radius():
    """The bin is a spherical cap, and a small one is the flat-disc limit."""
    assert sens.point_source_bin_sr(2.0) / sens.point_source_bin_sr(1.0) == pytest.approx(
        4.0, rel=1e-3
    )
    small = np.deg2rad(0.1)
    assert sens.point_source_bin_sr(0.1) == pytest.approx(np.pi * small**2, rel=1e-5)


def test_background_is_linear_in_exposure_and_bin():
    """Counting events in a bin is one product, so it scales like one."""
    flux = _FlatFlux(1.0e-10)
    aeff = np.stack([AEFF, AEFF], axis=1)
    cos_theta, weights = np.array([-0.8, -0.2]), np.array([0.5, 0.5])
    args = (flux, aeff, cos_theta, weights)
    one = sens.atmospheric_background_counts(*args, 1.0e8, LOG10_E, 1.0)
    assert one > 0.0
    np.testing.assert_allclose(
        sens.atmospheric_background_counts(*args, 2.0e8, LOG10_E, 1.0), 2.0 * one
    )
    np.testing.assert_allclose(
        sens.atmospheric_background_counts(*args, 1.0e8, LOG10_E, 0.5),
        one * sens.point_source_bin_sr(0.5) / sens.point_source_bin_sr(1.0),
    )
    # A window that throws away the bottom of the band can only lose events.
    assert sens.atmospheric_background_counts(
        *args, 1.0e8, LOG10_E, 1.0, emin_gev=1.0e6
    ) < one


def test_background_rides_the_sweep_of_each_declination():
    """One row of weights per declination gives one column of counts."""
    flux = _FlatFlux(1.0e-10)
    aeff = np.stack([AEFF, 2.0 * AEFF], axis=1)
    cos_theta = np.array([-0.8, -0.2])
    weights = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    counts = sens.atmospheric_background_counts(flux, aeff, cos_theta, weights, 1.0e8, LOG10_E)
    assert counts.shape == (3,)
    np.testing.assert_allclose(counts[1], 2.0 * counts[0])
    np.testing.assert_allclose(counts[2], 0.5 * (counts[0] + counts[1]))


def test_feldman_cousins_reproduces_the_published_table():
    """The unified construction, against the rows of Feldman and Cousins Table IV."""
    for background, expected in (
        (0.0, [2.44, 4.36, 5.91, 7.42, 8.60]),
        (1.0, [1.61, 3.36, 4.91, 6.42, 7.60]),
    ):
        limits = [sens.feldman_cousins_upper_limit(n, background) for n in range(5)]
        np.testing.assert_allclose(limits, expected, atol=0.01)


def test_sensitivity_reduces_to_the_background_free_limit():
    """No background to average over leaves the constant the module quotes."""
    assert float(sens.sensitivity_upper_limit(0.0)) == pytest.approx(
        sens.N_EVENTS_LIMIT, abs=0.01
    )


def test_sensitivity_grows_with_the_background():
    """More background is a weaker search, and eventually a square-root one."""
    background = np.array([0.0, 1.0, 4.0, 25.0, 100.0])
    n_events = sens.sensitivity_upper_limit(background)
    assert n_events.shape == background.shape
    assert np.all(np.diff(n_events) > 0.0)
    # Well above zero the average upper limit is the Gaussian one, a fixed
    # number of standard deviations of the background.
    ratio = n_events[-1] / n_events[-2]
    assert ratio == pytest.approx(np.sqrt(4.0), rel=0.15)


def test_background_weakens_every_converter():
    """The three converters take the excludable events, so one number moves all."""
    n_events = float(sens.sensitivity_upper_limit(9.0))
    assert n_events > sens.N_EVENTS_LIMIT
    free = sens.power_law_sensitivity(AEFF, 1.0e8, 2.0, LOG10_E)
    limited = sens.power_law_sensitivity(AEFF, 1.0e8, 2.0, LOG10_E, n_events=n_events)
    np.testing.assert_allclose(limited / free, n_events / sens.N_EVENTS_LIMIT)


def test_per_energy_events_broadcast_against_a_column_of_directions():
    """A background that varies across the band is an array on the energy axis."""
    events = np.linspace(2.44, 12.0, LOG10_E.size)
    stacked = np.stack([AEFF, 2.0 * AEFF], axis=1)
    flat = sens.single_event_sensitivity(stacked, 1.0e8, LOG10_E)
    scaled = sens.single_event_sensitivity(stacked, 1.0e8, LOG10_E, n_events=events)
    assert scaled.shape == stacked.shape
    np.testing.assert_allclose(scaled, flat * events[:, None])
    line = sens.line_sensitivity(stacked, 1.0e8, n_events=events)
    np.testing.assert_allclose(
        line, sens.line_sensitivity(stacked, 1.0e8) * events[:, None] / sens.N_EVENTS_LIMIT
    )


def test_psf_containment_is_a_weighted_quantile():
    """The containment is read off the released PSF column, not a bin edge."""
    from softpaws.response.irfs import SmearingMatrix

    # One (E_nu, dec) group of four PSF bins carrying equal weight, so the 68%
    # point falls between the second and the third and has to be interpolated.
    angles = np.array([0.5, 1.0, 2.0, 4.0])
    raw = np.zeros((4, 11))
    raw[:, 0], raw[:, 1] = 3.0, 4.0
    raw[:, 2], raw[:, 3] = -5.0, 5.0
    raw[:, 6], raw[:, 7] = angles, angles
    raw[:, 10] = 0.25
    containment = SmearingMatrix(raw).psf_containment_deg(0.68)
    assert containment.shape == (1, 1)
    assert containment[0, 0] == pytest.approx(1.0 + (0.68 - 0.5) / 0.25)
    # The median sits inside the second bin by the same construction.
    assert SmearingMatrix(raw).psf_containment_deg(0.5)[0, 0] == pytest.approx(1.0)


def test_bin_radius_may_vary_with_energy_and_direction():
    """A measured point spread is a grid, and the background has to take one."""
    flux = _FlatFlux(1.0e-10)
    aeff = np.stack([AEFF, AEFF], axis=1)
    cos_theta, weights = np.array([-0.8, -0.2]), np.array([[1.0, 0.0], [0.0, 1.0]])
    args = (flux, aeff, cos_theta, weights, 1.0e8, LOG10_E)
    flat = sens.atmospheric_background_density(*args, 1.0)

    # A constant grid is the constant, whichever shape it arrives in.
    np.testing.assert_allclose(
        sens.atmospheric_background_density(*args, np.full(LOG10_E.size, 1.0)), flat
    )
    grid = np.full((LOG10_E.size, 2), 1.0)
    grid[:, 1] = 0.5
    varied = sens.atmospheric_background_density(*args, grid)
    np.testing.assert_allclose(varied[:, 0], flat[:, 0])
    np.testing.assert_allclose(
        varied[:, 1],
        flat[:, 1] * sens.point_source_bin_sr(0.5) / sens.point_source_bin_sr(1.0),
    )
    # A bin that closes with energy takes the high end down and leaves the low.
    tapered = sens.atmospheric_background_density(
        *args, np.linspace(1.0, 0.25, LOG10_E.size)
    )
    np.testing.assert_allclose(tapered[0], flat[0])
    assert tapered[-1, 0] / flat[-1, 0] == pytest.approx(
        sens.point_source_bin_sr(0.25) / sens.point_source_bin_sr(1.0)
    )


def test_large_background_is_bounded_in_memory_and_matches_the_exact_answer():
    """Past its cap the limit follows the exact construction's own straight line.

    The exact construction holds every count a Poisson of the background can
    deliver, so its cost grows with that background and an unbounded one takes
    the machine down rather than merely taking a while. Above
    ``EXACT_BACKGROUND_MAX`` the answer continues along the line in ``sqrt(b)``
    the exact one is already on. The reference below is what that construction
    returned at ``b = 400`` before the cap was put in.
    """
    assert sens.sensitivity_upper_limit(400.0) == pytest.approx(35.788, rel=0.01)
    # The two branches meet either side of the join.
    below = float(sens.sensitivity_upper_limit(sens.EXACT_BACKGROUND_MAX))
    above = float(sens.sensitivity_upper_limit(sens.EXACT_BACKGROUND_MAX * 1.005))
    assert above == pytest.approx(below, rel=0.01)
    assert above > below
    # A background no counting experiment would accept still returns, quickly.
    huge = sens.sensitivity_upper_limit(np.array([1.0e4, 1.0e6]))
    assert np.all(np.isfinite(huge))
    assert huge[1] / huge[0] == pytest.approx(10.0, rel=0.05)


def test_exact_construction_refuses_a_background_it_cannot_hold():
    """The cap is enforced where the allocation happens, not left to the caller."""
    with pytest.raises(ValueError, match="does not fit in memory"):
        sens.feldman_cousins_upper_limit(0, sens.EXACT_BACKGROUND_MAX * 2.0)
