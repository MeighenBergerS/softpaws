"""Tests for the declination-resolved response.

The pinned numbers were computed once from
``git show pre-cleanup:examples/35_point_source_effective_area.py`` and
``:examples/46_declination_resolved_reach.py`` on the same inputs; every
lifted function reproduced them exactly (maximum relative difference zero).
"""

import pathlib

import numpy as np
import pytest

from softpaws.detectors import ARCA230, ARCA_OPTICS, ICECUBE, ICECUBE_OPTICS
from softpaws.response import declination as dec
from softpaws.response.first_principles import SPECIES

DATA_DIR = pathlib.Path(__file__).parents[1] / "src" / "softpaws" / "data" / "dataverse_files"
COS_THETA = np.array([-0.9, -0.4, 0.0, 0.3, 0.8])


def test_polar_band_directions():
    """A band maps to minus its sin(dec), sampled uniformly inside."""
    edges = np.array([-1.0, 0.0, 1.0])
    got = dec.polar_band_directions(edges, n_sub=2)
    np.testing.assert_allclose(got, [[0.75, 0.25], [-0.25, -0.75]])
    assert dec.polar_band_directions(np.linspace(-1, 1, 6)).shape == (5, dec.N_SUB_BAND)


def test_zenith_band_weights():
    """At the Pole a source sits at one zenith all day; at the equator it sweeps."""
    centres, weights = dec.zenith_band_weights(-90.0, np.array([30.0, -60.0]))
    assert centres.shape == (dec.N_COS_THETA,) and weights.shape == (2, dec.N_COS_THETA)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    # One band holds the whole day, at cos(theta) = -sin(dec).
    for row, declination in zip(weights, (30.0, -60.0)):
        assert row.max() == pytest.approx(1.0)
        assert centres[np.argmax(row)] == pytest.approx(-np.sin(np.deg2rad(declination)), abs=0.01)
    _, spread = dec.zenith_band_weights(0.0, np.array([0.0]))
    assert np.count_nonzero(spread) > 10


def test_point_source_sensitivity_and_range():
    """The ceiling is one quadrature, so it scales inversely with exposure."""
    energy = 10.0**dec.COMMON_LOG10_E
    aeff = 1.0e4 * (energy / dec.PIVOT_ENERGY_GEV) ** 0.5
    one = dec.point_source_sensitivity(aeff, 1.0e8, 2.0)
    assert dec.point_source_sensitivity(aeff, 2.0e8, 2.0) == pytest.approx(0.5 * one)
    # Columns are handled independently.
    stacked = dec.point_source_sensitivity(np.stack([aeff, 2.0 * aeff], axis=1), 1.0e8, 2.0)
    np.testing.assert_allclose(stacked, [one, 0.5 * one])
    lo, hi = dec.central_energy_range(aeff, 2.0)
    assert 5.0 <= lo < hi <= 8.0


@pytest.mark.parametrize(
    "site, reach_km, channels, want",
    [
        (ICECUBE, None, "both", [4055845.798522181, 696193.5097843928]),
        (ICECUBE, 0.0178, "mu", [2313194.613364296, 238301.39121354712]),
        (ARCA230, None, "both", [8409310.903071176, 1002972.489553515]),
    ],
)
def test_directional_effective_area(site, reach_km, channels, want):
    """Two directions at 1 PeV, against the pre-cleanup script."""
    got = dec.directional_effective_area_cm2(
        site, np.array([0.3, -0.9]), 1.0e3, reach_km=reach_km, channels=channels,
        log10_e=np.array([6.0]),
    )
    np.testing.assert_allclose(got[0], want, rtol=1e-10)


def test_band_average_is_the_mean_of_its_directions():
    edges = np.array([-1.0, 0.0, 1.0])
    log10_e = np.array([5.0])
    banded = dec.band_averaged_effective_area_cm2(
        ICECUBE, edges, 1.0e3, log10_e=log10_e, n_sub=2
    )
    directions = dec.polar_band_directions(edges, n_sub=2)
    per_direction = dec.directional_effective_area_cm2(
        ICECUBE, directions.ravel(), 1.0e3, log10_e=log10_e
    )
    np.testing.assert_allclose(banded, per_direction.reshape(1, 2, 2).mean(axis=2))


def test_band_statistics():
    """A pure power-law ratio has that slope as its tilt and no scatter."""
    log10_e = np.linspace(4.0, 8.0, 21)
    band = log10_e >= 5.0
    model = np.ones((log10_e.size, 2))
    published = np.stack([10.0 ** (0.2 * log10_e - 1.0), np.full(log10_e.size, 3.0)], axis=1)
    level, tilt, scatter = dec.band_statistics(log10_e, published, model, band)
    np.testing.assert_allclose(tilt, [0.2, 0.0], atol=1e-12)
    np.testing.assert_allclose(scatter, [0.0, 0.0], atol=1e-12)
    assert level[1] == pytest.approx(3.0)
    # A band with too few finite points is reported as missing, not guessed.
    published[band, 1] = np.nan
    assert np.isnan(dec.band_statistics(log10_e, published, model, band)[0][1])


@pytest.mark.slow
def test_derived_directional_effective_area():
    """The derived-optics family at 1 PeV, against the pre-cleanup script."""
    got = dec.derived_directional_effective_area_cm2(
        ICECUBE, ICECUBE_OPTICS, np.array([0.3, -0.9]), 8.0, ("mu",), SPECIES[0],
        log10_e=np.array([6.0]),
    )
    np.testing.assert_allclose(got[0], [6264395.661338991, 351286.58524662815], rtol=1e-10)
    with_tau = dec.derived_directional_effective_area_cm2(
        ICECUBE, ICECUBE_OPTICS, np.array([0.3]), 8.0, ("mu", "tau"), SPECIES[0],
        log10_e=np.array([6.0]),
    )
    assert with_tau[0, 0] > got[0, 0]


@pytest.mark.slow
def test_column_target_volume():
    """Below threshold the volume is zero; above it, the pinned value."""
    cos_theta = np.array([-0.8, 0.4])
    available = np.full(2, 5.0)
    zero = dec.column_target_volume_km3(
        ICECUBE, ICECUBE_OPTICS, 500.0, cos_theta, available, 8.0
    )
    np.testing.assert_array_equal(zero, np.zeros(2))
    got = dec.column_target_volume_km3(
        ICECUBE, ICECUBE_OPTICS, 1.0e6, cos_theta, available, 8.0, n_sides=6
    )
    assert np.all(got > 0.0) and got.shape == (2,)
    # A smaller halo weight keeps less of the reach-dilated body.
    less = dec.column_target_volume_km3(
        ICECUBE, ICECUBE_OPTICS, 1.0e6, cos_theta, available, 8.0, halo_weight=0.0, n_sides=6
    )
    assert np.all(less < got)


@pytest.mark.slow
def test_derived_band_average_uses_every_species():
    edges = np.array([0.0, 1.0])
    log10_e = np.array([5.0])
    both = dec.derived_band_averaged_effective_area_cm2(
        ARCA230, ARCA_OPTICS, edges, 8.0, ("mu",), SPECIES, log10_e=log10_e, n_sub=1
    )
    per_species = [
        dec.derived_directional_effective_area_cm2(
            ARCA230, ARCA_OPTICS, dec.polar_band_directions(edges, 1).ravel(), 8.0,
            ("mu",), xsec, log10_e=log10_e,
        )
        for xsec in SPECIES
    ]
    np.testing.assert_allclose(both, np.mean(per_species, axis=0))


@pytest.mark.skipif(not (DATA_DIR / "irfs").exists(), reason="the DR2 release is not on disk")
def test_banded_dr2_table():
    """The published table keeps its declination axis and its band edges."""
    from softpaws.data import banded_effective_area

    edges, aeff = banded_effective_area(DATA_DIR, dec.COMMON_LOG10_E)
    assert edges[0] == pytest.approx(-1.0) and edges[-1] == pytest.approx(1.0)
    assert aeff.shape == (dec.COMMON_LOG10_E.size, edges.size - 1)
    assert np.all(aeff >= 0.0) and np.any(aeff > 0.0)
