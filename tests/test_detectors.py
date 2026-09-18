"""Tests for the detector-site registry against the paper's geometry table."""

import json
import pathlib

import numpy as np
import pytest

from softpaws.constants import CM_PER_KM
from softpaws.detectors import (
    ARCA21,
    ARCA230,
    GEN2,
    ICECUBE,
    MAX_UPSTREAM_KM,
    SITES,
    TRIDENT,
    TRIDENT_2025,
    Site,
    get_site,
)

GEOMETRY = json.loads(
    (pathlib.Path(__file__).parent / "regression" / "baseline.json").read_text()
)["geometry_table"]


@pytest.mark.parametrize("site", [ICECUBE, ARCA230, ARCA21])
def test_geometry_table(site):
    """Table E.2: footprint radius, height, volume and projected areas."""
    want = GEOMETRY[site.name]
    assert site.radius_km == pytest.approx(want["footprint_radius_km"], abs=5e-4)
    assert site.height_km == pytest.approx(want["height_km"])
    assert float(site.detector_volume_km3()) == pytest.approx(want["volume_km3"], rel=0.01)
    assert site.below_km == pytest.approx(want["below_km"])
    if "depth_km" in want:
        assert site.depth_km == pytest.approx(want["depth_km"], abs=5e-3)

    cos_theta = np.linspace(-1.0, 1.0, 2001)
    area = site.projected_area_km2(cos_theta)
    low, high = want["proj_area_range_km2"]
    # The table's lower end came off a coarse zenith grid, so it sits above
    # the true horizon minimum for the cylinders; the upper end and the mean
    # are grid independent.
    assert float(area.min()) <= low * 1.01
    assert float(area.max()) == pytest.approx(high, rel=0.01)
    assert site.mean_projected_area_km2() == pytest.approx(want["proj_area_mean_km2"], rel=0.02)


def test_arca_vertical_column():
    """Table E.2: the vertical water column above the ARCA blocks."""
    neutrino_column, muon_column_km = ARCA230.columns(1.0)
    assert float(neutrino_column) == pytest.approx(
        GEOMETRY["ARCA230"]["column_vertical_g_cm2"], rel=0.01
    )
    assert float(muon_column_km) == pytest.approx(ARCA230.depth_km)


def test_columns_below_horizon():
    """Below the horizon the neutrino sees the Earth and the muon sees rock."""
    neutrino_column, muon_column_km = ICECUBE.columns(np.array([0.5, -0.5, -1.0]))
    assert neutrino_column[0] == pytest.approx(ICECUBE.depth_km / 0.5 * CM_PER_KM * 0.92)
    assert neutrino_column[2] > neutrino_column[1] > neutrino_column[0]
    assert muon_column_km[0] == pytest.approx(ICECUBE.depth_km / 0.5)
    assert np.all(muon_column_km[1:] == MAX_UPSTREAM_KM)


def test_sphere_and_replace():
    sphere = Site("ball", "sphere", 0.0, 1.0, 1.0, 0.5)
    assert np.allclose(sphere.projected_area_km2([-1.0, 0.0, 1.0]), np.pi * 0.25)
    assert float(sphere.detector_volume_km3()) == pytest.approx(4.0 / 3.0 * np.pi * 0.125)
    bigger = sphere.replace(radius_km=1.0)
    assert bigger.radius_km == 1.0 and sphere.radius_km == 0.5


def test_registry():
    assert get_site("icecube") is ICECUBE
    assert get_site("TRIDENT-2025") is TRIDENT_2025
    assert TRIDENT_2025.radius_km == 1.75 and TRIDENT_2025.depth_km == TRIDENT.depth_km
    assert GEN2.detector_volume_km3() == pytest.approx(7.9, rel=1e-6)
    assert len(SITES) == 8
    with pytest.raises(KeyError):
        get_site("DUMONT")


def test_validation():
    with pytest.raises(ValueError):
        Site("bad", "cube", 0.0, 1.0, 1.0, 0.5)
    with pytest.raises(ValueError):
        Site("flat", "cylinder", 0.0, 1.0, 1.0, 0.5, height_km=0.0)
