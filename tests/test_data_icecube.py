"""Tests for the IceCube DR2 conveniences. They need the release on disk."""

import json
import pathlib

import numpy as np
import pytest

from softpaws.data import icecube
from softpaws.data.schema import SEASONS

DATA_DIR = pathlib.Path(__file__).parents[1] / "src" / "softpaws" / "data" / "dataverse_files"
BASELINE = json.loads(
    (pathlib.Path(__file__).parent / "regression" / "baseline.json").read_text()
)["dr2_sample"]

pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "irfs").exists(), reason="the DR2 release is not on disk"
)

YEAR_S = 365.25 * 86400.0


def test_irf_season():
    assert icecube.irf_season("IC86_IV") == "IC86"
    assert icecube.irf_season("IC59") == "IC59"
    assert len(icecube.IC86_SEASONS) == 11
    assert all(s.startswith("IC86") for s in icecube.IC86_SEASONS)


def test_livetimes_match_the_paper():
    """The release is 13.6 years, of which IC86 is 10.73."""
    total = icecube.total_livetime_s(DATA_DIR) / YEAR_S
    ic86 = icecube.total_livetime_s(DATA_DIR, icecube.IC86_SEASONS) / YEAR_S
    assert total == pytest.approx(BASELINE["dr2_exposure_yr"], abs=0.05)
    assert ic86 == pytest.approx(BASELINE["ic86_livetime_yr"], abs=0.01)


def test_effective_area_cache_and_average():
    a = icecube.load_effective_area(DATA_DIR, "IC86_I")
    b = icecube.load_effective_area(DATA_DIR, "IC86_VII")
    assert a is b
    up = icecube.hemisphere_average(a, "upgoing")
    down = icecube.hemisphere_average(a, "downgoing")
    assert up.shape == a.log10_energy_centers.shape
    assert np.all(up >= a.values.min(axis=1)) and np.all(up <= a.values.max(axis=1))
    # Upgoing tracks dominate the through-going sample at 100 TeV.
    i = int(np.argmin(np.abs(a.log10_energy_centers - 5.0)))
    assert up[i] > down[i]
    with pytest.raises(ValueError):
        icecube.hemisphere_average(a, "sideways")


def test_livetime_weighted_effective_area():
    grid = np.linspace(3.0, 8.0, 26)
    curve, livetime_s = icecube.livetime_weighted_effective_area(DATA_DIR, grid)
    assert curve.shape == grid.shape and np.all(np.isfinite(curve))
    assert livetime_s == pytest.approx(icecube.total_livetime_s(DATA_DIR))
    ic86_only, _ = icecube.livetime_weighted_effective_area(
        DATA_DIR, grid, seasons=icecube.IC86_SEASONS
    )
    a = icecube.load_effective_area(DATA_DIR, "IC86_I")
    ic86 = icecube.hemisphere_average(a)
    np.testing.assert_allclose(
        ic86_only, np.interp(grid, a.log10_energy_centers, ic86)
    )
    assert len(SEASONS) == 14


@pytest.mark.slow
def test_load_events_counts_the_paper_sample():
    """908,280 upgoing IC86 events, as the paper counts them."""
    events = icecube.load_events(DATA_DIR)
    upgoing = int(np.sum(events.dec >= 0.0))
    assert upgoing == BASELINE["upgoing_events"]
    checked = icecube.load_events(DATA_DIR, seasons=("IC86_I",), within_uptime=True)
    assert 0 < checked.n_events <= icecube.load_events(DATA_DIR, seasons=("IC86_I",)).n_events
