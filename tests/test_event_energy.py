"""Tests for the single-event energy reconstruction."""

import json
import pathlib

import numpy as np
import pytest

from softpaws.comparison import event_energy as ee
from softpaws.transport.cross_section import bgr18_cross_section

BASELINE = json.loads(
    (pathlib.Path(__file__).parent / "regression" / "baseline.json").read_text()
)


def test_sea_path_and_two_layer_column():
    # Straight up from 3.4 km depth is 3.4 km; near the horizon it is long.
    assert ee.sea_path_km(90.0, 3.4) == pytest.approx(3.4, rel=1e-9)
    assert ee.sea_path_km(0.6, 3.4) > 100.0
    total, water, rock, path = ee.two_layer_column_g_cm2(0.6, 3.4, 3.5, ds_km=0.05)
    assert path == pytest.approx(ee.sea_path_km(0.6, 3.4), abs=0.1)
    # Upward from 3.4 km with the seabed at 3.5 km the ray never meets rock.
    assert rock == 0.0 and water > 0.0
    _, water_down, rock_down, _ = ee.two_layer_column_g_cm2(-1.0, 3.4, 3.5, ds_km=0.05)
    assert rock_down > 0.0 and water_down > 0.0
    assert total == pytest.approx((water * 1.03 + rock * 2.65) * 1e5)
    # Straight up: 3.4 km of water only.
    total_up, water_up, rock_up, _ = ee.two_layer_column_g_cm2(90.0, 3.4, 3.5, ds_km=0.01)
    assert rock_up == 0.0 and water_up == pytest.approx(3.4, abs=0.02)


def test_potential_density_csda_and_gaussian():
    w = np.linspace(0.0, 24.0, 481)
    csda = ee.potential_density("csda", w)
    assert np.allclose(csda, csda[0]) and 1.5 < csda[0] < 3.0
    gauss = ee.potential_density("gaussian", w, n_x=30)
    assert gauss.shape == w.shape and np.all(np.isfinite(gauss))
    with pytest.raises(ValueError):
        ee.potential_density("mean", w)


def test_measurement_and_likelihood_shapes():
    w = np.linspace(0.0, 24.0, 481)
    u = ee.potential_density("csda", w)
    like = ee.lognormal_measurement(np.array([7.5, 8.0, 8.5]), 1.2e8, (3.5e7, 3.8e8))
    assert like[1] > like[0] and like[1] > like[2]
    measurement = lambda log10_e: ee.lognormal_measurement(log10_e, 1.2e8, (3.5e7, 3.8e8))  # noqa: E731
    energy = np.logspace(7.5, 10.0, 6)
    raw = ee.energy_likelihood(u, w, energy, measurement)
    conditional = ee.energy_likelihood(u, w, energy, measurement, accept_gev=1.0e5)
    assert raw.shape == conditional.shape == (6,)
    assert np.all(conditional <= raw * 1.0 + 1e-300)


def test_survival_and_unity_energy():
    xs = bgr18_cross_section()
    column = ee.KM3_230213A.traversed_column_g_cm2
    s = ee.survival_through_column(np.array([1e7, 1e9]), column, xs)
    assert 0.0 < s[1] < s[0] < 1.0
    unity = ee.interaction_unity_energy_gev(column, xs)
    assert ee.survival_through_column(unity, column, xs) == pytest.approx(np.exp(-1.0), rel=0.02)


def test_parent_energy_posterior_and_quantile():
    rec = ee.parent_energy_posterior(2.9e5, 1.95, 2.38)
    for key in ("exact", "fp"):
        assert np.trapezoid(rec[key], rec["w"]) == pytest.approx(1.0)
    assert ee.quantile(rec["w"], rec["exact"], 0.5) > 0.0
    assert ee.quantile(rec["w"], rec["exact"], 0.9) > ee.quantile(rec["w"], rec["exact"], 0.5)


@pytest.mark.slow
def test_km3_event_energy_matches_the_paper():
    """Section V B: E^-2 prior, exact kernel: mode 186, median 250, 90% [53, 3561] PeV."""
    want = BASELINE["km3_event_energy_pev"]
    w = np.linspace(0.0, 24.0, 4801)
    log10_enu = np.linspace(7.0, 10.5, 351)
    energy = 10.0**log10_enu
    xs = bgr18_cross_section()
    event = ee.KM3_230213A
    measurement = lambda log10_e: ee.lognormal_measurement(  # noqa: E731
        log10_e, event.muon_energy_gev, event.muon_energy_90_gev
    )
    survival = ee.survival_through_column(energy, event.traversed_column_g_cm2, xs)
    for kind in ("exact", "gaussian", "csda"):
        u = ee.potential_density(kind, w)
        like = ee.energy_likelihood(u, w, energy, measurement)
        p = ee.energy_posterior(like, log10_enu, lambda e: (e / 1e5) ** -2.0, xs, survival)
        got = np.array(ee.posterior_summary(p, log10_enu)) / 1e6
        np.testing.assert_allclose(got, want["E-2"][kind], rtol=want["rtol"])
