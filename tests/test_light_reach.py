"""Tests for the derived light reach against the pre-cleanup example 45.

The pinned values were computed once from
``git show pre-cleanup:examples/45_first_principles_reach.py`` on the same
inputs, and the library reproduces them to floating-point identity. The
optics summary the paper quotes is checked against ``baseline.json``.
"""

import json
import pathlib

import numpy as np
import pytest

from softpaws.detectors import ARCA230, ARCA_OPTICS, ICECUBE, ICECUBE_OPTICS
from softpaws.response import light_reach as lr

BASELINE = json.loads(
    (pathlib.Path(__file__).parent / "regression" / "baseline.json").read_text()
)["first_principles_reach"]

ENERGY = np.array([1.0e4, 1.0e6, 1.0e8])
DISTANCE = np.array([10.0, 100.0, 300.0])
MIN_MODULES = 8.0

#: Per site: optics, prism geometry, and the pre-cleanup values.
SITES = {
    "IceCube": {
        "optics": ICECUBE_OPTICS,
        "geometry": ICECUBE,
        "n_sides": 6,
        "cherenkov": [254.21180494216918, 138.0208137636154, 86.50262807059921,
                      59.24555739436998, 43.101655509225914],
        "efficiency": [0.0, 0.07500000000000001, 0.212, 0.20133333333333334,
                       0.14279999999999998, 0.08199999999999999, 0.034, 0.0072, 0.0],
        "attenuation_spectrum": [26.563132345414388, 42.780934226981685, 55.555527777770834,
                                 60.57940037669571, 57.99689646869046, 45.26366506371308,
                                 30.370297989976983, 11.899999999999999, 7.0],
        "attenuation_length": 59.16079783099616,
        "module_area": 0.0081,
        "module_charge": [685.2750561248145, 12.97561301197828, 0.12307548766131764],
        "hit_radius": [38.68244061886604, 203.26558241141112, 430.12135017574536],
        "chord": 0.6990050352741266,
        "hit_probability": [1.0, 0.9999976838703322, 0.045022180783137536],
        "hit_count": [569.8935709575168, 284.9467854787584, 24.868251092808205],
        "reach_offset": [30.49698941024363, 240.07320337577977, 487.747956207446],
        "muon_threshold": 1791.0431055672432,
        "body_radius": [0.5946865729579999, 0.804262786923536, 1.0519375397552022],
        "body_height": [1.0609939788204872, 1.4801464067515595, 1.857747956207446],
        "body_weight": [0.999999996036236, 1.0, 1.0],
    },
    "ARCA230": {
        "optics": ARCA_OPTICS,
        "geometry": ARCA230,
        "n_sides": None,
        "cherenkov": [263.93531585618064, 143.3000606864582, 89.81132275661702,
                      61.511678844008806, 44.75027846696489],
        "efficiency": [0.0, 0.16499999999999998, 0.268, 0.228, 0.158, 0.0892, 0.0368,
                       0.008, 0.0],
        "attenuation_spectrum": [14.96, 38.760000000000005, 60.52, 77.18, 77.06666666666666,
                                 44.608, 17.68, 5.848000000000001, 3.4000000000000004],
        "attenuation_length": 68.0,
        "module_area": 0.0315,
        "module_charge": [3709.5802690725436, 81.02187502653237, 1.4127960718902566],
        "hit_radius": [90.75589007504735, 320.065192899959, 620.7836673041807],
        "chord": 0.5687449956483899,
        "hit_probability": [1.0, 1.0, 0.39151044383654476],
        "hit_count": [617.8308204714807, 308.91541023574035, 70.6628583554979],
        "reach_offset": [49.752671050184894, 308.8086873367453, 628.8692086683646],
        "muon_threshold": 1083.9428401670953,
        "body_radius": [0.5667526710501849, 0.8258086873367454, 1.1458692086683646],
        "body_height": [0.7315053421003698, 1.0208086873367455, 1.3408692086683647],
        "body_weight": [0.9999999999994144, 1.0, 1.0],
    },
}


def _same(got, want):
    np.testing.assert_allclose(np.asarray(got, dtype=float), np.asarray(want, dtype=float),
                               rtol=1e-10, atol=0.0)


def test_brightness_factor():
    _same(lr.brightness_factor(ENERGY),
          [14.697026444086465, 1519.9775091996605, 170586.27571595958])


@pytest.mark.parametrize("name", list(SITES))
def test_wavelength_chain(name):
    """Cherenkov yield, efficiency and attenuation on the wavelength grid."""
    site, want = SITES[name]["optics"], SITES[name]
    _same(lr.cherenkov_spectrum_per_m_per_nm(site)[::50], want["cherenkov"])
    _same(lr.detection_efficiency(site)[::25], want["efficiency"])
    _same(lr.attenuation_spectrum_m(site)[::25], want["attenuation_spectrum"])
    _same(lr.attenuation_length_m(site), want["attenuation_length"])
    _same(lr.module_area_m2(site), want["module_area"])


@pytest.mark.parametrize("name", list(SITES))
def test_charge_and_hits(name):
    """Collected charge, the one-pe radius and the coincidence probability."""
    site, want = SITES[name]["optics"], SITES[name]
    _same(lr.module_charge_pe(DISTANCE, 1.0e6, site), want["module_charge"])
    _same(lr.hit_radius_m(ENERGY, site), want["hit_radius"])
    _same(lr.hit_probability(DISTANCE, 1.0e6, site), want["hit_probability"])


@pytest.mark.parametrize("name", list(SITES))
def test_reach(name):
    """Mean chord, hit count, reach offset, threshold and the dilated body."""
    site, geometry, want = SITES[name]["optics"], SITES[name]["geometry"], SITES[name]
    chord = lr.instrumented_chord_km(geometry.radius_km, geometry.height_km, want["n_sides"])
    _same(chord, want["chord"])
    _same([lr.hit_count(x, 1.0e6, site, chord)[0] for x in (-1.0e4, 0.0, 200.0)],
          want["hit_count"])
    _same(lr.reach_offset_m(ENERGY, site, chord, MIN_MODULES), want["reach_offset"])
    _same(lr.muon_threshold_gev(site, MIN_MODULES, chord), want["muon_threshold"])
    radius, height, weight = lr.effective_body_km(
        geometry.radius_km, geometry.height_km, ENERGY, site, MIN_MODULES, want["n_sides"])
    _same(radius, want["body_radius"])
    _same(height, want["body_height"])
    _same(weight, want["body_weight"])


def test_attenuation_override_is_flat():
    """A fitted flat length replaces the whole spectrum."""
    from dataclasses import replace

    site = replace(ICECUBE_OPTICS, attenuation_override_m=40.0)
    assert np.all(lr.attenuation_spectrum_m(site) == 40.0)
    assert lr.attenuation_length_m(site) == 40.0


@pytest.mark.parametrize("name", list(SITES))
def test_paper_optics_summary(name):
    """The derived optics the paper quotes per site."""
    site, want = SITES[name]["optics"], BASELINE[name]
    rtol = BASELINE["rtol"]
    assert lr.attenuation_length_m(site) == pytest.approx(
        want["attenuation_length_400nm_m"], rel=rtol)
    assert site.module_density_per_km3 == pytest.approx(want["module_density_per_km3"], rel=rtol)
    assert float(lr.hit_radius_m(1.0e6, site)[0]) == pytest.approx(
        want["one_pe_radius_1pev_m"], rel=rtol)
