"""Tests for the first-principles effective areas against the pre-cleanup example 45.

The pinned values were computed once from
``git show pre-cleanup:examples/45_first_principles_reach.py`` on the same
inputs, with two exceptions noted at the ARCA effective-area test. The
column-level pieces run in well under a second; the effective areas are
marked slow.
"""

import importlib.util
import pathlib

import numpy as np
import pytest

from softpaws.detectors import ARCA230, ARCA_OPTICS, ICECUBE_OPTICS
from softpaws.response import first_principles as fp
from softpaws.transport.earth import overburden_km
from softpaws.utils.constants import RHO_ICE_G_CM3

PAPER_SCRIPTS = pathlib.Path(__file__).parents[1] / "scripts" / "2026_muon_transport"
EXAMPLE_45 = PAPER_SCRIPTS / "45_first_principles_reach.py"
ENERGY = np.array([1.0e4, 1.0e6, 1.0e8])
THRESHOLD_GEV = 100.0
N_ENERGY = 8
MIN_MODULES = 8.0


#: The deterministic range is integrated on a fixed lattice rather than on a
#: grid refined to each descent, which converged it and moved these pre-cleanup
#: pins by up to 2e-5. Anything the range does not reach still holds at
#: ``rtol = 1e-8``, which leaves room for the 1e-10 scatter between SciPy
#: builds on different Python versions.
QUADRATURE_RTOL = 5.0e-5


def _same(got, want, rtol=1e-8):
    np.testing.assert_allclose(np.asarray(got, dtype=float), np.asarray(want, dtype=float),
                               rtol=rtol, atol=0.0)


@pytest.mark.parametrize(
    "index, cc, nc, slope",
    [
        (0, [4.7e-35, 6.9e-34, 4.800000000000001e-33],
         [1.4999999999999997e-35, 2.6e-34, 1.9e-33],
         [0.7733376853629049, 0.4888371267874905, 0.3721287977844564]),
        (1, [3.0999999999999996e-35, 6.6e-34, 4.8e-33],
         [1.1e-35, 2.4e-34, 1.9e-33],
         [0.8696141945066015, 0.5184536269570298, 0.373074732296621]),
    ],
)
def test_isoscalar_species(index, cc, nc, slope):
    """Both species land on the CSMS table at its own energies."""
    xsec = fp.SPECIES[index]
    _same(xsec.cc(ENERGY), cc)
    _same(xsec.nc(ENERGY), nc)
    _same(xsec.local_slope(ENERGY), slope)


def test_column_profile():
    column, energy, total = fp.column_profile(1.0e6, THRESHOLD_GEV, N_ENERGY, RHO_ICE_G_CM3)
    _same(column, [0.0, 3.0426658234720563, 6.182152638799714, 9.435478251224854,
                   12.948189675697613, 16.55716364370917, 19.05541193403758,
                   20.19663599783324], rtol=QUADRATURE_RTOL)
    _same(energy, [1000000.0, 268269.57952797273, 71968.56730011529, 19306.977288832495,
                   5179.474679231213, 1389.4954943731389, 372.7593720314942, 100.0])
    _same(total, 20.19663599783324, rtol=QUADRATURE_RTOL)
    assert fp.column_profile(50.0, THRESHOLD_GEV, N_ENERGY) is None


def test_rock_range_ratio():
    cos_theta = np.array([-1.0, -0.5, -0.1, 0.3])
    ratio = fp.rock_range_ratio(1.0e6, THRESHOLD_GEV, cos_theta, ICECUBE_OPTICS, 1.0,
                                RHO_ICE_G_CM3)
    _same(ratio, [0.8039656217293253, 0.8084984048334695, 0.8940153189806164, 1.0],
          rtol=QUADRATURE_RTOL)
    _same(fp.rock_range_ratio(1.0e6, THRESHOLD_GEV, cos_theta, ICECUBE_OPTICS, 1.0,
                              RHO_ICE_G_CM3, far_source=None), np.ones(4))


def test_ic_column_volume():
    cos_theta = np.array([0.1, 0.5, 0.9])
    _same(fp.ic_column_volume_km3(1.0e6, THRESHOLD_GEV, cos_theta, ICECUBE_OPTICS, MIN_MODULES,
                                  N_ENERGY),
          [26.541334532796267, 25.9280249403473, 24.237636221273743], rtol=QUADRATURE_RTOL)
    _same(fp.ic_column_volume_km3(1.0e6, THRESHOLD_GEV, cos_theta, ICECUBE_OPTICS, None,
                                  N_ENERGY),
          [22.434331953168293, 21.622138488050656, 20.300252452726838], rtol=QUADRATURE_RTOL)
    _same(fp.ic_column_volume_km3(50.0, THRESHOLD_GEV, cos_theta, ICECUBE_OPTICS, MIN_MODULES,
                                  N_ENERGY), np.zeros(3))


def test_arca_column_volume():
    theta_deg = np.array([30.0, 90.0, 150.0])
    available_km = overburden_km(np.cos(np.deg2rad(theta_deg)), ARCA230.depth_km)
    _same(fp.arca_column_volume_km3(1.0e6, THRESHOLD_GEV, theta_deg, available_km, ARCA_OPTICS,
                                    MIN_MODULES, N_ENERGY),
          [21.516338428120523, 46.39894645825184, 57.3389587964313], rtol=QUADRATURE_RTOL)
    _same(fp.arca_column_volume_km3(1.0e6, THRESHOLD_GEV, theta_deg, available_km, ARCA_OPTICS,
                                    None, N_ENERGY),
          [8.81128292305936, 32.015896706458655, 41.13144010948358], rtol=QUADRATURE_RTOL)


def test_residuals():
    mean, scatter = fp.residuals(np.array([1.0, 2.0, 3.0]), np.array([1.1, 1.9, 3.3]),
                                 np.array([True, True, False]))
    _same([mean, scatter], [0.9782319760890369, 0.031834539934688634])


@pytest.mark.slow
def test_ic_effective_area():
    """Two energies, one channel and one species at a time."""
    grid = np.array([4.5, 6.0])
    _same(fp.ic_effective_area_cm2(ICECUBE_OPTICS, THRESHOLD_GEV, MIN_MODULES, N_ENERGY,
                                   ("mu",), fp.SPECIES[0], log10_e=grid),
          [404090.7623022614, 3760342.8236161144], rtol=QUADRATURE_RTOL)
    # The composite tau propagator applies the muon exponent after the tau one,
    # which carries the lattice shift twice and puts it just past QUADRATURE_RTOL.
    _same(fp.ic_effective_area_cm2(ICECUBE_OPTICS, THRESHOLD_GEV, MIN_MODULES, N_ENERGY,
                                   ("tau",), fp.SPECIES[1], log10_e=grid),
          [30037.2799140218, 655429.4140577874], rtol=1.0e-4)


@pytest.mark.slow
def test_arca_effective_area():
    """Two energies, the ``nu_mu`` channel, the neutrino species.

    Pinned against the script as it stood after step 1.3 of the cleanup, not
    the ``pre-cleanup`` tag: the sky average goes through
    :mod:`softpaws.transport.earth` since then, whose ``cos -> theta`` round
    trip moves one zenith sample by one ulp onto a ``prem_column``
    discontinuity at 103.49 degrees. That step changed this number by
    ``5.6e-6``; this one changes it by nothing.
    """
    _same(fp.arca_effective_area_cm2(ARCA_OPTICS, THRESHOLD_GEV, MIN_MODULES, N_ENERGY,
                                     ("mu",), fp.SPECIES[0], log10_e=np.array([5.0, 7.0])),
          [2555388.44642669, 33680994.456562325], rtol=QUADRATURE_RTOL)


@pytest.mark.slow
def test_build_model_icecube():
    """Both channels and both species, averaged, at two energies."""
    _same(fp.build_model("IceCube", ICECUBE_OPTICS, MIN_MODULES, N_ENERGY,
                         log10_e=np.array([4.5, 6.0])),
          [393831.79552468995, 4435382.909065569], rtol=QUADRATURE_RTOL)


def test_example_45_reexports():
    """The script still resolves every name the other examples load it for."""
    spec = importlib.util.spec_from_file_location("_example_45", EXAMPLE_45)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("ICECUBE_SITE", "ARCA_SITE", "Site", "SPECIES", "DEFAULT_MIN_MODULES",
                 "build_model", "fit_reach", "hit_count", "reach_offset_m", "effective_body_km",
                 "muon_threshold_gev", "column_profile", "rock_range_ratio",
                 "attenuation_length_m", "stochastic_muon_range_km", "load_example_32"):
        assert hasattr(module, name), name
    assert module.ICECUBE_SITE is ICECUBE_OPTICS
    assert module.DEFAULT_MIN_MODULES == MIN_MODULES
    assert module.SPECIES is fp.SPECIES
