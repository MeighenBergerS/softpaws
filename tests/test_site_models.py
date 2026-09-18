"""Tests for the per-site forward models of :mod:`softpaws._paper.site_fit`.

Every pinned number was computed once from the pre-cleanup examples 33 and 56
at the commit tagged ``pre-cleanup``, at the best-fit points of the cached
chains, and the two paths agree bit for bit. The one exception is the ARCA230
trigger curve, where the shared loader in :mod:`softpaws.data.published`
converts to cm^2 before interpolating and the old script after, which moves
the curve by 2e-15 in relative terms.
"""

import pathlib

import numpy as np
import pytest

from softpaws._paper import site_fit as sm
from softpaws.detectors import ARCA230, ICECUBE, PONE, TRIDENT

DATA_DIR = pathlib.Path(__file__).parents[1] / "src" / "softpaws" / "data" / "dataverse_files"

#: A parameter vector inside every prior box, used for the pinned curves.
#: The deterministic range is integrated on a fixed lattice rather than on a
#: grid refined to each descent, which converged it and moved these pre-cleanup
#: pins by up to 2e-5. Anything the range does not reach still holds at the
#: tighter default.
QUADRATURE_RTOL = 5.0e-5

THETA = np.array([0.8, 3.2, 1.0, 0.4538, 0.02])


# ---------------------------------------------------------------------------
# The parameter block
# ---------------------------------------------------------------------------


def test_parameter_names_split_into_three_groups():
    assert sm.PARAM_NAMES == ("eps_0", "log10_e_thr", "b_scale", "lam", "reach_km")
    assert sm.CORNER_PARAMS == ("log10_e_thr", "b_scale", "lam")
    assert sm.INSTRUMENT_PARAMS == ("eps_0", "reach_km")
    assert sm.PHYSICS_PARAMS == ("b_scale", "lam")
    assert set(sm.PARAM_LABELS) == set(sm.PARAM_NAMES)


def test_pinned_constants():
    assert sm.LAMBDA_BGR18 == 0.4538
    assert sm.LAMBDA_PIVOT_GEV == 1.0e6
    assert sm.SITE_FIT_REACH_PIVOT_GEV == 1.0e6
    assert sm.F_TAU == 1.0
    assert sm.REACH_EXAMPLE28_KM == 0.0193
    assert sm.SMEARING_LOG10_E_THR == 2.85
    assert sm.B_SCALE_FLOOR == pytest.approx(0.28747283844399724, rel=1e-12)


def test_priors_share_the_physics_box():
    for name in sm.PHYSICS_PARAMS:
        assert sm.PRIORS["IceCube"][name] == sm.PRIORS["ARCA230"][name]
    assert sm.PRIORS["IceCube"]["eps_0"] == (0.0, 1.0)
    assert sm.PRIORS["IceCube"]["reach_km"] == (-0.02, 0.10)
    assert sm.PRIORS["ARCA230"]["reach_km"] == (-0.05, 0.40)
    # Zero has to sit inside every reach box, so the data can say no reach is
    # needed, and the b_scale floor has to be the lower edge.
    for priors in sm.PRIORS.values():
        assert priors["reach_km"][0] < 0.0 < priors["reach_km"][1]
        assert priors["b_scale"][0] == sm.B_SCALE_FLOOR


def test_expected_values_are_the_derived_ones():
    assert sm.EXPECTED["IceCube"]["b_scale"] == 1.0
    assert sm.EXPECTED["IceCube"]["reach_km"] == sm.REACH_EXAMPLE28_KM
    for site in ("ARCA230", "P-ONE", "TRIDENT"):
        assert sm.EXPECTED[site] == {"b_scale": 1.0, "lam": sm.LAMBDA_BGR18}


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_geometry_comes_from_the_site_registry():
    assert sm.IC_RADIUS_KM == ICECUBE.radius_km == pytest.approx(0.5641895835477563, rel=1e-14)
    assert sm.IC_HEIGHT_KM == ICECUBE.height_km
    assert sm.IC_N_SIDES == ICECUBE.n_sides
    assert sm.IC_SIDE_COEFF == pytest.approx(2.100150271617328, rel=1e-14)
    assert sm.IC_ICE_BELOW_KM == pytest.approx(0.79866, rel=1e-14)
    assert sm.ARCA_BLOCK_RADIUS_KM == ARCA230.radius_km
    assert sm.ARCA_BLOCK_HEIGHT_KM == ARCA230.height_km
    assert sm.ARCA_N_BLOCKS == ARCA230.n_blocks
    assert sm.ARCA_DEPTH_KM == pytest.approx(3.184, rel=1e-14)
    assert sm.ARCA_WATER_BELOW_KM == pytest.approx(0.396, rel=1e-14)
    assert sm.ARCA_MAX_SEA_PATH_KM == 100.0


def test_energy_grids():
    assert sm.IC_LOG10_E.size == 26
    assert sm.IC_LOG10_E[0] == 3.0 and sm.IC_LOG10_E[-1] == 8.0
    assert sm.IC_FIT_BAND == (5.0, 7.8)
    assert sm.ARCA_LOG10_E.size == 21
    assert sm.ARCA_LOG10_E[0] == pytest.approx(4.0) and sm.ARCA_LOG10_E[-1] == pytest.approx(8.0)
    assert sm.ARCA_FIT_BAND == (5.0, 7.5)
    assert sm.DIGITIZED_FIT_BAND == (5.0, 6.9)
    assert sm.ARCA_N_ZENITH == 20 and sm.ARCA_N_RUNG == 32
    assert sm.IC_N_DEC == 40 and sm.IC_N_RUNG == 80


def test_water_sites_match_the_registry():
    pone, trident = sm.water_sites()
    for site, registry in ((pone, PONE), (trident, TRIDENT)):
        assert site.name == registry.name
        assert site.radius_km == registry.radius_km
        assert site.height_km == registry.height_km
        assert site.n_blocks == registry.n_blocks
        assert site.depth_km == registry.depth_km
        assert site.below_km == registry.below_km
        # Both collaborations simulate nu_mu only against pure absorption.
        assert site.f_tau == 0.0
        assert site.regeneration is False
        assert site.fit_band == sm.DIGITIZED_FIT_BAND
        assert site.reach_prior == (-0.05, 0.40)
    assert pone.selection_level == "trigger"
    assert trident.selection_level == "6 deg cut"


def test_arca_is_a_water_site_with_the_tau_channel_on():
    site = sm.ARCA230_WATER_SITE
    assert (site.radius_km, site.height_km, site.n_blocks) == (
        ARCA230.radius_km,
        ARCA230.height_km,
        ARCA230.n_blocks,
    )
    assert site.depth_km == ARCA230.depth_km
    assert site.below_km == ARCA230.below_km
    assert site.f_tau == sm.F_TAU
    assert site.regeneration is True


def test_projected_area_is_the_convex_body_one():
    radius = sm.ARCA_BLOCK_RADIUS_KM
    overhead = float(sm.arca_projected_area_km2(np.array(0.0), radius))
    horizon = float(sm.arca_projected_area_km2(np.array(90.0), radius))
    assert overhead == pytest.approx(1.6794263175707245, rel=1e-12)
    assert horizon == pytest.approx(1.3069760000000001, rel=1e-12)
    assert overhead == pytest.approx(sm.ARCA_N_BLOCKS * np.pi * radius**2)
    assert horizon == pytest.approx(
        sm.ARCA_N_BLOCKS * 2.0 * radius * sm.ARCA_BLOCK_HEIGHT_KM
    )
    # The ARCA helper is the water one bound to the ARCA230 site.
    theta = np.linspace(0.0, 180.0, 11)
    assert np.array_equal(
        sm.arca_projected_area_km2(theta, radius),
        sm.water_projected_area_km2(sm.ARCA230_WATER_SITE, theta, radius),
    )


# ---------------------------------------------------------------------------
# Cross-section tilt and published curves
# ---------------------------------------------------------------------------


def test_tilted_cc_recovers_the_table_at_the_bgr18_slope():
    energy = np.logspace(4.0, 8.0, 9)
    assert np.array_equal(sm.tilted_cc(energy, sm.LAMBDA_BGR18), sm.CROSS_SECTION.cc(energy))
    # A steeper slope pivots about LAMBDA_PIVOT_GEV and leaves it untouched.
    tilted = sm.tilted_cc(energy, sm.LAMBDA_BGR18 + 0.1)
    ratio = tilted / sm.CROSS_SECTION.cc(energy)
    assert ratio == pytest.approx((energy / sm.LAMBDA_PIVOT_GEV) ** 0.1)


def test_published_water_curves():
    arca = sm.arca230_trigger()
    assert arca.size == sm.ARCA_LOG10_E.size
    assert int(np.isfinite(arca).sum()) == 20
    assert arca[[5, 10, 15]] == pytest.approx(
        [1415842.5778521637, 6031109.826153485, 18625498.56856658], rel=1e-12
    )
    pone = sm.pone_allsky_cm2()
    assert int(np.isfinite(pone).sum()) == 15
    assert pone[[5, 10, 14]] == pytest.approx(
        [986618.6488643411, 4076784.0030745193, 8730678.85514139], rel=1e-12
    )
    trident = sm.trident_allsky_cm2()
    assert int(np.isfinite(trident).sum()) == 15
    assert trident[[5, 10, 14]] == pytest.approx(
        [5588261.886529056, 15617614.644383255, 30099381.26228172], rel=1e-12
    )
    assert sm.TRIDENT_BAND_WEIGHTS == pytest.approx([0.4, 0.2, 0.4])


# ---------------------------------------------------------------------------
# Forward models
# ---------------------------------------------------------------------------


def test_water_model_pinned_on_trident():
    """TRIDENT's ladder carries no regeneration, so this one is quick."""
    site = [s for s in sm.water_sites() if s.name == "TRIDENT"][0]
    weights, neutrino_column, muon_column_km = sm.water_columns(site)
    ladders = sm.water_ladders(site, neutrino_column)
    assert ladders["mu"][0].shape == (sm.ARCA_LOG10_E.size, 1)
    assert not np.any(ladders["tau"][1])
    aeff = sm.water_model(THETA, site, ladders, weights, muon_column_km)
    # The deterministic range is integrated on a fixed lattice rather than on a
    # grid refined to each descent, which converged it and moved this
    # pre-cleanup pin by 3e-6.
    assert aeff[[0, 10, 20]] == pytest.approx(
        [577098.7590180387, 12493443.381331533, 75020799.66142105], rel=QUADRATURE_RTOL
    )
    # A mask reproduces the same nodes bit for bit. It reads the same lattice,
    # so which energies are asked for together no longer moves the quadrature.
    select = (sm.ARCA_LOG10_E >= 5.0) & (sm.ARCA_LOG10_E <= 6.0)
    assert np.array_equal(
        sm.water_model(THETA, site, ladders, weights, muon_column_km, select), aeff[select]
    )


@pytest.mark.slow
def test_arca_model_is_the_water_model_at_arca_geometry():
    weights, neutrino_column, muon_column_km = sm.arca_columns()
    ladders = sm.arca_ladders(neutrino_column)
    assert ladders["mu"][1].shape == (sm.ARCA_LOG10_E.size, sm.ARCA_N_RUNG, sm.ARCA_N_ZENITH)
    aeff = sm.arca_model(THETA, ladders, weights, muon_column_km)
    assert aeff[[0, 10, 20]] == pytest.approx(
        [113867.22860647023, 4238862.002621933, 30854765.478698894], rel=QUADRATURE_RTOL
    )
    assert np.array_equal(
        aeff, sm.water_model(THETA, sm.ARCA230_WATER_SITE, ladders, weights, muon_column_km)
    )


@pytest.mark.slow
def test_icecube_model_pinned():
    ladders = sm.icecube_ladders()
    for flavour in ("mu", "tau"):
        assert len(ladders[flavour]) == 4
        for block in ladders[flavour]:
            assert block.shape == (sm.IC_LOG10_E.size, sm.IC_N_RUNG)
    aeff = sm.icecube_model(THETA, ladders)
    assert aeff[[0, 10, 20]] == pytest.approx(
        [1251.495659754909, 770702.0721737217, 5829577.117627461], rel=QUADRATURE_RTOL
    )
    select = (sm.IC_LOG10_E >= sm.IC_FIT_BAND[0]) & (sm.IC_LOG10_E <= sm.IC_FIT_BAND[1])
    assert sm.icecube_model(THETA, ladders, select) == pytest.approx(aeff[select], rel=1e-5)


@pytest.mark.slow
def test_truncation_stays_flat_in_the_truncation_depth():
    """The gamma-law truncation is the approximation under test."""
    rows = sm.truncation_table(log10_energies=(6.0,))
    assert len(rows) == 4
    ratios = np.array([row["ratio"] for row in rows])
    assert np.all(ratios > 0.8) and np.all(ratios < 1.2)
    assert ratios.std() < 0.03


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


def test_water_detector_binds_its_curve_and_priors():
    site = [s for s in sm.water_sites() if s.name == "TRIDENT"][0]
    detector = sm.build_water_detector(site)
    assert detector.name == "TRIDENT"
    assert detector.selection_level == "6 deg cut"
    assert detector.priors["reach_km"] == site.reach_prior
    assert detector.priors["b_scale"] == sm.PRIORS["ARCA230"]["b_scale"]
    assert int(detector.mask.sum()) == 10
    assert detector.observed == pytest.approx(sm.trident_allsky_cm2(), nan_ok=True)
    assert detector.chain is None and detector.best is None
    assert not hasattr(detector, "color")


@pytest.mark.slow
@pytest.mark.skipif(not (DATA_DIR / "irfs").exists(), reason="the DR2 release is not on disk")
def test_build_four_detectors():
    detectors = sm.build_four_detectors(DATA_DIR)
    assert [d.name for d in detectors] == ["IceCube", "ARCA230", "P-ONE", "TRIDENT"]
    assert [d.selection_level for d in detectors] == [
        "analysis",
        "trigger",
        "trigger",
        "6 deg cut",
    ]
    assert [int(d.mask.sum()) for d in detectors] == [14, 13, 10, 10]
    icecube = detectors[0]
    assert icecube.observed[[10, 15, 20]] == pytest.approx(
        [822463.4724004329, 2809662.8374168244, 5936223.078126195], rel=1e-10
    )
    assert icecube.predict(THETA, None)[[10, 15, 20]] == pytest.approx(
        [770702.0721737217, 2729630.7316403734, 5829577.117627461], rel=QUADRATURE_RTOL
    )
