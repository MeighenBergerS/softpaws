"""Tests for the reduced response of :mod:`softpaws.response.reduced`.

The pinned numbers come from the results the pre-cleanup examples 77, 78, 81
and 82 wrote to ``examples/output``: the deviances of
``77_reduced_fit_sigma05.json``, the cell fits of
``81_trident_2025_map_fit.json``, and the chains behind them. Every test that
reads one of those caches skips when it is not on disk.
"""

import json
import pathlib

import numpy as np
import pytest

from softpaws.response import reduced as rd
from softpaws.response import site_models as sm

DATA_DIR = pathlib.Path(__file__).parents[1] / "src" / "softpaws" / "data" / "dataverse_files"
PAPER_SCRIPTS = pathlib.Path(__file__).parents[1] / "scripts" / "2026_muon_transport"
CACHE = PAPER_SCRIPTS / "output"

needs_dr2 = pytest.mark.skipif(
    not (DATA_DIR / "irfs").exists(), reason="the DR2 release is not on disk"
)


def _cached(name: str) -> pathlib.Path:
    path = CACHE / name
    if not path.exists():
        pytest.skip(f"{name} is not in the paper scripts' output")
    return path


# ---------------------------------------------------------------------------
# Constants and the parameter packing
# ---------------------------------------------------------------------------


def test_pinned_constants():
    assert rd.REDUCED_SITES == ("IceCube", "ARCA230", "P-ONE", "TRIDENT")
    assert rd.EPS_FIXED == {"IceCube": 0.956, "ARCA230": 1.0, "P-ONE": 1.0, "TRIDENT": 1.0}
    assert rd.REACH_DERIVED_KM == {
        "IceCube": 0.059,
        "ARCA230": 0.068,
        "P-ONE": 0.068,
        "TRIDENT": 0.068,
    }
    assert rd.PREDICTED_REACH_M == {
        "IceCube": (30.0, 59.0),
        "ARCA230": (41.0, 50.0),
        "P-ONE": (24.0, 30.0),
        "TRIDENT": (15.0, 27.0),
    }
    assert rd.FREE2 == ("log10_e_thr", "reach_km")
    assert rd.FREE3 == ("eps_0", "log10_e_thr", "reach_km")
    assert set(rd.WALKER_SCATTER) == set(rd.FREE3)
    assert rd.PARAM_BOUNDS == {
        "eps_0": (0.05, 1.5),
        "log10_e_thr": (1.5, 4.5),
        "reach_km": (-0.08, 0.40),
    }
    assert rd.MAP_RADIUS_KM == 1.75
    assert rd.COS_EDGES.size == 13
    assert rd.COS_EDGES[0] == 1.0 and rd.COS_EDGES[-1] == -1.0


def test_full_theta_overwrites_only_the_free_parameters():
    fixed = {"eps_0": 0.9, "log10_e_thr": 3.0, "b_scale": 1.0, "lam": 0.4538, "reach_km": 0.03}
    theta = rd.full_theta(rd.FREE2, np.array([3.5, 0.01]), fixed)
    assert theta == pytest.approx([0.9, 3.5, 1.0, 0.4538, 0.01])
    theta = rd.full_theta(rd.FREE3, np.array([0.5, 2.5, -0.01]), fixed)
    assert theta == pytest.approx([0.5, 2.5, 1.0, 0.4538, -0.01])
    # The order of the five is the site models', whatever order the free ones
    # are given in.
    assert rd.full_theta(("lam",), np.array([0.6]), fixed)[
        sm.PARAM_NAMES.index("lam")
    ] == pytest.approx(0.6)


def test_deviance_rejects_a_model_that_returns_nothing_positive():
    detector = sm.Detector(
        name="stub",
        log10_e=np.array([5.0, 6.0]),
        observed=np.array([1.0, 2.0]),
        mask=np.ones(2, bool),
        predict=lambda theta, select=None: np.array([0.0, 1.0]),
        priors={},
        start=np.zeros(5),
        selection_level="stub",
    )
    value, residual = rd.deviance(detector, np.zeros(5), 0.05)
    assert value == np.inf and residual is None


def test_deviance_is_the_log_residual_chi_square():
    detector = sm.Detector(
        name="stub",
        log10_e=np.array([5.0, 6.0]),
        observed=np.array([np.e, 1.0]),
        mask=np.array([True, False]),
        predict=lambda theta, select=None: np.array([1.0]),
        priors={},
        start=np.zeros(5),
        selection_level="stub",
    )
    value, residual = rd.deviance(detector, np.zeros(5), 0.5)
    assert residual == pytest.approx([1.0])
    assert value == pytest.approx(4.0)


def test_log_probability_rejects_outside_the_prior_box():
    detector = sm.Detector(
        name="stub",
        log10_e=np.array([5.0]),
        observed=np.array([1.0]),
        mask=np.ones(1, bool),
        predict=lambda theta, select=None: np.array([1.0]),
        priors={"log10_e_thr": (2.0, 5.0), "reach_km": (-0.02, 0.10)},
        start=np.zeros(5),
        selection_level="stub",
    )
    fixed = dict.fromkeys(sm.PARAM_NAMES, 1.0)
    log_probability = rd.reduced_log_probability(detector, rd.FREE2, fixed, 0.05)
    assert log_probability(np.array([3.0, 0.01])) == pytest.approx(0.0)
    assert log_probability(np.array([9.0, 0.01])) == -np.inf
    assert log_probability(np.array([3.0, 0.5])) == -np.inf


# ---------------------------------------------------------------------------
# The chains the reduced fit wrote
# ---------------------------------------------------------------------------


def test_attach_reduced_chains(tmp_path):
    rows = np.array([[3.5, 0.007], [3.6, 0.008]])
    np.savez(tmp_path / "chains.npz", **{f"{s}_2p_chain": rows for s in rd.REDUCED_SITES})
    detectors = [
        sm.Detector(
            name=name,
            log10_e=np.zeros(1),
            observed=np.zeros(1),
            mask=np.ones(1, bool),
            predict=lambda theta, select=None: np.zeros(1),
            priors={},
            start=np.zeros(5),
            selection_level="stub",
        )
        for name in rd.REDUCED_SITES
    ]
    rd.attach_reduced_chains(detectors, tmp_path / "chains.npz")
    for detector in detectors:
        assert detector.chain.shape == (2, 5)
        assert detector.chain[0, 0] == rd.EPS_FIXED[detector.name]
        assert detector.chain[:, 2] == pytest.approx(1.0)
        assert detector.chain[:, 3] == pytest.approx(sm.LAMBDA_BGR18)
        assert detector.chain[:, 1] == pytest.approx(rows[:, 0])
        assert detector.chain[:, 4] == pytest.approx(rows[:, 1])


def test_chain_summary_reproduces_the_published_plane():
    """Example 78's table, from the 5%-per-node chains of example 77."""
    chains, sigma = rd.load_two_parameter_chains(_cached("77_chains_sigma05.npz"))
    assert sigma == pytest.approx(0.05)
    assert set(chains) == set(rd.REDUCED_SITES)
    summary = rd.reduced_chain_summary(chains)
    assert summary["IceCube"]["e_thr_gev"] == pytest.approx(
        [3329.5128132795044, 3550.256766011953, 3776.666407182039], rel=1e-10
    )
    assert summary["IceCube"]["reach_m"] == pytest.approx(
        [3.3806849596194293, 7.240200955019245, 10.911631873521472], rel=1e-10
    )
    assert summary["ARCA230"]["reach_m"][1] == pytest.approx(44.54713141446361, rel=1e-10)
    assert summary["P-ONE"]["reach_m"][1] == pytest.approx(10.694071845818158, rel=1e-10)
    assert summary["TRIDENT"]["reach_m"][1] == pytest.approx(-37.38409635362696, rel=1e-10)
    separations = rd.reach_separation_sigma(chains)
    assert [(a, b) for a, b, _ in separations] == [
        ("IceCube", "ARCA230"),
        ("IceCube", "P-ONE"),
        ("IceCube", "TRIDENT"),
        ("ARCA230", "P-ONE"),
        ("ARCA230", "TRIDENT"),
        ("P-ONE", "TRIDENT"),
    ]
    assert [value for _, _, value in separations] == pytest.approx(
        [
            7.491523773955288,
            0.8636378055945046,
            3.892515822533133,
            9.599938310988467,
            7.245505721615148,
            4.4061645772079645,
        ],
        rel=1e-10,
    )


# ---------------------------------------------------------------------------
# TRIDENT's 2025 map
# ---------------------------------------------------------------------------


def test_map_cells_are_read_in_cm2():
    cos_theta, log10_e, log10_a = rd.trident_2025_cells()
    assert cos_theta.shape == (12,) and log10_e.shape == (12,) and log10_a.shape == (12, 12)
    assert cos_theta[0] == pytest.approx(0.917) and cos_theta[-1] == pytest.approx(-0.917)
    assert log10_e[0] == pytest.approx(3.125) and log10_e[-1] == pytest.approx(5.875)
    # The release is log10 of m^2; the loader adds the four decades.
    assert log10_a[0, 0] == pytest.approx(4.0, abs=1e-12)
    assert log10_a[-1, -1] == pytest.approx(5.7379999999999995, abs=1e-12)


def test_map_model_covers_every_cos_bin():
    _, log10_e, _ = rd.trident_2025_cells()
    model = rd.MapModel(log10_e)
    assert model.site.name == "TRIDENT"
    assert model.site.radius_km == rd.MAP_RADIUS_KM
    assert len(model.weights) == 12
    # The bins partition the whole-sky zenith grid.
    assert sum(w.sum() for w in model.weights) == pytest.approx(model.zenith_weights.sum())
    out = model(np.array([0.7, 2.5, 1.0, sm.LAMBDA_BGR18, 0.03]))
    assert out.shape == (12, log10_e.size)


def test_trident_2025_detector_flattens_the_selected_cells():
    detector, n_cells = rd.trident_2025_detector(0.5, 5.0)
    assert n_cells == 24
    assert detector.name == "TRIDENT"
    assert detector.selection_level == "2025 map"
    assert detector.observed.size == 24
    assert detector.mask.all()
    assert detector.priors["eps_0"] == rd.PARAM_BOUNDS["eps_0"]
    assert detector.priors["reach_km"] == rd.PARAM_BOUNDS["reach_km"]
    assert detector.observed[[0, 12, 23]] == pytest.approx(
        [3435579.478998743, 4753352.2594280485, 8770008.21143634], rel=1e-10
    )
    predicted = detector.predict(np.array([0.7, 2.5, 1.0, sm.LAMBDA_BGR18, 0.03]), None)
    assert predicted[[0, 12, 23]] == pytest.approx(
        [4142363.638305092, 3656752.3715711785, 8888682.351679329], rel=1e-10
    )


def test_trident_2025_average_detector():
    detector, allsky, smoothing = rd.trident_2025_average_detector(
        _cached("82_trident2025_chain.npz"), 0.5
    )
    assert detector.log10_e.size == allsky.size == 8
    assert smoothing["rms"] == pytest.approx(0.030198709942562373, rel=1e-9)
    assert smoothing["max"] == pytest.approx(0.054783788766439155, rel=1e-9)
    assert detector.observed[0] == pytest.approx(567264.5861631881, rel=1e-10)
    assert detector.observed[-1] == pytest.approx(11788285.939314619, rel=1e-10)
    # The nadir cells cost the all-sky average about 20%.
    assert allsky[0] == pytest.approx(473878.1637877712, rel=1e-10)
    assert allsky[-1] == pytest.approx(9012996.755291779, rel=1e-10)
    assert detector.chain.shape[1] == 5
    assert detector.chain[:, 2] == pytest.approx(1.0)
    assert detector.chain[:, 3] == pytest.approx(sm.LAMBDA_BGR18)


@pytest.mark.slow
def test_fit_cells_reproduces_the_published_cell_fit():
    """Example 81's ``|cos| <= 0.5`` row, refitted from scratch."""
    stored = json.loads(_cached("81_trident_2025_map_fit.json").read_text())["|cos| <= 0.5"]
    cos_theta, log10_e_all, log10_a_all = rd.trident_2025_cells()
    keep = log10_e_all >= sm.ARCA_LOG10_E.min()
    log10_e, log10_a = log10_e_all[keep], log10_a_all[:, keep]
    model = rd.MapModel(log10_e)
    cells = np.broadcast_to((np.abs(cos_theta) <= 0.5)[:, None], log10_a.shape)
    best, chi2, (low, high), residual = rd.fit_cells(model, log10_a, cells, 0.043)
    assert best[0] == pytest.approx(stored["eps_0"], rel=1e-10)
    assert 10.0 ** best[1] == pytest.approx(stored["e_thr_gev"], rel=1e-10)
    assert 1.0e3 * best[2] == pytest.approx(stored["reach_m"], rel=1e-10)
    assert chi2 == pytest.approx(stored["chi2"], rel=1e-10)
    assert 1.0e3 * low == pytest.approx(stored["reach_68_m"][0], rel=1e-9)
    assert 1.0e3 * high == pytest.approx(stored["reach_68_m"][1], rel=1e-9)
    assert residual.shape == log10_a.shape


# ---------------------------------------------------------------------------
# The stored fit, node by node
# ---------------------------------------------------------------------------


@pytest.mark.slow
@needs_dr2
def test_deviance_matches_every_block_of_the_stored_fit():
    """The 5%-per-node fit of example 77, at the parameter vectors it stored."""
    stored = json.loads(_cached("77_reduced_fit_sigma05.json").read_text())
    detectors = {d.name: d for d in sm.build_four_detectors(DATA_DIR)}
    assert set(stored) == set(detectors)
    for name, entry in stored.items():
        detector = detectors[name]
        assert int(detector.mask.sum()) == entry["nodes"]
        for block in ("five_param", "2-param (E_thr, Lambda)", "3-param (+eps_0)"):
            value, residual = rd.deviance(detector, np.array(entry[block]["best"]), 0.05)
            assert value == pytest.approx(entry[block]["deviance"], rel=1e-10)
            assert residual.size == entry["nodes"]
