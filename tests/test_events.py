"""Tests for the DR2 event benchmark and its reconstructed-energy likelihood.

The pinned likelihood numbers were computed once with the pre-cleanup
``examples/51_dr2_flavor_fit.py`` (tag ``pre-cleanup``) on the cached fit
inputs, so they check the lift and not the physics. The benchmark totals come
from the ``event_benchmark_ic86`` block of ``baseline.json``.

The tests that fold a response need the DR2 release, example 51's cached fit
inputs and example 22's MCEq table, and skip when any of the three is
missing.
"""

import json
import pathlib

import numpy as np
import pytest

from softpaws.comparison import events, reco_likelihood
from softpaws.data.icecube import IC86_SEASONS, total_livetime_s

REPO = pathlib.Path(__file__).parents[1]
DATA_DIR = REPO / "src" / "softpaws" / "data" / "dataverse_files"
FIT_INPUTS = REPO / "scripts" / "2026_muon_transport" / "output" / "51_fit_inputs.npz"
MCEQ_TABLE = REPO / "scripts" / "2026_muon_transport" / "output" / "22_mceq_atmospheric_flux.npz"
BASELINE = json.loads((REPO / "tests" / "regression" / "baseline.json").read_text())[
    "event_benchmark_ic86"
]

#: Livetime of the eleven IC86 seasons [s].
LIVETIME_S = 338741712.981806

#: Sums the pre-cleanup script gives on the fit window, all counts.
PINS = {
    "data": 526.0,
    "folded_conv": 423.02723979862793,
    "folded_prompt": 18.97758348799397,
    "astro_mu": 286.74696886473305,
    "astro_tau": 19.892760131712535,
    "yield_ratio": 0.06076699138363693,
}

#: Expected counts at ``(norm, gamma, a_conv, a_prompt) = (0.9, 2.5, 1, 1)``:
#: the window total per flavour ratio, and the loudest bin at ``r = 0``.
EXPECTATION_SUM = {0.0: 672.4026635570693, 0.25: 618.3610268187542,
                   0.5: 564.3193900804392, 0.9: 477.8527712991353}
EXPECTATION_MAX_R0 = 128.28685820251877

#: Profiled objective and its nuisances at three flavour ratios.
DELTA_LL = {0.0: 86.85689428107993, 0.3: 86.9204242660956, 0.6: 87.14371632159987}
DELTA_LL_PARAMS_R0 = (0.36982794110739875, 2.445033733839422,
                      0.9903458823691711, 1.0273045727761936)


# ---------------------------------------------------------------------------
# Constants and the pieces that need no data
# ---------------------------------------------------------------------------


def test_anchors_come_from_the_published_fits():
    """The pivot is the combined fit and the anchors the 9.5-year tracks fit."""
    assert reco_likelihood.PIVOT_PHI0 == pytest.approx(1.80e-18)
    assert reco_likelihood.PIVOT_GAMMA == pytest.approx(2.52)
    assert reco_likelihood.TRACKS_GAMMA == pytest.approx((2.37, 0.09))
    assert reco_likelihood.TRACKS_PHI_MU[0] == pytest.approx(1.44e-18)
    assert reco_likelihood.TRACKS_PHI_MU[1] == pytest.approx(
        float(np.hypot(0.26e-18, 0.05 * 1.44e-18))
    )
    anchors = reco_likelihood.ANCHORS
    assert anchors["gamma"] == reco_likelihood.TRACKS_GAMMA
    assert anchors["phi_mu"][0] == pytest.approx(1.44 / 1.80)
    assert events.ASTRO_PHI == reco_likelihood.TRACKS_PHI_MU[0]
    assert events.ASTRO_GAMMA == reco_likelihood.TRACKS_GAMMA[0]


def test_fit_window_holds_thirteen_bins():
    """The window runs from 10^4.25 to 10^7.5 GeV in quarter-decade bins."""
    window = ((reco_likelihood.RECO_EDGES[:-1] >= reco_likelihood.FIT_RECO[0] - 1.0e-9)
              & (reco_likelihood.RECO_EDGES[1:] <= reco_likelihood.FIT_RECO[1] + 1.0e-9))
    assert window.sum() == 13


def test_physical_r_range_is_extremal_at_a_vertex():
    """A linear-fractional ratio is extremal at a vertex of the simplex."""
    def earth_composition(source):
        f_e, f_mu, f_tau = source
        return (0.6 * f_e + 0.2 * f_mu + 0.2 * f_tau,
                0.2 * f_e + 0.4 * f_mu + 0.2 * f_tau,
                0.2 * f_e + 0.4 * f_mu + 0.6 * f_tau)

    r_min, r_max = reco_likelihood.physical_r_range(earth_composition)
    assert r_min == pytest.approx(0.5)
    assert r_max == pytest.approx(0.75)


def test_combine_band_adds_linearly_over_bins():
    """A normalization error keeps its fraction however many bins are summed."""
    grid = np.array([[1.0, 2.0], [3.0, 4.0]])
    errors = {"conv": 0.25 * grid, "astro_gamma": (-0.1 * grid, 0.1 * grid)}
    total = events.combine_band(errors, lambda g: g.sum())
    assert total == pytest.approx(np.hypot(0.25 * 10.0, 0.1 * 10.0))
    per_row = events.combine_band(errors, lambda g: g.sum(axis=1))
    assert per_row == pytest.approx([np.hypot(0.25 * 3.0, 0.1 * 3.0),
                                     np.hypot(0.25 * 7.0, 0.1 * 7.0)])


# ---------------------------------------------------------------------------
# The likelihood on the cached fit inputs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fit_inputs():
    """Cached smearing marginal and banded responses of example 51."""
    for path, what in ((DATA_DIR / "irfs", "the DR2 release"),
                       (FIT_INPUTS, "example 51's cached fit inputs"),
                       (MCEQ_TABLE, "example 22's MCEq table")):
        if not path.exists():
            pytest.skip(f"{what} is not on disk")
    cached = reco_likelihood.cached_fit_inputs(FIT_INPUTS)
    assert cached is not None
    return cached


@pytest.fixture(scope="module")
def likelihood(fit_inputs):
    """The anchored likelihood on the observed IC86 counts."""
    enu_edges, dec_edges, marginal, responses = fit_inputs
    atmos = reco_likelihood.atmospheric_fluxes(MCEQ_TABLE, dec_edges)
    counts, _ = reco_likelihood.binned_events(DATA_DIR, dec_edges)
    return reco_likelihood.RecoLikelihood(
        enu_edges, marginal, responses, atmos, dec_edges, LIVETIME_S, counts,
        anchors=reco_likelihood.ANCHORS)


def test_livetime_is_the_ic86_exposure():
    """The pinned livetime is the release's own uptime files."""
    if not (DATA_DIR / "uptime").exists():
        pytest.skip("the DR2 release is not on disk")
    assert total_livetime_s(DATA_DIR, IC86_SEASONS) == pytest.approx(LIVETIME_S)


def test_binned_events_fill_the_window(likelihood):
    """The upgoing IC86 sample puts 526 events inside the fit window."""
    assert likelihood.data.sum() == pytest.approx(PINS["data"])
    assert likelihood.data.shape == (13, 10)


def test_folded_components_match_the_pre_cleanup_script(likelihood):
    """Background and signal folds, against the script the lift replaced."""
    assert likelihood._folded_conv.sum() == pytest.approx(PINS["folded_conv"], rel=1e-10)
    assert likelihood._folded_prompt.sum() == pytest.approx(PINS["folded_prompt"], rel=1e-10)
    assert likelihood._astro("mu", 2.37).sum() == pytest.approx(PINS["astro_mu"], rel=1e-10)
    assert likelihood._astro("tau", 2.37).sum() == pytest.approx(PINS["astro_tau"], rel=1e-10)
    assert likelihood.channel_yield_ratio(reco_likelihood.PIVOT_GAMMA) == pytest.approx(
        PINS["yield_ratio"], rel=1e-10)


def test_expectation_matches_the_pre_cleanup_script(likelihood):
    """The expectation is unchanged at four flavour ratios."""
    for r, total in EXPECTATION_SUM.items():
        expectation = likelihood.expectation(r, 0.9, 2.5, 1.0, 1.0)
        assert expectation.sum() == pytest.approx(total, rel=1e-10)
    assert likelihood.expectation(0.0, 0.9, 2.5, 1.0, 1.0).max() == pytest.approx(
        EXPECTATION_MAX_R0, rel=1e-10)


def test_delta_ll_matches_the_pre_cleanup_script(likelihood):
    """The profiled objective and its nuisances are unchanged."""
    for r, value in DELTA_LL.items():
        profiled, params = likelihood.delta_ll(r)
        assert profiled == pytest.approx(value, rel=1e-10)
        if r == 0.0:
            assert params == pytest.approx(DELTA_LL_PARAMS_R0, rel=1e-10)


# ---------------------------------------------------------------------------
# The benchmark itself
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_event_benchmark_reproduces_the_paper(fit_inputs):
    """Figure 5: 564.5 +- 109 predicted tracks against 526 observed."""
    enu_edges, dec_edges, marginal, responses = fit_inputs
    atmos = reco_likelihood.atmospheric_fluxes(MCEQ_TABLE, dec_edges)
    data, _ = reco_likelihood.binned_events(DATA_DIR, dec_edges)
    window = ((reco_likelihood.RECO_EDGES[:-1] >= reco_likelihood.FIT_RECO[0] - 1.0e-9)
              & (reco_likelihood.RECO_EDGES[1:] <= reco_likelihood.FIT_RECO[1] + 1.0e-9))

    components, errors = events.predict(responses, atmos, enu_edges, dec_edges,
                                        marginal, LIVETIME_S)
    total = sum(components.values())[window].sum()
    band = events.combine_band(errors, lambda g: g[window].sum())
    rtol = BASELINE["rtol"]
    assert total == pytest.approx(BASELINE["model_total"], rel=rtol)
    assert band == pytest.approx(BASELINE["model_total_error"], rel=rtol)
    assert data[window].sum() == BASELINE["data_total"]
    assert data[window].sum() / total == pytest.approx(BASELINE["data_over_model"], rel=rtol)
    assert components["prompt"][window].sum() == pytest.approx(BASELINE["prompt"], rel=rtol)
    assert components["astro_mu"][window].sum() == pytest.approx(BASELINE["astro_mu"], rel=rtol)
    astro = components["astro_mu"] + components["astro_tau"]
    share = components["astro_tau"][window].sum() / astro[window].sum()
    assert share == pytest.approx(BASELINE["tau_to_mu_share_astro_tracks"], rel=rtol)

    baseline, _ = events.predict(events.published_response(DATA_DIR, enu_edges, dec_edges),
                                 atmos, enu_edges, dec_edges, marginal, LIVETIME_S)
    irf_total = sum(baseline.values())[window].sum()
    assert irf_total == pytest.approx(BASELINE["published_irf_total"], rel=rtol)
    assert total / irf_total == pytest.approx(BASELINE["model_over_irf"], rel=rtol)
    assert data[window].sum() / irf_total == pytest.approx(BASELINE["data_over_irf"], rel=rtol)
