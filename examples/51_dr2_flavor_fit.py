"""Example 51 -- fitting the flavour ratio to the actual DR2 events.

Example 50 forecasts what through-going tracks can say about the flavour
composition; this example asks the data. The model's true-energy response
(the fitted configuration of example 45, ``nu_mu`` and ``nu_tau -> tau -> mu``
channels, banded in the smearing table's own declination bins) is folded
through IceCube's released smearing matrices into *reconstructed* muon-energy
space, and the same profile likelihood is run against the binned IC86 events
of IceTracks-DR2. The free parameters are the astrophysical normalization and
spectral index, the conventional and prompt atmospheric normalizations, and
the flavour ratio

.. math:: r = \\frac{f_\\tau}{f_\\mu + f_\\tau},

scanned with everything else profiled. The output is a measured ``r`` -- the
first number in this project fit to events instead of to a published
response -- drawn both as a profile curve and on the flavour triangle of
example 50, where the 68% interval is a wedge anchored at the ``nu_e``
vertex and the best fit a ray to the ``mu``-``tau`` edge.

The best fit can sit on a boundary of the scan, where the Wilks chi-square
calibration of the profile fails, so the quoted intervals come from a
Feldman-Cousins construction instead: at each truth ratio, Poisson
pseudo-experiments are drawn from the data's profiled expectation, the same
profile statistic is computed for each, and its 68th and 95th percentiles
replace the flat Wilks levels. The toys keep the prior centres fixed and
re-profile every nuisance, so whatever the scaled deviance and the priors do
to the statistic's distribution is calibrated away rather than assumed. Both
interval flavours are printed; the figures carry the calibrated one.

Two things make the fit well posed. First, IceCube's 9.5-year tracks fit
supplies external anchors used throughout: its index (common to all
flavours) and its ``nu_mu`` normalization, which already separates the
``tau -> mu`` tracks under a 1:1:1 assumption -- their own test puts that
assumption at 5% on the normalization and nothing on the index, and the 5%
is added to the prior width. Without the anchors the astrophysical total is
degenerate with the conventional normalization (a +-25% prior on ~450
background events is more freedom than the ~77-event excess, and the
sub-window data that would pin the background are where the model's turn-on
fails), and with one shared normalization and a free ratio the fit relabels
the whole excess as tau at several times the measured flux; the anchors
block both, which is what lets the ratio scan run over its full range.
Second, the standard-oscillation band of the Earth ratio, ``r`` in
[0.47, 0.53] for every source composition, is drawn on every figure. The
headline is the tau flux profiled at the anchored ``nu_mu`` flux
(figure 51f); the ratio scan, the triangle and the two planes (51a, 51c,
51d, 51e) are the supporting views, and a tail jackknife reruns the profile
with each Poisson-limited high-energy event removed in turn. Figure 51g
adds this work to IceCube's MESE flavour measurement: the MESE likelihood
is rebuilt from its two published contours (a radial power law through
both at every angle, quadratic where the 95% contour is clipped), this
work's ratio profile is added as a function constant along rays from the
``nu_e`` vertex, and the two samples are taken as independent. The same
figure carries the forecast for IceCube-Gen2 tracks: example 50's Gen2
geometry through this example's fold and anchors, a ten-year Asimov at
1:1:1, with IC86's smearing standing in for Gen2's.

Scope, stated plainly. IC86 seasons only, since the early configurations have
different geometries the model does not carry. Reconstructed energies are
fitted between 10^4.25 and 10^7.5 GeV under a scaled-deviance likelihood: a
10% fractional model systematic per bin keeps the tens-of-thousands-strong
atmospheric bins from dominating through percent-level shape residuals of the
folded model, while the Poisson-limited tail is untouched. Gaussian priors on
the atmospheric normalizations (conventional 1.0 +- 0.25, prompt 1.0 +- 0.25)
break the prompt-astro degeneracy the way the collaboration analyses do, and
the astrophysical index is bounded to (1.5, 4). Without these three
regularizations the fit walks into corners: the astro component becomes a
shape patch for the window edge, or the prompt normalization absorbs the
astrophysical flux entirely. The smearing fold uses
the energy marginal of the released 5D tables (point-spread and angular-error
axes summed), projected onto a common reconstructed-energy grid assuming a
uniform density inside each released interval. The released table is
``nu_mu`` CC simulation, so the tau channel folds through a decay-shifted
copy: a tau-chain muon carries only the ``tau -> mu`` decay fraction
(mean 0.35) of the energy a ``nu_mu`` muon would, so each tau bin takes the
reco distribution of the ``nu_mu`` bin at ``x E_nu``, averaged over the
decay spectrum. Without this shift the tau channel reconstructs ~0.5 dex
too hard and turns into a spectral-hardening dial degenerate with the
index, which a single PeV event can then drive to ``r = 1``.
The absolute selection layer
is example 45's fitted normalization, though the profile's free normalizations
absorb most of it. No systematic uncertainties enter beyond the four profiled
parameters, so the interval is statistical plus flux-model freedom -- a mini
analysis, not a collaboration-grade measurement.

Usage
-----
    python examples/51_dr2_flavor_fit.py
    python examples/51_dr2_flavor_fit.py --rebuild-cache
"""

import argparse
import importlib.util
import pathlib
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from softpaws.data.loader import compute_livetime_s, load_season, load_uptime
from softpaws.data.schema import SEASONS

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"
_DEFAULT_DATA_DIR = _HERE.parent / "src" / "softpaws" / "data" / "dataverse_files"

#: Cached banded responses and smearing marginal, keyed to this example.
_CACHE = _DEFAULT_OUT_DIR / "51_fit_inputs.npz"

#: Cached IceCube-Gen2 banded responses for the forecast of figure 51g.
_GEN2_CACHE = _DEFAULT_OUT_DIR / "51_gen2_responses.npz"

#: Atmospheric flux cache from example 22.
_MCEQ_CACHE = _HERE / "output" / "22_mceq_atmospheric_flux.npz"

#: Seasons entering the fit: the IC86 configurations share one IRF set.
IC86_SEASONS = tuple(s for s in SEASONS if s.startswith("IC86"))

#: Astrophysical pivot: the per-flavour normalization unit and the index seed,
#: from the combined fit (arXiv:2308.00191). ``N = 1`` in the fit means this
#: flux.
PIVOT_PHI0 = 1.80e-18
PIVOT_GAMMA = 2.52

#: Reconstructed-energy grid the fold projects onto, and the window the fit
#: uses. The grid is wider than the window so the fold conserves counts.
RECO_EDGES = np.arange(1.0, 8.51, 0.25)
FIT_RECO = (4.25, 7.5)

#: Fractional model-shape systematic per bin. The percent-level residuals of
#: the folded model would otherwise dominate the likelihood through the
#: highest-statistics atmospheric bins; each bin's deviance is scaled by
#: ``1 + (MODEL_SYS^2) mu``, which de-weights exactly those bins and leaves
#: the Poisson-limited tail untouched.
MODEL_SYS = 0.10

#: Gaussian priors on the atmospheric normalizations, the standard breakers
#: of the prompt-astro degeneracy: hadronic-model spread on both. An earlier
#: order-one prompt prior let the prompt normalization run to 1.6 and swallow
#: the astrophysical ``nu_mu`` (its norm fell to 0.5); at 0.25, the
#: perturbative-QCD spread, the astrophysical normalization comes back to
#: 0.9 x the combined fit.
CONV_PRIOR = (1.0, 0.25)
PROMPT_PRIOR = (1.0, 0.25)

#: Physical bounds on the astrophysical spectral index, without which the
#: astro component can degenerate into a shape patch for the atmospheric
#: window edge (a first fit ran to gamma = 15 with the normalization at
#: zero).
GAMMA_BOUNDS = (1.5, 4.0)

#: Fine true-energy grid of the model responses.
LOG10_E_GRID = np.linspace(3.0, 8.5, 111)

#: Spectral-index grid the astrophysical folds are cached on; the profile
#: interpolates between its nodes, which is what makes the pseudo-experiment
#: calibration affordable.
GAMMA_GRID = np.linspace(GAMMA_BOUNDS[0], GAMMA_BOUNDS[1], 51)

#: Cached toy distributions of the profile statistic.
_FC_CACHE = _DEFAULT_OUT_DIR / "51_fc_calibration.npz"

#: Reconstructed energy [log10 GeV] above which the tail jackknife removes
#: single events; the bins there hold at most one event each.
JACKKNIFE_LOG10_E = 5.5

#: Quadrature nodes on the muon energy fraction ``x`` of ``tau -> mu nu nu``
#: (unpolarized spectrum ``f(x) = 5/3 - 3x^2 + 4x^3/3``, mean 0.35), used to
#: shift the ``nu_mu``-simulation smearing onto the tau channel.
TAU_DECAY_X = np.linspace(0.025, 0.975, 20)

#: Cache-format version of the toy distributions; bump on any change to the
#: folded model so stale calibrations rebuild.
_FC_VERSION = 5

#: Example 45's fitted configuration.
FITTED_ATTENUATION_M = 42.0
FITTED_NORMALIZATION = 0.755


def load_example(stem: str, name: str):
    """Import a sibling example module by file stem."""
    spec = importlib.util.spec_from_file_location(name, _HERE / stem)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def physical_r_range(ex50) -> tuple[float, float]:
    """Earth flavour-ratio range standard oscillations allow.

    The Earth fractions are linear in the source composition, so the
    linear-fractional ratio ``f_tau / (f_mu + f_tau)`` is extremal at a
    vertex of the source simplex; the three pure-flavour sources are enough.

    Returns
    -------
    r_min, r_max : float
        Band of the ratio over all source compositions.
    """
    ratios = []
    for source in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
        _, f_mu, f_tau = ex50.earth_composition(source)
        ratios.append(f_tau / (f_mu + f_tau))
    return min(ratios), max(ratios)


#: Example 50, for the triangle frame and the oscillation matrix.
_EX50 = load_example("50_flavor_triangle.py", "_example_50")

#: Standard-oscillation band of the Earth ratio, drawn on every figure.
R_MIN_STD, R_MAX = physical_r_range(_EX50)

#: Flavour-ratio scan grid. The full range is safe only because the
#: ``nu_mu`` flux is anchored: without the anchor, tracks cannot tell the
#: two channel templates apart by shape and the fit relabels the whole
#: astrophysical excess as tau at several times the measured flux for free.
R_GRID = np.linspace(0.0, 1.0, 41)

#: Feldman-Cousins calibration: truth points the toy distributions are built
#: at, and the ratio grid each toy's global minimum is scanned on. Truths
#: stop at 0.9: they carry the anchored ``nu_mu`` flux, so the tau flux
#: diverges at ``r = 1``; the last threshold is held beyond.
FC_R_TRUE = np.linspace(0.0, 0.9, 10)
FC_R_SCAN = np.linspace(0.0, 1.0, 11)

#: Flux plane of figure 51d [combined-fit per-flavour flux units]. The scan
#: reaches the unconstrained minimum; the figure shows the physical range and
#: points at the rest.
PHI_MU_GRID = np.linspace(0.0, 1.6, 33)
PHI_TAU_GRID = np.linspace(0.0, 8.0, 33)
PHI_TAU_PLOT_MAX = 2.0

#: Count plane of figure 51e: expected astrophysical tracks in the fit window
#: from each channel, the shape profiled.
N_MU_GRID = np.linspace(0.0, 160.0, 33)
N_TAU_GRID = np.linspace(0.0, 100.0, 33)

#: IceCube's 9.5-year northern-tracks fit (arXiv:2111.10299): the index and
#: the ``nu_mu`` normalization [GeV^-1 cm^-2 s^-1 sr^-1 at 100 TeV], the
#: anchors of the whole analysis. Their fit includes ``tau -> mu`` at 1:1:1,
#: and their own with/without test (Aartsen et al. 2016) puts that
#: assumption at 5% on the normalization and nothing on the index; the 5%
#: is added to the normalization width in quadrature.
TRACKS_GAMMA = (2.37, 0.09)
TRACKS_PHI_MU = (1.44e-18, float(np.hypot(0.26e-18, 0.05 * 1.44e-18)))

#: Tau-flux scan at the anchored ``nu_mu`` flux [combined-fit units].
#: The anchors in combined-fit units, as the likelihood takes them.
PHI_TAU_SCAN = np.linspace(0.0, 6.0, 61)
ANCHORS = {"gamma": TRACKS_GAMMA,
           "phi_mu": (TRACKS_PHI_MU[0] / PIVOT_PHI0, TRACKS_PHI_MU[1] / PIVOT_PHI0)}


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=pathlib.Path, default=_DEFAULT_DATA_DIR,
                        help="Root of the DR2 release.")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Rebuild the cached responses and smearing marginal.")
    parser.add_argument("--toys", type=int, default=600,
                        help="Pseudo-experiments per truth point of the "
                             "Feldman-Cousins calibration.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Seed of the pseudo-experiment generator.")
    parser.add_argument("--rebuild-fc", action="store_true",
                        help="Rebuild the cached toy distributions.")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the figures, '51a' through '51g'.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Inputs: smearing marginal, banded responses, events
# ---------------------------------------------------------------------------


def smearing_marginal(data_dir: pathlib.Path):
    """Energy marginal of the IC86 smearing table on the common reco grid.

    Sums the released fractional counts over the point-spread and
    angular-error axes and projects each reconstructed-energy interval onto
    :data:`RECO_EDGES`, uniform density assumed inside an interval.

    Parameters
    ----------
    data_dir : pathlib.Path
        Root of the DR2 release.

    Returns
    -------
    enu_edges : np.ndarray, shape (n_enu + 1,)
        True-energy bin edges [log10 GeV].
    dec_edges : np.ndarray, shape (n_dec + 1,)
        Upgoing declination bin edges [deg].
    marginal : np.ndarray, shape (n_enu, n_dec, n_reco)
        ``P(reco bin | true bin)``; sums over reco to one where the released
        intervals lie inside the grid.
    """
    raw = np.genfromtxt(data_dir / "irfs" / "IC86_smearing.csv", comments="#")
    raw = raw[raw[:, 10] > 0.0]
    enu_lows = np.unique(raw[:, 0])
    enu_edges = np.append(enu_lows, raw[:, 1].max())
    dec_lows_all = np.unique(raw[:, 2])
    upgoing = dec_lows_all >= -1.0e-9
    dec_lows = dec_lows_all[upgoing]
    keep = raw[:, 2] >= dec_lows[0] - 1.0e-9
    rows = raw[keep]
    dec_edges = np.append(dec_lows, rows[:, 3].max())

    i_enu = np.searchsorted(enu_lows, rows[:, 0] + 1.0e-9) - 1
    i_dec = np.searchsorted(dec_lows, rows[:, 2] + 1.0e-9) - 1
    lo, hi, weight = rows[:, 4], rows[:, 5], rows[:, 10]
    width = np.maximum(hi - lo, 1.0e-12)

    marginal = np.zeros((enu_lows.size, dec_lows.size, RECO_EDGES.size - 1))
    for k in range(RECO_EDGES.size - 1):
        overlap = np.clip(np.minimum(hi, RECO_EDGES[k + 1])
                          - np.maximum(lo, RECO_EDGES[k]), 0.0, None)
        np.add.at(marginal, (i_enu, i_dec, k), weight * overlap / width)
    return enu_edges, dec_edges, marginal


def banded_responses(ex35, ex45, ex46, dec_edges_deg, site=None):
    """Model channel responses on the smearing declination bands [cm^2].

    Parameters
    ----------
    ex35, ex45, ex46 : ModuleType
        Examples 35, 45 and 46.
    dec_edges_deg : np.ndarray
        Upgoing declination bin edges [deg].
    site : optional
        Example 35 geometry; IceCube when omitted.

    Returns
    -------
    responses : dict of str -> np.ndarray
        ``"mu"`` and ``"tau"`` on (:data:`LOG10_E_GRID`, bands).
    """
    site = ex35.build_sites()[0] if site is None else site
    site45 = replace(ex45.ICECUBE_SITE, attenuation_override_m=FITTED_ATTENUATION_M)
    sin_dec_edges = np.sin(np.deg2rad(dec_edges_deg))
    coarse = ex35.COMMON_LOG10_E
    out = {}
    for channel in ("mu", "tau"):
        print(f"  building banded response, {channel} channel ...")
        banded = ex46.model_banded(ex35, ex45, site, site45, sin_dec_edges, 8.0,
                                   (channel,), efficiency=FITTED_NORMALIZATION)
        fine = np.empty((LOG10_E_GRID.size, banded.shape[1]))
        with np.errstate(divide="ignore"):
            log_a = np.log10(np.clip(banded, 1.0e-30, None))
        for j in range(banded.shape[1]):
            fine[:, j] = 10.0 ** np.interp(LOG10_E_GRID, coarse, log_a[:, j],
                                           left=-30.0, right=log_a[-1, j])
        out[channel] = fine
    return out


def fit_inputs(ex35, ex45, ex46, data_dir: pathlib.Path, rebuild: bool):
    """Smearing marginal and banded responses, cached across runs."""
    if _CACHE.exists() and not rebuild:
        cache = np.load(_CACHE)
        if np.array_equal(cache["reco_edges"], RECO_EDGES):
            print(f"  cached fit inputs from {_CACHE.name}")
            return (cache["enu_edges"], cache["dec_edges"], cache["marginal"],
                    {"mu": cache["response_mu"], "tau": cache["response_tau"]})
    print("  marginalizing the IC86 smearing table ...")
    enu_edges, dec_edges, marginal = smearing_marginal(data_dir)
    responses = banded_responses(ex35, ex45, ex46, dec_edges)
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(_CACHE, reco_edges=RECO_EDGES, enu_edges=enu_edges,
             dec_edges=dec_edges, marginal=marginal,
             response_mu=responses["mu"], response_tau=responses["tau"])
    return enu_edges, dec_edges, marginal, responses


def gen2_responses(ex35, ex45, ex46, dec_edges, rebuild: bool):
    """IceCube-Gen2 banded responses, example 50's geometry, cached."""
    if _GEN2_CACHE.exists() and not rebuild:
        cache = np.load(_GEN2_CACHE)
        if np.array_equal(cache["dec_edges"], dec_edges):
            print(f"  cached Gen2 responses from {_GEN2_CACHE.name}")
            return {"mu": cache["response_mu"], "tau": cache["response_tau"]}
    sites = {s.name: s for s in ex35.build_sites()}
    gen2 = replace(sites["IceCube"], name="IceCube-Gen2",
                   radius_km=_EX50.GEN2_RADIUS_KM, height_km=_EX50.GEN2_HEIGHT_KM)
    responses = banded_responses(ex35, ex45, ex46, dec_edges, site=gen2)
    np.savez(_GEN2_CACHE, dec_edges=dec_edges, response_mu=responses["mu"],
             response_tau=responses["tau"])
    return responses


def atmospheric_fluxes(dec_edges_deg):
    """Conventional and prompt fluxes on the fine grid per band.

    Example 22's MCEq table at the band-centre declinations
    [GeV^-1 cm^-2 s^-1 sr^-1].

    Raises
    ------
    FileNotFoundError
        Raised if example 22's cache is missing.
    """
    if not _MCEQ_CACHE.exists():
        raise FileNotFoundError(
            f"{_MCEQ_CACHE} not found; run example 22 once to tabulate the "
            "atmospheric flux.")
    atm = np.load(_MCEQ_CACHE)
    centers = 0.5 * (dec_edges_deg[:-1] + dec_edges_deg[1:])
    energy = 10.0**LOG10_E_GRID
    out = {}
    for component in ("conv", "prompt"):
        log_f = np.log10(np.clip(atm[component], 1.0e-99, None))
        on_dec = np.array([
            np.interp(centers, atm["dec_deg"], log_f[i])
            for i in range(atm["energy_gev"].size)
        ])
        grid = np.empty((energy.size, centers.size))
        for j in range(centers.size):
            grid[:, j] = 10.0 ** np.interp(np.log10(energy),
                                           np.log10(atm["energy_gev"]),
                                           on_dec[:, j])
        out[component] = grid
    return out


def binned_events(data_dir: pathlib.Path, dec_edges_deg):
    """IC86 upgoing events histogrammed on (reco grid, declination bands)."""
    counts = np.zeros((RECO_EDGES.size - 1, dec_edges_deg.size - 1))
    total = 0
    for season in IC86_SEASONS:
        events = load_season(data_dir / "events" / f"{season}_exp.csv")
        sel = events["dec"] >= dec_edges_deg[0]
        total += int(sel.sum())
        hist, _, _ = np.histogram2d(events["log10_energy"][sel],
                                    events["dec"][sel],
                                    bins=(RECO_EDGES, dec_edges_deg))
        counts += hist
    print(f"  {total:,} upgoing IC86 events, "
          f"{counts.sum():,.0f} inside the reco grid")
    return counts


# ---------------------------------------------------------------------------
# The likelihood in reconstructed space
# ---------------------------------------------------------------------------


class RecoLikelihood:
    """Poisson likelihood over (reco energy, declination band) bins.

    Attributes
    ----------
    data : np.ndarray
        Observed counts on the fit window.
    """

    def __init__(self, enu_edges, marginal, responses, atmos, dec_edges_deg,
                 livetime_s, data_counts, anchors=None):
        self._anchors = dict(anchors or {})
        self._enu_edges = enu_edges
        self._marginal = marginal
        self._responses = responses
        self._livetime_s = livetime_s
        self._d_omega = 2.0 * np.pi * np.diff(np.sin(np.deg2rad(dec_edges_deg)))
        self._window = ((RECO_EDGES[:-1] >= FIT_RECO[0] - 1.0e-9)
                        & (RECO_EDGES[1:] <= FIT_RECO[1] + 1.0e-9))
        self.data = data_counts[self._window]
        self._marginal_tau = self._shifted_marginal(enu_edges, marginal)
        self._folded_conv = self._fold(self._true_counts("mu", atmos["conv"]))
        self._folded_prompt = self._fold(self._true_counts("mu", atmos["prompt"]))
        self._astro_cache = {
            channel: np.stack([self._astro_direct(channel, gamma)
                               for gamma in GAMMA_GRID])
            for channel in ("mu", "tau")
        }

    def _true_counts(self, channel, flux):
        """Counts per (true bin, band) for one channel and flux grid."""
        energy = 10.0**LOG10_E_GRID
        integrand = self._responses[channel] * flux
        counts = np.zeros((self._enu_edges.size - 1, self._d_omega.size))
        for i in range(self._enu_edges.size - 1):
            sel = ((LOG10_E_GRID >= self._enu_edges[i])
                   & (LOG10_E_GRID <= self._enu_edges[i + 1]))
            if sel.sum() < 2:
                continue
            counts[i] = np.trapezoid(integrand[sel], energy[sel], axis=0)
        return self._livetime_s * counts * self._d_omega[None, :]

    @staticmethod
    def _shifted_marginal(enu_edges, marginal):
        """Tau-channel smearing: the ``nu_mu`` marginal at shifted energy.

        The released table is ``nu_mu`` CC simulation, whose muon carries the
        full ``(1 - y)`` share of the neutrino energy; in the tau chain the
        muon keeps only the decay fraction ``x`` of that share. A tau event
        at ``E_nu`` therefore reconstructs like a ``nu_mu`` event at
        ``x E_nu``, so each tau true-energy bin takes the reco distribution
        of the ``nu_mu`` bin containing ``x E_nu``, averaged over the decay
        spectrum on :data:`TAU_DECAY_X` and clipped at the table edge.
        """
        centers = 0.5 * (enu_edges[:-1] + enu_edges[1:])
        f = 5.0 / 3.0 - 3.0 * TAU_DECAY_X**2 + 4.0 * TAU_DECAY_X**3 / 3.0
        weights = f / f.sum()
        shifted = np.zeros_like(marginal)
        for i, center in enumerate(centers):
            for x, w in zip(TAU_DECAY_X, weights):
                j = int(np.clip(
                    np.searchsorted(enu_edges, center + np.log10(x)) - 1,
                    0, centers.size - 1))
                shifted[i] += w * marginal[j]
        return shifted

    def _fold(self, true_counts, channel="mu"):
        """True-space counts through the smearing marginal, window applied."""
        marginal = self._marginal_tau if channel == "tau" else self._marginal
        reco = np.einsum("ij,ijk->kj", true_counts, marginal)
        return reco[self._window]

    def _astro_direct(self, channel, gamma):
        energy = 10.0**LOG10_E_GRID
        flux = 2.0 * PIVOT_PHI0 * (energy[:, None] / 1.0e5) ** (-gamma)
        return self._fold(self._true_counts(
            channel, flux * np.ones((1, self._d_omega.size))), channel)

    def _astro(self, channel, gamma):
        """Folded astrophysical counts, interpolated on :data:`GAMMA_GRID`."""
        i = int(np.clip(np.searchsorted(GAMMA_GRID, gamma) - 1, 0,
                        GAMMA_GRID.size - 2))
        weight = ((gamma - GAMMA_GRID[i])
                  / (GAMMA_GRID[i + 1] - GAMMA_GRID[i]))
        cache = self._astro_cache[channel]
        return (1.0 - weight) * cache[i] + weight * cache[i + 1]

    def expectation(self, r, norm, gamma, a_conv, a_prompt):
        """Expected counts on the fit window at one parameter point."""
        astro = norm * ((1.0 - r) * self._astro("mu", gamma)
                        + r * self._astro("tau", gamma))
        return a_conv * self._folded_conv + a_prompt * self._folded_prompt + astro

    def _deviance(self, mu, phi_mu, gamma, a_conv, a_prompt, data):
        """Scaled deviance plus the prior penalties for one expectation.

        ``phi_mu`` [combined-fit units] and ``gamma`` enter only through the
        external anchors, when the likelihood carries them.
        """
        mu = np.clip(mu, 1.0e-12, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            deviance = 2.0 * (mu - data + np.where(
                data > 0.0, data * np.log(data / mu), 0.0))
        scaled = deviance / (1.0 + MODEL_SYS**2 * mu)
        penalty = (((a_conv - CONV_PRIOR[0]) / CONV_PRIOR[1]) ** 2
                   + ((a_prompt - PROMPT_PRIOR[0]) / PROMPT_PRIOR[1]) ** 2)
        for name, value in (("gamma", gamma), ("phi_mu", phi_mu)):
            if name in self._anchors:
                center, width = self._anchors[name]
                penalty += ((value - center) / width) ** 2
        return float(np.sum(scaled)) + penalty

    @staticmethod
    def _out_of_bounds(gamma, a_conv, a_prompt) -> bool:
        return (a_conv <= 0.0 or a_prompt < 0.0
                or not GAMMA_BOUNDS[0] <= gamma <= GAMMA_BOUNDS[1])

    def _objective(self, r, norm, gamma, a_conv, a_prompt, data):
        """Objective at one flavour-ratio parameter point."""
        if norm < 0.0 or self._out_of_bounds(gamma, a_conv, a_prompt):
            return 1.0e12
        return self._deviance(self.expectation(r, norm, gamma, a_conv, a_prompt),
                              2.0 * norm * (1.0 - r), gamma, a_conv, a_prompt,
                              data)

    def _objective_counts(self, n_mu, n_tau, gamma, a_conv, a_prompt, data):
        """Objective with each channel's astrophysical track count fixed.

        The shape still follows the index; only the window totals are held,
        so the count plane asks what the data say about *how many* tracks
        each channel contributes.
        """
        if self._out_of_bounds(gamma, a_conv, a_prompt):
            return 1.0e12
        a_mu, a_tau = self._astro("mu", gamma), self._astro("tau", gamma)
        astro = n_mu * a_mu / a_mu.sum() + n_tau * a_tau / a_tau.sum()
        mu = a_conv * self._folded_conv + a_prompt * self._folded_prompt + astro
        return self._deviance(mu, 2.0 * n_mu / a_mu.sum(), gamma, a_conv,
                              a_prompt, data)

    def channel_yield_ratio(self, gamma) -> float:
        """Tracks per unit flux, tau channel over ``nu_mu`` channel."""
        return float(self._astro("tau", gamma).sum()
                     / self._astro("mu", gamma).sum())

    @staticmethod
    def _minimize(function, starts):
        """Best Nelder-Mead result over a list of simplex seeds."""
        best = None
        for x0 in starts:
            trial = minimize(function, x0=np.asarray(x0, dtype=float),
                             method="Nelder-Mead",
                             options={"xatol": 1.0e-4, "fatol": 1.0e-6,
                                      "maxiter": 6000})
            if best is None or trial.fun < best.fun:
                best = trial
        return best.fun, best.x

    def delta_ll(self, r, data=None, warm_start=None,
                 use_default_start=True) -> tuple[float, np.ndarray]:
        """Profiled objective at one flavour ratio, and the nuisances.

        Parameters
        ----------
        r : float
            Flavour ratio the fit is conditioned on.
        data : np.ndarray, optional
            Counts on the fit window; the observed counts when omitted.
        warm_start : np.ndarray, optional
            Extra simplex seed, typically a neighbouring fit's nuisances.
        use_default_start : bool, optional
            Also seed from the global default point; toys switch this off
            once warm-started.
        """
        data = self.data if data is None else data
        starts = [(1.0, PIVOT_GAMMA, 1.0, 1.0)] if use_default_start else []
        if warm_start is not None:
            starts.append(warm_start)
        return self._minimize(
            lambda p: self._objective(r, p[0], p[1], p[2], p[3], data), starts)

    def delta_ll_fluxes(self, phi_mu, phi_tau, warm_start=None):
        """Profiled objective at fixed per-flavour fluxes.

        ``phi_mu`` and ``phi_tau`` are in units of the combined-fit
        per-flavour flux; the index and the atmospheric normalizations are
        profiled. The point maps onto :meth:`expectation` through
        ``norm = (phi_mu + phi_tau) / 2`` and
        ``r = phi_tau / (phi_mu + phi_tau)``.
        """
        total = phi_mu + phi_tau
        r, norm = (0.0, 0.0) if total <= 0.0 else (phi_tau / total, 0.5 * total)
        starts = [(PIVOT_GAMMA, 1.0, 1.0)]
        if warm_start is not None:
            starts.append(warm_start)
        return self._minimize(
            lambda p: self._objective(r, norm, p[0], p[1], p[2], self.data),
            starts)

    def delta_ll_counts(self, n_mu, n_tau, warm_start=None):
        """Profiled objective at fixed per-channel track counts."""
        starts = [(PIVOT_GAMMA, 1.0, 1.0)]
        if warm_start is not None:
            starts.append(warm_start)
        return self._minimize(
            lambda p: self._objective_counts(n_mu, n_tau, p[0], p[1], p[2],
                                             self.data), starts)

    def delta_ll_tau(self, phi_tau, warm_start=None):
        """Profiled objective at a fixed tau flux, the ``nu_mu`` flux free.

        With the anchors carried, this is the tau-flux profile at IceCube's
        measured ``nu_mu`` flux and index.
        """
        def objective(p):
            phi_mu, gamma, a_conv, a_prompt = p
            if phi_mu < 0.0:
                return 1.0e12
            total = phi_mu + phi_tau
            r, norm = ((0.0, 0.0) if total <= 0.0
                       else (phi_tau / total, 0.5 * total))
            return self._objective(r, norm, gamma, a_conv, a_prompt, self.data)

        starts = [(0.8, PIVOT_GAMMA, 1.0, 1.0)]
        if warm_start is not None:
            starts.append(warm_start)
        return self._minimize(objective, starts)

    def tau_profile(self):
        """``2 Delta ln L`` on :data:`PHI_TAU_SCAN`, floored at its minimum."""
        curve, warm = [], None
        for phi_tau in PHI_TAU_SCAN:
            value, warm = self.delta_ll_tau(phi_tau, warm)
            curve.append(value)
        curve = np.array(curve)
        return curve - curve.min()

    def _plane(self, x_grid, y_grid, fit):
        grid = np.empty((x_grid.size, y_grid.size))
        warm = None
        for i, x in enumerate(x_grid):
            for j, y in enumerate(y_grid):
                grid[i, j], warm = fit(x, y, warm)
        return grid - grid.min()

    def flux_plane(self):
        """``2 Delta ln L`` on (:data:`PHI_MU_GRID`, :data:`PHI_TAU_GRID`).

        Floored at its minimum over the whole plane, physical or not, so the
        valley the ratio scan walks along is visible in full.
        """
        return self._plane(PHI_MU_GRID, PHI_TAU_GRID, self.delta_ll_fluxes)

    def count_plane(self):
        """``2 Delta ln L`` on (:data:`N_MU_GRID`, :data:`N_TAU_GRID`)."""
        return self._plane(N_MU_GRID, N_TAU_GRID, self.delta_ll_counts)

    def profile(self, data=None):
        """``2 Delta ln L`` on :data:`R_GRID` with the per-point nuisances.

        Parameters
        ----------
        data : np.ndarray, optional
            Counts on the fit window; the observed counts when omitted.
        """
        curves, nuisances = [], []
        warm = None
        for r in R_GRID:
            value, params = self.delta_ll(r, data=data, warm_start=warm)
            curves.append(value)
            nuisances.append(params)
            warm = params
        curve = np.array(curves)
        return curve - curve.min(), np.array(nuisances)


def interval(curve, level):
    """Crossing points of the profile at one threshold.

    Parameters
    ----------
    curve : np.ndarray
        Profile statistic on :data:`R_GRID`.
    level : float or np.ndarray
        Threshold, either one Wilks level or a calibrated threshold per grid
        point.
    """
    below = curve <= level
    return float(R_GRID[below].min()), float(R_GRID[below].max())


def fc_calibration(likelihood, nuisances, n_toys, seed, rebuild):
    """Toy distributions of the profile statistic, cached across runs.

    At each truth point of :data:`FC_R_TRUE`, Poisson pseudo-experiments are
    drawn from an anchor-consistent truth -- the ``nu_mu`` flux and index at
    their anchors, the tau flux set by the ratio, the atmospheric
    normalizations at the data's profiled values -- and the statistic
    ``q(r_true)``, the fixed-ratio fit minus the toy's global minimum
    scanned on :data:`FC_R_SCAN`, is computed for each. Truths must respect
    the anchors: toys built at the profiled ``nu_mu`` flux of a high ratio
    (which the anchor drives to zero) carry the anchor penalty into every
    fit and can never be excluded. The prior centres stay fixed, matching
    the profile construction on the data.

    Parameters
    ----------
    likelihood : RecoLikelihood
        The fitted likelihood.
    nuisances : np.ndarray
        Profiled nuisances on :data:`R_GRID` from the data fit; the toy
        expectations are built at the truth point's entry.
    n_toys : int
        Pseudo-experiments per truth point.
    seed : int
        Generator seed.
    rebuild : bool
        Ignore the cache and rebuild.

    Returns
    -------
    q : np.ndarray, shape (FC_R_TRUE.size, n_toys)
        Statistic distributions, one row per truth point.
    """
    priors = np.array([CONV_PRIOR, PROMPT_PRIOR, TRACKS_GAMMA,
                       (TRACKS_PHI_MU[0] * 1e18, TRACKS_PHI_MU[1] * 1e18)])
    if _FC_CACHE.exists() and not rebuild:
        cache = np.load(_FC_CACHE)
        if (int(cache["n_toys"]) == n_toys and int(cache["seed"]) == seed
                and np.array_equal(cache["r_true"], FC_R_TRUE)
                and "priors" in cache and np.array_equal(cache["priors"], priors)
                and "version" in cache and int(cache["version"]) == _FC_VERSION):
            print(f"  cached toy distributions from {_FC_CACHE.name}")
            return cache["q"]
    rng = np.random.default_rng(seed)
    q = np.empty((FC_R_TRUE.size, n_toys))
    for i, r_true in enumerate(FC_R_TRUE):
        profiled = nuisances[int(np.argmin(np.abs(R_GRID - r_true)))]
        params = np.array((0.5 * ANCHORS["phi_mu"][0] / (1.0 - r_true),
                           ANCHORS["gamma"][0], profiled[2], profiled[3]))
        mu = likelihood.expectation(r_true, *params)
        for t in range(n_toys):
            toy = rng.poisson(mu).astype(float)
            fixed, warm = likelihood.delta_ll(r_true, data=toy,
                                              warm_start=params)
            lowest = fixed
            for r_scan in FC_R_SCAN:
                value, warm = likelihood.delta_ll(r_scan, data=toy,
                                                  warm_start=warm,
                                                  use_default_start=False)
                lowest = min(lowest, value)
            q[i, t] = fixed - lowest
        print(f"  r_true {r_true:.1f}: c68 {np.percentile(q[i], 68.27):5.2f}, "
              f"c95 {np.percentile(q[i], 95.0):5.2f}")
    _FC_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(_FC_CACHE, r_true=FC_R_TRUE, q=q, n_toys=n_toys, seed=seed,
             priors=priors, version=_FC_VERSION)
    return q


def jackknife_tail(likelihood):
    """Leave-one-out profiles over the Poisson-limited tail.

    Every occupied bin above :data:`JACKKNIFE_LOG10_E` loses its event in
    turn and the full profile is rerun. A best fit that crosses the scan
    range under a one-event change is a fluctuation reading, and the summary
    prints every leave-one-out fit so that fragility is visible.

    Parameters
    ----------
    likelihood : RecoLikelihood
        The fitted likelihood.

    Returns
    -------
    rows : list of tuple
        One ``(log10_e_center, band_index, r_hat, delta_ll_r0)`` per removed
        event.
    """
    centers = 0.5 * (RECO_EDGES[:-1] + RECO_EDGES[1:])[likelihood._window]
    rows = []
    for i, j in np.argwhere(likelihood.data > 0):
        if centers[i] < JACKKNIFE_LOG10_E:
            continue
        data = likelihood.data.copy()
        data[i, j] -= 1.0
        curve, _ = likelihood.profile(data=data)
        rows.append((float(centers[i]), int(j),
                     float(R_GRID[int(np.argmin(curve))]), float(curve[0])))
    return rows


def _save(fig, out_dir: pathlib.Path, stem: str) -> None:
    """Write one figure to ``out_dir`` as both PDF and PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {path.resolve()}")
    plt.close(fig)


def figure_profile(curve, c68, c95, out_dir) -> None:
    """The measured profile with Wilks and calibrated thresholds.

    The standard-oscillation band is shaded.

    Parameters
    ----------
    curve : np.ndarray
        Profile statistic on :data:`R_GRID`.
    c68, c95 : np.ndarray
        Calibrated thresholds on :data:`FC_R_TRUE`.
    """
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.0, 3.0))
        ax.axvspan(R_MIN_STD, R_MAX, color="0.93", zorder=0,
                   label="std. osc. band")
        top = min(max(1.3, 1.25 * max(curve.max(), c95.max())), 12.0)
        for level, note in ((1.0, r"$68\%$"), (3.84, r"$95\%$")):
            if level < top:
                ax.axhline(level, color="0.75", lw=0.7, ls=":")
                ax.text(R_GRID[-1] * 1.005, level, note, fontsize=7,
                        color="0.45", va="center")
        ax.plot(FC_R_TRUE, c68, color="#7570b3", lw=0.9, ls="--",
                label=r"FC $68\%$ threshold")
        ax.plot(FC_R_TRUE, c95, color="#7570b3", lw=0.9, ls="-.",
                label=r"FC $95\%$ threshold")
        ax.plot(R_GRID, curve, color="#e7298a", lw=1.3, label="profile")
        ax.set_xlim(0.0, R_GRID[-1])
        ax.set_ylim(0.0, top)
        ax.set_xlabel(r"$f_\tau \,/\, (f_\mu + f_\tau)$ at Earth")
        ax.set_ylabel(r"$2\,\Delta\ln L$ (IC86 data, anchored)")
        ax.legend(fontsize=6, frameon=False, loc="upper left")
        _save(fig, out_dir, "51a_dr2_profile")


def figure_spectra(likelihood, best, dec_edges_deg, out_dir) -> None:
    """Data against the best-fit components, summed over declination."""
    r, (norm, gamma, a_conv, a_prompt) = best
    window_centers = 0.5 * (RECO_EDGES[:-1] + RECO_EDGES[1:])[likelihood._window]
    components = {
        "atm. conventional": a_conv * likelihood._folded_conv,
        "atm. prompt": a_prompt * likelihood._folded_prompt,
        r"astro $\nu_\mu$": norm * (1.0 - r) * likelihood._astro("mu", gamma),
        r"astro $\nu_\tau \to \mu$": norm * r * likelihood._astro("tau", gamma),
    }
    colors = ("0.65", "0.4", "#7570b3", "#e7298a")
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.0, 3.0))
        total = np.zeros_like(window_centers)
        for (name, comp), color in zip(components.items(), colors):
            summed = comp.sum(axis=1)
            total += summed
            if summed.max() <= 0.0:
                name += r" ($= 0$ at best fit)"
            ax.step(window_centers, summed, where="mid", color=color, lw=1.1,
                    label=name)
        ax.step(window_centers, total, where="mid", color="k", lw=1.3,
                label="total")
        observed = likelihood.data.sum(axis=1)
        ax.errorbar(window_centers, observed, yerr=np.sqrt(observed), fmt="o",
                    color="k", ms=2.2, lw=0.8, capsize=1.5, label="IC86 data",
                    zorder=5)
        ax.set_yscale("log")
        ax.set_ylim(0.3, None)
        ax.set_xlabel(r"$\log_{10}(E_{\rm reco}\,/\,\mathrm{GeV})$")
        ax.set_ylabel("events per bin, upgoing")
        ax.legend(fontsize=6, frameon=False, loc="upper right")
        _save(fig, out_dir, "51b_dr2_spectra")


def figure_triangle(curve, interval68, interval95, out_dir) -> None:
    """The measured intervals and best fit on example 50's flavour triangle.

    Example 50's frame and published contours, with the DR2 measurement on
    top: the calibrated 68% and 95% intervals on ``r`` as nested wedges
    anchored at the ``nu_e`` vertex, the best-fit ratio as a ray to the
    ``mu``-``tau`` edge, and the standard-oscillation band as two dotted
    rays. A wedge that spans the whole physical scan says so in its label.
    Tracks constrain only ``f_tau / (f_mu + f_tau)``, so the ray -- every
    point of which is the same measurement -- is the honest shape of the
    best fit, the star at its far end a reading aid, not a point estimate of
    all three fractions.
    """
    r_best = float(R_GRID[int(np.argmin(curve))])
    color = "#e7298a"
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(4.4, 4.2))
        corners = _EX50.draw_triangle_frame(ax)

        for (lo, hi), level, alpha, zorder in (
            (interval68, "68", 0.18, 2.2),
            (interval95, "95", 0.07, 2.0),
        ):
            whole = lo <= R_GRID[0] and hi >= R_GRID[-1] - 1.0e-9
            note = " (whole range)" if whole else ""
            wedge = plt.Polygon(
                [corners["e"], _EX50._ternary_xy(0.0, 1.0 - lo, lo),
                 _EX50._ternary_xy(0.0, 1.0 - hi, hi)], closed=True,
                facecolor=color, edgecolor=color, lw=0.6, alpha=alpha,
                zorder=zorder,
                label=rf"IC86 tracks ${level}\%${note}")
            ax.add_patch(wedge)

        tip = _EX50._ternary_xy(0.0, 1.0 - r_best, r_best)
        ax.plot(*zip(corners["e"], tip), color=color, lw=1.1, ls="--",
                zorder=4, label=rf"best fit $\hat r = {r_best:.2f}$")
        ax.plot(*tip, marker="*", color=color, ms=7, ls="none", zorder=5)

        for r, label in ((R_MIN_STD, "std. osc. band, "
                          rf"$r \in [{R_MIN_STD:.2f}, {R_MAX:.2f}]$"),
                         (R_MAX, "_nolegend_")):
            bound = _EX50._ternary_xy(0.0, 1.0 - r, r)
            ax.plot(*zip(corners["e"], bound), color="0.4", lw=0.8, ls=":",
                    zorder=3, label=label)

        _EX50.draw_published_curves(ax)

        handles, labels = ax.get_legend_handles_labels()
        measured = ax.legend(handles[:4], labels[:4], fontsize=8,
                             frameon=False, loc="upper left",
                             bbox_to_anchor=(0.0, 1.16), handlelength=1.4,
                             labelspacing=0.3)
        ax.add_artist(measured)
        ax.legend(handles[4:], labels[4:], fontsize=8, frameon=False,
                  loc="upper right", bbox_to_anchor=(1.08, 1.16),
                  handlelength=1.6, labelspacing=0.3)
        _save(fig, out_dir, "51c_dr2_triangle")


def plane_minima(plane, x_grid, y_grid, slope):
    """Unconstrained and physical minima of a plane, as grid indices.

    The physical region is ``y <= slope * x``: the standard-oscillation
    ceiling in the plane's own units.
    """
    physical = y_grid[None, :] <= slope * x_grid[:, None] + 1.0e-9
    free = np.unravel_index(int(np.argmin(plane)), plane.shape)
    phys = np.unravel_index(int(np.argmin(np.where(physical, plane, np.inf))),
                            plane.shape)
    return free, phys


def _draw_plane(ax, x_grid, y_grid, plane, slopes, free, phys, reference,
                y_max):
    """Shared body of the flux and count planes.

    Parameters
    ----------
    slopes : tuple of float
        ``(lower, upper)`` slopes of the standard-oscillation band.
    free, phys : tuple of int
        Grid indices of the unconstrained and physical minima.
    reference : tuple of float
        The combined-fit 1:1:1 point in the plane's units.
    y_max : float
        Upper plot limit; an unconstrained minimum above it is pointed at.
    """
    from matplotlib.lines import Line2D

    x = np.array([0.0, x_grid[-1]])
    ax.fill_between(x, slopes[0] * x, slopes[1] * x, color="#7570b3",
                    alpha=0.18, lw=0, zorder=0, label="std. osc. band")
    ax.contour(x_grid, y_grid, plane.T, levels=[2.30, 5.99], colors="#e7298a",
               linestyles=["-", "--"], linewidths=[1.2, 0.9], zorder=2)
    x_free, y_free = x_grid[free[0]], y_grid[free[1]]
    if y_free <= y_max:
        ax.plot(x_free, y_free, "o", mfc="none", color="#e7298a", ms=5,
                zorder=4, label="unconstrained minimum")
    else:
        ax.annotate(f"unconstrained\nminimum at\n({x_free:.2g}, {y_free:.2g})",
                    xy=(x_free + 0.02 * x_grid[-1], 0.97 * y_max),
                    xytext=(x_free + 0.07 * x_grid[-1], 0.60 * y_max),
                    fontsize=6, color="#e7298a", ha="left", va="top",
                    arrowprops=dict(arrowstyle="->", color="#e7298a", lw=0.8),
                    zorder=4)
    ax.plot(x_grid[phys[0]], y_grid[phys[1]], "*", color="#e7298a", ms=8,
            zorder=5, label=rf"best fit, $r \leq {R_MAX:.2f}$")
    ax.plot(*reference, "+", color="k", ms=7, mew=1.2, zorder=5,
            label="combined fit, 1:1:1")
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color="#e7298a", lw=1.2),
                Line2D([], [], color="#e7298a", lw=0.9, ls="--")]
    labels += [r"$68\%$", r"$95\%$"]
    ax.set_xlim(0.0, x_grid[-1])
    ax.set_ylim(0.0, y_max)
    ax.legend(handles, labels, fontsize=6, frameon=False, loc="upper right")


def figure_flux_plane(plane, out_dir) -> None:
    """The ``(phi_mu, phi_tau)`` profile: the valley the ratio scan walks.

    The 68% and 95% contours (``2 Delta ln L`` of 2.30 and 5.99, two
    parameters), the standard-oscillation band, the physical best fit and
    the combined-fit point ``(1, 1)``. The axis stops at
    :data:`PHI_TAU_PLOT_MAX`; the unconstrained minimum lies far above it
    and is pointed at.
    """
    slopes = tuple(r / (1.0 - r) for r in (R_MIN_STD, R_MAX))
    free, phys = plane_minima(plane, PHI_MU_GRID, PHI_TAU_GRID, slopes[1])
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.0, 3.0))
        _draw_plane(ax, PHI_MU_GRID, PHI_TAU_GRID, plane, slopes, free, phys,
                    (1.0, 1.0), PHI_TAU_PLOT_MAX)
        ax.set_xlabel(r"$\Phi_{\nu_\mu} \,/\, \Phi_{\rm combined}$")
        ax.set_ylabel(r"$\Phi_{\nu_\tau} \,/\, \Phi_{\rm combined}$")
        _save(fig, out_dir, "51d_dr2_flux_plane")


def figure_count_plane(plane, likelihood, out_dir,
                       stem="51e_dr2_count_plane") -> None:
    """The same likelihood in expected-track units.

    Each axis is the number of astrophysical tracks a channel contributes
    to the fit window, so the valley reads directly as "the data count the
    excess and cannot tell who made it". The standard-oscillation band and
    the combined-fit point are converted with the channel yield ratio at
    the pivot index.
    """
    yield_ratio = likelihood.channel_yield_ratio(PIVOT_GAMMA)
    slopes = tuple(r / (1.0 - r) * yield_ratio for r in (R_MIN_STD, R_MAX))
    free, phys = plane_minima(plane, N_MU_GRID, N_TAU_GRID, slopes[1])
    n_mu_ref = 0.5 * likelihood._astro("mu", PIVOT_GAMMA).sum()
    n_tau_ref = 0.5 * likelihood._astro("tau", PIVOT_GAMMA).sum()
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.0, 3.0))
        _draw_plane(ax, N_MU_GRID, N_TAU_GRID, plane, slopes, free, phys,
                    (n_mu_ref, n_tau_ref), N_TAU_GRID[-1])
        ax.set_xlabel(r"astro $\nu_\mu$ tracks in the window")
        ax.set_ylabel(r"astro $\nu_\tau \to \mu$ tracks in the window")
        _save(fig, out_dir, stem)


def figure_tau_profile(tau_curve, out_dir) -> None:
    """The tau-flux profile at the anchored ``nu_mu`` flux -- the headline.

    The standard-oscillation band is the anchored ``nu_mu`` flux times the
    band of ``f_tau / f_mu`` at Earth; the 68% and one-sided 90% levels are
    Wilks, which the anchors make adequate (the plane is closed and the
    minimum is interior or at ``Phi_tau = 0``).
    """
    phi_mu = ANCHORS["phi_mu"][0]
    lo, hi = (phi_mu * r / (1.0 - r) for r in (R_MIN_STD, R_MAX))
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.0, 3.0))
        ax.axvspan(lo, hi, color="0.93", zorder=0, label="std. osc. band")
        top = min(max(4.5, 1.1 * tau_curve.max()), 12.0)
        for level, note in ((1.0, r"$68\%$"), (2.71, r"$90\%$")):
            ax.axhline(level, color="0.75", lw=0.7, ls=":")
            ax.text(PHI_TAU_SCAN[-1] * 1.01, level, note, fontsize=7,
                    color="0.45", va="center")
        ax.plot(PHI_TAU_SCAN, tau_curve, color="#e7298a", lw=1.3,
                label="profile")
        ax.set_xlim(0.0, PHI_TAU_SCAN[-1])
        ax.set_ylim(0.0, top)
        ax.set_xlabel(r"$\Phi_{\nu_\tau} \,/\, \Phi_{\rm combined}$")
        ax.set_ylabel(r"$2\,\Delta\ln L$ (IC86 data, anchored)")
        ax.legend(fontsize=6, frameon=False, loc="upper left")
        _save(fig, out_dir, "51f_dr2_tau_profile")


#: Wilks levels for two parameters, the triangle's degrees of freedom.
LEVEL68_2D, LEVEL95_2D = 2.30, 5.99


def _fractions_from_xy(x, y):
    """Inverse of example 50's ``_ternary_xy``."""
    f_mu = y / (np.sqrt(3.0) / 2.0)
    f_e = x - 0.5 * f_mu
    return f_e, f_mu, 1.0 - f_e - f_mu


def mese_surface(x, y):
    """MESE's ``2 Delta ln L`` on the triangle, rebuilt from its contours.

    About the published best fit, each direction gets a power law in radius
    that passes through the 68% and 95% contours exactly (levels 2.30 and
    5.99), so both published curves are reproduced at every angle. Where
    the 95% contour leaves the triangle the profile is quadratic through
    the 68% contour alone.

    Parameters
    ----------
    x, y : np.ndarray
        Cartesian triangle coordinates (example 50's orientation).

    Returns
    -------
    surface : np.ndarray
        ``2 Delta ln L`` relative to the MESE best fit.
    """
    cx, cy = _EX50._ternary_xy(*_EX50.ICECUBE_BEST_FIT)
    published = _EX50.load_published_curves()
    theta = np.linspace(-np.pi, np.pi, 1441)
    radius = {}
    for name in ("contour68", "contour95"):
        f = published[name]
        px, py = _EX50._ternary_xy(f[:, 0], f[:, 1], f[:, 2])
        ang, rad = np.arctan2(py - cy, px - cx), np.hypot(px - cx, py - cy)
        order = np.argsort(ang)
        ang, rad = ang[order], rad[order]
        wrapped_ang = np.concatenate([ang - 2 * np.pi, ang, ang + 2 * np.pi])
        r_theta = np.interp(theta, wrapped_ang, np.tile(rad, 3))
        steps = np.diff(np.append(ang, ang[0] + 2 * np.pi))
        for i in np.where(steps > 4.0 * np.median(steps))[0]:
            lo, hi = ang[i], ang[i] + steps[i]
            gap = ((theta > lo) & (theta < hi)) | ((theta + 2 * np.pi > lo)
                                                   & (theta + 2 * np.pi < hi))
            r_theta[gap] = np.nan
        radius[name] = r_theta
    rho, ang = np.hypot(x - cx, y - cy), np.arctan2(y - cy, x - cx)
    r68 = np.interp(ang, theta, radius["contour68"])
    r95 = np.interp(ang, theta, radius["contour95"])
    with np.errstate(invalid="ignore", divide="ignore"):
        power = np.where(np.isfinite(r95),
                         np.log(LEVEL95_2D / LEVEL68_2D) / np.log(r95 / r68), 2.0)
        return LEVEL68_2D * (rho / r68) ** power


def combined_surface(curve):
    """MESE plus this work on a triangle grid.

    This work's anchored profile in ``r = f_tau / (f_mu + f_tau)`` is
    constant along rays from the ``nu_e`` vertex; the two samples are taken
    as independent (MESE is a starting-event selection, DR2 a through-going
    one).

    Returns
    -------
    x, y : np.ndarray
        Grid coordinates.
    total : np.ndarray
        Combined ``2 Delta ln L``, floored at its minimum; NaN outside the
        triangle.
    """
    x, y = np.meshgrid(np.linspace(0.0, 1.0, 501),
                       np.linspace(0.0, np.sqrt(3.0) / 2.0, 434))
    f_e, f_mu, f_tau = _fractions_from_xy(x, y)
    inside = (f_e >= 0.0) & (f_mu >= 0.0) & (f_tau >= 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(f_mu + f_tau > 1.0e-9, f_tau / (f_mu + f_tau), 0.0)
    total = mese_surface(x, y) + np.interp(r, R_GRID, curve)
    total = np.where(inside, total, np.nan)
    return x, y, total - np.nanmin(total)


def figure_combined_triangle(curve, curve_gen2, out_dir) -> tuple[float, float, float]:
    """MESE combined with this work, and with the Gen2 forecast.

    The published MESE contours stay as drawn; the combined IC86 regions
    (filled, pink) and the combined Gen2-forecast contours (purple) go on
    top, both at the two-parameter Wilks levels.

    Returns
    -------
    best : tuple of float
        Combined IC86 best-fit fractions ``(f_e, f_mu, f_tau)``.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    x, y, total = combined_surface(curve)
    _, _, total_gen2 = combined_surface(curve_gen2)
    i, j = np.unravel_index(int(np.nanargmin(total)), total.shape)
    best = tuple(float(v) for v in _fractions_from_xy(x[i, j], y[i, j]))
    pink, purple = "#e7298a", "#7570b3"
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(4.4, 4.2))
        _EX50.draw_triangle_frame(ax)
        ax.contourf(x, y, total, levels=[0.0, LEVEL68_2D, LEVEL95_2D],
                    colors=[pink, pink], alpha=0.18, zorder=2)
        for surface, color, z in ((total, pink, 4), (total_gen2, purple, 5)):
            ax.contour(x, y, surface, levels=[LEVEL68_2D], colors=color,
                       linewidths=1.3, zorder=z)
            ax.contour(x, y, surface, levels=[LEVEL95_2D], colors=color,
                       linestyles="--", linewidths=1.0, zorder=z)
        ax.plot(x[i, j], y[i, j], marker="*", color=pink, ms=8, ls="none",
                zorder=6, label="combined best fit")
        _EX50.draw_published_curves(ax, skip=("icecube2022",))
        handles, labels = ax.get_legend_handles_labels()
        ours = [Patch(facecolor=pink, alpha=0.3, edgecolor=pink, lw=1.3),
                Line2D([], [], color=pink, ls="--", lw=1.0), handles[0],
                Line2D([], [], color=purple, lw=1.3),
                Line2D([], [], color=purple, ls="--", lw=1.0)]
        ours_labels = [r"MESE + IC86 tracks $68\%$",
                       r"MESE + IC86 tracks $95\%$", labels[0],
                       r"MESE + Gen2 forecast $68\%$",
                       r"MESE + Gen2 forecast $95\%$"]
        first = ax.legend(ours, ours_labels, fontsize=8, frameon=False,
                          loc="upper left", bbox_to_anchor=(0.0, 1.16),
                          handlelength=1.4, labelspacing=0.3)
        ax.add_artist(first)
        ax.legend(handles[1:], labels[1:], fontsize=8, frameon=False,
                  loc="upper right", bbox_to_anchor=(1.08, 1.16),
                  handlelength=1.6, labelspacing=0.3)
        _save(fig, out_dir, "51g_dr2_combined_triangle")
    return best


def main() -> None:
    args = parse_args()
    print("Loading examples 35, 45 and 46 ...")
    ex35 = load_example("35_point_source_effective_area.py", "_example_35")
    ex45 = load_example("45_first_principles_reach.py", "_example_45")
    ex46 = load_example("46_declination_resolved_reach.py", "_example_46")

    enu_edges, dec_edges, marginal, responses = fit_inputs(
        ex35, ex45, ex46, args.data_dir, args.rebuild_cache)
    atmos = atmospheric_fluxes(dec_edges)

    livetime_s = sum(
        compute_livetime_s(load_uptime(args.data_dir / "uptime" / f"{s}_exp.csv"))
        for s in IC86_SEASONS)
    print(f"  IC86 livetime: {livetime_s / (365.25 * 86400.0):.2f} yr")
    data_counts = binned_events(args.data_dir, dec_edges)

    likelihood = RecoLikelihood(enu_edges, marginal, responses, atmos,
                                dec_edges, livetime_s, data_counts,
                                anchors=ANCHORS)
    print(f"  fit window {FIT_RECO}: {likelihood.data.sum():,.0f} events")
    print(f"  anchors from IceCube's tracks fit: gamma {TRACKS_GAMMA[0]} +- "
          f"{TRACKS_GAMMA[1]}, nu_mu flux {TRACKS_PHI_MU[0] * 1e18:.2f} +- "
          f"{TRACKS_PHI_MU[1] * 1e18:.2f} x 1e-18 at 100 TeV "
          f"({ANCHORS['phi_mu'][0]:.2f} +- {ANCHORS['phi_mu'][1]:.2f} x combined fit)")
    print(f"  standard oscillations: r in [{R_MIN_STD:.2f}, {R_MAX:.2f}] at Earth")

    print("Profiling the flavour ratio against the data ...")
    curve, nuisances = likelihood.profile()
    i_best = int(np.argmin(curve))
    r_best = R_GRID[i_best]
    norm, gamma, a_conv, a_prompt = nuisances[i_best]

    print(f"\n  best fit r = {r_best:.2f}, Wilks 68% "
          f"[{interval(curve, 1.0)[0]:.2f}, {interval(curve, 1.0)[1]:.2f}], "
          f"95% [{interval(curve, 3.84)[0]:.2f}, {interval(curve, 3.84)[1]:.2f}]")
    print(f"  2 Delta lnL: r=0 at {curve[0]:.2f}, r=0.5 at "
          f"{curve[int(np.argmin(np.abs(R_GRID - 0.5)))]:.2f}, "
          f"r=1 at {curve[-1]:.2f}")
    print(f"  nuisances at best fit: astro norm {norm:.2f} x combined fit "
          f"(nu_mu flux {2.0 * norm * (1.0 - r_best):.2f}, nu_tau flux "
          f"{2.0 * norm * r_best:.2f}), gamma {gamma:.2f}, atm conv {a_conv:.2f}, "
          f"prompt {a_prompt:.2f}")
    predicted = likelihood.expectation(r_best, *nuisances[i_best]).sum()
    print(f"  predicted {predicted:,.0f} events against {likelihood.data.sum():,.0f}")

    print(f"\nCalibrating the interval with {args.toys} pseudo-experiments "
          "per truth point ...")
    q = fc_calibration(likelihood, nuisances, args.toys, args.seed,
                       args.rebuild_fc or args.rebuild_cache)
    c68 = np.percentile(q, 68.27, axis=1)
    c95 = np.percentile(q, 95.0, axis=1)
    fc68 = interval(curve, np.interp(R_GRID, FC_R_TRUE, c68))
    fc95 = interval(curve, np.interp(R_GRID, FC_R_TRUE, c95))
    p_taufree = float(np.mean(q[0] >= curve[0]))
    print(f"\n  FC 68% [{fc68[0]:.2f}, {fc68[1]:.2f}], "
          f"95% [{fc95[0]:.2f}, {fc95[1]:.2f}]")
    print(f"  tau-free (r = 0) p-value: {p_taufree:.2f}")

    print("\nTail jackknife, one event removed per row:")
    for log_e, j, r_hat, d0 in jackknife_tail(likelihood):
        print(f"  without 10^{log_e:.2f} GeV in dec "
              f"[{dec_edges[j]:.1f}, {dec_edges[j + 1]:.1f}]: "
              f"r_hat {r_hat:.2f}, 2 Delta lnL(r=0) {d0:.2f}")

    print("\nProfiling the tau flux at the anchored nu_mu flux ...")
    tau_curve = likelihood.tau_profile()
    i_tau = int(np.argmin(tau_curve))
    inside = PHI_TAU_SCAN[tau_curve <= 1.0]
    upper90 = PHI_TAU_SCAN[tau_curve <= 2.71].max()
    _, p_tau = likelihood.delta_ll_tau(PHI_TAU_SCAN[i_tau])
    tau_yield = 0.5 * likelihood._astro("tau", p_tau[1]).sum()
    print(f"  tau flux: best {PHI_TAU_SCAN[i_tau]:.2f}, 68% "
          f"[{inside.min():.2f}, {inside.max():.2f}], < {upper90:.2f} at 90% "
          f"(x combined fit; std. osc. expects "
          f"{ANCHORS['phi_mu'][0] * R_MIN_STD / (1 - R_MIN_STD):.2f}-"
          f"{ANCHORS['phi_mu'][0] * R_MAX / (1 - R_MAX):.2f})")
    print(f"  in tracks: best {PHI_TAU_SCAN[i_tau] * tau_yield:.0f}, "
          f"< {upper90 * tau_yield:.0f} at 90%; profiled nu_mu flux {p_tau[0]:.2f}, "
          f"gamma {p_tau[1]:.2f}, conv {p_tau[2]:.2f}, prompt {p_tau[3]:.2f}")

    print("\nScanning the (phi_mu, phi_tau) plane ...")
    plane = likelihood.flux_plane()
    free, phys = plane_minima(plane, PHI_MU_GRID, PHI_TAU_GRID,
                              R_MAX / (1.0 - R_MAX))
    print(f"  minimum: phi_mu {PHI_MU_GRID[free[0]]:.2f}, "
          f"phi_tau {PHI_TAU_GRID[free[1]]:.2f} (x combined fit); inside the "
          f"std. osc. ceiling: phi_mu {PHI_MU_GRID[phys[0]]:.2f}, phi_tau "
          f"{PHI_TAU_GRID[phys[1]]:.2f}, 2 Delta lnL {plane[phys]:.2f}")
    print("Scanning the (N_mu, N_tau) count plane ...")
    counts_plane = likelihood.count_plane()
    yield_ratio = likelihood.channel_yield_ratio(PIVOT_GAMMA)
    free_n, phys_n = plane_minima(counts_plane, N_MU_GRID, N_TAU_GRID,
                                  R_MAX / (1.0 - R_MAX) * yield_ratio)
    print(f"  tau/mu tracks per unit flux at gamma {PIVOT_GAMMA}: {yield_ratio:.3f}")
    print(f"  minimum: {N_MU_GRID[free_n[0]]:.0f} nu_mu + "
          f"{N_TAU_GRID[free_n[1]]:.0f} tau tracks; inside the ceiling: "
          f"{N_MU_GRID[phys_n[0]]:.0f} + {N_TAU_GRID[phys_n[1]]:.0f}, "
          f"2 Delta lnL {counts_plane[phys_n]:.2f}")

    print("\nGen2 forecast: same fold and anchors, example 50's Gen2 geometry, "
          f"{_EX50.FORECAST_YEARS:g} yr Asimov at 1:1:1 ...")
    responses_gen2 = gen2_responses(ex35, ex45, ex46, dec_edges, args.rebuild_cache)
    gen2 = RecoLikelihood(enu_edges, marginal, responses_gen2, atmos, dec_edges,
                          _EX50.FORECAST_YEARS * 365.25 * 86400.0,
                          np.zeros_like(data_counts), anchors=ANCHORS)
    asimov = gen2.expectation(0.5, ANCHORS["phi_mu"][0], TRACKS_GAMMA[0], 1.0, 1.0)
    curve_gen2, _ = gen2.profile(data=asimov)
    lo, hi = interval(curve_gen2, 1.0)
    astro_gen2 = (asimov - gen2.expectation(0.5, 0.0, TRACKS_GAMMA[0], 1.0, 1.0)).sum()
    print(f"  Gen2 tracks alone: {astro_gen2:.0f} astro tracks in the window, "
          f"Wilks 68% [{lo:.2f}, {hi:.2f}], 2 Delta lnL(r=0) {curve_gen2[0]:.2f}, "
          f"r=1 {curve_gen2[-1]:.2f}")

    print()
    figure_profile(curve, c68, c95, args.out_dir)
    figure_spectra(likelihood, (r_best, nuisances[i_best]), dec_edges,
                   args.out_dir)
    figure_triangle(curve, fc68, fc95, args.out_dir)
    figure_flux_plane(plane, args.out_dir)
    figure_count_plane(counts_plane, likelihood, args.out_dir)
    figure_tau_profile(tau_curve, args.out_dir)
    best = figure_combined_triangle(curve, curve_gen2, args.out_dir)
    print(f"combined MESE + IC86 tracks best fit: f_e {best[0]:.2f}, f_mu "
          f"{best[1]:.2f}, f_tau {best[2]:.2f} (MESE alone: "
          f"{_EX50.ICECUBE_BEST_FIT[0]:.2f}, {_EX50.ICECUBE_BEST_FIT[1]:.2f}, "
          f"{_EX50.ICECUBE_BEST_FIT[2]:.2f})")


if __name__ == "__main__":
    main()
