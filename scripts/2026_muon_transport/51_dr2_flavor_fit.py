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

from softpaws.comparison import reco_likelihood as _reco
from softpaws.comparison.feldman_cousins import cached_toys, profile_interval
from softpaws.data.icecube import (
    IC86_SEASONS,
)
from softpaws.data.loader import compute_livetime_s, load_uptime

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"
_DEFAULT_DATA_DIR = _HERE.parents[1] / "src" / "softpaws" / "data" / "dataverse_files"

#: Cached banded responses and smearing marginal, keyed to this example.
_CACHE = _DEFAULT_OUT_DIR / "51_fit_inputs.npz"

#: Cached IceCube-Gen2 banded responses for the forecast of figure 51g.
_GEN2_CACHE = _DEFAULT_OUT_DIR / "51_gen2_responses.npz"

#: Atmospheric flux cache from example 22.
_MCEQ_CACHE = _HERE / "output" / "22_mceq_atmospheric_flux.npz"

#: The fit's grids, priors, anchors and likelihood now live in
#: :mod:`softpaws.comparison.reco_likelihood`. They are bound here under
#: their old names because examples 52, 53, 54, 59, 61, 62 and 76 load this
#: file and read them from it.
PIVOT_PHI0 = _reco.PIVOT_PHI0
PIVOT_GAMMA = _reco.PIVOT_GAMMA
RECO_EDGES = _reco.RECO_EDGES
FIT_RECO = _reco.FIT_RECO
MODEL_SYS = _reco.MODEL_SYS
CONV_PRIOR = _reco.CONV_PRIOR
PROMPT_PRIOR = _reco.PROMPT_PRIOR
GAMMA_BOUNDS = _reco.GAMMA_BOUNDS
LOG10_E_GRID = _reco.LOG10_E_GRID
GAMMA_GRID = _reco.GAMMA_GRID
TAU_DECAY_X = _reco.TAU_DECAY_X
R_GRID = _reco.R_GRID
FC_R_TRUE = _reco.FC_R_TRUE
FC_R_SCAN = _reco.FC_R_SCAN
PHI_MU_GRID = _reco.PHI_MU_GRID
PHI_TAU_GRID = _reco.PHI_TAU_GRID
N_MU_GRID = _reco.N_MU_GRID
N_TAU_GRID = _reco.N_TAU_GRID
PHI_TAU_SCAN = _reco.PHI_TAU_SCAN
TRACKS_GAMMA = _reco.TRACKS_GAMMA
TRACKS_PHI_MU = _reco.TRACKS_PHI_MU
ANCHORS = _reco.ANCHORS
RecoLikelihood = _reco.RecoLikelihood

#: Upper limit of the tau-flux axis in figure 51d [combined-fit units]. The
#: scan reaches the unconstrained minimum, which lies far above the axis.
PHI_TAU_PLOT_MAX = 2.0

#: Cached toy distributions of the profile statistic.
_FC_CACHE = _DEFAULT_OUT_DIR / "51_fc_calibration.npz"

#: Reconstructed energy [log10 GeV] above which the tail jackknife removes
#: single events; the bins there hold at most one event each.
JACKKNIFE_LOG10_E = 5.5

#: Cache-format version of the toy distributions; bump on any change to the
#: folded model so stale calibrations rebuild.
_FC_VERSION = 5

#: Example 45's fitted configuration.
#: Example 45's IceCube pair, nu_mu only, with rock below the ice
#: (2026-09-02); the all-water kernel gave 42 m and 0.755.
FITTED_ATTENUATION_M = 33.7
FITTED_NORMALIZATION = 0.956


def load_example(stem: str, name: str):
    """Import a sibling example module by file stem."""
    spec = importlib.util.spec_from_file_location(name, _HERE / stem)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def physical_r_range(ex50) -> tuple[float, float]:
    """Earth flavour-ratio range standard oscillations allow.

    See :func:`softpaws.comparison.reco_likelihood.physical_r_range`.

    Parameters
    ----------
    ex50 : ModuleType
        Example 50, for its oscillation matrix.

    Returns
    -------
    r_min, r_max : float
        Band of the ratio over all source compositions.
    """
    return _reco.physical_r_range(ex50.earth_composition)


#: Example 50, for the triangle frame and the oscillation matrix.
_EX50 = load_example("50_flavor_triangle.py", "_example_50")

#: Standard-oscillation band of the Earth ratio, drawn on every figure.
R_MIN_STD, R_MAX = physical_r_range(_EX50)


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
# Inputs: banded responses, atmosphere, events
# ---------------------------------------------------------------------------


def banded_responses(ex35, ex45, ex46, dec_edges_deg, site=None):
    """Model channel responses on the smearing declination bands [cm^2].

    Wraps :func:`softpaws.comparison.reco_likelihood.banded_responses` with
    example 46's band-averaged effective area, evaluated in example 45's
    fitted configuration.

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

    def banded_aeff(channel, sin_dec_edges):
        print(f"  building banded response, {channel} channel ...")
        return ex46.model_banded(ex35, ex45, site, site45, sin_dec_edges, 8.0,
                                 (channel,), efficiency=FITTED_NORMALIZATION)

    return _reco.banded_responses(banded_aeff, dec_edges_deg, ex35.COMMON_LOG10_E)


def fit_inputs(ex35, ex45, ex46, data_dir: pathlib.Path, rebuild: bool):
    """Smearing marginal and banded responses, cached across runs."""
    cached = None if rebuild else _reco.cached_fit_inputs(_CACHE)
    if cached is not None:
        print(f"  cached fit inputs from {_CACHE.name}")
        return cached
    print("  marginalizing the IC86 smearing table ...")
    return _reco.fit_inputs(
        data_dir, _CACHE,
        lambda dec_edges: banded_responses(ex35, ex45, ex46, dec_edges),
        rebuild=True)


def gen2_responses(ex35, ex45, ex46, dec_edges, rebuild: bool):
    """IceCube-Gen2 banded responses, example 50's geometry, cached."""
    cached = None if rebuild else _reco.load_banded_responses(_GEN2_CACHE, dec_edges)
    if cached is not None:
        print(f"  cached Gen2 responses from {_GEN2_CACHE.name}")
        return cached
    sites = {s.name: s for s in ex35.build_sites()}
    gen2 = replace(sites["IceCube"], name="IceCube-Gen2",
                   radius_km=_EX50.GEN2_RADIUS_KM, height_km=_EX50.GEN2_HEIGHT_KM)
    responses = banded_responses(ex35, ex45, ex46, dec_edges, site=gen2)
    _reco.save_banded_responses(_GEN2_CACHE, dec_edges, responses)
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
    return _reco.atmospheric_fluxes(_MCEQ_CACHE, dec_edges_deg)


def binned_events(data_dir: pathlib.Path, dec_edges_deg):
    """IC86 upgoing events histogrammed on (reco grid, declination bands)."""
    counts, total = _reco.binned_events(data_dir, dec_edges_deg)
    print(f"  {total:,} upgoing IC86 events, "
          f"{counts.sum():,.0f} inside the reco grid")
    return counts


def interval(curve, level):
    """Crossing points of the profile on :data:`R_GRID`; see :func:`profile_interval`."""
    return profile_interval(R_GRID, curve, level)


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
    metadata = {"r_true": FC_R_TRUE, "n_toys": n_toys, "seed": seed, "priors": priors,
                "version": _FC_VERSION}

    def expectation(r_true):
        profiled = nuisances[int(np.argmin(np.abs(R_GRID - r_true)))]
        params = np.array((0.5 * ANCHORS["phi_mu"][0] / (1.0 - r_true),
                           ANCHORS["gamma"][0], profiled[2], profiled[3]))
        return likelihood.expectation(r_true, *params), params

    def build():
        rng = np.random.default_rng(seed)
        q = np.empty((FC_R_TRUE.size, n_toys))
        for i, r_true in enumerate(FC_R_TRUE):
            mu, params = expectation(r_true)
            for t in range(n_toys):
                toy = rng.poisson(mu).astype(float)
                fixed, warm = likelihood.delta_ll(r_true, data=toy, warm_start=params)
                lowest = fixed
                for r_scan in FC_R_SCAN:
                    value, warm = likelihood.delta_ll(r_scan, data=toy, warm_start=warm,
                                                      use_default_start=False)
                    lowest = min(lowest, value)
                q[i, t] = fixed - lowest
            print(f"  r_true {r_true:.1f}: c68 {np.percentile(q[i], 68.27):5.2f}, "
                  f"c95 {np.percentile(q[i], 95.0):5.2f}")
        return q

    if _FC_CACHE.exists() and not rebuild:
        print(f"  cached toy distributions from {_FC_CACHE.name} (if the metadata match)")
    return cached_toys(_FC_CACHE, metadata, build, rebuild)


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
