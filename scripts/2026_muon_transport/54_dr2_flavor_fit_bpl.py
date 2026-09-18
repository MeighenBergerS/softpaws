"""Example 54 -- the DR2 flavour fit under a single and a broken power law.

Example 51 measures the flavour ratio of the IC86 through-going tracks with
the astrophysical spectrum a single power law anchored to IceCube's 9.5-year
tracks fit. IceCube has since rejected the single power law at 4.7 sigma in
favour of a broken one (arXiv:2507.22233, 2507.22234), and example 53 shows
that the break sits below this window's reach while the index above it is
what the window measures. This example reruns example 51 twice, once under
each spectral hypothesis, and draws every figure with both:

* **SPL** (green): example 51 as is -- a single power law with the index and
  the ``nu_mu`` normalization anchored to the tracks fit,
  ``gamma = 2.37 +- 0.09`` and ``1.44 +- 0.27 x 1e-18`` at 100 TeV.
* **BPL** (pink): IceCube's joint cascades+tracks broken power law. The
  branch below the break (``gamma_1 = 1.31``, break at ``10^4.39`` GeV) is
  held at its published values, since the window cannot see it; the index
  above the break is the free spectral parameter, anchored to
  ``gamma_2 = 2.735 +- 0.071``, and the normalization -- the per-flavour flux
  at 100 TeV on the upper branch, the convention of example 53 -- is
  anchored to ``1.77 +- 0.19 x 1e-18``, widened by the same 5% in quadrature
  for the 1:1:1 assumption inside the published fit.

Everything else is example 51's: the fitted-configuration responses folded
through IceCube's smearing into reconstructed muon energy, the scaled
deviance with its atmospheric priors, the flavour ratio

.. math:: r = \\frac{f_\\tau}{f_\\mu + f_\\tau}

scanned with the nuisances profiled, Feldman-Cousins calibration of the
ratio interval from anchor-consistent toys, the tail jackknife, the tau-flux
profile at the anchored ``nu_mu`` flux, the two likelihood planes, the
flavour triangle, the MESE combination and the Gen2 forecast. The BPL run
reuses example 51's cached inputs and Gen2 responses; only the spectral
shape, the anchors and the toy calibration differ, so the comparison
isolates what the spectral hypothesis does to the flavour result.

The point of the comparison: under the single power law the pre-break index
over-predicts a few hundred TeV in this window by construction, and the
deficit there is what drains the tau channel. With the published
high-energy index the astrophysical template is softer, the anchored
``nu_mu`` flux sits higher, and the tau channel is asked a cleaner question.

Usage
-----
    python scripts/2026_muon_transport/54_dr2_flavor_fit_bpl.py
    python scripts/2026_muon_transport/54_dr2_flavor_fit_bpl.py --toys 200
"""

import argparse
import importlib.util
import pathlib
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from softpaws.data.loader import compute_livetime_s, load_uptime
from softpaws.fluxes import ICECUBE_BPL_2025, broken_power_law_shape

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"
_DEFAULT_DATA_DIR = _HERE.parents[1] / "src" / "softpaws" / "data" / "dataverse_files"

#: Colours from the style file: BPL pink, SPL green.
COLOR_BPL, COLOR_SPL = "#e7298a", "#1b9e77"

#: IceCube's joint cascades+tracks broken power law (arXiv:2507.22234): the
#: per-flavour flux at 100 TeV [1e-18 GeV^-1 cm^-2 s^-1 sr^-1] on the upper
#: branch with its symmetrized error, the index above the break with its
#: symmetrized error, and the fixed lower branch.
BPL_PHI0 = (ICECUBE_BPL_2025.phi0, ICECUBE_BPL_2025.phi0_err)
BPL_GAMMA_2 = (ICECUBE_BPL_2025.gamma_2, ICECUBE_BPL_2025.gamma_2_err)
BPL_GAMMA_1 = ICECUBE_BPL_2025.gamma_1
BPL_LOG_BREAK = ICECUBE_BPL_2025.log10_break_gev

#: Tau-flux scan at the anchored ``nu_mu`` flux [combined-fit units], wider
#: than example 51's: under the broken power law the profile is flatter and
#: the 90% limit lies beyond 6.
PHI_TAU_SCAN = np.linspace(0.0, 12.0, 61)

#: Cache-format version of the toy distributions of this example.
_FC_VERSION = 1


def load_example(stem: str, name: str):
    """Import a sibling example module by file stem."""
    spec = importlib.util.spec_from_file_location(name, _HERE / stem)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: Example 51 supplies the machinery, example 50 the triangle.
_EX51 = load_example("51_dr2_flavor_fit.py", "_example_51")
_EX50 = _EX51._EX50


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=pathlib.Path, default=_DEFAULT_DATA_DIR,
                        help="Root of the DR2 release.")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Rebuild example 51's cached responses and smearing marginal.")
    parser.add_argument("--toys", type=int, default=600,
                        help="Pseudo-experiments per truth point of the "
                             "Feldman-Cousins calibration.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Seed of the pseudo-experiment generator.")
    parser.add_argument("--rebuild-fc", action="store_true",
                        help="Rebuild the cached toy distributions.")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the figures, '54a' through '54g'.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# The two spectral hypotheses
# ---------------------------------------------------------------------------


def spl_shape(energy_gev, gamma):
    """Single power law, unit flux at the 100 TeV pivot."""
    return (energy_gev / 1.0e5) ** (-gamma)


def bpl_shape(energy_gev, gamma_2):
    """IceCube's broken power law, unit flux at 100 TeV on the upper branch."""
    return broken_power_law_shape(energy_gev, gamma_2, BPL_GAMMA_1, BPL_LOG_BREAK)


@dataclass
class Hypothesis:
    """One spectral hypothesis: shape, anchors, colour and caches.

    Attributes
    ----------
    key : str
        ``"spl"`` or ``"bpl"``.
    label : str
        Legend text.
    short : str
        ``"SPL"`` or ``"BPL"``, for crowded legends.
    color : str
        Line colour.
    shape : callable
        ``shape(energy_gev, gamma)``, unit per-flavour flux at 100 TeV.
    anchors : dict
        ``"gamma"`` and ``"phi_mu"`` priors as ``(centre, width)``; the flux
        in units of the combined-fit per-flavour flux.
    """

    key: str
    label: str
    short: str
    color: str
    shape: callable
    anchors: dict
    results: dict = field(default_factory=dict)

    @property
    def fc_cache(self) -> pathlib.Path:
        return _DEFAULT_OUT_DIR / f"54_fc_calibration_{self.key}.npz"


def build_hypotheses() -> list[Hypothesis]:
    """The two hypotheses, SPL first."""
    phi_width = float(np.hypot(BPL_PHI0[1], 0.05 * BPL_PHI0[0]))
    unit = _EX51.PIVOT_PHI0 * 1.0e18
    return [
        Hypothesis("spl", "single power law", "SPL", COLOR_SPL, spl_shape,
                   dict(_EX51.ANCHORS)),
        Hypothesis("bpl", "broken power law", "BPL", COLOR_BPL, bpl_shape,
                   {"gamma": BPL_GAMMA_2,
                    "phi_mu": (BPL_PHI0[0] / unit, phi_width / unit)}),
    ]


class ShapedLikelihood(_EX51.RecoLikelihood):
    """Example 51's likelihood with a pluggable astrophysical shape.

    The free spectral parameter keeps example 51's name ``gamma`` and its
    bounds and cache grid; for the broken power law it is the index above
    the break.
    """

    def __init__(self, shape, *args, **kwargs):
        self._shape = shape
        super().__init__(*args, **kwargs)

    def _astro_direct(self, channel, gamma):
        energy = 10.0**_EX51.LOG10_E_GRID
        flux = 2.0 * _EX51.PIVOT_PHI0 * self._shape(energy, gamma)[:, None]
        return self._fold(self._true_counts(
            channel, flux * np.ones((1, self._d_omega.size))), channel)


def tau_profile(likelihood):
    """``2 Delta ln L`` on :data:`PHI_TAU_SCAN`, floored at its minimum."""
    curve, warm = [], None
    for phi_tau in PHI_TAU_SCAN:
        value, warm = likelihood.delta_ll_tau(phi_tau, warm)
        curve.append(value)
    curve = np.array(curve)
    return curve - curve.min()


def fc_calibration(hypothesis, likelihood, nuisances, n_toys, seed, rebuild):
    """Example 51's Feldman-Cousins toys for one hypothesis, cached.

    Identical construction: at each truth point of ``FC_R_TRUE`` the toys
    are drawn from an anchor-consistent expectation (``nu_mu`` flux and
    index at the hypothesis's anchors, the tau flux set by the ratio, the
    atmospheric normalizations at the data's profiled values), and the
    statistic is the fixed-ratio fit minus the toy's global minimum on
    ``FC_R_SCAN``.

    Returns
    -------
    q : np.ndarray, shape (FC_R_TRUE.size, n_toys)
        Statistic distributions, one row per truth point.
    """
    ex = _EX51
    anchors = hypothesis.anchors
    priors = np.array([ex.CONV_PRIOR, ex.PROMPT_PRIOR, anchors["gamma"], anchors["phi_mu"]])
    cache = hypothesis.fc_cache
    if cache.exists() and not rebuild:
        saved = np.load(cache)
        if (int(saved["n_toys"]) == n_toys and int(saved["seed"]) == seed
                and np.array_equal(saved["r_true"], ex.FC_R_TRUE)
                and np.allclose(saved["priors"], priors)
                and int(saved["version"]) == _FC_VERSION
                and int(saved["version51"]) == ex._FC_VERSION):
            print(f"  cached toy distributions from {cache.name}")
            return saved["q"]
    rng = np.random.default_rng(seed)
    q = np.empty((ex.FC_R_TRUE.size, n_toys))
    for i, r_true in enumerate(ex.FC_R_TRUE):
        profiled = nuisances[int(np.argmin(np.abs(ex.R_GRID - r_true)))]
        params = np.array((0.5 * anchors["phi_mu"][0] / (1.0 - r_true),
                           anchors["gamma"][0], profiled[2], profiled[3]))
        mu = likelihood.expectation(r_true, *params)
        for t in range(n_toys):
            toy = rng.poisson(mu).astype(float)
            fixed, warm = likelihood.delta_ll(r_true, data=toy, warm_start=params)
            lowest = fixed
            for r_scan in ex.FC_R_SCAN:
                value, warm = likelihood.delta_ll(r_scan, data=toy, warm_start=warm,
                                                  use_default_start=False)
                lowest = min(lowest, value)
            q[i, t] = fixed - lowest
        print(f"  r_true {r_true:.1f}: c68 {np.percentile(q[i], 68.27):5.2f}, "
              f"c95 {np.percentile(q[i], 95.0):5.2f}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, r_true=ex.FC_R_TRUE, q=q, n_toys=n_toys, seed=seed, priors=priors,
             version=_FC_VERSION, version51=ex._FC_VERSION)
    return q


# ---------------------------------------------------------------------------
# Running one hypothesis
# ---------------------------------------------------------------------------


def run_hypothesis(hyp, inputs, args) -> None:
    """Example 51's full analysis under one hypothesis; results on ``hyp``."""
    ex = _EX51
    enu_edges, dec_edges, marginal, responses, atmos, livetime_s, data_counts, gen2_resp = inputs
    print(f"\n=== {hyp.label} ===")
    print(f"  anchors: gamma {hyp.anchors['gamma'][0]:.3f} +- {hyp.anchors['gamma'][1]:.3f}, "
          f"nu_mu flux {hyp.anchors['phi_mu'][0] * ex.PIVOT_PHI0 * 1e18:.2f} +- "
          f"{hyp.anchors['phi_mu'][1] * ex.PIVOT_PHI0 * 1e18:.2f} x 1e-18 at 100 TeV "
          f"({hyp.anchors['phi_mu'][0]:.2f} +- {hyp.anchors['phi_mu'][1]:.2f} x combined fit)")
    like = ShapedLikelihood(hyp.shape, enu_edges, marginal, responses, atmos, dec_edges,
                            livetime_s, data_counts, anchors=hyp.anchors)
    res = hyp.results
    res["likelihood"] = like

    print("Profiling the flavour ratio against the data ...")
    curve, nuisances = like.profile()
    i_best = int(np.argmin(curve))
    r_best = ex.R_GRID[i_best]
    norm, gamma, a_conv, a_prompt = nuisances[i_best]
    res.update(curve=curve, nuisances=nuisances, r_best=float(r_best), best=nuisances[i_best])
    w68, w95 = ex.interval(curve, 1.0), ex.interval(curve, 3.84)
    print(f"  best fit r = {r_best:.2f}, Wilks 68% [{w68[0]:.2f}, {w68[1]:.2f}], "
          f"95% [{w95[0]:.2f}, {w95[1]:.2f}]")
    print(f"  2 Delta lnL: r=0 at {curve[0]:.2f}, r=0.5 at "
          f"{curve[int(np.argmin(np.abs(ex.R_GRID - 0.5)))]:.2f}, r=1 at {curve[-1]:.2f}")
    print(f"  nuisances at best fit: astro norm {norm:.2f} x combined fit "
          f"(nu_mu flux {2.0 * norm * (1.0 - r_best):.2f}, nu_tau flux "
          f"{2.0 * norm * r_best:.2f}), gamma {gamma:.2f}, atm conv {a_conv:.2f}, "
          f"prompt {a_prompt:.2f}")
    predicted = like.expectation(r_best, *nuisances[i_best]).sum()
    print(f"  predicted {predicted:,.0f} events against {like.data.sum():,.0f}")

    print(f"Calibrating the interval with {args.toys} pseudo-experiments per truth point ...")
    q = fc_calibration(hyp, like, nuisances, args.toys, args.seed,
                       args.rebuild_fc or args.rebuild_cache)
    c68 = np.percentile(q, 68.27, axis=1)
    c95 = np.percentile(q, 95.0, axis=1)
    fc68 = ex.interval(curve, np.interp(ex.R_GRID, ex.FC_R_TRUE, c68))
    fc95 = ex.interval(curve, np.interp(ex.R_GRID, ex.FC_R_TRUE, c95))
    res.update(c68=c68, c95=c95, fc68=fc68, fc95=fc95,
               p_taufree=float(np.mean(q[0] >= curve[0])))
    print(f"  FC 68% [{fc68[0]:.2f}, {fc68[1]:.2f}], 95% [{fc95[0]:.2f}, {fc95[1]:.2f}]")
    print(f"  tau-free (r = 0) p-value: {res['p_taufree']:.2f}")

    print("Tail jackknife, one event removed per row:")
    for log_e, j, r_hat, d0 in ex.jackknife_tail(like):
        print(f"  without 10^{log_e:.2f} GeV in dec [{dec_edges[j]:.1f}, "
              f"{dec_edges[j + 1]:.1f}]: r_hat {r_hat:.2f}, 2 Delta lnL(r=0) {d0:.2f}")

    print("Profiling the tau flux at the anchored nu_mu flux ...")
    tau_curve = tau_profile(like)
    i_tau = int(np.argmin(tau_curve))
    inside = PHI_TAU_SCAN[tau_curve <= 1.0]
    upper90 = PHI_TAU_SCAN[tau_curve <= 2.71].max()
    edge = " (scan edge)" if upper90 >= PHI_TAU_SCAN[-1] else ""
    _, p_tau = like.delta_ll_tau(PHI_TAU_SCAN[i_tau])
    tau_yield = 0.5 * like._astro("tau", p_tau[1]).sum()
    phi_mu = hyp.anchors["phi_mu"][0]
    res.update(tau_curve=tau_curve, tau_best=float(PHI_TAU_SCAN[i_tau]),
               tau_68=(float(inside.min()), float(inside.max())), tau_90=float(upper90),
               tau_tracks_90=float(upper90 * tau_yield))
    print(f"  tau flux: best {res['tau_best']:.2f}, 68% [{inside.min():.2f}, "
          f"{inside.max():.2f}], < {upper90:.2f}{edge} at 90% (x combined fit; std. osc. expects "
          f"{phi_mu * ex.R_MIN_STD / (1 - ex.R_MIN_STD):.2f}-"
          f"{phi_mu * ex.R_MAX / (1 - ex.R_MAX):.2f})")
    print(f"  in tracks: best {res['tau_best'] * tau_yield:.0f}, "
          f"< {upper90 * tau_yield:.0f} at 90%; profiled nu_mu flux {p_tau[0]:.2f}, "
          f"gamma {p_tau[1]:.2f}, conv {p_tau[2]:.2f}, prompt {p_tau[3]:.2f}")

    print("Scanning the (phi_mu, phi_tau) plane ...")
    plane = like.flux_plane()
    free, phys = ex.plane_minima(plane, ex.PHI_MU_GRID, ex.PHI_TAU_GRID,
                                 ex.R_MAX / (1.0 - ex.R_MAX))
    res.update(flux_plane=plane)
    print(f"  minimum: phi_mu {ex.PHI_MU_GRID[free[0]]:.2f}, phi_tau "
          f"{ex.PHI_TAU_GRID[free[1]]:.2f} (x combined fit); inside the std. osc. ceiling: "
          f"phi_mu {ex.PHI_MU_GRID[phys[0]]:.2f}, phi_tau {ex.PHI_TAU_GRID[phys[1]]:.2f}, "
          f"2 Delta lnL {plane[phys]:.2f}")
    print("Scanning the (N_mu, N_tau) count plane ...")
    counts_plane = like.count_plane()
    yield_ratio = like.channel_yield_ratio(ex.PIVOT_GAMMA)
    free_n, phys_n = ex.plane_minima(counts_plane, ex.N_MU_GRID, ex.N_TAU_GRID,
                                     ex.R_MAX / (1.0 - ex.R_MAX) * yield_ratio)
    res.update(count_plane=counts_plane)
    print(f"  tau/mu tracks per unit flux at gamma {ex.PIVOT_GAMMA}: {yield_ratio:.3f}")
    print(f"  minimum: {ex.N_MU_GRID[free_n[0]]:.0f} nu_mu + {ex.N_TAU_GRID[free_n[1]]:.0f} "
          f"tau tracks; inside the ceiling: {ex.N_MU_GRID[phys_n[0]]:.0f} + "
          f"{ex.N_TAU_GRID[phys_n[1]]:.0f}, 2 Delta lnL {counts_plane[phys_n]:.2f}")

    print(f"Gen2 forecast: {_EX50.FORECAST_YEARS:g} yr Asimov at 1:1:1 under the same "
          "anchors ...")
    gen2 = ShapedLikelihood(hyp.shape, enu_edges, marginal, gen2_resp, atmos, dec_edges,
                            _EX50.FORECAST_YEARS * 365.25 * 86400.0,
                            np.zeros_like(data_counts), anchors=hyp.anchors)
    asimov = gen2.expectation(0.5, phi_mu, hyp.anchors["gamma"][0], 1.0, 1.0)
    curve_gen2, _ = gen2.profile(data=asimov)
    lo, hi = ex.interval(curve_gen2, 1.0)
    astro_gen2 = (asimov - gen2.expectation(0.5, 0.0, hyp.anchors["gamma"][0], 1.0, 1.0)).sum()
    res.update(curve_gen2=curve_gen2)
    print(f"  Gen2 tracks alone: {astro_gen2:.0f} astro tracks in the window, Wilks 68% "
          f"[{lo:.2f}, {hi:.2f}], 2 Delta lnL(r=0) {curve_gen2[0]:.2f}, r=1 {curve_gen2[-1]:.2f}")


# ---------------------------------------------------------------------------
# Figures, each with both hypotheses
# ---------------------------------------------------------------------------


def _save(fig, out_dir: pathlib.Path, stem: str) -> None:
    """Write one figure to ``out_dir`` as both PDF and PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {path.resolve()}")
    plt.close(fig)


def figure_profile(hyps, out_dir) -> None:
    """Both ratio profiles with their calibrated thresholds."""
    ex = _EX51
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.0, 3.0))
        ax.axvspan(ex.R_MIN_STD, ex.R_MAX, color="0.93", zorder=0, label="std. osc. band")
        top = min(max(1.3, 1.25 * max(max(h.results["curve"].max(), h.results["c95"].max())
                                       for h in hyps)), 12.0)
        for level, note in ((1.0, r"$68\%$"), (3.84, r"$95\%$")):
            if level < top:
                ax.axhline(level, color="0.75", lw=0.7, ls=":")
                ax.text(ex.R_GRID[-1] * 1.005, level, note, fontsize=7, color="0.45",
                        va="center")
        for h in hyps:
            ax.plot(ex.FC_R_TRUE, h.results["c68"], color=h.color, lw=0.8, ls="--")
            ax.plot(ex.FC_R_TRUE, h.results["c95"], color=h.color, lw=0.8, ls="-.")
            ax.plot(ex.R_GRID, h.results["curve"], color=h.color, lw=1.3, label=h.label)
        handles, labels = ax.get_legend_handles_labels()
        handles += [Line2D([], [], color="0.4", lw=0.8, ls="--"),
                    Line2D([], [], color="0.4", lw=0.8, ls="-.")]
        labels += [r"FC $68\%$ threshold", r"FC $95\%$ threshold"]
        ax.set_xlim(0.0, ex.R_GRID[-1])
        ax.set_ylim(0.0, top)
        ax.set_xlabel(r"$f_\tau \,/\, (f_\mu + f_\tau)$ at Earth")
        ax.set_ylabel(r"$2\,\Delta\ln L$ (IC86 data, anchored)")
        ax.legend(handles, labels, fontsize=6, frameon=False, loc="upper left")
        _save(fig, out_dir, "54a_dr2_profile")


def figure_spectra(hyps, out_dir) -> None:
    """Data against the best-fit components, one panel per hypothesis."""
    ex = _EX51
    with plt.style.context(str(_STYLE)):
        fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.0), sharey=True,
                                 gridspec_kw={"wspace": 0.06})
        for ax, h in zip(axes, hyps):
            like = h.results["likelihood"]
            r, (norm, gamma, a_conv, a_prompt) = h.results["r_best"], h.results["best"]
            centers = 0.5 * (ex.RECO_EDGES[:-1] + ex.RECO_EDGES[1:])[like._window]
            components = {
                "atm. conventional": (a_conv * like._folded_conv, "0.65", "-"),
                "atm. prompt": (a_prompt * like._folded_prompt, "0.4", "-"),
                r"astro $\nu_\mu$": (norm * (1.0 - r) * like._astro("mu", gamma), h.color, ":"),
                r"astro $\nu_\tau \to \mu$": (norm * r * like._astro("tau", gamma),
                                              h.color, "--"),
            }
            total = np.zeros_like(centers)
            for name, (comp, color, ls) in components.items():
                summed = comp.sum(axis=1)
                total += summed
                if summed.max() <= 0.0:
                    name += r" ($= 0$ at best fit)"
                ax.step(centers, summed, where="mid", color=color, lw=1.0, ls=ls, label=name)
            ax.step(centers, total, where="mid", color=h.color, lw=1.4,
                    label=f"total, {h.label}")
            observed = like.data.sum(axis=1)
            ax.errorbar(centers, observed, yerr=np.sqrt(observed), fmt="o", color="k",
                        ms=2.2, lw=0.8, capsize=1.5, label="IC86 data", zorder=5)
            ax.set_yscale("log")
            ax.set_ylim(0.3, None)
            ax.set_xlabel(r"$\log_{10}(E_{\rm reco}\,/\,\mathrm{GeV})$")
            ax.legend(fontsize=6, frameon=False, loc="upper right")
        axes[0].set_ylabel("events per bin, upgoing")
        _save(fig, out_dir, "54b_dr2_spectra")


def figure_triangle(hyps, out_dir) -> None:
    """Both calibrated wedges and best-fit rays on the flavour triangle."""
    ex = _EX51
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(4.4, 4.2))
        corners = _EX50.draw_triangle_frame(ax)
        ours, ours_labels = [], []
        for h in hyps:
            r_best = h.results["r_best"]
            for (lo, hi), level, alpha, zorder in ((h.results["fc68"], "68", 0.18, 2.2),
                                                   (h.results["fc95"], "95", 0.07, 2.0)):
                whole = lo <= ex.R_GRID[0] and hi >= ex.R_GRID[-1] - 1.0e-9
                note = " (whole range)" if whole else ""
                ax.add_patch(plt.Polygon(
                    [corners["e"], _EX50._ternary_xy(0.0, 1.0 - lo, lo),
                     _EX50._ternary_xy(0.0, 1.0 - hi, hi)], closed=True,
                    facecolor=h.color, edgecolor=h.color, lw=0.6, alpha=alpha, zorder=zorder))
                ours.append(Patch(facecolor=h.color, alpha=alpha + 0.1, edgecolor=h.color,
                                  lw=0.6))
                ours_labels.append(rf"{h.label} ${level}\%${note}")
            tip = _EX50._ternary_xy(0.0, 1.0 - r_best, r_best)
            ax.plot(*zip(corners["e"], tip), color=h.color, lw=1.1, ls="--", zorder=4)
            ax.plot(*tip, marker="*", color=h.color, ms=7, ls="none", zorder=5)
            ours.append(Line2D([], [], color=h.color, lw=1.1, ls="--", marker="*", ms=7))
            ours_labels.append(rf"best fit $\hat r = {r_best:.2f}$")
        for r in (ex.R_MIN_STD, ex.R_MAX):
            bound = _EX50._ternary_xy(0.0, 1.0 - r, r)
            ax.plot(*zip(corners["e"], bound), color="0.4", lw=0.8, ls=":", zorder=3)
        ours.append(Line2D([], [], color="0.4", lw=0.8, ls=":"))
        ours_labels.append(rf"std. osc. band, $r \in [{ex.R_MIN_STD:.2f}, {ex.R_MAX:.2f}]$")
        _EX50.draw_published_curves(ax)
        handles, labels = ax.get_legend_handles_labels()
        measured = ax.legend(ours, ours_labels, fontsize=7, frameon=False, loc="upper left",
                             bbox_to_anchor=(0.0, 1.20), handlelength=1.4, labelspacing=0.25)
        ax.add_artist(measured)
        ax.legend(handles, labels, fontsize=8, frameon=False, loc="upper right",
                  bbox_to_anchor=(1.08, 1.16), handlelength=1.6, labelspacing=0.3)
        _save(fig, out_dir, "54c_dr2_triangle")


def _draw_planes(ax, hyps, key, x_grid, y_grid, slopes, reference, y_max) -> None:
    """Shared body of the flux and count planes, both hypotheses."""
    x = np.array([0.0, x_grid[-1]])
    ax.fill_between(x, slopes[0] * x, slopes[1] * x, color="#7570b3", alpha=0.18, lw=0,
                    zorder=0, label="std. osc. band")
    for h in hyps:
        plane = h.results[key]
        free, phys = _EX51.plane_minima(plane, x_grid, y_grid, slopes[1])
        ax.contour(x_grid, y_grid, plane.T, levels=[2.30, 5.99], colors=h.color,
                   linestyles=["-", "--"], linewidths=[1.2, 0.9], zorder=2)
        x_free, y_free = x_grid[free[0]], y_grid[free[1]]
        if y_free <= y_max:
            ax.plot(x_free, y_free, "o", mfc="none", color=h.color, ms=5, zorder=4)
        else:
            ax.annotate(f"unconstrained\nminimum at\n({x_free:.2g}, {y_free:.2g})",
                        xy=(x_free + 0.02 * x_grid[-1], 0.97 * y_max),
                        xytext=(x_free + 0.07 * x_grid[-1], 0.60 * y_max), fontsize=6,
                        color=h.color, ha="left", va="top",
                        arrowprops=dict(arrowstyle="->", color=h.color, lw=0.8), zorder=4)
        ax.plot(x_grid[phys[0]], y_grid[phys[1]], "*", color=h.color, ms=8, zorder=5,
                label=h.label)
    ax.plot(*reference, "+", color="k", ms=7, mew=1.2, zorder=5, label="combined fit, 1:1:1")
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color="0.4", lw=1.2), Line2D([], [], color="0.4", lw=0.9, ls="--"),
                Line2D([], [], color="0.4", marker="o", mfc="none", ls="none", ms=5),
                Line2D([], [], color="0.4", marker="*", ls="none", ms=8)]
    labels += [r"$68\%$", r"$95\%$", "unconstrained minimum",
               rf"best fit, $r \leq {_EX51.R_MAX:.2f}$"]
    ax.set_xlim(0.0, x_grid[-1])
    ax.set_ylim(0.0, y_max)
    ax.legend(handles, labels, fontsize=6, frameon=False, loc="upper right")


def figure_flux_plane(hyps, out_dir) -> None:
    """The ``(phi_mu, phi_tau)`` profiles of both hypotheses."""
    ex = _EX51
    slopes = tuple(r / (1.0 - r) for r in (ex.R_MIN_STD, ex.R_MAX))
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.0, 3.0))
        _draw_planes(ax, hyps, "flux_plane", ex.PHI_MU_GRID, ex.PHI_TAU_GRID, slopes,
                     (1.0, 1.0), ex.PHI_TAU_PLOT_MAX)
        ax.set_xlabel(r"$\Phi_{\nu_\mu} \,/\, \Phi_{\rm combined}$")
        ax.set_ylabel(r"$\Phi_{\nu_\tau} \,/\, \Phi_{\rm combined}$")
        _save(fig, out_dir, "54d_dr2_flux_plane")


def figure_count_plane(hyps, out_dir) -> None:
    """The same likelihoods in expected-track units.

    The band and the reference point use the SPL yield ratio at the pivot
    index, as in example 51; the BPL's differs by a few percent.
    """
    ex = _EX51
    like = hyps[0].results["likelihood"]
    yield_ratio = like.channel_yield_ratio(ex.PIVOT_GAMMA)
    slopes = tuple(r / (1.0 - r) * yield_ratio for r in (ex.R_MIN_STD, ex.R_MAX))
    reference = (0.5 * like._astro("mu", ex.PIVOT_GAMMA).sum(),
                 0.5 * like._astro("tau", ex.PIVOT_GAMMA).sum())
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.0, 3.0))
        _draw_planes(ax, hyps, "count_plane", ex.N_MU_GRID, ex.N_TAU_GRID, slopes,
                     reference, ex.N_TAU_GRID[-1])
        ax.set_xlabel(r"astro $\nu_\mu$ tracks in the window")
        ax.set_ylabel(r"astro $\nu_\tau \to \mu$ tracks in the window")
        _save(fig, out_dir, "54e_dr2_count_plane")


def figure_tau_profile(hyps, out_dir) -> None:
    """Both tau-flux profiles at their anchored ``nu_mu`` fluxes.

    Each hypothesis has its own standard-oscillation band, since the
    anchored ``nu_mu`` flux differs; both are drawn in the hypothesis colour.
    """
    ex = _EX51
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.0, 3.0))
        top = min(max(4.5, 1.1 * max(h.results["tau_curve"].max() for h in hyps)), 12.0)
        for h in hyps:
            phi_mu = h.anchors["phi_mu"][0]
            lo, hi = (phi_mu * r / (1.0 - r) for r in (ex.R_MIN_STD, ex.R_MAX))
            ax.axvspan(lo, hi, color=h.color, alpha=0.12, lw=0, zorder=0)
            ax.plot(PHI_TAU_SCAN, h.results["tau_curve"], color=h.color, lw=1.3,
                    label=h.label)
        for level, note in ((1.0, r"$68\%$"), (2.71, r"$90\%$")):
            ax.axhline(level, color="0.75", lw=0.7, ls=":")
            ax.text(PHI_TAU_SCAN[-1] * 1.01, level, note, fontsize=7, color="0.45",
                    va="center")
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Patch(facecolor="0.6", alpha=0.3, lw=0))
        labels.append("std. osc. band (per hypothesis)")
        ax.set_xlim(0.0, PHI_TAU_SCAN[-1])
        ax.set_ylim(0.0, top)
        ax.set_xlabel(r"$\Phi_{\nu_\tau} \,/\, \Phi_{\rm combined}$")
        ax.set_ylabel(r"$2\,\Delta\ln L$ (IC86 data, anchored)")
        ax.legend(handles, labels, fontsize=6, frameon=False, loc="upper left")
        _save(fig, out_dir, "54f_dr2_tau_profile")


def figure_combined_triangle(hyps, out_dir) -> dict:
    """MESE combined with this work under both hypotheses, plus Gen2.

    The IC86 combinations are filled at 68% with the 95% dashed; the Gen2
    forecasts are the dotted 68% contours in the same colours.

    Returns
    -------
    best : dict of str -> tuple of float
        Combined best-fit fractions ``(f_e, f_mu, f_tau)`` per hypothesis key.
    """
    ex = _EX51
    best = {}
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(4.4, 4.2))
        _EX50.draw_triangle_frame(ax)
        ours, ours_labels = [], []
        for h in hyps:
            x, y, total = ex.combined_surface(h.results["curve"])
            _, _, total_gen2 = ex.combined_surface(h.results["curve_gen2"])
            i, j = np.unravel_index(int(np.nanargmin(total)), total.shape)
            best[h.key] = tuple(float(v) for v in ex._fractions_from_xy(x[i, j], y[i, j]))
            ax.contourf(x, y, total, levels=[0.0, ex.LEVEL68_2D], colors=[h.color],
                        alpha=0.15, zorder=2)
            ax.contour(x, y, total, levels=[ex.LEVEL68_2D], colors=h.color, linewidths=1.3,
                       zorder=4)
            ax.contour(x, y, total, levels=[ex.LEVEL95_2D], colors=h.color, linestyles="--",
                       linewidths=1.0, zorder=4)
            ax.contour(x, y, total_gen2, levels=[ex.LEVEL68_2D], colors=h.color,
                       linestyles=":", linewidths=1.1, zorder=5)
            ax.plot(x[i, j], y[i, j], marker="*", color=h.color, ms=8, ls="none", zorder=6)
            ours += [Patch(facecolor=h.color, alpha=0.3, edgecolor=h.color, lw=1.3),
                     Line2D([], [], color=h.color, ls="--", lw=1.0),
                     Line2D([], [], color=h.color, ls=":", lw=1.1)]
            ours_labels += [rf"MESE + IC86 tracks $68\%$, {h.short}",
                            rf"MESE + IC86 tracks $95\%$, {h.short}",
                            rf"MESE + Gen2 forecast $68\%$, {h.short}"]
        _EX50.draw_published_curves(ax, skip=("icecube2022",))
        handles, labels = ax.get_legend_handles_labels()
        first = ax.legend(ours, ours_labels, fontsize=7, frameon=False, loc="upper left",
                          bbox_to_anchor=(-0.04, 1.20), handlelength=1.4, labelspacing=0.25)
        ax.add_artist(first)
        ax.legend(handles, labels, fontsize=7, frameon=False, loc="upper right",
                  bbox_to_anchor=(1.10, 1.20), handlelength=1.6, labelspacing=0.25)
        _save(fig, out_dir, "54g_dr2_combined_triangle")
    return best


def main() -> None:
    args = parse_args()
    ex = _EX51
    print("Loading examples 35, 45 and 46 ...")
    ex35 = load_example("35_point_source_effective_area.py", "_example_35")
    ex45 = load_example("45_first_principles_reach.py", "_example_45")
    ex46 = load_example("46_declination_resolved_reach.py", "_example_46")

    enu_edges, dec_edges, marginal, responses = ex.fit_inputs(
        ex35, ex45, ex46, args.data_dir, args.rebuild_cache)
    atmos = ex.atmospheric_fluxes(dec_edges)
    livetime_s = sum(
        compute_livetime_s(load_uptime(args.data_dir / "uptime" / f"{s}_exp.csv"))
        for s in ex.IC86_SEASONS)
    print(f"  IC86 livetime: {livetime_s / (365.25 * 86400.0):.2f} yr")
    data_counts = ex.binned_events(args.data_dir, dec_edges)
    gen2_resp = ex.gen2_responses(ex35, ex45, ex46, dec_edges, args.rebuild_cache)
    print(f"  standard oscillations: r in [{ex.R_MIN_STD:.2f}, {ex.R_MAX:.2f}] at Earth")
    print(f"  BPL lower branch held at gamma_1 {BPL_GAMMA_1}, break 10^{BPL_LOG_BREAK} GeV "
          f"= {10**BPL_LOG_BREAK / 1e3:.0f} TeV")

    inputs = (enu_edges, dec_edges, marginal, responses, atmos, livetime_s, data_counts,
              gen2_resp)
    hyps = build_hypotheses()
    for h in hyps:
        run_hypothesis(h, inputs, args)

    print("\nSummary, SPL against BPL:")
    for h in hyps:
        r = h.results
        print(f"  {h.label}: r_hat {r['r_best']:.2f}, FC 68% [{r['fc68'][0]:.2f}, "
              f"{r['fc68'][1]:.2f}], 95% [{r['fc95'][0]:.2f}, {r['fc95'][1]:.2f}], "
              f"2 Delta lnL(r=1) {r['curve'][-1]:.2f}, tau flux < {r['tau_90']:.2f} "
              f"(< {r['tau_tracks_90']:.0f} tracks) at 90%, gamma {r['best'][1]:.2f}")

    print()
    figure_profile(hyps, args.out_dir)
    figure_spectra(hyps, args.out_dir)
    figure_triangle(hyps, args.out_dir)
    figure_flux_plane(hyps, args.out_dir)
    figure_count_plane(hyps, args.out_dir)
    figure_tau_profile(hyps, args.out_dir)
    best = figure_combined_triangle(hyps, args.out_dir)
    for h in hyps:
        b = best[h.key]
        print(f"combined MESE + IC86 tracks best fit, {h.label}: f_e {b[0]:.2f}, "
              f"f_mu {b[1]:.2f}, f_tau {b[2]:.2f}")
    print(f"MESE alone: {_EX50.ICECUBE_BEST_FIT[0]:.2f}, {_EX50.ICECUBE_BEST_FIT[1]:.2f}, "
          f"{_EX50.ICECUBE_BEST_FIT[2]:.2f}")


if __name__ == "__main__":
    main()
