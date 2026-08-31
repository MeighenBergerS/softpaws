"""Example 53 -- single against broken power law on the DR2 tracks.

IceCube rejects a single power law for the diffuse flux at 4.7 sigma in
favour of a broken one (arXiv:2507.22233, 2507.22234), with a break near
30 TeV: a hard index below it, ``gamma_2 = 2.74`` above. This example asks
what the same question looks like through this project's method -- example
51's fold of the fitted-configuration responses through IceCube's smearing
into reconstructed muon energy, fit to the binned IC86 events of
IceTracks-DR2 above ``10^4.25`` GeV -- and puts the two fits next to the
published spectra.

Both spectra are fit with the composition fixed at 1:1:1, both channels
folded, the atmospheric normalizations free under the same priors as
example 51 and no external anchors: a single power law in (normalization,
index) and a broken one in (normalization, index below, index above, break
energy), with the normalization the per-flavour flux at 100 TeV on the
upper branch and the lower branch continuous at the break. The published
broken power laws are drawn with the same convention. The break IceCube
sees sits below this window's reach, so the comparison is honest about
what tracks above ~50 TeV can and cannot say: the index above the break,
which they measure, against the location of the break, which they do not.

Usage
-----
    python examples/53_spectral_shape_tracks.py
    python examples/53_spectral_shape_tracks.py --published-response
"""

import argparse
import importlib.util
import itertools
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

from softpaws.data.loader import compute_livetime_s, load_uptime

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"
_DEFAULT_DATA_DIR = _HERE.parent / "src" / "softpaws" / "data" / "dataverse_files"

#: Pivot of every normalization [GeV]: per-flavour ``nu + nubar`` flux at
#: 100 TeV in units of 1e-18 GeV^-1 cm^-2 s^-1 sr^-1.
PIVOT_GEV = 1.0e5

#: IceCube's published spectra in that unit: the 9.5-year northern-tracks
#: single power law (arXiv:2111.10299) and the two broken power laws of
#: arXiv:2507.22234, joint cascades+tracks and MESE, as (phi0, gamma_1,
#: gamma_2, log10 E_break).
PUBLISHED_SPL = {"IceCube tracks 9.5 yr": (1.44, 2.37)}
PUBLISHED_BPL = {
    "IceCube cascades+tracks": (1.77, 1.31, 2.735, 4.39),
    "IceCube MESE": (2.28, 1.72, 2.839, 4.524),
}

#: Parameter bounds of the free fits.
GAMMA_RANGE = (1.0, 4.5)
LOG_BREAK_RANGE = (4.0, 7.0)

#: Profile grids for the 68% intervals.
GAMMA_SCAN = np.linspace(1.5, 4.0, 26)
LOG_BREAK_SCAN = np.linspace(4.0, 7.0, 16)

#: Colours: this work's two fits, and the published spectra.
COLOR_SPL, COLOR_BPL = "#7570b3", "#e7298a"
COLOR_PUBLISHED = {"IceCube tracks 9.5 yr": "0.35", "IceCube cascades+tracks": "#1b9e77",
                   "IceCube MESE": "#d95f02"}


def load_example(stem: str, name: str):
    """Import a sibling example module by file stem."""
    spec = importlib.util.spec_from_file_location(name, _HERE / stem)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=pathlib.Path, default=_DEFAULT_DATA_DIR,
                        help="Root of the DR2 release.")
    parser.add_argument("--published-response", action="store_true",
                        help="Use IceCube's published IC86 effective area for the "
                             "nu_mu channel instead of the model's.")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the figures, '53a' and '53b'.")
    return parser.parse_args()


def spl(energy_gev, phi0, gamma):
    """Single power law [1e-18 GeV^-1 cm^-2 s^-1 sr^-1 per flavour]."""
    return phi0 * (energy_gev / PIVOT_GEV) ** (-gamma)


def bpl(energy_gev, phi0, gamma_1, gamma_2, log_break):
    """Broken power law, pivot on the upper branch, continuous at the break."""
    e_break = 10.0 ** log_break
    above = phi0 * (energy_gev / PIVOT_GEV) ** (-gamma_2)
    below = phi0 * (e_break / PIVOT_GEV) ** (-gamma_2) * (energy_gev / e_break) ** (-gamma_1)
    return np.where(energy_gev >= e_break, above, below)


def published_response(ex51, data_dir, dec_edges, responses):
    """IceCube's published IC86 effective area on the smearing bands.

    The tau channel is rescaled by the same published-over-model ratio.
    """
    raw = np.genfromtxt(data_dir / "irfs" / "IC86_effectiveArea.csv", comments="#")
    e_lo, e_hi, d_lo, d_hi, area = raw.T
    sin_lo, sin_hi = np.sin(np.deg2rad(d_lo)), np.sin(np.deg2rad(d_hi))
    pub = np.zeros((ex51.LOG10_E_GRID.size, dec_edges.size - 1))
    for j in range(dec_edges.size - 1):
        band = np.sin(np.deg2rad(dec_edges[j:j + 2]))
        for k, log_e in enumerate(ex51.LOG10_E_GRID):
            sel = (e_lo <= log_e) & (log_e < e_hi)
            weight = np.clip(np.minimum(sin_hi[sel], band[1])
                             - np.maximum(sin_lo[sel], band[0]), 0.0, None)
            pub[k, j] = (area[sel] * weight).sum() / weight.sum()
    ratio = pub / np.clip(responses["mu"], 1.0e-30, None)
    return {"mu": pub, "tau": responses["tau"] * ratio}


class SpectralFit:
    """SPL and BPL fits of the reconstructed spectrum, composition 1:1:1.

    Parameters
    ----------
    likelihood : RecoLikelihood
        Example 51's likelihood without anchors.
    """

    def __init__(self, ex51, likelihood):
        self._ex51 = ex51
        self._like = likelihood
        self._energy = 10.0 ** ex51.LOG10_E_GRID
        self._n_bands = likelihood._d_omega.size

    def astro(self, shape):
        """Folded astrophysical counts for a per-flavour flux shape."""
        flux = 2.0 * self._ex51.PIVOT_PHI0 * shape[:, None] * np.ones((1, self._n_bands))
        like = self._like
        return 0.5 * (like._fold(like._true_counts("mu", flux), "mu")
                      + like._fold(like._true_counts("tau", flux), "tau"))

    def objective(self, astro, a_conv, a_prompt):
        like = self._like
        mu = a_conv * like._folded_conv + a_prompt * like._folded_prompt + astro
        return like._deviance(mu, 1.0, 2.5, a_conv, a_prompt, like.data)

    def _minimize(self, function, starts):
        best = None
        for x0 in starts:
            trial = minimize(function, x0=np.asarray(x0, dtype=float), method="Nelder-Mead",
                             options={"xatol": 1.0e-4, "fatol": 1.0e-6, "maxiter": 12000})
            if best is None or trial.fun < best.fun:
                best = trial
        return best.fun, best.x

    def fit_spl(self, gamma=None, warm=None):
        """SPL fit; ``gamma`` fixed when given (for the profile)."""
        def function(p):
            norm, g, a_conv, a_prompt = (p[0], gamma, p[1], p[2]) if gamma is not None else p
            if (norm < 0.0 or a_conv <= 0.0 or a_prompt < 0.0
                    or not GAMMA_RANGE[0] <= g <= GAMMA_RANGE[1]):
                return 1.0e12
            return self.objective(norm * self.astro(spl(self._energy, 1.0, g)),
                                  a_conv, a_prompt)
        if gamma is None:
            starts = [(0.5, g0, 1.0, 1.0) for g0 in (2.0, 2.4, 2.8)]
        else:
            starts = [(0.5, 1.0, 1.0)] + ([warm] if warm is not None else [])
        return self._minimize(function, starts)

    def fit_bpl(self, fixed=None, warm=None):
        """BPL fit; ``fixed`` maps parameter index (1, 2 or 3) to a value."""
        fixed = fixed or {}
        free = [i for i in range(4) if i not in fixed]

        def unpack(p):
            full = [None] * 4
            for i, v in zip(free, p[:len(free)]):
                full[i] = v
            for i, v in fixed.items():
                full[i] = v
            return full, p[len(free)], p[len(free) + 1]

        def function(p):
            (norm, g1, g2, log_b), a_conv, a_prompt = unpack(p)
            if norm < 0.0 or a_conv <= 0.0 or a_prompt < 0.0:
                return 1.0e12
            if not (GAMMA_RANGE[0] <= g1 <= GAMMA_RANGE[1]
                    and GAMMA_RANGE[0] <= g2 <= GAMMA_RANGE[1]
                    and LOG_BREAK_RANGE[0] <= log_b <= LOG_BREAK_RANGE[1]):
                return 1.0e12
            return self.objective(norm * self.astro(bpl(self._energy, 1.0, g1, g2, log_b)),
                                  a_conv, a_prompt)

        if not fixed:
            starts = [(0.5, g1, g2, lb, 1.0, 1.0)
                      for g1, g2, lb in itertools.product(
                          (1.5, 2.0, 2.5, 3.5), (2.0, 2.5, 3.0, 3.5), (4.5, 5.0, 5.5, 6.0))]
        else:
            seed = {0: 0.5, 1: 2.5, 2: 2.5, 3: 5.0}
            starts = [[seed[i] for i in free] + [1.0, 1.0]]
            if warm is not None:
                starts.append(warm)
        return self._minimize(function, starts)


def interval_from_profile(grid, values, level=1.0):
    """Best point and 68% crossing points of a profile."""
    values = np.asarray(values) - np.min(values)
    inside = grid[values <= level]
    return grid[int(np.argmin(values))], inside.min(), inside.max()


def window_reach(ex51, likelihood, shape):
    """Neutrino-energy range holding 90% of the window's astrophysical tracks."""
    flux = 2.0 * ex51.PIVOT_PHI0 * shape[:, None] * np.ones((1, likelihood._d_omega.size))
    per_true = np.zeros(likelihood._enu_edges.size - 1)
    for channel in ("mu", "tau"):
        true = likelihood._true_counts(channel, flux)
        marginal = likelihood._marginal_tau if channel == "tau" else likelihood._marginal
        per_true += np.einsum("ij,ijk->i", true, marginal[:, :, likelihood._window])
    cdf = np.cumsum(per_true) / per_true.sum()
    edges = likelihood._enu_edges
    lo = np.interp(0.05, cdf, edges[1:])
    hi = np.interp(0.95, cdf, edges[1:])
    return lo, hi


def _save(fig, out_dir: pathlib.Path, stem: str) -> None:
    """Write one figure to ``out_dir`` as both PDF and PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {path.resolve()}")
    plt.close(fig)


def figure_reco(ex51, likelihood, fit, spl_params, bpl_params, out_dir) -> None:
    """Reconstructed spectrum under both fits, with the data/model ratio."""
    window = likelihood._window
    centers = 0.5 * (ex51.RECO_EDGES[:-1] + ex51.RECO_EDGES[1:])[window]
    data = likelihood.data.sum(axis=1)
    energy = 10.0 ** ex51.LOG10_E_GRID
    n_s, g_s, c_s, p_s = spl_params
    n_b, g1, g2, lb, c_b, p_b = bpl_params
    astro_s = n_s * fit.astro(spl(energy, 1.0, g_s))
    astro_b = n_b * fit.astro(bpl(energy, 1.0, g1, g2, lb))
    atm_s = c_s * likelihood._folded_conv + p_s * likelihood._folded_prompt
    atm_b = c_b * likelihood._folded_conv + p_b * likelihood._folded_prompt
    total_s = (atm_s + astro_s).sum(axis=1)
    total_b = (atm_b + astro_b).sum(axis=1)
    atm_s = atm_s.sum(axis=1)
    with plt.style.context(str(_STYLE)):
        fig, (ax, ratio) = plt.subplots(2, 1, figsize=(3.0, 3.8), sharex=True,
                                        gridspec_kw={"height_ratios": (3, 1.3), "hspace": 0.05})
        ax.step(centers, atm_s, where="mid", color="0.6", lw=0.9, label="atmospheric (SPL fit)")
        ax.step(centers, astro_s.sum(axis=1), where="mid", color=COLOR_SPL, lw=0.9, ls=":",
                label="astro, SPL")
        ax.step(centers, astro_b.sum(axis=1), where="mid", color=COLOR_BPL, lw=0.9, ls=":",
                label="astro, BPL")
        ax.step(centers, total_s, where="mid", color=COLOR_SPL, lw=1.2, label="total, SPL")
        ax.step(centers, total_b, where="mid", color=COLOR_BPL, lw=1.2, label="total, BPL")
        ax.errorbar(centers, data, yerr=np.sqrt(data), fmt="o", color="k", ms=2.2, lw=0.8,
                    capsize=1.5, label="IC86 data", zorder=5)
        ax.set_yscale("log")
        ax.set_ylim(0.3, None)
        ax.set_ylabel("events per bin, upgoing")
        ax.legend(fontsize=6, frameon=False, loc="upper right")
        ok = data > 0
        for total, color, shift in ((total_s, COLOR_SPL, -0.02), (total_b, COLOR_BPL, 0.02)):
            ratio.errorbar(centers[ok] + shift, data[ok] / total[ok],
                           yerr=np.sqrt(data[ok]) / total[ok], fmt="o", color=color,
                           ms=2.2, lw=0.8, capsize=1.5)
        ratio.axhline(1.0, color="0.6", lw=0.7, ls=":")
        ratio.set_ylim(0.0, 2.5)
        ratio.set_xlim(4.25, 7.5)
        ratio.set_xlabel(r"$\log_{10}(E_{\rm reco}\,/\,\mathrm{GeV})$")
        ratio.set_ylabel("data / model")
        _save(fig, out_dir, "53a_spectral_shape_reco")


def figure_flux(spl_params, bpl_params, reach, out_dir) -> None:
    """Per-flavour ``E^2 Phi`` of the fits and the published spectra.

    This work's fits are drawn solid inside the window's reach and faint
    outside it, where they are extrapolations.
    """
    energy = np.logspace(3.0, 8.0, 300)
    scale = energy**2 * 1.0e-18
    n_s, g_s, _, _ = spl_params
    n_b, g1, g2, lb, _, _ = bpl_params
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.0, 3.0))
        ax.axvspan(10.0 ** reach[0], 10.0 ** reach[1], color="0.93", zorder=0,
                   label=r"window reach (90\% of astro tracks)")
        for name, (phi0, gamma) in PUBLISHED_SPL.items():
            ax.plot(energy, scale * spl(energy, phi0, gamma), color=COLOR_PUBLISHED[name],
                    lw=0.9, ls="--", label=f"{name}, SPL")
        for name, params in PUBLISHED_BPL.items():
            ax.plot(energy, scale * bpl(energy, *params), color=COLOR_PUBLISHED[name],
                    lw=0.9, label=f"{name}, BPL")
        inside = (energy >= 10.0 ** reach[0]) & (energy <= 10.0 ** reach[1])
        for curve, color, label in ((spl(energy, n_s * 1.8, g_s), COLOR_SPL, "this work, SPL"),
                                    (bpl(energy, n_b * 1.8, g1, g2, lb), COLOR_BPL,
                                     "this work, BPL")):
            ax.plot(energy, scale * curve, color=color, lw=1.4, alpha=0.3)
            ax.plot(energy[inside], (scale * curve)[inside], color=color, lw=1.6,
                    label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1.0e3, 1.0e8)
        ax.set_ylim(1.0e-9, 3.0e-7)
        ax.set_xlabel(r"$E_\nu$ [GeV]")
        ax.set_ylabel(r"$E^2 \Phi_{\nu+\bar\nu}$ per flavour [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
        ax.legend(fontsize=5.5, frameon=False, loc="upper right")
        _save(fig, out_dir, "53b_spectral_shape_flux")


def main() -> None:
    args = parse_args()
    print("Loading examples 35, 45, 46 and 51 ...")
    ex51 = load_example("51_dr2_flavor_fit.py", "_example_51")
    ex35 = load_example("35_point_source_effective_area.py", "_example_35")
    ex45 = load_example("45_first_principles_reach.py", "_example_45")
    ex46 = load_example("46_declination_resolved_reach.py", "_example_46")

    enu_edges, dec_edges, marginal, responses = ex51.fit_inputs(
        ex35, ex45, ex46, args.data_dir, False)
    if args.published_response:
        print("  nu_mu channel: IceCube's published IC86 effective area")
        responses = published_response(ex51, args.data_dir, dec_edges, responses)
    atmos = ex51.atmospheric_fluxes(dec_edges)
    livetime_s = sum(
        compute_livetime_s(load_uptime(args.data_dir / "uptime" / f"{s}_exp.csv"))
        for s in ex51.IC86_SEASONS)
    data_counts = ex51.binned_events(args.data_dir, dec_edges)
    likelihood = ex51.RecoLikelihood(enu_edges, marginal, responses, atmos, dec_edges,
                                     livetime_s, data_counts)
    fit = SpectralFit(ex51, likelihood)
    print(f"  fit window {ex51.FIT_RECO}: {likelihood.data.sum():,.0f} events")

    print("Fitting the single power law ...")
    obj_s, spl_params = fit.fit_spl()
    print("Fitting the broken power law ...")
    obj_b, bpl_params = fit.fit_bpl()
    delta = obj_s - obj_b
    p_value = chi2.sf(delta, 2)

    print("Profiling the indices and the break ...")
    prof_gamma, warm = [], None
    for g in GAMMA_SCAN:
        value, warm = fit.fit_spl(gamma=g, warm=warm)
        prof_gamma.append(value)
    gamma_best, gamma_lo, gamma_hi = interval_from_profile(GAMMA_SCAN, prof_gamma)
    profiles = {}
    for index, grid in ((1, GAMMA_SCAN), (2, GAMMA_SCAN), (3, LOG_BREAK_SCAN)):
        values, warm = [], None
        for x in grid:
            value, warm = fit.fit_bpl(fixed={index: x}, warm=warm)
            values.append(value)
        profiles[index] = interval_from_profile(grid, values)

    n_s, g_s, c_s, p_s = spl_params
    n_b, g1, g2, lb, c_b, p_b = bpl_params
    print(f"\n  SPL: flux at 100 TeV {n_s * 1.8:.2f}, gamma {g_s:.2f} "
          f"(68% [{gamma_lo:.2f}, {gamma_hi:.2f}]), conv {c_s:.2f}, prompt {p_s:.2f}")
    print(f"  BPL: flux at 100 TeV {bpl(PIVOT_GEV, n_b * 1.8, g1, g2, lb):.2f}, "
          f"gamma_1 {g1:.2f} (68% [{profiles[1][1]:.2f}, "
          f"{profiles[1][2]:.2f}]), gamma_2 {g2:.2f} (68% [{profiles[2][1]:.2f}, "
          f"{profiles[2][2]:.2f}]), break 10^{lb:.2f} GeV = {10**lb / 1e3:.0f} TeV "
          f"(68% [10^{profiles[3][1]:.2f}, 10^{profiles[3][2]:.2f}]), conv {c_b:.2f}, "
          f"prompt {p_b:.2f}")
    print(f"  2 Delta lnL (BPL over SPL) = {delta:.2f} for 2 extra parameters: "
          f"p = {p_value:.3f} (Wilks)")
    for name, (phi0, gamma) in PUBLISHED_SPL.items():
        print(f"  published {name}: flux at 100 TeV {phi0:.2f}, gamma {gamma:.2f}")
    for name, (phi0, gg1, gg2, lbb) in PUBLISHED_BPL.items():
        print(f"  published {name}: flux at 100 TeV {bpl(PIVOT_GEV, phi0, gg1, gg2, lbb):.2f}, "
              f"gamma_1 {gg1:.2f}, gamma_2 {gg2:.3f}, "
              f"break 10^{lbb:.2f} GeV = {10**lbb / 1e3:.0f} TeV")
    reach = window_reach(ex51, likelihood, spl(10.0 ** ex51.LOG10_E_GRID, 1.0, g_s))
    print(f"  window reach: 90% of the astro tracks come from E_nu in "
          f"[10^{reach[0]:.2f}, 10^{reach[1]:.2f}] GeV")

    print()
    figure_reco(ex51, likelihood, fit, spl_params, bpl_params, args.out_dir)
    figure_flux(spl_params, bpl_params, reach, args.out_dir)


if __name__ == "__main__":
    main()
