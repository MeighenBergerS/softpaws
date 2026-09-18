"""Example 58 -- KM3-230213A as a new component on top of IceCube's flux.

Examples 31 and 57 ask whether the event belongs to the flux IceCube
measures at 10-1000 TeV, and find it strained at 2-3 sigma. The other
reading is that it does not: a second, harder component -- cosmogenic
(GZK) neutrinos, or a separate diffuse population -- that is negligible at
IceCube's energies and carries the event. This example fits that second
component, under the one condition any diffuse component must meet: its
tail cannot produce events IceCube's extremely-high-energy search did not
see.

**The model.** ``Phi = Phi_BPL + Phi_new``: IceCube's broken power law held
at its published fit (arXiv:2507.22234, example 54's constants), plus either

* a log-Gaussian bump, ``E^2 Phi = A exp(-(log10 E - log10 E_p)^2 / 2 s^2)``
  with ``s = 0.5`` dex, the width of a typical cosmogenic spectrum, on a
  grid in ``(log10 E_p, log10 A)``; or
* one of the ten cosmogenic templates the KM3NeT release ships
  (``src/softpaws/data/km3net/uhe_release/flux_models``), times a free scale.

**The likelihood.** Three independent terms.

* ARCA21, the event: ``T Omega int dE Phi(E) A_eff(E) l(E)`` with example
  57's energy likelihood ``l`` (the muon measurement folded with the exact
  arrival kernel) and the released bright-track effective area.
* ARCA21, nothing else: ``exp(-mu_band)`` over the rest of the UHE band,
  example 31's second Poisson term.
* IceCube-EHE, nothing at all: the nine-year quasi-differential limit
  (Phys. Rev. D 98, 062003) is 2.44 events per energy decade at 90%, for an
  ``E^-2`` flux inside the decade. Read backwards, that fixes the search's
  exposure per decade, so for any model the expected count in a decade
  centred at ``E`` is ``2.44 <E^2 Phi>_decade / E^2 Phi_limit(E)``, with
  the log-average over the decade (an exposure growing like ``E`` across
  it). Non-overlapping decades from ``10^6.75`` to ``10^10.75`` each
  contribute ``exp(-N)``; the most constraining sliding decade is printed
  beside them.

**What comes out.** In the bump plane, the ARCA21 terms alone make a ridge
(one event in the window, any energy), and the IceCube term cuts it off
from above: the joint best fit is where the event is as probable as the
IceCube non-observation allows. The number to read is ``mu_event`` at that
point -- how likely the event was under the best new component IceCube
permits -- against the ~0.1 of the single-spectrum reading. For the
cosmogenic templates the same question is a scale factor: what multiple of
each model the event asks for, and what multiple IceCube-EHE forbids.

The IceCube side is a published limit, not a forward model of this project;
the ARCA21 side is example 31's. The bump width is a convention and
``--width`` moves it.

Usage
-----
    python scripts/2026_muon_transport/58_km3_event_new_component.py
    python scripts/2026_muon_transport/58_km3_event_new_component.py --width 0.3 --no-icecube
"""

import argparse
import importlib.util
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_DATA_DIR = _HERE.parents[1] / "src" / "softpaws" / "data" / "dataverse_files"
_DEFAULT_OUT_DIR = _HERE / "output"
_RELEASE = _HERE.parents[1] / "src" / "softpaws" / "data" / "km3net" / "uhe_release"

#: IceCube-EHE: events per decade at the 90% differential limit, and the
#: non-overlapping decades the likelihood sums.
EHE_EVENTS_PER_DECADE = 2.44
EHE_DECADE_CENTRES = np.array([7.25, 8.25, 9.25, 10.25])

#: Bump-plane grid: peak energy [log10 GeV] and peak ``E^2 Phi``
#: [log10 GeV cm^-2 s^-1 sr^-1].
LOG10_EP_GRID = np.linspace(7.0, 10.5, 36)
LOG10_A_GRID = np.linspace(-10.5, -6.5, 81)

#: KM3NeT's 90% neutrino-energy window for the event [GeV] (release quantiles);
#: ``mu_90`` counts expected events inside it, the number the collaboration
#: quotes as "events like this one".
EVENT_WINDOW_GEV = (7.24e7, 2.57e9)

#: Energy grid of every flux integral [log10 GeV].
LOG10_E = np.linspace(5.5, 11.0, 551)

#: Cosmogenic templates of the release, in file order.
TEMPLATES = sorted(p.name for p in (_RELEASE / "flux_models").glob("cosmogenic_*.json")
                   if "band" not in p.name)

COLORS = {"arca": "#d95f02", "joint": "#e7298a", "bpl": "#7570b3", "ehe": "0.25",
          "auger": "0.55", "band": "0.85"}


def load_example(stem: str, name: str):
    """Import a sibling example module by file stem."""
    spec = importlib.util.spec_from_file_location(name, _HERE / stem)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EX57 = load_example("57_km3_event_energy_and_bpl_tension.py", "_example_57")
_EX31 = _EX57._EX31


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--width", type=float, default=0.5,
                        help="Bump width in log10(E) [dex].")
    parser.add_argument("--no-icecube", action="store_true",
                        help="Drop the IceCube-EHE term (ARCA21 alone).")
    parser.add_argument("--extra-exposure", type=float, nargs="+", default=[0.0, 2.0, 5.0],
                        help="ARCA exposure added since the release with no further event, "
                             "in multiples of the ARCA21 release exposure (335 days of "
                             "ARCA21); the first value makes the figures.")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the figures, '58a' and '58b'.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Fluxes [GeV^-1 cm^-2 s^-1 sr^-1, per flavour]
# ---------------------------------------------------------------------------


def bpl_flux(energy_gev) -> np.ndarray:
    """IceCube's broken power law at its published fit."""
    return _EX57.BPL_ICECUBE[0] * 1.0e-18 * _EX57.bpl_shape(energy_gev, _EX57.BPL_ICECUBE[1])


def bump_flux(energy_gev, log10_ep: float, log10_a: float, width: float) -> np.ndarray:
    """Log-Gaussian bump in ``E^2 Phi``."""
    log10_e = np.log10(energy_gev)
    e2phi = 10.0**log10_a * np.exp(-0.5 * ((log10_e - log10_ep) / width) ** 2)
    return e2phi / energy_gev**2


def load_template(name: str):
    """One cosmogenic template: ``(energy_gev, e2phi)`` on its own grid."""
    d = json.load(open(_RELEASE / "flux_models" / name))
    energy = np.array(list(d["Energy [GeV]"].values()))
    key = next(k for k in d if k.startswith("log10(E^2 F(E)"))
    return energy, 10.0 ** np.array(list(d[key].values()))


def template_flux(template, energy_gev) -> np.ndarray:
    """A template interpolated in log-log, zero outside its grid."""
    t_energy, t_e2phi = template
    log_e2phi = np.interp(np.log10(energy_gev), np.log10(t_energy), np.log10(t_e2phi),
                          left=-np.inf, right=-np.inf)
    return 10.0**log_e2phi / energy_gev**2


def load_limit(name: str):
    """A differential limit: ``(energy_gev, e2phi)``."""
    d = json.load(open(_RELEASE / "flux_constraints" / f"limits_differential_{name}.json"))
    return (np.array(list(d["Energy [GeV]"].values())),
            np.array(list(d["E^2 F(E) [GeV.cm^-2.s^-1.sr^-1]"].values())))


# ---------------------------------------------------------------------------
# Likelihood terms
# ---------------------------------------------------------------------------


class Terms:
    """Precomputed pieces: ARCA21 event and band weights, IceCube-EHE exposure."""

    def __init__(self, energy_like: np.ndarray, use_icecube: bool, extra_exposure: float = 0.0):
        ex = _EX31
        self.energy = 10.0**LOG10_E
        km_log10_e, km_aeff = ex.arca21_published_aeff()
        aeff = ex._log_interp_aeff(km_log10_e, km_aeff, LOG10_E)
        like = np.interp(LOG10_E, _EX57.LOG10_ENU, energy_like, left=0.0, right=0.0)
        #: The one event sits in the whole ARCA exposure, release plus what came
        #: after it with nothing seen, so every ARCA term scales together.
        self.exposure_factor = 1.0 + extra_exposure
        exposure = ex.KM_LIVETIME_S * ex.KM_SOLID_ANGLE_SR * self.exposure_factor
        #: Event term weight: ``mu_event = int Phi w_event dE``.
        self.w_event = exposure * aeff * like
        band = (LOG10_E >= ex.KM_BAND_LOG10_E[0]) & (LOG10_E <= ex.KM_BAND_LOG10_E[1])
        self.w_band = np.where(band, exposure * aeff, 0.0)
        self.use_icecube = use_icecube
        self.ehe_energy, self.ehe_limit = load_limit("icecube-ehe")
        lo, hi = np.log10(EVENT_WINDOW_GEV)
        window = (LOG10_E >= lo) & (LOG10_E <= hi)
        #: Count weight over the event's 90% energy window: ``mu_90 = int Phi w_90 dE``.
        self.w_90 = np.where(window, exposure * aeff, 0.0)

    def mu_90(self, flux) -> float:
        """Expected ARCA21 events in the event's 90% neutrino-energy window."""
        return float(np.trapezoid(flux * self.w_90, self.energy))

    def mu_event(self, flux) -> float:
        return float(np.trapezoid(flux * self.w_event, self.energy))

    def mu_band(self, flux) -> float:
        return float(np.trapezoid(flux * self.w_band, self.energy))

    def ehe_counts(self, flux, centres=EHE_DECADE_CENTRES) -> np.ndarray:
        """Expected IceCube-EHE events per decade centred on ``centres``."""
        e2phi = flux * self.energy**2
        out = np.empty(len(centres))
        for k, centre in enumerate(centres):
            inside = (LOG10_E >= centre - 0.5) & (LOG10_E <= centre + 0.5)
            mean = np.trapezoid(e2phi[inside], LOG10_E[inside])  # log-average x 1 decade
            limit = 10.0 ** np.interp(centre, np.log10(self.ehe_energy), np.log10(self.ehe_limit))
            out[k] = EHE_EVENTS_PER_DECADE * mean / limit
        return out

    def log_l(self, flux) -> float:
        """``ln mu_event - mu_band - sum N_EHE``."""
        mu = self.mu_event(flux)
        value = (np.log(mu) if mu > 0.0 else -np.inf) - self.mu_band(flux)
        if self.use_icecube:
            value -= self.ehe_counts(flux).sum()
        return float(value)


def bump_plane(terms: Terms, width: float):
    """Log-likelihood on the bump grid, with and without the IceCube term."""
    energy = terms.energy
    base = bpl_flux(energy)
    with_ic = np.empty((LOG10_EP_GRID.size, LOG10_A_GRID.size))
    without = np.empty_like(with_ic)
    for i, ep in enumerate(LOG10_EP_GRID):
        for j, a in enumerate(LOG10_A_GRID):
            flux = base + bump_flux(energy, ep, a, width)
            mu = terms.mu_event(flux)
            core = (np.log(mu) if mu > 0.0 else -np.inf) - terms.mu_band(flux)
            without[i, j] = core
            with_ic[i, j] = core - terms.ehe_counts(flux).sum()
    return with_ic, without


def template_scan(terms: Terms, template, scales):
    """Log-likelihood and diagnostics against the template scale."""
    energy = terms.energy
    base = bpl_flux(energy)
    shape = template_flux(template, energy)
    rows = []
    for s in scales:
        flux = base + s * shape
        rows.append((terms.mu_event(flux), terms.mu_band(flux), terms.ehe_counts(flux).sum()))
    rows = np.array(rows)
    with np.errstate(divide="ignore"):
        log_l = np.log(rows[:, 0]) - rows[:, 1] - (rows[:, 2] if terms.use_icecube else 0.0)
    return log_l, rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _save(fig, out_dir: pathlib.Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {path.resolve()}")
    plt.close(fig)


def figure_plane(with_ic, without, terms, width, out_dir) -> None:
    """The bump plane: ARCA21 alone against ARCA21 + IceCube-EHE."""
    levels = 0.5 * chi2.isf([0.32, 0.05], df=2)
    # Where the bump alone would give 2.44 IceCube-EHE events in its best decade.
    energy = terms.energy
    excluded = np.empty((LOG10_EP_GRID.size, LOG10_A_GRID.size))
    for i, ep in enumerate(LOG10_EP_GRID):
        for j, a in enumerate(LOG10_A_GRID):
            excluded[i, j] = terms.ehe_counts(bump_flux(energy, ep, a, width)).max()
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.4, 3.2))
        ax.contourf(LOG10_EP_GRID, LOG10_A_GRID, (excluded > EHE_EVENTS_PER_DECADE).T.astype(float),
                    levels=[0.5, 1.5], colors=[COLORS["ehe"]], alpha=0.12)
        ax.contour(LOG10_EP_GRID, LOG10_A_GRID, -(without - without.max()).T, levels=levels,
                   colors=COLORS["arca"], linestyles=["-", "--"], linewidths=[1.1, 0.8])
        ax.contourf(LOG10_EP_GRID, LOG10_A_GRID, -(with_ic - with_ic.max()).T,
                    levels=[0.0, levels[0]], colors=[COLORS["joint"]], alpha=0.25)
        ax.contour(LOG10_EP_GRID, LOG10_A_GRID, -(with_ic - with_ic.max()).T, levels=levels,
                   colors=COLORS["joint"], linestyles=["-", "--"], linewidths=[1.1, 0.8])
        i, j = np.unravel_index(int(np.argmax(with_ic)), with_ic.shape)
        ax.plot(LOG10_EP_GRID[i], LOG10_A_GRID[j], "*", color=COLORS["joint"], ms=8)
        ax.set_xlabel(r"$\log_{10}(E_p\,/\,\mathrm{GeV})$")
        ax.set_ylabel(r"$\log_{10}$ peak $E^2\Phi$ [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
        from matplotlib.patches import Patch
        handles = [plt.Line2D([], [], color=COLORS["arca"], lw=1.1, label="ARCA21 alone"),
                   Patch(facecolor=COLORS["joint"], alpha=0.35,
                         label=r"ARCA21 + IceCube-EHE (68\%, 95\%)"),
                   Patch(facecolor=COLORS["ehe"], alpha=0.15,
                         label=r"bump alone $> 2.44$ EHE events")]
        ax.legend(handles=handles, fontsize=6, frameon=False, loc="lower right")
        ax.text(0.03, 0.97, rf"bump width {width:g} dex", transform=ax.transAxes, fontsize=6,
                va="top")
        _save(fig, out_dir, "58a_new_component_plane")


def figure_flux(terms, best_with, best_without, width, out_dir) -> None:
    """Fluxes against the limits and the release's cosmogenic band."""
    energy = np.logspace(4.0, 11.0, 400)
    scale = energy**2
    band = json.load(open(_RELEASE / "flux_models" / "cosmogenic_band.json"))
    b_e = np.array(list(band["Energy [GeV]"].values()))
    b_lo = 10.0 ** np.array(list(band[next(k for k in band if "min" in k)].values()))
    b_hi = 10.0 ** np.array(list(band[next(k for k in band if "max" in k)].values()))
    auger = load_limit("auger")
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.6, 3.0))
        ax.fill_between(b_e, b_lo, b_hi, color=COLORS["band"], lw=0, label="cosmogenic models")
        ax.plot(terms.ehe_energy, terms.ehe_limit, color=COLORS["ehe"], lw=1.0, ls=":",
                label=r"IceCube-EHE 90\% (2018)")
        ax.plot(*auger, color=COLORS["auger"], lw=1.0, ls=":", label=r"Auger 90\% (2023)")
        ax.plot(energy, scale * bpl_flux(energy), color=COLORS["bpl"], lw=1.1,
                label="IceCube BPL")
        curves = ((best_without, COLORS["arca"], "--", "bump, ARCA21 alone"),
                  (best_with, COLORS["joint"], "-", "bump, ARCA21 + IceCube-EHE"))
        for (ep, a), color, ls, label in curves:
            ax.plot(energy, scale * bump_flux(energy, ep, a, width), color=color, lw=1.2, ls=ls,
                    label=label)
        ax.axvspan(*EVENT_WINDOW_GEV, color="0.6", alpha=0.18, lw=0,
                   label=r"KM3-230213A 90\% energy window")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1.0e4, 1.0e11)
        ax.set_ylim(1.0e-11, 1.0e-6)
        ax.set_xlabel(r"$E_\nu$ [GeV]")
        ax.set_ylabel(r"$E^2 \Phi$ per flavour [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
        ax.legend(fontsize=5.5, frameon=False, loc="upper left")
        _save(fig, out_dir, "58b_new_component_flux")


def main() -> None:
    args = parse_args()
    print("Example 57's exact arrival kernel and the event's energy likelihood ...")
    u = _EX57.potential_density("exact")
    energy_like = _EX57.energy_likelihood(u, 10.0**_EX57.LOG10_ENU)
    print("\nARCA exposure since the release, in ARCA21-release units, with no further event:")
    print(f"  {'added':>6} {'joint best bump':>26} {'EHE N':>6} {'mu_90 release':>14} "
          f"{'mu_90 total':>12} {'P(>=1)':>7} {'ARCA-alone ridge peak':>22}")
    for extra in args.extra_exposure:
        t = Terms(energy_like, not args.no_icecube, extra)
        with_ic_k, without_k = bump_plane(t, args.width)
        i, j = np.unravel_index(int(np.argmax(with_ic_k)), with_ic_k.shape)
        ep, a = LOG10_EP_GRID[i], LOG10_A_GRID[j]
        flux = bpl_flux(t.energy) + bump_flux(t.energy, ep, a, args.width)
        mu_total = t.mu_90(flux)
        # ARCA-alone ridge: the peak height at which the band holds one event, at
        # the joint best fit's peak energy.
        col = int(np.argmin(np.abs(LOG10_EP_GRID - ep)))
        ridge = LOG10_A_GRID[int(np.argmax(without_k[col]))]
        print(f"  {extra:6.1f} 10^{ep:.2f} GeV, peak 10^{a:.2f} {t.ehe_counts(flux).sum():6.2f} "
              f"{mu_total / t.exposure_factor:14.3f} {mu_total:12.3f} "
              f"{1.0 - np.exp(-mu_total):7.3f} 10^{ridge:.2f} at 10^{ep:.2f} GeV")

    terms = Terms(energy_like, not args.no_icecube, args.extra_exposure[0])
    if args.extra_exposure[0] > 0.0:
        print(f"\nFigures and the tables below use {args.extra_exposure[0]:g} added exposures.")

    base = bpl_flux(terms.energy)
    print(f"\nIceCube BPL alone: mu_90 {terms.mu_90(base):.4f} events in the 90% window, "
          f"mu_band {terms.mu_band(base):.3f}, IceCube-EHE decades "
          f"{np.array2string(terms.ehe_counts(base), precision=3)}")
    print("  (mu_event below is a density in the measured muon energy: compare ratios only)")

    print(f"\nScanning the bump plane (width {args.width:g} dex) ...")
    with_ic, without = bump_plane(terms, args.width)
    for label, surface in (("ARCA21 alone", without), ("ARCA21 + IceCube-EHE", with_ic)):
        i, j = np.unravel_index(int(np.argmax(surface)), surface.shape)
        ep, a = LOG10_EP_GRID[i], LOG10_A_GRID[j]
        flux = base + bump_flux(terms.energy, ep, a, args.width)
        bump_only = bump_flux(terms.energy, ep, a, args.width)
        sliding = terms.ehe_counts(flux, np.linspace(7.25, 10.25, 61))
        edge = " (rails at the grid edge: the ARCA21 ridge is mu_band = 1 at any E_p)" \
            if a >= LOG10_A_GRID[-1] - 1.0e-9 or ep >= LOG10_EP_GRID[-1] - 1.0e-9 else ""
        print(f"  {label}: best bump at 10^{ep:.2f} GeV, peak E^2 Phi 10^{a:.2f}{edge}; "
              f"mu_90 {terms.mu_90(flux):.3f}, mu_event {terms.mu_event(flux):.4f} (bump share "
              f"{terms.mu_event(bump_only) / terms.mu_event(flux):.2f}), mu_band "
              f"{terms.mu_band(flux):.2f}, IceCube-EHE {terms.ehe_counts(flux).sum():.2f} events "
              f"(worst sliding decade {sliding.max():.2f} at "
              f"10^{np.linspace(7.25, 10.25, 61)[int(np.argmax(sliding))]:.2f})")
    def argbest(surface):
        i, j = np.unravel_index(int(np.argmax(surface)), surface.shape)
        return float(LOG10_EP_GRID[i]), float(LOG10_A_GRID[j])

    best_with, best_without = argbest(with_ic), argbest(without)
    gain = 2.0 * (with_ic.max() - terms.log_l(base))
    print(f"  2 Delta lnL of BPL + bump over BPL alone (with IceCube-EHE): {gain:.2f} "
          f"for 2 parameters")

    print("\nCosmogenic templates x scale (with IceCube-EHE unless --no-icecube):")
    print(f"  {'template':>34} {'best x':>8} {'68%':>16} {'mu_90':>7} {'EHE N':>7} "
          f"{'x at N=2.44':>12} {'mu_90 there':>12}")
    scales = np.logspace(-1.5, 2.5, 161)
    for name in TEMPLATES:
        template = load_template(name)
        log_l, rows = template_scan(terms, template, scales)
        k = int(np.argmax(log_l))
        inside = scales[log_l >= log_l[k] - 0.5]
        n_alone = np.array([terms.ehe_counts(s * template_flux(template, terms.energy)).max()
                            for s in scales])
        allowed = n_alone <= EHE_EVENTS_PER_DECADE
        x_limit = scales[allowed].max() if allowed.any() else np.nan
        label = name.removeprefix("cosmogenic_").removesuffix(".json")
        shape = template_flux(template, terms.energy)
        mu90_best = terms.mu_90(base + scales[k] * shape)
        mu90_limit = terms.mu_90(base + x_limit * shape) if np.isfinite(x_limit) else np.nan
        print(f"  {label:>34} {scales[k]:8.2f} [{inside.min():6.2f}, {inside.max():6.2f}] "
              f"{mu90_best:7.3f} {rows[k, 2]:7.2f} {x_limit:12.2f} {mu90_limit:12.3f}")

    print()
    figure_plane(with_ic, without, terms, args.width, args.out_dir)
    figure_flux(terms, best_with, best_without, args.width, args.out_dir)


if __name__ == "__main__":
    main()
