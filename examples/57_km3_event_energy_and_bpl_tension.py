"""Example 57 -- KM3-230213A: its neutrino energy in closed form, and the BPL tension.

Two things example 31 left open. Its docstring converts the event's muon
energy to a neutrino energy through the mean inelasticity alone and calls the
result a lower bound to be "replaced by the event's actual energy likelihood
before any number here is quoted"; and it fits a single power law, which
IceCube has since rejected in favour of a broken one (arXiv:2507.22234,
example 53). This example supplies both.

**The energy likelihood.** A muon produced at energy ``eps`` a distance ``X``
upstream arrives with log-loss ``w = ln(eps / E)`` distributed as the
subordinator of :mod:`softpaws.transport.loss_distribution`,
``E[e^{-s w}] = e^{-X Phi(s)}``. The production point is not known. Along
the nominal direction the release's topography puts 34 km of sea water in
front of the detector (behind it 104 km of rock and 4 km of water, 309 km
w.e. in all); a muon from even a 3 EeV parent reaches 120 PeV within ~7 km,
so the birth point is uniform upstream in water. The muon-energy law at the
detector is then the *potential density* of the subordinator,

.. math:: u(w) = \\int_0^\\infty P(w \\mid X)\\,dX,

which the exact kernel gives in closed form: a peak at ``w -> 0`` where the
muon was born close, a dip near ``w ~ 1``, and the renewal plateau
``1 / Phi'(0)`` beyond -- the inverse *mean log-loss rate*, below ``1 / b_mu``
because ``-ln(1 - y) > y``. Two references are drawn with it: the
Fokker-Planck Gaussian of the paper's Appendix, and the mean-loss (CSDA)
limit, for which ``u(w) = 1 / b_mu`` exactly and the posterior collapses onto
the flux times the cross section above ``E_mu / (1 - <y>)``. The neutrino-energy posterior is

.. math:: P(E_\\nu \\mid \\text{event}) \\propto \\Phi(E_\\nu)\\,\\sigma_{CC}(E_\\nu)\\,
    S(E_\\nu)\\int dE\\, L(E)\\, u\\bigl(\\ln[(1 - \\langle y\\rangle) E_\\nu / E]\\bigr),

with ``L`` the lognormal muon-energy measurement (120 PeV, 90% interval
35-380 PeV), ``S`` the survival through the sea path, and the flux prior one
of KM3NeT's ``E^-2``, example 31's single power law and IceCube's broken
power law. KM3NeT's own estimate under ``E^-2`` is drawn for comparison.

**The tension.** Example 31's profile-likelihood construction, with the flux
family made generic. The single power law is rerun as is; the broken power
law holds IceCube's lower branch and break (``gamma_1 = 1.31``,
``10^4.39`` GeV) and frees the normalization and the index above the break,
with the IceCube Asimov injected at IceCube's own fit. The KM3NeT event
enters either through example 31's hard energy window or through the energy
likelihood above -- the muon measurement folded with the arrival kernel,
normalized over the muons the selection would accept -- which is what the
31 docstring asked for. The compatibility test is example 31's, two degrees
of freedom, and is printed for every combination.

Usage
-----
    python examples/57_km3_event_energy_and_bpl_tension.py
    python examples/57_km3_event_energy_and_bpl_tension.py --n-gamma 31 --n-phi0 41
"""

import argparse
import importlib.util
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from softpaws.fluxes import ICECUBE_BPL_2025, ICECUBE_TRACKS_2022, broken_power_law_shape
from softpaws.transport.coefficients import (
    diffusion_coefficient,
    drift_coefficient,
    third_moment_coefficient,
)
from softpaws.transport.loss_distribution import (
    loss_density,
    loss_density_gaussian,
    loss_density_three_moment,
)
from softpaws.transport.source import mean_inelasticity
from softpaws.utils.constants import CM_PER_KM, RHO_WATER_G_CM3

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_DATA_DIR = _HERE.parent / "src" / "softpaws" / "data" / "dataverse_files"
_DEFAULT_OUT_DIR = _HERE / "output"

#: Avogadro's number [mol^-1], for the sea-path survival.
AVOGADRO = 6.02214076e23

#: KM3-230213A's muon energy [PeV]: the estimate and its 90% interval
#: (KM3NeT Collaboration, Nature 638 (2025) 376).
MU_PEV = 120.0
MU_90_PEV = (35.0, 380.0)

#: KM3NeT's neutrino-energy estimate under an ``E^-2`` prior [PeV], the median
#: and its 90% interval, as quoted in the same paper (72 PeV to 2.6 EeV).
KM3NET_ENU_PEV = (220.0, 72.0, 2600.0)

#: Elevation of the arrival direction above the horizon [deg] and the
#: detector depth [km]; the sea path in front of the detector follows.
ELEVATION_DEG = 0.6
DEPTH_KM = 3.4
EARTH_RADIUS_KM = 6371.0

#: The traversed column along the nominal direction from the release's
#: topography notebook (Zenodo 10.5281/zenodo.14860165): water 4 km, rock
#: 104 km, then 34 km of water in front of the detector; 142 km, 309 km w.e.
#: The neutrino survival uses the whole column; the muon's birth point lies
#: inside the last 34 km of water for any parent energy considered here.
TRAVERSED_COLUMN_KMWE = 309.0
WATER_BEFORE_DETECTOR_KM = 34.0

#: Reference energy at which the loss coefficients are frozen for the kernel
#: [GeV]; the kernel is scale invariant, and b_mu moves by 4% per decade here.
KERNEL_ENERGY_GEV = 1.0e8

#: Log-loss grid and the upstream integration for the potential density. The
#: grid must hold the whole loss law of the deepest slab (mean ``b X_max``
#: plus its spread), since :func:`loss_density` renormalizes on the grid.
W_GRID = np.linspace(0.0, 24.0, 4801)
X_MAX_KM = 30.0
N_X = 120
N_K = 2**13
#: Loss-family calibration behind the "exact" potential density: 3 uses the
#: three-moment family of Eq. (7), whose hard edge matches PROPOSAL's tail
#: (Appendix D); 2 is the digamma two-moment family, 8% low on ``Phi'(0)``
#: and so 8% long on the renewal plateau ``1 / Phi'(0)`` (Appendix C).
KERNEL_MOMENTS = 3

#: Muon energy below which the bright-track selection does not count a muon
#: [GeV]; only enters the normalization of the arrival density.
MU_ACCEPT_GEV = 1.0e5

#: Neutrino-energy grid of the posterior [log10 GeV].
LOG10_ENU = np.linspace(7.0, 10.5, 351)

#: IceCube's broken power law (example 54's constants): lower index and
#: break held, ``(phi0, gamma_2)`` free in the fit.
BPL_GAMMA_1 = ICECUBE_BPL_2025.gamma_1
BPL_LOG_BREAK = ICECUBE_BPL_2025.log10_break_gev
BPL_ICECUBE = (ICECUBE_BPL_2025.phi0, ICECUBE_BPL_2025.gamma_2)

#: Example 31's single power-law truth, and IceCube's 9.5-year tracks fit,
#: as the two single-power-law injections.
SPL_ICECUBE_TRACKS = (ICECUBE_TRACKS_2022.phi0, ICECUBE_TRACKS_2022.gamma)

COLORS = {"exact": "#e7298a", "gaussian": "#7570b3", "csda": "#1b9e77", "km3net": "0.3",
          "SPL": "#1b9e77", "BPL": "#e7298a", "IceCube": "#7570b3"}


def load_example(stem: str, name: str):
    """Import a sibling example module by file stem."""
    spec = importlib.util.spec_from_file_location(name, _HERE / stem)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EX31 = load_example("31_flux_contours_effective_area.py", "_example_31")


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=pathlib.Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--n-gamma", type=int, default=41)
    parser.add_argument("--n-phi0", type=int, default=57)
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the figures, '57a' and '57b'.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Part A: the arrival kernel and the energy posterior
# ---------------------------------------------------------------------------


def sea_path_km(elevation_deg: float = ELEVATION_DEG, depth_km: float = DEPTH_KM) -> float:
    """Path from the detector to the sea surface at a given elevation [km]."""
    r0 = EARTH_RADIUS_KM - depth_km
    s = np.sin(np.deg2rad(elevation_deg))
    return float(-r0 * s + np.sqrt((r0 * s) ** 2 + EARTH_RADIUS_KM**2 - r0**2))


def potential_density(kind: str) -> np.ndarray:
    """``u(w) = int_0^inf P(w | X) dX`` on :data:`W_GRID` [km per unit w].

    Parameters
    ----------
    kind : str
        ``"exact"`` (the subordinator), ``"gaussian"`` (Fokker-Planck) or
        ``"csda"`` (mean loss, ``1 / b_mu`` exactly).
    """
    b = float(np.squeeze(drift_coefficient(KERNEL_ENERGY_GEV)))
    d = float(np.squeeze(diffusion_coefficient(KERNEL_ENERGY_GEV)))
    t = float(np.squeeze(third_moment_coefficient(KERNEL_ENERGY_GEV)))
    if kind == "csda":
        return np.full(W_GRID.size, 1.0 / b)
    x_grid = np.linspace(0.0, X_MAX_KM, N_X + 1)[1:]
    u = np.zeros(W_GRID.size)
    for x in x_grid:
        if kind == "exact" and KERNEL_MOMENTS == 3:
            u += loss_density_three_moment(W_GRID, float(x), b, d, t, n_k=N_K)
        elif kind == "exact":
            u += loss_density(W_GRID, float(x), b, d, n_k=N_K)
        else:
            u += loss_density_gaussian(W_GRID, float(x), b, d)
    u *= x_grid[1] - x_grid[0]
    # The first slab, X in (0, dX): the loss is small and the density sharply
    # peaked, so it is taken as its trapezoid end point rather than resolved.
    return u


def muon_measurement(log10_e_mu) -> np.ndarray:
    """Lognormal likelihood of the measured muon energy, density in ``ln E``."""
    sigma = (np.log(MU_90_PEV[1] / MU_90_PEV[0])) / (2.0 * norm.isf(0.05))
    return norm.pdf(log10_e_mu * np.log(10.0), loc=np.log(MU_PEV * 1.0e6), scale=sigma)


def energy_likelihood(u: np.ndarray, energy_nu, normalize: bool = True) -> np.ndarray:
    """``l(E_nu) = int dw u(w) L(eps e^-w)``, the event's energy likelihood.

    The integral runs over the log-loss on the kernel's own grid, so the
    integrable spike of the exact ``u`` at ``w -> 0`` is sampled identically
    for every ``E_nu``. Integrating over a muon-energy grid instead lets the
    spike slide across that grid and prints a sawtooth into ``l(E_nu)``.

    With ``normalize`` the result is divided by the muons the selection
    accepts, ``int u dw`` over ``E_mu >= MU_ACCEPT_GEV``, which makes it the
    conditional density the *tension* term needs alongside ``A_eff`` (whose
    effective length is that same integral). The energy *posterior* carries
    ``sigma_CC`` and not ``A_eff``, so it takes the unnormalized rate
    ``Phi sigma u(w)``; dividing there would tilt it to low energies.
    """
    energy_nu = np.atleast_1d(np.asarray(energy_nu, dtype=float))
    eps = (1.0 - np.squeeze(mean_inelasticity(energy_nu))) * energy_nu
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (u[1:] + u[:-1]) * np.diff(W_GRID))])
    w_accept = np.log(eps / MU_ACCEPT_GEV)
    total = np.where(w_accept <= W_GRID[-1], np.interp(w_accept, W_GRID, cdf),
                     cdf[-1] + (w_accept - W_GRID[-1]) * u[-1])
    log10_e_mu = (np.log(eps)[:, None] - W_GRID[None, :]) / np.log(10.0)
    like = muon_measurement(log10_e_mu)
    value = np.trapezoid(u[None, :] * like, W_GRID, axis=1)
    return value / np.clip(total, 1.0e-300, None) if normalize else value


def sea_survival(energy_nu) -> np.ndarray:
    """Survival through the traversed column, :data:`TRAVERSED_COLUMN_KMWE`."""
    column = TRAVERSED_COLUMN_KMWE * CM_PER_KM * RHO_WATER_G_CM3
    sigma = _EX31.CROSS_SECTION.cc(energy_nu) + _EX31.CROSS_SECTION.nc(energy_nu)
    return np.exp(-AVOGADRO * sigma * column)


def flux_shape(name: str, energy_gev) -> np.ndarray:
    """Unit-normalization flux shapes of the three priors."""
    e = np.asarray(energy_gev, dtype=float)
    if name == "E^-2":
        return (e / 1.0e5) ** (-2.0)
    if name == "SPL":
        return (e / 1.0e5) ** (-_EX31.GAMMA_TRUTH)
    if name == "BPL":
        return bpl_shape(e, BPL_ICECUBE[1])
    raise ValueError(name)


def bpl_shape(energy_gev, gamma_2):
    """IceCube's broken power law, unit flux at 100 TeV on the upper branch."""
    return broken_power_law_shape(energy_gev, gamma_2, BPL_GAMMA_1, BPL_LOG_BREAK)


def posterior(likelihood: np.ndarray, prior: str) -> np.ndarray:
    """Normalized posterior on :data:`LOG10_ENU` for one flux prior."""
    energy = 10.0**LOG10_ENU
    weight = flux_shape(prior, energy) * _EX31.CROSS_SECTION.cc(energy) * sea_survival(energy)
    p = likelihood * weight * energy  # density in log10 E
    return p / np.trapezoid(p, LOG10_ENU)


def summarize_posterior(p: np.ndarray):
    """Mode, median and 90% interval [PeV]."""
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(LOG10_ENU))])
    lo, med, hi = np.interp([0.05, 0.5, 0.95], cdf, LOG10_ENU)
    mode = LOG10_ENU[int(np.argmax(p))]
    return tuple(10.0**v / 1.0e6 for v in (mode, med, lo, hi))


# ---------------------------------------------------------------------------
# Part B: the tension with a generic flux family
# ---------------------------------------------------------------------------


def expected_counts_flux(edges, aeff_log10_e, aeff, flux, livetime_s, solid_angle_sr,
                         n_sub=96) -> np.ndarray:
    """Example 31's :func:`expected_counts` for any flux function of energy [GeV]."""
    edges = np.asarray(edges, dtype=float)
    counts = np.empty(edges.size - 1)
    for i in range(edges.size - 1):
        log10_e = np.linspace(edges[i], edges[i + 1], n_sub)
        energy = 10.0**log10_e
        aeff_here = _EX31._log_interp_aeff(aeff_log10_e, aeff, log10_e)
        counts[i] = (np.trapezoid(flux(energy) * aeff_here, energy) * livetime_s
                     * solid_angle_sr)
    return counts


def flux_family(name: str):
    """``(shape(energy, index), label of the index, truth)`` per family."""
    if name == "SPL":
        return (lambda e, g: (e / 1.0e5) ** (-g)), r"\gamma", (_EX31.PHI0_TRUTH,
                                                                 _EX31.GAMMA_TRUTH)
    if name == "SPL (IceCube tracks)":
        return (lambda e, g: (e / 1.0e5) ** (-g)), r"\gamma", SPL_ICECUBE_TRACKS
    if name == "BPL":
        return bpl_shape, r"\gamma_2", BPL_ICECUBE
    raise ValueError(name)


def km_event_coefficients(shape, gamma_grid, km_log10_e, km_aeff, energy_like=None):
    """ARCA21 counts per unit ``phi0``: event term and band term, per index.

    With ``energy_like`` the event term is ``T Omega int dE phi A l(E)``, the
    energy likelihood replacing example 31's window; without it, the window.
    """
    ex = _EX31
    kwargs = dict(livetime_s=ex.KM_LIVETIME_S, solid_angle_sr=ex.KM_SOLID_ANGLE_SR)
    a_event, a_band = np.empty(gamma_grid.size), np.empty(gamma_grid.size)
    energy = 10.0**LOG10_ENU
    for j, g in enumerate(gamma_grid):
        def flux(e, g=g):
            return 1.0e-18 * shape(e, g)
        if energy_like is None:
            a_event[j] = expected_counts_flux(ex.KM_EDGES, km_log10_e, km_aeff, flux,
                                              **kwargs).sum()
        else:
            aeff_here = ex._log_interp_aeff(km_log10_e, km_aeff, LOG10_ENU)
            a_event[j] = (np.trapezoid(flux(energy) * aeff_here * energy_like, energy)
                          * ex.KM_LIVETIME_S * ex.KM_SOLID_ANGLE_SR)
        a_band[j] = expected_counts_flux(ex.KM_BAND_LOG10_E, km_log10_e, km_aeff, flux,
                                         n_sub=240, **kwargs).sum()
    return a_event, a_band


def ic_surface(shape, truth, phi0_grid, gamma_grid, ic_log10_e, ic_aeff, livetime_s):
    """IceCube profile log-likelihood on the grid, Asimov at ``truth``."""
    ex = _EX31

    def counts(phi0, g):
        return expected_counts_flux(ex.IC_FIT_EDGES, ic_log10_e, ic_aeff,
                                    lambda e: phi0 * 1.0e-18 * shape(e, g), livetime_s,
                                    ex.IC_SOLID_ANGLE_SR)

    signal = counts(*truth)
    observed = signal + ex.BKG_FRACTION * signal.sum() * ex.atmospheric_template(
        ex.IC_FIT_EDGES, 3.7)
    template = ex.atmospheric_template(ex.IC_FIT_EDGES, 3.7)
    signal_unit = np.array([counts(1.0, g) for g in gamma_grid])
    return ex.ic_log_likelihood(phi0_grid, signal_unit, observed, template)


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


def figure_energy(potentials, posteriors, out_dir) -> None:
    """Left: the three arrival kernels. Right: the neutrino-energy posteriors."""
    with plt.style.context(str(_STYLE)):
        fig, (ax_u, ax_p) = plt.subplots(1, 2, figsize=(6.2, 2.9),
                                         gridspec_kw={"wspace": 0.28})
        labels = {"exact": "exact kernel", "gaussian": "Fokker-Planck", "csda": "mean loss"}
        for kind, u in potentials.items():
            ax_u.plot(W_GRID, u, color=COLORS[kind], lw=1.2, label=labels[kind])
        ax_u.set_xlim(0.0, 6.0)
        # The exact kernel's integrable spike at w -> 0 (the muon born in the
        # last few hundred metres) would set the whole axis; cap it.
        ax_u.set_ylim(0.0, 4.0)
        ax_u.text(0.35, 3.75, r"exact $u(w) \to \infty$ as $w \to 0$", fontsize=6,
                  color=COLORS["exact"], va="top")
        ax_u.set_xlabel(r"$w = \ln(E_\mu^{\rm birth}\,/\,E_\mu^{\rm det})$")
        ax_u.set_ylabel(r"$u(w)$ [km]")
        ax_u.legend(fontsize=6, frameon=False, loc="lower right")

        styles = {"E^-2": "-", "SPL": "--", "BPL": ":"}
        for (prior, kind), p in posteriors.items():
            if kind != "exact" and prior != "E^-2":
                continue
            lw = 1.4 if kind == "exact" else 0.9
            shown = {"E^-2": r"$E^{-2}$", "SPL": "SPL", "BPL": "BPL"}[prior]
            label = f"{shown}, {labels[kind]}"
            ax_p.plot(LOG10_ENU, p, color=COLORS[kind], lw=lw, ls=styles[prior], label=label)
        mode, lo, hi = (np.log10(v * 1.0e6) for v in KM3NET_ENU_PEV)
        ax_p.axvspan(lo, hi, color="0.92", zorder=0)
        ax_p.axvline(mode, color=COLORS["km3net"], lw=0.9, ls="-.",
                     label=r"KM3NeT, $E^{-2}$ (90\%)")
        ax_p.set_xlim(LOG10_ENU[0], LOG10_ENU[-1])
        ax_p.set_ylim(0.0, None)
        ax_p.set_xlabel(r"$\log_{10}(E_\nu\,/\,\mathrm{GeV})$")
        ax_p.set_ylabel("posterior density")
        ax_p.legend(fontsize=5.5, frameon=False, loc="upper right")
        _save(fig, out_dir, "57a_km3_event_energy")


def figure_tension(results, phi0_grid, gamma_grid, out_dir) -> None:
    """IceCube and KM3NeT profile regions, one panel per flux family."""
    families = [f for f in ("SPL", "BPL") if f in results]
    with plt.style.context(str(_STYLE)):
        fig, axes = plt.subplots(1, len(families), figsize=(3.1 * len(families), 3.0),
                                 sharey=True, gridspec_kw={"wspace": 0.08})
        axes = np.atleast_1d(axes)
        for ax, family in zip(axes, families):
            r = results[family]
            ic = r["ic"] - r["ic"].max()
            km = r["km_like"] - r["km_like"].max()
            ax.contourf(gamma_grid, phi0_grid, -ic, levels=[0.0, *_EX31.DELTA_LNL_LEVELS],
                        colors=[COLORS["IceCube"]] * 2, alpha=0.25)
            ax.contour(gamma_grid, phi0_grid, -ic, levels=_EX31.DELTA_LNL_LEVELS,
                       colors=COLORS["IceCube"], linewidths=[1.1, 0.8])
            window = r["km_window"] - r["km_window"].max()
            ax.contourf(gamma_grid, phi0_grid, -window, levels=[0.0, _EX31.DELTA_LNL_LEVELS[0]],
                        colors=[COLORS[family]], alpha=0.1)
            ax.contour(gamma_grid, phi0_grid, -km, levels=_EX31.DELTA_LNL_LEVELS,
                       colors=COLORS[family], linewidths=[1.1, 0.8], linestyles=["-", "--"])
            ax.plot(gamma_grid, r["one_event"], color=COLORS[family], lw=0.8, ls=":")
            ax.plot(*r["truth"][::-1], marker="+", color="k", ms=7, mew=1.2)
            ax.set_yscale("log")
            ax.set_ylim(phi0_grid[0], phi0_grid[-1])
            ax.set_xlim(gamma_grid[0], gamma_grid[-1])
            ax.set_xlabel(rf"${r['index_label']}$")
            ax.text(0.04, 0.96, f"{family}\n{r['sigma_like']:.1f}$\\sigma$",
                    transform=ax.transAxes, fontsize=7, va="top")
        axes[0].set_ylabel(r"$\phi_0$ at 100 TeV "
                           r"[$10^{-18}$ GeV$^{-1}$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
        from matplotlib.patches import Patch

        handles = [plt.Line2D([], [], color=COLORS["IceCube"], lw=1.1, label="IceCube (Asimov)"),
                   plt.Line2D([], [], color="0.3", lw=1.1, label="ARCA21, energy likelihood"),
                   Patch(facecolor="0.3", alpha=0.15, lw=0, label=r"ARCA21, energy window (68\%)"),
                   plt.Line2D([], [], color="0.3", lw=0.8, ls=":", label="one event in window")]
        axes[-1].legend(handles=handles, fontsize=6, frameon=False, loc="lower right")
        _save(fig, out_dir, "57b_bpl_tension")


def main() -> None:
    args = parse_args()
    ex = _EX31

    print(f"Sea path to the surface at {ELEVATION_DEG} deg elevation from {DEPTH_KM} km on a "
          f"smooth sphere: {sea_path_km():.0f} km; release topography: "
          f"{WATER_BEFORE_DETECTOR_KM:.0f} km of water before the detector, "
          f"{TRAVERSED_COLUMN_KMWE:.0f} km w.e. traversed")
    b = float(np.squeeze(drift_coefficient(KERNEL_ENERGY_GEV)))
    print(f"Kernel at 10^{np.log10(KERNEL_ENERGY_GEV):.0f} GeV: b_mu {b:.3f} km^-1, "
          f"d_mu {float(np.squeeze(diffusion_coefficient(KERNEL_ENERGY_GEV))):.3f} km^-1")

    print("\nPotential densities of the three kernels ...")
    potentials = {kind: potential_density(kind) for kind in ("exact", "gaussian", "csda")}
    for kind, u in potentials.items():
        at = [u[np.searchsorted(W_GRID, w)] for w in (0.1, 1.0, 4.0, 8.0)]
        print(f"  {kind:>8}: u(0.1) {at[0]:.3f}, u(1) {at[1]:.3f}, u(4) {at[2]:.3f}, "
              f"u(8) {at[3]:.3f} km  (1/b = {1.0 / b:.3f})")

    energy = 10.0**LOG10_ENU
    likelihoods = {kind: energy_likelihood(u, energy) for kind, u in potentials.items()}
    rates = {kind: energy_likelihood(u, energy, normalize=False)
             for kind, u in potentials.items()}
    posteriors = {}
    print("\nNeutrino energy of KM3-230213A [PeV]: mode, median, 90% interval")
    print(f"  KM3NeT (E^-2): --, {KM3NET_ENU_PEV[0]:.0f}, [{KM3NET_ENU_PEV[1]:.0f}, "
          f"{KM3NET_ENU_PEV[2]:.0f}]")
    for prior in ("E^-2", "SPL", "BPL"):
        for kind in ("exact", "gaussian", "csda"):
            p = posterior(rates[kind], prior)
            posteriors[(prior, kind)] = p
            mode, med, lo, hi = summarize_posterior(p)
            print(f"  {prior:>5} prior, {kind:>8} kernel: {mode:6.0f}, {med:6.0f}, "
                  f"[{lo:.0f}, {hi:.0f}]")

    print(f"\nBuilding IceCube A_eff(E_nu), loading DR2 from: {args.data_dir}")
    ic_log10_e, ic_aeff, ic_livetime_s = ex.build_icecube_aeff(args.data_dir,
                                                               ex.DEFAULT_MUON_THRESHOLD_GEV)
    km_log10_e, km_aeff = ex.arca21_published_aeff()
    gamma_grid = np.linspace(*ex.GAMMA_RANGE, args.n_gamma)
    phi0_grid = np.logspace(-1.5, 2.7, args.n_phi0)
    like_exact = likelihoods["exact"]

    results = {}
    print("\nTension, IceCube Asimov against the ARCA21 event (2 dof):")
    print(f"  {'flux family':>22} {'truth':>14} {'event term':>18} {'TS':>6} {'sigma':>6} "
          f"{'IC best':>16} {'KM best':>16}")
    for family in ("SPL", "SPL (IceCube tracks)", "BPL"):
        shape, index_label, truth = flux_family(family)
        ic = ic_surface(shape, truth, phi0_grid, gamma_grid, ic_log10_e, ic_aeff, ic_livetime_s)
        row = {"ic": ic, "truth": truth, "index_label": index_label}
        for tag, energy_like in (("window", None), ("energy likelihood", like_exact)):
            a_event, a_band = km_event_coefficients(shape, gamma_grid, km_log10_e, km_aeff,
                                                    energy_like)
            km = ex.km_log_likelihood(phi0_grid[:, None], a_event, a_band, extended=True)
            ts, p, sigma = ex.compatibility(ic, km)
            ic_best, km_best = ex.best_fit(phi0_grid, gamma_grid, ic), ex.best_fit(
                phi0_grid, gamma_grid, km)
            print(f"  {family:>22} ({truth[0]:.2f}, {truth[1]:.2f}) {tag:>18} {ts:6.2f} "
                  f"{sigma:6.2f} ({ic_best[0]:6.2f}, {ic_best[1]:.2f}) "
                  f"({km_best[0]:6.2f}, {km_best[1]:.2f})")
            if tag == "window":
                row["one_event"] = ex.phi0_for_one_event(a_event)
                row["km_window"], row["sigma_window"] = km, sigma
            else:
                row["km_like"], row["sigma_like"] = km, sigma
        results[family] = row

    print()
    figure_energy(potentials, posteriors, args.out_dir)
    figure_tension(results, phi0_grid, gamma_grid, args.out_dir)


if __name__ == "__main__":
    main()
