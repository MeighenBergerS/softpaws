"""Example 60 -- heavy charged particles through the same kernel: the stau.

The (eps, n) forecast of example 59 says diffuse spectra cannot see modified
muon losses; this example shows where modified losses *are* visible: a
particle whose loss coefficients are orders of magnitude different. A
metastable stau (or any heavy charged slepton) radiates with the muon's
loss spectrum suppressed by its mass, so it crosses tens of kilometres of
rock and ice where every muon dies -- the near-horizontal excess of
Meighen-Berger et al., Phys. Lett. B 811 (2020) 135929 [arXiv:2005.07523],
whose IceCube analysis set ``m > 320`` GeV. Everything that analysis had to
simulate about propagation, this example produces in closed form from the
same kernel the rest of the project runs on.

**Coefficients.** The stau's radiative channels are the muon's, rescaled by
the kinematic-cutoff Jacobian of Reno, Sarcevic and Su
[Astropart. Phys. 24 (2005) 107]: pair production by ``m_mu / m``,
photonuclear by ``m_0 / m`` with ``m_0`` set here against their published
curve (:data:`NUC_MASS_GEV`), bremsstrahlung by ``(m_mu / m)^2``; ionization
is mass-blind, ``a = 2e-3 GeV cm^2/g``. The first figure validates the
resulting range against their fitted parameterization (their own quoted
accuracy is ~20%).

**What the kernel adds.** The same Jacobian suppresses the *second* moment
twice over, so the stau's diffusion-to-drift ratio collapses,
``d/b ~ m_0/m``, and the range distribution is narrow where the muon's is
broad: the "staus essentially do not lose energy stochastically" assumption
of the search becomes a computed number, the relative range spread
``sigma_X / X``, printed against the muon's. The horizontal transmission --
the probability of crossing the slant column with energy to spare -- follows
from the Gaussian log-loss law with the running ``alpha + beta E`` mean, and
with it the acceptance-weighted horizontal column per unit surface flux, the
transport factor a production pipeline (MCEq + MadGraph in the paper) folds
against the stau flux.

**Millicharge.** A particle of charge ``eps_q e`` scales every coefficient
-- ionization included -- by ``eps_q^2``: range grows as ``1 / eps_q^2``
while the deposit that makes the track visible shrinks by the same factor.
``--millicharge`` runs that variant at the muon mass.

Scope: transmission uses ice density along the whole slant path (the deep
horizontal chord is bedrock in reality); the stau is taken stable on the
crossing timescale (lifetimes with ``c tau gamma`` beyond ~100 km); the
production spectrum is out of scope, stated deliberately.

Usage
-----
    python scripts/future_bsm/60_stau_transport.py
    python scripts/future_bsm/60_stau_transport.py --millicharge 0.1 0.3
"""

import argparse
import importlib.util
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm as gauss

from softpaws.constants import CM_PER_KM  # noqa: F401  (unit note below)
from softpaws.transport.coefficients import proposal_loss_spectrum

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"

#: Masses [GeV]: the muon, and the stau scan.
M_MU = 0.10566
STAU_MASSES = (100.0, 200.0, 320.0, 450.0)

#: Ionization loss [GeV cm^2/g], mass-blind (RSS Eq. 4).
ALPHA_ION = 2.0e-3

#: Photonuclear cutoff mass [GeV]: the RSS Jacobian scale, set so the scaled
#: muon photonuclear coefficient meets their published stau curve
#: (2e-9 cm^2/g at 10^6 GeV for m = 150 GeV).
NUC_MASS_GEV = 0.38

#: RSS effective-beta parameterization (their Eqs. 24-25) for the validation.
RSS_BETA = (5.0e-9, 2.8e-10, 2.0e-10)
RSS_MASS_GEV = 150.0

#: Energy grid of the coefficient tables [GeV] and the y grid of the moments.
E_GRID = np.logspace(3.0, 10.0, 71)
Y_GRID = np.logspace(-7.0, -1e-6, 3001)

#: Minimum energy a particle must keep across the column [GeV] (RSS's E_0).
E_MIN_GEV = 1.0e3

#: IceCube depth [km], ice density [g/cm^3] and the zenith grid of the
#: horizontal window.
DEPTH_KM = 1.95
RHO_ICE = 0.92
EARTH_RADIUS_KM = 6371.0
ZENITH_DEG = np.linspace(80.0, 90.0, 41)

COLORS = {"muon": "0.35", 100.0: "#e7298a", 200.0: "#1b9e77", 320.0: "#7570b3",
          450.0: "#d95f02", "rss": "#e6ab02"}


def load_example(stem: str, name: str):
    """Import a sibling example module by file stem."""
    # The hubs this script loads live with the paper scripts; its own
    # siblings live here.
    path = _HERE / stem
    if not path.exists():
        path = _HERE.parent / "2026_muon_transport" / stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--millicharge", type=float, nargs="*", default=[],
                        help="Charges [e] for the millicharged variant at the muon mass.")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the figures, '60a' through '60c'.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Coefficients
# ---------------------------------------------------------------------------


def muon_channel_moments():
    """Per-channel first and second loss moments of the muon [cm^2/g].

    Returns
    -------
    moments : dict of str -> tuple(np.ndarray, np.ndarray)
        ``(beta, beta2)`` on :data:`E_GRID`, with ``beta = int y dGamma`` and
        ``beta2 = int y^2 dGamma`` per column density.
    """
    channels = ("bremsstrahlung", "pair production", "photonuclear")
    out = {ch: (np.empty(E_GRID.size), np.empty(E_GRID.size)) for ch in channels}
    for i, e in enumerate(E_GRID):
        spec = proposal_loss_spectrum(float(np.clip(e, 1.0e4, 1.0e8)), Y_GRID)
        for ch in channels:
            gamma = spec[ch]
            # The tabulated spectra are per km of water; per column density
            # [cm^2/g] divide by rho_water x 10^5 cm/km.
            out[ch][0][i] = np.trapezoid(gamma * Y_GRID, Y_GRID) / CM_PER_KM
            out[ch][1][i] = np.trapezoid(gamma * Y_GRID**2, Y_GRID) / CM_PER_KM
    return out


#: Per-channel mass scalings ``(power of m_mu/m for beta, extra y-suppression
#: scale m_0 [GeV])``: beta scales with the Jacobian, the second moment with
#: its square (the typical y itself scales as m_0 / m).
CHANNEL_SCALING = {"bremsstrahlung": 2.0, "pair production": 1.0, "photonuclear": 1.0}


def scaled_coefficients(moments, mass_gev: float, charge: float = 1.0):
    """``(beta(E), beta2(E), alpha)`` for a heavy charged particle [cm^2/g].

    Photonuclear uses the calibrated :data:`NUC_MASS_GEV` in place of the
    muon mass; the second moments carry one extra Jacobian power each.
    """
    jacobian = {"bremsstrahlung": (M_MU / mass_gev) ** 2,
                "pair production": M_MU / mass_gev,
                "photonuclear": min(NUC_MASS_GEV / mass_gev, 1.0)}
    #: Typical fractional loss per interaction, ``y_typ ~ m_0 / m``: the same
    #: kinematic cutoff that suppresses beta suppresses the loss size, so the
    #: second moment is ``beta x y_typ`` -- the Jacobian enters twice.
    y_scale = {"bremsstrahlung": M_MU, "pair production": M_MU,
               "photonuclear": NUC_MASS_GEV}
    beta = np.zeros(E_GRID.size)
    beta2 = np.zeros(E_GRID.size)
    for ch, (b1, b2) in moments.items():
        beta += b1 * jacobian[ch]
        if mass_gev > M_MU:
            beta2 += b1 * jacobian[ch] * min(y_scale[ch] / mass_gev, 1.0)
        else:
            beta2 += b2
    return charge**2 * beta, charge**2 * beta2, charge**2 * ALPHA_ION


def rss_beta(energy_gev, e0_gev=E_MIN_GEV, mass_gev=RSS_MASS_GEV):
    """RSS's fitted effective beta [cm^2/g]."""
    b0, b1, b2 = RSS_BETA
    return ((b0 + b1 * np.log(energy_gev / 1.0e10) + b2 * np.log(e0_gev / 1.0e3))
            * (RSS_MASS_GEV / mass_gev))


def csda_range_kmwe(beta, alpha, energy_gev, e0_gev=E_MIN_GEV):
    """``X = int dE / (alpha + beta(E) E)`` [km w.e.], on the running beta."""
    energy_gev = np.atleast_1d(energy_gev)
    out = np.empty(energy_gev.size)
    for i, e in enumerate(energy_gev):
        if e <= e0_gev:
            out[i] = 0.0
            continue
        x = np.logspace(np.log10(e0_gev), np.log10(e), 400)
        b = np.interp(np.log10(x), np.log10(E_GRID), beta)
        out[i] = np.trapezoid(1.0 / (alpha + b * x), x) * 1.0e-5
    return out


def range_spread(beta, beta2, alpha, energy_gev, e0_gev=E_MIN_GEV):
    """Relative range spread ``sigma_X / X`` from the loss variance.

    The first-passage variance of the CSDA-like descent: ``Var(X) =
    int dE beta2(E) E^2 / (alpha + beta E)^3`` (the log-loss variance per
    column, propagated along the mean trajectory).
    """
    energy_gev = np.atleast_1d(energy_gev)
    out = np.empty(energy_gev.size)
    for i, e in enumerate(energy_gev):
        if e <= e0_gev:
            out[i] = 0.0
            continue
        x = np.logspace(np.log10(e0_gev), np.log10(e), 400)
        b = np.interp(np.log10(x), np.log10(E_GRID), beta)
        b2 = np.interp(np.log10(x), np.log10(E_GRID), beta2)
        mean = np.trapezoid(1.0 / (alpha + b * x), x)
        var = np.trapezoid(b2 * x**2 / (alpha + b * x) ** 3, x)
        out[i] = np.sqrt(var) / mean
    return out


# ---------------------------------------------------------------------------
# Horizontal transmission
# ---------------------------------------------------------------------------


def slant_column_kmwe(zenith_deg):
    """Slant column to the surface [km w.e.], ice density throughout."""
    r0 = EARTH_RADIUS_KM - DEPTH_KM
    mu = np.cos(np.deg2rad(zenith_deg))
    length = -r0 * mu + np.sqrt((r0 * mu) ** 2 + EARTH_RADIUS_KM**2 - r0**2)
    return length * RHO_ICE


def transmission(beta, beta2, alpha, energy_gev: float, column_kmwe):
    """Probability of crossing the column with more than :data:`E_MIN_GEV`.

    The mean energy follows the running ``alpha + beta E`` descent; the
    fluctuation about it is Gaussian in the accumulated log-loss with the
    variance of :func:`range_spread`'s integrand. Exact for the muon within
    a few percent of the full kernel in this regime, and the stau's spread
    is so small the law barely matters.
    """
    column = np.atleast_1d(column_kmwe) * 1.0e5
    x_range = csda_range_kmwe(beta, alpha, np.array([energy_gev]))[0] * 1.0e5
    rel = range_spread(beta, beta2, alpha, np.array([energy_gev]))[0]
    sigma = rel * x_range
    return gauss.cdf((x_range - column) / np.maximum(sigma, 1.0e-30))


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


def figure_validation(moments, out_dir) -> None:
    """Left: our stau beta against RSS's fit. Right: ranges against theirs."""
    energy = np.logspace(4.0, 12.0, 100)
    with plt.style.context(str(_STYLE)):
        fig, (ax_b, ax_r) = plt.subplots(1, 2, figsize=(6.0, 2.8),
                                         gridspec_kw={"wspace": 0.3})
        for mass in (150.0, 250.0):
            beta, beta2, alpha = scaled_coefficients(moments, mass)
            b_on = np.interp(np.log10(energy), np.log10(E_GRID), beta)
            ls = "-" if mass == 150.0 else "--"
            ax_b.plot(energy, b_on, color="#e7298a", ls=ls, lw=1.2,
                      label=rf"this work, $m = {mass:.0f}$ GeV")
            ax_b.plot(energy, rss_beta(energy, mass_gev=mass), color=COLORS["rss"],
                      ls=ls, lw=1.1, label=rf"RSS fit, $m = {mass:.0f}$ GeV")
            ax_r.plot(energy, csda_range_kmwe(beta, alpha, energy), color="#e7298a",
                      ls=ls, lw=1.2)
            rss_b = rss_beta(energy, mass_gev=mass)
            ax_r.plot(energy, np.log(
                (ALPHA_ION + rss_b * energy) / (ALPHA_ION + rss_b * E_MIN_GEV))
                / rss_b * 1.0e-5, color=COLORS["rss"], ls=ls, lw=1.1)
        mu_beta, mu_beta2, _ = scaled_coefficients(moments, M_MU)
        ax_r.plot(energy, csda_range_kmwe(mu_beta, ALPHA_ION, energy), color=COLORS["muon"],
                  lw=1.0, ls=":", label="muon")
        ax_b.set_xscale("log")
        ax_b.set_yscale("log")
        ax_b.set_xlabel(r"$E$ [GeV]")
        ax_b.set_ylabel(r"$\beta_{\tilde\tau}$ [cm$^2$/g]")
        ax_b.legend(fontsize=5.5, frameon=False, loc="upper left")
        ax_r.set_xscale("log")
        ax_r.set_yscale("log")
        ax_r.set_xlabel(r"$E$ [GeV]")
        ax_r.set_ylabel(r"range to $10^3$ GeV [km w.e.]")
        ax_r.legend(fontsize=5.5, frameon=False, loc="lower right")
        _save(fig, out_dir, "60a_stau_validation")


def figure_transmission(moments, energy_gev, out_dir) -> None:
    """Transmission through the horizontal window, muon against staus."""
    column = slant_column_kmwe(ZENITH_DEG)
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.2, 3.0))
        mu = scaled_coefficients(moments, M_MU)
        ax.plot(ZENITH_DEG, transmission(*mu, energy_gev, column), color=COLORS["muon"],
                lw=1.2, ls=":", label=rf"muon, $10^{np.log10(energy_gev):.0f}$ GeV")
        for mass in STAU_MASSES:
            coefficients = scaled_coefficients(moments, mass)
            ax.plot(ZENITH_DEG, transmission(*coefficients, energy_gev, column),
                    color=COLORS[mass], lw=1.2, label=rf"$\tilde\tau$, {mass:.0f} GeV")
        ax.set_xlabel(r"zenith [deg]")
        ax.set_ylabel(r"$P$(cross with $E > 10^3$ GeV)")
        ax.set_ylim(0.0, 1.05)
        ax.legend(fontsize=6, frameon=False, loc="lower left",
                  title=rf"surface $E = 10^{np.log10(energy_gev):.0f}$ GeV",
                  title_fontsize=6)
        secax = ax.secondary_xaxis(
            "top", functions=(lambda z: np.interp(z, ZENITH_DEG, column),
                              lambda c: np.interp(c, column, ZENITH_DEG)))
        secax.set_xlabel("slant column [km w.e.]", fontsize=7)
        _save(fig, out_dir, "60b_stau_transmission")


def figure_acceptance(moments, charges, out_dir) -> None:
    """Horizon-window acceptance against mass, and the millicharged variant."""
    energies = np.logspace(4.0, 7.0, 31)
    column = slant_column_kmwe(ZENITH_DEG)
    weights = np.sin(np.deg2rad(ZENITH_DEG))

    def acceptance(coefficients):
        """Zenith-averaged crossing probability, per surface energy."""
        return np.array([np.average(transmission(*coefficients, e, column),
                                    weights=weights) for e in energies])

    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.2, 3.0))
        for mass in STAU_MASSES:
            ax.plot(energies, acceptance(scaled_coefficients(moments, mass)),
                    color=COLORS[mass], lw=1.2, label=rf"$\tilde\tau$, {mass:.0f} GeV")
        ax.plot(energies, acceptance(scaled_coefficients(moments, M_MU)),
                color=COLORS["muon"], lw=1.0, ls=":", label="muon")
        for q, ls in zip(charges, ("--", "-."), strict=False):
            ax.plot(energies, acceptance(scaled_coefficients(moments, M_MU, q)),
                    color="#66a61e", lw=1.1, ls=ls,
                    label=rf"millicharge $\varepsilon_q = {q:g}$")
        ax.set_xscale("log")
        ax.set_xlabel(r"surface energy [GeV]")
        ax.set_ylabel(r"horizon-window crossing fraction")
        ax.set_ylim(0.0, 1.05)
        ax.legend(fontsize=5.5, frameon=False, loc="upper left")
        _save(fig, out_dir, "60c_stau_acceptance")


def main() -> None:
    args = parse_args()
    print("Muon per-channel loss moments from the tabulated spectra ...")
    moments = muon_channel_moments()
    b_mu = sum(v[0] for v in moments.values())
    i6 = int(np.argmin(np.abs(E_GRID - 1.0e6)))
    print(f"  muon beta at 10^6 GeV: {b_mu[i6]:.3g} cm^2/g "
          f"(channels: " + ", ".join(f"{ch} {v[0][i6]:.2g}" for ch, v in moments.items())
          + ")")

    print("\nStau coefficients and RSS validation (their fit is good to ~20%):")
    print(f"  {'mass':>6} {'beta(1e6)':>10} {'RSS':>10} {'X(1e6)':>9} {'RSS':>9} "
          f"{'X(1e9)':>9} {'RSS':>9}  [km w.e.]")
    for mass in (150.0, 250.0):
        beta, beta2, alpha = scaled_coefficients(moments, mass)
        ours6 = float(np.interp(6.0, np.log10(E_GRID), beta))
        x6 = csda_range_kmwe(beta, alpha, np.array([1.0e6]))[0]
        x9 = csda_range_kmwe(beta, alpha, np.array([1.0e9]))[0]
        rb6 = float(rss_beta(1.0e6, mass_gev=mass))
        rx = [float(np.log((ALPHA_ION + rss_beta(e, mass_gev=mass) * e)
                           / (ALPHA_ION + rss_beta(e, mass_gev=mass) * E_MIN_GEV))
                    / rss_beta(e, mass_gev=mass) * 1.0e-5) for e in (1.0e6, 1.0e9)]
        print(f"  {mass:6.0f} {ours6:10.2e} {rb6:10.2e} {x6:9.0f} {rx[0]:9.0f} "
              f"{x9:9.0f} {rx[1]:9.0f}")

    print("\nRange spread sigma_X / X (the 'deterministic stau' assumption, computed):")
    print(f"  {'E [GeV]':>9} {'muon':>7} " + " ".join(f"{m:>7.0f}" for m in STAU_MASSES))
    mu_c = scaled_coefficients(moments, M_MU)
    stau_c = {m: scaled_coefficients(moments, m) for m in STAU_MASSES}
    for e in (1.0e5, 1.0e6, 1.0e7):
        row = [range_spread(*mu_c, np.array([e]))[0]]
        row += [range_spread(*stau_c[m], np.array([e]))[0] for m in STAU_MASSES]
        print(f"  {e:9.0e} " + " ".join(f"{v:7.3f}" for v in row))

    print("\nHorizontal window (80-90 deg, slant column "
          f"{slant_column_kmwe(80.0):.0f}-{slant_column_kmwe(90.0):.0f} km w.e.):")
    for e in (1.0e5, 1.0e6):
        mu_t = np.average(transmission(*mu_c, e, slant_column_kmwe(ZENITH_DEG)),
                          weights=np.sin(np.deg2rad(ZENITH_DEG)))
        line = f"  E {e:.0e}: muon {mu_t:.4f}"
        for m in STAU_MASSES:
            t = np.average(transmission(*stau_c[m], e, slant_column_kmwe(ZENITH_DEG)),
                           weights=np.sin(np.deg2rad(ZENITH_DEG)))
            line += f", {m:.0f} GeV {t:.3f}"
        print(line)

    print()
    figure_validation(moments, args.out_dir)
    figure_transmission(moments, 1.0e6, args.out_dir)
    figure_acceptance(moments, args.millicharge or [0.3], args.out_dir)


if __name__ == "__main__":
    main()
