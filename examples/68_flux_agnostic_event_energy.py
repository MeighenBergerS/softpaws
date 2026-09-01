"""Example 68 -- KM3-230213A's energy with no flux assumption at all.

Example 57's neutrino-energy posterior carries a flux prior, and the
published 220 PeV median inherits it. This example asks what the event
alone says. The ingredients that survive with no flux are the muon
measurement folded through the exact kernel (example 57's
``energy_likelihood`` with the potential density -- no flux enters there),
the interaction probability ``sigma_CC(E_nu)``, and the survival through
the matter the neutrino had to cross,

.. math:: \\ell(E_\\nu) = \\sigma_{CC}(E_\\nu)\\, e^{-N_A \\sigma_{tot} X}
    \\int dw\\, u(w)\\, L\\bigl((1 - \\langle y\\rangle) E_\\nu e^{-w}\\bigr).

**Where the direction enters.** The muon is born within ~30 km w.e. of the
detector whatever the arrival direction (the potential density's support),
so the transport fold is column-blind; the 1.5 degree (68%) direction
uncertainty (KM3NeT, Nature 638 (2025) 376) acts only through the
absorption column ``X``. The elevation interval ``0.6 +- 1.5`` deg is
mapped to columns with a two-layer curved Earth (sea to
:data:`SEABED_DEPTH_KM`, rock below): at ``+2.1`` deg the path is all
water, ~80 km w.e. -- a floor, since topography can only add matter; at
``-0.9`` deg the ray dips through seabed rock for several hundred km w.e.
The central direction takes the release's own 309 km w.e. (the smooth
model gives ~160; the difference is the real seabed, printed as the model
check).

**The finding.** Flux-agnostic, the distribution is one-sided. The muon
measurement sets a firm lower edge (tens of PeV), the transport fold is
flat per decade above it (``u ~ 1/b``: a 100 PeV and a 10 EeV parent make
the observed muon about equally well), and ``sigma_CC`` *rises* -- so the
likelihood grows with energy until absorption cuts it, at a few EeV along
the longest allowed path and beyond 100 EeV along the shortest. The
published median sits two decades below that cliff: the ceiling on
KM3-230213A's energy is the flux prior's, not the detector's. The overlay
of example 57's ``E^-2`` posterior makes the contrast the figure's point.

**The loss-law twist (figure 68b).** The lower edge is itself a model
statement. Under example 59's extra loss ``kappa = 1 + eps (E/PeV)^n``
the fold changes in exactly two places -- the light matches the datum at
``E_mu kappa(E_mu) ~ 120 PeV`` and the occupation density tilts by
``1 / kappa`` -- while the neutrino side (``sigma_CC``, absorption) is
untouched, so the right side of the picture stands and the left edge
slides down by ``kappa`` at the reinterpreted muon energy. Drawn at what
DR2 allows (example 67's :data:`~_example_67.EPS_DRAWN`), the
flux-agnostic floor drops from ~100 PeV to a few PeV. Together the two
panels say what the event alone knows: the ceiling belongs to the flux
assumption, the floor to the loss law, and the kernel makes each
assumption's role explicit.

Usage
-----
    python examples/68_flux_agnostic_event_energy.py
    python examples/68_flux_agnostic_event_energy.py --sigma-dir 3.0
"""

import argparse
import importlib.util
import pathlib

import matplotlib.pyplot as plt
import numpy as np

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"


def load_example(stem: str, name: str):
    """Import a sibling example module by file stem."""
    spec = importlib.util.spec_from_file_location(name, _HERE / stem)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: Example 57 supplies the kernel fold, the event constants and (through its
#: example-31 import) the neutrino cross sections; example 67 supplies the
#: loss laws DR2 allows (``EPS_DRAWN``) and, through it, example 59's kappa.
_EX57 = load_example("57_km3_event_energy_and_bpl_tension.py", "_example_57")
_EX67 = load_example("67_loss_model_energy_reconstruction.py", "_example_67")

#: Direction uncertainty [deg, 68%] (systematics-dominated; KM3NeT, Nature
#: 638 (2025) 376) about the ``+0.6`` deg elevation.
SIGMA_DIR_DEG = 1.5

#: Two-layer Earth: seabed depth [km] (rock below, sea above) and densities
#: [g cm^-3]; the detector sits at example 57's ``DEPTH_KM``.
SEABED_DEPTH_KM = 3.5
RHO_SEA, RHO_ROCK = 1.03, 2.65

#: Neutrino-energy grid [GeV], and the wider grid of the loss-law figure
#: (the BSM lower edges sit at a few PeV).
ENU_GRID = np.logspace(7.0, 11.0, 240)
ENU_GRID_BSM = np.logspace(6.4, 11.0, 260)

COLORS = {"transport": "0.55", "short": "#1b9e77", "central": "#7570b3",
          "long": "#e7298a", "prior": "0.2",
          "sm_family": ("0.3", "0.55", "0.75"),
          0.5: "#1b9e77", 1.0: "#7570b3", 2.0: "#e7298a"}


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sigma-dir", type=float, default=SIGMA_DIR_DEG,
                        help="Direction uncertainty [deg, 68%%].")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the figure, '68a'.")
    return parser.parse_args()


def column_gcm2(elevation_deg: float, ds_km: float = 0.05) -> tuple:
    """Matter column from the detector to the sea surface [g/cm^2].

    Two-layer curved Earth: rock below :data:`SEABED_DEPTH_KM`, sea above,
    the ray leaving the detector at ``elevation_deg`` above the local
    horizontal (negative = below).

    Returns
    -------
    total, water, rock : float
        Total, water and rock columns [g/cm^2], and the path length [km]
        as the fourth element.
    """
    r0 = _EX57.EARTH_RADIUS_KM - _EX57.DEPTH_KM
    r_bed = _EX57.EARTH_RADIUS_KM - SEABED_DEPTH_KM
    sin_a = np.sin(np.deg2rad(elevation_deg))
    water = rock = 0.0
    s = 0.0
    while True:
        s += ds_km
        r = np.sqrt(r0**2 + s**2 + 2.0 * r0 * s * sin_a)
        if r >= _EX57.EARTH_RADIUS_KM:
            break
        if r > r_bed:
            water += ds_km
        else:
            rock += ds_km
        if s > 3.0e3:  # safety: not a chord this example should see
            break
    to_gcm2 = 1.0e5
    return (water * RHO_SEA + rock * RHO_ROCK) * to_gcm2, water, rock, s


def survival(energy_nu, column_g_cm2: float) -> np.ndarray:
    """Survival through ``column_g_cm2`` of isoscalar matter."""
    sigma = (_EX57._EX31.CROSS_SECTION.cc(energy_nu)
             + _EX57._EX31.CROSS_SECTION.nc(energy_nu))
    return np.exp(-_EX57.AVOGADRO * sigma * column_g_cm2)


def tau_unity_gev(column_g_cm2: float) -> float:
    """Energy at which the column is one interaction length."""
    target = 1.0 / (_EX57.AVOGADRO * column_g_cm2)
    grid = np.logspace(6.0, 12.0, 400)
    sigma = _EX57._EX31.CROSS_SECTION.cc(grid) + _EX57._EX31.CROSS_SECTION.nc(grid)
    if sigma[-1] < target:
        return np.inf
    return float(np.exp(np.interp(np.log(target), np.log(sigma), np.log(grid))))


def bsm_fold(u: np.ndarray, energy_nu, eps: float, n: float) -> np.ndarray:
    """Example 57's event fold under the extra loss ``kappa``.

    Two insertions, both from example 59's kernel algebra: the light of a
    muon with true energy ``E_mu`` reads ``E_mu kappa(E_mu)`` against the
    standard calibration, and the occupation density scales as
    ``u / kappa`` (faster losses spend less column at every loss depth).
    ``eps = 0`` reproduces ``energy_likelihood(..., normalize=False)``.
    """
    energy_nu = np.atleast_1d(np.asarray(energy_nu, dtype=float))
    entry = (1.0 - np.squeeze(_EX57.mean_inelasticity(energy_nu))) * energy_nu
    log_e_mu = np.log(entry)[:, None] - _EX57.W_GRID[None, :]
    e_mu = np.exp(log_e_mu)
    kap = _EX67._EX59.kappa(e_mu, eps, n)
    like = _EX57.muon_measurement(np.log10(e_mu * kap))
    return np.trapezoid((u[None, :] / kap) * like, _EX57.W_GRID, axis=1)


def _save(fig, out_dir: pathlib.Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {path.resolve()}")
    plt.close(fig)


def figure_likelihood(curves, prior_curve, out_dir) -> None:
    """Max-normalized flux-agnostic likelihoods and the ``E^-2`` posterior."""
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.5, 3.1))
        # The three direction curves share one scale (the family maximum), so
        # more matter reads as a lower curve; the flux-free fold and the
        # posterior have different units and carry their own maxima.
        shared = max(v.max() for _, (c, ls, v) in curves.items() if ls == "--")
        for label, (color, ls, values) in curves.items():
            norm = values.max() if ls == ":" else shared
            ax.plot(ENU_GRID, values / norm, color=color, lw=1.2, ls=ls,
                    label=label)
        ax.plot(ENU_GRID, prior_curve / prior_curve.max(), color=COLORS["prior"],
                lw=1.0, ls="-", alpha=0.85,
                label=r"with $E^{-2}$ flux prior (example 57)")
        med, lo, hi = (v * 1.0e6 for v in _EX57.KM3NET_ENU_PEV)
        ax.axvline(med, color=COLORS["prior"], lw=0.6, ls=":")
        ax.text(med * 1.15, 2.0e-3, "KM3NeT median", rotation=90, fontsize=4.8,
                color=COLORS["prior"], va="bottom")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(ENU_GRID[0], ENU_GRID[-1])
        ax.set_ylim(1.0e-3, 2.5)
        ax.set_xlabel(r"$E_\nu$ [GeV]")
        ax.set_ylabel(r"likelihood per $\ln E_\nu$ (normalized)")
        ax.legend(fontsize=5.0, frameon=False, loc="upper left")
        _save(fig, out_dir, "68a_flux_agnostic_energy")


def figure_bsm(sm_curves, bsm_curves, out_dir) -> None:
    """Figure 68b: the flux-agnostic picture under the DR2-allowed loss laws.

    The standard-loss trio (grey, one per direction) shares its family
    maximum so more matter reads lower; each loss law is its own model, so
    each BSM curve carries its own maximum -- only the shapes compare.
    """
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.5, 3.1))
        shared = max(v.max() for _, v in sm_curves.items())
        for (label, values), color in zip(sm_curves.items(), COLORS["sm_family"]):
            ax.plot(ENU_GRID_BSM, values / shared, color=color, lw=1.0, ls="--",
                    label=label)
        # The three loss laws coincide wherever the fold is flat; distinct
        # styles keep each visible where they part (the left edge).
        styles = {0.5: ("-", 1.6), 1.0: ("--", 1.2), 2.0: (":", 1.4)}
        for (n, eps, tag), values in bsm_curves.items():
            ls, lw = styles[n]
            ax.plot(ENU_GRID_BSM, values / values.max(), color=COLORS[n], lw=lw, ls=ls,
                    label=rf"$\kappa = 1 + {eps:g}\,(E/\mathrm{{PeV}})^{{{n:g}}}$ ({tag})")
        med = _EX57.KM3NET_ENU_PEV[0] * 1.0e6
        ax.axvline(med, color=COLORS["prior"], lw=0.6, ls=":")
        ax.text(med * 1.15, 2.0e-3, "KM3NeT median", rotation=90, fontsize=4.8,
                color=COLORS["prior"], va="bottom")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(ENU_GRID_BSM[0], ENU_GRID_BSM[-1])
        ax.set_ylim(1.0e-3, 2.5)
        ax.set_xlabel(r"$E_\nu$ [GeV]")
        ax.set_ylabel(r"likelihood per $\ln E_\nu$ (normalized)")
        ax.legend(fontsize=5.0, frameon=False, loc="upper left",
                  title="standard losses (grey) vs extra loss", title_fontsize=5.0)
        _save(fig, out_dir, "68b_flux_agnostic_bsm")


def main() -> None:
    args = parse_args()
    elevations = {"short": _EX57.ELEVATION_DEG + args.sigma_dir,
                  "central": _EX57.ELEVATION_DEG,
                  "long": _EX57.ELEVATION_DEG - args.sigma_dir}
    print(f"Direction: elevation {_EX57.ELEVATION_DEG:g} +- {args.sigma_dir:g} deg (68%)")
    columns = {}
    print(f"  {'case':>8} {'elev [deg]':>11} {'path [km]':>10} {'water [km]':>11} "
          f"{'rock [km]':>10} {'column [km w.e.]':>17} {'tau=1 at [GeV]':>15}")
    for case, elev in elevations.items():
        total, water, rock, path = column_gcm2(elev)
        if case == "central":
            model_kmwe = total / 1.0e5
            total = _EX57.TRAVERSED_COLUMN_KMWE * 1.0e5  # the release's own value
        columns[case] = total
        print(f"  {case:>8} {elev:11.1f} {path:10.1f} {water:11.1f} {rock:10.1f} "
              f"{total / 1.0e5:17.0f} {tau_unity_gev(total):15.3g}")
    print(f"  (smooth-seabed model at the central direction: {model_kmwe:.0f} km w.e.; "
          f"the release's 309 includes the real topography -- the model is a floor)")

    print("\nBuilding the exact-kernel potential density (example 57) ...")
    u = _EX57.potential_density("exact")
    transport = _EX57.energy_likelihood(u, ENU_GRID, normalize=False)
    sigma_cc = _EX57._EX31.CROSS_SECTION.cc(ENU_GRID)

    curves = {"transport only (no flux, no column)":
              (COLORS["transport"], ":", transport)}
    for case, label in (("short", "-1 sigma matter"), ("central", "central direction"),
                        ("long", "+1 sigma matter")):
        values = transport * sigma_cc * survival(ENU_GRID, columns[case])
        kmwe = columns[case] / 1.0e5
        curves[f"{label}, {kmwe:.0f} km w.e."] = (COLORS[case], "--", values)

    prior_curve = (transport * sigma_cc * survival(ENU_GRID, columns["central"])
                   * _EX57.flux_shape("E^-2", ENU_GRID) * ENU_GRID)  # per ln E

    print("\nFlux-agnostic verdicts (likelihood-ratio 0.1 edges, per ln E):")
    for label, (color, ls, values) in list(curves.items())[1:]:
        norm = values / values.max()
        above = ENU_GRID[norm > 0.1]
        hi = f"{above[-1]:.3g}" if above[-1] < ENU_GRID[-1] * 0.99 else "none in window"
        print(f"  {label:>34}: lower edge {above[0]:.3g} GeV, upper edge {hi}")
    print("  (the lower edge is the muon measurement; the upper edge, where it exists, "
          "is absorption)")
    figure_likelihood(curves, prior_curve, args.out_dir)

    # Figure 68b: the same picture under the loss laws DR2 allows (example
    # 67), central column for the BSM curves -- the loss law only moves the
    # left edge, the direction only the right, so the families factorize.
    sigma_cc_b = _EX57._EX31.CROSS_SECTION.cc(ENU_GRID_BSM)
    transport_b = _EX57.energy_likelihood(u, ENU_GRID_BSM, normalize=False)
    sm_curves = {}
    for case, label in (("short", "-1 sigma matter"), ("central", "central direction"),
                        ("long", "+1 sigma matter")):
        sm_curves[f"standard, {label}"] = (transport_b * sigma_cc_b
                                           * survival(ENU_GRID_BSM, columns[case]))
    bsm_curves = {}
    print("\nUnder the DR2-allowed loss laws (central column), the floor moves:")
    for n, (eps, tag) in _EX67.EPS_DRAWN.items():
        values = (bsm_fold(u, ENU_GRID_BSM, eps, n) * sigma_cc_b
                  * survival(ENU_GRID_BSM, columns["central"]))
        bsm_curves[(n, eps, tag)] = values  # tag keeps its LaTeX escape
        above = ENU_GRID_BSM[values / values.max() > 0.1]
        print(f"  n = {n:g}, eps = {eps:g} ({tag.replace(chr(92), '')}): "
              f"lower edge {above[0]:.3g} GeV")
    figure_bsm(sm_curves, bsm_curves, args.out_dir)


if __name__ == "__main__":
    main()
