"""Example 61 -- an angle-dependent flux bound on heavy charged particles.

Example 60 gives the transport: a stau (or any heavy charged particle)
crosses Earth columns that absorb every muon, arrives with a percent-level
range spread, and deposits a fixed ~0.2 TeV crossing the detector. This
example turns that into a measurement without touching production physics:
whatever makes such particles, the number of track-like events IceCube
recorded from a direction caps their flux from that direction.

**The logic.** Because the particle's losses are quasi-deterministic
(example 60: range spread below 2%), the Earth is a step function: a
particle of surface energy above ``E_min(theta, m)`` -- the energy whose
range equals the PREM slant column -- arrives; below, it does not. The
in-detector signature is a through-going track with a deposit near
``alpha rho L ~ 0.2`` TeV, inside the IceTracks-DR2 sample's low
reconstructed energies. Counting *every* DR2 event in that window as a
candidate (no background subtraction, which is what makes the statement
agnostic), the 90% Poisson ceiling per declination band divided by the
band's exposure is an upper limit on the integral surface flux,

.. math:: \\Phi(> E_{\\rm min}(\\theta, m)) \\le
    \\frac{N_{90}(\\theta)}{T\\,\\Omega_{\\rm band}\\,\\bar A(\\theta)},

with ``A(theta)`` the detector's projected area. All the mass (or charge)
dependence lives in ``E_min``; the flux ceiling itself is one curve per
band. The result is reported both ways: the ceiling against zenith, and
``E_min`` against zenith per mass, which together are the full statement.

**Millicharge and the dim-track selection.** The same bound applies to a
millicharged particle with every coefficient scaled by ``eps_q^2`` -- the
columns become easier to cross, so ``E_min`` drops -- but the light yield
falls by the same ``eps_q^2``, so the selection efficiency must follow the
charge. The literature pins two anchors: a full minimum-ionizing muon is
IceCube's energy-scale calibration source, so ``eps_q = 1`` is fully
efficient; and IceCube's own fractional-charge search found the standard
trigger "significantly less efficient" at ``e/3`` -- 11% of the MIP light
(Van Driessche, PhD, Ghent 2019, via the Faint Particle Trigger paper,
arXiv:2411.00484). :func:`dim_track_efficiency` is a per-module Poisson
hit model on the string lattice with one free normalization set so that
``e/3`` sits at 50% efficiency, full charge at ~1; the flux ceiling per
charge divides by it. DeepCore's Faint Particle Trigger (deployed 2023)
would extend below ``e/3`` on the denser lattice; that selection is noted,
not modelled.

The candidate window, the projected area and the ceiling are deliberately
crude in the conservative direction: no selection efficiency is credited
(a smaller true efficiency would weaken the bound; quoting it at 1 is the
agnostic choice only for the *counting* side, so the printed variant with
example 45's fitted efficiency is the realistic one), and every observed
event counts as signal.

Usage
-----
    python scripts/future_bsm/61_heavy_track_flux_bound.py
    python scripts/future_bsm/61_heavy_track_flux_bound.py --deposit-window 2.0 3.25
"""

import argparse
import importlib.util
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.data.loader import compute_livetime_s, load_uptime
from softpaws.transport.attenuation import prem_column

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_DATA_DIR = _HERE.parents[1] / "src" / "softpaws" / "data" / "dataverse_files"
_DEFAULT_OUT_DIR = _HERE / "output"

#: Reconstructed-energy window of the candidate counting [log10 GeV]: the
#: crossing deposit (~0.2 TeV) with room for the radiative sprinkles and
#: the proxy resolution.
DEPOSIT_WINDOW = (2.0, 3.25)

#: Crossing deposit of a unit-charge quasi-MIP [GeV] (alpha rho L over ~1 km).
UNIT_DEPOSIT_GEV = 200.0

#: Dim-track selection model: modules a through-going track passes per
#: 125 m string spacing (60 DOMs within ~60 m over 1 km), the coincidence
#: count of the SMT8-like condition, and the anchor: charge 1/3 (11% of
#: MIP light) at 50% efficiency, per IceCube's fractional-charge search.
N_MODULES_NEAR_TRACK = 60
SMT_MODULES = 8
ANCHOR_CHARGE = 1.0 / 3.0
ANCHOR_EFFICIENCY = 0.5

#: Masses [GeV] and millicharges [e] of the E_min curves.
MASSES_GEV = (100.0, 200.0, 320.0, 450.0)
CHARGES = (1.0, 0.8, 0.7)

#: IceCube footprint [km^2], instrumented height [km].
FOOTPRINT_KM2 = 1.0
HEIGHT_KM = 1.0

COLORS = {100.0: "#e7298a", 200.0: "#1b9e77", 320.0: "#7570b3", 450.0: "#d95f02",
          "bound": "#e7298a", "eff": "#7570b3"}


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


_EX60 = load_example("60_stau_transport.py", "_example_60")
_EX51 = load_example("51_dr2_flavor_fit.py", "_example_51")


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=pathlib.Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--deposit-window", type=float, nargs=2, default=list(DEPOSIT_WINDOW),
                        help="Candidate reco window [log10 GeV].")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the figures, '61a' and '61b'.")
    return parser.parse_args()


def band_columns(dec_edges_deg):
    """PREM slant column at each band centre [km w.e.]; upgoing bands."""
    centers = 0.5 * (dec_edges_deg[:-1] + dec_edges_deg[1:])
    return centers, np.array([prem_column(float(d)) for d in centers]) / 1.0e5


def e_min_gev(moments, column_kmwe, mass_gev, charge=1.0):
    """Surface energy whose range equals the column, by bisection [GeV]."""
    beta, _, alpha = _EX60.scaled_coefficients(moments, mass_gev, charge)

    def range_of(log10_e):
        return _EX60.csda_range_kmwe(beta, alpha, np.array([10.0**log10_e]))[0]

    out = np.empty(np.atleast_1d(column_kmwe).shape)
    for i, col in enumerate(np.atleast_1d(column_kmwe)):
        lo, hi = 3.05, 12.0
        if range_of(hi) < col:
            out[i] = np.nan
            continue
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if range_of(mid) < col:
                lo = mid
            else:
                hi = mid
        out[i] = 10.0**hi
    return out


def projected_area_km2(dec_deg):
    """Upright-cylinder projection at the band's zenith (``theta = 90 + dec``)."""
    radius = np.sqrt(FOOTPRINT_KM2 / np.pi)
    cos_t = np.abs(np.cos(np.deg2rad(90.0 + np.asarray(dec_deg))))
    sin_t = np.sqrt(1.0 - cos_t**2)
    return np.pi * radius**2 * cos_t + 2.0 * radius * HEIGHT_KM * sin_t


def _mip_module_mean() -> float:
    """Per-module mean PE of a full MIP, set by the e/3 anchor.

    With ``N`` modules near the track and a hit probability
    ``1 - exp(-eps^2 mu)`` each, the SMT-like condition (``>= 8`` hits) is
    binomial; ``mu`` is solved so :data:`ANCHOR_CHARGE` gives
    :data:`ANCHOR_EFFICIENCY`.
    """
    from scipy.stats import binom

    def efficiency(mu, charge):
        p_hit = 1.0 - np.exp(-charge**2 * mu)
        return float(binom.sf(SMT_MODULES - 1, N_MODULES_NEAR_TRACK, p_hit))

    lo, hi = 0.01, 50.0
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        if efficiency(mid, ANCHOR_CHARGE) < ANCHOR_EFFICIENCY:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def dim_track_efficiency(charge) -> np.ndarray:
    """Selection efficiency for a through-going track of charge ``eps_q e``."""
    from scipy.stats import binom
    mu = _mip_module_mean()
    charge = np.atleast_1d(np.asarray(charge, dtype=float))
    p_hit = 1.0 - np.exp(-charge**2 * mu)
    return binom.sf(SMT_MODULES - 1, N_MODULES_NEAR_TRACK, p_hit)


def poisson_ul90(n_obs):
    """90% upper limit on the mean with all events as candidates.

    Exact for small counts (tabulated), Gaussian ``n + 1.28 sqrt(n)`` above.
    """
    table = {0: 2.30, 1: 3.89, 2: 5.32, 3: 6.68, 4: 7.99, 5: 9.27}
    n_obs = np.atleast_1d(n_obs)
    out = np.empty(n_obs.shape)
    for i, n in enumerate(n_obs):
        out[i] = table.get(int(n), n + 1.28 * np.sqrt(n)) if n <= 5 else n + 1.28 * np.sqrt(n)
    return out


def _save(fig, out_dir: pathlib.Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {path.resolve()}")
    plt.close(fig)


def figure_bound(centers, flux_ul, flux_ul_eff, out_dir) -> None:
    """The flux ceiling per band, agnostic and with the fitted efficiency."""
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.2, 3.0))
        ax.step(90.0 + centers, flux_ul, where="mid", color=COLORS["bound"], lw=1.3,
                label="agnostic (efficiency 1)")
        ax.step(90.0 + centers, flux_ul_eff, where="mid", color=COLORS["eff"], lw=1.1,
                ls="--", label=f"with fitted efficiency {_EX51.FITTED_NORMALIZATION:g}")
        ax.set_yscale("log")
        ax.set_xlabel(r"zenith [deg]")
        ax.set_ylabel(r"$\Phi_{90}(>E_{\min})$ [cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
        ax.legend(fontsize=6, frameon=False, loc="upper right")
        _save(fig, out_dir, "61a_heavy_track_flux_bound")


def figure_emin(centers, moments, columns, out_dir) -> None:
    """``E_min`` against zenith per mass, and the millicharge variants."""
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.2, 3.0))
        for mass in MASSES_GEV:
            ax.plot(90.0 + centers, e_min_gev(moments, columns, mass), color=COLORS[mass],
                    lw=1.2, label=rf"$\tilde\tau$, {mass:.0f} GeV")
        for q, ls in zip((0.8, 0.7), ("--", ":"), strict=False):
            deposit = q**2 * UNIT_DEPOSIT_GEV
            label = (rf"$\varepsilon_q = {q:g}$ ({deposit:.0f} GeV deposit)")
            ax.plot(90.0 + centers, e_min_gev(moments, columns, MASSES_GEV[0], q),
                    color="#66a61e", lw=1.1, ls=ls, label=label)
        ax.set_yscale("log")
        ax.set_xlabel(r"zenith [deg]")
        ax.set_ylabel(r"$E_{\min}$ [GeV]")
        ax.legend(fontsize=5.5, frameon=False, loc="upper left")
        _save(fig, out_dir, "61b_heavy_track_emin")


def main() -> None:
    args = parse_args()
    window = tuple(args.deposit_window)
    print("Muon channel moments (example 60) ...")
    moments = _EX60.muon_channel_moments()

    print("DR2 candidates per declination band in the deposit window "
          f"[{window[0]:g}, {window[1]:g}] log10 GeV ...")
    enu_edges, dec_edges, _, _ = _EX51.fit_inputs(
        load_example("35_point_source_effective_area.py", "_e35"),
        load_example("45_first_principles_reach.py", "_e45"),
        load_example("46_declination_resolved_reach.py", "_e46"), args.data_dir, False)
    counts = _EX51.binned_events(args.data_dir, dec_edges)
    reco_centers = 0.5 * (_EX51.RECO_EDGES[:-1] + _EX51.RECO_EDGES[1:])
    in_window = (reco_centers >= window[0]) & (reco_centers <= window[1])
    n_band = counts[in_window].sum(axis=0)
    livetime_s = sum(compute_livetime_s(load_uptime(args.data_dir / "uptime" / f"{s}_exp.csv"))
                     for s in _EX51.IC86_SEASONS)

    centers, columns = band_columns(dec_edges)
    d_omega = 2.0 * np.pi * np.diff(np.sin(np.deg2rad(dec_edges)))
    area_cm2 = projected_area_km2(centers) * 1.0e10
    exposure = livetime_s * d_omega * area_cm2
    ul90 = poisson_ul90(n_band)
    flux_ul = ul90 / exposure
    flux_ul_eff = flux_ul / _EX51.FITTED_NORMALIZATION

    print(f"\n  {'zenith':>8} {'column':>9} {'N_obs':>8} {'Phi_90':>10}   E_min [GeV] per mass "
          + " / ".join(f"{m:.0f}" for m in MASSES_GEV))
    for j, (dec, col) in enumerate(zip(centers, columns, strict=True)):
        e_mins = [e_min_gev(moments, np.array([col]), m)[0] for m in MASSES_GEV]
        print(f"  {90 + dec:8.1f} {col:9.0f} {n_band[j]:8.0f} {flux_ul[j]:10.2e}   "
              + " / ".join(f"{e:.2e}" for e in e_mins))

    print("\nDim-track selection efficiency (Poisson hit model, e/3 anchored at "
          f"{ANCHOR_EFFICIENCY:.0%}):")
    charges = np.array([1.0, 0.8, 0.6, 0.5, 0.4, 1.0 / 3.0, 0.3, 0.25, 0.2])
    eff = dim_track_efficiency(charges)
    for q, e in zip(charges, eff, strict=True):
        deposit = q**2 * UNIT_DEPOSIT_GEV
        note = "" if e > 0.01 else "   (out of reach for this selection)"
        print(f"  eps_q {q:5.2f}: efficiency {e:8.3f}, deposit {deposit:6.1f} GeV, "
          f"ceiling x {1.0 / max(e, 1e-12):8.3g}{note}")
    print("  DeepCore's Faint Particle Trigger (2023) extends below e/3 on the denser "
          "lattice; not modelled here.")

    print()
    figure_bound(centers, flux_ul, flux_ul_eff, args.out_dir)
    figure_emin(centers, moments, columns, args.out_dir)


if __name__ == "__main__":
    main()
