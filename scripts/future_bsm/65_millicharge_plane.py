"""Example 65 -- the millicharged plane from the horizon dim-track analysis.

Example 64 forecasts the horizon analysis for staus; a millicharged (or
fractionally charged) particle changes three dials at once, every one of
them already in the machinery: production scales as ``eps_q^2`` (the
Drell-Yan coupling), the losses scale as ``eps_q^2`` (so the Earth and the
ice open up -- ``E_min`` falls), and the light yield scales as ``eps_q^2``
(so the selection efficiency falls, example 61's anchored model). This
example scans the ``(mass, eps_q)`` plane and places the result next to
the collider searches.

**Where this plane sits in the landscape.** The well-known deep
millicharge constraints -- SLAC mQ (``eps ~ 1e-5``-``0.1`` for 0.1-1000
MeV), ArgoNeuT (``1e-3``-``0.1`` for 0.1-3 GeV), milliQan, SENSEI,
Super-Kamiokande (to ~1.5 GeV), SN1987A and cosmology at still lighter
masses -- all live at **masses below a few GeV**, where production is
meson decay (``pi^0``/``eta`` Dalitz, onia) with its enormous shower
multiplicities and detection is single faint scatters. Above ~10 GeV the
mesons are gone, production is Drell-Yan, and the deep-``eps`` bounds end;
the frontier is the CMS fractional-charge search. This example's plane
starts at 60 GeV on purpose: it is the heavy corner, where those famous
``eps ~ 1e-4`` limits do not apply. Reaching small ``eps`` at *low* mass
with a telescope needs the meson-decay production (the same Z-moment
algebra) and a faint-hit selection (the Faint Particle Trigger) -- a
different analysis, out of scope here.

**Why this plane is interesting.** CMS's fractional-charge search
(arXiv:2402.09932, 138 fb^-1, Drell-Yan production) excludes ``2e/3`` up
to 640 GeV -- but ``e/3`` only up to **60 GeV**: the dim track falls out
of the collider tracker exactly as it falls out of a standard telescope
selection. And above ``sqrt s / 2`` the collider has no reach at any
charge. The telescope's two openings are the low-charge edge (down to the
selection floor, ``eps_q ~ 0.25`` for a DR2-like selection; DeepCore's
Faint Particle Trigger extends lower) and the high-mass side, where
``E_min`` saturates at the ionization value and only the production
threshold matters.

**Signal.** A Dirac-fermion pair through the photon,
``sigma_hat = (4 pi alpha^2 e_q^2 eps_q^2 / 3 hat s) beta (3 - beta^2)/2
/ 3``, folded exactly as in example 63 (same PDFs, same Z-moment
atmospheric chain); the ``eps_q^2`` of the production factors out, so the
flux is computed once per mass. Transport and selection then follow
example 64 with the charge-scaled coefficients and efficiency.

**Output.** Contours of ``log10 mu_90`` -- the factor by which this
production model would need to be enhanced for a 90% exclusion -- on the
``(m, eps_q)`` plane, with the CMS exclusion (both anchors quoted above,
log-interpolated between), the LEP floor, the selection floor and the LHC
kinematic wall. Under direct leading-order Drell-Yan the telescope does
not reach the model anywhere (``mu_90 >= 10^8``; the example-63 production
caveat applies on top), so the figure is a sensitivity map and an honest
statement of where the openings are, not an exclusion.

Usage
-----
    python examples/65_millicharge_plane.py
    python examples/65_millicharge_plane.py --years 30
"""

import argparse
import importlib.util
import pathlib

import matplotlib.pyplot as plt
import numpy as np

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_DATA_DIR = _HERE.parents[1] / "src" / "softpaws" / "data" / "dataverse_files"
_DEFAULT_OUT_DIR = _HERE / "output"

#: Scan grids: mass [GeV] and charge [e].
MASSES_GEV = np.array([60.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0])
CHARGES = np.array([0.25, 1.0 / 3.0, 0.4, 0.5, 2.0 / 3.0, 0.8, 1.0])

#: CMS fractional-charge exclusion anchors (arXiv:2402.09932): (charge, max
#: excluded mass [GeV]); log-interpolated between, flat outside.
CMS_ANCHORS = ((1.0 / 3.0, 60.0), (0.5, 400.0), (2.0 / 3.0, 640.0), (0.9, 660.0))
LEP_FLOOR_GEV = 90.0
LHC_KINEMATIC_GEV = 6500.0

COLORS = {"cms": "0.82", "sens": "#e7298a", "floor": "#7570b3"}


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


_EX64 = load_example("64_horizon_dim_track_sensitivity.py", "_example_64")
_EX63 = _EX64._EX63
_EX61 = _EX64._EX61
_EX60 = _EX64._EX60


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=pathlib.Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--years", type=float, default=10.7)
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the figure, '65a'.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Fermion-pair Drell-Yan and the unit-charge flux per mass
# ---------------------------------------------------------------------------


def sigma_hat_fermion_cm2(s_hat, mass_gev):
    """Dirac-fermion pair through the photon, unit millicharge [cm^2]."""
    beta2 = 1.0 - 4.0 * mass_gev**2 / s_hat
    beta = np.sqrt(np.clip(beta2, 0.0, None))
    return np.where(beta2 > 0.0,
                    (4.0 * np.pi * _EX63.ALPHA_EM**2 / (3.0 * s_hat)
                     * beta * (3.0 - beta2) / 2.0 / 3.0 * _EX63.GEV2_TO_CM2),
                    0.0)


def surface_mcp_flux(energy_gev, mass_gev):
    """Unit-charge millicharged surface flux [GeV^-1 cm^-2 s^-1 sr^-1].

    Example 63's Z-moment chain with the fermion partonic cross section;
    multiply by ``eps_q^2`` for the physical charge.
    """
    original = _EX63.sigma_hat_cm2
    _EX63.sigma_hat_cm2 = sigma_hat_fermion_cm2
    try:
        return _EX63.surface_stau_flux(energy_gev, mass_gev)
    finally:
        _EX63.sigma_hat_cm2 = original


def scan_plane(moments, livetime_s, b_mu, b_nu):
    """``log10 mu_90`` on the ``(mass, charge)`` grid."""
    centers = 0.5 * (_EX64.ZENITH_DEG[:-1] + _EX64.ZENITH_DEG[1:])
    columns = _EX64.ice_column_kmwe(centers)
    d_omega = 2.0 * np.pi * np.abs(np.diff(np.cos(np.deg2rad(_EX64.ZENITH_DEG))))
    area_cm2 = _EX64.SIDE_AREA_KM2 * 1.0e10

    out = np.full((MASSES_GEV.size, CHARGES.size), np.inf)
    for i, m in enumerate(MASSES_GEV):
        grid = np.logspace(np.log10(2.0 * m) + 0.3, 8.0, 50)
        flux_unit = surface_mcp_flux(grid, float(m))
        print(f"  m {m:6.0f} GeV: unit-charge flux at grid peak "
              f"{flux_unit.max():.3e}")
        for k, q in enumerate(CHARGES):
            eff = float(_EX61.dim_track_efficiency(float(q))[0])
            if eff < 1.0e-4:
                continue
            counts = np.zeros(centers.size)
            for j, col in enumerate(columns):
                e_min = _EX61.e_min_gev(moments, np.array([col]), float(m), float(q))[0]
                if not np.isfinite(e_min):
                    continue
                inside = grid >= e_min
                if inside.sum() < 2:
                    continue
                counts[j] = (livetime_s * d_omega[j] * area_cm2 * eff * q**2
                             * np.trapezoid(flux_unit[inside], grid[inside]))
            limit = _EX64.mu_90(b_mu, b_nu, counts)
            out[i, k] = np.log10(limit) if np.isfinite(limit) and limit > 0 else np.inf
    return out


def cms_boundary(charge):
    """CMS max excluded mass at a charge, from the anchors [GeV]."""
    q = np.array([a[0] for a in CMS_ANCHORS])
    m = np.array([a[1] for a in CMS_ANCHORS])
    return np.where(charge < q[0], 0.0,
                    10.0 ** np.interp(charge, q, np.log10(m)))


def _save(fig, out_dir: pathlib.Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {path.resolve()}")
    plt.close(fig)


def figure_plane(log_mu90, out_dir) -> None:
    """The ``(m, eps_q)`` plane: sensitivity contours and the collider regions."""
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.6, 3.1))
        q_fine = np.linspace(CHARGES[0], 1.0, 100)
        ax.fill_betweenx(q_fine, 1.0, cms_boundary(q_fine), color=COLORS["cms"],
                         zorder=0, label=r"CMS excluded (138 fb$^{-1}$)")
        ax.axvline(LEP_FLOOR_GEV, color="0.55", lw=0.9, ls=":", label="LEP")
        ax.axvline(LHC_KINEMATIC_GEV, color="0.3", lw=1.0, ls="--")
        ax.text(LHC_KINEMATIC_GEV * 0.9, 0.3, r"LHC wall", rotation=90, fontsize=6,
                color="0.3", va="bottom", ha="right")
        ax.axhline(CHARGES[0], color=COLORS["floor"], lw=1.0, ls="-.")
        ax.text(70.0, CHARGES[0] * 1.03, "DR2-like selection floor (FPT extends lower)",
                fontsize=5.5, color=COLORS["floor"], va="bottom")
        finite = np.isfinite(log_mu90)
        levels = np.arange(np.floor(log_mu90[finite].min()), 22.0, 2.0)
        cs = ax.contour(MASSES_GEV, CHARGES, log_mu90.T, levels=levels,
                        colors=COLORS["sens"], linewidths=0.9)
        ax.clabel(cs, fontsize=5, fmt=r"$10^{%.0f}$")
        ax.plot([], [], color=COLORS["sens"], lw=0.9,
                label=r"IceCube horizon $\mu_{90}$ (this forecast)")
        ax.set_xscale("log")
        ax.set_xlim(55.0, 1.1 * LHC_KINEMATIC_GEV)
        ax.set_ylim(0.22, 1.02)
        ax.set_xlabel(r"mass [GeV]")
        ax.set_ylabel(r"charge $\varepsilon_q$ [e]")
        ax.legend(fontsize=5.5, frameon=False, loc="upper right")
        _save(fig, out_dir, "65a_millicharge_plane")


def main() -> None:
    args = parse_args()
    livetime_s = args.years * 3.156e7
    print(f"Horizon window, exposure {args.years:g} yr; charges "
          f"{CHARGES.round(2).tolist()}, masses {MASSES_GEV.astype(int).tolist()}")
    moments = _EX60.muon_channel_moments()
    b_mu = _EX64.muon_background(moments, livetime_s)
    b_nu = _EX64.neutrino_floor(args.data_dir, livetime_s)

    print("\nScanning the (mass, charge) plane (fermion Drell-Yan, unit strength) ...")
    log_mu90 = scan_plane(moments, livetime_s, b_mu, b_nu)

    print(f"\n  log10 mu_90 (rows = mass, columns = charge {CHARGES.round(2).tolist()}):")
    for i, m in enumerate(MASSES_GEV):
        row = " ".join("   inf" if not np.isfinite(v) else f"{v:6.1f}" for v in log_mu90[i])
        print(f"  m {m:6.0f}: {row}")
    best = np.unravel_index(int(np.nanargmin(np.where(np.isfinite(log_mu90), log_mu90,
                                                      np.nan))), log_mu90.shape)
    print(f"\n  most sensitive point: m = {MASSES_GEV[best[0]]:.0f} GeV, eps_q = "
          f"{CHARGES[best[1]]:.2f}, mu_90 = 10^{log_mu90[best]:.1f}")
    print("  (Under direct LO Drell-Yan nothing is excluded; the plane maps where the "
          "openings are: the low-charge edge CMS loses at e/3, and the far side of the "
          "LHC wall. The example-63 production caveat applies.)")

    figure_plane(log_mu90, args.out_dir)


if __name__ == "__main__":
    main()
