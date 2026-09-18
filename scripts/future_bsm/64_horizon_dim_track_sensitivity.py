"""Example 64 -- dim tracks from the horizon: the IceCube stau sensitivity.

Example 63 closed the through-Earth route analytically; the sensitivity
lives just *above* the horizon, where tens of kilometres of ice thin the
atmospheric muons by orders of magnitude while a stau walks through -- the
window of Meighen-Berger et al., PLB 811 (2020) 135929. This example builds
that analysis from this project's kernel end to end and forecasts the IC86
sensitivity in the plane the collider community reads: stau mass against
the pair-production cross section at ``sqrt s = 13`` TeV.

**Backgrounds.** Two templates on a fine zenith grid over
:data:`ZENITH_DEG`.

* *Punch-through muons*: the Gaisser surface flux with the curved-Earth
  ``cos theta*`` (production height :data:`PROD_HEIGHT_KM`), propagated
  through the slant ice column with the same Gaussian log-loss law the
  signal transport uses -- the kernel computes its own background. A muon
  counts when it arrives inside the deposit-equivalent window
  :data:`ARRIVAL_WINDOW_GEV`. This component falls by orders of magnitude
  across the window as the column grows.
* *Neutrino floor*: near the horizon the atmospheric-neutrino tracks are
  zenith-flat; their level is taken data-driven from DR2's first upgoing
  band in the deposit window, converted per steradian. Free normalization.

**Signal.** ``mu x Phi_DY(E; m)`` -- example 63's analytic Drell-Yan
Z-moment flux times a signal strength -- arriving above the *ice*
``E_min(theta, m)`` (example 60's coefficients; transmission is a step to
percent accuracy) with the fitted selection efficiency. The stau deposit
(~0.2 TeV) sits inside the window at every energy.

**Sensitivity.** Asimov at background; both background normalizations
profiled with example 51's scaled-deviance systematic; ``mu_90`` at
``2 Delta lnL = 2.71``. The y-axis converts through example 63's
LHC-validated cross section: ``sigma_90(m) = mu_90(m) x sigma_DY(13 TeV,
m)``, overlaid with the theory curve itself, ATLAS's excluded masses
(120-390 GeV, 139 fb^-1, massless LSP) and the LEP floor.

The example-63 caveat carries: this signal model is *direct leading-order
Drell-Yan from the primary-nucleon chain*, which sits far below what the
PLB simulation found; until that is reconciled against the original MCEq
tables, the sensitivity here is the honest floor of this production model,
not a reproduction of the published limit.

Usage
-----
    python scripts/future_bsm/64_horizon_dim_track_sensitivity.py
    python scripts/future_bsm/64_horizon_dim_track_sensitivity.py --years 20
"""

import argparse
import importlib.util
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm as gauss

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_DATA_DIR = _HERE.parents[1] / "src" / "softpaws" / "data" / "dataverse_files"
_DEFAULT_OUT_DIR = _HERE / "output"

#: Horizon window [deg from vertical, downgoing] and its binning.
ZENITH_DEG = np.linspace(85.0, 90.0, 21)

#: Arrival-energy window equivalent to the 0.1-1 TeV deposit cut [GeV]: an
#: arriving muon (or stau) in this range leaves a track-like deposit inside
#: the published window over a ~1 km crossing.
ARRIVAL_WINDOW_GEV = (50.0, 3000.0)

#: Gaisser muon-flux constants and the production height for ``cos theta*``.
PROD_HEIGHT_KM = 32.0
EARTH_RADIUS_KM = 6371.0

#: Detector: depth, side-on area, ice density; selection efficiency from
#: example 45's fit.
DEPTH_KM = 1.95
SIDE_AREA_KM2 = 1.0
RHO_ICE = 0.92
EFFICIENCY = 0.755

#: Systematics: fractional per-bin template uncertainty (example 51's form)
#: and the priors on the two background normalizations.
MODEL_SYS = 0.15
MU_NORM_PRIOR = (1.0, 0.30)
NU_NORM_PRIOR = (1.0, 0.25)

#: Mass grid of the sensitivity curve [GeV], and the collider overlays. The
#: grid runs far beyond the collider reach on purpose: the atmospheric
#: ``sqrt s = sqrt(2 m_N E)`` grows with primary energy, while the LHC ends
#: at ``sqrt s / 2`` -- the high-mass half of this figure is where a
#: telescope search has no collider competition at any cross section.
MASSES_GEV = np.array([90.0, 100.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0,
                       1500.0, 2000.0, 3000.0, 5000.0])
ATLAS_EXCLUDED_GEV = (120.0, 390.0)
LEP_FLOOR_GEV = 90.0
LHC_KINEMATIC_GEV = 6500.0

COLORS = {"mu": "#7570b3", "nu": "0.45", "sig": "#e7298a", "theory": "#1b9e77",
          "atlas": "0.85"}


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


_EX63 = load_example("63_analytic_stau_limit.py", "_example_63")
_EX61 = _EX63._EX61
_EX60 = _EX63._EX60
_EX51 = _EX63._EX51


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=pathlib.Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--years", type=float, default=10.7,
                        help="Exposure [yr]; the IC86 livetime by default.")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the figures, '64a' and '64b'.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Geometry and the muon background
# ---------------------------------------------------------------------------


def ice_column_kmwe(zenith_deg):
    """Slant ice column from the detector to the surface [km w.e.]."""
    r0 = EARTH_RADIUS_KM - DEPTH_KM
    mu = np.cos(np.deg2rad(zenith_deg))
    length = -r0 * mu + np.sqrt((r0 * mu) ** 2 + EARTH_RADIUS_KM**2 - r0**2)
    return length * RHO_ICE


def cos_theta_star(zenith_deg):
    """Zenith at the production height, curved Earth."""
    r = EARTH_RADIUS_KM / (EARTH_RADIUS_KM + PROD_HEIGHT_KM)
    sin2 = (r * np.sin(np.deg2rad(zenith_deg))) ** 2
    return np.sqrt(np.clip(1.0 - sin2, 0.0, 1.0))


def gaisser_muon_flux(energy_gev, zenith_deg):
    """Surface muon flux [GeV^-1 cm^-2 s^-1 sr^-1], conventional."""
    e = np.asarray(energy_gev, dtype=float)
    cs = cos_theta_star(zenith_deg)
    return 0.14 * e**-2.7 * (1.0 / (1.0 + 1.1 * e * cs / 115.0)
                             + 0.054 / (1.0 + 1.1 * e * cs / 850.0))


def arrival_probability(moments, coefficients, energy_surface, column_kmwe, window):
    """P(arrive with energy inside ``window``) for one particle type.

    Crossing with more than ``E`` left requires the range from the surface
    energy down to ``E`` to exceed the column; the range fluctuates with the
    kernel's log-loss variance (Gaussian law, example 60).
    """
    beta, beta2, alpha = coefficients

    def p_above(e_floor):
        x_range = _EX60.csda_range_kmwe(beta, alpha, np.atleast_1d(energy_surface),
                                        e0_gev=e_floor)
        rel = _EX60.range_spread(beta, beta2, alpha, np.atleast_1d(energy_surface),
                                 e0_gev=e_floor)
        sigma = np.maximum(rel * x_range, 1.0e-12)
        return gauss.cdf((x_range - column_kmwe) / sigma)

    return np.clip(p_above(window[0]) - p_above(window[1]), 0.0, None)


def muon_background(moments, livetime_s):
    """Punch-through muons per zenith bin in the arrival window."""
    coefficients = _EX60.scaled_coefficients(moments, _EX60.M_MU)
    centers = 0.5 * (ZENITH_DEG[:-1] + ZENITH_DEG[1:])
    columns = ice_column_kmwe(centers)
    d_omega = 2.0 * np.pi * np.abs(np.diff(np.cos(np.deg2rad(ZENITH_DEG))))
    area_cm2 = SIDE_AREA_KM2 * 1.0e10
    counts = np.zeros(centers.size)
    for j, (theta, col) in enumerate(zip(centers, columns)):
        e_s = np.logspace(np.log10(ARRIVAL_WINDOW_GEV[0]), 8.0, 120)
        flux = gaisser_muon_flux(e_s, theta)
        p_window = arrival_probability(moments, coefficients, e_s, col,
                                       ARRIVAL_WINDOW_GEV)
        counts[j] = livetime_s * d_omega[j] * area_cm2 * np.trapezoid(flux * p_window, e_s)
    return counts


def neutrino_floor(data_dir, livetime_s):
    """Zenith-flat neutrino-track template from DR2's first upgoing band."""
    ex35 = load_example("35_point_source_effective_area.py", "_e35")
    ex45 = load_example("45_first_principles_reach.py", "_e45")
    ex46 = load_example("46_declination_resolved_reach.py", "_e46")
    _, dec_edges, _, _ = _EX51.fit_inputs(ex35, ex45, ex46, data_dir, False)
    counts = _EX51.binned_events(data_dir, dec_edges)
    reco_centers = 0.5 * (_EX51.RECO_EDGES[:-1] + _EX51.RECO_EDGES[1:])
    window = ((reco_centers >= _EX61.DEPOSIT_WINDOW[0])
              & (reco_centers <= _EX61.DEPOSIT_WINDOW[1]))
    first_band = counts[window].sum(axis=0)[0]
    omega_band = 2.0 * np.pi * (np.sin(np.deg2rad(dec_edges[1])) - np.sin(np.deg2rad(dec_edges[0])))
    per_sr = first_band / omega_band  # over the DR2 livetime, absorbed by `livetime_s` scaling
    per_sr *= livetime_s / (10.73 * 3.156e7)
    d_omega = 2.0 * np.pi * np.abs(np.diff(np.cos(np.deg2rad(ZENITH_DEG))))
    return per_sr * d_omega


def signal_counts(moments, mass_gev, livetime_s):
    """Stau counts per zenith bin at unit signal strength (LO Drell-Yan)."""
    centers = 0.5 * (ZENITH_DEG[:-1] + ZENITH_DEG[1:])
    columns = ice_column_kmwe(centers)
    d_omega = 2.0 * np.pi * np.abs(np.diff(np.cos(np.deg2rad(ZENITH_DEG))))
    area_cm2 = SIDE_AREA_KM2 * 1.0e10
    counts = np.zeros(centers.size)
    for j, col in enumerate(columns):
        e_min = _EX61.e_min_gev(moments, np.array([col]), float(mass_gev))[0]
        if not np.isfinite(e_min):
            continue
        grid = np.logspace(np.log10(max(e_min, 2.0 * mass_gev)), 8.0, 40)
        flux = _EX63.surface_stau_flux(grid, float(mass_gev))
        counts[j] = (livetime_s * d_omega[j] * area_cm2 * EFFICIENCY
                     * np.trapezoid(flux, grid))
    return counts


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------


def mu_90(b_mu, b_nu, signal_unit):
    """Signal strength at ``2 Delta lnL = 2.71`` on the background Asimov."""
    data = b_mu + b_nu

    def deviance(params, mu_s):
        a_mu, a_nu = params
        model = np.clip(a_mu * b_mu + a_nu * b_nu + mu_s * signal_unit, 1.0e-12, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            dev = 2.0 * (model - data + np.where(data > 0.0,
                                                 data * np.log(data / model), 0.0))
        penalty = (((a_mu - MU_NORM_PRIOR[0]) / MU_NORM_PRIOR[1]) ** 2
                   + ((a_nu - NU_NORM_PRIOR[0]) / NU_NORM_PRIOR[1]) ** 2)
        return float(np.sum(dev / (1.0 + MODEL_SYS**2 * model))) + penalty

    def profiled(mu_s):
        result = minimize(lambda p: deviance(p, mu_s), x0=np.array([1.0, 1.0]),
                          method="Nelder-Mead",
                          options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 4000})
        return result.fun

    base = profiled(0.0)
    total = signal_unit.sum()
    if total <= 0.0:
        return np.inf
    lo, hi = 0.0, 1.0 / total
    while profiled(hi) - base < 2.71:
        hi *= 4.0
        if hi * total > 1.0e12:
            return np.inf
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if profiled(mid) - base < 2.71:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _save(fig, out_dir: pathlib.Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {path.resolve()}")
    plt.close(fig)


def figure_rates(b_mu, b_nu, sig_example, out_dir) -> None:
    """Zenith rates: the two backgrounds and an example signal."""
    centers = 0.5 * (ZENITH_DEG[:-1] + ZENITH_DEG[1:])
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.2, 3.0))
        ax.step(centers, b_mu, where="mid", color=COLORS["mu"], lw=1.2,
                label="punch-through muons")
        ax.step(centers, b_nu, where="mid", color=COLORS["nu"], lw=1.0, ls=":",
                label="neutrino floor (data-driven)")
        ax.step(centers, sig_example, where="mid", color=COLORS["sig"], lw=1.2, ls="--",
                label=r"$\tilde\tau$, 100 GeV, $\mu = 10^{5}$")
        ax.set_yscale("log")
        ax.set_ylim(1.0e-4, None)
        ax.set_xlabel(r"zenith [deg]")
        ax.set_ylabel("events per bin")
        ax.legend(fontsize=6, frameon=False, loc="lower left")
        _save(fig, out_dir, "64a_horizon_rates")


def figure_sensitivity(masses, sigma90_fb, theory_fb, out_dir) -> None:
    """Mass against the 13 TeV pair cross section, collider overlays."""
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.4, 3.0))
        ax.axvspan(*ATLAS_EXCLUDED_GEV, color=COLORS["atlas"], zorder=0,
                   label=r"ATLAS excluded (139 fb$^{-1}$)")
        ax.axvline(LEP_FLOOR_GEV, color="0.55", lw=0.9, ls=":", label="LEP")
        ax.axvline(LHC_KINEMATIC_GEV, color="0.3", lw=1.0, ls="--")
        ax.text(LHC_KINEMATIC_GEV * 0.92, 3.0e-2, r"LHC kinematic wall ($\sqrt s / 2$)",
                rotation=90, fontsize=6, color="0.3", va="bottom", ha="right")
        ax.plot(masses, theory_fb, color=COLORS["theory"], lw=1.3,
                label=r"$\sigma_{\rm DY}$ (this work, LO$\times K$)")
        ax.plot(masses, sigma90_fb, color=COLORS["sig"], lw=1.3, marker="o", ms=3,
                label=r"IceCube horizon 90\% (this forecast)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(80.0, 1.1 * LHC_KINEMATIC_GEV)
        ax.set_xlabel(r"$m_{\tilde\tau}$ [GeV]")
        ax.set_ylabel(r"$\sigma(pp \to \tilde\tau\tilde\tau)$ at 13 TeV [fb]")
        ax.legend(fontsize=5.5, frameon=False, loc="center left")
        _save(fig, out_dir, "64b_horizon_sensitivity")


def main() -> None:
    args = parse_args()
    livetime_s = args.years * 3.156e7
    print(f"Horizon window {ZENITH_DEG[0]:.0f}-{ZENITH_DEG[-1]:.0f} deg, "
          f"ice column {ice_column_kmwe(ZENITH_DEG[0]):.0f}-"
          f"{ice_column_kmwe(ZENITH_DEG[-1]):.0f} km w.e., exposure {args.years:g} yr")
    moments = _EX60.muon_channel_moments()

    print("Punch-through muon background from the kernel ...")
    b_mu = muon_background(moments, livetime_s)
    b_nu = neutrino_floor(args.data_dir, livetime_s)
    centers = 0.5 * (ZENITH_DEG[:-1] + ZENITH_DEG[1:])
    for j in (0, 8, 14, 19):
        print(f"  zenith {centers[j]:5.2f}: muons {b_mu[j]:10.3g}, nu floor {b_nu[j]:8.3g}")

    print("\nSensitivity scan (LO Drell-Yan signal, signal strength mu):")
    print(f"  {'mass':>6} {'signal(mu=1)':>13} {'mu_90':>10} {'sigma_DY(13)':>13} "
          f"{'sigma_90':>11} [fb]")
    sigma90_fb, theory_fb = [], []
    for m in MASSES_GEV:
        unit = signal_counts(moments, m, livetime_s)
        limit = mu_90(b_mu, b_nu, unit)
        sigma_dy, _, _ = _EX63.dy_cross_section_cm2(13000.0**2, float(m))
        sigma_dy_fb = sigma_dy / 1.0e-39
        theory_fb.append(sigma_dy_fb)
        sigma90_fb.append(limit * sigma_dy_fb if np.isfinite(limit) else np.inf)
        print(f"  {m:6.0f} {unit.sum():13.3e} {limit:10.3g} {sigma_dy_fb:13.3g} "
              f"{sigma90_fb[-1]:11.3g}")

    print("\n  (Signal model = direct LO Drell-Yan from the primary chain; the example-63 "
          "discrepancy against the PLB simulation applies to the y-axis conversion.)")
    sig_example = 1.0e5 * signal_counts(moments, 100.0, livetime_s)
    figure_rates(b_mu, b_nu, sig_example, args.out_dir)
    figure_sensitivity(MASSES_GEV, np.array(sigma90_fb), np.array(theory_fb), args.out_dir)


if __name__ == "__main__":
    main()
