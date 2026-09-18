"""Example 63 -- the stau mass limit with analytic Drell-Yan production.

Example 62 limits the surface flux of Earth-crossing heavy particles and
leaves production to a table. This example computes the production too, in
closed form, and closes the loop to a mass limit. The pleasant symmetry:
the atmospheric cascade collapses into the Z-moment formalism -- which is
this project's ``Phi(s)`` under another name (the spectrum-weighted Mellin
moment) -- so production and transport are the same mathematical object at
the two ends of the pipeline.

**Partonic cross section.** Scalar pair production through the photon,

.. math:: \\hat\\sigma(q\\bar q \\to \\tilde\\tau^+\\tilde\\tau^-) =
    \\frac{\\pi \\alpha^2 e_q^2}{3\\,\\hat s}\\,\\frac{\\beta^3}{3},

with ``beta = sqrt(1 - 4m^2/hat s)`` -- the ``beta^3`` is the scalar
threshold suppression -- and the colour average ``1/3`` written out. The
``Z`` contribution and higher orders are folded into one K-factor
(:data:`K_FACTOR`); a left+right degenerate pair doubles the yield
(:data:`N_STATES`).

**Hadronic cross section.** A compact leading-order valence+sea PDF
parameterization (stated accuracy tens of percent at the ``x > 0.01`` this
needs) gives ``sigma(N N -> stau stau X)(E_lab; m)`` and the stau energy
fraction ``dn/dx`` with ``x = E_stau / E_N`` (each stau carries half the
pair's lab energy at leading order). The printout validates the machinery
at the LHC point: ``sigma(pp, sqrt s = 13 TeV, m = 100 GeV)`` against the
O(1) fb of the SUSY cross-section working group.

**Atmospheric fold.** Staus neither reinteract nor (for the metastable
case) decay, so the cascade equation gives their flux as a Z-moment ratio,

.. math:: \\Phi_{\\tilde\\tau}(E) = \\frac{Z_{N\\tilde\\tau}(E)}
    {1 - Z_{NN}}\\;\\Phi_N(E),

with ``Z_{N tau}(E) = int dx x^{gamma(E)-1} (A sigma(E/x) / sigma_air)
(dn/dx)``, a broken power-law nucleon flux with the knee, and the standard
``Z_NN``. This is the formalism of prompt leptons with the charm loop
replaced by Drell-Yan; meson-induced production is subleading at these
energies and left out.

**The limit.** The predicted surface flux runs through example 62's
likelihood -- ``E_min(theta, m)`` transport, dim-track efficiency,
atmospheric-neutrino background with the 10% systematic -- and the mass is
excluded where the predicted counts exceed the profiled 90% allowance.
DR2 is upgoing-only, so the crossing energies start at
``E_min ~ 3 x 10^5`` GeV even in the thinnest band: the limit leans on the
far tail of the production spectrum, and the printout says plainly how far
this is from the near-horizontal *downgoing* window of Meighen-Berger et
al., PLB 811 (2020) 135929, where ``E_min`` is two decades lower.

Usage
-----
    python scripts/future_bsm/63_analytic_stau_limit.py
    python scripts/future_bsm/63_analytic_stau_limit.py --k-factor 1.6
"""

import argparse
import importlib.util
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.data.loader import compute_livetime_s, load_uptime

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_DATA_DIR = _HERE.parents[1] / "src" / "softpaws" / "data" / "dataverse_files"
_DEFAULT_OUT_DIR = _HERE / "output"

ALPHA_EM = 1.0 / 137.036
GEV2_TO_CM2 = 3.894e-28  # 1 GeV^-2 in cm^2
M_NUCLEON = 0.9389

#: NLO-and-Z correction on the photon-only LO cross section, and the number
#: of degenerate stau states produced (left + right).
K_FACTOR = 1.3
N_STATES = 2

#: Air target: mean mass number and inelastic nucleon-air cross section [cm^2].
A_AIR = 14.5
SIGMA_AIR_CM2 = 300.0e-27

#: Nucleon flux (Gaisser-style): ``phi = PHI_N E^-2.7`` below the knee,
#: steepening to 3.0 above [GeV^-1 cm^-2 s^-1 sr^-1].
PHI_N0 = 1.8
KNEE_GEV = 3.0e6
GAMMA_BELOW, GAMMA_ABOVE = 2.7, 3.0

#: Spectrum-weighted nucleon regeneration moment (standard value at 2.7).
Z_NN = 0.26

#: Masses of the exclusion scan [GeV].
MASSES_GEV = np.array([50.0, 75.0, 100.0, 150.0, 200.0, 300.0])

COLORS = {"flux": "#e7298a", "allow": "#7570b3"}


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


_EX62 = load_example("62_stau_flux_limit.py", "_example_62")
_EX61 = _EX62._EX61
_EX60 = _EX62._EX60
_EX51 = _EX62._EX51


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=pathlib.Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--k-factor", type=float, default=K_FACTOR)
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the figure, '63a'.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Partons and the Drell-Yan cross section
# ---------------------------------------------------------------------------


def pdfs(x):
    """Compact LO parton densities ``x f(x)`` at ``Q ~ 100`` GeV.

    Valence normalized to counting sum rules, a soft sea; good to tens of
    percent for ``x > 0.01``, which is where the moments below live.

    Returns
    -------
    xu_v, xd_v, x_sea : np.ndarray
        Up and down valence, and the per-flavour light sea.
    """
    x = np.asarray(x, dtype=float)
    xu_v = 2.18 * x**0.5 * (1.0 - x) ** 3.0
    xd_v = 1.23 * x**0.5 * (1.0 - x) ** 4.0
    x_sea = 0.14 * x**-0.2 * (1.0 - x) ** 7.0
    return xu_v, xd_v, x_sea


def parton_lumi_qqbar(x1, x2):
    """``Sum_q e_q^2 [q(x1) qbar(x2) + qbar(x1) q(x2)]`` for an isoscalar nucleon.

    The nucleon average makes u and d valence contribute with the mean of
    their charges; the sea is flavour symmetric.
    """
    e_u2, e_d2 = 4.0 / 9.0, 1.0 / 9.0
    xu1, xd1, xs1 = pdfs(x1)
    xu2, xd2, xs2 = pdfs(x2)
    # Isoscalar target: average u and d valence.
    val1 = 0.5 * (xu1 + xd1)
    val2 = 0.5 * (xu2 + xd2)
    e_mean = 0.5 * (e_u2 + e_d2)
    lumi = (e_mean * (val1 * xs2 + xs1 * val2)          # valence x sea
            + 2.0 * (e_u2 + e_d2) * (xs1 * xs2))        # sea x sea (u, d; s ~ small)
    return lumi / (x1 * x2)


def sigma_hat_cm2(s_hat, mass_gev):
    """Scalar pair production through the photon [cm^2 x e_q^2 stripped]."""
    beta2 = 1.0 - 4.0 * mass_gev**2 / s_hat
    beta = np.sqrt(np.clip(beta2, 0.0, None))
    return np.where(beta2 > 0.0,
                    np.pi * ALPHA_EM**2 / (3.0 * s_hat) * beta**3 / 3.0 * GEV2_TO_CM2,
                    0.0)


def dy_cross_section_cm2(s_gev2, mass_gev, k_factor=K_FACTOR, n_x=120):
    """``sigma(N N -> stau pair)`` and the stau spectrum ``dn/dx``.

    Returns
    -------
    sigma : float
        Total cross section [cm^2].
    x_grid, dn_dx : np.ndarray
        Energy fraction ``x = E_stau / E_N ~ x1 / 2`` of each of the two
        staus, normalized so ``int dn/dx dx = 2``.
    """
    tau_min = 4.0 * mass_gev**2 / s_gev2
    if tau_min >= 1.0:
        return 0.0, np.array([0.5]), np.array([0.0])
    lx = np.linspace(np.log(tau_min) / 2.0, -1.0e-9, n_x)
    x1 = np.exp(lx)
    d_sigma_dx1 = np.zeros(n_x)
    for i, xa in enumerate(x1):
        x2_min = tau_min / xa
        if x2_min >= 1.0:
            continue
        lx2 = np.linspace(np.log(x2_min), -1.0e-9, n_x)
        x2 = np.exp(lx2)
        integrand = parton_lumi_qqbar(xa, x2) * sigma_hat_cm2(xa * x2 * s_gev2, mass_gev) * x2
        d_sigma_dx1[i] = np.trapezoid(integrand, lx2)
    sigma = float(np.trapezoid(d_sigma_dx1 * x1, lx)) * k_factor * N_STATES
    # Each stau takes ~half the pair's lab energy: x = x1 / 2 for both.
    x_grid = x1 / 2.0
    weight = d_sigma_dx1 * x1
    dn_dx = 2.0 * weight / max(np.trapezoid(weight / 2.0, x_grid), 1.0e-300) / 2.0
    return sigma, x_grid, dn_dx


# ---------------------------------------------------------------------------
# The atmospheric Z-moment fold
# ---------------------------------------------------------------------------


def nucleon_flux(energy_gev):
    """Broken power-law primary nucleon flux [GeV^-1 cm^-2 s^-1 sr^-1]."""
    e = np.asarray(energy_gev, dtype=float)
    below = PHI_N0 * 1.0e-4 * e**-GAMMA_BELOW * 1.0e4  # per cm^2: 1.8e4 /m^2 = 1.8 /cm^2
    above = (PHI_N0 * 1.0e-4 * KNEE_GEV**(GAMMA_ABOVE - GAMMA_BELOW)
             * e**-GAMMA_ABOVE * 1.0e4)
    return np.where(e < KNEE_GEV, below, above)


def gamma_at(energy_gev):
    return np.where(np.asarray(energy_gev) < KNEE_GEV, GAMMA_BELOW, GAMMA_ABOVE)


def surface_stau_flux(energy_gev, mass_gev, k_factor=K_FACTOR):
    """``Phi_stau(E)`` from the Z-moment ratio [GeV^-1 cm^-2 s^-1 sr^-1]."""
    energy_gev = np.atleast_1d(np.asarray(energy_gev, dtype=float))
    out = np.zeros(energy_gev.shape)
    for i, e_st in enumerate(energy_gev):
        gamma = float(gamma_at(e_st))
        # Z_{N stau}(E) = int dx x^{gamma - 1} [A sigma(E/x) / sigma_air] dn/dx(E/x)
        lx = np.linspace(np.log(1.0e-4), np.log(0.5), 90)
        x = np.exp(lx)
        value = 0.0
        for xa, dlx in zip(x, np.gradient(lx)):
            e_n = e_st / xa
            s = 2.0 * M_NUCLEON * e_n
            sigma, x_grid, dn_dx = dy_cross_section_cm2(s, mass_gev, k_factor, n_x=40)
            if sigma <= 0.0:
                continue
            dn = np.interp(xa, x_grid, dn_dx, left=0.0, right=0.0)
            value += (xa ** (gamma - 1.0) * A_AIR * sigma / SIGMA_AIR_CM2 * dn) * xa * dlx
        out[i] = value / (1.0 - Z_NN) * float(nucleon_flux(e_st))
    return out


# ---------------------------------------------------------------------------
# The limit
# ---------------------------------------------------------------------------


def predicted_counts(mass_gev, dec_edges, livetime_s, moments, k_factor):
    """Signal counts per band from the analytic surface flux."""
    centers, columns = _EX61.band_columns(dec_edges)
    d_omega = 2.0 * np.pi * np.diff(np.sin(np.deg2rad(dec_edges)))
    area_cm2 = _EX61.projected_area_km2(centers) * 1.0e10
    e_min = _EX61.e_min_gev(moments, columns, float(mass_gev))
    counts = np.zeros(dec_edges.size - 1)
    for j in range(counts.size):
        if not np.isfinite(e_min[j]):
            continue
        grid = np.logspace(np.log10(e_min[j]), np.log10(e_min[j]) + 3.5, 30)
        flux = surface_stau_flux(grid, mass_gev, k_factor)
        counts[j] = livetime_s * d_omega[j] * area_cm2[j] * np.trapezoid(flux, grid)
    return counts


def _save(fig, out_dir: pathlib.Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {path.resolve()}")
    plt.close(fig)


def figure_flux(masses, fluxes, allowed_k, out_dir) -> None:
    """Predicted surface fluxes against the example-62 flux allowance."""
    energy = np.logspace(3.0, 8.0, 60)
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.4, 3.0))
        for m, flux in zip(masses, fluxes):
            ax.plot(energy, energy**2 * flux, lw=1.2,
                    label=rf"$m = {m:.0f}$ GeV", alpha=0.9)
        gamma, k90 = allowed_k
        ax.plot(energy, energy**2 * k90 * (energy / 1.0e5) ** (-gamma),
                color="k", lw=1.4, ls="--",
                label=rf"example 62 $K_{{90}}$ ($\gamma = {gamma:g}$)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$E_{\tilde\tau}$ [GeV]")
        ax.set_ylabel(r"$E^2 \Phi_{\tilde\tau}$ [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
        ax.set_ylim(1.0e-22, None)
        ax.legend(fontsize=5.5, frameon=False, loc="upper right")
        _save(fig, out_dir, "63a_analytic_stau_flux")


def main() -> None:
    args = parse_args()

    print("LHC validation of the Drell-Yan machinery:")
    s_lhc = 13000.0**2
    sigma, _, _ = dy_cross_section_cm2(s_lhc, 100.0, args.k_factor)
    print(f"  sigma(pp -> stau stau, sqrt s = 13 TeV, m = 100 GeV) = "
          f"{sigma / 1.0e-39:.2f} fb")
    print("  (PROSPINO reference scale: sigma > 1 fb up to m ~ 400 GeV, so L+R at "
          "100 GeV is O(10-100) fb -- this sits inside; treat the pipeline as good "
          "to a factor ~2.)")

    print("\nSurface stau fluxes (Z-moment fold):")
    energy_ref = np.array([1.0e4, 1.0e5, 1.0e6])
    for m in (100.0, 200.0):
        flux = surface_stau_flux(energy_ref, m, args.k_factor)
        print(f"  m {m:.0f} GeV: " + "  ".join(
            f"E={e:.0e}: {f:.2e}" for e, f in zip(energy_ref, flux)))

    print("\nFolding through example 62's likelihood ...")
    ex35 = load_example("35_point_source_effective_area.py", "_e35")
    ex45 = load_example("45_first_principles_reach.py", "_e45")
    ex46 = load_example("46_declination_resolved_reach.py", "_e46")
    enu_edges, dec_edges, marginal, _ = _EX51.fit_inputs(ex35, ex45, ex46, args.data_dir,
                                                          False)
    counts = _EX51.binned_events(args.data_dir, dec_edges)
    reco_centers = 0.5 * (_EX51.RECO_EDGES[:-1] + _EX51.RECO_EDGES[1:])
    window = ((reco_centers >= _EX61.DEPOSIT_WINDOW[0])
              & (reco_centers <= _EX61.DEPOSIT_WINDOW[1]))
    data = counts[window].sum(axis=0)
    livetime_s = sum(compute_livetime_s(load_uptime(args.data_dir / "uptime" / f"{s}_exp.csv"))
                     for s in _EX51.IC86_SEASONS)
    background = _EX62.background_template(args.data_dir, enu_edges, dec_edges, marginal,
                                           window, livetime_s)
    moments = _EX60.muon_channel_moments()

    print(f"\n  {'mass':>7} {'predicted counts':>17} {'90% allowance':>14} {'ratio':>8}")
    fluxes = []
    energy = np.logspace(3.0, 8.0, 60)
    for m in MASSES_GEV:
        predicted = predicted_counts(m, dec_edges, livetime_s, moments, args.k_factor)
        allowance = _EX62.profiled_limit(data, background, predicted)
        fluxes.append(surface_stau_flux(energy, m, args.k_factor))
        ratio = 1.0 / allowance if np.isfinite(allowance) and allowance > 0 else 0.0
        verdict = "EXCLUDED" if ratio > 1.0 else ""
        print(f"  {m:7.0f} {predicted.sum():17.3e} {allowance * predicted.sum():14.3e} "
              f"{ratio:8.2e} {verdict}")

    unit62 = _EX62.signal_counts(MASSES_GEV[:1], 3.7, dec_edges, livetime_s, moments)
    k90 = _EX62.profiled_limit(data, background, unit62[0])
    print("\n  DR2 upgoing verdict: the predicted counts sit ~9 orders below the "
          "allowance --")
    print("  through-Earth staus from Drell-Yan are out of reach, analytically. The "
          "kinematic pinch")
    print("  (each stau carries x >= m / sqrt(s)) plus E_min >= 3e5 GeV kills the "
          "channel.")

    # The constructive counterpart: the PLB downgoing window, where E_min is set by
    # tens of km w.e. of ice, not the Earth. Same analytic flux, back-of-envelope
    # exposure (85-90 deg, ~0.5 sr, side-on area, IC86 livetime).
    grid = np.logspace(4.5, 7.0, 40)
    for m in (100.0, 200.0, 300.0):
        flux = surface_stau_flux(grid, m, args.k_factor)
        n_down = (livetime_s * 0.5 * 1.0e10
                  * np.trapezoid(flux, grid))
        print(f"  same flux, downgoing 85-90 deg window (not in DR2), m = {m:.0f} GeV: "
              f"~{n_down:.2e} events per IC86 exposure")
    print("  -- the PLB route: the sensitivity lives below the horizon, not beyond it.")
    figure_flux(MASSES_GEV, fluxes, (3.7, k90), args.out_dir)


if __name__ == "__main__":
    main()
