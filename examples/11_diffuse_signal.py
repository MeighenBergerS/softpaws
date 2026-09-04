"""Tutorial 11 -- injecting a new diffuse signal.

Tutorials 05 and 09 built the response and calibrated it. This one runs it
backwards. Given a signal nobody has seen yet, how large does it have to be
before each telescope sees it?

Three injections cover most of what a proposal asks for. A single event is the
weakest claim an experiment can make, and the flux behind it is one event per
decade of neutrino energy. A line is a flux that arrives at one energy, which
is what a decaying or annihilating particle of fixed mass gives. A power law
spreads the same normalization over the whole band. All three are linear in
the normalization, so each is one division by the exposure.

Every sensitivity is run twice: once on the geometry alone, with nothing
fitted, and once with the one instrument number of tutorial 09 fitted at each
site against its own published effective area. The gap between the two is what
the instrument, rather than the transport, contributes to the answer.

Usage
-----
    python examples/11_diffuse_signal.py
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.detectors import ARCA230, ICECUBE, PONE, TRIDENT
from softpaws.fluxes import ICECUBE_TRACKS_2022, power_law_flux
from softpaws.response.declination import (
    REACH_FRACTIONS,
    REACH_PIVOT_GEV,
    REACH_REFERENCE_GEV,
    central_energy_range,
    directional_effective_area_cm2,
    fit_light_reach,
)
from softpaws.response.sensitivity import (
    N_EVENTS_LIMIT,
    PIVOT_ENERGY_GEV,
    line_sensitivity,
    power_law_sensitivity,
    single_event_sensitivity,
)
from softpaws.response.site_models import published_effective_area_cm2
from softpaws.transport.earth import zenith_grid

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"

#: Neutrino energies every curve here is built on [log10 GeV].
LOG10_E = np.arange(4.0, 8.01, 0.25)

#: Band the instrument number is fitted over [log10 GeV]. Published nodes
#: outside a site's own digitized range are skipped by the fit.
FIT_BAND = (5.0, 7.5)

#: Directions the fit and the sky average run over. The fit needs only enough
#: to average the published convention, since it is scanned many times.
N_ZENITH_FIT = 8
N_ZENITH = 24

#: Sites compared, in the order they are drawn.
SITES = (ICECUBE, ARCA230, PONE, TRIDENT)

#: Sky a diffuse flux covers [sr].
SKY_SR = 4.0 * np.pi

#: Energies the tables are quoted at [GeV].
ANCHOR_GEV = 1.0e6

#: Spectral indices scanned for the power-law injection.
GAMMA_GRID = np.linspace(1.5, 3.0, 16)

#: Spectral index of the power-law table.
GAMMA = 2.0

#: Window the power-law ceiling is recomputed over, to show what buying a
#: background-free search costs [log10 GeV].
RAISED_WINDOW_LOG10_GEV = 6.0

YEAR_S = 365.25 * 86400.0


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threshold-gev", type=float, default=1.0e3,
                        help="Muon selection threshold.")
    parser.add_argument("--livetime-yr", type=float, default=10.0,
                        help="Exposure of the search.")
    parser.add_argument("--gamma", type=float, default=GAMMA,
                        help="Spectral index of the power-law injection.")
    parser.add_argument("--emin-gev", type=float, default=None,
                        help="Bottom of the analysis window; the whole grid by default.")
    parser.add_argument("--data-dir", type=pathlib.Path, default=None,
                        help="Root of the DR2 release, for IceCube's published table.")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the figures.")
    return parser.parse_args()


def fit_site(site, data_dir: pathlib.Path | None, threshold_gev: float) -> tuple[float, float]:
    """Fit the one instrument number of tutorial 09 at one site.

    The published curve carries the sky it was averaged over, so the model is
    averaged the same way before the two are compared.

    Parameters
    ----------
    site : softpaws.detectors.Site
        The detector.
    data_dir : pathlib.Path or None
        Root of the DR2 release, read for IceCube alone.
    threshold_gev : float
        Muon selection threshold [GeV].

    Returns
    -------
    reach_km : float
        Fitted growth of the light reach per e-fold [km].
    residual_dex : float
        Root-mean-square residual of the best scanned point [dex].
    """
    published, cos_range = published_effective_area_cm2(site, LOG10_E, data_dir)
    theta_deg, weights = zenith_grid(N_ZENITH_FIT, cos_range)
    band = (LOG10_E >= FIT_BAND[0]) & (LOG10_E <= FIT_BAND[1])
    reach_km, residual, _ = fit_light_reach(
        site, np.cos(np.deg2rad(theta_deg)), weights, published[band], threshold_gev,
        log10_e=LOG10_E[band],
    )
    return reach_km, float(np.min(residual))


def sky_averaged(site, threshold_gev: float, reach_km: float | None) -> np.ndarray:
    """Effective area averaged over the whole sky [cm^2].

    Parameters
    ----------
    site : softpaws.detectors.Site
        The detector.
    threshold_gev : float
        Muon selection threshold [GeV].
    reach_km : float or None
        Fitted light reach per e-fold [km]. ``None`` keeps the footprint.

    Returns
    -------
    aeff : np.ndarray
        Effective area on :data:`LOG10_E` [cm^2].
    """
    theta_deg, weights = zenith_grid(N_ZENITH)
    cos_theta = np.cos(np.deg2rad(theta_deg))
    per_direction = directional_effective_area_cm2(
        site, cos_theta, threshold_gev, reach_km=reach_km, log10_e=LOG10_E
    )
    return per_direction @ weights


def report_fits(fits: dict[str, tuple[float, float]]) -> None:
    """Print the fitted instrument number of each site."""
    print("\nOne instrument number per detector, fitted to its own published table:\n")
    print(f"{'detector':>10} {'footprint':>10} {'R(1 PeV)':>10} "
          f"{'Lambda':>18} {'residual':>10}")
    for site in SITES:
        if site.name not in fits:
            continue
        reach_km, residual = fits[site.name]
        radius_m = site.radius_km * 1.0e3
        grown_m = radius_m + reach_km * 1.0e3 * np.log(ANCHOR_GEV / REACH_PIVOT_GEV)
        print(f"{site.name:>10} {radius_m:9.0f} m {grown_m:9.0f} m "
              f"{reach_km * 1.0e3:+9.1f} m/e-fold {residual:9.3f} dex")
    print("\n  The reach grows per e-fold about a pivot far above the band, so a")
    print("  positive value is a turn-on: inside the band the array responds to less")
    print("  than the footprint it owns and grows toward it. Three sites want that.")
    print("  ARCA230 wants a small halo instead, and it is the one site of the four")
    print("  whose published curve one number does not follow to better than 0.05 dex.")


def report_sensitivity(curves: dict[str, dict[str, np.ndarray]], livetime_s: float,
                       gamma: float, emin_gev: float | None) -> None:
    """Print the three injections, each on both responses."""
    anchor = int(np.argmin(np.abs(LOG10_E - np.log10(ANCHOR_GEV))))
    print(f"\nA diffuse signal seen in {livetime_s / YEAR_S:.0f} years, per flavour, "
          "with no background:\n")

    blocks = (
        (f"(a) a single event: E^2 phi giving one event per decade at "
         f"{ANCHOR_GEV:.0e} GeV [GeV cm^-2 s^-1 sr^-1]",
         lambda aeff: single_event_sensitivity(
             aeff, livetime_s, LOG10_E, solid_angle_sr=SKY_SR)[anchor]),
        (f"(b) a line at {ANCHOR_GEV:.0e} GeV: E Phi giving {N_EVENTS_LIMIT} events "
         "[GeV cm^-2 s^-1 sr^-1]",
         lambda aeff: ANCHOR_GEV * line_sensitivity(
             aeff, livetime_s, solid_angle_sr=SKY_SR)[anchor]),
        (f"(c) an E^-{gamma:.1f} power law: E^2 phi at {PIVOT_ENERGY_GEV:.0e} GeV giving "
         f"{N_EVENTS_LIMIT} events [GeV cm^-2 s^-1 sr^-1]",
         lambda aeff: float(power_law_sensitivity(
             aeff, livetime_s, gamma, LOG10_E, emin_gev, solid_angle_sr=SKY_SR))),
    )
    for title, quantity in blocks:
        print(f"  {title}")
        print(f"{'detector':>12} {'geometry':>12} {'fitted':>12} {'fitted/geometry':>17}")
        for name, pair in curves.items():
            geometry = quantity(pair["geometry"])
            fitted = quantity(pair["fitted"])
            print(f"{name:>12} {geometry:12.3g} {fitted:12.3g} {fitted / geometry:17.2f}")
        print()

    ratio = N_EVENTS_LIMIT * np.log(10.0)
    print(f"  Blocks (a) and (b) are the same statement: a line limit is {ratio:.1f} times")
    print("  the one-event flux, the events assumed times the decade the count is")
    print("  spread over. Only the power law asks anything about a spectrum.\n")

    measured = float(
        power_law_flux(PIVOT_ENERGY_GEV, ICECUBE_TRACKS_2022.phi0, ICECUBE_TRACKS_2022.gamma)[0]
        * PIVOT_ENERGY_GEV**2
    )
    best = min(
        float(power_law_sensitivity(pair["fitted"], livetime_s, gamma, LOG10_E, emin_gev,
                                    solid_angle_sr=SKY_SR))
        for pair in curves.values()
    )
    print(f"  The measured astrophysical flux is {measured:.2g} GeV cm^-2 s^-1 sr^-1 at "
          f"{PIVOT_ENERGY_GEV:.0e} GeV, which")
    print(f"  is {measured / best:.0f} times the best ceiling in block (c). A diffuse search "
          "is background")
    print("  limited and not signal limited wherever the atmospheric flux of tutorial 10")
    print("  survives, so block (c) is the floor a zero-background search would reach")
    print("  rather than a projected limit.\n")

    print(f"  Where the E^-{gamma:.1f} count comes from, and what a raised window costs:\n")
    window = 10.0**RAISED_WINDOW_LOG10_GEV
    print(f"{'detector':>12} {'90% [log10 E]':>16} {'ceiling':>12} "
          f"{f'above 10^{RAISED_WINDOW_LOG10_GEV:.0f}':>14} {'cost':>8}")
    for name, pair in curves.items():
        lo, hi = central_energy_range(pair["fitted"], gamma, emin_gev=10.0 ** LOG10_E[0],
                                      log10_e=LOG10_E)
        whole = float(power_law_sensitivity(pair["fitted"], livetime_s, gamma, LOG10_E,
                                            emin_gev, solid_angle_sr=SKY_SR))
        raised = float(power_law_sensitivity(pair["fitted"], livetime_s, gamma, LOG10_E,
                                             window, solid_angle_sr=SKY_SR))
        print(f"{name:>12} {f'{lo:.2f} to {hi:.2f}':>16} {whole:12.3g} {raised:14.3g} "
              f"{raised / whole:7.1f}x")
    print("\n  Every band starts at the bottom of the grid, so an E^-2 ceiling is set by")
    print("  where the analysis starts and not by the reach of the detector. Throwing")
    print(f"  away everything below 10^{RAISED_WINDOW_LOG10_GEV:.0f} GeV, which is what buys "
          "the zero background, costs")
    print("  the factor in the last column.")


def make_figures(curves: dict[str, dict[str, np.ndarray]], livetime_s: float,
                 out_stem: pathlib.Path) -> None:
    """Draw the differential sensitivity and its dependence on the spectrum."""
    energy = 10.0**LOG10_E
    with plt.style.context(str(_STYLE)):
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))

        for name, pair in curves.items():
            band = [single_event_sensitivity(pair[key], livetime_s, LOG10_E,
                                             solid_angle_sr=SKY_SR)
                    for key in ("geometry", "fitted")]
            line, = axes[0].plot(LOG10_E, band[1], lw=1.3, label=name)
            axes[0].fill_between(LOG10_E, band[0], band[1], color=line.get_color(),
                                 alpha=0.3, lw=0)
        measured = power_law_flux(
            energy, ICECUBE_TRACKS_2022.phi0, ICECUBE_TRACKS_2022.gamma
        ) * energy**2
        axes[0].plot(LOG10_E, measured, color="0.4", lw=1.6, ls="--")
        axes[0].text(6.3, 3.0e-8, "measured flux", color="0.4", ha="center", va="center",
                     fontsize=7)
        axes[0].set_yscale("log")
        axes[0].set_xlabel(r"$\log_{10}(E_\nu/\mathrm{GeV})$")
        axes[0].set_ylabel(r"$E^2\phi$ [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
        axes[0].set_title("one event per decade")
        axes[0].set_ylim(1.0e-13, 2.0e-7)
        axes[0].legend(frameon=False, fontsize=7, loc="lower right", ncol=2)

        for name, pair in curves.items():
            band = [np.array([power_law_sensitivity(pair[key], livetime_s, g, LOG10_E,
                                                    solid_angle_sr=SKY_SR)
                              for g in GAMMA_GRID])
                    for key in ("geometry", "fitted")]
            line, = axes[1].plot(GAMMA_GRID, band[1], lw=1.3, label=name)
            axes[1].fill_between(GAMMA_GRID, band[0], band[1], color=line.get_color(),
                                 alpha=0.3, lw=0)
        axes[1].set_yscale("log")
        axes[1].set_xlabel(r"$\gamma$")
        axes[1].set_ylabel(r"$E^2\phi$ at 100 TeV [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
        axes[1].set_title(f"{livetime_s / YEAR_S:.0f} years, power law")

        fig.tight_layout()
        for suffix in (".pdf", ".png"):
            fig.savefig(out_stem.with_suffix(suffix))
            print(f"Figure saved to: {out_stem.with_suffix(suffix)}")
        plt.close(fig)


def main() -> None:
    """Fit each site, turn the response into three sensitivities, and draw them."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    livetime_s = args.livetime_yr * YEAR_S

    fits, curves = {}, {}
    for site in SITES:
        print(f"Fitting {site.name} over {REACH_FRACTIONS.size} scanned radii "
              f"at {REACH_REFERENCE_GEV:.0e} GeV ...")
        try:
            reach_km, residual = fit_site(site, args.data_dir, args.threshold_gev)
        except FileNotFoundError:
            print(f"({site.name}: the DR2 release is not on disk, so this site is skipped.)")
            continue
        fits[site.name] = (reach_km, residual)
        curves[site.name] = {
            "geometry": sky_averaged(site, args.threshold_gev, None),
            "fitted": sky_averaged(site, args.threshold_gev, reach_km),
        }

    report_fits(fits)
    report_sensitivity(curves, livetime_s, args.gamma, args.emin_gev)
    make_figures(curves, livetime_s, args.out_dir / "11_diffuse_signal")


if __name__ == "__main__":
    main()
