"""Tutorial 12 -- injecting a new point source.

Tutorial 11 spread the injected signal over the whole sky. A source is one
direction instead, and that changes the answer twice. The solid angle is gone,
so the flux a search can reach is smaller. And the Earth is in the way for
part of the day, by an amount that depends on the declination of the source
and on where the detector sits, so the same source is a different measurement
at each site.

The three injections of tutorial 11 come along unchanged: a single event, a
line at one energy, and a power law. So does the pair of responses, the
geometry alone and the one instrument number fitted at each site against its
own published table.

What is new is the background. Tutorial 11 excluded 2.44 events whatever the
exposure, which is the Feldman-Cousins limit on nothing seen over nothing
expected. A point source is a direction, so what a search has to look through
is one bin of sky rather than the whole of it, and the atmospheric neutrinos
inside that bin are countable: MCEq gives the flux, the same response gives
the effective area, and the angular resolution gives the bin. The limit that
comes out is no longer background free, and against IceCube's own published
sensitivity it lands on the other side: a counting experiment in a fixed bin
is worse than the unbinned likelihood a real search uses, so the published
curve falls between the two, which is where it has to be.

Where the search is allowed to start is not a detail. An atmospheric spectrum
falls like ``E^-3.7`` and the signal here like ``E^-2``, so counting the whole
band charges a search most where its effective area is largest, and the
declination dependence the response has comes out flattened. Letting the
window move removes that, and the published shape is the evidence it should
be removed: across the upgoing sky that shape is the ceiling's, so whatever
the background costs a real search, it costs it almost equally at every
declination.

The bin is measured too, wherever there is a release to measure it from. The
DR2 smearing table carries a point-spread angle with every row, so the bin can
be the containment radius at each energy and declination rather than a chosen
constant, and for a Gaussian point spread that radius is the optimum for a
counting search to within 5%. It matters because it is not a constant: the
containment closes from about 1.4 degrees at 10 TeV to 0.4 at 10 PeV, and it
opens again toward the pole where the Earth has taken the high-energy end
away. Both run the way the published shape needs, and a fixed bin instead
leaves a tilt of 1.3 across the upgoing sky that is not in the data.

Usage
-----
    python examples/12_point_source_signal.py
    python examples/12_point_source_signal.py --bin-radius-deg 0.5
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.data import icecube_point_source_sensitivity, load_psf_table
from softpaws.detectors import ARCA230, ICECUBE, PONE, TRIDENT
from softpaws.fluxes import AtmosphericFlux, load_mceq_table
from softpaws.response.declination import (
    REACH_FRACTIONS,
    REACH_REFERENCE_GEV,
    directional_effective_area_cm2,
    fit_light_reach,
    zenith_band_weights,
)
from softpaws.response.sensitivity import (
    DEFAULT_BIN_RADIUS_DEG,
    N_EVENTS_LIMIT,
    OPTIMAL_CONTAINMENT,
    PIVOT_ENERGY_GEV,
    atmospheric_background_counts,
    atmospheric_background_density,
    line_sensitivity,
    optimized_window_sensitivity,
    power_law_sensitivity,
    psf_bin_radius_deg,
    sensitivity_upper_limit,
    single_event_sensitivity,
)
from softpaws.response.site_models import published_effective_area_cm2
from softpaws.transport.earth import zenith_grid

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"

#: Neutrino energies every curve here is built on [log10 GeV].
LOG10_E = np.arange(4.0, 8.01, 0.25)

#: Band the instrument number is fitted over [log10 GeV].
FIT_BAND = (5.0, 7.5)

#: Directions the fit averages over, and zenith bands the daily sweep of a
#: source is histogrammed into.
N_ZENITH_FIT = 8
N_COS_THETA = 60

#: Declinations the ceiling is drawn against, uniform in ``sin(dec)``.
SIN_DEC = np.linspace(-0.95, 0.95, 21)

#: Declinations the tables are printed at [deg].
TABLE_DEC_DEG = (-60.0, -30.0, 0.0, 30.0, 60.0)

#: Sites compared, in the order they are drawn.
SITES = (ICECUBE, ARCA230, PONE, TRIDENT)

#: Energy the single event and the line are quoted at [GeV].
ANCHOR_GEV = 1.0e6

#: Spectral index of the power-law injection.
GAMMA = 2.0

YEAR_S = 365.25 * 86400.0

#: Exposure of the published curve the comparison is drawn against [yr].
PUBLISHED_LIVETIME_YR = 14.0

#: Cached MCEq table, the one tutorial 10 builds.
MCEQ_TABLE = _DEFAULT_OUT_DIR / "22_mceq_atmospheric_flux.npz"

#: Cached point-spread containment, taken from the DR2 smearing table once.
PSF_TABLE = _DEFAULT_OUT_DIR / "12_psf_containment.npz"


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threshold-gev", type=float, default=1.0e3,
                        help="Muon selection threshold.")
    parser.add_argument("--livetime-yr", type=float, default=PUBLISHED_LIVETIME_YR,
                        help="Exposure of the search. Defaults to the exposure of the "
                             "published curve the figure is drawn against.")
    parser.add_argument("--gamma", type=float, default=GAMMA,
                        help="Spectral index of the power-law injection.")
    parser.add_argument("--dec-deg", type=float, default=0.0,
                        help="Declination the injections are tabulated at.")
    parser.add_argument("--data-dir", type=pathlib.Path, default=None,
                        help="Root of the DR2 release, for IceCube's published table.")
    parser.add_argument("--mceq-table", type=pathlib.Path, default=MCEQ_TABLE,
                        help="Cached MCEq atmospheric table; tutorial 10 builds it.")
    parser.add_argument("--bin-radius-deg", type=float, default=DEFAULT_BIN_RADIUS_DEG,
                        help="Angular radius of the bin the background is counted in.")
    parser.add_argument("--psf-table", type=pathlib.Path, default=PSF_TABLE,
                        help="Cached DR2 point-spread containment; built on first use.")
    parser.add_argument("--fixed-bin", action="store_true",
                        help="Use --bin-radius-deg everywhere instead of the measured PSF.")
    parser.add_argument("--channels", choices=("both", "mu"), default="both",
                        help="'mu' drops the tau channel, the convention a published "
                             "per-flavour nu_mu sensitivity is quoted in.")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the figures.")
    return parser.parse_args()


def load_background(path: pathlib.Path):
    """The MCEq background, or ``None`` if the table has not been built yet.

    Building the table runs MCEq once per declination and takes minutes, so
    tutorial 10 caches it and this one reads the cache. Without it the whole
    tutorial still runs, background free, which is what tutorial 11 does.

    Parameters
    ----------
    path : pathlib.Path
        Cached ``.npz`` table.

    Returns
    -------
    flux : softpaws.fluxes.AtmosphericFlux or None
        The interpolated table, or ``None`` if it is not on disk.
    """
    if not path.exists():
        print(f"({path} is not on disk, so the limits below stay background free.\n"
              " Run examples/10_atmospheric_background.py once to build it.)")
        return None
    return AtmosphericFlux(load_mceq_table(path))


def load_psf(path: pathlib.Path, data_dir: pathlib.Path | None, fixed: bool) -> dict | None:
    """IceCube's point-spread containment, or ``None`` to keep a fixed bin."""
    if fixed:
        return None
    try:
        table = load_psf_table(path, data_dir, quantile=OPTIMAL_CONTAINMENT)
    except FileNotFoundError:
        print(f"({path} is not on disk and the DR2 release is not either, so the\n"
              " background is counted in a bin of fixed radius.)")
        return None
    print(f"Point-spread containment at {100 * float(table['quantile']):.0f}%, "
          f"{table['containment_deg'].shape[0]} energies x "
          f"{table['containment_deg'].shape[1]} declinations.")
    return table


def fit_site(site, data_dir: pathlib.Path | None, threshold_gev: float,
             channels: str = "both") -> tuple[float, float]:
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
    channels : {"both", "mu"}, optional
        Flavours the response is built from. The published table is ``nu_mu``
        simulation, so the fit and the prediction must agree on this.

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
        log10_e=LOG10_E[band], channels=channels,
    )
    return reach_km, float(np.min(residual))


def swept_response(site, threshold_gev: float, reach_km: float | None,
                   dec_deg: np.ndarray,
                   channels: str = "both") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Response of each arrival direction, and the daily sweep of each declination.

    A source at a fixed declination sweeps a fixed set of zeniths as the Earth
    turns, so the response it sees is the directional effective area contracted
    with the time the source spends in each zenith band. At the Pole that
    contraction is a single band and nothing sweeps at all.

    The contraction is left to the caller because the signal and the background
    do it differently. A signal flux is the same from every direction, so the
    effective area can be averaged first. An atmospheric flux is not, so it has
    to be multiplied by the response direction by direction and averaged after.

    Parameters
    ----------
    site : softpaws.detectors.Site
        The detector.
    threshold_gev : float
        Muon selection threshold [GeV].
    reach_km : float or None
        Fitted light reach per e-fold [km]. ``None`` keeps the footprint.
    dec_deg : np.ndarray, shape (n_dec,)
        Source declinations [deg].
    channels : {"both", "mu"}, optional
        Flavours the response is built from.

    Returns
    -------
    per_direction : np.ndarray, shape (LOG10_E.size, N_COS_THETA)
        Effective area on :data:`LOG10_E` in each zenith band [cm^2].
    cos_theta : np.ndarray, shape (N_COS_THETA,)
        Cosine of the arrival zenith of each band.
    weights : np.ndarray, shape (n_dec, N_COS_THETA)
        Fraction of a sidereal day each declination spends in each band.
    """
    cos_theta, weights = zenith_band_weights(site.latitude_deg, dec_deg,
                                             n_cos_theta=N_COS_THETA)
    per_direction = directional_effective_area_cm2(
        site, cos_theta, threshold_gev, reach_km=reach_km, log10_e=LOG10_E,
        channels=channels,
    )
    return per_direction, cos_theta, weights


def background_limit(flux, sweep, aeff_cm2: np.ndarray, livetime_s: float, gamma: float,
                     bin_radius_deg: float | np.ndarray) -> tuple[np.ndarray, ...]:
    """Atmospheric events in the source bin, and the limit they leave.

    The whole band is counted once, to say how much background there is, and
    then the window is allowed to move, because where a search can start is
    what decides how much of that background it has to accept. See
    :func:`~softpaws.response.sensitivity.optimized_window_sensitivity`.

    Parameters
    ----------
    flux : softpaws.fluxes.AtmosphericFlux or None
        The MCEq background. ``None`` keeps the search background free.
    sweep : tuple
        As returned by :func:`swept_response`.
    aeff_cm2 : np.ndarray, shape (LOG10_E.size, n_dec)
        The same response contracted over the sweep [cm^2], on the same
        declinations the sweep was built for.
    livetime_s : float
        Exposure [s].
    gamma : float
        Spectral index of the injected flux.
    bin_radius_deg : float or np.ndarray
        Angular radius of the bin [deg], from :func:`bin_radius_grid`.

    Returns
    -------
    counts : np.ndarray, shape (n_dec,)
        Atmospheric events in the bin over the whole band.
    limit : np.ndarray or None, shape (n_dec,)
        Best ``E^2 phi`` a search excludes [GeV cm^-2 s^-1], or ``None``
        without a flux.
    emin_gev : np.ndarray or None, shape (n_dec,)
        Energy the best window starts at [GeV], or ``None`` without a flux.
    density : np.ndarray or None, shape (LOG10_E.size, n_dec)
        Atmospheric events per decade, or ``None`` without a flux.
    """
    per_direction, cos_theta, weights = sweep
    if flux is None:
        return np.zeros(weights.shape[0]), None, None, None
    args = (flux, per_direction, cos_theta, weights, livetime_s, LOG10_E, bin_radius_deg)
    counts = atmospheric_background_counts(*args)
    density = atmospheric_background_density(*args)
    limit, emin_gev = optimized_window_sensitivity(
        aeff_cm2, density, livetime_s, gamma, LOG10_E
    )
    return counts, limit, emin_gev, density


def report_fits(fits: dict[str, tuple[float, float]]) -> None:
    """Print the fitted instrument number of each site."""
    print("\nOne instrument number per detector, fitted to its own published table:\n")
    print(f"{'detector':>10} {'latitude':>10} {'Lambda':>18} {'residual':>10}")
    for site in SITES:
        if site.name not in fits:
            continue
        reach_km, residual = fits[site.name]
        print(f"{site.name:>10} {site.latitude_deg:9.1f}d {reach_km * 1.0e3:+9.1f} m/e-fold "
              f"{residual:9.3f} dex")


def report_source(curves: dict[str, dict[str, np.ndarray]], livetime_s: float,
                  gamma: float, dec_deg: float) -> None:
    """Print the three injections at one declination, each on both responses."""
    anchor = int(np.argmin(np.abs(LOG10_E - np.log10(ANCHOR_GEV))))
    print(f"\nA point source at declination {dec_deg:+.0f} degrees, seen in "
          f"{livetime_s / YEAR_S:.0f} years, with no background:\n")

    blocks = (
        (f"(a) a single event: E^2 phi giving one event per decade at "
         f"{ANCHOR_GEV:.0e} GeV [GeV cm^-2 s^-1]",
         lambda aeff: single_event_sensitivity(aeff, livetime_s, LOG10_E)[anchor]),
        (f"(b) a line at {ANCHOR_GEV:.0e} GeV: E Phi giving {N_EVENTS_LIMIT} events "
         "[GeV cm^-2 s^-1]",
         lambda aeff: ANCHOR_GEV * line_sensitivity(aeff, livetime_s)[anchor]),
        (f"(c) an E^-{gamma:.1f} power law: E^2 phi at {PIVOT_ENERGY_GEV:.0e} GeV giving "
         f"{N_EVENTS_LIMIT} events [GeV cm^-2 s^-1]",
         lambda aeff: float(power_law_sensitivity(aeff, livetime_s, gamma, LOG10_E))),
    )
    for title, quantity in blocks:
        print(f"  {title}")
        print(f"{'detector':>12} {'geometry':>12} {'fitted':>12} {'fitted/geometry':>17}")
        for name, pair in curves.items():
            geometry = quantity(pair["geometry"])
            fitted = quantity(pair["fitted"])
            print(f"{name:>12} {geometry:12.3g} {fitted:12.3g} {fitted / geometry:17.2f}")
        print()


def report_background(backgrounds: dict[str, tuple], ceilings: dict[str, dict],
                      radii: dict[str, np.ndarray | float], dec_deg: np.ndarray,
                      bin_radius_deg: float, have_flux: bool) -> None:
    """Print the atmospheric background, the window it forces, and its price."""
    if not have_flux:
        return
    print("\n  Atmospheric neutrinos in the source bin over the whole band, where the\n"
          "  search then starts, the bin it uses there, and what the three cost it:\n")
    print(f"{'detector':>12} {'':>8}" + "".join(f"{f'{d:+.0f}d':>12}" for d in TABLE_DEC_DEG))
    for name, (counts, emin_gev) in backgrounds.items():
        cost = ceilings[name]["background"] / ceilings[name]["fitted"]
        radius = radii[name]
        # The bin is quoted where the search actually starts, which is the
        # only energy on the grid that decides anything.
        if np.ndim(radius) == 0:
            at_window = np.full(dec_deg.size, float(radius))
        else:
            at_window = np.array([np.interp(np.log10(e), LOG10_E, radius[:, j])
                                  for j, e in enumerate(emin_gev)])
        for label, row, form in (("events", counts, "{:12.1f}"),
                                 ("E_min", np.log10(emin_gev), "{:12.2f}"),
                                 ("bin/deg", at_window, "{:12.2f}"),
                                 ("cost", cost, "{:11.2f}x")):
            values = "".join(form.format(np.interp(d, dec_deg, row)) for d in TABLE_DEC_DEG)
            print(f"{name if label == 'events' else '':>12} {label:>8}" + values)
    print(f"\n  With no background a search excludes {N_EVENTS_LIMIT} events whatever the")
    print("  exposure, and the cost row would be 1 everywhere. With one it excludes the")
    print("  average Feldman-Cousins upper limit over background-only outcomes, and the")
    print("  cost is what that does to the flux. E_min is log10 of the energy the best")
    print("  window starts at, and it is the whole story: an atmospheric spectrum falls")
    print("  faster than an astrophysical one, so a search whose response reaches high")
    print("  energy cuts the background away and pays almost nothing, while one the")
    print("  Earth has capped has nowhere to cut to and pays in full.")
    print("\n  The downgoing event counts are large because the release gives those")
    print("  directions almost no pointing below 10^4 GeV, tens of degrees rather")
    print("  than one, and a bin that wide holds most of the sky. Those are the")
    print("  events the window is cutting away, which is why the cost stays modest.")
    print("\n  Two things are missing from every downgoing entry anyway. Atmospheric")
    print("  muons are not modelled and outnumber these neutrinos by orders of")
    print("  magnitude wherever they reach the detector, and the MCEq table shipped")
    print("  here is the South Pole atmosphere, so only IceCube reads it at its own")
    print("  site, as only IceCube has a released point spread to bin with.")
    print()


def report_published(ceilings: dict[str, dict], dec_deg: np.ndarray,
                     livetime_s: float) -> None:
    """Put both IceCube limits next to IceCube's own published sensitivity."""
    band = ceilings.get("IceCube")
    if band is None:
        return
    published_sin, published_flux = icecube_point_source_sensitivity()
    years = livetime_s / YEAR_S
    print("  IceCube's own 14-year track sensitivity, against the two limits this "
          "model\n  brackets it with [GeV cm^-2 s^-1]:\n")
    if abs(years - PUBLISHED_LIVETIME_YR) > 0.1:
        print(f"  (The model runs at {years:.0f} years and the published curve at "
              f"{PUBLISHED_LIVETIME_YR:.0f}, so this is not\n   like for like. Pass "
              f"--livetime-yr {PUBLISHED_LIVETIME_YR:.0f} for the comparison that is.)\n")
    print(f"{'':>14}" + "".join(f"{f'{d:+.0f}d':>12}" for d in TABLE_DEC_DEG))
    at_table = np.sin(np.deg2rad(np.asarray(TABLE_DEC_DEG, dtype=float)))
    published = np.interp(at_table, published_sin, published_flux)
    ceiling = np.array([np.interp(d, dec_deg, band["fitted"]) for d in TABLE_DEC_DEG])
    limited = np.array([np.interp(d, dec_deg, band["background"]) for d in TABLE_DEC_DEG])
    for label, values in (("published", published), ("ceiling", ceiling),
                          ("+ background", limited)):
        print(f"{label:>14}" + "".join(f"{v:12.3g}" for v in values))
    print(f"{'over ceiling':>14}" + "".join(f"{v:11.2f}x" for v in published / ceiling))
    print(f"{'under limit':>14}" + "".join(f"{v:11.2f}x" for v in limited / published))
    print("\n  The ceiling counts no background at all and the limit under it counts")
    print("  every atmospheric neutrino in the bin, with no weighting by how much")
    print("  each one looks like the signal. A real search sits between the two, and")
    print("  IceCube's does across the upgoing sky, by the two factors above. Neither")
    print("  bound is fitted to it. Both fail in the downgoing sky, where atmospheric")
    print("  muons are the background and nothing here models them.")
    print()


def report_declination(ceilings: dict[str, np.ndarray], dec_deg: np.ndarray,
                       gamma: float) -> None:
    """Print the ceiling against declination, and where each site is best."""
    print(f"  The same E^-{gamma:.1f} ceiling against declination, fitted response "
          "[GeV cm^-2 s^-1]:\n")
    print(f"{'detector':>12}" + "".join(f"{f'{d:+.0f}d':>12}" for d in TABLE_DEC_DEG)
          + f"{'best':>8}{'spread':>9}")
    for name, ceiling in ceilings.items():
        row = f"{name:>12}"
        for wanted in TABLE_DEC_DEG:
            row += f"{np.interp(wanted, dec_deg, ceiling):12.3g}"
        best = int(np.argmin(ceiling))
        spread = float(np.max(ceiling) / np.min(ceiling))
        print(row + f"{f'{dec_deg[best]:+.0f}d':>8}{spread:8.2f}x")
    print("\n  IceCube sits at the Pole, where a declination is one zenith all day, so")
    print("  the ceiling is a clean function of it: the horizon is best and the")
    print("  downgoing sky, where the muon has to be born inside the overburden, is")
    print("  worst. Every other site turns under the sky, and the sweep averages the")
    print("  Earth away. The spread closes as the site approaches the equator, which")
    print("  is geometry and not instrument: TRIDENT sees every declination alike.")


def make_figures(ceilings: dict[str, np.ndarray], curves: dict[str, dict[str, np.ndarray]],
                 livetime_s: float, gamma: float, dec_deg: float,
                 out_stem: pathlib.Path) -> None:
    """Draw the ceiling against declination and the differential sensitivity."""
    with plt.style.context(str(_STYLE)):
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))

        # One line per detector, and it is the whole statement: the fitted
        # response, the atmospheric background in the measured bin, and the
        # window each declination is best searched in. No band, because there
        # is nothing left here that is a choice.
        for name, band in ceilings.items():
            axes[0].plot(SIN_DEC, band.get("background", band["fitted"]), lw=1.4)
        published_sin, published_flux = icecube_point_source_sensitivity()
        axes[0].plot(published_sin, published_flux, color="0.4", lw=1.6, ls="--")
        axes[0].text(-0.1, 3.0e-9, "IceCube, published", color="0.4", ha="center",
                     va="center", fontsize=7)
        axes[0].set_yscale("log")
        axes[0].set_xlabel(r"$\sin(\mathrm{dec})$")
        axes[0].set_ylabel(r"$E^2\phi$ at 100 TeV [GeV cm$^{-2}$ s$^{-1}$]")
        axes[0].set_title(f"{livetime_s / YEAR_S:.0f} years, $\\gamma = {gamma:.0f}$")
        axes[0].set_ylim(1.0e-11, 8.0e-9)

        for name, pair in curves.items():
            axes[1].plot(
                LOG10_E,
                single_event_sensitivity(pair["fitted"], livetime_s, LOG10_E,
                                         n_events=pair.get("n_events", N_EVENTS_LIMIT)),
                lw=1.4, label=name,
            )
        axes[1].set_yscale("log")
        axes[1].set_xlabel(r"$\log_{10}(E_\nu/\mathrm{GeV})$")
        axes[1].set_ylabel(r"$E^2\phi$ [GeV cm$^{-2}$ s$^{-1}$]")
        axes[1].set_title(f"per decade, dec ${dec_deg:+.0f}^\\circ$")
        axes[1].legend(frameon=False, fontsize=7, loc="upper left")

        fig.tight_layout()
        for suffix in (".pdf", ".png"):
            fig.savefig(out_stem.with_suffix(suffix))
            print(f"Figure saved to: {out_stem.with_suffix(suffix)}")
        plt.close(fig)


def main() -> None:
    """Fit each site, sweep the sky, and turn the response into three sensitivities."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    livetime_s = args.livetime_yr * YEAR_S
    dec_deg = np.rad2deg(np.arcsin(SIN_DEC))

    flux = load_background(args.mceq_table)
    psf = load_psf(args.psf_table, args.data_dir, args.fixed_bin)
    fits, ceilings, curves, backgrounds, radii = {}, {}, {}, {}, {}
    for site in SITES:
        print(f"Fitting {site.name} over {REACH_FRACTIONS.size} scanned radii "
              f"at {REACH_REFERENCE_GEV:.0e} GeV ...")
        try:
            reach_km, residual = fit_site(site, args.data_dir, args.threshold_gev,
                                          args.channels)
        except FileNotFoundError:
            print(f"({site.name}: the DR2 release is not on disk, so this site is skipped.)")
            continue
        fits[site.name] = (reach_km, residual)

        # The sweep contracts one set of directions with one set of weights per
        # declination, so the source costs nothing beyond the grid it rides on.
        print(f"Sweeping {site.name} over {SIN_DEC.size} declinations ...")
        sweeps = {
            key: swept_response(site, args.threshold_gev,
                                None if key == "geometry" else reach_km,
                                np.append(dec_deg, args.dec_deg), args.channels)
            for key in ("geometry", "fitted")
        }
        swept = {key: sweep[0] @ sweep[2].T for key, sweep in sweeps.items()}
        ceilings[site.name] = {
            key: power_law_sensitivity(aeff[:, :-1], livetime_s, args.gamma, LOG10_E)
            for key, aeff in swept.items()
        }
        curves[site.name] = {key: aeff[:, -1] for key, aeff in swept.items()}

        # The background rides the fitted response, the one that carries the
        # detector's own turn-on, and the same sweep the signal does.
        radius = psf_bin_radius_deg(psf, sweeps["fitted"][1], LOG10_E,
                                    args.bin_radius_deg)
        counts, limit, emin_gev, density = background_limit(
            flux, sweeps["fitted"], swept["fitted"], livetime_s, args.gamma, radius,
        )
        # For the report the bin is quoted per declination, which is the
        # sweep-weighted mean of the directions that declination visits.
        weights = sweeps["fitted"][2]
        radii[site.name] = radius if np.ndim(radius) == 0 else radius @ weights.T
        # The sweep carries one extra declination, the one the tables are
        # printed at, so the curves drop it the way the ceilings do.
        if limit is not None:
            backgrounds[site.name] = (counts[:-1], emin_gev[:-1])
            ceilings[site.name]["background"] = limit[:-1]
            # The differential curve is the same statement bin by bin: each
            # decade carries its own background and excludes its own flux.
            curves[site.name]["n_events"] = sensitivity_upper_limit(density[:, -1])

    report_fits(fits)
    report_source(curves, livetime_s, args.gamma, args.dec_deg)
    report_background(backgrounds, ceilings, radii, dec_deg, args.bin_radius_deg,
                      flux is not None)
    report_published(ceilings, dec_deg, livetime_s)
    report_declination({name: band["fitted"] for name, band in ceilings.items()},
                       dec_deg, args.gamma)
    make_figures(ceilings, curves, livetime_s, args.gamma, args.dec_deg,
                 args.out_dir / "12_point_source_signal")


if __name__ == "__main__":
    main()
