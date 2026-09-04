"""Example 49 -- the tau wall, and where its events sit in the DR2 sample.

Tau charged current is not terminal in the Earth: the tau decays back to a
``nu_tau`` before losing much energy, so the cascade crosses columns that
absorb every ``nu_mu``. For an equal-flavour flux the through-going track rate
therefore hands over from ``nu_mu`` to ``nu_tau -> tau -> mu`` along a sharp
diagonal in the zenith-energy plane -- the **tau wall** -- running from the
nadir at 10^6 GeV to ``cos(theta) ~ -0.35`` at 10^8 GeV. Figure (a) maps the
tau fraction of the track rate and draws the wall under both transmission
conventions, which is the construction's honest uncertainty: the full
regeneration ladder credits a down-scattered neutrino to its original energy,
pure survival removes it, and the published deep bands sit between the two.
The wall only moves toward the horizon under survival, so the tau fraction
quoted from the ladder is the conservative one.

Figure (b) prices the prediction in the real sample: the DR2 uptime-summed
exposure of 13.6 years, the MCEq atmospheric flux (conventional and prompt,
example 22's cache), and published astrophysical power laws. Above 100 TeV the
tau channel is ~13% of the astrophysical track rate; above 1 PeV at
``sin(dec) > 0.5`` the atmospheric expectation is 0.3 events while the tau
channel carries ~28% of the astrophysical one. Prompt matters nowhere: it is
1% of the atmospheric rate overall and the wall region is
atmospheric-background-free either way.

The astrophysical flux enters three ways -- the northern-tracks, cascade and
combined-fit single power laws -- and the tau *share* barely moves between
them, because it is a property of the response and not of the spectrum.

Usage
-----
    python examples/49_tau_wall.py
    python examples/49_tau_wall.py --first-principles
"""

import argparse
import importlib.util
import pathlib
from dataclasses import replace
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from softpaws.data.loader import compute_livetime_s, load_uptime
from softpaws.data.schema import SEASONS
from softpaws.fluxes import (
    ICECUBE_CASCADES_2020,
    ICECUBE_COMBINED_2023,
    ICECUBE_TRACKS_2022,
    AtmosphericFlux,
    load_mceq_table,
)
from softpaws.transport.attenuation import regenerated_transmission

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"
_DEFAULT_DATA_DIR = _HERE.parents[1] / "src" / "softpaws" / "data" / "dataverse_files"

#: Atmospheric ``nu_mu + nubar`` flux tabulated by example 22 (MCEq,
#: SIBYLL-2.3d on H3a over a South Pole atmosphere), conventional and prompt,
#: on an energy-by-declination grid. Example 22 must have been run once.
_MCEQ_CACHE = _HERE / "output" / "22_mceq_atmospheric_flux.npz"

#: Published astrophysical single power laws: per-flavour ``nu + nubar``
#: normalization at 100 TeV [1e-18 GeV^-1 cm^-2 s^-1 sr^-1] and spectral
#: index. Northern tracks (arXiv:2111.10299), cascades (arXiv:2001.09520) and
#: the combined fit (arXiv:2308.00191).
ASTRO_FLUXES = {
    "tracks": (ICECUBE_TRACKS_2022.phi0, ICECUBE_TRACKS_2022.gamma),
    "cascades": (ICECUBE_CASCADES_2020.phi0, ICECUBE_CASCADES_2020.gamma),
    "combined": (ICECUBE_COMBINED_2023.phi0, ICECUBE_COMBINED_2023.gamma),
}

#: Upgoing arrival directions of the wall map; ``cos(theta) = -1`` is the
#: nadir and ``-sin(dec)`` at the Pole.
MAP_COS_THETA = np.linspace(-0.999, -0.02, 25)

#: Component colours of the counts figure: greys for the atmosphere, the
#: shared purple for the astrophysical ``nu_mu`` and pink for the tau channel.
COMPONENT_COLOR = {
    "atm. conventional": "0.65",
    "atm. prompt": "0.4",
    r"astro $\nu_\mu$": "#7570b3",
    r"astro $\nu_\tau \to \mu$": "#e7298a",
}


def load_example(stem: str, name: str):
    """Import a sibling example module by file stem."""
    spec = importlib.util.spec_from_file_location(name, _HERE / stem)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=pathlib.Path, default=_DEFAULT_DATA_DIR,
                        help="Root of the DR2 release.")
    parser.add_argument("--first-principles", action="store_true",
                        help="Drop example 45's fitted attenuation override and "
                             "normalization and keep the derived optics.")
    parser.add_argument("--e-min-gev", type=float, default=1.0e5,
                        help="Energy above which the counts figure integrates.")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the two figures, '49a' and '49b'.")
    return parser.parse_args()


def directional(ex35, ex45, ex46, site, site45, cos_theta, flavours, efficiency,
                n_levels: int | None = None) -> np.ndarray:
    """Species-averaged directional effective area [cm^2].

    Parameters
    ----------
    ex35, ex45, ex46 : ModuleType
        Examples 35, 45 and 46.
    site : ex35.Site
        Detector geometry and medium.
    site45 : ex45.Site
        Detector optics.
    cos_theta : np.ndarray
        Arrival directions; ``+1`` is overhead.
    flavours : tuple of str
        Parent channels to sum.
    efficiency : float
        Flat normalization applied to the model.
    n_levels : int or None, optional
        Rungs of the ``nu_mu`` regeneration ladder. ``None`` keeps the shipped
        ladder; ``1`` is the pure-survival convention.

    Returns
    -------
    aeff : np.ndarray, shape (n_energy, n_dir)
        Effective area [cm^2] on ``ex35.COMMON_LOG10_E``.
    """
    shipped = ex46.regenerated_transmission
    if n_levels is not None:
        ex46.regenerated_transmission = partial(regenerated_transmission,
                                                n_levels=n_levels)
    try:
        per_species = [
            ex46.directional_aeff_cm2(ex35, ex45, site, site45, cos_theta, 8.0,
                                      flavours, xsec, None, 1.0, efficiency)
            for xsec in ex45.SPECIES
        ]
    finally:
        ex46.regenerated_transmission = shipped
    return np.mean(per_species, axis=0)


def atmospheric_flux(component: str, energy: np.ndarray,
                     dec_deg: np.ndarray) -> np.ndarray:
    """Surface atmospheric flux on an energy-by-band grid [GeV^-1 cm^-2 s^-1 sr^-1].

    Log-log interpolation of example 22's MCEq table in both energy and
    declination.

    Parameters
    ----------
    component : {"conv", "prompt"}
        Atmospheric component.
    energy : np.ndarray
        Neutrino energies [GeV].
    dec_deg : np.ndarray
        Band-centre declinations [deg].

    Returns
    -------
    flux : np.ndarray, shape (energy.size, dec_deg.size)
        ``nu_mu + nubar`` flux at the surface.

    Raises
    ------
    FileNotFoundError
        Raised if example 22's cache is missing.
    """
    if not _MCEQ_CACHE.exists():
        raise FileNotFoundError(
            f"{_MCEQ_CACHE} not found; run example 22 once to tabulate the "
            "atmospheric flux.")
    flux = AtmosphericFlux(load_mceq_table(_MCEQ_CACHE))
    return flux.component_on_grid(component, energy, dec_deg)


def band_counts(aeff, flux, energy, widths, livetime_s, e_min_gev=0.0):
    """Expected events per declination band over the exposure.

    Parameters
    ----------
    aeff : np.ndarray, shape (n_energy, n_band)
        Banded effective area [cm^2].
    flux : np.ndarray, shape (n_energy, n_band)
        Flux per band [GeV^-1 cm^-2 s^-1 sr^-1].
    energy : np.ndarray
        Neutrino energies [GeV].
    widths : np.ndarray
        Band widths in ``sin(dec)``.
    livetime_s : float
        Exposure [s].
    e_min_gev : float, optional
        Lower integration limit [GeV].

    Returns
    -------
    counts : np.ndarray, shape (n_band,)
        Expected events per band.
    """
    mask = energy >= e_min_gev
    per_band = np.trapezoid(aeff[mask] * flux[mask], energy[mask], axis=0)
    return livetime_s * 2.0 * np.pi * widths * per_band


def _save(fig, out_dir: pathlib.Path, stem: str) -> None:
    """Write one figure to ``out_dir`` as both PDF and PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {path.resolve()}")
    plt.close(fig)


def figure_wall(log10_e, fraction, fraction_survival, out_dir) -> None:
    """The tau fraction of the track rate, with the wall under both conventions."""
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.2, 3.0))
        mesh = ax.pcolormesh(log10_e, MAP_COS_THETA, fraction.T, cmap="viridis",
                             vmin=0.0, vmax=1.0, shading="nearest")
        fig.colorbar(mesh, ax=ax, label=r"$\nu_\tau$ fraction of track rate")
        for grid, style in ((fraction, "-"), (fraction_survival, "--")):
            ax.contour(log10_e, MAP_COS_THETA, grid.T, levels=[0.5],
                       colors="white", linewidths=1.2, linestyles=style)
        ax.legend(
            handles=[plt.Line2D([], [], color="white", ls=s, lw=1.2, label=n)
                     for n, s in (("ladder", "-"), ("survival", "--"))],
            loc="upper left", fontsize=8, frameon=False, handlelength=2.0,
            labelcolor="white", title=r"wall ($f_\tau = 0.5$)",
            title_fontsize=8, borderaxespad=0.8,
        )
        legend = ax.get_legend()
        legend.get_title().set_color("white")
        ax.set_xlabel(r"$\log_{10}(E_\nu\,/\,\mathrm{GeV})$")
        ax.set_ylabel(r"$\cos\theta$")
        _save(fig, out_dir, "49a_tau_wall_map")


def figure_counts(centers, widths, up, rows, e_min_gev, out_dir) -> None:
    """Expected DR2 events per unit ``sin(dec)``, by component."""
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.0, 3.0))
        for (name, row), color in zip(rows.items(), COMPONENT_COLOR.values()):
            ax.step(centers[up], row[up] / widths[up], where="mid", color=color,
                    lw=1.2, label=name)
        ax.set_yscale("log")
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel(r"$\sin\delta$")
        exponent = int(np.round(np.log10(e_min_gev)))
        ax.set_ylabel(r"events / $\Delta\sin\delta$, "
                      rf"$E_\nu > 10^{{{exponent}}}$ GeV")
        ax.legend(fontsize=8, frameon=False, loc="lower left")
        _save(fig, out_dir, "49b_dr2_band_counts")


def report(energy, widths, up, centers, a_mu, a_tau, conv, prompt, livetime_s) -> None:
    """Print the component counts for the three astrophysical fluxes."""
    yr = livetime_s / (365.25 * 86400.0)
    for e_min, label in ((0.0, "all E"), (1.0e5, "E > 100 TeV"), (1.0e6, "E > 1 PeV")):
        print(f"\n  [{label}] upgoing totals over {yr:.1f} yr")
        n_conv = band_counts(a_mu, conv, energy, widths, livetime_s, e_min)[up].sum()
        n_pr = band_counts(a_mu, prompt, energy, widths, livetime_s, e_min)[up].sum()
        print(f"  atmospheric: conventional {n_conv:11,.1f}   prompt {n_pr:8,.1f}")
        for name, (phi0, gamma) in ASTRO_FLUXES.items():
            flux = phi0 * 1.0e-18 * (energy[:, None] / 1.0e5) ** (-gamma)
            flux = flux * np.ones((1, centers.size))
            n_mu = band_counts(a_mu, flux, energy, widths, livetime_s, e_min)[up].sum()
            n_tau = band_counts(a_tau, flux, energy, widths, livetime_s, e_min)[up].sum()
            print(f"  astro {name:8s}: nu_mu {n_mu:9,.1f}   tau->mu {n_tau:8,.2f}   "
                  f"tau share {n_tau / (n_mu + n_tau):6.1%}")

    print("\n  wall region (sin dec > 0.5, E > 1 PeV):")
    wall = up & (centers > 0.5)
    n_atm = (band_counts(a_mu, conv, energy, widths, livetime_s, 1.0e6)
             + band_counts(a_mu, prompt, energy, widths, livetime_s, 1.0e6))[wall].sum()
    for name, (phi0, gamma) in ASTRO_FLUXES.items():
        flux = phi0 * 1.0e-18 * (energy[:, None] / 1.0e5) ** (-gamma)
        flux = flux * np.ones((1, centers.size))
        n_mu = band_counts(a_mu, flux, energy, widths, livetime_s, 1.0e6)[wall].sum()
        n_tau = band_counts(a_tau, flux, energy, widths, livetime_s, 1.0e6)[wall].sum()
        print(f"  {name:8s}: astro nu_mu {n_mu:6.2f}   tau->mu {n_tau:5.2f} "
              f"({n_tau / (n_mu + n_tau):5.1%})   atmospheric {n_atm:5.2f}")


def main() -> None:
    args = parse_args()
    print("Loading examples 35, 45 and 46 ...")
    ex35 = load_example("35_point_source_effective_area.py", "_example_35")
    ex45 = load_example("45_first_principles_reach.py", "_example_45")
    ex46 = load_example("46_declination_resolved_reach.py", "_example_46")

    site = ex35.build_sites()[0]
    site45 = ex45.ICECUBE_SITE
    efficiency = 1.0
    if not args.first_principles:
        site45 = replace(site45, attenuation_override_m=42.0)
        efficiency = 0.755
        print("  fitted configuration: Lambda 42 m, normalization 0.755")

    log10_e = ex35.COMMON_LOG10_E
    energy = 10.0**log10_e

    print("Building the wall map: nu_mu channel ...")
    map_mu = directional(ex35, ex45, ex46, site, site45, MAP_COS_THETA, ("mu",),
                         efficiency)
    print("Building the wall map: tau channel ...")
    map_tau = directional(ex35, ex45, ex46, site, site45, MAP_COS_THETA, ("tau",),
                          efficiency)
    print("Building the wall map: nu_mu channel, survival convention ...")
    map_mu_survival = directional(ex35, ex45, ex46, site, site45, MAP_COS_THETA,
                                  ("mu",), efficiency, n_levels=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.where(map_mu + map_tau > 0.0,
                            map_tau / (map_mu + map_tau), np.nan)
        fraction_survival = np.where(
            map_mu_survival + map_tau > 0.0,
            map_tau / (map_mu_survival + map_tau), np.nan)

    print("Building the banded areas for the counts ...")
    sin_dec_edges, _ = ex35.icecube_banded(args.data_dir)
    centers = 0.5 * (sin_dec_edges[:-1] + sin_dec_edges[1:])
    widths = np.diff(sin_dec_edges)
    up = centers > 0.0
    a_mu = ex46.model_banded(ex35, ex45, site, site45, sin_dec_edges, 8.0, ("mu",),
                             efficiency=efficiency)
    a_tau = ex46.model_banded(ex35, ex45, site, site45, sin_dec_edges, 8.0, ("tau",),
                              efficiency=efficiency)

    livetime_s = sum(
        compute_livetime_s(load_uptime(args.data_dir / "uptime" / f"{s}_exp.csv"))
        for s in SEASONS)
    print(f"  DR2 uptime-summed livetime: {livetime_s / (365.25 * 86400.0):.2f} yr")

    dec_deg = np.rad2deg(np.arcsin(np.clip(centers, -1.0, 1.0)))
    conv = atmospheric_flux("conv", energy, dec_deg)
    prompt = atmospheric_flux("prompt", energy, dec_deg)
    report(energy, widths, up, centers, a_mu, a_tau, conv, prompt, livetime_s)

    phi0, gamma = ASTRO_FLUXES["combined"]
    astro = phi0 * 1.0e-18 * (energy[:, None] / 1.0e5) ** (-gamma)
    astro = astro * np.ones((1, centers.size))
    rows = {
        "atm. conventional": band_counts(a_mu, conv, energy, widths, livetime_s,
                                         args.e_min_gev),
        "atm. prompt": band_counts(a_mu, prompt, energy, widths, livetime_s,
                                   args.e_min_gev),
        r"astro $\nu_\mu$": band_counts(a_mu, astro, energy, widths, livetime_s,
                                        args.e_min_gev),
        r"astro $\nu_\tau \to \mu$": band_counts(a_tau, astro, energy, widths,
                                                 livetime_s, args.e_min_gev),
    }
    print()
    figure_wall(log10_e, fraction, fraction_survival, args.out_dir)
    figure_counts(centers, widths, up, rows, args.e_min_gev, args.out_dir)


if __name__ == "__main__":
    main()
