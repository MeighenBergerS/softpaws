"""Example 52 -- a global through-going-track forecast on the flavour triangle.

Example 51 measured what IC86 through-going tracks add to IceCube's MESE
flavour result (little) and forecast IceCube-Gen2 the same way (little
more, because a through-going muon's reconstructed energy is nearly flat in
log E below ``E_nu`` and the tau wall leaks into the bulk). This example
extends that forecast to every current and planned track detector this
project models -- KM3NeT/ARCA230, Baikal-GVD, P-ONE, TRIDENT and
IceCube-Gen2 -- and adds them all to MESE at once.

Each detector runs through example 51's machinery unchanged: example 46's
directional effective areas on the ten declination bands of IceCube's
smearing table (a band's Earth column depends only on the local zenith, so
the Pole's ``zenith = 90 deg + dec`` maps every site onto the same bands),
example 51's decay-shifted fold, the IceCube-tracks anchors on the index
and the ``nu_mu`` flux, a ten-year Asimov at the oscillation-averaged 1:1:1
composition, and the profile in ``r = f_tau / (f_mu + f_tau)``. IC86 enters
as measured. The profiles add as independent samples, and the sum goes onto
MESE through example 51's reconstructed likelihood surface.

Idealizations, stated: IC86's smearing stands in for every detector's, the
South Pole atmospheric table is used at every site, the water sites take
example 50's trigger-times-selection layer, and Baikal-GVD takes the sea
water optics with the lake's ~22 m absorption length in place of 47 m.
Baikal-GVD's small count also reflects example 35's footprint convention:
fourteen 60 m-radius cluster cylinders, ~0.08 km^3 of instrumented water,
with muons crossing between clusters not counted.

Usage
-----
    python scripts/2026_muon_transport/52_global_tracks_forecast.py
    python scripts/2026_muon_transport/52_global_tracks_forecast.py --rebuild-responses
"""

import argparse
import importlib.util
import pathlib
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"
_DEFAULT_DATA_DIR = _HERE.parents[1] / "src" / "softpaws" / "data" / "dataverse_files"

#: Per-site responses on the smearing bands, cached one site at a time.
_RESPONSE_CACHE = _DEFAULT_OUT_DIR / "52_global_responses.npz"

#: Lake Baikal's absorption length [m], in place of sea water's 47 m.
BAIKAL_ATTENUATION_M = 22.0

#: Forecast detectors and their colours; IC86 is measured and drawn black.
FORECAST_COLOR = {
    "KM3NeT/ARCA230": "#1b9e77",
    "Baikal-GVD": "#66a61e",
    "P-ONE": "#d95f02",
    "TRIDENT": "#e6ab02",
    "IceCube-Gen2": "#7570b3",
}
GLOBAL_COLOR = "#1b9e77"


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
    parser.add_argument("--rebuild-responses", action="store_true",
                        help="Rebuild the cached per-site responses.")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the figures, '52a' and '52b'.")
    return parser.parse_args()


def roster(ex35, ex45, ex50):
    """Forecast detectors: example 35 geometry, example 45 optics, selection.

    Returns
    -------
    detectors : list of tuple
        ``(name, site35, site45, efficiency)`` per forecast detector.
    """
    sites = {s.name: s for s in ex35.build_sites()}
    ice = replace(ex45.ICECUBE_SITE, attenuation_override_m=42.0)
    water = replace(ex45.ARCA_SITE, attenuation_override_m=47.0)
    baikal = replace(ex45.ARCA_SITE, name="Baikal-GVD",
                     attenuation_override_m=BAIKAL_ATTENUATION_M)
    gen2 = replace(sites["IceCube"], name="IceCube-Gen2",
                   radius_km=ex50.GEN2_RADIUS_KM, height_km=ex50.GEN2_HEIGHT_KM)
    return [
        ("KM3NeT/ARCA230", sites["ARCA230"], replace(water, name="KM3NeT/ARCA230"),
         ex50.EFFICIENCY_WATER),
        ("Baikal-GVD", sites["Baikal-GVD"], baikal, ex50.EFFICIENCY_WATER),
        ("P-ONE", sites["P-ONE"], replace(water, name="P-ONE"), ex50.EFFICIENCY_WATER),
        ("TRIDENT", sites["TRIDENT"], replace(water, name="TRIDENT"),
         ex50.EFFICIENCY_WATER),
        ("IceCube-Gen2", gen2, replace(ice, name="IceCube-Gen2"), ex50.EFFICIENCY_ICE),
    ]


def site_responses(ex35, ex45, ex46, ex51, detectors, dec_edges, rebuild: bool):
    """Channel responses per detector on the smearing bands [cm^2], cached.

    The bands are IceCube declination bands; each maps to the local zenith
    through ``cos(zenith) = -sin(dec)``, which fixes the Earth column and
    hence the response at every site.

    Returns
    -------
    responses : dict of str -> dict
        ``responses[name]["mu" | "tau"]`` on (``ex51.LOG10_E_GRID``, bands).
    """
    cache = dict(np.load(_RESPONSE_CACHE)) if _RESPONSE_CACHE.exists() and not rebuild else {}
    if cache and not np.array_equal(cache.get("dec_edges"), dec_edges):
        cache = {}
    sin_centers = 0.5 * (np.sin(np.deg2rad(dec_edges[:-1])) + np.sin(np.deg2rad(dec_edges[1:])))
    cos_theta = -sin_centers
    coarse = ex35.COMMON_LOG10_E
    out = {}
    for name, site35, site45, efficiency in detectors:
        keys = (f"{name}_mu", f"{name}_tau")
        if all(k in cache for k in keys):
            print(f"  cached {name}")
            out[name] = {"mu": cache[keys[0]], "tau": cache[keys[1]]}
            continue
        out[name] = {}
        for channel in ("mu", "tau"):
            print(f"  building {name} {channel} ...", flush=True)
            per_species = [
                ex46.directional_aeff_cm2(ex35, ex45, site35, site45, cos_theta, 8.0,
                                          (channel,), xsec, None, 1.0, efficiency)
                for xsec in ex45.SPECIES
            ]
            banded = np.mean(per_species, axis=0)
            fine = np.empty((ex51.LOG10_E_GRID.size, banded.shape[1]))
            with np.errstate(divide="ignore"):
                log_a = np.log10(np.clip(banded, 1.0e-30, None))
            for j in range(banded.shape[1]):
                fine[:, j] = 10.0 ** np.interp(ex51.LOG10_E_GRID, coarse, log_a[:, j],
                                               left=-30.0, right=log_a[-1, j])
            out[name][channel] = fine
            cache[f"{name}_{channel}"] = fine
        cache["dec_edges"] = dec_edges
        _RESPONSE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(_RESPONSE_CACHE, **cache)
    return out


def asimov_profile(ex51, enu_edges, marginal, responses, atmos, dec_edges, years):
    """Anchored reco-space profile of a ten-year Asimov at 1:1:1.

    Returns
    -------
    curve : np.ndarray
        ``2 Delta ln L`` on ``ex51.R_GRID``.
    astro : float
        Astrophysical tracks in the fit window at the truth.
    """
    zeros = np.zeros((ex51.RECO_EDGES.size - 1, dec_edges.size - 1))
    like = ex51.RecoLikelihood(enu_edges, marginal, responses, atmos, dec_edges,
                               years * 365.25 * 86400.0, zeros, anchors=ex51.ANCHORS)
    truth = (0.5, ex51.ANCHORS["phi_mu"][0], ex51.TRACKS_GAMMA[0], 1.0, 1.0)
    asimov = like.expectation(*truth)
    background = like.expectation(0.5, 0.0, ex51.TRACKS_GAMMA[0], 1.0, 1.0)
    curve, _ = like.profile(data=asimov)
    return curve, float((asimov - background).sum())


def _save(fig, out_dir: pathlib.Path, stem: str) -> None:
    """Write one figure to ``out_dir`` as both PDF and PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {path.resolve()}")
    plt.close(fig)


def figure_profiles(ex51, curves, curve_ic86, curve_global, out_dir) -> None:
    """Anchored ratio profiles: IC86 measured, each forecast, and the sum."""
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.0, 3.0))
        ax.axvspan(ex51.R_MIN_STD, ex51.R_MAX, color="0.93", zorder=0,
                   label="std. osc. band")
        for level, note in ((1.0, r"$68\%$"), (3.84, r"$95\%$")):
            ax.axhline(level, color="0.75", lw=0.7, ls=":")
            ax.text(1.005, level, note, fontsize=7, color="0.45", va="center")
        ax.plot(ex51.R_GRID, curve_ic86, color="k", lw=1.2, label="IC86 (measured)")
        for name, curve in curves.items():
            ax.plot(ex51.R_GRID, curve, color=FORECAST_COLOR[name], lw=1.0,
                    label=f"{name} (10 yr)")
        ax.plot(ex51.R_GRID, curve_global, color=GLOBAL_COLOR, lw=1.6, ls="--",
                label="all tracks")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 8.0)
        ax.set_xlabel(r"$f_\tau \,/\, (f_\mu + f_\tau)$ at Earth")
        ax.set_ylabel(r"$2\,\Delta\ln L$ (anchored)")
        ax.legend(fontsize=5.5, frameon=False, loc="upper left")
        _save(fig, out_dir, "52a_global_tracks_profiles")


def figure_triangle(ex51, curve_ic86, curve_global, out_dir) -> tuple[float, ...]:
    """MESE alone, MESE + IC86 (measured), MESE + all tracks (forecast).

    Returns
    -------
    best : tuple of float
        Global best-fit fractions ``(f_e, f_mu, f_tau)``.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    ex50 = ex51._EX50
    x, y, total_ic = ex51.combined_surface(curve_ic86)
    _, _, total_all = ex51.combined_surface(curve_global)
    i, j = np.unravel_index(int(np.nanargmin(total_all)), total_all.shape)
    best = tuple(float(v) for v in ex51._fractions_from_xy(x[i, j], y[i, j]))
    pink = "#e7298a"
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(4.4, 4.2))
        ex50.draw_triangle_frame(ax)
        ax.contourf(x, y, total_all, levels=[0.0, ex51.LEVEL68_2D, ex51.LEVEL95_2D],
                    colors=[GLOBAL_COLOR, GLOBAL_COLOR], alpha=0.18, zorder=2)
        for surface, color, z in ((total_ic, pink, 4), (total_all, GLOBAL_COLOR, 5)):
            ax.contour(x, y, surface, levels=[ex51.LEVEL68_2D], colors=color,
                       linewidths=1.3, zorder=z)
            ax.contour(x, y, surface, levels=[ex51.LEVEL95_2D], colors=color,
                       linestyles="--", linewidths=1.0, zorder=z)
        ax.plot(x[i, j], y[i, j], marker="*", color=GLOBAL_COLOR, ms=8, ls="none",
                zorder=6, label="global best fit")
        ex50.draw_published_curves(ax, skip=("icecube2022",))
        handles, labels = ax.get_legend_handles_labels()
        ours = [Line2D([], [], color=pink, lw=1.3),
                Line2D([], [], color=pink, ls="--", lw=1.0),
                Patch(facecolor=GLOBAL_COLOR, alpha=0.3, edgecolor=GLOBAL_COLOR, lw=1.3),
                Line2D([], [], color=GLOBAL_COLOR, ls="--", lw=1.0), handles[0]]
        ours_labels = [r"MESE + IC86 tracks $68\%$", r"MESE + IC86 tracks $95\%$",
                       r"MESE + all tracks $68\%$ (forecast)",
                       r"MESE + all tracks $95\%$ (forecast)", labels[0]]
        first = ax.legend(ours, ours_labels, fontsize=8, frameon=False,
                          loc="upper left", bbox_to_anchor=(0.0, 1.16),
                          handlelength=1.4, labelspacing=0.3)
        ax.add_artist(first)
        ax.legend(handles[1:], labels[1:], fontsize=8, frameon=False,
                  loc="upper right", bbox_to_anchor=(1.08, 1.16),
                  handlelength=1.6, labelspacing=0.3)
        _save(fig, out_dir, "52b_global_tracks_triangle")
    return best


def main() -> None:
    args = parse_args()
    print("Loading examples 35, 45, 46, 50 and 51 ...")
    ex51 = load_example("51_dr2_flavor_fit.py", "_example_51")
    ex50 = ex51._EX50
    ex35 = load_example("35_point_source_effective_area.py", "_example_35")
    ex45 = load_example("45_first_principles_reach.py", "_example_45")
    ex46 = load_example("46_declination_resolved_reach.py", "_example_46")

    enu_edges, dec_edges, marginal, responses_ic = ex51.fit_inputs(
        ex35, ex45, ex46, args.data_dir, False)
    atmos = ex51.atmospheric_fluxes(dec_edges)

    print("IC86, measured ...")
    from softpaws.data.loader import compute_livetime_s, load_uptime
    livetime_s = sum(
        compute_livetime_s(load_uptime(args.data_dir / "uptime" / f"{s}_exp.csv"))
        for s in ex51.IC86_SEASONS)
    data_counts = ex51.binned_events(args.data_dir, dec_edges)
    ic86 = ex51.RecoLikelihood(enu_edges, marginal, responses_ic, atmos, dec_edges,
                               livetime_s, data_counts, anchors=ex51.ANCHORS)
    curve_ic86, _ = ic86.profile()

    detectors = roster(ex35, ex45, ex50)
    print("Per-site responses on the smearing bands ...")
    responses = site_responses(ex35, ex45, ex46, ex51, detectors, dec_edges,
                               args.rebuild_responses)

    print(f"\n  {'detector':>16} {'astro tracks':>12} {'2dLL(r=0)':>10} {'Wilks 68%':>14}")
    curves = {}
    for name, _, _, _ in detectors:
        curve, astro = asimov_profile(ex51, enu_edges, marginal, responses[name],
                                      atmos, dec_edges, ex50.FORECAST_YEARS)
        curves[name] = curve
        lo, hi = ex51.interval(curve, 1.0)
        print(f"  {name:>16} {astro:12.0f} {curve[0]:10.2f}   [{lo:.2f}, {hi:.2f}]")
    curve_global = curve_ic86 + sum(curves.values())
    curve_global -= curve_global.min()
    lo, hi = ex51.interval(curve_global, 1.0)
    print(f"  {'all tracks':>16} {'':>12} {curve_global[0]:10.2f}   [{lo:.2f}, {hi:.2f}]"
          f"   (IC86 measured: 2dLL(r=0) {curve_ic86[0]:.2f})")

    print()
    figure_profiles(ex51, curves, curve_ic86, curve_global, args.out_dir)
    best = figure_triangle(ex51, curve_ic86, curve_global, args.out_dir)
    print(f"global MESE + all tracks best fit: f_e {best[0]:.2f}, f_mu {best[1]:.2f}, "
          f"f_tau {best[2]:.2f}")


if __name__ == "__main__":
    main()
