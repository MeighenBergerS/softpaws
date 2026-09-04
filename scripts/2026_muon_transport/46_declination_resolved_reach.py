"""Example 46 -- the derived reach, band by band across the sky.

Example 45 builds the light reach from the medium's optics and the module's
area, with nothing fitted, and reproduces the published DR2 table to 0.025 dex
of shape. What it leaves behind is a residual in the *energy* dependence: our
transmission falls too slowly near 10^5 GeV and too fast above 10^7. Example 45
also shows that residual is not the regeneration ladder -- the transmission
log-slope is identical to two decimals across a threefold change in the
neutral-current inelasticity and an eightfold change in the number of rungs,
because the direct unscattered term is ``exp(-X n sigma_tot)`` and carries no
ladder dependence at all.

That leaves the declination axis, which the sky average throws away.

Why the average can hide it
---------------------------
Absorption varies by three decades across the upgoing sky, from 1.1e7 g/cm^2 at
the horizon to 1.1e10 at the nadir. A sky-averaged transmission is therefore a
sum over wildly different optical depths, and its energy dependence is set by
which bands still transmit at each energy. Two quite different things produce
the same averaged residual:

**A universal error.** If our absorption or our cross section is wrong, every
band is wrong the same way and the residual's energy tilt is the same in all of
them.

**A declination-dependent selection.** If instead DR2's efficiency varies across
the sky -- plausible, since the horizon carries the atmospheric-muon background
and needs tighter cuts than the nadir -- then each band is internally consistent
and only the average is tilted.

These are distinguishable, and the published table is binned finely enough to do
it. The diagnostic is the **tilt**: the slope of ``log10(published / model)``
against ``log10 E`` within one band. A tilt common to every band is ours; a tilt
that tracks the PREM column is not.

What this example changes against example 35
--------------------------------------------
Example 35 carries in a two-parameter reach fitted to the sky-averaged table.
Here the reach is example 45's, derived from Frank-Tamm yield, the medium's
attenuation length and the module's effective area, and applied to every band
without refitting. Three further corrections from example 45 come with it: the
``nu_mu`` channel alone, since the DR2 readme states the table is built from
simulated muon neutrino events; the neutrino/antineutrino average, since the
companion paper states the area assumes equal numbers of each; and the reach as
the *single* suppression mechanism, since letting it also truncate the range
suppresses the same physics twice and drives the model below the published
curve.

Usage
-----
    python examples/46_declination_resolved_reach.py
    python examples/46_declination_resolved_reach.py --min-modules 4
    python examples/46_declination_resolved_reach.py --with-tau
"""

import argparse
import importlib.util
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.response.declination import (
    band_statistics as _band_statistics,
)
from softpaws.response.declination import (
    column_target_volume_km3,
    derived_band_averaged_effective_area_cm2,
    derived_directional_effective_area_cm2,
)

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"
_DEFAULT_DATA_DIR = _HERE.parents[1] / "src" / "softpaws" / "data" / "dataverse_files"

#: Band the residuals are scored over, matching examples 35 and 45.
STATS_LOG10_E = (5.0, 7.8)

#: Energies the level is tabulated and drawn at.
SHOW_LOG10_E = (5.0, 6.0, 7.0)

#: Rungs of the regeneration ladder, as in example 35.
N_RUNG = 32
RUNG_DECADES = 4.0

BAND_COLOR = ("#7570b3", "#1b9e77", "#e7298a")


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
    parser.add_argument("--min-modules", type=float, default=None,
                        help="Modules that must register a coincident hit. "
                             "Defaults to example 45's value.")
    parser.add_argument("--first-principles", action="store_true",
                        help="Drop example 45's fitted attenuation override and "
                             "normalization and keep the derived optics.")
    parser.add_argument("--with-tau", action="store_true",
                        help="Add nu_tau -> tau -> mu. Wrong against a nu_mu table.")
    parser.add_argument("--scan-inelasticity", action="store_true",
                        help="Scan the mean neutral-current inelasticity band by band. "
                             "Only the deep-column bands respond, which is why the "
                             "sky-averaged scan of example 45 saw nothing.")
    parser.add_argument("--out", type=pathlib.Path,
                        default=_DEFAULT_OUT_DIR / "46_declination_resolved_reach",
                        help="Output stem; '_level' and '_tilt' figures are written.")
    return parser.parse_args()


def directional_aeff_cm2(
    ex35, ex45, site, site45, cos_theta: np.ndarray, min_modules: float,
    flavours: tuple[str, ...], cross_section, inelasticity_nc: float | None = None,
    halo_weight: float = 1.0, efficiency: float = 1.0,
) -> np.ndarray:
    """Effective area per arrival direction, with the derived reach [cm^2].

    See
    :func:`softpaws.response.declination.derived_directional_effective_area_cm2`.
    The ``ex35`` and ``ex45`` arguments are kept for the scripts that load this
    one; only ``ex35.COMMON_LOG10_E`` is still read from them.
    """
    return derived_directional_effective_area_cm2(
        site, site45, cos_theta, min_modules, flavours, cross_section,
        inelasticity_nc, halo_weight, efficiency, ex35.COMMON_LOG10_E,
    )


def column_volume_km3(ex45, site, site45, production_gev, cos_theta, available_km,
                      min_modules, halo_weight, n_sides, n_energy: int = 40):
    """Column target volume for one production energy and every direction [km^3].

    See :func:`softpaws.response.declination.column_target_volume_km3`. ``ex45``
    is kept for the scripts that load this one and is no longer read.
    """
    return column_target_volume_km3(
        site, site45, production_gev, cos_theta, available_km, min_modules,
        halo_weight, n_sides, n_energy,
    )


def model_banded(ex35, ex45, site, site45, sin_dec_edges, min_modules, flavours,
                 inelasticity_nc=None, halo_weight=1.0, efficiency=1.0):
    """Model effective area per published band, averaged over both species [cm^2].

    See
    :func:`softpaws.response.declination.derived_band_averaged_effective_area_cm2`.
    """
    return derived_band_averaged_effective_area_cm2(
        site, site45, sin_dec_edges, min_modules, flavours, ex45.SPECIES,
        inelasticity_nc, halo_weight, efficiency, ex35.COMMON_LOG10_E,
        ex35.N_SUB_BAND,
    )


def band_statistics(log10_e, published, model, band):
    """Level, tilt and scatter per band; see :func:`band_statistics`."""
    return _band_statistics(log10_e, published, model, band)


def report(sin_dec_centers, columns_g_cm2, level, tilt, scatter) -> None:
    """Print the level and tilt against declination, upgoing sky only."""
    up = sin_dec_centers > 0.0
    print("\n  Upgoing bands: level and energy tilt of published / model")
    print(f"  {'sin(dec)':>9} {'dec[deg]':>9} {'PREM X':>10} {'level':>7} "
          f"{'tilt':>8} {'scatter':>8}")
    for j in np.flatnonzero(up)[::5]:
        print(f"  {sin_dec_centers[j]:9.2f} "
              f"{np.rad2deg(np.arcsin(sin_dec_centers[j])):9.1f} "
              f"{columns_g_cm2[j]:10.2e} {level[j]:7.3f} "
              f"{tilt[j]:+8.3f} {scatter[j]:8.4f}")
    print("  (tilt is dex per decade of E_nu; zero means the energy shape is right)")

    good = up & np.isfinite(tilt)
    t, lv = tilt[good], level[good]
    print(f"\n  tilt  : mean {t.mean():+.3f}, band-to-band spread {t.std():.3f} dex/decade,"
          f" range {t.min():+.3f} to {t.max():+.3f}")
    print(f"  level : mean {10**np.mean(np.log10(lv)):.3f}, "
          f"spread {np.std(np.log10(lv)):.3f} dex")

    # A tilt common to every band belongs to the model; one that tracks the
    # column belongs to the sky. The correlation separates them.
    x = np.log10(np.clip(columns_g_cm2[good], 1.0, None))
    print(f"\n  correlation of tilt with log10(PREM column): "
          f"{np.corrcoef(x, t)[0, 1]:+.2f}")
    print(f"  correlation of level with log10(PREM column): "
          f"{np.corrcoef(x, np.log10(lv))[0, 1]:+.2f}")
    if abs(np.corrcoef(x, t)[0, 1]) < 0.5 and t.std() < 0.5 * abs(t.mean()):
        print("  -> the tilt is common to every band, so it is ours and not the sky.")
    else:
        print("  -> the tilt varies with the column, so it is declination dependent.")


def scan_inelasticity(ex35, ex45, site, site45, sin_dec_edges, min_modules, flavours,
                      sin_dec_centers, columns, log10_e, published, band) -> None:
    """Print the band-by-band tilt against the mean neutral-current inelasticity.

    The horizon bands do not respond at all and the deep ones respond strongly,
    which is the whole reason a sky-averaged version of this scan reports a null.
    Values above roughly 0.35 are outside the physics and are shown to establish
    that even an indefensible mean leaves the deepest bands tilted.
    """
    up = sin_dec_centers > 0.0
    log_column = np.log10(np.clip(columns, 1.0, None))
    marks = (0.02, 0.42, 0.82, 0.98)
    print("\n  Mean neutral-current inelasticity, band by band")
    print("  " + " ".join(f"sin(dec)={m:.2f}" for m in marks)
          + "     spread  corr(col)  level")
    for y in (0.15, 0.25, 0.35, 0.45, 0.60):
        model = model_banded(ex35, ex45, site, site45, sin_dec_edges, min_modules,
                             flavours, y)
        level, tilt, _ = band_statistics(log10_e, published, model, band)
        good = up & np.isfinite(tilt)
        row = " ".join(f"{tilt[int(np.argmin(np.abs(sin_dec_centers - m)))]:+13.3f}"
                       for m in marks)
        print(f"  <y> = {y:.2f} {row}  {tilt[good].std():7.3f}  "
              f"{np.corrcoef(log_column[good], tilt[good])[0, 1]:+8.2f}  "
              f"{10**np.mean(np.log10(level[good])):6.3f}")
    print("  (the horizon bands do not move at all; only the deep column responds)")


def make_figures(sin_dec_centers, log10_e, published, model, level, tilt, out) -> None:
    """Level against declination, then the energy tilt against declination."""
    up = sin_dec_centers > 0.0
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.2, 3.0))
        for target, color in zip(SHOW_LOG10_E, BAND_COLOR):
            i = int(np.argmin(np.abs(log10_e - target)))
            ax.plot(sin_dec_centers[up], (published[i] / model[i])[up], color=color,
                    lw=1.3, label=rf"$10^{{{target:.0f}}}$ GeV")
        ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.6)
        ax.set_xlabel(r"$\sin\delta$")
        ax.set_ylabel(r"published $/$ model")
        ax.legend(fontsize=6, frameon=False, loc="lower left")
        _save(fig, out.with_name(out.name + "_level"))

        fig, ax = plt.subplots(figsize=(3.2, 3.0))
        ax.axhline(0.0, color="0.6", lw=0.8, ls=":")
        ax.plot(sin_dec_centers[up], tilt[up], color="#1b9e77", lw=1.4)
        mean = np.nanmean(tilt[up])
        ax.axhline(mean, color="#7570b3", lw=1.0, ls="--")
        ax.text(0.03, mean, f"  mean {mean:+.3f}", color="#7570b3", fontsize=6,
                va="bottom", ha="left")
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel(r"$\sin\delta$")
        ax.set_ylabel("tilt of published / model [dex per decade]")
        _save(fig, out.with_name(out.name + "_tilt"))


def _save(fig, out_path: pathlib.Path) -> None:
    """Write both the vector and the raster copy of one figure."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_path.with_suffix(suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {path.resolve()}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    print("Loading examples 35 and 45 ...")
    ex35 = load_example("35_point_source_effective_area.py", "_example_35")
    ex45 = load_example("45_first_principles_reach.py", "_example_45")

    min_modules = (ex45.DEFAULT_MIN_MODULES if args.min_modules is None
                   else args.min_modules)
    flavours = ("mu", "tau") if args.with_tau else ("mu",)
    site = ex35.build_sites()[0]
    site45 = ex45.ICECUBE_SITE
    # Example 45's fitted configuration by default, as in example 47: the flat
    # attenuation override its shape fit lands on and the normalization its
    # fitted curve carries, both set on the sky average only.
    efficiency = 1.0
    if not args.first_principles:
        from dataclasses import replace
        # Example 45's nu_mu-only fit with rock below the ice (2026-09-02);
        # the all-water kernel gave 42 m and 0.755.
        site45 = replace(site45, attenuation_override_m=33.7)
        efficiency = 0.956
        print("  fitted configuration: Lambda 33.7 m, normalization 0.956 "
              "(--first-principles for derived optics)")
    print(f"  N = {min_modules:g} modules, channels {', '.join('nu_' + f for f in flavours)}, "
          f"Lambda = {ex45.attenuation_length_m(site45):.1f} m")

    sin_dec_edges, published = ex35.icecube_banded(args.data_dir)
    sin_dec_centers = 0.5 * (sin_dec_edges[:-1] + sin_dec_edges[1:])
    print(f"  {published.shape[1]} declination bands, {published.shape[0]} energies")

    print("Building the model band by band ...")
    model = model_banded(ex35, ex45, site, site45, sin_dec_edges, min_modules, flavours,
                         efficiency=efficiency)

    log10_e = ex35.COMMON_LOG10_E
    band = (log10_e >= STATS_LOG10_E[0]) & (log10_e <= STATS_LOG10_E[1])
    level, tilt, scatter = band_statistics(log10_e, published, model, band)
    columns, _ = site.columns(-sin_dec_centers)
    report(sin_dec_centers, columns, level, tilt, scatter)
    if args.scan_inelasticity:
        scan_inelasticity(ex35, ex45, site, site45, sin_dec_edges, min_modules,
                          flavours, sin_dec_centers, columns, log10_e, published, band)
    make_figures(sin_dec_centers, log10_e, published, model, level, tilt, args.out)


if __name__ == "__main__":
    main()
