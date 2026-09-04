"""Tutorial 06 -- the response band by band, and what it means for a source.

Averaging over the sky hides the one axis a point-source search cares about.
Seen from the South Pole a declination is a fixed zenith, so the published
effective area can be compared band by band with no time averaging at all.

This tutorial compares the model with the DR2 table in each published band,
adds the light reach that tutorial 05 left out, and turns the result into the
flux a background-free search would exclude at each declination.

Usage
-----
    python examples/06_declination_and_point_sources.py
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.data import banded_effective_area, icecube_point_source_sensitivity
from softpaws.detectors import ICECUBE
from softpaws.response.declination import (
    COMMON_LOG10_E,
    band_averaged_effective_area_cm2,
    band_statistics,
    point_source_sensitivity,
)

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"

#: Band the level and the tilt are scored over [log10 GeV].
STATS_BAND = (5.0, 7.8)

#: Light reach per e-fold of muon energy [km], and the energy it vanishes at.
REACH_KM = 0.0178
REACH_PIVOT_GEV = 10.0**9.67

#: Exposure and spectrum of the point-source ceiling.
LIVETIME_YR = 10.0
GAMMA = 2.0

YEAR_S = 365.25 * 86400.0


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threshold-gev", type=float, default=1.0e3,
                        help="Muon selection threshold.")
    parser.add_argument("--data-dir", type=pathlib.Path, default=None,
                        help="Root of the DR2 release.")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the figures.")
    return parser.parse_args()


def report(sin_dec: np.ndarray, published: np.ndarray, models: dict[str, np.ndarray]) -> None:
    """Print the level, tilt and scatter of each band, with and without the reach."""
    band = (COMMON_LOG10_E >= STATS_BAND[0]) & (COMMON_LOG10_E <= STATS_BAND[1])
    upgoing = sin_dec > 0.05
    print(f"Published over model, scored over 10^{STATS_BAND[0]} to "
          f"10^{STATS_BAND[1]} GeV:\n")
    for name, model in models.items():
        level, tilt, scatter = band_statistics(COMMON_LOG10_E, published, model, band)
        print(f"  {name}")
        print(f"{'sin(dec)':>10} {'level':>8} {'tilt':>8} {'scatter':>9}")
        for j in np.flatnonzero(upgoing)[::6]:
            print(f"{sin_dec[j]:10.2f} {level[j]:8.3f} {tilt[j]:+8.3f} {scatter[j]:9.4f}")
        keep = upgoing & np.isfinite(level)
        print(f"{'mean':>10} {np.nanmean(level[keep]):8.3f} "
              f"{np.nanmean(tilt[keep]):+8.3f} {np.nanmean(scatter[keep]):9.4f}\n")
    print("  The static model is low and its tilt is small: the shape is right and")
    print("  the normalization is not. Growing the body with the light the muon")
    print("  makes closes most of the gap without touching the energy dependence.")


def make_figures(sin_dec: np.ndarray, published: np.ndarray, models: dict[str, np.ndarray],
                 out_stem: pathlib.Path) -> None:
    """Draw the band ratios and the point-source ceiling."""
    upgoing = sin_dec > 0.05
    band = (COMMON_LOG10_E >= STATS_BAND[0]) & (COMMON_LOG10_E <= STATS_BAND[1])

    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        for name, model in models.items():
            level = band_statistics(COMMON_LOG10_E, published, model, band)[0]
            ax.plot(sin_dec[upgoing], level[upgoing], lw=1.3, label=name)
        ax.axhline(1.0, color="0.7", lw=0.8, zorder=0)
        ax.set_xlabel(r"$\sin(\mathrm{dec})$")
        ax.set_ylabel("published / model")
        ax.set_title("IceCube DR2, band by band")
        ax.set_ylim(0.0, 2.0)
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        for suffix in (".pdf", ".png"):
            path = out_stem.with_name(out_stem.name + "a").with_suffix(suffix)
            fig.savefig(path)
            print(f"Figure saved to: {path}")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        for name, model in models.items():
            ceiling = point_source_sensitivity(model, LIVETIME_YR * YEAR_S, GAMMA)
            ax.plot(sin_dec[upgoing], ceiling[upgoing], lw=1.3, label=f"model, {name}")
        pub_sin, pub_flux = icecube_point_source_sensitivity()
        north = pub_sin > 0.05
        ax.plot(pub_sin[north], pub_flux[north], color="0.4", lw=1.6, ls="--",
                label="IceCube, published")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\sin(\mathrm{dec})$")
        ax.set_ylabel(r"$E^2\phi$ at 100 TeV [GeV cm$^{-2}$ s$^{-1}$]")
        ax.set_title(f"{LIVETIME_YR:.0f} years, $\\gamma = {GAMMA:.0f}$")
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        for suffix in (".pdf", ".png"):
            path = out_stem.with_name(out_stem.name + "b").with_suffix(suffix)
            fig.savefig(path)
            print(f"Figure saved to: {path}")
        plt.close(fig)


def main() -> None:
    """Compare band by band and draw the ceiling."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading the published table, band by band ...")
    edges, published = banded_effective_area(args.data_dir, COMMON_LOG10_E)
    sin_dec = 0.5 * (edges[:-1] + edges[1:])

    models = {}
    print("Building the model with the instrumented footprint only ...")
    models["footprint"] = band_averaged_effective_area_cm2(
        ICECUBE, edges, args.threshold_gev, n_sub=3
    )
    print("Building the model with the light reach ...")
    models["with reach"] = band_averaged_effective_area_cm2(
        ICECUBE, edges, args.threshold_gev, reach_km=REACH_KM, pivot_gev=REACH_PIVOT_GEV,
        n_sub=3
    )

    report(sin_dec, published, models)
    make_figures(sin_dec, published, models, args.out_dir / "t06_declination")


if __name__ == "__main__":
    main()
