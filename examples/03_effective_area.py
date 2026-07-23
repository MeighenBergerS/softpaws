"""Example 03 — livetime-weighted effective area, split by hemisphere.

For each IceTracks-DR2 season, loads that season's own effective-area table,
averages it over declination within the upgoing (``dec > 0``) and downgoing
(``dec < 0``) hemispheres separately (weighted by solid angle, i.e. bin
width in ``sin(dec)``), then combines the per-season hemisphere curves onto
a common energy grid using livetime weighting.

Usage
-----
    python examples/03_effective_area.py
    python examples/03_effective_area.py --data-dir /path/to/dataverse_files
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.data.loader import compute_livetime_s, load_uptime, parse_aeff
from softpaws.data.schema import SEASONS

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_DATA_DIR = _HERE.parent / "src" / "softpaws" / "data" / "dataverse_files"
_DEFAULT_OUT_DIR = _HERE / "output"

COMMON_LOG10_E = np.linspace(2.0, 10.0, 81)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=_DEFAULT_DATA_DIR,
        help="Root of the DR2 data directory (must contain 'irfs/' and 'uptime/' subfolders).",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=_DEFAULT_OUT_DIR / "03_effective_area.pdf",
        help="Output file for the figure.",
    )
    return parser.parse_args()


def hemisphere_average(aeff, hemisphere: str) -> np.ndarray:
    """Solid-angle-weighted average of an EffectiveArea grid over one hemisphere.

    Parameters
    ----------
    aeff : softpaws.response.irfs.EffectiveArea
        Tabulated effective area for one season.
    hemisphere : {"upgoing", "downgoing"}
        ``"upgoing"`` selects ``dec > 0`` (``sin(dec) > 0``) bins;
        ``"downgoing"`` selects ``dec < 0`` bins.

    Returns
    -------
    curve : np.ndarray, shape (N,)
        Effective area [cm²] on ``aeff.log10_energy_centers``, averaged
        over the selected declination bins weighted by their width in
        ``sin(dec)``.
    """
    centers = aeff.sin_dec_centers
    mask = centers > 0.0 if hemisphere == "upgoing" else centers < 0.0

    widths = np.diff(aeff.sin_dec_edges)[mask]
    values = aeff.values[:, mask]

    return np.average(values, axis=1, weights=widths)


def _canonical_irf_season(season: str) -> str:
    return "IC86" if season.startswith("IC86") else season


def combine_seasons(data_dir: pathlib.Path) -> dict[str, np.ndarray]:
    """Load per-season effective areas and combine via livetime weighting.

    Only the effective-area table is needed here, so it is loaded directly
    via :func:`~softpaws.data.loader.parse_aeff` (and cached per canonical
    IRF season) rather than via ``load_irfs``, which would also parse the
    much larger smearing table unnecessarily.

    Returns
    -------
    combined : dict[str, np.ndarray]
        ``"upgoing"`` and ``"downgoing"`` -> combined curve [cm²] on
        ``COMMON_LOG10_E``.
    """
    irf_dir = data_dir / "irfs"
    uptime_dir = data_dir / "uptime"

    sums = {"upgoing": np.zeros_like(COMMON_LOG10_E), "downgoing": np.zeros_like(COMMON_LOG10_E)}
    weights = {"upgoing": 0.0, "downgoing": 0.0}
    aeff_cache: dict[str, object] = {}

    for season in SEASONS:
        irf_season = _canonical_irf_season(season)
        if irf_season not in aeff_cache:
            raw = np.genfromtxt(irf_dir / f"{irf_season}_effectiveArea.csv", comments="#")
            aeff_cache[irf_season] = parse_aeff(raw)
        aeff = aeff_cache[irf_season]

        uptime = load_uptime(uptime_dir / f"{season}_exp.csv")
        livetime_s = compute_livetime_s(uptime)

        for hemisphere in ("upgoing", "downgoing"):
            curve = hemisphere_average(aeff, hemisphere)
            interp = np.interp(COMMON_LOG10_E, aeff.log10_energy_centers, curve)
            sums[hemisphere] += livetime_s * interp
            weights[hemisphere] += livetime_s

        print(f"  {season:>10}: livetime = {livetime_s / 86400.0:8.1f} d")

    return {h: sums[h] / weights[h] for h in ("upgoing", "downgoing")}


def make_figure(combined: dict[str, np.ndarray], out_path: pathlib.Path) -> None:
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3, 3))

        ax.plot(COMMON_LOG10_E, combined["upgoing"], label="upgoing")
        ax.plot(COMMON_LOG10_E, combined["downgoing"], label="downgoing")

        ax.set_yscale("log")
        ax.set_xlabel(r"$\log_{10}(E_\nu\,/\,\mathrm{GeV})$")
        ax.set_ylabel(r"$A_{\rm eff}$ [cm$^2$]")
        ax.legend()

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png"):
            path = out_path.with_suffix(suffix)
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {path.resolve()}")
        plt.close(fig)


def main() -> None:
    args = parse_args()

    print(f"Loading IRFs from: {args.data_dir}")
    combined = combine_seasons(args.data_dir)

    make_figure(combined, args.out)


if __name__ == "__main__":
    main()
