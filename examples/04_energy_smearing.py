"""Example 04 — reconstructed-energy smearing for fixed injected energies.

For four injected neutrino energies (1 TeV, 10 TeV, 100 TeV, 1 PeV), finds
each season's own true-energy bin containing that energy, marginalizes the
smearing matrix over declination (weighted by solid angle, whole sky) and
over PSF/AngErr, then combines the per-season reconstructed-energy
distributions via livetime weighting. Each combined distribution is
normalized to unit area so the four shapes are directly comparable.

Usage
-----
    python examples/04_energy_smearing.py
    python examples/04_energy_smearing.py --data-dir /path/to/dataverse_files
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.data.loader import compute_livetime_s, load_uptime, parse_smearing
from softpaws.data.schema import SEASONS
from softpaws.response.irfs import SmearingMatrix

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_DATA_DIR = _HERE.parent / "src" / "softpaws" / "data" / "dataverse_files"
_DEFAULT_OUT_DIR = _HERE / "output"

COMMON_LOG10_RECO_EDGES = np.linspace(1.0, 9.0, 81)
COMMON_LOG10_RECO_CENTERS = 0.5 * (COMMON_LOG10_RECO_EDGES[:-1] + COMMON_LOG10_RECO_EDGES[1:])

INJECTED_LOG10_E = (3.0, 4.0, 5.0, 6.0)
INJECTED_LABELS = ("1 TeV", "10 TeV", "100 TeV", "1 PeV")


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
        default=_DEFAULT_OUT_DIR / "04_energy_smearing.pdf",
        help="Output file for the figure.",
    )
    return parser.parse_args()


def _canonical_irf_season(season: str) -> str:
    return "IC86" if season.startswith("IC86") else season


def whole_sky_response(sm: SmearingMatrix, log10_e_injected: float) -> np.ndarray:
    """Solid-angle-weighted, whole-sky reconstructed-energy distribution.

    Parameters
    ----------
    sm : softpaws.response.irfs.SmearingMatrix
        Smearing matrix for one season.
    log10_e_injected : float
        Injected true neutrino energy, log10(E_nu/GeV).

    Returns
    -------
    dist : np.ndarray, shape (n_reco,)
        Reconstruction-energy probability, marginalized over declination
        (weighted by ``sin(dec)`` bin width) and PSF/AngErr, on
        ``COMMON_LOG10_RECO_EDGES``.
    """
    i_enu = np.searchsorted(sm.log10_enu_edges, log10_e_injected, side="right") - 1
    i_enu = int(np.clip(i_enu, 0, len(sm.log10_enu_centers) - 1))

    matrix = sm.energy_response_matrix(COMMON_LOG10_RECO_EDGES)[i_enu]  # (n_dec, n_reco)

    sin_dec_edges = np.sin(np.deg2rad(sm.dec_edges))
    weights = np.diff(sin_dec_edges)

    return np.average(matrix, axis=0, weights=weights)


def combine_seasons(data_dir: pathlib.Path) -> np.ndarray:
    """Combine per-season, whole-sky response distributions via livetime weighting.

    Only the smearing table is needed here, so it is loaded directly via
    :func:`~softpaws.data.loader.parse_smearing` (and cached per canonical
    IRF season, since IC86 sub-seasons share one file) rather than via
    ``load_irfs``.

    Returns
    -------
    combined : np.ndarray, shape (4, n_reco)
        Combined, unit-area-normalized reconstructed-energy distribution
        for each of the four injected energies in :data:`INJECTED_LOG10_E`.
    """
    irf_dir = data_dir / "irfs"
    uptime_dir = data_dir / "uptime"

    sums = np.zeros((len(INJECTED_LOG10_E), len(COMMON_LOG10_RECO_CENTERS)))
    total_livetime_s = 0.0
    sm_cache: dict[str, SmearingMatrix] = {}

    for season in SEASONS:
        irf_season = _canonical_irf_season(season)
        if irf_season not in sm_cache:
            print(f"  parsing smearing table for {irf_season} ...")
            raw = np.loadtxt(irf_dir / f"{irf_season}_smearing.csv", comments="#")
            sm_cache[irf_season] = parse_smearing(raw)
        sm = sm_cache[irf_season]

        uptime = load_uptime(uptime_dir / f"{season}_exp.csv")
        livetime_s = compute_livetime_s(uptime)
        total_livetime_s += livetime_s

        for i, log10_e in enumerate(INJECTED_LOG10_E):
            sums[i] += livetime_s * whole_sky_response(sm, log10_e)

        print(f"  {season:>10}: livetime = {livetime_s / 86400.0:8.1f} d")

    combined = sums / total_livetime_s

    bin_widths = np.diff(COMMON_LOG10_RECO_EDGES)
    areas = np.sum(combined * bin_widths, axis=1, keepdims=True)
    return combined / areas


def make_figure(combined: np.ndarray, out_path: pathlib.Path) -> None:
    reco_edges_gev = 10.0**COMMON_LOG10_RECO_EDGES

    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3, 3))

        for dist, label in zip(combined, INJECTED_LABELS):
            ax.stairs(dist, reco_edges_gev, label=label)

        ax.set_xscale("log")
        ax.set_xlabel(r"$E_{\rm reco}$ [GeV]")
        ax.set_ylabel(r"density [dex$^{-1}$]")
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
