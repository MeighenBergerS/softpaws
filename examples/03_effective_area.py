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

from softpaws.data.icecube import (
    livetime_weighted_effective_area,
)

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


def combine_seasons(data_dir: pathlib.Path) -> dict[str, np.ndarray]:
    """Livetime-weighted effective area per hemisphere on ``COMMON_LOG10_E`` [cm^2]."""
    return {
        h: livetime_weighted_effective_area(data_dir, COMMON_LOG10_E, h)[0]
        for h in ("upgoing", "downgoing")
    }


def make_figure(combined: dict[str, np.ndarray], out_path: pathlib.Path) -> None:
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3, 3))

        ax.plot(COMMON_LOG10_E, combined["upgoing"], label="upgoing")
        np.savetxt("upgoing.txt", combined["upgoing"])
        np.savetxt("energy_grid.txt", COMMON_LOG10_E)
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
