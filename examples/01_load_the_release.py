"""Tutorial 01 -- what the IceCube release contains, and how to read it.

The IceTracks-DR2 release gives four things: reconstructed track events, a
binned effective area per detector configuration, a smearing matrix, and the
good-run uptime. This tutorial loads each of them, says how much exposure the
release carries, and draws the effective area over the two hemispheres.

It needs the release on disk; see the data guide for where to put it.

Usage
-----
    python examples/01_load_the_release.py
    SOFTPAWS_DATA_DIR=/data/neutrino python examples/01_load_the_release.py
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.data import (
    IC86_SEASONS,
    banded_effective_area,
    data_root,
    dr2_dir,
    livetime_weighted_effective_area,
    load_effective_area,
    load_events,
    total_livetime_s,
)
from softpaws.data.schema import SEASONS

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"

YEAR_S = 365.25 * 86400.0

#: Energies the curves are drawn on [log10 GeV].
LOG10_E = np.linspace(3.0, 8.0, 26)


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=pathlib.Path, default=None,
                        help="Root of the DR2 release; the default follows SOFTPAWS_DATA_DIR.")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the figure.")
    return parser.parse_args()


def report(data_dir) -> None:
    """Print the exposure, the sample size and one effective-area value."""
    print(f"Reading the release from: {data_dir or dr2_dir()}")
    print(f"  (the data root is {data_root()})")

    total_yr = total_livetime_s(data_dir) / YEAR_S
    ic86_yr = total_livetime_s(data_dir, IC86_SEASONS) / YEAR_S
    print(f"\n{len(SEASONS)} seasons, {total_yr:.2f} years of livetime in total")
    print(f"  of which the {len(IC86_SEASONS)} IC86 seasons carry {ic86_yr:.2f} years")

    events = load_events(data_dir, IC86_SEASONS)
    upgoing = int(np.sum(events.dec > 0.0))
    print(f"\n{events.n_events:,} IC86 events, {upgoing:,} of them upgoing")
    print(f"  reconstructed energy runs from 10^{events.log10_energy.min():.1f} "
          f"to 10^{events.log10_energy.max():.1f} GeV")

    aeff = load_effective_area(data_dir, "IC86_I")
    print(f"\nThe IC86 effective-area table is {aeff.values.shape[0]} energies "
          f"by {aeff.values.shape[1]} declination bands")
    print(f"  at 1 PeV, upgoing: {np.ravel(aeff(6.0, 0.5))[0]:.3g} cm^2")


def make_figure(data_dir, out_path: pathlib.Path) -> None:
    """Draw the livetime-weighted effective area over both hemispheres."""
    curves = {
        hemisphere: livetime_weighted_effective_area(data_dir, LOG10_E, hemisphere)[0]
        for hemisphere in ("upgoing", "downgoing")
    }
    edges, banded = banded_effective_area(data_dir, LOG10_E)
    centres = 0.5 * (edges[:-1] + edges[1:])

    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        # The bands behind, so the hemisphere averages read on top of them.
        for j in range(centres.size):
            if not np.any(banded[:, j] > 0.0):
                continue
            ax.plot(LOG10_E, banded[:, j], color="0.85", lw=0.5, zorder=1)
        for name, style in (("upgoing", "-"), ("downgoing", "--")):
            ax.plot(LOG10_E, curves[name], ls=style, lw=1.4, label=name, zorder=3)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\log_{10}(E_\nu/\mathrm{GeV})$")
        ax.set_ylabel(r"$A_{\rm eff}$ [cm$^2$]")
        ax.set_title("IceCube DR2, livetime weighted")
        ax.legend(frameon=False)
        fig.tight_layout()
        for suffix in (".pdf", ".png"):
            fig.savefig(out_path.with_suffix(suffix))
            print(f"Figure saved to: {out_path.with_suffix(suffix)}")
        plt.close(fig)


def main() -> None:
    """Report the release and draw its effective area."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report(args.data_dir)
    make_figure(args.data_dir, args.out_dir / "t01_load_the_release")


if __name__ == "__main__":
    main()
