"""01_load_the_release.py -- what the IceCube release contains.

The IceTracks-DR2 release gives four things: reconstructed track events, a
binned effective area per detector configuration, a smearing matrix, and the
good-run uptime. This example loads them, reports the exposure, and draws the
effective area over the two hemispheres.

It needs the release on disk. Place it under the package data directory or
point SOFTPAWS_DATA_DIR at it; see the data guide.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from softpaws.constants import SECONDS_PER_YEAR
from softpaws.data import (
    IC86_SEASONS,
    dr2_dir,
    livetime_weighted_effective_area,
    load_effective_area,
    load_events,
    total_livetime_s,
)

OUT = Path(__file__).parent / "output"
STYLE = [Path(__file__).parents[1] / "styles" / name  # the paper's style, without LaTeX
         for name in ("beacom_conformal.mplstyle", "no_latex.mplstyle")]
LOG10_E = np.linspace(3.0, 8.0, 26)  # neutrino energy [log10 GeV]


def main():
    print(f"Reading the release from {dr2_dir()}")
    print(f"Livetime: {total_livetime_s() / SECONDS_PER_YEAR:.2f} yr in total, "
          f"{total_livetime_s(seasons=IC86_SEASONS) / SECONDS_PER_YEAR:.2f} yr in IC86")

    events = load_events(seasons=IC86_SEASONS)
    print(f"IC86 events: {events.n_events:,}, of which {np.sum(events.dec > 0.0):,} upgoing")

    table = load_effective_area(None, "IC86_I")
    print(f"IC86 effective-area table: {table.values.shape[0]} energies "
          f"x {table.values.shape[1]} declination bands")

    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    for hemisphere in ("upgoing", "downgoing"):
        aeff, _ = livetime_weighted_effective_area(None, LOG10_E, hemisphere)
        ax.plot(LOG10_E, aeff, label=f"IceCube DR2, {hemisphere}")
    ax.set(yscale="log", xlabel=r"$\log_{10}(E_\nu/\mathrm{GeV})$",
           ylabel=r"Livetime-weighted $A_\mathrm{eff}$ [cm$^2$]")
    ax.set_box_aspect(1)
    ax.legend()
    OUT.mkdir(exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(OUT / f"01_load_the_release{suffix}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
