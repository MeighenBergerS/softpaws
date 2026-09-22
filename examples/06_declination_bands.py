"""06_declination_bands.py -- the response by declination.

Averaging over the sky hides the one axis a point-source search cares about.
Seen from the South Pole a declination is a fixed zenith, so IceCube's
published effective area can be compared with the model band by band.

This example makes that comparison twice, once with the instrumented footprint
alone and once with the light reach of example 09. The reach shrinks the
effective body below its pivot energy, which is how the model carries the
selection's turn-on. Example 12 turns the same response into the flux a
point-source search reaches.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from softpaws.data import banded_effective_area
from softpaws.detectors import ICECUBE
from softpaws.response import (
    COMMON_LOG10_E,
    band_averaged_effective_area_cm2,
    band_statistics,
    fit_published_reach,
)

OUT = Path(__file__).parent / "output"
STYLE = [Path(__file__).parents[1] / "styles" / name  # the paper's style, without LaTeX
         for name in ("beacom_conformal.mplstyle", "no_latex.mplstyle")]
THRESHOLD_GEV = 1.0e3  # muon selection threshold


def main():
    edges, published = banded_effective_area(None, COMMON_LOG10_E)
    sin_dec = 0.5 * (edges[:-1] + edges[1:])
    north = sin_dec > 0.05  # upgoing at the Pole
    scored = (COMMON_LOG10_E >= 5.0) & (COMMON_LOG10_E <= 7.8)
    reach_km, _ = fit_published_reach(ICECUBE, THRESHOLD_GEV)  # as in example 09

    models = {
        "Footprint": band_averaged_effective_area_cm2(ICECUBE, edges, THRESHOLD_GEV, n_sub=3),
        "With reach": band_averaged_effective_area_cm2(
            ICECUBE, edges, THRESHOLD_GEV, reach_km=reach_km, n_sub=3
        ),
    }

    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    for (name, aeff), color in zip(models.items(), ("C1", "C0"), strict=False):
        level, _, _ = band_statistics(COMMON_LOG10_E, published, aeff, scored)
        print(f"{name:>10}: published / model = {np.nanmean(level[north]):.2f} "
              "averaged over the northern bands")
        ax.plot(sin_dec[north], level[north], color=color, label=name)

    ax.axhline(1.0, color="0.7", lw=0.6, ls=":")
    ax.set(xlabel=r"$\sin\delta$", ylabel="Published / model", xlim=(0.0, 1.0),
           ylim=(0.0, 2.0))
    ax.set_box_aspect(1)
    ax.legend()
    OUT.mkdir(exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(OUT / f"06_declination_bands{suffix}", dpi=300,
                    bbox_inches="tight")


if __name__ == "__main__":
    main()
