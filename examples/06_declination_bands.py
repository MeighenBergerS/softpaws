"""06_declination_and_point_sources.py -- the response by declination.

Averaging over the sky hides the one axis a point-source search cares about.
Seen from the South Pole a declination is a fixed zenith, so IceCube's
published effective area can be compared with the model band by band.

This example makes that comparison twice, once with the instrumented footprint
alone and once with the light reach of example 09. The reach shrinks the
effective body below its pivot energy, which is how the model carries the
selection's turn-on. It then turns each response into the flux a
background-free search would exclude at each declination.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from softpaws.data import banded_effective_area, icecube_point_source_sensitivity
from softpaws.detectors import ICECUBE
from softpaws.response import (
    COMMON_LOG10_E,
    band_averaged_effective_area_cm2,
    band_statistics,
    fit_published_reach,
    point_source_sensitivity,
)
from softpaws.utils.constants import SECONDS_PER_YEAR

OUT = Path(__file__).parent / "output"
THRESHOLD_GEV = 1.0e3  # muon selection threshold
LIVETIME_S = 10.0 * SECONDS_PER_YEAR
GAMMA = 2.0  # spectral index of the source


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

    fig, (ax_level, ax_flux) = plt.subplots(1, 2, figsize=(10, 4))
    for name, aeff in models.items():
        level, _, _ = band_statistics(COMMON_LOG10_E, published, aeff, scored)
        print(f"{name:>10}: published / model = {np.nanmean(level[north]):.2f} "
              "averaged over the northern bands")
        ax_level.plot(sin_dec[north], level[north], label=name)

        ceiling = point_source_sensitivity(aeff, LIVETIME_S, GAMMA)
        ax_flux.plot(sin_dec[north], ceiling[north], label=f"Model, {name.lower()}")

    pub_sin, pub_flux = icecube_point_source_sensitivity()
    ax_flux.plot(pub_sin[pub_sin > 0.05], pub_flux[pub_sin > 0.05], "k--",
                 label="IceCube, published")
    ax_level.set(xlabel=r"$\sin\delta$", ylabel="Published / model", ylim=(0.0, 2.0))
    ax_flux.set(xlabel=r"$\sin\delta$", yscale="log",
                ylabel=r"$E^2\phi$ at 100 TeV [GeV cm$^{-2}$ s$^{-1}$]")
    ax_level.legend()
    ax_flux.legend()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "06_declination_and_point_sources.png", dpi=150)


if __name__ == "__main__":
    main()
