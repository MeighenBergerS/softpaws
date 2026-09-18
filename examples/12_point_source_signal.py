"""12_point_source_signal.py -- how bright must a point source be?

Example 11 spread the signal over the whole sky. A source is one direction, and
that changes the answer twice. The solid angle is gone, so a search reaches a
smaller flux. And the Earth is in the way for part of each day, by an amount
that depends on the source's declination and on where the detector sits, so the
same source is a different measurement at each site.

The background changes too. A search looks through one bin of sky around the
source, and the atmospheric neutrinos inside it can be counted: MCEq gives
their flux and the same response gives the effective area. The search also
picks the energy it starts at, since the atmospheric spectrum falls much faster
than the source's. This example draws the resulting limit against declination
for four sites and holds IceCube's against its published sensitivity.
Atmospheric muons are not modelled, so each site's downgoing sky is optimistic.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from softpaws.constants import SECONDS_PER_YEAR
from softpaws.data import icecube_point_source_sensitivity
from softpaws.detectors import ARCA230, ICECUBE, PONE, TRIDENT
from softpaws.fluxes import SHIPPED_TABLE, AtmosphericFlux, load_mceq_table
from softpaws.response import fit_published_reach, point_source_limit

OUT = Path(__file__).parent / "output"
STYLE = [Path(__file__).parents[1] / "styles" / name  # the paper's style, without LaTeX
         for name in ("beacom_conformal.mplstyle", "no_latex.mplstyle")]
THRESHOLD_GEV = 1.0e3  # muon selection threshold
LIVETIME_S = 14.0 * SECONDS_PER_YEAR  # the exposure of IceCube's published curve
SIN_DEC = np.linspace(-1.0, 1.0, 41)  # uniform in sin(dec) is uniform on the sky
COLORS = {"IceCube": "k", "ARCA230": "C0", "TRIDENT": "C1", "P-ONE": "C2"}


def main():
    background = AtmosphericFlux(load_mceq_table(SHIPPED_TABLE))
    dec_deg = np.rad2deg(np.arcsin(SIN_DEC))

    print("E^2 phi at 100 TeV excluded at 90% in 14 years, E^-2 source [GeV cm^-2 s^-1]:")
    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    for site in (ICECUBE, ARCA230, TRIDENT, PONE):
        try:
            reach_km, _ = fit_published_reach(site, THRESHOLD_GEV)
        except FileNotFoundError:
            print(f"{site.name:>8}: skipped, its published table needs the DR2 release")
            continue
        limit = point_source_limit(site, dec_deg, THRESHOLD_GEV, LIVETIME_S,
                                   reach_km=reach_km, background=background)
        best = np.argmin(limit)
        print(f"{site.name:>8} at {site.latitude_deg:+5.1f} deg latitude: best {limit[best]:.2g} "
              f"at dec {dec_deg[best]:+3.0f} deg, worst {limit.max():.2g}")
        ax.plot(SIN_DEC, limit, color=COLORS[site.name], label=site.name)

    published_sin_dec, published = icecube_point_source_sensitivity()
    ax.plot(published_sin_dec, published, "k--", label="IceCube, published")
    ax.set(yscale="log", xlim=(-1.0, 1.0), xlabel=r"$\sin\delta$",
           ylabel=r"$E^2\phi$ at 100 TeV [GeV cm$^{-2}$ s$^{-1}$]")
    ax.set_box_aspect(1)
    ax.legend()
    OUT.mkdir(exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(OUT / f"12_point_source_signal{suffix}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
