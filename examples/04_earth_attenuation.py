"""04_earth_attenuation.py -- what the Earth does to a neutrino on the way in.

A neutrino arriving from below crosses up to a full Earth diameter of matter.
Above a few tens of TeV that matter is no longer transparent, and by an EeV
the Earth is opaque except near the horizon. This example computes the column
along each arrival direction from the PREM profile, turns it into a survival
probability, and shows what neutral-current regeneration adds back.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from softpaws.detectors import ICECUBE
from softpaws.transport import bgr18_cross_section, earth
from softpaws.transport.attenuation import regenerated_transmission, survival_probability

OUT = Path(__file__).parent / "output"
STYLE = [Path(__file__).parents[1] / "styles" / name  # the paper's style, without LaTeX
         for name in ("beacom_conformal.mplstyle", "no_latex.mplstyle")]
COS_THETA = np.linspace(-1.0, 0.0, 121)  # upgoing directions
LOG10_E = np.linspace(3.0, 9.0, 121)  # neutrino energy [log10 GeV]


def main():
    cross_section = bgr18_cross_section()
    for dec_deg in (5.0, 30.0, 90.0):  # at the South Pole, declination is elevation
        column = earth.prem_column(dec_deg)
        survival = survival_probability(1.0e6, column, cross_section=cross_section)
        print(f"dec {dec_deg:4.0f} deg: column {column:.2e} g/cm^2, "
              f"1 PeV survival {np.squeeze(survival):.3f}")

    column = earth.prem_column(30.0)
    absorbed = np.squeeze(survival_probability(1.0e7, column, cross_section=cross_section))
    _, weights = regenerated_transmission(1.0e7, np.array([column]), cross_section)
    print(f"10 PeV at dec 30 deg: {absorbed:.4f} survive absorption, "
          f"{weights.sum():.4f} with regeneration")

    columns = np.ravel([earth.neutrino_column_g_cm2(c, ICECUBE.depth_km, ICECUBE.density_g_cm3)
                        for c in COS_THETA])
    energies = 10.0**LOG10_E
    survival = np.array([np.ravel(survival_probability(energies, c, cross_section=cross_section))
                         for c in columns])

    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    mesh = ax.pcolormesh(LOG10_E, COS_THETA, survival, vmin=0.0, vmax=1.0, cmap="magma",
                         rasterized=True)
    ax.set(xlabel=r"$\log_{10}(E_\nu/\mathrm{GeV})$", ylabel=r"$\cos\theta_z$")
    ax.set_box_aspect(1)
    fig.colorbar(mesh, ax=ax, label="Survival probability", fraction=0.046, pad=0.04)
    OUT.mkdir(exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(OUT / f"04_earth_attenuation{suffix}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
