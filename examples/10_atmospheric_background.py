"""10_atmospheric_background.py -- the background every track analysis sits on.

Most of what a neutrino telescope records is not astrophysical. Cosmic rays
hitting the atmosphere make muon neutrinos, and below a few hundred TeV these
outnumber the astrophysical ones by orders of magnitude. Where the two fluxes
cross decides the energy window an analysis can use.

MCEq solves the cascade equations for this background. A run takes minutes, so
the package ships a precomputed table; with the atm extra installed,
load_mceq_table(path, recompute=True) builds a new one. This example folds the
shipped table through IceCube's upgoing effective area and finds the crossing.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from softpaws.constants import SECONDS_PER_YEAR
from softpaws.detectors import ICECUBE
from softpaws.fluxes import ICECUBE_TRACKS_2022, SHIPPED_TABLE, AtmosphericFlux, load_mceq_table
from softpaws.response import sky_averaged_effective_area_cm2

OUT = Path(__file__).parent / "output"
STYLE = [Path(__file__).parents[1] / "styles" / name  # the paper's style, without LaTeX
         for name in ("beacom_conformal.mplstyle", "no_latex.mplstyle")]
THRESHOLD_GEV = 1.0e3  # muon selection threshold
LOG10_E = np.linspace(3.0, 8.0, 41)  # neutrino energy [log10 GeV]
LIVETIME_S = 10.0 * SECONDS_PER_YEAR
UPGOING_SR = 2.0 * np.pi  # solid angle of the upgoing sky


def main():
    flux = AtmosphericFlux(load_mceq_table(SHIPPED_TABLE))
    energy = 10.0**LOG10_E
    astrophysical = np.ravel(ICECUBE_TRACKS_2022.flux(energy))
    atmospheric = np.ravel(flux(energy, np.full_like(energy, 30.0)))  # at dec = 30 deg
    crossing = np.interp(0.0, np.log(atmospheric / astrophysical)[::-1], LOG10_E[::-1])
    print(f"The astrophysical flux takes over above 10^{crossing:.1f} GeV")

    aeff = sky_averaged_effective_area_cm2(
        ICECUBE, THRESHOLD_GEV, cos_range=(-1.0, 0.0), n_zenith=12, log10_e=LOG10_E
    )
    # Indicative only: one sky-averaged area, no reconstruction and no selection.
    exposure = aeff * LIVETIME_S * UPGOING_SR * np.gradient(energy)  # cm^2 s sr GeV
    for log10_min in (4.0, 5.0, 6.0):
        above = LOG10_E >= log10_min
        print(f"Upgoing events above 10^{log10_min:.0f} GeV in 10 years: "
              f"{np.sum((atmospheric * exposure)[above]):9.1f} atmospheric, "
              f"{np.sum((astrophysical * exposure)[above]):7.1f} astrophysical")

    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    ax.plot(LOG10_E, energy**2 * astrophysical, color="k", lw=1.4, label="Astrophysical")
    for dec_deg in (5.0, 30.0, 60.0, 85.0):
        values = np.ravel(flux(energy, np.full_like(energy, dec_deg)))
        ax.plot(LOG10_E, energy**2 * values, label=rf"Atmospheric, $\delta = {dec_deg:.0f}^\circ$")
    ax.set(yscale="log", xlim=(LOG10_E[0], LOG10_E[-1]), ylim=(1e-10, 1e-3),
           xlabel=r"$\log_{10}(E_\nu/\mathrm{GeV})$",
           ylabel=r"$E^2\phi$ [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    ax.set_box_aspect(1)
    ax.legend()
    OUT.mkdir(exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(OUT / f"10_atmospheric_background{suffix}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
