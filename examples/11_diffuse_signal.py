"""11_diffuse_signal.py -- how large must a new diffuse signal be?

Examples 05 and 09 built the response and calibrated it. This one runs it
backwards: for a signal nobody has seen yet, how large must it be before each
telescope sees it?

Three injections cover most of what a proposal asks for. A single event is the
weakest claim an experiment can make. A line is a flux at one energy, which is
what a decaying or annihilating particle of fixed mass gives. A power law
spreads the normalization over the whole band. All three are linear in the
normalization, so each is one division by the exposure. Every site carries the
light reach fitted to its own published table, as in example 09.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from softpaws.detectors import ARCA230, ICECUBE, PONE, TRIDENT
from softpaws.fluxes import ICECUBE_TRACKS_2022
from softpaws.response import (
    fit_published_reach,
    line_sensitivity,
    power_law_sensitivity,
    single_event_sensitivity,
    sky_averaged_effective_area_cm2,
)
from softpaws.utils.constants import SECONDS_PER_YEAR

OUT = Path(__file__).parent / "output"
STYLE = [Path(__file__).parents[1] / "styles" / name  # the paper's style, without LaTeX
         for name in ("beacom_conformal.mplstyle", "no_latex.mplstyle")]
THRESHOLD_GEV = 1.0e3  # muon selection threshold
LIVETIME_S = 10.0 * SECONDS_PER_YEAR
LOG10_E = np.arange(4.0, 8.01, 0.25)  # neutrino energy [log10 GeV]
PEV = 8  # index of 1 PeV in LOG10_E
SKY_SR = 4.0 * np.pi
COLORS = {"IceCube": "k", "ARCA230": "C0", "TRIDENT": "C1", "P-ONE": "C2"}


def main():
    print("Flux needed in 10 years, per flavour, no background [GeV cm^-2 s^-1 sr^-1]:")
    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    for site in (ICECUBE, ARCA230, TRIDENT, PONE):
        try:
            reach_km, _ = fit_published_reach(site, THRESHOLD_GEV, log10_e=LOG10_E)
        except FileNotFoundError:
            print(f"{site.name:>8}: skipped, its published table needs the DR2 release")
            continue
        aeff = sky_averaged_effective_area_cm2(
            site, THRESHOLD_GEV, reach_km=reach_km, log10_e=LOG10_E
        )
        one_event = single_event_sensitivity(aeff, LIVETIME_S, LOG10_E, solid_angle_sr=SKY_SR)
        line = 1.0e6 * line_sensitivity(aeff, LIVETIME_S, solid_angle_sr=SKY_SR)[PEV]
        power_law = power_law_sensitivity(aeff, LIVETIME_S, 2.0, LOG10_E, solid_angle_sr=SKY_SR)
        print(f"{site.name:>8}: one event at 1 PeV {one_event[PEV]:.2g}, "
              f"line at 1 PeV {line:.2g}, E^-2 at 100 TeV {float(power_law):.2g}")
        ax.plot(LOG10_E, one_event, color=COLORS[site.name], label=site.name)

    energy = 10.0**LOG10_E
    ax.plot(LOG10_E, energy**2 * ICECUBE_TRACKS_2022.flux(energy), "--", color="0.5",
            label="Measured flux")
    ax.text(0.95, 0.95, "One event per decade\nin 10 years", transform=ax.transAxes,
            ha="right", va="top")
    ax.set(yscale="log", xlim=(LOG10_E[0], LOG10_E[-1]),
           xlabel=r"$\log_{10}(E_\nu/\mathrm{GeV})$",
           ylabel=r"$E^2\phi$ [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    ax.set_box_aspect(1)
    ax.legend()
    OUT.mkdir(exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(OUT / f"11_diffuse_signal{suffix}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
