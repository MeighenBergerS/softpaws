"""05_effective_area.py -- an effective area built from the transport.

A neutrino telescope counts a muon if it is born inside the detector or within
one range of it. The effective area is therefore the projected area of the
instrumented body times that range, plus the body itself, times the cross
section and the target density. Nothing in this is specific to one detector,
so the same call serves every site.

This example builds the sky-averaged effective area of four detectors and
holds IceCube's upgoing curve against its published table.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from softpaws.data import dr2_dir, icecube_dr2_aeff
from softpaws.detectors import ARCA230, ICECUBE, PONE, TRIDENT
from softpaws.response import sky_averaged_effective_area_cm2

OUT = Path(__file__).parent / "output"
LOG10_E = np.linspace(4.0, 8.0, 17)  # neutrino energy [log10 GeV]
THRESHOLD_GEV = 1.0e3  # muon energy a selection needs


def main():
    fig, ax = plt.subplots()
    for site in (ICECUBE, ARCA230, PONE, TRIDENT):
        aeff = sky_averaged_effective_area_cm2(site, THRESHOLD_GEV, log10_e=LOG10_E)
        ax.plot(LOG10_E, aeff, label=site.name)
        print(f"{site.name:>8}: A_eff(1 PeV) = {np.interp(6.0, LOG10_E, aeff):.3g} cm^2")

    # IceCube publishes its upgoing effective area with the DR2 data release.
    try:
        published, _ = icecube_dr2_aeff(dr2_dir(), LOG10_E)
    except FileNotFoundError:
        print("DR2 release not found, so the published comparison is skipped.")
    else:
        upgoing = sky_averaged_effective_area_cm2(
            ICECUBE, THRESHOLD_GEV, cos_range=(-1.0, 0.0), log10_e=LOG10_E
        )
        ratio = (published / upgoing)[4:13]  # 10^5 to 10^7 GeV
        print(f"IceCube published / model: {ratio.min():.2f} to {ratio.max():.2f}")
        print("The table carries the selection's turn-on, which example 06 adds.")
        ax.plot(LOG10_E, published, "k--", label="IceCube, published (upgoing)")

    ax.set(yscale="log", xlabel=r"$\log_{10}(E_\nu/\mathrm{GeV})$",
           ylabel=r"$A_\mathrm{eff}$ [cm$^2$]")
    ax.legend()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "05_effective_area.png", dpi=150)


if __name__ == "__main__":
    main()
