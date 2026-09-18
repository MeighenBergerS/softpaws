"""09_fitting_the_light_reach.py -- one instrument number from a published table.

Example 05 left a gap: built from the instrumented footprint alone, the model
sits above IceCube's published table. A real selection turns on gradually, so
a detector responds to only part of its footprint at low energy and to more of
it as the muons get brighter.

The light reach captures that with one number. The effective radius grows by
Lambda per e-fold of muon energy about a pivot far above the table, so below
the pivot the body is smaller than the footprint. This example fits Lambda to
IceCube's published upgoing effective area and shows what it does to the
comparison.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from softpaws.data import dr2_dir, icecube_dr2_aeff
from softpaws.detectors import ICECUBE
from softpaws.response import fit_published_reach, sky_averaged_effective_area_cm2

OUT = Path(__file__).parent / "output"
STYLE = [Path(__file__).parents[1] / "styles" / name  # the paper's style, without LaTeX
         for name in ("beacom_conformal.mplstyle", "no_latex.mplstyle")]
THRESHOLD_GEV = 1.0e3  # muon selection threshold
LOG10_E = np.linspace(4.0, 8.0, 17)  # neutrino energy [log10 GeV]
UPGOING = (-1.0, 0.0)  # the sky the published table covers


def main():
    reach_km, residual_dex = fit_published_reach(ICECUBE, THRESHOLD_GEV)
    print(f"Fitted reach: {reach_km * 1e3:.1f} m per e-fold, residual {residual_dex:.3f} dex")

    published, _ = icecube_dr2_aeff(dr2_dir(), LOG10_E)
    models = {
        "Footprint": sky_averaged_effective_area_cm2(
            ICECUBE, THRESHOLD_GEV, cos_range=UPGOING, log10_e=LOG10_E
        ),
        "With reach": sky_averaged_effective_area_cm2(
            ICECUBE, THRESHOLD_GEV, cos_range=UPGOING, reach_km=reach_km, log10_e=LOG10_E
        ),
    }

    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    ax.plot(LOG10_E, published, color="k", lw=1.4, label="IceCube, published")
    for (name, model), color in zip(models.items(), ("C1", "C0")):
        ratio = (published / model)[4:15]  # 10^5 to 10^7.5 GeV
        print(f"{name:>10}: published / model = {np.exp(np.mean(np.log(ratio))):.2f}")
        ax.plot(LOG10_E, model, "--", color=color, label=f"Model, {name.lower()}")
    ax.set(yscale="log", xlabel=r"$\log_{10}(E_\nu/\mathrm{GeV})$",
           ylabel=r"Upgoing $A_\mathrm{eff}$ [cm$^2$]")
    ax.set_box_aspect(1)
    ax.legend()
    OUT.mkdir(exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(OUT / f"09_fitting_the_light_reach{suffix}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
