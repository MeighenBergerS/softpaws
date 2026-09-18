"""08_tau_induced_tracks.py -- the tracks a tau neutrino makes.

A through-going track is usually read as a muon neutrino, but it need not be
one. A tau neutrino makes a tau, and about one decay in six gives a muon that
crosses the detector like any other track. Where the Earth is opaque to muon
neutrinos this channel is no small correction: the tau neutrino regenerates on
the way through, and the muon neutrino does not.

This example computes the tau share of the track rate against energy and
arrival direction.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from softpaws.detectors import ICECUBE
from softpaws.response import directional_effective_area_cm2
from softpaws.transport.tau import BR_TAU_TO_MU, MEAN_Z, decay_length_km

OUT = Path(__file__).parent / "output"
STYLE = [Path(__file__).parents[1] / "styles" / name  # the paper's style, without LaTeX
         for name in ("beacom_conformal.mplstyle", "no_latex.mplstyle")]
THRESHOLD_GEV = 1.0e3  # muon selection threshold
LOG10_E = np.linspace(4.0, 8.0, 17)  # neutrino energy [log10 GeV]
COS_THETA = np.array([-0.99, -0.7, -0.4, -0.1])  # -1 is the nadir


def main():
    print(f"tau -> mu branching {BR_TAU_TO_MU:.3f}, mean muon energy fraction {MEAN_Z:.2f}")
    for energy in (1.0e6, 1.0e8):
        print(f"tau decay length at {energy:.0e} GeV: {decay_length_km(energy).item():.3g} km")

    muon_only = directional_effective_area_cm2(
        ICECUBE, COS_THETA, THRESHOLD_GEV, channels="mu", log10_e=LOG10_E
    )
    with_tau = directional_effective_area_cm2(
        ICECUBE, COS_THETA, THRESHOLD_GEV, channels="both", log10_e=LOG10_E
    )
    share = 1.0 - muon_only / with_tau

    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    for j, cos_theta in enumerate(COS_THETA):
        print(f"cos(theta) = {cos_theta:+.2f}: tau share {share[8, j]:.0%} at 1 PeV, "
              f"{share[-1, j]:.0%} at 100 PeV")
        ax.plot(LOG10_E, share[:, j], label=rf"$\cos\theta_z = {cos_theta:+.2f}$")
    ax.set(xlabel=r"$\log_{10}(E_\nu/\mathrm{GeV})$", ylabel="Tau share of the track rate",
           xlim=(LOG10_E[0], LOG10_E[-1]), ylim=(0.0, 1.0))
    ax.set_box_aspect(1)
    ax.legend()
    OUT.mkdir(exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(OUT / f"08_tau_induced_tracks{suffix}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
