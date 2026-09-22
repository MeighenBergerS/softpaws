"""03_range_and_loss_law.py -- how far a muon goes, and what it loses.

Two questions follow from the transport exponent. How far does a muon travel
before it falls below a threshold, and how is the energy it has left spread
after a fixed distance?

The first is a first-passage problem with a closed form. It comes out shorter
than the mean-loss range, because a muon that radiates hard early never gets
as far as the mean. The second is the loss law, whose tail decides how bright
a track can be. The Gaussian approximation misses that tail by orders of
magnitude.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from softpaws.transport import (
    diffusion_coefficient,
    drift_coefficient,
    muon_range_km,
    stochastic_muon_range_km,
    third_moment_coefficient,
)
from softpaws.transport.loss_distribution import (
    loss_density,
    loss_density_gaussian,
    loss_density_three_moment,
    survival_from_density,
)

OUT = Path(__file__).parent / "output"
STYLE = [Path(__file__).parents[1] / "styles" / name  # the paper's style, without LaTeX
         for name in ("beacom_conformal.mplstyle", "no_latex.mplstyle")]
THRESHOLD_GEV = 1.0e3  # the range runs down to this muon energy
ENERGY_GEV = 1.0e6  # muon energy of the loss law
DISTANCE_KM = 1.0  # distance the loss law is taken after [km of water]
W = np.linspace(-1.0, 8.0, 3001)  # log loss, w = ln(E_start / E_end)


def main():
    energies = 10.0 ** np.arange(4.0, 9.0)
    first_passage = stochastic_muon_range_km(energies, THRESHOLD_GEV)
    mean_loss = muon_range_km(energies, THRESHOLD_GEV)
    for e, fp, ml in zip(energies, first_passage, mean_loss, strict=True):
        print(f"{e:.0e} GeV muon: range {fp:5.2f} km, mean-loss range {ml:5.2f} km")

    b, d, t = (f(ENERGY_GEV).item() for f in
               (drift_coefficient, diffusion_coefficient, third_moment_coefficient))
    densities = {
        "Three moments": loss_density_three_moment(W, DISTANCE_KM, b, d, t),
        "Two moments": loss_density(W, DISTANCE_KM, b, d),
        "Gaussian": loss_density_gaussian(W, DISTANCE_KM, b, d),
    }
    print(f"Chance to lose more than a factor {np.exp(1.5):.1f} over {DISTANCE_KM:g} km:")
    for name, density in densities.items():
        print(f"  {name:>13}: {float(survival_from_density(1.5, W, density)):.1e}")

    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    for (name, density), style in zip(densities.items(), ("-", "--", ":"), strict=False):
        ax.plot(W, np.clip(density, 1e-12, None), style, label=name)
    ax.set(yscale="log", xlim=(0.0, 5.0), ylim=(1e-4, 5.0),
           xlabel=r"Log loss $w = \ln(\varepsilon/E)$", ylabel=r"Probability density $P(w)$")
    ax.set_box_aspect(1)
    ax.legend()
    OUT.mkdir(exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(OUT / f"03_range_and_loss_law{suffix}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
