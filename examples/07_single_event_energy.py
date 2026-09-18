"""07_single_event_energy.py -- the neutrino energy behind one observed track.

A detector measures the energy a muon has as it crosses, not the energy of the
neutrino that made it. Between the two lie an unknown interaction point and a
random loss history. This example inverts that for KM3-230213A, the brightest
track yet recorded.

Two ingredients enter. The potential density says how much column a muon spends
at each logarithmic loss on its way down, and the flux prior says which parent
energies were plausible to begin with. The answer moves with both, so the
reconstructed energy is a statement about the loss model as much as the event.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from softpaws.comparison.event_energy import (
    KM3_230213A,
    energy_likelihood,
    energy_posterior,
    lognormal_measurement,
    posterior_summary,
    potential_density,
    survival_through_column,
)
from softpaws.fluxes import ICECUBE_BPL_2025, ICECUBE_TRACKS_2022
from softpaws.transport import bgr18_cross_section

OUT = Path(__file__).parent / "output"
W = np.linspace(0.0, 24.0, 1601)  # logarithmic loss
LOG10_ENU = np.linspace(7.0, 10.5, 141)  # neutrino energy [log10 GeV]
PRIORS = {
    "E^-2": lambda e: (e / 1.0e5) ** -2.0,
    "Single power law": ICECUBE_TRACKS_2022.flux,
    "Broken power law": ICECUBE_BPL_2025.flux,
}


def main():
    event = KM3_230213A
    cross_section = bgr18_cross_section()
    energy = 10.0**LOG10_ENU
    survival = survival_through_column(energy, event.traversed_column_g_cm2, cross_section)
    print(f"{event.name}: a {event.muon_energy_gev / 1e6:.0f} PeV muon "
          f"through {event.traversed_column_kmwe:.0f} km water equivalent")

    def measurement(log10_e_mu):
        return lognormal_measurement(log10_e_mu, event.muon_energy_gev, event.muon_energy_90_gev)

    fig, ax = plt.subplots()
    for losses in ("exact", "gaussian"):  # the full loss law, and its Gaussian limit
        likelihood = energy_likelihood(potential_density(losses, W, n_x=120), W, energy,
                                       measurement)
        for name, prior in PRIORS.items():
            posterior = energy_posterior(likelihood, LOG10_ENU, prior, cross_section, survival)
            _, median, lo, hi = (v / 1.0e6 for v in posterior_summary(posterior, LOG10_ENU))
            print(f"  {losses:>8} losses, {name:>16} prior: median {median:4.0f} PeV, "
                  f"90% in [{lo:.0f}, {hi:.0f}]")
            if losses == "exact":
                ax.plot(LOG10_ENU, posterior, label=name)

    ax.set(xlabel=r"$\log_{10}(E_\nu/\mathrm{GeV})$", ylabel="Posterior density",
           title=event.name)
    ax.legend(title="Flux prior")
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "07_single_event_energy.png", dpi=150)


if __name__ == "__main__":
    main()
