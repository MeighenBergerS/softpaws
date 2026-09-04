"""Tutorial 07 -- the neutrino energy behind one observed track.

A detector measures the energy a muon has as it crosses, not the energy of
the neutrino that made it. Between the two lie an unknown interaction point
and a stochastic loss history. Inverting that is what this tutorial does, for
KM3-230213A, the brightest track yet recorded.

Two ingredients enter. The potential density says how much column a muon
spends at each log loss on its way down, and the flux prior says which parent
energies were plausible to begin with. The answer moves with both, which is
the point: the reconstructed energy is a statement about the loss model as
much as about the event.

Usage
-----
    python examples/07_single_event_energy.py
"""

import argparse
import pathlib

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

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"

#: Log-loss grid the potential density is built on.
W_GRID = np.linspace(0.0, 24.0, 1601)

#: Neutrino energies the posterior is drawn on [log10 GeV].
LOG10_ENU = np.linspace(7.0, 10.5, 141)

#: Slabs the potential density integrates over. Halving this moves the
#: medians by about 4%, so it is left at the value the paper uses.
N_X = 120

#: Loss families compared.
KINDS = ("exact", "gaussian", "csda")


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the figure.")
    return parser.parse_args()


def flux_priors() -> dict:
    """The three flux priors the posterior is compared under."""
    return {
        "E-2": lambda e: (e / 1.0e5) ** -2.0,
        "SPL": lambda e: ICECUBE_TRACKS_2022.flux(e),
        "BPL": lambda e: ICECUBE_BPL_2025.flux(e),
    }


def posteriors() -> dict:
    """The posterior of every loss family and flux prior."""
    event = KM3_230213A
    cross_section = bgr18_cross_section()
    energy = 10.0**LOG10_ENU
    survival = survival_through_column(energy, event.traversed_column_g_cm2, cross_section)

    def measurement(log10_e_mu):
        return lognormal_measurement(
            log10_e_mu, event.muon_energy_gev, event.muon_energy_90_gev
        )

    out = {}
    for kind in KINDS:
        print(f"Building the potential density, {kind} ...")
        u = potential_density(kind, W_GRID, n_x=N_X)
        likelihood = energy_likelihood(u, W_GRID, energy, measurement)
        for name, shape in flux_priors().items():
            out[(kind, name)] = energy_posterior(
                likelihood, LOG10_ENU, shape, cross_section, survival
            )
    return out


def report(results: dict) -> None:
    """Print the mode, the median and the 90% interval of every combination."""
    event = KM3_230213A
    print(f"\n{event.name}: a muon of {event.muon_energy_gev / 1e6:.0f} PeV "
          f"({event.muon_energy_90_gev[0] / 1e6:.0f} to "
          f"{event.muon_energy_90_gev[1] / 1e6:.0f} at 90%),")
    print(f"  arriving {event.elevation_deg:+.1f} deg above the horizon through "
          f"{event.traversed_column_kmwe:.0f} km w.e.\n")
    print(f"{'flux prior':>12} {'loss family':>13} {'mode':>8} {'median':>8} "
          f"{'90% interval':>20}   [PeV]")
    for (kind, prior), p in results.items():
        mode, median, lo, hi = (v / 1.0e6 for v in posterior_summary(p, LOG10_ENU))
        print(f"{prior:>12} {kind:>13} {mode:8.0f} {median:8.0f} "
              f"{f'[{lo:.0f}, {hi:.0f}]':>20}")
    print("\n  The prior moves the answer by more than the loss family does. A")
    print("  harder prior pulls the parent energy up; a Gaussian loss law, which")
    print("  cannot make a big loss, pushes it up too, by about a fifth.")


def make_figure(results: dict, out_path: pathlib.Path) -> None:
    """Draw the posterior of each flux prior, for the exact loss family."""
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        labels = {"E-2": r"$E^{-2}$", "SPL": "SPL", "BPL": "BPL"}
        for prior in flux_priors():
            ax.plot(LOG10_ENU, results[("exact", prior)], lw=1.4,
                    label=labels[prior])
        ax.set_xlabel(r"$\log_{10}(E_\nu/\mathrm{GeV})$")
        ax.set_ylabel(r"posterior density in $\log_{10}E$")
        ax.set_title("KM3-230213A")
        ax.legend(frameon=False, title="flux prior")
        fig.tight_layout()
        for suffix in (".pdf", ".png"):
            fig.savefig(out_path.with_suffix(suffix))
            print(f"Figure saved to: {out_path.with_suffix(suffix)}")
        plt.close(fig)


def main() -> None:
    """Reconstruct the event energy and draw the posterior."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = posteriors()
    report(results)
    make_figure(results, args.out_dir / "07_single_event_energy")


if __name__ == "__main__":
    main()
