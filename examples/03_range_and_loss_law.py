"""Tutorial 03 -- how far a muon goes, and how much it loses on the way.

Two questions follow from the transport exponent. How far does a muon of a
given energy travel before it falls below a threshold, and what is the
distribution of the energy it has left after a fixed distance?

The first is a first-passage problem and has a closed form. The second is the
loss law itself, and this tutorial shows why its tail matters: the Gaussian
approximation is wrong by orders of magnitude a factor of a few out, which is
exactly where a bright track lives.

Usage
-----
    python examples/03_range_and_loss_law.py
"""

import argparse
import pathlib

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

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"

#: Muon energies the range table runs over [log10 GeV].
LOG10_E = np.array([4.0, 5.0, 6.0, 7.0, 8.0])

#: Log losses the survival table is printed at, ``w = ln(E_produced / E_seen)``.
W_TABLE = (0.5, 1.0, 1.5, 2.0, 3.0)

#: Grid the loss density is built on.
W_GRID = np.linspace(-1.0, 8.0, 3001)


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threshold-gev", type=float, default=1.0e3,
                        help="Muon energy the range runs down to.")
    parser.add_argument("--energy-gev", type=float, default=1.0e6,
                        help="Muon energy of the loss law.")
    parser.add_argument("--distance-km", type=float, default=1.0,
                        help="Distance the loss law is evaluated after [km of water].")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the figure.")
    return parser.parse_args()


def report_range(threshold_gev: float) -> None:
    """Print the first-passage range against the continuous-slowing-down range."""
    energy = 10.0**LOG10_E
    csda = np.atleast_1d(muon_range_km(energy, threshold_gev))
    first_passage = np.atleast_1d(stochastic_muon_range_km(energy, threshold_gev))
    frozen = np.atleast_1d(
        stochastic_muon_range_km(energy, threshold_gev, kernel_evaluation="frozen")
    )
    print(f"Range of a muon in water down to {threshold_gev:.0f} GeV, in km:\n")
    print(f"{'log10(E/GeV)':>13} {'first passage':>14} {'CSDA':>8} {'ratio':>7} {'frozen':>8}")
    for i, log10_e in enumerate(LOG10_E):
        print(f"{log10_e:13.1f} {first_passage[i]:14.2f} {csda[i]:8.2f} "
              f"{first_passage[i] / csda[i]:7.3f} {frozen[i]:8.2f}")
    print("\n  The first-passage range is the shorter one: a muon that happens to")
    print("  radiate hard early never reaches the mean-loss distance. Freezing the")
    print("  kernel at the production energy shortens it further, by more the")
    print("  longer the lever arm.")


def report_loss_law(energy_gev: float, distance_km: float) -> dict[str, np.ndarray]:
    """Print and return the survival above a log loss, for three loss families."""
    b = float(np.squeeze(drift_coefficient(energy_gev)))
    d = float(np.squeeze(diffusion_coefficient(energy_gev)))
    t = float(np.squeeze(third_moment_coefficient(energy_gev)))
    families = {
        "three moment": loss_density_three_moment(W_GRID, distance_km, b, d, t),
        "two moment": loss_density(W_GRID, distance_km, b, d),
        "Gaussian": loss_density_gaussian(W_GRID, distance_km, b, d),
    }
    print(f"\nProbability that a {energy_gev:.0e} GeV muon has lost more than a factor")
    print(f"exp(w) of its energy over {distance_km:g} km of water:\n")
    header = f"{'w':>5} {'factor':>8}" + "".join(f"{name:>15}" for name in families)
    print(header)
    for w in W_TABLE:
        row = f"{w:5.1f} {np.exp(w):8.1f}"
        for density in families.values():
            row += f"{float(survival_from_density(w, W_GRID, density)):15.2e}"
        print(row)
    print("\n  The Gaussian form collapses in the tail. The two-moment family is")
    print("  closer but still low, and only the three-moment one tracks the hard")
    print("  end of the kernel, which is the part a bright track samples.")
    print("  The last Gaussian entry sits at the numerical floor of this grid;")
    print("  the true value there is about 1e-18.")
    return families


def make_figure(families: dict[str, np.ndarray], out_path: pathlib.Path) -> None:
    """Draw the three loss densities on the log-loss axis."""
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        for (name, density), style in zip(families.items(), ("-", "--", ":")):
            ax.plot(W_GRID, np.clip(density, 1e-12, None), ls=style, lw=1.3, label=name)
        ax.set_yscale("log")
        ax.set_xlim(0.0, 5.0)
        ax.set_ylim(1e-4, 5.0)
        ax.set_xlabel(r"log loss $w = \ln(\varepsilon/E)$")
        ax.set_ylabel(r"$P(w)$")
        ax.set_title("the loss law after 1 km of water")
        ax.legend(frameon=False)
        fig.tight_layout()
        for suffix in (".pdf", ".png"):
            fig.savefig(out_path.with_suffix(suffix))
            print(f"Figure saved to: {out_path.with_suffix(suffix)}")
        plt.close(fig)


def main() -> None:
    """Report the range and the loss law, then draw the loss law."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_range(args.threshold_gev)
    families = report_loss_law(args.energy_gev, args.distance_km)
    make_figure(families, args.out_dir / "t03_range_and_loss_law")


if __name__ == "__main__":
    main()
