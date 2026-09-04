"""Tutorial 02 -- the transport exponent and what it replaces.

A muon crossing matter loses energy in a mixture of many soft collisions and
a few hard ones. The transport exponent ``Phi(A)`` is the eigenvalue of that
collision operator on a power law of index ``A``: it says how fast a
power-law muon spectrum is attenuated, and it is the one quantity the rest
of the package is built on.

This tutorial evaluates ``Phi(A)`` from the shipped loss table and sets it
against the two truncations the literature uses, the drift-only ``A b_mu``
and the second-order Fokker-Planck form. They agree where the expansion is
valid and part company where it is not.

Usage
-----
    python examples/02_transport_exponent.py
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.transport import (
    diffusion_coefficient,
    drift_coefficient,
    log_loss_moments,
    third_moment_coefficient,
)
from softpaws.transport.eigenvalue import (
    phi_drift,
    phi_eigenvalue_three_moment,
    phi_fokker_planck,
)

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"

#: Spectral indices the exponent is drawn against.
A_GRID = np.linspace(0.05, 6.0, 200)

#: Indices the table prints, including the two exactness points.
A_TABLE = (0.14, 0.50, 0.98, 1.00, 2.00, 3.00, 4.00, 6.00)

#: Muon energies compared [GeV].
ENERGIES_GEV = (1.0e4, 1.0e6, 1.0e8)


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--energy-gev", type=float, default=1.0e6,
                        help="Muon energy the kernel is read at.")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the figure.")
    return parser.parse_args()


def kernel(energy_gev: float) -> tuple[float, float, float]:
    """The three loss moments of the kernel at one energy [km^-1]."""
    return tuple(
        float(np.squeeze(f(energy_gev)))
        for f in (drift_coefficient, diffusion_coefficient, third_moment_coefficient)
    )


def report(energy_gev: float) -> None:
    """Print the kernel, the exponent and the two truncations."""
    print("The kernel of water, per kilometre:\n")
    print(f"{'log10(E/GeV)':>13} {'b_mu':>8} {'d_mu':>8} {'t_mu':>8} {'first log':>10}")
    for e in ENERGIES_GEV:
        b, d, t = kernel(e)
        phi_prime = float(np.squeeze(log_loss_moments(e)[0]))
        print(f"{np.log10(e):13.1f} {b:8.3f} {d:8.3f} {t:8.3f} {phi_prime:9.3f}")

    b, d, t = kernel(energy_gev)
    print(f"\nThe exponent at {energy_gev:.3g} GeV, against its truncations:\n")
    print(f"{'A':>6} {'Phi(A)':>9} {'A b_mu':>9} {'Phi_2(A)':>9} {'Phi_2/Phi':>10}")
    for a in A_TABLE:
        phi = float(phi_eigenvalue_three_moment(a, b, d, t))
        drift = float(phi_drift(a, b))
        second = float(phi_fokker_planck(a, b, d))
        print(f"{a:6.2f} {phi:9.3f} {drift:9.3f} {second:9.3f} {second / phi:10.3f}")

    print("\n  Phi(1) = b_mu exactly, and Phi(2) = 2 b_mu - d_mu exactly:")
    print(f"    Phi(1) = {float(phi_eigenvalue_three_moment(1.0, b, d, t)):.6f}, "
          f"b_mu = {b:.6f}")
    print(f"    Phi(2) = {float(phi_eigenvalue_three_moment(2.0, b, d, t)):.6f}, "
          f"2 b_mu - d_mu = {2 * b - d:.6f}")
    print("\n  The drift form rises without bound; the second-order form turns over")
    print("  and falls, which is why neither can be trusted far from A = 1.")


def make_figure(energy_gev: float, out_path: pathlib.Path) -> None:
    """Draw the exponent and its two truncations against the spectral index."""
    b, d, t = kernel(energy_gev)
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        ax.plot(A_GRID, phi_eigenvalue_three_moment(A_GRID, b, d, t), lw=1.6,
                label=r"$\Phi(A)$")
        ax.plot(A_GRID, phi_drift(A_GRID, b), lw=1.0, ls="--", label=r"$A\,b_\mu$")
        ax.plot(A_GRID, phi_fokker_planck(A_GRID, b, d), lw=1.0, ls=":",
                label=r"$\Phi_2(A)$")
        ax.axvline(1.0, color="0.8", lw=0.6, zorder=0)
        ax.axvline(2.0, color="0.8", lw=0.6, zorder=0)
        ax.set_xlabel(r"spectral index $A$")
        ax.set_ylabel(r"$\Phi$ [km$^{-1}$]")
        exponent = int(round(np.log10(energy_gev)))
        ax.set_title(rf"water at $10^{{{exponent}}}$ GeV")
        ax.set_ylim(0.0, 2.5)
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        for suffix in (".pdf", ".png"):
            fig.savefig(out_path.with_suffix(suffix))
            print(f"Figure saved to: {out_path.with_suffix(suffix)}")
        plt.close(fig)


def main() -> None:
    """Report the exponent and draw it."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report(args.energy_gev)
    make_figure(args.energy_gev, args.out_dir / "t02_transport_exponent")


if __name__ == "__main__":
    main()
