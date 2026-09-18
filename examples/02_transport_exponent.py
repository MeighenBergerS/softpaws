"""02_transport_exponent.py -- the transport exponent and what it replaces.

A muon crossing matter loses energy in many soft collisions and a few hard
ones. The transport exponent Phi(A) is the eigenvalue of that collision
operator on a power law of index A: it says how fast a power-law spectrum of
muons is attenuated, and the rest of the package is built on it.

This example evaluates Phi(A) from the shipped loss table at 1 PeV and sets it
against the two truncations in the literature, the drift-only A b_mu and the
second-order form. All three agree at A = 1 and 2 and part company elsewhere.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from softpaws.transport import diffusion_coefficient, drift_coefficient, third_moment_coefficient
from softpaws.transport.eigenvalue import phi_drift, phi_eigenvalue_three_moment, phi_fokker_planck

OUT = Path(__file__).parent / "output"
ENERGY_GEV = 1.0e6
A = np.linspace(0.05, 6.0, 200)  # spectral index of the muon spectrum


def main():
    # The first three moments of the fractional loss per km of water.
    b, d, t = (f(ENERGY_GEV).item() for f in
               (drift_coefficient, diffusion_coefficient, third_moment_coefficient))
    print(f"Loss moments at {ENERGY_GEV:.0e} GeV: b = {b:.3f}, d = {d:.3f}, t = {t:.3f} /km")

    phi = phi_eigenvalue_three_moment(A, b, d, t)
    print(f"Phi(1) = {phi_eigenvalue_three_moment(1.0, b, d, t):.4f} = b_mu")
    print(f"Phi(2) = {phi_eigenvalue_three_moment(2.0, b, d, t):.4f} = 2 b_mu - d_mu "
          f"= {2 * b - d:.4f}")

    fig, ax = plt.subplots()
    ax.plot(A, phi, label=r"$\Phi(A)$")
    ax.plot(A, phi_drift(A, b), "--", label=r"$A\,b_\mu$")
    ax.plot(A, phi_fokker_planck(A, b, d), ":", label=r"second order")
    ax.set(xlabel="Spectral index $A$", ylabel=r"$\Phi$ [km$^{-1}$]", ylim=(0.0, 2.5))
    ax.legend()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "02_transport_exponent.png", dpi=150)


if __name__ == "__main__":
    main()
