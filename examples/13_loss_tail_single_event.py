"""Example 13 — parent-energy loss tail for a single UHE event.

The distribution-level companion to example 11 (cross section from one event).
Examples 05-08 and 12 only ever need the eigenvalue ``Phi(A)``, because a
power-law flux averages over the muon log-loss ``w = ln(eps / E)``. A *single*
ultra-high-energy track (KM3-230213A) instead samples the full law ``P(w)``
once, and its heavy tail -- the chance the parent neutrino was far more energetic
than the observed muon -- is exactly what the paper's Fokker-Planck Gaussian
throws away.

This script inverts the exact subordinator characteristic function
(:func:`softpaws.transport.loss_distribution.loss_density`) and puts ``P(w)``
next to the Fokker-Planck Gaussian, on a linear and a log-y panel over the
parent/observed energy ratio ``eps / E = e^w``. It then prints the tail
probabilities ``P(W > w)`` at fixed ``eps / E`` ratios, with the exact/FP ratio
column -- the headline number: the Gaussian under-predicts the tail by one to
three orders of magnitude, so a single-event cross-section extraction that
assumes it (as the paper's does) is biased.

Geometry and energy default to a KM3NeT-like column ``ell = 3 km`` at the
100 PeV Table 1 coefficients; both are overridable.

Usage
-----
    python examples/13_loss_tail_single_event.py
    python examples/13_loss_tail_single_event.py --ell-km 3 --energy-gev 1e8
    python examples/13_loss_tail_single_event.py --out-dir examples/output
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from softpaws.transport.coefficients import diffusion_coefficient, drift_coefficient
from softpaws.transport.loss_distribution import (
    gaussian_survival,
    loss_density,
    loss_density_gaussian,
    survival_from_density,
)

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"

# Parent/observed energy ratios eps / E at which to tabulate the tail.
_RATIOS = (3.0, 5.0, 10.0, 30.0, 100.0)

_EXACT_STYLE = {"color": "crimson", "lw": 2.2, "label": "exact (subordinator)"}
_FP_STYLE = {"color": "navy", "lw": 2.0, "ls": "--", "label": "Fokker-Planck (Gaussian)"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=_DEFAULT_OUT_DIR,
        help="Directory for the output figure.",
    )
    parser.add_argument(
        "--ell-km",
        type=float,
        default=3.0,
        help="Propagated column depth ell [km] (default: KM3NeT-like 3 km).",
    )
    parser.add_argument(
        "--energy-gev",
        type=float,
        default=1.0e8,
        help="Observed muon energy [GeV] setting the Table 1 coefficients.",
    )
    return parser.parse_args()


def _save(fig: plt.Figure, out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_path.with_suffix(suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {path.resolve()}")
    plt.close(fig)


def _add_ratio_axis(ax: plt.Axes) -> None:
    """Add a secondary top axis in the parent/observed energy ratio eps / E."""
    sec = ax.secondary_xaxis(
        "top",
        functions=(
            lambda w: np.exp(np.clip(w, -50.0, 50.0)),
            lambda ratio: np.log(np.clip(ratio, 1e-9, None)),
        ),
    )
    sec.set_xlabel(r"parent / observed energy ratio  $\varepsilon / E$")


def make_figure(out_path: pathlib.Path, ell_km: float, b_mu: float, d_mu: float) -> None:
    """Two panels of ``P(w)``, exact vs Fokker-Planck, linear and log-y."""
    mean = (b_mu + d_mu / 2.0) * ell_km
    w_grid = np.linspace(1e-3, mean + 7.0 * np.sqrt(d_mu * ell_km), 1600)
    exact = loss_density(w_grid, ell_km, b_mu, d_mu)
    gauss = loss_density_gaussian(w_grid, ell_km, b_mu, d_mu)

    with plt.style.context(str(_STYLE)):
        fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(6.6, 3.0))
        for ax, logy in ((ax_lin, False), (ax_log, True)):
            ax.plot(w_grid, exact, **_EXACT_STYLE)
            ax.plot(w_grid, gauss, **_FP_STYLE)
            ax.set_xlabel(r"log-loss  $w = \ln(\varepsilon / E)$")
            ax.set_ylabel("probability density")
            if logy:
                ax.set_yscale("log")
                ax.set_ylim(1e-6, 2.0)
            _add_ratio_axis(ax)

        ax_lin.legend(
            handles=[Line2D([], [], **_EXACT_STYLE), Line2D([], [], **_FP_STYLE)],
            fontsize="small",
            frameon=False,
        )
        fig.tight_layout()
        _save(fig, out_path)


def print_tail_table(ell_km: float, b_mu: float, d_mu: float) -> None:
    """Print exact vs Fokker-Planck tail probabilities at fixed ``eps / E``."""
    mean = (b_mu + d_mu / 2.0) * ell_km
    w_grid = np.linspace(0.0, mean + 9.0 * np.sqrt(d_mu * ell_km), 3000)
    density = loss_density(w_grid, ell_km, b_mu, d_mu)
    exact_mean = np.trapezoid(w_grid * density, w_grid)

    print(f"\n=== ell = {ell_km:g} km ===  exact mean w = {exact_mean:.3f}   FP mean = {mean:.3f}")
    print(f"{'eps/E':>7} {'w=lnF':>7} {'P_exact':>11} {'P_FP':>11} {'ratio':>9}")
    for ratio in _RATIOS:
        w = np.log(ratio)
        p_exact = float(survival_from_density(w, w_grid, density))
        p_fp = float(gaussian_survival(w, ell_km, b_mu, d_mu))
        print(f"{ratio:7g} {w:7.2f} {p_exact:11.2e} {p_fp:11.2e} {p_exact / p_fp:9.1f}")


def main() -> None:
    args = parse_args()
    b_mu = float(drift_coefficient(args.energy_gev)[0])
    d_mu = float(diffusion_coefficient(args.energy_gev)[0])
    print_tail_table(args.ell_km, b_mu, d_mu)
    make_figure(args.out_dir / "13_loss_tail_single_event", args.ell_km, b_mu, d_mu)


if __name__ == "__main__":
    main()
