"""Example 15 — parent-neutrino energy reconstruction for three UHE events.

Where example 13 shows the raw log-loss law ``P(w)`` for a single event, this
script turns it into what an analyst actually reports: the reconstructed
*parent-neutrino* energy of an observed muon track. Given a muon seen at energy
``E`` after propagating a column ``ell``, the parent neutrino had energy
``eps = E e^{w}`` with ``w = ln(eps / E) >= 0``. Folding the exact loss law
against a power-law flux prior ``phi ~ eps^{-1-gamma}`` gives the reconstructed
posterior over parent energy (per log-energy),

.. math:: p(\\ln\\varepsilon) \\;\\propto\\; e^{-\\gamma\\,w}\\,P(w),

and the same construction with the Fokker-Planck Gaussian in place of the exact
``P(w)`` gives the reconstruction the paper's drift-diffusion truncation implies.

The two agree near the peak, but they differ in the shape the steep flux prior
does not fully hide. The exact law (:mod:`softpaws.transport.loss_distribution`)
is both more concentrated just above ``E`` -- so its median parent energy sits a
little *below* the Gaussian's -- and heavier in the extreme upper tail, where the
subordinator keeps power that the Gaussian cuts off super-exponentially. The
crossover is out near the 99.9th percentile: there the exact reconstruction
allows the parent neutrino to have been a factor of several more energetic than
the Gaussian ever would. A reconstruction that assumes the Gaussian therefore
misjudges the rare, high-energy end -- exactly the regime a single UHE event
probes. This script puts the two posteriors side by side for three distinct
high-energy events spanning ~0.3 PeV to ~100 PeV, and prints the median, 90th-,
and 99.9th-percentile parent energy under each.

The three default events are illustrative track-like UHE detections (observed
muon energy and propagation column are approximate); override them by editing
``EVENTS`` or use ``--gamma`` to change the flux prior.

Usage
-----
    python examples/15_parent_energy_reconstruction.py
    python examples/15_parent_energy_reconstruction.py --gamma 2.0
    python examples/15_parent_energy_reconstruction.py --out-dir examples/output
"""

import argparse
import pathlib
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from softpaws.transport.coefficients import diffusion_coefficient, drift_coefficient
from softpaws.transport.loss_distribution import loss_density, loss_density_gaussian

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"

GAMMA_IC = 2.38  # IceCube 9.5 yr diffuse-flux best fit (Eq. 1.3), the flux prior

# Events are distinguished by color (from the style file's cycle), the two
# transport methods by line style.
_EVENT_COLORS = ("C0", "C1", "C2")
_EXACT_STYLE = {"ls": "-", "lw": 2.2}
_FP_STYLE = {"ls": "--", "lw": 2.0}

# Reconstructed parent energy runs from ~0.3 PeV to ~100 PeV across the events.
LOG10_EPS_LIM = (5.0, 10.0)


@dataclass(frozen=True)
class Event:
    """One observed muon track: name, muon energy [GeV], and column depth [km]."""

    name: str
    e_mu_gev: float
    ell_km: float


# Three distinct high-energy track-like detections. Observed muon energies and
# propagation columns are approximate, illustrative values.
EVENTS = (
    Event("IceCube-170922A", 2.9e5, 1.95),   # ~0.29 PeV, IceCube overburden
    Event("Glashow-like (6 PeV)", 6.05e6, 1.95),  # ~6 PeV, IceCube overburden
    Event("KM3-230213A", 1.2e8, 3.0),        # ~120 PeV, KM3NeT-like column
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=_DEFAULT_OUT_DIR,
        help="Directory for the output figure.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=GAMMA_IC,
        help="Spectral index of the power-law flux prior (default: 2.38).",
    )
    return parser.parse_args()


def reconstruct(event: Event, gamma: float) -> dict[str, np.ndarray]:
    """Reconstructed parent-energy posteriors (exact and FP) for one event.

    Returns the shared log-loss grid ``w``, the parent-energy grid ``eps`` [GeV],
    and the two posteriors ``p(ln eps) ~ e^{-gamma w} P(w)``, each normalized to
    unit area in ``w`` (equivalently in ``ln eps``).
    """
    b_mu = float(drift_coefficient(event.e_mu_gev)[0])
    d_mu = float(diffusion_coefficient(event.e_mu_gev)[0])

    mean = (b_mu + d_mu / 2.0) * event.ell_km
    w = np.linspace(1e-4, mean + 9.0 * np.sqrt(d_mu * event.ell_km), 2400)

    prior = np.exp(-gamma * w)  # power-law flux prior in log-parent-energy
    post_exact = prior * loss_density(w, event.ell_km, b_mu, d_mu)
    post_fp = prior * loss_density_gaussian(w, event.ell_km, b_mu, d_mu)
    post_exact /= np.trapezoid(post_exact, w)
    post_fp /= np.trapezoid(post_fp, w)

    return {
        "w": w,
        "eps": event.e_mu_gev * np.exp(w),
        "exact": post_exact,
        "fp": post_fp,
        "b_mu": b_mu,
        "d_mu": d_mu,
    }


def _quantile(w: np.ndarray, density: np.ndarray, q: float) -> float:
    """The ``q``-quantile of a density tabulated on the ascending grid ``w``."""
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(w))])
    cdf /= cdf[-1]
    return float(np.interp(q, cdf, w))


def print_table(event: Event, rec: dict[str, np.ndarray]) -> None:
    """Print reconstructed parent-energy quantiles, exact vs Fokker-Planck."""
    print(f"\n=== {event.name} ===  E_mu = {event.e_mu_gev:.3g} GeV, "
          f"ell = {event.ell_km:g} km  (b_mu = {rec['b_mu']:.3f}, d_mu = {rec['d_mu']:.4f})")
    print(f"{'quantile':>10} {'exact eps/E':>13} {'FP eps/E':>10} "
          f"{'exact eps [GeV]':>17} {'FP eps [GeV]':>15}")
    for q, tag in ((0.5, "median"), (0.9, "90th pct"), (0.999, "99.9th pct")):
        we = _quantile(rec["w"], rec["exact"], q)
        wf = _quantile(rec["w"], rec["fp"], q)
        eps_e = event.e_mu_gev * np.exp(we)
        eps_f = event.e_mu_gev * np.exp(wf)
        print(f"{tag:>10} {np.exp(we):13.2f} {np.exp(wf):10.2f} {eps_e:17.3g} {eps_f:15.3g}")


def make_figure(recs: list[dict[str, np.ndarray]], gamma: float, out_path: pathlib.Path) -> None:
    """All three events on one axes: reconstructed parent-energy posterior, exact vs FP."""
    jac = np.log(10.0)  # density in w to density in log10(eps): dw = ln(10) d log10(eps)

    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3, 3))

        for event, rec, color in zip(EVENTS, recs, _EVENT_COLORS):
            log10_eps = np.log10(rec["eps"])
            ax.plot(log10_eps, rec["exact"] * jac, color=color, **_EXACT_STYLE)
            ax.plot(log10_eps, rec["fp"] * jac, color=color, **_FP_STYLE)
            ax.axvline(np.log10(event.e_mu_gev), color=color, lw=0.8, ls=":")

        ax.set_xlim(*LOG10_EPS_LIM)
        ax.set_ylim(bottom=1e-3)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\log_{10}(\varepsilon_\nu\,/\,\mathrm{GeV})$")
        ax.set_ylabel("reconstructed posterior")
        ax.set_title(rf"parent-energy reconstruction ($\gamma = {gamma:g}$ flux prior)",
                     fontsize="small")

        event_handles = [
            Line2D([], [], color=c, lw=2.0, label=e.name)
            for e, c in zip(EVENTS, _EVENT_COLORS)
        ]
        method_handles = [
            Line2D([], [], color="0.3", label="exact (subordinator)", **_EXACT_STYLE),
            Line2D([], [], color="0.3", label="Fokker-Planck (Gaussian)", **_FP_STYLE),
        ]
        ax.legend(handles=event_handles + method_handles, fontsize="x-small", frameon=False)
        fig.tight_layout()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png"):
            path = out_path.with_suffix(suffix)
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {path.resolve()}")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    recs = [reconstruct(event, args.gamma) for event in EVENTS]
    for event, rec in zip(EVENTS, recs):
        print_table(event, rec)
    make_figure(recs, args.gamma, args.out_dir / "15_parent_energy_reconstruction")


if __name__ == "__main__":
    main()
