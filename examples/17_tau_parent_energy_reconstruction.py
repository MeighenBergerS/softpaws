"""Example 17 -- parent-energy reconstruction for a nu_tau-origin muon track.

The nu_tau counterpart of ``15_parent_energy_reconstruction.py``. There, an
observed muon track is assumed to come directly from nu_mu CC production, and
the exact subordinator (:mod:`softpaws.transport.loss_distribution`) versus
Fokker-Planck posteriors over the parent energy are compared. Here the same
observed muon is instead hypothesized to have come from the chain nu_tau -> CC
tau -> leptonic decay -> ordinary muon transport
(:mod:`softpaws.transport.tau`), and the reconstructed posterior over the
*parent tau* (~ parent nu_tau) energy is built from the closed-form composite
law derived there, :func:`~softpaws.transport.tau.tau_loss_density`.

The interesting physics is the tau decay-length truncation: unlike the direct
nu_mu case, the transport parameters entering the tau-origin likelihood
(``ell_tau(E_tau)``) depend on the *hypothesis itself*, not just on the
observed energy. A tau energetic enough that its mean decay length exceeds the
assumed total column ``ell_tau_total_km`` (from the nu_tau interaction point to
the detector) is unlikely to have decayed into a muon at all -- so the
tau-origin posterior is suppressed at the high-energy end relative to what a
naive reuse of the muon-only machinery would give, exactly where a
KM3-230213A-like reconstruction would otherwise be tempted to invoke an
extreme parent energy.

``ell_tau_total_km`` is a new, necessarily larger, illustrative geometry input
per event than ``15``'s muon-only ``ell_km``: it must contain room for both the
tau's own flight and the secondary muon's subsequent propagation. As in ``15``,
these are illustrative, not measured, values -- see the ``EVENTS`` table below.
Same three events as example 15 and 08's IceCube spectral index for the flux
prior.

Usage
-----
    python examples/17_tau_parent_energy_reconstruction.py
    python examples/17_tau_parent_energy_reconstruction.py --gamma 2.0
    python examples/17_tau_parent_energy_reconstruction.py --out-dir examples/output
"""

import argparse
import pathlib
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from softpaws.transport.coefficients import diffusion_coefficient, drift_coefficient
from softpaws.transport.loss_distribution import loss_density, loss_density_gaussian
from softpaws.transport.tau import tau_loss_density, tau_survival_before_decay

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"

GAMMA_IC = 2.38  # IceCube 9.5 yr diffuse-flux best fit (Eq. 1.3), the flux prior

_EVENT_COLORS = ("C0", "C1", "C2")
_EXACT_STYLE = {"ls": "-", "lw": 2.2}
_FP_STYLE = {"ls": "--", "lw": 2.0}
_TAU_STYLE = {"ls": ":", "lw": 2.2}

LOG10_EPS_LIM = (5.0, 10.0)


@dataclass(frozen=True)
class Event:
    """One observed muon track: name, muon energy [GeV], and two columns.

    ``ell_km`` is the muon-only propagation column (as in example 15).
    ``ell_tau_total_km`` is the illustrative *total* column from the assumed
    nu_tau interaction point to the detector, for the tau-origin hypothesis --
    necessarily larger than ``ell_km`` since it must also contain the tau's own
    flight before decay.
    """

    name: str
    e_mu_gev: float
    ell_km: float
    ell_tau_total_km: float


# Illustrative geometries: down-going IceCube-like events get a modest total
# column (a down-going nu_tau interaction has, at most, a few km of ice plus
# some atmosphere above it); the near-horizontal KM3-230213A gets a much larger
# one, since a near-horizontal path through Earth's crust is exactly the
# geometry that maximizes the available column for the tau to decay upstream
# (see softpaws.transport.tau).
EVENTS = (
    Event("IceCube-170922A", 2.9e5, 1.95, 10.0),
    Event("Glashow-like (6 PeV)", 6.05e6, 1.95, 15.0),
    Event("KM3-230213A", 1.2e8, 3.0, 50.0),
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
    """Reconstructed parent-energy posteriors: muon-origin (exact, FP) and tau-origin.

    Shares one ``w`` grid across all three so they are directly comparable. The
    grid is sized to comfortably cover both the muon-only spread (as in
    example 15) and the plotted eps window, plus headroom for the tau-origin
    posterior's own extra spread from the decay kinematics.
    """
    b_mu = float(drift_coefficient(event.e_mu_gev)[0])
    d_mu = float(diffusion_coefficient(event.e_mu_gev)[0])

    mean = (b_mu + d_mu / 2.0) * event.ell_km
    w_max_muon = mean + 9.0 * np.sqrt(d_mu * event.ell_km)
    w_max_plot = np.log(10.0 ** LOG10_EPS_LIM[1] / event.e_mu_gev)
    w_max = max(w_max_muon, w_max_plot) + 5.0
    w = np.linspace(1e-4, w_max, 3000)

    prior = np.exp(-gamma * w)  # power-law flux prior in log-parent-energy

    post_exact = prior * loss_density(w, event.ell_km, b_mu, d_mu)
    post_fp = prior * loss_density_gaussian(w, event.ell_km, b_mu, d_mu)
    post_exact /= np.trapezoid(post_exact, w)
    post_fp /= np.trapezoid(post_fp, w)

    e_tau_diag = event.e_mu_gev * np.exp(w)
    tau_density = tau_loss_density(w, event.ell_tau_total_km, e_tau_diag, b_mu, d_mu)
    tau_mass = np.trapezoid(prior * tau_density, w)
    post_tau = (prior * tau_density) / tau_mass

    return {
        "w": w,
        "eps": event.e_mu_gev * np.exp(w),
        "exact": post_exact,
        "fp": post_fp,
        "tau": post_tau,
        "tau_mass_fraction": tau_mass,  # unnormalized mass before dividing above
        "b_mu": b_mu,
        "d_mu": d_mu,
    }


def _quantile(w: np.ndarray, density: np.ndarray, q: float) -> float:
    """The ``q``-quantile of a density tabulated on the ascending grid ``w``."""
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(w))])
    cdf /= cdf[-1]
    return float(np.interp(q, cdf, w))


def print_table(event: Event, rec: dict[str, np.ndarray]) -> None:
    """Print reconstructed parent-energy quantiles: muon-origin vs. tau-origin."""
    print(
        f"\n=== {event.name} ===  E_mu = {event.e_mu_gev:.3g} GeV, "
        f"ell_mu = {event.ell_km:g} km, ell_tau_total = {event.ell_tau_total_km:g} km "
        f"(b_mu = {rec['b_mu']:.3f}, d_mu = {rec['d_mu']:.4f})"
    )
    e_tau_median = event.e_mu_gev * np.exp(_quantile(rec["w"], rec["tau"], 0.5))
    survival = tau_survival_before_decay(event.ell_tau_total_km, e_tau_median)
    print(
        f"  at the tau-origin median parent energy ({e_tau_median:.3g} GeV), "
        f"P(tau decays before ell_tau_total) = {survival:.3f}"
    )
    print(
        f"{'quantile':>10} {'exact eps [GeV]':>17} {'FP eps [GeV]':>15} "
        f"{'tau eps [GeV]':>15}"
    )
    for q, tag in ((0.5, "median"), (0.9, "90th pct"), (0.999, "99.9th pct")):
        eps_exact = event.e_mu_gev * np.exp(_quantile(rec["w"], rec["exact"], q))
        eps_fp = event.e_mu_gev * np.exp(_quantile(rec["w"], rec["fp"], q))
        eps_tau = event.e_mu_gev * np.exp(_quantile(rec["w"], rec["tau"], q))
        print(f"{tag:>10} {eps_exact:17.3g} {eps_fp:15.3g} {eps_tau:15.3g}")


def make_figure(recs: list[dict[str, np.ndarray]], gamma: float, out_path: pathlib.Path) -> None:
    """All three events on one axes: muon-origin (exact, FP) vs. tau-origin posterior."""
    jac = np.log(10.0)  # density in w to density in log10(eps): dw = ln(10) d log10(eps)

    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.4, 3.2))

        for event, rec, color in zip(EVENTS, recs, _EVENT_COLORS):
            log10_eps = np.log10(rec["eps"])
            ax.plot(log10_eps, rec["exact"] * jac, color=color, **_EXACT_STYLE)
            ax.plot(log10_eps, rec["fp"] * jac, color=color, **_FP_STYLE)
            ax.plot(log10_eps, rec["tau"] * jac, color=color, **_TAU_STYLE)
            ax.axvline(np.log10(event.e_mu_gev), color=color, lw=0.8, ls=":")

        ax.set_xlim(*LOG10_EPS_LIM)
        ax.set_ylim(bottom=1e-3)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\log_{10}(\varepsilon_\nu\,/\,\mathrm{GeV})$")
        ax.set_ylabel("reconstructed posterior")
        ax.set_title(
            rf"parent-energy reconstruction ($\gamma = {gamma:g}$ flux prior)",
            fontsize="small",
        )

        event_handles = [
            Line2D([], [], color=c, lw=2.0, label=e.name)
            for e, c in zip(EVENTS, _EVENT_COLORS)
        ]
        method_handles = [
            Line2D([], [], color="0.3", label="muon-origin, exact", **_EXACT_STYLE),
            Line2D([], [], color="0.3", label="muon-origin, Fokker-Planck", **_FP_STYLE),
            Line2D([], [], color="0.3", label="tau-origin (exact)", **_TAU_STYLE),
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
    make_figure(recs, args.gamma, args.out_dir / "17_tau_parent_energy_reconstruction")


if __name__ == "__main__":
    main()
