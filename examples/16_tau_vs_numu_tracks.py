"""Example 16 -- through-going muon tracks: true nu_mu vs. true nu_tau.

Compares two independent ways of making an observed muon track for the *same*
diffuse flux normalization ``(phi0, gamma)``: direct ``nu_mu`` charged-current
production (:class:`~softpaws.response.soft_volume.SoftVolumeResponse`), and
``nu_tau`` CC production of a tau that decays leptonically to a muon
(:mod:`softpaws.transport.tau`). Structured like ``08_exact_vs_fokker_planck.py``
-- two figures, each with a ratio panel sharing the x-axis -- but comparing
flavors instead of transport methods.

1. the tau/numu ratio ``N_tau_to_mu / N_numu`` vs muon energy, split into its
   two pieces: the flat branching-ratio-times-decay-moment prefactor, and the
   full ratio including the energy-growing ``ell_tau(q E_mu) Phi(A)`` bracket;
2. expected through-going track counts per energy bin, nu_mu-induced vs.
   nu_tau-induced (same flux, same exact soft-volume transport), as in
   ``06_soft_volume_event_rate.py``.

Both use the exact eigenvalue path (``method="exact"``), since the tau ratio
is built from the exact ``Phi(A)``. The point of the comparison: the tau
contribution is a percent-level correction at the IceCube diffuse-flux energies
(~1 PeV) but grows to a few-tens-of-percent effect by ~100 PeV, because the tau
decay length grows linearly with energy while the muon range does not (see
``softpaws.transport.tau`` for the derivation).

Usage
-----
    python examples/16_tau_vs_numu_tracks.py
    python examples/16_tau_vs_numu_tracks.py --column-depth-km 1.95
    python examples/16_tau_vs_numu_tracks.py --out-dir examples/output
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.response.soft_volume import (
    SoftVolumeResponse,
    tau_induced_expected_counts,
)
from softpaws.transport.eigenvalue import spectral_index
from softpaws.transport.soft_volume import sphere_radius_from_volume
from softpaws.transport.tau import BR_TAU_TO_MU, tau_to_muon_ratio, z_moment

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"

GAMMA_IC = 2.38  # IceCube 9.5 yr diffuse-flux best fit (Eq. 1.3)
PHI0_IC = 0.63  # best-fit flux normalization [1e-18 GeV^-1 cm^-2 s^-1 sr^-1]
LIVETIME_S = 9.5 * 365.25 * 86400.0  # IceCube 9.5 yr exposure
SOLID_ANGLE_SR = 2.0 * np.pi  # one hemisphere

RADIUS_KM = sphere_radius_from_volume(1.0)  # ~0.62 km, IceCube-like

# Ratio panel (as in example 05/08): continuous energy grid.
LOG10_E = np.linspace(5.0, 9.0, 81)  # 100 TeV to 1 EeV, in log10(E / GeV)

# Event-rate panel (as in example 06/08): half-decade bins.
LOG10_E_EDGES = np.arange(4.0, 9.01, 0.5)  # 10 TeV to 1 EeV

_FLAVOR_STYLE = {"nu_mu": {"ls": "-"}, "nu_tau": {"ls": "--"}}
_FLAVOR_HANDLES = [
    plt.Line2D([], [], color="k", ls="-", label=r"$\nu_\mu$ (direct)"),
    plt.Line2D([], [], color="k", ls="--", label=r"$\nu_\tau$ (via $\tau \to \mu$)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=_DEFAULT_OUT_DIR,
        help="Directory for the output figures.",
    )
    parser.add_argument(
        "--column-depth-km",
        type=float,
        default=1.95,
        help="Upstream column depth for the exact path [km] (default: IceCube-like "
        "down-going overburden).",
    )
    return parser.parse_args()


def _new_ratio_figure() -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """A stacked main/ratio figure pair sharing the x-axis."""
    fig, (ax, ax_ratio) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(3.2, 3.8),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )
    return fig, ax, ax_ratio


def _save(fig: plt.Figure, out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_path.with_suffix(suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {path.resolve()}")
    plt.close(fig)


def make_ratio_figure(out_path: pathlib.Path) -> None:
    """Tau/numu ratio vs energy: full ratio vs. its flat prefactor."""
    energy_gev = 10.0**LOG10_E
    a = spectral_index(GAMMA_IC)
    prefactor = BR_TAU_TO_MU * z_moment(a)  # the "1" (in-detector) piece only
    full_ratio = tau_to_muon_ratio(a, energy_gev)
    bracket = full_ratio / prefactor  # 1 + ell_tau(q E_mu) Phi(A)

    with plt.style.context(str(_STYLE)):
        fig, ax, ax_ratio = _new_ratio_figure()

        ax.axhline(prefactor, color="0.5", ls=":", lw=1.0, label="in-detector only")
        ax.plot(LOG10_E, full_ratio, color="C3", label="full (incl. soft term)")
        ax.set_yscale("log")
        ax.set_ylabel(r"$N_{\tau\to\mu}\,/\,N_{\nu_\mu}$")
        ax.legend(fontsize="small")

        ax_ratio.plot(LOG10_E, bracket, color="C3")
        ax_ratio.axhline(1.0, color="0.6", lw=0.8, ls=":")
        ax_ratio.set_ylabel(r"$1+\ell_\tau\Phi(A)$")
        ax_ratio.set_xlabel(r"$\log_{10}(E_\mu\,/\,\mathrm{GeV})$")

        _save(fig, out_path)


def make_event_rate_figure(out_path: pathlib.Path, column_depth_km: float) -> None:
    """Expected track counts per bin: nu_mu-induced vs. nu_tau-induced."""
    response = SoftVolumeResponse(RADIUS_KM, method="exact", column_depth_km=column_depth_km)

    counts_numu = response.expected_counts(
        LOG10_E_EDGES, PHI0_IC, GAMMA_IC, LIVETIME_S, SOLID_ANGLE_SR
    )
    counts_tau = tau_induced_expected_counts(
        response, LOG10_E_EDGES, PHI0_IC, GAMMA_IC, LIVETIME_S, SOLID_ANGLE_SR
    )

    with plt.style.context(str(_STYLE)):
        fig, ax, ax_ratio = _new_ratio_figure()

        ax.stairs(
            counts_numu, LOG10_E_EDGES, label=r"$\nu_\mu$ (direct)",
            color="C0", **_FLAVOR_STYLE["nu_mu"],
        )
        ax.stairs(
            counts_tau, LOG10_E_EDGES, label=r"$\nu_\tau$ (via $\tau\to\mu$)",
            color="C1", **_FLAVOR_STYLE["nu_tau"],
        )
        ax.set_yscale("log")
        ax.set_ylabel("expected tracks / bin (9.5 yr)")
        ax.legend(fontsize="small")

        ax_ratio.stairs(counts_tau / counts_numu, LOG10_E_EDGES, color="C1")
        ax_ratio.set_ylabel(r"$\nu_\tau\,/\,\nu_\mu$")
        ax_ratio.set_xlabel(r"$\log_{10}(E_\mu\,/\,\mathrm{GeV})$")

        _save(fig, out_path)

    print("Expected tracks per bin (9.5 yr):")
    print(f"{'log10(E/GeV)':>14} {'nu_mu':>12} {'nu_tau':>12} {'tau/numu':>10}")
    for lo, hi, n_mu, n_tau in zip(
        LOG10_E_EDGES[:-1], LOG10_E_EDGES[1:], counts_numu, counts_tau
    ):
        print(f"{lo:6.1f}-{hi:<6.1f} {n_mu:12.3g} {n_tau:12.3g} {n_tau / n_mu:10.3f}")


def main() -> None:
    args = parse_args()
    print(f"Exact path: column depth {args.column_depth_km:g} km")

    make_ratio_figure(args.out_dir / "16_tau_vs_numu_ratio")
    make_event_rate_figure(args.out_dir / "16_tau_vs_numu_counts", args.column_depth_km)


if __name__ == "__main__":
    main()
