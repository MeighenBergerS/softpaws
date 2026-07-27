"""Example 21 -- full IceCube comparison: Earth attenuation plus nu_tau.

Builds on ``14_earth_attenuation.py`` by adding the nu_tau contribution to its
per-event PREM Earth attenuation (Form B, the most physical of the three
variants compared there): the tau-induced muon-track rate from a
same-normalization ``nu_tau`` flux (:mod:`softpaws.transport.tau`,
``16_tau_vs_numu_tracks.py``), via the new
:func:`~softpaws.response.soft_volume.tau_induced_expected_counts_attenuated`.

Two soft-volume curves are shown, each independently fit to the observed data
above 1 PeV: ``numu`` (example 14's Form B baseline) and ``numu + nu_tau``
(the fuller prediction).

Note on energy smearing
------------------------
An IceCube energy-smearing migration (via the new
:func:`~softpaws.comparison.rates.soft_volume_smeared_counts`) was tried here
and deliberately left out. The published smearing table maps *true injected
neutrino energy* to reconstructed energy, and that migration is enormous --
for a 1.8 PeV true neutrino the upgoing table's mean reconstructed energy is
over two decades lower, since the published effective area extends to
interactions produced far outside the fiducial volume, where most of the
muon's energy is lost in transit. But the soft-volume model's energy variable
is already the *observed muon energy at the detector* (i.e. after its own
transport losses, Eq. 2.23). Feeding that into the neutrino-indexed smearing
table double-applies the muon-range/energy-loss physics -- once inside the
transport model, again via the smearing table's built-in muon-range effect --
which inflated the required flux normalization by ~90x in a quick test,
against ~4x for the IRF path and the unsmeared soft-volume path. A correct
treatment would need the detector-resolution component of the IceCube
response decoupled from the muon-range migration, which the public DR2
release does not provide separately. ``soft_volume_smeared_counts`` is kept in
:mod:`softpaws.comparison.rates` (tested) for whenever that decomposition
becomes available.

Caveat carried from example 14: this is a bare astrophysical-flux template
with no atmospheric-neutrino background, fit only above 1 PeV where the
upgoing sample is expected to be signal-dominated.

Usage
-----
    python examples/21_full_icecube_comparison.py
    python examples/21_full_icecube_comparison.py --data-dir /path/to/dataverse_files
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.comparison.rates import (
    fit_scale_factor,
    implied_efficiency,
    irf_expected_counts,
    observed_counts,
)
from softpaws.data.container import EventSet
from softpaws.data.loader import compute_livetime_s, load_all_seasons, load_irfs, load_uptime
from softpaws.response.soft_volume import (
    SoftVolumeResponse,
    power_law_flux,
    tau_induced_expected_counts_attenuated,
)

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_DATA_DIR = _HERE.parent / "src" / "softpaws" / "data" / "dataverse_files"
_DEFAULT_OUT_DIR = _HERE / "output"

IC86_SEASONS = (
    "IC86_I", "IC86_II", "IC86_III", "IC86_IV", "IC86_V", "IC86_VI",
    "IC86_VII", "IC86_VIII", "IC86_IX", "IC86_X", "IC86_XI",
)

RADIUS_KM = 0.62  # IceCube-like instrumented sphere
PHI0 = 0.63  # reference flux normalization [1e-18 GeV^-1 cm^-2 s^-1 sr^-1]
GAMMA = 2.38  # reference spectral index
LOG10_E_MIN_FIT = 6.0  # 1 PeV; below this the sample is background-dominated
LOG10_E_EDGES = np.arange(3.0, 8.01, 0.5)
DEC_MIN, DEC_MAX = 0.0, 90.0  # upgoing hemisphere


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=_DEFAULT_DATA_DIR,
        help="Root of the DR2 data directory.",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=_DEFAULT_OUT_DIR / "21_full_icecube_comparison.pdf",
        help="Output file for the figure.",
    )
    return parser.parse_args()


def load_ic86_observed_and_livetime(data_dir: pathlib.Path) -> tuple[np.ndarray, float]:
    raw = load_all_seasons(data_dir)
    events = EventSet(raw)
    mask = np.zeros(events.n_events, dtype=bool)
    for season in IC86_SEASONS:
        uptime = load_uptime(data_dir / "uptime" / f"{season}_exp.csv")
        for start, stop in uptime:
            mask |= (events.time >= start) & (events.time <= stop)
    ic86_events = EventSet(events.data[mask])

    counts = observed_counts(ic86_events, LOG10_E_EDGES, DEC_MIN, DEC_MAX)
    livetime_s = sum(
        compute_livetime_s(load_uptime(data_dir / "uptime" / f"{season}_exp.csv"))
        for season in IC86_SEASONS
    )
    return counts, livetime_s


def soft_volume_templates(response, livetime_s: float) -> dict[str, np.ndarray]:
    """Two soft-volume predictions: numu alone and numu + nu_tau.

    Both use the per-event PREM Earth attenuation (Form B) of example 14;
    ``response`` must be built with ``attenuation_column_g_cm2=None`` so
    :meth:`~softpaws.response.soft_volume.SoftVolumeResponse.
    expected_counts_attenuated` applies attenuation only once.
    """
    numu = response.expected_counts_attenuated(
        LOG10_E_EDGES, PHI0, GAMMA, livetime_s, DEC_MIN, DEC_MAX,
    )
    tau = tau_induced_expected_counts_attenuated(
        response, LOG10_E_EDGES, PHI0, GAMMA, livetime_s, DEC_MIN, DEC_MAX,
    )

    return {
        "numu": numu,
        "numu + nu_tau": numu + tau,
    }


def make_figure(
    observed: np.ndarray,
    irf_fitted: np.ndarray,
    soft_fitted: dict[str, np.ndarray],
    efficiencies: dict[str, float],
    out_path: pathlib.Path,
) -> None:
    styles = {
        "numu": (":", "0.55"),
        "numu + nu_tau": ("-", "C3"),
    }
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.4, 3.2))

        ax.stairs(observed, LOG10_E_EDGES, label="observed (IC86)", lw=1.8, color="k")
        ax.stairs(irf_fitted, LOG10_E_EDGES, label="IRF (fitted)", ls="--", color="C1")
        for name, counts in soft_fitted.items():
            ls, color = styles[name]
            ax.stairs(counts, LOG10_E_EDGES, label=f"soft: {name}", ls=ls, color=color)

        ax.set_yscale("log")
        ax.set_xlabel(r"$\log_{10}(E\,/\,\mathrm{GeV})$")
        ax.set_ylabel("tracks / bin")
        ax.legend(fontsize=6)

        eff_lines = "\n".join(
            rf"$\epsilon_\mathrm{{{short}}} = {eff:.2f}$"
            for short, eff in efficiencies.items()
        )
        ax.text(
            0.05, 0.05, eff_lines,
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6,
        )

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png"):
            path = out_path.with_suffix(suffix)
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {path.resolve()}")
        plt.close(fig)


def main() -> None:
    args = parse_args()

    print(f"Loading IC86 events from: {args.data_dir}")
    observed, livetime_s = load_ic86_observed_and_livetime(args.data_dir)
    print(f"  IC86 upgoing events: {observed.sum():,.0f}")
    print(f"  IC86 combined livetime: {livetime_s / 86400.0:.1f} d")

    print("Loading IC86 IRFs ...")
    irfs = load_irfs(args.data_dir / "irfs", "IC86_I")

    def flux_fn(energy_gev):
        return power_law_flux(energy_gev, PHI0, GAMMA)

    print("Computing IRF prediction ...")
    irf_template = irf_expected_counts(
        irfs["aeff"], irfs["smearing"], LOG10_E_EDGES, DEC_MIN, DEC_MAX, flux_fn, livetime_s,
    )

    print("Computing soft-volume predictions (numu / numu + nu_tau, Form B attenuation) ...")
    response = SoftVolumeResponse(radius_km=RADIUS_KM, method="exact")
    soft_templates = soft_volume_templates(response, livetime_s)

    scale_irf = fit_scale_factor(observed, irf_template, LOG10_E_EDGES, LOG10_E_MIN_FIT)
    soft_fitted: dict[str, np.ndarray] = {}
    efficiencies: dict[str, float] = {}
    short_names = {"numu": r"\nu_\mu", "numu + nu_tau": r"\nu_\mu+\nu_\tau"}
    for name, template in soft_templates.items():
        scale = fit_scale_factor(observed, template, LOG10_E_EDGES, LOG10_E_MIN_FIT)
        eff = implied_efficiency(observed, template, irf_template, LOG10_E_EDGES, LOG10_E_MIN_FIT)
        soft_fitted[name] = template * scale
        efficiencies[short_names[name]] = eff
        print(f"  {name:24s}: best-fit phi0 = {PHI0 * scale:6.3f},  eps = {eff:.3f}")
    print("  (paper efficiency eps_IC-TG ~ 0.45)")

    make_figure(observed, irf_template * scale_irf, soft_fitted, efficiencies, args.out)


if __name__ == "__main__":
    main()
