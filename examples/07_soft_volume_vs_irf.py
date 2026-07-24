"""Example 07 — observed vs. IRF- and soft-volume-predicted track counts.

Compares the observed IceTracks-DR2 reconstructed-energy histogram against two
predictions at the paper's best-fit diffuse flux (``phi0 = 0.63``,
``gamma = 2.38``; arXiv:2607.13143, Eq. 1.3): the published-IRF path
(effective area migrated through the energy smearing matrix) and the
drift-limit soft-volume path. Both predictions are fit to the observed data
with a single overall normalization (see
:func:`~softpaws.comparison.rates.fit_scale_factor`), and their ratio is
reported as a data-driven analogue of the paper's efficiency factor
``eps_IC-TG ~ 0.45`` (Eq. 1.4).

Restricted to the downgoing hemisphere (``dec < 0``), where the soft-volume
drift limit's implicit unattenuated-flux assumption is appropriate -- see the
module docstring of :mod:`softpaws.comparison.rates` and
``docs/soft_volume_notes.md``. Only the IC86 seasons are used, since IC86 is
the last (and longest-lived) detector configuration and its smearing table is
shared across all IC86 sub-seasons.

Caveat: the fit here is a bare single-parameter astrophysical-flux template
with no atmospheric-neutrino background model, unlike the paper's Eq. 4.2. The
observed IC86 sample below ~1 PeV is overwhelmingly atmospheric, not
astrophysical: ``observed / irf_template`` falls steeply from ~800 at 10 TeV to
~25 at 20 PeV, a textbook background signature. The default fit range
therefore starts at 1 PeV, where the diffuse astrophysical flux dominates. Even
so, treat the resulting efficiency as indicative, not a reproduction of the
paper's Section 4.1 fit.

Usage
-----
    python examples/07_soft_volume_vs_irf.py
    python examples/07_soft_volume_vs_irf.py --data-dir /path/to/dataverse_files
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
from softpaws.response.soft_volume import SoftVolumeResponse, power_law_flux

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
# (see the "Caveat" note in the module docstring above).
LOG10_E_EDGES = np.arange(3.0, 8.01, 0.5)
DEC_MIN, DEC_MAX = -90.0, 0.0  # downgoing hemisphere


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
        default=_DEFAULT_OUT_DIR / "07_soft_volume_vs_irf.pdf",
        help="Output file for the figure.",
    )
    return parser.parse_args()


def load_ic86_observed_and_livetime(data_dir: pathlib.Path) -> tuple[np.ndarray, float]:
    raw = load_all_seasons(data_dir)
    events = EventSet(raw)
    # Restrict to IC86 seasons so the observed sample matches the single IC86
    # IRF used for the prediction.
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


def make_figure(
    observed: np.ndarray,
    irf_fitted: np.ndarray,
    soft_fitted: np.ndarray,
    eff: float,
    out_path: pathlib.Path,
) -> None:
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3, 3))

        ax.stairs(observed, LOG10_E_EDGES, label="observed (IC86)", lw=1.8, color="k")
        ax.stairs(irf_fitted, LOG10_E_EDGES, label="IRF (fitted)", ls="--")
        ax.stairs(soft_fitted, LOG10_E_EDGES, label="soft volume (fitted)", ls=":")

        ax.set_yscale("log")
        ax.set_xlabel(r"$\log_{10}(E\,/\,\mathrm{GeV})$")
        ax.set_ylabel("tracks / bin")
        ax.legend()
        ax.text(
            0.05, 0.05, rf"$\epsilon = ${eff:.2f}",
            transform=ax.transAxes, ha="left", va="bottom",
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
    print(f"  IC86 downgoing events: {observed.sum():,.0f}")
    print(f"  IC86 combined livetime: {livetime_s / 86400.0:.1f} d")

    print("Loading IC86 IRFs ...")
    irfs = load_irfs(args.data_dir / "irfs", "IC86_I")

    def flux_fn(energy_gev):
        return power_law_flux(energy_gev, PHI0, GAMMA)

    print("Computing IRF prediction ...")
    irf_template = irf_expected_counts(
        irfs["aeff"], irfs["smearing"], LOG10_E_EDGES, DEC_MIN, DEC_MAX, flux_fn, livetime_s,
    )

    print("Computing soft-volume prediction ...")
    response = SoftVolumeResponse(radius_km=RADIUS_KM)
    solid_angle_sr = 2.0 * np.pi * (np.sin(np.deg2rad(DEC_MAX)) - np.sin(np.deg2rad(DEC_MIN)))
    soft_template = response.expected_counts(
        LOG10_E_EDGES, PHI0, GAMMA, livetime_s, solid_angle_sr,
    )

    scale_irf = fit_scale_factor(observed, irf_template, LOG10_E_EDGES, LOG10_E_MIN_FIT)
    scale_soft = fit_scale_factor(observed, soft_template, LOG10_E_EDGES, LOG10_E_MIN_FIT)
    eff = implied_efficiency(observed, soft_template, irf_template, LOG10_E_EDGES, LOG10_E_MIN_FIT)

    print(f"  best-fit phi0 (IRF)        : {PHI0 * scale_irf:.3f}")
    print(f"  best-fit phi0 (soft volume): {PHI0 * scale_soft:.3f}")
    print(f"  implied efficiency eps = scale_soft / scale_irf: {eff:.3f}  (paper: ~0.45)")

    make_figure(observed, irf_template * scale_irf, soft_template * scale_soft, eff, args.out)


if __name__ == "__main__":
    main()
