"""Example 12 — observed vs. IRF- and exact-soft-volume-predicted track counts.

Same comparison as ``07_soft_volume_vs_irf.py``, but the soft-volume path uses
the exact eigenvalue transport (``method="exact"``,
``docs/exact_soft_volume_notes.md``) instead of the paper's Fokker-Planck drift
limit. Compares the observed IceTracks-DR2 reconstructed-energy histogram
against the published-IRF path and the exact soft-volume path, both at the
paper's best-fit diffuse flux (``phi0 = 0.63``, ``gamma = 2.38``;
arXiv:2607.13143, Eq. 1.3) and both fit to the observed data with a single
overall normalization (see :func:`~softpaws.comparison.rates.fit_scale_factor`).
Their ratio is reported as a data-driven analogue of the paper's efficiency
factor ``eps_IC-TG ~ 0.45`` (Eq. 1.4); the exact path's ``I(A) ~ 0.8``
normalization is expected to shift this efficiency higher (``docs/
exact_soft_volume_notes.md``, Section 10).

Restricted to the upgoing hemisphere (``dec > 0``), where the atmospheric-muon
background is Earth-shielded and the sample is far less contaminated than
downgoing. Only the IC86 seasons are used, since IC86 is the last (and
longest-lived) detector configuration and its smearing table is shared across
all IC86 sub-seasons.

By default the exact path uses the infinite-column limit (``--column-depth-km``
unset), for an apples-to-apples comparison with example 07's drift form.
``--column-depth-km`` is provided for experimentation, but note that for the
upgoing sample the physically relevant column is the Earth chord along each
event's incidence angle (up to Earth's diameter for near-vertical upgoing
tracks), not IceCube's ~1.95 km overburden used for downgoing muons -- a single
scalar value here cannot represent that per-event geometry.

Caveat: the soft-volume path's unattenuated-flux assumption (``D_nu = 1`` in
Eq. 2.21; see the module docstring of :mod:`softpaws.comparison.rates` and
``docs/soft_volume_notes.md``) is only strictly valid for the downgoing
hemisphere. Upgoing neutrinos cross the Earth, and above ~100 TeV-1 PeV the
neutrino-nucleon cross section grows enough that Earth attenuation is no
longer negligible -- right in this example's fit range. Neither the drift nor
the exact soft-volume path models that attenuation (the exact path's
finite-column saturation factor above describes the *muon's* upstream column
after production, not Earth attenuation of the parent neutrino), so the
soft-volume prediction -- and hence the implied efficiency -- is likely
overestimated here, more so at the high-energy end of the fit range. Treat
the resulting efficiency as indicative, not a reproduction of the paper's
Section 4.1 fit, which does model attenuation.

Caveat: the fit here is also a bare single-parameter astrophysical-flux
template with no atmospheric-neutrino background model, unlike the paper's
Eq. 4.2. The default fit range starts at 1 PeV, where the diffuse
astrophysical flux is expected to dominate over the (much smaller, but
non-zero) upgoing atmospheric-neutrino background.

Usage
-----
    python examples/12_soft_volume_exact_vs_irf.py
    python examples/12_soft_volume_exact_vs_irf.py --column-depth-km 1.95
    python examples/12_soft_volume_exact_vs_irf.py --data-dir /path/to/dataverse_files
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
from softpaws.data.icecube import (
    IC86_SEASONS,
    load_events,
    total_livetime_s,
)
from softpaws.data.loader import load_irfs
from softpaws.fluxes import REFERENCE_SPL
from softpaws.response.soft_volume import SoftVolumeResponse, power_law_flux

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_DATA_DIR = _HERE.parent / "src" / "softpaws" / "data" / "dataverse_files"
_DEFAULT_OUT_DIR = _HERE / "output"


RADIUS_KM = 0.62  # IceCube-like instrumented sphere
PHI0 = REFERENCE_SPL.phi0  # [1e-18 GeV^-1 cm^-2 s^-1 sr^-1] at 100 TeV
GAMMA = REFERENCE_SPL.gamma
LOG10_E_MIN_FIT = 6.0  # 1 PeV; below this the sample is background-dominated
# (see the "Caveat" note in the module docstring above).
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
        "--column-depth-km",
        type=float,
        default=None,
        help="Upstream column depth for the exact soft-volume path [km]; "
        "default is the infinite-column limit (apples-to-apples with example "
        "07's drift form).",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=_DEFAULT_OUT_DIR / "12_soft_volume_exact_vs_irf.pdf",
        help="Output file for the figure.",
    )
    return parser.parse_args()


def load_ic86_observed_and_livetime(data_dir: pathlib.Path) -> tuple[np.ndarray, float]:
    """Binned IC86 upgoing counts and the IC86 livetime, from the library loaders."""
    events = load_events(data_dir, IC86_SEASONS, within_uptime=True)
    counts = observed_counts(events, LOG10_E_EDGES, DEC_MIN, DEC_MAX)
    return counts, total_livetime_s(data_dir, IC86_SEASONS)


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
        ax.stairs(soft_fitted, LOG10_E_EDGES, label="soft volume, exact (fitted)", ls=":")

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
    column = args.column_depth_km
    label = "infinite column" if column is None else f"column depth {column:g} km"
    print(f"Exact soft-volume path: {label}")

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

    print("Computing exact soft-volume prediction ...")
    response = SoftVolumeResponse(radius_km=RADIUS_KM, method="exact", column_depth_km=column)
    solid_angle_sr = 2.0 * np.pi * (np.sin(np.deg2rad(DEC_MAX)) - np.sin(np.deg2rad(DEC_MIN)))
    soft_template = response.expected_counts(
        LOG10_E_EDGES, PHI0, GAMMA, livetime_s, solid_angle_sr,
    )

    scale_irf = fit_scale_factor(observed, irf_template, LOG10_E_EDGES, LOG10_E_MIN_FIT)
    scale_soft = fit_scale_factor(observed, soft_template, LOG10_E_EDGES, LOG10_E_MIN_FIT)
    eff = implied_efficiency(observed, soft_template, irf_template, LOG10_E_EDGES, LOG10_E_MIN_FIT)

    print(f"  best-fit phi0 (IRF)               : {PHI0 * scale_irf:.3f}")
    print(f"  best-fit phi0 (soft volume, exact) : {PHI0 * scale_soft:.3f}")
    print(f"  implied efficiency eps = scale_soft / scale_irf: {eff:.3f}  (paper: ~0.45)")

    make_figure(observed, irf_template * scale_irf, soft_template * scale_soft, eff, args.out)


if __name__ == "__main__":
    main()
