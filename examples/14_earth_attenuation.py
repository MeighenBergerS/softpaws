"""Example 14 -- Earth attenuation of the upgoing soft-volume prediction.

Builds on ``12_soft_volume_exact_vs_irf.py`` by adding neutrino Earth
attenuation to the exact soft-volume path, in the two forms of
:mod:`softpaws.transport.attenuation`:

* **no attenuation** -- the ``D_nu = 1`` baseline of example 12;
* **closed form (Form A)** -- a single solid-angle-averaged, constant
  mean-density Earth column for the declination band
  (:func:`~softpaws.transport.attenuation.representative_column`), applied as an
  energy-dependent factor ``D_nu(E)`` on the flux;
* **per-event (Form B)** -- the layered-PREM column along each declination's
  Earth chord, folded into the solid-angle integral by
  :meth:`~softpaws.response.soft_volume.SoftVolumeResponse.expected_counts_attenuated`.

The point of the comparison is example 12's headline: without attenuation the
exact soft-volume path over-predicts the observed >1 PeV upgoing counts, so the
implied efficiency ``eps = scale_soft / scale_irf`` comes out far below the
paper's ``~0.45`` (arXiv:2607.13143, Eq. 1.4). Earth attenuation suppresses the
prediction most at high energy and near-vertical-up, which should raise ``eps``
toward the paper's value. Each variant is fit to the observed data with a single
overall normalization above 1 PeV, and its implied efficiency is reported.

Same sample as example 12: IC86 seasons, upgoing hemisphere (``dec > 0``), with
the caveats noted there (bare astrophysical template, no atmospheric-neutrino
background; the low-energy excess is background, not signal -- see the
``atmospheric-flux`` note).

Usage
-----
    python examples/14_earth_attenuation.py
    python examples/14_earth_attenuation.py --data-dir /path/to/dataverse_files
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
from softpaws.response.soft_volume import SoftVolumeResponse, power_law_flux
from softpaws.transport.attenuation import representative_column

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_DATA_DIR = _HERE.parent / "src" / "softpaws" / "data" / "dataverse_files"
_DEFAULT_OUT_DIR = _HERE / "output"


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
        default=_DEFAULT_OUT_DIR / "14_earth_attenuation.pdf",
        help="Output file for the figure.",
    )
    return parser.parse_args()


def load_ic86_observed_and_livetime(data_dir: pathlib.Path) -> tuple[np.ndarray, float]:
    """Binned IC86 upgoing counts and the IC86 livetime, from the library loaders."""
    events = load_events(data_dir, IC86_SEASONS, within_uptime=True)
    counts = observed_counts(events, LOG10_E_EDGES, DEC_MIN, DEC_MAX)
    return counts, total_livetime_s(data_dir, IC86_SEASONS)


def soft_volume_templates(livetime_s: float) -> dict[str, np.ndarray]:
    """Exact soft-volume counts with no / closed-form / per-event attenuation."""
    solid_angle_sr = 2.0 * np.pi * (np.sin(np.deg2rad(DEC_MAX)) - np.sin(np.deg2rad(DEC_MIN)))

    none = SoftVolumeResponse(radius_km=RADIUS_KM, method="exact")
    closed = SoftVolumeResponse(
        radius_km=RADIUS_KM,
        method="exact",
        attenuation_column_g_cm2=representative_column(DEC_MIN, DEC_MAX),
    )
    return {
        "no attenuation": none.expected_counts(
            LOG10_E_EDGES, PHI0, GAMMA, livetime_s, solid_angle_sr
        ),
        "closed form (Form A)": closed.expected_counts(
            LOG10_E_EDGES, PHI0, GAMMA, livetime_s, solid_angle_sr
        ),
        "per-event PREM (Form B)": none.expected_counts_attenuated(
            LOG10_E_EDGES, PHI0, GAMMA, livetime_s, DEC_MIN, DEC_MAX
        ),
    }


def make_figure(
    observed: np.ndarray,
    irf_fitted: np.ndarray,
    soft_fitted: dict[str, np.ndarray],
    efficiencies: dict[str, float],
    out_path: pathlib.Path,
) -> None:
    styles = {
        "no attenuation": (":", "0.55"),
        "closed form (Form A)": ("--", "C0"),
        "per-event PREM (Form B)": ("-.", "C2"),
    }
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.4, 3.2))

        ax.stairs(observed, LOG10_E_EDGES, label="observed (IC86)", lw=1.8, color="k")
        ax.stairs(irf_fitted, LOG10_E_EDGES, label="IRF (fitted)", ls="--", color="C3")
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

    print("Computing soft-volume predictions (no / closed / per-event attenuation) ...")
    soft_templates = soft_volume_templates(livetime_s)

    scale_irf = fit_scale_factor(observed, irf_template, LOG10_E_EDGES, LOG10_E_MIN_FIT)
    soft_fitted: dict[str, np.ndarray] = {}
    efficiencies: dict[str, float] = {}
    short_names = {
        "no attenuation": r"D_\nu=1",
        "closed form (Form A)": r"A",
        "per-event PREM (Form B)": r"B",
    }
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
