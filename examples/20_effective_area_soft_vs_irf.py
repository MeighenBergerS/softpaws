"""Example 20 — soft-volume implied effective area vs. the published IceCube A_eff.

Extends ``03_effective_area.py``'s livetime-combined, hemisphere-split IceCube
effective area with the effective area implied by the soft-volume forward
model (:mod:`softpaws.response.soft_volume`). The target-volume factorization
(arXiv:2607.13143, Eq. 2.20) gives ``dN/dE = V_target(E) n_N sigma_CC(E)
phi_nu(E)``, so

.. math:: A_\\mathrm{eff}^\\mathrm{soft}(E) = V_\\mathrm{target}(E)\\,n_N\\,\\sigma_\\mathrm{CC}(E)

is directly comparable to the published ``A_eff(E_nu, dec)`` -- and, unlike
every other comparison in this repo, needs no flux normalization fit: it is a
fit-free check of the transport physics and detector geometry alone.

Three soft-volume variants are shown, all at the paper's reference spectral
index (``gamma = 2.38``, Eq. 1.3), against which the soft volume itself has a
residual (mild) dependence -- see :meth:`~softpaws.response.soft_volume.
SoftVolumeResponse.effective_area_cm2`:

* **drift** -- the paper's leading Fokker-Planck form;
* **exact, infinite column** -- the exact eigenvalue transport with unlimited
  upstream ice, the convention used for the upgoing comparisons of examples
  12/14;
* **exact, finite column (1.95 km)** -- the same exact transport with a
  downgoing IceCube-like ice overburden, the convention used in example 16.

None of these three soft-volume curves carry Earth attenuation, angular
acceptance, or reconstruction/quality cuts -- all folded into the published
``A_eff`` -- so this comparison isolates how much of the published curve's
shape and normalization is transport physics versus detector effects.

Usage
-----
    python examples/20_effective_area_soft_vs_irf.py
    python examples/20_effective_area_soft_vs_irf.py --data-dir /path/to/dataverse_files
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.data.loader import compute_livetime_s, load_uptime, parse_aeff
from softpaws.data.schema import SEASONS
from softpaws.response.soft_volume import SoftVolumeResponse
from softpaws.transport.soft_volume import sphere_radius_from_volume

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_DATA_DIR = _HERE.parent / "src" / "softpaws" / "data" / "dataverse_files"
_DEFAULT_OUT_DIR = _HERE / "output"

COMMON_LOG10_E = np.linspace(2.0, 10.0, 81)
RADIUS_KM = sphere_radius_from_volume(1.0)  # ~0.62 km, IceCube-like
GAMMA = 2.38  # IceCube 9.5 yr diffuse-flux best fit (Eq. 1.3); see module docstring
DOWNGOING_COLUMN_KM = 1.95  # IceCube-like downgoing ice overburden (example 16's default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=_DEFAULT_DATA_DIR,
        help="Root of the DR2 data directory (must contain 'irfs/' and 'uptime/' subfolders).",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=_DEFAULT_OUT_DIR / "20_effective_area_soft_vs_irf.pdf",
        help="Output file for the figure.",
    )
    return parser.parse_args()


def hemisphere_average(aeff, hemisphere: str) -> np.ndarray:
    """Solid-angle-weighted average of an EffectiveArea grid over one hemisphere.

    Same as ``03_effective_area.py``'s function of the same name.
    """
    centers = aeff.sin_dec_centers
    mask = centers > 0.0 if hemisphere == "upgoing" else centers < 0.0

    widths = np.diff(aeff.sin_dec_edges)[mask]
    values = aeff.values[:, mask]

    return np.average(values, axis=1, weights=widths)


def _canonical_irf_season(season: str) -> str:
    return "IC86" if season.startswith("IC86") else season


def combine_seasons_icecube(data_dir: pathlib.Path) -> dict[str, np.ndarray]:
    """Load per-season IceCube effective areas and combine via livetime weighting.

    Same as ``03_effective_area.py``'s ``combine_seasons``.
    """
    irf_dir = data_dir / "irfs"
    uptime_dir = data_dir / "uptime"

    sums = {"upgoing": np.zeros_like(COMMON_LOG10_E), "downgoing": np.zeros_like(COMMON_LOG10_E)}
    weights = {"upgoing": 0.0, "downgoing": 0.0}
    aeff_cache: dict[str, object] = {}

    for season in SEASONS:
        irf_season = _canonical_irf_season(season)
        if irf_season not in aeff_cache:
            raw = np.genfromtxt(irf_dir / f"{irf_season}_effectiveArea.csv", comments="#")
            aeff_cache[irf_season] = parse_aeff(raw)
        aeff = aeff_cache[irf_season]

        uptime = load_uptime(uptime_dir / f"{season}_exp.csv")
        livetime_s = compute_livetime_s(uptime)

        for hemisphere in ("upgoing", "downgoing"):
            curve = hemisphere_average(aeff, hemisphere)
            interp = np.interp(COMMON_LOG10_E, aeff.log10_energy_centers, curve)
            sums[hemisphere] += livetime_s * interp
            weights[hemisphere] += livetime_s

    return {h: sums[h] / weights[h] for h in ("upgoing", "downgoing")}


def soft_volume_curves() -> dict[str, np.ndarray]:
    """Soft-volume implied effective area on ``COMMON_LOG10_E``, three variants."""
    energy_gev = 10.0**COMMON_LOG10_E

    drift = SoftVolumeResponse(radius_km=RADIUS_KM, method="drift")
    exact_inf = SoftVolumeResponse(radius_km=RADIUS_KM, method="exact", column_depth_km=None)
    exact_fin = SoftVolumeResponse(
        radius_km=RADIUS_KM, method="exact", column_depth_km=DOWNGOING_COLUMN_KM
    )
    return {
        "drift": drift.effective_area_cm2(energy_gev, GAMMA),
        "exact, infinite column": exact_inf.effective_area_cm2(energy_gev, GAMMA),
        f"exact, finite column ({DOWNGOING_COLUMN_KM:g} km)": exact_fin.effective_area_cm2(
            energy_gev, GAMMA
        ),
    }


def make_figure(
    icecube: dict[str, np.ndarray],
    soft: dict[str, np.ndarray],
    out_path: pathlib.Path,
) -> None:
    soft_styles = {
        "drift": ("--", "C0"),
        "exact, infinite column": (":", "C1"),
        f"exact, finite column ({DOWNGOING_COLUMN_KM:g} km)": ("-.", "C2"),
    }
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.4, 3.2))

        ax.plot(COMMON_LOG10_E, icecube["upgoing"], color="k", lw=1.8, label="IceCube, upgoing")
        ax.plot(
            COMMON_LOG10_E, icecube["downgoing"], color="0.5", lw=1.8, label="IceCube, downgoing",
        )
        for name, curve in soft.items():
            ls, color = soft_styles[name]
            ax.plot(COMMON_LOG10_E, curve, ls=ls, color=color, label=f"soft volume: {name}")

        ax.set_yscale("log")
        ax.set_xlabel(r"$\log_{10}(E_\nu\,/\,\mathrm{GeV})$")
        ax.set_ylabel(r"$A_{\rm eff}$ [cm$^2$]")
        ax.legend(fontsize=6)

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png"):
            path = out_path.with_suffix(suffix)
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {path.resolve()}")
        plt.close(fig)


def main() -> None:
    args = parse_args()

    print(f"Loading IceCube IRFs from: {args.data_dir}")
    icecube = combine_seasons_icecube(args.data_dir)

    print(f"Computing soft-volume implied effective area (gamma = {GAMMA}) ...")
    soft = soft_volume_curves()

    # Report the ratio at a few reference energies for a quick numeric sanity check.
    print(f"{'log10(E/GeV)':>14} {'IC upgoing':>12} {'IC downgoing':>13}", end="")
    for name in soft:
        print(f" {name:>28}", end="")
    print()
    for log10_e in (4.0, 5.0, 6.0, 7.0, 8.0):
        i = int(np.argmin(np.abs(COMMON_LOG10_E - log10_e)))
        print(f"{COMMON_LOG10_E[i]:14.1f} {icecube['upgoing'][i]:12.3g} "
              f"{icecube['downgoing'][i]:13.3g}", end="")
        for name in soft:
            print(f" {soft[name][i]:28.3g}", end="")
        print()

    make_figure(icecube, soft, args.out)


if __name__ == "__main__":
    main()
