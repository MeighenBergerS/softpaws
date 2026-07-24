"""Example 05 — drift-limit soft-volume enhancement versus muon energy.

Plots the total-to-instrumented volume ratio ``V_tot / V_det`` (arXiv:2607.13143,
Eq. 2.24) as a function of muon energy for two spherical detectors: IceCube
(``V_det ~ 1 km^3``, ``R ~ 0.62 km``) and the smaller KM3NeT (``R ~ 0.33 km``),
both in water and assuming the IceCube diffuse-flux index ``gamma = 2.38``. The
paper's reference figures of merit -- about 4x for IceCube and about 7.5x for
KM3NeT at 1 PeV -- are marked for comparison.

Usage
-----
    python examples/05_soft_volume_drift.py
    python examples/05_soft_volume_drift.py --out examples/output/05_soft_volume_drift.pdf
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.transport.soft_volume import sphere_radius_from_volume, volume_ratio_drift

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"

GAMMA_IC = 2.38  # IceCube 9.5 yr diffuse-flux best fit (Eq. 1.3)
LOG10_E = np.linspace(5.0, 9.0, 81)  # 100 TeV to 1 EeV, in log10(E / GeV)

# Spherical-detector radii [km]. IceCube from its ~1 km^3 instrumented volume;
# KM3NeT chosen to reproduce the paper's ~7.5x figure of merit.
DETECTORS = {
    "IceCube": sphere_radius_from_volume(1.0),
    "KM3NeT": 0.33,
}

# Paper reference figures of merit at 1 PeV (Section 2.3).
REFERENCE_POINTS = {
    "IceCube": (6.0, 4.0),
    "KM3NeT": (6.0, 7.5),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=_DEFAULT_OUT_DIR / "05_soft_volume_drift.pdf",
        help="Output file for the figure.",
    )
    return parser.parse_args()


def make_figure(out_path: pathlib.Path) -> None:
    energy_gev = 10.0**LOG10_E

    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3, 3))

        for name, radius_km in DETECTORS.items():
            ratio = volume_ratio_drift(radius_km, energy_gev, GAMMA_IC)
            (line,) = ax.plot(LOG10_E, ratio, label=f"{name} (R = {radius_km:.2f} km)")

            log10_e_ref, ratio_ref = REFERENCE_POINTS[name]
            ax.scatter(
                log10_e_ref, ratio_ref, color=line.get_color(), marker="o",
                zorder=5, edgecolor="k", linewidth=0.5,
            )

        ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
        ax.set_xlabel(r"$\log_{10}(E_\mu\,/\,\mathrm{GeV})$")
        ax.set_ylabel(r"$V_{\rm tot}\,/\,V_{\rm det}$")
        ax.legend()

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png"):
            path = out_path.with_suffix(suffix)
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {path.resolve()}")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    make_figure(args.out)


if __name__ == "__main__":
    main()
