"""Example 06 — drift-limit soft-volume track counts, by contribution.

Uses :class:`~softpaws.response.soft_volume.SoftVolumeResponse` to predict the
expected number of through-going muon tracks per energy bin for an IceCube-like
spherical detector (``R ~ 0.62 km``, water) exposed to the paper's best-fit
diffuse flux (``phi0 = 0.63``, ``gamma = 2.38``; arXiv:2607.13143, Eq. 1.3),
split into the instrumented-volume ("inside") and soft-volume contributions.

The soft-volume term dominates by a factor of a few -- the whole point of the
formalism. Counts are geometric (unit efficiency, no event selection or angular
acceptance beyond the hemisphere solid angle); multiply by ``eps ~ 0.45`` for a
rough detector-level estimate (see ``docs/soft_volume_notes.md``).

Usage
-----
    python examples/06_soft_volume_event_rate.py
    python examples/06_soft_volume_event_rate.py --out examples/output/06_soft_volume_event_rate.pdf
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.fluxes import REFERENCE_SPL
from softpaws.response.soft_volume import SoftVolumeResponse

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"

RADIUS_KM = 0.62  # IceCube-like instrumented sphere
PHI0 = REFERENCE_SPL.phi0  # [1e-18 GeV^-1 cm^-2 s^-1 sr^-1] at 100 TeV
GAMMA = REFERENCE_SPL.gamma
LIVETIME_S = 9.5 * 365.25 * 86400.0  # IceCube 9.5 yr exposure
SOLID_ANGLE_SR = 2.0 * np.pi  # upgoing hemisphere
LOG10_E_EDGES = np.arange(4.0, 9.01, 0.5)  # 10 TeV to 1 EeV, half-decade bins


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=_DEFAULT_OUT_DIR / "06_soft_volume_event_rate.pdf",
        help="Output file for the figure.",
    )
    return parser.parse_args()


def make_figure(out_path: pathlib.Path) -> None:
    response = SoftVolumeResponse(radius_km=RADIUS_KM)

    counts = {
        part: response.expected_counts(
            LOG10_E_EDGES, PHI0, GAMMA, LIVETIME_S, SOLID_ANGLE_SR, part=part
        )
        for part in ("total", "soft", "inside")
    }

    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3, 3))

        ax.stairs(counts["total"], LOG10_E_EDGES, label="total", lw=1.8)
        ax.stairs(counts["soft"], LOG10_E_EDGES, label="soft volume", ls="--")
        ax.stairs(counts["inside"], LOG10_E_EDGES, label="inside", ls=":")

        ax.set_yscale("log")
        ax.set_xlabel(r"$\log_{10}(E_\mu\,/\,\mathrm{GeV})$")
        ax.set_ylabel("expected tracks / bin (9.5 yr)")
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
