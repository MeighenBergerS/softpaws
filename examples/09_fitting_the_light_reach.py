"""Tutorial 09 -- reading the light reach off a published effective area.

Tutorial 05 left a gap: the model built from the instrumented footprint alone
sits about a quarter below the published table. The missing piece is the light
a bright muon makes outside the array, which lets a track be reconstructed
from further away the more energy it carries.

That growth is one number, a reach per e-fold of muon energy. This tutorial
scans it against IceCube's published upgoing effective area and shows what the
best value does to the comparison. One number, fitted to one curve, is the
whole of what the instrument contributes here.

Usage
-----
    python examples/09_fitting_the_light_reach.py
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.data import dr2_dir, icecube_dr2_aeff
from softpaws.detectors import ICECUBE
from softpaws.response.declination import directional_effective_area_cm2
from softpaws.transport.earth import zenith_grid

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"

#: Neutrino energies the comparison runs on [log10 GeV].
LOG10_E = np.linspace(4.0, 8.0, 17)

#: Band the reach is scored over [log10 GeV].
FIT_BAND = (5.0, 7.8)

#: Reaches scanned [km per e-fold of muon energy].
REACH_GRID_KM = np.linspace(0.0, 0.030, 11)

#: Energy at which the reach vanishes [GeV].
PIVOT_GEV = 10.0**9.67

#: Directions the upgoing average runs over.
N_ZENITH = 16


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threshold-gev", type=float, default=1.0e3,
                        help="Muon selection threshold.")
    parser.add_argument("--data-dir", type=pathlib.Path, default=None,
                        help="Root of the DR2 release.")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the figure.")
    return parser.parse_args()


def upgoing_model(threshold_gev: float, reach_km: float | None) -> np.ndarray:
    """Upgoing effective area, with or without a light reach [cm^2]."""
    theta_deg, weights = zenith_grid(N_ZENITH, (-1.0, 0.0))
    cos_theta = np.cos(np.deg2rad(theta_deg))
    per_direction = directional_effective_area_cm2(
        ICECUBE, cos_theta, threshold_gev, reach_km=reach_km, pivot_gev=PIVOT_GEV,
        log10_e=LOG10_E,
    )
    return per_direction @ weights


def scan(published: np.ndarray, threshold_gev: float) -> tuple[float, np.ndarray]:
    """Scan the reach for the value that puts the model on the published curve.

    Parameters
    ----------
    published : np.ndarray
        The published effective area on :data:`LOG10_E` [cm^2].
    threshold_gev : float
        Muon selection threshold [GeV].

    Returns
    -------
    best_km : float
        Reach that minimizes the mean-square residual in the logarithm.
    residual_dex : np.ndarray
        Root-mean-square residual [dex] at each scanned reach.
    """
    band = (LOG10_E >= FIT_BAND[0]) & (LOG10_E <= FIT_BAND[1])
    residual = np.empty(REACH_GRID_KM.size)
    for i, reach_km in enumerate(REACH_GRID_KM):
        model = upgoing_model(threshold_gev, reach_km if reach_km > 0.0 else None)
        residual[i] = float(np.sqrt(np.mean(np.log10(published[band] / model[band]) ** 2)))
        print(f"  reach {reach_km * 1e3:5.1f} m per e-fold: {residual[i]:.3f} dex")
    return float(REACH_GRID_KM[np.argmin(residual)]), residual


def main() -> None:
    """Scan the reach, report the best value, and draw the comparison."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading the published DR2 table ...")
    published = icecube_dr2_aeff(args.data_dir or dr2_dir(), LOG10_E)[0]

    print(f"\nScanning the reach against 10^{FIT_BAND[0]} to 10^{FIT_BAND[1]} GeV:")
    best_km, residual = scan(published, args.threshold_gev)

    band = (LOG10_E >= FIT_BAND[0]) & (LOG10_E <= FIT_BAND[1])
    curves = {
        "footprint": upgoing_model(args.threshold_gev, None),
        f"reach {best_km * 1e3:.0f} m": upgoing_model(args.threshold_gev, best_km),
    }
    print(f"\nThe best reach is {best_km * 1e3:.0f} m per e-fold of muon energy,")
    print(f"  against an instrumented radius of {ICECUBE.radius_km * 1e3:.0f} m.")
    print(f"  At 1 PeV it adds {best_km * np.log(1.0e6 / PIVOT_GEV) * -1e3:.0f} m "
          "to the effective footprint.\n")
    print(f"{'log10(E/GeV)':>13}" + "".join(f"{f'published/{n}':>22}" for n in curves))
    for log10_e in (5.0, 6.0, 7.0):
        i = int(np.argmin(np.abs(LOG10_E - log10_e)))
        row = f"{log10_e:13.1f}"
        for curve in curves.values():
            row += f"{published[i] / curve[i]:22.2f}"
        print(row)
    for name, curve in curves.items():
        ratio = published[band] / curve[band]
        print(f"\n  {name}: level {np.exp(np.mean(np.log(ratio))):.2f}, "
              f"{np.std(np.log10(ratio)):.3f} dex of shape left over")

    with plt.style.context(str(_STYLE)):
        fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.0))
        axes[0].plot(REACH_GRID_KM * 1e3, residual, lw=1.4)
        axes[0].axvline(best_km * 1e3, color="0.7", lw=0.8, zorder=0)
        axes[0].set_xlabel("reach [m per e-fold]")
        axes[0].set_ylabel("residual [dex]")
        axes[0].set_title("the scan")

        axes[1].plot(LOG10_E, published, color="0.5", lw=2.2, label="published", zorder=1)
        for name, curve in curves.items():
            axes[1].plot(LOG10_E, curve, lw=1.3, label=name, zorder=3)
        axes[1].set_yscale("log")
        axes[1].set_xlabel(r"$\log_{10}(E_\nu/\mathrm{GeV})$")
        axes[1].set_ylabel(r"$A_{\rm eff}$ [cm$^2$]")
        axes[1].set_title("IceCube DR2, upgoing")
        axes[1].legend(frameon=False, fontsize=7, loc="upper left")
        fig.tight_layout()
        out_path = args.out_dir / "09_fitting_the_light_reach"
        for suffix in (".pdf", ".png"):
            fig.savefig(out_path.with_suffix(suffix))
            print(f"Figure saved to: {out_path.with_suffix(suffix)}")
        plt.close(fig)


if __name__ == "__main__":
    main()
