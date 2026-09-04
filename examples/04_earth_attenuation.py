"""Tutorial 04 -- what the Earth does to a neutrino on the way in.

A neutrino arriving from below crosses up to a full Earth diameter of matter.
Above a few tens of TeV that matter is not transparent, and by an EeV the
Earth is opaque except near the horizon. This tutorial computes the column
along each arrival direction from the PREM profile, turns it into a survival
probability, and shows what neutral-current regeneration adds back.

Usage
-----
    python examples/04_earth_attenuation.py
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.detectors import ICECUBE
from softpaws.transport import bgr18_cross_section, earth
from softpaws.transport.attenuation import regenerated_transmission, survival_probability

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"

#: Declinations the table prints [deg]; at the Pole these are the zeniths.
DEC_TABLE_DEG = (0.0, 5.0, 15.0, 30.0, 60.0, 90.0)

#: Neutrino energies the table prints [log10 GeV].
LOG10_E_TABLE = (4.0, 5.0, 6.0, 7.0, 8.0)

#: Directions and energies the figure runs over.
COS_THETA = np.linspace(-1.0, 0.0, 121)
LOG10_E_GRID = np.linspace(3.0, 9.0, 121)


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the figure.")
    return parser.parse_args()


def report() -> None:
    """Print the chord, the column and the survival against declination."""
    cross_section = bgr18_cross_section()
    print("The Earth an upgoing neutrino crosses, seen from the South Pole:\n")
    print(f"{'dec [deg]':>10} {'chord [km]':>11} {'column [g/cm^2]':>16}"
          + "".join(f"{'D(1e%d)' % e:>10}" for e in (5, 6, 7)))
    for dec in DEC_TABLE_DEG:
        chord = float(np.squeeze(earth.earth_chord_length_km(dec)))
        column = earth.prem_column(dec)
        row = f"{dec:10.0f} {chord:11.0f} {column:16.3e}"
        for log10_e in (5, 6, 7):
            d_nu = float(np.squeeze(
                survival_probability(10.0**log10_e, column, cross_section=cross_section)
            ))
            row += f"{d_nu:10.3f}"
        print(row)
    print("\n  At the horizon the column vanishes and everything gets through. At the")
    print("  nadir a PeV neutrino has a few per cent left and an EeV one nothing.")

    print("\nNeutral-current regeneration puts some of it back, at lower energy.")
    column = earth.prem_column(30.0)
    for log10_e in (6.0, 7.0):
        energies, weights = regenerated_transmission(10.0**log10_e, np.array([column]),
                                                     cross_section)
        surviving = float(np.sum(weights))
        absorbed_only = float(np.squeeze(
            survival_probability(10.0**log10_e, column, cross_section=cross_section)
        ))
        print(f"  {log10_e:.0f}: absorption only {absorbed_only:.4f}, "
              f"with regeneration {surviving:.4f} spread over {energies.size} rungs")


def make_figure(out_path: pathlib.Path) -> None:
    """Draw the survival probability against energy and arrival direction."""
    cross_section = bgr18_cross_section()
    columns = np.array([
        earth.neutrino_column_g_cm2(c, ICECUBE.depth_km, ICECUBE.density_g_cm3)
        for c in COS_THETA
    ]).ravel()
    energies = 10.0**LOG10_E_GRID
    survival = np.array([
        np.ravel(survival_probability(energies, column, cross_section=cross_section))
        for column in columns
    ])

    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.4, 3.2))
        mesh = ax.pcolormesh(LOG10_E_GRID, COS_THETA, survival, vmin=0.0, vmax=1.0,
                             shading="auto", cmap="magma")
        ax.set_xlabel(r"$\log_{10}(E_\nu/\mathrm{GeV})$")
        ax.set_ylabel(r"$\cos\theta_z$")
        ax.set_title("Earth transmission")
        fig.colorbar(mesh, ax=ax, label=r"$D_\nu$")
        fig.tight_layout()
        for suffix in (".pdf", ".png"):
            fig.savefig(out_path.with_suffix(suffix))
            print(f"Figure saved to: {out_path.with_suffix(suffix)}")
        plt.close(fig)


def main() -> None:
    """Report the Earth columns and draw the transmission."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report()
    make_figure(args.out_dir / "t04_earth_attenuation")


if __name__ == "__main__":
    main()
