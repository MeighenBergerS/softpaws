"""Tutorial 10 -- the atmospheric background every track analysis sits on.

Most of what a neutrino telescope records is not astrophysical. Cosmic rays
hitting the atmosphere make muon neutrinos, and below a few hundred TeV they
outnumber the astrophysical ones by orders of magnitude. Where the two cross
is what decides the window an analysis can use.

MCEq solves the cascade equations for that background. Running it takes
minutes, so the table is built once and shipped with the package; this tutorial
loads that table
if it is there and builds it otherwise, then folds it through an effective
area and finds the crossing point.

Usage
-----
    python examples/10_atmospheric_background.py
    python examples/10_atmospheric_background.py --rebuild   # needs the atm extra
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.detectors import ICECUBE
from softpaws.fluxes import ICECUBE_TRACKS_2022, SHIPPED_TABLE, AtmosphericFlux, load_mceq_table
from softpaws.response.declination import directional_effective_area_cm2
from softpaws.transport.earth import zenith_grid

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"
_DEFAULT_TABLE = SHIPPED_TABLE

#: Neutrino energies the comparison runs on [log10 GeV].
LOG10_E = np.linspace(3.0, 8.0, 41)

#: Declinations the flux is drawn at [deg]; at the Pole these are zeniths.
DEC_DEG = np.array([5.0, 30.0, 60.0, 85.0])

#: Directions the upgoing rate average runs over.
N_ZENITH = 12

#: Exposure of the rate comparison [s].
LIVETIME_S = 10.0 * 365.25 * 86400.0

#: Solid angle of the upgoing sky [sr].
SOLID_ANGLE_SR = 2.0 * np.pi


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table", type=pathlib.Path, default=_DEFAULT_TABLE,
                        help="Cached MCEq table.")
    parser.add_argument("--rebuild", action="store_true",
                        help="Run MCEq again even if the cache exists; needs the atm extra.")
    parser.add_argument("--threshold-gev", type=float, default=1.0e3,
                        help="Muon selection threshold.")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the figure.")
    return parser.parse_args()


def report(flux: AtmosphericFlux, aeff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Print the two fluxes and the rates they give, and return them."""
    energy = 10.0**LOG10_E
    print("Per-flavour flux at 30 deg declination [GeV^-1 cm^-2 s^-1 sr^-1]:\n")
    print(f"{'log10(E/GeV)':>13} {'atmospheric':>14} {'astrophysical':>15} {'ratio':>9}")
    atmospheric = np.ravel(flux(energy, np.full_like(energy, 30.0)))
    astrophysical = np.ravel(ICECUBE_TRACKS_2022.flux(energy))
    for log10_e in (3.0, 4.0, 5.0, 6.0, 7.0):
        i = int(np.argmin(np.abs(LOG10_E - log10_e)))
        print(f"{log10_e:13.1f} {atmospheric[i]:14.3e} {astrophysical[i]:15.3e} "
              f"{atmospheric[i] / astrophysical[i]:9.2f}")

    # Where the ratio passes one, not where the difference is smallest: at high
    # energy both fluxes are tiny and their difference is meaningless.
    log_ratio = np.log(atmospheric / astrophysical)
    crossing = float(np.interp(0.0, log_ratio[::-1], LOG10_E[::-1]))
    print(f"\n  The two cross at about 10^{crossing:.1f} GeV. Below that the sample is")
    print("  atmospheric; above it, astrophysical.")

    widths = np.gradient(energy)
    atmospheric_rate = atmospheric * aeff * LIVETIME_S * SOLID_ANGLE_SR * widths
    astrophysical_rate = astrophysical * aeff * LIVETIME_S * SOLID_ANGLE_SR * widths
    print(f"\nUpgoing events in {LIVETIME_S / (365.25 * 86400.0):.0f} years, "
          "above a threshold:\n")
    print(f"{'above 10^x GeV':>15} {'atmospheric':>14} {'astrophysical':>15}")
    for log10_min in (4.0, 5.0, 6.0):
        keep = LOG10_E >= log10_min
        print(f"{log10_min:15.1f} {atmospheric_rate[keep].sum():14.1f} "
              f"{astrophysical_rate[keep].sum():15.1f}")
    print("\n  The counts are indicative: one flat effective area over the upgoing")
    print("  sky, no reconstruction and no selection beyond the muon threshold.")
    return atmospheric, astrophysical


def make_figure(flux: AtmosphericFlux, out_path: pathlib.Path) -> None:
    """Draw the atmospheric flux against energy for several declinations."""
    energy = 10.0**LOG10_E
    scaled = energy**2
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        for dec in DEC_DEG:
            values = np.ravel(flux(energy, np.full_like(energy, dec)))
            ax.plot(LOG10_E, scaled * values, lw=1.2,
                    label=rf"$\mathrm{{dec}} = {dec:.0f}^\circ$")
        ax.plot(LOG10_E, scaled * np.ravel(ICECUBE_TRACKS_2022.flux(energy)),
                color="0.4", lw=1.8, ls="--", label="astrophysical")
        ax.set_yscale("log")
        ax.set_ylim(1e-10, 1e-3)
        ax.set_xlabel(r"$\log_{10}(E_\nu/\mathrm{GeV})$")
        ax.set_ylabel(r"$E^2\phi$ [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
        ax.set_title("atmospheric against astrophysical")
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        for suffix in (".pdf", ".png"):
            fig.savefig(out_path.with_suffix(suffix))
            print(f"Figure saved to: {out_path.with_suffix(suffix)}")
        plt.close(fig)


def main() -> None:
    """Load or build the table, compare the two fluxes, and draw them."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.table.exists() and not args.rebuild:
        print(f"No cached MCEq table at {args.table}.")
        print("Install the atmospheric extra and rebuild it once:")
        print("    pip install -e '.[atm]'")
        print(f"    python {pathlib.Path(__file__).name} --rebuild")
        return

    print(f"Loading the MCEq table from {args.table} ...")
    flux = AtmosphericFlux(load_mceq_table(args.table, recompute=args.rebuild))
    print(f"  {flux.energy_gev.size} energies by {flux.dec_deg.size} declinations")

    print("Building an upgoing effective area to fold it through ...")
    theta_deg, weights = zenith_grid(N_ZENITH, (-1.0, 0.0))
    aeff = directional_effective_area_cm2(
        ICECUBE, np.cos(np.deg2rad(theta_deg)), args.threshold_gev, log10_e=LOG10_E
    ) @ weights

    report(flux, aeff)
    make_figure(flux, args.out_dir / "10_atmospheric_background")


if __name__ == "__main__":
    main()
