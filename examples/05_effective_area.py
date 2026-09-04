"""Tutorial 05 -- an effective area built from the transport.

An effective area is a target volume times a cross section times a nucleon
density. The target volume is what the transport gives: the projected area of
the instrumented body times the distance a muon can travel and still arrive
above threshold, plus the body itself. Nothing in that is specific to one
detector, so the same call serves every site in the registry.

This tutorial builds the sky-averaged effective area of four detectors and
compares IceCube's with its published table.

Usage
-----
    python examples/05_effective_area.py
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.data import dr2_dir, icecube_dr2_aeff
from softpaws.detectors import ARCA230, ICECUBE, PONE, TRIDENT
from softpaws.response.declination import directional_effective_area_cm2
from softpaws.transport.earth import zenith_grid

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"

#: Neutrino energies the curves are built on [log10 GeV].
LOG10_E = np.linspace(4.0, 8.0, 17)

#: Directions the sky average runs over.
N_ZENITH = 24

#: Sites compared, in the order they are drawn.
SITES = (ICECUBE, ARCA230, PONE, TRIDENT)


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threshold-gev", type=float, default=1.0e3,
                        help="Muon selection threshold.")
    parser.add_argument("--data-dir", type=pathlib.Path, default=None,
                        help="Root of the DR2 release, for the published comparison.")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the figure.")
    return parser.parse_args()


def sky_averaged(site, threshold_gev: float, hemisphere: str = "all") -> np.ndarray:
    """Effective area averaged over a hemisphere or the whole sky [cm^2].

    Parameters
    ----------
    site : softpaws.detectors.Site
        The detector.
    threshold_gev : float
        Muon selection threshold [GeV].
    hemisphere : {"all", "upgoing"}, optional
        ``"upgoing"`` keeps ``cos(theta) < 0``, which is the convention the
        DR2 table is quoted in.

    Returns
    -------
    aeff : np.ndarray
        Effective area on :data:`LOG10_E` [cm^2].
    """
    cos_range = (-1.0, 0.0) if hemisphere == "upgoing" else (-1.0, 1.0)
    theta_deg, weights = zenith_grid(N_ZENITH, cos_range)
    cos_theta = np.cos(np.deg2rad(theta_deg))
    per_direction = directional_effective_area_cm2(
        site, cos_theta, threshold_gev, log10_e=LOG10_E
    )
    return per_direction @ weights


def report(curves: dict[str, np.ndarray], published: np.ndarray | None) -> None:
    """Print the effective areas at three energies, and the published ratio."""
    print("Sky-averaged effective area [cm^2], from the instrumented footprint only:\n")
    print(f"{'log10(E/GeV)':>13}" + "".join(f"{name:>12}" for name in curves))
    for log10_e in (5.0, 6.0, 7.0):
        i = int(np.argmin(np.abs(LOG10_E - log10_e)))
        print(f"{log10_e:13.1f}" + "".join(f"{c[i]:12.3g}" for c in curves.values()))

    if published is None:
        return
    print("\nIceCube's upgoing model against the published DR2 table:\n")
    print(f"{'log10(E/GeV)':>13} {'published':>12} {'model':>12} {'published/model':>16}")
    model = curves["IceCube up"]
    for log10_e in (5.0, 6.0, 7.0):
        i = int(np.argmin(np.abs(LOG10_E - log10_e)))
        print(f"{log10_e:13.1f} {published[i]:12.3g} {model[i]:12.3g} "
              f"{published[i] / model[i]:16.2f}")
    band = (LOG10_E >= 5.0) & (LOG10_E <= 7.8)
    ratio = published[band] / model[band]
    print(f"\n  Over 10^5 to 10^7.8 GeV the ratio is {np.exp(np.mean(np.log(ratio))):.2f}")
    print(f"  with {np.std(np.log10(ratio)):.3f} dex of scatter, and nothing is fitted.")
    print("  The missing factor is the light the muon makes outside the array,")
    print("  which tutorial 06 puts back.")


def make_figure(curves: dict[str, np.ndarray], published: np.ndarray | None,
                out_path: pathlib.Path) -> None:
    """Draw the four sites and the published table."""
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.3, 3.2))
        if published is not None:
            ax.plot(LOG10_E, published, color="0.5", lw=2.2, zorder=1,
                    label="IceCube, published")
        for name, curve in curves.items():
            if name.endswith(" up"):
                continue
            ax.plot(LOG10_E, curve, lw=1.3, label=name, zorder=3)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\log_{10}(E_\nu/\mathrm{GeV})$")
        ax.set_ylabel(r"$A_{\rm eff}$ [cm$^2$]")
        ax.set_title("sky averaged, footprint only")
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        fig.tight_layout()
        for suffix in (".pdf", ".png"):
            fig.savefig(out_path.with_suffix(suffix))
            print(f"Figure saved to: {out_path.with_suffix(suffix)}")
        plt.close(fig)


def main() -> None:
    """Build the effective areas, report them and draw them."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    curves = {}
    for site in SITES:
        print(f"Building {site.name} ...")
        curves[site.name] = sky_averaged(site, args.threshold_gev)
    curves["IceCube up"] = sky_averaged(ICECUBE, args.threshold_gev, "upgoing")

    try:
        published = icecube_dr2_aeff(args.data_dir or dr2_dir(), LOG10_E)[0]
    except FileNotFoundError:
        print("\n(The DR2 release is not on disk, so the published comparison is skipped.)")
        published = None

    report(curves, published)
    make_figure(curves, published, args.out_dir / "05_effective_area")


if __name__ == "__main__":
    main()
