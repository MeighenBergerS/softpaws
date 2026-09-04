"""Tutorial 08 -- the tracks a tau neutrino makes, and why they are not optional.

A through-going track is usually read as a muon neutrino. It need not be. A
tau neutrino makes a tau, the tau decays, and about one time in six the decay
gives a muon that crosses the detector looking like any other track. At the
energies where the Earth is opaque to muon neutrinos this channel is not a
correction, because the tau regenerates on the way through and the muon one
does not.

This tutorial computes the tau contribution to the track rate against energy
and arrival direction, and says what share of the tracks it carries.

Usage
-----
    python examples/08_tau_induced_tracks.py
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.detectors import ICECUBE
from softpaws.response.declination import directional_effective_area_cm2
from softpaws.transport.tau import BR_TAU_TO_MU, MEAN_Z, decay_length_km, z_moment

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"

#: Neutrino energies the curves are built on [log10 GeV].
LOG10_E = np.linspace(4.0, 8.0, 17)

#: Arrival directions compared; ``-1`` is the nadir.
COS_THETA = np.array([-0.99, -0.7, -0.4, -0.1])


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threshold-gev", type=float, default=1.0e3,
                        help="Muon selection threshold.")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the figure.")
    return parser.parse_args()


def report_tau_properties() -> None:
    """Print what the tau brings to the track channel."""
    print("The tau, as the track channel sees it:\n")
    print(f"  branching to muons          {BR_TAU_TO_MU:.3f}")
    print(f"  mean muon energy fraction   {MEAN_Z:.2f}")
    print(f"  Z moment at A = 1           {float(np.squeeze(z_moment(1.0))):.3f}")
    print(f"\n{'log10(E/GeV)':>13} {'decay length [km]':>19}")
    for log10_e in (6.0, 7.0, 8.0, 9.0):
        length = float(np.squeeze(decay_length_km(10.0**log10_e)))
        print(f"{log10_e:13.1f} {length:19.3g}")
    print("\n  Above about 10^8 GeV the tau outruns the detector, and the channel")
    print("  stops being a prompt one.")


def build(threshold_gev: float) -> dict[str, np.ndarray]:
    """Effective area with and without the tau channel, per direction."""
    curves = {}
    for channels, name in (("mu", "nu_mu only"), ("both", "with tau")):
        print(f"Building {name} ...")
        curves[name] = directional_effective_area_cm2(
            ICECUBE, COS_THETA, threshold_gev, channels=channels, log10_e=LOG10_E
        )
    return curves


def report_share(curves: dict[str, np.ndarray]) -> np.ndarray:
    """Print and return the tau share of the track rate, per energy and direction."""
    share = 1.0 - curves["nu_mu only"] / curves["with tau"]
    print("\nFraction of the effective area the tau channel carries:\n")
    header = f"{'log10(E/GeV)':>13}" + "".join(f"{f'cos={c:+.2f}':>10}" for c in COS_THETA)
    print(header)
    for log10_e in (5.0, 6.0, 7.0, 8.0):
        i = int(np.argmin(np.abs(LOG10_E - log10_e)))
        print(f"{log10_e:13.1f}" + "".join(f"{v:10.1%}" for v in share[i]))
    print("\n  Near the horizon the tau carries a tenth or so, roughly flat in")
    print("  energy. Toward the nadir, where the Earth has removed the muon")
    print("  neutrinos, it takes over: above a PeV almost everything that")
    print("  arrives from below came in as a tau.")
    return share


def make_figure(share: np.ndarray, out_path: pathlib.Path) -> None:
    """Draw the tau share against energy for each arrival direction."""
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        for j, cos_theta in enumerate(COS_THETA):
            ax.plot(LOG10_E, share[:, j], lw=1.3,
                    label=rf"$\cos\theta = {cos_theta:+.2f}$")
        ax.set_xlabel(r"$\log_{10}(E_\nu/\mathrm{GeV})$")
        ax.set_ylabel("tau share of the track rate")
        ax.set_ylim(0.0, 1.0)
        ax.set_title("IceCube, upgoing")
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        for suffix in (".pdf", ".png"):
            fig.savefig(out_path.with_suffix(suffix))
            print(f"Figure saved to: {out_path.with_suffix(suffix)}")
        plt.close(fig)


def main() -> None:
    """Report the tau channel and draw its share."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_tau_properties()
    curves = build(args.threshold_gev)
    share = report_share(curves)
    make_figure(share, args.out_dir / "08_tau_induced_tracks")


if __name__ == "__main__":
    main()
