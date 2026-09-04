"""Example 83 -- example 73's data-space figure on the reduced response.

Example 73 draws the four published effective areas against the
five-parameter posterior predictive. This example redraws it with the
physics fixed (``b_scale = 1``, ``lam`` at BGR18) and only the instrument
numbers of example 77 sampled, and it replaces TRIDENT's 2022 sky average
by the 2025 map of Morton-Blake et al. (arXiv:2510.24395, Fig. 3a),
averaged over the cells where their selection is flat, ``|cos| <= 0.5``,
with the posterior of example 82 (selection normalization, threshold and
reach). The three other sites carry example 77's 5% chains with their
selection normalization fixed. The all-sky average of the 2025 map is
printed beside the fitted one, so the cost of the nadir selection is on
record.

Usage
-----
    python examples/83_four_detector_aeff_reduced.py
"""

import argparse
import importlib.util
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.response import reduced

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_DATA_DIR = _HERE.parents[1] / "src" / "softpaws" / "data" / "dataverse_files"
_DEFAULT_OUT_DIR = _HERE / "output"


def load_example(stem: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / stem)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EX73 = load_example("73_four_detector_aeff_fit.py", "_example_73")
_EX81 = load_example("81_trident_2025_map_fit.py", "_example_81")
_EX77 = _EX81._EX77
_EX56 = _EX77._EX56
_EX33 = _EX77._EX33

COS_MAX = 0.5
N_DRAWS = 200

LABEL_SPEC = {
    "IceCube": (4.6, 0.55, "center", "IceCube", "IceCube", None),  # under its line
    "ARCA230": (4.5, 1.55, "center", "ARCA230", "ARCA230", None),
    "P-ONE": (6.9, 1.0, "left", "ARCA230", "P-ONE", None),
    "TRIDENT": (4.75, 1.75, "center", "TRIDENT", "TRIDENT", None),  # above, following it
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=pathlib.Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--chains", type=pathlib.Path,
                        default=_DEFAULT_OUT_DIR / "77_chains_sigma05.npz")
    parser.add_argument("--trident-chain", type=pathlib.Path,
                        default=_DEFAULT_OUT_DIR / "82_trident2025_chain.npz")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR)
    return parser.parse_args()


def reduced_detectors(data_dir, chains_path):
    """Example 56's IceCube, ARCA230 and P-ONE with example 77's two-parameter chains."""
    detectors = [d for d in _EX56.build_detectors(data_dir) if d.name != "TRIDENT"]
    return reduced.attach_reduced_chains(detectors, chains_path)


def trident_2025(trident_chain_path):
    """The 2025 map averaged over ``|cos| <= COS_MAX`` and its posterior."""
    detector, allsky, smoothing = reduced.trident_2025_average_detector(
        trident_chain_path, COS_MAX
    )
    print("TRIDENT 2025 average smoothed by a log-log quadratic; residual rms "
          f"{smoothing['rms']:.3f} dex, max {smoothing['max']:.3f} dex")
    return detector, allsky


def figure(detectors, out_dir) -> None:
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        medians = {}
        for d in detectors:
            observed = np.asarray(d.observed, dtype=float)
            good = np.isfinite(observed) & (observed > 0.0)
            ax.plot(d.log10_e[good], observed[good], color="k", lw=1.2)
            med, lo, hi = _EX73.posterior_band(d, N_DRAWS)
            color = _EX56.SITE_COLORS[d.name]
            ax.fill_between(d.log10_e[good], lo[good], hi[good], color=color, alpha=0.45, lw=0)
            ax.plot(d.log10_e[good], med[good], color=color, lw=1.1, ls="--")
            medians[d.name] = (d.log10_e[good], med[good])
        ax.plot([], [], color="k", lw=1.2, label="Published")
        ax.fill_between([], [], [], color="0.4", alpha=0.45, label="Model")
        ax.set_yscale("log")
        ax.set_xlim(4.0, 8.0)
        ax.set_ylim(5.0e4, 2.0e8)
        ax.set_xlabel(r"$\log_{10}(E_\nu\,/\,\mathrm{GeV})$")
        ax.set_ylabel(r"$A_{\rm eff}$ [cm$^2$]")
        ax.set_box_aspect(1)
        ax.legend(frameon=False, loc="lower right")
        fig.canvas.draw()
        for d in detectors:
            x0, factor, ha, slope_of, anchor_on, y_at = LABEL_SPEC[d.name]
            angle = _EX73._curve_angle_deg(ax, *medians[slope_of], x0) if slope_of else 0.0
            x, y = medians[anchor_on]
            height = factor * float(np.interp(y_at if y_at is not None else x0, x, y))
            ax.text(x0, height, d.name, color=_EX56.SITE_COLORS[d.name], ha=ha, va="center",
                    rotation=angle, rotation_mode="anchor")
        _EX73._save(fig, out_dir, "83a_four_detector_aeff_reduced")


def main() -> None:
    args = parse_args()
    detectors = reduced_detectors(args.data_dir, args.chains)
    trident, allsky = trident_2025(args.trident_chain)
    detectors.append(trident)
    print("\nTRIDENT 2025 map averages [m^2]:")
    print(f"  {'log10E':>7} {'|cos|<=0.5':>11} {'all-sky':>9} {'ratio':>6}")
    for e, f, a in zip(trident.log10_e, trident.observed / 1e4, allsky / 1e4):
        print(f"  {e:7.2f} {f:11.1f} {a:9.1f} {a/f:6.3f}")
    print("\nResidual inside each fit window (published / posterior median):")
    for d in detectors:
        med, _, _ = _EX73.posterior_band(d, N_DRAWS)
        observed = np.asarray(d.observed, dtype=float)
        ratio = observed[d.mask] / med[d.mask]
        good = np.isfinite(ratio) & (ratio > 0.0)
        print(f"  {d.name:>10}: geometric mean {10**np.mean(np.log10(ratio[good])):.3f}, "
              f"scatter {np.std(np.log10(ratio[good])):.3f} dex")
    figure(detectors, args.out_dir)


if __name__ == "__main__":
    main()
