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
import types

import matplotlib.pyplot as plt
import numpy as np

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_DATA_DIR = _HERE.parent / "src" / "softpaws" / "data" / "dataverse_files"
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
    data = np.load(chains_path)
    detectors = [d for d in _EX56.build_detectors(data_dir) if d.name != "TRIDENT"]
    for d in detectors:
        fixed = {"eps_0": _EX77.EPS_FIXED[d.name], "log10_e_thr": 3.0, "b_scale": 1.0,
                 "lam": _EX33.LAMBDA_BGR18, "reach_km": 0.03}
        two = data[f"{d.name}_2p_chain"]
        d.chain = np.array([_EX77.full_theta(_EX77.FREE2, row, fixed) for row in two])
    return detectors


def trident_2025(trident_chain_path):
    """The 2025 map averaged over ``|cos| <= COS_MAX`` and its posterior."""
    cos_c, le_all, la_all = _EX81.load_map()
    keep = le_all >= _EX33.ARCA_LOG10_E.min()
    log10_e, log10_a = le_all[keep], la_all[:, keep]
    edges = _EX81.COS_EDGES
    dcos = np.abs(np.diff(edges))
    rows = np.abs(cos_c) <= COS_MAX
    area = 10 ** log10_a                                     # [m^2]
    fitted = np.average(area[rows], axis=0, weights=dcos[rows])
    allsky = np.average(area, axis=0, weights=dcos)
    # The map's cells carry their simulation's statistics as a checkerboard of
    # a few hundredths of a dex, and the eight-point average inherits a kink
    # near 10^5.3 GeV. A quadratic in log-log is the smoothest curve with the
    # right curvature over two decades; the residual it removes is reported.
    coeff = np.polyfit(log10_e, np.log10(fitted), 2)
    smoothed = 10 ** np.polyval(coeff, log10_e)
    print("TRIDENT 2025 average smoothed by a log-log quadratic; residual rms "
          f"{np.std(np.log10(fitted / smoothed)):.3f} dex, max {np.max(np.abs(np.log10(fitted / smoothed))):.3f} dex")
    fitted = smoothed
    model = _EX81.MapModel(log10_e)
    weights = [w for w, r in zip(model.weights, rows) if r]

    def predict(theta, select=None):
        curves = np.empty((len(weights), log10_e.size))
        for i, w in enumerate(weights):
            full = _EX56.water_model(theta, model.site, model.ladders, w, model.mcol, None)
            curves[i] = 10 ** np.interp(log10_e, model.grid, np.log10(full))
        pred = np.average(curves, axis=0, weights=dcos[rows])
        return pred if select is None else pred[select]

    chain = np.load(trident_chain_path)["chain"]
    fixed = {"eps_0": 0.7, "log10_e_thr": 2.5, "b_scale": 1.0, "lam": _EX33.LAMBDA_BGR18,
             "reach_km": 0.03}
    full = np.array([_EX77.full_theta(_EX77.FREE3, row, fixed) for row in chain])
    detector = types.SimpleNamespace(
        name="TRIDENT", log10_e=log10_e, observed=1.0e4 * fitted, mask=np.ones(log10_e.size, bool),
        predict=predict, chain=full, color=_EX56.COLORS["TRIDENT"])
    return detector, 1.0e4 * allsky


def figure(detectors, out_dir) -> None:
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        medians = {}
        for d in detectors:
            observed = np.asarray(d.observed, dtype=float)
            good = np.isfinite(observed) & (observed > 0.0)
            ax.plot(d.log10_e[good], observed[good], color="k", lw=1.2)
            med, lo, hi = _EX73.posterior_band(d, N_DRAWS)
            ax.fill_between(d.log10_e[good], lo[good], hi[good], color=d.color, alpha=0.45, lw=0)
            ax.plot(d.log10_e[good], med[good], color=d.color, lw=1.1, ls="--")
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
            ax.text(x0, height, d.name, color=d.color, ha=ha, va="center", rotation=angle,
                    rotation_mode="anchor")
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
