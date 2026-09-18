"""Example 73 -- four effective areas: published bands against the fitted model.

The corner plots of examples 33/56/72 summarize the fits in parameter
space; this example shows the same fits where the data live. For each of
the four sites -- IceCube, ARCA230, P-ONE and TRIDENT -- the published
(or digitized) effective area is drawn as a band of its assumed
fractional error (the same 15% the likelihood uses), and the model's
posterior predictive under example 72's ensemble-informed ``b_scale``
prior is drawn over it: the median curve and the 68% band from the
chain.

The two bands answer different questions. The published band is what the
fit was allowed to miss by; the posterior band is what the model, with
its transport priors from the loss ensemble, actually spans. Where the
posterior band sits inside the published one across the fit window, the
site is described; the fit windows are marked, and outside them the
curves are extrapolation.

Usage
-----
    python scripts/2026_muon_transport/73_four_detector_aeff_fit.py
    python scripts/2026_muon_transport/73_four_detector_aeff_fit.py --steps 4000
"""

import argparse
import importlib.util
import pathlib

import matplotlib.pyplot as plt
import numpy as np

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_DATA_DIR = _HERE.parents[1] / "src" / "softpaws" / "data" / "dataverse_files"
_DEFAULT_OUT_DIR = _HERE / "output"


def load_example(stem: str, name: str):
    """Import a sibling example module by file stem."""
    spec = importlib.util.spec_from_file_location(name, _HERE / stem)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EX72 = load_example("72_informed_transport_corner.py", "_example_72")
_EX56 = _EX72._EX56
_EX33 = _EX72._EX33

#: Posterior-predictive draws per site.
N_DRAWS = 150

COLORS = {"published": "0.45", "model": "#e7298a"}


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=pathlib.Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--walkers", type=int, default=40)
    parser.add_argument("--sigma", type=float, default=0.15,
                        help="Assumed fractional error on the observed areas.")
    parser.add_argument("--resample", action="store_true",
                        help="Refit even when the chain cache exists.")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the figure, '73a'.")
    return parser.parse_args()


def posterior_band(detector, n_draws: int, seed: int = 7) -> tuple:
    """Median and 68% band of the predicted area over the full grid."""
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, detector.chain.shape[0], size=n_draws)
    curves = []
    for k in picks:
        predicted = detector.predict(detector.chain[k], None)
        curves.append(np.asarray(predicted, dtype=float))
    stack = np.array(curves)
    with np.errstate(invalid="ignore"):
        return (np.nanmedian(stack, axis=0), np.nanpercentile(stack, 16, axis=0),
                np.nanpercentile(stack, 84, axis=0))


def _save(fig, out_dir: pathlib.Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {path.resolve()}")
    plt.close(fig)


def _curve_angle_deg(ax, x: np.ndarray, y: np.ndarray, x0: float) -> float:
    """Screen-space slope of a curve at ``x0``, for a label that follows it."""
    x1 = x0 + 0.25
    points = ax.transData.transform(
        np.column_stack([[x0, x1], np.interp([x0, x1], x, y)]))
    return float(np.degrees(np.arctan2(points[1, 1] - points[0, 1],
                                       points[1, 0] - points[0, 0])))


#: Label placement per site: text energy [log10 GeV], vertical factor on the
#: median, horizontal alignment, the curve whose slope the text follows
#: (``None`` = horizontal), the curve the factor applies to, and the energy
#: the height is read at (``None`` = the text energy).
LABEL_SPEC = {
    "IceCube": (4.0, 0.40, "center", "IceCube", "IceCube", 3.8),  # under its line
    "ARCA230": (4.5, 1.55, "center", "ARCA230", "ARCA230", None),  # ARCA-TRIDENT gap
    "P-ONE": (6.9, 1.0, "left", "ARCA230", "P-ONE", None),        # line end, ARCA slope
    "TRIDENT": (5.3, 1.55, "center", "TRIDENT", "TRIDENT", None),  # above, following it
}


def figure_single(detectors, sigma: float, out_dir) -> None:
    """All four sites on one axes, each in its own color.

    Per detector: the published area as a light band of its assumed
    fractional error, the model as the dashed posterior median with its
    68% band. The styles separate published from model; the colors
    separate the sites, with each name set beside its own curve.
    """
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        medians = {}
        for detector in detectors:
            log10_e = detector.log10_e
            observed = np.asarray(detector.observed, dtype=float)
            good = np.isfinite(observed) & (observed > 0.0)
            # The published curve as a black line: the agreement is tight
            # enough that a band on both sides hides the comparison.
            ax.plot(log10_e[good], observed[good], color="k", lw=1.2)
            # The model is drawn only where a published value exists: beyond
            # the digitized grids the predictive is unanchored extrapolation
            # (and single pathological draws can swamp the percentile band).
            med, lo, hi = posterior_band(detector, N_DRAWS)
            ax.fill_between(log10_e[good], lo[good], hi[good], color=detector.color,
                            alpha=0.45, lw=0)
            ax.plot(log10_e[good], med[good], color=detector.color, lw=1.1, ls="--")
            medians[detector.name] = (log10_e[good], med[good])
        ax.plot([], [], color="k", lw=1.2, label="Published")
        ax.fill_between([], [], [], color="0.4", alpha=0.45, label="Model")
        ax.set_yscale("log")
        ax.set_xlim(3.0, 8.0)
        ax.set_ylim(1.0e3, 2.0e8)
        ax.set_xlabel(r"$\log_{10}(E_\nu\,/\,\mathrm{GeV})$")
        ax.set_ylabel(r"$A_{\rm eff}$ [cm$^2$]")
        ax.set_box_aspect(1)
        ax.legend(frameon=False, loc="lower right")
        # Draw once to freeze the transform, then set each name on its curve.
        fig.canvas.draw()
        for detector in detectors:
            x0, factor, ha, slope_of, anchor_on, y_at = LABEL_SPEC[detector.name]
            angle = (_curve_angle_deg(ax, *medians[slope_of], x0)
                     if slope_of is not None else 0.0)
            x, y = medians[anchor_on]
            height = factor * float(np.interp(y_at if y_at is not None else x0, x, y))
            ax.text(x0, height, detector.name, color=detector.color,
                    ha=ha, va="center", rotation=angle, rotation_mode="anchor")
        _save(fig, out_dir, "73a_four_detector_aeff")


def main() -> None:
    args = parse_args()
    implied = _EX72.implied_b_scales()
    sigma_ens = float(np.array(list(implied.values())).std())
    print(f"Informed b_scale prior from the loss ensemble: N(1, {sigma_ens:.3f})")
    _EX33.B_SCALE_MEAN, _EX33.B_SCALE_STD = 1.0, sigma_ens
    detectors = _EX56.build_detectors(args.data_dir)
    cache = args.out_dir / "73_chains.npz"
    if cache.exists() and not args.resample:
        print(f"  cached chains from {cache.name}")
        data = np.load(cache)
        for detector in detectors:
            detector.chain = data[f"{detector.name}_chain"]
            detector.best = data[f"{detector.name}_best"]
    else:
        for seed, detector in enumerate(detectors, start=11):
            _EX33.run_fit(detector, args.steps, args.walkers, args.sigma, seed)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache,
                            **{f"{d.name}_chain": d.chain for d in detectors},
                            **{f"{d.name}_best": d.best for d in detectors})
        print(f"  chains cached to {cache.name}")

    print("\nResidual check inside each fit window (published / posterior median):")
    for detector in detectors:
        med, _, _ = posterior_band(detector, N_DRAWS)
        observed = np.asarray(detector.observed, dtype=float)
        ratio = observed[detector.mask] / med[detector.mask]
        good = np.isfinite(ratio) & (ratio > 0.0)
        print(f"  {detector.name:>10}: geometric mean {10**np.mean(np.log10(ratio[good])):.3f}, "
              f"scatter {np.std(np.log10(ratio[good])):.3f} dex")

    figure_single(detectors, args.sigma, args.out_dir)


if __name__ == "__main__":
    main()
