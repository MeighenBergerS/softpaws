"""Example 72 -- the four-site transport fit, informed by the loss-model error.

Two questions example 69's ensemble puts to the multi-detector fits of
examples 33 and 56. First, sensitivity: the fitted ``b_scale`` posteriors
(four sites, combined 0.91 +- 0.07) have always carried an ad-hoc prior,
``N(0.94, 0.15)`` -- can the detectors actually tell the published
photonuclear parameterizations apart? Each variant implies an effective
``b_scale`` (the log-average of ``kappa_1`` over the fit window,
``10^5``-``10^7`` GeV); its z-score under each site's posterior is the
answer. Second, the informed rerun: the ensemble is the *right* prior --
centred on the baseline table with the spread the published models
actually span -- so the corner is resampled under ``N(1, sigma_ens)``
and the four-way compatibility, the combined transport numbers and the
marginals are re-reported on defensible footing.

The honest caveat carries over from the single-detector lesson: an
effective-area level is degenerate with the efficiency scale, so much of
each ``b_scale`` posterior is its prior. The informative content is in
the *shifts* between sites and between priors, and in whether any
variant's implied value is pushed to the posterior's edge -- both are
quantified in the printout rather than eyeballed.

Usage
-----
    python examples/72_informed_transport_corner.py
    python examples/72_informed_transport_corner.py --steps 4000
"""

import argparse
import importlib.util
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.transport.loss_ensemble import implied_scales

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_DATA_DIR = _HERE.parent / "src" / "softpaws" / "data" / "dataverse_files"
_DEFAULT_OUT_DIR = _HERE / "output"


def load_example(stem: str, name: str):
    """Import a sibling example module by file stem."""
    spec = importlib.util.spec_from_file_location(name, _HERE / stem)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EX69 = load_example("69_loss_model_error_budget.py", "_example_69")
_EX56 = load_example("56_four_detector_posterior_corner.py", "_example_56")
_EX33 = _EX56._EX33

#: Fit window of the implied ``b_scale`` [log10 GeV] and the chain column.
IMPLIED_WINDOW = (5.0, 7.0)
B_SCALE_INDEX = _EX33.PARAM_NAMES.index("b_scale")

COLORS = {"default": "0.55", "informed": "#e7298a"}


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=pathlib.Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--walkers", type=int, default=40)
    parser.add_argument("--sigma", type=float, default=0.15,
                        help="Assumed fractional error on the observed areas.")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the figures, '72a' and '72b'.")
    return parser.parse_args()


def implied_b_scales() -> dict:
    """Effective ``b_scale`` each ensemble variant implies over the window."""
    _, ratios = _EX69.load_ensemble(False)
    return implied_scales(ratios, _EX69.E_GRID, IMPLIED_WINDOW)


def run_corner(args, prior_mean: float, prior_std: float, seed0: int) -> list:
    """Example 56's four fits under one ``b_scale`` prior."""
    _EX33.B_SCALE_MEAN, _EX33.B_SCALE_STD = prior_mean, prior_std
    detectors = _EX56.build_detectors(args.data_dir)
    for seed, detector in enumerate(detectors, start=seed0):
        _EX33.run_fit(detector, args.steps, args.walkers, args.sigma, seed)
    return detectors


def posterior_stats(detectors) -> dict:
    """Mean and standard deviation of each site's ``b_scale`` marginal."""
    return {d.name: (float(d.chain[:, B_SCALE_INDEX].mean()),
                     float(d.chain[:, B_SCALE_INDEX].std())) for d in detectors}


def _save(fig, out_dir: pathlib.Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {path.resolve()}")
    plt.close(fig)


def figure_b_scale(stats, implied, priors, out_dir) -> None:
    """Figure 72a: per-site ``b_scale`` under both priors, variants overlaid."""
    names = list(next(iter(stats.values())).keys())
    x = np.arange(len(names))
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.6, 3.0))
        for k, (tag, site_stats) in enumerate(stats.items()):
            mean = [site_stats[n][0] for n in names]
            std = [site_stats[n][1] for n in names]
            ax.errorbar(x + 0.12 * (k - 0.5), mean, yerr=std, fmt="o", ms=3.2,
                        lw=1.0, capsize=2, color=COLORS[tag],
                        label=f"{tag} prior N({priors[tag][0]:g}, {priors[tag][1]:.2f})")
        for label, value in implied.items():
            channel = _EX69.VARIANTS[label][0]
            ls = {"bremsstrahlung": "-", "pair production": "-",
                  "photonuclear": "--"}[channel]
            ax.axhline(value, color="0.65", lw=0.6, ls=ls)
            ax.text(len(names) - 0.45, value, label, fontsize=4.0, color="0.4",
                    va="center", ha="left")
        ax.axhline(1.0, color="0.2", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=6)
        ax.set_ylabel(r"$b_\mu$ scale")
        ax.legend(fontsize=5.2, frameon=False, loc="lower left")
        _save(fig, out_dir, "72a_b_scale_informed")


def main() -> None:
    args = parse_args()
    implied = implied_b_scales()
    values = np.array(list(implied.values()))
    sigma_ens = float(values.std())
    print("Implied b_scale per variant (log-average of kappa_1 over "
          f"1e{IMPLIED_WINDOW[0]:g}-1e{IMPLIED_WINDOW[1]:g} GeV):")
    for label, value in implied.items():
        print(f"  {label:>14}: {value:.3f}")
    print(f"  ensemble spread: std {sigma_ens:.3f}, half-range "
          f"{(values.max() - values.min()) / 2.0:.3f}")
    priors = {"default": (0.94, 0.15), "informed": (1.0, sigma_ens)}

    stats, four_way = {}, {}
    detectors_informed = None
    for tag, (mean, std) in priors.items():
        print(f"\n=== {tag} prior: b_scale ~ N({mean:g}, {std:.3f}) ===")
        detectors = run_corner(args, mean, std, seed0=11)
        stats[tag] = posterior_stats(detectors)
        four_way[tag] = _EX56.four_way_compatibility(detectors)
        if tag == "informed":
            detectors_informed = detectors
        for name, (m, s) in stats[tag].items():
            print(f"  {name:>10}: b_scale {m:.3f} +- {s:.3f} "
                  f"(shrinkage vs prior {s / std:.2f})")
        combined = four_way[tag]["combined_mean"]
        sigma_c = four_way[tag]["combined_sigma"]
        print(f"  combined: b {combined['b_scale']:.3f} +- {sigma_c['b_scale']:.3f}, "
              f"lam {combined['lam']:.3f} +- {sigma_c['lam']:.3f}; global "
              f"{four_way[tag]['global']['sigma']:.2f} sigma")

    print("\nCan the sites tell the photonuclear models apart? z-scores of the")
    print("implied b_scale under each informed posterior:")
    names = list(stats["informed"].keys())
    header = f"  {'variant':>14}" + "".join(f" {n:>10}" for n in names)
    print(header)
    for label, value in implied.items():
        row = "".join(f" {(value - stats['informed'][n][0]) / stats['informed'][n][1]:10.2f}"
                      for n in names)
        print(f"  {label:>14}" + row)
    print("  (|z| beyond ~2 with shrinkage well below 1 would be discrimination; "
          "shrinkage near 1 means the posterior is the prior)")

    figure_b_scale(stats, implied, priors, args.out_dir)
    _EX56.make_figure(detectors_informed,
                      args.out_dir / "72b_informed_corner.pdf")


if __name__ == "__main__":
    main()
