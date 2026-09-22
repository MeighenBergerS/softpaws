"""Example 70 -- the loss-model error bands on the effective areas.

Example 69 measured the kernel's cross-section error as an ensemble of
published parameterizations; this example propagates it to the paper's
central deliverables: the first-principles effective areas of example 45
(IceCube and ARCA230) and, through them, the point-source reach of
example 35's kind.

**Mechanics.** The transport enters example 45 through two names,
``drift_coefficient`` (the brightness factor and every range) and
``diffusion_coefficient``; both are patched -- in every loaded softpaws
module, so the library-internal aliases follow -- with the variant ratio
``kappa_1(E)`` (mean loss) and ``kappa_2(E)`` (second moment)
interpolated from example 69's ensemble. Example 45's *first-principles*
model is then rebuilt per variant: nothing is refitted, so the band is a
genuine prediction band, not a fit residual. The ``y``-moment and
log-loss-moment ratios differ by well under a percent, so one ratio
serves both.

**Outputs.** Figure 70a: the published curves against the baseline
prediction with its ensemble envelope, both detectors. Figure 70b: the
variant-to-baseline ratio of the effective areas. Two opposing effects
compose it: a faster-losing muon is *brighter* (light yield carries
``b``, pushing the area up a little) but *shorter-ranged* (the upstream
effective length carries ``1 / Phi'(0)``, pulling it down a lot), so
the range side dominates and the band tracks a diluted ``1 / kappa_1``
-- diluted because the range integrates the moments down the whole
energy ladder. Figure 70c: the band on a point-source flux limit,
``ratio = int A_base E^-gamma / int A_var E^-gamma`` above ``E_min``,
against the spectral index -- hard sources lean on the energies where
the photonuclear spread lives and carry the larger band.

Usage
-----
    python scripts/2026_muon_transport/70_aeff_error_bands.py
    python scripts/2026_muon_transport/70_aeff_error_bands.py --n-energy 120
"""

import argparse
import importlib.util
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.transport.coefficients import set_kernel_scaling
from softpaws.transport.loss_ensemble import variant_scaling

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


#: Point-source fold: spectral indices and the lower integration edge [GeV].
GAMMA_GRID = np.linspace(1.5, 3.5, 21)
PS_EMIN_GEV = 1.0e3

COLORS = {"published": "0.6", "base": "#e7298a", "band": "#e7298a",
          "arca": "#7570b3"}


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=pathlib.Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--n-energy", type=int, default=160,
                        help="Arrival-energy quadrature points per curve.")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the figures, '70a'-'70c'.")
    return parser.parse_args()


def install_patches() -> None:
    """Kept for the scripts that call it; the library hook needs no patching.

    The coefficients read :func:`softpaws.transport.coefficients.kernel_scaling`
    at call time, so :func:`set_variant` reaches every caller.
    """


def set_variant(ratios, label: str | None) -> None:
    """Activate one ensemble variant (``None`` restores the baseline)."""
    set_kernel_scaling(variant_scaling(ratios, label, _EX69.E_GRID))


#: Example 69 supplies the ensemble (from its cache); patches go in before
#: example 45 is loaded, so its from-imports bind the wrapped functions.
_EX69 = load_example("69_loss_model_error_budget.py", "_example_69")
install_patches()
_EX45 = load_example("45_first_principles_reach.py", "_example_45")


def build_curves(ex32, args, flavours=("mu", "tau")) -> dict:
    """First-principles effective areas per variant, both detectors."""
    _, ratios = _EX69.load_ensemble(False)
    published = {"IceCube": ex32.icecube_published(args.data_dir),
                 "ARCA": ex32.arca230_published(ex32.ARCA_LOG10_E)}
    sites = {"IceCube": _EX45.ICECUBE_SITE, "ARCA": _EX45.ARCA_SITE}
    curves = {which: {} for which in sites}
    for label in [None, *_EX69.VARIANTS]:
        set_variant(ratios, label)
        for which, site in sites.items():
            curves[which][label or "baseline"] = _EX45.build_model(
                ex32, which, site, _EX45.DEFAULT_MIN_MODULES, args.n_energy, flavours)
            print(f"  {label or 'baseline':>14} {which:>8} built")
    set_variant(ratios, None)
    return curves, published


def _save(fig, out_dir: pathlib.Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {path.resolve()}")
    plt.close(fig)


def figure_areas(ex32, curves, published, out_dir) -> None:
    """Figure 70a: published against the banded first-principles prediction."""
    grids = {"IceCube": ex32.IC_LOG10_E, "ARCA": ex32.ARCA_LOG10_E}
    with plt.style.context(str(_STYLE)):
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=False)
        for ax, which in zip(axes, grids, strict=True):
            log10_e = grids[which]
            stack = np.array([v for k, v in curves[which].items() if k != "baseline"])
            ax.fill_between(10.0**log10_e, stack.min(axis=0), stack.max(axis=0),
                            color=COLORS["band"], alpha=0.25, lw=0,
                            label="loss-model envelope")
            ax.plot(10.0**log10_e, curves[which]["baseline"], color=COLORS["base"],
                    lw=1.1, label="first principles (baseline)")
            ax.plot(10.0**log10_e, published[which], color=COLORS["published"], lw=2.0,
                    alpha=0.7, label="published", zorder=1)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(10.0**4.0, 10.0**9.0)
            ax.set_xlabel(r"$E_\nu$ [GeV]")
            ax.set_title("IceCube" if which == "IceCube" else "KM3NeT/ARCA230",
                         fontsize=7)
        axes[0].set_ylabel(r"$A_{\rm eff}$ [cm$^2$]")
        axes[0].legend(fontsize=5.2, frameon=False, loc="upper left")
        _save(fig, out_dir, "70a_aeff_bands")


def figure_ratio(ex32, curves, out_dir) -> None:
    """Figure 70b: variant-to-baseline effective-area ratios."""
    grids = {"IceCube": (ex32.IC_LOG10_E, "-"), "ARCA": (ex32.ARCA_LOG10_E, "--")}
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.5, 3.0))
        for which, (log10_e, ls) in grids.items():
            base = curves[which]["baseline"]
            with np.errstate(invalid="ignore", divide="ignore"):
                stack = np.array([v / base for k, v in curves[which].items()
                                  if k != "baseline"])
            label = "IceCube" if which == "IceCube" else "ARCA230"
            color = COLORS["base"] if which == "IceCube" else COLORS["arca"]
            ax.fill_between(10.0**log10_e, np.nanmin(stack, axis=0),
                            np.nanmax(stack, axis=0), color=color, alpha=0.18, lw=0)
            for row in stack:
                ax.plot(10.0**log10_e, row, color=color, lw=0.6, ls=ls, alpha=0.6)
            ax.plot([], [], color=color, ls=ls, label=label)
        ax.axhline(1.0, color="0.2", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlim(10.0**4.0, 10.0**9.0)
        ax.set_xlabel(r"$E_\nu$ [GeV]")
        ax.set_ylabel(r"$A_{\rm eff}$ variant / baseline")
        ax.legend(fontsize=5.5, frameon=False, loc="upper left")
        _save(fig, out_dir, "70b_aeff_ratio_bands")


def figure_point_source(ex32, curves, out_dir) -> None:
    """Figure 70c: the band a point-source flux limit inherits."""
    log10_e = ex32.IC_LOG10_E
    energy = 10.0**log10_e
    keep = energy >= PS_EMIN_GEV
    base = curves["IceCube"]["baseline"]
    ratios = {}
    for label, aeff in curves["IceCube"].items():
        if label == "baseline":
            continue
        row = np.empty(GAMMA_GRID.size)
        for i, gamma in enumerate(GAMMA_GRID):
            weight = energy[keep] ** (1.0 - gamma)  # per ln E
            row[i] = (np.trapezoid(base[keep] * weight, log10_e[keep])
                      / np.trapezoid(aeff[keep] * weight, log10_e[keep]))
        ratios[label] = row
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.4, 2.9))
        stack = np.array(list(ratios.values()))
        ax.fill_between(GAMMA_GRID, stack.min(axis=0), stack.max(axis=0),
                        color=COLORS["band"], alpha=0.2, lw=0, label="envelope")
        for label, row in ratios.items():
            ax.plot(GAMMA_GRID, row, lw=0.8, alpha=0.8,
                    label=f"{label} ({_EX69.VARIANTS[label][1]})")
        ax.axhline(1.0, color="0.2", lw=0.8)
        ax.set_xlabel(r"source spectral index $\gamma$")
        ax.set_ylabel("flux-limit ratio, variant / baseline")
        ax.legend(fontsize=4.4, frameon=False, loc="upper right")
        _save(fig, out_dir, "70c_point_source_band")
    return ratios


def main() -> None:
    args = parse_args()
    print("Loading example 32 (published curves) ...")
    ex32 = _EX45.load_example_32()
    print("Building the first-principles areas per ensemble variant ...")
    curves, published = build_curves(ex32, args)

    print("\nEffective-area envelope (variant/baseline) at the anchors:")
    for which in ("IceCube", "ARCA"):
        log10_e = ex32.IC_LOG10_E if which == "IceCube" else ex32.ARCA_LOG10_E
        base = curves[which]["baseline"]
        stack = np.array([v for k, v in curves[which].items() if k != "baseline"])
        cells = []
        for target in (5.0, 6.0, 7.0, 8.0):
            i = int(np.argmin(np.abs(log10_e - target)))
            if base[i] > 0.0:
                lo, hi = stack[:, i].min() / base[i], stack[:, i].max() / base[i]
                cells.append(f"1e{target:.0f}: {lo:.3f}-{hi:.3f}")
        print(f"  {which:>8}: " + "  ".join(cells))

    figure_areas(ex32, curves, published, args.out_dir)
    figure_ratio(ex32, curves, args.out_dir)
    ratios = figure_point_source(ex32, curves, args.out_dir)
    stack = np.array(list(ratios.values()))
    i2 = int(np.argmin(np.abs(GAMMA_GRID - 2.0)))
    print(f"\nPoint-source flux-limit band at gamma = 2: "
          f"{stack[:, i2].min():.3f} - {stack[:, i2].max():.3f} of the baseline limit.")


if __name__ == "__main__":
    main()
