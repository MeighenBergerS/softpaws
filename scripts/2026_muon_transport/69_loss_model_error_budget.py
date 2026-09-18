"""Example 69 -- the loss model's own error, energy resolved.

The kernel's inputs are PROPOSAL's parameterizations (bremsstrahlung and
pair production Kelner-Kokoulin-Petrukhin, photonuclear ALLM97 with
Butkevich-Mikheyev shadowing), so benchmarking against PROPOSAL validates
the transport algebra and says nothing about the cross sections
themselves. This example measures that remaining error the way the field
does: an ensemble of published parameterizations, one channel swapped at
a time, every variant pushed through the same quadrature the shipped
table uses (:func:`~softpaws.transport.coefficients.build_proposal_table`).

**What is varied.** Bremsstrahlung: Andreev-Bezrukov-Bugaev and the NLO
Sandrock-Soedingrekso-Rhode set. Pair production: the NLO set. Photonuclear
-- the channel that carries the budget -- Bezrukov-Bugaev (real-photon,
hard component), Block-Durand-Ha, ALLM91, and ALLM97 under the
Dutta-Reno-Sarcevic-Seckel shadowing. Because ``Phi`` is linear in the
loss spectrum, each swap propagates exactly: the drift ratio
``kappa_1 = Phi'_var(0) / Phi'(0)`` is the energy-reconstruction error,
the second-moment ratio ``kappa_2`` the error on everything
fluctuation-sensitive (the range spread, the ``A_eff`` fluctuation term,
punch-through tails).

**What this is not.** A spread of published models is a proxy for the
theory error, not a posterior; the project's own reducible errors -- the
6.9%-long range from the uncalibrated moment family, the 3% ionization
splice, frozen-vs-running coefficients -- are separate line items and are
not in these bands. Ionization itself is known to sub-percent and left
fixed.

**The verdict this buys.** The envelope on the mean loss is
percent-level at a TeV and grows with the photonuclear share toward
higher energy; at KM3-230213A's 120 PeV it is the honest error bar on
the standard-loss energy estimate (``E_hat = E kappa`` inverts to
``delta E / E ~ kappa_1 - 1``). Figure 69c places that envelope against
example 67's DR2-allowed BSM enhancements on the same axes: the model
error and the loss laws the below-PeV data cannot exclude live one to
three orders of magnitude apart, so the loss-model uncertainty can
neither fake nor hide them.

Usage
-----
    python scripts/2026_muon_transport/69_loss_model_error_budget.py
    python scripts/2026_muon_transport/69_loss_model_error_budget.py --rebuild
"""

import argparse
import importlib.util
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.transport import loss_ensemble as ensemble
from softpaws.transport.loss_ensemble import VARIANTS

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"
_CACHE = _DEFAULT_OUT_DIR / "69_ensemble.npz"


def load_example(stem: str, name: str):
    """Import a sibling example module by file stem."""
    spec = importlib.util.spec_from_file_location(name, _HERE / stem)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: Energy grid of the budget [GeV] (the shipped table's range).
E_GRID = ensemble.E_GRID

#: Moment orders kept: ``Phi'(0)`` (drift; the energy-reconstruction scale)
#: and ``Phi''(0)`` (the fluctuation scale).
N_MOMENTS = ensemble.N_MOMENTS

#: The KM3-230213A reference energy [GeV] for the printed error bar.
KM3_MU_GEV = 1.2e8

# The variants, the moments and the ensemble itself live in
# :mod:`softpaws.transport.loss_ensemble`; the names stay for the scripts that load this one.

COLORS = {"bremsstrahlung": ("#7570b3", "#8da0cb"), "pair production": ("0.45",),
          "photonuclear": ("#e7298a", "#d95f02", "#e6ab02", "#a6761d"),
          "band": "#e7298a"}


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rebuild", action="store_true",
                        help="Recompute the ensemble even if the cache exists.")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the figures, '69a'-'69c'.")
    return parser.parse_args()


def variant_parametrization(label: str):
    """See :func:`softpaws.transport.loss_ensemble.variant_parametrization`."""
    return ensemble.variant_parametrization(label)


def channel_moments(param, energy_gev: float, y: np.ndarray) -> np.ndarray:
    """See :func:`softpaws.transport.loss_ensemble.channel_moments`."""
    return ensemble.channel_moments(param, energy_gev, y, N_MOMENTS)


def build_ensemble() -> dict:
    """See :func:`softpaws.transport.loss_ensemble.build_ensemble`."""
    return ensemble.build_ensemble(E_GRID, VARIANTS, N_MOMENTS)


def load_ensemble(rebuild: bool) -> tuple:
    """The ensemble from the cache; see :func:`softpaws.transport.loss_ensemble.load_ensemble`."""
    if _CACHE.exists() and not rebuild:
        print(f"  cached ensemble from {_CACHE.name}")
    return ensemble.load_ensemble(_CACHE, rebuild)


def _save(fig, out_dir: pathlib.Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {path.resolve()}")
    plt.close(fig)


def _variant_colors():
    """One color per variant, grouped by channel family."""
    counters = {ch: 0 for ch in COLORS if isinstance(COLORS[ch], tuple)}
    out = {}
    for label, (channel, _) in VARIANTS.items():
        out[label] = COLORS[channel][counters[channel]]
        counters[channel] += 1
    return out


def figure_ratio(ratios, order: int, stem: str, out_dir) -> None:
    """Variant-to-baseline ratio of one moment, with the envelope."""
    colors = _variant_colors()
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.5, 3.0))
        stack = np.array([r[:, order] for r in ratios.values()])
        ax.fill_between(E_GRID, stack.min(axis=0), stack.max(axis=0),
                        color=COLORS["band"], alpha=0.12, lw=0, label="envelope")
        for label, values in ratios.items():
            ax.plot(E_GRID, values[:, order], color=colors[label], lw=1.1,
                    ls="--" if label.startswith("photo") else "-",
                    label=f"{label} ({VARIANTS[label][1]})")
        ax.axhline(1.0, color="0.2", lw=0.8)
        ax.axvline(KM3_MU_GEV, color="0.4", lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(E_GRID[0], E_GRID[-1])
        ax.set_xlabel(r"$E_\mu$ [GeV]")
        moment = (r"$\Phi'(0)$ (mean loss)", r"$\Phi''(0)$ (fluctuation scale)")[order]
        ax.set_ylabel(f"variant / baseline, {moment}")
        ax.legend(fontsize=4.6, frameon=False, loc="upper left", ncol=1)
        _save(fig, out_dir, stem)


def figure_versus_bsm(ratios, out_dir) -> None:
    """Figure 69c: the model-error envelope against the DR2-allowed BSM laws."""
    ex67 = load_example("67_loss_model_energy_reconstruction.py", "_e67")
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.5, 3.0))
        stack = np.array([r[:, 0] for r in ratios.values()])
        spread = np.abs(stack - 1.0).max(axis=0)
        ax.fill_between(E_GRID, 1.0e-4, spread, color=COLORS["band"], alpha=0.15, lw=0)
        ax.plot(E_GRID, spread, color=COLORS["band"], lw=1.3,
                label="loss-model error (ensemble envelope)")
        for n, (eps, tag) in ex67.EPS_DRAWN.items():
            kappa = ex67._EX59.kappa(E_GRID, eps, n)
            ax.plot(E_GRID, kappa - 1.0, color="0.35", lw=1.0,
                    ls={0.5: "-", 1.0: "--", 2.0: ":"}[n],
                    label=rf"BSM $\kappa - 1$, $n = {n:g}$ ({tag})")
        ax.axvline(KM3_MU_GEV, color="0.4", lw=0.6, ls=":")
        ax.text(KM3_MU_GEV * 1.2, 2.0e-4, "KM3-230213A muon", rotation=90,
                fontsize=4.8, color="0.4", va="bottom")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1.0e3, E_GRID[-1])
        ax.set_ylim(1.0e-4, 1.0e6)
        ax.set_xlabel(r"$E_\mu$ [GeV]")
        ax.set_ylabel(r"fractional loss enhancement $|\kappa - 1|$")
        ax.legend(fontsize=5.0, frameon=False, loc="upper left")
        _save(fig, out_dir, "69c_error_vs_bsm")


def main() -> None:
    args = parse_args()
    baseline, ratios = load_ensemble(args.rebuild)

    total = sum(baseline.values())
    print("\nBaseline channel shares of Phi'(0):")
    marks = (1.0e4, 1.0e6, 1.0e8, 1.0e10)
    header = "  {:>16}".format("E [GeV]") + "".join(f" {e:>9.0e}" for e in marks)
    print(header)
    for ch, values in baseline.items():
        shares = [np.interp(np.log10(e), np.log10(E_GRID), values[:, 0] / total[:, 0])
                  for e in marks]
        print(f"  {ch:>16}" + "".join(f" {s:9.2%}" for s in shares))

    print("\nVariant / baseline at the same marks (Phi'(0) | Phi''(0)):")
    for label, values in ratios.items():
        cells = []
        for e in marks:
            r1 = np.interp(np.log10(e), np.log10(E_GRID), values[:, 0])
            r2 = np.interp(np.log10(e), np.log10(E_GRID), values[:, 1])
            cells.append(f"{r1:6.3f}|{r2:5.3f}")
        print(f"  {label:>16} " + "  ".join(cells))

    stack = np.array([r[:, 0] for r in ratios.values()])
    envelope = np.abs(stack - 1.0).max(axis=0)
    spread_km3 = float(np.interp(np.log10(KM3_MU_GEV), np.log10(E_GRID), envelope))
    dis = np.array([np.abs(r[:, 0] - 1.0) for label, r in ratios.items()
                    if label != "photo BB"])
    dis_km3 = float(np.interp(np.log10(KM3_MU_GEV), np.log10(E_GRID), dis.max(axis=0)))
    print(f"\nAt the KM3-230213A muon ({KM3_MU_GEV:.2g} GeV): mean-loss envelope "
          f"+-{spread_km3:.1%} -> the 120 PeV estimate carries a +-"
          f"{spread_km3 * 120:.0f} PeV loss-model error bar.")
    print(f"Without the 1981 real-photon Bezrukov-Bugaev (the DIS-era subset): "
          f"+-{dis_km3:.1%}, +-{dis_km3 * 120:.0f} PeV.")
    print("(Separate, reducible line items not in these bands: the 6.9% moment-family "
          "range bias, the 3% ionization-splice convention.)")

    figure_ratio(ratios, 0, "69a_mean_loss_ratio", args.out_dir)
    figure_ratio(ratios, 1, "69b_fluctuation_ratio", args.out_dir)
    figure_versus_bsm(ratios, args.out_dir)


if __name__ == "__main__":
    main()
