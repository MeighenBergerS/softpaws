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
    python examples/69_loss_model_error_budget.py
    python examples/69_loss_model_error_budget.py --rebuild
"""

import argparse
import importlib.util
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.transport.coefficients import loss_spectrum_y_grid
from softpaws.utils.constants import CM_PER_KM, RHO_WATER_G_CM3

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"
_CACHE = _DEFAULT_OUT_DIR / "69_ensemble.npz"


def load_example(stem: str, name: str):
    """Import a sibling example module by file stem."""
    spec = importlib.util.spec_from_file_location(name, _HERE / stem)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: Energy grid of the budget [GeV] (the shipped table's range).
E_GRID = np.logspace(2.0, 10.0, 33)

#: Moment orders kept: ``Phi'(0)`` (drift; the energy-reconstruction scale)
#: and ``Phi''(0)`` (the fluctuation scale).
N_MOMENTS = 2

#: The KM3-230213A reference energy [GeV] for the printed error bar.
KM3_MU_GEV = 1.2e8

#: Variant label -> (channel to swap, human name). Constructors live in
#: :func:`variant_parametrization` to keep the optional import local.
VARIANTS = {
    "brems ABB": ("bremsstrahlung", "Andreev-Bezrukov-Bugaev"),
    "brems NLO": ("bremsstrahlung", "Sandrock-Soedingrekso-Rhode"),
    "pair NLO": ("pair production", "Sandrock-Soedingrekso-Rhode"),
    "photo BB": ("photonuclear", "Bezrukov-Bugaev + hard"),
    "photo BDH": ("photonuclear", "Block-Durand-Ha"),
    "photo ALLM91": ("photonuclear", "ALLM91"),
    "photo DRSS": ("photonuclear", "ALLM97, DRSS shadowing"),
}

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
    """PROPOSAL parametrization object for one variant channel."""
    import proposal as pp

    shadow_bm = pp.parametrization.photonuclear.ShadowButkevichMikheyev
    builders = {
        "brems ABB": lambda: pp.parametrization.bremsstrahlung.AndreevBezrukovBugaev(False),
        "brems NLO": lambda: pp.parametrization.bremsstrahlung.SandrockSoedingreksoRhode(False),
        "pair NLO": lambda: pp.parametrization.pairproduction.SandrockSoedingreksoRhode(False),
        "photo BB": lambda: pp.parametrization.photonuclear.BezrukovBugaev(True),
        "photo BDH": lambda: pp.parametrization.photonuclear.BlockDurandHa(shadow_bm()),
        "photo ALLM91": lambda: pp.parametrization.photonuclear.AbramowiczLevinLevyMaor91(
            shadow_bm()),
        "photo DRSS": lambda: pp.parametrization.photonuclear.AbramowiczLevinLevyMaor97(
            pp.parametrization.photonuclear.ShadowDuttaRenoSarcevicSeckel()),
    }
    return builders[label]()


def channel_moments(param, energy_gev: float, y: np.ndarray) -> np.ndarray:
    """Log-loss moments ``<(-ln(1-y))^n>`` of one channel [km^-1].

    The same quadrature and water convention as
    :func:`~softpaws.transport.coefficients.build_proposal_table`.
    """
    import proposal as pp

    particle = pp.particle.MuMinusDef()
    medium = pp.medium.Water()
    energy_mev = energy_gev * 1.0e3
    molar_mass = sum(c.atoms_in_molecule * c.atomic_number for c in medium.components)
    scale = CM_PER_KM * RHO_WATER_G_CM3 / medium.mass_density
    rate = np.zeros_like(y)
    for component in medium.components:
        limits = param.kinematic_limits(particle, component, energy_mev)
        inside = (y > limits.v_min) & (y < limits.v_max)
        per_gram = np.zeros_like(y)
        per_gram[inside] = [
            param.differential_crosssection(particle, component, energy_mev, value)
            for value in y[inside]
        ]
        weight = component.atoms_in_molecule * component.atomic_number / molar_mass
        rate += medium.mass_density * weight * per_gram
    rate *= scale
    log_loss = -np.log1p(-y)
    return np.array([np.trapezoid(log_loss**n * rate, y) for n in range(1, N_MOMENTS + 1)])


def build_ensemble() -> dict:
    """Baseline channel moments and every variant's swapped channel."""
    from softpaws.transport.coefficients import proposal_parametrizations

    y = loss_spectrum_y_grid()
    base_params = proposal_parametrizations()
    baseline = {ch: np.zeros((E_GRID.size, N_MOMENTS)) for ch in base_params}
    print("Baseline channels (KKP + KKP + ALLM97/BM) ...")
    for i, e in enumerate(E_GRID):
        for ch, param in base_params.items():
            baseline[ch][i] = channel_moments(param, float(e), y)
    swapped = {}
    for label, (channel, name) in VARIANTS.items():
        print(f"Variant {label} ({name}) ...")
        param = variant_parametrization(label)
        swapped[label] = np.array([channel_moments(param, float(e), y) for e in E_GRID])
    out = {f"base {ch}": v for ch, v in baseline.items()}
    out.update({f"swap {label}": v for label, v in swapped.items()})
    return out


def load_ensemble(rebuild: bool) -> tuple:
    """The ensemble, from the cache when it is present.

    Returns
    -------
    baseline : dict
        Channel -> ``(n_e, N_MOMENTS)`` baseline moments.
    ratios : dict
        Variant label -> ``(n_e, N_MOMENTS)`` total-moment ratio to baseline.
    """
    if _CACHE.exists() and not rebuild:
        print(f"  cached ensemble from {_CACHE.name}")
        data = dict(np.load(_CACHE))
    else:
        data = build_ensemble()
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(_CACHE, **data)
    baseline = {k[5:]: v for k, v in data.items() if k.startswith("base ")}
    total = sum(baseline.values())
    ratios = {}
    for label, (channel, _) in VARIANTS.items():
        variant_total = total - baseline[channel] + data[f"swap {label}"]
        ratios[label] = variant_total / total
    return baseline, ratios


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
