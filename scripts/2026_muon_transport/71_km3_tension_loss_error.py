"""Example 71 -- the KM3NeT tension under the loss-model error.

Example 57 put KM3-230213A at 2.8 sigma against IceCube's broken power
law; example 69 measured the loss model's own error. This example asks
the question the two together raise: does the cross-section uncertainty
alleviate the tension?

**How a variant enters.** The event's energy scale: KM3NeT's 120 PeV is
a light measurement read through the baseline loss model, so under a
variant with ratio ``kappa_1(E)`` the same light belongs to a muon with
``E kappa_1(E) = E_hat`` and the fold tilts by ``1 / kappa_1`` (the
example 68 algebra, with the ensemble ratio in place of a BSM law). The
event rate: the effective area follows the muon range, so the ARCA21
area optionally carries ``1 / kappa_1`` at the entry energy -- both
switches are reported separately. The IceCube anchor is held fixed: its
flux is measured at ``10^5``-``10^7`` GeV where the ensemble spread is
2-8%, an order below the event-side shift, and its response is the
released one.

**The finding.** The variants move the ``E^-2`` energy estimate by
roughly their ``kappa`` at 120 PeV (the published 220 PeV acquires a
-30 PeV / +45 PeV model-side uncertainty), and the tension moves with
the reinterpreted energy: softer-loss variants (ALLM91,
Block-Durand-Ha) pull the event down the spectrum toward the IceCube
flux and relieve a few tenths of a sigma; the harder Bezrukov-Bugaev
pushes the other way. The loss-model error alone does not dissolve the
tension -- it sets the width of the systematic band the tension number
should be quoted with.

Usage
-----
    python examples/71_km3_tension_loss_error.py
    python examples/71_km3_tension_loss_error.py --no-rate-scaling
"""

import argparse
import importlib.util
import pathlib

import matplotlib.pyplot as plt
import numpy as np

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


_EX69 = load_example("69_loss_model_error_budget.py", "_example_69")
_EX57 = load_example("57_km3_event_energy_and_bpl_tension.py", "_example_57")

#: Flux families of the tension table (example 57's names).
FAMILIES = ("SPL (IceCube tracks)", "BPL")

COLORS = {"sigma": "#e7298a", "median": "#7570b3", "base": "0.3"}


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=pathlib.Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--n-gamma", type=int, default=41)
    parser.add_argument("--n-phi0", type=int, default=121)
    parser.add_argument("--no-rate-scaling", action="store_true",
                        help="Drop the 1/kappa range scaling of the ARCA21 area.")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the figure, '71a'.")
    return parser.parse_args()


def kappa_interp(ratios, label):
    """``kappa_1(E)`` of one ensemble variant, log-interpolated."""
    return variant_scaling(ratios, label, _EX69.E_GRID).kappa_1


def variant_likelihood(u, energy_nu, kappa, normalize=True):
    """Example 57's event fold under the variant loss law.

    The two insertions of example 68's ``bsm_fold``: the light reads
    ``E_mu kappa(E_mu)`` against the baseline calibration, and the
    occupation density tilts by ``1 / kappa``. Normalization mirrors
    ``energy_likelihood``: the accepted-muon integral, with acceptance on
    the *light* energy.
    """
    energy_nu = np.atleast_1d(np.asarray(energy_nu, dtype=float))
    entry = (1.0 - np.squeeze(_EX57.mean_inelasticity(energy_nu))) * energy_nu
    log_e_mu = np.log(entry)[:, None] - _EX57.W_GRID[None, :]
    e_mu = np.exp(log_e_mu)
    kap = kappa(e_mu)
    weight = u[None, :] / kap
    like = _EX57.muon_measurement(np.log10(e_mu * kap))
    value = np.trapezoid(weight * like, _EX57.W_GRID, axis=1)
    if not normalize:
        return value
    accepted = np.where(e_mu * kap >= _EX57.MU_ACCEPT_GEV, weight, 0.0)
    total = np.trapezoid(accepted, _EX57.W_GRID, axis=1)
    return value / np.clip(total, 1.0e-300, None)


def _save(fig, out_dir: pathlib.Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {path.resolve()}")
    plt.close(fig)


def figure_summary(rows, base_sigma, base_median, out_dir) -> None:
    """Medians and tensions per variant, the baseline as reference lines."""
    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))
    with plt.style.context(str(_STYLE)):
        fig, (top, bottom) = plt.subplots(2, 1, figsize=(3.6, 3.6), sharex=True,
                                          height_ratios=(1, 1))
        med = np.array([r["median_pev"] for r in rows])
        lo = np.array([r["lo_pev"] for r in rows])
        hi = np.array([r["hi_pev"] for r in rows])
        top.errorbar(x, med, yerr=(med - lo, hi - med), fmt="o", ms=3,
                     color=COLORS["median"], lw=0.9, capsize=2)
        top.axhline(base_median, color=COLORS["base"], lw=0.8, ls=":")
        top.set_yscale("log")
        top.set_ylabel(r"$E_\nu$ median, $E^{-2}$ [PeV]")
        bottom.plot(x, [r["sigma_bpl"] for r in rows], "o", ms=3.5,
                    color=COLORS["sigma"], label="BPL")
        bottom.plot(x, [r["sigma_spl"] for r in rows], "s", ms=3,
                    color=COLORS["median"], alpha=0.7, label="SPL (tracks)")
        bottom.axhline(base_sigma, color=COLORS["base"], lw=0.8, ls=":",
                       label="baseline BPL")
        bottom.set_ylabel(r"tension [$\sigma$]")
        bottom.set_xticks(x)
        bottom.set_xticklabels(labels, rotation=45, ha="right", fontsize=5)
        bottom.legend(fontsize=5, frameon=False, loc="lower left")
        _save(fig, out_dir, "71a_tension_loss_error")


def main() -> None:
    args = parse_args()
    _, ratios = _EX69.load_ensemble(False)

    print("Building the exact-kernel potential density (example 57) ...")
    u = _EX57.potential_density("exact")
    energy = 10.0**_EX57.LOG10_ENU

    print(f"Loading responses (DR2 from {args.data_dir}) ...")
    ex31 = _EX57._EX31
    ic_log10_e, ic_aeff, ic_livetime_s = ex31.build_icecube_aeff(
        args.data_dir, ex31.DEFAULT_MUON_THRESHOLD_GEV)
    km_log10_e, km_aeff = ex31.arca21_published_aeff()
    gamma_grid = np.linspace(*ex31.GAMMA_RANGE, args.n_gamma)
    phi0_grid = np.logspace(-1.5, 2.7, args.n_phi0)

    surfaces = {}
    for family in FAMILIES:
        shape, _, truth = _EX57.flux_family(family)
        surfaces[family] = (shape, truth, _EX57.ic_surface(
            shape, truth, phi0_grid, gamma_grid, ic_log10_e, ic_aeff, ic_livetime_s))

    def tension(family, energy_like, aeff):
        shape, truth, ic = surfaces[family]
        a_event, a_band = _EX57.km_event_coefficients(shape, gamma_grid, km_log10_e,
                                                      aeff, energy_like)
        km = ex31.km_log_likelihood(phi0_grid[:, None], a_event, a_band, extended=True)
        _, _, sigma = ex31.compatibility(ic, km)
        return sigma

    print(f"\n  {'variant':>14} {'kappa(120PeV)':>13} {'E^-2 med [PeV]':>15} "
          f"{'sigma SPL':>10} {'sigma BPL':>10}")
    rows = []
    base_row = None
    for label in [None, *_EX69.VARIANTS]:
        kappa = (lambda e: np.ones_like(np.asarray(e, dtype=float))) if label is None \
            else kappa_interp(ratios, label)
        like = variant_likelihood(u, energy, kappa)
        rates = variant_likelihood(u, energy, kappa, normalize=False)
        med_p = _EX57.posterior(rates, "E^-2")
        _, med, lo, hi = _EX57.summarize_posterior(med_p)
        aeff = km_aeff if args.no_rate_scaling else (
            km_aeff / kappa(0.4 * 10.0**km_log10_e))
        row = {"label": label or "baseline",
               "median_pev": med, "lo_pev": lo, "hi_pev": hi,
               "sigma_spl": tension("SPL (IceCube tracks)", like, aeff),
               "sigma_bpl": tension("BPL", like, aeff)}
        k120 = float(kappa(1.2e8))
        print(f"  {row['label']:>14} {k120:13.3f} {med:8.0f} [{lo:.0f},{hi:.0f}] "
              f"{row['sigma_spl']:10.2f} {row['sigma_bpl']:10.2f}")
        if label is None:
            base_row = row
        else:
            rows.append(row)

    sig = np.array([r["sigma_bpl"] for r in rows])
    print(f"\nBPL tension band across the ensemble: {sig.min():.2f} - {sig.max():.2f} "
          f"sigma (baseline {base_row['sigma_bpl']:.2f}); the E^-2 median spans "
          f"{min(r['median_pev'] for r in rows):.0f} - "
          f"{max(r['median_pev'] for r in rows):.0f} PeV.")
    figure_summary(rows, base_row["sigma_bpl"], base_row["median_pev"], args.out_dir)


if __name__ == "__main__":
    main()
