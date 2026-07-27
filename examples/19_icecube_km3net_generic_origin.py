"""Example 19 -- IceCube and KM3NeT credible regions, both under a generic origin.

Extends ``18_km3net_generic_origin.py`` by applying the same generic
(``nu_mu`` or ``nu_tau``) origin treatment to *both* experiments, not just
KM3NeT. A through-going muon track cannot distinguish direct ``nu_mu`` CC
production from ``nu_tau`` CC production followed by ``tau -> mu`` decay
(:mod:`softpaws.transport.tau`), so in principle every track in the IceCube
diffuse sample carries the same flavor ambiguity the KM3NeT event does --
just at a much smaller, energy-dependent level (example 16: a few percent
at IceCube's typical few-PeV energies, tens of percent at KM3NeT's ~120 PeV).

Both experiments' injected/observed datasets are left exactly as in examples
10 and 18 (a numu-only Asimov truth for IceCube, one observed KM3NeT event);
what changes here is only the *model* used to fit them,
``FitConfig(origin="generic")`` adding the tau-induced contribution on top of
the direct nu_mu prediction (only defined for ``method="exact"``, since the
tau ratio needs the exact eigenvalue ``Phi(A)``). This isolates the effect of
consistently accounting for the flavor ambiguity in the *model*, without
also re-litigating what "actually" produced the data.

The expected outcome: IceCube's generic-origin region should barely move
relative to its numu-only region (the diffuse sample lives mostly at the
few-PeV energies where the tau correction is small), while KM3NeT's should
shift much more (as in example 18) -- directly showing that the flavor
ambiguity matters almost entirely for the single ultra-high-energy event, not
the bulk diffuse sample. Two Bayes factors are reported: the numu-only
baseline (matching examples 10/18) and the fully self-consistent generic-origin
one (both experiments' regions built with the same origin assumption).

MCMC precision is higher than examples 10/18 by default (more steps and
walkers), since the ``(phi0, gamma)`` contours were visibly jagged at the
lower default -- this script accordingly takes several minutes to run.

Usage
-----
    python examples/19_icecube_km3net_generic_origin.py
    python examples/19_icecube_km3net_generic_origin.py --steps 4000 --walkers 32
"""

import argparse
import pathlib

import corner
import matplotlib.pyplot as plt
import numpy as np

from softpaws.comparison.likelihood import (
    B_SCALE_MEAN,
    D_SCALE_MEDIAN,
    GAMMA_RANGE,
    GAMMA_TRUTH,
    PHI0_TRUTH,
    FitConfig,
    asimov_dataset,
    atmospheric_template,
    poisson_log_likelihood,
    run_mcmc,
    signal_counts,
)

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"

# IceCube-like detector and binning (unchanged from examples 10/18).
IC_RADIUS_KM = 0.62
IC_COLUMN_DEPTH_KM = 1.95
IC_EDGES = np.arange(4.0, 9.01, 0.5)
IC_SOLID_ANGLE_SR = 2.0 * np.pi
IC_LIVETIME_S = 9.5 * 365.25 * 86400.0
BKG_FRACTION = 0.3

# KM3NeT single UHE event KM3-230213A (E_mu = 120 PeV; 90% interval [35, 380] PeV).
KM_RADIUS_KM = 0.33
KM_COLUMN_DEPTH_KM = 2.5
KM_EDGES = np.array([np.log10(35.0e6), np.log10(380.0e6)])
KM_LIVETIME_S = 1000.0 * 86400.0  # approximate multi-year ARCA exposure
KM_SOLID_ANGLE_SR = 2.0 * np.pi
KM_OBSERVED = np.array([1.0])

# Line style keyed by (method, origin); color keyed by experiment.
LINE_STYLE = {
    ("diffusion", "numu"): "-",
    ("exact", "numu"): "--",
    ("exact", "generic"): ":",
}
LINE_LABEL = {
    ("diffusion", "numu"): "numu-only, diffusion",
    ("exact", "numu"): "numu-only, exact",
    ("exact", "generic"): "generic origin, exact (incl. tau)",
}
EXP_COLOR = {"IceCube": "#1b9e77", "KM3NeT": "#d95f02"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path,
                        default=_DEFAULT_OUT_DIR / "19_icecube_km3net_generic_origin.pdf")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--walkers", type=int, default=48)
    return parser.parse_args()


def ic_config(method: str, origin: str = "numu") -> FitConfig:
    return FitConfig(
        radius_km=IC_RADIUS_KM, log10_e_edges=IC_EDGES, livetime_s=IC_LIVETIME_S,
        solid_angle_sr=IC_SOLID_ANGLE_SR, method=method,
        column_depth_km=IC_COLUMN_DEPTH_KM if method == "exact" else None,
        origin=origin,
    )


def km_config(method: str, origin: str = "numu") -> FitConfig:
    return FitConfig(
        radius_km=KM_RADIUS_KM, log10_e_edges=KM_EDGES, livetime_s=KM_LIVETIME_S,
        solid_angle_sr=KM_SOLID_ANGLE_SR, method=method,
        column_depth_km=KM_COLUMN_DEPTH_KM if method == "exact" else None,
        origin=origin,
    )


def bayes_factor(km_conf: FitConfig, ic_samples: dict[str, np.ndarray],
                 n_draw: int = 20000, seed: int = 7) -> float:
    """MC estimate of BF = <L_KM>_flat / <L_KM>_IC-posterior (Eq. 4.5)."""
    rng = np.random.default_rng(seed)

    def mean_likelihood(phi0: np.ndarray, gamma: np.ndarray) -> float:
        vals = np.array([
            np.exp(poisson_log_likelihood(
                KM_OBSERVED, signal_counts(km_conf, p, g, B_SCALE_MEAN, D_SCALE_MEDIAN)))
            for p, g in zip(phi0, gamma)
        ])
        return float(np.mean(vals))

    phi0_flat = rng.uniform(0.0, 2.0, n_draw)
    gamma_flat = rng.uniform(*GAMMA_RANGE, n_draw)
    idx = rng.integers(0, ic_samples["phi0"].size, n_draw)

    num = mean_likelihood(phi0_flat, gamma_flat)
    den = mean_likelihood(ic_samples["phi0"][idx], ic_samples["gamma"][idx])
    return num / den if den > 0 else float("inf")


def make_figure(out_path: pathlib.Path, n_steps: int, n_walkers: int) -> None:
    template = atmospheric_template(IC_EDGES)
    signal = asimov_dataset(ic_config("diffusion"), PHI0_TRUTH, GAMMA_TRUTH)
    ic_observed = asimov_dataset(ic_config("diffusion"), PHI0_TRUTH, GAMMA_TRUTH,
                                 bkg_total=BKG_FRACTION * signal.sum())
    kwargs = dict(n_walkers=n_walkers, n_steps=n_steps, n_burn=n_steps // 3)

    posteriors = {}
    for method in ("diffusion", "exact"):
        print(f"  fitting IceCube numu-only ({method}) ...")
        posteriors[("IceCube", method, "numu")] = run_mcmc(
            ic_config(method), ic_observed, template, **kwargs
        )
        print(f"  fitting KM3NeT numu-only ({method}) ...")
        posteriors[("KM3NeT", method, "numu")] = run_mcmc(
            km_config(method), KM_OBSERVED, None, **kwargs
        )
    print("  fitting IceCube generic origin (exact, incl. tau) ...")
    posteriors[("IceCube", "exact", "generic")] = run_mcmc(
        ic_config("exact", origin="generic"), ic_observed, template, **kwargs
    )
    print("  fitting KM3NeT generic origin (exact, incl. tau) ...")
    posteriors[("KM3NeT", "exact", "generic")] = run_mcmc(
        km_config("exact", origin="generic"), KM_OBSERVED, None, **kwargs
    )

    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.4, 3.2))
        for (exp, method, origin), s in posteriors.items():
            color = EXP_COLOR[exp]
            # Fresh dict per call: corner.hist2d mutates contour_kwargs (caching
            # "colors"), so a shared dict would leak the previous curve's colour.
            contour_kwargs = {"colors": color, "linestyles": LINE_STYLE[(method, origin)]}
            corner.hist2d(
                s["gamma"], s["phi0"], ax=ax, levels=(0.68, 0.95),
                plot_datapoints=False, plot_density=False, no_fill_contours=True,
                color=color, contour_kwargs=contour_kwargs,
            )
        ax.plot(GAMMA_TRUTH, PHI0_TRUTH, "k*", ms=7, zorder=6)
        ax.set_xlim(1.8, 3.0)
        ax.set_ylim(0.0, 1.5)
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(r"$\phi_0$")

        handles = [plt.Line2D([], [], color=c, label=e) for e, c in EXP_COLOR.items()]
        handles += [plt.Line2D([], [], color="k", ls=ls, label=LINE_LABEL[key])
                    for key, ls in LINE_STYLE.items()]
        ax.legend(handles=handles, fontsize="x-small", loc="upper right")

        bf_numu = bayes_factor(km_config("exact"), posteriors[("IceCube", "exact", "numu")])
        bf_generic = bayes_factor(
            km_config("exact", origin="generic"), posteriors[("IceCube", "exact", "generic")]
        )
        ax.text(
            0.04, 0.04,
            f"Bayes factor (numu-only) $\\approx$ {bf_numu:.0f}\n"
            f"Bayes factor (generic, self-consistent) $\\approx$ {bf_generic:.0f}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize="small",
        )
        print(f"  Bayes factor, numu-only (exact)               ~ {bf_numu:.1f}  (paper: ~18)")
        print(f"  Bayes factor, generic origin (exact, both exps) ~ {bf_generic:.1f}")

        ic_phi0_numu = np.median(posteriors[("IceCube", "exact", "numu")]["phi0"])
        ic_phi0_generic = np.median(posteriors[("IceCube", "exact", "generic")]["phi0"])
        km_phi0_numu = np.median(posteriors[("KM3NeT", "exact", "numu")]["phi0"])
        km_phi0_generic = np.median(posteriors[("KM3NeT", "exact", "generic")]["phi0"])
        print(f"  IceCube median phi0: numu-only {ic_phi0_numu:.3f} -> "
              f"generic {ic_phi0_generic:.3f} "
              f"({100 * (ic_phi0_generic / ic_phi0_numu - 1.0):+.1f}%)")
        print(f"  KM3NeT  median phi0: numu-only {km_phi0_numu:.3f} -> "
              f"generic {km_phi0_generic:.3f} "
              f"({100 * (km_phi0_generic / km_phi0_numu - 1.0):+.1f}%)")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png"):
            path = out_path.with_suffix(suffix)
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"Figure saved to: {path.resolve()}")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    make_figure(args.out, args.steps, args.walkers)


if __name__ == "__main__":
    main()
