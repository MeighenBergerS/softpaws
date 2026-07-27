"""Example 18 -- KM3NeT credible region under a generic (nu_mu or nu_tau) origin.

Extends ``10_flux_contours_icecube_km3net.py`` with the tau-neutrino channel
(:mod:`softpaws.transport.tau`). There, the single KM3NeT UHE event
(KM3-230213A) is fit assuming it is a direct ``nu_mu`` CC track; the resulting
degenerate ``(phi0, gamma)`` banana sits in some tension with the IceCube
region, quantified by a Bayes factor. But a through-going muon track cannot
actually tell ``nu_mu`` and ``nu_tau`` origin apart -- a ``nu_tau`` CC
interaction followed by ``tau -> mu`` decay produces an identical track, and
the event sits at ~120 PeV, exactly where example 16 showed the tau channel
stops being a percent-level correction.

This script keeps everything from example 10 unchanged -- the IceCube region
(diffusion and exact) and the KM3NeT *numu-only* region (diffusion and exact,
for reference) -- and adds one new region: the KM3NeT event fit under a
*generic* origin, where the predicted rate at each ``(phi0, gamma)`` is the
direct ``nu_mu`` prediction plus the tau-induced contribution from a
same-normalization ``nu_tau`` flux (``FitConfig(origin="generic")``,
:func:`softpaws.comparison.likelihood.signal_counts`). This is only defined for
the exact forward model, since the tau ratio is built from the exact
eigenvalue ``Phi(A)``.

Since the generic-origin model predicts *more* events than the numu-only model
at the same ``(phi0, gamma)``, explaining the single observed event needs a
smaller normalization -- so the generic-origin banana should sit at lower
``phi0`` than the numu-only one, closer to the IceCube region. The printed
Bayes factors (numu-only vs. generic-origin, both exact) quantify whether that
shift meaningfully relaxes the tension the paper reports (~18).

Usage
-----
    python examples/18_km3net_generic_origin.py
    python examples/18_km3net_generic_origin.py --steps 4000
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

# IceCube-like detector and binning (unchanged from example 10).
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

METHOD_STYLE = {"diffusion": {"linestyles": "-"}, "exact": {"linestyles": "--"}}
GENERIC_STYLE = {"linestyles": ":"}
EXP_COLOR = {"IceCube": "#1b9e77", "KM3NeT": "#d95f02"}
GENERIC_COLOR = "#7570b3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path,
                        default=_DEFAULT_OUT_DIR / "18_km3net_generic_origin.pdf")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--walkers", type=int, default=32)
    return parser.parse_args()


def ic_config(method: str) -> FitConfig:
    return FitConfig(
        radius_km=IC_RADIUS_KM, log10_e_edges=IC_EDGES, livetime_s=IC_LIVETIME_S,
        solid_angle_sr=IC_SOLID_ANGLE_SR, method=method,
        column_depth_km=IC_COLUMN_DEPTH_KM if method == "exact" else None,
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
        print(f"  fitting IceCube ({method}) ...")
        posteriors[("IceCube", method)] = run_mcmc(ic_config(method), ic_observed,
                                                   template, **kwargs)
        print(f"  fitting KM3NeT numu-only ({method}) ...")
        posteriors[("KM3NeT", method)] = run_mcmc(
            km_config(method), KM_OBSERVED, None, **kwargs
        )
    print("  fitting KM3NeT generic origin (exact, incl. tau) ...")
    posteriors[("KM3NeT-generic", "exact")] = run_mcmc(
        km_config("exact", origin="generic"), KM_OBSERVED, None, **kwargs
    )

    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.4, 3.2))
        for (exp, method), s in posteriors.items():
            # Fresh dict per call: corner.hist2d mutates contour_kwargs (caching
            # "colors"), so a shared dict would leak the first experiment's colour.
            if exp == "KM3NeT-generic":
                color = GENERIC_COLOR
                contour_kwargs = {"colors": color, **GENERIC_STYLE}
            else:
                color = EXP_COLOR[exp]
                contour_kwargs = {"colors": color, **METHOD_STYLE[method]}
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
        handles += [plt.Line2D([], [], color="k", ls=st["linestyles"], label=m)
                    for m, st in METHOD_STYLE.items()]
        handles += [plt.Line2D([], [], color=GENERIC_COLOR, ls=GENERIC_STYLE["linestyles"],
                               label="KM3NeT generic origin\n(exact, incl. tau)")]
        ax.legend(handles=handles, fontsize="x-small", loc="upper right")

        bf_numu = bayes_factor(km_config("exact"), posteriors[("IceCube", "exact")])
        bf_generic = bayes_factor(
            km_config("exact", origin="generic"), posteriors[("IceCube", "exact")]
        )
        ax.text(
            0.04, 0.04,
            f"Bayes factor (numu-only) $\\approx$ {bf_numu:.0f}\n"
            f"Bayes factor (generic) $\\approx$ {bf_generic:.0f}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize="small",
        )
        print(f"  Bayes factor, numu-only (exact)   ~ {bf_numu:.1f}  (paper: ~18)")
        print(f"  Bayes factor, generic origin (exact) ~ {bf_generic:.1f}")

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
