"""Example 11 — cross section required for one UHE event (paper Fig. 8).

Reproduces the structure of Figure 8 of arXiv:2607.13143: the distribution of the
effective cross-section slope ``lambda`` (left) and the enhancement
``R = sigma / sigma_SM`` at 100 PeV (right) obtained by requiring one muon event
at ``E* = 100 PeV`` (Eq. 4.6), propagating the ``(phi0, gamma)`` posterior of each
experiment. Curves are shown for KM3NeT, IceCube (1 yr), and IceCube (9.5 yr).

The key addition over the paper: each distribution is computed with both the FP
drift-diffusion soft volume and the exact eigenvalue soft volume. The FP form has
a pole at ``lambda = gamma - 1`` (~1.38 for gamma = 2.38): as the required slope
approaches it the predicted event count diverges, so the FP ``lambda`` never
crosses the pole. The exact form keeps the finite-column saturation factor and so
stays finite, allowing solutions beyond the pole (see
``docs/exact_soft_volume_notes.md`` Part 11) -- exactly the pathology the notes
flag in the paper's cross-section extraction.

The IceCube ``(phi0, gamma)`` posterior comes from an Asimov fit at the paper's
best fit (see ``examples/09``); KM3NeT from its single UHE event. Exposures and
normalizations are approximate.

Usage
-----
    python examples/11_cross_section_from_one_event.py
    python examples/11_cross_section_from_one_event.py --steps 4000
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.comparison.likelihood import (
    GAMMA_TRUTH,
    PHI0_TRUTH,
    FitConfig,
    asimov_dataset,
    atmospheric_template,
    cross_section_enhancement,
    required_lambda,
    run_mcmc,
)

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"

E_STAR_GEV = 1.0e8  # 100 PeV
YEAR_S = 365.25 * 86400.0

# IceCube flux fit (Asimov) and KM3NeT single-event fit.
IC_RADIUS_KM = 0.62
IC_EDGES = np.arange(4.0, 9.01, 0.5)
IC_SOLID_ANGLE_SR = 2.0 * np.pi
IC_LIVETIME_S = 9.5 * YEAR_S
BKG_FRACTION = 0.3
KM_RADIUS_KM = 0.33
KM_EDGES = np.array([np.log10(35.0e6), np.log10(380.0e6)])
KM_LIVETIME_S = 1000.0 * 86400.0  # approximate multi-year ARCA exposure
KM_SOLID_ANGLE_SR = 2.0 * np.pi
KM_OBSERVED = np.array([1.0])

# Exposure and geometry per experiment for the Eq. (4.6) extraction.
EXPERIMENTS = {
    "IceCube (1 yr)": dict(radius_km=IC_RADIUS_KM, livetime_s=YEAR_S,
                           solid_angle_sr=IC_SOLID_ANGLE_SR, column_depth_km=1.95,
                           posterior="IceCube"),
    "IceCube (9.5 yr)": dict(radius_km=IC_RADIUS_KM, livetime_s=9.5 * YEAR_S,
                             solid_angle_sr=IC_SOLID_ANGLE_SR, column_depth_km=1.95,
                             posterior="IceCube"),
    "KM3NeT": dict(radius_km=KM_RADIUS_KM, livetime_s=KM_LIVETIME_S,
                   solid_angle_sr=KM_SOLID_ANGLE_SR, column_depth_km=2.5,
                   posterior="KM3NeT"),
}
EXP_COLOR = {"IceCube (1 yr)": "C2", "IceCube (9.5 yr)": "C1", "KM3NeT": "C4"}
METHOD_STYLE = {"diffusion": {"ls": "-"}, "exact": {"ls": "--"}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path,
                        default=_DEFAULT_OUT_DIR / "11_cross_section_from_one_event.pdf")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--walkers", type=int, default=32)
    parser.add_argument("--n-samples", type=int, default=2000,
                        help="Posterior draws to propagate through Eq. (4.6).")
    return parser.parse_args()


def extraction_config(exp_kwargs: dict, method: str) -> FitConfig:
    return FitConfig(
        radius_km=exp_kwargs["radius_km"], log10_e_edges=IC_EDGES,
        livetime_s=exp_kwargs["livetime_s"], solid_angle_sr=exp_kwargs["solid_angle_sr"],
        method=method,
        column_depth_km=exp_kwargs["column_depth_km"] if method == "exact" else None,
    )


def solve_lambda_R(config: FitConfig, phi0: np.ndarray, gamma: np.ndarray):
    """Solve Eq. (4.6) for lambda over posterior draws; return (lambda, R) arrays."""
    lams = np.array([required_lambda(config, p, g, E_STAR_GEV)
                     for p, g in zip(phi0, gamma)])
    lams = lams[np.isfinite(lams)]
    rs = np.array([cross_section_enhancement(lam, E_STAR_GEV) for lam in lams])
    return lams, rs


def make_figure(fits: dict[str, dict[str, np.ndarray]], out_path: pathlib.Path,
                n_samples: int, seed: int = 3) -> None:
    rng = np.random.default_rng(seed)

    with plt.style.context(str(_STYLE)):
        fig, (ax_lam, ax_r) = plt.subplots(1, 2, figsize=(6.4, 3.0))

        for exp, kwargs in EXPERIMENTS.items():
            post = fits[kwargs["posterior"]]
            idx = rng.integers(0, post["phi0"].size, min(n_samples, post["phi0"].size))
            phi0, gamma = post["phi0"][idx], post["gamma"][idx]
            for method, style in METHOD_STYLE.items():
                config = extraction_config(kwargs, method)
                lams, rs = solve_lambda_R(config, phi0, gamma)
                if lams.size == 0:
                    continue
                ax_lam.hist(lams, bins=40, range=(0.0, 2.5), density=True,
                            histtype="step", color=EXP_COLOR[exp], **style)
                ax_r.hist(rs, bins=40, range=(0.0, 50.0), density=True,
                          histtype="step", color=EXP_COLOR[exp], **style)
                print(f"  {exp:16s} {method:10s}  "
                      f"lambda = {np.median(lams):.2f}   R = {np.median(rs):.1f}")

        ax_lam.axvline(1.38, color="0.5", lw=0.8, ls=":")
        ax_lam.text(1.40, ax_lam.get_ylim()[1] * 0.95, r"FP pole ($\gamma=2.38$)",
                    fontsize="x-small", color="0.4", rotation=90, va="top")
        ax_lam.set_xlabel(r"$\lambda$")
        ax_lam.set_ylabel("posterior density")
        ax_r.set_xlabel(r"$R = \sigma / \sigma_{\rm SM}$ at 100 PeV")

        handles = [plt.Line2D([], [], color=c, label=e) for e, c in EXP_COLOR.items()]
        handles += [plt.Line2D([], [], color="k", ls=st["ls"], label=m)
                    for m, st in METHOD_STYLE.items()]
        ax_lam.legend(handles=handles, fontsize="x-small", loc="upper left")

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png"):
            path = out_path.with_suffix(suffix)
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"Figure saved to: {path.resolve()}")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    kwargs = dict(n_walkers=args.walkers, n_steps=args.steps, n_burn=args.steps // 3)

    # IceCube (phi0, gamma) posterior from an Asimov fit at the paper's best fit;
    # the diffusion (FP) fit provides the flux posterior propagated by both methods.
    ic_conf = FitConfig(radius_km=IC_RADIUS_KM, log10_e_edges=IC_EDGES,
                        livetime_s=IC_LIVETIME_S, solid_angle_sr=IC_SOLID_ANGLE_SR,
                        method="diffusion")
    signal = asimov_dataset(ic_conf, PHI0_TRUTH, GAMMA_TRUTH)
    ic_observed = asimov_dataset(ic_conf, PHI0_TRUTH, GAMMA_TRUTH,
                                 bkg_total=BKG_FRACTION * signal.sum())
    km_conf = FitConfig(radius_km=KM_RADIUS_KM, log10_e_edges=KM_EDGES,
                        livetime_s=KM_LIVETIME_S, solid_angle_sr=KM_SOLID_ANGLE_SR,
                        method="diffusion")

    print("  fitting IceCube flux (Asimov) ...")
    ic_post = run_mcmc(ic_conf, ic_observed, atmospheric_template(IC_EDGES), **kwargs)
    print("  fitting KM3NeT flux ...")
    km_post = run_mcmc(km_conf, KM_OBSERVED, None, **kwargs)

    make_figure({"IceCube": ic_post, "KM3NeT": km_post}, args.out, args.n_samples)


if __name__ == "__main__":
    main()
