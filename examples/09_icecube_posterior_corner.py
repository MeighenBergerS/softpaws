"""Example 09 — IceCube diffuse-flux posterior corner (paper Fig. 6).

Reproduces the structure of Figure 6 of arXiv:2607.13143: the joint posterior of
the diffuse-flux parameters ``(phi0, gamma)`` and the transport nuisances
``(b_mu, d_mu)`` from a binned Poisson fit (Eq. 4.2). Three forward models are
overlaid: the paper's ``drift`` and ``diffusion`` models, plus the exact
eigenvalue model (``docs/exact_soft_volume_notes.md``).

The fit is driven by an **Asimov dataset** injected from the paper's best-fit
diffuse flux (``phi0 = 0.63``, ``gamma = 2.38``) through the diffusion forward
model, with a floated atmospheric background. Injection is used instead of the
in-repo DR2 downgoing sample because the latter is dominated by cosmic-ray
atmospheric muons -- outside the neutrino soft-volume model's validity -- so it
cannot recover the paper's flux posteriors. All three models are then fit to the
same injected data.

The drift and diffusion models recover the injected truth; the exact model, which
carries the ``I(A) ~ 0.8`` inelasticity factor the paper drops, prefers a higher
``phi0`` to describe the same counts -- the normalization shift discussed in the
notes.

Usage
-----
    python examples/09_icecube_posterior_corner.py
    python examples/09_icecube_posterior_corner.py --steps 4000 --walkers 48
"""

import argparse
import pathlib

import corner
import matplotlib.pyplot as plt
import numpy as np

from softpaws.comparison.likelihood import (
    D_SCALE_LOGSTD,
    D_SCALE_MEDIAN,
    GAMMA_TRUTH,
    PHI0_TRUTH,
    FitConfig,
    asimov_dataset,
    atmospheric_template,
    run_mcmc,
)
from softpaws.transport.coefficients import diffusion_coefficient

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"

RADIUS_KM = 0.62
COLUMN_DEPTH_KM = 1.95  # IceCube overburden for the exact finite-column model
LOG10_E_EDGES = np.arange(4.0, 9.01, 0.5)  # 10 TeV to 1 EeV
SOLID_ANGLE_SR = 2.0 * np.pi  # downgoing hemisphere
LIVETIME_S = 9.5 * 365.25 * 86400.0  # IceCube 9.5 yr exposure
BKG_FRACTION = 0.3  # injected background as a fraction of the total signal

CORNER_LABELS = {
    "phi0": r"$\phi_0$",
    "gamma": r"$\gamma$",
    "b_mu": r"$b_\mu\,[\mathrm{km}^{-1}]$",
    "d_mu": r"$d_\mu\,[\mathrm{km}^{-1}]$",
}
METHOD_COLORS = {"drift": "C0", "diffusion": "C1", "exact": "C3"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path,
                        default=_DEFAULT_OUT_DIR / "09_icecube_posterior_corner.pdf")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--walkers", type=int, default=32)
    return parser.parse_args()


def config_for(method: str) -> FitConfig:
    return FitConfig(
        radius_km=RADIUS_KM, log10_e_edges=LOG10_E_EDGES, livetime_s=LIVETIME_S,
        solid_angle_sr=SOLID_ANGLE_SR, method=method,
        column_depth_km=COLUMN_DEPTH_KM if method == "exact" else None,
    )


def corner_columns(samples: dict[str, np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Stack (phi0, gamma, b_mu, d_mu); the drift model's d_mu is drawn from prior."""
    n = samples["phi0"].size
    if "d_mu" in samples:
        d_mu = samples["d_mu"]
    else:
        d0 = float(diffusion_coefficient(1.0e7)[0])
        d_scale = D_SCALE_MEDIAN * np.exp(D_SCALE_LOGSTD * rng.standard_normal(n))
        d_mu = d0 * d_scale
    return np.column_stack([samples["phi0"], samples["gamma"], samples["b_mu"], d_mu])


def make_figure(out_path: pathlib.Path, n_steps: int, n_walkers: int) -> None:
    # Inject Asimov data from the paper's best fit through the diffusion model.
    truth_config = config_for("diffusion")
    signal = asimov_dataset(truth_config, PHI0_TRUTH, GAMMA_TRUTH)
    template = atmospheric_template(LOG10_E_EDGES)
    observed = asimov_dataset(truth_config, PHI0_TRUTH, GAMMA_TRUTH,
                              bkg_total=BKG_FRACTION * signal.sum())
    rng = np.random.default_rng(42)

    all_samples = {}
    for method in ("drift", "diffusion", "exact"):
        print(f"  sampling {method} ...")
        all_samples[method] = run_mcmc(
            config_for(method), observed, background=template,
            n_walkers=n_walkers, n_steps=n_steps, n_burn=n_steps // 3,
        )

    stacked = np.vstack([corner_columns(all_samples[m], rng) for m in all_samples])
    ranges = [(np.percentile(stacked[:, i], 0.5), np.percentile(stacked[:, i], 99.5))
              for i in range(stacked.shape[1])]
    labels = list(CORNER_LABELS.values())

    with plt.style.context(str(_STYLE)):
        fig = None
        for method, samples in all_samples.items():
            fig = corner.corner(
                corner_columns(samples, rng), labels=labels, range=ranges,
                color=METHOD_COLORS[method], fig=fig, plot_datapoints=False,
                plot_density=False, levels=(0.68, 0.95), hist_kwargs={"density": True},
            )

        # Mark the injected flux truth on the phi0 and gamma histograms.
        axes = np.array(fig.axes).reshape(4, 4)
        axes[0, 0].axvline(PHI0_TRUTH, color="0.4", lw=0.8, ls=":")
        axes[1, 1].axvline(GAMMA_TRUTH, color="0.4", lw=0.8, ls=":")

        handles = [plt.Line2D([], [], color=c, label=m) for m, c in METHOD_COLORS.items()]
        fig.legend(handles=handles, loc="upper right", frameon=False, fontsize="large")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png"):
            path = out_path.with_suffix(suffix)
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"Figure saved to: {path.resolve()}")
        plt.close(fig)

    print(f"\n  injected truth: phi0 = {PHI0_TRUTH}, gamma = {GAMMA_TRUTH}")
    print("  best-fit (median) flux parameters:")
    for method, samples in all_samples.items():
        print(f"    {method:10s}  phi0 = {np.median(samples['phi0']):.2f}   "
              f"gamma = {np.median(samples['gamma']):.2f}")


def main() -> None:
    args = parse_args()
    make_figure(args.out, args.steps, args.walkers)


if __name__ == "__main__":
    main()
