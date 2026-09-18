"""Example 29 -- what a fit prefers, given example 28's first-principles model.

Example 28 builds the per-neutrino-energy effective area with nothing fitted --
the stochastic first-passage range (``docs/first_passage_range.md``),
neutral-current regeneration, and the ``nu_tau -> tau -> mu`` channel -- and
lands within 0.049 dex rms of the published IceCube upgoing table. This example
asks the complementary question: if the model's few physical handles are allowed
to float, where does the data put them, and do they land on the values they were
*derived* to have?

Structurally this is example 09's corner plot moved from the event-rate fit to
the effective-area fit, with the best-fit point marked in every panel alongside
the first-principles expectation.

Five parameters float, each with an independent expectation:

``eps_0``
    Constant selection efficiency, capped at 1: the model is a geometric
    ceiling, so no efficiency can exceed it. Example 28's residual runs 0.58 to
    0.84 over the fitted band, so a *constant* cannot describe it -- where the fit puts this, and
    what trend it leaves behind, is the point.
``log10(E_thr/GeV)``
    Muon selection threshold. The DR2 smearing matrix pins it independently: the
    5th percentile of accepted reconstructed muon energy is flat at ~700 GeV
    (``log10 = 2.85``) across three decades of ``E_nu``.
``b_scale``
    Transport nuisance rescaling ``b_mu``, and with it ``Phi'(0)``. Sets how fast
    the range grows with energy, ``dL / d ln E = 1 / Phi'(0)``, so this is the
    handle the residual *trend* pulls on. Theory value 1, and this is the one
    parameter carrying an informative prior -- the same truncated Gaussian
    ``N(0.94, 0.15)`` the event-rate fits use
    (:mod:`softpaws.comparison.likelihood`), since it is a calibrated transport
    quantity rather than a free knob.
``lambda``
    Effective slope of the charged-current cross section over the fitted band,
    implemented as a tilt of the BGR18 table about a 1 PeV pivot so that
    ``lambda = 0.454`` reproduces it exactly. Together with ``b_scale`` this is
    the other handle on how fast the model grows with energy.
``Lambda``
    Growth of the light reach per e-fold of muon energy, the shape parameter of
    :func:`~softpaws.transport.soft_volume.light_reach_radius_km`, with its
    pivot held at 1 PeV so that ``eps_0`` keeps the normalization. ``Lambda = 0``
    is the static-footprint model. This is the parameter the earlier
    four-parameter version of this fit did not have, and its absence is why
    that version had to push the residual trend into ``lambda``: example 28's
    ratio inversion gives ``Lambda = 19.3`` m per e-fold independently, so the
    question here is whether the posterior finds it and releases ``lambda``.

The ``nu_tau`` flux ratio is *not* floated: it is fixed by oscillations over
astrophysical baselines, so ``f_tau = 1`` throughout.

The flux index ``gamma`` is not floated either, and cannot be. A tabulated
effective area is differential in the neutrino energy, which makes the parent
monochromatic -- App. I's ``s -> 0`` case, where ``Phi(0) = 0`` and
``I(0) = 1``. No spectral index survives anywhere in ``A_eff``, so the
likelihood is exactly flat in ``gamma`` and its posterior would only return its
prior. Bringing ``gamma`` in requires folding a flux through the effective area
and fitting *counts*, which is example 09's territory, not this one.

**Scope of ``lambda``.** It tilts the *detection* cross section. The Earth
transmission ladders are precomputed at BGR18 and held there, both because
recomputing them per sample is not affordable and because the transmission is
exponentially sensitive to the cross section, so floating it there would drown
the effective-area shape this fit is about.

The likelihood is Gaussian in ``ln A_eff``. The published table carries no
uncertainties, so a flat fractional error is assumed (``--sigma``, default 15%):
posterior *widths* are therefore set by that assumption and are not
measurements, while the posterior *locations* are what this example is about.
The top of the DR2 simulation (100 PeV) is excluded, as in example 28.

The Earth transmission ladders do not depend on any fitted parameter, so they
are computed once up front and reused; the per-sample cost is then the closed
form of ``stochastic_muon_range_km``.

Usage
-----
    python scripts/2026_muon_transport/29_effective_area_posterior_corner.py
    python scripts/2026_muon_transport/29_effective_area_posterior_corner.py \
        --steps 6000 --sigma 0.10
"""

import argparse
import pathlib

import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np

from softpaws.comparison.likelihood import B_SCALE_MEAN, B_SCALE_STD
from softpaws.constants import CM_PER_KM
from softpaws.data.icecube import (
    livetime_weighted_effective_area,
)
from softpaws.transport.attenuation import flavour_transmission, prem_column
from softpaws.transport.cross_section import bgr18_cross_section
from softpaws.transport.muon_range import DEFAULT_MUON_THRESHOLD_GEV, stochastic_muon_range_km
from softpaws.transport.soft_volume import light_reach_radius_km
from softpaws.transport.source import MEAN_INELASTICITY, nucleon_number_density
from softpaws.transport.tau import BR_TAU_TO_MU, MEAN_Z

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_DATA_DIR = _HERE.parents[1] / "src" / "softpaws" / "data" / "dataverse_files"
_DEFAULT_OUT_DIR = _HERE / "output"

COMMON_LOG10_E = np.linspace(3.0, 8.0, 26)
STATS_LOG10_E = (5.0, 7.8)
# IceCube as an upright hexagonal prism: ~1 km^2 of footprint by 1 km of
# instrumented height, giving V_det = 1.00 km^3 exactly. RADIUS_KM is the
# area-equivalent radius of the hexagon, so pi R^2 is the footprint, and
# SIDE_COEFF is the prism perimeter divided by pi R, the coefficient of the
# side-projection term (2.0 for a cylinder, 2.10 for a hexagon).
FOOTPRINT_KM2 = 1.0
HEIGHT_KM = 1.0
N_SIDES = 6
RADIUS_KM = float(np.sqrt(FOOTPRINT_KM2 / np.pi))
SIDE_COEFF = float(2.0 * np.sqrt(np.pi * N_SIDES * np.tan(np.pi / N_SIDES)) / np.pi)
N_DEC = 40
N_RUNG = 80
CROSS_SECTION = bgr18_cross_section()

PARAM_NAMES = ("eps_0", "log10_e_thr", "b_scale", "lam", "reach_km")
CORNER_LABELS = [
    r"$\varepsilon_0$",
    r"$\log_{10}(E_{\rm thr}/{\rm GeV})$",
    r"$b_\mu$ scale",
    r"$\lambda$",
    r"$\Lambda$ [km]",
]

# The reach law's pivot is held at the same 1 PeV as the cross-section tilt, so
# that eps_0 carries the normalization and Lambda carries only the shape. Left
# free, the two would be degenerate: the fit constrains sqrt(eps_0) R_eff, not
# R_eff. Lambda = 0 recovers the static-footprint model of example 28 exactly.
REACH_PIVOT_GEV = 1.0e6

# Example 28 inverts the published-to-model ratio for the radius each energy
# demands and fits a straight line through it in ln E, giving 19.3 m per e-fold
# with no reference to this posterior. That is the independent expectation for
# Lambda, in the same sense that 0.454 is the independent expectation for lam.
REACH_EXAMPLE28_KM = 0.0193

# Effective log-log slope of the BGR18 CC cross section over the fitted band,
# and the pivot the tilt rotates about. lam = LAMBDA_BGR18 recovers the table.
LAMBDA_BGR18 = 0.4538
LAMBDA_PIVOT_GEV = 1.0e6

# Fixed by oscillations over astrophysical baselines, so not a fit parameter.
F_TAU = 1.0
# Where each parameter was derived to sit, independently of this fit.
EXPECTED = np.array([
    np.nan,                                  # eps_0: no first-principles value
    np.log10(DEFAULT_MUON_THRESHOLD_GEV),    # 1 TeV, the package default
    1.0,                                     # theory transport coefficients
    LAMBDA_BGR18,                            # the tabulated cross-section slope
    REACH_EXAMPLE28_KM,                      # slope of example 28's ratio inversion
])
# The smearing matrix's own handle on the threshold, shown for comparison.
SMEARING_LOG10_E_THR = 2.85

# Flat within these ranges. eps_0's upper edge is physical, not a convenience:
# the model is a geometric ceiling, so a selection cannot exceed it.
PRIORS = {
    "eps_0": (0.0, 1.0),
    "log10_e_thr": (2.0, 5.0),
    "b_scale": (0.0, 3.0),
    "lam": (0.0, 1.2),
    # Zero sits inside the range, so the data can say no reach is needed; the
    # negative side is kept open as a null check rather than as physics.
    "reach_km": (-0.02, 0.10),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=pathlib.Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=pathlib.Path,
                        default=_DEFAULT_OUT_DIR / "29_effective_area_posterior_corner.pdf")
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--walkers", type=int, default=48)
    parser.add_argument("--sigma", type=float, default=0.15,
                        help="Assumed fractional uncertainty on the published effective area.")
    return parser.parse_args()


def icecube_upgoing(data_dir: pathlib.Path):
    """Livetime-weighted DR2 effective area over the upgoing sky [cm^2].

    Delegates to :func:`softpaws.data.icecube.livetime_weighted_effective_area`
    on ``COMMON_LOG10_E``.
    """
    return livetime_weighted_effective_area(data_dir, COMMON_LOG10_E)[0]


def precompute_ladders() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Solid-angle-averaged transmission ladders, one per flavour.

    Neither ladder depends on a fitted parameter -- they are set by the cross
    section and the PREM column alone -- so they are built once and reused for
    every posterior sample.

    Returns
    -------
    ladders : dict
        ``"mu"`` and ``"tau"`` -> ``(energies, weights, weights_cos,
        weights_sin)`` with ``energies`` of shape ``(n_energy, N_RUNG)`` [GeV]
        and the three weight arrays the matching hemisphere averages of the
        arrival probability against 1, ``|cos theta_z|`` and ``sin theta_z``.

    Notes
    -----
    A prism presents a direction-dependent area, so the declination average no
    longer commutes with the target volume the way it did for a sphere. The
    volume is linear in the two geometry terms, ``pi R^2 |cos|`` and
    ``(P / pi) h sin``, so averaging the transmission against each of them
    separately keeps the result exact while still collapsing the declination
    axis once and for all. At the Pole ``|cos theta_z| = sin(dec)``.
    """
    dec_deg = np.linspace(0.5, 89.5, N_DEC)
    columns = np.array([prem_column(float(d)) for d in dec_deg])
    dec_rad = np.deg2rad(dec_deg)
    solid_angle = np.cos(dec_rad)
    cos_theta = np.sin(dec_rad)
    sin_theta = np.cos(dec_rad)

    ladders: dict[str, tuple[np.ndarray, ...]] = {}
    for flavour in ("mu", "tau"):
        energies = np.empty((COMMON_LOG10_E.size, N_RUNG))
        weights = np.empty((COMMON_LOG10_E.size, N_RUNG))
        weights_cos = np.empty((COMMON_LOG10_E.size, N_RUNG))
        weights_sin = np.empty((COMMON_LOG10_E.size, N_RUNG))
        for i, log10_e in enumerate(COMMON_LOG10_E):
            rung_energy, rung_weight = flavour_transmission(
                10.0**log10_e, columns, CROSS_SECTION, flavour=flavour,
                n_grid=N_RUNG, decades=4.0,
            )
            energies[i] = rung_energy
            weights[i] = np.average(rung_weight, axis=1, weights=solid_angle)
            weights_cos[i] = np.average(
                rung_weight * cos_theta[None, :], axis=1, weights=solid_angle
            )
            weights_sin[i] = np.average(
                rung_weight * sin_theta[None, :], axis=1, weights=solid_angle
            )
        ladders[flavour] = (energies, weights, weights_cos, weights_sin)
    return ladders


def model_aeff(
    theta: np.ndarray,
    ladders: dict[str, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Predicted effective area for one parameter vector.

    Parameters
    ----------
    theta : np.ndarray, shape (5,)
        ``(eps_0, log10_e_thr, b_scale, lam, reach_km)``.
    ladders : dict
        Output of :func:`precompute_ladders`.

    Returns
    -------
    aeff : np.ndarray, shape (COMMON_LOG10_E.size,)
        Effective area [cm^2].
    """
    eps_0, log10_e_thr, b_scale, lam, reach_km = theta
    threshold = 10.0**log10_e_thr
    n_nucleon = nucleon_number_density()

    total = np.zeros(COMMON_LOG10_E.size)
    channels = (
        ("mu", 1.0 - MEAN_INELASTICITY, 1.0),
        ("tau", MEAN_Z * (1.0 - MEAN_INELASTICITY), F_TAU * BR_TAU_TO_MU),
    )
    for flavour, muon_fraction, weight in channels:
        energies, arrival, arrival_cos, arrival_sin = ladders[flavour]
        muon_energy = muon_fraction * energies
        length = stochastic_muon_range_km(
            muon_energy.ravel(), threshold, b_scale=b_scale
        ).reshape(muon_energy.shape)
        radius = light_reach_radius_km(
            RADIUS_KM, muon_energy, reach_km, REACH_PIVOT_GEV
        )
        # BGR18 tilted about the pivot; lam = LAMBDA_BGR18 recovers the table.
        sigma = CROSS_SECTION.cc(energies) * (energies / LAMBDA_PIVOT_GEV) ** (
            lam - LAMBDA_BGR18
        )
        rate = n_nucleon * sigma * CM_PER_KM**3
        # Each geometry term carries its own declination average; see
        # precompute_ladders. V_det is isotropic and rides on the plain one.
        cap = np.pi * radius**2 * length * arrival_cos
        side = SIDE_COEFF * radius * HEIGHT_KM * length * arrival_sin
        v_det = np.pi * radius**2 * HEIGHT_KM * arrival
        total += weight * (rate * (cap + side + v_det)).sum(axis=1)
    return eps_0 * total


def log_probability(
    theta: np.ndarray,
    observed: np.ndarray,
    mask: np.ndarray,
    ladders: dict[str, tuple[np.ndarray, np.ndarray]],
    sigma_ln: float,
) -> float:
    """Flat-prior log posterior, Gaussian in ``ln A_eff``."""
    for value, name in zip(theta, PARAM_NAMES):
        low, high = PRIORS[name]
        if not low < value < high:
            return -np.inf
    # b_scale is a calibrated transport quantity, not a free knob: same
    # truncated Gaussian the event-rate fits use.
    log_prior = -0.5 * ((theta[2] - B_SCALE_MEAN) / B_SCALE_STD) ** 2
    predicted = model_aeff(theta, ladders)
    # Non-finite as well as non-positive: below b_scale ~ 0.287 the calibrated
    # kernel has d_mu >= b_mu and is no longer a loss spectrum at all, and this
    # prior still admits that corner. Example 33 guards the same way.
    if not np.all(np.isfinite(predicted)) or np.any(predicted <= 0.0):
        return -np.inf
    residual = np.log(observed[mask] / predicted[mask])
    return log_prior - 0.5 * float(np.sum((residual / sigma_ln) ** 2))


def summarize(chain: np.ndarray, best: np.ndarray) -> None:
    """Print posterior medians, credible intervals, and the expectations."""
    header = (
        f"\n{'parameter':>14} {'best fit':>9} {'median':>8} "
        f"{'16%':>8} {'84%':>8} {'expected':>9}"
    )
    print(header)
    for i, name in enumerate(PARAM_NAMES):
        lo, med, hi = np.percentile(chain[:, i], [16, 50, 84])
        expected = "--" if np.isnan(EXPECTED[i]) else f"{EXPECTED[i]:.2f}"
        print(f"{name:>14} {best[i]:9.3f} {med:8.3f} {lo:8.3f} {hi:8.3f} {expected:>9}")
    print(
        f"\n  (smearing matrix puts log10(E_thr/GeV) at "
        f"{SMEARING_LOG10_E_THR:.2f} independently)"
    )


def make_figure(
    chain: np.ndarray,
    best: np.ndarray,
    out_path: pathlib.Path,
) -> None:
    """Draw the corner plot, marking the best fit in every panel."""
    rc = {
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.labelsize": 14,
        "font.size": 12,
    }
    with plt.style.context(str(_STYLE)), plt.rc_context(rc):
        ranges = [(np.percentile(chain[:, i], 0.5), np.percentile(chain[:, i], 99.5))
                  for i in range(chain.shape[1])]
        base = np.array(plt.matplotlib.colors.to_rgb("C0"))
        # corner fills outside-in, so the alphas run 0 -> between -> inside.
        fills = [(*base, 0.0), (*base, 0.12), (*base, 0.30)]
        fig = corner.corner(
            chain, labels=CORNER_LABELS, range=ranges, color="C0",
            plot_datapoints=False, plot_density=False, levels=(0.68, 0.95),
            fill_contours=True, contourf_kwargs={"colors": fills},
            contour_kwargs={"linewidths": 1.0},
            hist_kwargs={"density": True, "lw": 1.4},
            label_kwargs={"fontsize": 15}, smooth=0.8,
        )
        # The best fit, in every 1D histogram and every 2D panel.
        corner.overplot_lines(fig, best, color="C3", lw=1.2, ls="-")
        corner.overplot_points(
            fig, best[None, :], marker="o", markersize=5.0,
            color="C3", markeredgecolor="w", markeredgewidth=0.6,
        )
        # The first-principles expectation, where one exists.
        corner.overplot_lines(fig, EXPECTED, color="0.35", lw=1.0, ls=":")

        handles = [
            plt.Line2D([], [], color="C3", lw=1.2, marker="o", markersize=5, label="best fit"),
            plt.Line2D([], [], color="0.35", lw=1.0, ls=":", label="first principles"),
        ]
        fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=15)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png"):
            path = out_path.with_suffix(suffix)
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"Figure saved to: {path.resolve()}")
        plt.close(fig)


def report_residuals(
    observed: np.ndarray,
    best: np.ndarray,
    mask: np.ndarray,
    ladders: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    """Print what the best fit leaves behind, energy by energy."""
    predicted = model_aeff(best, ladders)
    ratio = observed / predicted
    print(f"\n{'log10(E/GeV)':>13} {'A_eff^IC':>11} {'best fit':>11} {'ratio':>7}")
    for log10_e in (4.0, 5.0, 6.0, 7.0, 8.0):
        i = int(np.argmin(np.abs(COMMON_LOG10_E - log10_e)))
        flag = "  *" if COMMON_LOG10_E[i] > STATS_LOG10_E[1] else ""
        print(f"{COMMON_LOG10_E[i]:13.1f} {observed[i]:11.3g} {predicted[i]:11.3g} "
              f"{ratio[i]:7.2f}{flag}")
    residual = np.log10(ratio[mask])
    print(f"\n  rms {np.std(residual):.3f} dex, trend {residual[-1] - residual[0]:+.2f} dex "
          f"over the fitted band")


def main() -> None:
    args = parse_args()

    print(f"Loading IceCube IRFs from: {args.data_dir}")
    observed = icecube_upgoing(args.data_dir)
    mask = (COMMON_LOG10_E >= STATS_LOG10_E[0]) & (COMMON_LOG10_E <= STATS_LOG10_E[1])

    print("Precomputing Earth transmission ladders (parameter independent) ...")
    ladders = precompute_ladders()

    start = np.array([
        0.7, np.log10(DEFAULT_MUON_THRESHOLD_GEV), B_SCALE_MEAN, LAMBDA_BGR18,
        REACH_EXAMPLE28_KM,
    ])
    # Per-parameter scatter: reach_km lives on a scale two orders of magnitude
    # below the others, so a common 0.02 would throw walkers out of its prior.
    scatter = np.array([0.02, 0.02, 0.02, 0.02, 0.002])
    rng = np.random.default_rng(11)
    initial = start + scatter * rng.standard_normal((args.walkers, start.size))

    print(f"Sampling ({args.walkers} walkers x {args.steps} steps) ...")
    sampler = emcee.EnsembleSampler(
        args.walkers, start.size, log_probability,
        args=(observed, mask, ladders, args.sigma),
    )
    sampler.run_mcmc(initial, args.steps, progress=False)
    chain = sampler.get_chain(discard=args.steps // 3, flat=True)
    best = chain[np.argmax(sampler.get_log_prob(discard=args.steps // 3, flat=True))]

    summarize(chain, best)
    report_residuals(observed, best, mask, ladders)
    make_figure(chain, best, args.out)


if __name__ == "__main__":
    main()
