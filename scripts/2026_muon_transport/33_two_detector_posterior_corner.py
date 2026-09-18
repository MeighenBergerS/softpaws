"""Example 33 -- the same fit run against IceCube and against KM3NeT/ARCA230.

Example 29 floats the few physical handles of the first-principles effective
area against the published IceCube upgoing table and asks where the data puts
them. Example 30 ports the same construction to ARCA by supplying only the
*instrument* numbers -- a cylinder instead of a sphere, a finite sea-water
overburden, a 4 pi sky average. This example runs example 29's fit twice, once
per site, and puts the two posteriors on one set of axes.

The point is which parameters are allowed to be compared.

**Shared physics.** ``b_scale`` rescales the drift coefficient and with it
``Phi'(0)``, and ``lambda`` is the effective slope of the charged-current cross
section. Neither is a property of a site: the loss kernel and the Earth are the
same under the Mediterranean as under the Pole. Two posteriors that land on top
of each other here is the claim the figure is making; two that do not would say
the transport is being tuned per detector.

**Instrument response.** ``eps_0`` is a selection efficiency and ``Lambda`` is a
light reach in a different medium around a differently shaped array, so nothing
requires them to agree and a shared axis would only invite the wrong reading.
They are reported as marginals beside the corner instead of inside it.

``log10(E_thr/GeV)`` sits between the two. It is an instrument number, but it is
on a common scale and its ordering is a check rather than a coincidence claim:
ARCA230 is compared at *trigger* level, so its threshold should come out below
IceCube's analysis-level one.

**The two curves are not at the same selection level, and that is deliberate.**
IceCube is the livetime-weighted upgoing ``nu_mu`` table of IceTracks-DR2, an
analysis-level response. ARCA230 is the ``nu_mu`` effective area at trigger
level digitized from KM3NeT Collaboration, Eur. Phys. J. C 84 (2024) 885
[arXiv:2402.08363] Fig. 7. Trigger level is the strongest test of a geometric
ceiling -- it carries no quality or containment cuts, so it is the largest area
the instrument ever reports -- which is what makes it the right curve for
``b_scale`` and ``lambda``. It is the wrong curve for reading ``eps_0`` as a
physical efficiency against IceCube's, and the figure labels it as such.

The two fits are run **independently**. Tying ``b_scale`` and ``lambda`` across
the two would shrink both contours, but their agreement would then be imposed
rather than demonstrated, and demonstrating it is the whole point.

**The truncated range.** ARCA's largest site-specific effect is that a downgoing
muon cannot be born further upstream than the sea surface, so Eq. (16)'s
first-passage integral is cut at the available column: ``L = E[tau(w) ^ X]``.
Example 30 evaluates that by Gil-Pelaez inversion of the log-loss CDF, which
costs seconds per energy and cannot go inside a sampler. Here the first-passage
depth is matched to a gamma law on its first two renewal moments -- mean
``w / Phi'(0) - Phi''(0) / (2 Phi'(0)^2)`` and variance
``-w Phi''(0) / Phi'(0)^3`` -- whose limited expected value is an incomplete
gamma function (:func:`truncated_muon_range_km`). It reproduces the exact integral to
better than 0.2% from ``10^5`` to ``10^8`` GeV at every truncation depth;
``--check-truncation`` runs that comparison.

The likelihood, the 15% assumed fractional error, and the flat-prior treatment
are example 29's, unchanged; posterior *widths* are set by that assumption and
are not measurements, while the posterior *locations* are what the figure is
about.

Outputs, besides the figure, are written next to it and are what the paper
reads: ``33_overlap_region.json`` carries the per-parameter intervals, the
intersection of the two 68% regions, the product posterior on the shared pair
and the 2-dof compatibility test, and ``33_chains.npz`` carries both chains so
the figure can be redrawn without resampling.

Usage
-----
    python scripts/2026_muon_transport/33_two_detector_posterior_corner.py
    python scripts/2026_muon_transport/33_two_detector_posterior_corner.py --steps 6000 --sigma 0.10
    python scripts/2026_muon_transport/33_two_detector_posterior_corner.py --check-truncation
"""

import argparse
import json
import pathlib

import corner
import matplotlib.pyplot as plt
import numpy as np

from softpaws.comparison.likelihood import B_SCALE_MEAN, B_SCALE_STD
from softpaws.comparison.posterior import (
    credible_interval,
    inside_box,
    intersection,
    log_gaussian_in_log,
    marginal_summary,
    pairwise_compatibility,
    product_posterior,
    sample_posterior,
)
from softpaws.data.published import KM3NET_DIR
from softpaws.response import site_models
from softpaws.transport.muon_range import DEFAULT_MUON_THRESHOLD_GEV

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_DATA_DIR = _HERE.parents[1] / "src" / "softpaws" / "data" / "dataverse_files"
_ARCA230_TRIGGER_TABLE = KM3NET_DIR / "arca_trigger_level_eff.csv"
_DEFAULT_OUT_DIR = _HERE / "output"

#: Style-file colour of each posterior. Plotting stays in the scripts, so the
#: colours are not carried on the :class:`Detector` record.
SITE_COLORS = {"IceCube": "C0", "ARCA230": "C1"}

# ---------------------------------------------------------------------------
# The model now lives in softpaws.response.site_models. Everything below is
# re-exported so the sibling examples that load this script by path keep
# resolving; Phase 3 of the cleanup retires that helper and this block with it.
# ---------------------------------------------------------------------------

CROSS_SECTION = site_models.CROSS_SECTION

PARAM_NAMES = site_models.PARAM_NAMES
CORNER_PARAMS = site_models.CORNER_PARAMS
INSTRUMENT_PARAMS = site_models.INSTRUMENT_PARAMS
PHYSICS_PARAMS = site_models.PHYSICS_PARAMS
LABELS = site_models.PARAM_LABELS

LAMBDA_BGR18 = site_models.LAMBDA_BGR18
LAMBDA_PIVOT_GEV = site_models.LAMBDA_PIVOT_GEV
REACH_PIVOT_GEV = site_models.REACH_PIVOT_GEV
F_TAU = site_models.F_TAU
B_SCALE_FLOOR = site_models.B_SCALE_FLOOR
REACH_EXAMPLE28_KM = site_models.REACH_EXAMPLE28_KM
SMEARING_LOG10_E_THR = site_models.SMEARING_LOG10_E_THR
EXPECTED = site_models.EXPECTED
PRIORS = site_models.PRIORS

IC_LOG10_E = site_models.IC_LOG10_E
IC_FIT_BAND = site_models.IC_FIT_BAND
IC_HEIGHT_KM = site_models.IC_HEIGHT_KM
IC_N_SIDES = site_models.IC_N_SIDES
IC_RADIUS_KM = site_models.IC_RADIUS_KM
IC_SIDE_COEFF = site_models.IC_SIDE_COEFF
IC_ICE_BELOW_KM = site_models.IC_ICE_BELOW_KM
IC_N_DEC = site_models.IC_N_DEC
IC_N_RUNG = site_models.IC_N_RUNG

ARCA_BLOCK_RADIUS_KM = site_models.ARCA_BLOCK_RADIUS_KM
ARCA_BLOCK_HEIGHT_KM = site_models.ARCA_BLOCK_HEIGHT_KM
ARCA_N_BLOCKS = site_models.ARCA_N_BLOCKS
ARCA_DEPTH_KM = site_models.ARCA_DEPTH_KM
ARCA_WATER_BELOW_KM = site_models.ARCA_WATER_BELOW_KM
ARCA_MAX_SEA_PATH_KM = site_models.ARCA_MAX_SEA_PATH_KM
ARCA_LOG10_E = site_models.ARCA_LOG10_E
ARCA_FIT_BAND = site_models.ARCA_FIT_BAND
ARCA_N_ZENITH = site_models.ARCA_N_ZENITH
ARCA_N_RUNG = site_models.ARCA_N_RUNG

Detector = site_models.Detector
icecube_upgoing = site_models.icecube_upgoing
arca230_trigger = site_models.arca230_trigger
icecube_ladders = site_models.icecube_ladders
icecube_model = site_models.icecube_model
arca_zenith_grid = site_models.arca_zenith_grid
arca_columns = site_models.arca_columns
arca_ladders = site_models.arca_ladders
arca_projected_area_km2 = site_models.arca_projected_area_km2
arca_model = site_models.arca_model
_tilted_cc = site_models.tilted_cc


def parse_args() -> argparse.Namespace:
    """Command-line arguments; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=pathlib.Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=_DEFAULT_OUT_DIR / "33_two_detector_posterior_corner.pdf",
    )
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--walkers", type=int, default=48)
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.15,
        help="Assumed fractional uncertainty on each published effective area.",
    )
    parser.add_argument(
        "--check-truncation",
        action="store_true",
        help="Compare the closed-form truncated range against the Gil-Pelaez "
        "integral of example 30 and exit. Slow: seconds per energy.",
    )
    return parser.parse_args()


def check_truncation() -> None:
    """Print :func:`softpaws.response.site_models.truncation_table`."""
    print(f"\n{'log10(E_mu)':>12} {'X / L':>7} {'exact':>9} {'closed':>9} {'ratio':>7}")
    for row in site_models.truncation_table():
        print(
            f"{row['log10_e']:12.1f} {row['fraction']:7.2f} {row['exact']:9.3f} "
            f"{row['closed']:9.3f} {row['ratio']:7.4f}"
        )
    print("  exact: Eq. (16) cut at X by Gil-Pelaez inversion, as in example 30.")


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def log_probability(theta: np.ndarray, detector: Detector, sigma_ln: float) -> float:
    """Flat-prior log posterior, Gaussian in ``ln A_eff``.

    ``b_scale`` is the exception to the flat priors: it is a calibrated
    transport quantity rather than a free knob, so it carries the same truncated
    Gaussian the event-rate fits use (:mod:`softpaws.comparison.likelihood`),
    identically for both detectors.
    """
    if not inside_box(theta, [detector.priors[name] for name in PARAM_NAMES]):
        return -np.inf
    log_prior = -0.5 * ((theta[2] - B_SCALE_MEAN) / B_SCALE_STD) ** 2
    predicted = detector.predict(theta, detector.mask)
    return log_prior + log_gaussian_in_log(detector.observed[detector.mask], predicted, sigma_ln)


def run_fit(detector: Detector, steps: int, walkers: int, sigma: float, seed: int) -> None:
    """Sample one detector's posterior and store the chain on it."""
    # Per-parameter scatter: reach_km lives on a scale two orders of magnitude
    # below the others, so a common 0.02 would throw walkers out of its prior.
    scatter = np.array([0.02, 0.02, 0.02, 0.02, 0.002])
    print(f"Sampling {detector.name} ({walkers} walkers x {steps} steps) ...")
    detector.chain, detector.best = sample_posterior(
        log_probability, detector.start, scatter, walkers, steps, seed, args=(detector, sigma)
    )


# ---------------------------------------------------------------------------
# Overlap region
# ---------------------------------------------------------------------------


def _interval(chain: np.ndarray, index: int, level: float) -> tuple[float, float]:
    return credible_interval(chain[:, index], level)


def overlap_region(detectors: list[Detector]) -> dict:
    """Quantify where the two posteriors agree, for the paper to read.

    Three things, in increasing strength.

    *Per-parameter intersection.* The overlap of the two 68% credible intervals,
    parameter by parameter. Reported for all five, but only the ``PHYSICS_PARAMS``
    are required to overlap -- the instrument parameters describe different
    hardware and an empty intersection there is not a discrepancy.

    *Product posterior.* The two fits are independent, so on the shared subspace
    their joint constraint is the product of the two marginals. Evaluated by
    kernel density estimate on a grid, which is what the paper should quote as
    the combined measurement of ``b_scale`` and ``lambda``.

    *Compatibility.* The Mahalanobis distance between the two posterior means on
    the shared subspace, against the summed covariances, as a 2-dof chi-square.
    This is the number that answers "do they agree", and it is a
    profile-style statement rather than a Bayes factor for the reason
    ``docs/`` records: the Bayes factor moves by a factor of two with the prior
    box, and this does not.

    Parameters
    ----------
    detectors : list of Detector
        Exactly two, each with a sampled ``chain``.

    Returns
    -------
    region : dict
        JSON-serializable summary; see the keys built below.
    """
    first, second = detectors
    region: dict = {
        "detectors": {
            d.name: {
                "selection_level": d.selection_level,
                "fit_band_log10_e": [
                    float(d.log10_e[d.mask].min()),
                    float(d.log10_e[d.mask].max()),
                ],
                "n_points": int(d.mask.sum()),
            }
            for d in detectors
        },
        "marginals": {},
        "intersection_68": {},
        "physics_params": list(PHYSICS_PARAMS),
    }
    summaries = {
        d.name: marginal_summary(d.chain, PARAM_NAMES, d.priors, d.best) for d in detectors
    }
    for name in PARAM_NAMES:
        region["marginals"][name] = {d.name: summaries[d.name][name] for d in detectors}
        region["intersection_68"][name] = {
            **intersection([summaries[d.name][name]["ci68"] for d in detectors]),
            "comparable": name in CORNER_PARAMS,
        }
    indices = [PARAM_NAMES.index(name) for name in PHYSICS_PARAMS]
    region["product_posterior"] = product_posterior(
        [d.chain[:, indices] for d in detectors], PHYSICS_PARAMS
    )
    pair = pairwise_compatibility(first.chain[:, indices], second.chain[:, indices])
    region["compatibility"] = {
        "params": list(PHYSICS_PARAMS),
        "dof": pair["dof"],
        "chi2": pair["chi2"],
        "p_value": pair["p_value"],
        "sigma": pair["sigma"],
        "delta": {name: float(pair["delta"][k]) for k, name in enumerate(PHYSICS_PARAMS)},
    }
    region["note"] = (
        f"{first.name} is compared at {first.selection_level} level and "
        f"{second.name} at {second.selection_level} level. eps_0 is a selection "
        "efficiency against its own curve and the two are therefore not on the "
        "same footing; only the physics_params carry a comparison."
    )
    return region


def summarize(detectors: list[Detector], region: dict) -> None:
    """Print the per-detector posteriors and the overlap region."""
    for d in detectors:
        print(f"\n{d.name} ({d.selection_level} level)")
        print(
            f"{'parameter':>14} {'best fit':>9} {'median':>8} {'16%':>8} "
            f"{'84%':>8} {'expected':>9}"
        )
        for i, name in enumerate(PARAM_NAMES):
            lo, med, hi = np.percentile(d.chain[:, i], [16, 50, 84])
            value = EXPECTED[d.name].get(name)
            expected = "--" if value is None else f"{value:.3f}"
            print(f"{name:>14} {d.best[i]:9.3f} {med:8.3f} {lo:8.3f} {hi:8.3f} {expected:>9}")

    print("\nOverlap of the 68% intervals")
    for name in PARAM_NAMES:
        entry = region["intersection_68"][name]
        tag = "" if entry["comparable"] else "   (instrument, not compared)"
        if entry["empty"]:
            print(f"{name:>14}   empty{tag}")
        else:
            print(f"{name:>14}   [{entry['low']:.3f}, {entry['high']:.3f}]{tag}")

    print("\nProduct posterior on the shared subspace (both fits are independent)")
    for name in PHYSICS_PARAMS:
        entry = region["product_posterior"][name]
        lo, hi = entry["ci68"]
        print(f"{name:>14}   {entry['median']:.3f}  [{lo:.3f}, {hi:.3f}]")

    compatibility = region["compatibility"]
    print(
        f"\nCompatibility on {', '.join(PHYSICS_PARAMS)}: "
        f"chi2 = {compatibility['chi2']:.2f} for {compatibility['dof']} dof, "
        f"p = {compatibility['p_value']:.3f}, {compatibility['sigma']:.1f} sigma"
    )

    rails = [
        f"{name} of {detector} against its {edge} prior edge"
        for name, entry in region["marginals"].items()
        for detector, values in entry.items()
        for edge in ("low", "high")
        if values["rails_prior"][edge]
    ]
    if rails:
        print("\n  WARNING: 68% interval rails against a prior edge --")
        for rail in rails:
            print(f"    {rail}")
        print("    Quote these as bounds, not as measurements.")

    print(f"  {region['note']}")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _forest_panel(
    ax: plt.Axes,
    detectors: list[Detector],
    name: str,
    scale: float = 1.0,
) -> None:
    """One horizontal-whisker row per detector for an instrument parameter.

    The detector names ride above their own whiskers rather than on the y axis.
    As tick labels they sit outside the panel, where the last column of the
    corner leaves them no room and they collide with the diagonal beside it.
    """
    index = PARAM_NAMES.index(name)
    for row, d in enumerate(detectors):
        y = len(detectors) - 1 - row
        color = SITE_COLORS[d.name]
        lo95, hi95 = _interval(d.chain, index, 0.95)
        lo68, hi68 = _interval(d.chain, index, 0.68)
        median = float(np.median(d.chain[:, index])) * scale
        ax.plot([lo95 * scale, hi95 * scale], [y, y], color=color, lw=0.9, alpha=0.55)
        ax.plot([lo68 * scale, hi68 * scale], [y, y], color=color, lw=2.6, solid_capstyle="butt")
        ax.plot([median], [y], marker="o", ms=4.0, color=color,
                markeredgecolor="w", markeredgewidth=0.5, zorder=3)
        ax.text(lo95 * scale, y + 0.18, d.name, color=color, fontsize=8,
                ha="left", va="bottom")
        value = EXPECTED[d.name].get(name)
        if value is not None:
            ax.plot([value * scale], [y], marker="|", ms=9, color="0.35", zorder=4)

    ax.set_yticks([])
    # Headroom for the name above the topmost whisker.
    ax.set_ylim(-0.5, len(detectors) - 0.15)
    ax.set_xlabel(LABELS[name], fontsize=8, labelpad=1.5)
    ax.tick_params(axis="x", labelsize=8, pad=1.5)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def make_figure(detectors: list[Detector], out_path: pathlib.Path) -> None:
    """Draw the overlaid corner with the instrument marginals beside it.

    Three decluttering choices, all of them deliberate.

    The corner carries only ``CORNER_PARAMS``, so every panel in it is one where
    two overlapping contours mean something. The instrument parameters go into
    the upper triangle, which a corner plot otherwise leaves blank.

    The two posteriors are drawn differently rather than in two colours of the
    same weight: the first is filled, the second is unfilled and dashed. At two
    contour levels each, two translucent fills of equal weight are unreadable.

    No reference lines are drawn across the 2D panels. Marking a best fit in
    every panel costs two line segments per posterior per panel, which is where
    the ink actually goes; the derived values appear on the diagonal, where they
    read as values rather than as grid lines.
    """
    indices = [PARAM_NAMES.index(name) for name in CORNER_PARAMS]
    labels = [LABELS[name] for name in CORNER_PARAMS]
    k = len(indices)

    # Shared ranges, so the two corners agree about their own axes.
    ranges = []
    for i in indices:
        lo = min(np.percentile(d.chain[:, i], 0.5) for d in detectors)
        hi = max(np.percentile(d.chain[:, i], 99.5) for d in detectors)
        pad = 0.05 * (hi - lo)
        ranges.append((lo - pad, hi + pad))

    rc = {"xtick.labelsize": 9, "ytick.labelsize": 9, "axes.labelsize": 12, "font.size": 10}
    with plt.style.context(str(_STYLE)), plt.rc_context(rc):
        fig, _ = plt.subplots(k, k, figsize=(6.6, 6.6))
        for row, d in enumerate(detectors):
            color = SITE_COLORS[d.name]
            base = np.array(plt.matplotlib.colors.to_rgb(color))
            filled = row == 0
            # corner fills outside-in, so the alphas run 0 -> between -> inside.
            fills = [(*base, 0.0), (*base, 0.12), (*base, 0.28)]
            corner.corner(
                d.chain[:, indices], labels=labels, range=ranges, color=color, fig=fig,
                plot_datapoints=False, plot_density=False, levels=(0.68, 0.95),
                fill_contours=filled,
                contourf_kwargs={"colors": fills} if filled else None,
                contour_kwargs={"linewidths": 1.1, "linestyles": "-" if filled else "--"},
                hist_kwargs={"density": True, "lw": 1.3,
                             "ls": "-" if filled else "--", "histtype": "step"},
                label_kwargs={"fontsize": 12}, smooth=0.8, no_fill_contours=not filled,
            )

        axes = np.array(fig.axes[: k * k]).reshape((k, k))
        # The derived values, on the diagonal only. Both detectors expect the
        # same b_scale and lam -- that is what makes them shared -- so the
        # distinct values are drawn once rather than stacked.
        for column, name in enumerate(CORNER_PARAMS):
            values = {EXPECTED[d.name][name] for d in detectors if name in EXPECTED[d.name]}
            for value in values:
                axes[column, column].axvline(value, color="0.35", lw=1.0, ls=":", zorder=0)
        # The DR2 smearing matrix's independent handle on IceCube's threshold.
        if "log10_e_thr" in CORNER_PARAMS:
            column = CORNER_PARAMS.index("log10_e_thr")
            axes[column, column].axvline(
                SMEARING_LOG10_E_THR, color="0.35", lw=1.0, ls=(0, (1, 2)), zorder=0
            )

        # The blank upper triangle, put to work. It is L-shaped, not
        # rectangular: cell (1, 1) is the b_mu diagonal and has to be left
        # alone. The last column, rows 0 to k-2, is the tall free block, and
        # cell (0, 1) beside it takes the legend.
        column = axes[0, k - 1].get_position()
        legend_cell = axes[0, 1].get_position()
        left, width = column.x0, column.width
        bottom, top = axes[k - 2, k - 1].get_position().y0, column.y1
        span = top - bottom

        legend_handles = [
            plt.Line2D([], [], color=SITE_COLORS[d.name], lw=1.6,
                       ls="-" if i == 0 else "--",
                       label=f"{d.name} ({d.selection_level})")
            for i, d in enumerate(detectors)
        ]
        legend_handles.append(
            plt.Line2D([], [], color="0.35", lw=1.0, ls=":", label="first principles")
        )
        legend_ax = fig.add_axes(
            [legend_cell.x0, legend_cell.y0, legend_cell.width, legend_cell.height]
        )
        legend_ax.axis("off")
        legend_ax.legend(handles=legend_handles, loc="upper left", frameon=False,
                         fontsize=9, handlelength=1.8, borderaxespad=0.0)

        forest = fig.add_axes([left, top - 0.40 * span, width, 0.32 * span])
        _forest_panel(forest, detectors, "eps_0")
        forest.set_title("instrument response,\nnot compared", fontsize=9, pad=5.0, color="0.35")

        reach = fig.add_axes([left, bottom + 0.18 * span, width, 0.32 * span])
        # Metres per e-fold: the two reaches differ by an order of magnitude and
        # kilometres would put IceCube's on top of zero.
        _forest_panel(reach, detectors, "reach_km", scale=1.0e3)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png"):
            path = out_path.with_suffix(suffix)
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"Figure saved to: {path.resolve()}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_detectors(data_dir: pathlib.Path) -> list[Detector]:
    """Load both published curves and bind each to its forward model.

    A thin wrapper on :mod:`softpaws.response.site_models` that reports
    progress, since the two ladders take a few seconds each.
    """
    start = np.array(
        [0.7, np.log10(DEFAULT_MUON_THRESHOLD_GEV), B_SCALE_MEAN, LAMBDA_BGR18,
         REACH_EXAMPLE28_KM]
    )
    print(f"Loading IceCube IRFs from: {data_dir}")
    print("Precomputing IceCube transmission ladders (parameter independent) ...")
    icecube = site_models.build_icecube_detector(data_dir, start)

    print(f"Loading ARCA230 trigger-level curve from: {_ARCA230_TRIGGER_TABLE}")
    print("Precomputing ARCA transmission ladders (parameter independent) ...")
    arca = site_models.build_arca_detector(start)
    return [icecube, arca]


def report_residuals(detector: Detector) -> None:
    """Print what each best fit leaves behind, energy by energy."""
    predicted = detector.predict(detector.best)
    ratio = detector.observed / predicted
    print(
        f"\n{detector.name}: {'log10(E/GeV)':>13} {'published':>11} "
        f"{'best fit':>11} {'ratio':>7}"
    )
    for log10_e in (5.0, 6.0, 7.0, 8.0):
        i = int(np.argmin(np.abs(detector.log10_e - log10_e)))
        if not np.isfinite(detector.observed[i]):
            continue
        flag = "" if detector.mask[i] else "  *"
        print(
            f"{'':>{len(detector.name) + 1}} {detector.log10_e[i]:13.1f} "
            f"{detector.observed[i]:11.3g} {predicted[i]:11.3g} {ratio[i]:7.2f}{flag}"
        )
    residual = np.log10(ratio[detector.mask])
    print(
        f"  rms {np.std(residual):.3f} dex, trend {residual[-1] - residual[0]:+.2f} dex "
        f"over the fitted band. * outside it."
    )


def main() -> None:
    args = parse_args()

    if args.check_truncation:
        check_truncation()
        return

    detectors = build_detectors(args.data_dir)
    for seed, detector in enumerate(detectors, start=11):
        run_fit(detector, args.steps, args.walkers, args.sigma, seed)

    region = overlap_region(detectors)
    summarize(detectors, region)
    for detector in detectors:
        report_residuals(detector)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    region["provenance"] = {
        "steps": args.steps,
        "walkers": args.walkers,
        "assumed_fractional_sigma": args.sigma,
        "burn_in_fraction": 1.0 / 3.0,
        "reach_pivot_gev": REACH_PIVOT_GEV,
        "lambda_pivot_gev": LAMBDA_PIVOT_GEV,
        "lambda_bgr18": LAMBDA_BGR18,
    }
    json_path = args.out.with_name("33_overlap_region.json")
    json_path.write_text(json.dumps(region, indent=2) + "\n")
    print(f"\nOverlap region written to: {json_path.resolve()}")

    npz_path = args.out.with_name("33_chains.npz")
    np.savez_compressed(
        npz_path,
        param_names=np.array(PARAM_NAMES),
        **{f"{d.name}_chain": d.chain for d in detectors},
        **{f"{d.name}_best": d.best for d in detectors},
    )
    print(f"Chains written to: {npz_path.resolve()}")

    make_figure(detectors, args.out)


if __name__ == "__main__":
    main()
