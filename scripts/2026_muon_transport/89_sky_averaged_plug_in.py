"""Example 89 -- the one-line estimate from first principles, and plugged in.

Example 88 fits the threshold and the reach of the paper's estimate,
Eq. (skyavg), to each published effective area. This example asks what the
estimate gives with nothing fitted, and it makes the paper's figure for it.
Every published curve is drawn against the estimate in its first-principles
form, the reach as a power on top of the footprint,

    A_eff = eps_0 (E / 1 PeV)^k  n_N sigma_CC T [ <A_proj> L(E) + V_det ],
    k = Lambda (R + c h / 4) / (R^2 / 2 + c R h / 4),

with the inputs of :data:`FIRST_PRINCIPLES` per site: the normalization at
the collaboration's level, a 300 GeV trigger threshold at the sea sites and
IceCube's analysis-level turn-on, the reach the site's measured optics
predict, and for P-ONE, whose clusters are compact, ARCA's reach relative to
the radius. The printout also carries, per site, the fitted estimate of
example 88, the estimate at the full model's instrument numbers, the estimate
plugged in with no reach and one 1 TeV threshold, and the naive estimate of
the paper's Eq. (estimate) with the instrumented volume alone, so the cost
of each simplification is on record. Nothing is refitted, and example 88's
cache must exist.

The printout closes with two checks of the power-law form: ``k`` from the
formula against the slope the direct evaluation of the reach actually has,
and the power law against the direct evaluation; and with a
back-of-the-envelope fit of ``N (E / 1 PeV)^k`` on the plug-in curve.

Usage
-----
    python scripts/2026_muon_transport/89_sky_averaged_plug_in.py
    python scripts/2026_muon_transport/89_sky_averaged_plug_in.py --naive-threshold-gev 300
"""

import argparse
import importlib.util
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from softpaws.response import reduced
from softpaws.response import site_models as sm
from softpaws.transport.source import MEAN_INELASTICITY

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_DATA_DIR = _HERE.parents[1] / "src" / "softpaws" / "data" / "dataverse_files"
_DEFAULT_OUT_DIR = _HERE / "output"


def load_example(stem: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / stem)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EX88 = load_example("88_sky_averaged_analytic_fit.py", "_example_88")
_EX83, _EX73, _EX56 = _EX88._EX83, _EX88._EX73, _EX88._EX56

#: The variants, their line style and their legend entry.
VARIANTS = (
    ("first principles", "--", "First principles"),
    ("first principles, direct", "-.", "First principles, direct"),
    ("fitted", "-.", "Fitted"),
    ("full-model numbers", "-.", "Full-model numbers"),
    ("plug-in", ":", "Plug-in, no reach"),
    ("instrumented volume", ":", "Naive"),
)

#: The variants the figure draws beside the published curves; the ratio figure
#: adds the naive one. The printout carries all six.
FIGURE_VARIANTS = tuple(v for v in VARIANTS if v[0] == "first principles")
RATIO_VARIANTS = tuple(
    v for v in VARIANTS if v[0] in ("first principles", "instrumented volume")
)

#: The first-principles inputs per site: normalization, threshold [GeV] and
#: where the reach comes from. ``"optics"`` is the midpoint of the band the
#: site's measured optics predict, ``"none"`` is an analysis-level selection
#: whose quality cuts remove the halo, and ``"ARCA ratio"`` scales ARCA's
#: optics reach by the ratio of radii, for a cluster too compact for the bulk
#: optics to apply. IceCube's threshold is the turn-on of its analysis-level
#: selection; the sea sites get one trigger threshold.
FIRST_PRINCIPLES = {
    "IceCube": (0.956, 3.5e3, "none"),
    "ARCA230": (1.0, 300.0, "optics"),
    "P-ONE": (1.0, 300.0, "ARCA ratio"),
    "TRIDENT": (0.7, 300.0, "optics"),
}

#: Range constants of the paper's Eq. (skyavg), the loss moments at 100 TeV [km].
RANGE_SLOPE_KM = 2.17
RANGE_CONSTANT_KM = 0.81


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=pathlib.Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--chains", type=pathlib.Path,
                        default=_DEFAULT_OUT_DIR / "77_chains_sigma05.npz")
    parser.add_argument("--trident-chain", type=pathlib.Path,
                        default=_DEFAULT_OUT_DIR / "82_trident2025_chain.npz")
    parser.add_argument("--sky-cache", type=pathlib.Path,
                        default=_DEFAULT_OUT_DIR / "88_sky_averaged_analytic_fit.npz",
                        help="Example 88's cache, for the fitted threshold and reach.")
    parser.add_argument("--naive-threshold-gev", type=float, default=1.0e3,
                        help="The one muon threshold [GeV] every site gets when plugging in.")
    parser.add_argument("--naive-eps", type=float, default=1.0,
                        help="The normalization every site gets when plugging in.")
    parser.add_argument("--sigma", type=float, default=0.05,
                        help="Assumed fractional error per tabulated point, for the envelope fit.")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR)
    return parser.parse_args()


def first_principles_reach_km(name: str) -> float:
    """The reach of :data:`FIRST_PRINCIPLES` in km."""
    _, _, source = FIRST_PRINCIPLES[name]
    if source == "none":
        return 0.0
    if source == "optics":
        return 1.0e-3 * float(np.mean(reduced.PREDICTED_REACH_M[name]))
    arca_ratio = 1.0e-3 * float(np.mean(reduced.PREDICTED_REACH_M["ARCA230"])) \
        / sm.ARCA230_WATER_SITE.radius_km
    return arca_ratio * site_geometry(name)[0]


def site_geometry(name: str) -> tuple[float, float, float]:
    """Radius [km], height [km] and side coefficient of the site's projected area."""
    if name == "IceCube":
        return sm.IC_RADIUS_KM, sm.IC_HEIGHT_KM, sm.IC_SIDE_COEFF
    site = (
        sm.ARCA230_WATER_SITE if name == "ARCA230"
        else [s for s in sm.water_sites() if s.name == name][0]
    )
    return site.radius_km, site.height_km, 2.0


def analytic_k(name: str, reach_km: float, threshold_gev: float, plug_threshold_gev: float):
    """The two slopes that make ``k``: the reach's and the threshold's.

    Returns
    -------
    k_reach, k_threshold : float
        The log-slope of the mean projected area from the reach at the pivot,
        and the change of the range's log-slope at 1 PeV when the threshold
        moves from ``plug_threshold_gev`` to ``threshold_gev``.
    """
    radius, height, side = site_geometry(name)
    k_reach = reach_km * (radius + side * height / 4.0) / (
        radius**2 / 2.0 + side * radius * height / 4.0
    )
    muon = (1.0 - MEAN_INELASTICITY) * 1.0e6

    def length(threshold):
        return RANGE_SLOPE_KM * np.log(muon / threshold) + RANGE_CONSTANT_KM

    k_threshold = RANGE_SLOPE_KM / length(threshold_gev) \
        - RANGE_SLOPE_KM / length(plug_threshold_gev)
    return float(k_reach), float(k_threshold)


def measured_slope(log10_e, ratio, mask) -> float:
    """Log-log slope of a ratio of curves inside the fit window."""
    good = mask & np.isfinite(ratio) & (ratio > 0.0)
    return float(np.polyfit(log10_e[good], np.log10(ratio[good]), 1)[0])


def envelope_fit(log10_e, observed, plug_in, mask, sigma: float):
    """Fit ``N (E / 1 PeV)^k`` on top of the plug-in curve, least squares in the log."""

    def chi2(x):
        model = x[0] * plug_in * 10.0 ** (x[1] * (log10_e - 6.0))
        residual = np.log(observed[mask] / model[mask])
        residual = residual[np.isfinite(residual)]
        return np.sum(residual**2) / sigma**2

    best = minimize(chi2, (1.0, 0.1), method="Nelder-Mead")
    model = best.x[0] * plug_in * 10.0 ** (best.x[1] * (log10_e - 6.0))
    return float(best.x[0]), float(best.x[1]), model, float(best.fun)


def figure(rows: dict, out_dir: pathlib.Path) -> None:
    """The published curves against the three variants, one panel."""
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        anchors = {}
        for name, (log10_e, observed, curves) in rows.items():
            good = np.isfinite(observed) & (observed > 0.0)
            color = _EX56.SITE_COLORS[name]
            ax.plot(log10_e[good], observed[good], color="k", lw=1.2)
            for key, ls, _ in FIGURE_VARIANTS:
                ax.plot(log10_e[good], curves[key][good], color=color, lw=1.1, ls=ls)
            anchors[name] = (log10_e[good], observed[good])
        ax.plot([], [], color="k", lw=1.2, label="Published")
        for _, ls, label in FIGURE_VARIANTS:
            ax.plot([], [], color="0.4", lw=1.1, ls=ls, label=label)
        ax.set_yscale("log")
        ax.set_xlim(4.0, 8.0)
        ax.set_ylim(1.0e4, 1.0e8)
        ax.set_xlabel(r"$\log_{10}(E_\nu\,/\,\mathrm{GeV})$")
        ax.set_ylabel(r"$A_{\rm eff}$ [cm$^2$]")
        ax.set_box_aspect(1)
        ax.legend(frameon=False, loc="lower right")
        fig.canvas.draw()
        for name in rows:
            x0, factor, ha, slope_of, anchor_on, y_at = _EX88.LABEL_SPEC[name]
            angle = _EX73._curve_angle_deg(ax, *anchors[slope_of], x0) if slope_of else 0.0
            x, y = anchors[anchor_on]
            height = factor * float(np.interp(y_at if y_at is not None else x0, x, y))
            ax.text(x0, height, name, color=_EX56.SITE_COLORS[name], ha=ha, va="center",
                    rotation=angle, rotation_mode="anchor")
        _EX73._save(fig, out_dir, "89a_sky_averaged_plug_in")


def ratio_figure(rows: dict, out_dir: pathlib.Path) -> None:
    """Published over model for the three variants."""
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        for name, (log10_e, observed, curves) in rows.items():
            good = np.isfinite(observed) & (observed > 0.0)
            color = _EX56.SITE_COLORS[name]
            for key, ls, _ in RATIO_VARIANTS:
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = observed[good] / curves[key][good]
                ax.plot(log10_e[good], ratio, color=color, lw=1.1, ls=ls)
        ax.axhline(1.0, color="k", lw=0.8)
        for name in rows:
            ax.plot([], [], color=_EX56.SITE_COLORS[name], lw=1.1, label=name)
        for _, ls, label in RATIO_VARIANTS:
            ax.plot([], [], color="0.4", lw=1.1, ls=ls, label=label)
        ax.set_yscale("log")
        ax.set_xlim(4.0, 8.0)
        ax.set_ylim(0.5, 60.0)
        ax.set_yticks([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
        ax.set_yticklabels(["0.5", "1", "2", "5", "10", "20", "50"])
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlabel(r"$\log_{10}(E_\nu\,/\,\mathrm{GeV})$")
        ax.set_ylabel("Published / estimate")
        ax.set_box_aspect(1)
        ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=6)
        _EX73._save(fig, out_dir, "89b_sky_averaged_plug_in_ratio")


def main() -> None:
    args = parse_args()
    detectors = _EX83.reduced_detectors(args.data_dir, args.chains)
    trident, _ = _EX83.trident_2025(args.trident_chain)
    detectors.append(trident)
    predictors = _EX88.factorized_predictors(trident.log10_e)
    sky = np.load(args.sky_cache)

    print(f"Plug-in: reach 0, E_thr = {args.naive_threshold_gev:.0f} GeV, "
          f"eps_0 = {args.naive_eps:g} at every site")
    rows, summary, thetas_by_site = {}, [], {}
    for d in detectors:
        predict = predictors[d.name]
        observed = np.asarray(d.observed, dtype=float)
        theta_full = np.median(d.chain, axis=0)
        theta_sky = np.median(sky[f"{d.name}_chain"], axis=0)
        theta_naive = theta_full.copy()
        theta_naive[0] = args.naive_eps
        theta_naive[1] = np.log10(args.naive_threshold_gev)
        theta_naive[4] = 0.0
        eps_first, threshold_first, _ = FIRST_PRINCIPLES[d.name]
        theta_first = theta_full.copy()
        theta_first[0] = eps_first
        theta_first[1] = np.log10(threshold_first)
        theta_first[4] = first_principles_reach_km(d.name)
        thetas = {
            "first principles, direct": theta_first, "fitted": theta_sky,
            "full-model numbers": theta_full, "plug-in": theta_naive,
        }
        curves = {
            key: np.asarray(predict(theta, None), dtype=float) for key, theta in thetas.items()
        }
        thetas["instrumented volume"] = theta_naive
        curves["instrumented volume"] = np.asarray(
            predict(theta_naive, None, naive=True), dtype=float
        )
        theta_footprint = theta_first.copy()
        theta_footprint[4] = 0.0
        k_reach, _ = analytic_k(d.name, theta_first[4], threshold_first, threshold_first)
        thetas["first principles"] = theta_first
        curves["first principles"] = np.asarray(
            predict(theta_footprint, None), dtype=float
        ) * 10.0 ** (k_reach * (d.log10_e - 6.0))
        rows[d.name] = (d.log10_e, observed, curves)
        thetas_by_site[d.name] = thetas

        print(f"\n=== {d.name} ({d.selection_level} level) ===")
        print(f"  {'variant':20s} {'eps_0':>6s} {'E_thr':>7s} {'reach':>6s}  "
              f"{'pub/model':>9s} {'scatter':>8s}   published / model at 10^5, 10^6, 10^7")
        for key, _, _ in VARIANTS:
            theta = thetas[key]
            level, scatter = _EX88.window_stats(observed, curves[key], d.mask)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = observed / curves[key]
            by_energy = "  ".join(
                f"{np.interp(x, d.log10_e, ratio):.2f}"
                for x in _EX88.QUOTED_LOG10_E if d.log10_e.min() <= x <= d.log10_e.max()
            )
            print(f"  {key:20s} {theta[0]:6.2f} {10.0**theta[1]:7.0f} {1.0e3 * theta[4]:6.0f}  "
                  f"{level:9.2f} {scatter:8.3f}   {by_energy}")
            summary.append((d.name, key, level, scatter))

    print("\nThe reach as a power: k from the formula against the direct evaluation of Eq. (reach)")
    print(f"  {'site':>8s} {'numbers':>24s} {'eps_0':>6s} {'E_thr':>6s} {'reach':>6s}  "
          f"{'k formula':>9s} {'k direct':>9s}  {'power law / direct':>18s}")
    for d in detectors:
        log10_e, observed, curves = rows[d.name]
        predict = predictors[d.name]
        for key in ("fitted", "first principles, direct"):
            theta = thetas_by_site[d.name][key]
            if theta[4] == 0.0:
                continue
            theta_footprint = theta.copy()
            theta_footprint[4] = 0.0
            footprint = np.asarray(predict(theta_footprint, None), dtype=float)
            k_reach, _ = analytic_k(d.name, theta[4], 10.0**theta[1], 10.0**theta[1])
            with np.errstate(divide="ignore", invalid="ignore"):
                k_direct = measured_slope(log10_e, curves[key] / footprint, d.mask)
                power = footprint * 10.0 ** (k_reach * (log10_e - 6.0))
                dev = np.abs(power[d.mask] / curves[key][d.mask] - 1.0)
            dev = dev[np.isfinite(dev)]
            label = key.replace(", direct", "")
            print(f"  {d.name:>8s} {label:>24s} {theta[0]:6.2f} {10.0**theta[1]:6.0f} "
                  f"{1.0e3*theta[4]:6.0f}  {k_reach:9.3f} {k_direct:9.3f}  within {dev.max():5.1%}")

    print("\nBack of the envelope: N (E / 1 PeV)^k on the plug-in curve, fitted per table")
    print(f"  {'site':>8s}  {'N':>5s}  {'k':>6s}  {'scatter':>8s}  {'deviance':>9s}")
    envelopes = {}
    for d in detectors:
        log10_e, observed, curves = rows[d.name]
        n_fit, k_fit, model, dev = envelope_fit(log10_e, observed, curves["plug-in"], d.mask,
                                                args.sigma)
        _, scatter = _EX88.window_stats(observed, model, d.mask)
        envelopes[d.name] = (n_fit, k_fit)
        print(f"  {d.name:>8s}  {n_fit:5.2f}  {k_fit:6.3f}  {scatter:8.3f}  "
              f"{dev:6.1f}/{d.mask.sum() - 2}")

    print("\nSummary: published over the estimate, geometric mean inside the fit window")
    print(f"  {'site':>8s}  " + "  ".join(f"{key:>18s}" for key, _, _ in VARIANTS))
    for d in detectors:
        levels = [lv for name, key, lv, _ in summary if name == d.name]
        print(f"  {d.name:>8s}  " + "  ".join(f"{lv:18.2f}" for lv in levels))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_dir / "89_sky_averaged_plug_in.npz",
        naive_threshold_gev=np.array(args.naive_threshold_gev),
        naive_eps=np.array(args.naive_eps),
        **{f"{name}_envelope_N_k": np.array(v) for name, v in envelopes.items()},
        **{f"{name}_log10_e": v[0] for name, v in rows.items()},
        **{f"{name}_published": v[1] for name, v in rows.items()},
        **{f"{name}_{key.replace(', ', '_').replace(' ', '_').replace('-', '_')}": v[2][key]
           for name, v in rows.items() for key, _, _ in VARIANTS},
    )
    figure(rows, args.out_dir)
    ratio_figure(rows, args.out_dir)


if __name__ == "__main__":
    main()
