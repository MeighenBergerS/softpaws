"""Example 56 -- example 33's corner with P-ONE and TRIDENT added.

Example 33 fits the five handles of the first-principles effective area --
``eps_0``, ``log10(E_thr)``, ``b_scale``, ``lambda`` and the light reach
``Lambda`` -- to IceCube's upgoing table and to KM3NeT/ARCA230's trigger-level
sky average, independently, and puts the two posteriors on one corner. The
claim is that ``b_scale`` and ``lambda`` are properties of the loss kernel and
the Earth, not of a site, so two independent fits must land on top of each
other. This example adds two more sites and asks the same question of four.

* **P-ONE** -- the all-sky ``nu_mu`` effective area of the ICRC2023
  performance study at trigger level (three PMTs within 10 ns), the seven
  ten-line clusters of ~120 m radius and 1 km height at 2.16 km.
* **TRIDENT** -- the sky average of Extended Data Fig. 8 of the Nature
  Astronomy paper, built from its three ``cos(theta)`` bands with their
  solid-angle weights, for the 2 km-radius, 570 m-high array at 3.1 km, with
  the 6-degree angular-error cut.

Both curves are the digitized, smoothed ones of example 55. The two new sites
run through example 33's ARCA forward model with only the cylinder and the
overburden changed, and with two conventions matched to what the
collaborations simulate: ``nu_mu`` only, and pure absorption in the Earth
(example 55 shows the neutral-current ladder sits above their upgoing bands
by construction). IceCube and ARCA230 keep example 33's conventions exactly,
so their posteriors reproduce example 33; the four fits are fully
independent, as before.

What is compared is what example 33 compares: the corner carries
``log10(E_thr)``, ``b_scale`` and ``lambda``; ``eps_0`` and ``Lambda`` are
instrument numbers drawn as forest panels beside it. The compatibility test
becomes a four-way one: each site against the product of the other three on
the shared subspace, and the overall chi-square of the four means about their
precision-weighted centre, 2 (n - 1) degrees of freedom.

The digitized curves stop at 10^7 GeV, half a decade short of the two
reference sites' bands, and their assumed fractional error is example 33's
15% plus nothing for the digitization; their posteriors are correspondingly
wider and are the point, since ``b_scale`` and ``lambda`` are set by the UHE
end of a curve.

Usage
-----
    python examples/56_four_detector_posterior_corner.py
    python examples/56_four_detector_posterior_corner.py --steps 6000
"""

import argparse
import importlib.util
import json
import pathlib
from dataclasses import dataclass

import corner
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from softpaws.transport.attenuation import prem_column, regenerated_transmission
from softpaws.transport.source import nucleon_number_density
from softpaws.utils.constants import CM_PER_KM, RHO_WATER_G_CM3

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_DATA_DIR = _HERE.parent / "src" / "softpaws" / "data" / "dataverse_files"
_DEFAULT_OUT_DIR = _HERE / "output"


def load_example(stem: str, name: str):
    """Import a sibling example module by file stem."""
    spec = importlib.util.spec_from_file_location(name, _HERE / stem)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EX33 = load_example("33_two_detector_posterior_corner.py", "_example_33")
_EX55 = load_example("55_pone_trident_prediction.py", "_example_55")

#: Style-file colours, one per site: IceCube, ARCA230, P-ONE, TRIDENT.
COLORS = {"IceCube": "#7570b3", "ARCA230": "#1b9e77", "P-ONE": "#d95f02",
          "TRIDENT": "#e7298a"}

#: Fit band of the two digitized sites [log10 GeV]: from example 33's lower
#: edge to the last smoothed bin below the plotted edge.
DIGITIZED_FIT_BAND = (5.0, 6.9)

#: Solid-angle weights of TRIDENT's three ``cos(theta)`` bands.
TRIDENT_BAND_WEIGHTS = np.array([0.8, 0.4, 0.8]) / 2.0


def parse_args() -> argparse.Namespace:
    """Command-line arguments; the sampler ones mirror example 33's."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=pathlib.Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=pathlib.Path,
                        default=_DEFAULT_OUT_DIR / "56_four_detector_posterior_corner.pdf")
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--walkers", type=int, default=48)
    parser.add_argument("--sigma", type=float, default=0.15,
                        help="Assumed fractional error per node of every curve.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# A water site: example 33's ARCA model with the cylinder and overburden free
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WaterSite:
    """Upright-cylinder water detector for example 33's forward model."""

    name: str
    radius_km: float
    height_km: float
    n_blocks: int
    depth_km: float
    selection_level: str
    fit_band: tuple[float, float]
    reach_prior: tuple[float, float]
    #: ``tau -> mu`` channel weight; 0 for a ``nu_mu``-only simulation.
    f_tau: float
    #: Neutral-current regeneration in the Earth; False for pure absorption.
    regeneration: bool


def water_columns(site: WaterSite):
    """Example 33's :func:`arca_columns` for any depth."""
    theta_deg, weights = _EX33.arca_zenith_grid()
    cos_theta = np.cos(np.deg2rad(theta_deg))
    with np.errstate(divide="ignore", invalid="ignore"):
        downgoing_km = np.where(cos_theta > 0.0, site.depth_km / np.maximum(cos_theta, 1e-6),
                                np.inf)
    muon_column_km = np.minimum(downgoing_km, _EX33.ARCA_MAX_SEA_PATH_KM)
    water = np.where(np.isfinite(downgoing_km), muon_column_km, 0.0) * CM_PER_KM * RHO_WATER_G_CM3
    earth = np.array([prem_column(float(t) - 90.0) if t > 90.0 else 0.0 for t in theta_deg])
    return weights, np.where(theta_deg > 90.0, earth, water), muon_column_km


def water_ladders(site: WaterSite, neutrino_column):
    """Transmission ladders; a one-rung ``nu_mu`` ladder when regeneration is off."""
    if site.regeneration:
        return _EX33.arca_ladders(neutrino_column)
    grid = _EX33.ARCA_LOG10_E
    energies = np.empty((grid.size, 1))
    weights = np.empty((grid.size, 1, neutrino_column.size))
    for i, log10_e in enumerate(grid):
        energies[i], weights[i] = regenerated_transmission(
            10.0**log10_e, neutrino_column, _EX33.CROSS_SECTION, n_levels=1)
    return {"mu": (energies, weights), "tau": (energies, np.zeros_like(weights))}


def water_projected_area_km2(site: WaterSite, theta_deg, radius_km):
    """Convex-body projection of ``n_blocks`` upright cylinders [km^2]."""
    theta = np.deg2rad(theta_deg)
    cap = np.pi * np.asarray(radius_km) ** 2 * np.abs(np.cos(theta))
    side = 2.0 * np.asarray(radius_km) * site.height_km * np.sin(theta)
    return site.n_blocks * (cap + side)


def water_model(theta, site: WaterSite, ladders, zenith_weights, muon_column_km, select=None):
    """Example 33's :func:`arca_model` on a :class:`WaterSite` [cm^2]."""
    ex = _EX33
    eps_0, log10_e_thr, b_scale, lam, reach_km = theta
    threshold = 10.0**log10_e_thr
    n_nucleon = nucleon_number_density(RHO_WATER_G_CM3)
    theta_deg, _ = ex.arca_zenith_grid()
    nodes = slice(None) if select is None else select
    total = np.zeros(ex.ARCA_LOG10_E.size if select is None else int(np.sum(select)))
    channels = (("mu", 1.0 - ex.MEAN_INELASTICITY, 1.0),
                ("tau", ex.MEAN_Z * (1.0 - ex.MEAN_INELASTICITY), site.f_tau * ex.BR_TAU_TO_MU))
    for flavour, muon_fraction, weight in channels:
        if weight == 0.0:
            continue
        energies, arrival = ladders[flavour]
        energies, arrival = energies[nodes], arrival[nodes]
        muon_energy = muon_fraction * energies
        length = ex.truncated_muon_range_km(
            muon_energy[:, :, None], muon_column_km[None, None, :], threshold, b_scale)
        radius = ex.light_reach_radius_km(site.radius_km, muon_energy, reach_km,
                                          ex.REACH_PIVOT_GEV)
        area_km2 = water_projected_area_km2(site, theta_deg[None, None, :], radius[:, :, None])
        v_det_km3 = site.n_blocks * np.pi * radius**2 * site.height_km
        volume_km3 = area_km2 * length + v_det_km3[:, :, None]
        sigma = ex._tilted_cc(energies, lam)
        rate = n_nucleon * sigma[:, :, None] * volume_km3 * CM_PER_KM**3
        total += weight * np.average((arrival * rate).sum(axis=1), axis=1,
                                     weights=zenith_weights)
    return eps_0 * total


# ---------------------------------------------------------------------------
# The two digitized sky averages
# ---------------------------------------------------------------------------


def _on_grid(log10_e, log10_a):
    """Interpolate a smoothed curve onto example 33's ARCA grid [cm^2], NaN outside."""
    return 10.0 ** np.interp(_EX33.ARCA_LOG10_E, log10_e, log10_a, left=np.nan, right=np.nan)


def pone_allsky_cm2():
    """P-ONE's all-sky trigger-level curve on the grid."""
    return _on_grid(*_EX55.pone_allsky())


def trident_allsky_cm2():
    """TRIDENT's sky average from its three bands, solid-angle weighted, on the grid."""
    bands = [_on_grid(log10_e, log10_a) for _, _, log10_e, log10_a in _EX55.trident_bands()]
    return np.sum([w * b for w, b in zip(TRIDENT_BAND_WEIGHTS, bands)], axis=0)


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


def water_sites() -> list[WaterSite]:
    """The two new sites; example 35's geometries."""
    ex35 = load_example("35_point_source_effective_area.py", "_example_35")
    geometry = {s.name: s for s in ex35.build_sites()}
    pone, trident = geometry["P-ONE"], geometry["TRIDENT"]
    return [
        WaterSite("P-ONE", pone.radius_km, pone.height_km, pone.n_blocks, pone.depth_km,
                  "trigger", DIGITIZED_FIT_BAND, (-0.05, 0.40), 0.0, False),
        WaterSite("TRIDENT", trident.radius_km, trident.height_km, trident.n_blocks,
                  trident.depth_km, "6 deg cut", DIGITIZED_FIT_BAND, (-0.05, 0.40), 0.0, False),
    ]


def build_detectors(data_dir: pathlib.Path) -> list:
    """Example 33's two detectors plus the two water sites, recoloured."""
    ex = _EX33
    detectors = ex.build_detectors(data_dir)
    for d in detectors:
        d.color = COLORS[d.name]
    curves = {"P-ONE": pone_allsky_cm2, "TRIDENT": trident_allsky_cm2}
    start = detectors[1].start.copy()
    for site in water_sites():
        print(f"Building {site.name}: {site.n_blocks} x (r {site.radius_km:g} km, "
              f"h {site.height_km:g} km) at {site.depth_km:.2f} km, nu_mu only, "
              f"{'NC regeneration' if site.regeneration else 'pure absorption'} ...")
        observed = curves[site.name]()
        mask = ((ex.ARCA_LOG10_E >= site.fit_band[0]) & (ex.ARCA_LOG10_E <= site.fit_band[1])
                & np.isfinite(observed))
        zenith_weights, neutrino_column, muon_column_km = water_columns(site)
        ladders = water_ladders(site, neutrino_column)
        priors = dict(ex.PRIORS["ARCA230"])
        priors["reach_km"] = site.reach_prior
        ex.EXPECTED[site.name] = {"b_scale": 1.0, "lam": ex.LAMBDA_BGR18}
        detectors.append(ex.Detector(
            name=site.name, log10_e=ex.ARCA_LOG10_E, observed=observed, mask=mask,
            predict=(lambda theta, select=None, s=site, la=ladders, w=zenith_weights,
                     m=muon_column_km: water_model(theta, s, la, w, m, select)),
            priors=priors, start=start.copy(), color=COLORS[site.name],
            selection_level=site.selection_level))
    return detectors


# ---------------------------------------------------------------------------
# Four-way compatibility
# ---------------------------------------------------------------------------


def four_way_compatibility(detectors) -> dict:
    """Each site against the other three, and all four about their common centre.

    On the shared subspace the four posteriors are summarized by their means
    and covariances. The leave-one-out test is the Mahalanobis distance of
    one mean from the precision-weighted mean of the other three; the global
    test is the chi-square of all four about the common precision-weighted
    mean, with ``2 (n - 1)`` degrees of freedom.
    """
    ex = _EX33
    indices = [ex.PARAM_NAMES.index(name) for name in ex.PHYSICS_PARAMS]
    means = [d.chain[:, indices].mean(axis=0) for d in detectors]
    precisions = [np.linalg.inv(np.cov(d.chain[:, indices].T)) for d in detectors]

    def combine(subset):
        precision = sum(precisions[i] for i in subset)
        covariance = np.linalg.inv(precision)
        mean = covariance @ sum(precisions[i] @ means[i] for i in subset)
        return mean, covariance

    out = {"params": list(ex.PHYSICS_PARAMS), "leave_one_out": {}}
    n = len(detectors)
    for i, d in enumerate(detectors):
        others = [j for j in range(n) if j != i]
        mean, covariance = combine(others)
        delta = means[i] - mean
        chi_square = float(delta @ np.linalg.solve(covariance + np.linalg.inv(precisions[i]),
                                                   delta))
        p = float(chi2.sf(chi_square, len(indices)))
        out["leave_one_out"][d.name] = {
            "chi2": chi_square, "p_value": p,
            "sigma": float(np.sqrt(chi2.isf(p, 1))) if p > 0.0 else float("inf")}
    mean, covariance = combine(range(n))
    chi_square = float(sum((means[i] - mean) @ precisions[i] @ (means[i] - mean)
                           for i in range(n)))
    dof = len(indices) * (n - 1)
    p = float(chi2.sf(chi_square, dof))
    out["global"] = {"chi2": chi_square, "dof": dof, "p_value": p,
                     "sigma": float(np.sqrt(chi2.isf(p, 1))) if p > 0.0 else float("inf")}
    out["combined_mean"] = {name: float(mean[k]) for k, name in enumerate(ex.PHYSICS_PARAMS)}
    out["combined_sigma"] = {name: float(np.sqrt(covariance[k, k]))
                             for k, name in enumerate(ex.PHYSICS_PARAMS)}
    return out


def summarize(detectors, region, four_way) -> None:
    """Per-site posteriors, then the four-way tests."""
    ex = _EX33
    for d in detectors:
        print(f"\n{d.name} ({d.selection_level} level, {int(d.mask.sum())} nodes)")
        print(f"{'parameter':>14} {'best fit':>9} {'median':>8} {'16%':>8} {'84%':>8} "
              f"{'expected':>9}")
        for i, name in enumerate(ex.PARAM_NAMES):
            lo, med, hi = np.percentile(d.chain[:, i], [16, 50, 84])
            value = ex.EXPECTED[d.name].get(name)
            expected = "--" if value is None else f"{value:.3f}"
            print(f"{name:>14} {d.best[i]:9.3f} {med:8.3f} {lo:8.3f} {hi:8.3f} {expected:>9}")
    print("\nCombined (precision-weighted) on the shared subspace")
    for name in ex.PHYSICS_PARAMS:
        print(f"{name:>14}   {four_way['combined_mean'][name]:.3f} "
              f"+- {four_way['combined_sigma'][name]:.3f}")
    print("\nEach site against the other three")
    for name, entry in four_way["leave_one_out"].items():
        print(f"{name:>14}   chi2 {entry['chi2']:5.2f} / 2 dof, p = {entry['p_value']:.3f}, "
              f"{entry['sigma']:.1f} sigma")
    g = four_way["global"]
    print(f"\nAll four about their common centre: chi2 = {g['chi2']:.2f} for {g['dof']} dof, "
          f"p = {g['p_value']:.3f}, {g['sigma']:.1f} sigma")
    rails = [f"{name} of {det} against its {edge} prior edge"
             for name, entry in region["marginals"].items()
             for det, values in entry.items() for edge in ("low", "high")
             if values["rails_prior"][edge]]
    if rails:
        print("\n  WARNING: 68% interval rails against a prior edge --")
        for rail in rails:
            print(f"    {rail}")


def marginals(detectors) -> dict:
    """Example 33's per-parameter marginal summary, for any number of sites."""
    ex = _EX33
    region = {"marginals": {}}
    for i, name in enumerate(ex.PARAM_NAMES):
        region["marginals"][name] = {}
        for d in detectors:
            lo68, hi68 = ex._interval(d.chain, i, 0.68)
            lo95, hi95 = ex._interval(d.chain, i, 0.95)
            low, high = d.priors[name]
            span = high - low
            region["marginals"][name][d.name] = {
                "median": float(np.median(d.chain[:, i])), "best_fit": float(d.best[i]),
                "ci68": [lo68, hi68], "ci95": [lo95, hi95], "prior": [float(low), float(high)],
                "rails_prior": {"low": bool(lo68 - low < 0.02 * span),
                                "high": bool(high - hi68 < 0.02 * span)}}
    return region


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def make_figure(detectors, out_path: pathlib.Path) -> None:
    """Example 33's layout with four posteriors.

    The two reference sites keep example 33's styles (IceCube filled, ARCA
    dashed); the two new ones are unfilled solid and dotted, so four contour
    sets at two levels stay readable.
    """
    ex = _EX33
    indices = [ex.PARAM_NAMES.index(name) for name in ex.CORNER_PARAMS]
    labels = [ex.LABELS[name] for name in ex.CORNER_PARAMS]
    k = len(indices)
    styles = ["-", "--", "-", ":"]
    ranges = []
    for i in indices:
        lo = min(np.percentile(d.chain[:, i], 0.5) for d in detectors)
        hi = max(np.percentile(d.chain[:, i], 99.5) for d in detectors)
        pad = 0.05 * (hi - lo)
        ranges.append((lo - pad, hi + pad))

    rc = {"xtick.labelsize": 9, "ytick.labelsize": 9, "axes.labelsize": 12, "font.size": 10}
    with plt.style.context(str(_STYLE)), plt.rc_context(rc):
        fig, _ = plt.subplots(k, k, figsize=(7.2, 7.2))
        for row, d in enumerate(detectors):
            base = np.array(plt.matplotlib.colors.to_rgb(d.color))
            filled = row == 0
            fills = [(*base, 0.0), (*base, 0.12), (*base, 0.28)]
            corner.corner(
                d.chain[:, indices], labels=labels, range=ranges, color=d.color, fig=fig,
                plot_datapoints=False, plot_density=False, levels=(0.68, 0.95),
                fill_contours=filled, contourf_kwargs={"colors": fills} if filled else None,
                contour_kwargs={"linewidths": 1.1, "linestyles": styles[row]},
                hist_kwargs={"density": True, "lw": 1.3, "ls": styles[row], "histtype": "step"},
                label_kwargs={"fontsize": 12}, smooth=0.8, no_fill_contours=not filled)

        axes = np.array(fig.axes[: k * k]).reshape((k, k))
        for column, name in enumerate(ex.CORNER_PARAMS):
            values = {ex.EXPECTED[d.name][name] for d in detectors if name in ex.EXPECTED[d.name]}
            for value in values:
                axes[column, column].axvline(value, color="0.35", lw=1.0, ls=":", zorder=0)
        if "log10_e_thr" in ex.CORNER_PARAMS:
            column = ex.CORNER_PARAMS.index("log10_e_thr")
            axes[column, column].axvline(ex.SMEARING_LOG10_E_THR, color="0.35", lw=1.0,
                                         ls=(0, (1, 2)), zorder=0)

        column = axes[0, k - 1].get_position()
        legend_cell = axes[0, 1].get_position()
        left, width = column.x0, column.width
        bottom, top = axes[k - 2, k - 1].get_position().y0, column.y1
        span = top - bottom
        handles = [plt.Line2D([], [], color=d.color, lw=1.6, ls=styles[i],
                              label=f"{d.name} ({d.selection_level})")
                   for i, d in enumerate(detectors)]
        handles.append(plt.Line2D([], [], color="0.35", lw=1.0, ls=":", label="first principles"))
        legend_ax = fig.add_axes([legend_cell.x0, legend_cell.y0, legend_cell.width,
                                  legend_cell.height])
        legend_ax.axis("off")
        legend_ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8.5,
                         handlelength=1.8, borderaxespad=0.0)

        forest = fig.add_axes([left, top - 0.44 * span, width, 0.38 * span])
        ex._forest_panel(forest, detectors, "eps_0")
        forest.set_title("instrument response,\nnot compared", fontsize=9, pad=5.0,
                         color="0.35")
        reach = fig.add_axes([left, bottom + 0.10 * span, width, 0.38 * span])
        ex._forest_panel(reach, detectors, "reach_km", scale=1.0e3)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png"):
            path = out_path.with_suffix(suffix)
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"Figure saved to: {path.resolve()}")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    ex = _EX33
    detectors = build_detectors(args.data_dir)
    for seed, detector in enumerate(detectors, start=11):
        ex.run_fit(detector, args.steps, args.walkers, args.sigma, seed)

    region = marginals(detectors)
    four_way = four_way_compatibility(detectors)
    summarize(detectors, region, four_way)
    for detector in detectors:
        ex.report_residuals(detector)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    region["four_way"] = four_way
    region["provenance"] = {"steps": args.steps, "walkers": args.walkers,
                            "assumed_fractional_sigma": args.sigma,
                            "conventions": {d.name: d.selection_level for d in detectors},
                            "new_sites": "nu_mu only, pure absorption in the Earth"}
    json_path = args.out.with_name("56_four_way_region.json")
    json_path.write_text(json.dumps(region, indent=2) + "\n")
    print(f"\nRegion written to: {json_path.resolve()}")
    npz_path = args.out.with_name("56_chains.npz")
    np.savez_compressed(npz_path, param_names=np.array(ex.PARAM_NAMES),
                        **{f"{d.name}_chain": d.chain for d in detectors},
                        **{f"{d.name}_best": d.best for d in detectors})
    print(f"Chains written to: {npz_path.resolve()}")
    make_figure(detectors, args.out)


if __name__ == "__main__":
    main()
