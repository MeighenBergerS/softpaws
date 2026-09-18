"""Example 88 -- the sky-averaged response fitted to the published tables.

Example 83 draws the published effective areas against the full model,
Eq. (aeff), with two instrument numbers per site. This example asks what the
sky-averaged response of the paper's Eq. (skyavg) does on its own at all four
sites. It draws every published curve in black against that response twice:
at the instrument numbers the full model fitted (examples 77 and 82), and
refitted to the table with the sky-averaged response itself under the same
error model and the same held normalization (free at TRIDENT, as in
example 82).

The sky-averaged response factorizes the angle out. The Earth enters as one
transmission per table, ``T(E) = <exp(-X / Lambda_nu)>`` over the table's
solid angle, the projected area as the solid-angle mean of a convex body,
a quarter of its surface at the reach radius (Cauchy), and the range as the
closed form of Eq. (Lclosed) with one loss kernel in pure water. At a sea site the
downgoing half of the sky supplies only ``d / cos(theta)`` of water, so its
range is the hemisphere mean of ``min(L, d / cos theta)``, and the two
halves carry their own transmission. The exact solid-angle average of the
per-direction analytic response (example 86) is printed beside it, so the
cost of factorizing is on record.

The figure shows the published curves against two fitted forms: the
sky-averaged response with its threshold and reach fitted, and the naive
estimate of the paper's Eq. (estimate), the instrumented volume with the
same Earth factor and no range, fitted on its own terms with its
normalization and reach free. The unfitted curves of both are kept in the
printout and the cache.

The point is the fitted numbers, not the curves. A threshold and a reach have
physical meaning, the turn-on of the selection and the growth of the light
halo, and the reach is predicted from each site's measured optics. The
printout puts the sky-averaged refit beside the full-model fit and the
prediction, so the reader can see which physics the missing effects were
standing in for.

Usage
-----
    python scripts/2026_muon_transport/88_sky_averaged_analytic_fit.py
"""

import argparse
import dataclasses
import importlib.util
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.constants import CM_PER_KM, RHO_WATER_G_CM3
from softpaws.response import reduced
from softpaws.response import site_models as sm
from softpaws.transport.attenuation import survival_probability
from softpaws.transport.earth import prem_column
from softpaws.transport.soft_volume import light_reach_radius_km
from softpaws.transport.source import MEAN_INELASTICITY, nucleon_number_density

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_DATA_DIR = _HERE.parents[1] / "src" / "softpaws" / "data" / "dataverse_files"
_DEFAULT_OUT_DIR = _HERE / "output"


def load_example(stem: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / stem)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EX83 = load_example("83_four_detector_aeff_reduced.py", "_example_83")
_EX85 = load_example("85_closed_form_response.py", "_example_85")
_EX86 = load_example("86_closed_form_four_detectors.py", "_example_86")
_EX73 = _EX83._EX73
_EX56 = _EX83._EX56

QUOTED_LOG10_E = (5.0, 6.0, 7.0)

#: Label placement per site; see example 73's ``LABEL_SPEC``.
LABEL_SPEC = {
    "IceCube": (5.0, 0.55, "center", "IceCube", "IceCube", None),
    "ARCA230": (6.2, 1.5, "center", "ARCA230", "ARCA230", None),
    "P-ONE": (6.9, 1.0, "left", "ARCA230", "P-ONE", None),
    "TRIDENT": (4.75, 1.75, "center", "TRIDENT", "TRIDENT", None),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=pathlib.Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--chains", type=pathlib.Path,
                        default=_DEFAULT_OUT_DIR / "77_chains_sigma05.npz")
    parser.add_argument("--trident-chain", type=pathlib.Path,
                        default=_DEFAULT_OUT_DIR / "82_trident2025_chain.npz")
    parser.add_argument("--sigma", type=float, default=0.05,
                        help="Assumed fractional error per tabulated point.")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--walkers", type=int, default=16)
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR)
    parser.add_argument("--figure-only", action="store_true",
                        help="Redraw from the cached fits instead of refitting.")
    return parser.parse_args()


def exact_predictors(trident_log10_e: np.ndarray) -> dict:
    """The per-direction analytic response of Eq. (aeffcf), averaged exactly.

    Example 86's functions, wrapped to the ``predict(theta, select)`` signature
    the fit uses, so the cost of factorizing the angle out can be printed.
    """

    def icecube(theta, select=None):
        out = _EX86.closed_form_icecube(theta)
        return out if select is None else out[select]

    def water(site):
        zenith_weights, neutrino_column, muon_column_km = sm.water_columns(site)

        def predict(theta, select=None):
            out = _EX86.closed_form_water(
                theta, site, zenith_weights, neutrino_column, muon_column_km
            )
            return out if select is None else out[select]

        return predict

    def trident(theta, select=None):
        out = _EX86.closed_form_trident(theta, trident_log10_e, _EX83.COS_MAX)
        return out if select is None else out[select]

    pone = [s for s in sm.water_sites() if s.name == "P-ONE"][0]
    return {
        "IceCube": icecube,
        "ARCA230": water(sm.ARCA230_WATER_SITE),
        "P-ONE": water(pone),
        "TRIDENT": trident,
    }


def mean_projected_area_km2(
    radius_km, height_km: float, n_blocks: int, side_coeff: float,
    cos_theta: np.ndarray | None = None, weights: np.ndarray | None = None,
):
    """Solid-angle mean of the projected area, ``n_blocks`` bodies of Eq. (aproj).

    Over the whole sky ``<|cos theta|> = 1/2`` and ``<sin theta> = pi/4``, which
    is Cauchy's quarter of the surface. With ``cos_theta`` and ``weights`` the
    mean is taken over that part of the zenith grid instead.
    """
    if cos_theta is None:
        mean_cos, mean_sin = 0.5, np.pi / 4.0
    else:
        mean_cos = np.average(np.abs(cos_theta), weights=weights)
        mean_sin = np.average(np.sqrt(1.0 - cos_theta**2), weights=weights)
    radius_km = np.asarray(radius_km, dtype=float)
    footprint = np.pi * radius_km**2 * mean_cos
    side = side_coeff * radius_km * height_km * mean_sin
    return n_blocks * (footprint + side)


def capped_range_mean_km(length_km, depth_km: float, cos_theta: np.ndarray, weights: np.ndarray):
    """Hemisphere mean of ``min(L, d / cos theta)`` over the downgoing directions."""
    length_km = np.asarray(length_km, dtype=float)
    capped = np.minimum(length_km[:, None], depth_km / cos_theta[None, :])
    return np.average(capped, axis=1, weights=weights)


def factorized_predictors(trident_log10_e: np.ndarray) -> dict:
    """The sky-averaged response of Eq. (skyavg), per site."""
    n_nucleon = nucleon_number_density(RHO_WATER_G_CM3)
    energy = 10.0**sm.ARCA_LOG10_E

    ic_energy = 10.0**sm.IC_LOG10_E
    ic_columns = np.array([prem_column(float(d)) for d in _EX85.DEC_DEG])
    ic_transmission = np.average(
        survival_probability(ic_energy[:, None], ic_columns[None, :], sm.LAMBDA_BGR18,
                             sm.CROSS_SECTION),
        axis=1, weights=_EX85.SOLID_ANGLE,
    )

    def icecube(theta, select=None, naive=False):
        eps_0, log10_e_thr, _, lam, reach_km = theta
        muon = (1.0 - MEAN_INELASTICITY) * ic_energy
        length = _EX85.closed_form_range_km(muon, 10.0**log10_e_thr)
        radius = light_reach_radius_km(sm.IC_RADIUS_KM, muon, reach_km, sm.REACH_PIVOT_GEV)
        if naive:
            length, radius = 0.0 * length, sm.IC_RADIUS_KM + 0.0 * radius
        area = mean_projected_area_km2(radius, sm.IC_HEIGHT_KM, 1, sm.IC_SIDE_COEFF)
        v_det = np.pi * radius**2 * sm.IC_HEIGHT_KM
        out = eps_0 * n_nucleon * sm.tilted_cc(ic_energy, lam) * ic_transmission \
            * (area * length + v_det) * CM_PER_KM**3
        return out if select is None else out[select]

    theta_deg, weights = sm.arca_zenith_grid()
    cos_theta = np.cos(np.deg2rad(theta_deg))

    def water(site, band=None):
        """One sea site, its two hemispheres carried separately.

        ``band`` masks the zenith grid, for TRIDENT's flat-selection cells.
        """
        _, neutrino_column, _ = sm.water_columns(site)
        survival = survival_probability(energy[:, None], neutrino_column[None, :],
                                        sm.LAMBDA_BGR18, sm.CROSS_SECTION)
        keep = np.ones(cos_theta.size, bool) if band is None else band
        halves = []
        for mask in (keep & (cos_theta < 0.0), keep & (cos_theta > 0.0)):
            fraction = weights[mask].sum() / weights[keep].sum()
            if fraction == 0.0:
                halves.append((mask, 0.0, np.zeros(energy.size)))
                continue
            transmission = np.average(survival[:, mask], axis=1, weights=weights[mask])
            halves.append((mask, fraction, transmission))
        down_mask = halves[1][0]

        def predict(theta, select=None, naive=False):
            eps_0, log10_e_thr, _, lam, reach_km = theta
            muon = (1.0 - MEAN_INELASTICITY) * energy
            length = _EX85.closed_form_range_km(muon, 10.0**log10_e_thr)
            radius = light_reach_radius_km(site.radius_km, muon, reach_km, sm.REACH_PIVOT_GEV)
            if naive:
                length, radius = 0.0 * length, site.radius_km + 0.0 * radius
            area = mean_projected_area_km2(
                radius, site.height_km, site.n_blocks, 2.0,
                None if band is None else cos_theta[keep], None if band is None else weights[keep],
            )
            v_det = site.n_blocks * np.pi * radius**2 * site.height_km
            length_down = length
            if down_mask.any():
                length_down = capped_range_mean_km(
                    length, site.depth_km, cos_theta[down_mask], weights[down_mask]
                )
            out = np.zeros(energy.size)
            for (mask, fraction, transmission), seg in zip(halves, (length, length_down)):
                out += fraction * transmission * (area * seg + v_det)
            out *= eps_0 * n_nucleon * sm.tilted_cc(energy, lam) * CM_PER_KM**3
            return out if select is None else out[select]

        return predict

    # TRIDENT: the 2025 map's flat-selection band, averaged as example 82 does.
    cos_cells, _, _ = reduced.trident_2025_cells()
    model = reduced.MapModel(trident_log10_e)
    d_cos = np.abs(np.diff(reduced.COS_EDGES))
    rows = np.abs(cos_cells) <= _EX83.COS_MAX
    band_predictors = [
        water(model.site, band=(w > 0.0)) for w, row in zip(model.weights, rows) if row
    ]

    def trident(theta, select=None, naive=False):
        curves = np.empty((len(band_predictors), trident_log10_e.size))
        for i, predict in enumerate(band_predictors):
            curve = predict(theta, naive=naive)
            curves[i] = 10.0 ** np.interp(trident_log10_e, model.grid, np.log10(curve))
        out = np.average(curves, axis=0, weights=d_cos[rows])
        return out if select is None else out[select]

    pone = [s for s in sm.water_sites() if s.name == "P-ONE"][0]
    return {
        "IceCube": icecube,
        "ARCA230": water(sm.ARCA230_WATER_SITE),
        "P-ONE": water(pone),
        "TRIDENT": trident,
    }


def refit(detector, predict, sigma: float, steps: int, walkers: int, seed: int):
    """The instrument numbers refitted with the analytic response.

    Two numbers with the normalization held, as example 77, except at
    TRIDENT, where the normalization is free as in example 82.
    """
    analytic = dataclasses.replace(detector, predict=predict, chain=None, best=None)
    free = reduced.FREE3 if detector.name == "TRIDENT" else reduced.FREE2
    fixed = {
        "eps_0": 0.7 if detector.name == "TRIDENT" else reduced.EPS_FIXED[detector.name],
        "log10_e_thr": 2.5 if detector.name == "TRIDENT"
        else np.log10(sm.DEFAULT_MUON_THRESHOLD_GEV),
        "b_scale": 1.0, "lam": sm.LAMBDA_BGR18, "reach_km": 0.03,
    }
    best, chain = reduced.fit(analytic, free, fixed, sigma, steps, walkers, seed)
    full = np.array([reduced.full_theta(free, row, fixed) for row in chain])
    return analytic, reduced.full_theta(free, best, fixed), full


def refit_naive(detector, predict, sigma: float, steps: int, walkers: int, seed: int):
    """The naive estimate fitted on its own terms.

    The instrumented volume has no threshold, so its free numbers are the
    normalization and the reach, with priors wide enough that the fit can go
    wherever the table sends it.
    """
    def predict_naive(theta, select=None):
        return predict(theta, select, naive=True)

    priors = dict(detector.priors)
    priors["eps_0"] = (0.01, 500.0)
    priors["reach_km"] = (-0.08, 5.0)
    naive = dataclasses.replace(detector, predict=predict_naive, priors=priors,
                                chain=None, best=None)
    free = ("eps_0", "reach_km")
    fixed = {
        "eps_0": 0.7 if detector.name == "TRIDENT" else reduced.EPS_FIXED[detector.name],
        "log10_e_thr": np.log10(sm.DEFAULT_MUON_THRESHOLD_GEV),
        "b_scale": 1.0, "lam": sm.LAMBDA_BGR18, "reach_km": 0.03,
    }
    best, chain = reduced.fit(naive, free, fixed, sigma, steps, walkers, seed + 100)
    full = np.array([reduced.full_theta(free, row, fixed) for row in chain])
    return naive, reduced.full_theta(free, best, fixed), full


def window_stats(observed, predicted, mask):
    ratio = np.log10(observed[mask] / predicted[mask])
    good = np.isfinite(ratio)
    return 10 ** np.mean(ratio[good]), float(np.std(ratio[good]))


def quantiles(chain: np.ndarray, column: int) -> tuple[float, float, float]:
    lo, med, hi = np.percentile(chain[:, column], [16, 50, 84])
    return float(lo), float(med), float(hi)


def figure(rows: dict, out_dir: pathlib.Path) -> None:
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        anchors = {}
        for name, (log10_e, observed, _, fitted, _, naive_fitted) in rows.items():
            good = np.isfinite(observed) & (observed > 0.0)
            color = _EX56.SITE_COLORS[name]
            ax.plot(log10_e[good], observed[good], color="k", lw=1.2)
            ax.plot(log10_e[good], fitted[good], color=color, lw=1.1, ls="--")
            ax.plot(log10_e[good], naive_fitted[good], color=color, lw=1.1, ls=":")
            anchors[name] = (log10_e[good], observed[good])
        ax.plot([], [], color="k", lw=1.2, label="Published")
        ax.plot([], [], color="0.4", lw=1.1, ls="--", label="Sky-Averaged")
        ax.plot([], [], color="0.4", lw=1.1, ls=":", label="Naive")
        ax.set_yscale("log")
        ax.set_xlim(4.0, 8.0)
        ax.set_ylim(5.0e4, 2.0e8)
        ax.set_xlabel(r"$\log_{10}(E_\nu\,/\,\mathrm{GeV})$")
        ax.set_ylabel(r"$A_{\rm eff}$ [cm$^2$]")
        ax.set_box_aspect(1)
        ax.legend(frameon=False, loc="lower right")
        fig.canvas.draw()
        for name in rows:
            x0, factor, ha, slope_of, anchor_on, y_at = LABEL_SPEC[name]
            angle = _EX73._curve_angle_deg(ax, *anchors[slope_of], x0) if slope_of else 0.0
            x, y = anchors[anchor_on]
            height = factor * float(np.interp(y_at if y_at is not None else x0, x, y))
            ax.text(x0, height, name, color=_EX56.SITE_COLORS[name], ha=ha, va="center",
                    rotation=angle, rotation_mode="anchor")
        _EX73._save(fig, out_dir, "88a_sky_averaged_analytic_fit")


def main() -> None:
    args = parse_args()
    detectors = _EX83.reduced_detectors(args.data_dir, args.chains)
    trident, _ = _EX83.trident_2025(args.trident_chain)
    detectors.append(trident)
    predictors = factorized_predictors(trident.log10_e)
    exact = exact_predictors(trident.log10_e)
    cache_path = args.out_dir / "88_sky_averaged_analytic_fit.npz"
    cached = np.load(cache_path) if args.figure_only and cache_path.exists() else None

    rows, fitted_chains, naive_chains, summary = {}, {}, {}, []
    for seed, d in enumerate(detectors, start=41):
        predict = predictors[d.name]
        observed = np.asarray(d.observed, dtype=float)
        theta_full = np.median(d.chain, axis=0)
        unfitted = np.asarray(predict(theta_full, None), dtype=float)
        naive = np.asarray(predict(theta_full, None, naive=True), dtype=float)
        exact_unfitted = np.asarray(exact[d.name](theta_full, None), dtype=float)
        if cached is not None:
            chain = cached[f"{d.name}_chain"]
            analytic = dataclasses.replace(d, predict=predict, chain=None, best=None)
        else:
            analytic, best, chain = refit(d, predict, args.sigma, args.steps, args.walkers, seed)
        theta_fit = np.median(chain, axis=0)
        fitted = np.asarray(predict(theta_fit, None), dtype=float)
        if cached is not None and f"{d.name}_naive_chain" in cached.files:
            naive_chain = cached[f"{d.name}_naive_chain"]
            naive_det = dataclasses.replace(
                d, predict=lambda th, sel=None, _p=predict: _p(th, sel, naive=True),
                chain=None, best=None,
            )
        else:
            naive_det, _, naive_chain = refit_naive(
                d, predict, args.sigma, args.steps, args.walkers, seed
            )
        theta_naive = np.median(naive_chain, axis=0)
        naive_fitted = np.asarray(predict(theta_naive, None, naive=True), dtype=float)
        rows[d.name] = (d.log10_e, observed, unfitted, fitted, naive, naive_fitted)
        fitted_chains[d.name] = chain
        naive_chains[d.name] = naive_chain

        n = int(d.mask.sum())
        n_free = 3 if d.name == "TRIDENT" else 2
        dev_un, _ = reduced.deviance(analytic, theta_full, args.sigma)
        dev_fit, _ = reduced.deviance(analytic, theta_fit, args.sigma)
        e_lo, e_med, e_hi = (10.0**v for v in quantiles(chain, 1))
        r_lo, r_med, r_hi = (1.0e3 * v for v in quantiles(chain, 4))
        e_full_lo, e_full_med, e_full_hi = (10.0**v for v in quantiles(d.chain, 1))
        r_full_lo, r_full_med, r_full_hi = (1.0e3 * v for v in quantiles(d.chain, 4))
        pred_lo, pred_hi = reduced.PREDICTED_REACH_M[d.name]
        print(f"\n=== {d.name} ({d.selection_level} level, {n} nodes, sigma {args.sigma}) ===")
        level, scatter = window_stats(observed, unfitted, d.mask)
        print(f"  sky-averaged at the full-model numbers: published/model {level:.3f}, "
              f"scatter {scatter:.3f} dex, deviance {dev_un:.1f} / {n - n_free} dof")
        factor = "  ".join(
            f"{np.interp(x, d.log10_e, unfitted / exact_unfitted):.3f}"
            for x in QUOTED_LOG10_E if d.log10_e.min() <= x <= d.log10_e.max()
        )
        print(f"    factorized over exact solid-angle average at 10^5, 10^6, 10^7: {factor}")
        level, scatter = window_stats(observed, fitted, d.mask)
        print(f"  sky-averaged refitted: published/model {level:.3f}, scatter {scatter:.3f} dex, "
              f"deviance {dev_fit:.1f} / {n - n_free} dof")
        print(f"    E_thr [GeV]: full model {e_full_med:.0f} [{e_full_lo:.0f}, {e_full_hi:.0f}]"
              f"   sky-averaged {e_med:.0f} [{e_lo:.0f}, {e_hi:.0f}]")
        print(f"    reach [m]:   full model {r_full_med:.0f} [{r_full_lo:.0f}, {r_full_hi:.0f}]"
              f"   sky-averaged {r_med:.0f} [{r_lo:.0f}, {r_hi:.0f}]"
              f"   optics predict {pred_lo:.0f} to {pred_hi:.0f}")
        dev_naive, _ = reduced.deviance(naive_det, theta_naive, args.sigma)
        level, scatter = window_stats(observed, naive_fitted, d.mask)
        nv_lo, nv_med, nv_hi = quantiles(naive_chain, 0)
        nr_lo, nr_med, nr_hi = (1.0e3 * v for v in quantiles(naive_chain, 4))
        print(f"  naive refitted (eps_0 and reach free): published/model {level:.3f}, "
              f"scatter {scatter:.3f} dex, deviance {dev_naive:.1f} / {n - 2} dof")
        print(f"    eps_0 {nv_med:.2f} [{nv_lo:.2f}, {nv_hi:.2f}]   "
              f"reach {nr_med:.0f} m [{nr_lo:.0f}, {nr_hi:.0f}]")
        if d.name == "TRIDENT":
            n_lo, n_med, n_hi = quantiles(chain, 0)
            nf_lo, nf_med, nf_hi = quantiles(d.chain, 0)
            print(f"    eps_0:       full model {nf_med:.2f} [{nf_lo:.2f}, {nf_hi:.2f}]"
                  f"   sky-averaged {n_med:.2f} [{n_lo:.2f}, {n_hi:.2f}]")
        print("  published / sky-averaged by energy (unfitted, fitted):")
        for x in QUOTED_LOG10_E:
            if not (d.log10_e.min() <= x <= d.log10_e.max()):
                continue
            un = np.interp(x, d.log10_e, observed / unfitted)
            fi = np.interp(x, d.log10_e, observed / fitted)
            print(f"    10^{x:.0f}: {un:.3f}, {fi:.3f}")
        summary.append((d.name, e_full_med, e_med, r_full_med, r_med, pred_lo, pred_hi))

    print("\nSummary: instrument numbers, full model versus sky-averaged refit")
    print(f"  {'site':>8}  {'E_thr full':>10}  {'E_thr sky':>10}  "
          f"{'reach full':>10}  {'reach sky':>10}  {'optics':>12}")
    for name, ef, ea, rf, ra, plo, phi in summary:
        print(f"  {name:>8}  {ef:10.0f}  {ea:10.0f}  {rf:10.0f}  {ra:10.0f}  "
              f"{plo:5.0f} to {phi:3.0f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_dir / "88_sky_averaged_analytic_fit.npz",
        sigma=np.array(args.sigma),
        **{f"{name}_log10_e": v[0] for name, v in rows.items()},
        **{f"{name}_published": v[1] for name, v in rows.items()},
        **{f"{name}_unfitted": v[2] for name, v in rows.items()},
        **{f"{name}_fitted": v[3] for name, v in rows.items()},
        **{f"{name}_naive": v[4] for name, v in rows.items()},
        **{f"{name}_naive_fitted": v[5] for name, v in rows.items()},
        **{f"{name}_naive_chain": c for name, c in naive_chains.items()},
        **{f"{name}_chain": c for name, c in fitted_chains.items()},
    )
    figure(rows, args.out_dir)


if __name__ == "__main__":
    main()
