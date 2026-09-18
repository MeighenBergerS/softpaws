"""Example 44 -- the light reach evaluated where the muon is seen.

The reach law of Eq. (17) says the effective footprint radius grows
logarithmically with the muon's light output. Examples 30 and 32 evaluate it at
the muon's **production** energy, which :func:`light_reach_radius_km` flags in
its own Notes as an approximation: the arrival energy falls along the depth
integral, so a muon born far upstream is credited with a reach it no longer has.
This example removes that approximation.

Why the reach law has the form it does
--------------------------------------
It is derivable, not merely fitted. A muon's Cherenkov light per unit length
follows its energy loss, ``dE/dx = a + bE`` -- bare track constant, radiative
showers proportional to ``bE`` -- so the yield is ``Y ~ a + bE``. A module at
perpendicular distance ``d`` sees a flux falling as ``exp(-d/lambda)/d`` with
``lambda`` the effective photon attenuation length of the medium. Requiring a
fixed photoelectron count to register a track,

.. math:: Y \\, \\frac{e^{-d/\\lambda}}{d} = \\mathrm{const}
    \\quad\\Longrightarrow\\quad
    d_{\\max}(E) \\simeq d_0 + \\lambda \\ln\\!\\frac{a + bE}{a + bE_0}.

Two things follow that no fit was told to produce. The radius is logarithmic in
the muon energy, which is the form of Eq. (17); and its slope per e-fold **is**
the medium's attenuation length, so the fitted ``Lambda`` is a prediction to
check rather than a free number. Below the critical energy ``E_c = a/b`` the
ionization term takes over, the yield stops falling, and the radius saturates.

The consistent depth integral
-----------------------------
Writing the arrival energy as the integration variable rather than the column,

.. math:: V_{\\rm soft}(\\varepsilon, \\Omega) = \\int_{E_{\\rm thr}}^{\\varepsilon}
    A_{\\rm proj}\\big(\\Omega, R_{\\rm eff}(E)\\big) \\,
    \\left|\\frac{{\\rm d}\\ell}{{\\rm d}E}\\right| {\\rm d}E,

where ``ell(E) = L(eps -> E_thr) - L(E -> E_thr)`` is the column a muon has
travelled by the time it has fallen to ``E``. Each slice of the column is then
credited with the footprint the muon can actually light up *there*.

This matters because the two ends of the integral pull opposite ways. Near the
detector the muon is bright and the reach is a halo beyond the array; far
upstream it is close to threshold and lights less than the footprint. Evaluating
at the production energy keeps only the first.

What the example reports
------------------------
Three models against the published DR2 table: the static footprint, the reach at
production energy, and the reach at arrival energy. Then a scan over ``Lambda``,
fitting only the pivot at each point, so the preferred attenuation length can be
read off and compared with what is known about the ice.

Usage
-----
    python scripts/2026_muon_transport/44_depth_dependent_reach.py
    python scripts/2026_muon_transport/44_depth_dependent_reach.py --n-energy 60
"""

import argparse
import importlib.util
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

from softpaws.constants import CM_PER_KM
from softpaws.transport.muon_range import stochastic_muon_range_km
from softpaws.transport.soft_volume import light_reach_radius_km, prism_projected_area_km2
from softpaws.transport.source import MEAN_INELASTICITY, nucleon_number_density

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"
_DEFAULT_DATA_DIR = _HERE.parents[1] / "src" / "softpaws" / "data" / "dataverse_files"

#: Band the models are compared over, matching example 32's fit band.
FIT_BAND = (5.0, 7.8)

#: Effective photon attenuation length of the clearest deep South Pole ice [km],
#: ``lambda_eff = sqrt(lambda_abs lambda_scat / 3)`` with an absorption length of
#: 150 to 200 m near 400 nm and an effective scattering length of 50 to 70 m.
#: This is the scale the derivation in the module docstring predicts for
#: ``Lambda``, and it is an order-of-magnitude expectation: the argument fixes
#: the logarithmic *form* robustly and the coefficient only to the photon
#: transport scale, since it ignores the 1/d falloff and the fact that a whole
#: string of modules collects the light.
ICE_LAMBDA_RANGE_KM = (0.050, 0.070)

MODEL_COLOR = {
    "static footprint": "#1b9e77",
    "reach at production energy": "#7570b3",
    "reach at arrival energy": "#e7298a",
}
MODEL_STYLE = {
    "static footprint": ":",
    "reach at production energy": "--",
    "reach at arrival energy": "-",
}


def load_example_32():
    """Import example 32, whose instrument constants and ladder this reuses."""
    spec = importlib.util.spec_from_file_location(
        "_example_32", _HERE / "32_effective_area_comparison.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=pathlib.Path, default=_DEFAULT_DATA_DIR,
                        help="Directory holding the IceTracks-DR2 release.")
    parser.add_argument("--threshold", type=float, default=1.0e3,
                        help="Muon selection threshold [GeV].")
    parser.add_argument("--n-energy", type=int, default=40,
                        help="Points in the arrival-energy quadrature.")
    parser.add_argument("--out", type=pathlib.Path,
                        default=_DEFAULT_OUT_DIR / "44_depth_dependent_reach",
                        help="Output stem; .pdf and .png are both written.")
    return parser.parse_args()


def soft_volume_km3(
    ex32,
    production_gev: float,
    threshold_gev: float,
    cos_theta: np.ndarray,
    reach_km: float | None,
    pivot_gev: float,
    n_energy: int,
    at_arrival: bool,
) -> np.ndarray:
    """Column target volume for one production energy [km^3].

    Parameters
    ----------
    ex32 : ModuleType
        Example 32, for the instrument constants.
    production_gev : float
        Muon energy at production ``eps`` [GeV].
    threshold_gev : float
        Muon selection threshold [GeV].
    cos_theta : np.ndarray
        Arrival directions.
    reach_km : float or None
        Growth of the effective radius per e-fold [km]. ``None`` holds the
        radius at the instrumented footprint.
    pivot_gev : float
        Energy at which the effective radius equals the instrumented one [GeV].
    n_energy : int
        Points in the arrival-energy quadrature.
    at_arrival : bool
        Evaluate the reach at the muon's energy along the column (the consistent
        treatment) rather than at its production energy.

    Returns
    -------
    volume : np.ndarray
        Column volume [km^3], one entry per direction.

    Notes
    -----
    The column travelled by the time the muon has fallen to ``E`` is
    ``L(eps -> E_thr) - L(E -> E_thr)``, which is exact for the *mean*
    first-passage depth by the tower property, and lets the validated range
    function be called in its normal convention.
    """
    if production_gev <= threshold_gev:
        return np.zeros_like(np.asarray(cos_theta, dtype=float))

    total_km = float(np.atleast_1d(stochastic_muon_range_km(production_gev, threshold_gev))[0])
    if not np.isfinite(total_km) or total_km <= 0.0:
        return np.zeros_like(np.asarray(cos_theta, dtype=float))

    if reach_km is None:
        radius = np.full(1, ex32.IC_RADIUS_KM)
        area = prism_projected_area_km2(cos_theta, radius[0], ex32.IC_HEIGHT_KM,
                                        ex32.IC_N_SIDES)
        return area * total_km

    if not at_arrival:
        radius = float(light_reach_radius_km(
            ex32.IC_RADIUS_KM, production_gev, reach_km, pivot_gev)[0])
        area = prism_projected_area_km2(cos_theta, radius, ex32.IC_HEIGHT_KM,
                                        ex32.IC_N_SIDES)
        return area * total_km

    # Descending in energy, so the column ascends from 0 to the full range.
    energy = np.logspace(np.log10(production_gev), np.log10(threshold_gev), n_energy)
    column = total_km - stochastic_muon_range_km(energy, threshold_gev)
    radius = light_reach_radius_km(ex32.IC_RADIUS_KM, energy, reach_km, pivot_gev)
    area = prism_projected_area_km2(
        np.asarray(cos_theta, dtype=float)[None, :], radius[:, None],
        ex32.IC_HEIGHT_KM, ex32.IC_N_SIDES,
    )
    return np.trapezoid(area, column, axis=0)


def effective_area_cm2(
    ex32, threshold_gev: float, reach_km: float | None, pivot_gev: float,
    n_energy: int, at_arrival: bool,
) -> np.ndarray:
    """Upgoing-averaged nu_mu effective area with the regeneration ladder [cm^2].

    Parameters
    ----------
    ex32 : ModuleType
        Example 32.
    threshold_gev : float
        Muon selection threshold [GeV].
    reach_km : float or None
        Reach per e-fold [km]; ``None`` for the static footprint.
    pivot_gev : float
        Pivot energy [GeV].
    n_energy : int
        Points in the arrival-energy quadrature.
    at_arrival : bool
        Evaluate the reach along the column rather than at production.

    Returns
    -------
    aeff : np.ndarray
        Effective area [cm^2] on ``ex32.IC_LOG10_E``.
    """
    columns, weights, cos_theta = ex32.ic_upgoing_columns()
    n_nucleon = nucleon_number_density()
    v_det = np.pi * ex32.IC_RADIUS_KM**2 * ex32.IC_HEIGHT_KM
    out = np.empty(ex32.IC_LOG10_E.size)

    for i, e_nu in enumerate(10.0**ex32.IC_LOG10_E):
        rung_energy, rung_weight = ex32.regenerated_transmission(
            float(e_nu), columns, ex32.CROSS_SECTION
        )
        rate = np.zeros((rung_energy.size, cos_theta.size))
        for k, e_k in enumerate(rung_energy):
            volume = soft_volume_km3(
                ex32, (1.0 - MEAN_INELASTICITY) * float(e_k), threshold_gev,
                cos_theta, reach_km, pivot_gev, n_energy, at_arrival,
            )
            rate[k] = (volume + v_det) * CM_PER_KM**3
        rate *= n_nucleon * ex32.CROSS_SECTION.cc(rung_energy)[:, None]
        out[i] = np.average((rung_weight * rate).sum(axis=0), weights=weights)
    return out


def fit_pivot(ex32, published, band, threshold_gev, reach_km, n_energy, at_arrival):
    """Best-fit pivot energy and the residual scatter it leaves.

    Returns
    -------
    pivot_gev : float
        Pivot minimizing the root-mean-square log residual [GeV].
    rms : float
        Root-mean-square of ``log10(published / model)`` over the band [dex].
    """
    def cost(log_pivot: float) -> float:
        model = effective_area_cm2(ex32, threshold_gev, reach_km, 10**log_pivot,
                                   n_energy, at_arrival)
        return float(np.sqrt(np.mean(np.log10(published[band] / model[band]) ** 2)))

    opt = minimize_scalar(cost, bounds=(3.5, 7.5), method="bounded",
                          options={"xatol": 0.02})
    return 10**opt.x, float(opt.fun)


def main() -> None:
    args = parse_args()
    print("Loading example 32 ...")
    ex32 = load_example_32()
    published = ex32.icecube_published(args.data_dir)
    band = (ex32.IC_LOG10_E >= FIT_BAND[0]) & (ex32.IC_LOG10_E <= FIT_BAND[1])

    print("Scanning the reach per e-fold, fitting only the pivot at each point ...")
    lambdas = np.array([0.020, 0.030, 0.040, 0.050, 0.060, 0.070, 0.085,
                        0.100, 0.120, 0.150, 0.200])
    scan = []
    for lam in lambdas:
        pivot, rms = fit_pivot(ex32, published, band, args.threshold, float(lam),
                               args.n_energy, True)
        scan.append((lam, pivot, rms))
        print(f"    Lambda = {lam * 1e3:5.0f} m/e-fold -> pivot {pivot:10,.0f} GeV, "
              f"{rms:.4f} dex")
    scan = np.array(scan)
    best = scan[np.argmin(scan[:, 2])]
    print(f"\n  preferred Lambda = {best[0] * 1e3:.0f} m per e-fold, "
          f"pivot {best[1]:,.0f} GeV, {best[2]:.4f} dex")
    print(f"  deep-ice expectation: {ICE_LAMBDA_RANGE_KM[0] * 1e3:.0f} to "
          f"{ICE_LAMBDA_RANGE_KM[1] * 1e3:.0f} m")

    print("\nBuilding the three models ...")
    curves = {}
    curves["static footprint"] = effective_area_cm2(
        ex32, args.threshold, None, 1.0e6, args.n_energy, False)
    pivot_prod, rms_prod = fit_pivot(ex32, published, band, args.threshold,
                                     float(best[0]), args.n_energy, False)
    curves["reach at production energy"] = effective_area_cm2(
        ex32, args.threshold, float(best[0]), pivot_prod, args.n_energy, False)
    curves["reach at arrival energy"] = effective_area_cm2(
        ex32, args.threshold, float(best[0]), float(best[1]), args.n_energy, True)

    print("\n  ratio of the published table to each model, over the fit band")
    for name, curve in curves.items():
        res = np.log10(published[band] / curve[band])
        print(f"    {name:28s}: mean {10**np.mean(res):.3f}, "
              f"{np.sqrt(np.mean(res**2)):.4f} dex rms about unity, "
              f"{np.std(res):.4f} dex about its own mean")
    print(f"    (production-energy pivot {pivot_prod:,.0f} GeV, {rms_prod:.4f} dex)")

    make_figure(ex32, published, curves, scan, args.out)


def make_figure(ex32, published, curves, scan, out_path: pathlib.Path) -> None:
    """Left: published over model. Right: the scan over the reach per e-fold."""
    with plt.style.context(str(_STYLE)):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 2.8))

        ax.axhline(1.0, color="0.6", lw=0.8, ls=":", zorder=1)
        for name, curve in curves.items():
            ax.plot(ex32.IC_LOG10_E, published / curve, color=MODEL_COLOR[name],
                    ls=MODEL_STYLE[name], lw=1.3, label=name, zorder=3)
        ax.set_xlim(*FIT_BAND)
        ax.set_ylim(0.0, 1.4)
        ax.set_xlabel(r"$\log_{10}(E_\nu\,/\,\mathrm{GeV})$")
        ax.set_ylabel(r"published $/$ model")
        ax.legend(loc="lower right", fontsize=5.5, frameon=False, handlelength=2.2)

        bx.axvspan(ICE_LAMBDA_RANGE_KM[0] * 1e3, ICE_LAMBDA_RANGE_KM[1] * 1e3,
                   color="#1b9e77", alpha=0.12, lw=0, zorder=0)
        bx.text(0.5 * sum(ICE_LAMBDA_RANGE_KM) * 1e3, 0.95, "deep ice",
                transform=bx.get_xaxis_transform(), ha="center", va="top",
                fontsize=6, color="#1b9e77")
        bx.plot(scan[:, 0] * 1e3, scan[:, 2], color="#e7298a", lw=1.3, zorder=3)
        bx.set_xlabel(r"$\Lambda$ [m per e-fold]")
        bx.set_ylabel("residual [dex]")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png"):
            path = out_path.with_suffix(suffix)
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {path.resolve()}")
        plt.close(fig)


if __name__ == "__main__":
    main()
