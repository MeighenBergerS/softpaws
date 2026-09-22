"""Example 86 -- the closed-form response against the full one at four sites.

Example 83 draws the four published effective areas against the full
response, Eq. (aeff) of the paper. This example drops the published curves
and draws the full response in black against the closed form of Eq. (aeffcf)
in each site's colour, at the same two instrument numbers per site, so the
figure shows what the last step of Section III buys at every detector.

The closed form is Eq. (aeffstd) with the transport put in by hand: the
frozen closed-form range of Eq. (Lclosed) in pure water at ``(1 - <y_w>) E_nu``,
the survival factor ``exp(-X / Lambda_nu)`` along the chord, the reach in the
projected area, ``V_det`` for the starting tracks, and the ``nu_mu`` channel
alone. At a sea site the muon cannot be born above the surface, so the range
is capped at the column upstream of the detector, ``min(L, X)``, the hand
version of the truncated expectation the full model carries. Each site keeps
its own sky average: IceCube's upgoing hemisphere, the whole sky at ARCA230
and P-ONE, and the ``|cos theta_z| <= 0.5`` band of the 2025 map at TRIDENT.

P-ONE and TRIDENT are simulated with pure absorption and no tau channel, so
at those two sites the closed form and the full response differ only in the
range, which is the cleanest test of the range side alone. The example prints
the ratio of the two at three energies per site and inside each fit window.

Usage
-----
    python scripts/2026_muon_transport/86_closed_form_four_detectors.py
"""

import argparse
import importlib.util
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws._paper import reduced
from softpaws._paper import site_fit as sm
from softpaws.constants import CM_PER_KM, RHO_WATER_G_CM3
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
_EX73 = _EX83._EX73
_EX56 = _EX83._EX56

#: Energies the printout quotes [log10 GeV].
QUOTED_LOG10_E = (5.0, 6.0, 7.0)

#: Label placement per site; see example 73's ``LABEL_SPEC``.
LABEL_SPEC = {
    "IceCube": (4.6, 0.55, "center", "IceCube", "IceCube", None),
    "ARCA230": (4.5, 1.55, "center", "ARCA230", "ARCA230", None),
    "P-ONE": (7.3, 0.55, "center", "P-ONE", "P-ONE", None),
    "TRIDENT": (4.75, 1.75, "center", "TRIDENT", "TRIDENT", None),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=pathlib.Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--chains", type=pathlib.Path,
                        default=_DEFAULT_OUT_DIR / "77_chains_sigma05.npz")
    parser.add_argument("--trident-chain", type=pathlib.Path,
                        default=_DEFAULT_OUT_DIR / "82_trident2025_chain.npz")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR)
    return parser.parse_args()


def closed_form_water(
    theta: np.ndarray,
    site: sm.WaterSite,
    zenith_weights: np.ndarray,
    neutrino_column: np.ndarray,
    muon_column_km: np.ndarray,
) -> np.ndarray:
    """Eq. (aeffcf) averaged over a water site's sky, on :data:`sm.ARCA_LOG10_E`.

    The same assembly as :func:`softpaws._paper.site_fit.water_model`
    with the closed-form pieces in place of the full ones: pure absorption
    for the arrival, the frozen water range capped at the upstream column for
    the length, and no tau channel.
    """
    eps_0, log10_e_thr, _, lam, reach_km = theta
    threshold = 10.0**log10_e_thr
    n_nucleon = nucleon_number_density(RHO_WATER_G_CM3)
    theta_deg, _ = sm.arca_zenith_grid()
    energy = 10.0**sm.ARCA_LOG10_E
    arrival = survival_probability(
        energy[:, None], neutrino_column[None, :], lam, sm.CROSS_SECTION
    )
    muon = (1.0 - MEAN_INELASTICITY) * energy
    length = np.minimum(
        _EX85.closed_form_range_km(muon, threshold)[:, None], muon_column_km[None, :]
    )
    radius = light_reach_radius_km(site.radius_km, muon, reach_km, sm.SITE_FIT_REACH_PIVOT_GEV)
    area_km2 = sm.water_projected_area_km2(site, theta_deg[None, :], radius[:, None])
    v_det_km3 = site.n_blocks * np.pi * radius**2 * site.height_km
    volume_km3 = area_km2 * length + v_det_km3[:, None]
    sigma = sm.tilted_cc(energy, lam)
    rate = n_nucleon * sigma[:, None] * volume_km3 * CM_PER_KM**3
    return eps_0 * np.average(arrival * rate, axis=1, weights=zenith_weights)


def closed_form_icecube(theta: np.ndarray) -> np.ndarray:
    """Eq. (aeffcf) over IceCube's upgoing hemisphere, on :data:`sm.IC_LOG10_E`."""
    eps_0, log10_e_thr, _, _, reach_km = theta
    columns = np.array([prem_column(float(d)) for d in _EX85.DEC_DEG])
    per_dec = _EX85.response_per_declination(
        eps_0, 10.0**log10_e_thr, reach_km, None, columns, False, False, False
    )
    return _EX85.band_average(per_dec, 0.0, 90.0)


def closed_form_trident(theta: np.ndarray, log10_e: np.ndarray, cos_max: float) -> np.ndarray:
    """Eq. (aeffcf) averaged over the flat-selection band of the 2025 map."""
    cos_theta, _, _ = reduced.trident_2025_cells()
    model = reduced.MapModel(log10_e)
    _, neutrino_column, _ = sm.water_columns(model.site)
    d_cos = np.abs(np.diff(reduced.COS_EDGES))
    rows = np.abs(cos_theta) <= cos_max
    curves = []
    for weights, row in zip(model.weights, rows, strict=True):
        if not row:
            continue
        full = closed_form_water(
            theta, model.site, weights, neutrino_column, model.muon_column_km
        )
        curves.append(10.0 ** np.interp(log10_e, model.grid, np.log10(full)))
    return np.average(np.array(curves), axis=0, weights=d_cos[rows])


def closed_form(detector: sm.SiteFit, theta: np.ndarray) -> np.ndarray:
    """The closed form on the detector's own grid and sky average."""
    if detector.name == "IceCube":
        return closed_form_icecube(theta)
    if detector.name == "TRIDENT":
        return closed_form_trident(theta, detector.log10_e, _EX83.COS_MAX)
    site = (
        sm.ARCA230_WATER_SITE if detector.name == "ARCA230"
        else [s for s in sm.water_sites() if s.name == detector.name][0]
    )
    zenith_weights, neutrino_column, muon_column_km = sm.water_columns(site)
    return closed_form_water(theta, site, zenith_weights, neutrino_column, muon_column_km)


def figure(curves: dict, out_dir: pathlib.Path) -> None:
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        for name, (log10_e, full, closed) in curves.items():
            good = np.isfinite(full) & (full > 0.0) & (closed > 0.0)
            color = _EX56.SITE_COLORS[name]
            ax.plot(log10_e[good], full[good], color="k", lw=1.2)
            ax.plot(log10_e[good], closed[good], color=color, lw=1.1, ls="--")
        ax.plot([], [], color="k", lw=1.2, label="Full response, Eq.~(21)")
        ax.plot([], [], color="0.4", lw=1.1, ls="--", label="Closed form, Eq.~(19)")
        ax.set_yscale("log")
        ax.set_xlim(4.0, 8.0)
        ax.set_ylim(5.0e4, 2.0e8)
        ax.set_xlabel(r"$\log_{10}(E_\nu\,/\,\mathrm{GeV})$")
        ax.set_ylabel(r"$A_{\rm eff}$ [cm$^2$]")
        ax.set_box_aspect(1)
        ax.legend(frameon=False, loc="lower right")
        fig.canvas.draw()
        anchors = {name: (x, f) for name, (x, f, _) in curves.items()}
        for name in curves:
            x0, factor, ha, slope_of, anchor_on, y_at = LABEL_SPEC[name]
            angle = _EX73._curve_angle_deg(ax, *anchors[slope_of], x0) if slope_of else 0.0
            x, y = anchors[anchor_on]
            height = factor * float(np.interp(y_at if y_at is not None else x0, x, y))
            ax.text(x0, height, name, color=_EX56.SITE_COLORS[name], ha=ha, va="center",
                    rotation=angle, rotation_mode="anchor")
        _EX73._save(fig, out_dir, "86a_closed_form_four_detectors")


def main() -> None:
    args = parse_args()
    detectors = _EX83.reduced_detectors(args.data_dir, args.chains)
    trident, _ = _EX83.trident_2025(args.trident_chain)
    detectors.append(trident)

    curves = {}
    print("Closed form, Eq. (aeffcf), over the full response, Eq. (aeff), at the "
          "posterior medians of examples 77 and 82")
    print(f"  {'site':>8}  " + "  ".join(f"10^{x:.0f}" for x in QUOTED_LOG10_E)
          + "   window: geometric mean, scatter [dex]")
    for d in detectors:
        theta = np.median(d.chain, axis=0)
        full = np.asarray(d.predict(theta, None), dtype=float)
        closed = closed_form(d, theta)
        curves[d.name] = (d.log10_e, full, closed)
        ratio = closed / full
        cells = "  ".join(
            f"{np.interp(x, d.log10_e, ratio):5.3f}" if d.log10_e.min() <= x <= d.log10_e.max()
            else "   --" for x in QUOTED_LOG10_E
        )
        inside = np.log10(ratio[d.mask])
        print(f"  {d.name:>8}  {cells}   {10**np.mean(inside):.3f}, {np.std(inside):.3f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_dir / "86_closed_form_four_detectors.npz",
        **{f"{name}_log10_e": v[0] for name, v in curves.items()},
        **{f"{name}_full": v[1] for name, v in curves.items()},
        **{f"{name}_closed": v[2] for name, v in curves.items()},
    )
    figure(curves, args.out_dir)


if __name__ == "__main__":
    main()
