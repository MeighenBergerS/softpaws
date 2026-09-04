"""Example 74 -- example 35 remade with the loss-model error bands.

Example 35's four figures carry single model lines; example 69 measured
the loss model's error and example 70 established how to propagate it
(patch ``drift_coefficient``, ``diffusion_coefficient`` *and*
``log_loss_moments`` -- the last one is where the first-passage range
reads the kernel). This example reruns example 35's pipeline under the
two envelope-defining variants of the ensemble -- Bezrukov-Bugaev above,
ALLM91 below; every other variant lies between them (example 69) -- and
redraws the four figures with the model lines wearing their bands.

The four panels, mirroring 35a-35d: (a) published against modelled
effective area in four upgoing declination bands; (b) the residual
band-by-band across the sky at three energies; (c) the modelled
point-source sensitivity against IceCube's published curve, matched to
the published analysis' setup; (d) the sensitivity of the five sites
against source declination. Bands widen with energy exactly as example
70's ratio figure says they must -- a few percent at ``10^5`` GeV,
+-6-8% by ``10^7``-``10^8`` -- so on the log axes of (a), (c) and (d)
they are visible but thin: the declination story of example 35 is not
at risk from the loss model, and the figures now say so quantitatively.

Panels (c) and (d) are no longer background-free ceilings. A point
source is one direction, so the sky a search looks through is one bin of
it, and the atmospheric neutrinos inside that bin are countable: MCEq
gives the flux, the same response gives the effective area, and the
released point spread gives the bin. Two things about that bin matter
and neither is a choice. It is the 68% containment, which for a Gaussian
point spread is the radius maximizing signal over the root of background
to within 5%. And it varies, closing as the energy rises and opening
again wherever the Earth has taken the high-energy end away, which is
what carries the declination dependence of the answer.

The window follows from the same argument. An atmospheric spectrum falls
faster than an astrophysical one, so where a search starts decides how
much background it has to accept, and a fixed lower edge charges a
search most exactly where its effective area is largest. Letting the
edge move removes that: it sits high where the response reaches high
energy and drops where the Earth has capped it. Panel (c) lands on the
published curve across the upgoing sky as a result, with nothing in it
fitted to that curve.

Usage
-----
    python examples/74_point_source_with_bands.py
    python examples/74_point_source_with_bands.py --gamma 2.5
"""

import argparse
import importlib.util
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.data import load_psf_table
from softpaws.fluxes import AtmosphericFlux, load_mceq_table
from softpaws.response.sensitivity import (
    OPTIMAL_CONTAINMENT,
    atmospheric_background_density,
    optimized_window_sensitivity,
    psf_bin_radius_deg,
)
from softpaws.transport.coefficients import set_kernel_scaling
from softpaws.transport.loss_ensemble import variant_scaling

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_DATA_DIR = _HERE.parents[1] / "src" / "softpaws" / "data" / "dataverse_files"
_DEFAULT_OUT_DIR = _HERE / "output"
_CACHE_DIR = _HERE / "cache"


def load_example(stem: str, name: str):
    """Import a sibling example module by file stem."""
    spec = importlib.util.spec_from_file_location(name, _HERE / stem)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: Example 70 installs the coefficient patches at import; example 35 is loaded
#: afterwards so its from-imports bind the wrapped functions.
_EX70 = load_example("70_aeff_error_bands.py", "_example_70")
_EX69 = _EX70._EX69
_EX35 = load_example("35_point_source_effective_area.py", "_example_35")

#: The envelope pair; every other ensemble member lies between them.
ENVELOPE = ("photo BB", "photo ALLM91")

#: The fitted transport scale (example 72's informed four-site corner,
#: b = 0.989 +- 0.028 with rock below the array), applied to both log-loss
#: moments exactly as the fits apply it. The fitted lambda (0.465 vs BGR18's
#: 0.4538) has no hook in example 35's machinery; its <= 4% UHE tilt sits
#: inside the drawn bands.
B_SCALE_FITTED = 0.989

COLORS = {"pub": "0.3", "model": "#e7298a"}

#: The atmospheric background, tabulated once by example 22.
MCEQ_TABLE = _CACHE_DIR / "22_mceq_atmospheric_flux.npz"

#: IceCube's released point spread, taken off the DR2 smearing table once. It
#: is the only measured one, so the other four sites read it as a stand-in.
PSF_TABLE = _CACHE_DIR / "74_psf_containment.npz"


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=pathlib.Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--threshold-gev", type=float,
                        default=_EX35.DEFAULT_MUON_THRESHOLD_GEV)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--bin-radius-deg", type=float, default=1.0,
                        help="Bin radius used where there is no released point spread.")
    parser.add_argument("--livetime-yr", type=float, default=10.0)
    parser.add_argument("--reach-km", type=float, default=_EX35.REACH_KM)
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the figures, '74a'-'74d'.")
    return parser.parse_args()


def activate(ratios, label) -> None:
    """One ensemble variant on top of the fitted transport scale."""
    set_kernel_scaling(variant_scaling(ratios, label, _EX69.E_GRID, prefactor=B_SCALE_FITTED))


def limit(bands, cos_theta, weights, radius, flux, livetime_s, gamma):
    """Flux a search excludes, with the background counted and the window free.

    The background rides the same directions the signal does, and it is
    multiplied by the response direction by direction before the daily sweep
    is averaged, because an atmospheric flux changes by an order of magnitude
    between the horizon and the vertical while a source flux does not.

    Parameters
    ----------
    bands : np.ndarray, shape (n_e, n_dir)
        Effective area per arrival direction [cm^2].
    cos_theta : np.ndarray, shape (n_dir,)
        Cosine of the arrival zenith of each direction.
    weights : np.ndarray, shape (n_dec, n_dir)
        Fraction of a sidereal day each declination spends in each direction.
    radius : np.ndarray or float
        Bin radius against energy and direction [deg].
    flux : softpaws.fluxes.AtmosphericFlux
        The MCEq background.
    livetime_s : float
        Exposure [s].
    gamma : float
        Spectral index of the injected flux.

    Returns
    -------
    e2_flux : np.ndarray, shape (n_dec,)
        ``E^2 dN/dE`` at the pivot [GeV cm^-2 s^-1].
    emin_gev : np.ndarray, shape (n_dec,)
        Energy the best window starts at [GeV].
    """
    density = atmospheric_background_density(
        flux, bands, cos_theta, weights, livetime_s, _EX35.COMMON_LOG10_E, radius)
    return optimized_window_sensitivity(
        bands @ weights.T, density, livetime_s, gamma, _EX35.COMMON_LOG10_E)


def build_variant(args, sites, sin_dec_edges, dec_grid, flux, psf) -> dict:
    """Example 35's model objects under the active loss variant."""
    icecube = sites[0]
    # The static, unfitted footprint, deliberately: the banded panels show the
    # declination shape with nothing calibrated, and the level they leave is
    # the flat factor the hemisphere calibration of example 45 absorbs.
    static = _EX35.icecube_model_banded(icecube, sin_dec_edges, args.threshold_gev)
    livetime_s = args.livetime_yr * 365.25 * 24.0 * 3600.0
    published_s = _EX35.PUBLISHED_LIVETIME_YR * 365.25 * 24.0 * 3600.0
    sensitivity, window, matched, matched_window = {}, {}, None, None
    for site in sites:
        cos_theta, weights = _EX35.zenith_band_weights(site.latitude_deg, dec_grid)
        bands = _EX35.directional_aeff_cm2(site, cos_theta, args.threshold_gev)
        radius = psf_bin_radius_deg(psf, cos_theta, _EX35.COMMON_LOG10_E,
                                    args.bin_radius_deg)
        sensitivity[site.name], window[site.name] = limit(
            bands, cos_theta, weights, radius, flux, livetime_s, args.gamma)
        if site is icecube:
            matched, matched_window = limit(
                bands, cos_theta, weights, radius, flux, published_s,
                _EX35.PUBLISHED_GAMMA)
    return {"static": static, "sensitivity": sensitivity, "window": window,
            "matched": matched, "matched_window": matched_window}


def _save(fig, out_dir: pathlib.Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {path.resolve()}")
    plt.close(fig)


def _band(variants, pick):
    """Elementwise min and max of ``pick(variant)`` across the ensemble runs."""
    stack = np.array([pick(v) for v in variants.values()])
    return stack.min(axis=0), stack.max(axis=0)


#: Declinations the printed tables are quoted at [deg].
TABLE_DEC_DEG = (-60.0, -30.0, 0.0, 30.0, 60.0)


def report_window(sites, dec_grid, variants) -> None:
    """Print where the search starts, which is what carries the declination shape."""
    baseline = variants["baseline"]
    print("\n  Where the best window starts [log10(E/GeV)]:\n")
    print(f"{'detector':>12}" + "".join(f"{f'{d:+.0f}d':>10}" for d in TABLE_DEC_DEG))
    for site in sites:
        row = np.log10(baseline["window"][site.name])
        print(f"{site.name:>12}"
              + "".join(f"{np.interp(d, dec_grid, row):10.2f}" for d in TABLE_DEC_DEG))
    print("\n  It sits high where the response reaches high energy and the background")
    print("  can be cut away, and drops where the Earth has capped the response and")
    print("  there is nothing left to cut to.")


def report_matched(dec_grid, variants, published_curve) -> None:
    """Print the matched IceCube curve against the published one."""
    pub_sin_dec, pub_flux = published_curve
    at_table = np.sin(np.deg2rad(np.asarray(TABLE_DEC_DEG, dtype=float)))
    published = np.interp(at_table, pub_sin_dec, pub_flux)
    model = np.array([np.interp(d, dec_grid, variants["baseline"]["matched"])
                      for d in TABLE_DEC_DEG])
    print("\n  Matched to the published analysis [GeV cm^-2 s^-1]:\n")
    print(f"{'':>12}" + "".join(f"{f'{d:+.0f}d':>12}" for d in TABLE_DEC_DEG))
    for label, values in (("published", published), ("model", model)):
        print(f"{label:>12}" + "".join(f"{v:12.3g}" for v in values))
    print(f"{'ratio':>12}" + "".join(f"{v:11.2f}x" for v in model / published))
    print("\n  Nothing here is fitted to that curve. The upgoing sky is the like-for-")
    print("  like half; the downgoing one is not, because atmospheric muons are the")
    print("  background there and this model has none.")


def figure_bands(sin_dec_centers, published, variants, out_dir) -> None:
    """74a: four upgoing declination bands, the model line wearing its band."""
    show = [int(np.argmin(np.abs(sin_dec_centers - s))) for s in (0.1, 0.4, 0.7, 0.95)]
    log10_e = _EX35.COMMON_LOG10_E
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        for j, color in zip(show, ("C0", "C1", "C2", "C3")):
            dec = np.rad2deg(np.arcsin(sin_dec_centers[j]))
            ax.plot(log10_e, published[:, j], color=color, lw=1.4)
            lo, hi = _band(variants, lambda v, j=j: v["static"][:, j])
            ax.fill_between(log10_e, lo, hi, color=color, alpha=0.35, lw=0)
            ax.plot(log10_e, variants["baseline"]["static"][:, j], color=color,
                    lw=0.9, ls="--")
            ax.text(7.5, 0.6 * np.interp(7.5, log10_e, published[:, j]),
                    rf"$\delta = {dec:.0f}^\circ$", color=color, fontsize=8,
                    ha="center", va="center")
        ax.plot([], [], color="0.3", lw=1.4, label="IceCube")
        ax.plot([], [], color="0.3", lw=0.9, ls="--", label="Model")
        ax.fill_between([], [], [], color="0.3", alpha=0.35,
                        label="Loss-model band")
        ax.set_yscale("log")
        ax.set_xlim(log10_e[0], log10_e[-1])
        ax.set_xlabel(r"$\log_{10}(E_\nu\,/\,\mathrm{GeV})$")
        ax.set_ylabel(r"$A_{\rm eff}$ [cm$^2$]")
        ax.set_box_aspect(1)
        ax.legend(loc="lower right")
        _save(fig, out_dir, "74a_effective_area_bands")


def figure_residual(sin_dec_centers, published, variants, out_dir) -> None:
    """74b: the residual across the sky, with the band on the model side."""
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        for log10_e, color in zip((5.0, 6.0, 7.0), ("C0", "C1", "C2")):
            i = int(np.argmin(np.abs(_EX35.COMMON_LOG10_E - log10_e)))
            lo, hi = _band(variants, lambda v, i=i: published[i] / v["static"][i])
            ax.fill_between(sin_dec_centers, lo, hi, color=color, alpha=0.30, lw=0)
            ax.plot(sin_dec_centers, published[i] / variants["baseline"]["static"][i],
                    color=color, lw=1.1, label=rf"$10^{{{log10_e:.0f}}}$ GeV")
        ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
        ax.axvspan(-1.0, 0.0, color="0.88", alpha=0.7, lw=0)
        ax.set_yscale("log")
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(1.0e-3, 3.0)
        ax.set_xlabel(r"$\sin\delta$", fontsize=8)
        ax.set_ylabel("IceCube / model", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, loc="lower right")
        _save(fig, out_dir, "74b_residual_by_declination")


def figure_sensitivity(dec_grid, variants, published_curve, out_dir) -> None:
    """74c: the matched sensitivity ceiling, labels on the curves as in 35c."""
    pub_sin_dec, pub_flux = published_curve
    sin_dec = np.sin(np.deg2rad(dec_grid))
    matched = variants["baseline"]["matched"]
    label_x = 0.3
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        ax.plot(pub_sin_dec, pub_flux, color="k", lw=1.4)
        lo, hi = _band(variants, lambda v: v["matched"])
        ax.fill_between(sin_dec, lo, hi, color="C0", alpha=0.35, lw=0)
        ax.plot(sin_dec, matched, color="C0", lw=1.1, ls="--")
        ax.fill_between(sin_dec, matched,
                        np.interp(sin_dec, pub_sin_dec, pub_flux),
                        color="C0", alpha=0.12, lw=0)
        ax.axvspan(-1.0, 0.0, color="0.88", alpha=0.7, lw=0)
        ax.set_yscale("log")
        ax.set_xlim(-1.0, 1.0)
        ax.set_xlabel(r"$\sin\delta$")
        ax.set_ylabel(r"$E^2\,\mathrm{d}N/\mathrm{d}E$ [GeV cm$^{-2}$ s$^{-1}$]")
        ax.set_box_aspect(1)
        # The labels replace a legend, so they sit on their curves: draw once
        # to freeze the transform, then take the angle off the screen.
        fig.canvas.draw()
        for x, y, name, color in ((pub_sin_dec, pub_flux, "IceCube", "k"),
                                  (sin_dec, matched, "Model", "C0")):
            ax.text(label_x, 1.18 * float(np.interp(label_x, x, y)), name,
                    color=color, ha="center", va="bottom",
                    rotation=_EX35._curve_angle_deg(ax, x, y, label_x),
                    rotation_mode="anchor")
        _save(fig, out_dir, "74c_published_sensitivity")


def figure_site_ceiling(sites, dec_grid, variants, out_dir) -> None:
    """74d: the five-site ultra-high-energy ceiling, banded."""
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        floor, top = np.inf, 0.0
        for site in sites:
            color = _EX35.SITE_COLORS[site.name]
            lo, hi = _band(variants, lambda v, n=site.name: v["sensitivity"][n])
            ax.fill_between(dec_grid, lo, hi, color=color, alpha=0.30, lw=0)
            central = variants["baseline"]["sensitivity"][site.name]
            ax.plot(dec_grid, central, color=color, lw=1.2,
                    ls=_EX35.SITE_LINESTYLES[site.name], label=site.name)
            floor, top = min(floor, central.min()), max(top, central.max())
        ax.set_yscale("log")
        ax.set_xlim(-90.0, 90.0)
        low, high = 0.7 * floor, 25.0 * top
        ax.set_ylim(low, high)
        # NGC 1068 and TXS 0506+056 are six degrees apart, so NGC 1068's label
        # sits to the left of its line, as in example 35's original.
        label_side = {"NGC 1068": -1.0}
        for name, dec in _EX35.REFERENCE_SOURCES:
            ax.axvline(dec, color="0.7", lw=0.6, ls=":")
            ax.text(dec + 2.8 * label_side.get(name, 1.0),
                    low * (high / low) ** 0.98, name, rotation=90,
                    ha="center", va="top", color="0.45")
        ax.set_xticks([-90, -45, 0, 45, 90])
        ax.set_xlabel(r"Source declination $\delta$ [deg]")
        ax.set_ylabel(r"$E^2\,\mathrm{d}N/\mathrm{d}E$ [GeV cm$^{-2}$ s$^{-1}$]")
        ax.set_box_aspect(1)
        ax.legend(loc="upper right")
        _save(fig, out_dir, "74d_site_ceiling")


def load_background(args):
    """The MCEq flux and IceCube's point spread, both cached beside the script."""
    flux = AtmosphericFlux(load_mceq_table(MCEQ_TABLE))
    psf = load_psf_table(PSF_TABLE, args.data_dir, quantile=OPTIMAL_CONTAINMENT)
    containment = psf["containment_deg"]
    print(f"Point spread at {100 * float(psf['quantile']):.0f}% containment: "
          f"{containment.shape[0]} energies x {containment.shape[1]} declinations")
    return flux, psf


def main() -> None:
    args = parse_args()
    _, ratios = _EX69.load_ensemble(False)
    sites = _EX35.build_sites()
    print(f"Loading IceCube IRFs from: {args.data_dir}")
    sin_dec_edges, published = _EX35.icecube_banded(args.data_dir)
    sin_dec_centers = 0.5 * (sin_dec_edges[:-1] + sin_dec_edges[1:])
    dec_grid = np.linspace(-90.0, 90.0, _EX35.N_DEC_GRID)
    flux, psf = load_background(args)

    print(f"Fitted transport scale applied throughout: b_scale = {B_SCALE_FITTED}")
    variants = {}
    for label in ["baseline", *ENVELOPE]:
        print(f"Building example 35's pipeline under {label} ...")
        activate(ratios, None if label == "baseline" else label)
        variants[label] = build_variant(args, sites, sin_dec_edges, dec_grid, flux, psf)
    _EX70.set_variant(ratios, None)

    for j_target in (0.4, 0.95):
        j = int(np.argmin(np.abs(sin_dec_centers - j_target)))
        for e_target in (5.0, 7.0):
            i = int(np.argmin(np.abs(_EX35.COMMON_LOG10_E - e_target)))
            lo, hi = _band(variants, lambda v, i=i, j=j: v["static"][i, j])
            base = variants["baseline"]["static"][i, j]
            if base > 0.0:
                print(f"  band at sin dec {sin_dec_centers[j]:.2f}, 1e{e_target:.0f} GeV: "
                      f"{lo / base:.3f} - {hi / base:.3f}")

    report_window(sites, dec_grid, variants)
    published_curve = _EX35.load_published_sensitivity(_EX35._PUBLISHED_SENSITIVITY)
    report_matched(dec_grid, variants, published_curve)
    figure_bands(sin_dec_centers, published, variants, args.out_dir)
    figure_residual(sin_dec_centers, published, variants, args.out_dir)
    figure_sensitivity(dec_grid, variants, published_curve, args.out_dir)
    figure_site_ceiling(sites, dec_grid, variants, args.out_dir)


if __name__ == "__main__":
    main()
