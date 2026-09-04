"""Example 47 -- band effective areas, point-source sensitivity and site ceiling.

Redoes the three figures example 35 contributes to the paper -- its (a), (c) and
(d) -- on the model example 45 derives and example 46 validates band by band.
Example 46 already replaces 35's (b).

What changes against example 35
-------------------------------
Example 35 carries in a two-parameter reach fitted to the sky-averaged DR2
table. Five things are different here, and all of them were established against
data rather than chosen:

``reach``
    Derived from the Frank-Tamm yield, the medium's photon attenuation length
    and the module's effective area (example 45). A hit is a local coincidence
    with Poisson per-receiver probability, the multiplicity is met at Poisson
    probability, and nothing in the optical chain is set against an
    effective-area curve.

``fitted configuration``
    The default run carries example 45's *fitted* model to the angular axis:
    the per-site flat attenuation override its shape fit lands on and the
    free normalization its fitted curve needs
    (:data:`FITTED_ATTENUATION_M`, :data:`FITTED_NORMALIZATION`), with the
    halo fully counted, exactly as in example 45. Both numbers were set
    against *sky-averaged* curves only, so every angular shape here is a
    prediction. ``--first-principles`` keeps the derived wavelength-resolved
    optics instead, and ``--halo-weight`` blends the reach-dilated halo, the
    configuration the sky-averaged refit scan prefers at 0.75.

``channels``
    ``nu_mu`` alone against a published table, since the DR2 readme states the
    area is averaged over *simulated muon neutrino events*. Predicting an
    observed track rate is a different operation and does need
    ``nu_tau -> tau -> mu``; ``--with-tau`` switches to it.

``species``
    Neutrino and antineutrino averaged, each propagated with its own cross
    section and pinned to the CSMS isoscalar values, since both published
    tables are that average.

What is new against example 35's figure set
-------------------------------------------
A fourth figure: the KM3NeT/ARCA230 zenith-resolved effective area, digitized
from Fig. 7(b) of arXiv:2402.08363 (the final track selection), against the
same model in the same ``cos(theta)`` bands. IceCube's declination bands and
ARCA's zenith bands together test whether one halo weight and one flat
efficiency per detector carry the *angular* dependence, which the sky-averaged
fits never constrained.

Where this model is weak, and it matters here
---------------------------------------------
Example 46 measures the neutral-current regeneration contribution band by band.
At the horizon it is a 0 to 3% correction; at the nadir above 10^7 GeV it is the
**entire** signal, reaching a factor 444 at ``sin(dec) = 0.98`` and 10^7 GeV.
Direct transmission there is dead and everything the detector sees arrived by
down-scattering, so the deepest bands rest on the regeneration ladder rather
than on the transport this paper is about.

That is a caveat specific to *point sources near the celestial pole*, which is
exactly what figures (c) and (d) plot. The sky-averaged benchmarks are safe,
since they are dominated by bands where regeneration is a percent-level
correction. The polar end of the declination curves is not, and the figures mark
it rather than hiding it.

Usage
-----
    python examples/47_point_source_derived_reach.py
    python examples/47_point_source_derived_reach.py --halo-weight 1.0
    python examples/47_point_source_derived_reach.py --emin-gev 1e4
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"
_DEFAULT_DATA_DIR = _HERE.parents[1] / "src" / "softpaws" / "data" / "dataverse_files"

#: Digitized KM3NeT/ARCA230 zenith-resolved effective area, Fig. 7(b) of
#: arXiv:2402.08363: the **final track selection** for ``nu_mu`` CC, in bands
#: of ``cos(theta)``. Column headers carry the band edges as
#: ``km3net_<hi>g<lo>``, largest ``cos(theta)`` first; X is ``E_nu`` [GeV] and
#: Y is ``A_eff`` [m^2].
_KM3NET_ANGLE_CSV = (_HERE.parents[1] / "src" / "softpaws" / "data" / "km3net"
                     / "km3net_angle_dependent.csv")

#: Example 45's fitted configuration, from its shipped ``--numu-only`` run:
#: the flat attenuation override its shape fit lands on [m] and the free
#: normalization its fitted curve carries, per detector. The default run uses
#: these, so figures here are the *fitted* model of example 45 taken to the
#: angular axis it was never fitted on. ``--first-principles`` drops the
#: override and keeps each site's derived wavelength-resolved optics.
#: Example 45's fitted pair per site, nu_mu only, with rock below the ice or
#: the sea floor (2026-09-02). The all-water kernel gave 42 / 47 m and
#: 0.755 / 0.720.
FITTED_ATTENUATION_M = {"IceCube": 33.7, "ARCA230": 41.6}
FITTED_NORMALIZATION = {"IceCube": 0.956, "ARCA230": 0.817}

#: Fraction of the reach-dilated halo the selection accepts. Example 45
#: carries no halo blend, so its fitted configuration corresponds to 1; the
#: sky-averaged refit scan prefers 0.75 with *derived* optics, reachable via
#: ``--halo-weight 0.75 --first-principles``.
DEFAULT_HALO_WEIGHT = 1.0

#: Declination bands drawn in the effective-area figure, in ``sin(dec)``.
SHOW_SIN_DEC = (0.1, 0.4, 0.7, 0.95)

#: Beyond this ``sin(dec)`` the regeneration ladder carries most of the signal
#: above 10^7 GeV (example 46), so the curve is drawn but flagged.
REGENERATION_CAUTION_SIN_DEC = 0.72

BAND_COLOR = ("#7570b3", "#1b9e77", "#e7298a", "#e6ab02")


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=pathlib.Path, default=_DEFAULT_DATA_DIR,
                        help="Root of the DR2 release.")
    parser.add_argument("--min-modules", type=float, default=None,
                        help="Modules that must register a coincident hit.")
    parser.add_argument("--halo-weight", type=float, default=DEFAULT_HALO_WEIGHT,
                        help="Fraction of the reach-dilated halo the selection "
                             "accepts; 1 (the default, example 45's convention) "
                             "counts every track the light admits.")
    parser.add_argument("--first-principles", action="store_true",
                        help="Drop the fitted attenuation overrides and keep the "
                             "derived wavelength-resolved optics.")
    parser.add_argument("--efficiency-icecube", type=float,
                        default=FITTED_NORMALIZATION["IceCube"],
                        help="Flat normalization applied at IceCube; defaults to "
                             "example 45's fitted value.")
    parser.add_argument("--efficiency-arca", type=float,
                        default=FITTED_NORMALIZATION["ARCA230"],
                        help="Flat normalization applied at ARCA230, and reused "
                             "for the proposed water sites; defaults to example "
                             "45's fitted value.")
    parser.add_argument("--with-tau", action="store_true",
                        help="Add nu_tau -> tau -> mu, for a rate and not a table.")
    parser.add_argument("--gamma", type=float, default=2.0,
                        help="Spectral index of the point source.")
    parser.add_argument("--emin-gev", type=float, default=1.0e5,
                        help="Bottom of the analysis window for the site ceiling [GeV].")
    parser.add_argument("--livetime-yr", type=float, default=10.0,
                        help="Exposure for the site ceiling [yr].")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the three figures, written as '47a'-'47c'.")
    return parser.parse_args()


def optics_for(ex45, site, fitted: bool = False) -> object:
    """Optical medium and module to use for one of example 35's sites.

    IceCube and ARCA have published optics and modules, and example 45 carries
    both. The three proposed sites do not, so the sea-water pair is reused for
    the two ocean sites and Baikal is given the lake's much shorter absorption
    length. Those three enter only the site-ceiling forecast, never a comparison
    against data, and the assumption is stated rather than buried.

    Parameters
    ----------
    ex45 : ModuleType
        Example 45, for its site definitions.
    site : ex35.Site
        The detector whose optics are wanted.
    fitted : bool, optional
        Apply example 45's fitted flat attenuation override
        (:data:`FITTED_ATTENUATION_M`) where one exists. The proposed sites
        have no fit and always keep their assumed optics.

    Returns
    -------
    optics : ex45.Site
        Medium and module to build the reach from.
    """
    from dataclasses import replace

    if site.name == "IceCube":
        base = ex45.ICECUBE_SITE
    elif site.name == "Baikal-GVD":
        # Lake Baikal is far less transparent than the Mediterranean: absorption
        # ~22 m against ~68 m, with comparable scattering.
        base = replace(ex45.ARCA_SITE, name="Baikal-GVD", absorption_m=22.0,
                       scattering_m=30.0, density_g_cm3=1.0)
    else:
        base = replace(ex45.ARCA_SITE, name=site.name)
    if fitted and site.name in FITTED_ATTENUATION_M:
        base = replace(base, attenuation_override_m=FITTED_ATTENUATION_M[site.name])
    return base


def banded_model(ex35, ex45, ex46, site, sin_dec_edges, min_modules, flavours,
                 halo_weight, efficiency, fitted):
    """Model effective area per published declination band [cm^2]."""
    return ex46.model_banded(ex35, ex45, site, optics_for(ex45, site, fitted),
                             sin_dec_edges, min_modules, flavours, None,
                             halo_weight, efficiency)


def directional_model(ex35, ex45, ex46, site, cos_theta, min_modules, flavours,
                      halo_weight, efficiency, fitted):
    """Model effective area per arrival direction, species-averaged [cm^2]."""
    optics = optics_for(ex45, site, fitted)
    return np.mean(
        [ex46.directional_aeff_cm2(ex35, ex45, site, optics, cos_theta,
                                   min_modules, flavours, xsec, None,
                                   halo_weight, efficiency)
         for xsec in ex45.SPECIES],
        axis=0,
    )


def km3net_banded(path: pathlib.Path = _KM3NET_ANGLE_CSV):
    """Digitized ARCA230 effective area per ``cos(theta)`` band.

    Returns
    -------
    bands : list of tuple
        One entry ``(cos_hi, cos_lo, log10_e, aeff_cm2)`` per band, points
        sorted in energy. ``cos_hi`` is the largest ``cos(theta)`` of the
        band; ``-1`` is the nadir, so the last band is the deepest column.
    """
    with open(path) as handle:
        names = [n for n in handle.readline().strip().split(",") if n]
    raw = np.genfromtxt(path, delimiter=",", skip_header=2)
    bands = []
    for k, name in enumerate(names):
        hi, lo = name.removeprefix("km3net_").split("g")
        x, y = raw[:, 2 * k], raw[:, 2 * k + 1]
        good = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
        order = np.argsort(x[good])
        bands.append((float(hi), float(lo),
                      np.log10(x[good][order]), 1.0e4 * y[good][order]))
    return bands


def arca_banded_model(ex35, ex45, ex46, site, bands, min_modules, flavours,
                      halo_weight, efficiency, fitted, n_sub: int = 8):
    """Model effective area in the digitized ``cos(theta)`` bands [cm^2].

    Each band is averaged uniformly in ``cos(theta)``, which is the
    solid-angle weighting, over ``n_sub`` sub-directions.

    Returns
    -------
    model : np.ndarray, shape (n_band, n_energy)
        Effective area [cm^2] on ``ex35.COMMON_LOG10_E``.
    """
    curves = []
    for cos_hi, cos_lo, _, _ in bands:
        edges = np.linspace(cos_lo, cos_hi, n_sub + 1)
        cos_theta = 0.5 * (edges[:-1] + edges[1:])
        curves.append(directional_model(
            ex35, ex45, ex46, site, cos_theta, min_modules, flavours,
            halo_weight, efficiency, fitted).mean(axis=1))
    return np.array(curves)


def report_arca_bands(ex35, bands, model) -> None:
    """Print the per-band residuals against the digitized Fig. 7(b) points."""
    print("\n  ARCA230 cos(theta) bands, published / model "
          "(Fig. 7b, final track selection)")
    print(f"  {'cos(theta) band':>17} {'mean':>8} {'rms':>8}   points at E >= 10^4 GeV")
    for (cos_hi, cos_lo, log10_e, aeff), curve in zip(bands, model):
        mask = log10_e >= 4.0
        table = np.log10(np.clip(curve, 1.0e-30, None))
        matched = 10.0 ** np.interp(log10_e[mask], ex35.COMMON_LOG10_E, table)
        res = np.log10(aeff[mask] / matched)
        print(f"  [{cos_hi:5.2f}, {cos_lo:5.2f}] {10.0**res.mean():8.3f} "
              f"{res.std():8.4f}   {mask.sum():3d}")


def figure_arca_bands(ex35, bands, model, out_dir) -> None:
    """Digitized and model ARCA230 effective areas per ``cos(theta)`` band."""
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        for (cos_hi, cos_lo, log10_e, aeff), curve, color in zip(
                bands, model, BAND_COLOR):
            ax.plot(log10_e, aeff, color=color, lw=2.2, alpha=0.5)
            ax.plot(ex35.COMMON_LOG10_E, curve, color=color, lw=1.1)
            ax.text(7.35, 1.8 * np.interp(7.35, log10_e, aeff,
                                          right=aeff[-1]),
                    rf"$[{cos_hi:g},\,{cos_lo:g}]$", color=color, fontsize=6,
                    ha="center", va="bottom")
        ax.set_yscale("log")
        ax.set_xlim(3.0, 8.0)
        ax.set_xlabel(r"$\log_{10}(E_\nu\,/\,\mathrm{GeV})$")
        ax.set_ylabel(r"$A_{\rm eff}$ [cm$^2$]")
        _save(fig, out_dir, "47d_arca_bands")


def figure_arca_ratio(ex35, bands, model, out_dir) -> None:
    """Digitized over model, per ``cos(theta)`` band, at the digitized points."""
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        ax.axhline(1.0, color="0.6", lw=0.8, ls=":", zorder=1)
        for (cos_hi, cos_lo, log10_e, aeff), curve, color in zip(
                bands, model, BAND_COLOR):
            table = np.log10(np.clip(curve, 1.0e-30, None))
            matched = 10.0 ** np.interp(log10_e, ex35.COMMON_LOG10_E, table)
            ax.plot(log10_e, aeff / matched, color=color, lw=1.3,
                    label=rf"$[{cos_hi:g},\,{cos_lo:g}]$")
        ax.set_yscale("log")
        ax.set_xlim(3.0, 8.0)
        ax.set_ylim(0.05, 3.0)
        ax.set_xlabel(r"$\log_{10}(E_\nu\,/\,\mathrm{GeV})$")
        ax.set_ylabel(r"KM3NeT $/$ model")
        ax.legend(fontsize=6, frameon=False, loc="lower left",
                  title=r"$\cos\theta$ band", title_fontsize=6)
        _save(fig, out_dir, "47e_arca_ratio")


def figure_bands(ex35, sin_dec_centers, published, model, out_dir) -> None:
    """Published and model effective area in a few upgoing bands (replaces 35a)."""
    log10_e = ex35.COMMON_LOG10_E
    show = [int(np.argmin(np.abs(sin_dec_centers - s))) for s in SHOW_SIN_DEC]
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        for j, color in zip(show, BAND_COLOR):
            dec = np.rad2deg(np.arcsin(sin_dec_centers[j]))
            ax.plot(log10_e, published[:, j], color=color, lw=2.2, alpha=0.5)
            ax.plot(log10_e, model[:, j], color=color, lw=1.1)
            ax.text(7.5, 0.55 * np.interp(7.5, log10_e, published[:, j]),
                    rf"$\delta = {dec:.0f}^\circ$", color=color, fontsize=7,
                    ha="center", va="center")
        ax.plot([], [], color="0.3", lw=2.2, alpha=0.5, label="Published")
        ax.plot([], [], color="0.3", lw=1.1, label="First principles")
        ax.set_yscale("log")
        ax.set_xlim(4.0, 8.0)
        ax.set_xlabel(r"$\log_{10}(E_\nu\,/\,\mathrm{GeV})$")
        ax.set_ylabel(r"$A_{\rm eff}$ [cm$^2$]")
        ax.legend(fontsize=6, frameon=False, loc="lower right")
        _save(fig, out_dir, "47a_effective_area_bands")


def figure_published_sensitivity(dec_grid, matched, published_curve, out_dir) -> None:
    """Model ceiling against IceCube's published sensitivity (replaces 35c)."""
    pub_sin_dec, pub_flux = published_curve
    sin_dec = np.sin(np.deg2rad(dec_grid))
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        ax.plot(pub_sin_dec, pub_flux, color="#e7298a", lw=2.2, alpha=0.5,
                label="IceCube 14-yr PSTracks")
        ax.plot(sin_dec, matched, color="#7570b3", lw=1.3, label="First principles")
        ax.fill_between(sin_dec, matched, np.interp(sin_dec, pub_sin_dec, pub_flux),
                        color="#7570b3", alpha=0.10, lw=0)
        ax.axvspan(-1.0, 0.0, color="0.88", alpha=0.7, lw=0)
        ax.axvspan(REGENERATION_CAUTION_SIN_DEC, 1.0, color="#d95f02", alpha=0.12, lw=0)
        ax.text(0.5 * (REGENERATION_CAUTION_SIN_DEC + 1.0), 0.03,
                "regeneration\ndominated", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=5.5, color="#d95f02")
        ax.set_yscale("log")
        ax.set_xlim(-1.0, 1.0)
        ax.set_xlabel(r"$\sin\delta$")
        ax.set_ylabel(r"$E^2\,\mathrm{d}N/\mathrm{d}E$ [GeV cm$^{-2}$ s$^{-1}$]")
        ax.legend(fontsize=6, frameon=False, loc="upper center")
        _save(fig, out_dir, "47b_published_sensitivity")


def figure_site_ceiling(sites, dec_grid, sensitivity, out_dir) -> None:
    """Ultra-high-energy point-source ceiling for the five sites (replaces 35d)."""
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        for site in sites:
            ax.plot(dec_grid, sensitivity[site.name], color=site.color, lw=1.2,
                    ls=site.linestyle, label=site.name)
        ax.set_yscale("log")
        ax.set_xlim(-90.0, 90.0)
        curves = np.concatenate([sensitivity[s.name] for s in sites])
        low, high = 0.7 * curves.min(), 25.0 * curves.max()
        ax.set_ylim(low, high)
        caution = np.rad2deg(np.arcsin(REGENERATION_CAUTION_SIN_DEC))
        ax.axvspan(caution, 90.0, color="#d95f02", alpha=0.12, lw=0)
        label_side = {"NGC 1068": -1.0}
        for name, dec in ex35_reference_sources():
            ax.axvline(dec, color="0.7", lw=0.6, ls=":")
            ax.text(dec + 2.8 * label_side.get(name, 1.0), low * (high / low) ** 0.98,
                    name, rotation=90, fontsize=6, ha="center", va="top", color="0.45")
        ax.set_xticks([-90, -45, 0, 45, 90])
        ax.set_xlabel(r"source declination $\delta$ [deg]")
        ax.set_ylabel(r"$E^2\,\mathrm{d}N/\mathrm{d}E$ [GeV cm$^{-2}$ s$^{-1}$]")
        ax.legend(fontsize=6, frameon=False, loc="upper right")
        _save(fig, out_dir, "47c_site_ceiling")


_REFERENCE_SOURCES: tuple = ()


def ex35_reference_sources():
    """The marked sources, carried from example 35 at import time."""
    return _REFERENCE_SOURCES


def _save(fig, out_dir: pathlib.Path, stem: str) -> None:
    """Write one figure to ``out_dir`` as both PDF and PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {path.resolve()}")
    plt.close(fig)


def main() -> None:
    global _REFERENCE_SOURCES
    args = parse_args()

    print("Loading examples 35, 45 and 46 ...")
    ex46_mod = _HERE / "46_declination_resolved_reach.py"
    import importlib.util

    def load(stem, name):
        spec = importlib.util.spec_from_file_location(name, _HERE / stem)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    ex35 = load("35_point_source_effective_area.py", "_example_35")
    ex45 = load("45_first_principles_reach.py", "_example_45")
    ex46 = load(ex46_mod.name, "_example_46")
    _REFERENCE_SOURCES = ex35.REFERENCE_SOURCES

    min_modules = (ex45.DEFAULT_MIN_MODULES if args.min_modules is None
                   else args.min_modules)
    flavours = ("mu", "tau") if args.with_tau else ("mu",)
    sites = ex35.build_sites()
    icecube, arca = sites[0], sites[1]
    efficiency = {site.name: args.efficiency_arca for site in sites}
    efficiency["IceCube"] = args.efficiency_icecube
    mode = ("first principles" if args.first_principles
            else "fitted Lambda 33.7 m (IceCube) / 41.6 m (ARCA230)")
    print(f"  N = {min_modules:g} modules, halo weight {args.halo_weight:g}, {mode},")
    print(f"  normalization {args.efficiency_icecube:g} (IceCube) / "
          f"{args.efficiency_arca:g} (ARCA230),")
    print(f"  channels {', '.join('nu_' + f for f in flavours)}, "
          f"nu and nubar averaged")

    print("Loading the banded DR2 table ...")
    sin_dec_edges, published = ex35.icecube_banded(args.data_dir)
    sin_dec_centers = 0.5 * (sin_dec_edges[:-1] + sin_dec_edges[1:])

    print("Building the banded model ...")
    fitted = not args.first_principles
    model = banded_model(ex35, ex45, ex46, icecube, sin_dec_edges, min_modules,
                         flavours, args.halo_weight, efficiency["IceCube"], fitted)
    report_bands(ex35, sin_dec_centers, published, model)

    print("\nBuilding the ARCA230 zenith bands (Fig. 7b) ...")
    arca_bands = km3net_banded()
    arca_model = arca_banded_model(ex35, ex45, ex46, arca, arca_bands,
                                   min_modules, flavours, args.halo_weight,
                                   efficiency["ARCA230"], fitted)
    report_arca_bands(ex35, arca_bands, arca_model)

    print("\nBuilding point-source sensitivities ...")
    livetime_s = args.livetime_yr * 365.25 * 24.0 * 3600.0
    dec_grid = np.linspace(-90.0, 90.0, ex35.N_DEC_GRID)
    sensitivity: dict[str, np.ndarray] = {}
    matched = None
    for site in sites:
        print(f"  {site.name} ...")
        cos_theta, weights = ex35.zenith_band_weights(site.latitude_deg, dec_grid)
        bands = directional_model(ex35, ex45, ex46, site, cos_theta, min_modules,
                                  flavours, args.halo_weight, efficiency[site.name],
                                  fitted)
        sensitivity[site.name] = ex35.point_source_sensitivity(
            bands @ weights.T, livetime_s, args.gamma, args.emin_gev)
        if site is icecube:
            matched = ex35.point_source_sensitivity(
                bands @ weights.T,
                ex35.PUBLISHED_LIVETIME_YR * 365.25 * 24.0 * 3600.0,
                ex35.PUBLISHED_GAMMA, ex35.PUBLISHED_EMIN_GEV)

    published_curve = ex35.load_published_sensitivity(ex35._PUBLISHED_SENSITIVITY)
    ex35.report_published(dec_grid, matched, published_curve)

    print()
    figure_bands(ex35, sin_dec_centers, published, model, args.out_dir)
    figure_arca_bands(ex35, arca_bands, arca_model, args.out_dir)
    figure_arca_ratio(ex35, arca_bands, arca_model, args.out_dir)
    figure_published_sensitivity(dec_grid, matched, published_curve, args.out_dir)
    figure_site_ceiling(sites, dec_grid, sensitivity, args.out_dir)


def report_bands(ex35, sin_dec_centers, published, model) -> None:
    """Print the band-by-band level and the ceiling check."""
    log10_e = ex35.COMMON_LOG10_E
    lo, hi = ex35.STATS_LOG10_E
    band = (log10_e >= lo) & (log10_e <= hi)
    up = sin_dec_centers > 0.0

    print(f"\n  Band residual over 1e{lo:g}-1e{hi:g} GeV (published / model)")
    print(f"  {'sin(dec)':>9} {'dec[deg]':>9} {'mean':>8} {'rms':>8}")
    for j in np.flatnonzero(up)[::6]:
        res = np.log10(published[band, j] / model[band, j])
        print(f"  {sin_dec_centers[j]:9.2f} "
              f"{np.rad2deg(np.arcsin(sin_dec_centers[j])):9.1f} "
              f"{10**np.mean(res):8.3f} {np.std(res):8.4f}")
    res = np.log10(published[np.ix_(band, up)] / model[np.ix_(band, up)])
    print(f"  overall {10**np.mean(res):.3f}, rms {np.std(res):.4f} dex, "
          f"band-to-band spread {np.std(res.mean(axis=0)):.4f} dex")
    worst = np.nanmax(published[np.ix_(band, up)] / model[np.ix_(band, up)])
    print(f"  ceiling {'held' if worst <= 1.0 else 'VIOLATED'}: "
          f"max published / model = {worst:.3f}")


if __name__ == "__main__":
    main()
