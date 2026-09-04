"""Example 45 -- the reach law with nothing fitted.

Example 44 showed that the gap between our effective-area ceiling and the
published tables is a **depth-dependent light reach**: the effective footprint
has to be evaluated at the energy the muon has where it is seen, not at the
energy it was born with. That example still fitted the two numbers the reach law
carries, the growth per e-fold ``Lambda`` and the pivot ``E_piv``. Here both are
derived, and the two-detector comparison of example 32 is repeated with nothing
adjusted against a published curve.

Deriving the reach
------------------
A muon's Cherenkov output per unit length follows its energy loss. The bare
track radiates at the Frank-Tamm rate

.. math:: \\frac{{\\rm d}N_\\gamma}{{\\rm d}x} = 2\\pi\\alpha
    \\left(\\frac{1}{\\lambda_1} - \\frac{1}{\\lambda_2}\\right)
    \\left(1 - n^{-2}\\right),

and each GeV of radiative loss makes an electromagnetic shower carrying
``L_em ~ 4 m`` of charged track, so the yield is

.. math:: N'(E) = \\frac{{\\rm d}N_\\gamma}{{\\rm d}x}\\,
    \\left(1 + L_{\\rm em}\\, b_\\mu E\\right),

the ``a + bE`` of the loss law with its own coefficients. ``L_em b_mu`` is
density independent, since the shower track length scales as ``1 / rho`` and
``b_mu`` as ``rho``, so the ratio sets a light critical energy near 700 GeV
against the 537 GeV of the loss law.

Light leaves a track cylindrically and is attenuated on the effective photon
length ``Lambda``, which is ``sqrt(lambda_abs lambda_scat / 3)`` where scattering
dominates, as in deep ice, and ``lambda_abs`` where it does not, as in sea water.
Summing the collected charge over the modules within one attenuation length of
the track, the module density cancels and the detection condition reduces to a
charge per nearby module,

.. math:: \\frac{A_{\\rm mod} N'(E)}{\\pi \\Lambda}\\, f(x) \\ge q,

with ``A_mod`` the module's angle-averaged effective area and ``f(x)`` the
fraction of the surrounding solid angle the array fills. A track inside the
array is surrounded, so ``f = 1``; a track at distance ``x`` beyond the boundary
sees the array on one side and through the ice, so ``f = e^{-x/Lambda} / 2``.
Two numbers follow from the one condition:

``E_thr``
    The muon threshold, where a crossing track first meets the condition.

``R_eff(E) = R_det + Lambda ln[N'(E) / N'(E_piv)]``
    The effective footprint, where the pivot is set by ``N'(E_piv) =
    2 N'(E_thr)``. **The array responds beyond its own footprint once the muon
    is twice as bright as threshold**, because the array surrounds a track
    inside it and stands on one side of a track outside it.

Two conditions follow, and keeping them apart is what makes the reach derivable.
A *hit* is a local coincidence -- an HLC pair on neighbouring modules at
IceCube, two photomultipliers of one module at KM3NeT -- with each receiver
converting its share of the collected charge into at least one photoelectron at
Poisson probability (:func:`hit_probability`). What a selection then demands is
a **multiplicity**: enough of those hits, and IceCube's simple-majority trigger
asks for eight (:data:`DEFAULT_MIN_MODULES`). The mean count over the in-array
transverse plane (:func:`hit_count`) sets the reach, the Poisson probability of
meeting the multiplicity at that mean makes the threshold a smooth turn-on, and
no charge is left to choose.

Every remaining input is an instrument or medium number: the photocathode area,
the module density, the photon detection efficiency against wavelength and the
medium's absorption and scattering against wavelength. **Wavelength is not a
detail here.** Taking the efficiency flat at its peak across the nominal
300-600 nm band overstates the collected light by about a factor of two, since
the real response is a bump ~120 nm wide, cut off below by the housing glass and
above by the cathode; and deep ice is opaque past 500 nm, so the red half of the
band never reaches a distant module at all. Both integrals are done properly.

What the array counts, and what it does not
-------------------------------------------
The reach says where a track is bright enough. It does not say whether the track
is *reconstructible*, and a published response is built from reconstructed
tracks. A line that clips a corner of the array crosses the instrumented volume
and leaves again with no lever arm, and every such line is counted by the
convex-body projection of :func:`prism_projected_area_km2`. Requiring a minimum
path ``l`` inside the volume removes them, and the removal is not a flat
efficiency: the lines it removes are concentrated at oblique incidence, which is
exactly where the intact prism's cap and side terms *add*.

That single requirement fixes a residual the reach cannot. Without it the two
models sit a factor ~1.2 above both published curves at the low-energy end,
where the reach is inert and the Earth is transparent, and the factor is nearly
the same at two detectors of quite different shape -- because for any roughly
equidimensional convex body the direction-averaged area sits ~1.18 above the
equal-volume sphere's, whatever the shape. See :data:`DEFAULT_MIN_TRACK_KM` and
:func:`~softpaws.transport.soft_volume.eroded_prism_target_km2`.

The two act on different things, which is why both are needed. ``l`` sets the
level and the zenith dependence and is energy independent; ``q`` sets how fast
the footprint grows with energy and so carries the slope.

What the example reports
------------------------
Two curves per detector against its published one. ``First principles`` carries
the whole model with every number above taken from the medium and the module.
``Fitted`` floats the attenuation length ``Lambda`` against the same curve, so
the fitted value is directly comparable with what is known about the ice and the
water. That comparison is the test: an argument that predicts the coefficient is
worth more than one that only predicts the shape. ``Lambda`` and ``q`` are
floated together, which is the same two-parameter freedom example 32 gave its
fitted reach.

An optional scan over ``q`` is available with ``--scan``. It shifts ``R_eff``
rigidly and moves the threshold with it, so ``q`` sets the level and every
*shape* stays a prediction.

Both ``nu_mu`` and ``nu_tau -> tau -> mu`` are summed. The DR2 readme describes
the table as an average over simulated muon neutrino events, which reads as a
``nu_mu`` response, and its own deepest declination bands contradict that: a
``nu_mu`` flux cannot cross the full Earth diameter at 10^8 GeV, because escaping
requires degrading three decades through interactions that are neutral current
only 30% of the time. The tau cascade can, since its charged current is not
terminal. See :data:`DEFAULT_FLAVOURS`.

Where the physics lives
-----------------------
The optics chain is :mod:`softpaws.response.light_reach` and the effective-area
assembly is :mod:`softpaws.response.first_principles`. This script imports both
and re-exports their names, so the scripts that load it by path keep working:
``ex45.hit_count``, ``ex45.build_model`` and the rest resolve here with the
signatures they always had. What stays in the script is the reporting, the
multiplicity scan and the two figures.

Usage
-----
    python examples/45_first_principles_reach.py
    python examples/45_first_principles_reach.py --min-modules 4
    python examples/45_first_principles_reach.py --numu-only
    python examples/45_first_principles_reach.py --scan
"""

import argparse
import importlib.util
import pathlib
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from softpaws.detectors import (
    ANCHOR_NM,
    ARCA_OPTICS,
    DEFAULT_MIN_TRACK_KM,
    ICECUBE_OPTICS,
    Optics,
)
from softpaws.response import first_principles as _fp
from softpaws.response.first_principles import (
    ANCHORS,
    ARCA_BAND,
    CROSS_SECTION,
    CSMS_LOG10_E,
    CSMS_PB,
    DEFAULT_FLAVOURS,
    FAR_MEDIUM_SOURCE,
    IC_BAND,
    SPECIES,
    IsoscalarCrossSection,
    column_profile,
    residuals,
    rock_range_ratio,
)
from softpaws.response.light_reach import (
    DEFAULT_MIN_MODULES,
    EM_TRACK_LENGTH_M_PER_GEV,
    FINE_STRUCTURE,
    HLC_PARTNERS,
    PROJECTED_FRACTION,
    WAVELENGTH_NM,
    attenuation_length_m,
    attenuation_spectrum_m,
    brightness_factor,
    cherenkov_spectrum_per_m_per_nm,
    detection_efficiency,
    effective_body_km,
    hit_count,
    hit_probability,
    hit_radius_m,
    instrumented_chord_km,
    module_area_m2,
    module_charge_pe,
    muon_threshold_gev,
    reach_offset_m,
)
from softpaws.transport.soft_volume import DEFAULT_MUON_THRESHOLD_GEV, stochastic_muon_range_km
from softpaws.utils.constants import M_PER_KM

#: Names the scripts that load this one by path reach for. Listing them keeps
#: the re-exports explicit; the definitions live in the library.
__all__ = [
    "ANCHORS",
    "ANCHOR_NM",
    "ARCA_BAND",
    "ARCA_SITE",
    "CROSS_SECTION",
    "CSMS_LOG10_E",
    "CSMS_PB",
    "DEFAULT_FLAVOURS",
    "DEFAULT_MIN_MODULES",
    "DEFAULT_MIN_TRACK_KM",
    "DEFAULT_MUON_THRESHOLD_GEV",
    "EM_TRACK_LENGTH_M_PER_GEV",
    "FAR_MEDIUM_SOURCE",
    "FINE_STRUCTURE",
    "HLC_PARTNERS",
    "ICECUBE_SITE",
    "IC_BAND",
    "IsoscalarCrossSection",
    "PROJECTED_FRACTION",
    "SPECIES",
    "Site",
    "WAVELENGTH_NM",
    "arca_column_volume_km3",
    "arca_effective_area_cm2",
    "attenuation_length_m",
    "attenuation_spectrum_m",
    "brightness_factor",
    "build_model",
    "cherenkov_spectrum_per_m_per_nm",
    "column_profile",
    "detection_efficiency",
    "detector_curves",
    "effective_body_km",
    "fit_reach",
    "hit_count",
    "hit_probability",
    "hit_radius_m",
    "ic_column_volume_km3",
    "ic_effective_area_cm2",
    "instrumented_chord_km",
    "load_example_32",
    "module_area_m2",
    "module_charge_pe",
    "muon_threshold_gev",
    "reach_offset_m",
    "residuals",
    "rock_range_ratio",
    "stochastic_muon_range_km",
]

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"
_DEFAULT_DATA_DIR = _HERE.parent / "src" / "softpaws" / "data" / "dataverse_files"

#: Energy range both figures are drawn over, ``log10(E_nu / GeV)``.
PLOT_BAND = (4.0, 9.0)

#: Colour carries which curve it is and line style carries which detector, so
#: the two legends factorize. Shared with example 32.
MODEL_COLOR = {
    "Published": "#e7298a",
    "First principles": "#7570b3",
    "Fitted": "#1b9e77",
}
MODEL_WIDTH = {"Published": 2.4, "First principles": 1.0, "Fitted": 1.0}
MODEL_ZORDER = {"Published": 1, "First principles": 3, "Fitted": 2}
#: The published curve is drawn wide, pale and underneath, so the model curves
#: reading on top of it stay legible exactly where they agree with it.
MODEL_ALPHA = {"Published": 0.5, "First principles": 1.0, "Fitted": 1.0}
MODELS = ("First principles", "Fitted")
DETECTOR_STYLE = {"IceCube": "-", "KM3NeT/ARCA230": "--"}


# The optical media and modules live in :mod:`softpaws.detectors.optics`;
# the names below are kept for the scripts that load this one.
Site = Optics
ICECUBE_SITE = ICECUBE_OPTICS
ARCA_SITE = ARCA_OPTICS


def load_example_32():
    """Import example 32, whose published curves this reuses."""
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
    parser.add_argument("--min-modules", type=float, default=DEFAULT_MIN_MODULES,
                        help="Modules that must collect at least one photoelectron.")
    parser.add_argument("--min-track-km", type=float, default=None,
                        help="Override the minimum in-detector path [km] at BOTH sites. "
                             "By default each keeps its own; see DEFAULT_MIN_TRACK_KM.")
    parser.add_argument("--n-energy", type=int, default=40,
                        help="Points in the arrival-energy quadrature.")
    parser.add_argument("--numu-only", action="store_true",
                        help="Drop the nu_tau -> tau -> mu channel. Fails the ceiling in "
                             "the deepest declination bands; see DEFAULT_FLAVOURS.")
    parser.add_argument("--scan", action="store_true",
                        help="Also scan the per-module charge, which sets the level.")
    parser.add_argument("--out", type=pathlib.Path,
                        default=_DEFAULT_OUT_DIR / "45_first_principles_reach",
                        help="Output stem; '_area' and '_ratio' figures are written, "
                             "each as .pdf and .png.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# The effective-area builders, with the signature the loading scripts use
# ---------------------------------------------------------------------------
#
# These used to take example 32 as their first argument, for its geometry,
# columns and transmission. All of that now lives in the library, so the
# argument is accepted and not read; the loading scripts pass it unchanged.


def ic_column_volume_km3(ex32, production_gev, threshold_gev, cos_theta, site, min_modules,
                         n_energy):
    """IceCube target volume [km^3].

    See :func:`softpaws.response.first_principles.ic_column_volume_km3`.
    """
    return _fp.ic_column_volume_km3(production_gev, threshold_gev, cos_theta, site,
                                    min_modules, n_energy)


def ic_effective_area_cm2(ex32, site, threshold_gev, min_modules, n_energy,
                          flavours=DEFAULT_FLAVOURS, cross_section=None):
    """IceCube effective area [cm^2].

    See :func:`softpaws.response.first_principles.ic_effective_area_cm2`.
    """
    return _fp.ic_effective_area_cm2(site, threshold_gev, min_modules, n_energy, flavours,
                                     cross_section)


def arca_column_volume_km3(ex32, production_gev, threshold_gev, theta_deg, available_km, site,
                           min_modules, n_energy):
    """ARCA230 column volume [km^3].

    See :func:`softpaws.response.first_principles.arca_column_volume_km3`.
    """
    return _fp.arca_column_volume_km3(production_gev, threshold_gev, theta_deg, available_km,
                                      site, min_modules, n_energy)


def arca_effective_area_cm2(ex32, site, threshold_gev, min_modules, n_energy,
                            flavours=DEFAULT_FLAVOURS, cross_section=None):
    """ARCA230 effective area [cm^2].

    See :func:`softpaws.response.first_principles.arca_effective_area_cm2`.
    """
    return _fp.arca_effective_area_cm2(site, threshold_gev, min_modules, n_energy, flavours,
                                       cross_section)


def build_model(ex32, which, site, min_modules, n_energy, flavours=DEFAULT_FLAVOURS):
    """The full model for one detector [cm^2].

    See :func:`softpaws.response.first_principles.build_model`.
    """
    return _fp.build_model(which, site, min_modules, n_energy, flavours)


def fit_reach(ex32, which, site, min_modules, n_energy, published, band, flavours):
    """Float the attenuation length.

    See :func:`softpaws.response.first_principles.fit_reach`.
    """
    return _fp.fit_reach(which, site, min_modules, n_energy, published, band, flavours)


def detector_curves(ex32, which, site, min_modules, n_energy, published, band,
                    flavours=DEFAULT_FLAVOURS):
    """The three curves of one detector.

    See :func:`softpaws.response.first_principles.detector_curves`.
    """
    return _fp.detector_curves(which, site, min_modules, n_energy, published, band, flavours)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def report_site(site: Site, min_modules: float, radius_km: float, height_km: float,
                n_sides: int | None) -> None:
    """Print the derived light-reach numbers for one site."""
    lam = attenuation_length_m(site)
    chord_km = instrumented_chord_km(radius_km, height_km, n_sides)
    threshold = muon_threshold_gev(site, min_modules, chord_km)
    spectrum = cherenkov_spectrum_per_m_per_nm(site)
    efficiency = detection_efficiency(site)
    bare = float(np.trapezoid(spectrum, WAVELENGTH_NM))
    detectable = float(np.trapezoid(spectrum * efficiency, WAVELENGTH_NM))
    print(f"\n  {site.name}")
    print(f"    attenuation length at {ANCHOR_NM:.0f} nm  {lam:6.1f} m "
          f"(band {attenuation_spectrum_m(site).min():.0f} to "
          f"{attenuation_spectrum_m(site).max():.0f})")
    print(f"    photocathode area           {module_area_m2(site) * 1e4:6.1f} cm^2 projected")
    print(f"    bare-track yield            {bare:6.2e} photons/m")
    print(f"    detectable yield            {detectable:6.2e} pe/m "
          f"(mean efficiency {detectable / bare:.3f} against a flat "
          f"{site.quantum_efficiency:.2f})")
    print(f"    module density              {site.module_density_per_km3:6.0f} per km^3")
    print(f"    instrumented mean chord     {chord_km * M_PER_KM:6.0f} m")
    print(f"    one-pe radius at 1 PeV      {float(hit_radius_m(1.0e6, site)[0]):6.0f} m")
    print(f"    reach at 1 PeV              "
          f"{float(reach_offset_m(np.array([1.0e6]), site, chord_km, min_modules)[0]):6.0f} m")
    print(f"    muon threshold              {threshold:10,.0f} GeV")
    # Where the vertical reach stops growing, which is the medium and not the light.
    for label, headroom in (("above", site.headroom_above_m),
                            ("below", site.headroom_below_m)):
        print(f"    headroom {label}              {headroom:6.0f} m "
              f"({headroom / lam:.1f} attenuation lengths)")
    note = "" if site.min_track_km > 0.0 else "  (trigger level; none required)"
    print(f"    minimum in-detector track   {site.min_track_km * M_PER_KM:6.0f} m{note}")


def report_anchors(ex32, detector: str, curves: dict[str, np.ndarray],
                   band: np.ndarray) -> None:
    """Print ``published / model`` at :data:`ANCHORS`, where a flat ratio is legible.

    Parameters
    ----------
    ex32 : ModuleType
        Example 32, for the energy grids.
    detector : str
        Detector name, selecting which grid the curves sit on.
    curves : dict of str -> np.ndarray
        Curves keyed as in :data:`MODEL_COLOR`.
    band : np.ndarray
        Boolean mask of the comparison band. The ceiling test is restricted to
        it, since below threshold the model has no acceptance and the ratio
        there says nothing about a geometric ceiling.
    """
    log10_e = ex32.IC_LOG10_E if detector == "IceCube" else ex32.ARCA_LOG10_E
    for target in ANCHORS:
        i = int(np.argmin(np.abs(log10_e - target)))
        row = "  ".join(
            f"{model} {curves['Published'][i] / curves[model][i]:.3f}" for model in MODELS
        )
        print(f"      log10(E/GeV) = {log10_e[i]:.1f}   {row}")
    with np.errstate(divide="ignore"):
        ratio = curves["Published"] / curves["First principles"]
    valid = band & np.isfinite(ratio)
    j = int(np.nanargmax(np.where(valid, ratio, -np.inf)))
    verdict = "held" if ratio[j] <= 1.0 else "VIOLATED"
    print(f"      ceiling {verdict}: max published/model = {ratio[j]:.3f} "
          f"at log10(E/GeV) = {log10_e[j]:.1f}")


def scan_charge(ex32, which, site, published, band, multiplicities, n_energy, flavours,
                radius_km, height_km, n_sides):
    """Shape scatter left by the reach model against the required multiplicity.

    Returns
    -------
    table : np.ndarray
        Columns ``(modules, threshold [GeV], mean ratio, scatter [dex])``.
    """
    chord_km = instrumented_chord_km(radius_km, height_km, n_sides)
    rows = []
    for modules in multiplicities:
        model = build_model(ex32, which, site, float(modules), n_energy, flavours)
        mean, scatter = residuals(published, model, band)
        threshold = muon_threshold_gev(site, float(modules), chord_km)
        rows.append((modules, threshold, mean, scatter))
        print(f"    N = {modules:5.1f} modules -> E_thr {threshold:9,.0f} GeV, "
              f"mean {mean:.3f}, {scatter:.4f} dex")
    return np.array(rows)


def main() -> None:
    args = parse_args()
    print("Loading example 32 ...")
    ex32 = load_example_32()

    flavours = ("mu",) if args.numu_only else DEFAULT_FLAVOURS
    icecube_site, arca_site = ICECUBE_SITE, ARCA_SITE
    if args.min_track_km is not None:
        icecube_site = replace(icecube_site, min_track_km=args.min_track_km)
        arca_site = replace(arca_site, min_track_km=args.min_track_km)
    print(f"\nChannels summed: {', '.join('nu_' + f for f in flavours)}")
    print("\nDerived from the medium and the module:")
    report_site(icecube_site, args.min_modules, ex32.IC_RADIUS_KM, ex32.IC_HEIGHT_KM,
                ex32.IC_N_SIDES)
    report_site(arca_site, args.min_modules, ex32.ARCA230_RADIUS_KM,
                ex32.BLOCK_HEIGHT_KM, None)

    ic_published = ex32.icecube_published(args.data_dir)
    arca_published = ex32.arca230_published(ex32.ARCA_LOG10_E)
    ic_band = (ex32.IC_LOG10_E >= IC_BAND[0]) & (ex32.IC_LOG10_E <= IC_BAND[1])
    arca_band = (ex32.ARCA_LOG10_E >= ARCA_BAND[0]) & (ex32.ARCA_LOG10_E <= ARCA_BAND[1])

    print("\nBuilding IceCube ...")
    ic_curves, ic_fit = detector_curves(ex32, "IceCube", icecube_site, args.min_modules,
                                        args.n_energy, ic_published, ic_band, flavours)
    print("Building ARCA230 ...")
    arca_curves, arca_fit = detector_curves(ex32, "ARCA", arca_site, args.min_modules,
                                            args.n_energy, arca_published, arca_band,
                                            flavours)

    print(f"\n  published over model, at N = {args.min_modules:g} modules")
    for name, curves, band in (("IceCube", ic_curves, ic_band),
                               ("KM3NeT/ARCA230", arca_curves, arca_band)):
        print(f"    {name}")
        for model in MODELS:
            mean, scatter = residuals(curves["Published"], curves[model], band)
            print(f"      {model:17s}: mean {mean:.3f}, {scatter:.4f} dex of shape")
        report_anchors(ex32, name, curves, band)

    scan = {}
    if args.scan:
        charges = np.array([2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0])
        print("\nScanning the required multiplicity, which shifts the level and moves")
        print("the threshold with it, so only the shape scatter is a test:")
        for name, which, site, published, band, geometry in (
            ("IceCube", "IceCube", icecube_site, ic_published, ic_band,
             (ex32.IC_RADIUS_KM, ex32.IC_HEIGHT_KM, ex32.IC_N_SIDES)),
            ("KM3NeT/ARCA230", "ARCA", arca_site, arca_published, arca_band,
             (ex32.ARCA230_RADIUS_KM, ex32.BLOCK_HEIGHT_KM, None)),
        ):
            print(f"  {name}")
            scan[name] = scan_charge(ex32, which, site, published, band, charges,
                                     args.n_energy, flavours, *geometry)

    summarize(ic_curves, ic_band, ic_fit, arca_curves, arca_band, arca_fit,
              icecube_site, arca_site)
    detectors = {"IceCube": (ex32.IC_LOG10_E, ic_curves, ic_band),
                 "KM3NeT/ARCA230": (ex32.ARCA_LOG10_E, arca_curves, arca_band)}
    fit_notes = {
        "IceCube": (attenuation_length_m(icecube_site), *ic_fit),
        "KM3NeT/ARCA230": (attenuation_length_m(arca_site), *arca_fit),
    }
    make_area_figure(detectors, args.out.with_name(args.out.name + "_area"), fit_notes)
    make_ratio_figure(detectors, args.out.with_name(args.out.name + "_ratio"), fit_notes)


def summarize(ic_curves, ic_band, ic_fit, arca_curves, arca_band, arca_fit,
              icecube_site=ICECUBE_SITE, arca_site=ARCA_SITE) -> None:
    """State where the first-principles curves land, and what the fit costs.

    The fitted attenuation length is deliberately *not* reported against the
    derived one. ``attenuation_override_m`` replaces the whole wavelength
    spectrum by a single flat value, while the derived number is the length at
    :data:`ANCHOR_NM`, the clarity peak. Those are different quantities -- the
    derived spectrum spans 7 to 61 m at IceCube -- so their ratio measures the
    substitution and not the medium, and quoting it invited a comparison the
    model cannot support.
    """
    print("\n  Free normalization the fitted curve needs, and the flat attenuation")
    print("  length the shape fit lands on (the calibrated pair examples 46 and 47 carry):")
    for site, (length_m, norm) in ((icecube_site, ic_fit), (arca_site, arca_fit)):
        print(f"    {site.name:16s} {norm:.3f}   Lambda {length_m:5.1f} m")

    ic_mean, ic_shape = residuals(ic_curves["Published"], ic_curves["First principles"], ic_band)
    arca_mean, arca_shape = residuals(
        arca_curves["Published"], arca_curves["First principles"], arca_band)
    print(f"\n  Level: {ic_mean:.3f} at IceCube, {arca_mean:.3f} at ARCA230, with no number")
    print("  in the optical chain set against either published curve. A hit is a")
    print("  local coincidence and the multiplicity is met at Poisson probability,")
    print("  so the threshold is a smooth turn-on, and the cross sections are the")
    print("  shipped nu-p tables pinned to the CSMS isoscalar values. Against the")
    print("  DR2 table use --numu-only; that release is built from simulated")
    print("  nu_mu events.")
    print(f"  Shape left over: {ic_shape:.4f} dex at IceCube, {arca_shape:.4f} dex at ARCA230.")


def _save(fig, out_path: pathlib.Path) -> None:
    """Write both the vector and the raster copy of one figure."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_path.with_suffix(suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {path.resolve()}")
    plt.close(fig)


def _annotate_fit(ax, fit_notes: dict[str, tuple[float, float, float]] | None) -> None:
    """Write what the ``Fitted`` curve floated, in the top-right corner.

    One line per detector: the attenuation length from the derived value at
    :data:`ANCHOR_NM` to the fitted flat override, and the free normalization.
    The two lengths are different quantities -- a spectrum's anchor against a
    flat substitute, see :func:`summarize` -- so the arrow states what the fit
    did, not a measurement of the medium.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to annotate.
    fit_notes : dict of str -> tuple of float, or None
        Per detector ``(derived length [m], fitted length [m], normalization)``.
    """
    if not fit_notes:
        return
    short = {"IceCube": "IC", "KM3NeT/ARCA230": "ARCA"}
    lines = []
    for detector, (derived_m, fitted_m, norm) in fit_notes.items():
        lines.append(f"{short.get(detector, detector)}: "
                     f"$\\Lambda$ {derived_m:.0f}$\\to${fitted_m:.0f} m")
        lines.append(f"norm 1$\\to${norm:.2f}")
    ax.text(0.97, 0.975, "\n".join(lines), transform=ax.transAxes,
            ha="right", va="top", fontsize=8, linespacing=1.4,
            multialignment="right", color=MODEL_COLOR["Fitted"], zorder=5,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                  "edgecolor": "none", "alpha": 0.85})


def make_area_figure(detectors, out_path: pathlib.Path, fit_notes=None) -> None:
    """Both detectors' effective areas on one panel, as in example 32.

    Colour carries which curve it is and line style carries which detector, so
    the six lines need three plus two legend entries. ``fit_notes`` states in
    the corner what the ``Fitted`` curve floated; see :func:`_annotate_fit`.
    """
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.0, 3.0))
        for detector, (log10_e, curves, _) in detectors.items():
            for name, curve in curves.items():
                ax.plot(log10_e, curve, color=MODEL_COLOR[name],
                        ls=DETECTOR_STYLE[detector], lw=MODEL_WIDTH[name],
                        alpha=MODEL_ALPHA[name], zorder=MODEL_ZORDER[name])
        ax.set_yscale("log")
        ax.set_xlim(*PLOT_BAND)
        ax.set_ylim(1.0e4, 1.0e9)
        ax.set_xlabel(r"$\log_{10}(E_\nu\,/\,\mathrm{GeV})$")
        ax.set_ylabel(r"$A_{\rm eff}$ [cm$^2$]")
        _annotate_fit(ax, fit_notes)

        curve_legend = ax.legend(
            handles=[plt.Line2D([], [], color=c, lw=MODEL_WIDTH[n], alpha=MODEL_ALPHA[n],
                                label=n)
                     for n, c in MODEL_COLOR.items()],
            loc="upper left", fontsize=8, frameon=False, handlelength=2.0,
        )
        ax.add_artist(curve_legend)
        ax.legend(
            handles=[plt.Line2D([], [], color="k", ls=s, lw=1.0, label=d)
                     for d, s in DETECTOR_STYLE.items()],
            loc="lower right", fontsize=8, frameon=False, handlelength=2.4,
        )
        _save(fig, out_path)


def make_ratio_figure(detectors, out_path: pathlib.Path, fit_notes=None) -> None:
    """Published over model, which is where the shape agreement is legible.

    ``fit_notes`` states in the corner what the ``Fitted`` curve floated; see
    :func:`_annotate_fit`.
    """
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.0, 3.0))
        ax.axhline(1.0, color="0.6", lw=0.8, ls=":", zorder=1)
        for detector, (log10_e, curves, _) in detectors.items():
            for name in MODELS:
                # Drawn over the whole plotted range; the fit still uses its own
                # narrower band, so the curve extends past what was fitted. The
                # model is zero below the light threshold, so the ratio is left
                # infinite there and matplotlib drops those points.
                with np.errstate(divide="ignore"):
                    ratio = curves["Published"] / curves[name]
                ax.plot(log10_e, ratio, color=MODEL_COLOR[name],
                        ls=DETECTOR_STYLE[detector], lw=1.3, zorder=3)
        ax.set_xlim(*PLOT_BAND)
        ax.set_ylim(0.0, 2.0)
        ax.set_xlabel(r"$\log_{10}(E_\nu\,/\,\mathrm{GeV})$")
        ax.set_ylabel(r"published $/$ model")
        _annotate_fit(ax, fit_notes)

        curve_legend = ax.legend(
            handles=[plt.Line2D([], [], color=MODEL_COLOR[n], lw=1.3, label=n)
                     for n in MODELS],
            loc="upper left", fontsize=8, frameon=False, handlelength=2.0,
        )
        ax.add_artist(curve_legend)
        ax.legend(
            handles=[plt.Line2D([], [], color="k", ls=s, lw=1.0, label=d)
                     for d, s in DETECTOR_STYLE.items()],
            loc="lower right", fontsize=8, frameon=False, handlelength=2.4,
        )
        _save(fig, out_path)


if __name__ == "__main__":
    main()
