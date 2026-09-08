"""Example 35 -- declination-resolved effective area and point-source reach.

Examples 28-34 all average over direction: 28 and 29 collapse the DR2 table to
one upgoing-hemisphere curve, 30 and 32 work with sky averages, and 34
integrates the halo over the whole sky. A point source is the first observable
that is *differential* in declination, and that axis is worth isolating for a
reason that has nothing to do with point sources themselves.

**The declination dependence carries no free parameters.** The three instrument
handles of Section VII -- the selection efficiency ``eps_0``, the threshold
``E_thr``, and the reach ``Lambda`` of Eq. (17) -- are all properties of the
hardware, and none of them knows where a neutrino came from. Everything that
varies with declination is transport and geometry: the PREM column the neutrino
crosses, the column the muon has available upstream of the detector, and the
area the instrument projects towards the arrival direction. So holding the
model against the published table *band by band* tests the same physics as the
sky-averaged benchmark, with the one thing the sky average could hide -- an
error that integrates away -- exposed instead.

The DR2 effective area is natively binned this way: ``IC86_effectiveArea.csv``
carries 50 uniform ``sin(dec)`` bands, and example 28 averaged over them. Figures
(a) and (b) put them back.

Three things follow from the geometry, and the figures separate them.

**Upgoing (dec > 0 at the Pole).** The neutrino crosses a PREM chord that grows
from zero at the horizon to the full diameter at the nadir, so the Earth turns
opaque from the bottom of the sky upward as the energy rises. The muon, born in
rock, has effectively unlimited column. This is the regime example 28 fitted,
and the residual should be flat across it if the transport is right.

**Downgoing (dec < 0 at the Pole).** The neutrino arrives essentially
unattenuated, but the muon cannot be born above the ice: its available column is
the ~2 km of overburden divided by ``cos(theta_z)``, not the ~16 km of water
equivalent its free range would cover. This is the truncation of example 34,
here for a power-law source rather than a line. Overlaid on that is something
the model does not have and does not claim to: the atmospheric-muon veto, which
removes essentially the whole downgoing sky at 10 TeV and still suppresses it
twentyfold at 100 TeV. What the figure shows is that the suppression *lifts* --
by 1 PeV the downgoing bands are within 25% of the upgoing ones. Above that they
overshoot: at 10 PeV the published downgoing area exceeds this model by a third,
so the geometric bound the upgoing sky respects to a few percent fails in the one
corner where the column truncation bites hardest, and the reach of Eq. (17),
fitted upgoing with its pivot far above the band, runs the wrong way to help.
That is reported rather than tuned away; the upgoing benchmark does not rest on
it, but it is where this model is weakest.

**Off the Pole.** At any other latitude a fixed declination sweeps zenith over a
sidereal day, so the effective area has to be averaged along the source's
zenith track. For an upright cylinder that also sweeps the projected area
between ``pi R^2`` overhead and ``2 R h`` at the horizon. Both averages are one
quadrature here; a Monte Carlo pipeline pays for them in simulation statistics.
The consequence is visible in figure (d): a polar site has strongly
declination-dependent reach and a mid-latitude one is nearly uniform, because
the sweep averages the two opposed effects together.

Figures (c) and (d) turn that into the observable a point-source search quotes.
Both are background-free geometric ceilings in the sense of example 34 -- no
angular cut, no selection efficiency, no background model, and no reach -- so
they must lie *below* anything a real search achieves, and how far below is the
price of the things they leave out.

Figure (c) is that check, and it is a second parameter-free benchmark on an
observable quite unlike an effective area. IceCube's 14-year PSTracks 90% CL
median sensitivity (arXiv:2507.07275) is a declination curve for the same
channel this model computes -- through-going muon tracks -- so the model is run
at that paper's own spectrum, exposure and energy range and laid against it. The
ceiling holds at every declination. It sits a factor of 3 below the achieved
sensitivity at the horizon, 4 across the northern sky, and 18 across the
southern one, which is the atmospheric-muon background appearing as the gap it
should be rather than as a fitted nuisance. The shape is reproduced without
anything being tuned to it: both curves are minimal at the horizon and rise to
either pole, for the two opposed reasons above.

Figure (d) is the forecast the check earns, and one choice in it is not cosmetic.
The sensitivity is an integral over energy, and where that integral sits decides
whether it has any declination structure at all: below ~100 TeV the Earth is
transparent from every direction and the muon range is short enough that no site
runs out of column, so a ceiling integrated from 1 TeV is flat in declination to
a factor of two whatever the transport does. The window is therefore cut at
``--emin-gev``, 100 TeV by default, which is the band this paper is about. Over
it IceCube's ceiling spans a factor of eight across the sky while ARCA230's
spans 1.2 -- a polar site trades the two effects against declination and a
mid-latitude one averages them away over each sidereal day.

Two numbers are carried in rather than fitted. The reach ``Lambda`` and its
pivot come from example 28's fit to the DR2 upgoing average; applying that one
pair to all 50 bands is a prediction, not a fit, and figure (b) shows it as the
dashed curves.

One caveat spans the two halves of this set. The effective area of figures (a)
and (b) is the DR2 release, whose 14 seasons span 2008-2022 for 13.6 yr of
good-run livetime, while the sensitivity of figure (c) is the 14-year PSTracks
selection, and the selection was not frozen between them. (Note that the DR2
readme describes the event files as a "10 year sample"; that sentence is
inherited from the DR1 readme and is contradicted by the release's own
season list and good-run lists.) The
model is the same in both, so the comparison is fair in shape; the normalization
of the figure (c) gap carries that difference.

Usage
-----
    python examples/35_point_source_effective_area.py
    python examples/35_point_source_effective_area.py --gamma 2.5
    python examples/35_point_source_effective_area.py --emin-gev 1e3
    python examples/35_point_source_effective_area.py --data-dir /path/to/dataverse_files
    python examples/35_point_source_effective_area.py --out-dir /path/to/figures

Writes four standalone figures, ``35a``-``35d``, in the order described above.
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.data.icecube import banded_effective_area
from softpaws.data.published import icecube_point_source_sensitivity
from softpaws.detectors import ARCA230, GVD, ICECUBE, PONE, TRIDENT, Site
from softpaws.response.declination import (
    band_averaged_effective_area_cm2,
    directional_effective_area_cm2,
)
from softpaws.response.declination import (
    central_energy_range as _central_energy_range,
)
from softpaws.response.declination import (
    point_source_sensitivity as _point_source_sensitivity,
)
from softpaws.response.declination import (
    polar_band_directions as _polar_band_directions,
)
from softpaws.response.declination import (
    zenith_band_weights as _zenith_band_weights,
)
from softpaws.transport.cross_section import bgr18_cross_section
from softpaws.transport.muon_range import DEFAULT_MUON_THRESHOLD_GEV

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_DATA_DIR = _HERE.parents[1] / "src" / "softpaws" / "data" / "dataverse_files"
_DEFAULT_OUT_DIR = _HERE / "output"
_PUBLISHED_SENSITIVITY = (
    _HERE.parents[1]
    / "src"
    / "softpaws"
    / "data"
    / "bounds"
    / "icecube_14year_track_sensitivity_e2.csv"
)

CROSS_SECTION = bgr18_cross_section()

# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------

# Same grid as example 28, so the two benchmarks are read on identical energies.
COMMON_LOG10_E = np.linspace(3.0, 8.0, 26)

# The DR2 simulation runs to 100 PeV (DR2_readme.txt) and the selection is still
# turning on below 100 TeV, so residuals are scored on this band only.
STATS_LOG10_E = (5.0, 7.8)

# Sub-samples per published declination band, spaced uniformly in sin(dec) --
# which is uniform in solid angle, the weighting the table itself uses. The PREM
# column varies fastest just below the horizon, where a band centre alone is not
# representative.
N_SUB_BAND = 5

# Arrival-zenith bands. The effective area depends on the arrival direction only
# through cos(theta_z), so it is evaluated once per band and the declination
# enters only through the time each source spends there -- which makes the whole
# declination scan cost one pass over these bands, and makes it worth taking
# many. It has to be many: at the Pole the hour angle drops out and each
# declination maps to a single band, so this binning *is* the resolution of
# figure (c) there, and a coarse grid shows up as visible steps.
N_COS_THETA = 180
# Hour-angle samples per sidereal day. Irrelevant at the Pole, where the zenith
# of a given declination never changes.
N_HOUR_ANGLE = 192
# Declination grid for figure (c).
N_DEC_GRID = 73

# Rungs of the neutral-current / tau regeneration ladder, as in example 34.
N_RUNG = 32
RUNG_DECADES = 4.0

# Longest path a near-horizontal muon can have in the detector medium [km].
# Only a cap on the 1/cos(theta) divergence; it exceeds every muon range here.

# Feldman-Cousins 90% CL upper limit on the signal for zero observed background.
N_EVENTS_LIMIT = 2.44

# Pivot at which the quoted point-source flux normalization is defined [GeV].
PIVOT_ENERGY_GEV = 1.0e5

# Bottom of the analysis window for figure (d) [GeV]. Below this the Earth is
# transparent from every direction and no site runs out of muon column, so the
# ceiling carries no declination information; see the module docstring.
DEFAULT_EMIN_GEV = 1.0e5

# ---------------------------------------------------------------------------
# The published point-source sensitivity, for figure (c)
# ---------------------------------------------------------------------------

# IceCube's 14-year PSTracks 90% CL median sensitivity (arXiv:2507.07275, the
# PSTracks curve of their Figs. 4 and 13), digitized as (sin(dec), E^2 dN/dE).
# Everything below is fixed by that paper rather than chosen here:
#
#   * gamma = 2, which is also why the pivot does not matter -- E^2 phi is
#     constant for an E^-2 source, so the comparison is pivot-free even though
#     both sides happen to quote it at 100 TeV.
#   * y-axis in TeV cm^-2 s^-1, per flavor, hence the factor below.
#   * 14 years, which the model is held to rather than DR2's own 13.6.
#   * their central 90% sensitive energy range for gamma = 2 runs from about
#     1 TeV near the horizon upward (their App. D), so the model is integrated
#     from the bottom of the common grid rather than over the UHE window.
#
# A 90% CL median sensitivity is what a real search achieves against its real
# background. The model is a background-free ceiling, so it must lie *below*
# this curve everywhere; how far below is the price of background and selection,
# and it is not the same price in the two hemispheres.
PUBLISHED_LIVETIME_YR = 14.0
PUBLISHED_GAMMA = 2.0
PUBLISHED_EMIN_GEV = 1.0e3
TEV_TO_GEV = 1.0e3

# ---------------------------------------------------------------------------
# Instrument handles carried in from example 28
# ---------------------------------------------------------------------------

# Reach law of Eq. (17), fitted there against the DR2 upgoing average:
# Lambda = 17.8 m per e-fold (41 m per decade) with the pivot above the fitted
# band, so it is an extrapolation there too. Nothing about it is refitted here;
# applying one pair to every declination band is the prediction figure (b) tests.
#
# Carried in by hand, so it has to be re-read from example 28 whenever the range
# changes: the frozen-kernel range gave 21.3 m per e-fold and 10^8.72 GeV, and
# leaving those in place while the range ran produced a residual that belonged
# to the stale constant and not to the declination.
REACH_KM = 0.0178
REACH_PIVOT_GEV = 10.0**9.67

# ---------------------------------------------------------------------------
# Sites
# ---------------------------------------------------------------------------

# The site geometries live in :mod:`softpaws.detectors`; this script only
# adds the figure styling for each.
SITE_COLORS = {
    "IceCube": "k",
    "ARCA230": "C0",
    "TRIDENT": "C1",
    "P-ONE": "C2",
    "Baikal-GVD": "C3",
}
SITE_LINESTYLES = {name: "-" for name in SITE_COLORS}

# ---------------------------------------------------------------------------
# Reference declinations
# ---------------------------------------------------------------------------

# Marked in figure (c). NGC 1068 and TXS 0506+056 are the two sources IceCube has
# reported evidence for; the Galactic Center anchors the halo signal of example
# 34 on the same axis. NGC 1068 sits within a hundredth of a degree of the
# celestial equator, which at the Pole is the horizon -- the least defensible
# band in this whole calculation, and the same regime flagged for KM3-230213A in
# Section X.C.
REFERENCE_SOURCES = (
    ("NGC 1068", -0.013),
    ("TXS 0506+056", 5.693),
    ("Galactic Center", -28.936),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=_DEFAULT_DATA_DIR,
        help="Root of the DR2 data directory (must contain 'irfs/' and 'uptime/').",
    )
    parser.add_argument(
        "--threshold-gev",
        type=float,
        default=DEFAULT_MUON_THRESHOLD_GEV,
        help="Muon selection threshold [GeV]; the DR2 smearing matrix supports ~700-1000.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=2.0,
        help="Spectral index of the point source used for figure (c).",
    )
    parser.add_argument(
        "--emin-gev",
        type=float,
        default=DEFAULT_EMIN_GEV,
        help="Bottom of the analysis energy window for figure (c) [GeV].",
    )
    parser.add_argument(
        "--livetime-yr",
        type=float,
        default=10.0,
        help="Exposure used for figure (c), applied to every site [yr].",
    )
    parser.add_argument(
        "--reach-km",
        type=float,
        default=REACH_KM,
        help="Growth of the light reach per e-fold [km], carried in from example 28.",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=_DEFAULT_OUT_DIR,
        help="Directory for the four figures, written as '35a'-'35d'.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Truncated first-passage range
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Sites
# ---------------------------------------------------------------------------


def build_sites() -> list[Site]:
    """The five detectors figure (c) compares.

    IceCube is first because it is the only one with a published
    declination-resolved effective area, so it is the site figures (a) and (b)
    validate the model against. The other four are the instrumented footprint
    and nothing else. The geometries are the published layouts of
    :mod:`softpaws.detectors`.

    Returns
    -------
    sites : list of Site
        Detector definitions, IceCube first.
    """
    return [ICECUBE, ARCA230, TRIDENT, PONE, GVD]


# ---------------------------------------------------------------------------
# Directional effective area
# ---------------------------------------------------------------------------


def directional_aeff_cm2(
    site: Site,
    cos_theta: np.ndarray,
    threshold_gev: float,
    reach_km: float | None = None,
    pivot_gev: float = REACH_PIVOT_GEV,
    channels: str = "both",
) -> np.ndarray:
    """Effective area per arrival direction, on ``COMMON_LOG10_E``.

    See :func:`softpaws.response.declination.directional_effective_area_cm2`.
    """
    return directional_effective_area_cm2(
        site, cos_theta, threshold_gev, reach_km, pivot_gev, channels,
        COMMON_LOG10_E, CROSS_SECTION,
    )


# ---------------------------------------------------------------------------
# Panel (a) and (b): the IceCube band-by-band benchmark
# ---------------------------------------------------------------------------


def icecube_banded(data_dir: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Livetime-weighted DR2 effective area, keeping the declination axis.

    See :func:`softpaws.data.icecube.banded_effective_area`.
    """
    return banded_effective_area(data_dir, COMMON_LOG10_E)


def polar_band_directions(sin_dec_edges: np.ndarray) -> np.ndarray:
    """Directions sampling each published band; see :func:`polar_band_directions`."""
    return _polar_band_directions(sin_dec_edges, N_SUB_BAND)


def icecube_model_banded(
    site: Site,
    sin_dec_edges: np.ndarray,
    threshold_gev: float,
    reach_km: float | None = None,
    pivot_gev: float = REACH_PIVOT_GEV,
) -> np.ndarray:
    """Model effective area averaged within each published declination band.

    See :func:`softpaws.response.declination.band_averaged_effective_area_cm2`.
    """
    return band_averaged_effective_area_cm2(
        site, sin_dec_edges, threshold_gev, reach_km, pivot_gev,
        COMMON_LOG10_E, CROSS_SECTION, N_SUB_BAND,
    )


# ---------------------------------------------------------------------------
# Figure (c): point-source sensitivity
# ---------------------------------------------------------------------------


def zenith_band_weights(latitude_deg: float, dec_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Daily time fraction per zenith band; see :func:`zenith_band_weights`."""
    return _zenith_band_weights(latitude_deg, dec_deg, N_COS_THETA, N_HOUR_ANGLE)


def point_source_sensitivity(
    aeff_cm2: np.ndarray,
    livetime_s: float,
    gamma: float,
    emin_gev: float = DEFAULT_EMIN_GEV,
) -> np.ndarray:
    """Background-free point-source ceiling; see :func:`point_source_sensitivity`."""
    return _point_source_sensitivity(
        aeff_cm2, livetime_s, gamma, emin_gev, COMMON_LOG10_E, PIVOT_ENERGY_GEV, N_EVENTS_LIMIT
    )


def load_published_sensitivity(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """IceCube's published point-source sensitivity, in GeV cm^-2 s^-1.

    See :func:`softpaws.data.published.icecube_point_source_sensitivity`.
    """
    return icecube_point_source_sensitivity(path)


def central_energy_range(
    aeff_cm2: np.ndarray,
    gamma: float,
    emin_gev: float = DEFAULT_EMIN_GEV,
) -> tuple[float, float]:
    """Central 90% energy range of the signal; see :func:`central_energy_range`."""
    return _central_energy_range(aeff_cm2, gamma, emin_gev, COMMON_LOG10_E, PIVOT_ENERGY_GEV)


# ---------------------------------------------------------------------------
# Reporting and plotting
# ---------------------------------------------------------------------------


def report_bands(
    sin_dec_centers: np.ndarray,
    published: np.ndarray,
    static: np.ndarray,
    with_reach: np.ndarray,
) -> None:
    """Print the band-by-band residuals behind figures (a) and (b)."""
    lo, hi = STATS_LOG10_E
    band = (COMMON_LOG10_E >= lo) & (COMMON_LOG10_E <= hi)
    upgoing = sin_dec_centers > 0.0

    print(f"\nBand-by-band residual over 1e{lo:g}-1e{hi:g} GeV (published / model)")
    print(f"{'sin(dec)':>9} {'dec [deg]':>10} {'static':>18} {'+ reach':>18}")
    for j in np.flatnonzero(upgoing)[::6]:
        dec = np.rad2deg(np.arcsin(sin_dec_centers[j]))
        r_static = np.log10(published[band, j] / static[band, j])
        r_reach = np.log10(published[band, j] / with_reach[band, j])
        print(
            f"{sin_dec_centers[j]:9.2f} {dec:10.1f} "
            f"{np.mean(r_static):+8.3f} +- {np.std(r_static):5.3f} "
            f"{np.mean(r_reach):+8.3f} +- {np.std(r_reach):5.3f}"
        )
    print("  (mean +- rms in dex; zero is perfect, nothing is fitted per band)")

    for label, model in (("static", static), ("+ reach", with_reach)):
        residual = np.log10(published[np.ix_(band, upgoing)] / model[np.ix_(band, upgoing)])
        # Spread of the band means is the quantity the sky average cannot see:
        # an offset common to every band is a normalization, a spread is not.
        band_means = residual.mean(axis=0)
        print(
            f"\n  {label:>8}: overall {np.mean(residual):+.3f} dex, "
            f"rms {np.std(residual):.3f} dex over the upgoing sky"
        )
        print(
            f"           {'':>0}band-to-band spread of the mean {np.std(band_means):.3f} dex, "
            f"range {band_means.min():+.3f} to {band_means.max():+.3f}"
        )

    # The downgoing sky is suppressed by the atmospheric-muon veto, which the
    # model does not have. The interesting part is that the suppression lifts:
    # once the veto stops biting, a hemisphere with a hard column truncation and
    # nothing fitted lands on the same ratio as the upgoing one.
    down = sin_dec_centers < 0.0
    print("\n  Downgoing sky, where the model has no veto (median published / model)")
    print(f"{'log10(E/GeV)':>13} {'downgoing':>12} {'upgoing':>12} {'ratio':>8}")
    for log10_e in (4.0, 5.0, 6.0, 7.0, 7.8):
        i = int(np.argmin(np.abs(COMMON_LOG10_E - log10_e)))
        med_down = float(np.median(published[i, down] / static[i, down]))
        med_up = float(np.median(published[i, upgoing] / static[i, upgoing]))
        print(f"{log10_e:13.1f} {med_down:12.3f} {med_up:12.3f} {med_down / med_up:8.2f}")
    print("  (the last column reaching 1 is the veto turning off, not a fit)")

    # A geometric ceiling requires published / model <= 1. It holds where the
    # model was built to hold and fails in the corner where the truncation is
    # hardest, which is worth stating rather than averaging away.
    print("\n  Ceiling check, max published / model over the scored band (should be <= 1)")
    for label, mask in (("upgoing", upgoing), ("downgoing", ~upgoing)):
        for name, model in (("static", static), ("+ reach", with_reach)):
            block = (published / model)[np.ix_(band, mask)]
            j = int(np.unravel_index(int(np.nanargmax(block)), block.shape)[1])
            i = int(np.unravel_index(int(np.nanargmax(block)), block.shape)[0])
            print(
                f"{label:>12} {name:>8}: {np.nanmax(block):5.2f} at "
                f"sin(dec) = {sin_dec_centers[mask][j]:+.2f}, "
                f"log10(E/GeV) = {COMMON_LOG10_E[band][i]:.1f}"
            )
    print(
        "  The static model holds the bound upgoing to a few percent even point by point.\n"
        "  Downgoing above ~10 PeV it does not: the muon there is capped at ~2 km of ice\n"
        "  while the array is accepting tracks from well outside its footprint, and the\n"
        "  reach runs the wrong way to help, since below its pivot Eq. (17) shrinks the\n"
        "  radius rather than growing it. Nothing here fits that corner and the upgoing\n"
        "  benchmark does not rest on it, but it is where this model is weakest."
    )


def report_sites(
    sites: list[Site],
    dec_grid: np.ndarray,
    sensitivity: dict[str, np.ndarray],
    aeff_at_sources: dict[str, np.ndarray],
    gamma: float,
    emin_gev: float,
) -> None:
    """Print the point-source sensitivities behind figure (d)."""
    print(
        f"\nPoint-source ceiling, E^2 phi at 100 TeV [GeV cm^-2 s^-1], "
        f"gamma = {gamma:g}, E_nu > 10^{np.log10(emin_gev):.0f} GeV"
    )
    header = " ".join(f"{name:>14}" for name, _ in REFERENCE_SOURCES)
    print(f"{'site':>12} {header} {'best dec':>9} {'sky span':>9}")
    for site in sites:
        curve = sensitivity[site.name]
        values = " ".join(
            f"{np.interp(dec, dec_grid, curve):14.2e}" for _, dec in REFERENCE_SOURCES
        )
        best = dec_grid[int(np.argmin(curve))]
        print(f"{site.name:>12} {values} {best:8.0f}d {curve.max() / curve.min():8.1f}x")
    print("  (sky span is worst declination over best: how uniform the site's coverage is)")

    print("\n  central 90% energy range of the signal [log10(E_nu/GeV)]")
    for site in sites:
        ranges = " ".join(
            "{:>5.1f}-{:<5.1f}".format(
                *central_energy_range(aeff_at_sources[site.name][:, k], gamma, emin_gev)
            )
            for k in range(len(REFERENCE_SOURCES))
        )
        print(f"{site.name:>12} {ranges}")

    reference = sites[0]
    print(f"\n  relative to {reference.name}, median over declination")
    for site in sites[1:]:
        ratio = float(np.median(sensitivity[reference.name] / sensitivity[site.name]))
        verdict = f"{ratio:.2f}x better" if ratio > 1.0 else f"{1.0 / ratio:.2f}x worse"
        print(f"{site.name:>12} {verdict:>16}")


def report_published(
    dec_grid: np.ndarray,
    matched: np.ndarray,
    published: tuple[np.ndarray, np.ndarray],
) -> None:
    """Print the gap between the model ceiling and the achieved sensitivity."""
    pub_sin_dec, pub_flux = published
    sin_dec = np.sin(np.deg2rad(dec_grid))
    inside = (sin_dec >= pub_sin_dec.min()) & (sin_dec <= pub_sin_dec.max())
    interpolated = np.interp(sin_dec, pub_sin_dec, pub_flux)
    gap = interpolated / matched

    print(
        f"\nAgainst IceCube's published 14-year PSTracks sensitivity "
        f"(gamma = {PUBLISHED_GAMMA:g}, {PUBLISHED_LIVETIME_YR:g} yr, "
        f"E_nu > 10^{np.log10(PUBLISHED_EMIN_GEV):.0f} GeV)"
    )
    print(f"{'sin(dec)':>9} {'published':>12} {'model':>12} {'ratio':>7}")
    for target in (-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 0.95):
        j = int(np.argmin(np.abs(sin_dec - target)))
        print(
            f"{sin_dec[j]:9.2f} {interpolated[j]:12.2e} {matched[j]:12.2e} {gap[j]:7.1f}"
        )
    print("  (ratio > 1 means the ceiling lies below what the search achieves, as it must)")

    north = inside & (sin_dec > 0.0)
    south = inside & (sin_dec < 0.0)
    print(
        f"\n  northern (upgoing) sky: ceiling is {np.median(gap[north]):.1f}x below the "
        "achieved sensitivity"
    )
    print(
        f"  southern (downgoing) sky: {np.median(gap[south]):.1f}x below -- the "
        "atmospheric-muon background the ceiling does not carry"
    )
    violated = np.flatnonzero(inside & (gap < 1.0))
    if violated.size:
        print(
            f"  WARNING: ceiling exceeded at {violated.size} declinations, "
            f"worst {1.0 / gap[violated].min():.2f}x"
        )
    else:
        print("  the ceiling is respected at every declination, which is the check")


def _save(fig: "plt.Figure", out_dir: pathlib.Path, stem: str) -> None:
    """Write one figure to ``out_dir`` as both PDF and PNG.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to write.
    out_dir : pathlib.Path
        Destination directory, created if missing.
    stem : str
        File name without a suffix.
    """
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {path.resolve()}")
    plt.close(fig)


def _curve_angle_deg(ax: "plt.Axes", x: np.ndarray, y: np.ndarray, x0: float) -> float:
    """On-screen angle of a curve at ``x0``, in degrees.

    A data-space slope is not the angle a label should carry: the axes are log
    in ``y`` and linear in ``x``, and the box is not square. Transforming two
    nearby points through ``ax.transData`` gives the angle actually drawn, so
    the label lies along the curve whatever the aspect ratio turns out to be.
    Call only after the limits are set and the canvas has been drawn.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes the curve lives in, with its limits already final.
    x, y : np.ndarray
        The curve, sorted ascending in ``x``.
    x0 : float
        Where along the curve to measure.

    Returns
    -------
    angle : float
        Rotation [deg] for a text label lying along the curve.
    """
    span = 0.04 * (np.max(x) - np.min(x))
    ends = np.array([x0 - span, x0 + span])
    points = ax.transData.transform(
        np.column_stack([ends, np.interp(ends, x, y)])
    )
    delta = points[1] - points[0]
    return float(np.degrees(np.arctan2(delta[1], delta[0])))


def figure_bands(
    sin_dec_centers: np.ndarray,
    published: np.ndarray,
    static: np.ndarray,
    out_dir: pathlib.Path,
) -> None:
    """Published and model effective area in a few upgoing declination bands."""
    show_bands = [
        int(np.argmin(np.abs(sin_dec_centers - s))) for s in (0.1, 0.4, 0.7, 0.95)
    ]
    label_x = 7.5

    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        for j, color in zip(show_bands, ("C0", "C1", "C2", "C3")):
            dec = np.rad2deg(np.arcsin(sin_dec_centers[j]))
            ax.plot(COMMON_LOG10_E, published[:, j], color=color, lw=1.4)
            ax.plot(COMMON_LOG10_E, static[:, j], color=color, lw=1.0, ls="--")
            # Sat between this band's curve and the next one down: the bands are
            # only a factor of two or three apart here, so a larger drop would
            # land the label on its neighbour.
            ax.text(
                label_x,
                0.6 * np.interp(label_x, COMMON_LOG10_E, published[:, j]),
                rf"$\delta = {dec:.0f}^\circ$",
                color=color,
                fontsize=8,
                ha="center",
                va="center",
            )
        ax.plot([], [], color="0.3", lw=1.4, label="IceCube")
        ax.plot([], [], color="0.3", lw=1.0, ls="--", label="Model")
        ax.set_yscale("log")
        ax.set_xlim(COMMON_LOG10_E[0], COMMON_LOG10_E[-1])
        ax.set_xlabel(r"$\log_{10}(E_\nu\,/\,\mathrm{GeV})$", fontsize=8)
        ax.set_ylabel(r"$A_{\rm eff}$ [cm$^2$]", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, loc="lower right")
        _save(fig, out_dir, "35a_effective_area_bands")


def figure_residual(
    sin_dec_centers: np.ndarray,
    published: np.ndarray,
    static: np.ndarray,
    with_reach: np.ndarray,
    out_dir: pathlib.Path,
) -> None:
    """Residual against the published table, band by band across the sky."""
    show_energies = (5.0, 6.0, 7.0)
    text_x = -0.5

    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        lowest = None
        for log10_e, color in zip(show_energies, ("C0", "C1", "C2")):
            i = int(np.argmin(np.abs(COMMON_LOG10_E - log10_e)))
            ratio = published[i] / static[i]
            ax.plot(
                sin_dec_centers,
                ratio,
                color=color,
                lw=1.1,
                label=rf"$10^{{{log10_e:.0f}}}$ GeV",
            )
            ax.plot(sin_dec_centers, published[i] / with_reach[i], color=color, lw=0.9, ls="--")
            if lowest is None:
                lowest = ratio
        ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
        ax.axvspan(-1.0, 0.0, color="0.88", alpha=0.7, lw=0)
        # Tucked under the lowest curve, which is the 10^5 GeV one: that is where
        # the veto bites hardest and so where the empty space is.
        ax.text(
            text_x,
            0.45 * float(np.interp(text_x, sin_dec_centers, lowest)),
            "downgoing:\nveto suppressed",
            fontsize=8,
            ha="center",
            va="top",
        )
        ax.set_yscale("log")
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(1.0e-3, 3.0)
        ax.set_xlabel(r"$\sin\delta$", fontsize=8)
        ax.set_ylabel("IceCube / model", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, loc="lower right")
        _save(fig, out_dir, "35b_residual_by_declination")


def figure_published_sensitivity(
    dec_grid: np.ndarray,
    matched: np.ndarray,
    published_sensitivity: tuple[np.ndarray, np.ndarray],
    out_dir: pathlib.Path,
) -> None:
    """Model ceiling against IceCube's published point-source sensitivity."""
    pub_sin_dec, pub_flux = published_sensitivity
    sin_dec = np.sin(np.deg2rad(dec_grid))
    label_x = 0.3

    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        ax.plot(pub_sin_dec, pub_flux, color="k", lw=1.4)
        ax.plot(sin_dec, matched, color="C0", lw=1.1, ls="--")
        ax.fill_between(
            sin_dec,
            matched,
            np.interp(sin_dec, pub_sin_dec, pub_flux),
            color="C0",
            alpha=0.12,
            lw=0,
        )
        ax.axvspan(-1.0, 0.0, color="0.88", alpha=0.7, lw=0)
        ax.set_yscale("log")
        ax.set_xlim(-1.0, 1.0)
        ax.set_xlabel(r"$\sin\delta$", fontsize=8)
        ax.set_ylabel(r"$E^2\,\mathrm{d}N/\mathrm{d}E$ [GeV cm$^{-2}$ s$^{-1}$]", fontsize=8)
        ax.tick_params(labelsize=8)

        # The labels replace a legend, so they have to sit on their curves: draw
        # once to freeze the transform, then take the angle off the screen.
        fig.canvas.draw()
        for x, y, name, color in (
            (pub_sin_dec, pub_flux, "IceCube", "k"),
            (sin_dec, matched, "Model", "C0"),
        ):
            ax.text(
                label_x,
                1.18 * float(np.interp(label_x, x, y)),
                name,
                fontsize=8,
                color=color,
                ha="center",
                va="bottom",
                rotation=_curve_angle_deg(ax, x, y, label_x),
                rotation_mode="anchor",
            )
        _save(fig, out_dir, "35c_published_sensitivity")


def figure_site_ceiling(
    sites: list[Site],
    dec_grid: np.ndarray,
    sensitivity: dict[str, np.ndarray],
    out_dir: pathlib.Path,
) -> None:
    """Ultra-high-energy point-source ceiling for the five sites."""
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        for site in sites:
            ax.plot(
                dec_grid,
                sensitivity[site.name],
                color=SITE_COLORS[site.name],
                lw=1.2,
                ls=SITE_LINESTYLES[site.name],
                label=site.name,
            )
        ax.set_yscale("log")
        ax.set_xlim(-90.0, 90.0)
        # Headroom above the curves so the legend and the source markers have
        # somewhere to live that is not on top of the data. At 8 pt the source
        # names are tall enough that this has to be generous.
        curves = np.concatenate([sensitivity[site.name] for site in sites])
        low, high = 0.7 * curves.min(), 25.0 * curves.max()
        ax.set_ylim(low, high)
        # NGC 1068 and TXS 0506+056 are six degrees apart, so their labels would
        # collide if both sat on the same side. Putting NGC 1068 to the left of
        # its line separates them without staggering them in height.
        label_side = {"NGC 1068": -1.0}
        for name, dec in REFERENCE_SOURCES:
            ax.axvline(dec, color="0.7", lw=0.6, ls=":")
            ax.text(
                dec + 2.8 * label_side.get(name, 1.0),
                low * (high / low) ** 0.98,
                name,
                rotation=90,
                fontsize=8,
                ha="center",
                va="top",
                color="0.45",
            )
        ax.set_xticks([-90, -45, 0, 45, 90])
        ax.set_xlabel(r"source declination $\delta$ [deg]", fontsize=8)
        ax.set_ylabel(r"$E^2\,\mathrm{d}N/\mathrm{d}E$ [GeV cm$^{-2}$ s$^{-1}$]", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, loc="upper right")
        _save(fig, out_dir, "35d_site_ceiling")


def main() -> None:
    args = parse_args()
    sites = build_sites()
    icecube = sites[0]

    print(f"Loading IceCube IRFs from: {args.data_dir}")
    sin_dec_edges, published = icecube_banded(args.data_dir)
    sin_dec_centers = 0.5 * (sin_dec_edges[:-1] + sin_dec_edges[1:])
    print(f"  {published.shape[1]} declination bands kept, {published.shape[0]} energies")

    print("Building the model band by band (nothing fitted) ...")
    static = icecube_model_banded(icecube, sin_dec_edges, args.threshold_gev)
    print(
        f"  carrying in example 28's reach: {args.reach_km * 1e3:.1f} m per e-fold, "
        f"pivot 10^{np.log10(REACH_PIVOT_GEV):.2f} GeV"
    )
    with_reach = icecube_model_banded(
        icecube, sin_dec_edges, args.threshold_gev, reach_km=args.reach_km
    )
    report_bands(sin_dec_centers, published, static, with_reach)

    print("\nBuilding point-source sensitivities ...")
    livetime_s = args.livetime_yr * 365.25 * 24.0 * 3600.0
    dec_grid = np.linspace(-90.0, 90.0, N_DEC_GRID)
    source_decs = np.array([dec for _, dec in REFERENCE_SOURCES])

    sensitivity: dict[str, np.ndarray] = {}
    aeff_at_sources: dict[str, np.ndarray] = {}
    for site in sites:
        print(f"  {site.name} ...")
        cos_theta, weights = zenith_band_weights(site.latitude_deg, dec_grid)
        # (n_e, n_band) once, then contracted with the per-declination exposure.
        bands = directional_aeff_cm2(site, cos_theta, args.threshold_gev)
        sensitivity[site.name] = point_source_sensitivity(
            bands @ weights.T, livetime_s, args.gamma, args.emin_gev
        )
        _, source_weights = zenith_band_weights(site.latitude_deg, source_decs)
        aeff_at_sources[site.name] = bands @ source_weights.T
        if site is icecube:
            # Figure (c) is held to the published analysis' own setup rather than
            # to the ultra-high-energy window and exposure figure (d) chooses.
            matched = point_source_sensitivity(
                bands @ weights.T,
                PUBLISHED_LIVETIME_YR * 365.25 * 24.0 * 3600.0,
                PUBLISHED_GAMMA,
                PUBLISHED_EMIN_GEV,
            )

    published_curve = load_published_sensitivity(_PUBLISHED_SENSITIVITY)
    report_published(dec_grid, matched, published_curve)
    report_sites(sites, dec_grid, sensitivity, aeff_at_sources, args.gamma, args.emin_gev)

    print()
    figure_bands(sin_dec_centers, published, static, args.out_dir)
    figure_residual(sin_dec_centers, published, static, with_reach, args.out_dir)
    figure_published_sensitivity(dec_grid, matched, published_curve, args.out_dir)
    figure_site_ceiling(sites, dec_grid, sensitivity, args.out_dir)


if __name__ == "__main__":
    main()
