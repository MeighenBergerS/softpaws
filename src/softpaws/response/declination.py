"""Effective areas resolved by arrival direction, and what a point source sees.

A published effective area is averaged over a declination band, and a point
source is a fixed declination the sky sweeps past the detector. Both need the
response before the direction average, which is what this module builds. Two
families live here.

The first takes the light reach as a fitted number, ``reach_km`` per e-fold
about a pivot, and is the parameter-light form the tabulated comparisons of
Section IV run on. The second takes the reach from the published optics of
:mod:`softpaws.response.light_reach`, erodes the body by the minimum
in-detector track and carries the Poisson multiplicity weight, which is the
first-principles form of Section IV B.

Both end in the same place: an effective area per direction, averaged inside
each published band, and from that a background-free point-source ceiling.

Notes
-----
``cos_theta = +1`` is vertically downgoing and ``-1`` the nadir, as in
:mod:`softpaws.transport.earth`. Seen from the South Pole a source at
declination ``dec`` arrives at ``cos_theta = -sin(dec)``, so the upgoing sky
is ``dec > 0``.
"""

from __future__ import annotations

import numpy as np

from ..detectors import Optics, Site
from ..transport.attenuation import flavour_transmission, regenerated_transmission
from ..transport.cross_section import CrossSection
from ..transport.soft_volume import (
    DEFAULT_MUON_THRESHOLD_GEV,
    eroded_prism_target_km2,
    light_reach_radius_km,
    truncated_muon_range_km,
    two_medium_range_ratio,
)
from ..transport.source import MEAN_INELASTICITY, mean_inelasticity, nucleon_number_density
from ..transport.tau import BR_TAU_TO_MU, MEAN_Z
from ..utils.constants import CM_PER_KM
from .effective_area import default_cross_section
from .first_principles import column_profile, rock_range_ratio
from .light_reach import effective_body_km
from .sensitivity import N_EVENTS_LIMIT, PIVOT_ENERGY_GEV, power_law_sensitivity

__all__ = [
    "COMMON_LOG10_E",
    "DEFAULT_EMIN_GEV",
    "N_COS_THETA",
    "N_EVENTS_LIMIT",
    "N_HOUR_ANGLE",
    "N_RUNG",
    "N_SUB_BAND",
    "PIVOT_ENERGY_GEV",
    "REACH_FRACTIONS",
    "REACH_PIVOT_GEV",
    "REACH_REFERENCE_GEV",
    "RUNG_DECADES",
    "band_averaged_effective_area_cm2",
    "band_statistics",
    "central_energy_range",
    "column_target_volume_km3",
    "derived_band_averaged_effective_area_cm2",
    "derived_directional_effective_area_cm2",
    "directional_effective_area_cm2",
    "fit_light_reach",
    "point_source_sensitivity",
    "polar_band_directions",
    "zenith_band_weights",
]

#: Neutrino energies every curve here is returned on [log10 GeV].
COMMON_LOG10_E = np.linspace(3.0, 8.0, 26)

#: Directions sampled inside each published declination band.
N_SUB_BAND = 5

#: Zenith bands a source's daily sweep is histogrammed into.
N_COS_THETA = 180

#: Hour angles sampled over a sidereal day.
N_HOUR_ANGLE = 192

#: Rungs of the neutral-current regeneration ladder, and the decades it spans.
N_RUNG = 32
RUNG_DECADES = 4.0

#: Bottom of the analysis window [GeV]. This is what decides whether a
#: sensitivity carries declination information: raise it and the Earth-absorbed
#: high-energy end, where the declination dependence lives, is all that is left.
DEFAULT_EMIN_GEV = 1.0e5

#: Energy at which a fitted light reach vanishes [GeV].
REACH_PIVOT_GEV = 10.0**9.67


def directional_effective_area_cm2(
    site: Site,
    cos_theta: np.ndarray,
    threshold_gev: float,
    reach_km: float | None = None,
    pivot_gev: float = REACH_PIVOT_GEV,
    channels: str = "both",
    log10_e: np.ndarray = COMMON_LOG10_E,
    cross_section: CrossSection | None = None,
) -> np.ndarray:
    """Effective area per arrival direction, in the published convention.

    A tabulated effective area is differential in the neutrino energy, so the
    parent is monochromatic: no spectral weighting enters anywhere, and the
    length is the first-passage range rather than ``1 / Phi(A)``. The assembly
    keeps the neutral-current and tau ladders and credits every rung to the
    surface energy. Nothing is averaged over direction, and the first-passage
    integral is cut at whatever column the direction actually supplies.

    Parameters
    ----------
    site : Site
        Detector geometry and medium.
    cos_theta : np.ndarray, shape (n_dir,)
        Cosine of the arrival zenith for each direction; ``+1`` is overhead.
    threshold_gev : float
        Muon selection threshold [GeV].
    reach_km : float or None, optional
        Growth of the light reach per e-fold [km]. ``None``, the default, keeps
        the static instrumented radius and leaves the result parameter-free.
    pivot_gev : float, optional
        Energy at which the reach vanishes [GeV]. Ignored when ``reach_km`` is
        ``None``.
    channels : {"both", "mu"}, optional
        ``"mu"`` keeps only the ``nu_mu`` charged current, which is the flavour
        the DR2 tables were generated from. ``"both"``, the default, adds
        ``nu_tau -> tau -> mu``, which a through-going track cannot distinguish.
    log10_e : np.ndarray, optional
        Neutrino energies [log10 GeV].
    cross_section : CrossSection, optional
        Cross-section model; ``None`` uses
        :func:`softpaws.response.effective_area.default_cross_section`.

    Returns
    -------
    aeff : np.ndarray, shape (log10_e.size, n_dir)
        Effective area [cm^2] at each energy and arrival direction.
    """
    cross_section = default_cross_section() if cross_section is None else cross_section
    cos_theta = np.atleast_1d(np.asarray(cos_theta, dtype=float))
    neutrino_column, muon_column_km = site.columns(cos_theta)
    n_nucleon = nucleon_number_density(site.density_g_cm3)
    energy = 10.0 ** np.asarray(log10_e, dtype=float)

    ladders = [("mu", 1.0 - MEAN_INELASTICITY, 1.0)]
    if channels == "both":
        ladders.append(("tau", MEAN_Z * (1.0 - MEAN_INELASTICITY), BR_TAU_TO_MU))

    total = np.zeros((energy.size, cos_theta.size))
    for flavour, muon_fraction, branching in ladders:
        # One diagonalization serves every column, so the whole set of
        # directions costs the same as a single one.
        rungs = [
            flavour_transmission(
                float(e_nu),
                neutrino_column,
                cross_section,
                flavour=flavour,
                n_grid=N_RUNG,
                decades=RUNG_DECADES,
            )
            for e_nu in energy
        ]
        # The rock correction below the horizon is solved per direction, so it
        # is taken for every rung of every energy at once: the entry energy
        # belongs to the direction, and asking for one parent at a time would
        # walk that solve again for each of them.
        rock_all = two_medium_range_ratio(
            muon_fraction * np.concatenate([rung[0] for rung in rungs]),
            threshold_gev,
            cos_theta,
            site.below_km + 0.5 * site.height_km,
            site.density_g_cm3,
        )
        cut = np.cumsum([0] + [rung[0].size for rung in rungs])
        for i, (rung_energy, rung_weight) in enumerate(rungs):
            muon_energy = muon_fraction * rung_energy
            # (n_rung, n_dir): each rung's muon under each direction's column,
            # shortened below the horizon by the rock beneath the optical medium.
            rock = rock_all[cut[i]:cut[i + 1]]
            length = (
                truncated_muon_range_km(
                    muon_energy[:, None],
                    muon_column_km[None, :],
                    threshold_gev,
                    site.density_g_cm3,
                )
                * rock
            )
            radius = (
                np.full(muon_energy.shape, site.radius_km)
                if reach_km is None
                else light_reach_radius_km(site.radius_km, muon_energy, reach_km, pivot_gev)
            )
            area_km2 = site.projected_area_km2(cos_theta[None, :], radius[:, None])
            volume_km3 = area_km2 * length + site.detector_volume_km3(radius)[:, None]
            rate = (
                n_nucleon * cross_section.cc(rung_energy)[:, None] * volume_km3 * CM_PER_KM**3
            )
            total[i] += branching * np.sum(rung_weight * rate, axis=0)
    return total


#: Effective radius at :data:`REACH_REFERENCE_GEV` that :func:`fit_light_reach`
#: scans, as a fraction of the instrumented footprint radius. The range covers
#: a detector that responds to two thirds of its footprint at 1 PeV and one
#: that already responds past it.
REACH_FRACTIONS = np.linspace(0.6, 1.2, 13)

#: Energy the scanned fraction is quoted at [GeV].
REACH_REFERENCE_GEV = 1.0e6


def fit_light_reach(
    site: Site,
    cos_theta: np.ndarray,
    weights: np.ndarray,
    published_aeff_cm2: np.ndarray,
    threshold_gev: float,
    log10_e: np.ndarray = COMMON_LOG10_E,
    fractions: np.ndarray = REACH_FRACTIONS,
    reference_gev: float = REACH_REFERENCE_GEV,
    pivot_gev: float = REACH_PIVOT_GEV,
    channels: str = "both",
) -> tuple[float, np.ndarray, np.ndarray]:
    """Scan the one instrument number against a published effective area.

    The reach enters as ``Lambda`` per e-fold about a pivot, and the pivot sits
    far above the band any table covers, so a scan in ``Lambda`` is a scan with
    a lever arm of some eight e-folds on it. Two metres per e-fold then move a
    120 m footprint as far as twenty move a 2 km one, and no single grid serves
    both. Scanning the effective radius at ``reference_gev`` instead, in units
    of the footprint the site already has, puts every detector on one grid.

    Parameters
    ----------
    site : Site
        Detector geometry and medium.
    cos_theta : np.ndarray, shape (n_dir,)
        Arrival directions the published average runs over.
    weights : np.ndarray, shape (n_dir,)
        Weight of each direction in that average; sums to one.
    published_aeff_cm2 : np.ndarray, shape (log10_e.size,)
        The published curve on the same energy grid [cm^2]. Nodes that are not
        positive and finite are skipped.
    threshold_gev : float
        Muon selection threshold [GeV].
    log10_e : np.ndarray, optional
        Neutrino energies the comparison runs on [log10 GeV].
    fractions : np.ndarray, optional
        Effective radius at ``reference_gev``, as a fraction of the footprint
        radius. See :data:`REACH_FRACTIONS`.
    reference_gev : float, optional
        Energy the fraction is quoted at [GeV]. See :data:`REACH_REFERENCE_GEV`.
    pivot_gev : float, optional
        Energy at which the reach vanishes [GeV].
    channels : {"both", "mu"}, optional
        Channels summed; see :func:`directional_effective_area_cm2`.

    Returns
    -------
    reach_km : float
        Reach per e-fold [km] of the scanned point that fits best.
    residual_dex : np.ndarray, shape (fractions.size,)
        Root-mean-square of ``log10(published / model)`` at each scanned point,
        infinite where the reach empties the detector.
    models : np.ndarray, shape (fractions.size, log10_e.size)
        The averaged model at each scanned point [cm^2].
    """
    published = np.asarray(published_aeff_cm2, dtype=float)
    usable = np.isfinite(published) & (published > 0.0)
    if not np.any(usable):
        raise ValueError("The published curve has no positive node to fit against.")
    # Below the pivot the reach is negative, so a fraction under one is a
    # positive Lambda: the array responds to less than its own footprint.
    lever = np.log(reference_gev / pivot_gev)
    reaches = site.radius_km * (np.asarray(fractions, dtype=float) - 1.0) / lever

    models = np.empty((reaches.size, np.size(log10_e)))
    residual = np.full(reaches.size, np.inf)
    for i, reach_km in enumerate(reaches):
        models[i] = (
            directional_effective_area_cm2(
                site, cos_theta, threshold_gev, reach_km=float(reach_km),
                pivot_gev=pivot_gev, channels=channels, log10_e=log10_e,
            )
            @ np.asarray(weights, dtype=float)
        )
        if np.all(np.isfinite(models[i][usable]) & (models[i][usable] > 0.0)):
            ratio = np.log10(published[usable] / models[i][usable])
            residual[i] = float(np.sqrt(np.mean(ratio**2)))
    return float(reaches[int(np.argmin(residual))]), residual, models


def polar_band_directions(
    sin_dec_edges: np.ndarray, n_sub: int = N_SUB_BAND
) -> np.ndarray:
    """Arrival directions sampling each published band, seen from the Pole.

    At the South Pole the hour angle drops out of the zenith relation and every
    declination maps to one fixed zenith, ``cos(theta_z) = -sin(dec)``. A source
    in the northern sky is therefore permanently upgoing and one in the southern
    sky permanently downgoing, which is what makes IceCube the clean site for a
    declination-resolved test: no time averaging enters at all.

    Parameters
    ----------
    sin_dec_edges : np.ndarray, shape (n_dec + 1,)
        Published band edges in ``sin(dec)``.
    n_sub : int, optional
        Directions sampled inside each band.

    Returns
    -------
    cos_theta : np.ndarray, shape (n_dec, n_sub)
        Sub-sample directions within each band, uniform in ``sin(dec)``.
    """
    sin_dec_edges = np.asarray(sin_dec_edges, dtype=float)
    lo, hi = sin_dec_edges[:-1], sin_dec_edges[1:]
    fraction = (np.arange(n_sub) + 0.5) / n_sub
    sin_dec = lo[:, None] + (hi - lo)[:, None] * fraction[None, :]
    return -sin_dec


def band_averaged_effective_area_cm2(
    site: Site,
    sin_dec_edges: np.ndarray,
    threshold_gev: float,
    reach_km: float | None = None,
    pivot_gev: float = REACH_PIVOT_GEV,
    log10_e: np.ndarray = COMMON_LOG10_E,
    cross_section: CrossSection | None = None,
    n_sub: int = N_SUB_BAND,
) -> np.ndarray:
    """Model effective area averaged within each published declination band.

    Uniform in ``sin(dec)`` is uniform in solid angle, which is how the
    published band average is built.

    Parameters
    ----------
    site : Site
        Detector geometry and medium; the band construction assumes the Pole.
    sin_dec_edges : np.ndarray, shape (n_dec + 1,)
        Published band edges in ``sin(dec)``.
    threshold_gev : float
        Muon selection threshold [GeV].
    reach_km : float or None, optional
        Growth of the light reach per e-fold [km]; ``None`` stays static.
    pivot_gev : float, optional
        Energy at which the reach vanishes [GeV].
    log10_e : np.ndarray, optional
        Neutrino energies [log10 GeV].
    cross_section : CrossSection, optional
        Cross-section model.
    n_sub : int, optional
        Directions sampled inside each band.

    Returns
    -------
    aeff : np.ndarray, shape (log10_e.size, n_dec)
        Effective area [cm^2], band-averaged uniformly in ``sin(dec)``.
    """
    directions = polar_band_directions(sin_dec_edges, n_sub)
    n_dec, n_sub = directions.shape
    per_direction = directional_effective_area_cm2(
        site,
        directions.ravel(),
        threshold_gev,
        reach_km=reach_km,
        pivot_gev=pivot_gev,
        log10_e=log10_e,
        cross_section=cross_section,
    )
    return per_direction.reshape(-1, n_dec, n_sub).mean(axis=2)


def zenith_band_weights(
    latitude_deg: float,
    dec_deg: np.ndarray,
    n_cos_theta: int = N_COS_THETA,
    n_hour_angle: int = N_HOUR_ANGLE,
) -> tuple[np.ndarray, np.ndarray]:
    """Fraction of a sidereal day each declination spends in each zenith band.

    A source at declination ``delta`` seen from latitude ``phi`` has

    .. math:: \\cos\\theta_z = \\sin\\phi\\,\\sin\\delta
        + \\cos\\phi\\,\\cos\\delta\\,\\cos H,

    with the hour angle ``H`` sweeping uniformly over a sidereal day. Since the
    effective area depends on the arrival direction only through
    ``cos(theta_z)``, binning that sweep is all the geometry a point source
    needs: the direction-averaged area is the band areas contracted with these
    weights, exactly.

    Parameters
    ----------
    latitude_deg : float
        Geographic latitude of the site [deg].
    dec_deg : np.ndarray, shape (n_dec,)
        Source declinations [deg].
    n_cos_theta : int, optional
        Number of zenith bands.
    n_hour_angle : int, optional
        Hour angles sampled over the day.

    Returns
    -------
    cos_theta : np.ndarray, shape (n_cos_theta,)
        Band centres in ``cos(theta_z)``, descending from near ``+1``.
    weights : np.ndarray, shape (n_dec, n_cos_theta)
        Time fraction in each band; each row sums to one.
    """
    phi = np.deg2rad(latitude_deg)
    delta = np.deg2rad(np.atleast_1d(np.asarray(dec_deg, dtype=float)))
    hour = np.linspace(0.0, 2.0 * np.pi, n_hour_angle, endpoint=False)

    cos_theta_z = (
        np.sin(phi) * np.sin(delta)[:, None]
        + np.cos(phi) * np.cos(delta)[:, None] * np.cos(hour)[None, :]
    )
    edges = np.linspace(-1.0, 1.0, n_cos_theta + 1)
    weights = np.vstack([np.histogram(row, bins=edges)[0] for row in cos_theta_z]).astype(float)
    weights /= weights.sum(axis=1, keepdims=True)
    return 0.5 * (edges[:-1] + edges[1:]), weights


def point_source_sensitivity(
    aeff_cm2: np.ndarray,
    livetime_s: float,
    gamma: float,
    emin_gev: float = DEFAULT_EMIN_GEV,
    log10_e: np.ndarray = COMMON_LOG10_E,
    pivot_gev: float = PIVOT_ENERGY_GEV,
    n_events: float = N_EVENTS_LIMIT,
) -> np.ndarray:
    """Flux normalization a background-free search would exclude.

    For ``phi(E) = phi_0 (E / E_piv)^-gamma`` the expected count is linear in
    ``phi_0``, so the limit is one quadrature,

    .. math:: \\phi_0^{\\rm lim} = \\frac{N_{\\rm lim}}
        {T \\int_{E_{\\min}} \\dd E\\, A_{\\rm eff}(E) (E/E_{\\rm piv})^{-\\gamma}}.

    Parameters
    ----------
    aeff_cm2 : np.ndarray, shape (log10_e.size, ...)
        Effective area [cm^2] on the energy grid, energy first.
    livetime_s : float
        Exposure [s].
    gamma : float
        Spectral index of the assumed source.
    emin_gev : float, optional
        Bottom of the analysis window [GeV]; see :data:`DEFAULT_EMIN_GEV`.
    log10_e : np.ndarray, optional
        Neutrino energies [log10 GeV].
    pivot_gev : float, optional
        Pivot energy of the quoted flux [GeV].
    n_events : float, optional
        Events the search excludes; see :data:`N_EVENTS_LIMIT`.

    Returns
    -------
    e2_flux : np.ndarray
        ``E^2 phi`` at ``pivot_gev`` [GeV cm^-2 s^-1].

    See Also
    --------
    softpaws.response.sensitivity.power_law_sensitivity : The same limit for a
        diffuse flux as well, and the implementation this calls.
    """
    return power_law_sensitivity(
        aeff_cm2, livetime_s, gamma, log10_e, emin_gev, pivot_gev, n_events
    )


def central_energy_range(
    aeff_cm2: np.ndarray,
    gamma: float,
    emin_gev: float = DEFAULT_EMIN_GEV,
    log10_e: np.ndarray = COMMON_LOG10_E,
    pivot_gev: float = PIVOT_ENERGY_GEV,
) -> tuple[float, float]:
    """Central 90% energy range of the signal a sensitivity comes from.

    Parameters
    ----------
    aeff_cm2 : np.ndarray, shape (log10_e.size,)
        Effective area [cm^2] on the energy grid.
    gamma : float
        Spectral index of the assumed source.
    emin_gev : float, optional
        Bottom of the analysis window [GeV].
    log10_e : np.ndarray, optional
        Neutrino energies [log10 GeV].
    pivot_gev : float, optional
        Pivot energy of the assumed flux [GeV].

    Returns
    -------
    log10_lo, log10_hi : float
        ``log10(E_nu / GeV)`` bracketing the central 90% of the expected count.
    """
    log10_e = np.asarray(log10_e, dtype=float)
    energy = 10.0**log10_e
    window = energy >= emin_gev
    inside = log10_e[window]
    # Per log-decade, so the quantiles read off the axis the figure uses.
    integrand = (aeff_cm2 * (energy / pivot_gev) ** (-gamma) * energy)[window]
    cumulative = np.concatenate(
        ([0.0], np.cumsum(0.5 * np.diff(inside) * (integrand[1:] + integrand[:-1])))
    )
    cumulative /= cumulative[-1]
    return (
        float(np.interp(0.05, cumulative, inside)),
        float(np.interp(0.95, cumulative, inside)),
    )


# ---------------------------------------------------------------------------
# The derived-optics family: the light reach comes from the published optics
# ---------------------------------------------------------------------------


def column_target_volume_km3(
    site: Site,
    optics: Optics,
    production_gev: float,
    cos_theta: np.ndarray,
    available_km: np.ndarray,
    min_modules: float,
    halo_weight: float = 1.0,
    n_sides: int | None = None,
    n_energy: int = 40,
    threshold_gev: float = DEFAULT_MUON_THRESHOLD_GEV,
) -> np.ndarray:
    """Column target volume for one production energy and every direction [km^3].

    The first-principles construction, direction resolved: the truncated
    first-passage range against the available upstream column, the effective
    footprint evaluated at the energy the muon has where it is seen (averaged
    over the part of the column it can actually have covered), the body eroded
    by the minimum in-detector track and dilated by the light reach, the
    Poisson multiplicity weight, and the halo blended at ``halo_weight``.

    Parameters
    ----------
    site : Site
        Detector geometry and medium.
    optics : Optics
        The same detector's optical medium and module.
    production_gev : float
        Muon energy at production [GeV].
    cos_theta : np.ndarray
        Cosine of the arrival zenith.
    available_km : np.ndarray
        Upstream column available in each direction [km of detector medium].
    min_modules : float
        Modules that must register a coincident hit.
    halo_weight : float, optional
        Fraction of the reach-dilated halo the selection accepts.
    n_sides : int or None, optional
        Cross-section of the instrumented body; ``None`` is a cylinder.
    n_energy : int, optional
        Points in the arrival-energy quadrature.
    threshold_gev : float, optional
        Nominal muon threshold the range runs to [GeV].

    Returns
    -------
    volume : np.ndarray
        Target volume [km^3], one entry per direction.
    """
    if production_gev <= threshold_gev:
        return np.zeros_like(cos_theta)
    truncated = np.clip(
        np.atleast_1d(
            truncated_muon_range_km(
                production_gev, available_km, threshold_gev, site.density_g_cm3
            )
        ),
        0.0,
        None,
    )
    # Below the bedrock or the sea floor the muon is in rock, which shortens
    # every upgoing column; downgoing directions are the optical medium alone.
    ratio = rock_range_ratio(
        production_gev, threshold_gev, cos_theta, optics, site.height_km, site.density_g_cm3
    )
    truncated = truncated * ratio
    profile = column_profile(production_gev, threshold_gev, n_energy, site.density_g_cm3)
    if profile is None:
        return np.zeros_like(cos_theta)
    column, energy, _ = profile

    radius, height, weight = effective_body_km(
        site.radius_km, site.height_km, energy, optics, min_modules, n_sides
    )
    area_d, vol_d = eroded_prism_target_km2(
        cos_theta[None, :],
        radius[:, None],
        height[:, None],
        optics.min_track_km,
        n_sides,
        site.n_blocks,
    )
    area_0, vol_0 = eroded_prism_target_km2(
        cos_theta, site.radius_km, site.height_km, optics.min_track_km, n_sides, site.n_blocks
    )
    area = weight[:, None] * ((1.0 - halo_weight) * area_0[None, :] + halo_weight * area_d)
    vol = (1.0 - halo_weight) * vol_0 + halo_weight * vol_d
    clipped = np.minimum(column[:, None] * ratio[None, :], available_km[None, :])
    span = clipped[-1]
    mean_area = np.where(
        span > 0.0,
        np.trapezoid(area, clipped, axis=0) / np.where(span > 0.0, span, 1.0),
        weight[0] * area_0,
    )
    return mean_area * truncated + weight[0] * vol[0]


def derived_directional_effective_area_cm2(
    site: Site,
    optics: Optics,
    cos_theta: np.ndarray,
    min_modules: float,
    flavours: tuple[str, ...],
    cross_section: CrossSection,
    inelasticity_nc: float | None = None,
    halo_weight: float = 1.0,
    efficiency: float = 1.0,
    log10_e: np.ndarray = COMMON_LOG10_E,
) -> np.ndarray:
    """Effective area per arrival direction, with the derived light reach [cm^2].

    The builder of :func:`directional_effective_area_cm2` with the
    first-principles machinery substituted in: the body dilated by the derived
    light reach and eroded by the minimum in-detector track, the Poisson
    multiplicity weight carrying the threshold turn-on, and the range running
    to the nominal threshold so the dimming enters once.

    Parameters
    ----------
    site : Site
        Detector geometry, medium and latitude.
    optics : Optics
        The same detector's optical medium and optical module.
    cos_theta : np.ndarray, shape (n_dir,)
        Cosine of the arrival zenith; ``+1`` is overhead.
    min_modules : float
        Modules that must register a coincident hit.
    flavours : tuple of str
        Parent channels to sum, ``"mu"`` and optionally ``"tau"``.
    cross_section : CrossSection
        Cross section of the incident species, used for the interaction and for
        the Earth absorption alike.
    inelasticity_nc : float, optional
        Mean neutral-current inelasticity. ``None`` keeps the library default.
    halo_weight : float, optional
        Fraction of the reach-dilated halo the selection accepts. The
        acceptance is blended as ``(1 - w) A_static + w A_dilated``, so 1 (the
        default) counts every track the light condition admits and 0 keeps the
        instrumented body only.
    efficiency : float, optional
        Flat selection efficiency applied to the whole curve.
    log10_e : np.ndarray, optional
        Neutrino energies [log10 GeV].

    Returns
    -------
    aeff : np.ndarray, shape (log10_e.size, n_dir)
        Effective area [cm^2].
    """
    cos_theta = np.atleast_1d(np.asarray(cos_theta, dtype=float))
    neutrino_column, muon_column_km = site.columns(cos_theta)
    n_nucleon = nucleon_number_density(site.density_g_cm3)
    energy = 10.0 ** np.asarray(log10_e, dtype=float)
    n_sides = site.n_sides if site.shape == "prism" else None

    total = np.zeros((energy.size, cos_theta.size))
    for flavour in flavours:
        for i, e_nu in enumerate(energy):
            if flavour == "mu":
                extra = {} if inelasticity_nc is None else {"mean_inelasticity": inelasticity_nc}
                rung_energy, rung_weight = regenerated_transmission(
                    float(e_nu), neutrino_column, cross_section, **extra
                )
                branching = 1.0
                muon_gev = (1.0 - mean_inelasticity(rung_energy)) * rung_energy
            else:
                extra = (
                    {} if inelasticity_nc is None else {"mean_inelasticity_nc": inelasticity_nc}
                )
                rung_energy, rung_weight = flavour_transmission(
                    float(e_nu),
                    neutrino_column,
                    cross_section,
                    flavour="tau",
                    n_grid=N_RUNG,
                    decades=RUNG_DECADES,
                    **extra,
                )
                branching = BR_TAU_TO_MU
                muon_gev = MEAN_Z * (1.0 - mean_inelasticity(rung_energy)) * rung_energy
            rate = np.zeros((rung_energy.size, cos_theta.size))
            for k, e_mu in enumerate(muon_gev):
                rate[k] = column_target_volume_km3(
                    site,
                    optics,
                    float(e_mu),
                    cos_theta,
                    muon_column_km,
                    min_modules,
                    halo_weight,
                    n_sides,
                )
            rate *= (
                n_nucleon * cross_section.cc(rung_energy)[:, None] * branching * CM_PER_KM**3
            )
            total[i] += np.sum(rung_weight * rate, axis=0)
    return efficiency * total


def derived_band_averaged_effective_area_cm2(
    site: Site,
    optics: Optics,
    sin_dec_edges: np.ndarray,
    min_modules: float,
    flavours: tuple[str, ...],
    species,
    inelasticity_nc: float | None = None,
    halo_weight: float = 1.0,
    efficiency: float = 1.0,
    log10_e: np.ndarray = COMMON_LOG10_E,
    n_sub: int = N_SUB_BAND,
) -> np.ndarray:
    """Model effective area per published band, averaged over species [cm^2].

    Parameters
    ----------
    site : Site
        Detector geometry, medium and latitude.
    optics : Optics
        The same detector's optical medium and optical module.
    sin_dec_edges : np.ndarray, shape (n_dec + 1,)
        Published band edges in ``sin(dec)``.
    min_modules : float
        Modules that must register a coincident hit.
    flavours : tuple of str
        Parent channels to sum.
    species : sequence of CrossSection
        Incident species averaged over, as in
        :data:`softpaws.response.first_principles.SPECIES`.
    inelasticity_nc : float, optional
        Mean neutral-current inelasticity.
    halo_weight : float, optional
        Fraction of the reach-dilated halo the selection accepts.
    efficiency : float, optional
        Flat selection efficiency.
    log10_e : np.ndarray, optional
        Neutrino energies [log10 GeV].
    n_sub : int, optional
        Directions sampled inside each band.

    Returns
    -------
    aeff : np.ndarray, shape (log10_e.size, n_dec)
        Effective area [cm^2], band-averaged uniformly in ``sin(dec)``.
    """
    directions = polar_band_directions(sin_dec_edges, n_sub)
    n_dec, n_sub = directions.shape
    per_species = [
        derived_directional_effective_area_cm2(
            site,
            optics,
            directions.ravel(),
            min_modules,
            flavours,
            xsec,
            inelasticity_nc,
            halo_weight,
            efficiency,
            log10_e,
        )
        for xsec in species
    ]
    averaged = np.mean(per_species, axis=0)
    return averaged.reshape(-1, n_dec, n_sub).mean(axis=2)


def band_statistics(
    log10_e: np.ndarray, published: np.ndarray, model: np.ndarray, band: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Level and tilt of ``published / model`` within each declination band.

    The level is the geometric mean of the ratio over the scored band and the
    tilt is the slope of its base-ten logarithm against ``log10 E``, so a tilt
    of zero means the model has the right energy dependence in that band
    whatever its normalization.

    Parameters
    ----------
    log10_e : np.ndarray
        ``log10(E_nu / GeV)`` grid.
    published, model : np.ndarray, shape (n_energy, n_dec)
        Effective areas [cm^2].
    band : np.ndarray
        Boolean mask of the scored energies.

    Returns
    -------
    level : np.ndarray, shape (n_dec,)
        Geometric-mean ratio.
    tilt : np.ndarray, shape (n_dec,)
        Slope of ``log10(published / model)`` [dex per decade of energy].
    scatter : np.ndarray, shape (n_dec,)
        Root-mean-square about the fitted line [dex].
    """
    x = np.asarray(log10_e, dtype=float)[band]
    level = np.full(published.shape[1], np.nan)
    tilt = np.full(published.shape[1], np.nan)
    scatter = np.full(published.shape[1], np.nan)
    for j in range(published.shape[1]):
        y = published[band, j] / model[band, j]
        ok = np.isfinite(y) & (y > 0.0)
        if ok.sum() < 3:
            continue
        logy = np.log10(y[ok])
        slope, intercept = np.polyfit(x[ok], logy, 1)
        level[j] = 10.0 ** np.mean(logy)
        tilt[j] = slope
        scatter[j] = float(np.std(logy - (slope * x[ok] + intercept)))
    return level, tilt, scatter
