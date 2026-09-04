"""Effective areas with the light reach derived and nothing fitted.

The column integral of a track detector's entering term is evaluated at the
energy the muon has where it is seen, not at the energy it was born with,
and the instrumented body it enters is dilated by the light reach of
:mod:`softpaws.response.light_reach` at that energy. Both terms of the can,
the extrusion over the column and the instrumented volume, are carried
together, so a reach that grows the body grows both, and a muon born below
threshold contributes neither.

Two detectors are assembled: IceCube, averaged over the upgoing sky with
layered-PREM columns (:func:`ic_effective_area_cm2`), and KM3NeT/ARCA230,
averaged over the whole sky with the finite sea-water overburden above it
(:func:`arca_effective_area_cm2`). Each sums the ``nu_mu`` channel and the
``nu_tau -> tau -> mu`` channel (:data:`DEFAULT_FLAVOURS`) and averages
neutrino and antineutrino, each propagated through the Earth with its own
cross section pinned to the CSMS isoscalar values
(:class:`IsoscalarCrossSection`, :data:`SPECIES`). :func:`build_model` is the
whole model for one detector; :func:`fit_reach` floats a flat attenuation
length against a published curve so that the fitted value is comparable with
what is known about the medium.

Geometry comes from :mod:`softpaws.detectors`, the Earth columns from
:mod:`softpaws.transport.earth`, and the published energy grids from the
effective-area engine.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import minimize_scalar

from softpaws.detectors import ARCA230, ICECUBE, MAX_UPSTREAM_KM, Optics, Site
from softpaws.response.effective_area import (
    ARCA_LOG10_E,
    IC_LOG10_E,
    N_ZENITH,
    ic_upgoing_columns,
)
from softpaws.response.light_reach import DEFAULT_MIN_MODULES, effective_body_km
from softpaws.transport.attenuation import flavour_transmission, regenerated_transmission
from softpaws.transport.cross_section import CrossSection, bgr18_cross_section
from softpaws.transport.earth import neutrino_column_g_cm2, overburden_km, zenith_grid
from softpaws.transport.soft_volume import (
    DEFAULT_MUON_THRESHOLD_GEV,
    eroded_prism_target_km2,
    stochastic_muon_range_km,
    truncated_muon_range_km,
    two_medium_range_ratio,
)
from softpaws.transport.source import mean_inelasticity, nucleon_number_density
from softpaws.transport.tau import BR_TAU_TO_MU, MEAN_Z
from softpaws.utils.constants import CM_PER_KM, M_PER_KM, RHO_WATER_G_CM3

__all__ = [
    "ANCHORS",
    "ARCA_BAND",
    "ARCA_LOG10_E",
    "CROSS_SECTION",
    "CSMS_LOG10_E",
    "CSMS_PB",
    "DEFAULT_FLAVOURS",
    "FAR_MEDIUM_SOURCE",
    "IC_BAND",
    "IC_LOG10_E",
    "IsoscalarCrossSection",
    "SPECIES",
    "arca_column_volume_km3",
    "arca_effective_area_cm2",
    "build_model",
    "column_profile",
    "detector_curves",
    "fit_reach",
    "ic_column_volume_km3",
    "ic_effective_area_cm2",
    "residuals",
    "rock_range_ratio",
]

#: Band the models are compared over at each site, ``log10(E_nu / GeV)``.
IC_BAND = (5.0, 7.8)
ARCA_BAND = (4.0, 7.5)

#: Energies the ratio is tabulated at, ``log10(E_nu / GeV)``.
ANCHORS = (5.0, 6.0, 7.0)

#: Channels summed. The DR2 readme calls the released table an average over
#: "simulated muon neutrino events", which reads as a ``nu_mu`` response, but
#: the table's own deepest declination bands rule that out: at
#: ``sin(dec) = 0.98`` it stays flat at 2-3e5 cm^2 from 10^5.6 to 10^8.6, and
#: a ``nu_mu``-only model falls to 8.9e1 cm^2 there. A 10^8 GeV neutrino
#: crossing 1.02e10 g/cm^2 must degrade to ~10^5 GeV to escape, which takes
#: ~24 successive scatters, and each interaction is neutral current only 30%
#: of the time. For ``nu_tau`` the charged current is not terminal, since the
#: tau decays back to a ``nu_tau`` at ~0.4 of the energy, so that cascade
#: alone survives the full Earth diameter.
DEFAULT_FLAVOURS = ("mu", "tau")

#: Medium below the optical one, entered through the two-medium first passage
#: of :func:`softpaws.transport.soft_volume.two_medium_range_ratio` for every
#: upgoing direction: PROPOSAL's standard rock, whose ``Phi'(0)`` per unit
#: column sits 27% above water at 1 PeV. An upgoing muon is born below the
#: bedrock or the sea floor and crosses only the near column of ice or water
#: before it is seen, so most of its range is in this medium. ``None`` keeps
#: the water kernel for the whole range; the difference is a factor 0.80 to
#: 0.86 on the entering term over the upgoing sky beyond ~6 degrees of the
#: horizon, and none inside it.
FAR_MEDIUM_SOURCE = "proposal_rock"

#: CSMS isoscalar cross sections [pb] (Cooper-Sarkar, Mertsch and Sarkar,
#: arXiv:1106.3723, Tables 1 and 2), on ``CSMS_LOG10_E``. The shipped BGR18
#: tables are ``nu-p`` used as per-nucleon, which under-counts an isoscalar
#: target by 17% at 10^5 GeV falling to ~11% at 10^7, the valence-quark
#: difference between protons and neutrons. :class:`IsoscalarCrossSection`
#: pins each channel to these values.
CSMS_LOG10_E = np.array([4.0, np.log10(5.0e4), 5.0, np.log10(2.0e5),
                         np.log10(5.0e5), 6.0, np.log10(2.0e6), np.log10(5.0e6),
                         7.0, np.log10(2.0e7), np.log10(5.0e7), 8.0])
CSMS_PB = {
    "nu": {"cc": (47.0, 140.0, 210.0, 310.0, 490.0, 690.0, 950.0, 1400.0,
                  1900.0, 2600.0, 3700.0, 4800.0),
           "nc": (15.0, 49.0, 75.0, 110.0, 180.0, 260.0, 360.0, 540.0,
                  730.0, 980.0, 1400.0, 1900.0)},
    "nubar": {"cc": (31.0, 110.0, 180.0, 270.0, 460.0, 660.0, 920.0, 1400.0,
                     1900.0, 2500.0, 3700.0, 4800.0),
              "nc": (11.0, 39.0, 64.0, 99.0, 170.0, 240.0, 350.0, 530.0,
                     730.0, 980.0, 1400.0, 1900.0)},
}


class IsoscalarCrossSection(CrossSection):
    """A shipped ``nu-p`` table pinned to the CSMS isoscalar values.

    Scales the base model's charged- and neutral-current channels by the
    ratio of the CSMS isoscalar value to the base value at the CSMS energies,
    interpolated in ``log E`` and held at the ends. Both channels scale, so
    the Earth attenuation the total sets stays consistent with the
    interaction rate.

    Parameters
    ----------
    base : softpaws.transport.cross_section.CrossSection
        The shipped table to correct.
    species : {"nu", "nubar"}
        Which CSMS column to pin to.

    Notes
    -----
    ``local_slope`` delegates to the base model: the correction drifts by
    ~0.2 in ``ln sigma`` over nine e-folds of energy, a slope shift of ~0.02,
    below the smoothing spline's own uncertainty.
    """

    def __init__(self, base: CrossSection, species: str):
        """Wrap a base model and pin it to the reference values of one species.

        Parameters
        ----------
        base : CrossSection
            Model the correction is applied to.
        species : str
            Key of :data:`CSMS_PB`, the species to pin against.
        """
        self._base = base
        self._log_ratio = {}
        for channel in ("cc", "nc"):
            target = np.asarray(CSMS_PB[species][channel], dtype=float) * 1.0e-36
            ours = np.array([
                float(np.atleast_1d(getattr(base, channel)(10.0**log_e))[0])
                for log_e in CSMS_LOG10_E
            ])
            self._log_ratio[channel] = np.log(target / ours)

    def _scale(self, channel: str, energy_gev) -> np.ndarray:
        """Ratio of the CSMS value to the base table, interpolated in ``log E``."""
        log_e = np.log10(np.asarray(energy_gev, dtype=float))
        return np.exp(np.interp(log_e, CSMS_LOG10_E, self._log_ratio[channel]))

    def cc(self, energy_gev):
        """Charged-current cross section [cm^2], pinned to CSMS.

        Parameters
        ----------
        energy_gev : float or np.ndarray
            Neutrino energy [GeV].

        Returns
        -------
        sigma : np.ndarray
            Cross section per nucleon [cm^2].
        """
        return self._base.cc(energy_gev) * self._scale("cc", energy_gev)

    def nc(self, energy_gev):
        """Neutral-current cross section [cm^2], pinned to CSMS.

        Parameters
        ----------
        energy_gev : float or np.ndarray
            Neutrino energy [GeV].

        Returns
        -------
        sigma : np.ndarray
            Cross section per nucleon [cm^2].
        """
        return self._base.nc(energy_gev) * self._scale("nc", energy_gev)

    def local_slope(self, energy_gev):
        """Local slope of the base model; see the class notes.

        Parameters
        ----------
        energy_gev : float or np.ndarray
            Neutrino energy [GeV].

        Returns
        -------
        slope : np.ndarray
            ``d ln sigma / d ln E`` of the base table.
        """
        return self._base.local_slope(energy_gev)


#: The shipped neutrino table, the default cross section of every builder.
CROSS_SECTION = bgr18_cross_section()

#: Both published tables are neutrino/antineutrino averages: KM3NeT writes
#: ``A_eff(nu_i + nubar_i) / 2`` and the DR2 companion paper (arXiv:2605.19040)
#: states the area is "averaged assuming an equal number of neutrinos and
#: antineutrinos". The comparand therefore needs both species, each carrying
#: its own Earth absorption, and each pinned to its own CSMS isoscalar column.
SPECIES = (IsoscalarCrossSection(CROSS_SECTION, "nu"),
           IsoscalarCrossSection(bgr18_cross_section("BGR18_nubar"), "nubar"))


# ---------------------------------------------------------------------------
# The column a muon has covered, and the medium it covered it in
# ---------------------------------------------------------------------------


def column_profile(
    production_gev: float, threshold_gev: float, n_energy: int,
    density_g_cm3: float = RHO_WATER_G_CM3,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Column travelled against the energy the muon has there.

    The column a muon has covered by the time it has fallen to ``E`` is
    ``L(eps -> E_thr) - L(E -> E_thr)``, exact for the mean first-passage
    depth by the tower property, so the validated range function is called
    in its normal convention throughout.

    Parameters
    ----------
    production_gev : float
        Muon energy at production [GeV].
    threshold_gev : float
        Muon selection threshold [GeV].
    n_energy : int
        Points in the quadrature.
    density_g_cm3 : float, optional
        Density of the detector medium [g cm^-3].

    Returns
    -------
    profile : tuple or None
        ``(column [km], arrival energy [GeV], total range [km])``, or ``None``
        when the muon is born below threshold.
    """
    if production_gev <= threshold_gev:
        return None
    total = float(np.atleast_1d(
        stochastic_muon_range_km(production_gev, threshold_gev, density_g_cm3))[0])
    if not np.isfinite(total) or total <= 0.0:
        return None
    energy = np.logspace(np.log10(production_gev), np.log10(threshold_gev), n_energy)
    column = total - stochastic_muon_range_km(energy, threshold_gev, density_g_cm3)
    return column, energy, total


def rock_range_ratio(
    production_gev: float, threshold_gev: float, cos_theta: np.ndarray, site: Optics,
    height_km: float, density_g_cm3: float, far_source: str | None = FAR_MEDIUM_SOURCE,
) -> np.ndarray:
    """Two-medium range over the single-medium one, per arrival direction.

    The near column is the optical medium between the far medium and the
    centre of the instrumented body, ``(headroom below + h/2) / |cos theta|``,
    in the same geometric units as every other length here, so the ratio
    multiplies the column-depth lengths of the entering term directly. It is
    1 above the horizon, where the overburden is the optical medium
    throughout, and 1 everywhere when ``far_source`` is ``None``.

    Parameters
    ----------
    production_gev : float
        Muon energy at production [GeV].
    threshold_gev : float
        Muon selection threshold [GeV].
    cos_theta : np.ndarray
        Cosine of the arrival zenith; ``+1`` is overhead.
    site : Optics
        Detector optics, for the headroom below the instrumented volume.
    height_km : float
        Instrumented height [km].
    density_g_cm3 : float
        Density of the optical medium [g cm^-3], the unit the lengths are in.
    far_source : str or None, optional
        Loss table of the medium below the optical one; see
        :data:`FAR_MEDIUM_SOURCE`.

    Returns
    -------
    ratio : np.ndarray
        Range ratio, one entry per direction, in ``(0, 1]``.
    """
    vertical_km = site.headroom_below_m / M_PER_KM + 0.5 * height_km
    return two_medium_range_ratio(
        production_gev, threshold_gev, cos_theta, vertical_km, density_g_cm3,
        far_source=far_source)


def _muon_rungs(
    e_nu: float, flavour: str, columns: np.ndarray, xsec: CrossSection,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Energies a neutrino arrives with, their weights, and the muons they make.

    Parameters
    ----------
    e_nu : float
        Neutrino energy at the surface [GeV].
    flavour : {"mu", "tau"}
        Parent channel.
    columns : np.ndarray
        Earth columns of the directions averaged over [g cm^-2].
    xsec : CrossSection
        Cross section of the incident species.

    Returns
    -------
    rungs : np.ndarray
        Neutrino energies at the detector [GeV].
    rung_weight : np.ndarray, shape (rungs, directions)
        Transmitted flux at each rung and direction.
    muon_gev : np.ndarray
        Muon energy each rung produces [GeV].
    branching : float
        Branching fraction into a muon: 1 for ``nu_mu``,
        :data:`~softpaws.transport.tau.BR_TAU_TO_MU` for ``nu_tau``.
    """
    if flavour == "mu":
        rungs, rung_weight = regenerated_transmission(float(e_nu), columns, xsec)
        return rungs, rung_weight, (1.0 - mean_inelasticity(rungs)) * rungs, 1.0
    rungs, rung_weight = flavour_transmission(float(e_nu), columns, xsec, flavour="tau")
    return rungs, rung_weight, MEAN_Z * (1.0 - mean_inelasticity(rungs)) * rungs, BR_TAU_TO_MU


# ---------------------------------------------------------------------------
# IceCube
# ---------------------------------------------------------------------------


def ic_column_volume_km3(
    production_gev: float, threshold_gev: float, cos_theta: np.ndarray,
    site: Optics, min_modules: float | None, n_energy: int, geometry: Site = ICECUBE,
) -> np.ndarray:
    """Target volume for one production energy [km^3].

    The entering term is the silhouette extruded upstream over the column the
    muon can cover, and the instrumented volume is the same extrusion
    continued to the back face, so the two add to the volume of the body
    extruded by ``L``. Both are carried here, since a light reach dilates the
    body and so moves both of them; returning only the column term and adding
    a fixed ``V_det`` outside would grow one and hold the other. A muon born
    below threshold gets neither.

    Parameters
    ----------
    production_gev : float
        Muon energy at production [GeV].
    threshold_gev : float
        Muon selection threshold [GeV].
    cos_theta : np.ndarray
        Arrival directions, ``|cos theta_z|`` of an upgoing hemisphere.
    site : Optics
        Detector optics.
    min_modules : float or None
        Modules that must fire. ``None`` holds the body at the instrumented one.
    n_energy : int
        Points in the arrival-energy quadrature.
    geometry : Site, optional
        Instrumented prism. Defaults to :data:`~softpaws.detectors.ICECUBE`.

    Returns
    -------
    volume : np.ndarray
        Target volume [km^3], one entry per direction.
    """
    zeros = np.zeros_like(np.asarray(cos_theta, dtype=float))
    density = geometry.density_g_cm3
    profile = column_profile(production_gev, threshold_gev, n_energy, density)
    if profile is None:
        return zeros
    column, energy, total = profile
    # Below the ice the muon is in rock, which shortens every upgoing column.
    # ``ic_upgoing_columns`` hands back ``|cos theta_z| = sin(dec)`` for a
    # hemisphere that is upgoing by construction, so the sign is restored here.
    ratio = rock_range_ratio(
        production_gev, threshold_gev, -np.abs(np.asarray(cos_theta, dtype=float)),
        site, geometry.height_km, density)

    if min_modules is None:
        area, volume = eroded_prism_target_km2(
            cos_theta, geometry.radius_km, geometry.height_km, site.min_track_km,
            geometry.n_sides)
        return area * total * ratio + volume

    radius, height, weight = effective_body_km(
        geometry.radius_km, geometry.height_km, energy, site, min_modules,
        geometry.n_sides)
    area, volume = eroded_prism_target_km2(
        np.asarray(cos_theta, dtype=float)[None, :], radius[:, None],
        height[:, None], site.min_track_km, geometry.n_sides,
    )
    # The instrumented term belongs to a vertex inside the body, which the muon
    # leaves at essentially its production energy, so it is taken at ``energy[0]``.
    return (np.trapezoid(weight[:, None] * area, column[:, None] * ratio[None, :], axis=0)
            + weight[0] * volume[0])


def ic_effective_area_cm2(
    site: Optics, threshold_gev: float, min_modules: float | None, n_energy: int,
    flavours: tuple[str, ...] = DEFAULT_FLAVOURS, cross_section: CrossSection | None = None,
    log10_e: np.ndarray = IC_LOG10_E, geometry: Site = ICECUBE,
) -> np.ndarray:
    """Upgoing-averaged IceCube effective area, both channels [cm^2].

    Parameters
    ----------
    site : Optics
        Detector optics.
    threshold_gev : float
        Muon selection threshold [GeV].
    min_modules : float or None
        Modules that must fire, or ``None`` for the instrumented footprint.
    n_energy : int
        Points in the arrival-energy quadrature.
    flavours : tuple of str, optional
        Parent channels to sum. Defaults to :data:`DEFAULT_FLAVOURS`.
    cross_section : softpaws.transport.cross_section.CrossSection, optional
        Cross section of the incident species, used for both the interaction
        and the Earth absorption. Defaults to :data:`CROSS_SECTION`.
    log10_e : np.ndarray, optional
        Neutrino energies to evaluate at, ``log10(E_nu / GeV)``. Defaults to
        :data:`IC_LOG10_E`.
    geometry : Site, optional
        Instrumented prism. Defaults to :data:`~softpaws.detectors.ICECUBE`.

    Returns
    -------
    aeff : np.ndarray
        Effective area [cm^2] on ``log10_e``.
    """
    xsec = CROSS_SECTION if cross_section is None else cross_section
    columns, weights, cos_theta = ic_upgoing_columns()
    # South Pole ice, not the module-level water default. `n L` is exactly
    # density-invariant so the entering term does not care, but `V_det` is a
    # geometric volume and scales with it.
    n_nucleon = nucleon_number_density(geometry.density_g_cm3)
    log10_e = np.asarray(log10_e, dtype=float)
    out = np.zeros(log10_e.size)

    for i, e_nu in enumerate(10.0**log10_e):
        for flavour in flavours:
            rungs, rung_weight, muon_gev, branching = _muon_rungs(e_nu, flavour, columns, xsec)
            rate = np.zeros((rungs.size, cos_theta.size))
            for k, e_mu in enumerate(muon_gev):
                volume = ic_column_volume_km3(
                    float(e_mu), threshold_gev, cos_theta, site, min_modules, n_energy,
                    geometry,
                )
                rate[k] = volume * CM_PER_KM**3
            rate *= n_nucleon * xsec.cc(rungs)[:, None] * branching
            out[i] += np.average((rung_weight * rate).sum(axis=0), weights=weights)
    return out


# ---------------------------------------------------------------------------
# KM3NeT/ARCA230
# ---------------------------------------------------------------------------


def arca_column_volume_km3(
    production_gev: float, threshold_gev: float, theta_deg: np.ndarray,
    available_km: np.ndarray, site: Optics, min_modules: float | None, n_energy: int,
    geometry: Site = ARCA230,
) -> np.ndarray:
    """Column target volume at ARCA230, truncated at the available column [km^3].

    The site supplies a finite upstream column, so the length is the
    truncated first-passage range ``E[tau ^ X]``. The reach enters as the
    mean projected area over the part of the column the muon can actually
    have travelled, which keeps the validated range intact and reduces to the
    plain column integral where the column is unlimited. As at IceCube, the
    instrumented volume is carried here and not added outside, so that the
    reach dilates both terms of the can and a sub-threshold muon contributes
    neither.

    Parameters
    ----------
    production_gev : float
        Muon energy at production [GeV].
    threshold_gev : float
        Muon selection threshold [GeV].
    theta_deg : np.ndarray
        Zenith samples [deg].
    available_km : np.ndarray
        Upstream sea-water column available in each direction [km].
    site : Optics
        Detector optics.
    min_modules : float or None
        Modules that must fire, or ``None`` for the instrumented footprint.
    n_energy : int
        Points in the arrival-energy quadrature.
    geometry : Site, optional
        Instrumented cylinders. Defaults to :data:`~softpaws.detectors.ARCA230`.

    Returns
    -------
    volume : np.ndarray
        Column volume [km^3], one entry per zenith.
    """
    radius_km = geometry.radius_km
    height_km = geometry.height_km
    n_blocks = geometry.n_blocks
    density = geometry.density_g_cm3
    cos_theta = np.cos(np.deg2rad(np.asarray(theta_deg, dtype=float)))
    if production_gev <= threshold_gev:
        return np.zeros_like(theta_deg)

    truncated = np.atleast_1d(truncated_muon_range_km(
        production_gev, available_km, threshold_gev, kernel_evaluation="running"))
    truncated = np.clip(truncated, 0.0, None)
    # Below the sea floor the muon is in rock, which shortens every upgoing
    # column; the overburden above is water throughout and is untouched.
    ratio = rock_range_ratio(production_gev, threshold_gev, cos_theta, site,
                             height_km, density)
    truncated = truncated * ratio

    if min_modules is None:
        area, volume = eroded_prism_target_km2(
            cos_theta, radius_km, height_km, site.min_track_km, None, n_blocks)
        return area * truncated + volume

    profile = column_profile(production_gev, threshold_gev, n_energy, density)
    if profile is None:
        return np.zeros_like(theta_deg)
    column, energy, _ = profile

    radius, height, weight = effective_body_km(radius_km, height_km, energy, site,
                                               min_modules, None)
    area, volume = eroded_prism_target_km2(
        cos_theta[None, :], radius[:, None], height[:, None], site.min_track_km,
        None, n_blocks)
    area = weight[:, None] * area
    clipped = np.minimum(column[:, None] * ratio[None, :], available_km[None, :])
    span = clipped[-1]
    mean_area = np.where(
        span > 0.0, np.trapezoid(area, clipped, axis=0) / np.where(span > 0.0, span, 1.0),
        weight[0] * eroded_prism_target_km2(cos_theta, radius_km, height_km,
                                            site.min_track_km, None, n_blocks)[0],
    )
    return mean_area * truncated + weight[0] * volume[0]


def arca_effective_area_cm2(
    site: Optics, threshold_gev: float, min_modules: float | None, n_energy: int,
    flavours: tuple[str, ...] = DEFAULT_FLAVOURS, cross_section: CrossSection | None = None,
    log10_e: np.ndarray = ARCA_LOG10_E, geometry: Site = ARCA230, n_zenith: int = N_ZENITH,
) -> np.ndarray:
    """Sky-averaged ARCA230 effective area, both channels [cm^2].

    Parameters
    ----------
    site : Optics
        Detector optics.
    threshold_gev : float
        Muon selection threshold [GeV].
    min_modules : float or None
        Modules that must fire, or ``None`` for the instrumented footprint.
    n_energy : int
        Points in the arrival-energy quadrature.
    flavours : tuple of str, optional
        Parent channels to sum. Defaults to :data:`DEFAULT_FLAVOURS`.
    cross_section : softpaws.transport.cross_section.CrossSection, optional
        Cross section of the incident species, used for both the interaction
        and the Earth absorption. Defaults to :data:`CROSS_SECTION`.
    log10_e : np.ndarray, optional
        Neutrino energies to evaluate at, ``log10(E_nu / GeV)``. Defaults to
        :data:`ARCA_LOG10_E`.
    geometry : Site, optional
        Instrumented cylinders. Defaults to :data:`~softpaws.detectors.ARCA230`.
    n_zenith : int, optional
        Equal-solid-angle zenith slices of the sky average.

    Returns
    -------
    aeff : np.ndarray
        Effective area [cm^2] on ``log10_e``.
    """
    xsec = CROSS_SECTION if cross_section is None else cross_section
    theta_deg, weights = zenith_grid(n_zenith)
    cos_theta = np.cos(np.deg2rad(theta_deg))
    density = geometry.density_g_cm3
    columns = neutrino_column_g_cm2(cos_theta, geometry.depth_km, density, MAX_UPSTREAM_KM)
    available_km = overburden_km(cos_theta, geometry.depth_km, MAX_UPSTREAM_KM)
    n_nucleon = nucleon_number_density(density)
    log10_e = np.asarray(log10_e, dtype=float)
    out = np.zeros(log10_e.size)

    for i, e_nu in enumerate(10.0**log10_e):
        for flavour in flavours:
            rungs, rung_weight, muon_gev, branching = _muon_rungs(e_nu, flavour, columns, xsec)
            rate = np.zeros((rungs.size, theta_deg.size))
            for k, e_mu in enumerate(muon_gev):
                volume = arca_column_volume_km3(
                    float(e_mu), threshold_gev, theta_deg, available_km, site,
                    min_modules, n_energy, geometry,
                )
                rate[k] = volume * CM_PER_KM**3
            rate *= n_nucleon * xsec.cc(rungs)[:, None] * branching
            out[i] += np.average((rung_weight * rate).sum(axis=0), weights=weights)
    return out


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_model(
    which: str, site: Optics, min_modules: float = DEFAULT_MIN_MODULES, n_energy: int = 40,
    flavours: tuple[str, ...] = DEFAULT_FLAVOURS,
    species: tuple[CrossSection, ...] = SPECIES,
    **builder_kwargs,
) -> np.ndarray:
    """The full model for one detector, threshold and reach both derived [cm^2].

    Averaged over neutrino and antineutrino, each propagated through the
    Earth with its own cross section, since both published tables are that
    average.

    The light condition enters once. The range runs to the nominal threshold,
    and every track segment carries the probability that its Poisson-thinned
    hits meet the multiplicity, the weight of
    :func:`~softpaws.response.light_reach.effective_body_km`, so the threshold
    is a smooth turn-on and not a cut. Truncating the range at
    :func:`~softpaws.response.light_reach.muon_threshold_gev` as well would
    suppress the same physics twice.

    Parameters
    ----------
    which : {"IceCube", "ARCA"}
        Which detector to build.
    site : Optics
        Detector optics.
    min_modules : float, optional
        Modules that must fire. Defaults to
        :data:`~softpaws.response.light_reach.DEFAULT_MIN_MODULES`.
    n_energy : int, optional
        Points in the arrival-energy quadrature.
    flavours : tuple of str, optional
        Parent channels to sum. Defaults to :data:`DEFAULT_FLAVOURS`.
    species : tuple of CrossSection, optional
        Cross sections averaged over. Defaults to :data:`SPECIES`.
    **builder_kwargs
        Passed to :func:`ic_effective_area_cm2` or
        :func:`arca_effective_area_cm2`, for example ``log10_e``.

    Returns
    -------
    aeff : np.ndarray
        Effective area [cm^2] on that detector's grid.
    """
    build = ic_effective_area_cm2 if which == "IceCube" else arca_effective_area_cm2
    curves = [build(site, DEFAULT_MUON_THRESHOLD_GEV, min_modules, n_energy, flavours, xsec,
                    **builder_kwargs)
              for xsec in species]
    return np.mean(curves, axis=0)


def residuals(
    published: np.ndarray, model: np.ndarray, band: np.ndarray
) -> tuple[float, float]:
    """Geometric-mean ratio and shape scatter of ``published / model`` over a band.

    Parameters
    ----------
    published : np.ndarray
        Published effective area [cm^2].
    model : np.ndarray
        Model effective area on the same grid [cm^2].
    band : np.ndarray
        Boolean mask of the energies compared.

    Returns
    -------
    mean : float
        Geometric mean of ``published / model`` over the band.
    scatter : float
        Standard deviation of ``log10(published / model)`` over the band [dex].
    """
    valid = band & np.isfinite(published) & np.isfinite(model) & (model > 0.0)
    res = np.log10(published[valid] / model[valid])
    return float(10**np.mean(res)), float(np.std(res))


def fit_reach(
    which: str, site: Optics, min_modules: float, n_energy: int, published: np.ndarray,
    band: np.ndarray, flavours: tuple[str, ...] = DEFAULT_FLAVOURS, **builder_kwargs,
) -> tuple[float, float, np.ndarray]:
    """Float a flat attenuation length against one published curve.

    ``Lambda`` sets how fast the footprint grows per e-fold, which is shape
    alone, so it is fitted against the scatter about a free normalization.
    Fitting it against the level instead drives it to the bound, since no
    attenuation length can supply a normalization, and returns a number no
    medium has.

    The returned curve carries that free normalization, so it lands on the
    published one by construction and only its shape is a statement.

    Parameters
    ----------
    which : {"IceCube", "ARCA"}
        Which detector to build.
    site : Optics
        Detector optics, whose derived attenuation length starts the search.
    min_modules : float
        Modules that must fire, held at the derived value.
    n_energy : int
        Points in the arrival-energy quadrature.
    published : np.ndarray
        Published effective area [cm^2].
    band : np.ndarray
        Boolean mask of the energies the fit runs over.
    flavours : tuple of str, optional
        Parent channels to sum. Defaults to :data:`DEFAULT_FLAVOURS`.
    **builder_kwargs
        Passed to :func:`build_model`, for example ``log10_e``.

    Returns
    -------
    length_m : float
        Fitted attenuation length [m], to compare against the medium's optics.
    normalization : float
        Factor the fitted shape needs to reach the published level.
    curve : np.ndarray
        Normalized effective area [cm^2].
    """
    def model_at(log10_lambda: float) -> np.ndarray:
        trial = replace(site, attenuation_override_m=10.0**log10_lambda)
        return build_model(which, trial, min_modules, n_energy, flavours, **builder_kwargs)

    def shape_cost(log10_lambda: float) -> float:
        return residuals(published, model_at(log10_lambda), band)[1]

    opt = minimize_scalar(shape_cost, bounds=(1.0, 2.5), method="bounded",
                          options={"xatol": 0.005})
    curve = model_at(opt.x)
    normalization = residuals(published, curve, band)[0]
    return 10.0**opt.x, normalization, curve * normalization


def detector_curves(
    which: str, site: Optics, min_modules: float, n_energy: int, published: np.ndarray,
    band: np.ndarray, flavours: tuple[str, ...] = DEFAULT_FLAVOURS, **builder_kwargs,
) -> tuple[dict[str, np.ndarray], tuple[float, float]]:
    """Published, first-principles and fitted effective areas for one detector [cm^2].

    Parameters
    ----------
    which : {"IceCube", "ARCA"}
        Which detector to build.
    site : Optics
        Detector optics.
    min_modules : float
        Modules that must fire.
    n_energy : int
        Points in the arrival-energy quadrature.
    published : np.ndarray
        Published effective area [cm^2] on the detector's grid.
    band : np.ndarray
        Boolean mask of the energies the fit runs over.
    flavours : tuple of str, optional
        Parent channels to sum. Defaults to :data:`DEFAULT_FLAVOURS`.
    **builder_kwargs
        Passed to :func:`build_model`, for example ``log10_e``.

    Returns
    -------
    curves : dict of str -> np.ndarray
        Keyed ``"Published"``, ``"First principles"`` and ``"Fitted"``.
    fit : tuple of float
        ``(attenuation length [m], normalization)`` from the fit.
    """
    first_principles = build_model(which, site, min_modules, n_energy, flavours,
                                   **builder_kwargs)
    length_m, normalization, fitted = fit_reach(
        which, site, min_modules, n_energy, published, band, flavours, **builder_kwargs)
    return {
        "Published": published,
        "First principles": first_principles,
        "Fitted": fitted,
    }, (length_m, normalization)
