r"""Light reach of a detector, derived from its optics and its module.

A muon's Cherenkov output per unit length follows its energy loss. The bare
track radiates at the Frank-Tamm rate, and each GeV of radiative loss builds
an electromagnetic shower carrying :data:`EM_TRACK_LENGTH_M_PER_GEV` of
charged track, so the yield is

.. math:: N'(E) = \\frac{{\\rm d}N_\\gamma}{{\\rm d}x}\\,
    \\left(1 + L_{\\rm em}\\, b_\\mu E\\right),

the ``a + bE`` of the loss law with its own coefficients. Light leaves the
track cylindrically and is attenuated on the medium's effective photon
length, which is ``sqrt(lambda_abs lambda_scat / 3)`` where scattering
dominates, as in deep ice, and ``lambda_abs`` where it does not, as in sea
water. Every integral here is resolved in wavelength on
:data:`WAVELENGTH_NM`, because the photocathode and the medium each admit a
band ~120 nm wide, and taking either flat across the nominal 300 to 600 nm
overstates the collected light by about a factor of two.

Two conditions follow. A *hit* is a local coincidence, an HLC pair on
neighbouring modules at IceCube or two photomultipliers of one module at
KM3NeT, with each receiver converting its share of the collected charge into
at least one photoelectron at Poisson probability
(:func:`hit_probability`). A selection then demands a *multiplicity*: enough
of those hits, eight for IceCube's simple-majority trigger
(:data:`DEFAULT_MIN_MODULES`). The mean count over the in-array transverse
plane (:func:`hit_count`) sets the reach (:func:`reach_offset_m`), the
Poisson probability of meeting the multiplicity at that mean turns the
threshold into a smooth turn-on (:func:`effective_body_km`), and no charge is
left to choose.

Every input is an instrument or medium number carried by an
:class:`~softpaws.detectors.Optics` record: the photocathode area, the module
density, the photon detection efficiency against wavelength, and the medium's
absorption and scattering against wavelength.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammainc

from softpaws.detectors import ANCHOR_NM, Optics
from softpaws.transport.coefficients import drift_coefficient
from softpaws.transport.muon_range import DEFAULT_MUON_THRESHOLD_GEV
from softpaws.utils.constants import M_PER_KM, RHO_WATER_G_CM3

__all__ = [
    "DEFAULT_MIN_MODULES",
    "EM_TRACK_LENGTH_M_PER_GEV",
    "FINE_STRUCTURE",
    "HLC_PARTNERS",
    "PROJECTED_FRACTION",
    "WAVELENGTH_NM",
    "attenuation_length_m",
    "attenuation_spectrum_m",
    "brightness_factor",
    "cherenkov_spectrum_per_m_per_nm",
    "detection_efficiency",
    "effective_body_km",
    "hit_count",
    "hit_probability",
    "hit_radius_m",
    "instrumented_chord_km",
    "module_area_m2",
    "module_charge_pe",
    "muon_threshold_gev",
    "reach_offset_m",
]

#: Fine-structure constant, for the Frank-Tamm yield.
FINE_STRUCTURE = 7.2973525693e-3

#: Wavelengths every optical integral runs over [nm]. Wide enough that the
#: photocathode and the medium, not the grid, decide where the band ends.
WAVELENGTH_NM = np.linspace(280.0, 680.0, 201)

#: Charged track length in an electromagnetic shower [m GeV^-1], at water
#: density. Paired with the water-density ``b_mu`` in
#: :func:`brightness_factor`, so the product that enters the yield is density
#: independent.
EM_TRACK_LENGTH_M_PER_GEV = 4.0

#: Coincidence partners of a single-PMT module: IceCube's HLC accepts the
#: nearest or next-to-nearest neighbour on the same string, up or down. Part of
#: the trigger definition, and a count, so it is not a tunable.
HLC_PARTNERS = 4

#: Modules that must register a coincident hit for the track to count. A hit is
#: a local coincidence, an HLC pair at IceCube or two photomultipliers of one
#: module at KM3NeT, with each receiver firing on one photoelectron at Poisson
#: probability (:func:`hit_probability`); what a trigger then demands is a
#: multiplicity, and IceCube's simple-majority trigger asks for eight.
#: Everything else the condition needs is a published instrument number: the
#: module density, the photocathode area, the efficiency curve and the medium's
#: optics. See :func:`hit_count`.
DEFAULT_MIN_MODULES = 8.0

#: Fraction of its photocathode area a module presents to an arriving photon.
#: A sphere uniformly covered with photocathode of area ``A`` presents ``A / 4``
#: from every direction; a single flat photomultiplier facing one hemisphere
#: with cosine acceptance averages to the same quarter over the full sky.
PROJECTED_FRACTION = 0.25


# ---------------------------------------------------------------------------
# The light-yield model, resolved in wavelength
# ---------------------------------------------------------------------------


def cherenkov_spectrum_per_m_per_nm(site: Optics) -> np.ndarray:
    r"""Frank-Tamm photon yield of a bare relativistic track [m^-1 nm^-1].

    .. math:: \\frac{{\\rm d}^2N_\\gamma}{{\\rm d}x\\,{\\rm d}\\lambda}
        = \\frac{2\\pi\\alpha}{\\lambda^2}\\left(1 - n^{-2}\\right),

    on :data:`WAVELENGTH_NM`. Integrating it over 300 to 600 nm returns the
    ~3.3e4 photons per metre that a single-number treatment starts from, but
    the ``1 / lambda^2`` decides *which* photons those are, and neither the
    photocathode nor the medium treats them alike.

    Parameters
    ----------
    site : Optics
        Detector optics, for the refractive index.

    Returns
    -------
    spectrum : np.ndarray
        Photons per metre of track per nanometre, on :data:`WAVELENGTH_NM`.
    """
    lam_m = WAVELENGTH_NM * 1.0e-9
    per_m_per_m = (2.0 * np.pi * FINE_STRUCTURE / lam_m**2
                   * (1.0 - site.refractive_index**-2))
    return per_m_per_m * 1.0e-9


def detection_efficiency(site: Optics) -> np.ndarray:
    """Probability that a photon reaching the module produces a photoelectron.

    The photocathode's quantum efficiency times the transmission of the
    pressure sphere and the optical gel, tabulated together on
    :data:`WAVELENGTH_NM`. Taking it flat at its peak across the whole band,
    which is what a single ``quantum_efficiency`` does, counts photons the
    module cannot convert: the response is a bump ~120 nm wide sitting inside
    a 300 nm band, cut off below by the glass and above by the cathode.

    Parameters
    ----------
    site : Optics
        Detector optics, for the tabulated efficiency curve.

    Returns
    -------
    efficiency : np.ndarray
        Photon detection efficiency on :data:`WAVELENGTH_NM`.
    """
    grid, values = np.asarray(site.efficiency_nm), np.asarray(site.efficiency)
    return np.interp(WAVELENGTH_NM, grid, values, left=0.0, right=0.0)


def attenuation_spectrum_m(site: Optics) -> np.ndarray:
    """Effective photon attenuation length of the medium, per wavelength [m].

    Where scattering is short against absorption the transport is diffusive
    and the flux falls on ``sqrt(lambda_abs lambda_scat / 3)``; where it is
    not, the light travels ballistically and the length is ``lambda_abs``.
    The shorter of the two selects the applicable limit at each wavelength.

    Both lengths are the site's own single-wavelength values carried by a
    tabulated shape, normalized at :data:`~softpaws.detectors.ANCHOR_NM`, so
    ``absorption_m`` and ``scattering_m`` keep their published meaning and
    only the wavelength dependence is added. Deep ice is clearest near 400 nm
    and opaque by 600 nm, so the red half of the nominal band is gone long
    before the reach is interesting, and the light that survives to large
    distance is a narrow window near the clarity peak.

    Parameters
    ----------
    site : Optics
        Detector optics.

    Returns
    -------
    length_m : np.ndarray
        Attenuation length [m] on :data:`WAVELENGTH_NM`, or a flat
        ``attenuation_override_m`` when a fit has set one.
    """
    if site.attenuation_override_m is not None:
        return np.full_like(WAVELENGTH_NM, float(site.attenuation_override_m))
    shape_abs = np.interp(WAVELENGTH_NM, np.asarray(site.absorption_nm),
                          np.asarray(site.absorption_shape))
    shape_scat = np.interp(WAVELENGTH_NM, np.asarray(site.scattering_nm),
                           np.asarray(site.scattering_shape))
    anchor_abs = np.interp(ANCHOR_NM, np.asarray(site.absorption_nm),
                           np.asarray(site.absorption_shape))
    anchor_scat = np.interp(ANCHOR_NM, np.asarray(site.scattering_nm),
                            np.asarray(site.scattering_shape))
    absorption = site.absorption_m * shape_abs / anchor_abs
    scattering = site.scattering_m * shape_scat / anchor_scat
    diffusive = np.sqrt(absorption * scattering / 3.0)
    return np.minimum(diffusive, absorption)


def attenuation_length_m(site: Optics) -> float:
    """Attenuation length at the clarity peak [m], for reporting only.

    The model integrates :func:`attenuation_spectrum_m` and never uses a
    single number; this is the value at :data:`~softpaws.detectors.ANCHOR_NM`,
    which is what a data sheet quotes and what a fitted length is comparable
    against.

    Parameters
    ----------
    site : Optics
        Detector optics.

    Returns
    -------
    length_m : float
        Attenuation length at the anchor wavelength [m], or the flat
        ``attenuation_override_m`` when a fit has set one.
    """
    if site.attenuation_override_m is not None:
        return float(site.attenuation_override_m)
    return float(np.interp(ANCHOR_NM, WAVELENGTH_NM, attenuation_spectrum_m(site)))


def module_area_m2(site: Optics) -> float:
    """Geometric photocathode area a module presents to an arriving photon [m^2].

    The photocathode area times :data:`PROJECTED_FRACTION`. It carries no
    efficiency: the conversion probability sits inside the wavelength
    integral, since it is the one thing in the chain that varies fastest
    across the band.

    Parameters
    ----------
    site : Optics
        Detector optics.

    Returns
    -------
    area_m2 : float
        Projected photocathode area [m^2].
    """
    return site.cathode_area_m2 * PROJECTED_FRACTION


def brightness_factor(energy_gev: float | np.ndarray) -> np.ndarray:
    """Track brightness relative to a minimum-ionizing muon.

    ``1 + L_em b_mu E``: the bare track plus the electromagnetic showers of
    the radiative loss, which carry ``L_em`` metres of charged track per GeV.
    The shower light has the same Cherenkov spectrum as the bare track, so
    this factors out of every wavelength integral.

    Parameters
    ----------
    energy_gev : float or np.ndarray
        Muon energy [GeV].

    Returns
    -------
    brightness : np.ndarray
        Light yield relative to the bare track.
    """
    energy = np.asarray(energy_gev, dtype=float)
    b_water = drift_coefficient(energy, RHO_WATER_G_CM3) / M_PER_KM
    return 1.0 + EM_TRACK_LENGTH_M_PER_GEV * b_water * energy


def module_charge_pe(
    distance_m: float | np.ndarray, energy_gev: float | np.ndarray, site: Optics
) -> np.ndarray:
    r"""Photoelectrons a module collects from a track passing at a distance.

    Light leaves a long track cylindrically, so the fluence at perpendicular
    distance ``d`` is the yield per metre spread over ``2 pi d`` and
    attenuated on the medium's length. Summing over the band,

    .. math:: Q(d, E) = \\frac{A_{\\rm mod}}{2\\pi d}\\, Y(E) \\int {\\rm d}\\lambda\\;
        \\frac{{\\rm d}^2N_\\gamma}{{\\rm d}x\\,{\\rm d}\\lambda}\\,
        \\eta(\\lambda)\\, {\\rm e}^{-d / \\Lambda(\\lambda)},

    with ``Y`` the brightness of :func:`brightness_factor`. The wavelength
    integral is where a single-number treatment loses: ``eta`` and ``Lambda``
    peak in the same narrow window, and the exponential narrows it further
    with distance, so the effective band shrinks as the reach grows.

    Parameters
    ----------
    distance_m : float or np.ndarray
        Perpendicular distance from the track [m].
    energy_gev : float or np.ndarray
        Muon energy [GeV], broadcast against ``distance_m``.
    site : Optics
        Detector optics.

    Returns
    -------
    charge_pe : np.ndarray
        Collected charge [photoelectrons].
    """
    distance = np.atleast_1d(np.asarray(distance_m, dtype=float))
    weight = (cherenkov_spectrum_per_m_per_nm(site) * detection_efficiency(site))
    exponent = -distance[..., None] / attenuation_spectrum_m(site)
    collected = np.trapezoid(weight * np.exp(exponent), WAVELENGTH_NM, axis=-1)
    geometry = module_area_m2(site) / (2.0 * np.pi * np.maximum(distance, 1.0e-6))
    return geometry * collected * brightness_factor(energy_gev)


#: Distances the one-photoelectron radius is tabulated on [m], log spaced so the
#: inversion stays accurate over the four decades of brightness in play.
_HIT_DISTANCE_M = np.logspace(-1.0, 3.2, 400)

#: Signed offsets the mean hit count is tabulated on [m], for the inversion in
#: :func:`reach_offset_m`. The positive end comfortably exceeds any reach in
#: play; the negative end only has to cover the interpolation edge, since a
#: track the condition wants *inside* the array is handled by the multiplicity
#: weight of :func:`effective_body_km` and the offset is clipped at zero.
_REACH_OFFSET_M = np.linspace(-100.0, 1200.0, 261)

#: The wedge kernel of :func:`hit_count` on that offset grid,
#: ``2 d arccos(clip(x / d))``, tabulated once: the mean count at every offset
#: is then one matrix product with the hit probabilities.
_REACH_KERNEL = 2.0 * _HIT_DISTANCE_M[:, None] * np.arccos(
    np.clip(_REACH_OFFSET_M[None, :] / _HIT_DISTANCE_M[:, None], -1.0, 1.0))

#: Trapezoid quadrature weights of the distance grid, so the count integrals
#: reduce to matrix products against :data:`_REACH_KERNEL`.
_D_STEP = np.diff(_HIT_DISTANCE_M)
_D_TRAPZ = np.concatenate(
    [[0.5 * _D_STEP[0]], 0.5 * (_D_STEP[:-1] + _D_STEP[1:]), [0.5 * _D_STEP[-1]]])


def hit_radius_m(energy_gev: float | np.ndarray, site: Optics) -> np.ndarray:
    """Distance at which a module still collects one photoelectron [m].

    A reporting scale only: the counting that enters the model is Poisson
    over :func:`hit_probability`, with no step at any radius. This is the
    distance at which the *mean* charge falls to one photoelectron, which is
    what a single-number summary of the optics can be compared against.

    Parameters
    ----------
    energy_gev : float or np.ndarray
        Muon energy [GeV].
    site : Optics
        Detector optics.

    Returns
    -------
    radius_m : np.ndarray
        Radius of the one-photoelectron cylinder [m]. Zero where even a
        module on the track does not reach one photoelectron.
    """
    charge = module_charge_pe(_HIT_DISTANCE_M, np.atleast_1d(energy_gev)[..., None], site)
    # Q falls monotonically with distance, so the crossing is unique. Locate it
    # with one argmax per row and interpolate in log(charge) against
    # log(distance), where both are close to straight.
    log_d = np.log(_HIT_DISTANCE_M)
    with np.errstate(divide="ignore"):
        log_q = np.log(np.clip(charge, 1.0e-300, None))
    faint = charge < 1.0
    crossed = faint.any(axis=-1)
    hi = np.argmax(faint, axis=-1)
    lo = np.clip(hi - 1, 0, None)
    span = np.take_along_axis(log_q, lo[..., None], -1)[..., 0] - \
        np.take_along_axis(log_q, hi[..., None], -1)[..., 0]
    start = np.take_along_axis(log_q, lo[..., None], -1)[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(span > 0.0, start / span, 0.0)
    radius = np.exp(log_d[lo] + frac * (log_d[hi] - log_d[lo]))
    radius = np.where(hi == 0, 0.0, radius)
    return np.where(crossed, radius, _HIT_DISTANCE_M[-1])


def instrumented_chord_km(radius_km: float, height_km: float, n_sides: int | None) -> float:
    """Mean chord of the instrumented body [km].

    ``<c> = 4V / S`` for any convex body, so this needs no new number. It is
    the length of track the array has to work with, and therefore the length
    over which :func:`hit_count` counts modules.

    Parameters
    ----------
    radius_km : float
        Footprint radius of the body [km].
    height_km : float
        Height of the body [km].
    n_sides : int or None
        Sides of the prism cross-section, or ``None`` for a cylinder.

    Returns
    -------
    chord_km : float
        Mean chord [km].
    """
    volume = np.pi * radius_km**2 * height_km
    if n_sides is None:
        perimeter = 2.0 * np.pi * radius_km
    else:
        perimeter = 2.0 * radius_km * np.sqrt(np.pi * n_sides * np.tan(np.pi / n_sides))
    surface = 2.0 * np.pi * radius_km**2 + perimeter * height_km
    return float(4.0 * volume / surface)


def hit_probability(
    distance_m: float | np.ndarray, energy_gev: float | np.ndarray, site: Optics
) -> np.ndarray:
    """Probability that a module at a distance registers a *hit*.

    A hit is a local coincidence, because that is what both instruments
    count: IceCube's simple-majority trigger counts HLC hits, and KM3NeT's L1
    is two photomultipliers of one module within ~10 ns, since in sea water
    a single photoelectron is indistinguishable from potassium-40 decay. Each
    receiver converts its mean charge into at least one photoelectron with
    Poisson probability ``p = 1 - e^{-Q}``, and the coincidence admits every
    partner the definition allows:

    - A single-PMT module (IceCube) pairs with its nearest or next-to-nearest
      neighbours on the same string, up or down, so a firing module counts
      when *any* of :data:`HLC_PARTNERS` partners at essentially the same
      track distance also fires: ``p (1 - (1 - p)^4)``.
    - A multi-PMT module (KM3NeT) splits its collected charge over the
      ``illuminated_pmts`` that face the track, and counts when any two fire:
      ``1 - (1-p)^m - m p (1-p)^(m-1)`` with ``p = 1 - e^{-Q/m}``.

    Both reduce to the module firing outright when the track is bright. The
    partner sum matters in the dim limit, where a fixed-pair rule
    under-counts by the number of partners, enough to push the IceCube
    turn-on from 1.8 TeV to 3.7 TeV and visibly suppress the effective area
    below 100 TeV. A step function, every module inside the one-photoelectron
    radius fires and none outside, is worse still, putting the threshold at
    5.3 TeV where the array demonstrably triggers below 1 TeV.

    Parameters
    ----------
    distance_m : float or np.ndarray
        Perpendicular distance from the track [m].
    energy_gev : float or np.ndarray
        Muon energy [GeV], broadcast against ``distance_m``.
    site : Optics
        Detector optics.

    Returns
    -------
    probability : np.ndarray
        Probability that the module registers a coincident hit.
    """
    charge = module_charge_pe(distance_m, energy_gev, site)
    if site.illuminated_pmts is None:
        single = 1.0 - np.exp(-charge)
        return single * (1.0 - (1.0 - single) ** HLC_PARTNERS)
    m = float(site.illuminated_pmts)
    single = 1.0 - np.exp(-charge / m)
    return 1.0 - (1.0 - single) ** m - m * single * (1.0 - single) ** (m - 1.0)


def hit_count(
    offset_m: float, energy_gev: float | np.ndarray, site: Optics, chord_km: float,
) -> np.ndarray:
    r"""Mean number of hit modules, for a track at a signed distance from the boundary.

    Each module hits with :func:`hit_probability`, so the mean count is that
    probability integrated over the in-array part of the transverse plane. A
    circle of radius ``d`` around a track at signed offset ``x`` from the
    boundary keeps the fraction ``arccos(x / d) / pi`` of its circumference
    inside, hence

    .. math:: \\bar N(x, E) = \\rho_{\\rm mod}\\,\\langle c\\rangle \\int
        2\\,d\\,\\arccos\\!\\left({\\rm clip}(x / d)\\right) p_{\\rm hit}(d, E)\\,
        {\\rm d}d,

    with the mean chord as the track length in view. For a step ``p_hit``
    this is the circular-segment count of a disc treatment; the Poisson form
    differs where it matters, in the dim limit, where the count becomes
    linear in the collected charge.

    Parameters
    ----------
    offset_m : float
        Signed distance of the track from the boundary [m]; positive is outside.
    energy_gev : float or np.ndarray
        Muon energy [GeV].
    site : Optics
        Detector optics.
    chord_km : float
        Mean chord of the instrumented body [km], from
        :func:`instrumented_chord_km`.

    Returns
    -------
    count : np.ndarray
        Mean number of modules registering a coincident hit.
    """
    probability = hit_probability(
        _HIT_DISTANCE_M, np.atleast_1d(np.asarray(energy_gev, dtype=float))[..., None],
        site)
    wedge = 2.0 * _HIT_DISTANCE_M * np.arccos(
        np.clip(float(offset_m) / _HIT_DISTANCE_M, -1.0, 1.0))
    density_per_m3 = site.module_density_per_km3 / M_PER_KM**3
    count = (probability * _D_TRAPZ) @ wedge
    return density_per_m3 * count * chord_km * M_PER_KM


def _mean_counts(
    energy_gev: np.ndarray, site: Optics, chord_km: float
) -> tuple[np.ndarray, np.ndarray]:
    """Mean hit counts on :data:`_REACH_OFFSET_M`, and for a central track.

    One evaluation of the optics serves both: the reach inversion needs the
    count against the offset, the multiplicity weight needs the count deep
    inside the array, where the wedge is the full circle.

    Parameters
    ----------
    energy_gev : np.ndarray
        Muon energy [GeV].
    site : Optics
        Detector optics.
    chord_km : float
        Mean chord of the instrumented body [km].

    Returns
    -------
    counts : np.ndarray, shape (energy, offset)
        Mean hit count at each tabulated offset.
    central : np.ndarray, shape (energy,)
        Mean hit count for a central crossing track.
    """
    probability = hit_probability(_HIT_DISTANCE_M, energy_gev[..., None], site)
    scale = site.module_density_per_km3 / M_PER_KM**3 * chord_km * M_PER_KM
    weighted = probability * _D_TRAPZ
    return (scale * (weighted @ _REACH_KERNEL),
            scale * (weighted @ (2.0 * np.pi * _HIT_DISTANCE_M)))


def _invert_reach_m(counts: np.ndarray, min_modules: float) -> np.ndarray:
    """Offset at which each row of ``counts`` falls to ``min_modules`` [m].

    Parameters
    ----------
    counts : np.ndarray, shape (energy, offset)
        Mean hit counts on :data:`_REACH_OFFSET_M`, from :func:`_mean_counts`.
    min_modules : float
        Mean number of hits demanded.

    Returns
    -------
    offset_m : np.ndarray, shape (energy,)
        Signed offset [m], held at the edges of the tabulated offsets.
    """
    return np.array([
        np.interp(min_modules, row[::-1], _REACH_OFFSET_M[::-1],
                  left=_REACH_OFFSET_M[-1], right=_REACH_OFFSET_M[0])
        for row in counts
    ])


def reach_offset_m(
    energy_gev: np.ndarray, site: Optics, chord_km: float, min_modules: float
) -> np.ndarray:
    """How far outside the boundary a track can be and still make ``min_modules``.

    Inverts the mean count of :func:`hit_count` in the signed offset. The
    count falls monotonically with the offset, so the crossing is unique and
    one tabulated count against offset serves every energy at once.

    Parameters
    ----------
    energy_gev : np.ndarray
        Muon energy [GeV].
    site : Optics
        Detector optics.
    chord_km : float
        Mean chord of the instrumented body [km].
    min_modules : float
        Mean number of hits demanded.

    Returns
    -------
    offset_m : np.ndarray
        Signed distance [m], held at the edges of the tabulated offsets. The
        negative edge does not need to be deep: below it the acceptance is
        carried by the multiplicity weight of :func:`effective_body_km`.
    """
    energy = np.atleast_1d(np.asarray(energy_gev, dtype=float))
    counts, _ = _mean_counts(energy, site, chord_km)
    return _invert_reach_m(counts, min_modules)


def muon_threshold_gev(site: Optics, min_modules: float, chord_km: float) -> float:
    """Muon energy at which a track through the array first fires ``min_modules``.

    Deep inside the array the segment is the whole disc, so the condition is
    ``rho_mod pi d_1^2 <c> = N_min``: the hit radius has to reach a definite
    value, and that fixes an energy. This is the threshold the *light* sets,
    and with Poisson-thinned hits it is the midpoint of a turn-on: the
    multiplicity weight of :func:`effective_body_km` passes ~55% of central
    tracks here, more above, fewer below. The range still runs to the nominal
    threshold; the weight carries the dimming.

    Parameters
    ----------
    site : Optics
        Detector optics.
    min_modules : float
        Modules that must fire.
    chord_km : float
        Mean chord of the instrumented body [km].

    Returns
    -------
    threshold_gev : float
        Muon threshold [GeV], held at or above
        :data:`~softpaws.transport.muon_range.DEFAULT_MUON_THRESHOLD_GEV`.
        ``nan`` when no energy up to ``10^12`` GeV meets the multiplicity.
    """
    def gap(log10_e: float) -> float:
        deep = -1.0e4
        count = np.atleast_1d(hit_count(deep, 10.0**log10_e, site, chord_km))
        return float(count[0]) - min_modules

    if gap(12.0) < 0.0:
        return float("nan")
    if gap(0.0) > 0.0:
        return DEFAULT_MUON_THRESHOLD_GEV
    return max(10.0 ** brentq(gap, 0.0, 12.0, xtol=1.0e-4), DEFAULT_MUON_THRESHOLD_GEV)


def effective_body_km(
    radius_km: float, height_km: float, energy_gev: float | np.ndarray, site: Optics,
    min_modules: float, n_sides: int | None = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Instrumented body dilated by the derived light reach, and its weight [km].

    The reach is where a track still makes ``min_modules`` mean hits
    (:func:`reach_offset_m`). It is a distance from a *boundary*, so the same
    reach moves the body in every direction and not only radially.
    Vertically it is capped by the medium: both sites are layered, with an
    upper boundary at the ice or sea surface and a lower one at bedrock or
    seabed, and a track outside the optical medium is neither radiating into
    it nor visible through it. The horizontal directions carry no such cap.

    Dim muons are carried by the third return, a multiplicity weight: the
    hits are Poisson-thinned, so a track whose *mean* count sits below
    ``min_modules`` still meets the selection with probability
    ``P(N >= min_modules)``, evaluated for a central crossing. The weight is
    what turns the threshold into a smooth turn-on; emptying the body below
    the mean-count threshold instead is a delta-function condition that
    collapses the model at low energy, where the published response falls
    smoothly. The weight applies the dimming exactly once: the range keeps
    its nominal lower limit, and the offset is clipped at zero because
    sub-threshold acceptance belongs to the weight.

    Parameters
    ----------
    radius_km : float
        Instrumented footprint radius [km].
    height_km : float
        Instrumented height [km].
    energy_gev : float or np.ndarray
        Muon energy where the track is seen [GeV].
    site : Optics
        Detector optics.
    min_modules : float
        Mean number of hits demanded.
    n_sides : int or None, optional
        Cross-section of the instrumented body, for its mean chord.

    Returns
    -------
    radius : np.ndarray
        Effective radius [km].
    height : np.ndarray
        Effective height [km], grown at each end cap and held inside the medium.
    weight : np.ndarray
        Probability that a central crossing track meets the multiplicity.
    """
    chord_km = instrumented_chord_km(radius_km, height_km, n_sides)
    energy = np.atleast_1d(np.asarray(energy_gev, dtype=float))
    counts, central = _mean_counts(energy, site, chord_km)
    offset_km = np.clip(_invert_reach_m(counts, min_modules) / M_PER_KM, 0.0, None)
    above_km = np.minimum(offset_km, site.headroom_above_m / M_PER_KM)
    below_km = np.minimum(offset_km, site.headroom_below_m / M_PER_KM)
    # P(N >= k) for a Poisson mean is the regularized lower incomplete gamma.
    weight = gammainc(min_modules, central)
    return (np.clip(radius_km + offset_km, 0.0, None),
            np.clip(height_km + above_km + below_km, 0.0, None),
            weight)
