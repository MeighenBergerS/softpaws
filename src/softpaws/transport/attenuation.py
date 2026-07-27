"""Earth attenuation of the neutrino flux before it reaches the detector.

The soft-volume forward model (:mod:`softpaws.response.soft_volume`) assumes an
unattenuated flux (``D_nu = 1``, arXiv:2607.13143 Eq. 2.4), which is only valid
for the downgoing hemisphere. Upgoing neutrinos traverse the Earth, and above
~100 TeV the neutrino-nucleon cross section grows enough that the Earth becomes
opaque -- the dominant effect shaping the upgoing UHE spectrum.

The survival probability along a chord of column depth ``X`` [g cm^-2] is

.. math:: D_\\nu(E) = \\exp\\!\\bigl[-N_A\\,\\sigma_\\mathrm{tot}(E)\\,X\\bigr],

with ``sigma_tot`` the total (CC + NC) neutrino-nucleon cross section. Only
absorption is modelled: neutral-current down-scattering and tau regeneration
are neglected (a ``nu_mu`` disappearance picture).

This module provides two columns:

- a **constant mean-density** chord (:func:`mean_density_column`), whose
  solid-angle average over a declination band (:func:`representative_column`)
  gives the single scalar used by the closed-form attenuation of
  :class:`~softpaws.response.soft_volume.SoftVolumeResponse`;
- a **layered PREM** chord (:func:`prem_column`), used by the per-event
  effective solid angle (:func:`effective_solid_angle`) that
  :meth:`~softpaws.response.soft_volume.SoftVolumeResponse.expected_counts_attenuated`
  integrates over the band.

Geometry follows the South Pole detector convention (see
``docs/soft_volume_notes.md``): a source at declination ``dec`` arrives at
zenith ``90 deg + dec``, so the upgoing hemisphere is ``dec > 0``. The Earth
chord for a (near-)surface detector is ``L(dec) = 2 R_Earth sin(dec)``: it
vanishes at the horizon (``dec = 0``) and reaches the full diameter for
vertically upgoing tracks (``dec = 90 deg``).
"""

from __future__ import annotations

import numpy as np

from ..utils.constants import (
    AVOGADRO_PER_MOL,
    CM_PER_KM,
    EARTH_RADIUS_KM,
    RHO_EARTH_MEAN_G_CM3,
)
from .source import DEFAULT_LAMBDA, cc_cross_section

# Total-to-CC cross-section ratio. The neutral-current channel adds ~40% to the
# charged-current cross section at UHE, so sigma_tot ~ 1.4 sigma_CC; the flux is
# attenuated by both channels while detection uses CC only.
TOTAL_TO_CC_RATIO = 1.4


def total_cross_section(
    energy_gev: float | np.ndarray,
    lam: float = DEFAULT_LAMBDA,
) -> np.ndarray:
    """Total neutrino-nucleon cross section for Earth attenuation.

    Reuses the charged-current power law of
    :func:`softpaws.transport.source.cc_cross_section` scaled by
    :data:`TOTAL_TO_CC_RATIO` to include the neutral-current channel.

    Parameters
    ----------
    energy_gev : float or np.ndarray
        Neutrino energy [GeV].
    lam : float, optional
        Cross-section slope. Defaults to
        :data:`~softpaws.transport.source.DEFAULT_LAMBDA`.

    Returns
    -------
    sigma : np.ndarray
        Total cross section per nucleon [cm^2].
    """
    return TOTAL_TO_CC_RATIO * cc_cross_section(energy_gev, lam)


def survival_probability(
    energy_gev: float | np.ndarray,
    column_g_cm2: float | np.ndarray,
    lam: float = DEFAULT_LAMBDA,
) -> np.ndarray:
    """Neutrino survival probability ``D_nu`` through a column (Eq. 2.4).

    Parameters
    ----------
    energy_gev : float or np.ndarray
        Neutrino energy [GeV].
    column_g_cm2 : float or np.ndarray
        Traversed column depth ``X`` [g cm^-2]. Broadcast against
        ``energy_gev``.
    lam : float, optional
        Cross-section slope. Defaults to
        :data:`~softpaws.transport.source.DEFAULT_LAMBDA`.

    Returns
    -------
    d_nu : np.ndarray
        Survival probability ``exp(-N_A sigma_tot X)``, in ``[0, 1]``.
    """
    sigma = total_cross_section(energy_gev, lam)
    optical_depth = AVOGADRO_PER_MOL * sigma * np.asarray(column_g_cm2, dtype=float)
    return np.exp(-optical_depth)


def earth_chord_length_km(declination_deg: float | np.ndarray) -> np.ndarray:
    """Chord length through the Earth for a surface detector.

    ``L(dec) = 2 R_Earth sin(dec)`` for the South Pole geometry: zero at the
    horizon (``dec = 0``) and the full diameter at the nadir (``dec = 90``).

    Parameters
    ----------
    declination_deg : float or np.ndarray
        Source declination [deg]; the upgoing hemisphere is ``dec > 0``.

    Returns
    -------
    length : np.ndarray
        Chord length [km]. Zero for downgoing directions (``dec <= 0``).
    """
    dec = np.atleast_1d(np.asarray(declination_deg, dtype=float))
    length = 2.0 * EARTH_RADIUS_KM * np.sin(np.deg2rad(dec))
    return np.clip(length, 0.0, None)


def mean_density_column(declination_deg: float | np.ndarray) -> np.ndarray:
    """Column depth of a constant mean-density Earth chord.

    ``X(dec) = rho_mean * L(dec)`` with the Earth mean density
    :data:`~softpaws.utils.constants.RHO_EARTH_MEAN_G_CM3`. This is the
    closed-form column; :func:`prem_column` is the layered refinement.

    Parameters
    ----------
    declination_deg : float or np.ndarray
        Source declination [deg].

    Returns
    -------
    column : np.ndarray
        Column depth [g cm^-2].
    """
    length_cm = earth_chord_length_km(declination_deg) * CM_PER_KM
    return RHO_EARTH_MEAN_G_CM3 * length_cm


def representative_column(dec_min_deg: float, dec_max_deg: float) -> float:
    """Solid-angle-averaged mean-density column over a declination band.

    The closed-form attenuation replaces the per-direction column by this single
    scalar. With ``dOmega = 2 pi cos(dec) d(dec)`` and the constant-density
    column ``X(dec) = rho_mean * 2 R_Earth sin(dec)``, the solid-angle average
    reduces analytically to

    .. math:: \\langle X \\rangle = \\rho_\\mathrm{mean}\\,R_\\mathrm{Earth}\\,
        (\\sin\\mathrm{dec}_\\max + \\sin\\mathrm{dec}_\\min).

    Parameters
    ----------
    dec_min_deg, dec_max_deg : float
        Declination band edges [deg], with ``dec_max > dec_min``.

    Returns
    -------
    column : float
        Representative column depth [g cm^-2].

    Raises
    ------
    ValueError
        Raised if ``dec_max_deg <= dec_min_deg``.
    """
    if dec_max_deg <= dec_min_deg:
        raise ValueError(
            f"dec_max_deg ({dec_max_deg}) must exceed dec_min_deg ({dec_min_deg})."
        )
    sin_min = np.sin(np.deg2rad(dec_min_deg))
    sin_max = np.sin(np.deg2rad(dec_max_deg))
    radius_cm = EARTH_RADIUS_KM * CM_PER_KM
    return float(RHO_EARTH_MEAN_G_CM3 * radius_cm * (sin_max + sin_min))


# ---------------------------------------------------------------------------
# PREM (Dziewonski & Anderson 1981) piecewise-polynomial density profile.
# Each row is (outer radius [km], polynomial coefficients in x = r / R_Earth,
# ascending order). Density in g cm^-3.
# ---------------------------------------------------------------------------
_PREM_SHELLS = (
    (1221.5, (13.0885, 0.0, -8.8381)),
    (3480.0, (12.5815, -1.2638, -3.6426, -5.5281)),
    (5701.0, (7.9565, -6.4761, 5.5283, -3.0807)),
    (5771.0, (5.3197, -1.4836)),
    (5971.0, (11.2494, -8.0298)),
    (6151.0, (7.1089, -3.8045)),
    (6346.6, (2.6910, 0.6924)),
    (6356.0, (2.9000,)),
    (6368.0, (2.6000,)),
    (EARTH_RADIUS_KM, (1.0200,)),
)


def prem_density(radius_km: float | np.ndarray) -> np.ndarray:
    """PREM density at a given radius.

    Parameters
    ----------
    radius_km : float or np.ndarray
        Radius from the Earth center [km]. Radii beyond
        :data:`~softpaws.utils.constants.EARTH_RADIUS_KM` return zero (vacuum).

    Returns
    -------
    density : np.ndarray
        Mass density [g cm^-3].
    """
    r = np.atleast_1d(np.asarray(radius_km, dtype=float))
    x = r / EARTH_RADIUS_KM
    density = np.zeros_like(r)
    filled = np.zeros_like(r, dtype=bool)
    for outer, coeffs in _PREM_SHELLS:
        in_shell = (~filled) & (r <= outer)
        if np.any(in_shell):
            density[in_shell] = np.polynomial.polynomial.polyval(x[in_shell], coeffs)
            filled |= in_shell
    return density


def prem_column(
    declination_deg: float,
    n_steps: int = 512,
) -> float:
    """Column depth of a layered PREM Earth chord.

    Integrates the PREM density along the chord to the detector. The chord has
    impact parameter ``b = R_Earth sin(nadir)`` with nadir angle
    ``90 deg - dec``, so the radius at path length ``s`` from the chord midpoint
    is ``sqrt(b^2 + s^2)``.

    Parameters
    ----------
    declination_deg : float
        Source declination [deg]; ``dec <= 0`` (downgoing) returns zero.
    n_steps : int, optional
        Number of trapezoidal steps along the chord. Defaults to 512.

    Returns
    -------
    column : float
        Column depth [g cm^-2].
    """
    length_km = float(earth_chord_length_km(declination_deg)[0])
    if length_km <= 0.0:
        return 0.0
    half_km = 0.5 * length_km
    nadir_rad = np.deg2rad(90.0 - declination_deg)
    b_km = EARTH_RADIUS_KM * np.sin(nadir_rad)
    s_km = np.linspace(-half_km, half_km, n_steps)
    r_km = np.sqrt(b_km**2 + s_km**2)
    density = prem_density(r_km)
    # Integrate over path length; convert km -> cm for a g cm^-2 column.
    return float(np.trapezoid(density, s_km) * CM_PER_KM)


def effective_solid_angle(
    energy_gev: np.ndarray,
    dec_min_deg: float,
    dec_max_deg: float,
    lam: float = DEFAULT_LAMBDA,
    n_dec: int = 64,
) -> np.ndarray:
    """Attenuation-weighted solid angle of a declination band (per-event form).

    For an isotropic flux and a uniform target, the neutrino energy, target
    volume, cross section, and flux factor out of the band integral, leaving

    .. math:: \\Omega_\\mathrm{eff}(E) = \\int_\\mathrm{band}
        D_\\nu(E, \\mathrm{dec})\\,2\\pi\\cos\\mathrm{dec}\\,d\\mathrm{dec},

    with the per-direction PREM column. Using this in place of the geometric
    solid angle applies Earth attenuation event by event. As ``E -> 0`` (or with
    no attenuation) it recovers ``2 pi (sin dec_max - sin dec_min)``.

    Parameters
    ----------
    energy_gev : np.ndarray
        Neutrino energies [GeV].
    dec_min_deg, dec_max_deg : float
        Declination band edges [deg].
    lam : float, optional
        Cross-section slope. Defaults to
        :data:`~softpaws.transport.source.DEFAULT_LAMBDA`.
    n_dec : int, optional
        Number of declination samples for the band integral. Defaults to 64.

    Returns
    -------
    omega_eff : np.ndarray, shape (energy_gev.size,)
        Effective solid angle [sr] at each energy.
    """
    energy = np.atleast_1d(np.asarray(energy_gev, dtype=float))
    dec = np.linspace(dec_min_deg, dec_max_deg, n_dec)
    columns = np.array([prem_column(d) for d in dec])  # [g cm^-2], shape (n_dec,)

    # D_nu[energy, dec] and the solid-angle weight 2 pi cos(dec).
    d_nu = survival_probability(energy[:, None], columns[None, :], lam)
    weight = 2.0 * np.pi * np.cos(np.deg2rad(dec))
    integrand = d_nu * weight[None, :]
    return np.trapezoid(integrand, np.deg2rad(dec), axis=1)
