"""Earth geometry: PREM density, chords and columns, overburden, zenith grids.

Everything a neutrino or a muon meets on the way to the detector, as a
function of arrival direction. The PREM profile of Dziewonski and Anderson
(1981) gives the column of an upgoing chord; the medium above the detector
gives the column of a downgoing one; and the zenith grid with its solid-angle
weights is what every sky average runs over.

Notes
-----
``cos_theta = +1`` is vertically downgoing (from above, through the
overburden) and ``-1`` vertically upgoing (through the Earth). A source at
declination ``dec`` seen from the South Pole arrives at
``cos_theta = -sin(dec)``, so the upgoing hemisphere is ``dec > 0``.
"""

from __future__ import annotations

import numpy as np

from ..utils.constants import CM_PER_KM, EARTH_RADIUS_KM, RHO_EARTH_MEAN_G_CM3

__all__ = [
    "MAX_UPSTREAM_KM",
    "earth_chord_length_km",
    "mean_density_column",
    "neutrino_column_g_cm2",
    "overburden_km",
    "prem_column",
    "prem_density",
    "representative_column",
    "zenith_grid",
]

#: Longest medium column a muon is given upstream of the detector [km]. Below
#: the horizon the muon is born in rock, which never runs out, and this caps
#: the near-horizon overburden so that no integral runs to infinity. It exceeds
#: every muon range in the problem, so results are insensitive to it.
MAX_UPSTREAM_KM = 100.0


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
    r"""Solid-angle-averaged mean-density column over a declination band.

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


def zenith_grid(
    n_zenith: int,
    cos_range: tuple[float, float] = (-1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Zenith samples and their solid-angle weights over a band of the sky.

    The band is cut into ``n_zenith`` slices of equal ``cos(theta)``, which
    is equal solid angle, and each slice is represented by its centre.

    Parameters
    ----------
    n_zenith : int
        Number of slices.
    cos_range : tuple of float, optional
        Band of ``cos(theta)`` to cover. Defaults to the full sky. The output
        runs from ``+1`` (downgoing) to ``-1`` (upgoing) whatever order the
        two limits are given in.

    Returns
    -------
    theta_deg : np.ndarray, shape (n_zenith,)
        Zenith angle of each slice centre [deg], downgoing first.
    weights : np.ndarray, shape (n_zenith,)
        Solid-angle weights within the band, normalized to sum to one, so a
        weighted average over them is the average over that band.

    Examples
    --------
    >>> theta, w = zenith_grid(4)
    >>> theta.round(1)
    array([ 41.4,  75.5, 104.5, 138.6])
    >>> float(w.sum())
    1.0
    """
    cos_lo, cos_hi = sorted(cos_range)
    edges = np.linspace(cos_hi, cos_lo, n_zenith + 1)
    cos_theta = 0.5 * (edges[:-1] + edges[1:])
    theta_deg = np.rad2deg(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    return theta_deg, np.full(n_zenith, 1.0 / n_zenith)


def overburden_km(
    cos_theta: float | np.ndarray,
    depth_km: float,
    cap_km: float = MAX_UPSTREAM_KM,
) -> np.ndarray:
    """Length of medium a muon can be born in, upstream of the detector [km].

    A downgoing muon is produced in the medium between the surface and the
    detector, a path of ``depth / cos(theta)``. An upgoing muon comes through
    rock, which supplies far more column than any muon survives, so its
    length is effectively infinite and ``cap_km`` stands in for it.

    Parameters
    ----------
    cos_theta : float or np.ndarray
        Cosine of the arrival zenith; ``+1`` is overhead.
    depth_km : float
        Depth of the instrumented centre below the surface of the medium [km].
    cap_km : float, optional
        Longest length returned [km]; see :data:`MAX_UPSTREAM_KM`.

    Returns
    -------
    length_km : np.ndarray
        Available upstream length, as a length of the detector medium [km].
    """
    cos_theta = np.asarray(cos_theta, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        downgoing = np.where(cos_theta > 0.0, depth_km / np.maximum(cos_theta, 1.0e-6), np.inf)
    return np.minimum(downgoing, cap_km)


def neutrino_column_g_cm2(
    cos_theta: float | np.ndarray,
    depth_km: float,
    density_g_cm3: float,
    cap_km: float = MAX_UPSTREAM_KM,
) -> np.ndarray:
    """Column a neutrino traverses before reaching the detector [g cm^-2].

    Upgoing directions get the layered-PREM Earth chord of
    :func:`prem_column`, evaluated at a declination equal to the angle below
    the horizon. Downgoing directions get the medium above the detector,
    which is negligible except within a degree or so of the horizon, where
    the ``1 / cos(theta)`` path reaches ``10^7`` g cm^-2 and starts to matter
    above 100 PeV. Exactly at the horizon the column is zero.

    Parameters
    ----------
    cos_theta : float or np.ndarray
        Cosine of the arrival zenith; ``+1`` is overhead, ``-1`` the nadir.
    depth_km : float
        Depth of the instrumented centre below the surface of the medium [km].
    density_g_cm3 : float
        Density of the medium above the detector [g cm^-3].
    cap_km : float, optional
        Cap on the downgoing path; see :func:`overburden_km`.

    Returns
    -------
    column : np.ndarray
        Column depth [g cm^-2].
    """
    cos_theta = np.asarray(cos_theta, dtype=float)
    above = cos_theta > 0.0
    medium = overburden_km(cos_theta, depth_km, cap_km) * CM_PER_KM * density_g_cm3
    theta_deg = np.rad2deg(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    earth = np.array(
        [prem_column(float(t) - 90.0) if t > 90.0 else 0.0 for t in np.atleast_1d(theta_deg)]
    ).reshape(np.shape(theta_deg))
    return np.where(above, medium, earth)
