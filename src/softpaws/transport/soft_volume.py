"""Soft volume in the drift limit.

The soft volume is the effective target region for through-going muon tracks: a
muon produced outside the instrumented volume can still drift into it, so the
target is larger than the detector itself (arXiv:2607.13143, Section 2.3).

In the drift limit (``d_mu -> 0``, deterministic continuous slowing down) and
with the simplifying assumptions of Section 2.3 -- a spherical perfectly
absorbing detector, a power-law neutrino flux ``phi ~ E^-gamma``, a power-law CC
cross section ``sigma ~ E^lambda``, and a constant near-detector medium -- the
soft volume has the closed form (Eq. 2.23)

.. math:: V_\\mathrm{soft}(E) = \\frac{\\pi R_\\mathrm{det}^2}{b_\\mu(E)\\,A},
    \\qquad A \\equiv \\gamma - \\lambda - 1 > 0.

The muon range ``1/b_mu`` is further weighted by the spectral penalty ``1/A``
for producing a higher-energy parent neutrino. This module implements that
formula and the associated figure of merit (Eq. 2.24). The full Green's-function
convolution and the diffusion correction are deferred to later modules.
"""

from __future__ import annotations

import numpy as np

from ..utils.constants import RHO_WATER_G_CM3
from .coefficients import drift_coefficient
from .source import DEFAULT_LAMBDA


def spectral_penalty(gamma: float, lam: float = DEFAULT_LAMBDA) -> float:
    """Spectral penalty exponent ``A = gamma - lambda - 1``.

    Parameters
    ----------
    gamma : float
        Neutrino flux spectral index, ``phi_nu ~ E^-gamma``.
    lam : float, optional
        CC cross-section slope, ``sigma_CC ~ E^lambda``. Defaults to
        :data:`DEFAULT_LAMBDA`.

    Returns
    -------
    A : float
        Penalty exponent ``gamma - lambda - 1``.

    Raises
    ------
    ValueError
        Raised if ``A <= 0``, where the soft-volume integral (Eq. 2.22)
        diverges and the drift closed form does not apply.
    """
    a = gamma - lam - 1.0
    if a <= 0.0:
        raise ValueError(
            f"Spectral penalty A = gamma - lambda - 1 = {a:.3f} must be positive; "
            f"the soft-volume integral diverges for gamma <= lambda + 1."
        )
    return a


def sphere_radius_from_volume(volume_km3: float) -> float:
    """Radius of a sphere with the given volume.

    Parameters
    ----------
    volume_km3 : float
        Sphere volume [km^3].

    Returns
    -------
    radius : float
        Sphere radius [km].
    """
    return (3.0 * volume_km3 / (4.0 * np.pi)) ** (1.0 / 3.0)


def soft_volume_drift(
    radius_km: float,
    energy_gev: float | np.ndarray,
    gamma: float,
    lam: float = DEFAULT_LAMBDA,
    density_g_cm3: float = RHO_WATER_G_CM3,
) -> np.ndarray:
    """Drift-limit soft volume as a function of muon energy (Eq. 2.23).

    Parameters
    ----------
    radius_km : float
        Radius of the spherical detector [km].
    energy_gev : float or np.ndarray
        Observed muon energy [GeV].
    gamma : float
        Neutrino flux spectral index, ``phi_nu ~ E^-gamma``.
    lam : float, optional
        CC cross-section slope. Defaults to :data:`DEFAULT_LAMBDA`.
    density_g_cm3 : float, optional
        Target-medium density [g cm^-3]. Defaults to water.

    Returns
    -------
    v_soft : np.ndarray
        Soft volume [km^3].
    """
    a = spectral_penalty(gamma, lam)
    b_mu = drift_coefficient(energy_gev, density_g_cm3)
    proj_area = np.pi * radius_km**2
    return proj_area / (b_mu * a)


def volume_ratio_drift(
    radius_km: float,
    energy_gev: float | np.ndarray,
    gamma: float,
    lam: float = DEFAULT_LAMBDA,
    density_g_cm3: float = RHO_WATER_G_CM3,
) -> np.ndarray:
    """Total-to-detector volume ratio in the drift limit (Eq. 2.24).

    The total target volume is the instrumented sphere plus its soft volume,
    ``V_tot / V_det = 1 + 3 / (4 R_det b_mu A)``. For the IceCube diffuse flux
    (``A ~ 1``) this recovers the paper's figure of merit
    ``1 + 3 / (4 R_det b_mu)``.

    Parameters
    ----------
    radius_km : float
        Radius of the spherical detector [km].
    energy_gev : float or np.ndarray
        Observed muon energy [GeV].
    gamma : float
        Neutrino flux spectral index, ``phi_nu ~ E^-gamma``.
    lam : float, optional
        CC cross-section slope. Defaults to :data:`DEFAULT_LAMBDA`.
    density_g_cm3 : float, optional
        Target-medium density [g cm^-3]. Defaults to water.

    Returns
    -------
    ratio : np.ndarray
        Dimensionless ``V_tot / V_det``.
    """
    a = spectral_penalty(gamma, lam)
    b_mu = drift_coefficient(energy_gev, density_g_cm3)
    return 1.0 + 3.0 / (4.0 * radius_km * b_mu * a)
