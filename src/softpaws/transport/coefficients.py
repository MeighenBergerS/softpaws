"""Muon drift and diffusion transport coefficients.

The soft expansion of the QED collision operator reduces muon energy loss to a
drift-diffusion process governed by two coefficients (arXiv:2607.13143,
Eq. 2.8): the drift ``b_mu`` [km^-1], the mean fractional energy loss per unit
length (so the muon range is ``1/b_mu``), and the diffusion ``d_mu`` [km^-1],
the variance of that loss.

This module provides the paper's Table 1 reference values in water and a simple
log-energy interpolation between them. Only ``b_mu`` enters the drift limit;
``d_mu`` is tabulated here for the later diffusion extension.
"""

from __future__ import annotations

import numpy as np

from ..utils.constants import RHO_WATER_G_CM3

# ---------------------------------------------------------------------------
# Table 1 of arXiv:2607.13143: total QED coefficients for muons in water
# (rho = 1.02 g/cm^3), at two reference energies.
# ---------------------------------------------------------------------------

_REF_LOG10_E = np.array([15.0, 17.0])  # log10(E / GeV) for 1 PeV and 100 PeV
_REF_B_MU = np.array([0.35, 0.40])  # drift [km^-1]
_REF_D_MU = np.array([0.0766, 0.0982])  # diffusion [km^-1]


def drift_coefficient(
    energy_gev: float | np.ndarray,
    density_g_cm3: float = RHO_WATER_G_CM3,
) -> np.ndarray:
    """Muon drift coefficient ``b_mu`` at a given energy.

    Parameters
    ----------
    energy_gev : float or np.ndarray
        Muon energy [GeV].
    density_g_cm3 : float, optional
        Target-medium density [g cm^-3]. Defaults to water. The coefficient
        scales linearly with density, since it is proportional to the target
        number density.

    Returns
    -------
    b_mu : np.ndarray
        Drift coefficient [km^-1].

    Notes
    -----
    Values are log-linearly interpolated in ``log10(E / GeV)`` between the two
    Table 1 reference energies (1 PeV and 100 PeV) and clipped to the endpoints
    outside that range. The energy dependence of the QED coefficients is slow
    (logarithmic), so this is adequate in the drift limit.
    """
    log10_e = np.log10(np.atleast_1d(np.asarray(energy_gev, dtype=float)))
    b_water = np.interp(log10_e, _REF_LOG10_E, _REF_B_MU)
    return b_water * (density_g_cm3 / RHO_WATER_G_CM3)


def diffusion_coefficient(
    energy_gev: float | np.ndarray,
    density_g_cm3: float = RHO_WATER_G_CM3,
) -> np.ndarray:
    """Muon diffusion coefficient ``d_mu`` at a given energy.

    Parameters
    ----------
    energy_gev : float or np.ndarray
        Muon energy [GeV].
    density_g_cm3 : float, optional
        Target-medium density [g cm^-3]. Defaults to water. Scales linearly
        with density.

    Returns
    -------
    d_mu : np.ndarray
        Diffusion coefficient [km^-1].

    Notes
    -----
    Interpolated as in :func:`drift_coefficient`. Not used in the drift limit;
    tabulated for the diffusion extension.
    """
    log10_e = np.log10(np.atleast_1d(np.asarray(energy_gev, dtype=float)))
    d_water = np.interp(log10_e, _REF_LOG10_E, _REF_D_MU)
    return d_water * (density_g_cm3 / RHO_WATER_G_CM3)
