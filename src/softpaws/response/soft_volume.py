"""Soft-volume forward model: neutrino flux to muon track rate (drift limit).

This is the soft-volume counterpart to the IRF path in :mod:`softpaws.response.irfs`.
Both map an incident neutrino flux to a predicted track rate; keeping them
interchangeable is what enables the head-to-head comparison.

In the drift limit with the Section 2.3 approximations, the master formula
(arXiv:2607.13143, Eq. 2.20) factorizes: the differential track rate at observed
muon energy ``E`` is the target volume times the local weak-rate density,

.. math:: \\frac{dN}{dt\\,dE\\,d\\Omega}
    = \\bigl[V_\\mathrm{det} + V_\\mathrm{soft}(E)\\bigr]\\,
      n_N\\,\\sigma_\\mathrm{CC}(E)\\,\\phi_\\nu(E),

where the soft volume ``V_soft`` carries the muon-transport enhancement (Eq. 2.23)
and ``V_det`` is the instrumented sphere. The drift closed form assumes a
power-law neutrino flux, so this model is parametrized directly by ``(phi0, gamma)``.
"""

from __future__ import annotations

import numpy as np

from ..transport.soft_volume import soft_volume_drift
from ..transport.source import DEFAULT_LAMBDA, cc_cross_section, nucleon_number_density
from ..utils.constants import CM_PER_KM, RHO_WATER_G_CM3

# Flux pivot energy for the power-law parametrization (Eq. 4.1): 100 TeV.
FLUX_PIVOT_GEV = 1.0e5


def power_law_flux(
    energy_gev: float | np.ndarray,
    phi0: float,
    gamma: float,
) -> np.ndarray:
    """Single power-law diffuse neutrino flux (Eq. 4.1).

    Parameters
    ----------
    energy_gev : float or np.ndarray
        Neutrino energy [GeV].
    phi0 : float
        Flux normalization in units of ``1e-18 GeV^-1 cm^-2 s^-1 sr^-1`` at the
        100 TeV pivot; of order unity for typical diffuse fluxes.
    gamma : float
        Spectral index, ``phi_nu ~ E^-gamma``.

    Returns
    -------
    flux : np.ndarray
        Differential flux [GeV^-1 cm^-2 s^-1 sr^-1].
    """
    energy = np.atleast_1d(np.asarray(energy_gev, dtype=float))
    return (phi0 * 1.0e-18) * (energy / FLUX_PIVOT_GEV) ** (-gamma)


class SoftVolumeResponse:
    """Drift-limit soft-volume forward model for a spherical detector.

    Parameters
    ----------
    radius_km : float
        Radius of the spherical instrumented volume [km].
    density_g_cm3 : float, optional
        Target-medium density [g cm^-3]. Defaults to water.

    Attributes
    ----------
    radius_km : float
        Detector radius [km].
    density_g_cm3 : float
        Medium density [g cm^-3].
    n_nucleon_cm3 : float
        Target nucleon number density [cm^-3].
    v_det_cm3 : float
        Instrumented sphere volume [cm^3].

    Examples
    --------
    >>> resp = SoftVolumeResponse(radius_km=0.62)
    >>> rate = resp.differential_rate(1.0e6, phi0=0.63, gamma=2.38)
    >>> float(rate[0]) > 0
    True
    """

    def __init__(
        self,
        radius_km: float,
        density_g_cm3: float = RHO_WATER_G_CM3,
    ) -> None:
        self.radius_km = radius_km
        self.density_g_cm3 = density_g_cm3
        self.n_nucleon_cm3 = nucleon_number_density(density_g_cm3)
        radius_cm = radius_km * CM_PER_KM
        self.v_det_cm3 = 4.0 / 3.0 * np.pi * radius_cm**3

    def target_volume_cm3(
        self,
        energy_gev: float | np.ndarray,
        gamma: float,
        lam: float = DEFAULT_LAMBDA,
        part: str = "total",
    ) -> np.ndarray:
        """Effective target volume at a given muon energy.

        Parameters
        ----------
        energy_gev : float or np.ndarray
            Observed muon energy [GeV].
        gamma : float
            Neutrino flux spectral index.
        lam : float, optional
            CC cross-section slope. Defaults to :data:`DEFAULT_LAMBDA`.
        part : {"total", "soft", "inside"}, optional
            Which contribution to return. ``"inside"`` is the instrumented
            sphere, ``"soft"`` the transport-enhanced soft volume, ``"total"``
            their sum.

        Returns
        -------
        volume : np.ndarray
            Target volume [cm^3], broadcast to the shape of ``energy_gev``.
        """
        v_soft = soft_volume_drift(self.radius_km, energy_gev, gamma, lam, self.density_g_cm3)
        v_soft_cm3 = v_soft * CM_PER_KM**3
        if part == "soft":
            return v_soft_cm3
        if part == "inside":
            return np.full_like(v_soft_cm3, self.v_det_cm3)
        if part == "total":
            return self.v_det_cm3 + v_soft_cm3
        raise ValueError(f"part must be 'total', 'soft', or 'inside', got {part!r}.")

    def weak_rate_density(
        self,
        energy_gev: float | np.ndarray,
        phi0: float,
        gamma: float,
        lam: float = DEFAULT_LAMBDA,
    ) -> np.ndarray:
        """Local weak-rate density ``n_N sigma_CC(E) phi_nu(E)``.

        Parameters
        ----------
        energy_gev : float or np.ndarray
            Observed muon energy [GeV] (approximately the neutrino energy).
        phi0, gamma : float
            Power-law flux parameters (see :func:`power_law_flux`).
        lam : float, optional
            CC cross-section slope. Defaults to :data:`DEFAULT_LAMBDA`.

        Returns
        -------
        rate_density : np.ndarray
            Weak-rate density [cm^-3 GeV^-1 s^-1 sr^-1].
        """
        sigma = cc_cross_section(energy_gev, lam)
        flux = power_law_flux(energy_gev, phi0, gamma)
        return self.n_nucleon_cm3 * sigma * flux

    def differential_rate(
        self,
        energy_gev: float | np.ndarray,
        phi0: float,
        gamma: float,
        lam: float = DEFAULT_LAMBDA,
        part: str = "total",
    ) -> np.ndarray:
        """Differential track rate ``dN / (dt dE dOmega)`` (Eq. 2.20).

        Parameters
        ----------
        energy_gev : float or np.ndarray
            Observed muon energy [GeV].
        phi0, gamma : float
            Power-law flux parameters (see :func:`power_law_flux`).
        lam : float, optional
            CC cross-section slope. Defaults to :data:`DEFAULT_LAMBDA`.
        part : {"total", "soft", "inside"}, optional
            Which target-volume contribution to use.

        Returns
        -------
        rate : np.ndarray
            Differential track rate [GeV^-1 s^-1 sr^-1].
        """
        volume = self.target_volume_cm3(energy_gev, gamma, lam, part)
        return volume * self.weak_rate_density(energy_gev, phi0, gamma, lam)

    def expected_counts(
        self,
        log10_energy_edges: np.ndarray,
        phi0: float,
        gamma: float,
        livetime_s: float,
        solid_angle_sr: float,
        lam: float = DEFAULT_LAMBDA,
        part: str = "total",
        n_subdivisions: int = 64,
    ) -> np.ndarray:
        """Expected track counts per muon-energy bin.

        Integrates the differential rate over each energy bin and over the given
        solid angle and livetime. Angular acceptance, attenuation, and detector
        efficiency are not modelled (see ``docs/soft_volume_notes.md``); the
        result is a geometric through-going estimate.

        Parameters
        ----------
        log10_energy_edges : np.ndarray, shape (n_bins + 1,)
            Muon-energy bin edges in ``log10(E / GeV)``.
        phi0, gamma : float
            Power-law flux parameters (see :func:`power_law_flux`).
        livetime_s : float
            Exposure time [s].
        solid_angle_sr : float
            Solid angle over which the (isotropic) flux is integrated [sr].
        lam : float, optional
            CC cross-section slope. Defaults to :data:`DEFAULT_LAMBDA`.
        part : {"total", "soft", "inside"}, optional
            Which target-volume contribution to use.
        n_subdivisions : int, optional
            Number of log-spaced sample points per bin for the energy integral.

        Returns
        -------
        counts : np.ndarray, shape (n_bins,)
            Expected number of tracks in each energy bin.
        """
        edges = np.asarray(log10_energy_edges, dtype=float)
        counts = np.empty(len(edges) - 1)
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            energy = np.logspace(lo, hi, n_subdivisions)
            rate = self.differential_rate(energy, phi0, gamma, lam, part)
            counts[i] = np.trapezoid(rate, energy)
        return counts * livetime_s * solid_angle_sr
