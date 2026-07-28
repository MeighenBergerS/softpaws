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
formula and the associated figure of merit (Eq. 2.24).

It also provides the *exact* soft volume (:func:`soft_volume_exact`,
``docs/exact_soft_volume_notes.md``), which replaces the drift range ``1/(b_mu A)``
by ``I(A) (1 - e^{-Phi(A) x}) / Phi(A)`` with the exact eigenvalue ``Phi(A)`` and a
finite upstream column depth ``x``. The saturation factor keeps the result finite
where the drift form diverges (``A -> 0``) or goes negative (``A < 0``, the
cross-section pole).

Finally it provides the *range* target volume (:func:`range_target_volume_km3`),
which is a different quantity from all of the above and exists to match the
convention of the published IceCube effective area. The soft volume is
differential in the observed muon energy and already spectrally weighted: its
finite length ``1/(b_mu A)`` is a spectral attenuation length, set by how much
rarer the higher-energy parent neutrino is. The published ``A_eff(E_nu)`` instead
fixes the neutrino energy and integrates over every muon energy that survives the
event selection, so its length is the full muon range down to the analysis
threshold, which grows logarithmically with energy. Comparing the two directly
is an apples-to-oranges comparison; see ``examples/20_effective_area_soft_vs_irf.py``.
"""

from __future__ import annotations

import numpy as np

from ..utils.constants import RHO_WATER_G_CM3
from .coefficients import critical_energy_gev, diffusion_coefficient, drift_coefficient
from .eigenvalue import phi_eigenvalue, spectral_index
from .source import DEFAULT_LAMBDA, inelasticity_factor

# Muon energy below which a track no longer passes an IceCube-like through-going
# selection. Used as the lower limit of the muon range in
# :func:`range_target_volume_km3`; the resulting volume depends on it only
# logarithmically.
DEFAULT_MUON_THRESHOLD_GEV = 1.0e3


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
    b_scale: float = 1.0,
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
    b_scale : float, optional
        Multiplicative rescaling of the Table 1 drift coefficient ``b_mu``, used
        as a transport nuisance parameter in the data fits (Section 3). Defaults
        to 1 (the theoretical value).

    Returns
    -------
    v_soft : np.ndarray
        Soft volume [km^3].
    """
    a = spectral_penalty(gamma, lam)
    b_mu = b_scale * drift_coefficient(energy_gev, density_g_cm3)
    proj_area = np.pi * radius_km**2
    return proj_area / (b_mu * a)


def soft_volume_diffusion(
    radius_km: float,
    energy_gev: float | np.ndarray,
    gamma: float,
    lam: float = DEFAULT_LAMBDA,
    density_g_cm3: float = RHO_WATER_G_CM3,
    b_scale: float = 1.0,
    d_scale: float = 1.0,
) -> np.ndarray:
    """Diffusion-corrected soft volume (Eq. 2.25).

    The paper's drift-diffusion model multiplies the drift soft volume by the
    leading diffusion correction ``1 - d_mu / (2 b_mu)``,

    .. math:: V_\\mathrm{soft}^\\mathrm{diff}(E) = V_\\mathrm{soft}^\\mathrm{drift}(E)
        \\left(1 - \\frac{d_\\mu}{2 b_\\mu}\\right),

    an ``O(d_mu/2 b_mu) ~ 10%`` reduction. This is the ``method="diffusion"``
    forward model used to reproduce the diffusion contours of the paper's Fig. 6.

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
    b_scale, d_scale : float, optional
        Multiplicative rescalings of the Table 1 drift and diffusion coefficients,
        used as transport nuisance parameters (Section 3). Default to 1.

    Returns
    -------
    v_soft : np.ndarray
        Soft volume [km^3].
    """
    b_mu = b_scale * drift_coefficient(energy_gev, density_g_cm3)
    d_mu = d_scale * diffusion_coefficient(energy_gev, density_g_cm3)
    v_drift = soft_volume_drift(radius_km, energy_gev, gamma, lam, density_g_cm3, b_scale)
    return v_drift * (1.0 - d_mu / (2.0 * b_mu))


def muon_range_km(
    energy_gev: float | np.ndarray,
    threshold_gev: float = DEFAULT_MUON_THRESHOLD_GEV,
    density_g_cm3: float = RHO_WATER_G_CM3,
    b_scale: float = 1.0,
) -> np.ndarray:
    """Muon range from a starting energy down to a detection threshold.

    Integrating the continuous-slowing-down loss law ``-dE/dx = a_mu + b_mu E``
    from ``E`` down to ``E_thr`` gives

    .. math:: R(E \\to E_\\mathrm{thr}) = \\frac{1}{b_\\mu}
        \\ln\\frac{E + E_c}{E_\\mathrm{thr} + E_c},
        \\qquad E_c = a_\\mu / b_\\mu.

    Unlike the soft volume's spectral length ``1/(b_mu A)``, this carries no
    spectral weighting: it is the distance a muon of energy ``E`` can travel and
    still arrive above threshold, and it grows logarithmically with ``E``.

    Parameters
    ----------
    energy_gev : float or np.ndarray
        Muon energy at production [GeV].
    threshold_gev : float, optional
        Muon energy below which the track is not selected [GeV]. Defaults to
        :data:`DEFAULT_MUON_THRESHOLD_GEV`.
    density_g_cm3 : float, optional
        Target-medium density [g cm^-3]. Defaults to water.
    b_scale : float, optional
        Multiplicative rescaling of the Table 1 drift coefficient. Defaults to 1.

    Returns
    -------
    range_km : np.ndarray
        Muon range [km], clipped at zero for muons born below threshold.

    Notes
    -----
    ``b_mu`` is evaluated at the starting energy ``E`` rather than integrated
    along the track. Because ``b_mu`` moves by only 14% between 1 PeV and 100 PeV
    (Table 1), this is a percent-level approximation over the range of interest.
    """
    energy = np.atleast_1d(np.asarray(energy_gev, dtype=float))
    b_mu = b_scale * drift_coefficient(energy, density_g_cm3)
    e_crit = critical_energy_gev(energy, density_g_cm3, b_scale)
    ratio = (energy + e_crit) / (threshold_gev + e_crit)
    return np.clip(np.log(ratio) / b_mu, 0.0, None)


def range_target_volume_km3(
    radius_km: float,
    energy_gev: float | np.ndarray,
    threshold_gev: float = DEFAULT_MUON_THRESHOLD_GEV,
    density_g_cm3: float = RHO_WATER_G_CM3,
    b_scale: float = 1.0,
) -> np.ndarray:
    """Target volume for through-going tracks, in the muon-range convention.

    A muon of energy ``E`` produced anywhere in the upstream column
    ``pi R_det^2 R(E -> E_thr)`` reaches the detector above threshold, and the
    instrumented sphere itself contributes its own mean chord. Since
    ``pi R_det^2 (4 R_det / 3) = V_det`` exactly, the two add to

    .. math:: V_\\mathrm{range}(E) = \\pi R_\\mathrm{det}^2\\,
        R(E \\to E_\\mathrm{thr}) + V_\\mathrm{det},

    the same ``V_det + V_soft`` decomposition as the drift form, with the
    spectral length ``1/(b_mu A)`` replaced by the threshold range. This is the
    quantity that matches the convention of the published IceCube effective area,
    which is tabulated at fixed neutrino energy and integrated over all selected
    muon energies.

    Parameters
    ----------
    radius_km : float
        Radius of the spherical detector [km].
    energy_gev : float or np.ndarray
        Muon energy at production [GeV].
    threshold_gev : float, optional
        Muon selection threshold [GeV]. Defaults to
        :data:`DEFAULT_MUON_THRESHOLD_GEV`.
    density_g_cm3 : float, optional
        Target-medium density [g cm^-3]. Defaults to water.
    b_scale : float, optional
        Multiplicative rescaling of the Table 1 drift coefficient. Defaults to 1.

    Returns
    -------
    volume : np.ndarray
        Target volume [km^3], broadcast to the shape of ``energy_gev``.

    Notes
    -----
    Detection efficiency is not included: this is the geometric ceiling that a
    perfect through-going selection with threshold ``E_thr`` would reach. The
    ratio of the published ``A_eff`` to this volume is therefore an estimate of
    the selection efficiency (see ``examples/20_effective_area_soft_vs_irf.py``).

    A muon born below threshold is never selected, so the volume drops to zero
    rather than to ``V_det`` for ``E <= E_thr``.
    """
    proj_area = np.pi * radius_km**2
    v_det = 4.0 / 3.0 * np.pi * radius_km**3
    range_km = muon_range_km(energy_gev, threshold_gev, density_g_cm3, b_scale)
    return np.where(range_km > 0.0, proj_area * range_km + v_det, 0.0)


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


def saturation_factor(
    phi_per_km: float | np.ndarray,
    column_depth_km: float,
) -> np.ndarray:
    """Effective attenuated range ``(1 - e^{-Phi x}) / Phi`` over a finite column.

    This is the geometric factor that turns the infinite-column range ``1/Phi``
    into the value delivered by an upstream column of depth ``x``
    (``docs/exact_soft_volume_notes.md`` Part 7). It is finite for every sign of
    ``Phi``: as ``Phi -> 0`` it tends to ``x`` (no losses, the whole column
    accumulates), and for ``Phi < 0`` it continues to
    ``(e^{|Phi| x} - 1) / |Phi|``, curing the drift-form divergence at and beyond
    the cross-section pole.

    Parameters
    ----------
    phi_per_km : float or np.ndarray
        Collision eigenvalue ``Phi(A)`` [km^-1] (see
        :func:`softpaws.transport.eigenvalue.phi_eigenvalue`).
    column_depth_km : float
        Available upstream column depth ``x`` [km].

    Returns
    -------
    eff_range : np.ndarray
        ``(1 - e^{-Phi x}) / Phi`` [km].
    """
    phi = np.asarray(phi_per_km, dtype=float)
    x = float(column_depth_km)
    small = np.abs(phi) < 1e-12
    safe_phi = np.where(small, 1.0, phi)
    # -expm1(-Phi x) = 1 - e^{-Phi x}, evaluated stably for both signs of Phi.
    eff_range = -np.expm1(-safe_phi * x) / safe_phi
    return np.where(small, x, eff_range)


def soft_volume_exact(
    radius_km: float,
    energy_gev: float | np.ndarray,
    gamma: float,
    lam: float = DEFAULT_LAMBDA,
    column_depth_km: float | None = None,
    density_g_cm3: float = RHO_WATER_G_CM3,
    include_inelasticity: bool = True,
    b_scale: float = 1.0,
    d_scale: float = 1.0,
) -> np.ndarray:
    """Exact soft volume with the eigenvalue ``Phi(A)`` and a finite column.

    Implements ``docs/exact_soft_volume_notes.md`` Part 7:

    .. math:: V_\\mathrm{soft}(E) = \\pi R_\\mathrm{det}^2\\,
        \\mathcal{I}(A)\\,\\frac{1 - e^{-\\Phi(A) x}}{\\Phi(A)},

    with ``Phi(A)`` the exact eigenvalue (:func:`phi_eigenvalue`) and
    ``I(A) ~ 0.8`` the inelasticity factor. Compared with
    :func:`soft_volume_drift`, this replaces the leading eigenvalue ``b_mu A`` by
    the full ``Phi(A)``, keeps the ``I(A)`` normalization, and applies the
    finite-column saturation factor.

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
    column_depth_km : float or None, optional
        Available upstream column depth ``x`` [km]. If ``None`` (the default) the
        infinite-column limit ``1/Phi(A)`` is used, which requires ``Phi(A) > 0``.
    density_g_cm3 : float, optional
        Target-medium density [g cm^-3]. Defaults to water.
    include_inelasticity : bool, optional
        Whether to fold in ``I(A)``. Set ``False`` for a geometry-only volume that
        is directly comparable with :func:`soft_volume_drift`.
    b_scale, d_scale : float, optional
        Multiplicative rescalings of the Table 1 drift and diffusion coefficients,
        used as transport nuisance parameters (Section 3). Default to 1.

    Returns
    -------
    v_soft : np.ndarray
        Soft volume [km^3].

    Raises
    ------
    ValueError
        Raised if ``column_depth_km is None`` while ``Phi(A) <= 0``, where the
        infinite-column limit diverges; pass a finite ``column_depth_km`` instead.
    """
    a = spectral_index(gamma, lam)
    b_mu = b_scale * drift_coefficient(energy_gev, density_g_cm3)
    d_mu = d_scale * diffusion_coefficient(energy_gev, density_g_cm3)
    phi = phi_eigenvalue(a, b_mu, d_mu)
    proj_area = np.pi * radius_km**2

    if column_depth_km is None:
        if np.any(phi <= 0.0):
            raise ValueError(
                f"Phi(A) <= 0 for A = {a:.3f}: the infinite-column soft volume "
                "diverges. Pass a finite column_depth_km to use the saturation factor."
            )
        eff_range = 1.0 / phi
    else:
        eff_range = saturation_factor(phi, column_depth_km)

    factor = inelasticity_factor(a) if include_inelasticity else 1.0
    return proj_area * factor * eff_range
