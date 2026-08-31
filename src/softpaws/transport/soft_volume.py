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

:func:`soft_volume_attenuated_exact` (Eq. 11, App. C.2-C.3 of
``docs/2026_softvolume.pdf``) is the further generalization that folds parent-
neutrino Earth attenuation directly into that same depth integral, rather than
applying a decoupled multiplicative survival probability
(:mod:`softpaws.transport.attenuation`) on top of it -- see
:meth:`softpaws.response.soft_volume.SoftVolumeResponse.expected_counts_coupled_attenuation`.

:func:`scale_breaking_saturation_factor` implements App. F's leading-order
scale-breaking correction (Eq. F4): a nonzero ``beta`` (``dGamma/dy ~ E^beta``,
e.g. the slow photonuclear rise) makes the effective spectral index run with
propagated depth rather than staying fixed at ``A``. Off by default (``beta
= 0``), since ``coefficients.py`` finds the corresponding LPM suppression
negligible for muons in the covered range and this package ships no
calibrated nonzero ``beta``.

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
from scipy.special import gammainc, polygamma

from ..utils.constants import RHO_WATER_G_CM3
from .coefficients import (
    DEFAULT_SOURCE,
    critical_energy_gev,
    diffusion_coefficient,
    drift_coefficient,
    ionization_coefficient,
    log_loss_moments,
)
from .eigenvalue import (
    phi_eigenvalue,
    phi_eigenvalue_derivative,
    spectral_index,
    two_moment_loss_spectrum,
)
from .source import DEFAULT_LAMBDA, inelasticity_factor

# Muon energy below which a track no longer passes an IceCube-like through-going
# selection. Used as the lower limit of the muon range in
# :func:`range_target_volume_km3`; the resulting volume depends on it only
# logarithmically.
DEFAULT_MUON_THRESHOLD_GEV = 1.0e3

# Matching energy between the two regimes of :func:`stochastic_muon_range_km`:
# radiative and stochastic above, deterministic and ionizing below. It has to sit
# well above the critical energy ``E_c ~ 600`` GeV, where the scale-invariant
# kernel that the first-passage derivation assumes stops describing the losses,
# and low enough that the radiative treatment still covers most of the range.
# 10 TeV is 17 E_c and leaves one decade to the default threshold.
DEFAULT_IONIZATION_MATCH_GEV = 1.0e4


def spectral_penalty(
    gamma: float,
    lam: float | np.ndarray = DEFAULT_LAMBDA,
) -> float | np.ndarray:
    """Spectral penalty exponent ``A = gamma - lambda - 1``.

    Parameters
    ----------
    gamma : float
        Neutrino flux spectral index, ``phi_nu ~ E^-gamma``.
    lam : float or np.ndarray, optional
        CC cross-section slope, ``sigma_CC ~ E^lambda``. Defaults to
        :data:`DEFAULT_LAMBDA`. An array is the local slope of a tabulated
        cross section (:meth:`~softpaws.transport.cross_section.CrossSection.
        local_slope`), one value per energy.

    Returns
    -------
    A : float or np.ndarray
        Penalty exponent ``gamma - lambda - 1``.

    Raises
    ------
    ValueError
        Raised if ``A <= 0`` anywhere, where the soft-volume integral
        (Eq. 2.22) diverges and the drift closed form does not apply.
    """
    a = gamma - lam - 1.0
    if np.any(a <= 0.0):
        raise ValueError(
            f"Spectral penalty A = gamma - lambda - 1 = {np.min(a):.3f} must be positive; "
            f"the soft-volume integral diverges for gamma <= lambda + 1."
        )
    return a


def dynamic_projected_radius_km(
    radius_km: float,
    energy_gev: float | np.ndarray,
    light_yield_length_km: float,
    density_g_cm3: float = RHO_WATER_G_CM3,
    b_scale: float = 1.0,
    source: str = DEFAULT_SOURCE,
) -> np.ndarray:
    """Energy-growing lateral trigger radius from stochastic light yield.

    Not part of arXiv:2607.13143: the paper's soft volume uses a fixed
    projected area ``pi R_det^2``, which is only right if a muon's ability to
    trigger the detector from a given lateral distance doesn't depend on its
    energy. Above the critical energy ``E_c`` (:func:`critical_energy_gev`)
    radiative losses -- bremsstrahlung, pair production, photonuclear --
    start dominating the loss rate, and the resulting stochastic light
    output lets a track trigger strings from beyond ``R_det``. This
    parametrizes that reach as growing logarithmically with energy above
    ``E_c``,

    .. math:: R_\\mathrm{eff}(E) = R_\\mathrm{det} + L\\,
        \\max\\!\\bigl[0,\\ \\ln(E / E_c(E))\\bigr],

    clipped to ``R_det`` at and below ``E_c`` so the term is inactive for
    TeV-scale (ionization-dominated) muons. ``E_c`` is reused rather than
    refit, so the only new phenomenological parameter is the slope ``L``.

    Parameters
    ----------
    radius_km : float
        Static detector radius ``R_det`` [km].
    energy_gev : float or np.ndarray
        Muon energy [GeV].
    light_yield_length_km : float
        Growth length ``L`` [km] per e-fold of energy above ``E_c`` -- a
        phenomenological nuisance parameter, not a derived quantity (see
        ``examples/26_dynamic_response_effective_area.py``, which fits it
        against the published IceCube effective area rather than asserting a
        value).
    density_g_cm3 : float, optional
        Target-medium density [g cm^-3]. Defaults to water.
    b_scale : float, optional
        Multiplicative rescaling of the drift coefficient feeding ``E_c``.
        Defaults to 1.
    source : {"proposal", "table1"}, optional
        Transport-coefficient tabulation feeding ``E_c``; see
        :mod:`softpaws.transport.coefficients`.

    Returns
    -------
    r_eff : np.ndarray
        Effective projected radius [km], broadcast to the shape of
        ``energy_gev``.
    """
    energy = np.atleast_1d(np.asarray(energy_gev, dtype=float))
    e_crit = critical_energy_gev(energy, density_g_cm3, b_scale, source)
    growth = np.clip(np.log(energy / e_crit), 0.0, None)
    return radius_km + light_yield_length_km * growth


def dynamic_projected_area_km2(
    radius_km: float,
    energy_gev: float | np.ndarray,
    light_yield_length_km: float,
    density_g_cm3: float = RHO_WATER_G_CM3,
    b_scale: float = 1.0,
    source: str = DEFAULT_SOURCE,
) -> np.ndarray:
    """Energy-growing projected area ``pi R_eff(E)^2``.

    See :func:`dynamic_projected_radius_km` for the growth law and its
    physical motivation.

    Parameters
    ----------
    radius_km : float
        Static detector radius ``R_det`` [km].
    energy_gev : float or np.ndarray
        Muon energy [GeV].
    light_yield_length_km : float
        Growth length ``L`` [km] per e-fold of energy above ``E_c``.
    density_g_cm3 : float, optional
        Target-medium density [g cm^-3]. Defaults to water.
    b_scale : float, optional
        Multiplicative rescaling of the drift coefficient feeding ``E_c``.
        Defaults to 1.
    source : {"proposal", "table1"}, optional
        Transport-coefficient tabulation feeding ``E_c``; see
        :mod:`softpaws.transport.coefficients`.

    Returns
    -------
    area : np.ndarray
        Effective projected area [km^2], broadcast to the shape of
        ``energy_gev``.
    """
    r_eff = dynamic_projected_radius_km(
        radius_km, energy_gev, light_yield_length_km, density_g_cm3, b_scale, source,
    )
    return np.pi * r_eff**2


def prism_projected_area_km2(
    cos_theta: float | np.ndarray,
    radius_km: float | np.ndarray,
    height_km: float | np.ndarray,
    n_sides: int | None = 6,
    n_blocks: int = 1,
) -> np.ndarray:
    """Projected area of an upright convex prism, averaged over azimuth.

    A sphere presents ``pi R^2`` from every direction, which is convenient and
    wrong for an array that is as wide as it is tall. This is the projection of
    an upright prism of regular ``n_sides`` cross-section,

    .. math:: A_\\mathrm{proj}(\\theta_z) = \\pi R^2 |\\cos\\theta_z|
        + \\frac{P}{\\pi} h \\sin\\theta_z,

    with ``R`` the area-equivalent radius of the cross-section, so that the
    footprint is ``pi R^2`` and the instrumented volume ``pi R^2 h`` whatever
    ``n_sides`` is, and ``P = 2 R sqrt(pi n tan(pi/n))`` its perimeter. The
    ``P / pi`` in the side term is the azimuthal average of the silhouette width
    of a convex cross-section, which for a circle (``n_sides=None``) returns the
    ``2 R h sin theta`` of the cylinder form used for the ARCA blocks.

    Two consequences are worth stating because they are easy to guess wrongly.
    The two terms *add* at oblique incidence, so the maximum is neither face-on
    value: a 1 km^2 by 1 km hexagonal prism presents 1.00 km^2 vertically and
    1.19 km^2 horizontally but 1.55 km^2 at ``theta_z = 50``. And its
    direction-averaged area is ``S/4 = 1.43`` km^2 by Cauchy's formula, above
    the 1.21 km^2 of the equal-volume sphere, because the sphere minimises
    surface area at fixed volume and therefore also minimises mean projected
    area. Replacing a sphere by any equal-volume prism raises the ceiling.

    Parameters
    ----------
    cos_theta : float or np.ndarray
        Cosine of the arrival zenith angle. Only its magnitude is used, so
        upgoing and downgoing arrivals of the same obliquity agree.
    radius_km : float or np.ndarray
        Area-equivalent radius of the cross-section [km], broadcast against
        ``cos_theta``. An energy-dependent radius is how the reach law of
        :func:`light_reach_radius_km` enters.
    height_km : float or np.ndarray
        Instrumented height of one prism [km], broadcast against ``cos_theta``
        and ``radius_km``. An energy-dependent height is how a light reach that
        extends the boundary vertically as well as radially enters.
    n_sides : int or None, optional
        Sides of the regular cross-section; 6 (the default) for an IceCube-like
        hexagonal footprint, ``None`` for a circular one. A hexagon has a 5%
        longer perimeter than the circle of equal area, which is the whole of
        its effect on this expression.
    n_blocks : int, optional
        Number of identical prisms. Defaults to 1.

    Returns
    -------
    area : np.ndarray
        Projected area [km^2], broadcast over ``cos_theta`` and ``radius_km``.

    Raises
    ------
    ValueError
        Raised if ``n_sides`` is given and is less than 3.
    """
    if n_sides is not None and n_sides < 3:
        raise ValueError(f"n_sides must be at least 3 or None for a circle, got {n_sides}.")
    cos_abs = np.abs(np.asarray(cos_theta, dtype=float))
    sin_theta = np.sqrt(np.clip(1.0 - cos_abs**2, 0.0, 1.0))
    radius = np.asarray(radius_km, dtype=float)
    if n_sides is None:
        perimeter = 2.0 * np.pi * radius
    else:
        perimeter = 2.0 * radius * np.sqrt(np.pi * n_sides * np.tan(np.pi / n_sides))
    cap = np.pi * radius**2 * cos_abs
    side = perimeter / np.pi * np.asarray(height_km, dtype=float) * sin_theta
    return n_blocks * (cap + side)


def eroded_prism_target_km2(
    cos_theta: float | np.ndarray,
    radius_km: float | np.ndarray,
    height_km: float | np.ndarray,
    min_chord_km: float = 0.0,
    n_sides: int | None = 6,
    n_blocks: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Target area and volume of a prism that a track must cross for ``min_chord_km``.

    :func:`prism_projected_area_km2` counts every line that touches the body,
    including the ones that clip a corner and leave again. A reconstructed track
    is not made that way: it needs a lever arm inside the instrumented volume,
    and a selection that demands one throws the clipping tracks away. Imposing a
    minimum in-detector path ``l`` is therefore a *selection* statement with a
    geometric consequence, and the consequence is the whole of what it does here.

    The lines whose chord through a convex body ``K`` exceeds ``l`` are exactly
    the lines meeting the erosion ``K n (K - l n)``, so both the area and the
    volume follow from that one body. For an upright prism the erosion factorizes:
    the cross-section becomes the lens of two copies offset by ``l sin(theta)``,
    and the height falls to ``h - l cos(theta)``. Hence

    .. math:: A(\\theta) = A_{\\rm lens}\\,|\\cos\\theta|
        + w_{\\rm lens}\\,(h - l|\\cos\\theta|)\\,\\sin\\theta,
        \\qquad V(\\theta) = A_{\\rm lens}\\,(h - l|\\cos\\theta|),

    with ``A_lens`` and ``w_lens`` the area and the perpendicular width of the
    lens. ``V`` is the volume in which a vertex still leaves ``l`` of track
    before the muon exits, which is the right instrumented term to pair with
    ``A``: entering tracks need a chord, starting ones need a remaining path.

    The erosion is strongest at oblique incidence, which is where the intact
    prism's cap and side terms add. That is the whole reason it matters: it
    removes the oblique enhancement that a published declination dependence does
    not carry, and takes the direction-averaged area down towards the
    equal-volume sphere's.

    Parameters
    ----------
    cos_theta : float or np.ndarray
        Cosine of the arrival zenith angle; only its magnitude is used.
    radius_km : float or np.ndarray
        Area-equivalent radius of the cross-section [km].
    height_km : float or np.ndarray
        Instrumented height of one prism [km].
    min_chord_km : float, optional
        Minimum path inside the instrumented volume [km]. Defaults to 0, which
        returns :func:`prism_projected_area_km2` and the intact volume exactly.
    n_sides : int or None, optional
        Sides of the regular cross-section; ``None`` for a circle. The lens is
        computed for the circle of equal area and the perimeter enters only
        through the side term, which is the same Cauchy-level treatment of the
        cross-section shape that :func:`prism_projected_area_km2` makes.
    n_blocks : int, optional
        Number of identical prisms. Defaults to 1.

    Returns
    -------
    area_km2 : np.ndarray
        Projected area of the eroded body [km^2].
    volume_km3 : np.ndarray
        Volume of the eroded body [km^3].

    Raises
    ------
    ValueError
        Raised if ``n_sides`` is given and is less than 3, or if
        ``min_chord_km`` is negative.

    Examples
    --------
    >>> a, v = eroded_prism_target_km2(0.5, 0.5642, 1.0, 0.0)
    >>> bool(np.isclose(a, prism_projected_area_km2(0.5, 0.5642, 1.0)))
    True
    """
    if n_sides is not None and n_sides < 3:
        raise ValueError(f"n_sides must be at least 3 or None for a circle, got {n_sides}.")
    if min_chord_km < 0.0:
        raise ValueError(f"min_chord_km must be non-negative, got {min_chord_km}.")
    cos_abs = np.abs(np.asarray(cos_theta, dtype=float))
    sin_theta = np.sqrt(np.clip(1.0 - cos_abs**2, 0.0, 1.0))
    radius = np.asarray(radius_km, dtype=float)
    height = np.asarray(height_km, dtype=float)
    ell = float(min_chord_km)

    if n_sides is None:
        perimeter = 2.0 * np.pi * radius
    else:
        perimeter = 2.0 * radius * np.sqrt(np.pi * n_sides * np.tan(np.pi / n_sides))
    # Cauchy's mean width, carried as a correction on the circle's 2R so that
    # the l -> 0 limit reproduces prism_projected_area_km2 for any cross-section.
    with np.errstate(divide="ignore", invalid="ignore"):
        width_factor = np.where(
            radius > 0.0, perimeter / (2.0 * np.pi * np.maximum(radius, 1e-12)), 1.0)

    offset = np.clip(ell * sin_theta, 0.0, 2.0 * radius)
    half = np.clip(0.5 * offset / np.maximum(radius, 1e-12), 0.0, 1.0)
    lens = (2.0 * radius**2 * np.arccos(half)
            - 0.5 * offset * np.sqrt(np.clip(4.0 * radius**2 - offset**2, 0.0, None)))
    width = 2.0 * np.sqrt(np.clip(radius**2 - 0.25 * offset**2, 0.0, None)) * width_factor
    eroded_height = np.clip(height - ell * cos_abs, 0.0, None)

    area = n_blocks * (lens * cos_abs + width * eroded_height * sin_theta)
    volume = n_blocks * lens * eroded_height
    return area, volume


def light_reach_radius_km(
    radius_km: float,
    energy_gev: float | np.ndarray,
    reach_km: float,
    pivot_gev: float,
) -> np.ndarray:
    """Instrumented radius plus a signed, logarithmically growing light reach.

    :func:`dynamic_projected_radius_km` clips its growth at ``R_det``, so it can
    only ever describe a detector responding to *more* than its footprint. That
    forbids the opposite regime, which is just as real: while the trigger is
    still turning on, a dim track lights too few modules to use the whole array
    and the detector responds to *less* than its footprint. Dropping the clip
    and letting the reach change sign covers both with one mechanism,

    .. math:: R_\\mathrm{eff}(E) = \\max\\!\\left[0,\\;
        R_\\mathrm{det} + \\Lambda \\ln(E / E_\\mathrm{piv})\\right],

    with ``Lambda`` the growth per e-fold and ``E_piv`` the energy at which the
    detector responds to exactly its own footprint. Above ``E_piv`` the reach is
    a physical halo. Below it the negative branch is a stand-in for partial
    occupancy rather than a literal distance, and should be read as a
    phenomenological turn-on.

    Both parameters describe the medium and the muon rather than the array, so
    one pair applies to every configuration at a site with only ``R_det``
    changing. That is what makes the law transferable between a full detector
    and a partially built one, and therefore testable.

    Parameters
    ----------
    radius_km : float
        Instrumented footprint radius ``R_det`` [km].
    energy_gev : float or np.ndarray
        Muon energy [GeV] setting the light output.
    reach_km : float
        Growth of the reach per e-fold of energy ``Lambda`` [km].
    pivot_gev : float
        Energy ``E_piv`` at which the effective radius equals ``R_det`` [GeV].

    Returns
    -------
    radius : np.ndarray
        Effective radius [km], clipped at zero.

    Notes
    -----
    The reach is a statement about the muon's light output *where it is seen*,
    so the energy passed should be the muon energy at the detector. Callers
    working at fixed neutrino energy, where the arrival energy varies along the
    depth integral, necessarily approximate this by the production energy;
    that overestimates the reach for muons born far upstream. In the
    muon-energy-differential convention the observed energy is the natural
    argument and no approximation is involved.
    """
    energy = np.atleast_1d(np.asarray(energy_gev, dtype=float))
    radius = np.clip(radius_km + reach_km * np.log(energy / pivot_gev), 0.0, None)
    return radius.reshape(np.shape(energy_gev)) if np.ndim(energy_gev) else radius


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
    source: str = DEFAULT_SOURCE,
    light_yield_length_km: float | None = None,
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
    source : {"proposal", "table1"}, optional
        Transport-coefficient tabulation; see
        :mod:`softpaws.transport.coefficients`.
    b_scale : float, optional
        Multiplicative rescaling of the drift coefficient ``b_mu``, used
        as a transport nuisance parameter in the data fits (Section 3). Defaults
        to 1 (the theoretical value).
    light_yield_length_km : float or None, optional
        If set, replaces the static projected area ``pi R_det^2`` with the
        energy-growing :func:`dynamic_projected_area_km2` (not part of the
        paper; see that function's docstring). ``None`` (the default)
        preserves the static-radius behaviour.

    Returns
    -------
    v_soft : np.ndarray
        Soft volume [km^3].
    """
    a = spectral_penalty(gamma, lam)
    b_mu = b_scale * drift_coefficient(energy_gev, density_g_cm3, source)
    if light_yield_length_km is None:
        proj_area = np.pi * radius_km**2
    else:
        proj_area = dynamic_projected_area_km2(
            radius_km, energy_gev, light_yield_length_km, density_g_cm3, b_scale, source,
        )
    return proj_area / (b_mu * a)


def soft_volume_diffusion(
    radius_km: float,
    energy_gev: float | np.ndarray,
    gamma: float,
    lam: float = DEFAULT_LAMBDA,
    density_g_cm3: float = RHO_WATER_G_CM3,
    b_scale: float = 1.0,
    d_scale: float = 1.0,
    source: str = DEFAULT_SOURCE,
    light_yield_length_km: float | None = None,
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
    light_yield_length_km : float or None, optional
        If set, replaces the static projected area with the energy-growing
        :func:`dynamic_projected_area_km2` (see :func:`soft_volume_drift`).
        ``None`` (the default) preserves the static-radius behaviour.

    Returns
    -------
    v_soft : np.ndarray
        Soft volume [km^3].
    """
    b_mu = b_scale * drift_coefficient(energy_gev, density_g_cm3, source)
    d_mu = d_scale * diffusion_coefficient(energy_gev, density_g_cm3, source)
    v_drift = soft_volume_drift(
        radius_km, energy_gev, gamma, lam, density_g_cm3, b_scale, source,
        light_yield_length_km=light_yield_length_km,
    )
    return v_drift * (1.0 - d_mu / (2.0 * b_mu))


def muon_range_km(
    energy_gev: float | np.ndarray,
    threshold_gev: float = DEFAULT_MUON_THRESHOLD_GEV,
    density_g_cm3: float = RHO_WATER_G_CM3,
    b_scale: float = 1.0,
    source: str = DEFAULT_SOURCE,
    kernel_evaluation: str = "running",
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
    source : {"proposal", "table1"}, optional
        Transport-coefficient tabulation; see
        :mod:`softpaws.transport.coefficients`.
    b_scale : float, optional
        Multiplicative rescaling of the drift coefficient. Defaults to 1.

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
    if kernel_evaluation not in ("frozen", "running"):
        raise ValueError(
            f"kernel_evaluation must be 'frozen' or 'running', got {kernel_evaluation!r}."
        )
    energy = np.atleast_1d(np.asarray(energy_gev, dtype=float))
    if kernel_evaluation == "frozen":
        b_mu = b_scale * drift_coefficient(energy, density_g_cm3, source)
        e_crit = critical_energy_gev(energy, density_g_cm3, b_scale, source)
        ratio = (energy + e_crit) / (threshold_gev + e_crit)
        return np.clip(np.log(ratio) / b_mu, 0.0, None)

    # Running: -dE/dx = a_mu + b_mu(E) E has no closed-form integral once b_mu
    # itself runs, so integrate dL = dE / (a_mu + b_mu(E) E) on a shared
    # logarithmic grid. Frozen b_mu recovers the closed form above exactly.
    top = float(np.max(energy))
    if top <= threshold_gev:
        return np.zeros_like(energy)
    log10_grid = np.linspace(
        np.log10(threshold_gev),
        np.log10(top),
        max(2, int(np.ceil((np.log10(top) - np.log10(threshold_gev)) * 48)) + 1),
    )
    grid = 10.0**log10_grid
    b_grid = b_scale * drift_coefficient(grid, density_g_cm3, source)
    a_mu = b_scale * ionization_coefficient(density_g_cm3)
    integrand = grid / (a_mu + b_grid * grid)
    ln_grid = log10_grid * np.log(10.0)
    cumulative = np.concatenate(
        [[0.0], np.cumsum(np.diff(ln_grid) * 0.5 * (integrand[1:] + integrand[:-1]))]
    )
    return np.clip(np.interp(np.log10(energy), log10_grid, cumulative), 0.0, None)


def _log_loss_moments_at(
    energy_gev: np.ndarray,
    density_g_cm3: float,
    b_scale: float,
    source: str,
    log_loss_source: str,
) -> tuple[np.ndarray, np.ndarray]:
    """``Phi'(0)`` and ``-Phi''(0)`` at each energy, by whichever route is asked for.

    Factored out of :func:`stochastic_muon_range_km` so that the frozen and
    running evaluations read the kernel the same way and differ only in *where*
    they read it.
    """
    if log_loss_source == "table" and source != "table1":
        first, second, _ = log_loss_moments(energy_gev, density_g_cm3, source)
        return b_scale * first, b_scale * second
    # Table 1 tabulates no log-loss columns, so there the family is the only
    # route; it is 8% low on the first moment and 56% low on the second.
    b_mu = b_scale * drift_coefficient(energy_gev, density_g_cm3, source)
    d_mu = diffusion_coefficient(energy_gev, density_g_cm3, source)
    kappa, p = two_moment_loss_spectrum(b_mu, d_mu)
    return kappa * polygamma(1, p + 1.0), -kappa * polygamma(2, p + 1.0)


def _running_radiative_length_km(
    energy_gev: np.ndarray,
    floor_gev: float,
    density_g_cm3: float,
    b_scale: float,
    source: str,
    log_loss_source: str,
    nodes_per_decade: int,
) -> np.ndarray:
    """The radiative segment with the kernel followed down the trajectory.

    The rate at which log energy is shed is a local quantity, so over a descent
    spanning decades the depth accumulates as ``dL / dlnE = 1 / Phi'(0; E)``
    and the frozen form ``ln(eps/E_floor) / Phi'(0; eps)`` is the value of that
    integrand at the *top* of the descent, where the loss rate is highest. It is
    therefore short, one-sidedly and by more the further the muon falls.

    Evaluated once on a shared logarithmic grid and interpolated, so the cost is
    independent of how many production energies are asked for.
    """
    energy = np.atleast_1d(np.asarray(energy_gev, dtype=float))
    top = float(np.max(energy))
    if top <= floor_gev:
        return np.zeros_like(energy)
    log10_lo, log10_hi = np.log10(floor_gev), np.log10(top)
    n_nodes = max(2, int(np.ceil((log10_hi - log10_lo) * nodes_per_decade)) + 1)
    log10_grid = np.linspace(log10_lo, log10_hi, n_nodes)
    first, _ = _log_loss_moments_at(
        10.0**log10_grid, density_g_cm3, b_scale, source, log_loss_source
    )
    ln_grid = log10_grid * np.log(10.0)
    integrand = 1.0 / first
    cumulative = np.concatenate(
        [[0.0], np.cumsum(np.diff(ln_grid) * 0.5 * (integrand[1:] + integrand[:-1]))]
    )
    return np.interp(np.log10(energy), log10_grid, cumulative)


def stochastic_muon_range_km(
    energy_gev: float | np.ndarray,
    threshold_gev: float = DEFAULT_MUON_THRESHOLD_GEV,
    density_g_cm3: float = RHO_WATER_G_CM3,
    b_scale: float = 1.0,
    source: str = DEFAULT_SOURCE,
    ell_max_scale: float = 2.5,
    n_ell: int = 161,
    method: str = "closed",
    include_ionization: bool = True,
    match_energy_gev: float = DEFAULT_IONIZATION_MATCH_GEV,
    log_loss_source: str = "table",
    kernel_evaluation: str = "running",
    running_nodes_per_decade: int = 48,
) -> np.ndarray:
    """Muon range to a threshold, averaged over the exact stochastic loss law.

    :func:`muon_range_km` answers "how far does the *average* muon get before
    dropping below ``E_thr``" by integrating the continuous-slowing-down law,
    which makes the arrival-above-threshold probability a step function at that
    mean range. The losses are not deterministic, though
    (:mod:`softpaws.transport.loss_distribution`), so the honest length is the
    expected one,

    .. math:: L(\\varepsilon) = \\int_0^\\infty d\\ell\\;
        \\mathbb{P}\\bigl[\\,W(\\ell) < \\ln(\\varepsilon / E_\\mathrm{thr})\\,\\bigr],

    with ``W`` the accumulated log-loss subordinator whose CDF is
    :func:`softpaws.transport.loss_distribution.log_loss_cdf`. Replacing the
    step by the true CDF shortens the length by ~7% at 1 PeV and ~12% at
    100 PeV: the loss law is right-skewed, so more muons fall short of the mean
    range than overshoot it.

    That integral has a closed form (``docs/first_passage_range.md``). ``W`` is
    non-decreasing, so ``{W(ell) < w}`` is exactly ``{tau(w) > ell}`` for the
    first-passage depth ``tau``, and the integral collapses to ``E[tau(w)]`` --
    the expected distance at which the muon first drops below threshold. That
    renewal function has Laplace transform ``1 / (s Phi(s))``, whose small-``s``
    expansion gives

    .. math:: L(\\varepsilon) = \\frac{\\ln(\\varepsilon / E_\\mathrm{thr})}
        {\\Phi'(0)} - \\frac{\\Phi''(0)}{2\\,\\Phi'(0)^2},
        \\qquad \\Phi'(0) = \\langle -\\ln(1-y)\\rangle,\\;
        -\\Phi''(0) = \\langle \\ln^2(1-y)\\rangle,

    i.e. the CSDA formula with ``b_mu = <y>`` replaced by ``<-ln(1-y)>``, plus a
    constant. Since ``-ln(1-y) >= y`` for every positive loss spectrum, the
    stochastic range is always the shorter one.

    **Ionization and the two-regime range.** The expansion above is purely
    radiative, but a threshold of 1 TeV sits within a factor of two of the muon
    critical energy in water (``E_c = a_mu / b_mu ~ 600`` GeV,
    :func:`~softpaws.transport.coefficients.critical_energy_gev`), so the last
    e-fold of the range -- the one that sets where a tabulated effective area
    turns on -- is not radiative at all. Ionization cannot simply be added to
    ``Phi``: it removes a fixed amount of energy per unit length and not a fixed
    *fraction*, so it is additive in ``E`` and not in ``ln E``, which is the
    structure the whole subordinator derivation rests on.

    With ``include_ionization`` (the default) the range is therefore spliced at a
    matching energy ``E_*`` chosen well above ``E_c``,

    .. math:: L(\\varepsilon) = \\underbrace{\\frac{\\ln(\\varepsilon/E_*)}
        {\\Phi'(0)} - \\frac{\\Phi''(0)}{2\\Phi'(0)^2}}_{\\text{stochastic,
        radiative}} \\;+\\; \\underbrace{\\frac{1}{b_\\mu(E_a)}
        \\ln\\frac{E_a + E_c}{E_\\mathrm{thr} + E_c}}_{\\text{deterministic,
        with ionization}},

    which leaves the derivation untouched above ``E_*`` and appends an almost
    constant offset below it, so the result is still closed form. Muons born
    below ``E_*`` get the deterministic range alone (:func:`muon_range_km`).
    Passing ``include_ionization=False`` recovers the purely radiative range,
    which is what Table E.1 of the paper contrasts against ``R_CSDA``.

    The deterministic segment starts at ``E_a = E_* exp(-<overshoot>)`` and not
    at ``E_*``, because a first passage overshoots the level it crosses. By
    Wald's identity ``E[W(tau)] = w + <overshoot>`` with
    ``<overshoot> = -Phi''(0) / 2 Phi'(0)`` in log energy, which is the same
    quantity the renewal constant above measures in depth, so starting the CSDA
    segment at ``E_*`` would count that stretch of track twice. It is worth 0.4
    km, and leaving it in shows up as a 0.4 km discontinuity at ``E_*``.

    The splice shortens the range at every energy, by 17% at ``E_nu = 10^4``
    GeV, where the muon never enters the radiative regime at all, through 6.7%
    at ``10^6`` to 4.3% at ``10^8``. Under ``kernel_evaluation="frozen"`` it
    instead changed sign across the band (-13%, -1.6%, +1.4%), because two
    effects ran against each other in the spliced decade: ionization shortened
    the range while ceasing to apply the production-energy ``Phi'(0)`` to a muon
    that is by then at TeV energies lengthened it, and the two nearly cancelled.
    Running the kernel down the trajectory already carries the second of those,
    so the splice is left doing only the physical job it exists for.

    ``match_energy_gev`` is a convention and the answer moves with it: a factor
    of three either way from the default changes the range by about 3%, which is
    the systematic this treatment carries. Lowering it keeps more of the track
    stochastic, which is right down to ``E_c``; raising it keeps more of the
    ionization, which matters most near threshold. Neither limit is uniformly
    better, and only a deterministic drift inside ``W(ell)`` would remove the
    choice.

    Unlike the soft volume's spectral length ``1/Phi(A)``, this carries no
    spectral weighting -- it is the right length for a **monochromatic** parent,
    which is what a tabulated effective area is differential in (App. I's
    ``s -> 0`` case; see :func:`dm_line_target_volume_km3`).

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
        Multiplicative rescaling of the drift coefficient. Defaults to 1.
    source : {"proposal", "table1"}, optional
        Transport-coefficient tabulation; see
        :mod:`softpaws.transport.coefficients`.
    ell_max_scale : float, optional
        Upper limit of the depth integral, in units of the deterministic range
        (:func:`muon_range_km`). The CDF is already below ``1e-3`` by twice the
        deterministic range, so the default truncation is harmless. Used by
        ``method="quadrature"`` only.
    n_ell : int, optional
        Number of depth samples in the integral. Used by
        ``method="quadrature"`` only.
    method : {"closed", "quadrature"}, optional
        ``"closed"`` (the default) evaluates the renewal expansion above, two
        polygamma calls and no inversion. ``"quadrature"`` evaluates the depth
        integral directly against
        :func:`softpaws.transport.loss_distribution.log_loss_cdf`; it is orders
        of magnitude slower and is kept as the independent check that the
        closed form is exercised against.
    include_ionization : bool, optional
        Splice the deterministic ionizing range below ``match_energy_gev`` onto
        the radiative first passage above it, as derived above. Defaults to
        ``True``. ``False`` gives the purely radiative range.
    match_energy_gev : float, optional
        Matching energy ``E_*`` [GeV] between the two regimes, which has to sit
        well above the critical energy for the splice to be meaningful.
        Defaults to :data:`DEFAULT_IONIZATION_MATCH_GEV`. Ignored when
        ``include_ionization`` is ``False``, and also when it falls at or below
        ``threshold_gev``, where there is no ionizing segment left to splice and
        the range degrades to the purely radiative one.
    log_loss_source : {"table", "family"}, optional
        Where ``Phi'(0)`` and ``Phi''(0)`` come from. ``"table"`` (the default)
        reads them from the tabulated spectrum via
        :func:`~softpaws.transport.coefficients.log_loss_moments`; ``"family"``
        reconstructs them from the two-moment calibration of ``b_mu`` and
        ``d_mu``. The family is 8% low on the first moment and 56% low on the
        second, because ``-ln(1-y)`` weights the hard end of the kernel that a
        fit to the ``y``-moments does not constrain, and the resulting range is
        6.9% long over ``10^5`` to ``10^8`` GeV. The bias is almost pure
        normalization -- 0.7% rms of residual tilt across that band -- so it
        moves an effective-area ceiling and leaves its shape alone. Kept as an
        option because ``source="table1"`` has no log-loss columns and because
        ``method="quadrature"`` checks against the family's own kernel.
    kernel_evaluation : {"running", "frozen"}, optional
        Where along the descent the kernel is read. ``"running"`` (the default)
        follows it down, so the radiative segment is
        ``int dlnE / Phi'(0; E)`` and both renewal constants are read at the
        level being crossed. ``"frozen"`` holds the production-energy kernel for
        the whole descent, which is the closed form of Eq.~(C4) as written.

        Freezing is short, one-sidedly, and by more the further the muon falls,
        because it evaluates the loss rate at the top of the descent where that
        rate is highest. Against a PROPOSAL propagation (example 39, stopping at
        100 TeV) the mean range is 4.0% rms and 6.5% worst over ``w = 1.15`` to
        ``5.76``, against 1.4% and 2.5% running. Effective areas reach
        ``w ~ 9``, where the two differ by 11%. The gap is a rising tilt and not
        a normalization: 0% at ``10^4`` GeV of parent energy, 2.1% at ``10^5``,
        4.4% at ``10^6``, 7.3% at ``10^7`` and 11.3% at ``10^8``.

        Kept selectable because the frozen form is what the drift-diffusion
        literature evaluates and what ``method="quadrature"`` can check.
    running_nodes_per_decade : int, optional
        Grid density for the running integral. The integrand ``1/Phi'(0; E)``
        varies by ``E^-beta`` with ``beta ~ 0.028``, so it is nearly linear in
        ``lnE``: the default is converged to 6 mm over four decades, two orders
        below the 4 cm at which the closed form tracks its own depth integral.

    Returns
    -------
    range_km : np.ndarray
        Expected range [km], zero for muons born below threshold.

    Raises
    ------
    ValueError
        Raised if ``method`` or ``log_loss_source`` is not one of its supported
        values, or if the two are combined incompatibly.

    Notes
    -----
    ``b_mu`` and ``d_mu`` are evaluated once, at the production energy, rather
    than followed down the track -- the same percent-level approximation
    :func:`muon_range_km` makes and justifies. Against a PROPOSAL Monte Carlo
    that cost is 4.5% rms on the range, dropping to 1.8% if the moments are
    integrated down the trajectory instead
    (``examples/39_range_moment_estimator.py``).

    With ``log_loss_source="family"`` the two methods agree to better than 4 cm
    over ``10^4`` to ``10^8`` GeV. The
    residual is the depth grid: it is sized to the range to ``E_thr`` while the
    integrand falls off on the shorter range to ``E_*``, so the spliced form is
    sampled more coarsely than the purely radiative one, which agrees to 2 mm.
    The closed form is an expansion in ``1 / ln(eps / E_thr)``, so it should
    not be pushed to ``eps -> E_thr``, where the length vanishes anyway.
    """
    if method not in ("closed", "quadrature"):
        raise ValueError(f"method must be 'closed' or 'quadrature', got {method!r}.")
    if log_loss_source not in ("table", "family"):
        raise ValueError(
            f"log_loss_source must be 'table' or 'family', got {log_loss_source!r}."
        )
    if kernel_evaluation not in ("frozen", "running"):
        raise ValueError(
            f"kernel_evaluation must be 'frozen' or 'running', got {kernel_evaluation!r}."
        )
    if method == "quadrature" and kernel_evaluation != "frozen":
        # The depth integral builds one kernel and holds it for the whole
        # descent, so it is a frozen calculation by construction. Checking the
        # running closed form against it would measure the running, not the
        # renewal expansion the check exists for.
        raise ValueError(
            "method='quadrature' freezes the kernel at the production energy, so "
            "it needs kernel_evaluation='frozen'."
        )
    if method == "quadrature" and log_loss_source != "family":
        # The depth integral runs against log_loss_cdf, which builds the
        # two-moment family's kernel. Checking a table-based closed form against
        # it would compare two different kernels and disagree by ~7% by
        # construction, so the internal check is pinned to the family.
        raise ValueError(
            "method='quadrature' checks the closed form against the two-moment "
            "family's own depth integral, so it needs log_loss_source='family'."
        )

    energy = np.atleast_1d(np.asarray(energy_gev, dtype=float))
    deterministic = muon_range_km(
        energy, threshold_gev, density_g_cm3, b_scale, source, kernel_evaluation
    )
    b_mu = b_scale * drift_coefficient(energy, density_g_cm3, source)
    d_mu = diffusion_coefficient(energy, density_g_cm3, source)
    selectable = (energy > threshold_gev) & (deterministic > 0.0)

    # Phi'(0) = <-ln(1-y)> and -Phi''(0) = <ln^2(1-y)>, both per unit length,
    # here at the production energy.
    first, second = _log_loss_moments_at(
        energy, density_g_cm3, b_scale, source, log_loss_source
    )

    # Above the matching energy the first passage is radiative and stochastic; below it
    # the muon is within reach of E_c and slows deterministically. The stochastic part
    # already carries the muon *past* E_*, since first passage overshoots the level it
    # crosses: by Wald, E[W(tau)] = w + <overshoot> with <overshoot> = -Phi''(0)/2Phi'(0)
    # in log energy, which is the same quantity the renewal constant measures in depth.
    # The deterministic segment therefore starts at the mean arrival energy and not at
    # E_*, and its b_mu is evaluated there, which is where the muon actually is.
    # A threshold at or above E_* leaves no ionizing segment to splice on, since
    # the muon stops counting while it is still radiative. Clamping the floor
    # degrades the two-regime range back to the purely radiative one, which is
    # what a fitted threshold above E_* should get.
    # Both renewal constants -- the mean overshoot in log energy and the
    # constant it contributes to the depth -- belong to the *crossing*, so a
    # running evaluation reads them at the level being crossed. Freezing reads
    # everything at production, which is what makes it a frozen calculation.
    stochastic_floor = float(
        match_energy_gev
        if include_ionization and match_energy_gev > threshold_gev
        else threshold_gev
    )
    if kernel_evaluation == "running":
        crossing_first, crossing_second = _log_loss_moments_at(
            np.array([stochastic_floor]), density_g_cm3, b_scale, source, log_loss_source
        )
    else:
        crossing_first, crossing_second = first, second

    if include_ionization and match_energy_gev > threshold_gev:
        # A first passage crosses its level from above, so the overshoot is
        # non-negative and the arrival energy lies in [E_thr, E_*]. Both bounds
        # bind only where the calibrated kernel is not a valid loss spectrum at
        # all -- ``d_mu / b_mu >= 1`` puts ``p + 1 <= 0`` -- which a sampler
        # exploring an unphysical b_scale does reach, and where an unclamped
        # exponential would return an infinite range instead of a wrong one.
        overshoot = np.clip(crossing_second / (2.0 * crossing_first), 0.0, None)
        arrival_gev = np.clip(
            stochastic_floor * np.exp(-overshoot), threshold_gev, stochastic_floor
        )
        offset_km = muon_range_km(
            arrival_gev, threshold_gev, density_g_cm3, b_scale, source, kernel_evaluation
        )
    else:
        offset_km = np.zeros_like(energy)
    stochastic_regime = selectable & (energy > stochastic_floor)

    if method == "closed":
        with np.errstate(divide="ignore", invalid="ignore"):
            if kernel_evaluation == "running":
                radiative_km = _running_radiative_length_km(
                    energy,
                    stochastic_floor,
                    density_g_cm3,
                    b_scale,
                    source,
                    log_loss_source,
                    running_nodes_per_decade,
                )
            else:
                radiative_km = np.log(energy / stochastic_floor) / first
            length = (
                radiative_km
                + crossing_second / (2.0 * crossing_first**2)
                + offset_km
            )
        # Muons born below E_* never enter the radiative regime, so the deterministic
        # range is the whole of their answer.
        out = np.where(stochastic_regime, length, np.where(selectable, deterministic, 0.0))
        return out.reshape(np.shape(energy_gev)) if np.ndim(energy_gev) else out

    from .loss_distribution import log_loss_cdf

    out = np.zeros(energy.shape[0])
    for i, eps in enumerate(energy):
        if not selectable[i]:
            continue
        if not stochastic_regime[i]:
            out[i] = float(deterministic[i])
            continue
        ell = np.linspace(0.0, ell_max_scale * float(deterministic[i]), n_ell)
        cdf = log_loss_cdf(np.log(eps / stochastic_floor), ell, float(b_mu[i]), float(d_mu[i]))
        out[i] = float(np.trapezoid(cdf, ell)) + float(offset_km[i])
    return out.reshape(np.shape(energy_gev)) if np.ndim(energy_gev) else out


def _running_variance_rate_km2(
    energy_gev: np.ndarray,
    floor_gev: float,
    density_g_cm3: float,
    b_scale: float,
    source: str,
    nodes_per_decade: int = 48,
) -> np.ndarray:
    """``int dlnE (-Phi''(0; E)) / Phi'(0; E)^3``, the running form of the term
    linear in ``w`` in :func:`stochastic_muon_range_variance_km2`."""
    energy = np.atleast_1d(np.asarray(energy_gev, dtype=float))
    top = float(np.max(energy))
    if top <= floor_gev:
        return np.zeros_like(energy)
    log10_grid = np.linspace(
        np.log10(floor_gev),
        np.log10(top),
        max(2, int(np.ceil((np.log10(top) - np.log10(floor_gev)) * nodes_per_decade)) + 1),
    )
    first, second, _ = log_loss_moments(10.0**log10_grid, density_g_cm3, source)
    first, second = b_scale * first, b_scale * second
    ln_grid = log10_grid * np.log(10.0)
    integrand = second / first**3
    cumulative = np.concatenate(
        [[0.0], np.cumsum(np.diff(ln_grid) * 0.5 * (integrand[1:] + integrand[:-1]))]
    )
    return np.interp(np.log10(energy), log10_grid, cumulative)


def stochastic_muon_range_variance_km2(
    energy_gev: float | np.ndarray,
    threshold_gev: float = DEFAULT_MUON_THRESHOLD_GEV,
    density_g_cm3: float = RHO_WATER_G_CM3,
    b_scale: float = 1.0,
    source: str = DEFAULT_SOURCE,
    kernel_evaluation: str = "running",
) -> np.ndarray:
    """Variance of the muon range to a threshold, from the same first passage.

    :func:`stochastic_muon_range_km` returns the *mean* depth at which a muon
    first falls below ``E_thr``. Individual muons scatter about it by tens of
    percent, and that spread has a closed form built from the same kernel. With
    ``w = ln(varepsilon / E_thr)``,

    .. math:: \\mathrm{Var}(R) = \\frac{-\\Phi''(0)\\,w}{\\Phi'(0)^3}
        - \\frac{\\Phi'''(0)}{3\\,\\Phi'(0)^3}
        + \\frac{\\Phi''(0)^2}{4\\,\\Phi'(0)^4}.

    The structure mirrors the mean, and both constants are moments of the same
    stationary overshoot. A first passage crosses its level from above; the
    *mean* overshoot ``-Phi''(0) / 2 Phi'(0)`` is the constant in the mean, and
    its *variance* ``Phi'''(0) / 3 Phi'(0) - Phi''(0)^2 / 4 Phi'(0)^2``, divided
    by ``Phi'(0)^2`` to turn log-energy into depth, is the constant here. It
    enters negatively: a muon that overshoots further crossed its level sooner.

    Both terms matter. Against a direct simulation of PROPOSAL's kernel the
    leading term alone runs 8 to 25% high over ``w = 3.5`` to ``9.2``; with the
    constant the agreement is better than 2%, and against PROPOSAL itself the
    spread comes out to 3.5% rms with nothing fitted. The second-order
    drift-diffusion transport, by contrast, is 30 to 40% low at every ``w`` and
    worsens with distance, because it carries ``d_mu = <y^2>`` where this
    quantity needs ``<ln^2(1-y)>``, and the two differ by a factor of four. See
    ``examples/39_range_moment_estimator.py``.

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
        Multiplicative rescaling of the kernel normalization. All three moments
        scale with it, so the variance scales as ``1 / b_scale^2``.
    source : {"proposal"}, optional
        Transport-coefficient tabulation; see
        :mod:`softpaws.transport.coefficients`. Needs the log-loss moment
        columns, which ``"table1"`` does not have.

    Returns
    -------
    variance_km2 : np.ndarray
        Variance of the range [km^2 w.e.], zero below threshold.

    Notes
    -----
    Purely radiative, with no counterpart to the ionization splice of
    :func:`stochastic_muon_range_km`. Below the matching energy the loss is
    deterministic and adds no variance of its own, but the energy at which the
    muon *arrives* there fluctuates by the overshoot, and that fluctuation is
    anticorrelated with the first-passage depth above it. Reproducing the
    spliced variance therefore needs that covariance, which is not derived here;
    for a threshold at or below the critical energy this result is the
    stochastic part alone.

    The expansion is in ``1 / w`` and its constant term is negative, so it
    returns zero rather than a negative variance for ``w`` below about one,
    where a muon reaches the threshold in a handful of collisions and no
    expansion of this kind applies.
    """
    if kernel_evaluation not in ("frozen", "running"):
        raise ValueError(
            f"kernel_evaluation must be 'frozen' or 'running', got {kernel_evaluation!r}."
        )
    energy = np.atleast_1d(np.asarray(energy_gev, dtype=float))
    phi_prime, phi_second, phi_third = (
        b_scale * moment for moment in log_loss_moments(energy, density_g_cm3, source)
    )
    if kernel_evaluation == "running":
        # Both constants belong to the crossing, so both are read at the level
        # being crossed; only the term linear in w accumulates down the descent.
        phi_prime, phi_second, phi_third = (
            b_scale * np.reshape(moment, ())
            for moment in log_loss_moments(np.array([threshold_gev]), density_g_cm3, source)
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.log(energy / threshold_gev)
        overshoot_variance = (
            phi_third / (3.0 * phi_prime) - phi_second**2 / (4.0 * phi_prime**2)
        ) / phi_prime**2
        if kernel_evaluation == "running":
            linear = _running_variance_rate_km2(
                energy, threshold_gev, density_g_cm3, b_scale, source
            )
        else:
            linear = phi_second * w / phi_prime**3
        variance = linear - overshoot_variance

    out = np.where(energy > threshold_gev, np.clip(variance, 0.0, None), 0.0)
    return out.reshape(np.shape(energy_gev)) if np.ndim(energy_gev) else out


def truncated_muon_range_km(
    energy_gev: float | np.ndarray,
    column_km: float | np.ndarray,
    threshold_gev: float = DEFAULT_MUON_THRESHOLD_GEV,
    density_g_cm3: float = RHO_WATER_G_CM3,
    b_scale: float = 1.0,
    source: str = DEFAULT_SOURCE,
    kernel_evaluation: str = "running",
) -> np.ndarray:
    """First-passage range cut at a finite upstream column [km].

    :func:`stochastic_muon_range_km` integrates the first-passage probability to
    infinite depth, which is right whenever the medium supplies more column than
    any muon survives and wrong whenever it does not. A downgoing track is always
    the second case, since the muon cannot be born above the ice; so is any
    detector under a few km of water, where an upgoing muon at the top of the
    band would need more column than the site has. The honest length is then the
    *limited* expectation

    .. math:: L(\\varepsilon, X) = \\mathbb{E}[\\tau(w) \\wedge X]
        = \\int_0^X {\\rm d}\\ell\\;\\mathbb{P}[W(\\ell) < w].

    Evaluating that integral directly needs the log-loss CDF at every depth.
    Matching a gamma law to the first two moments of the first-passage depth --
    :func:`stochastic_muon_range_km` and
    :func:`stochastic_muon_range_variance_km2` -- turns it into an incomplete
    gamma function instead, at no cost in accuracy that matters here: example
    33's ``--check-truncation`` holds it against a direct Gil-Pelaez inversion.

    Both moments are built from the tabulated log-loss moments, so this agrees
    with :func:`stochastic_muon_range_km` with ``include_ionization=False`` in
    the ``column_km -> inf`` limit. There is no ionization splice: the truncation
    is only interesting where the column runs out well before the muon reaches
    the critical energy.

    Parameters
    ----------
    energy_gev : float or np.ndarray
        Muon energy at production [GeV].
    column_km : float or np.ndarray
        Available upstream column, as a length of the medium [km]. Broadcast
        against ``energy_gev``; ``inf`` returns the untruncated range.
    threshold_gev : float, optional
        Muon energy below which the track is not selected [GeV]. Defaults to
        :data:`DEFAULT_MUON_THRESHOLD_GEV`.
    density_g_cm3 : float, optional
        Medium density [g cm^-3]. Defaults to water.
    b_scale : float, optional
        Multiplicative rescaling of the kernel normalization.
    source : {"proposal"}, optional
        Transport-coefficient tabulation; see
        :mod:`softpaws.transport.coefficients`.

    Returns
    -------
    length_km : np.ndarray
        Expected truncated range [km], zero for muons born below threshold.
    """
    if kernel_evaluation not in ("frozen", "running"):
        raise ValueError(
            f"kernel_evaluation must be 'frozen' or 'running', got {kernel_evaluation!r}."
        )
    energy = np.asarray(energy_gev, dtype=float)
    # The tabulated moments come back at least one-dimensional; keep the shape of
    # the input so a scalar energy gives a scalar length, as the callers assume.
    constant_at = energy if kernel_evaluation == "frozen" else np.array([threshold_gev])
    phi_prime, phi_second, _ = (
        np.reshape(b_scale * moment, np.shape(energy) if kernel_evaluation == "frozen" else ())
        for moment in log_loss_moments(constant_at, density_g_cm3, source)
    )

    selectable = energy > threshold_gev
    w = np.where(selectable, np.log(np.maximum(energy, threshold_gev) / threshold_gev), 0.0)
    if kernel_evaluation == "running":
        radiative = np.reshape(
            _running_radiative_length_km(
                np.maximum(energy, threshold_gev),
                threshold_gev,
                density_g_cm3,
                b_scale,
                source,
                "table",
                48,
            ),
            np.shape(energy),
        )
    else:
        radiative = w / phi_prime
    mean = radiative + phi_second / (2.0 * phi_prime**2)
    variance = np.reshape(
        stochastic_muon_range_variance_km2(
            np.maximum(energy, threshold_gev * (1.0 + 1.0e-12)),
            threshold_gev,
            density_g_cm3,
            b_scale,
            source,
            kernel_evaluation,
        ),
        np.shape(energy),
    )

    column = np.asarray(column_km, dtype=float)
    # An infinite column is the untruncated case; the general expression below
    # would evaluate inf * 0 on it.
    capped = np.where(np.isfinite(column), column, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        shape = mean**2 / variance
        x = capped / (variance / mean)
        limited = mean * gammainc(shape + 1.0, x) + capped * (1.0 - gammainc(shape, x))
    # A few e-folds above threshold the variance expansion floors at zero, where
    # the first passage is effectively deterministic and the limited expectation
    # is just the shorter of the two lengths.
    limited = np.where(variance > 0.0, limited, np.minimum(mean, capped))
    limited = np.where(np.isfinite(column), limited, mean)
    return np.where(selectable & (mean > 0.0), np.clip(limited, 0.0, None), 0.0)


def range_target_volume_km3(
    radius_km: float,
    energy_gev: float | np.ndarray,
    threshold_gev: float = DEFAULT_MUON_THRESHOLD_GEV,
    density_g_cm3: float = RHO_WATER_G_CM3,
    b_scale: float = 1.0,
    source: str = DEFAULT_SOURCE,
    light_yield_length_km: float | None = None,
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
    source : {"proposal", "table1"}, optional
        Transport-coefficient tabulation; see
        :mod:`softpaws.transport.coefficients`.
    b_scale : float, optional
        Multiplicative rescaling of the drift coefficient. Defaults to 1.
    light_yield_length_km : float or None, optional
        If set, replaces the static projected area with the energy-growing
        :func:`dynamic_projected_area_km2` (see
        :func:`dynamic_projected_radius_km`'s docstring for the motivation:
        stochastic light yield above the critical energy letting a track
        trigger from beyond ``R_det``, not part of the paper). Only the
        lateral projected area grows; the instrumented sphere ``V_det``
        below is left untouched. ``None`` (the default) preserves the
        static-radius behaviour.

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
    if light_yield_length_km is None:
        proj_area = np.pi * radius_km**2
    else:
        proj_area = dynamic_projected_area_km2(
            radius_km, energy_gev, light_yield_length_km, density_g_cm3, b_scale, source,
        )
    v_det = 4.0 / 3.0 * np.pi * radius_km**3
    range_km = muon_range_km(energy_gev, threshold_gev, density_g_cm3, b_scale, source)
    return np.where(range_km > 0.0, proj_area * range_km + v_det, 0.0)


def dm_line_target_volume_km3(radius_km: float, column_depth_km: float) -> float:
    """Target volume for a monochromatic (dark matter line) source (App. I).

    A delta-function line at ``E_nu = m_chi`` has no continuum spectral index
    to average over, so it excites the ``s = 0`` mode of the same eigenvalue
    formalism the rest of the package uses (``docs/2026_softvolume_vs_implementation.md``
    S:2.4). At ``s = 0``, ``Phi(0) = 0`` identically
    (:func:`softpaws.transport.eigenvalue.phi_eigenvalue`) and
    ``I(0) = 1`` (:func:`softpaws.transport.source.inelasticity_factor`), so
    :func:`soft_volume_exact`'s ``I(A) A_proj (1 - e^{-Phi(A) x}) / Phi(A)``
    collapses to the plain geometric column ``A_proj x`` -- no exponential
    range-shortening at all, since there is no spectral index left to shorten
    the range against. This is the literal ``s -> 0`` instance of the same
    soft-volume machinery used for the power-law case, offered here for
    side-by-side comparison against the muon-range convention
    (:func:`range_target_volume_km3`), not as a replacement for it -- see
    ``examples/34_dm_line_sensitivity.py``, which uses the muon-range
    convention for exactly that reason.

    Parameters
    ----------
    radius_km : float
        Radius of the spherical detector [km].
    column_depth_km : float
        Available upstream column depth ``x`` [km] (see
        :mod:`softpaws.transport.attenuation`; a couple km of ice for a
        downgoing line source at the South Pole).

    Returns
    -------
    volume : float
        Target volume [km^3], ``A_proj x + V_det``.

    Notes
    -----
    Unlike :func:`range_target_volume_km3`, this carries no muon-energy
    dependence and so no threshold cutoff: because ``Phi(0) = 0`` removes the
    energy/range weighting entirely, the ``s = 0`` treatment has no way to
    encode "the muon fell below the analysis threshold before reaching the
    detector." It grows *linearly* in the available column ``x`` rather than
    logarithmically, unlike the muon range -- expect the two conventions to
    diverge visibly at large ``x``.
    """
    proj_area = np.pi * radius_km**2
    v_det = 4.0 / 3.0 * np.pi * radius_km**3
    return proj_area * float(column_depth_km) + v_det


def volume_ratio_drift(
    radius_km: float,
    energy_gev: float | np.ndarray,
    gamma: float,
    lam: float = DEFAULT_LAMBDA,
    density_g_cm3: float = RHO_WATER_G_CM3,
    source: str = DEFAULT_SOURCE,
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
    b_mu = drift_coefficient(energy_gev, density_g_cm3, source)
    return 1.0 + 3.0 / (4.0 * radius_km * b_mu * a)


def saturation_factor(
    phi_per_km: float | np.ndarray,
    column_depth_km: float,
    inv_lambda_per_km: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Effective attenuated range over a finite column (Eq. 9, or Eq. 11 coupled).

    With ``inv_lambda_per_km = 0`` (the default) this is the plain transparent-
    Earth factor ``(1 - e^{-Phi x}) / Phi`` that turns the infinite-column range
    ``1/Phi`` into the value delivered by an upstream column of depth ``x``
    (``docs/exact_soft_volume_notes.md`` Part 7). It is finite for every sign of
    ``Phi``: as ``Phi -> 0`` it tends to ``x`` (no losses, the whole column
    accumulates), and for ``Phi < 0`` it continues to
    ``(e^{|Phi| x} - 1) / |Phi|``, curing the drift-form divergence at and beyond
    the cross-section pole.

    With a nonzero ``inv_lambda_per_km = 1/Lambda_nu``, this is instead the
    paper's coupled ``R_nu(x, A)`` (Eq. 11, App. C.2-C.3 of
    ``docs/2026_softvolume.pdf``),

    .. math:: R_\\nu(x, A) = \\frac{e^{-x/\\Lambda_\\nu} - e^{-x\\Phi(A)}}
        {\\Phi(A) - 1/\\Lambda_\\nu},

    which folds the parent neutrino's own Earth attenuation into the same
    depth integral that produces the soft volume, rather than applying it as a
    separate multiplicative flux factor. Setting ``inv_lambda_per_km = 0``
    recovers the plain factor above identically, and the removable singularity
    at ``Phi(A) == 1/Lambda_nu`` is handled by its L'Hopital limit
    ``x e^{-x/Lambda_nu}`` (the same pattern as
    :func:`softpaws.transport.tau.tau_loss_density`'s composite-symbol pole).

    Parameters
    ----------
    phi_per_km : float or np.ndarray
        Collision eigenvalue ``Phi(A)`` [km^-1] (see
        :func:`softpaws.transport.eigenvalue.phi_eigenvalue`).
    column_depth_km : float
        Available upstream column depth ``x`` [km].
    inv_lambda_per_km : float or np.ndarray, optional
        Parent-neutrino attenuation rate ``1/Lambda_nu`` [km^-1] (see
        :func:`softpaws.transport.attenuation.neutrino_interaction_length_km`).
        Defaults to ``0`` (no neutrino attenuation, today's behaviour).

    Returns
    -------
    eff_range : np.ndarray
        ``R_nu(x, A)`` [km], or the plain saturation factor when
        ``inv_lambda_per_km = 0``.
    """
    phi = np.asarray(phi_per_km, dtype=float)
    inv_lambda = np.asarray(inv_lambda_per_km, dtype=float)
    x = float(column_depth_km)
    denom = phi - inv_lambda
    small = np.abs(denom) < 1e-12
    safe_denom = np.where(small, 1.0, denom)
    numer = np.exp(-inv_lambda * x) - np.exp(-phi * x)
    coupled = numer / safe_denom
    # L'Hopital limit at Phi(A) == 1/Lambda_nu: d/dPhi[e^{-x/Lambda}-e^{-x Phi}]
    # over d/dPhi[Phi - 1/Lambda] at fixed inv_lambda is x e^{-x Phi}.
    limit_value = x * np.exp(-phi * x)
    return np.where(small, limit_value, coupled)


def scale_breaking_saturation_factor(
    phi_per_km: float | np.ndarray,
    phi_prime_per_km: float | np.ndarray,
    beta: float,
    column_depth_km: float,
    n_steps: int = 256,
) -> np.ndarray:
    """Saturation factor with the App. F running-spectral-index correction.

    App. F treats a scale-breaking loss rate ``dGamma/dy = E^beta kappa(y)``
    as a shift operator on the Mellin index, whose leading-order (first
    order in ``beta``) consequence (Eq. F4 of ``docs/2026_softvolume.pdf``)
    is that the *effective* spectral index runs with propagated distance
    ``ell``,

    .. math:: s(\\ell) \\approx A + \\beta\\,\\ell\\,\\Phi(A).

    Substituting this into the propagator and Taylor-expanding ``Phi(s(ell))``
    to the same order gives a Gaussian-modified exponent replacing the plain
    ``Phi(A) ell`` of Eq. 9,

    .. math:: \\int_0^x d\\ell\\,\\exp\\!\\left[-\\ell\\,\\Phi(A)
        - \\frac{\\beta}{2}\\,\\ell^2\\,\\Phi(A)\\,\\Phi'(A)\\right],

    evaluated here by quadrature (the antiderivative is an error function, but
    the quadrature is simpler and just as robust for a finite, smooth,
    monotonically decaying integrand). Reduces to :func:`saturation_factor`
    exactly at ``beta = 0``.

    **Scope note.** This implements Eq. F4's leading-order truncation, not
    the full Eq. F3 series: for a source that is genuinely a single power-law
    mode (the case this package treats throughout), Eq. F3's source term
    ``S_hat(xi, s + k beta)`` is only well defined at ``k = 0`` -- the
    compressed paper does not spell out how a single-mode source's higher
    shifted terms are meant to be regularized (compare App. H, which handles
    exactly this kind of non-single-mode bookkeeping explicitly for a cutoff
    source but not for this shift operator). Eq. F4 sidesteps that ambiguity
    entirely by working with the running index directly, at the cost of being
    a first-order (small-``beta``) approximation rather than an exact
    resummation.

    Parameters
    ----------
    phi_per_km : float or np.ndarray
        Collision eigenvalue ``Phi(A)`` [km^-1].
    phi_prime_per_km : float or np.ndarray
        Derivative ``d Phi / dA`` [km^-1] (see
        :func:`softpaws.transport.eigenvalue.phi_eigenvalue_derivative`).
    beta : float
        Scale-breaking exponent (``dGamma/dy ~ E^beta``); ``0`` recovers the
        plain treatment.
    column_depth_km : float
        Available upstream column depth ``x`` [km].
    n_steps : int, optional
        Number of quadrature points. Defaults to 256.

    Returns
    -------
    eff_range : np.ndarray
        The running-index-corrected effective range [km], broadcast to the
        shape of ``phi_per_km``.
    """
    if beta == 0.0:
        return saturation_factor(phi_per_km, column_depth_km)
    phi = np.asarray(phi_per_km, dtype=float)
    phi_prime = np.asarray(phi_prime_per_km, dtype=float)
    x = float(column_depth_km)
    ell = np.linspace(0.0, x, n_steps)
    exponent = ell * phi[..., None] + 0.5 * beta * ell**2 * (phi * phi_prime)[..., None]
    return np.trapezoid(np.exp(-exponent), ell, axis=-1)


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
    source: str = DEFAULT_SOURCE,
    beta: float = 0.0,
    light_yield_length_km: float | None = None,
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
    beta : float, optional
        Scale-breaking exponent of App. F (``dGamma/dy ~ E^beta``), applying
        the Eq. F4 running-spectral-index correction
        (:func:`scale_breaking_saturation_factor`) instead of the plain
        saturation factor. Defaults to ``0`` (off, today's behaviour); requires
        a finite ``column_depth_km``.
    light_yield_length_km : float or None, optional
        If set, replaces the static projected area with the energy-growing
        :func:`dynamic_projected_area_km2` (see :func:`soft_volume_drift`).
        ``None`` (the default) preserves the static-radius behaviour.

    Returns
    -------
    v_soft : np.ndarray
        Soft volume [km^3].

    Raises
    ------
    ValueError
        Raised if ``column_depth_km is None`` while ``Phi(A) <= 0`` (the
        infinite-column limit diverges) or while ``beta != 0`` (App. F's
        running index has no infinite-column limit); pass a finite
        ``column_depth_km`` instead.
    """
    a = spectral_index(gamma, lam)
    b_mu = b_scale * drift_coefficient(energy_gev, density_g_cm3, source)
    d_mu = d_scale * diffusion_coefficient(energy_gev, density_g_cm3, source)
    phi = phi_eigenvalue(a, b_mu, d_mu)
    if light_yield_length_km is None:
        proj_area = np.pi * radius_km**2
    else:
        proj_area = dynamic_projected_area_km2(
            radius_km, energy_gev, light_yield_length_km, density_g_cm3, b_scale, source,
        )

    if column_depth_km is None:
        if beta != 0.0:
            raise ValueError(
                "beta != 0 requires a finite column_depth_km (App. F's running-index "
                "correction has no infinite-column limit)."
            )
        if np.any(phi <= 0.0):
            raise ValueError(
                f"Phi(A) <= 0 for A = {a:.3f}: the infinite-column soft volume "
                "diverges. Pass a finite column_depth_km to use the saturation factor."
            )
        eff_range = 1.0 / phi
    elif beta == 0.0:
        eff_range = saturation_factor(phi, column_depth_km)
    else:
        phi_prime = phi_eigenvalue_derivative(a, b_mu, d_mu)
        eff_range = scale_breaking_saturation_factor(phi, phi_prime, beta, column_depth_km)

    factor = inelasticity_factor(a) if include_inelasticity else 1.0
    return proj_area * factor * eff_range


def soft_volume_attenuated_exact(
    radius_km: float,
    energy_gev: float | np.ndarray,
    gamma: float,
    column_depth_km: float,
    inv_lambda_nu_per_km: float | np.ndarray,
    lam: float = DEFAULT_LAMBDA,
    density_g_cm3: float = RHO_WATER_G_CM3,
    b_scale: float = 1.0,
    d_scale: float = 1.0,
    source: str = DEFAULT_SOURCE,
) -> tuple[np.ndarray, np.ndarray]:
    """Coupled inside/soft volume terms with parent-neutrino attenuation (Eq. 11).

    Generalizes :func:`soft_volume_exact` by folding the parent neutrino's own
    Earth attenuation into the same depth integral that produces the soft
    volume (App. C.2-C.3 of ``docs/2026_softvolume.pdf``), instead of applying
    a decoupled multiplicative survival probability
    (:mod:`softpaws.transport.attenuation`) on top of it.

    App. C.2's soft-volume integral only covers muons produced *upstream*
    (production depth ``xi`` from 0 to ``x``), each weighted by the parent
    neutrino's survival ``e^{-xi/Lambda_nu}`` to that production point:

    .. math:: V_\\mathrm{soft}(x, A) = A_\\mathrm{proj}\\,\\mathcal{I}(A)\\,
        R_\\nu(x, A), \\qquad R_\\nu(x, A) = \\frac{e^{-x/\\Lambda_\\nu}
        - e^{-x\\Phi(A)}}{\\Phi(A) - 1/\\Lambda_\\nu}.

    Muons produced *inside* the detector are produced at ``xi = x`` exactly,
    so they instead pick up a single flat survival factor ``e^{-x/Lambda_nu}``
    -- the parent neutrino's own survival probability to the detector, not the
    integrated ``R_nu``:

    .. math:: V_\\mathrm{det}^\\mathrm{eff}(x, A) = \\mathcal{I}(A)\\,
        e^{-x/\\Lambda_\\nu}\\,V_\\mathrm{det}.

    Together these replace :meth:`softpaws.response.soft_volume.
    SoftVolumeResponse.target_volume_cm3`'s decoupled ``D_nu(E) * [V_det +
    V_soft]`` with the exact ``e^{-x/Lambda_nu} V_det + V_soft(x, A)``, using
    the **unattenuated** flux (see
    :meth:`~softpaws.response.soft_volume.SoftVolumeResponse.
    expected_counts_coupled_attenuation`).

    Parameters
    ----------
    radius_km : float
        Radius of the spherical detector [km].
    energy_gev : float or np.ndarray
        Observed muon energy [GeV].
    gamma : float
        Neutrino flux spectral index, ``phi_nu ~ E^-gamma``.
    column_depth_km : float
        Column depth ``x`` [km] from the surface of the Earth to the detector
        along the relevant direction (small, a couple km of ice, for
        downgoing; up to the full Earth chord for upgoing -- see
        :mod:`softpaws.transport.attenuation`).
    inv_lambda_nu_per_km : float or np.ndarray
        Parent-neutrino attenuation rate ``1/Lambda_nu`` [km^-1] at
        ``energy_gev`` (see
        :func:`softpaws.transport.attenuation.neutrino_interaction_length_km`),
        expressed at the same reference ``density_g_cm3``.
    lam : float, optional
        CC cross-section slope. Defaults to :data:`DEFAULT_LAMBDA`.
    density_g_cm3 : float, optional
        Target-medium density [g cm^-3]. Defaults to water.
    b_scale, d_scale : float, optional
        Multiplicative rescalings of the Table 1 drift and diffusion
        coefficients (Section 3). Default to 1.
    source : {"proposal", "table1"}, optional
        Transport-coefficient tabulation; see
        :mod:`softpaws.transport.coefficients`.

    Returns
    -------
    v_det_eff_cm3 : np.ndarray
        Attenuated inside-detector term ``I(A) e^{-x/Lambda_nu} V_det`` [km^3].
    v_soft_cm3 : np.ndarray
        Coupled soft-volume term ``A_proj I(A) R_nu(x, A)`` [km^3].
    """
    a = spectral_index(gamma, lam)
    b_mu = b_scale * drift_coefficient(energy_gev, density_g_cm3, source)
    d_mu = d_scale * diffusion_coefficient(energy_gev, density_g_cm3, source)
    phi = phi_eigenvalue(a, b_mu, d_mu)
    proj_area = np.pi * radius_km**2
    v_det = 4.0 / 3.0 * np.pi * radius_km**3

    inv_lambda = np.asarray(inv_lambda_nu_per_km, dtype=float)
    factor = inelasticity_factor(a)
    r_nu = saturation_factor(phi, column_depth_km, inv_lambda)

    v_soft = proj_area * factor * r_nu
    # Broadcast rather than fill: v_det_eff otherwise depends on neither the
    # (possibly array-valued) energy nor phi, so it wouldn't inherit their shape.
    v_det_survival = factor * np.exp(-inv_lambda * float(column_depth_km))
    v_det_eff = np.zeros_like(v_soft) + v_det_survival * v_det
    return v_det_eff, v_soft
