"""Soft volume in the drift limit.

The soft volume is the effective target region for through-going muon tracks: a
muon produced outside the instrumented volume can still drift into it, so the
target is larger than the detector itself. The drift-limit form below is the
earlier treatment of Palmisano et al. (arXiv:2607.13143, Section 2.3), kept as
a limit and a cross-check of the transport exponent.

In the drift limit (``d_mu -> 0``, deterministic continuous slowing down) and
with their simplifying assumptions -- a spherical perfectly
absorbing detector, a power-law neutrino flux ``phi ~ E^-gamma``, a power-law CC
cross section ``sigma ~ E^lambda``, and a constant near-detector medium -- the
soft volume has the closed form (their Eq. 2.23)

.. math:: V_\\mathrm{soft}(E) = \\frac{\\pi R_\\mathrm{det}^2}{b_\\mu(E)\\,A},
    \\qquad A \\equiv \\gamma - \\lambda - 1 > 0.

The muon range ``1/b_mu`` is further weighted by the spectral penalty ``1/A``
for producing a higher-energy parent neutrino. This module implements that
formula and the associated figure of merit (their Eq. 2.24).

It also provides the *exact* soft volume (:func:`soft_volume_exact`,
``docs/theory/exact_soft_volume.md``), which replaces the drift range ``1/(b_mu A)``
by ``I(A) (1 - e^{-Phi(A) x}) / Phi(A)`` with the exact eigenvalue ``Phi(A)`` and a
finite upstream column depth ``x``. The saturation factor keeps the result finite
where the drift form diverges (``A -> 0``) or goes negative (``A < 0``, the
cross-section pole).

:func:`soft_volume_attenuated_exact` (Eq. 11 and App. C of the method
paper) is the further generalization that folds parent-
neutrino Earth attenuation directly into that same depth integral, rather than
applying a decoupled multiplicative survival probability
(:mod:`softpaws.transport.attenuation`) on top of it -- see
:meth:`softpaws.response.soft_volume.SoftVolumeResponse.expected_counts_coupled_attenuation`.

:func:`scale_breaking_saturation_factor` implements the leading-order
scale-breaking correction of an earlier draft of the method paper (its App. F,
Eq. F4): a nonzero ``beta`` (``dGamma/dy ~ E^beta``,
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
is an apples-to-oranges comparison. The ranges themselves live in
:mod:`softpaws.transport.muon_range`.
"""

from __future__ import annotations

import numpy as np

from ..utils.constants import RHO_WATER_G_CM3
from .coefficients import (
    DEFAULT_SOURCE,
    critical_energy_gev,
    diffusion_coefficient,
    drift_coefficient,
)
from .eigenvalue import (
    phi_eigenvalue,
    phi_eigenvalue_derivative,
    spectral_index,
)
from .muon_range import (
    DEFAULT_MUON_THRESHOLD_GEV,
    muon_range_km,
)
from .source import DEFAULT_LAMBDA, inelasticity_factor


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

    Not part of Palmisano et al. (arXiv:2607.13143), whose soft volume uses a
    fixed projected area ``pi R_det^2``. That is only right if a muon's ability to
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
    """Diffusion-corrected soft volume (Palmisano et al., Eq. 2.25).

    Their drift-diffusion model multiplies the drift soft volume by the
    leading diffusion correction ``1 - d_mu / (2 b_mu)``,

    .. math:: V_\\mathrm{soft}^\\mathrm{diff}(E) = V_\\mathrm{soft}^\\mathrm{drift}(E)
        \\left(1 - \\frac{d_\\mu}{2 b_\\mu}\\right),

    an ``O(d_mu/2 b_mu) ~ 10%`` reduction. This is the ``method="diffusion"``
    forward model used to reproduce the diffusion contours of their Fig. 6.

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
        trigger from beyond ``R_det``, not part of Palmisano et al.). Only the
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
    """Target volume for a monochromatic (dark matter line) source.

    An earlier draft of the method paper carried this as its App. I. A
    delta-function line at ``E_nu = m_chi`` has no continuum spectral index
    to average over, so it excites the ``s = 0`` mode of the same eigenvalue
    formalism the rest of the package uses (``docs/theory/paper_to_code.md``
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
    """Total-to-detector volume ratio in the drift limit (Palmisano et al., Eq. 2.24).

    The total target volume is the instrumented sphere plus its soft volume,
    ``V_tot / V_det = 1 + 3 / (4 R_det b_mu A)``. For the IceCube diffuse flux
    (``A ~ 1``) this recovers their figure of merit
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
    (``docs/theory/exact_soft_volume.md`` Part 7). It is finite for every sign of
    ``Phi``: as ``Phi -> 0`` it tends to ``x`` (no losses, the whole column
    accumulates), and for ``Phi < 0`` it continues to
    ``(e^{|Phi| x} - 1) / |Phi|``, curing the drift-form divergence at and beyond
    the cross-section pole.

    With a nonzero ``inv_lambda_per_km = 1/Lambda_nu``, this is instead the
    method paper's coupled ``R_nu(x, A)`` (Sec. III A and App. A),

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
    """Saturation factor with a running-spectral-index correction.

    App. F of an earlier draft of the method paper treats a scale-breaking
    loss rate ``dGamma/dy = E^beta kappa(y)``
    as a shift operator on the Mellin index, whose leading-order (first
    order in ``beta``) consequence (its Eq. F4)
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
    ``S_hat(xi, s + k beta)`` is only well defined at ``k = 0`` -- that
    draft does not spell out how a single-mode source's higher
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

    Implements ``docs/theory/exact_soft_volume.md`` Part 7:

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
        Scale-breaking exponent (``dGamma/dy ~ E^beta``), applying
        the running-spectral-index correction
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
        infinite-column limit diverges) or while ``beta != 0`` (the
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
                "beta != 0 requires a finite column_depth_km (the running-index "
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
    volume (App. A of the method paper), instead of applying
    a decoupled multiplicative survival probability
    (:mod:`softpaws.transport.attenuation`) on top of it.

    That soft-volume integral only covers muons produced *upstream*
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
