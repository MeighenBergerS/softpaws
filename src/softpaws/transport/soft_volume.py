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
from scipy.special import polygamma

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
    two_moment_loss_spectrum,
)
from .source import DEFAULT_LAMBDA, inelasticity_factor

# Muon energy below which a track no longer passes an IceCube-like through-going
# selection. Used as the lower limit of the muon range in
# :func:`range_target_volume_km3`; the resulting volume depends on it only
# logarithmically.
DEFAULT_MUON_THRESHOLD_GEV = 1.0e3


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
    energy = np.atleast_1d(np.asarray(energy_gev, dtype=float))
    b_mu = b_scale * drift_coefficient(energy, density_g_cm3, source)
    e_crit = critical_energy_gev(energy, density_g_cm3, b_scale, source)
    ratio = (energy + e_crit) / (threshold_gev + e_crit)
    return np.clip(np.log(ratio) / b_mu, 0.0, None)


def stochastic_muon_range_km(
    energy_gev: float | np.ndarray,
    threshold_gev: float = DEFAULT_MUON_THRESHOLD_GEV,
    density_g_cm3: float = RHO_WATER_G_CM3,
    b_scale: float = 1.0,
    source: str = DEFAULT_SOURCE,
    ell_max_scale: float = 2.5,
    n_ell: int = 161,
    method: str = "closed",
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

    Returns
    -------
    range_km : np.ndarray
        Expected range [km], zero for muons born below threshold.

    Raises
    ------
    ValueError
        Raised if ``method`` is not one of the two supported values.

    Notes
    -----
    ``b_mu`` and ``d_mu`` are evaluated once, at the production energy, rather
    than followed down the track -- the same percent-level approximation
    :func:`muon_range_km` makes and justifies.

    The two methods agree to better than 2 mm over ``10^4`` to ``10^8`` GeV.
    The closed form is an expansion in ``1 / ln(eps / E_thr)``, so it should
    not be pushed to ``eps -> E_thr``, where the length vanishes anyway.
    """
    if method not in ("closed", "quadrature"):
        raise ValueError(f"method must be 'closed' or 'quadrature', got {method!r}.")

    energy = np.atleast_1d(np.asarray(energy_gev, dtype=float))
    deterministic = muon_range_km(energy, threshold_gev, density_g_cm3, b_scale, source)
    b_mu = b_scale * drift_coefficient(energy, density_g_cm3, source)
    d_mu = diffusion_coefficient(energy, density_g_cm3, source)
    selectable = (energy > threshold_gev) & (deterministic > 0.0)

    if method == "closed":
        kappa, p = two_moment_loss_spectrum(b_mu, d_mu)
        # Phi'(0) = <-ln(1-y)> and -Phi''(0) = <ln^2(1-y)>, both per unit length.
        first = kappa * polygamma(1, p + 1.0)
        second = -kappa * polygamma(2, p + 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            length = np.log(energy / threshold_gev) / first + second / (2.0 * first**2)
        out = np.where(selectable, length, 0.0)
        return out.reshape(np.shape(energy_gev)) if np.ndim(energy_gev) else out

    from .loss_distribution import log_loss_cdf

    out = np.zeros(energy.shape[0])
    for i, eps in enumerate(energy):
        if not selectable[i]:
            continue
        ell = np.linspace(0.0, ell_max_scale * float(deterministic[i]), n_ell)
        cdf = log_loss_cdf(np.log(eps / threshold_gev), ell, float(b_mu[i]), float(d_mu[i]))
        out[i] = float(np.trapezoid(cdf, ell))
    return out.reshape(np.shape(energy_gev)) if np.ndim(energy_gev) else out


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
    ``examples/23_dm_lines.py``.

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
