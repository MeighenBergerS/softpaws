"""Earth attenuation of the neutrino flux before it reaches the detector.

The soft-volume forward model (:mod:`softpaws.response.soft_volume`) assumes an
unattenuated flux (``D_nu = 1``, Palmisano et al., arXiv:2607.13143, Eq. 2.4),
which is only valid
for the downgoing hemisphere. Upgoing neutrinos traverse the Earth, and above
~100 TeV the neutrino-nucleon cross section grows enough that the Earth becomes
opaque -- the dominant effect shaping the upgoing UHE spectrum.

The survival probability along a chord of column depth ``X`` [g cm^-2] is

.. math:: D_\\nu(E) = \\exp\\!\\bigl[-N_A\\,\\sigma_\\mathrm{tot}(E)\\,X\\bigr],

with ``sigma_tot`` the total (CC + NC) neutrino-nucleon cross section. That is
pure absorption -- a ``nu_mu`` disappearance picture -- which is right for the
flux at a fixed energy but not for a tabulated effective area, where a neutrino
that scattered down on the way in still counts towards its original energy.
:func:`regenerated_transmission` and :func:`flavour_transmission` keep the
down-scattered population instead: the former on a geometric ladder for
``nu_mu``, the latter on a log-energy grid that also covers ``nu_tau``, whose
charged-current interaction regenerates the neutrino rather than terminating it
and so leaves the Earth far more transparent at UHE.

This module provides two columns:

- a **constant mean-density** chord (:func:`mean_density_column`), whose
  solid-angle average over a declination band (:func:`representative_column`)
  gives the single scalar used by the closed-form attenuation of
  :class:`~softpaws.response.soft_volume.SoftVolumeResponse`;
- a **layered PREM** chord (:func:`prem_column`), used by the per-event
  effective solid angle (:func:`effective_solid_angle`) that
  :meth:`~softpaws.response.soft_volume.SoftVolumeResponse.expected_counts_attenuated`
  integrates over the band.

Both of the above (Forms A and B) apply ``D_nu`` as a flat multiplicative
factor on the flux, decoupled from the muon-transport saturation factor of
:mod:`softpaws.transport.soft_volume`. This is the paper's transparent-Earth
approximation (Eq. 9-10) plus a bolted-on survival probability, valid only
when ``D_nu`` varies slowly over the ~few-``Phi(A)^-1`` km-w.e. range the soft
volume is produced in. :func:`neutrino_interaction_length_km` instead gives
``Lambda_nu`` in the same km^-1 units as ``Phi(A)``, letting
:func:`softpaws.transport.soft_volume.saturation_factor` combine the two
analytically into the paper's exact ``R_nu(x, A)`` (Eq. 11, App. C.2-C.3) --
**Form C**, used by
:meth:`~softpaws.response.soft_volume.SoftVolumeResponse.expected_counts_coupled_attenuation`
for the strongly-absorbed upgoing UHE regime where Forms A/B are an
uncontrolled approximation.

Geometry follows the South Pole detector convention (see
``docs/theory/soft_volume.md``): a source at declination ``dec`` arrives at
zenith ``90 deg + dec``, so the upgoing hemisphere is ``dec > 0``. The Earth
chord for a (near-)surface detector is ``L(dec) = 2 R_Earth sin(dec)``: it
vanishes at the horizon (``dec = 0``) and reaches the full diameter for
vertically upgoing tracks (``dec = 90 deg``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..utils.constants import (
    AVOGADRO_PER_MOL,
    CM_PER_KM,
    RHO_WATER_G_CM3,
)
from ..utils.constants import TOTAL_TO_CC_RATIO as _TOTAL_TO_CC_RATIO
from .earth import (  # noqa: F401  (re-exported; these lived here before transport.earth)
    earth_chord_length_km,
    mean_density_column,
    prem_column,
    prem_density,
    representative_column,
)
from .source import DEFAULT_LAMBDA, cc_cross_section

if TYPE_CHECKING:  # avoids a circular import: cross_section imports from source
    from .cross_section import CrossSection

# Total-to-CC cross-section ratio. The neutral-current channel adds ~40% to the
# charged-current cross section at UHE, so sigma_tot ~ 1.4 sigma_CC; the flux is
# attenuated by both channels while detection uses CC only.
TOTAL_TO_CC_RATIO = _TOTAL_TO_CC_RATIO


def total_cross_section(
    energy_gev: float | np.ndarray,
    lam: float = DEFAULT_LAMBDA,
    cross_section: "CrossSection | None" = None,
) -> np.ndarray:
    """Total neutrino-nucleon cross section for Earth attenuation.

    By default reuses the charged-current power law of
    :func:`softpaws.transport.source.cc_cross_section` scaled by
    :data:`TOTAL_TO_CC_RATIO` to include the neutral-current channel. The
    tabulated ratio runs 1.41-1.47 over eight decades, so that constant is good
    to a few percent and essentially all of this model's error comes from the
    power law itself.

    Parameters
    ----------
    energy_gev : float or np.ndarray
        Neutrino energy [GeV].
    lam : float, optional
        Cross-section slope of the power-law model. Defaults to
        :data:`~softpaws.transport.source.DEFAULT_LAMBDA`. Ignored when
        ``cross_section`` is given.
    cross_section : softpaws.transport.cross_section.CrossSection, optional
        Explicit cross-section model, whose own CC and NC channels are summed.
        ``None`` (the default) uses the scaled power law.

    Returns
    -------
    sigma : np.ndarray
        Total cross section per nucleon [cm^2].
    """
    if cross_section is not None:
        return cross_section.total(energy_gev)
    return TOTAL_TO_CC_RATIO * cc_cross_section(energy_gev, lam)


def survival_probability(
    energy_gev: float | np.ndarray,
    column_g_cm2: float | np.ndarray,
    lam: float = DEFAULT_LAMBDA,
    cross_section: "CrossSection | None" = None,
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
    cross_section : softpaws.transport.cross_section.CrossSection, optional
        Explicit cross-section model; see :func:`total_cross_section`. The
        survival probability is exponential in the cross section, so this is
        the most cross-section-sensitive quantity in the package.

    Returns
    -------
    d_nu : np.ndarray
        Survival probability ``exp(-N_A sigma_tot X)``, in ``[0, 1]``.
    """
    sigma = total_cross_section(energy_gev, lam, cross_section)
    optical_depth = AVOGADRO_PER_MOL * sigma * np.asarray(column_g_cm2, dtype=float)
    return np.exp(-optical_depth)


def neutrino_interaction_length_km(
    energy_gev: float | np.ndarray,
    density_g_cm3: float = RHO_WATER_G_CM3,
    lam: float = DEFAULT_LAMBDA,
    cross_section: "CrossSection | None" = None,
) -> np.ndarray:
    """Neutrino interaction length ``Lambda_nu``, in the transport's length units.

    ``Lambda_nu = 1 / (N_A sigma_tot(E) rho)``, expressed in km at the given
    reference density (water by default, giving the km-water-equivalent
    convention Palmisano et al. quote: ``Lambda_nu ~ O(10^3) km.w.e.``). This
    matches the km^-1 units :func:`softpaws.transport.eigenvalue.phi_eigenvalue`
    already uses for ``Phi(A)``, so the two combine directly in
    :func:`softpaws.transport.soft_volume.saturation_factor`'s coupled form
    (Eq. 11 of ``paper/main.tex``).

    Parameters
    ----------
    energy_gev : float or np.ndarray
        Neutrino energy [GeV].
    density_g_cm3 : float, optional
        Reference medium density [g cm^-3] setting the length unit. Defaults to
        water, matching the default ``density_g_cm3`` of
        :func:`~softpaws.transport.soft_volume.soft_volume_exact`.
    lam : float, optional
        Cross-section slope. Defaults to
        :data:`~softpaws.transport.source.DEFAULT_LAMBDA`.
    cross_section : softpaws.transport.cross_section.CrossSection, optional
        Explicit cross-section model; see :func:`total_cross_section`.

    Returns
    -------
    lambda_nu : np.ndarray
        Interaction length [km] at the reference density.
    """
    sigma = total_cross_section(energy_gev, lam, cross_section)
    inv_lambda_per_km = AVOGADRO_PER_MOL * sigma * density_g_cm3 * CM_PER_KM
    return 1.0 / inv_lambda_per_km


def effective_solid_angle(
    energy_gev: np.ndarray,
    dec_min_deg: float,
    dec_max_deg: float,
    lam: float = DEFAULT_LAMBDA,
    n_dec: int = 64,
    cross_section: "CrossSection | None" = None,
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
    cross_section : softpaws.transport.cross_section.CrossSection, optional
        Explicit cross-section model; see :func:`total_cross_section`.

    Returns
    -------
    omega_eff : np.ndarray, shape (energy_gev.size,)
        Effective solid angle [sr] at each energy.
    """
    energy = np.atleast_1d(np.asarray(energy_gev, dtype=float))
    dec = np.linspace(dec_min_deg, dec_max_deg, n_dec)
    columns = np.array([prem_column(d) for d in dec])  # [g cm^-2], shape (n_dec,)

    # D_nu[energy, dec] and the solid-angle weight 2 pi cos(dec).
    d_nu = survival_probability(energy[:, None], columns[None, :], lam, cross_section)
    weight = 2.0 * np.pi * np.cos(np.deg2rad(dec))
    integrand = d_nu * weight[None, :]
    return np.trapezoid(integrand, np.deg2rad(dec), axis=1)


# ---------------------------------------------------------------------------
# Neutral-current regeneration
# ---------------------------------------------------------------------------

# Mean neutral-current inelasticity at UHE. The NC and CC inelasticity
# distributions have very similar shapes above a TeV; this is the NC mean,
# slightly above the CC :data:`~softpaws.transport.source.MEAN_INELASTICITY`
# because NC samples a marginally harder part of the same parton kinematics.
NC_MEAN_INELASTICITY = 0.25

# Rungs of the down-scattering ladder. Each rung is a factor (1 - <y>) in
# energy, so the default reaches 0.25% of the injected energy -- far below the
# point where the surviving neutrino can still make a selectable muon, so the
# result is insensitive to adding more.
NC_REGENERATION_LEVELS = 24


def regenerated_transmission(
    energy_gev: float,
    column_g_cm2: float | np.ndarray,
    cross_section: "CrossSection | None" = None,
    lam: float = DEFAULT_LAMBDA,
    mean_inelasticity: float = NC_MEAN_INELASTICITY,
    n_levels: int = NC_REGENERATION_LEVELS,
) -> tuple[np.ndarray, np.ndarray]:
    """Arriving-neutrino energy ladder and weights, with NC down-scattering kept.

    :func:`survival_probability` is pure absorption: it removes the neutrino on
    *any* interaction, charged- or neutral-current. That is right for the flux
    at a fixed energy, but wrong for a tabulated effective area. A neutrino that
    neutral-current scatters is not gone -- it continues at lower energy and can
    still interact charged-current near the detector, and the published response
    credits the resulting event to the **original** energy. Absorbing it
    instead is what makes a pure-``sigma_tot`` model undershoot the published
    ``A_eff`` at UHE, where the Earth is many interaction lengths deep.

    This solves the coupled transport down a discrete energy ladder
    ``E_k = E (1 - <y>)^k``, where the neutral-current interaction is modelled
    as a fixed fractional energy loss (the same delta-at-the-mean treatment
    :func:`softpaws.transport.source.inelasticity_factor` uses for the weak
    vertex):

    .. math:: \\frac{d\\varphi_k}{dX} = -N_A\\,\\sigma_\\mathrm{tot}(E_k)\\,
        \\varphi_k + N_A\\,\\sigma_\\mathrm{NC}(E_{k-1})\\,\\varphi_{k-1},
        \\qquad \\varphi_k(0) = \\delta_{k0}.

    The generator is lower-bidiagonal with distinct diagonal entries, so it is
    diagonalized once and evaluated at every requested column at no extra cost.

    Parameters
    ----------
    energy_gev : float
        Injected neutrino energy [GeV] at the surface.
    column_g_cm2 : float or np.ndarray
        Traversed column depth(s) ``X`` [g cm^-2].
    cross_section : softpaws.transport.cross_section.CrossSection, optional
        Cross-section model providing separate ``cc`` and ``nc`` channels.
        ``None`` (the default) falls back to the analytic power law with the
        fixed :data:`TOTAL_TO_CC_RATIO` split.
    lam : float, optional
        Cross-section slope of the power-law fallback. Ignored when
        ``cross_section`` is given.
    mean_inelasticity : float, optional
        Mean NC inelasticity ``<y>``. Defaults to
        :data:`NC_MEAN_INELASTICITY`.
    n_levels : int, optional
        Number of rungs on the ladder. Defaults to
        :data:`NC_REGENERATION_LEVELS`; ``1`` reproduces
        :func:`survival_probability` exactly (no regeneration).

    Returns
    -------
    energies_gev : np.ndarray, shape (n_levels,)
        Arriving energies ``E (1 - <y>)^k`` [GeV].
    weights : np.ndarray, shape (n_levels, n_column)
        Probability of arriving at each rung without having interacted
        charged-current, one column per entry of ``column_g_cm2``. Summing over
        rungs gives the total non-CC survival, which exceeds
        :func:`survival_probability` by exactly the regenerated population.

    Notes
    -----
    Tau regeneration is still not modelled -- this is a ``nu_mu`` ladder only.
    Neglecting the spread of the NC inelasticity distribution smears the ladder
    less than the underlying cross-section uncertainty at these energies.
    """
    columns = np.atleast_1d(np.asarray(column_g_cm2, dtype=float))
    k = np.arange(int(n_levels))
    energies = float(energy_gev) * (1.0 - float(mean_inelasticity)) ** k

    if cross_section is not None:
        sigma_cc = np.atleast_1d(cross_section.cc(energies))
        sigma_nc = np.atleast_1d(cross_section.nc(energies))
    else:
        sigma_cc = np.atleast_1d(cc_cross_section(energies, lam))
        sigma_nc = (TOTAL_TO_CC_RATIO - 1.0) * sigma_cc
    loss = AVOGADRO_PER_MOL * (sigma_cc + sigma_nc)
    feed = AVOGADRO_PER_MOL * sigma_nc

    if n_levels == 1:
        return energies, np.exp(-loss[0] * columns)[None, :]

    generator = np.diag(-loss) + np.diag(feed[:-1], -1)
    # Lower-bidiagonal with distinct diagonal entries, so the eigenvalues are
    # the diagonal itself and one decomposition serves every column depth.
    eigenvalues, vectors = np.linalg.eig(generator)
    start = np.zeros(int(n_levels))
    start[0] = 1.0
    coefficients = np.linalg.solve(vectors, start)
    weights = vectors @ (np.exp(np.outer(eigenvalues, columns)) * coefficients[:, None])
    return energies, np.clip(np.real(weights), 0.0, None)


# Mean fraction of the tau energy carried away by the regenerated tau neutrino
# in tau -> nu_tau X. Distinct from softpaws.transport.tau.MEAN_Z, which is the
# fraction carried by the *muon* in the leptonic channel.
TAU_TO_NUTAU_ENERGY_FRACTION = 0.4


def flavour_transmission(
    energy_gev: float,
    column_g_cm2: float | np.ndarray,
    cross_section: "CrossSection | None" = None,
    lam: float = DEFAULT_LAMBDA,
    flavour: str = "mu",
    decades: float = 5.0,
    n_grid: int = 100,
    mean_inelasticity_nc: float = NC_MEAN_INELASTICITY,
    mean_inelasticity_cc: float = 0.2,
    nu_tau_fraction: float = TAU_TO_NUTAU_ENERGY_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Arriving-neutrino spectrum after a column, per flavour, on a log-energy grid.

    Generalizes :func:`regenerated_transmission`, which handles ``nu_mu`` on a
    geometric ladder of a single step size. The ``nu_tau`` cascade needs two
    step sizes at once -- neutral current moves the neutrino by
    ``1 - <y>_NC``, whereas charged current makes a tau that promptly decays
    back to a ``nu_tau`` at roughly ``<z_nu> (1 - <y>_CC)`` of the original
    energy -- so a single-ratio ladder cannot represent it and a log-energy
    grid with interpolated feed is used instead.

    The distinction between the flavours is what charged current does:

    ``"mu"``
        Charged current is terminal. A ``nu_mu`` that interacts charged-current
        makes a muon, which at these energies loses energy rather than decaying,
        so the neutrino is gone. Only neutral current regenerates.
    ``"tau"``
        Charged current is *not* terminal. The tau it makes decays back to a
        ``nu_tau`` before losing much energy, so the Earth stays far more
        transparent to ``nu_tau`` than to ``nu_mu`` at UHE. Both channels
        regenerate.

    Parameters
    ----------
    energy_gev : float
        Injected neutrino energy [GeV] at the surface.
    column_g_cm2 : float or np.ndarray
        Traversed column depth(s) ``X`` [g cm^-2].
    cross_section : softpaws.transport.cross_section.CrossSection, optional
        Cross-section model providing separate ``cc`` and ``nc`` channels.
        ``None`` (the default) uses the analytic power law with the fixed
        :data:`TOTAL_TO_CC_RATIO` split.
    lam : float, optional
        Cross-section slope of the power-law fallback.
    flavour : {"mu", "tau"}, optional
        Which cascade to solve; see above. Defaults to ``"mu"``.
    decades : float, optional
        Span of the log-energy grid below the injected energy.
    n_grid : int, optional
        Number of grid points over that span.
    mean_inelasticity_nc, mean_inelasticity_cc : float, optional
        Mean neutral- and charged-current inelasticities.
    nu_tau_fraction : float, optional
        Mean ``E_nu / E_tau`` in the tau decay, used only for ``flavour="tau"``.
        Defaults to :data:`TAU_TO_NUTAU_ENERGY_FRACTION`.

    Returns
    -------
    energies_gev : np.ndarray, shape (n_grid,)
        Grid energies [GeV], descending from ``energy_gev``.
    weights : np.ndarray, shape (n_grid, n_column)
        Probability of arriving at each grid energy without having been
        removed, one column per entry of ``column_g_cm2``.

    Raises
    ------
    ValueError
        Raised if ``flavour`` is not ``"mu"`` or ``"tau"``.

    Notes
    -----
    The tau is treated as decaying where it was produced. Its decay length
    reaches ~5 km w.e. at 100 PeV against a ~2700 km w.e. interaction length,
    so the displacement is negligible for the transmission, though not for the
    muon that the decay produces.

    Feed that lands between grid points is split linearly in ``log E`` between
    the two neighbours, which conserves the total rate exactly and the mean
    log-energy to the grid resolution.
    """
    if flavour not in ("mu", "tau"):
        raise ValueError(f"flavour must be 'mu' or 'tau', got {flavour!r}.")

    columns = np.atleast_1d(np.asarray(column_g_cm2, dtype=float))
    log_energies = np.log(float(energy_gev)) - np.linspace(
        0.0, float(decades) * np.log(10.0), int(n_grid)
    )
    energies = np.exp(log_energies)
    step = log_energies[0] - log_energies[1]

    if cross_section is not None:
        sigma_cc = np.atleast_1d(cross_section.cc(energies))
        sigma_nc = np.atleast_1d(cross_section.nc(energies))
    else:
        sigma_cc = np.atleast_1d(cc_cross_section(energies, lam))
        sigma_nc = (TOTAL_TO_CC_RATIO - 1.0) * sigma_cc

    generator = np.diag(-AVOGADRO_PER_MOL * (sigma_cc + sigma_nc))

    def _feed(rate: np.ndarray, fraction: float) -> None:
        """Scatter ``rate`` from each node down to ``fraction`` of its energy."""
        offset = -np.log(fraction) / step
        low = np.floor(offset).astype(int)
        frac = offset - low
        for i in range(int(n_grid)):
            for target, share in ((low + i, 1.0 - frac), (low + i + 1, frac)):
                if 0 <= target < n_grid and share > 0.0:
                    generator[target, i] += rate[i] * share

    _feed(AVOGADRO_PER_MOL * sigma_nc, 1.0 - mean_inelasticity_nc)
    if flavour == "tau":
        _feed(
            AVOGADRO_PER_MOL * sigma_cc,
            nu_tau_fraction * (1.0 - mean_inelasticity_cc),
        )

    eigenvalues, vectors = np.linalg.eig(generator)
    start = np.zeros(int(n_grid))
    start[0] = 1.0
    coefficients = np.linalg.solve(vectors, start)
    weights = vectors @ (np.exp(np.outer(eigenvalues, columns)) * coefficients[:, None])
    return energies, np.clip(np.real(weights), 0.0, None)
