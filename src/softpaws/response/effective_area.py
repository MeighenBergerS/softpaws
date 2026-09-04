"""Effective areas from the soft-volume transport, for one site at a time.

The per-neutrino-energy effective area of Sec. VI of the paper is the
nucleon density, the charged-current cross section and a target volume: the
projected area of the instrumented body times the muon range, plus the body
itself. Nothing in that construction is site-specific. The transport exponent,
the first-passage range and the flavour-dependent Earth transmission are
properties of the loss kernel and of the Earth, so the same functions serve
IceCube and the water detectors given only each site's instrument numbers.

Two families live here. The ``ic_*`` functions average over the upgoing
hemisphere seen from the Pole on a declination grid, as the IceTracks-DR2
table does. :func:`arca_effective_area` averages over a band of ``cos(theta)``
on the full sky with a finite overburden, as the KM3NeT, P-ONE and TRIDENT
figures do. Both keep the neutral-current-degraded population on an energy
ladder and credit every rung to the surface energy, which is how a published
effective area is built.

The two instrument numbers a published curve can be inverted for, the
selection threshold and the light reach, enter as arguments. Nothing here is
fitted; :func:`fit_reach_law` is the one-line least-squares step the examples
run on the radii :func:`ic_required_radius_km` and
:func:`required_footprint_radius_km` return.

Notes
-----
``cos_theta = +1`` is vertically downgoing and ``-1`` the nadir, as in
:mod:`softpaws.transport.earth`. Effective areas are in cm^2, lengths in km,
energies in GeV.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq

from ..detectors import ARCA230, ICECUBE, MAX_UPSTREAM_KM
from ..transport.attenuation import flavour_transmission, regenerated_transmission
from ..transport.coefficients import diffusion_coefficient, drift_coefficient
from ..transport.cross_section import CrossSection, bgr18_cross_section
from ..transport.earth import neutrino_column_g_cm2, overburden_km, prem_column, zenith_grid
from ..transport.loss_distribution import log_loss_cdf
from ..transport.soft_volume import (
    light_reach_radius_km,
    muon_range_km,
    prism_projected_area_km2,
    truncated_muon_range_km,
)
from ..transport.source import MEAN_INELASTICITY, nucleon_number_density
from ..transport.tau import BR_TAU_TO_MU, MEAN_Z
from ..utils.constants import CM_PER_KM, RHO_WATER_G_CM3

__all__ = [
    "ARCA_FIT_BAND",
    "ARCA_LOG10_E",
    "IC_FIT_BAND",
    "IC_LOG10_E",
    "N_DEC",
    "N_ZENITH",
    "arca_effective_area",
    "cylinder_projected_area_km2",
    "default_cross_section",
    "effective_volume_km3",
    "first_passage_length_table",
    "fit_reach_law",
    "ic_effective_area_regenerated",
    "ic_effective_area_tau_channel",
    "ic_mean_target_volume_cm3",
    "ic_required_radius_km",
    "ic_target_volume_cm3",
    "ic_upgoing_columns",
    "required_footprint_radius_km",
    "truncated_range_from_table_km",
    "truncated_range_km",
]

#: Neutrino-energy grid of the IceCube curves [log10 GeV]: 0.2 dex from 1 TeV
#: to 100 PeV, the span of the DR2 effective-area table.
IC_LOG10_E = np.linspace(3.0, 8.0, 26)

#: Band of :data:`IC_LOG10_E` the IceCube reach law is calibrated over
#: [log10 GeV]. Below 100 TeV the DR2 selection is still turning on.
IC_FIT_BAND = (5.0, 7.8)

#: Number of declination slices of the upgoing hemisphere.
N_DEC = 60

#: Neutrino-energy grid of the water-site curves [log10 GeV]: 0.2 dex from
#: 10 TeV to 10 EeV, the span of the ARCA figures.
ARCA_LOG10_E = np.arange(4.0, 10.01, 0.2)

#: Band of :data:`ARCA_LOG10_E` the ARCA230 reach law is calibrated over
#: [log10 GeV]. The digitized trigger curve saturates at the edge of the
#: published figure in its last tenth of a decade, so the fit stops short.
ARCA_FIT_BAND = (4.0, 7.5)

#: Number of equal-``cos(theta)`` slices of the sky.
N_ZENITH = 90


@functools.lru_cache(maxsize=1)
def default_cross_section() -> CrossSection:
    """The BGR18 tabulated cross section, built once and shared.

    Returns
    -------
    cross_section : CrossSection
        :func:`~softpaws.transport.cross_section.bgr18_cross_section` with its
        default table.
    """
    return bgr18_cross_section()


def _cross_section(cross_section: CrossSection | None) -> CrossSection:
    return default_cross_section() if cross_section is None else cross_section


# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------


def fit_reach_law(
    log10_e: np.ndarray, required_radius_km: np.ndarray, radius_km: float
) -> tuple[float, float]:
    """Least-squares reach law through the radii a published curve demands.

    :func:`ic_required_radius_km` and :func:`required_footprint_radius_km`
    invert a published effective area for the footprint that would reproduce
    it. Those radii are close to linear in ``ln E``, so one straight-line fit
    fixes both parameters of
    :func:`~softpaws.transport.soft_volume.light_reach_radius_km` without
    running the forward model.

    Parameters
    ----------
    log10_e : np.ndarray
        ``log10(E_nu / GeV)`` of the points to fit.
    required_radius_km : np.ndarray
        Radius each point demands [km]; ``NaN`` entries are dropped.
    radius_km : float
        Instrumented radius [km], used to locate the pivot.

    Returns
    -------
    reach_km : float
        Growth of the reach per e-fold of energy [km].
    pivot_gev : float
        Energy at which the effective radius equals ``radius_km`` [GeV].
    """
    valid = np.isfinite(required_radius_km)
    ln_e = np.log(10.0 ** np.asarray(log10_e)[valid])
    slope, intercept = np.polyfit(ln_e, np.asarray(required_radius_km)[valid], 1)
    return float(slope), float(np.exp((radius_km - intercept) / slope))


def truncated_range_km(
    energy_mu_gev: np.ndarray,
    column_km: np.ndarray,
    threshold_gev: float,
    kernel_evaluation: str = "running",
    density_g_cm3: float = RHO_WATER_G_CM3,
) -> np.ndarray:
    """Truncated first-passage range ``E[tau ^ X]`` on an energy-by-column grid [km].

    Eq. (16) of the paper is ``L = Integral_0^inf d_ell P[W(ell) < w_star]``.
    Cutting the integral at a finite ``X`` gives ``E[tau(w_star) ^ X]``, the
    length available when the muon cannot be born further upstream than
    ``X``. This is :func:`~softpaws.transport.soft_volume.truncated_muon_range_km`
    broadcast to a ``(n_energy, n_column)`` table and clipped at zero.

    Parameters
    ----------
    energy_mu_gev : np.ndarray, shape (n,)
        Muon energy at production [GeV].
    column_km : np.ndarray, shape (m,)
        Available upstream column ``X``, as a length of the medium [km];
        ``inf`` returns the untruncated length.
    threshold_gev : float
        Muon selection threshold [GeV].
    kernel_evaluation : {"running", "frozen"}, optional
        Where along the descent the loss kernel is read.
    density_g_cm3 : float, optional
        Medium density [g cm^-3]. Defaults to water.

    Returns
    -------
    length : np.ndarray, shape (n, m)
        Effective length [km].
    """
    energy = np.atleast_1d(np.asarray(energy_mu_gev, dtype=float))
    column = np.atleast_1d(np.asarray(column_km, dtype=float))
    return np.clip(
        truncated_muon_range_km(
            energy[:, None],
            column[None, :],
            threshold_gev,
            density_g_cm3=density_g_cm3,
            kernel_evaluation=kernel_evaluation,
        ),
        0.0,
        None,
    )


def first_passage_length_table(
    energy_mu_gev: np.ndarray, threshold_gev: float, n_ell: int = 401
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative first-passage length against depth, frozen two-moment kernel.

    Integrates ``P[W(ell) < w_star]`` from the Gil-Pelaez inversion of
    :func:`~softpaws.transport.loss_distribution.log_loss_cdf` on a depth grid,
    with the drift and diffusion coefficients read at the production energy.
    This is the route :func:`truncated_range_km` superseded: it reads the loss
    moments off the two-moment family and freezes them at production, which is
    8% low on ``Phi'(0)`` and 56% low on ``-Phi''(0)``. It is kept for the
    contrast curve of example 31, whose numbers were built on it, and is not
    the range the paper quotes.

    Parameters
    ----------
    energy_mu_gev : np.ndarray, shape (n,)
        Muon energies at production [GeV], the rows of the table.
    threshold_gev : float
        Muon selection threshold [GeV].
    n_ell : int, optional
        Number of depth nodes. The grid runs to 2.5 times the longest
        continuous-slowing-down range in ``energy_mu_gev``.

    Returns
    -------
    ell_km : np.ndarray, shape (n_ell,)
        Depth nodes [km].
    cumulative_km : np.ndarray, shape (n, n_ell)
        ``Integral_0^ell P[W < w_star]`` at each node [km]; zero rows for
        energies at or below threshold.
    """
    energy = np.asarray(energy_mu_gev, dtype=float)
    deterministic = muon_range_km(energy, threshold_gev)
    ell_km = np.linspace(0.0, 2.5 * float(np.max(deterministic)), n_ell)
    b_mu = np.atleast_1d(drift_coefficient(energy))
    d_mu = np.atleast_1d(diffusion_coefficient(energy))
    cumulative = np.zeros((energy.size, n_ell))
    for i, eps in enumerate(energy):
        if eps <= threshold_gev:
            continue
        cdf = log_loss_cdf(np.log(eps / threshold_gev), ell_km, float(b_mu[i]), float(d_mu[i]))
        cumulative[i] = cumulative_trapezoid(cdf, ell_km, initial=0.0)
    return ell_km, cumulative


def truncated_range_from_table_km(
    energy_mu_gev: np.ndarray,
    column_km: np.ndarray,
    grid_log10_e: np.ndarray,
    ell_km: np.ndarray,
    cumulative_km: np.ndarray,
) -> np.ndarray:
    """Truncated first-passage range ``E[tau ^ X]`` by table lookup [km].

    Reads :func:`first_passage_length_table` at the column, then interpolates
    across the table's energy rows in ``log10(E)``.

    Parameters
    ----------
    energy_mu_gev : np.ndarray, shape (n,)
        Muon energy at production [GeV].
    column_km : np.ndarray, shape (m,)
        Available upstream column [km]; values beyond the table's last node
        are clipped to it.
    grid_log10_e : np.ndarray
        ``log10(E_mu / GeV)`` of the table rows.
    ell_km, cumulative_km : np.ndarray
        Output of :func:`first_passage_length_table`.

    Returns
    -------
    length : np.ndarray, shape (n, m)
        Effective length [km].
    """
    x = np.minimum(np.atleast_1d(column_km), ell_km[-1])
    per_row = np.array([np.interp(x, ell_km, row) for row in cumulative_km])
    log10_e = np.log10(np.atleast_1d(energy_mu_gev))
    out = np.empty((log10_e.size, x.size))
    for j in range(x.size):
        out[:, j] = np.interp(log10_e, grid_log10_e, per_row[:, j])
    return np.clip(out, 0.0, None)


# ---------------------------------------------------------------------------
# IceCube: upgoing hemisphere on a declination grid
# ---------------------------------------------------------------------------


def ic_upgoing_columns(n_dec: int = N_DEC) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PREM columns, solid-angle weights and zenith cosines, upgoing hemisphere.

    The hemisphere is cut into ``n_dec`` declination slices of equal width
    between 0.5 and 89.5 degrees. IceCube sits at the Pole, so a source at
    declination ``dec`` arrives at ``|cos theta_z| = sin(dec)``.

    Parameters
    ----------
    n_dec : int, optional
        Number of declination slices. Defaults to :data:`N_DEC`.

    Returns
    -------
    columns : np.ndarray, shape (n_dec,)
        Layered-PREM Earth column of each slice [g cm^-2].
    weights : np.ndarray, shape (n_dec,)
        Solid-angle weight of each slice, ``cos(dec)``, not normalized.
    cos_theta : np.ndarray, shape (n_dec,)
        ``|cos theta_z| = sin(dec)`` of each slice.
    """
    dec_deg = np.linspace(0.5, 89.5, n_dec)
    columns = np.array([prem_column(float(d)) for d in dec_deg])
    dec_rad = np.deg2rad(dec_deg)
    return columns, np.cos(dec_rad), np.sin(dec_rad)


def ic_target_volume_cm3(
    length_km: np.ndarray,
    radius_km: float | np.ndarray = ICECUBE.radius_km,
    cos_theta: float | np.ndarray = 0.0,
    height_km: float = ICECUBE.height_km,
    n_sides: int | None = ICECUBE.n_sides,
) -> np.ndarray:
    """Prism target volume, projected column plus the body itself [cm^3].

    Parameters
    ----------
    length_km : np.ndarray
        Muon range to threshold [km].
    radius_km : float or np.ndarray, optional
        Area-equivalent footprint radius [km], broadcast against
        ``length_km``. Defaults to the IceCube layout.
    cos_theta : float or np.ndarray, optional
        Cosine of the arrival zenith. An array is appended as a trailing axis,
        since the projected area depends on the arrival direction where the
        instrumented volume does not.
    height_km : float, optional
        Instrumented height [km]. Defaults to the IceCube layout.
    n_sides : int or None, optional
        Sides of the prism cross-section; ``None`` for a cylinder. Defaults
        to the IceCube hexagon.

    Returns
    -------
    volume : np.ndarray
        ``A_proj(cos_theta) L + pi R^2 h`` [cm^3].
    """
    radius = np.asarray(radius_km, dtype=float)
    length = np.asarray(length_km, dtype=float)
    zenith = np.asarray(cos_theta, dtype=float)
    if zenith.ndim:
        radius = radius[..., None]
        length = length[..., None]
    proj_area = prism_projected_area_km2(zenith, radius, height_km, n_sides)
    return (proj_area * length + np.pi * radius**2 * height_km) * CM_PER_KM**3


def ic_mean_target_volume_cm3(
    length_km: np.ndarray,
    radius_km: float | np.ndarray = ICECUBE.radius_km,
    n_dec: int = N_DEC,
    height_km: float = ICECUBE.height_km,
    n_sides: int | None = ICECUBE.n_sides,
) -> np.ndarray:
    """Target volume averaged over the upgoing hemisphere [cm^3].

    Parameters
    ----------
    length_km : np.ndarray
        Muon range to threshold [km].
    radius_km : float or np.ndarray, optional
        Footprint radius [km]; see :func:`ic_target_volume_cm3`.
    n_dec : int, optional
        Declination slices of :func:`ic_upgoing_columns`.
    height_km : float, optional
        Instrumented height [km].
    n_sides : int or None, optional
        Sides of the prism cross-section.

    Returns
    -------
    volume : np.ndarray
        Solid-angle mean of :func:`ic_target_volume_cm3` [cm^3].
    """
    _, weights, cos_theta = ic_upgoing_columns(n_dec)
    return np.average(
        ic_target_volume_cm3(length_km, radius_km, cos_theta, height_km, n_sides),
        axis=-1,
        weights=weights,
    )


def ic_required_radius_km(
    ratio: np.ndarray,
    lengths: np.ndarray,
    n_dec: int = N_DEC,
    height_km: float = ICECUBE.height_km,
    n_sides: int | None = ICECUBE.n_sides,
) -> np.ndarray:
    """Footprint radius that would scale the target volume by ``ratio``.

    Inverts :func:`ic_mean_target_volume_cm3` at each energy for the radius
    that multiplies the instrumented-radius volume by ``ratio``, holding the
    height fixed.

    Parameters
    ----------
    ratio : np.ndarray
        Required scaling of the mean target volume, one per energy.
    lengths : np.ndarray
        Muon range at each energy [km].
    n_dec : int, optional
        Declination slices of :func:`ic_upgoing_columns`.
    height_km : float, optional
        Instrumented height [km].
    n_sides : int or None, optional
        Sides of the prism cross-section.

    Returns
    -------
    radius : np.ndarray
        Required radius [km], ``NaN`` where ``ratio`` is not finite or not
        positive.
    """
    out = np.full(np.shape(ratio), np.nan)
    for i, (r, length) in enumerate(zip(np.atleast_1d(ratio), lengths)):
        if not np.isfinite(r) or r <= 0.0:
            continue
        one = np.array([length])
        target = r * float(ic_mean_target_volume_cm3(one, n_dec=n_dec, height_km=height_km,
                                                     n_sides=n_sides)[0])
        out[i] = brentq(
            lambda x: float(ic_mean_target_volume_cm3(one, x, n_dec, height_km, n_sides)[0])
            - target,
            1.0e-4,
            50.0,
        )
    return out


def ic_effective_area_regenerated(
    length_km: np.ndarray,
    threshold_gev: float,
    reach_km: float | None = None,
    pivot_gev: float = 1.0e6,
    log10_e: np.ndarray = IC_LOG10_E,
    cross_section: CrossSection | None = None,
    n_dec: int = N_DEC,
    radius_km: float = ICECUBE.radius_km,
    height_km: float = ICECUBE.height_km,
    n_sides: int | None = ICECUBE.n_sides,
    mask_subthreshold: bool = True,
) -> np.ndarray:
    """Direct ``nu_mu`` channel, NC regeneration kept, upgoing-averaged [cm^2].

    Each surface energy is spread over the ladder of
    :func:`~softpaws.transport.attenuation.regenerated_transmission`; every
    rung's muon gets its own range, read off ``length_km`` in ``log10(E)``,
    and its own reach-dilated radius when a reach law is given.

    Parameters
    ----------
    length_km : np.ndarray, shape (log10_e.size,)
        Muon range to threshold at the muon energy ``(1 - <y>) E_nu`` of each
        grid point [km].
    threshold_gev : float
        Muon selection threshold [GeV].
    reach_km : float or None, optional
        Growth of the light reach per e-fold [km]; ``None`` keeps the static
        footprint.
    pivot_gev : float, optional
        Energy at which the reach vanishes [GeV]. Ignored when ``reach_km``
        is ``None``.
    log10_e : np.ndarray, optional
        ``log10(E_nu / GeV)`` grid. Defaults to :data:`IC_LOG10_E`.
    cross_section : CrossSection or None, optional
        Neutrino cross section; ``None`` uses :func:`default_cross_section`.
    n_dec : int, optional
        Declination slices of :func:`ic_upgoing_columns`.
    radius_km, height_km : float, optional
        Instrumented footprint radius and height [km]. Default to IceCube.
    n_sides : int or None, optional
        Sides of the prism cross-section. Defaults to the IceCube hexagon.
    mask_subthreshold : bool, optional
        Whether a rung whose muon is born below threshold also loses the
        instrumented volume, not only the column. ``True`` (the default) is
        the consistent choice: the body counts starting events, and a
        starting event whose muon cannot be selected is not one. ``False``
        reproduces the earlier construction of examples 28 and 31.

    Returns
    -------
    aeff : np.ndarray, shape (log10_e.size,)
        Effective area [cm^2].
    """
    xsec = _cross_section(cross_section)
    energy = 10.0**log10_e
    columns, weights, cos_theta = ic_upgoing_columns(n_dec)
    n_nucleon = nucleon_number_density()
    out = np.empty(energy.size)
    for i, e_nu in enumerate(energy):
        rung_energy, rung_weight = regenerated_transmission(float(e_nu), columns, xsec)
        rung_length = np.interp(
            np.log10(rung_energy), log10_e, length_km, left=0.0, right=length_km[-1]
        )
        below = (1.0 - MEAN_INELASTICITY) * rung_energy <= threshold_gev
        rung_length[below] = 0.0
        selectable = (~below)[:, None] if mask_subthreshold else 1.0
        rung_radius = (
            radius_km
            if reach_km is None
            else light_reach_radius_km(
                radius_km, (1.0 - MEAN_INELASTICITY) * rung_energy, reach_km, pivot_gev
            )
        )
        rung_rate = (
            n_nucleon * xsec.cc(rung_energy)[:, None]
            * ic_target_volume_cm3(rung_length, rung_radius, cos_theta, height_km, n_sides)
            * selectable
        )
        out[i] = np.average((rung_weight * rung_rate).sum(axis=0), weights=weights)
    return out


def ic_effective_area_tau_channel(
    length_km: np.ndarray,
    threshold_gev: float,
    log10_e: np.ndarray = IC_LOG10_E,
    cross_section: CrossSection | None = None,
    n_dec: int = N_DEC,
    radius_km: float = ICECUBE.radius_km,
    height_km: float = ICECUBE.height_km,
    n_sides: int | None = ICECUBE.n_sides,
    mask_subthreshold: bool = True,
) -> np.ndarray:
    """``nu_tau -> tau -> mu`` channel, upgoing-averaged, static footprint [cm^2].

    The chain of Sec. IV: it costs the branching ratio ``B_{tau->mu}`` and a
    factor ``<z>`` in muon energy, and gains the charged-current-regenerating
    Earth transmission of
    :func:`~softpaws.transport.attenuation.flavour_transmission`.

    Parameters
    ----------
    length_km : np.ndarray, shape (log10_e.size,)
        Muon range to threshold at the muon energy ``(1 - <y>) E_nu`` of each
        grid point [km]; the tau channel reads it at the energy whose direct
        muon matches the chain's.
    threshold_gev : float
        Muon selection threshold [GeV].
    log10_e : np.ndarray, optional
        ``log10(E_nu / GeV)`` grid. Defaults to :data:`IC_LOG10_E`.
    cross_section : CrossSection or None, optional
        Neutrino cross section; ``None`` uses :func:`default_cross_section`.
    n_dec : int, optional
        Declination slices of :func:`ic_upgoing_columns`.
    radius_km, height_km : float, optional
        Instrumented footprint radius and height [km]. Default to IceCube.
    n_sides : int or None, optional
        Sides of the prism cross-section. Defaults to the IceCube hexagon.
    mask_subthreshold : bool, optional
        See :func:`ic_effective_area_regenerated`.

    Returns
    -------
    aeff : np.ndarray, shape (log10_e.size,)
        Effective area [cm^2].
    """
    xsec = _cross_section(cross_section)
    energy = 10.0**log10_e
    columns, weights, cos_theta = ic_upgoing_columns(n_dec)
    n_nucleon = nucleon_number_density()
    muon_fraction = MEAN_Z * (1.0 - MEAN_INELASTICITY)
    out = np.empty(energy.size)
    for i, e_nu in enumerate(energy):
        rung_energy, rung_weight = flavour_transmission(float(e_nu), columns, xsec, flavour="tau")
        rung_length = np.interp(
            np.log10(rung_energy * muon_fraction / (1.0 - MEAN_INELASTICITY)),
            log10_e, length_km, left=0.0, right=length_km[-1],
        )
        below = muon_fraction * rung_energy <= threshold_gev
        rung_length[below] = 0.0
        selectable = (~below)[:, None] if mask_subthreshold else 1.0
        rung_rate = (
            n_nucleon * xsec.cc(rung_energy)[:, None]
            * ic_target_volume_cm3(rung_length, radius_km, cos_theta, height_km, n_sides)
            * BR_TAU_TO_MU * selectable
        )
        out[i] = np.average((rung_weight * rung_rate).sum(axis=0), weights=weights)
    return out


# ---------------------------------------------------------------------------
# Water sites: full sky on a cos(theta) grid, finite overburden
# ---------------------------------------------------------------------------


def cylinder_projected_area_km2(
    theta_deg: np.ndarray,
    radius_km: float | np.ndarray,
    n_blocks: int,
    height_km: float | np.ndarray = ARCA230.height_km,
) -> np.ndarray:
    """Projected area of upright cylindrical building blocks [km^2].

    A cylinder of radius ``R`` and height ``h`` presents ``pi R^2`` overhead
    and ``2 R h`` at the horizon; the convex-body projection interpolates
    between them as ``pi R^2 |cos theta| + 2 R h sin theta``. The blocks'
    projections add, which ignores the mutual shadowing of two adjacent blocks
    near the horizon.

    Parameters
    ----------
    theta_deg : np.ndarray
        Zenith angle [deg].
    radius_km : float or np.ndarray
        Footprint radius of one block [km], broadcast against ``theta_deg``,
        so an energy-dependent radius can be passed as a column.
    n_blocks : int
        Number of building blocks.
    height_km : float or np.ndarray, optional
        Instrumented height of one block [km]. Defaults to the ARCA block.
        A larger one is how a light reach that extends the boundary
        vertically as well as radially enters.

    Returns
    -------
    area : np.ndarray
        Projected area [km^2], broadcast to the shape of the inputs.
    """
    theta = np.deg2rad(theta_deg)
    cap = np.pi * np.asarray(radius_km) ** 2 * np.abs(np.cos(theta))
    side = 2.0 * np.asarray(radius_km) * np.asarray(height_km) * np.sin(theta)
    return n_blocks * (cap + side)


def arca_effective_area(
    radius_km: float,
    n_blocks: int,
    threshold_gev: float,
    depth_km: float,
    flavour: str,
    kernel_evaluation: str = "running",
    reach_km: float | None = None,
    pivot_gev: float = 1.0e6,
    log10_e: np.ndarray = ARCA_LOG10_E,
    height_km: float = ARCA230.height_km,
    cos_range: tuple[float, float] = (-1.0, 1.0),
    n_zenith: int = N_ZENITH,
    truncate: bool = True,
    mask_subthreshold: bool = True,
    density_g_cm3: float = RHO_WATER_G_CM3,
    max_upstream_km: float = MAX_UPSTREAM_KM,
    cross_section: CrossSection | None = None,
    truncated_range: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Solid-angle-averaged effective area of a cylinder array [cm^2].

    Assembles Sec. VI of the paper with the two changes a water site needs:
    the length is the first-passage range truncated at the available column,
    and the projected area follows the cylinder. The neutral-current-degraded
    population is kept on an energy ladder and each rung is credited to the
    surface energy, matching how a published effective area is built.

    Parameters
    ----------
    radius_km : float
        Footprint radius of one building block [km].
    n_blocks : int
        Number of building blocks.
    threshold_gev : float
        Muon selection threshold [GeV].
    depth_km : float
        Depth of the instrumented centre below the surface of the medium [km].
    flavour : {"mu", "tau"}
        ``"mu"`` is the direct charged-current muon. ``"tau"`` is the
        ``nu_tau -> tau -> mu`` chain, which costs ``B_{tau->mu}`` and a factor
        ``<z>`` in muon energy but gains the regenerating Earth transmission.
    kernel_evaluation : {"running", "frozen"}, optional
        Where along the descent the loss kernel is read; see
        :func:`truncated_range_km`.
    reach_km : float or None, optional
        Growth of the light reach per e-fold [km]. ``None`` keeps the static
        footprint; a value dilates both the projected area and the body, since
        both describe what the detector responds to.
    pivot_gev : float, optional
        Energy at which the reach vanishes [GeV].
    log10_e : np.ndarray, optional
        ``log10(E_nu / GeV)`` grid. Defaults to :data:`ARCA_LOG10_E`.
    height_km : float, optional
        Instrumented height of one block [km]. Defaults to the ARCA block.
    cos_range : tuple of float, optional
        Band of ``cos(theta)`` to average over. Defaults to the full sky.
    n_zenith : int, optional
        Slices of :func:`~softpaws.transport.earth.zenith_grid`.
    truncate : bool, optional
        Whether to cut the first-passage integral at the available column.
        ``False`` reproduces Eq. (16) as written, which sizes the overburden
        effect.
    mask_subthreshold : bool, optional
        Whether a sub-threshold rung loses the instrumented volume along with
        the column; see :func:`ic_effective_area_regenerated`. ``False``
        reproduces examples 30 and 31.
    density_g_cm3 : float, optional
        Density of the medium [g cm^-3]. Defaults to sea water.
    max_upstream_km : float, optional
        Cap on the near-horizontal path; see
        :data:`~softpaws.transport.earth.MAX_UPSTREAM_KM`.
    cross_section : CrossSection or None, optional
        Neutrino cross section; ``None`` uses :func:`default_cross_section`.
    truncated_range : callable or None, optional
        Replacement for :func:`truncated_range_km`, called as
        ``truncated_range(energy_mu_gev, column_km)`` and returning the
        ``(n_energy, n_column)`` length table. Example 31 passes
        :func:`truncated_range_from_table_km` through this.

    Returns
    -------
    aeff : np.ndarray, shape (log10_e.size,)
        Effective area [cm^2], averaged over the requested band.
    """
    xsec = _cross_section(cross_section)
    theta_deg, weights = zenith_grid(n_zenith, cos_range)
    cos_theta = np.cos(np.deg2rad(theta_deg))
    columns = neutrino_column_g_cm2(cos_theta, depth_km, density_g_cm3, max_upstream_km)
    available_km = overburden_km(cos_theta, depth_km, max_upstream_km)
    if not truncate:
        available_km = np.full_like(available_km, np.inf)
    static_area_km2 = cylinder_projected_area_km2(theta_deg, radius_km, n_blocks, height_km)
    static_v_det_km3 = n_blocks * np.pi * radius_km**2 * height_km
    n_nucleon = nucleon_number_density(density_g_cm3)

    muon_fraction = 1.0 - MEAN_INELASTICITY
    if flavour == "tau":
        muon_fraction *= MEAN_Z

    out = np.empty(log10_e.size)
    for i, e_nu in enumerate(10.0**log10_e):
        if flavour == "tau":
            rung_energy, rung_weight = flavour_transmission(
                float(e_nu), columns, xsec, flavour="tau"
            )
        else:
            rung_energy, rung_weight = regenerated_transmission(float(e_nu), columns, xsec)
        # (n_rung, n_theta) length: each rung's muon, each direction's column.
        if truncated_range is None:
            length = truncated_range_km(
                rung_energy * muon_fraction, available_km, threshold_gev,
                kernel_evaluation, density_g_cm3,
            )
        else:
            length = truncated_range(rung_energy * muon_fraction, available_km)
        below = rung_energy * muon_fraction <= threshold_gev
        length[below, :] = 0.0

        if reach_km is None:
            area_km2 = static_area_km2[None, :]
            v_det_km3 = static_v_det_km3
        else:
            # The light reach follows the muon, so each rung gets its own radius.
            r_eff = light_reach_radius_km(
                radius_km, rung_energy * muon_fraction, reach_km, pivot_gev
            )[:, None]
            area_km2 = cylinder_projected_area_km2(theta_deg[None, :], r_eff, n_blocks, height_km)
            v_det_km3 = n_blocks * np.pi * r_eff**2 * height_km

        volume_km3 = area_km2 * length + v_det_km3
        if mask_subthreshold:
            volume_km3 = volume_km3 * (~below)[:, None]
        rate = n_nucleon * xsec.cc(rung_energy)[:, None] * volume_km3 * CM_PER_KM**3
        if flavour == "tau":
            rate = rate * BR_TAU_TO_MU
        out[i] = np.average((rung_weight * rate).sum(axis=0), weights=weights)
    return out


def required_footprint_radius_km(
    ratio: np.ndarray,
    radius_km: float,
    n_blocks: int,
    height_km: float = ARCA230.height_km,
    n_zenith: int = N_ZENITH,
) -> np.ndarray:
    """Footprint radius that would scale the sky-averaged projected area by ``ratio``.

    Read when the implied efficiency ``A_eff^published / A_eff^model`` comes
    out above one, which a geometric ceiling forbids. The deficit then has to
    sit in the geometry, and the natural place is the projected area: a bright
    muon triggers from outside the instrumented footprint. This inverts the
    sky-averaged cylinder projection for the radius that closes the gap,
    holding the height fixed.

    Parameters
    ----------
    ratio : np.ndarray
        Required scaling of the sky-averaged projected area.
    radius_km : float
        Nominal footprint radius of one building block [km].
    n_blocks : int
        Number of building blocks.
    height_km : float, optional
        Instrumented height of one block [km]. Defaults to the ARCA block.
    n_zenith : int, optional
        Slices of :func:`~softpaws.transport.earth.zenith_grid`.

    Returns
    -------
    radius : np.ndarray
        Required footprint radius [km], ``NaN`` where ``ratio`` is not finite
        or not positive.
    """
    theta_deg, weights = zenith_grid(n_zenith)

    def mean_area(r: float) -> float:
        return float(np.average(
            cylinder_projected_area_km2(theta_deg, r, n_blocks, height_km), weights=weights
        ))

    base = mean_area(radius_km)
    out = np.full(np.shape(ratio), np.nan)
    for i, r in enumerate(np.atleast_1d(ratio)):
        if not np.isfinite(r) or r <= 0.0:
            continue
        out[i] = brentq(lambda x: mean_area(x) - r * base, 1.0e-3, 50.0)
    return out


def effective_volume_km3(
    aeff_cm2: np.ndarray,
    log10_e: np.ndarray = ARCA_LOG10_E,
    density_g_cm3: float = RHO_WATER_G_CM3,
    cross_section: CrossSection | None = None,
) -> np.ndarray:
    """Effective target volume implied by an effective area [km^3].

    Inverting ``A_eff = n_N sigma_CC V_eff`` gives the transmission-weighted
    volume the detector behaves as, which is the quantity Eqs. (10) and (15)
    of the paper predict. Quoted at the medium's nucleon density, so it is a
    volume of that medium and not a column.

    Parameters
    ----------
    aeff_cm2 : np.ndarray
        Effective area [cm^2] on ``log10_e``.
    log10_e : np.ndarray, optional
        ``log10(E_nu / GeV)`` grid. Defaults to :data:`ARCA_LOG10_E`.
    density_g_cm3 : float, optional
        Density of the medium [g cm^-3]. Defaults to sea water.
    cross_section : CrossSection or None, optional
        Neutrino cross section; ``None`` uses :func:`default_cross_section`.

    Returns
    -------
    volume : np.ndarray
        Effective volume [km^3].
    """
    energy = 10.0**log10_e
    denominator = nucleon_number_density(density_g_cm3) * _cross_section(cross_section).cc(energy)
    return aeff_cm2 / denominator / CM_PER_KM**3
