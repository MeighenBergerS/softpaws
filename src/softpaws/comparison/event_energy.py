"""The neutrino energy behind one observed muon track.

A track detector measures a muon energy somewhere along the muon's descent
from the neutrino interaction, an unknown distance upstream. The arrival
kernel of the transport, integrated over that distance, gives the potential
density ``u(w)``: how much column a muon spends at each log loss ``w`` on its
way down. Folded with the measurement it gives the event's energy
likelihood, and with a flux prior, the cross section and the survival
through the matter in front of the detector, the neutrino-energy posterior.

The functions here are those of the KM3-230213A analysis (Section V B of the
paper) and of the single-track reconstructions that precede it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from ..transport.coefficients import (
    diffusion_coefficient,
    drift_coefficient,
    third_moment_coefficient,
)
from ..transport.cross_section import CrossSection
from ..transport.loss_distribution import (
    loss_density,
    loss_density_gaussian,
    loss_density_three_moment,
)
from ..transport.source import mean_inelasticity
from ..utils.constants import AVOGADRO_PER_MOL, CM_PER_KM, EARTH_RADIUS_KM, RHO_WATER_G_CM3

__all__ = [
    "KM3_230213A",
    "TrackEvent",
    "energy_likelihood",
    "energy_posterior",
    "interaction_unity_energy_gev",
    "lognormal_measurement",
    "parent_energy_posterior",
    "posterior_summary",
    "potential_density",
    "quantile",
    "sea_path_km",
    "survival_through_column",
    "two_layer_column_g_cm2",
]


@dataclass(frozen=True)
class TrackEvent:
    """One observed muon track and the matter in front of the detector.

    Attributes
    ----------
    name : str
        Event name.
    muon_energy_gev : float
        Central estimate of the muon energy at the detector [GeV].
    muon_energy_90_gev : tuple of float
        Lower and upper edge of the 90% interval on the muon energy [GeV].
    elevation_deg : float
        Elevation of the arrival direction above the local horizon [deg];
        negative is below.
    depth_km : float
        Detector depth below the surface of the medium [km].
    traversed_column_kmwe : float
        Column the neutrino traverses to the detector [km water equivalent].
    water_before_detector_km : float
        Water in front of the detector along the arrival direction [km]; the
        muon is born inside it for any parent energy of interest.
    reference : str
        Where the numbers come from.
    """

    name: str
    muon_energy_gev: float
    muon_energy_90_gev: tuple[float, float]
    elevation_deg: float
    depth_km: float
    traversed_column_kmwe: float
    water_before_detector_km: float
    reference: str = ""

    @property
    def traversed_column_g_cm2(self) -> float:
        """The traversed column in ``g cm^-2``."""
        return self.traversed_column_kmwe * CM_PER_KM * RHO_WATER_G_CM3


#: KM3-230213A (KM3NeT Collaboration, Nature 638 (2025) 376): a 120 PeV muon
#: with a 90% interval of 35 to 380 PeV, arriving 0.6 deg above the horizon
#: at a 3.4 km deep detector. The column along the nominal direction from the
#: release's topography notebook (Zenodo 10.5281/zenodo.14860165) is 4 km of
#: water, 104 km of rock, then 34 km of water in front of the detector: 142
#: km, 309 km water equivalent.
KM3_230213A = TrackEvent(
    name="KM3-230213A",
    muon_energy_gev=1.2e8,
    muon_energy_90_gev=(3.5e7, 3.8e8),
    elevation_deg=0.6,
    depth_km=3.4,
    traversed_column_kmwe=309.0,
    water_before_detector_km=34.0,
    reference="Nature 638 (2025) 376; Zenodo 10.5281/zenodo.14860165",
)


def sea_path_km(
    elevation_deg: float, depth_km: float, earth_radius_km: float = EARTH_RADIUS_KM
) -> float:
    """Path from a detector to the surface of its medium at a given elevation [km].

    Parameters
    ----------
    elevation_deg : float
        Elevation of the direction above the local horizon [deg].
    depth_km : float
        Detector depth below the surface [km].
    earth_radius_km : float, optional
        Radius of the surface [km].

    Returns
    -------
    path_km : float
        Straight-line path to the surface [km].
    """
    r0 = earth_radius_km - depth_km
    s = np.sin(np.deg2rad(elevation_deg))
    return float(-r0 * s + np.sqrt((r0 * s) ** 2 + earth_radius_km**2 - r0**2))


def two_layer_column_g_cm2(
    elevation_deg: float,
    depth_km: float,
    seabed_depth_km: float,
    rho_sea_g_cm3: float = 1.03,
    rho_rock_g_cm3: float = 2.65,
    ds_km: float = 0.05,
    earth_radius_km: float = EARTH_RADIUS_KM,
) -> tuple[float, float, float, float]:
    """Matter column from a detector to the surface through a two-layer Earth.

    Rock below the seabed, sea above it, on a curved Earth, with the ray
    leaving the detector at ``elevation_deg`` above the local horizontal.

    Parameters
    ----------
    elevation_deg : float
        Elevation of the direction above the local horizon [deg]; negative
        is below.
    depth_km : float
        Detector depth below the sea surface [km].
    seabed_depth_km : float
        Seabed depth below the sea surface [km].
    rho_sea_g_cm3, rho_rock_g_cm3 : float, optional
        Densities of the two layers [g cm^-3].
    ds_km : float, optional
        Step of the path integration [km].
    earth_radius_km : float, optional
        Radius of the sea surface [km].

    Returns
    -------
    total_g_cm2 : float
        Total column [g cm^-2].
    water_km, rock_km : float
        Path lengths in each layer [km].
    path_km : float
        Total path length [km].
    """
    r0 = earth_radius_km - depth_km
    r_bed = earth_radius_km - seabed_depth_km
    sin_a = np.sin(np.deg2rad(elevation_deg))
    water = rock = 0.0
    s = 0.0
    while True:
        s += ds_km
        r = np.sqrt(r0**2 + s**2 + 2.0 * r0 * s * sin_a)
        if r >= earth_radius_km:
            break
        if r > r_bed:
            water += ds_km
        else:
            rock += ds_km
        if s > 3.0e3:  # not a chord this function is meant for
            break
    total = (water * rho_sea_g_cm3 + rock * rho_rock_g_cm3) * CM_PER_KM
    return total, water, rock, s


def potential_density(
    kind: str,
    w_grid: np.ndarray,
    kernel_energy_gev: float = 1.0e8,
    x_max_km: float = 30.0,
    n_x: int = 120,
    n_k: int = 2**13,
    n_moments: int = 3,
) -> np.ndarray:
    """Potential density ``u(w) = int_0^inf P(w | X) dX`` on a log-loss grid.

    The column a muon spends at each log loss ``w`` on its way down, per unit
    ``w`` [km]. The kernel is frozen at ``kernel_energy_gev``; it is scale
    invariant, and the drift moves by about 4% per decade there.

    Parameters
    ----------
    kind : {"exact", "gaussian", "csda"}
        ``"exact"`` is the subordinator (the loss law itself), ``"gaussian"``
        the Fokker-Planck form, ``"csda"`` the mean loss, which gives
        ``1 / b_mu`` exactly.
    w_grid : np.ndarray
        Log-loss grid [dimensionless]. It must hold the whole loss law of the
        deepest slab, mean ``b x_max`` plus its spread, because the loss
        density renormalizes on the grid.
    kernel_energy_gev : float, optional
        Energy the loss coefficients are read at [GeV].
    x_max_km : float, optional
        Deepest slab integrated [km].
    n_x : int, optional
        Number of slabs.
    n_k : int, optional
        Fourier nodes of the loss density.
    n_moments : {2, 3}, optional
        Loss family behind ``"exact"``: 3 is the three-moment family whose
        hard edge matches PROPOSAL's tail; 2 the two-moment digamma family,
        8% low on the first log-loss moment.

    Returns
    -------
    u : np.ndarray
        Potential density on ``w_grid`` [km per unit ``w``].

    Raises
    ------
    ValueError
        Raised if ``kind`` is not one of the three names.
    """
    if kind not in ("exact", "gaussian", "csda"):
        raise ValueError(f"kind must be 'exact', 'gaussian' or 'csda', got {kind!r}.")
    b = float(np.squeeze(drift_coefficient(kernel_energy_gev)))
    d = float(np.squeeze(diffusion_coefficient(kernel_energy_gev)))
    t = float(np.squeeze(third_moment_coefficient(kernel_energy_gev)))
    w_grid = np.asarray(w_grid, dtype=float)
    if kind == "csda":
        return np.full(w_grid.size, 1.0 / b)
    x_grid = np.linspace(0.0, x_max_km, n_x + 1)[1:]
    u = np.zeros(w_grid.size)
    for x in x_grid:
        if kind == "exact" and n_moments == 3:
            u += loss_density_three_moment(w_grid, float(x), b, d, t, n_k=n_k)
        elif kind == "exact":
            u += loss_density(w_grid, float(x), b, d, n_k=n_k)
        else:
            u += loss_density_gaussian(w_grid, float(x), b, d)
    # The first slab, X in (0, dX): the loss is small and the density sharply
    # peaked, so it is taken as its trapezoid end point rather than resolved.
    return u * (x_grid[1] - x_grid[0])


def lognormal_measurement(
    log10_e: np.ndarray, central_gev: float, interval_90_gev: tuple[float, float]
) -> np.ndarray:
    """Lognormal likelihood of a measured energy, as a density in ``ln E``.

    The width is set by the 90% interval, taken symmetric in the logarithm.

    Parameters
    ----------
    log10_e : np.ndarray
        Trial energies [log10 GeV].
    central_gev : float
        Central estimate [GeV].
    interval_90_gev : tuple of float
        Lower and upper edge of the 90% interval [GeV].

    Returns
    -------
    density : np.ndarray
        Likelihood density in ``ln E`` at each trial energy.
    """
    sigma = np.log(interval_90_gev[1] / interval_90_gev[0]) / (2.0 * norm.isf(0.05))
    return norm.pdf(np.asarray(log10_e) * np.log(10.0), loc=np.log(central_gev), scale=sigma)


def energy_likelihood(
    u: np.ndarray,
    w_grid: np.ndarray,
    energy_nu: float | np.ndarray,
    measurement: Callable[[np.ndarray], np.ndarray],
    accept_gev: float | None = None,
) -> np.ndarray:
    """The event's energy likelihood ``l(E_nu) = int dw u(w) L(eps e^-w)``.

    The integral runs over the log loss on the kernel's own grid, so the
    integrable spike of the exact ``u`` at ``w -> 0`` is sampled identically
    for every ``E_nu``.

    Parameters
    ----------
    u : np.ndarray
        Potential density on ``w_grid``, from :func:`potential_density`.
    w_grid : np.ndarray
        Log-loss grid.
    energy_nu : float or np.ndarray
        Neutrino energies [GeV].
    measurement : callable
        Likelihood of the muon measurement as a function of ``log10 E_mu``,
        a density in ``ln E``; see :func:`lognormal_measurement`.
    accept_gev : float or None, optional
        Muon energy below which the selection does not count a muon [GeV].
        When given, the result is divided by the muons the selection
        accepts, ``int u dw`` over ``E_mu >= accept_gev``, which is the
        conditional density a rate comparison needs alongside the effective
        area. ``None`` returns the unnormalized fold, which is what the
        energy posterior takes alongside the cross section.

    Returns
    -------
    likelihood : np.ndarray
        The likelihood at each ``energy_nu``.
    """
    energy_nu = np.atleast_1d(np.asarray(energy_nu, dtype=float))
    w_grid = np.asarray(w_grid, dtype=float)
    eps = (1.0 - np.squeeze(mean_inelasticity(energy_nu))) * energy_nu
    log10_e_mu = (np.log(eps)[:, None] - w_grid[None, :]) / np.log(10.0)
    value = np.trapezoid(u[None, :] * measurement(log10_e_mu), w_grid, axis=1)
    if accept_gev is None:
        return value
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (u[1:] + u[:-1]) * np.diff(w_grid))])
    w_accept = np.log(eps / accept_gev)
    total = np.where(
        w_accept <= w_grid[-1],
        np.interp(w_accept, w_grid, cdf),
        cdf[-1] + (w_accept - w_grid[-1]) * u[-1],
    )
    return value / np.clip(total, 1.0e-300, None)


def survival_through_column(
    energy_nu: float | np.ndarray, column_g_cm2: float, cross_section: CrossSection
) -> np.ndarray:
    """Neutrino survival through a column of isoscalar matter.

    Parameters
    ----------
    energy_nu : float or np.ndarray
        Neutrino energies [GeV].
    column_g_cm2 : float
        Traversed column [g cm^-2].
    cross_section : CrossSection
        Cross-section model; charged and neutral current both remove the
        neutrino from the beam.

    Returns
    -------
    survival : np.ndarray
        Survival probability in ``[0, 1]``.
    """
    sigma = cross_section.cc(energy_nu) + cross_section.nc(energy_nu)
    return np.exp(-AVOGADRO_PER_MOL * sigma * column_g_cm2)


def interaction_unity_energy_gev(column_g_cm2: float, cross_section: CrossSection) -> float:
    """Energy at which a column is one interaction length.

    Parameters
    ----------
    column_g_cm2 : float
        Traversed column [g cm^-2].
    cross_section : CrossSection
        Cross-section model.

    Returns
    -------
    energy_gev : float
        The energy [GeV], or infinity if the column is thinner than one
        interaction length up to ``10^12`` GeV.
    """
    target = 1.0 / (AVOGADRO_PER_MOL * column_g_cm2)
    grid = np.logspace(6.0, 12.0, 400)
    sigma = cross_section.cc(grid) + cross_section.nc(grid)
    if sigma[-1] < target:
        return np.inf
    return float(np.exp(np.interp(np.log(target), np.log(sigma), np.log(grid))))


def energy_posterior(
    likelihood: np.ndarray,
    log10_enu: np.ndarray,
    flux_shape: Callable[[np.ndarray], np.ndarray],
    cross_section: CrossSection,
    survival: np.ndarray | None = None,
) -> np.ndarray:
    """Normalized neutrino-energy posterior on a ``log10 E`` grid.

    The weight is the flux shape times the charged-current cross section
    times the survival, and the posterior is a density in ``log10 E``.

    Parameters
    ----------
    likelihood : np.ndarray
        Energy likelihood on ``10**log10_enu``, from :func:`energy_likelihood`
        without ``accept_gev``.
    log10_enu : np.ndarray
        Neutrino-energy grid [log10 GeV].
    flux_shape : callable
        Flux prior as a function of energy [GeV], up to normalization.
    cross_section : CrossSection
        Cross-section model.
    survival : np.ndarray or None, optional
        Survival factor on the grid; ``None`` for no attenuation.

    Returns
    -------
    posterior : np.ndarray
        Density in ``log10 E``, unit area on the grid.
    """
    energy = 10.0 ** np.asarray(log10_enu, dtype=float)
    weight = flux_shape(energy) * cross_section.cc(energy)
    if survival is not None:
        weight = weight * survival
    p = likelihood * weight * energy
    return p / np.trapezoid(p, log10_enu)


def posterior_summary(
    posterior: np.ndarray, log10_enu: np.ndarray, level: float = 0.90
) -> tuple[float, float, float, float]:
    """Mode, median and central interval of a posterior on a ``log10 E`` grid.

    Parameters
    ----------
    posterior : np.ndarray
        Density in ``log10 E``.
    log10_enu : np.ndarray
        The grid [log10 GeV].
    level : float, optional
        Probability content of the interval.

    Returns
    -------
    mode, median, low, high : float
        All in GeV.
    """
    log10_enu = np.asarray(log10_enu, dtype=float)
    cdf = np.concatenate(
        [[0.0], np.cumsum(0.5 * (posterior[1:] + posterior[:-1]) * np.diff(log10_enu))]
    )
    tail = 0.5 * (1.0 - level)
    lo, med, hi = np.interp([tail, 0.5, 1.0 - tail], cdf, log10_enu)
    mode = log10_enu[int(np.argmax(posterior))]
    return tuple(10.0**v for v in (mode, med, lo, hi))


def quantile(w: np.ndarray, density: np.ndarray, q: float) -> float:
    """The ``q``-quantile of a density tabulated on an ascending grid.

    Parameters
    ----------
    w : np.ndarray
        Grid.
    density : np.ndarray
        Density on the grid; it need not be normalized.
    q : float
        Probability in ``[0, 1]``.

    Returns
    -------
    value : float
        The quantile.
    """
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(w))])
    cdf /= cdf[-1]
    return float(np.interp(q, cdf, w))


def parent_energy_posterior(
    muon_energy_gev: float, ell_km: float, gamma: float, n_w: int = 2400
) -> dict[str, np.ndarray]:
    """Parent-energy posterior of one muon seen after a known column.

    For a muon of energy ``E`` measured after ``ell_km`` of medium, the parent
    energy is ``eps = E e^w`` with ``w`` the log loss, and a power-law flux
    prior weights it by ``e^{-gamma w}``. Both the exact loss law and its
    Fokker-Planck form are returned, each normalized to unit area in ``w``.

    Parameters
    ----------
    muon_energy_gev : float
        Measured muon energy [GeV].
    ell_km : float
        Column the muon crossed [km].
    gamma : float
        Spectral index of the flux prior.
    n_w : int, optional
        Number of grid points in ``w``.

    Returns
    -------
    result : dict
        ``"w"`` the log-loss grid, ``"eps"`` the parent energies [GeV],
        ``"exact"`` and ``"fp"`` the two posteriors, and ``"b_mu"``,
        ``"d_mu"`` the coefficients used [km^-1].
    """
    b_mu = float(np.squeeze(drift_coefficient(muon_energy_gev)))
    d_mu = float(np.squeeze(diffusion_coefficient(muon_energy_gev)))
    mean = (b_mu + d_mu / 2.0) * ell_km
    w = np.linspace(1e-4, mean + 9.0 * np.sqrt(d_mu * ell_km), n_w)
    prior = np.exp(-gamma * w)
    post_exact = prior * loss_density(w, ell_km, b_mu, d_mu)
    post_fp = prior * loss_density_gaussian(w, ell_km, b_mu, d_mu)
    post_exact /= np.trapezoid(post_exact, w)
    post_fp /= np.trapezoid(post_fp, w)
    return {
        "w": w,
        "eps": muon_energy_gev * np.exp(w),
        "exact": post_exact,
        "fp": post_fp,
        "b_mu": b_mu,
        "d_mu": d_mu,
    }
