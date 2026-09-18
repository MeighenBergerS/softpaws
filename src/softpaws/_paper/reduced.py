"""The reduced response: two instrument numbers per detector.

The five-parameter fit of :mod:`softpaws.response.site_models` floats three
numbers that are not the instrument's to set. ``b_scale`` returns its prior
at every site, ``lam`` is the cross-section slope and one number in nature,
and ``eps_0`` is a selection efficiency the collaborations quote. This
module fixes all three and refits each published table with only the
threshold ``log10_e_thr`` and the light reach ``reach_km`` free, so the
question becomes whether two instrument numbers describe a table as well as
five did.

A three-parameter variant leaves ``eps_0`` free with the physics still
fixed, which is what the TRIDENT 2025 effective-area map needs: the map is
published after trigger and quality cuts, so its normalization is not one
of the levels :data:`EPS_FIXED` records.

Notes
-----
Nothing here plots or prints. The forward models, the parameter names and
the published curves come from :mod:`softpaws.response.site_models`, and
the sampler from :mod:`softpaws.comparison.posterior`. Both the sampler and
the map loader are imported inside the functions that use them, because the
package layering runs data -> response -> comparison and a module-level
import the other way would close a cycle.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Callable

import numpy as np

from softpaws.detectors import TRIDENT_2025

from .site_fit import (
    ARCA_LOG10_E,
    LAMBDA_BGR18,
    PARAM_NAMES,
    PRIORS,
    SiteFit,
    WaterSite,
    arca_zenith_grid,
    water_columns,
    water_ladders,
    water_model,
    water_sites,
)

__all__ = [
    "COS_EDGES",
    "EPS_FIXED",
    "FREE2",
    "FREE3",
    "MAP_RADIUS_KM",
    "PARAM_BOUNDS",
    "PREDICTED_REACH_M",
    "REACH_DERIVED_KM",
    "REDUCED_SITES",
    "WALKER_SCATTER",
    "MapModel",
    "attach_reduced_chains",
    "deviance",
    "fit",
    "fit_cells",
    "full_theta",
    "load_two_parameter_chains",
    "reach_separation_sigma",
    "reduced_chain_summary",
    "reduced_log_probability",
    "trident_2025_average_detector",
    "trident_2025_cells",
    "trident_2025_detector",
]

#: The four sites the reduced fit runs on, in figure order.
REDUCED_SITES = ("IceCube", "ARCA230", "P-ONE", "TRIDENT")

#: Fixed selection efficiency per site. IceCube carries the through-going
#: analysis-level efficiency the collaboration quotes; the water tables are at
#: trigger or proposal level, with nothing to lose by construction.
EPS_FIXED = {"IceCube": 0.956, "ARCA230": 1.0, "P-ONE": 1.0, "TRIDENT": 1.0}

#: Reach each medium's bulk optics imply [km per e-fold]: the optical
#: attenuation lengths of 59 m in ice and 68 m in water.
REACH_DERIVED_KM = {"IceCube": 0.059, "ARCA230": 0.068, "P-ONE": 0.068, "TRIDENT": 0.068}

#: Reach each site's own measured optics predict [m], as ``(low, high)``: the
#: effective attenuation length at 400 nm, the shorter of the absorption length
#: and the diffusive length ``sqrt(abs * scat / 3)``. IceCube: absorption 110 to
#: 200 m with effective scattering 25 to 50 m, from the dust-layer average to
#: the clearest ice. ARCA230: Capo Passero absorption ~50 m at 470 nm (ANTARES
#: site 60 m), scaled to 400 nm. P-ONE: STRAW 28 m at 450 nm (35 m in the
#: proposal), scaled. TRIDENT: measured effective attenuation 15 to 27 m over
#: 405 to 460 nm.
PREDICTED_REACH_M = {
    "IceCube": (30.0, 59.0),
    "ARCA230": (41.0, 50.0),
    "P-ONE": (24.0, 30.0),
    "TRIDENT": (15.0, 27.0),
}

#: The two free parameters of the reduced fit.
FREE2 = ("log10_e_thr", "reach_km")

#: The three free parameters of the variant that also floats ``eps_0``.
FREE3 = ("eps_0", "log10_e_thr", "reach_km")

#: Initial walker scatter per free parameter. ``reach_km`` lives two orders of
#: magnitude below the others, so a common value would throw walkers out of it.
WALKER_SCATTER = {"eps_0": 0.01, "log10_e_thr": 0.02, "reach_km": 0.002}


# ---------------------------------------------------------------------------
# The reduced fit
# ---------------------------------------------------------------------------


def full_theta(
    free_names: tuple[str, ...],
    free_values: np.ndarray,
    fixed: dict[str, float],
) -> np.ndarray:
    """Assemble the full five-parameter vector from the free ones.

    Parameters
    ----------
    free_names : tuple of str
        Names of the free parameters, a subset of :data:`PARAM_NAMES`.
    free_values : np.ndarray
        Their values, in the same order.
    fixed : dict
        Value of every one of :data:`PARAM_NAMES`; the free ones are
        overwritten.

    Returns
    -------
    theta : np.ndarray, shape (5,)
        ``(eps_0, log10_e_thr, b_scale, lam, reach_km)``.
    """
    theta = np.array([fixed[name] for name in PARAM_NAMES], dtype=float)
    for name, value in zip(free_names, free_values):
        theta[PARAM_NAMES.index(name)] = value
    return theta


def deviance(
    detector: SiteFit, theta: np.ndarray, sigma_ln: float
) -> tuple[float, np.ndarray | None]:
    """Chi-square of one parameter vector against a published curve.

    Parameters
    ----------
    detector : SiteFit
        The site, whose ``mask`` selects the nodes that enter.
    theta : np.ndarray, shape (5,)
        The parameter vector.
    sigma_ln : float
        Assumed fractional error per node, as a width in the logarithm.

    Returns
    -------
    deviance : float
        ``sum (ln(observed / predicted) / sigma_ln)^2``, or ``inf`` where the
        model returns nothing positive and finite.
    residual : np.ndarray or None
        ``ln(observed / predicted)`` on the masked nodes, ``None`` when the
        deviance is infinite.
    """
    predicted = detector.predict(theta, detector.mask)
    if not np.all(np.isfinite(predicted)) or np.any(predicted <= 0.0):
        return np.inf, None
    residual = np.log(detector.observed[detector.mask] / predicted)
    return float(np.sum((residual / sigma_ln) ** 2)), residual


def reduced_log_probability(
    detector: SiteFit,
    free_names: tuple[str, ...],
    fixed: dict[str, float],
    sigma_ln: float,
) -> Callable[[np.ndarray], float]:
    """Build the flat-prior log posterior over the free parameters.

    Parameters
    ----------
    detector : SiteFit
        The site, whose ``priors`` box the free parameters.
    free_names : tuple of str
        Names of the free parameters.
    fixed : dict
        Value of every one of :data:`PARAM_NAMES`.
    sigma_ln : float
        Assumed fractional error per node, as a width in the logarithm.

    Returns
    -------
    log_probability : callable
        ``log_probability(values)`` over the free parameters alone.
    """

    def log_probability(values: np.ndarray) -> float:
        for name, value in zip(free_names, values):
            low, high = detector.priors[name]
            if not low < value < high:
                return -np.inf
        dev, _ = deviance(detector, full_theta(free_names, values, fixed), sigma_ln)
        return -0.5 * dev

    return log_probability


def fit(
    detector: SiteFit,
    free_names: tuple[str, ...],
    fixed: dict[str, float],
    sigma_ln: float,
    steps: int,
    walkers: int,
    seed: int,
    n_starts: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Locate the best fit, then sample around it.

    A Nelder-Mead search from ``n_starts`` random points inside the prior box
    finds the minimum first, so that the ensemble starts on it rather than
    spending its burn-in walking there.

    Parameters
    ----------
    detector : SiteFit
        The site.
    free_names : tuple of str
        Names of the free parameters.
    fixed : dict
        Value of every one of :data:`PARAM_NAMES`.
    sigma_ln : float
        Assumed fractional error per node, as a width in the logarithm.
    steps : int
        Steps per walker.
    walkers : int
        Number of walkers.
    seed : int
        Seed of the multi-start search and of the initial walker scatter.
    n_starts : int, optional
        Random starting points of the search, besides ``fixed`` itself.

    Returns
    -------
    best : np.ndarray, shape (len(free_names),)
        Highest-posterior sample of the chain.
    chain : np.ndarray, shape (n_samples, len(free_names))
        The flattened chain after burn-in.
    """
    from scipy.optimize import minimize

    from softpaws.comparison.posterior import sample_posterior

    log_probability = reduced_log_probability(detector, free_names, fixed, sigma_ln)
    start = np.array([fixed[name] for name in free_names], dtype=float)
    best, best_value = start, -log_probability(start)
    rng = np.random.default_rng(seed)
    for _ in range(n_starts):
        x0 = np.array([rng.uniform(*detector.priors[name]) for name in free_names])
        if not np.isfinite(log_probability(x0)):
            continue
        result = minimize(
            lambda x: -log_probability(x),
            x0,
            method="Nelder-Mead",
            options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 2000},
        )
        if result.fun < best_value:
            best, best_value = result.x, result.fun
    scatter = np.array([WALKER_SCATTER[name] for name in free_names])
    chain, sampled_best = sample_posterior(
        log_probability, best, scatter, walkers, steps, seed
    )
    return sampled_best, chain


# ---------------------------------------------------------------------------
# Reading the chains back
# ---------------------------------------------------------------------------


def load_two_parameter_chains(
    path: pathlib.Path, sites: tuple[str, ...] = REDUCED_SITES
) -> tuple[dict[str, np.ndarray], float]:
    """Read the two-parameter chains of a reduced fit.

    Parameters
    ----------
    path : pathlib.Path
        The ``npz`` archive the fit wrote.
    sites : tuple of str, optional
        Sites to read. Defaults to :data:`REDUCED_SITES`.

    Returns
    -------
    chains : dict
        Site -> array of shape ``(n_samples, 2)``, columns
        ``log10(E_thr / GeV)`` and ``Lambda`` [m per e-fold].
    sigma : float
        Assumed fractional error per node the fit ran at.
    """
    data = np.load(path)
    chains = {}
    for site in sites:
        chain = data[f"{site}_2p_chain"]
        chains[site] = np.column_stack([chain[:, 0], 1.0e3 * chain[:, 1]])
    return chains, float(data["sigma"])


def reduced_chain_summary(chains: dict[str, np.ndarray]) -> dict[str, dict[str, tuple]]:
    """Median and central 68% interval of each two-parameter posterior.

    Parameters
    ----------
    chains : dict
        Site -> array of shape ``(n_samples, 2)``; see
        :func:`load_two_parameter_chains`.

    Returns
    -------
    summary : dict
        Site -> ``{"e_thr_gev": (low, median, high), "reach_m": (low, median,
        high)}``, with the threshold in GeV and the reach in m per e-fold.
    """
    summary = {}
    for site, chain in chains.items():
        e_thr = 10.0 ** np.percentile(chain[:, 0], [16, 50, 84])
        reach = np.percentile(chain[:, 1], [16, 50, 84])
        summary[site] = {
            "e_thr_gev": tuple(float(v) for v in e_thr),
            "reach_m": tuple(float(v) for v in reach),
        }
    return summary


def reach_separation_sigma(chains: dict[str, np.ndarray]) -> list[tuple[str, str, float]]:
    """Separation of every pair of reach medians, in combined 68% half-widths.

    Parameters
    ----------
    chains : dict
        Site -> array of shape ``(n_samples, 2)``; see
        :func:`load_two_parameter_chains`.

    Returns
    -------
    pairs : list of tuple
        ``(first, second, separation)``, one per unordered pair, in the order
        the sites appear in ``chains``.
    """
    names = list(chains)
    pairs = []
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            qa = np.percentile(chains[first][:, 1], [16, 50, 84])
            qb = np.percentile(chains[second][:, 1], [16, 50, 84])
            width = np.hypot(0.5 * (qa[2] - qa[0]), 0.5 * (qb[2] - qb[0]))
            pairs.append((first, second, float(abs(qa[1] - qb[1]) / width)))
    return pairs


def attach_reduced_chains(
    detectors: list[SiteFit], chains_path: pathlib.Path
) -> list[SiteFit]:
    """Give each detector the five-column form of its two-parameter chain.

    The reduced fit samples only the threshold and the reach, so the other
    three columns are filled with the values the fit held them at.

    Parameters
    ----------
    detectors : list of SiteFit
        Sites to fill in, each of which must have a chain in the archive.
    chains_path : pathlib.Path
        The ``npz`` archive the reduced fit wrote.

    Returns
    -------
    detectors : list of SiteFit
        The same objects, with ``chain`` set.
    """
    data = np.load(chains_path)
    for detector in detectors:
        fixed = {
            "eps_0": EPS_FIXED[detector.name],
            "log10_e_thr": 3.0,
            "b_scale": 1.0,
            "lam": LAMBDA_BGR18,
            "reach_km": 0.03,
        }
        two = data[f"{detector.name}_2p_chain"]
        detector.chain = np.array([full_theta(FREE2, row, fixed) for row in two])
    return detectors


# ---------------------------------------------------------------------------
# TRIDENT's 2025 effective-area map
# ---------------------------------------------------------------------------

#: Footprint radius of the 2025 reference layout [km]: 1000 strings at 100 m
#: average spacing, about 9.6 km^2.
MAP_RADIUS_KM = TRIDENT_2025.radius_km

#: ``cos(theta_z)`` bin edges of the map, downgoing first.
COS_EDGES = np.linspace(1.0, -1.0, 13)

#: Prior box of the three instrument numbers the map is fitted with.
PARAM_BOUNDS = {"eps_0": (0.05, 1.5), "log10_e_thr": (1.5, 4.5), "reach_km": (-0.08, 0.40)}


def trident_2025_cells(
    path: pathlib.Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TRIDENT's 2025 effective-area map, cell by cell.

    Parameters
    ----------
    path : pathlib.Path or None, optional
        Table to read; see :func:`softpaws.data.published.trident_2025_map`.

    Returns
    -------
    cos_theta : np.ndarray, shape (12,)
        ``cos(theta_z)`` bin centres, downgoing first.
    log10_e : np.ndarray, shape (12,)
        ``log10(E_nu / GeV)`` bin centres.
    log10_aeff_cm2 : np.ndarray, shape (12, 12)
        ``log10`` of the effective area [cm^2], one row per ``cos_theta``.
    """
    from softpaws.data.published import trident_2025_map

    return trident_2025_map(path)


class MapModel:
    """Model effective area on the map's grid, per ``cos(theta_z)`` bin.

    Attributes
    ----------
    site : WaterSite
        TRIDENT at the 2025 reference footprint.
    zenith_weights : np.ndarray
        Solid-angle weights of the whole-sky zenith grid.
    muon_column_km : np.ndarray
        Column available upstream of the detector [km of sea water].
    ladders : dict
        Transmission ladders of the site.
    weights : list of np.ndarray
        One masked copy of ``zenith_weights`` per ``cos(theta_z)`` bin.
    grid : np.ndarray
        Energy grid the forward model is evaluated on.
    log10_e : np.ndarray
        Energy grid the map is read on, as ``log10(E_nu / GeV)``.
    """

    def __init__(self, log10_e: np.ndarray, site: WaterSite | None = None) -> None:
        """Bind the site's columns and ladders to the map's cell layout.

        Parameters
        ----------
        log10_e : np.ndarray
            Energy bin centres of the map, as ``log10(E_nu / GeV)``.
        site : WaterSite or None, optional
            The detector. ``None`` uses TRIDENT at :data:`MAP_RADIUS_KM`.
        """
        if site is None:
            base = [s for s in water_sites() if s.name == "TRIDENT"][0]
            site = dataclasses.replace(base, radius_km=MAP_RADIUS_KM)
        self.site = site
        self.zenith_weights, neutrino_column, self.muon_column_km = water_columns(site)
        self.ladders = water_ladders(site, neutrino_column)
        theta_deg, _ = arca_zenith_grid()
        cos_theta = np.cos(np.deg2rad(theta_deg))
        self.weights = [
            np.where(
                (cos_theta >= COS_EDGES[i + 1]) & (cos_theta < COS_EDGES[i]),
                self.zenith_weights,
                0.0,
            )
            for i in range(COS_EDGES.size - 1)
        ]
        self.grid = ARCA_LOG10_E
        self.log10_e = log10_e

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        """Model effective area on the map's cells.

        Parameters
        ----------
        theta : np.ndarray, shape (5,)
            ``(eps_0, log10_e_thr, b_scale, lam, reach_km)``.

        Returns
        -------
        log10_aeff_cm2 : np.ndarray, shape (len(weights), log10_e.size)
            ``log10`` of the effective area [cm^2], one row per band.
        """
        out = np.empty((len(self.weights), self.log10_e.size))
        for i, weights in enumerate(self.weights):
            full = water_model(
                theta, self.site, self.ladders, weights, self.muon_column_km, None
            )
            out[i] = np.interp(self.log10_e, self.grid, np.log10(full))
        return out


def fit_cells(
    model: MapModel,
    log10_a: np.ndarray,
    cells: np.ndarray,
    sigma_dex: float,
) -> tuple[np.ndarray, float, tuple[float, float], np.ndarray]:
    """Fit the three instrument numbers over a selection of map cells.

    The physics is held at ``b_scale = 1`` and ``lam`` at BGR18, so only the
    selection normalization, the threshold and the reach move. The reach is
    then profiled: the other two are refitted at fixed reach until the
    chi-square rises by one.

    Parameters
    ----------
    model : MapModel
        The forward model on the map's cells.
    log10_a : np.ndarray
        ``log10`` of the published effective area [cm^2], same shape as the
        model's output.
    cells : np.ndarray
        Boolean mask of the cells that enter the chi-square.
    sigma_dex : float
        Assumed error per cell [dex].

    Returns
    -------
    best : np.ndarray, shape (3,)
        ``(eps_0, log10_e_thr, reach_km)`` at the minimum.
    chi2 : float
        Chi-square there.
    reach_68 : tuple of float
        Profile interval of ``reach_km`` [km per e-fold].
    residual : np.ndarray
        ``log10`` of published over model, on every cell.
    """
    from scipy.optimize import minimize, minimize_scalar

    def theta_of(x):
        return np.array([x[0], x[1], 1.0, LAMBDA_BGR18, x[2]])

    def chi_square(x):
        for value, (low, high) in zip(x, PARAM_BOUNDS.values()):
            if not low < value < high:
                return 1.0e9
        residual = log10_a[cells] - model(theta_of(x))[cells]
        return float(np.sum((residual / sigma_dex) ** 2))

    starts = ([0.7, 2.5, 0.02], [0.5, 3.0, -0.03], [0.9, 2.2, 0.06], [0.6, 3.4, 0.0])
    best = min(
        (
            minimize(
                chi_square,
                x0,
                method="Nelder-Mead",
                options={"xatol": 1e-4, "fatol": 1e-3, "maxiter": 4000},
            )
            for x0 in starts
        ),
        key=lambda result: result.fun,
    )

    def profile(reach):
        result = minimize(
            lambda y: chi_square([y[0], y[1], reach]),
            best.x[:2],
            method="Nelder-Mead",
            options={"xatol": 1e-4, "fatol": 1e-3},
        )
        return result.fun

    low = minimize_scalar(
        lambda r: (profile(r) - best.fun - 1.0) ** 2,
        bounds=(PARAM_BOUNDS["reach_km"][0], best.x[2]),
        method="bounded",
    ).x
    high = minimize_scalar(
        lambda r: (profile(r) - best.fun - 1.0) ** 2,
        bounds=(best.x[2], PARAM_BOUNDS["reach_km"][1]),
        method="bounded",
    ).x
    residual = log10_a - model(theta_of(best.x))
    return best.x, best.fun, (float(low), float(high)), residual


def trident_2025_detector(
    cos_max: float = 0.5,
    log10_e_min: float = 5.0,
    path: pathlib.Path | None = None,
) -> tuple[SiteFit, int]:
    """The 2025 map, cell by cell, as one detector for the reduced fit.

    Only the cells where the published selection is flat are kept,
    ``|cos(theta_z)| <= cos_max``, and only the decade the two TRIDENT tables
    share. The map is published after trigger and quality cuts, so the
    selection normalization is left free.

    Parameters
    ----------
    cos_max : float, optional
        Widest ``|cos(theta_z)|`` kept.
    log10_e_min : float, optional
        Lowest energy kept, as ``log10(E_nu / GeV)``.
    path : pathlib.Path or None, optional
        Table to read; see :func:`trident_2025_cells`.

    Returns
    -------
    detector : SiteFit
        TRIDENT, with every kept cell as one node of a flattened curve.
    n_cells : int
        Number of cells that survived both cuts.
    """
    cos_theta, log10_e_all, log10_a_all = trident_2025_cells(path)
    keep = log10_e_all >= log10_e_min
    log10_e, log10_a = log10_e_all[keep], log10_a_all[:, keep]
    rows = np.abs(cos_theta) <= cos_max
    model = MapModel(log10_e)
    # Only the selected cos bins are needed; drop the others from the loop.
    model.weights = [w for w, row in zip(model.weights, rows) if row]
    observed = 10.0 ** log10_a[rows].ravel()

    def predict(theta, select=None):
        out = np.empty((int(rows.sum()), log10_e.size))
        for i, weights in enumerate(model.weights):
            full = water_model(
                theta, model.site, model.ladders, weights, model.muon_column_km, None
            )
            out[i] = np.interp(log10_e, model.grid, np.log10(full))
        predicted = 10.0**out.ravel()
        return predicted if select is None else predicted[select]

    priors = dict(PRIORS["ARCA230"])
    priors["reach_km"] = PARAM_BOUNDS["reach_km"]
    priors["eps_0"] = PARAM_BOUNDS["eps_0"]
    detector = SiteFit(
        name="TRIDENT",
        log10_e=np.tile(log10_e, int(rows.sum())),
        observed=observed,
        mask=np.ones(observed.size, bool),
        predict=predict,
        priors=priors,
        start=np.array([0.7, 2.5, 1.0, LAMBDA_BGR18, 0.03]),
        selection_level="2025 map",
    )
    return detector, int(rows.sum()) * log10_e.size


def trident_2025_average_detector(
    chain_path: pathlib.Path,
    cos_max: float = 0.5,
    log10_e_min: float | None = None,
    path: pathlib.Path | None = None,
) -> tuple[SiteFit, np.ndarray, dict[str, float]]:
    """The 2025 map averaged over its flat-selection cells, with its posterior.

    The map's cells carry their simulation's statistics as a checkerboard of a
    few hundredths of a dex, and the band average inherits a kink near
    ``10^5.3`` GeV. A quadratic in log-log is the smoothest curve with the
    right curvature over two decades, and the residual it removes is reported.

    Parameters
    ----------
    chain_path : pathlib.Path
        The ``npz`` archive :func:`trident_2025_detector`'s fit wrote.
    cos_max : float, optional
        Widest ``|cos(theta_z)|`` the average runs over.
    log10_e_min : float or None, optional
        Lowest energy kept, as ``log10(E_nu / GeV)``. ``None`` uses the bottom
        of the water-site energy grid.
    path : pathlib.Path or None, optional
        Table to read; see :func:`trident_2025_cells`.

    Returns
    -------
    detector : SiteFit
        TRIDENT, with the smoothed band average as its published curve and the
        five-column form of the chain.
    allsky_cm2 : np.ndarray
        The same average over every band [cm^2], so the cost of the nadir
        selection is on record.
    smoothing : dict
        ``"rms"`` and ``"max"`` of the residual the smoothing removes [dex].
    """
    if log10_e_min is None:
        log10_e_min = float(ARCA_LOG10_E.min())
    cos_theta, log10_e_all, log10_a_all = trident_2025_cells(path)
    keep = log10_e_all >= log10_e_min
    log10_e, log10_a = log10_e_all[keep], log10_a_all[:, keep]
    d_cos = np.abs(np.diff(COS_EDGES))
    rows = np.abs(cos_theta) <= cos_max
    area = 10.0**log10_a
    fitted = np.average(area[rows], axis=0, weights=d_cos[rows])
    allsky = np.average(area, axis=0, weights=d_cos)
    coefficients = np.polyfit(log10_e, np.log10(fitted), 2)
    smoothed = 10.0 ** np.polyval(coefficients, log10_e)
    smoothing = {
        "rms": float(np.std(np.log10(fitted / smoothed))),
        "max": float(np.max(np.abs(np.log10(fitted / smoothed)))),
    }

    model = MapModel(log10_e)
    weights_per_band = [w for w, row in zip(model.weights, rows) if row]

    def predict(theta, select=None):
        curves = np.empty((len(weights_per_band), log10_e.size))
        for i, weights in enumerate(weights_per_band):
            full = water_model(
                theta, model.site, model.ladders, weights, model.muon_column_km, None
            )
            curves[i] = 10.0 ** np.interp(log10_e, model.grid, np.log10(full))
        predicted = np.average(curves, axis=0, weights=d_cos[rows])
        return predicted if select is None else predicted[select]

    fixed = {
        "eps_0": 0.7,
        "log10_e_thr": 2.5,
        "b_scale": 1.0,
        "lam": LAMBDA_BGR18,
        "reach_km": 0.03,
    }
    chain = np.load(chain_path)["chain"]
    priors = dict(PRIORS["ARCA230"])
    priors["reach_km"] = PARAM_BOUNDS["reach_km"]
    priors["eps_0"] = PARAM_BOUNDS["eps_0"]
    detector = SiteFit(
        name="TRIDENT",
        log10_e=log10_e,
        observed=smoothed,
        mask=np.ones(log10_e.size, bool),
        predict=predict,
        priors=priors,
        start=np.array([fixed[name] for name in PARAM_NAMES]),
        selection_level="2025 map",
        chain=np.array([full_theta(FREE3, row, fixed) for row in chain]),
    )
    return detector, allsky, smoothing
