"""Published effective areas, put on the model's energy grid.

Each site's published curve is read from :mod:`softpaws.data.published`,
digitized step curves are smoothed, and the result is interpolated onto the
grid the model is evaluated on, together with the sky it is averaged over.
"""

from __future__ import annotations

import pathlib
from typing import Callable

import numpy as np

from softpaws.detectors import Site

from ..constants import IC_LOG10_E, PUBLISHED_WATER_LOG10_E, SMOOTH_BIN_DEX, SMOOTH_WINDOW

__all__ = [
    "PUBLISHED_SKY",
    "TRIDENT_BAND_WEIGHTS",
    "arca230_trigger",
    "icecube_upgoing",
    "on_arca_grid",
    "pone_allsky_cm2",
    "published_curve",
    "published_effective_area_cm2",
    "trident_allsky_cm2",
]

#: Solid-angle weights of TRIDENT's three published ``cos(theta)`` bands.
TRIDENT_BAND_WEIGHTS = np.array([0.8, 0.4, 0.8]) / 2.0


def icecube_upgoing(data_dir: pathlib.Path) -> np.ndarray:
    """Livetime-weighted DR2 effective area over the upgoing sky.

    Parameters
    ----------
    data_dir : pathlib.Path
        Root of the IceTracks-DR2 release, holding ``irfs/`` and ``uptime/``.

    Returns
    -------
    aeff_cm2 : np.ndarray, shape (IC_LOG10_E.size,)
        Effective area on :data:`IC_LOG10_E` [cm^2].
    """
    from softpaws.data.published import icecube_dr2_aeff

    return icecube_dr2_aeff(data_dir, IC_LOG10_E)[0]


def arca230_trigger() -> np.ndarray:
    """Digitized full-ARCA ``nu_mu`` effective area at trigger level.

    Returns
    -------
    aeff_cm2 : np.ndarray, shape (PUBLISHED_WATER_LOG10_E.size,)
        Effective area on :data:`PUBLISHED_WATER_LOG10_E` [cm^2], ``NaN`` outside the
        digitized range.
    """
    from softpaws.data.published import arca230_trigger_level_aeff, interpolate_aeff

    return interpolate_aeff(PUBLISHED_WATER_LOG10_E, *arca230_trigger_level_aeff())


def _smooth_digitized(log10_e: np.ndarray, log10_a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Smooth a digitized step-function curve.

    Median of ``log10_a`` in :data:`SMOOTH_BIN_DEX` bins of ``log10_e``, then a
    :data:`SMOOTH_WINDOW`-bin running mean, evaluated at the bin centres.

    Parameters
    ----------
    log10_e : np.ndarray
        ``log10(E_nu / GeV)`` of the digitized points.
    log10_a : np.ndarray
        ``log10`` of the digitized effective area.

    Returns
    -------
    centers : np.ndarray
        Bin centres that hold at least one digitized point.
    smoothed : np.ndarray
        The smoothed curve on ``centers``.
    """
    lo = np.floor(log10_e.min() / SMOOTH_BIN_DEX) * SMOOTH_BIN_DEX
    edges = np.arange(lo, log10_e.max() + SMOOTH_BIN_DEX, SMOOTH_BIN_DEX)
    index = np.clip(np.digitize(log10_e, edges) - 1, 0, edges.size - 2)
    centers, medians = [], []
    for k in range(edges.size - 1):
        selected = index == k
        if selected.any():
            centers.append(0.5 * (edges[k] + edges[k + 1]))
            medians.append(np.median(log10_a[selected]))
    centers, medians = np.array(centers), np.array(medians)
    half = SMOOTH_WINDOW // 2
    padded = np.pad(medians, half, mode="edge")
    kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
    return centers, np.convolve(padded, kernel, mode="valid")


def on_arca_grid(log10_e: np.ndarray, log10_a: np.ndarray) -> np.ndarray:
    """Put a smoothed curve on the water-site grid by log-log interpolation.

    Parameters
    ----------
    log10_e : np.ndarray
        ``log10(E_nu / GeV)`` of the curve, ascending.
    log10_a : np.ndarray
        ``log10`` of the effective area [cm^2].

    Returns
    -------
    aeff_cm2 : np.ndarray, shape (PUBLISHED_WATER_LOG10_E.size,)
        Effective area [cm^2], ``NaN`` outside the curve.
    """
    return 10.0 ** np.interp(PUBLISHED_WATER_LOG10_E, log10_e, log10_a, left=np.nan, right=np.nan)


def pone_allsky_cm2() -> np.ndarray:
    """P-ONE's all-sky trigger-level curve, smoothed, on the water-site grid.

    Returns
    -------
    aeff_cm2 : np.ndarray, shape (PUBLISHED_WATER_LOG10_E.size,)
        Effective area [cm^2], ``NaN`` outside the digitized range.
    """
    from softpaws.data.published import pone_allsky_aeff

    log10_e, aeff_cm2 = pone_allsky_aeff()
    return on_arca_grid(*_smooth_digitized(log10_e, np.log10(aeff_cm2)))


def trident_allsky_cm2() -> np.ndarray:
    """TRIDENT's sky average from its three bands, on the water-site grid.

    The three published ``cos(theta)`` bands are smoothed, put on the grid and
    combined with the solid-angle weights of :data:`TRIDENT_BAND_WEIGHTS`.

    Returns
    -------
    aeff_cm2 : np.ndarray, shape (PUBLISHED_WATER_LOG10_E.size,)
        Effective area [cm^2], ``NaN`` outside the digitized range.
    """
    from softpaws.data.published import TRIDENT_COS_BANDS, trident_band_aeff

    bands = []
    for cos_lo, cos_hi in TRIDENT_COS_BANDS:
        log10_e, aeff_cm2 = trident_band_aeff(cos_lo, cos_hi)
        bands.append(on_arca_grid(*_smooth_digitized(log10_e, np.log10(aeff_cm2))))
    return np.sum([w * b for w, b in zip(TRIDENT_BAND_WEIGHTS, bands)], axis=0)


#: Published sky average of each water site, by name.
published_curve: dict[str, Callable[[], np.ndarray]] = {
    "P-ONE": pone_allsky_cm2,
    "TRIDENT": trident_allsky_cm2,
}


#: Sky each published curve is averaged over, as a ``cos(theta)`` range. The
#: DR2 table is quoted over the upgoing sky and every water table over the
#: whole of it, which is the average a model has to be compared in.
PUBLISHED_SKY: dict[str, tuple[float, float]] = {
    "IceCube": (-1.0, 0.0),
    "ARCA230": (-1.0, 1.0),
    "P-ONE": (-1.0, 1.0),
    "TRIDENT": (-1.0, 1.0),
}


def published_effective_area_cm2(
    site: Site, log10_e: np.ndarray, data_dir: pathlib.Path | None = None
) -> tuple[np.ndarray, tuple[float, float]]:
    """One site's published effective area, on a grid, with the sky it covers.

    Parameters
    ----------
    site : Site
        The detector. One of the four in :data:`PUBLISHED_SKY`.
    log10_e : np.ndarray
        Neutrino energies to evaluate at [log10 GeV].
    data_dir : pathlib.Path or None, optional
        Root of the IceTracks-DR2 release, read for IceCube alone. ``None``
        uses :func:`softpaws.data.paths.dr2_dir`.

    Returns
    -------
    aeff_cm2 : np.ndarray, shape (log10_e.size,)
        Published effective area [cm^2], ``NaN`` outside the published range.
    cos_range : tuple of float
        The ``cos(theta)`` band the curve is averaged over.

    Raises
    ------
    KeyError
        Raised for a site with no published curve here.
    """
    from softpaws.data.published import interpolate_aeff

    if site.name not in PUBLISHED_SKY:
        raise KeyError(f"No published effective area ships for {site.name!r}.")
    if site.name == "IceCube":
        from softpaws.data.paths import dr2_dir

        grid, aeff = IC_LOG10_E, icecube_upgoing(dr2_dir() if data_dir is None else data_dir)
    elif site.name == "ARCA230":
        grid, aeff = PUBLISHED_WATER_LOG10_E, arca230_trigger()
    else:
        grid, aeff = PUBLISHED_WATER_LOG10_E, published_curve[site.name]()
    good = np.isfinite(aeff) & (aeff > 0.0)
    return interpolate_aeff(log10_e, grid[good], aeff[good]), PUBLISHED_SKY[site.name]
