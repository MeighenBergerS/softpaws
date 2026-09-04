"""Feldman-Cousins calibration of a profile-likelihood interval.

A profile statistic ``q(r) = -2 ln L(r) + 2 ln L_min`` scanned over a
parameter ``r`` gives a Wilks interval where it crosses ``1.0`` (68%) or
``3.84`` (95%). Near a physical boundary Wilks fails, and the threshold at
each true ``r`` is calibrated instead from pseudo-experiments: the
distribution of the same statistic under toys drawn at that truth. The
functions here run the toys for any model that can produce an expectation
and evaluate the statistic, cache the result, and read intervals off the
calibrated thresholds.
"""

from __future__ import annotations

import logging
import pathlib

import numpy as np

__all__ = [
    "cached_toys",
    "calibrated_thresholds",
    "profile_interval",
    "toy_statistics",
]

logger = logging.getLogger(__name__)


def profile_interval(
    grid: np.ndarray, curve: np.ndarray, level: float | np.ndarray
) -> tuple[float, float]:
    """Crossing points of a profile statistic at one threshold.

    Parameters
    ----------
    grid : np.ndarray
        Parameter values the statistic is tabulated on.
    curve : np.ndarray
        Profile statistic on ``grid``.
    level : float or np.ndarray
        Threshold, either one Wilks level or a calibrated threshold per grid
        point.

    Returns
    -------
    low, high : float
        Lowest and highest grid values where the statistic is at or below
        the threshold.

    Raises
    ------
    ValueError
        Raised if the statistic is above the threshold everywhere.
    """
    grid = np.asarray(grid, dtype=float)
    below = np.asarray(curve) <= level
    if not np.any(below):
        raise ValueError("The profile statistic exceeds the threshold on the whole grid.")
    return float(grid[below].min()), float(grid[below].max())


def calibrated_thresholds(
    q: np.ndarray,
    truths: np.ndarray,
    grid: np.ndarray,
    percentiles: tuple[float, ...] = (68.27, 95.0),
) -> dict[float, np.ndarray]:
    """Per-grid-point thresholds from the toy distributions at each truth.

    Parameters
    ----------
    q : np.ndarray, shape (n_truths, n_toys)
        Statistic distributions, one row per truth point.
    truths : np.ndarray, shape (n_truths,)
        The truth points.
    grid : np.ndarray
        Parameter grid to interpolate the thresholds onto.
    percentiles : tuple of float, optional
        Percentiles of the toy distribution that define the thresholds.

    Returns
    -------
    thresholds : dict
        Percentile -> threshold on ``grid``, linearly interpolated between
        the truth points.
    """
    return {
        p: np.interp(grid, truths, np.percentile(q, p, axis=1)) for p in percentiles
    }


def toy_statistics(
    truths: np.ndarray,
    expectation,
    statistic,
    n_toys: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Distributions of a profile statistic under Poisson pseudo-experiments.

    Parameters
    ----------
    truths : np.ndarray, shape (n_truths,)
        Truth points.
    expectation : callable
        ``expectation(truth)`` -> expected counts (array) at that truth; the
        toys are Poisson draws of it.
    statistic : callable
        ``statistic(truth, toy)`` -> the profile statistic of one toy at the
        truth it was drawn from: the fixed fit minus the toy's global
        minimum.
    n_toys : int
        Pseudo-experiments per truth point.
    rng : np.random.Generator
        Random generator of the Poisson draws.

    Returns
    -------
    q : np.ndarray, shape (n_truths, n_toys)
        Statistic distributions, one row per truth point.
    """
    truths = np.asarray(truths, dtype=float)
    q = np.empty((truths.size, n_toys))
    for i, truth in enumerate(truths):
        mu = expectation(truth)
        for t in range(n_toys):
            q[i, t] = statistic(truth, rng.poisson(mu).astype(float))
        c68, c95 = np.percentile(q[i], 68.27), np.percentile(q[i], 95.0)
        logger.info("truth %.2f: c68 %.2f, c95 %.2f", truth, c68, c95)
    return q


def cached_toys(
    path: str | pathlib.Path,
    metadata: dict,
    build,
    rebuild: bool = False,
) -> np.ndarray:
    """Toy distributions from a cache file, rebuilt when the metadata differs.

    Parameters
    ----------
    path : str or pathlib.Path
        Cache file (``.npz``).
    metadata : dict
        Values the cache must match to be valid (truth points, number of
        toys, seed, priors, a version number, ...). Arrays are compared
        element-wise, everything else by equality.
    build : callable
        ``build()`` -> ``q``, called when the cache is missing or stale.
    rebuild : bool, optional
        Ignore the cache.

    Returns
    -------
    q : np.ndarray
        The toy distributions.
    """
    path = pathlib.Path(path)
    if path.exists() and not rebuild:
        with np.load(path) as saved:
            valid = all(
                key in saved and np.array_equal(np.asarray(saved[key]), np.asarray(value))
                for key, value in metadata.items()
            )
            if valid:
                logger.info("Cached toy distributions from %s", path)
                return np.asarray(saved["q"])
    q = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, q=q, **metadata)
    return q
