"""Posterior sampling and the statistics read off a set of chains.

The fits of the paper sample a handful of parameters per detector with an
affine-invariant ensemble sampler and then ask three questions of the
chains: what each marginal says (and whether it is the prior box talking),
what the product of independent posteriors says on a shared subspace, and
whether two or more posteriors agree there. The functions here answer them
for any chains; the forward models and the priors stay with the fits.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2, gaussian_kde

__all__ = [
    "credible_interval",
    "global_compatibility",
    "inside_box",
    "intersection",
    "leave_one_out_compatibility",
    "log_gaussian_in_log",
    "marginal_summary",
    "pairwise_compatibility",
    "product_posterior",
    "sample_posterior",
]


def inside_box(theta: np.ndarray, bounds) -> bool:
    """Whether every parameter lies strictly inside its prior box.

    Parameters
    ----------
    theta : np.ndarray
        Parameter values.
    bounds : sequence of tuple of float
        ``(low, high)`` per parameter, in the same order.

    Returns
    -------
    inside : bool
        True if ``low < value < high`` for every parameter.
    """
    return all(low < value < high for value, (low, high) in zip(theta, bounds))


def log_gaussian_in_log(observed: np.ndarray, predicted: np.ndarray, sigma_ln: float) -> float:
    """Log likelihood that is Gaussian in ``ln(observed / predicted)``.

    A prediction that is not finite or not positive is outside the model,
    not merely unlikely, and returns ``-inf``.

    Parameters
    ----------
    observed : np.ndarray
        Observed values, positive.
    predicted : np.ndarray
        Predicted values on the same points.
    sigma_ln : float
        Assumed fractional error, as a width in the logarithm.

    Returns
    -------
    log_likelihood : float
        ``-0.5 sum (ln(observed / predicted) / sigma_ln)^2``, or ``-inf``.
    """
    predicted = np.asarray(predicted, dtype=float)
    if not np.all(np.isfinite(predicted)) or np.any(predicted <= 0.0):
        return -np.inf
    residual = np.log(np.asarray(observed, dtype=float) / predicted)
    return -0.5 * float(np.sum((residual / sigma_ln) ** 2))


def sample_posterior(
    log_probability,
    start: np.ndarray,
    scatter: np.ndarray,
    walkers: int,
    steps: int,
    seed: int,
    args: tuple = (),
    discard_fraction: float = 1.0 / 3.0,
    processes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a posterior with an affine-invariant ensemble and flatten the chain.

    Parameters
    ----------
    log_probability : callable
        ``log_probability(theta, *args)``, the log posterior.
    start : np.ndarray
        Central starting point of the walkers.
    scatter : np.ndarray
        Per-parameter Gaussian scatter of the initial walkers about ``start``.
    walkers : int
        Number of walkers.
    steps : int
        Steps per walker.
    seed : int
        Seed of the initial scatter.
    args : tuple, optional
        Extra positional arguments of ``log_probability``.
    discard_fraction : float, optional
        Fraction of the steps discarded as burn-in.
    processes : int or None, optional
        Worker processes to evaluate the ensemble with. ``None``, the default,
        runs in this process. The walkers of one step are independent, so
        splitting them across cores is close to linear in the number of them
        and leaves the chain unchanged: the proposal draws come from the
        sampler's own generator and the results are gathered back in walker
        order. Only worth it where one evaluation costs more than the round
        trip, which for the forward models here means milliseconds and up.
        ``log_probability`` and everything in ``args`` have to be picklable,
        so both must be reachable by import and not defined in a script that
        another one loads by path.

    Returns
    -------
    chain : np.ndarray, shape (n_samples, n_parameters)
        The flattened chain after burn-in.
    best : np.ndarray, shape (n_parameters,)
        The sample of highest log probability.
    """
    import contextlib
    import multiprocessing

    import emcee

    start = np.asarray(start, dtype=float)
    rng = np.random.default_rng(seed)
    initial = start + np.asarray(scatter, dtype=float) * rng.standard_normal(
        (walkers, start.size)
    )
    opened = (
        multiprocessing.Pool(processes)
        if processes is not None and processes > 1
        else contextlib.nullcontext()
    )
    with opened as pool:
        sampler = emcee.EnsembleSampler(
            walkers, start.size, log_probability, args=args, pool=pool
        )
        sampler.run_mcmc(initial, steps, progress=False)
    discard = int(steps * discard_fraction)
    chain = sampler.get_chain(discard=discard, flat=True)
    best = chain[np.argmax(sampler.get_log_prob(discard=discard, flat=True))]
    return chain, best


def credible_interval(samples: np.ndarray, level: float) -> tuple[float, float]:
    """Central credible interval of one-dimensional samples.

    Parameters
    ----------
    samples : np.ndarray
        Samples of one parameter.
    level : float
        Probability content, for example ``0.68``.

    Returns
    -------
    low, high : float
        The interval edges.
    """
    half = 100.0 * (1.0 - level) / 2.0
    lo, hi = np.percentile(samples, [half, 100.0 - half])
    return float(lo), float(hi)


def marginal_summary(
    chain: np.ndarray,
    names: tuple[str, ...],
    priors: dict[str, tuple[float, float]],
    best: np.ndarray | None = None,
    rail_fraction: float = 0.02,
) -> dict[str, dict]:
    """Median, credible intervals and prior-rail flags of every marginal.

    A 68% edge sitting on a prior edge means the prior box is doing the
    constraining and not the data: the interval is then a bound, not a
    measurement, and must not be quoted as one.

    Parameters
    ----------
    chain : np.ndarray, shape (n_samples, n_parameters)
        The flattened chain.
    names : tuple of str
        Parameter names, one per column.
    priors : dict
        Name -> ``(low, high)`` prior box.
    best : np.ndarray or None, optional
        Best-fit point; ``None`` omits the ``"best_fit"`` entry.
    rail_fraction : float, optional
        Fraction of the prior width within which a 68% edge counts as
        railed against the prior.

    Returns
    -------
    summary : dict
        Name -> ``{"median", "best_fit", "ci68", "ci95", "prior",
        "rails_prior"}``.
    """
    out = {}
    for i, name in enumerate(names):
        lo68, hi68 = credible_interval(chain[:, i], 0.68)
        lo95, hi95 = credible_interval(chain[:, i], 0.95)
        low, high = priors[name]
        span = high - low
        entry = {
            "median": float(np.median(chain[:, i])),
            "ci68": [lo68, hi68],
            "ci95": [lo95, hi95],
            "prior": [float(low), float(high)],
            "rails_prior": {
                "low": bool(lo68 - low < rail_fraction * span),
                "high": bool(high - hi68 < rail_fraction * span),
            },
        }
        if best is not None:
            entry["best_fit"] = float(best[i])
        out[name] = entry
    return out


def intersection(intervals) -> dict:
    """Overlap of several intervals.

    Parameters
    ----------
    intervals : sequence of tuple of float
        ``(low, high)`` per posterior.

    Returns
    -------
    overlap : dict
        ``{"low", "high", "empty"}``; ``empty`` is True when the intervals
        do not meet.
    """
    lo = max(interval[0] for interval in intervals)
    hi = min(interval[1] for interval in intervals)
    return {"low": float(lo), "high": float(hi), "empty": bool(hi <= lo)}


def product_posterior(
    chains: list[np.ndarray],
    names: tuple[str, ...],
    n_grid: int = 160,
    bound_percentiles: tuple[float, float] = (0.5, 99.5),
) -> dict:
    """Product of independent posteriors on a shared subspace.

    Each chain is turned into a kernel density estimate on a common grid
    spanning the union of the chains' central ranges, the densities are
    multiplied, and the marginals of the product are summarized.

    Parameters
    ----------
    chains : list of np.ndarray
        One ``(n_samples, len(names))`` array per posterior, columns in the
        order of ``names``.
    names : tuple of str
        Names of the shared parameters.
    n_grid : int, optional
        Grid points per axis.
    bound_percentiles : tuple of float, optional
        Percentiles of each chain that bound the grid.

    Returns
    -------
    product : dict
        ``"params"``, then per name ``{"median", "ci68"}``, and ``"mode"``.
    """
    bounds = []
    for k in range(len(names)):
        lo = min(np.percentile(c[:, k], bound_percentiles[0]) for c in chains)
        hi = max(np.percentile(c[:, k], bound_percentiles[1]) for c in chains)
        bounds.append((float(lo), float(hi)))
    axes = [np.linspace(lo, hi, n_grid) for lo, hi in bounds]
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.vstack([m.ravel() for m in mesh])
    density = np.ones(points.shape[1])
    for c in chains:
        density *= gaussian_kde(c.T)(points)
    density = density.reshape(mesh[0].shape)
    total = density.sum()
    out: dict = {"params": list(names)}
    for axis, (name, grid) in enumerate(zip(names, axes)):
        others = tuple(a for a in range(density.ndim) if a != axis)
        marginal = density.sum(axis=others) / total
        cumulative = np.cumsum(marginal)
        median, lo68, hi68 = np.interp([0.5, 0.16, 0.84], cumulative, grid)
        out[name] = {"median": float(median), "ci68": [float(lo68), float(hi68)]}
    peak = np.unravel_index(np.argmax(density), density.shape)
    out["mode"] = {
        name: float(grid[peak[axis]]) for axis, (name, grid) in enumerate(zip(names, axes))
    }
    return out


def _sigma(p_value: float) -> float:
    return float(np.sqrt(chi2.isf(p_value, 1))) if p_value > 0.0 else float("inf")


def pairwise_compatibility(chain_a: np.ndarray, chain_b: np.ndarray) -> dict:
    """Mahalanobis distance between two posterior means, as a chi-square.

    The two posteriors are summarized by their means and covariances; the
    distance between the means against the summed covariances is a
    chi-square with as many degrees of freedom as parameters.

    Parameters
    ----------
    chain_a, chain_b : np.ndarray
        ``(n_samples, n_parameters)`` chains on the same parameters.

    Returns
    -------
    result : dict
        ``{"dof", "chi2", "p_value", "sigma", "delta"}``, where ``delta`` is
        the difference of the means.
    """
    means = [chain_a.mean(axis=0), chain_b.mean(axis=0)]
    covariances = [np.cov(chain_a.T), np.cov(chain_b.T)]
    delta = means[0] - means[1]
    chi_square = float(delta @ np.linalg.solve(covariances[0] + covariances[1], delta))
    dof = delta.size
    p_value = float(chi2.sf(chi_square, dof))
    return {
        "dof": dof,
        "chi2": chi_square,
        "p_value": p_value,
        "sigma": _sigma(p_value),
        "delta": delta,
    }


def _precision_combine(means, precisions, subset):
    precision = sum(precisions[i] for i in subset)
    covariance = np.linalg.inv(precision)
    mean = covariance @ sum(precisions[i] @ means[i] for i in subset)
    return mean, covariance


def leave_one_out_compatibility(chains: list[np.ndarray]) -> list[dict]:
    """Each posterior against the precision-weighted mean of the others.

    Parameters
    ----------
    chains : list of np.ndarray
        ``(n_samples, n_parameters)`` chains on the same parameters.

    Returns
    -------
    results : list of dict
        Per chain, ``{"chi2", "p_value", "sigma"}`` of its mean against the
        combination of the others.
    """
    means = [c.mean(axis=0) for c in chains]
    precisions = [np.linalg.inv(np.cov(c.T)) for c in chains]
    n = len(chains)
    out = []
    for i in range(n):
        mean, covariance = _precision_combine(means, precisions, [j for j in range(n) if j != i])
        delta = means[i] - mean
        chi_square = float(
            delta @ np.linalg.solve(covariance + np.linalg.inv(precisions[i]), delta)
        )
        p_value = float(chi2.sf(chi_square, delta.size))
        out.append({"chi2": chi_square, "p_value": p_value, "sigma": _sigma(p_value)})
    return out


def global_compatibility(chains: list[np.ndarray], names: tuple[str, ...]) -> dict:
    """All posteriors about their common precision-weighted mean.

    Parameters
    ----------
    chains : list of np.ndarray
        ``(n_samples, len(names))`` chains on the same parameters.
    names : tuple of str
        Parameter names, for the combined-mean entries.

    Returns
    -------
    result : dict
        ``{"chi2", "dof", "p_value", "sigma", "combined_mean",
        "combined_sigma"}`` with ``2 (n - 1)`` degrees of freedom for two
        parameters, and in general ``len(names) (n - 1)``.
    """
    means = [c.mean(axis=0) for c in chains]
    precisions = [np.linalg.inv(np.cov(c.T)) for c in chains]
    n = len(chains)
    mean, covariance = _precision_combine(means, precisions, range(n))
    chi_square = float(
        sum((means[i] - mean) @ precisions[i] @ (means[i] - mean) for i in range(n))
    )
    dof = len(names) * (n - 1)
    p_value = float(chi2.sf(chi_square, dof))
    return {
        "chi2": chi_square,
        "dof": dof,
        "p_value": p_value,
        "sigma": _sigma(p_value),
        "combined_mean": {name: float(mean[k]) for k, name in enumerate(names)},
        "combined_sigma": {name: float(np.sqrt(covariance[k, k])) for k, name in enumerate(names)},
    }
