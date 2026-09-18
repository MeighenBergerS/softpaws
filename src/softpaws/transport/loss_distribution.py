r"""Full distribution of the accumulated muon log-energy loss.

Where :mod:`softpaws.transport.eigenvalue` gives the single eigenvalue ``Phi(A)``
that a power-law flux excites, this module gives the whole probability law behind
it. A muon observed at energy ``E`` that was produced at energy ``eps`` after
propagating a column ``ell`` has accumulated a log-energy loss

.. math:: w = \\ln(\\varepsilon / E) \\ge 0,

which, under the scale-invariant QED loss rates, is a *subordinator* (a
non-decreasing Levy jump process) whose Laplace exponent is exactly the Mellin
symbol of the collision operator (``docs/theory/exact_soft_volume.md``):

.. math:: \\mathbb{E}\\bigl[e^{-s\\,w(\\ell)}\\bigr] = e^{-\\ell\\,\\Phi(s)}.

A power-law flux ``phi ~ E^{-1-A}`` averages over ``w`` and so only ever sees the
one number ``Phi(A)`` -- the regime of examples 05-08 and 12. A **single** event
(such as KM3-230213A) instead samples ``P(w)`` once, and the heavy tail of that
law -- the chance the parent neutrino was far more energetic than the observed
muon -- is what the Fokker-Planck Gaussian of Palmisano et al. throws away.

This module inverts ``e^{-ell Phi(-i k)}`` to recover ``P(w)`` exactly
(:func:`loss_density`), alongside the Fokker-Planck Gaussian reference
(:func:`loss_density_gaussian`, :func:`gaussian_survival`) it is meant to
supersede. It reuses the same two-moment loss family as the eigenvalue path, via
the complex-capable :func:`softpaws.transport.eigenvalue.phi_symbol`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.special import erfc, polygamma

from ..constants import _GIL_PELAEZ_CHUNK
from .eigenvalue import (
    phi_symbol,
    phi_symbol_three_moment,
    three_moment_loss_spectrum,
    two_moment_loss_spectrum,
)


def invert_log_loss_symbol(
    w_grid: np.ndarray,
    ell_km: float,
    symbol: Callable[[np.ndarray], np.ndarray],
    n_k: int = 2**14,
) -> np.ndarray:
    r"""Density ``P(w)`` from any subordinator symbol, by Fourier inversion.

    The characteristic function of the log-loss is ``phi_w(k) = exp(-ell Phi(-i k))``
    for *any* Laplace exponent ``Phi``, so the inversion

    .. math:: P(w) = \\frac{1}{2\\pi}\\int \\phi_w(k)\\,e^{-i k w}\\,dk

    is shared by every loss family. It is evaluated deterministically on a uniform
    ``k`` grid whose span is fixed by the ``w`` spacing (``k_max = pi / dw``), which
    resolves the far tail without Monte-Carlo noise. The result is clipped at zero
    and renormalized to unit area over ``w_grid``.

    Pass a calibrated closed form (as :func:`loss_density` and
    :func:`loss_density_three_moment` do) or a numerical ``Phi`` built from
    tabulated differential cross sections.

    Parameters
    ----------
    w_grid : np.ndarray
        Uniformly spaced grid of log-loss values ``w = ln(eps / E)`` (>= 0).
    ell_km : float
        Propagated column depth ``ell`` [km].
    symbol : callable
        Laplace exponent ``Phi(s)`` [km^-1], called once with the complex array
        ``s = -i k``. Must accept and return complex arrays.
    n_k : int, optional
        Number of nodes on the ``k`` grid.

    Returns
    -------
    density : np.ndarray
        Probability density ``P(w)``, normalized to unit area over ``w_grid``.

    Raises
    ------
    ValueError
        Raised if ``n_k`` is too small for the requested ``w`` span; see the
        aliasing note below.

    Notes
    -----
    ``w_grid`` must be uniformly spaced; the ``k`` span is derived from its step.

    The ``k`` sum is a Riemann sum of spacing ``dk = 2 pi / (dw n_k)``, so the
    density it returns is the true one periodized with period ``2 pi / dk =
    dw n_k``. The ``w`` span must therefore fit inside one period, which for a
    grid starting near zero means ``w_grid.size < n_k``. Violating it does not
    fail loudly on its own -- the result is still positive and still normalizes
    to one, it is simply the wrong density -- so it is checked here instead.
    """
    w = np.asarray(w_grid, dtype=float)
    step = w[1] - w[0]
    alias_period = step * n_k
    span = w[-1] - w[0]
    if alias_period <= span:
        raise ValueError(
            f"n_k = {n_k} aliases the inversion over this w grid: the period "
            f"dw * n_k = {alias_period:.3g} does not cover the span {span:.3g}. "
            f"Raise n_k above {int(np.ceil(span / step))} or coarsen w_grid."
        )
    k_max = np.pi / step
    k = np.linspace(-k_max, k_max, n_k, endpoint=False)
    cf = np.exp(-ell_km * np.asarray(symbol(-1j * k)))

    density = np.empty(w.shape[0])
    # Chunk over w to bound the memory of the outer product cf(k) * exp(-i w k).
    for i in range(0, w.shape[0], 200):
        w_chunk = w[i : i + 200][:, None]
        density[i : i + 200] = (cf[None, :] * np.exp(-1j * w_chunk * k[None, :])).sum(1).real
    density *= (k[1] - k[0]) / (2.0 * np.pi)

    density = np.clip(density, 0.0, None)
    return density / np.trapezoid(density, w)


def loss_density_three_moment(
    w_grid: np.ndarray,
    ell_km: float,
    b_mu: float,
    d_mu: float,
    t_mu: float,
    n_k: int = 2**14,
) -> np.ndarray:
    """Density ``P(w)`` of the log-loss from the three-moment loss family.

    Same inversion as :func:`loss_density`, but with the loss spectrum calibrated
    to three ``y``-moments instead of two
    (:func:`softpaws.transport.eigenvalue.three_moment_loss_spectrum`). The third
    moment is what fixes the hard end of ``dGamma/dy``, and with it the tail of
    ``P(w)``: benchmarked against a PROPOSAL Monte Carlo this tracks the tail to
    tens of percent out to ``eps / E ~ 50``, where the two-moment form is a factor
    70 low and the Fokker-Planck Gaussian is astronomically off
    (``examples/27_proposal_cross_section_and_loss.py``).

    Parameters
    ----------
    w_grid : np.ndarray
        Uniformly spaced grid of log-loss values ``w = ln(eps / E)`` (>= 0).
    ell_km : float
        Propagated column depth ``ell`` [km].
    b_mu, d_mu, t_mu : float
        First three ``y``-moments of the loss spectrum [km^-1] (see
        :mod:`softpaws.transport.coefficients`).
    n_k : int, optional
        Number of nodes on the ``k`` grid of the inversion.

    Returns
    -------
    density : np.ndarray
        Probability density ``P(w)``, normalized to unit area over ``w_grid``.
    """
    kappa, q, p = three_moment_loss_spectrum(b_mu, d_mu, t_mu)
    return invert_log_loss_symbol(
        w_grid,
        ell_km,
        lambda s: phi_symbol_three_moment(s, float(kappa), float(q), float(p)),
        n_k,
    )


def loss_density(
    w_grid: np.ndarray,
    ell_km: float,
    b_mu: float,
    d_mu: float,
    n_k: int = 2**14,
) -> np.ndarray:
    r"""Exact density ``P(w)`` of the log-loss, by characteristic-function inversion.

    The characteristic function of the subordinator is
    ``phi_w(k) = exp(-ell Phi(-i k))``, so

    .. math:: P(w) = \\frac{1}{2\\pi}\\int \\phi_w(k)\\,e^{-i k w}\\,dk.

    The integral is evaluated deterministically on a uniform ``k`` grid whose
    span is fixed by the ``w`` spacing (``k_max = pi / dw``), which resolves the
    far tail without Monte-Carlo noise. The result is clipped at zero and
    renormalized to unit area over ``w_grid``.

    Parameters
    ----------
    w_grid : np.ndarray
        Uniformly spaced grid of log-loss values ``w = ln(eps / E)`` (>= 0) at
        which to evaluate the density.
    ell_km : float
        Propagated column depth ``ell`` [km].
    b_mu : float
        Total drift coefficient ``b_mu`` [km^-1] (see
        :func:`softpaws.transport.coefficients.drift_coefficient`).
    d_mu : float
        Total diffusion coefficient ``d_mu`` [km^-1]. Must satisfy
        ``0 < d_mu < b_mu`` for the two-moment family to be defined.
    n_k : int, optional
        Number of nodes on the ``k`` grid of the inversion.

    Returns
    -------
    density : np.ndarray
        Probability density ``P(w)`` [dimensionless per unit ``w``], normalized to
        unit area over ``w_grid``, same shape as ``w_grid``.

    Notes
    -----
    ``w_grid`` must be uniformly spaced; the ``k`` span is derived from its step.
    """
    kappa, p = two_moment_loss_spectrum(b_mu, d_mu)
    return invert_log_loss_symbol(
        w_grid, ell_km, lambda s: phi_symbol(s, kappa, p), n_k
    )


def loss_density_gaussian(
    w_grid: np.ndarray,
    ell_km: float,
    b_mu: float,
    d_mu: float,
) -> np.ndarray:
    """Fokker-Planck (Gaussian) reference density for the log-loss.

    The drift-diffusion truncation of Palmisano et al. makes ``w`` normal with the same first
    two moments as the exact process: mean ``M ell`` and variance ``d_mu ell``,
    where ``M = b_mu + d_mu / 2``. This is the density that :func:`loss_density`
    supersedes; it matches the exact law near the peak but decays far too fast in
    the tail.

    Parameters
    ----------
    w_grid : np.ndarray
        Grid of log-loss values ``w`` at which to evaluate the density.
    ell_km : float
        Propagated column depth ``ell`` [km].
    b_mu : float
        Total drift coefficient ``b_mu`` [km^-1].
    d_mu : float
        Total diffusion coefficient ``d_mu`` [km^-1].

    Returns
    -------
    density : np.ndarray
        Gaussian density, same shape as ``w_grid``.
    """
    w = np.asarray(w_grid, dtype=float)
    mean = (b_mu + d_mu / 2.0) * ell_km
    var = d_mu * ell_km
    return np.exp(-((w - mean) ** 2) / (2.0 * var)) / np.sqrt(2.0 * np.pi * var)


def gaussian_survival(
    w: float | np.ndarray,
    ell_km: float,
    b_mu: float,
    d_mu: float,
) -> np.ndarray:
    """Fokker-Planck tail probability ``P(W > w)``.

    Survival function of the Gaussian reference of
    :func:`loss_density_gaussian`, ``0.5 erfc((w - M ell) / sqrt(2 d_mu ell))``.

    Parameters
    ----------
    w : float or np.ndarray
        Log-loss threshold(s) ``w = ln(eps / E)``.
    ell_km : float
        Propagated column depth ``ell`` [km].
    b_mu : float
        Total drift coefficient ``b_mu`` [km^-1].
    d_mu : float
        Total diffusion coefficient ``d_mu`` [km^-1].

    Returns
    -------
    survival : np.ndarray
        Gaussian ``P(W > w)``, same shape as ``w``.
    """
    w = np.asarray(w, dtype=float)
    mean = (b_mu + d_mu / 2.0) * ell_km
    z = (w - mean) / np.sqrt(d_mu * ell_km)
    return 0.5 * erfc(z / np.sqrt(2.0))


def survival_from_density(
    w_query: float | np.ndarray,
    w_grid: np.ndarray,
    density: np.ndarray,
) -> np.ndarray:
    """Tail probability ``P(W > w)`` from a tabulated density.

    Integrates ``density`` over ``w_grid`` with the trapezoidal rule and returns
    ``1 - CDF`` at ``w_query`` by linear interpolation. Used to read the exact
    tail off :func:`loss_density`.

    Parameters
    ----------
    w_query : float or np.ndarray
        Log-loss threshold(s) at which to evaluate the survival probability.
    w_grid : np.ndarray
        Grid on which ``density`` is tabulated (ascending).
    density : np.ndarray
        Probability density ``P(w)`` on ``w_grid``.

    Returns
    -------
    survival : np.ndarray
        ``P(W > w_query)``, same shape as ``w_query``.
    """
    w = np.asarray(w_grid, dtype=float)
    dens = np.asarray(density, dtype=float)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(w))])
    return 1.0 - np.interp(w_query, w, cdf)


def log_loss_cdf(
    w_query: float,
    ell_km: float | np.ndarray,
    b_mu: float,
    d_mu: float,
    k_max: float = 1.0e3,
    n_k: int = 400_000,
    chunk: int = _GIL_PELAEZ_CHUNK,
) -> np.ndarray:
    r"""Probability ``P(W <= w_query)`` at each propagated depth, by Gil-Pelaez.

    Where :func:`loss_density` inverts the characteristic function onto a whole
    ``w`` grid and :func:`survival_from_density` then integrates it, this goes
    straight to the cumulative probability at a *single* ``w``, using the
    Gil-Pelaez inversion formula

    .. math:: F(w) = \\frac{1}{2} - \\frac{1}{\\pi}\\int_0^\\infty
        \\frac{\\Im\\bigl[e^{-i k w}\\,\\varphi(k)\\bigr]}{k}\\,dk,
        \\qquad \\varphi(k) = e^{-\\ell\\,\\Phi(-i k)}.

    That is what the per-neutrino-energy target volume needs
    (:func:`softpaws.transport.muon_range.stochastic_muon_range_km`): a muon
    born at ``eps`` is still above an analysis threshold ``E_thr`` after
    propagating ``ell`` exactly when ``W < ln(eps / E_thr)``, one threshold per
    energy but every depth along the column.

    Parameters
    ----------
    w_query : float
        Log-loss threshold ``w = ln(eps / E_thr)``.
    ell_km : float or np.ndarray
        Propagated column depth(s) ``ell`` [km]. Vectorized: the whole depth
        integral is one call.
    b_mu : float
        Drift coefficient ``b_mu`` [km^-1].
    d_mu : float
        Diffusion coefficient ``d_mu`` [km^-1]. Must satisfy ``0 < d_mu < b_mu``
        for the two-moment family to be defined.
    k_max : float, optional
        Upper limit of the ``k`` quadrature. The symbol grows only
        logarithmically, so the characteristic function decays as a power law
        ``k^{-ell kappa}`` and the tail matters most at small ``ell``.
    n_k : int, optional
        Number of quadrature nodes on ``(0, k_max]``.
    chunk : int, optional
        Number of ``k`` nodes held in memory at once. The full outer product
        would be ``n_ell * n_k`` complex numbers -- gigabytes for a fine grid --
        so the quadrature is accumulated in slices instead.

    Returns
    -------
    cdf : np.ndarray
        ``P(W <= w_query)`` clipped to ``[0, 1]``, one entry per ``ell_km``.

    Notes
    -----
    Agrees with :func:`loss_density` plus :func:`survival_from_density` to a few
    times ``1e-3`` absolute; the residual is the ``k``-truncation both share.
    Unlike that route it needs no ``w`` grid, so it neither aliases nor forces a
    span/resolution trade-off.
    """
    kappa, p = two_moment_loss_spectrum(b_mu, d_mu)
    ell = np.atleast_1d(np.asarray(ell_km, dtype=float))
    w = float(w_query)
    step = k_max / n_k
    # Trapezoid over k in [0, k_max]: accumulate every interior node at full
    # weight, then halve the two endpoints. The k -> 0 endpoint is removable --
    # Phi(-ik) -> -i k Phi'(0) makes the integrand tend to ell Phi'(0) - w, with
    # Phi'(0) = kappa psi'(p + 1) the mean log-loss per unit length.
    total = ell * (kappa * polygamma(1, p + 1.0)) - w
    total *= 0.5
    for start in range(1, n_k + 1, chunk):
        k = np.arange(start, min(start + chunk, n_k + 1), dtype=float) * step
        symbol = np.asarray(phi_symbol(-1j * k, kappa, p))
        # exp(-ell Phi(-ik)) e^{-i k w} / k, summed over this k slice only.
        phase = np.exp(-1j * k * w) / k
        block = np.exp(-ell[:, None] * symbol[None, :]) * phase[None, :]
        weights = np.ones(k.shape[0])
        if k[-1] >= k_max:
            weights[-1] = 0.5
        total += (np.imag(block) * weights[None, :]).sum(axis=1)
    return np.clip(0.5 - total * step / np.pi, 0.0, 1.0)
