"""Full distribution of the accumulated muon log-energy loss.

Where :mod:`softpaws.transport.eigenvalue` gives the single eigenvalue ``Phi(A)``
that a power-law flux excites, this module gives the whole probability law behind
it. A muon observed at energy ``E`` that was produced at energy ``eps`` after
propagating a column ``ell`` has accumulated a log-energy loss

.. math:: w = \\ln(\\varepsilon / E) \\ge 0,

which, under the scale-invariant QED loss rates, is a *subordinator* (a
non-decreasing Levy jump process) whose Laplace exponent is exactly the Mellin
symbol of the collision operator (``docs/exact_soft_volume_notes.md``):

.. math:: \\mathbb{E}\\bigl[e^{-s\\,w(\\ell)}\\bigr] = e^{-\\ell\\,\\Phi(s)}.

A power-law flux ``phi ~ E^{-1-A}`` averages over ``w`` and so only ever sees the
one number ``Phi(A)`` -- the regime of examples 05-08 and 12. A **single** event
(such as KM3-230213A) instead samples ``P(w)`` once, and the heavy tail of that
law -- the chance the parent neutrino was far more energetic than the observed
muon -- is what the paper's Fokker-Planck Gaussian throws away.

This module inverts ``e^{-ell Phi(-i k)}`` to recover ``P(w)`` exactly
(:func:`loss_density`), alongside the Fokker-Planck Gaussian reference
(:func:`loss_density_gaussian`, :func:`gaussian_survival`) it is meant to
supersede. It reuses the same two-moment loss family as the eigenvalue path, via
the complex-capable :func:`softpaws.transport.eigenvalue.phi_symbol`.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfc

from .eigenvalue import phi_symbol, two_moment_loss_spectrum


def loss_density(
    w_grid: np.ndarray,
    ell_km: float,
    b_mu: float,
    d_mu: float,
    n_k: int = 2**14,
) -> np.ndarray:
    """Exact density ``P(w)`` of the log-loss, by characteristic-function inversion.

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
    w = np.asarray(w_grid, dtype=float)
    kappa, p = two_moment_loss_spectrum(b_mu, d_mu)

    dw = w[1] - w[0]
    k_max = np.pi / dw
    k = np.linspace(-k_max, k_max, n_k, endpoint=False)
    dk = k[1] - k[0]
    cf = np.exp(-ell_km * phi_symbol(-1j * k, kappa, p))

    density = np.empty(w.shape[0])
    # Chunk over w to bound the memory of the outer product cf(k) * exp(-i w k).
    for i in range(0, w.shape[0], 200):
        w_chunk = w[i : i + 200][:, None]
        density[i : i + 200] = (cf[None, :] * np.exp(-1j * w_chunk * k[None, :])).sum(1).real
    density *= dk / (2.0 * np.pi)

    density = np.clip(density, 0.0, None)
    return density / np.trapezoid(density, w)


def loss_density_gaussian(
    w_grid: np.ndarray,
    ell_km: float,
    b_mu: float,
    d_mu: float,
) -> np.ndarray:
    """Fokker-Planck (Gaussian) reference density for the log-loss.

    The paper's drift-diffusion truncation makes ``w`` normal with the same first
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
