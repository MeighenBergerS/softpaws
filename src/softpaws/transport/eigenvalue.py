"""Exact eigenvalue of the QED collision operator.

Where :mod:`softpaws.transport.coefficients` provides the drift ``b_mu`` and
diffusion ``d_mu`` of the paper's Fokker-Planck expansion, this module provides
the *exact* attenuation constant that supersedes them.

The QED energy-loss operator is scale-invariant, so power laws ``phi ~ E^-1-s``
are its eigenfunctions with eigenvalue (``docs/exact_soft_volume_notes.md``, and
``docs/2026_softvolume.pdf`` Part 2)

.. math:: \\Phi(s) = \\int_0^1 dy\\, \\frac{d\\Gamma}{dy}
    \\bigl[\\,1 - (1 - y)^s\\,\\bigr].

A single power-law source excites exactly one mode ``s = A = gamma - lambda - 1``,
so the whole transport problem collapses to multiplication by the one number
``Phi(A)``. The bracket ``[1 - (1-y)^s]`` is IR- and UV-finite with no cutoffs:
the drift-diffusion moments ``b_mu``, ``d_mu`` are just the first two terms of its
binomial expansion, ``Phi(A) = A b_mu - A(A-1)/2 d_mu + ...`` (Part 10.1), and the
series terminates at integer ``A``, giving the exactness identities
``Phi(1) = b_mu`` and ``Phi(2) = 2 b_mu - d_mu`` (Part 10.2).

Table 1 of the paper fixes only ``b_mu`` and ``d_mu``, so the primary evaluation
here calibrates a two-parameter loss spectrum to those two moments and evaluates
``Phi(A)`` in closed form. It is exact at ``A = 1, 2`` by construction and degrades
for ``A >~ 3`` (higher moments); the definitive version is a direct quadrature of
``Phi(A)`` against tabulated differential cross sections, provided by
:func:`phi_eigenvalue_quadrature`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.special import digamma

from ..utils.constants import RHO_WATER_G_CM3
from .coefficients import diffusion_coefficient, drift_coefficient
from .source import DEFAULT_LAMBDA


def spectral_index(gamma: float, lam: float = DEFAULT_LAMBDA) -> float:
    """Source spectral index ``A = gamma - lambda - 1``.

    Unlike :func:`softpaws.transport.soft_volume.spectral_penalty`, this does not
    require ``A > 0``: the exact treatment stays finite for ``A <= 0`` through the
    saturation factor, so the sign is left for the caller to interpret (``A < 0``
    signals the cross-section pole, ``docs/2026_softvolume.pdf`` Part 11.2).

    Parameters
    ----------
    gamma : float
        Neutrino flux spectral index, ``phi_nu ~ E^-gamma``.
    lam : float, optional
        CC cross-section slope, ``sigma_CC ~ E^lambda``. Defaults to
        :data:`softpaws.transport.source.DEFAULT_LAMBDA`.

    Returns
    -------
    A : float
        Spectral index ``gamma - lambda - 1``.
    """
    return gamma - lam - 1.0


def two_moment_loss_spectrum(
    b_mu: float | np.ndarray,
    d_mu: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calibrate the two-parameter loss spectrum ``dGamma/dy = kappa (1-y)^p / y``.

    The family is fixed by matching its first two ``y``-moments to the drift and
    diffusion coefficients (``docs/2026_softvolume.pdf`` Part 10.4):

    .. math:: b_\\mu = \\frac{\\kappa}{p+1}, \\qquad
        d_\\mu = \\frac{\\kappa}{(p+1)(p+2)}
        \\;\\Longrightarrow\\; \\frac{d_\\mu}{b_\\mu} = \\frac{1}{p+2}.

    Parameters
    ----------
    b_mu : float or np.ndarray
        Drift coefficient [km^-1].
    d_mu : float or np.ndarray
        Diffusion coefficient [km^-1]. Must satisfy ``0 < d_mu < b_mu`` (i.e.
        ``p > -1``) for the shape parameter to be well defined.

    Returns
    -------
    kappa : np.ndarray
        Overall normalization [km^-1].
    p : np.ndarray
        Shape exponent of the ``(1 - y)^p`` softening.
    """
    b = np.asarray(b_mu, dtype=float)
    d = np.asarray(d_mu, dtype=float)
    p = b / d - 2.0
    kappa = b * (p + 1.0)
    return kappa, p


def phi_eigenvalue(
    spectral_index_value: float | np.ndarray,
    b_mu: float | np.ndarray,
    d_mu: float | np.ndarray,
) -> np.ndarray:
    """Exact collision eigenvalue ``Phi(A)`` from the two calibrated moments.

    Using the closed form of the two-moment family (Part 10.4),

    .. math:: \\Phi(A) = \\kappa\\,\\bigl[\\psi(p + A + 1) - \\psi(p + 1)\\bigr],

    with ``psi`` the digamma function. This reproduces ``Phi(1) = b_mu`` and
    ``Phi(2) = 2 b_mu - d_mu`` exactly and grows monotonically in ``A`` (unlike the
    Fokker-Planck truncation, which turns over).

    Parameters
    ----------
    spectral_index_value : float or np.ndarray
        Source spectral index ``A`` (see :func:`spectral_index`).
    b_mu : float or np.ndarray
        Drift coefficient [km^-1].
    d_mu : float or np.ndarray
        Diffusion coefficient [km^-1]. If non-positive, the drift limit
        ``Phi = A b_mu`` is returned.

    Returns
    -------
    phi : np.ndarray
        Eigenvalue ``Phi(A)`` [km^-1].

    Notes
    -----
    Exact only at ``A = 1, 2``; other ``A`` inherit whatever the two-moment family
    gets wrong about third and higher moments. For a definitive value pass
    tabulated differential cross sections to :func:`phi_eigenvalue_quadrature`.
    """
    a = np.asarray(spectral_index_value, dtype=float)
    b = np.asarray(b_mu, dtype=float)
    d = np.asarray(d_mu, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa, p = two_moment_loss_spectrum(b, d)
        phi = kappa * (digamma(p + a + 1.0) - digamma(p + 1.0))
    return np.where(d > 0.0, phi, a * b)


def phi_drift(
    spectral_index_value: float | np.ndarray,
    b_mu: float | np.ndarray,
) -> np.ndarray:
    """Leading (drift-only) eigenvalue ``Phi ~ A b_mu``.

    First term of the binomial expansion of ``Phi(A)`` (Part 10.1); this is the
    approximation used by the drift-limit soft volume in
    :func:`softpaws.transport.soft_volume.soft_volume_drift`.

    Parameters
    ----------
    spectral_index_value : float or np.ndarray
        Source spectral index ``A``.
    b_mu : float or np.ndarray
        Drift coefficient [km^-1].

    Returns
    -------
    phi : np.ndarray
        ``A b_mu`` [km^-1].
    """
    a = np.asarray(spectral_index_value, dtype=float)
    return a * np.asarray(b_mu, dtype=float)


def phi_fokker_planck(
    spectral_index_value: float | np.ndarray,
    b_mu: float | np.ndarray,
    d_mu: float | np.ndarray,
) -> np.ndarray:
    """Second-order (Fokker-Planck) eigenvalue ``A b_mu - A(A-1)/2 d_mu``.

    The paper's truncation of ``Phi(A)`` (Part 10.1), kept here for head-to-head
    comparison with the exact :func:`phi_eigenvalue`. It agrees with the exact
    value to well under a percent near ``A = 1`` (where the IceCube spectrum sits)
    and diverges from it for ``A >~ 3``.

    Parameters
    ----------
    spectral_index_value : float or np.ndarray
        Source spectral index ``A``.
    b_mu : float or np.ndarray
        Drift coefficient [km^-1].
    d_mu : float or np.ndarray
        Diffusion coefficient [km^-1].

    Returns
    -------
    phi : np.ndarray
        Truncated eigenvalue [km^-1].
    """
    a = np.asarray(spectral_index_value, dtype=float)
    b = np.asarray(b_mu, dtype=float)
    d = np.asarray(d_mu, dtype=float)
    return a * b - 0.5 * a * (a - 1.0) * d


def phi_eigenvalue_quadrature(
    spectral_index_value: float | np.ndarray,
    dgamma_dy: Callable[[np.ndarray], np.ndarray],
    n_points: int = 8192,
) -> np.ndarray:
    """Exact eigenvalue by direct quadrature of ``Phi(A)``.

    Evaluates ``Phi(A) = int_0^1 dy (dGamma/dy) [1 - (1-y)^A]`` for an arbitrary
    differential loss rate. This is the definitive path (Part 10.4): pass a
    tabulated or modelled ``dGamma/dy`` (e.g. from PROPOSAL) and no expansion in
    ``y`` or energy cutoff is needed. The integrand is finite as ``y -> 0`` even
    for a ``1/y`` bremsstrahlung tail, since ``1 - (1-y)^A -> A y``.

    Parameters
    ----------
    spectral_index_value : float or np.ndarray
        Source spectral index ``A`` (scalar or array).
    dgamma_dy : callable
        Differential loss rate ``dGamma/dy`` [km^-1] as a function of the
        fractional energy loss ``y`` in ``(0, 1)``; must accept and return arrays.
    n_points : int, optional
        Number of quadrature nodes on ``(0, 1)``.

    Returns
    -------
    phi : np.ndarray
        Eigenvalue ``Phi(A)`` [km^-1], broadcast to the shape of
        ``spectral_index_value``.
    """
    a = np.atleast_1d(np.asarray(spectral_index_value, dtype=float))
    # Nodes on the open interval; the endpoints contribute nothing (the y -> 0
    # divergence of dGamma/dy is cancelled by the bracket, y -> 1 gives (1-y)^A -> 0).
    y = np.linspace(0.0, 1.0, n_points + 2)[1:-1]
    g = np.asarray(dgamma_dy(y), dtype=float)
    bracket = 1.0 - (1.0 - y)[None, :] ** a[:, None]
    phi = np.trapezoid(g[None, :] * bracket, y, axis=1)
    return phi.reshape(np.shape(spectral_index_value))


def phi_eigenvalue_at_energy(
    spectral_index_value: float | np.ndarray,
    energy_gev: float | np.ndarray,
    density_g_cm3: float = RHO_WATER_G_CM3,
) -> np.ndarray:
    """Exact eigenvalue at a muon energy, using the Table 1 coefficients.

    Convenience wrapper that reads ``b_mu(E)``, ``d_mu(E)`` from
    :mod:`softpaws.transport.coefficients` and feeds them to
    :func:`phi_eigenvalue`.

    Parameters
    ----------
    spectral_index_value : float or np.ndarray
        Source spectral index ``A``.
    energy_gev : float or np.ndarray
        Muon energy [GeV].
    density_g_cm3 : float, optional
        Target-medium density [g cm^-3]. Defaults to water.

    Returns
    -------
    phi : np.ndarray
        Eigenvalue ``Phi(A)`` [km^-1].
    """
    b_mu = drift_coefficient(energy_gev, density_g_cm3)
    d_mu = diffusion_coefficient(energy_gev, density_g_cm3)
    return phi_eigenvalue(spectral_index_value, b_mu, d_mu)
