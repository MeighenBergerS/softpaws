"""The transport exponent: eigenvalue of the QED collision operator.

Where :mod:`softpaws.transport.coefficients` provides the drift ``b_mu`` and
diffusion ``d_mu`` of the earlier Fokker-Planck expansion (Palmisano et al.,
arXiv:2607.13143), this module provides the attenuation constant that
supersedes them, with every loss moment kept.

The QED energy-loss operator is scale-invariant, so power laws ``phi ~ E^-1-s``
are its eigenfunctions with eigenvalue (``docs/theory/exact_soft_volume.md``, and
``paper/main.tex`` Part 2)

.. math:: \\Phi(s) = \\int_0^1 dy\\, \\frac{d\\Gamma}{dy}
    \\bigl[\\,1 - (1 - y)^s\\,\\bigr].

A single power-law source excites exactly one mode ``s = A = gamma - lambda - 1``,
so the whole transport problem collapses to multiplication by the one number
``Phi(A)``. The bracket ``[1 - (1-y)^s]`` is IR- and UV-finite with no cutoffs:
the drift-diffusion moments ``b_mu``, ``d_mu`` are just the first two terms of its
binomial expansion, ``Phi(A) = A b_mu - A(A-1)/2 d_mu + ...`` (Part 10.1), and the
series terminates at integer ``A``, giving the exactness identities
``Phi(1) = b_mu`` and ``Phi(2) = 2 b_mu - d_mu`` (Part 10.2).

Table 1 of Palmisano et al. fixes only ``b_mu`` and ``d_mu``, so the primary evaluation
here calibrates a two-parameter loss spectrum to those two moments and evaluates
``Phi(A)`` in closed form. It is exact at ``A = 1, 2`` by construction and degrades
for ``A >~ 3`` (higher moments); the definitive version is a direct quadrature of
``Phi(A)`` against tabulated differential cross sections, provided by
:func:`phi_eigenvalue_quadrature`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.special import digamma, gamma, loggamma, polygamma

from ..utils.constants import RHO_WATER_G_CM3
from .coefficients import DEFAULT_SOURCE, diffusion_coefficient, drift_coefficient
from .source import DEFAULT_LAMBDA


def spectral_index(gamma: float, lam: float = DEFAULT_LAMBDA) -> float:
    """Source spectral index ``A = gamma - lambda - 1``.

    Unlike :func:`softpaws.transport.soft_volume.spectral_penalty`, this does not
    require ``A > 0``: the exact treatment stays finite for ``A <= 0`` through the
    saturation factor, so the sign is left for the caller to interpret (``A < 0``
    signals the cross-section pole, ``paper/main.tex`` Part 11.2).

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
    diffusion coefficients (``paper/main.tex`` Part 10.4):

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


def three_moment_loss_spectrum(
    b_mu: float | np.ndarray,
    d_mu: float | np.ndarray,
    t_mu: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calibrate the loss spectrum ``dGamma/dy = kappa y^(q-1) (1-y)^p`` to three moments.

    One-parameter generalization of :func:`two_moment_loss_spectrum`, which is the
    ``q = 0`` member of this family. Freeing the soft exponent ``q`` lets the third
    ``y``-moment be matched as well, and the extra freedom lands almost entirely on
    the hard end: calibrated to PROPOSAL, ``p`` moves from about ``+2.1`` to about
    ``-0.17``, so ``dGamma/dy`` no longer vanishes as ``y -> 1`` and the rare
    catastrophic losses that dominate the fluctuations survive. See
    ``examples/27_proposal_cross_section_and_loss.py``.

    The moments of the family are Beta functions, ``<y^n> = kappa B(n + q, p + 1)``,
    so the two ratios

    .. math:: r_1 = \\frac{d_\\mu}{b_\\mu} = \\frac{1 + q}{2 + q + p}, \\qquad
        r_2 = \\frac{t_\\mu}{d_\\mu} = \\frac{2 + q}{3 + q + p}

    invert in closed form.

    Parameters
    ----------
    b_mu : float or np.ndarray
        First moment ``<y>``, the drift coefficient [km^-1].
    d_mu : float or np.ndarray
        Second moment ``<y^2>``, the diffusion coefficient [km^-1].
    t_mu : float or np.ndarray
        Third moment ``<y^3>`` [km^-1] (see
        :func:`softpaws.transport.coefficients.third_moment_coefficient`).

    Returns
    -------
    kappa : np.ndarray
        Overall normalization [km^-1].
    q : np.ndarray
        Soft exponent, so that ``dGamma/dy ~ y^(q-1)`` as ``y -> 0``. Negative for
        real loss spectra, whose soft pile-up is steeper than ``1/y``.
    p : np.ndarray
        Shape exponent of the ``(1 - y)^p`` hard-end softening.

    Raises
    ------
    ValueError
        Raised if the moments are not those of a positive spectrum on ``(0, 1)``,
        i.e. if the solution leaves ``q > -1``, ``p > -1``.

    Notes
    -----
    Moment log-convexity guarantees ``r_2 >= r_1`` for any positive spectrum, so
    the denominator ``r_2 - r_1`` is non-negative; it vanishes only in the
    degenerate single-jump-size limit.
    """
    b = np.asarray(b_mu, dtype=float)
    d = np.asarray(d_mu, dtype=float)
    t = np.asarray(t_mu, dtype=float)
    r_1 = d / b
    r_2 = t / d
    q = (2.0 * r_1 - r_2 - r_1 * r_2) / (r_2 - r_1)
    p = (1.0 + q) / r_1 - 2.0 - q
    if np.any(q <= -1.0) or np.any(p <= -1.0):
        raise ValueError(
            "three-moment calibration left the convergent domain (need q > -1 and "
            f"p > -1, got q = {q}, p = {p}); check that t_mu is the third moment "
            "of the same spectrum as b_mu and d_mu."
        )
    kappa = b / _beta(1.0 + q, p + 1.0)
    return kappa, q, p


def _beta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``B(a, b)`` by its Gamma-function definition, valid for non-integer ``a < 0``.

    ``scipy.special.beta`` returns ``inf`` for negative arguments; the analytic
    continuation is what the three-moment family needs, since the individual Beta
    functions of :func:`phi_symbol_three_moment` are evaluated at ``q < 0`` even
    though their difference is finite.
    """
    return np.asarray(gamma(a) * gamma(b) / gamma(a + b))


def phi_symbol(
    s: complex | np.ndarray,
    kappa: float | np.ndarray,
    p: float | np.ndarray,
) -> np.ndarray:
    """Mellin symbol ``Phi(s)`` of the two-moment loss family, for any ``s``.

    Closed form of the family ``dGamma/dy = kappa (1-y)^p / y`` (Part 10.4),

    .. math:: \\Phi(s) = \\kappa\\,\\bigl[\\psi(s + p + 1) - \\psi(p + 1)\\bigr],

    with ``psi`` the digamma function. Unlike :func:`phi_eigenvalue`, this keeps
    the argument's dtype, so it accepts **complex** ``s``. That is what the
    characteristic function of the log-loss subordinator needs
    (:func:`softpaws.transport.loss_distribution.loss_density`, evaluated at
    ``s = -i k``); for the real spectral index ``A`` use :func:`phi_eigenvalue`.

    Parameters
    ----------
    s : complex or np.ndarray
        Mellin variable. Real or complex; the dtype is preserved.
    kappa : float or np.ndarray
        Loss-spectrum normalization ``kappa`` [km^-1] (see
        :func:`two_moment_loss_spectrum`).
    p : float or np.ndarray
        Shape exponent ``p`` of the ``(1 - y)^p`` softening.

    Returns
    -------
    phi : np.ndarray
        Symbol ``Phi(s)`` [km^-1], matching the (broadcast) shape and dtype of
        ``s``.
    """
    s = np.asarray(s)
    kappa = np.asarray(kappa, dtype=float)
    p = np.asarray(p, dtype=float)
    return kappa * (digamma(s + p + 1.0) - digamma(p + 1.0))


# Below this |q| the two Beta functions of the three-moment symbol cancel to
# working precision and the q -> 0 digamma limit is used instead. The limit is
# accurate to O(q), so the switch costs at most ~1e-6 relative.
_Q_DIGAMMA_LIMIT = 1.0e-6


def phi_symbol_three_moment(
    s: complex | np.ndarray,
    kappa: float,
    q: float,
    p: float,
) -> np.ndarray:
    """Mellin symbol ``Phi(s)`` of the three-moment loss family, for any ``s``.

    Closed form of ``dGamma/dy = kappa y^(q-1) (1-y)^p``,

    .. math:: \\Phi(s) = \\kappa\\,\\Gamma(q)\\,\\left[
        \\frac{\\Gamma(p+1)}{\\Gamma(q+p+1)}
        - \\frac{\\Gamma(p+s+1)}{\\Gamma(q+p+s+1)}\\right],

    which reduces to :func:`phi_symbol` as ``q -> 0``. Both Beta functions
    diverge at ``q = 0`` while their difference stays finite, so the ``q -> 0``
    digamma limit is substituted below :data:`_Q_DIGAMMA_LIMIT`.

    Like :func:`phi_symbol` this accepts **complex** ``s``, which is what the
    characteristic function of
    :func:`softpaws.transport.loss_distribution.loss_density_three_moment` needs.
    The Gamma ratios are formed through ``loggamma``: ``Gamma(p + s + 1)``
    overflows for large ``|Im s|``, whereas the ratio grows only like ``s^-q``.

    Parameters
    ----------
    s : complex or np.ndarray
        Mellin variable. Real or complex; the dtype is preserved.
    kappa : float
        Loss-spectrum normalization [km^-1] (see
        :func:`three_moment_loss_spectrum`).
    q : float
        Soft exponent, ``> -1``.
    p : float
        Hard-end exponent, ``> -1``.

    Returns
    -------
    phi : np.ndarray
        Symbol ``Phi(s)`` [km^-1], matching the shape and dtype of ``s``.
    """
    s = np.asarray(s)
    if abs(float(q)) < _Q_DIGAMMA_LIMIT:
        return phi_symbol(s, kappa, p)
    # exp(loggamma(a) - loggamma(b)) rather than gamma(a) / gamma(b): the latter
    # is inf / inf for the large imaginary arguments of the inversion. The
    # arguments are cast to complex because q + p + 1 may be negative, where the
    # real branch of loggamma is undefined but the complex one carries the sign.
    complex_s = s.astype(complex)
    reference = np.exp(loggamma(complex(p + 1.0)) - loggamma(complex(q + p + 1.0)))
    shifted = np.exp(
        loggamma(complex_s + (p + 1.0)) - loggamma(complex_s + (q + p + 1.0))
    )
    phi = kappa * gamma(q) * (reference - shifted)
    return phi if np.iscomplexobj(s) else np.real(phi)


def n_moment_coefficients(
    mu: np.ndarray,
    q: float,
    p: float,
    dps: int = 80,
) -> np.ndarray:
    """Calibrate the ``N``-moment loss family to the first ``N`` moments.

    Generalization of :func:`two_moment_loss_spectrum` and
    :func:`three_moment_loss_spectrum` to arbitrary order. The family is
    ``dGamma/dy = y^(q-1) (1-y)^p sum_j a_j y^j``, whose moments are Beta
    functions, ``<y^n> = sum_j a_j B(n + q + j, p + 1)``, so matching
    ``mu_1 ... mu_N`` is a dense ``N x N`` linear system in the ``a_j`` and the
    exponent stays in closed form (:func:`phi_symbol_n_moment`). The base
    exponents ``(q, p)`` are held fixed, normally at their three-moment
    calibration.

    The matrix is a Beta-function Hankel matrix, and reconstructing a density
    from its moments on ``(0, 1)`` is the classical ill-posed Hausdorff problem:
    the condition number grows about three decades per added moment, from
    ``1e4`` at ``N = 4`` to ``1e13`` at ``N = 10``. The solve is therefore done
    in extended precision with ``mpmath`` and only the result is returned in
    double precision. Note what is and is not well posed here: the *exponent*
    converges geometrically in ``N`` while the reconstructed *shape* does not
    converge at all, because both the moments and ``Phi`` are blind to the soft
    region that carries most of the collisions. See
    ``examples/37_moment_convergence.py``.

    Parameters
    ----------
    mu : np.ndarray
        Moments ``mu_1 ... mu_N`` [km^-1], setting ``N = len(mu)``. The first two
        are the drift and diffusion coefficients ``b_mu`` and ``d_mu``.
    q : float
        Soft exponent of the base weight, so ``dGamma/dy ~ y^(q-1)`` as
        ``y -> 0``. Negative for real loss spectra.
    p : float
        Hard exponent of the base weight, the ``(1 - y)^p`` softening.
    dps : int, optional
        ``mpmath`` working precision [decimal digits]. The default of ``80``
        covers ``N <= 10``.

    Returns
    -------
    a : np.ndarray
        Polynomial coefficients ``a_0 ... a_{N-1}`` [km^-1].

    Raises
    ------
    ImportError
        Raised if ``mpmath`` is not installed. Double precision is not enough
        for this solve, so there is no fallback.
    """
    try:
        import mpmath as mp
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "n_moment_coefficients needs mpmath: the moment matrix is too "
            "ill-conditioned to solve in double precision."
        ) from exc
    n = len(mu)
    with mp.workdps(dps):
        q_mp, p_mp = mp.mpf(float(q)), mp.mpf(float(p))
        matrix = mp.matrix(n, n)
        rhs = mp.matrix(n, 1)
        for i in range(n):
            for j in range(n):
                matrix[i, j] = mp.beta(mp.mpf(i + 1) + q_mp + j, p_mp + 1)
            rhs[i] = mp.mpf(float(mu[i]))
        solution = mp.lu_solve(matrix, rhs)
    return np.array([float(v) for v in solution])


def n_moment_loss_spectrum(
    a: np.ndarray,
    q: float,
    p: float,
    y: np.ndarray,
) -> np.ndarray:
    """Reconstructed ``dGamma/dy`` of the ``N``-moment loss family.

    Evaluates ``y^(q-1) (1-y)^p sum_j a_j y^j`` for the coefficients returned by
    :func:`n_moment_coefficients`.

    We caution that this reproduces the moments of the loss spectrum and not its
    shape. Against a tabulated spectrum the reconstruction is off by orders of
    magnitude below ``y ~ 1e-4`` for every ``N``, and adding moments moves the
    error around instead of reducing it. Use it to audit the calibration, and
    the tabulated spectrum or the log-augmented family of ``examples/38`` when
    the shape itself is the target.

    Parameters
    ----------
    a : np.ndarray
        Polynomial coefficients ``a_0 ... a_{N-1}`` [km^-1].
    q : float
        Soft exponent of the base weight.
    p : float
        Hard exponent of the base weight.
    y : np.ndarray
        Grid of fractional energy losses in ``(0, 1)``.

    Returns
    -------
    dgamma_dy : np.ndarray
        Differential loss rate [km^-1] on ``y``.
    """
    polynomial = np.zeros_like(np.asarray(y, dtype=float))
    for j, a_j in enumerate(a):
        polynomial = polynomial + a_j * y**j
    return y ** (q - 1.0) * (1.0 - y) ** p * polynomial


def _beta_complex(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``B(a, b)`` via complex ``loggamma``, valid for negative non-integer ``a``.

    The base soft exponent ``q`` is negative for real loss spectra, so ``q + j``
    lands on the negative real axis where the real-valued ``loggamma`` is
    undefined. Promoting to complex keeps the branch bookkeeping correct. Unlike
    :func:`_beta`, this survives the large imaginary arguments of the
    characteristic function.
    """
    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)
    return np.exp(loggamma(a) + loggamma(b) - loggamma(a + b))


def phi_symbol_n_moment(
    s: complex | np.ndarray,
    a: np.ndarray,
    q: float,
    p: float,
) -> np.ndarray:
    """Mellin symbol ``Phi(s)`` of the ``N``-moment loss family, for any ``s``.

    Term by term, ``int_0^1 dy y^(q-1+j) (1-y)^p [1 - (1-y)^s]`` is a difference
    of Beta functions, so the exponent stays closed form at every order,

    .. math:: \\Phi(s) = \\sum_j a_j
        \\bigl[\\,B(q+j,\\,p+1) - B(q+j,\\,p+1+s)\\,\\bigr].

    The ``j = 0`` term diverges as ``q -> 0`` while its difference stays finite,
    so the digamma limit of :func:`phi_symbol` is substituted there, as in
    :func:`phi_symbol_three_moment`. Like the two- and three-moment symbols this
    accepts **complex** ``s``, which the log-loss inversion needs.

    Calibrated to PROPOSAL at ``1`` PeV in water, the error at ``A = 8`` against
    a direct quadrature of the tabulated spectrum falls from ``15.7%`` at
    ``N = 2`` to ``0.52%`` at ``N = 3`` and ``0.032%`` at ``N = 4``, then gains
    roughly a factor of five per moment.

    Parameters
    ----------
    s : complex or np.ndarray
        Mellin variable. Real or complex; the dtype is preserved.
    a : np.ndarray
        Polynomial coefficients from :func:`n_moment_coefficients` [km^-1].
    q : float
        Soft exponent of the base weight.
    p : float
        Hard exponent of the base weight.

    Returns
    -------
    phi : np.ndarray
        Symbol ``Phi(s)`` [km^-1], matching the shape and dtype of ``s``.
    """
    s_array = np.asarray(s)
    total = np.zeros(s_array.shape, dtype=complex)
    for j, a_j in enumerate(a):
        if abs(q + j) < _Q_DIGAMMA_LIMIT:
            term = digamma(p + 1.0 + s_array) - digamma(p + 1.0)
        else:
            term = _beta_complex(q + j, p + 1.0) - _beta_complex(q + j, p + 1.0 + s_array)
        total = total + a_j * term
    return total if np.iscomplexobj(s_array) else total.real


def phi_eigenvalue_three_moment(
    spectral_index_value: float | np.ndarray,
    b_mu: float | np.ndarray,
    d_mu: float | np.ndarray,
    t_mu: float | np.ndarray,
) -> np.ndarray:
    """Collision eigenvalue ``Phi(A)`` from three calibrated moments.

    Three-moment counterpart of :func:`phi_eigenvalue`. Exact at ``A = 1, 2, 3``
    by construction -- the binomial expansion of ``1 - (1-y)^A`` terminates there,
    giving ``Phi(1) = b_mu``, ``Phi(2) = 2 b_mu - d_mu`` and
    ``Phi(3) = 3 b_mu - 3 d_mu + t_mu`` -- and benchmarked against PROPOSAL it
    stays within 0.5% out to ``A = 8``, where the two-moment form is 16% low
    (``examples/27_proposal_cross_section_and_loss.py``).

    Parameters
    ----------
    spectral_index_value : float or np.ndarray
        Source spectral index ``A`` (see :func:`spectral_index`).
    b_mu, d_mu, t_mu : float or np.ndarray
        First three ``y``-moments of the loss spectrum [km^-1].

    Returns
    -------
    phi : np.ndarray
        Eigenvalue ``Phi(A)`` [km^-1].
    """
    a = np.asarray(spectral_index_value, dtype=float)
    kappa, q, p = three_moment_loss_spectrum(b_mu, d_mu, t_mu)
    return phi_symbol_three_moment(a, float(kappa), float(q), float(p))


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
        phi = phi_symbol(a, kappa, p)
    return np.where(d > 0.0, phi, a * b)


def phi_eigenvalue_derivative(
    spectral_index_value: float | np.ndarray,
    b_mu: float | np.ndarray,
    d_mu: float | np.ndarray,
) -> np.ndarray:
    """Derivative ``d Phi / dA`` of the exact eigenvalue, from the trigamma function.

    Differentiating the closed form of :func:`phi_eigenvalue`,

    .. math:: \\frac{d\\Phi}{dA}(A) = \\kappa\\,\\psi_1(p + A + 1),

    with ``psi_1`` the trigamma function. This is the ingredient App. B's own
    Bernstein-function property (``Phi' > 0``, ``Phi'' < 0``) refers to, and
    what :func:`softpaws.transport.soft_volume.
    scale_breaking_saturation_factor` needs for the App. F running-index
    correction (Eq. F4 of ``paper/main.tex``).

    Parameters
    ----------
    spectral_index_value : float or np.ndarray
        Source spectral index ``A``.
    b_mu : float or np.ndarray
        Drift coefficient [km^-1].
    d_mu : float or np.ndarray
        Diffusion coefficient [km^-1]. If non-positive, the drift limit
        ``d Phi/dA = b_mu`` is returned (the derivative of ``Phi = A b_mu``).

    Returns
    -------
    phi_prime : np.ndarray
        ``d Phi / dA`` [km^-1], always positive (Bernstein-function property).
    """
    a = np.asarray(spectral_index_value, dtype=float)
    b = np.asarray(b_mu, dtype=float)
    d = np.asarray(d_mu, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa, p = two_moment_loss_spectrum(b, d)
        deriv = kappa * polygamma(1, a + p + 1.0)
    return np.where(d > 0.0, deriv, b)


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

    The truncation of ``Phi(A)`` used by Palmisano et al., kept here for
    head-to-head comparison with :func:`phi_eigenvalue`. It agrees with the exact
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
    source: str = DEFAULT_SOURCE,
) -> np.ndarray:
    """Exact eigenvalue at a muon energy, from the tabulated coefficients.

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
    source : {"proposal", "table1"}, optional
        Transport-coefficient tabulation; see
        :mod:`softpaws.transport.coefficients`.

    Returns
    -------
    phi : np.ndarray
        Eigenvalue ``Phi(A)`` [km^-1].
    """
    b_mu = drift_coefficient(energy_gev, density_g_cm3, source)
    d_mu = diffusion_coefficient(energy_gev, density_g_cm3, source)
    return phi_eigenvalue(spectral_index_value, b_mu, d_mu)
