"""Exponential-cutoff sources via real-space log-loss convolution (App. H).

A pure power-law source excites exactly one Mellin mode, ``s = A``
(:mod:`softpaws.transport.eigenvalue`), which is what makes the closed-form
soft volume (Eq. 9) possible. Sec. VII.C of ``docs/2026_softvolume.pdf`` notes
that this breaks down for ``A < 0`` (the cross-section pole,
:func:`softpaws.transport.soft_volume.spectral_penalty`): the effective range
scales as ``exp(x |Phi(A)|)``, which for a realistic Earth-crossing column
(``x ~ 10^4`` km.w.e.) is an "enormous exponential enhancement." Sec. VII.C
"strongly recommend[s] mitigating this by applying a physical spectral cutoff
to the parent neutrino flux," derived in App. H for a source
``phi_nu ~ E^-gamma e^{-E/E0}``.

**Why this isn't the paper's literal Cahen-Mellin pole series.** App. H's own
route (Eq. H1-H3) sums the source's Mellin poles at ``s = A, A-1, A-2, ...``
via a Taylor expansion of the transport kernel ``h(s)`` about ``s = A``,
truncated to a handful of terms. That series is asymptotic rather than
rapidly convergent in exactly the regime it exists to fix: for the same
large-``x``, negative-``A`` case Sec. VII.C describes, ``h(s)`` itself scales
like ``exp(x |Phi(s)|)`` at every stencil point near ``A``, so the individual
Taylor terms overflow before any cancellation can bring them back down to the
finite physical answer -- a truncated real-axis series is not a numerically
stable way to evaluate it there.

This module instead evaluates the *same physical convolution* directly in
real (energy) space, which stays numerically well behaved throughout: App. B
already establishes that the muon's accumulated log-loss ``w = ln(eps / E)``
over a column ``ell`` is a subordinator with an exactly known density
(:func:`softpaws.transport.loss_distribution.loss_density`, always a proper,
normalized, non-negative probability law -- no exponentially large
intermediate quantities anywhere). Folding the cutoff source through that
density and integrating over the upstream production depth
(:func:`cutoff_soft_rate_density`) is mathematically the same convolution
App. H performs in Mellin space, just carried out without ever forming the
divergent-looking intermediate series.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .loss_distribution import loss_density


def cutoff_soft_rate_density(
    energy_gev: float | np.ndarray,
    e0_cutoff_gev: float,
    column_depth_km: float,
    b_mu: float,
    d_mu: float,
    weak_rate_density_gev: Callable[[np.ndarray], np.ndarray],
    n_xi: int = 12,
    n_w: int = 512,
    w_span: float | None = None,
    n_k: int = 4096,
) -> np.ndarray:
    """Transported (soft) rate density for a source with an exponential cutoff.

    Generalizes Eq. 9's ``S(E) (1 - e^{-x Phi(A)}) / Phi(A)`` to a source that
    is not a single power-law mode, ``S(eps) e^{-eps / E0}``. A muon observed
    at energy ``E`` and produced at column depth ``xi`` (so it propagated
    ``ell = x - xi``) was produced at energy ``eps = E e^w``, with ``w`` drawn
    from the exact log-loss density at that ``ell``
    (:func:`~softpaws.transport.loss_distribution.loss_density`). Averaging
    the cutoff source over ``w`` and integrating over the upstream production
    depth gives

    .. math:: \\phi_\\mathrm{soft}(x, E) = \\int_0^x d\\xi \\int_0^\\infty dw\\,
        p_{x-\\xi}(w)\\, S(E e^w)\\, e^{-E e^w / E_0}\\, e^w,

    evaluated by a midpoint rule in ``xi`` (excluding the ``xi = x`` endpoint,
    which is the in-detector population handled separately -- see
    :meth:`softpaws.response.soft_volume.SoftVolumeResponse.
    differential_rate_with_cutoff`) and a trapezoidal rule in ``w``. Reduces
    to the plain treatment as ``e0_cutoff_gev -> inf``.

    Parameters
    ----------
    energy_gev : float or np.ndarray
        Observed muon energy [GeV].
    e0_cutoff_gev : float
        Spectral cutoff energy ``E0`` [GeV] of the parent neutrino flux.
    column_depth_km : float
        Available upstream column depth ``x`` [km].
    b_mu, d_mu : float
        Drift and diffusion coefficients [km^-1], evaluated at a representative
        energy (see the caller's docstring for the approximation this makes).
    weak_rate_density_gev : callable
        The **uncut** weak-rate density ``n_N sigma_CC(eps) phi_nu(eps)``
        (e.g. :meth:`softpaws.response.soft_volume.SoftVolumeResponse.
        weak_rate_density`, with the cutoff applied here instead), evaluated
        at a production energy ``eps`` [GeV]; must accept and return arrays.
    n_xi : int, optional
        Number of production-depth samples. Defaults to 12.
    n_w : int, optional
        Number of log-loss grid points per :func:`~softpaws.transport.
        loss_distribution.loss_density` call. Defaults to 512.
    w_span : float or None, optional
        Upper end of the log-loss grid. ``None`` (the default) sizes it
        automatically from the transport width at the full column
        (``b_mu * x + 10 sqrt(d_mu * x)``) and the cutoff's own reach
        (``ln(e0_cutoff_gev / min(energy_gev))``), whichever is larger --
        :func:`~softpaws.transport.loss_distribution.loss_density`'s
        characteristic-function inversion needs the grid *spacing*
        ``w_span / n_w`` to resolve the density's actual width, so an
        unnecessarily large fixed span (relative to a typical few-km column)
        wastes resolution and biases the result low; see the module
        docstring's numerical note.
    n_k : int, optional
        Number of inversion nodes passed to :func:`~softpaws.transport.
        loss_distribution.loss_density`. Defaults to 4096.

    Returns
    -------
    rate_density : np.ndarray
        Rate density integrated over the upstream column [cm^-3 GeV^-1 s^-1
        sr^-1 km], broadcast to the shape of ``energy_gev``. Multiply by the
        projected area [cm^2] to get a differential rate contribution
        [GeV^-1 s^-1 sr^-1], matching how ``A_proj`` multiplies the plain
        soft-volume range elsewhere in the package.
    """
    energy = np.atleast_1d(np.asarray(energy_gev, dtype=float))
    x = float(column_depth_km)
    d_xi = x / n_xi
    ell = x - (np.arange(n_xi) + 0.5) * d_xi  # midpoint rule, excludes xi = x

    if w_span is None:
        # Whichever factor decays first sets the integrand's actual extent: the
        # transport density p_ell(w) (width ~ b_mu x + few sqrt(d_mu x)) or the
        # cutoff's own e^{-eps/E0} suppression (kicks in around w ~ ln(E0/E)).
        # Sizing the grid to the larger of the two -- as if both had to be
        # resolved -- wastes resolution on a region where the integrand is
        # already negligible and biases the result low (see the module
        # docstring's numerical note); take the smaller instead, each with a
        # margin so the true decay is captured rather than clipped early.
        transport_width = b_mu * x + 10.0 * np.sqrt(max(d_mu, 0.0) * x) + 1.0
        cutoff_reach = np.log(e0_cutoff_gev / float(energy.min())) + 5.0
        w_span = max(min(transport_width, cutoff_reach), 1.0)
    w = np.linspace(0.0, w_span, n_w)
    exp_w = np.exp(w)

    total = np.zeros_like(energy)
    for ell_i in ell:
        p_w = loss_density(w, float(ell_i), b_mu, d_mu, n_k)
        eps = energy[:, None] * exp_w[None, :]
        integrand = (
            p_w[None, :]
            * weak_rate_density_gev(eps)
            * np.exp(-eps / e0_cutoff_gev)
            * exp_w[None, :]
        )
        total += np.trapezoid(integrand, w, axis=1) * d_xi
    return total
