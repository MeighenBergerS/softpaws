r"""Tau-neutrino contribution to the through-going muon sample.

A ``nu_tau`` charged-current interaction makes a tau instead of a muon. At the
soft-volume regime's energies (>~100 TeV) the tau's decay length grows past its
radiative range, so -- unlike the muon -- it barely loses energy before it
decays: ``b_tau^{-1} ~ 35 km`` versus a decay length ``ell_tau ~ 5 km`` at
100 PeV. The tau therefore free-streams at (near-)fixed energy and then decays;
roughly 17% of the time (:data:`BR_TAU_TO_MU`) that decay is leptonic to a
muon, which then propagates through the ordinary soft-volume physics of
:mod:`softpaws.transport.eigenvalue`.

This is a second, independent way to make a through-going muon track, so it
adds to (does not replace) the direct ``nu_mu`` prediction of
:mod:`softpaws.response.soft_volume`. The chain is three scale-invariant
operators composed:

1. ``nu_tau`` CC production of a tau (same weak vertex as :mod:`.source`,
   drift-free survival to the decay point);
2. leptonic decay to a muon carrying fraction ``z = E_mu / E_tau`` of the
   tau's energy, with density ``g(z)`` (:func:`decay_spectrum`) -- a *second*
   Mellin mode, since a power-law flux picks up the moment ``<z^A>``
   (:func:`z_moment`);
3. ordinary muon soft-volume transport of that secondary muon, i.e. the same
   eigenvalue ``Phi(A)`` used throughout :mod:`softpaws.transport.eigenvalue`.

For a source spectral index ``A`` the ratio of tau-induced to direct-``nu_mu``
muon tracks at observed energy ``E_mu`` works out to

.. math:: \\frac{N_{\\tau\\to\\mu}}{N_{\\nu_\\mu}}(E_\\mu) \\approx
    B_{\\tau\\to\\mu}\\,\\langle z^A\\rangle_g\\,
    \\bigl[\\,1 + \\ell_\\tau(q E_\\mu)\\,\\Phi(A)\\,\\bigr],
    \\qquad q \\equiv 1/\\langle z\\rangle_g,

implemented here as :func:`tau_to_muon_ratio`. The bracket's "1" is the
in-detector piece (a tau produced and decaying right at the observed energy);
the ``ell_tau Phi(A)`` term is the tau analogue of the muon soft volume --
upstream tau production reaching the detector via the decay-length scale
instead of the muon range, with the same eigenvalue ``Phi(A)`` governing the
subsequent muon leg. Because ``ell_tau`` grows linearly with energy while the
muon range ``1/Phi(A)`` is roughly flat, this bracket -- and hence the whole
tau contribution -- grows sharply with energy: of order 5% at 1 PeV, but of
order 30-40% at 100 PeV, squarely in the KM3NeT regime.

Beyond the population-averaged ratio above, :func:`tau_loss_density` gives the
full single-event law needed for parent-energy reconstruction (the tau
counterpart of :mod:`softpaws.transport.loss_distribution`): the observed
muon's total log-loss ``w = ln(E_tau / E_obs) = -ln(z) + w_mu`` composes the
decay fraction with the *ordinary* muon subordinator run over whatever column
remains after the tau's own (random, exponentially distributed) flight length.
Marginalizing that random remaining column against the tau's truncated
decay-length distribution gives a closed-form Mellin-Laplace symbol,

.. math:: \\Psi(s; E_\\tau, \\ell) = \\langle z^s \\rangle\\,
    \\frac{e^{-\\ell / \\ell_\\tau(E_\\tau)} - e^{-\\ell\\,\\Phi(s)}}
    {\\ell_\\tau(E_\\tau)\\,\\Phi(s) - 1},

with ``ell`` the assumed total column from the ``nu_tau`` interaction point to
the detector. This is *defective*: ``Psi(0) = 1 - e^{-ell/ell_tau(E_tau)}``,
the probability the tau decays before reaching the detector at all (see
:func:`tau_survival_before_decay`) -- a tau produced too energetically for its
decay length to fit inside ``ell`` simply does not make an in-medium muon via
this channel, and that missing probability mass must not be renormalized away
when it feeds a posterior over ``E_tau`` (unlike the muon-only case, where the
transport parameters do not depend on the hypothesis being tested, here
``ell_tau(E_tau)`` does, so this suppression is the point).
"""

from __future__ import annotations

import numpy as np

from ..utils.constants import C_KM_PER_S, M_TAU_GEV, RHO_WATER_G_CM3, TAU_LIFETIME_S
from .eigenvalue import phi_eigenvalue_at_energy, phi_symbol, two_moment_loss_spectrum

# tau -> mu nu_mu nu_tau branching ratio (PDG).
BR_TAU_TO_MU = 0.1739

# <z> of the polarized tau -> mu decay spectrum (see decay_spectrum), exact:
# z_moment(1) = 1/6 + 2/15 = 3/10.
MEAN_Z = 0.3


def decay_spectrum(z: float | np.ndarray) -> np.ndarray:
    r"""Muon energy-fraction spectrum ``g(z)`` from polarized tau decay.

    For the leptonic decay ``tau -> mu nu_mu nu_tau``, the V-A matrix element
    in the tau rest frame is ``dGamma/(dx dcos(theta)) ~ x^2 [(3-2x) -
    h(1-2x)cos(theta)]`` with ``x = 2 E_mu^* / m_tau`` and ``h`` the tau
    helicity. A tau produced by ``nu_tau`` CC is (ultra-relativistically)
    fully left-handed, ``h = -1``. Boosting to the lab with
    ``z = x(1 + cos(theta)) / 2`` and integrating out ``cos(theta))`` gives the
    closed form

    .. math:: g(z) = 2 (1 - z)^2 (1 + 2 z), \\qquad z \\in [0, 1],

    normalized to unit area with mean ``<z> = 3/10`` (:data:`MEAN_Z`).

    Parameters
    ----------
    z : float or np.ndarray
        Muon energy fraction ``z = E_mu / E_tau``, in ``[0, 1]``.

    Returns
    -------
    density : np.ndarray
        ``g(z)``, same shape as ``z``.
    """
    zz = np.asarray(z, dtype=float)
    return 2.0 * (1.0 - zz) ** 2 * (1.0 + 2.0 * zz)


def z_symbol(s: complex | np.ndarray) -> np.ndarray:
    r"""Mellin symbol ``<z^s>_g`` of the tau decay spectrum, for any ``s``.

    Writing :func:`decay_spectrum` as ``g(z) = 2(1-z)^2 + 4z(1-z)^2`` and using
    ``int_0^1 z^s (1-z)^2 dz = B(s+1, 3) = 2 / [(s+1)(s+2)(s+3)]``,

    .. math:: \\langle z^s \\rangle_g = \\frac{4}{(s+1)(s+2)(s+3)}
        + \\frac{8}{(s+2)(s+3)(s+4)}.

    Unlike :func:`z_moment`, this keeps the argument's dtype, so it accepts
    **complex** ``s``. That is what the characteristic function of ``-ln(z)``
    needs (:func:`tau_loss_density`, evaluated at ``s = -i k``); for the real
    spectral index ``A`` use :func:`z_moment` (mirrors the
    :func:`~.eigenvalue.phi_symbol` / :func:`~.eigenvalue.phi_eigenvalue` split).

    Parameters
    ----------
    s : complex or np.ndarray
        Mellin variable. Real or complex; the dtype is preserved.

    Returns
    -------
    moment : np.ndarray
        ``<z^s>_g``, matching the (broadcast) shape and dtype of ``s``.
    """
    s = np.asarray(s)
    term1 = 4.0 / ((s + 1.0) * (s + 2.0) * (s + 3.0))
    term2 = 8.0 / ((s + 2.0) * (s + 3.0) * (s + 4.0))
    return term1 + term2


def z_moment(spectral_index_value: float | np.ndarray) -> np.ndarray:
    """Mellin moment ``<z^A>_g`` of the tau decay spectrum, in closed form.

    Real-argument counterpart of :func:`z_symbol` (see there for the derivation).
    Reduces to 1 at ``A = 0`` (normalization) and to :data:`MEAN_Z` at ``A = 1``.

    Parameters
    ----------
    spectral_index_value : float or np.ndarray
        Source spectral index ``A`` (see
        :func:`softpaws.transport.eigenvalue.spectral_index`).

    Returns
    -------
    moment : np.ndarray
        ``<z^A>_g``, dimensionless.
    """
    a = np.asarray(spectral_index_value, dtype=float)
    return z_symbol(a)


def decay_length_km(energy_gev: float | np.ndarray) -> np.ndarray:
    """Mean lab-frame tau decay length ``ell_tau = (E / m_tau) c tau0``.

    Parameters
    ----------
    energy_gev : float or np.ndarray
        Tau energy [GeV].

    Returns
    -------
    length : np.ndarray
        Mean decay length [km]. ``~5 km`` at ``E_tau = 100`` PeV.
    """
    energy = np.asarray(energy_gev, dtype=float)
    return (energy / M_TAU_GEV) * C_KM_PER_S * TAU_LIFETIME_S


def tau_survival_before_decay(
    ell_km: float,
    e_tau_gev: float | np.ndarray,
) -> np.ndarray:
    """Probability the tau decays before reaching the end of the column.

    ``P(x_decay < ell) = 1 - e^{-ell / ell_tau(E_tau)}``, for a tau flight
    length ``x_decay`` exponentially distributed with mean
    :func:`decay_length_km`. This is the total mass ``Psi(0)`` of
    :func:`tau_loss_density` (see its module-level derivation): a tau too
    energetic for its decay length to fit inside ``ell`` simply does not
    produce an in-medium muon via this channel.

    Parameters
    ----------
    ell_km : float
        Total column depth from the ``nu_tau`` interaction point to the
        detector [km].
    e_tau_gev : float or np.ndarray
        Tau energy [GeV].

    Returns
    -------
    probability : np.ndarray
        ``P(x_decay < ell)``, in ``[0, 1]``.
    """
    ell_tau = decay_length_km(e_tau_gev)
    return 1.0 - np.exp(-ell_km / ell_tau)


def tau_to_muon_ratio(
    spectral_index_value: float | np.ndarray,
    energy_gev: float | np.ndarray,
    density_g_cm3: float = RHO_WATER_G_CM3,
) -> np.ndarray:
    """Ratio of tau-induced to direct-``nu_mu`` muon tracks, ``N_tau / N_numu``.

    Implements the module-level formula: the tau decay branching ratio and
    Mellin moment times the bracket ``[1 + ell_tau(q E_mu) Phi(A)]``, with
    ``Phi(A)`` the exact muon eigenvalue (:func:`~.eigenvalue.phi_eigenvalue_at_energy`)
    evaluated at the *observed* muon energy (the same convention used
    throughout :mod:`softpaws.transport.soft_volume`), and ``q E_mu`` the
    typical parent tau energy for that observed muon.

    Parameters
    ----------
    spectral_index_value : float or np.ndarray
        Source spectral index ``A = gamma - lambda - 1``.
    energy_gev : float or np.ndarray
        Observed muon energy [GeV].
    density_g_cm3 : float, optional
        Target-medium density [g cm^-3]. Defaults to water.

    Returns
    -------
    ratio : np.ndarray
        ``N_tau_to_mu / N_numu`` at each energy, for a ``nu_tau`` flux with the
        same normalization and spectral index as the ``nu_mu`` flux.
    """
    a = np.asarray(spectral_index_value, dtype=float)
    e_mu = np.asarray(energy_gev, dtype=float)
    q = 1.0 / MEAN_Z
    ell_tau = decay_length_km(q * e_mu)
    phi = phi_eigenvalue_at_energy(a, e_mu, density_g_cm3)
    return BR_TAU_TO_MU * z_moment(a) * (1.0 + ell_tau * phi)


def tau_loss_density(
    w_grid: np.ndarray,
    ell_km: float,
    e_tau_gev: float | np.ndarray,
    b_mu: float,
    d_mu: float,
    n_k: int = 2**14,
) -> np.ndarray:
    r"""Defective density of ``w = ln(E_tau / E_obs)`` for the tau-origin chain.

    Inverts the closed-form symbol derived in the module docstring,

    .. math:: \\Psi(s) = \\langle z^s \\rangle\\,
        \\frac{e^{-\\ell/\\ell_\\tau} - e^{-\\ell\\,\\Phi(s)}}
        {\\ell_\\tau\\,\\Phi(s) - 1},

    by the same characteristic-function inversion as
    :func:`softpaws.transport.loss_distribution.loss_density`, evaluating
    ``Phi`` and ``<z^s>`` at ``s = -i k`` via :func:`~.eigenvalue.phi_symbol`
    and :func:`z_symbol`. The removable singularity at ``ell_tau Phi(s) = 1``
    is handled by its L'Hopital limit, ``(ell/ell_tau) e^{-ell/ell_tau}``.

    Unlike :func:`~softpaws.transport.loss_distribution.loss_density`, the
    result is **not** renormalized to unit area: its integral over ``w`` is
    :func:`tau_survival_before_decay`, which is physically meaningful and must
    survive into any posterior built from this density (see the module
    docstring). ``e_tau_gev`` broadcasts against ``w_grid``: pass a scalar
    for the density of one fixed tau-energy hypothesis (e.g. to check the
    total-mass identity), or an array matching ``w_grid`` -- typically
    ``E_mu * exp(w_grid)`` -- to read off, at each grid point, the likelihood
    of the *specific* tau-energy hypothesis that ``w`` corresponds to (the
    diagonal evaluation a parent-energy reconstruction needs, since
    ``ell_tau`` depends on the hypothesis being tested rather than being a
    fixed nuisance parameter).

    Parameters
    ----------
    w_grid : np.ndarray
        Uniformly spaced grid of log-loss values ``w = ln(E_tau / E_obs)``
        (>= 0) at which to evaluate the density.
    ell_km : float
        Total column depth from the ``nu_tau`` interaction point to the
        detector [km].
    e_tau_gev : float or np.ndarray
        Tau energy [GeV] at which to evaluate ``ell_tau``; broadcasts against
        ``w_grid`` (see above).
    b_mu : float
        Total muon drift coefficient ``b_mu`` [km^-1], evaluated at the
        observed muon energy (as in
        :func:`~softpaws.transport.loss_distribution.loss_density`).
    d_mu : float
        Total muon diffusion coefficient ``d_mu`` [km^-1].
    n_k : int, optional
        Number of nodes on the ``k`` grid of the inversion.

    Returns
    -------
    density : np.ndarray
        Defective probability density ``P(w)``, same shape as ``w_grid``.

    Notes
    -----
    ``w_grid`` must be uniformly spaced; the ``k`` span is derived from its
    step, as in :func:`~softpaws.transport.loss_distribution.loss_density`.
    """
    w = np.asarray(w_grid, dtype=float)
    ell_tau = np.broadcast_to(decay_length_km(np.asarray(e_tau_gev, dtype=float)), w.shape)
    kappa, p = two_moment_loss_spectrum(b_mu, d_mu)

    dw = w[1] - w[0]
    k_max = np.pi / dw
    k = np.linspace(-k_max, k_max, n_k, endpoint=False)
    dk = k[1] - k[0]
    s = -1j * k

    phi = phi_symbol(s, kappa, p)  # (n_k,), shared across all w
    z_cf = z_symbol(s)  # (n_k,), shared across all w

    density = np.empty(w.shape[0])
    for i in range(0, w.shape[0], 200):
        ell_tau_chunk = ell_tau[i : i + 200][:, None]
        w_chunk = w[i : i + 200][:, None]

        numer = np.exp(-ell_km / ell_tau_chunk) - np.exp(-ell_km * phi[None, :])
        denom = ell_tau_chunk * phi[None, :] - 1.0
        small = np.abs(denom) < 1e-9
        safe_denom = np.where(small, 1.0, denom)
        limit_value = (ell_km / ell_tau_chunk) * np.exp(-ell_km / ell_tau_chunk)
        decay_integral = np.where(small, limit_value, numer / safe_denom)

        cf = z_cf[None, :] * decay_integral
        density[i : i + 200] = (cf * np.exp(-1j * w_chunk * k[None, :])).sum(1).real

    density *= dk / (2.0 * np.pi)
    return np.clip(density, 0.0, None)
