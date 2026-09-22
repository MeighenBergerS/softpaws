"""Bayesian inference layer for the diffuse-flux and cross-section fits.

Builds the binned Poisson likelihood of Palmisano et al. (arXiv:2607.13143)
Eq. (4.2) on top of the soft-volume forward model
(:class:`softpaws.response.soft_volume.SoftVolumeResponse`) and drives it with
an ``emcee`` sampler. This is what reproduces the Section 4 figures of that
earlier work, which softpaws keeps as a cross-check:

- **Fig. 6 / 7** — posteriors of the diffuse-flux parameters ``(phi0, gamma)`` and
  the transport nuisances ``(b_mu, d_mu)``, for the ``drift``, ``diffusion``, and
  the new ``exact`` forward models.
- **Fig. 8** — the cross-section slope ``lambda`` required for one muon event at
  ``100 PeV`` (Eq. 4.6), via :func:`required_lambda`. Here the FP drift form
  diverges at the pole ``lambda = gamma - 1`` while the eigenvalue form stays
  finite (see ``docs/theory/exact_soft_volume.md``).

Data caveats (this is a DR2-based *proxy*, not the Ref. [28] dataset of
Palmisano et al.):

- The IceCube counts come from the in-repo IceTracks-DR2 downgoing events.
- The atmospheric background is a floated power-law template
  (:func:`atmospheric_template`), not their fixed per-bin background.

The transport-nuisance priors follow the MC calibration of their Section 3
(Eq. 3.10):
``b_mu`` rescaled by ``b_scale ~ N(0.94, 0.15)`` and ``d_mu`` by a log-normal
``d_scale`` matching ``1.5 (+1.6 / -0.8)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import emcee
import numpy as np
from scipy.optimize import brentq

from ..constants import GAMMA_TRUTH, PHI0_TRUTH
from ..response.soft_volume import SoftVolumeResponse, tau_induced_expected_counts
from ..transport.source import DEFAULT_LAMBDA, E0_CROSS_GEV

# Flat-prior support for the flux parameters (uniform within these ranges).
PHI0_RANGE = (0.0, 5.0)
GAMMA_RANGE = (1.5, 3.5)

# Transport-nuisance priors (Section 3, Eq. 3.10), as rescalings of Table 1.
B_SCALE_MEAN, B_SCALE_STD = 0.94, 0.15  # b_scale ~ N(mean, std), truncated > 0
D_SCALE_MEDIAN, D_SCALE_LOGSTD = 1.5, 0.6  # d_scale ~ lognormal, truncated > 0

# The SM reference cross-section slope, used for R = sigma / sigma_SM.
LAMBDA_SM = DEFAULT_LAMBDA


def asimov_dataset(config: "FitConfig", phi0: float = PHI0_TRUTH, gamma: float = GAMMA_TRUTH,
                   b_scale: float = 1.0, d_scale: float = 1.0,
                   bkg_total: float = 0.0, bkg_slope: float = 3.7) -> np.ndarray:
    """Expected (Asimov) counts from an injected truth flux, plus optional background.

    The Asimov dataset is the noise-free expectation at a chosen truth; fitting it
    recovers that truth, so it is the standard way to show a pipeline reproduces
    the expected posteriors. Here the IceCube figures inject the best-fit flux of
    Palmisano et al. through the forward model rather than fitting the atmospheric-muon-swamped
    DR2 downgoing sample (which lies outside the neutrino soft-volume model's
    validity).

    Parameters
    ----------
    config : FitConfig
        Fit configuration whose forward model generates the signal.
    phi0, gamma : float, optional
        Injected diffuse-flux truth. Default to the diffusion best fit of
        Palmisano et al.
    b_scale, d_scale : float, optional
        Injected transport-coefficient truth. Default to the theoretical values.
    bkg_total : float, optional
        Total background events to add, distributed by :func:`atmospheric_template`.
        Defaults to none.
    bkg_slope : float, optional
        Atmospheric template slope for the injected background.

    Returns
    -------
    counts : np.ndarray
        Expected counts per bin (signal + background).
    """
    counts = signal_counts(config, phi0, gamma, b_scale, d_scale)
    if bkg_total > 0.0 and np.asarray(config.log10_e_edges).size > 2:
        counts = counts + bkg_total * atmospheric_template(config.log10_e_edges, bkg_slope)
    return counts


@dataclass
class FitConfig:
    """Static configuration of a binned soft-volume fit.

    Parameters
    ----------
    radius_km : float
        Spherical-detector radius [km].
    log10_e_edges : np.ndarray
        Muon-energy bin edges in ``log10(E / GeV)``.
    livetime_s : float
        Exposure time [s].
    solid_angle_sr : float
        Solid angle of the analysis region [sr].
    method : {"drift", "diffusion", "exact"}, optional
        Soft-volume forward model. Defaults to ``"diffusion"``.
    column_depth_km : float or None, optional
        Upstream column depth for the exact method [km].
    lam : float, optional
        CC cross-section slope held fixed in the *flux* fit. Defaults to
        :data:`softpaws.transport.source.DEFAULT_LAMBDA`.
    density_g_cm3 : float, optional
        Target-medium density [g cm^-3].
    n_subdivisions : int, optional
        Log-spaced sample points per bin for the signal integral. Kept modest for
        MCMC speed.
    origin : {"numu", "generic"}, optional
        Assumed flavor origin of the signal. ``"numu"`` (the default) is the
        direct ``nu_mu`` CC prediction alone. ``"generic"`` adds the tau-induced
        contribution (:func:`softpaws.response.soft_volume.tau_induced_expected_counts`)
        from a ``nu_tau`` flux with the *same* ``(phi0, gamma)``, i.e. the event
        sample is agnostic to whether a given track came from direct ``nu_mu``
        production or ``nu_tau -> tau -> mu``. Only defined for ``method="exact"``,
        since the tau ratio is built from the exact eigenvalue ``Phi(A)``
        (:mod:`softpaws.transport.tau`).
    """

    radius_km: float
    log10_e_edges: np.ndarray
    livetime_s: float
    solid_angle_sr: float
    method: str = "diffusion"
    column_depth_km: float | None = None
    lam: float = DEFAULT_LAMBDA
    density_g_cm3: float = 1.02
    n_subdivisions: int = 16
    origin: str = "numu"
    background: np.ndarray | None = field(default=None, repr=False)

    def uses_diffusion(self) -> bool:
        """Whether the model has a free diffusion nuisance ``d_scale``."""
        return self.method in ("diffusion", "exact")


def atmospheric_template(log10_e_edges: np.ndarray, slope: float = 3.7) -> np.ndarray:
    """Unit-sum background shape ``propto integral E^-slope dE`` per bin.

    A pragmatic stand-in for the fixed per-bin atmospheric background of
    Palmisano et al.
    (which is not available for the DR2 release). Scaled by a free ``bkg_norm``
    nuisance in the fit, so ``bkg_norm`` is the total expected background count.

    Parameters
    ----------
    log10_e_edges : np.ndarray, shape (n_bins + 1,)
        Muon-energy bin edges in ``log10(E / GeV)``.
    slope : float, optional
        Falling power-law index of the atmospheric muon spectrum. Defaults to
        3.7, typical of the conventional atmospheric flux.

    Returns
    -------
    shape : np.ndarray, shape (n_bins,)
        Per-bin fractions summing to 1.
    """
    edges = 10.0 ** np.asarray(log10_e_edges, dtype=float)
    if np.isclose(slope, 1.0):
        integral = np.diff(np.log(edges))
    else:
        integral = (edges[1:] ** (1.0 - slope) - edges[:-1] ** (1.0 - slope)) / (1.0 - slope)
    return integral / integral.sum()


def poisson_log_likelihood(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Binned Poisson log-likelihood ``sum_i (k_i ln N_i - N_i)`` (Eq. 4.2).

    The ``ln k_i!`` term is dropped, as it is constant in the model parameters.

    Parameters
    ----------
    observed : np.ndarray
        Observed counts ``k_i`` per bin.
    predicted : np.ndarray
        Expected counts ``N_i`` per bin (signal + background). Must be positive
        wherever it is evaluated.

    Returns
    -------
    log_l : float
        Poisson log-likelihood (up to the parameter-independent constant).
    """
    k = np.asarray(observed, dtype=float)
    n = np.asarray(predicted, dtype=float)
    if np.any(n <= 0.0):
        return -np.inf
    return float(np.sum(k * np.log(n) - n))


def free_param_names(method: str) -> list[str]:
    """Ordered names of the free parameters for a given forward model."""
    names = ["phi0", "gamma", "b_scale"]
    if method in ("diffusion", "exact"):
        names.append("d_scale")
    names.append("bkg_norm")
    return names


def _theta_to_dict(theta: Sequence[float], method: str) -> dict[str, float]:
    return dict(zip(free_param_names(method), theta, strict=True))


def signal_counts(config: FitConfig, phi0: float, gamma: float,
                  b_scale: float, d_scale: float) -> np.ndarray:
    """Expected astrophysical counts per bin for the configured forward model.

    With ``config.origin == "generic"`` this adds the tau-induced contribution
    from a same-normalization ``nu_tau`` flux to the direct ``nu_mu`` prediction
    (see :attr:`FitConfig.origin`).

    Raises
    ------
    ValueError
        Raised if ``config.origin == "generic"`` while ``config.method !=
        "exact"``, since the tau ratio needs the exact eigenvalue ``Phi(A)``.
    """
    response = SoftVolumeResponse(
        config.radius_km,
        config.density_g_cm3,
        method=config.method,
        column_depth_km=config.column_depth_km,
        b_scale=b_scale,
        d_scale=d_scale,
    )
    counts = response.expected_counts(
        config.log10_e_edges,
        phi0,
        gamma,
        config.livetime_s,
        config.solid_angle_sr,
        lam=config.lam,
        n_subdivisions=config.n_subdivisions,
    )
    if config.origin == "generic":
        if config.method != "exact":
            raise ValueError(
                "FitConfig.origin='generic' (tau-inclusive) requires method='exact', "
                f"got {config.method!r}."
            )
        counts = counts + tau_induced_expected_counts(
            response,
            config.log10_e_edges,
            phi0,
            gamma,
            config.livetime_s,
            config.solid_angle_sr,
            lam=config.lam,
            n_subdivisions=config.n_subdivisions,
        )
    return counts


def log_prior(theta: Sequence[float], method: str) -> float:
    """Log-prior for the flux and transport-nuisance parameters."""
    p = _theta_to_dict(theta, method)
    if not (PHI0_RANGE[0] < p["phi0"] < PHI0_RANGE[1]):
        return -np.inf
    if not (GAMMA_RANGE[0] < p["gamma"] < GAMMA_RANGE[1]):
        return -np.inf
    if p["b_scale"] <= 0.0 or p["bkg_norm"] < 0.0:
        return -np.inf

    # b_scale: truncated Gaussian about the MC-calibrated value.
    lp = -0.5 * ((p["b_scale"] - B_SCALE_MEAN) / B_SCALE_STD) ** 2

    # d_scale: right-skewed log-normal, positive.
    if method in ("diffusion", "exact"):
        d = p["d_scale"]
        if d <= 0.0:
            return -np.inf
        lp += -np.log(d) - 0.5 * ((np.log(d) - np.log(D_SCALE_MEDIAN)) / D_SCALE_LOGSTD) ** 2

    return float(lp)


def log_posterior(theta: Sequence[float], config: FitConfig, observed: np.ndarray) -> float:
    """Log-posterior = log-prior + Poisson log-likelihood."""
    lp = log_prior(theta, config.method)
    if not np.isfinite(lp):
        return -np.inf
    p = _theta_to_dict(theta, config.method)
    d_scale = p.get("d_scale", 1.0)
    # The eigenvalue family and saturation factor can legitimately over/underflow
    # where a walker wanders into an unphysical region; those samples are rejected
    # by the finiteness check below, so the transient warnings are not informative.
    with np.errstate(all="ignore"):
        signal = signal_counts(config, p["phi0"], p["gamma"], p["b_scale"], d_scale)
    background = p["bkg_norm"] * config.background if config.background is not None else 0.0
    predicted = signal + background
    # Reject unphysical regions (e.g. d_mu >= b_mu, where the two-moment
    # eigenvalue family is undefined and the soft volume is non-finite).
    if not np.all(np.isfinite(predicted)):
        return -np.inf
    return lp + poisson_log_likelihood(observed, predicted)


def initial_walkers(method: str, n_walkers: int, rng: np.random.Generator) -> np.ndarray:
    """Starting positions for the ensemble, scattered around a sensible guess."""
    guess = {"phi0": 0.7, "gamma": 2.38, "b_scale": 0.94, "d_scale": 1.5, "bkg_norm": 1.0}
    names = free_param_names(method)
    center = np.array([guess[name] for name in names])
    scatter = 0.05 * np.abs(center) + 1e-3
    return center + scatter * rng.standard_normal((n_walkers, len(names)))


def run_mcmc(config: FitConfig, observed: np.ndarray, background: np.ndarray | None = None,
             n_walkers: int = 32, n_steps: int = 3000, n_burn: int = 1000,
             seed: int = 0) -> dict[str, np.ndarray]:
    """Sample the posterior and return flat chains keyed by parameter name.

    Parameters
    ----------
    config : FitConfig
        Fit configuration; its ``background`` field is set from ``background``.
    observed : np.ndarray
        Observed counts per bin.
    background : np.ndarray or None, optional
        Per-bin background *shape* (summing to 1); ``None`` for no background
        (e.g. the KM3NeT single-event fit).
    n_walkers, n_steps, n_burn : int, optional
        Ensemble size, chain length, and burn-in to discard.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    samples : dict of str -> np.ndarray
        Flattened post-burn-in chains, one array per free parameter, plus
        ``"b_mu"`` and (if applicable) ``"d_mu"`` in physical units [km^-1].
    """
    config.background = background
    rng = np.random.default_rng(seed)
    names = free_param_names(config.method)
    ndim = len(names)
    p0 = initial_walkers(config.method, n_walkers, rng)

    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior, args=(config, observed))
    sampler.run_mcmc(p0, n_steps, progress=False)
    flat = sampler.get_chain(discard=n_burn, flat=True)

    samples = {name: flat[:, i] for i, name in enumerate(names)}

    # Physical transport coefficients at 10 PeV, for the corner plot axes.
    from ..transport.coefficients import diffusion_coefficient, drift_coefficient

    b0 = float(drift_coefficient(1.0e7, config.density_g_cm3)[0])
    samples["b_mu"] = samples["b_scale"] * b0
    if config.uses_diffusion():
        d0 = float(diffusion_coefficient(1.0e7, config.density_g_cm3)[0])
        samples["d_mu"] = samples["d_scale"] * d0
    return samples


def cross_section_enhancement(lam: float, e_star_gev: float,
                              e0_gev: float = E0_CROSS_GEV) -> float:
    """Cross-section enhancement ``R = sigma(lambda) / sigma(lambda_SM)`` at ``E*``.

    Since ``sigma propto (E / E0)^lambda``, at fixed ``E*`` this is
    ``(E* / E0)^(lambda - lambda_SM)``.

    Parameters
    ----------
    lam : float
        Fitted effective cross-section slope.
    e_star_gev : float
        Reference energy [GeV] (100 PeV for Fig. 8).
    e0_gev : float, optional
        Cross-section pivot energy [GeV]. Defaults to
        :data:`softpaws.transport.source.E0_CROSS_GEV` (10 PeV).

    Returns
    -------
    R : float
        Enhancement relative to the SM reference ``lambda = 0.4``.
    """
    return float((e_star_gev / e0_gev) ** (lam - LAMBDA_SM))


def _events_at_energy(config: FitConfig, phi0: float, gamma: float, lam: float,
                      e_star_gev: float) -> float:
    """Expected muon events in one e-fold at ``E*`` (Eq. 4.6 structure)."""
    response = SoftVolumeResponse(
        config.radius_km,
        config.density_g_cm3,
        method=config.method,
        column_depth_km=config.column_depth_km,
    )
    volume_cm3 = response.target_volume_cm3(e_star_gev, gamma, lam=lam, part="total")
    weak = response.weak_rate_density(e_star_gev, phi0, gamma, lam=lam)
    rate = float((volume_cm3 * weak)[0])  # [GeV^-1 s^-1 sr^-1]
    return rate * e_star_gev * config.livetime_s * config.solid_angle_sr


def required_lambda(config: FitConfig, phi0: float, gamma: float,
                    e_star_gev: float = 1.0e8,
                    lam_bounds: tuple[float, float] | None = None) -> float:
    """Cross-section slope giving one muon event at ``E*`` (Eq. 4.6).

    Solves ``N_mu(E*; lambda) = 1`` for ``lambda`` by bracketing root-find. For
    the FP (drift/diffusion) models the soft volume diverges at
    ``lambda = gamma - 1`` (the ``A = 0`` pole), so the search is confined below
    it; the exact model with a finite column stays finite, so a wider bracket is
    used and the returned value can exceed the pole.

    Parameters
    ----------
    config : FitConfig
        Fit configuration (its ``method`` selects FP vs exact, ``livetime_s`` and
        ``solid_angle_sr`` set the exposure).
    phi0, gamma : float
        Diffuse-flux parameters, typically drawn from a posterior.
    e_star_gev : float, optional
        Reference muon energy [GeV]. Defaults to 100 PeV.
    lam_bounds : tuple of float, optional
        Search bracket for ``lambda``. Defaults to ``(-0.9, gamma - 1 - 1e-4)``
        for the FP models and ``(-0.9, 3.0)`` for the exact model.

    Returns
    -------
    lam : float
        Required cross-section slope, or ``nan`` if no root lies in the bracket.
    """
    if lam_bounds is None:
        if config.method == "exact":
            lam_bounds = (-0.9, 3.0)
        else:
            lam_bounds = (-0.9, gamma - 1.0 - 1e-4)

    def log_n(lam: float) -> float:
        n = _events_at_energy(config, phi0, gamma, lam, e_star_gev)
        return np.log(n) if n > 0 else -np.inf

    lo, hi = lam_bounds
    f_lo, f_hi = log_n(lo), log_n(hi)
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0.0:
        return float("nan")
    return float(brentq(log_n, lo, hi, xtol=1e-4))
