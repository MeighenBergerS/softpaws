"""Observed vs. predicted reconstructed-energy track counts.

Puts the published-IRF forward model (:mod:`softpaws.response.irfs`) and the
soft-volume forward model (:mod:`softpaws.response.soft_volume`) on a common
footing, in reconstructed-energy bins comparable to the observed IceTracks-DR2
event histogram: the IRF path is migrated from true neutrino energy to
reconstructed energy via the smearing matrix's :meth:`~softpaws.response.irfs.
SmearingMatrix.energy_response_matrix`, and the soft-volume path already
predicts *muon energy at the detector*, which is used directly as the
reconstructed-energy proxy (Palmisano et al., arXiv:2607.13143, Section 2.1).

Notes
-----
The soft-volume drift limit implemented in
:mod:`softpaws.transport.soft_volume` assumes an unattenuated neutrino flux
(Eq. 2.21 takes ``D_nu = 1``), which is only realistic for the downgoing
hemisphere. The published effective area, by contrast, already has Earth
attenuation baked in for every direction. Comparisons here should therefore be
restricted to the downgoing hemisphere (``dec < 0`` at IceCube) until the
transport path grows an explicit attenuation model; see
``docs/theory/soft_volume.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ..constants import _BISECTION_STEPS
from ..data.container import EventSet
from ..response.irfs import EffectiveArea, SmearingMatrix


def observed_counts(
    events: EventSet,
    log10_e_edges: np.ndarray,
    dec_min: float,
    dec_max: float,
) -> np.ndarray:
    """Histogram of observed reconstructed energies within a declination band.

    Parameters
    ----------
    events : softpaws.data.container.EventSet
        Observed IceTracks-DR2 events.
    log10_e_edges : np.ndarray, shape (n_bins + 1,)
        Reconstructed-energy bin edges in ``log10(E/GeV)``.
    dec_min : float
        Minimum declination [deg].
    dec_max : float
        Maximum declination [deg].

    Returns
    -------
    counts : np.ndarray, shape (n_bins,)
        Observed event counts per bin.
    """
    subset = events.filter_dec(dec_min, dec_max)
    counts, _ = np.histogram(subset.log10_energy, bins=log10_e_edges)
    return counts


def irf_expected_counts(
    aeff: EffectiveArea,
    smearing: SmearingMatrix,
    log10_e_reco_edges: np.ndarray,
    dec_min: float,
    dec_max: float,
    flux_fn: Callable[[np.ndarray], np.ndarray],
    livetime_s: float,
    n_subdivisions: int = 16,
) -> np.ndarray:
    """Predicted reconstructed-energy track counts from the IRF path.

    For each true-energy/declination group of the smearing table restricted to
    ``[dec_min, dec_max]``, computes the expected number of true muon-neutrino
    events, ``A_eff * (solid angle) * (integrated flux) * livetime``, then
    migrates it into reconstructed-energy bins using that group's row of
    ``smearing.energy_response_matrix``.

    Parameters
    ----------
    aeff : softpaws.response.irfs.EffectiveArea
        Effective area for the same season as ``smearing``.
    smearing : softpaws.response.irfs.SmearingMatrix
        Smearing matrix for one season; its own (true-energy, declination)
        bin grid sets the integration grid.
    log10_e_reco_edges : np.ndarray, shape (n_reco + 1,)
        Reconstructed-energy bin edges in ``log10(E/GeV)``, matching the
        binning used for :func:`observed_counts`.
    dec_min : float
        Minimum declination [deg].
    dec_max : float
        Maximum declination [deg].
    flux_fn : callable
        Differential neutrino flux, ``flux_fn(E_nu_gev) -> dphi/dE`` in
        ``GeV^-1 cm^-2 s^-1 sr^-1``. See
        :func:`~softpaws.response.soft_volume.power_law_flux`.
    livetime_s : float
        Exposure time [s].
    n_subdivisions : int, optional
        Number of log-spaced sample points per true-energy bin for the flux
        integral.

    Returns
    -------
    counts : np.ndarray, shape (n_reco,)
        Expected reconstructed-energy track counts per bin.
    """
    return irf_expected_counts_directional(
        aeff,
        smearing,
        log10_e_reco_edges,
        dec_min,
        dec_max,
        lambda energy, _dec: flux_fn(energy),
        livetime_s,
        n_subdivisions,
    )


def irf_expected_counts_directional(
    aeff: EffectiveArea,
    smearing: SmearingMatrix,
    log10_e_reco_edges: np.ndarray,
    dec_min: float,
    dec_max: float,
    flux_fn: Callable[[np.ndarray, float], np.ndarray],
    livetime_s: float,
    n_subdivisions: int = 16,
) -> np.ndarray:
    """IRF-path counts for a flux that also depends on direction.

    Same calculation as :func:`irf_expected_counts` -- which is a thin wrapper
    around this one -- except that the flux is evaluated separately in every
    declination bin of the smearing table. That matters for an atmospheric flux,
    which is strongly zenith dependent (steeply falling overhead, harder near the
    horizon where the parent mesons decay before interacting); an isotropic flux
    such as the astrophysical power law does not need it.

    No Earth attenuation is applied here: the published effective area already
    includes it for every direction.

    Parameters
    ----------
    aeff : softpaws.response.irfs.EffectiveArea
        Effective area for the same season as ``smearing``.
    smearing : softpaws.response.irfs.SmearingMatrix
        Smearing matrix for one season; its own (true-energy, declination)
        bin grid sets the integration grid.
    log10_e_reco_edges : np.ndarray, shape (n_reco + 1,)
        Reconstructed-energy bin edges in ``log10(E/GeV)``.
    dec_min : float
        Minimum declination [deg].
    dec_max : float
        Maximum declination [deg].
    flux_fn : callable
        Differential neutrino flux, ``flux_fn(E_nu_gev, dec_deg) -> dphi/dE`` in
        ``GeV^-1 cm^-2 s^-1 sr^-1``, called with an array of energies and one
        scalar declination at a time.
    livetime_s : float
        Exposure time [s].
    n_subdivisions : int, optional
        Number of log-spaced sample points per true-energy bin for the flux
        integral.

    Returns
    -------
    counts : np.ndarray, shape (n_reco,)
        Expected reconstructed-energy track counts per bin.
    """
    response = smearing.energy_response_matrix(log10_e_reco_edges)  # (n_enu, n_dec, n_reco)

    dec_centers = smearing.dec_centers
    dec_mask = (dec_centers >= dec_min) & (dec_centers <= dec_max)

    sin_dec_edges = np.sin(np.deg2rad(smearing.dec_edges))
    solid_angle_per_dec = 2.0 * np.pi * np.diff(sin_dec_edges)  # steradians

    enu_edges = smearing.log10_enu_edges
    enu_centers = smearing.log10_enu_centers

    n_reco = response.shape[2]
    counts = np.zeros(n_reco)

    for i, (lo, hi) in enumerate(zip(enu_edges[:-1], enu_edges[1:], strict=True)):
        energy = np.logspace(lo, hi, n_subdivisions)

        for j in np.nonzero(dec_mask)[0]:
            flux_integral = np.trapezoid(flux_fn(energy, float(dec_centers[j])), energy)
            aeff_val = float(aeff(enu_centers[i], dec_centers[j])[0])
            n_true = aeff_val * flux_integral * solid_angle_per_dec[j] * livetime_s
            counts += n_true * response[i, j, :]

    return counts


def soft_volume_smeared_counts(
    true_counts_fn: Callable[[np.ndarray], np.ndarray],
    smearing: SmearingMatrix,
    log10_e_reco_edges: np.ndarray,
    dec_min: float,
    dec_max: float,
) -> np.ndarray:
    """Migrate a soft-volume true-count spectrum through the IceCube smearing table.

    The soft-volume path predicts *muon energy at the detector* directly and, by
    default, is compared to data by using that energy as the reconstructed-energy
    proxy (see the module docstring) -- unlike the IRF path, it is never actually
    smeared. This applies the published :meth:`~softpaws.response.irfs.
    SmearingMatrix.energy_response_matrix` to the soft-volume prediction too, so
    the two paths are compared on the same reconstructed-energy footing.

    ``smearing``'s response matrix is defined on injected *neutrino* energy;
    treating the soft-volume model's muon-energy bins as if they were neutrino-
    energy bins is an approximation (E_mu ~ E_nu for a through-going track, but
    not exact), not a re-derivation of the smearing table for muon energy.

    Parameters
    ----------
    true_counts_fn : callable
        Given the smearing table's own true-energy bin edges
        (``smearing.log10_enu_edges``), returns the soft-volume expected counts
        in each of those bins over the declination band ``[dec_min, dec_max]``.
        Build this from :meth:`~softpaws.response.soft_volume.SoftVolumeResponse.
        expected_counts` (or ``expected_counts_attenuated``), optionally adding
        :func:`~softpaws.response.soft_volume.tau_induced_expected_counts`.
    smearing : softpaws.response.irfs.SmearingMatrix
        Smearing matrix for one season; its true-energy/declination bin grid
        sets the migration.
    log10_e_reco_edges : np.ndarray, shape (n_reco + 1,)
        Reconstructed-energy bin edges in ``log10(E/GeV)``, matching the
        binning used for :func:`observed_counts`.
    dec_min : float
        Minimum declination [deg].
    dec_max : float
        Maximum declination [deg].

    Returns
    -------
    counts : np.ndarray, shape (n_reco,)
        Smeared reconstructed-energy track counts per bin.
    """
    response = smearing.energy_response_matrix(log10_e_reco_edges)  # (n_enu, n_dec, n_reco)

    dec_centers = smearing.dec_centers
    dec_mask = (dec_centers >= dec_min) & (dec_centers <= dec_max)
    if not np.any(dec_mask):
        return np.zeros(response.shape[2])

    sin_dec_edges = np.sin(np.deg2rad(smearing.dec_edges))
    solid_angle_per_dec = 2.0 * np.pi * np.diff(sin_dec_edges)

    # The soft-volume rate does not depend on direction, so the band-averaged
    # response is a plain solid-angle average over the declination bins in band
    # (as in example 04's whole_sky_response), not an aeff-weighted one.
    mean_response = np.average(
        response[:, dec_mask, :], axis=1, weights=solid_angle_per_dec[dec_mask],
    )  # (n_enu, n_reco)

    true_counts = true_counts_fn(smearing.log10_enu_edges)  # (n_enu,)
    return true_counts @ mean_response


def fit_scale_factor(
    observed: np.ndarray,
    template: np.ndarray,
    log10_e_edges: np.ndarray,
    log10_e_min_fit: float = 4.0,
) -> float:
    """Single-parameter normalization that best matches a template to data.

    For a template that scales linearly with the flux normalization (as both
    forward models do) and Poisson-distributed observed counts with no
    background, the maximum-likelihood scale factor for bins in the fit range
    is the simple ratio of summed counts, ``sum(observed) / sum(template)``.

    Parameters
    ----------
    observed : np.ndarray, shape (n_bins,)
        Observed counts per bin (see :func:`observed_counts`).
    template : np.ndarray, shape (n_bins,)
        Predicted counts per bin at some reference flux normalization.
    log10_e_edges : np.ndarray, shape (n_bins + 1,)
        Bin edges in ``log10(E/GeV)``, used only to select the fit range.
    log10_e_min_fit : float, optional
        Lower edge of the fit range in ``log10(E/GeV)``. Defaults to 10 TeV,
        matching the IceCube diffuse-flux analysis (Palmisano et al.,
        arXiv:2607.13143, Section 4.1).

    Returns
    -------
    scale : float
        Best-fit multiplicative scale factor.

    Raises
    ------
    ValueError
        Raised if the template sums to zero or less in the fit range.
    """
    centers = 0.5 * (log10_e_edges[:-1] + log10_e_edges[1:])
    mask = centers >= log10_e_min_fit

    template_sum = template[mask].sum()
    if template_sum <= 0:
        raise ValueError("Template sums to zero or less in the fit range; cannot fit a scale.")

    return float(observed[mask].sum() / template_sum)


def fit_scale_factor_with_background(
    observed: np.ndarray,
    template: np.ndarray,
    background: np.ndarray,
    log10_e_edges: np.ndarray,
    log10_e_min_fit: float = 4.0,
) -> float:
    r"""Signal normalization on top of a fixed background prediction.

    The background-free :func:`fit_scale_factor` has a closed-form solution; with
    an additive background held at its predicted normalization the Poisson
    maximum likelihood for the signal scale ``k`` is instead the root of

    .. math:: \\sum_i s_i\\left(\\frac{o_i}{k\\,s_i + b_i} - 1\\right) = 0,

    which is monotonically decreasing in ``k`` and is solved here by bisection.
    Use this when the fit range reaches down into the background-dominated part
    of the spectrum, where :func:`fit_scale_factor` would absorb the background
    into the signal normalization.

    Parameters
    ----------
    observed : np.ndarray, shape (n_bins,)
        Observed counts per bin (see :func:`observed_counts`).
    template : np.ndarray, shape (n_bins,)
        Predicted signal counts per bin at some reference flux normalization.
    background : np.ndarray, shape (n_bins,)
        Predicted background counts per bin, held fixed.
    log10_e_edges : np.ndarray, shape (n_bins + 1,)
        Bin edges in ``log10(E/GeV)``, used only to select the fit range.
    log10_e_min_fit : float, optional
        Lower edge of the fit range in ``log10(E/GeV)``.

    Returns
    -------
    scale : float
        Best-fit multiplicative scale factor on ``template``. Zero if the
        background alone already over-predicts the data everywhere in the fit
        range.

    Raises
    ------
    ValueError
        Raised if the template sums to zero or less in the fit range.
    """
    centers = 0.5 * (log10_e_edges[:-1] + log10_e_edges[1:])
    mask = centers >= log10_e_min_fit

    signal = np.asarray(template, dtype=float)[mask]
    bkg = np.asarray(background, dtype=float)[mask]
    data = np.asarray(observed, dtype=float)[mask]

    if signal.sum() <= 0:
        raise ValueError("Template sums to zero or less in the fit range; cannot fit a scale.")

    def score(scale: float) -> float:
        predicted = scale * signal + bkg
        # A bin predicting nothing where something was seen pulls the scale up
        # without bound; one predicting nothing where nothing was seen is inert.
        if np.any((predicted <= 0.0) & (data > 0.0)):
            return np.inf
        safe = np.where(predicted > 0.0, predicted, 1.0)
        return float(np.sum(signal * (np.where(predicted > 0.0, data / safe, 0.0) - 1.0)))

    if score(0.0) <= 0.0:
        return 0.0

    # Dropping the background can only raise the fitted scale, so the
    # background-free solution of fit_scale_factor brackets the root from above.
    upper = float(data.sum() / signal.sum())
    lower = 0.0
    for _ in range(_BISECTION_STEPS):
        middle = 0.5 * (lower + upper)
        if score(middle) > 0.0:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def fit_component_scales(
    observed: np.ndarray,
    templates: Sequence[np.ndarray],
    log10_e_edges: np.ndarray,
    log10_e_min_fit: float = 4.0,
    n_iterations: int = 500,
    tol: float = 1e-12,
) -> np.ndarray:
    r"""Joint Poisson maximum-likelihood normalizations of several templates.

    Generalizes :func:`fit_scale_factor` to a prediction that is a sum of
    independently normalized components, ``N_i = sum_c k_c t_{ci}`` -- an
    astrophysical signal plus an atmospheric background whose normalization is
    left free, say, as an analysis does when the background model carries its own
    (large) flux uncertainty. Unlike :func:`fit_scale_factor_with_background`,
    which holds the background at its predicted normalization, every component
    here floats.

    Solved by the expectation-maximization iteration for Poisson mixtures,

    .. math:: k_c \\leftarrow k_c\\,\\frac{\\sum_i o_i t_{ci} / N_i}
        {\\sum_i t_{ci}},

    whose fixed point satisfies the likelihood equations for all components at
    once and which never leaves the physical region ``k_c >= 0``.

    Parameters
    ----------
    observed : np.ndarray, shape (n_bins,)
        Observed counts per bin (see :func:`observed_counts`).
    templates : sequence of np.ndarray
        Predicted counts per bin for each component, at some reference
        normalization. All of shape ``(n_bins,)``.
    log10_e_edges : np.ndarray, shape (n_bins + 1,)
        Bin edges in ``log10(E/GeV)``, used only to select the fit range.
    log10_e_min_fit : float, optional
        Lower edge of the fit range in ``log10(E/GeV)``.
    n_iterations : int, optional
        Maximum number of iterations.
    tol : float, optional
        Relative change in every scale below which the iteration stops.

    Returns
    -------
    scales : np.ndarray, shape (n_components,)
        Best-fit multiplicative scale factor for each template.

    Raises
    ------
    ValueError
        Raised if any template sums to zero or less in the fit range.
    """
    centers = 0.5 * (log10_e_edges[:-1] + log10_e_edges[1:])
    mask = centers >= log10_e_min_fit

    stack = np.array([np.asarray(t, dtype=float)[mask] for t in templates])  # (n_comp, n_fit)
    data = np.asarray(observed, dtype=float)[mask]

    totals = stack.sum(axis=1)
    if np.any(totals <= 0.0):
        raise ValueError("Every template must sum to more than zero in the fit range.")

    scales = np.full(stack.shape[0], data.sum() / (stack.shape[0] * totals))
    for _ in range(n_iterations):
        predicted = scales @ stack
        safe = np.where(predicted > 0.0, predicted, 1.0)
        weights = np.where(predicted > 0.0, data / safe, 0.0)
        updated = scales * (stack @ weights) / totals
        if np.all(np.abs(updated - scales) <= tol * np.maximum(scales, 1.0)):
            scales = updated
            break
        scales = updated
    return scales


def implied_efficiency(
    observed: np.ndarray,
    soft_template: np.ndarray,
    irf_template: np.ndarray,
    log10_e_edges: np.ndarray,
    log10_e_min_fit: float = 4.0,
) -> float:
    """Efficiency implied by comparing the idealized and IRF fit normalizations.

    Both templates are evaluated at the same reference flux normalization, so
    their best-fit scale factors (:func:`fit_scale_factor`) are directly
    comparable. The ratio ``scale_soft / scale_irf`` is the flux normalization
    the idealized soft-volume detector needs, relative to what the
    efficiency-and-acceptance-aware IRF path needs, to describe the same data.
    This is the data-driven analogue of the efficiency factor
    ``eps_IC-TG = phi0 / phi0_IceCube`` defined in Palmisano et al.,
    arXiv:2607.13143, Eq. 1.4.

    Parameters
    ----------
    observed : np.ndarray, shape (n_bins,)
        Observed counts per bin (see :func:`observed_counts`).
    soft_template : np.ndarray, shape (n_bins,)
        Soft-volume predicted counts at the reference flux normalization.
    irf_template : np.ndarray, shape (n_bins,)
        IRF predicted counts at the same reference flux normalization.
    log10_e_edges : np.ndarray, shape (n_bins + 1,)
        Bin edges in ``log10(E/GeV)``, used only to select the fit range.
    log10_e_min_fit : float, optional
        Lower edge of the fit range in ``log10(E/GeV)``.

    Returns
    -------
    efficiency : float
        Implied efficiency ``scale_soft / scale_irf``.
    """
    scale_soft = fit_scale_factor(observed, soft_template, log10_e_edges, log10_e_min_fit)
    scale_irf = fit_scale_factor(observed, irf_template, log10_e_edges, log10_e_min_fit)
    return scale_soft / scale_irf
