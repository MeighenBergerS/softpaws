"""Benchmarks comparing the soft-volume and IRF forward models.

Tools to place the soft-volume predictions (:mod:`softpaws.response`) and the
published-IRF predictions side by side and validate both against the observed
IceCube DR2 event distributions.
"""

from .event_energy import (
    KM3_230213A,
    TrackEvent,
    energy_likelihood,
    energy_posterior,
    parent_energy_posterior,
    potential_density,
)
from .events import (
    astro_flux,
    combine_band,
    predict,
    published_response,
)
from .feldman_cousins import cached_toys, calibrated_thresholds, profile_interval, toy_statistics
from .posterior import (
    credible_interval,
    global_compatibility,
    leave_one_out_compatibility,
    marginal_summary,
    pairwise_compatibility,
    product_posterior,
    sample_posterior,
)
from .rates import (
    fit_scale_factor,
    implied_efficiency,
    irf_expected_counts,
    observed_counts,
)
from .reco_likelihood import (
    RecoLikelihood,
    atmospheric_fluxes,
    banded_responses,
    binned_events,
    fit_inputs,
    physical_r_range,
    smearing_marginal,
    true_counts,
)

__all__ = [
    "KM3_230213A",
    "TrackEvent",
    "cached_toys",
    "calibrated_thresholds",
    "credible_interval",
    "energy_likelihood",
    "energy_posterior",
    "global_compatibility",
    "leave_one_out_compatibility",
    "marginal_summary",
    "pairwise_compatibility",
    "parent_energy_posterior",
    "potential_density",
    "product_posterior",
    "profile_interval",
    "sample_posterior",
    "toy_statistics",
    "RecoLikelihood",
    "astro_flux",
    "atmospheric_fluxes",
    "banded_responses",
    "binned_events",
    "combine_band",
    "fit_inputs",
    "fit_scale_factor",
    "implied_efficiency",
    "irf_expected_counts",
    "observed_counts",
    "physical_r_range",
    "predict",
    "published_response",
    "smearing_marginal",
    "true_counts",
]
