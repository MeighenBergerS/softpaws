"""Benchmarks comparing the soft-volume and IRF forward models.

Tools to place the soft-volume predictions (:mod:`softpaws.response`) and the
published-IRF predictions side by side and validate both against the observed
IceCube DR2 event distributions.
"""

from .rates import (
    fit_scale_factor,
    implied_efficiency,
    irf_expected_counts,
    observed_counts,
)

__all__ = [
    "fit_scale_factor",
    "implied_efficiency",
    "irf_expected_counts",
    "observed_counts",
]
