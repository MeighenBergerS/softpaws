"""High-energy muon transport.

Two treatments of the QED collision operator:

- the **eigenvalue** treatment (``eigenvalue``, ``soft_volume_exact``), which
  diagonalizes the collision operator with power laws under the assumption
  that the loss kernel is locally scale invariant. It gives the transport
  exponent ``Phi(s)`` with every loss moment kept, and with no small-``y``
  expansion and no energy cutoff. This is the physics core, described in
  Meighen-Berger, *Analytical High-Energy Muon Transport for Neutrino
  Telescopes* (2026); see ``docs/theory/exact_soft_volume.md``.
- the second-order **drift-diffusion** expansion (``coefficients``,
  ``soft_volume`` drift form), where soft energy losses dominate and rare hard
  scatters are perturbative. It is the earlier treatment of Palmisano,
  Redigolo, Tammaro and Tesi (arXiv:2607.13143), recovered here as the first
  two terms of ``Phi(s)`` and kept as a cross-check; see
  ``docs/theory/soft_volume.md``.

The range of a muon to a threshold, the first-passage generator built on
``Phi(s)``, lives in ``muon_range``; the effective areas of
:mod:`softpaws.response` are built on it. Both soft-volume treatments feed
the same master formula and are exposed through interchangeable
:mod:`softpaws.response` predictors.
"""

from . import (
    attenuation,
    coefficients,
    cross_section,
    earth,
    eigenvalue,
    loss_distribution,
    loss_ensemble,
    muon_range,
    soft_volume,
    source,
    tau,
)
from .coefficients import (
    KernelScaling,
    diffusion_coefficient,
    drift_coefficient,
    log_loss_moments,
    scaled_kernel,
    set_kernel_scaling,
    third_moment_coefficient,
)
from .cross_section import CrossSection, PowerLawCrossSection, bgr18_cross_section
from .earth import neutrino_column_g_cm2, overburden_km, prem_column, zenith_grid
from .eigenvalue import phi_eigenvalue, phi_eigenvalue_at_energy, spectral_index
from .loss_distribution import loss_density, loss_density_three_moment
from .muon_range import (
    DEFAULT_MUON_THRESHOLD_GEV,
    muon_range_km,
    stochastic_muon_range_km,
    truncated_muon_range_km,
    two_medium_muon_range_km,
)

__all__ = [
    "CrossSection",
    "DEFAULT_MUON_THRESHOLD_GEV",
    "KernelScaling",
    "PowerLawCrossSection",
    "attenuation",
    "bgr18_cross_section",
    "coefficients",
    "cross_section",
    "diffusion_coefficient",
    "drift_coefficient",
    "earth",
    "eigenvalue",
    "log_loss_moments",
    "loss_distribution",
    "loss_ensemble",
    "muon_range",
    "muon_range_km",
    "neutrino_column_g_cm2",
    "overburden_km",
    "phi_eigenvalue",
    "phi_eigenvalue_at_energy",
    "prem_column",
    "scaled_kernel",
    "set_kernel_scaling",
    "soft_volume",
    "source",
    "spectral_index",
    "stochastic_muon_range_km",
    "tau",
    "third_moment_coefficient",
    "truncated_muon_range_km",
    "two_medium_muon_range_km",
    "zenith_grid",
]
