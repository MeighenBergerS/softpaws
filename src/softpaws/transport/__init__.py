"""Soft-volume muon transport.

Two treatments of the QED collision operator from Palmisano, *The soft volume of
ultra-high energy neutrinos experiments* (arXiv:2607.13143):

- the paper's second-order **drift-diffusion** expansion (``coefficients``,
  ``soft_volume`` drift form), where soft energy losses dominate and rare hard
  scatters are perturbative; and
- the **exact eigenvalue** treatment (``eigenvalue``, ``soft_volume_exact``),
  which diagonalises the exact collision operator with power laws, giving the
  attenuation constant ``Phi(A)`` without any small-``y`` expansion or energy
  cutoff. See ``docs/exact_soft_volume_notes.md``.

Both feed the same soft-volume master formula and are exposed through
interchangeable :mod:`softpaws.response` predictors.
"""

from . import (
    attenuation,
    coefficients,
    cross_section,
    earth,
    eigenvalue,
    loss_distribution,
    loss_ensemble,
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
from .soft_volume import (
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
