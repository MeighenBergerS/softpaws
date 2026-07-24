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
