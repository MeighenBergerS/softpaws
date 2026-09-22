"""softpaws — a first-principles forward model for UHE neutrino telescopes.

Implements the analytic muon transport of Meighen-Berger, *Estimating
High-Energy Neutrino Effective Areas from Muon Propagation* (2026). The locally
scale-invariant collision operator is diagonalized by power laws, so a muon's
whole loss history collapses to a single transport exponent ``Phi(s)``, which
then serves as a first-passage generator for the range of the muon and for the
detector response built on it. The result is benchmarked against the published
effective areas of IceCube, KM3NeT/ARCA, P-ONE and TRIDENT, against IceCube's
own instrument response functions (effective area + smearing matrix), and
against IceCube data.

Modules
-------
constants
    Units (the code computes in GeV, cm, s and rad), physical constants and defaults.
data
    Load the IceCube releases and the published curves of the other detectors.
detectors
    Published geometry, medium, and optics for each site.
fluxes
    Power laws, the published IceCube fits, and the atmospheric background.
transport
    The loss kernel, the transport exponent, ranges, Earth, and the tau channel.
response
    Forward models mapping a neutrino flux to an effective area or event rate.
comparison
    Likelihoods, the event benchmark, and single-track energy reconstruction.
"""

__version__ = "1.0.0"
