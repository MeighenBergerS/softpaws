"""softpaws — a first-principles forward model for UHE neutrino telescopes.

Implements the analytic muon transport of Meighen-Berger, *Analytical
High-Energy Muon Transport for Neutrino Telescopes* (2026). The locally
scale-invariant collision operator is diagonalized by power laws, so a muon's
whole loss history collapses to a single transport exponent ``Phi(s)``, which
then serves as a first-passage generator for the range of the muon and for the
detector response built on it. The result is benchmarked against the published
effective areas of IceCube, KM3NeT/ARCA, P-ONE and TRIDENT, against IceCube's
own instrument response functions (effective area + smearing matrix), and
against IceCube data.

Subpackages
-----------
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
utils
    Physical constants, unit conversions, and shared helpers.
"""

__version__ = "0.1.0"
