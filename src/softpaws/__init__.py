"""softpaws — a soft-volume forward model for UHE neutrino telescopes.

Implements the drift-diffusion (soft-volume) muon-transport response of
Palmisano, *The soft volume of ultra-high energy neutrinos experiments*
(arXiv:2607.13143), and benchmarks it against the published IceCube instrument
response functions (effective area + smearing matrix) and IceCube data.

Subpackages
-----------
data
    Load IceCube DR2 events, effective areas, and smearing matrices.
transport
    Soft-volume drift-diffusion muon transport.
response
    Forward models mapping a neutrino flux to an event rate.
comparison
    Compare the soft-volume and IRF forward models against data.
utils
    Physical constants, unit conversions, and shared helpers.
"""

__version__ = "0.1.0"
