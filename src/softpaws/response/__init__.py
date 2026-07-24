"""Forward models mapping a neutrino flux to an observed event rate.

Provides two interchangeable paths for comparison:

- the soft-volume model built from :mod:`softpaws.transport`, and
- the published IceCube IRF model (effective area convolved with the energy and
  angular smearing matrices).
"""

from .irfs import EffectiveArea, PointSpreadFunction, SmearingMatrix
from .soft_volume import SoftVolumeResponse, power_law_flux

__all__ = [
    "EffectiveArea",
    "PointSpreadFunction",
    "SmearingMatrix",
    "SoftVolumeResponse",
    "power_law_flux",
]
