"""Published detector layouts.

A :class:`Site` is the geometry, medium, depth and latitude of one neutrino
telescope, and an :class:`Optics` record is its medium clarity and optical
module. The constants are the layouts the paper compares, and :data:`SITES`
collects them by name.

Examples
--------
>>> from softpaws.detectors import ICECUBE, get_site
>>> round(ICECUBE.detector_volume_km3(), 6)
1.0
>>> get_site("ARCA230").n_blocks
2
"""

from .optics import ANCHOR_NM, ARCA_OPTICS, DEFAULT_MIN_TRACK_KM, ICECUBE_OPTICS, Optics
from .sites import (
    ARCA21,
    ARCA230,
    GEN2,
    GVD,
    ICECUBE,
    MAX_UPSTREAM_KM,
    PONE,
    SITES,
    TRIDENT,
    TRIDENT_2025,
    Site,
    get_site,
)

__all__ = [
    "ANCHOR_NM",
    "ARCA21",
    "ARCA230",
    "ARCA_OPTICS",
    "DEFAULT_MIN_TRACK_KM",
    "GEN2",
    "GVD",
    "ICECUBE",
    "ICECUBE_OPTICS",
    "MAX_UPSTREAM_KM",
    "Optics",
    "PONE",
    "SITES",
    "Site",
    "TRIDENT",
    "TRIDENT_2025",
    "get_site",
]
