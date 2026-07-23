"""Loaders for the IceCube IceTracks-DR2 release.

Reads reconstructed muon-track events, binned effective areas, and smearing
matrices from the data release (DOI 10.7910/DVN/MMIIZA; arXiv:2605.19040). The
raw files live under ``src/softpaws/data/dataverse_files/`` and are not
committed here (see ``.gitignore``).
"""

from .container import EventSet
from .loader import (
    compute_livetime_s,
    load_all_seasons,
    load_irfs,
    load_season,
    load_uptime,
)
from .schema import EVENTS_DTYPE, SEASONS

__all__ = [
    "EVENTS_DTYPE",
    "SEASONS",
    "EventSet",
    "compute_livetime_s",
    "load_all_seasons",
    "load_irfs",
    "load_season",
    "load_uptime",
]
