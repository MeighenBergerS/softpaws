"""Loaders for the IceCube IceTracks-DR2 release.

Reads reconstructed muon-track events, binned effective areas, and smearing
matrices from the data release (DOI 10.7910/DVN/MMIIZA; arXiv:2605.19040). The
raw files live under ``src/softpaws/data/dataverse_files/`` and are not
committed here (see ``.gitignore``).
"""

from .container import EventSet
from .icecube import (
    IC86_SEASONS,
    hemisphere_average,
    irf_season,
    livetime_weighted_effective_area,
    load_effective_area,
    load_events,
    season_livetime_s,
    total_livetime_s,
)
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
    "IC86_SEASONS",
    "SEASONS",
    "EventSet",
    "compute_livetime_s",
    "hemisphere_average",
    "irf_season",
    "livetime_weighted_effective_area",
    "load_effective_area",
    "load_events",
    "load_all_seasons",
    "load_irfs",
    "load_season",
    "load_uptime",
    "season_livetime_s",
    "total_livetime_s",
]
