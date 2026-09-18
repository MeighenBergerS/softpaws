"""Loaders for the IceCube IceTracks-DR2 release and the published curves.

Reads reconstructed muon-track events, binned effective areas, and smearing
matrices from the data release (DOI 10.7910/DVN/MMIIZA; arXiv:2605.19040). The
raw files live under ``src/softpaws/data/dataverse_files/`` and are not
committed here (see ``.gitignore``). :mod:`softpaws.data.published` reads the
effective-area tables of the other detectors that ship with the package.
"""

from .container import EventSet
from .icecube import (
    IC86_SEASONS,
    banded_effective_area,
    hemisphere_average,
    irf_season,
    livetime_weighted_effective_area,
    load_effective_area,
    load_events,
    load_psf_table,
    total_livetime_s,
)
from .loader import (
    compute_livetime_s,
    load_irfs,
    load_season,
    load_uptime,
)
from .paths import DATA_DIR_VARIABLE, DR2_DOI, data_root, dr2_dir, hese_dir
from .published import (
    PONE_ZENITH_BANDS_DEG,
    TRIDENT_COS_BANDS,
    arca21_bright_track_aeff,
    arca230_angle_dependent_aeff,
    arca230_quoted_fit,
    arca230_trigger_level_aeff,
    icecube_dr2_aeff,
    icecube_point_source_sensitivity,
    interpolate_aeff,
    pone_allsky_aeff,
    pone_band_aeff,
    trident_2025_map,
    trident_band_aeff,
)
from .schema import EVENTS_DTYPE, SEASONS

__all__ = [
    "DATA_DIR_VARIABLE",
    "DR2_DOI",
    "data_root",
    "dr2_dir",
    "hese_dir",
    "EVENTS_DTYPE",
    "IC86_SEASONS",
    "PONE_ZENITH_BANDS_DEG",
    "SEASONS",
    "TRIDENT_COS_BANDS",
    "EventSet",
    "arca21_bright_track_aeff",
    "arca230_angle_dependent_aeff",
    "arca230_quoted_fit",
    "arca230_trigger_level_aeff",
    "compute_livetime_s",
    "banded_effective_area",
    "hemisphere_average",
    "icecube_dr2_aeff",
    "icecube_point_source_sensitivity",
    "interpolate_aeff",
    "irf_season",
    "livetime_weighted_effective_area",
    "load_effective_area",
    "load_events",
    "load_psf_table",
    "load_irfs",
    "load_season",
    "load_uptime",
    "pone_allsky_aeff",
    "pone_band_aeff",
    "total_livetime_s",
    "trident_2025_map",
    "trident_band_aeff",
]
