"""Column names and dtypes for IceTracks-DR2 data files.

Notes
-----
Field names follow the IceTracks-DR2 data release conventions.
See DOI 10.7910/DVN/MMIIZA and arXiv:2605.19040 for the full specification.

Raw CSV column names (with units in brackets) are mapped to short internal
field names via ``CSV_TO_FIELD``.
"""

import numpy as np

# Mapping from raw CSV header names to internal structured-array field names.
CSV_TO_FIELD: dict[str, str] = {
    "run":          "run",
    "event":        "event",
    "subevent":     "subevent",
    "MJD[days]":    "time",
    "log10(E/GeV)": "log10_energy",
    "AngErr[deg]":  "sigma",
    "RA[deg]":      "ra",
    "Dec[deg]":     "dec",
    "Azimuth[deg]": "azimuth",
    "Zenith[deg]":  "zenith",
}

EVENTS_DTYPE = np.dtype([
    ("run",          np.int64),
    ("event",        np.int64),
    ("subevent",     np.int32),
    ("time",         np.float64),   # MJD arrival time [days]
    ("log10_energy", np.float64),   # log10 of reconstructed muon energy [GeV]
    ("sigma",        np.float64),   # Angular uncertainty estimate [deg]
    ("ra",           np.float64),   # Right ascension [deg, J2000]
    ("dec",          np.float64),   # Declination [deg, J2000]
    ("azimuth",      np.float64),   # Local azimuth [deg]
    ("zenith",       np.float64),   # Local zenith [deg]
])

# Season labels in chronological order.
SEASONS: tuple[str, ...] = (
    "IC40", "IC59", "IC79",
    "IC86_I", "IC86_II", "IC86_III", "IC86_IV",
    "IC86_V", "IC86_VI", "IC86_VII", "IC86_VIII",
    "IC86_IX", "IC86_X", "IC86_XI",
)
