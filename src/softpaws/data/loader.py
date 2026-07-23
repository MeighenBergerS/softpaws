"""Load IceTracks-DR2 event and IRF files from the local data directory.

Notes
-----
All data files are whitespace-delimited CSVs with a single ``#``-prefixed
header line. The standard layout expected here is::

    <data_dir>/
        events/   IC40_exp.csv, IC59_exp.csv, ...
        irfs/     IC40_effectiveArea.csv, IC86_smearing.csv, ...
        uptime/   IC40_exp.csv, ...

The canonical ``data_dir`` for this project is
``src/softpaws/data/dataverse_files/``.
"""

import pathlib
import re

import numpy as np

from ..response.irfs import EffectiveArea, SmearingMatrix
from .schema import CSV_TO_FIELD, EVENTS_DTYPE

_SEASON_FILE_PATTERN = re.compile(r"^(IC\d+(?:_[IVX]+)?)_exp\.csv$")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_all_seasons(data_dir: str | pathlib.Path) -> np.ndarray:
    """Load and concatenate events from all seasons in ``data_dir``.

    Parameters
    ----------
    data_dir : str or pathlib.Path
        Root of the DR2 data directory, expected to contain an ``events/``
        subdirectory with one CSV file per IceCube season.

    Returns
    -------
    events : np.ndarray
        Structured array with dtype ``EVENTS_DTYPE`` containing every
        event across all seasons, sorted by MJD arrival time.

    Raises
    ------
    FileNotFoundError
        Raised if ``data_dir/events/`` does not exist or contains no
        recognisable season files.
    """
    events_dir = pathlib.Path(data_dir) / "events"
    if not events_dir.is_dir():
        raise FileNotFoundError(f"Events directory not found: {events_dir}")

    season_files = sorted(
        p for p in events_dir.glob("*_exp.csv")
        if _SEASON_FILE_PATTERN.match(p.name)
    )
    if not season_files:
        raise FileNotFoundError(f"No season files found in {events_dir}")

    chunks = [load_season(p) for p in season_files]
    combined = np.concatenate(chunks)
    combined.sort(order="time")
    return combined


def load_season(path: str | pathlib.Path) -> np.ndarray:
    """Load a single IceTracks-DR2 season CSV file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a season events CSV (e.g. ``IC86_I_exp.csv``).

    Returns
    -------
    events : np.ndarray
        Structured array with dtype ``EVENTS_DTYPE``.

    Raises
    ------
    FileNotFoundError
        Raised if ``path`` does not exist.
    ValueError
        Raised if required columns are absent from the file header.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    csv_cols = _read_header(path)
    raw = np.genfromtxt(path, comments="#")

    if raw.ndim == 1:
        raw = raw[np.newaxis, :]

    col_idx = {name: i for i, name in enumerate(csv_cols)}
    missing = [c for c in CSV_TO_FIELD if c not in col_idx]
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}")

    n = len(raw)
    out = np.empty(n, dtype=EVENTS_DTYPE)
    for csv_name, field_name in CSV_TO_FIELD.items():
        out[field_name] = raw[:, col_idx[csv_name]]

    return out


def load_irfs(
    irf_dir: str | pathlib.Path,
    season: str,
    uptime_dir: str | pathlib.Path | None = None,
) -> dict:
    """Load and parse IRFs for a given IceCube season.

    Parameters
    ----------
    irf_dir : str or pathlib.Path
        Path to the ``irfs/`` subdirectory.
    season : str
        Season label, e.g. ``"IC86_I"``.  IC86-II and later share the
        IC86 IRF files, so any ``IC86_*`` label maps to ``IC86``.
    uptime_dir : str or pathlib.Path, optional
        Path to the ``uptime/`` subdirectory.  When provided the livetime
        is computed and stored under the ``"livetime_s"`` key.

    Returns
    -------
    irfs : dict
        ``"aeff"`` → :class:`~softpaws.response.irfs.EffectiveArea`
        ``"smearing"`` → :class:`~softpaws.response.irfs.SmearingMatrix`
        ``"livetime_s"`` → float (seconds), only present when *uptime_dir* given.

    Raises
    ------
    FileNotFoundError
        Raised if expected IRF files are not found.

    Notes
    -----
    Loading and parsing the smearing matrix (~570 MB, 3.36 M rows) takes
    roughly 30 s on a typical workstation.  The result is not cached; keep
    the returned dict alive for the duration of your analysis.
    """
    irf_dir = pathlib.Path(irf_dir)
    irf_season = _canonical_irf_season(season)

    aeff_path = irf_dir / f"{irf_season}_effectiveArea.csv"
    smear_path = irf_dir / f"{irf_season}_smearing.csv"

    for p in (aeff_path, smear_path):
        if not p.exists():
            raise FileNotFoundError(p)

    raw_aeff = np.genfromtxt(aeff_path, comments="#")
    raw_smear = np.loadtxt(smear_path, comments="#")

    result: dict = {
        "aeff": parse_aeff(raw_aeff),
        "smearing": parse_smearing(raw_smear),
    }

    if uptime_dir is not None:
        uptime_path = pathlib.Path(uptime_dir) / f"{season}_exp.csv"
        if uptime_path.exists():
            uptime = load_uptime(uptime_path)
            result["livetime_s"] = compute_livetime_s(uptime)
        else:
            raise FileNotFoundError(uptime_path)

    return result


# ---------------------------------------------------------------------------
# IRF parsers
# ---------------------------------------------------------------------------


def parse_aeff(raw: np.ndarray) -> EffectiveArea:
    """Build an :class:`EffectiveArea` from the raw 5-column CSV array.

    Parameters
    ----------
    raw : np.ndarray
        Shape ``(N, 5)``.  Columns: log10(E_nu/GeV)_min/max,
        Dec_nu_min/max [deg], A_Eff [cm²].

    Returns
    -------
    EffectiveArea
        Interpolator in (log10_E_nu, sin(dec)) space with values in cm².
    """
    # Unique bin edges from the min/max columns
    log10_e_edges = np.unique(np.concatenate([raw[:, 0], raw[:, 1]]))
    dec_edges_deg = np.unique(np.concatenate([raw[:, 2], raw[:, 3]]))
    sin_dec_edges = np.sin(np.deg2rad(dec_edges_deg))

    n_e = len(log10_e_edges) - 1
    n_d = len(dec_edges_deg) - 1

    # Map each row to (i_e, i_d) bin indices
    i_e = np.searchsorted(log10_e_edges[:-1], raw[:, 0], side="left")
    i_d = np.searchsorted(dec_edges_deg[:-1], raw[:, 2], side="left")

    values = np.zeros((n_e, n_d))
    for row, ie, id_ in zip(raw, i_e, i_d):
        values[ie, id_] = row[4]

    return EffectiveArea(log10_e_edges, sin_dec_edges, values)


def parse_smearing(raw: np.ndarray) -> SmearingMatrix:
    """Build a :class:`SmearingMatrix` from the raw 11-column CSV array.

    Parameters
    ----------
    raw : np.ndarray
        Shape ``(N, 11)``.  Columns: log10(E_nu)_min/max, dec_min/max [deg],
        log10(E_reco)_min/max, PSF_min/max [deg],
        AngErr_min/max [deg], Fractional_Counts.

    Returns
    -------
    SmearingMatrix
    """
    return SmearingMatrix(raw)


# ---------------------------------------------------------------------------
# Uptime / livetime
# ---------------------------------------------------------------------------


def load_uptime(path: str | pathlib.Path) -> np.ndarray:
    """Load a good-run (uptime) list CSV.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to an uptime CSV with columns ``MJD_start`` and ``MJD_stop``.

    Returns
    -------
    uptime : np.ndarray
        Shape ``(N, 2)``, columns [MJD_start, MJD_stop].
    """
    return np.genfromtxt(path, comments="#")


def compute_livetime_s(uptime: np.ndarray) -> float:
    """Sum the duration of good-run intervals in seconds.

    Parameters
    ----------
    uptime : np.ndarray
        Shape ``(N, 2)`` with columns [MJD_start, MJD_stop].

    Returns
    -------
    float
        Total livetime in seconds.
    """
    durations_mjd = uptime[:, 1] - uptime[:, 0]
    return float(durations_mjd.sum()) * 86400.0


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _read_header(path: pathlib.Path) -> list[str]:
    with open(path) as fh:
        line = fh.readline().lstrip("# ").rstrip()
    return line.split()


def _canonical_irf_season(season: str) -> str:
    if season.startswith("IC86"):
        return "IC86"
    return season
