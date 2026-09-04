"""IceCube DR2 conveniences: seasons, livetimes, and combined effective areas.

The IceTracks-DR2 release ships one effective-area table per detector
configuration and one event and uptime file per season. The helpers here
answer the questions every analysis asks first: which seasons share a
response, how much livetime each one carries, and what the livetime-weighted
effective area looks like over one hemisphere on a common energy grid.
"""

from __future__ import annotations

import functools
import pathlib

import numpy as np

from softpaws.response.irfs import EffectiveArea

from .container import EventSet
from .loader import (
    _canonical_irf_season,
    compute_livetime_s,
    load_season,
    load_uptime,
    parse_aeff,
)
from .paths import dr2_dir, require
from .schema import SEASONS

__all__ = [
    "IC86_SEASONS",
    "hemisphere_average",
    "irf_season",
    "livetime_weighted_effective_area",
    "load_effective_area",
    "load_events",
    "season_livetime_s",
    "total_livetime_s",
]

#: The eleven IC86 seasons, which share one instrument response.
IC86_SEASONS = tuple(s for s in SEASONS if s.startswith("IC86"))


def _root(data_dir: str | pathlib.Path | None) -> pathlib.Path:
    """The release root, defaulting to :func:`softpaws.data.paths.dr2_dir`."""
    return dr2_dir() if data_dir is None else pathlib.Path(data_dir)


def irf_season(season: str) -> str:
    """Name of the response tables a season uses.

    IC86-II and later share the IC86 files, so every ``IC86_*`` label maps to
    ``"IC86"``; the three partial-detector seasons keep their own.

    Parameters
    ----------
    season : str
        Season label, for example ``"IC86_IV"``.

    Returns
    -------
    name : str
        File stem of the response tables, for example ``"IC86"``.

    Examples
    --------
    >>> irf_season("IC86_IV")
    'IC86'
    >>> irf_season("IC79")
    'IC79'
    """
    return _canonical_irf_season(season)


@functools.lru_cache(maxsize=None)
def _cached_effective_area(irf_path: str) -> EffectiveArea:
    return parse_aeff(np.genfromtxt(irf_path, comments="#"))


def load_effective_area(data_dir: str | pathlib.Path | None, season: str) -> EffectiveArea:
    """Effective-area table of one season, parsed once and cached.

    Only the effective-area file is read. The smearing table of the same
    season is ~600 MB and is left to :func:`softpaws.data.load_irfs`.

    Parameters
    ----------
    data_dir : str or pathlib.Path or None
        Root of the DR2 release, holding ``irfs/`` and ``uptime/``; ``None``
        uses :func:`softpaws.data.paths.dr2_dir`.
    season : str
        Season label; see :func:`irf_season`.

    Returns
    -------
    aeff : EffectiveArea
        The parsed table.

    Raises
    ------
    FileNotFoundError
        Raised if the season's effective-area file is missing.
    """
    path = _root(data_dir) / "irfs" / f"{irf_season(season)}_effectiveArea.csv"
    require(path, "IceTracks-DR2 release")
    return _cached_effective_area(str(path.resolve()))


def season_livetime_s(data_dir: str | pathlib.Path | None, season: str) -> float:
    """Good-run livetime of one season [s].

    Parameters
    ----------
    data_dir : str or pathlib.Path or None
        Root of the DR2 release; ``None`` uses :func:`softpaws.data.paths.dr2_dir`.
    season : str
        Season label.

    Returns
    -------
    livetime_s : float
        Livetime summed over the season's good-run windows [s].
    """
    path = _root(data_dir) / "uptime" / f"{season}_exp.csv"
    return compute_livetime_s(load_uptime(require(path, "IceTracks-DR2 release")))


def total_livetime_s(
    data_dir: str | pathlib.Path | None = None, seasons: tuple[str, ...] = SEASONS
) -> float:
    """Good-run livetime summed over several seasons [s].

    Parameters
    ----------
    data_dir : str or pathlib.Path or None
        Root of the DR2 release; ``None`` uses :func:`softpaws.data.paths.dr2_dir`.
    seasons : tuple of str, optional
        Seasons to sum. Defaults to the whole release.

    Returns
    -------
    livetime_s : float
        Total livetime [s].
    """
    return float(sum(season_livetime_s(data_dir, s) for s in seasons))


def hemisphere_average(aeff: EffectiveArea, hemisphere: str = "upgoing") -> np.ndarray:
    """Solid-angle-weighted average of a table over one hemisphere.

    Parameters
    ----------
    aeff : EffectiveArea
        Tabulated effective area for one season.
    hemisphere : {"upgoing", "downgoing"}, optional
        ``"upgoing"`` averages the ``sin(dec) > 0`` bins, ``"downgoing"`` the
        ``sin(dec) < 0`` bins. At the Pole a positive declination is below
        the horizon.

    Returns
    -------
    curve : np.ndarray
        Effective area [cm^2] on ``aeff.log10_energy_centers``, averaged over
        the selected declination bins weighted by their width in ``sin(dec)``.

    Raises
    ------
    ValueError
        Raised if ``hemisphere`` is not one of the two names.
    """
    if hemisphere not in ("upgoing", "downgoing"):
        raise ValueError(f"hemisphere must be 'upgoing' or 'downgoing', got {hemisphere!r}.")
    centers = aeff.sin_dec_centers
    mask = centers > 0.0 if hemisphere == "upgoing" else centers < 0.0
    widths = np.diff(aeff.sin_dec_edges)[mask]
    return np.average(aeff.values[:, mask], axis=1, weights=widths)


def livetime_weighted_effective_area(
    data_dir: str | pathlib.Path | None,
    log10_e: np.ndarray,
    hemisphere: str = "upgoing",
    seasons: tuple[str, ...] = SEASONS,
) -> tuple[np.ndarray, float]:
    """Effective area over one hemisphere, averaged over seasons by livetime.

    Each season's table is averaged over the hemisphere with
    :func:`hemisphere_average`, interpolated onto ``log10_e``, and weighted
    by that season's livetime.

    Parameters
    ----------
    data_dir : str or pathlib.Path or None
        Root of the DR2 release; ``None`` uses :func:`softpaws.data.paths.dr2_dir`.
    log10_e : np.ndarray
        Neutrino energies the curve is returned on [log10 GeV].
    hemisphere : {"upgoing", "downgoing"}, optional
        Hemisphere to average over.
    seasons : tuple of str, optional
        Seasons to combine. Defaults to the whole release.

    Returns
    -------
    aeff_cm2 : np.ndarray
        Livetime-weighted effective area on ``log10_e`` [cm^2].
    livetime_s : float
        Total livetime of the seasons combined [s].
    """
    log10_e = np.asarray(log10_e, dtype=float)
    total = np.zeros_like(log10_e)
    livetime_total = 0.0
    for season in seasons:
        aeff = load_effective_area(data_dir, season)
        livetime_s = season_livetime_s(data_dir, season)
        curve = hemisphere_average(aeff, hemisphere)
        total += livetime_s * np.interp(log10_e, aeff.log10_energy_centers, curve)
        livetime_total += livetime_s
    return total / livetime_total, livetime_total


def load_events(
    data_dir: str | pathlib.Path | None = None,
    seasons: tuple[str, ...] = IC86_SEASONS,
    within_uptime: bool = False,
) -> EventSet:
    """Reconstructed events of several seasons in one container.

    Parameters
    ----------
    data_dir : str or pathlib.Path or None
        Root of the DR2 release, holding ``events/`` and ``uptime/``; ``None``
        uses :func:`softpaws.data.paths.dr2_dir`.
    seasons : tuple of str, optional
        Seasons to load. Defaults to the eleven IC86 seasons, which share one
        response.
    within_uptime : bool, optional
        If True, keep only events whose time falls inside a good-run window
        of the loaded seasons. The release files already respect the good-run
        list, so this is a check and not a selection.

    Returns
    -------
    events : EventSet
        The concatenated events.
    """
    data_dir = _root(data_dir)
    require(data_dir / "events", "IceTracks-DR2 release")
    data = np.concatenate([load_season(data_dir / "events" / f"{s}_exp.csv") for s in seasons])
    events = EventSet(data)
    if within_uptime:
        mask = np.zeros(events.n_events, dtype=bool)
        for season in seasons:
            for start, stop in load_uptime(data_dir / "uptime" / f"{season}_exp.csv"):
                mask |= (events.time >= start) & (events.time <= stop)
        events = EventSet(events.data[mask])
    return events
