"""Published effective-area curves the model is compared against.

Every loader reads one table shipped under ``src/softpaws/data/`` (or the
IceTracks-DR2 release for IceCube), returns it on the table's own energy
grid as ``log10(E_nu / GeV)`` and ``A_eff`` [cm^2], and says in its docstring
which paper and figure the table came from. Nothing is smoothed or
interpolated here; :func:`interpolate_aeff` is the one shared step for
putting a table on a model grid.

The tables are of three kinds, and the difference matters when a model curve
is held against them. A *trigger-level* area (ARCA230, P-ONE) carries no
analysis cuts and is the largest area an instrument reports, so it is the
right thing to hold a geometric ceiling against. A *selection-level* area
(ARCA21 bright track, ARCA230 final track selection, TRIDENT with its angular
cut, the DR2 table) sits below it by the selection efficiency. The two are
not interchangeable in a likelihood.
"""

from __future__ import annotations

import pathlib

import numpy as np

from .icecube import livetime_weighted_effective_area

__all__ = [
    "KM3NET_DIR",
    "PONE_DIR",
    "PONE_ZENITH_BANDS_DEG",
    "TRIDENT_COS_BANDS",
    "TRIDENT_DIR",
    "arca21_bright_track_aeff",
    "arca230_angle_dependent_aeff",
    "arca230_quoted_fit",
    "arca230_trigger_level_aeff",
    "icecube_dr2_aeff",
    "icecube_point_source_sensitivity",
    "interpolate_aeff",
    "pone_allsky_aeff",
    "pone_band_aeff",
    "trident_2025_map",
    "trident_band_aeff",
]

_DATA_DIR = pathlib.Path(__file__).parent

#: Directory of the KM3NeT tables.
KM3NET_DIR = _DATA_DIR / "km3net"

#: Directory of the P-ONE tables.
PONE_DIR = _DATA_DIR / "pone"

#: Directory of the TRIDENT tables.
TRIDENT_DIR = _DATA_DIR / "trident"

#: P-ONE zenith bands ``(lo, hi)`` [deg] in file order; 180 is the nadir.
PONE_ZENITH_BANDS_DEG = ((0, 30), (30, 60), (60, 90), (90, 120), (120, 150), (150, 180))

#: TRIDENT ``cos(theta)`` bands ``(lo, hi)`` in file order; -1 is the nadir.
TRIDENT_COS_BANDS = ((-1.0, -0.2), (-0.2, 0.2), (0.2, 1.0))


def interpolate_aeff(
    log10_e: np.ndarray, table_log10_e: np.ndarray, table_aeff_cm2: np.ndarray
) -> np.ndarray:
    """Put a tabulated curve on a grid by log-log interpolation.

    Parameters
    ----------
    log10_e : np.ndarray
        ``log10(E_nu / GeV)`` to evaluate at.
    table_log10_e : np.ndarray
        ``log10(E_nu / GeV)`` of the table, ascending.
    table_aeff_cm2 : np.ndarray
        Effective area of the table [cm^2], positive.

    Returns
    -------
    aeff_cm2 : np.ndarray
        Effective area on ``log10_e`` [cm^2], ``NaN`` outside the table.
    """
    return 10.0 ** np.interp(
        log10_e, table_log10_e, np.log10(table_aeff_cm2), left=np.nan, right=np.nan
    )


def _read_xy(path: pathlib.Path, comments: str = "#") -> tuple[np.ndarray, np.ndarray]:
    """Two-column table, finite and positive rows only, sorted in the first column."""
    raw = np.genfromtxt(path, delimiter=",", comments=comments)
    x, y = raw[:, 0], raw[:, 1]
    good = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
    order = np.argsort(x[good])
    return x[good][order], y[good][order]


# ---------------------------------------------------------------------------
# KM3NeT/ARCA
# ---------------------------------------------------------------------------


def arca230_trigger_level_aeff(path: pathlib.Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Digitized full-ARCA ``nu_mu`` effective area at trigger level.

    Digitized from KM3NeT Collaboration, Eur. Phys. J. C 84 (2024) 885
    [arXiv:2402.08363] Fig. 7, for the two-building-block detector (230
    detection units), over ``10^3`` to ``10^8`` GeV.

    Trigger level asks only that the event produce enough coincident hits,
    with none of the quality and containment cuts an analysis adds on top, so
    this is the largest effective area the instrument reports and the hardest
    for a footprint-based bound to accommodate.

    Parameters
    ----------
    path : pathlib.Path or None, optional
        Table to read. Defaults to ``arca_trigger_level_eff.csv`` in
        :data:`KM3NET_DIR`.

    Returns
    -------
    log10_e : np.ndarray
        ``log10(E_nu / GeV)`` of the digitized points, ascending.
    aeff_cm2 : np.ndarray
        Effective area [cm^2].
    """
    e_gev, a_m2 = _read_xy(path or KM3NET_DIR / "arca_trigger_level_eff.csv")
    return np.log10(e_gev), 1.0e4 * a_m2


def arca230_quoted_fit(log10_e: np.ndarray) -> np.ndarray:
    """Analytic parametrization of the ARCA230 trigger curve quoted in the literature.

    Several phenomenology papers quote

    .. math:: A_\\mathrm{eff} = 2\\,[0.20\\,(E/E_0)^{-0.51} + 0.46\\,(E/E_0)^{-0.06}]^{-6.4}
        \\ \\mathrm{m}^2, \\quad E_0 = 10^4\\ \\mathrm{GeV},

    over ``10^3`` to ``10^8`` GeV, attributed to the same figure as
    :func:`arca230_trigger_level_aeff`. It runs a factor 1.95 above the
    digitized curve across five decades, which is flat enough to identify:
    the leading 2 doubles a curve that is already the two-block effective
    area. It is kept so that the discrepancy stays visible.

    Parameters
    ----------
    log10_e : np.ndarray
        ``log10(E_nu / GeV)``.

    Returns
    -------
    aeff_cm2 : np.ndarray
        Effective area [cm^2], ``NaN`` outside the quoted validity.
    """
    x = 10.0 ** (np.asarray(log10_e, dtype=float) - 4.0)
    aeff_m2 = 2.0 * (0.20 * x**-0.51 + 0.46 * x**-0.06) ** -6.4
    return np.where((log10_e >= 3.0) & (log10_e <= 8.0), aeff_m2 * 1.0e4, np.nan)


def arca21_bright_track_aeff(path: pathlib.Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Released ARCA21 bright-track, all-flavour, sky-averaged effective area.

    The selection that recorded KM3-230213A, from the event's own data
    release: KM3NeT Collaboration, Nature 638 (2025) 376, Zenodo record
    10.5281/zenodo.14860165 (``uhe-event-v1.0``), file
    ``effective_area_brighttrackselection_allflavour_skyavg.json``. Its
    stated convention is ``N = 4 pi T Integral A_eff(E) Phi_per-flavour(E) dE``.
    The table runs to ``10^11`` GeV and is zero below its threshold; only the
    positive rows are returned.

    Parameters
    ----------
    path : pathlib.Path or None, optional
        Table to read. Defaults to
        ``arca21_aeff_brighttrack_allflavour_skyavg.csv`` in :data:`KM3NET_DIR`.

    Returns
    -------
    log10_e : np.ndarray
        ``log10(E_nu / GeV)`` of the tabulated points with positive area.
    aeff_cm2 : np.ndarray
        Effective area [cm^2].
    """
    e_gev, a_cm2 = _read_xy(path or KM3NET_DIR / "arca21_aeff_brighttrack_allflavour_skyavg.csv")
    return np.log10(e_gev), a_cm2


def arca230_angle_dependent_aeff(
    path: pathlib.Path | None = None,
) -> list[tuple[float, float, np.ndarray, np.ndarray]]:
    """Digitized ARCA230 effective area per ``cos(theta)`` band, final track selection.

    Fig. 7(b) of KM3NeT Collaboration, Eur. Phys. J. C 84 (2024) 885
    [arXiv:2402.08363]: the ``nu_mu`` charged-current effective area after
    the final track selection, in four bands of ``cos(theta)``. The column
    headers of the table carry the band edges as ``km3net_<hi>g<lo>``,
    largest ``cos(theta)`` first.

    Parameters
    ----------
    path : pathlib.Path or None, optional
        Table to read. Defaults to ``km3net_angle_dependent.csv`` in
        :data:`KM3NET_DIR`.

    Returns
    -------
    bands : list of tuple
        One entry ``(cos_hi, cos_lo, log10_e, aeff_cm2)`` per band, points
        sorted in energy. ``cos_hi`` is the largest ``cos(theta)`` of the
        band; ``-1`` is the nadir, so the last band is the deepest column.
    """
    path = path or KM3NET_DIR / "km3net_angle_dependent.csv"
    with open(path) as handle:
        names = [n for n in handle.readline().strip().split(",") if n]
    raw = np.genfromtxt(path, delimiter=",", skip_header=2)
    bands = []
    for k, name in enumerate(names):
        hi, lo = name.removeprefix("km3net_").split("g")
        x, y = raw[:, 2 * k], raw[:, 2 * k + 1]
        good = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
        order = np.argsort(x[good])
        bands.append((float(hi), float(lo), np.log10(x[good][order]), 1.0e4 * y[good][order]))
    return bands


# ---------------------------------------------------------------------------
# IceCube
# ---------------------------------------------------------------------------


def icecube_dr2_aeff(
    data_dir: str | pathlib.Path, log10_e: np.ndarray, hemisphere: str = "upgoing"
) -> tuple[np.ndarray, float]:
    """Livetime-weighted IceTracks-DR2 effective area over one hemisphere.

    The ``nu_mu`` effective-area tables of the IceTracks-DR2 release
    (DOI 10.7910/DVN/MMIIZA), one per detector configuration, averaged over
    the declination bins of one hemisphere and over the 14 seasons weighted
    by each season's good-run livetime; see
    :func:`~softpaws.data.icecube.livetime_weighted_effective_area`. The
    table is analysis level: it carries the through-going track selection.

    Parameters
    ----------
    data_dir : str or pathlib.Path
        Root of the DR2 release, holding ``irfs/`` and ``uptime/``.
    log10_e : np.ndarray
        ``log10(E_nu / GeV)`` to return the curve on.
    hemisphere : {"upgoing", "downgoing"}, optional
        Hemisphere to average over.

    Returns
    -------
    aeff_cm2 : np.ndarray
        Effective area on ``log10_e`` [cm^2].
    livetime_s : float
        Total good-run livetime of the release [s].
    """
    return livetime_weighted_effective_area(data_dir, log10_e, hemisphere)


# ---------------------------------------------------------------------------
# P-ONE and TRIDENT
# ---------------------------------------------------------------------------


def pone_allsky_aeff(data_dir: pathlib.Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Digitized P-ONE all-sky ``nu_mu`` effective area at trigger level.

    From the P-ONE performance study presented at ICRC 2023: seven clusters
    of 120 m radius and 1 km height at 2.66 km in the Cascadia Basin,
    trigger level, ``nu_mu`` with neutrino and antineutrino averaged. The
    curve was digitized from a step-function plot, so the raw points carry
    the risers; the release is in m^2 and is converted here.

    Parameters
    ----------
    data_dir : pathlib.Path or None, optional
        Directory holding ``pone_allsky.csv``. Defaults to :data:`PONE_DIR`.

    Returns
    -------
    log10_e : np.ndarray
        ``log10(E_nu / GeV)`` of the digitized points, ascending.
    aeff_cm2 : np.ndarray
        Effective area [cm^2].
    """
    e_gev, a_m2 = _read_xy((data_dir or PONE_DIR) / "pone_allsky.csv")
    return np.log10(e_gev), 1.0e4 * a_m2


def pone_band_aeff(
    zenith_lo_deg: int, zenith_hi_deg: int, data_dir: pathlib.Path | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Digitized P-ONE ``nu_mu`` effective area in one zenith band, trigger level.

    Same source as :func:`pone_allsky_aeff`, in the six 30-degree zenith
    bands of :data:`PONE_ZENITH_BANDS_DEG`, with 180 degrees the nadir.

    Parameters
    ----------
    zenith_lo_deg, zenith_hi_deg : int
        Band edges [deg], one of the pairs in :data:`PONE_ZENITH_BANDS_DEG`.
    data_dir : pathlib.Path or None, optional
        Directory holding ``pone_<lo>_<hi>.csv``. Defaults to :data:`PONE_DIR`.

    Returns
    -------
    log10_e : np.ndarray
        ``log10(E_nu / GeV)`` of the digitized points, ascending.
    aeff_cm2 : np.ndarray
        Effective area [cm^2].
    """
    e_gev, a_m2 = _read_xy((data_dir or PONE_DIR) / f"pone_{zenith_lo_deg}_{zenith_hi_deg}.csv")
    return np.log10(e_gev), 1.0e4 * a_m2


def trident_band_aeff(
    cos_lo: float, cos_hi: float, data_dir: pathlib.Path | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Digitized TRIDENT ``nu_mu`` effective area in one ``cos(theta)`` band.

    Extended Data Fig. 8 of the TRIDENT Nature Astronomy paper (Ye et al.,
    2023): a 2 km-radius, 570 m-high array at 3.1 km in the South China Sea,
    ``nu_mu`` with neutrino and antineutrino averaged, after a 6-degree
    angular-error cut, in the three ``cos(theta)`` bands of
    :data:`TRIDENT_COS_BANDS` with -1 the nadir. The energy axis of the
    release is already ``log10(E / GeV)``; the area is in m^2 and is
    converted here.

    Parameters
    ----------
    cos_lo, cos_hi : float
        Band edges, one of the pairs in :data:`TRIDENT_COS_BANDS`.
    data_dir : pathlib.Path or None, optional
        Directory holding ``trident_<lo>_<hi>.csv``. Defaults to
        :data:`TRIDENT_DIR`.

    Returns
    -------
    log10_e : np.ndarray
        ``log10(E_nu / GeV)`` of the digitized points, ascending.
    aeff_cm2 : np.ndarray
        Effective area [cm^2].

    Raises
    ------
    ValueError
        Raised if ``(cos_lo, cos_hi)`` is not one of the tabulated bands.
    """
    stems = dict(zip(TRIDENT_COS_BANDS, ("trident_-1_-0.2", "trident_-0.2_0.2", "trident_0.2_1.0")))
    try:
        stem = stems[(float(cos_lo), float(cos_hi))]
    except KeyError:
        raise ValueError(
            f"({cos_lo}, {cos_hi}) is not a TRIDENT band; the bands are {TRIDENT_COS_BANDS}."
        ) from None
    log10_e, a_m2 = _read_xy((data_dir or TRIDENT_DIR) / f"{stem}.csv")
    return log10_e, 1.0e4 * a_m2


def trident_2025_map(path: pathlib.Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TRIDENT's 2025 ``nu_mu`` effective-area map in ``cos(theta_z)`` and energy.

    Morton-Blake et al. (arXiv:2510.24395), Fig. 3a: the ``nu_mu``
    charged-current track effective area of the reference TRIDENT layout
    (1000 strings at 100 m, 20 hDOMs at 30 m) on a 12 x 12 grid in
    ``cos(theta_z)`` and ``log10 E``, after trigger, edge and track-extension
    cuts. The map was digitized from the embedded raster; the release is
    ``log10`` of m^2 and is converted here.

    Parameters
    ----------
    path : pathlib.Path or None, optional
        Table to read. Defaults to ``trident_2025_fig3a_log10aeff_m2.csv`` in
        :data:`TRIDENT_DIR`.

    Returns
    -------
    cos_theta : np.ndarray, shape (12,)
        ``cos(theta_z)`` bin centres, downgoing first.
    log10_e : np.ndarray, shape (12,)
        ``log10(E_nu / GeV)`` bin centres, 0.25 dex from 3.125 to 5.875.
    log10_aeff_cm2 : np.ndarray, shape (12, 12)
        ``log10`` of the effective area [cm^2], one row per ``cos_theta``.
    """
    raw = np.loadtxt(path or TRIDENT_DIR / "trident_2025_fig3a_log10aeff_m2.csv", delimiter=",")
    cos_theta, log10_a_m2 = raw[:, 0], raw[:, 1:]
    log10_e = 3.0 + 0.25 * (np.arange(log10_a_m2.shape[1]) + 0.5)
    return cos_theta, log10_e, log10_a_m2 + 4.0


def icecube_point_source_sensitivity(
    path: pathlib.Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """IceCube's 14-year through-going-track point-source sensitivity.

    Digitized from the collaboration's published sensitivity against source
    declination, for an ``E^-2`` source over the whole energy range.

    Parameters
    ----------
    path : pathlib.Path, optional
        Two-column CSV of ``sin(dec)`` and ``E^2 dN/dE`` [TeV cm^-2 s^-1].
        ``None`` uses the table shipped in ``softpaws/data/bounds``.

    Returns
    -------
    sin_dec : np.ndarray
        Source ``sin(dec)``, sorted ascending and clipped to the unit interval.
    e2_flux : np.ndarray
        ``E^2 dN/dE`` per flavour [GeV cm^-2 s^-1].
    """
    if path is None:
        path = _DATA_DIR / "bounds" / "icecube_14year_track_sensitivity_e2.csv"
    raw = np.loadtxt(path, delimiter=",")
    order = np.argsort(raw[:, 0])
    # The digitization overshoots |sin(dec)| = 1 by a few parts in a thousand.
    return np.clip(raw[order, 0], -1.0, 1.0), raw[order, 1] * 1.0e3
