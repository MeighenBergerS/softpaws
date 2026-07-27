"""Binned observed datasets for the Section 4 comparison fits.

Assembles the in-repo IceTracks-DR2 events into the binned muon-energy histograms
used by the diffuse-flux likelihood (:mod:`softpaws.comparison.likelihood`). Only
the IC86 seasons are combined, matching the single IC86 IRF, and only the
downgoing hemisphere is kept, where the soft-volume drift limit's unattenuated-flux
assumption applies (see :mod:`softpaws.comparison.rates`).

This is the DR2 *proxy* for the paper's IceCube 9.5 yr through-going dataset
(Ref. [28] of arXiv:2607.13143), which is not part of this release.
"""

from __future__ import annotations

import pathlib

import numpy as np

from ..data.container import EventSet
from ..data.loader import compute_livetime_s, load_all_seasons, load_uptime
from .rates import observed_counts

IC86_SEASONS = (
    "IC86_I", "IC86_II", "IC86_III", "IC86_IV", "IC86_V", "IC86_VI",
    "IC86_VII", "IC86_VIII", "IC86_IX", "IC86_X", "IC86_XI",
)


def load_ic86_downgoing(
    data_dir: str | pathlib.Path,
    log10_e_edges: np.ndarray,
    dec_min: float = -90.0,
    dec_max: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Binned downgoing IC86 counts and combined livetime.

    Parameters
    ----------
    data_dir : str or pathlib.Path
        Root of the DR2 data directory (``events/`` and ``uptime/``).
    log10_e_edges : np.ndarray, shape (n_bins + 1,)
        Muon-energy bin edges in ``log10(E / GeV)``.
    dec_min, dec_max : float, optional
        Declination band [deg]. Defaults to the downgoing hemisphere.

    Returns
    -------
    counts : np.ndarray, shape (n_bins,)
        Observed events per bin.
    livetime_s : float
        Combined IC86 livetime [s].
    """
    data_dir = pathlib.Path(data_dir)
    events = EventSet(load_all_seasons(data_dir))

    mask = np.zeros(events.n_events, dtype=bool)
    livetime_s = 0.0
    for season in IC86_SEASONS:
        uptime = load_uptime(data_dir / "uptime" / f"{season}_exp.csv")
        for start, stop in uptime:
            mask |= (events.time >= start) & (events.time <= stop)
        livetime_s += compute_livetime_s(uptime)

    ic86 = EventSet(events.data[mask])
    counts = observed_counts(ic86, log10_e_edges, dec_min, dec_max)
    return counts.astype(float), livetime_s
