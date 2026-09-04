"""Numpy-backed container for IceTracks-DR2 muon-track events."""

from __future__ import annotations

import numpy as np

from softpaws.data.schema import EVENTS_DTYPE


class EventSet:
    """Structured container for a set of IceCube muon-track events.

    Parameters
    ----------
    data : np.ndarray
        Structured array with dtype ``EVENTS_DTYPE``.

    Attributes
    ----------
    data : np.ndarray
        The underlying structured array.
    n_events : int
        Number of events in the set.

    Examples
    --------
    >>> import numpy as np
    >>> from softpaws.data.schema import EVENTS_DTYPE
    >>> from softpaws.data.container import EventSet
    >>> arr = np.zeros(3, dtype=EVENTS_DTYPE)
    >>> evs = EventSet(arr)
    >>> evs.n_events
    3
    """

    def __init__(self, data: np.ndarray) -> None:
        """Wrap a structured array of events.

        Parameters
        ----------
        data : np.ndarray
            Structured array with dtype :data:`EVENTS_DTYPE`.

        Raises
        ------
        ValueError
            Raised if the array does not carry that dtype.
        """
        if data.dtype != EVENTS_DTYPE:
            raise ValueError(f"Expected dtype {EVENTS_DTYPE}, got {data.dtype}.")
        self._data = data

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def data(self) -> np.ndarray:
        """The underlying structured array."""
        return self._data

    @property
    def n_events(self) -> int:
        """Number of events."""
        return len(self._data)

    @property
    def run(self) -> np.ndarray:
        """IceCube run number for each event."""
        return self._data["run"]

    @property
    def event(self) -> np.ndarray:
        """Event ID within the season."""
        return self._data["event"]

    @property
    def subevent(self) -> np.ndarray:
        """Subevent ID (distinguishes coincident triggers)."""
        return self._data["subevent"]

    @property
    def time(self) -> np.ndarray:
        """Arrival time of all events [MJD]."""
        return self._data["time"]

    @property
    def log10_energy(self) -> np.ndarray:
        """log10 of reconstructed muon energy proxy [GeV]."""
        return self._data["log10_energy"]

    @property
    def sigma(self) -> np.ndarray:
        """Per-event angular uncertainty estimate [deg]."""
        return self._data["sigma"]

    @property
    def ra(self) -> np.ndarray:
        """Right ascension [deg, J2000]."""
        return self._data["ra"]

    @property
    def dec(self) -> np.ndarray:
        """Declination [deg, J2000]."""
        return self._data["dec"]

    @property
    def azimuth(self) -> np.ndarray:
        """Local azimuth of reconstructed direction [deg]."""
        return self._data["azimuth"]

    @property
    def zenith(self) -> np.ndarray:
        """Local zenith of reconstructed direction [deg]."""
        return self._data["zenith"]

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_dec(self, dec_min: float, dec_max: float) -> "EventSet":
        """Return a new EventSet restricted to a declination band.

        Parameters
        ----------
        dec_min : float
            Minimum declination [deg].
        dec_max : float
            Maximum declination [deg].

        Returns
        -------
        subset : EventSet
            New EventSet containing only events within the band.
        """
        mask = (self._data["dec"] >= dec_min) & (self._data["dec"] <= dec_max)
        return EventSet(self._data[mask])

    def filter_energy(self, log10_emin: float, log10_emax: float) -> "EventSet":
        """Return a new EventSet restricted to an energy range.

        Parameters
        ----------
        log10_emin : float
            Minimum log10(energy/GeV).
        log10_emax : float
            Maximum log10(energy/GeV).

        Returns
        -------
        subset : EventSet
            New EventSet containing only events in the energy range.
        """
        mask = (
            (self._data["log10_energy"] >= log10_emin)
            & (self._data["log10_energy"] <= log10_emax)
        )
        return EventSet(self._data[mask])

    def filter_time(self, mjd_start: float, mjd_stop: float) -> "EventSet":
        """Return a new EventSet restricted to a time window.

        Parameters
        ----------
        mjd_start : float
            Start of the window [MJD].
        mjd_stop : float
            End of the window [MJD].

        Returns
        -------
        subset : EventSet
            New EventSet containing only events within the time window.
        """
        mask = (self._data["time"] >= mjd_start) & (self._data["time"] <= mjd_stop)
        return EventSet(self._data[mask])

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of events."""
        return self.n_events

    def __repr__(self) -> str:
        """Short summary naming the number of events."""
        return f"EventSet(n_events={self.n_events:,})"
