"""Effective area and smearing matrix for IceTracks-DR2.

Three classes
-------------
EffectiveArea
    A_eff(log10_E_nu, dec) [cm²], bilinearly interpolated.
PointSpreadFunction
    PSF containment angle (deg), legacy simple class kept for compatibility.
SmearingMatrix
    Full 5-D conditional response table mapping true (E_nu, dec) to
    reconstructed (E_reco, PSF_angle, AngErr) for signal injection.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class EffectiveArea:
    """Tabulated effective area as a function of energy and declination.

    Parameters
    ----------
    log10_energy_edges : np.ndarray
        Bin edges in log10(E_nu/GeV), shape ``(N+1,)``.
    sin_dec_edges : np.ndarray
        Bin edges in sin(declination), shape ``(M+1,)``.
    values : np.ndarray
        Effective area values [cm²], shape ``(N, M)``.

    Notes
    -----
    Interpolation uses the bin centres derived from the provided edges.
    Queries outside the tabulated range are clipped to the boundary.
    Values are in cm² to match the IceTracks-DR2 CSV files and be consistent
    with the signal-rate formula (J in GeV² cm⁻⁵, σv in cm³ s⁻¹).

    Examples
    --------
    >>> import numpy as np
    >>> log10_e = np.linspace(2, 8, 7)
    >>> sin_dec = np.linspace(-1, 1, 5)
    >>> vals = np.ones((6, 4))
    >>> aeff = EffectiveArea(log10_e, sin_dec, vals)
    >>> float(aeff(4.0, 0.0))
    1.0
    """

    def __init__(
        self,
        log10_energy_edges: np.ndarray,
        sin_dec_edges: np.ndarray,
        values: np.ndarray,
    ) -> None:
        """Wrap a binned effective-area table.

        Parameters
        ----------
        log10_energy_edges : np.ndarray, shape (n_energy + 1,)
            Neutrino-energy bin edges [log10 GeV].
        sin_dec_edges : np.ndarray, shape (n_dec + 1,)
            Declination bin edges in ``sin(dec)``.
        values : np.ndarray, shape (n_energy, n_dec)
            Effective area in each bin [cm^2].
        """
        self._log10_e_edges = np.asarray(log10_energy_edges, dtype=float)
        self._sin_dec_edges = np.asarray(sin_dec_edges, dtype=float)
        self._log10_e_centers = 0.5 * (log10_energy_edges[:-1] + log10_energy_edges[1:])
        self._sin_dec_centers = 0.5 * (sin_dec_edges[:-1] + sin_dec_edges[1:])
        self._values = np.asarray(values, dtype=float)
        self._interp = RegularGridInterpolator(
            (self._log10_e_centers, self._sin_dec_centers),
            self._values,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    def __call__(self, log10_energy: float | np.ndarray, dec: float | np.ndarray) -> np.ndarray:
        """Evaluate the effective area at given energy and declination.

        Parameters
        ----------
        log10_energy : float or np.ndarray
            log10(E_nu/GeV) at which to evaluate.
        dec : float or np.ndarray
            Declination [deg] at which to evaluate.

        Returns
        -------
        aeff : np.ndarray
            Effective area [cm²].
        """
        sin_dec = np.sin(np.deg2rad(dec))
        pts = np.column_stack([np.atleast_1d(log10_energy), np.atleast_1d(sin_dec)])
        return self._interp(pts)

    @property
    def log10_energy_centers(self) -> np.ndarray:
        """Tabulated bin centres in log10(E_nu/GeV), shape ``(N,)``."""
        return self._log10_e_centers

    @property
    def sin_dec_centers(self) -> np.ndarray:
        """Tabulated bin centres in sin(declination), shape ``(M,)``."""
        return self._sin_dec_centers

    @property
    def sin_dec_edges(self) -> np.ndarray:
        """Tabulated bin edges in sin(declination), shape ``(M+1,)``."""
        return self._sin_dec_edges

    @property
    def values(self) -> np.ndarray:
        """Raw tabulated effective area grid [cm²], shape ``(N, M)``."""
        return self._values


class PointSpreadFunction:
    """Tabulated PSF containment angles as a function of energy and declination.

    Parameters
    ----------
    log10_energy_edges : np.ndarray
        Bin edges in log10(E/GeV), shape ``(N+1,)``.
    sin_dec_edges : np.ndarray
        Bin edges in sin(declination), shape ``(M+1,)``.
    quantiles : np.ndarray
        PSF containment angles [deg], shape ``(N, M)``.
        Typically the 68% or 50% containment radius.

    Examples
    --------
    >>> import numpy as np
    >>> log10_e = np.linspace(2, 8, 7)
    >>> sin_dec = np.linspace(-1, 1, 5)
    >>> q = np.ones((6, 4)) * 0.5
    >>> psf = PointSpreadFunction(log10_e, sin_dec, q)
    >>> float(psf(4.0, 0.0))
    0.5
    """

    def __init__(
        self,
        log10_energy_edges: np.ndarray,
        sin_dec_edges: np.ndarray,
        quantiles: np.ndarray,
    ) -> None:
        """Wrap a binned table of angular-error quantiles.

        Parameters
        ----------
        log10_energy_edges : np.ndarray, shape (n_energy + 1,)
            Neutrino-energy bin edges [log10 GeV].
        sin_dec_edges : np.ndarray, shape (n_dec + 1,)
            Declination bin edges in ``sin(dec)``.
        quantiles : np.ndarray, shape (n_energy, n_dec)
            Containment angle in each bin [deg].
        """
        self._log10_e_centers = 0.5 * (log10_energy_edges[:-1] + log10_energy_edges[1:])
        self._sin_dec_centers = 0.5 * (sin_dec_edges[:-1] + sin_dec_edges[1:])
        self._quantiles = np.asarray(quantiles, dtype=float)
        self._interp = RegularGridInterpolator(
            (self._log10_e_centers, self._sin_dec_centers),
            self._quantiles,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    def __call__(self, log10_energy: float | np.ndarray, dec: float | np.ndarray) -> np.ndarray:
        """Evaluate the PSF containment angle at given energy and declination.

        Parameters
        ----------
        log10_energy : float or np.ndarray
            log10(E/GeV) at which to evaluate.
        dec : float or np.ndarray
            Declination [deg] at which to evaluate.

        Returns
        -------
        psi : np.ndarray
            PSF containment angle [deg].
        """
        sin_dec = np.sin(np.deg2rad(dec))
        pts = np.column_stack([np.atleast_1d(log10_energy), np.atleast_1d(sin_dec)])
        return self._interp(pts)


class SmearingMatrix:
    """5-D conditional response matrix for IceTracks-DR2.

    Maps true neutrino (log10_E_nu, dec) to the joint distribution over
    reconstructed (log10_E_reco, PSF_angle, AngErr) via the
    ``Fractional_Counts`` column of the IceTracks-DR2 smearing CSV.

    Parameters
    ----------
    raw : np.ndarray
        Raw 11-column array loaded from ``<season>_smearing.csv``.
        Columns: log10(E_nu)_min/max, dec_min/max [deg],
        log10(E_reco)_min/max, PSF_min/max [deg],
        AngErr_min/max [deg], Fractional_Counts.

    Notes
    -----
    The table has 14 true-energy bins × 30 declination bins = 420 groups,
    each with 8 000 conditional response entries.  The constructor indexes
    the flat table and pre-computes per-group CDFs so that
    :meth:`sample` runs in O(N) time.

    Examples
    --------
    >>> import numpy as np
    >>> from softpaws.data.loader import load_irfs
    >>> irfs = load_irfs("src/softpaws/data/dataverse_files/irfs", "IC86_I")
    >>> sm = irfs["smearing"]                               # doctest: +SKIP
    >>> rng = np.random.default_rng(0)
    >>> e, psf, ae = sm.sample(np.array([3.5]), np.array([0.0]), rng)  # doctest: +SKIP
    """

    def __init__(self, raw: np.ndarray) -> None:
        """Parse a smearing table into its bin structure and probabilities.

        Parameters
        ----------
        raw : np.ndarray, shape (n_rows, 7)
            Rows of the released smearing file: the true-energy and
            declination bin edges, the reconstructed-energy and
            angular-error bin edges, and the fractional counts.
        """
        # ---- discover bin structure ----------------------------------------
        enu_bins = np.unique(raw[:, :2], axis=0)   # (N_enu, 2)
        dec_bins = np.unique(raw[:, 2:4], axis=0)  # (N_dec, 2)

        self._log10_enu_min: np.ndarray = enu_bins[:, 0]
        self._log10_enu_max: np.ndarray = enu_bins[:, 1]
        self._dec_min: np.ndarray = dec_bins[:, 0]
        self._dec_max: np.ndarray = dec_bins[:, 1]
        self._n_enu: int = len(enu_bins)
        self._n_dec: int = len(dec_bins)

        n_groups = self._n_enu * self._n_dec
        if raw.shape[0] % n_groups != 0:
            raise ValueError(
                f"Smearing table has {raw.shape[0]} rows but "
                f"{self._n_enu} E_nu bins × {self._n_dec} dec bins = {n_groups} groups "
                f"do not divide evenly."
            )
        self._n_per_group: int = raw.shape[0] // n_groups

        # ---- sort rows into (n_enu, n_dec, n_per_group) order --------------
        # Assign bin indices to each row
        i_enu = np.searchsorted(self._log10_enu_min, raw[:, 0], side="right") - 1
        i_dec = np.searchsorted(self._dec_min, raw[:, 2], side="right") - 1
        i_enu = np.clip(i_enu, 0, self._n_enu - 1)
        i_dec = np.clip(i_dec, 0, self._n_dec - 1)

        # Flat group index + position within group (for stable sort)
        group_idx = i_enu * self._n_dec + i_dec
        within_idx = np.zeros(len(raw), dtype=int)
        seen = {}
        for k, g in enumerate(group_idx):
            within_idx[k] = seen.get(g, 0)
            seen[g] = within_idx[k] + 1

        sort_order = np.lexsort((within_idx, group_idx))
        # Keep only the 7 response columns: Ereco_min/max, PSF_min/max, AngErr_min/max, Frac
        data_sorted = raw[sort_order, 4:]  # (N_total, 7)

        self._data = data_sorted.reshape(
            self._n_enu, self._n_dec, self._n_per_group, 7
        ).astype(float)

        # ---- pre-compute normalised CDFs per group -------------------------
        fracs = self._data[:, :, :, 6]                          # (n_enu, n_dec, n_per)
        totals = fracs.sum(axis=2, keepdims=True)
        totals = np.where(totals > 0, totals, 1.0)
        self._cdf = np.cumsum(fracs / totals, axis=2)           # (n_enu, n_dec, n_per)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        log10_E_nu: np.ndarray,
        dec: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample reconstructed properties for given true-neutrino events.

        Parameters
        ----------
        log10_E_nu : np.ndarray, shape (N,)
            True neutrino log10(E_nu / GeV).
        dec : np.ndarray, shape (N,)
            True neutrino declination [deg] (approximately the source dec).
        rng : np.random.Generator
            NumPy random generator.

        Returns
        -------
        log10_E_reco : np.ndarray, shape (N,)
            Reconstructed muon energy proxy log10(E_reco / GeV).
        psf_angle : np.ndarray, shape (N,)
            Angular opening between true and reconstructed direction [deg].
        ang_err : np.ndarray, shape (N,)
            Per-event estimated angular uncertainty σ [deg].
        """
        log10_E_nu = np.asarray(log10_E_nu, dtype=float)
        dec = np.asarray(dec, dtype=float)
        n = log10_E_nu.size

        # Bin indices
        i_enu = np.clip(
            np.searchsorted(self._log10_enu_min, log10_E_nu, side="right") - 1,
            0, self._n_enu - 1,
        )
        i_dec = np.clip(
            np.searchsorted(self._dec_min, dec, side="right") - 1,
            0, self._n_dec - 1,
        )

        log10_E_reco = np.empty(n)
        psf_angle = np.empty(n)
        ang_err = np.empty(n)

        u = rng.uniform(size=n)
        for k in range(n):
            cdf = self._cdf[i_enu[k], i_dec[k]]
            idx = min(int(np.searchsorted(cdf, u[k])), self._n_per_group - 1)
            row = self._data[i_enu[k], i_dec[k], idx]
            log10_E_reco[k] = rng.uniform(row[0], row[1])
            psf_angle[k] = rng.uniform(row[2], row[3])
            ang_err[k] = rng.uniform(row[4], row[5])

        return log10_E_reco, psf_angle, ang_err

    # ------------------------------------------------------------------
    # Energy migration matrix
    # ------------------------------------------------------------------

    @property
    def log10_enu_edges(self) -> np.ndarray:
        """True-energy bin edges [log10(GeV)], shape (n_enu+1,)."""
        return np.append(self._log10_enu_min, self._log10_enu_max[-1])

    @property
    def log10_enu_centers(self) -> np.ndarray:
        """True-energy bin centres [log10(GeV)], shape (n_enu,)."""
        return 0.5 * (self._log10_enu_min + self._log10_enu_max)

    @property
    def dec_edges(self) -> np.ndarray:
        """Declination bin edges [deg], shape (n_dec+1,)."""
        return np.append(self._dec_min, self._dec_max[-1])

    @property
    def dec_centers(self) -> np.ndarray:
        """Declination bin centres [deg], shape (n_dec,)."""
        return 0.5 * (self._dec_min + self._dec_max)

    def psf_containment_deg(self, quantile: float = 0.68) -> np.ndarray:
        """Containment radius of the point spread, per true-energy and declination bin.

        The released table carries a ``PSF_angle`` bin with every row, so the
        angular resolution is a weighted quantile of that column inside each
        group. It is the number a point-source search needs and an effective
        area does not: how much sky has to be looked through to keep a source,
        which falls as the energy rises and the track leaves more light.

        Parameters
        ----------
        quantile : float, optional
            Containment to return. See
            :data:`~softpaws.response.sensitivity.OPTIMAL_CONTAINMENT` for why
            68% is the one a counting bin wants.

        Returns
        -------
        containment_deg : np.ndarray, shape (n_enu, n_dec)
            Angle containing ``quantile`` of the reconstructed events [deg].
            ``NaN`` in bins the release leaves empty.
        """
        angle = 0.5 * (self._data[:, :, :, 2] + self._data[:, :, :, 3])
        weight = self._data[:, :, :, 6]
        order = np.argsort(angle, axis=2)
        angle = np.take_along_axis(angle, order, axis=2)
        cumulative = np.cumsum(np.take_along_axis(weight, order, axis=2), axis=2)

        total = cumulative[:, :, -1]
        with np.errstate(divide="ignore", invalid="ignore"):
            cdf = cumulative / total[:, :, None]
        # The first node at or past the quantile, and the one before it, so the
        # answer is interpolated rather than rounded to a bin edge.
        upper = np.argmax(cdf >= quantile, axis=2)
        lower = np.maximum(upper - 1, 0)
        take = lambda a, i: np.take_along_axis(a, i[:, :, None], axis=2)[:, :, 0]  # noqa: E731
        cdf_lo, cdf_hi = take(cdf, lower), take(cdf, upper)
        angle_lo, angle_hi = take(angle, lower), take(angle, upper)
        span = np.where(cdf_hi > cdf_lo, cdf_hi - cdf_lo, 1.0)
        result = angle_lo + (angle_hi - angle_lo) * (quantile - cdf_lo) / span
        return np.where(total > 0.0, result, np.nan)

    def energy_response_matrix(self, log10_reco_edges: np.ndarray) -> np.ndarray:
        """Marginal energy migration P(E_reco_bin | E_nu_bin, dec_bin).

        Marginalises over PSF angle and AngErr, summing ``Fractional_Counts``
        for all smearing-table entries whose reconstructed energy mid-point
        falls inside each output reco bin.

        Parameters
        ----------
        log10_reco_edges : ndarray, shape (n_reco+1,)
            Reconstructed energy bin edges in log10(E/GeV).

        Returns
        -------
        matrix : ndarray, shape (n_enu, n_dec, n_reco)
            Probability of reconstructing in each reco bin.  Values may sum
            to < 1 along axis 2 when ``log10_reco_edges`` do not cover the
            full smearing-table E_reco range.
        """
        n_reco = len(log10_reco_edges) - 1
        ereco_mid = 0.5 * (self._data[:, :, :, 0] + self._data[:, :, :, 1])
        fracs     = self._data[:, :, :, 6]

        matrix = np.zeros((self._n_enu, self._n_dec, n_reco))
        for i_r in range(n_reco):
            lo, hi = log10_reco_edges[i_r], log10_reco_edges[i_r + 1]
            in_bin = (ereco_mid >= lo) & (ereco_mid < hi)
            matrix[:, :, i_r] = np.sum(fracs * in_bin, axis=2)

        return matrix
