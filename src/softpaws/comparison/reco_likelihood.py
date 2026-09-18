"""Reconstructed-energy likelihood of the IceCube DR2 upgoing track sample.

The forward model of :mod:`softpaws.response` predicts a rate against true
neutrino energy and declination. The events of the IceTracks-DR2 release
carry a reconstructed muon-energy proxy instead, so a comparison has to fold
the model through the released smearing table first. This module does that
fold and wraps it in a Poisson likelihood over (reconstructed energy,
declination band) bins:

* :func:`smearing_marginal` sums the released 5D table over its
  point-spread and angular-error axes and projects it onto one common
  reconstructed-energy grid.
* :func:`banded_responses` puts a model effective area on the same
  declination bands as that table.
* :func:`atmospheric_fluxes` reads the MCEq background and
  :func:`binned_events` histograms the observed events.
* :class:`RecoLikelihood` folds the model, adds the background, and profiles
  the astrophysical normalization, the spectral index and the two
  atmospheric normalizations at a fixed flavour ratio.

The grids, priors and external anchors the fit uses are module constants, so
a script and its consumers share one definition of the fit window.

Notes
-----
The release is documented at <https://doi.org/10.7910/DVN/MMIIZA>.
"""

from __future__ import annotations

import logging
import pathlib
from collections.abc import Callable, Sequence

import numpy as np
from scipy.optimize import minimize

from softpaws.data.icecube import IC86_SEASONS, load_events
from softpaws.data.paths import dr2_dir, require
from softpaws.fluxes import (
    FLUX_UNIT,
    ICECUBE_COMBINED_2023,
    ICECUBE_TRACKS_2022,
    AtmosphericFlux,
    load_mceq_table,
)

from ..constants import (
    FC_R_SCAN,
    FC_R_TRUE,
    LOG10_E_GRID,
    MODEL_SYS,
    N_MU_GRID,
    N_TAU_GRID,
    PHI_MU_GRID,
    PHI_TAU_GRID,
    PHI_TAU_SCAN,
    R_GRID,
    RECO_EDGES,
    TAU_DECAY_X,
)

__all__ = [
    "ANCHORS",
    "CHANNELS",
    "CONV_PRIOR",
    "FIT_RECO",
    "FC_R_SCAN",
    "FC_R_TRUE",
    "GAMMA_BOUNDS",
    "GAMMA_GRID",
    "LOG10_E_GRID",
    "MODEL_SYS",
    "N_MU_GRID",
    "N_TAU_GRID",
    "PHI_MU_GRID",
    "PHI_TAU_GRID",
    "PHI_TAU_SCAN",
    "PIVOT_GAMMA",
    "PIVOT_PHI0",
    "PROMPT_PRIOR",
    "RECO_EDGES",
    "R_GRID",
    "RecoLikelihood",
    "TAU_DECAY_X",
    "TRACKS_GAMMA",
    "TRACKS_PHI_MU",
    "atmospheric_fluxes",
    "banded_responses",
    "binned_events",
    "cached_fit_inputs",
    "fit_inputs",
    "load_banded_responses",
    "physical_r_range",
    "save_banded_responses",
    "smearing_marginal",
    "true_counts",
]

logger = logging.getLogger(__name__)

#: Astrophysical pivot: the per-flavour normalization unit and the index
#: seed, from IceCube's combined fit (arXiv:2308.00191). ``N = 1`` in the fit
#: means this flux.
PIVOT_PHI0 = ICECUBE_COMBINED_2023.phi0 * FLUX_UNIT
PIVOT_GAMMA = ICECUBE_COMBINED_2023.gamma

FIT_RECO = (4.25, 7.5)


#: Gaussian priors on the atmospheric normalizations, the standard breakers
#: of the prompt-astro degeneracy: hadronic-model spread on both. An earlier
#: order-one prompt prior let the prompt normalization run to 1.6 and swallow
#: the astrophysical ``nu_mu`` (its norm fell to 0.5); at 0.25, the
#: perturbative-QCD spread, the astrophysical normalization comes back to
#: 0.9 x the combined fit.
CONV_PRIOR = (1.0, 0.25)
PROMPT_PRIOR = (1.0, 0.25)

#: Physical bounds on the astrophysical spectral index, without which the
#: astro component can degenerate into a shape patch for the atmospheric
#: window edge (a first fit ran to gamma = 15 with the normalization at
#: zero).
GAMMA_BOUNDS = (1.5, 4.0)


#: Spectral-index grid the astrophysical folds are cached on; the profile
#: interpolates between its nodes, which is what makes the pseudo-experiment
#: calibration affordable.
GAMMA_GRID = np.linspace(GAMMA_BOUNDS[0], GAMMA_BOUNDS[1], 51)

#: The two track channels the model carries: a ``nu_mu`` charged-current
#: muon and a ``nu_tau -> tau -> mu`` one.
CHANNELS = ("mu", "tau")


#: IceCube's 9.5-year northern-tracks fit (arXiv:2111.10299): the index and
#: the ``nu_mu`` normalization [GeV^-1 cm^-2 s^-1 sr^-1 at 100 TeV], the
#: anchors of the whole analysis. Their fit includes ``tau -> mu`` at 1:1:1,
#: and their own with/without test (Aartsen et al. 2016) puts that
#: assumption at 5% on the normalization and nothing on the index; the 5%
#: is added to the normalization width in quadrature.
TRACKS_GAMMA = (ICECUBE_TRACKS_2022.gamma, ICECUBE_TRACKS_2022.gamma_err)
TRACKS_PHI_MU = (ICECUBE_TRACKS_2022.phi0 * FLUX_UNIT,
                 float(np.hypot(ICECUBE_TRACKS_2022.phi0_err, 0.05 * ICECUBE_TRACKS_2022.phi0))
                 * FLUX_UNIT)

#: The same anchors in combined-fit units, as :class:`RecoLikelihood` takes
#: them.
ANCHORS = {"gamma": TRACKS_GAMMA,
           "phi_mu": (TRACKS_PHI_MU[0] / PIVOT_PHI0, TRACKS_PHI_MU[1] / PIVOT_PHI0)}


def physical_r_range(earth_composition: Callable) -> tuple[float, float]:
    """Earth flavour-ratio range standard oscillations allow.

    The Earth fractions are linear in the source composition, so the
    linear-fractional ratio ``f_tau / (f_mu + f_tau)`` is extremal at a
    vertex of the source simplex; the three pure-flavour sources are enough.

    Parameters
    ----------
    earth_composition : callable
        Maps a source composition ``(f_e, f_mu, f_tau)`` to the fractions
        after standard oscillations.

    Returns
    -------
    r_min, r_max : float
        Band of the ratio over all source compositions.
    """
    ratios = []
    for source in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
        _, f_mu, f_tau = earth_composition(source)
        ratios.append(f_tau / (f_mu + f_tau))
    return min(ratios), max(ratios)


# ---------------------------------------------------------------------------
# Inputs: smearing marginal, banded responses, atmosphere, events
# ---------------------------------------------------------------------------


def smearing_marginal(
    data_dir: str | pathlib.Path | None = None,
    reco_edges: np.ndarray = RECO_EDGES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Energy marginal of the IC86 smearing table on a common reco grid.

    Sums the released fractional counts over the point-spread and
    angular-error axes and projects each reconstructed-energy interval onto
    ``reco_edges``, uniform density assumed inside an interval.

    Parameters
    ----------
    data_dir : str or pathlib.Path or None, optional
        Root of the DR2 release; ``None`` uses
        :func:`softpaws.data.paths.dr2_dir`.
    reco_edges : np.ndarray, optional
        Reconstructed-energy bin edges [log10 GeV].

    Returns
    -------
    enu_edges : np.ndarray, shape (n_enu + 1,)
        True-energy bin edges [log10 GeV].
    dec_edges : np.ndarray, shape (n_dec + 1,)
        Upgoing declination bin edges [deg].
    marginal : np.ndarray, shape (n_enu, n_dec, n_reco)
        ``P(reco bin | true bin)``; sums over reco to one where the released
        intervals lie inside the grid.
    """
    data_dir = dr2_dir() if data_dir is None else pathlib.Path(data_dir)
    path = require(data_dir / "irfs" / "IC86_smearing.csv", "IceTracks-DR2 release")
    raw = np.genfromtxt(path, comments="#")
    raw = raw[raw[:, 10] > 0.0]
    enu_lows = np.unique(raw[:, 0])
    enu_edges = np.append(enu_lows, raw[:, 1].max())
    dec_lows_all = np.unique(raw[:, 2])
    upgoing = dec_lows_all >= -1.0e-9
    dec_lows = dec_lows_all[upgoing]
    keep = raw[:, 2] >= dec_lows[0] - 1.0e-9
    rows = raw[keep]
    dec_edges = np.append(dec_lows, rows[:, 3].max())

    i_enu = np.searchsorted(enu_lows, rows[:, 0] + 1.0e-9) - 1
    i_dec = np.searchsorted(dec_lows, rows[:, 2] + 1.0e-9) - 1
    lo, hi, weight = rows[:, 4], rows[:, 5], rows[:, 10]
    width = np.maximum(hi - lo, 1.0e-12)

    marginal = np.zeros((enu_lows.size, dec_lows.size, reco_edges.size - 1))
    for k in range(reco_edges.size - 1):
        overlap = np.clip(np.minimum(hi, reco_edges[k + 1])
                          - np.maximum(lo, reco_edges[k]), 0.0, None)
        np.add.at(marginal, (i_enu, i_dec, k), weight * overlap / width)
    return enu_edges, dec_edges, marginal


def banded_responses(
    banded_aeff: Callable,
    dec_edges_deg: np.ndarray,
    coarse_log10_e: np.ndarray,
    channels: Sequence[str] = CHANNELS,
    log10_e_grid: np.ndarray = LOG10_E_GRID,
) -> dict[str, np.ndarray]:
    """Model channel responses on the smearing declination bands [cm^2].

    The caller supplies the effective area itself, band by band and channel
    by channel, on whatever coarse energy grid it prefers; this function
    interpolates it in the logarithm onto ``log10_e_grid``, which is the
    grid the likelihood integrates over.

    Parameters
    ----------
    banded_aeff : callable
        Called as ``banded_aeff(channel, sin_dec_edges)`` and returns the
        band-averaged effective area [cm^2] of that channel, of shape
        ``(coarse_log10_e.size, n_bands)``.
    dec_edges_deg : np.ndarray
        Upgoing declination bin edges [deg].
    coarse_log10_e : np.ndarray
        Energy grid the supplied areas are given on [log10 GeV].
    channels : sequence of str, optional
        Channels to build.
    log10_e_grid : np.ndarray, optional
        Fine energy grid the responses are returned on [log10 GeV].

    Returns
    -------
    responses : dict of str -> np.ndarray
        One entry per channel, on (``log10_e_grid``, bands) [cm^2].
    """
    sin_dec_edges = np.sin(np.deg2rad(dec_edges_deg))
    out = {}
    for channel in channels:
        banded = np.asarray(banded_aeff(channel, sin_dec_edges), dtype=float)
        fine = np.empty((log10_e_grid.size, banded.shape[1]))
        with np.errstate(divide="ignore"):
            log_a = np.log10(np.clip(banded, 1.0e-30, None))
        for j in range(banded.shape[1]):
            fine[:, j] = 10.0 ** np.interp(log10_e_grid, coarse_log10_e, log_a[:, j],
                                           left=-30.0, right=log_a[-1, j])
        out[channel] = fine
    return out


def cached_fit_inputs(
    cache_path: str | pathlib.Path,
    reco_edges: np.ndarray = RECO_EDGES,
    channels: Sequence[str] = CHANNELS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]] | None:
    """Fit inputs from an earlier run, or ``None`` if the cache cannot serve.

    Parameters
    ----------
    cache_path : str or pathlib.Path
        ``.npz`` file written by :func:`fit_inputs`.
    reco_edges : np.ndarray, optional
        Reconstructed-energy bin edges the caller wants; a cache built on a
        different grid is refused.
    channels : sequence of str, optional
        Channels the cache is expected to hold.

    Returns
    -------
    inputs : tuple or None
        ``(enu_edges, dec_edges, marginal, responses)`` as
        :func:`fit_inputs` returns them, or ``None``.
    """
    cache_path = pathlib.Path(cache_path)
    if not cache_path.exists():
        return None
    cache = np.load(cache_path)
    if not np.array_equal(cache["reco_edges"], reco_edges):
        return None
    logger.info("Cached fit inputs from %s", cache_path)
    return (cache["enu_edges"], cache["dec_edges"], cache["marginal"],
            {channel: cache[f"response_{channel}"] for channel in channels})


def fit_inputs(
    data_dir: str | pathlib.Path | None,
    cache_path: str | pathlib.Path,
    build_responses: Callable[[np.ndarray], dict[str, np.ndarray]],
    rebuild: bool = False,
    reco_edges: np.ndarray = RECO_EDGES,
    channels: Sequence[str] = CHANNELS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Smearing marginal and banded responses, cached across runs.

    Parameters
    ----------
    data_dir : str or pathlib.Path or None
        Root of the DR2 release; ``None`` uses
        :func:`softpaws.data.paths.dr2_dir`.
    cache_path : str or pathlib.Path
        ``.npz`` file the inputs are read from and written to.
    build_responses : callable
        Called as ``build_responses(dec_edges)`` when the cache cannot
        serve, and returns the responses of :func:`banded_responses`.
    rebuild : bool, optional
        Ignore the cache and rebuild.
    reco_edges : np.ndarray, optional
        Reconstructed-energy bin edges [log10 GeV].
    channels : sequence of str, optional
        Channels the responses hold.

    Returns
    -------
    enu_edges : np.ndarray
        True-energy bin edges of the smearing table [log10 GeV].
    dec_edges : np.ndarray
        Upgoing declination bin edges [deg].
    marginal : np.ndarray
        Smearing marginal, as :func:`smearing_marginal` returns it.
    responses : dict of str -> np.ndarray
        Channel responses on (``LOG10_E_GRID``, bands) [cm^2].
    """
    if not rebuild:
        cached = cached_fit_inputs(cache_path, reco_edges, channels)
        if cached is not None:
            return cached
    enu_edges, dec_edges, marginal = smearing_marginal(data_dir, reco_edges)
    responses = build_responses(dec_edges)
    cache_path = pathlib.Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, reco_edges=reco_edges, enu_edges=enu_edges,
             dec_edges=dec_edges, marginal=marginal,
             **{f"response_{channel}": responses[channel] for channel in channels})
    return enu_edges, dec_edges, marginal, responses


def load_banded_responses(
    cache_path: str | pathlib.Path,
    dec_edges_deg: np.ndarray,
    channels: Sequence[str] = CHANNELS,
) -> dict[str, np.ndarray] | None:
    """Banded responses from an earlier run, or ``None`` if the bands differ.

    Parameters
    ----------
    cache_path : str or pathlib.Path
        ``.npz`` file written by :func:`save_banded_responses`.
    dec_edges_deg : np.ndarray
        Declination bin edges the caller wants [deg].
    channels : sequence of str, optional
        Channels the cache is expected to hold.

    Returns
    -------
    responses : dict of str -> np.ndarray or None
        The cached responses, or ``None``.
    """
    cache_path = pathlib.Path(cache_path)
    if not cache_path.exists():
        return None
    cache = np.load(cache_path)
    if not np.array_equal(cache["dec_edges"], dec_edges_deg):
        return None
    logger.info("Cached banded responses from %s", cache_path)
    return {channel: cache[f"response_{channel}"] for channel in channels}


def save_banded_responses(
    cache_path: str | pathlib.Path,
    dec_edges_deg: np.ndarray,
    responses: dict[str, np.ndarray],
) -> None:
    """Write banded responses and their declination bands to a cache file.

    Parameters
    ----------
    cache_path : str or pathlib.Path
        Destination ``.npz`` file.
    dec_edges_deg : np.ndarray
        Declination bin edges the responses are banded on [deg].
    responses : dict of str -> np.ndarray
        Channel responses on (``LOG10_E_GRID``, bands) [cm^2].
    """
    cache_path = pathlib.Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, dec_edges=dec_edges_deg,
             **{f"response_{channel}": grid for channel, grid in responses.items()})


def atmospheric_fluxes(
    table_path: str | pathlib.Path,
    dec_edges_deg: np.ndarray,
    log10_e_grid: np.ndarray = LOG10_E_GRID,
) -> dict[str, np.ndarray]:
    """Conventional and prompt fluxes on the fine grid, one column per band.

    Parameters
    ----------
    table_path : str or pathlib.Path
        MCEq table, as :func:`softpaws.fluxes.build_mceq_table` writes it.
    dec_edges_deg : np.ndarray
        Declination bin edges [deg]; each band is taken at its centre.
    log10_e_grid : np.ndarray, optional
        Energy grid the fluxes are returned on [log10 GeV].

    Returns
    -------
    fluxes : dict of str -> np.ndarray
        ``"conv"`` and ``"prompt"`` on (``log10_e_grid``, bands)
        [GeV^-1 cm^-2 s^-1 sr^-1].

    Raises
    ------
    FileNotFoundError
        Raised if the table is missing.
    """
    table_path = pathlib.Path(table_path)
    if not table_path.exists():
        raise FileNotFoundError(
            f"{table_path} not found; tabulate the atmospheric flux once with "
            "softpaws.fluxes.build_mceq_table.")
    flux = AtmosphericFlux(load_mceq_table(table_path))
    centers = 0.5 * (dec_edges_deg[:-1] + dec_edges_deg[1:])
    energy = 10.0**log10_e_grid
    return {c: flux.component_on_grid(c, energy, centers) for c in ("conv", "prompt")}


def binned_events(
    data_dir: str | pathlib.Path | None,
    dec_edges_deg: np.ndarray,
    reco_edges: np.ndarray = RECO_EDGES,
    seasons: tuple[str, ...] = IC86_SEASONS,
) -> tuple[np.ndarray, int]:
    """IC86 upgoing events histogrammed on (reco grid, declination bands).

    Parameters
    ----------
    data_dir : str or pathlib.Path or None
        Root of the DR2 release; ``None`` uses
        :func:`softpaws.data.paths.dr2_dir`.
    dec_edges_deg : np.ndarray
        Declination bin edges [deg]; events below the first edge are cut.
    reco_edges : np.ndarray, optional
        Reconstructed-energy bin edges [log10 GeV].
    seasons : tuple of str, optional
        Seasons to combine. Defaults to the eleven IC86 seasons.

    Returns
    -------
    counts : np.ndarray, shape (n_reco, n_dec)
        Events per bin.
    n_upgoing : int
        Events above the first declination edge, whether or not their
        ``log10_energy`` falls inside the reco grid.
    """
    events = load_events(data_dir, seasons)
    sel = events.dec >= dec_edges_deg[0]
    counts, _, _ = np.histogram2d(events.log10_energy[sel], events.dec[sel],
                                  bins=(reco_edges, dec_edges_deg))
    return counts, int(sel.sum())


def true_counts(
    response: np.ndarray,
    flux: np.ndarray,
    enu_edges: np.ndarray,
    d_omega: np.ndarray,
    livetime_s: float,
    log10_e_grid: np.ndarray = LOG10_E_GRID,
) -> np.ndarray:
    """Expected counts per (true-energy bin, declination band).

    The response and the flux live on the fine grid; the integral over each
    true-energy bin of the smearing table is a trapezoid over the fine grid
    points inside it, so a bin holding fewer than two points contributes
    nothing.

    Parameters
    ----------
    response : np.ndarray
        Channel response on (``log10_e_grid``, bands) [cm^2].
    flux : np.ndarray
        Differential flux on the same grid [GeV^-1 cm^-2 s^-1 sr^-1].
    enu_edges : np.ndarray
        True-energy bin edges of the smearing table [log10 GeV].
    d_omega : np.ndarray
        Solid angle per declination band [sr].
    livetime_s : float
        Exposure [s].
    log10_e_grid : np.ndarray, optional
        Energy grid the response and the flux are given on [log10 GeV].

    Returns
    -------
    counts : np.ndarray, shape (n_enu, n_dec)
        Expected events per true-energy bin and band.
    """
    energy = 10.0**log10_e_grid
    integrand = response * flux
    counts = np.zeros((enu_edges.size - 1, d_omega.size))
    for i in range(enu_edges.size - 1):
        sel = (log10_e_grid >= enu_edges[i]) & (log10_e_grid <= enu_edges[i + 1])
        if sel.sum() < 2:
            continue
        counts[i] = np.trapezoid(integrand[sel], energy[sel], axis=0)
    return livetime_s * counts * d_omega[None, :]


# ---------------------------------------------------------------------------
# The likelihood in reconstructed space
# ---------------------------------------------------------------------------


class RecoLikelihood:
    """Poisson likelihood over (reco energy, declination band) bins.

    The model enters as one response per channel, the background as the two
    atmospheric fluxes, and the free parameters are the flavour ratio ``r``,
    the astrophysical normalization in units of :data:`PIVOT_PHI0`, the
    spectral index, and the two atmospheric normalizations. Each bin's
    deviance is scaled by ``1 + MODEL_SYS^2 mu``, which de-weights the
    highest-statistics atmospheric bins and leaves the Poisson-limited tail
    untouched, and Gaussian priors hold the atmospheric normalizations and,
    when anchors are supplied, the index and the ``nu_mu`` flux.

    Parameters
    ----------
    enu_edges : np.ndarray
        True-energy bin edges of the smearing table [log10 GeV].
    marginal : np.ndarray
        Smearing marginal, as :func:`smearing_marginal` returns it.
    responses : dict of str -> np.ndarray
        ``"mu"`` and ``"tau"`` responses on (:data:`LOG10_E_GRID`, bands)
        [cm^2].
    atmos : dict of str -> np.ndarray
        ``"conv"`` and ``"prompt"`` fluxes on the same grid.
    dec_edges_deg : np.ndarray
        Declination bin edges [deg].
    livetime_s : float
        Exposure [s].
    data_counts : np.ndarray
        Observed counts on (:data:`RECO_EDGES`, bands).
    anchors : dict, optional
        External Gaussian anchors, ``{"gamma": (centre, width),
        "phi_mu": (centre, width)}``, the ``nu_mu`` flux in units of
        :data:`PIVOT_PHI0`.

    Attributes
    ----------
    data : np.ndarray
        Observed counts on the fit window.
    """

    def __init__(self, enu_edges, marginal, responses, atmos, dec_edges_deg,
                 livetime_s, data_counts, anchors=None):
        self._anchors = dict(anchors or {})
        self._enu_edges = enu_edges
        self._marginal = marginal
        self._responses = responses
        self._livetime_s = livetime_s
        self._d_omega = 2.0 * np.pi * np.diff(np.sin(np.deg2rad(dec_edges_deg)))
        self._window = ((RECO_EDGES[:-1] >= FIT_RECO[0] - 1.0e-9)
                        & (RECO_EDGES[1:] <= FIT_RECO[1] + 1.0e-9))
        self.data = data_counts[self._window]
        self._marginal_tau = self._shifted_marginal(enu_edges, marginal)
        self._folded_conv = self._fold(self._true_counts("mu", atmos["conv"]))
        self._folded_prompt = self._fold(self._true_counts("mu", atmos["prompt"]))
        self._astro_cache = {
            channel: np.stack([self._astro_direct(channel, gamma)
                               for gamma in GAMMA_GRID])
            for channel in CHANNELS
        }

    def _true_counts(self, channel, flux):
        """Counts per (true bin, band) for one channel and flux grid."""
        return true_counts(self._responses[channel], flux, self._enu_edges,
                           self._d_omega, self._livetime_s)

    @staticmethod
    def _shifted_marginal(enu_edges, marginal):
        """Tau-channel smearing: the ``nu_mu`` marginal at shifted energy.

        The released table is ``nu_mu`` CC simulation, whose muon carries the
        full ``(1 - y)`` share of the neutrino energy; in the tau chain the
        muon keeps only the decay fraction ``x`` of that share. A tau event
        at ``E_nu`` therefore reconstructs like a ``nu_mu`` event at
        ``x E_nu``, so each tau true-energy bin takes the reco distribution
        of the ``nu_mu`` bin containing ``x E_nu``, averaged over the decay
        spectrum on :data:`TAU_DECAY_X` and clipped at the table edge.
        """
        centers = 0.5 * (enu_edges[:-1] + enu_edges[1:])
        f = 5.0 / 3.0 - 3.0 * TAU_DECAY_X**2 + 4.0 * TAU_DECAY_X**3 / 3.0
        weights = f / f.sum()
        shifted = np.zeros_like(marginal)
        for i, center in enumerate(centers):
            for x, w in zip(TAU_DECAY_X, weights):
                j = int(np.clip(
                    np.searchsorted(enu_edges, center + np.log10(x)) - 1,
                    0, centers.size - 1))
                shifted[i] += w * marginal[j]
        return shifted

    def _fold(self, counts, channel="mu"):
        """True-space counts through the smearing marginal, window applied."""
        marginal = self._marginal_tau if channel == "tau" else self._marginal
        reco = np.einsum("ij,ijk->kj", counts, marginal)
        return reco[self._window]

    def _astro_direct(self, channel, gamma):
        """Folded astrophysical counts at one index, without interpolation."""
        energy = 10.0**LOG10_E_GRID
        flux = 2.0 * PIVOT_PHI0 * (energy[:, None] / 1.0e5) ** (-gamma)
        return self._fold(self._true_counts(
            channel, flux * np.ones((1, self._d_omega.size))), channel)

    def _astro(self, channel, gamma):
        """Folded astrophysical counts, interpolated on :data:`GAMMA_GRID`."""
        i = int(np.clip(np.searchsorted(GAMMA_GRID, gamma) - 1, 0,
                        GAMMA_GRID.size - 2))
        weight = ((gamma - GAMMA_GRID[i])
                  / (GAMMA_GRID[i + 1] - GAMMA_GRID[i]))
        cache = self._astro_cache[channel]
        return (1.0 - weight) * cache[i] + weight * cache[i + 1]

    def expectation(self, r, norm, gamma, a_conv, a_prompt):
        """Expected counts on the fit window at one parameter point.

        Parameters
        ----------
        r : float
            Flavour ratio ``f_tau / (f_mu + f_tau)`` at Earth.
        norm : float
            Astrophysical normalization [units of :data:`PIVOT_PHI0`].
        gamma : float
            Astrophysical spectral index.
        a_conv, a_prompt : float
            Atmospheric normalizations.

        Returns
        -------
        mu : np.ndarray
            Expected counts on (window bins, bands).
        """
        astro = norm * ((1.0 - r) * self._astro("mu", gamma)
                        + r * self._astro("tau", gamma))
        return a_conv * self._folded_conv + a_prompt * self._folded_prompt + astro

    def _deviance(self, mu, phi_mu, gamma, a_conv, a_prompt, data):
        """Scaled deviance plus the prior penalties for one expectation.

        ``phi_mu`` [combined-fit units] and ``gamma`` enter only through the
        external anchors, when the likelihood carries them.
        """
        mu = np.clip(mu, 1.0e-12, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            deviance = 2.0 * (mu - data + np.where(
                data > 0.0, data * np.log(data / mu), 0.0))
        scaled = deviance / (1.0 + MODEL_SYS**2 * mu)
        penalty = (((a_conv - CONV_PRIOR[0]) / CONV_PRIOR[1]) ** 2
                   + ((a_prompt - PROMPT_PRIOR[0]) / PROMPT_PRIOR[1]) ** 2)
        for name, value in (("gamma", gamma), ("phi_mu", phi_mu)):
            if name in self._anchors:
                center, width = self._anchors[name]
                penalty += ((value - center) / width) ** 2
        return float(np.sum(scaled)) + penalty

    @staticmethod
    def _out_of_bounds(gamma, a_conv, a_prompt) -> bool:
        """True if a nuisance point is outside the physical bounds."""
        return (a_conv <= 0.0 or a_prompt < 0.0
                or not GAMMA_BOUNDS[0] <= gamma <= GAMMA_BOUNDS[1])

    def _objective(self, r, norm, gamma, a_conv, a_prompt, data):
        """Objective at one flavour-ratio parameter point."""
        if norm < 0.0 or self._out_of_bounds(gamma, a_conv, a_prompt):
            return 1.0e12
        return self._deviance(self.expectation(r, norm, gamma, a_conv, a_prompt),
                              2.0 * norm * (1.0 - r), gamma, a_conv, a_prompt,
                              data)

    def _objective_counts(self, n_mu, n_tau, gamma, a_conv, a_prompt, data):
        """Objective with each channel's astrophysical track count fixed.

        The shape still follows the index; only the window totals are held,
        so the count plane asks what the data say about *how many* tracks
        each channel contributes.
        """
        if self._out_of_bounds(gamma, a_conv, a_prompt):
            return 1.0e12
        a_mu, a_tau = self._astro("mu", gamma), self._astro("tau", gamma)
        astro = n_mu * a_mu / a_mu.sum() + n_tau * a_tau / a_tau.sum()
        mu = a_conv * self._folded_conv + a_prompt * self._folded_prompt + astro
        return self._deviance(mu, 2.0 * n_mu / a_mu.sum(), gamma, a_conv,
                              a_prompt, data)

    def channel_yield_ratio(self, gamma) -> float:
        """Tracks per unit flux, tau channel over ``nu_mu`` channel.

        Parameters
        ----------
        gamma : float
            Astrophysical spectral index.

        Returns
        -------
        ratio : float
            Window counts of the tau channel over the ``nu_mu`` channel.
        """
        return float(self._astro("tau", gamma).sum()
                     / self._astro("mu", gamma).sum())

    @staticmethod
    def _minimize(function, starts):
        """Best Nelder-Mead result over a list of simplex seeds."""
        best = None
        for x0 in starts:
            trial = minimize(function, x0=np.asarray(x0, dtype=float),
                             method="Nelder-Mead",
                             options={"xatol": 1.0e-4, "fatol": 1.0e-6,
                                      "maxiter": 6000})
            if best is None or trial.fun < best.fun:
                best = trial
        return best.fun, best.x

    def delta_ll(self, r, data=None, warm_start=None,
                 use_default_start=True) -> tuple[float, np.ndarray]:
        """Profiled objective at one flavour ratio, and the nuisances.

        Parameters
        ----------
        r : float
            Flavour ratio the fit is conditioned on.
        data : np.ndarray, optional
            Counts on the fit window; the observed counts when omitted.
        warm_start : np.ndarray, optional
            Extra simplex seed, typically a neighbouring fit's nuisances.
        use_default_start : bool, optional
            Also seed from the global default point; toys switch this off
            once warm-started.

        Returns
        -------
        value : float
            Profiled objective.
        params : np.ndarray
            Nuisances ``(norm, gamma, a_conv, a_prompt)`` at the minimum.
        """
        data = self.data if data is None else data
        starts = [(1.0, PIVOT_GAMMA, 1.0, 1.0)] if use_default_start else []
        if warm_start is not None:
            starts.append(warm_start)
        return self._minimize(
            lambda p: self._objective(r, p[0], p[1], p[2], p[3], data), starts)

    def delta_ll_fluxes(self, phi_mu, phi_tau, warm_start=None):
        """Profiled objective at fixed per-flavour fluxes.

        ``phi_mu`` and ``phi_tau`` are in units of the combined-fit
        per-flavour flux; the index and the atmospheric normalizations are
        profiled. The point maps onto :meth:`expectation` through
        ``norm = (phi_mu + phi_tau) / 2`` and
        ``r = phi_tau / (phi_mu + phi_tau)``.

        Parameters
        ----------
        phi_mu, phi_tau : float
            Per-flavour fluxes [units of :data:`PIVOT_PHI0`].
        warm_start : np.ndarray, optional
            Extra simplex seed ``(gamma, a_conv, a_prompt)``.

        Returns
        -------
        value : float
            Profiled objective.
        params : np.ndarray
            Nuisances ``(gamma, a_conv, a_prompt)`` at the minimum.
        """
        total = phi_mu + phi_tau
        r, norm = (0.0, 0.0) if total <= 0.0 else (phi_tau / total, 0.5 * total)
        starts = [(PIVOT_GAMMA, 1.0, 1.0)]
        if warm_start is not None:
            starts.append(warm_start)
        return self._minimize(
            lambda p: self._objective(r, norm, p[0], p[1], p[2], self.data),
            starts)

    def delta_ll_counts(self, n_mu, n_tau, warm_start=None):
        """Profiled objective at fixed per-channel track counts.

        Parameters
        ----------
        n_mu, n_tau : float
            Astrophysical tracks each channel puts in the fit window.
        warm_start : np.ndarray, optional
            Extra simplex seed ``(gamma, a_conv, a_prompt)``.

        Returns
        -------
        value : float
            Profiled objective.
        params : np.ndarray
            Nuisances ``(gamma, a_conv, a_prompt)`` at the minimum.
        """
        starts = [(PIVOT_GAMMA, 1.0, 1.0)]
        if warm_start is not None:
            starts.append(warm_start)
        return self._minimize(
            lambda p: self._objective_counts(n_mu, n_tau, p[0], p[1], p[2],
                                             self.data), starts)

    def delta_ll_tau(self, phi_tau, warm_start=None):
        """Profiled objective at a fixed tau flux, the ``nu_mu`` flux free.

        With the anchors carried, this is the tau-flux profile at IceCube's
        measured ``nu_mu`` flux and index.

        Parameters
        ----------
        phi_tau : float
            Tau flux [units of :data:`PIVOT_PHI0`].
        warm_start : np.ndarray, optional
            Extra simplex seed ``(phi_mu, gamma, a_conv, a_prompt)``.

        Returns
        -------
        value : float
            Profiled objective.
        params : np.ndarray
            Nuisances ``(phi_mu, gamma, a_conv, a_prompt)`` at the minimum.
        """
        def objective(p):
            phi_mu, gamma, a_conv, a_prompt = p
            if phi_mu < 0.0:
                return 1.0e12
            total = phi_mu + phi_tau
            r, norm = ((0.0, 0.0) if total <= 0.0
                       else (phi_tau / total, 0.5 * total))
            return self._objective(r, norm, gamma, a_conv, a_prompt, self.data)

        starts = [(0.8, PIVOT_GAMMA, 1.0, 1.0)]
        if warm_start is not None:
            starts.append(warm_start)
        return self._minimize(objective, starts)

    def tau_profile(self):
        """``2 Delta ln L`` on :data:`PHI_TAU_SCAN`, floored at its minimum.

        Returns
        -------
        curve : np.ndarray
            Profile statistic on :data:`PHI_TAU_SCAN`.
        """
        curve, warm = [], None
        for phi_tau in PHI_TAU_SCAN:
            value, warm = self.delta_ll_tau(phi_tau, warm)
            curve.append(value)
        curve = np.array(curve)
        return curve - curve.min()

    def _plane(self, x_grid, y_grid, fit):
        """``2 Delta ln L`` over a grid of two held parameters."""
        grid = np.empty((x_grid.size, y_grid.size))
        warm = None
        for i, x in enumerate(x_grid):
            for j, y in enumerate(y_grid):
                grid[i, j], warm = fit(x, y, warm)
        return grid - grid.min()

    def flux_plane(self):
        """``2 Delta ln L`` on (:data:`PHI_MU_GRID`, :data:`PHI_TAU_GRID`).

        Floored at its minimum over the whole plane, physical or not, so the
        valley the ratio scan walks along is visible in full.

        Returns
        -------
        plane : np.ndarray
            Profile statistic on the two grids.
        """
        return self._plane(PHI_MU_GRID, PHI_TAU_GRID, self.delta_ll_fluxes)

    def count_plane(self):
        """``2 Delta ln L`` on (:data:`N_MU_GRID`, :data:`N_TAU_GRID`).

        Returns
        -------
        plane : np.ndarray
            Profile statistic on the two grids.
        """
        return self._plane(N_MU_GRID, N_TAU_GRID, self.delta_ll_counts)

    def profile(self, data=None):
        """``2 Delta ln L`` on :data:`R_GRID` with the per-point nuisances.

        Parameters
        ----------
        data : np.ndarray, optional
            Counts on the fit window; the observed counts when omitted.

        Returns
        -------
        curve : np.ndarray
            Profile statistic on :data:`R_GRID`, floored at its minimum.
        nuisances : np.ndarray, shape (R_GRID.size, 4)
            Profiled ``(norm, gamma, a_conv, a_prompt)`` per grid point.
        """
        curves, nuisances = [], []
        warm = None
        for r in R_GRID:
            value, params = self.delta_ll(r, data=data, warm_start=warm)
            curves.append(value)
            nuisances.append(params)
            warm = params
        curve = np.array(curves)
        return curve - curve.min(), np.array(nuisances)
