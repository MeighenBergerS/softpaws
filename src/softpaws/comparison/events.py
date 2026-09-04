"""The DR2 event benchmark: a prediction with every input pinned outside.

:mod:`softpaws.comparison.reco_likelihood` fits the model to the IC86 upgoing
events with four profiled parameters. This module does the simpler and
harder thing: it predicts the same sample with nothing fitted. The
atmospheric background is the MCEq table at normalization one, the
astrophysical flux is IceCube's 9.5-year tracks measurement at a 1:1:1
composition, and the model responses are folded through the released
smearing marginal exactly as the fit folds them.

The band on the prediction is external-input uncertainty only: the hadronic
spread on each atmospheric normalization and the tracks measurement's own
normalization and index errors. Each source is a normalization error,
correlated across every bin, so it adds linearly over whatever bins are
summed and only the sources combine in quadrature -- which is what
:func:`combine_band` does.

:func:`published_response` gives the baseline the model has to be judged
against: the same fluxes through IceCube's own released effective area. If
both land on the data, the benchmark tests the fluxes and the smearing, not
the transport.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable

import numpy as np

from softpaws.data.loader import parse_aeff
from softpaws.data.paths import dr2_dir, require

from .reco_likelihood import (
    LOG10_E_GRID,
    TRACKS_GAMMA,
    TRACKS_PHI_MU,
    RecoLikelihood,
    true_counts,
)

__all__ = [
    "ASTRO_GAMMA",
    "ASTRO_GAMMA_ERR",
    "ASTRO_PHI",
    "ASTRO_PHI_ERR",
    "ATM_ERR",
    "astro_flux",
    "combine_band",
    "predict",
    "published_response",
]

#: The externally pinned astrophysical flux: IceCube's 9.5-year tracks fit,
#: per flavour at the 100 TeV pivot (arXiv:2111.10299), applied to the
#: ``nu_mu`` and ``nu_tau`` channels alike (1:1:1).
ASTRO_PHI, ASTRO_PHI_ERR = TRACKS_PHI_MU
ASTRO_GAMMA, ASTRO_GAMMA_ERR = TRACKS_GAMMA

#: Hadronic-model spread carried as the atmospheric normalization error.
ATM_ERR = 0.25


def astro_flux(
    gamma: float,
    phi0: float = ASTRO_PHI,
    log10_e_grid: np.ndarray = LOG10_E_GRID,
) -> np.ndarray:
    """Per-flavour power law on the fine grid [GeV^-1 cm^-2 s^-1 sr^-1].

    Parameters
    ----------
    gamma : float
        Spectral index.
    phi0 : float, optional
        Normalization at the 100 TeV pivot
        [GeV^-1 cm^-2 s^-1 sr^-1]; the tracks fit by default.
    log10_e_grid : np.ndarray, optional
        Energy grid [log10 GeV].

    Returns
    -------
    flux : np.ndarray, shape (n_energy, 1)
        Differential flux, one column so it broadcasts over the bands.
    """
    energy = 10.0**log10_e_grid
    return phi0 * (energy[:, None] / 1.0e5) ** (-gamma)


def published_response(
    data_dir: str | pathlib.Path | None,
    enu_edges: np.ndarray,
    dec_edges: np.ndarray,
    log10_e_grid: np.ndarray = LOG10_E_GRID,
) -> dict[str, np.ndarray]:
    """IceCube's own DR2 effective area on the model's response grid.

    The released ``IC86_effectiveArea.csv`` is an average over each (true
    energy, true declination) bin, so it is broadcast piecewise-constant
    onto ``log10_e_grid`` inside each smearing bin and taken at the band's
    own declination. The table is a ``nu_mu`` area, so the tau channel is
    zero in this response.

    Parameters
    ----------
    data_dir : str or pathlib.Path or None
        Root of the DR2 release; ``None`` uses
        :func:`softpaws.data.paths.dr2_dir`.
    enu_edges : np.ndarray
        True-energy bin edges of the smearing table [log10 GeV].
    dec_edges : np.ndarray
        Upgoing declination bin edges [deg].
    log10_e_grid : np.ndarray, optional
        Energy grid the response is returned on [log10 GeV].

    Returns
    -------
    responses : dict of str -> np.ndarray
        ``"mu"`` on (``log10_e_grid``, bands) [cm^2], and ``"tau"`` of zeros.
    """
    data_dir = dr2_dir() if data_dir is None else pathlib.Path(data_dir)
    path = require(data_dir / "irfs" / "IC86_effectiveArea.csv", "IceTracks-DR2 release")
    raw = np.genfromtxt(path, comments="#")
    aeff = parse_aeff(raw)
    dec_centers = 0.5 * (dec_edges[:-1] + dec_edges[1:])
    log10_e = np.clip(log10_e_grid, enu_edges[0] + 1.0e-9, enu_edges[-1] - 1.0e-9)
    i_enu = np.searchsorted(enu_edges, log10_e, side="right") - 1
    table_e_centers = aeff.log10_energy_centers
    table_sin_edges = aeff.sin_dec_edges
    i_table_e = np.searchsorted(table_e_centers, 0.5 * (enu_edges[i_enu] + enu_edges[i_enu + 1]))
    i_table_e = np.clip(i_table_e - 1, 0, table_e_centers.size - 1)
    i_table_d = np.clip(np.searchsorted(table_sin_edges, np.sin(np.deg2rad(dec_centers)),
                                        side="right") - 1, 0, table_sin_edges.size - 2)
    mu = aeff.values[np.ix_(i_table_e, i_table_d)]
    return {"mu": mu, "tau": np.zeros_like(mu)}


def predict(
    responses: dict[str, np.ndarray],
    atmos: dict[str, np.ndarray],
    enu_edges: np.ndarray,
    dec_edges: np.ndarray,
    marginal: np.ndarray,
    livetime_s: float,
    gamma: float = ASTRO_GAMMA,
    gamma_err: float = ASTRO_GAMMA_ERR,
    phi0: float = ASTRO_PHI,
    phi0_err: float = ASTRO_PHI_ERR,
    atm_err: float = ATM_ERR,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray | tuple]]:
    """Reconstructed-space components on the full reco grid, per band.

    Parameters
    ----------
    responses : dict of str -> np.ndarray
        ``"mu"`` and ``"tau"`` responses on (``LOG10_E_GRID``, bands) [cm^2].
    atmos : dict of str -> np.ndarray
        ``"conv"`` and ``"prompt"`` fluxes on the same grid.
    enu_edges : np.ndarray
        True-energy bin edges of the smearing table [log10 GeV].
    dec_edges : np.ndarray
        Declination bin edges [deg].
    marginal : np.ndarray
        Smearing marginal, as
        :func:`~softpaws.comparison.reco_likelihood.smearing_marginal`
        returns it.
    livetime_s : float
        Exposure [s].
    gamma, gamma_err : float, optional
        Astrophysical index and its error.
    phi0, phi0_err : float, optional
        Per-flavour normalization at 100 TeV and its error
        [GeV^-1 cm^-2 s^-1 sr^-1].
    atm_err : float, optional
        Fractional error on each atmospheric normalization.

    Returns
    -------
    components : dict of str -> np.ndarray
        ``"conv"``, ``"prompt"``, ``"astro_mu"``, ``"astro_tau"`` on
        (reco bins, declination bands).
    errors : dict of str -> np.ndarray or tuple
        Signed one-sigma grids per external input, same shape, each fully
        correlated across bins; combine with :func:`combine_band`.
    """
    d_omega = 2.0 * np.pi * np.diff(np.sin(np.deg2rad(dec_edges)))
    marginal_tau = RecoLikelihood._shifted_marginal(enu_edges, marginal)

    def fold(counts, tau=False):
        return np.einsum("ij,ijk->kj", counts, marginal_tau if tau else marginal)

    ones = np.ones((1, d_omega.size))
    flux = astro_flux(gamma, phi0) * ones
    components = {
        "conv": fold(true_counts(responses["mu"], atmos["conv"], enu_edges,
                                 d_omega, livetime_s)),
        "prompt": fold(true_counts(responses["mu"], atmos["prompt"], enu_edges,
                                   d_omega, livetime_s)),
        "astro_mu": fold(true_counts(responses["mu"], flux, enu_edges,
                                     d_omega, livetime_s)),
        "astro_tau": fold(true_counts(responses["tau"], flux, enu_edges,
                                      d_omega, livetime_s), tau=True),
    }

    astro = components["astro_mu"] + components["astro_tau"]
    spread = []
    for varied_gamma in (gamma - gamma_err, gamma + gamma_err):
        flux_g = astro_flux(varied_gamma, phi0) * ones
        varied = (fold(true_counts(responses["mu"], flux_g, enu_edges,
                                   d_omega, livetime_s))
                  + fold(true_counts(responses["tau"], flux_g, enu_edges,
                                     d_omega, livetime_s), tau=True))
        spread.append(varied - astro)
    errors = {
        "conv": atm_err * components["conv"],
        "prompt": atm_err * components["prompt"],
        "astro_norm": phi0_err / phi0 * astro,
        "astro_gamma": (spread[0], spread[1]),
    }
    return components, errors


def combine_band(errors: dict, reduce: Callable) -> np.ndarray:
    """One-sigma external-input band after summing bins with ``reduce``.

    Every entry of ``errors`` is one normalization or index error, so it is
    fully correlated across bins and its grid adds linearly under ``reduce``;
    the sources then add in quadrature. Adding the per-bin grids in
    quadrature across bins instead would shrink a 25% normalization error
    by roughly the square root of the number of bins.

    Parameters
    ----------
    errors : dict of str -> np.ndarray or tuple
        Signed one-sigma grids per source, as returned by :func:`predict`.
        ``"astro_gamma"`` holds the two signed index variations, which
        change sign across the pivot and so must be reduced before taking
        their magnitude.
    reduce : callable
        Maps a (reco bins, declination bands) grid to the sum wanted, for
        example ``lambda g: g[window].sum()`` or ``lambda g: g.sum(axis=1)``.

    Returns
    -------
    band : np.ndarray or float
        One-sigma band of whatever shape ``reduce`` returns.
    """
    terms = []
    for key, grid in errors.items():
        if key == "astro_gamma":
            lo, hi = grid
            terms.append(0.5 * (np.abs(reduce(lo)) + np.abs(reduce(hi))))
        else:
            terms.append(reduce(grid))
    return np.sqrt(sum(term**2 for term in terms))
