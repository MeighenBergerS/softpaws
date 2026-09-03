"""Atmospheric ``nu_mu`` flux from an MCEq table.

MCEq (Fedynitch et al.) solves the cascade equations for the conventional
and prompt atmospheric neutrino fluxes at one zenith angle at a time. The
table built here samples it over declination once, caches the result as an
``.npz`` file, and :class:`AtmosphericFlux` interpolates that table wherever
a forward model needs the background.

MCEq is an optional dependency (``pip install softpaws[atm]``). Loading and
interpolating an existing table needs only NumPy and SciPy.
"""

from __future__ import annotations

import logging
import pathlib

import numpy as np
from scipy.interpolate import RegularGridInterpolator

__all__ = [
    "ATMOSPHERE",
    "INTERACTION_MODEL",
    "PRIMARY_MODEL",
    "TABLE_KEYS",
    "AtmosphericFlux",
    "build_mceq_table",
    "load_mceq_table",
]

logger = logging.getLogger(__name__)

#: MCEq hadronic interaction model. SIBYLL-2.3d and H3a are the standard
#: conventional and prompt baseline.
INTERACTION_MODEL = "SIBYLL2.3d"
#: MCEq primary cosmic-ray model.
PRIMARY_MODEL = "H3a"
#: MCEq atmosphere: the South Pole profile MCEq ships for IceCube, in
#: January (austral summer, the thinner atmosphere).
ATMOSPHERE = ("SouthPole", "January")
#: Arrays a table holds.
TABLE_KEYS = ("energy_gev", "dec_deg", "conv", "prompt")


def build_mceq_table(
    path: str | pathlib.Path,
    dec_range_deg: tuple[float, float] = (0.0, 90.0),
    n_dec: int = 19,
    interaction_model: str = INTERACTION_MODEL,
    primary_model: str = PRIMARY_MODEL,
    atmosphere: tuple[str, str] = ATMOSPHERE,
) -> dict[str, np.ndarray]:
    """Run MCEq once per declination and cache the ``nu_mu`` fluxes.

    A declination ``dec`` at the South Pole is a production zenith of
    ``90 deg - dec``, so ``dec = 90`` is the vertical upgoing flux and
    ``dec = 0`` the horizontal one.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination ``.npz`` file.
    dec_range_deg : tuple of float, optional
        Declination range sampled [deg].
    n_dec : int, optional
        Number of declinations sampled, evenly, over the range.
    interaction_model : str, optional
        MCEq hadronic interaction model.
    primary_model : str, optional
        MCEq primary cosmic-ray model, one of the Hillas-Gaisser variants.
    atmosphere : tuple of str, optional
        MCEq ``MSIS00_IC`` atmosphere location and month.

    Returns
    -------
    table : dict of np.ndarray
        ``energy_gev`` (n_e,), ``dec_deg`` (n_dec,), and the fluxes ``conv``
        and ``prompt`` of shape ``(n_e, n_dec)`` in
        ``GeV^-1 cm^-2 s^-1 sr^-1``, summed over ``nu_mu`` and ``nu_mu_bar``.

    Raises
    ------
    ImportError
        Raised if MCEq or crflux is not installed.
    """
    import importlib.util  # noqa: F401  (MCEq 1.4.1 uses it without importing it)

    try:
        import crflux.models as pm
        from MCEq.core import MCEqRun
    except ImportError as error:
        raise ImportError(
            "Building the atmospheric table needs MCEq and crflux: "
            "pip install 'softpaws[atm]'."
        ) from error

    path = pathlib.Path(path)
    dec_deg = np.linspace(dec_range_deg[0], dec_range_deg[1], n_dec)
    mceq = MCEqRun(
        interaction_model=interaction_model,
        primary_model=(pm.HillasGaisser2012, primary_model),
        density_model=("MSIS00_IC", atmosphere),
        theta_deg=0.0,
    )
    conv = np.empty((mceq.e_grid.size, n_dec))
    prompt = np.empty_like(conv)
    for j, dec in enumerate(dec_deg):
        theta = 90.0 - dec
        logger.info("MCEq: dec = %5.1f deg (production zenith %5.1f deg)", dec, theta)
        mceq.set_theta_deg(theta)
        mceq.solve()
        conv[:, j] = mceq.get_solution("conv_numu") + mceq.get_solution("conv_antinumu")
        prompt[:, j] = mceq.get_solution("pr_numu") + mceq.get_solution("pr_antinumu")

    table = {
        "energy_gev": np.asarray(mceq.e_grid, dtype=float),
        "dec_deg": dec_deg,
        "conv": conv,
        "prompt": prompt,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        interaction_model=interaction_model,
        primary_model=primary_model,
        atmosphere="/".join(atmosphere),
        **table,
    )
    logger.info("Flux table saved to %s", path.resolve())
    return table


def load_mceq_table(path: str | pathlib.Path, recompute: bool = False) -> dict[str, np.ndarray]:
    """Load a cached MCEq table, building it first if needed.

    Parameters
    ----------
    path : str or pathlib.Path
        Table file.
    recompute : bool, optional
        Rebuild with :func:`build_mceq_table` even if ``path`` exists.

    Returns
    -------
    table : dict of np.ndarray
        As returned by :func:`build_mceq_table`.
    """
    path = pathlib.Path(path)
    if recompute or not path.exists():
        logger.info("Building the MCEq flux table at %s (this takes minutes)", path)
        return build_mceq_table(path)
    with np.load(path) as data:
        table = {key: np.asarray(data[key]) for key in TABLE_KEYS}
        logger.info(
            "Loaded MCEq flux table %s: %s / %s / %s, %d declinations",
            path.resolve(),
            data["interaction_model"],
            data["primary_model"],
            data["atmosphere"],
            table["dec_deg"].size,
        )
    return table


class AtmosphericFlux:
    """Interpolated atmospheric ``nu_mu`` flux from an MCEq table.

    Bilinear in ``(log10 E, dec)`` on the logarithm of the flux, which is the
    interpolation both variables are smooth in. Callable in the two-argument
    form the forward models expect.

    Parameters
    ----------
    table : dict of np.ndarray
        Table from :func:`load_mceq_table`.
    include_prompt : bool, optional
        Whether to add the prompt (charm) component to the conventional one.

    Attributes
    ----------
    energy_gev : np.ndarray
        Tabulated energies [GeV].
    dec_deg : np.ndarray
        Tabulated declinations [deg].
    flux : np.ndarray, shape (n_e, n_dec)
        Tabulated flux [GeV^-1 cm^-2 s^-1 sr^-1].
    table : dict of np.ndarray
        The table itself, for :meth:`component_on_grid`.
    """

    def __init__(self, table: dict[str, np.ndarray], include_prompt: bool = True) -> None:
        self.table = table
        self.energy_gev = np.asarray(table["energy_gev"], dtype=float)
        self.dec_deg = np.asarray(table["dec_deg"], dtype=float)
        flux = np.asarray(table["conv"], dtype=float)
        if include_prompt:
            flux = flux + np.asarray(table["prompt"], dtype=float)
        self.flux = flux
        # A floor keeps the logarithm finite where the cascade solution underflows.
        floor = flux[flux > 0.0].min() * 1e-10
        self._interp = RegularGridInterpolator(
            (np.log10(self.energy_gev), self.dec_deg),
            np.log(np.maximum(flux, floor)),
            bounds_error=False,
            fill_value=None,  # linear extrapolation off the ends of the table
        )

    def __call__(
        self,
        energy_gev: float | np.ndarray,
        dec_deg: float | np.ndarray,
    ) -> np.ndarray:
        """Differential flux at an energy and declination.

        Parameters
        ----------
        energy_gev : float or np.ndarray
            Neutrino energy [GeV].
        dec_deg : float or np.ndarray
            Source declination [deg]; broadcast against ``energy_gev``.

        Returns
        -------
        flux : np.ndarray
            Differential flux [GeV^-1 cm^-2 s^-1 sr^-1].
        """
        energy, dec = np.broadcast_arrays(
            np.asarray(energy_gev, dtype=float), np.asarray(dec_deg, dtype=float)
        )
        points = np.stack([np.log10(energy), dec], axis=-1)
        return np.exp(self._interp(points))

    def component_on_grid(
        self,
        component: str,
        energy_gev: np.ndarray,
        dec_deg: np.ndarray,
    ) -> np.ndarray:
        """One component on an energy-by-declination grid, clamped at the table edges.

        Interpolates ``log10`` of the flux linearly in declination and then in
        ``log10 E``, holding the edge values outside the table instead of
        extrapolating as :meth:`__call__` does.

        Parameters
        ----------
        component : {"conv", "prompt"}
            Which component to read.
        energy_gev : np.ndarray, shape (n_e,)
            Neutrino energies [GeV].
        dec_deg : np.ndarray, shape (n_dec,)
            Declinations [deg]. Negative values are folded to their absolute
            value, since the table covers one hemisphere.

        Returns
        -------
        flux : np.ndarray, shape (n_e, n_dec)
            Differential flux [GeV^-1 cm^-2 s^-1 sr^-1].
        """
        energy_gev = np.asarray(energy_gev, dtype=float)
        dec_deg = np.abs(np.asarray(dec_deg, dtype=float))
        log_f = np.log10(np.clip(np.asarray(self.table[component], dtype=float), 1.0e-99, None))
        on_dec = np.array(
            [np.interp(dec_deg, self.dec_deg, log_f[i]) for i in range(self.energy_gev.size)]
        )
        out = np.empty((energy_gev.size, dec_deg.size))
        for j in range(dec_deg.size):
            out[:, j] = 10.0 ** np.interp(
                np.log10(energy_gev), np.log10(self.energy_gev), on_dec[:, j]
            )
        return out
