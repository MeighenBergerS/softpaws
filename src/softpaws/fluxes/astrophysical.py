"""Astrophysical neutrino fluxes: power laws and the published fits.

Every flux here is a per-flavour ``nu + nubar`` flux at Earth, differential
in energy, in ``GeV^-1 cm^-2 s^-1 sr^-1``. Normalizations are quoted at the
100 TeV pivot in units of ``1e-18``, which puts them near one for the
measured diffuse flux.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "FLUX_PIVOT_GEV",
    "FLUX_UNIT",
    "ICECUBE_BPL_2025",
    "ICECUBE_CASCADES_2020",
    "ICECUBE_COMBINED_2023",
    "ICECUBE_TRACKS_2022",
    "REFERENCE_SPL",
    "BrokenPowerLawFit",
    "PowerLawFit",
    "broken_power_law_flux",
    "broken_power_law_shape",
    "power_law_flux",
]

#: Pivot energy of every normalization here [GeV].
FLUX_PIVOT_GEV = 1.0e5

#: Unit of the normalizations, ``1e-18 GeV^-1 cm^-2 s^-1 sr^-1``.
FLUX_UNIT = 1.0e-18


def power_law_flux(
    energy_gev: float | np.ndarray,
    phi0: float,
    gamma: float,
) -> np.ndarray:
    """Single power-law diffuse flux.

    Parameters
    ----------
    energy_gev : float or np.ndarray
        Neutrino energy [GeV].
    phi0 : float
        Normalization at the 100 TeV pivot [1e-18 GeV^-1 cm^-2 s^-1 sr^-1].
    gamma : float
        Spectral index, ``phi ~ E^-gamma``.

    Returns
    -------
    flux : np.ndarray
        Differential flux [GeV^-1 cm^-2 s^-1 sr^-1].

    Examples
    --------
    >>> float(power_law_flux(1.0e5, 1.0, 2.0))
    1e-18
    """
    energy = np.atleast_1d(np.asarray(energy_gev, dtype=float))
    return (phi0 * FLUX_UNIT) * (energy / FLUX_PIVOT_GEV) ** (-gamma)


def broken_power_law_shape(
    energy_gev: float | np.ndarray,
    gamma_2: float,
    gamma_1: float = 1.31,
    log10_break_gev: float = 4.39,
) -> np.ndarray:
    """Broken power law with unit flux at 100 TeV on the upper branch.

    The lower branch is continuous at the break. The defaults are the fixed
    lower index and break of :data:`ICECUBE_BPL_2025`.

    Parameters
    ----------
    energy_gev : float or np.ndarray
        Neutrino energy [GeV].
    gamma_2 : float
        Spectral index above the break.
    gamma_1 : float, optional
        Spectral index below the break.
    log10_break_gev : float, optional
        Break energy [log10 GeV].

    Returns
    -------
    shape : np.ndarray
        Flux relative to its value at the pivot.
    """
    energy = np.asarray(energy_gev, dtype=float)
    e_break = 10.0**log10_break_gev
    above = (energy / FLUX_PIVOT_GEV) ** (-gamma_2)
    below = (e_break / FLUX_PIVOT_GEV) ** (-gamma_2) * (energy / e_break) ** (-gamma_1)
    return np.where(energy >= e_break, above, below)


def broken_power_law_flux(
    energy_gev: float | np.ndarray,
    phi0: float,
    gamma_2: float,
    gamma_1: float = 1.31,
    log10_break_gev: float = 4.39,
) -> np.ndarray:
    """Broken power-law diffuse flux.

    Parameters
    ----------
    energy_gev : float or np.ndarray
        Neutrino energy [GeV].
    phi0 : float
        Normalization at the 100 TeV pivot on the upper branch
        [1e-18 GeV^-1 cm^-2 s^-1 sr^-1].
    gamma_2 : float
        Spectral index above the break.
    gamma_1 : float, optional
        Spectral index below the break.
    log10_break_gev : float, optional
        Break energy [log10 GeV].

    Returns
    -------
    flux : np.ndarray
        Differential flux [GeV^-1 cm^-2 s^-1 sr^-1].
    """
    return (phi0 * FLUX_UNIT) * broken_power_law_shape(
        energy_gev, gamma_2, gamma_1, log10_break_gev
    )


@dataclass(frozen=True)
class PowerLawFit:
    """A published single power-law fit.

    Attributes
    ----------
    name : str
        Short label.
    phi0 : float
        Per-flavour normalization at 100 TeV [1e-18 GeV^-1 cm^-2 s^-1 sr^-1].
    gamma : float
        Spectral index.
    phi0_err : float
        Symmetrized uncertainty on ``phi0`` [same units]; zero if not quoted.
    gamma_err : float
        Symmetrized uncertainty on ``gamma``; zero if not quoted.
    reference : str
        Where the numbers come from.
    """

    name: str
    phi0: float
    gamma: float
    phi0_err: float = 0.0
    gamma_err: float = 0.0
    reference: str = ""

    def flux(self, energy_gev: float | np.ndarray) -> np.ndarray:
        """Differential flux of the fit at ``energy_gev`` [GeV^-1 cm^-2 s^-1 sr^-1]."""
        return power_law_flux(energy_gev, self.phi0, self.gamma)


@dataclass(frozen=True)
class BrokenPowerLawFit:
    """A published broken power-law fit.

    Attributes
    ----------
    name : str
        Short label.
    phi0 : float
        Per-flavour normalization at 100 TeV on the upper branch
        [1e-18 GeV^-1 cm^-2 s^-1 sr^-1].
    gamma_2 : float
        Spectral index above the break.
    gamma_1 : float
        Spectral index below the break, held fixed in the fit.
    log10_break_gev : float
        Break energy [log10 GeV], held fixed in the fit.
    phi0_err : float
        Symmetrized uncertainty on ``phi0``.
    gamma_2_err : float
        Symmetrized uncertainty on ``gamma_2``.
    reference : str
        Where the numbers come from.
    """

    name: str
    phi0: float
    gamma_2: float
    gamma_1: float
    log10_break_gev: float
    phi0_err: float = 0.0
    gamma_2_err: float = 0.0
    reference: str = ""

    def shape(self, energy_gev: float | np.ndarray) -> np.ndarray:
        """Flux relative to its value at the pivot; see :func:`broken_power_law_shape`."""
        return broken_power_law_shape(energy_gev, self.gamma_2, self.gamma_1, self.log10_break_gev)

    def flux(self, energy_gev: float | np.ndarray) -> np.ndarray:
        """Differential flux of the fit at ``energy_gev`` [GeV^-1 cm^-2 s^-1 sr^-1]."""
        return broken_power_law_flux(
            energy_gev, self.phi0, self.gamma_2, self.gamma_1, self.log10_break_gev
        )


#: The reference single power law of Palmisano (arXiv:2607.13143, Eq. 1.3),
#: IceCube's 9.5-year diffuse best fit as that paper quotes it.
REFERENCE_SPL = PowerLawFit("reference", 0.63, 2.38, reference="arXiv:2607.13143 Eq. 1.3")

#: IceCube's 9.5-year northern-tracks fit. The fit includes ``tau -> mu`` at
#: 1:1:1; the collaboration's own with/without test puts that assumption at
#: 5% on the normalization and nothing on the index.
ICECUBE_TRACKS_2022 = PowerLawFit(
    "IceCube tracks", 1.44, 2.37, phi0_err=0.26, gamma_err=0.09, reference="arXiv:2111.10299"
)

#: IceCube's cascade fit.
ICECUBE_CASCADES_2020 = PowerLawFit("IceCube cascades", 1.66, 2.53, reference="arXiv:2001.09520")

#: IceCube's combined tracks-plus-cascades fit.
ICECUBE_COMBINED_2023 = PowerLawFit("IceCube combined", 1.80, 2.52, reference="arXiv:2308.00191")

#: IceCube's joint cascades-plus-tracks broken power law: the upper branch is
#: free, the lower index and the break are held.
ICECUBE_BPL_2025 = BrokenPowerLawFit(
    "IceCube BPL",
    phi0=1.77,
    gamma_2=2.735,
    gamma_1=1.31,
    log10_break_gev=4.39,
    phi0_err=0.185,
    gamma_2_err=0.071,
    reference="arXiv:2507.22234",
)
