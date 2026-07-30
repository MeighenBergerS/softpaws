"""Neutrino-nucleon cross-section models.

:mod:`softpaws.transport.source` provides the paper's analytic cross section, a
single power law ``sigma_CC = sigma0 (E / E0)^lambda`` anchored at 10 PeV
(arXiv:2607.13143, Eq. 2.5). That form is what makes the soft-volume closed
form work -- a power-law flux times a power-law cross section excites exactly
one Mellin mode ``A = gamma - lambda - 1`` -- and it is also what
:func:`softpaws.comparison.likelihood.required_lambda` treats as the quantity
being measured, so it is kept as a parametrized model here
(:class:`PowerLawCrossSection`).

Extrapolated away from its anchor, though, that power law is badly wrong: it
overshoots a modern calculation by a factor of ~2.7 at 10 TeV and ~9 at 1 TeV,
which is enough to dominate any comparison against the published IceCube
effective area at those energies. :class:`TabulatedCrossSection` reads a
tabulated calculation instead (:func:`softpaws.data.loader.
load_cross_section_table`) and derives ``lambda`` from it as a **local** slope
``lambda_eff(E) = d ln sigma_CC / d ln E``, which is the same
local-power-law approximation :meth:`~softpaws.response.soft_volume.
SoftVolumeResponse.expected_counts_from_flux` already makes on the flux side.

Both value and slope come from one smoothing spline in log-log space, so they
stay mutually consistent. A spline rather than linear interpolation because
``lambda`` feeds ``A``, and differentiating a linear interpolant would give a
piecewise-constant staircase; a smoothing rather than an interpolating spline
because the tables are digitized, and the residual jitter would otherwise show
up in the derivative. The cost is ~1% on the cross section itself, negligible
against the factor-of-several errors this is here to remove.

Ships with **BGR18** (Bertone, Gauld and Rojo, arXiv:1808.02034), tabulated for
``nu_mu`` on a proton target.

Notes
-----
The BGR18 tables are ``nu-p``. Above ~1 PeV the sea-quark contribution
dominates and ``sigma_(nu p) ~ sigma_(nu n)``, so treating them as per-nucleon
values is safe; below that, valence quarks make ``sigma_(nu n)`` appreciably
larger, and the per-nucleon value for an isoscalar target (the Earth) or a
proton-rich one (water ice, ten protons to eight neutrons) is correspondingly
higher. Until a ``nu-n`` table is supplied, sub-PeV cross sections here are
underestimates by up to a few tens of percent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.interpolate import make_smoothing_spline

from .source import DEFAULT_LAMBDA, E0_CROSS_GEV, SIGMA0_CM2

# Neutral-to-charged current ratio of the analytic model. The tabulated BGR18
# ratio runs 1.41-1.47 over eight decades, so this constant is good to a few
# percent; see softpaws.transport.attenuation.TOTAL_TO_CC_RATIO.
DEFAULT_TOTAL_TO_CC = 1.4

# Reference model shipped with the package.
DEFAULT_TABLE_MODEL = "BGR18"


class CrossSection(ABC):
    """Interface shared by the analytic and tabulated cross-section models.

    Implementations expose the charged-current cross section that sources
    muons, the total that attenuates the flux in the Earth, and the local
    log-log slope ``lambda`` that sets the soft volume's spectral index
    ``A = gamma - lambda - 1``.
    """

    @abstractmethod
    def cc(self, energy_gev: float | np.ndarray) -> np.ndarray:
        """Charged-current cross section [cm^2]."""

    @abstractmethod
    def nc(self, energy_gev: float | np.ndarray) -> np.ndarray:
        """Neutral-current cross section [cm^2]."""

    @abstractmethod
    def local_slope(self, energy_gev: float | np.ndarray) -> np.ndarray:
        """Local slope ``lambda(E) = d ln sigma_CC / d ln E``."""

    def total(self, energy_gev: float | np.ndarray) -> np.ndarray:
        """Total (CC + NC) cross section [cm^2], which sets Earth attenuation.

        Parameters
        ----------
        energy_gev : float or np.ndarray
            Neutrino energy [GeV].

        Returns
        -------
        sigma : np.ndarray
            Total cross section per target [cm^2].
        """
        return self.cc(energy_gev) + self.nc(energy_gev)


class PowerLawCrossSection(CrossSection):
    """The paper's single power law ``sigma_CC = sigma0 (E / E0)^lambda``.

    Parameters
    ----------
    lam : float, optional
        Cross-section slope. Defaults to
        :data:`~softpaws.transport.source.DEFAULT_LAMBDA`.
    sigma0_cm2 : float, optional
        Cross section at the anchor energy [cm^2]. Defaults to
        :data:`~softpaws.transport.source.SIGMA0_CM2`.
    e0_gev : float, optional
        Anchor energy [GeV]. Defaults to
        :data:`~softpaws.transport.source.E0_CROSS_GEV`.
    total_to_cc : float, optional
        Ratio of the total to the charged-current cross section. Defaults to
        :data:`DEFAULT_TOTAL_TO_CC`.

    Examples
    --------
    >>> model = PowerLawCrossSection()
    >>> float(model.local_slope(1.0e6)[0])
    0.4
    """

    def __init__(
        self,
        lam: float = DEFAULT_LAMBDA,
        sigma0_cm2: float = SIGMA0_CM2,
        e0_gev: float = E0_CROSS_GEV,
        total_to_cc: float = DEFAULT_TOTAL_TO_CC,
    ) -> None:
        self.lam = lam
        self.sigma0_cm2 = sigma0_cm2
        self.e0_gev = e0_gev
        self.total_to_cc = total_to_cc

    def cc(self, energy_gev: float | np.ndarray) -> np.ndarray:
        """Charged-current cross section [cm^2]."""
        energy = np.atleast_1d(np.asarray(energy_gev, dtype=float))
        return self.sigma0_cm2 * (energy / self.e0_gev) ** self.lam

    def nc(self, energy_gev: float | np.ndarray) -> np.ndarray:
        """Neutral-current cross section [cm^2], a fixed fraction of the CC one."""
        return (self.total_to_cc - 1.0) * self.cc(energy_gev)

    def local_slope(self, energy_gev: float | np.ndarray) -> np.ndarray:
        """Local slope, which for a power law is the constant ``lambda``."""
        energy = np.atleast_1d(np.asarray(energy_gev, dtype=float))
        return np.full_like(energy, self.lam)


class TabulatedCrossSection(CrossSection):
    """Cross section interpolated from tabulated CC and NC calculations.

    Each channel is represented by a smoothing spline of ``ln sigma`` against
    ``ln E``, so the cross section and its local slope come from the same
    object. Outside the tabulated range the spline is continued as a power law
    with the boundary slope, which is continuous in both value and slope and
    avoids the runaway of a cubic extrapolation.

    Parameters
    ----------
    energy_cc_gev, sigma_cc_cm2 : np.ndarray
        Charged-current table: energies [GeV] and cross sections [cm^2].
    energy_nc_gev, sigma_nc_cm2 : np.ndarray
        Neutral-current table, on its own energy grid.
    name : str, optional
        Label for the model, used in ``repr``.

    Attributes
    ----------
    name : str
        Model label.
    energy_range_gev : tuple of float
        Tabulated range of the charged-current table [GeV], outside which
        values are power-law continuations.

    Examples
    --------
    >>> model = bgr18_cross_section()
    >>> bool(model.cc(1.0e6)[0] > model.cc(1.0e5)[0])
    True
    """

    def __init__(
        self,
        energy_cc_gev: np.ndarray,
        sigma_cc_cm2: np.ndarray,
        energy_nc_gev: np.ndarray,
        sigma_nc_cm2: np.ndarray,
        name: str = "tabulated",
    ) -> None:
        self.name = name
        self._cc = _LogLogSpline(energy_cc_gev, sigma_cc_cm2)
        self._nc = _LogLogSpline(energy_nc_gev, sigma_nc_cm2)
        self.energy_range_gev = (float(energy_cc_gev[0]), float(energy_cc_gev[-1]))

    def __repr__(self) -> str:
        lo, hi = self.energy_range_gev
        return f"{type(self).__name__}(name={self.name!r}, range=({lo:.3g}, {hi:.3g}) GeV)"

    def cc(self, energy_gev: float | np.ndarray) -> np.ndarray:
        """Charged-current cross section [cm^2]."""
        return self._cc.value(energy_gev)

    def nc(self, energy_gev: float | np.ndarray) -> np.ndarray:
        """Neutral-current cross section [cm^2]."""
        return self._nc.value(energy_gev)

    def local_slope(self, energy_gev: float | np.ndarray) -> np.ndarray:
        """Local slope ``lambda(E) = d ln sigma_CC / d ln E``.

        This is the ``lambda`` of the soft-volume model, so it is the *charged
        current* slope: it is CC interactions that source the muons whose
        spectrum the soft volume weights.
        """
        return self._cc.slope(energy_gev)


class _LogLogSpline:
    """Smoothing spline of ``ln y`` against ``ln x`` with power-law continuation."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        log_x = np.log(np.asarray(x, dtype=float))
        log_y = np.log(np.asarray(y, dtype=float))
        # Generalized cross-validation picks the smoothing strength; on the
        # shipped tables it reproduces sigma to ~1% while removing the
        # digitization wobble from the derivative.
        self._spline = make_smoothing_spline(log_x, log_y)
        self._derivative = self._spline.derivative()
        self._log_x_min = float(log_x[0])
        self._log_x_max = float(log_x[-1])

    def _clipped(self, x: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        log_x = np.log(np.atleast_1d(np.asarray(x, dtype=float)))
        return log_x, np.clip(log_x, self._log_x_min, self._log_x_max)

    def value(self, x: float | np.ndarray) -> np.ndarray:
        log_x, inside = self._clipped(x)
        # Inside the table the offset term vanishes; outside it continues the
        # curve as a power law with the boundary slope.
        log_y = self._spline(inside) + self._derivative(inside) * (log_x - inside)
        return np.exp(log_y)

    def slope(self, x: float | np.ndarray) -> np.ndarray:
        _, inside = self._clipped(x)
        return self._derivative(inside)


def bgr18_cross_section(model: str = DEFAULT_TABLE_MODEL) -> TabulatedCrossSection:
    """Load the tabulated cross section shipped with the package.

    Parameters
    ----------
    model : str, optional
        Table directory under ``src/softpaws/data/xsec/``. Defaults to
        :data:`DEFAULT_TABLE_MODEL`.

    Returns
    -------
    cross_section : TabulatedCrossSection
        Model built from the charged- and neutral-current tables.
    """
    # Imported here rather than at module scope: softpaws.data.loader returns
    # softpaws.response.irfs objects, so importing it from transport at load
    # time would close an import cycle. Nothing here runs at import.
    from ..data.loader import load_cross_section_table

    energy_cc, sigma_cc = load_cross_section_table(model, "cc")
    energy_nc, sigma_nc = load_cross_section_table(model, "nc")
    return TabulatedCrossSection(energy_cc, sigma_cc, energy_nc, sigma_nc, name=model)
