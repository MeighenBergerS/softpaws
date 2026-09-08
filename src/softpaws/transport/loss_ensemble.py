"""The loss model's own error: an ensemble of published parametrizations.

The kernel's inputs are PROPOSAL's parametrizations (Kelner-Kokoulin-Petrukhin
bremsstrahlung and pair production, ALLM97 photonuclear with
Butkevich-Mikheyev shadowing). Benchmarking against PROPOSAL validates the
transport algebra and says nothing about the cross sections themselves. This
module measures that remaining error the way the field does: an ensemble of
published parametrizations, one channel swapped at a time, every variant
pushed through the same quadrature the shipped table uses.

Because the exponent ``Phi`` is linear in the loss spectrum, each swap
propagates exactly as a ratio of moments at each energy: ``kappa_1`` on the
first log-loss moment is the energy-reconstruction error, ``kappa_2`` on the
second is the error on everything fluctuation sensitive. Installing those
ratios through :func:`softpaws.transport.coefficients.set_kernel_scaling`
pushes one variant through every calculation without rebuilding the table.

Building the ensemble needs PROPOSAL (``pip install softpaws[transport]``);
loading a cached one needs only NumPy.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import numpy as np

from ..utils.constants import CM_PER_KM, RHO_WATER_G_CM3
from .coefficients import KernelScaling, loss_spectrum_y_grid, proposal_parametrizations

__all__ = [
    "E_GRID",
    "N_MOMENTS",
    "VARIANTS",
    "build_ensemble",
    "channel_moments",
    "implied_scales",
    "load_ensemble",
    "variant_parametrization",
    "variant_ratios",
    "variant_scaling",
]

logger = logging.getLogger(__name__)

#: Energy grid of the budget [GeV], the shipped table's range.
E_GRID = np.logspace(2.0, 10.0, 33)

#: Moment orders kept: the first log-loss moment (the drift, the
#: energy-reconstruction scale) and the second (the fluctuation scale).
N_MOMENTS = 2

#: Variant label -> (channel it swaps, name of the parametrization).
#: Constructors live in :func:`variant_parametrization` to keep the PROPOSAL
#: import local.
VARIANTS = {
    "brems ABB": ("bremsstrahlung", "Andreev-Bezrukov-Bugaev"),
    "brems NLO": ("bremsstrahlung", "Sandrock-Soedingrekso-Rhode"),
    "pair NLO": ("pair production", "Sandrock-Soedingrekso-Rhode"),
    "photo BB": ("photonuclear", "Bezrukov-Bugaev + hard"),
    "photo BDH": ("photonuclear", "Block-Durand-Ha"),
    "photo ALLM91": ("photonuclear", "ALLM91"),
    "photo DRSS": ("photonuclear", "ALLM97, DRSS shadowing"),
}


def variant_parametrization(label: str) -> Any:
    """PROPOSAL parametrization object for one variant's swapped channel.

    Parameters
    ----------
    label : str
        Key of :data:`VARIANTS`.

    Returns
    -------
    parametrization : object
        The PROPOSAL parametrization.

    Raises
    ------
    ImportError
        Raised if PROPOSAL is not installed.
    KeyError
        Raised if ``label`` is not a variant.
    """
    try:
        import proposal as pp
    except ImportError as error:
        raise ImportError(
            "The loss ensemble needs PROPOSAL: pip install 'softpaws[transport]'."
        ) from error

    shadow_bm = pp.parametrization.photonuclear.ShadowButkevichMikheyev
    builders = {
        "brems ABB": lambda: pp.parametrization.bremsstrahlung.AndreevBezrukovBugaev(False),
        "brems NLO": lambda: pp.parametrization.bremsstrahlung.SandrockSoedingreksoRhode(False),
        "pair NLO": lambda: pp.parametrization.pairproduction.SandrockSoedingreksoRhode(False),
        "photo BB": lambda: pp.parametrization.photonuclear.BezrukovBugaev(True),
        "photo BDH": lambda: pp.parametrization.photonuclear.BlockDurandHa(shadow_bm()),
        "photo ALLM91": lambda: pp.parametrization.photonuclear.AbramowiczLevinLevyMaor91(
            shadow_bm()
        ),
        "photo DRSS": lambda: pp.parametrization.photonuclear.AbramowiczLevinLevyMaor97(
            pp.parametrization.photonuclear.ShadowDuttaRenoSarcevicSeckel()
        ),
    }
    return builders[label]()


def channel_moments(
    param: Any, energy_gev: float, y: np.ndarray, n_moments: int = N_MOMENTS
) -> np.ndarray:
    """Log-loss moments ``<(-ln(1-y))^n>`` of one channel [km^-1].

    The same quadrature and water convention as
    :func:`softpaws.transport.coefficients.build_proposal_table`.

    Parameters
    ----------
    param : object
        A PROPOSAL parametrization.
    energy_gev : float
        Muon energy [GeV].
    y : np.ndarray
        Fractional-loss grid, from
        :func:`softpaws.transport.coefficients.loss_spectrum_y_grid`.
    n_moments : int, optional
        Number of moments returned, from the first.

    Returns
    -------
    moments : np.ndarray, shape (n_moments,)
        The moments [km^-1] at water density.
    """
    import proposal as pp

    particle = pp.particle.MuMinusDef()
    medium = pp.medium.Water()
    energy_mev = energy_gev * 1.0e3
    molar_mass = sum(c.atoms_in_molecule * c.atomic_number for c in medium.components)
    scale = CM_PER_KM * RHO_WATER_G_CM3 / medium.mass_density
    rate = np.zeros_like(y)
    for component in medium.components:
        limits = param.kinematic_limits(particle, component, energy_mev)
        inside = (y > limits.v_min) & (y < limits.v_max)
        per_gram = np.zeros_like(y)
        per_gram[inside] = [
            param.differential_crosssection(particle, component, energy_mev, value)
            for value in y[inside]
        ]
        weight = component.atoms_in_molecule * component.atomic_number / molar_mass
        rate += medium.mass_density * weight * per_gram
    rate *= scale
    log_loss = -np.log1p(-y)
    return np.array([np.trapezoid(log_loss**n * rate, y) for n in range(1, n_moments + 1)])


def build_ensemble(
    energy_grid: np.ndarray = E_GRID,
    variants: dict[str, tuple[str, str]] = VARIANTS,
    n_moments: int = N_MOMENTS,
) -> dict[str, np.ndarray]:
    """Baseline channel moments and every variant's swapped channel.

    Parameters
    ----------
    energy_grid : np.ndarray, optional
        Muon energies the moments are tabulated at [GeV].
    variants : dict, optional
        Variants to build; see :data:`VARIANTS`.
    n_moments : int, optional
        Number of moments per energy.

    Returns
    -------
    ensemble : dict of np.ndarray
        ``"base <channel>"`` for each baseline channel and
        ``"swap <label>"`` for each variant, each of shape
        ``(n_energy, n_moments)`` [km^-1].
    """
    y = loss_spectrum_y_grid()
    base_params = proposal_parametrizations()
    baseline = {ch: np.zeros((energy_grid.size, n_moments)) for ch in base_params}
    logger.info("Baseline channels (KKP + KKP + ALLM97/BM)")
    for i, e in enumerate(energy_grid):
        for ch, param in base_params.items():
            baseline[ch][i] = channel_moments(param, float(e), y, n_moments)
    swapped = {}
    for label, (_, name) in variants.items():
        logger.info("Variant %s (%s)", label, name)
        param = variant_parametrization(label)
        swapped[label] = np.array(
            [channel_moments(param, float(e), y, n_moments) for e in energy_grid]
        )
    out = {f"base {ch}": v for ch, v in baseline.items()}
    out.update({f"swap {label}": v for label, v in swapped.items()})
    return out


def variant_ratios(
    ensemble: dict[str, np.ndarray],
    variants: dict[str, tuple[str, str]] = VARIANTS,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Split an ensemble into baseline channels and total-moment ratios.

    Parameters
    ----------
    ensemble : dict of np.ndarray
        As returned by :func:`build_ensemble` or read from its cache.
    variants : dict, optional
        Variants present; see :data:`VARIANTS`.

    Returns
    -------
    baseline : dict of np.ndarray
        Channel -> ``(n_energy, n_moments)`` baseline moments.
    ratios : dict of np.ndarray
        Variant label -> ``(n_energy, n_moments)`` ratio of the variant's
        total moments to the baseline's.
    """
    baseline = {k[5:]: v for k, v in ensemble.items() if k.startswith("base ")}
    total = sum(baseline.values())
    ratios = {}
    for label, (channel, _) in variants.items():
        variant_total = total - baseline[channel] + ensemble[f"swap {label}"]
        ratios[label] = variant_total / total
    return baseline, ratios


def load_ensemble(
    path: str | pathlib.Path, rebuild: bool = False
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """The ensemble from a cache file, building and saving it if needed.

    Parameters
    ----------
    path : str or pathlib.Path
        Cache file (``.npz``).
    rebuild : bool, optional
        Rebuild with PROPOSAL even if the cache exists.

    Returns
    -------
    baseline : dict of np.ndarray
        Channel -> ``(n_energy, n_moments)`` baseline moments.
    ratios : dict of np.ndarray
        Variant label -> ``(n_energy, n_moments)`` total-moment ratio.
    """
    path = pathlib.Path(path)
    if path.exists() and not rebuild:
        logger.info("Cached ensemble from %s", path)
        data = dict(np.load(path))
    else:
        data = build_ensemble()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **data)
    return variant_ratios(data)


def variant_scaling(
    ratios: dict[str, np.ndarray],
    label: str | None,
    energy_grid: np.ndarray = E_GRID,
    prefactor: float = 1.0,
) -> KernelScaling | None:
    """The :class:`~softpaws.transport.coefficients.KernelScaling` of one variant.

    The ratios are interpolated linearly in ``log10 E`` and held at the ends
    of the grid.

    Parameters
    ----------
    ratios : dict of np.ndarray
        Variant ratios from :func:`load_ensemble`.
    label : str or None
        Variant to activate. ``None`` is the baseline, which returns
        ``None`` unless ``prefactor`` differs from one.
    energy_grid : np.ndarray, optional
        Energies the ratios are tabulated at [GeV].
    prefactor : float, optional
        Constant factor multiplied into both ratios, for a fitted transport
        scale on top of the variant.

    Returns
    -------
    scaling : KernelScaling or None
        Factors to install with
        :func:`softpaws.transport.coefficients.set_kernel_scaling`.
    """
    log_grid = np.log10(energy_grid)
    if label is None:
        if prefactor == 1.0:
            return None
        return KernelScaling(lambda e: prefactor, lambda e: prefactor)
    values = ratios[label]

    def make(column):
        def kappa(energy_gev):
            log_e = np.log10(np.asarray(energy_gev, dtype=float))
            return prefactor * np.interp(log_e, log_grid, values[:, column])

        return kappa

    return KernelScaling(make(0), make(1))


def implied_scales(
    ratios: dict[str, np.ndarray],
    energy_grid: np.ndarray = E_GRID,
    window_log10_gev: tuple[float, float] = (5.0, 7.0),
) -> dict[str, float]:
    """Effective transport scale each variant implies over an energy window.

    The geometric mean of ``kappa_1`` across the window, which is what a
    constant ``b_scale`` fitted over that window would return.

    Parameters
    ----------
    ratios : dict of np.ndarray
        Variant ratios from :func:`load_ensemble`.
    energy_grid : np.ndarray, optional
        Energies the ratios are tabulated at [GeV].
    window_log10_gev : tuple of float, optional
        Window [log10 GeV].

    Returns
    -------
    scales : dict of float
        Variant label -> implied scale.
    """
    log_e = np.log10(energy_grid)
    keep = (log_e >= window_log10_gev[0]) & (log_e <= window_log10_gev[1])
    return {
        label: float(np.exp(np.mean(np.log(values[keep, 0])))) for label, values in ratios.items()
    }
