"""The range of a muon to a threshold, and the target volume it sets.

A muon born with energy ``E`` travels until its energy falls below the
threshold the event selection can still see. The mean-loss (CSDA) range
integrates ``dE/dx = a + b E``; the first-passage range treats the same
losses as the stochastic process they are and asks where the energy first
crosses the threshold, which the transport exponent ``Phi(s)`` answers in
closed form. Both are implemented here, with the two-medium correction for a
muon that spends part of its range in the rock beneath the optical medium,
the truncation to a finite column, and the variance of the first-passage
length.

The range is the length a published effective area is built on: the
neutrino energy is fixed and every muon energy that survives the selection is
integrated over, so the length is the full range to threshold and grows
logarithmically with energy. The spectrally weighted soft volume of
:mod:`softpaws.transport.soft_volume` is a different quantity, whose length
``1 / Phi(A)`` is set by how much rarer the higher-energy parent neutrino is;
that module builds the range target volume from the functions here.
"""

from __future__ import annotations

import functools

import numpy as np
from scipy.special import gammainc, polygamma

from ..utils.constants import RHO_WATER_G_CM3
from .coefficients import (
    DEFAULT_SOURCE,
    critical_energy_gev,
    diffusion_coefficient,
    drift_coefficient,
    ionization_coefficient,
    kernel_scaling_token,
    log_loss_moments,
)
from .eigenvalue import two_moment_loss_spectrum

__all__ = [
    "DEFAULT_IONIZATION_MATCH_GEV",
    "DEFAULT_MUON_THRESHOLD_GEV",
    "muon_range_km",
    "stochastic_muon_range_km",
    "stochastic_muon_range_variance_km2",
    "truncated_muon_range_km",
    "two_medium_muon_range_km",
    "two_medium_range_ratio",
]

# Muon energy below which a track no longer passes an IceCube-like through-going
# selection. Used as the lower limit of the muon range in
# :func:`range_target_volume_km3`; the resulting volume depends on it only
# logarithmically.
DEFAULT_MUON_THRESHOLD_GEV = 1.0e3

# Matching energy between the two regimes of :func:`stochastic_muon_range_km`:
# radiative and stochastic above, deterministic and ionizing below. It has to sit
# well above the critical energy ``E_c ~ 600`` GeV, where the scale-invariant
# kernel that the first-passage derivation assumes stops describing the losses,
# and low enough that the radiative treatment still covers most of the range.
# 10 TeV is 17 E_c and leaves one decade to the default threshold.
DEFAULT_IONIZATION_MATCH_GEV = 1.0e4


def muon_range_km(
    energy_gev: float | np.ndarray,
    threshold_gev: float = DEFAULT_MUON_THRESHOLD_GEV,
    density_g_cm3: float = RHO_WATER_G_CM3,
    b_scale: float = 1.0,
    source: str = DEFAULT_SOURCE,
    kernel_evaluation: str = "running",
) -> np.ndarray:
    """Muon range from a starting energy down to a detection threshold.

    Integrating the continuous-slowing-down loss law ``-dE/dx = a_mu + b_mu E``
    from ``E`` down to ``E_thr`` gives

    .. math:: R(E \\to E_\\mathrm{thr}) = \\frac{1}{b_\\mu}
        \\ln\\frac{E + E_c}{E_\\mathrm{thr} + E_c},
        \\qquad E_c = a_\\mu / b_\\mu.

    Unlike the soft volume's spectral length ``1/(b_mu A)``, this carries no
    spectral weighting: it is the distance a muon of energy ``E`` can travel and
    still arrive above threshold, and it grows logarithmically with ``E``.

    Parameters
    ----------
    energy_gev : float or np.ndarray
        Muon energy at production [GeV].
    threshold_gev : float, optional
        Muon energy below which the track is not selected [GeV]. Defaults to
        :data:`DEFAULT_MUON_THRESHOLD_GEV`.
    density_g_cm3 : float, optional
        Target-medium density [g cm^-3]. Defaults to water.
    source : {"proposal", "table1"}, optional
        Transport-coefficient tabulation; see
        :mod:`softpaws.transport.coefficients`.
    b_scale : float, optional
        Multiplicative rescaling of the drift coefficient. Defaults to 1.

    Returns
    -------
    range_km : np.ndarray
        Muon range [km], clipped at zero for muons born below threshold.

    Notes
    -----
    ``b_mu`` is evaluated at the starting energy ``E`` rather than integrated
    along the track. Because ``b_mu`` moves by only 14% between 1 PeV and 100 PeV
    (Table 1), this is a percent-level approximation over the range of interest.
    """
    if kernel_evaluation not in ("frozen", "running"):
        raise ValueError(
            f"kernel_evaluation must be 'frozen' or 'running', got {kernel_evaluation!r}."
        )
    energy = np.atleast_1d(np.asarray(energy_gev, dtype=float))
    if kernel_evaluation == "frozen":
        b_mu = b_scale * drift_coefficient(energy, density_g_cm3, source)
        e_crit = critical_energy_gev(energy, density_g_cm3, b_scale, source)
        ratio = (energy + e_crit) / (threshold_gev + e_crit)
        return np.clip(np.log(ratio) / b_mu, 0.0, None)

    # Running: -dE/dx = a_mu + b_mu(E) E has no closed-form integral once b_mu
    # itself runs, so integrate dL = dE / (a_mu + b_mu(E) E) along a shared
    # logarithmic lattice. Frozen b_mu recovers the closed form above exactly.
    return _depth_between_km(
        "csda",
        energy,
        threshold_gev,
        density_g_cm3,
        b_scale,
        source,
        "table",
        _CSDA_NODES_PER_DECADE,
    )


def _log_loss_moments_at(
    energy_gev: np.ndarray,
    density_g_cm3: float,
    b_scale: float,
    source: str,
    log_loss_source: str,
) -> tuple[np.ndarray, np.ndarray]:
    """``Phi'(0)`` and ``-Phi''(0)`` at each energy, by whichever route is asked for.

    Factored out of :func:`stochastic_muon_range_km` so that the frozen and
    running evaluations read the kernel the same way and differ only in *where*
    they read it.
    """
    if log_loss_source == "table" and source != "table1":
        first, second, _ = log_loss_moments(energy_gev, density_g_cm3, source)
        return b_scale * first, b_scale * second
    # Table 1 tabulates no log-loss columns, so there the family is the only
    # route; it is 8% low on the first moment and 56% low on the second.
    b_mu = b_scale * drift_coefficient(energy_gev, density_g_cm3, source)
    d_mu = diffusion_coefficient(energy_gev, density_g_cm3, source)
    kappa, p = two_moment_loss_spectrum(b_mu, d_mu)
    return kappa * polygamma(1, p + 1.0), -kappa * polygamma(2, p + 1.0)


#: Energy span of the cached depth curves [log10 GeV]. The lower edge sits
#: below any threshold ever asked for and the upper one above the ``10^12``
#: bracket of :func:`_near_entry_energy_gev`, so every descent is a difference
#: of two points inside the span. Widening it does not move a single value,
#: since the node spacing is fixed at ``1 / nodes_per_decade`` and not by the
#: endpoints.
_CURVE_LOG10_LO = 0.0
_CURVE_LOG10_HI = 15.0

#: Lattice density of the deterministic curve. The radiative one takes its own
#: from ``running_nodes_per_decade``, whose integrand ``1 / Phi'(0; E)`` is
#: nearly flat in ``lnE``. This one is not: ``E / (a_mu + b_mu E)`` rises
#: exponentially in ``lnE`` below the critical energy and flattens above it, so
#: it wants the finer lattice. At this density the deterministic range is
#: converged to 5e-7, which the earlier grid -- refined to the span of each
#: descent and so finest close to threshold -- reached only to 3e-5.
_CSDA_NODES_PER_DECADE = 3072


@functools.lru_cache(maxsize=32)
def _base_depth_curve(
    kind: str,
    source: str,
    log_loss_source: str,
    nodes_per_decade: int,
    token: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative depth against ``log10 E``, in water and at ``b_scale = 1``.

    Every descent this module integrates -- the deterministic
    ``dL = dE / (a_mu + b_mu(E) E)``, the radiative ``dL = dlnE / Phi'(0; E)``
    and the variance rate that goes with it -- has an integrand that depends on
    the energy alone once the medium and the kernel are fixed. The depth
    between any two energies is therefore a difference of one cumulative curve,
    which is built once here and read by interpolation afterwards.

    Building it on a fixed lattice rather than on a grid spanning the descent
    asked for also makes the answer independent of *which* energies are
    requested together: the older per-call grid ran from the floor to the
    largest energy in the batch, so a muon evaluated alone and the same muon
    evaluated inside an array got trapezoid rules of slightly different step.

    Parameters
    ----------
    kind : {"csda", "radiative", "variance"}
        Which integrand to accumulate: ``dE / (a_mu + b_mu E)``,
        ``dlnE / Phi'(0)`` or ``dlnE (-Phi''(0)) / Phi'(0)^3``.
    source : str
        Transport-coefficient tabulation.
    log_loss_source : {"table", "family"}
        Where the log-loss moments come from; ``"csda"`` ignores it.
    nodes_per_decade : int
        Lattice density.
    token : int
        :func:`~softpaws.transport.coefficients.kernel_scaling_token`, which
        changes when the installed kernel scaling does and so drops the entry.

    Returns
    -------
    log10_grid, cumulative : np.ndarray
        The lattice and the depth accumulated from its lower edge [km].
    """
    del token  # a cache key only; the scaling is read through the coefficients
    n_nodes = int(round((_CURVE_LOG10_HI - _CURVE_LOG10_LO) * nodes_per_decade)) + 1
    log10_grid = np.linspace(_CURVE_LOG10_LO, _CURVE_LOG10_HI, n_nodes)
    grid = 10.0**log10_grid
    if kind == "csda":
        b_grid = drift_coefficient(grid, RHO_WATER_G_CM3, source)
        a_mu = ionization_coefficient(RHO_WATER_G_CM3)
        integrand = grid / (a_mu + b_grid * grid)
    elif kind == "variance":
        first, second, _ = log_loss_moments(grid, RHO_WATER_G_CM3, source)
        integrand = second / first**3
    else:
        first, _ = _log_loss_moments_at(
            grid, RHO_WATER_G_CM3, 1.0, source, log_loss_source
        )
        integrand = 1.0 / first
    ln_grid = log10_grid * np.log(10.0)
    cumulative = np.concatenate(
        [[0.0], np.cumsum(np.diff(ln_grid) * 0.5 * (integrand[1:] + integrand[:-1]))]
    )
    return log10_grid, cumulative


@functools.lru_cache(maxsize=64)
def _scaled_depth_curve(
    kind: str,
    source: str,
    log_loss_source: str,
    nodes_per_decade: int,
    density_g_cm3: float,
    b_scale: float,
    token: int,
) -> tuple[np.ndarray, np.ndarray]:
    """As :func:`_base_depth_curve`, for a medium of the given density.

    Every integrand carries ``b_scale`` and the density as one overall factor:
    ``a_mu`` and ``b_mu`` are each proportional to both, so the deterministic
    integrand ``E / (a_mu + b_mu E)`` scales as ``1 / (b_scale rho)`` exactly,
    and so does ``1 / Phi'(0)`` whenever the log-loss moments are read from the
    table. The variance rate ``-Phi''(0) / Phi'(0)^3`` carries the same factor
    squared, since both moments scale together. The two-moment family is the
    exception -- it reconstructs the moments from ``b_mu`` and ``d_mu``, and
    only the first of those carries ``b_scale`` -- so that route rebuilds the
    curve instead.
    """
    factored = kind in ("csda", "variance") or (
        log_loss_source == "table" and source != "table1"
    )
    if factored:
        log10_grid, cumulative = _base_depth_curve(
            kind, source, log_loss_source, nodes_per_decade, token
        )
        power = 2 if kind == "variance" else 1
        scale = ((RHO_WATER_G_CM3 / density_g_cm3) / b_scale) ** power
        return log10_grid, cumulative * scale

    n_nodes = int(round((_CURVE_LOG10_HI - _CURVE_LOG10_LO) * nodes_per_decade)) + 1
    log10_grid = np.linspace(_CURVE_LOG10_LO, _CURVE_LOG10_HI, n_nodes)
    first, _ = _log_loss_moments_at(
        10.0**log10_grid, density_g_cm3, b_scale, source, log_loss_source
    )
    ln_grid = log10_grid * np.log(10.0)
    integrand = 1.0 / first
    cumulative = np.concatenate(
        [[0.0], np.cumsum(np.diff(ln_grid) * 0.5 * (integrand[1:] + integrand[:-1]))]
    )
    return log10_grid, cumulative


def _depth_between_km(
    kind: str,
    energy_gev: np.ndarray,
    floor_gev: float | np.ndarray,
    density_g_cm3: float,
    b_scale: float,
    source: str,
    log_loss_source: str,
    nodes_per_decade: int,
) -> np.ndarray:
    """Depth accumulated descending from ``energy_gev`` to ``floor_gev`` [km].

    A difference of two readings of :func:`_scaled_depth_curve`, clipped at
    zero for a muon born at or below the floor. Both arguments broadcast, so a
    per-direction floor costs no more than a single one.
    """
    log10_grid, cumulative = _scaled_depth_curve(
        kind,
        source,
        log_loss_source,
        nodes_per_decade,
        float(density_g_cm3),
        float(b_scale),
        kernel_scaling_token(),
    )
    top = np.interp(np.log10(energy_gev), log10_grid, cumulative)
    bottom = np.interp(np.log10(floor_gev), log10_grid, cumulative)
    return np.clip(top - bottom, 0.0, None)


def _running_radiative_length_km(
    energy_gev: np.ndarray,
    floor_gev: float,
    density_g_cm3: float,
    b_scale: float,
    source: str,
    log_loss_source: str,
    nodes_per_decade: int,
) -> np.ndarray:
    """The radiative segment with the kernel followed down the trajectory.

    The rate at which log energy is shed is a local quantity, so over a descent
    spanning decades the depth accumulates as ``dL / dlnE = 1 / Phi'(0; E)``
    and the frozen form ``ln(eps/E_floor) / Phi'(0; eps)`` is the value of that
    integrand at the *top* of the descent, where the loss rate is highest. It is
    therefore short, one-sidedly and by more the further the muon falls.

    Read off :func:`_base_depth_curve`, so the cost is independent of how many
    production energies are asked for and of how far each of them falls.
    """
    energy = np.atleast_1d(np.asarray(energy_gev, dtype=float))
    return _depth_between_km(
        "radiative",
        energy,
        floor_gev,
        density_g_cm3,
        b_scale,
        source,
        log_loss_source,
        nodes_per_decade,
    )


def stochastic_muon_range_km(
    energy_gev: float | np.ndarray,
    threshold_gev: float = DEFAULT_MUON_THRESHOLD_GEV,
    density_g_cm3: float = RHO_WATER_G_CM3,
    b_scale: float = 1.0,
    source: str = DEFAULT_SOURCE,
    ell_max_scale: float = 2.5,
    n_ell: int = 161,
    method: str = "closed",
    include_ionization: bool = True,
    match_energy_gev: float = DEFAULT_IONIZATION_MATCH_GEV,
    log_loss_source: str = "table",
    kernel_evaluation: str = "running",
    running_nodes_per_decade: int = 48,
) -> np.ndarray:
    """Muon range to a threshold, averaged over the exact stochastic loss law.

    :func:`muon_range_km` answers "how far does the *average* muon get before
    dropping below ``E_thr``" by integrating the continuous-slowing-down law,
    which makes the arrival-above-threshold probability a step function at that
    mean range. The losses are not deterministic, though
    (:mod:`softpaws.transport.loss_distribution`), so the honest length is the
    expected one,

    .. math:: L(\\varepsilon) = \\int_0^\\infty d\\ell\\;
        \\mathbb{P}\\bigl[\\,W(\\ell) < \\ln(\\varepsilon / E_\\mathrm{thr})\\,\\bigr],

    with ``W`` the accumulated log-loss subordinator whose CDF is
    :func:`softpaws.transport.loss_distribution.log_loss_cdf`. Replacing the
    step by the true CDF shortens the length by ~7% at 1 PeV and ~12% at
    100 PeV: the loss law is right-skewed, so more muons fall short of the mean
    range than overshoot it.

    That integral has a closed form (``docs/theory/first_passage_range.md``). ``W`` is
    non-decreasing, so ``{W(ell) < w}`` is exactly ``{tau(w) > ell}`` for the
    first-passage depth ``tau``, and the integral collapses to ``E[tau(w)]`` --
    the expected distance at which the muon first drops below threshold. That
    renewal function has Laplace transform ``1 / (s Phi(s))``, whose small-``s``
    expansion gives

    .. math:: L(\\varepsilon) = \\frac{\\ln(\\varepsilon / E_\\mathrm{thr})}
        {\\Phi'(0)} - \\frac{\\Phi''(0)}{2\\,\\Phi'(0)^2},
        \\qquad \\Phi'(0) = \\langle -\\ln(1-y)\\rangle,\\;
        -\\Phi''(0) = \\langle \\ln^2(1-y)\\rangle,

    i.e. the CSDA formula with ``b_mu = <y>`` replaced by ``<-ln(1-y)>``, plus a
    constant. Since ``-ln(1-y) >= y`` for every positive loss spectrum, the
    stochastic range is always the shorter one.

    **Ionization and the two-regime range.** The expansion above is purely
    radiative, but a threshold of 1 TeV sits within a factor of two of the muon
    critical energy in water (``E_c = a_mu / b_mu ~ 600`` GeV,
    :func:`~softpaws.transport.coefficients.critical_energy_gev`), so the last
    e-fold of the range -- the one that sets where a tabulated effective area
    turns on -- is not radiative at all. Ionization cannot simply be added to
    ``Phi``: it removes a fixed amount of energy per unit length and not a fixed
    *fraction*, so it is additive in ``E`` and not in ``ln E``, which is the
    structure the whole subordinator derivation rests on.

    With ``include_ionization`` (the default) the range is therefore spliced at a
    matching energy ``E_*`` chosen well above ``E_c``,

    .. math:: L(\\varepsilon) = \\underbrace{\\frac{\\ln(\\varepsilon/E_*)}
        {\\Phi'(0)} - \\frac{\\Phi''(0)}{2\\Phi'(0)^2}}_{\\text{stochastic,
        radiative}} \\;+\\; \\underbrace{\\frac{1}{b_\\mu(E_a)}
        \\ln\\frac{E_a + E_c}{E_\\mathrm{thr} + E_c}}_{\\text{deterministic,
        with ionization}},

    which leaves the derivation untouched above ``E_*`` and appends an almost
    constant offset below it, so the result is still closed form. Muons born
    below ``E_*`` get the deterministic range alone (:func:`muon_range_km`).
    Passing ``include_ionization=False`` recovers the purely radiative range,
    which is what Table E.1 of the paper contrasts against ``R_CSDA``.

    The deterministic segment starts at ``E_a = E_* exp(-<overshoot>)`` and not
    at ``E_*``, because a first passage overshoots the level it crosses. By
    Wald's identity ``E[W(tau)] = w + <overshoot>`` with
    ``<overshoot> = -Phi''(0) / 2 Phi'(0)`` in log energy, which is the same
    quantity the renewal constant above measures in depth, so starting the CSDA
    segment at ``E_*`` would count that stretch of track twice. It is worth 0.4
    km, and leaving it in shows up as a 0.4 km discontinuity at ``E_*``.

    The splice shortens the range at every energy, by 17% at ``E_nu = 10^4``
    GeV, where the muon never enters the radiative regime at all, through 6.7%
    at ``10^6`` to 4.3% at ``10^8``. Under ``kernel_evaluation="frozen"`` it
    instead changed sign across the band (-13%, -1.6%, +1.4%), because two
    effects ran against each other in the spliced decade: ionization shortened
    the range while ceasing to apply the production-energy ``Phi'(0)`` to a muon
    that is by then at TeV energies lengthened it, and the two nearly cancelled.
    Running the kernel down the trajectory already carries the second of those,
    so the splice is left doing only the physical job it exists for.

    ``match_energy_gev`` is a convention and the answer moves with it: a factor
    of three either way from the default changes the range by about 3%, which is
    the systematic this treatment carries. Lowering it keeps more of the track
    stochastic, which is right down to ``E_c``; raising it keeps more of the
    ionization, which matters most near threshold. Neither limit is uniformly
    better, and only a deterministic drift inside ``W(ell)`` would remove the
    choice.

    Unlike the soft volume's spectral length ``1/Phi(A)``, this carries no
    spectral weighting -- it is the right length for a **monochromatic** parent,
    which is what a tabulated effective area is differential in (App. I's
    ``s -> 0`` case; see :func:`dm_line_target_volume_km3`).

    Parameters
    ----------
    energy_gev : float or np.ndarray
        Muon energy at production [GeV].
    threshold_gev : float, optional
        Muon energy below which the track is not selected [GeV]. Defaults to
        :data:`DEFAULT_MUON_THRESHOLD_GEV`.
    density_g_cm3 : float, optional
        Target-medium density [g cm^-3]. Defaults to water.
    b_scale : float, optional
        Multiplicative rescaling of the drift coefficient. Defaults to 1.
    source : {"proposal", "table1"}, optional
        Transport-coefficient tabulation; see
        :mod:`softpaws.transport.coefficients`.
    ell_max_scale : float, optional
        Upper limit of the depth integral, in units of the deterministic range
        (:func:`muon_range_km`). The CDF is already below ``1e-3`` by twice the
        deterministic range, so the default truncation is harmless. Used by
        ``method="quadrature"`` only.
    n_ell : int, optional
        Number of depth samples in the integral. Used by
        ``method="quadrature"`` only.
    method : {"closed", "quadrature"}, optional
        ``"closed"`` (the default) evaluates the renewal expansion above, two
        polygamma calls and no inversion. ``"quadrature"`` evaluates the depth
        integral directly against
        :func:`softpaws.transport.loss_distribution.log_loss_cdf`; it is orders
        of magnitude slower and is kept as the independent check that the
        closed form is exercised against.
    include_ionization : bool, optional
        Splice the deterministic ionizing range below ``match_energy_gev`` onto
        the radiative first passage above it, as derived above. Defaults to
        ``True``. ``False`` gives the purely radiative range.
    match_energy_gev : float, optional
        Matching energy ``E_*`` [GeV] between the two regimes, which has to sit
        well above the critical energy for the splice to be meaningful.
        Defaults to :data:`DEFAULT_IONIZATION_MATCH_GEV`. Ignored when
        ``include_ionization`` is ``False``, and also when it falls at or below
        ``threshold_gev``, where there is no ionizing segment left to splice and
        the range degrades to the purely radiative one.
    log_loss_source : {"table", "family"}, optional
        Where ``Phi'(0)`` and ``Phi''(0)`` come from. ``"table"`` (the default)
        reads them from the tabulated spectrum via
        :func:`~softpaws.transport.coefficients.log_loss_moments`; ``"family"``
        reconstructs them from the two-moment calibration of ``b_mu`` and
        ``d_mu``. The family is 8% low on the first moment and 56% low on the
        second, because ``-ln(1-y)`` weights the hard end of the kernel that a
        fit to the ``y``-moments does not constrain, and the resulting range is
        6.9% long over ``10^5`` to ``10^8`` GeV. The bias is almost pure
        normalization -- 0.7% rms of residual tilt across that band -- so it
        moves an effective-area ceiling and leaves its shape alone. Kept as an
        option because ``source="table1"`` has no log-loss columns and because
        ``method="quadrature"`` checks against the family's own kernel.
    kernel_evaluation : {"running", "frozen"}, optional
        Where along the descent the kernel is read. ``"running"`` (the default)
        follows it down, so the radiative segment is
        ``int dlnE / Phi'(0; E)`` and both renewal constants are read at the
        level being crossed. ``"frozen"`` holds the production-energy kernel for
        the whole descent, which is the closed form of Eq.~(C4) as written.

        Freezing is short, one-sidedly, and by more the further the muon falls,
        because it evaluates the loss rate at the top of the descent where that
        rate is highest. Against a PROPOSAL propagation (example 39, stopping at
        100 TeV) the mean range is 4.0% rms and 6.5% worst over ``w = 1.15`` to
        ``5.76``, against 1.4% and 2.5% running. Effective areas reach
        ``w ~ 9``, where the two differ by 11%. The gap is a rising tilt and not
        a normalization: 0% at ``10^4`` GeV of parent energy, 2.1% at ``10^5``,
        4.4% at ``10^6``, 7.3% at ``10^7`` and 11.3% at ``10^8``.

        Kept selectable because the frozen form is what the drift-diffusion
        literature evaluates and what ``method="quadrature"`` can check.
    running_nodes_per_decade : int, optional
        Grid density for the running integral. The integrand ``1/Phi'(0; E)``
        varies by ``E^-beta`` with ``beta ~ 0.028``, so it is nearly linear in
        ``lnE``: the default is converged to 6 mm over four decades, two orders
        below the 4 cm at which the closed form tracks its own depth integral.

    Returns
    -------
    range_km : np.ndarray
        Expected range [km], zero for muons born below threshold.

    Raises
    ------
    ValueError
        Raised if ``method`` or ``log_loss_source`` is not one of its supported
        values, or if the two are combined incompatibly.

    Notes
    -----
    ``b_mu`` and ``d_mu`` are evaluated once, at the production energy, rather
    than followed down the track -- the same percent-level approximation
    :func:`muon_range_km` makes and justifies. Against a PROPOSAL Monte Carlo
    that cost is 4.5% rms on the range, dropping to 1.8% if the moments are
    integrated down the trajectory instead
    (``examples/39_range_moment_estimator.py``).

    With ``log_loss_source="family"`` the two methods agree to better than 4 cm
    over ``10^4`` to ``10^8`` GeV. The
    residual is the depth grid: it is sized to the range to ``E_thr`` while the
    integrand falls off on the shorter range to ``E_*``, so the spliced form is
    sampled more coarsely than the purely radiative one, which agrees to 2 mm.
    The closed form is an expansion in ``1 / ln(eps / E_thr)``, so it should
    not be pushed to ``eps -> E_thr``, where the length vanishes anyway.
    """
    if method not in ("closed", "quadrature"):
        raise ValueError(f"method must be 'closed' or 'quadrature', got {method!r}.")
    if log_loss_source not in ("table", "family"):
        raise ValueError(
            f"log_loss_source must be 'table' or 'family', got {log_loss_source!r}."
        )
    if kernel_evaluation not in ("frozen", "running"):
        raise ValueError(
            f"kernel_evaluation must be 'frozen' or 'running', got {kernel_evaluation!r}."
        )
    if method == "quadrature" and kernel_evaluation != "frozen":
        # The depth integral builds one kernel and holds it for the whole
        # descent, so it is a frozen calculation by construction. Checking the
        # running closed form against it would measure the running, not the
        # renewal expansion the check exists for.
        raise ValueError(
            "method='quadrature' freezes the kernel at the production energy, so "
            "it needs kernel_evaluation='frozen'."
        )
    if method == "quadrature" and log_loss_source != "family":
        # The depth integral runs against log_loss_cdf, which builds the
        # two-moment family's kernel. Checking a table-based closed form against
        # it would compare two different kernels and disagree by ~7% by
        # construction, so the internal check is pinned to the family.
        raise ValueError(
            "method='quadrature' checks the closed form against the two-moment "
            "family's own depth integral, so it needs log_loss_source='family'."
        )

    energy = np.atleast_1d(np.asarray(energy_gev, dtype=float))
    deterministic = muon_range_km(
        energy, threshold_gev, density_g_cm3, b_scale, source, kernel_evaluation
    )
    selectable = (energy > threshold_gev) & (deterministic > 0.0)

    # Phi'(0) = <-ln(1-y)> and -Phi''(0) = <ln^2(1-y)>, both per unit length,
    # here at the production energy.
    first, second = _log_loss_moments_at(
        energy, density_g_cm3, b_scale, source, log_loss_source
    )

    # Above the matching energy the first passage is radiative and stochastic; below it
    # the muon is within reach of E_c and slows deterministically. The stochastic part
    # already carries the muon *past* E_*, since first passage overshoots the level it
    # crosses: by Wald, E[W(tau)] = w + <overshoot> with <overshoot> = -Phi''(0)/2Phi'(0)
    # in log energy, which is the same quantity the renewal constant measures in depth.
    # The deterministic segment therefore starts at the mean arrival energy and not at
    # E_*, and its b_mu is evaluated there, which is where the muon actually is.
    # A threshold at or above E_* leaves no ionizing segment to splice on, since
    # the muon stops counting while it is still radiative. Clamping the floor
    # degrades the two-regime range back to the purely radiative one, which is
    # what a fitted threshold above E_* should get.
    # Both renewal constants -- the mean overshoot in log energy and the
    # constant it contributes to the depth -- belong to the *crossing*, so a
    # running evaluation reads them at the level being crossed. Freezing reads
    # everything at production, which is what makes it a frozen calculation.
    stochastic_floor = float(
        match_energy_gev
        if include_ionization and match_energy_gev > threshold_gev
        else threshold_gev
    )
    if kernel_evaluation == "running":
        crossing_first, crossing_second = _log_loss_moments_at(
            np.array([stochastic_floor]), density_g_cm3, b_scale, source, log_loss_source
        )
    else:
        crossing_first, crossing_second = first, second

    if include_ionization and match_energy_gev > threshold_gev:
        # A first passage crosses its level from above, so the overshoot is
        # non-negative and the arrival energy lies in [E_thr, E_*]. Both bounds
        # bind only where the calibrated kernel is not a valid loss spectrum at
        # all -- ``d_mu / b_mu >= 1`` puts ``p + 1 <= 0`` -- which a sampler
        # exploring an unphysical b_scale does reach, and where an unclamped
        # exponential would return an infinite range instead of a wrong one.
        overshoot = np.clip(crossing_second / (2.0 * crossing_first), 0.0, None)
        arrival_gev = np.clip(
            stochastic_floor * np.exp(-overshoot), threshold_gev, stochastic_floor
        )
        offset_km = muon_range_km(
            arrival_gev, threshold_gev, density_g_cm3, b_scale, source, kernel_evaluation
        )
    else:
        offset_km = np.zeros_like(energy)
    stochastic_regime = selectable & (energy > stochastic_floor)

    if method == "closed":
        with np.errstate(divide="ignore", invalid="ignore"):
            if kernel_evaluation == "running":
                radiative_km = _running_radiative_length_km(
                    energy,
                    stochastic_floor,
                    density_g_cm3,
                    b_scale,
                    source,
                    log_loss_source,
                    running_nodes_per_decade,
                )
            else:
                radiative_km = np.log(energy / stochastic_floor) / first
            length = (
                radiative_km
                + crossing_second / (2.0 * crossing_first**2)
                + offset_km
            )
        # Muons born below E_* never enter the radiative regime, so the deterministic
        # range is the whole of their answer.
        out = np.where(stochastic_regime, length, np.where(selectable, deterministic, 0.0))
        return out.reshape(np.shape(energy_gev)) if np.ndim(energy_gev) else out

    from .loss_distribution import log_loss_cdf

    # Only the depth integral needs the y-moments; the closed form reads the
    # log-loss ones instead, so these are built here and not above.
    b_mu = b_scale * drift_coefficient(energy, density_g_cm3, source)
    d_mu = diffusion_coefficient(energy, density_g_cm3, source)
    out = np.zeros(energy.shape[0])
    for i, eps in enumerate(energy):
        if not selectable[i]:
            continue
        if not stochastic_regime[i]:
            out[i] = float(deterministic[i])
            continue
        ell = np.linspace(0.0, ell_max_scale * float(deterministic[i]), n_ell)
        cdf = log_loss_cdf(np.log(eps / stochastic_floor), ell, float(b_mu[i]), float(d_mu[i]))
        out[i] = float(np.trapezoid(cdf, ell)) + float(offset_km[i])
    return out.reshape(np.shape(energy_gev)) if np.ndim(energy_gev) else out


def two_medium_muon_range_km(
    energy_gev: float | np.ndarray,
    threshold_gev: float = DEFAULT_MUON_THRESHOLD_GEV,
    near_column_km: float = 0.0,
    near_source: str = DEFAULT_SOURCE,
    far_source: str = "proposal_rock",
    density_g_cm3: float = RHO_WATER_G_CM3,
    b_scale: float = 1.0,
    **range_kwargs,
) -> np.ndarray:
    """Range to threshold through a far medium first and a near one last.

    An upgoing muon at IceCube or ARCA is born in the bedrock or the sea floor
    and only enters the ice or the water for the last stretch before the
    array, so the kernel it descends through is rock for most of its range.
    Because the log-loss subordinator only moves down, the descent splits at
    one energy: the muon enters the near medium at the ``E_1`` for which the
    near medium's own range to threshold equals the near column,

    .. math:: L_\\mathrm{near}(E_1 \\to E_\\mathrm{thr}) = X_\\mathrm{near},

    and the total range is the far-medium first passage from the production
    energy down to that level plus the near column,

    .. math:: L(\\varepsilon) = L_\\mathrm{far}(\\varepsilon \\to E_1)
        + X_\\mathrm{near}, \\qquad \\varepsilon > E_1 .

    A muon whose whole near-medium range fits inside ``X_near`` never sees the
    far medium and gets :func:`stochastic_muon_range_km` in the near medium
    alone. ``E_1`` depends on the near column and the threshold and not on the
    production energy, so it is solved once per arrival direction.

    Every column here is water equivalent, as everywhere in the model, so a
    near column of ice at density 0.918 enters as its water-equivalent length.
    The ionization coefficient is taken from the near medium throughout; the
    far-medium value is ~15% lower in standard rock, which is worth a few
    percent of the last kilometre and nothing elsewhere.

    Parameters
    ----------
    energy_gev : float or np.ndarray
        Muon energy at production [GeV].
    threshold_gev : float, optional
        Muon energy below which the track is not selected [GeV]. Defaults to
        :data:`DEFAULT_MUON_THRESHOLD_GEV`.
    near_column_km : float, optional
        Column of the near medium between the far medium and the detector,
        along the arrival direction [km w.e.]. Zero (the default) puts the
        whole range in the far medium.
    near_source : str, optional
        Coefficient source for the near medium; see
        :mod:`softpaws.transport.coefficients`. Defaults to water.
    far_source : str, optional
        Coefficient source for the far medium. Defaults to
        ``"proposal_rock"``, PROPOSAL's standard rock.
    density_g_cm3 : float, optional
        Reference density [g cm^-3]; cancels, as every length is a column.
    b_scale : float, optional
        Multiplicative rescaling of the drift coefficient, applied to both
        media. Defaults to 1.
    **range_kwargs
        Passed to :func:`stochastic_muon_range_km` for both segments
        (``include_ionization``, ``match_energy_gev``, ``kernel_evaluation``,
        ``log_loss_source``, ``running_nodes_per_decade``).

    Returns
    -------
    range_km : np.ndarray
        Expected range [km w.e.], zero for muons born below threshold.
    """
    energy = np.atleast_1d(np.asarray(energy_gev, dtype=float))
    near = float(near_column_km)
    range_items = tuple(sorted(range_kwargs.items()))

    if near <= 0.0:
        out = stochastic_muon_range_km(
            energy, threshold_gev, density_g_cm3, b_scale, far_source, **range_kwargs
        )
        return out.reshape(np.shape(energy_gev)) if np.ndim(energy_gev) else out

    within_near = stochastic_muon_range_km(
        energy, threshold_gev, density_g_cm3, b_scale, near_source, **range_kwargs
    )
    if float(np.max(within_near)) <= near:
        # Even the most energetic muon asked for stops inside the near column.
        return within_near.reshape(np.shape(energy_gev)) if np.ndim(energy_gev) else within_near

    entry_gev = _near_entry_energy_gev(
        near, float(threshold_gev), float(density_g_cm3), float(b_scale), near_source,
        range_items,
    )
    beyond = stochastic_muon_range_km(
        energy, entry_gev, density_g_cm3, b_scale, far_source, **range_kwargs
    )
    out = np.where(energy > entry_gev, near + beyond, within_near)
    return out.reshape(np.shape(energy_gev)) if np.ndim(energy_gev) else out


def two_medium_range_ratio(
    production_gev: float | np.ndarray,
    threshold_gev: float,
    cos_theta: float | np.ndarray,
    near_vertical_km: float,
    density_g_cm3: float = RHO_WATER_G_CM3,
    b_scale: float = 1.0,
    far_source: str | None = "proposal_rock",
    near_source: str = DEFAULT_SOURCE,
    **range_kwargs,
) -> np.ndarray:
    """Two-medium range over the single-medium one, per arrival direction.

    The factor by which the rock below an array shortens the entering term of
    an upgoing effective area. The near column is the optical medium between
    the far medium and the point the muon is seen at,
    ``near_vertical_km / |cos theta|``, in the same units as every other
    length here, so the ratio multiplies a column-depth length directly. It
    is 1 above the horizon, where the overburden is the optical medium
    throughout, and 1 everywhere when ``far_source`` is ``None``.

    Parameters
    ----------
    production_gev : float or np.ndarray
        Muon energy at production [GeV]. An array is answered in one pass: the
        entry energy belongs to the direction and not to the parent, so a whole
        transmission ladder costs one descent per direction.
    threshold_gev : float
        Muon energy below which the track is not selected [GeV].
    cos_theta : float or np.ndarray
        Cosine of the arrival zenith; ``+1`` is overhead, ``-1`` the nadir.
    near_vertical_km : float
        Vertical extent of the optical medium below the point the muon is
        seen at [km], typically the headroom below the instrumented volume
        plus half its height.
    density_g_cm3 : float, optional
        Density of the optical medium [g cm^-3], the unit the lengths are in.
    b_scale : float, optional
        Multiplicative rescaling of the drift coefficient, applied to both
        media. Defaults to 1.
    far_source : str or None, optional
        Coefficient source for the far medium. Defaults to
        ``"proposal_rock"``; ``None`` disables the correction.
    near_source : str, optional
        Coefficient source for the near medium. Defaults to water.
    **range_kwargs
        Passed to :func:`stochastic_muon_range_km` for both media.

    Returns
    -------
    ratio : np.ndarray
        Range ratio in ``(0, 1]``, of shape ``(n_dir,)`` for a scalar
        production energy and ``(n_energy, n_dir)`` for an array of them.
    """
    production = np.atleast_1d(np.asarray(production_gev, dtype=float))
    cos_theta = np.atleast_1d(np.asarray(cos_theta, dtype=float))
    ratio = np.ones((production.size, cos_theta.size))
    upgoing = cos_theta < 0.0
    if far_source is None or not upgoing.any():
        return ratio if np.ndim(production_gev) else ratio[0]
    single = np.asarray(stochastic_muon_range_km(
        production, threshold_gev, density_g_cm3, b_scale, near_source, **range_kwargs
    ))
    # A muon with no single-medium range has no ratio to take, and keeps the 1
    # that leaves the entering term to the optical medium alone.
    usable = np.isfinite(single) & (single > 0.0)
    if usable.any():
        for i in np.flatnonzero(upgoing):
            near_km = near_vertical_km / max(-float(cos_theta[i]), 1.0e-3)
            two = np.asarray(two_medium_muon_range_km(
                production, threshold_gev, near_km, near_source, far_source,
                density_g_cm3, b_scale, **range_kwargs
            ))
            ratio[usable, i] = two[usable] / single[usable]
    return ratio if np.ndim(production_gev) else ratio[0]


@functools.lru_cache(maxsize=8192)
def _near_entry_energy_gev(
    near_column_km: float,
    threshold_gev: float,
    density_g_cm3: float,
    b_scale: float,
    near_source: str,
    range_items: tuple,
) -> float:
    """The energy at which the near medium's range to threshold equals the near column.

    Cached, because the entry energy depends on the arrival direction and the
    threshold and not on the production energy, so a sky-resolved effective
    area asks for the same few hundred values many thousand times.
    """
    from scipy.optimize import brentq

    range_kwargs = dict(range_items)

    def shortfall(log10_e):
        return float(np.squeeze(stochastic_muon_range_km(
            10.0**log10_e, threshold_gev, density_g_cm3, b_scale, near_source, **range_kwargs
        ))) - near_column_km

    # The caller has checked that some energy below its own maximum reaches the
    # column, and the coefficient tables stop at 10^10 GeV, so the bracket is safe.
    return 10.0**brentq(shortfall, np.log10(threshold_gev) + 1.0e-6, 12.0)


def _running_variance_rate_km2(
    energy_gev: np.ndarray,
    floor_gev: float,
    density_g_cm3: float,
    b_scale: float,
    source: str,
    nodes_per_decade: int = 48,
) -> np.ndarray:
    """``int dlnE (-Phi''(0; E)) / Phi'(0; E)^3``, the running form of the term
    linear in ``w`` in :func:`stochastic_muon_range_variance_km2`."""
    energy = np.atleast_1d(np.asarray(energy_gev, dtype=float))
    return _depth_between_km(
        "variance",
        energy,
        floor_gev,
        density_g_cm3,
        b_scale,
        source,
        "table",
        nodes_per_decade,
    )


def stochastic_muon_range_variance_km2(
    energy_gev: float | np.ndarray,
    threshold_gev: float = DEFAULT_MUON_THRESHOLD_GEV,
    density_g_cm3: float = RHO_WATER_G_CM3,
    b_scale: float = 1.0,
    source: str = DEFAULT_SOURCE,
    kernel_evaluation: str = "running",
) -> np.ndarray:
    """Variance of the muon range to a threshold, from the same first passage.

    :func:`stochastic_muon_range_km` returns the *mean* depth at which a muon
    first falls below ``E_thr``. Individual muons scatter about it by tens of
    percent, and that spread has a closed form built from the same kernel. With
    ``w = ln(varepsilon / E_thr)``,

    .. math:: \\mathrm{Var}(R) = \\frac{-\\Phi''(0)\\,w}{\\Phi'(0)^3}
        - \\frac{\\Phi'''(0)}{3\\,\\Phi'(0)^3}
        + \\frac{\\Phi''(0)^2}{4\\,\\Phi'(0)^4}.

    The structure mirrors the mean, and both constants are moments of the same
    stationary overshoot. A first passage crosses its level from above; the
    *mean* overshoot ``-Phi''(0) / 2 Phi'(0)`` is the constant in the mean, and
    its *variance* ``Phi'''(0) / 3 Phi'(0) - Phi''(0)^2 / 4 Phi'(0)^2``, divided
    by ``Phi'(0)^2`` to turn log-energy into depth, is the constant here. It
    enters negatively: a muon that overshoots further crossed its level sooner.

    Both terms matter. Against a direct simulation of PROPOSAL's kernel the
    leading term alone runs 8 to 25% high over ``w = 3.5`` to ``9.2``; with the
    constant the agreement is better than 2%, and against PROPOSAL itself the
    spread comes out to 3.5% rms with nothing fitted. The second-order
    drift-diffusion transport, by contrast, is 30 to 40% low at every ``w`` and
    worsens with distance, because it carries ``d_mu = <y^2>`` where this
    quantity needs ``<ln^2(1-y)>``, and the two differ by a factor of four. See
    ``examples/39_range_moment_estimator.py``.

    Parameters
    ----------
    energy_gev : float or np.ndarray
        Muon energy at production [GeV].
    threshold_gev : float, optional
        Muon energy below which the track is not selected [GeV]. Defaults to
        :data:`DEFAULT_MUON_THRESHOLD_GEV`.
    density_g_cm3 : float, optional
        Target-medium density [g cm^-3]. Defaults to water.
    b_scale : float, optional
        Multiplicative rescaling of the kernel normalization. All three moments
        scale with it, so the variance scales as ``1 / b_scale^2``.
    source : {"proposal"}, optional
        Transport-coefficient tabulation; see
        :mod:`softpaws.transport.coefficients`. Needs the log-loss moment
        columns, which ``"table1"`` does not have.

    Returns
    -------
    variance_km2 : np.ndarray
        Variance of the range [km^2 w.e.], zero below threshold.

    Notes
    -----
    Purely radiative, with no counterpart to the ionization splice of
    :func:`stochastic_muon_range_km`. Below the matching energy the loss is
    deterministic and adds no variance of its own, but the energy at which the
    muon *arrives* there fluctuates by the overshoot, and that fluctuation is
    anticorrelated with the first-passage depth above it. Reproducing the
    spliced variance therefore needs that covariance, which is not derived here;
    for a threshold at or below the critical energy this result is the
    stochastic part alone.

    The expansion is in ``1 / w`` and its constant term is negative, so it
    returns zero rather than a negative variance for ``w`` below about one,
    where a muon reaches the threshold in a handful of collisions and no
    expansion of this kind applies.
    """
    if kernel_evaluation not in ("frozen", "running"):
        raise ValueError(
            f"kernel_evaluation must be 'frozen' or 'running', got {kernel_evaluation!r}."
        )
    energy = np.atleast_1d(np.asarray(energy_gev, dtype=float))
    phi_prime, phi_second, phi_third = (
        b_scale * moment for moment in log_loss_moments(energy, density_g_cm3, source)
    )
    if kernel_evaluation == "running":
        # Both constants belong to the crossing, so both are read at the level
        # being crossed; only the term linear in w accumulates down the descent.
        phi_prime, phi_second, phi_third = (
            b_scale * np.reshape(moment, ())
            for moment in log_loss_moments(np.array([threshold_gev]), density_g_cm3, source)
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.log(energy / threshold_gev)
        overshoot_variance = (
            phi_third / (3.0 * phi_prime) - phi_second**2 / (4.0 * phi_prime**2)
        ) / phi_prime**2
        if kernel_evaluation == "running":
            linear = _running_variance_rate_km2(
                energy, threshold_gev, density_g_cm3, b_scale, source
            )
        else:
            linear = phi_second * w / phi_prime**3
        variance = linear - overshoot_variance

    out = np.where(energy > threshold_gev, np.clip(variance, 0.0, None), 0.0)
    return out.reshape(np.shape(energy_gev)) if np.ndim(energy_gev) else out


def truncated_muon_range_km(
    energy_gev: float | np.ndarray,
    column_km: float | np.ndarray,
    threshold_gev: float = DEFAULT_MUON_THRESHOLD_GEV,
    density_g_cm3: float = RHO_WATER_G_CM3,
    b_scale: float = 1.0,
    source: str = DEFAULT_SOURCE,
    kernel_evaluation: str = "running",
) -> np.ndarray:
    """First-passage range cut at a finite upstream column [km].

    :func:`stochastic_muon_range_km` integrates the first-passage probability to
    infinite depth, which is right whenever the medium supplies more column than
    any muon survives and wrong whenever it does not. A downgoing track is always
    the second case, since the muon cannot be born above the ice; so is any
    detector under a few km of water, where an upgoing muon at the top of the
    band would need more column than the site has. The honest length is then the
    *limited* expectation

    .. math:: L(\\varepsilon, X) = \\mathbb{E}[\\tau(w) \\wedge X]
        = \\int_0^X {\\rm d}\\ell\\;\\mathbb{P}[W(\\ell) < w].

    Evaluating that integral directly needs the log-loss CDF at every depth.
    Matching a gamma law to the first two moments of the first-passage depth --
    :func:`stochastic_muon_range_km` and
    :func:`stochastic_muon_range_variance_km2` -- turns it into an incomplete
    gamma function instead, at no cost in accuracy that matters here: example
    33's ``--check-truncation`` holds it against a direct Gil-Pelaez inversion.

    Both moments are built from the tabulated log-loss moments, so this agrees
    with :func:`stochastic_muon_range_km` with ``include_ionization=False`` in
    the ``column_km -> inf`` limit. There is no ionization splice: the truncation
    is only interesting where the column runs out well before the muon reaches
    the critical energy.

    Parameters
    ----------
    energy_gev : float or np.ndarray
        Muon energy at production [GeV].
    column_km : float or np.ndarray
        Available upstream column, as a length of the medium [km]. Broadcast
        against ``energy_gev``; ``inf`` returns the untruncated range.
    threshold_gev : float, optional
        Muon energy below which the track is not selected [GeV]. Defaults to
        :data:`DEFAULT_MUON_THRESHOLD_GEV`.
    density_g_cm3 : float, optional
        Medium density [g cm^-3]. Defaults to water.
    b_scale : float, optional
        Multiplicative rescaling of the kernel normalization.
    source : {"proposal"}, optional
        Transport-coefficient tabulation; see
        :mod:`softpaws.transport.coefficients`.

    Returns
    -------
    length_km : np.ndarray
        Expected truncated range [km], zero for muons born below threshold.
    """
    if kernel_evaluation not in ("frozen", "running"):
        raise ValueError(
            f"kernel_evaluation must be 'frozen' or 'running', got {kernel_evaluation!r}."
        )
    energy = np.asarray(energy_gev, dtype=float)
    # The tabulated moments come back at least one-dimensional; keep the shape of
    # the input so a scalar energy gives a scalar length, as the callers assume.
    constant_at = energy if kernel_evaluation == "frozen" else np.array([threshold_gev])
    phi_prime, phi_second, _ = (
        np.reshape(b_scale * moment, np.shape(energy) if kernel_evaluation == "frozen" else ())
        for moment in log_loss_moments(constant_at, density_g_cm3, source)
    )

    selectable = energy > threshold_gev
    w = np.where(selectable, np.log(np.maximum(energy, threshold_gev) / threshold_gev), 0.0)
    if kernel_evaluation == "running":
        radiative = np.reshape(
            _running_radiative_length_km(
                np.maximum(energy, threshold_gev),
                threshold_gev,
                density_g_cm3,
                b_scale,
                source,
                "table",
                48,
            ),
            np.shape(energy),
        )
    else:
        radiative = w / phi_prime
    mean = radiative + phi_second / (2.0 * phi_prime**2)
    variance = np.reshape(
        stochastic_muon_range_variance_km2(
            np.maximum(energy, threshold_gev * (1.0 + 1.0e-12)),
            threshold_gev,
            density_g_cm3,
            b_scale,
            source,
            kernel_evaluation,
        ),
        np.shape(energy),
    )

    column = np.asarray(column_km, dtype=float)
    # An infinite column is the untruncated case; the general expression below
    # would evaluate inf * 0 on it.
    capped = np.where(np.isfinite(column), column, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        shape = mean**2 / variance
        x = capped / (variance / mean)
        limited = mean * gammainc(shape + 1.0, x) + capped * (1.0 - gammainc(shape, x))
    # A few e-folds above threshold the variance expansion floors at zero, where
    # the first passage is effectively deterministic and the limited expectation
    # is just the shorter of the two lengths.
    limited = np.where(variance > 0.0, limited, np.minimum(mean, capped))
    limited = np.where(np.isfinite(column), limited, mean)
    return np.where(selectable & (mean > 0.0), np.clip(limited, 0.0, None), 0.0)
