"""Muon drift and diffusion transport coefficients.

The soft expansion of the QED collision operator reduces muon energy loss to a
drift-diffusion process governed by two coefficients (arXiv:2607.13143,
Eq. 2.8): the drift ``b_mu`` [km^-1], the mean fractional energy loss per unit
length (so the muon range is ``1/b_mu``), and the diffusion ``d_mu`` [km^-1],
the variance of that loss. In the eigenvalue language of
:mod:`softpaws.transport.eigenvalue` they are the first two ``y``-moments of the
loss spectrum, ``b_mu = <y>`` and ``d_mu = <y^2>`` per unit length.

Two sources are available, selected by the ``source`` argument:

``"table1"``
    The paper's Table 1: two reference values in water, at 1 PeV and 100 PeV,
    log-energy interpolated and clipped outside. This is what reproduces the
    paper.
``"proposal"`` (the default)
    A table computed with PROPOSAL (Koehne et al., arXiv:1809.07740) over
    ``10^2`` to ``10^10`` GeV, summing bremsstrahlung, ``e+e-`` pair production
    and the photonuclear channel. Built by :func:`build_proposal_table` and
    shipped under ``src/softpaws/data/coefficients/``.
``"proposal_rock"``
    The same table in PROPOSAL's standard rock (``Z = 11``, ``A = 22``,
    ``rho = 2.65``), stored per km of *water-equivalent* column so it compares
    with the water table at equal column depth. Its ``Z^2/A`` puts ``b_mu``
    and the log-loss moments 26-30% above water at ``10^4`` to ``10^6`` GeV
    and 20-24% above at ``10^7`` to ``10^9``. An upgoing muon at IceCube or
    ARCA spends most of its range below the ice or the sea floor, which is
    what this source is for.

The two differ by more than the clipping: PROPOSAL puts ``b_mu`` about 7-8%
above Table 1 at *both* of its anchor energies and ``d_mu`` 13-20% above, and
below 1 PeV -- where Table 1 has no data at all and is held flat -- the gap
reaches 15% at 1 TeV, with the sign reversed. Since ``Phi(1) = b_mu`` exactly
and the IceCube spectrum sits at ``A ~ 1``, the soft volume goes as ``1/b_mu``
and inherits those differences directly.

The Landau-Pomeranchuk-Migdal effect is deliberately *not* included: checked
against PROPOSAL's own LPM switch, the suppression of muon radiative losses is
below ``10^-5`` even at ``10^10`` GeV, three decades above the top of the range
used here. LPM does matter for the electromagnetic showers the radiated photons
initiate, but that is a detector-response effect, and for this comparison it
lives inside the published IceCube response rather than in the transport.
"""

from __future__ import annotations

import contextlib
import dataclasses
import pathlib
from typing import Callable

import numpy as np

from ..utils.constants import CM_PER_KM, RHO_WATER_G_CM3

# ---------------------------------------------------------------------------
# Table 1 of arXiv:2607.13143: total QED coefficients for muons in water
# (rho = 1.02 g/cm^3), at two reference energies.
# ---------------------------------------------------------------------------

_REF_LOG10_E = np.array([6.0, 8.0])  # log10(E / GeV) for 1 PeV and 100 PeV
_REF_B_MU = np.array([0.35, 0.40])  # drift [km^-1]
_REF_D_MU = np.array([0.0766, 0.0982])  # diffusion [km^-1]

# Table 1 stops at two moments, so the third-moment column has no Table 1
# counterpart; see third_moment_coefficient.
_TABLE1_N_MOMENTS = 2

# ---------------------------------------------------------------------------
# PROPOSAL-computed table, shipped with the package.
# ---------------------------------------------------------------------------

_COEFFICIENT_DIR = pathlib.Path(__file__).parents[1] / "data" / "coefficients"
PROPOSAL_TABLE_PATH = _COEFFICIENT_DIR / "proposal_muon_water.csv"
PROPOSAL_ROCK_TABLE_PATH = _COEFFICIENT_DIR / "proposal_muon_rock.csv"

TABLE1_SOURCE = "table1"
PROPOSAL_SOURCE = "proposal"
PROPOSAL_ROCK_SOURCE = "proposal_rock"
DEFAULT_SOURCE = PROPOSAL_SOURCE

#: PROPOSAL-built sources, the medium each is computed in (a ``proposal.medium``
#: class name) and the shipped table. Every table is stored per km of
#: water-equivalent column, so the sources compare at equal column depth and
#: differ only through the medium's ``Z``, ``A`` and composition.
_PROPOSAL_SOURCES = {
    PROPOSAL_SOURCE: ("Water", PROPOSAL_TABLE_PATH),
    PROPOSAL_ROCK_SOURCE: ("StandardRock", PROPOSAL_ROCK_TABLE_PATH),
}

_proposal_tables: dict[str, tuple[np.ndarray, ...]] = {}


def _load_proposal_table(source: str = PROPOSAL_SOURCE) -> tuple[np.ndarray, ...]:
    """Read and cache a shipped PROPOSAL coefficient table.

    Parameters
    ----------
    source : {"proposal", "proposal_rock"}, optional
        Which table; see :data:`_PROPOSAL_SOURCES`. Defaults to water.

    Returns
    -------
    columns : tuple of np.ndarray
        ``log10(E / GeV)`` followed by the tabulated ``y``-moments [km^-1] at
        :data:`~softpaws.utils.constants.RHO_WATER_G_CM3` -- ``b_mu``, ``d_mu``
        and, for tables built since the third moment was added, ``t_mu``.

    Raises
    ------
    FileNotFoundError
        Raised if the table is missing; rebuild it with
        :func:`build_proposal_table`.
    """
    if source not in _proposal_tables:
        medium, path = _PROPOSAL_SOURCES[source]
        if not path.exists():
            raise FileNotFoundError(
                f"PROPOSAL coefficient table not found at {path}; rebuild it with "
                f"build_proposal_table(medium={medium!r}) (requires the 'proposal' "
                f"package), or pass source='{TABLE1_SOURCE}'."
            )
        table = np.loadtxt(path, delimiter=",")
        _proposal_tables[source] = (np.log10(table[:, 0]), *table[:, 1:].T)
    return _proposal_tables[source]



# ---------------------------------------------------------------------------
# Kernel scaling hook, for the loss-model error budget
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class KernelScaling:
    """Energy-dependent factors that rescale the tabulated kernel moments.

    Used by :mod:`softpaws.transport.loss_ensemble` to push one alternative
    loss parametrization through every calculation without rebuilding the
    table: the ensemble is linear in the loss spectrum, so a swapped channel
    is a ratio of moments at each energy.

    Attributes
    ----------
    kappa_1 : callable
        Ratio of the first moments, ``Phi'_var(0) / Phi'(0)`` and
        ``b_var / b``, as a function of the muon energy [GeV].
    kappa_2 : callable
        Ratio of the second moments, applied to ``d_mu``, to ``-Phi''(0)``
        and to the third log-loss moment.
    """

    kappa_1: Callable[[np.ndarray], np.ndarray]
    kappa_2: Callable[[np.ndarray], np.ndarray]


_KERNEL_SCALING: KernelScaling | None = None


def set_kernel_scaling(scaling: KernelScaling | None) -> None:
    """Install (or with ``None`` remove) a global :class:`KernelScaling`.

    While a scaling is installed, :func:`drift_coefficient`,
    :func:`diffusion_coefficient` and :func:`log_loss_moments` multiply their
    results by its factors at the requested energy, so everything built on
    them sees the alternative kernel.

    Parameters
    ----------
    scaling : KernelScaling or None
        The factors to apply, or ``None`` for the shipped table.
    """
    global _KERNEL_SCALING
    _KERNEL_SCALING = scaling


def kernel_scaling() -> KernelScaling | None:
    """The installed :class:`KernelScaling`, or ``None``."""
    return _KERNEL_SCALING


@contextlib.contextmanager
def scaled_kernel(scaling: KernelScaling | None):
    """Install a :class:`KernelScaling` for a block and restore the old one after.

    Parameters
    ----------
    scaling : KernelScaling or None
        The factors to apply inside the block.

    Examples
    --------
    >>> with scaled_kernel(None):
    ...     b = drift_coefficient(1.0e6)
    """
    previous = _KERNEL_SCALING
    set_kernel_scaling(scaling)
    try:
        yield
    finally:
        set_kernel_scaling(previous)


def _scale(value, energy_gev, order: int):
    """Apply the installed scaling of the given moment order, if any."""
    if _KERNEL_SCALING is None:
        return value
    factor = _KERNEL_SCALING.kappa_1 if order == 1 else _KERNEL_SCALING.kappa_2
    return value * factor(energy_gev)


def _interpolate(
    energy_gev: float | np.ndarray,
    source: str,
    column: int,
) -> np.ndarray:
    """Log-energy interpolation of a coefficient from the selected source."""
    log10_e = np.log10(np.atleast_1d(np.asarray(energy_gev, dtype=float)))
    if source == TABLE1_SOURCE:
        if column >= _TABLE1_N_MOMENTS:
            raise ValueError(
                f"source={TABLE1_SOURCE!r} tabulates only b_mu and d_mu; the third "
                f"moment needs source={PROPOSAL_SOURCE!r}."
            )
        reference = (_REF_LOG10_E, (_REF_B_MU, _REF_D_MU)[column])
    elif source in _PROPOSAL_SOURCES:
        grid = _load_proposal_table(source)
        if column + 1 >= len(grid):
            raise ValueError(
                f"the PROPOSAL table at {_PROPOSAL_SOURCES[source][1]} has only "
                f"{len(grid) - 1} moment column(s); rebuild it with "
                "build_proposal_table() to add the third moment."
            )
        reference = (grid[0], grid[column + 1])
    else:
        raise ValueError(
            f"source must be {TABLE1_SOURCE!r}, {PROPOSAL_SOURCE!r} or "
            f"{PROPOSAL_ROCK_SOURCE!r}, got {source!r}."
        )
    return np.interp(log10_e, *reference)


def proposal_parametrizations() -> dict[str, object]:
    """The three radiative channels the shipped table is built from.

    Bremsstrahlung and ``e+e-`` pair production (both Kelner-Kokoulin-Petrukhin)
    and photonuclear (ALLM97 with Butkevich-Mikheyev shadowing). Ionization is
    excluded, matching Table 1's convention; it enters separately through
    :func:`ionization_coefficient`.

    Requires the optional ``proposal`` dependency.

    Returns
    -------
    parametrizations : dict
        Channel label to PROPOSAL parametrization object.
    """
    import proposal as pp

    return {
        "bremsstrahlung": pp.parametrization.bremsstrahlung.KelnerKokoulinPetrukhin(False),
        "pair production": pp.parametrization.pairproduction.KelnerKokoulinPetrukhin(False),
        "photonuclear": pp.parametrization.photonuclear.AbramowiczLevinLevyMaor97(
            pp.parametrization.photonuclear.ShadowButkevichMikheyev()
        ),
    }


def loss_spectrum_y_grid(
    n_soft: int = 3000,
    n_hard: int = 2000,
    log10_y_min: float = -12.0,
    log10_one_minus_y_min: float = -8.0,
) -> np.ndarray:
    """Grid in ``y`` resolving both the soft pile-up and the ``y -> 1`` edge.

    Log spaced in ``y`` below the midpoint and log spaced in ``1 - y`` above it.
    A single log grid in ``y`` leaves the hard edge coarse, which costs about a
    percent on the second moment; this grid closes the first three moments of
    PROPOSAL's spectrum to better than 0.1%.

    Parameters
    ----------
    n_soft, n_hard : int, optional
        Number of nodes below and above ``y = 0.5``.
    log10_y_min : float, optional
        Smallest ``log10(y)`` on the soft branch.
    log10_one_minus_y_min : float, optional
        Smallest ``log10(1 - y)`` on the hard branch.

    Returns
    -------
    y : np.ndarray
        Ascending, strictly increasing grid of fractional energy losses in
        ``(0, 1)``.
    """
    soft = np.logspace(log10_y_min, np.log10(0.5), n_soft)
    hard = 1.0 - np.logspace(log10_one_minus_y_min, np.log10(0.5), n_hard)[::-1]
    return np.unique(np.concatenate([soft, hard]))


def proposal_loss_spectrum(
    energy_gev: float, y: np.ndarray, medium: str = "Water"
) -> dict[str, np.ndarray]:
    """PROPOSAL's differential loss rate ``dGamma/dy``, per channel.

    PROPOSAL's ``differential_crosssection`` is per unit column density, per
    component, so the rate per unit length is the mass-fraction-weighted sum over
    components times the mass density. The result is rescaled to
    :data:`~softpaws.utils.constants.RHO_WATER_G_CM3`, matching the convention of
    the shipped tables: a rate per km of water-equivalent column, whatever the
    medium, so that media compare at equal column depth.

    This is the spectrum whose moments :func:`build_proposal_table` tabulates and
    which the calibrated families of :mod:`softpaws.transport.eigenvalue`
    approximate. Requires the optional ``proposal`` dependency.

    Parameters
    ----------
    energy_gev : float
        Muon energy [GeV].
    y : np.ndarray
        Fractional energy losses in ``(0, 1)``; see :func:`loss_spectrum_y_grid`.
    medium : str, optional
        A ``proposal.medium`` class name, ``"Water"`` (the default), ``"Ice"``
        or ``"StandardRock"``. Ice and water coincide per unit column; standard
        rock sits 20 to 30% above them through its ``Z^2/A``.

    Returns
    -------
    spectrum : dict
        Channel label to ``dGamma/dy`` [km^-1], plus the key ``"total"``. Values
        outside a channel's kinematic limits are zero.
    """
    import proposal as pp

    particle = pp.particle.MuMinusDef()
    medium = getattr(pp.medium, medium)()
    energy_mev = energy_gev * 1.0e3

    # PROPOSAL's atomic_number is the atomic mass, so these are mass fractions.
    molar_mass = sum(c.atoms_in_molecule * c.atomic_number for c in medium.components)
    scale = CM_PER_KM * RHO_WATER_G_CM3 / medium.mass_density

    spectrum: dict[str, np.ndarray] = {}
    for label, param in proposal_parametrizations().items():
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
        spectrum[label] = rate * scale
    spectrum["total"] = sum(spectrum.values())
    return spectrum


def build_proposal_table(
    path: str | pathlib.Path | None = None,
    log10_e_min: float = 2.0,
    log10_e_max: float = 10.0,
    points_per_decade: int = 8,
    medium: str = "Water",
) -> np.ndarray:
    """Tabulate the first three ``y``-moments with PROPOSAL and write them to disk.

    Sums the three radiative channels of :func:`proposal_parametrizations` with no
    energy cut, so every loss is continuous and PROPOSAL's ``dEdx`` and ``dE2dx``
    are exactly the first two moments the drift-diffusion expansion needs.

    PROPOSAL exposes no third-moment accumulator, so ``t_mu = <y^3>`` is obtained
    by quadrature of :func:`proposal_loss_spectrum` on
    :func:`loss_spectrum_y_grid`. The same quadrature reproduces ``dEdx`` and
    ``dE2dx`` to better than 0.1%, which is what validates it; that closure is
    asserted here rather than assumed.

    Only needed to regenerate the shipped table, and with
    :func:`proposal_loss_spectrum` the only place the optional ``proposal``
    dependency is used.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        Destination CSV. Defaults to the shipped table for ``medium``,
        :data:`PROPOSAL_TABLE_PATH` for water and
        :data:`PROPOSAL_ROCK_TABLE_PATH` for standard rock.
    log10_e_min, log10_e_max : float, optional
        Range of the table in ``log10(E / GeV)``. The default upper limit stays
        below the energy where PROPOSAL's own interpolation tables break down.
    points_per_decade : int, optional
        Sampling density.
    medium : str, optional
        A ``proposal.medium`` class name; see :func:`proposal_loss_spectrum`.
        Defaults to ``"Water"``.

    Returns
    -------
    table : np.ndarray, shape (n, 7)
        Columns of energy [GeV], the ``y``-moments ``b_mu``, ``d_mu``, ``t_mu``,
        and the log-loss moments ``Phi'(0)``, ``-Phi''(0)``, ``Phi'''(0)``, all
        [km^-1] at :data:`~softpaws.utils.constants.RHO_WATER_G_CM3`.

    Raises
    ------
    RuntimeError
        Raised if the quadrature of ``dGamma/dy`` disagrees with PROPOSAL's own
        ``dEdx`` or ``dE2dx`` by more than 1%, which would mean the reconstructed
        spectrum feeding ``t_mu`` cannot be trusted.
    """
    import proposal as pp

    if path is None:
        path = {name: table for name, table in _PROPOSAL_SOURCES.values()}[medium]
    medium_name = medium
    particle = pp.particle.MuMinusDef()
    medium = getattr(pp.medium, medium_name)()
    # v_cut = 1 puts every loss in the continuous part; the third flag switches
    # on the second moment (dE2dx), which is zero without it.
    cuts = pp.EnergyCutSettings(np.inf, 1, True)
    cross_sections = [
        pp.crosssection.make_crosssection(p, particle, medium, cuts, True)
        for p in proposal_parametrizations().values()
    ]

    n_points = int(round((log10_e_max - log10_e_min) * points_per_decade)) + 1
    energy_gev = np.logspace(log10_e_min, log10_e_max, n_points)
    # PROPOSAL works in MeV, and ``calculate_dEdx`` and ``calculate_dE2dx`` are
    # per unit column density (MeV cm^2 g^-1), not per cm, so no medium density
    # enters: the table is stored per km of water-equivalent column whatever the
    # medium, matching Table 1's convention. Dividing by the medium density here
    # was invisible for water and wrong by 2.65 for standard rock.
    scale = CM_PER_KM * RHO_WATER_G_CM3
    b_mu = np.array(
        [sum(c.calculate_dEdx(e * 1.0e3) for c in cross_sections) / (e * 1.0e3) for e in energy_gev]
    ) * scale
    d_mu = np.array(
        [
            sum(c.calculate_dE2dx(e * 1.0e3) for c in cross_sections) / (e * 1.0e3) ** 2
            for e in energy_gev
        ]
    ) * scale

    y = loss_spectrum_y_grid()
    log_loss = -np.log1p(-y)
    t_mu = np.empty_like(energy_gev)
    phi_moments = np.empty((energy_gev.size, 3))
    for i, energy in enumerate(energy_gev):
        spectrum = proposal_loss_spectrum(energy, y, medium_name)["total"]
        for order, reference in ((1, b_mu[i]), (2, d_mu[i])):
            quadrature = np.trapezoid(y**order * spectrum, y)
            # 0.1% with PROPOSAL's shipped interpolation tables; freshly generated
            # ones (written to /tmp on first use) sit 2% from the quadrature in
            # water and rock alike, so the guard admits that.
            if abs(quadrature / reference - 1.0) > 0.025:
                raise RuntimeError(
                    f"dGamma/dy quadrature disagrees with PROPOSAL's moment {order} "
                    f"at E = {energy:.3g} GeV: {quadrature:.6g} vs {reference:.6g}."
                )
        t_mu[i] = np.trapezoid(y**3 * spectrum, y)
        for order in (1, 2, 3):
            phi_moments[i, order - 1] = np.trapezoid(log_loss**order * spectrum, y)

    table = np.column_stack([energy_gev, b_mu, d_mu, t_mu, phi_moments])
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        table,
        delimiter=",",
        header=(
            f"Muon transport coefficients from PROPOSAL in {medium_name} "
            f"(rho = {medium.mass_density} g/cm^3), per km of water-equivalent "
            f"column at rho = {RHO_WATER_G_CM3} g/cm^3.\n"
            "Bremsstrahlung + e+e- pair production (Kelner-Kokoulin-Petrukhin) + "
            "photonuclear (ALLM97, Butkevich-Mikheyev shadowing); no ionization.\n"
            "b_mu = <y>, d_mu = <y^2>, t_mu = <y^3> per unit length.\n"
            "The last three are the log-loss moments the first-passage range needs, "
            "Phi^(n)(0) = <(-ln(1-y))^n>, which no family calibrated to the "
            "y-moments reproduces: a two-moment fit is 8% low on the first and 56% "
            "low on the second.\n"
            "E [GeV], b_mu [km^-1], d_mu [km^-1], t_mu [km^-1], "
            "phi1 [km^-1], phi2 [km^-1], phi3 [km^-1]"
        ),
    )
    return table

# Ionization (Bethe) energy loss for muons in water, a_mu ~ 2.0e-3 GeV cm^2 g^-1
# (PDG muon tables; it varies only logarithmically from 1 GeV to 100 TeV). Table 1
# tabulates the radiative b_mu and d_mu alone, which is all the soft-volume drift
# limit needs. The constant term is needed to close the loss law
# ``-dE/dx = a_mu + b_mu E`` at low energy, and so to define the muon range.
IONIZATION_A_GEV_CM2_G = 2.0e-3


def drift_coefficient(
    energy_gev: float | np.ndarray,
    density_g_cm3: float = RHO_WATER_G_CM3,
    source: str = DEFAULT_SOURCE,
) -> np.ndarray:
    """Muon drift coefficient ``b_mu`` at a given energy.

    Parameters
    ----------
    energy_gev : float or np.ndarray
        Muon energy [GeV].
    density_g_cm3 : float, optional
        Target-medium density [g cm^-3]. Defaults to water. The coefficient
        scales linearly with density, since it is proportional to the target
        number density.
    source : {"proposal", "table1"}, optional
        Which tabulation to interpolate; see the module docstring. Defaults to
        :data:`DEFAULT_SOURCE`.

    Returns
    -------
    b_mu : np.ndarray
        Drift coefficient [km^-1].

    Notes
    -----
    Values are log-linearly interpolated in ``log10(E / GeV)`` and clipped to
    the endpoints of the selected source: for ``"table1"`` just the two
    reference energies (1 PeV and 100 PeV), for ``"proposal"`` the full table
    outside that range. The energy dependence of the QED coefficients is slow
    (logarithmic), so this is adequate in the drift limit.
    """
    b_water = _interpolate(energy_gev, source, 0)
    return _scale(b_water * (density_g_cm3 / RHO_WATER_G_CM3), energy_gev, 1)


def diffusion_coefficient(
    energy_gev: float | np.ndarray,
    density_g_cm3: float = RHO_WATER_G_CM3,
    source: str = DEFAULT_SOURCE,
) -> np.ndarray:
    """Muon diffusion coefficient ``d_mu`` at a given energy.

    Parameters
    ----------
    energy_gev : float or np.ndarray
        Muon energy [GeV].
    density_g_cm3 : float, optional
        Target-medium density [g cm^-3]. Defaults to water. Scales linearly
        with density.
    source : {"proposal", "table1"}, optional
        Which tabulation to interpolate; see the module docstring. Defaults to
        :data:`DEFAULT_SOURCE`.

    Returns
    -------
    d_mu : np.ndarray
        Diffusion coefficient [km^-1].

    Notes
    -----
    Interpolated as in :func:`drift_coefficient`. Not used in the drift limit;
    tabulated for the diffusion extension.
    """
    d_water = _interpolate(energy_gev, source, 1)
    return _scale(d_water * (density_g_cm3 / RHO_WATER_G_CM3), energy_gev, 2)


def third_moment_coefficient(
    energy_gev: float | np.ndarray,
    density_g_cm3: float = RHO_WATER_G_CM3,
    source: str = DEFAULT_SOURCE,
) -> np.ndarray:
    """Muon third loss moment ``t_mu = <y^3>`` at a given energy.

    The extra input the three-moment loss family needs
    (:func:`softpaws.transport.eigenvalue.three_moment_loss_spectrum`). Where
    ``b_mu`` and ``d_mu`` fix the mean and variance of the energy loss, ``t_mu``
    fixes its skewness, which is what pins the hard end of ``dGamma/dy`` and so
    the tail of the log-loss law.

    Parameters
    ----------
    energy_gev : float or np.ndarray
        Muon energy [GeV].
    density_g_cm3 : float, optional
        Target-medium density [g cm^-3]. Defaults to water. Scales linearly with
        density, as for the lower moments.
    source : {"proposal"}, optional
        Which tabulation to interpolate. Defaults to :data:`DEFAULT_SOURCE`.

    Returns
    -------
    t_mu : np.ndarray
        Third moment [km^-1].

    Raises
    ------
    ValueError
        Raised for ``source="table1"``, which tabulates only two moments, or if
        the shipped PROPOSAL table predates the third-moment column.
    """
    t_water = _interpolate(energy_gev, source, 2)
    return t_water * (density_g_cm3 / RHO_WATER_G_CM3)


def log_loss_moments(
    energy_gev: float | np.ndarray,
    density_g_cm3: float = RHO_WATER_G_CM3,
    source: str = DEFAULT_SOURCE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three log-loss moments of the kernel, from the tabulated spectrum.

    Where ``b_mu``, ``d_mu`` and ``t_mu`` are moments of the fractional loss
    ``y``, these are moments of the *logarithmic* loss ``-ln(1-y)``, which is
    what the first-passage range of
    :func:`softpaws.transport.soft_volume.stochastic_muon_range_km` and its
    variance are built from:

    .. math:: \\Phi'(0) = \\langle -\\ln(1-y)\\rangle, \\quad
        -\\Phi''(0) = \\langle \\ln^2(1-y)\\rangle, \\quad
        \\Phi'''(0) = \\langle -\\ln^3(1-y)\\rangle .

    They are read from the shipped table rather than reconstructed from
    ``b_mu``, ``d_mu``, ``t_mu``, because no family calibrated to the
    ``y``-moments reproduces them. ``-ln(1-y)`` diverges as ``y -> 1`` where
    ``y`` saturates, so these moments are dominated by the hard end of the
    kernel that a calibrated family gets wrong: against PROPOSAL's own spectrum
    the two-moment family of
    :func:`softpaws.transport.eigenvalue.two_moment_loss_spectrum` is 8% low on
    the first, 56% low on the second and 87% low on the third, and the
    three-moment family is 1% high, 15% high and 40% high.

    Parameters
    ----------
    energy_gev : float or np.ndarray
        Muon energy [GeV].
    density_g_cm3 : float, optional
        Target-medium density [g cm^-3]. Defaults to water. Scales linearly with
        density, as for the ``y``-moments.
    source : {"proposal"}, optional
        Which tabulation to interpolate. Defaults to :data:`DEFAULT_SOURCE`.

    Returns
    -------
    phi_prime, phi_second, phi_third : np.ndarray
        ``Phi'(0)``, ``-Phi''(0)`` and ``Phi'''(0)`` [km^-1], all positive.

    Raises
    ------
    ValueError
        Raised for ``source="table1"``, which tabulates no log-loss moments, or
        if the shipped PROPOSAL table predates these columns.
    """
    scale = density_g_cm3 / RHO_WATER_G_CM3
    return tuple(
        _scale(_interpolate(energy_gev, source, column) * scale, energy_gev, order)
        for column, order in ((3, 1), (4, 2), (5, 2))
    )


def ionization_coefficient(density_g_cm3: float = RHO_WATER_G_CM3) -> float:
    """Muon ionization loss coefficient ``a_mu``.

    Parameters
    ----------
    density_g_cm3 : float, optional
        Target-medium density [g cm^-3]. Defaults to water. Scales linearly with
        density, as for the QED coefficients.

    Returns
    -------
    a_mu : float
        Ionization loss [GeV km^-1], the constant term of
        ``-dE/dx = a_mu + b_mu E``.
    """
    return IONIZATION_A_GEV_CM2_G * density_g_cm3 * CM_PER_KM


def critical_energy_gev(
    energy_gev: float | np.ndarray,
    density_g_cm3: float = RHO_WATER_G_CM3,
    b_scale: float = 1.0,
    source: str = DEFAULT_SOURCE,
) -> np.ndarray:
    """Muon critical energy ``E_c = a_mu / b_mu``.

    Above ``E_c`` radiative losses dominate and the drift limit of
    :mod:`softpaws.transport.soft_volume` applies; below it the muon slows at the
    near-constant ionization rate. In water ``E_c ~ 570 GeV``.

    Parameters
    ----------
    energy_gev : float or np.ndarray
        Muon energy [GeV], at which ``b_mu`` is evaluated. The critical energy
        inherits ``b_mu``'s slow (logarithmic) energy dependence.
    density_g_cm3 : float, optional
        Target-medium density [g cm^-3]. Defaults to water. Cancels between
        ``a_mu`` and ``b_mu``, so ``E_c`` is density independent.
    b_scale : float, optional
        Multiplicative rescaling of the drift coefficient. Defaults to 1.
    source : {"proposal", "table1"}, optional
        Which tabulation to interpolate; see the module docstring. Defaults to
        :data:`DEFAULT_SOURCE`.

    Returns
    -------
    e_crit : np.ndarray
        Critical energy [GeV], broadcast to the shape of ``energy_gev``.
    """
    b_mu = b_scale * drift_coefficient(energy_gev, density_g_cm3, source)
    return ionization_coefficient(density_g_cm3) / b_mu
