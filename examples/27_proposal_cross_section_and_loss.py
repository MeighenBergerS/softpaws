"""Example 27 — softpaws loss spectrum and log-loss law against PROPOSAL.

Examples 08 and 13 compare the paper's Fokker-Planck truncation against the
exact eigenvalue treatment, but both sides of that comparison are *ours*. This
script benchmarks the transport input itself against PROPOSAL (Koehne et al.,
arXiv:1809.07740), in three steps that follow the same quantity from the
differential cross section to the observable log-loss law:

1. **Cross section.** PROPOSAL's differential loss rate ``dGamma/dy`` for the
   three radiative channels, against the calibrated families of
   :mod:`softpaws.transport.eigenvalue`: the two-parameter
   ``kappa (1 - y)^p / y`` fixed by ``b_mu`` and ``d_mu``, and the
   three-parameter ``kappa y^(q-1) (1 - y)^p`` fixed by ``b_mu``, ``d_mu`` and
   ``t_mu``. All are PROPOSAL-derived here -- the shipped coefficient table comes
   from the same channels -- so each agrees on the moments it was calibrated to
   by construction, and any difference is purely a failure of *shape*.
2. **Eigenvalue.** The soft volume only ever needs ``Phi(A)``, so the shape
   error is projected onto that one number: a direct quadrature of PROPOSAL's
   ``dGamma/dy`` against the two closed forms, the Fokker-Planck truncation, and
   the drift-only limit.
3. **Loss law.** The full distribution of ``w = ln(eps / E)`` from a PROPOSAL
   Monte-Carlo propagation, against the subordinator inversion driven by
   PROPOSAL's own ``dGamma/dy``, the two closed-form inversions, and the
   Fokker-Planck Gaussian.

The first headline is that the **mean is not the interesting benchmark**. The
mean survival ``<E_f / E_i>`` reproduces ``exp(-b_mu ell)`` to about a percent,
and ``Phi(A)`` agrees within a few percent near ``A ~ 1`` where the IceCube
spectrum sits -- yet the median muon loses roughly half the mean log-loss, and
the Fokker-Planck Gaussian understates the tail ``P(W > 1.5)`` by two to three
orders of magnitude.

The second is that **one more moment fixes almost all of it**. Matching only
``b_mu`` and ``d_mu`` pins the ``y ~ 0.1-0.7`` window that generates them and
leaves the hard bremsstrahlung tail wrong by three orders of magnitude at
``y = 0.99`` -- and it is those rare hard scatters that set the fluctuations.
Adding ``t_mu = <y^3>`` moves the freed parameter onto exactly that end (``p``
drops from ``+2.1`` to ``-0.18``, so ``dGamma/dy`` no longer vanishes as
``y -> 1``), which brings ``Phi(A)`` inside a percent out to ``A = 8`` and the
log-loss tail into agreement with the Monte Carlo. The closed form survives:
the symbol is a difference of Beta functions.

Requires the optional ``proposal`` dependency (``pip install -e ".[transport]"``).
The PROPOSAL differential cross sections live in
:mod:`softpaws.transport.coefficients`, which needs them to tabulate ``t_mu``;
what stays local here is only the auditing -- the Monte Carlo, the numerical
``Phi``, and the comparisons.

Usage
-----
    python examples/27_proposal_cross_section_and_loss.py
    python examples/27_proposal_cross_section_and_loss.py --energy-gev 1e8 --ell-km 3
    python examples/27_proposal_cross_section_and_loss.py --n-muons 20000
"""

import argparse
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np

from softpaws.transport.coefficients import (
    diffusion_coefficient,
    drift_coefficient,
    loss_spectrum_y_grid,
    proposal_loss_spectrum,
    proposal_parametrizations,
    third_moment_coefficient,
)
from softpaws.transport.eigenvalue import (
    phi_drift,
    phi_eigenvalue,
    phi_eigenvalue_three_moment,
    phi_fokker_planck,
    three_moment_loss_spectrum,
    two_moment_loss_spectrum,
)
from softpaws.transport.loss_distribution import (
    gaussian_survival,
    invert_log_loss_symbol,
    loss_density,
    loss_density_three_moment,
    survival_from_density,
)
from softpaws.utils.constants import CM_PER_KM, RHO_WATER_G_CM3

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"

# Fractional energy losses at which to tabulate the dGamma/dy comparison, chosen
# to straddle the soft pile-up, the moment-generating window, and the hard tail.
_Y_PROBES = (1.0e-6, 1.0e-4, 1.0e-2, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99)

# Spectral indices for the eigenvalue table. A ~ 1 is the IceCube regime.
_A_PROBES = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)

# Log-loss thresholds for the tail table, i.e. eps / E = e^w of 1.6 to 20.
_W_PROBES = (0.5, 1.0, 1.5, 2.0, 3.0)

# (1 - y) is floored before taking its log so that y = 1 stays finite; the
# integrand carries no weight there.
_ONE_MINUS_Y_FLOOR = 1.0e-16

# Panels (a) and (b) carry no secondary axis, so their titles need extra padding to
# line up with panel (c), whose title clears the eps / E axis on top.
_TITLE_PAD = 18.0

_PROPOSAL_STYLE = {"color": "black", "lw": 1.6, "label": "PROPOSAL"}
_TWO_MOMENT_STYLE = {
    "color": "#e7298a",
    "lw": 1.6,
    "ls": "--",
    "label": r"two-moment $(b_\mu, d_\mu)$",
}
_THREE_MOMENT_STYLE = {
    "color": "#0868ac",
    "lw": 1.6,
    "ls": (0, (5, 1, 1, 1)),
    "label": r"three-moment $(b_\mu, d_\mu, t_\mu)$",
}
_FP_STYLE = {"color": "#7570b3", "lw": 1.4, "ls": ":", "label": "Fokker-Planck"}
_DRIFT_STYLE = {"color": "#66a61e", "lw": 1.2, "ls": "-.", "label": r"drift only ($A b_\mu$)"}
_CHANNEL_STYLES = {
    "bremsstrahlung": {"color": "#1b9e77", "lw": 0.9, "alpha": 0.8},
    "pair production": {"color": "#d95f02", "lw": 0.9, "alpha": 0.8},
    "photonuclear": {"color": "#e6ab02", "lw": 0.9, "alpha": 0.8},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=_DEFAULT_OUT_DIR,
        help="Directory for the output figure.",
    )
    parser.add_argument(
        "--energy-gev",
        type=float,
        default=1.0e6,
        help="Initial muon energy [GeV] at which to compare (default: 1 PeV).",
    )
    parser.add_argument(
        "--ell-km",
        type=float,
        default=1.0,
        help="Propagated column depth ell [km] for the loss-law comparison.",
    )
    parser.add_argument(
        "--n-muons",
        type=int,
        default=4000,
        help="Number of muons in the PROPOSAL Monte Carlo.",
    )
    parser.add_argument(
        "--ecut-mev",
        type=float,
        default=500.0,
        help="PROPOSAL energy cut [MeV] separating stochastic from continuous losses.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed for the PROPOSAL random generator.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# PROPOSAL differential loss rate
# ---------------------------------------------------------------------------


def proposal_moments(energy_gev: float) -> tuple[float, float]:
    """PROPOSAL's ``b_mu`` and ``d_mu`` evaluated directly, without the table.

    Uses ``dEdx`` and ``dE2dx`` with every loss made continuous, which is exactly
    how :func:`softpaws.transport.coefficients.build_proposal_table` fills the
    shipped table. Serves as the reference for the moment closure of the
    reconstructed ``dGamma/dy``, free of the table's interpolation error.

    Parameters
    ----------
    energy_gev : float
        Muon energy [GeV].

    Returns
    -------
    b_mu, d_mu : float
        Drift and diffusion coefficients [km^-1] at
        :data:`~softpaws.utils.constants.RHO_WATER_G_CM3`.
    """
    import proposal as pp

    particle = pp.particle.MuMinusDef()
    medium = pp.medium.Water()
    # v_cut = 1 makes every loss continuous, so dEdx and dE2dx are the first two
    # y-moments with no cut dependence.
    cuts = pp.EnergyCutSettings(np.inf, 1, True)
    cross_sections = [
        pp.crosssection.make_crosssection(param, particle, medium, cuts, True)
        for param in proposal_parametrizations().values()
    ]
    energy_mev = energy_gev * 1.0e3
    scale = CM_PER_KM * RHO_WATER_G_CM3 / medium.mass_density
    b_mu = sum(c.calculate_dEdx(energy_mev) for c in cross_sections) / energy_mev
    d_mu = sum(c.calculate_dE2dx(energy_mev) for c in cross_sections) / energy_mev**2
    return b_mu * scale, d_mu * scale


# ---------------------------------------------------------------------------
# Eigenvalue and log-loss law from a tabulated dGamma/dy
# ---------------------------------------------------------------------------


def phi_from_spectrum(
    s: complex | np.ndarray,
    y: np.ndarray,
    dgamma_dy: np.ndarray,
    chunk: int = 64,
) -> np.ndarray:
    """Mellin symbol ``Phi(s)`` by quadrature of a tabulated loss rate.

    Evaluates ``Phi(s) = int_0^1 dy (dGamma/dy) [1 - (1 - y)^s]`` on the supplied
    grid, for real or complex ``s``. This is the same integral as
    :func:`softpaws.transport.eigenvalue.phi_eigenvalue_quadrature`, but on the
    caller's grid and accepting complex arguments: PROPOSAL's ``dGamma/dy`` rises
    faster than ``1/y``, which a linear grid cannot resolve, and complex ``s`` is
    what the characteristic function of :func:`subordinator_density` needs.

    Parameters
    ----------
    s : complex or np.ndarray
        Mellin variable. Real or complex.
    y : np.ndarray
        Ascending grid of fractional energy losses in ``(0, 1)``.
    dgamma_dy : np.ndarray
        Differential loss rate [km^-1] on ``y``.
    chunk : int, optional
        Number of ``s`` values evaluated per block, bounding peak memory.

    Returns
    -------
    phi : np.ndarray
        ``Phi(s)`` [km^-1], with the shape of ``s`` and a complex dtype if ``s``
        is complex.
    """
    s_array = np.atleast_1d(np.asarray(s))
    log_one_minus_y = np.log1p(-np.clip(y, None, 1.0 - _ONE_MINUS_Y_FLOOR))
    out = np.empty(s_array.shape, dtype=np.result_type(s_array.dtype, float)).reshape(-1)
    flat = s_array.reshape(-1)
    for start in range(0, flat.size, chunk):
        block = flat[start : start + chunk][:, None]
        bracket = 1.0 - np.exp(block * log_one_minus_y[None, :])
        out[start : start + chunk] = np.trapezoid(dgamma_dy[None, :] * bracket, y, axis=1)
    return out.reshape(s_array.shape)


def subordinator_density(
    w_grid: np.ndarray,
    ell_km: float,
    y: np.ndarray,
    dgamma_dy: np.ndarray,
    n_k: int = 2**13,
) -> np.ndarray:
    """Density ``P(w)`` of the log-loss driven by PROPOSAL's tabulated loss rate.

    Feeds :func:`phi_from_spectrum` to
    :func:`softpaws.transport.loss_distribution.invert_log_loss_symbol`, so the
    law follows PROPOSAL's differential cross sections rather than any calibrated
    family. This is the reference the closed forms are judged against.

    Parameters
    ----------
    w_grid : np.ndarray
        Uniformly spaced grid of log-loss values ``w = ln(eps / E) >= 0``.
    ell_km : float
        Propagated column depth ``ell`` [km].
    y : np.ndarray
        Grid of fractional energy losses.
    dgamma_dy : np.ndarray
        Differential loss rate [km^-1] on ``y``.
    n_k : int, optional
        Number of nodes on the ``k`` grid of the inversion.

    Returns
    -------
    density : np.ndarray
        ``P(w)``, normalized to unit area over ``w_grid``.
    """
    return invert_log_loss_symbol(
        w_grid, ell_km, lambda s: phi_from_spectrum(s, y, dgamma_dy), n_k
    )


# ---------------------------------------------------------------------------
# PROPOSAL Monte Carlo
# ---------------------------------------------------------------------------


def propagate_muons(
    energy_gev: float,
    ell_km: float,
    n_muons: int,
    ecut_mev: float,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Propagate muons through water and read their energy at a fixed column.

    Uses the same three radiative channels as the rest of the script, with a
    finite energy cut so that losses above ``ecut_mev`` are sampled stochastically
    -- which is the whole point of the comparison. The energy is taken at exactly
    ``ell_km`` via PROPOSAL's ``get_state_for_distance``, so muons that were
    tracked further do not bias the result.

    Parameters
    ----------
    energy_gev : float
        Initial muon energy [GeV].
    ell_km : float
        Column depth at which to read the energy [km].
    n_muons : int
        Number of muons to propagate.
    ecut_mev : float
        Energy cut [MeV] separating stochastic from continuous losses.
    seed : int
        Seed for PROPOSAL's random generator.

    Returns
    -------
    final_energy_gev : np.ndarray, shape (n_stopped_excluded,)
        Muon energy [GeV] at ``ell_km``, for the muons that reached it.
    n_stopped : int
        Number of muons that fell below the tracking threshold before ``ell_km``
        and are excluded from ``final_energy_gev``.
    """
    import proposal as pp

    particle = pp.particle.MuMinusDef()
    medium = pp.medium.Water()
    # v_cut = 1 with a finite ecut: losses above ecut_mev are stochastic, the
    # rest continuous. This is PROPOSAL's standard stochastic configuration.
    cuts = pp.EnergyCutSettings(ecut_mev, 1, False)
    cross_sections = [
        pp.crosssection.make_crosssection(param, particle, medium, cuts, True)
        for param in proposal_parametrizations().values()
    ]

    collection = pp.PropagationUtilityCollection()
    collection.displacement = pp.make_displacement(cross_sections, True)
    collection.interaction = pp.make_interaction(cross_sections, True)
    collection.time = pp.make_time(cross_sections, particle, True)
    utility = pp.PropagationUtility(collection=collection)
    geometry = pp.geometry.Sphere(pp.Cartesian3D(0, 0, 0), 1.0e20)
    density = pp.density_distribution.density_homogeneous(medium.mass_density)
    propagator = pp.Propagator(particle, [(geometry, utility, density)])

    pp.RandomGenerator.get().set_seed(seed)
    ell_cm = ell_km * CM_PER_KM
    # Track well below the cut so that a muon only stops if it truly ranges out.
    min_energy_mev = 10.0 * ecut_mev

    energies = []
    n_stopped = 0
    for _ in range(n_muons):
        state = pp.particle.ParticleState()
        state.type = particle.particle_type
        state.position = pp.Cartesian3D(0, 0, 0)
        state.direction = pp.Cartesian3D(0, 0, 1)
        state.energy = energy_gev * 1.0e3
        state.propagated_distance = 0.0
        track = propagator.propagate(state, max_distance=ell_cm, min_energy=min_energy_mev)
        if track.final_state().propagated_distance < ell_cm * (1.0 - 1.0e-9):
            n_stopped += 1
            continue
        energies.append(track.get_state_for_distance(ell_cm).energy / 1.0e3)
    return np.asarray(energies), n_stopped


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_moment_closure(
    energy_gev: float,
    y: np.ndarray,
    total: np.ndarray,
    b_table: float,
    d_table: float,
    t_table: float,
) -> None:
    """Check the reconstructed ``dGamma/dy`` against PROPOSAL's own moments."""
    b_direct, d_direct = proposal_moments(energy_gev)
    b_quad = float(np.trapezoid(y * total, y))
    d_quad = float(np.trapezoid(y**2 * total, y))
    t_quad = float(np.trapezoid(y**3 * total, y))
    print("\n=== 1a. Moment closure of the reconstructed dGamma/dy ===")
    print(
        f"{'moment':>8} {'quadrature':>12} {'PROPOSAL':>12} {'ratio':>8} "
        f"{'table':>12} {'ratio':>8}"
    )
    for name, quad, direct, table in (
        ("b_mu", b_quad, b_direct, b_table),
        ("d_mu", d_quad, d_direct, d_table),
    ):
        print(
            f"{name:>8} {quad:12.6f} {direct:12.6f} {quad / direct:8.4f} "
            f"{table:12.6f} {table / direct:8.4f}"
        )
    # PROPOSAL has no third-moment accumulator, so t_mu has no independent
    # reference: it comes from this same quadrature, which is exactly why the two
    # rows above are checked first.
    print(f"{'t_mu':>8} {t_quad:12.6f} {'--':>12} {'--':>8} {t_table:12.6f} {'--':>8}")
    print(
        "  quadrature/PROPOSAL validates the dGamma/dy reconstruction; "
        "table/PROPOSAL is the\n  shipped table's interpolation error at this energy. "
        "t_mu is quadrature-only, so\n  the b_mu and d_mu closure is what licenses it."
    )


def print_spectrum_table(
    y: np.ndarray,
    total: np.ndarray,
    two: tuple[float, float],
    three: tuple[float, float, float],
) -> None:
    """Tabulate PROPOSAL's ``dGamma/dy`` against both calibrated families."""
    kappa_2, p_2 = two
    kappa_3, q_3, p_3 = three
    print("\n=== 1b. dGamma/dy: PROPOSAL vs the calibrated families ===")
    print(
        f"{'y':>8} {'PROPOSAL':>12} {'two-moment':>12} {'ratio':>9} "
        f"{'three-mom':>12} {'ratio':>8}"
    )
    for probe in _Y_PROBES:
        reference = float(np.interp(probe, y, total))
        value_2 = kappa_2 * (1.0 - probe) ** p_2 / probe
        value_3 = kappa_3 * probe ** (q_3 - 1.0) * (1.0 - probe) ** p_3
        print(
            f"{probe:8g} {reference:12.4g} {value_2:12.4g} {reference / value_2:9.3f} "
            f"{value_3:12.4g} {reference / value_3:8.3f}"
        )
    print(
        "  Two moments fix y ~ 0.1-0.7, the window that generates them, and leave the\n"
        "  soft pile-up and the hard tail orders of magnitude off. The third moment buys\n"
        "  the hard end: p drops below zero, so dGamma/dy no longer dies as y -> 1."
    )


def print_eigenvalue_table(
    y: np.ndarray,
    total: np.ndarray,
    b_mu: float,
    d_mu: float,
    t_mu: float,
    ell_km: float,
    final_energy_gev: np.ndarray,
    energy_gev: float,
) -> None:
    """Tabulate ``Phi(A)`` from PROPOSAL, the model forms, and the Monte Carlo."""
    a = np.array(_A_PROBES)
    phi_quad = phi_from_spectrum(a, y, total).real
    phi_two = phi_eigenvalue(a, b_mu, d_mu)
    phi_three = phi_eigenvalue_three_moment(a, b_mu, d_mu, t_mu)
    phi_fp = phi_fokker_planck(a, b_mu, d_mu)
    phi_dr = phi_drift(a, b_mu)
    # Phi(A) is minus the log of the A-th moment of the survival ratio, per unit
    # column: E[(E_f / E_i)^A] = exp(-ell Phi(A)).
    ratio = final_energy_gev / energy_gev
    phi_mc = np.array([-np.log(np.mean(ratio**value)) / ell_km for value in a])

    print("\n=== 2. Collision eigenvalue Phi(A) [km^-1] ===")
    print(
        f"{'A':>5} {'quadrature':>11} {'MC':>9} {'two-mom':>9} {'three-mom':>10} "
        f"{'Fokker-Pl':>10} {'drift':>8}   ratios to quadrature"
    )
    for i, value in enumerate(a):
        print(
            f"{value:5g} {phi_quad[i]:11.4f} {phi_mc[i]:9.4f} {phi_two[i]:9.4f} "
            f"{phi_three[i]:10.4f} {phi_fp[i]:10.4f} {phi_dr[i]:8.4f}   "
            f"{phi_mc[i] / phi_quad[i]:5.3f} {phi_two[i] / phi_quad[i]:5.3f} "
            f"{phi_three[i] / phi_quad[i]:5.3f} {phi_fp[i] / phi_quad[i]:5.3f} "
            f"{phi_dr[i] / phi_quad[i]:5.3f}"
        )
    print(
        "  Quadrature and Monte Carlo agree to a few percent -- both are PROPOSAL.\n"
        "  The three-moment form is exact at A = 1, 2, 3 and holds to sub-percent\n"
        "  well past them, where the two-moment form and the truncations drift off."
    )


def print_loss_table(
    w_mc: np.ndarray,
    w_grid: np.ndarray,
    exact_proposal: np.ndarray,
    exact_two_moment: np.ndarray,
    exact_three_moment: np.ndarray,
    ell_km: float,
    b_mu: float,
    d_mu: float,
    energy_gev: float,
    final_energy_gev: np.ndarray,
) -> None:
    """Report the mean survival, then the tail probabilities that break it."""
    fp_mean = (b_mu + d_mu / 2.0) * ell_km
    print(f"\n=== 3a. Mean vs typical, ell = {ell_km:g} km ===")
    print(
        f"  <E_f / E_i>   MC {np.mean(final_energy_gev / energy_gev):.4f}   "
        f"exp(-b_mu ell) {np.exp(-b_mu * ell_km):.4f}   "
        f"ratio {np.mean(final_energy_gev / energy_gev) / np.exp(-b_mu * ell_km):.4f}"
    )
    print(
        f"  mean w        MC {w_mc.mean():.4f}   Fokker-Planck {fp_mean:.4f}   "
        f"ratio {w_mc.mean() / fp_mean:.4f}"
    )
    print(
        f"  median w      MC {np.median(w_mc):.4f}   "
        f"i.e. {np.median(w_mc) / w_mc.mean():.2f} of the mean -- the law is not symmetric"
    )
    print(f"  std dev w     MC {w_mc.std():.4f}   Fokker-Planck {np.sqrt(d_mu * ell_km):.4f}")

    print("\n=== 3b. Tail probability P(W > w) ===")
    print(
        f"{'w':>6} {'eps/E':>7} {'MC':>10} {'PROPOSAL':>10} {'three-mom':>10} "
        f"{'two-mom':>10} {'Gaussian':>10}"
    )
    for probe in _W_PROBES:
        n_above = int((w_mc > probe).sum())
        print(
            f"{probe:6.2f} {np.exp(probe):7.1f} {n_above / w_mc.size:10.2e} "
            f"{float(survival_from_density(probe, w_grid, exact_proposal)):10.2e} "
            f"{float(survival_from_density(probe, w_grid, exact_three_moment)):10.2e} "
            f"{float(survival_from_density(probe, w_grid, exact_two_moment)):10.2e} "
            f"{float(gaussian_survival(probe, ell_km, b_mu, d_mu)):10.2e}"
            f"   (MC n = {n_above})"
        )
    print(
        "  The Gaussian collapses and the two-moment inversion runs out of tail; the\n"
        "  three-moment closed form tracks the MC and the full PROPOSAL inversion, at\n"
        "  the cost of one extra tabulated number."
    )


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_path.with_suffix(suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {path.resolve()}")
    plt.close(fig)


def _panel_spectrum(
    ax: plt.Axes,
    y: np.ndarray,
    spectrum: dict[str, np.ndarray],
    two: tuple[float, float],
    three: tuple[float, float, float],
) -> None:
    """Loss rate per unit ``ln y``, whose area under the curve is ``b_mu``."""
    kappa_2, p_2 = two
    kappa_3, q_3, p_3 = three
    # Total underneath, channels over it: pair production sits on top of the total
    # across the whole soft plateau and would otherwise be hidden by it.
    ax.plot(y, y * spectrum["total"], **_PROPOSAL_STYLE)
    for label, style in _CHANNEL_STYLES.items():
        ax.plot(y, y * spectrum[label], label=label, **style)
    ax.plot(y, kappa_2 * (1.0 - y) ** p_2, **_TWO_MOMENT_STYLE)
    ax.plot(y, kappa_3 * y**q_3 * (1.0 - y) ** p_3, **_THREE_MOMENT_STYLE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.0e-9, 1.0)
    ax.set_ylim(1.0e-3, 1.0e3)
    ax.set_xlabel(r"fractional energy loss  $y$")
    ax.set_ylabel(r"$y\,\mathrm{d}\Gamma/\mathrm{d}y$  [km$^{-1}$]")
    ax.set_title(r"(a) loss spectrum  ($\int \mathrm{d}\ln y = b_\mu$)", pad=_TITLE_PAD)
    ax.legend(fontsize=5.0, loc="lower left")


def _panel_eigenvalue(
    ax: plt.Axes,
    y: np.ndarray,
    total: np.ndarray,
    b_mu: float,
    d_mu: float,
    t_mu: float,
    ell_km: float,
    final_energy_gev: np.ndarray,
    energy_gev: float,
) -> None:
    """Ratio of each ``Phi(A)`` model to the PROPOSAL quadrature."""
    a = np.linspace(0.2, 6.0, 120)
    reference = phi_from_spectrum(a, y, total).real
    ax.axhline(1.0, **_PROPOSAL_STYLE)
    ax.plot(a, phi_eigenvalue(a, b_mu, d_mu) / reference, **_TWO_MOMENT_STYLE)
    ax.plot(
        a, phi_eigenvalue_three_moment(a, b_mu, d_mu, t_mu) / reference, **_THREE_MOMENT_STYLE
    )
    ax.plot(a, phi_fokker_planck(a, b_mu, d_mu) / reference, **_FP_STYLE)
    ax.plot(a, phi_drift(a, b_mu) / reference, **_DRIFT_STYLE)

    a_mc = np.array(_A_PROBES)
    ratio = final_energy_gev / energy_gev
    phi_mc = np.array([-np.log(np.mean(ratio**value)) / ell_km for value in a_mc])
    ax.plot(
        a_mc,
        phi_mc / phi_from_spectrum(a_mc, y, total).real,
        "o",
        ms=2.5,
        color="black",
        mfc="none",
        label="PROPOSAL MC",
    )
    ax.axvline(1.0, color="grey", lw=0.6, ls=(0, (1, 3)))
    ax.set_xlabel(r"spectral index  $A = \gamma - \lambda - 1$")
    ax.set_ylabel(r"$\Phi(A)\;/\;\Phi_{\rm PROPOSAL}(A)$")
    ax.set_title(r"(b) eigenvalue, relative to PROPOSAL", pad=_TITLE_PAD)
    ax.set_ylim(0.6, 1.6)
    ax.legend(fontsize=5.0, loc="upper left")


def _panel_loss_law(
    ax: plt.Axes,
    w_mc: np.ndarray,
    w_grid: np.ndarray,
    exact_proposal: np.ndarray,
    exact_two_moment: np.ndarray,
    exact_three_moment: np.ndarray,
    ell_km: float,
    b_mu: float,
    d_mu: float,
) -> None:
    """Tail probability ``P(W > w)``, Monte Carlo against the analytic laws.

    The survival function rather than the density: it is what the single-event
    inference actually uses, it needs no binning, and it makes the orders of
    magnitude between the laws directly readable.
    """
    w_max = 4.0
    ordered = np.sort(w_mc)
    survival_mc = 1.0 - np.arange(ordered.size) / ordered.size
    ax.step(ordered, survival_mc, where="post", color="black", lw=0.7, alpha=0.5)
    # A marker-only proxy keeps the legend readable next to the smooth curves.
    ax.plot([], [], color="black", lw=0.7, alpha=0.5, label="PROPOSAL MC")
    ax.plot(w_grid, survival_from_density(w_grid, w_grid, exact_proposal), **_PROPOSAL_STYLE)
    ax.plot(
        w_grid, survival_from_density(w_grid, w_grid, exact_three_moment), **_THREE_MOMENT_STYLE
    )
    ax.plot(w_grid, survival_from_density(w_grid, w_grid, exact_two_moment), **_TWO_MOMENT_STYLE)
    ax.plot(w_grid, gaussian_survival(w_grid, ell_km, b_mu, d_mu), **_FP_STYLE)
    ax.axvline(
        (b_mu + d_mu / 2.0) * ell_km, color="grey", lw=0.6, ls=(0, (1, 3)), label=r"mean $w$ (FP)"
    )
    ax.set_yscale("log")
    ax.set_ylim(1.0e-4, 1.5)
    ax.set_xlim(0.0, w_max)
    ax.set_xlabel(r"log-loss  $w = \ln(\varepsilon / E)$")
    ax.set_ylabel(r"tail probability  $P(W > w)$")
    ax.set_title("(c) log-loss law")
    secondary = ax.secondary_xaxis(
        "top",
        functions=(
            lambda w: np.exp(np.clip(w, -50.0, 50.0)),
            lambda ratio: np.log(np.clip(ratio, 1.0e-9, None)),
        ),
    )
    secondary.set_xscale("log")
    secondary.set_xticks([1, 3, 10, 30])
    secondary.set_xlabel(r"$\varepsilon / E$", labelpad=1.5)
    ax.legend(fontsize=5.0, loc="lower left")


def make_figure(
    out_path: pathlib.Path,
    y: np.ndarray,
    spectrum: dict[str, np.ndarray],
    two: tuple[float, float],
    three: tuple[float, float, float],
    b_mu: float,
    d_mu: float,
    t_mu: float,
    ell_km: float,
    energy_gev: float,
    final_energy_gev: np.ndarray,
    w_mc: np.ndarray,
    w_grid: np.ndarray,
    exact_proposal: np.ndarray,
    exact_two_moment: np.ndarray,
    exact_three_moment: np.ndarray,
) -> None:
    """Three panels: loss spectrum, eigenvalue ratio, log-loss law."""
    with plt.style.context(str(_STYLE)):
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1))
        _panel_spectrum(axes[0], y, spectrum, two, three)
        _panel_eigenvalue(
            axes[1], y, spectrum["total"], b_mu, d_mu, t_mu, ell_km, final_energy_gev, energy_gev
        )
        _panel_loss_law(
            axes[2],
            w_mc,
            w_grid,
            exact_proposal,
            exact_two_moment,
            exact_three_moment,
            ell_km,
            b_mu,
            d_mu,
        )
        fig.suptitle(
            rf"muon in water, $E_\mu = 10^{{{np.log10(energy_gev):.0f}}}$ GeV, "
            rf"$\ell = {ell_km:g}$ km",
            fontsize=8.0,
        )
        fig.tight_layout()
        _save(fig, out_path)


def main() -> None:
    args = parse_args()
    energy_gev = args.energy_gev

    b_mu = float(drift_coefficient(energy_gev)[0])
    d_mu = float(diffusion_coefficient(energy_gev)[0])
    t_mu = float(third_moment_coefficient(energy_gev)[0])
    two = tuple(float(v) for v in two_moment_loss_spectrum(b_mu, d_mu))
    three = tuple(float(v) for v in three_moment_loss_spectrum(b_mu, d_mu, t_mu))

    print(f"=== softpaws vs PROPOSAL, muon in water at E = {energy_gev:.3g} GeV ===")
    print(
        f"  shipped table: b_mu = {b_mu:.5f}, d_mu = {d_mu:.5f}, t_mu = {t_mu:.5f} km^-1"
    )
    print(f"  two-moment family:   kappa = {two[0]:.5f} km^-1, p = {two[1]:.5f}")
    print(
        f"  three-moment family: kappa = {three[0]:.5f} km^-1, q = {three[1]:.5f}, "
        f"p = {three[2]:.5f}"
    )

    y = loss_spectrum_y_grid()
    spectrum = proposal_loss_spectrum(energy_gev, y)
    print_moment_closure(energy_gev, y, spectrum["total"], b_mu, d_mu, t_mu)
    print_spectrum_table(y, spectrum["total"], two, three)

    start = time.perf_counter()
    final_energy_gev, n_stopped = propagate_muons(
        energy_gev, args.ell_km, args.n_muons, args.ecut_mev, args.seed
    )
    print(
        f"\nPropagated {args.n_muons} muons over {args.ell_km:g} km "
        f"in {time.perf_counter() - start:.1f} s ({n_stopped} ranged out and were dropped)."
    )
    print(
        f"  b_mu at the median final energy ({np.median(final_energy_gev):.3g} GeV) is "
        f"{float(drift_coefficient(np.median(final_energy_gev))[0]):.5f} km^-1, "
        "the scale breaking the\n  analytic laws neglect."
    )

    print_eigenvalue_table(
        y, spectrum["total"], b_mu, d_mu, t_mu, args.ell_km, final_energy_gev, energy_gev
    )

    w_mc = np.log(energy_gev / final_energy_gev)
    w_grid = np.linspace(1.0e-3, max(4.0, 1.05 * w_mc.max()), 1500)
    exact_proposal = subordinator_density(w_grid, args.ell_km, y, spectrum["total"])
    exact_two_moment = loss_density(w_grid, args.ell_km, b_mu, d_mu)
    exact_three_moment = loss_density_three_moment(w_grid, args.ell_km, b_mu, d_mu, t_mu)

    print_loss_table(
        w_mc,
        w_grid,
        exact_proposal,
        exact_two_moment,
        exact_three_moment,
        args.ell_km,
        b_mu,
        d_mu,
        energy_gev,
        final_energy_gev,
    )

    make_figure(
        args.out_dir / "27_proposal_cross_section_and_loss",
        y,
        spectrum,
        two,
        three,
        b_mu,
        d_mu,
        t_mu,
        args.ell_km,
        energy_gev,
        final_energy_gev,
        w_mc,
        w_grid,
        exact_proposal,
        exact_two_moment,
        exact_three_moment,
    )


if __name__ == "__main__":
    main()
