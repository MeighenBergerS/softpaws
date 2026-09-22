"""Example 37 -- how many loss moments does the transport exponent need?

Example 27 benchmarks the calibrated loss families against PROPOSAL and stops at
three moments. Its panel (a) leaves an obvious loose end: even with ``t_mu``
matched, the reconstructed ``dGamma/dy`` still differs from PROPOSAL's by tens of
percent across most of the ``y`` range, while panels (b) and (c) look converged.
Either the agreement in (b) and (c) is luck, or the shape in (a) does not matter.
This example settles which, by pushing the calibration to ``N = 4, 5, 6, 10``
moments and watching all three quantities at once.

**The N-moment family.** The two- and three-moment spectra of
:mod:`softpaws.transport.eigenvalue` are the first two members of

.. math:: \\frac{d\\Gamma}{dy} = y^{q-1} (1-y)^p \\sum_{j=0}^{N-1} a_j y^j,

with the base exponents ``(q, p)`` fixed once by the three-moment calibration and
the ``a_j`` solved from the moments. Every moment is a Beta function,
``<y^n> = sum_j a_j B(n + q + j, p + 1)``, so matching ``mu_1 ... mu_N`` is a
linear system, and the exponent stays closed form,

.. math:: \\Phi(s) = \\sum_j a_j \\bigl[ B(q+j,\\,p+1) - B(q+j,\\,p+1+s) \\bigr].

The claim that the closed form survives extra moments is therefore not special to
``N = 3``. It survives to any order, as a finite sum of Beta functions.

**What the example finds.** The three panels disagree with each other, and that
is the result.

1. **The exponent converges geometrically.** Against a direct quadrature of
   PROPOSAL's own ``dGamma/dy``, the drift-diffusion pair is 16% low at
   ``A = 8``; three moments bring that to 0.52%, four to 0.032%, and each
   further moment gains roughly another factor of five, until the comparison
   hits the 0.001% floor of the reference quadrature itself.
2. **The log-loss law converges with it.** The deep tail ``P(W > 2.5)``, which
   the two-moment family gets wrong by a factor of 1.9, is within 2% from three
   moments on.
3. **The shape never converges.** The reconstructed ``dGamma/dy`` sits ~40% low
   near ``y ~ 4 x 10^-3`` for *every* ``N >= 3``, and the extra freedom spends
   itself on oscillations above ``y ~ 10^-2`` that grow with ``N`` rather than
   shrink. Adding moments moves the shape around; it does not move it closer.

**Why that is consistent.** The moments and the exponent are blind to the same
places. Both weight the spectrum by something that vanishes at the soft end --
``y^n`` for the moments, and ``[1 - (1-y)^s] -> s y`` for ``Phi`` -- so the
region below ``y ~ 10^-3`` carries 96% of the collisions but only 8% of ``mu_1``
and 13% of ``Phi(8)``. It is precisely the region no moment can constrain, and
precisely the region ``Phi`` does not ask about. The shape error that survives
lives where neither quantity has support, which is why it is permanent and why it
is harmless.

**The cost of asking for more.** The moment matrix is a Beta-function Hankel
matrix, and reconstructing a density from its moments on ``(0, 1)`` is the
classical ill-posed Hausdorff problem. The condition number grows by about three
decades per added moment, from ``10^4`` at ``N = 4`` to ``10^13`` at ``N = 10``,
and the solved coefficients cancel to six significant figures by ``N = 10``. The
linear solve is therefore done in extended precision with ``mpmath``. The
practical reading is the useful one: the exponent is well posed where the
cross-section shape is not.

Three moments is the right place to stop, and not because the series stalls. It
does not. It is because 0.5% on ``Phi`` is already an order of magnitude below
the ~15% that scale invariance itself is violated by between 1 and 100 PeV
(Section~II of the draft), so a fourth moment buys accuracy the rest of the model
cannot use.

The reference here is the subordinator inversion driven by PROPOSAL's tabulated
``dGamma/dy``, not a Monte-Carlo propagation. Example 27 already validates that
inversion against the Monte Carlo; repeating it would cost minutes and change
nothing.

Requires the optional ``proposal`` dependency (``pip install -e ".[transport]"``)
and ``mpmath``.

Usage
-----
    python scripts/2026_muon_transport/37_moment_convergence.py
    python scripts/2026_muon_transport/37_moment_convergence.py --energy-gev 1e8 --ell-km 5
    python scripts/2026_muon_transport/37_moment_convergence.py --n-moments 2 3 4 8
    python scripts/2026_muon_transport/37_moment_convergence.py --exact-family two
    python scripts/2026_muon_transport/37_moment_convergence.py --out-dir /path/to/figures

Writes two figures. ``37_moment_convergence`` is the three-panel convergence
study above. ``37_exponent_vs_proposal`` is the standalone square figure that
belongs beside the one of example 36: the same exponent and truncations, with
PROPOSAL's tabulated loss spectrum integrated directly and drawn as a black
dotted line. Read together with the shape panel, the pair carries the two claims
the draft needs -- that the exponent and the log-loss tail are reproduced, and
that the fractional-loss spectrum behind them is not, because the integral does
not depend on its shape.

The closed forms carry a pole at ``A = -(p + 1)``, the negative-index
convergence edge. For the three-moment calibration ``p ~ -0.18`` puts it at
``A ~ -0.82``, inside the plotted range, so the curve is drawn only where it
converges and the edge is marked. The two-moment family has ``p ~ +2.1`` and its
edge sits at ``-3.1``, off the plot, which is why the figure of example 36 shows
no such feature.
"""

import argparse
import pathlib
from functools import partial

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

from softpaws.transport.coefficients import loss_spectrum_y_grid, proposal_loss_spectrum
from softpaws.transport.eigenvalue import (
    n_moment_coefficients,
    n_moment_loss_spectrum,
    phi_drift,
    phi_eigenvalue,
    phi_eigenvalue_three_moment,
    phi_fokker_planck,
    phi_symbol,
    phi_symbol_n_moment,
    phi_symbol_three_moment,
    three_moment_loss_spectrum,
    two_moment_loss_spectrum,
)
from softpaws.transport.loss_distribution import invert_log_loss_symbol

_HERE = pathlib.Path(__file__).resolve().parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"

# Keeps (1 - y)^s finite on the hard edge of the grid, as in example 27.
_ONE_MINUS_Y_FLOOR = 1.0e-12

# Spectral indices the exponent is compared at. A = 0.98 is the IceCube diffuse
# value of Eq. (13) of the draft; the rest bracket it.
_A_GRID = np.array([0.5, 0.98, 2.0, 3.0, 5.0, 8.0])

# Log-loss thresholds for the tail table. W > 2.5 is roughly the regime a single
# ultra-high-energy event samples (docs/first_passage_range.md).
_W_THRESHOLDS = (1.0, 1.5, 2.5)

# y values the shape table is printed at.
_Y_PROBE = np.array([1.0e-3, 1.0e-2, 0.1, 0.5, 0.9, 0.99])

# Range of the standalone exponent figure, matching example 36 so the two can be
# read side by side.
_FIG_A_MIN, _FIG_A_MAX = -1.2, 11.0
_A_ICECUBE = 0.98  # gamma = 2.38, lambda = 0.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--energy-gev",
        type=float,
        default=1.0e6,
        help="Muon energy at which the loss spectrum is evaluated [GeV].",
    )
    parser.add_argument(
        "--ell-km",
        type=float,
        default=3.0,
        help="Propagated column depth for the log-loss law [km water equivalent].",
    )
    parser.add_argument(
        "--n-moments",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 6, 10],
        help="Numbers of moments to calibrate. Values below 2 are dropped.",
    )
    parser.add_argument(
        "--dps",
        type=int,
        default=80,
        help="mpmath working precision for the moment solve [decimal digits].",
    )
    parser.add_argument(
        "--exact-family",
        choices=("two", "three"),
        default="three",
        help="Closed form drawn as the exact exponent in the standalone figure. "
        "'three' is the Table F.1 calibration and sits within 1%% of PROPOSAL "
        "over the whole range; 'two' is the one-digamma form of the main text, "
        "which is 16%% high at A = 8.",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=_DEFAULT_OUT_DIR,
        help="Directory for the output figures.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Moments and the N-moment closed form
# ---------------------------------------------------------------------------


def spectrum_moments(y: np.ndarray, dgamma_dy: np.ndarray, n_max: int) -> np.ndarray:
    """Moments ``mu_n = <y^n>`` of a tabulated loss rate.

    Parameters
    ----------
    y : np.ndarray
        Ascending grid of fractional energy losses in ``(0, 1)``.
    dgamma_dy : np.ndarray
        Differential loss rate [km^-1] on ``y``.
    n_max : int
        Highest moment order to return.

    Returns
    -------
    mu : np.ndarray
        Moments ``mu_1 ... mu_n_max`` [km^-1]. ``mu_1`` and ``mu_2`` are the
        drift and diffusion coefficients ``b_mu`` and ``d_mu``.
    """
    return np.array([np.trapezoid(dgamma_dy * y**n, y) for n in range(1, n_max + 1)])


def moment_matrix_condition(n: int, q: float, p: float) -> float:
    """Double-precision condition number of the ``N``-moment Beta matrix.

    Reported to show why the solve of
    :func:`~softpaws.transport.eigenvalue.n_moment_coefficients` needs
    extended precision, and why large ``N`` is not merely useless but hostile.

    Parameters
    ----------
    n : int
        Number of moments matched.
    q, p : float
        Base-weight exponents, as in
        :func:`~softpaws.transport.eigenvalue.n_moment_coefficients`.

    Returns
    -------
    cond : float
        Two-norm condition number.
    """
    matrix = np.array(
        [[float(mp.beta(mp.mpf(i + 1) + q + j, p + 1.0)) for j in range(n)] for i in range(n)]
    )
    return float(np.linalg.cond(matrix))


def phi_from_spectrum(
    s: complex | np.ndarray, y: np.ndarray, dgamma_dy: np.ndarray, chunk: int = 64
) -> np.ndarray:
    """Reference ``Phi(s)`` by quadrature of a tabulated loss rate.

    Same integral as :func:`~softpaws.transport.eigenvalue.phi_symbol_n_moment`,
    evaluated numerically on the supplied
    grid instead of in closed form. This is the reference every calibrated family
    is judged against. Mirrors the helper of example 27.

    Parameters
    ----------
    s : complex or np.ndarray
        Mellin variable.
    y : np.ndarray
        Ascending grid of fractional energy losses in ``(0, 1)``.
    dgamma_dy : np.ndarray
        Differential loss rate [km^-1] on ``y``.
    chunk : int, optional
        Number of ``s`` values evaluated per block, bounding peak memory.

    Returns
    -------
    phi : np.ndarray
        ``Phi(s)`` [km^-1], with the shape of ``s``.
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


# ---------------------------------------------------------------------------
# Model assembly
# ---------------------------------------------------------------------------


def build_models(mu: np.ndarray, n_moments: list[int], dps: int) -> dict[int, dict]:
    """Calibrate every requested family and package its three evaluators.

    ``N = 2`` and ``N = 3`` use the library closed forms
    (:func:`~softpaws.transport.eigenvalue.two_moment_loss_spectrum` and
    :func:`~softpaws.transport.eigenvalue.three_moment_loss_spectrum`), which are
    the families the draft actually quotes. ``N >= 4`` uses the polynomial
    extension built on the three-moment base exponents, of which the three-moment
    family is the ``a = (kappa, 0, 0)`` member.

    Parameters
    ----------
    mu : np.ndarray
        Moments ``mu_1 ...`` [km^-1] of the reference spectrum.
    n_moments : list of int
        Numbers of moments to calibrate.
    dps : int
        mpmath working precision [decimal digits].

    Returns
    -------
    models : dict
        Maps ``N`` to a dict with keys ``spectrum`` (callable of ``y``), ``phi``
        (callable of ``s``), ``cond`` (float or NaN), and ``cancellation`` (the
        coefficient cancellation ratio, or NaN).
    """
    kappa2, p2 = two_moment_loss_spectrum(mu[0], mu[1])
    kappa3, q3, p3 = three_moment_loss_spectrum(mu[0], mu[1], mu[2])
    q_base, p_base = float(q3), float(p3)

    models: dict[int, dict] = {}
    for n in sorted(n_moments):
        if n < 2:
            continue
        if n == 2:
            # dGamma/dy = kappa (1-y)^p / y is the q -> 0 member of the family,
            # where both Beta functions of phi_symbol_n_moment diverge and only their
            # difference is finite. The library carries the digamma closed form.
            kappa, p = float(kappa2), float(p2)
            models[n] = {
                "spectrum": lambda y, kappa=kappa, p=p: kappa * (1.0 - y) ** p / y,
                "phi": lambda s, kappa=kappa, p=p: phi_symbol(s, kappa, p),
                "cond": float("nan"),
                "cancellation": float("nan"),
            }
            continue
        if n == 3:
            kappa, q, p = float(kappa3), q_base, p_base
            models[n] = {
                "spectrum": lambda y, kappa=kappa, q=q, p=p: kappa
                * y ** (q - 1.0)
                * (1.0 - y) ** p,
                "phi": lambda s, kappa=kappa, q=q, p=p: phi_symbol_three_moment(
                    s, kappa, q, p
                ),
                "cond": float("nan"),
                "cancellation": float("nan"),
                "coefficients": np.array([kappa, 0.0, 0.0]),
            }
            continue
        a = n_moment_coefficients(mu[:n], q_base, p_base, dps=dps)
        models[n] = {
            "spectrum": lambda y, a=a: n_moment_loss_spectrum(a, q_base, p_base, y),
            "phi": lambda s, a=a: phi_symbol_n_moment(s, a, q_base, p_base),
            "cond": moment_matrix_condition(n, q_base, p_base) if n > 3 else float("nan"),
            "cancellation": float(np.abs(a).sum() / abs(a.sum())),
            "coefficients": a,
        }
    return models, (q_base, p_base)


def survival(density: np.ndarray, w_grid: np.ndarray, threshold: float) -> float:
    """Tail probability ``P(W > threshold)`` of a log-loss density.

    Parameters
    ----------
    density : np.ndarray
        ``P(w)`` on ``w_grid``, normalized to unit area.
    w_grid : np.ndarray
        Ascending grid of log-loss values ``w = ln(eps / E) >= 0``.
    threshold : float
        Log-loss threshold.

    Returns
    -------
    probability : float
        Integrated tail above ``threshold``.
    """
    mask = w_grid >= threshold
    return float(np.trapezoid(density[mask], w_grid[mask]))


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def print_moment_closure(mu: np.ndarray, models: dict, y: np.ndarray) -> None:
    """Check that each calibrated family reproduces the moments it was fitted to."""
    print("\nMoment closure: <y^n>_model / <y^n>_PROPOSAL")
    orders = range(1, min(len(mu), 6) + 1)
    print("  N  " + "".join(f"   n={n}" for n in orders) + "     cond      cancel")
    for n, model in models.items():
        spectrum = model["spectrum"](y)
        ratios = [np.trapezoid(spectrum * y**k, y) / mu[k - 1] for k in orders]
        cond = model["cond"]
        cancel = model["cancellation"]
        cond_s = "       -" if not np.isfinite(cond) else f"{cond:8.1e}"
        cancel_s = "       -" if not np.isfinite(cancel) else f"{cancel:8.1e}"
        print(f"  {n:<2} " + "".join(f" {r:6.3f}" for r in ratios) + f"  {cond_s}  {cancel_s}")


def print_reference_floor(energy_gev: float) -> np.ndarray:
    """Grid error of the reference quadrature, below which the tables mean nothing.

    Every error quoted against ``Phi_exact`` is limited by how well the default
    ``y`` grid resolves PROPOSAL's own spectrum. Refining the grid by a factor of
    eight measures that floor directly.

    Parameters
    ----------
    energy_gev : float
        Muon energy [GeV].

    Returns
    -------
    floor : np.ndarray
        Relative change [%] of ``Phi`` on :data:`_A_GRID` between the default
        grid and an eight-times finer one.
    """
    coarse_y = loss_spectrum_y_grid()
    fine_y = loss_spectrum_y_grid(24000, 16000)
    out = []
    for grid in (coarse_y, fine_y):
        spectrum = proposal_loss_spectrum(energy_gev, grid)["total"]
        out.append(phi_from_spectrum(_A_GRID, grid, spectrum))
    floor = np.abs(out[0] / out[1] - 1.0) * 100.0
    print("\nReference floor: |Phi_exact(default grid) / Phi_exact(8x finer) - 1| [%]")
    print("     " + "".join(f" {v:7.4f}" for v in floor))
    print("     errors below this in the next table are not meaningful.")
    return floor


def print_eigenvalue_table(models: dict, phi_reference: np.ndarray) -> None:
    """Relative error of ``Phi(A)`` against the PROPOSAL quadrature."""
    print("\nTransport exponent: |Phi_N(A) / Phi_exact(A) - 1| [%]")
    print("  N  " + "".join(f"  A={a:<5.2f}" for a in _A_GRID))
    print(
        "  --  "
        + "".join(f" {v:7.4f}" for v in phi_reference)
        + "   <- Phi_exact [1/km]"
    )
    for n, model in models.items():
        phi = np.asarray(model["phi"](_A_GRID), dtype=float)
        error = np.abs(phi / phi_reference - 1.0) * 100.0
        print(f"  {n:<2} " + "".join(f" {v:7.3f}" for v in error))


def print_shape_table(models: dict, y: np.ndarray, dgamma_dy: np.ndarray) -> None:
    """Pointwise ratio of the reconstructed loss spectrum to PROPOSAL's."""
    print("\nShape: (dGamma/dy)_N / PROPOSAL")
    reference = np.interp(_Y_PROBE, y, dgamma_dy)
    print("  N  " + "".join(f"  y={v:<7.0e}" for v in _Y_PROBE))
    for n, model in models.items():
        ratio = model["spectrum"](_Y_PROBE) / reference
        print(f"  {n:<2} " + "".join(f" {v:9.3f}" for v in ratio))


def print_tail_table(tails: dict, reference_tail: list[float]) -> None:
    """Log-loss tail probabilities against the PROPOSAL-driven inversion."""
    print("\nLog-loss tail: P(W > w), and the ratio to PROPOSAL")
    header = "".join(f"   P(W>{w:.1f})    x ref" for w in _W_THRESHOLDS)
    print("  N  " + header)
    print(
        "  --  "
        + "".join(f" {v:10.3e}        -" for v in reference_tail)
        + "   <- PROPOSAL"
    )
    for n, values in tails.items():
        cells = "".join(
            f" {v:10.3e} {v / r:8.3f}" for v, r in zip(values, reference_tail, strict=True)
        )
        print(f"  {n:<2} " + cells)


def print_support_table(y: np.ndarray, dgamma_dy: np.ndarray) -> None:
    """Where the collision rate, the first moment, and ``Phi`` actually live.

    The argument for why the shape error is harmless: the moments and ``Phi``
    suppress the soft end that carries almost all of the collisions.
    """
    print("\nSupport: fraction of each integral contributed per decade in y")
    log_one_minus_y = np.log1p(-np.clip(y, None, 1.0 - _ONE_MINUS_Y_FLOOR))
    mu_1 = np.trapezoid(dgamma_dy * y, y)
    phi_8 = np.trapezoid(dgamma_dy * (1.0 - np.exp(8.0 * log_one_minus_y)), y)
    rate_total = np.trapezoid(dgamma_dy, y)
    phi_integrand = dgamma_dy * (1.0 - np.exp(8.0 * log_one_minus_y))
    edges = [0.0, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 0.5, 0.9, 1.0]
    print("       y range      mu_1    Phi(8)      rate")
    for low, high in zip(edges, edges[1:], strict=False):
        mask = (y >= low) & (y < high)
        if mask.sum() < 2:
            continue
        print(
            f"  {low:7.0e}-{high:<7.0e} "
            f"{np.trapezoid((dgamma_dy * y)[mask], y[mask]) / mu_1:8.4f} "
            f"{np.trapezoid(phi_integrand[mask], y[mask]) / phi_8:9.4f} "
            f"{np.trapezoid(dgamma_dy[mask], y[mask]) / rate_total:9.4f}"
        )
    print(
        f"\n  Gamma_tot = {rate_total:.4g} /km over this grid and still growing as the\n"
        f"  grid extends to smaller y, while Phi(1) = mu_1 = {mu_1:.4g} /km is finite."
    )


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_path.with_suffix(suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  wrote {path}")


def _panel_shape(ax, models: dict, y: np.ndarray, dgamma_dy: np.ndarray) -> None:
    """Panel (a): reconstructed spectrum over PROPOSAL's, versus ``y``."""
    ax.axhline(1.0, color="0.35", lw=0.8, zorder=1)
    good = dgamma_dy > 0.0
    for n, model in models.items():
        ax.plot(
            y[good], model["spectrum"](y[good]) / dgamma_dy[good], lw=1.2, label=f"$N={n}$"
        )
    ax.set_xscale("log")
    ax.set_xlim(1.0e-5, 1.0)
    ax.set_ylim(0.0, 2.5)
    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"$(\mathrm{d}\Gamma/\mathrm{d}y)_N\,/\,\mathrm{PROPOSAL}$")
    ax.set_title(r"(a) shape", fontsize=8)
    ax.legend(ncol=2, fontsize=8)


def _panel_eigenvalue(ax, models: dict, a_grid: np.ndarray, phi_reference: np.ndarray) -> None:
    """Panel (b): relative error of the exponent, versus spectral index."""
    for n, model in models.items():
        phi = np.asarray(model["phi"](a_grid), dtype=float)
        ax.plot(a_grid, np.abs(phi / phi_reference - 1.0) * 100.0, lw=1.2, label=f"$N={n}$")
    ax.axvline(0.98, color="0.35", lw=0.8, ls=":", zorder=1)
    ax.set_yscale("log")
    ax.set_xlim(a_grid[0], a_grid[-1])
    ax.set_ylim(1.0e-3, 1.0e2)
    ax.set_xlabel(r"$A$")
    ax.set_ylabel(r"$|\Phi_N(A)/\Phi(A) - 1|$ [\%]")
    ax.set_title(r"(b) transport exponent", fontsize=8)


def _panel_loss_law(ax, densities: dict, reference: np.ndarray, w_grid: np.ndarray) -> None:
    """Panel (c): log-loss survival function against the PROPOSAL inversion."""
    ax.plot(
        w_grid,
        1.0 - np.concatenate([[0.0], np.cumsum(np.diff(w_grid) * reference[:-1])]),
        color="0.2",
        lw=1.6,
        label="PROPOSAL",
    )
    for n, density in densities.items():
        tail = 1.0 - np.concatenate([[0.0], np.cumsum(np.diff(w_grid) * density[:-1])])
        ax.plot(w_grid, tail, lw=1.1, ls="--", label=f"$N={n}$")
    ax.set_yscale("log")
    ax.set_xlim(0.0, 4.0)
    ax.set_ylim(1.0e-3, 1.2)
    ax.set_xlabel(r"$w = \ln(\varepsilon/E)$")
    ax.set_ylabel(r"$P(W > w)$")
    ax.set_title(r"(c) log-loss law", fontsize=8)
    ax.legend(ncol=2, fontsize=8)


def _label_along_curve(ax, curve, a, text, color, offset_points=5.0) -> None:
    """Write a label along a curve, rotated to its local slope.

    The rotation is read from the display transform, so the axis limits and the
    box aspect must both be final before this is called.

    Parameters
    ----------
    ax : plt.Axes
        Axes holding the curve.
    curve : callable
        The curve, as a function of the spectral index ``A``.
    a : float
        Spectral index at which to place the label.
    text : str
        Label text.
    color : str
        Label color, matched to the curve it names.
    offset_points : float, optional
        Perpendicular offset in points. Positive sits above the line, negative
        below it.
    """
    step = 0.02 * (_FIG_A_MAX - _FIG_A_MIN)
    ends = np.array([a - step, a + step])
    (x0, y0), (x1, y1) = ax.transData.transform(
        np.column_stack([ends, np.asarray(curve(ends), dtype=float)])
    )
    ax.annotate(
        text,
        xy=(a, float(np.asarray(curve(np.array([a])), dtype=float)[0])),
        xytext=(0.0, offset_points),
        textcoords="offset points",
        color=color,
        rotation=np.degrees(np.arctan2(y1 - y0, x1 - x0)),
        rotation_mode="anchor",
        ha="center",
        va="bottom" if offset_points >= 0.0 else "top",
    )


def figure_exponent_vs_proposal(
    y: np.ndarray,
    dgamma_dy: np.ndarray,
    mu: np.ndarray,
    family: str,
    out_dir: pathlib.Path,
) -> None:
    """Standalone square figure: the exponent and its truncations against PROPOSAL.

    The companion to the figure of example 36. That one shows the closed form
    against its own truncations; this one adds PROPOSAL's tabulated loss spectrum,
    integrated directly, so the reader can see that the exponent is reproduced
    even though the loss spectrum behind it is not.

    Parameters
    ----------
    y : np.ndarray
        Grid of fractional energy losses.
    dgamma_dy : np.ndarray
        PROPOSAL's differential loss rate [km^-1] on ``y``.
    mu : np.ndarray
        Moments ``mu_1, mu_2, mu_3`` [km^-1] of that spectrum.
    family : {"two", "three"}
        Which closed form to draw as the exact exponent.
    out_dir : pathlib.Path
        Directory for the figure.
    """
    b_mu, d_mu, t_mu = (float(v) for v in mu[:3])
    a_grid = np.linspace(_FIG_A_MIN, _FIG_A_MAX, 600)
    a_die = 1.0 + 2.0 * b_mu / d_mu

    # The closed forms carry a pole at A = -(p + 1), where the hard edge of the
    # calibrated spectrum stops being integrable against (1-y)^A. That is the
    # negative-index convergence edge of the draft's appendix. For the two-moment
    # family p ~ +2.1 puts it at -3.1, off the plot; for the three-moment family
    # p ~ -0.18 puts it at -0.82, inside it, so the curve is drawn only where it
    # converges.
    if family == "two":
        _, p_edge = two_moment_loss_spectrum(b_mu, d_mu)
        exact = partial(phi_eigenvalue, b_mu=b_mu, d_mu=d_mu)
    else:
        _, _, p_edge = three_moment_loss_spectrum(b_mu, d_mu, t_mu)
        exact = partial(phi_eigenvalue_three_moment, b_mu=b_mu, d_mu=d_mu, t_mu=t_mu)
    a_edge = -(float(np.ravel(p_edge)[0]) + 1.0)
    exact_label = r"closed form $\Phi(A)$"
    a_exact = a_grid[a_grid > a_edge + 0.04]
    drift = partial(phi_drift, b_mu=b_mu)
    second = partial(phi_fokker_planck, b_mu=b_mu, d_mu=d_mu)
    proposal = partial(phi_from_spectrum, y=y, dgamma_dy=dgamma_dy)

    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        c_exact, c_drift, c_second = colors[0], colors[2], colors[1]

        # Where the exponent changes sign the mode grows instead of attenuating.
        ax.axhspan(-1.0, 0.0, color="0.92", lw=0, zorder=0)
        ax.axhline(0.0, color="0.4", lw=0.6, zorder=1)

        ax.plot(a_grid, drift(a_grid), color=c_drift, ls="--")
        ax.plot(a_grid, second(a_grid), color=c_second, ls="-.")
        ax.plot(a_exact, exact(a_exact), color=c_exact)
        ax.plot(a_grid, proposal(a_grid), color="black", ls=":", lw=1.4, zorder=6)

        # The binomial series terminates at integer A, so the truncations touch
        # the exact curve there rather than merely approaching it.
        for a in (1.0, 2.0):
            ax.plot(
                a,
                float(np.asarray(exact(np.array([a])), dtype=float)[0]),
                "o",
                ms=3.5,
                mfc="white",
                mec=c_exact,
                mew=1.0,
                zorder=7,
            )

        ax.axvline(_A_ICECUBE, color="0.55", lw=0.6, ls=":")
        ax.text(
            _A_ICECUBE - 0.22,
            1.90,
            rf"IceCube, $A={_A_ICECUBE:g}$",
            color="0.35",
            rotation=90,
            va="top",
            ha="center",
        )
        ax.plot(a_die, 0.0, "v", ms=4, color=c_second, zorder=5)
        ax.annotate(r"$\Phi<0$: growing mode", xy=(2.0, -0.06), color="0.35", va="top")

        ax.set_xlim(_FIG_A_MIN, _FIG_A_MAX)
        ax.set_ylim(-1.0, 2.0)
        ax.set_box_aspect(1.0)
        ax.set_xlabel(r"Effective spectral index $A=\gamma-\lambda-1$")
        ax.set_ylabel(r"$\Phi(A)$ [km$^{-1}$]")

        # Curves carry their own names along their own slopes, as in example 36.
        # The limits and the box aspect must be final first, since the rotations
        # are read from the display transform.
        _label_along_curve(ax, drift, 2.5, r"$A\,b_\mu$", c_drift)
        _label_along_curve(ax, exact, 5.2, exact_label, c_exact, offset_points=4.0)
        _label_along_curve(
            ax, second, 8.0, r"$A b_\mu - \frac{1}{2}A(A-1)d_\mu$", c_second
        )
        _label_along_curve(ax, proposal, 6.3, r"PROPOSAL", "black", offset_points=-5.0)

        _save(fig, out_dir / "37_exponent_vs_proposal")
        plt.close(fig)


def print_figure_table(y: np.ndarray, dgamma_dy: np.ndarray, mu: np.ndarray) -> None:
    """Both closed forms against PROPOSAL over the range of the standalone figure."""
    b_mu, d_mu, t_mu = (float(v) for v in mu[:3])
    probe = np.array([-1.0, -0.5, 0.5, _A_ICECUBE, 2.0, 3.0, 5.0, 8.0, 11.0])
    reference = phi_from_spectrum(probe, y, dgamma_dy)
    two = np.asarray(phi_eigenvalue(probe, b_mu=b_mu, d_mu=d_mu), dtype=float)
    three = np.asarray(
        phi_eigenvalue_three_moment(probe, b_mu=b_mu, d_mu=d_mu, t_mu=t_mu), dtype=float
    )
    print("\nStandalone figure: closed forms against PROPOSAL over its A range")
    print("  A          " + "".join(f" {a:8.2f}" for a in probe))
    print("  PROPOSAL   " + "".join(f" {v:8.4f}" for v in reference) + "   [1/km]")
    print("  2-mom err% " + "".join(f" {v:8.2f}" for v in np.abs(two / reference - 1.0) * 100.0))
    print("  3-mom err% " + "".join(f" {v:8.2f}" for v in np.abs(three / reference - 1.0) * 100.0))
    print(
        "  Below A = 0 both closed forms leave PROPOSAL, because the hard edge of\n"
        "  the calibrated spectrum differs from the tabulated one and that is what\n"
        "  the negative-index convergence edge is sensitive to."
    )


def make_figure(
    models: dict,
    y: np.ndarray,
    dgamma_dy: np.ndarray,
    densities: dict,
    reference_density: np.ndarray,
    w_grid: np.ndarray,
    out_dir: pathlib.Path,
) -> None:
    """Assemble and write the three-panel convergence figure."""
    a_dense = np.linspace(0.2, 8.0, 160)
    phi_reference = phi_from_spectrum(a_dense, y, dgamma_dy)
    with plt.style.context(str(_STYLE)):
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1))
        _panel_shape(axes[0], models, y, dgamma_dy)
        _panel_eigenvalue(axes[1], models, a_dense, phi_reference)
        _panel_loss_law(axes[2], densities, reference_density, w_grid)
        fig.tight_layout()
        _save(fig, out_dir / "37_moment_convergence")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    n_max = max(args.n_moments)

    print(
        f"Muon energy {args.energy_gev:.3g} GeV, column {args.ell_km:g} km w.e., "
        f"matching up to {n_max} moments."
    )

    y = loss_spectrum_y_grid()
    dgamma_dy = proposal_loss_spectrum(args.energy_gev, y)["total"]
    mu = spectrum_moments(y, dgamma_dy, n_max)
    print("\nPROPOSAL moments [1/km]:")
    for n, value in enumerate(mu[: min(len(mu), 6)], start=1):
        print(f"  mu_{n} = {value:.6g}")

    models, (q_base, p_base) = build_models(mu, args.n_moments, args.dps)
    print(f"\nThree-moment base exponents: q = {q_base:+.4f}, p = {p_base:+.4f}")

    phi_reference = phi_from_spectrum(_A_GRID, y, dgamma_dy)

    w_grid = np.linspace(0.0, 6.0, 2001)
    reference_density = invert_log_loss_symbol(
        w_grid, args.ell_km, lambda s: phi_from_spectrum(s, y, dgamma_dy)
    )
    reference_tail = [survival(reference_density, w_grid, t) for t in _W_THRESHOLDS]

    densities: dict[int, np.ndarray] = {}
    tails: dict[int, list[float]] = {}
    for n, model in models.items():
        density = invert_log_loss_symbol(w_grid, args.ell_km, model["phi"])
        densities[n] = density
        tails[n] = [survival(density, w_grid, t) for t in _W_THRESHOLDS]

    print_moment_closure(mu, models, y)
    print_reference_floor(args.energy_gev)
    print_eigenvalue_table(models, phi_reference)
    print_shape_table(models, y, dgamma_dy)
    print_tail_table(tails, reference_tail)
    print_support_table(y, dgamma_dy)
    print_figure_table(y, dgamma_dy, mu)

    print()
    figure_exponent_vs_proposal(y, dgamma_dy, mu, args.exact_family, args.out_dir)
    make_figure(
        models, y, dgamma_dy, densities, reference_density, w_grid, args.out_dir
    )


if __name__ == "__main__":
    main()
