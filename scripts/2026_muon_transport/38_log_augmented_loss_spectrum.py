"""Example 38 -- a closed-form loss spectrum that also has the right shape.

Example 37 shows that the calibrated families reproduce ``Phi`` to a fraction of
a percent while their ``dGamma/dy`` stays wrong by orders of magnitude below
``y ~ 10^-4``, and that no number of moments repairs the shape. That is the right
answer if all one wants is ``Phi``. It is the wrong answer if one wants a loss
spectrum -- to sample collisions, to study the Landau-Pomeranchuk-Migdal
transition, or simply to answer a referee who points at the figure. This example
builds the family that does both.

**Why the single power fails.** The two- and three-moment families are a fixed
power ``y^(q-1)`` at the soft end. PROPOSAL's spectrum is not: its local slope
``d ln(dGamma/dy) / d ln y`` runs from about ``-1.05`` at ``y = 10^-6`` to
``-2.06`` at ``y = 10^-2``, because pair production carries 98-99% of the soft
rate and its shape runs with screening. A fixed power cannot follow a running
one, so the ratio of the two diverges as a power of ``1/y``.

**Why a mixture cannot fix it either.** The obvious repair is a positive
superposition of Beta kernels, one per channel. It provably fails. For any
positive combination of powers,

.. math:: \\frac{d^2 \\ln f}{d(\\ln y)^2} = \\mathrm{Var}_w(a_k) \\ge 0,

so the local slope of a positive mixture can only *increase* with ``y``. Between
``y = 10^-6`` and ``10^-2`` PROPOSAL's slope *decreases* by a full unit. The
``(1-y)^p`` factor bends the slope down by only ``-p y / (1-y)``, which is 0.03
at ``y = 10^-2``. The obstruction is structural, not a matter of conditioning or
of choosing a better basis.

**What does work.** A logarithmic factor bends the slope in the missing
direction, and it is the physically correct form, since the running comes from
screening logarithms in the first place. Take

.. math:: \\frac{d\\Gamma}{dy} = y^{q-1} (1-y)^p \\sum_{m=0}^{M} c_m \\ln^m(1/y).

Every moment and the exponent itself stay closed form, because

.. math:: \\int_0^1 dy\\, y^{a-1} (1-y)^p \\ln^m(1/y) = (-1)^m \\partial_a^m B(a, p+1),

and derivatives of a Beta function with respect to its first argument are that
Beta function times polynomials in digamma and its derivatives. So

.. math:: \\Phi(s) = \\sum_m c_m (-1)^m
    \\bigl[ \\partial_a^m B(a, p+1) - \\partial_a^m B(a, p+1+s) \\bigr]_{a=q}.

**How it is fitted, and how that differs from a calibration.** With ``M = 6``
there are seven coefficients. Three are spent imposing ``b_mu``, ``d_mu`` and
``t_mu`` exactly; the remaining four are least-squared against the tabulated
shape on a log grid. The result beats the three-moment family on both counts at
once: the shape lands within about 5% from ``y = 10^-7`` to ``y = 0.9``, where
the three-moment family is off by a factor of 180 at the low end, and ``Phi``
improves by an order of magnitude, to 0.055% at ``A = 8`` against 0.62%.

The cost is conceptual and worth stating plainly. The two- and three-moment
families are *calibrated* from three published numbers and need nothing else.
This one is *fitted* to a tabulated spectrum, so it needs ``q``, ``p``, ``M`` and
seven coefficients per energy. It is a better representation of the loss
spectrum and a worse closure. Use the moment families when the target is
``Phi``, and this one when the target is ``dGamma/dy``.

Usage
-----
    python examples/38_log_augmented_loss_spectrum.py
    python examples/38_log_augmented_loss_spectrum.py --energy-gev 1e8
    python examples/38_log_augmented_loss_spectrum.py --max-log-power 8
    python examples/38_log_augmented_loss_spectrum.py --out-dir /path/to/figures

``M = 6`` is the practical minimum. Three of the coefficients are spent on the
moment constraints, so ``M = 4`` leaves only two for the shape and the fit
degrades badly, to an rms log-ratio of 1.5 against 0.04 at ``M = 6``.

Writes one figure, ``38_log_augmented_loss_spectrum``, and prints the slope
monotonicity check, the fitted coefficients, and the shape, moment, exponent and
tail comparisons against PROPOSAL.
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import null_space
from scipy.special import comb as binomial
from scipy.special import digamma, gammaln, polygamma

from softpaws.transport.coefficients import loss_spectrum_y_grid, proposal_loss_spectrum
from softpaws.transport.eigenvalue import three_moment_loss_spectrum
from softpaws.transport.loss_distribution import invert_log_loss_symbol

_HERE = pathlib.Path(__file__).resolve().parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"

# Keeps (1 - y)^s finite on the hard edge of the grid, as in examples 27 and 37.
_ONE_MINUS_Y_FLOOR = 1.0e-12

# Spectral indices the exponent is compared at, as in example 37.
_A_GRID = np.array([0.5, 0.98, 2.0, 3.0, 5.0, 8.0])

# y values the shape and slope tables are printed at.
_Y_PROBE = np.array([1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 0.5, 0.9])

# Log-loss thresholds for the tail table.
_W_THRESHOLDS = (1.0, 1.5, 2.5)

# Search grids for the base exponents. Physical soft slopes q - 1 run between
# about -1 and -0.6, so q stays positive and the Beta functions of the closed
# form need no analytic continuation.
_Q_SCAN = np.arange(0.05, 0.90, 0.05)
_P_SCAN = (0.0, 0.5, 1.0, 2.0)


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
        help="Propagated column depth for the log-loss tail table [km water equivalent].",
    )
    parser.add_argument(
        "--max-log-power",
        type=int,
        default=6,
        help="Largest power M of ln(1/y) allowed in the fit. Needs M >= 3.",
    )
    parser.add_argument(
        "--fit-y-min",
        type=float,
        default=1.0e-7,
        help="Lower edge of the shape-fit window in y. Below the pair-production "
        "kinematic threshold the spectrum turns over and is not a power law.",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=_DEFAULT_OUT_DIR,
        help="Directory for the output figure.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Beta-function derivatives and the closed forms built on them
# ---------------------------------------------------------------------------


def beta_derivatives(a: float, b: float, m_max: int) -> np.ndarray:
    """Derivatives ``d^m/da^m B(a, b)`` for ``m = 0 ... m_max``.

    Writing ``B = exp(g)`` with ``g(a) = lnGamma(a) + lnGamma(b) - lnGamma(a+b)``,
    the derivatives follow the exponential formula

    .. math:: D_m = \\sum_{j=0}^{m-1} \\binom{m-1}{j} D_j\\, g^{(m-j)},

    with ``g^(k) = psi^(k-1)(a) - psi^(k-1)(a+b)``. Evaluating them this way is
    exact and fast, where numerical differentiation would be neither.

    Parameters
    ----------
    a : float
        First Beta argument. Must be positive.
    b : float
        Second Beta argument. Must be positive.
    m_max : int
        Highest derivative order required.

    Returns
    -------
    derivatives : np.ndarray
        Array of length ``m_max + 1`` holding ``d^m B / da^m``.
    """
    g = np.empty(m_max + 1)
    for k in range(1, m_max + 1):
        if k == 1:
            g[k] = digamma(a) - digamma(a + b)
        else:
            g[k] = polygamma(k - 1, a) - polygamma(k - 1, a + b)
    out = np.empty(m_max + 1)
    out[0] = np.exp(gammaln(a) + gammaln(b) - gammaln(a + b))
    for m in range(1, m_max + 1):
        out[m] = sum(binomial(m - 1, j, exact=True) * out[j] * g[m - j] for j in range(m))
    return out


def log_family_moments(q: float, p: float, m_max: int, orders: tuple[int, ...]) -> np.ndarray:
    """Moment matrix of the log-augmented family.

    Entry ``[n, m]`` is ``int_0^1 dy y^n * y^(q-1) (1-y)^p ln^m(1/y)``, which is
    ``(-1)^m d^m/da^m B(a, p+1)`` at ``a = n + q``.

    Parameters
    ----------
    q, p : float
        Base-weight exponents.
    m_max : int
        Largest power of ``ln(1/y)`` in the family.
    orders : tuple of int
        Moment orders ``n`` to evaluate.

    Returns
    -------
    matrix : np.ndarray
        Shape ``(len(orders), m_max + 1)`` matrix, in units of the coefficients.
    """
    signs = np.array([(-1.0) ** m for m in range(m_max + 1)])
    return np.array([signs * beta_derivatives(n + q, p + 1.0, m_max) for n in orders])


def log_family_phi(
    coefficients: np.ndarray, q: float, p: float, s: float | np.ndarray
) -> np.ndarray:
    """Transport exponent ``Phi(s)`` of the log-augmented family, in closed form.

    Term by term the ``y`` integral of ``y^(q-1)(1-y)^p ln^m(1/y) [1 - (1-y)^s]``
    is a difference of Beta derivatives, so

    .. math:: \\Phi(s) = \\sum_m c_m (-1)^m
        \\bigl[\\partial_a^m B(a,p+1) - \\partial_a^m B(a,p+1+s)\\bigr]_{a=q}.

    Parameters
    ----------
    coefficients : np.ndarray
        Coefficients ``c_0 ... c_M`` [km^-1].
    q, p : float
        Base-weight exponents.
    s : float or np.ndarray
        Mellin variable. Real only; the complex case needs a complex polygamma
        and is not required here, since the log-loss law is obtained by
        quadrature instead.

    Returns
    -------
    phi : np.ndarray
        ``Phi(s)`` [km^-1], with the shape of ``s``.
    """
    m_max = len(coefficients) - 1
    signs = np.array([(-1.0) ** m for m in range(m_max + 1)])
    at_zero = signs * beta_derivatives(q, p + 1.0, m_max)
    s_array = np.atleast_1d(np.asarray(s, dtype=float))
    out = np.empty(s_array.shape).reshape(-1)
    for i, s_value in enumerate(s_array.reshape(-1)):
        shifted = signs * beta_derivatives(q, p + 1.0 + s_value, m_max)
        out[i] = float(np.dot(coefficients, at_zero - shifted))
    return out.reshape(s_array.shape)


def log_family_spectrum(
    coefficients: np.ndarray, q: float, p: float, y: np.ndarray
) -> np.ndarray:
    """Loss spectrum ``dGamma/dy`` of the log-augmented family.

    Parameters
    ----------
    coefficients : np.ndarray
        Coefficients ``c_0 ... c_M`` [km^-1].
    q, p : float
        Base-weight exponents.
    y : np.ndarray
        Grid of fractional energy losses in ``(0, 1)``.

    Returns
    -------
    dgamma_dy : np.ndarray
        Differential loss rate [km^-1].
    """
    log_inverse = np.log(1.0 / y)
    series = np.zeros_like(y)
    for m, c_m in enumerate(coefficients):
        series = series + c_m * log_inverse**m
    return y ** (q - 1.0) * (1.0 - y) ** p * series


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def fit_log_family(
    fit_y: np.ndarray,
    fit_dgamma: np.ndarray,
    mu: np.ndarray,
    m_max: int,
) -> tuple[np.ndarray, float, float, float]:
    """Fit the shape subject to exact ``b_mu``, ``d_mu`` and ``t_mu``.

    For each ``(q, p, M)`` on the search grid the three moment constraints are
    imposed exactly by writing ``c = c_0 + Z z``, with ``c_0`` a particular
    solution and ``Z`` a basis for the null space of the moment matrix. The
    remaining freedom ``z`` is least-squared against the tabulated shape with
    relative weights, so every decade in ``y`` counts equally. Candidates whose
    spectrum goes negative anywhere are rejected.

    Parameters
    ----------
    fit_y : np.ndarray
        Log-spaced grid of fractional energy losses used for the shape fit.
    fit_dgamma : np.ndarray
        Reference ``dGamma/dy`` [km^-1] on ``fit_y``. Must be strictly positive.
    mu : np.ndarray
        Reference moments ``(b_mu, d_mu, t_mu)`` [km^-1].
    m_max : int
        Largest power of ``ln(1/y)`` allowed.

    Returns
    -------
    coefficients : np.ndarray
        Fitted ``c_0 ... c_M`` [km^-1].
    q, p : float
        Chosen base exponents.
    rms : float
        Root-mean-square of ``ln(fit / reference)`` over the fit window.

    Raises
    ------
    RuntimeError
        Raised if no candidate on the search grid stays positive.
    """
    weights = 1.0 / fit_dgamma
    log_inverse = np.log(1.0 / fit_y)
    best: tuple[float, np.ndarray, float, float] | None = None
    for q in _Q_SCAN:
        for p in _P_SCAN:
            for m in range(3, m_max + 1):
                base = fit_y ** (q - 1.0) * (1.0 - fit_y) ** p
                design = base[:, None] * np.array([log_inverse**k for k in range(m + 1)]).T
                constraint = log_family_moments(q, p, m, (1, 2, 3))
                particular = np.linalg.lstsq(constraint, mu, rcond=None)[0]
                basis = null_space(constraint)
                if basis.shape[1] == 0:
                    continue
                residual = 1.0 - (design @ particular) * weights
                free = np.linalg.lstsq((design @ basis) * weights[:, None], residual, rcond=None)[0]
                coefficients = particular + basis @ free
                model = design @ coefficients
                if np.any(model <= 0.0):
                    continue
                rms = float(np.sqrt(np.mean(np.log(model / fit_dgamma) ** 2)))
                if best is None or rms < best[0]:
                    best = (rms, coefficients, float(q), float(p))
    if best is None:
        raise RuntimeError(
            "no positive candidate on the (q, p, M) search grid; widen _Q_SCAN or "
            "raise --max-log-power."
        )
    rms, coefficients, q, p = best
    return coefficients, q, p, rms


# ---------------------------------------------------------------------------
# Reference helpers
# ---------------------------------------------------------------------------


def phi_from_spectrum(
    s: complex | np.ndarray, y: np.ndarray, dgamma_dy: np.ndarray, chunk: int = 64
) -> np.ndarray:
    """``Phi(s)`` by quadrature of a tabulated loss rate, as in examples 27 and 37."""
    s_array = np.atleast_1d(np.asarray(s))
    log_one_minus_y = np.log1p(-np.clip(y, None, 1.0 - _ONE_MINUS_Y_FLOOR))
    out = np.empty(s_array.shape, dtype=np.result_type(s_array.dtype, float)).reshape(-1)
    flat = s_array.reshape(-1)
    for start in range(0, flat.size, chunk):
        block = flat[start : start + chunk][:, None]
        bracket = 1.0 - np.exp(block * log_one_minus_y[None, :])
        out[start : start + chunk] = np.trapezoid(dgamma_dy[None, :] * bracket, y, axis=1)
    return out.reshape(s_array.shape)


def local_slope(y: np.ndarray, dgamma_dy: np.ndarray) -> np.ndarray:
    """Local logarithmic slope ``d ln(dGamma/dy) / d ln y``.

    Parameters
    ----------
    y : np.ndarray
        Ascending grid of fractional energy losses.
    dgamma_dy : np.ndarray
        Differential loss rate [km^-1]. Non-positive entries return NaN.

    Returns
    -------
    slope : np.ndarray
        Local slope, NaN where the rate vanishes.
    """
    positive = dgamma_dy > 0.0
    logged = np.full_like(y, np.nan)
    logged[positive] = np.log(dgamma_dy[positive])
    return np.gradient(logged, np.log(y))


def slope_on_log_grid(
    y: np.ndarray,
    dgamma_dy: np.ndarray,
    y_min: float = 1.0e-8,
    y_max: float = 0.95,
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Local slope resampled onto a coarse log grid, for plotting.

    The working grid spaces points 0.004 decades apart, so differentiating on it
    turns the channel thresholds -- pair production at ``y = 2.1e-9``,
    photonuclear at ``y = 1.5e-7`` -- into spikes that swamp the physical slope.
    Interpolating the log spectrum onto a coarse grid first shows the slope the
    argument is about.

    Parameters
    ----------
    y : np.ndarray
        Ascending grid of fractional energy losses.
    dgamma_dy : np.ndarray
        Differential loss rate [km^-1].
    y_min, y_max : float, optional
        Range of the output grid.
    n_points : int, optional
        Number of output nodes.

    Returns
    -------
    probe : np.ndarray
        Log-spaced grid of ``y`` values.
    slope : np.ndarray
        Local slope on ``probe``.
    """
    probe = np.logspace(np.log10(y_min), np.log10(y_max), n_points)
    positive = dgamma_dy > 0.0
    logged = np.interp(np.log(probe), np.log(y[positive]), np.log(dgamma_dy[positive]))
    return probe, np.gradient(logged, np.log(probe))


def survival(density: np.ndarray, w_grid: np.ndarray, threshold: float) -> float:
    """Tail probability ``P(W > threshold)`` of a log-loss density."""
    mask = w_grid >= threshold
    return float(np.trapezoid(density[mask], w_grid[mask]))


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def print_slope_obstruction(y: np.ndarray, dgamma_dy: np.ndarray) -> None:
    """Show that PROPOSAL's slope falls, which no positive mixture can reproduce."""
    grid, slope = slope_on_log_grid(y, dgamma_dy)
    print("\nLocal slope d ln(dGamma/dy) / d ln y of PROPOSAL's total")
    print("  a positive mixture of powers can only have a NON-DECREASING slope")
    previous = None
    falls = 0
    for probe in _Y_PROBE:
        value = float(np.interp(np.log(probe), np.log(grid), slope))
        flag = ""
        if previous is not None and value < previous - 1.0e-9:
            flag = "  <- decreasing"
            falls += 1
        print(f"    y = {probe:8.0e}   slope = {value:+7.3f}{flag}")
        previous = value
    print(
        f"  {falls} decreasing steps, so a positive Beta mixture is ruled out "
        "structurally."
    )


def print_shape_table(
    y: np.ndarray, dgamma_dy: np.ndarray, three_moment: np.ndarray, log_family: np.ndarray
) -> None:
    """Pointwise ratio of each closed form to PROPOSAL's spectrum."""
    reference = np.interp(_Y_PROBE, y, dgamma_dy)
    print("\nShape: (dGamma/dy)_model / PROPOSAL")
    print("  model     " + "".join(f" y={v:<8.0e}" for v in _Y_PROBE))
    for label, model in (("3-moment", three_moment), ("log family", log_family)):
        ratio = np.interp(_Y_PROBE, y, model) / reference
        print(f"  {label:<10}" + "".join(f" {v:10.3f}" for v in ratio))


def print_moment_table(
    y: np.ndarray, dgamma_dy: np.ndarray, three_moment: np.ndarray, log_family: np.ndarray
) -> None:
    """Moment closure of each closed form against PROPOSAL."""
    orders = (1, 2, 3, 4, 6)
    print("\nMoment closure: <y^n>_model / <y^n>_PROPOSAL")
    print("  model     " + "".join(f"     n={n}" for n in orders))
    for label, model in (("3-moment", three_moment), ("log family", log_family)):
        ratios = [
            np.trapezoid(model * y**n, y) / np.trapezoid(dgamma_dy * y**n, y) for n in orders
        ]
        print(f"  {label:<10}" + "".join(f" {v:9.4f}" for v in ratios))


def print_eigenvalue_table(
    phi_reference: np.ndarray, phi_three: np.ndarray, phi_log: np.ndarray
) -> None:
    """Relative error of the exponent for each closed form."""
    print("\nTransport exponent: |Phi_model(A) / Phi_exact(A) - 1| [%]")
    print("  model     " + "".join(f"  A={a:<5.2f}" for a in _A_GRID))
    print("  --        " + "".join(f" {v:7.4f}" for v in phi_reference) + "   <- [1/km]")
    for label, phi in (("3-moment", phi_three), ("log family", phi_log)):
        print(
            f"  {label:<10}"
            + "".join(f" {v:7.3f}" for v in np.abs(phi / phi_reference - 1.0) * 100.0)
        )


def print_tail_table(tails: dict[str, list[float]], reference: list[float]) -> None:
    """Log-loss tail probabilities against the PROPOSAL-driven inversion."""
    print("\nLog-loss tail: P(W > w), and the ratio to PROPOSAL")
    print("  model     " + "".join(f"   P(W>{w:.1f})    x ref" for w in _W_THRESHOLDS))
    print("  --        " + "".join(f" {v:10.3e}        -" for v in reference))
    for label, values in tails.items():
        cells = "".join(f" {v:10.3e} {v / r:8.3f}" for v, r in zip(values, reference))
        print(f"  {label:<10}" + cells)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_path.with_suffix(suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  wrote {path}")


def _panel_slope(ax, y, dgamma_dy, three_moment, log_family) -> None:
    """Panel (a): the local slope, which is where the single power law fails."""
    probe, slope_reference = slope_on_log_grid(y, dgamma_dy)
    ax.plot(probe, slope_reference, color="0.2", lw=1.6, label="PROPOSAL")
    for model, style, label in (
        (three_moment, "--", "3-moment"),
        (log_family, "-.", "log family"),
    ):
        ax.plot(*slope_on_log_grid(y, model), lw=1.2, ls=style, label=label)
    ax.set_xscale("log")
    ax.set_ylim(-2.6, 0.2)
    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"$\mathrm{d}\ln(\mathrm{d}\Gamma/\mathrm{d}y)\,/\,\mathrm{d}\ln y$")
    ax.set_title(r"(a) local slope", fontsize=8)
    ax.legend(fontsize=6)


def _panel_shape(ax, y, dgamma_dy, three_moment, log_family) -> None:
    """Panel (b): ratio to PROPOSAL, the figure example 37 leaves unresolved."""
    good = dgamma_dy > 0.0
    reference = dgamma_dy[good]
    ax.axhline(1.0, color="0.35", lw=0.8, zorder=1)
    ax.plot(y[good], three_moment[good] / reference, lw=1.2, ls="--", label="3-moment")
    ax.plot(y[good], log_family[good] / reference, lw=1.2, ls="-.", label="log family")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.0e-8, 1.0)
    ax.set_ylim(1.0e-1, 1.0e4)
    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"$(\mathrm{d}\Gamma/\mathrm{d}y)_{\rm model}\,/\,\mathrm{PROPOSAL}$")
    ax.set_title(r"(b) shape", fontsize=8)
    ax.legend(fontsize=6)


def _panel_eigenvalue(ax, a_dense, phi_reference, phi_three, phi_log) -> None:
    """Panel (c): the exponent, which the log family also improves."""
    ax.plot(
        a_dense,
        np.abs(phi_three / phi_reference - 1.0) * 100.0,
        lw=1.2,
        ls="--",
        label="3-moment",
    )
    ax.plot(
        a_dense, np.abs(phi_log / phi_reference - 1.0) * 100.0, lw=1.2, ls="-.", label="log family"
    )
    ax.axvline(0.98, color="0.35", lw=0.8, ls=":", zorder=1)
    ax.set_yscale("log")
    ax.set_xlim(a_dense[0], a_dense[-1])
    ax.set_ylim(1.0e-4, 1.0e1)
    ax.set_xlabel(r"$A$")
    ax.set_ylabel(r"$|\Phi_{\rm model}(A)/\Phi(A) - 1|$ [\%]")
    ax.set_title(r"(c) transport exponent", fontsize=8)
    ax.legend(fontsize=6)


def make_figure(
    y, dgamma_dy, three_moment, log_family, coefficients, q, p, out_dir
) -> None:
    """Assemble and write the three-panel figure."""
    a_dense = np.linspace(0.2, 8.0, 160)
    phi_reference = phi_from_spectrum(a_dense, y, dgamma_dy)
    phi_three = phi_from_spectrum(a_dense, y, three_moment)
    phi_log = log_family_phi(coefficients, q, p, a_dense)
    with plt.style.context(str(_STYLE)):
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1))
        _panel_slope(axes[0], y, dgamma_dy, three_moment, log_family)
        _panel_shape(axes[1], y, dgamma_dy, three_moment, log_family)
        _panel_eigenvalue(axes[2], a_dense, phi_reference, phi_three, phi_log)
        fig.tight_layout()
        _save(fig, out_dir / "38_log_augmented_loss_spectrum")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if args.max_log_power < 3:
        raise SystemExit("--max-log-power must be at least 3 to leave shape freedom.")

    print(f"Muon energy {args.energy_gev:.3g} GeV, column {args.ell_km:g} km w.e.")

    y = loss_spectrum_y_grid()
    dgamma_dy = proposal_loss_spectrum(args.energy_gev, y)["total"]
    mu = np.array([np.trapezoid(dgamma_dy * y**n, y) for n in (1, 2, 3)])
    print(f"PROPOSAL moments [1/km]: b_mu = {mu[0]:.6g}, d_mu = {mu[1]:.6g}, t_mu = {mu[2]:.6g}")

    print_slope_obstruction(y, dgamma_dy)

    kappa, q_three, p_three = three_moment_loss_spectrum(*mu)
    three_moment = (
        float(kappa) * y ** (float(q_three) - 1.0) * (1.0 - y) ** float(p_three)
    )
    print(
        f"\nThree-moment calibration: kappa = {float(kappa):.5g}, "
        f"q = {float(q_three):+.4f}, p = {float(p_three):+.4f}"
    )

    fit_y = np.logspace(np.log10(args.fit_y_min), np.log10(0.999), 600)
    fit_dgamma = np.interp(fit_y, y, dgamma_dy)
    positive = fit_dgamma > 0.0
    coefficients, q, p, rms = fit_log_family(
        fit_y[positive], fit_dgamma[positive], mu, args.max_log_power
    )
    log_family = log_family_spectrum(coefficients, q, p, y)
    print(
        f"Log-augmented fit:        M = {len(coefficients) - 1}, "
        f"q = {q:+.4f}, p = {p:+.4f}, rms ln-ratio = {rms:.4f}"
    )
    print(f"  c = {np.array2string(coefficients, precision=5)}")
    if rms > 0.25:
        print(
            f"  WARNING: rms log-ratio {rms:.2f} is far above the ~0.04 reachable at "
            f"M = 6. Three coefficients go to the moment constraints, so raise "
            f"--max-log-power to leave more freedom for the shape."
        )

    print_shape_table(y, dgamma_dy, three_moment, log_family)
    print_moment_table(y, dgamma_dy, three_moment, log_family)

    phi_reference = phi_from_spectrum(_A_GRID, y, dgamma_dy)
    print_eigenvalue_table(
        phi_reference,
        phi_from_spectrum(_A_GRID, y, three_moment),
        log_family_phi(coefficients, q, p, _A_GRID),
    )

    w_grid = np.linspace(0.0, 6.0, 2001)
    reference_density = invert_log_loss_symbol(
        w_grid, args.ell_km, lambda s: phi_from_spectrum(s, y, dgamma_dy)
    )
    tails = {}
    for label, spectrum in (("3-moment", three_moment), ("log family", log_family)):
        density = invert_log_loss_symbol(
            w_grid, args.ell_km, lambda s, v=spectrum: phi_from_spectrum(s, y, v)
        )
        tails[label] = [survival(density, w_grid, t) for t in _W_THRESHOLDS]
    print_tail_table(tails, [survival(reference_density, w_grid, t) for t in _W_THRESHOLDS])

    print()
    make_figure(y, dgamma_dy, three_moment, log_family, coefficients, q, p, args.out_dir)


if __name__ == "__main__":
    main()
