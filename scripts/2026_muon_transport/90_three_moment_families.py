"""Example 90 -- how much of Figure 2 do three matched moments guarantee?

Figure 2 (example 37) shows the three-moment Beta family

.. math:: \\frac{d\\Gamma}{dy} = \\kappa\\, y^{q-1} (1-y)^p

on top of PROPOSAL's exponent out to ``A = 11``. That agreement has two possible
sources. Either matching ``<y>``, ``<y^2>`` and ``<y^3>`` forces it, and any
kernel with those moments would do as well, or it comes from the particular
shape of the Beta family. This example separates the two.

**The envelope.** Write the exponent as an integral over the measure
``dmu = y dGamma``,

.. math:: \\Phi(s) = \\int_0^1 d\\mu(y)\\, f_s(y), \\qquad
          f_s(y) = \\frac{1 - (1-y)^s}{y}.

Matching three moments fixes the mass, mean and second moment of ``mu``. At
``s = 1, 2, 3`` the function ``f_s`` is a polynomial of degree ``s - 1``, so every
kernel with those moments gives the same ``Phi`` there. At any other ``s`` the
remaining freedom in ``mu`` enters. The smallest and largest ``Phi(s)`` over all
positive measures with the three moments is a linear program on a grid of ``y``
nodes, which this example solves exactly. Above ``A = 3`` the solutions are the
two-point measures the moment problem predicts: an atom at ``y = 1`` plus one at
``y = (mu_2 - mu_3)/(mu_1 - mu_2)`` for the lower edge, and an atom at ``y = 0``,
a pure continuous loss, plus one at ``y = mu_3/mu_2`` for the upper edge.

**Two other smooth kernels.** The envelope is set by delta functions, which no
cross section looks like. The realistic question is how far a smooth kernel with
the same moments can move. The library already solves the family

.. math:: \\frac{d\\Gamma}{dy} = y^{q'-1} (a_0 + a_1 y + a_2 y^2)

for the three coefficients at any base exponent ``q'``
(:func:`~softpaws.transport.eigenvalue.n_moment_coefficients`), with ``Phi`` in
closed form as a sum of Beta functions. Choosing ``q'`` sets the soft end of the
kernel. The kernel stays positive on ``(0, 1)`` only for ``q'`` between about
``-0.3`` and ``-1``, so the example draws those two extremes. The soft end then
runs as ``y^{-1.3}`` in one and ``y^{-1.95}`` in the other, around the Beta
family's ``y^{-1.74}``. PROPOSAL's spectrum is no single power law. It steepens
from ``y^{-1.1}`` near ``y = 10^{-5}`` to ``y^{-2.05}`` near ``y = 0.05``, and is
close to ``y^{-1.7}`` in the decade around ``y = 0.01``.

**What the example finds.**

1. **Agreement to ``A = 3`` is guaranteed, and not only at the integers.**
   Between ``A = 1`` and ``3`` the envelope stays below 0.5%. Below ``A = 1`` it
   opens again, to ``+5%`` at ``A = 0.5``, because ``f_s`` stops being close to
   a polynomial there.
2. **Beyond ``A = 3`` the moments allow much more than the figure shows.** At
   ``A = 10`` the envelope runs from ``-15%`` to ``+9%`` around PROPOSAL.
3. **Smooth kernels use only a small part of that room.** The two alternatives
   sit at ``-5.6%`` and ``+1.4%`` at ``A = 10``, and the Beta family at
   ``-0.8%``. The soft end decides the sign. At large ``s``,
   ``1 - (1-y)^s`` saturates once ``y > 1/s``, so ``Phi(s)`` counts the
   collisions harder than ``y ~ 1/A`` and grows as ``s^{-q'}``. A softer kernel
   falls behind and a steeper one runs ahead. ``Phi(10)`` draws most of its
   weight from ``y = 10^{-3}`` to ``0.16``, with its median at ``y = 0.012``,
   where PROPOSAL's local slope is close to the Beta family's ``-1.74``. That
   is why the Beta family tracks PROPOSAL.
4. **Each further moment closes most of what is left.** Matching ``<y^4>`` as
   well shrinks the envelope at ``A = 10`` to ``-3.5%`` and ``+3.2%``, and
   ``<y^5>`` shrinks it further. The moments converge on the exponent even
   though no finite number of them fixes the spectrum.

This is the mirror image of the endpoint problem of the range. The range needs
``Phi'(0)`` and ``Phi''(0)``, which probe ``y -> 1``, where three moments do not
pin the family down (Appendix A of the paper). Large ``A`` probes ``y -> 0``,
where the Beta family happens to have the right shape.

**Where it matters.** Astrophysical fluxes give ``A`` near 1 to 2, and
conventional atmospheric neutrinos with ``gamma = 3.7`` and ``lambda = 0.4`` give
``A = 2.3``. Every result of the paper sits where the moments guarantee the
exponent. The large-``A`` agreement of Figure 2 is a property of the Beta shape,
and the text should say so.

Requires the optional ``proposal`` dependency (``pip install -e ".[transport]"``)
and ``mpmath``.

Usage
-----
    python scripts/2026_muon_transport/90_three_moment_families.py
    python scripts/2026_muon_transport/90_three_moment_families.py --q-soft -0.25
    python scripts/2026_muon_transport/90_three_moment_families.py --out-dir /path/to/figures

Writes three square figures. ``90a_three_moment_families`` is Figure 2 with the
truncations replaced by the two alternative kernels and the three-moment
envelope. ``90b_three_moment_ratio`` is the same content as a ratio to PROPOSAL,
where the percent-level differences are visible, with the four-moment envelope
drawn inside the three-moment one. ``90c_moment_envelopes`` drops the individual
kernels and shows only the envelopes for one to five matched moments, nested,
as a ratio to PROPOSAL. One moment fixes only ``b_mu``, and since ``f_A`` runs
monotonically from ``A`` at ``y = 0`` to 1 at ``y = 1`` its envelope is
``b_mu`` to ``A b_mu``, with the first-order truncation as its edge. Two moments
pin ``Phi`` only at ``A = 1`` and ``2``, and the band is already ``4%`` wide at ``A = 3``, which is
the point the third moment closes. ``90d_moment_envelopes_absolute`` draws the
same nested envelopes as ``Phi(A)`` itself, on the axes of Figure 2, with the
paper's three-moment closed form and the first- and second-order truncations of
Figure 2 on top. The first-order line
``A b_mu`` is the upper edge of the one-moment envelope. The second-order curve
lies outside the two-moment envelope everywhere except ``A = 0, 1, 2``: it
integrates the same two moments against the truncated weight
``A - A(A-1) y / 2``, and ``f_A`` differs from that weight with the sign of
``A(A-1)(A-2)`` at every ``y``. No positive loss spectrum reproduces it, which
is the same fact that lets it turn negative at ``A_die``.
"""

import argparse
import pathlib
import warnings
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from scipy.optimize import linprog

from softpaws.transport.coefficients import loss_spectrum_y_grid, proposal_loss_spectrum
from softpaws.transport.eigenvalue import (
    n_moment_coefficients,
    n_moment_loss_spectrum,
    phi_eigenvalue_three_moment,
    phi_symbol_n_moment,
    three_moment_loss_spectrum,
)

_HERE = pathlib.Path(__file__).resolve().parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"

# Keeps (1 - y)^s finite on the hard edge of the grid, as in example 37.
_ONE_MINUS_Y_FLOOR = 1.0e-12

# Range of both figures. The envelope is unbounded below for A < 0, where
# (1 - y)^A diverges at the hard edge, so the figures start at zero.
_A_MIN, _A_MAX = 0.0, 11.0
_A_ICECUBE = 0.98  # gamma = 2.38, lambda = 0.4
_A_ATMOSPHERIC = 2.3  # gamma = 3.7, lambda = 0.4

# Spectral indices of the printed table.
_A_TABLE = np.array([0.5, _A_ICECUBE, 1.5, 2.5, 3.0, 5.0, 8.0, 10.0, 11.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--energy-gev",
        type=float,
        default=1.0e6,
        help="Muon energy at which the loss spectrum is evaluated [GeV].",
    )
    parser.add_argument(
        "--q-soft",
        type=float,
        default=-0.30,
        help="Base exponent of the softer alternative kernel, y^(q-1) at small y.",
    )
    parser.add_argument(
        "--q-steep",
        type=float,
        default=-0.95,
        help="Base exponent of the steeper alternative kernel, y^(q-1) at small y.",
    )
    parser.add_argument(
        "--n-nodes",
        type=int,
        default=2500,
        help="Nodes per side of the y grid of the envelope linear program.",
    )
    parser.add_argument(
        "--n-a",
        type=int,
        default=111,
        help="Number of spectral indices at which the envelope is solved.",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=_DEFAULT_OUT_DIR,
        help="Directory for the output figures.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Reference and kernels
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
        Moments ``mu_1 ... mu_n_max`` [km^-1].
    """
    return np.array([np.trapezoid(dgamma_dy * y**n, y) for n in range(1, n_max + 1)])


def phi_from_spectrum(s: np.ndarray, y: np.ndarray, dgamma_dy: np.ndarray) -> np.ndarray:
    """Reference ``Phi(s)`` by quadrature of a tabulated loss rate.

    Parameters
    ----------
    s : np.ndarray
        Real spectral indices.
    y : np.ndarray
        Ascending grid of fractional energy losses in ``(0, 1)``.
    dgamma_dy : np.ndarray
        Differential loss rate [km^-1] on ``y``.

    Returns
    -------
    phi : np.ndarray
        ``Phi(s)`` [km^-1], with the shape of ``s``.
    """
    s_array = np.atleast_1d(np.asarray(s, dtype=float))
    log_one_minus_y = np.log1p(-np.clip(y, None, 1.0 - _ONE_MINUS_Y_FLOOR))
    out = np.array([
        np.trapezoid(dgamma_dy * -np.expm1(value * log_one_minus_y), y) for value in s_array
    ])
    return out.reshape(s_array.shape)


def alternative_kernel(mu: np.ndarray, q_base: float) -> dict:
    """Three-moment kernel ``y^(q-1) (a_0 + a_1 y + a_2 y^2)`` at a chosen base.

    Parameters
    ----------
    mu : np.ndarray
        Moments ``mu_1, mu_2, mu_3`` [km^-1] to match.
    q_base : float
        Base exponent. The kernel runs as ``y^(q_base - 1)`` at small ``y``.

    Returns
    -------
    kernel : dict
        Keys ``q``, ``a`` (the three coefficients), ``phi`` (callable of real
        ``s``) and ``min_rate`` (the smallest ``dGamma/dy`` on a fine grid).

    Raises
    ------
    ValueError
        If the matched kernel turns negative anywhere on ``(0, 1)``, since it
        is then not a loss spectrum.
    """
    coefficients = np.asarray(n_moment_coefficients(mu[:3], q_base, 0.0), dtype=float)
    y_check = np.logspace(-12.0, np.log10(1.0 - 1.0e-9), 20000)
    rate = np.asarray(n_moment_loss_spectrum(coefficients, q_base, 0.0, y_check), dtype=float)
    if np.any(rate < 0.0):
        raise ValueError(
            f"The three-moment kernel with base q = {q_base:g} turns negative; "
            "choose a base exponent between about -0.3 and -1."
        )

    def phi(s: np.ndarray, a: np.ndarray = coefficients, q: float = q_base) -> np.ndarray:
        s_complex = np.asarray(s, dtype=complex)
        return np.real(np.asarray(phi_symbol_n_moment(s_complex, a, q, 0.0)))

    return {"q": q_base, "a": coefficients, "phi": phi, "min_rate": float(rate.min())}


# ---------------------------------------------------------------------------
# The envelope of all kernels with the same moments
# ---------------------------------------------------------------------------


def envelope_nodes(n_per_side: int) -> np.ndarray:
    """Nodes on ``[0, 1]`` for the envelope, dense at both endpoints.

    The extremal measures sit at the endpoints and at interior points that move
    with the moments, so the grid has to resolve both ends.
    """
    soft = np.logspace(-7.0, np.log10(0.5), n_per_side)
    return np.unique(np.concatenate([[0.0, 1.0], soft, 1.0 - soft]))


def f_weight(s: float, nodes: np.ndarray) -> np.ndarray:
    """``f_s(y) = [1 - (1-y)^s] / y`` with its limits ``s`` at 0 and 1 at 1."""
    out = np.empty_like(nodes)
    at_zero, at_one = nodes == 0.0, nodes == 1.0
    inner = ~(at_zero | at_one)
    out[at_zero] = s
    out[at_one] = 1.0
    out[inner] = -np.expm1(s * np.log1p(-nodes[inner])) / nodes[inner]
    return out


def moment_envelope(
    a_grid: np.ndarray, mu: np.ndarray, n_moments: int, nodes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Smallest and largest ``Phi(s)`` over positive kernels with fixed moments.

    Solves two linear programs per spectral index, in the measure
    ``dmu = y dGamma`` placed on ``nodes``. The first ``n_moments`` moments of
    ``dGamma`` are the zeroth to ``(n_moments - 1)``-th of ``mu``.

    Parameters
    ----------
    a_grid : np.ndarray
        Positive spectral indices.
    mu : np.ndarray
        Moments ``mu_1 ...`` [km^-1] of the reference spectrum.
    n_moments : int
        Number of moments held fixed.
    nodes : np.ndarray
        Support grid on ``[0, 1]``.

    Returns
    -------
    lower, upper : np.ndarray
        Envelope of ``Phi`` [km^-1] on ``a_grid``.
    """
    constraints = np.vstack([nodes**k for k in range(n_moments)])
    targets = mu[:n_moments]
    lower = np.empty(a_grid.size)
    upper = np.empty(a_grid.size)
    for i, s in enumerate(a_grid):
        weight = f_weight(float(s), nodes)
        low = linprog(weight, A_eq=constraints, b_eq=targets, bounds=(0, None), method="highs")
        high = linprog(-weight, A_eq=constraints, b_eq=targets, bounds=(0, None), method="highs")
        if not (low.success and high.success):
            raise RuntimeError(f"Envelope linear program failed at A = {s:g}.")
        lower[i], upper[i] = low.fun, -high.fun
    return lower, upper


def two_point_extremes(
    mu: np.ndarray, s: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """The two-point measures that bound ``Phi`` above ``A = 3``.

    Parameters
    ----------
    mu : np.ndarray
        Moments ``mu_1, mu_2, mu_3`` [km^-1].
    s : np.ndarray
        Spectral indices.

    Returns
    -------
    lower, upper : np.ndarray
        ``Phi(s)`` [km^-1] of the two measures.
    y_lower, y_upper : float
        Interior atom of each: the lower measure adds an atom at ``y = 1``, the
        upper one an atom at ``y = 0``, a continuous loss.
    """
    m1, m2, m3 = (float(v) for v in mu[:3])
    y_lower = (m2 - m3) / (m1 - m2)
    c_lower = (m1 - m2) / (y_lower * (1.0 - y_lower))
    c_hard = m1 - c_lower * y_lower
    lower = c_hard + c_lower * -np.expm1(s * np.log1p(-y_lower))

    y_upper = m3 / m2
    c_upper = m2 / y_upper**2
    drift = m1 - c_upper * y_upper
    upper = drift * s + c_upper * -np.expm1(s * np.log1p(-y_upper))
    return lower, upper, y_lower, y_upper


# ---------------------------------------------------------------------------
# Printout
# ---------------------------------------------------------------------------


def print_table(rows: dict, reference: np.ndarray) -> None:
    """Every curve as a percentage of PROPOSAL at the table indices."""
    print("\nPhi(A) against PROPOSAL [%], the envelopes as lower / upper edge")
    print("  A              " + "".join(f" {a:7.2f}" for a in _A_TABLE))
    print("  PROPOSAL [1/km]" + "".join(f" {v:7.4f}" for v in reference))
    for label, values in rows.items():
        if isinstance(values, tuple):
            low = 100.0 * (values[0] / reference - 1.0)
            high = 100.0 * (values[1] / reference - 1.0)
            print(f"  {label:<15}" + "".join(f" {v:+7.2f}" for v in low))
            print(f"  {'':<15}" + "".join(f" {v:+7.2f}" for v in high))
        else:
            error = 100.0 * (values / reference - 1.0)
            print(f"  {label:<15}" + "".join(f" {v:+7.2f}" for v in error))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_path.with_suffix(suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  wrote {path}")


def _label_along_curve(
    ax, curve, a, text, color, offset_points=5.0, rotation_delta=0.0
) -> None:
    """Write a label along a curve, rotated to its local slope.

    The rotation is read from the display transform, so the axis limits and the
    box aspect must both be final before this is called.
    """
    step = 0.02 * (_A_MAX - _A_MIN)
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
        rotation=np.degrees(np.arctan2(y1 - y0, x1 - x0)) + rotation_delta,
        rotation_mode="anchor",
        ha="center",
        va="bottom" if offset_points >= 0.0 else "top",
        zorder=20,
    )


def _index_markers(ax, y_text: float) -> None:
    """Vertical guides at the IceCube and atmospheric indices."""
    for a, text in ((_A_ICECUBE, "IceCube"), (_A_ATMOSPHERIC, "Atmospheric")):
        ax.axvline(a, color="0.55", lw=0.6, ls=":")
        ax.text(a - 0.22, y_text, text, color="0.35", rotation=90, va="top", ha="center")


def figure_absolute(
    curves: dict, envelope: tuple, a_grid: np.ndarray, out_dir: pathlib.Path
) -> None:
    """Figure 2 with the truncations replaced by the alternative kernels."""
    with plt.style.context(str(_STYLE)):
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        ax.fill_between(a_grid, *envelope, color="0.85", lw=0, zorder=0)

        ax.plot(a_grid, curves["soft"]["phi"](a_grid), color=colors[1], ls="--")
        ax.plot(a_grid, curves["steep"]["phi"](a_grid), color=colors[3], ls="-.")
        ax.plot(a_grid, curves["beta"](a_grid), color=colors[0])
        ax.plot(a_grid, curves["proposal"](a_grid), color="black", ls=":", lw=1.4, zorder=6)

        # Every kernel with the three moments passes through these points.
        for a in (1.0, 2.0, 3.0):
            ax.plot(
                a, float(curves["beta"](np.array([a]))[0]), "o", ms=3.5,
                mfc="white", mec="black", mew=0.8, zorder=7,
            )

        _index_markers(ax, 2.75)
        ax.set_xlim(_A_MIN, _A_MAX)
        ax.set_ylim(0.0, 2.8)
        ax.set_box_aspect(1.0)
        ax.set_xlabel(r"Effective spectral index $A=\gamma-\lambda-1$")
        ax.set_ylabel(r"$\Phi(A)$ [km$^{-1}$]")

        _label_along_curve(ax, curves["proposal"], 5.0, "PROPOSAL", "black", offset_points=4.0)
        _label_along_curve(ax, curves["beta"], 3.6, "Beta family", colors[0], offset_points=-4.0)
        _label_along_curve(
            ax, curves["steep"]["phi"], 9.3, r"$y^{-1.95}$ soft end", colors[3], offset_points=4.0
        )
        _label_along_curve(
            ax, curves["soft"]["phi"], 9.2, r"$y^{-1.3}$ soft end", colors[1], offset_points=-3.0
        )
        _label_along_curve(
            ax, lambda a: np.interp(a, a_grid, envelope[0]), 7.9,
            "Same three moments", "0.45", offset_points=-4.0,
        )

        _save(fig, out_dir / "90a_three_moment_families")
        plt.close(fig)


def figure_ratio(
    curves: dict,
    envelope3: tuple,
    envelope4: tuple,
    a_grid: np.ndarray,
    out_dir: pathlib.Path,
) -> None:
    """Every curve and both envelopes as a ratio to PROPOSAL."""
    reference = curves["proposal"](a_grid)
    with plt.style.context(str(_STYLE)):
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        ax.fill_between(a_grid, envelope3[0] / reference, envelope3[1] / reference,
                        color="0.88", lw=0, zorder=0)
        ax.fill_between(a_grid, envelope4[0] / reference, envelope4[1] / reference,
                        color="0.72", lw=0, zorder=1)
        ax.axhline(1.0, color="black", ls=":", lw=1.4, zorder=6)

        ratios = {
            "beta": curves["beta"](a_grid) / reference,
            "soft": curves["soft"]["phi"](a_grid) / reference,
            "steep": curves["steep"]["phi"](a_grid) / reference,
        }
        ax.plot(a_grid, ratios["soft"], color=colors[1], ls="--")
        ax.plot(a_grid, ratios["steep"], color=colors[3], ls="-.")
        ax.plot(a_grid, ratios["beta"], color=colors[0])

        _index_markers(ax, 1.145)
        ax.set_xlim(_A_MIN, _A_MAX)
        ax.set_ylim(0.8, 1.15)
        ax.set_box_aspect(1.0)
        ax.set_xlabel(r"Effective spectral index $A=\gamma-\lambda-1$")
        ax.set_ylabel(r"$\Phi(A)$ / PROPOSAL")

        def along(values):
            return lambda a: np.interp(a, a_grid, values)

        _label_along_curve(ax, along(ratios["beta"]), 7.0, "Beta family", colors[0],
                           offset_points=-4.0)
        _label_along_curve(ax, along(ratios["steep"]), 5.6, r"$y^{-1.95}$", colors[3],
                           offset_points=4.0)
        _label_along_curve(ax, along(ratios["soft"]), 7.5, r"$y^{-1.3}$", colors[1],
                           offset_points=-4.0)
        _label_along_curve(ax, along(envelope3[0] / reference), 8.6, "Three moments",
                           "0.45", offset_points=-3.0)
        _label_along_curve(ax, along(envelope4[1] / reference), 9.4, "Four moments", "0.3",
                           offset_points=3.0)

        _save(fig, out_dir / "90b_three_moment_ratio")
        plt.close(fig)


def figure_envelopes(
    proposal, envelopes: dict, a_grid: np.ndarray, out_dir: pathlib.Path
) -> None:
    """The envelopes for two to five moments, nested, over PROPOSAL.

    Parameters
    ----------
    proposal : callable
        PROPOSAL's ``Phi`` [km^-1] as a function of the spectral index.
    envelopes : dict
        Maps the number of matched moments to its ``(lower, upper)`` envelope
        [km^-1] on ``a_grid``.
    a_grid : np.ndarray
        Spectral indices.
    out_dir : pathlib.Path
        Directory for the figure.
    """
    reference = proposal(a_grid)
    names = {
        1: "One moment", 2: "Two moments", 3: "Three moments", 4: "Four moments",
        5: "Five moments",
    }
    # Label positions along each lower edge, in A, and their offsets in points.
    # The five-moment band is too thin to carry text and gets an arrow instead.
    placement = {2: (7.6, -3.0), 3: (9.0, -3.0), 4: (9.3, -3.0)}
    with plt.style.context(str(_STYLE)):
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        shade = {1: colors[5], 2: colors[2], 3: colors[0], 4: colors[1], 5: colors[3]}
        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        for n in sorted(envelopes):
            lower, upper = (edge / reference for edge in envelopes[n])
            # An opaque tint, drawn widest first, so each visible region carries
            # one color and the nested bands do not mix.
            tint = 0.55 + 0.45 * np.asarray(to_rgb(shade[n]))
            ax.fill_between(a_grid, lower, upper, color=tint, lw=0, zorder=n)
            ax.plot(a_grid, lower, color=shade[n], lw=0.7, zorder=n)
            ax.plot(a_grid, upper, color=shade[n], lw=0.7, zorder=n)
        ax.axhline(1.0, color="black", ls=":", lw=1.4, zorder=10)

        _index_markers(ax, 1.34)
        ax.set_xlim(_A_MIN, _A_MAX)
        ax.set_ylim(0.55, 1.35)
        ax.set_box_aspect(1.0)
        ax.set_xlabel(r"Effective spectral index $A=\gamma-\lambda-1$")
        ax.set_ylabel(r"$\Phi(A)$ / PROPOSAL")

        for n, (a_label, offset) in placement.items():
            edge = envelopes[n][0] / reference
            _label_along_curve(
                ax, lambda a, e=edge: np.interp(a, a_grid, e), a_label, names[n],
                shade[n], offset_points=offset,
            )
        # Both one-moment edges leave the axes beyond A ~ 3, so its label sits
        # inside its own fill.
        ax.text(8.0, 0.58, names[1], color=shade[1], ha="center", va="center", zorder=20)
        ax.annotate(
            names[5], xy=(7.6, 1.0), xytext=(4.6, 1.24), color=shade[5], ha="center",
            va="bottom", zorder=20,
            arrowprops={"arrowstyle": "-|>", "color": shade[5], "lw": 0.7,
                        "shrinkA": 1.0, "shrinkB": 0.0},
        )

        _save(fig, out_dir / "90c_moment_envelopes")
        plt.close(fig)


def figure_envelopes_absolute(
    proposal,
    closed_form,
    envelopes: dict,
    a_grid: np.ndarray,
    b_mu: float,
    d_mu: float,
    out_dir: pathlib.Path,
) -> None:
    """The envelopes for two to five moments, nested, as ``Phi(A)`` itself.

    Parameters
    ----------
    proposal : callable
        PROPOSAL's ``Phi`` [km^-1] as a function of the spectral index.
    closed_form : callable
        The paper's three-moment ``Phi`` [km^-1], Eq. (phithree).
    envelopes : dict
        Maps the number of matched moments to its ``(lower, upper)`` envelope
        [km^-1] on ``a_grid``.
    a_grid : np.ndarray
        Spectral indices.
    b_mu, d_mu : float
        First and second moments of the loss spectrum [km^-1], for the
        truncations drawn on top.
    out_dir : pathlib.Path
        Directory for the figure.
    """
    names = {
        1: "One moment", 2: "Two moments", 3: "Three moments", 4: "Four moments",
        5: "Five moments",
    }
    # Label positions along each lower edge, in A, and their offsets in points.
    # The four- and five-moment bands are too thin here to carry text and get
    # arrows from the empty upper left instead.
    placement = {2: (8.0, -3.0), 3: (9.2, -6.0)}
    with plt.style.context(str(_STYLE)):
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        shade = {1: colors[5], 2: colors[2], 3: colors[0], 4: colors[1], 5: colors[3]}
        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        for n in sorted(envelopes):
            lower, upper = envelopes[n]
            # An opaque tint, drawn widest first, so each visible region carries
            # one color and the nested bands do not mix.
            tint = 0.55 + 0.45 * np.asarray(to_rgb(shade[n]))
            ax.fill_between(a_grid, lower, upper, color=tint, lw=0, zorder=n)
            ax.plot(a_grid, lower, color=shade[n], lw=0.7, zorder=n)
            ax.plot(a_grid, upper, color=shade[n], lw=0.7, zorder=n)
        # The closed form sits on PROPOSAL, so it is drawn solid underneath the
        # dotted reference, as in example 37.
        ax.plot(a_grid, closed_form(a_grid), color=shade[3], lw=1.3, zorder=9)
        ax.plot(a_grid, proposal(a_grid), color="black", ls=":", lw=1.4, zorder=10)

        def first_order(a):
            return b_mu * np.asarray(a)

        def second_order(a):
            a = np.asarray(a)
            return a * b_mu - 0.5 * a * (a - 1.0) * d_mu

        truncation_color = "0.25"
        ax.plot(a_grid, first_order(a_grid), color=truncation_color, ls="--", lw=1.0, zorder=11)
        ax.plot(a_grid, second_order(a_grid), color=truncation_color, ls="-.", lw=1.0, zorder=11)

        _index_markers(ax, 3.35)
        ax.set_xlim(_A_MIN, _A_MAX)
        ax.set_ylim(0.0, 3.4)
        ax.set_box_aspect(1.0)
        ax.set_xlabel(r"Effective spectral index $A=\gamma-\lambda-1$")
        ax.set_ylabel(r"$\Phi(A)$ [km$^{-1}$]")

        # The one-moment lower edge is flat at b_mu, so its label starts at A = 4
        # just under it.
        ax.annotate(
            names[1], xy=(4.0, b_mu), xytext=(0.0, -3.0), textcoords="offset points",
            color=shade[1], ha="left", va="top", zorder=20,
        )
        _label_along_curve(ax, first_order, 8.1, r"$A\,b_\mu$", truncation_color,
                           offset_points=4.0)
        # The second-order curve bends over, so its label sits flat under the peak
        # with an arrow to the curve, as for PROPOSAL.
        a_second = 6.3
        ax.annotate(
            r"$A b_\mu - \frac{1}{2}A(A-1)d_\mu$",
            xy=(a_second, float(second_order(a_second))), xytext=(4.6, 0.62),
            color=truncation_color, ha="center", va="top", zorder=20,
            arrowprops={"arrowstyle": "-|>", "color": truncation_color, "lw": 0.7,
                        "shrinkA": 1.0, "shrinkB": 0.0},
        )
        a_closed = 5.2
        ax.annotate(
            "Used here",
            xy=(a_closed, float(closed_form(np.array([a_closed]))[0])),
            xytext=(2.2, 1.98), color=shade[3], ha="center", va="bottom", zorder=20,
            arrowprops={"arrowstyle": "-|>", "color": shade[3], "lw": 0.7,
                        "shrinkA": 1.0, "shrinkB": 0.0},
        )
        # PROPOSAL runs through every band, so its label sits in the empty upper
        # left with an arrow.
        a_proposal = 3.8
        ax.annotate(
            "PROPOSAL", xy=(a_proposal, float(proposal(np.array([a_proposal]))[0])),
            xytext=(1.5, 1.6), color="black", ha="center", va="bottom", zorder=20,
            arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 0.7,
                        "shrinkA": 1.0, "shrinkB": 0.0},
        )
        for n, (a_label, offset) in placement.items():
            edge = envelopes[n][0]
            _label_along_curve(
                ax, lambda a, e=edge: np.interp(a, a_grid, e), a_label, names[n],
                shade[n], offset_points=offset,
            )
        # Arrow targets: the five-moment band straddles PROPOSAL, and the part of
        # the four-moment band above it is the gap between the two upper edges.
        a_five, a_four = 7.0, 9.3
        targets = {
            5: (a_five, float(proposal(np.array([a_five]))[0])),
            4: (a_four, float(np.interp(
                a_four, a_grid, 0.5 * (envelopes[4][1] + envelopes[5][1])
            ))),
        }
        text_at = {5: (5.0, 2.85), 4: (5.4, 3.2)}
        for n in (5, 4):
            ax.annotate(
                names[n], xy=targets[n], xytext=text_at[n], color=shade[n], ha="center",
                va="bottom", zorder=20,
                arrowprops={"arrowstyle": "-|>", "color": shade[n], "lw": 0.7,
                            "shrinkA": 1.0, "shrinkB": 0.0},
            )

        _save(fig, out_dir / "90d_moment_envelopes_absolute")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    y = loss_spectrum_y_grid()
    dgamma_dy = proposal_loss_spectrum(args.energy_gev, y)["total"]
    mu = spectrum_moments(y, dgamma_dy, 5)
    b_mu, d_mu, t_mu = (float(v) for v in mu[:3])
    _, q_beta, p_beta = three_moment_loss_spectrum(b_mu, d_mu, t_mu)
    print(f"Muon energy {args.energy_gev:.3g} GeV")
    print("PROPOSAL moments [1/km]: " + ", ".join(f"mu_{n} = {v:.5f}" for n, v in enumerate(mu, 1)))
    print(f"Beta family: q = {float(q_beta):+.4f}, p = {float(p_beta):+.4f}, "
          f"soft end y^{float(q_beta) - 1.0:+.2f}")

    soft = alternative_kernel(mu, args.q_soft)
    steep = alternative_kernel(mu, args.q_steep)
    for name, kernel in (("softer", soft), ("steeper", steep)):
        print(f"Alternative, {name}: q = {kernel['q']:+.2f}, "
              f"a = ({', '.join(f'{v:+.4f}' for v in kernel['a'])}), "
              f"smallest dGamma/dy = {kernel['min_rate']:.3g} /km")

    curves = {
        "proposal": partial(phi_from_spectrum, y=y, dgamma_dy=dgamma_dy),
        "beta": lambda s: np.asarray(
            phi_eigenvalue_three_moment(s, b_mu=b_mu, d_mu=d_mu, t_mu=t_mu), dtype=float
        ),
        "soft": soft,
        "steep": steep,
    }

    nodes = envelope_nodes(args.n_nodes)
    print(f"\nSolving the envelopes on {nodes.size} nodes ...")
    table1 = moment_envelope(_A_TABLE, mu, 1, nodes)
    table2 = moment_envelope(_A_TABLE, mu, 2, nodes)
    table3 = moment_envelope(_A_TABLE, mu, 3, nodes)
    table4 = moment_envelope(_A_TABLE, mu, 4, nodes)
    table5 = moment_envelope(_A_TABLE, mu, 5, nodes)
    reference = curves["proposal"](_A_TABLE)
    print_table(
        {
            "Beta family": curves["beta"](_A_TABLE),
            f"q' = {args.q_soft:+.2f}": soft["phi"](_A_TABLE),
            f"q' = {args.q_steep:+.2f}": steep["phi"](_A_TABLE),
            "1-moment env.": table1,
            "2-moment env.": table2,
            "3-moment env.": table3,
            "4-moment env.": table4,
            "5-moment env.": table5,
        },
        reference,
    )

    high_a = _A_TABLE[_A_TABLE > 3.0]
    low2, high2, y_low, y_high = two_point_extremes(mu, high_a)
    in_table = _A_TABLE > 3.0
    print(f"\nTwo-point measures above A = 3: atoms at {{{y_low:.3f}, 1}} and {{0, {y_high:.3f}}}")
    print("  largest difference to the linear program: "
          f"{100.0 * np.max(np.abs(low2 / table3[0][in_table] - 1.0)):.3f}% (lower), "
          f"{100.0 * np.max(np.abs(high2 / table3[1][in_table] - 1.0)):.3f}% (upper)")

    a_grid = np.linspace(0.02, _A_MAX, args.n_a)
    envelope1 = moment_envelope(a_grid, mu, 1, nodes)
    envelope2 = moment_envelope(a_grid, mu, 2, nodes)
    envelope3 = moment_envelope(a_grid, mu, 3, nodes)
    envelope4 = moment_envelope(a_grid, mu, 4, nodes)
    envelope5 = moment_envelope(a_grid, mu, 5, nodes)
    between = (a_grid >= 1.0) & (a_grid <= 3.0)
    width = (envelope3[1] - envelope3[0]) / curves["proposal"](a_grid)
    print("  three-moment envelope width for 1 <= A <= 3: "
          f"at most {100.0 * width[between].max():.2f}%")

    print()
    figure_absolute(curves, envelope3, a_grid, args.out_dir)
    figure_ratio(curves, envelope3, envelope4, a_grid, args.out_dir)
    analytic = (b_mu * np.minimum(a_grid, 1.0), b_mu * np.maximum(a_grid, 1.0))
    print("  one-moment envelope against b_mu and A b_mu: largest difference "
          f"{100.0 * max(np.max(np.abs(e / a - 1.0)) for e, a in zip(envelope1, analytic)):.3f}%")
    all_envelopes = {1: envelope1, 2: envelope2, 3: envelope3, 4: envelope4, 5: envelope5}
    figure_envelopes(curves["proposal"], all_envelopes, a_grid, args.out_dir)
    figure_envelopes_absolute(
        curves["proposal"], curves["beta"], all_envelopes, a_grid, b_mu, d_mu, args.out_dir
    )


if __name__ == "__main__":
    main()
