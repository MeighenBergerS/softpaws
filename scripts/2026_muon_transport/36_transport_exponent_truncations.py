"""Example 36 -- the transport exponent against its truncations.

Draws the figure that belongs with Appendix F (Table~\\ref{tab:fp}) and
Section~V of ``docs/main.tex``: the exact Levy exponent ``Phi(A)`` laid against
the drift-only and second-order forms that the earlier drift-diffusion
literature uses.

The exponent is the whole content of the exact solution. A power-law source
excites one mode ``s = A = gamma - lambda - 1``, the collision operator is
diagonal on it, and the transport problem collapses to multiplication by the
single number ``Phi(A)``. Expanding the bracket ``[1 - (1-y)^A]`` binomially
gives ``Phi(A) = A b_mu - A(A-1) d_mu / 2 + ...``, so the drift limit and the
Fokker-Planck form are the first and second partial sums of that series -- not
independent approximations, but truncations with a stated domain.

Three features of the figure are the argument.

**Tangency at A = 1 and A = 2.** The binomial series terminates at integer
``A``, so the truncations are not merely close there, they are exact:
``Phi(1) = b_mu`` and ``Phi(2) = 2 b_mu - d_mu``. The IceCube diffuse index sits
at ``A = 0.98``, a fifth of a percent from the first of those points, which is
why the drift-diffusion approximation worked as well as it did. It was a
controlled expansion in ``(A - 1)``, and the exact form says so.

**Turnover at A_die.** The second-order form is a downward parabola. Its second
root ``A_die = 1 + 2 b_mu / d_mu`` is near ten for water: past it the truncated
exponent is negative, which would mean muons gaining energy as they propagate.
The exact ``Phi(A)`` keeps growing, monotonically and without bound, because it
is a Bernstein function. Nothing physical happens at ``A_die``; it is the edge
of the truncation's domain, and the point of drawing it is that the edge can now
be located.

**Phi < 0 for A < 0.** Negative spectral indices are not pathological here. The
exponent turns negative, the mode grows rather than attenuates, and the
first-passage picture of Section V inverts along with it. The closed form stays
finite down to ``A > -(1 + p)``, where the bremsstrahlung tail of the kernel
finally binds.

Calibration follows Table F.1: PROPOSAL water coefficients near 20 TeV,
``b_mu = 0.349`` and ``d_mu = 0.079 km^-1``. The drawn exact curve is the
two-moment family ``dGamma/dy = kappa (1-y)^p / y``, whose one-digamma closed
form is the ``Phi(A)`` of the main text; the table printed alongside the figure
adds the three-moment calibration ``kappa y^(q-1) (1-y)^p`` that Table F.1
itself uses. The two agree by construction at ``A = 0, 1, 2`` and separate
slowly beyond, by a few percent at ``A = 4``, because the third moment controls
how much weight the kernel keeps near ``y -> 1``. That difference touches none
of the three features above.

Usage
-----
    python examples/36_transport_exponent_truncations.py
    python examples/36_transport_exponent_truncations.py --energy-gev 1e6
    python examples/36_transport_exponent_truncations.py --out-dir /path/to/figures

Writes one figure, ``36_transport_exponent``, and prints the Table F.1
comparison it is drawn from.
"""

import argparse
import pathlib
from collections.abc import Callable
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from softpaws.transport.coefficients import (
    diffusion_coefficient,
    drift_coefficient,
    third_moment_coefficient,
)
from softpaws.transport.eigenvalue import (
    phi_drift,
    phi_eigenvalue,
    phi_eigenvalue_three_moment,
    phi_fokker_planck,
    two_moment_loss_spectrum,
)
from softpaws.utils.constants import RHO_WATER_G_CM3

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"

# Table F.1 is quoted at the tabulated water values near 20 TeV rather than at
# the 1 PeV reference of the other tables: the three columns scale almost
# linearly with b_mu, so their comparison is insensitive to the choice, while
# A_die -- a ratio of the two coefficients -- is not.
REFERENCE_ENERGY_GEV = 2.0e4

A_ICECUBE = 0.98  # gamma = 2.38, lambda = 0.4, Sec. V A
A_MIN, A_MAX = -1.2, 11.0
TABLE_F1_INDICES = (0.14, 0.50, 0.98, 1.00, 2.00, 3.00, 4.00, 6.00)


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
        default=REFERENCE_ENERGY_GEV,
        help="Muon energy at which the transport coefficients are read [GeV]; "
        "the Table F.1 convention is 20 TeV.",
    )
    return parser.parse_args()


def transport_moments(energy_gev: float) -> tuple[float, float, float]:
    """Read ``b_mu``, ``d_mu`` and ``t_mu`` for water at one muon energy.

    Parameters
    ----------
    energy_gev : float
        Muon energy [GeV].

    Returns
    -------
    b_mu, d_mu, t_mu : float
        First three ``y``-moments of the PROPOSAL loss spectrum [km^-1].
    """
    moments = (drift_coefficient, diffusion_coefficient, third_moment_coefficient)
    return tuple(float(np.ravel(moment(energy_gev, RHO_WATER_G_CM3))[0]) for moment in moments)


def report_table_f1(b_mu: float, d_mu: float, t_mu: float) -> None:
    """Print the Table F.1 columns at the calibration this figure uses."""
    kappa, p = (float(np.ravel(x)[0]) for x in two_moment_loss_spectrum(b_mu, d_mu))
    a_die = 1.0 + 2.0 * b_mu / d_mu
    print(
        f"b_mu = {b_mu:.4f} /km   d_mu = {d_mu:.4f} /km   t_mu = {t_mu:.4f} /km\n"
        f"d/b  = {d_mu / b_mu:.3f}   kappa = {kappa:.3f} /km   p = {p:.3f}   "
        f"A_die = {a_die:.2f}"
    )
    print(f"\n{'A':>6}{'Phi_2mom':>10}{'Phi_3mom':>10}{'A b_mu':>9}{'2nd order':>11}")
    for a in TABLE_F1_INDICES:
        print(
            f"{a:6.2f}{float(phi_eigenvalue(a, b_mu, d_mu)):10.3f}"
            f"{float(phi_eigenvalue_three_moment(a, b_mu, d_mu, t_mu)):10.3f}"
            f"{float(phi_drift(a, b_mu)):9.3f}"
            f"{float(phi_fokker_planck(a, b_mu, d_mu)):11.3f}"
        )


def _save(fig: plt.Figure, out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_path.with_suffix(suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {path.resolve()}")
    plt.close(fig)


def label_along_curve(
    ax: plt.Axes,
    curve: Callable[[np.ndarray], np.ndarray],
    a: float,
    text: str,
    color: str,
) -> None:
    """Write ``text`` on a curve at ``A = a``, rotated to follow it.

    The rotation is read off the curve in *display* coordinates, so it tracks the
    drawn slope rather than the mathematical one and stays right whatever the
    figure's aspect ratio and axis limits are. Call this after the limits are
    final.

    Parameters
    ----------
    ax : plt.Axes
        Axes holding the curve.
    curve : callable
        The curve, as a function of the spectral index ``A``.
    a : float
        Spectral index at which to place the label.
    text : str
        Label, sitting just above the curve.
    color : str
        Label color, matched to the curve it names.
    """
    step = 0.02 * (A_MAX - A_MIN)
    ends = np.array([a - step, a + step])
    (x0, y0), (x1, y1) = ax.transData.transform(np.column_stack([ends, curve(ends)]))
    ax.annotate(
        text,
        xy=(a, float(curve(np.array([a]))[0])),
        # Clear of the line by a few points: the baseline follows the tangent,
        # but subscripts and fractions hang below it.
        xytext=(0.0, 5.0),
        textcoords="offset points",
        color=color,
        fontsize=7,
        rotation=np.degrees(np.arctan2(y1 - y0, x1 - x0)),
        rotation_mode="anchor",
        ha="center",
        va="bottom",
    )


def figure_exponent(
    b_mu: float,
    d_mu: float,
    out_path: pathlib.Path,
) -> None:
    """The exact exponent against the drift-only and second-order truncations."""
    a_grid = np.linspace(A_MIN, A_MAX, 600)
    a_die = 1.0 + 2.0 * b_mu / d_mu

    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        c_exact, c_drift, c_second = colors[0], colors[2], colors[1]

        # Where the exponent changes sign the mode grows instead of attenuating.
        ax.axhspan(-1.0, 0.0, color="0.92", lw=0, zorder=0)
        ax.axhline(0.0, color="0.4", lw=0.6, zorder=1)

        exact = partial(phi_eigenvalue, b_mu=b_mu, d_mu=d_mu)
        drift = partial(phi_drift, b_mu=b_mu)
        second = partial(phi_fokker_planck, b_mu=b_mu, d_mu=d_mu)

        ax.plot(a_grid, exact(a_grid), color=c_exact)
        ax.plot(a_grid, drift(a_grid), color=c_drift, ls="--")
        ax.plot(a_grid, second(a_grid), color=c_second, ls="-.")

        # The binomial series terminates at integer A, so the truncations touch
        # the exact curve there rather than merely approaching it.
        for a in (1.0, 2.0):
            ax.plot(
                a,
                float(exact(a)),
                "o",
                ms=3.5,
                mfc="white",
                mec=c_exact,
                mew=1.0,
                zorder=5,
            )
        ax.annotate(
            r"$\Phi(1)=b_\mu$",
            xy=(1.0, float(exact(1.0))),
            xytext=(1.35, -0.30),
            fontsize=6.8,
            arrowprops=dict(arrowstyle="-", lw=0.5, color="0.4"),
        )
        ax.annotate(
            r"$\Phi(2)=2b_\mu-d_\mu$",
            xy=(2.0, float(exact(2.0))),
            xytext=(2.6, 0.12),
            fontsize=6.8,
            arrowprops=dict(arrowstyle="-", lw=0.5, color="0.4"),
        )

        ax.axvline(A_ICECUBE, color="0.55", lw=0.6, ls=":")
        ax.text(
            A_ICECUBE - 0.22,
            1.90,
            rf"IceCube, $A={A_ICECUBE:g}$",
            fontsize=6.5,
            color="0.35",
            rotation=90,
            va="top",
            ha="center",
        )

        ax.plot(a_die, 0.0, "v", ms=4, color=c_second, zorder=5)
        ax.annotate(
            rf"$A_{{\mathrm{{die}}}}={a_die:.1f}$",
            xy=(a_die, 0.0),
            xytext=(7.4, -0.28),
            fontsize=6.8,
            color=c_second,
        )
        ax.annotate(r"$\Phi<0$: growing mode", xy=(-1.05, -0.72), fontsize=6.5, color="0.35")

        ax.set_xlim(A_MIN, A_MAX)
        ax.set_ylim(-1.0, 2.0)
        ax.set_xlabel(r"effective spectral index $A=\gamma-\lambda-1$")
        ax.set_ylabel(r"$\Phi(A)$ [km$^{-1}$]")

        # Each curve carries its own name, in its own color and along its own
        # slope; a legend would have to sit somewhere, and everywhere it could
        # sit is either a curve or an annotation. The limits must be final
        # first, since the rotations are read from the display transform.
        label_along_curve(ax, drift, 3.5, r"$A\,b_\mu$", c_drift)
        label_along_curve(ax, exact, 6.0, r"exact $\Phi(A)$", c_exact)
        label_along_curve(
            ax, second, 8.0, r"$A b_\mu - \frac{1}{2}A(A-1)d_\mu$", c_second
        )

        _save(fig, out_path)


def main() -> None:
    args = parse_args()
    b_mu, d_mu, t_mu = transport_moments(args.energy_gev)
    print(f"PROPOSAL water coefficients at E_mu = {args.energy_gev:.3g} GeV")
    report_table_f1(b_mu, d_mu, t_mu)
    print()
    figure_exponent(b_mu, d_mu, args.out_dir / "36_transport_exponent")


if __name__ == "__main__":
    main()
