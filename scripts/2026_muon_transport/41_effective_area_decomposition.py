"""Example 41 -- what builds the IceCube effective area: starting and entering events.

Example 32 shows that the parameter-free curve tracks the published DR2 table.
It does not show *where* the area comes from, and that is the question an analyst
asks when deciding whether the transport treatment matters to their own result.
Equation (18) answers it, because the target volume -- the generation "can" -- is
linear in the muon range,

.. math:: A_{\\rm eff} = n_N \\sum_k w_k \\sigma_{\\rm CC}(E_k)
    \\left[ A_{\\rm proj}(\\Omega, E_k)\\, L(E_k) + V_{\\rm det} \\right],

so splitting ``L`` splits ``A_eff`` additively. The bracket is already a split by
event topology: ``V_det`` counts **starting** events, whose neutrino vertex lies
inside the instrumented volume, and ``A_proj L`` counts **entering** events,
whose muon is born outside and arrives above threshold. Since ``L`` is a range to
threshold, entering covers through-going and stopping tracks alike. The soft
volume of Ref. [Palmisano:2026sid] is the entering term.

We split the entering term against the continuous-slowing-down range ``R_CSDA``,
the distance a muon would travel if it lost energy at the mean rate
``-dE/dx = a_mu + b_mu E`` with no fluctuations. Both ranges carry the same
ionization splice and the same running kernel, so they differ only in their
treatment of fluctuations (Table III). Three terms follow.

``starting``
    ``V_det``, the instrumented volume, fixed in size so its share falls as the
    range grows. It carries no transport at all: its energy dependence is the
    charged-current cross section and the Earth transmission alone.

``entering, mean loss``
    ``A_proj R_CSDA``, the can extension a muon earns by shedding energy
    continuously at the average rate. This is the term any treatment gets, and
    to leading order it is also what the drift-diffusion truncation gets, since
    that expansion carries ``Phi'(0) -> b_mu`` where the exact exponent carries
    ``Phi'(0) = <-ln(1-y)>``.

``entering, fluctuations``
    ``A_proj (L - R_CSDA)``, and it is **negative**. Radiative loss is
    multiplicative, so ``<-ln(1-y)> > <y>`` strictly by Jensen's inequality and
    the first-passage range is shorter than the continuous-slowing-down range.
    A mean-loss muon and a real muon shed the same energy on average, and the
    real one crosses threshold sooner, because one hard bremsstrahlung ends the
    track where the mean-loss picture keeps coasting.

The point of the figure is the size of the third term against the first: the
fluctuation correction removes several times more effective area than starting
events contribute, and the gap widens with energy because the lever arm
``ln(eps / E_thr)`` grows while ``V_det`` does not.

What this does and does not bias is worth stating. A Monte Carlo that propagates
muons through the can recovers the fluctuation term by construction, since it
samples the real loss law and an oversized can merely generates muons that die
before arriving. A forward model that sets the can length from the mean loss rate
does not, and overestimates its own effective area by the amount printed here.

Everything is averaged over the upgoing hemisphere with the PREM chords, the
neutral-current regeneration ladder, and the ``nu_tau -> tau -> mu`` channel of
example 32, whose functions this example imports rather than copies -- the three
curves are the same computation called with three ranges, so the decomposition
is exact by construction and the total reproduces example 32's ``first
principles`` curve line for line.

Usage
-----
    python examples/41_effective_area_decomposition.py
    python examples/41_effective_area_decomposition.py --threshold 1e4
"""

import argparse
import importlib.util
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.transport.soft_volume import (
    DEFAULT_MUON_THRESHOLD_GEV,
    muon_range_km,
    stochastic_muon_range_km,
)
from softpaws.transport.source import MEAN_INELASTICITY

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"
_EXAMPLE_32 = _HERE / "32_effective_area_comparison.py"

# Colour carries which term it is, and the total is black so it reads as the sum
# rather than as a fourth contribution. From the shared style's prop_cycle.
TERM_COLOR = {
    "starting": "#1b9e77",
    "entering, mean loss": "#7570b3",
    "entering, fluctuations": "#e7298a",
}
TERM_STYLE = {
    "starting": ":",
    "entering, mean loss": "--",
    "entering, fluctuations": "-.",
}

# Energies the printed table reports at, chosen to span the DR2 band.
REPORT_LOG10_E = (5.0, 6.0, 7.0, 8.0)


def load_example_32():
    """Import example 32 as a module, since its filename cannot be imported.

    Reusing its effective-area functions rather than copying them is deliberate.
    The two examples must decompose exactly the curve example 32 draws, and a
    private copy of the target volume or the regeneration ladder would drift
    from it silently.

    Returns
    -------
    module : ModuleType
        Example 32, with its module-level instrument constants bound.
    """
    spec = importlib.util.spec_from_file_location("_example_32", _EXAMPLE_32)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_MUON_THRESHOLD_GEV,
        help="Muon selection threshold [GeV].",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=_DEFAULT_OUT_DIR / "41_effective_area_decomposition",
        help="Output stem; .pdf and .png are both written.",
    )
    return parser.parse_args()


def effective_area(ex32, length_km: np.ndarray, threshold_gev: float) -> np.ndarray:
    """Upgoing-averaged effective area for a given muon range [cm^2].

    Both channels of Eq. (18), summed as example 32 sums them, since a
    through-going selection cannot tell them apart.

    Parameters
    ----------
    ex32 : ModuleType
        Example 32, from :func:`load_example_32`.
    length_km : np.ndarray
        Muon range to threshold [km], indexed by ``ex32.IC_LOG10_E``.
    threshold_gev : float
        Muon selection threshold [GeV].

    Returns
    -------
    aeff : np.ndarray
        Effective area [cm^2] on the ``ex32.IC_LOG10_E`` grid.
    """
    return ex32.ic_effective_area_regenerated(
        length_km, threshold_gev
    ) + ex32.ic_effective_area_tau_channel(length_km, threshold_gev)


def decompose(ex32, threshold_gev: float) -> dict[str, np.ndarray]:
    """Split the effective area into starting and entering contributions.

    The target volume is linear in the range, so evaluating the same effective
    area at three ranges -- zero, continuous-slowing-down, and first-passage --
    gives three terms that sum to the total identically. The zero-range call
    isolates the starting term, since it leaves only ``V_det``.

    Parameters
    ----------
    ex32 : ModuleType
        Example 32, from :func:`load_example_32`.
    threshold_gev : float
        Muon selection threshold [GeV].

    Returns
    -------
    terms : dict of str -> np.ndarray
        Keyed by ``"starting"``, ``"entering, mean loss"``,
        ``"entering, fluctuations"`` and ``"total"``, each [cm^2] on the
        ``ex32.IC_LOG10_E`` grid.

    Notes
    -----
    The zero-range call still credits ``V_det`` on rungs whose muon is born below
    threshold, exactly as Eq. (18) is implemented in example 32. That term is
    common to all three calls, so it cancels from the mean-loss and fluctuation
    differences and leaves the total equal to example 32's curve.
    """
    energy_mu = (1.0 - MEAN_INELASTICITY) * 10.0**ex32.IC_LOG10_E
    range_csda = muon_range_km(energy_mu, threshold_gev)
    range_first_passage = stochastic_muon_range_km(energy_mu, threshold_gev)

    aeff_geometric = effective_area(ex32, np.zeros_like(range_csda), threshold_gev)
    aeff_mean = effective_area(ex32, range_csda, threshold_gev)
    aeff_total = effective_area(ex32, range_first_passage, threshold_gev)

    return {
        "starting": aeff_geometric,
        "entering, mean loss": aeff_mean - aeff_geometric,
        "entering, fluctuations": aeff_total - aeff_mean,
        "total": aeff_total,
        "_range_csda_km": range_csda,
        "_range_first_passage_km": range_first_passage,
    }


def report(ex32, terms: dict[str, np.ndarray]) -> None:
    """Print the decomposition and the cost of ignoring fluctuations."""
    total = terms["total"]
    print("\n  Contributions to A_eff, upgoing-averaged, as a fraction of the total")
    print("  log10(E_nu/GeV)    starting   ent. mean   ent. fluct.   "
          "L/R_CSDA   A_eff too high by")
    for log10_e in REPORT_LOG10_E:
        i = int(np.argmin(np.abs(ex32.IC_LOG10_E - log10_e)))
        ratio = terms["_range_first_passage_km"][i] / terms["_range_csda_km"][i]
        # What a mean-loss propagation would report, against the correct answer.
        overestimate = (
            terms["starting"][i] + terms["entering, mean loss"][i]
        ) / total[i] - 1.0
        print(
            f"  {ex32.IC_LOG10_E[i]:>13.1f}   "
            f"{terms['starting'][i] / total[i]:>9.3f}   "
            f"{terms['entering, mean loss'][i] / total[i]:>9.3f}   "
            f"{terms['entering, fluctuations'][i] / total[i]:>11.3f}   "
            f"{ratio:>8.3f}   {overestimate:>16.1%}"
        )

    i_pev = int(np.argmin(np.abs(ex32.IC_LOG10_E - 6.0)))
    size = abs(terms["entering, fluctuations"][i_pev]) / terms["starting"][i_pev]
    print(
        f"\n  At 1 PeV the fluctuation term removes {size:.1f} times the effective area "
        f"starting events contribute."
    )


def make_figure(ex32, terms: dict[str, np.ndarray], out_path: pathlib.Path) -> None:
    """Draw the three terms and their sum on one panel.

    The fluctuation term is negative, so it is drawn as its magnitude and said
    to be negative in the legend. Absolute units are kept throughout, since the
    question the figure answers is how much effective area each term is worth.

    Parameters
    ----------
    ex32 : ModuleType
        Example 32, from :func:`load_example_32`.
    terms : dict of str -> np.ndarray
        From :func:`decompose`.
    out_path : pathlib.Path
        Output file; both ``.pdf`` and ``.png`` are written.
    """
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.2, 3.0))

        ax.plot(ex32.IC_LOG10_E, terms["total"], color="k", ls="-", lw=2.0, zorder=4,
                label="total")
        for name in ("starting", "entering, mean loss", "entering, fluctuations"):
            label = name if "fluct" not in name else name + " (negative)"
            ax.plot(ex32.IC_LOG10_E, np.abs(terms[name]), color=TERM_COLOR[name],
                    ls=TERM_STYLE[name], lw=1.2, zorder=3, label=label)

        ax.set_yscale("log")
        ax.set_xlim(4.0, 8.0)
        # The lower bound clips the cusp where the two ranges cross near
        # threshold, which is a zero of the difference and not a feature.
        ax.set_ylim(3.0e4, 3.0e7)
        ax.set_xlabel(r"$\log_{10}(E_\nu\,/\,\mathrm{GeV})$")
        ax.set_ylabel(r"contribution to $A_{\rm eff}$ [cm$^2$]")
        ax.legend(loc="lower right", fontsize=6, frameon=False, handlelength=2.4)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png"):
            path = out_path.with_suffix(suffix)
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {path.resolve()}")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    print("Loading example 32's effective-area construction ...")
    ex32 = load_example_32()

    print("Decomposing the upgoing-averaged effective area ...")
    terms = decompose(ex32, args.threshold)

    report(ex32, terms)
    make_figure(ex32, terms, args.out)


if __name__ == "__main__":
    main()
