"""Example 43 -- implied selection efficiency, HESE starting events against DR2 tracks.

Examples 32 and 42 each divide a published effective area by a first-principles
one, and each reads the quotient as the efficiency of a selection the model does
not attempt to describe. They do it on samples that share almost nothing: DR2 is
a through-going northern-sky track selection dominated by the **entering** term
of Eq. (18), and HESE is an all-sky veto-defined starting sample that is purely
the ``V_det`` term. Different topology, different threshold variable, different
sky, different backgrounds.

Putting the two quotients on one axis asks whether they behave the same way. The
question matters because the two ratios have different content. For DR2 the
quotient tests the entering term, so the transport is inside it. For HESE the
quotient tests the starting term, where no transport enters at all. If the two
curves plateau at the same value and turn on the same way, then whatever the
first-principles construction is getting right, it is not getting right by an
accident of the through-going geometry.

For this comparison the HESE prediction uses the **fiducial** volume the veto
leaves rather than the full instrumented km^3 (see
:data:`~examples.42_hese_starting_events.HESE_FIDUCIAL_FRACTION`), because the
DR2 quotient is already against a volume the selection can use. With the
geometric factor removed from both, what is left on each curve is a selection
efficiency and nothing else, and the two become comparable.

Two caveats on the comparison, neither of which the ratio can hide:

* the sky coverage differs, since DR2 is upgoing and HESE all-sky. The report
  below also prints the HESE quotient restricted to the upgoing hemisphere, and
  the two agree, which is what the isotropy of the HESE selection predicts.
* the HESE fiducial fraction is estimated from the veto geometry quoted in the
  release paper and carries perhaps 10% of its own, so the **level** of the HESE
  curve is uncertain by that much where its **shape** is not.

``nu_tau`` is left out. Our ladder credits regenerated rungs that the sample's
60 TeV deposited-energy threshold rejects, and example 42 shows the resulting
zenith drift, so the tau quotient is not a clean efficiency.

Usage
-----
    python examples/43_implied_efficiency_hese_vs_dr2.py
"""

import argparse
import importlib.util
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.data.hese import effective_area_cm2, load_hese_mc

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"
_DEFAULT_DATA_DIR = _HERE.parent / "src" / "softpaws" / "data" / "dataverse_files"

CURVE_COLOR = {
    "DR2 tracks, entering": "#1b9e77",
    r"HESE starting, $\nu_\mu$": "#7570b3",
    r"HESE starting, $\nu_e$": "#e7298a",
}
CURVE_STYLE = {
    "DR2 tracks, entering": "-",
    r"HESE starting, $\nu_\mu$": "--",
    r"HESE starting, $\nu_e$": "-.",
}

# Only these two are drawn. The nu_e quotient is still computed and reported,
# where in the figure it would say the same thing the nu_mu curve says, offset
# by the inelasticity, and that is a sentence rather than a line.
PLOTTED = ("DR2 tracks, entering", r"HESE starting, $\nu_\mu$")

#: Inline labels replacing the legend: text, abscissa, and which side of the
#: curve to sit on (``+1`` above, ``-1`` below). The offset is applied
#: perpendicular to the curve in points, so a steep curve does not swallow it.
INLINE_LABEL = {
    "DR2 tracks, entering": ("DR2", 5.5, -1),
    r"HESE starting, $\nu_\mu$": ("HESE", 5.5, -1),
}
LABEL_OFFSET_POINTS = 9.0

#: The HESE analysis threshold, a cut on reconstructed deposited energy [GeV].
HESE_DEPOSITED_CUT_GEV = 6.0e4

#: First colour of the shared style's cycle, used for the gap to the ceiling.
STYLE_PINK = "#e7298a"

ALL_SKY = (-1.0, 1.0)
UPGOING = (-1.0, 0.0)


def load_example(name: str):
    """Import a numbered example as a module, since the filename cannot be imported."""
    spec = importlib.util.spec_from_file_location(f"_{name}", _HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir", type=pathlib.Path, default=_DEFAULT_DATA_DIR,
        help="Directory holding the IceTracks-DR2 release.",
    )
    parser.add_argument("--threshold", type=float, default=1.0e3,
                        help="Muon selection threshold for the DR2 model [GeV].")
    parser.add_argument(
        "--out", type=pathlib.Path,
        default=_DEFAULT_OUT_DIR / "43_implied_efficiency_hese_vs_dr2",
        help="Output stem; .pdf and .png are both written.",
    )
    return parser.parse_args()


def hese_efficiency(
    ex42, flavour: str, cos_zenith_range: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Published-over-model quotient for the HESE starting sample.

    Parameters
    ----------
    ex42 : ModuleType
        Example 42, from :func:`load_example`.
    flavour : {"e", "mu"}
        Neutrino flavour.
    cos_zenith_range : tuple of float
        Band edges in ``cos(zenith)``.

    Returns
    -------
    log10_energy : np.ndarray
        Bin centres, ``log10(E_nu / GeV)``.
    efficiency : np.ndarray
        Ratio of the release's effective area to ours on the fiducial volume.
    """
    mc = load_hese_mc()
    bins = ex42.ENERGY_BINS_GEV
    centre = np.sqrt(bins[1:] * bins[:-1])
    published, _ = effective_area_cm2(
        mc, bins, flavour=flavour, interaction="cc", cos_zenith_range=cos_zenith_range
    )
    ours = ex42.predicted_starting_area_cm2(
        centre, cos_zenith_range, flavour, volume_cm3=ex42.HESE_FIDUCIAL_CM3
    )
    return np.log10(centre), published / ours


def dr2_efficiency(ex32, data_dir: pathlib.Path, threshold_gev: float):
    """Published-over-model quotient for the DR2 through-going table.

    Parameters
    ----------
    ex32 : ModuleType
        Example 32, from :func:`load_example`.
    data_dir : pathlib.Path
        Directory holding the IceTracks-DR2 release.
    threshold_gev : float
        Muon selection threshold [GeV].

    Returns
    -------
    log10_energy : np.ndarray
        ``log10(E_nu / GeV)`` grid.
    efficiency : np.ndarray
        Ratio of the published table to the parameter-free model.
    """
    curves, _ = ex32.icecube_curves(data_dir, threshold_gev)
    return ex32.IC_LOG10_E, curves["published"] / curves["first principles"]


def report(curves: dict[str, tuple[np.ndarray, np.ndarray]], upgoing: dict) -> None:
    """Print the plateau of each quotient and the sky-coverage check."""
    print("\n  Implied selection efficiency, plateau above 300 TeV")
    for name, (log10_e, efficiency) in curves.items():
        band = (log10_e >= np.log10(3.0e5)) & (log10_e <= 7.0)
        values = efficiency[band]
        values = values[np.isfinite(values) & (values > 0)]
        print(f"    {name:32s}: {values.mean():.3f}  "
              f"(rms {np.std(np.log10(values)):.3f} dex across the band)")

    print("\n  Sky-coverage check, HESE all-sky against HESE upgoing only")
    for flavour, (all_sky, up) in upgoing.items():
        band = np.isfinite(all_sky) & np.isfinite(up) & (all_sky > 0) & (up > 0)
        print(f"    nu_{flavour:4s}: mean ratio upgoing/all-sky "
              f"{np.mean(up[band] / all_sky[band]):.3f}")


def curve_angle_deg(ax, log10_e: np.ndarray, efficiency: np.ndarray, x0: float) -> float:
    """Screen angle of a curve at ``x0``, in degrees.

    A data-space slope is not the angle a label should be rotated by, since the
    axes are not square in data units. This maps two points either side of
    ``x0`` through ``transData`` and takes the angle there, so the label lies
    along the curve as drawn.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes the curve is drawn on, already laid out.
    log10_e, efficiency : np.ndarray
        The curve.
    x0 : float
        Abscissa to take the angle at.

    Returns
    -------
    angle_deg : float
        Rotation for :meth:`~matplotlib.axes.Axes.text` [deg].
    """
    span = 0.12
    pair = np.array(
        [[x, np.interp(x, log10_e, efficiency)] for x in (x0 - span, x0 + span)]
    )
    (x1, y1), (x2, y2) = ax.transData.transform(pair)
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))


def make_figure(curves: dict, out_path: pathlib.Path) -> None:
    """Draw the two quotients on one panel, labelled inline.

    Parameters
    ----------
    curves : dict
        Curve name -> (``log10(E_nu / GeV)``, efficiency).
    out_path : pathlib.Path
        Output file; both ``.pdf`` and ``.png`` are written.
    """
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.4, 3.0))
        # The ceiling. A published area above it would falsify the construction.
        ax.axhline(1.0, color="0.6", lw=0.8, ls=":", zorder=1)

        # Below the deposited-energy cut the HESE quotient is a statement about
        # the threshold and not about the model, so shade it out.
        hese_colour = CURVE_COLOR[r"HESE starting, $\nu_\mu$"]
        cut = np.log10(HESE_DEPOSITED_CUT_GEV)
        ax.axvspan(4.0, cut, color=hese_colour, alpha=0.1, lw=0, zorder=0)
        ax.text(0.5 * (4.0 + cut), 1.1, "HESE cut", ha="center", va="center",
                fontsize=6, color=hese_colour, zorder=4)

        # The gap between the ceiling and the DR2 quotient is what the selection
        # throws away, so shade it and say so.
        dr2_x, dr2_y = curves["DR2 tracks, entering"]
        ax.fill_between(dr2_x, dr2_y, 1.0, color=STYLE_PINK, alpha=0.05, lw=0, zorder=0)

        for name in PLOTTED:
            log10_e, efficiency = curves[name]
            ax.plot(log10_e, efficiency, color=CURVE_COLOR[name], ls=CURVE_STYLE[name],
                    lw=1.3, zorder=3)

        ax.set_xlim(4.0, 8.0)
        ax.set_ylim(0.0, 1.25)
        ax.set_xlabel(r"$\log_{10}(E_\nu\,/\,\mathrm{GeV})$")
        ax.set_ylabel(r"published $/$ first principles")

        # Inline labels replace the legend, so the angle has to be taken after
        # the axes are laid out and the transform is final.
        fig.canvas.draw()
        for name, (text, x0, side) in INLINE_LABEL.items():
            log10_e, efficiency = curves[name]
            y0 = float(np.interp(x0, log10_e, efficiency))
            angle = curve_angle_deg(ax, log10_e, efficiency, x0)
            normal = np.radians(angle + side * 90.0)
            ax.annotate(
                text, xy=(x0, y0),
                xytext=(LABEL_OFFSET_POINTS * np.cos(normal),
                        LABEL_OFFSET_POINTS * np.sin(normal)),
                textcoords="offset points", color=CURVE_COLOR[name], fontsize=7,
                ha="center", va="center", zorder=4,
                rotation=angle, rotation_mode="anchor",
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png"):
            path = out_path.with_suffix(suffix)
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {path.resolve()}")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    print("Loading examples 32 and 42 ...")
    ex32 = load_example("32_effective_area_comparison")
    ex42 = load_example("42_hese_starting_events")

    print("Building the DR2 quotient ...")
    curves = {"DR2 tracks, entering": dr2_efficiency(ex32, args.data_dir, args.threshold)}

    print("Building the HESE quotient ...")
    upgoing = {}
    for flavour, label in (("mu", r"HESE starting, $\nu_\mu$"),
                           ("e", r"HESE starting, $\nu_e$")):
        curves[label] = hese_efficiency(ex42, flavour, ALL_SKY)
        upgoing[flavour] = (curves[label][1], hese_efficiency(ex42, flavour, UPGOING)[1])

    report(curves, upgoing)
    make_figure(curves, args.out)


if __name__ == "__main__":
    main()
