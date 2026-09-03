"""Example 78 -- the two instrument numbers of example 77 on one plane.

Example 77 fixes the physics (``b_scale = 1``, ``lam`` at BGR18, ``eps_0``
at the selection plateau) and refits each published effective area with
only the threshold and the reach free. This example puts the four
two-parameter posteriors on one corner, in example 56's layout, so the
question of whether the sites agree can be read off the contours. The
diagonal carries the marginals, the lower-left cell the
``(log10 E_thr, Lambda)`` plane at 68% and 95%, and the upper-right cell
the legend. Reference lines mark the derived reach of App. F (59 m in
ice, 68 m in water), ``Lambda = 0``, and the threshold the DR2 smearing
matrix implies.

Usage
-----
    python examples/77_reduced_response_fit.py --sigma 0.05
    python examples/78_reduced_response_plane.py
    python examples/78_reduced_response_plane.py --chains path/to/77_chains.npz --tag sigma05
"""

import argparse
import importlib.util
import pathlib

import corner
import matplotlib.pyplot as plt
import numpy as np

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"


def load_example(stem: str, name: str):
    """Import a sibling example module by file stem."""
    spec = importlib.util.spec_from_file_location(name, _HERE / stem)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EX77 = load_example("77_reduced_response_fit.py", "_example_77")
_EX56 = _EX77._EX56
_EX33 = _EX77._EX33

SITES = ("IceCube", "ARCA230", "P-ONE", "TRIDENT")
LEVELS = {"IceCube": "analysis", "ARCA230": "trigger", "P-ONE": "trigger", "TRIDENT": "6 deg cut"}
LABELS = [r"$\log_{10}(E_{\mathrm{thr}}/\mathrm{GeV})$", r"$\Lambda$ [m]"]
STYLES = ["-", "--", "-", ":"]

#: Reach each site's own optics predict [m], as (low, high): the effective
#: attenuation length at 400 nm, the shorter of the absorption length and the
#: diffusive length sqrt(abs * scat / 3). IceCube: absorption 110-200 m with
#: effective scattering 25-50 m (dust-layer average to clearest ice).
#: ARCA230: Capo Passero absorption ~50 m at 470 nm (ANTARES site 60 m),
#: scaled to 400 nm. P-ONE: STRAW 28 m at 450 nm (35 m in the proposal),
#: scaled. TRIDENT: measured effective attenuation 15-27 m over 405-460 nm.
PREDICTED_M = {"IceCube": (30.0, 59.0), "ARCA230": (41.0, 50.0), "P-ONE": (24.0, 30.0),
               "TRIDENT": (15.0, 27.0)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chains", type=pathlib.Path, default=_DEFAULT_OUT_DIR / "77_chains.npz")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR)
    return parser.parse_args()


def load_chains(path: pathlib.Path) -> tuple[dict, float]:
    """Return ``{site: (n, 2) array of [log10 E_thr, Lambda in m]}`` and the error model."""
    data = np.load(path)
    sigma = float(data["sigma"])
    chains = {}
    for site in SITES:
        chain = data[f"{site}_2p_chain"]
        chains[site] = np.column_stack([chain[:, 0], 1.0e3 * chain[:, 1]])
    return chains, sigma


def summarize(chains: dict, sigma: float) -> None:
    print(f"Two-parameter posteriors, error model {100*sigma:.0f}% per node")
    print(f"{'site':>8}  {'E_thr [GeV]':>22}  {'Lambda [m]':>22}")
    for site, chain in chains.items():
        lo_e, med_e, hi_e = 10 ** np.percentile(chain[:, 0], [16, 50, 84])
        lo_l, med_l, hi_l = np.percentile(chain[:, 1], [16, 50, 84])
        print(f"{site:>8}  {med_e:8.0f} [{lo_e:6.0f}, {hi_e:6.0f}]  "
              f"{med_l:8.1f} [{lo_l:6.1f}, {hi_l:6.1f}]")
    # Pairwise separation of the reach medians in units of the combined 68% half-widths.
    print("Reach separation between sites, in sigma:")
    names = list(chains)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            qa = np.percentile(chains[a][:, 1], [16, 50, 84])
            qb = np.percentile(chains[b][:, 1], [16, 50, 84])
            width = np.hypot(0.5 * (qa[2] - qa[0]), 0.5 * (qb[2] - qb[0]))
            print(f"  {a:>8} vs {b:<8} {abs(qa[1] - qb[1]) / width:5.1f}")


def make_figure(chains: dict, sigma: float, out_path: pathlib.Path) -> None:
    ranges = []
    for k in range(2):
        lo = min(np.percentile(c[:, k], 0.5) for c in chains.values())
        hi = max(np.percentile(c[:, k], 99.5) for c in chains.values())
        pad = 0.08 * (hi - lo)
        ranges.append((lo - pad, hi + pad))
    # The reach axis is held at a fixed window: a wide posterior would otherwise
    # set the frame and the predicted strips would not be readable.
    ranges[1] = (max(ranges[1][0], -60.0), min(max(ranges[1][1], 65.0), 120.0))


    rc = {"xtick.labelsize": 8, "ytick.labelsize": 8, "axes.labelsize": 8, "font.size": 8}
    with plt.style.context(str(_STYLE)), plt.rc_context(rc):
        fig, _ = plt.subplots(2, 2, figsize=(4.6, 4.6))
        for row, site in enumerate(SITES):
            color = _EX56.COLORS[site]
            base = np.array(plt.matplotlib.colors.to_rgb(color))
            filled = False
            fills = [(*base, 0.0), (*base, 0.12), (*base, 0.28)]
            corner.corner(
                chains[site], labels=LABELS, range=ranges, color=color, fig=fig,
                plot_datapoints=False, plot_density=False, levels=(0.68, 0.95),
                fill_contours=filled, contourf_kwargs={"colors": fills} if filled else None,
                contour_kwargs={"linewidths": 1.1, "linestyles": STYLES[row]},
                hist_kwargs={"density": True, "lw": 1.3, "ls": STYLES[row], "histtype": "step"},
                label_kwargs={"fontsize": 8}, smooth=0.8, no_fill_contours=not filled)

        axes = np.array(fig.axes[:4]).reshape((2, 2))
        plane, thr_ax, reach_ax = axes[1, 0], axes[0, 0], axes[1, 1]
        # The reach each site's measured optics predict, as a band in the
        # site's colour across the whole threshold range.
        for site in SITES:
            lo, hi = PREDICTED_M[site]
            plane.axhspan(lo, hi, facecolor=_EX56.COLORS[site], edgecolor="none", alpha=0.16,
                          zorder=0)
        plane.axhline(0.0, color="0.6", lw=0.7, zorder=0)
        thr_ax.axvline(_EX33.SMEARING_LOG10_E_THR, color="0.35", lw=0.9, ls=(0, (1, 2)), zorder=0)
        plane.axvline(_EX33.SMEARING_LOG10_E_THR, color="0.35", lw=0.9, ls=(0, (1, 2)), zorder=0)

        pct = r"\%" if plt.rcParams["text.usetex"] else "%"
        handles = [plt.Line2D([], [], color=_EX56.COLORS[s], lw=1.6, ls=STYLES[i],
                              label=f"{s} ({LEVELS[s]})") for i, s in enumerate(SITES)]
        handles.append(plt.matplotlib.patches.Patch(facecolor="0.5", alpha=0.3, lw=0,
                                                    label="Predicted from optics"))
        legend_ax = axes[0, 1]
        legend_ax.axis("off")
        legend_ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8,
                         handlelength=1.8, borderaxespad=0.0,
                         title=f"Error model {100*sigma:.0f}{pct} per node", title_fontsize=8)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png"):
            path = out_path.with_suffix(suffix)
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"Figure saved to: {path.resolve()}")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    chains, sigma = load_chains(args.chains)
    summarize(chains, sigma)
    tag = f"_{args.tag}" if args.tag else ""
    make_figure(chains, sigma, args.out_dir / f"78_reduced_response_plane{tag}.pdf")


if __name__ == "__main__":
    main()
