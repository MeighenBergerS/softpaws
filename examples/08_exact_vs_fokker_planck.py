"""Example 08 — exact eigenvalue vs. Fokker-Planck soft volume.

Puts the exact soft-volume forward model
(``docs/exact_soft_volume_notes.md``) side by side with the paper's
Fokker-Planck / drift limit, on exactly the two observables of examples 05 and
06:

1. the total-to-instrumented volume ratio ``V_tot / V_det`` vs muon energy
   (as in ``05_soft_volume_drift.py``), and
2. the expected through-going track counts per energy bin
   (as in ``06_soft_volume_event_rate.py``).

Each observable is written to its own figure, with a lower panel sharing the
x-axis that shows the exact/Fokker-Planck ratio.

Both paths use the same ``(phi0, gamma)`` flux and the same Table 1 transport
coefficients. The Fokker-Planck curve is the drift limit ``V_soft ~ A_proj /
(b_mu A)`` (method ``"drift"``); the exact curve replaces ``b_mu A`` by the exact
eigenvalue ``Phi(A)`` and folds in the inelasticity factor ``I(A) ~ 0.8``
(method ``"exact"``). At the IceCube spectral index ``A = 0.98`` the eigenvalues
agree to well under a percent (the exactness identity ``Phi(1) = b_mu``), so the
ratio is dominated by the ``I(A)`` normalization the paper drops.

By default the exact path uses the infinite-column limit, for an apples-to-apples
comparison with the drift form. Pass ``--column-depth-km 1.95`` to add the
finite-column saturation factor (roughly IceCube's overburden for down-going
muons), which suppresses the exact prediction further.

Usage
-----
    python examples/08_exact_vs_fokker_planck.py
    python examples/08_exact_vs_fokker_planck.py --column-depth-km 1.95
    python examples/08_exact_vs_fokker_planck.py --out-dir examples/output
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from softpaws.response.soft_volume import SoftVolumeResponse
from softpaws.transport.soft_volume import sphere_radius_from_volume

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"

GAMMA_IC = 2.38  # IceCube 9.5 yr diffuse-flux best fit (Eq. 1.3)
PHI0_IC = 0.63  # best-fit flux normalization [1e-18 GeV^-1 cm^-2 s^-1 sr^-1]
LIVETIME_S = 9.5 * 365.25 * 86400.0  # IceCube 9.5 yr exposure
SOLID_ANGLE_SR = 2.0 * np.pi  # one hemisphere

# Volume-ratio panel (as in example 05): continuous energy grid, two detectors.
LOG10_E = np.linspace(5.0, 9.0, 81)  # 100 TeV to 1 EeV, in log10(E / GeV)
DETECTORS = {
    "IceCube": sphere_radius_from_volume(1.0),  # ~0.62 km
    "KM3NeT": 0.33,
}

# Event-rate panel (as in example 06): half-decade bins, IceCube-like sphere.
RADIUS_KM = sphere_radius_from_volume(1.0)
LOG10_E_EDGES = np.arange(4.0, 9.01, 0.5)  # 10 TeV to 1 EeV

# Line styles shared by both figures: solid = Fokker-Planck, dashed = exact.
_FP_STYLE = {"ls": "-"}
_EXACT_STYLE = {"ls": "--"}
_METHOD_HANDLES = [
    Line2D([], [], color="k", ls="-", label="Fokker-Planck"),
    Line2D([], [], color="k", ls="--", label="exact"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=_DEFAULT_OUT_DIR,
        help="Directory for the output figures.",
    )
    parser.add_argument(
        "--column-depth-km",
        type=float,
        default=None,
        help="Upstream column depth for the exact path [km]; default is the "
        "infinite-column limit (apples-to-apples with the drift form).",
    )
    return parser.parse_args()


def _new_ratio_figure() -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """A stacked main/ratio figure pair sharing the x-axis."""
    fig, (ax, ax_ratio) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(3.2, 3.8),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )
    ax_ratio.axhline(1.0, color="0.6", lw=0.8, ls=":")
    ax_ratio.set_ylabel("exact / FP")
    return fig, ax, ax_ratio


def _save(fig: plt.Figure, out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_path.with_suffix(suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {path.resolve()}")
    plt.close(fig)


def make_volume_ratio_figure(out_path: pathlib.Path, column_depth_km: float | None) -> None:
    """Volume ratio ``V_tot / V_det`` vs energy: exact vs Fokker-Planck (cf. 05)."""
    energy_gev = 10.0**LOG10_E

    with plt.style.context(str(_STYLE)):
        fig, ax, ax_ratio = _new_ratio_figure()

        for name, radius_km in DETECTORS.items():
            fp = SoftVolumeResponse(radius_km, method="drift")
            exact = SoftVolumeResponse(radius_km, method="exact", column_depth_km=column_depth_km)

            ratio_fp = fp.target_volume_cm3(energy_gev, GAMMA_IC) / fp.v_det_cm3
            ratio_exact = exact.target_volume_cm3(energy_gev, GAMMA_IC) / exact.v_det_cm3

            (line,) = ax.plot(LOG10_E, ratio_fp, label=name, **_FP_STYLE)
            ax.plot(LOG10_E, ratio_exact, color=line.get_color(), **_EXACT_STYLE)
            ax_ratio.plot(LOG10_E, ratio_exact / ratio_fp, color=line.get_color())

        ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
        ax.set_ylabel(r"$V_{\rm tot}\,/\,V_{\rm det}$")
        detector_handles = ax.get_legend_handles_labels()[0]
        ax.legend(handles=detector_handles + _METHOD_HANDLES, fontsize="small")

        ax_ratio.set_xlabel(r"$\log_{10}(E_\mu\,/\,\mathrm{GeV})$")

        _save(fig, out_path)


def make_event_rate_figure(out_path: pathlib.Path, column_depth_km: float | None) -> None:
    """Expected track counts per bin: exact vs Fokker-Planck (cf. 06)."""
    fp = SoftVolumeResponse(RADIUS_KM, method="drift")
    exact = SoftVolumeResponse(RADIUS_KM, method="exact", column_depth_km=column_depth_km)

    parts = ("total", "soft")
    counts_fp = {
        p: fp.expected_counts(LOG10_E_EDGES, PHI0_IC, GAMMA_IC, LIVETIME_S, SOLID_ANGLE_SR, part=p)
        for p in parts
    }
    counts_exact = {
        p: exact.expected_counts(
            LOG10_E_EDGES, PHI0_IC, GAMMA_IC, LIVETIME_S, SOLID_ANGLE_SR, part=p
        )
        for p in parts
    }

    with plt.style.context(str(_STYLE)):
        fig, ax, ax_ratio = _new_ratio_figure()

        for part in parts:
            patch = ax.stairs(counts_fp[part], LOG10_E_EDGES, label=part, **_FP_STYLE)
            color = patch.get_edgecolor()
            ax.stairs(counts_exact[part], LOG10_E_EDGES, color=color, **_EXACT_STYLE)
            ax_ratio.stairs(counts_exact[part] / counts_fp[part], LOG10_E_EDGES, color=color)

        ax.set_yscale("log")
        ax.set_ylabel("expected tracks / bin (9.5 yr)")
        contribution_handles = ax.get_legend_handles_labels()[0]
        ax.legend(handles=contribution_handles + _METHOD_HANDLES, fontsize="small")

        ax_ratio.set_ylim(0.7, 1.05)
        ax_ratio.set_xlabel(r"$\log_{10}(E_\mu\,/\,\mathrm{GeV})$")

        _save(fig, out_path)


def main() -> None:
    args = parse_args()
    column = args.column_depth_km
    label = "infinite column" if column is None else f"column depth {column:g} km"
    print(f"Exact path: {label}")

    make_volume_ratio_figure(args.out_dir / "08_exact_vs_fp_volume", column)
    make_event_rate_figure(args.out_dir / "08_exact_vs_fp_counts", column)


if __name__ == "__main__":
    main()
