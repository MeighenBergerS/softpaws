"""Example 87 -- from the closed form to the full response, one effect at a time, at ARCA230.

Section III of the paper takes the closed-form response, Eq. (aeffcf), to the
full one, Eq. (aeff), by restoring three things: the medium (the rock below
the array and the running, spliced, column-limited range in place of the
frozen closed form), the Earth (the neutral-current regeneration ladder in
place of pure absorption), and the tau channel. This example draws that ladder
at ARCA230, the whole-sky trigger-level site where all three are on, at the
two instrument numbers of example 77.

The figure shows every rung as a ratio to the full response, labelled on the
curve; the areas themselves lie too close to read. The four rungs are
cumulative, so the last one is Eq. (aeff) and the example asserts that it
reproduces :func:`softpaws.response.site_models.water_model` exactly.

Usage
-----
    python scripts/2026_muon_transport/87_arca_closed_form_ladder.py
"""

import argparse
import importlib.util
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.response import site_models as sm
from softpaws.transport.attenuation import survival_probability
from softpaws.transport.muon_range import truncated_muon_range_km, two_medium_range_ratio
from softpaws.transport.soft_volume import light_reach_radius_km
from softpaws.transport.source import MEAN_INELASTICITY, nucleon_number_density
from softpaws.utils.constants import CM_PER_KM, RHO_WATER_G_CM3

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"


def load_example(stem: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / stem)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EX85 = load_example("85_closed_form_response.py", "_example_85")

SITE = sm.ARCA230_WATER_SITE
ENERGY_GEV = 10.0**sm.ARCA_LOG10_E
QUOTED_LOG10_E = (5.0, 6.0, 7.0)

#: The rungs, cumulative: (ladder, medium, tau) switches and the legend text.
RUNGS = (
    ((False, False, False), "Estimate"),
    ((False, True, False), "+ rock"),
    ((True, True, False), "+ Earth regeneration"),
    ((True, True, True), r"+ $\nu_\tau$ channel (full model)"),
)

#: Okabe-Ito, distinct from the site colours of the four-detector figures.
RUNG_COLORS = ("#E69F00", "#56B4E9", "#009E73", "k")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chains", type=pathlib.Path,
                        default=_DEFAULT_OUT_DIR / "77_chains_sigma05.npz")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR)
    return parser.parse_args()


def arca_response(
    theta: np.ndarray,
    ladders: dict,
    ladder: bool,
    medium: bool,
    tau: bool,
) -> np.ndarray:
    """ARCA230's sky-averaged effective area [cm^2] with three switches.

    The assembly of :func:`softpaws.response.site_models.water_model`, with
    each closed-form piece swapped in when its switch is off: pure absorption
    for the ladder, the frozen water range capped at the upstream column for
    the medium, and no second source for the tau channel.
    """
    eps_0, log10_e_thr, b_scale, lam, reach_km = theta
    threshold = 10.0**log10_e_thr
    n_nucleon = nucleon_number_density(RHO_WATER_G_CM3)
    theta_deg, _ = sm.arca_zenith_grid()
    cos_theta = np.cos(np.deg2rad(theta_deg))
    zenith_weights, neutrino_column, muon_column_km = sm.water_columns(SITE)
    below_centre_km = SITE.below_km + 0.5 * SITE.height_km

    total = np.zeros(ENERGY_GEV.size)
    channels = [("mu", 1.0 - MEAN_INELASTICITY, 1.0)]
    if tau:
        tau_fraction = sm.MEAN_Z * (1.0 - MEAN_INELASTICITY)
        channels.append(("tau", tau_fraction, SITE.f_tau * sm.BR_TAU_TO_MU))
    for flavour, muon_fraction, weight in channels:
        if ladder:
            energies, arrival, rock = ladders[flavour]
        else:
            energies = ENERGY_GEV[:, None]
            arrival = survival_probability(
                ENERGY_GEV[:, None], neutrino_column[None, :], lam, sm.CROSS_SECTION
            )[:, None, :]
            rock = np.stack([
                two_medium_range_ratio(
                    muon_fraction * energies[i], sm.DEFAULT_MUON_THRESHOLD_GEV,
                    cos_theta, below_centre_km,
                )
                for i in range(ENERGY_GEV.size)
            ])
        muon_energy = muon_fraction * energies
        if medium:
            length = truncated_muon_range_km(
                muon_energy[:, :, None], muon_column_km[None, None, :], threshold, b_scale
            ) * rock
        else:
            length = np.minimum(
                _EX85.closed_form_range_km(muon_energy, threshold)[:, :, None],
                muon_column_km[None, None, :],
            )
        radius = light_reach_radius_km(SITE.radius_km, muon_energy, reach_km, sm.REACH_PIVOT_GEV)
        area_km2 = sm.water_projected_area_km2(SITE, theta_deg[None, None, :], radius[:, :, None])
        v_det_km3 = SITE.n_blocks * np.pi * radius**2 * SITE.height_km
        volume_km3 = area_km2 * length + v_det_km3[:, :, None]
        sigma = sm.tilted_cc(energies, lam)
        rate = n_nucleon * sigma[:, :, None] * volume_km3 * CM_PER_KM**3
        total += weight * np.average((arrival * rate).sum(axis=1), axis=1, weights=zenith_weights)
    return eps_0 * total


def _curve_angle_deg(ax, x: np.ndarray, y: np.ndarray, x0: float) -> float:
    """Screen-space slope of a curve at ``x0``, for a label that follows it."""
    x1 = x0 + 0.25
    points = ax.transData.transform(np.column_stack([[x0, x1], np.interp([x0, x1], x, y)]))
    return float(np.degrees(np.arctan2(points[1, 1] - points[0, 1], points[1, 0] - points[0, 0])))


#: Label per rung: text, energy [log10 GeV], vertical offset on the ratio,
#: horizontal alignment, and whether the text follows the curve's slope.
LABELS = (
    ("Analytic", 7.3, -0.022, "center", True),
    ("+ Medium", 4.7, -0.022, "center", True),
    ("+ Earth Regeneration", 6.7, +0.022, "center", True),
    (r"+ $\nu_\tau$", 4.7, -0.022, "center", False),
)


def figure(curves: list, out_dir: pathlib.Path) -> None:
    full = curves[-1][1]
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        ratios = []
        for (_, area), color in zip(curves, RUNG_COLORS):
            ratio = area / full
            ratios.append(ratio)
            ax.plot(sm.ARCA_LOG10_E, ratio, color=color, lw=1.2 if color == "k" else 1.1,
                    ls="-" if color == "k" else "--")
        ax.set_xlim(4.0, 8.0)
        ax.set_ylim(0.68, 1.04)
        ax.set_yticks([0.7, 0.8, 0.9, 1.0])
        ax.set_xlabel(r"$\log_{10}(E_\nu\,/\,\mathrm{GeV})$")
        ax.set_ylabel("Ratio To Full")
        ax.set_box_aspect(1)
        fig.canvas.draw()
        for (text, x0, offset, ha, follow), ratio, color in zip(LABELS, ratios, RUNG_COLORS):
            angle = _curve_angle_deg(ax, sm.ARCA_LOG10_E, ratio, x0) if follow else 0.0
            y0 = float(np.interp(x0, sm.ARCA_LOG10_E, ratio)) + offset
            ax.text(x0, y0, text, color=color, ha=ha, va="center",
                    rotation=angle, rotation_mode="anchor")
        out_dir.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png"):
            path = out_dir / f"87a_arca_closed_form_ladder{suffix}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {path.resolve()}")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    chain = np.load(args.chains)["ARCA230_2p_chain"]
    median = np.median(chain, axis=0)
    theta = np.array([1.0, median[0], 1.0, sm.LAMBDA_BGR18, median[1]])
    print("ARCA230 instrument numbers (example 77 medians)")
    print(f"  eps_0 = 1, E_thr = {10**median[0]:.0f} GeV, reach = {1.0e3 * median[1]:.0f} m")

    zenith_weights, neutrino_column, muon_column_km = sm.water_columns(SITE)
    ladders = sm.water_ladders(SITE, neutrino_column)
    library = sm.water_model(theta, SITE, ladders, zenith_weights, muon_column_km)

    curves = []
    for switches, label in RUNGS:
        curves.append((label, arca_response(theta, ladders, *switches)))
    full = curves[-1][1]
    deviation = np.max(np.abs(full / library - 1.0))
    assert deviation < 1.0e-10, deviation
    print(f"  last rung against the library model: max |ratio - 1| = {deviation:.1e}")

    print("\nEach rung over the full response")
    print("  log10(E_nu/GeV):" + "".join(f"{x:8.1f}" for x in QUOTED_LOG10_E))
    for label, area in curves:
        ratio = area / full
        cells = "".join(f"{np.interp(x, sm.ARCA_LOG10_E, ratio):8.3f}" for x in QUOTED_LOG10_E)
        print(f"  {label.replace('~', ' '):>36}:{cells}")
    print("\nEach effect alone, as the step it adds (rung over the rung before)")
    for (_, before), (label, area) in zip(curves[:-1], curves[1:]):
        step = np.interp(QUOTED_LOG10_E, sm.ARCA_LOG10_E, area / before) - 1.0
        cells = "".join(f"{s:+8.1%}" for s in step)
        print(f"  {label.replace('~', ' '):>36}:{cells}")

    np.savez(
        args.out_dir / "87_arca_closed_form_ladder.npz",
        log10_e=sm.ARCA_LOG10_E, theta=theta,
        **{f"rung_{i}": area for i, (_, area) in enumerate(curves)},
    )
    figure(curves, args.out_dir)


if __name__ == "__main__":
    main()
