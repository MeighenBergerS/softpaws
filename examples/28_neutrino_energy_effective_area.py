"""Example 28 -- effective area in the published (per-neutrino-energy) convention.

Examples 20 and 26 compare the *soft volume* against the published IceCube
effective area. That comparison mixes two conventions. The soft volume of
Eq. (10) is differential in the **observed muon energy** and already spectrally
weighted, so its length is ``1/Phi(A) ~ 2.4 km``; the published ``A_eff`` is
tabulated at fixed **neutrino energy** and integrated over every muon energy
that survives the selection. Reading one against the other at the same
numerical energy is not like-for-like: the DR2 smearing matrix puts the median
reconstructed muon energy more than two decades below ``E_nu``, and the offset
grows with energy.

This example builds the effective area in the published convention instead. At
fixed ``E_nu`` the source is monochromatic, which is App. I's ``s -> 0`` case:
``Phi(0) = 0`` and ``I(0) = 1``, so there is no spectral shortening left and the
length is fixed purely by whether the muon reaches the detector above the
analysis threshold,

    A_eff(E_nu) = n_N sigma_CC(E_nu) [A_proj L(E_nu) + V_det] D_nu(E_nu),

    L(E_nu) = Integral_0^inf d_ell P[muon born at (1 - <y_w>) E_nu is above E_thr
              after propagating ell].

Two ingredients are new relative to examples 20 and 26.

**The length is stochastic, not deterministic.**
:func:`~softpaws.transport.soft_volume.muon_range_km` puts the arrival
probability at a step function on the mean CSDA range. The real loss law is
right-skewed (:mod:`softpaws.transport.loss_distribution`), so more muons fall
short of the mean than overshoot it, and
:func:`~softpaws.transport.soft_volume.stochastic_muon_range_km` -- the exact
log-loss CDF integrated over depth -- comes out ~7% shorter at 1 PeV and ~12%
shorter at 100 PeV.

**Neutral-current scattering does not delete the neutrino.**
:func:`~softpaws.transport.attenuation.survival_probability` removes it on any
interaction. The published response instead credits an event to the neutrino's
*surface* energy even if it neutral-current scattered on the way in, so pure
absorption undershoots at UHE where the Earth is many interaction lengths deep.
:func:`~softpaws.transport.attenuation.regenerated_transmission` keeps the
down-scattered population on an energy ladder and lets it interact at the
detector.

The selection threshold is not fitted here: the DR2 smearing matrix pins it. The
5th percentile of accepted reconstructed muon energy sits at ~700 GeV and stays
there across three decades of ``E_nu`` (slope 0.02 in ``log-log``, against 1.0
for a threshold that scaled with the neutrino energy), so a fixed
``E_thr = 1 TeV`` is what the data support. Pass ``--threshold`` to vary it.

Nothing in the resulting curve is fitted -- no normalization, no column, no
efficiency.

Usage
-----
    python examples/28_neutrino_energy_effective_area.py
    python examples/28_neutrino_energy_effective_area.py --threshold 700
    python examples/28_neutrino_energy_effective_area.py --data-dir /path/to/dataverse_files
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.data.loader import compute_livetime_s, load_uptime, parse_aeff
from softpaws.data.schema import SEASONS
from softpaws.transport.attenuation import (
    prem_column,
    regenerated_transmission,
    survival_probability,
)
from softpaws.transport.cross_section import bgr18_cross_section
from softpaws.transport.soft_volume import (
    DEFAULT_MUON_THRESHOLD_GEV,
    muon_range_km,
    sphere_radius_from_volume,
    stochastic_muon_range_km,
)
from softpaws.transport.source import MEAN_INELASTICITY, nucleon_number_density
from softpaws.utils.constants import CM_PER_KM

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_DATA_DIR = _HERE.parent / "src" / "softpaws" / "data" / "dataverse_files"
_DEFAULT_OUT_DIR = _HERE / "output"

# Coarser than example 26's 61-point grid: every point costs one exact log-loss
# inversion, and the curve is smooth on this scale.
COMMON_LOG10_E = np.linspace(3.0, 8.0, 26)
RADIUS_KM = sphere_radius_from_volume(1.0)  # ~0.62 km, IceCube-like

# Declination samples for the upgoing-hemisphere average. The published table is
# binned in sin(dec), so the average is taken with sin(dec) weighting.
N_DEC = 60

CROSS_SECTION = bgr18_cross_section()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=_DEFAULT_DATA_DIR,
        help="Root of the DR2 data directory (must contain 'irfs/' and 'uptime/').",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_MUON_THRESHOLD_GEV,
        help="Muon selection threshold [GeV]; the DR2 smearing matrix supports ~700-1000.",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=_DEFAULT_OUT_DIR / "28_neutrino_energy_effective_area.pdf",
        help="Output file for the figure.",
    )
    return parser.parse_args()


def _canonical_irf_season(season: str) -> str:
    return "IC86" if season.startswith("IC86") else season


def icecube_upgoing(data_dir: pathlib.Path) -> np.ndarray:
    """Livetime-weighted IceCube effective area, averaged over the upgoing sky.

    Parameters
    ----------
    data_dir : pathlib.Path
        Root of the DR2 data directory.

    Returns
    -------
    aeff : np.ndarray, shape (COMMON_LOG10_E.size,)
        Effective area [cm^2], solid-angle averaged over ``sin(dec) > 0``.
    """
    irf_dir = data_dir / "irfs"
    uptime_dir = data_dir / "uptime"

    total = np.zeros_like(COMMON_LOG10_E)
    total_livetime_s = 0.0
    aeff_cache: dict[str, object] = {}

    for season in SEASONS:
        irf_season = _canonical_irf_season(season)
        if irf_season not in aeff_cache:
            raw = np.genfromtxt(irf_dir / f"{irf_season}_effectiveArea.csv", comments="#")
            aeff_cache[irf_season] = parse_aeff(raw)
        aeff = aeff_cache[irf_season]

        livetime_s = compute_livetime_s(load_uptime(uptime_dir / f"{season}_exp.csv"))
        upgoing = aeff.sin_dec_centers > 0.0
        curve = np.average(
            aeff.values[:, upgoing], axis=1, weights=np.diff(aeff.sin_dec_edges)[upgoing]
        )
        total += livetime_s * np.interp(COMMON_LOG10_E, aeff.log10_energy_centers, curve)
        total_livetime_s += livetime_s

    return total / total_livetime_s


def upgoing_columns() -> tuple[np.ndarray, np.ndarray]:
    """PREM column depths and solid-angle weights over the upgoing hemisphere.

    Returns
    -------
    columns_g_cm2 : np.ndarray, shape (N_DEC,)
        Layered-PREM column along each declination's Earth chord [g cm^-2].
    weights : np.ndarray, shape (N_DEC,)
        Solid-angle weights, ``cos(dec)``, for averaging over the hemisphere.
    """
    dec_deg = np.linspace(0.5, 89.5, N_DEC)
    columns = np.array([prem_column(float(d)) for d in dec_deg])
    return columns, np.cos(np.deg2rad(dec_deg))


def target_volume_cm3(
    length_km: np.ndarray,
    radius_km: float = RADIUS_KM,
) -> np.ndarray:
    """Target volume for a monochromatic parent: projected column plus detector.

    Parameters
    ----------
    length_km : np.ndarray
        Effective muon length [km] at each neutrino energy.
    radius_km : float, optional
        Radius of the spherical instrumented volume [km].

    Returns
    -------
    volume : np.ndarray
        Target volume [cm^3].
    """
    proj_area = np.pi * radius_km**2
    v_det = 4.0 / 3.0 * np.pi * radius_km**3
    return (proj_area * length_km + v_det) * CM_PER_KM**3


def effective_area_absorbed(volume_cm3: np.ndarray) -> np.ndarray:
    """Effective area with pure-absorption Earth attenuation (today's treatment).

    Parameters
    ----------
    volume_cm3 : np.ndarray
        Target volume [cm^3] at each energy of ``COMMON_LOG10_E``.

    Returns
    -------
    aeff : np.ndarray
        Effective area [cm^2], averaged over the upgoing hemisphere.
    """
    energy = 10.0**COMMON_LOG10_E
    columns, weights = upgoing_columns()
    survival = survival_probability(
        energy[:, None], columns[None, :], cross_section=CROSS_SECTION
    )
    d_nu = np.average(survival, axis=1, weights=weights)
    return volume_cm3 * nucleon_number_density() * CROSS_SECTION.cc(energy) * d_nu


def effective_area_regenerated(
    length_km: np.ndarray,
    threshold_gev: float,
) -> np.ndarray:
    """Effective area with neutral-current regeneration kept.

    A neutrino that scattered down to ``E_k`` before reaching the detector still
    interacts there, with the cross section and the muon length of ``E_k`` --
    but the event is credited to the surface energy ``E_nu``, matching how the
    published table is built. Each rung of the ladder therefore carries its own
    target volume.

    Parameters
    ----------
    length_km : np.ndarray
        Effective muon length [km] on ``COMMON_LOG10_E``, reused by
        interpolation for the down-scattered rungs.
    threshold_gev : float
        Muon selection threshold [GeV], used only to zero out rungs that have
        fallen below it.

    Returns
    -------
    aeff : np.ndarray
        Effective area [cm^2], averaged over the upgoing hemisphere.
    """
    energy = 10.0**COMMON_LOG10_E
    columns, weights = upgoing_columns()
    n_nucleon = nucleon_number_density()

    out = np.empty(energy.size)
    for i, e_nu in enumerate(energy):
        rung_energy, rung_weight = regenerated_transmission(
            float(e_nu), columns, CROSS_SECTION
        )
        # Rungs below threshold cannot make a selectable muon at all; above it,
        # reuse the tabulated length by log-interpolation rather than rerunning
        # the inversion for every rung of every energy.
        rung_length = np.interp(
            np.log10(rung_energy), COMMON_LOG10_E, length_km, left=0.0, right=length_km[-1]
        )
        rung_length[(1.0 - MEAN_INELASTICITY) * rung_energy <= threshold_gev] = 0.0
        rung_volume = target_volume_cm3(rung_length)
        rung_rate = n_nucleon * CROSS_SECTION.cc(rung_energy) * rung_volume
        # Sum the ladder at each declination, then average over solid angle.
        per_dec = (rung_weight * rung_rate[:, None]).sum(axis=0)
        out[i] = np.average(per_dec, weights=weights)
    return out


def make_figure(
    icecube: np.ndarray,
    curves: dict[str, np.ndarray],
    out_path: pathlib.Path,
) -> None:
    """Draw the two-panel comparison and write it to disk."""
    with plt.style.context(str(_STYLE)):
        fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.6))

        ax = axes[0]
        ax.plot(COMMON_LOG10_E, icecube, color="k", lw=1.8, label="IceCube, upgoing")
        for (name, curve), color in zip(curves.items(), ("C0", "C1", "C3")):
            ax.plot(COMMON_LOG10_E, curve, lw=1.1, color=color, label=name)
        ax.set_yscale("log")
        ax.set_ylabel(r"$A_{\rm eff}$ [cm$^2$]")
        ax.set_title("(a) per-neutrino-energy effective area", fontsize=7)
        ax.legend(fontsize=5.5, loc="upper left")

        ax = axes[1]
        for (name, curve), color in zip(curves.items(), ("C0", "C1", "C3")):
            ax.plot(COMMON_LOG10_E, icecube / curve, lw=1.1, color=color, label=name)
        ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_ylim(0.3, 4.0)
        ax.set_ylabel("IceCube / model")
        ax.set_title("(b) residual, no free parameters", fontsize=7)
        ax.legend(fontsize=5.5, loc="upper left")

        for ax in axes:
            ax.set_xlim(COMMON_LOG10_E[0], COMMON_LOG10_E[-1])
            ax.set_xlabel(r"$\log_{10}(E_\nu\,/\,\mathrm{GeV})$")

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png"):
            path = out_path.with_suffix(suffix)
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {path.resolve()}")
        plt.close(fig)


def report(
    icecube: np.ndarray,
    curves: dict[str, np.ndarray],
    lengths: dict[str, np.ndarray],
) -> None:
    """Print the lengths and the residuals behind the figure."""
    print(f"\n{'log10(E/GeV)':>13} {'L_CSDA':>8} {'L_stoch':>8} {'ratio':>6}")
    for log10_e in (4.0, 5.0, 6.0, 7.0, 8.0):
        i = int(np.argmin(np.abs(COMMON_LOG10_E - log10_e)))
        print(
            f"{COMMON_LOG10_E[i]:13.1f} {lengths['deterministic'][i]:8.2f} "
            f"{lengths['stochastic'][i]:8.2f} "
            f"{lengths['stochastic'][i] / lengths['deterministic'][i]:6.3f}"
        )

    print(f"\n{'log10(E/GeV)':>13} {'A_eff^IC':>11} " + " ".join(f"{n:>26}" for n in curves))
    for log10_e in (4.0, 5.0, 6.0, 7.0, 8.0):
        i = int(np.argmin(np.abs(COMMON_LOG10_E - log10_e)))
        ratios = " ".join(f"{icecube[i] / c[i]:26.2f}" for c in curves.values())
        print(f"{COMMON_LOG10_E[i]:13.1f} {icecube[i]:11.3g} {ratios}")
    print("  (columns are IceCube / model; 1.0 is perfect, no parameters were fitted)")

    band = (COMMON_LOG10_E >= 5.0) & (COMMON_LOG10_E <= 8.0)
    print()
    for name, curve in curves.items():
        residual = np.log10(icecube[band] / curve[band])
        print(f"  {name:>26}: rms residual 1e5-1e8 = {np.std(residual):.3f} dex")


def main() -> None:
    args = parse_args()

    print(f"Loading IceCube IRFs from: {args.data_dir}")
    icecube = icecube_upgoing(args.data_dir)

    energy_mu = (1.0 - MEAN_INELASTICITY) * 10.0**COMMON_LOG10_E
    print(f"Computing muon lengths (threshold = {args.threshold:g} GeV) ...")
    lengths = {
        "deterministic": muon_range_km(energy_mu, args.threshold),
        "stochastic": stochastic_muon_range_km(energy_mu, args.threshold),
    }

    print("Building effective areas ...")
    curves = {
        "CSDA range, absorption only": effective_area_absorbed(
            target_volume_cm3(lengths["deterministic"])
        ),
        "stochastic, absorption only": effective_area_absorbed(
            target_volume_cm3(lengths["stochastic"])
        ),
        "stochastic + NC regeneration": effective_area_regenerated(
            lengths["stochastic"], args.threshold
        ),
    }

    report(icecube, curves, lengths)
    make_figure(icecube, curves, args.out)


if __name__ == "__main__":
    main()
