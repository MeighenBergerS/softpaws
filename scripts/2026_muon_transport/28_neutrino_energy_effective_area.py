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
:func:`~softpaws.transport.muon_range.muon_range_km` puts the arrival
probability at a step function on the mean CSDA range. The real loss law is
right-skewed (:mod:`softpaws.transport.loss_distribution`), so more muons fall
short of the mean than overshoot it, and
:func:`~softpaws.transport.muon_range.stochastic_muon_range_km` -- the exact
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
    python scripts/2026_muon_transport/28_neutrino_energy_effective_area.py
    python scripts/2026_muon_transport/28_neutrino_energy_effective_area.py --threshold 700
    python scripts/2026_muon_transport/28_neutrino_energy_effective_area.py \
        --data-dir /path/to/dataverse_files
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from softpaws.constants import CM_PER_KM
from softpaws.data.icecube import (
    livetime_weighted_effective_area,
)
from softpaws.transport.attenuation import (
    flavour_transmission,
    prem_column,
    regenerated_transmission,
    survival_probability,
)
from softpaws.transport.cross_section import bgr18_cross_section
from softpaws.transport.muon_range import (
    DEFAULT_MUON_THRESHOLD_GEV,
    muon_range_km,
    stochastic_muon_range_km,
)
from softpaws.transport.soft_volume import light_reach_radius_km, prism_projected_area_km2
from softpaws.transport.source import MEAN_INELASTICITY, nucleon_number_density
from softpaws.transport.tau import BR_TAU_TO_MU, MEAN_Z

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_DATA_DIR = _HERE.parents[1] / "src" / "softpaws" / "data" / "dataverse_files"
_DEFAULT_OUT_DIR = _HERE / "output"

# Coarser than example 26's 61-point grid: every point costs one exact log-loss
# inversion, and the curve is smooth on this scale.
COMMON_LOG10_E = np.linspace(3.0, 8.0, 26)

# IceCube as an upright hexagonal prism and not a sphere. The array is ~1 km^2 of
# footprint by 1 km of instrumented height, which reproduces V_det = 1.00 km^3
# exactly, where an equal-volume sphere reproduces the volume but presents the
# same 1.21 km^2 in every direction. RADIUS_KM is the area-equivalent radius of
# the hexagon, so pi R^2 is the footprint and pi R^2 h the instrumented volume.
FOOTPRINT_KM2 = 1.0
HEIGHT_KM = 1.0
N_SIDES = 6
RADIUS_KM = float(np.sqrt(FOOTPRINT_KM2 / np.pi))  # ~0.564 km

# Declination samples for the upgoing-hemisphere average. The published table is
# binned in sin(dec), so the average is taken with sin(dec) weighting.
N_DEC = 60

# The simulation behind the DR2 tables runs to 100 PeV (DR2_readme.txt), which is
# the top of COMMON_LOG10_E, so the last point sits on the boundary of the
# tabulation. It is plotted but excluded from the residual statistics.
STATS_LOG10_E = (5.0, 7.8)

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
        "--kernel-evaluation",
        choices=("running", "frozen"),
        default="running",
        help=(
            "Where along the descent the loss kernel is read. 'running' follows it "
            "down; 'frozen' holds the production-energy value, which is the closed "
            "form of Eq. (C4) as written and is short by 3-11% across this band."
        ),
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=_DEFAULT_OUT_DIR / "28_neutrino_energy_effective_area.pdf",
        help="Output file for the figure.",
    )
    return parser.parse_args()


def icecube_upgoing(data_dir: pathlib.Path):
    """Livetime-weighted DR2 effective area over the upgoing sky [cm^2].

    Delegates to :func:`softpaws.data.icecube.livetime_weighted_effective_area`
    on ``COMMON_LOG10_E``.
    """
    return livetime_weighted_effective_area(data_dir, COMMON_LOG10_E)[0]


def upgoing_columns() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PREM columns, solid-angle weights and zenith cosines, upgoing hemisphere.

    Returns
    -------
    columns_g_cm2 : np.ndarray, shape (N_DEC,)
        Layered-PREM column along each declination's Earth chord [g cm^-2].
    weights : np.ndarray, shape (N_DEC,)
        Solid-angle weights, ``cos(dec)``, for averaging over the hemisphere.
    cos_theta : np.ndarray, shape (N_DEC,)
        Cosine of the arrival zenith angle. IceCube sits at the Pole, so a
        source at declination ``dec`` arrives at ``cos(theta_z) = -sin(dec)``
        and only the magnitude enters the projected area.
    """
    dec_deg = np.linspace(0.5, 89.5, N_DEC)
    columns = np.array([prem_column(float(d)) for d in dec_deg])
    dec_rad = np.deg2rad(dec_deg)
    return columns, np.cos(dec_rad), np.sin(dec_rad)


def target_volume_cm3(
    length_km: np.ndarray,
    radius_km: float | np.ndarray = RADIUS_KM,
    cos_theta: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Target volume for a monochromatic parent: projected column plus detector.

    The projected area now depends on the arrival direction, so this returns one
    volume per (energy, declination) pair when ``cos_theta`` is an array. The
    detector volume itself does not: a prism of footprint ``pi R^2`` and height
    ``HEIGHT_KM`` holds ``pi R^2 h`` whatever direction it is viewed from.

    Parameters
    ----------
    length_km : np.ndarray
        Effective muon length [km] at each neutrino energy.
    radius_km : float or np.ndarray, optional
        Area-equivalent radius of the prism cross-section [km]. An array is
        broadcast against ``length_km``, which is how the reach law of
        :func:`~softpaws.transport.soft_volume.light_reach_radius_km` enters.
    cos_theta : float or np.ndarray, optional
        Cosine of the arrival zenith. Appended as a trailing axis when it is an
        array. Defaults to 0, the horizontal arrival.

    Returns
    -------
    volume : np.ndarray
        Target volume [cm^3], of shape ``length_km.shape + cos_theta.shape``.
    """
    radius = np.asarray(radius_km, dtype=float)
    length = np.asarray(length_km, dtype=float)
    zenith = np.asarray(cos_theta, dtype=float)
    if zenith.ndim:
        radius = radius[..., None]
        length = length[..., None]
    proj_area = prism_projected_area_km2(zenith, radius, HEIGHT_KM, N_SIDES)
    v_det = np.pi * radius**2 * HEIGHT_KM
    return (proj_area * length + v_det) * CM_PER_KM**3


def mean_target_volume_cm3(
    length_km: np.ndarray,
    radius_km: float | np.ndarray = RADIUS_KM,
) -> np.ndarray:
    """Target volume averaged over the upgoing hemisphere at fixed energy.

    Parameters
    ----------
    length_km : np.ndarray
        Effective muon length [km] at each neutrino energy.
    radius_km : float or np.ndarray, optional
        Area-equivalent radius of the prism cross-section [km].

    Returns
    -------
    volume : np.ndarray
        Solid-angle-averaged target volume [cm^3], one per energy.
    """
    _, weights, cos_theta = upgoing_columns()
    return np.average(
        target_volume_cm3(length_km, radius_km, cos_theta), axis=-1, weights=weights
    )


def required_radius_km(ratio: np.ndarray, radius_km: float = RADIUS_KM) -> np.ndarray:
    """Sphere radius that would scale the target volume by ``ratio``.

    The implied selection efficiency is the published area divided by the
    model. Where it departs from a constant, the departure can be pushed into
    the geometry, and for a sphere the projected area dominates the target
    volume, so the radius that would absorb it is close to ``R sqrt(ratio)``.
    Solving the full cubic instead keeps the small ``V_det`` term honest.

    Parameters
    ----------
    ratio : np.ndarray
        Required scaling of the target volume at each energy.
    radius_km : float, optional
        Nominal instrumented radius [km].

    Returns
    -------
    radius : np.ndarray
        Required radius [km], ``NaN`` where ``ratio`` is not finite.
    """
    lengths = stochastic_muon_range_km(
        (1.0 - MEAN_INELASTICITY) * 10.0**COMMON_LOG10_E, DEFAULT_MUON_THRESHOLD_GEV
    )
    out = np.full(np.shape(ratio), np.nan)
    for i, (r, length) in enumerate(zip(np.atleast_1d(ratio), lengths)):
        if not np.isfinite(r) or r <= 0.0:
            continue
        target = r * float(mean_target_volume_cm3(np.array([length]))[0])
        out[i] = brentq(
            lambda x: float(mean_target_volume_cm3(np.array([length]), x)[0]) - target,
            1.0e-4,
            50.0,
        )
    return out


def fit_reach_law(log10_e: np.ndarray, radius_needed_km: np.ndarray) -> tuple[float, float]:
    """Least-squares reach law through the radii the published curve demands.

    Parameters
    ----------
    log10_e : np.ndarray
        ``log10(E_nu / GeV)`` of the points to fit.
    radius_needed_km : np.ndarray
        Radius each point demands [km]; ``NaN`` entries are dropped.

    Returns
    -------
    reach_km : float
        Growth of the reach per e-fold of energy [km].
    pivot_gev : float
        Energy at which the effective radius equals the instrumented one [GeV].
    """
    valid = np.isfinite(radius_needed_km)
    ln_e = np.log(10.0 ** np.asarray(log10_e)[valid])
    slope, intercept = np.polyfit(ln_e, np.asarray(radius_needed_km)[valid], 1)
    return float(slope), float(np.exp((RADIUS_KM - intercept) / slope))


def effective_area_absorbed(length_km: np.ndarray) -> np.ndarray:
    """Effective area with pure-absorption Earth attenuation (today's treatment).

    The projected area and the transmission both depend on the arrival
    direction, so the average is taken over their product and not over the
    transmission alone.

    Parameters
    ----------
    length_km : np.ndarray
        Effective muon length [km] at each energy of ``COMMON_LOG10_E``.

    Returns
    -------
    aeff : np.ndarray
        Effective area [cm^2], averaged over the upgoing hemisphere.
    """
    energy = 10.0**COMMON_LOG10_E
    columns, weights, cos_theta = upgoing_columns()
    survival = survival_probability(
        energy[:, None], columns[None, :], cross_section=CROSS_SECTION
    )
    volume = target_volume_cm3(length_km, RADIUS_KM, cos_theta)
    per_dec = volume * survival
    return (
        np.average(per_dec, axis=1, weights=weights)
        * nucleon_number_density()
        * CROSS_SECTION.cc(energy)
    )


def effective_area_regenerated(
    length_km: np.ndarray,
    threshold_gev: float,
    reach_km: float | None = None,
    pivot_gev: float = 1.0e6,
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
    reach_km : float or None, optional
        Growth of the light reach per e-fold [km]. ``None`` keeps the static
        instrumented radius.
    pivot_gev : float, optional
        Energy at which the reach vanishes [GeV]. Ignored when ``reach_km`` is
        ``None``.

    Returns
    -------
    aeff : np.ndarray
        Effective area [cm^2], averaged over the upgoing hemisphere.
    """
    energy = 10.0**COMMON_LOG10_E
    columns, weights, cos_theta = upgoing_columns()
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
        rung_radius = (
            RADIUS_KM
            if reach_km is None
            else light_reach_radius_km(
                RADIUS_KM, (1.0 - MEAN_INELASTICITY) * rung_energy, reach_km, pivot_gev
            )
        )
        rung_volume = target_volume_cm3(rung_length, rung_radius, cos_theta)
        rung_rate = n_nucleon * CROSS_SECTION.cc(rung_energy)[:, None] * rung_volume
        # Sum the ladder at each declination, then average over solid angle.
        per_dec = (rung_weight * rung_rate).sum(axis=0)
        out[i] = np.average(per_dec, weights=weights)
    return out


def effective_area_tau_channel(
    length_km: np.ndarray,
    threshold_gev: float,
) -> np.ndarray:
    """Muon-track effective area from an equal-normalization ``nu_tau`` flux.

    A through-going track cannot tell a direct ``nu_mu`` CC muon from one made
    by ``nu_tau -> tau -> mu``, so the second channel adds to the observed rate.
    It matters more and more with energy for one reason: charged current
    regenerates a ``nu_tau`` but terminates a ``nu_mu``, so the Earth stays
    transparent to ``nu_tau`` long after it has gone opaque to ``nu_mu``
    (:func:`~softpaws.transport.attenuation.flavour_transmission`).

    The muon is born about four times lower in energy than in the direct
    channel -- ``<z> (1 - <y_w>) E_nu`` with ``<z> = 0.3`` against
    ``(1 - <y_w>) E_nu`` -- which shortens its range, and the branching ratio
    costs a further factor ``B_{tau->mu} = 0.174``. Neither offsets the
    transmission gain at UHE.

    **This is a model-side addition, not a like-for-like term.** The DR2 tables
    are muon-neutrino effective areas, generated from ``nu_mu`` simulation
    (``DR2_readme.txt``), so this channel is not inside the published number it
    is being compared against. It is reported as a separate curve for that
    reason, and it assumes ``phi_nu_tau = phi_nu_mu`` at Earth, which is what
    the effective-area convention forces once the flux is divided out.

    Parameters
    ----------
    length_km : np.ndarray
        Effective muon length [km] on ``COMMON_LOG10_E``, reused for the
        down-scattered rungs by log-interpolation.
    threshold_gev : float
        Muon selection threshold [GeV].

    Returns
    -------
    aeff : np.ndarray
        Effective area [cm^2], averaged over the upgoing hemisphere.
    """
    energy = 10.0**COMMON_LOG10_E
    columns, weights, cos_theta = upgoing_columns()
    n_nucleon = nucleon_number_density()
    # Muon energy from the two-step decay chain, as a fraction of the parent.
    muon_fraction = MEAN_Z * (1.0 - MEAN_INELASTICITY)

    out = np.empty(energy.size)
    for i, e_nu in enumerate(energy):
        rung_energy, rung_weight = flavour_transmission(
            float(e_nu), columns, CROSS_SECTION, flavour="tau"
        )
        rung_length = np.interp(
            np.log10(rung_energy * muon_fraction / (1.0 - MEAN_INELASTICITY)),
            COMMON_LOG10_E,
            length_km,
            left=0.0,
            right=length_km[-1],
        )
        rung_length[muon_fraction * rung_energy <= threshold_gev] = 0.0
        rung_rate = (
            n_nucleon
            * CROSS_SECTION.cc(rung_energy)[:, None]
            * target_volume_cm3(rung_length, RADIUS_KM, cos_theta)
            * BR_TAU_TO_MU
        )
        per_dec = (rung_weight * rung_rate).sum(axis=0)
        out[i] = np.average(per_dec, weights=weights)
    return out


def make_figure(
    icecube: np.ndarray,
    curves: dict[str, np.ndarray],
    out_path: pathlib.Path,
) -> None:
    """Draw the two-panel comparison and write it to disk."""
    with plt.style.context(str(_STYLE)):
        fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.0))

        colors = ("C0", "C1", "C3", "C2", "C4")

        ax = axes[0]
        ax.plot(COMMON_LOG10_E, icecube, color="k", lw=1.8, label="IceCube, upgoing")
        for (name, curve), color in zip(curves.items(), colors):
            ax.plot(COMMON_LOG10_E, curve, lw=1.1, color=color, label=name)
        ax.set_yscale("log")
        ax.set_ylabel(r"$A_{\rm eff}$ [cm$^2$]")
        ax.set_title("(a) per-neutrino-energy effective area", fontsize=7)
        ax.legend(fontsize=5.5, loc="upper left")

        ax = axes[1]
        for (name, curve), color in zip(curves.items(), colors):
            ax.plot(COMMON_LOG10_E, icecube / curve, lw=1.1, color=color, label=name)
        ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_ylim(0.3, 4.0)
        ax.set_ylabel("IceCube / model")
        ax.set_title("(b) residual; only the reach law is fitted", fontsize=7)
        ax.legend(fontsize=5.5, loc="upper left")

        for ax in axes:
            ax.set_xlim(COMMON_LOG10_E[0], COMMON_LOG10_E[-1])
            ax.set_xlabel(r"$\log_{10}(E_\nu\,/\,\mathrm{GeV})$")
            # Beyond the top of the DR2 simulation; shown, but not scored.
            ax.axvspan(STATS_LOG10_E[1], COMMON_LOG10_E[-1], color="0.85", alpha=0.5, lw=0)

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

    print(f"\n{'log10(E/GeV)':>13} {'A_eff^IC':>11} " + " ".join(f"{n:>28}" for n in curves))
    for log10_e in (4.0, 5.0, 6.0, 7.0, 8.0):
        i = int(np.argmin(np.abs(COMMON_LOG10_E - log10_e)))
        ratios = " ".join(f"{icecube[i] / c[i]:28.2f}" for c in curves.values())
        flag = "  *" if COMMON_LOG10_E[i] > STATS_LOG10_E[1] else ""
        print(f"{COMMON_LOG10_E[i]:13.1f} {icecube[i]:11.3g} {ratios}{flag}")
    print("  (columns are IceCube / model; 1.0 is perfect, no parameters were fitted)")
    print("  * beyond the top of the DR2 simulation (100 PeV); excluded from the statistics")

    lo, hi = STATS_LOG10_E
    band = (COMMON_LOG10_E >= lo) & (COMMON_LOG10_E <= hi)
    print()
    for name, curve in curves.items():
        residual = np.log10(icecube[band] / curve[band])
        trend = residual[-1] - residual[0]
        print(
            f"  {name:>28}: rms {np.std(residual):.3f} dex, "
            f"trend {trend:+.2f} dex over 1e{lo:g}-1e{hi:g}"
        )


def main() -> None:
    args = parse_args()

    print(f"Loading IceCube IRFs from: {args.data_dir}")
    icecube = icecube_upgoing(args.data_dir)

    energy_mu = (1.0 - MEAN_INELASTICITY) * 10.0**COMMON_LOG10_E
    print(
        f"Computing muon lengths (threshold = {args.threshold:g} GeV, "
        f"kernel {args.kernel_evaluation}) ..."
    )
    lengths = {
        "deterministic": muon_range_km(
            energy_mu, args.threshold, kernel_evaluation=args.kernel_evaluation
        ),
        "stochastic": stochastic_muon_range_km(
            energy_mu, args.threshold, kernel_evaluation=args.kernel_evaluation
        ),
    }

    print("Building effective areas ...")
    curves = {
        "CSDA range, absorption only": effective_area_absorbed(lengths["deterministic"]),
        "stochastic, absorption only": effective_area_absorbed(lengths["stochastic"]),
        "stochastic + NC regeneration": effective_area_regenerated(
            lengths["stochastic"], args.threshold
        ),
    }
    print("Adding the nu_tau -> tau -> mu channel ...")
    curves["+ nu_tau -> tau -> mu"] = curves["stochastic + NC regeneration"] + (
        effective_area_tau_channel(lengths["stochastic"], args.threshold)
    )

    # --- Fit the reach law, the same two-parameter form calibrated on ARCA. ---
    print("\nFitting the reach law against the DR2 upgoing table ...")
    lo, hi = STATS_LOG10_E
    band = (COMMON_LOG10_E >= lo) & (COMMON_LOG10_E <= hi)
    ratio = np.full(COMMON_LOG10_E.size, np.nan)
    ratio[band] = (icecube / curves["+ nu_tau -> tau -> mu"])[band]
    needed = required_radius_km(ratio)
    reach_km, pivot_gev = fit_reach_law(COMMON_LOG10_E, needed)
    print(f"  reach   Lambda = {reach_km * 1e3:6.1f} m per e-fold "
          f"({reach_km * 1e3 * np.log(10.0):.0f} m per decade)")
    print(f"  pivot   E_piv  = 10^{np.log10(pivot_gev):.2f} GeV")
    print(f"  instrumented R = {RADIUS_KM * 1e3:6.0f} m")
    print("  required radius by energy:")
    for log10_e in (5.0, 6.0, 7.0, 7.8):
        i = int(np.argmin(np.abs(COMMON_LOG10_E - log10_e)))
        if np.isfinite(needed[i]):
            print(f"    log10(E/GeV) = {COMMON_LOG10_E[i]:4.1f}   {needed[i] * 1e3:6.0f} m")

    print("  forward-running with the fitted reach ...")
    curves["+ fitted reach"] = effective_area_regenerated(
        lengths["stochastic"], args.threshold, reach_km=reach_km, pivot_gev=pivot_gev
    ) + effective_area_tau_channel(lengths["stochastic"], args.threshold)

    report(icecube, curves, lengths)
    make_figure(icecube, curves, args.out)


if __name__ == "__main__":
    main()
