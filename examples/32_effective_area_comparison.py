"""Example 32 -- IceCube and KM3NeT/ARCA230 effective areas in one panel.

Examples 28 and 30 build the per-neutrino-energy effective area of Sec. VI
separately for the two sites, each against its own published curve and each on
its own axes. Nothing in the construction is site-specific -- the exponent
``Phi``, the first-passage range ``L = E[tau(w_star)]`` and the flavour-dependent
Earth transmission are properties of the loss kernel and of the Earth -- so the
two belong on the same axes, where the claim is visible in one look: the same
machinery, given only each site's instrument numbers, tracks two published
effective areas that differ by two orders of magnitude and span seven decades of
energy between them.

Three curves per detector, and the distinction between them is the point.

``published``
    What the collaboration reports. For IceCube, the livetime-weighted upgoing
    ``nu_mu`` table of the IceTracks-DR2 release. For ARCA230, the ``nu_mu``
    effective area at **trigger level** digitized from KM3NeT Collaboration,
    Eur. Phys. J. C 84 (2024) 885 [arXiv:2402.08363] Fig. 7.

``first principles``
    The model with the instrumented footprint and nothing fitted: a sphere of
    the IceCube volume, or ARCA's as-built cylinders. This is the curve that
    tests the transport, and it is a geometric ceiling -- it may sit above the
    published area, since a real selection throws events away, but a published
    area above it would falsify the construction.

``fitted reach``
    The same model with the two-parameter light reach of Eq. (17) fitted to
    that detector's own published curve. It is a diagnostic, not a result: it
    locates the residual in the instrument response rather than in ``Phi``, and
    quoting its agreement against the curve it was fitted to would be circular.

Both detectors are shown with the ``nu_mu`` channel plus ``nu_tau -> tau -> mu``,
since a track selection cannot tell them apart. ARCA230 rather than ARCA21: the
trigger-level curve is the strongest available test of a geometric ceiling, and
it is the configuration whose reach law example 30 transfers.

Usage
-----
    python examples/32_effective_area_comparison.py
    python examples/32_effective_area_comparison.py --threshold 1e4
    python examples/32_effective_area_comparison.py --data-dir /path/to/dataverse_files
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq

from softpaws.data.loader import compute_livetime_s, load_uptime, parse_aeff
from softpaws.data.schema import SEASONS
from softpaws.transport.attenuation import (
    flavour_transmission,
    prem_column,
    regenerated_transmission,
)
from softpaws.transport.coefficients import diffusion_coefficient, drift_coefficient
from softpaws.transport.cross_section import bgr18_cross_section
from softpaws.transport.loss_distribution import log_loss_cdf
from softpaws.transport.soft_volume import (
    DEFAULT_MUON_THRESHOLD_GEV,
    light_reach_radius_km,
    muon_range_km,
    sphere_radius_from_volume,
    stochastic_muon_range_km,
)
from softpaws.transport.source import MEAN_INELASTICITY, nucleon_number_density
from softpaws.transport.tau import BR_TAU_TO_MU, MEAN_Z
from softpaws.utils.constants import CM_PER_KM, RHO_WATER_G_CM3

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_DATA_DIR = _HERE.parent / "src" / "softpaws" / "data" / "dataverse_files"
_KM3NET_DIR = _HERE.parent / "src" / "softpaws" / "data" / "km3net"
_ARCA230_TRIGGER_TABLE = _KM3NET_DIR / "arca_trigger_level_eff.csv"
_DEFAULT_OUT_DIR = _HERE / "output"

CROSS_SECTION = bgr18_cross_section()

# --- IceCube, example 28's numbers ------------------------------------------
IC_RADIUS_KM = sphere_radius_from_volume(1.0)  # ~0.62 km
IC_LOG10_E = np.linspace(3.0, 8.0, 26)
IC_FIT_BAND = (5.0, 7.8)  # band the reach law is calibrated over
N_DEC = 60

# --- ARCA230, example 30's numbers ------------------------------------------
ARCA230_RADIUS_KM = 0.517
BLOCK_HEIGHT_KM = 0.632
N_BLOCKS_FULL = 2
ARCA_DEPTH_KM = 3.5 - 0.5 * BLOCK_HEIGHT_KM
ARCA_LOG10_E = np.arange(4.0, 10.01, 0.2)
ARCA_FIT_BAND = (4.0, 7.5)  # digitized trigger curve saturates past the top
MAX_SEA_PATH_KM = 100.0
RHO_SEA_G_CM3 = RHO_WATER_G_CM3
N_ZENITH = 90
N_ELL = 401

# Colour carries which curve it is, line style carries which detector, so the
# two legends factorize.
# From the shared style's prop_cycle. Orange is skipped because example 31
# already spends it on ARCA, and here the detector is carried by line style.
CURVE_COLOR = {
    "published": "#e7298a",
    "first principles": "#7570b3",
    "fitted reach": "#1b9e77",
}
# The published curve is drawn wide and underneath, so that the two model
# curves reading on top of it stay legible exactly where they agree with it --
# which, being the result, is most of the range.
CURVE_WIDTH = {"published": 2.4, "first principles": 1.0, "fitted reach": 1.0}
CURVE_ZORDER = {"published": 1, "first principles": 3, "fitted reach": 2}
DETECTOR_STYLE = {"IceCube": "-", "KM3NeT/ARCA230": "--"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=pathlib.Path, default=_DEFAULT_DATA_DIR,
                        help="Root of the DR2 data directory (must contain 'irfs/' and 'uptime/').")
    parser.add_argument("--threshold", type=float, default=DEFAULT_MUON_THRESHOLD_GEV,
                        help="Muon selection threshold [GeV], shared by both sites.")
    parser.add_argument("--out", type=pathlib.Path,
                        default=_DEFAULT_OUT_DIR / "32_effective_area_comparison.pdf")
    return parser.parse_args()


def fit_reach_law(
    log10_e: np.ndarray, required_radius_km: np.ndarray, radius_km: float
) -> tuple[float, float]:
    """Least-squares reach law through the radii a published curve demands.

    Shared between the two sites: both invert a required-radius sequence that is
    close to linear in ``ln E`` for the same
    :func:`~softpaws.transport.soft_volume.light_reach_radius_km` form.

    Parameters
    ----------
    log10_e : np.ndarray
        ``log10(E_nu / GeV)`` of the points to fit.
    required_radius_km : np.ndarray
        Radius each point demands [km]; ``NaN`` entries are dropped.
    radius_km : float
        Instrumented radius [km], used to locate the pivot.

    Returns
    -------
    reach_km : float
        Growth of the reach per e-fold of energy [km].
    pivot_gev : float
        Energy at which the effective radius equals ``radius_km`` [GeV].
    """
    valid = np.isfinite(required_radius_km)
    ln_e = np.log(10.0 ** np.asarray(log10_e)[valid])
    slope, intercept = np.polyfit(ln_e, np.asarray(required_radius_km)[valid], 1)
    return float(slope), float(np.exp((radius_km - intercept) / slope))


# ---------------------------------------------------------------------------
# IceCube
# ---------------------------------------------------------------------------


def _canonical_irf_season(season: str) -> str:
    return "IC86" if season.startswith("IC86") else season


def icecube_published(data_dir: pathlib.Path) -> np.ndarray:
    """Livetime-weighted published IceCube effective area, upgoing sky [cm^2]."""
    irf_dir = data_dir / "irfs"
    uptime_dir = data_dir / "uptime"
    total = np.zeros_like(IC_LOG10_E)
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
        total += livetime_s * np.interp(IC_LOG10_E, aeff.log10_energy_centers, curve)
        total_livetime_s += livetime_s
    return total / total_livetime_s


def ic_upgoing_columns() -> tuple[np.ndarray, np.ndarray]:
    """PREM column depths and solid-angle weights over the upgoing hemisphere."""
    dec_deg = np.linspace(0.5, 89.5, N_DEC)
    columns = np.array([prem_column(float(d)) for d in dec_deg])
    return columns, np.cos(np.deg2rad(dec_deg))


def ic_target_volume_cm3(
    length_km: np.ndarray, radius_km: float | np.ndarray = IC_RADIUS_KM
) -> np.ndarray:
    """Spherical target volume, projected column plus detector [cm^3]."""
    radius = np.asarray(radius_km, dtype=float)
    return (np.pi * radius**2 * length_km + 4.0 / 3.0 * np.pi * radius**3) * CM_PER_KM**3


def ic_required_radius_km(ratio: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Sphere radius that would scale the target volume by ``ratio`` at each energy."""
    out = np.full(np.shape(ratio), np.nan)
    for i, (r, length) in enumerate(zip(np.atleast_1d(ratio), lengths)):
        if not np.isfinite(r) or r <= 0.0:
            continue
        target = r * float(ic_target_volume_cm3(np.array([length]))[0])
        out[i] = brentq(
            lambda x: float(ic_target_volume_cm3(np.array([length]), x)[0]) - target, 1.0e-4, 50.0
        )
    return out


def ic_effective_area_regenerated(
    length_km: np.ndarray,
    threshold_gev: float,
    reach_km: float | None = None,
    pivot_gev: float = 1.0e6,
) -> np.ndarray:
    """Direct nu_mu channel, NC regeneration kept, upgoing-averaged [cm^2]."""
    energy = 10.0**IC_LOG10_E
    columns, weights = ic_upgoing_columns()
    n_nucleon = nucleon_number_density()
    out = np.empty(energy.size)
    for i, e_nu in enumerate(energy):
        rung_energy, rung_weight = regenerated_transmission(float(e_nu), columns, CROSS_SECTION)
        rung_length = np.interp(
            np.log10(rung_energy), IC_LOG10_E, length_km, left=0.0, right=length_km[-1]
        )
        rung_length[(1.0 - MEAN_INELASTICITY) * rung_energy <= threshold_gev] = 0.0
        rung_radius = (
            IC_RADIUS_KM
            if reach_km is None
            else light_reach_radius_km(
                IC_RADIUS_KM, (1.0 - MEAN_INELASTICITY) * rung_energy, reach_km, pivot_gev
            )
        )
        rung_rate = (
            n_nucleon * CROSS_SECTION.cc(rung_energy)
            * ic_target_volume_cm3(rung_length, rung_radius)
        )
        out[i] = np.average((rung_weight * rung_rate[:, None]).sum(axis=0), weights=weights)
    return out


def ic_effective_area_tau_channel(length_km: np.ndarray, threshold_gev: float) -> np.ndarray:
    """nu_tau -> tau -> mu channel, upgoing-averaged, static footprint [cm^2]."""
    energy = 10.0**IC_LOG10_E
    columns, weights = ic_upgoing_columns()
    n_nucleon = nucleon_number_density()
    muon_fraction = MEAN_Z * (1.0 - MEAN_INELASTICITY)
    out = np.empty(energy.size)
    for i, e_nu in enumerate(energy):
        rung_energy, rung_weight = flavour_transmission(
            float(e_nu), columns, CROSS_SECTION, flavour="tau"
        )
        rung_length = np.interp(
            np.log10(rung_energy * muon_fraction / (1.0 - MEAN_INELASTICITY)),
            IC_LOG10_E, length_km, left=0.0, right=length_km[-1],
        )
        rung_length[muon_fraction * rung_energy <= threshold_gev] = 0.0
        rung_rate = (
            n_nucleon * CROSS_SECTION.cc(rung_energy)
            * ic_target_volume_cm3(rung_length) * BR_TAU_TO_MU
        )
        out[i] = np.average((rung_weight * rung_rate[:, None]).sum(axis=0), weights=weights)
    return out


def icecube_curves(
    data_dir: pathlib.Path, threshold_gev: float
) -> tuple[dict[str, np.ndarray], float]:
    """Published, parameter-free, and fitted-reach IceCube effective areas [cm^2].

    Returns
    -------
    curves : dict of str -> np.ndarray
        Keyed by ``"published"``, ``"first principles"`` and ``"fitted reach"``.
    reach_km : float
        Fitted growth of the light reach per e-fold [km].
    """
    published = icecube_published(data_dir)
    energy_mu = (1.0 - MEAN_INELASTICITY) * 10.0**IC_LOG10_E
    length = stochastic_muon_range_km(energy_mu, threshold_gev)
    tau = ic_effective_area_tau_channel(length, threshold_gev)
    base = ic_effective_area_regenerated(length, threshold_gev) + tau

    band = (IC_LOG10_E >= IC_FIT_BAND[0]) & (IC_LOG10_E <= IC_FIT_BAND[1])
    ratio = np.full_like(IC_LOG10_E, np.nan)
    ratio[band] = (published / base)[band]
    reach_km, pivot_gev = fit_reach_law(
        IC_LOG10_E, ic_required_radius_km(ratio, length), IC_RADIUS_KM
    )
    fitted = ic_effective_area_regenerated(
        length, threshold_gev, reach_km=reach_km, pivot_gev=pivot_gev
    ) + tau

    for name, curve in (("first principles", base), ("fitted reach", fitted)):
        residual = np.log10(published[band] / curve[band])
        print(f"  IceCube {name:17s}: {np.std(residual):.3f} dex rms against the published table")
    return {"published": published, "first principles": base, "fitted reach": fitted}, reach_km


# ---------------------------------------------------------------------------
# KM3NeT/ARCA230
# ---------------------------------------------------------------------------


def zenith_grid() -> tuple[np.ndarray, np.ndarray]:
    """Zenith samples and normalized weights over the full sky."""
    edges = np.linspace(1.0, -1.0, N_ZENITH + 1)
    cos_theta = 0.5 * (edges[:-1] + edges[1:])
    theta_deg = np.rad2deg(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    return theta_deg, np.full(N_ZENITH, 1.0 / N_ZENITH)


def projected_area_km2(
    theta_deg: np.ndarray, radius_km: float | np.ndarray, n_blocks: int
) -> np.ndarray:
    """Projected area of upright cylindrical building blocks [km^2]."""
    theta = np.deg2rad(theta_deg)
    cap = np.pi * np.asarray(radius_km) ** 2 * np.abs(np.cos(theta))
    side = 2.0 * np.asarray(radius_km) * BLOCK_HEIGHT_KM * np.sin(theta)
    return n_blocks * (cap + side)


def upstream_column_km(theta_deg: np.ndarray, depth_km: float) -> np.ndarray:
    """Sea-water column available upstream of the detector [km of water]."""
    cos_theta = np.cos(np.deg2rad(theta_deg))
    with np.errstate(divide="ignore", invalid="ignore"):
        downgoing = np.where(cos_theta > 0.0, depth_km / np.maximum(cos_theta, 1e-6), np.inf)
    return np.minimum(downgoing, MAX_SEA_PATH_KM)


def earth_column_g_cm2(theta_deg: np.ndarray, depth_km: float) -> np.ndarray:
    """Column depth traversed by the neutrino before reaching the detector [g cm^-2]."""
    water_km = upstream_column_km(theta_deg, depth_km)
    water = np.where(np.isfinite(water_km), water_km, 0.0) * CM_PER_KM * RHO_SEA_G_CM3
    earth = np.array([prem_column(float(t) - 90.0) if t > 90.0 else 0.0 for t in theta_deg])
    return np.where(theta_deg > 90.0, earth, water)


def build_length_table(
    energy_mu_gev: np.ndarray, threshold_gev: float
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative first-passage length against depth, on the muon-energy grid."""
    deterministic = muon_range_km(energy_mu_gev, threshold_gev)
    ell_km = np.linspace(0.0, 2.5 * float(np.max(deterministic)), N_ELL)
    b_mu = np.atleast_1d(drift_coefficient(energy_mu_gev))
    d_mu = np.atleast_1d(diffusion_coefficient(energy_mu_gev))
    cumulative = np.zeros((energy_mu_gev.size, N_ELL))
    for i, eps in enumerate(energy_mu_gev):
        if eps <= threshold_gev:
            continue
        cdf = log_loss_cdf(np.log(eps / threshold_gev), ell_km, float(b_mu[i]), float(d_mu[i]))
        cumulative[i] = cumulative_trapezoid(cdf, ell_km, initial=0.0)
    return ell_km, cumulative


def truncated_range_km(
    energy_mu_gev: np.ndarray,
    column_km: np.ndarray,
    grid_log10_e: np.ndarray,
    ell_km: np.ndarray,
    cumulative_km: np.ndarray,
) -> np.ndarray:
    """Truncated first-passage range ``E[tau ^ X]``, by table lookup [km]."""
    x = np.minimum(np.atleast_1d(column_km), ell_km[-1])
    per_row = np.array([np.interp(x, ell_km, row) for row in cumulative_km])
    log10_e = np.log10(np.atleast_1d(energy_mu_gev))
    out = np.empty((log10_e.size, x.size))
    for j in range(x.size):
        out[:, j] = np.interp(log10_e, grid_log10_e, per_row[:, j])
    return np.clip(out, 0.0, None)


def arca_effective_area(
    radius_km: float,
    n_blocks: int,
    threshold_gev: float,
    depth_km: float,
    grid_log10_e: np.ndarray,
    ell_km: np.ndarray,
    cumulative_km: np.ndarray,
    flavour: str,
    reach_km: float | None = None,
    pivot_gev: float = 1.0e6,
) -> np.ndarray:
    """Sky-averaged effective area for one parent flavour [cm^2]."""
    theta_deg, weights = zenith_grid()
    columns = earth_column_g_cm2(theta_deg, depth_km)
    available_km = upstream_column_km(theta_deg, depth_km)
    static_area_km2 = projected_area_km2(theta_deg, radius_km, n_blocks)
    static_v_det_km3 = n_blocks * np.pi * radius_km**2 * BLOCK_HEIGHT_KM
    n_nucleon = nucleon_number_density(RHO_SEA_G_CM3)

    muon_fraction = 1.0 - MEAN_INELASTICITY
    if flavour == "tau":
        muon_fraction *= MEAN_Z

    out = np.empty(ARCA_LOG10_E.size)
    for i, e_nu in enumerate(10.0**ARCA_LOG10_E):
        if flavour == "tau":
            rung_energy, rung_weight = flavour_transmission(
                float(e_nu), columns, CROSS_SECTION, flavour="tau"
            )
        else:
            rung_energy, rung_weight = regenerated_transmission(float(e_nu), columns, CROSS_SECTION)
        length = truncated_range_km(
            rung_energy * muon_fraction, available_km, grid_log10_e, ell_km, cumulative_km
        )
        length[rung_energy * muon_fraction <= threshold_gev, :] = 0.0

        if reach_km is None:
            area_km2 = static_area_km2[None, :]
            v_det_km3 = static_v_det_km3
        else:
            r_eff = light_reach_radius_km(
                radius_km, rung_energy * muon_fraction, reach_km, pivot_gev
            )[:, None]
            area_km2 = projected_area_km2(theta_deg[None, :], r_eff, n_blocks)
            v_det_km3 = n_blocks * np.pi * r_eff**2 * BLOCK_HEIGHT_KM

        volume_km3 = area_km2 * length + v_det_km3
        rate = n_nucleon * CROSS_SECTION.cc(rung_energy)[:, None] * volume_km3 * CM_PER_KM**3
        if flavour == "tau":
            rate = rate * BR_TAU_TO_MU
        out[i] = np.average((rung_weight * rate).sum(axis=0), weights=weights)
    return out


def arca230_published(log10_e: np.ndarray) -> np.ndarray:
    """Digitized full-ARCA nu_mu trigger-level effective area [cm^2]."""
    table = np.genfromtxt(_ARCA230_TRIGGER_TABLE, delimiter=",", comments="#")
    table = table[np.argsort(table[:, 0])]
    return 1.0e4 * 10.0 ** np.interp(
        log10_e, np.log10(table[:, 0]), np.log10(table[:, 1]), left=np.nan, right=np.nan
    )


def required_footprint_radius_km(ratio: np.ndarray, radius_km: float, n_blocks: int) -> np.ndarray:
    """Footprint radius that would scale the projected area by ``ratio``."""
    theta_deg, weights = zenith_grid()

    def mean_area(r: float) -> float:
        return float(np.average(projected_area_km2(theta_deg, r, n_blocks), weights=weights))

    base = mean_area(radius_km)
    out = np.full(np.shape(ratio), np.nan)
    for i, r in enumerate(np.atleast_1d(ratio)):
        if not np.isfinite(r) or r <= 0.0:
            continue
        out[i] = brentq(lambda x: mean_area(x) - r * base, 1.0e-3, 50.0)
    return out


def arca230_curves(threshold_gev: float, depth_km: float) -> tuple[dict[str, np.ndarray], float]:
    """Published, parameter-free, and fitted-reach ARCA230 effective areas [cm^2].

    Returns
    -------
    curves : dict of str -> np.ndarray
        Keyed by ``"published"``, ``"first principles"`` and ``"fitted reach"``.
    reach_km : float
        Fitted growth of the light reach per e-fold [km].
    """
    table_log10_e = np.arange(2.0, 10.01, 0.2)
    ell_km, cumulative_km = build_length_table(10.0**table_log10_e, threshold_gev)
    kwargs = dict(
        threshold_gev=threshold_gev, depth_km=depth_km,
        grid_log10_e=table_log10_e, ell_km=ell_km, cumulative_km=cumulative_km,
    )
    tau = arca_effective_area(ARCA230_RADIUS_KM, N_BLOCKS_FULL, flavour="tau", **kwargs)
    base = arca_effective_area(ARCA230_RADIUS_KM, N_BLOCKS_FULL, flavour="mu", **kwargs) + tau

    published = arca230_published(ARCA_LOG10_E)
    band = (ARCA_LOG10_E >= ARCA_FIT_BAND[0]) & (ARCA_LOG10_E <= ARCA_FIT_BAND[1])
    required = np.full(ARCA_LOG10_E.size, np.nan)
    required[band] = required_footprint_radius_km(
        (published / base)[band], ARCA230_RADIUS_KM, N_BLOCKS_FULL
    )
    reach_km, pivot_gev = fit_reach_law(ARCA_LOG10_E, required, ARCA230_RADIUS_KM)
    fitted = arca_effective_area(
        ARCA230_RADIUS_KM, N_BLOCKS_FULL, flavour="mu",
        reach_km=reach_km, pivot_gev=pivot_gev, **kwargs
    ) + arca_effective_area(
        ARCA230_RADIUS_KM, N_BLOCKS_FULL, flavour="tau",
        reach_km=reach_km, pivot_gev=pivot_gev, **kwargs
    )

    for name, curve in (("first principles", base), ("fitted reach", fitted)):
        residual = np.log10(published[band] / curve[band])
        print(f"  ARCA230 {name:17s}: {np.std(residual):.3f} dex rms against the trigger curve")
    return {"published": published, "first principles": base, "fitted reach": fitted}, reach_km


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def make_figure(
    curves: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]], out_path: pathlib.Path
) -> None:
    """Draw both detectors on one panel, with the two legends factorized.

    Colour says which of the three curves a line is, line style says which
    detector, so the six lines need only three plus two legend entries rather
    than six.

    Parameters
    ----------
    curves : dict
        Detector name -> (``log10(E_nu / GeV)`` grid, curve dict keyed as in
        :data:`CURVE_COLOR`).
    out_path : pathlib.Path
        Output file; both ``.pdf`` and ``.png`` are written.
    """
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.0, 3.0))

        for detector, (log10_e, detector_curves) in curves.items():
            for name, curve in detector_curves.items():
                ax.plot(log10_e, curve, color=CURVE_COLOR[name], ls=DETECTOR_STYLE[detector],
                        lw=CURVE_WIDTH[name], zorder=CURVE_ZORDER[name])

        ax.set_yscale("log")
        ax.set_xlim(3.0, 10.0)
        ax.set_ylim(3.0e2, 1.0e9)
        ax.set_xlabel(r"$\log_{10}(E_\nu\,/\,\mathrm{GeV})$")
        ax.set_ylabel(r"$A_{\rm eff}$ [cm$^2$]")

        curve_legend = ax.legend(
            handles=[
                plt.Line2D([], [], color=c, lw=CURVE_WIDTH[n], label=n)
                for n, c in CURVE_COLOR.items()
            ],
            loc="upper left", fontsize=6, frameon=False, handlelength=2.0,
        )
        ax.add_artist(curve_legend)
        ax.legend(
            handles=[
                plt.Line2D([], [], color="k", ls=s, lw=1.0, label=d)
                for d, s in DETECTOR_STYLE.items()
            ],
            loc="lower right", fontsize=6, frameon=False, handlelength=2.4,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png"):
            path = out_path.with_suffix(suffix)
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {path.resolve()}")
        plt.close(fig)


def report(curves: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]]) -> None:
    """Print the implied selection efficiency, which a ceiling holds below one."""
    for detector, (log10_e, detector_curves) in curves.items():
        ratio = detector_curves["published"] / detector_curves["first principles"]
        valid = np.isfinite(ratio)
        print(f"\n  {detector}: published / first principles")
        for target in (4.0, 5.0, 6.0, 7.0, 8.0):
            i = int(np.argmin(np.abs(log10_e - target)))
            if valid[i]:
                print(f"    log10(E/GeV) = {log10_e[i]:4.1f}   {ratio[i]:7.3f}")
        print(f"    range over the overlap: {np.nanmin(ratio[valid]):.3f} - "
              f"{np.nanmax(ratio[valid]):.3f}")


def main() -> None:
    args = parse_args()

    print(f"Building the IceCube effective area, loading DR2 from: {args.data_dir}")
    ic_curves, ic_reach_km = icecube_curves(args.data_dir, args.threshold)

    print("Building the ARCA230 effective area ...")
    arca_curves, arca_reach_km = arca230_curves(args.threshold, ARCA_DEPTH_KM)

    print(f"\n  fitted reach: IceCube {ic_reach_km * 1e3:.1f} m/e-fold, "
          f"ARCA230 {arca_reach_km * 1e3:.1f} m/e-fold")

    curves = {
        "IceCube": (IC_LOG10_E, ic_curves),
        "KM3NeT/ARCA230": (ARCA_LOG10_E, arca_curves),
    }
    report(curves)
    make_figure(curves, args.out)


if __name__ == "__main__":
    main()
