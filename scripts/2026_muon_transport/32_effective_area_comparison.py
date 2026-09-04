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

from softpaws.data.published import arca230_trigger_level_aeff, icecube_dr2_aeff, interpolate_aeff
from softpaws.detectors import ARCA230, ICECUBE, MAX_UPSTREAM_KM
from softpaws.response import effective_area as engine
from softpaws.response.effective_area import default_cross_section, fit_reach_law
from softpaws.transport.attenuation import (  # noqa: F401  (example 45 reaches these via ex32)
    flavour_transmission,
    regenerated_transmission,
)
from softpaws.transport.earth import neutrino_column_g_cm2, overburden_km
from softpaws.transport.earth import zenith_grid as earth_zenith_grid
from softpaws.transport.soft_volume import DEFAULT_MUON_THRESHOLD_GEV, stochastic_muon_range_km
from softpaws.transport.source import MEAN_INELASTICITY
from softpaws.utils.constants import RHO_WATER_G_CM3

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_DATA_DIR = _HERE.parents[1] / "src" / "softpaws" / "data" / "dataverse_files"
_DEFAULT_OUT_DIR = _HERE / "output"

CROSS_SECTION = default_cross_section()

# --- IceCube, example 28's numbers ------------------------------------------
# IceCube as an upright hexagonal prism: ~1 km^2 of footprint by 1 km of
# instrumented height, giving V_det = 1.00 km^3 exactly. IC_RADIUS_KM is the
# area-equivalent radius of the hexagon, so pi R^2 is the footprint.
IC_HEIGHT_KM = ICECUBE.height_km
IC_N_SIDES = ICECUBE.n_sides
IC_RADIUS_KM = ICECUBE.radius_km  # ~0.564 km
IC_FOOTPRINT_KM2 = float(np.pi * IC_RADIUS_KM**2)
IC_LOG10_E = engine.IC_LOG10_E
IC_FIT_BAND = engine.IC_FIT_BAND  # band the reach law is calibrated over
N_DEC = engine.N_DEC

# --- ARCA230, example 30's numbers ------------------------------------------
ARCA230_RADIUS_KM = ARCA230.radius_km
BLOCK_HEIGHT_KM = ARCA230.height_km
N_BLOCKS_FULL = ARCA230.n_blocks
ARCA_DEPTH_KM = ARCA230.depth_km
ARCA_LOG10_E = engine.ARCA_LOG10_E
ARCA_FIT_BAND = engine.ARCA_FIT_BAND  # digitized trigger curve saturates past the top
MAX_SEA_PATH_KM = MAX_UPSTREAM_KM
RHO_SEA_G_CM3 = RHO_WATER_G_CM3
N_ZENITH = engine.N_ZENITH

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


# ---------------------------------------------------------------------------
# IceCube. Thin wrappers over softpaws.response.effective_area with this
# script's constants; examples 41, 43, 44 and 45 reach the engine through them.
# ---------------------------------------------------------------------------


def icecube_published(data_dir: pathlib.Path) -> np.ndarray:
    """Livetime-weighted published IceCube effective area, upgoing sky [cm^2]."""
    return icecube_dr2_aeff(data_dir, IC_LOG10_E)[0]


def ic_upgoing_columns() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PREM columns, weights and zenith cosines; see :func:`engine.ic_upgoing_columns`."""
    return engine.ic_upgoing_columns(N_DEC)


def ic_target_volume_cm3(
    length_km: np.ndarray,
    radius_km: float | np.ndarray = IC_RADIUS_KM,
    cos_theta: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Prism target volume [cm^3]; see :func:`engine.ic_target_volume_cm3`."""
    return engine.ic_target_volume_cm3(length_km, radius_km, cos_theta, IC_HEIGHT_KM, IC_N_SIDES)


def ic_mean_target_volume_cm3(
    length_km: np.ndarray, radius_km: float | np.ndarray = IC_RADIUS_KM
) -> np.ndarray:
    """Upgoing-averaged target volume [cm^3]; see :func:`engine.ic_mean_target_volume_cm3`."""
    return engine.ic_mean_target_volume_cm3(
        length_km, radius_km, N_DEC, IC_HEIGHT_KM, IC_N_SIDES
    )


def ic_required_radius_km(ratio: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Radius scaling the target volume by ``ratio``; see :func:`engine.ic_required_radius_km`."""
    return engine.ic_required_radius_km(ratio, lengths, N_DEC, IC_HEIGHT_KM, IC_N_SIDES)


def ic_effective_area_regenerated(
    length_km: np.ndarray,
    threshold_gev: float,
    reach_km: float | None = None,
    pivot_gev: float = 1.0e6,
) -> np.ndarray:
    """Direct nu_mu channel [cm^2]; see :func:`engine.ic_effective_area_regenerated`."""
    return engine.ic_effective_area_regenerated(
        length_km, threshold_gev, reach_km, pivot_gev, IC_LOG10_E, CROSS_SECTION, N_DEC,
        IC_RADIUS_KM, IC_HEIGHT_KM, IC_N_SIDES,
    )


def ic_effective_area_tau_channel(length_km: np.ndarray, threshold_gev: float) -> np.ndarray:
    """nu_tau -> tau -> mu channel [cm^2]; see :func:`engine.ic_effective_area_tau_channel`."""
    return engine.ic_effective_area_tau_channel(
        length_km, threshold_gev, IC_LOG10_E, CROSS_SECTION, N_DEC,
        IC_RADIUS_KM, IC_HEIGHT_KM, IC_N_SIDES,
    )


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


def zenith_grid(
    cos_range: tuple[float, float] = (-1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Zenith samples and solid-angle weights; see :func:`softpaws.transport.earth.zenith_grid`."""
    return earth_zenith_grid(N_ZENITH, cos_range)


def projected_area_km2(
    theta_deg: np.ndarray, radius_km: float | np.ndarray, n_blocks: int,
    height_km: float | np.ndarray = BLOCK_HEIGHT_KM,
) -> np.ndarray:
    """Cylinder projected area [km^2]; see :func:`engine.cylinder_projected_area_km2`."""
    return engine.cylinder_projected_area_km2(theta_deg, radius_km, n_blocks, height_km)


def upstream_column_km(theta_deg: np.ndarray, depth_km: float) -> np.ndarray:
    """Medium available upstream [km]; see :func:`softpaws.transport.earth.overburden_km`."""
    return overburden_km(np.cos(np.deg2rad(theta_deg)), depth_km, MAX_SEA_PATH_KM)


def earth_column_g_cm2(theta_deg: np.ndarray, depth_km: float) -> np.ndarray:
    """Neutrino column [g cm^-2]; see :func:`softpaws.transport.earth.neutrino_column_g_cm2`."""
    return neutrino_column_g_cm2(
        np.cos(np.deg2rad(theta_deg)), depth_km, RHO_SEA_G_CM3, MAX_SEA_PATH_KM
    )


def truncated_range_km(
    energy_mu_gev: np.ndarray,
    column_km: np.ndarray,
    threshold_gev: float,
    kernel_evaluation: str = "running",
) -> np.ndarray:
    """Truncated first-passage range [km]; see :func:`engine.truncated_range_km`."""
    return engine.truncated_range_km(energy_mu_gev, column_km, threshold_gev, kernel_evaluation)


def arca_effective_area(
    radius_km: float,
    n_blocks: int,
    threshold_gev: float,
    depth_km: float,
    flavour: str,
    kernel_evaluation: str = "running",
    reach_km: float | None = None,
    pivot_gev: float = 1.0e6,
) -> np.ndarray:
    """Sky-averaged effective area, one flavour [cm^2]; see :func:`engine.arca_effective_area`."""
    return engine.arca_effective_area(
        radius_km, n_blocks, threshold_gev, depth_km, flavour, kernel_evaluation,
        reach_km, pivot_gev, log10_e=ARCA_LOG10_E, height_km=BLOCK_HEIGHT_KM,
        n_zenith=N_ZENITH, density_g_cm3=RHO_SEA_G_CM3, max_upstream_km=MAX_SEA_PATH_KM,
        cross_section=CROSS_SECTION,
    )


def arca230_published(log10_e: np.ndarray) -> np.ndarray:
    """Digitized full-ARCA nu_mu trigger-level effective area [cm^2]."""
    return interpolate_aeff(log10_e, *arca230_trigger_level_aeff())


def required_footprint_radius_km(ratio: np.ndarray, radius_km: float, n_blocks: int) -> np.ndarray:
    """Footprint radius scaling the projected area by ``ratio``; see the engine."""
    return engine.required_footprint_radius_km(
        ratio, radius_km, n_blocks, BLOCK_HEIGHT_KM, N_ZENITH
    )


def arca230_curves(threshold_gev: float, depth_km: float) -> tuple[dict[str, np.ndarray], float]:
    """Published, parameter-free, and fitted-reach ARCA230 effective areas [cm^2].

    Returns
    -------
    curves : dict of str -> np.ndarray
        Keyed by ``"published"``, ``"first principles"`` and ``"fitted reach"``.
    reach_km : float
        Fitted growth of the light reach per e-fold [km].
    """
    kwargs = dict(threshold_gev=threshold_gev, depth_km=depth_km)
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
