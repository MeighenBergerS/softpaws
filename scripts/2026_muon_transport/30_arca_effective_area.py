"""Example 30 -- predicting the KM3NeT/ARCA effective area and effective volume.

Example 28 builds the per-neutrino-energy effective area of Sec. VI of the draft
and benchmarks it against IceCube. Nothing in that construction is specific to
ice: the exponent ``Phi``, the first-passage range ``L = E[tau(w_star)]``, and
the flavour-dependent Earth transmission are properties of the loss kernel and
of the Earth, not of the site. Porting them to ARCA is therefore a matter of
supplying the *instrument* numbers, and this example makes the list of those
numbers explicit by requiring every one of them as a named constant.

Three of them are genuinely new relative to the IceCube case.

**A finite overburden.** Eq. (16) of the draft integrates the first-passage
probability to infinite depth. That is right for IceCube's upgoing hemisphere,
where the Earth supplies more column than any muon can survive, and wrong for
ARCA, which sits under only ~3.2 km of sea water. For a downgoing neutrino the
muon cannot be produced further upstream than the sea surface, so the length is
the *truncated* first passage

    L(E_nu, X) = Integral_0^X d_ell P[W(ell) < w_star] = E[tau(w_star) ^ X],

which is what :func:`truncated_range_km` computes. Above ~1 PeV this cuts the
downgoing acceptance by a factor of a few, and it is the single largest
ARCA-specific effect.

**An anisotropic detector.** The draft carries one number ``A_proj``. ARCA is a
flat cylinder -- footprint radius ~517 m per building block against an
instrumented height of 632 m -- so its projected area runs from ``pi R^2``
overhead to ``2 R h`` at the horizon and the two differ by 30%. The isotropic
sphere of example 28 has no such freedom.

**A 4 pi convention.** The published ARCA numbers used here are averaged over
the whole sky, not over the upgoing hemisphere. Downgoing directions have no
Earth in front of them, which is what keeps ARCA's effective area rising past
100 PeV where the upgoing sky has long gone opaque.

Note that the medium density cancels out of the effective area itself:
``n_N sigma A_proj L`` is ``rho N_A sigma A_proj (X / rho)``, so only the column
``X`` matters and sea water versus ice is irrelevant to the normalization. It
matters here only through the geometry, by setting how much column the 3.2 km of
water above the detector actually is.

Two published references are compared against, and they are not the same
measurement.

``ARCA230``
    The full detector, two building blocks. Compared against the ``nu_mu``
    effective area at **trigger level** of the KM3NeT Collaboration,
    Eur. Phys. J. C 84 (2024) 885 [arXiv:2402.08363] Fig. 7, digitized over
    ``10^3``-``10^8`` GeV (:func:`arca230_published_trigger`). Trigger level is
    the strongest test available: it carries no analysis cuts, so it is the
    largest area the instrument reports. The analytic parametrization of the
    same figure that circulates in the literature runs a flat factor ``1.95``
    higher (:func:`arca230_quoted_fit`) and should not be used.

``ARCA21``
    The 21-line detector that recorded KM3-230213A. Compared against the
    tabulated all-flavour, sky-averaged, bright-track effective area released
    with KM3NeT Collaboration, Nature 638 (2025) 376
    (``src/softpaws/data/km3net/``), which runs to ``10^11`` GeV.

The model is a geometric ceiling in both cases, so the meaningful statement is
the implied selection efficiency ``A_eff^published / A_eff^model``, which must
not exceed one.

Usage
-----
    python examples/30_arca_effective_area.py
    python examples/30_arca_effective_area.py --threshold 1e4
    python examples/30_arca_effective_area.py --depth-km 3.2
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.data import published
from softpaws.data.published import (
    arca21_bright_track_aeff,
    arca230_trigger_level_aeff,
    interpolate_aeff,
)
from softpaws.detectors import ARCA21, ARCA230, MAX_UPSTREAM_KM
from softpaws.response import effective_area as engine
from softpaws.response.effective_area import default_cross_section, fit_reach_law
from softpaws.transport.earth import neutrino_column_g_cm2, overburden_km
from softpaws.transport.earth import zenith_grid as earth_zenith_grid
from softpaws.transport.muon_range import DEFAULT_MUON_THRESHOLD_GEV, stochastic_muon_range_km
from softpaws.transport.source import MEAN_INELASTICITY
from softpaws.utils.constants import CM_PER_KM, RHO_WATER_G_CM3

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"

# ---------------------------------------------------------------------------
# Instrument numbers. Every constant below is an ARCA input the draft does not
# supply; each carries the published source it was taken from.
# ---------------------------------------------------------------------------

# Building-block footprint radius [km] and instrumented height [km].
#
# Measured from the as-built ARCA21 geometry shipped with the KM3-230213A
# release (``data/supplementary/detector/detector.json.gz``) rather than taken
# from the design values, which differ: 21 detection units, 18 optical modules
# each, mean vertical spacing 36.8 m, optical modules spanning 58.1-689.6 m
# above the anchor, so 631.5 m of instrumented height. The convex hull of the
# unit positions is 1.53e5 m^2, i.e. 7292 m^2 per unit, which scaled to the
# 115 units of a full block gives 8.39e5 m^2 and R = 517 m.
#
# The LoI (arXiv:1601.07459) quotes R = 500 m and z = 612 m for 0.48 km^3 per
# block; the as-built array is slightly wider and taller than that.
BLOCK_RADIUS_KM = ARCA230.radius_km
BLOCK_HEIGHT_KM = ARCA230.height_km
N_BLOCKS_FULL = ARCA230.n_blocks

# ARCA21: the equivalent radius of the measured convex hull above. Close to the
# 214 m that scaling a full block by sqrt(21/115) would give, so the 21 units
# were already deployed as a compact cluster rather than spread over the full
# footprint.
ARCA21_RADIUS_KM = ARCA21.radius_km

# Depth of the instrumented volume's centre below the sea surface [km]. Seabed
# at 3500 m at the Capo Passero site, with the instrumented span standing on it.
DETECTOR_DEPTH_KM = ARCA230.depth_km

# Longest sea-water path a near-horizontal muon can have [km]. Only a cap on the
# 1/cos(theta) divergence; it exceeds every muon range in the problem, so the
# result is insensitive to it.
MAX_SEA_PATH_KM = MAX_UPSTREAM_KM

# Sea water at the site, used to turn the geometric path length into a column.
RHO_SEA_G_CM3 = RHO_WATER_G_CM3

# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------

# 0.2 dex.
COMMON_LOG10_E = engine.ARCA_LOG10_E

# Zenith sampling. theta = 0 is straight down through the sea, theta = 180 is
# straight up through the Earth.
N_ZENITH = engine.N_ZENITH

CROSS_SECTION = default_cross_section()

# BGR18 stops just below 10^10 GeV; beyond that both the cross section and the
# PROPOSAL transport coefficients are extrapolations.
TABULATED_TOP_LOG10_E = 9.98

# Top of the band the reach law is fitted over. The digitized trigger curve
# saturates at the edge of the published figure in its last tenth of a decade,
# so the fit stops short of it.
FIT_TOP_LOG10_E = engine.ARCA_FIT_BAND[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_MUON_THRESHOLD_GEV,
        help="Muon selection threshold [GeV]. No ARCA equivalent of the DR2 "
        "smearing matrix is public, so the IceCube value is carried over.",
    )
    parser.add_argument(
        "--depth-km",
        type=float,
        default=DETECTOR_DEPTH_KM,
        help="Depth of the instrumented volume below the sea surface [km].",
    )
    parser.add_argument(
        "--block-radius-km",
        type=float,
        default=BLOCK_RADIUS_KM,
        help="Footprint radius of one ARCA building block [km].",
    )
    parser.add_argument(
        "--block-height-km",
        type=float,
        default=BLOCK_HEIGHT_KM,
        help="Instrumented height of a detection unit [km]. The as-built optical "
        "modules span 632 m; an acceptance height a little beyond the end modules "
        "is defensible, so this is worth varying.",
    )
    parser.add_argument(
        "--kernel-evaluation",
        choices=("running", "frozen"),
        default="running",
        help=(
            "Where along the descent the loss kernel is read. 'running' follows it "
            "down; 'frozen' holds the production-energy value, which is the closed "
            "form of Eq. (C4) as written."
        ),
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=_DEFAULT_OUT_DIR / "30_arca_effective_area.pdf",
        help="Output file for the figure.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def zenith_grid(
    cos_range: tuple[float, float] = (-1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Zenith samples and solid-angle weights; see :func:`softpaws.transport.earth.zenith_grid`."""
    return earth_zenith_grid(N_ZENITH, cos_range)


def projected_area_km2(
    theta_deg: np.ndarray,
    radius_km: float | np.ndarray,
    n_blocks: int,
) -> np.ndarray:
    """Cylinder projected area [km^2] at the current ``BLOCK_HEIGHT_KM``.

    See :func:`softpaws.response.effective_area.cylinder_projected_area_km2`.
    ``BLOCK_HEIGHT_KM`` is read at call time so that ``--block-height-km``
    reaches every projection.
    """
    return engine.cylinder_projected_area_km2(theta_deg, radius_km, n_blocks, BLOCK_HEIGHT_KM)


def upstream_column_km(theta_deg: np.ndarray, depth_km: float) -> np.ndarray:
    """Medium available upstream [km]; see :func:`softpaws.transport.earth.overburden_km`."""
    return overburden_km(np.cos(np.deg2rad(theta_deg)), depth_km, MAX_SEA_PATH_KM)


def earth_column_g_cm2(theta_deg: np.ndarray, depth_km: float) -> np.ndarray:
    """Neutrino column [g cm^-2]; see :func:`softpaws.transport.earth.neutrino_column_g_cm2`."""
    return neutrino_column_g_cm2(
        np.cos(np.deg2rad(theta_deg)), depth_km, RHO_SEA_G_CM3, MAX_SEA_PATH_KM
    )


# ---------------------------------------------------------------------------
# Truncated first-passage range
# ---------------------------------------------------------------------------


def truncated_range_km(
    energy_mu_gev: np.ndarray,
    column_km: np.ndarray,
    threshold_gev: float,
    kernel_evaluation: str = "running",
) -> np.ndarray:
    """Truncated first-passage range ``E[tau ^ X]`` [km], shape ``(n, m)``.

    See :func:`softpaws.response.effective_area.truncated_range_km`, which
    is the closed form; the private Gil-Pelaez table this once built read the
    loss moments off the two-moment family, frozen at production.
    """
    return engine.truncated_range_km(energy_mu_gev, column_km, threshold_gev, kernel_evaluation)


# ---------------------------------------------------------------------------
# Effective area
# ---------------------------------------------------------------------------


def effective_area(
    radius_km: float,
    n_blocks: int,
    threshold_gev: float,
    depth_km: float,
    flavour: str,
    kernel_evaluation: str = "running",
    cos_range: tuple[float, float] = (-1.0, 1.0),
    truncate: bool = True,
    reach_km: float | None = None,
    pivot_gev: float = 1.0e6,
) -> np.ndarray:
    """Solid-angle-averaged effective area for one parent flavour [cm^2].

    See :func:`softpaws.response.effective_area.arca_effective_area`. This
    example keeps the instrumented volume of a sub-threshold rung
    (``mask_subthreshold=False``), as its printed tables were built.
    """
    return engine.arca_effective_area(
        radius_km, n_blocks, threshold_gev, depth_km, flavour, kernel_evaluation,
        reach_km, pivot_gev, log10_e=COMMON_LOG10_E, height_km=BLOCK_HEIGHT_KM,
        cos_range=cos_range, n_zenith=N_ZENITH, truncate=truncate, mask_subthreshold=False,
        density_g_cm3=RHO_SEA_G_CM3, max_upstream_km=MAX_SEA_PATH_KM, cross_section=CROSS_SECTION,
    )


# ---------------------------------------------------------------------------
# Published references
# ---------------------------------------------------------------------------


def arca230_published_trigger(log10_e: np.ndarray) -> np.ndarray:
    """Digitized full-ARCA ``nu_mu`` effective area at trigger level [cm^2].

    See :func:`softpaws.data.published.arca230_trigger_level_aeff`; ``NaN``
    outside the digitized range.
    """
    return interpolate_aeff(log10_e, *arca230_trigger_level_aeff())


def arca230_quoted_fit(log10_e: np.ndarray) -> np.ndarray:
    """Analytic parametrization quoted in the literature, a factor 1.95 high [cm^2].

    See :func:`softpaws.data.published.arca230_quoted_fit`.
    """
    return published.arca230_quoted_fit(log10_e)


def arca21_published(log10_e: np.ndarray) -> np.ndarray:
    """Tabulated ARCA21 bright-track, all-flavour, sky-averaged area [cm^2].

    See :func:`softpaws.data.published.arca21_bright_track_aeff`; ``NaN``
    where the table is zero or absent.
    """
    return interpolate_aeff(log10_e, *arca21_bright_track_aeff())


# ---------------------------------------------------------------------------
# Reporting and plotting
# ---------------------------------------------------------------------------


def effective_volume_km3(aeff_cm2: np.ndarray) -> np.ndarray:
    """Effective target volume implied by an effective area [km^3 of sea water].

    See :func:`softpaws.response.effective_area.effective_volume_km3`.
    """
    return engine.effective_volume_km3(aeff_cm2, COMMON_LOG10_E, RHO_SEA_G_CM3, CROSS_SECTION)


def report(curves: dict[str, np.ndarray], lengths: dict[str, np.ndarray]) -> None:
    """Print the lengths, the volumes, and the implied efficiencies."""
    print(f"\n{'log10(E_nu/GeV)':>16} {'L_free':>8} {'L_down':>8} {'ratio':>7}")
    print(f"{'':>16} {'[km]':>8} {'[km]':>8}")
    for log10_e in (5.0, 6.0, 7.0, 8.0, 9.0, 10.0):
        i = int(np.argmin(np.abs(COMMON_LOG10_E - log10_e)))
        free, down = lengths["free"][i], lengths["vertical_down"][i]
        print(f"{COMMON_LOG10_E[i]:16.1f} {free:8.2f} {down:8.2f} {down / free:7.3f}")
    print("  L_free: Eq. (16) untruncated. L_down: truncated at the vertical overburden.")

    print(f"\n{'log10(E_nu/GeV)':>16} " + " ".join(f"{n:>26}" for n in curves))
    for log10_e in (5.0, 6.0, 7.0, 8.0, 9.0, 10.0):
        i = int(np.argmin(np.abs(COMMON_LOG10_E - log10_e)))
        row = " ".join(f"{c[i]:26.3g}" for c in curves.values())
        flag = "  *" if COMMON_LOG10_E[i] > TABULATED_TOP_LOG10_E else ""
        print(f"{COMMON_LOG10_E[i]:16.1f} {row}{flag}")
    print("  Effective areas in cm^2. * beyond the top of the BGR18 cross-section table.")

    volumes = {name: effective_volume_km3(curve) for name, curve in curves.items()}
    print(f"\n{'log10(E_nu/GeV)':>16} " + " ".join(f"{n:>26}" for n in volumes))
    for log10_e in (5.0, 6.0, 7.0, 8.0, 9.0, 10.0):
        i = int(np.argmin(np.abs(COMMON_LOG10_E - log10_e)))
        row = " ".join(f"{v[i]:26.3g}" for v in volumes.values())
        print(f"{COMMON_LOG10_E[i]:16.1f} {row}")
    print("  Effective volumes in km^3 of sea water, A_eff / (n_N sigma_CC).")


def required_footprint_radius_km(
    ratio: np.ndarray,
    radius_km: float,
    n_blocks: int,
) -> np.ndarray:
    """Footprint radius that would scale the projected area by ``ratio`` [km].

    See :func:`softpaws.response.effective_area.required_footprint_radius_km`.
    """
    return engine.required_footprint_radius_km(
        ratio, radius_km, n_blocks, BLOCK_HEIGHT_KM, N_ZENITH
    )


def efficiency_summary(
    label: str,
    published: np.ndarray,
    model: np.ndarray,
    radius_km: float | None = None,
    n_blocks: int = 1,
) -> None:
    """Print the implied selection efficiency ``published / model``.

    Parameters
    ----------
    label : str
        Name of the comparison.
    published, model : np.ndarray
        Effective areas [cm^2] on ``COMMON_LOG10_E``.
    radius_km : float or None, optional
        If given, also report the footprint radius that would bring the model
        up to the published curve wherever the efficiency exceeds one
        (:func:`required_footprint_radius_km`).
    n_blocks : int, optional
        Number of building blocks, for that inversion.
    """
    ratio = published / model
    valid = np.isfinite(ratio)
    if not valid.any():
        print(f"  {label}: no overlap")
        return
    needed = (
        required_footprint_radius_km(ratio, radius_km, n_blocks)
        if radius_km is not None
        else np.full_like(ratio, np.nan)
    )
    print(f"\n  {label}: implied efficiency A_eff^pub / A_eff^model")
    for log10_e in (4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0):
        i = int(np.argmin(np.abs(COMMON_LOG10_E - log10_e)))
        if not valid[i]:
            continue
        extra = "" if np.isnan(needed[i]) else f"   R_eff = {needed[i] * 1e3:6.0f} m"
        print(f"    log10(E/GeV) = {COMMON_LOG10_E[i]:4.1f}   {ratio[i]:8.3f}{extra}")
    print(
        f"    range over the overlap: {np.nanmin(ratio[valid]):.3f} - "
        f"{np.nanmax(ratio[valid]):.3f}"
    )


def make_figure(
    curves: dict[str, np.ndarray],
    published: dict[str, np.ndarray],
    untruncated: np.ndarray,
    out_path: pathlib.Path,
) -> None:
    """Draw the three-panel comparison and write it to disk."""
    with plt.style.context(str(_STYLE)):
        fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.6))

        ax = axes[0]
        ax.plot(
            COMMON_LOG10_E, published["ARCA230, nu_mu trigger level"],
            color="k", lw=1.8, label="ARCA230 published, trigger",
        )
        ax.plot(
            COMMON_LOG10_E, published["ARCA230, quoted analytic fit"],
            color="0.55", lw=0.9, ls=":", label=r"quoted fit ($1.95\times$ high)",
        )
        ax.plot(COMMON_LOG10_E, curves["ARCA230 nu_mu"], color="C0", lw=1.1, label="model, nu_mu")
        ax.plot(
            COMMON_LOG10_E, curves["ARCA230 nu_mu + nu_tau"],
            color="C2", lw=1.1, label="model, + nu_tau -> mu",
        )
        ax.plot(
            COMMON_LOG10_E, untruncated,
            color="C0", lw=0.9, ls="--", label="nu_mu, Eq. (16) untruncated",
        )
        ax.plot(
            COMMON_LOG10_E, curves["ARCA230, fitted reach"],
            color="C4", lw=1.4, label="model, fitted reach law",
        )
        ax.set_title("(a) full ARCA, per-flavour", fontsize=7)
        ax.set_ylabel(r"$A_{\rm eff}$ [cm$^2$]")
        ax.legend(fontsize=5.5, loc="upper left")

        ax = axes[1]
        ax.plot(
            COMMON_LOG10_E, published["ARCA21, all-flavour (Nature 2025)"],
            color="k", lw=1.8, label="ARCA21 published",
        )
        ax.plot(
            COMMON_LOG10_E, curves["ARCA21 nu_mu + nu_tau"],
            color="C1", lw=1.1, label="model, static footprint",
        )
        ax.plot(
            COMMON_LOG10_E, curves["ARCA21, fitted reach"],
            color="C4", lw=1.4, label="model, reach law from ARCA230",
        )
        ax.set_title("(b) ARCA21, bright-track selection", fontsize=7)
        ax.set_ylabel(r"$A_{\rm eff}$ [cm$^2$]")
        ax.legend(fontsize=5.5, loc="upper left")

        ax = axes[2]
        ax.plot(
            COMMON_LOG10_E,
            published["ARCA230, nu_mu trigger level"] / curves["ARCA230 nu_mu + nu_tau"],
            color="C2", lw=1.1, label="ARCA230, trigger",
        )
        ax.plot(
            COMMON_LOG10_E,
            published["ARCA21, all-flavour (Nature 2025)"] / curves["ARCA21 nu_mu + nu_tau"],
            color="C1", lw=1.1, label="ARCA21, static",
        )
        ax.plot(
            COMMON_LOG10_E,
            published["ARCA21, all-flavour (Nature 2025)"] / curves["ARCA21, fitted reach"],
            color="C4", lw=1.4, label="ARCA21, reach law",
        )
        ax.plot(
            COMMON_LOG10_E,
            published["ARCA230, nu_mu trigger level"] / curves["ARCA230, fitted reach"],
            color="C4", lw=0.9, ls="--", label="ARCA230, reach law",
        )
        ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 5.0)
        ax.set_ylabel("published / model")
        ax.set_title("(c) implied selection efficiency", fontsize=7)
        ax.legend(fontsize=5.5, loc="upper left")

        for ax in axes[:2]:
            ax.set_yscale("log")
        for ax in axes:
            ax.set_xlim(COMMON_LOG10_E[0], COMMON_LOG10_E[-1])
            ax.set_xlabel(r"$\log_{10}(E_\nu\,/\,\mathrm{GeV})$")
            ax.axvspan(TABULATED_TOP_LOG10_E, COMMON_LOG10_E[-1], color="0.85", alpha=0.5, lw=0)

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png"):
            path = out_path.with_suffix(suffix)
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {path.resolve()}")
        plt.close(fig)


def main() -> None:
    global BLOCK_HEIGHT_KM
    args = parse_args()
    BLOCK_HEIGHT_KM = args.block_height_km
    radius_km = args.block_radius_km

    energy_mu = (1.0 - MEAN_INELASTICITY) * 10.0**COMMON_LOG10_E
    theta_deg, _ = zenith_grid()
    available = upstream_column_km(theta_deg, args.depth_km)
    lengths = {
        "free": stochastic_muon_range_km(
            energy_mu, args.threshold, kernel_evaluation=args.kernel_evaluation
        ),
        "vertical_down": truncated_range_km(
            energy_mu, np.array([args.depth_km]), args.threshold, args.kernel_evaluation
        )[:, 0],
    }
    print(
        f"  detector depth {args.depth_km:.2f} km, vertical overburden "
        f"{args.depth_km * RHO_SEA_G_CM3 * CM_PER_KM:.3g} g/cm^2, "
        f"horizontal cap {available.max():.0f} km"
    )

    print("Building ARCA230 effective areas ...")
    kwargs = dict(
        threshold_gev=args.threshold,
        depth_km=args.depth_km,
        kernel_evaluation=args.kernel_evaluation,
    )
    a230_mu = effective_area(radius_km, N_BLOCKS_FULL, flavour="mu", **kwargs)
    a230_tau = effective_area(radius_km, N_BLOCKS_FULL, flavour="tau", **kwargs)

    print("Building ARCA21 effective areas ...")
    a21_mu = effective_area(ARCA21_RADIUS_KM, 1, flavour="mu", **kwargs)
    a21_tau = effective_area(ARCA21_RADIUS_KM, 1, flavour="tau", **kwargs)

    curves = {
        "ARCA230 nu_mu": a230_mu,
        "ARCA230 nu_mu + nu_tau": a230_mu + a230_tau,
        "ARCA21 nu_mu + nu_tau": a21_mu + a21_tau,
    }
    publisheds = {
        "ARCA230, nu_mu trigger level": arca230_published_trigger(COMMON_LOG10_E),
        "ARCA230, quoted analytic fit": arca230_quoted_fit(COMMON_LOG10_E),
        "ARCA21, all-flavour (Nature 2025)": arca21_published(COMMON_LOG10_E),
    }
    ratio = publisheds["ARCA230, quoted analytic fit"] / publisheds[
        "ARCA230, nu_mu trigger level"
    ]
    print(
        f"\n  Quoted analytic fit / digitized trigger curve: "
        f"{np.nanmin(ratio):.2f}-{np.nanmax(ratio):.2f} over the overlap "
        f"(a flat factor, i.e. a spurious doubling in the quoted form)."
    )

    # --- Calibrate the reach law on ARCA230, then transfer it to ARCA21. ---
    print("\nFitting the reach law against the ARCA230 trigger curve ...")
    trigger = publisheds["ARCA230, nu_mu trigger level"]
    # Fit only where the digitization is trustworthy: the top of the published
    # figure saturates, and below 10^4 GeV the model is far from its own regime.
    band = (COMMON_LOG10_E >= 4.0) & (COMMON_LOG10_E <= FIT_TOP_LOG10_E)
    required = np.full(COMMON_LOG10_E.size, np.nan)
    required[band] = required_footprint_radius_km(
        (trigger / curves["ARCA230 nu_mu + nu_tau"])[band], radius_km, N_BLOCKS_FULL
    )
    reach_km, pivot_gev = fit_reach_law(COMMON_LOG10_E, required, radius_km)
    print(
        f"  reach   Lambda   = {reach_km * 1e3:6.1f} m per e-fold "
        f"({reach_km * 1e3 * np.log(10.0):.0f} m per decade)"
    )
    print(f"  pivot   E_piv    = 10^{np.log10(pivot_gev):.2f} GeV")
    print(f"  instrumented R   = {radius_km * 1e3:6.0f} m")

    law = dict(reach_km=reach_km, pivot_gev=pivot_gev)
    print("  forward-running ARCA230 with the fitted law ...")
    a230_law = effective_area(radius_km, N_BLOCKS_FULL, flavour="mu", **kwargs, **law)
    a230_law += effective_area(radius_km, N_BLOCKS_FULL, flavour="tau", **kwargs, **law)
    residual = np.log10(trigger[band] / a230_law[band])
    print(
        f"  residual against the trigger curve: {np.std(residual):.3f} dex rms, "
        f"{np.max(np.abs(residual)):.3f} dex max, over 1e4-1e{FIT_TOP_LOG10_E:g} GeV"
    )

    print("  transferring the same law to ARCA21 ...")
    a21_law = effective_area(ARCA21_RADIUS_KM, 1, flavour="mu", **kwargs, **law)
    a21_law += effective_area(ARCA21_RADIUS_KM, 1, flavour="tau", **kwargs, **law)
    curves["ARCA230, fitted reach"] = a230_law
    curves["ARCA21, fitted reach"] = a21_law
    publisheds["ARCA230, fitted-reach radius"] = required

    print("Breaking ARCA230 down by zenith band ...")
    bands = {
        "up, cos < -0.5": (-1.0, -0.5),
        "up, -0.5 to 0": (-0.5, 0.0),
        "down, 0 to 0.5": (0.0, 0.5),
        "down, cos > 0.5": (0.5, 1.0),
    }
    band_curves = {
        label: effective_area(radius_km, N_BLOCKS_FULL, flavour="mu",
                              cos_range=rng, **kwargs)
        for label, rng in bands.items()
    }
    # Eq. (16) as written, with no overburden cut, to size the truncation.
    band_curves["all sky, no truncation"] = effective_area(
        radius_km, N_BLOCKS_FULL, flavour="mu", truncate=False, **kwargs
    )
    print(f"\n{'log10(E_nu/GeV)':>16} " + " ".join(f"{n:>23}" for n in band_curves))
    for log10_e in (5.0, 6.0, 7.0, 8.0):
        i = int(np.argmin(np.abs(COMMON_LOG10_E - log10_e)))
        row = " ".join(f"{c[i]:23.3g}" for c in band_curves.values())
        print(f"{COMMON_LOG10_E[i]:16.1f} {row}")
    print("  ARCA230 nu_mu effective area [cm^2], averaged within each cos(theta) band.")
    print("  cos(theta) = +1 is vertically downgoing through the sea.")

    report(curves, lengths)
    efficiency_summary(
        "ARCA230 nu_mu, trigger level",
        publisheds["ARCA230, nu_mu trigger level"],
        curves["ARCA230 nu_mu + nu_tau"],
        radius_km=radius_km,
        n_blocks=N_BLOCKS_FULL,
    )
    efficiency_summary(
        "ARCA21 all-flavour, static footprint",
        publisheds["ARCA21, all-flavour (Nature 2025)"],
        curves["ARCA21 nu_mu + nu_tau"],
    )
    efficiency_summary(
        "ARCA21 all-flavour, reach law transferred from ARCA230",
        publisheds["ARCA21, all-flavour (Nature 2025)"],
        curves["ARCA21, fitted reach"],
    )
    make_figure(curves, publisheds, band_curves["all sky, no truncation"], args.out)


if __name__ == "__main__":
    main()
