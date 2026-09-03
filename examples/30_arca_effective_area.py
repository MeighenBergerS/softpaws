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
from scipy.optimize import brentq

from softpaws.detectors import ARCA21, ARCA230, MAX_UPSTREAM_KM
from softpaws.transport.attenuation import (
    flavour_transmission,
    prem_column,
    regenerated_transmission,
)
from softpaws.transport.cross_section import bgr18_cross_section
from softpaws.transport.soft_volume import (
    DEFAULT_MUON_THRESHOLD_GEV,
    light_reach_radius_km,
    stochastic_muon_range_km,
    truncated_muon_range_km,
)
from softpaws.transport.source import MEAN_INELASTICITY, nucleon_number_density
from softpaws.transport.tau import BR_TAU_TO_MU, MEAN_Z
from softpaws.utils.constants import CM_PER_KM, RHO_WATER_G_CM3

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_KM3NET_DIR = _HERE.parent / "src" / "softpaws" / "data" / "km3net"
_ARCA21_TABLE = _KM3NET_DIR / "arca21_aeff_brighttrack_allflavour_skyavg.csv"
_ARCA230_TRIGGER_TABLE = _KM3NET_DIR / "arca_trigger_level_eff.csv"
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
COMMON_LOG10_E = np.arange(4.0, 10.01, 0.2)

# Zenith sampling. theta = 0 is straight down through the sea, theta = 180 is
# straight up through the Earth.
N_ZENITH = 90

CROSS_SECTION = bgr18_cross_section()

# BGR18 stops just below 10^10 GeV; beyond that both the cross section and the
# PROPOSAL transport coefficients are extrapolations.
TABULATED_TOP_LOG10_E = 9.98

# Top of the band the reach law is fitted over. The digitized trigger curve
# saturates at the edge of the published figure in its last tenth of a decade,
# so the fit stops short of it.
FIT_TOP_LOG10_E = 7.5


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
    """Zenith samples and their solid-angle weights over a band of the sky.

    Parameters
    ----------
    cos_range : tuple of float, optional
        Band of ``cos(theta)`` to cover. Defaults to the full sky. The
        convention matches the published ARCA figures: ``cos(theta) = +1`` is
        vertically downgoing through the sea, ``-1`` vertically upgoing through
        the Earth.

    Returns
    -------
    theta_deg : np.ndarray, shape (N_ZENITH,)
        Zenith angle [deg].
    weights : np.ndarray, shape (N_ZENITH,)
        Solid-angle weights within the band, normalized to sum to one, so a
        weighted average over them is the average over that band.
    """
    cos_lo, cos_hi = sorted(cos_range)
    edges = np.linspace(cos_hi, cos_lo, N_ZENITH + 1)
    cos_theta = 0.5 * (edges[:-1] + edges[1:])
    theta_deg = np.rad2deg(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    weights = np.full(N_ZENITH, 1.0 / N_ZENITH)
    return theta_deg, weights


def projected_area_km2(
    theta_deg: np.ndarray,
    radius_km: float | np.ndarray,
    n_blocks: int,
) -> np.ndarray:
    """Projected area of upright cylinders seen from a given zenith angle.

    A cylinder of radius ``R`` and height ``h`` presents ``pi R^2`` overhead and
    ``2 R h`` at the horizon; the convex-body projection interpolates between
    them as ``pi R^2 |cos theta| + 2 R h sin theta``.

    Parameters
    ----------
    theta_deg : np.ndarray
        Zenith angle [deg].
    radius_km : float or np.ndarray
        Footprint radius of one building block [km]. Broadcast against
        ``theta_deg``, so an energy-dependent radius can be passed as a column.
    n_blocks : int
        Number of building blocks. Their projections are added, which ignores
        the mutual shadowing of two adjacent blocks near the horizon.

    Returns
    -------
    area : np.ndarray
        Projected area [km^2], broadcast to the shape of the two inputs.
    """
    theta = np.deg2rad(theta_deg)
    cap = np.pi * np.asarray(radius_km) ** 2 * np.abs(np.cos(theta))
    side = 2.0 * np.asarray(radius_km) * BLOCK_HEIGHT_KM * np.sin(theta)
    return n_blocks * (cap + side)


def fit_reach_law(
    log10_e: np.ndarray,
    required_radius_km: np.ndarray,
    radius_km: float,
) -> tuple[float, float]:
    """Least-squares reach law through the radii a published curve demands.

    :func:`required_footprint_radius_km` inverts a published effective area for
    the footprint that would reproduce it. Those radii are close to linear in
    ``ln E``, so one straight-line fit fixes both parameters of
    :func:`~softpaws.transport.soft_volume.light_reach_radius_km` without ever
    running the forward model, which is what keeps the fit cheap enough to be
    worth doing.

    Parameters
    ----------
    log10_e : np.ndarray
        ``log10(E_nu / GeV)`` of the points to fit.
    required_radius_km : np.ndarray
        Radius each point demands [km]; ``NaN`` entries are dropped.
    radius_km : float
        Instrumented footprint radius [km], used to locate the pivot.

    Returns
    -------
    reach_km : float
        Growth of the reach per e-fold of energy [km].
    pivot_gev : float
        Energy at which the effective radius equals the instrumented one [GeV].
    """
    valid = np.isfinite(required_radius_km)
    ln_e = np.log(10.0 ** np.asarray(log10_e)[valid])
    slope, intercept = np.polyfit(ln_e, np.asarray(required_radius_km)[valid], 1)
    return float(slope), float(np.exp((radius_km - intercept) / slope))


def upstream_column_km(theta_deg: np.ndarray, depth_km: float) -> np.ndarray:
    """Sea-water column available upstream of the detector [km of water].

    Downgoing muons are produced in the water between the sea surface and the
    detector, a path of ``depth / cos(theta)``. Upgoing muons come through rock,
    which supplies far more column than any muon survives, so the limit is
    effectively infinite; ``MAX_SEA_PATH_KM`` stands in for it.

    Parameters
    ----------
    theta_deg : np.ndarray
        Zenith angle [deg].
    depth_km : float
        Depth of the instrumented volume below the sea surface [km].

    Returns
    -------
    column_km : np.ndarray
        Available column, expressed as a length of water at ``RHO_SEA_G_CM3``.
    """
    cos_theta = np.cos(np.deg2rad(theta_deg))
    with np.errstate(divide="ignore", invalid="ignore"):
        downgoing = np.where(cos_theta > 0.0, depth_km / np.maximum(cos_theta, 1e-6), np.inf)
    return np.minimum(downgoing, MAX_SEA_PATH_KM)


def earth_column_g_cm2(theta_deg: np.ndarray, depth_km: float) -> np.ndarray:
    """Column depth traversed by the neutrino before reaching the detector.

    Upgoing directions get the layered-PREM Earth chord of
    :func:`~softpaws.transport.attenuation.prem_column`, evaluated at a
    declination equal to the angle below the horizon. Downgoing directions get
    the sea water above the detector, which is negligible except within a
    degree or so of the horizon, where the 1/cos path reaches ``10^7`` g cm^-2
    and starts to matter above 100 PeV.

    Parameters
    ----------
    theta_deg : np.ndarray
        Zenith angle [deg].
    depth_km : float
        Depth of the instrumented volume below the sea surface [km].

    Returns
    -------
    column : np.ndarray
        Column depth [g cm^-2].
    """
    water_km = upstream_column_km(theta_deg, depth_km)
    water = np.where(np.isfinite(water_km), water_km, 0.0) * CM_PER_KM * RHO_SEA_G_CM3
    earth = np.array(
        [prem_column(float(t) - 90.0) if t > 90.0 else 0.0 for t in theta_deg]
    )
    return np.where(theta_deg > 90.0, earth, water)


# ---------------------------------------------------------------------------
# Truncated first-passage range
# ---------------------------------------------------------------------------


def truncated_range_km(
    energy_mu_gev: np.ndarray,
    column_km: np.ndarray,
    threshold_gev: float,
    kernel_evaluation: str = "running",
) -> np.ndarray:
    """Truncated first-passage range ``E[tau ^ X]``, from the library.

    Eq. (16) of the draft is ``L = Integral_0^inf d_ell P[W(ell) < w_star]``.
    Cutting the integral at a finite ``X`` gives ``E[tau(w_star) ^ X]``, the
    length available when the muon cannot be born further upstream than ``X``.

    This used to build its own table, running the Gil-Pelaez inversion of
    :func:`~softpaws.transport.loss_distribution.log_loss_cdf` on a depth grid
    and interpolating. That route carried two approximations the library no
    longer makes: it read the loss moments off the *two-moment family*, which is
    8% low on ``Phi'(0)`` and 56% low on ``-Phi''(0)`` because ``-ln(1-y)``
    weights the hard end of the kernel a fit to the ``y``-moments does not
    constrain, and it froze them at the production energy. The library form is
    closed (a pair of incomplete gamma functions matched to the first two
    first-passage moments), so it needs no table at all.

    Parameters
    ----------
    energy_mu_gev : np.ndarray, shape (n,)
        Muon energy at production [GeV].
    column_km : np.ndarray, shape (m,)
        Available upstream column ``X`` [km of water]; ``inf`` is allowed and
        returns the untruncated length.
    threshold_gev : float
        Muon selection threshold [GeV].
    kernel_evaluation : {"running", "frozen"}, optional
        Passed through to
        :func:`~softpaws.transport.soft_volume.truncated_muon_range_km`.

    Returns
    -------
    length : np.ndarray, shape (n, m)
        Effective length [km].
    """
    energy = np.atleast_1d(np.asarray(energy_mu_gev, dtype=float))
    column = np.atleast_1d(np.asarray(column_km, dtype=float))
    return np.clip(
        truncated_muon_range_km(
            energy[:, None],
            column[None, :],
            threshold_gev,
            kernel_evaluation=kernel_evaluation,
        ),
        0.0,
        None,
    )


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

    Assembles the pieces of Sec. VI of the draft with the two ARCA-specific
    changes: the length is truncated at the available column, and the projected
    area follows the cylinder rather than a sphere. As in example 28, the
    neutral-current-degraded population is kept on an energy ladder and each
    rung is credited to the surface energy, matching how a published effective
    area is built.

    Parameters
    ----------
    radius_km : float
        Footprint radius of one building block [km].
    n_blocks : int
        Number of building blocks.
    threshold_gev : float
        Muon selection threshold [GeV].
    depth_km : float
        Depth of the instrumented volume below the sea surface [km].
    kernel_evaluation : {"running", "frozen"}
        Where along the descent the loss kernel is read; see
        :func:`truncated_range_km`.
    flavour : {"mu", "tau"}
        ``"mu"`` is the direct charged-current muon. ``"tau"`` is the
        ``nu_tau -> tau -> mu`` chain of Sec. IV, which costs the branching
        ratio ``B_{tau->mu}`` and a factor ``<z>`` in muon energy but gains the
        charged-current-regenerating Earth transmission.
    cos_range : tuple of float, optional
        Band of ``cos(theta)`` to average over. Defaults to the full sky.
    truncate : bool, optional
        Whether to cut the first-passage integral at the available column. Set
        to ``False`` to reproduce Eq. (16) as written, which is what isolates
        the size of the overburden effect.
    reach_km : float or None, optional
        Growth of the light reach per e-fold of energy [km]. ``None``, the
        default, uses the static instrumented footprint; a value activates
        :func:`~softpaws.transport.soft_volume.light_reach_radius_km` for both
        the projected area and the instrumented volume, since both describe
        what the detector responds to.
    pivot_gev : float, optional
        Energy at which the reach vanishes [GeV]. Ignored when ``reach_km`` is
        ``None``.

    Returns
    -------
    aeff : np.ndarray, shape (COMMON_LOG10_E.size,)
        Effective area [cm^2], averaged over the requested band.
    """
    theta_deg, weights = zenith_grid(cos_range)
    columns = earth_column_g_cm2(theta_deg, depth_km)
    available_km = upstream_column_km(theta_deg, depth_km)
    if not truncate:
        available_km = np.full_like(available_km, np.inf)
    static_area_km2 = projected_area_km2(theta_deg, radius_km, n_blocks)
    static_v_det_km3 = n_blocks * np.pi * radius_km**2 * BLOCK_HEIGHT_KM
    n_nucleon = nucleon_number_density(RHO_SEA_G_CM3)

    muon_fraction = (1.0 - MEAN_INELASTICITY)
    if flavour == "tau":
        muon_fraction *= MEAN_Z

    out = np.empty(COMMON_LOG10_E.size)
    for i, e_nu in enumerate(10.0**COMMON_LOG10_E):
        if flavour == "tau":
            rung_energy, rung_weight = flavour_transmission(
                float(e_nu), columns, CROSS_SECTION, flavour="tau"
            )
        else:
            rung_energy, rung_weight = regenerated_transmission(
                float(e_nu), columns, CROSS_SECTION
            )
        # (n_rung, n_theta) length: each rung's muon, each direction's column.
        length = truncated_range_km(
            rung_energy * muon_fraction, available_km, threshold_gev, kernel_evaluation
        )
        length[rung_energy * muon_fraction <= threshold_gev, :] = 0.0

        if reach_km is None:
            area_km2 = static_area_km2[None, :]
            v_det_km3 = static_v_det_km3
        else:
            # The light reach follows the muon, so each rung gets its own radius.
            r_eff = light_reach_radius_km(
                radius_km, rung_energy * muon_fraction, reach_km, pivot_gev
            )[:, None]
            area_km2 = projected_area_km2(theta_deg[None, :], r_eff, n_blocks)
            v_det_km3 = n_blocks * np.pi * r_eff**2 * BLOCK_HEIGHT_KM

        volume_km3 = area_km2 * length + v_det_km3
        rate = (
            n_nucleon
            * CROSS_SECTION.cc(rung_energy)[:, None]
            * volume_km3
            * CM_PER_KM**3
        )
        if flavour == "tau":
            rate = rate * BR_TAU_TO_MU
        out[i] = np.average((rung_weight * rate).sum(axis=0), weights=weights)
    return out


# ---------------------------------------------------------------------------
# Published references
# ---------------------------------------------------------------------------


def arca230_published_trigger(log10_e: np.ndarray) -> np.ndarray:
    """Digitized full-ARCA ``nu_mu`` effective area at trigger level [cm^2].

    Read from ``src/softpaws/data/km3net/arca_trigger_level_eff.csv``, digitized
    from KM3NeT Collaboration, Eur. Phys. J. C 84 (2024) 885 [arXiv:2402.08363]
    Fig. 7, for the two-building-block detector.

    Trigger level is the right thing to hold a geometric ceiling against. It
    asks only that the event produce enough coincident hits, with none of the
    quality and containment cuts an analysis adds on top, so it is the largest
    effective area the instrument ever reports and the hardest for a
    footprint-based bound to accommodate.

    Parameters
    ----------
    log10_e : np.ndarray
        ``log10(E_nu / GeV)``.

    Returns
    -------
    aeff : np.ndarray
        Effective area [cm^2], ``NaN`` outside the digitized range.
    """
    table = np.genfromtxt(_ARCA230_TRIGGER_TABLE, delimiter=",", comments="#")
    table = table[np.argsort(table[:, 0])]
    log10_table = np.log10(table[:, 0])
    return 1.0e4 * 10.0 ** np.interp(
        log10_e, log10_table, np.log10(table[:, 1]), left=np.nan, right=np.nan
    )


def arca230_quoted_fit(log10_e: np.ndarray) -> np.ndarray:
    """Analytic parametrization of the same curve quoted in the literature [cm^2].

    Several phenomenology papers quote

        A_eff = 2 [0.20 (E/E0)^-0.51 + 0.46 (E/E0)^-0.06]^-6.4 m^2,
        E0 = 10^4 GeV,

    over ``10^3``-``10^8`` GeV, attributed to the same figure. It runs a factor
    ``1.95`` above :func:`arca230_published_trigger` across five decades, which
    is flat enough to identify: the leading ``2`` doubles a curve that is
    already the two-block effective area. It is kept here only so that the
    discrepancy is visible rather than propagated.

    Parameters
    ----------
    log10_e : np.ndarray
        ``log10(E_nu / GeV)``.

    Returns
    -------
    aeff : np.ndarray
        Effective area [cm^2], masked to ``NaN`` outside the quoted validity.
    """
    x = 10.0 ** (log10_e - 4.0)
    aeff_m2 = 2.0 * (0.20 * x**-0.51 + 0.46 * x**-0.06) ** -6.4
    return np.where((log10_e >= 3.0) & (log10_e <= 8.0), aeff_m2 * 1.0e4, np.nan)


def arca21_published(log10_e: np.ndarray) -> np.ndarray:
    """Tabulated ARCA21 bright-track, all-flavour, sky-averaged area [cm^2].

    Parameters
    ----------
    log10_e : np.ndarray
        ``log10(E_nu / GeV)``.

    Returns
    -------
    aeff : np.ndarray
        Effective area [cm^2], ``NaN`` where the table is zero or absent.
    """
    table = np.genfromtxt(_ARCA21_TABLE, delimiter=",", comments="#")
    log10_table = np.log10(table[:, 0])
    positive = table[:, 1] > 0.0
    interpolated = 10.0 ** np.interp(
        log10_e,
        log10_table[positive],
        np.log10(table[positive, 1]),
        left=np.nan,
        right=np.nan,
    )
    return interpolated


# ---------------------------------------------------------------------------
# Reporting and plotting
# ---------------------------------------------------------------------------


def effective_volume_km3(aeff_cm2: np.ndarray) -> np.ndarray:
    """Effective target volume implied by an effective area [km^3].

    Inverting ``A_eff = n_N sigma_CC V_eff`` gives the transmission-weighted
    volume the detector behaves as, which is the quantity Eqs. (10) and (15) of
    the draft actually predict. Quoted at the sea-water nucleon density, so it
    is a genuine volume of water rather than a column.

    Parameters
    ----------
    aeff_cm2 : np.ndarray
        Effective area [cm^2] on ``COMMON_LOG10_E``.

    Returns
    -------
    volume : np.ndarray
        Effective volume [km^3].
    """
    energy = 10.0**COMMON_LOG10_E
    denominator = nucleon_number_density(RHO_SEA_G_CM3) * CROSS_SECTION.cc(energy)
    return aeff_cm2 / denominator / CM_PER_KM**3


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
    """Footprint radius that would scale the projected area by ``ratio``.

    Read only when the implied efficiency comes out above one, which a
    geometric ceiling forbids. The deficit then has to sit in the geometry, and
    the natural place is the projected area: a bright muon triggers from
    outside the instrumented footprint, which is the ``A_proj(E)`` of Sec. IV.C
    with a growth length the draft never fixes. This inverts the sky-averaged
    cylinder projection for the radius that would close the gap, holding the
    instrumented height fixed.

    Parameters
    ----------
    ratio : np.ndarray
        Required scaling of the sky-averaged projected area.
    radius_km : float
        Nominal footprint radius of one building block [km].
    n_blocks : int
        Number of building blocks.

    Returns
    -------
    radius : np.ndarray
        Required footprint radius [km], ``NaN`` where ``ratio`` is not finite.
    """
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
