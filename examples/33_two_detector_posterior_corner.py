"""Example 33 -- the same fit run against IceCube and against KM3NeT/ARCA230.

Example 29 floats the few physical handles of the first-principles effective
area against the published IceCube upgoing table and asks where the data puts
them. Example 30 ports the same construction to ARCA by supplying only the
*instrument* numbers -- a cylinder instead of a sphere, a finite sea-water
overburden, a 4 pi sky average. This example runs example 29's fit twice, once
per site, and puts the two posteriors on one set of axes.

The point is which parameters are allowed to be compared.

**Shared physics.** ``b_scale`` rescales the drift coefficient and with it
``Phi'(0)``, and ``lambda`` is the effective slope of the charged-current cross
section. Neither is a property of a site: the loss kernel and the Earth are the
same under the Mediterranean as under the Pole. Two posteriors that land on top
of each other here is the claim the figure is making; two that do not would say
the transport is being tuned per detector.

**Instrument response.** ``eps_0`` is a selection efficiency and ``Lambda`` is a
light reach in a different medium around a differently shaped array, so nothing
requires them to agree and a shared axis would only invite the wrong reading.
They are reported as marginals beside the corner instead of inside it.

``log10(E_thr/GeV)`` sits between the two. It is an instrument number, but it is
on a common scale and its ordering is a check rather than a coincidence claim:
ARCA230 is compared at *trigger* level, so its threshold should come out below
IceCube's analysis-level one.

**The two curves are not at the same selection level, and that is deliberate.**
IceCube is the livetime-weighted upgoing ``nu_mu`` table of IceTracks-DR2, an
analysis-level response. ARCA230 is the ``nu_mu`` effective area at trigger
level digitized from KM3NeT Collaboration, Eur. Phys. J. C 84 (2024) 885
[arXiv:2402.08363] Fig. 7. Trigger level is the strongest test of a geometric
ceiling -- it carries no quality or containment cuts, so it is the largest area
the instrument ever reports -- which is what makes it the right curve for
``b_scale`` and ``lambda``. It is the wrong curve for reading ``eps_0`` as a
physical efficiency against IceCube's, and the figure labels it as such.

The two fits are run **independently**. Tying ``b_scale`` and ``lambda`` across
the two would shrink both contours, but their agreement would then be imposed
rather than demonstrated, and demonstrating it is the whole point.

**The truncated range.** ARCA's largest site-specific effect is that a downgoing
muon cannot be born further upstream than the sea surface, so Eq. (16)'s
first-passage integral is cut at the available column: ``L = E[tau(w) ^ X]``.
Example 30 evaluates that by Gil-Pelaez inversion of the log-loss CDF, which
costs seconds per energy and cannot go inside a sampler. Here the first-passage
depth is matched to a gamma law on its first two renewal moments -- mean
``w / Phi'(0) - Phi''(0) / (2 Phi'(0)^2)`` and variance
``-w Phi''(0) / Phi'(0)^3`` -- whose limited expected value is an incomplete
gamma function (:func:`truncated_range_km`). It reproduces the exact integral to
better than 0.2% from ``10^5`` to ``10^8`` GeV at every truncation depth;
``--check-truncation`` runs that comparison.

The likelihood, the 15% assumed fractional error, and the flat-prior treatment
are example 29's, unchanged; posterior *widths* are set by that assumption and
are not measurements, while the posterior *locations* are what the figure is
about.

Outputs, besides the figure, are written next to it and are what the paper
reads: ``33_overlap_region.json`` carries the per-parameter intervals, the
intersection of the two 68% regions, the product posterior on the shared pair
and the 2-dof compatibility test, and ``33_chains.npz`` carries both chains so
the figure can be redrawn without resampling.

Usage
-----
    python examples/33_two_detector_posterior_corner.py
    python examples/33_two_detector_posterior_corner.py --steps 6000 --sigma 0.10
    python examples/33_two_detector_posterior_corner.py --check-truncation
"""

import argparse
import json
import pathlib
from dataclasses import dataclass, field
from typing import Callable

import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammainc, polygamma
from scipy.stats import chi2, gaussian_kde

from softpaws.comparison.likelihood import B_SCALE_MEAN, B_SCALE_STD
from softpaws.data.loader import compute_livetime_s, load_uptime, parse_aeff
from softpaws.data.schema import SEASONS
from softpaws.transport.attenuation import flavour_transmission, prem_column
from softpaws.transport.coefficients import diffusion_coefficient, drift_coefficient
from softpaws.transport.cross_section import bgr18_cross_section
from softpaws.transport.eigenvalue import two_moment_loss_spectrum
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
_ARCA230_TRIGGER_TABLE = (
    _HERE.parent / "src" / "softpaws" / "data" / "km3net" / "arca_trigger_level_eff.csv"
)
_DEFAULT_OUT_DIR = _HERE / "output"

CROSS_SECTION = bgr18_cross_section()

# ---------------------------------------------------------------------------
# Parameters. The same five for both detectors, split by what may be compared.
# ---------------------------------------------------------------------------

PARAM_NAMES = ("eps_0", "log10_e_thr", "b_scale", "lam", "reach_km")

# Drawn as the corner. b_scale and lam are shared physics; log10_e_thr is an
# instrument number on a common scale, kept here because its ordering between
# trigger and analysis level is a check worth seeing.
CORNER_PARAMS = ("log10_e_thr", "b_scale", "lam")
# Drawn as marginals beside the corner: different medium, different array.
INSTRUMENT_PARAMS = ("eps_0", "reach_km")
# The subspace where agreement is a physics statement, and where the product
# posterior and the compatibility test are evaluated.
PHYSICS_PARAMS = ("b_scale", "lam")

LABELS = {
    "eps_0": r"$\varepsilon_0$",
    "log10_e_thr": r"$\log_{10}(E_{\rm thr}/{\rm GeV})$",
    "b_scale": r"$b_\mu$ scale",
    "lam": r"$\lambda$",
    "reach_km": r"$\Lambda$ [m per e-fold]",
}

# Effective log-log slope of the BGR18 CC cross section over the fitted band,
# and the pivot the tilt rotates about. lam = LAMBDA_BGR18 recovers the table.
LAMBDA_BGR18 = 0.4538
LAMBDA_PIVOT_GEV = 1.0e6

# The reach law's pivot, held at the cross-section tilt's pivot so that eps_0
# carries the normalization and Lambda carries only the shape. Same value for
# both detectors, without which the two Lambda marginals would not be on the
# same footing even as marginals.
REACH_PIVOT_GEV = 1.0e6

# Fixed by oscillations over astrophysical baselines, so not a fit parameter.
F_TAU = 1.0

# Lower edge of the b_scale prior, and a physical boundary rather than a
# convenience. The two-moment loss spectrum dGamma/dy = kappa (1-y)^p / y is
# calibrated by p = b_mu / d_mu - 2, so it exists only for d_mu < b_mu. Rescaling
# b_mu without rescaling d_mu therefore has a floor: below b_scale = max(d_mu /
# b_mu) over the energies in play, p drops past -1, kappa turns negative and
# there is no loss spectrum to propagate. Evaluated over 10^2 to 10^10 GeV, that
# floor is 0.287, which sits 4.4 sigma below the b_scale prior mean -- far
# outside where either posterior lives, but reachable by a walker proposal,
# where it would otherwise surface as a NaN rather than as a rejection.
B_SCALE_FLOOR = float(
    np.max(
        diffusion_coefficient(np.logspace(2.0, 10.0, 200))
        / drift_coefficient(np.logspace(2.0, 10.0, 200))
    )
)

# Example 28 inverts the published-to-model ratio for the radius each energy
# demands and fits a straight line through it in ln E. That is the independent
# expectation for IceCube's Lambda, in the same sense that LAMBDA_BGR18 is the
# independent expectation for lam. ARCA has no counterpart yet.
REACH_EXAMPLE28_KM = 0.0143
# The DR2 smearing matrix's own handle on IceCube's threshold: the 5th
# percentile of accepted reconstructed muon energy, flat at ~700 GeV across
# three decades of E_nu. ARCA publishes no equivalent.
SMEARING_LOG10_E_THR = 2.85

# Where each parameter was derived to sit, independently of these fits. NaN
# marks a parameter with no first-principles value for that detector.
EXPECTED = {
    "IceCube": {
        "log10_e_thr": np.log10(DEFAULT_MUON_THRESHOLD_GEV),
        "b_scale": 1.0,
        "lam": LAMBDA_BGR18,
        "reach_km": REACH_EXAMPLE28_KM,
    },
    "ARCA230": {
        "b_scale": 1.0,
        "lam": LAMBDA_BGR18,
    },
}

# Flat within these ranges. The two shared-physics parameters carry identical
# priors by construction -- a per-detector prior on b_scale or lam would put the
# difference between the posteriors partly in the prior. Only the instrument
# parameters differ, and only where the instrument demands it.
PRIORS = {
    "IceCube": {
        # eps_0's upper edge is physical, not a convenience: the model is a
        # geometric ceiling, so a selection cannot exceed it.
        "eps_0": (0.0, 1.0),
        "log10_e_thr": (2.0, 5.0),
        "b_scale": (B_SCALE_FLOOR, 3.0),
        "lam": (0.0, 1.2),
        # Zero sits inside the range, so the data can say no reach is needed;
        # the negative side is kept open as a null check rather than as physics.
        "reach_km": (-0.02, 0.10),
    },
    "ARCA230": {
        "eps_0": (0.0, 1.0),
        "log10_e_thr": (2.0, 5.0),
        "b_scale": (B_SCALE_FLOOR, 3.0),
        "lam": (0.0, 1.2),
        # Wider than IceCube's: ARCA's blocks are 517 m across against a
        # kilometre-scale sphere, so the reach the trigger-level curve demands
        # is a larger fraction of the footprint. Example 30's inversion of the
        # same curve lands inside this range.
        "reach_km": (-0.05, 0.40),
    },
}

# ---------------------------------------------------------------------------
# IceCube. Example 29's model, unchanged.
# ---------------------------------------------------------------------------

IC_LOG10_E = np.linspace(3.0, 8.0, 26)
# The top of the DR2 simulation (100 PeV) is excluded, as in examples 28 and 29.
IC_FIT_BAND = (5.0, 7.8)
IC_RADIUS_KM = sphere_radius_from_volume(1.0)
IC_N_DEC = 40
IC_N_RUNG = 80

# ---------------------------------------------------------------------------
# ARCA230. Example 30's instrument numbers, with its grids thinned for sampling.
# ---------------------------------------------------------------------------

# Building-block footprint radius [km] and instrumented height [km], measured
# from the as-built ARCA21 geometry and scaled to a full block; see example 30.
ARCA_BLOCK_RADIUS_KM = 0.517
ARCA_BLOCK_HEIGHT_KM = 0.632
ARCA_N_BLOCKS = 2
# Depth of the instrumented volume's centre below the sea surface [km]. Seabed
# at 3500 m at the Capo Passero site, with the instrumented span standing on it.
ARCA_DEPTH_KM = 3.5 - 0.5 * ARCA_BLOCK_HEIGHT_KM
# Longest sea-water path a near-horizontal muon can have [km]. Only a cap on the
# 1/cos(theta) divergence; it exceeds every muon range in the problem.
ARCA_MAX_SEA_PATH_KM = 100.0

ARCA_LOG10_E = np.arange(4.0, 8.01, 0.2)
# The digitized trigger curve saturates at the edge of the published figure in
# its last half decade, so the fit stops short of it. The lower edge matches
# IceCube's, so the two bands start together.
ARCA_FIT_BAND = (5.0, 7.5)
# Thinned from example 30's 90 zenith bands and the library's 100 rungs: the
# projected area and the column both vary smoothly with zenith, and the ladder
# weight falls off long before the last rung. Against a 48 x 80 grid at the
# first-principles point these move the fitted band by 4e-4 dex, three orders of
# magnitude below the 0.065 dex the assumed uncertainty puts on each node.
ARCA_N_ZENITH = 20
ARCA_N_RUNG = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=pathlib.Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=_DEFAULT_OUT_DIR / "33_two_detector_posterior_corner.pdf",
    )
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--walkers", type=int, default=48)
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.15,
        help="Assumed fractional uncertainty on each published effective area.",
    )
    parser.add_argument(
        "--check-truncation",
        action="store_true",
        help="Compare the closed-form truncated range against the Gil-Pelaez "
        "integral of example 30 and exit. Slow: seconds per energy.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Truncated first-passage range
# ---------------------------------------------------------------------------


def truncated_range_km(
    energy_mu_gev: np.ndarray,
    column_km: np.ndarray,
    threshold_gev: float,
    b_scale: float = 1.0,
    density_g_cm3: float = RHO_WATER_G_CM3,
) -> np.ndarray:
    """Expected first-passage range cut at a finite available column [km].

    :func:`~softpaws.transport.soft_volume.stochastic_muon_range_km` evaluates
    ``E[tau(w)]``, the expected depth at which a muon born at ``epsilon`` first
    falls below threshold, integrating Eq. (16) to infinity. ARCA sits under
    only ~3.2 km of sea water, so a downgoing muon cannot be born further
    upstream than the surface and the honest length is the *limited* expectation

    .. math:: L(\\varepsilon, X) = \\mathbb{E}[\\tau(w_\\star) \\wedge X]
        = \\int_0^X d\\ell\\;\\mathbb{P}[W(\\ell) < w_\\star].

    Evaluating that integral directly needs the log-loss CDF at every depth,
    which is one Gil-Pelaez inversion per energy and far too slow to sit inside
    a sampler. Renewal theory gives the first two moments of ``tau`` in closed
    form from the same two-moment loss spectrum the untruncated range uses,

    .. math:: \\mathbb{E}[\\tau] = \\frac{w}{\\Phi'(0)}
        - \\frac{\\Phi''(0)}{2\\Phi'(0)^2}, \\qquad
        \\mathrm{Var}[\\tau] = -\\frac{w\\,\\Phi''(0)}{\\Phi'(0)^3},

    and matching a gamma law to them turns the limited expectation into

    .. math:: \\mathbb{E}[\\tau \\wedge X] = \\mathbb{E}[\\tau]\\,P(k+1, X/\\theta)
        + X\\,[1 - P(k, X/\\theta)],

    with ``P`` the regularized lower incomplete gamma function. The gamma law is
    a *shape* assumption on ``tau``, not on the loss spectrum: it is exact in
    both limits that matter (``X -> inf`` returns the untruncated mean by
    construction, ``X -> 0`` returns ``X``), and ``--check-truncation`` shows it
    holds to better than 0.2% in between.

    Parameters
    ----------
    energy_mu_gev : np.ndarray
        Muon energy at production [GeV].
    column_km : np.ndarray
        Available upstream column ``X``, as a length of water at
        ``density_g_cm3`` [km]. Broadcast against ``energy_mu_gev``; ``inf`` is
        allowed and returns the untruncated range.
    threshold_gev : float
        Muon energy below which the track is not selected [GeV].
    b_scale : float, optional
        Multiplicative rescaling of the drift coefficient. Defaults to 1.
    density_g_cm3 : float, optional
        Target-medium density [g cm^-3]. Defaults to water.

    Returns
    -------
    length : np.ndarray
        Expected truncated range [km], zero for muons born below threshold.
    """
    energy = np.asarray(energy_mu_gev, dtype=float)
    b_mu = b_scale * drift_coefficient(energy, density_g_cm3)
    d_mu = diffusion_coefficient(energy, density_g_cm3)
    kappa, p = two_moment_loss_spectrum(b_mu, d_mu)
    # Phi'(0) = <-ln(1-y)> and -Phi''(0) = <ln^2(1-y)>, both per unit length.
    first = kappa * polygamma(1, p + 1.0)
    second = -kappa * polygamma(2, p + 1.0)

    # Muons born below threshold get w = 0, which keeps the mean and variance
    # finite and non-negative; they are zeroed out at the end.
    selectable = energy > threshold_gev
    w = np.where(selectable, np.log(np.maximum(energy, threshold_gev) / threshold_gev), 0.0)
    mean = w / first + second / (2.0 * first**2)
    variance = w * second / first**3

    column = np.asarray(column_km, dtype=float)
    # An infinite column is the untruncated case; feeding it to the general
    # expression below would evaluate ``inf * 0``. The upgoing sky is the reason
    # it appears: rock supplies more column than any muon survives.
    capped = np.where(np.isfinite(column), column, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma_shape = mean**2 / variance
        x = capped / (variance / mean)
        limited = mean * gammainc(gamma_shape + 1.0, x) + capped * (
            1.0 - gammainc(gamma_shape, x)
        )
    limited = np.where(np.isfinite(column), limited, mean)
    return np.where(selectable & (mean > 0.0), np.clip(limited, 0.0, None), 0.0)


def check_truncation() -> None:
    """Compare :func:`truncated_range_km` against the Gil-Pelaez integral."""
    from scipy.integrate import cumulative_trapezoid

    from softpaws.transport.loss_distribution import log_loss_cdf

    threshold = DEFAULT_MUON_THRESHOLD_GEV
    print(f"\n{'log10(E_mu)':>12} {'X / L':>7} {'exact':>9} {'closed':>9} {'ratio':>7}")
    for log10_e in (5.0, 6.0, 7.0, 8.0):
        energy = np.array([10.0**log10_e])
        b_mu = float(drift_coefficient(energy)[0])
        d_mu = float(diffusion_coefficient(energy)[0])
        deterministic = float(muon_range_km(energy, threshold)[0])
        ell = np.linspace(0.0, 3.0 * deterministic, 601)
        cdf = log_loss_cdf(np.log(energy[0] / threshold), ell, b_mu, d_mu)
        cumulative = cumulative_trapezoid(cdf, ell, initial=0.0)
        untruncated = float(cumulative[-1])
        for fraction in (0.25, 0.5, 1.0, 2.0):
            column = fraction * untruncated
            exact = float(np.interp(column, ell, cumulative))
            closed = float(truncated_range_km(energy, np.array([column]), threshold)[0])
            print(
                f"{log10_e:12.1f} {fraction:7.2f} {exact:9.3f} {closed:9.3f} "
                f"{closed / exact:7.4f}"
            )
    print("  exact: Eq. (16) cut at X by Gil-Pelaez inversion, as in example 30.")


# ---------------------------------------------------------------------------
# Published references
# ---------------------------------------------------------------------------


def _canonical_irf_season(season: str) -> str:
    return "IC86" if season.startswith("IC86") else season


def icecube_upgoing(data_dir: pathlib.Path) -> np.ndarray:
    """Livetime-weighted IceCube effective area over the upgoing sky.

    Parameters
    ----------
    data_dir : pathlib.Path
        Root of the DR2 data directory.

    Returns
    -------
    aeff : np.ndarray, shape (IC_LOG10_E.size,)
        Effective area [cm^2].
    """
    total = np.zeros_like(IC_LOG10_E)
    total_livetime_s = 0.0
    cache: dict[str, object] = {}
    for season in SEASONS:
        key = _canonical_irf_season(season)
        if key not in cache:
            raw = np.genfromtxt(data_dir / "irfs" / f"{key}_effectiveArea.csv", comments="#")
            cache[key] = parse_aeff(raw)
        aeff = cache[key]
        livetime_s = compute_livetime_s(load_uptime(data_dir / "uptime" / f"{season}_exp.csv"))
        upgoing = aeff.sin_dec_centers > 0.0
        curve = np.average(
            aeff.values[:, upgoing], axis=1, weights=np.diff(aeff.sin_dec_edges)[upgoing]
        )
        total += livetime_s * np.interp(IC_LOG10_E, aeff.log10_energy_centers, curve)
        total_livetime_s += livetime_s
    return total / total_livetime_s


def arca230_trigger() -> np.ndarray:
    """Digitized full-ARCA ``nu_mu`` effective area at trigger level [cm^2].

    Read from ``src/softpaws/data/km3net/arca_trigger_level_eff.csv``, digitized
    from KM3NeT Collaboration, Eur. Phys. J. C 84 (2024) 885 [arXiv:2402.08363]
    Fig. 7, for the two-building-block detector.

    Returns
    -------
    aeff : np.ndarray, shape (ARCA_LOG10_E.size,)
        Effective area [cm^2], ``NaN`` outside the digitized range.
    """
    table = np.genfromtxt(_ARCA230_TRIGGER_TABLE, delimiter=",", comments="#")
    table = table[np.argsort(table[:, 0])]
    return 1.0e4 * 10.0 ** np.interp(
        ARCA_LOG10_E,
        np.log10(table[:, 0]),
        np.log10(table[:, 1]),
        left=np.nan,
        right=np.nan,
    )


# ---------------------------------------------------------------------------
# Geometry and transmission, precomputed once per detector
# ---------------------------------------------------------------------------


def icecube_ladders() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Hemisphere-averaged transmission ladders for IceCube, one per flavour.

    Neither ladder depends on a fitted parameter -- they are set by the cross
    section and the PREM column alone -- so they are built once and reused for
    every posterior sample.

    Returns
    -------
    ladders : dict
        ``"mu"`` and ``"tau"`` -> ``(energies, weights)``, both of shape
        ``(IC_LOG10_E.size, IC_N_RUNG)``, with ``energies`` in GeV and
        ``weights`` the hemisphere-averaged arrival probabilities.
    """
    dec_deg = np.linspace(0.5, 89.5, IC_N_DEC)
    columns = np.array([prem_column(float(d)) for d in dec_deg])
    solid_angle = np.cos(np.deg2rad(dec_deg))

    ladders: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for flavour in ("mu", "tau"):
        energies = np.empty((IC_LOG10_E.size, IC_N_RUNG))
        weights = np.empty((IC_LOG10_E.size, IC_N_RUNG))
        for i, log10_e in enumerate(IC_LOG10_E):
            rung_energy, rung_weight = flavour_transmission(
                10.0**log10_e, columns, CROSS_SECTION, flavour=flavour,
                n_grid=IC_N_RUNG, decades=4.0,
            )
            energies[i] = rung_energy
            weights[i] = np.average(rung_weight, axis=1, weights=solid_angle)
        ladders[flavour] = (energies, weights)
    return ladders


def arca_zenith_grid() -> tuple[np.ndarray, np.ndarray]:
    """Zenith samples and their solid-angle weights over the whole sky.

    The convention matches the published ARCA figures: ``cos(theta) = +1`` is
    vertically downgoing through the sea, ``-1`` vertically upgoing through the
    Earth.

    Returns
    -------
    theta_deg : np.ndarray, shape (ARCA_N_ZENITH,)
        Zenith angle [deg].
    weights : np.ndarray, shape (ARCA_N_ZENITH,)
        Solid-angle weights, normalized to sum to one.
    """
    edges = np.linspace(1.0, -1.0, ARCA_N_ZENITH + 1)
    cos_theta = 0.5 * (edges[:-1] + edges[1:])
    theta_deg = np.rad2deg(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    return theta_deg, np.full(ARCA_N_ZENITH, 1.0 / ARCA_N_ZENITH)


def arca_projected_area_km2(
    theta_deg: np.ndarray,
    radius_km: float | np.ndarray,
) -> np.ndarray:
    """Projected area of the upright ARCA cylinders at a given zenith angle.

    A cylinder of radius ``R`` and height ``h`` presents ``pi R^2`` overhead and
    ``2 R h`` at the horizon; the convex-body projection interpolates between
    them as ``pi R^2 |cos theta| + 2 R h sin theta``.

    Parameters
    ----------
    theta_deg : np.ndarray
        Zenith angle [deg].
    radius_km : float or np.ndarray
        Footprint radius of one building block [km]. Broadcast against
        ``theta_deg``, so an energy-dependent radius can be passed.

    Returns
    -------
    area : np.ndarray
        Projected area [km^2] of all ``ARCA_N_BLOCKS`` blocks.
    """
    theta = np.deg2rad(theta_deg)
    cap = np.pi * np.asarray(radius_km) ** 2 * np.abs(np.cos(theta))
    side = 2.0 * np.asarray(radius_km) * ARCA_BLOCK_HEIGHT_KM * np.sin(theta)
    return ARCA_N_BLOCKS * (cap + side)


def arca_columns() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zenith weights, neutrino column, and available muon column for ARCA.

    Upgoing directions get the layered-PREM Earth chord, evaluated at a
    declination equal to the angle below the horizon, and effectively unlimited
    upstream column for the muon -- rock supplies far more than any muon
    survives. Downgoing directions get the sea water above the detector, which
    is negligible for the neutrino except within a degree of the horizon, and
    which is the whole of the muon's available column.

    Returns
    -------
    weights : np.ndarray, shape (ARCA_N_ZENITH,)
        Solid-angle weights over the whole sky.
    neutrino_column : np.ndarray, shape (ARCA_N_ZENITH,)
        Column traversed before reaching the detector [g cm^-2].
    muon_column_km : np.ndarray, shape (ARCA_N_ZENITH,)
        Column available upstream of the detector, as a length of sea water
        [km].
    """
    theta_deg, weights = arca_zenith_grid()
    cos_theta = np.cos(np.deg2rad(theta_deg))
    with np.errstate(divide="ignore", invalid="ignore"):
        downgoing_km = np.where(
            cos_theta > 0.0, ARCA_DEPTH_KM / np.maximum(cos_theta, 1e-6), np.inf
        )
    muon_column_km = np.minimum(downgoing_km, ARCA_MAX_SEA_PATH_KM)

    water = np.where(np.isfinite(downgoing_km), np.minimum(downgoing_km, ARCA_MAX_SEA_PATH_KM), 0.0)
    water = water * CM_PER_KM * RHO_WATER_G_CM3
    earth = np.array([prem_column(float(t) - 90.0) if t > 90.0 else 0.0 for t in theta_deg])
    neutrino_column = np.where(theta_deg > 90.0, earth, water)
    return weights, neutrino_column, muon_column_km


def arca_ladders(
    neutrino_column: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-zenith transmission ladders for ARCA, one per flavour.

    Unlike IceCube's, these are *not* averaged over direction: the muon's
    available column varies with zenith too, so the average has to wait until
    the truncated range has been applied.

    Parameters
    ----------
    neutrino_column : np.ndarray, shape (ARCA_N_ZENITH,)
        Column traversed before reaching the detector [g cm^-2].

    Returns
    -------
    ladders : dict
        ``"mu"`` and ``"tau"`` -> ``(energies, weights)`` with ``energies`` of
        shape ``(n_energy, ARCA_N_RUNG)`` [GeV] and ``weights`` of shape
        ``(n_energy, ARCA_N_RUNG, ARCA_N_ZENITH)``.
    """
    ladders: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for flavour in ("mu", "tau"):
        energies = np.empty((ARCA_LOG10_E.size, ARCA_N_RUNG))
        weights = np.empty((ARCA_LOG10_E.size, ARCA_N_RUNG, ARCA_N_ZENITH))
        for i, log10_e in enumerate(ARCA_LOG10_E):
            rung_energy, rung_weight = flavour_transmission(
                10.0**log10_e, neutrino_column, CROSS_SECTION, flavour=flavour,
                n_grid=ARCA_N_RUNG, decades=4.0,
            )
            energies[i] = rung_energy
            weights[i] = rung_weight
        ladders[flavour] = (energies, weights)
    return ladders


# ---------------------------------------------------------------------------
# Forward models
# ---------------------------------------------------------------------------


def _tilted_cc(energy_gev: np.ndarray, lam: float) -> np.ndarray:
    """BGR18 charged-current cross section tilted about the pivot [cm^2].

    ``lam = LAMBDA_BGR18`` recovers the tabulated values exactly.
    """
    return CROSS_SECTION.cc(energy_gev) * (energy_gev / LAMBDA_PIVOT_GEV) ** (
        lam - LAMBDA_BGR18
    )


def icecube_model(
    theta: np.ndarray,
    ladders: dict[str, tuple[np.ndarray, np.ndarray]],
    select: np.ndarray | None = None,
) -> np.ndarray:
    """Predicted IceCube effective area for one parameter vector [cm^2].

    Parameters
    ----------
    theta : np.ndarray, shape (5,)
        ``(eps_0, log10_e_thr, b_scale, lam, reach_km)``.
    ladders : dict
        Output of :func:`icecube_ladders`.
    select : np.ndarray or None, optional
        Boolean mask over the energy grid. ``None``, the default, evaluates the
        whole grid; the sampler passes the fitted band, since the nodes outside
        it enter no likelihood and cost the same as the ones that do.

    Returns
    -------
    aeff : np.ndarray
        Effective area [cm^2], on the whole grid or on the selected nodes.
    """
    eps_0, log10_e_thr, b_scale, lam, reach_km = theta
    threshold = 10.0**log10_e_thr
    n_nucleon = nucleon_number_density()
    nodes = slice(None) if select is None else select

    total = np.zeros(IC_LOG10_E.size if select is None else int(np.sum(select)))
    channels = (
        ("mu", 1.0 - MEAN_INELASTICITY, 1.0),
        ("tau", MEAN_Z * (1.0 - MEAN_INELASTICITY), F_TAU * BR_TAU_TO_MU),
    )
    for flavour, muon_fraction, weight in channels:
        energies, arrival = ladders[flavour]
        energies, arrival = energies[nodes], arrival[nodes]
        muon_energy = muon_fraction * energies
        length = stochastic_muon_range_km(
            muon_energy.ravel(), threshold, b_scale=b_scale
        ).reshape(muon_energy.shape)
        radius = light_reach_radius_km(IC_RADIUS_KM, muon_energy, reach_km, REACH_PIVOT_GEV)
        volume_cm3 = (
            np.pi * radius**2 * length + 4.0 / 3.0 * np.pi * radius**3
        ) * CM_PER_KM**3
        sigma = _tilted_cc(energies, lam)
        total += weight * (arrival * n_nucleon * sigma * volume_cm3).sum(axis=1)
    return eps_0 * total


def arca_model(
    theta: np.ndarray,
    ladders: dict[str, tuple[np.ndarray, np.ndarray]],
    zenith_weights: np.ndarray,
    muon_column_km: np.ndarray,
    select: np.ndarray | None = None,
) -> np.ndarray:
    """Predicted ARCA230 effective area for one parameter vector [cm^2].

    The same assembly as :func:`icecube_model` with the two site-specific
    changes of example 30: the first-passage length is truncated at the column
    available upstream of the detector, and the projected area follows the
    cylinder rather than a sphere.

    Parameters
    ----------
    theta : np.ndarray, shape (5,)
        ``(eps_0, log10_e_thr, b_scale, lam, reach_km)``.
    ladders : dict
        Output of :func:`arca_ladders`.
    zenith_weights : np.ndarray, shape (ARCA_N_ZENITH,)
        Solid-angle weights over the whole sky.
    muon_column_km : np.ndarray, shape (ARCA_N_ZENITH,)
        Column available upstream of the detector [km of sea water].
    select : np.ndarray or None, optional
        Boolean mask over the energy grid; see :func:`icecube_model`.

    Returns
    -------
    aeff : np.ndarray
        Sky-averaged effective area [cm^2], on the whole grid or on the
        selected nodes.
    """
    eps_0, log10_e_thr, b_scale, lam, reach_km = theta
    threshold = 10.0**log10_e_thr
    n_nucleon = nucleon_number_density(RHO_WATER_G_CM3)
    theta_deg, _ = arca_zenith_grid()
    nodes = slice(None) if select is None else select

    total = np.zeros(ARCA_LOG10_E.size if select is None else int(np.sum(select)))
    channels = (
        ("mu", 1.0 - MEAN_INELASTICITY, 1.0),
        ("tau", MEAN_Z * (1.0 - MEAN_INELASTICITY), F_TAU * BR_TAU_TO_MU),
    )
    for flavour, muon_fraction, weight in channels:
        energies, arrival = ladders[flavour]
        energies, arrival = energies[nodes], arrival[nodes]
        muon_energy = muon_fraction * energies
        # (n_energy, n_rung, n_zenith): each rung's muon under each direction's
        # overburden. The reach follows the muon, so the radius has no zenith
        # axis, but the projected area it feeds does.
        length = truncated_range_km(
            muon_energy[:, :, None], muon_column_km[None, None, :], threshold, b_scale
        )
        radius = light_reach_radius_km(
            ARCA_BLOCK_RADIUS_KM, muon_energy, reach_km, REACH_PIVOT_GEV
        )
        area_km2 = arca_projected_area_km2(theta_deg[None, None, :], radius[:, :, None])
        v_det_km3 = ARCA_N_BLOCKS * np.pi * radius**2 * ARCA_BLOCK_HEIGHT_KM
        volume_km3 = area_km2 * length + v_det_km3[:, :, None]
        sigma = _tilted_cc(energies, lam)
        rate = n_nucleon * sigma[:, :, None] * volume_km3 * CM_PER_KM**3
        total += weight * np.average((arrival * rate).sum(axis=1), axis=1, weights=zenith_weights)
    return eps_0 * total


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


@dataclass
class Detector:
    """One detector's published curve, forward model, priors, and results."""

    name: str
    log10_e: np.ndarray
    observed: np.ndarray
    mask: np.ndarray
    # (theta, select) -> effective area, on the whole grid when select is None.
    predict: Callable[[np.ndarray, np.ndarray | None], np.ndarray]
    priors: dict[str, tuple[float, float]]
    start: np.ndarray
    color: str
    selection_level: str
    chain: np.ndarray = field(default=None)
    best: np.ndarray = field(default=None)


def log_probability(theta: np.ndarray, detector: Detector, sigma_ln: float) -> float:
    """Flat-prior log posterior, Gaussian in ``ln A_eff``.

    ``b_scale`` is the exception to the flat priors: it is a calibrated
    transport quantity rather than a free knob, so it carries the same truncated
    Gaussian the event-rate fits use (:mod:`softpaws.comparison.likelihood`),
    identically for both detectors.
    """
    for value, name in zip(theta, PARAM_NAMES):
        low, high = detector.priors[name]
        if not low < value < high:
            return -np.inf
    log_prior = -0.5 * ((theta[2] - B_SCALE_MEAN) / B_SCALE_STD) ** 2
    predicted = detector.predict(theta, detector.mask)
    # A large positive reach drives the effective radius through zero at the
    # bottom of the ladder, which a geometric ceiling cannot represent; anything
    # non-positive or non-finite is outside the model rather than merely
    # unlikely, so it is rejected instead of scored.
    if not np.all(np.isfinite(predicted)) or np.any(predicted <= 0.0):
        return -np.inf
    residual = np.log(detector.observed[detector.mask] / predicted)
    return log_prior - 0.5 * float(np.sum((residual / sigma_ln) ** 2))


def run_fit(detector: Detector, steps: int, walkers: int, sigma: float, seed: int) -> None:
    """Sample one detector's posterior and store the chain on it."""
    # Per-parameter scatter: reach_km lives on a scale two orders of magnitude
    # below the others, so a common 0.02 would throw walkers out of its prior.
    scatter = np.array([0.02, 0.02, 0.02, 0.02, 0.002])
    rng = np.random.default_rng(seed)
    initial = detector.start + scatter * rng.standard_normal((walkers, detector.start.size))

    print(f"Sampling {detector.name} ({walkers} walkers x {steps} steps) ...")
    sampler = emcee.EnsembleSampler(
        walkers, detector.start.size, log_probability, args=(detector, sigma)
    )
    sampler.run_mcmc(initial, steps, progress=False)
    detector.chain = sampler.get_chain(discard=steps // 3, flat=True)
    detector.best = detector.chain[
        np.argmax(sampler.get_log_prob(discard=steps // 3, flat=True))
    ]


# ---------------------------------------------------------------------------
# Overlap region
# ---------------------------------------------------------------------------


def _interval(chain: np.ndarray, index: int, level: float) -> tuple[float, float]:
    half = 100.0 * (1.0 - level) / 2.0
    lo, hi = np.percentile(chain[:, index], [half, 100.0 - half])
    return float(lo), float(hi)


def overlap_region(detectors: list[Detector]) -> dict:
    """Quantify where the two posteriors agree, for the paper to read.

    Three things, in increasing strength.

    *Per-parameter intersection.* The overlap of the two 68% credible intervals,
    parameter by parameter. Reported for all five, but only the ``PHYSICS_PARAMS``
    are required to overlap -- the instrument parameters describe different
    hardware and an empty intersection there is not a discrepancy.

    *Product posterior.* The two fits are independent, so on the shared subspace
    their joint constraint is the product of the two marginals. Evaluated by
    kernel density estimate on a grid, which is what the paper should quote as
    the combined measurement of ``b_scale`` and ``lambda``.

    *Compatibility.* The Mahalanobis distance between the two posterior means on
    the shared subspace, against the summed covariances, as a 2-dof chi-square.
    This is the number that answers "do they agree", and it is a
    profile-style statement rather than a Bayes factor for the reason
    ``docs/`` records: the Bayes factor moves by a factor of two with the prior
    box, and this does not.

    Parameters
    ----------
    detectors : list of Detector
        Exactly two, each with a sampled ``chain``.

    Returns
    -------
    region : dict
        JSON-serializable summary; see the keys built below.
    """
    first, second = detectors
    region: dict = {
        "detectors": {
            d.name: {
                "selection_level": d.selection_level,
                "fit_band_log10_e": [
                    float(d.log10_e[d.mask].min()),
                    float(d.log10_e[d.mask].max()),
                ],
                "n_points": int(d.mask.sum()),
            }
            for d in detectors
        },
        "marginals": {},
        "intersection_68": {},
        "physics_params": list(PHYSICS_PARAMS),
    }

    for i, name in enumerate(PARAM_NAMES):
        region["marginals"][name] = {}
        for d in detectors:
            lo68, hi68 = _interval(d.chain, i, 0.68)
            lo95, hi95 = _interval(d.chain, i, 0.95)
            low, high = d.priors[name]
            # A 68% edge sitting on a prior edge means the prior box is doing
            # the constraining, not the data: the interval is then a bound, not
            # a measurement, and must not be quoted as one.
            span = high - low
            railed = [
                bool(lo68 - low < 0.02 * span),
                bool(high - hi68 < 0.02 * span),
            ]
            region["marginals"][name][d.name] = {
                "median": float(np.median(d.chain[:, i])),
                "best_fit": float(d.best[i]),
                "ci68": [lo68, hi68],
                "ci95": [lo95, hi95],
                "prior": [float(low), float(high)],
                "rails_prior": {"low": railed[0], "high": railed[1]},
            }
        lo = max(region["marginals"][name][d.name]["ci68"][0] for d in detectors)
        hi = min(region["marginals"][name][d.name]["ci68"][1] for d in detectors)
        region["intersection_68"][name] = {
            "low": float(lo),
            "high": float(hi),
            "empty": bool(hi <= lo),
            "comparable": name in CORNER_PARAMS,
        }

    # Product posterior on the shared subspace.
    indices = [PARAM_NAMES.index(name) for name in PHYSICS_PARAMS]
    bounds = []
    for i in indices:
        lo = min(np.percentile(d.chain[:, i], 0.5) for d in detectors)
        hi = max(np.percentile(d.chain[:, i], 99.5) for d in detectors)
        bounds.append((float(lo), float(hi)))
    axes = [np.linspace(lo, hi, 160) for lo, hi in bounds]
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.vstack([m.ravel() for m in mesh])
    density = np.ones(points.shape[1])
    for d in detectors:
        density *= gaussian_kde(d.chain[:, indices].T)(points)
    density = density.reshape(mesh[0].shape)
    total = density.sum()

    region["product_posterior"] = {"params": list(PHYSICS_PARAMS)}
    for axis, (name, grid) in enumerate(zip(PHYSICS_PARAMS, axes)):
        others = tuple(a for a in range(density.ndim) if a != axis)
        marginal = density.sum(axis=others) / total
        cumulative = np.cumsum(marginal)
        median, lo68, hi68 = np.interp([0.5, 0.16, 0.84], cumulative, grid)
        region["product_posterior"][name] = {
            "median": float(median),
            "ci68": [float(lo68), float(hi68)],
        }
    peak = np.unravel_index(np.argmax(density), density.shape)
    region["product_posterior"]["mode"] = {
        name: float(grid[peak[axis]]) for axis, (name, grid) in enumerate(zip(PHYSICS_PARAMS, axes))
    }

    # Compatibility on the shared subspace.
    means = [d.chain[:, indices].mean(axis=0) for d in detectors]
    covariances = [np.cov(d.chain[:, indices].T) for d in detectors]
    delta = means[0] - means[1]
    chi_square = float(delta @ np.linalg.solve(covariances[0] + covariances[1], delta))
    p_value = float(chi2.sf(chi_square, len(indices)))
    region["compatibility"] = {
        "params": list(PHYSICS_PARAMS),
        "dof": len(indices),
        "chi2": chi_square,
        "p_value": p_value,
        "sigma": float(np.sqrt(chi2.isf(p_value, 1))) if p_value > 0.0 else float("inf"),
        "delta": {name: float(delta[k]) for k, name in enumerate(PHYSICS_PARAMS)},
    }
    region["note"] = (
        f"{first.name} is compared at {first.selection_level} level and "
        f"{second.name} at {second.selection_level} level. eps_0 is a selection "
        "efficiency against its own curve and the two are therefore not on the "
        "same footing; only the physics_params carry a comparison."
    )
    return region


def summarize(detectors: list[Detector], region: dict) -> None:
    """Print the per-detector posteriors and the overlap region."""
    for d in detectors:
        print(f"\n{d.name} ({d.selection_level} level)")
        print(
            f"{'parameter':>14} {'best fit':>9} {'median':>8} {'16%':>8} "
            f"{'84%':>8} {'expected':>9}"
        )
        for i, name in enumerate(PARAM_NAMES):
            lo, med, hi = np.percentile(d.chain[:, i], [16, 50, 84])
            value = EXPECTED[d.name].get(name)
            expected = "--" if value is None else f"{value:.3f}"
            print(f"{name:>14} {d.best[i]:9.3f} {med:8.3f} {lo:8.3f} {hi:8.3f} {expected:>9}")

    print("\nOverlap of the 68% intervals")
    for name in PARAM_NAMES:
        entry = region["intersection_68"][name]
        tag = "" if entry["comparable"] else "   (instrument, not compared)"
        if entry["empty"]:
            print(f"{name:>14}   empty{tag}")
        else:
            print(f"{name:>14}   [{entry['low']:.3f}, {entry['high']:.3f}]{tag}")

    print("\nProduct posterior on the shared subspace (both fits are independent)")
    for name in PHYSICS_PARAMS:
        entry = region["product_posterior"][name]
        lo, hi = entry["ci68"]
        print(f"{name:>14}   {entry['median']:.3f}  [{lo:.3f}, {hi:.3f}]")

    compatibility = region["compatibility"]
    print(
        f"\nCompatibility on {', '.join(PHYSICS_PARAMS)}: "
        f"chi2 = {compatibility['chi2']:.2f} for {compatibility['dof']} dof, "
        f"p = {compatibility['p_value']:.3f}, {compatibility['sigma']:.1f} sigma"
    )

    rails = [
        f"{name} of {detector} against its {edge} prior edge"
        for name, entry in region["marginals"].items()
        for detector, values in entry.items()
        for edge in ("low", "high")
        if values["rails_prior"][edge]
    ]
    if rails:
        print("\n  WARNING: 68% interval rails against a prior edge --")
        for rail in rails:
            print(f"    {rail}")
        print("    Quote these as bounds, not as measurements.")

    print(f"  {region['note']}")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _forest_panel(
    ax: plt.Axes,
    detectors: list[Detector],
    name: str,
    scale: float = 1.0,
) -> None:
    """One horizontal-whisker row per detector for an instrument parameter.

    The detector names ride above their own whiskers rather than on the y axis.
    As tick labels they sit outside the panel, where the last column of the
    corner leaves them no room and they collide with the diagonal beside it.
    """
    index = PARAM_NAMES.index(name)
    for row, d in enumerate(detectors):
        y = len(detectors) - 1 - row
        lo95, hi95 = _interval(d.chain, index, 0.95)
        lo68, hi68 = _interval(d.chain, index, 0.68)
        median = float(np.median(d.chain[:, index])) * scale
        ax.plot([lo95 * scale, hi95 * scale], [y, y], color=d.color, lw=0.9, alpha=0.55)
        ax.plot([lo68 * scale, hi68 * scale], [y, y], color=d.color, lw=2.6, solid_capstyle="butt")
        ax.plot([median], [y], marker="o", ms=4.0, color=d.color,
                markeredgecolor="w", markeredgewidth=0.5, zorder=3)
        ax.text(lo95 * scale, y + 0.18, d.name, color=d.color, fontsize=8,
                ha="left", va="bottom")
        value = EXPECTED[d.name].get(name)
        if value is not None:
            ax.plot([value * scale], [y], marker="|", ms=9, color="0.35", zorder=4)

    ax.set_yticks([])
    # Headroom for the name above the topmost whisker.
    ax.set_ylim(-0.5, len(detectors) - 0.15)
    ax.set_xlabel(LABELS[name], fontsize=10, labelpad=1.5)
    ax.tick_params(axis="x", labelsize=8, pad=1.5)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def make_figure(detectors: list[Detector], out_path: pathlib.Path) -> None:
    """Draw the overlaid corner with the instrument marginals beside it.

    Three decluttering choices, all of them deliberate.

    The corner carries only ``CORNER_PARAMS``, so every panel in it is one where
    two overlapping contours mean something. The instrument parameters go into
    the upper triangle, which a corner plot otherwise leaves blank.

    The two posteriors are drawn differently rather than in two colours of the
    same weight: the first is filled, the second is unfilled and dashed. At two
    contour levels each, two translucent fills of equal weight are unreadable.

    No reference lines are drawn across the 2D panels. Marking a best fit in
    every panel costs two line segments per posterior per panel, which is where
    the ink actually goes; the derived values appear on the diagonal, where they
    read as values rather than as grid lines.
    """
    indices = [PARAM_NAMES.index(name) for name in CORNER_PARAMS]
    labels = [LABELS[name] for name in CORNER_PARAMS]
    k = len(indices)

    # Shared ranges, so the two corners agree about their own axes.
    ranges = []
    for i in indices:
        lo = min(np.percentile(d.chain[:, i], 0.5) for d in detectors)
        hi = max(np.percentile(d.chain[:, i], 99.5) for d in detectors)
        pad = 0.05 * (hi - lo)
        ranges.append((lo - pad, hi + pad))

    rc = {"xtick.labelsize": 9, "ytick.labelsize": 9, "axes.labelsize": 12, "font.size": 10}
    with plt.style.context(str(_STYLE)), plt.rc_context(rc):
        fig, _ = plt.subplots(k, k, figsize=(6.6, 6.6))
        for row, d in enumerate(detectors):
            base = np.array(plt.matplotlib.colors.to_rgb(d.color))
            filled = row == 0
            # corner fills outside-in, so the alphas run 0 -> between -> inside.
            fills = [(*base, 0.0), (*base, 0.12), (*base, 0.28)]
            corner.corner(
                d.chain[:, indices], labels=labels, range=ranges, color=d.color, fig=fig,
                plot_datapoints=False, plot_density=False, levels=(0.68, 0.95),
                fill_contours=filled,
                contourf_kwargs={"colors": fills} if filled else None,
                contour_kwargs={"linewidths": 1.1, "linestyles": "-" if filled else "--"},
                hist_kwargs={"density": True, "lw": 1.3,
                             "ls": "-" if filled else "--", "histtype": "step"},
                label_kwargs={"fontsize": 12}, smooth=0.8, no_fill_contours=not filled,
            )

        axes = np.array(fig.axes[: k * k]).reshape((k, k))
        # The derived values, on the diagonal only. Both detectors expect the
        # same b_scale and lam -- that is what makes them shared -- so the
        # distinct values are drawn once rather than stacked.
        for column, name in enumerate(CORNER_PARAMS):
            values = {EXPECTED[d.name][name] for d in detectors if name in EXPECTED[d.name]}
            for value in values:
                axes[column, column].axvline(value, color="0.35", lw=1.0, ls=":", zorder=0)
        # The DR2 smearing matrix's independent handle on IceCube's threshold.
        if "log10_e_thr" in CORNER_PARAMS:
            column = CORNER_PARAMS.index("log10_e_thr")
            axes[column, column].axvline(
                SMEARING_LOG10_E_THR, color="0.35", lw=1.0, ls=(0, (1, 2)), zorder=0
            )

        # The blank upper triangle, put to work. It is L-shaped, not
        # rectangular: cell (1, 1) is the b_mu diagonal and has to be left
        # alone. The last column, rows 0 to k-2, is the tall free block, and
        # cell (0, 1) beside it takes the legend.
        column = axes[0, k - 1].get_position()
        legend_cell = axes[0, 1].get_position()
        left, width = column.x0, column.width
        bottom, top = axes[k - 2, k - 1].get_position().y0, column.y1
        span = top - bottom

        legend_handles = [
            plt.Line2D([], [], color=d.color, lw=1.6, ls="-" if i == 0 else "--",
                       label=f"{d.name} ({d.selection_level})")
            for i, d in enumerate(detectors)
        ]
        legend_handles.append(
            plt.Line2D([], [], color="0.35", lw=1.0, ls=":", label="first principles")
        )
        legend_ax = fig.add_axes(
            [legend_cell.x0, legend_cell.y0, legend_cell.width, legend_cell.height]
        )
        legend_ax.axis("off")
        legend_ax.legend(handles=legend_handles, loc="upper left", frameon=False,
                         fontsize=9, handlelength=1.8, borderaxespad=0.0)

        forest = fig.add_axes([left, top - 0.40 * span, width, 0.32 * span])
        _forest_panel(forest, detectors, "eps_0")
        forest.set_title("instrument response,\nnot compared", fontsize=9, pad=5.0, color="0.35")

        reach = fig.add_axes([left, bottom + 0.18 * span, width, 0.32 * span])
        # Metres per e-fold: the two reaches differ by an order of magnitude and
        # kilometres would put IceCube's on top of zero.
        _forest_panel(reach, detectors, "reach_km", scale=1.0e3)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png"):
            path = out_path.with_suffix(suffix)
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"Figure saved to: {path.resolve()}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_detectors(data_dir: pathlib.Path) -> list[Detector]:
    """Load both published curves and bind each to its forward model."""
    print(f"Loading IceCube IRFs from: {data_dir}")
    ic_observed = icecube_upgoing(data_dir)
    ic_mask = (IC_LOG10_E >= IC_FIT_BAND[0]) & (IC_LOG10_E <= IC_FIT_BAND[1])

    print("Precomputing IceCube transmission ladders (parameter independent) ...")
    ic_ladders = icecube_ladders()

    print(f"Loading ARCA230 trigger-level curve from: {_ARCA230_TRIGGER_TABLE}")
    arca_observed = arca230_trigger()
    arca_mask = (
        (ARCA_LOG10_E >= ARCA_FIT_BAND[0])
        & (ARCA_LOG10_E <= ARCA_FIT_BAND[1])
        & np.isfinite(arca_observed)
    )

    print("Precomputing ARCA transmission ladders (parameter independent) ...")
    zenith_weights, neutrino_column, muon_column_km = arca_columns()
    ladders = arca_ladders(neutrino_column)

    start = np.array(
        [0.7, np.log10(DEFAULT_MUON_THRESHOLD_GEV), B_SCALE_MEAN, LAMBDA_BGR18,
         REACH_EXAMPLE28_KM]
    )
    return [
        Detector(
            name="IceCube",
            log10_e=IC_LOG10_E,
            observed=ic_observed,
            mask=ic_mask,
            predict=lambda theta, select=None: icecube_model(theta, ic_ladders, select),
            priors=PRIORS["IceCube"],
            start=start.copy(),
            color="C0",
            selection_level="analysis",
        ),
        Detector(
            name="ARCA230",
            log10_e=ARCA_LOG10_E,
            observed=arca_observed,
            mask=arca_mask,
            predict=lambda theta, select=None: arca_model(
                theta, ladders, zenith_weights, muon_column_km, select
            ),
            priors=PRIORS["ARCA230"],
            start=start.copy(),
            color="C1",
            selection_level="trigger",
        ),
    ]


def report_residuals(detector: Detector) -> None:
    """Print what each best fit leaves behind, energy by energy."""
    predicted = detector.predict(detector.best)
    ratio = detector.observed / predicted
    print(
        f"\n{detector.name}: {'log10(E/GeV)':>13} {'published':>11} "
        f"{'best fit':>11} {'ratio':>7}"
    )
    for log10_e in (5.0, 6.0, 7.0, 8.0):
        i = int(np.argmin(np.abs(detector.log10_e - log10_e)))
        if not np.isfinite(detector.observed[i]):
            continue
        flag = "" if detector.mask[i] else "  *"
        print(
            f"{'':>{len(detector.name) + 1}} {detector.log10_e[i]:13.1f} "
            f"{detector.observed[i]:11.3g} {predicted[i]:11.3g} {ratio[i]:7.2f}{flag}"
        )
    residual = np.log10(ratio[detector.mask])
    print(
        f"  rms {np.std(residual):.3f} dex, trend {residual[-1] - residual[0]:+.2f} dex "
        f"over the fitted band. * outside it."
    )


def main() -> None:
    args = parse_args()

    if args.check_truncation:
        check_truncation()
        return

    detectors = build_detectors(args.data_dir)
    for seed, detector in enumerate(detectors, start=11):
        run_fit(detector, args.steps, args.walkers, args.sigma, seed)

    region = overlap_region(detectors)
    summarize(detectors, region)
    for detector in detectors:
        report_residuals(detector)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    region["provenance"] = {
        "steps": args.steps,
        "walkers": args.walkers,
        "assumed_fractional_sigma": args.sigma,
        "burn_in_fraction": 1.0 / 3.0,
        "reach_pivot_gev": REACH_PIVOT_GEV,
        "lambda_pivot_gev": LAMBDA_PIVOT_GEV,
        "lambda_bgr18": LAMBDA_BGR18,
    }
    json_path = args.out.with_name("33_overlap_region.json")
    json_path.write_text(json.dumps(region, indent=2) + "\n")
    print(f"\nOverlap region written to: {json_path.resolve()}")

    npz_path = args.out.with_name("33_chains.npz")
    np.savez_compressed(
        npz_path,
        param_names=np.array(PARAM_NAMES),
        **{f"{d.name}_chain": d.chain for d in detectors},
        **{f"{d.name}_best": d.best for d in detectors},
    )
    print(f"Chains written to: {npz_path.resolve()}")

    make_figure(detectors, args.out)


if __name__ == "__main__":
    main()
