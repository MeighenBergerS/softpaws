"""Per-site forward models for the published effective-area tables.

Every detector the paper compares runs through the same five-number
parameter vector, :data:`PARAM_NAMES`:

``eps_0``
    Overall selection efficiency, a pure normalization.
``log10_e_thr``
    Muon energy the selection accepts, as ``log10(E / GeV)``.
``b_scale``
    Rescaling of the drift coefficient, and with it of ``Phi'(0)``.
``lam``
    Effective log-log slope of the charged-current cross section.
``reach_km``
    Light reach, as the extra radius the body gains per e-fold of muon
    energy above :data:`REACH_PIVOT_GEV`.

The first two and the last are instrument numbers, the middle two are
properties of the loss kernel and of the Earth. Two independent fits
landing on the same ``b_scale`` and ``lam`` is the claim the four-detector
corner makes, so the two are never given per-site priors.

Two geometries are implemented. IceCube is an upright hexagonal prism
under a hemisphere-averaged transmission ladder, because at the Pole a
declination is a zenith and the average commutes with everything except
the direction-dependent projected area. The water sites are upright
cylinders under per-zenith ladders, because their muons see a finite
overburden that varies with direction, so the average has to wait until
the truncated first-passage length has been applied. KM3NeT/ARCA230,
P-ONE and TRIDENT are the same model at different geometry, ``f_tau``,
regeneration setting and depth.

Notes
-----
Nothing here plots or prints. The published curves come from
:mod:`softpaws.data.published`, the layouts from :mod:`softpaws.detectors`
and the columns from :mod:`softpaws.transport.earth`.

The loaders are imported inside the functions that read a table.
:mod:`softpaws.data` imports :mod:`softpaws.response.irfs` for its container
types, so the package layering runs data -> response, and a module-level
import the other way would close the cycle.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from softpaws.detectors import ARCA230, ICECUBE, MAX_UPSTREAM_KM, PONE, TRIDENT
from softpaws.transport.attenuation import (
    flavour_transmission,
    prem_column,
    regenerated_transmission,
)
from softpaws.transport.coefficients import diffusion_coefficient, drift_coefficient
from softpaws.transport.cross_section import bgr18_cross_section
from softpaws.transport.earth import neutrino_column_g_cm2, overburden_km
from softpaws.transport.earth import zenith_grid as _zenith_grid
from softpaws.transport.soft_volume import (
    DEFAULT_MUON_THRESHOLD_GEV,
    light_reach_radius_km,
    muon_range_km,
    stochastic_muon_range_km,
    truncated_muon_range_km,
    two_medium_range_ratio,
)
from softpaws.transport.source import MEAN_INELASTICITY, nucleon_number_density
from softpaws.transport.tau import BR_TAU_TO_MU, MEAN_Z
from softpaws.utils.constants import CM_PER_KM, RHO_WATER_G_CM3

__all__ = [
    "ARCA230_WATER_SITE",
    "ARCA_BLOCK_HEIGHT_KM",
    "ARCA_BLOCK_RADIUS_KM",
    "ARCA_DEPTH_KM",
    "ARCA_FIT_BAND",
    "ARCA_LOG10_E",
    "ARCA_MAX_SEA_PATH_KM",
    "ARCA_N_BLOCKS",
    "ARCA_N_RUNG",
    "ARCA_N_ZENITH",
    "ARCA_WATER_BELOW_KM",
    "B_SCALE_FLOOR",
    "CORNER_PARAMS",
    "CROSS_SECTION",
    "DIGITIZED_FIT_BAND",
    "EXPECTED",
    "F_TAU",
    "IC_FIT_BAND",
    "IC_HEIGHT_KM",
    "IC_ICE_BELOW_KM",
    "IC_LOG10_E",
    "IC_N_DEC",
    "IC_N_RUNG",
    "IC_N_SIDES",
    "IC_RADIUS_KM",
    "IC_SIDE_COEFF",
    "INSTRUMENT_PARAMS",
    "LAMBDA_BGR18",
    "LAMBDA_PIVOT_GEV",
    "PARAM_LABELS",
    "PARAM_NAMES",
    "PHYSICS_PARAMS",
    "PRIORS",
    "REACH_EXAMPLE28_KM",
    "REACH_PIVOT_GEV",
    "SMEARING_LOG10_E_THR",
    "TRIDENT_BAND_WEIGHTS",
    "Detector",
    "WaterSite",
    "arca230_trigger",
    "arca_columns",
    "arca_ladders",
    "arca_model",
    "arca_projected_area_km2",
    "arca_zenith_grid",
    "build_arca_detector",
    "build_detectors",
    "build_four_detectors",
    "build_icecube_detector",
    "build_water_detector",
    "default_start",
    "icecube_ladders",
    "icecube_model",
    "icecube_upgoing",
    "on_arca_grid",
    "pone_allsky_cm2",
    "published_curve",
    "tilted_cc",
    "trident_allsky_cm2",
    "truncation_table",
    "water_columns",
    "water_ladders",
    "water_model",
    "water_projected_area_km2",
    "water_sites",
]

#: Charged-current and neutral-current cross sections every model here uses.
CROSS_SECTION = bgr18_cross_section()

# ---------------------------------------------------------------------------
# Parameters. The same five for every detector, split by what may be compared.
# ---------------------------------------------------------------------------

#: The full parameter vector, in the order every ``theta`` array carries.
PARAM_NAMES = ("eps_0", "log10_e_thr", "b_scale", "lam", "reach_km")

#: Drawn as a corner. ``b_scale`` and ``lam`` are shared physics;
#: ``log10_e_thr`` is an instrument number on a common scale, kept here
#: because its ordering between trigger and analysis level is worth seeing.
CORNER_PARAMS = ("log10_e_thr", "b_scale", "lam")

#: Reported as marginals beside the corner: different medium, different array.
INSTRUMENT_PARAMS = ("eps_0", "reach_km")

#: The subspace where agreement between two sites is a physics statement.
PHYSICS_PARAMS = ("b_scale", "lam")

#: Math labels of the five parameters, for figures.
PARAM_LABELS = {
    "eps_0": r"$\varepsilon_0$",
    "log10_e_thr": r"$\log_{10}(E_{\rm thr}/{\rm GeV})$",
    "b_scale": r"$b_\mu$ scale",
    "lam": r"$\lambda$",
    "reach_km": r"$\Lambda$ [m per e-fold]",
}

#: Effective log-log slope of the BGR18 charged-current cross section over the
#: fitted band. ``lam = LAMBDA_BGR18`` recovers the tabulated values.
LAMBDA_BGR18 = 0.4538

#: Pivot the cross-section tilt rotates about [GeV].
LAMBDA_PIVOT_GEV = 1.0e6

#: Pivot of the reach law [GeV], held at the cross-section tilt's pivot so that
#: ``eps_0`` carries the normalization and ``reach_km`` carries only the shape.
#: The same value at every site, without which two ``Lambda`` marginals would
#: not be on the same footing even as marginals.
REACH_PIVOT_GEV = 1.0e6

#: Tau-neutrino fraction of the astrophysical flux, fixed by oscillations over
#: astrophysical baselines and so not a fit parameter.
F_TAU = 1.0

#: Lower edge of the ``b_scale`` prior, and a physical boundary rather than a
#: convenience. The two-moment loss spectrum ``dGamma/dy = kappa (1-y)^p / y``
#: is calibrated by ``p = b_mu / d_mu - 2``, so it exists only for
#: ``d_mu < b_mu``. Rescaling ``b_mu`` without rescaling ``d_mu`` therefore has
#: a floor: below ``max(d_mu / b_mu)`` over the energies in play, ``p`` drops
#: past -1, ``kappa`` turns negative and there is no loss spectrum to
#: propagate. Over ``10^2`` to ``10^10`` GeV that floor is 0.287, far outside
#: where any posterior lives but reachable by a walker proposal, where it would
#: otherwise surface as a NaN rather than as a rejection.
B_SCALE_FLOOR = float(
    np.max(
        diffusion_coefficient(np.logspace(2.0, 10.0, 200))
        / drift_coefficient(np.logspace(2.0, 10.0, 200))
    )
)

#: Independent expectation for IceCube's light reach [km per e-fold]: the
#: published-to-model ratio inverted for the radius each energy demands, with a
#: straight line fitted through it in ``ln E``. The water sites have no
#: counterpart.
REACH_EXAMPLE28_KM = 0.0193

#: The DR2 smearing matrix's own handle on IceCube's threshold: the 5th
#: percentile of accepted reconstructed muon energy, flat at ~700 GeV across
#: three decades of ``E_nu``, as ``log10(E / GeV)``. No water site publishes an
#: equivalent.
SMEARING_LOG10_E_THR = 2.85

#: Where each parameter was derived to sit, independently of any fit. A missing
#: key marks a parameter with no first-principles value for that detector.
EXPECTED: dict[str, dict[str, float]] = {
    "IceCube": {
        "log10_e_thr": float(np.log10(DEFAULT_MUON_THRESHOLD_GEV)),
        "b_scale": 1.0,
        "lam": LAMBDA_BGR18,
        "reach_km": REACH_EXAMPLE28_KM,
    },
    "ARCA230": {"b_scale": 1.0, "lam": LAMBDA_BGR18},
    "P-ONE": {"b_scale": 1.0, "lam": LAMBDA_BGR18},
    "TRIDENT": {"b_scale": 1.0, "lam": LAMBDA_BGR18},
}

#: Flat prior boxes, per site. The two shared-physics parameters carry
#: identical priors by construction: a per-site prior on ``b_scale`` or ``lam``
#: would put part of the difference between two posteriors in the prior. Only
#: the instrument parameters differ, and only where the instrument demands it.
#: ``eps_0``'s upper edge is physical, since the model is a geometric ceiling
#: and a selection cannot exceed it. Zero sits inside every ``reach_km`` range,
#: so the data can say no reach is needed.
PRIORS: dict[str, dict[str, tuple[float, float]]] = {
    "IceCube": {
        "eps_0": (0.0, 1.0),
        "log10_e_thr": (2.0, 5.0),
        "b_scale": (B_SCALE_FLOOR, 3.0),
        "lam": (0.0, 1.2),
        "reach_km": (-0.02, 0.10),
    },
    "ARCA230": {
        "eps_0": (0.0, 1.0),
        "log10_e_thr": (2.0, 5.0),
        "b_scale": (B_SCALE_FLOOR, 3.0),
        "lam": (0.0, 1.2),
        # Wider than IceCube's: ARCA's blocks are 517 m across against a
        # kilometre-scale prism, so the reach a trigger-level curve demands is
        # a larger fraction of the footprint.
        "reach_km": (-0.05, 0.40),
    },
}

# ---------------------------------------------------------------------------
# IceCube: an upright hexagonal prism under a hemisphere-averaged ladder
# ---------------------------------------------------------------------------

#: Energy grid of the IceCube fit, as ``log10(E_nu / GeV)``.
IC_LOG10_E = np.linspace(3.0, 8.0, 26)

#: Fitted band [log10 GeV]. The top of the DR2 simulation, 100 PeV, is left out.
IC_FIT_BAND = (5.0, 7.8)

#: Instrumented height [km] of the prism.
IC_HEIGHT_KM = ICECUBE.height_km

#: Number of sides of the prism cross-section.
IC_N_SIDES = ICECUBE.n_sides

#: Area-equivalent radius of the hexagonal footprint [km].
IC_RADIUS_KM = ICECUBE.radius_km

#: The prism perimeter divided by ``pi R``, the coefficient of the
#: side-projection term.
IC_SIDE_COEFF = float(2.0 * np.sqrt(np.pi * IC_N_SIDES * np.tan(np.pi / IC_N_SIDES)) / np.pi)

#: Declination samples of the hemisphere average.
IC_N_DEC = 40

#: Rungs of the IceCube transmission ladder.
IC_N_RUNG = 80

#: Ice below the centre of the instrumented body, along the vertical, in the
#: water-equivalent units the ranges are in: 370 m under the deepest module and
#: half the array above that, at 0.918 g cm^-3. Rock lies beneath, and the
#: two-medium range shortens every upgoing entering term for it.
IC_ICE_BELOW_KM = (ICECUBE.below_km + 0.5 * IC_HEIGHT_KM) * 0.918

# ---------------------------------------------------------------------------
# Water sites: upright cylinders under per-zenith ladders
# ---------------------------------------------------------------------------

#: Footprint radius of one ARCA building block [km].
ARCA_BLOCK_RADIUS_KM = ARCA230.radius_km

#: Instrumented height of one ARCA building block [km].
ARCA_BLOCK_HEIGHT_KM = ARCA230.height_km

#: Number of ARCA building blocks in the ARCA230 layout.
ARCA_N_BLOCKS = ARCA230.n_blocks

#: Depth of the ARCA instrumented centre below the sea surface [km].
ARCA_DEPTH_KM = ARCA230.depth_km

#: Sea water between the centre of the ARCA body and the sea floor [km].
ARCA_WATER_BELOW_KM = ARCA230.below_km + 0.5 * ARCA_BLOCK_HEIGHT_KM

#: Longest sea-water path a near-horizontal muon can have [km]. Only a cap on
#: the ``1 / cos(theta)`` divergence; it exceeds every muon range in play.
ARCA_MAX_SEA_PATH_KM = MAX_UPSTREAM_KM

#: Energy grid of every water-site fit, as ``log10(E_nu / GeV)``.
ARCA_LOG10_E = np.arange(4.0, 8.01, 0.2)

#: Fitted band of the ARCA230 trigger curve [log10 GeV]. The digitized curve
#: saturates at the edge of the published figure in its last half decade, so
#: the fit stops short of it; the lower edge matches IceCube's.
ARCA_FIT_BAND = (5.0, 7.5)

#: Zenith bands of the water-site sky average. Both the projected area and the
#: column vary smoothly with zenith, so a thinned grid costs 4e-4 dex against a
#: 48-band one, three orders of magnitude below the assumed node error.
ARCA_N_ZENITH = 20

#: Rungs of a water-site transmission ladder.
ARCA_N_RUNG = 32

#: Fitted band of the two digitized sky averages [log10 GeV]: from the ARCA
#: lower edge to the last smoothed bin below the plotted edge.
DIGITIZED_FIT_BAND = (5.0, 6.9)

#: Solid-angle weights of TRIDENT's three published ``cos(theta)`` bands.
TRIDENT_BAND_WEIGHTS = np.array([0.8, 0.4, 0.8]) / 2.0

#: Bin width [dex] of the median filter applied to a digitized step curve.
SMOOTH_BIN_DEX = 0.1

#: Width [bins] of the running mean applied after that median filter.
SMOOTH_WINDOW = 5


# ---------------------------------------------------------------------------
# Published curves
# ---------------------------------------------------------------------------


def icecube_upgoing(data_dir: pathlib.Path) -> np.ndarray:
    """Livetime-weighted DR2 effective area over the upgoing sky.

    Parameters
    ----------
    data_dir : pathlib.Path
        Root of the IceTracks-DR2 release, holding ``irfs/`` and ``uptime/``.

    Returns
    -------
    aeff_cm2 : np.ndarray, shape (IC_LOG10_E.size,)
        Effective area on :data:`IC_LOG10_E` [cm^2].
    """
    from softpaws.data.published import icecube_dr2_aeff

    return icecube_dr2_aeff(data_dir, IC_LOG10_E)[0]


def arca230_trigger() -> np.ndarray:
    """Digitized full-ARCA ``nu_mu`` effective area at trigger level.

    Returns
    -------
    aeff_cm2 : np.ndarray, shape (ARCA_LOG10_E.size,)
        Effective area on :data:`ARCA_LOG10_E` [cm^2], ``NaN`` outside the
        digitized range.
    """
    from softpaws.data.published import arca230_trigger_level_aeff, interpolate_aeff

    return interpolate_aeff(ARCA_LOG10_E, *arca230_trigger_level_aeff())


def _smooth_digitized(log10_e: np.ndarray, log10_a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Smooth a digitized step-function curve.

    Median of ``log10_a`` in :data:`SMOOTH_BIN_DEX` bins of ``log10_e``, then a
    :data:`SMOOTH_WINDOW`-bin running mean, evaluated at the bin centres.

    Parameters
    ----------
    log10_e : np.ndarray
        ``log10(E_nu / GeV)`` of the digitized points.
    log10_a : np.ndarray
        ``log10`` of the digitized effective area.

    Returns
    -------
    centers : np.ndarray
        Bin centres that hold at least one digitized point.
    smoothed : np.ndarray
        The smoothed curve on ``centers``.
    """
    lo = np.floor(log10_e.min() / SMOOTH_BIN_DEX) * SMOOTH_BIN_DEX
    edges = np.arange(lo, log10_e.max() + SMOOTH_BIN_DEX, SMOOTH_BIN_DEX)
    index = np.clip(np.digitize(log10_e, edges) - 1, 0, edges.size - 2)
    centers, medians = [], []
    for k in range(edges.size - 1):
        selected = index == k
        if selected.any():
            centers.append(0.5 * (edges[k] + edges[k + 1]))
            medians.append(np.median(log10_a[selected]))
    centers, medians = np.array(centers), np.array(medians)
    half = SMOOTH_WINDOW // 2
    padded = np.pad(medians, half, mode="edge")
    kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
    return centers, np.convolve(padded, kernel, mode="valid")


def on_arca_grid(log10_e: np.ndarray, log10_a: np.ndarray) -> np.ndarray:
    """Put a smoothed curve on :data:`ARCA_LOG10_E` by log-log interpolation.

    Parameters
    ----------
    log10_e : np.ndarray
        ``log10(E_nu / GeV)`` of the curve, ascending.
    log10_a : np.ndarray
        ``log10`` of the effective area [cm^2].

    Returns
    -------
    aeff_cm2 : np.ndarray, shape (ARCA_LOG10_E.size,)
        Effective area [cm^2], ``NaN`` outside the curve.
    """
    return 10.0 ** np.interp(ARCA_LOG10_E, log10_e, log10_a, left=np.nan, right=np.nan)


def pone_allsky_cm2() -> np.ndarray:
    """P-ONE's all-sky trigger-level curve, smoothed, on :data:`ARCA_LOG10_E`.

    Returns
    -------
    aeff_cm2 : np.ndarray, shape (ARCA_LOG10_E.size,)
        Effective area [cm^2], ``NaN`` outside the digitized range.
    """
    from softpaws.data.published import pone_allsky_aeff

    log10_e, aeff_cm2 = pone_allsky_aeff()
    return on_arca_grid(*_smooth_digitized(log10_e, np.log10(aeff_cm2)))


def trident_allsky_cm2() -> np.ndarray:
    """TRIDENT's sky average from its three bands, on :data:`ARCA_LOG10_E`.

    The three published ``cos(theta)`` bands are smoothed, put on the grid and
    combined with the solid-angle weights of :data:`TRIDENT_BAND_WEIGHTS`.

    Returns
    -------
    aeff_cm2 : np.ndarray, shape (ARCA_LOG10_E.size,)
        Effective area [cm^2], ``NaN`` outside the digitized range.
    """
    from softpaws.data.published import TRIDENT_COS_BANDS, trident_band_aeff

    bands = []
    for cos_lo, cos_hi in TRIDENT_COS_BANDS:
        log10_e, aeff_cm2 = trident_band_aeff(cos_lo, cos_hi)
        bands.append(on_arca_grid(*_smooth_digitized(log10_e, np.log10(aeff_cm2))))
    return np.sum([w * b for w, b in zip(TRIDENT_BAND_WEIGHTS, bands)], axis=0)


#: Published sky average of each water site, by name.
published_curve: dict[str, Callable[[], np.ndarray]] = {
    "P-ONE": pone_allsky_cm2,
    "TRIDENT": trident_allsky_cm2,
}


# ---------------------------------------------------------------------------
# Cross-section tilt
# ---------------------------------------------------------------------------


def tilted_cc(energy_gev: np.ndarray, lam: float) -> np.ndarray:
    """BGR18 charged-current cross section tilted about the pivot.

    Parameters
    ----------
    energy_gev : np.ndarray
        Neutrino energy [GeV].
    lam : float
        Effective log-log slope. ``lam = LAMBDA_BGR18`` recovers the tabulated
        values exactly.

    Returns
    -------
    sigma : np.ndarray
        Charged-current cross section per nucleon [cm^2].
    """
    return CROSS_SECTION.cc(energy_gev) * (energy_gev / LAMBDA_PIVOT_GEV) ** (lam - LAMBDA_BGR18)


# ---------------------------------------------------------------------------
# IceCube geometry and transmission
# ---------------------------------------------------------------------------


def icecube_ladders() -> dict[str, tuple[np.ndarray, ...]]:
    """Hemisphere-averaged transmission ladders for IceCube, one per flavour.

    Neither ladder depends on a fitted parameter, since both are set by the
    cross section and the PREM column alone, so they are built once and reused
    for every posterior sample.

    Returns
    -------
    ladders : dict
        ``"mu"`` and ``"tau"`` -> ``(energies, weights, weights_cos,
        weights_sin)``, all of shape ``(IC_LOG10_E.size, IC_N_RUNG)``, with
        ``energies`` in GeV and the three weight arrays the hemisphere averages
        of the arrival probability against 1, ``|cos theta_z|`` and
        ``sin theta_z``.

    Notes
    -----
    The prism of :func:`icecube_model` presents a direction-dependent area, so
    the declination average no longer commutes with the target volume. The
    volume is linear in the two geometry terms, so averaging the transmission
    against each separately is exact and still collapses the declination axis
    once. At the Pole ``|cos theta_z| = sin(dec)``.
    """
    dec_deg = np.linspace(0.5, 89.5, IC_N_DEC)
    columns = np.array([prem_column(float(d)) for d in dec_deg])
    dec_rad = np.deg2rad(dec_deg)
    solid_angle = np.cos(dec_rad)
    cos_theta = np.sin(dec_rad)
    sin_theta = np.cos(dec_rad)

    ladders: dict[str, tuple[np.ndarray, ...]] = {}
    for flavour in ("mu", "tau"):
        muon_fraction = (1.0 - MEAN_INELASTICITY) * (1.0 if flavour == "mu" else MEAN_Z)
        energies = np.empty((IC_LOG10_E.size, IC_N_RUNG))
        weights = np.empty((IC_LOG10_E.size, IC_N_RUNG))
        weights_cos = np.empty((IC_LOG10_E.size, IC_N_RUNG))
        weights_sin = np.empty((IC_LOG10_E.size, IC_N_RUNG))
        for i, log10_e in enumerate(IC_LOG10_E):
            rung_energy, rung_weight = flavour_transmission(
                10.0**log10_e,
                columns,
                CROSS_SECTION,
                flavour=flavour,
                n_grid=IC_N_RUNG,
                decades=4.0,
            )
            # The rock below the ice shortens the entering term, which the two
            # geometry weights multiply and the instrumented term does not. The
            # ratio is read at the default threshold and kernel scale; it moves
            # by under 2% across the fitted range of either, so the ladders
            # stay parameter independent.
            rock = np.array(
                [
                    two_medium_range_ratio(
                        float(muon_fraction * e),
                        DEFAULT_MUON_THRESHOLD_GEV,
                        -cos_theta,
                        IC_ICE_BELOW_KM,
                    )
                    for e in rung_energy
                ]
            )
            energies[i] = rung_energy
            weights[i] = np.average(rung_weight, axis=1, weights=solid_angle)
            weights_cos[i] = np.average(
                rung_weight * rock * cos_theta[None, :], axis=1, weights=solid_angle
            )
            weights_sin[i] = np.average(
                rung_weight * rock * sin_theta[None, :], axis=1, weights=solid_angle
            )
        ladders[flavour] = (energies, weights, weights_cos, weights_sin)
    return ladders


def icecube_model(
    theta: np.ndarray,
    ladders: dict[str, tuple[np.ndarray, ...]],
    select: np.ndarray | None = None,
) -> np.ndarray:
    """Predicted IceCube effective area for one parameter vector.

    Parameters
    ----------
    theta : np.ndarray, shape (5,)
        ``(eps_0, log10_e_thr, b_scale, lam, reach_km)``.
    ladders : dict
        Output of :func:`icecube_ladders`.
    select : np.ndarray or None, optional
        Boolean mask over :data:`IC_LOG10_E`. ``None``, the default, evaluates
        the whole grid; a sampler passes the fitted band, since the nodes
        outside it enter no likelihood and cost the same as the ones that do.

    Returns
    -------
    aeff_cm2 : np.ndarray
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
        energies, arrival, arrival_cos, arrival_sin = ladders[flavour]
        energies, arrival = energies[nodes], arrival[nodes]
        arrival_cos, arrival_sin = arrival_cos[nodes], arrival_sin[nodes]
        muon_energy = muon_fraction * energies
        length = stochastic_muon_range_km(
            muon_energy.ravel(), threshold, b_scale=b_scale
        ).reshape(muon_energy.shape)
        radius = light_reach_radius_km(IC_RADIUS_KM, muon_energy, reach_km, REACH_PIVOT_GEV)
        sigma = tilted_cc(energies, lam)
        # Each geometry term carries its own declination average; see
        # icecube_ladders. V_det is isotropic and rides on the plain one.
        cap = np.pi * radius**2 * length * arrival_cos
        side = IC_SIDE_COEFF * radius * IC_HEIGHT_KM * length * arrival_sin
        v_det = np.pi * radius**2 * IC_HEIGHT_KM * arrival
        total += weight * (n_nucleon * sigma * CM_PER_KM**3 * (cap + side + v_det)).sum(axis=1)
    return eps_0 * total


# ---------------------------------------------------------------------------
# Water sites
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WaterSite:
    """One upright-cylinder water detector, as :func:`water_model` sees it.

    Attributes
    ----------
    name : str
        Detector name, as printed in tables and figures.
    radius_km : float
        Footprint radius of one building block [km].
    height_km : float
        Instrumented height of one building block [km].
    n_blocks : int
        Number of identical blocks.
    depth_km : float
        Depth of the instrumented centre below the sea surface [km].
    selection_level : str
        Selection the published curve is quoted at, for example ``"trigger"``.
    fit_band : tuple of float
        Fitted band [log10 GeV].
    reach_prior : tuple of float
        Flat prior box of ``reach_km`` [km per e-fold].
    f_tau : float
        ``tau -> mu`` channel weight; 0 for a ``nu_mu``-only simulation.
    regeneration : bool
        Neutral-current regeneration in the Earth; ``False`` for the pure
        absorption the collaborations simulate.
    below_km : float
        Water between the bottom of the instrumented volume and the sea floor
        [km]. Rock lies beneath, and shortens every upgoing entering term.
    """

    name: str
    radius_km: float
    height_km: float
    n_blocks: int
    depth_km: float
    selection_level: str
    fit_band: tuple[float, float]
    reach_prior: tuple[float, float]
    f_tau: float
    regeneration: bool
    below_km: float


#: KM3NeT/ARCA230 as a :class:`WaterSite`: two building blocks, the tau channel
#: on, and neutral-current regeneration in the Earth.
ARCA230_WATER_SITE = WaterSite(
    name="ARCA230",
    radius_km=ARCA_BLOCK_RADIUS_KM,
    height_km=ARCA_BLOCK_HEIGHT_KM,
    n_blocks=ARCA_N_BLOCKS,
    depth_km=ARCA_DEPTH_KM,
    selection_level="trigger",
    fit_band=ARCA_FIT_BAND,
    reach_prior=PRIORS["ARCA230"]["reach_km"],
    f_tau=F_TAU,
    regeneration=True,
    below_km=ARCA230.below_km,
)


def water_sites() -> list[WaterSite]:
    """P-ONE and TRIDENT in the convention their collaborations simulate.

    Both curves are ``nu_mu`` only and both are compared against pure
    absorption in the Earth, since a neutral-current ladder sits above their
    upgoing bands by construction. P-ONE's strings stand on the Cascadia Basin
    floor; TRIDENT's block sits about 100 m above the South China Sea bed.

    Returns
    -------
    sites : list of WaterSite
        P-ONE first, then TRIDENT.
    """
    return [
        WaterSite(
            name=PONE.name,
            radius_km=PONE.radius_km,
            height_km=PONE.height_km,
            n_blocks=PONE.n_blocks,
            depth_km=PONE.depth_km,
            selection_level="trigger",
            fit_band=DIGITIZED_FIT_BAND,
            reach_prior=(-0.05, 0.40),
            f_tau=0.0,
            regeneration=False,
            below_km=PONE.below_km,
        ),
        WaterSite(
            name=TRIDENT.name,
            radius_km=TRIDENT.radius_km,
            height_km=TRIDENT.height_km,
            n_blocks=TRIDENT.n_blocks,
            depth_km=TRIDENT.depth_km,
            selection_level="6 deg cut",
            fit_band=DIGITIZED_FIT_BAND,
            reach_prior=(-0.05, 0.40),
            f_tau=0.0,
            regeneration=False,
            below_km=TRIDENT.below_km,
        ),
    ]


def arca_zenith_grid() -> tuple[np.ndarray, np.ndarray]:
    """Whole-sky zenith grid every water site shares.

    Returns
    -------
    theta_deg : np.ndarray, shape (ARCA_N_ZENITH,)
        Zenith angle [deg]; see :func:`softpaws.transport.earth.zenith_grid`.
    weights : np.ndarray, shape (ARCA_N_ZENITH,)
        Solid-angle weights, summing to one.
    """
    return _zenith_grid(ARCA_N_ZENITH)


def water_columns(site: WaterSite) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zenith weights and the two columns of one water site.

    Parameters
    ----------
    site : WaterSite
        The detector, which sets the depth.

    Returns
    -------
    weights : np.ndarray, shape (ARCA_N_ZENITH,)
        Solid-angle weights over the whole sky.
    neutrino_column : np.ndarray, shape (ARCA_N_ZENITH,)
        Column traversed before reaching the detector [g cm^-2].
    muon_column_km : np.ndarray, shape (ARCA_N_ZENITH,)
        Column available upstream of the detector [km of sea water].
    """
    theta_deg, weights = arca_zenith_grid()
    cos_theta = np.cos(np.deg2rad(theta_deg))
    neutrino_column = neutrino_column_g_cm2(
        cos_theta, site.depth_km, RHO_WATER_G_CM3, ARCA_MAX_SEA_PATH_KM
    )
    muon_column_km = overburden_km(cos_theta, site.depth_km, ARCA_MAX_SEA_PATH_KM)
    return weights, neutrino_column, muon_column_km


def arca_columns() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zenith weights and the two columns of ARCA230.

    Returns
    -------
    weights : np.ndarray, shape (ARCA_N_ZENITH,)
        Solid-angle weights over the whole sky.
    neutrino_column : np.ndarray, shape (ARCA_N_ZENITH,)
        Column traversed before reaching the detector [g cm^-2].
    muon_column_km : np.ndarray, shape (ARCA_N_ZENITH,)
        Column available upstream of the detector [km of sea water].
    """
    return water_columns(ARCA230_WATER_SITE)


def arca_ladders(
    neutrino_column: np.ndarray,
    water_below_km: float = ARCA_WATER_BELOW_KM,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per-zenith transmission ladders with neutral-current regeneration.

    Unlike IceCube's, these are not averaged over direction: the muon's
    available column varies with zenith too, so the average has to wait until
    the truncated range has been applied.

    Parameters
    ----------
    neutrino_column : np.ndarray, shape (ARCA_N_ZENITH,)
        Column traversed before reaching the detector [g cm^-2].
    water_below_km : float, optional
        Water between the centre of the instrumented body and the sea floor,
        along the vertical [km]. Defaults to :data:`ARCA_WATER_BELOW_KM`.

    Returns
    -------
    ladders : dict
        ``"mu"`` and ``"tau"`` -> ``(energies, weights, rock)`` with
        ``energies`` of shape ``(ARCA_LOG10_E.size, ARCA_N_RUNG)`` [GeV],
        ``weights`` of shape ``(ARCA_LOG10_E.size, ARCA_N_RUNG,
        ARCA_N_ZENITH)``, and ``rock`` the two-medium range ratio of the same
        shape, 1 above the horizon and 0.80 to 0.86 below it, read at the
        default threshold and kernel scale.
    """
    theta_deg, _ = arca_zenith_grid()
    cos_theta = np.cos(np.deg2rad(theta_deg))
    ladders: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for flavour in ("mu", "tau"):
        muon_fraction = (1.0 - MEAN_INELASTICITY) * (1.0 if flavour == "mu" else MEAN_Z)
        energies = np.empty((ARCA_LOG10_E.size, ARCA_N_RUNG))
        weights = np.empty((ARCA_LOG10_E.size, ARCA_N_RUNG, ARCA_N_ZENITH))
        rock = np.ones_like(weights)
        for i, log10_e in enumerate(ARCA_LOG10_E):
            rung_energy, rung_weight = flavour_transmission(
                10.0**log10_e,
                neutrino_column,
                CROSS_SECTION,
                flavour=flavour,
                n_grid=ARCA_N_RUNG,
                decades=4.0,
            )
            energies[i] = rung_energy
            weights[i] = rung_weight
            rock[i] = np.array(
                [
                    two_medium_range_ratio(
                        float(muon_fraction * e),
                        DEFAULT_MUON_THRESHOLD_GEV,
                        cos_theta,
                        water_below_km,
                    )
                    for e in rung_energy
                ]
            )
        ladders[flavour] = (energies, weights, rock)
    return ladders


def water_ladders(
    site: WaterSite, neutrino_column: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Transmission ladders of one water site.

    A site with ``regeneration`` set gets :func:`arca_ladders`. A site without
    it gets a one-rung ``nu_mu`` ladder of pure absorption and an empty tau
    channel, which is the convention the P-ONE and TRIDENT studies simulate.

    Parameters
    ----------
    site : WaterSite
        The detector, which sets the overburden and the regeneration setting.
    neutrino_column : np.ndarray, shape (ARCA_N_ZENITH,)
        Column traversed before reaching the detector [g cm^-2].

    Returns
    -------
    ladders : dict
        ``"mu"`` and ``"tau"`` -> ``(energies, weights, rock)``; see
        :func:`arca_ladders`.
    """
    below_centre_km = site.below_km + 0.5 * site.height_km
    if site.regeneration:
        return arca_ladders(neutrino_column, below_centre_km)
    theta_deg, _ = arca_zenith_grid()
    cos_theta = np.cos(np.deg2rad(theta_deg))
    energies = np.empty((ARCA_LOG10_E.size, 1))
    weights = np.empty((ARCA_LOG10_E.size, 1, neutrino_column.size))
    rock = np.ones_like(weights)
    for i, log10_e in enumerate(ARCA_LOG10_E):
        energies[i], weights[i] = regenerated_transmission(
            10.0**log10_e, neutrino_column, CROSS_SECTION, n_levels=1
        )
        rock[i] = two_medium_range_ratio(
            float((1.0 - MEAN_INELASTICITY) * energies[i, 0]),
            DEFAULT_MUON_THRESHOLD_GEV,
            cos_theta,
            below_centre_km,
        )[None, :]
    return {"mu": (energies, weights, rock), "tau": (energies, np.zeros_like(weights), rock)}


def water_projected_area_km2(
    site: WaterSite,
    theta_deg: np.ndarray,
    radius_km: float | np.ndarray,
) -> np.ndarray:
    """Convex-body projection of the site's upright cylinders.

    A cylinder of radius ``R`` and height ``h`` presents ``pi R^2`` overhead
    and ``2 R h`` at the horizon; the convex-body projection interpolates
    between them as ``pi R^2 |cos theta| + 2 R h sin theta``.

    Parameters
    ----------
    site : WaterSite
        The detector, which sets the height and the number of blocks.
    theta_deg : np.ndarray
        Zenith angle [deg].
    radius_km : float or np.ndarray
        Footprint radius of one block [km]. Broadcast against ``theta_deg``,
        so an energy-dependent radius can be passed.

    Returns
    -------
    area_km2 : np.ndarray
        Projected area of all ``site.n_blocks`` blocks [km^2].
    """
    theta = np.deg2rad(theta_deg)
    cap = np.pi * np.asarray(radius_km) ** 2 * np.abs(np.cos(theta))
    side = 2.0 * np.asarray(radius_km) * site.height_km * np.sin(theta)
    return site.n_blocks * (cap + side)


def arca_projected_area_km2(
    theta_deg: np.ndarray,
    radius_km: float | np.ndarray,
) -> np.ndarray:
    """Convex-body projection of the two ARCA230 building blocks.

    Parameters
    ----------
    theta_deg : np.ndarray
        Zenith angle [deg].
    radius_km : float or np.ndarray
        Footprint radius of one block [km].

    Returns
    -------
    area_km2 : np.ndarray
        Projected area of both blocks [km^2].
    """
    return water_projected_area_km2(ARCA230_WATER_SITE, theta_deg, radius_km)


def water_model(
    theta: np.ndarray,
    site: WaterSite,
    ladders: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    zenith_weights: np.ndarray,
    muon_column_km: np.ndarray,
    select: np.ndarray | None = None,
) -> np.ndarray:
    """Predicted sky-averaged effective area of one water site.

    The same assembly as :func:`icecube_model` with the two changes a water
    site forces: the first-passage length is truncated at the column available
    upstream of the detector, and the projected area follows the cylinder
    rather than a prism.

    Parameters
    ----------
    theta : np.ndarray, shape (5,)
        ``(eps_0, log10_e_thr, b_scale, lam, reach_km)``.
    site : WaterSite
        The detector.
    ladders : dict
        Output of :func:`water_ladders`.
    zenith_weights : np.ndarray, shape (ARCA_N_ZENITH,)
        Solid-angle weights. Zeros outside a band restrict the average to it.
    muon_column_km : np.ndarray, shape (ARCA_N_ZENITH,)
        Column available upstream of the detector [km of sea water].
    select : np.ndarray or None, optional
        Boolean mask over :data:`ARCA_LOG10_E`; see :func:`icecube_model`.

    Returns
    -------
    aeff_cm2 : np.ndarray
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
        ("tau", MEAN_Z * (1.0 - MEAN_INELASTICITY), site.f_tau * BR_TAU_TO_MU),
    )
    for flavour, muon_fraction, weight in channels:
        if weight == 0.0:
            continue
        energies, arrival, rock = ladders[flavour]
        energies, arrival, rock = energies[nodes], arrival[nodes], rock[nodes]
        muon_energy = muon_fraction * energies
        # (n_energy, n_rung, n_zenith): each rung's muon under each direction's
        # overburden. The reach follows the muon, so the radius has no zenith
        # axis, but the projected area it feeds does. Upgoing directions carry
        # the rock below the sea floor through the ladder's range ratio.
        length = (
            truncated_muon_range_km(
                muon_energy[:, :, None], muon_column_km[None, None, :], threshold, b_scale
            )
            * rock
        )
        radius = light_reach_radius_km(site.radius_km, muon_energy, reach_km, REACH_PIVOT_GEV)
        area_km2 = water_projected_area_km2(site, theta_deg[None, None, :], radius[:, :, None])
        v_det_km3 = site.n_blocks * np.pi * radius**2 * site.height_km
        volume_km3 = area_km2 * length + v_det_km3[:, :, None]
        sigma = tilted_cc(energies, lam)
        rate = n_nucleon * sigma[:, :, None] * volume_km3 * CM_PER_KM**3
        total += weight * np.average((arrival * rate).sum(axis=1), axis=1, weights=zenith_weights)
    return eps_0 * total


def arca_model(
    theta: np.ndarray,
    ladders: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    zenith_weights: np.ndarray,
    muon_column_km: np.ndarray,
    select: np.ndarray | None = None,
) -> np.ndarray:
    """Predicted ARCA230 effective area for one parameter vector.

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
        Boolean mask over :data:`ARCA_LOG10_E`; see :func:`icecube_model`.

    Returns
    -------
    aeff_cm2 : np.ndarray
        Sky-averaged effective area [cm^2].
    """
    return water_model(
        theta, ARCA230_WATER_SITE, ladders, zenith_weights, muon_column_km, select
    )


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


@dataclass
class Detector:
    """One detector's published curve, forward model, priors and results.

    Attributes
    ----------
    name : str
        Detector name, as printed in tables and figures.
    log10_e : np.ndarray
        Energy grid of the published curve, as ``log10(E_nu / GeV)``.
    observed : np.ndarray
        Published effective area on that grid [cm^2].
    mask : np.ndarray
        Boolean mask of the nodes the likelihood reads.
    predict : Callable
        ``predict(theta, select)`` -> effective area [cm^2], on the whole grid
        when ``select`` is ``None``.
    priors : dict
        Parameter name -> flat prior box ``(low, high)``.
    start : np.ndarray, shape (5,)
        Central starting point of the walkers.
    selection_level : str
        Selection the published curve is quoted at, for example ``"trigger"``.
    chain : np.ndarray or None
        Flattened posterior chain, filled in by the fit.
    best : np.ndarray or None
        Best-fit parameter vector, filled in by the fit.
    """

    name: str
    log10_e: np.ndarray
    observed: np.ndarray
    mask: np.ndarray
    predict: Callable[[np.ndarray, np.ndarray | None], np.ndarray]
    priors: dict[str, tuple[float, float]]
    start: np.ndarray
    selection_level: str
    chain: np.ndarray = field(default=None)
    best: np.ndarray = field(default=None)


def default_start() -> np.ndarray:
    """Central starting point every site's walkers begin from.

    Returns
    -------
    start : np.ndarray, shape (5,)
        ``eps_0`` at 0.7 and the other four at the value each was derived to
        take, independently of any fit.
    """
    from softpaws.comparison.likelihood import B_SCALE_MEAN

    return np.array(
        [
            0.7,
            np.log10(DEFAULT_MUON_THRESHOLD_GEV),
            B_SCALE_MEAN,
            LAMBDA_BGR18,
            REACH_EXAMPLE28_KM,
        ]
    )


def build_icecube_detector(
    data_dir: pathlib.Path, start: np.ndarray | None = None
) -> Detector:
    """IceCube's upgoing DR2 table bound to :func:`icecube_model`.

    Parameters
    ----------
    data_dir : pathlib.Path
        Root of the IceTracks-DR2 release.
    start : np.ndarray or None, optional
        Starting point of the walkers. ``None`` uses :func:`default_start`.

    Returns
    -------
    detector : Detector
        IceCube, at analysis level, with its transmission ladders bound in.
    """
    observed = icecube_upgoing(data_dir)
    mask = (IC_LOG10_E >= IC_FIT_BAND[0]) & (IC_LOG10_E <= IC_FIT_BAND[1])
    ladders = icecube_ladders()
    return Detector(
        name="IceCube",
        log10_e=IC_LOG10_E,
        observed=observed,
        mask=mask,
        predict=lambda theta, select=None: icecube_model(theta, ladders, select),
        priors=PRIORS["IceCube"],
        start=default_start() if start is None else np.asarray(start, dtype=float).copy(),
        selection_level="analysis",
    )


def build_arca_detector(start: np.ndarray | None = None) -> Detector:
    """ARCA230's trigger-level curve bound to :func:`arca_model`.

    Parameters
    ----------
    start : np.ndarray or None, optional
        Starting point of the walkers. ``None`` uses :func:`default_start`.

    Returns
    -------
    detector : Detector
        ARCA230, at trigger level, with its transmission ladders bound in.
    """
    observed = arca230_trigger()
    mask = (
        (ARCA_LOG10_E >= ARCA_FIT_BAND[0])
        & (ARCA_LOG10_E <= ARCA_FIT_BAND[1])
        & np.isfinite(observed)
    )
    zenith_weights, neutrino_column, muon_column_km = arca_columns()
    ladders = arca_ladders(neutrino_column)
    return Detector(
        name="ARCA230",
        log10_e=ARCA_LOG10_E,
        observed=observed,
        mask=mask,
        predict=lambda theta, select=None: arca_model(
            theta, ladders, zenith_weights, muon_column_km, select
        ),
        priors=PRIORS["ARCA230"],
        start=default_start() if start is None else np.asarray(start, dtype=float).copy(),
        selection_level="trigger",
    )


def build_water_detector(
    site: WaterSite,
    start: np.ndarray | None = None,
    observed: np.ndarray | None = None,
) -> Detector:
    """One water site's published sky average bound to :func:`water_model`.

    Parameters
    ----------
    site : WaterSite
        The detector.
    start : np.ndarray or None, optional
        Starting point of the walkers. ``None`` uses :func:`default_start`.
    observed : np.ndarray or None, optional
        Published curve on :data:`ARCA_LOG10_E` [cm^2]. ``None`` reads the
        site's entry in :data:`published_curve`.

    Returns
    -------
    detector : Detector
        The site, with its columns and transmission ladders bound in.
    """
    if observed is None:
        observed = published_curve[site.name]()
    mask = (
        (ARCA_LOG10_E >= site.fit_band[0])
        & (ARCA_LOG10_E <= site.fit_band[1])
        & np.isfinite(observed)
    )
    zenith_weights, neutrino_column, muon_column_km = water_columns(site)
    ladders = water_ladders(site, neutrino_column)
    priors = dict(PRIORS["ARCA230"])
    priors["reach_km"] = site.reach_prior
    return Detector(
        name=site.name,
        log10_e=ARCA_LOG10_E,
        observed=observed,
        mask=mask,
        predict=lambda theta, select=None: water_model(
            theta, site, ladders, zenith_weights, muon_column_km, select
        ),
        priors=priors,
        start=default_start() if start is None else np.asarray(start, dtype=float).copy(),
        selection_level=site.selection_level,
    )


def build_detectors(
    data_dir: pathlib.Path, start: np.ndarray | None = None
) -> list[Detector]:
    """The two reference sites, IceCube and ARCA230.

    The two curves are deliberately not at the same selection level. IceCube is
    the livetime-weighted upgoing ``nu_mu`` table of IceTracks-DR2, an
    analysis-level response, while ARCA230 is compared at trigger level, which
    carries no quality or containment cuts and so is the strongest test of a
    geometric ceiling.

    Parameters
    ----------
    data_dir : pathlib.Path
        Root of the IceTracks-DR2 release.
    start : np.ndarray or None, optional
        Starting point of the walkers. ``None`` uses :func:`default_start`.

    Returns
    -------
    detectors : list of Detector
        IceCube first, then ARCA230.
    """
    return [build_icecube_detector(data_dir, start), build_arca_detector(start)]


def build_four_detectors(
    data_dir: pathlib.Path, start: np.ndarray | None = None
) -> list[Detector]:
    """The two reference sites plus P-ONE and TRIDENT.

    Parameters
    ----------
    data_dir : pathlib.Path
        Root of the IceTracks-DR2 release.
    start : np.ndarray or None, optional
        Starting point of the walkers. ``None`` uses :func:`default_start`.

    Returns
    -------
    detectors : list of Detector
        IceCube, ARCA230, P-ONE, TRIDENT.
    """
    detectors = build_detectors(data_dir, start)
    for site in water_sites():
        detectors.append(build_water_detector(site, start))
    return detectors


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def truncation_table(
    log10_energies: tuple[float, ...] = (5.0, 6.0, 7.0, 8.0),
    fractions: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0),
    threshold_gev: float = DEFAULT_MUON_THRESHOLD_GEV,
    n_grid: int = 601,
) -> list[dict[str, float]]:
    """Closed-form truncated range against the Gil-Pelaez integral.

    The reference builds the two-moment family's kernel, where the closed form
    reads its log-loss moments from the table, so the two differ by the ~7% the
    family costs. What this checks is the gamma-law truncation: the ratio has
    to be flat in ``X / L``, which is the approximation under test.

    Parameters
    ----------
    log10_energies : tuple of float, optional
        Muon energies to compare at, as ``log10(E / GeV)``.
    fractions : tuple of float, optional
        Truncation depths, as fractions of the untruncated range.
    threshold_gev : float, optional
        Muon energy the range is measured down to [GeV].
    n_grid : int, optional
        Nodes of the depth grid the reference integral runs on.

    Returns
    -------
    rows : list of dict
        One entry per ``(log10_energy, fraction)`` with keys ``"log10_e"``,
        ``"fraction"``, ``"exact"``, ``"closed"`` and ``"ratio"``; the two
        lengths are in km.

    Notes
    -----
    Slow: seconds per energy, so this never goes inside a sampler.
    """
    from scipy.integrate import cumulative_trapezoid

    from softpaws.transport.loss_distribution import log_loss_cdf

    rows: list[dict[str, float]] = []
    for log10_e in log10_energies:
        energy = np.array([10.0**log10_e])
        b_mu = float(drift_coefficient(energy)[0])
        d_mu = float(diffusion_coefficient(energy)[0])
        deterministic = float(muon_range_km(energy, threshold_gev)[0])
        ell = np.linspace(0.0, 3.0 * deterministic, n_grid)
        cdf = log_loss_cdf(np.log(energy[0] / threshold_gev), ell, b_mu, d_mu)
        cumulative = cumulative_trapezoid(cdf, ell, initial=0.0)
        untruncated = float(cumulative[-1])
        for fraction in fractions:
            column = fraction * untruncated
            exact = float(np.interp(column, ell, cumulative))
            closed = float(
                truncated_muon_range_km(energy, np.array([column]), threshold_gev)[0]
            )
            rows.append(
                {
                    "log10_e": float(log10_e),
                    "fraction": float(fraction),
                    "exact": exact,
                    "closed": closed,
                    "ratio": closed / exact,
                }
            )
    return rows
