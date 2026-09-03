"""Example 31 -- IceCube vs KM3NeT flux constraints, effective-area convention.

Redoes example 10's Fig. 7 (arXiv:2607.13143) -- the ``(phi0, gamma)`` regions
from IceCube through-going muons against the single KM3NeT UHE event
KM3-230213A, and the tension between them -- on top of the effective-area
machinery examples 28 and 30 built and validated, instead of example 10's
placeholder instrument.

**What changes relative to example 10.** Example 10 drives both experiments
through :class:`~softpaws.response.soft_volume.SoftVolumeResponse`, a spherical
detector of guessed radius and an ad hoc Earth column (``KM_RADIUS_KM = 0.33``,
``KM_COLUMN_DEPTH_KM = 2.5``, a 1000-day livetime, a 2 pi solid angle). Every
one of those KM3NeT numbers was a placeholder. This example replaces them with
the frozen results of examples 28 and 30: for IceCube the DR2 upgoing table
reproduced by :func:`build_icecube_aeff`, and for KM3NeT the published ARCA21
**bright-track** effective area released with the event itself
(:func:`arca21_published_aeff`), on its own 4 pi convention and its actual
335-day exposure.

**Why the published ARCA21 curve and not the model.** Example 30 ports the
soft-volume construction to ARCA and calibrates a light-reach law against the
ARCA230 **trigger-level** figure. That curve is a geometric ceiling: trigger
level carries no analysis cuts, so it is the largest area the instrument ever
reports. KM3-230213A was selected by the bright-track selection, and example 30
prints the gap between the two -- an implied efficiency of 0.18-0.26 across this
event's energy window, i.e. the ceiling runs 4-6x above the response that
actually recorded the event. Driving the KM3NeT likelihood with the ceiling
lowers the flux needed for one event by that same factor and all but erases the
tension, so this example uses the published bright-track table for inference and
keeps the ceiling only as a printed contrast (:func:`aeff_ratio_report`, and the
two ``mu = 1`` normalizations in :func:`report`). The asymmetry is the point:
IceCube's curve here is calibrated to an *analysis-level* published table, so its
KM3NeT counterpart has to be analysis level too.

**Why there are no posterior contours.** With one event and a sub-unity
expectation, ``L = mu e^-mu`` is very nearly linear in ``phi0`` over any
plausible range, so a Bayesian posterior has no interior maximum in ``phi0`` and
simply piles up against whatever prior wall it is given: the median ``phi0``
tracks the upper bound of a flat prior almost exactly (0.75, 1.50, 2.52, 5.09
for walls at 1.5, 3, 5, 10). A credible contour drawn from that is a picture of
the prior box, and its apparent overlap with IceCube is set by where the box was
cut. Everything below is instead a profile-likelihood ratio, which carries no
prior at all.

**What bounds the KM3NeT constraint.** Both likelihood forms bound ``phi0`` at
every fixed ``gamma`` -- that is what a prior wall was standing in for before --
but they differ along ``gamma``. The window-only form, one event inside its own
energy interval and no statement about the rest of the spectrum, is exactly
degenerate: its ridge is the locus where ARCA21 expects one event in that
window (:func:`phi0_for_one_event`), flat in ``gamma``. Adding a second Poisson
term, that no other bright track appeared over the rest of the UHE band
(:data:`KM_BAND_LOG10_E`), tilts that ridge and puts an interior best fit at
``gamma ~ 1.8``, because a flux hard enough to make one 100 PeV event likely
would also have produced several lower-energy ones. The tilt is gentle, so the
95% band still spans the whole plotted ``gamma`` range; what the figure reads
off is the ``phi0`` offset at IceCube's ``gamma``, not a closed blob. The
two-bin form assumes the band is background free, which is why the band is a
named constant rather than being buried in the likelihood.

**KM3-230213A's energy.** The event is quoted as a muon-energy interval
(``E_mu ~ 120 PeV``, 90% CI ``[35, 380] PeV``), but the A_eff convention here is
differential in ``E_nu``. The interval is converted with the same mean
inelasticity used everywhere else in this model (``E_nu = E_mu / (1 - <y>)``),
which is a *lower bound* on the parent energy: the muon is measured after
propagation losses, not at production, so the Collaboration's own neutrino-energy
estimate is both higher and considerably wider. Using the narrow low window is
conservative in the direction that matters here -- it understates ARCA's
acceptance and so overstates the tension -- and it should be replaced by the
event's actual energy likelihood before any number here is quoted.

**What the IceCube side is.** An Asimov dataset injected at the paper's Table 2
best fit (``PHI0_TRUTH``, ``GAMMA_TRUTH``), not the observed DR2 events. So the
green region is a *forecast* over DR2's full 13.6-year good-run exposure,
centred on the injected truth by construction, rather than IceCube's published
contour -- and that normalization sits below IceCube's own 9.5-year ``nu_mu``
fit (a different, shorter dataset), which would reduce the tension. Every
tension number below inherits that caveat.

Usage
-----
    python examples/31_flux_contours_effective_area.py
    python examples/31_flux_contours_effective_area.py --data-dir /path/to/dataverse_files
    python examples/31_flux_contours_effective_area.py --no-ceiling   # skip the slow contrast curve
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import chi2, norm

from softpaws.comparison.likelihood import (
    GAMMA_RANGE,
    GAMMA_TRUTH,
    PHI0_TRUTH,
    atmospheric_template,
    poisson_log_likelihood,
)
from softpaws.data.icecube import (
    livetime_weighted_effective_area,
)
from softpaws.detectors import ARCA21, ARCA230, ICECUBE, MAX_UPSTREAM_KM
from softpaws.response.soft_volume import power_law_flux
from softpaws.transport.attenuation import (
    flavour_transmission,
    prem_column,
    regenerated_transmission,
)
from softpaws.transport.coefficients import diffusion_coefficient, drift_coefficient
from softpaws.transport.cross_section import bgr18_cross_section
from softpaws.transport.earth import neutrino_column_g_cm2, overburden_km
from softpaws.transport.earth import zenith_grid as earth_zenith_grid
from softpaws.transport.loss_distribution import log_loss_cdf
from softpaws.transport.soft_volume import (
    DEFAULT_MUON_THRESHOLD_GEV,
    light_reach_radius_km,
    muon_range_km,
    prism_projected_area_km2,
    stochastic_muon_range_km,
)
from softpaws.transport.source import MEAN_INELASTICITY, nucleon_number_density
from softpaws.transport.tau import BR_TAU_TO_MU, MEAN_Z
from softpaws.utils.constants import CM_PER_KM, RHO_WATER_G_CM3

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_DATA_DIR = _HERE.parent / "src" / "softpaws" / "data" / "dataverse_files"
_KM3NET_DIR = _HERE.parent / "src" / "softpaws" / "data" / "km3net"
_ARCA21_TABLE = _KM3NET_DIR / "arca21_aeff_brighttrack_allflavour_skyavg.csv"
_ARCA230_TRIGGER_TABLE = _KM3NET_DIR / "arca_trigger_level_eff.csv"
_DEFAULT_OUT_DIR = _HERE / "output"

CROSS_SECTION = bgr18_cross_section()

# ---------------------------------------------------------------------------
# Frozen instrument numbers. IceCube from example 28, ARCA21 from example 30 --
# both replace guesses example 10 made up (KM_RADIUS_KM = 0.33 km,
# KM_COLUMN_DEPTH_KM = 2.5 km, a 1000-day livetime, a 2 pi solid angle).
# ---------------------------------------------------------------------------

# IceCube: DR2 upgoing sky. Threshold pinned by the smearing matrix's
# 5th-percentile accepted muon energy (example 28), not fitted here.
#
# The exposure is summed from the release's own good-run lists rather than
# assumed: DR2 covers 2008-2022 across 14 seasons, and the good-run total is
# 4963.4 days = 13.589 yr, not the 9.5 yr of the separate Abbasi:2021qfz
# diffuse nu_mu measurement. Convolving the 14-season A_eff with a 9.5-year
# livetime, as earlier versions of this example did, understates the IceCube
# expectation by a factor of 1.43.
# IceCube as an upright hexagonal prism: ~1 km^2 of footprint by 1 km of
# instrumented height, which gives V_det = 1.00 km^3 exactly. IC_RADIUS_KM is the
# area-equivalent radius of the hexagon, so pi R^2 is the footprint.
IC_HEIGHT_KM = ICECUBE.height_km
IC_N_SIDES = ICECUBE.n_sides
IC_RADIUS_KM = ICECUBE.radius_km  # ~0.564 km
IC_FOOTPRINT_KM2 = float(np.pi * IC_RADIUS_KM**2)
IC_SOLID_ANGLE_SR = 2.0 * np.pi  # upgoing hemisphere only
IC_AEFF_LOG10_E = np.linspace(3.0, 8.0, 26)
IC_STATS_LOG10_E = (5.0, 7.8)  # band example 28 calibrated the reach law over
IC_FIT_EDGES = np.arange(5.0, 8.01, 0.5)  # E_nu bins for the flux fit
BKG_FRACTION = 0.3

# ARCA21: the 21-line detector that recorded KM3-230213A. The published
# bright-track table is the response used for inference; the geometry below only
# feeds the trigger-level contrast curve.
ARCA21_RADIUS_KM = ARCA21.radius_km
ARCA230_RADIUS_KM = ARCA230.radius_km
BLOCK_HEIGHT_KM = ARCA230.height_km
ARCA_DEPTH_KM = ARCA230.depth_km
N_BLOCKS_FULL = ARCA230.n_blocks
KM_LIVETIME_S = 335.0 * 86400.0  # actual ARCA21 bright-track exposure
KM_SOLID_ANGLE_SR = 4.0 * np.pi  # the released table is sky-averaged, not upgoing
ARCA_AEFF_LOG10_E = np.arange(4.0, 10.01, 0.2)
ARCA_FIT_TOP_LOG10_E = 7.5  # digitized trigger curve saturates past this
MAX_SEA_PATH_KM = MAX_UPSTREAM_KM
RHO_SEA_G_CM3 = RHO_WATER_G_CM3
N_ZENITH = 90
N_ELL = 401

# KM3-230213A: E_mu ~ 120 PeV, 90% interval [35, 380] PeV, converted to E_nu
# through the mean inelasticity (see the module docstring's caveat -- this is a
# lower bound on the parent energy, not the Collaboration's own estimate).
KM_EVENT_MU_PEV = (35.0, 380.0)
KM_EDGES = np.log10(np.array(KM_EVENT_MU_PEV) * 1.0e6 / (1.0 - MEAN_INELASTICITY))

# Band over which "no other bright track was recorded" is asserted, closing the
# KM3NeT region (see the docstring). Taken to be background free, which is the
# one assumption the two-bin likelihood adds over the window-only form.
KM_BAND_LOG10_E = np.array([7.0, 10.0])

# Upper edges of the flat phi0 prior, used *only* for the Bayes-factor
# sensitivity table. Nothing in the profile-likelihood results depends on them,
# which is the point of quoting both.
PHI0_PRIOR_TOPS = (1.5, 3.0, 5.0)

# ---------------------------------------------------------------------------
# Li, Machado, Naredo-Tuero and Schwemberger (arXiv:2502.04508), whose 3.5 sigma
# is the published number this example's 1.9 sigma has to be reconciled against.
# Their stated assumptions, for the ingredient ladder of `tension_ladder`:
#
#   * the IceCube side is a Gaussian *prior* on the flux parameters at the
#     collaboration's combined fit, not a forward-modelled event distribution;
#   * the event enters through P(N_hit | E_nu) integrated over all E_nu, whose
#     90% interval is 23-2400 PeV under an E^-2 prior and 4-760 PeV under the
#     E^-2.52 diffuse prior; we can only impose a window, so both are tried;
#   * the ARCA21 exposure is quoted as 288 days;
#   * no statement is made about the rest of the UHE band, i.e. no second
#     Poisson term. They note that adding KM3NeT's own non-observation, with a
#     flat energy probability across the window, returns 1.9 sigma.
# ---------------------------------------------------------------------------
LI_SIGMA_PUBLISHED = 3.5
LI_EDGES = np.log10(np.array([23.0, 2400.0]) * 1.0e6)
LI_EDGES_DIFFUSE_PRIOR = np.log10(np.array([4.0, 760.0]) * 1.0e6)
LI_LIVETIME_S = 288.0 * 86400.0
# IceCube combined fit as they quote it: phi0 = 1.83 +0.13 -0.16, gamma = 2.52 +- 0.04.
LI_PHI0, LI_PHI0_SIGMA = 1.83, (0.16, 0.13)
LI_GAMMA, LI_GAMMA_SIGMA = 2.52, 0.04

# Profile-likelihood contour levels for two free parameters.
DELTA_LNL_LEVELS = 0.5 * chi2.isf([0.32, 0.05], df=2)  # ~1.15, ~3.00

# Keyed by detector, not collaboration: the orange region is ARCA21's own
# likelihood, and the one-event locus drawn with it is ARCA21's too.
EXP_COLOR = {"IceCube": "#1b9e77", "ARCA21": "#d95f02"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=pathlib.Path, default=_DEFAULT_DATA_DIR,
                        help="Root of the DR2 data directory (must contain 'irfs/' and 'uptime/').")
    parser.add_argument("--out", type=pathlib.Path,
                        default=_DEFAULT_OUT_DIR / "31_flux_contours_effective_area.pdf")
    parser.add_argument("--threshold", type=float, default=DEFAULT_MUON_THRESHOLD_GEV,
                        help="Muon selection threshold [GeV], shared by both instruments.")
    parser.add_argument("--no-ceiling", action="store_true",
                        help="Skip the trigger-level contrast curve, which is the slow part.")
    parser.add_argument("--n-gamma", type=int, default=161)
    parser.add_argument("--n-phi0", type=int, default=241)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# IceCube A_eff(E_nu) -- example 28's construction, trimmed to its final curve.
# ---------------------------------------------------------------------------


def icecube_upgoing(data_dir: pathlib.Path):
    """Livetime-weighted DR2 effective area over the upgoing sky [cm^2].

    Delegates to :func:`softpaws.data.icecube.livetime_weighted_effective_area`
    on ``IC_AEFF_LOG10_E``.
    """
    return livetime_weighted_effective_area(data_dir, IC_AEFF_LOG10_E)


def ic_upgoing_columns() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PREM columns, solid-angle weights and zenith cosines, upgoing hemisphere.

    IceCube sits at the Pole, so a source at declination ``dec`` arrives at
    ``|cos theta_z| = sin(dec)``, which is the third return value.
    """
    dec_deg = np.linspace(0.5, 89.5, 60)
    columns = np.array([prem_column(float(d)) for d in dec_deg])
    dec_rad = np.deg2rad(dec_deg)
    return columns, np.cos(dec_rad), np.sin(dec_rad)


def ic_target_volume_cm3(
    length_km: np.ndarray,
    radius_km: float | np.ndarray = IC_RADIUS_KM,
    cos_theta: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Prism target volume, projected column plus detector [cm^3].

    An array ``cos_theta`` is appended as a trailing axis, since the projected
    area depends on the arrival direction where the instrumented volume does not.
    """
    radius = np.asarray(radius_km, dtype=float)
    length = np.asarray(length_km, dtype=float)
    zenith = np.asarray(cos_theta, dtype=float)
    if zenith.ndim:
        radius = radius[..., None]
        length = length[..., None]
    proj_area = prism_projected_area_km2(zenith, radius, IC_HEIGHT_KM, IC_N_SIDES)
    v_det = np.pi * radius**2 * IC_HEIGHT_KM
    return (proj_area * length + v_det) * CM_PER_KM**3


def ic_mean_target_volume_cm3(
    length_km: np.ndarray, radius_km: float | np.ndarray = IC_RADIUS_KM
) -> np.ndarray:
    """Target volume averaged over the upgoing hemisphere [cm^3]."""
    _, weights, cos_theta = ic_upgoing_columns()
    return np.average(
        ic_target_volume_cm3(length_km, radius_km, cos_theta), axis=-1, weights=weights
    )


def ic_required_radius_km(ratio: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Footprint radius that would scale the target volume by ``ratio`` at each energy."""
    out = np.full(np.shape(ratio), np.nan)
    for i, (r, length) in enumerate(zip(np.atleast_1d(ratio), lengths)):
        if not np.isfinite(r) or r <= 0.0:
            continue
        target = r * float(ic_mean_target_volume_cm3(np.array([length]))[0])
        out[i] = brentq(
            lambda x: float(ic_mean_target_volume_cm3(np.array([length]), x)[0]) - target,
            1.0e-4,
            50.0,
        )
    return out


def fit_reach_law(
    log10_e: np.ndarray, required_radius_km: np.ndarray, radius_km: float
) -> tuple[float, float]:
    """Least-squares reach law through the radii a published curve demands.

    Shared between IceCube and ARCA: both invert a required-radius sequence
    that is close to linear in ``ln E`` for the same
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


def ic_effective_area_regenerated(
    length_km: np.ndarray,
    threshold_gev: float,
    reach_km: float | None = None,
    pivot_gev: float = 1.0e6,
) -> np.ndarray:
    """Direct nu_mu channel, NC regeneration kept, upgoing-averaged [cm^2]."""
    energy = 10.0**IC_AEFF_LOG10_E
    columns, weights, cos_theta = ic_upgoing_columns()
    n_nucleon = nucleon_number_density()
    out = np.empty(energy.size)
    for i, e_nu in enumerate(energy):
        rung_energy, rung_weight = regenerated_transmission(float(e_nu), columns, CROSS_SECTION)
        rung_length = np.interp(
            np.log10(rung_energy), IC_AEFF_LOG10_E, length_km, left=0.0, right=length_km[-1]
        )
        rung_length[(1.0 - MEAN_INELASTICITY) * rung_energy <= threshold_gev] = 0.0
        rung_radius = (
            IC_RADIUS_KM
            if reach_km is None
            else light_reach_radius_km(
                IC_RADIUS_KM, (1.0 - MEAN_INELASTICITY) * rung_energy, reach_km, pivot_gev
            )
        )
        rung_volume = ic_target_volume_cm3(rung_length, rung_radius, cos_theta)
        rung_rate = n_nucleon * CROSS_SECTION.cc(rung_energy)[:, None] * rung_volume
        per_dec = (rung_weight * rung_rate).sum(axis=0)
        out[i] = np.average(per_dec, weights=weights)
    return out


def ic_effective_area_tau_channel(length_km: np.ndarray, threshold_gev: float) -> np.ndarray:
    """nu_tau -> tau -> mu channel, upgoing-averaged, static footprint [cm^2]."""
    energy = 10.0**IC_AEFF_LOG10_E
    columns, weights, cos_theta = ic_upgoing_columns()
    n_nucleon = nucleon_number_density()
    muon_fraction = MEAN_Z * (1.0 - MEAN_INELASTICITY)
    out = np.empty(energy.size)
    for i, e_nu in enumerate(energy):
        rung_energy, rung_weight = flavour_transmission(
            float(e_nu), columns, CROSS_SECTION, flavour="tau"
        )
        rung_length = np.interp(
            np.log10(rung_energy * muon_fraction / (1.0 - MEAN_INELASTICITY)),
            IC_AEFF_LOG10_E, length_km, left=0.0, right=length_km[-1],
        )
        rung_length[muon_fraction * rung_energy <= threshold_gev] = 0.0
        rung_rate = (
            n_nucleon * CROSS_SECTION.cc(rung_energy)[:, None]
            * ic_target_volume_cm3(rung_length, IC_RADIUS_KM, cos_theta) * BR_TAU_TO_MU
        )
        per_dec = (rung_weight * rung_rate).sum(axis=0)
        out[i] = np.average(per_dec, weights=weights)
    return out


def build_icecube_aeff(
    data_dir: pathlib.Path, threshold_gev: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """IceCube nu_mu + nu_tau->mu effective area with the fitted reach law.

    Calibrates the reach law against the published DR2 upgoing table, exactly
    as example 28's ``"+ fitted reach"`` curve (0.007 dex rms residual there).
    That table is analysis level, which is what obliges the KM3NeT side of this
    example to use an analysis-level curve too.

    Returns
    -------
    log10_e : np.ndarray
        ``log10(E_nu / GeV)`` grid.
    aeff : np.ndarray
        Effective area [cm^2] on that grid.
    livetime_s : float
        Total DR2 good-run livetime summed over all seasons [s].
    """
    published, livetime_s = icecube_upgoing(data_dir)
    energy_mu = (1.0 - MEAN_INELASTICITY) * 10.0**IC_AEFF_LOG10_E
    length = stochastic_muon_range_km(energy_mu, threshold_gev)
    base = ic_effective_area_regenerated(length, threshold_gev) + ic_effective_area_tau_channel(
        length, threshold_gev
    )

    band = (IC_AEFF_LOG10_E >= IC_STATS_LOG10_E[0]) & (IC_AEFF_LOG10_E <= IC_STATS_LOG10_E[1])
    ratio = np.full_like(IC_AEFF_LOG10_E, np.nan)
    ratio[band] = (published / base)[band]
    needed = ic_required_radius_km(ratio, length)
    reach_km, pivot_gev = fit_reach_law(IC_AEFF_LOG10_E, needed, IC_RADIUS_KM)

    curve = ic_effective_area_regenerated(
        length, threshold_gev, reach_km=reach_km, pivot_gev=pivot_gev
    ) + ic_effective_area_tau_channel(length, threshold_gev)
    residual = np.log10(published[band] / curve[band])
    print(
        f"  IceCube: reach = {reach_km * 1e3:.1f} m/e-fold, "
        f"residual {np.std(residual):.3f} dex rms against the published table"
    )
    print(
        f"  IceCube: DR2 good-run livetime = {livetime_s / 86400.0:.1f} d "
        f"= {livetime_s / (365.25 * 86400.0):.3f} yr, summed over the release"
    )
    return IC_AEFF_LOG10_E, curve, livetime_s


# ---------------------------------------------------------------------------
# ARCA21 A_eff(E_nu). The published bright-track table drives the inference; the
# trigger-level model below is built only to show how far above it sits.
# ---------------------------------------------------------------------------


def arca21_published_aeff() -> tuple[np.ndarray, np.ndarray]:
    """Released ARCA21 bright-track, all-flavour, sky-averaged area [cm^2].

    The selection that recorded KM3-230213A, from the event's own data release
    (KM3NeT Collaboration, Nature 638 (2025) 376). Its stated convention is
    ``N = 4 pi T Integral A_eff(E) Phi_per-flavour(E) dE``, which is what
    :data:`KM_SOLID_ANGLE_SR` and :func:`power_law_flux` implement.

    Returns
    -------
    log10_e : np.ndarray
        ``log10(E_nu / GeV)`` of the tabulated points with positive area.
    aeff : np.ndarray
        Effective area [cm^2].
    """
    table = np.genfromtxt(_ARCA21_TABLE, delimiter=",", comments="#")
    positive = table[:, 1] > 0.0
    return np.log10(table[positive, 0]), table[positive, 1]


def zenith_grid(
    cos_range: tuple[float, float] = (-1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Zenith samples and solid-angle weights; see :func:`softpaws.transport.earth.zenith_grid`."""
    return earth_zenith_grid(N_ZENITH, cos_range)


def projected_area_km2(
    theta_deg: np.ndarray, radius_km: float | np.ndarray, n_blocks: int
) -> np.ndarray:
    """Projected area of upright cylindrical building blocks [km^2]."""
    theta = np.deg2rad(theta_deg)
    cap = np.pi * np.asarray(radius_km) ** 2 * np.abs(np.cos(theta))
    side = 2.0 * np.asarray(radius_km) * BLOCK_HEIGHT_KM * np.sin(theta)
    return n_blocks * (cap + side)


def upstream_column_km(theta_deg: np.ndarray, depth_km: float) -> np.ndarray:
    """Medium available upstream [km]; see :func:`softpaws.transport.earth.overburden_km`."""
    return overburden_km(np.cos(np.deg2rad(theta_deg)), depth_km, MAX_SEA_PATH_KM)


def earth_column_g_cm2(theta_deg: np.ndarray, depth_km: float) -> np.ndarray:
    """Neutrino column [g cm^-2]; see :func:`softpaws.transport.earth.neutrino_column_g_cm2`."""
    return neutrino_column_g_cm2(
        np.cos(np.deg2rad(theta_deg)), depth_km, RHO_SEA_G_CM3, MAX_SEA_PATH_KM
    )


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

    out = np.empty(ARCA_AEFF_LOG10_E.size)
    for i, e_nu in enumerate(10.0**ARCA_AEFF_LOG10_E):
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


def arca230_published_trigger(log10_e: np.ndarray) -> np.ndarray:
    """Digitized full-ARCA nu_mu trigger-level effective area [cm^2]."""
    table = np.genfromtxt(_ARCA230_TRIGGER_TABLE, delimiter=",", comments="#")
    table = table[np.argsort(table[:, 0])]
    log10_table = np.log10(table[:, 0])
    return 1.0e4 * 10.0 ** np.interp(
        log10_e, log10_table, np.log10(table[:, 1]), left=np.nan, right=np.nan
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


def build_arca_ceiling_aeff(
    threshold_gev: float, depth_km: float
) -> tuple[np.ndarray, np.ndarray]:
    """ARCA21 trigger-level *ceiling*, example 30's ``"ARCA21, fitted reach"``.

    Calibrates the reach law on the full two-block ARCA230 against its published
    **trigger-level** curve, then transfers it unchanged to ARCA21. Trigger level
    carries no analysis cuts, so this is the largest area the instrument reports
    and it sits 4-6x above the bright-track selection that recorded
    KM3-230213A -- example 30 prints the ratio directly. It is built here only to
    be reported alongside the real constraint, never to infer with, and
    ``--no-ceiling`` skips it.

    Returns
    -------
    log10_e : np.ndarray
        ``log10(E_nu / GeV)`` grid.
    aeff : np.ndarray
        Effective area [cm^2] on that grid.
    """
    table_log10_e = np.arange(2.0, 10.01, 0.2)
    ell_km, cumulative_km = build_length_table(10.0**table_log10_e, threshold_gev)
    kwargs = dict(
        threshold_gev=threshold_gev, depth_km=depth_km,
        grid_log10_e=table_log10_e, ell_km=ell_km, cumulative_km=cumulative_km,
    )

    a230_mu = arca_effective_area(ARCA230_RADIUS_KM, N_BLOCKS_FULL, flavour="mu", **kwargs)
    a230_tau = arca_effective_area(ARCA230_RADIUS_KM, N_BLOCKS_FULL, flavour="tau", **kwargs)
    trigger = arca230_published_trigger(ARCA_AEFF_LOG10_E)
    band = (ARCA_AEFF_LOG10_E >= 4.0) & (ARCA_AEFF_LOG10_E <= ARCA_FIT_TOP_LOG10_E)
    required = np.full(ARCA_AEFF_LOG10_E.size, np.nan)
    required[band] = required_footprint_radius_km(
        (trigger / (a230_mu + a230_tau))[band], ARCA230_RADIUS_KM, N_BLOCKS_FULL
    )
    reach_km, pivot_gev = fit_reach_law(ARCA_AEFF_LOG10_E, required, ARCA230_RADIUS_KM)

    a21_mu = arca_effective_area(
        ARCA21_RADIUS_KM, 1, flavour="mu", reach_km=reach_km, pivot_gev=pivot_gev, **kwargs
    )
    a21_tau = arca_effective_area(
        ARCA21_RADIUS_KM, 1, flavour="tau", reach_km=reach_km, pivot_gev=pivot_gev, **kwargs
    )
    print(
        f"  ARCA230 -> ARCA21 ceiling: reach = {reach_km * 1e3:.1f} m/e-fold "
        "(calibrated on ARCA230 trigger level, transferred)"
    )
    return ARCA_AEFF_LOG10_E, a21_mu + a21_tau


def aeff_ratio_report(
    ceiling: tuple[np.ndarray, np.ndarray], published: tuple[np.ndarray, np.ndarray]
) -> None:
    """Print the trigger-level ceiling against the bright-track table."""
    print("\n  ARCA21 response across KM3-230213A's window "
          f"(log10 E_nu = {KM_EDGES[0]:.2f} to {KM_EDGES[1]:.2f}):")
    print(f"    {'log10(E/GeV)':>13} {'bright track':>14} {'trigger ceiling':>17} {'ratio':>8}")
    for log10_e in (7.0, 7.5, 8.0, 8.5, 9.0):
        pub = float(_log_interp_aeff(published[0], published[1], np.array([log10_e]))[0])
        ceil = float(_log_interp_aeff(ceiling[0], ceiling[1], np.array([log10_e]))[0])
        flag = "  <-" if KM_EDGES[0] <= log10_e <= KM_EDGES[1] else ""
        print(f"    {log10_e:13.1f} {pub:14.3g} {ceil:17.3g} {pub / ceil:8.3f}{flag}")
    print("    Inference uses the bright-track column; the ceiling is drawn for contrast only.")


# ---------------------------------------------------------------------------
# Expected counts
# ---------------------------------------------------------------------------


def _log_interp_aeff(
    log10_e_grid: np.ndarray, aeff: np.ndarray, log10_e_query: np.ndarray
) -> np.ndarray:
    """Log-log interpolation of a positive, smooth effective-area curve."""
    log_aeff = np.log(np.clip(aeff, 1.0e-300, None))
    interp = np.interp(log10_e_query, log10_e_grid, log_aeff, left=-700.0, right=log_aeff[-1])
    return np.exp(interp)


def expected_counts(
    log10_e_edges: np.ndarray,
    aeff_log10_e: np.ndarray,
    aeff: np.ndarray,
    phi0: float,
    gamma: float,
    livetime_s: float,
    solid_angle_sr: float,
    n_sub: int = 96,
) -> np.ndarray:
    """Poisson-mean counts per bin, ``T Omega Integral_bin dE phi(E) A_eff(E)``.

    Parameters
    ----------
    log10_e_edges : np.ndarray, shape (n_bins + 1,)
        ``log10(E_nu / GeV)`` bin edges.
    aeff_log10_e, aeff : np.ndarray
        Tabulated effective area [cm^2] and its energy grid.
    phi0, gamma : float
        Diffuse-flux parameters of :func:`~softpaws.response.soft_volume.power_law_flux`.
    livetime_s : float
        Exposure time [s].
    solid_angle_sr : float
        Solid angle of the analysis region [sr].
    n_sub : int, optional
        Log-spaced quadrature points per bin. The default is converged to better
        than 0.1% on the widest bin used here; 24 was 0.5% high.

    Returns
    -------
    counts : np.ndarray, shape (n_bins,)
        Expected counts per bin.
    """
    edges = np.asarray(log10_e_edges, dtype=float)
    counts = np.empty(edges.size - 1)
    for i in range(edges.size - 1):
        log10_e = np.linspace(edges[i], edges[i + 1], n_sub)
        energy = 10.0**log10_e
        flux = power_law_flux(energy, phi0, gamma)
        aeff_here = _log_interp_aeff(aeff_log10_e, aeff, log10_e)
        counts[i] = np.trapezoid(flux * aeff_here, energy) * livetime_s * solid_angle_sr
    return counts


def asimov_ic(
    aeff_log10_e: np.ndarray, aeff: np.ndarray, livetime_s: float,
    phi0: float = PHI0_TRUTH, gamma: float = GAMMA_TRUTH,
    bkg_total: float = 0.0, bkg_slope: float = 3.7,
) -> np.ndarray:
    """Expected (Asimov) IceCube counts from an injected truth flux, plus background."""
    counts = expected_counts(
        IC_FIT_EDGES, aeff_log10_e, aeff, phi0, gamma, livetime_s, IC_SOLID_ANGLE_SR
    )
    if bkg_total > 0.0:
        counts = counts + bkg_total * atmospheric_template(IC_FIT_EDGES, bkg_slope)
    return counts


# ---------------------------------------------------------------------------
# Profile likelihoods.
#
# Both instruments' expected counts are strictly proportional to ``phi0`` at
# fixed ``gamma``, so a single pass over a ``gamma`` grid fixes the whole
# two-dimensional surface exactly. That is what makes an MCMC unnecessary here,
# and with it the prior box that was setting the KM3NeT contour.
# ---------------------------------------------------------------------------


def km_rate_coefficients(
    gamma_grid: np.ndarray,
    aeff_log10_e: np.ndarray,
    aeff: np.ndarray,
    edges: np.ndarray = KM_EDGES,
    livetime_s: float = KM_LIVETIME_S,
) -> tuple[np.ndarray, np.ndarray]:
    """ARCA21 expected counts at ``phi0 = 1``, in the event window and the band.

    Parameters
    ----------
    gamma_grid : np.ndarray
        Spectral indices to tabulate.
    aeff_log10_e, aeff : np.ndarray
        Effective area [cm^2] and its ``log10(E_nu / GeV)`` grid.
    edges : np.ndarray, optional
        ``log10(E_nu / GeV)`` edges of the event window. Defaults to
        :data:`KM_EDGES`; :func:`tension_ladder` passes Ref. [Li et al.]'s
        interval instead.
    livetime_s : float, optional
        ARCA21 exposure [s]. Defaults to :data:`KM_LIVETIME_S`.

    Returns
    -------
    a_window : np.ndarray
        Counts per unit ``phi0`` inside ``edges``.
    a_band : np.ndarray
        Counts per unit ``phi0`` over :data:`KM_BAND_LOG10_E`, the range across
        which no other bright track is asserted.
    """
    kwargs = dict(livetime_s=livetime_s, solid_angle_sr=KM_SOLID_ANGLE_SR)
    a_window = np.array([
        expected_counts(edges, aeff_log10_e, aeff, 1.0, g, **kwargs).sum() for g in gamma_grid
    ])
    a_band = np.array([
        expected_counts(KM_BAND_LOG10_E, aeff_log10_e, aeff, 1.0, g, n_sub=240, **kwargs).sum()
        for g in gamma_grid
    ])
    return a_window, a_band


def km_log_likelihood(
    phi0: np.ndarray, a_window: np.ndarray, a_band: np.ndarray, extended: bool
) -> np.ndarray:
    """Log-likelihood of one bright track at KM3-230213A's energy.

    Parameters
    ----------
    phi0 : np.ndarray, shape (n_phi0, 1)
        Flux normalization, broadcast against the ``gamma`` axis.
    a_window, a_band : np.ndarray, shape (n_gamma,)
        Output of :func:`km_rate_coefficients`.
    extended : bool
        ``False`` gives the window-only Poisson term ``ln mu_win - mu_win``,
        which says nothing about the rest of the spectrum. ``True`` adds the
        second bin -- zero events over the rest of :data:`KM_BAND_LOG10_E` --
        giving ``ln mu_win - mu_band``. Only the extended form has an interior
        maximum in ``gamma``; the window-only form is degenerate along the
        ``mu_win = 1`` ridge (:func:`phi0_for_one_event`).

    Returns
    -------
    log_l : np.ndarray, shape (n_phi0, n_gamma)
        Log-likelihood, up to the parameter-independent ``ln k!``.
    """
    mu_window = phi0 * a_window
    penalty = phi0 * (a_band if extended else a_window)
    with np.errstate(divide="ignore"):
        return np.log(mu_window) - penalty


def phi0_for_one_event(a_window: np.ndarray) -> np.ndarray:
    """Flux at which ARCA21 expects exactly one event in the event's window.

    The prior-free statement this dataset supports: below this locus the event
    required an upward fluctuation. It is also the ridge of the window-only
    likelihood, whose maximum sits at ``mu_win = 1`` for every ``gamma``.
    """
    return 1.0 / a_window


def ic_signal_coefficients(
    gamma_grid: np.ndarray, aeff_log10_e: np.ndarray, aeff: np.ndarray, livetime_s: float
) -> np.ndarray:
    """IceCube per-bin expected counts at ``phi0 = 1``, one row per ``gamma``."""
    return np.array([
        expected_counts(
            IC_FIT_EDGES, aeff_log10_e, aeff, 1.0, g, livetime_s, IC_SOLID_ANGLE_SR
        )
        for g in gamma_grid
    ])


def ic_log_likelihood(
    phi0_grid: np.ndarray, signal_unit: np.ndarray, observed: np.ndarray, template: np.ndarray
) -> np.ndarray:
    """IceCube log-likelihood, profiled over the free atmospheric normalization.

    Parameters
    ----------
    phi0_grid : np.ndarray, shape (n_phi0,)
        Flux normalizations to evaluate.
    signal_unit : np.ndarray, shape (n_gamma, n_bins)
        Per-bin counts at ``phi0 = 1`` (:func:`ic_signal_coefficients`).
    observed : np.ndarray, shape (n_bins,)
        Observed counts.
    template : np.ndarray, shape (n_bins,)
        Unit-sum atmospheric background shape.

    Returns
    -------
    log_l : np.ndarray, shape (n_phi0, n_gamma)
        Profile log-likelihood, maximized over ``bkg_norm >= 0`` at each point.
    """
    out = np.empty((phi0_grid.size, signal_unit.shape[0]))
    for j in range(signal_unit.shape[0]):
        for i, phi0 in enumerate(phi0_grid):
            signal = phi0 * signal_unit[j]

            def negative(bkg_norm: float, signal: np.ndarray = signal) -> float:
                return -poisson_log_likelihood(observed, signal + bkg_norm * template)

            result = minimize_scalar(
                negative, bounds=(0.0, 10.0 * float(observed.sum()) + 1.0), method="bounded"
            )
            out[i, j] = -result.fun
    return out


def compatibility(
    ic_log_l: np.ndarray, km_log_l: np.ndarray
) -> tuple[float, float, float]:
    """Likelihood-ratio test that both datasets share one ``(phi0, gamma)``.

    Compares the two datasets fitted independently against the two fitted
    jointly, on the common grid. The test statistic has two degrees of freedom,
    the parameters the joint fit is forced to share.

    Parameters
    ----------
    ic_log_l, km_log_l : np.ndarray, shape (n_phi0, n_gamma)
        Profile log-likelihood surfaces on the same grid.

    Returns
    -------
    ts : float
        ``2 (max ln L_IC + max ln L_KM - max (ln L_IC + ln L_KM))``.
    p_value : float
        Corresponding chi-square tail probability.
    sigma : float
        The same probability as a two-sided Gaussian significance.
    """
    ts = 2.0 * (np.max(ic_log_l) + np.max(km_log_l) - np.max(ic_log_l + km_log_l))
    p_value = float(chi2.sf(ts, df=2))
    return float(ts), p_value, float(norm.isf(0.5 * p_value))


def flux_prior_surface(phi0_grid: np.ndarray, gamma_grid: np.ndarray) -> np.ndarray:
    """Gaussian constraint on ``(phi0, gamma)``, the IceCube side of Ref. Li et al.

    Their IceCube input is not a forward-modelled event distribution but a
    Gaussian prior at the collaboration's combined fit, so the ladder needs the
    same object on the same grid. The asymmetric ``phi0`` error is carried as a
    two-sided Gaussian.

    Parameters
    ----------
    phi0_grid, gamma_grid : np.ndarray
        Grid axes shared with the other likelihood surfaces.

    Returns
    -------
    log_l : np.ndarray, shape (n_phi0, n_gamma)
        Log-likelihood, zero at the quoted central values.
    """
    sigma_lo, sigma_hi = LI_PHI0_SIGMA
    delta_phi0 = phi0_grid[:, None] - LI_PHI0
    sigma_phi0 = np.where(delta_phi0 < 0.0, sigma_lo, sigma_hi)
    delta_gamma = gamma_grid[None, :] - LI_GAMMA
    return -0.5 * (delta_phi0 / sigma_phi0) ** 2 - 0.5 * (delta_gamma / LI_GAMMA_SIGMA) ** 2


def tension_ladder(
    phi0_grid: np.ndarray,
    gamma_grid: np.ndarray,
    km_log10_e: np.ndarray,
    km_aeff: np.ndarray,
    ic_log_l: np.ndarray,
) -> None:
    """Walk from Ref. Li et al.'s configuration to ours, one ingredient at a time.

    Every rung is scored with the same profile-likelihood ratio on two degrees
    of freedom, so the differences between rungs are attributable to the
    ingredient that changed and not to the statistic. The published $3.5\\sigma$
    is a Bayes factor over a marginalized posterior and is printed for reference
    only; the gap between it and the first rung is the part of their analysis
    this machinery cannot reproduce, which is the event's energy likelihood
    ``P(N_hit | E_nu)`` in place of a window.

    The decomposition is order dependent, since the ingredients are not
    independent. This is one stated path.

    Parameters
    ----------
    phi0_grid, gamma_grid : np.ndarray
        Grid axes shared with the other likelihood surfaces.
    km_log10_e, km_aeff : np.ndarray
        Published ARCA21 bright-track effective area [cm^2] and its grid.
    ic_log_l : np.ndarray, shape (n_phi0, n_gamma)
        Our IceCube profile likelihood, the last ingredient to be swapped in.
    """
    li_ic = flux_prior_surface(phi0_grid, gamma_grid)
    cache: dict[tuple[float, float, float], tuple[np.ndarray, np.ndarray]] = {}

    def sigma_for(edges: np.ndarray, livetime_s: float, extended: bool, ic: np.ndarray) -> float:
        key = (float(edges[0]), float(edges[-1]), livetime_s)
        if key not in cache:
            cache[key] = km_rate_coefficients(gamma_grid, km_log10_e, km_aeff, edges, livetime_s)
        a_window, a_band = cache[key]
        km = km_log_likelihood(phi0_grid[:, None], a_window, a_band, extended=extended)
        return compatibility(ic, km)[2]

    config = dict(edges=LI_EDGES, livetime_s=LI_LIVETIME_S, extended=False, ic=li_ic)
    rows = [
        ("their configuration, our statistic", {}),
        ("+ our event window, E_nu = E_mu / (1 - <y_w>)", dict(edges=KM_EDGES)),
        ("+ the ARCA21 bright-track exposure, 335 d", dict(livetime_s=KM_LIVETIME_S)),
        ("+ KM3NeT's non-observation above the window", dict(extended=True)),
        ("+ our IceCube forward model, DR2 Asimov", dict(ic=ic_log_l)),
    ]

    print("\n  Ingredient ladder from Ref. [Li et al.] to this example:")
    print(f"    {'their published Bayes factor':<46} {LI_SIGMA_PUBLISHED:5.2f} sigma")
    previous = None
    for label, change in rows:
        config.update(change)
        sigma = sigma_for(**config)
        step = "     --" if previous is None else f"  {sigma - previous:+5.2f}"
        print(f"    {label:<46} {sigma:5.2f} sigma {step}")
        previous = sigma

    alt = sigma_for(LI_EDGES_DIFFUSE_PRIOR, LI_LIVETIME_S, False, li_ic)
    print(f"    (first rung with their E^-2.52 energy interval: {alt:.2f} sigma)")


def li_cross_checks(
    km_log10_e: np.ndarray,
    km_aeff: np.ndarray,
    ic_log10_e: np.ndarray,
    ic_aeff: np.ndarray,
    ic_livetime_s: float,
) -> None:
    """Reproduce the two event counts Ref. [Li et al.] quotes, as a response check.

    They quote $0.005$ events at KM3NeT over $72$-$2600$ PeV for IceCube's
    nominal flux, and $75$ events at IceCube over the same interval for the
    normalization their one event implies. Both are direct statements about the
    two responses, so recomputing them with ours says whether the responses
    agree before any statistic is applied.

    Parameters
    ----------
    km_log10_e, km_aeff : np.ndarray
        Published ARCA21 bright-track effective area [cm^2] and its grid.
    ic_log10_e, ic_aeff : np.ndarray
        Our IceCube effective area [cm^2] and its grid.
    ic_livetime_s : float
        DR2 good-run livetime [s].
    """
    edges = np.log10(np.array([72.0, 2600.0]) * 1.0e6)
    km_counts = expected_counts(
        edges, km_log10_e, km_aeff, LI_PHI0, LI_GAMMA, LI_LIVETIME_S, KM_SOLID_ANGLE_SR
    ).sum()
    km_unit = expected_counts(
        edges, km_log10_e, km_aeff, 1.0, LI_GAMMA, LI_LIVETIME_S, KM_SOLID_ANGLE_SR
    ).sum()
    ic_counts = expected_counts(
        edges, ic_log10_e, ic_aeff, 1.0 / km_unit, LI_GAMMA, ic_livetime_s, IC_SOLID_ANGLE_SR
    ).sum()
    print("\n  Cross-check against the two counts Ref. [Li et al.] quotes over 72-2600 PeV:")
    print(f"    ARCA21 at their nominal flux: {km_counts:.4f} events   (they quote 0.005)")
    print(f"    phi0 for one ARCA21 event at gamma = {LI_GAMMA}: {1.0 / km_unit:.1f}")
    print(f"    IceCube at that phi0:         {ic_counts:.1f} events     (they quote 75)")
    print("    the IceCube number holds A_eff flat above 10^8 GeV, so it is a lower bound")


def bayes_factor(
    phi0_grid: np.ndarray, gamma_grid: np.ndarray,
    km_log_l: np.ndarray, ic_log_l: np.ndarray, phi0_top: float,
) -> float:
    """BF = <L_KM>_flat-prior / <L_KM>_IceCube-weighted (Eq. 4.5), by quadrature.

    Kept because the paper quotes it, but it is reported alongside its prior box
    rather than on its own: the numerator averages over a flat
    ``(phi0, gamma)`` box, so the answer moves with where that box is cut.
    :func:`compatibility` is the prior-free statement.

    Parameters
    ----------
    phi0_grid, gamma_grid : np.ndarray
        Grid axes of the two likelihood surfaces.
    km_log_l, ic_log_l : np.ndarray, shape (n_phi0, n_gamma)
        Profile log-likelihood surfaces.
    phi0_top : float
        Upper edge of the flat ``phi0`` prior.

    Returns
    -------
    bf : float
        Ratio of the two averages.
    """
    inside = (phi0_grid[:, None] <= phi0_top) & np.ones_like(gamma_grid, dtype=bool)
    km_l = np.where(inside, np.exp(km_log_l), 0.0)
    weight = np.where(inside, np.exp(ic_log_l - np.max(ic_log_l)), 0.0)

    def integrate(surface: np.ndarray) -> float:
        return float(np.trapezoid(np.trapezoid(surface, gamma_grid, axis=1), phi0_grid))

    flat_mean = integrate(km_l) / integrate(inside.astype(float))
    ic_mean = integrate(km_l * weight) / integrate(weight)
    return flat_mean / ic_mean if ic_mean > 0.0 else float("inf")


# ---------------------------------------------------------------------------
# Reporting and plotting
# ---------------------------------------------------------------------------


def best_fit(
    phi0_grid: np.ndarray, gamma_grid: np.ndarray, log_l: np.ndarray
) -> tuple[float, float]:
    """Grid maximum of a log-likelihood surface, as ``(phi0, gamma)``."""
    i, j = np.unravel_index(np.argmax(log_l), log_l.shape)
    return float(phi0_grid[i]), float(gamma_grid[j])


def report(
    phi0_grid: np.ndarray, gamma_grid: np.ndarray,
    ic_log_l: np.ndarray, km_window: np.ndarray, km_extended: np.ndarray,
    a_window: np.ndarray, a_window_ceiling: np.ndarray | None,
    ic_best: tuple[float, float], km_best: tuple[float, float],
) -> None:
    """Print the best fits, the tension, and the Bayes-factor sensitivity."""
    ic_phi0, ic_gamma = ic_best
    km_phi0, km_gamma = km_best
    print(f"\n  IceCube (Asimov, DR2 exposure): phi0 = {ic_phi0:.2f}, gamma = {ic_gamma:.2f}")
    print(f"  ARCA21 (one event, two-bin): phi0 = {km_phi0:.2f}, gamma = {km_gamma:.2f}")

    j = int(np.argmin(np.abs(gamma_grid - ic_gamma)))
    print(f"\n  At IceCube's gamma = {ic_gamma:.2f}, one ARCA21 event in the window needs")
    print(f"    phi0 = {phi0_for_one_event(a_window)[j]:.1f}   (bright-track A_eff)")
    if a_window_ceiling is not None:
        print(f"    phi0 = {phi0_for_one_event(a_window_ceiling)[j]:.1f}   "
              "(trigger-level ceiling -- the response this example used to infer with)")
    print(f"  against IceCube's own phi0 = {ic_phi0:.2f}.")

    ts, p_value, sigma = compatibility(ic_log_l, km_extended)
    print("\n  Compatibility of the two datasets (profile-likelihood ratio, 2 dof):")
    print(f"    TS = {ts:.2f}, p = {p_value:.4f}, {sigma:.2f} sigma")
    ts_w, _, sigma_w = compatibility(ic_log_l, km_window)
    print(f"    window-only ARCA21 likelihood: TS = {ts_w:.2f}, {sigma_w:.2f} sigma "
          "(no assumption about the rest of the band, but degenerate in gamma)")

    print("\n  Bayes factor (Eq. 4.5), quoted with its prior box since it has one:")
    for extended, name in ((True, "two-bin "), (False, "window  ")):
        surface = km_extended if extended else km_window
        for top in PHI0_PRIOR_TOPS:
            bf = bayes_factor(phi0_grid, gamma_grid, surface, ic_log_l, top)
            print(f"    {name} likelihood, phi0 prior (0, {top:3.1f}): BF = {bf:6.1f}")


def outer_boundary_log_phi0(
    phi0_grid: np.ndarray, delta: np.ndarray, level: float, upper: bool
) -> np.ndarray:
    """``log10(phi0)`` of a likelihood-ratio contour, one value per ``gamma``.

    Parameters
    ----------
    phi0_grid : np.ndarray, shape (n_phi0,)
        Normalization axis of the surface.
    delta : np.ndarray, shape (n_phi0, n_gamma)
        Log-likelihood deficit from the global maximum.
    level : float
        Contour to trace.
    upper : bool
        Whether to take the top edge of the region rather than the bottom.

    Returns
    -------
    edge : np.ndarray, shape (n_gamma,)
        Contour position, ``NaN`` at any ``gamma`` the region does not reach.
    """
    edge = np.full(delta.shape[1], np.nan)
    for j in range(delta.shape[1]):
        inside = np.flatnonzero(delta[:, j] <= level)
        if inside.size:
            edge[j] = np.log10(phi0_grid[inside[-1] if upper else inside[0]])
    return edge


def label_along_edge(
    ax: plt.Axes, gamma_grid: np.ndarray, edge: np.ndarray,
    gamma_at: float, offset_dex: float, text: str, colour: str, va: str,
) -> None:
    """Write ``text`` beside a contour, rotated to run parallel to it on screen.

    The rotation has to be computed in display coordinates rather than from the
    data slope, because the axes are not square and the ordinate is a log axis
    plotted linearly in ``log10(phi0)``.

    Parameters
    ----------
    ax : plt.Axes
        Axes to draw on, with its limits already fixed.
    gamma_grid : np.ndarray
        Abscissa of ``edge``.
    edge : np.ndarray
        Contour position in ``log10(phi0)`` (:func:`outer_boundary_log_phi0`).
    gamma_at : float
        Where along the contour to anchor the label.
    offset_dex : float
        Perpendicular-ish offset from the contour [dex]; sign chooses the side.
    text : str
        Label.
    colour : str
        Text colour, matching the region it names.
    va : {"top", "bottom"}
        Vertical alignment, so the offset pushes the text clear of the contour.
    """
    valid = np.isfinite(edge)
    span = 0.5 * (gamma_grid[-1] - gamma_grid[0]) / 20.0
    lo, hi = gamma_at - span, gamma_at + span
    y_lo, y_hi = np.interp([lo, hi], gamma_grid[valid], edge[valid])
    (x0, y0), (x1, y1) = ax.transData.transform([(lo, y_lo), (hi, y_hi)])
    angle = float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
    ax.text(gamma_at, float(np.interp(gamma_at, gamma_grid[valid], edge[valid])) + offset_dex,
            text, color=colour, rotation=angle, rotation_mode="anchor",
            ha="center", va=va, fontsize=7)


def make_figure(
    out_path: pathlib.Path,
    phi0_grid: np.ndarray, gamma_grid: np.ndarray,
    ic_log_l: np.ndarray, km_extended: np.ndarray,
    ic_best: tuple[float, float], km_best: tuple[float, float],
) -> None:
    """Draw the two profile-likelihood regions and their best fits.

    Each region is named by a rotated label running alongside it rather than by
    a legend: ARCA21 above its band, IceCube below its ellipse.

    Parameters
    ----------
    out_path : pathlib.Path
        Output file; both ``.pdf`` and ``.png`` are written.
    phi0_grid, gamma_grid : np.ndarray
        Grid axes of the likelihood surfaces.
    ic_log_l, km_extended : np.ndarray, shape (n_phi0, n_gamma)
        Profile log-likelihood surfaces for IceCube and ARCA21.
    ic_best, km_best : tuple of float
        ``(phi0, gamma)`` maxima of the two surfaces.
    """
    log_phi0 = np.log10(phi0_grid)
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.6, 3.6))

        deltas = {}
        for label, surface in (("IceCube", ic_log_l), ("ARCA21", km_extended)):
            colour = EXP_COLOR[label]
            deltas[label] = delta = np.max(surface) - surface
            ax.contourf(gamma_grid, log_phi0, delta, levels=[0.0, *DELTA_LNL_LEVELS],
                        colors=[(*plt.matplotlib.colors.to_rgb(colour), a) for a in (0.45, 0.2)])
            ax.contour(gamma_grid, log_phi0, delta, levels=DELTA_LNL_LEVELS,
                       colors=colour, linestyles=["-", "--"], linewidths=0.9)

        for (phi0, gamma), colour, marker, size in (
            (ic_best, EXP_COLOR["IceCube"], "*", 8),
            (km_best, EXP_COLOR["ARCA21"], "D", 4),
        ):
            ax.plot(gamma, np.log10(phi0), marker=marker, color=colour, ls="",
                    ms=size, mec="k", mew=0.5, zorder=6)

        ax.set_xlim(1.6, 3.0)
        ax.set_ylim(np.log10(0.05), np.log10(300.0))
        ticks = np.array([0.1, 1.0, 10.0, 100.0])
        ax.set_yticks(np.log10(ticks))
        ax.set_yticklabels([f"{t:g}" for t in ticks])
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(r"$\phi_0$  [$10^{-18}\,$GeV$^{-1}$cm$^{-2}$s$^{-1}$sr$^{-1}$]")

        # The rotations are read off the transform, so the limits must be final.
        fig.canvas.draw()
        outer = DELTA_LNL_LEVELS[-1]
        label_along_edge(
            ax, gamma_grid, outer_boundary_log_phi0(phi0_grid, deltas["ARCA21"], outer, True),
            gamma_at=1.80, offset_dex=0.26, text="ARCA21",
            colour=EXP_COLOR["ARCA21"], va="bottom",
        )
        label_along_edge(
            ax, gamma_grid, outer_boundary_log_phi0(phi0_grid, deltas["IceCube"], outer, False),
            gamma_at=2.58, offset_dex=-0.24, text="IceCube",
            colour=EXP_COLOR["IceCube"], va="top",
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png"):
            path = out_path.with_suffix(suffix)
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"Figure saved to: {path.resolve()}")
        plt.close(fig)


def main() -> None:
    args = parse_args()

    print(f"Building IceCube A_eff(E_nu), loading DR2 from: {args.data_dir}")
    ic_log10_e, ic_aeff, ic_livetime_s = build_icecube_aeff(args.data_dir, args.threshold)

    print("Loading the released ARCA21 bright-track effective area ...")
    km_log10_e, km_aeff = arca21_published_aeff()

    ceiling = None
    if not args.no_ceiling:
        print("Building the ARCA21 trigger-level ceiling for contrast ...")
        ceiling = build_arca_ceiling_aeff(args.threshold, ARCA_DEPTH_KM)
        aeff_ratio_report(ceiling, (km_log10_e, km_aeff))

    gamma_grid = np.linspace(*GAMMA_RANGE, args.n_gamma)
    phi0_grid = np.logspace(-1.5, 2.7, args.n_phi0)

    print("\nEvaluating the profile likelihoods on the grid ...")
    a_window, a_band = km_rate_coefficients(gamma_grid, km_log10_e, km_aeff)
    km_window = km_log_likelihood(phi0_grid[:, None], a_window, a_band, extended=False)
    km_extended = km_log_likelihood(phi0_grid[:, None], a_window, a_band, extended=True)
    a_window_ceiling = (
        None if ceiling is None else km_rate_coefficients(gamma_grid, *ceiling)[0]
    )

    signal = asimov_ic(ic_log10_e, ic_aeff, ic_livetime_s)
    ic_observed = asimov_ic(
        ic_log10_e, ic_aeff, ic_livetime_s, bkg_total=BKG_FRACTION * signal.sum()
    )
    ic_log_l = ic_log_likelihood(
        phi0_grid, ic_signal_coefficients(gamma_grid, ic_log10_e, ic_aeff, ic_livetime_s),
        ic_observed, atmospheric_template(IC_FIT_EDGES),
    )

    ic_best = best_fit(phi0_grid, gamma_grid, ic_log_l)
    km_best = best_fit(phi0_grid, gamma_grid, km_extended)
    report(
        phi0_grid, gamma_grid, ic_log_l, km_window, km_extended,
        a_window, a_window_ceiling, ic_best, km_best,
    )
    tension_ladder(phi0_grid, gamma_grid, km_log10_e, km_aeff, ic_log_l)
    li_cross_checks(km_log10_e, km_aeff, ic_log10_e, ic_aeff, ic_livetime_s)
    make_figure(args.out, phi0_grid, gamma_grid, ic_log_l, km_extended, ic_best, km_best)


if __name__ == "__main__":
    main()
