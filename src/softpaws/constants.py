"""Every fixed number softpaws uses: units, physical constants and defaults.

Multiply by a unit to pass a value in, divide by it to read one out:

>>> from softpaws.constants import PeV, km
>>> radius = 1.0 * km          # stored in cm
>>> radius / km
1.0

The units compute in GeV, cm, s and rad. Most of the older names below still
carry a unit suffix and are in the units their suffix names; they move to
the base units as their modules are rewritten.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Units: the base units are GeV, cm, s and rad
# ---------------------------------------------------------------------------

GeV = 1.0
MeV = 1.0e-3 * GeV
TeV = 1.0e3 * GeV
PeV = 1.0e6 * GeV
EeV = 1.0e9 * GeV

cm = 1.0
nm = 1.0e-7 * cm
m = 1.0e2 * cm
km = 1.0e5 * cm

s = 1.0
day = 86400.0 * s
year = 365.25 * day

rad = 1.0
deg = np.pi / 180.0 * rad
sr = 1.0

g = 1.0

# ---------------------------------------------------------------------------
# Fundamental constants
# ---------------------------------------------------------------------------

AVOGADRO_PER_MOL = 6.02214076e23

# Speed of light, for lab-frame decay lengths (gamma * c * tau0).
C_KM_PER_S = 2.99792458e5

# ---------------------------------------------------------------------------
# Particle masses [GeV]
# ---------------------------------------------------------------------------

M_ELECTRON_GEV = 0.51099895e-3
M_MUON_GEV = 0.1056583755
M_PION_GEV = 0.13957039
M_TAU_GEV = 1.77686

# ---------------------------------------------------------------------------
# Particle lifetimes [s], rest frame
# ---------------------------------------------------------------------------

TAU_LIFETIME_S = 2.903e-13

# ---------------------------------------------------------------------------
# Medium densities [g cm^-3]
# ---------------------------------------------------------------------------
# The muon transport coefficients are calibrated in pure water; South
# Pole ice is provided for later reuse. See ``docs/theory/soft_volume.md``.

RHO_WATER_G_CM3 = 1.02
RHO_ICE_G_CM3 = 0.92
#: Fresh water, for Lake Baikal [g cm^-3].
RHO_LAKE_G_CM3 = 1.00

# ---------------------------------------------------------------------------
# Earth
# ---------------------------------------------------------------------------
# Used by the neutrino Earth-attenuation model (``softpaws.transport.attenuation``).
# The mean density feeds the constant-density closed-form column; the layered
# PREM profile (Dziewonski & Anderson 1981) feeds the per-event column.

EARTH_RADIUS_KM = 6371.0
RHO_EARTH_MEAN_G_CM3 = 5.513

# ---------------------------------------------------------------------------
# Length conversions
# ---------------------------------------------------------------------------

CM_PER_KM = 1.0e5
KM_PER_CM = 1.0e-5
CM_PER_M = 1.0e2
M_PER_KM = 1.0e3

# ---------------------------------------------------------------------------
# Time conversions
# ---------------------------------------------------------------------------

#: Seconds in a Julian year, the unit livetimes and exposures are quoted in.
SECONDS_PER_YEAR = 365.25 * 86400.0

# ---------------------------------------------------------------------------
# Neutrino cross sections
# ---------------------------------------------------------------------------

#: Total-to-charged-current cross-section ratio, ``(sigma_CC + sigma_NC) / sigma_CC``
#: (Gandhi et al. 1998; Cooper-Sarkar et al. 2011), constant to a few percent
#: over the energies here.
TOTAL_TO_CC_RATIO = 1.4


# ---------------------------------------------------------------------------
# Muon loss coefficients (used by ``softpaws.transport.coefficients``)
# ---------------------------------------------------------------------------

# Table 1 stops at two moments, so the third-moment column has no Table 1
# counterpart; see third_moment_coefficient.
_TABLE1_N_MOMENTS = 2

# Ionization (Bethe) energy loss for muons in water, a_mu ~ 2.0e-3 GeV cm^2 g^-1
# (PDG muon tables; it varies only logarithmically from 1 GeV to 100 TeV). Table 1
# tabulates the radiative b_mu and d_mu alone, which is all the soft-volume drift
# limit needs. The constant term is needed to close the loss law
# ``-dE/dx = a_mu + b_mu E`` at low energy, and so to define the muon range.
IONIZATION_A_GEV_CM2_G = 2.0e-3

# ---------------------------------------------------------------------------
# Transport exponent: numerical settings (used by ``softpaws.transport.eigenvalue``)
# ---------------------------------------------------------------------------

# Below this |q| the two Beta functions of the three-moment symbol cancel to
# working precision and the q -> 0 digamma limit is used instead. The limit is
# accurate to O(q), so the switch costs at most ~1e-6 relative.
_Q_DIGAMMA_LIMIT = 1.0e-6

# ---------------------------------------------------------------------------
# Earth geometry (used by ``softpaws.transport.earth``)
# ---------------------------------------------------------------------------

#: Longest medium column a muon is given upstream of the detector [km]. Below
#: the horizon the muon is born in rock, which never runs out, and this caps
#: the near-horizon overburden so that no integral runs to infinity. It exceeds
#: every muon range in the problem, so results are insensitive to it.
MAX_UPSTREAM_KM = 100.0

# ---------------------------------------------------------------------------
# Earth attenuation and regeneration (used by ``softpaws.transport.attenuation``)
# ---------------------------------------------------------------------------

# Mean neutral-current inelasticity at UHE. The NC and CC inelasticity
# distributions have very similar shapes above a TeV; this is the NC mean,
# slightly above the CC :data:`~softpaws.transport.source.MEAN_INELASTICITY`
# because NC samples a marginally harder part of the same parton kinematics.
NC_MEAN_INELASTICITY = 0.25

# Rungs of the down-scattering ladder. Each rung is a factor (1 - <y>) in
# energy, so the default reaches 0.25% of the injected energy -- far below the
# point where the surviving neutrino can still make a selectable muon, so the
# result is insensitive to adding more.
NC_REGENERATION_LEVELS = 24

# Mean fraction of the tau energy carried away by the regenerated tau neutrino
# in tau -> nu_tau X. Distinct from softpaws.transport.tau.MEAN_Z, which is the
# fraction carried by the *muon* in the leptonic channel.
TAU_TO_NUTAU_ENERGY_FRACTION = 0.4

# ---------------------------------------------------------------------------
# Neutrino cross section and muon source (used by ``softpaws.transport.source``)
# ---------------------------------------------------------------------------

# UHE CC cross-section power law, sigma_CC = sigma0 (E / E0)^lambda
# (Palmisano et al., arXiv:2607.13143, Eq. 2.5). sigma0 matches the MadGraph result at E0 = 10 PeV
# with the default LHAPDF set; lambda ~ 0.4 follows the small-x PDF behaviour.
SIGMA0_CM2 = 1.48e-33

E0_CROSS_GEV = 1.0e7  # 10 PeV

DEFAULT_LAMBDA = 0.4

# Average CC (DIS) inelasticity <y_w>; near-elastic at UHE (Section 2.3). This is
# the asymptotic value, reached above ~10 PeV; use mean_inelasticity() wherever
# the answer is wanted over a range of energies.
MEAN_INELASTICITY = 0.2

# ---------------------------------------------------------------------------
# Muon range (used by ``softpaws.transport.muon_range``)
# ---------------------------------------------------------------------------

# Muon energy below which a track no longer passes an IceCube-like through-going
# selection. Used as the lower limit of the muon range in
# :func:`range_target_volume_km3`; the resulting volume depends on it only
# logarithmically.
DEFAULT_MUON_THRESHOLD_GEV = 1.0e3

# Matching energy between the two regimes of :func:`stochastic_muon_range_km`:
# radiative and stochastic above, deterministic and ionizing below. It has to sit
# well above the critical energy ``E_c ~ 600`` GeV, where the scale-invariant
# kernel that the first-passage derivation assumes stops describing the losses,
# and low enough that the radiative treatment still covers most of the range.
# 10 TeV is 17 E_c and leaves one decade to the default threshold.
DEFAULT_IONIZATION_MATCH_GEV = 1.0e4

#: Energy span of the cached depth curves [log10 GeV]. The lower edge sits
#: below any threshold ever asked for and the upper one above the ``10^12``
#: bracket of :func:`_near_entry_energy_gev`, so every descent is a difference
#: of two points inside the span. Widening it does not move a single value,
#: since the node spacing is fixed at ``1 / nodes_per_decade`` and not by the
#: endpoints.
_CURVE_LOG10_LO = 0.0

_CURVE_LOG10_HI = 15.0

#: Lattice density of the deterministic curve. The radiative one takes its own
#: from ``running_nodes_per_decade``, whose integrand ``1 / Phi'(0; E)`` is
#: nearly flat in ``lnE``. This one is not: ``E / (a_mu + b_mu E)`` rises
#: exponentially in ``lnE`` below the critical energy and flattens above it, so
#: it wants the finer lattice. At this density the deterministic range is
#: converged to 5e-7, which the earlier grid -- refined to the span of each
#: descent and so finest close to threshold -- reached only to 3e-5.
_CSDA_NODES_PER_DECADE = 3072

# ---------------------------------------------------------------------------
# Loss distribution: numerical settings (used by ``softpaws.transport.loss_distribution``)
# ---------------------------------------------------------------------------

# Chunk size for the Gil-Pelaez quadrature. The integrand is an (n_ell, n_k)
# complex array, so a bare outer product over a fine ``k`` grid runs to
# gigabytes; chunking over ``k`` bounds it at this many nodes at a time while
# leaving the accumulated result bitwise-comparable to within quadrature error.
_GIL_PELAEZ_CHUNK = 4096

# ---------------------------------------------------------------------------
# Loss-model ensemble (used by ``softpaws.transport.loss_ensemble``)
# ---------------------------------------------------------------------------

#: Energy grid of the budget [GeV], the shipped table's range.
E_GRID = np.logspace(2.0, 10.0, 33)

#: Moment orders kept: the first log-loss moment (the drift, the
#: energy-reconstruction scale) and the second (the fluctuation scale).
N_MOMENTS = 2

# ---------------------------------------------------------------------------
# Tau channel (used by ``softpaws.transport.tau``)
# ---------------------------------------------------------------------------

# tau -> mu nu_mu nu_tau branching ratio (PDG).
BR_TAU_TO_MU = 0.1739

# <z> of the polarized tau -> mu decay spectrum (see decay_spectrum), exact:
# z_moment(1) = 1/6 + 2/15 = 3/10.
MEAN_Z = 0.3

# ---------------------------------------------------------------------------
# Astrophysical fluxes (used by ``softpaws.fluxes.astrophysical``)
# ---------------------------------------------------------------------------

#: Pivot energy of every normalization here [GeV].
FLUX_PIVOT_GEV = 1.0e5

#: Unit of the normalizations, ``1e-18 GeV^-1 cm^-2 s^-1 sr^-1``.
FLUX_UNIT = 1.0e-18

# ---------------------------------------------------------------------------
# Direction-resolved response (used by ``softpaws.response.declination``)
# ---------------------------------------------------------------------------

#: Neutrino energies every curve here is returned on [log10 GeV].
COMMON_LOG10_E = np.linspace(3.0, 8.0, 26)

#: Directions sampled inside each published declination band.
N_SUB_BAND = 5

#: Zenith bands a source's daily sweep is histogrammed into.
N_COS_THETA = 180

#: Hour angles sampled over a sidereal day.
N_HOUR_ANGLE = 192

#: Rungs of the neutral-current regeneration ladder, and the decades it spans.
N_RUNG = 32

RUNG_DECADES = 4.0

#: Bottom of the analysis window [GeV]. This is what decides whether a
#: sensitivity carries declination information: raise it and the Earth-absorbed
#: high-energy end, where the declination dependence lives, is all that is left.
DEFAULT_EMIN_GEV = 1.0e5

#: Energy at which a fitted light reach vanishes [GeV].
REACH_PIVOT_GEV = 10.0**9.67

#: Effective radius at :data:`REACH_REFERENCE_GEV` that :func:`fit_light_reach`
#: scans, as a fraction of the instrumented footprint radius. The range covers
#: a detector that responds to two thirds of its footprint at 1 PeV and one
#: that already responds past it.
REACH_FRACTIONS = np.linspace(0.6, 1.2, 13)

#: Energy the scanned fraction is quoted at [GeV].
REACH_REFERENCE_GEV = 1.0e6

# ---------------------------------------------------------------------------
# Effective-area grids (used by ``softpaws.response.effective_area``)
# ---------------------------------------------------------------------------

#: Neutrino-energy grid of the IceCube curves [log10 GeV]: 0.2 dex from 1 TeV
#: to 100 PeV, the span of the DR2 effective-area table.
IC_LOG10_E = np.linspace(3.0, 8.0, 26)

#: Number of declination slices of the upgoing hemisphere.
N_DEC = 60

#: Neutrino-energy grid of the water-site curves [log10 GeV]: 0.2 dex from
#: 10 TeV to 10 EeV, the span of the ARCA figures.
ARCA_LOG10_E = np.arange(4.0, 10.01, 0.2)

#: Number of equal-``cos(theta)`` slices of the sky.
N_ZENITH = 90

# ---------------------------------------------------------------------------
# Light yield and light reach (used by ``softpaws.response.light_reach``)
# ---------------------------------------------------------------------------

#: Fine-structure constant, for the Frank-Tamm yield.
FINE_STRUCTURE = 7.2973525693e-3

#: Wavelengths every optical integral runs over [nm]. Wide enough that the
#: photocathode and the medium, not the grid, decide where the band ends.
WAVELENGTH_NM = np.linspace(280.0, 680.0, 201)

#: Charged track length in an electromagnetic shower [m GeV^-1], at water
#: density. Paired with the water-density ``b_mu`` in
#: :func:`brightness_factor`, so the product that enters the yield is density
#: independent.
EM_TRACK_LENGTH_M_PER_GEV = 4.0

#: Coincidence partners of a single-PMT module: IceCube's HLC accepts the
#: nearest or next-to-nearest neighbour on the same string, up or down. Part of
#: the trigger definition, and a count, so it is not a tunable.
HLC_PARTNERS = 4

#: Modules that must register a coincident hit for the track to count. A hit is
#: a local coincidence, an HLC pair at IceCube or two photomultipliers of one
#: module at KM3NeT, with each receiver firing on one photoelectron at Poisson
#: probability (:func:`hit_probability`); what a trigger then demands is a
#: multiplicity, and IceCube's simple-majority trigger asks for eight.
#: Everything else the condition needs is a published instrument number: the
#: module density, the photocathode area, the efficiency curve and the medium's
#: optics. See :func:`hit_count`.
DEFAULT_MIN_MODULES = 8.0

#: Fraction of its photocathode area a module presents to an arriving photon.
#: A sphere uniformly covered with photocathode of area ``A`` presents ``A / 4``
#: from every direction; a single flat photomultiplier facing one hemisphere
#: with cosine acceptance averages to the same quarter over the full sky.
PROJECTED_FRACTION = 0.25

#: Distances the one-photoelectron radius is tabulated on [m], log spaced so the
#: inversion stays accurate over the four decades of brightness in play.
_HIT_DISTANCE_M = np.logspace(-1.0, 3.2, 400)

#: Signed offsets the mean hit count is tabulated on [m], for the inversion in
#: :func:`reach_offset_m`. The positive end comfortably exceeds any reach in
#: play; the negative end only has to cover the interpolation edge, since a
#: track the condition wants *inside* the array is handled by the multiplicity
#: weight of :func:`effective_body_km` and the offset is clipped at zero.
_REACH_OFFSET_M = np.linspace(-100.0, 1200.0, 261)

#: Trapezoid quadrature weights of the distance grid, so the count integrals
#: reduce to matrix products against :data:`_REACH_KERNEL`.
_D_STEP = np.diff(_HIT_DISTANCE_M)

# ---------------------------------------------------------------------------
# Sensitivities and upper limits (used by ``softpaws.response.sensitivity``)
# ---------------------------------------------------------------------------

#: Events a background-free search excludes at 90% confidence (Feldman-Cousins,
#: zero observed on zero background).
N_EVENTS_LIMIT = 2.44

#: Pivot energy of a quoted power-law flux [GeV].
PIVOT_ENERGY_GEV = 1.0e5

#: Natural logarithm of ten, the width of a decade in ``ln(E)``.
LN10 = float(np.log(10.0))

#: Confidence level the Feldman-Cousins construction is built at.
CONFIDENCE_LEVEL = 0.9

#: Radius of the bin a point-source background is counted in [deg]. About the
#: angular resolution a through-going track is reconstructed to above a TeV,
#: which is what sets how much sky an unbinned search has to look through.
#: A measured point-spread function replaces it wherever one is available.
DEFAULT_BIN_RADIUS_DEG = 1.0

#: Background above which the exact construction is replaced by its large-count
#: form. The construction below costs memory linear in the background, because
#: it holds every count a Poisson of that mean can deliver, so an unbounded one
#: is not merely slow. By here the average upper limit is a straight line in
#: ``sqrt(b)`` to better than a per cent, and :func:`sensitivity_upper_limit`
#: continues along that line instead.
EXACT_BACKGROUND_MAX = 200.0

#: Relative spacing the background is rounded to before the exact construction
#: is cached. A window scan asks for thousands of backgrounds that differ in
#: the fourth figure, and every distinct one is a construction from scratch,
#: so without this the cache never hits. The average upper limit grows no
#: faster than the root of the background, so a grid this fine moves it by
#: under half of this, which is far below anything else in the answer.
BACKGROUND_CACHE_STEP = 0.005

#: Containment of the point-spread function that makes the best counting bin.
#: For a Gaussian the radius maximizing signal over the root of the background
#: is ``1.585 sigma``, and the 68% containment radius is ``1.510 sigma``, so
#: taking the published containment as the bin is the optimum to 5% with
#: nothing fitted.
OPTIMAL_CONTAINMENT = 0.68

# ---------------------------------------------------------------------------
# Event benchmark (used by ``softpaws.comparison.events``)
# ---------------------------------------------------------------------------

#: Hadronic-model spread carried as the atmospheric normalization error.
ATM_ERR = 0.25

# ---------------------------------------------------------------------------
# Likelihood defaults (used by ``softpaws.comparison.likelihood``)
# ---------------------------------------------------------------------------

# Paper best-fit diffuse flux (Table 2, diffusion model), injected as the ground
# truth for the Asimov datasets driving the Section 4 figures.
PHI0_TRUTH = 0.63

GAMMA_TRUTH = 2.38

# ---------------------------------------------------------------------------
# Event rates: numerical settings (used by ``softpaws.comparison.rates``)
# ---------------------------------------------------------------------------

# Bisection steps used by fit_scale_factor_with_background. Each step halves the
# bracket, so this reaches the floating-point resolution of the upper bound.
_BISECTION_STEPS = 80

# ---------------------------------------------------------------------------
# Flavour likelihood grids (used by ``softpaws.comparison.reco_likelihood``)
# ---------------------------------------------------------------------------

#: Reconstructed-energy grid the fold projects onto, and the window the fit
#: uses. The grid is wider than the window so the fold conserves counts.
RECO_EDGES = np.arange(1.0, 8.51, 0.25)

#: Fractional model-shape systematic per bin. The percent-level residuals of
#: the folded model would otherwise dominate the likelihood through the
#: highest-statistics atmospheric bins; each bin's deviance is scaled by
#: ``1 + (MODEL_SYS^2) mu``, which de-weights exactly those bins and leaves
#: the Poisson-limited tail untouched.
MODEL_SYS = 0.10

#: Fine true-energy grid of the model responses.
LOG10_E_GRID = np.linspace(3.0, 8.5, 111)

#: Quadrature nodes on the muon energy fraction ``x`` of ``tau -> mu nu nu``
#: (unpolarized spectrum ``f(x) = 5/3 - 3x^2 + 4x^3/3``, mean 0.35), used to
#: shift the ``nu_mu``-simulation smearing onto the tau channel.
TAU_DECAY_X = np.linspace(0.025, 0.975, 20)

#: Flavour-ratio scan grid. The full range is safe only because the
#: ``nu_mu`` flux is anchored: without the anchor, tracks cannot tell the
#: two channel templates apart by shape and the fit relabels the whole
#: astrophysical excess as tau at several times the measured flux for free.
R_GRID = np.linspace(0.0, 1.0, 41)

#: Feldman-Cousins calibration: truth points the toy distributions are built
#: at, and the ratio grid each toy's global minimum is scanned on. Truths
#: stop at 0.9: they carry the anchored ``nu_mu`` flux, so the tau flux
#: diverges at ``r = 1``; the last threshold is held beyond.
FC_R_TRUE = np.linspace(0.0, 0.9, 10)

FC_R_SCAN = np.linspace(0.0, 1.0, 11)

#: Flux plane [combined-fit per-flavour flux units]. The scan reaches the
#: unconstrained minimum, which lies outside the physical range.
PHI_MU_GRID = np.linspace(0.0, 1.6, 33)

PHI_TAU_GRID = np.linspace(0.0, 8.0, 33)

#: Count plane: expected astrophysical tracks in the fit window from each
#: channel, the shape profiled.
N_MU_GRID = np.linspace(0.0, 160.0, 33)

N_TAU_GRID = np.linspace(0.0, 100.0, 33)

#: Tau-flux scan at the anchored ``nu_mu`` flux [combined-fit units].
PHI_TAU_SCAN = np.linspace(0.0, 6.0, 61)
