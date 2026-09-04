"""Example 34 -- dark-matter line sensitivity with the current effective area.

Example 23 built the ``chi chi -> nu nubar`` line sensitivity on the range-based
``threshold_effective_area_cm2``: a muon range down to threshold, a sphere of the
instrumented volume, pure absorption in the Earth, and no direction dependence
beyond a J-factor-weighted survival probability. Everything the effective-area
work of examples 28 and 30 added since then bears directly on that calculation,
and this example redoes it with the current model:

* the **stochastic first-passage range** in place of the continuous-slowing-down
  one (``docs/first_passage_range.md``),
* **neutral-current regeneration** and the ``nu_tau -> tau -> mu`` channel, both
  of which a track selection cannot separate from direct ``nu_mu``,
* and a **finite upstream column**, which is the change that matters most here
  and which example 23 did not have at all.

**Why the overburden is the story.** The Galactic Center sits at
``dec = -29 deg``, which at the South Pole is 61 deg from the zenith: *downgoing*.
Example 23 noted this and drew the right conclusion for the neutrino -- the
Earth barely attenuates a signal that never enters it -- but the same fact
constrains the muon far more tightly. A downgoing muon cannot be born above the
ice, so its available column is the ~2 km of overburden rather than the ~15 km
of water-equivalent range a 1 PeV muon would otherwise cover. Truncating the
first-passage integral there costs a factor of a few in the J-weighted effective
area, in the direction that *weakens* the predicted sensitivity, which is the
direction example 23's residual asked for.

**ARCA230 sees the same halo from the other hemisphere.** At the Capo Passero
latitude the Galactic Center is below the horizon for most of a sidereal day, so
the signal arrives *upgoing* through the Earth: the neutrino is attenuated where
IceCube's is not, but the muon gets an unlimited upstream column where IceCube's
is capped. The two effects pull in opposite directions and neither site is
uniformly better, which is what makes the comparison worth drawing.
:func:`zenith_exposure` is where this enters -- it turns the J-factor's
declination profile into the distribution of arrival zeniths each site actually
sees, averaged over hour angle.

**Why no published curve is drawn.** Example 23 set its prediction against the
IceCube-Gen2 sensitivity digitized from Arguelles et al., *Dark Matter
Annihilation to Neutrinos*, Rev. Mod. Phys. 93 (2021) 035007
[arXiv:1912.09486], and that comparison does not hold up. Three mismatches, and
none of them can be repaired from this side:

*It is all-flavor.* Their Table III lists IceCube-Gen2 under "All Flavors". The
model here is ``nu_mu`` charged current plus ``nu_tau -> tau -> mu``, which is
what a through-going track selection sees and nothing else. At these energies a
diffuse all-flavor sensitivity is carried mostly by cascades.

*It is not a line search.* Their Sec. IV states plainly: "We have recast the
estimates of diffuse flux sensitivity given in (Aartsen et al., 2019) to
estimate the sensitivity to dark matter annihilation." A diffuse-flux
sensitivity integrated over the sky and over a broad energy range, converted
afterwards into an annihilation cross section, is a different statistical object
from a monochromatic line search, and the Fig. 4 caption confirms the whole
high-mass panel is built "by converting either the detected flux or the reported
upper limit into a conservative upper bound".

*The top of the range is a different detector.* The Gen2 diffuse sensitivity
above ~10 PeV comes from the radio array, not from the optical in-ice array this
model represents.

Two further differences were mismatched in example 23 and *are* repairable, so
they have been repaired here rather than dropped: that work assumes **five**
years, not ten, and a **generalized** NFW profile with ``gamma = 1.2``,
``rho_0 = 0.4 GeV cm^-3`` and ``R_0 = 8.127 kpc`` (Benito et al. 2019), against
example 23's ``gamma = 1``, ``0.3 GeV cm^-3`` and ``8.5 kpc``. The halo
parameters alone move the J-factor by a factor of a few, so they are worth
matching to the community standard whether or not anything is overlaid. This
example therefore uses the generalized profile and reports the shift.

What remains is a prediction rather than a comparison, which is the honest
product: given each site's geometry and location, this is the line sensitivity a
background-free through-going-track search would have. One curve per site, the
parameter-free instrumented footprint, which is a geometric ceiling.

**The light reach of Eq. (17) is deliberately not applied.** It would have to be
evaluated at the muon's energy *where it is seen*, which depends on how far the
muon travelled, and that is exactly the ``W(E, ell)`` kernel Sec. III warns does
not factor out of the depth integral. Evaluating it at the production energy
instead -- the approximation
:func:`~softpaws.transport.soft_volume.light_reach_radius_km` documents -- shifts
the effective radius by ``Lambda Phi'(0) <L>``, which at 10 PeV is 3% of
``R_det`` for Gen2 but 53% for ARCA230, 140% for P-ONE and 448% for Baikal-GVD.
On the small-footprint arrays the correction is larger than the quantity it
corrects, so the reach is left out rather than shown with a caveat. Doing it
properly means keeping ``R_eff`` inside the depth integral, which the
monochromatic case permits -- there is no diagonalization at ``s = 0`` to
protect -- but that is a different calculation from the illustration here.

Caveats carried over from example 23: the search is treated as background free at
every mass, so :data:`N_EVENTS_LIMIT` is a zero-background Feldman-Cousins limit
and the curves are ceilings rather than forecasts; the tabulated cross section is
``nu_mu`` on an isoscalar target; the full sky is used with no angular cut and no
energy window, both of which a real line search would impose; and the halo is a
single profile with no substructure and no profile uncertainty, which is the
largest astrophysical error on the vertical scale and is not shown.

Usage
-----
    python examples/34_dm_line_sensitivity.py
    python examples/34_dm_line_sensitivity.py --livetime-yr 15
    python examples/34_dm_line_sensitivity.py --no-electroweak
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.detectors import ARCA230, GEN2, GVD, PONE, TRIDENT, Site
from softpaws.transport.attenuation import flavour_transmission
from softpaws.transport.cross_section import bgr18_cross_section
from softpaws.transport.eigenvalue import phi_eigenvalue
from softpaws.transport.soft_volume import (
    DEFAULT_MUON_THRESHOLD_GEV,
    truncated_muon_range_km,
)
from softpaws.transport.source import (
    MEAN_INELASTICITY,
    inelasticity_factor,
    nucleon_number_density,
)
from softpaws.transport.tau import BR_TAU_TO_MU, MEAN_Z
from softpaws.utils.constants import CM_PER_KM

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"

CROSS_SECTION = bgr18_cross_section()

# ---------------------------------------------------------------------------
# Halo
# ---------------------------------------------------------------------------

# Generalized Navarro-Frenk-White halo,
#
#     rho(r) = rho_s 2^(3-gamma) / [ (r/r_s)^gamma (1 + r/r_s)^(3-gamma) ],
#
# normalized to the local density at the solar radius. The parameters are the
# ones Arguelles et al. (arXiv:1912.09486) adopt from the Benito et al. (2019)
# fit, and are the reason this example does not inherit example 23's halo: a
# prediction quoted on a different profile from everyone else's cannot be
# compared with anything. gamma = 1 recovers the ordinary NFW cusp.
RHO_SUN_GEV_CM3 = 0.4
GAMMA_SLOPE = 1.2
R_SUN_KPC = 8.127
R_SCALE_KPC = 20.0
R_VIRIAL_KPC = 200.0
KPC_TO_CM = 3.0856775814913673e21

# Example 23's halo, kept only so the shift can be reported rather than hidden.
LEGACY_HALO = {"rho": 0.3, "gamma": 1.0, "r_sun": 8.5}

# Galactic Center in equatorial (J2000) coordinates.
GC_RA_DEG = 266.405
GC_DEC_DEG = -28.936

# Feldman-Cousins 90% CL upper limit on the signal for zero observed background.
N_EVENTS_LIMIT = 2.44

# ---------------------------------------------------------------------------
# Electroweak depletion of the line
# ---------------------------------------------------------------------------

# Matching scale q_W below which the electroweak shower is not developed, and
# the SU(2)_L coupling there. HDMSpectra (Bauer, Rodd and Webber, JHEP 06 (2021)
# 121 [arXiv:2007.15001]) starts its evolution at q_W = 100 GeV.
EW_SCALE_GEV = 100.0
ALPHA_2 = 0.0338
# Weak isospin of the emitting particle. A Standard Model neutrino is purely
# left-handed and therefore always sits in a doublet, so T = 1/2 at every mass.
EW_ISOSPIN = 0.5

# ---------------------------------------------------------------------------
# Sites
# ---------------------------------------------------------------------------

# IceCube-Gen2 baseline: ~8 km^3 instrumented, taken as an upright hexagonal
# prism of that volume, at the geographic South Pole. Gen2 extends the footprint
# and not the depth, so the height is the same 1 km of instrumented string that
# IceCube spans and the footprint carries the whole of the volume increase.
# The site geometries live in :mod:`softpaws.detectors`. Gen2 is the 7.9 km^3
# prism of the technical design report unless --volume-km3 says otherwise.
GEN2_VOLUME_KM3 = float(GEN2.detector_volume_km3())
SITE_COLORS = {"IceCube-Gen2": "C0", "ARCA230": "C1", "TRIDENT": "C2", "P-ONE": "C3",
               "Baikal-GVD": "C4"}


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------

# Arrival-direction bands. The J-factor and the exposure are both smooth in
# cos(theta), so this is finer than either needs.
N_COS_THETA = 24
# Hour-angle samples used to turn a declination into an arrival-zenith
# distribution. Irrelevant at the South Pole, where the zenith is fixed.
N_HOUR_ANGLE = 96
# Rungs of the neutral-current / tau regeneration ladder.
N_RUNG = 32

# Mass range. The lower edge is where a 1 TeV muon threshold stops biting; the
# upper edge is where the BGR18 cross section and the transport coefficients
# both become extrapolations of their tabulations.
MASS_RANGE_GEV = (1.0e4, 1.0e10)
N_MASS = 49

# Below this a line search is background limited: the atmospheric muon and
# neutrino fluxes are still well above the astrophysical one, so the
# zero-background ceiling these curves assume is not reachable. Shaded in the
# figure rather than cut, since the curves themselves stay well defined.
ATMOSPHERIC_TOP_GEV = 1.0e5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume-km3", type=float, default=GEN2_VOLUME_KM3,
                        help="Instrumented volume of IceCube-Gen2 [km^3].")
    parser.add_argument("--livetime-yr", type=float, default=10.0,
                        help="Exposure time, used for both sites [yr].")
    parser.add_argument("--threshold-gev", type=float, default=DEFAULT_MUON_THRESHOLD_GEV,
                        help="Muon selection threshold [GeV].")
    parser.add_argument("--no-electroweak", action="store_true",
                        help="Keep the tree-level line, with no electroweak depletion.")
    parser.add_argument("--out", type=pathlib.Path,
                        default=_DEFAULT_OUT_DIR / "34_dm_line_sensitivity.pdf",
                        help="Output file for the figure.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Halo J-factor
# ---------------------------------------------------------------------------


def nfw_density(
    radius_kpc: float | np.ndarray,
    rho_sun_gev_cm3: float = RHO_SUN_GEV_CM3,
    gamma: float = GAMMA_SLOPE,
    r_sun_kpc: float = R_SUN_KPC,
) -> np.ndarray:
    """Generalized NFW dark matter density profile.

    Parameters
    ----------
    radius_kpc : float or np.ndarray
        Galactocentric radius [kpc].
    rho_sun_gev_cm3 : float, optional
        Density at the solar radius, which fixes the normalization.
    gamma : float, optional
        Inner slope. ``1`` is the ordinary NFW cusp.
    r_sun_kpc : float, optional
        Distance from the Sun to the Galactic Center [kpc].

    Returns
    -------
    density : np.ndarray
        Dark matter mass density [GeV cm^-3], zero beyond :data:`R_VIRIAL_KPC`.
    """
    radius = np.atleast_1d(np.asarray(radius_kpc, dtype=float))

    def shape(r: np.ndarray) -> np.ndarray:
        x = np.maximum(r, 1.0e-3) / R_SCALE_KPC
        return 2.0 ** (3.0 - gamma) / (x**gamma * (1.0 + x) ** (3.0 - gamma))

    # Regulate the r -> 0 cusp; the enclosed J-factor converges regardless.
    rho_s = rho_sun_gev_cm3 / float(shape(np.array([r_sun_kpc]))[0])
    return np.where(radius <= R_VIRIAL_KPC, rho_s * shape(radius), 0.0)


def los_j_factor(
    psi_deg: float | np.ndarray,
    n_steps: int = 512,
    **halo: float,
) -> np.ndarray:
    """Line-of-sight annihilation J-factor at an angle from the Galactic Center.

    Parameters
    ----------
    psi_deg : float or np.ndarray
        Angular separation from the Galactic Center [deg].
    n_steps : int, optional
        Number of trapezoidal steps along the line of sight. Defaults to 512.
    **halo : float
        Overrides passed through to :func:`nfw_density`, used to price the
        difference against example 23's profile.

    Returns
    -------
    j_los : np.ndarray
        ``\\int dl \\rho^2`` along the line of sight [GeV^2 cm^-5].
    """
    r_sun = halo.get("r_sun_kpc", R_SUN_KPC)
    psi_rad = np.deg2rad(np.atleast_1d(np.asarray(psi_deg, dtype=float)))
    l_max = r_sun * np.cos(psi_rad) + np.sqrt(
        R_VIRIAL_KPC**2 - (r_sun * np.sin(psi_rad)) ** 2
    )
    fraction = np.linspace(0.0, 1.0, n_steps)
    l_kpc = l_max[:, None] * fraction[None, :]
    r_kpc = np.sqrt(r_sun**2 + l_kpc**2 - 2.0 * r_sun * l_kpc * np.cos(psi_rad)[:, None])
    integrand = nfw_density(r_kpc, **halo) ** 2
    return np.trapezoid(integrand, l_kpc, axis=1) * KPC_TO_CM


def ew_line_survival(mass_gev: float | np.ndarray) -> np.ndarray:
    """Fraction of the annihilation neutrinos that stay on the line [0, 1].

    Above the electroweak scale ``chi chi -> nu nubar`` is not monochromatic.
    Each primary neutrino can radiate a ``W`` or ``Z``, which both removes it
    from the line at ``E_nu = m_chi`` and starts a shower that fills a continuum
    below. HDMSpectra describes exactly this: energy conservation is maintained
    because "there is a considerable contribution to a delta-function at
    ``x = 1``, associated with events where an initial ``W`` or ``Z`` was never
    emitted and thus no subsequent shower developed."

    The coefficient of that delta function is a no-emission probability, and at
    leading double logarithm it is the isospin Sudakov factor of their
    Eq. (B.7),

    .. math:: \\Delta^{(T)}(Q) \\sim \\exp\\left[-T(T+1)\\,
        \\frac{\\alpha_2}{2\\pi}\\,\\ln^2\\!\\left(\\frac{Q}{q_W}\\right)\\right],

    with ``T`` the weak isospin of the emitting particle. A Standard Model
    neutrino is left-handed, so ``T = 1/2`` and ``T(T+1) = 3/4`` at every mass.

    **Annihilation, not decay.** HDMSpectra works with decays, where ``x = 2E/m``
    puts each daughter at ``Q = m/2``. Annihilation of two particles of mass
    ``m_chi`` is the decay of one particle of mass ``2 m_chi``, so here
    ``Q = m_chi`` -- each outgoing neutrino carries the full ``m_chi``.

    Applied as a multiplicative factor on the signal, this is the statement that
    the analysis keeps **only the line** and discards the continuum. That is
    conservative twice over: the continuum neutrinos are real signal, and a
    threshold-based track selection would accept some of them. What it buys is
    that the parent stays monochromatic, so the whole calculation remains App.
    I's ``s -> 0`` case with ``Phi(0) = 0``; folding the continuum in instead
    would reintroduce a spectral weight and a different calculation entirely.

    Parameters
    ----------
    mass_gev : float or np.ndarray
        Dark matter mass ``m_chi`` [GeV].

    Returns
    -------
    survival : np.ndarray
        Fraction of neutrinos remaining at ``E_nu = m_chi``. Clamped to one
        below :data:`EW_SCALE_GEV`, where the double logarithm does not apply.

    Notes
    -----
    This is the leading double logarithm with a fixed coupling. It neglects the
    running of ``alpha_2``, the single-logarithmic terms, and the hypercharge
    contribution, which carries no isospin double logarithm. HDMSpectra puts the
    error from missing next-to-leading logarithms at "size O(10%) up to the EeV
    scale", which is the accuracy to claim here -- and it is small against the
    halo-profile uncertainty that is not shown at all.
    """
    scale = np.maximum(np.asarray(mass_gev, dtype=float), EW_SCALE_GEV)
    casimir = EW_ISOSPIN * (EW_ISOSPIN + 1.0)
    exponent = casimir * ALPHA_2 / (2.0 * np.pi) * np.log(scale / EW_SCALE_GEV) ** 2
    return np.exp(-exponent)


def dm_line_flux_general_s(
    psi_deg: float | np.ndarray,
    sigma_v_cm3_s: float,
    mass_gev: float | np.ndarray,
    spectral_index_s: float = 0.0,
    b_mu: float = 1.0,
    d_mu: float = 0.1,
    n_steps: int = 512,
) -> np.ndarray:
    """Literal Eq. I.2 (App. I) evaluation of the DM-line flux, at general ``s``.

    .. math:: \\frac{dN}{dE\\,d\\Omega} = \\frac{\\langle\\sigma v\\rangle}
        {2 m_\\chi^2}\\, \\mathcal{I}(s)
        \\int d\\ell\\, \\rho^2(r(\\psi,\\ell))\\, e^{-\\ell\\,\\Phi(s)}.

    Carried over from example 23, which this example replaces, because it is
    what licenses the plain J-factor formula used everywhere else here. At the
    physical value ``s = 0`` -- the only one that makes sense for a genuine
    monochromatic line, which has no continuum spectral index to average over --
    ``Phi(0) = 0`` identically
    (:func:`softpaws.transport.eigenvalue.phi_eigenvalue`) and ``I(0) = 1``, so
    this reduces exactly to ``(sigma_v / 2 m_chi^2) * los_j_factor(psi)``.
    :func:`check_appendix_i` verifies that rather than asserting it.

    ``ell`` is converted from kpc to km to match ``Phi``'s km^-1 convention, so
    the expression stays dimensionally honest away from ``s = 0`` too: for any
    nonzero ``s``, ``e^{-ell Phi(s)}`` is astronomically small over kpc-scale
    distances, which is *why* the line-of-sight propagator is only ever
    evaluated at ``s = 0`` in practice. ``b_mu`` and ``d_mu`` therefore only
    matter for exercising the general-``s`` shape and are inert at ``s = 0``.

    Parameters
    ----------
    psi_deg : float or np.ndarray
        Angular separation from the Galactic Center [deg].
    sigma_v_cm3_s : float
        Velocity-averaged annihilation cross section [cm^3 s^-1].
    mass_gev : float or np.ndarray
        Dark matter mass ``m_chi`` [GeV]; the line sits at ``E_nu = m_chi``.
    spectral_index_s : float, optional
        Mellin index ``s`` at which to evaluate Eq. I.2. Defaults to ``0``.
    b_mu, d_mu : float, optional
        Muon drift/diffusion coefficients [km^-1] feeding ``Phi(s)``; inert at
        ``s = 0``. Placeholders, not physical values.
    n_steps : int, optional
        Number of trapezoidal steps along the line of sight. Defaults to 512.

    Returns
    -------
    flux : np.ndarray
        ``dN / (dE dOmega)``, in the convention of :func:`los_j_factor` scaled
        by the annihilation prefactor.
    """
    phi_s = float(phi_eigenvalue(spectral_index_s, b_mu, d_mu))
    psi_rad = np.deg2rad(np.atleast_1d(np.asarray(psi_deg, dtype=float)))
    l_max = R_SUN_KPC * np.cos(psi_rad) + np.sqrt(
        R_VIRIAL_KPC**2 - (R_SUN_KPC * np.sin(psi_rad)) ** 2
    )
    fraction = np.linspace(0.0, 1.0, n_steps)
    l_kpc = l_max[:, None] * fraction[None, :]
    r_kpc = np.sqrt(R_SUN_KPC**2 + l_kpc**2 - 2.0 * R_SUN_KPC * l_kpc * np.cos(psi_rad)[:, None])
    l_km = l_kpc * KPC_TO_CM / CM_PER_KM
    integrand = nfw_density(r_kpc) ** 2 * np.exp(-l_km * phi_s)
    j_weighted = np.trapezoid(integrand, l_kpc, axis=1) * KPC_TO_CM
    factor = float(inelasticity_factor(spectral_index_s))
    return (sigma_v_cm3_s / (2.0 * np.asarray(mass_gev, dtype=float) ** 2)) * factor * j_weighted


def check_appendix_i(mass_gev: np.ndarray) -> None:
    """Verify Eq. I.2 at ``s = 0`` against the plain J-factor formula."""
    probe_sigma_v = 1.0e-23  # arbitrary; the check is on the ratio, not the scale
    probe_mass = mass_gev[[0, mass_gev.size // 2, -1]]
    direct = dm_line_flux_general_s(
        np.zeros_like(probe_mass), probe_sigma_v, probe_mass, spectral_index_s=0.0
    )
    reference = probe_sigma_v / (2.0 * probe_mass**2) * los_j_factor(0.0)[0]
    error = float(np.max(np.abs(direct / reference - 1.0)))
    print(f"  max |Eq. I.2(s=0) / plain J-factor - 1| = {error:.2e} (should be ~0)")


def sky_j_profile(
    n_dec: int = 181,
    n_ra: int = 360,
    n_psi: int = 400,
    **halo: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Declination profile of the halo J-factor and its all-sky total.

    Parameters
    ----------
    n_dec, n_ra, n_psi : int, optional
        Samples in declination, right ascension, and the interpolation grid in
        angular separation from the Galactic Center.
    **halo : float
        Overrides passed through to :func:`nfw_density`.

    Returns
    -------
    dec_deg : np.ndarray, shape (n_dec,)
        Declination samples [deg].
    dj_ddec : np.ndarray, shape (n_dec,)
        ``dJ/d(dec)`` [GeV^2 cm^-5 rad^-1], including the ``cos(dec)`` weight.
    j_total : float
        All-sky J-factor [GeV^2 cm^-5].
    """
    # The J-factor spans orders of magnitude between the Galactic Center and the
    # anticenter, so tabulate it on a log-spaced grid in psi and interpolate the
    # logarithm rather than evaluating the line-of-sight integral per sky pixel.
    psi_grid = np.concatenate(([0.0], np.logspace(-2.0, np.log10(180.0), n_psi)))
    log_j_grid = np.log(los_j_factor(psi_grid, **halo))

    dec_deg = np.linspace(-90.0, 90.0, n_dec)
    ra_deg = np.linspace(0.0, 360.0, n_ra, endpoint=False)
    dec_rad, ra_rad = np.deg2rad(dec_deg), np.deg2rad(ra_deg)
    gc_dec_rad, gc_ra_rad = np.deg2rad(GC_DEC_DEG), np.deg2rad(GC_RA_DEG)

    cos_psi = np.sin(dec_rad)[:, None] * np.sin(gc_dec_rad) + np.cos(dec_rad)[:, None] * np.cos(
        gc_dec_rad
    ) * np.cos(ra_rad[None, :] - gc_ra_rad)
    psi_deg = np.rad2deg(np.arccos(np.clip(cos_psi, -1.0, 1.0)))
    j_pixel = np.exp(np.interp(psi_deg, psi_grid, log_j_grid))

    dj_ddec = np.cos(dec_rad) * j_pixel.mean(axis=1) * 2.0 * np.pi
    j_total = float(np.trapezoid(dj_ddec, dec_rad))
    return dec_deg, dj_ddec, j_total


def zenith_exposure(
    latitude_deg: float,
    dec_deg: np.ndarray,
    dj_ddec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """J-weighted distribution of arrival zeniths at a site, over a sidereal day.

    A source at declination ``delta`` seen from latitude ``phi`` has

    .. math:: \\cos\\theta_z = \\sin\\phi\\,\\sin\\delta
        + \\cos\\phi\\,\\cos\\delta\\,\\cos H,

    with the hour angle ``H`` sweeping uniformly over a sidereal day. Binning
    that in ``cos(theta_z)``, weighted by ``dJ/d(dec)``, gives the fraction of
    the signal arriving from each band -- which is what the effective area has
    to be averaged over.

    At the South Pole ``cos(phi) = 0``, the hour angle drops out, and every
    declination maps to one fixed zenith, ``cos(theta_z) = -sin(delta)``. The
    Galactic Center at ``dec = -29 deg`` therefore sits permanently at
    ``cos(theta_z) = +0.48``, well above the horizon. At the Capo Passero
    latitude the same source spends most of the day below it.

    The convention matches example 30: ``cos(theta) = +1`` is a source at the
    zenith, so the neutrino travels straight down through the detector medium,
    and ``-1`` is a source at the nadir, straight up through the Earth.

    Parameters
    ----------
    latitude_deg : float
        Geographic latitude of the site [deg].
    dec_deg : np.ndarray
        Declination samples [deg], from :func:`sky_j_profile`.
    dj_ddec : np.ndarray
        ``dJ/d(dec)``, from :func:`sky_j_profile`.

    Returns
    -------
    cos_theta : np.ndarray, shape (N_COS_THETA,)
        Band centres in ``cos(theta_z)``, descending from near ``+1``.
    weights : np.ndarray, shape (N_COS_THETA,)
        J-weighted, time-averaged fraction of the signal in each band, summing
        to one.
    """
    phi = np.deg2rad(latitude_deg)
    delta = np.deg2rad(np.asarray(dec_deg, dtype=float))
    hour = np.linspace(0.0, 2.0 * np.pi, N_HOUR_ANGLE, endpoint=False)

    cos_theta_z = (
        np.sin(phi) * np.sin(delta)[:, None]
        + np.cos(phi) * np.cos(delta)[:, None] * np.cos(hour)[None, :]
    )
    # dJ/d(dec) is a density in declination, so it carries the local spacing.
    spacing = np.gradient(delta)
    sample_weight = np.repeat((dj_ddec * spacing)[:, None], N_HOUR_ANGLE, axis=1) / N_HOUR_ANGLE

    edges = np.linspace(1.0, -1.0, N_COS_THETA + 1)
    binned, _ = np.histogram(
        cos_theta_z.ravel(), bins=edges[::-1], weights=sample_weight.ravel()
    )
    weights = binned[::-1]
    return 0.5 * (edges[:-1] + edges[1:]), weights / weights.sum()


# ---------------------------------------------------------------------------
# Truncated first-passage range
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Sites
# ---------------------------------------------------------------------------


def build_sites(volume_km3: float) -> list[Site]:
    """The detectors this example compares, from :mod:`softpaws.detectors`."""
    gen2 = GEN2.replace(radius_km=float(np.sqrt(volume_km3 / (np.pi * GEN2.height_km))))
    # The three water sites below TRIDENT have no published effective area of
    # their own, so they are the instrumented footprint and nothing else --
    # which is all any site gets here, since the reach is not applied.
    return [gen2, ARCA230, TRIDENT, PONE, GVD]


# ---------------------------------------------------------------------------
# Effective area
# ---------------------------------------------------------------------------


def site_effective_area_cm2(
    site: Site,
    mass_gev: np.ndarray,
    cos_theta: np.ndarray,
    weights: np.ndarray,
    threshold_gev: float,
    truncate: bool = True,
) -> np.ndarray:
    """J- and exposure-weighted effective area for a line at ``E_nu = m_chi``.

    A tabulated line fixes the neutrino energy, so this is App. I's ``s -> 0``
    case throughout: no spectral weighting anywhere, and the length is the
    monochromatic first-passage range rather than ``1/Phi(A)``. The assembly is
    otherwise example 28's, with the neutral-current and tau ladders kept and
    each rung credited to the surface energy.

    Parameters
    ----------
    site : Site
        Detector geometry and location.
    mass_gev : np.ndarray, shape (n_mass,)
        Dark matter mass, equal to the neutrino energy [GeV].
    cos_theta, weights : np.ndarray, shape (N_COS_THETA,)
        Arrival-direction bands and their J-weighted exposure fractions.
    threshold_gev : float
        Muon selection threshold [GeV].
    truncate : bool, optional
        Whether to cut the first-passage integral at the available upstream
        column. ``False`` reproduces Eq. (16) as written, which is example 23's
        treatment and what isolates the size of the overburden effect.

    Returns
    -------
    aeff : np.ndarray, shape (n_mass,)
        Effective area [cm^2], already averaged over arrival direction with the
        Earth attenuation folded in.
    """
    theta_deg = np.rad2deg(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    neutrino_column, muon_column_km = site.columns(cos_theta)
    if not truncate:
        muon_column_km = np.full_like(muon_column_km, np.inf)
    n_nucleon = nucleon_number_density(site.density_g_cm3)

    total = np.zeros(np.size(mass_gev))
    channels = (
        ("mu", 1.0 - MEAN_INELASTICITY, 1.0),
        ("tau", MEAN_Z * (1.0 - MEAN_INELASTICITY), BR_TAU_TO_MU),
    )
    for flavour, muon_fraction, branching in channels:
        for i, mass in enumerate(np.atleast_1d(mass_gev)):
            rung_energy, rung_weight = flavour_transmission(
                float(mass), neutrino_column, CROSS_SECTION, flavour=flavour,
                n_grid=N_RUNG, decades=4.0,
            )
            muon_energy = muon_fraction * rung_energy
            # (n_rung, n_theta): each rung's muon under each band's overburden.
            length = truncated_muon_range_km(
                muon_energy[:, None], muon_column_km[None, :], threshold_gev,
                site.density_g_cm3,
            )
            radius = np.full(muon_energy.shape, site.radius_km)
            area_km2 = site.projected_area_km2(theta_deg[None, :], radius[:, None])
            volume_km3 = area_km2 * length + site.detector_volume_km3(radius)[:, None]
            rate = (
                n_nucleon
                * CROSS_SECTION.cc(rung_energy)[:, None]
                * volume_km3
                * CM_PER_KM**3
            )
            total[i] += branching * float(
                np.sum(weights[None, :] * rung_weight * rate)
            )
    return total


def sigma_v_sensitivity(
    mass_gev: np.ndarray,
    aeff_cm2: np.ndarray,
    livetime_s: float,
    j_total: float,
    line_fraction: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Invert the line-flux relation for the annihilation cross section.

    .. math:: \\langle\\sigma v\\rangle_\\mathrm{lim}
        = \\frac{12\\pi\\, m_\\chi^2\\, N_\\mathrm{lim}}
               {J\\, T\\, \\langle A_\\mathrm{eff}\\rangle},

    where the ``12 pi`` collects the Majorana ``1/(8 pi)``, the two neutrinos
    per annihilation, and the ``1/3`` muon-flavor fraction after oscillation
    over Galactic distances. The Earth attenuation that example 23 carried as a
    separate ``<D_nu>`` is inside ``<A_eff>`` here, since the transmission is
    evaluated per arrival direction along with everything else.

    Parameters
    ----------
    mass_gev : np.ndarray
        Dark matter mass [GeV].
    aeff_cm2 : np.ndarray
        Direction-averaged effective area [cm^2].
    livetime_s : float
        Exposure time [s].
    j_total : float
        All-sky halo J-factor [GeV^2 cm^-5].
    line_fraction : np.ndarray or float, optional
        Fraction of the neutrinos still at ``E_nu = m_chi``, from
        :func:`ew_line_survival`. Defaults to one, the tree-level line.

    Returns
    -------
    sigma_v : np.ndarray
        Upper limit on ``<sigma v>`` [cm^3 s^-1].
    """
    return (
        12.0 * np.pi * np.asarray(mass_gev, dtype=float) ** 2 * N_EVENTS_LIMIT
        / (j_total * livetime_s * aeff_cm2 * np.asarray(line_fraction, dtype=float))
    )


# ---------------------------------------------------------------------------
# Reporting and plotting
# ---------------------------------------------------------------------------


def report_exposure(site: Site, cos_theta: np.ndarray, weights: np.ndarray) -> None:
    """Print how much of the halo signal each site sees from above and below."""
    above = float(np.sum(weights[cos_theta > 0.0]))
    print(
        f"  {site.name:13s} lat {site.latitude_deg:+6.1f} deg: "
        f"{above:5.1%} of the J-weighted signal arrives from above the horizon, "
        f"{1.0 - above:5.1%} through the Earth"
    )


def report_halo_shift(j_total: float) -> None:
    """Price the halo profile against example 23's, which used a different one.

    The J-factor is the whole vertical normalization of the sensitivity, so a
    profile change is a rigid shift of every curve. Reporting it keeps the
    difference from example 23 visible instead of buried in a constant.
    """
    legacy = sky_j_profile(
        rho_sun_gev_cm3=LEGACY_HALO["rho"],
        gamma=LEGACY_HALO["gamma"],
        r_sun_kpc=LEGACY_HALO["r_sun"],
    )[2]
    print(
        f"  example 23 halo (gamma = {LEGACY_HALO['gamma']:.1f}, "
        f"rho_0 = {LEGACY_HALO['rho']:.1f} GeV cm^-3, "
        f"R_0 = {LEGACY_HALO['r_sun']:.1f} kpc): J = {legacy:.3g}"
    )
    print(
        f"  ratio {j_total / legacy:.2f}: every sensitivity here is that factor "
        f"stronger than example 23's for this reason alone"
    )


def report_sensitivities(mass_gev: np.ndarray, curves: dict[str, np.ndarray]) -> None:
    """Print the predicted sensitivities, and how the sites rank against each other."""
    print(f"\n{'m_chi [GeV]':>12} " + " ".join(f"{n:>14}" for n in curves))
    for mass in (1.0e5, 1.0e6, 1.0e7, 1.0e8, 1.0e9):
        i = int(np.argmin(np.abs(mass_gev - mass)))
        row = " ".join(f"{c[i]:14.3g}" for c in curves.values())
        print(f"{mass_gev[i]:12.3g} {row}")
    print("  Upper limits on <sigma v> in cm^3 s^-1; smaller is stronger.")

    # Ranked against the strongest site, which is the only comparison the
    # geometry supports: same halo, same livetime, same selection assumptions.
    stack = np.vstack(list(curves.values()))
    best = np.argmin(np.median(stack, axis=1))
    reference = list(curves)[int(best)]
    print(f"\n  Relative to {reference}, median over the range:")
    for name, curve in curves.items():
        ratio = curve / curves[reference]
        print(f"    {name:14s} {np.median(ratio):5.2f}x  "
              f"({ratio.min():.2f} to {ratio.max():.2f})")


def report_truncation(
    site: Site,
    mass_gev: np.ndarray,
    truncated: np.ndarray,
    free: np.ndarray,
) -> None:
    """Print what the finite overburden costs, which is the change that matters.

    Example 23 integrated the first-passage probability to infinite depth for
    every direction. That is right for an upgoing signal, where the muon is born
    in rock that never runs out, and wrong for this one: the Galactic Center is
    above the horizon at the South Pole, so the muon has only the ice overburden
    to be born in.
    """
    ratio = truncated / free
    print(f"\n  {site.name}: cost of the finite upstream column")
    for mass in (1.0e5, 1.0e6, 1.0e7, 1.0e8):
        i = int(np.argmin(np.abs(mass_gev - mass)))
        print(
            f"    m_chi = {mass_gev[i]:8.3g} GeV   sensitivity weaker by "
            f"{ratio[i]:5.2f}x"
        )


def make_figure(
    mass_gev: np.ndarray,
    curves: dict[str, np.ndarray],
    sites: list[Site],
    livetime_yr: float,
    out_path: pathlib.Path,
) -> None:
    """Draw the single sensitivity panel.

    One curve per site: the parameter-free instrumented footprint. Neither a
    published curve nor a light-reach band is overlaid; see the module docstring
    for why in each case.

    Parameters
    ----------
    mass_gev : np.ndarray
        Dark matter masses [GeV].
    curves : dict
        Site name -> sensitivity [cm^3 s^-1].
    sites : list of Site
        Used for the colours and for the annotation.
    livetime_yr : float
        Exposure, for the annotation.
    out_path : pathlib.Path
        Destination; both PDF and PNG are written.
    """
    colors = SITE_COLORS
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.6, 3.6))

        for name, curve in curves.items():
            ax.plot(mass_gev, curve, color=colors[name], lw=1.5, label=name)

        # Below the atmospheric top a real search is background limited, so a
        # zero-background ceiling is not a sensitivity anyone would quote there.
        ax.axvspan(mass_gev.min(), ATMOSPHERIC_TOP_GEV, color="0.88", zorder=0, lw=0)
        ax.text(
            np.sqrt(mass_gev.min() * ATMOSPHERIC_TOP_GEV), 0.5,
            "atmospheric\nbackground", transform=ax.get_xaxis_transform(),
            ha="center", va="center", fontsize=5.5, color="0.45", rotation=90,
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$m_\chi$ [GeV]")
        ax.set_ylabel(r"$\langle \sigma v \rangle$ [cm$^3$ s$^{-1}$]")
        ax.set_xlim(mass_gev.min(), mass_gev.max())
        ax.legend(fontsize=6, loc="upper left", frameon=False)
        # Three stacked lines rather than one multi-line string, so the gap
        # under the channel can be set independently of the line spacing.
        ax.text(0.97, 0.155, "background-free prediction", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6, color="0.35")
        ax.text(0.97, 0.085, rf"$\chi\chi \to \nu\bar{{\nu}}$, {livetime_yr:.0f} yr",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6)
        ax.text(0.97, 0.02, rf"gNFW $\gamma={GAMMA_SLOPE}$, tracks only",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6)

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png"):
            path = out_path.with_suffix(suffix)
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {path.resolve()}")
        plt.close(fig)


def main() -> None:
    args = parse_args()

    mass_gev = np.logspace(
        np.log10(MASS_RANGE_GEV[0]), np.log10(MASS_RANGE_GEV[1]), N_MASS
    )
    print(f"Mass grid: {mass_gev.size} points, "
          f"m_chi = {mass_gev[0]:.3g} to {mass_gev[-1]:.3g} GeV")

    print(
        f"\nComputing halo J-factor (generalized NFW, gamma = {GAMMA_SLOPE}, "
        f"rho_0 = {RHO_SUN_GEV_CM3} GeV cm^-3, R_0 = {R_SUN_KPC} kpc) ..."
    )
    dec_deg, dj_ddec, j_total = sky_j_profile()
    print(f"  all-sky J = {j_total:.3g} GeV^2 cm^-5")
    report_halo_shift(j_total)

    print("\nChecking Eq. I.2 (App. I) at s=0 against the plain J-factor formula ...")
    check_appendix_i(mass_gev)

    line_fraction = (
        np.ones_like(mass_gev) if args.no_electroweak else ew_line_survival(mass_gev)
    )
    if args.no_electroweak:
        print("\nElectroweak depletion of the line: disabled (--no-electroweak).")
    else:
        print("\nElectroweak depletion of the line (HDMSpectra Eq. B.7, T = 1/2):")
        for mass in (1.0e5, 1.0e6, 1.0e7, 1.0e8, 1.0e9, 1.0e10):
            i = int(np.argmin(np.abs(mass_gev - mass)))
            print(
                f"    m_chi = {mass_gev[i]:8.3g} GeV   {line_fraction[i]:.3f} of the "
                f"neutrinos stay on the line"
            )
        print("  The continuum below the line is discarded, so every curve is "
              "conservative.")

    sites = build_sites(args.volume_km3)
    livetime_s = args.livetime_yr * 365.25 * 86400.0
    print(f"\nArrival geometry of the halo signal ({args.livetime_yr:.0f} yr each):")
    exposures = {}
    for site in sites:
        cos_theta, weights = zenith_exposure(site.latitude_deg, dec_deg, dj_ddec)
        exposures[site.name] = (cos_theta, weights)
        report_exposure(site, cos_theta, weights)

    curves: dict[str, np.ndarray] = {}
    for site in sites:
        cos_theta, weights = exposures[site.name]
        print(f"\nBuilding {site.name} effective area (R_det = {site.radius_km:.2f} km) ...")
        aeff = site_effective_area_cm2(
            site, mass_gev, cos_theta, weights, args.threshold_gev
        )
        curves[site.name] = sigma_v_sensitivity(
            mass_gev, aeff, livetime_s, j_total, line_fraction
        )

    report_sensitivities(mass_gev, curves)

    print("\nWhat the finite upstream column costs, site by site:")
    for site in sites:
        cos_theta, weights = exposures[site.name]
        free_aeff = site_effective_area_cm2(
            site, mass_gev, cos_theta, weights, args.threshold_gev, truncate=False
        )
        free = sigma_v_sensitivity(
            mass_gev, free_aeff, livetime_s, j_total, line_fraction
        )
        report_truncation(site, mass_gev, curves[site.name], free)

    make_figure(mass_gev, curves, sites, args.livetime_yr, args.out)


if __name__ == "__main__":
    main()
