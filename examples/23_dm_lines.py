"""Example 23 -- Dark matter neutrino lines: soft-volume sensitivity for IceCube-Gen2.

Applies the soft-volume forward model to a *line* signal rather than the diffuse
power law of examples 12--14: dark matter annihilating in the Galactic halo
through ``chi chi -> nu nubar`` puts all of its neutrinos at a single energy
``E_nu = m_chi``, so the whole forward model collapses onto the effective area
evaluated at one point.

The prediction is built entirely from package pieces, with no tuned
normalization:

* **Effective area** -- the range-based, flux-independent
  :meth:`~softpaws.response.soft_volume.SoftVolumeResponse.threshold_effective_area_cm2`
  is the right convention here for the same reason it was in example 20: a line
  fixes the *neutrino* energy and the analysis accepts every muon above
  threshold, so the target volume is the muon range down to threshold, not the
  spectrally weighted soft volume. The detector is scaled to IceCube-Gen2 by
  instrumented volume (:func:`~softpaws.transport.soft_volume.sphere_radius_from_volume`).
* **Cross section** -- the tabulated BGR18 calculation
  (:func:`~softpaws.transport.cross_section.bgr18_cross_section`), not the
  analytic power law. The published curve runs to 7.4 EeV, far outside the range
  where a single power law is usable.
* **Earth attenuation** -- the halo signal is not isotropic, so ``D_nu`` is
  averaged over the sky with the *J-factor* as the weight
  (:func:`~softpaws.transport.attenuation.prem_column`). This matters more than
  it does for a diffuse flux: the Galactic Center sits at ``dec ~ -29 deg``,
  which is downgoing at the South Pole and so unabsorbed, and the J-weighted
  survival probability saturates near that downgoing fraction instead of falling
  to zero.

App. I of ``docs/2026_softvolume.pdf`` derives the DM-line rate from the same
``Phi(s)``-weighted line-of-sight propagator as everything else in the
package, evaluated at ``s = 0`` (a monochromatic line has no continuum
spectral index to average over, and ``Phi(0) = 0`` identically). This example
now makes that connection explicit and checked, not just asserted in prose:
:func:`dm_line_flux_general_s` implements Eq. I.2 literally, at general
``s``, and ``main()`` verifies its ``s = 0`` value matches the plain J-factor
formula (:func:`los_j_factor`) to numerical precision. It also plots a second
effective-area convention, :func:`sigma_v_sensitivity_soft_volume`, built
from the ``s = 0`` instance of the near-detector soft volume
(:func:`~softpaws.transport.soft_volume.dm_line_target_volume_km3`) rather
than the muon-range convention above, so the two conventions' disagreement is
visible on the same plot rather than only discussed in comments.

With a fixed signal requirement of :data:`N_EVENTS_LIMIT` events, the sensitivity
follows from the line flux (Eq. 1 below) by inversion,

.. math:: \\langle\\sigma v\\rangle_\\mathrm{lim}
    = \\frac{12\\pi\\, m_\\chi^2\\, N_\\mathrm{lim}}
           {J\\, T\\, A_\\mathrm{eff}(m_\\chi)\\, \\langle D_\\nu\\rangle(m_\\chi)},

where the ``12 pi`` collects the Majorana ``1/(8 pi)``, the two neutrinos per
annihilation, and the ``1/3`` muon-flavor fraction after oscillation over
Galactic distances.

The comparison curve is the published IceCube-Gen2 projected sensitivity shipped
in ``src/softpaws/data/bounds/`` and read through
:func:`~softpaws.data.loader.load_dm_line_bounds`, so both curves are for the
same detector. Panel (b) reports the residual as the *implied* event
requirement, ``N_lim x (published / analytic)`` -- the number of soft-volume
tracks the published curve behaves as though it demanded. Read against the
zero-background floor :data:`N_EVENTS_LIMIT` it separates into two regimes:

* **above the floor** (below ~4e8 GeV) the published curve is weaker than a
  background-free track count would allow, by ~1e3 at 20 TeV falling to ~1 near
  the crossing. That is the expected cost of the atmospheric background of
  example 22 plus the selection efficiency of example 20, both of which shrink
  with energy -- so the residual's slope is largely those two effects, not a
  failure of the transport.
* **below the floor** (above ~4e8 GeV) the published curve is *stronger* than
  the model can produce from any number of tracks, which no efficiency can
  explain. This is the signature of sensitivity the soft volume does not
  describe: IceCube-Gen2's projected EeV reach comes from its radio array and
  from all-flavor cascades, neither of which is a through-going muon in the
  optical in-ice array that this model represents.

So the comparison is only a like-for-like test of the transport in the middle of
the range; at the ends it measures what the published analysis has that this
forward model does not.

Further caveats: the analysis is treated as background free at every mass;
the tabulated cross section is ``nu_mu`` on a proton target, so nu/nubar and
isospin differences are neglected; and the ``D_nu = 1`` curve is drawn to show
that Earth attenuation is nearly irrelevant here -- unlike the diffuse upgoing
samples of example 14 -- because the Galactic Center is downgoing at the South
Pole and dominates the J-factor.

Usage
-----
    python examples/23_dm_lines.py
    python examples/23_dm_lines.py --volume-km3 8.0 --livetime-yr 10.0
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.data.loader import load_dm_line_bounds
from softpaws.response.soft_volume import SoftVolumeResponse
from softpaws.transport.attenuation import prem_column, survival_probability
from softpaws.transport.cross_section import bgr18_cross_section
from softpaws.transport.eigenvalue import phi_eigenvalue
from softpaws.transport.soft_volume import (
    DEFAULT_MUON_THRESHOLD_GEV,
    dm_line_target_volume_km3,
    sphere_radius_from_volume,
)
from softpaws.transport.source import inelasticity_factor, nucleon_number_density
from softpaws.utils.constants import CM_PER_KM

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"

# IceCube-Gen2 baseline: ~8 km^3 instrumented, 10 years of data.
GEN2_VOLUME_KM3 = 8.0
GEN2_LIVETIME_YR = 10.0

# Navarro-Frenk-White halo, normalized to the local density at the solar radius.
RHO_SUN_GEV_CM3 = 0.3
R_SUN_KPC = 8.5
R_SCALE_KPC = 20.0
R_VIRIAL_KPC = 200.0
KPC_TO_CM = 3.0856775814913673e21
KPC_TO_KM = KPC_TO_CM / CM_PER_KM

# Galactic Center in equatorial (J2000) coordinates. Its negative declination is
# the reason Earth attenuation is mild for this signal at the South Pole.
GC_RA_DEG = 266.405
GC_DEC_DEG = -28.936

# Feldman-Cousins 90% CL upper limit on the signal for zero observed background.
N_EVENTS_LIMIT = 2.44

# Near-detector ice column for the soft-volume (Eq. I.2 / App. I) sensitivity
# curve: the Galactic Center is downgoing at the South Pole (GC_DEC_DEG < 0),
# so this is a couple km of ice overburden, not the full Earth chord -- see
# docs/exact_soft_volume_notes.md's "IceCube sits ~1.95 km deep."
ICE_COLUMN_KM = 1.95

CROSS_SECTION = bgr18_cross_section()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--volume-km3",
        type=float,
        default=GEN2_VOLUME_KM3,
        help="Instrumented volume of the detector [km^3].",
    )
    parser.add_argument(
        "--livetime-yr",
        type=float,
        default=GEN2_LIVETIME_YR,
        help="Exposure time [yr].",
    )
    parser.add_argument(
        "--threshold-gev",
        type=float,
        default=DEFAULT_MUON_THRESHOLD_GEV,
        help="Muon selection threshold [GeV].",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=_DEFAULT_OUT_DIR / "23_dm_lines.pdf",
        help="Output file for the figure.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Halo J-factor
# ---------------------------------------------------------------------------


def nfw_density(radius_kpc: float | np.ndarray) -> np.ndarray:
    """NFW dark matter density profile.

    Parameters
    ----------
    radius_kpc : float or np.ndarray
        Galactocentric radius [kpc].

    Returns
    -------
    density : np.ndarray
        Dark matter mass density [GeV cm^-3], zero beyond
        :data:`R_VIRIAL_KPC`.
    """
    radius = np.atleast_1d(np.asarray(radius_kpc, dtype=float))
    rho_s = RHO_SUN_GEV_CM3 * (R_SUN_KPC / R_SCALE_KPC) * (1.0 + R_SUN_KPC / R_SCALE_KPC) ** 2
    # Regulate the r -> 0 cusp; the enclosed J-factor converges regardless.
    x = np.maximum(radius, 1.0e-3) / R_SCALE_KPC
    density = rho_s / (x * (1.0 + x) ** 2)
    return np.where(radius <= R_VIRIAL_KPC, density, 0.0)


def los_j_factor(psi_deg: float | np.ndarray, n_steps: int = 512) -> np.ndarray:
    """Line-of-sight annihilation J-factor at an angle from the Galactic Center.

    Parameters
    ----------
    psi_deg : float or np.ndarray
        Angular separation from the Galactic Center [deg].
    n_steps : int, optional
        Number of trapezoidal steps along the line of sight. Defaults to 512.

    Returns
    -------
    j_los : np.ndarray
        ``\\int dl \\rho^2`` along the line of sight [GeV^2 cm^-5].
    """
    psi_rad = np.deg2rad(np.atleast_1d(np.asarray(psi_deg, dtype=float)))
    # Path length at which the line of sight leaves the virial sphere.
    l_max = R_SUN_KPC * np.cos(psi_rad) + np.sqrt(
        R_VIRIAL_KPC**2 - (R_SUN_KPC * np.sin(psi_rad)) ** 2
    )
    fraction = np.linspace(0.0, 1.0, n_steps)
    l_kpc = l_max[:, None] * fraction[None, :]
    r_kpc = np.sqrt(R_SUN_KPC**2 + l_kpc**2 - 2.0 * R_SUN_KPC * l_kpc * np.cos(psi_rad)[:, None])
    integrand = nfw_density(r_kpc) ** 2
    return np.trapezoid(integrand, l_kpc, axis=1) * KPC_TO_CM


def dm_line_flux_general_s(
    psi_deg: float | np.ndarray,
    sigma_v_cm3_s: float,
    mass_gev: float,
    spectral_index_s: float = 0.0,
    b_mu: float = 1.0,
    d_mu: float = 0.1,
    n_steps: int = 512,
) -> np.ndarray:
    """Literal Eq. I.2 (App. I) evaluation of the DM-line flux, at general ``s``.

    .. math:: \\frac{dN}{dE\\,d\\Omega} = \\frac{\\langle\\sigma v\\rangle}
        {2 m_\\chi^2}\\, \\mathcal{I}(s)
        \\int d\\ell\\, \\rho^2(r(\\psi,\\ell))\\, e^{-\\ell\\,\\Phi(s)}.

    At the physical value ``s = 0`` -- the only value that makes sense for a
    genuine monochromatic line, which has no continuum spectral index to
    average over -- ``Phi(0) = 0`` identically
    (:func:`softpaws.transport.eigenvalue.phi_eigenvalue`) regardless of
    ``b_mu``/``d_mu``, and ``I(0) = 1``, so this reduces exactly to
    ``(sigma_v / 2 m_chi^2) * los_j_factor(psi)`` -- the ordinary,
    unattenuated J-factor formula already used throughout indirect-detection
    literature and already computed by :func:`los_j_factor`. That equivalence
    is checked explicitly in ``main()`` below, not just asserted.

    ``ell`` is converted from kpc to km to match ``Phi``'s km^-1 convention
    (the same length unit used everywhere else this package evaluates
    ``Phi``), so this function stays dimensionally honest away from ``s = 0``
    too: for any nonzero ``s``, ``e^{-ell Phi(s)}`` is astronomically small
    over kpc-scale distances (``Phi ~ O(0.1-1) km^-1``), which is *why* the
    line-of-sight propagator is only ever evaluated at ``s = 0`` in practice
    -- not a physical convention choice, but a mathematical consequence of
    the length scales involved. ``b_mu``, ``d_mu`` therefore only matter for
    exercising the formula's general-``s`` shape; the ``s = 0`` evaluation
    used everywhere else in this example does not depend on them at all.

    Parameters
    ----------
    psi_deg : float or np.ndarray
        Angular separation from the Galactic Center [deg].
    sigma_v_cm3_s : float
        Velocity-averaged annihilation cross section [cm^3 s^-1].
    mass_gev : float
        Dark matter mass ``m_chi`` [GeV]; the line sits at ``E_nu = m_chi``.
    spectral_index_s : float, optional
        Mellin index ``s`` at which to evaluate Eq. I.2. Defaults to ``0``,
        the physical value for a monochromatic line.
    b_mu, d_mu : float, optional
        Muon drift/diffusion coefficients [km^-1] feeding ``Phi(s)``; inert
        at ``s = 0`` (see above). Defaults are placeholders, not physical
        values -- only their sign and positivity matter away from ``s = 0``.
    n_steps : int, optional
        Number of trapezoidal steps along the line of sight. Defaults to 512.

    Returns
    -------
    flux : np.ndarray
        ``dN / (dE dOmega)`` [GeV^-1 sr^-1 ... via ``sigma_v``'s cm^3 s^-1],
        same convention as :func:`los_j_factor` scaled by the annihilation
        prefactor.
    """
    phi_s = float(phi_eigenvalue(spectral_index_s, b_mu, d_mu))
    psi_rad = np.deg2rad(np.atleast_1d(np.asarray(psi_deg, dtype=float)))
    l_max = R_SUN_KPC * np.cos(psi_rad) + np.sqrt(
        R_VIRIAL_KPC**2 - (R_SUN_KPC * np.sin(psi_rad)) ** 2
    )
    fraction = np.linspace(0.0, 1.0, n_steps)
    l_kpc = l_max[:, None] * fraction[None, :]
    r_kpc = np.sqrt(R_SUN_KPC**2 + l_kpc**2 - 2.0 * R_SUN_KPC * l_kpc * np.cos(psi_rad)[:, None])
    l_km = l_kpc * KPC_TO_KM
    integrand = nfw_density(r_kpc) ** 2 * np.exp(-l_km * phi_s)
    j_weighted = np.trapezoid(integrand, l_kpc, axis=1) * KPC_TO_CM
    factor = float(inelasticity_factor(spectral_index_s))
    return (sigma_v_cm3_s / (2.0 * mass_gev**2)) * factor * j_weighted


def sky_j_profile(
    n_dec: int = 181,
    n_ra: int = 360,
    n_psi: int = 400,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Declination profile of the halo J-factor and its all-sky total.

    Averages the line-of-sight J-factor over right ascension at each
    declination, which is all the direction dependence the calculation needs:
    Earth attenuation depends on declination alone.

    Parameters
    ----------
    n_dec : int, optional
        Number of declination samples. Defaults to 181.
    n_ra : int, optional
        Number of right-ascension samples used for the average. Defaults to 360.
    n_psi : int, optional
        Number of points on the interpolation grid in angular separation from
        the Galactic Center. Defaults to 400.

    Returns
    -------
    dec_deg : np.ndarray, shape (n_dec,)
        Declination samples [deg].
    dj_ddec : np.ndarray, shape (n_dec,)
        J-factor per unit declination, ``dJ/d(dec)`` [GeV^2 cm^-5 rad^-1],
        including the ``cos(dec)`` solid-angle weight.
    j_total : float
        All-sky J-factor ``\\int d\\Omega \\int dl \\rho^2`` [GeV^2 cm^-5].
    """
    # The J-factor spans orders of magnitude between the Galactic Center and the
    # anticenter, so tabulate it on a log-spaced grid in psi and interpolate the
    # logarithm rather than evaluating the line-of-sight integral per sky pixel.
    psi_grid = np.concatenate(([0.0], np.logspace(-2.0, np.log10(180.0), n_psi)))
    log_j_grid = np.log(los_j_factor(psi_grid))

    dec_deg = np.linspace(-90.0, 90.0, n_dec)
    ra_deg = np.linspace(0.0, 360.0, n_ra, endpoint=False)
    dec_rad, ra_rad = np.deg2rad(dec_deg), np.deg2rad(ra_deg)
    gc_dec_rad, gc_ra_rad = np.deg2rad(GC_DEC_DEG), np.deg2rad(GC_RA_DEG)

    cos_psi = np.sin(dec_rad)[:, None] * np.sin(gc_dec_rad) + np.cos(dec_rad)[:, None] * np.cos(
        gc_dec_rad
    ) * np.cos(ra_rad[None, :] - gc_ra_rad)
    psi_deg = np.rad2deg(np.arccos(np.clip(cos_psi, -1.0, 1.0)))
    j_pixel = np.exp(np.interp(psi_deg, psi_grid, log_j_grid))

    # dJ/d(dec) = cos(dec) * \int dRA <dJ/dOmega>, with the RA integral done as a
    # mean over the uniform grid times its 2 pi span.
    dj_ddec = np.cos(dec_rad) * j_pixel.mean(axis=1) * 2.0 * np.pi
    j_total = float(np.trapezoid(dj_ddec, dec_rad))
    return dec_deg, dj_ddec, j_total


def j_weighted_survival(
    energy_gev: np.ndarray,
    dec_deg: np.ndarray,
    dj_ddec: np.ndarray,
) -> np.ndarray:
    """Earth survival probability averaged over the sky with the J-factor weight.

    Parameters
    ----------
    energy_gev : np.ndarray, shape (n_energy,)
        Neutrino energy [GeV].
    dec_deg : np.ndarray, shape (n_dec,)
        Declination samples [deg], as returned by :func:`sky_j_profile`.
    dj_ddec : np.ndarray, shape (n_dec,)
        J-factor per unit declination, as returned by :func:`sky_j_profile`.

    Returns
    -------
    d_nu : np.ndarray, shape (n_energy,)
        J-weighted mean survival probability at each energy, in ``[0, 1]``.
    """
    columns = np.array([prem_column(d) for d in dec_deg])
    survival = survival_probability(
        np.asarray(energy_gev, dtype=float)[:, None],
        columns[None, :],
        cross_section=CROSS_SECTION,
    )
    dec_rad = np.deg2rad(dec_deg)
    weighted = np.trapezoid(survival * dj_ddec[None, :], dec_rad, axis=1)
    return weighted / np.trapezoid(dj_ddec, dec_rad)


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------


def sigma_v_sensitivity(
    mass_gev: np.ndarray,
    response: SoftVolumeResponse,
    livetime_s: float,
    j_total: float,
    survival: np.ndarray,
    threshold_gev: float = DEFAULT_MUON_THRESHOLD_GEV,
) -> np.ndarray:
    """Analytic ``<sigma v>`` sensitivity for a monoenergetic neutrino line.

    Inverts the line-flux relation for the annihilation cross section that would
    produce :data:`N_EVENTS_LIMIT` detected tracks.

    Parameters
    ----------
    mass_gev : np.ndarray, shape (n,)
        Dark matter mass ``m_chi`` [GeV]; the line sits at ``E_nu = m_chi``.
    response : softpaws.response.soft_volume.SoftVolumeResponse
        Detector response supplying the effective area.
    livetime_s : float
        Exposure time [s].
    j_total : float
        All-sky halo J-factor [GeV^2 cm^-5].
    survival : np.ndarray, shape (n,)
        J-weighted Earth survival probability at each mass.
    threshold_gev : float, optional
        Muon selection threshold [GeV]. Defaults to
        :data:`~softpaws.transport.soft_volume.DEFAULT_MUON_THRESHOLD_GEV`.

    Returns
    -------
    sigma_v : np.ndarray, shape (n,)
        Upper limit on ``<sigma v>`` [cm^3 s^-1].

    Notes
    -----
    The ``12 pi`` prefactor is ``8 pi`` for self-conjugate (Majorana) dark
    matter, divided by the two neutrinos produced per annihilation and
    multiplied by three for the ``nu_mu`` fraction of an equal-flavor mix at
    Earth.
    """
    aeff_cm2 = response.threshold_effective_area_cm2(mass_gev, threshold_gev)
    return (
        12.0
        * np.pi
        * mass_gev**2
        * N_EVENTS_LIMIT
        / (j_total * livetime_s * aeff_cm2 * survival)
    )


def sigma_v_sensitivity_soft_volume(
    mass_gev: np.ndarray,
    radius_km: float,
    livetime_s: float,
    j_total: float,
    survival: np.ndarray,
    column_depth_km: float = ICE_COLUMN_KM,
) -> np.ndarray:
    """``<sigma v>`` sensitivity using the ``s = 0`` soft-volume convention (App. I).

    Same inversion as :func:`sigma_v_sensitivity`, but with the effective area
    built from :func:`~softpaws.transport.soft_volume.dm_line_target_volume_km3`
    (the literal ``s -> 0`` instance of the eigenvalue formalism) instead of
    the muon-range convention (:func:`~softpaws.transport.soft_volume.
    range_target_volume_km3`, via :meth:`~softpaws.response.soft_volume.
    SoftVolumeResponse.threshold_effective_area_cm2`). The two are expected to
    diverge, most visibly at high mass: the soft-volume term grows *linearly*
    in ``column_depth_km`` while the muon range grows only logarithmically in
    energy, and -- per ``dm_line_target_volume_km3``'s own docstring -- the
    ``s = 0`` treatment has no threshold cutoff at all, so it does not fall
    off the way a real through-going selection would at low mass either. This
    curve exists to make that divergence visible, not to replace the
    muon-range curve as the recommended convention.

    Parameters
    ----------
    mass_gev : np.ndarray, shape (n,)
        Dark matter mass ``m_chi`` [GeV].
    radius_km : float
        Detector radius [km].
    livetime_s : float
        Exposure time [s].
    j_total : float
        All-sky halo J-factor [GeV^2 cm^-5].
    survival : np.ndarray, shape (n,)
        J-weighted Earth survival probability at each mass.
    column_depth_km : float, optional
        Near-detector column depth [km]. Defaults to :data:`ICE_COLUMN_KM`.

    Returns
    -------
    sigma_v : np.ndarray, shape (n,)
        Upper limit on ``<sigma v>`` [cm^3 s^-1].
    """
    volume_cm3 = dm_line_target_volume_km3(radius_km, column_depth_km) * CM_PER_KM**3
    aeff_cm2 = volume_cm3 * nucleon_number_density() * CROSS_SECTION.cc(mass_gev)
    return (
        12.0
        * np.pi
        * mass_gev**2
        * N_EVENTS_LIMIT
        / (j_total * livetime_s * aeff_cm2 * survival)
    )


def make_figure(
    mass_gev: np.ndarray,
    analytic: np.ndarray,
    analytic_no_atten: np.ndarray,
    soft_volume_curve: np.ndarray,
    published: np.ndarray,
    out_path: pathlib.Path,
) -> None:
    """Draw the sensitivity comparison and its residual.

    Parameters
    ----------
    mass_gev : np.ndarray, shape (n,)
        Dark matter masses [GeV].
    analytic : np.ndarray, shape (n,)
        Muon-range-convention sensitivity including Earth attenuation
        [cm^3 s^-1] (:func:`sigma_v_sensitivity`).
    analytic_no_atten : np.ndarray, shape (n,)
        Muon-range-convention sensitivity with ``D_nu = 1`` [cm^3 s^-1].
    soft_volume_curve : np.ndarray, shape (n,)
        ``s = 0`` soft-volume-convention sensitivity, including Earth
        attenuation (:func:`sigma_v_sensitivity_soft_volume`).
    published : np.ndarray, shape (n,)
        Published IceCube-Gen2 sensitivity on the same mass grid [cm^3 s^-1].
    out_path : pathlib.Path
        Destination for the figure; both PDF and PNG are written.
    """
    with plt.style.context(str(_STYLE)):
        fig, (ax, ax_ratio) = plt.subplots(
            2, 1, figsize=(3.4, 4.4), sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1.0]},
        )

        ax.plot(mass_gev, analytic, color="C0", lw=1.4,
                label=r"muon range, $\langle D_\nu\rangle$")
        ax.plot(mass_gev, analytic_no_atten, color="C0", lw=1.0, ls=":",
                label=r"muon range, $D_\nu=1$")
        ax.plot(mass_gev, soft_volume_curve, color="C2", lw=1.2, ls="-.",
                label=r"soft volume ($s=0$), $\langle D_\nu\rangle$")
        ax.plot(mass_gev, published, color="k", lw=1.4, ls="--",
                label="IceCube-Gen2 (published)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(r"$\langle \sigma v \rangle$ [cm$^3$ s$^{-1}$]")
        ax.legend(fontsize=6, loc="upper left")
        ax.text(0.97, 0.05, r"$\chi\chi \to \nu\bar{\nu}$, NFW halo",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6)

        implied = N_EVENTS_LIMIT * published / analytic
        ax_ratio.plot(mass_gev, implied, color="C1", lw=1.4)
        ax_ratio.axhline(N_EVENTS_LIMIT, color="0.5", lw=0.8, ls=":")
        # Below the zero-background floor no track count can reproduce the
        # published curve, so the shaded band flags where it must come from
        # channels outside this model (radio array, all-flavor cascades).
        ax_ratio.axhspan(implied.min() * 0.5, N_EVENTS_LIMIT, color="0.85", zorder=0)
        ax_ratio.set_xscale("log")
        ax_ratio.set_yscale("log")
        ax_ratio.set_ylim(implied.min() * 0.5, implied.max() * 2.0)
        ax_ratio.set_xlabel(r"$m_\chi$ [GeV]")
        ax_ratio.set_ylabel(r"implied $N_\mathrm{lim}$")
        ax_ratio.text(0.03, 0.08, "not reachable by tracks", transform=ax_ratio.transAxes,
                      ha="left", va="bottom", fontsize=5, color="0.35")

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png"):
            path = out_path.with_suffix(suffix)
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {path.resolve()}")
        plt.close(fig)


def main() -> None:
    args = parse_args()

    print("Loading published IceCube-Gen2 line sensitivity ...")
    mass_gev, published = load_dm_line_bounds("icecubegen2")
    print(f"  {mass_gev.size} points, m_chi = {mass_gev[0]:.3g} to {mass_gev[-1]:.3g} GeV")

    print("Computing NFW halo J-factor ...")
    dec_deg, dj_ddec, j_total = sky_j_profile()
    print(f"  all-sky J = {j_total:.3g} GeV^2 cm^-5")

    print("Averaging Earth attenuation over the sky ...")
    survival = j_weighted_survival(mass_gev, dec_deg, dj_ddec)
    print(f"  <D_nu>: {survival[0]:.3f} at {mass_gev[0]:.3g} GeV "
          f"-> {survival[-1]:.3f} at {mass_gev[-1]:.3g} GeV")

    radius_km = sphere_radius_from_volume(args.volume_km3)
    livetime_s = args.livetime_yr * 365.25 * 86400.0
    print(f"Building IceCube-Gen2 response: V = {args.volume_km3:.1f} km^3 "
          f"(R = {radius_km:.2f} km), T = {args.livetime_yr:.1f} yr")
    response = SoftVolumeResponse(radius_km=radius_km, cross_section=CROSS_SECTION)

    analytic = sigma_v_sensitivity(
        mass_gev, response, livetime_s, j_total, survival, args.threshold_gev,
    )
    analytic_no_atten = sigma_v_sensitivity(
        mass_gev, response, livetime_s, j_total, np.ones_like(mass_gev), args.threshold_gev,
    )

    ratio = published / analytic
    print(f"  published / analytic: median {np.median(ratio):.2f}, "
          f"range {ratio.min():.2f} to {ratio.max():.2f}")
    print(f"  implied N_lim: median {N_EVENTS_LIMIT * np.median(ratio):.1f} "
          f"(analytic assumes {N_EVENTS_LIMIT})")

    print("Checking Eq. I.2 (App. I) at s=0 against the plain J-factor formula ...")
    sigma_v_probe = 1.0e-23  # arbitrary; the check is on the ratio, not the scale
    check_mass = mass_gev[[0, mass_gev.size // 2, -1]]
    # dm_line_flux_general_s is evaluated at a fixed sky angle (looking
    # straight at the Galactic Center, psi=0), sweeping the probe masses.
    psi_probe = np.zeros_like(check_mass)
    direct = dm_line_flux_general_s(psi_probe, sigma_v_probe, check_mass, spectral_index_s=0.0)
    reference = sigma_v_probe / (2.0 * check_mass**2) * los_j_factor(0.0)[0]
    max_rel_err = float(np.max(np.abs(direct / reference - 1.0)))
    print(f"  max |Eq. I.2(s=0) / plain J-factor - 1| = {max_rel_err:.2e} (should be ~0)")

    print("Computing the s=0 soft-volume sensitivity curve (App. I) ...")
    soft_volume_curve = sigma_v_sensitivity_soft_volume(
        mass_gev, radius_km, livetime_s, j_total, survival,
    )
    sv_ratio = soft_volume_curve / analytic
    print(f"  soft-volume(s=0) / muon-range: median {np.median(sv_ratio):.2f}, "
          f"range {sv_ratio.min():.2f} to {sv_ratio.max():.2f}")

    make_figure(mass_gev, analytic, analytic_no_atten, soft_volume_curve, published, args.out)


if __name__ == "__main__":
    main()
