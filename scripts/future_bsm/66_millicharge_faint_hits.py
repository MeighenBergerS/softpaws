"""Example 66 -- millicharged particles the way IceCube can actually see them.

Example 65's plane is the heavy Drell-Yan corner; the corner where the
world's constraints live -- and where IceCube has published nothing yet --
is MeV-GeV masses at small charge, with *meson-decay* production and a
*faint-hit* signature. This example forecasts that analysis: the same
Z-moment production algebra as example 63 with the Drell-Yan loop replaced
by neutral-meson decays, and a multiple-scattering detection model built
for DeepCore's Faint Particle Trigger.

**Production** (channels as in Arguelles, Kelly, Munoz, arXiv:2104.13924).
Every neutral meson with ``m_meson > 2 m_chi`` feeds the flux: Dalitz-like
three-body decays of ``pi^0`` and ``eta`` (branching
``eps^2 x BR(m -> gamma e e)``) and two-body decays of ``rho``, ``omega``,
``phi``, ``J/psi`` and ``Upsilon`` (``eps^2 x BR(m -> e e)``), each with a
phase-space suppression toward threshold. The cascade side is the familiar
ratio ``Z_m / (1 - Z_NN) x Phi_N`` with delta-function decay kinematics
(``x = 1/2`` two-body, ``1/4`` Dalitz). Light-meson moments are constants
(Feynman scaling); the printed validation against the reference's Fig. 2
holds them to a factor of a few.

**Onia from first principles.** Charm and bottom production is threshold
dominated, so ``Z_{N J/psi}`` and ``Z_{N Upsilon}`` rise steeply with
energy -- a constant moment overshoots the low-energy flux by two orders
of magnitude. This example computes them: LO ``gg -> Q Qbar`` plus
``q qbar -> Q Qbar`` with example 63's compact PDFs (a gluon added,
momentum-sum normalized), a colour-evaporation fraction into the 1S state
anchored to the fixed-target cross sections (``sigma_Jpsi ~ 0.3-0.5 ub``
and ``sigma_bb ~ 10-30 nb`` at ``sqrt s = 38.8`` GeV), folded into the
spectrum-weighted moment exactly as example 63 folds Drell-Yan.

**Beyond the mesons.** Two channels the reference does not carry:
``Upsilon`` decays, and the continuum ``q qbar -> gamma* -> chi chibar``
-- example 63's Drell-Yan fold with a Dirac fermion in the final state,
open at any mass for pair masses above :data:`M_DY_MIN_GEV` (below that
the resonance rows already are the continuum). The reference (and the
JUNO forecast built on it) stops at ``m_chi = m_Jpsi / 2 = 1.55`` GeV;
here ``Upsilon`` carries the flux to 4.7 GeV and the continuum past it --
at ``m_chi = 2`` GeV the continuum is already twenty times the
``Upsilon`` channel, and above ``m_Upsilon / 2`` it is the only source,
connecting this forecast to example 65's heavy plane with no gap.

**Propagation.** At these charges the loss is pure ionization,
``eps^2 alpha``, so the overburden is one number per direction:
``E_min(theta) = eps^2 alpha X(theta)`` -- the closed-form limit of the
project's transport, reached exactly.

**Detection** (trigger of arXiv:2411.00484, deployed November 2023). An
MCP crossing DeepCore scatters on electrons with
``dsigma/dE_r ~ 2 pi alpha^2 eps^2 / (m_e E_r^2)``; each scatter above
:data:`E_VIS_GEV` lights a DOM with probability :data:`P_HIT` (the one
free instrumental number, scanned over a band). The Faint Particle Trigger
slides a 2500 ns window over all DeepCore HLC and SLC hits and asks for
5-19 hits with at least 10 velocity-consistent pairs -- a 350 m crossing
fits in one window, and ``>= 5`` track-consistent hits give exactly
``C(5,2) = 10`` pairs, so :data:`N_HITS` ``= 5`` is the published trigger
floor, not a guess. Signal efficiency is the Poisson tail of the expected
hit count. The sensitivity assumes the line-like selection is
background-free at 90% (2.44 events), with the ``p_hit`` band as the
honesty interval.

**Output.** ``eps_90(m)`` for ten years of FPT operation, drawn over the
approximate current exclusions (SLAC mQ, ArgoNeuT, milliQan demonstrator,
Super-K recast; boundaries are labeled approximate). Below ~1.5 GeV the
existing limits are stronger -- the honest onia moments cost the old
placeholder claim there. The new territory is ``m ~ 3-15`` GeV at
``eps ~ 0.1-0.3``: past ArgoNeuT's mass edge, below the milliQan
demonstrator ceiling, and beyond where any meson-decay forecast can go --
carried first by ``Upsilon`` and then by the continuum Drell-Yan.

Usage
-----
    python scripts/future_bsm/66_millicharge_faint_hits.py
    python scripts/future_bsm/66_millicharge_faint_hits.py --p-hit 0.5 --n-hits 8
"""

import argparse
import importlib.util
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"

ALPHA_EM = 1.0 / 137.036
M_E_GEV = 5.11e-4
GEV2_TO_CM2 = 3.894e-28


def load_example(stem: str, name: str):
    """Import a sibling example module by file stem."""
    # The hubs this script loads live with the paper scripts; its own
    # siblings live here.
    path = _HERE / stem
    if not path.exists():
        path = _HERE.parent / "2026_muon_transport" / stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: Example 63 supplies the compact PDFs and the air-target constants.
_EX63 = load_example("63_analytic_stau_limit.py", "_example_63")

#: Mesons: mass [GeV], ``BR(m -> e e (gamma))``, spectrum-weighted production
#: moment ``Z_Nm`` (constant for the light mesons -- Feynman scaling; ``None``
#: for the onia, whose energy-dependent moments are computed below), Dalitz
#: or two-body.
MESONS = {
    "pi0": (0.1350, 1.17e-2, 4.0e-2, "dalitz"),
    "eta": (0.5479, 6.9e-3, 4.0e-3, "dalitz"),
    "rho": (0.7753, 4.72e-5, 1.0e-2, "twobody"),
    "omega": (0.7827, 7.36e-5, 1.0e-2, "twobody"),
    "phi": (1.0195, 2.98e-4, 1.0e-3, "twobody"),
    "jpsi": (3.0969, 5.97e-2, None, "twobody"),
    "upsilon": (9.4604, 2.38e-2, None, "twobody"),
}

#: Mean energy fraction the MCP takes from the parent.
X_MEAN = {"dalitz": 0.25, "twobody": 0.5}

#: Primary nucleon flux and cascade moment (example 63's values).
PHI_N0, GAMMA_CR, Z_NN = 1.8, 2.7, 0.26

#: Heavy quarks: mass [GeV] and the colour-evaporation fraction of pairs that
#: bind into the 1S onium (anchored to fixed-target data in the printout).
M_CHARM, M_BOTTOM = 1.27, 4.18
F_JPSI, F_UPSILON = 0.015, 0.005
ONIA = {"jpsi": (M_CHARM, F_JPSI), "upsilon": (M_BOTTOM, F_UPSILON)}

#: NLO correction on the LO pair cross section, and the one-loop coupling.
K_QQ = 2.0
LAMBDA_QCD_GEV = 0.30

#: Continuum Drell-Yan ``q qbar -> gamma* -> chi chibar`` -- the channel the
#: meson list misses between the ``phi`` and the onia and the only one above
#: ``m_Upsilon / 2``. The compact PDFs are trusted only for pair masses above
#: :data:`M_DY_MIN_GEV` (below that the resonances ARE the ``rho``/``omega``/
#: ``phi`` rows); example 63's K factor carries over.
M_DY_MIN_GEV = 1.1
K_DY = 1.3

#: Ionization loss [GeV cm^2/g] at unit charge.
ALPHA_ION = 2.0e-3

#: DeepCore: crossing path [m], electron density of ice [cm^-3], vertical
#: overburden [km w.e.] and the downgoing cos(theta) range used.
PATH_M = 350.0
N_E_ICE = 3.1e23
VERTICAL_KMWE = 0.92 * 2.1
COS_RANGE = (0.3, 1.0)

#: Detection model: visible-recoil threshold, per-scatter hit probability
#: (with its band), and the FPT hit floor -- the trigger's 2500 ns window
#: asks for 5-19 hits with >= 10 velocity-consistent pairs (arXiv:2411.00484),
#: and 5 track-consistent hits give exactly C(5,2) = 10 pairs.
E_VIS_GEV = 0.01
P_HIT = 0.2
P_HIT_BAND = (0.05, 0.5)
N_HITS = 5

#: Exposure [yr] of FPT operation (the trigger is live since November 2023)
#: and the background-free 90% count.
YEARS = 10.0
N90 = 2.44

#: Mass grid [GeV]; the top decade is continuum-Drell-Yan territory.
MASSES_GEV = np.logspace(np.log10(0.01), np.log10(15.0), 29)

#: Approximate current exclusions, (mass [GeV], eps) polylines (upper edge of
#: the excluded region is above these lines); labeled approximate on purpose.
CURRENT = {
    "SLAC mQ (approx.)": ([0.01, 0.1], [2.0e-4, 2.0e-3]),
    "ArgoNeuT (approx.)": ([0.1, 1.0, 3.0], [1.5e-3, 5.0e-3, 3.0e-2]),
    "milliQan demo (approx.)": ([0.03, 1.0, 4.7], [1.0e-2, 3.0e-2, 3.0e-1]),
    "Super-K recast (approx.)": ([0.1, 0.5, 1.5], [2.0e-3, 4.0e-3, 1.5e-2]),
}

COLORS = {"sens": "#e7298a", "band": "#e7298a", "cur": "0.55"}


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--p-hit", type=float, default=P_HIT)
    parser.add_argument("--n-hits", type=int, default=N_HITS)
    parser.add_argument("--years", type=float, default=YEARS)
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the figure, '66a'.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Onia production: LO heavy-quark pairs + colour evaporation
# ---------------------------------------------------------------------------


def alpha_s(mu_gev):
    """One-loop coupling, four flavours."""
    return 12.0 * np.pi / (25.0 * np.log(mu_gev**2 / LAMBDA_QCD_GEV**2))


def gluon_pdf(x):
    """``x g(x)`` at ``Q ~ 2m_c``, carrying the momentum-sum ~42%."""
    x = np.asarray(x, dtype=float)
    return 1.5 * x**-0.2 * (1.0 - x) ** 5.0


def _sigma_hat_gg(s_hat, m_q, a_s):
    """LO ``gg -> Q Qbar`` [cm^2] (Combridge)."""
    rho = 4.0 * m_q**2 / s_hat
    beta = np.sqrt(np.clip(1.0 - rho, 0.0, None))
    log_term = np.log(np.clip((1.0 + beta) / np.clip(1.0 - beta, 1.0e-12, None), 1.0, None))
    value = ((1.0 + rho + rho**2 / 16.0) * log_term
             - (7.0 / 4.0 + 31.0 * rho / 16.0) * beta)
    return np.where(rho < 1.0,
                    np.pi * a_s**2 / (3.0 * s_hat) * value * GEV2_TO_CM2, 0.0)


def _sigma_hat_qq(s_hat, m_q, a_s):
    """LO ``q qbar -> Q Qbar`` [cm^2]."""
    rho = 4.0 * m_q**2 / s_hat
    beta = np.sqrt(np.clip(1.0 - rho, 0.0, None))
    return np.where(rho < 1.0,
                    8.0 * np.pi * a_s**2 / (27.0 * s_hat) * (1.0 + rho / 2.0)
                    * beta * GEV2_TO_CM2, 0.0)


def qq_pair_cross_section_cm2(s_gev2, m_q, n_x=40):
    """``sigma(N N -> Q Qbar X)`` and the pair spectrum ``dn/dx``.

    Mirrors example 63's Drell-Yan integral with the photon replaced by the
    two LO QCD channels; the pair (and, in colour evaporation, the onium)
    carries the projectile parton's fraction, ``x = x1``.

    Returns
    -------
    sigma : float
        LO x ``K_QQ`` pair cross section [cm^2].
    x_grid, dn_dx : np.ndarray
        Pair energy fraction and its distribution, normalized to 1.
    """
    tau_min = 4.0 * m_q**2 / s_gev2
    if tau_min >= 1.0:
        return 0.0, np.array([0.5]), np.array([0.0])
    a_s = alpha_s(2.0 * m_q)
    lx = np.linspace(np.log(tau_min) / 2.0, -1.0e-9, n_x)
    x1 = np.exp(lx)
    d_sigma_dx1 = np.zeros(n_x)
    for i, xa in enumerate(x1):
        x2_min = tau_min / xa
        if x2_min >= 1.0:
            continue
        lx2 = np.linspace(np.log(x2_min), -1.0e-9, n_x)
        x2 = np.exp(lx2)
        s_hat = xa * x2 * s_gev2
        xu1, xd1, xs1 = _EX63.pdfs(xa)
        xu2, xd2, xs2 = _EX63.pdfs(x2)
        lumi_gg = gluon_pdf(xa) * gluon_pdf(x2) / (xa * x2)
        val1, val2 = xu1 + xd1, xu2 + xd2
        lumi_qq = (val1 * xs2 + xs1 * val2 + 4.0 * xs1 * xs2) / (xa * x2)
        integrand = (lumi_gg * _sigma_hat_gg(s_hat, m_q, a_s)
                     + lumi_qq * _sigma_hat_qq(s_hat, m_q, a_s)) * x2
        d_sigma_dx1[i] = np.trapezoid(integrand, lx2)
    sigma = float(np.trapezoid(d_sigma_dx1 * x1, lx)) * K_QQ
    weight = d_sigma_dx1 * x1
    norm = max(np.trapezoid(weight, x1), 1.0e-300)
    return sigma, x1, weight / norm


_Z_ONIUM_CACHE = {}


def z_onium(name, energy_gev):
    """Energy-dependent ``Z_{N onium}`` -- spectrum-weighted, like example 63.

    ``Z(E) = int dx x^{gamma-1} [A F sigma_QQ(E/x) / sigma_air] dn/dx``,
    built once on a grid per onium and interpolated in log-log.
    """
    if name not in _Z_ONIUM_CACHE:
        m_q, f_frac = ONIA[name]
        e_grid = np.logspace(1.0, 6.5, 23)
        z_grid = np.zeros(e_grid.size)
        lx = np.linspace(np.log(1.0e-3), np.log(0.999), 40)
        x = np.exp(lx)
        dlx = np.gradient(lx)
        for i, e_m in enumerate(e_grid):
            value = 0.0
            for xa, dl in zip(x, dlx):
                s = 2.0 * _EX63.M_NUCLEON * (e_m / xa)
                sigma, x_grid, dn_dx = qq_pair_cross_section_cm2(s, m_q)
                if sigma <= 0.0:
                    continue
                dn = np.interp(xa, x_grid, dn_dx, left=0.0, right=0.0)
                value += (xa ** (GAMMA_CR - 1.0) * _EX63.A_AIR * f_frac * sigma
                          / _EX63.SIGMA_AIR_CM2 * dn) * xa * dl
            z_grid[i] = value
        good = z_grid > 0.0
        _Z_ONIUM_CACHE[name] = (np.log10(e_grid[good]), np.log10(z_grid[good]))
    log_e, log_z = _Z_ONIUM_CACHE[name]
    e = np.asarray(energy_gev, dtype=float)
    with np.errstate(divide="ignore"):
        out = 10.0 ** np.interp(np.log10(e), log_e, log_z, left=-np.inf)
    return np.where(np.isfinite(out), out, 0.0)


def _sigma_hat_chi(s_hat, mass_chi):
    """LO ``q qbar -> gamma* -> chi chibar`` [cm^2 x e_q^2 eps^2 stripped]."""
    beta2 = 1.0 - 4.0 * mass_chi**2 / s_hat
    beta = np.sqrt(np.clip(beta2, 0.0, None))
    return np.where(beta2 > 0.0,
                    4.0 * np.pi * ALPHA_EM**2 / (9.0 * s_hat)
                    * beta * (3.0 - beta**2) / 2.0 * GEV2_TO_CM2, 0.0)


def dy_chi_cross_section_cm2(s_gev2, mass_chi, n_x=40):
    """``sigma(N N -> chi chibar X)`` at ``eps = 1`` and the per-chi ``dn/dx``.

    Example 63's Drell-Yan integral with the scalar replaced by a Dirac
    fermion; each chi carries half the pair energy, ``x = x1 / 2``, and
    ``dn/dx`` is normalized to 2 (both chi count).
    """
    s_hat_min = max(4.0 * mass_chi**2, M_DY_MIN_GEV**2)
    tau_min = s_hat_min / s_gev2
    if tau_min >= 1.0:
        return 0.0, np.array([0.5]), np.array([0.0])
    lx = np.linspace(np.log(tau_min) / 2.0, -1.0e-9, n_x)
    x1 = np.exp(lx)
    d_sigma_dx1 = np.zeros(n_x)
    for i, xa in enumerate(x1):
        x2_min = tau_min / xa
        if x2_min >= 1.0:
            continue
        lx2 = np.linspace(np.log(x2_min), -1.0e-9, n_x)
        x2 = np.exp(lx2)
        integrand = (_EX63.parton_lumi_qqbar(xa, x2)
                     * _sigma_hat_chi(xa * x2 * s_gev2, mass_chi) * x2)
        d_sigma_dx1[i] = np.trapezoid(integrand, lx2)
    sigma = float(np.trapezoid(d_sigma_dx1 * x1, lx)) * K_DY
    x_grid = x1 / 2.0
    weight = d_sigma_dx1 * x1
    norm = max(np.trapezoid(weight, x_grid), 1.0e-300)
    return sigma, x_grid, 2.0 * weight / norm


_Z_DY_CACHE = {}


def z_dy(mass_chi, energy_gev):
    """``Z_{N chi}(E)`` of the continuum, per chi (the factor 2 is inside)."""
    key = round(float(mass_chi), 6)
    if key not in _Z_DY_CACHE:
        e_grid = np.logspace(0.0, 6.5, 23)
        z_grid = np.zeros(e_grid.size)
        lx = np.linspace(np.log(1.0e-3), np.log(0.5), 40)
        x = np.exp(lx)
        dlx = np.gradient(lx)
        for i, e_chi in enumerate(e_grid):
            value = 0.0
            for xa, dl in zip(x, dlx):
                s = 2.0 * _EX63.M_NUCLEON * (e_chi / xa)
                sigma, x_grid, dn_dx = dy_chi_cross_section_cm2(s, mass_chi)
                if sigma <= 0.0:
                    continue
                dn = np.interp(xa, x_grid, dn_dx, left=0.0, right=0.0)
                value += (xa ** (GAMMA_CR - 1.0) * _EX63.A_AIR * sigma
                          / _EX63.SIGMA_AIR_CM2 * dn) * xa * dl
            z_grid[i] = value
        good = z_grid > 0.0
        _Z_DY_CACHE[key] = (np.log10(e_grid[good]), np.log10(z_grid[good]))
    log_e, log_z = _Z_DY_CACHE[key]
    if log_e.size == 0:
        return np.zeros(np.asarray(energy_gev, dtype=float).shape)
    e = np.asarray(energy_gev, dtype=float)
    with np.errstate(divide="ignore"):
        out = 10.0 ** np.interp(np.log10(e), log_e, log_z, left=-np.inf)
    return np.where(np.isfinite(out), out, 0.0)


# ---------------------------------------------------------------------------
# The meson fold
# ---------------------------------------------------------------------------


def phase_space(mass_chi, mass_meson, kind):
    """Threshold suppression of the millicharged branching."""
    r = (2.0 * mass_chi / mass_meson) ** 2
    if r >= 1.0:
        return 0.0
    if kind == "twobody":
        return (1.0 + 0.5 * r) * np.sqrt(1.0 - r)
    return (1.0 - r) ** 3  # Dalitz-like, soft


def nucleon_flux(energy_gev):
    """Primary nucleon flux [GeV^-1 cm^-2 s^-1 sr^-1] (below-knee power law)."""
    return PHI_N0 * np.asarray(energy_gev, dtype=float) ** -GAMMA_CR


def mcp_flux(energy_gev, mass_chi, eps=1.0, only=None):
    """Surface MCP flux from the meson channels [GeV^-1 cm^-2 s^-1 sr^-1].

    Delta-function decay kinematics: each channel contributes
    ``2 Z_m BR / (1 - Z_NN) x Phi_N(E / x) / x``. Pass ``only`` (a channel
    name) to isolate one meson's contribution.
    """
    e = np.asarray(energy_gev, dtype=float)
    total = np.zeros(e.shape)
    for name, (mass_m, br_ee, z_m, kind) in MESONS.items():
        if only is not None and name != only:
            continue
        factor = phase_space(mass_chi, mass_m, kind)
        if factor <= 0.0:
            continue
        x = X_MEAN[kind]
        z = z_onium(name, e / x) if z_m is None else z_m
        total += 2.0 * z * br_ee * factor / (1.0 - Z_NN) * nucleon_flux(e / x) / x
    if only is None or only == "dy":
        # Continuum Drell-Yan: the fold already carries the decay kinematics.
        total += z_dy(mass_chi, e) / (1.0 - Z_NN) * nucleon_flux(e)
    return eps**2 * total


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def n_scatters(eps):
    """Expected visible scatters over the DeepCore crossing."""
    sigma_cm2 = (2.0 * np.pi * ALPHA_EM**2 * eps**2 / (M_E_GEV * E_VIS_GEV)) * GEV2_TO_CM2
    return sigma_cm2 * N_E_ICE * PATH_M * 100.0


def efficiency(eps, n_hits, p_hit):
    """Poisson tail: at least ``n_hits`` lit DOMs from the scatters."""
    return float(poisson.sf(n_hits - 1, n_scatters(eps) * p_hit))


def signal_rate(mass_chi, eps, years):
    """Expected FPT-selected events over the exposure, downgoing sky."""
    cos_grid = np.linspace(*COS_RANGE, 8)
    livetime_s = years * 3.156e7
    # DeepCore's footprint: ~150 m radius of denser strings.
    area_cm2 = np.pi * (150.0e2) ** 2
    total = 0.0
    for k in range(cos_grid.size - 1):
        cos_c = 0.5 * (cos_grid[k] + cos_grid[k + 1])
        d_omega = 2.0 * np.pi * (cos_grid[k + 1] - cos_grid[k])
        e_min = max(eps**2 * ALPHA_ION * VERTICAL_KMWE / cos_c * 1.0e5, 1.0)
        grid = np.logspace(np.log10(e_min), 4.0, 60)
        flux = mcp_flux(grid, mass_chi, eps)
        total += livetime_s * d_omega * area_cm2 * np.trapezoid(flux, grid)
    return total


def eps_90(mass_chi, n_hits, p_hit, years, n90=N90):
    """Charge at which the expected selected events reach ``n90``."""
    def events(eps):
        return signal_rate(mass_chi, eps, years) * efficiency(eps, n_hits, p_hit)

    lo, hi = 1.0e-5, 0.5
    if events(hi) < n90:
        return np.inf
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        if events(mid) < n90:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def _save(fig, out_dir: pathlib.Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {path.resolve()}")
    plt.close(fig)


def figure_plane(masses, central, band_lo, band_hi, out_dir, n_hits) -> None:
    """``eps_90(m)`` with the approximate current exclusions."""
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.6, 3.1))
        for label, (m, e) in CURRENT.items():
            ax.plot(m, e, color=COLORS["cur"], lw=0.9)
            ax.text(m[-1], e[-1] * 1.1, label, fontsize=4.8, color=COLORS["cur"],
                    ha="right", va="bottom")
        finite = np.isfinite(central)
        ax.fill_between(masses[finite], band_lo[finite], band_hi[finite],
                        color=COLORS["band"], alpha=0.2, lw=0,
                        label=rf"$p_{{\rm hit}}$ {P_HIT_BAND[0]:g}-{P_HIT_BAND[1]:g}")
        ax.plot(masses[finite], central[finite], color=COLORS["sens"], lw=1.4,
                label=rf"DeepCore FPT forecast (90\%, $\geq {n_hits}$ hits)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.01, 20.0)
        ax.set_ylim(1.0e-4, 0.5)
        ax.set_xlabel(r"$m_\chi$ [GeV]")
        ax.set_ylabel(r"charge $\varepsilon$ [e]")
        ax.legend(fontsize=5.5, frameon=False, loc="lower right")
        _save(fig, out_dir, "66a_millicharge_faint_hits")


def validate_production() -> None:
    """Anchor the onia machinery on data and on arXiv:2104.13924 Fig. 2."""
    print("Onia validation (LO x K, colour evaporation):")
    for label, sqrt_s, m_q, f_1s, anchor in (
            ("charm ", 38.8, M_CHARM, F_JPSI, "sigma_cc ~ 30-50 ub, sigma_Jpsi ~ 0.3-0.5 ub"),
            ("charm ", 200.0, M_CHARM, F_JPSI, "sigma_cc ~ 300-800 ub"),
            ("bottom", 38.8, M_BOTTOM, F_UPSILON, "sigma_bb ~ 10-30 nb, sigma_Y1S ~ 0.1 nb")):
        sigma, _, _ = qq_pair_cross_section_cm2(sqrt_s**2, m_q)
        print(f"  {label} sqrt s = {sqrt_s:5.1f}: sigma_QQ = {sigma/1e-30:9.3g} ub, "
              f"x F -> {f_1s * sigma/1e-30:9.3g} ub   ({anchor})")
    print("  Z_Jpsi(E):   " + "  ".join(
        f"{e:.0e}: {z_onium('jpsi', e):.2e}" for e in (1e2, 1e3, 1e4, 1e5)))
    print("  Z_Upsilon(E):" + "  ".join(
        f"  {e:.0e}: {z_onium('upsilon', e):.2e}" for e in (1e3, 1e4, 1e5)))
    print("  (the old constant placeholders were 5e-6 and 5e-8)")
    sigma_dy, _, _ = dy_chi_cross_section_cm2(38.8**2, 2.0)
    print(f"  continuum DY (m_chi = 2, eps = 1, sqrt s = 38.8): sigma = "
          f"{sigma_dy/1e-33:.3g} nb  (dimuon M > 4 GeV at this energy is ~0.1-1 nb)")
    for m_ref in (2.0, 4.0):
        parts = {name: mcp_flux(np.array([100.0]), m_ref, only=name)[0]
                 for name in ("jpsi", "upsilon", "dy")}
        print(f"  channel split at m = {m_ref:g}, E = 100: " + "  ".join(
            f"{k}: {v:.2e}" for k, v in parts.items()))

    print("AKM (arXiv:2104.13924) Fig. 2, m = 10 MeV, vertical, eps^-2 flux:")
    for name, e_ref, target in (("pi0", 10.0, "~1-3e-7"), ("pi0", 1.0e3, "~7e-13"),
                                ("jpsi", 10.0, "~3.5e-12"), ("jpsi", 1.0e3, "~2e-15")):
        ours = mcp_flux(np.array([e_ref]), 0.01, only=name)[0]
        print(f"  {name:5s} at E = {e_ref:6.0f}: ours {ours:9.2e}  vs AKM {target}")


def main() -> None:
    args = parse_args()
    print(f"DeepCore FPT forecast: path {PATH_M:.0f} m, E_vis {E_VIS_GEV * 1e3:.0f} MeV, "
          f">= {args.n_hits} hits at p_hit {args.p_hit:g}, {args.years:g} yr, "
          f"background-free {N90} events")
    print(f"  visible scatters at eps = 0.03: {n_scatters(0.03):.2f}; at 0.1: "
          f"{n_scatters(0.1):.1f}")
    validate_production()
    e_ref = np.array([1.0, 10.0, 100.0])
    for m in (0.01, 0.5, 3.5):
        flux = mcp_flux(e_ref, m)
        print(f"  eps^-2 flux, m = {m:g} GeV: " + "  ".join(
            f"E={e:.0f}: {f:.2e}" for e, f in zip(e_ref, flux)))

    print("\n  eps_90(m):")
    central, lo_b, hi_b = [], [], []
    for m in MASSES_GEV:
        c = eps_90(m, args.n_hits, args.p_hit, args.years)
        lo = eps_90(m, args.n_hits, P_HIT_BAND[1], args.years)
        hi = eps_90(m, args.n_hits, P_HIT_BAND[0], args.years)
        central.append(c)
        lo_b.append(lo)
        hi_b.append(hi)
        if m in MASSES_GEV[::4]:
            print(f"  m {m:7.3f} GeV: eps_90 {c:.3g}  (band {lo:.3g} - {hi:.3g})")
    central, lo_b, hi_b = map(np.array, (central, lo_b, hi_b))
    figure_plane(MASSES_GEV, central, lo_b, hi_b, args.out_dir, args.n_hits)


if __name__ == "__main__":
    main()
