"""Example 50 -- what through-going tracks alone say about flavour.

The tau wall of example 49 makes the zenith-energy distribution of
through-going tracks flavour sensitive: ``nu_tau -> tau -> mu`` survives
columns that absorb every ``nu_mu``, so the deep sky at high energy carries
the tau fraction. This example turns that into a mini analysis. For each
detector, expected counts are built on a zenith-energy grid for the
atmospheric flux (MCEq conventional and prompt, separate free normalizations)
and an astrophysical power law split between the ``nu_mu`` and tau channels,
and an Asimov profile likelihood is run in the flavour ratio

.. math:: r = \\frac{f_\\tau}{f_\\mu + f_\\tau},

with the astrophysical normalization, its spectral index and both atmospheric
normalizations profiled at every point. Tracks are blind to ``nu_e``, so
``r`` is the whole flavour content of the observable: in the flavour triangle
the constraint is a wedge anchored at the ``nu_e`` vertex, and the honest
statement of this method is that wedge, not a closed contour.

The truth is the combined-fit spectrum at the oscillation-averaged 1:1:1
composition (``r = 0.5``); the headline number per detector is the
significance with which a tau-free flux (1:1:0, ``r = 0``) is rejected.
IceCube enters at its real DR2 exposure and the forecast detectors at a
common ten years, with the selection layers this project measured: the DR2
normalization at IceCube and Gen2, and the trigger normalization times the
published-track-selection layer at the water sites.

Forecast idealizations, stated: true-energy binning (reconstruction smearing
would widen every wedge -- and does: example 51's reco-space Asimov at the
same 1:1:1 truth finds no IceCube sensitivity at all, 2 Delta lnL(r = 1) of
0.18, and no Gen2 sensitivity either. The tau wall is intact before
smearing -- Gen2's deep sky above 1 PeV holds 22 ``nu_mu`` and 11 tau tracks
in both examples -- but a through-going muon's entry energy is nearly flat
in log E from a TeV up to ``E_nu`` because its vertex position along the
track is unknown, so the released smearing puts the median reco energy of
a multi-PeV neutrino at 10^3.9 and six of those eleven tau tracks below the
fit window; the ideal counting statistic drops from 32 in true energy to
0.8 in reco), the South Pole atmospheric table used at all sites,
the regeneration ladder for ``nu_mu`` (the survival convention only lowers
the ``nu_mu`` floor in the wall region, tightening the wedges), and
footprint-plus-optics definitions for the detectors nobody has built.

Usage
-----
    python scripts/2026_muon_transport/50_flavor_triangle.py
    python scripts/2026_muon_transport/50_flavor_triangle.py --rebuild-grids
"""

import argparse
import importlib.util
import pathlib
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from softpaws.data.loader import compute_livetime_s, load_uptime
from softpaws.data.schema import SEASONS
from softpaws.detectors import GEN2

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"
_DEFAULT_DATA_DIR = _HERE.parents[1] / "src" / "softpaws" / "data" / "dataverse_files"

#: Cached response grids per detector, ``<name>_mu`` and ``<name>_tau`` on
#: (:data:`LOG10_E_GRID`, :data:`COS_THETA_CENTERS`).
_GRID_CACHE = _DEFAULT_OUT_DIR / "50_response_grids.npz"

#: Atmospheric flux cache from example 22.
_MCEQ_CACHE = _HERE / "output" / "22_mceq_atmospheric_flux.npz"

#: Truth spectrum: the combined-fit single power law (arXiv:2308.00191),
#: per-flavour ``nu + nubar`` at 100 TeV [GeV^-1 cm^-2 s^-1 sr^-1].
TRUTH_PHI0 = 1.80e-18
TRUTH_GAMMA = 2.52

#: IceCube's published three-flavour best fit at Earth (arXiv:2510.24957),
#: drawn on the triangle for context; each fraction is nonzero at 90% CL.
ICECUBE_BEST_FIT = (0.30, 0.37, 0.33)

#: Upgoing zenith cells and energy-bin edges of the counting grid.
COS_THETA_EDGES = np.linspace(-0.999, -0.02, 13)
COS_THETA_CENTERS = 0.5 * (COS_THETA_EDGES[:-1] + COS_THETA_EDGES[1:])
LOG10_E_EDGES = np.arange(4.0, 8.01, 0.5)

#: Fine energy grid the responses are tabulated on, for the in-bin integrals.
LOG10_E_GRID = np.linspace(4.0, 8.0, 81)

#: Forecast exposure of every detector but IceCube, which enters at its real
#: uptime-summed DR2 livetime [yr].
FORECAST_YEARS = 10.0

#: Flavour-ratio scan grid.
R_GRID = np.linspace(0.0, 1.0, 41)

#: Per-detector selection normalization: the DR2 value at the ice sites, and
#: the ARCA trigger normalization times the published-track-selection layer
#: (0.82, example 47) at the water sites.
EFFICIENCY_ICE = 0.755
EFFICIENCY_WATER = 0.720 * 0.82

#: IceCube-Gen2: 120 further strings on a 240 m grid, ~7.9 km^3 instrumented,
#: modelled as an IceCube-optics prism of that volume.
GEN2_RADIUS_KM = GEN2.radius_km
GEN2_HEIGHT_KM = GEN2.height_km

#: Detectors whose wedges the triangle figure draws; the profile figure keeps
#: the full roster. The wedge fills take the style palette's pink and purple.
TRIANGLE_DETECTORS = ("IceCube", "IceCube-Gen2")
TRIANGLE_COLOR = {"IceCube": "#e7298a", "IceCube-Gen2": "#7570b3"}
TRIANGLE_ALPHA = 0.1

#: Detector colours in the two figures.
SITE_COLOR = {
    "IceCube": "k",
    "KM3NeT/ARCA230": "#1b9e77",
    "IceCube-Gen2": "#7570b3",
    "P-ONE": "#d95f02",
    "TRIDENT": "#e7298a",
}


def load_example(stem: str, name: str):
    """Import a sibling example module by file stem."""
    spec = importlib.util.spec_from_file_location(name, _HERE / stem)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=pathlib.Path, default=_DEFAULT_DATA_DIR,
                        help="Root of the DR2 release, for the livetime.")
    parser.add_argument("--rebuild-grids", action="store_true",
                        help="Rebuild the cached response grids.")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the two figures, '50a' and '50b'.")
    return parser.parse_args()


def roster(ex35, ex45):
    """The five detectors: example 35 geometry, example 45 optics, selection.

    Returns
    -------
    detectors : list of tuple
        One entry ``(site35, site45, efficiency, years)`` per detector;
        IceCube's years are filled in later from the DR2 uptime.
    """
    sites = {s.name: s for s in ex35.build_sites()}
    ice_optics = replace(ex45.ICECUBE_SITE, attenuation_override_m=42.0)
    water_optics = replace(ex45.ARCA_SITE, attenuation_override_m=47.0)
    gen2 = replace(sites["IceCube"], name="IceCube-Gen2",
                   radius_km=GEN2_RADIUS_KM, height_km=GEN2_HEIGHT_KM)
    return [
        (sites["IceCube"], ice_optics, EFFICIENCY_ICE, None),
        (sites["ARCA230"], replace(water_optics, name="KM3NeT/ARCA230"),
         EFFICIENCY_WATER, FORECAST_YEARS),
        (gen2, replace(ice_optics, name="IceCube-Gen2"), EFFICIENCY_ICE,
         FORECAST_YEARS),
        (sites["P-ONE"], replace(water_optics, name="P-ONE"), EFFICIENCY_WATER,
         FORECAST_YEARS),
        (sites["TRIDENT"], replace(water_optics, name="TRIDENT"),
         EFFICIENCY_WATER, FORECAST_YEARS),
    ]


def display_name(site35) -> str:
    """Roster name of one detector, ARCA carrying its KM3NeT prefix."""
    return "KM3NeT/ARCA230" if site35.name == "ARCA230" else site35.name


def response_grids(ex35, ex45, ex46, detectors, rebuild: bool) -> dict:
    """Per-detector channel responses on the fine grid [cm^2], cached.

    Parameters
    ----------
    ex35, ex45, ex46 : ModuleType
        Examples 35, 45 and 46.
    detectors : list of tuple
        The roster.
    rebuild : bool
        Ignore the cache and rebuild.

    Returns
    -------
    grids : dict of str -> np.ndarray
        ``"<name>_mu"`` and ``"<name>_tau"`` of shape
        ``(LOG10_E_GRID.size, COS_THETA_CENTERS.size)``.
    """
    wanted = [f"{display_name(s)}_{c}" for s, _, _, _ in detectors
              for c in ("mu", "tau")]
    if _GRID_CACHE.exists() and not rebuild:
        cache = dict(np.load(_GRID_CACHE))
        if all(k in cache for k in wanted):
            print(f"  cached response grids from {_GRID_CACHE.name}")
            return cache

    grids = {}
    coarse = ex35.COMMON_LOG10_E
    for site35, site45, efficiency, _ in detectors:
        name = display_name(site35)
        for flavours, channel in ((("mu",), "mu"), (("tau",), "tau")):
            print(f"  building {name} {channel} ...")
            per_species = [
                ex46.directional_aeff_cm2(ex35, ex45, site35, site45,
                                          COS_THETA_CENTERS, 8.0, flavours,
                                          xsec, None, 1.0, efficiency)
                for xsec in ex45.SPECIES
            ]
            coarse_grid = np.mean(per_species, axis=0)
            fine = np.empty((LOG10_E_GRID.size, COS_THETA_CENTERS.size))
            with np.errstate(divide="ignore"):
                log_a = np.log10(np.clip(coarse_grid, 1.0e-30, None))
            for j in range(COS_THETA_CENTERS.size):
                fine[:, j] = 10.0 ** np.interp(LOG10_E_GRID, coarse, log_a[:, j])
            grids[f"{name}_{channel}"] = fine
    _GRID_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(_GRID_CACHE, **grids)
    return grids


def atmospheric_grids() -> tuple[np.ndarray, np.ndarray]:
    """Conventional and prompt fluxes on the fine grid [GeV^-1 cm^-2 s^-1 sr^-1].

    Example 22's South Pole table, indexed by the declination equivalent of
    each zenith cell; used at every site, which is a stated forecast
    idealization.

    Returns
    -------
    conv, prompt : np.ndarray
        Fluxes of shape ``(LOG10_E_GRID.size, COS_THETA_CENTERS.size)``.

    Raises
    ------
    FileNotFoundError
        Raised if example 22's cache is missing.
    """
    if not _MCEQ_CACHE.exists():
        raise FileNotFoundError(
            f"{_MCEQ_CACHE} not found; run example 22 once to tabulate the "
            "atmospheric flux.")
    atm = np.load(_MCEQ_CACHE)
    dec_deg = np.rad2deg(np.arcsin(np.clip(-COS_THETA_CENTERS, 0.0, 1.0)))
    energy = 10.0**LOG10_E_GRID
    out = []
    for component in ("conv", "prompt"):
        log_f = np.log10(np.clip(atm[component], 1.0e-99, None))
        on_dec = np.array([
            np.interp(dec_deg, atm["dec_deg"], log_f[i])
            for i in range(atm["energy_gev"].size)
        ])
        grid = np.empty((energy.size, dec_deg.size))
        for j in range(dec_deg.size):
            grid[:, j] = 10.0 ** np.interp(np.log10(energy),
                                           np.log10(atm["energy_gev"]),
                                           on_dec[:, j])
        out.append(grid)
    return out[0], out[1]


def binned_counts(response, flux, livetime_s) -> np.ndarray:
    """Expected counts on the zenith-energy grid.

    Parameters
    ----------
    response : np.ndarray
        Effective area [cm^2] on the fine grid.
    flux : np.ndarray
        Flux [GeV^-1 cm^-2 s^-1 sr^-1] on the fine grid.
    livetime_s : float
        Exposure [s].

    Returns
    -------
    counts : np.ndarray, shape (n_e_bin, n_zenith)
        Expected events per bin.
    """
    energy = 10.0**LOG10_E_GRID
    integrand = response * flux
    d_omega = 2.0 * np.pi * np.diff(COS_THETA_EDGES)
    counts = np.empty((LOG10_E_EDGES.size - 1, COS_THETA_CENTERS.size))
    for i in range(LOG10_E_EDGES.size - 1):
        sel = (LOG10_E_GRID >= LOG10_E_EDGES[i]) & (LOG10_E_GRID <= LOG10_E_EDGES[i + 1])
        counts[i] = np.trapezoid(integrand[sel], energy[sel], axis=0)
    return livetime_s * counts * d_omega[None, :]


class TrackLikelihood:
    """Asimov Poisson likelihood in the flavour ratio for one detector.

    Attributes
    ----------
    asimov : np.ndarray
        Truth counts on the grid, built at ``r = 0.5`` and the truth spectrum.
    """

    def __init__(self, a_mu, a_tau, conv, prompt, livetime_s):
        self._a_mu, self._a_tau = a_mu, a_tau
        self._livetime_s = livetime_s
        self._conv = binned_counts(a_mu, conv, livetime_s)
        self._prompt = binned_counts(a_mu, prompt, livetime_s)
        self.asimov = self.expectation(0.5, 1.0, TRUTH_GAMMA, 1.0, 1.0)

    def _astro(self, channel, gamma) -> np.ndarray:
        response = self._a_mu if channel == "mu" else self._a_tau
        energy = 10.0**LOG10_E_GRID
        flux = 2.0 * TRUTH_PHI0 * (energy[:, None] / 1.0e5) ** (-gamma)
        return binned_counts(response, flux * np.ones((1, COS_THETA_CENTERS.size)),
                             self._livetime_s)

    def expectation(self, r, norm, gamma, a_conv, a_prompt) -> np.ndarray:
        """Expected counts at one parameter point.

        ``norm`` rescales the summed ``nu_mu + nu_tau`` astrophysical flux,
        whose per-flavour truth is ``TRUTH_PHI0``; the factor 2 inside
        :meth:`_astro` carries the two flavours the track channel sees.
        """
        astro = norm * ((1.0 - r) * self._astro("mu", gamma)
                        + r * self._astro("tau", gamma))
        return a_conv * self._conv + a_prompt * self._prompt + astro

    def delta_ll(self, r) -> float:
        """Profiled ``2 Delta ln L`` at one flavour ratio against the truth."""
        data = self.asimov

        def negative_ll(params):
            norm, gamma, a_conv, a_prompt = params
            if norm <= 0.0 or a_conv <= 0.0 or a_prompt <= 0.0:
                return 1.0e12
            mu = np.clip(self.expectation(r, norm, gamma, a_conv, a_prompt),
                         1.0e-12, None)
            return 2.0 * float(np.sum(mu - data * np.log(mu)))

        best = minimize(negative_ll, x0=(1.0, TRUTH_GAMMA, 1.0, 1.0),
                        method="Nelder-Mead",
                        options={"xatol": 1.0e-4, "fatol": 1.0e-6, "maxiter": 4000})
        return best.fun

    def profile(self) -> np.ndarray:
        """``2 Delta ln L`` on :data:`R_GRID`, floored at its minimum."""
        curve = np.array([self.delta_ll(r) for r in R_GRID])
        return curve - curve.min()


def interval(curve: np.ndarray, level: float) -> tuple[float, float]:
    """Crossing points of the profile at one ``2 Delta ln L`` level."""
    below = curve <= level
    lo = R_GRID[below].min()
    hi = R_GRID[below].max()
    return float(lo), float(hi)


def figure_profiles(profiles: dict[str, np.ndarray], out_dir) -> None:
    """The profiled ``2 Delta ln L`` against the flavour ratio, all detectors."""
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.0, 3.0))
        for level, note in ((1.0, r"$68\%$"), (3.84, r"$95\%$")):
            ax.axhline(level, color="0.75", lw=0.7, ls=":")
            ax.text(1.005, level, note, fontsize=7, color="0.45", va="center")
        for name, curve in profiles.items():
            ax.plot(R_GRID, curve, color=SITE_COLOR[name], lw=1.2, label=name)
        ax.axvline(0.5, color="0.6", lw=0.7, ls="--")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 12.0)
        ax.set_xlabel(r"$f_\tau \,/\, (f_\mu + f_\tau)$ at Earth")
        ax.set_ylabel(r"$2\,\Delta\ln L$")
        ax.legend(fontsize=7, frameon=False, loc="upper right")
        _save(fig, out_dir, "50a_flavor_profiles")


def _ternary_xy(f_e, f_mu, f_tau) -> tuple[float, float]:
    """Barycentric to Cartesian in the arXiv:2510.24957 orientation.

    ``nu_tau`` sits at the lower left, ``nu_e`` at the lower right and
    ``nu_mu`` on top, so the bottom axis reads the ``nu_e`` fraction.
    """
    return f_e + 0.5 * f_mu, f_mu * np.sqrt(3.0) / 2.0


def load_published_curves() -> dict[str, np.ndarray]:
    """The curves of arXiv:2510.24957 Fig. 1, extracted from its vector paths.

    Returns
    -------
    curves : dict of str -> np.ndarray
        Keyed ``contour68``, ``contour95``, ``std_osc`` and ``icecube2022``;
        each an ``(n, 3)`` array of ``(f_e, f_mu, f_tau)``.
    """
    path = (_HERE.parents[1] / "src" / "softpaws" / "data" / "icecube"
            / "flavor2510_fig1.csv")
    names = np.genfromtxt(path, delimiter=",", usecols=(0,), dtype=str)
    values = np.genfromtxt(path, delimiter=",", usecols=(1, 2, 3))
    return {name: values[names == name] for name in np.unique(names)}


def earth_composition(source: tuple[float, float, float]) -> tuple[float, ...]:
    """Oscillation-averaged flavour fractions at Earth for a source ratio.

    Uses the averaged conversion matrix ``P_ab = sum_i |U_ai|^2 |U_bi|^2``
    with current global-fit mixing angles and ``delta_CP = 0``, which is the
    convention the standard triangle markers are drawn with.

    Parameters
    ----------
    source : tuple of float
        Source composition ``(f_e, f_mu, f_tau)``, any normalization.

    Returns
    -------
    fractions : tuple of float
        Earth composition, summing to one.
    """
    th12, th23, th13 = np.radians((33.44, 49.0, 8.57))
    c12, s12 = np.cos(th12), np.sin(th12)
    c23, s23 = np.cos(th23), np.sin(th23)
    c13, s13 = np.cos(th13), np.sin(th13)
    pmns = np.array([
        [c12 * c13, s12 * c13, s13],
        [-s12 * c23 - c12 * s23 * s13, c12 * c23 - s12 * s23 * s13, s23 * c13],
        [s12 * s23 - c12 * c23 * s13, -c12 * s23 - s12 * c23 * s13, c23 * c13],
    ])
    probability = pmns**2 @ pmns.T**2
    at_earth = probability @ (np.asarray(source, dtype=float) / np.sum(source))
    return tuple(at_earth / at_earth.sum())


#: Source scenarios the standard triangle marks, oscillation averaged to
#: Earth, in the marker scheme of arXiv:2510.24957: pion decay (circle),
#: muon-damped (square) and neutron decay (triangle).
SOURCE_MARKERS = (
    (r"$1\!:\!2\!:\!0$ source", (1.0, 2.0, 0.0), "o", "#e376c2"),
    (r"$0\!:\!1\!:\!0$ source", (0.0, 1.0, 0.0), "s", "#2b9e2b"),
    (r"$1\!:\!0\!:\!0$ source", (1.0, 0.0, 0.0), "^", "#ff7d0c"),
)


def draw_triangle_frame(ax) -> dict[str, tuple[float, float]]:
    """Ternary grid, edges and axis labels of the flavour triangle.

    The arXiv:2510.24957 orientation (``nu_e`` fraction along the bottom,
    ``nu_tau`` lower left, ``nu_mu`` on top) with a 0.1 ternary grid and
    fraction ticks on all three sides.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on; limits, aspect and axis visibility are set here.

    Returns
    -------
    corners : dict of str -> tuple of float
        Cartesian corner coordinates, keyed ``"tau"``, ``"e"`` and ``"mu"``.
    """
    root3half = np.sqrt(3.0) / 2.0
    corners = {"tau": (0.0, 0.0), "e": (1.0, 0.0), "mu": (0.5, root3half)}

    for value in np.arange(0.1, 0.95, 0.1):
        for a, b in (
            (_ternary_xy(value, 0.0, 1.0 - value),
             _ternary_xy(value, 1.0 - value, 0.0)),
            (_ternary_xy(0.0, value, 1.0 - value),
             _ternary_xy(1.0 - value, value, 0.0)),
            (_ternary_xy(0.0, 1.0 - value, value),
             _ternary_xy(1.0 - value, 0.0, value)),
        ):
            ax.plot(*zip(a, b, strict=True), color="0.88", lw=0.4, zorder=1)
        x, y = _ternary_xy(value, 0.0, 1.0 - value)
        ax.text(x, y - 0.033, f"{value:.1f}", ha="center", va="top",
                fontsize=8, color="0.35", rotation=-60)
        x, y = _ternary_xy(0.0, 1.0 - value, value)
        ax.text(x - 0.030, y + 0.014, f"{value:.1f}", ha="right",
                va="center", fontsize=8, color="0.35", rotation=60)
        x, y = _ternary_xy(1.0 - value, value, 0.0)
        ax.text(x + 0.030, y + 0.014, f"{value:.1f}", ha="left",
                va="center", fontsize=8, color="0.35")

    triangle = [corners["tau"], corners["e"], corners["mu"], corners["tau"]]
    ax.plot(*zip(*triangle, strict=True), color="k", lw=0.9, zorder=3)
    ax.text(0.5, -0.115, r"$\nu_e$ fraction", ha="center",
            va="top", fontsize=8)
    ax.text(0.115, 0.50, r"$\nu_\tau$ fraction",
            ha="center", va="center", fontsize=8, rotation=60)
    ax.text(0.885, 0.50, r"$\nu_\mu$ fraction",
            ha="center", va="center", fontsize=8, rotation=-60)

    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.16, 1.06)
    ax.set_aspect("equal")
    ax.axis("off")
    return corners


def draw_published_curves(ax, skip=()) -> None:
    """The extracted arXiv:2510.24957 contours and their MESE best fit.

    The 68% and 95% CL contours, the previous-measurement contour, the
    standard-oscillation allowed region and the best-fit star, each with its
    legend label; ``skip`` names curves to leave out.
    """
    published = load_published_curves()
    curve_style = {
        "contour68": dict(color="k", lw=1.3, ls="-",
                          label=r"IceCube MESE $68\%$ CL"),
        "contour95": dict(color="k", lw=1.3, ls="--",
                          label=r"IceCube MESE $95\%$ CL"),
        "icecube2022": dict(color="#8a544a", lw=0.9, ls=":",
                            label=r"IceCube (2022) $68\%$ CL"),
        "std_osc": dict(color="0.4", lw=0.9, ls="-.",
                        label="std. osc. allowed"),
    }
    for name, style in curve_style.items():
        if name in skip:
            continue
        frac = published[name]
        x, y = _ternary_xy(frac[:, 0], frac[:, 1], frac[:, 2])
        ax.plot(x, y, zorder=4, **style)

    ax.plot(*_ternary_xy(*ICECUBE_BEST_FIT), marker="*", color="k", ms=7,
            ls="none", zorder=5, label="MESE best fit")


def figure_triangle(intervals: dict[str, tuple[float, float]], out_dir) -> None:
    """The flavour triangle in the style of arXiv:2510.24957 Fig. 1.

    The frame of :func:`draw_triangle_frame`, the published curves of
    :func:`draw_published_curves` -- and on top of them, each detector's
    tracks-only 68% wedge anchored at the ``nu_e`` vertex, with the interval
    segments stacked along the ``mu``-``tau`` edge. Tracks constrain only
    ``f_tau / (f_mu + f_tau)``, so the wedge is the honest contour shape of
    this method.
    """
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(4.4, 4.2))
        corners = draw_triangle_frame(ax)

        for name, (lo, hi) in intervals.items():
            if name not in TRIANGLE_DETECTORS:
                continue
            color = TRIANGLE_COLOR[name]
            edge_lo = _ternary_xy(0.0, 1.0 - lo, lo)
            edge_hi = _ternary_xy(0.0, 1.0 - hi, hi)
            wedge = plt.Polygon([corners["e"], edge_lo, edge_hi], closed=True,
                                facecolor=color, edgecolor=color, lw=0.6,
                                alpha=TRIANGLE_ALPHA, zorder=2, label=name)
            ax.add_patch(wedge)

        draw_published_curves(ax)

        handles, labels = ax.get_legend_handles_labels()
        n_wedges = len(TRIANGLE_DETECTORS)
        detector_part = ax.legend(handles[:n_wedges], labels[:n_wedges],
                                  fontsize=8, frameon=False, loc="upper left",
                                  bbox_to_anchor=(0.0, 1.16),
                                  handlelength=1.4, labelspacing=0.3,
                                  title="tracks-only 68\\% (this work)",
                                  title_fontsize=8, alignment="left")
        ax.add_artist(detector_part)
        ax.legend(handles[n_wedges:], labels[n_wedges:], fontsize=8,
                  frameon=False, loc="upper right", bbox_to_anchor=(1.08, 1.16),
                  handlelength=1.6, labelspacing=0.3)
        _save(fig, out_dir, "50b_flavor_triangle")


def _save(fig, out_dir: pathlib.Path, stem: str) -> None:
    """Write one figure to ``out_dir`` as both PDF and PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {path.resolve()}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    print("Loading examples 35, 45 and 46 ...")
    ex35 = load_example("35_point_source_effective_area.py", "_example_35")
    ex45 = load_example("45_first_principles_reach.py", "_example_45")
    ex46 = load_example("46_declination_resolved_reach.py", "_example_46")

    detectors = roster(ex35, ex45)
    grids = response_grids(ex35, ex45, ex46, detectors, args.rebuild_grids)
    conv, prompt = atmospheric_grids()

    dr2_livetime_s = sum(
        compute_livetime_s(load_uptime(args.data_dir / "uptime" / f"{s}_exp.csv"))
        for s in SEASONS)
    print(f"  IceCube at its DR2 livetime, "
          f"{dr2_livetime_s / (365.25 * 86400.0):.2f} yr; forecasts at "
          f"{FORECAST_YEARS:g} yr")

    profiles, intervals68 = {}, {}
    print(f"\n  {'detector':>16} {'68% on r':>16} {'reject 1:1:0':>13}")
    for site35, _, _, years in detectors:
        name = display_name(site35)
        livetime_s = (dr2_livetime_s if years is None
                      else years * 365.25 * 86400.0)
        likelihood = TrackLikelihood(grids[f"{name}_mu"], grids[f"{name}_tau"],
                                     conv, prompt, livetime_s)
        curve = likelihood.profile()
        profiles[name] = curve
        intervals68[name] = interval(curve, 1.0)
        sigma = np.sqrt(max(curve[0], 0.0))
        lo, hi = intervals68[name]
        print(f"  {name:>16} [{lo:5.2f}, {hi:5.2f}] {sigma:12.1f}σ")

    print("\n  source compositions, oscillation averaged to Earth:")
    for label, source, _, _ in SOURCE_MARKERS:
        f_e, f_mu, f_tau = earth_composition(source)
        clean = label.replace("$", "").replace("\\!", "").replace(" source", "")
        print(f"  {clean:>7s} -> ({f_e:.2f}, {f_mu:.2f}, {f_tau:.2f})")

    print()
    figure_profiles(profiles, args.out_dir)
    figure_triangle(intervals68, args.out_dir)


if __name__ == "__main__":
    main()
