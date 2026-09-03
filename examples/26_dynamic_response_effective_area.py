"""Example 26 — adding an energy-growing projected area to example 20.

Same comparison as ``20_effective_area_soft_vs_irf.py``, extended with the
energy-dependent projected area
:func:`~softpaws.transport.soft_volume.dynamic_projected_area_km2` (not part
of arXiv:2607.13143 -- see that function's docstring for the physical
motivation and the discussion this example follows up on).

Example 20's own numbers motivate the addition: the implied selection
efficiency ``eps = A_eff^IC / A_eff^range`` dips to ~0.5 around 5-15 TeV (a
genuine selection-cut effect), then rises smoothly through ``eps = 1`` around
20 PeV and reaches ``eps ~ 1.2`` at 100 PeV, still climbing. Above ~20 PeV the
published IceCube effective area exceeds even the geometric muon-range
ceiling, which assumes a fixed detector radius ``R_det``. The physical
picture: above the critical energy ``E_c`` (~570 GeV in water/ice) radiative
losses dominate, and the resulting stochastic light output lets a track
trigger strings from beyond ``R_det`` -- growing with energy.

The one new phenomenological parameter is the growth length ``L`` (km per
e-fold of energy above ``E_c``). Rather than asserting a value, it is fit
here to the horizon-band residual **above** ``log10(E_nu / GeV) = 7`` --
comfortably past where the static-radius efficiency has already turned on and
crossed 1 (example 20), isolating the pure geometric-growth regime from the
low-energy selection cuts. The fitted ``L`` is reported next to the ~100 m
photon-absorption-length ballpark quoted in the ice-optical-properties
literature, as a sanity check, not a derivation.

**What this does and doesn't fix.** The growth term clips to zero at and
below ``E_c``, so it is inactive for TeV-scale muons and does not touch the
~0.5 efficiency floor around 5-15 TeV -- that is a genuine event-selection
effect (quality cuts, background rejection), not a geometry effect, and
fixing it is explicitly not the goal here (see the parent discussion). The
goal is the high-energy excess; a flatter low-energy efficiency, if any, is a
bonus of using the same single fitted parameter, not something separately
tuned for.

Usage
-----
    python examples/26_dynamic_response_effective_area.py
    python examples/26_dynamic_response_effective_area.py --data-dir /path/to/dataverse_files
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

from softpaws.data.icecube import (
    irf_season as canonical_irf_season,
)
from softpaws.data.loader import compute_livetime_s, load_uptime, parse_aeff
from softpaws.data.schema import SEASONS
from softpaws.response.soft_volume import SoftVolumeResponse
from softpaws.transport.attenuation import prem_column, survival_probability
from softpaws.transport.cross_section import bgr18_cross_section
from softpaws.transport.soft_volume import (
    DEFAULT_MUON_THRESHOLD_GEV,
    range_target_volume_km3,
    sphere_radius_from_volume,
)
from softpaws.transport.source import (
    MEAN_INELASTICITY,
    nucleon_number_density,
)
from softpaws.utils.constants import CM_PER_KM, M_PER_KM

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_DATA_DIR = _HERE.parent / "src" / "softpaws" / "data" / "dataverse_files"
_DEFAULT_OUT_DIR = _HERE / "output"

# Upper edge is 100 PeV, the top of the simulation behind the published response.
COMMON_LOG10_E = np.linspace(2.0, 8.0, 61)
RADIUS_KM = sphere_radius_from_volume(1.0)  # ~0.62 km, IceCube-like
GAMMA = 2.38  # IceCube 9.5 yr diffuse-flux best fit (Eq. 1.3); see module docstring
DOWNGOING_COLUMN_KM = 1.95  # IceCube-like downgoing ice overburden (example 16's default)
HORIZON_SIN_DEC = 0.11  # half-width of the horizon band, |sin(dec)| < this

# Where the static-radius efficiency has already turned on and crossed 1
# (example 20), so the residual above this energy isolates the geometric
# light-yield growth from the low-energy selection-efficiency turn-on.
FIT_LOG10_E_MIN = 7.0

# Literature ballpark for the photon absorption length in South Pole ice
# (bulk-averaged, order of magnitude only -- e.g. the SPICE ice models put it
# at ~100-200 m depending on depth and wavelength). Used only as a sanity
# check on the fitted growth length, not as an input.
ICE_ABSORPTION_LENGTH_M = 100.0

# Tabulated cross section rather than the paper's power law; see example 20.
CROSS_SECTION = bgr18_cross_section()

_RANGE_LABEL = r"range, $E_{\rm thr} = 1$ TeV"
_DYNAMIC_LABEL = r"range, dynamic $A_{\rm proj}$"
_SOFT_STYLES = {
    "drift": ("--", "C0"),
    "exact, infinite column": (":", "C1"),
    f"exact, finite column ({DOWNGOING_COLUMN_KM:g} km)": ("-.", "C2"),
    _RANGE_LABEL: ("-", "C3"),
    _DYNAMIC_LABEL: ("-", "C4"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=_DEFAULT_DATA_DIR,
        help="Root of the DR2 data directory (must contain 'irfs/' and 'uptime/' subfolders).",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=_DEFAULT_OUT_DIR / "26_dynamic_response_effective_area.pdf",
        help="Output file for the figure.",
    )
    return parser.parse_args()


def combine_seasons_icecube(data_dir: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Livetime-weighted IceCube effective area, resolved in declination.

    Identical to example 20's function of the same name; duplicated rather
    than imported, since examples in this repo are self-contained scripts.

    Parameters
    ----------
    data_dir : pathlib.Path
        Root of the DR2 data directory.

    Returns
    -------
    values : np.ndarray, shape (n_energy, n_dec)
        Combined effective area [cm^2] on ``COMMON_LOG10_E`` and the tabulated
        ``sin(dec)`` bins.
    sin_dec_edges : np.ndarray, shape (n_dec + 1,)
        Bin edges in ``sin(dec)``.
    """
    irf_dir = data_dir / "irfs"
    uptime_dir = data_dir / "uptime"

    total = None
    total_livetime_s = 0.0
    sin_dec_edges = None
    aeff_cache: dict[str, object] = {}

    for season in SEASONS:
        irf_season = canonical_irf_season(season)
        if irf_season not in aeff_cache:
            raw = np.genfromtxt(irf_dir / f"{irf_season}_effectiveArea.csv", comments="#")
            aeff_cache[irf_season] = parse_aeff(raw)
        aeff = aeff_cache[irf_season]

        uptime = load_uptime(uptime_dir / f"{season}_exp.csv")
        livetime_s = compute_livetime_s(uptime)

        interp = np.column_stack([
            np.interp(COMMON_LOG10_E, aeff.log10_energy_centers, aeff.values[:, j])
            for j in range(aeff.values.shape[1])
        ])
        if total is None:
            total = np.zeros_like(interp)
            sin_dec_edges = aeff.sin_dec_edges
        total += livetime_s * interp
        total_livetime_s += livetime_s

    return total / total_livetime_s, sin_dec_edges


def survival_grid(sin_dec_centers: np.ndarray) -> np.ndarray:
    """Neutrino survival probability on the (energy, declination) grid.

    Identical to example 20's function of the same name.
    """
    dec_deg = np.rad2deg(np.arcsin(sin_dec_centers))
    columns = np.array([prem_column(d) for d in dec_deg])
    return survival_probability(
        10.0**COMMON_LOG10_E[:, None], columns[None, :], cross_section=CROSS_SECTION,
    )


def band_average(values: np.ndarray, sin_dec_edges: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Solid-angle-weighted average of an (energy, declination) grid over a band.

    Identical to example 20's function of the same name.
    """
    weights = np.diff(sin_dec_edges)[mask]
    return np.average(values[:, mask], axis=1, weights=weights)


def horizon_detector_area(
    icecube: np.ndarray,
    sin_dec_edges: np.ndarray,
    d_nu: np.ndarray,
) -> np.ndarray:
    """Detector-only IceCube effective area in the horizon band.

    Identical to example 20's function of the same name.
    """
    centers = 0.5 * (sin_dec_edges[:-1] + sin_dec_edges[1:])
    horizon = np.abs(centers) < HORIZON_SIN_DEC
    corrected = icecube[:, horizon] / d_nu[:, horizon]
    return np.average(corrected, axis=1, weights=np.diff(sin_dec_edges)[horizon])


def selection_efficiency(icecube_horizon: np.ndarray, range_area: np.ndarray) -> np.ndarray:
    """Ratio of the corrected IceCube area to a range model's geometric ceiling.

    Identical to example 20's function of the same name.
    """
    return np.divide(
        icecube_horizon,
        range_area,
        out=np.full_like(icecube_horizon, np.nan),
        where=range_area > 0.0,
    )


def range_area_cm2(light_yield_length_km: float | None) -> np.ndarray:
    """Range-convention effective area [cm^2] on ``COMMON_LOG10_E``.

    Parameters
    ----------
    light_yield_length_km : float or None
        Forwarded to :func:`~softpaws.transport.soft_volume.
        range_target_volume_km3`; ``None`` is the static-radius geometric
        ceiling (example 20's curve).

    Returns
    -------
    area : np.ndarray, shape (COMMON_LOG10_E.size,)
        Effective area [cm^2].
    """
    energy_nu = 10.0**COMMON_LOG10_E
    volume_km3 = range_target_volume_km3(
        RADIUS_KM,
        (1.0 - MEAN_INELASTICITY) * energy_nu,
        DEFAULT_MUON_THRESHOLD_GEV,
        light_yield_length_km=light_yield_length_km,
    )
    sigma = CROSS_SECTION.cc(energy_nu)
    return volume_km3 * CM_PER_KM**3 * nucleon_number_density() * sigma


def fit_light_yield_length_km(ic_horizon: np.ndarray) -> float:
    """Best-fit growth length ``L``, from the horizon-band residual above ``E_c``.

    Minimizes the sum of squared ``log(eps_dynamic)`` over
    ``log10(E_nu / GeV) >= FIT_LOG10_E_MIN``, i.e. drives the dynamic-model
    efficiency toward 1 in the regime where the static-radius ceiling is
    already known (example 20) to undershoot the published effective area.

    Parameters
    ----------
    ic_horizon : np.ndarray, shape (COMMON_LOG10_E.size,)
        Absorption-corrected IceCube effective area [cm^2] in the horizon band
        (:func:`horizon_detector_area`).

    Returns
    -------
    l_fit_km : float
        Best-fit growth length [km].
    """
    fit_mask = COMMON_LOG10_E >= FIT_LOG10_E_MIN

    def objective(l_km: float) -> float:
        eps = ic_horizon[fit_mask] / range_area_cm2(l_km)[fit_mask]
        return float(np.sum(np.log(eps) ** 2))

    result = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")
    return float(result.x)


def soft_volume_curves(l_fit_km: float) -> dict[str, np.ndarray]:
    """Target volume [km^3] on ``COMMON_LOG10_E``, static curves plus the dynamic one.

    Parameters
    ----------
    l_fit_km : float
        Fitted growth length [km] for the dynamic-``A_proj`` curve
        (:func:`fit_light_yield_length_km`).

    Returns
    -------
    curves : dict of str to np.ndarray
        Target volume [km^3] for each model variant; see example 20's
        function of the same name for the first four. The fifth,
        :data:`_DYNAMIC_LABEL`, is the range convention with the fitted
        energy-growing projected area.
    """
    energy_gev = 10.0**COMMON_LOG10_E

    drift = SoftVolumeResponse(radius_km=RADIUS_KM, method="drift", cross_section=CROSS_SECTION)
    exact_inf = SoftVolumeResponse(
        radius_km=RADIUS_KM, method="exact", column_depth_km=None, cross_section=CROSS_SECTION
    )
    exact_fin = SoftVolumeResponse(
        radius_km=RADIUS_KM, method="exact", column_depth_km=DOWNGOING_COLUMN_KM,
        cross_section=CROSS_SECTION,
    )
    to_km3 = 1.0 / CM_PER_KM**3
    return {
        "drift": drift.target_volume_cm3(energy_gev, GAMMA) * to_km3,
        "exact, infinite column": exact_inf.target_volume_cm3(energy_gev, GAMMA) * to_km3,
        f"exact, finite column ({DOWNGOING_COLUMN_KM:g} km)": (
            exact_fin.target_volume_cm3(energy_gev, GAMMA) * to_km3
        ),
        _RANGE_LABEL: range_target_volume_km3(
            RADIUS_KM,
            (1.0 - MEAN_INELASTICITY) * energy_gev,
            DEFAULT_MUON_THRESHOLD_GEV,
        ),
        _DYNAMIC_LABEL: range_target_volume_km3(
            RADIUS_KM,
            (1.0 - MEAN_INELASTICITY) * energy_gev,
            DEFAULT_MUON_THRESHOLD_GEV,
            light_yield_length_km=l_fit_km,
        ),
    }


def make_figure(
    icecube: np.ndarray,
    sin_dec_edges: np.ndarray,
    d_nu: np.ndarray,
    volumes: dict[str, np.ndarray],
    areas: dict[str, np.ndarray],
    l_fit_km: float,
    out_path: pathlib.Path,
) -> None:
    """Draw the three-panel comparison and write it to disk."""
    sin_dec_centers = 0.5 * (sin_dec_edges[:-1] + sin_dec_edges[1:])
    upgoing = sin_dec_centers > 0.0
    downgoing = sin_dec_centers < 0.0

    ic_up = band_average(icecube, sin_dec_edges, upgoing)
    ic_down = band_average(icecube, sin_dec_edges, downgoing)
    ic_horizon = horizon_detector_area(icecube, sin_dec_edges, d_nu)

    sigma = CROSS_SECTION.cc(10.0**COMMON_LOG10_E)
    n_nucleon = nucleon_number_density()
    v_icecube = ic_horizon / (n_nucleon * sigma) / CM_PER_KM**3

    with plt.style.context(str(_STYLE)):
        fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.5))

        # (a) Effective area, hemisphere averaged, both sides attenuated.
        ax = axes[0]
        ax.plot(COMMON_LOG10_E, ic_up, color="k", lw=1.6, label="IceCube, up")
        ax.plot(COMMON_LOG10_E, ic_down, color="0.55", lw=1.6, label="IceCube, down")
        d_nu_up = band_average(d_nu, sin_dec_edges, upgoing)
        for name, area in areas.items():
            ls, color = _SOFT_STYLES[name]
            ax.plot(COMMON_LOG10_E, area * d_nu_up, ls=ls, color=color, lw=1.0)
            ax.plot(COMMON_LOG10_E, area, ls=ls, color=color, lw=1.0, alpha=0.3)
        ax.set_yscale("log")
        ax.set_ylim(1e2, 1e9)
        ax.set_ylabel(r"$A_{\rm eff}$ [cm$^2$]")
        ax.set_title("(a) hemisphere average", fontsize=7)
        data_legend = ax.legend(fontsize=5.5, loc="upper left")
        proxies = [
            plt.Line2D([], [], color="0.3", lw=1.0),
            plt.Line2D([], [], color="0.3", lw=1.0, alpha=0.3),
        ]
        ax.legend(
            proxies,
            [r"model $\times\,D_\nu$ (up)", "model, no attenuation (down)"],
            fontsize=5.5,
            loc="lower right",
        )
        ax.add_artist(data_legend)

        # (b) Implied target volume at the horizon, absorption divided out.
        ax = axes[1]
        ax.plot(COMMON_LOG10_E, v_icecube, color="k", lw=1.6, label="IceCube, horizon")
        for name, volume in volumes.items():
            ls, color = _SOFT_STYLES[name]
            lw = 1.4 if name == _DYNAMIC_LABEL else 1.0
            ax.plot(COMMON_LOG10_E, volume, ls=ls, color=color, lw=lw, label=name)
        ax.set_yscale("log")
        ax.set_ylim(0.3, 2e2)
        ax.set_ylabel(r"$V_{\rm target}$ [km$^3$]")
        ax.set_title("(b) implied target volume", fontsize=7)
        ax.legend(fontsize=5, loc="upper left")

        # (c) What is left over: the selection turn-on, static vs. dynamic A_proj.
        ax = axes[2]
        eps_static = selection_efficiency(ic_horizon, areas[_RANGE_LABEL])
        eps_dynamic = selection_efficiency(ic_horizon, areas[_DYNAMIC_LABEL])
        ax.plot(COMMON_LOG10_E, eps_static, color="C3", lw=1.2, label="static " + r"$R_{\rm det}$")
        ax.plot(
            COMMON_LOG10_E, eps_dynamic, color="C4", lw=1.6,
            label=rf"dynamic, $L={l_fit_km * M_PER_KM:.0f}$ m",
        )
        ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
        ax.axvline(FIT_LOG10_E_MIN, color="0.6", lw=0.6, ls="--")
        ax.set_ylim(0.0, 1.3)
        ax.set_ylabel(r"$\varepsilon = A_{\rm eff}^{\rm IC} / A_{\rm eff}^{\rm range}$")
        ax.set_title("(c) implied selection efficiency", fontsize=7)
        ax.legend(fontsize=5.5, loc="upper left")

        for ax in axes:
            ax.set_xlim(COMMON_LOG10_E[0], COMMON_LOG10_E[-1])
            ax.set_xlabel(r"$\log_{10}(E_\nu\,/\,\mathrm{GeV})$")

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png"):
            path = out_path.with_suffix(suffix)
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {path.resolve()}")
        plt.close(fig)


def report(
    icecube: np.ndarray,
    sin_dec_edges: np.ndarray,
    d_nu: np.ndarray,
    volumes: dict[str, np.ndarray],
    areas: dict[str, np.ndarray],
    l_fit_km: float,
) -> None:
    """Print the horizon-band numbers behind panels (b) and (c)."""
    ic_horizon = horizon_detector_area(icecube, sin_dec_edges, d_nu)
    sigma = CROSS_SECTION.cc(10.0**COMMON_LOG10_E)
    v_icecube = ic_horizon / (nucleon_number_density() * sigma) / CM_PER_KM**3
    eps_static = selection_efficiency(ic_horizon, areas[_RANGE_LABEL])
    eps_dynamic = selection_efficiency(ic_horizon, areas[_DYNAMIC_LABEL])

    print(
        f"\nFitted light-yield growth length: L = {l_fit_km * M_PER_KM:.1f} m "
        f"(fit region log10(E/GeV) >= {FIT_LOG10_E_MIN:g}; "
        f"literature ice-absorption-length ballpark ~ {ICE_ABSORPTION_LENGTH_M:.0f} m)\n"
    )

    header = (
        f"{'log10(E/GeV)':>13} {'A_eff^IC':>11} {'V_IC':>8} {'V_range':>8} "
        f"{'V_dynamic':>9} {'eps_stat':>8} {'eps_dyn':>8}"
    )
    print(header)
    for log10_e in (4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0):
        i = int(np.argmin(np.abs(COMMON_LOG10_E - log10_e)))
        print(
            f"{COMMON_LOG10_E[i]:13.1f} {ic_horizon[i]:11.3g} {v_icecube[i]:8.2f} "
            f"{volumes[_RANGE_LABEL][i]:8.2f} {volumes[_DYNAMIC_LABEL][i]:9.2f} "
            f"{eps_static[i]:8.2f} {eps_dynamic[i]:8.2f}"
        )


def main() -> None:
    args = parse_args()

    print(f"Loading IceCube IRFs from: {args.data_dir}")
    icecube, sin_dec_edges = combine_seasons_icecube(args.data_dir)
    sin_dec_centers = 0.5 * (sin_dec_edges[:-1] + sin_dec_edges[1:])

    print("Computing PREM survival probability per declination bin ...")
    d_nu = survival_grid(sin_dec_centers)

    ic_horizon = horizon_detector_area(icecube, sin_dec_edges, d_nu)
    print(f"Fitting light-yield growth length above log10(E/GeV) = {FIT_LOG10_E_MIN:g} ...")
    l_fit_km = fit_light_yield_length_km(ic_horizon)

    print(f"Computing soft-volume target volumes (gamma = {GAMMA}) ...")
    volumes = soft_volume_curves(l_fit_km)
    sigma = CROSS_SECTION.cc(10.0**COMMON_LOG10_E)
    n_nucleon = nucleon_number_density()
    areas = {k: v * CM_PER_KM**3 * n_nucleon * sigma for k, v in volumes.items()}

    report(icecube, sin_dec_edges, d_nu, volumes, areas, l_fit_km)
    make_figure(icecube, sin_dec_edges, d_nu, volumes, areas, l_fit_km, args.out)


if __name__ == "__main__":
    main()
