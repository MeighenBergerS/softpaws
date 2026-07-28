"""Example 20 — soft-volume target volume vs. the published IceCube A_eff.

Extends ``03_effective_area.py``'s livetime-combined IceCube effective area with
the target volume implied by the soft-volume forward model
(:mod:`softpaws.response.soft_volume`). The target-volume factorization
(arXiv:2607.13143, Eq. 2.20) gives ``dN/dE = V_target(E) n_N sigma_CC(E)
phi_nu(E)``, so ``A_eff = V_target n_N sigma_CC`` is nominally comparable to the
published ``A_eff(E_nu, dec)`` -- and needs no flux normalization fit.

Making that comparison honest takes three corrections, all applied here.

**Earth attenuation.** The published table is tabulated in *true* neutrino
declination and has absorption folded in: at 80 PeV it falls by a factor ~470
from the horizon to the nadir. The soft-volume model assumes ``D_nu = 1``. Both
sides are therefore put on the same footing with the per-declination PREM
survival probability (:mod:`softpaws.transport.attenuation`): the model curves
are attenuated bin by bin before the hemisphere average, and the horizon-band
IceCube curve is divided by ``D_nu`` to recover a detector-only effective area.
The horizon band keeps that deconvolution below a factor ~2 at all energies.

**Convention.** ``V_soft(E)`` is differential in the *observed muon* energy and
is already spectrally weighted -- its length ``1/(b_mu A)`` is a spectral
attenuation length, finite only because the parent flux falls, and so nearly
energy independent. The published ``A_eff(E_nu)`` fixes the *neutrino* energy and
integrates over every muon energy that survives the selection, so its length is
the muon range down to threshold, which grows logarithmically. The range curve
(:meth:`~softpaws.response.soft_volume.SoftVolumeResponse.threshold_effective_area_cm2`)
is the like-for-like quantity; the soft-volume curves are shown alongside it to
make the size of the convention difference visible.

**Efficiency.** What is left after both corrections is the event selection. The
third panel reports it as ``epsilon(E) = A_eff^IceCube / A_eff^range``, a
turn-on curve rather than a discrepancy.

Volumes, not effective areas, carry the physics content: ``sigma_CC`` is common
to both sides and only steepens every curve by the same ``E^lambda``, so panel
(b) divides it out.

The energy axis stops at 100 PeV, the upper limit of the simulation behind the
published response (``DR2_readme.txt``); the tabulated bins above it are
extrapolation.

Usage
-----
    python examples/20_effective_area_soft_vs_irf.py
    python examples/20_effective_area_soft_vs_irf.py --data-dir /path/to/dataverse_files
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.data.loader import compute_livetime_s, load_uptime, parse_aeff
from softpaws.data.schema import SEASONS
from softpaws.response.soft_volume import SoftVolumeResponse
from softpaws.transport.attenuation import prem_column, survival_probability
from softpaws.transport.soft_volume import (
    DEFAULT_MUON_THRESHOLD_GEV,
    range_target_volume_km3,
    sphere_radius_from_volume,
)
from softpaws.transport.source import (
    MEAN_INELASTICITY,
    cc_cross_section,
    nucleon_number_density,
)
from softpaws.utils.constants import CM_PER_KM

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

_RANGE_LABEL = r"range, $E_{\rm thr} = 1$ TeV"
_SOFT_STYLES = {
    "drift": ("--", "C0"),
    "exact, infinite column": (":", "C1"),
    f"exact, finite column ({DOWNGOING_COLUMN_KM:g} km)": ("-.", "C2"),
    _RANGE_LABEL: ("-", "C3"),
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
        default=_DEFAULT_OUT_DIR / "20_effective_area_soft_vs_irf.pdf",
        help="Output file for the figure.",
    )
    return parser.parse_args()


def _canonical_irf_season(season: str) -> str:
    return "IC86" if season.startswith("IC86") else season


def combine_seasons_icecube(data_dir: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Livetime-weighted IceCube effective area, resolved in declination.

    Unlike ``03_effective_area.py``'s hemisphere-averaged version, the
    declination structure is kept, because the per-declination Earth attenuation
    has to be applied before any average over solid angle.

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
        irf_season = _canonical_irf_season(season)
        if irf_season not in aeff_cache:
            raw = np.genfromtxt(irf_dir / f"{irf_season}_effectiveArea.csv", comments="#")
            aeff_cache[irf_season] = parse_aeff(raw)
        aeff = aeff_cache[irf_season]

        uptime = load_uptime(uptime_dir / f"{season}_exp.csv")
        livetime_s = compute_livetime_s(uptime)

        # Interpolate each declination column onto the common energy grid.
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

    Parameters
    ----------
    sin_dec_centers : np.ndarray, shape (n_dec,)
        Bin centres in ``sin(dec)``.

    Returns
    -------
    d_nu : np.ndarray, shape (n_energy, n_dec)
        PREM survival probability, in ``[0, 1]``.
    """
    dec_deg = np.rad2deg(np.arcsin(sin_dec_centers))
    columns = np.array([prem_column(d) for d in dec_deg])
    return survival_probability(10.0**COMMON_LOG10_E[:, None], columns[None, :])


def band_average(values: np.ndarray, sin_dec_edges: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Solid-angle-weighted average of an (energy, declination) grid over a band.

    Parameters
    ----------
    values : np.ndarray, shape (n_energy, n_dec)
        Grid to average.
    sin_dec_edges : np.ndarray, shape (n_dec + 1,)
        Bin edges in ``sin(dec)``; the widths are the solid-angle weights.
    mask : np.ndarray, shape (n_dec,)
        Boolean selection of declination bins.

    Returns
    -------
    average : np.ndarray, shape (n_energy,)
        Weighted average over the selected bins.
    """
    weights = np.diff(sin_dec_edges)[mask]
    return np.average(values[:, mask], axis=1, weights=weights)


def horizon_detector_area(
    icecube: np.ndarray,
    sin_dec_edges: np.ndarray,
    d_nu: np.ndarray,
) -> np.ndarray:
    """Detector-only IceCube effective area in the horizon band.

    Divides the per-declination Earth absorption back out of the published
    values, then averages over solid angle. Restricting to the horizon keeps the
    deconvolution factor below ~2 at every energy; over the full hemisphere it
    would reach ``1e3`` and amplify the tabulation systematics with it.

    Parameters
    ----------
    icecube : np.ndarray, shape (n_energy, n_dec)
        Published effective area [cm^2].
    sin_dec_edges : np.ndarray, shape (n_dec + 1,)
        Bin edges in ``sin(dec)``.
    d_nu : np.ndarray, shape (n_energy, n_dec)
        Neutrino survival probability.

    Returns
    -------
    aeff : np.ndarray, shape (n_energy,)
        Absorption-corrected effective area [cm^2].
    """
    centers = 0.5 * (sin_dec_edges[:-1] + sin_dec_edges[1:])
    horizon = np.abs(centers) < HORIZON_SIN_DEC
    # Mask first, so the opaque bins (where D_nu underflows to zero) are never
    # divided by.
    corrected = icecube[:, horizon] / d_nu[:, horizon]
    return np.average(corrected, axis=1, weights=np.diff(sin_dec_edges)[horizon])


def selection_efficiency(icecube_horizon: np.ndarray, range_area: np.ndarray) -> np.ndarray:
    """Ratio of the corrected IceCube area to the range model's geometric ceiling.

    Parameters
    ----------
    icecube_horizon : np.ndarray, shape (n_energy,)
        Absorption-corrected IceCube effective area [cm^2].
    range_area : np.ndarray, shape (n_energy,)
        Range-model effective area [cm^2], zero below the muon threshold.

    Returns
    -------
    efficiency : np.ndarray, shape (n_energy,)
        Implied efficiency; NaN where the range model vanishes.
    """
    return np.divide(
        icecube_horizon,
        range_area,
        out=np.full_like(icecube_horizon, np.nan),
        where=range_area > 0.0,
    )


def soft_volume_curves() -> dict[str, np.ndarray]:
    """Soft-volume implied target volume [km^3] on ``COMMON_LOG10_E``.

    Returns
    -------
    curves : dict of str to np.ndarray
        Target volume [km^3] for each model variant. The three soft-volume
        variants are evaluated at ``gamma = GAMMA`` and are differential in the
        observed muon energy; the range variant is flux independent and takes the
        neutrino energy.
    """
    energy_gev = 10.0**COMMON_LOG10_E

    drift = SoftVolumeResponse(radius_km=RADIUS_KM, method="drift")
    exact_inf = SoftVolumeResponse(radius_km=RADIUS_KM, method="exact", column_depth_km=None)
    exact_fin = SoftVolumeResponse(
        radius_km=RADIUS_KM, method="exact", column_depth_km=DOWNGOING_COLUMN_KM
    )
    to_km3 = 1.0 / CM_PER_KM**3
    return {
        "drift": drift.target_volume_cm3(energy_gev, GAMMA) * to_km3,
        "exact, infinite column": exact_inf.target_volume_cm3(energy_gev, GAMMA) * to_km3,
        f"exact, finite column ({DOWNGOING_COLUMN_KM:g} km)": (
            exact_fin.target_volume_cm3(energy_gev, GAMMA) * to_km3
        ),
        # The range variant is a function of the neutrino energy, so the muon is
        # born at (1 - <y_w>) E_nu. Multiplying this by n_N sigma_CC(E_nu)
        # reproduces SoftVolumeResponse.threshold_effective_area_cm2 exactly.
        _RANGE_LABEL: range_target_volume_km3(
            RADIUS_KM,
            (1.0 - MEAN_INELASTICITY) * energy_gev,
            DEFAULT_MUON_THRESHOLD_GEV,
        ),
    }


def make_figure(
    icecube: np.ndarray,
    sin_dec_edges: np.ndarray,
    d_nu: np.ndarray,
    volumes: dict[str, np.ndarray],
    areas: dict[str, np.ndarray],
    out_path: pathlib.Path,
) -> None:
    """Draw the three-panel comparison and write it to disk."""
    sin_dec_centers = 0.5 * (sin_dec_edges[:-1] + sin_dec_edges[1:])
    upgoing = sin_dec_centers > 0.0
    downgoing = sin_dec_centers < 0.0

    ic_up = band_average(icecube, sin_dec_edges, upgoing)
    ic_down = band_average(icecube, sin_dec_edges, downgoing)
    ic_horizon = horizon_detector_area(icecube, sin_dec_edges, d_nu)

    sigma = cc_cross_section(10.0**COMMON_LOG10_E)
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
        # Two legends: the data curves, and what the model's two line weights mean.
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
            ax.plot(COMMON_LOG10_E, volume, ls=ls, color=color, lw=1.0, label=name)
        ax.set_yscale("log")
        ax.set_ylim(0.3, 2e2)
        ax.set_ylabel(r"$V_{\rm target}$ [km$^3$]")
        ax.set_title("(b) implied target volume", fontsize=7)
        ax.legend(fontsize=5.5, loc="upper left")

        # (c) What is left over: the selection turn-on.
        ax = axes[2]
        efficiency = selection_efficiency(ic_horizon, areas[_RANGE_LABEL])
        ax.plot(COMMON_LOG10_E, efficiency, color="k", lw=1.6)
        ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
        ax.set_ylim(0.0, 1.3)
        ax.set_ylabel(r"$\varepsilon = A_{\rm eff}^{\rm IC} / A_{\rm eff}^{\rm range}$")
        ax.set_title("(c) implied selection efficiency", fontsize=7)

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
) -> None:
    """Print the horizon-band numbers behind panels (b) and (c)."""
    ic_horizon = horizon_detector_area(icecube, sin_dec_edges, d_nu)
    sigma = cc_cross_section(10.0**COMMON_LOG10_E)
    v_icecube = ic_horizon / (nucleon_number_density() * sigma) / CM_PER_KM**3
    efficiency = selection_efficiency(ic_horizon, areas[_RANGE_LABEL])

    header = f"{'log10(E/GeV)':>13} {'A_eff^IC':>11} {'V_IC':>8} {'V_drift':>8} {'V_range':>8}"
    print(f"{header} {'eff':>6}")
    for log10_e in (4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0):
        i = int(np.argmin(np.abs(COMMON_LOG10_E - log10_e)))
        print(
            f"{COMMON_LOG10_E[i]:13.1f} {ic_horizon[i]:11.3g} {v_icecube[i]:8.2f} "
            f"{volumes['drift'][i]:8.2f} {volumes[_RANGE_LABEL][i]:8.2f} {efficiency[i]:6.2f}"
        )


def main() -> None:
    args = parse_args()

    print(f"Loading IceCube IRFs from: {args.data_dir}")
    icecube, sin_dec_edges = combine_seasons_icecube(args.data_dir)
    sin_dec_centers = 0.5 * (sin_dec_edges[:-1] + sin_dec_edges[1:])

    print("Computing PREM survival probability per declination bin ...")
    d_nu = survival_grid(sin_dec_centers)

    print(f"Computing soft-volume target volumes (gamma = {GAMMA}) ...")
    volumes = soft_volume_curves()
    sigma = cc_cross_section(10.0**COMMON_LOG10_E)
    n_nucleon = nucleon_number_density()
    areas = {k: v * CM_PER_KM**3 * n_nucleon * sigma for k, v in volumes.items()}

    report(icecube, sin_dec_edges, d_nu, volumes, areas)
    make_figure(icecube, sin_dec_edges, d_nu, volumes, areas, args.out)


if __name__ == "__main__":
    main()
