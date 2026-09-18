"""Example 48 -- what loss fluctuations do to the range, and what that costs.

The first-passage range is shorter than the continuous-slowing-down one --
``Phi'(0) = <-ln(1-y)>`` exceeds ``b_mu = <y>`` for every positive loss
spectrum, so one hard bremsstrahlung ends a track the mean-loss picture keeps
coasting. This example shows the effect at both levels of the paper's
validation ladder.

Figure (a) is the muon level, head to head with Monte Carlo: muons are
propagated in PROPOSAL (the three radiative channels of the shipped
coefficient table plus ionization) down to a 10 TeV threshold, and their mean
stopping depth is drawn against the closed-form first-passage range and the
mean-loss law, everything as a ratio to the mean-loss range so the effect is
legible. The PROPOSAL points land on the first-passage curve -- 0.98-1.03,
agreement within the Monte Carlo statistics -- while the mean-loss law runs
9-23% long. The 10 TeV threshold is what makes the comparison a test of the
exponent alone: the last e-fold above a 1 TeV threshold sits at the muon
critical energy, where ionization dominates, and ionization removes a fixed
energy per length and not a fixed fraction, so it cannot live inside the
subordinator -- the closed form stitches in a deterministic mean-loss
segment there. Rerunning with ``--threshold-gev 1e3`` shows that boundary:
the closed form comes out 8-18% long at 10^4-10^5 GeV against a PROPOSAL
that keeps fluctuating through the splice region, still far inside the
mean-loss gap. The exponent is exact where the transport is radiative; the
splice is the stated boundary.

Figure (b) is the response level: the banded IceCube model built twice, once
per range law with every other ingredient identical, against the published
DR2 declination bands. The mean-loss law inflates the effective area by
11-13% and adds a tilt of about -0.02 dex per decade, growing the residual
scatter by half in the two near-horizon bands where Earth absorption is a
percent-level effect and the shape is transport alone.

What this is, and is not
------------------------
The DR2 tables are simulation products -- IceCube propagates muons with
PROPOSAL -- so figure (b) is a consistency statement at the response level:
any analytic effective-area model built on a mean-loss range inherits the
tilt, and the closed form removes it without Monte Carlo. Event data cannot
see the difference: a tilt of 0.02 dex per decade is exactly degenerate with
a spectral-index shift of 0.02, far inside the published index uncertainty.

The muon-level ratio is density invariant, so figure (a) is computed in water
(the library's reference medium) while figure (b) runs in ice.

Usage
-----
    python scripts/2026_muon_transport/48_fluctuation_cost.py
    python scripts/2026_muon_transport/48_fluctuation_cost.py --threshold-gev 1e3
    python scripts/2026_muon_transport/48_fluctuation_cost.py --first-principles
"""

import argparse
import importlib.util
import pathlib
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from softpaws.constants import CM_PER_KM
from softpaws.response import declination as _declination
from softpaws.response import first_principles as _first_principles
from softpaws.transport.coefficients import proposal_parametrizations
from softpaws.transport.muon_range import muon_range_km, stochastic_muon_range_km

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"
_DEFAULT_DATA_DIR = _HERE.parents[1] / "src" / "softpaws" / "data" / "dataverse_files"

#: Cached PROPOSAL stopping depths, so the figure re-renders without
#: re-propagating. ``--resample`` refreshes it.
_SAMPLE_CACHE = _DEFAULT_OUT_DIR / "48_proposal_ranges.npz"

def sample_grid(threshold_gev: float) -> np.ndarray:
    """Energies the Monte Carlo samples, from a decade above threshold [log10 GeV]."""
    return np.arange(max(4.0, np.log10(threshold_gev) + 1.0), 8.01, 0.5)

#: Band the response-level residuals are scored over, matching examples 45-47.
STATS_LOG10_E = (5.0, 7.8)

#: Near-horizon declination bands figure (b) draws, in ``sin(dec)``. Earth
#: absorption is a percent-level effect in both, so the residual shape there
#: is transport alone.
SHOW_SIN_DEC = (0.02, 0.22)

#: Colour carries the range law and line style carries the band, so the two
#: legends factorize as in example 45.
LAW_COLOR = {"first passage": "#7570b3", "mean loss": "#e7298a"}
BAND_STYLE = ("-", "--")


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
                        help="Root of the DR2 release.")
    parser.add_argument("--n-muons", type=int, default=1000,
                        help="Muons propagated per sampled energy.")
    parser.add_argument("--threshold-gev", type=float, default=1.0e4,
                        help="Muon threshold of figure (a). The 10 TeV default "
                             "tests the exponent alone; 1e3 exposes the "
                             "ionization-splice boundary.")
    parser.add_argument("--resample", action="store_true",
                        help="Re-propagate instead of reading the cached depths.")
    parser.add_argument("--first-principles", action="store_true",
                        help="Drop example 45's fitted attenuation override and "
                             "normalization and keep the derived optics.")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the two figures, '48a' and '48b'.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Figure (a): the muon level against PROPOSAL
# ---------------------------------------------------------------------------


def proposal_mean_ranges_km(log10_e: np.ndarray, n_muons: int,
                            threshold_gev: float, seed: int = 8):
    """Mean PROPOSAL stopping depth per sampled energy [km].

    Propagates muons in water with the shipped coefficient table's three
    radiative channels plus Bethe-Bloch ionization, losses above 500 MeV
    sampled stochastically, and reads the depth at which each muon first
    falls below ``threshold_gev`` -- the same first-crossing convention the
    closed form computes.

    Parameters
    ----------
    log10_e : np.ndarray
        Sampled energies [log10 GeV].
    n_muons : int
        Muons per energy.
    threshold_gev : float
        Muon threshold [GeV].
    seed : int, optional
        Seed for PROPOSAL's random generator.

    Returns
    -------
    mean_km : np.ndarray
        Mean stopping depth [km] on ``log10_e``.
    error_km : np.ndarray
        Standard error of the mean [km].
    """
    import proposal as pp

    particle = pp.particle.MuMinusDef()
    medium = pp.medium.Water()
    cuts = pp.EnergyCutSettings(500.0, 1, False)
    params = list(proposal_parametrizations().values())
    params.append(pp.parametrization.ionization.BetheBlochRossi(cuts))
    cross = [pp.crosssection.make_crosssection(p, particle, medium, cuts, True)
             for p in params]
    collection = pp.PropagationUtilityCollection()
    collection.displacement = pp.make_displacement(cross, True)
    collection.interaction = pp.make_interaction(cross, True)
    collection.time = pp.make_time(cross, particle, True)
    utility = pp.PropagationUtility(collection=collection)
    geometry = pp.geometry.Sphere(pp.Cartesian3D(0, 0, 0), 1.0e20)
    density = pp.density_distribution.density_homogeneous(medium.mass_density)
    propagator = pp.Propagator(particle, [(geometry, utility, density)])
    pp.RandomGenerator.get().set_seed(seed)

    threshold_mev = threshold_gev * 1.0e3
    mean_km = np.empty(log10_e.size)
    error_km = np.empty(log10_e.size)
    for i, sample in enumerate(log10_e):
        depths = np.empty(n_muons)
        for k in range(n_muons):
            state = pp.particle.ParticleState()
            state.type = particle.particle_type
            state.position = pp.Cartesian3D(0, 0, 0)
            state.direction = pp.Cartesian3D(0, 0, 1)
            state.energy = 10.0 ** (sample + 3.0)
            state.propagated_distance = 0.0
            track = propagator.propagate(state, max_distance=1.0e12,
                                         min_energy=threshold_mev)
            depths[k] = track.final_state().propagated_distance / CM_PER_KM
        mean_km[i] = depths.mean()
        error_km[i] = depths.std() / np.sqrt(n_muons)
        print(f"  log10 E = {sample:.1f}: {mean_km[i]:7.3f} +- {error_km[i]:.3f} km")
    return mean_km, error_km


def sampled_ranges(n_muons: int, threshold_gev: float, resample: bool):
    """PROPOSAL mean ranges, from the cache unless a resample is asked for."""
    grid = sample_grid(threshold_gev)
    if _SAMPLE_CACHE.exists() and not resample:
        cache = np.load(_SAMPLE_CACHE)
        if (np.array_equal(cache["log10_e"], grid)
                and float(cache.get("threshold_gev", 0.0)) == threshold_gev):
            print(f"  cached PROPOSAL depths from {_SAMPLE_CACHE.name}")
            return grid, cache["mean_km"], cache["error_km"]
    print("Propagating in PROPOSAL ...")
    mean_km, error_km = proposal_mean_ranges_km(grid, n_muons, threshold_gev)
    _SAMPLE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(_SAMPLE_CACHE, log10_e=grid, mean_km=mean_km,
             error_km=error_km, n_muons=n_muons, threshold_gev=threshold_gev)
    return grid, mean_km, error_km


def figure_range(sample_log10_e, mean_km, error_km, threshold_gev: float,
                 out_dir: pathlib.Path) -> None:
    """Range to threshold as a ratio to the mean-loss law, with PROPOSAL points."""
    grid = np.linspace(sample_log10_e[0], sample_log10_e[-1], 121)
    energy = 10.0**grid
    csda = muon_range_km(energy, threshold_gev)
    first_passage = stochastic_muon_range_km(energy, threshold_gev)
    csda_at_samples = muon_range_km(10.0**sample_log10_e, threshold_gev)

    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.0, 3.0))
        ax.axhline(1.0, color=LAW_COLOR["mean loss"], lw=1.2, label="mean loss")
        ax.plot(grid, first_passage / csda, color=LAW_COLOR["first passage"],
                lw=1.2, label="first passage")
        ax.errorbar(sample_log10_e, mean_km / csda_at_samples,
                    yerr=error_km / csda_at_samples, fmt="o", color="k",
                    ms=2.5, lw=0.8, capsize=1.5, label="PROPOSAL", zorder=5)
        ax.set_xlim(sample_log10_e[0], sample_log10_e[-1])
        ax.set_ylim(0.70, 1.05)
        ax.set_xlabel(r"$\log_{10}(E_\mu\,/\,\mathrm{GeV})$")
        exponent = int(np.round(np.log10(threshold_gev)))
        ax.set_ylabel(rf"range to $10^{{{exponent}}}$ GeV $/$ mean-loss range")
        ax.legend(fontsize=8, frameon=False, loc="lower left")
        _save(fig, out_dir, "48a_range_vs_proposal")


# ---------------------------------------------------------------------------
# Figure (b): the response level against the published bands
# ---------------------------------------------------------------------------


def csda_range(energy_gev, threshold_gev, density_g_cm3):
    """Mean-loss range with the shipped signature of the stochastic one [km]."""
    return muon_range_km(energy_gev, threshold_gev, density_g_cm3)


def csda_truncated(energy_gev, column_km, threshold_gev, density_g_cm3):
    """Mean-loss range cut at the available column [km]."""
    return np.minimum(muon_range_km(energy_gev, threshold_gev, density_g_cm3),
                      np.asarray(column_km, dtype=float))


def banded(ex35, ex45, ex46, site, site45, sin_dec_edges, efficiency,
           mean_loss: bool) -> np.ndarray:
    """Banded IceCube model under one range law [cm^2].

    Parameters
    ----------
    ex35, ex45, ex46 : ModuleType
        Examples 35, 45 and 46.
    site : ex35.Site
        IceCube geometry and medium.
    site45 : ex45.Site
        IceCube optics, with the fitted override where requested.
    sin_dec_edges : np.ndarray
        Published band edges in ``sin(dec)``.
    efficiency : float
        Flat normalization applied to the model.
    mean_loss : bool
        Substitute the continuous-slowing-down law for the first-passage
        range, leaving every other ingredient untouched.

    Returns
    -------
    aeff : np.ndarray
        Effective area [cm^2] per band on ``ex35.COMMON_LOG10_E``.
    """
    # The range enters through the library modules the banded model calls, not
    # through the example namespaces the cleanup emptied.
    shipped = (_first_principles.stochastic_muon_range_km,
               _declination.truncated_muon_range_km)
    if mean_loss:
        _first_principles.stochastic_muon_range_km = csda_range
        _declination.truncated_muon_range_km = csda_truncated
    try:
        return ex46.model_banded(ex35, ex45, site, site45, sin_dec_edges, 8.0,
                                 ("mu",), efficiency=efficiency)
    finally:
        (_first_principles.stochastic_muon_range_km,
         _declination.truncated_muon_range_km) = shipped


def report(published, phi_model, csda_model, sin_dec_centers, log10_e, band) -> None:
    """Print level, tilt and scatter per shown band under both range laws."""
    for label, model in (("first passage", phi_model), ("mean loss", csda_model)):
        print(f"\n  {label}")
        print(f"  {'sin(dec)':>9} {'level':>7} {'tilt':>8} {'scatter':>8}")
        for target in SHOW_SIN_DEC:
            j = int(np.argmin(np.abs(sin_dec_centers - target)))
            res = np.log10(published[band, j] / model[band, j])
            tilt = np.polyfit(log10_e[band], res, 1)[0]
            print(f"  {sin_dec_centers[j]:9.2f} {10.0**res.mean():7.3f} "
                  f"{tilt:+8.3f} {res.std():8.4f}")


def figure_bands(published, phi_model, csda_model, sin_dec_centers, log10_e,
                 band, out_dir: pathlib.Path) -> None:
    """Band effective areas under both range laws, with the ratio beneath."""
    with plt.style.context(str(_STYLE)):
        fig, (ax_area, ax_ratio) = plt.subplots(
            2, 1, figsize=(3.0, 4.4), sharex=True,
            gridspec_kw={"height_ratios": (2.0, 1.2), "hspace": 0.06})
        for target, style in zip(SHOW_SIN_DEC, BAND_STYLE):
            j = int(np.argmin(np.abs(sin_dec_centers - target)))
            ax_area.plot(log10_e[band], published[band, j], color="0.75", lw=2.4,
                         ls=style, alpha=0.8, zorder=1)
            for label, model in (("first passage", phi_model),
                                 ("mean loss", csda_model)):
                ax_area.plot(log10_e[band], model[band, j], color=LAW_COLOR[label],
                             ls=style, lw=1.1, zorder=2)
                ax_ratio.plot(log10_e[band], published[band, j] / model[band, j],
                              color=LAW_COLOR[label], ls=style, lw=1.1)
        ax_area.set_yscale("log")
        ax_area.set_ylabel(r"$A_{\rm eff}$ [cm$^2$]")
        law_handles = [plt.Line2D([], [], color="0.75", lw=2.4, alpha=0.8,
                                  label="published")]
        law_handles += [plt.Line2D([], [], color=c, lw=1.1, label=n)
                        for n, c in LAW_COLOR.items()]
        law_legend = ax_area.legend(handles=law_handles, loc="upper left",
                                    fontsize=8, frameon=False, handlelength=2.0)
        ax_area.add_artist(law_legend)
        ax_area.legend(
            handles=[plt.Line2D([], [], color="k", ls=s, lw=1.0,
                                label=rf"$\sin\delta = {t:g}$")
                     for t, s in zip(SHOW_SIN_DEC, BAND_STYLE)],
            loc="lower right", fontsize=8, frameon=False, handlelength=2.4,
        )
        ax_ratio.axhline(1.0, color="0.6", lw=0.8, ls=":", zorder=1)
        ax_ratio.set_ylim(0.6, 1.3)
        ax_ratio.set_xlim(*STATS_LOG10_E)
        ax_ratio.set_xlabel(r"$\log_{10}(E_\nu\,/\,\mathrm{GeV})$")
        ax_ratio.set_ylabel(r"published $/$ model")
        _save(fig, out_dir, "48b_band_effective_areas")


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

    print("Figure (a): the muon level against PROPOSAL")
    grid, mean_km, error_km = sampled_ranges(args.n_muons, args.threshold_gev,
                                             args.resample)
    csda_at = muon_range_km(10.0**grid, args.threshold_gev)
    phi_at = stochastic_muon_range_km(10.0**grid, args.threshold_gev)
    print(f"  {'log10 E':>8} {'PROPOSAL/CSDA':>14} {'Phi/CSDA':>9} {'Phi/PROPOSAL':>13}")
    for i, sample in enumerate(grid):
        print(f"  {sample:8.1f} {mean_km[i] / csda_at[i]:14.3f} "
              f"{phi_at[i] / csda_at[i]:9.3f} {phi_at[i] / mean_km[i]:13.3f}")

    print("\nFigure (b): the response level against the published bands")
    print("Loading examples 35, 45 and 46 ...")
    ex35 = load_example("35_point_source_effective_area.py", "_example_35")
    ex45 = load_example("45_first_principles_reach.py", "_example_45")
    ex46 = load_example("46_declination_resolved_reach.py", "_example_46")

    site = ex35.build_sites()[0]
    site45 = ex45.ICECUBE_SITE
    efficiency = 1.0
    if not args.first_principles:
        site45 = replace(site45, attenuation_override_m=42.0)
        efficiency = 0.755
        print("  fitted configuration: Lambda 42 m, normalization 0.755")

    sin_dec_edges, published = ex35.icecube_banded(args.data_dir)
    sin_dec_centers = 0.5 * (sin_dec_edges[:-1] + sin_dec_edges[1:])
    log10_e = ex35.COMMON_LOG10_E
    band = (log10_e >= STATS_LOG10_E[0]) & (log10_e <= STATS_LOG10_E[1])

    print("Building the banded model, first-passage range ...")
    phi_model = banded(ex35, ex45, ex46, site, site45, sin_dec_edges, efficiency,
                       mean_loss=False)
    print("Building the banded model, mean-loss range ...")
    csda_model = banded(ex35, ex45, ex46, site, site45, sin_dec_edges, efficiency,
                        mean_loss=True)
    report(published, phi_model, csda_model, sin_dec_centers, log10_e, band)

    print()
    figure_range(grid, mean_km, error_km, args.threshold_gev, args.out_dir)
    figure_bands(published, phi_model, csda_model, sin_dec_centers, log10_e, band,
                 args.out_dir)


if __name__ == "__main__":
    main()
