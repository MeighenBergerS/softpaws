"""Example 62 -- a zenith-shape limit on heavy charged particles from DR2.

Example 61's ceiling counts every event as a candidate; this example fits
instead. In the deposit window the upgoing bands hold ~10^5 events each,
almost all of them atmospheric-neutrino muons, and their zenith shape is
predictable; a flux of Earth-crossing heavy particles adds counts with a
very different shape, the steep imprint of ``E_min(theta, m)``. The fit
profiles the background normalization and limits the signal.

**Background.** MCEq's conventional and prompt fluxes (example 22) folded
through IceCube's *released* IC86 effective area and smearing -- the
analysis-level response, valid at these low energies where this project's
own forward model is not (its turn-on region, examples 45/51). One free
normalization; a 10% per-bin scaled-deviance systematic (example 51's),
which is what actually sets the limit: with ~10^5 events per band the
statistical error is 0.3% while the template is good to percent.

**Signal.** A surface flux ``K (E / 100 TeV)^{-gamma}`` of heavy charged
particles (the shape a steeply falling production spectrum takes; ``gamma``
scanned). A particle arrives if its surface energy exceeds
``E_min(theta, m)`` -- example 61's PREM crossing condition, a step to
percent accuracy by example 60 -- and is selected with example 61's
dim-track efficiency at its charge. Signal counts per band:

.. math:: s_j = T\\,\\Omega_j\\,\\bar A(\\theta_j)\\,\\epsilon(q)\\,
    \\int_{E_{\\min}(\\theta_j)} K\\,(E/10^5)^{-\\gamma}\\,dE .

**The limit.** Profile likelihood over the ten bands; ``K_90`` where the
profiled ``2 Delta lnL`` crosses 2.71. Reported against mass at unit
charge, and against charge at a light mass. Converting ``K_90(m)`` into a
mass limit requires the production spectrum (MCEq + Drell-Yan in
Meighen-Berger et al., PLB 811 (2020) 135929); ``--production-table``
accepts one as ``(E [GeV], flux [GeV^-1 cm^-2 s^-1 sr^-1])`` per mass and
compares it to the limit directly. Without a table the result stays a flux
limit, deliberately.

Scope: the deposit-window containment is taken as 1 (the 0.2 TeV crossing
deposit sits mid-window); the signal zenith dependence within a band is the
band-centre value; downgoing bands (the PLB signal region proper) are not
in the DR2 upgoing selection, so this limit leans on the thin-chord bands
just below the horizon.

Usage
-----
    python examples/62_stau_flux_limit.py
    python examples/62_stau_flux_limit.py --gamma 3.0 3.7
"""

import argparse
import importlib.util
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

from softpaws.data.loader import compute_livetime_s, load_uptime

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_DATA_DIR = _HERE.parent / "src" / "softpaws" / "data" / "dataverse_files"
_DEFAULT_OUT_DIR = _HERE / "output"

#: Per-bin fractional systematic of the background template (example 51's
#: scaled deviance); this, not sqrt(N), limits the analysis.
MODEL_SYS = 0.10

#: Signal spectral indices scanned, and the flux pivot [GeV].
GAMMA_SCAN = (3.0, 3.7)
PIVOT_GEV = 1.0e5

#: Masses [GeV] of the limit curve at unit charge, and charges at the
#: lightest mass.
MASSES_GEV = np.array([100.0, 150.0, 200.0, 320.0, 450.0, 700.0, 1000.0])
CHARGES = np.array([1.0, 0.8, 0.6, 0.5, 0.4, 1.0 / 3.0, 0.3, 0.25])

COLORS = {"data": "k", "bkg": "#7570b3", "sig": "#e7298a", 3.0: "#1b9e77", 3.7: "#e7298a"}


def load_example(stem: str, name: str):
    """Import a sibling example module by file stem."""
    spec = importlib.util.spec_from_file_location(name, _HERE / stem)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EX61 = load_example("61_heavy_track_flux_bound.py", "_example_61")
_EX60 = _EX61._EX60
_EX51 = _EX61._EX51


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=pathlib.Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--gamma", type=float, nargs="+", default=list(GAMMA_SCAN),
                        help="Signal spectral indices.")
    parser.add_argument("--production-table", type=pathlib.Path, default=None,
                        help="CSV of (E [GeV], flux [GeV^-1 cm^-2 s^-1 sr^-1]) to compare "
                             "against the limit (one mass per file).")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the figures, '62a' and '62b'.")
    return parser.parse_args()


def published_band_aeff(data_dir, dec_edges, log10_e_centers):
    """Released IC86 effective area averaged onto the smearing bands [cm^2]."""
    raw = np.genfromtxt(data_dir / "irfs" / "IC86_effectiveArea.csv", comments="#")
    e_lo, e_hi, d_lo, d_hi, area = raw.T
    sin_lo, sin_hi = np.sin(np.deg2rad(d_lo)), np.sin(np.deg2rad(d_hi))
    out = np.zeros((log10_e_centers.size, dec_edges.size - 1))
    for j in range(dec_edges.size - 1):
        band = np.sin(np.deg2rad(dec_edges[j:j + 2]))
        for k, log_e in enumerate(log10_e_centers):
            sel = (e_lo <= log_e) & (log_e < e_hi)
            weight = np.clip(np.minimum(sin_hi[sel], band[1])
                             - np.maximum(sin_lo[sel], band[0]), 0.0, None)
            if weight.sum() > 0.0:
                out[k, j] = (area[sel] * weight).sum() / weight.sum()
    return out


def background_template(data_dir, enu_edges, dec_edges, marginal, window, livetime_s):
    """Expected atmospheric-neutrino counts per band in the deposit window."""
    atm = np.load(_EX51._MCEQ_CACHE)
    centers_dec = 0.5 * (dec_edges[:-1] + dec_edges[1:])
    enu_centers = 0.5 * (enu_edges[:-1] + enu_edges[1:])
    aeff = published_band_aeff(data_dir, dec_edges, enu_centers)
    d_omega = 2.0 * np.pi * np.diff(np.sin(np.deg2rad(dec_edges)))
    counts = np.zeros(dec_edges.size - 1)
    for i in range(enu_edges.size - 1):
        e_bin = np.logspace(enu_edges[i], enu_edges[i + 1], 12)
        flux = np.zeros((e_bin.size, centers_dec.size))
        for comp in ("conv", "prompt"):
            log_f = np.log10(np.clip(atm[comp], 1.0e-99, None))
            on_dec = np.array([np.interp(centers_dec, atm["dec_deg"], log_f[k])
                               for k in range(atm["energy_gev"].size)])
            for j in range(centers_dec.size):
                flux[:, j] += 10.0 ** np.interp(np.log10(e_bin), np.log10(atm["energy_gev"]),
                                                on_dec[:, j])
        per_band = np.trapezoid(flux * aeff[i][None, :], e_bin, axis=0)
        counts += per_band * marginal[i, :, window].sum(axis=0) * d_omega
    return livetime_s * counts


def signal_counts(masses_or_charge, gamma, dec_edges, livetime_s, moments, charge=1.0):
    """Signal counts per band at ``K = 1`` for each mass (or one charge)."""
    centers, columns = _EX61.band_columns(dec_edges)
    d_omega = 2.0 * np.pi * np.diff(np.sin(np.deg2rad(dec_edges)))
    area_cm2 = _EX61.projected_area_km2(centers) * 1.0e10
    eff = float(_EX61.dim_track_efficiency(charge)[0])
    out = np.zeros((np.atleast_1d(masses_or_charge).size, dec_edges.size - 1))
    for k, m in enumerate(np.atleast_1d(masses_or_charge)):
        e_min = _EX61.e_min_gev(moments, columns, float(m), charge)
        with np.errstate(invalid="ignore"):
            integral = PIVOT_GEV / (gamma - 1.0) * (e_min / PIVOT_GEV) ** (1.0 - gamma)
        out[k] = np.where(np.isfinite(e_min),
                          livetime_s * d_omega * area_cm2 * eff * integral, 0.0)
    return out


def profiled_limit(data, background, signal_unit):
    """90% CL on ``K`` by profile likelihood with the scaled deviance.

    The background normalization is profiled; each band's deviance is
    de-weighted by ``1 + MODEL_SYS^2 mu`` exactly as in example 51.
    """
    def deviance(k, b_norm):
        mu = np.clip(b_norm * background + k * signal_unit, 1.0e-12, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            dev = 2.0 * (mu - data + np.where(data > 0, data * np.log(data / mu), 0.0))
        return float(np.sum(dev / (1.0 + MODEL_SYS**2 * mu)))

    def profiled(k):
        result = minimize_scalar(lambda b: deviance(k, b), bounds=(0.1, 10.0),
                                 method="bounded")
        return result.fun

    base = profiled(0.0)
    if signal_unit.sum() <= 0.0:
        return np.inf
    scale = 1.0 / signal_unit.sum()
    lo, hi = 0.0, scale
    while profiled(hi) - base < 2.71:
        hi *= 4.0
        if hi > 1.0e30 * scale:
            return np.inf
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if profiled(mid) - base < 2.71:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _save(fig, out_dir: pathlib.Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {path.resolve()}")
    plt.close(fig)


def figure_bands(centers, data, background, b_fit, example_signal, out_dir) -> None:
    """Counts per band: data, fitted background, an example signal x10."""
    with plt.style.context(str(_STYLE)):
        fig, (ax, ratio) = plt.subplots(2, 1, figsize=(3.2, 3.4), sharex=True,
                                        gridspec_kw={"height_ratios": (2.4, 1.2),
                                                     "hspace": 0.06})
        zenith = 90.0 + centers
        ax.errorbar(zenith, data, yerr=np.sqrt(data), fmt="o", color=COLORS["data"],
                    ms=2.4, lw=0.8, capsize=1.5, label="DR2, deposit window", zorder=5)
        ax.step(zenith, b_fit * background, where="mid", color=COLORS["bkg"], lw=1.2,
                label="atm. neutrinos (fitted norm)")
        ax.step(zenith, example_signal, where="mid", color=COLORS["sig"], lw=1.1, ls="--",
                label=r"signal at $10 \times K_{90}$ (450 GeV)")
        ax.set_yscale("log")
        ax.set_ylabel("events per band")
        ax.legend(fontsize=6, frameon=False, loc="lower left")
        ratio.axhline(1.0, color="0.6", lw=0.7, ls=":")
        band = MODEL_SYS
        ratio.axhspan(1.0 - band, 1.0 + band, color="0.93", zorder=0)
        ratio.errorbar(zenith, data / (b_fit * background),
                       yerr=np.sqrt(data) / (b_fit * background), fmt="o",
                       color=COLORS["data"], ms=2.4, lw=0.8, capsize=1.5)
        ratio.set_xlabel(r"zenith [deg]")
        ratio.set_ylabel("data / bkg")
        _save(fig, out_dir, "62a_stau_limit_bands")


def figure_limits(limits_mass, limits_charge, gammas, out_dir) -> None:
    """``K_90`` against mass (left, unit charge) and against charge (right)."""
    with plt.style.context(str(_STYLE)):
        fig, (ax_m, ax_q) = plt.subplots(1, 2, figsize=(6.0, 2.8),
                                         gridspec_kw={"wspace": 0.3})
        for g in gammas:
            ax_m.plot(MASSES_GEV, limits_mass[g], color=COLORS.get(g, "0.3"), lw=1.2,
                      marker="o", ms=3, label=rf"$\gamma = {g:g}$")
            ax_q.plot(CHARGES, limits_charge[g], color=COLORS.get(g, "0.3"), lw=1.2,
                      marker="o", ms=3)
        ax_m.set_xscale("log")
        ax_m.set_yscale("log")
        ax_m.set_xlabel(r"mass [GeV]")
        ax_m.set_ylabel(r"$K_{90}$ at 100 TeV [GeV$^{-1}$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
        ax_m.legend(fontsize=6, frameon=False, loc="upper left")
        ax_q.set_yscale("log")
        ax_q.set_xlabel(r"charge $\varepsilon_q$ [e]")
        ax_q.set_ylabel(r"$K_{90}$ at 100 TeV (m = 100 GeV)")
        _save(fig, out_dir, "62b_stau_limit_curves")


def main() -> None:
    args = parse_args()
    print("Inputs: smearing bands, deposit-window counts, MCEq background ...")
    ex35 = load_example("35_point_source_effective_area.py", "_e35")
    ex45 = load_example("45_first_principles_reach.py", "_e45")
    ex46 = load_example("46_declination_resolved_reach.py", "_e46")
    enu_edges, dec_edges, marginal, _ = _EX51.fit_inputs(ex35, ex45, ex46, args.data_dir,
                                                          False)
    counts = _EX51.binned_events(args.data_dir, dec_edges)
    reco_centers = 0.5 * (_EX51.RECO_EDGES[:-1] + _EX51.RECO_EDGES[1:])
    window = ((reco_centers >= _EX61.DEPOSIT_WINDOW[0])
              & (reco_centers <= _EX61.DEPOSIT_WINDOW[1]))
    data = counts[window].sum(axis=0)
    livetime_s = sum(compute_livetime_s(load_uptime(args.data_dir / "uptime" / f"{s}_exp.csv"))
                     for s in _EX51.IC86_SEASONS)
    background = background_template(args.data_dir, enu_edges, dec_edges, marginal, window,
                                     livetime_s)
    centers, _ = _EX61.band_columns(dec_edges)
    print(f"  window {tuple(_EX61.DEPOSIT_WINDOW)}: data {data.sum():,.0f}, "
          f"raw template {background.sum():,.0f} "
          f"(template / data = {background.sum() / data.sum():.2f})")

    moments = _EX60.muon_channel_moments()
    limits_mass, limits_charge = {}, {}
    print("\n  K_90 [GeV^-1 cm^-2 s^-1 sr^-1 at 100 TeV], unit charge:")
    print(f"  {'gamma':>6} " + " ".join(f"{m:>9.0f}" for m in MASSES_GEV))
    for g in args.gamma:
        unit = signal_counts(MASSES_GEV, g, dec_edges, livetime_s, moments)
        limits_mass[g] = np.array([profiled_limit(data, background, unit[k])
                                   for k in range(MASSES_GEV.size)])
        print(f"  {g:6.1f} " + " ".join(f"{v:9.2e}" for v in limits_mass[g]))
    print("\n  K_90 against charge (m = 100 GeV):")
    print(f"  {'gamma':>6} " + " ".join(f"{q:>9.2f}" for q in CHARGES))
    for g in args.gamma:
        row = []
        for q in CHARGES:
            unit = signal_counts([100.0], g, dec_edges, livetime_s, moments, charge=float(q))
            row.append(profiled_limit(data, background, unit[0]))
        limits_charge[g] = np.array(row)
        print(f"  {g:6.1f} " + " ".join(f"{v:9.2e}" for v in row))

    if args.production_table is not None:
        table = np.loadtxt(args.production_table, delimiter=",")
        k_at_pivot = float(np.interp(np.log10(PIVOT_GEV), np.log10(table[:, 0]),
                                     np.log10(np.clip(table[:, 1], 1e-99, None))))
        print(f"\n  production table at 100 TeV: 10^{k_at_pivot:.2f}; compare to K_90 above.")

    g0 = args.gamma[-1]
    unit_450 = signal_counts([450.0], g0, dec_edges, livetime_s, moments)[0]
    result = minimize_scalar(
        lambda b: float(np.sum((background * b - data) ** 2 / np.maximum(data, 1.0))),
        bounds=(0.1, 10.0), method="bounded")
    b_fit = float(result.x)
    print(f"\n  fitted background normalization: {b_fit:.2f}")
    figure_bands(centers, data, background, b_fit,
                 10.0 * limits_mass[g0][MASSES_GEV.tolist().index(450.0)] * unit_450,
                 args.out_dir)
    figure_limits(limits_mass, limits_charge, args.gamma, args.out_dir)


if __name__ == "__main__":
    main()
