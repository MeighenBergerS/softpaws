"""Example 55 -- P-ONE and TRIDENT effective areas as parameter-free predictions.

Examples 45 to 47 set the model's two water-site numbers -- the flat
attenuation length and the normalization -- against KM3NeT/ARCA230's
sky-averaged curve, and example 47 showed the same configuration carries the
*angular* dependence of ARCA's Fig. 7(b). This example takes the same water
configuration, unchanged, to two detectors it has never seen: P-ONE (seven
clusters of 120 m radius and 1 km height at 2.66 km, ICRC2023 performance
study, trigger level, six zenith bands plus the all-sky curve) and TRIDENT
(a 2 km-radius, 570 m-high array at 3.1 km, Extended Data Fig. 8 of the
Nature Astronomy paper, three ``cos(theta)`` bands with a 6-degree
angular-error cut). Only the instrumented footprint and the depth change;
the optics, the coincidence rule, the reach and the normalization are
ARCA's. Every curve here is therefore a prediction, not a fit.

The published curves were digitized from step-function plots, so the raw
points carry the risers and the reader's hand. They are smoothed here --
median in 0.1-dex energy bins, then a five-bin running mean in ``log10``
-- and drawn as a band of ``+-`` :data:`BAND_DEX` around the smoothed
curve, a generous allowance for the digitization and the selection-level
conventions the two collaborations use.

Conventions. Both releases are in m^2 and are converted to cm^2. P-ONE
bins in zenith angle with 180 degrees the nadir; TRIDENT bins in
``cos(theta)`` with ``-1`` the nadir. Both map onto the model's
``cos(theta) = cos(zenith)``, negative for upgoing. P-ONE's energy axis is
in GeV, TRIDENT's in ``log10(E / GeV)``. The model is ``nu_mu`` only,
neutrino and antineutrino averaged, matching what both collaborations
simulate.

Earth transmission. Both collaborations weight their events with a
survival probability, ``exp(-N sigma_tot X)``, and no neutral-current
down-scattering (P-ONE through LeptonWeighter; TRIDENT states no Earth
treatment at all). The model's regeneration ladder therefore sits above
their upgoing bands by construction, and the default here is pure
absorption to match them; ``--with-regeneration`` restores the physics
default of the rest of the project. With the ladder in, TRIDENT's upgoing
band reads 0.67 with a -0.2 dex/decade tilt; without it 0.87 with the same
mild slope as every other band.

What to read off. The upgoing bands test the Earth column and the reach
together; the downgoing bands test the overburden and the geometric
footprint alone, since there is no Earth in the way. A level offset common
to all bands is the selection layer (trigger for P-ONE, a quality cut for
TRIDENT), and is reported as such. What remains is one band-independent
slope, the model's reach growing faster with energy than the simulated
areas, the same residual ARCA shows.

Usage
-----
    python examples/55_pone_trident_prediction.py
    python examples/55_pone_trident_prediction.py --with-regeneration
"""

import argparse
import importlib.util
import pathlib

import matplotlib.pyplot as plt
import numpy as np

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"
_DATA = _HERE.parent / "src" / "softpaws" / "data"

#: Half-width of the drawn band around each smoothed published curve [dex].
BAND_DEX = 0.12

#: Energy bin for the digitization smoothing [dex] and the running-mean width.
SMOOTH_BIN_DEX = 0.1
SMOOTH_WINDOW = 5

#: Energy above which the residuals are reported [log10 GeV], and the
#: decades at which the ratio is printed.
REPORT_LOG10_E = 4.0
REPORT_DECADES = (5.0, 6.0, 6.9)

#: P-ONE zenith bands [deg] in file order; 180 is the nadir.
PONE_BANDS = ((0, 30), (30, 60), (60, 90), (90, 120), (120, 150), (150, 180))

#: TRIDENT ``cos(theta)`` bands ``(lo, hi)`` and their file stems; -1 is the nadir.
TRIDENT_BANDS = (((-1.0, -0.2), "trident_-1_-0.2"), ((-0.2, 0.2), "trident_-0.2_0.2"),
                 ((0.2, 1.0), "trident_0.2_1.0"))

#: Style-file colours, one per band.
COLORS = ("#e7298a", "#1b9e77", "#d95f02", "#7570b3", "#66a61e", "#e6ab02")


def load_example(stem: str, name: str):
    """Import a sibling example module by file stem."""
    spec = importlib.util.spec_from_file_location(name, _HERE / stem)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--efficiency", type=float, default=None,
                        help="Flat normalization for both sites; defaults to example "
                             "45's ARCA230 value, 0.817 with rock below the sea floor.")
    parser.add_argument("--halo-weight", type=float, default=1.0,
                        help="Fraction of the reach-dilated halo counted (example 47).")
    parser.add_argument("--pone-radius-km", type=float, default=None,
                        help="Override the per-cluster footprint radius of P-ONE [km]; "
                             "example 35's value when omitted.")
    parser.add_argument("--with-regeneration", action="store_true",
                        help="Keep the neutral-current down-scattering ladder in the "
                             "Earth transmission. The default is pure absorption, which "
                             "is what both published simulations use.")
    parser.add_argument("--with-tau", action="store_true",
                        help="Add the nu_tau -> tau -> mu channel.")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the figures, '55a' through '55d'.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Digitized curves
# ---------------------------------------------------------------------------


def smooth_digitized(log10_e, log10_a):
    """Smooth a digitized step-function curve.

    Median of ``log10_a`` in :data:`SMOOTH_BIN_DEX` bins of ``log10_e``,
    then a :data:`SMOOTH_WINDOW`-bin running mean, evaluated at the bin
    centres.

    Returns
    -------
    log10_e_s, log10_a_s : np.ndarray
        Smoothed curve on the bin centres that hold at least one point.
    """
    lo = np.floor(log10_e.min() / SMOOTH_BIN_DEX) * SMOOTH_BIN_DEX
    edges = np.arange(lo, log10_e.max() + SMOOTH_BIN_DEX, SMOOTH_BIN_DEX)
    index = np.clip(np.digitize(log10_e, edges) - 1, 0, edges.size - 2)
    centers, medians = [], []
    for k in range(edges.size - 1):
        sel = index == k
        if sel.any():
            centers.append(0.5 * (edges[k] + edges[k + 1]))
            medians.append(np.median(log10_a[sel]))
    centers, medians = np.array(centers), np.array(medians)
    half = SMOOTH_WINDOW // 2
    padded = np.pad(medians, half, mode="edge")
    kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
    return centers, np.convolve(padded, kernel, mode="valid")


def _read_xy(path: pathlib.Path):
    raw = np.loadtxt(path, delimiter=",")
    x, y = raw[:, 0], raw[:, 1]
    good = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
    order = np.argsort(x[good])
    return x[good][order], y[good][order]


def pone_bands():
    """Digitized P-ONE effective areas per zenith band.

    Returns
    -------
    bands : list of tuple
        ``(cos_lo, cos_hi, log10_e, log10_aeff_cm2)`` per band, smoothed;
        ``cos_lo < cos_hi``.
    """
    out = []
    for z_lo, z_hi in PONE_BANDS:
        e_gev, a_m2 = _read_xy(_DATA / "pone" / f"pone_{z_lo}_{z_hi}.csv")
        log10_e, log10_a = smooth_digitized(np.log10(e_gev), np.log10(1.0e4 * a_m2))
        cos = sorted((np.cos(np.deg2rad(z_lo)), np.cos(np.deg2rad(z_hi))))
        out.append((cos[0], cos[1], log10_e, log10_a))
    return out


def pone_allsky():
    """Digitized P-ONE all-sky effective area, smoothed."""
    e_gev, a_m2 = _read_xy(_DATA / "pone" / "pone_allsky.csv")
    return smooth_digitized(np.log10(e_gev), np.log10(1.0e4 * a_m2))


def trident_bands():
    """Digitized TRIDENT effective areas per ``cos(theta)`` band, smoothed."""
    out = []
    for (cos_lo, cos_hi), stem in TRIDENT_BANDS:
        log10_e, a_m2 = _read_xy(_DATA / "trident" / f"{stem}.csv")
        log10_e, log10_a = smooth_digitized(log10_e, np.log10(1.0e4 * a_m2))
        out.append((cos_lo, cos_hi, log10_e, log10_a))
    return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def band_model(ex35, ex45, ex46, ex47, site, bands, flavours, halo_weight, efficiency,
               n_sub: int = 8):
    """Model effective area averaged over each ``cos(theta)`` band [cm^2].

    Example 47's ARCA machinery: the fitted water optics, uniform average
    in ``cos(theta)`` over ``n_sub`` sub-directions.

    Returns
    -------
    model : np.ndarray, shape (n_band, n_energy)
        On ``ex35.COMMON_LOG10_E``.
    """
    arca_style = [(hi, lo, None, None) for lo, hi, _, _ in bands]
    return ex47.arca_banded_model(ex35, ex45, ex46, site, arca_style,
                                  ex45.DEFAULT_MIN_MODULES, flavours, halo_weight,
                                  efficiency, True, n_sub=n_sub)


def allsky_model(ex35, ex45, ex46, ex47, site, flavours, halo_weight, efficiency,
                 n_sub: int = 24):
    """Model effective area averaged uniformly over the sphere [cm^2]."""
    edges = np.linspace(-1.0, 1.0, n_sub + 1)
    cos_theta = 0.5 * (edges[:-1] + edges[1:])
    return ex47.directional_model(ex35, ex45, ex46, site, cos_theta,
                                  ex45.DEFAULT_MIN_MODULES, flavours, halo_weight,
                                  efficiency, True).mean(axis=1)


def residuals(ex35, log10_e, log10_a, curve):
    """Published minus model [dex] at the smoothed points above the report energy."""
    mask = log10_e >= REPORT_LOG10_E
    table = np.log10(np.clip(curve, 1.0e-30, None))
    matched = np.interp(log10_e[mask], ex35.COMMON_LOG10_E, table)
    return log10_a[mask] - matched


def report(ex35, name, bands, model, labels) -> None:
    """Print the per-band level and scatter of published over model."""
    print(f"\n  {name}: published / model, E >= 10^{REPORT_LOG10_E:g} GeV")
    print(f"  {'band':>18} {'mean':>8} {'rms':>8} {'tilt/dec':>10}"
          + "".join(f" {'@10^' + str(d):>8}" for d in REPORT_DECADES))
    all_res = []
    for (lo, hi, log10_e, log10_a), curve, label in zip(bands, model, labels):
        res = residuals(ex35, log10_e, log10_a, curve)
        x = log10_e[log10_e >= REPORT_LOG10_E]
        tilt = np.polyfit(x, res, 1)[0] if x.size > 2 else np.nan
        all_res.append(res)
        table = np.log10(np.clip(curve, 1.0e-30, None))
        at = [10.0 ** (np.interp(d, log10_e, log10_a) - np.interp(d, ex35.COMMON_LOG10_E, table))
              if log10_e.min() <= d <= log10_e.max() else np.nan for d in REPORT_DECADES]
        print(f"  {label:>18} {10.0**res.mean():8.3f} {res.std():8.3f} {tilt:10.3f}"
              + "".join(f" {v:8.2f}" for v in at))
    all_res = np.concatenate(all_res)
    print(f"  {'all bands':>18} {10.0**all_res.mean():8.3f} {all_res.std():8.3f}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _save(fig, out_dir: pathlib.Path, stem: str) -> None:
    """Write one figure to ``out_dir`` as both PDF and PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {path.resolve()}")
    plt.close(fig)


def figure_bands(ex35, name, bands, model, labels, stem, out_dir, ylim) -> None:
    """Published bands (smoothed, with the digitization band) and model lines."""
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        for (lo, hi, log10_e, log10_a), curve, label, color in zip(bands, model, labels,
                                                                   COLORS):
            ax.fill_between(log10_e, 10.0 ** (log10_a - BAND_DEX),
                            10.0 ** (log10_a + BAND_DEX), color=color, alpha=0.22, lw=0)
            ax.plot(ex35.COMMON_LOG10_E, curve, color=color, lw=1.1, label=label)
        ax.plot([], [], color="0.5", lw=1.1, label="model (lines)")
        ax.fill_between([], [], [], color="0.5", alpha=0.3, lw=0,
                        label=rf"published $\pm{BAND_DEX:g}$ dex")
        ax.set_yscale("log")
        ax.set_xlim(3.0, 8.0)
        ax.set_ylim(*ylim)
        ax.set_xlabel(r"$\log_{10}(E_\nu\,/\,\mathrm{GeV})$")
        ax.set_ylabel(r"$A_{\rm eff}$ [cm$^2$]")
        ax.legend(fontsize=5.5, frameon=False, loc="upper left", ncol=2, title=name,
                  title_fontsize=6.5)
        _save(fig, out_dir, stem)


def figure_ratios(ex35, sets, out_dir) -> None:
    """Published over model per band, one panel per detector."""
    with plt.style.context(str(_STYLE)):
        fig, axes = plt.subplots(1, len(sets), figsize=(3.2 * len(sets), 3.0), sharey=True,
                                 gridspec_kw={"wspace": 0.06})
        for ax, (name, bands, model, labels) in zip(axes, sets):
            ax.axhspan(10.0**-BAND_DEX, 10.0**BAND_DEX, color="0.93", zorder=0)
            ax.axhline(1.0, color="0.6", lw=0.8, ls=":", zorder=1)
            for (lo, hi, log10_e, log10_a), curve, label, color in zip(bands, model, labels,
                                                                       COLORS):
                table = np.log10(np.clip(curve, 1.0e-30, None))
                matched = np.interp(log10_e, ex35.COMMON_LOG10_E, table)
                ax.plot(log10_e, 10.0 ** (log10_a - matched), color=color, lw=1.2,
                        label=label)
            ax.set_yscale("log")
            ax.set_xlim(3.0, 8.0)
            ax.set_ylim(0.1, 10.0)
            ax.set_xlabel(r"$\log_{10}(E_\nu\,/\,\mathrm{GeV})$")
            ax.legend(fontsize=5.5, frameon=False, loc="upper left", ncol=2, title=name,
                      title_fontsize=6.5)
        axes[0].set_ylabel("published / model")
        _save(fig, out_dir, "55c_pone_trident_ratio")


def figure_allsky(ex35, published, model, out_dir) -> None:
    """P-ONE's all-sky curve against the sphere-averaged model."""
    log10_e, log10_a = published
    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        ax.fill_between(log10_e, 10.0 ** (log10_a - BAND_DEX), 10.0 ** (log10_a + BAND_DEX),
                        color=COLORS[0], alpha=0.22, lw=0,
                        label=rf"P-ONE all-sky $\pm{BAND_DEX:g}$ dex")
        ax.plot(ex35.COMMON_LOG10_E, model, color=COLORS[0], lw=1.2, label="model")
        ax.set_yscale("log")
        ax.set_xlim(3.0, 8.0)
        ax.set_ylim(1.0e3, 1.0e8)
        ax.set_xlabel(r"$\log_{10}(E_\nu\,/\,\mathrm{GeV})$")
        ax.set_ylabel(r"$A_{\rm eff}$ [cm$^2$]")
        ax.legend(fontsize=6, frameon=False, loc="upper left")
        _save(fig, out_dir, "55d_pone_allsky")


def main() -> None:
    args = parse_args()
    print("Loading examples 35, 45, 46 and 47 ...")
    ex35 = load_example("35_point_source_effective_area.py", "_example_35")
    ex45 = load_example("45_first_principles_reach.py", "_example_45")
    ex46 = load_example("46_declination_resolved_reach.py", "_example_46")
    ex47 = load_example("47_point_source_derived_reach.py", "_example_47")
    if not args.with_regeneration:
        _pure = ex46.regenerated_transmission

        def _absorb_only(*a, **k):
            k["n_levels"] = 1
            return _pure(*a, **k)

        ex46.regenerated_transmission = _absorb_only
        print("  Earth transmission: pure absorption, no NC regeneration")
    sites = {s.name: s for s in ex35.build_sites()}
    if args.pone_radius_km is not None:
        from dataclasses import replace
        sites["P-ONE"] = replace(sites["P-ONE"], radius_km=args.pone_radius_km)
    efficiency = (ex47.FITTED_NORMALIZATION["ARCA230"] if args.efficiency is None
                  else args.efficiency)
    flavours = ("mu", "tau") if args.with_tau else ("mu",)
    print(f"  water configuration: Lambda {ex47.FITTED_ATTENUATION_M['ARCA230']:g} m, "
          f"normalization {efficiency:g}, N = {ex45.DEFAULT_MIN_MODULES:g} modules, "
          f"halo weight {args.halo_weight:g}, channels "
          f"{', '.join('nu_' + f for f in flavours)}")
    for name in ("P-ONE", "TRIDENT"):
        s = sites[name]
        print(f"  {name}: {s.n_blocks} x (r {s.radius_km:g} km, h {s.height_km:g} km) "
              f"at {s.depth_km:.2f} km, latitude {s.latitude_deg:g} deg")

    print("Loading the digitized curves ...")
    pone = pone_bands()
    pone_sky = pone_allsky()
    trident = trident_bands()
    pone_labels = [rf"$[{lo:g},\,{hi:g}]^\circ$" for lo, hi in PONE_BANDS]
    trident_labels = [rf"$[{lo:g},\,{hi:g}]$" for (lo, hi), _ in TRIDENT_BANDS]

    print("Building the P-ONE bands ...")
    pone_model = band_model(ex35, ex45, ex46, ex47, sites["P-ONE"], pone, flavours,
                            args.halo_weight, efficiency)
    print("Building the P-ONE all-sky curve ...")
    pone_sky_model = allsky_model(ex35, ex45, ex46, ex47, sites["P-ONE"], flavours,
                                  args.halo_weight, efficiency)
    print("Building the TRIDENT bands ...")
    trident_model = band_model(ex35, ex45, ex46, ex47, sites["TRIDENT"], trident, flavours,
                               args.halo_weight, efficiency)

    report(ex35, "P-ONE (zenith bands, trigger level)", pone, pone_model, pone_labels)
    report(ex35, "P-ONE (all-sky)", [(-1.0, 1.0) + pone_sky], [pone_sky_model], ["all-sky"])
    report(ex35, "TRIDENT (cos theta bands, 6 deg cut)", trident, trident_model,
           trident_labels)

    print()
    figure_bands(ex35, "P-ONE, zenith", pone, pone_model, pone_labels, "55a_pone_bands",
                 args.out_dir, (1.0e3, 1.0e8))
    figure_bands(ex35, r"TRIDENT, $\cos\theta$", trident, trident_model, trident_labels,
                 "55b_trident_bands", args.out_dir, (1.0e4, 1.0e9))
    figure_ratios(ex35, [("P-ONE", pone, pone_model, pone_labels),
                         ("TRIDENT", trident, trident_model, trident_labels)], args.out_dir)
    figure_allsky(ex35, pone_sky, pone_sky_model, args.out_dir)


if __name__ == "__main__":
    main()
