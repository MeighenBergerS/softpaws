"""Example 76 -- the forward model held against the DR2 events, nothing fitted.

Every comparison so far held the model against a published response, and
example 51 fitted it to events with profiled nuisances. This example does the
simpler and harder thing: it predicts the IC86 event sample with every
parameter pinned externally and reports where the prediction stands. The
model side is example 51's machinery unchanged -- the fitted-configuration
responses of example 45 (``nu_mu`` and ``nu_tau -> tau -> mu`` channels,
banded in the smearing table's declination bins) folded through the released
IC86 smearing marginal -- but nothing is profiled: the conventional and
prompt atmospheric fluxes are example 22's MCEq tables at normalization one,
and the astrophysical flux is IceCube's 9.5-year tracks measurement
(``1.44e-18`` [GeV^-1 cm^-2 s^-1 sr^-1] per flavour at 100 TeV, index 2.37)
at a 1:1:1 composition, with the tau channel folded through the decay-shifted
smearing. The data side is the binned 10.7-year IC86 upgoing sample.

The output is the predicted reconstructed-energy spectrum and declination
distribution against the events, with data/model ratios printed per decade.
The shaded band on the total is external-input uncertainty only: the +-25%
hadronic spread on each atmospheric normalization and the tracks
measurement's own normalization and index errors, in quadrature. The model
carries a known boundary: below a reconstructed 10^4.25 GeV the turn-on and
proxy region under-predicts progressively, which is why example 51 fits
above that line; this example draws the line and quotes the ratios on both
sides of it. Everything above the line is a prediction with zero fitted
parameters.

Usage
-----
    python examples/76_dr2_event_benchmark.py
    python examples/76_dr2_event_benchmark.py --rebuild-cache
"""

import argparse
import importlib.util
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from softpaws.data.loader import compute_livetime_s, load_uptime

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"


def _load_example(stem: str, name: str):
    """Import a sibling example module by file stem."""
    spec = importlib.util.spec_from_file_location(name, _HERE / stem)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: Example 51, for the smearing marginal, the banded responses, the
#: atmospheric fluxes, the event binning, and the tracks-fit constants.
_EX51 = _load_example("51_dr2_flavor_fit.py", "_example_51")

#: The externally pinned astrophysical flux: IceCube's 9.5-year tracks fit,
#: per flavour at the 100 TeV pivot (arXiv:2111.10299), applied to the
#: ``nu_mu`` and ``nu_tau`` channels alike (1:1:1).
ASTRO_PHI, ASTRO_PHI_ERR = _EX51.TRACKS_PHI_MU
ASTRO_GAMMA, ASTRO_GAMMA_ERR = _EX51.TRACKS_GAMMA

#: Hadronic-model spread carried as the atmospheric normalization error.
ATM_ERR = 0.25

#: Reconstructed energy [log10 GeV] above which the astrophysical component
#: is expected to compete with the atmosphere; the declination figure's
#: second panel cuts here.
HIGH_RECO = 5.0


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=pathlib.Path,
                        default=_EX51._DEFAULT_DATA_DIR,
                        help="Root of the DR2 release.")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Rebuild example 51's cached responses.")
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for figures 76a and 76b.")
    return parser.parse_args()


def load_inputs(data_dir: pathlib.Path, rebuild: bool):
    """Example 51's cached smearing marginal and banded responses.

    The heavy response build is only triggered when the cache is missing or
    a rebuild is requested; examples 35, 45 and 46 load lazily in that case.
    """
    if _EX51._CACHE.exists() and not rebuild:
        return _EX51.fit_inputs(None, None, None, data_dir, False)
    ex35 = _load_example("35_point_source_effective_area.py", "_example_35")
    ex45 = _load_example("45_first_principles_reach.py", "_example_45")
    ex46 = _load_example("46_declination_resolved_reach.py", "_example_46")
    return _EX51.fit_inputs(ex35, ex45, ex46, data_dir, rebuild)


def true_counts(response, flux, enu_edges, d_omega, livetime_s):
    """Expected counts per (true-energy bin, declination band).

    Mirrors :meth:`RecoLikelihood._true_counts` for a standalone response and
    flux grid, both on ``LOG10_E_GRID`` x bands.

    Parameters
    ----------
    response : np.ndarray
        Channel response [cm^2].
    flux : np.ndarray
        Differential flux [GeV^-1 cm^-2 s^-1 sr^-1].
    enu_edges : np.ndarray
        True-energy bin edges of the smearing table [log10 GeV].
    d_omega : np.ndarray
        Solid angle per declination band [sr].
    livetime_s : float
        Summed IC86 livetime [s].
    """
    energy = 10.0**_EX51.LOG10_E_GRID
    integrand = response * flux
    counts = np.zeros((enu_edges.size - 1, d_omega.size))
    for i in range(enu_edges.size - 1):
        sel = ((_EX51.LOG10_E_GRID >= enu_edges[i])
               & (_EX51.LOG10_E_GRID <= enu_edges[i + 1]))
        if sel.sum() < 2:
            continue
        counts[i] = np.trapezoid(integrand[sel], energy[sel], axis=0)
    return livetime_s * counts * d_omega[None, :]


def astro_flux(gamma):
    """Per-flavour tracks-fit power law on the fine grid [GeV^-1 cm^-2 s^-1 sr^-1]."""
    energy = 10.0**_EX51.LOG10_E_GRID
    return ASTRO_PHI * (energy[:, None] / 1.0e5) ** (-gamma)


def predict(responses, atmos, enu_edges, dec_edges, marginal, livetime_s):
    """Reconstructed-space components on the full reco grid, per band.

    Returns
    -------
    components : dict of str -> np.ndarray
        ``"conv"``, ``"prompt"``, ``"astro_mu"``, ``"astro_tau"`` on
        (reco bins, declination bands).
    band : np.ndarray
        One-sigma external-input uncertainty on the summed model, same shape.
    """
    d_omega = 2.0 * np.pi * np.diff(np.sin(np.deg2rad(dec_edges)))
    marginal_tau = _EX51.RecoLikelihood._shifted_marginal(enu_edges, marginal)

    def fold(true, tau=False):
        return np.einsum("ij,ijk->kj", true, marginal_tau if tau else marginal)

    ones = np.ones((1, d_omega.size))
    flux = astro_flux(ASTRO_GAMMA) * ones
    components = {
        "conv": fold(true_counts(responses["mu"], atmos["conv"], enu_edges,
                                 d_omega, livetime_s)),
        "prompt": fold(true_counts(responses["mu"], atmos["prompt"], enu_edges,
                                   d_omega, livetime_s)),
        "astro_mu": fold(true_counts(responses["mu"], flux, enu_edges,
                                     d_omega, livetime_s)),
        "astro_tau": fold(true_counts(responses["tau"], flux, enu_edges,
                                      d_omega, livetime_s), tau=True),
    }

    astro = components["astro_mu"] + components["astro_tau"]
    spread = []
    for gamma in (ASTRO_GAMMA - ASTRO_GAMMA_ERR, ASTRO_GAMMA + ASTRO_GAMMA_ERR):
        flux_g = astro_flux(gamma) * ones
        varied = (fold(true_counts(responses["mu"], flux_g, enu_edges,
                                   d_omega, livetime_s))
                  + fold(true_counts(responses["tau"], flux_g, enu_edges,
                                     d_omega, livetime_s), tau=True))
        spread.append(varied - astro)
    gamma_sigma = 0.5 * (np.abs(spread[0]) + np.abs(spread[1]))
    band = np.sqrt((ATM_ERR * components["conv"]) ** 2
                   + (ATM_ERR * components["prompt"]) ** 2
                   + (ASTRO_PHI_ERR / ASTRO_PHI * astro) ** 2
                   + gamma_sigma**2)
    return components, band


def report(components, band, data):
    """Print the data/model comparison; return the summary dict."""
    centers = 0.5 * (_EX51.RECO_EDGES[:-1] + _EX51.RECO_EDGES[1:])
    window = ((_EX51.RECO_EDGES[:-1] >= _EX51.FIT_RECO[0] - 1.0e-9)
              & (_EX51.RECO_EDGES[1:] <= _EX51.FIT_RECO[1] + 1.0e-9))
    total = sum(components.values())

    print("\nComponent totals inside the prediction window "
          f"(reco 10^{_EX51.FIT_RECO[0]} to 10^{_EX51.FIT_RECO[1]} GeV):")
    for name, grid in components.items():
        print(f"  {name:<10s} {grid[window].sum():10.1f}")
    model_w = total[window].sum()
    band_w = np.sqrt((band[window] ** 2).sum())
    data_w = data[window].sum()
    print(f"  model      {model_w:10.1f} +- {band_w:.1f} (external inputs)")
    print(f"  data       {data_w:10.0f}")
    print(f"  data/model {data_w / model_w:10.3f}")

    astro = components["astro_mu"] + components["astro_tau"]
    tau_share = components["astro_tau"][window].sum() / astro[window].sum()
    print(f"  tau -> mu share of astrophysical tracks: {tau_share:.3f}")

    print("\nData/model per half-decade (summed over bands):")
    model_e = total.sum(axis=1)
    data_e = data.sum(axis=1)
    for lo in np.arange(3.0, 8.0, 0.5):
        sel = (centers > lo) & (centers < lo + 0.5)
        if model_e[sel].sum() <= 0.0:
            continue
        if lo + 0.5 <= _EX51.FIT_RECO[0]:
            marker = "   (below the window)"
        elif lo < _EX51.FIT_RECO[0]:
            marker = "   (straddles the window edge)"
        else:
            marker = ""
        print(f"  10^{lo:.1f} - 10^{lo + 0.5:.1f}: "
              f"{data_e[sel].sum() / model_e[sel].sum():7.2f}{marker}")

    occupied = window[:, None] & (total > 0.0)
    mu = np.clip(total[occupied], 1.0e-12, None)
    observed = data[occupied]
    with np.errstate(divide="ignore", invalid="ignore"):
        deviance = 2.0 * (mu - observed + np.where(
            observed > 0.0, observed * np.log(observed / mu), 0.0))
    scaled = deviance / (1.0 + _EX51.MODEL_SYS**2 * mu)
    print(f"\nWindow deviance over {occupied.sum()} (reco, band) bins, "
          "zero fitted parameters:")
    print(f"  Poisson {deviance.sum():8.1f}   scaled (10% shape systematic) "
          f"{scaled.sum():8.1f}")
    return {"window": window, "centers": centers, "total": total,
            "tau_share": tau_share}


def poisson_errors(counts):
    """Garwood central 68% Poisson intervals, as (lower, upper) error bars.

    The square-root error bar of a one-count bin reaches zero and draws a
    line to the floor of a log axis; the exact interval does not.
    """
    with np.errstate(invalid="ignore"):
        lower = counts - chi2.ppf(0.159, 2.0 * counts) / 2.0
    upper = chi2.ppf(0.841, 2.0 * (counts + 1.0)) / 2.0 - counts
    return np.where(counts > 0.0, lower, 0.0), upper


def _save(fig, out_dir: pathlib.Path, stem: str) -> None:
    """Write one figure to ``out_dir`` as both PDF and PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {path.resolve()}")
    plt.close(fig)


def figure_spectrum(components, band, data, out_dir) -> None:
    """Figure 76a: reconstructed-energy spectrum and the data/model ratio."""
    centers = 0.5 * (_EX51.RECO_EDGES[:-1] + _EX51.RECO_EDGES[1:])
    total = sum(components.values())
    model_e = total.sum(axis=1)
    band_e = np.sqrt((band**2).sum(axis=1))
    data_e = data.sum(axis=1)
    atmos_e = (components["conv"] + components["prompt"]).sum(axis=1)
    with plt.style.context(str(_STYLE)):
        fig, (ax, axr) = plt.subplots(
            2, 1, sharex=True, figsize=(3.4, 4.6),
            gridspec_kw={"height_ratios": (2.6, 1.0), "hspace": 0.08})
        ax.plot(centers, atmos_e, color="0.45", ls="-", lw=0.9,
                label="Atmospheric")
        ax.plot(centers, components["astro_mu"].sum(axis=1), color="C0",
                lw=0.9, label=r"Astro $\nu_\mu$")
        ax.plot(centers, components["astro_tau"].sum(axis=1), color="C1",
                lw=0.9, label=r"Astro $\nu_\tau \to \tau \to \mu$")
        ax.plot(centers, model_e, color="k", lw=1.4,
                label="Model, nothing fitted")
        ax.fill_between(centers, model_e - band_e, model_e + band_e,
                        color="k", alpha=0.15, lw=0)
        sel = data_e > 0.0
        err_lo, err_hi = poisson_errors(data_e[sel])
        ax.errorbar(centers[sel], data_e[sel], yerr=(err_lo, err_hi),
                    fmt="o", color="k", ms=2.5, lw=0.8, label="IC86 events")
        ax.set_yscale("log")
        ax.set_ylim(0.3, 3.0e5)
        ax.set_ylabel("Events per 0.25 dex")
        ax.set_box_aspect(1)
        ax.legend(loc="upper right")

        shown = sel & (model_e > 0.0)
        axr.errorbar(centers[shown], data_e[shown] / model_e[shown],
                     yerr=(err_lo[shown[sel]] / model_e[shown],
                           err_hi[shown[sel]] / model_e[shown]),
                     fmt="o", color="k", ms=2.5, lw=0.8)
        axr.fill_between(centers, 1.0 - band_e / np.clip(model_e, 1e-12, None),
                         1.0 + band_e / np.clip(model_e, 1e-12, None),
                         color="k", alpha=0.15, lw=0)
        axr.axhline(1.0, color="0.4", lw=0.6)
        axr.set_yscale("log")
        axr.set_ylim(0.3, 20.0)
        axr.set_xlim(3.0, 8.0)
        axr.set_xlabel(r"$\log_{10}(E_{\mathrm{reco}} / \mathrm{GeV})$")
        axr.set_ylabel("Data / model")
        axr.set_box_aspect(1.0 / 2.6)
        _save(fig, out_dir, "76a_reco_spectrum")


def figure_declination(components, band, data, dec_edges, out_dir) -> None:
    """Figures 76b/76c: declination over the window and above the high cut."""
    sin_edges = np.sin(np.deg2rad(dec_edges))
    sin_centers = 0.5 * (sin_edges[:-1] + sin_edges[1:])
    total = sum(components.values())
    atmos = components["conv"] + components["prompt"]
    window = ((_EX51.RECO_EDGES[:-1] >= _EX51.FIT_RECO[0] - 1.0e-9)
              & (_EX51.RECO_EDGES[1:] <= _EX51.FIT_RECO[1] + 1.0e-9))
    high = _EX51.RECO_EDGES[:-1] >= HIGH_RECO - 1.0e-9
    panels = (
        ("76b_declination", window, r"$E_{\mathrm{reco}} > 10^{4.25}$ GeV"),
        ("76c_declination_high", window & high,
         rf"$E_{{\mathrm{{reco}}}} > 10^{{{HIGH_RECO:.0f}}}$ GeV"),
    )
    with plt.style.context(str(_STYLE)):
        for stem, sel, title in panels:
            fig, ax = plt.subplots(figsize=(3.0, 3.0))
            model_d = total[sel].sum(axis=0)
            band_d = np.sqrt((band[sel] ** 2).sum(axis=0))
            data_d = data[sel].sum(axis=0)
            atmos_d = atmos[sel].sum(axis=0)
            ax.stairs(model_d, sin_edges, color="k", lw=1.2, label="Model")
            ax.stairs(np.clip(model_d - band_d, 0.0, None), sin_edges,
                      baseline=model_d + band_d, fill=True, color="k",
                      alpha=0.15, lw=0)
            ax.stairs(atmos_d, sin_edges, color="0.45", lw=0.8, ls="--",
                      label="Atmospheric")
            occupied = data_d > 0.0
            err_lo, err_hi = poisson_errors(data_d[occupied])
            ax.errorbar(sin_centers[occupied], data_d[occupied],
                        yerr=(err_lo, err_hi), fmt="o",
                        color="k", ms=2.5, lw=0.8, label="IC86 events")
            ax.set_title(title)
            ax.set_xlabel(r"$\sin\delta$")
            ax.set_ylabel("Events per band")
            ax.set_yscale("log")
            ax.set_box_aspect(1)
            ax.legend(loc="lower left")
            _save(fig, out_dir, stem)


def main() -> None:
    args = parse_args()
    print("Loading example 51's inputs ...")
    enu_edges, dec_edges, marginal, responses = load_inputs(
        args.data_dir, args.rebuild_cache)
    atmos = _EX51.atmospheric_fluxes(dec_edges)

    livetime_s = sum(
        compute_livetime_s(load_uptime(args.data_dir / "uptime" / f"{s}_exp.csv"))
        for s in _EX51.IC86_SEASONS)
    print(f"  IC86 livetime: {livetime_s / (365.25 * 86400.0):.2f} yr")
    data = _EX51.binned_events(args.data_dir, dec_edges)

    print("Predicting with pinned inputs: MCEq atmosphere x 1.0, tracks flux "
          f"{ASTRO_PHI * 1e18:.2f}e-18 at gamma {ASTRO_GAMMA} per flavour, 1:1:1")
    components, band = predict(responses, atmos, enu_edges, dec_edges, marginal,
                               livetime_s)
    report(components, band, data)

    figure_spectrum(components, band, data, args.out_dir)
    figure_declination(components, band, data, dec_edges, args.out_dir)


if __name__ == "__main__":
    main()
