"""Example 22 -- MCEq atmospheric background, added to example 21.

Every comparison up to and including ``21_full_icecube_comparison.py`` fits a
bare astrophysical template above 1 PeV, because below that the upgoing
IceTracks-DR2 sample is dominated by atmospheric neutrinos that the model does
not predict. This example computes that background from first principles with
**MCEq** (https://github.com/mceq-project/MCEq), which solves the coupled
cascade equations for the air shower, and folds it through both forward models,
so the comparison can be extended down to 10 TeV.

The calculation has two stages:

1. **Tabulate.** MCEq is run once per zenith angle on a South Pole atmosphere
   and the resulting ``nu_mu + nu_mu_bar`` fluxes (conventional and prompt,
   separately) are cached to an ``.npz`` table. This is the slow step -- of
   order ten minutes, dominated by the near-horizontal angles, whose curved
   atmosphere takes the longest to integrate -- so it is skipped whenever the
   table is already on disk.
2. **Integrate.** The table is interpolated in ``(E, dec)`` and pushed through
   the two response paths: the soft-volume model via the new
   :meth:`~softpaws.response.soft_volume.SoftVolumeResponse.
   expected_counts_from_flux`, and the published IRFs via the new
   :func:`~softpaws.comparison.rates.irf_expected_counts_directional`. Neither
   the flux's spectral shape nor its zenith dependence is a power law, which is
   what those two entry points add over the ones example 21 uses.

Fitting
-------
The fit is in two stages, which keeps every free parameter in the region that
actually constrains it.

First the **atmospheric flux normalization** ``k_atm`` is measured on the IRF
path over 10-100 TeV (:func:`calibrate_atmospheric_normalization`), where the
sample is background dominated and holds ~2000 events. It has to be the IRF
path: the published response is the one containing IceCube's selection
efficiency, so a mismatch there is attributable to the flux model rather than to
the detector. Because ``k_atm`` then corrects the *flux*, it carries over to the
soft-volume path unchanged.

Second the **astrophysical normalization** is fitted above 100 TeV with the
background frozen at ``k_atm`` times its prediction
(:func:`~softpaws.comparison.rates.fit_scale_factor_with_background`). Floating
the background there instead would leave two normalizations against three
populated bins holding seventeen events, and it duly returns nonsense.

Why the fit does not reach further down
---------------------------------------
An earlier version of this example fitted both normalizations jointly from
10 TeV. That is not defensible, and the reason is worth recording. The
efficiency of the IceCube selection is strongly energy dependent -- example 20
measures it climbing from ~0 near 100 GeV to ~1 by 1 PeV -- so a *single*
efficiency factor, whether fitted or implied, is a constant chasing a turn-on
curve. It returns the event-weighted mean of that curve over the fit window,
which is why moving the threshold from 10 TeV to 300 TeV walked the fitted
``phi0`` between 0.72 and 1.14 with the atmospheric normalization trading
against it.

Nor can the turn-on simply be fitted as a nuisance. It is not physics: it is
IceCube's trigger, filter, quality cuts, and atmospheric-muon rejection for this
particular selection, and example 20's range convention has already divided out
the part that *is* physics (muon range and geometric target volume). Fitting it
would measure their cuts through our model, with every transport and flux error
free to hide in it; importing it from example 20 is circular, since it is
defined there as the ratio that makes the two effective areas agree. The
soft-volume model is therefore only predictive where the selection is fully
efficient, and 100 TeV is chosen as a compromise: high enough that the turn-on
is nearly complete, low enough to retain more than the two events above 1 PeV.
Whatever residual inefficiency remains at 100-300 TeV shows up here as the
soft-volume background over-predicting the data, and is reported as a ceiling on
the efficiency in that window rather than fitted away.

Geometry
--------
Atmospheric neutrinos arriving upgoing at the South Pole are produced in the
atmosphere on the opposite side of the Earth, so a source declination ``dec``
(arrival zenith ``90 deg + dec``, the convention of
:mod:`softpaws.transport.attenuation`) corresponds to an MCEq production zenith
``theta = 90 deg - dec``: vertically upgoing tracks come from showers that were
vertical overhead on the far side, and horizontal ones from horizontal showers.
Their propagation through the Earth is then the same PREM absorption
(:func:`~softpaws.transport.attenuation.prem_column`) applied to the
astrophysical flux in example 14 -- which is why the background dies off above
a few hundred TeV rather than following the atmospheric spectrum forever.

Caveats
-------
* The soft-volume path models no detection efficiency, so its background, like
  its signal, is a geometric ceiling; the IRF path's background is the realistic
  one.
* The DR2 release provides one instrument response per detector configuration,
  not per season, and ``load_irfs`` maps every ``IC86_*`` label onto the single
  IC86 response. That is the correct response for the summed IC86 livetime used
  here; the earlier configurations (IC40, IC59, IC79) are excluded along with
  their events, since the soft-volume model has one fixed detector radius and
  cannot represent a partial detector.
* Both paths evaluate the atmospheric flux in the energy variable their own
  model uses -- observed muon energy for the soft-volume path (Eq. 2.23),
  true neutrino energy plus smearing for the IRF path -- exactly as they treat
  the astrophysical flux. See the energy-smearing note in example 21.
* Atmospheric *muons* are not modelled and do not need to be: they cannot cross
  the Earth, and the sample here is upgoing only.

Usage
-----
    python examples/22_atmospheric_background_mceq.py
    python examples/22_atmospheric_background_mceq.py --recompute-table
    python examples/22_atmospheric_background_mceq.py --data-dir /path/to/dataverse_files

Requires the optional MCEq dependency (``pip install -e ".[atm]"``), but only
when the flux table has to be built.
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.comparison.rates import (
    fit_component_scales,
    fit_scale_factor_with_background,
    irf_expected_counts,
    irf_expected_counts_directional,
    observed_counts,
)
from softpaws.data.icecube import (
    IC86_SEASONS,
    load_events,
    total_livetime_s,
)
from softpaws.data.loader import load_irfs
from softpaws.fluxes import (
    ATMOSPHERE,
    INTERACTION_MODEL,
    PRIMARY_MODEL,
    REFERENCE_SPL,
    AtmosphericFlux,
    build_mceq_table,
    load_mceq_table,
)
from softpaws.response.soft_volume import (
    SoftVolumeResponse,
    power_law_flux,
    tau_induced_expected_counts_attenuated,
)
from softpaws.transport.cross_section import bgr18_cross_section

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parents[1] / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_DATA_DIR = _HERE.parents[1] / "src" / "softpaws" / "data" / "dataverse_files"
_DEFAULT_OUT_DIR = _HERE / "output"
_DEFAULT_TABLE = _DEFAULT_OUT_DIR / "22_mceq_atmospheric_flux.npz"


RADIUS_KM = 0.62  # IceCube-like instrumented sphere
PHI0 = REFERENCE_SPL.phi0  # [1e-18 GeV^-1 cm^-2 s^-1 sr^-1] at 100 TeV
GAMMA = REFERENCE_SPL.gamma
LOG10_E_MIN_FIT = 5.0  # 100 TeV; the signal window
LOG10_E_MIN_BKG_CAL = 4.0  # 10 TeV; lower edge of the background calibration window
LOG10_E_EDGES = np.arange(3.0, 8.01, 0.5)
DEC_MIN, DEC_MAX = 0.0, 90.0  # upgoing hemisphere

# MCEq settings. SIBYLL-2.3d and H3a are the standard conventional/prompt
# atmospheric-neutrino baseline; the atmosphere is the South Pole profile MCEq
# ships for IceCube, in January (austral summer, the thinner atmosphere).
N_DEC_TABLE = 13  # production zeniths, evenly spaced in declination over the band


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=_DEFAULT_DATA_DIR,
        help="Root of the DR2 data directory.",
    )
    parser.add_argument(
        "--table",
        type=pathlib.Path,
        default=_DEFAULT_TABLE,
        help="Cached MCEq atmospheric-flux table.",
    )
    parser.add_argument(
        "--recompute-table",
        action="store_true",
        help="Rebuild the MCEq table even if it already exists.",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Use the conventional atmospheric component only, dropping charm decay.",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=_DEFAULT_OUT_DIR / "22_atmospheric_background_mceq.pdf",
        help="Output file for the figure.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Stage 1: tabulate the atmospheric flux per zenith angle with MCEq.
# ---------------------------------------------------------------------------


def build_flux_table(path: pathlib.Path, n_dec: int = N_DEC_TABLE) -> dict[str, np.ndarray]:
    """Run MCEq once per declination; see :func:`softpaws.fluxes.build_mceq_table`."""
    print(f"Building MCEq flux table ({n_dec} zenith angles; this takes minutes) ...")
    return build_mceq_table(path, (DEC_MIN, DEC_MAX), n_dec, INTERACTION_MODEL, PRIMARY_MODEL,
                            ATMOSPHERE)


def load_flux_table(path: pathlib.Path, recompute: bool = False) -> dict[str, np.ndarray]:
    """Cached MCEq table, built if missing; see :func:`softpaws.fluxes.load_mceq_table`."""
    if recompute or not path.exists():
        return build_flux_table(path)
    table = load_mceq_table(path)
    print(f"Loaded MCEq flux table: {path.resolve()} ({table['dec_deg'].size} zenith angles)")
    return table


# ---------------------------------------------------------------------------
# Stage 2: integrate the table into background counts (example 21's machinery).
# ---------------------------------------------------------------------------


def load_ic86_observed_and_livetime(data_dir: pathlib.Path) -> tuple[np.ndarray, float]:
    """Binned IC86 upgoing counts and the IC86 livetime, from the library loaders."""
    events = load_events(data_dir, IC86_SEASONS, within_uptime=True)
    counts = observed_counts(events, LOG10_E_EDGES, DEC_MIN, DEC_MAX)
    return counts, total_livetime_s(data_dir, IC86_SEASONS)


def calibrate_atmospheric_normalization(
    observed: np.ndarray,
    astro: np.ndarray,
    atmospheric: np.ndarray,
) -> float:
    """Atmospheric flux normalization measured below the signal window.

    The MCEq prediction is scaled to the data over
    ``[LOG10_E_MIN_BKG_CAL, LOG10_E_MIN_FIT)``, where the sample is
    background dominated and carries ~2000 events, rather than being left to
    float in the signal window, where only a handful of events constrain it.
    Both components are floated here so the ~10% astrophysical contamination of
    the calibration window does not bias the result.

    This must be run on the **IRF** templates: it is the published response that
    contains IceCube's selection efficiency, so a discrepancy there is
    attributable to the atmospheric flux model rather than to the detector. The
    resulting factor is then a property of the flux, and can be carried over to
    the soft-volume path.

    Parameters
    ----------
    observed : np.ndarray, shape (n_bins,)
        Observed counts per bin.
    astro, atmospheric : np.ndarray, shape (n_bins,)
        IRF-path astrophysical and atmospheric templates at their reference
        normalizations.

    Returns
    -------
    k_atm : float
        Multiplicative correction to the MCEq atmospheric prediction.
    """
    centers = 0.5 * (LOG10_E_EDGES[:-1] + LOG10_E_EDGES[1:])
    in_window = (centers >= LOG10_E_MIN_BKG_CAL) & (centers < LOG10_E_MIN_FIT)
    index = np.nonzero(in_window)[0]
    edges = LOG10_E_EDGES[index[0] : index[-1] + 2]

    _, k_atm = fit_component_scales(
        observed[in_window],
        [astro[in_window], atmospheric[in_window]],
        edges,
        edges[0],
    )
    return float(k_atm)


def soft_volume_signal(response: SoftVolumeResponse, livetime_s: float) -> np.ndarray:
    """Astrophysical ``numu + nu_tau`` soft-volume template of example 21."""
    numu = response.expected_counts_attenuated(
        LOG10_E_EDGES, PHI0, GAMMA, livetime_s, DEC_MIN, DEC_MAX,
    )
    tau = tau_induced_expected_counts_attenuated(
        response, LOG10_E_EDGES, PHI0, GAMMA, livetime_s, DEC_MIN, DEC_MAX,
    )
    return numu + tau


def make_figure(
    observed: np.ndarray,
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: pathlib.Path,
) -> None:
    """Observed counts against the fitted signal plus atmospheric background.

    The observed counts carry Poisson statistical errors ``sqrt(N)``. Empty bins
    are left out, having neither a value nor an error a logarithmic axis can
    show.

    Parameters
    ----------
    observed : np.ndarray, shape (n_bins,)
        Observed counts per bin.
    predictions : dict
        Maps a label to ``(fitted signal, fitted background)`` counts per bin.
    out_path : pathlib.Path
        Output file; written with both ``.pdf`` and ``.png`` suffixes.
    """
    centers = 0.5 * (LOG10_E_EDGES[:-1] + LOG10_E_EDGES[1:])
    filled = observed > 0
    styles = {"soft": ("-", "C3"), "IRF": ("--", "C1")}

    with plt.style.context(str(_STYLE)):
        fig, ax = plt.subplots(figsize=(3.4, 3.2))

        ax.errorbar(
            centers[filled],
            observed[filled],
            yerr=np.sqrt(observed[filled]),
            fmt="o",
            ms=2.5,
            lw=0.8,
            capsize=1.5,
            color="k",
            label="observed (IC86)",
            zorder=5,
        )
        for name, (signal, background) in predictions.items():
            ls, color = styles[name]
            ax.stairs(
                signal + background, LOG10_E_EDGES,
                label=f"{name}: astro + atm", ls=ls, color=color,
            )
            ax.stairs(
                background, LOG10_E_EDGES,
                label=f"{name}: atm alone", ls=":", color=color, alpha=0.7,
            )
        ax.axvline(LOG10_E_MIN_FIT, color="0.7", lw=0.6, zorder=0)
        ax.set_yscale("log")
        ax.set_ylim(1.0e-3, 5.0 * max(observed.max(), 1.0))
        ax.set_xlabel(r"$\log_{10}(E\,/\,\mathrm{GeV})$")
        ax.set_ylabel("tracks / bin")
        ax.legend(fontsize=6)

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png"):
            path = out_path.with_suffix(suffix)
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {path.resolve()}")
        plt.close(fig)


def main() -> None:
    args = parse_args()

    table = load_flux_table(args.table, args.recompute_table)
    atm_flux = AtmosphericFlux(table, include_prompt=not args.no_prompt)

    print(f"Loading IC86 events from: {args.data_dir}")
    observed, livetime_s = load_ic86_observed_and_livetime(args.data_dir)
    print(f"  IC86 upgoing events: {observed.sum():,.0f}")
    print(f"  IC86 combined livetime: {livetime_s / 86400.0:.1f} d")

    print("Loading IC86 IRFs ...")
    irfs = load_irfs(args.data_dir / "irfs", "IC86_I")

    def astro_flux(energy_gev):
        return power_law_flux(energy_gev, PHI0, GAMMA)

    print("Computing IRF signal and atmospheric background ...")
    irf_signal = irf_expected_counts(
        irfs["aeff"], irfs["smearing"], LOG10_E_EDGES, DEC_MIN, DEC_MAX, astro_flux, livetime_s,
    )
    irf_background = irf_expected_counts_directional(
        irfs["aeff"], irfs["smearing"], LOG10_E_EDGES, DEC_MIN, DEC_MAX, atm_flux, livetime_s,
    )

    print("Computing soft-volume signal and atmospheric background ...")
    # BGR18 rather than the paper's power law: the atmospheric background lives
    # at 1-100 TeV, where extrapolating the power law down from its 10 PeV anchor
    # overshoots by factors of 2.7 to 9.
    response = SoftVolumeResponse(
        radius_km=RADIUS_KM, method="exact", cross_section=bgr18_cross_section(),
    )
    soft_signal = soft_volume_signal(response, livetime_s)
    soft_background = response.expected_counts_from_flux(
        LOG10_E_EDGES, atm_flux, livetime_s, DEC_MIN, DEC_MAX,
    )

    print("Atmospheric prediction vs. data, per bin (MCEq normalization as computed):")
    for i, (lo, hi) in enumerate(zip(LOG10_E_EDGES[:-1], LOG10_E_EDGES[1:])):
        print(
            f"  {lo:4.1f}-{hi:4.1f}: observed {observed[i]:9,.0f}   "
            f"IRF atm {irf_background[i]:11,.1f}   soft atm {soft_background[i]:11,.1f}"
        )

    k_atm = calibrate_atmospheric_normalization(observed, irf_signal, irf_background)
    print(
        f"Atmospheric flux normalization from 10^{LOG10_E_MIN_BKG_CAL:.1f}-"
        f"10^{LOG10_E_MIN_FIT:.1f} GeV on the IRF path: k_atm = {k_atm:.2f}"
    )

    print(f"Fitting the astrophysical normalization above 10^{LOG10_E_MIN_FIT:.1f} GeV ...")
    in_fit = 0.5 * (LOG10_E_EDGES[:-1] + LOG10_E_EDGES[1:]) >= LOG10_E_MIN_FIT
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, signal, raw_background in (
        ("soft", soft_signal, soft_background),
        ("IRF", irf_signal, irf_background),
    ):
        # k_atm corrects the flux, not the response, so it carries over to the
        # soft-volume path unchanged. Nothing else about the background floats:
        # in the signal window it would be unconstrained.
        background = raw_background * k_atm
        scale = fit_scale_factor_with_background(
            observed, signal, background, LOG10_E_EDGES, LOG10_E_MIN_FIT,
        )
        predictions[name] = (signal * scale, background)
        print(
            f"  {name:4s}: best-fit phi0 = {PHI0 * scale:6.3f}   "
            f"(in window: observed {observed[in_fit].sum():.0f}, "
            f"atmospheric {background[in_fit].sum():5.1f}, "
            f"astrophysical at reference {signal[in_fit].sum():5.1f})"
        )
        if background[in_fit].sum() > observed[in_fit].sum():
            ceiling = observed[in_fit].sum() / background[in_fit].sum()
            print(
                f"        background alone over-predicts the data, so the selection "
                f"efficiency in this window is at most {ceiling:.2f}"
            )

    make_figure(observed, predictions, args.out)


if __name__ == "__main__":
    main()
