"""Example 40 -- what a 30% photonuclear uncertainty does to the transport.

The paper takes PROPOSAL's kernel as given and reports a 12% shape agreement
against the published IceCube effective area, which reads as though the kernel
were exact. It is not. Photonuclear scattering is the least controlled of the
three radiative channels -- the low-``Q^2`` nuclear response is non-perturbative,
parametrizations disagree, and shadowing at ultra-high energy is uncertain -- and
Ref.~[Palmisano:2026sid] carries an explicit, energy-independent 30% uncertainty
on the photonuclear rate for exactly this reason. This script propagates that
same assignment through the exact transport.

Three questions, in order of how much they matter.

**How much of the exponent is photonuclear?** The channel decomposition is a
by-product and worth quoting, because it is the statement being stressed. With
the ALLM97 parametrization and Butkevich-Mikheyev shadowing that PROPOSAL uses
here, the channel carries 21% of ``b_mu`` and 29% of ``d_mu`` at 1 PeV in water.
Ref.~[Palmisano:2026sid] quotes 25% and about 50% for the same quantities with
its own choice of parametrization, so the share itself is parametrization
dependent at the level being scanned. Bremsstrahlung dominates ``d_mu`` on this
kernel.

**How much does the range move?** The tabulated effective area is proportional to
the range to threshold, so a swing in ``Phi'(0)`` passes through to the
prediction essentially undiluted. This is the number that belongs next to the
12%.

**Does it move the shape or only the normalization?** This is the question that
decides whether the comparison in Sec. VI is contaminated. The paper's 12% is
the root-mean-square scatter about a *free* normalization, so a photonuclear
shift that is common to every energy is absorbed by that normalization and costs
nothing. The script therefore splits the range swing into its mean offset across
the band and the residual tilt about that offset, and only the second competes
with the 12%.

The scan is a flat rescaling of the photonuclear differential rate by
``1 +- 0.30`` at every ``y``, which is what Ref.~[Palmisano:2026sid] assigns. A
shape uncertainty within the channel would act differently, and more sharply on
``Phi'(0)`` than on ``b_mu``, since the two weight the hard end differently.

Examples
--------
::

    python examples/40_photonuclear_systematic.py
    python examples/40_photonuclear_systematic.py --pn-uncertainty 0.15
    python examples/40_photonuclear_systematic.py --threshold-gev 1e4
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from softpaws.transport.coefficients import (
    loss_spectrum_y_grid,
    proposal_loss_spectrum,
)

_HERE = pathlib.Path(__file__).parent
_STYLE = _HERE.parent / "styles" / "beacom_conformal.mplstyle"
_DEFAULT_OUT_DIR = _HERE / "output"

REFERENCE_ENERGY_GEV = 1.0e6
DEFAULT_THRESHOLD_GEV = 1.0e3
DEFAULT_PN_UNCERTAINTY = 0.30
# The band over which the paper quotes the effective-area shape agreement,
# expressed as muon energies at production.
LOG10_ENERGY_SCAN = np.arange(4.0, 8.5, 0.5)
# The IceCube diffuse index, A = gamma - lambda - 1.
ICECUBE_INDEX = 0.98
# The shape scatter the paper reports for the IceCube effective area.
PUBLISHED_SHAPE_SCATTER = 0.12


# ---------------------------------------------------------------------------
# Kernel moments under a rescaled photonuclear channel
# ---------------------------------------------------------------------------


def kernel_moments(
    energy_gev: float,
    pn_scale: float,
    y: np.ndarray | None = None,
) -> dict[str, float]:
    """Moments of PROPOSAL's loss spectrum with the photonuclear rate rescaled.

    All five quantities are quadratures of the same ``dGamma/dy``, so a channel
    rescaling propagates to them consistently. The two log-loss moments weight
    the hard end more heavily than the plain ``y`` moments do, which is why they
    respond differently to the same rescaling.

    Parameters
    ----------
    energy_gev : float
        Muon energy [GeV].
    pn_scale : float
        Multiplicative factor on the photonuclear differential rate.
    y : np.ndarray, optional
        Grid of fractional energy losses. Defaults to
        :func:`~softpaws.transport.coefficients.loss_spectrum_y_grid`.

    Returns
    -------
    moments : dict
        Keys ``b_mu``, ``d_mu``, ``t_mu``, ``phi_prime`` (``Phi'(0)``) and
        ``phi_second`` (``-Phi''(0)``), all [km^-1].
    """
    grid = loss_spectrum_y_grid() if y is None else y
    spectrum = proposal_loss_spectrum(energy_gev, grid)
    total = (
        spectrum["bremsstrahlung"]
        + spectrum["pair production"]
        + pn_scale * spectrum["photonuclear"]
    )
    log_loss = -np.log1p(-grid)
    return {
        "b_mu": float(np.trapezoid(total * grid, grid)),
        "d_mu": float(np.trapezoid(total * grid**2, grid)),
        "t_mu": float(np.trapezoid(total * grid**3, grid)),
        "phi_prime": float(np.trapezoid(total * log_loss, grid)),
        "phi_second": float(np.trapezoid(total * log_loss**2, grid)),
    }


def phi_of_index(
    energy_gev: float,
    pn_scale: float,
    index: np.ndarray,
    y: np.ndarray | None = None,
) -> np.ndarray:
    """Exact exponent ``Phi(A)`` by quadrature, with the photonuclear rate rescaled.

    Parameters
    ----------
    energy_gev : float
        Muon energy [GeV].
    pn_scale : float
        Multiplicative factor on the photonuclear differential rate.
    index : np.ndarray
        Spectral indices ``A`` at which to evaluate the exponent.
    y : np.ndarray, optional
        Grid of fractional energy losses.

    Returns
    -------
    phi : np.ndarray
        ``Phi(A)`` [km^-1], same shape as ``index``.
    """
    grid = loss_spectrum_y_grid() if y is None else y
    spectrum = proposal_loss_spectrum(energy_gev, grid)
    total = (
        spectrum["bremsstrahlung"]
        + spectrum["pair production"]
        + pn_scale * spectrum["photonuclear"]
    )
    bracket = 1.0 - (1.0 - grid)[None, :] ** np.atleast_1d(index)[:, None]
    return np.trapezoid(total[None, :] * bracket, grid, axis=1)


def radiative_range_km(
    energy_gev: float,
    threshold_gev: float,
    moments: dict[str, float],
) -> float:
    """Purely radiative first-passage range, Eq.~(C4) without the ionization splice.

    The splice below ``E_*`` is deterministic and carries no photonuclear
    dependence worth tracking here, so the bare radiative range isolates the
    effect being measured.

    Parameters
    ----------
    energy_gev : float
        Muon energy at production [GeV].
    threshold_gev : float
        Muon energy below which the track is not selected [GeV].
    moments : dict
        Output of :func:`kernel_moments`.

    Returns
    -------
    range_km : float
        Expected range [km w.e.].
    """
    w = np.log(energy_gev / threshold_gev)
    return w / moments["phi_prime"] + moments["phi_second"] / (
        2.0 * moments["phi_prime"] ** 2
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def report_channel_shares(energy_gev: float) -> None:
    """Print each channel's share of the moments that drive the transport.

    Parameters
    ----------
    energy_gev : float
        Muon energy [GeV].
    """
    y = loss_spectrum_y_grid()
    spectrum = proposal_loss_spectrum(energy_gev, y)
    log_loss = -np.log1p(-y)
    labels = ("bremsstrahlung", "pair production", "photonuclear")
    weights = {"b_mu": y, "d_mu": y**2, "Phi'(0)": log_loss}

    print(f"Channel shares at E_mu = {energy_gev:.3g} GeV, water")
    header = f"{'channel':>16} " + " ".join(f"{name:>12}" for name in weights)
    print(header)
    print("-" * len(header))
    totals = {
        name: float(np.trapezoid(spectrum["total"] * weight, y))
        for name, weight in weights.items()
    }
    for label in labels:
        row = " ".join(
            f"{100.0 * float(np.trapezoid(spectrum[label] * weight, y)) / totals[name]:>11.1f}%"
            for name, weight in weights.items()
        )
        print(f"{label:>16} {row}")
    print(f"{'total [km^-1]':>16} " + " ".join(f"{totals[n]:>12.4f}" for n in weights))
    print()


def report_swing(
    energy_gev: float,
    threshold_gev: float,
    pn_uncertainty: float,
) -> None:
    """Print the effect of the photonuclear rescaling on the transport quantities.

    Parameters
    ----------
    energy_gev : float
        Muon energy [GeV].
    threshold_gev : float
        Muon energy threshold [GeV].
    pn_uncertainty : float
        Fractional uncertainty on the photonuclear rate.
    """
    central = kernel_moments(energy_gev, 1.0)
    low = kernel_moments(energy_gev, 1.0 - pn_uncertainty)
    high = kernel_moments(energy_gev, 1.0 + pn_uncertainty)

    print(f"Photonuclear rate rescaled by 1 +- {pn_uncertainty:.2f} "
          f"at E_mu = {energy_gev:.3g} GeV")
    header = f"{'quantity':>14} {'central':>10} {'-30%':>10} {'+30%':>10} {'swing':>9}"
    print(header)
    print("-" * len(header))
    for key, label in (
        ("b_mu", "b_mu"), ("d_mu", "d_mu"), ("t_mu", "t_mu"),
        ("phi_prime", "Phi'(0)"), ("phi_second", "-Phi''(0)"),
    ):
        swing = 100.0 * 0.5 * (high[key] - low[key]) / central[key]
        print(f"{label:>14} {central[key]:>10.4f} {low[key]:>10.4f} "
              f"{high[key]:>10.4f} {swing:>8.1f}%")

    phi_central = float(phi_of_index(energy_gev, 1.0, np.array([ICECUBE_INDEX]))[0])
    phi_low = float(
        phi_of_index(energy_gev, 1.0 - pn_uncertainty, np.array([ICECUBE_INDEX]))[0]
    )
    phi_high = float(
        phi_of_index(energy_gev, 1.0 + pn_uncertainty, np.array([ICECUBE_INDEX]))[0]
    )
    swing = 100.0 * 0.5 * (phi_high - phi_low) / phi_central
    print(f"{'Phi(A=0.98)':>14} {phi_central:>10.4f} {phi_low:>10.4f} "
          f"{phi_high:>10.4f} {swing:>8.1f}%")

    range_central = radiative_range_km(energy_gev, threshold_gev, central)
    range_low = radiative_range_km(energy_gev, threshold_gev, low)
    range_high = radiative_range_km(energy_gev, threshold_gev, high)
    swing = 100.0 * 0.5 * (range_low - range_high) / range_central
    print(f"{'L [km w.e.]':>14} {range_central:>10.3f} {range_low:>10.3f} "
          f"{range_high:>10.3f} {swing:>8.1f}%")
    print()


def report_shape_versus_normalization(
    log10_energies: np.ndarray,
    threshold_gev: float,
    pn_uncertainty: float,
) -> dict[str, np.ndarray]:
    """Split the range swing into a common offset and a residual tilt.

    The effective-area comparison of Sec. VI floats a normalization, so only the
    part of the photonuclear swing that varies across the band can contaminate
    the quoted shape agreement.

    Parameters
    ----------
    log10_energies : np.ndarray
        ``log10(E_mu / GeV)`` at which to evaluate the range.
    threshold_gev : float
        Muon energy threshold [GeV].
    pn_uncertainty : float
        Fractional uncertainty on the photonuclear rate.

    Returns
    -------
    result : dict
        Arrays keyed ``log10_energy`` and ``ratio_low``, ``ratio_high``, the
        range ratios to the central kernel.
    """
    energies = 10.0**log10_energies
    ratios = {}
    for label, scale in (("low", 1.0 - pn_uncertainty), ("high", 1.0 + pn_uncertainty)):
        ratio = np.array([
            radiative_range_km(energy, threshold_gev, kernel_moments(energy, scale))
            / radiative_range_km(energy, threshold_gev, kernel_moments(energy, 1.0))
            for energy in energies
        ])
        ratios[f"ratio_{label}"] = ratio

    print("Range ratio to the central kernel, across the comparison band")
    header = f"{'log10(E/GeV)':>13} {'-30%':>9} {'+30%':>9}"
    print(header)
    print("-" * len(header))
    for index, value in enumerate(log10_energies):
        print(f"{value:>13.1f} {ratios['ratio_low'][index]:>9.4f} "
              f"{ratios['ratio_high'][index]:>9.4f}")
    print()

    for label in ("low", "high"):
        ratio = ratios[f"ratio_{label}"]
        offset = float(np.mean(ratio))
        tilt = float(np.std(ratio / offset))
        print(f"  {label:>4}: common offset {100.0 * (offset - 1.0):+6.2f}%, "
              f"residual tilt {100.0 * tilt:5.2f}% rms")
    print(f"  for scale, the published shape scatter is "
          f"{100.0 * PUBLISHED_SHAPE_SCATTER:.0f}%")
    print()

    ratios["log10_energy"] = log10_energies
    return ratios


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def figure_systematic(
    energy_gev: float,
    pn_uncertainty: float,
    ratios: dict[str, np.ndarray],
    stem: pathlib.Path,
) -> None:
    """Draw the exponent band and the range swing across the band.

    Parameters
    ----------
    energy_gev : float
        Muon energy at which the exponent is drawn [GeV].
    pn_uncertainty : float
        Fractional uncertainty on the photonuclear rate.
    ratios : dict
        Output of :func:`report_shape_versus_normalization`.
    stem : pathlib.Path
        Output path without extension; ``.pdf`` and ``.png`` are written.
    """
    plt.style.use(_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

    ax = axes[0]
    # Away from A = 0, where Phi vanishes and the ratio is undefined.
    index = np.linspace(0.1, 8.0, 60)
    central = phi_of_index(energy_gev, 1.0, index)
    low = phi_of_index(energy_gev, 1.0 - pn_uncertainty, index)
    high = phi_of_index(energy_gev, 1.0 + pn_uncertainty, index)
    ax.plot(index, 100.0 * (high / central - 1.0), color="C0", lw=1.4,
            label=f"$+{100 * pn_uncertainty:.0f}\\%$ photonuclear")
    ax.plot(index, 100.0 * (low / central - 1.0), color="C3", lw=1.4,
            label=f"$-{100 * pn_uncertainty:.0f}\\%$ photonuclear")
    ax.axvline(ICECUBE_INDEX, color="0.5", lw=0.9, ls=":")
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set_xlabel(r"$A$")
    ax.set_ylabel(r"$\Delta\Phi(A)/\Phi(A)$ [\%]")
    ax.set_title("(a) the exponent")
    ax.legend(frameon=False, fontsize="small")

    ax = axes[1]
    ax.axhspan(-100.0 * PUBLISHED_SHAPE_SCATTER, 100.0 * PUBLISHED_SHAPE_SCATTER,
               color="0.88", lw=0, label="published shape scatter")
    ax.plot(ratios["log10_energy"], 100.0 * (ratios["ratio_high"] - 1.0),
            color="C0", lw=1.4, marker="o", ms=3.0,
            label=f"$+{100 * pn_uncertainty:.0f}\\%$")
    ax.plot(ratios["log10_energy"], 100.0 * (ratios["ratio_low"] - 1.0),
            color="C3", lw=1.4, marker="o", ms=3.0,
            label=f"$-{100 * pn_uncertainty:.0f}\\%$")
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set_xlabel(r"$\log_{10}(E_\mu/\mathrm{GeV})$")
    ax.set_ylabel(r"$\Delta L / L$ [\%]")
    ax.set_title("(b) the range to threshold")
    ax.legend(frameon=False, fontsize="small")

    fig.tight_layout()
    for suffix in (".pdf", ".png"):
        fig.savefig(stem.with_suffix(suffix))
    plt.close(fig)
    print(f"wrote {stem.with_suffix('.pdf')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR,
                        help="Directory for the output figure.")
    parser.add_argument("--energy-gev", type=float, default=REFERENCE_ENERGY_GEV,
                        help="Muon energy for the single-energy tables [GeV].")
    parser.add_argument("--threshold-gev", type=float, default=DEFAULT_THRESHOLD_GEV,
                        help="Muon energy threshold for the range [GeV].")
    parser.add_argument("--pn-uncertainty", type=float, default=DEFAULT_PN_UNCERTAINTY,
                        help="Fractional uncertainty on the photonuclear rate; "
                        "Ref. [Palmisano:2026sid] assigns 0.30.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    report_channel_shares(args.energy_gev)
    report_swing(args.energy_gev, args.threshold_gev, args.pn_uncertainty)
    ratios = report_shape_versus_normalization(
        LOG10_ENERGY_SCAN, args.threshold_gev, args.pn_uncertainty
    )
    figure_systematic(
        args.energy_gev, args.pn_uncertainty, ratios,
        args.out_dir / "40_photonuclear_systematic",
    )


if __name__ == "__main__":
    main()
