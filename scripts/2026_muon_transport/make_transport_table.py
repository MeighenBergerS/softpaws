"""Generate the transport validation table and the kernel rows of the inputs table.

Writes ``output/transport_table.tex``, the body of the range-to-threshold table
that Appendix~\\ref{app:renewal} carries: the two loss rates of the kernel and
four ranges for a muon in water with a 1 TeV threshold, over five decades of
production energy. It also prints the kernel rows that the inputs table quotes,
water at 1 and 100 PeV and standard rock at 1 PeV, so the two tables cannot
drift apart.

Every entry comes from the library, so the paper never carries hand-transcribed
transport numbers; rerun after any change to the loss tabulation or to the
range construction.

Usage
-----
    python scripts/2026_muon_transport/make_transport_table.py
"""

import pathlib

import numpy as np

from softpaws.transport.coefficients import drift_coefficient, log_loss_moments
from softpaws.transport.muon_range import (
    DEFAULT_IONIZATION_MATCH_GEV,
    muon_range_km,
    stochastic_muon_range_km,
)

_HERE = pathlib.Path(__file__).parent
_OUT = _HERE / "output" / "transport_table.tex"

#: Production energies of the table columns [log10 GeV].
LOG10_E = np.arange(4.0, 9.0)

#: Muon threshold of the table [GeV].
THRESHOLD_GEV = 1.0e3


def rows() -> dict[str, np.ndarray]:
    """The table's rows, all in water at :data:`THRESHOLD_GEV`."""
    energy = 10.0**LOG10_E
    common = {"threshold_gev": THRESHOLD_GEV}
    return {
        "b": np.atleast_1d(drift_coefficient(energy)),
        "phi1": np.atleast_1d(log_loss_moments(energy)[0]),
        "L": np.atleast_1d(stochastic_muon_range_km(energy, **common)),
        "csda": np.atleast_1d(muon_range_km(energy, **common)),
        "radiative": np.atleast_1d(
            stochastic_muon_range_km(energy, include_ionization=False, **common)
        ),
        "frozen": np.atleast_1d(
            stochastic_muon_range_km(energy, kernel_evaluation="frozen", **common)
        ),
    }


def kernel_rows() -> dict[str, tuple[float, float]]:
    """The drift and log-loss rates the inputs table quotes, per unit column."""
    out = {}
    for name, energy, source in (
        ("water 1 PeV", 1.0e6, "proposal"),
        ("water 100 PeV", 1.0e8, "proposal"),
        ("standard rock 1 PeV", 1.0e6, "proposal_rock"),
    ):
        grid = np.atleast_1d(energy)
        b = float(np.atleast_1d(drift_coefficient(grid, source=source))[0])
        phi1 = float(np.atleast_1d(log_loss_moments(grid, source=source)[0])[0])
        out[name] = (b, phi1)
    return out


def cells(values: np.ndarray, digits: int) -> str:
    """One row of the tabular body."""
    return " & ".join(f"{v:.{digits}f}" for v in values)


def main() -> None:
    """Write the table body and report the kernel rows."""
    r = rows()
    header = " & ".join(f"${int(x)}$" for x in LOG10_E)
    lines = [
        "% Machine-written by scripts/2026_muon_transport/make_transport_table.py; "
        "do not edit by hand.",
        "% Source: the softpaws library, PROPOSAL tabulation, water at 1.02 g/cm^3,",
        f"% E_thr = {THRESHOLD_GEV:.0f} GeV. Ranges in km of water equivalent, rates in km^-1.",
        "\\begin{ruledtabular}",
        "\\begin{tabular}{lccccc}",
        f"$\\log_{{10}}(E_\\mu/\\mathrm{{GeV}})$ & {header} \\\\",
        "\\colrule",
        f"$b_\\mu = \\langle y \\rangle$          & {cells(r['b'], 3)} \\\\",
        f"$\\Phi'(0) = \\langle -\\ln(1{{-}}y)\\rangle$ & {cells(r['phi1'], 3)} \\\\",
        "\\colrule",
        f"$L$, Eq.~(\\ref{{eq:Lclosed}}) & {cells(r['L'], 2)} \\\\",
        f"$R_{{\\mathrm{{CSDA}}}}$         & {cells(r['csda'], 2)} \\\\",
        f"$L$, purely radiative       & {cells(r['radiative'], 2)} \\\\",
        "\\colrule",
        f"$L$, kernel frozen at production & {cells(r['frozen'], 2)} \\\\",
        "\\end{tabular}",
        "\\end{ruledtabular}",
    ]
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {_OUT}")

    ratio = r["L"] / r["csda"]
    print("\nRatios the prose reads off the table")
    for x, value in zip(LOG10_E, ratio, strict=True):
        print(f"  L / R_CSDA at 10^{int(x)} GeV: {value:.3f}")
    frozen = r["frozen"] / r["L"] - 1.0
    print("  frozen against running: " + ", ".join(f"{v:+.1%}" for v in frozen))
    print(f"  b_mu / Phi'(0): {r['b'][-1] / r['phi1'][-1]:.3f} at 10^8 GeV")

    print("\nKernel rows of the inputs table [km^-1]")
    for name, (b, phi1) in kernel_rows().items():
        print(f"  {name}: b_mu = {b:.3f}, Phi'(0) = {phi1:.3f}")

    print("\nThe Jensen asymptote b_mu / Phi'(0)")
    ends = np.array([1.0e3, 1.0e9])
    b_end = np.atleast_1d(drift_coefficient(ends))
    phi_end = np.atleast_1d(log_loss_moments(ends)[0])
    print(f"  1 TeV: {b_end[0] / phi_end[0]:.3f}, 1 EeV: {b_end[1] / phi_end[1]:.3f}")

    star = np.atleast_1d(DEFAULT_IONIZATION_MATCH_GEV)
    first, second, _ = (float(np.atleast_1d(m)[0]) for m in log_loss_moments(star))
    overshoot = second / (2.0 * first)
    arrival = float(star[0]) * np.exp(-overshoot)
    double_count = float(
        np.atleast_1d(muon_range_km(star, threshold_gev=arrival))[0]
    )
    print("\nThe ionization splice")
    print(f"  E_* = {star[0]:.0f} GeV, mean overshoot = {overshoot:.3f} in log energy")
    print(f"  arrival energy E_a = {arrival:.3g} GeV")
    print(f"  track counted twice if the segment starts at E_*: {double_count:.2f} km")


if __name__ == "__main__":
    main()
