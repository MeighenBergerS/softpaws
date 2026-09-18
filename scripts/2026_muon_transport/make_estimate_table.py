"""Write the paper's table of everything in the one-line estimate but the cross section.

``output/estimate_table.tex``: one column per site, one row per input of
Eq. (skyavg) as Figure 3 evaluates it with nothing fitted: the normalization,
the threshold and the reach of example 89's first-principles inputs, the
power ``k`` of Eq. (kreach) they imply, the mean projected area ``S / 4`` and
the instrumented volume of the site's geometry, and the transmission ``T``
over the upgoing part of the site's sky at three energies, for the BGR18
cross section with ``sigma_tot = 1.4 sigma_CC`` and the PREM Earth. The
downgoing half of a sea site's sky is transparent to better than a percent
at every energy the tables cover, so it is stated in the caption instead of
tabulated.

The table is machine-written so the paper never carries hand-transcribed
numbers; the transport table and the recipe table follow the same rule.

Usage
-----
    python scripts/2026_muon_transport/make_estimate_table.py
"""

import importlib.util
import pathlib

import numpy as np

from softpaws._paper import site_fit as sm
from softpaws.transport.attenuation import survival_probability
from softpaws.transport.earth import prem_column

_HERE = pathlib.Path(__file__).parent
_OUT = _HERE / "output" / "estimate_table.tex"

#: Energies of the transmission rows [log10 GeV].
LOG10_E = np.array([5.0, 6.0, 7.0])

#: Column order.
SITES = ("IceCube", "ARCA230", "P-ONE", "TRIDENT")


def load_example(stem: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / stem)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def upgoing_transmission(name: str) -> np.ndarray:
    """Mean survival over the upgoing part of the site's sky, solid-angle weighted."""
    energy = 10.0 ** LOG10_E[:, None]
    if name == "IceCube":
        dec_deg = np.linspace(0.5, 89.5, sm.IC_N_DEC)
        columns = np.array([prem_column(float(d)) for d in dec_deg])
        survival = survival_probability(energy, columns[None, :], sm.LAMBDA_BGR18,
                                        sm.CROSS_SECTION)
        return np.average(survival, axis=1, weights=np.cos(np.deg2rad(dec_deg)))
    site = (
        sm.ARCA230_WATER_SITE if name == "ARCA230"
        else [s for s in sm.water_sites() if s.name == name][0]
    )
    theta_deg, weights = sm.arca_zenith_grid()
    up = np.cos(np.deg2rad(theta_deg)) < 0.0
    _, neutrino_column, _ = sm.water_columns(site)
    survival = survival_probability(energy, neutrino_column[None, :], sm.LAMBDA_BGR18,
                                    sm.CROSS_SECTION)
    return np.average(survival[:, up], axis=1, weights=weights[up])


def main() -> None:
    """Assemble the rows from example 89's inputs and the library, and write."""
    ex89 = load_example("89_sky_averaged_plug_in.py", "_example_89")
    columns = {}
    for name in SITES:
        eps_0, threshold_gev, _ = ex89.FIRST_PRINCIPLES[name]
        reach_km = ex89.first_principles_reach_km(name)
        radius, height, side = ex89.site_geometry(name)
        n_blocks = 1 if name == "IceCube" else (
            sm.ARCA230_WATER_SITE.n_blocks if name == "ARCA230"
            else [s for s in sm.water_sites() if s.name == name][0].n_blocks
        )
        k_reach, _ = ex89.analytic_k(name, reach_km, threshold_gev, threshold_gev)
        area = ex89._EX88.mean_projected_area_km2(radius, height, n_blocks, side)
        v_det = n_blocks * np.pi * radius**2 * height
        columns[name] = {
            "eps": f"{eps_0:.3g}", "thr": f"{threshold_gev:.0f}",
            "reach": f"{1.0e3 * reach_km:.0f}", "k": f"{k_reach:.2f}",
            "area": f"{float(area):.2f}", "vdet": f"{float(v_det):.2f}",
            "T": [f"{v:.2f}" for v in upgoing_transmission(name)],
        }

    def row(label: str, key: str, index: int | None = None) -> str:
        cells = [columns[s][key] if index is None else columns[s][key][index] for s in SITES]
        return label + " & " + " & ".join(f"${c}$" for c in cells) + r" \\"

    lines = [
        "% Machine-written by scripts/2026_muon_transport/make_estimate_table.py; "
        "do not edit by hand.",
        "% Sources: example 89's first-principles inputs, the site geometry of the "
        "softpaws library, BGR18 with sigma_tot = 1.4 sigma_CC, PREM.",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{l" + "c" * len(SITES) + "}",
        " & " + " & ".join(SITES) + r" \\",
        r"\colrule",
        row(r"$\epsilon_0$", "eps"),
        row(r"$E_{\mathrm{thr}}$ [GeV]", "thr"),
        row(r"$\Lambda$ [m]", "reach"),
        row(r"$k$, Eq.~(\ref{eq:kreach})", "k"),
        row(r"$\langle A_{\mathrm{proj}} \rangle$ [km$^2$]", "area"),
        row(r"$V_{\mathrm{det}}$ [km$^3$]", "vdet"),
        r"\colrule",
    ]
    for i, log10_e in enumerate(LOG10_E):
        lines.append(row(rf"$T$, $10^{{{log10_e:.0f}}}$~GeV", "T", i))
    lines += [r"\end{tabular}", r"\end{ruledtabular}"]
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {_OUT}")
    print("\n".join(lines[4:-2]))


if __name__ == "__main__":
    main()
