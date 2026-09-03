"""Generate the plug-in recipe table from the reduced-response fit caches.

Reads the two-parameter chains of example 77 (physics fixed, ``eps_0`` held
at the collaboration's level, ``log10_e_thr`` and ``reach_km`` sampled at
5% per node) for IceCube, ARCA230 and P-ONE, and the three-parameter chain
of example 82 for TRIDENT (its 2025 map, ``eps_0`` free), and writes
``docs/recipe_table.tex``: one row per detector with the held or fitted
normalization, the threshold, the fitted reach, the reach the site's
measured optics predict, and a worked reference value, the effective area
at 1 PeV evaluated at the posterior medians.

The table is machine-written so the paper never carries hand-transcribed
fit numbers; rerun after any refit that touches the chains.

Usage
-----
    .venv/bin/python docs/make_recipe_table.py
"""

import importlib.util
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).parent
_EXAMPLES = _HERE.parent / "examples"
_CHAINS = _EXAMPLES / "output" / "77_chains_sigma05.npz"
_TRIDENT_CHAIN = _EXAMPLES / "output" / "82_trident2025_chain.npz"
_OUT = _HERE / "recipe_table.tex"

SITES = ("IceCube", "ARCA230", "P-ONE", "TRIDENT")

#: Reference energy of the worked value [log10 GeV].
REFERENCE_LOG10_E = 6.0


def load_example(stem: str, name: str):
    """Import a sibling example module by file stem."""
    spec = importlib.util.spec_from_file_location(name, _EXAMPLES / stem)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quantiles(values):
    """Median and central 68% half-widths of a chain column."""
    med = float(np.median(values))
    lo, hi = np.percentile(values, [15.865, 84.135])
    return med, med - float(lo), float(hi) - med


def fmt(value, minus, plus, digits=2):
    """``m^{+p}_{-m}`` with symmetric collapse when the sides agree."""
    if round(minus, digits) == round(plus, digits):
        return f"${value:.{digits}f} \\pm {minus:.{digits}f}$"
    return f"${value:.{digits}f}^{{+{plus:.{digits}f}}}_{{-{minus:.{digits}f}}}$"


def main() -> None:
    ex83 = load_example("83_four_detector_aeff_reduced.py", "_example_83")
    ex77 = ex83._EX77
    ex78 = load_example("78_reduced_response_plane.py", "_example_78")
    detectors = ex83.reduced_detectors(ex83._DEFAULT_DATA_DIR, _CHAINS)
    trident, _ = ex83.trident_2025(_TRIDENT_CHAIN)
    detectors.append(trident)
    by_name = {d.name: d for d in detectors}

    lines = [
        "% Machine-written by docs/make_recipe_table.py -- do not edit by hand.",
        "% Sources: examples/output/77_chains_sigma05.npz (IceCube, ARCA230, P-ONE; "
        "physics fixed, eps_0 held) and 82_trident2025_chain.npz (TRIDENT 2025 map, eps_0 free).",
        "\\begin{tabular}{lccccc}",
        "\\hline\\hline",
        ("detector & $\\varepsilon_0$ & $E_{\\mathrm{thr}}$ [GeV] & "
         "$\\Lambda$ fitted [m] & $\\Lambda$ predicted [m] & "
         "$A_{\\mathrm{eff}}(1~\\mathrm{PeV})$ [m$^2$] \\\\"),
        "\\hline",
    ]
    for site in SITES:
        d = by_name[site]
        chain = d.chain                      # full theta rows: eps_0, log10_e_thr, b, lam, reach_km
        e_med, e_lo, e_hi = quantiles(chain[:, 1])
        e_thr = (10 ** e_med, 10 ** e_med - 10 ** (e_med - e_lo), 10 ** (e_med + e_hi) - 10 ** e_med)
        r_med, r_lo, r_hi = quantiles(1.0e3 * chain[:, 4])
        if site == "TRIDENT":
            n_med, n_lo, n_hi = quantiles(chain[:, 0])
            eps_cell = fmt(n_med, n_lo, n_hi) + " (free)"
        else:
            eps_cell = f"${ex77.EPS_FIXED[site]:.3g}$ (held)"
        theta = np.median(chain, axis=0)
        predicted = np.asarray(d.predict(theta, None), dtype=float)
        area = 10 ** np.interp(REFERENCE_LOG10_E, d.log10_e,
                               np.log10(np.clip(predicted, 1.0e-30, None)))
        lo_p, hi_p = ex78.PREDICTED_M[site]
        cells = [
            site,
            eps_cell,
            fmt(e_thr[0], e_thr[1], e_thr[2], 0),
            fmt(r_med, r_lo, r_hi, 0),
            f"${lo_p:.0f}$ to ${hi_p:.0f}$",
            f"${float(f'{area / 1.0e4:.3g}'):.0f}$",
        ]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\hline\\hline", "\\end{tabular}"]
    _OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {_OUT}")
    print("\n".join(lines[5:-2]))


if __name__ == "__main__":
    main()
