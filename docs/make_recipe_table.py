"""Generate the plug-in recipe table from the fit caches.

Reads the four-site informed-prior chains of example 73 (the quotable
calibration, parameters ``eps_0``, ``log10_e_thr``, ``b_scale``, ``lam``,
``reach_km`` per site), rebuilds each site's forward model through example
56's ``build_detectors``, and writes ``docs/recipe_table.tex`` -- one shared
transport block plus one row per detector with a worked reference value,
the effective area at 1 PeV evaluated at the posterior medians.

The table is machine-written so the paper never carries hand-transcribed
fit numbers; rerun after any refit that touches ``73_chains.npz``.

Usage
-----
    .venv/bin/python docs/make_recipe_table.py
"""

import importlib.util
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).parent
_EXAMPLES = _HERE.parent / "examples"
_CHAINS = _EXAMPLES / "output" / "73_chains.npz"
_OUT = _HERE / "recipe_table.tex"

#: Chain column order, example 33's PARAM_NAMES (examples 56/72/73 share it).
PARAMS = ("eps_0", "log10_e_thr", "b_scale", "lam", "reach_km")

SITES = ("IceCube", "ARCA230", "P-ONE", "TRIDENT")

#: Reference energy of the worked value [log10 GeV].
REFERENCE_LOG10_E = 6.0


def load_example(stem: str, name: str):
    """Import a sibling example module by file stem."""
    spec = importlib.util.spec_from_file_location(name, _EXAMPLES / stem)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summaries():
    """Posterior median and central 68% width per parameter per site."""
    chains = np.load(_CHAINS)
    out = {}
    for site in SITES:
        chain = chains[f"{site}_chain"]
        med = np.median(chain, axis=0)
        lo, hi = np.percentile(chain, [15.865, 84.135], axis=0)
        out[site] = {name: (med[i], med[i] - lo[i], hi[i] - med[i])
                     for i, name in enumerate(PARAMS)}
        out[site]["_median_theta"] = med
    return out


def reference_areas(stats):
    """Model effective area at the reference energy, posterior medians [cm^2]."""
    ex56 = load_example("56_four_detector_posterior_corner.py", "_example_56")
    detectors = ex56.build_detectors(ex56._DEFAULT_DATA_DIR)
    areas = {}
    for detector in detectors:
        theta = stats[detector.name]["_median_theta"]
        predicted = detector.predict(theta, None)
        areas[detector.name] = 10.0 ** np.interp(
            REFERENCE_LOG10_E, detector.log10_e,
            np.log10(np.clip(predicted, 1.0e-30, None)))
    return areas


def fmt(value, minus, plus, digits=2):
    """``m^{+p}_{-m}`` with symmetric collapse when the sides agree."""
    if round(minus, digits) == round(plus, digits):
        return f"${value:.{digits}f} \\pm {minus:.{digits}f}$"
    return f"${value:.{digits}f}^{{+{plus:.{digits}f}}}_{{-{minus:.{digits}f}}}$"


def main() -> None:
    stats = summaries()
    areas = reference_areas(stats)

    lines = [
        "% Machine-written by docs/make_recipe_table.py -- do not edit by hand.",
        "% Source: examples/output/73_chains.npz (informed-prior four-site fit).",
        "\\begin{tabular}{lcccccc}",
        "\\hline\\hline",
        ("detector & $\\varepsilon_0$ & $E_{\\mathrm{thr}}$ [TeV] & "
         "$b_{\\mathrm{scale}}$ & $\\lambda$ & $\\Lambda$ [m/$e$-fold] & "
         "$A_{\\mathrm{eff}}(1~\\mathrm{PeV})$ [m$^2$] \\\\"),
        "\\hline",
    ]
    for site in SITES:
        row = stats[site]
        e_thr = tuple(10.0 ** np.array([
            row["log10_e_thr"][0],
            row["log10_e_thr"][0] - row["log10_e_thr"][1],
            row["log10_e_thr"][0] + row["log10_e_thr"][2]]) / 1.0e3)
        cells = [
            site,
            fmt(*row["eps_0"]),
            fmt(e_thr[0], e_thr[0] - e_thr[1], e_thr[2] - e_thr[0], 1),
            fmt(*row["b_scale"]),
            fmt(*row["lam"]),
            fmt(1.0e3 * row["reach_km"][0], 1.0e3 * row["reach_km"][1],
                1.0e3 * row["reach_km"][2], 0),
            f"${float(f'{areas[site] / 1.0e4:.3g}'):.0f}$",
        ]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\hline\\hline", "\\end{tabular}"]

    _OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {_OUT}")
    print(f"\nWorked reference values, A_eff at 10^{REFERENCE_LOG10_E:.0f} GeV "
          "(posterior medians):")
    for site in SITES:
        print(f"  {site:<10s} {areas[site]:10.1f} cm^2")


if __name__ == "__main__":
    main()
