"""Example 80 -- what makes TRIDENT's upgoing band grow slowly.

Example 79 shows the negative reach of TRIDENT's sky average is an upgoing
statement. The downgoing band, where the overburden caps the range, is
reproduced with zero reach, while the upgoing band falls 0.155 dex against
the model across the window and the horizon band 0.03 dex. Two things in
their simulation could do that and ours cannot see from the sky average.
The Earth could absorb more strongly than BGR18 through a PREM column, or
the vertex-generation volume could be finite, so that muons born far from
the array are never made. This example holds the reach at the value the
measured optics give (20 m), frees only the threshold, and scans an
absorption scale on the column and a cap on the muon column, band by band.

Usage
-----
    python scripts/2026_muon_transport/80_trident_upgoing_scan.py
"""

import importlib.util
import pathlib

import numpy as np
from scipy.optimize import minimize_scalar

_HERE = pathlib.Path(__file__).parent


def load_example(stem: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / stem)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EX77 = load_example("77_reduced_response_fit.py", "_example_77")
_EX56 = _EX77._EX56
_EX55 = _EX56._EX55
_EX33 = _EX77._EX33

REACH_KM = 0.020        # the measured South China Sea attenuation length
SIGMA = 0.05
ABSORPTION_SCALES = (1.0, 1.1, 1.2, 1.3, 1.5)
COLUMN_CAPS_KM = (1.0, 2.0, 3.0, 5.0, 10.0, None)


def main() -> None:
    site = [s for s in _EX56.water_sites() if s.name == "TRIDENT"][0]
    zenith_weights, neutrino_column, muon_column_km = _EX56.water_columns(site)
    theta_deg, _ = _EX33.arca_zenith_grid()
    cos_theta = np.cos(np.deg2rad(theta_deg))
    grid = _EX33.ARCA_LOG10_E
    bands = list(zip(("up", "horizon", "down"), _EX55.trident_bands(), strict=True))
    print(f"TRIDENT bands, reach fixed at {1e3*REACH_KM:.0f} m, eps_0 = 1, physics fixed, "
          f"E_thr free; deviance at {100*SIGMA:.0f}% per node, tilt = last/first node [dex]")
    for key, (lo, hi, log10_e, log10_a) in bands:
        observed = _EX56._on_grid(log10_e, log10_a)
        mask = (grid >= site.fit_band[0]) & (grid <= site.fit_band[1]) & np.isfinite(observed)
        weights = np.where((cos_theta >= lo) & (cos_theta < hi), zenith_weights, 0.0)
        print(f"\n=== {key} ({lo:+.1f} < cos < {hi:+.1f}) ===")
        print(f"  {'cap [km]':>9} " + " ".join(f"{'x'+str(f):>22}" for f in ABSORPTION_SCALES))
        for cap in COLUMN_CAPS_KM:
            column = muon_column_km if cap is None else np.minimum(muon_column_km, cap)
            row = []
            for f in ABSORPTION_SCALES:
                ladders = _EX56.water_ladders(site, f * neutrino_column)
                def dev_of(log10_thr, ladders=ladders, weights=weights, column=column,
                           mask=mask, observed=observed):
                    theta = np.array([1.0, log10_thr, 1.0, _EX33.LAMBDA_BGR18, REACH_KM])
                    pred = _EX56.water_model(theta, site, ladders, weights, column, mask)
                    if not np.all(np.isfinite(pred)) or np.any(pred <= 0):
                        return np.inf, None
                    res = np.log(observed[mask] / pred)
                    return float(np.sum((res / SIGMA) ** 2)), res
                best = minimize_scalar(lambda x: dev_of(x)[0], bounds=(1.5, 4.5), method="bounded")
                dev, res = dev_of(best.x)
                row.append(f"{dev:6.1f} {10**best.x:6.0f}GeV {(res[-1]-res[0])/np.log(10):+.3f}")
            print(f"  {str(cap) if cap else 'none':>9} " + " ".join(f"{r:>22}" for r in row))


if __name__ == "__main__":
    main()
