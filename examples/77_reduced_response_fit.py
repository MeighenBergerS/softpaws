"""Example 77 -- the response with the physics fixed: threshold and reach only.

Examples 33/56/72/73 fit five constants per site. Three of them are not
the instrument's to set. ``b_scale`` returns its prior at every site,
``lam`` is the cross-section slope and one number in nature, and
``eps_0`` is a selection efficiency the collaborations quote. This
example fixes all three -- ``b_scale = 1``, ``lam`` at BGR18, ``eps_0``
at the quoted analysis-level plateau for IceCube and at unity for the
trigger- and proposal-level water tables -- and refits each site with
only ``log10(E_thr)`` and ``Lambda`` free. The deviance of the reduced
fit is compared with the cached five-parameter best fit of example 73 on
the same nodes and the same 15% error model, so the question is whether
two instrument numbers describe each table as well as five did.

A three-parameter variant leaves ``eps_0`` free with the physics fixed,
to show what efficiency each table asks for once the slope and the
kernel can no longer absorb it.

Usage
-----
    python examples/77_reduced_response_fit.py
    python examples/77_reduced_response_fit.py --steps 3000 --eps-icecube 0.95
"""

import argparse
import importlib.util
import json
import pathlib

import emcee
import numpy as np
from scipy.optimize import minimize

_HERE = pathlib.Path(__file__).parent
_DEFAULT_DATA_DIR = _HERE.parent / "src" / "softpaws" / "data" / "dataverse_files"
_DEFAULT_OUT_DIR = _HERE / "output"


def load_example(stem: str, name: str):
    """Import a sibling example module by file stem."""
    spec = importlib.util.spec_from_file_location(name, _HERE / stem)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EX56 = load_example("56_four_detector_posterior_corner.py", "_example_56")
_EX33 = _EX56._EX33

#: Fixed selection efficiency per site. IceCube: the through-going
#: analysis-level efficiency the paper quotes; water sites: trigger or
#: proposal level, nothing to lose by construction.
EPS_FIXED = {"IceCube": 0.956, "ARCA230": 1.0, "P-ONE": 1.0, "TRIDENT": 1.0}

#: Derived reach expectation per medium [km per e-fold], the optical
#: attenuation lengths of App. F (59 m ice, 68 m water).
REACH_DERIVED_KM = {"IceCube": 0.059, "ARCA230": 0.068, "P-ONE": 0.068, "TRIDENT": 0.068}

FREE2 = ("log10_e_thr", "reach_km")
FREE3 = ("eps_0", "log10_e_thr", "reach_km")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=pathlib.Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--walkers", type=int, default=16)
    parser.add_argument("--sigma", type=float, default=0.15)
    parser.add_argument("--eps-icecube", type=float, default=EPS_FIXED["IceCube"])
    parser.add_argument("--out-dir", type=pathlib.Path, default=_DEFAULT_OUT_DIR)
    return parser.parse_args()


def full_theta(free_names, free_values, fixed):
    theta = np.array([fixed[n] for n in _EX33.PARAM_NAMES], dtype=float)
    for name, value in zip(free_names, free_values):
        theta[_EX33.PARAM_NAMES.index(name)] = value
    return theta


def deviance(detector, theta, sigma_ln):
    predicted = detector.predict(theta, detector.mask)
    if not np.all(np.isfinite(predicted)) or np.any(predicted <= 0.0):
        return np.inf, None
    residual = np.log(detector.observed[detector.mask] / predicted)
    return float(np.sum((residual / sigma_ln) ** 2)), residual


def make_logprob(detector, free_names, fixed, sigma_ln):
    def logprob(values):
        for name, value in zip(free_names, values):
            low, high = detector.priors[name]
            if not low < value < high:
                return -np.inf
        dev, _ = deviance(detector, full_theta(free_names, values, fixed), sigma_ln)
        return -0.5 * dev
    return logprob


def fit(detector, free_names, fixed, sigma_ln, steps, walkers, seed):
    logprob = make_logprob(detector, free_names, fixed, sigma_ln)
    start = np.array([fixed[n] for n in free_names])
    # Best fit first, from a small grid of starts so the sampler begins on it.
    best, best_val = start, -logprob(start)
    rng = np.random.default_rng(seed)
    for _ in range(6):
        x0 = np.array([rng.uniform(*detector.priors[n]) for n in free_names])
        if not np.isfinite(logprob(x0)):
            continue
        res = minimize(lambda x: -logprob(x), x0, method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 2000})
        if res.fun < best_val:
            best, best_val = res.x, res.fun
    scatter = np.array([{"eps_0": 0.01, "log10_e_thr": 0.02, "reach_km": 0.002}[n]
                        for n in free_names])
    initial = best + scatter * rng.standard_normal((walkers, len(free_names)))
    sampler = emcee.EnsembleSampler(walkers, len(free_names), logprob)
    sampler.run_mcmc(initial, steps, progress=False)
    chain = sampler.get_chain(discard=steps // 3, flat=True)
    lp = sampler.get_log_prob(discard=steps // 3, flat=True)
    best = chain[np.argmax(lp)]
    return best, chain


def main() -> None:
    args = parse_args()
    EPS_FIXED["IceCube"] = args.eps_icecube
    detectors = _EX56.build_detectors(args.data_dir)
    cache = args.out_dir / "73_chains.npz"
    five = np.load(cache)
    out = {}
    chains = {"sigma": np.array(args.sigma), "free2": np.array(FREE2), "free3": np.array(FREE3)}
    print(f"\nPhysics fixed: b_scale = 1, lam = {_EX33.LAMBDA_BGR18} (BGR18); "
          f"eps_0 fixed at {EPS_FIXED}")
    for seed, d in enumerate(detectors, start=21):
        fixed = {"eps_0": EPS_FIXED[d.name], "log10_e_thr": np.log10(_EX33.DEFAULT_MUON_THRESHOLD_GEV),
                 "b_scale": 1.0, "lam": _EX33.LAMBDA_BGR18, "reach_km": 0.03}
        n = int(d.mask.sum())
        best5 = five[f"{d.name}_best"]
        dev5, res5 = deviance(d, best5, args.sigma)
        print(f"\n=== {d.name} ({d.selection_level} level, {n} nodes) ===")
        print(f"  five-parameter best (ex73): eps_0 {best5[0]:.3f}, log10 E_thr {best5[1]:.2f}, "
              f"b {best5[2]:.3f}, lam {best5[3]:.3f}, Lambda {1e3*best5[4]:.0f} m; "
              f"deviance {dev5:.2f} / {n - 5} dof, rms {np.std(res5)/np.log(10):.3f} dex, "
              f"trend {(res5[-1]-res5[0])/np.log(10):+.3f} dex")
        entry = {"nodes": n, "five_param": {"best": best5.tolist(), "deviance": dev5}}
        for tag, free in (("2-param (E_thr, Lambda)", FREE2), ("3-param (+eps_0)", FREE3)):
            best, chain = fit(d, free, fixed, args.sigma, args.steps, args.walkers, seed)
            theta = full_theta(free, best, fixed)
            dev, res = deviance(d, theta, args.sigma)
            q = {name: np.percentile(chain[:, k], [16, 50, 84]) for k, name in enumerate(free)}
            print(f"  {tag}: deviance {dev:.2f} / {n - len(free)} dof, "
                  f"rms {np.std(res)/np.log(10):.3f} dex, trend {(res[-1]-res[0])/np.log(10):+.3f} dex")
            for name in free:
                lo, med, hi = q[name]
                scale, unit = (1e3, " m") if name == "reach_km" else (1.0, "")
                extra = ""
                if name == "reach_km":
                    extra = f"   (derived {1e3*REACH_DERIVED_KM[d.name]:.0f} m)"
                if name == "log10_e_thr":
                    extra = f"   (E_thr {10**med:.0f} GeV, 68% {10**lo:.0f}-{10**hi:.0f})"
                print(f"      {name:>12}: best {scale*theta[_EX33.PARAM_NAMES.index(name)]:.3f}{unit}, "
                      f"median {scale*med:.3f}{unit} [{scale*lo:.3f}, {scale*hi:.3f}]{extra}")
            entry[tag] = {"best": theta.tolist(), "deviance": dev,
                          "quantiles": {k: v.tolist() for k, v in q.items()}}
            chains[f"{d.name}_{len(free)}p_chain"] = chain
            chains[f"{d.name}_{len(free)}p_best"] = theta
        out[d.name] = entry
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "77_reduced_fit.json"
    path.write_text(json.dumps(out, indent=2))
    np.savez(args.out_dir / "77_chains.npz", **chains)
    print(f"\nWritten to {path} and 77_chains.npz")


if __name__ == "__main__":
    main()
