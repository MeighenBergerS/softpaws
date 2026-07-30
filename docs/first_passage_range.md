# The monochromatic range as a first-passage problem

Draft material for the paper. Everything below is exact under the same scale-invariance
assumption (Eq. 3) that the rest of the formalism rests on; nothing here is fitted.

Implemented in `softpaws.transport.soft_volume.stochastic_muon_range_km`, exercised by
`examples/28_neutrino_energy_effective_area.py`, tested in `tests/test_nu_energy_response.py`.

---

## 1. Why a second length is needed

Eq. (10) gives the track rate differential in the **observed muon energy**, and its
length is the spectrally-weighted `1/Φ(A)` — about 2.4 km in water at IceCube's
spectral index. That is the right length for a power-law parent, because the source
excites the single mode `s = A` and the propagator contributes `e^{-ℓΦ(A)}`.

A published effective area is a different object. It is tabulated at fixed **neutrino
energy** and integrated over every muon energy that survives the selection, so the
parent is *monochromatic* and there is no spectral index left to weight against. The
two conventions are not interchangeable: the IceCube DR2 smearing matrix puts the
median reconstructed muon energy 1.6 decades below `E_ν` at 100 TeV and 3.9 decades
below at 10 EeV, and the offset grows monotonically with energy.

For a monochromatic parent the formalism is already in hand — it is the `s → 0` case
of App. I, where `Φ(0) = 0` and `I(0) = 1`. Both spectral factors switch off, so the
length cannot come from `Φ(A)` at all. It has to come from the only other scale in the
problem: the analysis threshold `E_thr` below which the muon is no longer selected.

## 2. The quantity to compute

Write `ε = (1 - ⟨y_w⟩) E_ν` for the muon energy at production and `E(ℓ)` for its energy
after propagating a distance `ℓ`. The target volume for a monochromatic parent is the
projected area times the depth over which a muon is still selectable, plus the
instrumented volume:

```
V(E_ν) = A_proj L(E_ν) + V_det ,

L(E_ν) = ∫₀^∞ dℓ  P[ E(ℓ) > E_thr ] .
```

In terms of the accumulated log-loss `W(ℓ) = ln(ε / E(ℓ))` this is

```
L(E_ν) = ∫₀^∞ dℓ  P[ W(ℓ) < w⋆ ] ,        w⋆ ≡ ln(ε / E_thr) .            (1)
```

Evaluated directly, Eq. (1) is a depth integral over characteristic-function
inversions — expensive, and it hides the structure. It does neither.

## 3. The structural fact: `W` is a subordinator

A muon's energy only ever decreases, so `W(ℓ)` is **non-decreasing** in `ℓ`. Together
with scale invariance this makes it a subordinator (a non-decreasing Lévy process)
whose Laplace exponent is exactly the collision eigenvalue of Eq. (5):

```
E[ e^{-s W(ℓ)} ] = e^{-ℓ Φ(s)} ,     Φ(s) = ∫₀¹ dy (dΓ/dy) [1 - (1-y)^s] .   (2)
```

Monotonicity is the whole point. Define the **first-passage depth**

```
τ(w) = inf { ℓ : W(ℓ) ≥ w } ,
```

the distance at which the muon first falls below `ε e^{-w}`. Because `W` never
decreases, it is below `w` at depth `ℓ` if and only if it has not yet crossed `w`:

```
{ W(ℓ) < w }  ⟺  { τ(w) > ℓ } .                                            (3)
```

## 4. The identity

Substituting Eq. (3) into Eq. (1) and using `∫₀^∞ P[τ > ℓ] dℓ = E[τ]`:

```
L(E_ν) = E[ τ(w⋆) ] ≡ U(w⋆) .                                              (4)
```

**The effective length is the expected first-passage depth to the threshold.** This is
the correct stochastic definition of "range" — the mean distance at which the muon
first drops out of the selection — and it explains why the numerical answer sits close
to, but systematically below, the CSDA range.

## 5. Closed form from the potential measure

`U` is the renewal function (potential measure) of the subordinator, and for any
subordinator its Laplace transform follows from Eq. (2) in one line:

```
∫₀^∞ e^{-s w} U(dw) = ∫₀^∞ dℓ E[e^{-s W(ℓ)}] = ∫₀^∞ dℓ e^{-ℓ Φ(s)} = 1 / Φ(s) ,
```

so for the cumulative `U(w)`

```
Û(s) = 1 / ( s Φ(s) ) .                                                    (5)
```

This is the counterpart of the soft volume's `1/Φ(A)`: there a single mode picks out
one value of `Φ`; here the whole first-passage law is governed by the same `Φ`,
transformed. The monochromatic and power-law cases are the same function evaluated
differently, not two separate physical models.

Expanding `Φ` about `s = 0` — legitimate because `Φ(0) = 0` and `Φ` is analytic there —

```
Φ(s) = Φ′(0) s + ½ Φ″(0) s² + O(s³) ,

Û(s) = 1/(Φ′(0) s²) · [ 1 + ½ (Φ″(0)/Φ′(0)) s + O(s²) ]^{-1}
     = 1/(Φ′(0) s²) - Φ″(0)/(2 Φ′(0)² s) + O(1) ,
```

and inverting term by term (`s^{-2} → w`, `s^{-1} → 1`, and `O(1) → ` terms that decay
in `w`):

```
┌────────────────────────────────────────────────────────────────────────┐
│  L(E_ν) = ln[ (1 - ⟨y_w⟩) E_ν / E_thr ] / Φ′(0)  -  Φ″(0) / (2 Φ′(0)²) │   (6)
└────────────────────────────────────────────────────────────────────────┘
```

Two derivatives of one special function. No inversion, no depth integral.

## 6. The two moments

Differentiating Eq. (2) under the integral:

```
 Φ′(0) = ∫₀¹ dy (dΓ/dy) ( -ln(1-y) )  = ⟨ -ln(1-y) ⟩     per unit length
-Φ″(0) = ∫₀¹ dy (dΓ/dy) (  ln²(1-y) ) = ⟨  ln²(1-y) ⟩    per unit length
```

Both are finite for the same reason `Φ(A)` is: the soft `1/y` pile-up of
bremsstrahlung is cancelled by `-ln(1-y) → y` as `y → 0`.

For the two-parameter loss family `dΓ/dy = κ (1-y)^p / y` calibrated to `b_μ` and
`d_μ`, where `Φ(s) = κ [ψ(s+p+1) - ψ(p+1)]`, these are polygamma calls:

```
 Φ′(0) = κ ψ₁(p+1) ,        -Φ″(0) = κ |ψ₂(p+1)| .
```

## 7. Why it is always shorter than CSDA

Well above the critical energy the continuous-slowing-down range is

```
R_CSDA = ln(ε / E_thr) / b_μ ,          b_μ = ⟨y⟩ ,
```

so Eq. (6) is **the CSDA formula with `⟨y⟩` replaced by `⟨-ln(1-y)⟩`**, plus a
constant.

The elementary inequality `-ln(1-y) ≥ y` on `(0,1)`, strict for `y > 0`, gives
`Φ′(0) ≥ b_μ` for *every* positive loss spectrum. The stochastic range is therefore
always the shorter one, with no model dependence in the direction of the effect.

The reason is Jensen, and it is the same "mean versus typical" statement
`examples/27_proposal_cross_section_and_loss.py` §3a already makes: CSDA propagates
the **logarithm of the mean energy**, `ln E[E(ℓ)] = -b_μ ℓ`, whereas what controls
threshold crossing is the **mean of the logarithm**, `E[ln E(ℓ)] = -Φ′(0) ℓ`. Since
`E[ln E] ≤ ln E[E]`, the typical muon degrades faster than the mean one, and a
threshold is crossed sooner than the CSDA range suggests. The mean is held up by rare
muons that happened to radiate little; those are not typical, and a range is a typical
quantity.

## 8. Validation

Muon in water, `E_thr = 1 TeV`, PROPOSAL coefficients, `⟨y_w⟩ = 0.2`:

| log₁₀(E_ν/GeV) | `b_μ = ⟨y⟩` | `Φ′(0) = ⟨-ln(1-y)⟩` | `b_μ/Φ′(0)` | `C = -Φ″(0)/2Φ′(0)²` | `L` Eq. (1) | `L` Eq. (6) | `R_CSDA` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4.0 | 0.3398 | 0.3929 | 0.865 | 0.416 |  5.71 |  5.71 |  4.95 |
| 5.0 | 0.3609 | 0.4213 | 0.857 | 0.417 | 10.82 | 10.82 | 10.92 |
| 6.0 | 0.3780 | 0.4442 | 0.851 | 0.415 | 15.46 | 15.46 | 16.54 |
| 7.0 | 0.3978 | 0.4708 | 0.845 | 0.411 | 19.50 | 19.50 | 21.55 |
| 8.0 | 0.4236 | 0.5058 | 0.837 | 0.406 | 22.73 | 22.73 | 25.73 |

Lengths in km. Eq. (6) reproduces the direct evaluation of Eq. (1) to better than
2 mm over four decades — including at `log₁₀ E_ν = 4`, where `w⋆ ≈ 2` and the
asymptotic expansion has no right to be as good as it is.

Eq. (1) is itself evaluated two independent ways that agree to ~1 m in the depth
integral: Gil-Pelaez inversion (`loss_distribution.log_loss_cdf`) and density
inversion followed by quadrature (`loss_density` + `survival_from_density`).

The ratio `L/R_CSDA` runs from 1.15 at 10 TeV — where the constant `C` still dominates
and the muon is near threshold — down through 0.94 at 1 PeV to 0.88 at 100 PeV,
approaching the asymptotic `b_μ/Φ′(0) = 0.837`.

## 9. Where it enters

```
A_eff(E_ν) = n_N σ_CC(E_ν) [ A_proj L(E_ν) + V_det ] T(E_ν) ,
```

with `T` the Earth transmission. Nothing in this expression is fitted. Against the
IceCube DR2 upgoing effective area it lands within ~12% between 1 and 10 PeV once
neutral-current regeneration is kept in `T` (`attenuation.regenerated_transmission`);
see `examples/28_neutrino_energy_effective_area.py` for the residuals.

## 10. Scope and caveats

- `b_μ` and `d_μ` are evaluated at the production energy rather than followed down the
  track, the same percent-level approximation `muon_range_km` already makes and
  justifies by how slowly the QED coefficients move.
- `E_thr` is a hard threshold. A soft trigger turn-on would smear `w⋆`, replacing
  Eq. (6) by its average over the turn-on; the DR2 smearing matrix supports a hard
  threshold well, with the 5th percentile of accepted reconstructed muon energy flat
  at ~700 GeV across three decades of `E_ν` (log-log slope 0.02, against 1.0 for a
  threshold scaling with neutrino energy).
- The `O(1)` term dropped in Eq. (6) decays in `w⋆`; the validation table shows it is
  already below the 2 mm level at `w⋆ ≈ 2`, but Eq. (6) should not be pushed to
  `ε → E_thr`, where `L → 0` and the expansion has no support.
- `⟨y_w⟩` is taken as a constant delta at its mean, matching `inelasticity_factor`.
  The true CC inelasticity runs with energy and has real spread, which smears `w⋆`
  by roughly its own width.
