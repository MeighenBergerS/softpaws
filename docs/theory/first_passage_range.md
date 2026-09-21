# The monochromatic range as a first-passage problem

Working notes behind the method paper's Sec. II E, "The Range to Threshold", and its
App. A.2 of the same name. Everything below is exact under the same local scale-invariance
assumption (`eq:scaleinv`, Sec. II A of the method paper) that the rest of the formalism
rests on; nothing here is fitted. Eq. (1) below is the method paper's `eq:Ldef`, Eq. (5)
is `eq:potential`, the two moments of §6 are `eq:Phimoments`, and Eq. (6) is `eq:Lclosed`.

Implemented in `softpaws.transport.muon_range.stochastic_muon_range_km`, exercised by
`scripts/2026_muon_transport/28_neutrino_energy_effective_area.py`, tested in
`tests/test_nu_energy_response.py`.

---

## 1. Why a second length is needed

The track-rate equation of the method paper (`eq:rate`, Sec. III A) gives the track rate
differential in the **observed muon energy**, and its
length is the spectrally-weighted `1/Φ(A)` — about 2.7 km in water at IceCube's
diffuse spectral index. That is the right length for a power-law parent, because the source
excites the single mode `s = A` and the propagator contributes `e^{-ℓΦ(A)}`.

A published effective area is a different object. It is tabulated at fixed **neutrino
energy** and integrated over every muon energy that survives the selection, so the
parent is *monochromatic* and there is no spectral index left to weight against. The
two conventions are not interchangeable: the IceCube DR2 smearing matrix puts the
median reconstructed muon energy 1.6 decades below `E_ν` at 100 TeV and 3.9 decades
below at 10 EeV, and the offset grows monotonically with energy.

For a monochromatic parent the formalism is already in hand — it is the `s → 0` case
of the method paper's App. A, "The Transport", where `Φ(0) = 0` and `I(0) = 1`. Both spectral factors switch off, so the
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
whose Laplace exponent is exactly the collision eigenvalue of the method paper
(`eq:Phi`, Sec. II A):

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

The reason is Jensen, and it is the same "mean versus typical" statement that
`scripts/2026_muon_transport/27_proposal_cross_section_and_loss.py` already makes in
its "3a. Mean vs typical" report: CSDA propagates
the **logarithm of the mean energy**, `ln E[E(ℓ)] = -b_μ ℓ`, whereas what controls
threshold crossing is the **mean of the logarithm**, `E[ln E(ℓ)] = -Φ′(0) ℓ`. Since
`E[ln E] ≤ ln E[E]`, the typical muon degrades faster than the mean one, and a
threshold is crossed sooner than the CSDA range suggests. The mean is held up by rare
muons that happened to radiate little; those are not typical, and a range is a typical
quantity.

## 8. Validation

Muon in water, `E_thr = 1 TeV`, the shipped PROPOSAL kernel table, as the
library computes them today (`make_transport_table.py` in the scripts directory
writes the same rows into the paper):

| log₁₀(E_μ/GeV) | `b_μ = ⟨y⟩` | `Φ′(0) = ⟨-ln(1-y)⟩` | `b_μ/Φ′(0)` | `C = -Φ″(0)/2Φ′(0)²` | `L`, running | `L`, frozen | `R_CSDA` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4.0 | 0.3424 | 0.4283 | 0.800 | 0.798 |  5.73 |  5.53 |  5.73 |
| 5.0 | 0.3626 | 0.4610 | 0.787 | 0.811 | 10.76 | 10.36 | 12.10 |
| 6.0 | 0.3797 | 0.4869 | 0.780 | 0.800 | 15.62 | 14.78 | 18.30 |
| 7.0 | 0.4000 | 0.5164 | 0.775 | 0.772 | 20.21 | 18.64 | 24.21 |
| 8.0 | 0.4265 | 0.5549 | 0.769 | 0.736 | 24.52 | 21.80 | 29.79 |

Lengths in km of water. The frozen column evaluates Eq. (6) with the
coefficients held at the production energy, which is the form derived above;
the running column follows them down the track, as App. A.2 of the method
paper does, and is the library default. Both splice on to a deterministic
ionization segment below 10 TeV, which is why the 10 TeV row meets `R_CSDA`.

Eq. (1) is itself evaluated two independent ways that agree to ~1 m in the depth
integral: Gil-Pelaez inversion (`softpaws.transport.loss_distribution.log_loss_cdf`)
and density inversion followed by quadrature (`loss_density` and then
`survival_from_density`, both in `softpaws.transport.loss_distribution`).

The ratio `L/R_CSDA` runs from unity at 10 TeV, where the ionization segment
carries the range, through 0.85 at 1 PeV to 0.82 at 100 PeV, approaching the
asymptotic `b_μ/Φ′(0) = 0.77`. Fluctuations can only remove range, so it never
exceeds unity.

## 9. Where it enters

```
A_eff(E_ν) = n_N σ_CC(E_ν) [ A_proj L(E_ν) + V_det ] T(E_ν) ,
```

with `T` the Earth transmission. Nothing in this expression is fitted.

Against the IceCube DR2 upgoing effective area, with neutral-current regeneration
kept in `T` (`softpaws.transport.attenuation.regenerated_transmission`) and the
`ν_τ → τ → μ` channel added (`softpaws.transport.attenuation.flavour_transmission`,
`flavour="tau"`), the residual over
10⁵–10⁷·⁸ GeV is 0.044 dex rms with a +0.15 dex trend across the band. The implied
selection efficiency `A_eff^IC / A_eff^model` then runs 0.56 → 0.82 and stays below
one everywhere, as an efficiency must — which it does not without those two terms.
See `scripts/2026_muon_transport/28_neutrino_energy_effective_area.py`. The top decade
is excluded
from the statistics: the simulation behind the DR2 tables stops at 100 PeV.

Note that the DR2 tables are *muon-neutrino* effective areas, so the `ν_τ` channel
is a model-side addition rather than a like-for-like term, and it assumes
`φ_ντ = φ_νμ` at Earth. Script 28 keeps it as a separate curve for that reason.

## 10. Scope and caveats

- Eq. (6) evaluates `b_μ` and `d_μ` at the production energy and does not follow them
  down the track, which shortens the range by 4% at 100 TeV and 11% at 100 PeV
  against the running form (App. A.2 of the method paper). In code this is the `kernel_evaluation="frozen"` option of
  `softpaws.transport.muon_range.stochastic_muon_range_km` and of
  `softpaws.transport.muon_range.muon_range_km`. Both default to `"running"`, which
  reads the kernel along the descent as in App. A.2 of the method paper.
- `E_thr` is a hard threshold. A soft trigger turn-on would smear `w⋆`, replacing
  Eq. (6) by its average over the turn-on; the DR2 smearing matrix supports a hard
  threshold well, with the 5th percentile of accepted reconstructed muon energy flat
  at ~700 GeV across three decades of `E_ν` (log-log slope 0.02, against 1.0 for a
  threshold scaling with neutrino energy).
- The `O(1)` term dropped in Eq. (6) decays in `w⋆`; the validation table shows it is
  already below the 2 mm level at `w⋆ ≈ 2`, but Eq. (6) should not be pushed to
  `ε → E_thr`, where `L → 0` and the expansion has no support.
- `⟨y_w⟩` is taken as a constant delta at its mean, matching
  `softpaws.transport.source.inelasticity_factor`.
  The true CC inelasticity runs with energy and has real spread, which smears `w⋆`
  by roughly its own width.

## 11. The loss law along the descent

Sections 3 to 6 hold the kernel fixed at the production energy, and Sec. 10
notes that the *range* is then short by up to 11% at 100 PeV. For the *law*
`P(w)` of a single track the same freezing is far worse once the muon falls
through decades where the drift changes: from `10^14` GeV the photonuclear
drift is twice its `10^10` GeV value, so a kernel read at the top of the
descent overstates the mean log-loss over 34 km of water by a factor of 2.5.

The subordinator of Sec. 3 generalizes to an additive process whose exponent
accumulates along the mean descent `dlnE/dl = −Φ′(0; E)`,

```
Ψ(s; ℓ) = ∫_0^ℓ dl Φ(s; E(l)) = ∫_0^{v(ℓ)} dv Φ(s; ε e^{−v}) / Φ′(0; ε e^{−v}),
```

with `v(ℓ)` the mean log-loss at depth `ℓ`, so that
`E[e^{−sW(ℓ)}] = e^{−Ψ(s; ℓ)}` and `P(w)` follows by the same inversion as
before. The kernel at each energy is the three-moment family of Sec. 8 read
from the shipped table, and `Φ′(0; E)` is that family's own mean rate, so the
mean of the resulting law is exactly `v(ℓ)`. Below the ionization matching
energy of Sec. 10 the kernel is held fixed, as the range does.

In code this is `softpaws.transport.loss_distribution.running_log_loss_symbol`
for the exponent and `loss_density_running` for the law; the frozen forms
`loss_density_three_moment` and `loss_density` remain for a kernel read once.
