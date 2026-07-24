# Soft-volume forward model — physics notes

Distilled reference for implementing the `transport/` → `response/` soft-volume
path in `softpaws`. Source: Palmisano, Redigolo, Tammaro, Tesi,
*The soft volume of ultra-high energy neutrinos experiments*
([arXiv:2607.13143](https://arxiv.org/abs/2607.13143), companion of 2507.10665).
Equation numbers below refer to that paper. Local copy: `docs/2607.13143.pdf`.

---

## 1. The one-sentence idea

A muon-neutrino CC interaction makes a muon that can travel ~km before dying.
So for **track events** the effective target is not the instrumented volume
`V_det` but a larger **soft volume** `V_soft` set by the muon's energy-loss
range. The paper builds a fast, semi-analytic map

```
neutrino flux  ϕ_ν(E_ν, Ω)   →   muon track rate  dN/dE dΩ
```

by solving a Boltzmann transport equation for the muon distribution, reduced to
a **drift–diffusion (Fokker–Planck) equation in energy** via a soft expansion of
the QED collision operator.

Two contributions to every track bin (Eq. 1.1, 2.12):

```
⟨N⟩ ∝ V_det              (muon born inside detector)
     + A_proj · (muon QED range)    (muon born outside, drifts in)  ← soft volume
```

with `A_proj = π R_det²` the projected area of a spherical detector.

---

## 2. Governing equations

### 2.1 Stationary Boltzmann transport (Eq. 2.1, 2.9)

Steady state (`∂_t f = 0`, source/detector vary slower than R⊕/c ≈ 0.01 s),
ultra-relativistic (`v ≈ c = 1`), collinear (muon direction = parent-ν
direction, since `Δθ ∝ m_μ/E ≪ 1`). The muon flux per unit energy
`ϕ_μ(E, x)` along a fixed line of sight (x = column depth) obeys:

```
∂ϕ_μ/∂x  −  ∂/∂E[ b_μ(E) E ϕ_μ ]  −  ½ ∂²/∂E²[ d_μ(E) E² ϕ_μ ]  =  (E²/(2π)³) C_weak     (2.9)
```

- **Drift term** `b_μ(E)` — mean fractional energy loss per unit length. Muon range ≈ `1/b_μ`.
- **Diffusion term** `d_μ(E)` — stochastic spread of energy loss (variance).
- **Source** `C_weak` — neutrino CC production of muons (Section 2, below).
- `x` has units of length (column depth); coefficients are per-length (km⁻¹).

### 2.2 Transport coefficients (Eq. 2.8)

First two moments of the QED energy-loss spectrum `dΓ/dy = n_T dσ/dy`,
`y ≡ ΔE/E` the fractional energy loss:

```
b_μ(E) = ∫ dy · y   · dΓ(E,y)/dy          (drift  = 1st moment)
d_μ(E) = ∫ dy · y²  · dΓ(E,y)/dy          (diffusion = 2nd moment)
```

Three radiative processes contribute (Section 3, Fig. 3–4):
- **e⁺e⁻ pair production (PP)** — softest (`dσ/dy ~ (1−y)/y`), dominates `b_μ`, expansion-friendly. `y_min = 4 m_e/E`.
- **Bremsstrahlung (B)** — `dσ/dy ~ 1/y`, hard tail, IR-divergent (needs `y_min` cutoff).
- **Photonuclear (PN)** — `dσ/dy ~ 1/y`, hard tail, dominant theory uncertainty (~30%). `y_min ≈ m_π/E`.

Reference values in **water** (ρ = 1.02 g/cm³), Table 1:

| E_μ | b_μ [km⁻¹] | d_μ [km⁻¹] | d_μ/b_μ |
|-----|-----------|-----------|---------|
| 1 PeV | 0.35 | 0.077 | ~0.22 |
| 100 PeV | 0.40 | 0.098 | ~0.25 |

Per-process at 1 PeV: PP (0.17, 0.004), B (0.091, 0.043), PN (0.088, 0.030).
Compounds add incoherently weighted by atoms: `b_water = 2 b_H + b_O`.
Coefficients depend only *logarithmically* on E — treated as constant in the
baseline solution, energy-dependent version in Appendix A.1.

### 2.3 Green's function / propagation kernel (Eq. 2.16, A.16)

Solving 2.9 for an impulse source `δ(x−ξ)δ(E−ε)` with constant coefficients.
In log-energy `u = log E`, `ψ_μ = E ϕ_μ`, it's plain drift-diffusion (Eq. A.14):

```
∂_x ψ = M_μ ∂_u ψ + (d_μ/2) ∂²_u ψ ,     M_μ ≡ b_μ + d_μ/2
```

Gaussian kernel (a **log-normal in E**), for propagation length ℓ = x − ξ > 0:

```
G(E,x; ε,ξ) = θ(x−ξ) θ(ε−E) · 1/(E √(2π d_μ ℓ))
              · exp[ −( log(ε/E) − (b_μ + d_μ/2) ℓ )² / (2 d_μ ℓ) ]         (2.16)
```

Interpretation: probability that a muon born at ξ with energy ε arrives at x
with energy E. `θ(ε−E)` enforces energy is only lost.

**Drift limit** `d_μ → 0` (Eq. 2.17): kernel collapses to a delta on the
deterministic continuous-slowing-down trajectory `E(x) = E exp[b_μ(x−ξ)]`:

```
G → (ε/E) δ(ε − E e^{b_μ(x−ξ)}) θ(x−ξ)
```

Good below ~300 GeV (ionization/Bethe–Bloch regime) and as a leading-order
approximation everywhere.

---

## 3. The weak source term (muon production)

Muon born from `ν_μ` CC DIS with `ε = (1−y_w) E_ν`, `y_w` = DIS inelasticity
(`⟨y_w⟩ ≈ 0.2`, so near-elastic; using the mean vs full `P(y_w)` is a ~1% effect).
Rewritten (Eq. 2.19):

```
(ε²/(2π)³) C_weak = n_N(ξ) ∫₀¹ dy_w [ P(y_w)/(1−y_w) ] D_ν(ε_{y_w}, ξ, Ω) σ_CC(ε_{y_w}) ϕ_ν^⊕(ε_{y_w})
```

with `ε_{y_w} = ε/(1−y_w)`, `P(y_w) = σ⁻¹ dσ/dy_w` the inelasticity distribution.

**Neutrino CC cross section** (Eq. 2.5) — power law above 10 PeV:
```
σ_CC(E_ν) = σ₀ (E_ν/E₀)^λ ,   σ₀ = 1.48e-33 cm² at E₀ = 10 PeV,   λ ≈ 0.4
```
(from small-x PDF behavior; below 10 PeV use MadGraph/tabulated).

**Attenuation** `D_ν` (Eq. 2.4) — Earth absorption of neutrinos before the
interaction point:
```
D_ν ≈ exp[ −n_N σ_νN(E_ν) L(x; Ω) ]
```
`L` = column depth along the line of sight, from PREM Earth model (concentric
constant-density shells; depends only on incoming polar/zenith angle).

---

## 4. The master formula (Eq. 2.13, 2.20)

Full differential track rate = inside term (∝ V_det) + outside/soft term (∝ A_proj):

```
dN/(dt dE dΩ) = V_det · n_N · ∫ dE_ν ϕ_ν(E_ν,Ω)|det · dσ_CC/dE           (inside)
              + A_proj · ∫₀ˣ dξ ∫_E^∞ dε · G(E,x; ε,ξ) · (ε²/(2π)³) C_weak(ξ,ε)   (soft)
```

- Green's function evaluated at detector surface `x = L(Ω_ν) − R_det`.
- Integration limits: Earth surface `ξ = 0` → detector surface `ξ = x`.
- Not analytic in general → **numerical**, but cheap (seconds for many bins).
- Efficiency/acceptance folded in as `V_det → ε_i V_det`, `A_proj → ε_i A_proj`
  (or an energy/direction-dependent response `ε(E_μ, Ω)`).

**Soft volume definition** (Eq. 2.14):
```
V_soft(E) · [n_N σ_νN(E) ϕ_ν^⊕(E)] = A_proj ϕ_μ(E,Ω)|det
```

---

## 5. Analytic soft-volume estimates (Section 2.3) — sanity checks

With simplifying assumptions (spherical perfectly-absorbing detector; `σ_CC ∝ E^λ`,
λ≈0.4; single power-law flux `ϕ_ν ∝ E^{-γ}`; step-function attenuation — up-going
absorbed, down-going free; constant near-detector density `n_N`), the source
reduces to `C_weak = n_N σ_νN ϕ_ν (ε/E)^{λ−γ}` (Eq. 2.21) and:

**Drift limit** (Eq. 2.23), with `A ≡ γ − λ − 1 > 0`:
```
V_soft|drift = A_proj / (b_μ A) = π R_det² / (b_μ (γ − λ − 1))
```
Range `1/b_μ` enhanced further by the spectral penalty `1/A`.

**Figure of merit** (Eq. 2.24), for IceCube's flux (A ≈ 1):
```
V_tot / V_det ≈ 1 + (3/4) / (R_det b_μ)
```
IceCube (R_det ≈ 0.62 km, b_μ ≈ 0.35/km @ 1 PeV) → **V_soft ≈ 4× V_det**.
KM3NeT (smaller) → ~7.5×.

**Diffusion correction** (Eq. 2.25): negative, `V_soft|diff ≈ V_soft|drift (1 − d_μ/2b_μ)`,
~10% shift. Small — diffusion barely moves event counts even with O(1) `d_μ` error.

Energy-dependent coefficients (Appendix A.1): replace `1/b_μ` by a
spectrum-weighted average of `1/b_μ(u)`; interpretation unchanged.

---

## 6. Calibrating the coefficients (Section 3.1)

`b_μ, d_μ` are given priors from a **once-and-for-all** MC of muon propagation
(step by exponential MFP, sample `y` from `dΓ/dy`, stop at `E ≤ 0.1 E₀`,
`E₀ = 10 PeV`). Two fit models:

- **Drift model:** range–energy relation `⟨R⟩ = −(1/b_μ) log(E_f/E₀)` (Eq. 3.2).
  Result `b_μ^MC/b_μ = 1.03 (+0.50/−0.22)`.
- **Drift-diffusion:** log-normal in `z = log(E/ε)`, `G(z|x) ∝ exp[−(z + M_μ x)²/(2 d_μ x)]`
  (Eq. 3.5). Fit via first two moments of the range `R` (Eq. 3.8–3.9):
  ```
  ⟨R⟩ = z/M_μ + d/M_μ²
  ⟨R²⟩ − ⟨R⟩² = z d/M_μ³ + 2 d²/M_μ⁴
  ```
  Result `b_μ^MC/b_μ = 0.94 ± 0.15`, `d_μ^MC/d_μ = 1.5 (+1.6/−0.8)`.

`d_μ` is poorly constrained (probes the hard tail) but that's OK — it only enters
the soft volume as `d_μ/2b_μ`.

For our purposes we can start by adopting the **Table 1 central values** and treat
the fit spread as a systematic band; the full MC calibration is optional.

---

## 7. Approximations & caveats (read before trusting a number)

1. **Idealized detector.** Perfectly absorbing sphere, unit efficiency, every
   crossing muon counted. No event selection, angular acceptance, reconstruction,
   or energy cuts. Prediction = *geometric through-going rate*, not detector-level.
   Empirically for IceCube through-going: `ε_IC-TG ≈ 0.45` (≈ the θ_zenith>85° cut).
2. **Soft expansion is only marginally convergent.** Global `d_μ/b_μ ~ 0.2`;
   truncating Kramers–Moyal at 2nd order (Pawula's theorem) is not guaranteed
   positivity-preserving. Hard B/PN tails (`dσ/dy ~ 1/y`, `y ~ 1`) can break it.
   Expect ~tens-of-% error, dominated by diffusion. Hard scatters can be added
   perturbatively (Appendix B), giving ~25% floor — comparable to QED theory error.
3. **Collinear / no angular deflection.** Muon inherits ν direction. Fine at UHE.
4. **Constant, homogeneous medium** near detector (ice/water). Earth profile via
   PREM only affects neutrino attenuation, not muon propagation.
5. **Coefficients ~constant in E** (log-slow). Fails for PN at very high E → use A.1.
6. **`θ(ε−E)` support** is imposed by hand; small Gaussian tail at ε<E is an
   artifact of 2nd-order FP.
7. **Taus negligible** (~5% of track sample; Section 2.4) — ignore for now.
8. **Fully-differential vs inclusive:** retaining the E_ν↔E_μ,production↔E_μ,det
   mapping matters little for steep power laws (parents with E_ν ≫ E_μ suppressed),
   but is the whole point for peaked/exotic (e.g. DM) fluxes.

Benchmark fit result to aim at reproducing (IceCube 9.5 yr, diffusion model, Eq. 1.3):
`γ = 2.38 (+0.11/−0.09)`, `ϕ₀ = 0.63 (+0.14/−0.13) ×10⁻¹⁸ GeV⁻¹ s⁻¹ cm⁻² sr⁻¹`.

---

## 8. Numerical recipe → mapping to `softpaws`

Data flows `data/ → transport/ → response/ → comparison/`. Proposed pieces:

**`utils/`** — constants (m_e, m_μ, m_π, N_A), unit conversions (GeV↔cm↔km,
densities), PREM shells + `L(zenith)` column-depth helper (Eq. footnote p.11).

**`transport/`** — the physics core:
- `coefficients`: `b_μ(E)`, `d_μ(E)` — start from Table 1 constants (water/ice),
  interface to swap in energy-dependent / MC-calibrated values later.
- `green`: `G(E, x; ε, ξ)` log-normal kernel (Eq. 2.16) + drift-limit delta (2.17);
  optional slow-coefficient version (A.24).
- `source`: `C_weak` (Eq. 2.19) — needs `σ_CC(E_ν)` (Eq. 2.5), `P(y_w)` (or ⟨y_w⟩),
  `D_ν` attenuation (Eq. 2.4).
- `flux`: convolve G with source → `ϕ_μ(E, Ω)|det` (Eq. 2.15).
- `soft_volume`: `V_soft(E)` (Eq. 2.14/2.22); analytic drift/diffusion forms
  (2.23/2.25) as fast path + validation.

**`response/`** — `SoftVolumeResponse` alongside the existing IRF path in
`response/irfs.py`, both exposing the same `flux → dN/dE dΩ` interface (master
formula Eq. 2.20). This interchangeability is what enables the head-to-head.

**`comparison/`** — put the two predictions and the DR2 events side by side;
reproduce the power-law fit (Eq. 4.1–4.2, Poisson likelihood over ≥10 TeV bins),
extract `ε_IC-TG`.

**Suggested build order:** utils constants → drift-limit V_soft (closed form,
Eq. 2.23, validate against the "4×" figure of merit) → full Green's-function
convolution (Eq. 2.20) → wrap as `response/` predictor → comparison fit. Add MC
coefficient calibration (Section 3.1) and hard-scatter correction (Appendix B)
only if needed for accuracy.
