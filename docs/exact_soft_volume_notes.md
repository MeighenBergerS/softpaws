# Exact soft-volume forward model — physics notes

Distilled reference for the **exact** soft-volume solution derived in
`docs/2026_softvolume.pdf`. This is a first-principles re-derivation that
replaces the paper's Fokker–Planck (drift–diffusion) expansion with an *exact*
eigenvalue treatment of the QED collision operator. It is the intended physics
core for the `transport/` → `response/` path.

Read alongside [`soft_volume_notes.md`](soft_volume_notes.md), which distills
the **paper's** approach (Palmisano et al.,
[arXiv:2607.13143](https://arxiv.org/abs/2607.13143)). The paper's Fokker–Planck
result is recovered here as the first two terms of an exact series — see
[§8, Contrast](#8-contrast-with-the-paper-and-with-soft_volume_notesmd).

Part numbers below refer to `docs/2026_softvolume.pdf`.

---

## 1. The one-sentence idea

The muon energy-loss operator is scale-invariant (rates depend on the fractional
loss `y`, not on energy `E`). Scale-invariant operators are **diagonalised by
power laws**, and the astrophysical source *is* a power law. So the entire
integro-differential transport problem collapses to **multiplication by a single
number** `Φ(A)` — no propagator, no kernel, no convolution, and no energy
cutoffs. (Part 0.)

Where the paper expands the collision operator in small `y` to get a differential
Fokker–Planck equation, this treatment keeps the collision operator exact and
diagonalises it instead.

---

## 2. The transport equation and the exact collision operator (Parts 1–2)

Same starting point as the paper: steady-state (`∂_t f = 0`), ultra-relativistic
(`v = c = 1`), collinear (`Δθ ~ m_μ/E ≪ 1`, so `Ω` is a spectator label). The
muon flux per unit energy `ϕ_μ(x, E)` along a fixed line of sight (`x` = column
depth [km w.e.]) obeys

```
∂ϕ_μ(x,E)/∂x  =  C_QED[E; ϕ_μ]  +  S(x,E)
```

The QED collision operator is kept as the **exact integral operator** (Part 1.4),
not expanded:

```
C_QED[E; ϕ_μ] = − ϕ_μ(E) ∫₀¹ dy dΓ(E,y)/dy                (loss)
              + ∫₀¹ dy/(1−y) · dΓ(E_y,y)/dy · ϕ_μ(E_y)     (gain),   E_y ≡ E/(1−y)
```

with `dΓ/dy = n_T dσ/dy` the differential rate per unit length to lose fraction
`y`. The `1/(1−y)` is the Jacobian from `E = (1−y)E'`. The gain term is evaluated
at the *incoming* energy `E_y = E/(1−y)` (a muon arriving at `E` after losing 90%
was radiating at `10E`). Its non-locality — `ϕ_μ` inside the integral at a
different energy — is the entire difficulty, and is exactly what the eigenvalue
trick avoids.

### 2.1 Power laws are eigenfunctions (Assumption 4: scale-invariant rates)

Substituting `ϕ_μ(E) = E^{−1−s}` and using `ϕ_μ(E_y) = (1−y)^{1+s} ϕ_μ(E)`, the
`1/(1−y)` Jacobian cancels one power of `(1−y)` and the operator becomes diagonal:

```
C_QED[E; E^{−1−s}] = − Φ(s) · E^{−1−s}
```

with the **exact eigenvalue function** (the central object — every physical
statement below is about this one function):

```
        ┌────────────────────────────────────────┐
        │  Φ(s) = ∫₀¹ dy (dΓ/dy) [ 1 − (1−y)^s ]  │
        └────────────────────────────────────────┘
```

The bracket `[1 − (1−y)^s]` is the net effect of a collision of severity `y` on a
population of spectral index `s`: `1` is the muon leaving `E`, `(1−y)^s` is the
compensating arrival from above. Hard spectrum (small `s`) → nearly complete
compensation → small `Φ`. Steep spectrum (large `s`) → little replenishment →
large `Φ`.

**Finite with no cutoffs** (this is the key practical win over the paper):
- *IR-finite.* Bremsstrahlung `dΓ/dy ~ κ/y` diverges at `y→0`, but
  `1 − (1−y)^s → sy`, so the integrand → `sκ`, finite. The loss-term divergence
  is cancelled by the gain term (an infinitely soft photon does nothing).
  **No `y_min = 10⁻⁷` needed.**
- *UV-finite.* As `y→1`, `(1−y)^s → 0`, bracket → 1: a catastrophic collision
  simply removes the muon. Hard brems/photonuclear tails are integrated exactly.
  **No `y_cut` machinery (paper's Appendices A/B) needed.**

### 2.2 Mellin transform (formal statement, Part 3)

Define `ϕ̂(x,s) ≡ ∫₀^∞ dE E^s ϕ_μ(x,E)`. Then `Ĉ_QED(s) = −Φ(s) ϕ̂(s)`:
multiplication, not convolution. Structurally `Φ(s)` is a Mellin symbol /
anomalous dimension — the same object as a moment of a DGLAP splitting function.
The Green's function is `e^{−Φ(s)ℓ}`, one number per mode; because a single
power-law source excites exactly one `s`, you never need the inverse transform.

---

## 3. Solving the transport equation (Part 4)

Mellin-transforming the PDE turns the integro-differential equation into a linear
first-order **ODE in depth**, mode by mode:

```
∂ϕ̂(x,s)/∂x = −Φ(s) ϕ̂(x,s) + Ŝ(x,s)
```

- **First order in `x` ⇒ initial-value problem in depth** (depth plays the role
  of time). The *only* boundary condition allowed is `ϕ_μ(x=0, E) = 0` (no muons
  enter at the Earth's surface, Assumption 5). You cannot additionally impose a
  condition at the detector without over-determining the system.
- **This justifies the paper's "free flux at the surface" shortcut.** A perfectly
  absorbing detector at `x_det` affects only `x > x_det`, which nobody observes,
  so the detector never back-reacts on the flux arriving at it. Not an
  approximation — forced by the hyperbolic structure.
- (Atmospheric muons violate Assumption 5 for down-going directions; handled as a
  separately measured background, not through this equation.)

Integrating factor `e^{Φ(s)x}` gives the exact solution:

```
ϕ̂(x,s) = ∫₀ˣ dξ e^{−Φ(s)(x−ξ)} Ŝ(ξ,s)
```

Muons produced at depth `ξ` propagate `ℓ = x − ξ` and are attenuated by
`e^{−Φ(s)ℓ}` — attenuation of the *spectral-mode amplitude*, not of muon number
(QED collisions don't destroy muons). **Exact so far: no expansion in `y`, no
Fokker–Planck, no truncation.**

---

## 4. The source term (Part 5)

Built from scratch, same physics as the paper but carried exactly. Muon born from
`ν_μ` CC DIS with `ε = (1−y_w) E_ν`, `y_w` the weak inelasticity
(`⟨y_w⟩ ≈ 0.2`), `P(y_w) = σ⁻¹ dσ/dy_w`:

```
S(ε) = n_N ∫₀¹ dy_w/(1−y_w) · P(y_w) · [ϕ_ν σ_νN](ε/(1−y_w))
```

Same `1/(1−y)` Jacobian structure as the QED gain term, same reason. With
power-law flux `ϕ_ν ∝ E_ν^{−γ}` (Assumption 6) and power-law cross section
`σ_νN ∝ E_ν^λ`, `λ ≈ 0.4` (Assumption 7), and step-function attenuation +
constant near-detector density (Assumption 8), the product is a single power
`ϕ_ν σ_νN ∝ E_ν^{λ−γ}`, and the `y_w` integral **factorises off completely** into
a pure number. This defines the two key quantities:

```
        ┌──────────────────────┐   ┌──────────────────────────────┐
        │  A ≡ γ − λ − 1        │   │  I(A) ≡ ⟨(1−y_w)^A⟩_P ≃ 0.8   │
        └──────────────────────┘   └──────────────────────────────┘

  S(ε) = n_N I(A) σ_νN(ε) ϕ_ν^⊕(ε)  ∝  ε^{λ−γ} = ε^{−1−A}
```

- **`A` is not ad hoc.** It is flux steepness `γ`, minus cross-section growth `λ`,
  minus one for the Jacobian. It measures how sharply the observable-weighted
  population falls with energy.
- **The source is exactly the eigenfunction with `s = A`.** This is the whole
  game — skip the inverse Mellin transform entirely.
- **`I(A)` is a genuine normalisation factor** the paper mostly drops. `I(1) =
  1 − ⟨y_w⟩ ≃ 0.8` exactly when `A ≈ 1` (no averaging approximation needed).

---

## 5. Master solution and event rate (Parts 6–8)

Because the source excites exactly one mode, the ODE solves directly in energy
space (`S(ε) = S₀ ε^{−1−A}`, `ξ`-independent by Assumption 8):

```
        ┌──────────────────────────────────────┐
        │  ϕ_μ(x,E) = S(E) · (1 − e^{−Φ(A)x})/Φ(A)  │
        └──────────────────────────────────────┘
```

Sanity checks: `x→0` → `S(E)x` (thin slab, linear, none lost); `x→∞` →
`S(E)/Φ(A)` (equilibrium); `Φ→0` → `S(E)x` for all `x` (no losses, whole column
accumulates — **finite**, remember this one).

**Soft volume** (Part 7) — the target volume that, producing muons with no
propagation losses, would give the observed arrival rate:

```
  V_soft(E) = A_proj · I(A)/Φ(A) · (1 − e^{−Φ(A)x})
```

Three factors: `A_proj/Φ(A)` = area × effective range `1/Φ(A)` (range weighted by
spectral replenishment, not the CSDA range); `I(A) ≃ 0.8`; and the **saturation
factor** `(1 − e^{−Φx})` = finite upstream column.

**Master formula** (Part 8) — two disjoint populations: produced inside
(`V_det = 4/3 π R_det³`) and produced outside, arriving through the projected
surface (`A_proj = π R_det²` for a sphere, plane-parallel Assumption 10). `I(A)`
multiplies both (same production vertex):

```
        ┌─────────────────────────────────────────────────────────────────────┐
        │  dN/(dt dE dΩ) = I(A) n_N σ_νN(E) ϕ_ν^⊕(E) · [ V_det                  │
        │                     + A_proj/Φ(A) · (1 − e^{−Φ(A)x}) ]                │
        └─────────────────────────────────────────────────────────────────────┘
```

With `R_det ≃ 0.62 km` and `1/Φ ≃ 2.9 km`, the soft term is ~4× the instrumented
volume — the paper's central claim, now with the corrected coefficient (`I(A)`
and exact `Φ(A)`).

---

## 6. The saturation factor `(1 − e^{−Φx})` — why it matters (Parts 7.4, 11.2)

Assumption 9 is a perfect absorber (unit efficiency, no cuts/acceptance/threshold
— the prediction is a geometric through-going rate). Two reasons to keep the full
saturation factor instead of the paper's `x→∞` idealisation:

**(a) Finite depth.** With `Φ ≃ 0.35 km⁻¹`:

| x [km]        | 1    | 2    | 3    | 5    | 10   |
|---------------|------|------|------|------|------|
| `1 − e^{−Φx}` | 0.30 | 0.50 | 0.65 | 0.82 | 0.97 |

IceCube sits ~1.95 km deep. For near-vertical down-going muons the available
column is only that deep, so `x→∞` **overestimates the rate by ~2×**. Their
`θ_zenith > 85°` cut removes this region (so their diffuse fit is unaffected), but
it matters for any all-sky or point-source application.

**(b) It cures divergences.** As `Φ→0`, `(1−e^{−Φx})/Φ → x`, finite — the `1/Φ`
pole at `A→0` is an artifact of pretending the upstream medium is infinite.
Keeping `x` removes it. For `Φ < 0` (see §7) it continues analytically to a
finite, positive value:

```
  (1 − e^{−Φx})/Φ  ──Φ<0──→  (e^{|Φ|x} − 1)/|Φ|
```

---

## 7. The cross-section pole (Part 11.2)

The denominator `Φ(A)` crosses zero when `λ > γ − 1` (i.e. `A < 0`). For
`γ = 2.38` the **pole sits at `λ = 1.38`**. Physically, when the cross section
grows faster than the flux falls, more distant / higher-energy parent neutrinos
dominate and the infinite-column integral genuinely diverges (`(1−y)^A > 1`, so
`Φ(A) < 0`, the observable-weighted population is no longer attenuated). This is
not a numerical artifact — it is the signal that the `x→∞` assumption has failed.

**Consequence for the cross-section extraction:** the paper's Fig. 8 quoted modes
`λ_IC = 1.48` (1 yr) and `1.66` (9.5 yr) both sit *past* the pole, where Eq. (4.6)
returns a *negative* event count. That the distributions pile up at
`λ ~ 1.2–1.7`, straddling the pole, looks like the pole rather than the data —
worth redoing with the saturation factor before believing `R = σ/σ_SM ~ 10`.
(Caveat: possible their code holds `λ = 0.4` inside `A` while floating it
elsewhere, which would be inconsistent but avoids the sign flip.)

---

## 8. Contrast with the paper (and with `soft_volume_notes.md`)

`soft_volume_notes.md` documents the **paper's Fokker–Planck path**; this file
documents the **exact eigenvalue path**. They describe the same physics; the exact
version *contains* the paper's as a truncation.

| Aspect                | Paper / `soft_volume_notes.md` (Fokker–Planck)                  | This derivation (exact eigenvalue)                              |
|-----------------------|-----------------------------------------------------------------|-----------------------------------------------------------------|
| Collision operator    | Expanded in small `y` → drift `b_μ` + diffusion `d_μ`           | Kept exact; diagonalised by power laws                          |
| Governing solution    | Drift–diffusion PDE in energy (Eq. 2.9)                          | ODE in depth per Mellin mode; single mode `s = A`               |
| Propagator            | Log-normal Green's function `G(E,x;ε,ξ)` (Eq. 2.16), convolved   | None — multiply by `e^{−Φ(A)ℓ}`; no kernel, no convolution      |
| Attenuation constant  | `b_μ` (drift), `d_μ` (diffusion) — first two `y`-moments         | `Φ(A) = ∫ dΓ/dy [1−(1−y)^A]` — the *full* eigenvalue            |
| Energy cutoffs        | Needs `y_min` (IR) and `y_cut` (UV, Appendices A/B)              | IR- and UV-finite; **no cutoffs**                               |
| Range factor          | `V_soft ∝ 1/(b_μ A)`, with an unphysical `1/A` pole at `A→0`     | `V_soft ∝ I(A)(1−e^{−Φ(A)x})/Φ(A)`, finite everywhere           |
| Column depth          | Typically `x→∞`                                                 | Finite-`x` saturation factor `(1−e^{−Φx})` kept                 |
| Normalisation `I(A)`  | Mostly dropped (their B.5 carries `1+A`; 2.21–2.23 drop it)      | Kept: `I(A) ≃ 0.8`                                              |
| Evaluation            | Numerical Green's-function convolution                          | Closed form once `Φ(A)` is known                                |

### 8.1 The Fokker–Planck result is the first two terms (Part 10)

Binomial-expanding `1 − (1−y)^A` and integrating against `dΓ/dy` (with moments
`b_μ = ∫ y dΓ/dy`, `d_μ = ∫ y² dΓ/dy`, `t_μ = ∫ y³ dΓ/dy`):

```
  Φ(A) = A b_μ − A(A−1)/2 · d_μ + A(A−1)(A−2)/6 · t_μ − …
```

Truncating at second order gives exactly the paper's `Φ ≃ A' b_μ` with
`A' = A[1 − (d_μ/2b_μ)(A−1)]`. **The paper's Fokker–Planck result is the first
two terms of the exact series.**

### 8.2 Exactness identities — why the paper's fit survives

The series **terminates for positive integer `A`** (every term beyond the first
carries a factor `(A−1)`):

```
  Φ(1) = b_μ              (since 1 − (1−y) = y identically)
  Φ(2) = 2b_μ − d_μ       (since 1 − (1−y)² = 2y − y² identically)
```

Kramers–Moyal truncation at order `n` is exact for integer `A ≤ n`. IceCube's fit
gives `γ = 2.38`, `λ = 0.4` ⇒ **`A = 0.98`**, sitting essentially *on* the `A = 1`
exactness point. There, `Φ = b_μ` and nothing else — the entire hard tail of
brems + photonuclear (the paper's ~25% Appendix-B systematic) **cancels
identically between loss and gain**. The residual error is
`(A−1)·d_μ/2b_μ ≈ 0.02 × 0.11 ≈ 0.2%`.

So the soft expansion survives **not** because it is marginally convergent
(`d_μ/b_μ ~ 0.2`, the paper's stated reason) but because the expansion parameter
is `(A−1)d_μ/2b_μ`, and IceCube's spectrum happens to sit exactly where it
vanishes. The paper notes `A ≈ 1` twice without noticing it protects them.

### 8.3 Numeric comparison of `Φ(A)` (Part 10.4, Table)

| A    | Φ(A) [km⁻¹] | A·b_μ (drift) | A'·b_μ (Fokker–Planck) | V_FP / V_exact |
|------|-------------|---------------|------------------------|----------------|
| 0.14 | 0.058       | 0.049         | 0.054                  | 0.93           |
| 0.50 | 0.190       | 0.175         | 0.184                  | 0.97           |
| 0.98 | 0.343       | 0.342         | 0.343                  | 0.999          |
| 1.00 | 0.349       | 0.349         | 0.349                  | 1.000          |
| 2.00 | 0.621       | 0.698         | 0.621                  | 1.000          |
| 3.00 | 0.860       | 1.047         | 0.817                  | 0.95           |
| 4.00 | 1.077       | 1.396         | 0.936                  | 0.87           |
| 6.00 | 1.532       | 2.094         | 0.945                  | 0.62           |

Fokker–Planck agrees to <1% near `A = 1–2` but degrades badly for `A ≳ 3`, and its
`A'` (hence `V_soft`) changes sign at `A > 1 + 2b_μ/d_μ ≈ 10.1`. The exact `Φ`
grows monotonically forever, roughly `b_PP·A + b_B·ln A` (brems saturates
logarithmically — a `1/y` spectrum with catastrophic losses can only remove a
muon once).

---

## 9. Evaluating `Φ(A)` in practice (Part 10.4)

Table 1 gives only `b_μ` and `d_μ`, but `Φ(A)` needs the full shape of `dΓ/dy`.
Two options:

**(a) Two-parameter family** calibrated to the two moments. With
`dΓ/dy = κ (1−y)^p / y`:

```
  b = κ/(p+1),   d = κ/[(p+1)(p+2)]   ⇒   d/b = 1/(p+2)
  Φ(A) = κ [ ψ(p+A+1) − ψ(p+1) ]       (ψ = digamma; closed form)
```

Fitting per process at 1 PeV in water: brems `p_B = 0.12` (≈ pure `1/y` tail),
photonuclear `p_PN = 0.97`; pair production has `d/b ≃ 0.024`, so soft that two
terms suffice:

```
  Φ(A) ≃ [A b_PP − A(A−1)/2 d_PP]                        (pair production)
       + κ_B [ψ(A+p_B+1) − ψ(p_B+1)]                     (bremsstrahlung)
       + κ_PN [ψ(A+p_PN+1) − ψ(p_PN+1)]                  (photonuclear)
```

This reproduces `Φ(1) = 0.34900 = b_μ` and `Φ(2) = 0.62139 = 2b_μ − d_μ` to five
digits — the identities are *not* built in, so this validates the construction.
Caveat: rows other than `A = 1, 2` inherit whatever the two-moment family gets
wrong about third and higher moments.

**(b) Definitive version** — a single 1D quadrature of `Φ(A) = ∫ dΓ/dy [1−(1−y)^A]`
against PROPOSAL's differential cross sections. Cheap, tabulated once.
`Φ'(0) = ∫ dΓ/dy ln(1/(1−y)) ≥ b_μ` is the log-loss rate; `Φ` interpolates between
it at `s=0` and `b_μ` at `s=1` — the "drift coefficient" is just `Φ` at one
spectral index, not a universal propagation constant.

---

## 10. Consequences and things to revisit (Parts 11–12)

- **Diffuse fit.** Replacing `A'b_μ → Φ(A)` moves `V_soft` by <1% across
  IceCube's whole credible region `A = 0.98⁺⁰·¹¹₋₀.₀₉` — far inside their 30%
  photonuclear uncertainty. Their fit is fine, but the protection is the
  exactness identity, not marginal convergence. The one real shift is
  normalisation: including `I(A) ≃ 0.8` moves the inferred efficiency
  `ε_IC-TG ≃ 0.45 → 0.56`, closer to their geometric estimate `[0.37, 0.53]`.
- **Energy-dependent rates (Assumption 4).** Photonuclear grows from 25%→35% of
  `b_μ` over 1→100 PeV. Power laws stop being exact eigenfunctions, but for slow
  variation use the WKB form `e^{−Φ(A)ℓ} → exp[−∫ dη Φ(A; E(η))]` (the content of
  the paper's Appendix A.1, without a kernel).
- **Non-power-law sources** (DM line, cutoff, transient) excite many Mellin modes
  → the *only* place you actually need the inverse transform. Then use the exact
  propagator: accumulated loss `w = ln(ε/E)` is a compound Poisson subordinator
  with Laplace exponent `Φ(s)`, so `E[e^{−s w(ℓ)}] = e^{−ℓ Φ(s)}`, manifestly
  supported on `w ≥ 0` (no spurious energy gain, no `θ(ε−E)` patch — contrast the
  Fokker–Planck kernel's hand-imposed `θ(ε−E)` and its small Gaussian `ε<E`
  artifact).
- **Detector response (Assumption 9).** Everything here is a geometric
  through-going rate; a real `ε(E_μ, Ω)` multiplies the bracket and needs the
  collaboration's (non-public) response maps.

---

## 11. Mapping to `softpaws`

The exact path simplifies the numerical recipe in `soft_volume_notes.md` §8:

- **`transport/coefficients`** — instead of just `b_μ, d_μ`, provide `Φ(A)`:
  either the closed-form two-moment digamma family (§9a) or a tabulated 1D
  quadrature against PROPOSAL cross sections (§9b). Keep `b_μ, d_μ` as inputs and
  for cross-checks via the exactness identities `Φ(1)=b_μ`, `Φ(2)=2b_μ−d_μ`.
- **`transport/`** — no Green's-function convolution needed for power-law fluxes;
  `ϕ_μ(x,E) = S(E)(1−e^{−Φ(A)x})/Φ(A)` is closed form. Keep the subordinator
  propagator `e^{−ℓΦ(s)}` only for the non-power-law path.
- **`transport/soft_volume`** — `V_soft = A_proj I(A)(1−e^{−Φ(A)x})/Φ(A)`; keep
  the finite-`x` saturation factor (do **not** default to `x→∞`).
- **`transport/source`** — `S(ε) = n_N I(A) σ_νN(ε) ϕ_ν^⊕(ε)`, with
  `A = γ−λ−1`, `I(A) = ⟨(1−y_w)^A⟩_P`.
- **`response/`** — the `SoftVolumeResponse` predictor is now the closed-form
  master formula of §5, alongside the IRF path, same `flux → dN/dE dΩ` interface.
