# The transport exponent — physics notes

Distilled reference for the eigenvalue solution softpaws is built on, derived
in the method paper, Meighen-Berger, *Estimating High-Energy Neutrino Effective Areas from
Muon Propagation* (2026). It replaces the earlier Fokker–Planck
(drift–diffusion) expansion with an eigenvalue treatment of the QED collision
operator, keeping every loss moment. This is the physics core of the
`transport/` → `response/` path.

Read alongside [Prior work: the drift–diffusion soft volume](soft_volume.md),
which distills the earlier approach of Palmisano, Redigolo, Tammaro and Tesi
([arXiv:2607.13143](https://arxiv.org/abs/2607.13143)). Their Fokker–Planck
result is recovered here as the first two terms of the series — see
[§8, Contrast](#8-contrast-with-the-drift-diffusion-path).

**Naming convention on this page.** "The method paper" is Meighen-Berger
(2026), and the section, appendix and table references attached to it follow
its current numbering: Roman-numbered sections I to VI and lettered appendices
A to D, with App. A "The Transport" holding A.1 The Loss Kernel, A.2 The Range
to Threshold, A.3 The Drift-Diffusion Limit and A.4 Tau Transport. "Palmisano
et al." is arXiv:2607.13143, and every equation, figure, table and appendix
number attached to that name is theirs.

---

## 1. The one-sentence idea

The muon energy-loss operator is scale-invariant (rates depend on the fractional
loss `y`, not on energy `E`). Scale-invariant operators are **diagonalised by
power laws**, and the astrophysical source *is* a power law. So the entire
integro-differential transport problem collapses to **multiplication by a single
number** `Φ(A)` — no propagator, no kernel, no convolution, and no energy
cutoffs. (Method paper, Sec. I and Sec. II.A.)

Where Palmisano et al. expand the collision operator in small `y` to get a
differential Fokker–Planck equation, this treatment keeps the collision operator
exact and diagonalises it instead.

---

## 2. The transport equation and the exact collision operator (method paper, Sec. II and App. A)

Same starting point as Palmisano et al.: steady-state (`∂_t f = 0`),
ultra-relativistic (`v = c = 1`), collinear (`Δθ ~ m_μ/E ≪ 1`, so `Ω` is a
spectator label). The muon flux per unit energy `ϕ_μ(x, E)` along a fixed line of
sight (`x` = column depth [km w.e.]) obeys

```
∂ϕ_μ(x,E)/∂x  =  C_QED[E; ϕ_μ]  +  S(x,E)
```

The QED collision operator is kept as the **exact integral operator** (the
collision operator the method paper writes down at the opening of Sec. II), not
expanded:

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

### 2.1 Power laws are eigenfunctions (scale-invariant rates, the one assumption of the method paper, Sec. II.A)

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

**Finite with no cutoffs** (this is the key practical win over Palmisano et al.):
- *IR-finite.* Bremsstrahlung `dΓ/dy ~ κ/y` diverges at `y→0`, but
  `1 − (1−y)^s → sy`, so the integrand → `sκ`, finite. The loss-term divergence
  is cancelled by the gain term (an infinitely soft photon does nothing).
  **No `y_min = 10⁻⁷` needed.**
- *UV-finite.* As `y→1`, `(1−y)^s → 0`, bracket → 1: a catastrophic collision
  simply removes the muon. Hard brems/photonuclear tails are integrated exactly.
  **No `y_cut` machinery needed** (Appendices A and B of Palmisano et al.).

### 2.2 Mellin transform (formal statement, method paper App. A)

Define `ϕ̂(x,s) ≡ ∫₀^∞ dE E^s ϕ_μ(x,E)`. Then `Ĉ_QED(s) = −Φ(s) ϕ̂(s)`:
multiplication, not convolution. Structurally `Φ(s)` is a Mellin symbol /
anomalous dimension — the same object as a moment of a DGLAP splitting function.
The Green's function is `e^{−Φ(s)ℓ}`, one number per mode; because a single
power-law source excites exactly one `s`, you never need the inverse transform.

---

## 3. Solving the transport equation (method paper, Sec. III.A and App. A)

Mellin-transforming the PDE turns the integro-differential equation into a linear
first-order **ODE in depth**, mode by mode:

```
∂ϕ̂(x,s)/∂x = −Φ(s) ϕ̂(x,s) + Ŝ(x,s)
```

- **First order in `x` ⇒ initial-value problem in depth** (depth plays the role
  of time). The *only* boundary condition allowed is `ϕ_μ(x=0, E) = 0` (no muons
  enter at the Earth's surface, the condition `ϕ̂(0, s) = 0` the method paper
  imposes in App. A). You cannot additionally impose a condition at the detector
  without over-determining the system.
- **This justifies the "free flux at the surface" shortcut of Palmisano et al.**
  A perfectly absorbing detector at `x_det` affects only `x > x_det`, which
  nobody observes, so the detector never back-reacts on the flux arriving at it.
  Not an approximation — forced by the hyperbolic structure. The method paper
  states the boundary condition and does not spell this argument out.
- (Atmospheric muons violate that boundary condition for down-going directions;
  handled as a separately measured background, not through this equation.)

Integrating factor `e^{Φ(s)x}` gives the exact solution:

```
ϕ̂(x,s) = ∫₀ˣ dξ e^{−Φ(s)(x−ξ)} Ŝ(ξ,s)
```

Muons produced at depth `ξ` propagate `ℓ = x − ξ` and are attenuated by
`e^{−Φ(s)ℓ}` — attenuation of the *spectral-mode amplitude*, not of muon number
(QED collisions don't destroy muons). **Exact so far: no expansion in `y`, no
Fokker–Planck, no truncation.**

---

## 4. The source term (method paper, Sec. II.C)

Built from scratch, same physics as Palmisano et al. but carried exactly. Muon
born from `ν_μ` CC DIS with `ε = (1−y_w) E_ν`, `y_w` the weak inelasticity
(`⟨y_w⟩ ≈ 0.2`), `P(y_w) = σ⁻¹ dσ/dy_w`:

```
S(ε) = n_N ∫₀¹ dy_w/(1−y_w) · P(y_w) · [ϕ_ν σ_νN](ε/(1−y_w))
```

Same `1/(1−y)` Jacobian structure as the QED gain term, same reason. With
power-law flux `ϕ_ν ∝ E_ν^{−γ}` and power-law cross section `σ_νN ∝ E_ν^λ`,
`λ ≈ 0.4` (both stated in the method paper, Sec. II.C), and step-function
attenuation + constant near-detector density (the transparent-Earth idealization
the method paper starts from in Sec. III.A), the product is a single power
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
- **`I(A)` is a genuine normalisation factor** Palmisano et al. mostly drop.
  `I(1) = 1 − ⟨y_w⟩ ≃ 0.8` exactly when `A ≈ 1` (no averaging approximation
  needed).

---

## 5. Master solution and event rate (method paper, Sec. III.A)

Because the source excites exactly one mode, the ODE solves directly in energy
space (`S(ε) = S₀ ε^{−1−A}`, `ξ`-independent under the same constant-density
idealization):

```
        ┌──────────────────────────────────────┐
        │  ϕ_μ(x,E) = S(E) · (1 − e^{−Φ(A)x})/Φ(A)  │
        └──────────────────────────────────────┘
```

Sanity checks: `x→0` → `S(E)x` (thin slab, linear, none lost); `x→∞` →
`S(E)/Φ(A)` (equilibrium); `Φ→0` → `S(E)x` for all `x` (no losses, whole column
accumulates — **finite**, remember this one).

**Soft volume** (the second term of the track rate in the method paper,
Sec. III.A) — the target volume that, producing muons with no propagation
losses, would give the observed arrival rate:

```
  V_soft(E) = A_proj · I(A)/Φ(A) · (1 − e^{−Φ(A)x})
```

Three factors: `A_proj/Φ(A)` = area × effective range `1/Φ(A)` (range weighted by
spectral replenishment, not the CSDA range); `I(A) ≃ 0.8`; and the **saturation
factor** `(1 − e^{−Φx})` = finite upstream column.

**Master formula** (the track rate of the method paper, Sec. III.A) — two
disjoint populations: produced inside (`V_det = 4/3 π R_det³`) and produced
outside, arriving through the projected surface (`A_proj = π R_det²` for a
sphere, the constant-projected-area detector the method paper takes first in
Sec. III.A). `I(A)` multiplies both (same production vertex):

```
        ┌─────────────────────────────────────────────────────────────────────┐
        │  dN/(dt dE dΩ) = I(A) n_N σ_νN(E) ϕ_ν^⊕(E) · [ V_det                  │
        │                     + A_proj/Φ(A) · (1 − e^{−Φ(A)x}) ]                │
        └─────────────────────────────────────────────────────────────────────┘
```

With `R_det ≃ 0.62 km` and `1/Φ ≃ 2.9 km`, the soft term is ~4× the instrumented
volume — the central claim of Palmisano et al., now with the corrected
coefficient (`I(A)` and exact `Φ(A)`).

---

## 6. The saturation factor `(1 − e^{−Φx})` — why it matters (method paper, Sec. III.A)

The method paper names the saturation factor in Sec. III.A and takes the
transparent-Earth limit `1/Λ_ν → 0` in App. A. The depth table and the `Φ < 0`
continuation below are this page's own and are not in the current method paper.

The detector here is a perfect absorber (unit efficiency, no
cuts/acceptance/threshold — the prediction is a geometric through-going rate),
an idealization the method paper has since replaced by the two-number acceptance
of Sec. III.B. Two reasons to keep the full saturation factor instead of the
`x→∞` idealisation of Palmisano et al.:

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

## 7. The cross-section pole (not in the current method paper)

The current method paper does not carry this pole. It records only `Φ(0) = 0`
with `Φ' > 0` (Sec. II.A) and the convergence edge of the calibrated exponent at
negative `A` (App. A.1). The derivation is kept here.

The denominator `Φ(A)` crosses zero when `λ > γ − 1` (i.e. `A < 0`). For
`γ = 2.38` the **pole sits at `λ = 1.38`**. Physically, when the cross section
grows faster than the flux falls, more distant / higher-energy parent neutrinos
dominate and the infinite-column integral genuinely diverges (`(1−y)^A > 1`, so
`Φ(A) < 0`, the observable-weighted population is no longer attenuated). This is
not a numerical artifact — it is the signal that the `x→∞` assumption has failed.

**Consequence for the cross-section extraction:** Fig. 8 of Palmisano et al.
quoted modes `λ_IC = 1.48` (1 yr) and `1.66` (9.5 yr) both sit *past* the pole,
where their Eq. (4.6) returns a *negative* event count. That the distributions
pile up at `λ ~ 1.2–1.7`, straddling the pole, looks like the pole rather than
the data — worth redoing with the saturation factor before believing
`R = σ/σ_SM ~ 10`. (Caveat: possible their code holds `λ = 0.4` inside `A` while
floating it elsewhere, which would be inconsistent but avoids the sign flip.)

---

## 8. Contrast with the drift-diffusion path

`docs/theory/soft_volume.md` documents the **Fokker–Planck path of Palmisano et
al.**; this file documents the **exact eigenvalue path**. They describe the same
physics; the exact version *contains* theirs as a truncation (method paper,
Sec. II.D and App. A.3).

| Aspect                | Palmisano et al. / [prior work](soft_volume.md) (Fokker–Planck) | This derivation (exact eigenvalue)                              |
|-----------------------|-----------------------------------------------------------------|-----------------------------------------------------------------|
| Collision operator    | Expanded in small `y` → drift `b_μ` + diffusion `d_μ`           | Kept exact; diagonalised by power laws                          |
| Governing solution    | Drift–diffusion PDE in energy (their Eq. 2.9)                    | ODE in depth per Mellin mode; single mode `s = A`               |
| Propagator            | Log-normal Green's function `G(E,x;ε,ξ)` (their Eq. 2.16), convolved | None — multiply by `e^{−Φ(A)ℓ}`; no kernel, no convolution  |
| Attenuation constant  | `b_μ` (drift), `d_μ` (diffusion) — first two `y`-moments         | `Φ(A) = ∫ dΓ/dy [1−(1−y)^A]` — the *full* eigenvalue            |
| Energy cutoffs        | Needs `y_min` (IR) and `y_cut` (UV, their Appendices A/B)        | IR- and UV-finite; **no cutoffs**                               |
| Range factor          | `V_soft ∝ 1/(b_μ A)`, with an unphysical `1/A` pole at `A→0`     | `V_soft ∝ I(A)(1−e^{−Φ(A)x})/Φ(A)`, finite everywhere           |
| Column depth          | Typically `x→∞`                                                 | Finite-`x` saturation factor `(1−e^{−Φx})` kept                 |
| Normalisation `I(A)`  | Mostly dropped (their B.5 carries `1+A`; 2.21–2.23 drop it)      | Kept: `I(A) ≃ 0.8`                                              |
| Evaluation            | Numerical Green's-function convolution                          | Closed form once `Φ(A)` is known                                |

### 8.1 The Fokker–Planck result is the first two terms (method paper, Sec. II.D)

Binomial-expanding `1 − (1−y)^A` and integrating against `dΓ/dy` (with moments
`b_μ = ∫ y dΓ/dy`, `d_μ = ∫ y² dΓ/dy`, `t_μ = ∫ y³ dΓ/dy`):

```
  Φ(A) = A b_μ − A(A−1)/2 · d_μ + A(A−1)(A−2)/6 · t_μ − …
```

Truncating at second order gives exactly the `Φ ≃ A' b_μ` of Palmisano et al.
with `A' = A[1 − (d_μ/2b_μ)(A−1)]`. **The Fokker–Planck result of Palmisano et
al. is the first two terms of the exact series.**

### 8.2 Exactness identities — why the fit of Palmisano et al. survives (method paper, Sec. II.D)

The series **terminates for positive integer `A`** (every term beyond the first
carries a factor `(A−1)`):

```
  Φ(1) = b_μ              (since 1 − (1−y) = y identically)
  Φ(2) = 2b_μ − d_μ       (since 1 − (1−y)² = 2y − y² identically)
```

Kramers–Moyal truncation at order `n` is exact for integer `A ≤ n`. IceCube's fit
gives `γ = 2.38`, `λ = 0.4` ⇒ **`A = 0.98`**, sitting essentially *on* the `A = 1`
exactness point. There, `Φ = b_μ` and nothing else — the entire hard tail of
brems + photonuclear (the ~25% Appendix-B systematic of Palmisano et al., which
the method paper resums in App. A.3) **cancels identically between loss and
gain**. The residual error is `(A−1)·d_μ/2b_μ ≈ 0.02 × 0.11 ≈ 0.2%`.

So the soft expansion survives **not** because it is marginally convergent
(`d_μ/b_μ ~ 0.2`, the reason Palmisano et al. state) but because the expansion
parameter is `(A−1)d_μ/2b_μ`, and IceCube's spectrum happens to sit exactly
where it vanishes. Palmisano et al. note `A ≈ 1` twice without noticing it
protects them.

### 8.3 Numeric comparison of `Φ(A)` (the table of the exponent against its truncations in the method paper, App. A)

The method paper's table is now built on PROPOSAL's quadrature moments,
`b_μ = 0.380 km⁻¹` at 1 PeV in water, so its rows differ in absolute value from
the ones below, which use the earlier `b_μ = 0.349`, and its sign change sits at
`A_die = 9.3` instead of the 10.1 quoted here.

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

## 9. Evaluating `Φ(A)` in practice (method paper, App. A.1 and A.2)

Table 1 of Palmisano et al. gives only `b_μ` and `d_μ`, but `Φ(A)` needs the full
shape of `dΓ/dy`. Two options:

**(a) Two-parameter family** calibrated to the two moments (the two-moment form
of the method paper, App. A.1). With `dΓ/dy = κ (1−y)^p / y`:

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
spectral index, not a universal propagation constant. The method paper takes
this route for the log-loss moments `Φ'(0)` and `Φ''(0)` that set the range
(App. A.2), and for `Φ(s)` itself it now uses a three-moment family matched to
`b_μ`, `d_μ` and `t_μ`, a difference of two Beta functions (App. A.1).

---

## 10. Consequences and things to revisit (method paper, Sec. II.D and App. A)

- **Diffuse fit.** Replacing `A'b_μ → Φ(A)` moves `V_soft` by <1% across
  IceCube's whole credible region `A = 0.98⁺⁰·¹¹₋₀.₀₉` — far inside their 30%
  photonuclear uncertainty. Their fit is fine, but the protection is the
  exactness identity, not marginal convergence (the method paper draws the same
  conclusion in Sec. II.D). The one real shift is normalisation: including
  `I(A) ≃ 0.8` moves the inferred efficiency `ε_IC-TG ≃ 0.45 → 0.56`, closer to
  their geometric estimate `[0.37, 0.53]`. The efficiency shift is this page's
  own and is not in the method paper.
- **Energy-dependent rates** (the scale-invariance assumption, method paper
  Sec. II.A). Photonuclear grows from 25%→35% of `b_μ` over 1→100 PeV. Power
  laws stop being exact eigenfunctions, but for slow variation use the WKB form
  `e^{−Φ(A)ℓ} → exp[−∫ dη Φ(A; E(η))]` (the content of Appendix A.1 of Palmisano
  et al., without a kernel). The method paper follows the running kernel down a
  track in App. A.2 and treats the photonuclear rise as a shift of the spectral
  argument in App. A.
- **Non-power-law sources** (DM line, cutoff, transient) excite many Mellin modes
  → the *only* place you actually need the inverse transform. Then use the exact
  propagator: accumulated loss `w = ln(ε/E)` is a compound Poisson subordinator
  with Laplace exponent `Φ(s)` (the method paper, App. A, identifies it as a
  subordinator but not a compound Poisson one, because the `1/y` bremsstrahlung
  tail makes the total jump rate infinite), so `E[e^{−s w(ℓ)}] = e^{−ℓ Φ(s)}`,
  manifestly supported on `w ≥ 0` (no spurious energy gain, no `θ(ε−E)` patch —
  contrast the Fokker–Planck kernel's hand-imposed `θ(ε−E)` and its small
  Gaussian `ε<E` artifact).
- **Detector response** (the perfect-absorber idealization of §6). Everything
  here is a geometric through-going rate; a real `ε(E_μ, Ω)` multiplies the
  bracket and needs the collaboration's (non-public) response maps. The method
  paper has since replaced this with the two-number acceptance of Sec. III.B and
  the released IceCube response of Sec. IV.D.

---

## 11. Mapping to `softpaws`

The exact path simplifies the numerical recipe in [prior work](soft_volume.md) §8:

- **`transport/coefficients`** — supplies `b_μ, d_μ` (`drift_coefficient`,
  `diffusion_coefficient`) as inputs and for cross-checks via the exactness
  identities `Φ(1)=b_μ`, `Φ(2)=2b_μ−d_μ`.
- **`transport/eigenvalue`** provides `Φ(A)`: `phi_eigenvalue` and
  `phi_symbol` for the closed-form two-moment digamma family (§9a),
  `phi_eigenvalue_three_moment` and `phi_symbol_three_moment` for the
  three-moment Beta-function family the method paper uses, and
  `phi_eigenvalue_quadrature` for the tabulated 1D quadrature against PROPOSAL
  cross sections (§9b). `spectral_index` builds `A = γ−λ−1`.
- **`transport/loss_distribution`** — no Green's-function convolution needed for
  power-law fluxes; `ϕ_μ(x,E) = S(E)(1−e^{−Φ(A)x})/Φ(A)` is closed form. The
  subordinator propagator `e^{−ℓΦ(s)}` is kept here (`loss_density`,
  `log_loss_cdf`) only for the non-power-law path.
- **`transport/soft_volume`** — `soft_volume_exact` and `saturation_factor`
  give `V_soft = A_proj I(A)(1−e^{−Φ(A)x})/Φ(A)`; keep the finite-`x`
  saturation factor (do **not** default to `x→∞`). The two target volumes
  `range_target_volume_km3` and `dm_line_target_volume_km3` stay in this module,
  while the muon ranges they call (`muon_range_km`, `stochastic_muon_range_km`,
  `two_medium_muon_range_km`, `truncated_muon_range_km`) live in
  `transport/muon_range`.
- **`transport/source`** — `S(ε) = n_N I(A) σ_νN(ε) ϕ_ν^⊕(ε)`, with
  `A = γ−λ−1`, `I(A) = ⟨(1−y_w)^A⟩_P` (`inelasticity_factor`,
  `nucleon_number_density`, `cc_cross_section`).
- **`response/`** — the `SoftVolumeResponse` predictor in `response/soft_volume`
  is the closed-form master formula of §5, alongside the IRF path in
  `response/irfs`, same `flux → dN/dE dΩ` interface.
