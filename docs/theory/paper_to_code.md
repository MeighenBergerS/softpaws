# `2026_softvolume.pdf` vs. the `softpaws` implementation

Line-by-line comparison of the equations in the current `docs/2026_softvolume.pdf`
("Exact Analytic Solutions for Neutrino-Induced Lepton Transport") against what
`src/softpaws/` actually computes. Section/appendix labels below (`Sec. II`,
`App. D`, ...) are the ones printed in the current PDF.

Read alongside `docs/exact_soft_volume_notes.md`, which distills the physics of
this paper in detail but cites it by an older "Part N" numbering that no longer
matches the PDF's Roman-numeral sections and lettered appendices — its numeric
content (Table E.1 in particular) checks out against the current PDF, only its
cross-references are stale. This document instead maps each equation directly
to file:line in the code, and calls out where the code diverges from, extends
beyond, or hasn't yet caught up to the paper.

---

## 1. What matches exactly

| Paper | Code |
|---|---|
| `Φ(s) = ∫₀¹ dy dΓ/dy [1-(1-y)^s]` (Eq. 5) | [`eigenvalue.py:101-137`](https://github.com/MeighenBergerS/softpaws/blob/main/src/softpaws/transport/eigenvalue.py#L101-L137) `phi_symbol`, [`:243-279`](https://github.com/MeighenBergerS/softpaws/blob/main/src/softpaws/transport/eigenvalue.py#L243-L279) `phi_eigenvalue_quadrature` |
| `A ≡ γ - λ - 1` (Eq. 7) | [`eigenvalue.py:42-63`](https://github.com/MeighenBergerS/softpaws/blob/main/src/softpaws/transport/eigenvalue.py#L42-L63) `spectral_index`, [`soft_volume.py:60-93`](https://github.com/MeighenBergerS/softpaws/blob/main/src/softpaws/transport/soft_volume.py#L60-L93) `spectral_penalty` |
| `S(ε) = n_N I(A) σ_νN(ε) φ_ν^⊕(ε)`, `I(A)=⟨(1-y_w)^A⟩` (Eqs. 6, 8) | [`source.py:56-89`](https://github.com/MeighenBergerS/softpaws/blob/main/src/softpaws/transport/source.py#L56-L89) `inelasticity_factor` |
| `φ(x,E) = S(E)(1-e^{-xΦ(A)})/Φ(A)` (Eq. 9) | [`soft_volume.py:393-468`](https://github.com/MeighenBergerS/softpaws/blob/main/src/softpaws/transport/soft_volume.py#L393-L468) `soft_volume_exact`, [`:357-390`](https://github.com/MeighenBergerS/softpaws/blob/main/src/softpaws/transport/soft_volume.py#L357-L390) `saturation_factor` |
| `dN/(dt dE dΩ) = I(A) n_N σ_νN φ_ν^⊕ [V_det + A_proj/Φ(A)(1-e^{-xΦ(A)})]` (Eq. 10) | [`response/soft_volume.py`](https://github.com/MeighenBergerS/softpaws/blob/main/src/softpaws/response/soft_volume.py) `SoftVolumeResponse` with `method="exact"` |
| Fokker-Planck as Taylor expansion, `Φ(A)=Ab_μ - A(A-1)/2 d_μ + ...` (Eq. 13) | [`eigenvalue.py:211-240`](https://github.com/MeighenBergerS/softpaws/blob/main/src/softpaws/transport/eigenvalue.py#L211-L240) `phi_fokker_planck`, `phi_drift` |
| Exactness identities `Φ(1)=b_μ`, `Φ(2)=2b_μ-d_μ` (Eqs. 14-15) | Same functions; docstrings assert these, and Table E.1's numbers reproduce to 5 digits (`exact_soft_volume_notes.md` §8.3) |
| Two-moment calibrated kernel, `Φ(s)=κ[ψ(s+p+1)-ψ(p+1)]` (App. E.1-E.2) | [`eigenvalue.py:66-98`](https://github.com/MeighenBergerS/softpaws/blob/main/src/softpaws/transport/eigenvalue.py#L66-L98) `two_moment_loss_spectrum`, [`:101-137`](https://github.com/MeighenBergerS/softpaws/blob/main/src/softpaws/transport/eigenvalue.py#L101-L137) `phi_symbol` |
| Subordinator / characteristic-function inversion of the log-loss law (App. B, Eq. B1) | [`loss_distribution.py`](https://github.com/MeighenBergerS/softpaws/blob/main/src/softpaws/transport/loss_distribution.py) `loss_density` (inverts `e^{-ℓΦ(-ik)}` on a `k`-grid) |
| Tau composite symbol `I(s,ℓ) = (e^{-ℓ/ℓ̃τ}-e^{-ℓΦμ})/(Φμ-1/ℓ̃τ)` (App. D, Eq. D1) | [`tau.py:248-343`](https://github.com/MeighenBergerS/softpaws/blob/main/src/softpaws/transport/tau.py#L248-L343) `tau_loss_density`, using exactly this closed form with the L'Hopital limit at the removable pole |
| `⟨z^s⟩` tau-decay moment, used the same way as App. D's decay moment `M(A)` | [`tau.py:112-161`](https://github.com/MeighenBergerS/softpaws/blob/main/src/softpaws/transport/tau.py#L112-L161) `z_symbol`/`z_moment` |

The exact-eigenvalue core (Secs. III-IV, App. A-C, E) is implemented faithfully
and is the best-covered part of the paper.

---

## 2. Former physics gaps, now closed (with caveats)

All four gaps below are now implemented. None was a drop-in transcription of
the paper's equations as literally printed — each required either an
interpretive choice the compressed appendices leave open, or (in two cases)
a deliberate departure from the paper's own derivation route once it turned
out to be numerically unstable. Those choices are documented in detail in the
referenced modules' docstrings; this section summarizes them and should be
read as "implemented, here's exactly what that means" rather than "matches
the paper verbatim."

### 2.1 Parent-neutrino attenuation — Eq. (11), implemented as Form C

`R_ν(x,A) = (e^{-x/Λ_ν} - e^{-xΦ(A)}) / (Φ(A) - 1/Λ_ν)` is now implemented
literally: [`attenuation.py`](https://github.com/MeighenBergerS/softpaws/blob/main/src/softpaws/transport/attenuation.py)
`neutrino_interaction_length_km` gives `Λ_ν` in the same km^-1 convention
`Φ(A)` uses, and
[`soft_volume.py`](https://github.com/MeighenBergerS/softpaws/blob/main/src/softpaws/transport/soft_volume.py)
`saturation_factor` was generalized with an optional `inv_lambda_per_km`
argument (default `0`, so every existing caller is unaffected) to compute
`R_ν` directly, including the L'Hopital limit at the removable singularity
`Φ(A) = 1/Λ_ν`. `soft_volume_attenuated_exact` assembles the full coupled
master formula — including the detail that in-detector production (at depth
`x` exactly) picks up a flat `e^{-x/Λ_ν}` survival factor while upstream
production (`0 ≤ ξ ≤ x`) picks up the integrated `R_ν`.
`SoftVolumeResponse.expected_counts_coupled_attenuation` is the response-layer
entry point ("Form C", alongside the pre-existing decoupled Forms A/B). This
one item matches the paper's derivation without a scope-narrowing
substitution — the closest of the four to "implemented as printed."

**Key realization along the way:** the column depth `x` used by the muon-
transport saturation factor and the column depth feeding `D_ν` were always
meant to be the same declination-dependent quantity (`exact_soft_volume_notes.md`
§5-6: "IceCube sits ~1.95 km deep" for downgoing; the full Earth chord for
upgoing). Forms A/B compute them as two independent numbers; Form C uses one.

### 2.2 Scale-breaking (App. F) — implemented as Eq. F4's leading-order running index, not the full Eq. F3 series

[`eigenvalue.py`](https://github.com/MeighenBergerS/softpaws/blob/main/src/softpaws/transport/eigenvalue.py)
`phi_eigenvalue_derivative` and
[`soft_volume.py`](https://github.com/MeighenBergerS/softpaws/blob/main/src/softpaws/transport/soft_volume.py)
`scale_breaking_saturation_factor` implement Eq. F4's first-order-in-`β`
running spectral index, `s(ℓ) ≈ A + βℓΦ(A)`, substituted into the propagator
and Taylor-expanded to give a Gaussian-modified exponent (evaluated by
quadrature). `soft_volume_exact` gained a `beta` parameter, `0` by default.

**Deliberately not Eq. F3's full series**: that series' source term
`Ŝ(ξ, s+kβ)` is only well defined at `k=0` for a genuine single-power-law
source (the case this package treats throughout) — the compressed paper
never spells out how the higher shifted terms are meant to be regularized
for a single-mode source (contrast App. H, which handles exactly this kind
of bookkeeping explicitly for a *cutoff* source). Eq. F4 sidesteps that
ambiguity by construction, at the cost of being a small-`β` approximation.
No calibrated nonzero `β` ships with the package — `coefficients.py` already
found the LPM suppression negligible for muons in the covered range, so
this closes the structural gap without asserting an unverified photonuclear
exponent.

### 2.3 Cutoff sources (App. H) — implemented via real-space convolution, not the literal Cahen-Mellin series

[`cutoff_source.py`](https://github.com/MeighenBergerS/softpaws/blob/main/src/softpaws/transport/cutoff_source.py) and
`SoftVolumeResponse.differential_rate_with_cutoff` implement Sec. VII.C's
recommended mitigation for the `A<0` blowup.

**Deliberately not Eq. H3's literal pole series**: a first implementation
attempt (Touchard/Bell-polynomial Taylor coefficients of `h(s) = I(s) R(s,x)`
around `s=A`) reproduced the plain treatment correctly in the mild regime but
**overflowed to `NaN`** in exactly the large-`x`, negative-`A` regime App. H
exists to fix — `h(s)` itself scales like `exp(x|Φ(s)|)` at every stencil
point near `A`, so individual series terms blow up before any cancellation
can bring them back down; a truncated real-axis series is not numerically
stable there. The shipped implementation instead evaluates the *same*
physical convolution directly in energy space, via App. B's already-exact,
already-tested log-loss density
([`loss_distribution.py`](https://github.com/MeighenBergerS/softpaws/blob/main/src/softpaws/transport/loss_distribution.py)
`loss_density`, which is a proper bounded probability density with no
exponentially large intermediates anywhere), integrated over the upstream
production depth. Verified to stay finite, positive, and monotonic at
`A<0`, `x=10⁴ km.w.e.` — the case that overflowed under the literal series —
and to reduce to the plain treatment as `E0 → ∞`. It is markedly slower than
the rest of the package (an FFT-style inversion per production-depth slice),
so it is exposed as an explicit opt-in method rather than wired into any
default path.

### 2.4 Dark matter lines (App. I) — Eq. I.2 implemented and checked; the `s=0` line-of-sight limit turns out to already equal the pre-existing J-factor calculation

`examples/34_dm_line_sensitivity.py::dm_line_flux_general_s` now implements Eq. I.2
literally, at general `s`. Working out what it says at the physical value
`s=0` (a monochromatic line has no continuum spectral index to average over):
`Φ(0)=0` identically, so the line-of-sight propagator `e^{-ℓΦ(0)}` is exactly
`1` for every `ℓ`, and Eq. I.2 collapses to the plain, unattenuated J-factor
integral — which `los_j_factor` already computed before this change. `main()`
now checks this equivalence numerically (`max |Eq. I.2(s=0) / plain J-factor
- 1|`, printed each run) rather than asserting it in prose. That part of the
original gap was benign: the general machinery, evaluated where the physics
actually calls for it, reproduces what was already coded.

What genuinely changed: `transport/soft_volume.py::dm_line_target_volume_km3`
adds the literal `s→0` instance of the *detector-side* soft-volume formalism
(`Φ(0)=0` makes the near-detector effective range grow linearly in the
available column rather than logarithmically, unlike the muon-range
convention), plotted in the example alongside — not in place of — the
pre-existing `threshold_effective_area_cm2` muon-range curve. App. I itself
only discusses the halo line of sight, not this near-detector piece, so this
is the item with the least direct paper backing among the four; it is kept
as an explicit, separately-labeled comparison rather than a replacement.

---

## 3. Places the code goes beyond the paper

These aren't discrepancies, just capability the paper doesn't cover — noted so
they aren't mistaken for implementing a paper equation that doesn't exist:

- **Tabulated BGR18 cross section** (`cross_section.py`, `bgr18_cross_section`)
  as an alternative to the paper's single power law `σ_νN ∝ E^λ` (Eq. 5-region
  analogue in Sec. III.B) — used in `examples/34_dm_line_sensitivity.py` because the power
  law isn't valid up to the DM masses probed there.
- **PREM-layered per-event Earth column** (`attenuation.py:225-284`,
  `prem_density`/`prem_column`) vs. the paper's implicit constant-density
  picture; this refines the column depth `X` that feeds `D_ν` in Forms A/B,
  and also feeds Form C's `x` directly (§2.1) — PREM refines the column
  regardless of which of the three attenuation forms uses it.
- **`tau_to_muon_ratio`** (`tau.py:211-245`), a closed-form population-averaged
  `N_τ→μ/N_νμ` ratio — structurally consistent with App. D's composite symbol
  but is the codebase's own construction; the paper states only the general
  `V_τ_soft` form (Eq. 12) and appendix mechanics, not this specific ratio.
- **Dynamic (energy-growing) projected area** (`soft_volume.py`
  `dynamic_projected_radius_km`/`dynamic_projected_area_km2`), a
  phenomenological detector-response term with no counterpart in the paper:
  above the critical energy `E_c` (`critical_energy_gev`), stochastic
  radiative losses let a track trigger from lateral distances beyond the
  static `R_det`, growing logarithmically with energy. Its one free
  parameter (a growth length `L`) is fit in `examples/26_dynamic_response_
  effective_area.py` against the high-energy residual left over after
  example 20's comparison, not asserted from theory.

---

## 4. Summary table

| Paper section | Status in `softpaws` |
|---|---|
| Sec. II-III, App. A-B (transport eq., diagonalization, Mellin, subordinator) | Implemented |
| Sec. IV.A, App. C.1 (basic soft volume, Duhamel) | Implemented |
| Sec. IV.B, App. C.2-C.3 (parent-ν attenuation folded into `R_ν(x,A)`, Eq. 11) | Implemented as printed ("Form C", §2.1) |
| Sec. V, App. D (tau-induced tracks) | Implemented (`tau.py`) |
| Sec. VI, App. E (Fokker-Planck as truncation, exactness identities, calibrated kernel) | Implemented |
| Sec. VII (data-analysis implications: IceCube robustness, KM3NeT pole) | Reproduced numerically (see `exact_soft_volume_notes.md` §6-8), not separate code but a direct consequence of §1's formulas |
| Sec. VIII, App. F-G (scale breaking, LPM shift operator) | Eq. F4 leading-order running index implemented; full Eq. F3 series not (ambiguous source term, §2.2) |
| App. H (cutoff sources, Cahen-Mellin) | Implemented via real-space convolution; literal pole series overflows in the regime it's meant to fix, so not used (§2.3) |
| App. I (dark matter lines) | Eq. I.2 implemented and checked; `s=0` limit equals the pre-existing J-factor calculation; detector-side `s=0` volume offered as a comparison, not App. I's own topic (§2.4) |
| App. J (performance/methodology tables) | N/A (not a physics result) |
