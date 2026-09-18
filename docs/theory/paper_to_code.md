# The method paper vs. the `softpaws` implementation

Line-by-line comparison of the equations in the method paper, Meighen-Berger,
*Estimating High-Energy Neutrino Effective Areas from Muon Propagation* (2026), against
what `src/softpaws/` computes. Sections are cited by their Roman numerals
(Sec. II.E) and appendices by their letters (App. A.2). Equations are named by
what they say and, where that helps, by their label in the paper's source
(`eq:Lclosed`), so a reference survives renumbering.

Read alongside [The transport exponent](exact_soft_volume.md), which distills
the physics of the eigenvalue solution but cites the paper by an older
"Part N" numbering, and [The range to threshold](first_passage_range.md). This
document instead maps each equation to a module and function in the code, and
calls out where the code diverges from, extends beyond, or has not caught up
to the paper.

Throughout, "the paper" means the method paper. The earlier drift-diffusion
calculation of Palmisano et al. (arXiv:2607.13143) is covered separately in
[Prior work: the drift-diffusion soft volume](soft_volume.md).

---

## 1. What matches

| Paper | Code |
|---|---|
| Transport exponent `Φ(s) = ∫₀¹ dy dΓ/dy [1-(1-y)^s]` (Sec. II.A, `eq:Phi`) | `transport.eigenvalue.phi_eigenvalue_quadrature` (direct quadrature of a tabulated `dΓ/dy`), `phi_eigenvalue` (real `A`), `phi_symbol` and `phi_symbol_three_moment` (complex `s`, for the calibrated families) |
| Effective spectral index `A ≡ γ - λ - 1` (Sec. II.C, `eq:Adef`) | `transport.eigenvalue.spectral_index`, `transport.soft_volume.spectral_penalty` |
| Source mode `S(ε) = n_N I(A) σ_νN(ε) φ_ν^⊕(ε)`, `I(A) = ⟨(1-y_w)^A⟩` (Sec. II.C, `eq:source`, `eq:sourcemode`) | `transport.source.inelasticity_factor` |
| Transparent-Earth solution `φ(x,E) = S(E)(1-e^{-xΦ(A)})/Φ(A)` (Sec. III.A, `eq:softbasic`) | `transport.soft_volume.soft_volume_exact`, `transport.soft_volume.saturation_factor` |
| Track rate `dN/(dt dE dΩ) = I(A) n_N σ_νN φ_ν^⊕ [V_det + A_proj (1-e^{-xΦ(A)})/Φ(A)]` (Sec. III.A, `eq:rate`) | `response.soft_volume.SoftVolumeResponse` with `method="exact"` |
| Kramers-Moyal expansion `Φ(A) = A b_μ - A(A-1)/2 d_μ + ...` (Sec. II.D, `eq:km`) and the drift-diffusion range `L_DD` (App. A.3, `eq:LDD`) | `transport.eigenvalue.phi_fokker_planck`, `phi_drift`, `transport.soft_volume.soft_volume_drift`, `soft_volume_diffusion` |
| Identities `Φ(1) = b_μ`, `Φ(2) = 2b_μ - d_μ` and the sign change at `A_die = 1 + 2b_μ/d_μ` (Sec. II.D, `eq:exact1`, `eq:exact2`, `eq:die`) | Same functions. With the paper's `b_μ = 0.380`, `d_μ = 0.092`, `t_μ = 0.056` km⁻¹, `phi_eigenvalue_three_moment` against `phi_fokker_planck` reproduces every row of the paper's truncation table (App. A, `tab:fp`) to within one unit in its last printed digit, and gives `A_die = 9.26`. The table in [the transport-exponent notes](exact_soft_volume.md) §8.3 is the same comparison at an older calibration (`b_μ = 0.349`), so its rows differ from the paper's. |
| Calibrated kernel (Sec. II.B, App. A.1): the two-moment family `Φ(s) = κ[ψ(s+p+1) - ψ(p+1)]` and the three-moment family as a difference of Beta functions (`eq:app:phithree`) | `transport.eigenvalue.two_moment_loss_spectrum` with `phi_symbol`, and `three_moment_loss_spectrum` with `phi_symbol_three_moment` and `phi_eigenvalue_three_moment` |
| Log-loss moments `Φ'(0) = ⟨-ln(1-y)⟩`, `-Φ''(0) = ⟨ln²(1-y)⟩` by quadrature of the tabulated spectrum (App. A.2, `eq:Phimoments`) | `transport.coefficients.log_loss_moments` (reads the shipped PROPOSAL table, as the paper prescribes, and adds `Φ'''(0)` for the variance) |
| Range to threshold: first-passage definition and closed form (Sec. II.E, `eq:Ldef`, `eq:Lclosed`), running kernel (`eq:Lrun`), ionization splice at `E_* = 10 TeV` (`eq:Lsplice`), two media (`eq:Ltwomedium`), variance (`eq:Lvar`), all in App. A.2 | `transport.muon_range.stochastic_muon_range_km` (`kernel_evaluation="running"` is the default and `"frozen"` gives `eq:Lclosed`, `match_energy_gev` defaults to `DEFAULT_IONIZATION_MATCH_GEV` = 10 TeV), `two_medium_muon_range_km`, `two_medium_range_ratio`, `stochastic_muon_range_variance_km2`. The continuous-slowing-down range the paper compares against is `muon_range_km`. |
| Subordinator and Laplace exponent (App. A, `eq:app:levykhintchine`), inversion of the log-loss law (Sec. II.F, `tab:losslaw`) | `transport.loss_distribution.invert_log_loss_symbol` (any symbol), `loss_density` (two-moment), `loss_density_three_moment`, `log_loss_cdf` |
| Tau decay moment `M(A) = 4/((A+1)(A+2)(A+3)) + 8/((A+2)(A+3)(A+4))` (App. A.4, `eq:decaymoment`) | `transport.tau.z_moment`, `transport.tau.z_symbol` (complex `s`) |
| Tau composite propagator at a fixed decay length, "an exponential in `1/ℓ_τ + Φ_τ` against one in `Φ_μ`" (App. A.4, the prose after `eq:vsofttau`) | `transport.tau.tau_loss_density` inverts the closed form `⟨z^s⟩ (e^{-ℓ/ℓ_τ} - e^{-ℓΦ_μ}) / (ℓ_τ Φ_μ - 1)` with the L'Hopital limit at the removable pole. The code drops the tau's own losses (`Φ_τ = 0`), which the module docstring states and which the paper's prompt limit also assumes. |
| Two-channel rate with `B_τ→μ M(A) V_τ` (Sec. III.C, `eq:totalrate`) and the tau's regenerating Earth transmission (Sec. III.D) | `response.effective_area.ic_effective_area_tau_channel`, `arca_effective_area` with `flavour="tau"`, `transport.attenuation.flavour_transmission` |
| Regeneration ladder `dφ_k/dX = -N_A σ_tot(E_k) φ_k + N_A σ_NC(E_{k-1}) φ_{k-1}` (Sec. III.D, `eq:ladder`) | `transport.attenuation.regenerated_transmission` (one diagonalization, evaluated at every column) |
| Reach ansatz `R_eff(E) = max[0, R_det + Λ ln(E/E_piv)]` (Sec. III.B, `eq:reach`) | `transport.soft_volume.light_reach_radius_km`, fitted in `response.reduced.fit` and predicted from the optics in `response.light_reach` (App. B.1) |
| Tabulated effective area `A_eff(E_ν, Ω) = n_N Σ_k w_k σ_CC(E_k) [A_proj L(E_k) + V_det]` (Sec. IV, `eq:aeff`) and its one-line form (Sec. IV.C, `eq:recipe`) | `response.effective_area.ic_effective_area_regenerated`, `arca_effective_area`, `response.site_models`, `response.reduced` (two instrument numbers per site), `response.first_principles` (reach derived, nothing fitted) |
| Point-source limit `φ_0^lim(δ) = N_lim / (T ∫ dE A_eff (E/E_piv)^{-γ})` with the Feldman-Cousins average over background-only outcomes (Sec. V.A, `eq:pslim`) | `response.sensitivity.power_law_sensitivity`, `sensitivity_upper_limit`, `optimized_window_sensitivity` |
| Potential density with transform `1/Φ(s)` (Sec. V.B, App. A.2, `eq:potential`) | `comparison.event_energy.potential_density`, `energy_posterior` |

The transport core (Sec. II, App. A) and the range to threshold on which every
effective area rests are implemented as printed and are the best-covered parts
of the paper.

---

## 2. Former physics gaps, now closed (with caveats)

The four items below were added after the transport core, against an earlier
draft of the paper whose appendices derived them. None was a drop-in
transcription. Each required either an interpretive choice or a deliberate
departure from a derivation route that turned out to be numerically unstable,
and the current paper has since dropped three of the four derivations, keeping
only the statement each rests on. The choices are documented in the referenced
modules' docstrings. Read this section as "implemented, and here is what that
means against the paper as it stands now".

### 2.1 Parent-neutrino attenuation, `R_ν(x, A)`, implemented as Form C

The modified effective range `R_ν(x,A) = (e^{-x/Λ_ν} - e^{-xΦ(A)}) / (Φ(A) - 1/Λ_ν)`
(Sec. III.A, `eq:Rnu`, derived in App. A by variation of constants) is
implemented literally. `transport.attenuation.neutrino_interaction_length_km`
gives `Λ_ν` in the same km⁻¹ convention `Φ(A)` uses, and
`transport.soft_volume.saturation_factor` takes an optional `inv_lambda_per_km`
argument (default `0`, so every existing caller is unaffected) that computes
`R_ν` directly, including the L'Hopital limit at the removable singularity
`Φ(A) = 1/Λ_ν`. `transport.soft_volume.soft_volume_attenuated_exact` assembles
the coupled rate, with the detail that in-detector production (at depth `x`
exactly) picks up a flat `e^{-x/Λ_ν}` survival factor while upstream production
(`0 ≤ ξ ≤ x`) picks up the integrated `R_ν`. The paper's channel volumes
(App. A.4, `eq:vsoftmu`, `eq:vsofttau`) write the same attenuation inside the
depth integral. `response.soft_volume.SoftVolumeResponse.expected_counts_coupled_attenuation`
is the response-layer entry point ("Form C", alongside the decoupled Forms A
and B). This is the one item of the four that matches the paper's derivation
without a scope-narrowing substitution, and `tests/test_event_rate.py` holds it
against a hand calculation.

**Key realization along the way:** the column depth `x` used by the
muon-transport saturation factor and the column depth feeding the survival
factor `D_ν` were always the same declination-dependent quantity
([the transport-exponent notes](exact_soft_volume.md) §5 and §6: "IceCube sits
~1.95 km deep" for downgoing, the full Earth chord for upgoing). Forms A and B
compute them as two independent numbers. Form C uses one, and takes it from
the layered PREM chord (`transport.earth.prem_column`).

### 2.2 Scale breaking: a running spectral index the paper no longer prints

The paper now treats the photonuclear rise `E^β` in one place. It shifts the
spectral argument instead of spoiling the diagonalization (App. A,
`eq:shiftop`), every transport formula holds for the frozen kernel, and the
running enters by applying the frozen results locally, which App. A.2 does for
the range (`eq:Lrun`). An earlier draft went further and expanded the shift
operator into a series with a leading-order running index. That appendix is
gone.

The code keeps both routes. The paper's route is
`transport.muon_range.stochastic_muon_range_km` with
`kernel_evaluation="running"`, the default, and
`response.effective_area.truncated_range_km` exposes the same switch for the
tabulated areas. The earlier draft's route is
`transport.eigenvalue.phi_eigenvalue_derivative` (the trigamma derivative of
the two-moment exponent) and
`transport.soft_volume.scale_breaking_saturation_factor`, which substitute the
first-order running index `s(ℓ) ≈ A + βℓΦ(A)` into the propagator and integrate
the Gaussian-modified exponent by quadrature. `soft_volume_exact` takes a
`beta` parameter, `0` by default, and `tests/test_eigenvalue.py` checks the
`β = 0` limit and the quadrature. The full series of the earlier draft was never
implemented, because that draft left its source term undefined for a single
power-law mode, and the current paper does not carry the series, so there is
nothing left to catch up to. No calibrated nonzero `β` ships with the package.
`transport.coefficients` records that the LPM suppression of muon radiative
losses is negligible in the covered range, and no photonuclear exponent has
been asserted.

### 2.3 Cutoff sources: real-space convolution, with no series left in the paper

Sec. II.C states that a source expressible as a sum of power laws,
exponential cutoffs included, evolves mode by mode, and that is the paper's
whole treatment of cutoffs. An earlier draft derived a Cahen-Mellin pole
series for an exponentially cut source and recommended it as the mitigation
for the `A < 0` blowup of the transparent-Earth factor. That appendix is gone.

The code implements the cutoff a different way.
`transport.cutoff_source.cutoff_soft_rate_density` and
`response.soft_volume.SoftVolumeResponse.differential_rate_with_cutoff`
evaluate the physical convolution directly in energy space, using the
log-loss density of `transport.loss_distribution.loss_density` (a bounded
probability density with no exponentially large intermediates), integrated
over the upstream production depth. `tests/test_cutoff_source.py` checks that
the result stays finite, positive and monotonic at `A < 0` with `x = 10⁴` km
of column, and that it reduces to the plain treatment as `E_0 → ∞`. The
module docstrings record that a first attempt at the literal pole series
overflowed to `NaN` in exactly that regime, which is why the convolution route
was taken (unverified: that attempt is not in the repository, and the series
is no longer in the paper). The convolution is markedly slower than the rest
of the package (an FFT-style inversion per production-depth slice), so it is
an explicit opt-in method and not part of any default path.

### 2.4 Dark matter lines: the line-of-sight formula and the `s = 0` limit

An earlier draft carried an appendix on monochromatic sources with a
line-of-sight formula at general Mellin index `s`. The current paper mentions
monochromatic lines only in Sec. II.C, as one of the sources that evolve mode
by mode, and Sec. IV notes that a table indexed by `E_ν` carries no spectral
index, so `Φ(0) = 0` and the depth integral collapses onto the mean
first-passage depth.

`scripts/2026_muon_transport/34_dm_line_sensitivity.py` still implements that
earlier formula in `dm_line_flux_general_s`, at general `s`. At the physical
value `s = 0` (a line has no continuum index to average over) `Φ(0) = 0`
identically, the line-of-sight propagator `e^{-ℓΦ(0)}` is `1` for every `ℓ`,
and the formula collapses to the plain, unattenuated J-factor integral that
`los_j_factor` computes. The script's `main()` checks this numerically
(`max |general-s formula at s = 0 / plain J-factor - 1|`, printed each run)
instead of asserting it in prose. That part of the original gap was benign.

What changed in the library: `transport.soft_volume.dm_line_target_volume_km3`
is the literal `s → 0` instance of the detector-side soft-volume formalism
(`Φ(0) = 0` makes the near-detector effective range grow linearly in the
available column instead of logarithmically, unlike the muon-range
convention), plotted in the script beside, not in place of, the
`SoftVolumeResponse.threshold_effective_area_cm2` muon-range curve. The paper
never discussed this near-detector piece, so it is the item with the least
paper backing among the four, and it is kept as a separately labeled
comparison.

---

## 3. Places the code goes beyond the paper

These are not discrepancies, just capability the paper does not cover, noted
so they are not mistaken for implementing a paper equation that does not
exist:

- **`tau_to_muon_ratio`** (`transport.tau.tau_to_muon_ratio`), a closed-form
  population-averaged `N_τ→μ / N_νμ` ratio with the bracket
  `[1 + ℓ_τ(qE_μ) Φ(A)]`. It is consistent with the two-channel rate of
  Sec. III.C (`eq:totalrate`) and the channel volumes of App. A.4, but the
  ratio itself is the codebase's own construction. The tau helpers of
  `response.soft_volume` (`tau_induced_differential_rate` and its
  `expected_counts` variants) are built on it.
- **Clipped dynamic projected area**
  (`transport.soft_volume.dynamic_projected_radius_km`,
  `dynamic_projected_area_km2`), `R_eff(E) = R_det + L max[0, ln(E/E_c)]`,
  with the growth switched on at the critical energy `E_c`
  (`transport.coefficients.critical_energy_gev`). This is the precursor of the
  paper's reach (`eq:reach`), which the code implements separately as
  `light_reach_radius_km` with a signed reach and a fitted pivot. The clipped
  form has no counterpart in the paper. Its growth length `L` was fitted in
  `scripts/2026_muon_transport/26_dynamic_response_effective_area.py`
  (`fit_light_yield_length_km`) against the horizon-band residual of the
  static-footprint model above `E_c`. That script's baseline comparison, an
  earlier example 20, no longer ships.
- **Analytic continuation of the saturation factor to `Φ(A) < 0`**
  (`transport.soft_volume.saturation_factor`), which continues
  `(1 - e^{-Φx})/Φ` to `(e^{|Φ|x} - 1)/|Φ|` past the cross-section pole. The
  current paper discusses only the sign change of the truncated form at
  `A_die` (Sec. II.D, App. A.3), not the pole of the exact form, which
  [the transport-exponent notes](exact_soft_volume.md) §7 treat from an
  earlier draft.
- **Finite-column range** (`transport.muon_range.truncated_muon_range_km`,
  `response.effective_area.truncated_range_km`), the first-passage range cut
  at the available upstream column, `E[τ(w) ∧ X]`, from a gamma law matched to
  the two range moments. The paper carries the available column as an
  instrument input (App. B, `tab:inputs`) and states the range as the
  infinite-column expectation.
- **The running-index and cutoff machinery of §2.2 and §2.3**, and the
  `s = 0` target volume of §2.4, which the current paper no longer derives.

Two items an earlier version of this page listed here have since become the
paper's own inputs. The tabulated BGR18 cross section
(`transport.cross_section.bgr18_cross_section`, `TabulatedCrossSection`) is
what Sec. IV and App. B fix the physics at, with the power law `σ_νN ∝ E^λ` of
Sec. II.C (`PowerLawCrossSection`) kept as the idealization that defines `A`.
The layered PREM column (`transport.earth.prem_column`, `prem_density`,
re-exported by `transport.attenuation`) is the column the regeneration ladder
(Sec. III.D) and the one-line response (Sec. IV.C) take along each chord.

---

## 4. Summary table

| Paper section | Status in `softpaws` |
|---|---|
| Sec. II.A, App. A (transport equation, diagonalization, Laplace exponent, shift operator) | Implemented (`transport.eigenvalue`, `transport.loss_distribution`). The shift operator enters only through the running-index correction of §2.2. |
| Sec. II.B, App. A.1 (calibrated kernel, two and three moments) | Implemented (`transport.eigenvalue`, `transport.coefficients`) |
| Sec. II.C (power-law sources, `A`, `I(A)`) | Implemented (`transport.source`, `transport.eigenvalue.spectral_index`) |
| Sec. II.D, App. A.3 (drift-diffusion expansion as a limit, identities, `A_die`) | Implemented and reproduced numerically (`phi_fokker_planck`, `phi_drift`, `soft_volume_drift`), see §1 above and [the transport-exponent notes](exact_soft_volume.md) §8 |
| Sec. II.E, II.F, App. A.2 (range to threshold, running kernel, splice, two media, variance, validation against PROPOSAL) | Implemented (`transport.muon_range`, `transport.coefficients.log_loss_moments`). The PROPOSAL range comparison is `scripts/2026_muon_transport/39_range_moment_estimator.py`. |
| Sec. III.A (effective volume, `R_ν`) | Implemented as printed ("Form C", §2.1) |
| Sec. III.B, App. B.1 (reach ansatz, predicted from the optics) | Implemented (`transport.soft_volume.light_reach_radius_km`, `response.light_reach`, `response.reduced`) |
| Sec. III.C, App. A.4 (tau-induced tracks) | Implemented (`transport.tau`, `response.effective_area.ic_effective_area_tau_channel`, `arca_effective_area`) |
| Sec. III.D (regeneration ladder, flavour transmission) | Implemented (`transport.attenuation.regenerated_transmission`, `flavour_transmission`) |
| Sec. IV, IV.C, App. B (tabulated effective areas, two instrument numbers per site, one-line response) | Implemented (`response.effective_area`, `response.site_models`, `response.reduced`, `response.first_principles`, `detectors`) |
| Sec. IV.A (loss-model error budget) | Implemented (`transport.loss_ensemble`) |
| Sec. IV.B, IV.D, App. C (declination bands, the event sample) | Implemented (`response.declination`, `response.irfs`, `comparison.rates`, `comparison.events`), with scripts 46, 74 and 76 |
| Sec. V.A, App. C.1 (point-source limit) | Implemented (`response.sensitivity`), with script 84 |
| Sec. V.B (energy of KM3-230213A, potential density) | Implemented (`comparison.event_energy`), with script 57 |
| Sec. V.C, App. D (tension, ingredient ladder) | Implemented (`comparison.likelihood`, `comparison.event_energy`), with scripts 57, 71 and 75 |
