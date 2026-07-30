import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# ==========================================
# Parameters & Constants
# ==========================================
x_depth = 1e4        # Total column depth (km.w.e.)
A_index = 0.98       # Effective spectral index A (e.g., IceCube diffuse fit)
Phi_A = 0.343        # Loss exponent Phi(A) in km^-1 for A=0.98
A_0 = 1.0            # Baseline projected area (km^2)

# Energy range for the plot (GeV)
energies = jnp.logspace(3, 9, 100)

# ==========================================
# 1. Paper Formulation (Static Area)
# ==========================================
# V_eff = A_proj * [1 - exp(-x * Phi(A))] / Phi(A)
def calc_v_eff_paper(Phi, x, A_proj):
    return A_proj * (1.0 - jnp.exp(-x * Phi)) / Phi

# The paper's effective volume is independent of E if Phi(A) is fixed
v_eff_paper = jnp.full_like(energies, calc_v_eff_paper(Phi_A, x_depth, A_0))

# ==========================================
# 2. Hybrid Formulation (Dynamic Response)
# ==========================================
# Toy model for W(E, l): Acceptance increases with residual energy.
# We assume initial energy E decays over distance l roughly as E_res = E * exp(-l * Phi_A)
# The trigger area grows logarithmically with this residual energy above a threshold.
def W_response(E, l, Phi, A_base, E_threshold=1e3, growth_rate=0.1):
    E_res = E * jnp.exp(-l * Phi)
    # Area enhancement kicks in for energies above E_threshold
    enhancement = jnp.where(E_res > E_threshold, growth_rate * jnp.log(E_res / E_threshold), 0.0)
    return A_base * (1.0 + enhancement)

# Integration of W(E, l) * exp(-l * Phi_A) over dl
@jax.jit
def calc_v_eff_hybrid(E, Phi, x, A_base):
    # Setup integration grid for distance l
    l_vals = jnp.linspace(0, x, 1000)
    dl = l_vals[1] - l_vals[0]
    
    # Compute integrand: W(E, l) * Green's Function
    integrand = W_response(E, l_vals, Phi, A_base) * jnp.exp(-l_vals * Phi)
    
    # Trapezoidal rule integration
    return jnp.trapezoid(integrand, dx=dl)

# Vectorize over the energy array
vmap_hybrid = jax.vmap(calc_v_eff_hybrid, in_axes=(0, None, None, None))
v_eff_hybrid = vmap_hybrid(energies, Phi_A, x_depth, A_0)

# ==========================================
# Plotting
# ==========================================
plt.figure(figsize=(8, 6))

plt.plot(energies, v_eff_paper, color='black', linestyle='--', linewidth=2, 
         label='Analytic Paper Model ($W = A_0$)')
plt.plot(energies, v_eff_hybrid, color='purple', linewidth=2, 
         label='Hybrid Transport Model ($W(E, \ell)$)')

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Muon Energy $E$ (GeV)', fontsize=12)
plt.ylabel(r'Effective Volume $V_{\rm eff}$ (km$^3$.w.e.)', fontsize=12)
plt.title('Comparison of Effective Volume Predictions', fontsize=14)
plt.legend(fontsize=11, loc='upper left')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()

# Display the generated figure
plt.savefig('effective_volume_comparison.png', dpi=300)