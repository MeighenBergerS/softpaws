import numpy as np
from scipy.special import digamma, gamma, polygamma
from scipy.integrate import quad
import matplotlib.pyplot as plt

# =============================================================================
# 1. THE CORE KERNEL (Levy Exponent)
# =============================================================================
# Using the closed-form from the paper: dGamma/dy = kappa * (1-y)^p / y
# Phi(s) = kappa * [psi(p + s + 1) - psi(p + 1)]
# Calibrated for water/ice at 1 PeV: Phi(1) = b_mu = 0.349 km^-1.
# We choose p = 0.5 (effective shape) -> kappa = b_mu * (p+1).
b_mu = 0.349  # km^-1
p = 0.5
kappa = b_mu * (p + 1)  # Ensures Phi(1) = b_mu exactly

def Phi(s):
    """Levy exponent. Handles s=0 gracefully."""
    if np.abs(s) < 1e-12:
        return 0.0
    return kappa * (digamma(p + s + 1) - digamma(p + 1))

# =============================================================================
# 2. RECIPE 1 & 2: DIRECT MUON & TAU-INDUCED SOFT VOLUMES
# =============================================================================
def V_mu(A, x):
    """Effective volume for a direct muon track."""
    ph = Phi(A)
    if np.abs(ph) < 1e-12:
        return x  # geometric limit as Phi -> 0
    return (1.0 - np.exp(-x * ph)) / ph

def V_tau(A, x, l_tau=50.0):
    """Effective volume for a tau -> muon track (assuming tau decays before losing energy)."""
    ph = Phi(A)
    denom = l_tau * ph - 1.0
    if np.abs(denom) < 1e-12:
        # Series expansion for the rare case where l_tau*Phi = 1
        return 0.5 * x * (x / l_tau)  # simplified limit
    term1 = l_tau * (1.0 - np.exp(-x / l_tau))
    term2 = (1.0 - np.exp(-x * ph)) / ph if np.abs(ph) > 1e-12 else x
    return (term1 - term2) / denom

# =============================================================================
# 3. RECIPE 4: DARK MATTER LINE-OF-SIGHT INTEGRAL (NFW Profile)
# =============================================================================
R_sun = 8.5  # kpc

def nfw_profile(r):
    """Simplified NFW: rho(r) = 1 / (r * (1+r)^2) in arbitrary units."""
    if r < 1e-6:
        return 1.0  # central cusp cap
    return 1.0 / (r * (1.0 + r)**2)

def dm_los_integral(s, psi_deg=45.0):
    """
    Computes integral over line-of-sight of e^{-l * Phi(s)} * rho^2(r).
    psi: pointing angle relative to Galactic Center (degrees).
    """
    psi = np.radians(psi_deg)
    def integrand(l):
        r = np.sqrt(R_sun**2 + l**2 - 2 * R_sun * l * np.cos(psi))
        rho2 = nfw_profile(r)**2
        return np.exp(-l * Phi(s)) * rho2
    # Integrate from 0 to 100 kpc (halo edge)
    res, err = quad(integrand, 0, 100, limit=200)
    return res

# =============================================================================
# 4. RECIPE 3: AGN / TRANSIENT WITH EXPONENTIAL CUTOFF (Numerical Inverse Mellin)
# =============================================================================
def flux_with_cutoff(E_obs, x, gamma_nu=2.3, E0=100.0):
    """
    Computes the leptonic flux at observed energy E_obs (PeV)
    for a neutrino source spectrum: dN/dE_nu ∝ E_nu^{-gamma_nu} exp(-E_nu / E0).
    Uses saddle-point approximation to collapse the integral.
    """
    A0 = gamma_nu - 0.4 - 1.0  # approximate A (assuming lambda ~ 0.4)
    ph = Phi(A0)
    
    # CORRECTED LINE: Use polygamma(1, z) for the first derivative of digamma
    dph = kappa * (polygamma(1, p + A0 + 1) - polygamma(1, p + 1))
    
    # Asymptotic expression from the saddle-point derivation
    # Note: x (km) and E0 (PeV) are treated as dimensionless ratios here.
    # If E0 is in PeV, ensure x is scaled appropriately for your units.
    numerator = np.exp(-E_obs / E0) * (1.0 - np.exp(-x * ph))
    denominator = ph * (1.0 - x * dph / E0)
    
    # Avoid division by zero if denominator hits zero (rare)
    if np.abs(denominator) < 1e-12:
        return numerator / (ph * 1e-12)
    return numerator / denominator


# =============================================================================
# 5. EXECUTE & PLOT RESULTS
# =============================================================================
if __name__ == "__main__":
    x_earth = 10.0  # km water equivalent (typical vertical depth)
    
    # ---- A. Scan over spectral index A ----
    A_vals = np.linspace(-1.0, 3.0, 200)
    Phi_vals = [Phi(A) for A in A_vals]
    V_mu_vals = [V_mu(A, x_earth) for A in A_vals]
    V_tau_vals = [V_tau(A, x_earth) for A in A_vals]

    print("=" * 60)
    print("NUMERICAL RESULTS FOR KEY SCENARIOS")
    print("=" * 60)
    
    # KM3NeT / IceCube point (A = 0.98)
    A_ic = 0.98
    print(f"\n[1] IceCube Diffuse / KM3NeT UHE (A = {A_ic:.2f}):")
    print(f"    Phi(A)  = {Phi(A_ic):.4f} km^-1")
    print(f"    V_mu     = {V_mu(A_ic, x_earth):.4f} km")
    print(f"    V_tau    = {V_tau(A_ic, x_earth):.4f} km (tau decay length = 50 km)")
    
    # Cross-section pole test (A negative)
    A_neg = -0.5
    print(f"\n[2] High Cross-Section Slope (A = {A_neg:.1f}):")
    print(f"    Phi(A)  = {Phi(A_neg):.4f} km^-1 (NEGATIVE! -> exponential growth)")
    print(f"    V_mu     = {V_mu(A_neg, x_earth):.4f} km (SATURATED by geometric factor)")
    print(f"    (FP would have given negative volume here – our formula stays positive)")

    # Dark Matter Line
    print(f"\n[3] Dark Matter Line-of-Sight Integral (s = 1, psi = 45 deg):")
    dm_val = dm_los_integral(1.0, 45.0)
    print(f"    Integral = {dm_val:.4e}  (arbitrary density units)")

    # AGN Cutoff example
    E_obs = 10.0  # PeV
    print(f"\n[4] AGN Cutoff Flux at E_obs = {E_obs} PeV, x = {x_earth} km:")
    flux_cut = flux_with_cutoff(E_obs, x_earth)
    print(f"    Relative Flux = {flux_cut:.4e} (arbitrary normalization)")

    # ---- B. Plotting ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Phi(s) vs s
    ax1 = axes[0, 0]
    ax1.plot(A_vals, Phi_vals, 'b-', linewidth=2)
    ax1.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax1.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Spectral Index $A$', fontsize=12)
    ax1.set_ylabel('Lévy Exponent $\Phi(A)$ [km$^{-1}$]', fontsize=12)
    ax1.set_title('Exact Transport Kernel', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Soft Volumes (V_mu vs V_tau)
    ax2 = axes[0, 1]
    ax2.plot(A_vals, V_mu_vals, 'r-', label='Direct $\mu$', linewidth=2)
    ax2.plot(A_vals, V_tau_vals, 'g--', label='$\\tau \\to \mu$', linewidth=2)
    ax2.axvline(0.98, color='k', linestyle=':', alpha=0.7, label='IceCube/KM3NeT')
    ax2.set_xlabel('Spectral Index $A$', fontsize=12)
    ax2.set_ylabel('Soft Volume $V_{\mathrm{eff}}$ [km]', fontsize=12)
    ax2.set_title('Direct vs Tau-Induced Tracks ($x=10$ km)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Geometric Saturation (V_mu for different column depths)
    ax3 = axes[1, 0]
    for x_test in [2, 5, 10, 20]:
        V_test = [V_mu(A, x_test) for A in A_vals]
        ax3.plot(A_vals, V_test, label=f'x = {x_test} km')
    ax3.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Spectral Index $A$', fontsize=12)
    ax3.set_ylabel('Soft Volume $V_{\mu}$ [km]', fontsize=12)
    ax3.set_title('Geometric Saturation with Earth Depth', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: AGN Cutoff vs Pure Power-Law
    ax4 = axes[1, 1]
    E_range = np.logspace(-1, 2, 100)  # 0.1 to 100 PeV
    flux_cutoff_vals = [flux_with_cutoff(E, x_earth) for E in E_range]
    # Compare to power-law (A=0.98) shape E^{-1-A} = E^{-1.98}
    E_norm = E_range / E_range[0]
    power_law = E_norm**(-1.98)
    ax4.loglog(E_range, flux_cutoff_vals, 'b-', label='With Cutoff ($E_0=100$ PeV)', linewidth=2)
    ax4.loglog(E_range, power_law * flux_cutoff_vals[0] / power_law[0], 
               'r--', label='Pure Power-Law', alpha=0.7)
    ax4.set_xlabel('Detected Energy $E$ [PeV]', fontsize=12)
    ax4.set_ylabel('Relative Flux', fontsize=12)
    ax4.set_title('AGN Source: Exponential Cutoff Effect', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('neutrino_transport_results.png', dpi=150)
    plt.show()