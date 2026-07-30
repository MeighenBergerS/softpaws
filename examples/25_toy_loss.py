import numpy as np
import matplotlib.pyplot as plt

# Constants and grid for fractional energy loss y
y = np.logspace(-4, 0, 500)
E_mu = 1e6  # Muon energy in GeV (1 PeV)

# 1. Toy/Parametrized Differential Cross-sections dsigma/dy (approximate shapes)
# e+e- pair production (dominant at low y, ~ (1-y)/y)
dsigma_pp = (1.0 / y) * (1.0 - y + 0.75 * y**2) * 1e-6 

# Bremsstrahlung (~ 1/y behavior with typical QED form)
dsigma_brem = (4.0 / 3.0) * (1.0 - y + 0.75 * y**2) / y * 1e-7

# Photonuclear (flat-ish or rising slightly with energy)
dsigma_pn = 0.5 * (1.0 - y)**2 / y * 1e-7

dsigma_total = dsigma_pp + dsigma_brem + dsigma_pn

# 2. Plotting Top Panel: Energy Loss Distributions dsigma/dy
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

ax1.loglog(y, dsigma_total, label='Total', color='black', lw=2)
ax1.loglog(y, dsigma_pp, label='$e^+e^-$ Pair Production', color='green', ls='--')
ax1.loglog(y, dsigma_brem, label='Bremsstrahlung', color='red', ls=':')
ax1.loglog(y, dsigma_pn, label='Photonuclear', color='purple', ls='-.')

ax1.set_xlabel(r'Fractional Energy Loss $y$', fontsize=12)
ax1.set_ylabel(r'${\rm d}\sigma/{\rm d}y$ [arbitrary units]', fontsize=12)
ax1.set_title(r'Energy Loss Distributions ($E_\mu = 1$ PeV in water)', fontsize=14)
ax1.legend(loc='upper right')
ax1.grid(True, which="both", ls="--", alpha=0.5)

# 3. Plotting Bottom Panel: Transport Coefficients b_mu and d_mu as function of E
energies = np.logspace(3, 9, 50) # 1 TeV to 1 EeV
# Parametrizing standard logarithmic rise from radiative dominance + photonuclear rise
b_mu = 2.0 + 0.1 * np.log(energies / 1e3) 
d_mu = 0.45 + 0.05 * np.log(energies / 1e3)

ax2.semilogx(energies, b_mu, label='Drift Coefficient $b_\mu$', color='blue', lw=2)
ax2.semilogx(energies, d_mu, label='Diffusion Coefficient $d_\mu$', color='orange', lw=2, ls='--')

ax2.set_xlabel(r'Muon Energy $E_\mu$ (GeV)', fontsize=12)
ax2.set_ylabel(r'Transport Coefficients [km$^{-1}$]', fontsize=12)
ax2.set_title('Drift and Diffusion Coefficients vs. Energy', fontsize=14)
ax2.legend(loc='upper left')
ax2.grid(True, which="both", ls="--", alpha=0.5)

plt.tight_layout()
plt.savefig('energy_loss_distributions.png', dpi=300)