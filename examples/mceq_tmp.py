#!/usr/bin/env python3
"""
Downgoing (vertical, theta = 0 deg) atmospheric muon-neutrino flux at IceCube,
computed with MCEq (https://github.com/mceq-project/MCEq).

Shows the total flux together with the pion, kaon and prompt (charm) parent
contributions, scaled by E^3.

Requires: MCEq, crflux, numpy, matplotlib
    pip install MCEq
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import crflux.models as pm
from MCEq.core import MCEqRun

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
THETA_DEG = 0.0          # 0 deg = vertically downgoing at the detector
INTERACTION_MODEL = "SIBYLL23D"
PRIMARY_MODEL = (pm.HillasGaisser2012, "H3a")
ATMOSPHERE = ("CORSIKA", ('USStd', None))
OUTFILE = "icecube_downgoing_numu.png"

# Set to True if you have a working LaTeX installation and want real LaTeX
# typesetting; otherwise matplotlib's mathtext renders the same expressions.
USE_TEX = False

COL_TOTAL = "black"
COL_PION = "#e7298a"
COL_KAON = "#1b9e77"
COL_PROMPT = "#e6ab02"

FS = 8  # global font size

# ----------------------------------------------------------------------
# MCEq run
# ----------------------------------------------------------------------
mceq = MCEqRun(
    interaction_model=INTERACTION_MODEL,
    primary_model=PRIMARY_MODEL,
    theta_deg=THETA_DEG,
    density_model=ATMOSPHERE,
)

mceq.solve()

e_grid = mceq.e_grid  # GeV


def flux(prefix, mag=3):
    """nu_mu + nu_mu-bar flux (times E^mag) for a given parent-particle group.

    prefix: '' (total), 'pi_', 'k_' or 'pr_'
    """
    names = ("numu", "antinumu")
    if prefix == "":
        keys = ["total_" + n for n in names]
    else:
        keys = [prefix + n for n in names]
    return sum(mceq.get_solution(k, mag) for k in keys)


total = flux("")
pion = flux("pi_")
kaon = flux("k_")
prompt = flux("pr_")

# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
plt.rcParams.update(
    {
        "text.usetex": USE_TEX,
        "font.family": "serif",
        "font.size": FS,
        "axes.labelsize": FS,
        "axes.titlesize": FS,
        "xtick.labelsize": FS,
        "ytick.labelsize": FS,
        "legend.fontsize": FS,
        "axes.grid": False,
    }
)

fig, ax = plt.subplots(figsize=(3, 3))

ax.loglog(e_grid, total, color=COL_TOTAL, lw=1.2, label=r"Total")
ax.loglog(e_grid, pion, color=COL_PION, lw=1.2, label=r"$\pi^{\pm}$")
ax.loglog(e_grid, kaon, color=COL_KAON, lw=1.2, label=r"$K^{\pm}$")
ax.loglog(e_grid, prompt, color=COL_PROMPT, lw=1.2, label=r"Prompt")

ax.set_xlim(1e1, 1e5)
ax.set_ylim(1e-4, 1e0)

ax.set_xlabel(r"$E_{\nu}\ \mathrm{[GeV]}$")
ax.set_ylabel(
    r"$E_{\nu}^{3}\,\Phi_{\nu_{\mu}+\bar{\nu}_{\mu}}\ "
    r"\mathrm{[GeV^{2}\,cm^{-2}\,s^{-1}\,sr^{-1}]}$"
)

ax.grid(False)
ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=FS)

leg = ax.legend(loc="lower left", frameon=False, fontsize=FS,
                handlelength=1.5, labelspacing=0.3)

ax.text(
    0.95,
    0.95,
    r"$\theta_{\mathrm{zen}} = 0^{\circ}$",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=FS,
)

fig.tight_layout(pad=0.3)
fig.savefig(OUTFILE, dpi=300)
print("wrote", OUTFILE)