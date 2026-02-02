#!/usr/bin/env python3
"""Verify first-principles parameters."""

from cgc.parameters import (
    BETA_0, N_G_FROM_BETA, Z_TRANS_DERIVED, ALPHA_SCREENING,
    RHO_THRESH_DEFAULT, MU_NAIVE, MU_LYALPHA_LIMIT, MU_BEST_FIT
)

print('='*70)
print('SDCG FIRST-PRINCIPLES PARAMETERS')
print('='*70)
print()
print(f'beta_0 = {BETA_0:.3f}  (from SM conformal anomaly)')
print(f'n_g = {N_G_FROM_BETA:.5f}  (from RG flow: beta_0^2/4pi^2)')
print(f'z_trans = {Z_TRANS_DERIVED:.2f}  (from cosmic evolution)')
print(f'alpha = {ALPHA_SCREENING:.1f}  (from V(phi) ~ phi^-1)')
print(f'rho_thresh = {RHO_THRESH_DEFAULT:.0f} rho_crit  (from virial equilibrium)')
print()
print('THE mu PROBLEM:')
print(f'  Naive RG: mu = {MU_NAIVE:.2f}  (way too large)')
print(f'  Lya limit: mu < {MU_LYALPHA_LIMIT}')
print(f'  Best fit: mu = {MU_BEST_FIT}')
print()
print('  -> REQUIRES NEW PHYSICS AT meV SCALE')
