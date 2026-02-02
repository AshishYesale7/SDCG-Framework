#!/usr/bin/env python3
"""
CGC Real Data Analysis - Publication Quality Run
"""

from cgc.mcmc import run_mcmc
from cgc.data_loader import load_real_data
import numpy as np

print('='*70)
print('CGC REAL DATA ANALYSIS - PUBLICATION QUALITY RUN')
print('='*70)

# Load real cosmological data
print('\nLoading real cosmological data...')
data = load_real_data(verbose=True)

print('\nRunning MCMC with real data...')
print('  - 3000 steps x 32 walkers = 96,000 likelihood evaluations')
print('  - This will take ~30 seconds...\n')

sampler, chains = run_mcmc(
    data, 
    n_steps=3000,      # More steps for convergence
    n_walkers=32, 
    save_chains=True, 
    verbose=True,
    include_sne=False,
    include_lyalpha=False
)

# Compute proper statistics
print('\n' + '='*70)
print('FINAL PARAMETER CONSTRAINTS (Real Data)')
print('='*70)

param_names = ['omega_b', 'omega_cdm', 'h', 'ln10As', 'n_s', 'tau',
               'mu', 'n_g', 'z_trans', 'rho_thresh']

print('\n{:12s}  {:20s}  {:20s}'.format('Parameter', 'Mean +/- Std', '68% CI'))
print('-'*60)

for i, name in enumerate(param_names):
    mean = np.mean(chains[:, i])
    std = np.std(chains[:, i])
    low = np.percentile(chains[:, i], 16)
    high = np.percentile(chains[:, i], 84)
    print('{:12s}  {:8.4f} +/- {:8.4f}  [{:7.4f}, {:7.4f}]'.format(
        name, mean, std, low, high))

# CGC-specific summary
print('\n' + '='*70)
print('CGC THEORY CONSTRAINTS')
print('='*70)

mu_mean, mu_std = np.mean(chains[:, 6]), np.std(chains[:, 6])
ng_mean, ng_std = np.mean(chains[:, 7]), np.std(chains[:, 7])
zt_mean, zt_std = np.mean(chains[:, 8]), np.std(chains[:, 8])

print(f'\n  mu (CGC coupling)     = {mu_mean:.4f} +/- {mu_std:.4f}')
print(f'  n_g (spectral index)  = {ng_mean:.4f} +/- {ng_std:.4f}')
print(f'  z_trans               = {zt_mean:.3f} +/- {zt_std:.3f}')

# Detection significance
mu_sigma = abs(mu_mean) / mu_std
print(f'\n  mu detection significance: {mu_sigma:.1f} sigma')

if mu_sigma > 3:
    print('  >> Strong evidence for CGC (>3 sigma)')
elif mu_sigma > 2:
    print('  >> Moderate evidence for CGC (2-3 sigma)')
else:
    print('  >> Weak/no evidence for CGC (<2 sigma)')

# Save results
np.savez('results/cgc_real_analysis_final.npz',
         chains=chains,
         param_names=param_names)
print('\nResults saved to results/cgc_real_analysis_final.npz')
