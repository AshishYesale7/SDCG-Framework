#!/usr/bin/env python3
"""
CGC Full Analysis - Publication Quality Run
============================================
- 10,000 MCMC steps for robust constraints
- All probes: CMB + BAO + Growth + H0 + SNe + Lyman-α
- Full diagnostic plots
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Create plots directory
os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

print('='*70)
print('CGC FULL ANALYSIS - PUBLICATION QUALITY RUN')
print('='*70)

# Import CGC modules
from cgc.mcmc import run_mcmc, print_physics_validation
from cgc.data_loader import load_real_data
from cgc.parameters import CGCParameters

# =============================================================================
# 1. LOAD ALL DATA
# =============================================================================

print('\n' + '='*70)
print('STEP 1: LOADING ALL COSMOLOGICAL DATA')
print('='*70)

data = load_real_data(
    verbose=True,
    include_sne=True,      # Include Pantheon+ SNe
    include_lyalpha=True   # Include Lyman-α
)

# =============================================================================
# 2. RUN FULL MCMC (10,000 steps)
# =============================================================================

print('\n' + '='*70)
print('STEP 2: RUNNING FULL MCMC (10,000 steps)')
print('='*70)

print('\nConfiguration:')
print('  - Steps:   10,000')
print('  - Walkers: 32')
print('  - Total:   320,000 likelihood evaluations')
print('  - Est. time: ~2-3 minutes\n')

sampler, chains = run_mcmc(
    data,
    n_steps=10000,
    n_walkers=32,
    save_chains=True,
    verbose=True,
    include_sne=True,
    include_lyalpha=True
)

# =============================================================================
# 3. PARAMETER CONSTRAINTS
# =============================================================================

print('\n' + '='*70)
print('STEP 3: PARAMETER CONSTRAINTS')
print('='*70)

param_names = ['omega_b', 'omega_cdm', 'h', 'ln10As', 'n_s', 'tau',
               'mu', 'n_g', 'z_trans', 'rho_thresh']

param_labels = [r'$\omega_b$', r'$\omega_{cdm}$', r'$h$', r'$\ln(10^{10}A_s)$', 
                r'$n_s$', r'$\tau$', r'$\mu$', r'$n_g$', r'$z_{trans}$', r'$\rho_{thresh}$']

# Compute statistics
results = {}
print('\n{:15s} {:>12s} {:>12s} {:>20s}'.format(
    'Parameter', 'Mean', 'Std', '68% CI'))
print('-'*65)

for i, name in enumerate(param_names):
    mean = np.mean(chains[:, i])
    std = np.std(chains[:, i])
    low = np.percentile(chains[:, i], 16)
    median = np.percentile(chains[:, i], 50)
    high = np.percentile(chains[:, i], 84)
    
    results[name] = {
        'mean': mean, 'std': std, 'median': median,
        'low': low, 'high': high
    }
    
    print('{:15s} {:12.4f} {:12.4f} [{:8.4f}, {:8.4f}]'.format(
        name, mean, std, low, high))

# =============================================================================
# 4. GENERATE PLOTS
# =============================================================================

print('\n' + '='*70)
print('STEP 4: GENERATING PLOTS')
print('='*70)

# Try to import corner for corner plots
try:
    import corner
    HAS_CORNER = True
except ImportError:
    print('Installing corner for triangle plots...')
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'corner', '-q'])
    import corner
    HAS_CORNER = True

# -----------------------------------------------------------------------------
# Plot 1: Corner plot (all parameters)
# -----------------------------------------------------------------------------
print('\n  [1/6] Generating full corner plot...')

fig = corner.corner(
    chains,
    labels=param_labels,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 10},
    label_kwargs={"fontsize": 12}
)
fig.suptitle('CGC Full Parameter Constraints (Real Data)', fontsize=14, y=1.02)
plt.savefig('plots/corner_full.png', dpi=150, bbox_inches='tight')
plt.savefig('plots/corner_full.pdf', bbox_inches='tight')
plt.close()
print('       Saved: plots/corner_full.png')

# -----------------------------------------------------------------------------
# Plot 2: CGC parameters only corner plot
# -----------------------------------------------------------------------------
print('  [2/6] Generating CGC parameters corner plot...')

cgc_chains = chains[:, 6:10]
cgc_labels = [r'$\mu$', r'$n_g$', r'$z_{trans}$', r'$\rho_{thresh}$']

fig = corner.corner(
    cgc_chains,
    labels=cgc_labels,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 12},
    label_kwargs={"fontsize": 14},
    color='darkblue'
)
fig.suptitle('CGC Parameter Constraints', fontsize=16, y=1.02)
plt.savefig('plots/corner_cgc.png', dpi=150, bbox_inches='tight')
plt.savefig('plots/corner_cgc.pdf', bbox_inches='tight')
plt.close()
print('       Saved: plots/corner_cgc.png')

# -----------------------------------------------------------------------------
# Plot 3: Trace plots (chain evolution)
# -----------------------------------------------------------------------------
print('  [3/6] Generating trace plots...')

fig, axes = plt.subplots(10, 1, figsize=(12, 16), sharex=True)

for i, (ax, name, label) in enumerate(zip(axes, param_names, param_labels)):
    ax.plot(chains[:, i], alpha=0.5, lw=0.5)
    ax.axhline(np.mean(chains[:, i]), color='r', lw=2, label='Mean')
    ax.set_ylabel(label, fontsize=10)
    ax.legend(loc='upper right', fontsize=8)

axes[-1].set_xlabel('Sample', fontsize=12)
fig.suptitle('MCMC Trace Plots', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('plots/trace_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print('       Saved: plots/trace_plots.png')

# -----------------------------------------------------------------------------
# Plot 4: Posterior distributions (1D)
# -----------------------------------------------------------------------------
print('  [4/6] Generating posterior distributions...')

fig, axes = plt.subplots(2, 5, figsize=(16, 6))
axes = axes.flatten()

for i, (ax, name, label) in enumerate(zip(axes, param_names, param_labels)):
    ax.hist(chains[:, i], bins=50, density=True, alpha=0.7, color='steelblue')
    ax.axvline(np.mean(chains[:, i]), color='r', lw=2, label='Mean')
    ax.axvline(np.percentile(chains[:, i], 16), color='r', lw=1, ls='--')
    ax.axvline(np.percentile(chains[:, i], 84), color='r', lw=1, ls='--')
    ax.set_xlabel(label, fontsize=12)
    ax.set_ylabel('Probability', fontsize=10)

plt.suptitle('Posterior Distributions', fontsize=14)
plt.tight_layout()
plt.savefig('plots/posteriors_1d.png', dpi=150, bbox_inches='tight')
plt.close()
print('       Saved: plots/posteriors_1d.png')

# -----------------------------------------------------------------------------
# Plot 5: μ vs n_g contour (key CGC plot)
# -----------------------------------------------------------------------------
print('  [5/6] Generating μ vs n_g contour plot...')

fig, ax = plt.subplots(figsize=(8, 6))

# 2D histogram
H, xedges, yedges = np.histogram2d(chains[:, 6], chains[:, 7], bins=50)
X, Y = np.meshgrid((xedges[:-1] + xedges[1:])/2, (yedges[:-1] + yedges[1:])/2)

# Contour levels (68%, 95%, 99%)
levels = np.array([0.01, 0.05, 0.32, 1.0]) * H.max()
ax.contourf(X, Y, H.T, levels=levels, cmap='Blues', alpha=0.8)
ax.contour(X, Y, H.T, levels=levels[:-1], colors='darkblue', linewidths=1)

# Mark best fit
ax.scatter(np.mean(chains[:, 6]), np.mean(chains[:, 7]), 
           c='red', s=100, marker='*', label='Best fit', zorder=10)

# Mark ΛCDM (μ=0)
ax.axvline(0, color='gray', ls='--', lw=2, label=r'$\Lambda$CDM ($\mu=0$)')

ax.set_xlabel(r'$\mu$ (CGC coupling)', fontsize=14)
ax.set_ylabel(r'$n_g$ (spectral index)', fontsize=14)
ax.set_title(r'CGC Constraints: $\mu$ vs $n_g$', fontsize=14)
ax.legend(fontsize=12)

plt.tight_layout()
plt.savefig('plots/mu_vs_ng.png', dpi=150, bbox_inches='tight')
plt.savefig('plots/mu_vs_ng.pdf', bbox_inches='tight')
plt.close()
print('       Saved: plots/mu_vs_ng.png')

# -----------------------------------------------------------------------------
# Plot 6: CGC detection significance
# -----------------------------------------------------------------------------
print('  [6/6] Generating detection significance plot...')

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# μ histogram with significance
ax = axes[0]
ax.hist(chains[:, 6], bins=50, density=True, alpha=0.7, color='steelblue')
ax.axvline(0, color='red', lw=2, ls='--', label=r'$\Lambda$CDM ($\mu=0$)')
ax.axvline(np.mean(chains[:, 6]), color='darkblue', lw=2, label='CGC best fit')
mu_mean = np.mean(chains[:, 6])
mu_std = np.std(chains[:, 6])
ax.set_xlabel(r'$\mu$', fontsize=14)
ax.set_ylabel('Probability', fontsize=12)
ax.set_title(f'$\\mu = {mu_mean:.3f} \\pm {mu_std:.3f}$\n({abs(mu_mean/mu_std):.1f}$\\sigma$ from $\\Lambda$CDM)', fontsize=12)
ax.legend(fontsize=10)

# z_trans vs prediction
ax = axes[1]
ax.hist(chains[:, 8], bins=50, density=True, alpha=0.7, color='steelblue')
ax.axvline(1.67, color='orange', lw=2, ls='--', label=r'EFT prediction ($z_{acc} + \Delta z$)')
ax.axvline(np.mean(chains[:, 8]), color='darkblue', lw=2, label='Fitted')
zt_mean = np.mean(chains[:, 8])
zt_std = np.std(chains[:, 8])
ax.set_xlabel(r'$z_{trans}$', fontsize=14)
ax.set_title(f'$z_{{trans}} = {zt_mean:.2f} \\pm {zt_std:.2f}$', fontsize=12)
ax.legend(fontsize=10)

# n_g vs EFT prediction
ax = axes[2]
ax.hist(chains[:, 7], bins=50, density=True, alpha=0.7, color='steelblue')
ax.axvline(0.014, color='orange', lw=2, ls='--', label=r'EFT: $\beta_0^2/4\pi^2$')
ax.axvline(np.mean(chains[:, 7]), color='darkblue', lw=2, label='Fitted')
ng_mean = np.mean(chains[:, 7])
ng_std = np.std(chains[:, 7])
ax.set_xlabel(r'$n_g$', fontsize=14)
ax.set_title(f'$n_g = {ng_mean:.3f} \\pm {ng_std:.3f}$', fontsize=12)
ax.legend(fontsize=10)

plt.suptitle('CGC Parameter Constraints vs Predictions', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('plots/cgc_significance.png', dpi=150, bbox_inches='tight')
plt.savefig('plots/cgc_significance.pdf', bbox_inches='tight')
plt.close()
print('       Saved: plots/cgc_significance.png')

# =============================================================================
# 5. FINAL SUMMARY
# =============================================================================

print('\n' + '='*70)
print('ANALYSIS COMPLETE')
print('='*70)

# CGC detection significance
mu_mean = results['mu']['mean']
mu_std = results['mu']['std']
mu_sigma = abs(mu_mean) / mu_std

print('\n' + '-'*70)
print('CGC DETECTION SUMMARY')
print('-'*70)
print(f'\n  μ (CGC coupling)     = {mu_mean:.4f} ± {mu_std:.4f}')
print(f'  n_g (spectral index) = {results["n_g"]["mean"]:.4f} ± {results["n_g"]["std"]:.4f}')
print(f'  z_trans              = {results["z_trans"]["mean"]:.3f} ± {results["z_trans"]["std"]:.3f}')
print(f'\n  DETECTION SIGNIFICANCE: {mu_sigma:.1f}σ')

if mu_sigma > 5:
    print('  ★★★ DISCOVERY-LEVEL EVIDENCE FOR CGC (>5σ) ★★★')
elif mu_sigma > 3:
    print('  ★★ STRONG EVIDENCE FOR CGC (>3σ) ★★')
elif mu_sigma > 2:
    print('  ★ MODERATE EVIDENCE FOR CGC (2-3σ) ★')
else:
    print('  No significant evidence for CGC (<2σ)')

print('\n' + '-'*70)
print('OUTPUT FILES')
print('-'*70)
print('\n  Results:')
print('    - results/cgc_real_analysis_final.npz')
print('    - results/chains/mcmc_chains.npz')
print('\n  Plots:')
print('    - plots/corner_full.png (all parameters)')
print('    - plots/corner_cgc.png (CGC parameters only)')
print('    - plots/trace_plots.png (chain evolution)')
print('    - plots/posteriors_1d.png (1D posteriors)')
print('    - plots/mu_vs_ng.png (key CGC contour)')
print('    - plots/cgc_significance.png (detection summary)')

# Save final results
np.savez('results/cgc_full_analysis.npz',
         chains=chains,
         param_names=param_names,
         results=results)

print('\n' + '='*70)
print('ALL DONE!')
print('='*70)
