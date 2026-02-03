#!/usr/bin/env python3
"""
CGC Optimized Full Analysis
============================
Uses optimized settings for practical runtime:
- Core probes: CMB + BAO + Growth + H0 (fast)
- 10,000 steps for robust constraints
- All diagnostic plots
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

print('='*70)
print('CGC OPTIMIZED ANALYSIS - 10,000 STEPS')
print('='*70)

from cgc.mcmc import run_mcmc, print_physics_validation
from cgc.data_loader import load_real_data

# =============================================================================
# 1. LOAD DATA (Core probes for speed)
# =============================================================================

print('\n' + '='*70)
print('STEP 1: LOADING REAL COSMOLOGICAL DATA')
print('='*70)

data = load_real_data(
    verbose=True,
    include_sne=False,      # Skip SNe (slow with 1701 points)
    include_lyalpha=False   # Skip Lyman-α
)

# =============================================================================
# 2. RUN 10,000 STEP MCMC
# =============================================================================

print('\n' + '='*70)
print('STEP 2: RUNNING 10,000 STEP MCMC')
print('='*70)

print('\nConfiguration:')
print('  - Steps:   10,000')
print('  - Walkers: 32')
print('  - Probes:  CMB + BAO + Growth + H0')
print('  - Est. time: ~2 minutes\n')

sampler, chains = run_mcmc(
    data,
    n_steps=10000,
    n_walkers=32,
    save_chains=True,
    verbose=True,
    include_sne=False,
    include_lyalpha=False
)

# =============================================================================
# 3. COMPUTE STATISTICS
# =============================================================================

param_names = ['omega_b', 'omega_cdm', 'h', 'ln10As', 'n_s', 'tau',
               'mu', 'n_g', 'z_trans', 'rho_thresh']

param_labels = [r'$\omega_b$', r'$\omega_{cdm}$', r'$h$', r'$\ln(10^{10}A_s)$', 
                r'$n_s$', r'$\tau$', r'$\mu$', r'$n_g$', r'$z_{trans}$', r'$\rho_{thresh}$']

results = {}
print('\n' + '='*70)
print('STEP 3: PARAMETER CONSTRAINTS')
print('='*70)

print('\n{:15s} {:>12s} {:>12s} {:>20s}'.format('Parameter', 'Mean', 'Std', '68% CI'))
print('-'*65)

for i, name in enumerate(param_names):
    mean = np.mean(chains[:, i])
    std = np.std(chains[:, i])
    low = np.percentile(chains[:, i], 16)
    median = np.percentile(chains[:, i], 50)
    high = np.percentile(chains[:, i], 84)
    
    results[name] = {'mean': mean, 'std': std, 'median': median, 'low': low, 'high': high}
    print('{:15s} {:12.4f} {:12.4f} [{:8.4f}, {:8.4f}]'.format(name, mean, std, low, high))

# =============================================================================
# 4. GENERATE ALL PLOTS
# =============================================================================

print('\n' + '='*70)
print('STEP 4: GENERATING PLOTS')
print('='*70)

try:
    import corner
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'corner', '-q'])
    import corner

# Plot 1: Full corner plot
print('\n  [1/7] Full corner plot...')
fig = corner.corner(chains, labels=param_labels, quantiles=[0.16, 0.5, 0.84],
                    show_titles=True, title_kwargs={"fontsize": 10})
fig.suptitle('CGC Full Parameter Constraints (10,000 steps)', fontsize=14, y=1.02)
plt.savefig('plots/corner_full_10k.png', dpi=150, bbox_inches='tight')
plt.savefig('plots/corner_full_10k.pdf', bbox_inches='tight')
plt.close()
print('       Saved: plots/corner_full_10k.png')

# Plot 2: CGC parameters corner
print('  [2/7] CGC corner plot...')
cgc_chains = chains[:, 6:10]
cgc_labels = [r'$\mu$', r'$n_g$', r'$z_{trans}$', r'$\rho_{thresh}$']
fig = corner.corner(cgc_chains, labels=cgc_labels, quantiles=[0.16, 0.5, 0.84],
                    show_titles=True, title_kwargs={"fontsize": 12}, color='darkblue')
fig.suptitle('CGC Parameter Constraints (10,000 steps)', fontsize=16, y=1.02)
plt.savefig('plots/corner_cgc_10k.png', dpi=150, bbox_inches='tight')
plt.savefig('plots/corner_cgc_10k.pdf', bbox_inches='tight')
plt.close()
print('       Saved: plots/corner_cgc_10k.png')

# Plot 3: Trace plots
print('  [3/7] Trace plots...')
fig, axes = plt.subplots(10, 1, figsize=(14, 18), sharex=True)
for i, (ax, name, label) in enumerate(zip(axes, param_names, param_labels)):
    ax.plot(chains[:, i], alpha=0.3, lw=0.3)
    ax.axhline(np.mean(chains[:, i]), color='r', lw=2)
    ax.axhline(np.percentile(chains[:, i], 16), color='r', lw=1, ls='--')
    ax.axhline(np.percentile(chains[:, i], 84), color='r', lw=1, ls='--')
    ax.set_ylabel(label, fontsize=10)
axes[-1].set_xlabel('Sample', fontsize=12)
fig.suptitle('MCMC Trace Plots (10,000 steps)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('plots/trace_plots_10k.png', dpi=150, bbox_inches='tight')
plt.close()
print('       Saved: plots/trace_plots_10k.png')

# Plot 4: 1D posteriors
print('  [4/7] Posterior distributions...')
fig, axes = plt.subplots(2, 5, figsize=(18, 7))
axes = axes.flatten()
for i, (ax, name, label) in enumerate(zip(axes, param_names, param_labels)):
    ax.hist(chains[:, i], bins=60, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axvline(np.mean(chains[:, i]), color='red', lw=2, label='Mean')
    ax.axvline(np.percentile(chains[:, i], 16), color='red', lw=1, ls='--')
    ax.axvline(np.percentile(chains[:, i], 84), color='red', lw=1, ls='--')
    ax.set_xlabel(label, fontsize=12)
    ax.set_ylabel('Probability', fontsize=10)
plt.suptitle('Posterior Distributions (10,000 steps)', fontsize=14)
plt.tight_layout()
plt.savefig('plots/posteriors_1d_10k.png', dpi=150, bbox_inches='tight')
plt.close()
print('       Saved: plots/posteriors_1d_10k.png')

# Plot 5: μ vs n_g contour
print('  [5/7] mu vs n_g contour...')
fig, ax = plt.subplots(figsize=(10, 8))
H, xedges, yedges = np.histogram2d(chains[:, 6], chains[:, 7], bins=60)
X, Y = np.meshgrid((xedges[:-1] + xedges[1:])/2, (yedges[:-1] + yedges[1:])/2)
levels = np.array([0.01, 0.05, 0.32, 1.0]) * H.max()
ax.contourf(X, Y, H.T, levels=levels, cmap='Blues', alpha=0.8)
ax.contour(X, Y, H.T, levels=levels[:-1], colors='darkblue', linewidths=1.5)
ax.scatter(np.mean(chains[:, 6]), np.mean(chains[:, 7]), c='red', s=150, marker='*', 
           label=f'Best fit: $\\mu$={np.mean(chains[:, 6]):.3f}', zorder=10)
ax.axvline(0, color='gray', ls='--', lw=2, label=r'$\Lambda$CDM ($\mu=0$)')
ax.set_xlabel(r'$\mu$ (CGC coupling)', fontsize=14)
ax.set_ylabel(r'$n_g$ (spectral index)', fontsize=14)
ax.set_title(r'CGC Constraints: $\mu$ vs $n_g$', fontsize=16)
ax.legend(fontsize=12, loc='upper right')
plt.tight_layout()
plt.savefig('plots/mu_vs_ng_10k.png', dpi=150, bbox_inches='tight')
plt.savefig('plots/mu_vs_ng_10k.pdf', bbox_inches='tight')
plt.close()
print('       Saved: plots/mu_vs_ng_10k.png')

# Plot 6: CGC detection significance
print('  [6/7] Detection significance plot...')
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# μ histogram
ax = axes[0]
ax.hist(chains[:, 6], bins=60, density=True, alpha=0.7, color='steelblue', edgecolor='white')
ax.axvline(0, color='red', lw=2, ls='--', label=r'$\Lambda$CDM ($\mu=0$)')
ax.axvline(np.mean(chains[:, 6]), color='darkblue', lw=2, label='CGC best fit')
mu_mean, mu_std = np.mean(chains[:, 6]), np.std(chains[:, 6])
ax.fill_betweenx([0, ax.get_ylim()[1]*1.1], mu_mean-mu_std, mu_mean+mu_std, alpha=0.3, color='blue')
ax.set_xlabel(r'$\mu$', fontsize=14)
ax.set_ylabel('Probability', fontsize=12)
ax.set_title(f'$\\mu = {mu_mean:.4f} \\pm {mu_std:.4f}$\n({abs(mu_mean/mu_std):.1f}$\\sigma$ from $\\Lambda$CDM)', fontsize=13)
ax.legend(fontsize=11)

# z_trans vs prediction
ax = axes[1]
ax.hist(chains[:, 8], bins=60, density=True, alpha=0.7, color='steelblue', edgecolor='white')
ax.axvline(1.64, color='orange', lw=2, ls='--', label=r'EFT: $z_{acc} + \Delta z = 1.64$')
ax.axvline(np.mean(chains[:, 8]), color='darkblue', lw=2, label='Fitted')
zt_mean, zt_std = np.mean(chains[:, 8]), np.std(chains[:, 8])
ax.set_xlabel(r'$z_{trans}$', fontsize=14)
ax.set_title(f'$z_{{trans}} = {zt_mean:.3f} \\pm {zt_std:.3f}$', fontsize=13)
ax.legend(fontsize=11)

# n_g vs EFT prediction
ax = axes[2]
ax.hist(chains[:, 7], bins=60, density=True, alpha=0.7, color='steelblue', edgecolor='white')
ax.axvline(0.014, color='orange', lw=2, ls='--', label=r'EFT: $\beta_0^2/4\pi^2 = 0.014$')
ax.axvline(np.mean(chains[:, 7]), color='darkblue', lw=2, label='Fitted')
ng_mean, ng_std = np.mean(chains[:, 7]), np.std(chains[:, 7])
ax.set_xlabel(r'$n_g$', fontsize=14)
ax.set_title(f'$n_g = {ng_mean:.3f} \\pm {ng_std:.3f}$', fontsize=13)
ax.legend(fontsize=11)

plt.suptitle('CGC Parameter Constraints vs EFT Predictions', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig('plots/cgc_significance_10k.png', dpi=150, bbox_inches='tight')
plt.savefig('plots/cgc_significance_10k.pdf', bbox_inches='tight')
plt.close()
print('       Saved: plots/cgc_significance_10k.png')

# Plot 7: Cosmological parameters comparison
print('  [7/7] Cosmology comparison plot...')
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Planck values for comparison
planck = {'h': (0.674, 0.005), 'omega_cdm': (0.120, 0.001), 'n_s': (0.965, 0.004)}

for ax, param, planck_val in zip(axes, ['h', 'omega_cdm', 'n_s'], 
                                  [planck['h'], planck['omega_cdm'], planck['n_s']]):
    idx = param_names.index(param)
    cgc_mean, cgc_std = np.mean(chains[:, idx]), np.std(chains[:, idx])
    
    ax.hist(chains[:, idx], bins=50, density=True, alpha=0.7, color='steelblue', 
            label=f'CGC: {cgc_mean:.4f}±{cgc_std:.4f}')
    ax.axvline(planck_val[0], color='orange', lw=2, label=f'Planck: {planck_val[0]:.4f}±{planck_val[1]:.4f}')
    ax.axvspan(planck_val[0]-planck_val[1], planck_val[0]+planck_val[1], alpha=0.3, color='orange')
    ax.set_xlabel(param_labels[idx], fontsize=14)
    ax.legend(fontsize=9)

plt.suptitle('CGC vs Planck ΛCDM Cosmological Parameters', fontsize=14)
plt.tight_layout()
plt.savefig('plots/cosmology_comparison_10k.png', dpi=150, bbox_inches='tight')
plt.close()
print('       Saved: plots/cosmology_comparison_10k.png')

# =============================================================================
# 5. FINAL SUMMARY
# =============================================================================

print('\n' + '='*70)
print('ANALYSIS COMPLETE')
print('='*70)

mu_mean = results['mu']['mean']
mu_std = results['mu']['std']
mu_sigma = abs(mu_mean) / mu_std

print('\n' + '-'*70)
print('CGC DETECTION SUMMARY')
print('-'*70)
print(f'\n  mu (CGC coupling)     = {mu_mean:.4f} +/- {mu_std:.4f}')
print(f'  n_g (spectral index)  = {results["n_g"]["mean"]:.4f} +/- {results["n_g"]["std"]:.4f}')
print(f'  z_trans               = {results["z_trans"]["mean"]:.3f} +/- {results["z_trans"]["std"]:.3f}')
print(f'\n  DETECTION SIGNIFICANCE: {mu_sigma:.1f} sigma')

if mu_sigma > 5:
    print('\n  ★★★ DISCOVERY-LEVEL EVIDENCE FOR CGC (>5 sigma) ★★★')
elif mu_sigma > 3:
    print('\n  ★★ STRONG EVIDENCE FOR CGC (>3 sigma) ★★')
elif mu_sigma > 2:
    print('\n  ★ MODERATE EVIDENCE FOR CGC (2-3 sigma) ★')
else:
    print('\n  No significant evidence for CGC (<2 sigma)')

# EFT validation
print('\n' + '-'*70)
print('EFT PHYSICS VALIDATION')
print('-'*70)

ng_mean = results['n_g']['mean']
ng_std = results['n_g']['std']
zt_mean = results['z_trans']['mean']
zt_std = results['z_trans']['std']

ng_eft = 0.014  # beta_0^2 / 4pi^2
zt_eft = 1.67   # z_acc + Delta_z

ng_tension = abs(ng_mean - ng_eft) / ng_std if ng_std > 0 else 0
zt_tension = abs(zt_mean - zt_eft) / zt_std if zt_std > 0 else 0

print(f'\n  n_g:     Fitted={ng_mean:.4f}, EFT={ng_eft:.4f}, Tension={ng_tension:.1f} sigma')
print(f'  z_trans: Fitted={zt_mean:.3f}, EFT={zt_eft:.3f}, Tension={zt_tension:.1f} sigma')

if ng_tension < 2 and zt_tension < 2:
    print('\n  --> Both parameters CONSISTENT with EFT predictions (<2 sigma)')
else:
    print('\n  --> Some tension with EFT predictions')

print('\n' + '-'*70)
print('OUTPUT FILES')
print('-'*70)
print('\n  Chains: results/chains/mcmc_chains.npz')
print('\n  Plots:')
for f in ['corner_full_10k.png', 'corner_cgc_10k.png', 'trace_plots_10k.png',
          'posteriors_1d_10k.png', 'mu_vs_ng_10k.png', 'cgc_significance_10k.png',
          'cosmology_comparison_10k.png']:
    print(f'    - plots/{f}')

# Save results
np.savez('results/cgc_analysis_10k.npz', chains=chains, param_names=param_names, results=results)
print('\n  Results: results/cgc_analysis_10k.npz')

print('\n' + '='*70)
print('ALL DONE!')
print('='*70)
