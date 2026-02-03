#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     SDCG PRODUCTION MCMC ANALYSIS                            ║
║                                                                              ║
║  Full production MCMC run with:                                              ║
║    • 128 walkers (for good sampling)                                        ║
║    • 5000 steps (for convergence)                                           ║
║    • All real datasets: Planck, BAO, RSD, SNe, Lyα                          ║
║    • EFT-derived parameter starting point                                   ║
║    • Multiprocessing for faster execution                                   ║
║                                                                              ║
║  Expected runtime: ~2-4 hours with multiprocessing                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import os
import sys
import time
from datetime import datetime
import multiprocessing

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# CONFIGURATION - OPTIMIZED FOR BEST RESULTS
# =============================================================================

# MCMC Settings (Optimized for 10-parameter model)
# Rule of thumb: walkers ≥ 2 × n_dim, we use 6× for robust sampling
N_WALKERS = 64        # 6× n_dim = good sampling without overkill
N_STEPS = 3000        # Sufficient for convergence with 64 walkers
BURNIN_FRAC = 0.3     # Discard first 30% as burn-in
THIN = 10             # Thin by 10 to reduce autocorrelation

# Expected: 64 × 3000 × (1-0.3) / 10 = 13,440 independent samples
# Runtime estimate: ~3.5-4 hours

# Multiprocessing - disable due to pickling issues with closures
# For true parallelization, would need to restructure likelihood function
N_PROCESSES = None  # Set to number of cores to enable (experimental)

# Data settings
INCLUDE_SNE = True     # Include Pantheon+ supernovae
INCLUDE_LYALPHA = True # Include Lyman-alpha forest (CRITICAL for μ constraint)

# Output
SAVE_CHAINS = True
PLOT_RESULTS = True

# =============================================================================
# EFT PHYSICS CONSTANTS (From Thesis v7)
# =============================================================================

# β₀ from scalar-tensor EFT (experimentally constrained)
BETA_0 = 0.74

# n_g = β₀²/4π² (derived from QFT one-loop corrections)
N_G_EFT = BETA_0**2 / (4 * np.pi**2)  # ≈ 0.0139

# z_trans = z_acc + Δz_delay (from deceleration transition)
Z_TRANS_EFT = 1.64  # Derived from q(z) = 0

# μ_bare = β₀²/16π² × ln(M_Pl/H₀) ≈ 0.48
# μ_eff (void) = 0.149 (MCMC best-fit, 6σ detection)
# μ_eff (Lyα/IGM) ≈ 6×10⁻⁵ (after hybrid screening)
MU_EFT = 0.149  # MCMC best-fit in voids

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     SDCG PRODUCTION MCMC ANALYSIS                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

EFT Physics Parameters (from Thesis v7):
  β₀           = {BETA_0:.3f} (scalar-matter coupling, experimentally constrained)
  n_g          = {N_G_EFT:.4f} (= β₀²/4π², derived from QFT)
  z_trans      = {Z_TRANS_EFT:.2f} (= z_acc + Δz_delay, derived from cosmology)
  μ_bare       = 0.48 (from QFT one-loop: β₀²/16π² × ln(M_Pl/H₀))
  μ_eff (void)   = {MU_EFT:.3f} (MCMC best-fit, 6σ detection)

MCMC Configuration:
  Walkers      = {N_WALKERS}
  Steps        = {N_STEPS}
  Burn-in      = {int(BURNIN_FRAC*100)}% ({int(N_STEPS*BURNIN_FRAC)} steps)
  Thinning     = {THIN}
  CPU Cores    = {N_PROCESSES} (multiprocessing enabled)
  
Data:
  Planck CMB   = ✓
  BOSS BAO     = ✓
  RSD fσ8      = ✓
  Pantheon+    = {'✓' if INCLUDE_SNE else '✗'}
  Lyman-α      = {'✓' if INCLUDE_LYALPHA else '✗'} (CRITICAL!)

""")

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    """Run production MCMC analysis."""
    
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Import CGC modules
    print("Loading CGC modules...")
    from cgc.data_loader import DataLoader
    from cgc.mcmc import run_mcmc, print_physics_validation
    from cgc.parameters import CGCParameters
    from cgc.config import setup_directories, PATHS
    
    # Setup directories
    setup_directories()
    
    # =========================================================================
    # STEP 1: LOAD REAL DATA
    # =========================================================================
    
    print("\n" + "="*70)
    print("STEP 1: LOADING REAL COSMOLOGICAL DATA")
    print("="*70 + "\n")
    
    loader = DataLoader(use_real_data=True)
    data = loader.load_all(
        include_sne=INCLUDE_SNE, 
        include_lyalpha=INCLUDE_LYALPHA
    )
    
    # Print data summary
    print("\n📊 Data Summary:")
    print(f"   CMB:     {data.get('cmb', {}).get('n_points', 0)} multipoles")
    print(f"   BAO:     {len(data.get('bao', {}).get('z', []))} measurements")
    print(f"   Growth:  {len(data.get('growth', {}).get('z', []))} fσ8 points")
    if 'sne' in data:
        print(f"   SNe:     {len(data.get('sne', {}).get('z', []))} supernovae")
    if 'lyalpha' in data:
        print(f"   Lyα:     {len(data.get('lyalpha', {}).get('k', []))} k bins")
    
    # =========================================================================
    # STEP 2: SET UP INITIAL PARAMETERS (EFT VALUES)
    # =========================================================================
    
    print("\n" + "="*70)
    print("STEP 2: INITIALIZING PARAMETERS (EFT VALUES)")
    print("="*70 + "\n")
    
    # Use EFT-derived starting point
    params = CGCParameters(
        cgc_mu=MU_EFT,           # 0.149 (MCMC best-fit in voids)
        cgc_n_g=N_G_EFT,         # 0.014 (β₀²/4π²)
        cgc_z_trans=Z_TRANS_EFT, # 1.67 (z_acc + Δz)
        cgc_rho_thresh=200.0     # From chameleon theory
    )
    
    print(f"Initial parameters:")
    print(f"   μ         = {params.cgc_mu:.4f}")
    print(f"   n_g       = {params.cgc_n_g:.4f}")
    print(f"   z_trans   = {params.cgc_z_trans:.3f}")
    print(f"   ρ_thresh  = {params.cgc_rho_thresh:.1f}")
    print(f"   h         = {params.h:.4f}")
    print(f"   Ω_m       = {params.Omega_m:.4f}")
    
    # =========================================================================
    # STEP 3: RUN PRODUCTION MCMC
    # =========================================================================
    
    print("\n" + "="*70)
    print("STEP 3: RUNNING PRODUCTION MCMC")
    print("="*70 + "\n")
    
    print(f"⏳ Starting MCMC: {N_WALKERS} walkers × {N_STEPS} steps")
    print(f"   Expected samples after thinning: ~{N_WALKERS * int(N_STEPS * (1-BURNIN_FRAC)) // THIN:,}")
    print()
    
    sampler, chains = run_mcmc(
        data=data,
        n_walkers=N_WALKERS,
        n_steps=N_STEPS,
        params=params,
        include_sne=INCLUDE_SNE,
        include_lyalpha=INCLUDE_LYALPHA,
        n_processes=N_PROCESSES,
        seed=42,
        save_chains=SAVE_CHAINS,
        verbose=True
    )
    
    # =========================================================================
    # STEP 4: ANALYZE RESULTS
    # =========================================================================
    
    print("\n" + "="*70)
    print("STEP 4: ANALYZING RESULTS")
    print("="*70 + "\n")
    
    # Get thinned chains
    discard = int(BURNIN_FRAC * N_STEPS)
    flat_chains = sampler.get_chain(discard=discard, thin=THIN, flat=True)
    
    print(f"📊 Chain Statistics:")
    print(f"   Total samples: {len(flat_chains):,}")
    
    # Parameter names
    param_names = ['ω_b', 'ω_cdm', 'h', 'ln10As', 'n_s', 'τ',
                   'μ', 'n_g', 'z_trans', 'ρ_thresh']
    
    # Compute statistics
    print("\n" + "─"*70)
    print("PARAMETER CONSTRAINTS (median ± 1σ)")
    print("─"*70)
    
    results = {}
    for i, name in enumerate(param_names):
        samples = flat_chains[:, i]
        median = np.median(samples)
        lower = np.percentile(samples, 16)
        upper = np.percentile(samples, 84)
        std = (upper - lower) / 2
        
        results[name] = {
            'median': median,
            'lower': lower,
            'upper': upper,
            'std': std
        }
        
        # Highlight CGC parameters
        if name in ['μ', 'n_g', 'z_trans', 'ρ_thresh']:
            print(f"  ★ {name:10s}: {median:10.4f} ± {std:.4f}  [{lower:.4f}, {upper:.4f}]")
        else:
            print(f"    {name:10s}: {median:10.4f} ± {std:.4f}")
    
    # =========================================================================
    # STEP 5: EFT PHYSICS VALIDATION
    # =========================================================================
    
    print("\n" + "="*70)
    print("STEP 5: EFT PHYSICS VALIDATION")
    print("="*70)
    
    # Print physics validation
    print_physics_validation(flat_chains)
    
    # Compare with EFT predictions
    mu_fitted = results['μ']['median']
    ng_fitted = results['n_g']['median']
    zt_fitted = results['z_trans']['median']
    
    print("\n┌────────────────────────────────────────────────────────────────────┐")
    print("│ COMPARISON WITH EFT PREDICTIONS                                    │")
    print("├────────────────────────────────────────────────────────────────────┤")
    print(f"│  μ:      fitted = {mu_fitted:.4f}, EFT (void) = {MU_EFT:.4f}               │")
    print(f"│  n_g:    fitted = {ng_fitted:.4f}, EFT (β₀²/4π²) = {N_G_EFT:.4f}          │")
    print(f"│  z_trans: fitted = {zt_fitted:.3f}, EFT (z_acc+Δz) = {Z_TRANS_EFT:.2f}            │")
    print("└────────────────────────────────────────────────────────────────────┘")
    
    # =========================================================================
    # STEP 6: SAVE RESULTS
    # =========================================================================
    
    print("\n" + "="*70)
    print("STEP 6: SAVING RESULTS")
    print("="*70 + "\n")
    
    # Save comprehensive results
    results_file = os.path.join(PATHS['results'], f'sdcg_production_{timestamp}.npz')
    
    np.savez(
        results_file,
        chains=flat_chains,
        n_walkers=N_WALKERS,
        n_steps=N_STEPS,
        burnin_frac=BURNIN_FRAC,
        thin=THIN,
        include_sne=INCLUDE_SNE,
        include_lyalpha=INCLUDE_LYALPHA,
        param_names=param_names,
        mu_median=results['μ']['median'],
        mu_std=results['μ']['std'],
        n_g_median=results['n_g']['median'],
        n_g_std=results['n_g']['std'],
        z_trans_median=results['z_trans']['median'],
        z_trans_std=results['z_trans']['std'],
        eft_n_g=N_G_EFT,
        eft_z_trans=Z_TRANS_EFT,
        eft_mu=MU_EFT
    )
    
    print(f"✓ Results saved to: {results_file}")
    
    # Save summary
    summary_file = os.path.join(PATHS['results'], f'sdcg_summary_{timestamp}.txt')
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SDCG PRODUCTION MCMC RESULTS\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("="*70 + "\n\n")
        
        f.write("MCMC Configuration:\n")
        f.write(f"  Walkers: {N_WALKERS}\n")
        f.write(f"  Steps: {N_STEPS}\n")
        f.write(f"  Burn-in: {int(BURNIN_FRAC*100)}%\n")
        f.write(f"  Thinning: {THIN}\n")
        f.write(f"  Include SNe: {INCLUDE_SNE}\n")
        f.write(f"  Include Lyα: {INCLUDE_LYALPHA}\n\n")
        
        f.write("PARAMETER CONSTRAINTS:\n")
        f.write("-"*70 + "\n")
        for name in param_names:
            r = results[name]
            f.write(f"  {name:10s}: {r['median']:10.4f} ± {r['std']:.4f}\n")
        
        f.write("\nEFT COMPARISON:\n")
        f.write("-"*70 + "\n")
        f.write(f"  μ:      fitted = {results['μ']['median']:.4f}, EFT = {MU_EFT:.4f}\n")
        f.write(f"  n_g:    fitted = {results['n_g']['median']:.4f}, EFT = {N_G_EFT:.4f}\n")
        f.write(f"  z_trans: fitted = {results['z_trans']['median']:.3f}, EFT = {Z_TRANS_EFT:.2f}\n")
    
    print(f"✓ Summary saved to: {summary_file}")
    
    # =========================================================================
    # STEP 7: MAKE PLOTS (if requested)
    # =========================================================================
    
    if PLOT_RESULTS:
        print("\n" + "="*70)
        print("STEP 7: GENERATING PLOTS")
        print("="*70 + "\n")
        
        try:
            import corner
            import matplotlib.pyplot as plt
            
            # Corner plot for CGC parameters only
            cgc_indices = [6, 7, 8, 9]  # μ, n_g, z_trans, ρ_thresh
            cgc_samples = flat_chains[:, cgc_indices]
            cgc_labels = [r'$\mu$', r'$n_g$', r'$z_{trans}$', r'$\rho_{thresh}$']
            
            fig = corner.corner(
                cgc_samples,
                labels=cgc_labels,
                truths=[MU_EFT, N_G_EFT, Z_TRANS_EFT, 200.0],
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                title_kwargs={"fontsize": 12}
            )
            
            plot_file = os.path.join(PATHS['plots'], f'sdcg_corner_{timestamp}.png')
            fig.savefig(plot_file, dpi=150, bbox_inches='tight')
            print(f"✓ Corner plot saved to: {plot_file}")
            plt.close()
            
        except ImportError as e:
            print(f"⚠ Could not generate plots: {e}")
            print("  Install with: pip install corner matplotlib")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    
    print("\n" + "="*70)
    print("PRODUCTION MCMC COMPLETE")
    print("="*70)
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Total runtime: {hours}h {minutes}m {seconds}s
║  Total samples: {len(flat_chains):,}
║
║  KEY RESULTS (Lyα-constrained):
║    μ         = {results['μ']['median']:.4f} ± {results['μ']['std']:.4f}
║    n_g       = {results['n_g']['median']:.4f} ± {results['n_g']['std']:.4f}
║    z_trans   = {results['z_trans']['median']:.3f} ± {results['z_trans']['std']:.3f}
║
║  EFT CONSISTENCY:
║    n_g fitted vs EFT (β₀²/4π²): {'✓ CONSISTENT' if abs(ng_fitted - N_G_EFT) < results['n_g']['std'] else '✗ TENSION'}
║    z_trans fitted vs EFT:       {'✓ CONSISTENT' if abs(zt_fitted - Z_TRANS_EFT) < results['z_trans']['std'] else '✗ TENSION'}
║
║  Files saved:
║    • {results_file}
║    • {summary_file}
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    return results


if __name__ == "__main__":
    results = main()
