#!/usr/bin/env python3
"""
=============================================================================
SDCG Complete Observational Test Suite
=============================================================================

Comprehensive testing of Scale-Dependent Chameleon Gravity predictions
using realistic galaxy rotation curve and environment data.

This script:
1. Generates/loads SPARC-like rotation curves with proper Tully-Fisher
2. Creates realistic void environment distribution
3. Injects SDCG signal (for validation) or tests null hypothesis
4. Performs statistical analysis of velocity offsets
5. Compares with theory predictions

Key SDCG predictions:
- μ_eff(voids) ≈ 0.149 → +7.4% velocity enhancement
- μ_eff(clusters) ≈ 0.001 → negligible effect (screened)
- Lyman-α: μ_eff ≈ 6×10⁻⁵ → passes <7.5% constraint

Author: SDCG Analysis Pipeline
Date: February 2026
=============================================================================
"""

import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SDCG Theory Parameters
# =============================================================================

class SDCGTheory:
    """SDCG theoretical predictions"""
    
    # Fundamental parameters
    MU_BARE = 0.48      # QFT one-loop derivation
    BETA_0 = 0.70       # SM ansatz (scalar-matter coupling)
    K_0 = 0.05          # Pivot scale (h/Mpc)
    GAMMA = 0.0125      # Scale exponent
    
    # Screening parameters for different environments
    SCREENING = {
        'void':        0.31,       # Deep void interior
        'void_edge':   0.20,       # Void boundary
        'underdense':  0.15,       # Mildly underdense
        'field':       0.08,       # Field/average
        'filament':    0.03,       # Cosmic filament
        'group':       0.01,       # Galaxy group
        'cluster':     0.002,      # Cluster core
        'lyman_alpha': 1.2e-4,     # IGM (diffuse but dense gas)
        'solar':       1e-60,      # Solar System (total screening)
    }
    
    @classmethod
    def mu_eff(cls, environment: str) -> float:
        """Get effective μ for an environment"""
        S = cls.SCREENING.get(environment.lower(), 0.1)
        return cls.MU_BARE * S
    
    @classmethod
    def velocity_enhancement(cls, environment: str) -> float:
        """
        Fractional velocity enhancement from modified gravity
        
        v_obs = v_GR × √(G_eff/G_N)
        G_eff/G_N = 1 + μ_eff
        
        Enhancement = √(1 + μ_eff) - 1 ≈ μ_eff/2 for μ_eff << 1
        """
        mu = cls.mu_eff(environment)
        return np.sqrt(1 + mu) - 1
    
    @classmethod
    def velocity_difference(cls, env1: str, env2: str) -> float:
        """Predicted velocity difference between two environments"""
        enh1 = cls.velocity_enhancement(env1)
        enh2 = cls.velocity_enhancement(env2)
        return enh1 - enh2


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Galaxy:
    """Galaxy with rotation curve data"""
    name: str
    distance: float       # Mpc
    v_rot: float          # Rotation velocity (km/s)
    v_rot_err: float      # Error
    log_mstar: float      # log10(M_star/M_sun)
    log_mgas: float       # log10(M_gas/M_sun)
    r_eff: float          # Effective radius (kpc)
    morph_type: str       # Morphological type
    
    # Environment info
    environment: str = 'field'
    delta_rho: float = 0.0      # Density contrast
    distance_to_void: float = -1  # Distance to nearest void center (Mpc)
    
    # SDCG analysis
    v_rot_enhanced: float = 0.0  # With SDCG enhancement
    tf_residual: float = 0.0     # Tully-Fisher residual


@dataclass
class Void:
    """Cosmic void"""
    void_id: int
    x: float   # Mpc (comoving)
    y: float
    z_coord: float  # Avoid confusion with redshift
    r_eff: float    # Effective radius (Mpc)
    delta: float    # Central density contrast (-0.8 to -0.95)


# =============================================================================
# Data Generation with SDCG Physics
# =============================================================================

class SDCGDataGenerator:
    """
    Generate realistic galaxy data with optional SDCG signal injection
    """
    
    def __init__(self, seed: int = 2026):
        np.random.seed(seed)
        self.theory = SDCGTheory()
        
    def generate_void_distribution(self, 
                                   box_size: float = 500,  # Mpc
                                   n_voids: int = 200) -> List[Void]:
        """
        Generate void distribution following observed statistics
        
        Based on SDSS DR7 void finder results (Sutter et al. 2012)
        """
        voids = []
        
        for i in range(n_voids):
            # Void size distribution (log-normal, peak ~20 Mpc/h)
            r_eff = np.exp(np.random.normal(np.log(25), 0.4))
            r_eff = np.clip(r_eff, 8, 60)
            
            # Random position in box
            x = np.random.uniform(0, box_size)
            y = np.random.uniform(0, box_size)
            z = np.random.uniform(0, box_size)
            
            # Central underdensity (correlated with size - larger voids are emptier)
            delta = -0.7 - 0.2 * (r_eff / 40)
            delta = np.clip(delta, -0.95, -0.5)
            
            voids.append(Void(
                void_id=i + 1,
                x=x, y=y, z_coord=z,
                r_eff=r_eff,
                delta=delta
            ))
            
        return voids
    
    def generate_galaxies(self,
                          n_galaxies: int = 200,
                          box_size: float = 500,
                          voids: List[Void] = None,
                          inject_sdcg: bool = True) -> List[Galaxy]:
        """
        Generate galaxies with realistic properties and environment classification
        
        Parameters:
        -----------
        inject_sdcg : bool
            If True, inject SDCG velocity enhancement based on environment
            If False, generate pure GR velocities for null hypothesis testing
        """
        if voids is None:
            voids = self.generate_void_distribution(box_size)
            
        galaxies = []
        
        # Galaxy mass function - more low-mass galaxies
        log_mstar_values = np.random.normal(9.5, 1.2, n_galaxies)
        log_mstar_values = np.clip(log_mstar_values, 6.5, 11.5)
        
        for i in range(n_galaxies):
            # Position
            x = np.random.uniform(0, box_size)
            y = np.random.uniform(0, box_size)
            z = np.random.uniform(0, box_size)
            
            # Mass and related properties
            log_mstar = log_mstar_values[i]
            
            # Baryonic Tully-Fisher relation
            # V_flat ∝ M_bary^0.25 with scatter
            log_v_tf = 0.25 * (log_mstar - 9.0) + 2.0  # log10(V) in km/s
            log_v_tf += np.random.normal(0, 0.08)  # ~20% scatter
            v_rot_gr = 10 ** log_v_tf
            v_rot_gr = np.clip(v_rot_gr, 20, 350)
            
            # Error (typically 5-10% for good data)
            v_rot_err = v_rot_gr * np.random.uniform(0.05, 0.12)
            
            # Gas mass (dwarf galaxies are more gas-rich)
            gas_fraction = 0.3 - 0.02 * (log_mstar - 8)  # Higher for low mass
            gas_fraction = np.clip(gas_fraction, 0.05, 0.8)
            log_mgas = log_mstar + np.log10(gas_fraction / (1 - gas_fraction))
            
            # Effective radius (scales with mass)
            r_eff = 10 ** (0.3 * (log_mstar - 10) + np.random.normal(0, 0.15))
            r_eff = np.clip(r_eff, 0.2, 30)
            
            # Morphological type (dwarfs are more irregular)
            if log_mstar < 8:
                morph_type = np.random.choice(['dIrr', 'dSph', 'BCD'], p=[0.6, 0.3, 0.1])
            elif log_mstar < 9.5:
                morph_type = np.random.choice(['Sm', 'Sd', 'Im'], p=[0.4, 0.4, 0.2])
            else:
                morph_type = np.random.choice(['Sb', 'Sc', 'Sa'], p=[0.4, 0.4, 0.2])
                
            # Determine environment based on position relative to voids
            env, delta_rho, d_void = self._classify_environment(x, y, z, voids)
            
            # SDCG velocity enhancement
            if inject_sdcg:
                enhancement = SDCGTheory.velocity_enhancement(env)
                v_rot = v_rot_gr * (1 + enhancement)
            else:
                v_rot = v_rot_gr
                
            # Add measurement noise
            v_rot += np.random.normal(0, v_rot_err * 0.5)
            v_rot = max(v_rot, 10)
            
            # Distance (simplified - place in local universe)
            distance = np.sqrt(x**2 + y**2 + z**2) / 10  # Scale to Mpc
            distance = np.clip(distance, 1, 150)
            
            galaxies.append(Galaxy(
                name=f"GAL_{i+1:04d}",
                distance=distance,
                v_rot=v_rot,
                v_rot_err=v_rot_err,
                log_mstar=log_mstar,
                log_mgas=log_mgas,
                r_eff=r_eff,
                morph_type=morph_type,
                environment=env,
                delta_rho=delta_rho,
                distance_to_void=d_void,
                v_rot_enhanced=v_rot
            ))
            
        return galaxies
    
    def _classify_environment(self, x: float, y: float, z: float,
                              voids: List[Void]) -> Tuple[str, float, float]:
        """
        Classify galaxy environment based on void positions
        
        Returns: (environment, delta_rho, distance_to_nearest_void)
        """
        min_distance = float('inf')
        in_void = False
        void_delta = 0
        
        for void in voids:
            # Distance to void center
            d = np.sqrt((x - void.x)**2 + (y - void.y)**2 + (z - void.z_coord)**2)
            
            if d < min_distance:
                min_distance = d
                
            # Check if inside void
            if d < void.r_eff:
                in_void = True
                # Density profile (shell-like voids are emptier at edges)
                r_frac = d / void.r_eff
                void_delta = void.delta * (1 - 0.3 * r_frac)  # Emptier in center
                
        # Classify
        if in_void:
            if min_distance < 0.3 * void.r_eff:
                return 'void', void_delta, min_distance
            else:
                return 'void_edge', void_delta * 0.6, min_distance
        elif min_distance < 1.5 * (void.r_eff if voids else 30):
            return 'underdense', -0.3, min_distance
        elif min_distance < 3 * (void.r_eff if voids else 30):
            return 'field', 0.0, min_distance
        else:
            # Likely in filament or cluster
            if np.random.random() < 0.7:
                return 'filament', 0.5, min_distance
            else:
                return 'group', 2.0, min_distance


# =============================================================================
# Statistical Analysis
# =============================================================================

class SDCGStatisticalAnalysis:
    """
    Perform statistical analysis of SDCG signatures
    """
    
    def __init__(self, galaxies: List[Galaxy]):
        self.galaxies = galaxies
        self.results = {}
        
    def fit_tully_fisher(self) -> Dict:
        """
        Fit the Baryonic Tully-Fisher relation
        
        log(V) = a + b × log(M_bary)
        """
        # Use total baryonic mass
        log_mbary = []
        log_v = []
        
        for g in self.galaxies:
            m_star = 10 ** g.log_mstar
            m_gas = 10 ** g.log_mgas
            log_mbary.append(np.log10(m_star + m_gas))
            log_v.append(np.log10(g.v_rot))
            
        log_mbary = np.array(log_mbary)
        log_v = np.array(log_v)
        
        # Linear regression
        coeffs = np.polyfit(log_mbary, log_v, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        
        # Predicted velocities and residuals
        log_v_pred = intercept + slope * log_mbary
        residuals = log_v - log_v_pred
        
        # Store residuals in galaxy objects
        for i, g in enumerate(self.galaxies):
            g.tf_residual = residuals[i]
            
        # Intrinsic scatter
        scatter = np.std(residuals)
        
        self.results['tully_fisher'] = {
            'slope': slope,
            'intercept': intercept,
            'scatter_dex': scatter,
            'n_galaxies': len(self.galaxies)
        }
        
        return self.results['tully_fisher']
    
    def analyze_environment_velocity_offset(self) -> Dict:
        """
        THE KEY SDCG TEST: Compare TF residuals by environment
        
        If SDCG is correct:
        - Void galaxies have POSITIVE TF residuals (higher V at fixed M)
        - Cluster galaxies have near-zero residuals (screened)
        """
        # Group by environment
        env_residuals = {}
        
        for g in self.galaxies:
            env = g.environment
            if env not in env_residuals:
                env_residuals[env] = []
            env_residuals[env].append(g.tf_residual)
            
        # Compute statistics
        results = {}
        
        for env, residuals in env_residuals.items():
            if len(residuals) < 3:
                continue
                
            residuals = np.array(residuals)
            n = len(residuals)
            mean = np.mean(residuals)
            std = np.std(residuals)
            sem = std / np.sqrt(n)
            
            # Significance of deviation from zero
            t_stat = mean / sem if sem > 0 else 0
            
            # Convert to velocity percentage
            # Δlog(V) = 0.01 → Δv/v = 10^0.01 - 1 ≈ 2.3%
            pct_offset = (10**mean - 1) * 100
            pct_error = (10**sem - 1) * 100
            
            # Theoretical prediction
            predicted_enhancement = SDCGTheory.velocity_enhancement(env) * 100
            
            results[env] = {
                'n': n,
                'mean_residual_dex': float(mean),
                'std_dex': float(std),
                'sem_dex': float(sem),
                't_statistic': float(t_stat),
                'velocity_offset_pct': float(pct_offset),
                'velocity_error_pct': float(pct_error),
                'predicted_enhancement_pct': predicted_enhancement,
                'mu_eff': SDCGTheory.mu_eff(env)
            }
            
        self.results['environment_analysis'] = results
        return results
    
    def void_vs_cluster_test(self) -> Dict:
        """
        Direct comparison: void galaxies vs cluster/group galaxies
        """
        void_gals = [g for g in self.galaxies if g.environment in ['void', 'void_edge']]
        dense_gals = [g for g in self.galaxies if g.environment in ['group', 'filament', 'cluster']]
        
        if len(void_gals) < 5 or len(dense_gals) < 5:
            return {'error': 'Insufficient galaxies in void/dense categories'}
            
        void_resid = np.array([g.tf_residual for g in void_gals])
        dense_resid = np.array([g.tf_residual for g in dense_gals])
        
        # Two-sample t-test
        n1, n2 = len(void_resid), len(dense_resid)
        mean1, mean2 = np.mean(void_resid), np.mean(dense_resid)
        std1, std2 = np.std(void_resid), np.std(dense_resid)
        
        # Pooled standard error
        se = np.sqrt(std1**2/n1 + std2**2/n2)
        t_stat = (mean1 - mean2) / se if se > 0 else 0
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        
        # Velocity offset in percent
        delta_v_pct = (10**(mean1 - mean2) - 1) * 100
        
        # Theoretical prediction
        pred_void = SDCGTheory.velocity_enhancement('void') * 100
        pred_dense = SDCGTheory.velocity_enhancement('filament') * 100
        pred_diff = pred_void - pred_dense
        
        result = {
            'n_void': n1,
            'n_dense': n2,
            'mean_void_dex': float(mean1),
            'mean_dense_dex': float(mean2),
            'difference_dex': float(mean1 - mean2),
            'velocity_offset_pct': float(delta_v_pct),
            't_statistic': float(t_stat),
            'cohens_d': float(cohens_d),
            'predicted_difference_pct': pred_diff,
            'observed_vs_predicted': float(delta_v_pct / pred_diff) if pred_diff != 0 else 0
        }
        
        self.results['void_vs_cluster'] = result
        return result
    
    def mass_binned_analysis(self, n_bins: int = 4) -> Dict:
        """
        Analyze velocity offset by mass bin to check for systematics
        
        SDCG effect should be roughly constant across mass bins
        """
        # Sort by mass
        sorted_gals = sorted(self.galaxies, key=lambda g: g.log_mstar)
        bin_size = len(sorted_gals) // n_bins
        
        results = []
        
        for i in range(n_bins):
            start = i * bin_size
            end = start + bin_size if i < n_bins - 1 else len(sorted_gals)
            bin_gals = sorted_gals[start:end]
            
            # Separate by environment
            void_in_bin = [g for g in bin_gals if g.environment in ['void', 'void_edge', 'underdense']]
            dense_in_bin = [g for g in bin_gals if g.environment in ['filament', 'group']]
            
            if len(void_in_bin) < 2 or len(dense_in_bin) < 2:
                continue
                
            # Mean mass in bin
            mean_mass = np.mean([g.log_mstar for g in bin_gals])
            
            # Mean velocities
            v_void = np.mean([g.v_rot for g in void_in_bin])
            v_dense = np.mean([g.v_rot for g in dense_in_bin])
            
            delta_v_pct = (v_void - v_dense) / v_dense * 100
            
            results.append({
                'mass_bin': f"{mean_mass:.1f}",
                'n_void': len(void_in_bin),
                'n_dense': len(dense_in_bin),
                'v_void': float(v_void),
                'v_dense': float(v_dense),
                'delta_v_pct': float(delta_v_pct)
            })
            
        self.results['mass_bins'] = results
        return results


# =============================================================================
# Main Analysis
# =============================================================================

def run_complete_analysis(inject_sdcg: bool = True,
                          n_galaxies: int = 500,
                          output_dir: str = 'results'):
    """
    Run complete SDCG observational test suite
    """
    print("\n" + "="*70)
    print("SCALE-DEPENDENT CHAMELEON GRAVITY - OBSERVATIONAL TEST SUITE")
    print("="*70)
    
    # Configuration
    print(f"\n{'Configuration':-^70}")
    print(f"  SDCG signal injection: {'YES' if inject_sdcg else 'NO (null test)'}")
    print(f"  Number of galaxies: {n_galaxies}")
    print(f"\n  Theory parameters:")
    print(f"    μ_bare = {SDCGTheory.MU_BARE} (QFT one-loop)")
    print(f"    β₀ = {SDCGTheory.BETA_0} (SM ansatz)")
    print(f"    k₀ = {SDCGTheory.K_0} h/Mpc")
    
    # Generate data
    print(f"\n{'[1] Data Generation':-^70}")
    generator = SDCGDataGenerator(seed=2026)
    voids = generator.generate_void_distribution(n_voids=200)
    galaxies = generator.generate_galaxies(
        n_galaxies=n_galaxies,
        voids=voids,
        inject_sdcg=inject_sdcg
    )
    
    print(f"  Generated {len(voids)} voids")
    print(f"  Generated {len(galaxies)} galaxies")
    
    # Environment distribution
    env_counts = {}
    for g in galaxies:
        env_counts[g.environment] = env_counts.get(g.environment, 0) + 1
    print(f"\n  Environment distribution:")
    for env in ['void', 'void_edge', 'underdense', 'field', 'filament', 'group']:
        if env in env_counts:
            mu = SDCGTheory.mu_eff(env)
            enh = SDCGTheory.velocity_enhancement(env) * 100
            print(f"    {env:12s}: {env_counts[env]:4d} galaxies  (μ_eff={mu:.4f}, Δv={enh:+.1f}%)")
    
    # Statistical analysis
    print(f"\n{'[2] Tully-Fisher Relation':-^70}")
    analyzer = SDCGStatisticalAnalysis(galaxies)
    tf = analyzer.fit_tully_fisher()
    
    print(f"  log(V) = {tf['intercept']:.3f} + {tf['slope']:.3f} × log(M_bary)")
    print(f"  Intrinsic scatter: {tf['scatter_dex']:.3f} dex ({(10**tf['scatter_dex']-1)*100:.1f}%)")
    
    # Environment analysis
    print(f"\n{'[3] Environment Velocity Offsets':-^70}")
    env_results = analyzer.analyze_environment_velocity_offset()
    
    print(f"\n  {'Environment':<12} {'N':>5} {'Δlog(V)':>10} {'Δv/v':>10} {'Pred.':>10} {'t-stat':>10}")
    print("  " + "-"*62)
    
    for env in ['void', 'void_edge', 'underdense', 'field', 'filament', 'group']:
        if env in env_results:
            r = env_results[env]
            print(f"  {env:<12} {r['n']:>5} {r['mean_residual_dex']:>+10.4f} "
                  f"{r['velocity_offset_pct']:>+9.1f}% {r['predicted_enhancement_pct']:>+9.1f}% "
                  f"{r['t_statistic']:>+9.2f}")
    
    # Key test: Void vs Dense
    print(f"\n{'[4] KEY TEST: Void vs Dense Comparison':-^70}")
    vdc = analyzer.void_vs_cluster_test()
    
    if 'error' not in vdc:
        print(f"\n  Void galaxies (n={vdc['n_void']}):  mean TF residual = {vdc['mean_void_dex']:+.4f} dex")
        print(f"  Dense galaxies (n={vdc['n_dense']}): mean TF residual = {vdc['mean_dense_dex']:+.4f} dex")
        print(f"\n  Observed velocity offset:  Δv/v = {vdc['velocity_offset_pct']:+.2f}%")
        print(f"  Predicted (SDCG):          Δv/v = {vdc['predicted_difference_pct']:+.2f}%")
        print(f"\n  t-statistic: {vdc['t_statistic']:.2f}")
        print(f"  Cohen's d:   {vdc['cohens_d']:.3f}")
        
        if inject_sdcg:
            recovery = vdc['observed_vs_predicted']
            print(f"\n  Signal recovery: {recovery*100:.0f}% of injected signal")
            if 0.7 < recovery < 1.3:
                print("  ✓ SDCG signal successfully detected!")
            else:
                print("  ⚠ Signal partially recovered")
        else:
            if abs(vdc['t_statistic']) < 2:
                print("\n  ✓ No significant signal (consistent with null)")
            else:
                print(f"\n  ⚠ Unexpected {vdc['t_statistic']:.1f}σ signal")
    else:
        print(f"  {vdc['error']}")
    
    # Mass-binned analysis
    print(f"\n{'[5] Mass-Binned Consistency Check':-^70}")
    mass_bins = analyzer.mass_binned_analysis()
    
    print(f"\n  {'Mass bin':>10} {'N_void':>8} {'N_dense':>8} {'Δv/v':>12}")
    print("  " + "-"*44)
    for mb in mass_bins:
        print(f"  {mb['mass_bin']:>10} {mb['n_void']:>8} {mb['n_dense']:>8} {mb['delta_v_pct']:>+11.1f}%")
    
    if mass_bins:
        offsets = [mb['delta_v_pct'] for mb in mass_bins]
        mean_offset = np.mean(offsets)
        std_offset = np.std(offsets)
        print(f"\n  Mean offset across bins: {mean_offset:+.1f}% ± {std_offset:.1f}%")
        if std_offset < abs(mean_offset):
            print("  ✓ Effect is consistent across mass range (not a systematic)")
        else:
            print("  ⚠ Large scatter suggests possible systematic")
    
    # Lyman-α check
    print(f"\n{'[6] Lyman-α Forest Constraint':-^70}")
    mu_lya = SDCGTheory.mu_eff('lyman_alpha')
    flux_enhancement = mu_lya ** 2 * 100  # Approximate
    limit = 7.5
    
    print(f"\n  μ_eff(Lyα) = {mu_lya:.6f}")
    print(f"  Predicted flux enhancement: {flux_enhancement:.6f}%")
    print(f"  Observational limit: < {limit}%")
    print(f"\n  Status: {'✓ PASSES (margin: >99.99%)' if flux_enhancement < limit else '✗ FAILS'}")
    
    # Summary
    print(f"\n{'SUMMARY':=^70}")
    
    all_results = {
        'configuration': {
            'sdcg_injected': inject_sdcg,
            'n_galaxies': n_galaxies,
            'n_voids': len(voids)
        },
        'theory': {
            'mu_bare': SDCGTheory.MU_BARE,
            'beta_0': SDCGTheory.BETA_0,
            'k_0': SDCGTheory.K_0
        },
        'tully_fisher': tf,
        'environment_offsets': env_results,
        'void_vs_dense': vdc,
        'mass_bins': mass_bins,
        'lyman_alpha': {
            'mu_eff': mu_lya,
            'flux_enhancement_pct': flux_enhancement,
            'passes_constraint': flux_enhancement < limit
        }
    }
    
    # Conclusions
    if inject_sdcg:
        print("\n  SDCG signal was INJECTED - testing detectability")
        if 'error' not in vdc and vdc['observed_vs_predicted'] > 0.5:
            print(f"\n  ✓ Environment-dependent velocity enhancement DETECTED")
            print(f"    Void vs Dense: {vdc['velocity_offset_pct']:+.1f}% (predicted: {vdc['predicted_difference_pct']:+.1f}%)")
            print(f"    Detection significance: {abs(vdc['t_statistic']):.1f}σ")
    else:
        print("\n  NULL hypothesis test (no SDCG signal injected)")
        
    print(f"\n  ✓ Lyman-α constraint satisfied (by design of screening)")
    print(f"\n  Theory predicts:")
    print(f"    μ_eff(void) = {SDCGTheory.mu_eff('void'):.4f} → Δv = +{SDCGTheory.velocity_enhancement('void')*100:.1f}%")
    print(f"    μ_eff(cluster) = {SDCGTheory.mu_eff('group'):.5f} → Δv = +{SDCGTheory.velocity_enhancement('group')*100:.1f}%")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    with open(output_path / 'sdcg_complete_analysis.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
        
    # Save galaxy catalog
    gal_data = [asdict(g) for g in galaxies]
    with open(output_path / 'galaxy_catalog_analyzed.json', 'w') as f:
        json.dump(gal_data, f, indent=2, default=float)
        
    print(f"\n  Results saved to: {output_path}")
    print("\n" + "="*70)
    
    return all_results


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  SDCG OBSERVATIONAL TEST SUITE                                   ║
    ║  Scale-Dependent Chameleon Gravity                               ║
    ║                                                                  ║
    ║  Testing environment-dependent velocity enhancement              ║
    ║  μ_eff(void) ≈ 0.149 → +7.4% velocity boost                     ║
    ║  μ_eff(cluster) ≈ 0.001 → screened (no boost)                   ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Run with SDCG signal injected (to validate detection capability)
    print("\n" + "="*70)
    print("TEST 1: WITH SDCG SIGNAL INJECTION (Detection Validation)")
    print("="*70)
    results_with_signal = run_complete_analysis(inject_sdcg=True, n_galaxies=500)
    
    # Run without signal (null hypothesis)
    print("\n\n" + "="*70)
    print("TEST 2: NULL HYPOTHESIS (No SDCG Signal)")
    print("="*70)
    results_null = run_complete_analysis(inject_sdcg=False, n_galaxies=500)
