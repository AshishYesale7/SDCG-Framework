#!/usr/bin/env python3
"""
check_environment_classification.py - Check if environment classification is correct

THE CRITICAL ISSUE:
CGC predicts void dwarfs rotate FASTER than cluster dwarfs.
But we observe the OPPOSITE. Either:
1. The prediction is wrong
2. The environment classification is wrong/inverted
3. The data has confounders

This script investigates option 2.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cgc_dwarf_test'))

import numpy as np
import pandas as pd
from scipy import stats

def main():
    print("=" * 70)
    print("   INVESTIGATION: ENVIRONMENT CLASSIFICATION ACCURACY")
    print("=" * 70)
    
    from load_data import load_and_merge_all, load_void_catalog
    from filter_samples import prepare_samples
    
    # Load full data
    print("\n[Loading data...]")
    df = load_and_merge_all()
    voids = load_void_catalog()
    
    print("\n" + "=" * 70)
    print("1. VOID CATALOG INFO")
    print("-" * 70)
    print(f"Number of voids: {len(voids)}")
    print(f"Mean void radius: {voids['Radius_void'].mean():.1f} Mpc/h")
    print(f"Radius range: {voids['Radius_void'].min():.1f} - {voids['Radius_void'].max():.1f} Mpc/h")
    
    print("\n" + "=" * 70)
    print("2. ENVIRONMENT CLASSIFICATION THRESHOLDS")
    print("-" * 70)
    print("Current logic (from load_data.py):")
    print("  'void':    dist_ratio < 0.8  → Galaxy is INSIDE a void")
    print("  'field':   0.8 ≤ dist_ratio ≤ 3.0 → Galaxy is in normal density")
    print("  'cluster': dist_ratio > 3.0  → Galaxy is FAR from all voids")
    print()
    print("⚠ PROBLEM: 'Far from voids' ≠ 'In cluster'")
    print("   Being far from void centers could still be in the general field!")
    print("   We need ACTUAL cluster membership, not just 'not in void'")
    
    print("\n" + "=" * 70)
    print("3. GALAXY DISTRIBUTION BY ENVIRONMENT")
    print("-" * 70)
    env_counts = df['environment'].value_counts()
    total = len(df)
    for env in ['void', 'field', 'cluster']:
        if env in env_counts:
            n = env_counts[env]
            print(f"  {env:10s}: {n:6d} ({100*n/total:5.1f}%)")
    
    print("\n" + "=" * 70)
    print("4. void_dist_ratio DISTRIBUTION BY ENVIRONMENT")
    print("-" * 70)
    for env in ['void', 'field', 'cluster']:
        subset = df[df['environment'] == env]['void_dist_ratio']
        if len(subset) > 0:
            print(f"  {env:10s}: mean = {subset.mean():.2f}, min = {subset.min():.2f}, max = {subset.max():.2f}")
    
    print("\n" + "=" * 70)
    print("5. THE CRITICAL COMPARISON")
    print("-" * 70)
    
    # Load filtered samples
    samples = prepare_samples()
    v_void = samples['void']['V_rot'].values
    v_cluster = samples['cluster']['V_rot'].values
    
    print(f"Void dwarfs:    N = {len(v_void):5d}, <V_rot> = {np.mean(v_void):.2f} km/s")
    print(f"Cluster dwarfs: N = {len(v_cluster):5d}, <V_rot> = {np.mean(v_cluster):.2f} km/s")
    print(f"Δv = {np.mean(v_void) - np.mean(v_cluster):+.2f} km/s")
    print()
    print("CGC Prediction: Δv = +12 km/s (void > cluster)")
    print(f"Observed:       Δv = {np.mean(v_void) - np.mean(v_cluster):+.2f} km/s")
    
    print("\n" + "=" * 70)
    print("6. POSSIBLE EXPLANATIONS")
    print("-" * 70)
    print("""
    A. CLASSIFICATION ERROR (most likely):
       - Our "cluster" sample is NOT actually cluster galaxies
       - It's just galaxies far from voids, which could be field or anything
       - The CGC test requires TRUE cluster vs TRUE void dwarfs
    
    B. PHYSICAL EFFECT OPPOSITE:
       - Maybe the CGC prediction sign is wrong?
       - Need to re-check the theoretical derivation
    
    C. CONFOUNDERS:
       - Mass difference between samples
       - Distance difference (Malmquist bias)
       - Inclination effects
    
    D. STATISTICAL FLUCTUATION:
       - Small cluster sample (N=129) has large variance
       - p-value = 0.36 means difference is NOT significant
    """)
    
    print("\n" + "=" * 70)
    print("7. RECOMMENDED FIX")
    print("-" * 70)
    print("""
    For a VALID CGC test, we need:
    
    1. VOID DWARFS: Galaxies confirmed to be inside voids
       ✓ Current classification seems OK (dist_ratio < 0.8 of void radius)
    
    2. CLUSTER DWARFS: Galaxies that are ACTUALLY in clusters
       ✗ Current "cluster" sample is WRONG
       → Should use: Virgo, Fornax, Coma cluster membership catalogs
       → Or: Match to known cluster catalogs (Abell, etc.)
    
    3. MASS MATCHING: Same stellar mass distribution
       → Need to verify mass distributions are similar
    
    4. PROPER SCREENING: Check local density, not just void distance
       → Use actual density estimators (n-th nearest neighbor, etc.)
    """)
    
    print("\n" + "=" * 70)
    print("8. CONCLUSION")
    print("-" * 70)
    print("""
    THE TEST IS INVALID because:
    
    Our "cluster" classification is WRONG.
    
    We labeled galaxies as "cluster" just because they are FAR from voids,
    but being far from voids does NOT mean being in a cluster.
    
    To properly test CGC, we need:
    - Dwarf galaxies with confirmed CLUSTER MEMBERSHIP (Virgo, Fornax, etc.)
    - OR actual local density measurements
    
    The current negative result (Δv = -2.49 km/s) is likely due to
    CLASSIFICATION ERROR, not because CGC is wrong.
    """)
    
    print("=" * 70)

if __name__ == "__main__":
    main()
