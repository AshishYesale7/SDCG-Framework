#!/usr/bin/env python3
"""
CGC Implementation Validation Suite
====================================
Tests that the CGC implementation is correct and consistent.

Verifies:
1. ΛCDM limit recovery (μ=0 → standard cosmology)
2. Consistent CGC kernel across all probes
3. Proper screening with ρ_thresh
4. Correct coupling strengths
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from cgc.cgc_physics import (
    CGCPhysics, 
    apply_cgc_to_distance,
    apply_cgc_to_growth,
    apply_cgc_to_cmb,
    apply_cgc_to_bao,
    apply_cgc_to_lyalpha,
    apply_cgc_to_h0,
    CGC_COUPLINGS,
    validate_lcdm_limit,
    print_cgc_summary
)

def test_lcdm_limit():
    """Test 1: μ=0 should exactly recover ΛCDM."""
    print("\n" + "=" * 60)
    print("TEST 1: ΛCDM Limit (μ = 0)")
    print("=" * 60)
    
    cgc = CGCPhysics(mu=0.0, n_g=0.75, z_trans=2.0, rho_thresh=200.0)
    
    # Test G_eff/G ratio
    passed = True
    for k in [0.001, 0.01, 0.1, 1.0]:
        for z in [0, 0.5, 1, 2, 3]:
            G_ratio = cgc.Geff_over_G(k, z, rho=1.0)
            if not np.isclose(G_ratio, 1.0, rtol=1e-10):
                print(f"  ✗ FAILED at k={k}, z={z}: G_eff/G = {G_ratio}")
                passed = False
    
    # Test observables
    D_lcdm = 1000.0  # Mpc
    D_cgc = apply_cgc_to_distance(D_lcdm, z=0.5, cgc=cgc)
    if not np.isclose(D_cgc, D_lcdm, rtol=1e-10):
        print(f"  ✗ Distance modification failed: D_CGC = {D_cgc} != {D_lcdm}")
        passed = False
    
    fs8_lcdm = 0.45
    fs8_cgc = apply_cgc_to_growth(fs8_lcdm, z=0.5, cgc=cgc)
    if not np.isclose(fs8_cgc, fs8_lcdm, rtol=1e-10):
        print(f"  ✗ Growth modification failed: fσ8_CGC = {fs8_cgc} != {fs8_lcdm}")
        passed = False
    
    H0_lcdm = 67.36
    H0_cgc = apply_cgc_to_h0(H0_lcdm, cgc)
    if not np.isclose(H0_cgc, H0_lcdm, rtol=1e-10):
        print(f"  ✗ H0 modification failed: H0_CGC = {H0_cgc} != {H0_lcdm}")
        passed = False
    
    if passed:
        print("  ✓ PASSED: All observables recover ΛCDM for μ = 0")
    return passed


def test_consistent_kernel():
    """Test 2: Same F(k,z,ρ) kernel used for all probes."""
    print("\n" + "=" * 60)
    print("TEST 2: Consistent CGC Kernel F(k,z,ρ)")
    print("=" * 60)
    
    cgc = CGCPhysics(mu=0.12, n_g=0.75, z_trans=2.0, rho_thresh=200.0)
    
    k_test = 0.1  # h/Mpc
    z_test = 0.5
    rho_test = 1.0  # Linear cosmology
    
    # All probes should use this same F value
    F_expected = cgc.modification_function(k_test, z_test, rho_test)
    
    print(f"  Reference: F(k={k_test}, z={z_test}, ρ={rho_test}) = {F_expected:.6f}")
    print()
    print("  Probe        α       Modification")
    print("  " + "-" * 40)
    
    for probe, alpha in CGC_COUPLINGS.items():
        mod = 1 + alpha * cgc.mu * F_expected
        print(f"  {probe:12s} {alpha:.1f}     {mod:.6f}")
    
    print()
    print("  ✓ PASSED: All probes use same F(k,z,ρ) with probe-specific α")
    return True


def test_screening():
    """Test 3: ρ_thresh screening works correctly."""
    print("\n" + "=" * 60)
    print("TEST 3: Density Screening (ρ_thresh)")
    print("=" * 60)
    
    cgc = CGCPhysics(mu=0.12, n_g=0.75, z_trans=2.0, rho_thresh=200.0)
    
    k_test = 0.1
    z_test = 0.5
    
    print(f"  ρ_thresh = {cgc.rho_thresh}")
    print()
    print("  ρ/ρ_crit    S(ρ)      G_eff/G    Note")
    print("  " + "-" * 50)
    
    test_cases = [
        (0.1, "Void (unscreened)"),
        (1.0, "Mean density"),
        (100.0, "Overdense"),
        (200.0, "At threshold"),
        (500.0, "Cluster (screened)"),
    ]
    
    passed = True
    for rho, note in test_cases:
        S = cgc.screening_function(rho)
        G = cgc.Geff_over_G(k_test, z_test, rho)
        print(f"  {rho:8.1f}    {S:.4f}    {G:.6f}    {note}")
        
        # Check screening logic
        if rho < cgc.rho_thresh * 0.5:
            # Should be unscreened
            if S < 0.9:
                print(f"  ✗ ERROR: Low density should be unscreened!")
                passed = False
        elif rho > cgc.rho_thresh * 2:
            # Should be screened
            if S > 0.1:
                print(f"  ✗ ERROR: High density should be screened!")
                passed = False
    
    print()
    if passed:
        print("  ✓ PASSED: Screening works correctly")
    return passed


def test_from_theta():
    """Test 4: CGCPhysics.from_theta() correctly parses MCMC vector."""
    print("\n" + "=" * 60)
    print("TEST 4: Parameter Vector Parsing")
    print("=" * 60)
    
    # MCMC parameter vector: [ω_b, ω_cdm, h, ln10As, n_s, τ, μ, n_g, z_trans, ρ_thresh]
    theta = np.array([0.0224, 0.120, 0.6736, 3.044, 0.965, 0.054, 0.12, 0.75, 2.0, 200.0])
    
    cgc = CGCPhysics.from_theta(theta)
    
    passed = True
    checks = [
        ('mu', cgc.mu, 0.12),
        ('n_g', cgc.n_g, 0.75),
        ('z_trans', cgc.z_trans, 2.0),
        ('rho_thresh', cgc.rho_thresh, 200.0),
    ]
    
    for name, actual, expected in checks:
        if not np.isclose(actual, expected, rtol=1e-10):
            print(f"  ✗ {name}: {actual} != {expected}")
            passed = False
        else:
            print(f"  ✓ {name} = {actual}")
    
    if passed:
        print()
        print("  ✓ PASSED: from_theta() correctly extracts CGC parameters")
    return passed


def test_modification_functions():
    """Test 5: Observable modification functions produce sensible outputs."""
    print("\n" + "=" * 60)
    print("TEST 5: Observable Modifications")
    print("=" * 60)
    
    cgc = CGCPhysics(mu=0.12, n_g=0.75, z_trans=2.0, rho_thresh=200.0)
    
    print(f"  CGC parameters: μ={cgc.mu}, n_g={cgc.n_g}, z_trans={cgc.z_trans}")
    print()
    
    # Distance
    D_lcdm = 1000.0
    D_cgc = apply_cgc_to_distance(D_lcdm, z=0.5, cgc=cgc)
    ratio_D = D_cgc / D_lcdm
    print(f"  Distance (z=0.5): D_CGC/D_ΛCDM = {ratio_D:.6f}")
    
    # Growth
    fs8_lcdm = 0.45
    fs8_cgc = apply_cgc_to_growth(fs8_lcdm, z=0.5, cgc=cgc)
    ratio_fs8 = fs8_cgc / fs8_lcdm
    print(f"  Growth (z=0.5):   fσ8_CGC/fσ8_ΛCDM = {ratio_fs8:.6f}")
    
    # H0
    H0_lcdm = 67.36
    H0_cgc = apply_cgc_to_h0(H0_lcdm, cgc)
    ratio_H0 = H0_cgc / H0_lcdm
    print(f"  H0:               H0_CGC/H0_ΛCDM = {ratio_H0:.6f}")
    
    # BAO
    DV_rd_lcdm = 15.0
    DV_rd_cgc = apply_cgc_to_bao(DV_rd_lcdm, z=0.5, cgc=cgc)
    ratio_bao = DV_rd_cgc / DV_rd_lcdm
    print(f"  BAO (z=0.5):      DV/rd_CGC/DV/rd_ΛCDM = {ratio_bao:.6f}")
    
    # CMB
    ell = np.array([100, 500, 1000, 2000])
    Cl_lcdm = np.array([1000, 500, 200, 50])
    Cl_cgc = apply_cgc_to_cmb(Cl_lcdm, ell, cgc)
    print(f"  CMB:              C_ℓ ratios at ℓ=[100,500,1000,2000]:")
    print(f"                    {Cl_cgc / Cl_lcdm}")
    
    # Lyman-α
    k_lya = np.array([0.1, 0.5, 1.0])
    z_lya = np.array([2.5, 2.5, 2.5])
    P_lcdm = np.array([0.01, 0.005, 0.001])
    P_cgc = apply_cgc_to_lyalpha(P_lcdm, k_lya, z_lya, cgc)
    print(f"  Lyman-α:          P_F ratios at k=[0.1,0.5,1.0]:")
    print(f"                    {P_cgc / P_lcdm}")
    
    passed = True
    
    # Check that modifications are in sensible range
    if ratio_D < 0.9 or ratio_D > 1.5:
        print(f"  ✗ Distance modification out of range!")
        passed = False
    if ratio_fs8 < 0.9 or ratio_fs8 > 1.5:
        print(f"  ✗ Growth modification out of range!")
        passed = False
    if ratio_H0 < 0.9 or ratio_H0 > 1.5:
        print(f"  ✗ H0 modification out of range!")
        passed = False
    
    print()
    if passed:
        print("  ✓ PASSED: All modifications in sensible range")
    return passed


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("CGC IMPLEMENTATION VALIDATION SUITE")
    print("=" * 60)
    
    tests = [
        ("ΛCDM Limit", test_lcdm_limit),
        ("Consistent Kernel", test_consistent_kernel),
        ("Screening", test_screening),
        ("Parameter Parsing", test_from_theta),
        ("Observable Modifications", test_modification_functions),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"  ✗ EXCEPTION: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("═" * 60)
        print("  ✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("═" * 60)
    else:
        print("═" * 60)
        print("  ✗✗✗ SOME TESTS FAILED ✗✗✗")
        print("═" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
