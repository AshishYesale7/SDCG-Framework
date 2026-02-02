#!/usr/bin/env python3
"""
Test CGC equations match CGC_EQUATIONS_REFERENCE.txt
"""
from cgc.cgc_physics import (
    CGCPhysics, 
    apply_cgc_to_sne_distance, 
    apply_cgc_to_bao, 
    apply_cgc_to_cmb, 
    apply_cgc_to_growth, 
    apply_cgc_to_h0,
    apply_cgc_to_lyalpha
)
import numpy as np

print("=" * 60)
print("CGC EQUATIONS VERIFICATION")
print("Checking implementation matches CGC_EQUATIONS_REFERENCE.txt")
print("=" * 60)

# Test parameters - Lyα-constrained values (OFFICIAL for thesis v6)
# Analysis B: μ = 0.045 ± 0.019 (2.4σ, Lyα-consistent)
# EFT predictions: n_g = β₀²/4π² = 0.014, z_trans = 1.67
MU_OFFICIAL = 0.045  # Lyα-constrained
N_G_EFT = 0.014      # From β₀²/4π² with β₀ = 0.74
Z_TRANS_EFT = 1.67   # From z_acc + Δz

# For comparison: Analysis A (unconstrained, violates Lyα)
# MU_UNCONSTRAINED = 0.411 (9.4σ, but 136% Lyα enhancement exceeds 7.5% limit)

cgc = CGCPhysics(mu=MU_OFFICIAL, n_g=N_G_EFT, z_trans=Z_TRANS_EFT, rho_thresh=200.0)
print(f"\nUsing Lyα-constrained parameters:")
print(f"  μ = {MU_OFFICIAL}, n_g = {N_G_EFT}, z_trans = {Z_TRANS_EFT}")

all_passed = True

# Test 1: SNe distance modification
print("\n[1] SNe Distance: D_L^CGC = D_L × [1 + 0.5μ × (1 - exp(-z/z_trans))]")
z = 1.0
sne_ratio = apply_cgc_to_sne_distance(1.0, z, cgc)
expected_sne = 1 + 0.5 * MU_OFFICIAL * (1 - np.exp(-z/Z_TRANS_EFT))
match = np.isclose(sne_ratio, expected_sne, rtol=1e-6)
print(f"  z=1.0: {sne_ratio:.8f} vs expected {expected_sne:.8f} {'✓' if match else '✗'}")
all_passed &= match

# Test 2: BAO modification
print("\n[2] BAO: (D_V/r_d)^CGC = (D_V/r_d)^LCDM × [1 + μ × (1+z)^(-n_g)]")
z = 0.5
bao_ratio = apply_cgc_to_bao(1.0, z, cgc)
expected_bao = 1 + MU_OFFICIAL * (1+z)**(-N_G_EFT)
match = np.isclose(bao_ratio, expected_bao, rtol=1e-6)
print(f"  z=0.5: {bao_ratio:.8f} vs expected {expected_bao:.8f} {'✓' if match else '✗'}")
all_passed &= match

# Test 3: CMB modification
print("\n[3] CMB: D_ℓ^CGC = D_ℓ × [1 + μ × (ℓ/1000)^(n_g/2)]")
ell = 1000
cmb_ratio = apply_cgc_to_cmb(np.array([1.0]), np.array([ell]), cgc)[0]
expected_cmb = 1 + MU_OFFICIAL * (ell/1000)**(N_G_EFT/2)
match = np.isclose(cmb_ratio, expected_cmb, rtol=1e-6)
print(f"  ℓ=1000: {cmb_ratio:.8f} vs expected {expected_cmb:.8f} {'✓' if match else '✗'}")
all_passed &= match

# Test 4: Growth modification
print("\n[4] Growth: fσ8_CGC = fσ8 × [1 + 0.1μ × (1+z)^(-n_g)]")
z = 0.5
growth_ratio = apply_cgc_to_growth(1.0, z, cgc)
expected_growth = 1 + 0.1 * MU_OFFICIAL * (1+z)**(-N_G_EFT)
match = np.isclose(growth_ratio, expected_growth, rtol=1e-6)
print(f"  z=0.5: {growth_ratio:.8f} vs expected {expected_growth:.8f} {'✓' if match else '✗'}")
all_passed &= match

# Test 5: H0 modification
print("\n[5] H0: H0_eff = H0 × (1 + 0.1μ)")
h0_ratio = apply_cgc_to_h0(70.0, cgc) / 70.0
expected_h0 = 1 + 0.1 * MU_OFFICIAL
match = np.isclose(h0_ratio, expected_h0, rtol=1e-6)
print(f"  {h0_ratio:.8f} vs expected {expected_h0:.8f} {'✓' if match else '✗'}")
all_passed &= match

# Test 6: Lyman-α modification
print("\n[6] Lyman-α: P_F^CGC = P_F × [1 + μ × (k/k_CGC)^n_g × W(z)]")
print("  (k_CGC = 0.1(1+μ), W(z) = Gaussian window)")
k = 0.1
z = 2.0
lya = apply_cgc_to_lyalpha(np.array([1.0]), np.array([k]), np.array([z]), cgc)[0]
k_cgc = 0.1 * (1 + MU_OFFICIAL)
W_z = np.exp(-0.5 * ((z - Z_TRANS_EFT) / 1.5)**2)
expected_lya = 1 + MU_OFFICIAL * (k / k_cgc)**N_G_EFT * W_z
match = np.isclose(lya, expected_lya, rtol=1e-6)
print(f"  k=0.1, z=2: {lya:.8f} vs expected {expected_lya:.8f} {'✓' if match else '✗'}")
all_passed &= match

# Test 7: ΛCDM limit (μ=0)
print("\n[7] ΛCDM Limit (μ=0): All modifications = 1.0")
cgc0 = CGCPhysics(mu=0.0, n_g=N_G_EFT, z_trans=Z_TRANS_EFT, rho_thresh=200.0)
sne_0 = apply_cgc_to_sne_distance(1.0, 1.0, cgc0)
bao_0 = apply_cgc_to_bao(1.0, 0.5, cgc0)
cmb_0 = apply_cgc_to_cmb(np.array([1.0]), np.array([1000]), cgc0)[0]
growth_0 = apply_cgc_to_growth(1.0, 0.5, cgc0)
h0_0 = apply_cgc_to_h0(70.0, cgc0) / 70.0

lcdm_ok = all([
    np.isclose(sne_0, 1.0, rtol=1e-10),
    np.isclose(bao_0, 1.0, rtol=1e-10),
    np.isclose(cmb_0, 1.0, rtol=1e-10),
    np.isclose(growth_0, 1.0, rtol=1e-10),
    np.isclose(h0_0, 1.0, rtol=1e-10),
])
print(f"  SNe:    {sne_0:.10f}")
print(f"  BAO:    {bao_0:.10f}")
print(f"  CMB:    {cmb_0:.10f}")
print(f"  Growth: {growth_0:.10f}")
print(f"  H0:     {h0_0:.10f}")
print(f"  ΛCDM limit: {'✓ PASSED' if lcdm_ok else '✗ FAILED'}")
all_passed &= lcdm_ok

# Summary
print("\n" + "=" * 60)
if all_passed:
    print("✓ ALL TESTS PASSED - CGC equations match reference!")
else:
    print("✗ SOME TESTS FAILED - Check equations!")
print("=" * 60)
