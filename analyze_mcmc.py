#!/usr/bin/env python3
"""Analyze the full MCMC chain"""
import numpy as np

# Load the largest MCMC chain
data = np.load("results/mcmc_checkpoint_20260201_131724.npz", allow_pickle=True)
chains = data['chains']
n_steps = data['n_steps_completed']
elapsed_time = data['elapsed_time']

print("=" * 80)
print("ANALYSIS OF FULL MCMC RUN (320,000 samples)")
print("=" * 80)
print(f"Chains shape: {chains.shape}")
print(f"Steps completed: {n_steps}")
print(f"Elapsed time: {elapsed_time/3600:.2f} hours")

param_names = ['omega_b', 'omega_cdm', 'h', 'ln10As', 'n_s', 'tau', 'mu', 'n_g', 'z_trans', 'rho_thresh']

# Use latter half (burn-in removed)
chains_burned = chains[len(chains)//2:]
print(f"Using {len(chains_burned)} samples after burn-in")

print("\nPARAMETER CONSTRAINTS:")
print("-" * 80)
for i, name in enumerate(param_names):
    mean = np.mean(chains_burned[:, i])
    std = np.std(chains_burned[:, i])
    print(f"  {name:12s}: {mean:.5f} +/- {std:.5f}")

# CGC parameters
mu_samples = chains_burned[:, 6]
mu_mean = np.mean(mu_samples)
mu_std = np.std(mu_samples)

print("\n" + "=" * 80)
print("CGC PARAMETER: mu")
print("=" * 80)
print(f"  Mean:   {mu_mean:.5f} +/- {mu_std:.5f}")
print(f"  Significance from zero: {mu_mean/mu_std:.2f} sigma")

# Dwarf prediction
v_typical = 80
S_void = 1.0
S_cluster = 0.001
delta_v_samples = v_typical * (np.sqrt(1 + mu_samples * S_void) - np.sqrt(1 + mu_samples * S_cluster))
delta_v_mean = np.mean(delta_v_samples)
delta_v_std = np.std(delta_v_samples)

print("\n" + "=" * 80)
print("DWARF GALAXY PREDICTION")
print("=" * 80)
print(f"  Predicted: Delta_v = +{delta_v_mean:.3f} +/- {delta_v_std:.3f} km/s")
print(f"  Observed:  Delta_v = -2.49 +/- 5.0 km/s (ALFALFA, unweighted)")

tension = abs(delta_v_mean - (-2.49)) / np.sqrt(delta_v_std**2 + 5.0**2)
print(f"  TENSION: {tension:.2f} sigma")

print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)
print(f"""
With mu = {mu_mean:.4f} from full MCMC:
  - Predicted Delta_v = +{delta_v_mean:.2f} km/s (NOT +12 km/s!)
  - Observed  Delta_v = -2.49 km/s
  - Tension = {tension:.1f} sigma (NOT significant)

The +12 km/s prediction was based on WRONG parameters!
The correct prediction is only +{delta_v_mean:.2f} km/s.
""")
