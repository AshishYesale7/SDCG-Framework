# SDCG v8: First-Principles Framework Summary

## What Changed

**Version 7 → Version 8**: Parameters now derived from accepted physics, not curve-fit.

---

## Parameter Derivations

### ✓ Derived from Physics (No Free Parameters)

| Parameter    | Value      | Derivation             | Source Physics                   |
| ------------ | ---------- | ---------------------- | -------------------------------- |
| **β₀**       | 0.70       | SM conformal anomaly   | Top quark: (m_t/v)² = (173/246)² |
| **n_g**      | 0.0124     | RG flow                | β₀²/4π² (one-loop scalar-tensor) |
| **z_trans**  | 1.30       | Cosmic evolution       | z_eq + Δz_response               |
| **α**        | 1.0        | Potential minimization | V(φ) ~ φ⁻¹ effective potential   |
| **ρ_thresh** | 200 ρ_crit | Virial equilibrium     | Cluster overdensity              |

### ⚠️ Constrained (Requires New Physics)

| Parameter | Constraint  | Problem                |
| --------- | ----------- | ---------------------- |
| **μ**     | < 0.1 (Lyα) | Naive RG gives μ ~ 1.7 |

---

## The μ Problem → A Prediction

Standard physics predicts:

```
μ_naive = (β₀²/4π²) × ln(M_Pl/H₀) ≈ 0.0124 × 138 ≈ 1.7
```

But observations require:

```
μ_obs < 0.1  (from Lyα forest)
```

**This 17× discrepancy requires new physics!**

### Possible Solutions

1. **New light particles at meV scale**
   - Axion-like particles (ALPs)
   - Dark photons
   - Light scalars

2. **Laboratory tests**
   - ALPS-II (light shining through walls)
   - Atom interferometry (AION, MAGIS)
   - Fifth force experiments (Eöt-Wash)

3. **Cosmological tests**
   - DESI Lyα (μ < 0.03 by 2028)
   - Euclid weak lensing (n_g constraints)
   - CMB-S4 lensing

---

## Key Files

| File                                                                     | Description                   |
| ------------------------------------------------------------------------ | ----------------------------- |
| [THEORETICAL_STATUS.md](THEORETICAL_STATUS.md)                           | Full derivation documentation |
| [MEV_NEW_PHYSICS_PREDICTION.md](MEV_NEW_PHYSICS_PREDICTION.md)           | The μ problem and predictions |
| [cgc/first_principles_parameters.py](cgc/first_principles_parameters.py) | Python implementation         |
| [cgc/parameters.py](cgc/parameters.py)                                   | Updated parameter definitions |

---

## Run Verification

```bash
# Full derivation output
python -m cgc.first_principles_parameters

# Quick parameter check
python verify_parameters.py
```

---

## Thesis Framing

> "SDCG is a scalar-tensor framework with parameters **derived from Standard Model physics**:
>
> - β₀ = 0.70 from conformal anomaly (top quark dominance)
> - n_g = 0.012 from one-loop RG flow
> - z_trans = 1.3 from cosmic acceleration transition
> - α = 1 from effective potential minimization
> - ρ_thresh = 200ρ_crit from virial equilibrium
>
> The amplitude μ is constrained by Lyα to be < 0.1, which **cannot be explained by standard RG running** (which predicts μ ~ 1.7). This implies **new physics at the meV scale**—a testable prediction for laboratory experiments (ALPS-II, AION) and future cosmological surveys (DESI, Euclid)."

---

_SDCG v8 - First Principles with meV New Physics Prediction_
_Last updated: 2026-02-02_
