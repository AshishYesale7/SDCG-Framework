# CGC Project Value Consistency Report
## Generated: January 2025

---

## ⚠️ CRITICAL: VALUE INCONSISTENCIES DETECTED

The project contains **THREE DIFFERENT** sets of μ values from different analysis runs:

---

## 📊 VALUE COMPARISON TABLE

| Parameter | OLD (v2-v5) | Analysis A (10k MCMC) | Analysis B (+Lyα) | EFT Theory |
|-----------|-------------|----------------------|-------------------|------------|
| **μ** | 0.149 ± 0.025 | 0.411 ± 0.044 | 0.045 ± 0.019 | - |
| **n_g** | 0.138 ± 0.014 | 0.647 ± 0.203 | 0.647 ± 0.203 | 0.014 |
| **z_trans** | 1.64 ± 0.10 | 2.43 ± 1.44 | 2.43 ± 1.44 | 1.67 |
| **Detection** | 6.0σ | 9.4σ | 2.4σ | - |
| **H₀ resolution** | ~36% | 49.5% | 5.4% | - |
| **Lyα enhancement** | ~11% | 136% ❌ | 6.5% ✅ | - |

---

## 📁 FILES WITH OUTDATED VALUES (μ = 0.149)

### Thesis Chapters (NEED UPDATING):
1. `CGC_THESIS_CHAPTER.tex` - μ = 0.149 ± 0.025
2. `CGC_THESIS_CHAPTER_v2.tex` - μ = 0.149 ± 0.025 (6σ)
3. `CGC_THESIS_CHAPTER_v3.tex` - μ = 0.149 ± 0.025 (6σ)
4. `CGC_THESIS_CHAPTER_v4.tex` - μ = 0.149 ± 0.025 (6σ)
5. `CGC_THESIS_CHAPTER_v5.tex` - μ = 0.149 ± 0.025

### Analysis/Test Files:
6. `test_cgc_formulas.py` - μ = 0.149, n_g = 0.138
7. `cgc_desi_analysis.py` - μ = 0.149
8. `CGC_PHYSICS_ANALYSIS.md` - μ = 0.149 ± 0.025
9. `CGC_EQUATIONS_REFERENCE.txt` - μ = 0.149 ± 0.025
10. `cgc/cgc_physics.py` - comment mentions μ = 0.149

---

## 📁 FILES WITH NEW VALUES

### Analysis A (no Lyα constraint):
- `run_lace_v6_analysis.py` - μ = 0.4113 ± 0.0436
- `run_lace_comprehensive_v6.py` - μ = 0.4113
- `run_thesis_lyalpha_transparency.py` - Both μ = 0.411 and μ = 0.045

### Analysis B (with Lyα constraint):
- `run_lace_joint_mcmc_v6.py` - μ = 0.045 (Sol 1)
- `thesis_materials/cgc_lyalpha_table.tex` - Both analyses compared

---

## 🤔 THE CORE ISSUE

### What happened:
1. **Original MCMC** (unknown date): Found μ = 0.149 ± 0.025 (6σ)
2. **10k MCMC** (Jan 30, 2025): Found μ = 0.411 ± 0.044 (9.4σ)
3. **Joint MCMC + Lyα** (Jan 30, 2025): Found μ = 0.045 ± 0.019 (2.4σ)

### Why they differ:
- **Original** → Used smaller dataset or different priors
- **10k MCMC** → Full data without Lyα constraint → VIOLATES Lyα bounds!
- **Joint +Lyα** → Includes Lyα likelihood → Respects DESI 7.5% limit

---

## ✅ WHICH VALUES TO USE?

### RECOMMENDED: Analysis B (μ = 0.045 ± 0.019)

**Reasons:**
1. ✅ Respects Lyα systematic limit (6.5% < 7.5%)
2. ✅ Self-consistent with all observational bounds
3. ✅ 2.4σ detection still indicates genuine effect
4. ✅ Most conservative, defensible in peer review

### ALTERNATIVE: Present BOTH Transparently

If using this approach, clearly state:
- **Without Lyα**: μ = 0.411 ± 0.044 (9.4σ) — but violates Lyα bounds
- **With Lyα**: μ = 0.045 ± 0.019 (2.4σ) — fully self-consistent

---

## 🔧 FILES THAT NEED UPDATING

If adopting Analysis B values:

### HIGH PRIORITY:
1. `CGC_THESIS_CHAPTER_v6.tex` (create new version with correct values)
2. `CGC_EQUATIONS_REFERENCE.txt` → Update summary values
3. `test_cgc_formulas.py` → Update test parameters

### MEDIUM PRIORITY:
4. `CGC_PHYSICS_ANALYSIS.md` → Update discussion
5. `cgc_desi_analysis.py` → Update analysis values
6. `cgc/cgc_physics.py` → Update comments

### LOW PRIORITY (historical):
7-10. Old thesis versions (v2-v5) can remain as version history

---

## 📋 RECOMMENDED FINAL VALUES FOR THESIS v6

```
╔═══════════════════════════════════════════════════════════════════╗
║                    OFFICIAL CGC PARAMETERS                        ║
╠═══════════════════════════════════════════════════════════════════╣
║  Parameter           Value                Note                    ║
╠═══════════════════════════════════════════════════════════════════╣
║  μ                   0.045 ± 0.019        2.4σ detection          ║
║  n_g                 0.647 ± 0.203        Fitted from MCMC        ║
║  n_g (EFT)           0.014                β₀²/4π² prediction      ║
║  z_trans             2.43 ± 1.44          Fitted from MCMC        ║
║  z_trans (EFT)       1.67                 z_acc + Δz prediction   ║
║                                                                   ║
║  H₀ resolution       5.4%                 Reduces 4.8σ → 4.55σ    ║
║  Lyα enhancement     6.5%                 Within 7.5% bound ✅     ║
╠═══════════════════════════════════════════════════════════════════╣
║                    ALTERNATIVE PRESENTATION                       ║
╠═══════════════════════════════════════════════════════════════════╣
║  Without Lyα (μ)     0.411 ± 0.044        9.4σ (but violates Lyα) ║
║  With Lyα (μ)        0.045 ± 0.019        2.4σ (self-consistent)  ║
╚═══════════════════════════════════════════════════════════════════╝
```

---

## 🎯 ACTION ITEMS

### Immediate:
- [ ] Create CGC_THESIS_CHAPTER_v6.tex with transparent Lyα analysis
- [ ] Update CGC_EQUATIONS_REFERENCE.txt with both value sets
- [ ] Run validation tests with new parameters

### Before Submission:
- [ ] Verify all thesis figures use consistent μ values
- [ ] Check that all equations produce correct H₀ predictions
- [ ] Ensure Lyα falsifiability is clearly stated

---

## 📊 SUMMARY

| Metric | OLD (μ=0.149) | Analysis A (μ=0.411) | Analysis B (μ=0.045) |
|--------|---------------|----------------------|----------------------|
| Detection significance | 6σ | 9.4σ | 2.4σ |
| H₀ tension resolution | ~36% | 49.5% | 5.4% |
| Lyα compatible? | Unknown | ❌ NO (136%) | ✅ YES (6.5%) |
| Peer-review defensible | ⚠️ | ❌ NO | ✅ YES |

**Recommendation**: Use Analysis B (μ = 0.045) as the official value, 
or transparently present BOTH analyses showing how Lyα constrains μ.

---

*Report generated by comprehensive value consistency analysis*
