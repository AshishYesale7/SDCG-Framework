# Expanded Dwarf Galaxy Dataset Status Report

## Date: February 3, 2026

## Summary

We have successfully expanded the dataset from **72 galaxies** to **3,295 galaxies**:

| Metric                   | Original | Expanded | Increase |
| ------------------------ | -------- | -------- | -------- |
| Total galaxies           | 72       | 3,295    | 46×      |
| Void candidates          | 27       | 74       | 2.7×     |
| Cluster candidates       | 29       | 262      | 9×       |
| With rotation velocities | 56       | 3,272    | 58×      |

## Downloaded Data Sources

### ✅ Successfully Downloaded

1. **ALFALFA α.40** (Haynes et al. 2011)
   - File: `data/alfalfa/alfalfa_a40.csv`
   - Size: 15,856 HI sources
   - Usable dwarfs: 3,251 (W50 < 150 km/s, D < 100 Mpc)
   - Void candidates: 51, Cluster: 241

2. **SPARC Database** (Lelli et al. 2016)
   - File: `data/sparc/sparc_data.mrt`
   - Size: 175 galaxies with high-quality rotation curves
   - Status: Needs fixed-width format parsing

3. **VGS - Void Galaxy Survey** (Kreckel et al. 2012)
   - File: `data/vgs/vgs_votable.xml`
   - Size: 60 void galaxies with δ < -0.5

4. **Manual Catalogs** (Literature compilation)
   - Void dwarfs: 23 verified from Pustilnik+2019, Karachentsev+2013
   - Cluster dwarfs: 21 verified from Toloba+2015, Eigenthaler+2018

### ❌ Need Alternative Approach

1. **LITTLE THINGS** - VizieR TAP query failed
   - Recommendation: Direct download from NRAO
   - URL: https://science.nrao.edu/science/surveys/littlethings

2. **Local Volume Galaxy Catalog** - VizieR TAP failed
   - Recommendation: Try VizieR web interface manually
   - URL: https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/AJ/145/101

3. **NGVS Virgo dwarfs** - VizieR TAP failed
   - Recommendation: Use VCC or EVCC catalogs
4. **Fornax dwarfs** - VizieR TAP failed
   - Recommendation: Use FCC catalog or FDS papers

## Environment Classification Method

Galaxies classified by proximity to known structures:

### Clusters (within angular radius & distance)

- Virgo: RA=187.7°, Dec=12.4°, D=16.5 Mpc, r=8°
- Fornax: RA=54.6°, Dec=-35.5°, D=19 Mpc, r=4°
- Coma: RA=195°, Dec=28°, D=100 Mpc, r=2.5°

### Voids (within angular radius & distance range)

- Local Void: RA=295°, Dec=5°, D=1-25 Mpc, r=40°
- Lynx-Cancer: RA=130°, Dec=40°, D=10-35 Mpc, r=25°
- CVn Void: RA=190°, Dec=35°, D=3-15 Mpc, r=15°

## Preliminary Analysis Results

**Current result (ALFALFA-based):**

- Void mean: 40.8 ± 3.1 km/s (N=51)
- Cluster mean: 41.7 ± 1.4 km/s (N=262)
- Δv = -1.0 ± 3.4 km/s (0.3σ)

⚠️ **Note:** This differs from the original +14.7 km/s result because:

1. ALFALFA W50 is line width, not rotation velocity
2. Needs proper inclination correction
3. Original analysis used carefully selected samples with resolved rotation curves

## Recommendations for Improvement

### Priority 1: Fix SPARC Parsing

The SPARC database has 175 galaxies with **actual rotation velocities** (Vflat column).
These need proper fixed-width format parsing.

### Priority 2: Cross-match with Void Catalogs

Use published void catalogs to improve environment classification:

- Pan et al. (2012) SDSS void catalog
- Rojas et al. (2005) SDSS void galaxies
- Pustilnik+2011,2019 Lynx-Cancer void

### Priority 3: Add LITTLE THINGS

41 nearby dwarfs with excellent HI rotation curves from VLA.

### Priority 4: Proper V_rot Calculation

For ALFALFA: V_rot = W50 / (2 × sin(i))
Need inclination estimates from optical axis ratios.

## File Locations

```
data/
├── alfalfa/
│   └── alfalfa_a40.csv          # 15,856 HI sources
├── sparc/
│   └── sparc_data.mrt           # 175 rotation curves
├── vgs/
│   ├── vgs_votable.xml          # 60 void galaxies
│   └── vgs_hi.tsv               # HI properties
├── dwarfs/
│   ├── verified_void_dwarfs.json    # 23 manual
│   └── verified_cluster_dwarfs.json # 21 manual
├── expanded_dwarf_dataset.json      # Compiled dataset
├── sdcg_analysis_subset.json        # For SDCG test
└── download_manifest_v2.json        # Download log
```

## Next Steps

1. Fix SPARC parsing to extract all 175 galaxies with Vflat
2. Apply better void identification using density field reconstructions
3. Match with original 72-galaxy sample to validate
4. Re-run SDCG analysis with expanded+validated dataset
