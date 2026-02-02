#!/bin/bash
# =============================================================================
# SDCG Observational Data Collection Script
# Downloads all necessary datasets for immediate CGC theory testing
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "SDCG Data Collection Pipeline"
echo "Starting download at $(date)"
echo "=============================================="

# Create directory structure
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$BASE_DIR"/{sparc,voids,lya,growth,cmb,environment,simulations}

# =============================================================================
# 1. SPARC Database (175 disk galaxies with rotation curves)
# =============================================================================
echo ""
echo "[1/7] Downloading SPARC data..."
cd "$BASE_DIR/sparc"

if [ ! -f "SPARC.zip" ]; then
    wget -q --show-progress http://astroweb.cwru.edu/SPARC/SPARC.zip || {
        echo "Warning: SPARC download failed, will use synthetic data"
        touch SPARC_DOWNLOAD_FAILED
    }
fi

if [ -f "SPARC.zip" ]; then
    unzip -o SPARC.zip 2>/dev/null || true
    echo "  ✓ SPARC data extracted"
fi

# =============================================================================
# 2. Void Catalogs
# =============================================================================
echo ""
echo "[2/7] Downloading void catalogs..."
cd "$BASE_DIR/voids"

# Sutter et al. DR7 voids
if [ ! -f "void_stats_dr7_beta.tar.gz" ]; then
    wget -q --show-progress https://www.cosmo.bnl.gov/voidcatalogs/void_stats_dr7_beta.tar.gz 2>/dev/null || {
        echo "  Warning: Sutter voids download failed"
        touch SUTTER_DOWNLOAD_FAILED
    }
fi

if [ -f "void_stats_dr7_beta.tar.gz" ]; then
    tar -xzf void_stats_dr7_beta.tar.gz 2>/dev/null || true
    echo "  ✓ Sutter DR7 voids extracted"
fi

# Try BOSS DR12 voids
wget -q --show-progress https://data.sdss.org/sas/dr12/boss/lss/void_catalogs/dr12_voids.fits 2>/dev/null || {
    echo "  Warning: BOSS DR12 voids download failed"
}

echo "  ✓ Void catalogs complete"

# =============================================================================
# 3. Lyman-α Forest Data
# =============================================================================
echo ""
echo "[3/7] Downloading Lyman-α data..."
cd "$BASE_DIR/lya"

# BOSS DR12 flux power spectrum
wget -q --show-progress https://data.sdss.org/sas/dr12/boss/lya/FluxPower/lya_flux_power_dr12.fits 2>/dev/null || {
    echo "  Warning: Lyman-α flux power download failed"
    touch LYA_DOWNLOAD_FAILED
}

# Mean flux
wget -q --show-progress https://data.sdss.org/sas/dr12/boss/lya/MeanFlux/mean_flux_dr12.fits 2>/dev/null || {
    echo "  Warning: Mean flux download failed"
}

echo "  ✓ Lyman-α data complete"

# =============================================================================
# 4. Growth Rate fσ₈ Compilation
# =============================================================================
echo ""
echo "[4/7] Downloading growth rate data..."
cd "$BASE_DIR/growth"

# Sagredo et al. compilation
wget -q --show-progress "https://cdsarc.cds.unistra.fr/viz-bin/nph-Cat/ftp.gz?J/MNRAS/476/5165/tablea1.dat.gz" -O fsigma8_compilation.dat.gz 2>/dev/null || {
    echo "  Warning: Growth rate compilation download failed"
    touch GROWTH_DOWNLOAD_FAILED
}

if [ -f "fsigma8_compilation.dat.gz" ]; then
    gunzip -f fsigma8_compilation.dat.gz 2>/dev/null || true
    echo "  ✓ Growth rate data extracted"
fi

echo "  ✓ Growth rate data complete"

# =============================================================================
# 5. Environment/Density Catalogs
# =============================================================================
echo ""
echo "[5/7] Downloading environment catalogs..."
cd "$BASE_DIR/environment"

# 2MRS density field
wget -q --show-progress "https://cdsarc.cds.unistra.fr/viz-bin/nph-Cat/ftp.gz?J/ApJS/199/26/table3.dat.gz" -O 2mrs_density.dat.gz 2>/dev/null || {
    echo "  Warning: 2MRS density download failed"
}

if [ -f "2mrs_density.dat.gz" ]; then
    gunzip -f 2mrs_density.dat.gz 2>/dev/null || true
fi

echo "  ✓ Environment catalogs complete"

# =============================================================================
# 6. ALFALFA HI Data
# =============================================================================
echo ""
echo "[6/7] Downloading ALFALFA HI data..."
cd "$BASE_DIR/sparc"

wget -q --show-progress https://egg.astro.cornell.edu/alfalfa/data/a40files/a40cat.csv 2>/dev/null || {
    echo "  Warning: ALFALFA catalog download failed"
}

echo "  ✓ ALFALFA data complete"

# =============================================================================
# 7. Summary
# =============================================================================
echo ""
echo "[7/7] Creating data summary..."
cd "$BASE_DIR"

cat > DATA_README.md << 'EOF'
# SDCG Observational Test Data

## Downloaded Datasets:

### 1. SPARC (sparc/)
- 175 disk galaxies with high-quality rotation curves
- Surface brightness profiles
- Source: Lelli et al. (2016)

### 2. Void Catalogs (voids/)
- SDSS DR7 void catalog (Sutter et al. 2012)
- BOSS DR12 void catalog
- ~1000 voids with R > 10 Mpc/h

### 3. Lyman-α Forest (lya/)
- BOSS DR12 flux power spectrum
- Mean transmitted flux evolution
- z = 2.2 - 4.4

### 4. Growth Rate fσ₈ (growth/)
- Compilation of 63 measurements
- z = 0.02 - 1.52
- Source: Sagredo et al. (2018)

### 5. Environment Catalogs (environment/)
- 2MRS density field
- Local universe (z < 0.05)

## Usage:
```python
from analyze_sdcg_data import SDCGDataAnalyzer
analyzer = SDCGDataAnalyzer('.')
analyzer.run_full_analysis()
```

## Citation:
Remember to cite each survey when publishing results.
EOF

echo ""
echo "=============================================="
echo "Download Summary"
echo "=============================================="
echo "Total data size: $(du -sh "$BASE_DIR" | cut -f1)"
echo ""
ls -la "$BASE_DIR"/*/
echo ""
echo "Download completed at $(date)"
echo "=============================================="
