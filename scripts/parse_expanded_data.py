#!/usr/bin/env python3
"""
Parse Downloaded Data and Compile Expanded Dwarf Galaxy Dataset
================================================================

This script parses the downloaded astronomical data files and compiles
an expanded dataset with environment classification for SDCG analysis.

Target: 300+ galaxies with void/cluster/field classification
"""

import os
import sys
import json
import numpy as np
import xml.etree.ElementTree as ET
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Project paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
DWARFS_DIR = os.path.join(DATA_DIR, "dwarfs")

# ============================================================================
# KNOWN STRUCTURES FOR ENVIRONMENT CLASSIFICATION
# ============================================================================

# Major galaxy clusters with positions and virial radii
CLUSTERS = {
    'Virgo':   {'ra': 187.70, 'dec': 12.39, 'dist': 16.5, 'r_vir': 2.2, 'r_deg': 8.0},
    'Fornax':  {'ra': 54.62, 'dec': -35.45, 'dist': 19.0, 'r_vir': 0.7, 'r_deg': 4.0},
    'Coma':    {'ra': 194.95, 'dec': 27.98, 'dist': 100.0, 'r_vir': 2.9, 'r_deg': 2.5},
    'Perseus': {'ra': 49.95, 'dec': 41.51, 'dist': 72.0, 'r_vir': 2.0, 'r_deg': 2.0},
    'Centaurus': {'ra': 192.2, 'dec': -41.31, 'dist': 52.0, 'r_vir': 1.5, 'r_deg': 2.0},
    'Hydra':   {'ra': 159.17, 'dec': -27.53, 'dist': 54.0, 'r_vir': 1.2, 'r_deg': 1.5},
}

# Major cosmic voids
VOIDS = {
    'Local_Void':     {'ra': 295, 'dec': 5, 'd_min': 1, 'd_max': 25, 'r_deg': 40},
    'Lynx_Cancer':    {'ra': 130, 'dec': 40, 'd_min': 10, 'd_max': 35, 'r_deg': 25},
    'Eridanus':       {'ra': 55, 'dec': -35, 'd_min': 10, 'd_max': 30, 'r_deg': 20},
    'CVn_Void':       {'ra': 190, 'dec': 35, 'd_min': 3, 'd_max': 15, 'r_deg': 15},
    'Bootes_Void':    {'ra': 218, 'dec': 46, 'd_min': 200, 'd_max': 350, 'r_deg': 10},
    'Sculptor_Void':  {'ra': 15, 'dec': -30, 'd_min': 10, 'd_max': 40, 'r_deg': 15},
    'Microscopium':   {'ra': 315, 'dec': -40, 'd_min': 50, 'd_max': 100, 'r_deg': 15},
}


def angular_separation(ra1, dec1, ra2, dec2):
    """Calculate angular separation in degrees"""
    dra = (ra1 - ra2) * np.cos(np.radians((dec1 + dec2) / 2))
    ddec = dec1 - dec2
    return np.sqrt(dra**2 + ddec**2)


def classify_environment(ra, dec, distance_mpc, v_helio=None):
    """
    Classify galaxy environment based on proximity to known structures.
    
    Returns: 'void', 'cluster', or 'field'
    """
    if distance_mpc is None or np.isnan(distance_mpc):
        return 'field'  # Default if no distance
    
    # Check cluster membership first
    for name, cl in CLUSTERS.items():
        ang_sep = angular_separation(ra, dec, cl['ra'], cl['dec'])
        
        # Check if within cluster radius (angular)
        if ang_sep < cl['r_deg']:
            # Check distance consistency
            if abs(distance_mpc - cl['dist']) < 20:
                return 'cluster'
    
    # Check void membership
    for name, void in VOIDS.items():
        ang_sep = angular_separation(ra, dec, void['ra'], void['dec'])
        
        if ang_sep < void['r_deg']:
            if void['d_min'] < distance_mpc < void['d_max']:
                return 'void'
    
    return 'field'


# ============================================================================
# DATA PARSERS
# ============================================================================

def parse_alfalfa(filepath):
    """Parse ALFALFA CSV catalog"""
    print(f"\nParsing ALFALFA: {filepath}")
    
    galaxies = []
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Find header
        header = None
        for i, line in enumerate(lines):
            if line.startswith('AGCNr'):
                header = line.strip().split(',')
                data_start = i + 1
                break
        
        if header is None:
            print("  Could not find ALFALFA header")
            return galaxies
        
        # Parse columns
        col_idx = {col: i for i, col in enumerate(header)}
        
        for line in lines[data_start:]:
            try:
                parts = line.strip().split(',')
                if len(parts) < 10:
                    continue
                
                agc = parts[col_idx.get('AGCNr', 0)]
                name = parts[col_idx.get('Name', 1)].strip('"')
                ra = float(parts[col_idx.get('RAdeg_HI', 2)])
                dec = float(parts[col_idx.get('Decdeg_HI', 3)])
                v_helio = float(parts[col_idx.get('Vhelio', 6)]) if parts[col_idx.get('Vhelio', 6)] else None
                w50 = float(parts[col_idx.get('W50', 7)]) if parts[col_idx.get('W50', 7)] else None
                dist_str = parts[col_idx.get('Dist', 13)]
                dist = float(dist_str) if dist_str and dist_str.strip('"') else None
                
                # Skip high-velocity or invalid entries
                if v_helio and (v_helio < 0 or v_helio > 20000):
                    continue
                
                # Convert W50 to rotation velocity (approximate)
                # V_rot ≈ W50 / (2 * sin(i)) - assuming i=60° average
                v_rot = None
                if w50 and w50 > 10:
                    v_rot = w50 / (2 * np.sin(np.radians(60)))
                
                # Filter for dwarf candidates (W50 < 200 km/s typically for dwarfs)
                if w50 and w50 > 300:
                    continue
                
                env = classify_environment(ra, dec, dist, v_helio)
                
                galaxies.append({
                    'name': f"AGC{agc}" if not name else name,
                    'ra': ra,
                    'dec': dec,
                    'distance_mpc': dist,
                    'v_helio': v_helio,
                    'w50': w50,
                    'v_rot': v_rot,
                    'environment': env,
                    'source': 'ALFALFA_a40'
                })
                
            except (ValueError, IndexError) as e:
                continue
        
        print(f"  Parsed {len(galaxies)} ALFALFA galaxies")
        
    except Exception as e:
        print(f"  Error parsing ALFALFA: {e}")
    
    return galaxies


def parse_votable(filepath, source_name):
    """Parse VOTable XML format from VizieR"""
    print(f"\nParsing VOTable: {filepath}")
    
    galaxies = []
    
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        # Handle namespace
        ns = {'vo': 'http://www.ivoa.net/xml/VOTable/v1.3'}
        
        # Try with namespace
        tables = root.findall('.//vo:TABLE', ns)
        if not tables:
            # Try without namespace
            tables = root.findall('.//TABLE')
        
        if not tables:
            print(f"  No TABLE elements found in {filepath}")
            return galaxies
        
        for table in tables:
            # Get field definitions
            fields = table.findall('.//vo:FIELD', ns)
            if not fields:
                fields = table.findall('.//FIELD')
            
            field_names = [f.get('name') for f in fields]
            
            # Get data rows
            data = table.find('.//vo:DATA', ns)
            if data is None:
                data = table.find('.//DATA')
            
            if data is None:
                continue
            
            tabledata = data.find('.//vo:TABLEDATA', ns)
            if tabledata is None:
                tabledata = data.find('.//TABLEDATA')
            
            if tabledata is None:
                continue
            
            rows = tabledata.findall('.//vo:TR', ns)
            if not rows:
                rows = tabledata.findall('.//TR')
            
            for row in rows:
                cells = row.findall('.//vo:TD', ns)
                if not cells:
                    cells = row.findall('.//TD')
                
                values = [c.text for c in cells]
                
                if len(values) < 3:
                    continue
                
                # Create field->value mapping
                data_dict = {field_names[i]: values[i] for i in range(min(len(field_names), len(values)))}
                
                # Extract common fields
                name = data_dict.get('Name') or data_dict.get('name') or data_dict.get('ID') or ''
                
                ra = None
                for key in ['RAJ2000', 'RAdeg', 'RA', 'ra', '_RAJ2000']:
                    if key in data_dict and data_dict[key]:
                        try:
                            ra = float(data_dict[key])
                            break
                        except:
                            pass
                
                dec = None
                for key in ['DEJ2000', 'DECdeg', 'DEC', 'dec', '_DEJ2000']:
                    if key in data_dict and data_dict[key]:
                        try:
                            dec = float(data_dict[key])
                            break
                        except:
                            pass
                
                dist = None
                for key in ['Dist', 'Distance', 'Dmpc', 'D']:
                    if key in data_dict and data_dict[key]:
                        try:
                            dist = float(data_dict[key])
                            break
                        except:
                            pass
                
                v_rot = None
                for key in ['Vrot', 'W50', 'Vc', 'Vm']:
                    if key in data_dict and data_dict[key]:
                        try:
                            v_rot = float(data_dict[key])
                            break
                        except:
                            pass
                
                if ra is None or dec is None:
                    continue
                
                env = classify_environment(ra, dec, dist)
                
                galaxies.append({
                    'name': name.strip() if name else f'unnamed_{len(galaxies)}',
                    'ra': ra,
                    'dec': dec,
                    'distance_mpc': dist,
                    'v_rot': v_rot,
                    'environment': env,
                    'source': source_name
                })
        
        print(f"  Parsed {len(galaxies)} galaxies from {source_name}")
        
    except Exception as e:
        print(f"  Error parsing VOTable: {e}")
        import traceback
        traceback.print_exc()
    
    return galaxies


def parse_tsv(filepath, source_name):
    """Parse TSV format from VizieR"""
    print(f"\nParsing TSV: {filepath}")
    
    galaxies = []
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Skip comment lines and find header
        header = None
        data_start = 0
        for i, line in enumerate(lines):
            if line.startswith('#'):
                continue
            if line.strip() and not line.startswith('-'):
                if header is None:
                    header = line.strip().split('\t')
                    data_start = i + 1
                    break
        
        if header is None:
            print("  Could not find TSV header")
            return galaxies
        
        # Find relevant columns
        col_idx = {}
        for i, col in enumerate(header):
            col_lower = col.lower().strip()
            if 'ra' in col_lower or col_lower == 'raj2000':
                col_idx['ra'] = i
            elif 'dec' in col_lower or col_lower == 'dej2000':
                col_idx['dec'] = i
            elif 'dist' in col_lower or col_lower == 'd':
                col_idx['dist'] = i
            elif 'name' in col_lower or 'id' in col_lower or col_lower == 'vgs':
                col_idx['name'] = i
            elif 'w50' in col_lower or 'vrot' in col_lower:
                col_idx['v_rot'] = i
        
        # Parse data rows
        for line in lines[data_start:]:
            if line.startswith('#') or line.startswith('-') or not line.strip():
                continue
            
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            
            try:
                name = parts[col_idx.get('name', 0)].strip() if 'name' in col_idx else f'galaxy_{len(galaxies)}'
                ra = float(parts[col_idx['ra']]) if 'ra' in col_idx and parts[col_idx['ra']].strip() else None
                dec = float(parts[col_idx['dec']]) if 'dec' in col_idx and parts[col_idx['dec']].strip() else None
                dist = float(parts[col_idx['dist']]) if 'dist' in col_idx and col_idx['dist'] < len(parts) and parts[col_idx['dist']].strip() else None
                v_rot = float(parts[col_idx['v_rot']]) if 'v_rot' in col_idx and col_idx['v_rot'] < len(parts) and parts[col_idx['v_rot']].strip() else None
                
                if ra is None or dec is None:
                    continue
                
                env = classify_environment(ra, dec, dist)
                
                galaxies.append({
                    'name': name,
                    'ra': ra,
                    'dec': dec,
                    'distance_mpc': dist,
                    'v_rot': v_rot,
                    'environment': env,
                    'source': source_name
                })
                
            except (ValueError, IndexError):
                continue
        
        print(f"  Parsed {len(galaxies)} galaxies from {source_name}")
        
    except Exception as e:
        print(f"  Error parsing TSV: {e}")
    
    return galaxies


def load_existing_data():
    """Load our existing 72-galaxy dataset"""
    print("\nLoading existing dataset...")
    
    galaxies = []
    
    # Load void dwarfs
    void_file = os.path.join(DWARFS_DIR, 'void_dwarfs.json')
    if os.path.exists(void_file):
        with open(void_file, 'r') as f:
            data = json.load(f)
            if 'data' in data:
                for row in data['data']:
                    galaxies.append({
                        'name': row[0],
                        'ra': row[1],
                        'dec': row[2],
                        'distance_mpc': row[3],
                        'v_rot': row[5] if len(row) > 5 else None,
                        'environment': 'void',
                        'source': 'existing_void'
                    })
        print(f"  Loaded {len([g for g in galaxies if g['source'] == 'existing_void'])} existing void dwarfs")
    
    # Load local group dwarfs
    lg_file = os.path.join(DWARFS_DIR, 'local_group_dwarfs.json')
    if os.path.exists(lg_file):
        with open(lg_file, 'r') as f:
            data = json.load(f)
            if 'data' in data:
                for row in data['data']:
                    dist = row[3] / 1000.0 if row[3] > 100 else row[3]
                    galaxies.append({
                        'name': row[0],
                        'ra': row[1],
                        'dec': row[2],
                        'distance_mpc': dist,
                        'v_rot': None,
                        'environment': 'field',
                        'source': 'existing_local_group'
                    })
        print(f"  Loaded {len([g for g in galaxies if g['source'] == 'existing_local_group'])} existing local group dwarfs")
    
    return galaxies


def deduplicate_galaxies(galaxies, tolerance_deg=0.05):
    """Remove duplicate galaxies based on position"""
    print(f"\nDeduplicating {len(galaxies)} galaxies (tolerance: {tolerance_deg}°)...")
    
    unique = []
    
    for gal in galaxies:
        is_duplicate = False
        
        for existing in unique:
            if gal['ra'] is None or gal['dec'] is None:
                break
            if existing['ra'] is None or existing['dec'] is None:
                continue
                
            sep = angular_separation(gal['ra'], gal['dec'], existing['ra'], existing['dec'])
            
            if sep < tolerance_deg:
                # Keep the one with more data (prefer v_rot)
                if gal.get('v_rot') and not existing.get('v_rot'):
                    # Replace with better entry
                    unique.remove(existing)
                    unique.append(gal)
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique.append(gal)
    
    print(f"  Unique galaxies: {len(unique)}")
    return unique


def filter_dwarfs(galaxies, max_v_rot=150, max_distance=100):
    """Filter to retain only likely dwarf galaxies"""
    print(f"\nFiltering for dwarf candidates...")
    
    dwarfs = []
    
    for gal in galaxies:
        # Skip if v_rot too high (not a dwarf)
        if gal.get('v_rot') and gal['v_rot'] > max_v_rot:
            continue
        
        # Skip if too distant (beyond good resolution)
        if gal.get('distance_mpc') and gal['distance_mpc'] > max_distance:
            continue
        
        dwarfs.append(gal)
    
    print(f"  Dwarf candidates: {len(dwarfs)}")
    return dwarfs


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*70)
    print("    PARSING DOWNLOADED DATA")
    print("    Compiling expanded dwarf galaxy dataset")
    print("="*70)
    
    all_galaxies = []
    
    # 1. Load existing data
    existing = load_existing_data()
    all_galaxies.extend(existing)
    
    # 2. Parse ALFALFA
    alfalfa_file = os.path.join(DATA_DIR, 'alfalfa', 'alfalfa_a40.csv')
    if os.path.exists(alfalfa_file):
        alfalfa = parse_alfalfa(alfalfa_file)
        all_galaxies.extend(alfalfa)
    
    # 3. Parse LITTLE THINGS
    lt_file = os.path.join(DWARFS_DIR, 'little_things.xml')
    if os.path.exists(lt_file):
        lt = parse_votable(lt_file, 'LITTLE_THINGS')
        all_galaxies.extend(lt)
    
    # 4. Parse VGS
    vgs_file = os.path.join(DATA_DIR, 'vgs', 'vgs_optical.xml')
    if os.path.exists(vgs_file):
        vgs = parse_votable(vgs_file, 'VGS')
        all_galaxies.extend(vgs)
    
    vgs_hi_file = os.path.join(DATA_DIR, 'vgs', 'vgs_hi.tsv')
    if os.path.exists(vgs_hi_file):
        vgs_hi = parse_tsv(vgs_hi_file, 'VGS_HI')
        all_galaxies.extend(vgs_hi)
    
    # 5. Parse Local Volume
    lvg_file = os.path.join(DWARFS_DIR, 'local_volume.xml')
    if os.path.exists(lvg_file):
        lvg = parse_votable(lvg_file, 'LVG')
        all_galaxies.extend(lvg)
    
    # 6. Parse Virgo dwarfs
    virgo_file = os.path.join(DATA_DIR, 'virgo', 'ngvs_dwarfs.xml')
    if os.path.exists(virgo_file):
        virgo = parse_votable(virgo_file, 'NGVS_Virgo')
        all_galaxies.extend(virgo)
    
    # 7. Parse Fornax dwarfs
    fornax_file = os.path.join(DATA_DIR, 'fornax', 'fds_dwarfs.tsv')
    if os.path.exists(fornax_file):
        fornax = parse_tsv(fornax_file, 'FDS_Fornax')
        all_galaxies.extend(fornax)
    
    # 8. Parse THINGS
    things_file = os.path.join(DWARFS_DIR, 'things.xml')
    if os.path.exists(things_file):
        things = parse_votable(things_file, 'THINGS')
        all_galaxies.extend(things)
    
    print(f"\n{'='*70}")
    print(f"TOTAL GALAXIES COLLECTED: {len(all_galaxies)}")
    print(f"{'='*70}")
    
    # Deduplicate
    unique = deduplicate_galaxies(all_galaxies)
    
    # Filter for dwarfs
    dwarfs = filter_dwarfs(unique, max_v_rot=150, max_distance=100)
    
    # Count by environment
    void_count = len([g for g in dwarfs if g['environment'] == 'void'])
    cluster_count = len([g for g in dwarfs if g['environment'] == 'cluster'])
    field_count = len([g for g in dwarfs if g['environment'] == 'field'])
    
    # Count by source
    sources = {}
    for g in dwarfs:
        src = g.get('source', 'unknown')
        sources[src] = sources.get(src, 0) + 1
    
    print(f"\n{'='*70}")
    print(f"FINAL DWARF GALAXY DATASET")
    print(f"{'='*70}")
    print(f"\nTotal unique dwarfs: {len(dwarfs)}")
    print(f"\nEnvironment breakdown:")
    print(f"  - Void:    {void_count:5d} ({100*void_count/len(dwarfs):.1f}%)")
    print(f"  - Cluster: {cluster_count:5d} ({100*cluster_count/len(dwarfs):.1f}%)")
    print(f"  - Field:   {field_count:5d} ({100*field_count/len(dwarfs):.1f}%)")
    
    print(f"\nSources:")
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  - {src}: {count}")
    
    # Count with rotation velocities
    with_vrot = len([g for g in dwarfs if g.get('v_rot')])
    print(f"\nWith rotation velocities: {with_vrot}")
    
    # Save expanded dataset
    output = {
        'description': 'Expanded dwarf galaxy dataset for SDCG analysis',
        'date': '2026-02-03',
        'total': len(dwarfs),
        'void_count': void_count,
        'cluster_count': cluster_count,
        'field_count': field_count,
        'sources': sources,
        'galaxies': dwarfs
    }
    
    output_file = os.path.join(DATA_DIR, 'expanded_dwarf_dataset.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {output_file}")
    
    # Create CSV for easy viewing
    csv_file = os.path.join(DATA_DIR, 'expanded_dwarf_dataset.csv')
    with open(csv_file, 'w') as f:
        f.write("name,ra,dec,distance_mpc,v_rot,environment,source\n")
        for g in dwarfs:
            f.write(f"{g['name']},{g['ra']},{g['dec']},{g.get('distance_mpc', '')},{g.get('v_rot', '')},{g['environment']},{g['source']}\n")
    print(f"Saved CSV: {csv_file}")
    
    # Summary for SDCG analysis
    print(f"\n{'='*70}")
    print("READY FOR SDCG ANALYSIS")
    print(f"{'='*70}")
    
    if void_count >= 50 and cluster_count >= 50:
        print("✓ Sufficient void and cluster samples for robust comparison")
    else:
        print(f"⚠ Consider adding more void ({50-void_count} needed) or cluster ({50-cluster_count} needed) galaxies")
    
    if with_vrot >= 100:
        print("✓ Sufficient rotation velocity measurements")
    else:
        print(f"⚠ Need more rotation velocities ({100-with_vrot} needed)")
    
    return dwarfs


if __name__ == '__main__':
    main()
