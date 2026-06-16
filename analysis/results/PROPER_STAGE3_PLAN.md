# How to Properly Redo Stage 3

## Overview of the problem

The current Stage 3 (`04d_docking_features.py`) has a fundamental design flaw: **it computes a drug-contact feature for rpoB only, then fills all other 12 genes with a constant 92.0 Å.** This means:
- The AUROC improvement (0.910 → 0.938) primarily reflects the model learning "is this rpoB?" (7/21 hotspots) — not genuine drug-binding information
- 2 of the 4 "missed" hotspots (pncA V125, gyrB N538) could never be helped by a rifampicin-only feature
- The other 2 (rpoB V170, I491) are genuinely close to rifampicin (4.0 Å, 3.3 Å) and correctly ranked low — that part is sound

A proper Stage 3 requires **per-gene drug-contact features** where every resistance gene gets its own drug-specific distance.

---

## Approach: Hybrid Co-Crystal + Docking

### Principle

Every drug-target pair gets its own `drug_distance_gene` feature. Where a co-crystal structure exists, use it. Where one doesn't, run AutoDock Vina.

### Drug-target pairs

| Gene | Drug | Approach | Expected source |
|------|------|----------|----------------|
| rpoB | Rifampicin | Co-crystal | 5UHB (already done, fix alignment) |
| katG | Isoniazid | **Docking** | No co-crystal with INH; katG is prodrug activator |
| embB | Ethambutol | **Docking** | No M. tb co-crystal with EMB |
| gyrA | Moxifloxacin | Co-crystal | 5BTC (M. tb gyrA with moxifloxacin) |
| gyrB | Moxifloxacin | Homology + Docking | No direct co-crystal; use gyrA-based binding site |
| pncA | Pyrazinamide | **Docking** | PZA is substrate; active site is known |
| rpsL | Streptomycin | Co-crystal | Ribosomal structures with streptomycin (e.g., 4KIQ from E. coli; need M. tb homolog) |
| eis | Amikacin | Docking | Aminoglycoside acetyltransferase; co-crystals available |
| tap | Amikacin | Docking | Efflux pump; limited structural data |
| mmpR5 | Bedaquiline | Docking | Some cryo-EM structures available |
| mmpL5 | Bedaquiline | Docking | Some cryo-EM structures available |
| tlyA | Capreomycin | Docking | rRNA methyltransferase |
| inhA | Isoniazid | **Docking** | INH-NAD adduct inhibits InhA; co-crystal 4BII (M. tb InhA + NADH) |

---

## Detailed Protocol

### Part 1: Structural Preparation

For each gene, prepare the protein structure:

```python
def prepare_protein(pdb_path, output_pdbqt):
    """
    Step 1: Clean the PDB structure for AutoDock Vina.

    Operations:
    1. Remove water molecules (HOH)
    2. Remove heteroatoms (unless they're catalytic ions like heme Fe)
    3. Add polar hydrogens (pH 7.4)
    4. Assign Gasteiger charges
    5. Merge non-polar hydrogens
    6. Output as PDBQT (AutoDock format)

    Tools:
    - obabel (OpenBabel): obabel input.pdb -o pdbqt -xr -xh --gen3d -p 7.4
    - Or meeko / AutoDockTools prepare_receptor4.py
    - Or ADFRsuite's prepare_receptor
    """
```

**For each gene:**
| Gene | PDB source | Special handling |
|------|-----------|-----------------|
| rpoB | AlphaFold (P9WGY9) or 5UHB chain C | Use 5UHB directly (cryo-EM, already has RFP bound) |
| katG | AlphaFold (P9WIE5) or 2CAS | 2CAS is crystal (2.0 Å); heme group must be kept |
| embB | AlphaFold (P9WNL7) | Large TMD protein; C-terminal domain |
| gyrA | 5BTC chain A | Already has MOX bound; use crystal directly |
| gyrB | AlphaFold (P9WG45) | Use gyrA binding site to position grid |
| pncA | AlphaFold (I6XD65) | Small protein, active site is known |
| rpsL | AlphaFold (P9WH63) | Small ribosomal protein |
| others | AlphaFold | Standard preparation |

### Part 2: Ligand Preparation

```python
def prepare_ligand(smiles, output_pdbqt, pH=7.4):
    """
    Step 2: Prepare ligand for AutoDock Vina.

    Operations:
    1. Generate 3D conformation from SMILES
    2. Enumerate tautomers at pH 7.4
    3. Assign Gasteiger charges
    4. Detect rotatable bonds
    5. Output as PDBQT

    Tools:
    - rdkit: Chem.MolFromSmiles -> EmbedMultipleConfs -> AddHs
    - meeko: smiles_to_pdbqt (handles protonation, tautomers)
    - obabel: obabel smiles -o pdbqt --gen3d -p 7.4
    """
```

**Ligands from SMILES:**
```python
DRUG_SMILES = {
    "rifampicin": "CC1=C(C(=O)C2=C(C3=C(C(=C(C=C3O)C(=O)NCC4=CC=CC=C4)O)C(=C2O1)C)O)OC5C(C(C(C(O5)C)O)N)O",
    "isoniazid": "C1=CC(=CN=C1)C(=O)NN",
    "ethambutol": "CCN(CC)C(CO)C(CO)NCC",
    "moxifloxacin": "CC1COC2=C(C(=CC(=C2C1)N3C=C(C(=O)C3=O)C(=O)O)F)N4CCNCC4",
    "pyrazinamide": "C1=CN=C(C=N1)C(=O)N",
    "streptomycin": "C1C(C(C(C(O1)OC2C(C(C(C(O2)CN)O)O)O)N)N)O",
    "amikacin": ...,
    "bedaquiline": ...,
    "capreomycin": ...,
}
```

### Part 3: Docking Grid Definition

For each protein, define the search box:

```python
def define_grid_box(protein_pdbqt, known_residues=None, center=None, size=25.0):
    """
    Define the AutoDock Vina search space.

    Two strategies:
    A) Known binding site residues: center = mean(CA_coords of known residues)
    B) Co-crystal ligand: center = centroid of bound ligand

    Grid size:
    - Default: 25 Å × 25 Å × 25 Å (covers most binding pockets)
    - Adjust: 2× the ligand's radius of gyration + 5 Å buffer

    Output: config.txt for Vina
    """
    return {
        "center_x": center[0], "center_y": center[1], "center_z": center[2],
        "size_x": size, "size_y": size, "size_z": size,
    }
```

**Grid centers per gene:**

| Gene | Center reference | Rationale |
|------|-----------------|-----------|
| rpoB | Centroid of RRDR residues 426-452 | Known rifampicin binding pocket |
| katG | Heme iron (FE) coordinates | Active site; INH activation |
| embB | Centroid of residues 295-420 | EMB-binding transmembrane domain |
| gyrA | D94 CA (or MOX centroid from 5BTC) | Known quinolone binding |
| gyrB | N538 CA | Known FQ resistance position |
| pncA | Centroid of active site (residues 4-20, 67-85) | PZA substrate binding pocket |
| rpsL | K43 CA | Known streptomycin contact |
| inhA | Centroid of NADH binding site | INH-NAD adduct target |

### Part 4: Running AutoDock Vina

```python
import subprocess

def run_vina(receptor_pdbqt, ligand_pdbqt, box_config, output_pdbqt,
             exhaustiveness=32, num_modes=20, energy_range=3):
    """
    Run AutoDock Vina.

    Parameters:
    - exhaustiveness: 32 (default is 8; higher = more thorough)
    - num_modes: 20 (number of binding modes to generate)
    - energy_range: 3 (kcal/mol; max energy diff from best mode)

    Command:
    vina --receptor receptor.pdbqt \
         --ligand ligand.pdbqt \
         --center_x ... --center_y ... --center_z ... \
         --size_x ... --size_y ... --size_z ... \
         --exhaustiveness 32 \
         --num_modes 20 \
         --energy_range 3 \
         --out output.pdbqt

    Output: Multi-model PDBQT file with ranked poses.
    """
    cmd = [
        "vina",
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligand_pdbqt),
        f"--center_x", str(box_config["center_x"]),
        f"--center_y", str(box_config["center_y"]),
        f"--center_z", str(box_config["center_z"]),
        f"--size_x", str(box_config["size_x"]),
        f"--size_y", str(box_config["size_y"]),
        f"--size_z", str(box_config["size_z"]),
        f"--exhaustiveness", str(exhaustiveness),
        f"--num_modes", str(num_modes),
        f"--energy_range", str(energy_range),
        "--out", str(output_pdbqt),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    parse_vina_output(result.stdout)

def parse_vina_output(vina_stdout):
    """Extract binding energies from Vina output."""
    # Lines look like:
    # 1    -9.5   0.000   0.000
    # 2    -9.2   1.523   2.101
    modes = []
    in_table = False
    for line in vina_stdout.split("\n"):
        if "mode |   affinity | dist from best" in line:
            in_table = True
            continue
        if in_table and line.strip() and not line.startswith("---"):
            parts = line.split()
            if len(parts) >= 4:
                mode = int(parts[0])
                affinity = float(parts[1])  # kcal/mol
                rmsd_lb = float(parts[2])
                rmsd_ub = float(parts[3])
                modes.append({
                    "mode": mode, "affinity": affinity,
                    "rmsd_lb": rmsd_lb, "rmsd_ub": rmsd_ub,
                })
    return modes
```

**Run times estimate:** ~5-15 minutes per drug-target pair at exhaustiveness=32 on a modern CPU.

### Part 5: Per-Residue Feature Extraction from Docked Poses

```python
import numpy as np
from Bio.PDB import PDBParser

def compute_per_residue_docking_features(protein_pdb, docked_ligand_pdbqt,
                                         pose_number=1):
    """
    For the top-ranked pose of the docked ligand, compute per-residue metrics.

    Features:
    1. min_distance: Min Euclidean distance from any residue atom to any ligand atom
    2. contact_count: Number of ligand atoms within 4 Å of the residue
    3. is_contact: 1 if any atom pair within 4 Å
    4. closest_ligand_atom_type: Atom type of nearest ligand atom (categorical)

    Note: Vina output is in PDBQT format. The coordinates of each pose
    are separated by "ENDMDL" / "MODEL" records. Use pose_number=1 for
    the top-ranked pose.
    """
    # Parse docked poses
    # ... extract atoms for pose_number ...

    parser = PDBParser(QUIET=True)
    protein = parser.get_structure("protein", protein_pdb)

    results = {}
    for chain in protein[0]:
        for residue in chain:
            if residue.get_id()[0].startswith("H_"):
                continue
            # Get all heavy atom coordinates
            res_coords = np.array([
                a.get_vector().get_array()
                for a in residue.get_atoms()
                if a.element != "H"
            ])
            if len(res_coords) == 0:
                continue

            # Compute distances to all ligand atoms
            # ligand_coords: (N_ligand_atoms, 3)
            distances = np.linalg.norm(
                res_coords[:, None] - ligand_coords[None, :],
                axis=-1
            )
            min_dist = np.min(distances)
            n_contacts = np.sum(distances < 4.0)

            results[(chain.get_id(), residue.get_id()[1])] = {
                "docking_min_distance": round(min_dist, 2),
                "docking_contact_count": n_contacts,
                "docking_is_contact": int(min_dist < 4.0),
            }

    return results
```

### Part 6: Co-Crystal Distances (for rpoB, gyrA, rpsL)

For genes with available co-crystal structures, the procedure is the same as the current code BUT:

1. **Fix the alignment** — replace the greedy per-residue matching with proper Smith-Waterman:
   ```python
   from Bio import pairwise2
   align = pairwise2.align.globalms(
       pdb_seq, af_seq, 2, -1, -2, -1
   )
   # Use the full alignment, not greedy per-residue matching
   ```

2. **Compute per-residue distances** — same min-distance computation but without the constant-fill for non-rpoB genes

3. **Available co-crystals:**
   - rpoB + RFP: 5UHB (cryo-EM, 3.5 Å) — already done
   - gyrA + MOX: 5BTC (X-ray, 2.85 Å) — M. tuberculosis gyrase with moxifloxacin
   - rpsL + streptomycin: Need to find M. tuberculosis or close homolog structure

### Part 7: Feature Integration

Instead of a single `drug_distance` column that's 92.0 for 12/13 genes:

```python
# One feature per gene, but stored as a single column
# Values are NaN for genes without a computed docking distance
df["drug_distance"] = np.nan  # Initialize all NaN

for gene in RESISTANCE_GENES:
    gene_mask = df["gene"] == gene
    if gene in docking_results:
        for pos, dist in docking_results[gene].items():
            mask = gene_mask & (df["residue_pos"] == pos)
            df.loc[mask, "drug_distance"] = dist

# Fill remaining NaN with a large number
# (these are residues far from any drug binding site)
default_dist = df["drug_distance"].max() + 10 if df["drug_distance"].notna().any() else 100
df["drug_distance"] = df["drug_distance"].fillna(default_dist)

# Binary contact feature
df["drug_contact"] = (df["drug_distance"] <= 5.0).astype(int)
```

### Part 8: Validation of Docking Quality

For each drug-target pair with a known co-crystal:
```python
def validate_docking(known_co_crystal_pdb, docked_pose_pdbqt):
    """
    Redock the ligand into the protein and compare to the
    crystallographic pose. Compute RMSD of ligand heavy atoms.

    Thresholds:
    - RMSD < 2.0 Å: excellent (docking succeeded)
    - RMSD < 3.0 Å: acceptable
    - RMSD > 3.0 Å: docking failed; use co-crystal directly
    """
    # Align protein structures
    # Extract ligand coordinates from both
    # Compute heavy-atom RMSD after protein alignment
    return rmsd
```

For targets without co-crystals, validate by:
- Known resistance mutations should be near the docked ligand
- Check docking energy is reasonable (< -6 kcal/mol for good binders)

---

## Expected Impact on Results

### What should improve

| Current problem | With proper Stage 3 |
|----------------|---------------------|
| drug_distance = 92.0 for 12/13 genes | Each gene has a meaningful drug distance |
| AUROC gain is primarily rpoB-discrimination | AUROC gain reflects genuine drug-binding per gene |
| pncA V125 and gyrB N538 "not rescued" due to missing feature | V125 gets pncA-PZA distance; N538 gets gyrB-MOX distance |
| D435 rank decreased (27 vs 20) | Fixed alignment may fix this |

### What should NOT change

- V170 and I491 will still rank low — they're genuinely 4.0 Å and 3.3 Å from RFP but penalized for transversion + fitness cost. This is correct.
- The overall conclusion "drug proximity is necessary but insufficient" still holds for the rpoB cases.

### Hypothetical results

| Metric | Current | With proper Stage 3 (estimate) |
|--------|---------|-------------------------------|
| AUROC | 0.938 | 0.945-0.955 (better per-gene signal) |
| Hotspots in Top 20 | 17/21 | 18-19/21 (V125 may be rescued by PZA docking) |
| Non-rescued | V170, I491, V125, N538 | V170, I491 (same reasons; correct) |

---

## What NOT to do

1. **Don't use the current greedy alignment** — it's fragile. Use proper Smith-Waterman.
2. **Don't fill non-rpoB with a constant** — defeats the purpose of per-gene features.
3. **Don't report AUROC without per-gene breakdown** — the current 0.938 is inflated by the rpoB prior.
4. **Don't claim "docking"** when it's co-crystal distance. If you run Vina, call it docking. If you use co-crystals, call it co-crystal contact distance.
5. **Don't try to dock to pncA with PZA expecting a "binding affinity" interpretation** — PZA is a substrate, not an inhibitor. The docking should find the active site, but the resistance mechanism (loss of enzyme activity) is different from target-based drugs.

---

## Implementation Plan

### What to build

A new script (e.g., `07_proper_stage3.py`) that:

```python
# Phase 1: For each gene, get or compute drug distances
for gene, drug in DRUG_TARGET_PAIRS:
    if gene in CO_CRYSTAL_STRUCTURES:
        distances = co_crystal_distance(gene, co_crystal_pdb)
    else:
        distances = dock_drug_to_protein(gene, drug, alphafold_pdb)

# Phase 2: Build per-gene drug distance feature
df = load_residue_data()
for gene in genes:
    for pos, dist in distances[gene].items():
        df.loc[(df.gene == gene) & (df.residue_pos == pos), "drug_distance"] = dist

# Phase 3: Retrain and evaluate
model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# ... 5-fold CV with and without docking features ...

# Phase 4: Per-gene breakdown
for gene in genes:
    print(f"  {gene}: AUROC without={:.3f}, with docking={:.3f}")
```

### Dependencies to install

```bash
pip install meeko    # ligand preparation (SMILES → PDBQT)
pip install rdkit    # cheminformatics (tautomers, conformers)
# AutoDock Vina binary: https://vina.scripps.edu/download/
# obabel for alternative prep: conda install -c conda-forge openbabel
```

### Time estimate

| Step | Time |
|------|------|
| Co-crystal distance (rpoB fix, gyrA, rpsL) | ~30 min (mostly finding correct structures) |
| Docking preparation (proteins + ligands) | ~1-2 hours (per target, first time) |
| Docking runs (10 targets × 15 min) | ~2-3 hours (can parallelize) |
| Feature extraction + model retraining | ~30 min |
| **Total** | **~5-7 hours** |
