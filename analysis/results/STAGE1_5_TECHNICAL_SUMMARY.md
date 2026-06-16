# Stage 1.5 Technical Summary: Rifampicin Co-Crystal Contact Distance

**Warning:** This step is **NOT** molecular docking. Despite the script name (`04d_docking_features.py`) and mentions of AutoDock Vina in docstrings, no docking software is ever called. The feature is simply the Euclidean distance from each rpoB residue to the rifampicin molecule in the 5UHB cryo-EM structure.

---

## Algorithm (what the code actually does)

### Inputs
- **5UHB cryo-EM structure** (`data/pdb/crystal/5UHB_rpoB.pdb`): M. tuberculosis RNA polymerase holoenzyme bound to rifampicin (RFP). Chain C = rpoB subunit.
- **AlphaFold rpoB structure** (`data/pdb/alphafold/rpoB_P9WGY9_alphafold.pdb`): Full-length M. tuberculosis rpoB predicted by AlphaFold2.
- **Residue-level feature table** (`analysis/results/hotspot_model/residue_hotspot_data.csv`): 6,566 residues across 13 genes with sequence features + Stage 1 structural features.

### Step-by-step

#### 1. Extract RFP coordinates (lines 178-195)

Load 5UHB via Biopython PDBParser. Iterate all chains, find residues whose ID starts with `H_` (heteroatom) and resname is `RFP`. Collect all atom coordinates as a numpy array `(N_atoms, 3)`.

#### 2. Greedy sequence-based CA alignment (lines 206-256)

Extract C-alpha atoms from:
- 5UHB chain C (rpoB subunit, may be truncated)
- AlphaFold rpoB (full length)

Walk through 5UHB CA atoms in order; for each, find the next matching amino acid in the AlphaFold structure within a ±50-residue window:

```python
af_start = 0
for cry_resid, cry_aa, ... in cry_items:
    for af_idx in range(af_start, min(af_start + 50, len(af_items))):
        if af_aa == cry_aa:
            # Match! Record CA pair, advance af_start
```

This produces `N_aligned` paired CA atoms. Apply Biopython Superimposer to compute the rigid-body rotation/translation that minimizes RMSD of the paired CAs, then transform ALL AlphaFold atoms into the 5UHB reference frame.

#### 3. Compute per-residue min distance to RFP (lines 260-274)

For every AlphaFold residue (after transformation):
- Collect all atom coordinates `(M_atoms, 3)`
- Compute pairwise Euclidean distance matrix `(M_atoms × N_RFP_atoms)`
- Take the minimum over all atom pairs → `drug_distance` for that residue

#### 4. Map to H37Rv genome positions (lines 276-283)

`compute_position_mapping("rpoB")` aligns the AlphaFold PDB sequence to the H37Rv genome sequence (extracted from the feature table). Uses Biopython `pairwise2.globalms` with scoring (match=2, mismatch=-1, gap_open=-2, gap_extend=-1). Returns a dict: `{H37Rv_position: PDB_residue_number}`.

Then: for each genome position, look up the PDB resid, find its computed distance, store as `("rpoB", genome_pos) → min_dist`.

#### 5. Build the drug_distance feature (lines 298-344)

- Initialize `drug_distance = NaN` for all 6,566 residues
- For rpoB: assign the computed RFP distance for each position
- For all other genes (12/13): fill with `max(rpoB_distances) + 10 = 92.0 Å`
- Also create `drug_contact = (drug_distance <= 5.0 Å).astype(int)`

#### 6. Retrain and evaluate (lines 351-497)

Add `drug_distance` and `drug_contact` to the Stage 1 feature set (base features + sasa_relative + esm2_intolerance + contact_density_3d). Retrain logistic regression with 5-fold StratifiedKFold cross-validation. Compare AUROC with and without docking features.

---

## Results as currently computed

| Metric | Stage 1 | Stage 1 + docking |
|--------|---------|-------------------|
| AUROC | 0.910 | 0.938 |
| Known hotspots in Top 20 | 17/21 | 17/21 |

Four "missed" hotspots checked for rescue:

| Residue | Gene | Drug dist | Rank (with docking) | Rescued? |
|---------|------|-----------|---------------------|----------|
| V170 | rpoB | 4.0 Å | 59 | NO |
| I491 | rpoB | 3.3 Å | 40 | NO |
| V125 | pncA | **92.0 Å** (default) | 168 | NO |
| N538 | gyrB | **92.0 Å** (default) | 161 | NO |

---

## Known issues with the current implementation

### 1. Not docking — it's co-crystal distance

The script is named "docking," defines DRUG_SMILES for 7 drugs, and configures a VINA_PATH — but **AutoDock Vina is never invoked**. The actual computation is measuring distances from an existing co-crystal. This is a static structure analysis, not a docking simulation.

To recreate what the code actually does: you need a co-crystal structure with the drug bound. The "drug_distance" feature is simply the minimum Euclidean distance from any atom in the residue to any atom in the ligand.

### 2. Only rpoB gets meaningful data

The feature is computed for rpoB only (the sole co-crystal structure in the pipeline). All 12 other genes receive `drug_distance = 92.0 Å` — a constant. The model trivially learns that:
- rpoB has small drug distances → more likely to be hotspots (7/21 hotspots are in rpoB)
- Non-rpoB has large drug distances → less likely to be hotspots (diluted over 12 genes)

The AUROC improvement (0.910 → 0.938) primarily reflects this rpoB-vs-non-rpoB prior, not genuine drug-binding information for each target.

### 3. pncA V125 and gyrB N538 were never testable

These residues are NOT in rpoB. They receive the default 92.0 Å distance. The claim "drug proximity is necessary but insufficient" does NOT apply to them — they were missed by Stage 1 structural features and the docking feature provides zero information for them. Their failure is a Stage 1 limitation (allostery, dynamics, compensatory evolution), not a docking limitation.

### 4. D435 rank may have decreased

rpoB D435 was rank #20 in Stage 1. With docking features, it's rank #27. This is unexpected — if drug proximity is informative, a residue 3.0 Å from RFP should rank higher, not lower.

### 5. The greedy alignment is fragile

The CA atom matching (lines 234-241) finds corresponding residues by sliding through the AlphaFold sequence looking for identical amino acids. This works if sequences are identical but fails on:
- Truncated constructs (5UHB may lack terminal regions)
- Strain-specific substitutions
- Insertions/deletions

The ±50-residue window prevents catastrophic desync but creates edge cases near termini.

### 6. No per-gene docking was performed

DRUG_SMILES exist for isoniazid, ethambutol, moxifloxacin, pyrazinamide, and streptomycin, but none were docked. Per-gene docking would require:
- Preparing each protein (protonation state, energy minimization)
- Preparing each ligand (tautomers, conformers, protonation at pH 7.4)
- Running AutoDock Vina with appropriate grid box (centered on known binding site)
- Scoring top poses and computing per-residue contact frequencies

---

## How to recreate correctly

### Option A: Co-crystal distance (current approach, but fix bugs)

```
1. Download 5UHB from RCSB PDB (https://files.rcsb.org/download/5UHB.pdb)
2. Extract rpoB chain + RFP ligand into 5UHB_rpoB.pdb
3. Download AlphaFold rpoB from EBI (AF-P9WGY9-F1-model_v6.pdb)
4. Structural alignment:
   a. Extract CA atoms from 5UHB chain C
   b. Extract CA atoms from AlphaFold
   c. Run global sequence alignment (Biopython pairwise2) 
      — NOT greedy per-residue matching
   d. Use aligned CA pairs for Superimposer
5. Transform AlphaFold into 5UHB frame
6. For each residue: min Euclidean distance to any RFP atom
7. Map PDB residue numbers → H37Rv genome positions
8. Feature vector: [drug_distance, drug_contact_binary]
9. Retrain logistic regression with 5-fold CV
```

### Option B: True molecular docking (what the pipeline should have done)

```
For each (drug, target) pair:
  1. Download/clean target structure (AlphaFold or crystal)
  2. Prepare protein: add hydrogens, assign charges, energy minimize
  3. Prepare ligand: enumerate tautomers, protonate at pH 7.4
  4. Define grid box centered on known/putative binding site
  5. Run AutoDock Vina: exhaustiveness=8, num_modes=20
  6. Cluster poses, select top-ranked cluster
  7. For each residue: compute centroid distance to ligand
  8. Feature: [min_docking_distance, docking_contact_count]

Drug-target pairs to run:
  - Rifampicin → rpoB (5UHB control, already have co-crystal)
  - Isoniazid → katG (prodrug activator, tricky)
  - Ethambutol → embB (emb arabinosyltransferase)
  - Moxifloxacin → gyrA, gyrB (DNA gyrase)
  - Pyrazinamide → pncA (prodrug activator, tricky)
  - Streptomycin → rpsL (ribosomal protein S12)
```

### Option C: The honest minimum for the paper

Rename the step to "Co-crystal Contact Distance Analysis" and report it honestly:
- Only rpoB receives the feature
- AUROC improvement is specific to rpoB-discrimination
- The 4 non-rescued residues: 2 are genuinely in the binding pocket (V170, I491) and correctly ranked low due to fitness/accessibility; 2 are in non-target genes and were never going to be rescued by a rifampicin feature
