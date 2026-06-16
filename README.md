# Forecasting Emerging Tuberculosis Drug Resistance

**Structural hotspot prediction → mutation forecasting → prospective clinical validation**
Aly Dhedhi, Pranava Kumar, Vinay Singamsetty, Li-Lun Ho, Manolis Kellis
Kellis Lab, MIT

---

## Overview

Can we predict which TB resistance mutations will emerge *before* they appear in resistance catalogs?

This project builds a multi-stage framework that:

1. **Identifies structural signatures of known resistance hotspots** across 13 M. tuberculosis resistance genes
2. **Forecasts specific SNV-accessible mutations** most likely to emerge under drug pressure
3. **Validates predictions prospectively** in 12,287 independent CRyPTIC clinical isolates

### Key Results

| Metric | Circular (old) | LR (corrected) | **XGBoost (final)** |
|--------|:--------------:|:--------------:|:-------------------:|
| AUROC | 0.990 | 0.869 | **0.943** |
| AUPRC | 0.523 | 0.206 | **0.249** (59× random) |
| F1 | 0.552 | 0.286 | **0.339** |
| Recall | 0.547 | 0.185 | **0.370** (10/27 hotspots) |
| Top-20 known mutations | ~5/22 | 1/33 | **6/33** |
| CRyPTIC FDR-significant | 25 | 19 | **19** |

**Prospective validation in CRyPTIC (12,287 isolates):**

| Tier | Count | Description |
|------|:-----:|-------------|
| 0 | 30 | Known WHO mutations (pipeline sanity check) |
| 1 | **19** | FDR-significant novel predictions (q < 0.05) |
| 2 | 33 | Observed with resistance enrichment (low power) |
| 3 | 29 | Observed, no phenotype data (pncA/rpsL blind spots) |
| 4 | 187 | Forecast-only (prospective surveillance targets) |

### Critical Bug Fix: Circular drug_proximity

The original model had an inflated AUROC (0.990) because:

1. **Pocket-overlap circularity**: Training positives were inside the pocket definition — a hotspot residue had distance 0 to "the pocket" (itself), giving drug_proximity=1.0 for 24/27 positives
2. **Fix 1 — 3D pocket dilation**: For narrow-pocket genes, dilated pockets to include all residues within 10Å in the AlphaFold structure (katG: 1→20, embB: 3→82, gyrB: 1→14, rpsL: 2→19)
3. **Fix 2 — Self-exclusion**: Query residue excluded from pocket when computing min heavy-atom distance

After the fix, drug_proximity dropped from the dominant feature (LR coefficient +6.71) to 4th importance (0.075). Homoplasy-based features now dominate.

---

## Data Sources & Sample Sizes

### Training data

| Source | Description | Location |
|--------|-------------|----------|
| TBDB/GenTB genomes | ~100 M. tuberculosis genomes | `data/metadata/all_tb_metadata.csv` |
| Resistance genes | 13 protein-coding genes (rpoB, katG, embB, gyrA, gyrB, rpsL, pncA, inhA, eis, tap, mmpR5, mmpL5, tlyA) | ~6,338 residues total |
| Known hotspots | 27 positives (21 original WHO + 6 inhA) | 0.43% positive rate |
| Training negatives | ~6,300 non-hotspot residues | across 12/13 genes (mmpL5 dropped) |

### CRyPTIC validation

| Metric | Count |
|--------|:-----:|
| Total clinical isolates | **12,287** |
| RIF phenotyped | 12,285 |
| INH phenotyped | 12,286 |
| EMB phenotyped | 12,287 |
| MXF phenotyped | 12,287 |
| Drugs with binary phenotypes | 13 (AMI, BDQ, CFZ, DLM, EMB, ETH, INH, KAN, LEV, LZD, MXF, RIF, RFB) |
| Mutation table | 1.5 GB (`data/cryptic/MUTATIONS.csv.gz`) |
| Phenotypes | `data/metadata/cryptic_phenotypes.csv` |

### Mycobacterial genomes (for MSA/Shannon entropy)

| Source | Count | Location |
|--------|:-----:|----------|
| NCBI RefSeq genomes | 10 | `data/genomes/GCF_*.fasta` (~4.4 MB each) |

### Structural data

| Source | Description | Location |
|--------|-------------|----------|
| AlphaFold2 (EBI) | Predicted structures for all 13 resistance proteins | `data/pdb/` |
| 5UHB (PDB) | Cryo-EM RNA polymerase + rifampicin co-crystal | rpoB drug pocket (1.53Å RMSD) |
| 5BS8 (PDB) | Gyrase A + fluoroquinolone co-crystal | gyrA drug pocket (1.59Å RMSD) |

### Other data files

| File | Size | Location |
|------|:----:|----------|
| H37Rv reference genome | ~4.4 MB | `data/reference/` |
| CRyPTIC supplement (tar.gz) | 16 MB | `data/cryptic_supplement.tar.gz` |
| PDB archives (tar.gz) | 1.7 MB | `data/pdbs.tar.gz` |

---

## Pipeline: Stage by Stage

### Stage 0: Sequence-Based Hotspot Prediction

`04b_hotspot_model.py` — Logistic regression using 12 sequence-derived features across ~6,300 residues in 12 resistance genes.

- **AUROC: 0.888**
- **7/21 known hotspots in Top 20 (original set)**
- Features: strand propensity, helix propensity, hydrophobicity, charge, volume, hbond, conservation (BLOSUM62), inner/outer distance, substitution counts

### Stage 1: Structural Features

`04c_stage1_features.py` — Adds three structural biology features from AlphaFold structures:

1. **SASA** — relative solvent accessibility
2. **ESM-2 intolerance** — evolutionary sequence model substitution tolerance
3. **3D contact density** — spatially neighboring residues within 8Å

- **AUROC: 0.910**

### Stage 3: XGBoost + Drug Proximity + pLDDT + Mutation Sensitivity

`04d_docking_features.py` — Replaced LogisticRegression with **XGBoost** (`scale_pos_weight=10, max_depth=6, learning_rate=0.05, n_estimators=300`). Drug proximity computed via dilated pocket-distance proxy (10Å dilation) with query-residue self-exclusion.

- **AUROC: 0.943** (5-fold CV)
- **XGBoost feature importance**: homoplasy_count (0.221), homoplasy_alleles (0.190), inner_distance (0.109), drug_proximity (0.075), plddt_score (0.074), sasa_relative (0.073), plddt_environment (0.066), volume (0.059), strand_propensity (0.036), contact_density_3d (0.033), conservation_blosum (0.024), mutation_sensitivity (0.000)
- **18 features total**: 12 base sequence, 5 structural (SASA, ESM-2, 3D contact, pLDDT score, pLDDT environment), 1 drug_proximity (saturating transform)

Note: Feature mutation_sensitivity was added but showed zero importance and was removed from the final model. ESM-2 intolerance was also non-contributory and was excluded.

### Mutation Forecasting

`04e_mutation_forecasting.py` — For each of the top hotspot residues, enumerate all SNV-accessible mutations. Score by:

```
P(emergence) = P(hotspot | features) × fitness × accessibility
```

- Known resistance mutations in top 20: **6/33**
- Known resistance mutations in top 50: **14/33**
- Known resistance mutations in top 100: **20/33**
- Watchlist: **329 candidate mutations** (`analysis/results/forecasting/emergence_watchlist.csv`)

### Leave-One-Gene-Out Validation

`05_leave_one_gene_out.py` — Uses LogisticRegression (not XGBoost, which overfits to training genes when an entire gene is held out).

| Gene | Known | Top-20 | Top-50 | Top-100 |
|------|:-----:|:------:|:------:|:-------:|
| embB | 9 | 2 | 4 | 4 |
| gyrA | 5 | 1 | 3 | 4 |
| pncA | 4 | 2 | 2 | 2 |
| rpoB | 10 | 1 | 5 | 7 |
| katG | 3 | 0 | 1 | 1 |
| rpsL | 2 | 0 | 2 | 2 |
| gyrB | 1 | — | — | — |

**Aggregate: ~50% Top-50 recall for unseen genes.** LR generalizes better than XGBoost when entire genes are held out.

### CRyPTIC Validation

`08_cryptic_validation_full.py` — Cross-references 329 watchlist mutations against 1.4 GB mutation matrix from 12,287 clinical isolates.

- **87 novel watchlist mutations observed** in clinical isolates
- **19 FDR-significant** (Benjamini-Hochberg q < 0.05)
- **52 novel mutations with phenotype data**, 40/52 (77%) enriched in resistant isolates

### Tiered Categorization

`09_stress_tests.py` — FDR correction, leakage analysis, literature cross-reference, tier assignment.

### Feature Ablation

`scripts/04f_evolutionary_features.py` — Attempted to compute Shannon entropy from MSA of 10 mycobacterial genomes, but external APIs blocked (UniProt BLAST returns 404) and no local alignment tools (MAFFT, BLAST+) are installed. Mutation_sensitivity (BLOSUM-based approximation) showed zero importance.

---

## Strongest Tier 1 Validated Predictions

| Mutation | Gene | Rank | Carriers | R% | OR | FDR p |
|----------|------|:----:|:--------:|:--:|:--:|:-----:|
| Q445R | embB | 37 | 56 | 100% | inf | 2.5e-36 |
| D94A | gyrA | 93 | 147 | 59% | 9.2 | 5.4e-36 |
| G406S | embB | 6 | 99 | 75% | 11.1 | 8.3e-20 |
| Q497K | embB | 121 | 71 | 84% | 20.2 | 1.2e-20 |
| D435G | rpoB | 41 | 61 | 90% | 14.7 | 3.1e-16 |
| H445R | rpoB | 31 | 33 | 97% | 49.4 | 1.2e-11 |
| I491L | rpoB | 220 | 20 | 100% | inf | 1.8e-8 |
| S315G | katG | 4 | 4 | 75% | — | (low n) |

---

## Project Structure

```
tb-resistance-discovery/
├── scripts/                     # Analysis pipeline (17 scripts)
│   ├── 01_download_data.py      # Data acquisition
│   ├── 04_resistance_forecasting.py  # Pipeline orchestration
│   ├── 04b_hotspot_model.py     # Stage 0: sequence-only model
│   ├── 04c_stage1_features.py   # Stage 1: structural features
│   ├── 04d_docking_features.py  # XGBoost model + drug proximity (dilated pocket fix)
│   ├── 04e_mutation_forecasting.py # P(emergence) pipeline
│   ├── 04f_evolutionary_features.py # MSA/Shannon entropy (prototype, blocked)
│   ├── 05_leave_one_gene_out.py # Cross-gene generalization
│   ├── 06_failure_analysis.py   # Missed mutation case studies
│   ├── 08_cryptic_validation_full.py # CRyPTIC mutation matrix cross-reference
│   ├── 09_stress_tests.py       # FDR correction, tiering
│   ├── 10_generate_figures.py   # Figure data generation
│   ├── 11_render_figures.py     # Publication-quality PNG rendering
│   └── 12_audit.py              # Self-audit (157 checks)
│
├── analysis/results/
│   ├── hotspot_model/           # Model outputs
│   │   ├── ranked_predictions.csv           # 6,338 residues
│   │   ├── feature_importance.csv           # XGBoost importance
│   │   ├── alphafold_validation.json
│   │   └── cross_validation_results.csv     # 5-fold CV
│   ├── forecasting/             # Mutation-level outputs
│   │   ├── emergence_watchlist.csv           # 329 candidates
│   │   ├── cryptic_validation_results.csv    # Full CRyPTIC cross-reference
│   │   ├── cryptic_tiered_validation.csv     # Tier 1-4
│   │   ├── cryptic_fdr_analysis.csv          # FDR enrichment
│   │   ├── leave_one_gene_out_results.csv    # LOO validation
│   │   └── failure_analysis.json             # 5 case studies
│   └── figures/                 # Publication figures (6 main + 3 supp)
│       ├── Figure_1.png through Figure_6.png
│       └── Figure_S1.png through Figure_S3.png
│
├── data/
│   ├── cryptic/                 # CRyPTIC tables (MUTATIONS.csv.gz: 1.5 GB)
│   ├── genomes/                 # 10 mycobacterial FASTA files (~4.4 MB each)
│   ├── metadata/                # Phenotypes, assembly metadata
│   │   └── cryptic_phenotypes.csv  # 12,287 samples × 43 columns
│   ├── pdb/                     # AlphaFold + co-crystal PDB structures
│   └── demo/                    # Demo VCF (117 samples)
│
└── requirements.txt             # pandas, numpy, scipy, sklearn, xgboost, biopython, matplotlib, seaborn
```

---

## Figures (9 total: 6 main + 3 supplementary)

| Figure | Description |
|--------|-------------|
| Figure 1 | Pipeline overview + sample statistics |
| Figure 2 | Stage-by-stage AUROC improvement, drug proximity per-gene, failure case studies |
| Figure 3 | XGBoost feature importance (SHAP/permutation) |
| Figure 4 | Emergence watchlist: mutation-level scores and status breakdown |
| Figure 5 | CRyPTIC validation cascade: FDR significance and tier distribution |
| Figure 6 | Clinical impact: carrier counts by tier and enrichment |
| Figure S1 | ROC curves: Stage 0, 1, 3 comparison |
| Figure S2 | Leave-one-gene-out cross-validation results |
| Figure S3 | Feature correlation matrix |

---

## Requirements

Python 3.12+ with standard scientific Python stack:
- `pandas`, `numpy`, `scipy`
- `scikit-learn`, `xgboost`
- `biopython`, `requests`
- `matplotlib`, `seaborn` (figure plotting)

Full environment: see `environment.yml` or `requirements.txt`.

---

## Reproducibility: Getting the Data

### 1. Extract committed archives

```bash
python extract_data.py
```

### 2. Download CRyPTIC MUTATIONS table (1.5 GB)

```bash
python scripts/download_cryptic_data.py
```

Alternatively, download manually from [Zenodo v1.1.1](https://zenodo.org/records/15679731) and place at `data/cryptic/MUTATIONS.csv.gz`.

### 3. Run the pipeline

Scripts are numbered in dependency order:

```bash
python scripts/01_download_data.py
python scripts/04_resistance_forecasting.py
python scripts/04b_hotspot_model.py
python scripts/04c_stage1_features.py
python scripts/04d_docking_features.py
python scripts/04e_mutation_forecasting.py
python scripts/05_leave_one_gene_out.py
python scripts/06_failure_analysis.py
python scripts/08_cryptic_validation_full.py
python scripts/09_stress_tests.py
python scripts/10_generate_figures.py
python scripts/11_render_figures.py
python scripts/12_audit.py        # 157 self-checks
```

---

## Self-Audit

`scripts/12_audit.py` runs 157 automated checks covering:
- Directory/file existence (all scripts, all output files)
- Syntax validation (all 17 scripts)
- Pipeline dependency ordering
- CRyPTIC data integrity (12,287 samples, 13 drugs, mutation count, FDR results)
- Model output schema (ranked predictions, feature importance, watchlist)
- Figure completeness (6 main + 3 supplementary)
- Package availability
- Claim consistency (hotspot count, FDR count, tier counts, AUROC progression)

**Current status: 157/157 pass, 0 failures.**

---

## Limitations & Caveats

1. **Small positive set.** Only 27 known hotspot residues across 13 genes (33 known mutations). Training on ~6,300 residues with 27 positives limits statistical power.

2. **Modest FDR yield.** 19 of ~300 testable predictions (6.3%) survive Benjamini-Hochberg correction at q < 0.05 — modest but above the 5% expected under the global null.

3. **Phenotype blind spots.** pncA mutations (e.g., Q10R, rank #8, 155 carriers) cannot be evaluated because CRyPTIC lacks pyrazinamide binary phenotypes. 29 Tier 3 mutations fall into this blind spot.

4. **No directly comparable benchmark.** No existing model predicts *emergence* of unseen resistance mutations. Closest: WHO catalog (retrospective curation), CRyPTIC ML (genotype→phenotype), ESM-1v saturation (mutational tolerance, not emergence under selection).

5. **External API blocked.** UniProt BLAST and NCBI Entrez unavailable — prevents MSA-based Shannon entropy computation. 10 mycobacterial genomes available in `data/genomes/` if alignment tools (MAFFT, BLAST+) are installed locally.

---

## Reference

Walker, T. M., et al. (2021). "The 2021 WHO catalogue of Mycobacterium tuberculosis complex mutations associated with drug resistance." *The Lancet Microbe*.

The CRyPTIC Consortium (2022). "A data compendium of Mycobacterium tuberculosis antibiotic resistance." *bioRxiv*.
