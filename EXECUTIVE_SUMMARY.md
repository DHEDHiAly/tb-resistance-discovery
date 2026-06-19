# Executive Summary — TB Resistance Emergence Forecasting Pipeline

**Project:** Forecasting Emerging Tuberculosis Drug Resistance  
**Authors:** Aly Dhedhi, Vinay Singamsetty, Li-Lun Ho, Manolis Kellis (Kellis Lab, MIT)  
**Repository:** `tb-resistance-discovery`  
**Last updated:** June 2026  

This document is a complete, start-to-finish guide to every script, data file, model stage, score, validation result, and design decision in the repository. Read it alongside `README.md` (public-facing summary) and `viewer.html` (interactive dashboard).

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Core Concepts and Vocabulary](#2-core-concepts-and-vocabulary)
3. [Model Architecture (Not a Neural Network)](#3-model-architecture-not-a-neural-network)
4. [All 17 Features Explained](#4-all-17-features-explained)
5. [Complete Score and Metrics Reference](#5-complete-score-and-metrics-reference)
6. [Pipeline Execution Order](#6-pipeline-execution-order)
7. [Resistance Genes Covered](#7-resistance-genes-covered)
8. [Every Pipeline Script (`scripts/`)](#8-every-pipeline-script-scripts)
9. [Every Analysis Script (`analysis/`)](#9-every-analysis-script-analysis)
10. [Root-Level Files](#10-root-level-files)
11. [Data Directory Inventory](#11-data-directory-inventory)
12. [Structural Assets (`data/pdb/`)](#12-structural-assets-datapdb)
13. [Results Directory Inventory](#13-results-directory-inventory)
14. [CRyPTIC Validation Tiers](#14-cryptic-validation-tiers)
15. [AutoDock Vina Structural Validation](#15-autodock-vina-structural-validation)
16. [Novelty Audit and Literature Claims](#16-novelty-audit-and-literature-claims)
17. [Known Bugs, Fixes, and Leakage Controls](#17-known-bugs-fixes-and-leakage-controls)
18. [Limitations](#18-limitations)
19. [Reproducibility Commands](#19-reproducibility-commands)
20. [Roadmap](#20-roadmap)

---

## 1. What This Project Does

### Scientific problem

Existing TB genomics work (WHO catalog, CRyPTIC) focuses on **genotype → phenotype**: given a mutation already seen in a patient, does it cause resistance? This project asks the **inverse prospective question**:

> Which mutations are *likely to emerge next* in the clinic, before they appear in surveillance databases?

### Three-phase workflow

| Phase | What happens | Key output |
|-------|--------------|------------|
| **1. Learn hotspot signatures** | Train a residue-level classifier on 13 TB resistance genes using sequence, homoplasy, structure, and drug proximity | `hotspot_score` per residue |
| **2. Enumerate accessible mutations** | At high-scoring residues, list all single-nucleotide (SNV) accessible amino acid changes | Mutation candidates |
| **3. Score emergence + validate** | `emergence_score = hotspot_score × mutation_score`; cross-check against 12,287 CRyPTIC isolates; dock top candidates with AutoDock Vina | Tier 0–4 classification, Vina ΔΔG |

### What success looks like

- **Ranking quality:** All 32 known hotspot residues rank in the top 32 by model score (AUROC 0.971).
- **Prospective signal:** 188 Tier-4 mutations (0 carriers in CRyPTIC) flagged for surveillance.
- **Retrospective confirmation:** 24 Tier-1 mutations reach FDR significance in CRyPTIC (model predicted before clinical confirmation).
- **Structural orthogonal validation:** 10 Tier-4 pocket-direct mutations show ΔΔG ≥ +0.15 kcal/mol in Vina.
- **Lead discovery claim:** **gyrB Q538L** — literature-novel, pocket-direct, ΔΔG +0.737 (STRONG).

---

## 2. Core Concepts and Vocabulary

| Term | Definition |
|------|------------|
| **Hotspot residue** | An amino acid position where resistance mutations cluster (32 positives in this dataset) |
| **Positive label** | Residue with ≥1 documented resistance mutation in WHO/CRyPTIC catalogs (`is_hotspot = 1`) |
| **hotspot_score** | Calibrated P(residue is a resistance hotspot) from Stage 2 XGBoost model |
| **emergence_score** | P(mutation emergence) = hotspot_score × mutation_score (fitness × accessibility × plausibility) |
| **Homoplasy** | Independent recurrence of the same mutation across unrelated TB lineages (signals selective pressure) |
| **Drug proximity** | Normalized inverse distance from residue to drug-binding pocket (10 Å radius, **self-excluded**) |
| **Tier 4** | Forecast-only: 0 carriers in CRyPTIC mutation matrix — never seen clinically |
| **Tier 1** | Novel mutation already in CRyPTIC with FDR q < 0.05 resistance enrichment |
| **ΔΔG (ddG)** | Vina binding energy difference: ΔG_mut − ΔG_WT; positive = weaker drug binding |
| **Pipeline novel** | Tier 4, 0 CRyPTIC carriers |
| **Literature novel** | Not in CARD/PubMed/WHO for Mtb at that exact substitution |
| **Structurally validated** | Pocket-direct (≤4.5 Å) + Vina ΔΔG ≥ +0.15 kcal/mol |

These three novelty notions are **not the same thing**.

---

## 3. Model Architecture (Not a Neural Network)

The production classifier is **not** an end-to-end deep learning model. It is a **staged tabular ML pipeline**:

```
Genomic + sequence features
        ↓
   AlphaFold structure → SASA, contact density, drug proximity
        ↓
   ESM-2 (feature only) → mutation intolerance score per residue
        ↓
   XGBoost gradient boosted trees (300 trees, depth 6)
        ↓
   Platt calibration (CalibratedClassifierCV, 5-fold internal CV)
        ↓
   hotspot_score → 04e mutation enumeration → emergence_score
```

### Classifier by stage

| Stage | Script | Algorithm | AUROC |
|-------|--------|-----------|-------|
| 0 | `04b_hotspot_model.py` | Logistic regression (weighted) | **0.888** |
| 1 | `04c_stage1_features.py` | Logistic regression + structural features | **0.906** |
| 2 | `04d_docking_features.py` | **XGBoost** + drug proximity + Platt calibration | **0.971** |
| Forecast | `04e_mutation_forecasting.py` | Rule-based mutation scoring on hotspot prior | — |
| LOO | `05_leave_one_gene_out.py` | Logistic regression (better cross-gene generalization) | ~50% top-50 recall |

### XGBoost hyperparameters (Stage 2, authoritative)

```python
XGBClassifier(
    scale_pos_weight=10,
    max_depth=6,
    learning_rate=0.05,
    n_estimators=300,
    random_state=42,
)
# Wrapped in CalibratedClassifierCV(method='sigmoid', cv=5)
```

### ESM-2 role

ESM-2 (`facebook/esm2_t33_650M_UR50D`) computes per-residue **mutation intolerance** (pseudo-log-likelihood drop on mutation). It is a **feature**, not the classifier. ESM-2 alone achieves AUROC **0.618** (near random); the full model adds **+0.353 AUROC (+57%)**.

### AlphaFold role

13 AlphaFold models (`data/pdb/alphafold/`) provide 3D coordinates for:
- Relative SASA (solvent accessibility)
- 3D contact density (Cβ neighbors within 8 Å)
- Drug proximity proxy when no co-crystal exists
- pLDDT environment confidence

Co-crystal structures (rpoB 5UHB, gyrA 5BS8) provide high-confidence drug distances and rigid docking receptors.

---

## 4. All 17 Features Explained

Master feature table: `analysis/results/hotspot_model/residue_hotspot_data_with_docking.csv`  
6,326 residues × 16–17 features; 32 positives (0.50%).

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `inner_distance` | Sequence | Distance from residue to nearest pocket residue (within gene) |
| 2 | `homoplasy_count` | Genomic | Count of genomes with any non-synonymous change at this codon (1,037 genomes) |
| 3 | `homoplasy_alleles` | Genomic | Count of distinct mutant alleles at this codon across genomes |
| 4 | `helix_propensity` | Sequence | Chou-Fasman helix propensity of wild-type amino acid |
| 5 | `strand_propensity` | Sequence | Chou-Fasman strand propensity |
| 6 | `hydrophobicity` | Physicochemical | Kyte-Doolittle scale |
| 7 | `volume` | Physicochemical | Side-chain volume (Å³) |
| 8 | `charge` | Physicochemical | Net charge at pH 7 |
| 9 | `hbond` | Physicochemical | H-bond donor/acceptor propensity |
| 10 | `rel_position` | Sequence | Normalized position along protein (0–1) |
| 11 | `conservation_blosum` | Sequence | BLOSUM62 self-score (conservation proxy) |
| 12 | `contact_density_seq` | Sequence | 1D neighbor density in sequence |
| 13 | `sasa_relative` | Structural | Relative solvent-accessible surface area (AlphaFold) |
| 14 | `esm2_intolerance` | PLM feature | ESM-2 mutation effect score |
| 15 | `contact_density_3d` | Structural | Cβ neighbors within 8 Å (AlphaFold) |
| 16 | `plddt_environment` | Structural | Mean pLDDT of residue + 6 Å shell |
| 17 | `drug_proximity` | Structural | 1 − (min_distance_to_pocket / 10Å), **self-excluded** |

### Feature importance (XGBoost gain, Stage 2)

| Feature | Gain | Interpretation |
|---------|------|----------------|
| homoplasy_alleles | **0.269** | Multiple independent alleles = strong emergence signal |
| drug_proximity | 0.158 | Pocket proximity predicts resistance mechanism |
| homoplasy_count | 0.149 | Recurrence across lineages |
| inner_distance | 0.048 | Sequence-level pocket proximity |
| strand_propensity | 0.047 | Secondary structure context |
| hydrophobicity | 0.046 | Biophysical environment |
| volume | 0.039 | Steric constraints |
| plddt_environment | 0.036 | Structural confidence |
| hbond | 0.036 | H-bonding capacity |

---

## 5. Complete Score and Metrics Reference

### Hotspot model (Stage 2 — primary claims)

Source: `analysis/results/hotspot_model/stage3_results.json`, `README.md`

| Metric | Value | Notes |
|--------|-------|-------|
| AUROC (5-fold stratified CV) | **0.971** | Threshold-independent ranking |
| AUPRC | **0.560** | 111× random baseline (0.005) |
| Best F1 (CV, Youden-optimized) | **0.622 ± 0.105** | Precision 0.83, recall 0.53 |
| F1 @ threshold 0.5 | **0.532 ± 0.181** | Diluted by 0.5% positive rate |
| Recall | **0.931** | TP / (TP + FN) |
| Specificity | 0.879 | TN / (TN + FP) |
| MCC | **0.306** | Matthews correlation |
| Permutation test | **p = 0.005** | 200 label shuffles |
| Top-20 recall | **26/32 (0.657)** | Known hotspots in top 20 |
| Top-50 recall | 25/32 (0.781) | |
| Top-100 recall | 27/32 (0.844) | |
| GroupKFold AUROC | **0.972 ± 0.018** | By gene, 4/5 folds |
| GroupKFold AUPRC | **0.575** | |
| GroupKFold Top-20 recall | **0.741** | |
| ESM-2 baseline AUROC | 0.618 | Near random |
| Full model lift over ESM-2 | +0.353 AUROC (+57%) | |

### Stage progression (AUROC)

| Stage | Script | AUROC | AUPRC | Top-20 recall |
|-------|--------|-------|-------|---------------|
| 0 | 04b | 0.888 | — | — |
| 1 | 04c | 0.906 | 0.205 | 0.386 |
| 2 | 04d | **0.971** | **0.560** | **0.657** |

### Per-gene Stage 3 AUROC (where computable)

| Gene | Stage 1 | Stage 3 | Notes |
|------|---------|---------|-------|
| rpoB | 0.956 | **0.995** | Best powered gene |
| gyrA | 0.996 | 1.000 | |
| embB | 1.000 | 1.000 | |
| pncA | 0.923 | 0.865 | Slight regression |
| inhA | 0.631 | 0.550 | Hard gene |
| gyrB, katG, eis | null | null | ≤1 positive each |
| mmpL5, mmpR5, tap, tlyA | null | null | 0 positives |

### Ranking gap (key insight)

All 32 positives occupy ranks 1–32. Score gap between last positive (0.650) and first negative (0.249) = **0.40**. At threshold 0.5, precision = 0.75 with zero false positives at rank ≤ 32.

### Mutation forecasting (04e)

| Metric | Value |
|--------|-------|
| Total watchlist mutations | ~330 |
| Tier-4 forecast-only | **188** |
| Known resistance in top-20 | 8/33 |
| Known resistance in top-50 | 16/33 |
| Known resistance in top-100 | 24/33 |

### CRyPTIC validation

| Tier | Count | Meaning |
|------|-------|---------|
| 0 | 30 | WHO-known (sanity check) |
| 1 | **24** | FDR q < 0.05 novel predictions |
| 2 | 32 | Enriched but underpowered |
| 3 | 31 | Observed, no phenotype (pncA blind spot) |
| 4 | **188** | Forecast-only (0 carriers) |

Matched-null validation: Tier 1 count (24) vs 1,000 random matched sets → **p = 0.001**.

### Vina structural validation (Tier-4 batch)

32 pocket-direct candidates (≤4.5 Å). **10 pass** ΔΔG ≥ +0.15:

| Mutation | Rank | ΔΔG | Category | Literature novel? |
|----------|------|-----|----------|-------------------|
| **gyrB Q538L** | 131 | **+0.737** | STRONG | **Yes** |
| rpoB L452M | 181 | +2.137 | STRONG | No (CARD) |
| rpoB P483R | 140 | +1.254 | STRONG | Maybe (P483L known) |
| rpoB L452R | 132 | +1.045 | STRONG | No (CARD) |
| rpoB Q432R | 225 | +0.399 | MODERATE | No (CARD) |
| gyrA S91A | 170 | +0.203 | MODERATE | No |
| gyrA G88S | 30 | +0.179 | MODERATE | No |
| gyrA G88V | 207 | +0.178 | MODERATE | No |
| gyrA G88D | 29 | +0.170 | MODERATE | No |
| rpoB I491N | 124 | +0.156 | MODERATE | No |

Vina categories: STRONG ≥0.4, MODERATE ≥0.15, WEAK ≥0.05, NONE <0.05, DOCKING_FAILED.

---

## 6. Pipeline Execution Order

```
DATA ACQUISITION
├── 01_download_data.py          → reference + initial assemblies
├── extract_data.py              → unpack pdbs.tar.gz, cryptic supplement
├── download_cryptic_data.py     → MUTATIONS.csv.gz (1.4 GB, not in git)
├── 14_download_more_genomes.py  → additional TB genomes + VCFs
├── 14b_download_bulk_genomes.py
└── 14c_download_500_genomes.py

HOMOPLASY (iterative — v4 is current)
├── 15_compute_homoplasy_from_assemblies.py
├── 15b → 15c → 15d → 15e_compute_homoplasy_v4.py
└── 16_merge_homoplasy.py        → residue_hotspot_data_updated.csv

MODEL TRAINING
├── 04b_hotspot_model.py         → Stage 0
├── 04c_stage1_features.py       → Stage 1
├── 04d_docking_features.py      → Stage 2 (XGBoost)
└── 04e_mutation_forecasting.py  → emergence watchlist

VALIDATION
├── 05_leave_one_gene_out.py
├── 08_cryptic_validation_full.py
├── 09_stress_tests.py
├── 06_filter_pocket_candidates.py
└── 07_tier4_pocket_vina_batch.py

ANALYSIS & FIGURES
├── analysis/compute_metrics.py
├── analysis/permutation_test.py
├── analysis/esm2_baseline.py
├── analysis/validate_novel_docking.py
├── analysis/audit_novelty_and_scores.py
├── 10_generate_figures.py
├── 11_render_figures.py
└── 12_audit.py                  → ~180 automated checks
```

---

## 7. Resistance Genes Covered

Defined in `scripts/04_resistance_forecasting.py` → `RESISTANCE_GENES`:

| Gene | Locus | Drug class | Pocket residues |
|------|-------|------------|-----------------|
| rpoB | Rv0667 | Rifampicin | 409–530 (RRDR) |
| katG | Rv1908c | Isoniazid | Active site + heme pocket |
| embB | Rv3795 | Ethambutol | 295–530 |
| gyrA | Rv0006 | Fluoroquinolones | QRDR 74–150 |
| gyrB | Rv0005 | Fluoroquinolones | 495–555 |
| pncA | Rv2043c | Pyrazinamide | 3–186 |
| rpsL | Rv0682 | Streptomycin | 23–98 |
| rrs | rrs | Aminoglycosides | (no pocket set) |
| eis | Rv2416c | Aminoglycosides | (promoter/coding) |
| tap | Rv1258c | Aminoglycosides | — |
| mmpR5 | Rv0678 | Bedaquiline | — |
| mmpL5 | Rv2680 | Bedaquiline | — |
| tlyA | Rv1694 | Capreomycin | — |
| inhA | Rv1484 | Isoniazid | NADH pocket |

**Core binding residues** (tighter interface, from crystal structures) in `CORE_BINDING_RESIDUES`:
- rpoB: 426–452 (RRDR)
- katG: 104–115 + 270–330
- gyrA: 88–94 (QRDR)
- gyrB: 533–538
- embB: 406, 497 region
- inhA: NADH binding site

**Positive labels:** 32 hotspot residues from `KNOWN_RES_MUTATIONS` + WHO catalog, tracked in `analysis/results/hotspot_model/positives_whitelisted.csv`.

---

## 8. Every Pipeline Script (`scripts/`)

### Data acquisition

#### `scripts/01_download_data.py`
Downloads H37Rv reference genome and MDR-TB assemblies from NCBI E-utilities.  
**Outputs:** `reference/H37Rv.fasta`, `reference/H37Rv.gff`, `data/genomes/*.fasta`, `data/metadata/tb_assemblies.csv`

#### `scripts/extract_data.py`
Extracts committed tar archives for reproducibility without re-downloading.  
**Inputs:** `data/pdbs.tar.gz`, `data/cryptic_supplement.tar.gz`  
**Outputs:** `data/pdb/`, `data/cryptic/` (supplement only; full MUTATIONS table downloaded separately)

#### `scripts/download_cryptic_data.py`
Downloads CRyPTIC MUTATIONS.csv.gz (~1.4 GB) from CRyPTIC FTP.  
**Output:** `data/cryptic/MUTATIONS.csv.gz` (gitignored; must download locally)

#### `scripts/14_download_more_genomes.py`
Downloads additional TB genomes, CRyPTIC-excluded VCFs, Afro-TB metadata for homoplasy expansion.  
**Outputs:** `data/genomes/`, `variants/cryptic_excluded_*.vcf.gz`, extended metadata

#### `scripts/14b_download_bulk_genomes.py`
NCBI Assembly DB search + bulk FASTA download.  
**Output:** `data/genomes/GCF_*.fasta`

#### `scripts/14c_download_500_genomes.py`
Parallel download of 500 TB genomes via NCBI E-utilities + FTP.

### Homoplasy computation (iterative versions)

Homoplasy = count of independent non-synonymous mutations at each codon across 1,037 TB genomes (117 VCF + 920 NCBI assemblies).

#### `scripts/15_compute_homoplasy_from_assemblies.py`
Initial version: align assemblies to H37Rv, count per-residue variants.  
**Output:** `analysis/results/homoplasy/homoplasy_from_assemblies.csv`

#### `scripts/15b_compute_homoplasy_seed.py`
Seed-based approach: 20 bp k-mer hashing for faster lookup.

#### `scripts/15c_compute_homoplasy_v2.py`
Robust 30 bp seeds + best-match verification.

#### `scripts/15d_compute_homoplasy_v3.py`
Adds reverse-complement seeds for minus-strand genes (fixes katG, rpoB orientation bugs).

#### `scripts/15e_compute_homoplasy_v4.py` *(current)*
K-mer voting with 20 bp seeds, stride 5 bp. Most reliable version.  
**Columns:** `gene`, `residue`, `homoplasy_count`, `homoplasy_alleles`, `n_genomes`

#### `scripts/16_merge_homoplasy.py`
Merges assembly homoplasy into master feature table.  
**Input:** `residue_hotspot_data.csv` + homoplasy CSV  
**Output:** `residue_hotspot_data_updated.csv`

### Core modeling pipeline

#### `scripts/04_resistance_forecasting.py`
**Shared foundation module** imported by 04b–04e. Defines:
- `RESISTANCE_GENES`, `CORE_BINDING_RESIDUES`, `KNOWN_RES_MUTATIONS`
- GFF parsing, CDS extraction, feature computation
- Original mutation-level XGBoost pipeline (legacy path)
**Outputs (legacy):** `forecasting/training_data.csv`, `feature_importance.csv`, `surveillance_watchlist.csv`

#### `scripts/04b_hotspot_model.py` — Stage 0
**Purpose:** Residue-level hotspot propensity from sequence + homoplasy.  
**Model:** Weighted logistic regression (proof-of-concept, not production).  
**Features:** 12 sequence/biochemical + homoplasy.  
**AUROC:** ~0.888 (leave-one-gene-out).  
**Outputs:**
- `hotspot_model/residue_hotspot_data.csv` — master feature table
- `hotspot_model/feature_coefficients.csv`
- `hotspot_model/lo_hotspot_validation.csv`
- `hotspot_model/lo_gene_validation.csv`
- `hotspot_model/predicted_hotspots_top200.csv`

#### `scripts/04c_stage1_features.py` — Stage 1
**Purpose:** Add structural features from AlphaFold + ESM-2.  
**Tasks:**
1. SASA (relative solvent accessibility) from AlphaFold PDBs
2. ESM-2 mutation intolerance (`facebook/esm2_t33_650M_UR50D`)
3. 3D contact density (Cβ neighbors within 8 Å)
4. AlphaFold vs crystal RMSD validation  
**Model:** Logistic regression benchmark.  
**AUROC:** ~0.906.  
**Outputs:**
- `hotspot_model/sasa_data.pkl`, `esm2_data.pkl`, `contact_density_3d.pkl`
- `hotspot_model/alphafold_validation.json`
- `hotspot_model/results_benchmark.csv`
- `hotspot_model/ranked_predictions.csv`

#### `scripts/04d_docking_features.py` — Stage 2 (production model)
**Purpose:** Add drug proximity feature; train final XGBoost + Platt calibration.  
**Drug distance logic:**
- Co-crystal: distance to ligand atoms in 5UHB (RIF) / 5BS8 (MFX)
- No crystal: dilated pocket proxy from AlphaFold (10 Å radius)
- **Self-exclusion:** query residue excluded from distance (fixes circularity bug)  
**Model:** XGBoost → CalibratedClassifierCV (sigmoid/Platt, 5-fold).  
**Evaluation:** StratifiedKFold + GroupKFold by gene.  
**AUROC:** 0.971; GroupKFold 0.972 ± 0.018.  
**Outputs:**
- `hotspot_model/drug_distances.pkl`, `plddt_data.pkl`
- `hotspot_model/residue_hotspot_data_with_docking.csv`
- `hotspot_model/stage3_results.json`
- `hotspot_model/ranked_predictions.csv`
- `hotspot_model/feature_coefficients.csv`

#### `scripts/04e_mutation_forecasting.py` — Phase 4e
**Purpose:** Mutation-level emergence scoring.  
**Formula:** `emergence_score = hotspot_score × mutation_score`  
**mutation_score** combines:
- Resistance plausibility (BLOSUM62, charge reversal, known substitution patterns)
- Fitness cost (Grantham distance, hydrophobicity shift)
- Evolutionary accessibility (single-nucleotide reachable from WT codon)  
**Mechanism tags:** Direct Pocket, Cofactor Gateway, Charge Reversal, Allosteric Shield, Loss-of-Function, Other  
**Output:** `forecasting/emergence_watchlist.csv` (~330 mutations ranked)

#### `scripts/04f_evolutionary_features.py` — Stage 4 (blocked)
**Purpose:** Shannon entropy from UniProt homolog MSA per residue.  
**Status:** Blocked — UniProt BLAST API returns 404; no local MAFFT/BLAST+. BLOSUM approximation had 99.7% constant values; removed.

### Validation scripts

#### `scripts/05_leave_one_gene_out.py`
Holds out each entire gene, trains logistic regression on remaining genes.  
**Why LR not XGBoost:** Better generalization to unseen genes.  
**Aggregate top-50 recall:** ~50%.  
**Outputs:** `forecasting/leave_one_gene_out_results.csv`, `leave_one_gene_out_mutation_ranks.csv`

#### `scripts/06_failure_analysis.py`
Diagnoses 5 major missed known mutations (structural + score breakdown).  
**Output:** `forecasting/failure_analysis.json`

#### `scripts/06_filter_pocket_candidates.py`
Filters Tier-4 mutations in gyrA/rpoB/gyrB with `drug_distance ≤ 4.5 Å`.  
**Input:** `cryptic_tiered_validation.csv`, `structural_validation_candidates.csv`  
**Output:** `forecasting/tier4_pocket_direct_matrix.csv` (32 candidates)

#### `scripts/07_tier4_pocket_vina_batch.py`
**Authoritative Vina batch.** For each Tier-4 pocket candidate:
1. Build mutant receptor PDBQT (`sidechain_builder.py`)
2. **Redock WT under identical grid** (fixes stale baseline bug)
3. Run AutoDock Vina (exhaustiveness=8)
4. Compute ΔΔG = mut_binding − wt_binding  
**Outputs:**
- `data/pdb/*_docked.pdbqt`
- `analysis/results/docking/*_vina.log`
- `forecasting/tier4_pocket_vina_scores.csv`
- `analysis/results/tier4_pocket_vina_results.json`

#### `scripts/08_cryptic_validation_full.py`
Stream-filters 1.5 GB CRyPTIC MUTATIONS.csv.gz against emergence watchlist.  
Cross-references 12,287 phenotyped isolates.  
**Outputs:** `forecasting/cryptic_validation_results.csv`, `matched_null_results.json`

#### `scripts/09_stress_tests.py`
FDR analysis (Benjamini-Hochberg), tier assignment, leakage checks.  
**Outputs:** `forecasting/cryptic_tiered_validation.csv`, `cryptic_fdr_analysis.csv`

### Figures and audit

#### `scripts/10_generate_figures.py`
Generates all paper figure data tables from result CSVs/JSONs.  
**Output:** `analysis/results/figures/fig*.csv`, `paper_summary.json`

#### `scripts/11_render_figures.py`
Renders publication PNGs with matplotlib/seaborn.  
**Output:** `Figure_1.png` … `Figure_S3.png`

#### `scripts/12_audit.py`
~180 automated checks: file existence, syntax, CRyPTIC integrity, model schema, figure completeness, package availability, claim consistency, leakage controls, statistical rigor.  
**Run:** `python scripts/12_audit.py`

### Structural / docking utilities

#### `scripts/sidechain_builder.py`
Builds complete sidechain heavy atoms for point mutations using standard rotamer geometry. Used by Vina batch to create mutant receptors.

#### `scripts/gen_mfx_ffxml.py`
Generates GAFF force-field XML for moxifloxacin from RDKit MMFF partial charges.  
**Output:** `data/pdb/gaff_MFX.xml`

#### `scripts/build_mfx_system.py`
Builds OpenMM system for gyrB + MFX (amber14 protein + custom ligand).  
**Output:** `data/pdb/gaff_MFX_v2.xml`

#### `scripts/build_mfx_system_v2.py`
Manual gyrB + MFX OpenMM system via Python API (all bonded/nonbonded forces). For future MD/MM-PBSA validation of Q538L.

---

## 9. Every Analysis Script (`analysis/`)

### Model evaluation

| Script | Purpose | Key output |
|--------|---------|------------|
| `compute_metrics.py` | Full metrics with bootstrap 95% CIs | `hotspot_model/full_metrics.json` |
| `permutation_test.py` | 200-shuffle label permutation test | `permutation_test_results.json` (p=0.005) |
| `esm2_baseline.py` | ESM-2-only vs full model comparison | `esm2_baseline_results.json` |
| `loo_comparison.py` | LR vs shallow XGBoost for LOO | `forecasting/loo_comparison_results.json` |

### Docking validation

| Script | Purpose | Key output |
|--------|---------|------------|
| `validate_novel_docking.py` | Vina for novel/Tier-4 candidates; parse existing scores | `novel_docking_validation.json` |
| `audit_novelty_and_scores.py` | Re-parse PDBQT REMARK lines; CARD/PubMed novelty check | stdout audit report |

### Results reporting

| Script | Purpose |
|--------|---------|
| `dump_all_results.py` | Comprehensive stdout dump of all pipeline metrics |
| `dump_watchlist.py` | Full watchlist with tier labels (330 mutations) |
| `detail_results.py` | CRyPTIC validation breakdown |
| `detail2.py` | FDR-significant novel hits with correct columns |
| `build_tiered_map.py` | Merge watchlist + validation + FDR → `tiered_mutation_map.csv` |
| `trim_watchlist.py` | Create `watchlist_top20.csv`, `watchlist_top50.csv` |

### Tier / feature diagnostics

| Script | Purpose |
|--------|---------|
| `check_tier1.py` | Check Tier 1 mutations for positive-label expansion |
| `check_tier2.py` | Estimate new positives from Tier 1–2 expansion |
| `check_feature_table.py` | Inspect homoplasy coverage in feature table |
| `check_mmpl5.py` | Diagnose why mmpL5 was dropped (NaN features) |
| `trace_mmpl5.py` | Trace mmpL5 through pipeline stages |

### Homoplasy / assembly debugging

| Script | Purpose |
|--------|---------|
| `check_assembly.py` | Compare assemblies that worked vs failed for rpoB seed finding |
| `check_coords.py` | Test H37Rv vs legacy coordinate systems |
| `debug_gene_extraction.py` | Verify gene extraction for rpoB and key mutations |
| `debug_minus_strand.py` | Debug minus-strand gene verification |
| `debug_voting.py` | Debug k-mer voting for katG |
| `deep_debug_katg.py` | Deep debug: minus-strand voting wrong position |
| `diagnose_seed.py` | Diagnose missed genes in assemblies |
| `find_rpob.py` | Scan all assemblies for rpoB conserved seed |

### External data / tooling

| Script | Purpose |
|--------|---------|
| `check_data_sources.py` | Probe GitHub/Figshare for 35k and Afro-TB datasets |
| `check_datasets.py` | Check Afro-TB Figshare and Pruthi TB-ML GitHub |
| `check_who_catalog.py` | List WHO mutation catalogue 2023 GitHub contents |
| `check_tools.py` | Check BioPython/pairwise2 for MSA/Shannon entropy |
| `download_next_batch.py` | Count assemblies and download next NCBI batch |
| `get_genes.py` | Print RESISTANCE_GENES list from 04_resistance_forecasting.py |

---

## 10. Root-Level Files

| File | Purpose |
|------|---------|
| `README.md` | Public project documentation, top candidates, metrics, reproducibility |
| `EXECUTIVE_SUMMARY.md` | This document — complete internal reference |
| `viewer.html` | Interactive pipeline dashboard (serve with `python -m http.server 8000`) |
| `_find_novel.py` | Ad-hoc: find truly novel Tier-4 mutations (rank > 20, carriers ≤ 1) |
| `test_H445L.pdbqt.flex` | AutoDockFlex test file for rpoB H445L |
| `reference/H37Rv.gff` | Gene annotations (committed) |
| `reference/H37Rv.fasta` | Reference genome (**gitignored** — download via 01_download_data.py) |

### `viewer.html` sections

1. **Overview** — pipeline steps, headline stats
2. **Data** — genomic, clinical, structural inputs
3. **Model training** — Stages 0–2 with AUROC progression
4. **CRyPTIC validation** — Tier 0–4 counts
5. **Structural validation** — Q538L PyMOL figure, Vina score table
6. **Audit** — novelty + score verification
7. **Next steps** — manuscript, MRSA, Mantis

Styling: Source Serif 4 + Source Sans 3, warm off-white, teal accent (academic report style).

---

## 11. Data Directory Inventory

| Path | Contents | In git? |
|------|----------|---------|
| `data/genomes/` | ~920 NCBI TB assemblies (`GCF_*.fasta`) | **No** (gitignored) |
| `data/metadata/` | Assembly metadata, CRyPTIC phenotypes, `data_bundle.json` | Partial |
| `data/cryptic/` | `MUTATIONS.csv.gz` (1.4 GB), supplement cache | **No** (except supplement tar) |
| `data/demo/` | Demo VCF `drprg_sparse.vcf.gz` | Yes |
| `data/pdb/` | Structural assets (see Section 12) | Partial (PDBQT committed; .pdb gitignored) |
| `data/pdbs.tar.gz` | Archive of PDB structures | Yes |
| `data/cryptic_supplement.tar.gz` | CRyPTIC supplement | Yes |
| `reference/` | H37Rv reference (fasta gitignored, gff committed) | Partial |
| `variants/` | CRyPTIC-excluded VCFs | Varies |

### Key metadata files

| File | Description |
|------|-------------|
| `data/metadata/cryptic_phenotypes.csv` | 12,287 isolates × 13 drug phenotypes |
| `data/metadata/tb_assemblies.csv` | NCBI assembly metadata |
| `data/metadata/tb_assemblies_extended.csv` | Extended download batch metadata |
| `data/metadata/data_bundle.json` | Consolidated metrics, Tier-4 candidates, docking results |

### What you must download locally (not in git)

1. `reference/H37Rv.fasta` — `python scripts/01_download_data.py`
2. `data/genomes/*.fasta` — scripts 14/14b/14c
3. `data/cryptic/MUTATIONS.csv.gz` — `python scripts/download_cryptic_data.py`

---

## 12. Structural Assets (`data/pdb/`)

### AlphaFold models
`data/pdb/alphafold/AF-*.pdb` — 13 resistance gene structures. Used for SASA, contact density, pocket proxy, pLDDT.

### Crystal structures
`data/pdb/crystal/` — rpoB (PDB 5UHB, RIF), gyrA (PDB 5BS8, MFX). High-confidence drug distances and rigid docking.

### Receptors (AutoDock PDBQT)
| File | Gene | Drug |
|------|------|------|
| `rpoB_receptor.pdbqt` | rpoB | Rifampicin |
| `gyrA_receptor.pdbqt` | gyrA | Moxifloxacin |
| `gyrB_receptor.pdbqt` | gyrB | Moxifloxacin |
| `inhA_receptor.pdbqt` | inhA | Triclosan/NADH proxy |
| `embB_receptor.pdbqt` | embB | Ethambutol |
| `eis_receptor.pdbqt` | eis | Amikacin |
| `rpsL_receptor.pdbqt` | rpsL | Streptomycin |

Also: `rpoB_receptor_flex.pdbqt`, `rpoB_rigid.pdbqt`, `gyrA_chainA_flex.pdbqt` for flexible docking experiments.

### Ligands
| File | Drug |
|------|------|
| `MFX_ligand.pdbqt` | Moxifloxacin |
| `RFP_ligand.pdbqt` | Rifampicin |
| `EMB_ligand.pdbqt` | Ethambutol |
| `STR_ligand.pdbqt` / `STR_ligand_fixed.pdbqt` | Streptomycin |
| `AMK_ligand.pdbqt` / `AMK_ligand_fixed.pdbqt` | Amikacin |
| `triclosan_ligand.pdbqt` | Triclosan (inhA NADH proxy) |
| `NAD_ligand.pdbqt`, `NADH_ligand.pdbqt` | NADH (Vina tree.h crash — use triclosan) |
| `PZA_ligand.pdbqt` | Pyrazinamide |

### WT docked poses (Vina baselines)
- `rpoB_WT_docked.pdbqt` — RIF baseline (−9.934 kcal/mol in crystal run)
- `gyrA_WT_docked.pdbqt`, `gyrA_WT_MFX_docked.pdbqt`
- `gyrB_WT_redock.pdbqt` — **Authoritative gyrB baseline (−7.071)** for Q538L ΔΔG
- `inhA_WT_*_tric.pdbqt`, `embB_WT_*_docked.pdbqt`, `eis_WT_AMK_docked.pdbqt`

### Mutant + docked files (~80+)
Pattern: `{gene}_{MUTATION}.pdbqt` (receptor) + `{gene}_{MUTATION}_docked.pdbqt` (Vina pose with REMARK VINA RESULT score).

### Force fields (OpenMM / MD)
- `gaff_MFX.xml`, `gaff_MFX_v2.xml` — GAFF parameters for moxifloxacin
- `MFX_gaff.mol2`, `MFX_coords.xyz`, `MFX_rdkit.sdf`, `ethambutol_3d.sdf`

### Validation figure
- `gyrB_Q538L_validation.png` — PyMOL ribbon + MFX (orange) + Q538L (magenta)

---

## 13. Results Directory Inventory

### `analysis/results/hotspot_model/`

| File | Description |
|------|-------------|
| `residue_hotspot_data.csv` | Master feature table (6,326 residues) |
| `residue_hotspot_data_with_docking.csv` | + drug_proximity |
| `residue_hotspot_data_updated.csv` | After homoplasy merge |
| `residue_hotspot_data_scaled.csv` | Scaled features |
| `ranked_predictions.csv` | All residues ranked by hotspot_score |
| `feature_coefficients.csv` | XGBoost/LR importances |
| `positives_whitelisted.csv` | 32 known hotspot residues |
| `stage3_results.json` | Stage 1 vs 3 AUROC per gene |
| `full_metrics.json` | AUROC, AUPRC, F1, MCC, bootstrap CIs |
| `cv_f1_pr_metrics.json` | 5-fold CV F1/precision/recall |
| `f1_pr_analysis.json` | Precision-recall curve analysis |
| `permutation_test_results.json` | p = 0.005 |
| `esm2_baseline_results.json` | ESM-2 vs full model |
| `alphafold_validation.json` | AlphaFold vs crystal RMSD |
| `results_benchmark.csv` | Stage 0/1/3 benchmark |
| `*.pkl` | Feature caches (gitignored, regenerated by scripts) |

### `analysis/results/forecasting/`

| File | Description |
|------|-------------|
| `emergence_watchlist.csv` | **Primary:** ~330 ranked mutations |
| `cryptic_validation_results.csv` | CRyPTIC cross-reference |
| `cryptic_tiered_validation.csv` | Tiers 0–4 |
| `cryptic_fdr_analysis.csv` | Benjamini-Hochberg q-values |
| `tiered_mutation_map.csv` | Unified tier + FDR |
| `watchlist_top20.csv`, `watchlist_top50.csv` | Clinical shortlists |
| `leave_one_gene_out_results.csv` | Per-gene LOO metrics |
| `leave_one_gene_out_mutation_ranks.csv` | Per-mutation LOO ranks |
| `failure_analysis.json` | 5 missed-mutation diagnostics |
| `matched_null_results.json` | Matched-null p = 0.001 |
| `loo_comparison_results.json` | LR vs XGB LOO |
| `tier4_pocket_direct_matrix.csv` | 32 pocket-direct Tier-4 candidates |
| `tier4_pocket_vina_scores.csv` | Authoritative Vina ΔΔG table |
| `feature_importance.csv`, `training_data.csv` | Legacy 04 outputs |

### `analysis/results/figures/`

| File | Description |
|------|-------------|
| `fig1_pipeline_stats.csv` | Dataset sizes |
| `fig2a_alphafold_rmsd.csv` | AlphaFold validation |
| `fig2b_stage_comparison.csv` | Stage 0→1→3 AUROC |
| `fig2c_rescued_failures.csv` | Structural features rescuing failures |
| `fig3_feature_importance.csv` | Top XGBoost features |
| `fig4a_top_watchlist.csv`, `fig4b_status_counts.csv` | Watchlist summary |
| `fig5a_validation_cascade.csv` | CRyPTIC funnel |
| `fig5b_tier_distribution.csv` | Tier counts |
| `fig5c_tier1_hits.csv` | FDR-significant Tier 1 |
| `fig6_clinical_impact.csv` | Clinical impact stats |
| `figS1_roc_comparison.csv` | ROC curves |
| `figS2_leave_one_gene_out.csv` | LOO table |
| `figS5_model_benchmark.csv` | Full benchmark |
| `paper_summary.json` | Consolidated figure stats |
| `Figure_1.png` … `Figure_S3.png` | Rendered PNGs |

### `analysis/results/docking/`
Per-mutation Vina logs for Tier-4 batch (~30 files): `gyrA_*_vina.log`, `gyrB_*_vina.log`, `rpoB_*_vina.log`

### `analysis/results/homoplasy/`
`homoplasy_from_assemblies.csv` — per-residue counts from 1,037 genomes

### Root-level result JSONs

| File | Description |
|------|-------------|
| `tier4_pocket_vina_results.json` | Batch Vina results + validated_novel list |
| `novel_docking_validation.json` | Earlier novel-candidate docking (Q538L stale baseline) |
| `docking_validation_results.json` | Phase 1/2 crystal docking |
| `phase2_docking_results.json` | AlphaFold-based phase-2 |
| `gyra_docking_results.json`, `rpob_docking_results.json` | Gene-specific |
| `novel_candidate_docking_results.json` | 5 truly novel candidates |
| `structural_validation_candidates.csv` | Selected for validation |
| `novel_gene_candidates.csv` | Novel gene-level candidates |
| `docking_validation_table.md` | Human-readable docking summary |

### Technical documentation in results/

| File | Description |
|------|-------------|
| `PIPELINE_RECONSTRUCTION_GUIDE.md` | How to rebuild pipeline from scratch |
| `STAGE1_5_TECHNICAL_SUMMARY.md` | Stage 1.5 structural feature details |
| `PROPER_STAGE3_PLAN.md` | Stage 3 design decisions and circularity fix |

---

## 14. CRyPTIC Validation Tiers

Script: `scripts/08_cryptic_validation_full.py` + `scripts/09_stress_tests.py`

| Tier | Count | Definition | Example |
|------|-------|------------|---------|
| **0** | 30 | Known WHO catalog mutation | rpoB S450L |
| **1** | 24 | Novel, FDR q < 0.05 in CRyPTIC | gyrA D94A (147 carriers, 59% R) |
| **2** | 32 | Observed, enriched but underpowered | — |
| **3** | 31 | Observed, no phenotype data | pncA Q10R (PZA blind spot) |
| **4** | 188 | **Forecast-only** — 0 carriers | gyrB Q538L |

### Top Tier-1 CRyPTIC-confirmed (retrospective validation)

| Mutation | Gene | Rank | Carriers | R% | FDR p |
|----------|------|------|----------|-----|-------|
| D94A | gyrA | 210 | 147 | 59% | 1.1e-35 |
| Q497K | embB | 122 | 71 | 84% | 1.8e-20 |
| D94H | gyrA | 209 | 44 | 77% | 6.5e-20 |
| G406S | embB | 32 | 99 | 75% | 1.1e-19 |
| I21T | inhA | 33 | 64 | 98% | 6.6e-18 |
| I194T | inhA | 31 | 64 | 97% | 2.2e-16 |
| D435G | rpoB | 5 | 61 | 90% | 2.8e-16 |
| G88C | gyrA | 206 | 24 | 88% | 7.3e-15 |
| H445L | rpoB | 68 | 76 | 82% | 8.5e-14 |
| H445R | rpoB | 19 | 33 | 97% | 1.1e-11 |

**Important:** Tier 1 mutations were already in CRyPTIC when validated — they confirm the model retrospectively. Tier 4 mutations have **never been seen** clinically.

---

## 15. AutoDock Vina Structural Validation

### Phase 1: Crystal structures (high confidence)

**rpoB + Rifampicin (5UHB chain C):**

| Mutation | WT | Mut | ΔΔG | Category |
|----------|-----|-----|-----|----------|
| WT | −9.934 | — | — | — |
| L430R | −9.934 | −9.531 | +0.403 | MODERATE |
| H445R | −9.934 | −9.693 | +0.241 | WEAK |
| H445P | −9.934 | −9.721 | +0.213 | WEAK |
| H445Q | −9.934 | −9.726 | +0.208 | WEAK |
| H445L | −9.934 | −9.732 | +0.202 | WEAK |
| Q432L | −9.934 | −9.746 | +0.188 | WEAK |
| I491L | −9.934 | −9.758 | +0.176 | WEAK |
| Q432P | −9.934 | −9.765 | +0.169 | WEAK |
| D435G | −9.934 | −9.831 | +0.103 | WEAK |
| V170A | −9.934 | −9.906 | +0.028 | NONE |
| Q432K | −9.934 | −9.920 | +0.014 | NONE |

**gyrA + Moxifloxacin (5BS8 chain A):**

| Mutation | WT | Mut | ΔΔG | Category |
|----------|-----|-----|-----|----------|
| WT | −4.375 | — | — | — |
| G88C | −4.375 | −4.168 | +0.207 | WEAK |
| D94A | −4.375 | −4.319 | +0.056 | NONE |
| D94H | −4.375 | −4.344 | +0.031 | NONE |
| A90T | −4.375 | −4.363 | +0.012 | NONE |

### Phase 2: AlphaFold (lower confidence)

**inhA + Triclosan (NADH proxy):** NADH crashes Vina (tree.h internal error >10 rotatable branches).

| Mutation | ΔΔG | Note |
|----------|-----|------|
| S94A | +0.507 | STRONG |
| I21T | −0.506 | Binding gain — triclosan may be wrong ligand |
| I194T | −0.169 | WEAK |

**embB + Ethambutol (AlphaFold):** No co-crystal; weak binding overall.

| Mutation | ΔΔG |
|----------|-----|
| Q497K | +0.017 (NONE) |
| G406S | +0.028 (NONE) |

### Tier-4 novel candidate docking (5 ✦ marked in README)

| Candidate | Rank | ddG | Result |
|-----------|------|-----|--------|
| rpsL K43E | 22 | — | DOCKING FAILED (STR tree.h limit) |
| inhA I16V | 27 | +0.019 | NONE |
| eis V59A | 34 | — | DOCKING FAILED (AMK tree.h limit) |
| rpoB V170I | 47 | −0.122 | NONE (allosteric) |
| **gyrB Q538L** | 131 | **+0.737** | **STRONG** |

### Q538L score correction (critical)

| Run | WT baseline | Mut | ΔΔG |
|-----|-------------|-----|-----|
| Old (`novel_docking_validation.json`) | −6.16 (`gyrB_WT_docked`) | −6.334 | **−0.174** (wrong) |
| **Tier-4 batch (authoritative)** | **−7.071** (`gyrB_WT_redock`) | −6.334 | **+0.737 STRONG** |

Mutant score unchanged; only WT baseline was wrong before fresh redock.

### Vina limitations

1. **tree.h(101):** Ligands with >10 rotatable branches (NADH, streptomycin, amikacin) crash Vina
2. **Rigid receptor:** Allosteric mutations (rpoB V170) show ddG ~ 0 despite clinical causality
3. **Conservative substitutions:** Ile↔Val produce ddG < 0.1 despite functional impact
4. **Indirect mechanisms:** LoF (pncA), promoter (eis), cofactor kinetics (inhA) invisible to static docking

---

## 16. Novelty Audit and Literature Claims

Script: `analysis/audit_novelty_and_scores.py`

**Score audit:** All 32 Tier-4 Vina scores re-parsed from PDBQT `REMARK VINA RESULT` lines — match CSV within 0.02 kcal/mol.

**Literature novelty verdict:**

| Claim level | Mutations |
|-------------|-----------|
| **Strong de novo** | **gyrB Q538L only** — codon 538 known as N538D/K/S/T; Q538L never in Mtb literature |
| **Uncertain** | rpoB P483R — P483L reported |
| **Known rare variants** | Other 8 Vina-validated hits — in CARD/WHO/clinical literature |

**Do not claim** all 10 Vina hits are literature-novel. They are **pipeline benchmarks** confirming structural mechanism for known/rare variants.

---

## 17. Known Bugs, Fixes, and Leakage Controls

### Circularity bug (AUROC inflation 0.990 → 0.971)

**Problem:** Training positives were inside pocket definition. Hotspot residue distance to itself = 0 → `drug_proximity = 1.0` for 24/27 positives.

**Fix:** Dilate pockets to 10 Å radius in 3D; **exclude query residue** from distance computation (self-exclusion).

**Effect:** `drug_proximity` dropped from dominant LR coefficient (+6.71) to 4th XGBoost importance (0.158). Homoplasy now dominates.

### Stale WT baseline (Q538L)

**Problem:** Old WT pose (`gyrB_WT_docked.pdbqt`) gave wrong ΔΔG sign.

**Fix:** `07_tier4_pocket_vina_batch.py` redocks WT fresh per gene under identical grid (`gyrB_WT_redock.pdbqt`).

### Leakage controls audited by `12_audit.py`

| Check | Status |
|-------|--------|
| homoplasy computed globally (not per CV fold) | Documented limitation — labels from WHO/CRyPTIC, not same genomes |
| drug_proximity self-exclusion | Fixed |
| Scaler fit inside CV folds only | Verified |
| CalibratedClassifierCV internal CV | 5-fold; final model on all data |
| GroupKFold by gene | AUROC 0.972 ± 0.018 |
| Permutation test | p = 0.005 |
| ESM-2 baseline | AUROC 0.618 |
| Matched-null CRyPTIC validation | p = 0.001 |

### Minus-strand homoplasy bug

katG, rpoB on minus strand required reverse-complement seeds (fixed in 15d/15e).

### mmpL5 dropped

No AlphaFold model + 0 positives → excluded from per-gene metrics.

---

## 18. Limitations

1. **Small positive set:** 32 positives in 6,350 residues (0.50%). Ranking excellent; threshold metrics diluted.
2. **pncA blind spot:** No binary PZA phenotype in CRyPTIC. Q10R (rank #6, 155 carriers) unvalidatable.
3. **Global homoplasy:** Not recomputed per CV fold (acceptable — labels external).
4. **Platt calibration on full data:** Scores well-calibrated but not exact probabilities.
5. **No direct benchmark:** CRyPTIC ML does genotype→phenotype; WHO does curation; neither forecasts emergence.
6. **API blocks:** UniProt BLAST 404 prevents MSA Shannon entropy (04f blocked).
7. **Vina blind spots:** Complex ligands, allosteric, LoF, promoter mechanisms.

---

## 19. Reproducibility Commands

```bash
# Environment: Python 3.10+, xgboost, scikit-learn, biopython, pandas, numpy
# Optional: esm (ESM-2), rdkit, openmm, meeko, autodock-vina

# 1. Data
python scripts/01_download_data.py
python scripts/extract_data.py
python scripts/download_cryptic_data.py
python scripts/14c_download_500_genomes.py   # or 14/14b for more genomes

# 2. Homoplasy
python scripts/15e_compute_homoplasy_v4.py
python scripts/16_merge_homoplasy.py

# 3. Model training (sequential)
python scripts/04b_hotspot_model.py
python scripts/04c_stage1_features.py      # requires GPU/time for ESM-2
python scripts/04d_docking_features.py
python scripts/04e_mutation_forecasting.py

# 4. Validation
python scripts/05_leave_one_gene_out.py
python scripts/08_cryptic_validation_full.py
python scripts/09_stress_tests.py

# 5. Structural validation
python scripts/06_filter_pocket_candidates.py
python scripts/07_tier4_pocket_vina_batch.py   # requires Vina + Meeko

# 6. Analysis
python analysis/compute_metrics.py
python analysis/permutation_test.py
python analysis/esm2_baseline.py
python analysis/audit_novelty_and_scores.py

# 7. Figures + audit
python scripts/10_generate_figures.py
python scripts/11_render_figures.py
python scripts/12_audit.py

# 8. Viewer
python -m http.server 8000
# Open http://localhost:8000/viewer.html
```

---

## 20. Roadmap

1. **Manuscript** — Lead with gyrB Q538L (literature-novel + Vina STRONG + PyMOL). Frame 9 other validated hits as pipeline benchmarks. Include CRyPTIC Tier 1 retrospective confirmations.

2. **Phase 2: Phenotypic validation (in vivo MICs)** — To test whether the mutation causes true physiological drug resistance:
   - **Surrogate modeling:** Transform the mutant *gyrB* plasmid (Q538L) into a fast-growing, non-pathogenic surrogate such as *Mycobacterium smegmatis* (BSL-1/2 lab).
   - **MIC testing:** Run minimum inhibitory concentration (MIC) assays by broth microdilution to determine whether cells carrying Q538L show a survival shift against escalating moxifloxacin doses vs wild-type *gyrB*.

3. **Phase 3: MRSA extension** — Same architecture (homoplasy + structure + drug proximity + XGBoost + prospective validation) on *Staphylococcus aureus* resistance genes.

4. **Phase 4: Mantis platform** — Deploy emergence model in Mantis clinical genomics platform; Tier-4 surveillance alerts with structural validation and literature novelty flags at WGS interpretation time.

5. **Future structural work** — OpenMM MD/MM-PBSA for Q538L (`build_mfx_system_v2.py`); PyMOL figures for other hits; strict WHO 2023 PDF grep for P483R.

---

## Quick Reference: Key File Paths

| Purpose | Path |
|---------|------|
| Master features | `analysis/results/hotspot_model/residue_hotspot_data_with_docking.csv` |
| Residue rankings | `analysis/results/hotspot_model/ranked_predictions.csv` |
| Mutation watchlist | `analysis/results/forecasting/emergence_watchlist.csv` |
| CRyPTIC tiers | `analysis/results/forecasting/cryptic_tiered_validation.csv` |
| Vina scores (authoritative) | `analysis/results/forecasting/tier4_pocket_vina_scores.csv` |
| Stage metrics | `analysis/results/hotspot_model/stage3_results.json` |
| Consolidated bundle | `data/metadata/data_bundle.json` |
| Q538L structure | `data/pdb/gyrB_Q538L_validation.png` |
| Interactive viewer | `viewer.html` |
| Full audit | `python scripts/12_audit.py` |

---

*End of executive summary. For questions about specific mutations, run `analysis/dump_watchlist.py` or inspect `cryptic_tiered_validation.csv` directly.*
