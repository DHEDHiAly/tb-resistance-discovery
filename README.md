# Forecasting Emerging Tuberculosis Drug Resistance

**Structural hotspot prediction → mutation forecasting → prospective clinical validation**

Kellis Lab, MIT

---

## Overview

Can we predict which TB resistance mutations will emerge *before* they appear in resistance catalogs?

This project builds a multi-stage framework that:

1. **Identifies structural signatures of known resistance hotspots** across 13 M. tuberculosis resistance genes
2. **Forecasts specific SNV-accessible mutations** most likely to emerge under drug pressure
3. **Validates predictions prospectively** in 12,287 independent CRyPTIC clinical isolates

### Key Results

| Stage | Method | AUROC | Hotspots in Top 20 |
|-------|--------|:-----:|:------------------:|
| 0 | Sequence features (strand, helix, charge) | 0.888 | 7/21 |
| 1 | + Structural features (SASA, ESM-2, 3D contact density) | **0.910** | **17/21** |
| 1.5 | + Drug docking distance | 0.938 | 17/21 |

**Prospective validation in CRyPTIC (12,287 isolates):**

| Tier | Count | Description |
|------|:-----:|-------------|
| 1 | **22** | Validated novel predictions (FDR q < 0.05) |
| 2 | 32 | Observed with resistance enrichment (low power) |
| 3 | 27 | Observed, no phenotype data (pncA/rpsL) |
| 4 | 179 | Forecast-only (prospective surveillance targets) |

30 known WHO mutations confirmed as pipeline sanity check.

---

## Project Structure

```
tb-resistance-discovery/
├── scripts/                     # Analysis pipeline
│   ├── 01_download_data.py      # Data acquisition (reference, CRyPTIC phenotypes)
│   ├── 04_resistance_forecasting.py  # Shared utilities + Phase 4 XGBoost model
│   ├── 04b_hotspot_model.py     # Stage 0: residue-level logistic regression
│   ├── 04c_stage1_features.py   # Stage 1: structural features (SASA, ESM-2, 3D contact)
│   ├── 04d_docking_features.py  # Task 7: drug docking features from 5UHB
│   ├── 04e_mutation_forecasting.py # Mutation-level P(emergence) pipeline
│   ├── 05_leave_one_gene_out.py # Leave-one-gene-out cross-validation
│   ├── 06_failure_analysis.py   # Case studies of missed mutations
│   ├── 08_cryptic_validation_full.py # CRyPTIC mutation matrix + cross-reference
│   ├── 09_stress_tests.py       # FDR correction, leakage check, tiered categorization
│   └── 10_generate_figures.py   # Paper figure data (6 main + 4 supplementary)
│
├── analysis/results/
│   ├── hotspot_model/           # Stage 0-1.5 outputs
│   │   ├── ranked_predictions.csv
│   │   ├── feature_coefficients.csv
│   │   ├── alphafold_validation.json
│   │   └── ranked_predictions_with_docking.csv
│   └── forecasting/             # Mutation-level outputs
│       ├── emergence_watchlist.csv           # 315 candidate mutations
│       ├── cryptic_validation_results.csv    # Full CRyPTIC cross-reference
│       ├── cryptic_tiered_validation.csv     # Tier 1-4 categorization
│       ├── cryptic_fdr_analysis.csv          # FDR-corrected enrichment tests
│       ├── leave_one_gene_out_results.csv    # LOO validation
│       └── failure_analysis.json             # Missed mutation case studies
│
├── data/
│   ├── cryptic/                 # Downloaded CRyPTIC tables
│   │   └── cache/               # Filtered mutation matrix cache
│   ├── metadata/
│   │   └── cryptic_phenotypes.csv  # 12,287 samples, 13 drug phenotypes
│   ├── demo/                    # Demo VCF (117 samples)
│   └── reference/               # H37Rv genome, GFF, AlphaFold PDBs
```

---

## Pipeline: Step by Step

### Stage 0: Sequence-Based Hotspot Prediction

`04b_hotspot_model.py` — Logistic regression using 12 sequence-derived features (strand propensity, helix propensity, hydrophobicity, charge, volume, hbond, conservation, etc.) across ~6,600 residues in 13 resistance genes.

- **AUROC: 0.888**
- **7/21 known hotspots in Top 20**
- Major failures: rpoB D435 at #597, V170 at #953, L452 at #526

### Stage 1: Structural Features

`04c_stage1_features.py` — Adds three structural biology features:

1. **SASA** — relative solvent accessibility from AlphaFold structures
2. **ESM-2 intolerance** — evolutionary sequence model predicts substitution tolerance
3. **3D contact density** — number of spatially neighboring residues within 8Å

- **AUROC: 0.910**
- **17/21 known hotspots rescued to Top 20**
- D435: #597 → #20, V170: #953 → #24, L452: #526 → #19

### Stage 1.5: Drug Docking

`04d_docking_features.py` — Rifampicin contact distances from 5UHB co-crystal structure, aligned to AlphaFold rpoB.

- **AUROC: 0.938**
- 4 missed hotspots (V170, I491, V125, N538) NOT rescued — drug proximity is necessary but insufficient
- Implicates allostery, dynamics, and fitness effects

### Mutation Forecasting

`04e_mutation_forecasting.py` — For each of the top 50 hotspot residues, enumerate all SNV-accessible mutations (315 total). Score by:

```
P(emergence) = P(hotspot | features) × weighted combination of:
  - Resistance potential (0.45)
  - Fitness preservation (0.30)
  - Evolutionary accessibility (0.25)
```

### Leave-One-Gene-Out Validation

`05_leave_one_gene_out.py` — Tests generalization to unseen genes.

| Gene | Top-20 | Top-50 | Top-100 |
|------|:------:|:------:|:-------:|
| embB | 2/9 | 4/9 | 4/9 |
| gyrA | 1/5 | 3/5 | 4/5 |
| pncA | 2/4 | 2/4 | 2/4 |
| rpoB | 1/10 | 5/10 | 7/10 |
| katG | 0/3 | 1/3 | 1/3 |
| rpsL | 0/2 | 2/2 | 2/2 |

**Aggregate: 52% Top-50 recall for unseen genes.**

### CRyPTIC Validation

`08_cryptic_validation_full.py` — Downloads CRyPTIC MUTATIONS table (1.4 GB), filters for 12,287 phenotype-matched samples, extracts all resistance gene mutations, and cross-references the 315 watchlist predictions.

**81 of 290 watchlist mutations observed clinically.** Of those with phenotype data, **38/54 (70%) enriched in resistant isolates.**

### Tiered Categorization

`09_stress_tests.py` — FDR correction (Benjamini-Hochberg), leakage analysis, literature cross-reference, tier assignment.

### Figures

`10_generate_figures.py` — Produces all data tables for 6 main + 4 supplementary figures.

---

## Strongest Tier 1 Validated Predictions

| Mutation | Gene | Rank | Carriers | R% | OR | FDR p |
|----------|------|:----:|:--------:|:--:|:--:|:-----:|
| Q445R | embB | 37 | 56 | 100% | inf | 2.5e-36 |
| D94A | gyrA | 93 | 147 | 59% | 9.2 | 5.4e-36 |
| G406S | embB | **6** | 99 | 75% | 11.1 | 8.3e-20 |
| Q497K | embB | 121 | 71 | 84% | 20.2 | 1.2e-20 |
| D435G | rpoB | 41 | 61 | 90% | 14.7 | 3.1e-16 |
| H445R | rpoB | 31 | 33 | 97% | 49.4 | 1.2e-11 |
| I491L | rpoB | 220 | 20 | 100% | inf | 1.8e-8 |
| S315G | katG | **4** | 4 | 75% | — | (low n) |

---

## Data Sources

- **CRyPTIC Consortium**: 12,287 M. tuberculosis clinical isolates with drug susceptibility phenotypes for 13 antibiotics. Data from [EBI FTP](https://ftp.ebi.ac.uk/pub/databases/cryptic/).
- **AlphaFold2**: Predicted structures for all 13 resistance proteins (downloaded from EBI AlphaFold DB).
- **5UHB**: Cryo-EM structure of M. tuberculosis RNA polymerase with rifampicin (PDB: 5UHB).
- **WHO Catalog**: Known resistance mutations from the WHO mutation catalog (Walker et al., 2021).
- **ESM-2**: Evolutionary scale modeling for substitution intolerance scores.

---

## Requirements

Python 3.12+ with standard scientific Python stack:
- `pandas`, `numpy`, `scipy`
- `scikit-learn`, `xgboost`
- `biopython`, `requests`
- `matplotlib`, `seaborn` (for figure plotting)

Full environment: see `environment.yml` or `requirements.txt`.

---

## Reference

Walker, T. M., et al. (2021). "The 2021 WHO catalogue of Mycobacterium tuberculosis complex mutations associated with drug resistance." *The Lancet Microbe*.

The CRyPTIC Consortium (2022). "A data compendium of Mycobacterium tuberculosis antibiotic resistance." *bioRxiv*.
