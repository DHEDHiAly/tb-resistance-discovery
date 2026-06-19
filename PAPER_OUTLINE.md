# Paper Outline — Forecasting Emerging Tuberculosis Drug Resistance

**Working title:** *Prospective forecasting of emerging M. tuberculosis resistance mutations using homoplasy, structural features, and clinical validation*

**Authors:** Aly Dhedhi, Vinay Singamsetty, Li-Lun Ho, Manolis Kellis (Kellis Lab, MIT)

**Authoritative metrics:** `analysis/results/PUBLICATION_METRICS.md` (regenerate: `python scripts/13_final_publication_audit.py`)

---

## Abstract (~250 words)

**Background.** Genomic surveillance catalogs known resistance mutations retrospectively. Can we forecast which mutations will emerge before they appear in clinical databases?

**Methods.** We trained a residue-level hotspot classifier across 13 TB resistance genes (6,350 residues, 32 positive hotspots) using homoplasy from 1,037 genomes, AlphaFold structural features, ESM-2 mutation intolerance, and drug-binding proximity. XGBoost with Platt calibration (Stage 2) achieved AUROC **0.971** and AUPRC **0.560** (5-fold CV). We enumerated SNV-accessible mutations, scored P(emergence), and prospectively validated against **12,287 CRyPTIC** isolates. Top Tier-4 pocket-direct candidates were structurally validated with AutoDock Vina (fresh WT redock).

**Results.** All 32 known hotspots ranked in the top 32 by full-model score. **24** novel predictions reached FDR significance in CRyPTIC (matched-null p < 0.001). **188** mutations remain forecast-only (0 carriers). **10** Tier-4 pocket-direct mutations showed Vina ΔΔG ≥ +0.15 kcal/mol; **gyrB Q538L** (ΔΔG +0.737) is literature-novel at this substitution in Mtb.

**Conclusions.** Integrating evolutionary recurrence, structure, and drug proximity enables prospective resistance emergence forecasting with independent clinical and biophysical validation.

**Keywords:** tuberculosis, antimicrobial resistance, machine learning, CRyPTIC, homoplasy, AlphaFold, prospective validation

---

## 1. Introduction

### 1.1 Clinical problem
- TB drug resistance is detected after mutations accumulate in surveillance catalogs (WHO, CRyPTIC).
- Genotype→phenotype tools predict resistance given a mutation already seen; they do not forecast *emergence*.

### 1.2 Gap
- No standard benchmark for prospective emergence forecasting at residue/mutation resolution.
- Need: rank unseen mutations for surveillance before clinical detection.

### 1.3 Contributions
1. Staged feature pipeline: sequence → structure (ESM-2, SASA, contacts) → drug proximity → XGBoost.
2. Mutation-level emergence scoring with CRyPTIC tiered validation (Tiers 0–4).
3. Orthogonal AutoDock Vina validation for Tier-4 pocket-direct candidates.
4. Lead discovery: **gyrB Q538L** — pipeline-novel, literature-novel, structurally validated.

---

## 2. Results

### 2.1 Pipeline overview
**Figure 1** — Study design schematic (`Figure_1.png`)

| Stage | Input | Model | Key metric |
|-------|-------|-------|------------|
| 0 | Sequence + homoplasy | Logistic regression | AUROC 0.888 |
| 1 | + AlphaFold, ESM-2 | Logistic regression | AUROC 0.906 |
| 2 | + Drug proximity | XGBoost + Platt | AUROC **0.971**, AUPRC **0.560** |

### 2.2 Hotspot model performance
**Figure 2** — Stage comparison, AlphaFold RMSD (`Figure_2.png`)  
**Figure S1** — ROC curve from 5-fold OOF predictions (`Figure_S1.png`)  
**Figure S_PR** — Precision–recall curve (`Figure_S_PR.png`)

| Metric | Stratified 5-fold CV | GroupKFold by gene |
|--------|---------------------|-------------------|
| AUROC | 0.968 ± 0.034 | **0.974 ± 0.018** |
| AUPRC | 0.465 ± 0.157 (92× random) | 0.586 ± 0.226 |
| Best F1 | 0.550 ± 0.119 | 0.676 ± 0.191 |
| Top-20 recall | 0.662 (21/32 per-fold avg) | **0.741** |

**Full-model ranking (calibrated):** all 32 positives in ranks 1–32; score gap 0.40 between last positive and first negative.

**Figure 3** — Feature importance (`Figure_3.png`): homoplasy_alleles (0.269), drug_proximity (0.158), homoplasy_count (0.149).

**Controls:** Permutation test p = 0.005; ESM-2-only AUROC 0.618 (+0.35 lift with full model).

### 2.3 Mutation-level emergence forecasting
**Figure 4** — Top watchlist mutations by emergence score (`Figure_4.png`)

- 332 ranked mutations in emergence watchlist.
- P(emergence) = hotspot_score × mutation_score (fitness, accessibility, plausibility).
- **188 Tier-4** forecast-only (0 CRyPTIC carriers).

### 2.4 CRyPTIC prospective validation
**Figure 5** — Validation cascade and tier distribution (`Figure_5.png`)

| Tier | N | Interpretation |
|------|---|----------------|
| 0 | 30 | WHO-known (sanity check) |
| 1 | **24** | FDR q < 0.05 — novel, clinically confirmed |
| 2 | 32 | Enriched, underpowered |
| 3 | 31 | Observed, no phenotype (pncA blind spot) |
| 4 | **188** | Forecast-only surveillance targets |

Matched-null: real Tier-1 count (24) vs null mean 9.3, **p = 0.001**.

**Table — Top Tier-1 hits:** D94A (gyrA), G406S/Q497K (embB), D435G/H445R (rpoB), I21T/I194T (inhA), etc.

### 2.5 Structural validation (AutoDock Vina)
**Lead figure:** gyrB Q538L + moxifloxacin (`data/pdb/gyrB_Q538L_validation.png`)

| Mutation | ΔΔG | Category | Literature novel? |
|----------|-----|----------|-------------------|
| **gyrB Q538L** | **+0.737** | STRONG | **Yes** |
| rpoB L452M | +2.137 | STRONG | No (CARD) |
| rpoB P483R | +1.254 | STRONG | Uncertain |
| rpoB L452R | +1.045 | STRONG | No |
| + 6 moderate hits | ≥ +0.15 | MODERATE | Mostly known rare |

32 Tier-4 pocket-direct candidates docked; 10 pass ΔΔG ≥ +0.15 threshold.

**Critical methods note:** WT baseline redocked per gene under identical grid (Q538L authoritative ΔΔG +0.737).

### 2.6 Clinical impact summary
**Figure 6** — Surveillance utility (`Figure_6.png`)

- Tier-4 list = prospective WGS surveillance panel.
- Distinguish pipeline-novel vs literature-novel vs structurally validated.

---

## 3. Discussion

### 3.1 Why ranking beats threshold F1 at 0.5% prevalence
- F1 @ 0.5 diluted by 6,318 negatives; AUROC/AUPRC and top-K recall are appropriate metrics.

### 3.2 Homoplasy as dominant signal
- Independent allelic recurrence across 1,037 genomes captures selective pressure not visible in structure alone.

### 3.3 Circularity fix for drug proximity
- Self-exclusion + 10 Å dilated pocket removed AUROC inflation (0.990 → 0.971).

### 3.4 Vina limitations
- tree.h crash for complex ligands (STR, AMK, NADH).
- Rigid receptor blind to allosteric mechanisms (rpoB V170).
- Conservative substitutions yield low ΔΔG despite clinical relevance.

### 3.5 Limitations
- 32 positives; global homoplasy not per-fold; pncA phenotype blind spot; no direct emergence benchmark.

### 3.6 Future work
- **Phase 2 — Phenotypic validation (in vivo MICs):** Transform mutant *gyrB* (Q538L) into *Mycobacterium smegmatis* as a BSL-1/2 surrogate; run broth microdilution MIC assays to test for a moxifloxacin resistance shift vs wild-type *gyrB*.
- MRSA extension; Mantis platform integration; MD/MM-PBSA for Q538L.

---

## 4. Methods

### 4.1 Data
- H37Rv reference; 13 resistance genes; 1,037 genomes for homoplasy.
- CRyPTIC MUTATIONS.csv.gz (12,287 phenotyped isolates).
- AlphaFold structures; co-crystals rpoB (5UHB), gyrA (5BS8).

### 4.2 Feature engineering
- 16 Stage-2 features (see `PUBLICATION_METRICS.md`).
- Drug proximity: co-crystal distance or 10 Å pocket proxy; query residue excluded.

### 4.3 Models
- Stage 0–1: logistic regression.
- Stage 2: XGBoost (depth 6, lr 0.05, 300 trees, scale_pos_weight=10) + CalibratedClassifierCV (Platt, 5-fold).

### 4.4 Cross-validation
- StratifiedKFold (5) for global metrics.
- GroupKFold by gene for conservative generalization estimate.

### 4.5 Emergence scoring
- `04e_mutation_forecasting.py`: SNV enumeration + emergence_score.

### 4.6 CRyPTIC validation
- `08_cryptic_validation_full.py`, `09_stress_tests.py`: tiers, Benjamini–Hochberg FDR, matched-null.

### 4.7 Docking
- `06_filter_pocket_candidates.py` (≤ 4.5 Å).
- `07_tier4_pocket_vina_batch.py`: mutant build, WT redock, Vina, ΔΔG.

### 4.8 Statistical analysis
- Permutation test (200 shuffles); bootstrap 95% CIs for F1/precision/recall.

---

## 5. Figures and Tables (submission checklist)

| ID | File | Content |
|----|------|---------|
| **Fig 1** | `analysis/results/figures/Figure_1.png` | Pipeline overview |
| **Fig 2** | `Figure_2.png` | Model performance: stage table + ROC + PR |
| **Fig 3** | `Figure_3.png` | XGBoost feature importance |
| **Fig 4** | `Figure_4.png` | CRyPTIC tier distribution + Tier 1 hits |
| **Fig S2** | `Figure_S2.png` | Leave-one-gene-out validation |
| **Fig 5 (struct)** | `data/pdb/gyrB_Q538L_validation.png` | Q538L PyMOL structural validation |

Supporting CSVs (not separate figures): `fig2b_stage_comparison.csv`, `fig3_feature_importance.csv`, `fig5a/b/c_*.csv`, `fig_roc_curve.csv`, `fig_pr_curve.csv`, `figS2_leave_one_gene_out.csv`.

**Removed as redundant:** old Figure 4 (stale watchlist), Figure 5/6 (duplicated CRyPTIC/infographic), separate S1/S_PR (merged into Figure 2).

---

## 6. Reproducibility

```bash
python scripts/13_final_publication_audit.py
python scripts/10_generate_figures.py
python scripts/11_render_figures.py
python scripts/12_audit.py
```

---

## 7. Target venues (internal)

1. *Nature Communications* or *Nature Microbiology* — lead with Q538L discovery + CRyPTIC prospective validation.
2. *Bioinformatics* / *PLOS Comp Bio* — methods + benchmark emphasis.
3. Conference: ISMB / ASM Microbe poster with `viewer.html` demo.
