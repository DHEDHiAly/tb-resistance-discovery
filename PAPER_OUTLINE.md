# Paper Outline — Forecasting Emerging Tuberculosis Drug Resistance

**Working title:** *Prospective forecasting of emerging M. tuberculosis resistance mutations using homoplasy, structural features, and clinical validation*

**Authors:** Aly Dhedhi, Vinay Singamsetty, Li-Lun Ho, Manolis Kellis (Kellis Lab, MIT)

**Authoritative metrics:** `analysis/results/PUBLICATION_METRICS.md` (regenerate: `python scripts/13_final_publication_audit.py`)

**Last metrics sync:** June 2026

---

## Abstract (~250 words)

**Background.** Genomic surveillance catalogs known resistance mutations retrospectively. Can we forecast which mutations will emerge before they appear in clinical databases?

**Methods.** We trained a residue-level hotspot classifier across 13 TB resistance genes (6,350 residues, 32 positive hotspots, 0.50% prevalence) using homoplasy from 1,037 genomes, AlphaFold structural features, ESM-2 mutation intolerance, and drug-binding proximity (16 features; self-excluded pocket distance). A staged pipeline progressed from logistic regression (Stages 0–1) to XGBoost with Platt calibration (Stage 2). We enumerated SNV-accessible mutations, scored P(emergence), prospectively validated against **12,287 CRyPTIC** isolates, and structurally validated Tier-4 pocket-direct candidates with AutoDock Vina (fresh WT redock per gene).

**Results.** Stratified 5-fold CV: AUROC **0.968 ± 0.034**, AUPRC **0.465 ± 0.157** (92× random), best F1 **0.550 ± 0.119** (precision 0.631, recall 0.562). Stage ablation peaked at AUROC 0.971 / AUPRC 0.560. GroupKFold by gene: AUROC **0.974 ± 0.018**, AUPRC **0.586 ± 0.226**, top-20 recall **0.741**. All 32 known hotspots ranked in the top 32 by calibrated full-model score (score gap 0.40). **24** novel predictions reached FDR significance in CRyPTIC (matched-null p < 0.001); **188** remain forecast-only (0 carriers). **10/32** Tier-4 pocket-direct dockings passed ΔΔG ≥ +0.15 kcal/mol. **gyrB Q538L** (ΔΔG **+0.737**, literature-novel) is the lead de novo finding.

**Conclusions.** Integrating homoplasy, structure, and drug proximity enables prospective resistance emergence forecasting with independent clinical and biophysical validation.

**Keywords:** tuberculosis, antimicrobial resistance, machine learning, precision–recall, CRyPTIC, homoplasy, AlphaFold, prospective validation

---

## Key results at a glance

| Domain | Metric | Value |
|--------|--------|-------|
| **Dataset** | Residues / positives | 6,350 / 32 (0.50%) |
| | Resistance genes | 13 |
| | Homoplasy genomes | 1,037 |
| | CRyPTIC isolates | 12,287 |
| **Primary CV (stratified 5-fold)** | AUROC | **0.968 ± 0.034** |
| | AUPRC | **0.465 ± 0.157** (92× random) |
| | Best F1 | **0.550 ± 0.119** (P 0.631 / R 0.562) |
| | Top-20 recall | **0.662** (21/32) |
| | Top-50 / Top-100 recall | 0.829 / 0.857 |
| **Stage ablation (Stage 2)** | AUROC / AUPRC | 0.971 / **0.560** (111× random) |
| | Top-20 recall | 0.657 |
| **PR operating points (OOF curve)** | Precision @ recall ≥ 0.25 / 0.50 | **0.80** / **0.43** |
| **GroupKFold (by gene)** | AUROC | **0.974 ± 0.018** |
| | AUPRC | **0.586 ± 0.226** |
| | Best F1 | 0.676 ± 0.191 |
| | Top-20 recall | **0.741** |
| **Full-model ranking** | All positives in top N | **32/32** in top 32 |
| | Top-20 recall | 20/32 (62.5%) |
| | Score gap (pos − neg) | **0.401** |
| **Controls** | Permutation test | **p = 0.005** |
| | ESM-2-only AUROC | 0.618 (+0.35 lift) |
| **CRyPTIC** | Tier 1 (FDR q < 0.05) | **24** |
| | Tier 4 (forecast-only) | **188** |
| | Matched-null p | **0.001** |
| **Vina (Tier-4 pocket)** | Docked / validated | 32 / **10** (ΔΔG ≥ 0.15) |
| **Lead hit** | gyrB Q538L ΔΔG | **+0.737** (STRONG) |

---

## 1. Introduction

### 1.1 Clinical problem
- TB drug resistance is detected after mutations accumulate in surveillance catalogs (WHO, CRyPTIC).
- Genotype→phenotype tools predict resistance given a mutation already seen; they do not forecast *emergence*.

### 1.2 Gap
- No standard benchmark for prospective emergence forecasting at residue/mutation resolution.
- Need: rank unseen mutations for surveillance before clinical detection.

### 1.3 Contributions
1. Staged feature pipeline: sequence → structure (ESM-2, SASA, contacts) → drug proximity → XGBoost + Platt calibration.
2. Mutation-level emergence scoring with CRyPTIC tiered validation (Tiers 0–4) and matched-null enrichment test.
3. Orthogonal AutoDock Vina validation for Tier-4 pocket-direct candidates (fresh WT redock).
4. Lead discovery: **gyrB Q538L** — pipeline-novel, literature-novel, structurally validated (ΔΔG +0.737).

---

## 2. Results

### 2.1 Pipeline overview
**Figure 1** — Study design (`analysis/results/figures/Figure_1.png`)

| Stage | Script | Model | AUROC | AUPRC | Top-20 recall (CV) |
|-------|--------|-------|-------|-------|-------------------|
| 0 | `04b` | Logistic regression | 0.888 | — | — |
| 1 | `04c` | + structural features | 0.906 | 0.205 | 0.386 |
| 2 | `04d` | XGBoost + drug proximity + Platt | **0.971** | **0.560** | **0.657** |

332 mutations in emergence watchlist; P(emergence) = hotspot_score × mutation_score.

---

### 2.2 Hotspot model performance (ROC, PR, F1)
**Figure 2** — Stage progression table + ROC + precision–recall curves (`Figure_2.png`)  
Curve data: `fig_roc_curve.csv`, `fig_pr_curve.csv`

#### Stratified 5-fold cross-validation (primary)

| Metric | Value |
|--------|-------|
| AUROC | **0.968 ± 0.034** |
| AUPRC | **0.465 ± 0.157** (92× random) |
| Best F1 (per-fold optimal threshold) | **0.550 ± 0.119** |
| Best F1 precision / recall | 0.631 / 0.562 |
| F1 @ fixed threshold 0.5 | 0.384 ± 0.142 |
| Top-20 recall (CV) | **0.662** (21/32) |
| Top-50 / Top-100 recall | 0.829 / 0.857 |

#### Precision–recall (PR) by evaluation scheme

Random baseline AUPRC = positive rate = **0.005** (32 / 6,350).

| Evaluation | AUPRC | vs random | Notes |
|------------|-------|-----------|-------|
| Stage 1 (structural) | **0.205** | 41× | Logistic regression |
| Stage 2 ablation (`04d`) | **0.560** | 111× | Feature-addition benchmark |
| **Stratified 5-fold CV** | **0.465 ± 0.157** | **92×** | **Primary headline** |
| Pooled OOF curve (Fig 2C) | **0.435** | 87× | Single ROC/PR plot |
| GroupKFold by gene | **0.586 ± 0.226** | 117× | Conservative holdout |

**Per-fold AUPRC (stratified 5-fold):** 0.613 · 0.537 · 0.619 · 0.295 · 0.259  
**Per-fold AUPRC (GroupKFold, 4 folds):** 0.855 · 0.313 · 0.420 · 0.756

#### PR-derived F1 and operating points

| Metric | Stratified CV (mean) | Pooled OOF | GroupKFold (mean) |
|--------|---------------------|------------|-------------------|
| Best F1 | **0.550 ± 0.119** | 0.500 | **0.676 ± 0.191** |
| Precision @ best F1 | **0.631** | 0.536 | — |
| Recall @ best F1 | **0.562** | 0.469 | — |
| F1 @ threshold 0.5 | 0.384 ± 0.142 | 0.400 | — |
| Precision @ threshold 0.5 | — | 0.556 | — |
| Recall @ threshold 0.5 | — | 0.313 | — |

**Precision at fixed recall (pooled OOF PR curve, `fig_pr_curve.csv`):**

| Recall ≥ | Precision |
|----------|-----------|
| 0.25 | **0.80** |
| 0.50 | **0.43** |
| 0.75 | 0.14 |
| 1.00 | 0.012 |

At recall ≥ 0.25, **4 of 5** predicted positives are true hotspots (precision 0.80). At recall ≥ 0.50, precision remains **0.43** — ~86× the random rate.

#### GroupKFold by gene (conservative generalization)

| Metric | Value |
|--------|-------|
| AUROC | **0.974 ± 0.018** |
| AUPRC | **0.586 ± 0.226** |
| Best F1 | 0.676 ± 0.191 |
| Top-20 recall | **0.741** |

#### Full-model ranking (calibrated XGBoost, all residues)

| Metric | Value |
|--------|-------|
| All 32 positives in top 32 | **Yes** (ranks 1–32) |
| Top-20 recall | **20/32** (62.5%) |
| Top-50 / Top-100 recall | 32/32 |
| Last positive score / first negative | 0.650 / 0.249 (gap **0.401**) |

**Interpretation for paper:** At 0.5% prevalence, threshold-based F1 is diluted (F1@0.5 = 0.38); **AUROC, AUPRC, and top-K recall** are the primary ranking metrics. PR curve (Figure 2C) shows strong enrichment over random baseline (0.005).

**Figure 3** — XGBoost feature importance (`Figure_3.png`, `fig3_feature_importance.csv`)

| Feature | Gain |
|---------|------|
| homoplasy_alleles | **0.269** |
| drug_proximity | 0.158 |
| homoplasy_count | 0.149 |
| inner_distance | 0.048 |
| strand_propensity | 0.047 |
| hydrophobicity | 0.046 |

**Controls:** Permutation test **p = 0.005** (200 shuffles). ESM-2-only AUROC **0.618**; full model lift **+0.350 AUROC**.

**Figure S2** — Leave-one-gene-out (`Figure_S2.png`, `figS2_leave_one_gene_out.csv`)

---

### 2.3 CRyPTIC prospective validation
**Figure 4** — Tier distribution + Tier 1 hits (`Figure_4.png`)  
Data: `fig5b_tier_distribution.csv`, `fig5c_tier1_hits.csv`

| Tier | N | Interpretation |
|------|---|----------------|
| 0 | 30 | WHO-known (pipeline sanity check) |
| 1 | **24** | FDR q < 0.05 — novel, clinically confirmed |
| 2 | 32 | Enriched, underpowered |
| 3 | 31 | Observed, no phenotype (pncA blind spot) |
| 4 | **188** | Forecast-only (0 CRyPTIC carriers) |

**Matched-null validation:** Tier 1 count **24** vs null mean **9.3** (1,000 permutations), **p = 0.001**.

#### Representative Tier 1 hits (FDR q < 0.05)

| Mutation | Gene | Rank | Carriers | R% | FDR p |
|----------|------|------|----------|-----|-------|
| D94A | gyrA | 210 | 147 | 59% | 1.1×10⁻³⁵ |
| Q497K | embB | 122 | 71 | 84% | 1.8×10⁻²⁰ |
| D94H | gyrA | 209 | 44 | 77% | 6.5×10⁻²⁰ |
| G406S | embB | 32 | 99 | 75% | 1.1×10⁻¹⁹ |
| I21T | inhA | 33 | 64 | 98% | 6.6×10⁻¹⁸ |
| I194T | inhA | 31 | 64 | 97% | 2.2×10⁻¹⁶ |
| D435G | rpoB | 5 | 61 | 90% | 2.8×10⁻¹⁶ |
| G88C | gyrA | 206 | 24 | 88% | 7.3×10⁻¹⁵ |
| H445L | rpoB | 68 | 76 | 82% | 8.5×10⁻¹⁴ |
| H445R | rpoB | 19 | 33 | 97% | 1.1×10⁻¹¹ |

Tier 1 = retrospective clinical confirmation of model-predicted novel mutations. Tier 4 = prospective surveillance targets not yet seen in clinic.

---

### 2.4 Structural validation (AutoDock Vina, Tier-4 pocket-direct)
**Structural figure:** gyrB Q538L + moxifloxacin (`data/pdb/gyrB_Q538L_validation.png`)  
Scores: `analysis/results/forecasting/tier4_pocket_vina_scores.csv`

**Filter:** Tier 4, gyrA/rpoB/gyrB, drug_distance ≤ 4.5 Å → **32 candidates**  
**Pass criterion:** ΔΔG ≥ +0.15 kcal/mol (mut − WT, fresh WT redock per gene)

#### All 10 structurally validated mutations

| Mutation | Rank | Emergence score | ΔΔG | Category | Literature novel? |
|----------|------|-----------------|-----|----------|-------------------|
| rpoB L452M | 181 | 0.200 | **+2.137** | STRONG | No (CARD) |
| rpoB P483R | 140 | 0.227 | **+1.254** | STRONG | Uncertain (P483L known) |
| rpoB L452R | 132 | 0.233 | **+1.045** | STRONG | No (CARD) |
| **gyrB Q538L** | 131 | 0.234 | **+0.737** | STRONG | **Yes (de novo)** |
| rpoB Q432R | 225 | 0.091 | +0.399 | MODERATE | No (CARD) |
| gyrA S91A | 170 | 0.206 | +0.203 | MODERATE | No |
| gyrA G88S | 30 | 0.418 | +0.179 | MODERATE | No |
| gyrA G88V | 207 | 0.167 | +0.178 | MODERATE | No |
| gyrA G88D | 29 | 0.423 | +0.170 | MODERATE | No |
| rpoB I491N | 124 | 0.240 | +0.156 | MODERATE | No |

**Q538L authoritative baseline:** WT redock −7.071 kcal/mol; mutant −6.334; ΔΔG **+0.737**. Earlier stale WT pose (−6.16) incorrectly gave negative ΔΔG.

**Vina categories:** STRONG ≥ 0.40 · MODERATE ≥ 0.15 · WEAK ≥ 0.05 · NONE < 0.05

Only **gyrB Q538L** satisfies pipeline-novel + literature-novel + structural STRONG. The other nine are pipeline benchmarks confirming pocket-direct mechanism for known/rare variants.

---

### 2.5 Novelty definitions (critical for claims)

| Term | Definition | Q538L |
|------|------------|-------|
| Pipeline novel | Tier 4, 0 CRyPTIC carriers | Yes |
| Literature novel | Not in CARD/PubMed/WHO for Mtb at this substitution | **Yes** |
| Structurally validated | Pocket-direct + ΔΔG ≥ +0.15 | **Yes (+0.737)** |

---

## 3. Discussion

### 3.1 Why ranking beats threshold F1 at 0.5% prevalence
- 6,318 negatives dilute F1@0.5 (0.38); AUROC (0.97), AUPRC (0.56), and top-K recall are appropriate.
- PR curve shows model maintains precision well above random (0.005) across recall range.

### 3.2 Homoplasy as dominant signal
- `homoplasy_alleles` (gain 0.269) and `homoplasy_count` (0.149) top XGBoost features.
- Independent allelic recurrence across 1,037 genomes captures selective pressure invisible to structure alone.

### 3.3 Circularity fix for drug proximity
- Self-exclusion + 10 Å dilated pocket removed AUROC inflation (0.990 → 0.971).

### 3.4 Vina limitations
- tree.h crash for complex ligands (streptomycin, amikacin, NADH).
- Rigid receptor blind to allosteric mechanisms (e.g. rpoB V170I, ΔΔG −0.12).
- Conservative substitutions yield low ΔΔG despite clinical relevance.

### 3.5 Limitations
- 32 positive residues; global homoplasy not recomputed per CV fold.
- pncA phenotype blind spot (31 Tier 3).
- No direct prospective emergence benchmark besides CRyPTIC tiers.

### 3.6 Future work / roadmap

1. **Manuscript** — Lead with gyrB Q538L; frame 9 other Vina hits as pipeline benchmarks; include CRyPTIC Tier 1 retrospective confirmations.

2. **Phase 2: Phenotypic validation (in vivo MICs)** — Test whether Q538L causes true physiological drug resistance:
   - **Surrogate modeling:** Transform mutant *gyrB* plasmid (Q538L) into *Mycobacterium smegmatis* (BSL-1/2).
   - **MIC testing:** Broth microdilution MIC vs escalating moxifloxacin; compare Q538L vs wild-type *gyrB*.

3. **Phase 3: MRSA extension** — Same architecture on *Staphylococcus aureus* resistance genes.

4. **Phase 4: Mantis platform** — Deploy Tier-4 surveillance alerts at WGS interpretation time.

5. **Structural follow-up** — MD/MM-PBSA for Q538L (`build_mfx_system_v2.py`).

---

## 4. Methods (summary)

| Component | Detail |
|-----------|--------|
| Data | H37Rv; 13 genes; 1,037 genomes; CRyPTIC 12,287 isolates |
| Features | 16 (homoplasy, sequence, SASA, ESM-2, contacts, pLDDT, drug proximity) |
| Stage 2 model | XGBoost (depth 6, lr 0.05, 300 trees, `scale_pos_weight=10`) + Platt calibration |
| CV | StratifiedKFold (5) + GroupKFold by gene |
| Emergence | `04e`: SNV enumeration × hotspot_score |
| CRyPTIC | `08`/`09`: tiers, Benjamini–Hochberg FDR, matched-null |
| Docking | `06`/`07`: ≤4.5 Å filter; WT redock; Vina ΔΔG |
| Stats | Permutation test (p=0.005); bootstrap CIs for F1 |

---

## 5. Figures (submission checklist)

| ID | File | Content |
|----|------|---------|
| **Fig 1** | `Figure_1.png` | Pipeline overview |
| **Fig 2** | `Figure_2.png` | Stage metrics + **ROC + PR curves** |
| **Fig 3** | `Figure_3.png` | XGBoost feature importance (gain) |
| **Fig 4** | `Figure_4.png` | CRyPTIC tiers + Tier 1 table |
| **Fig S2** | `Figure_S2.png` | Leave-one-gene-out |
| **Fig 5 (struct)** | `data/pdb/gyrB_Q538L_validation.png` | Q538L PyMOL validation |

**Supporting data (tables, not figures):** `PUBLICATION_METRICS.md`, `fig2b_stage_comparison.csv`, `fig_roc_curve.csv`, `fig_pr_curve.csv`, `tier4_pocket_vina_scores.csv`, `emergence_watchlist.csv`

---

## 6. Reproducibility

```bash
python scripts/13_final_publication_audit.py   # metrics + ROC/PR CSVs
python scripts/10_generate_figures.py          # table CSVs
python scripts/11_render_figures.py            # Figure_1–4 + Figure_S2
python scripts/12_audit.py
```

---

## 7. Target venues (internal)

1. *Nature Communications* / *Nature Microbiology* — Q538L discovery + CRyPTIC prospective validation + PR/AUROC performance.
2. *Bioinformatics* / *PLOS Computational Biology* — methods + benchmark emphasis.
3. ISMB / ASM Microbe — poster with `viewer.html`.
