# Paper Outline — Emergence Forecasting Model for Protein Hotspots

**Working title:** *Emergence Forecasting Model for protein hotspots: residue-level mutation forecasting validated in M. tuberculosis drug resistance*

**Authors:** Aly Dhedhi, Vinay Singamsetty, Li-Lun Ho, Manolis Kellis (Kellis Lab, MIT)

**Authoritative metrics:** `analysis/results/PUBLICATION_METRICS.md` · regenerate: `python scripts/13_final_publication_audit.py`

**Last sync:** June 2026

---

## Narrative arc (what this paper is really about)

This is **not** a TB resistance catalog paper. The contribution is a **reusable emergence-forecasting architecture**:

1. **Learn** where pathogenic change concentrates (residue-level hotspot propensity).
2. **Enumerate** all accessible single-nucleotide variants at those sites.
3. **Score** P(emergence) = P(hotspot) × P(mutation | biochemistry, structure, selective pressure).
4. **Validate** prospectively against independent clinical/population databases and orthogonal biophysics.

**Tuberculosis** is the **only implemented benchmark** in this repository (12,287 CRyPTIC isolates, 1,037 homoplasy genomes, Vina structural validation, lead hit **gyrB Q538L**).

**Scope note:** MRSA and Alzheimer's are **planned next steps only** — not implemented, not validated, and **not part of this manuscript's results**.

---

## Abstract (~280 words)

**Background.** Genomic surveillance catalogs resistance mutations retrospectively (WHO, CRyPTIC). Genotype→phenotype tools score variants already seen in clinic. The open question: **which mutations have not yet been observed but are likely to emerge next?**

**Methods.** We introduce an **Emergence Forecasting Model (EFM)** — a staged, residue-level classifier that integrates (i) population homoplasy from 1,037 genomes, (ii) AlphaFold structural features, (iii) ESM-2 mutation intolerance, and (iv) self-excluded drug-binding proximity, trained with XGBoost and Platt calibration on 6,350 residues across 13 *M. tuberculosis* resistance genes (32 positive hotspots, 0.50% prevalence). Accessible SNVs are enumerated at high-risk codons; each receives P(emergence). We evaluate with stratified 5-fold CV, GroupKFold by gene, and **precision–recall analysis** at 0.5% prevalence (random baseline AUPRC = 0.005). TB predictions are validated against 12,287 CRyPTIC isolates (tiered FDR framework) and AutoDock Vina (Tier-4 pocket-direct, fresh WT redock).

**Results (TB benchmark).** Primary CV: AUROC **0.968 ± 0.034**, **AUPRC 0.465 ± 0.157** (92× random), **precision 0.631 / recall 0.562** at optimal F1 (**F1 0.550 ± 0.119**). At fixed recall ≥ 0.50, precision **0.432** (~86× random). GroupKFold: AUPRC **0.586 ± 0.226**, precision **0.765 / recall 0.668** at best F1. All 32 known hotspots rank in the top 32. **24** predictions retrospectively confirmed in CRyPTIC (Tier 1, matched-null p = 0.001); **188** remain forecast-only (Tier 4). Lead de novo finding: **gyrB Q538L** (ΔΔG +0.737, literature-novel).

**Conclusions.** EFM enables prospective TB resistance mutation forecasting with strong precision–recall performance and independent CRyPTIC/Vina validation. The architecture is designed to transfer to other domains; **MRSA and Alzheimer's application are future work** (§5).

**Keywords:** emergence forecasting, precision–recall, homoplasy, XGBoost, tuberculosis, antimicrobial resistance, prospective validation

---

## 1. Introduction

### 1.1 The retrospective trap
- Genotype→phenotype tools (Mykrobe, Pathogenwatch, PHASTest) answer: *given this mutation, is the isolate resistant?*
- ClinVar/gnomAD answer: *have we seen this variant before?*
- Neither ranks **unseen** substitutions for **surveillance priority** before clinical detection.

### 1.2 The Emergence Forecasting Model (EFM) — core idea
EFM reframes the problem as **ranking residues and mutations by P(emergence)** using:
- **Selective pressure** (homoplasy / population recurrence),
- **Structural vulnerability** (AlphaFold SASA, contacts, pLDDT),
- **Functional constraint** (ESM-2 intolerance as a feature),
- **Mechanistic proximity** (distance to ligand-binding pocket, self-excluded to prevent circularity).

TB drug resistance is the **first and only full instantiation** reported here. The pipeline is gene-agnostic in design, but **all results in this paper are TB-only**.

### 1.3 Contributions (this work)
1. **Novel model:** Staged EFM (sequence → structure → drug proximity → XGBoost + Platt) with explicit P(emergence) decomposition and SNV enumeration.
2. **Rigorous PR evaluation:** Full precision–recall reporting at 0.5% prevalence — the operative metric regime for rare hotspot discovery.
3. **Prospective validation framework:** CRyPTIC tier system (Tiers 0–4) + matched-null enrichment + orthogonal Vina ΔΔG.
4. **TB lead discovery:** **gyrB Q538L** — pipeline-novel, literature-novel, structurally STRONG.

---

## 2. Methods *(primary technical section)*

### 2.1 Problem formulation

**Input:** Reference proteome for a gene set G; positive hotspot residues H⁺ (from curated catalogs); population genomic data; optional structures and ligand definitions.

**Output:** Ranked list of SNV-accessible mutations with emergence scores and validation tiers.

**Two-level factorization:**

```
P(emergence | mutation m at residue r) = P(hotspot | r, features) × P(mutation | m, r, biochemistry)
```

- **P(hotspot):** Stage 2 XGBoost classifier (16 features, calibrated).
- **P(mutation):** Weighted combination of resistance plausibility (45%), fitness (30%), evolutionary accessibility (25%).

---

### 2.2 Data sources (TB instantiation)

| Source | N | Role |
|--------|---|------|
| H37Rv reference (GFF + FASTA) | 13 genes, 6,350 residues | Search space |
| WHO 2021 resistance catalog | 32 hotspot residues | Training labels (0.50% prevalence) |
| TB genome assemblies | 1,037 | Homoplasy (`15e_compute_homoplasy_v4.py`) |
| CRyPTIC | 12,287 isolates | Prospective / retrospective validation |
| AlphaFold PDBs | 13 proteins | SASA, contacts, pLDDT, pocket geometry |
| Co-crystals | 5UHB (rpoB+RIF), 5BS8 (gyrA+MFX) | Drug distance for rpoB, gyrA |

**Positive definition:** Residue appears in any WHO-catalog resistance mutation for that gene. **Not** mutation-level labels at training — residue-level hotspot propensity only.

---

### 2.3 Feature engineering (16 Stage-2 features)

#### 2.3.1 Sequence and population (Stage 0 — `04b`)
- Homoplasy count and allele diversity per codon (k-mer voting gene alignment across 1,037 genomes).
- Biophysical: helix/strand propensity, hydrophobicity, volume, charge, H-bond capacity.
- `inner_distance`: sequence distance to known binding core.
- BLOSUM62 self-score, contact density, relative gene position.

#### 2.3.2 Structural (Stage 1 — `04c`)
- **SASA** (relative solvent accessibility from AlphaFold).
- **ESM-2 intolerance** (`facebook/esm2_t33_650M_UR50D`): pseudo-log-likelihood drop on mutation — **feature only**, not the classifier (ESM-2 alone AUROC 0.618).
- **3D contact density** and **pLDDT** (local confidence environment).

#### 2.3.3 Drug proximity (Stage 2 — `04d`)
- Per-residue minimum atom distance to dilated drug pocket (10 Å expansion from curated pocket residues).
- **Self-exclusion:** query residue excluded from pocket set (fixes circularity: AUROC 0.990 → 0.971 ablation).
- Transform: `drug_proximity = 1 / (1 + drug_distance / 10)`.

**Stage progression (ablation):**

| Stage | Model | AUROC | AUPRC | Top-20 recall |
|-------|-------|-------|-------|---------------|
| 0 | Logistic regression | 0.888 | — | — |
| 1 | + structural | 0.906 | 0.205 | 0.386 |
| 2 | XGBoost + drug + Platt | **0.971** | **0.560** | **0.657** |

---

### 2.4 Hotspot classifier (Stage 2)

**Algorithm:** XGBoost (`max_depth=6`, `learning_rate=0.05`, `n_estimators=300`, `scale_pos_weight=10`, `subsample=0.8`, `colsample_bytree=0.8`).

**Calibration:** Platt scaling via `CalibratedClassifierCV(method='sigmoid', cv=5)` on out-of-fold predictions.

**Output:** `ranked_predictions.csv` — all 6,350 residues ranked by `hotspot_score`.

**Top features (XGBoost gain):** homoplasy_alleles **0.269**, drug_proximity 0.158, homoplasy_count 0.149.

---

### 2.5 Mutation enumeration and emergence scoring (`04e`)

1. Select **top 50 residues** by `hotspot_score` + force-include all WHO hotspots.
2. For each codon, enumerate all **SNV-accessible** amino acid changes from H37Rv CDS (transitions/transversions; exclude stops and synonymous).
3. Per-mutation features: BLOSUM62, charge/size/hydrophobicity change, H-bond loss, transition flag, `drug_distance`, `inner_distance`.
4. **Resistance score** = pocket proximity × biochemical disruptiveness.
5. **Mutation score** = 0.45×resistance + 0.30×fitness + 0.25×accessibility (min-max normalized per batch).
6. **Emergence score** = `hotspot_score × mutation_score`.

**Output:** `emergence_watchlist.csv` (~305–332 mutations ranked).

---

### 2.6 Cross-validation and evaluation protocol

#### 2.6.1 Why precision–recall is mandatory
At **0.50% prevalence** (32/6,350), accuracy and F1@0.5 are misleading (6,318 negatives). **AUPRC, precision at fixed recall, and top-K recall** are primary metrics. Random baseline: **AUPRC = 0.005**.

#### 2.6.2 CV schemes
- **Stratified 5-fold CV** (primary headline): `StratifiedKFold(n_splits=5)`, per-fold optimal F1 threshold.
- **GroupKFold by gene** (conservative): entire gene held out — tests cross-gene generalization.
- **Full-model ranking:** Calibrated XGBoost on all residues; rank all 32 positives.

#### 2.6.3 Precision–recall metrics (authoritative — include in Results tables and Figure 2C)

**Stratified 5-fold CV (primary):**

| Metric | Value |
|--------|-------|
| AUROC | **0.968 ± 0.034** |
| **AUPRC** | **0.465 ± 0.157** (92× random) |
| **Best F1** | **0.550 ± 0.119** |
| **Precision @ best F1** | **0.631** |
| **Recall @ best F1** | **0.562** |
| F1 @ threshold 0.5 | 0.384 ± 0.142 |
| **Precision @ threshold 0.5** | **0.667** |
| **Recall @ threshold 0.5** | **0.324** |
| Top-20 recall | **0.662** (21/32) |
| Top-50 / Top-100 recall | 0.829 / 0.857 |
| Per-fold AUPRC | 0.613 · 0.537 · 0.619 · 0.294 · 0.259 |

**Pooled OOF PR curve (Figure 2C, `fig_pr_curve.csv` — 501-point smooth envelope):**

| Metric | Value |
|--------|-------|
| AUPRC | **0.435** |
| Precision @ best F1 | 0.536 |
| Recall @ best F1 | 0.469 |
| **Precision @ recall ≥ 0.25** | **0.800** |
| **Precision @ recall ≥ 0.50** | **0.432** |
| Precision @ recall ≥ 0.75 | 0.140 |

**GroupKFold by gene:**

| Metric | Value |
|--------|-------|
| AUROC | **0.974 ± 0.018** |
| **AUPRC** | **0.586 ± 0.226** |
| Best F1 | **0.676 ± 0.191** |
| **Precision @ best F1** | **0.765** |
| **Recall @ best F1** | **0.668** |
| Top-20 recall | **0.741** |
| Per-fold AUPRC | 0.855 · 0.313 · 0.420 · 0.756 |

**Full-model ranking:**

| Metric | Value |
|--------|-------|
| All positives in top 32 | **32/32** |
| Top-20 recall | 20/32 |
| Score gap (last pos − first neg) | **0.401** |

**Controls:** Permutation test **p = 0.005**; ESM-2-only AUROC 0.618 (+0.35 lift).

---

### 2.7 CRyPTIC prospective validation (`08`, `09`)

**Tier assignment:**

| Tier | Definition | N |
|------|------------|---|
| 0 | WHO-known mutation | 30 |
| 1 | Watchlist mutation observed + FDR q < 0.05 | **24** |
| 2 | Observed, enriched, underpowered | 32 |
| 3 | Observed, no phenotype (pncA/rpsL blind spot) | 31 |
| 4 | **Forecast-only — 0 CRyPTIC carriers** | **188** |

**Matched-null test:** Tier 1 count 24 vs null mean 9.3 (1,000 gene/carrier-matched permutations), **p = 0.001**.

---

### 2.8 Structural validation (`06`, `07`)

**Filter:** Tier 4, gene ∈ {rpoB, gyrA, gyrB}, `drug_distance ≤ 4.5 Å` → 32 candidates.

**AutoDock Vina:** Fresh WT redock per gene; ΔΔG = binding_mut − binding_WT; pass if **ΔΔG ≥ +0.15 kcal/mol**.

**Categories:** STRONG ≥ 0.40 · MODERATE ≥ 0.15 · WEAK ≥ 0.05.

---

### 2.9 Statistical software and reproducibility

```bash
python scripts/13_final_publication_audit.py   # AUROC, AUPRC, precision, recall, PR/ROC CSVs
python scripts/10_generate_figures.py
python scripts/11_render_figures.py            # Figure 2C = smooth PR curve
python scripts/12_audit.py                     # ~180 automated checks
```

All metrics: `analysis/results/publication_metrics.json`, `PUBLICATION_METRICS.md`.

---

## 3. Results (TB proof-of-concept)

### 3.1 Model performance — precision–recall headline (Figure 2)

**Figure 2** — Stage ablation table + ROC + **precision–recall curve** (`Figure_2.png`).

**Report in main text (not supplement):**

> At 0.5% prevalence, EFM achieves **AUPRC 0.465 ± 0.157** (92× random baseline 0.005), with **precision 0.631 and recall 0.562** at the per-fold optimal F1 threshold. At recall ≥ 0.50, precision remains **0.432** (~86× random). GroupKFold by gene yields **AUPRC 0.586** with **precision 0.765 / recall 0.668** at best F1.

All 32 WHO hotspot residues occupy ranks 1–32 (score gap 0.40).

### 3.2 CRyPTIC validation (Figure 4)
- **24 Tier-1** mutations retrospectively confirmed (FDR q < 0.05); matched-null p = 0.001.
- **188 Tier-4** forecast-only mutations (0 carriers) — prospective surveillance list.
- Representative Tier 1: gyrA D94A (147 carriers, 59% R), rpoB D435G (rank 5), embB G406S (rank 32).

### 3.3 Structural validation and lead hit
- **10/32** Tier-4 pocket-direct mutations pass Vina ΔΔG ≥ 0.15.
- **gyrB Q538L** (rank 131, emergence 0.234): ΔΔG **+0.737**, 1.34 Å from moxifloxacin, **literature-novel** (Q538L not reported in Mtb; N538D/K/S/T only).
- Other nine Vina hits = pipeline benchmarks (known CARD variants).

**Structural figure:** `data/pdb/gyrB_Q538L_validation.png`

### 3.4 Feature ablation insight
Homoplasy alleles (gain 0.269) dominate over drug proximity (0.158) — selective pressure signal is necessary but structure/mechanism refine ranking.

---

## 4. Discussion

### 4.1 EFM as a general framework (design intent only)
The staged architecture (population → structure → mechanism → calibrated trees) is **not TB-specific in design**, but **only TB is implemented and validated in this work**.

### 4.2 Why precision–recall must lead the narrative
At 0.5% prevalence, F1@0.5 = 0.38 despite excellent ranking (32/32 in top 32). Reviewers will ask about "low F1" — answer with **AUPRC 0.465**, **P = 0.80 @ R ≥ 0.25**, and top-K recall. Figure 2C is the primary classifier figure.

### 4.3 Limitations (this work)
- TB only: 32 positive residues; homoplasy not recomputed per CV fold.
- Vina: rigid receptor; misses allosteric resistance.
- **MRSA and Alzheimer's not yet attempted** — see §5 for planned extensions.

---

## 5. Next steps *(not implemented — do not report as results)*

Everything in this section is **future work**. None of it has been run in this repository.

### 5.1 Phase 2 — Phenotypic validation (TB)
- *M. smegmatis* surrogate + broth microdilution MIC for **gyrB Q538L** vs moxifloxacin.

### 5.2 Phase 3 — MRSA extension *(planned)*
Apply the same EFM pipeline to *S. aureus*:
- **Genes:** *mecA*, *femAB*, *rpoB*, *gyrA*, *grlA/B*, *dfrB*, *fusA*
- **Labels:** CARD + EUCAST resistance catalog
- **Homoplasy:** Public *S. aureus* WGS (PATRIC, NCBI Pathogen Detection)
- **Validation:** AST-matched WGS cohorts (e.g. CDC NHSN); same tier + matched-null logic as CRyPTIC

### 5.3 Phase 4 — Alzheimer's extension *(planned)*
Apply EFM to AD-relevant genes:
- **Genes:** APP, PSEN1, PSEN2, BACE1, MAPT
- **Labels:** ClinVar pathogenic/likely-pathogenic missense residues
- **Population signal:** gnomAD allele recurrence; ADSP/ADNI enrichment
- **Validation:** Temporal holdout of newly deposited ClinVar variants; matched-null vs gnomAD

### 5.4 Phase 5 — Clinical deployment (Mantis)
Integrate Tier-4 surveillance alerts into WGS interpretation workflows.

### 5.5 Manuscript framing
- **Report:** EFM model + TB benchmark + PR metrics + Q538L + CRyPTIC tiers.
- **Do not report:** MRSA or Alzheimer's metrics, figures, or validation — only list as §5 next steps.

---

## 6. Figures

| ID | File | Content |
|----|------|---------|
| **Fig 1** | `Figure_1.png` | EFM pipeline (general) + TB instantiation |
| **Fig 2** | `Figure_2.png` | Stage ablation + ROC + **precision–recall (AUPRC 0.435 OOF curve)** |
| **Fig 3** | `Figure_3.png` | Feature importance |
| **Fig 4** | `Figure_4.png` | CRyPTIC tiers + Tier 1 table |
| **Fig 5** | `gyrB_Q538L_validation.png` | Lead structural validation |
| **Fig S2** | `Figure_S2.png` | Leave-one-gene-out |

**Extended Data (recommended):** Full PR metric table (§2.6.3), all 10 Vina hits.

---

## 7. Target venues

1. ***Nature Communications* / *Nature Microbiology*** — EFM methodology + PR evaluation + TB benchmark (Q538L, CRyPTIC).
2. ***Bioinformatics* / *PLOS Computational Biology*** — Methods-first with full §2 reproduction.

---

## Appendix: Quick-reference PR scores for every section

| Section | Precision | Recall | AUPRC | F1 |
|---------|-----------|--------|-------|-----|
| Abstract / primary CV | **0.631** | **0.562** | **0.465 ± 0.157** | **0.550 ± 0.119** |
| Pooled OOF (Fig 2C) | 0.536 @ best F1 | 0.469 | **0.435** | 0.500 |
| P @ R ≥ 0.25 / 0.50 | **0.800** / **0.432** | — | — | — |
| GroupKFold | **0.765** | **0.668** | **0.586 ± 0.226** | **0.676 ± 0.191** |
| F1 @ threshold 0.5 | 0.667 | 0.324 | — | 0.384 ± 0.142 |
