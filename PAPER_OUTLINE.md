# Paper Outline — Emergence Forecasting Model for Protein Hotspots

**Working title:** *A general framework for forecasting emergent pathogenic mutations: residue-level hotspot modeling validated in tuberculosis, with extensions to MRSA and Alzheimer's disease*

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

**Tuberculosis** is the **proof-of-concept benchmark** (12,287 CRyPTIC isolates, 1,037 homoplasy genomes, Vina structural validation, lead hit **gyrB Q538L**).

**MRSA** and **Alzheimer's disease** are **direct transfer targets** — same math, different genes, labels, and validation cohorts.

---

## Abstract (~280 words)

**Background.** Genomic medicine is retrospective: we annotate variants after they appear in ClinVar, CRyPTIC, or gnomAD. For antimicrobial resistance and neurodegeneration, the clinically urgent question is different: **which mutations have not yet been seen but are likely to emerge next?**

**Methods.** We introduce an **Emergence Forecasting Model (EFM)** — a staged, residue-level classifier that integrates (i) population homoplasy from 1,037 genomes, (ii) AlphaFold structural features, (iii) ESM-2 mutation intolerance, and (iv) self-excluded drug-binding proximity, trained with XGBoost and Platt calibration on 6,350 residues across 13 *M. tuberculosis* resistance genes (32 positive hotspots, 0.50% prevalence). Accessible SNVs are enumerated at high-risk codons; each receives P(emergence). We evaluate with stratified 5-fold CV, GroupKFold by gene, and **precision–recall analysis** at 0.5% prevalence (random baseline AUPRC = 0.005). TB predictions are validated against 12,287 CRyPTIC isolates (tiered FDR framework) and AutoDock Vina (Tier-4 pocket-direct, fresh WT redock).

**Results (TB benchmark).** Primary CV: AUROC **0.968 ± 0.034**, **AUPRC 0.465 ± 0.157** (92× random), **precision 0.631 / recall 0.562** at optimal F1 (**F1 0.550 ± 0.119**). At fixed recall ≥ 0.50, precision **0.432** (~86× random). GroupKFold: AUPRC **0.586 ± 0.226**, precision **0.765 / recall 0.668** at best F1. All 32 known hotspots rank in the top 32. **24** predictions retrospectively confirmed in CRyPTIC (Tier 1, matched-null p = 0.001); **188** remain forecast-only (Tier 4). Lead de novo finding: **gyrB Q538L** (ΔΔG +0.737, literature-novel).

**Conclusions.** EFM provides a transferable blueprint for prospective mutation surveillance. We outline deployment to **MRSA** (mecA/fem/rpoB resistance genes) and **Alzheimer's** (APP, PSEN1, BACE1 pathogenic hotspot forecasting before ClinVar/gnomAD saturation).

**Keywords:** emergence forecasting, precision–recall, homoplasy, XGBoost, antimicrobial resistance, MRSA, Alzheimer's disease, prospective validation

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

TB drug resistance is the **first full instantiation**; the architecture is **pathogen- and disease-agnostic**.

### 1.3 Contributions
1. **Novel model:** Staged EFM (sequence → structure → drug proximity → XGBoost + Platt) with explicit P(emergence) decomposition and SNV enumeration.
2. **Rigorous PR evaluation:** Full precision–recall reporting at 0.5% prevalence — the operative metric regime for rare hotspot discovery.
3. **Prospective validation framework:** CRyPTIC tier system (Tiers 0–4) + matched-null enrichment + orthogonal Vina ΔΔG.
4. **TB lead discovery:** **gyrB Q538L** — pipeline-novel, literature-novel, structurally STRONG.
5. **Transfer roadmap:** Concrete gene lists and validation strategies for **MRSA** and **Alzheimer's**.

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

### 2.9 Transferability: applying EFM to MRSA

**Conceptual mapping:**

| EFM component | TB (this paper) | MRSA extension |
|---------------|-----------------|----------------|
| Gene set | 13 TB resistance genes | *mecA*, *femAB*, *rpoB*, *gyrA*, *grlA/B*, *dfrB*, *fusA* |
| Hotspot labels | WHO TB catalog | CARD + EUCAST + SCC*mec* literature |
| Homoplasy | 1,037 TB genomes | Public *S. aureus* collections (e.g. PATRIC, NCBI Pathogen Detection) |
| Structure | AlphaFold + co-crystals | AlphaFold + PDB (mecA/penicillin, quinolone-DNA gyrase) |
| Ligand proximity | RIF, MFX, INH-NAD, EMB | Oxacillin, moxifloxacin, vancomycin (where applicable) |
| Clinical validation | CRyPTIC 12,287 | CDC NHSN / public WGS + AST matched cohorts |
| Forecast tier | Tier 4 (0 carriers) | Mutations absent from surveillance but ranked by EFM |

**MRSA-specific value:** mecA promoter and femAB pathway mutations emerge under β-lactam pressure before appearing in local antibiograms — EFM ranks **accessible SNVs at hotspot codons** for hospital surveillance dashboards (Phase 4: Mantis platform).

**Expected PR regime:** MRSA resistance hotspots are similarly rare at residue resolution; **AUPRC and precision @ recall ≥ 0.25** remain the operative metrics (not accuracy).

---

### 2.10 Transferability: applying EFM to Alzheimer's disease

**Conceptual mapping:**

| EFM component | TB (this paper) | Alzheimer's extension |
|---------------|-----------------|----------------------|
| Gene set | TB resistance genes | **APP**, **PSEN1**, **PSEN2**, **BACE1**, **MAPT** (tau) |
| Hotspot labels | WHO resistance residues | ClinVar pathogenic/likely-pathogenic **missense** at known AD loci |
| Homoplasy / population signal | TB homoplasy | gnomAD v4 allele recurrence; ADSP/ADNI carrier enrichment |
| Structure | AlphaFold | AlphaFold + cryo-EM (γ-secretase, BACE1) |
| Ligand proximity | Anti-TB drugs | γ-secretase modulators, BACE inhibitors (mechanistic pocket) |
| Clinical validation | CRyPTIC phenotypes | Longitudinal cohorts: variants that **appear** in AD cases after model forecast |
| Forecast tier | Tier 4 (0 carriers) | Variants **absent from ClinVar** but high P(emergence) at APP/PSEN1 cleavage sites |

**Alzheimer's-specific framing:** AD genetics is dominated by **known familial mutations** (APP Swedish, PSEN1 E280A). EFM asks: **which accessible substitutions at structurally critical APP/PSEN1/BACE1 residues have not yet been reported but plausibly will?** — analogous to forecasting Q538L before CRyPTIC detection.

**Validation strategy:**
1. Train on ClinVar pathogenic residues (exclude VUS).
2. Hold out recently deposited pathogenic variants (temporal split).
3. Test enrichment in AD case-control WGS vs matched-null (same matched-null logic as CRyPTIC Tier 1).

**Caveat for paper:** Alzheimer's transfer is **proposed** — TB results are **demonstrated**. Frame AD as Discussion + Future Work with explicit gene list and validation design.

---

### 2.11 Statistical software and reproducibility

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

### 4.1 EFM as a general framework
The staged architecture (population → structure → mechanism → calibrated trees) is **not TB-specific**. Any domain with (i) sparse hotspot labels, (ii) population recurrence data, (iii) protein structure, and (iv) an independent validation cohort can run the same pipeline.

### 4.2 Why precision–recall must lead the narrative
At 0.5% prevalence, F1@0.5 = 0.38 despite excellent ranking (32/32 in top 32). Reviewers will ask about "low F1" — answer with **AUPRC 0.465**, **P = 0.80 @ R ≥ 0.25**, and top-K recall. Figure 2C is the primary classifier figure.

### 4.3 MRSA deployment path
- Near-term: swap gene table + CARD labels + *S. aureus* homoplasy; validate against public AST-matched WGS.
- Clinical hook: pre-emptive alerts for ranked Tier-4 *mecA*/*fem*/*rpoB* SNVs in hospital WGS pipelines.

### 4.4 Alzheimer's deployment path
- Train on APP/PSEN1/BACE1 ClinVar pathogenic residues; gnomAD as carrier matrix.
- Forecast accessible substitutions at γ-secretase cleavage sites and BACE1 catalytic pocket before ClinVar saturation.
- Longitudinal validation: variants that emerge in ADSP cases after model publication.

### 4.5 Limitations
- TB demo: 32 positive residues; homoplasy not recomputed per CV fold.
- Vina: rigid receptor; misses allosteric resistance.
- MRSA/AD: proposed extensions, not yet run in this repository.

### 4.6 Roadmap
1. **Manuscript** — Lead with EFM methodology + PR metrics; TB as benchmark; Q538L as de novo case study.
2. **Phase 2** — *M. smegmatis* MIC for Q538L (phenotypic proof).
3. **Phase 3** — MRSA EFM instantiation.
4. **Phase 4** — Alzheimer's gene set + gnomAD validation.
5. **Phase 5** — Mantis clinical WGS integration.

---

## 5. Figures

| ID | File | Content |
|----|------|---------|
| **Fig 1** | `Figure_1.png` | EFM pipeline (general) + TB instantiation |
| **Fig 2** | `Figure_2.png` | Stage ablation + ROC + **precision–recall (AUPRC 0.435 OOF curve)** |
| **Fig 3** | `Figure_3.png` | Feature importance |
| **Fig 4** | `Figure_4.png` | CRyPTIC tiers + Tier 1 table |
| **Fig 5** | `gyrB_Q538L_validation.png` | Lead structural validation |
| **Fig S2** | `Figure_S2.png` | Leave-one-gene-out |

**Extended Data (recommended):** Full PR metric table (§2.6.3), all 10 Vina hits, MRSA/AD gene mapping tables.

---

## 6. Target venues

1. ***Nature Communications* / *Nature Methods*** — General EFM framework + PR evaluation + cross-disease roadmap; TB as benchmark.
2. ***Nature Microbiology*** — If emphasizing Q538L + CRyPTIC clinical story.
3. ***Bioinformatics* / *PLOS Computational Biology*** — Methods-first with full §2 reproduction.

---

## Appendix: Quick-reference PR scores for every section

| Section | Precision | Recall | AUPRC | F1 |
|---------|-----------|--------|-------|-----|
| Abstract / primary CV | **0.631** | **0.562** | **0.465 ± 0.157** | **0.550 ± 0.119** |
| Pooled OOF (Fig 2C) | 0.536 @ best F1 | 0.469 | **0.435** | 0.500 |
| P @ R ≥ 0.25 / 0.50 | **0.800** / **0.432** | — | — | — |
| GroupKFold | **0.765** | **0.668** | **0.586 ± 0.226** | **0.676 ± 0.191** |
| F1 @ threshold 0.5 | 0.667 | 0.324 | — | 0.384 ± 0.142 |
