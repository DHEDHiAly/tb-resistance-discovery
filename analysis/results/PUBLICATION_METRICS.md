# Publication Metrics (Authoritative)

Generated: 2026-06-19T15:33:56.656085+00:00

## Dataset
- Residues: **6350** across 13 resistance genes
- Positive hotspots: **32** (0.50%)
- Stage 2 features: **16**

## Hotspot Model — Stratified 5-Fold CV (XGBoost, Stage 2)

| Metric | Value |
|--------|-------|
| AUROC | **0.968 ± 0.034** |
| AUPRC | **0.465 ± 0.157** (92× random) |
| Best F1 (per-fold optimal) | **0.550 ± 0.119** |
| Best F1 precision / recall | 0.631 / 0.562 |
| F1 @ threshold 0.5 | 0.384 ± 0.142 |
| Top-20 recall | **0.662** (21/32) |
| Top-50 recall | 0.829 |
| Top-100 recall | 0.857 |

## GroupKFold by Gene (conservative)

| Metric | Value |
|--------|-------|
| AUROC | **0.974 ± 0.018** |
| AUPRC | **0.586 ± 0.226** |
| Best F1 | 0.676 ± 0.191 |
| Top-20 recall | **0.741** |

## Stage Progression (AUROC)

| Stage | AUROC | AUPRC | Top-20 recall |
|-------|-------|-------|---------------|
| 0 (sequence) | 0.888 | — | — |
| 1 (structural) | 0.906 | 0.205 | 0.386 |
| 2 (XGBoost + drug) | **0.971** | **0.560** | **0.657** |

## CRyPTIC Prospective Validation (12,287 isolates)

- Tier 0 (WHO known): **30**
- Tier 1 (FDR q<0.05): **24**
- Tier 2 (Enriched): **32**
- Tier 3 (No phenotype): **31**
- Tier 4 (Forecast-only): **188**

## Full-Model Ranking (calibrated XGBoost on all residues)

- All 32 positives occupy ranks 1–32
- Top-20 recall: **20/32** (62.5%)
- Top-50 recall: **32/32**
- Top-100 recall: **32/32**
- Score gap (last positive − first negative): **0.4009**

## Matched-null validation
- Tier 1 count: **24** vs null mean 9.3 (p = 0.000999000999000999)

## Vina structural validation (Tier-4 pocket-direct)
- Candidates docked: **32**
- Structurally validated (ΔΔG ≥ 0.15): **10**

## Permutation test
- p = **0.004975124378109453**

## ESM-2 baseline
- ESM-2 only AUROC: **0.6178**
- Full model lift: **+0.3497** AUROC

---
*Source: `python scripts/13_final_publication_audit.py`*
