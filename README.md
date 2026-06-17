# Forecasting Emerging Tuberculosis Drug Resistance

Aly Dhedhi, Pranava Kumar, Vinay Singamsetty, Li-Lun Ho, Manolis Kellis
Kellis Lab, MIT

---

## What this project does

This pipeline predicts which M. tuberculosis drug resistance mutations are likely to emerge next -- before they appear in clinical resistance catalogs. It works in three phases:

1. Learn structural signatures of known resistance hotspots across 13 TB resistance genes
2. Enumerate all single-nucleotide accessible mutations at the most hotspot-like residues
3. Score mutations by P(emergence) = P(hotspot) x fitness x accessibility, then prospectively validate against 12,287 independent CRyPTIC clinical isolates

---

## Model performance

Dataset: 6,326 residues across 12 genes. 32 positive (hotspot) residues (0.51% of all residues). 16 features: homoplasy, sequence properties, structural (AlphaFold), and drug proximity.

Metrics are from 5-fold stratified cross-validation. The class imbalance is extreme (1 positive per 197 negatives), which affects threshold-dependent metrics.

| Metric | Value | 95% CI | Notes |
|--------|-------|--------|-------|
| AUROC | 0.917 | [0.874, 0.963] | Threshold-independent ranking |
| AUPRC | 0.186 (37x random) | [0.140, 0.476] | Precision-recall area |
| F1 (Youden threshold) | 0.063 | [0.042, 0.088] | At optimal threshold (0.0009) |
| Precision | 0.033 | [0.022, 0.046] | TP / (TP + FP) at threshold |
| Recall | 0.844 | [0.710, 0.957] | TP / (TP + FN) at threshold |
| Specificity | 0.873 | — | TN / (TN + FP) |
| MCC | 0.151 | — | Balanced correlation coefficient |
| Permutation test | p = 0.005 | — | 0/200 shuffled labels beat real AUROC |
| Top-20 recall | 0.219 (7/32) | — | Hotspot residues in top 20 |
| Top-50 recall | 0.375 (12/32) | — | Hotspot residues in top 50 |

ESM-2 alone (baseline): AUROC 0.618 (near random). Full model lift: +0.299 AUROC (+48%).

At the optimal threshold (Youden's J = 0.72), the model identifies 27/32 known hotspot residues (84% recall) with 801 false positives across 6,294 negatives. The low precision is expected: in a 0.51% positive rate problem with 32 known positives, even with AUROC 0.917, most high-scoring candidates will be novel predictions rather than already-known hotspots -- which is the intended application.

### Why F1 is low but the model is useful

A model predicting 1-in-200 events will have low F1 at any reasonable threshold. The relevant metric is: do the top-ranked predictions enrich for real resistance? In CRyPTIC validation, 19 of the top-ranked novel predictions reach FDR significance (q < 0.05), and 36/51 (71%) with phenotype data show resistance enrichment.

---

## Prospective validation in CRyPTIC (12,287 clinical isolates)

| Tier | Count | Meaning |
|------|-------|---------|
| 0 | 30 | Known WHO mutations (pipeline sanity check) |
| 1 | 19 | FDR-significant novel predictions (q < 0.05) |
| 2 | 33 | Observed, enriched in resistant isolates (underpowered) |
| 3 | 44 | Observed, no phenotype data (pncA blind spot) |
| 4 | 190 | Forecast-only (prospective surveillance targets) |

Matched-null validation: Tier 1 enrichment was tested against 1,000 random mutation sets matched by gene and carrier count. The real Tier 1 count (19) significantly exceeds the null distribution (p < 0.05), confirming that enrichment is not driven by gene-level confounders or detection bias.

### Top validated novel predictions

| Mutation | Gene | Rank | Carriers | R% | FDR p |
|----------|------|------|----------|----|-------|
| D94A | gyrA | 93 | 147 | 59% | 8.8e-36 |
| G406S | embB | 6 | 99 | 75% | 8.5e-20 |
| Q497K | embB | 121 | 71 | 84% | 1.4e-20 |
| I21T | inhA | 27 | 64 | 98% | 5.3e-18 |
| I194T | inhA | 21 | 64 | 97% | 1.7e-16 |
| D435G | rpoB | 41 | 61 | 90% | 2.2e-16 |
| H445R | rpoB | 31 | 33 | 97% | 8.5e-12 |
| Q10R | pncA | 11 | 155 | No data | Blind spot |

---

## Pipeline: stage by stage

### Stage 0: Sequence-only model
`04b_hotspot_model.py` -- 12 features from amino acid biochemistry (hydrophobicity, charge, volume, helix propensity, etc.) plus homoplasy counts from ~100 TB genomes. AUROC = 0.888.

### Stage 1: Structural features
`04c_stage1_features.py` -- Adds SASA (solvent accessibility), ESM-2 mutation intolerance, and 3D contact density from AlphaFold structures. AUROC = 0.910.

### Stage 2: Drug proximity + XGBoost + calibration
`04d_docking_features.py` -- Adds per-residue distance to drug-binding pocket (co-crystal or dilated pocket proxy, 10A radius, self-excluded). Replaces logistic regression with XGBoost (scale_pos_weight=10, max_depth=6, lr=0.05, 300 trees). Platt calibration via CalibratedClassifierCV.

AUROC = 0.917 [0.874, 0.963], AUPRC = 0.186 (37x random).

Feature importance (XGBoost gain): homoplasy_count (0.27), drug_proximity (0.10), inner_distance (0.09), hydrophobicity (0.05), homoplasy_alleles (0.05). The remaining 11 features account for 0.44 combined.

GroupKFold by gene is provided as a secondary evaluation (more conservative: trains on 4/5 of genes, tests on held-out genes).

### Mutation forecasting
`04e_mutation_forecasting.py` -- For the top hotspot-scoring residues, enumerate all SNV-accessible mutations. Score by: emergence = hotspot_score x mutation_score, where mutation_score combines resistance plausibility, fitness cost, and evolutionary accessibility.

Known resistance mutations in top-20: 4/33. Top-50: 13/33. Top-100: 19/33.

Clinical watchlists: `watchlist_top20.csv`, `watchlist_top50.csv`.

### Leave-one-gene-out validation
`05_leave_one_gene_out.py` -- Holds out each entire resistance gene, trains on the rest (logistic regression, which generalizes better than XGBoost for unseen genes). Aggregate top-50 recall: ~50%. 

### CRyPTIC cross-reference
`08_cryptic_validation_full.py` -- Stream-filters a 1.5 GB mutation matrix from 12,287 isolates against the watchlist. Assigns tiers 0-4. Computes matched-null validation.

### Feature ablation
`04f_evolutionary_features.py` -- Shannon entropy from MSA is blocked by API restrictions (UniProt BLAST returns 404; no local MAFFT/BLAST+). The BLOSUM-based mutation_sensitivity approximation had 2 unique values (99.7% constant) and was removed.

---

## Circularity bug and fix

The original model reported AUROC 0.990. The inflation came from:

1. Pocket overlap: training positives were inside the pocket definition. A hotspot residue's distance to itself was zero, giving drug_proximity=1.0 for 24/27 positives.
2. Fix: dilute pockets to 10A radius in 3D structure and exclude the query residue from distance computation (self-exclusion).

After the fix, drug_proximity dropped from the dominant feature (+6.71 in LR) to 4th importance (0.099). Homoplasy-based features now dominate.

---

## Data sources

| Data | Size | Location |
|------|------|----------|
| TB genomes (~100) | — | `data/demo/drprg_sparse.vcf.gz` |
| Resistance genes | 13 proteins, ~6,300 residues | `data/reference/H37Rv.{gff,fasta}` |
| Known hotspots | 32 positives, tracked in `positives_whitelisted.csv` | WHO catalog + CRyPTIC Tier 1-2 |
| CRyPTIC isolates | 12,287 | `data/cryptic/MUTATIONS.csv.gz` (1.5 GB) |
| CRyPTIC phenotypes | 12,287 x 13 drugs | `data/metadata/cryptic_phenotypes.csv` |
| Mycobacterial genomes | 10 NCBI RefSeq | `data/genomes/GCF_*.fasta` |
| AlphaFold structures | 13 proteins | `data/pdb/alphafold/` |
| Co-crystal structures | rpoB (5UHB), gyrA (5BS8) | `data/pdb/crystal/` |

---

## Project structure

```
tb-resistance-discovery/
scripts/               -- 17 pipeline scripts (numbered by dependency)
 04b_hotspot_model.py      Stage 0: sequence model
 04c_stage1_features.py    Stage 1: structural features
 04d_docking_features.py   Stage 2: XGBoost + drug proximity + Platt calibration
 04e_mutation_forecasting.py  P(emergence) scoring and watchlist generation
 05_leave_one_gene_out.py  Cross-gene generalization
 08_cryptic_validation_full.py  CRyPTIC cross-reference + matched-null validation
 12_audit.py               Full audit (~180 checks)
analysis/results/
 hotspot_model/           Model outputs (ranked predictions, metrics, feature importance)
 forecasting/             Watchlist, CRyPTIC validation, clinical top-20/50
 figures/                 9 publication figures
data/
 cryptic/                CRyPTIC mutation table (1.5 GB)
 genomes/                10 mycobacterial genomes
 metadata/               Clinical phenotypes
 pdb/                    AlphaFold + co-crystal structures
```

---

## Reproducibility

```bash
python scripts/04b_hotspot_model.py
python scripts/04c_stage1_features.py
python scripts/04d_docking_features.py
python scripts/04e_mutation_forecasting.py
python scripts/05_leave_one_gene_out.py
python scripts/08_cryptic_validation_full.py
python scripts/09_stress_tests.py
python scripts/10_generate_figures.py
python scripts/11_render_figures.py
python scripts/12_audit.py     # ~180 automated checks
```

The `12_audit.py` script checks: file existence, syntax, CRyPTIC data integrity, model output schema, figure completeness, package availability, claim consistency, leakage (homoplasy globality, drug_proximity self-exclusion, scaler placement, calibration, GroupKFold), and statistical rigor (permutation test, bootstrap CIs, ESM-2 baseline, matched-null validation).

---

## Limitations

1. Small positive set. 32 known positives in 6,326 residues (0.51%). Limits statistical power and makes threshold-dependent metrics (F1, precision) appear weak even though the model is ranking novel predictions correctly.

2. Phenotype blind spots. pncA (pyrazinamide) has no binary phenotype in CRyPTIC. Mutation Q10R (rank #11, 155 carriers) cannot be validated. 44 Tier 3 mutations are stuck here.

3. Homoplasy computed globally. homoplasy_count is computed from the full VCF, not per CV fold. This is acceptable because labels come from WHO/CRyPTIC catalogs, not from the same genomes. Documented as a known limitation in the self-audit.

4. Platt calibration on full data. CalibratedClassifierCV uses 5-fold internal CV, but the final model applies to all data. Emergence scores are well-calibrated but not exact probabilities.

5. No comparable benchmark. Existing work focuses on genotype-to-phenotype prediction (CRyPTIC ML) or retrospective curation (WHO catalog), not prospective emergence forecasting.

6. External APIs blocked. UniProt BLAST and NCBI Entrez are not accessible from this environment, preventing MSA-based Shannon entropy computation. 10 mycobacterial genomes are available locally if alignment tools are installed.

---

## References

Walker TM, Miotto P, Köser CU, et al. (2022). The 2021 WHO catalogue of Mycobacterium tuberculosis complex mutations associated with drug resistance: a genotypic analysis. *The Lancet Microbe*, 3(4): e265-e273. DOI: 10.1016/S2666-5247(21)00301-3

The CRyPTIC Consortium (2022). A data compendium of Mycobacterium tuberculosis antibiotic resistance. *bioRxiv*. DOI: 10.1101/2022.08.09.503396

Jumper J, Evans R, Pritzel A, et al. (2021). Highly accurate protein structure prediction with AlphaFold. *Nature*, 596: 583-589. DOI: 10.1038/s41586-021-03819-2

Rives A, Meier J, Sercu T, et al. (2021). Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. *PNAS*, 118(15): e2016239118. DOI: 10.1073/pnas.2016239118

Hunt ML, Murrell B, et al. (2025). Deep mutational scanning of M. tuberculosis rifampicin resistance. *bioRxiv*. DOI: 10.1101/2025.02.10.637030

Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society B*, 57(1): 289-300.

Youden WJ (1950). Index for rating diagnostic tests. *Cancer*, 3(1): 32-35.

Pedregosa F, Varoquaux G, Gramfort A, et al. (2011). Scikit-learn: machine learning in Python. *Journal of Machine Learning Research*, 12: 2825-2830.

Chen T, Guestrin C (2016). XGBoost: a scalable tree boosting system. *KDD '16*, 785-794. DOI: 10.1145/2939672.2939785

Cochrane G, Karsch-Mizrachi I, Takagi T (2016). The International Nucleotide Sequence Database Collaboration. *Nucleic Acids Research*, 44(D1): D48-D50.
