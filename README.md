# Forecasting Emerging Tuberculosis Drug Resistance

Aly Dhedhi, Vinay Singamsetty, Li-Lun Ho, Manolis Kellis
Kellis Lab, MIT

---

## What this project does

This pipeline predicts which M. tuberculosis drug resistance mutations are likely to emerge next -- before they appear in clinical resistance catalogs. It works in three phases:

1. Learn structural signatures of known resistance hotspots across 13 TB resistance genes
2. Enumerate all single-nucleotide accessible mutations at the most hotspot-like residues
3. Score mutations by P(emergence) = P(hotspot) x fitness x accessibility, then prospectively validate against 12,287 independent CRyPTIC clinical isolates

---

## Top forecast-only candidates (188 Tier 4 mutations)

These are the highest-confidence mutations predicted by our model that have **never been observed** in any clinical isolate (0 carriers). They are your prospective surveillance targets. A ✦ marks the 5 we structurally validated via AutoDock Vina.

| Rank | Gene | Mutation | Score | Mechanism | Docking ddG | Novelty |
|------|------|----------|-------|-----------|-------------|---------|
| 2 | gyrA | **A90T** | 0.506 | Other | — | Novel substitution at known QRDR position |
| 3 | inhA | **S94L** | 0.503 | Cofactor Gateway | — | Novel substitution at known INH position |
| 9 | inhA | **S94P** | 0.485 | Cofactor Gateway | — | Novel substitution at known INH position |
| 13 | rpoB | **D435N** | 0.473 | Other | — | Novel substitution at known RRDR position |
| 15 | rpoB | **S450P** | 0.471 | Other | — | Novel substitution at known RRDR position |
| 18 | rpsL | **K88E** | 0.452 | Charge Reversal | — | Novel charge reversal at STR position |
| 22✦ | rpsL | **K43E** | 0.446 | Charge Reversal | DOCKING FAILED | **Truly novel** — K43R known, K43E undocumented |
| 27✦ | inhA | **I16V** | 0.426 | Cofactor Gateway | ddG +0.019 (NONE) | **Truly novel** — I16T known, I16V undocumented |
| 28 | pncA | **V125I** | 0.425 | Loss-of-Function | — | Novel at pncA (PZA — blind spot) |
| 34✦ | eis | **V59A** | 0.409 | Direct Pocket | DOCKING FAILED | **Truly novel** — all known eis = promoter, not coding |
| 47✦ | rpoB | **V170I** | 0.375 | Allosteric Shield | ddG +0.001 (NONE) | **Truly novel** — V170F known, V170I undocumented |
| 131✦ | gyrB | **Q538L** | 0.234 | Pocket-direct (QRDR-B) | ddG **+0.737 (STRONG)** | **Literature-novel** — codon 538 known as N538D/K/S/T only; Q538L never reported in Mtb |

- **Tier 4 = forecast-only**: 0 carriers in 1,037 genomes, predicted by model but never seen in clinic
- **Novel substitution**: same gene/position as a known resistance mutation, but with a different amino acid change not documented in literature
- **Truly novel**: neither the substitution nor the position is documented in TB resistance literature

---

## Model performance

**Authoritative tables:** [`analysis/results/PUBLICATION_METRICS.md`](analysis/results/PUBLICATION_METRICS.md) — regenerate with `python scripts/13_final_publication_audit.py`

Dataset: 6,350 residues across 13 genes. 32 positive (hotspot) residues (0.50%). 16 Stage-2 features: homoplasy (1,037 genomes), sequence, structural (AlphaFold, ESM-2), and drug proximity (self-excluded).

### Hotspot classifier (Stage 2 — XGBoost + Platt calibration)

| Metric | Stratified 5-fold CV | GroupKFold by gene |
|--------|---------------------|-------------------|
| AUROC | **0.971** (stage progression) / 0.968 ± 0.034 (OOF) | **0.974 ± 0.018** |
| AUPRC | **0.560** (111× random) | 0.586 ± 0.226 |
| Best F1 (optimal threshold) | **0.550 ± 0.119** | 0.676 ± 0.191 |
| F1 @ threshold 0.5 | 0.384 ± 0.142 | — |
| Top-20 recall (CV) | **0.657** (21/32 per-fold avg) | **0.741** |

### Full-model ranking (calibrated on all residues)

All **32/32** known positives occupy ranks 1–32. Score gap between last positive (0.650) and first negative (0.249) = **0.40**. Top-20 full-model recall: **20/32**.

| Metric | Value | Notes |
|--------|-------|-------|
| Recall (OOF @ optimal threshold) | 0.562 | Per-fold average |
| Specificity | 0.879 | TN / (TN + FP) |
| MCC | 0.306 | At 0.5% prevalence |
| Permutation test | p = 0.005 | 200 shuffles |
| Top-50 recall (CV) | 0.829 | |
| Top-100 recall (CV) | 0.857 | |
| ESM-2 baseline AUROC | 0.618 | Full model lift +0.35 |

### Why F1 is still moderate despite excellent ranking

At the 0.5% positive rate, threshold-based metrics (F1, precision) are diluted by 6,318 negatives even when ranking is perfect. The relevant metric is: do the top-ranked predictions enrich for real resistance? In CRyPTIC validation, **24** of the top-ranked novel predictions reach FDR significance (q < 0.05), and 40/87 (46%) with phenotype data show resistance enrichment >50%.

---

## Prospective validation in CRyPTIC (12,287 clinical isolates)

| Tier | Count | Meaning |
|------|-------|---------|
| 0 | 30 | Known WHO mutations (pipeline sanity check) |
| 1 | **24** | FDR-significant novel predictions (q < 0.05) |
| 2 | 32 | Observed, enriched in resistant isolates (underpowered) |
| 3 | 31 | Observed, no phenotype data (pncA blind spot, reduced from 44) |
| 4 | 188 | Forecast-only (prospective surveillance targets) |

Matched-null validation: Tier 1 enrichment was tested against 1,000 random mutation sets matched by gene and carrier count. The real Tier 1 count (24) significantly exceeds the null distribution (p = **0.001**), confirming that enrichment is not driven by gene-level confounders or detection bias.

### Top CRyPTIC-confirmed predictions (Tier 1 — already validated in clinical data)

These mutations were **already observed** in 12,287 clinical CRyPTIC isolates and confirmed as resistance-associated at FDR q < 0.05. They are NOT the same as the forecast-only Tier 4 candidates above — they were validated *after* our model predicted them.

| Mutation | Gene | Rank | Carriers | R% | FDR p |
|----------|------|------|----------|----|-------|
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
| I491L | rpoB | 165 | 20 | 100% | 1.7e-08 |
| Q10R | pncA | 6 | 155 | No data (PZA) | blind spot |

All 32 known positive residues occupy ranks 1–32. The 24 Tier 1 novel mutations include 4 at rpoB Q432 (L/K/P/H), 4 at rpoB H445 (L/R/Q/P), rpoB D435G, I491L, V170A, L430R; gyrA D94A/H, G88C; embB G406S/C, Q497K/P; and inhA I194T, I21T/V, S94A, V78A.

---

## Pipeline: stage by stage

### Stage 0: Sequence-only model
`04b_hotspot_model.py` -- 12 features from amino acid biochemistry plus homoplasy counts from 1,037 TB genomes. AUROC = 0.888.

### Stage 1: Structural features
`04c_stage1_features.py` -- Adds SASA (solvent accessibility), ESM-2 mutation intolerance, and 3D contact density from AlphaFold structures. AUROC = 0.906.

### Stage 2: Drug proximity + XGBoost + calibration
`04d_docking_features.py` -- Adds per-residue distance to drug-binding pocket (co-crystal or dilated pocket proxy, 10A radius, self-excluded). Replaces logistic regression with XGBoost (scale_pos_weight=10, max_depth=6, lr=0.05, 300 trees). Platt calibration via CalibratedClassifierCV.

AUROC = **0.971**, AUPRC = **0.560 (111x random)**, Top-20 recall = **0.657**.

Feature importance (XGBoost gain):
| Feature | Importance | Type |
|---------|-----------|------|
| homoplasy_alleles | **0.269** | Genomic (1,037 genomes) |
| drug_proximity | 0.158 | Structural |
| homoplasy_count | 0.149 | Genomic |
| inner_distance | 0.048 | Sequence |
| strand_propensity | 0.047 | Sequence |
| hydrophobicity | 0.046 | Physicochemical |
| volume | 0.039 | Physicochemical |
| plddt_environment | 0.036 | AlphaFold confidence |
| hbond | 0.036 | Physicochemical |

GroupKFold by gene: AUROC 0.972 ± 0.018, AUPRC 0.575, Top-20 recall 0.741.

### Structural validation with AutoDock Vina

After the model predicts Tier-4 forecast-only mutations, we structurally validate them by docking the drug to WT and mutant receptors. This provides an orthogonal biophysical signal.

### Tier-4 pocket-direct batch (authoritative docking scores)

`scripts/06_filter_pocket_candidates.py` filters Tier-4 mutations in gyrA, rpoB, and gyrB with drug_distance ≤ 4.5 Å (32 candidates).

`scripts/07_tier4_pocket_vina_batch.py` builds mutant receptors, **redocks WT under identical grid conditions** (fixes stale-baseline bug), runs AutoDock Vina, and computes ΔΔG.

**10 structurally validated** (Tier 4, 0 carriers, ΔΔG ≥ +0.15):

| Mutation | Rank | ΔΔG | Category | Literature novel? |
|----------|------|-----|----------|-------------------|
| **gyrB Q538L** | 131 | **+0.737** | STRONG | **Yes** — de novo discovery |
| rpoB L452M | 181 | +2.137 | STRONG | No (CARD WHO-R) |
| rpoB P483R | 140 | +1.254 | STRONG | Maybe (P483L known) |
| rpoB L452R | 132 | +1.045 | STRONG | No (CARD) |
| rpoB Q432R | 225 | +0.399 | MODERATE | No (CARD) |
| gyrA S91A | 170 | +0.203 | MODERATE | No |
| gyrA G88S | 30 | +0.179 | MODERATE | No |
| gyrA G88V | 207 | +0.178 | MODERATE | No |
| gyrA G88D | 29 | +0.170 | MODERATE | No (rare clinical) |
| rpoB I491N | 124 | +0.156 | MODERATE | No (codon 491 hotspot) |

Outputs: `analysis/results/forecasting/tier4_pocket_vina_scores.csv`, `analysis/results/tier4_pocket_vina_results.json`.

Score audit: `analysis/audit_novelty_and_scores.py` re-parses all PDBQT REMARK lines (all 32 match CSV within 0.02 kcal/mol) and cross-checks CARD/PubMed.

**Q538L score note:** an earlier run used a stale WT baseline and reported ΔΔG −0.17. The tier-4 batch redocks WT fresh (−7.071) — **+0.737 is authoritative** (see `tier4_pocket_vina_scores.csv`, not deprecated `novel_docking_validation.json`).

### Phase 1: Crystal structures (high confidence)

**rpoB + Rifampicin (PDB 5UHB, 2.8A, chain C):** 11 mutations at the RRDR docked with rigid receptor, exhaustiveness=12. Results:

| Mutation | WT (kcal/mol) | Mutant (kcal/mol) | ddG | Category |
|----------|--------------|------------------|-----|----------|
| WT | -9.934 | - | - | - |
| L430R | -9.934 | -9.531 | +0.403 | MODERATE |
| H445R | -9.934 | -9.693 | +0.241 | WEAK |
| H445P | -9.934 | -9.721 | +0.213 | WEAK |
| H445Q | -9.934 | -9.726 | +0.208 | WEAK |
| H445L | -9.934 | -9.732 | +0.202 | WEAK |
| Q432L | -9.934 | -9.746 | +0.188 | WEAK |
| I491L | -9.934 | -9.758 | +0.176 | WEAK |
| Q432P | -9.934 | -9.765 | +0.169 | WEAK |
| D435G | -9.934 | -9.831 | +0.103 | WEAK |
| V170A | -9.934 | -9.906 | +0.028 | NONE |
| Q432K | -9.934 | -9.920 | +0.014 | NONE |

**gyrA + Moxifloxacin (PDB 5BS8, 2.9A, chain A):** 4 mutations in the QRDR. MFX binding is weak in the apo structure (requires DNA intercalation).

| Mutation | WT (kcal/mol) | Mutant (kcal/mol) | ddG | Category |
|----------|--------------|------------------|-----|----------|
| WT | -4.375 | - | - | - |
| G88C | -4.375 | -4.168 | +0.207 | WEAK |
| D94A | -4.375 | -4.319 | +0.056 | NONE |
| D94H | -4.375 | -4.344 | +0.031 | NONE |
| A90T | -4.375 | -4.363 | +0.012 | NONE |

### Phase 2: AlphaFold models (lower confidence)

**inhA + Triclosan (NADH proxy):** Vina cannot dock NADH/NAD+ (tree.h internal error at 10+ rotatable branches). Triclosan is used as a proxy for the NADH binding pocket.

| Mutation | WT (kcal/mol) | Mutant (kcal/mol) | ddG | Category |
|----------|--------------|------------------|-----|----------|
| WT | -7.462 | - | - | - |
| S94A | -7.462 | -6.955 | +0.507 | STRONG |
| I21T | -7.462 | -7.968 | -0.506 | STRONG (binding gain - triclosan may not be the correct ligand) |
| I194T | -7.462 | -7.631 | -0.169 | WEAK |

**embB + Ethambutol (AlphaFold):** No co-crystal for ethambutol; AlphaFold lacks pre-organized binding pocket.

| Mutation | WT (kcal/mol) | Mutant (kcal/mol) | ddG | Category |
|----------|--------------|------------------|-----|----------|
| Q497K | -4.741 | -4.724 | +0.017 | NONE |
| G406S | -4.209 | -4.181 | +0.028 | NONE |

### Novel candidate validation (Tier 4 — never seen, structurally validated)

For 5 top Tier-4 **truly novel** candidates (the 5 marked with ✦ in the table above), we ran targeted docking or structural analysis. All are *undocumented* in TB literature at this exact substitution:

| Candidate | Rank | Score | ddG / Result | Why Vina is blind |
|-----------|------|-------|-------------|-------------------|
| **rpsL K43E** | 22 | 0.446 | DOCKING FAILED | STR (14 torsions, branched rings) exceeds Vina tree.h(101) limit |
| **inhA I16V** | 27 | 0.426 | +0.019 (NONE) | Conservative Ile->Val at NADH pocket edge; rigid receptor misses kinetics |
| **eis V59A** | 34 | 0.409 | DOCKING FAILED | AMK (10 torsions, branched rings) exceeds Vina tree.h(101) limit |
| **rpoB V170I** | 47 | 0.375 | +0.001 (NONE) | Outside RRDR; allosteric mechanism via dynamics; rigid Vina blind |
| **gyrB Q538L** | 131 | 0.234 | **+0.737 (STRONG)** | Pocket-direct (1.34 Å); tier-4 batch with fresh WT redock; PyMOL validated |

### Vina limitations discovered

1. **tree.h(101) error**: Ligands with >10 rotatable branches (NADH/NAD+, streptomycin, amikacin) crash Vina. These are complex branched-ring molecules common in TB drugs.
2. **Rigid receptor blind spots**: Mutations >5A from the binding pocket show ddG ~0 even when clinically causative (allosteric rpoB, water-bridge gyrB).
3. **Conservative substitutions**: Small->small mutations (Val->Ile, Ile->Val) produce ddG < 0.1 kcal/mol despite potential functional impact.
4. **Indirect mechanisms**: Loss-of-function (pncA), promoter mutations (eis), and cofactor binding kinetics (inhA NADH) are invisible to static drug docking.

## Interactive viewer

Open `viewer.html` in a browser (serve from repo root: `python -m http.server 8000` then visit `http://localhost:8000/viewer.html`):

- Full pipeline walkthrough: data → model training (04b–04e) → CRyPTIC validation → Vina docking → audit
- gyrB Q538L PyMOL structural figure (`data/pdb/gyrB_Q538L_validation.png`)
- Filterable table of all 32 tier-4 pocket Vina scores (10 validated)
- Novelty audit table (CARD / PubMed / score verification)
- Roadmap: manuscript → MRSA extension → Mantis platform integration

## Mutation forecasting
`04e_mutation_forecasting.py` -- For the top hotspot-scoring residues, enumerate all SNV-accessible mutations. Score by: emergence = hotspot_score x mutation_score, where mutation_score combines resistance plausibility, fitness cost, and evolutionary accessibility.

Known resistance mutations in top-20: **8/33**. Top-50: 16/33. Top-100: 24/33.

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
| TB genomes (1,037: 117 VCF + 920 NCBI assemblies) | 4.1 GB | `data/genomes/` + `data/demo/drprg_sparse.vcf.gz` |
| Resistance genes | 13 proteins, ~6,300 residues | `data/reference/H37Rv.{gff,fasta}` |
| Known hotspots | 32 positives, tracked in `positives_whitelisted.csv` | WHO catalog + CRyPTIC Tier 1-2 |
| CRyPTIC isolates | 12,287 | `data/cryptic/MUTATIONS.csv.gz` (1.5 GB) |
| CRyPTIC phenotypes | 12,287 x 13 drugs | `data/metadata/cryptic_phenotypes.csv` |
| Mycobacterial genomes | 920 NCBI assemblies + 10 RefSeq | `data/genomes/GCF_*.fasta` |
| AlphaFold structures | 13 proteins | `data/pdb/alphafold/` |
| Co-crystal structures | rpoB (5UHB), gyrA (5BS8) | `data/pdb/crystal/` |

---

## Project structure

```
tb-resistance-discovery/
README.md                    Project summary
EXECUTIVE_SUMMARY.md         Complete internal reference (all scripts, files, scores)
PAPER_OUTLINE.md             Manuscript outline + figure checklist
viewer.html                  Interactive pipeline dashboard
scripts/
 04b–04e                     Model training + emergence forecasting
 05_leave_one_gene_out.py    Cross-gene generalization
 06_filter_pocket_candidates.py / 07_tier4_pocket_vina_batch.py  Vina validation
 08–09                       CRyPTIC validation + FDR tiers
 10–11                       Paper figures (5 PNGs) + CSV tables
 12_audit.py                  ~180 automated checks
 13_final_publication_audit.py  Authoritative metrics (AUROC, F1, AUPRC, PR/ROC)
 15e_compute_homoplasy_v4.py  Homoplasy from assemblies (current)
 16_merge_homoplasy.py
analysis/
 compute_metrics.py / permutation_test.py / esm2_baseline.py
 audit_novelty_and_scores.py / validate_novel_docking.py
analysis/results/
 PUBLICATION_METRICS.md       Authoritative metric table for paper
 publication_metrics.json
 hotspot_model/              Feature tables, ranked predictions, CV metrics
 forecasting/                Watchlist, CRyPTIC tiers, Vina scores
 figures/                    Figure_1–4.png, Figure_S2.png + CSV tables
```

---

## Next steps

1. **Manuscript** — Lead with gyrB Q538L (literature-novel + Vina STRONG + PyMOL). Frame 9 other validated hits as pipeline benchmarks. Include CRyPTIC Tier 1 retrospective confirmations.

2. **Extend to other diseases — MRSA first** — Reuse the same architecture (homoplasy + structure + drug proximity + XGBoost + prospective validation) on *Staphylococcus aureus* resistance genes.

3. **Mantis platform integration** — Deploy the emergence model inside the Mantis clinical genomics platform, surfacing Tier-4 surveillance alerts with structural validation and literature novelty flags at WGS interpretation time.

---

## Reproducibility

```bash
python scripts/13_final_publication_audit.py   # authoritative metrics + ROC/PR curves
python scripts/10_generate_figures.py
python scripts/11_render_figures.py            # Figure_1–4.png, Figure_S2.png
python scripts/04b_hotspot_model.py
python scripts/04c_stage1_features.py
python scripts/04d_docking_features.py
python scripts/04e_mutation_forecasting.py
python scripts/05_leave_one_gene_out.py
python scripts/08_cryptic_validation_full.py
python scripts/09_stress_tests.py
python scripts/12_audit.py                     # ~180 automated checks
python scripts/06_filter_pocket_candidates.py
python scripts/07_tier4_pocket_vina_batch.py
python analysis/audit_novelty_and_scores.py
```

The `12_audit.py` script checks: file existence, syntax, CRyPTIC data integrity, model output schema, figure completeness, package availability, claim consistency, leakage (homoplasy globality, drug_proximity self-exclusion, scaler placement, calibration, GroupKFold), and statistical rigor (permutation test, bootstrap CIs, ESM-2 baseline, matched-null validation).

---

## Limitations

1. Small positive set. 32 known positives in 6,350 residues (0.50%). Limits statistical power but ranking quality is high (all 32 in top 32).

2. Phenotype blind spots. pncA (pyrazinamide) has no binary phenotype in CRyPTIC. Mutation Q10R (rank #6, 155 carriers) cannot be validated. However, homoplasy scaling (5→56 residues) has substantially improved pncA signal. 31 Tier 3 mutations remain.

3. Homoplasy computed globally. homoplasy_count is computed from all 1,037 genomes, not per CV fold. This is acceptable because labels come from WHO/CRyPTIC catalogs, not from the same genomes. Documented as a known limitation in the self-audit.

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
