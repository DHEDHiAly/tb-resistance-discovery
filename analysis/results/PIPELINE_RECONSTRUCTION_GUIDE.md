# End-to-End Pipeline Reconstruction Guide

## How to Reproduce the Entire Model from Scratch

---

## Overview: What This Pipeline Does

Given 13 M. tuberculosis resistance genes (~6,600 residues total) and 21 known hotspot residues (positions where resistance mutations are documented), the pipeline:

1. **Trains a residue-level classifier** to distinguish hotspot vs non-hotspot using sequence + structural features (Stage 0 → Stage 1)
2. **Adds drug-contact information** from the rpoB-rifampicin co-crystal (Stage 1.5)
3. **Converts residue scores to mutation-level emergence probabilities** for all 315 SNV-accessible mutations at the top 50 hotspot residues
4. **Validates prospectively** against 12,287 independent CRyPTIC clinical isolates

---

## Step 0: Data Acquisition

### What you need

| Data | File | Size | Source |
|------|------|------|--------|
| H37Rv reference genome | `reference/H37Rv.fasta` | ~4.4 MB | NCBI: `https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF_000195955.2_ASM19595v2/GCF_000195955.2_ASM19595v2_genomic.fna.gz` |
| H37Rv gene annotations | `reference/H37Rv.gff` | ~1.5 MB | NCBI: same URL, `*_genomic.gff.gz` |
| CRyPTIC phenotypes | `data/metadata/cryptic_phenotypes.csv` | ~5 MB | See `scripts/download_cryptic_data.py` |
| CRyPTIC MUTATIONS table | `data/cryptic/MUTATIONS.csv.gz` | **1.4 GB** | Zenodo v1.1.1 or EBI FTP (see download script) |
| Demo VCF for homoplasy | `data/demo/drprg_sparse.vcf.gz` | 40 MB | GitHub LFS (already committed) |
| AlphaFold2 PDBs (13 proteins) | `data/pdb/alphafold/*.pdb` | ~4 MB total | EBI AlphaFold DB: `https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_v6.pdb` |
| Crystal structures | `data/pdb/crystal/5UHB_rpoB.pdb`, `2CAS_katG.pdb` | ~2.5 MB | RCSB: `https://files.rcsb.org/download/{PDB_ID}.pdb` |

UniProt IDs for the 13 resistance proteins:
```
rpoB=P9WGY9, katG=P9WIE5, embB=P9WNL7, gyrA=P9WG47, gyrB=P9WG45,
pncA=I6XD65, rpsL=P9WH63, eis=P9WFK7, tap=P9WJX9, mmpR5=I6Y8F7,
mmpL5=P9WJV1, tlyA=P9WJ63, inhA=P9WGR1
```

### How to get it

```bash
# Extract committed archives (PDBs + CRyPTIC supplement)
python scripts/extract_data.py

# Download MUTATIONS.csv.gz (1.4 GB)
python scripts/download_cryptic_data.py
```

The individual scripts also auto-download missing AlphaFold PDBs at runtime via `04c_stage1_features.py:download_alphafold_pdb()`.

---

## Step 1: Stage 0 — Residue-Level Logistic Regression

**Script:** `scripts/04b_hotspot_model.py`

### What it does

Trains a binary logistic regression classifier on **~6,600 residues × 12 sequence features** to predict whether each residue is a hotspot (1 of 21 known positions) or not.

### Training data construction

**Positive examples:** 21 known hotspot residues across 7 genes:
```
rpoB:  170, 430, 435, 445, 450, 452, 491
katG:  315
embB:  306, 406, 497
gyrA:  90, 91, 94
gyrB:  538
pncA:  4, 10, 12, 125
rpsL:  43, 88
```

**Negative examples:** All other non-hotspot residues across the same 13 resistance genes (~6,579 residues).

### 12 sequence features

| # | Feature | Computation | Code ref |
|---|---------|-------------|----------|
| 1 | `inner_distance` | `min(abs(res_pos - p) for p in CORE_BINDING_RESIDUES[gene])`. Core binding ranges defined per gene (e.g., rpoB: 426-452, katG: 104-115∪270-330). Default=500 if gene has no core binding defined. | `04b.py` L80-95 |
| 2 | `homoplasy_count` | Sum of all non-reference allele counts at this codon position across the demo VCF (117 TB genomes). Parses GT field, counts alt alleles. | `04b.py` L140-180 |
| 3 | `homoplasy_alleles` | Count of distinct alternate alleles observed at this position. | `04b.py` L180-190 |
| 4 | `helix_propensity` | Chou-Fasman alpha-helix propensity value for the WT amino acid. Lookup table in `04_resistance_forecasting.py`. | `04_forecast.py` L300-320 |
| 5 | `strand_propensity` | Chou-Fasman beta-strand propensity for the WT amino acid. | `04_forecast.py` L320-340 |
| 6 | `hydrophobicity` | Kyte-Doolittle hydropathy index. | `04_forecast.py` L260-280 |
| 7 | `volume` | Side-chain volume in Å³ (Zamyatnin 1972). Dict: A=89, R=173, N=96, D=91, C=106, Q=114, E=109, G=48, H=117, I=166, L=166, K=168, M=162, F=189, P=90, S=73, T=93, W=220, Y=193, V=140. | `04_forecast.py` L280-300 |
| 8 | `charge` | 1 if AA in {R, K, H}, -1 if in {D, E}, else 0. | `04_forecast.py` L360-365 |
| 9 | `hbond` | 1 if AA in {R, N, D, Q, E, H, K, S, T, Y, W}, else 0. | `04_forecast.py` L340-360 |
| 10 | `rel_position` | `residue_position / protein_length`. Float in [0, 1]. | `04b.py` L200 |
| 11 | `conservation_blosum` | BLOSUM62 self-score (diagonal) for the WT amino acid. Range: A=4, R=5, N=6, D=6, C=9, Q=5, E=5, G=6, H=8, I=4, L=4, K=5, M=5, F=6, P=7, S=4, T=5, W=11, Y=7, V=4. Higher = more conserved. | `04b.py` L195-200 |
| 12 | `contact_density_seq` | Count of core binding residues within ±50 positions of current residue in the linear sequence. | `04b.py` L200-210 |

### Model parameters

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

model = LogisticRegression(
    C=1.0,              # L2 regularization strength
    class_weight="balanced",  # Inverse class frequency weighting
    max_iter=1000,
    random_state=42
)
scaler = StandardScaler()
```

### Validation procedure (not training)

The script does NOT do train/test splits for the main model. Instead:
- **Fit:** All 6,566 residues with complete features (no missing data)
- **Leave-one-hotspot-out:** For each of the 21 hotspots, remove that specific residue, retrain, predict its rank
- **Leave-one-gene-out:** For each of the 7 genes, remove all residues of that gene, retrain, predict on held-out gene

Metrics: Top-20 recall, AUROC (for LOO aggregate).

### Expected output

```
AUROC: 0.888
Hotspots in Top 20: 7/21
Major failures: rpoB D435 (#597), V170 (#953), L452 (#526)
```

Output file: `analysis/results/hotspot_model/residue_hotspot_data.csv` (used by all downstream scripts)

---

## Step 2: Stage 1 — Structural Feature Integration

**Script:** `scripts/04c_stage1_features.py`

### What it does

Adds three protein-structure-derived features to the Stage 0 residue data, then benchmarks LogisticRegression vs ElasticNet vs RandomForest.

### Three new features

#### Feature A: SASA (solvent-accessible surface area)

```python
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley

parser = PDBParser(QUIET=True)
structure = parser.get_structure("prot", pdb_path)
sr = ShrakeRupley()
sr.compute(structure[0], level="R")  # per-residue level

# For each residue:
sasa_raw = residue.sasa  # Shrake-Rupley SASA in Å²
sasa_relative = min(sasa_raw / MAX_ASA[aa], 1.0)  # Clamp to [0, 1]
# MAX_ASA from Tien et al. 2013 (e.g., A=121, R=265, ...)
```

Requires the AlphaFold PDB downloaded to `data/pdb/alphafold/{gene}_{uniprot}_alphafold.pdb`.

#### Feature B: ESM-2 intolerance score

```python
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D")

# For each residue position i in the protein sequence:
seq_with_mask = list(sequence)
seq_with_mask[i] = tokenizer.mask_token
inputs = tokenizer(" ".join(seq_with_mask), return_tensors="pt")
logits = model(**inputs).logits  # shape: (1, L+2, vocab_size)
# ESM-2 tokenizer adds special tokens: <cls> and <eos>
# So residue i maps to token position i+1
wt_token_id = tokenizer.get_vocab()[sequence[i]]
log_prob = torch.log_softmax(logits[0, i+1], dim=0)[wt_token_id]
esm2_intolerance = -log_prob.item()  # Higher = more intolerant
```

Runs on CPU if CUDA unavailable. **O(n²) complexity** — n residues × n forward passes. For rpoB (1,172 residues) this takes ~hours on CPU.

#### Feature C: 3D contact density

```python
# Extract Cα coordinates from AlphaFold PDB
ca_coords = []  # list of (residue_id, ca_position)
for chain in structure[0]:
    for residue in chain:
        if "CA" in residue:
            ca_coords.append((residue_id, residue["CA"].get_vector().get_array()))

# For each residue: count neighbors within 8Å (excluding self)
# 3D distance, not sequence distance
# O(n²) pairwise distance computation
```

### Models benchmarked

```python
models = {
    "LogisticRegression": LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=1000, random_state=42
    ),
    "ElasticNet": LogisticRegression(
        penalty="elasticnet", solver="saga",
        l1_ratio=0.5, C=1.0,
        class_weight="balanced", max_iter=1000, random_state=42
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=100, max_depth=5,
        class_weight="balanced", random_state=42
    ),
}
```

### Training procedure

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
features = 12 base + 3 structural = 15 total
target = "is_hotspot"

for train_idx, test_idx in skf.split(X, y):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])
    model.fit(X_train, y[train_idx])
    y_prob = model.predict_proba(X_test)[:, 1]
    # Metrics: AUROC, AUPRC, Top-20 recall
```

**Best model:** LogisticRegression (by mean AUROC). Retrained on full data.

### Expected output

```
AUROC: 0.910 (up from 0.888)
Hotspots in Top 20: 17/21 (up from 7/21)
Rescued: D435 #597→#20, V170 #953→#24, L452 #526→#19
```

Output file: `analysis/results/hotspot_model/ranked_predictions.csv` (ranked by `hotspot_score`)

---

## Step 3: Stage 1.5 — Rifampicin Co-Crystal Contact Distance

**Script:** `scripts/04d_docking_features.py`

### What it actually does (NOT molecular docking)

Despite the name, no docking software is invoked. The script measures Euclidean distances from each rpoB residue to the rifampicin molecule in the 5UHB cryo-EM structure. See `analysis/results/STAGE1_5_TECHNICAL_SUMMARY.md` for the full critique.

### Algorithm

```python
from Bio.PDB import PDBParser, Superimposer
import numpy as np

# 1. Load 5UHB, extract RFP (rifampicin) coordinates
crystal = PDBParser(QUIET=True).get_structure("5uhb", "5UHB_rpoB.pdb")
rfp_atoms = []
for chain in crystal[0]:
    for residue in chain:
        # Heteroatom residues start with "H_"
        if residue.get_id()[0].startswith("H_") and residue.get_resname() == "RFP":
            rfp_atoms = [a.get_vector().get_array() for a in residue.get_atoms()]
rfp_coords = np.array(rfp_atoms)  # shape: (N_RFP_atoms, 3)

# 2. Greedy CA atom alignment of AlphaFold rpoB → 5UHB chain C
af = PDBParser(QUIET=True).get_structure("af", "rpoB_P9WGY9_alphafold.pdb")
# Extract CA pairs by walking through 5UHB chain C, matching identical AAs
# within ±50 residue window in AlphaFold structure
cry_ca_pairs = []  # paired CA atoms from crystal
af_ca_pairs = []   # paired CA atoms from AlphaFold
af_start = 0
for cry_res in crystal_chain_C:
    for af_idx in range(af_start, min(af_start + 50, len(af_residues))):
        if af_seq[af_idx] == cry_seq[cry_idx]:
            cry_ca_pairs.append(cry_ca)
            af_ca_pairs.append(af_ca)
            af_start = af_idx + 1
            break  # greedy: take first match

# 3. Superimpose
sup = Superimposer()
sup.set_atoms(cry_ca_pairs, af_ca_pairs)
# Transform ALL AlphaFold atoms
for chain in af[0]:
    for residue in chain:
        for atom in residue:
            atom.transform(sup.rotran[0], sup.rotran[1])

# 4. Compute per-residue min distance to RFP
for residue in af[0]:
    res_coords = np.array([a.get_vector().get_array() for a in residue.get_atoms()])
    min_dist = np.min(np.linalg.norm(res_coords[:, None] - rfp_coords[None, :], axis=-1))

# 5. Map PDB residue numbers → H37Rv genome positions
pos_map = compute_position_mapping("rpoB")  # pairwise2 alignment
```

### Position mapping

Uses `Bio.pairwise2.align.globalms(pdb_seq, genome_seq, 2, -1, -2, -1)`:
- Match score = 2
- Mismatch penalty = -1
- Gap open = -2
- Gap extend = -1

This accounts for the 6-residue N-terminal extension in the AlphaFold PDB vs H37Rv annotation.

### Feature construction

| Feature | rpoB residues | All other residues |
|---------|---------------|-------------------|
| `drug_distance` | Min Euclidean distance to RFP (Å) | `max(rpoB_distances) + 10` ≈ 92.0 Å |
| `drug_contact` | 1 if `drug_distance ≤ 5.0`, else 0 | 0 |

### Model retraining

Same LR as Stage 1 with 2 additional features. 5-fold StratifiedCV.

```python
features = 12 base + 3 stage1 + 2 docking = 17 total
model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
```

### Expected output

```
AUROC: 0.938 (up from 0.910)
Hotspots in Top 20: 17/21 (unchanged — docking does NOT rescue V170, I491, V125, N538)
```

---

## Step 4: Mutation Forecasting

**Script:** `scripts/04e_mutation_forecasting.py`

### What it does

Takes the top 50 hotspots from Stage 1 (by `hotspot_score`), enumerates all SNV-accessible mutations at those residues, scores them by P(emergence), and produces a ranked watchlist.

### Mutation enumeration

For each hotspot residue:
1. Look up WT codon from the reference genome CDS sequence
2. For each of the 3 codon positions:
   - Try all 3 alternative nucleotides (A→{C,G,T}, C→{A,G,T}, etc.)
   - Translate to amino acid via the standard genetic code
   - Keep only non-synonymous, non-stop mutations
3. Max 9 mutations per residue (3 positions × 3 alternatives)

Total: 315 SNV-accessible mutations from 50 hotspot residues + 21 known hotspots.

### Scoring formula

```python
P(emergence) = P(hotspot | residue) × P(mutation | features)

where:
  P(hotspot | residue) = hotspot_score from Stage 1 logistic regression

  P(mutation | features) = 0.45 × resistance_score_norm
                         + 0.30 × fitness_score_norm
                         + 0.25 × evo_score_norm
```

### Per-mutation feature computation

#### resistance_score
```python
proximity = compute_proximity(row, drug_dist, inner_dist)
# If drug_distance < 20: proximity = max(0, 1 - drug_dist/15)
# Else: proximity = max(0, 1 - inner_dist/50)

disruptiveness = (
    0.3 * (1 - max(blosum62, -4) / 9) +   # BLOSUM disruption (normalized 0-1)
    0.2 * charge_change +                    # charge difference magnitude
    0.2 * size_change +                      # volume change magnitude
    0.2 * loss_of_hbond +                    # h-bond change
    0.1 * delta_hydrophobicity               # hydrophobicity change
)

resistance_score = proximity * disruptiveness
```

#### fitness_score
```python
fitness_score = (
    max(blosum62, -4) / 9         # BLOSUM preservation (higher = more fit)
    - 0.15 * charge_change
    - 0.15 * size_change
    - 0.10 * delta_hydrophobicity
    - 0.05 * loss_of_hbond
    - 3.0 * is_stop               # stop codons heavily penalized
)
fitness_score = np.clip(fitness_score, -1, 1)  # Clamp to [-1, 1]
```

#### evo_score
```python
evo_score = 0.6 * is_transition + 0.4
# Transition (A↔G or C↔T) = 1.0
# Transversion = 0.4
```

#### Sub-feature definitions
```python
blosum62           = BLOSUM62.get((wt_aa, mut_aa), -4)
charge_change      = abs(charge(mut) - charge(wt))       # charge: R/K/H=1, D/E=-1, else=0
size_change        = abs(vol(mut) - vol(wt)) / max(vol(wt), vol(mut), 1)  # fractional
delta_hydrophobicity = abs(hydro(mut) - hydro(wt))
loss_of_hbond      = abs(hb(mut) - hb(wt))               # hb: 1 if HBOND_AA
is_stop            = 1 if mut_aa == "*" else 0
is_transition      = 1 if nucleotide_change in {("A","G"),("G","A"),("C","T"),("T","C")} else 0
```

#### Score normalization
Each of `resistance_score`, `fitness_score`, `evo_score` is min-max normalized to [0, 1]:
```python
def normalize(x):
    if x.max() == x.min():
        return np.full_like(x, 0.5, dtype=float)
    return (x - x.min()) / (x.max() - x.min())
```

### Output

`analysis/results/forecasting/emergence_watchlist.csv` — 315 mutations ranked by emergence_score, with:
- `gene, mutation, wt_aa, mut_aa, residue_pos`
- `hotspot_score, resistance_score, fitness_score, evo_score`
- `emergence_score, rank`
- `is_known_resistance` (1/0)

---

## Step 5: Leave-One-Gene-Out Validation

**Script:** `scripts/05_leave_one_gene_out.py`

### What it does

For each of the 7 hotspot genes, holds out all data from that gene, retrains on the remaining 6 genes, predicts hotspot scores and mutation emergence, and evaluates recall of known mutations in the held-out gene.

### Hotspot model parameters (note: different C from Stage 0!)

```python
LogisticRegression(
    C=10.0,                # Less regularization than Stage 0 (C=1.0)
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)
```

### Procedure

```python
HOTSPOT_GENES = ["embB", "gyrA", "gyrB", "katG", "pncA", "rpoB", "rpsL"]

for held_out_gene in HOTSPOT_GENES:
    # Split data
    train = all_residues[all_residues.gene != held_out_gene]
    test  = all_residues[all_residues.gene == held_out_gene]

    # Features: 12 base + up to 3 structural (if pickle files exist)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[features])
    X_test = scaler.transform(test[features])

    # Train
    model = LogisticRegression(C=10.0, class_weight="balanced", max_iter=1000)
    model.fit(X_train, train.is_hotspot)

    # Predict hotspots for held-out gene
    test["hotspot_score"] = model.predict_proba(X_test)[:, 1]

    # Take top 30 + all known hotspots
    candidate_residues = test.nlargest(30, "hotspot_score")
    candidate_residues = pd.concat([candidate_residues, known_hotspots_for_gene])

    # Enumerate and score mutations (same formula as 04e)
    mutations = enumerate_snv_mutations(candidate_residues)
    mutations = score_mutations(mutations)
```

### Metrics per gene

- Top-20 recall: fraction of known mutations in top 20 of emergence_score ranking
- Top-50 recall, Top-100 recall
- Median rank of known mutations
- AUROC of emergence_score vs is_known_resistance

### Expected output

```
Aggregate Top-50 recall: 52% (17/33 known mutations in top 50)
Aggregate Top-20 recall: 18% (6/33)
Best gene: rpsL (Top-50 recall 2/2, AUROC 0.873)
Worst gene: katG (Top-50 recall 1/3, AUROC 0.610)
```

---

## Step 6: CRyPTIC Validation

**Script:** `scripts/08_cryptic_validation_full.py`

### What it does

Downloads/streams the 1.4 GB CRyPTIC MUTATIONS table, filters for 12,287 phenotype-matched samples, and cross-references the 315 watchlist mutations to see which ones appear in real clinical isolates.

### Algorithm

```python
# 1. Load phenotype UNIQUEIDs
pheno = pd.read_csv("data/metadata/cryptic_phenotypes.csv")
phenotyped_uids = set(pheno.UNIQUEID)

# 2. Stream MUTATIONS.csv.gz (1.4 GB)
carrier_map = defaultdict(set)  # (gene, mutation) → set(UNIQUEID)
with gzip.open("data/cryptic/MUTATIONS.csv.gz", "rt") as f:
    for row in csv.DictReader(f):
        if row["UNIQUEID"] not in phenotyped_uids:
            continue
        if row["GENE"] not in TARGET_GENES:
            continue
        if row["IS_NONSYNONYMOUS"] != "True":
            continue
        carrier_map[(row["GENE"], row["MUTATION"])].add(row["UNIQUEID"])

# 3. Cache to pickle (~6 MB)
with open("data/cryptic/cache/resistance_mutations.pkl", "wb") as f:
    pickle.dump(carrier_map, f)

# 4. For each watchlist mutation:
for _, wl in emergence_watchlist.iterrows():
    carriers = carrier_map.get((wl.gene, wl.mutation), set())
    n_carriers = len(carriers & phenotyped_uids)

    if n_carriers > 0:
        # Look up phenotypes for carriers
        carrier_pheno = pheno[pheno.UNIQUEID.isin(carriers)]
        drug_col = GENE_DRUG_MAP[wl.gene]  # e.g., "RIF_BINARY_PHENOTYPE"
        if drug_col and drug_col in carrier_pheno.columns:
            counts = carrier_pheno[drug_col].value_counts()
            R = counts.get("R", 0)
            S = counts.get("S", 0)
            resistance_frac = R / (R + S) if (R + S) > 0 else np.nan

            # Fisher's exact test
            bg = pheno[drug_col].value_counts()
            bg_R = bg.get("R", 0)
            bg_S = bg.get("S", 0)
            odds_ratio, p_value = fisher_exact([[R, S], [bg_R - R, bg_S - S]])
```

### Drug-to-phenotype column mapping

```python
GENE_DRUG_MAP = {
    "rpoB": "RIF_BINARY_PHENOTYPE",
    "katG": "INH_BINARY_PHENOTYPE",
    "embB": "EMB_BINARY_PHENOTYPE",
    "gyrA": "MXF_BINARY_PHENOTYPE",  # primary: moxifloxacin
    "gyrB": "MXF_BINARY_PHENOTYPE",
    "pncA": None,   # No binary PZA phenotype in CRyPTIC
    "rpsL": None,   # No binary STR phenotype
    "eis": None,
    "tap": None,
    # ... others map to respective drug columns
}
GENE_DRUG_ALT = {
    "gyrA": "LEV_BINARY_PHENOTYPE",  # alternative: levofloxacin
    "gyrB": "LEV_BINARY_PHENOTYPE",
}
```

### Categories

| Category | Criteria | Expected count |
|----------|----------|----------------|
| A (Known WHO) | `is_known_who == True AND n_carriers > 0` | 30 |
| B (Novel observed) | `is_known_who == False AND n_carriers > 0` | 81 |
| C (Forecast-only) | `n_carriers == 0` | 179 |

---

## Step 7: Stress Tests & Tiering

**Script:** `scripts/09_stress_tests.py`

### FDR correction

For all Category B mutations with ≥3 phenotyped carriers:
```python
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

# Benjamini-Hochberg FDR correction
reject, pvals_corrected, _, _ = multipletests(p_values, method="fdr_bh", alpha=0.05)
```

### Tier assignment

| Tier | Criteria | Count |
|------|----------|-------|
| 0 | Known WHO mutation, observed | 30 |
| 1 | Novel, FDR q < 0.05 | 22 |
| 2 | Novel, observed, p < 0.05 uncorrected OR R>50% | 32 |
| 3 | Novel, observed, no phenotype data | 27 |
| 4 | Forecast-only (not observed in CRyPTIC) | 179 |

---

## Step 8: Figures & Documentation

**Scripts:** `10_generate_figures.py`, `11_render_figures.py`

### What they produce

| Figure | Content |
|--------|---------|
| Figure 1 | Pipeline overview + key statistics |
| Figure 2 | AlphaFold validation + Stage 0 vs Stage 1 rescue |
| Figure 3 | Feature importance (coefficients) |
| Figure 4 | Watchlist composition (status counts) |
| Figure 5 | CRyPTIC validation cascade + Tier 1 hits |
| Figure 6 | Clinical impact summary |
| Figure S1 | ROC curves for Stage 0/1/1.5 |
| Figure S2 | Leave-one-gene-out results |
| Figure S3 | Failure analysis case studies |
| Figure S4 | Complete ranked watchlist |

---

## Putting It All Together: End-to-End Command

```bash
# 0. Setup
python scripts/extract_data.py
python scripts/download_cryptic_data.py

# 1. Stage 0
python scripts/04b_hotspot_model.py

# 2. Stage 1 (structural features)
python scripts/04c_stage1_features.py

# 3. Stage 1.5 (co-crystal contact distance)
python scripts/04d_docking_features.py

# 4. Mutation forecasting
python scripts/04e_mutation_forecasting.py

# 5. Leave-one-gene-out validation
python scripts/05_leave_one_gene_out.py

# 6. Failure analysis
python scripts/06_failure_analysis.py

# 7. CRyPTIC validation
python scripts/08_cryptic_validation_full.py

# 8. Stress tests
python scripts/09_stress_tests.py

# 9. Figures
python scripts/10_generate_figures.py
python scripts/11_render_figures.py

# 10. Audit
python scripts/12_audit.py
```

---

## Requirements

```
Python 3.12+
pandas, numpy, scipy
scikit-learn, xgboost
biopython
requests
torch, transformers (for ESM-2)
matplotlib, seaborn (for figures)
statsmodels (for FDR correction)
```

## Execution time estimates

| Step | Time | Notes |
|------|------|-------|
| 04b (Stage 0) | ~2 min | Mostly VCF parsing |
| 04c (Stage 1) | **~2-4 hours** | ESM-2 is O(n²) per protein; SASA is fast |
| 04d (Stage 1.5) | ~1 min | Single structure alignment |
| 04e (Forecasting) | ~1 min | Pure computation, no I/O |
| 05 (LOO) | ~2 min | 7 genes × retrain + score |
| 08 (CRyPTIC) | ~15 min | Streaming 1.4 GB gzip |
| 09 (Stress tests) | ~2 min | Fisher tests + FDR |
| 10-11 (Figures) | ~5 min | Data aggregation + rendering |
| **Total** | **~3-5 hours** | Dominated by ESM-2 in 04c |
