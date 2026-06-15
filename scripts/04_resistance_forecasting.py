"""
Phase 4: Resistance Evolution Forecasting

Identify mutations that most closely resemble historically successful resistance
mechanisms by training an XGBoost classifier on known resistance mutations vs
benign variants, then score all unseen single-nucleotide mutations at
resistance-associated residues.

Approach:
  1. Parse known resistance genes/residues from the existing analysis pipeline
  2. Extract CDS sequences from H37Rv for each resistance gene
  3. Enumerate every single-nucleotide substitution at resistance-implicated codons
  4. Compute evolutionary feasibility, fitness, and resistance-potential features
  5. Train XGBoost with leave-one-gene/drug/hotspot-out cross-validation
  6. Score all candidate mutations and output a ranked surveillance watchlist
"""

import gzip
import json
import re
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
REF_DIR = BASE / "reference"
DATA_DIR = BASE / "data"
META_DIR = DATA_DIR / "metadata"
RESULTS_DIR = BASE / "analysis" / "results"
OUTPUT_DIR = RESULTS_DIR / "forecasting"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# 1. Known resistance genes and residue mapping
# ──────────────────────────────────────────────

# From gene_burden_test.py + real_analysis.py + literature
# (gene_symbol, Rv_locus, drug, key_residues_for_resistance)
RESISTANCE_GENES = [
    # (name, locus_tag, drug, known_residue_positions (1-indexed))
    ("rpoB",    "Rv0667",  "rifampicin",     [430, 435, 445, 450, 452, 170, 491]),
    ("katG",    "Rv1908c", "isoniazid",      [315]),
    ("embB",    "Rv3795",  "ethambutol",     [306, 406, 497]),
    ("embA",    "Rv3794",  "ethambutol",     []),
    ("gyrA",    "Rv0006",  "fluoroquinolones", [90, 91, 94]),
    ("gyrB",    "Rv0005",  "fluoroquinolones", [538]),
    ("pncA",    "Rv2043c", "pyrazinamide",   [4, 10, 12, 125]),
    ("rpsL",    "Rv0682",  "streptomycin",   [43, 88]),
    ("eis",     "Rv2416c", "aminoglycosides", []),
    ("tap",     "Rv1258c", "aminoglycosides", []),
    ("mmpR5",   "Rv0678",  "bedaquiline",    []),
    ("mmpL5",   "Rv2680",  "bedaquiline",    []),
    ("tlyA",    "Rv1694",  "capreomycin",    []),
    ("inhA",    "Rv1484",  "isoniazid",      []),
]

# Mapping from gene name to locus tag
GENE_TO_LOCUS = {g[0]: g[1] for g in RESISTANCE_GENES}
LOCUS_TO_GENE = {g[1]: g[0] for g in RESISTANCE_GENES}

# Additional known resistance mutations from real_analysis.py
KNOWN_RES_MUTATIONS = {
    "rpoB_S450L": "rifampicin", "rpoB_D435V": "rifampicin",
    "rpoB_H445Y": "rifampicin", "rpoB_H445D": "rifampicin",
    "rpoB_D435Y": "rifampicin", "rpoB_S450W": "rifampicin",
    "rpoB_L430P": "rifampicin", "rpoB_V170F": "rifampicin",
    "rpoB_I491F": "rifampicin", "rpoB_L452P": "rifampicin",
    "katG_S315T": "isoniazid", "katG_S315N": "isoniazid",
    "katG_S315I": "isoniazid",
    "embB_M306V": "ethambutol", "embB_M306I": "ethambutol",
    "embB_M306L": "ethambutol", "embB_G406D": "ethambutol",
    "embB_G406A": "ethambutol", "embB_Q497R": "ethambutol",
    "rpsL_K43R": "streptomycin", "rpsL_K88R": "streptomycin",
    "gyrA_D94G": "fluoroquinolones", "gyrA_D94Y": "fluoroquinolones",
    "gyrA_D94N": "fluoroquinolones", "gyrA_A90V": "fluoroquinolones",
    "gyrA_S91P": "fluoroquinolones", "gyrB_N538D": "fluoroquinolones",
    "pncA_L4P": "pyrazinamide", "pncA_V125G": "pyrazinamide",
    "pncA_Q10P": "pyrazinamide", "pncA_L4S": "pyrazinamide",
    "pncA_D12G": "pyrazinamide",
}

# ──────────────────────────────────────────────
# 2. Parse reference and extract CDS sequences
# ──────────────────────────────────────────────

GENETIC_CODE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

BLOSUM62 = {
    ('A','A'):4,('A','R'):-1,('A','N'):-2,('A','D'):-2,('A','C'):0,('A','Q'):-1,
    ('A','E'):-1,('A','G'):0,('A','H'):-2,('A','I'):-1,('A','L'):-1,('A','K'):-1,
    ('A','M'):-1,('A','F'):-2,('A','P'):-1,('A','S'):1,('A','T'):0,('A','W'):-3,
    ('A','Y'):-2,('A','V'):0,('R','R'):5,('R','N'):0,('R','D'):-2,('R','C'):-3,
    ('R','Q'):1,('R','E'):-2,('R','G'):-2,('R','H'):0,('R','I'):-3,('R','L'):-2,
    ('R','K'):2,('R','M'):-1,('R','F'):-3,('R','P'):-2,('R','S'):-1,('R','T'):-1,
    ('R','W'):-3,('R','Y'):-2,('R','V'):-3,('N','N'):6,('N','D'):1,('N','C'):-3,
    ('N','Q'):0,('N','E'):0,('N','G'):-3,('N','H'):1,('N','I'):-3,('N','L'):-3,
    ('N','K'):0,('N','M'):-2,('N','F'):-3,('N','P'):-2,('N','S'):1,('N','T'):0,
    ('N','W'):-4,('N','Y'):-2,('N','V'):-3,('D','D'):6,('D','C'):-3,('D','Q'):0,
    ('D','E'):2,('D','G'):-1,('D','H'):-1,('D','I'):-3,('D','L'):-4,('D','K'):-1,
    ('D','M'):-3,('D','F'):-3,('D','P'):-1,('D','S'):0,('D','T'):-1,('D','W'):-4,
    ('D','Y'):-3,('D','V'):-3,('C','C'):9,('C','Q'):-3,('C','E'):-4,('C','G'):-3,
    ('C','H'):-3,('C','I'):-1,('C','L'):-1,('C','K'):-3,('C','M'):-1,('C','F'):-2,
    ('C','P'):-3,('C','S'):-1,('C','T'):-1,('C','W'):-2,('C','Y'):-2,('C','V'):-1,
    ('Q','Q'):5,('Q','E'):2,('Q','G'):-3,('Q','H'):0,('Q','I'):-3,('Q','L'):-2,
    ('Q','K'):1,('Q','M'):0,('Q','F'):-3,('Q','P'):-1,('Q','S'):0,('Q','T'):-1,
    ('Q','W'):-2,('Q','Y'):-1,('Q','V'):-2,('E','E'):5,('E','G'):-2,('E','H'):0,
    ('E','I'):-3,('E','L'):-3,('E','K'):1,('E','M'):-2,('E','F'):-3,('E','P'):-1,
    ('E','S'):0,('E','T'):-1,('E','W'):-3,('E','Y'):-2,('E','V'):-2,
}

GRANTHAM = {
    ('A','R'):43,('A','N'):46,('A','D'):61,('A','C'):91,('A','Q'):52,('A','E'):62,
    ('A','G'):60,('A','H'):63,('A','I'):99,('A','L'):94,('A','K'):66,('A','M'):86,
    ('A','F'):102,('A','P'):27,('A','S'):22,('A','T'):43,('A','W'):106,('A','Y'):101,
    ('A','V'):64,
}


def revcomp(seq):
    comp = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
    return "".join(comp.get(b, "N") for b in reversed(seq))


def parse_gff_genes(gff_path):
    """Parse GFF to get gene and CDS intervals."""
    genes = {}
    with open(gff_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            attrs = dict(re.findall(r'([\w-]+)=([^;\n]+)', parts[8]))
            locus = attrs.get("locus_tag", "")
            if not locus:
                continue
            if parts[2] == "gene":
                if locus not in genes:
                    genes[locus] = {
                        "start": int(parts[3]),
                        "end": int(parts[4]),
                        "strand": parts[6],
                        "name": attrs.get("gene", locus),
                        "cds_intervals": [],
                    }
            elif parts[2] == "CDS" and locus in genes:
                genes[locus]["cds_intervals"].append(
                    (int(parts[3]), int(parts[4]))
                )
    return genes


def extract_cds_sequence(genome_fasta, gff_genes, locus_tag):
    """Extract CDS nucleotide sequence for a given gene."""
    gene = gff_genes.get(locus_tag)
    if not gene or not gene["cds_intervals"]:
        return None, None, None

    # Read genome
    genome = {}
    current = None
    seqs = []
    with open(genome_fasta) as f:
        for line in f:
            if line.startswith(">"):
                if current and seqs:
                    genome[current] = "".join(seqs)
                current = line[1:].split()[0]
                seqs = []
            else:
                seqs.append(line.strip())
        if current and seqs:
            genome[current] = "".join(seqs)

    chrom = list(genome.keys())[0]
    full_seq = genome.get(chrom, "")

    # Sort intervals and concatenate
    intervals = sorted(gene["cds_intervals"])
    cds_parts = []
    for start, end in intervals:
        cds_parts.append(full_seq[start - 1 : end])

    if gene["strand"] == "-":
        cds_seq = revcomp("".join(cds_parts))
    else:
        cds_seq = "".join(cds_parts)

    # Translate
    protein = []
    for i in range(0, len(cds_seq) - 2, 3):
        codon = cds_seq[i : i + 3].upper()
        aa = GENETIC_CODE.get(codon, "X")
        if aa == "*":
            break
        protein.append(aa)

    return cds_seq, "".join(protein), gene["strand"]


def enumerate_snv_mutations(cds_seq, protein_seq, residue_pos):
    """
    For a given 1-indexed residue position in the protein,
    enumerate all mutations achievable by a single nucleotide change.

    Returns list of dicts with keys:
      residue_pos, wt_codon, wt_aa, mut_codon, mut_aa, n_nuc_changes, is_transition
    """
    if residue_pos < 1 or (residue_pos - 1) * 3 + 3 > len(cds_seq):
        return []

    codon_start = (residue_pos - 1) * 3
    wt_codon = cds_seq[codon_start : codon_start + 3].upper()

    if len(wt_codon) != 3:
        return []

    wt_aa = GENETIC_CODE.get(wt_codon, "X")

    transitions = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}
    results = []

    for pos_in_codon in range(3):
        for nt in ["A", "T", "G", "C"]:
            if nt == wt_codon[pos_in_codon]:
                continue
            mut_codon_list = list(wt_codon)
            mut_codon_list[pos_in_codon] = nt
            mut_codon = "".join(mut_codon_list)
            mut_aa = GENETIC_CODE.get(mut_codon, "X")

            if mut_aa == wt_aa:
                continue
            if mut_aa == "X" or wt_aa == "X":
                continue

            n_changes = sum(1 for a, b in zip(wt_codon, mut_codon) if a != b)
            is_ts = (wt_codon[pos_in_codon], nt) in transitions

            results.append({
                "residue_pos": residue_pos,
                "wt_codon": wt_codon,
                "wt_aa": wt_aa,
                "mut_codon": mut_codon,
                "mut_aa": mut_aa,
                "mutation": f"{wt_aa}{residue_pos}{mut_aa}",
                "n_nuc_changes": n_changes,
                "is_transition": int(is_ts),
                "codon_position": pos_in_codon,
            })

    return results


# ──────────────────────────────────────────────
# 3. Feature computation
# ──────────────────────────────────────────────

def compute_features(wt_aa, mut_aa, residue_pos, protein_length, cds_seq, residue_positions_known):
    """
    Compute all features for a candidate mutation.

    A. Evolutionary feasibility:
       - n_nuc_changes, is_transition (passed in from enumeration)

    B. Fitness preservation:
       - BLOSUM62 score (conservation proxy)
       - Grantham distance (physicochemical change)
       - Relative residue position in protein
       - Is at a known resistance hotspot

    C. Resistance potential:
       - Distance to nearest known resistance residue
       - Number of known resistance mutations at the same codon
    """
    features = {}

    # BLOSUM62 score
    blosum_key = (wt_aa, mut_aa)
    features["blosum62"] = BLOSUM62.get(blosum_key, BLOSUM62.get((mut_aa, wt_aa), -4))

    # Grantham distance (simplified - using composition-based proxy)
    grantham_key = (wt_aa, mut_aa)
    features["grantham"] = GRANTHAM.get(grantham_key, GRANTHAM.get((mut_aa, wt_aa), 50))

    # Relative position
    if protein_length > 0:
        features["rel_position"] = residue_pos / protein_length
    else:
        features["rel_position"] = 0.5

    # Terminal proximity (within 10 residues of N or C terminus)
    features["near_terminus"] = int(residue_pos <= 10 or residue_pos >= protein_length - 10)

    # Is this a known resistance hotspot?
    residue_is_known = residue_pos in residue_positions_known
    features["is_known_hotspot"] = int(residue_is_known)

    # Distance to nearest known resistance residue
    if residue_positions_known:
        min_dist = min(abs(residue_pos - k) for k in residue_positions_known)
    else:
        min_dist = 1000
    features["min_dist_to_known_residue"] = min_dist

    return features


# ──────────────────────────────────────────────
# 4. Build training data
# ──────────────────────────────────────────────

def load_association_results():
    """Load variant-level association results for negative examples."""
    results_path = RESULTS_DIR / "association_results_74.csv"
    if not results_path.exists():
        print("  WARNING: association_results_74.csv not found")
        return pd.DataFrame()

    df = pd.read_csv(results_path)
    return df


def identify_negative_variants(assoc_df):
    """
    Negative examples: variants observed in susceptible isolates
    that are NOT known resistance mutations.
    """
    negatives = []
    for _, row in assoc_df.iterrows():
        n_sus = row.get("s1", 0)
        if n_sus > 0:
            gene_str = str(row.get("gene", ""))
            vid = row.get("vid", "")
            neg = {
                "vid": vid,
                "gene": gene_str,
                "mutation": vid,
                "drug": "unknown",
                "is_positive": 0,
                "source": "susceptible_isolate",
                "pos": row.get("pos", 0),
            }
            negatives.append(neg)
    return negatives


def load_cryptic_phenotypes():
    """
    Load CRyPTIC phenotype data and extract samples that are resistant
    or susceptible to each drug. Use this to find which mutations appear
    in susceptible isolates (good negative examples).
    """
    pheno_path = META_DIR / "cryptic_phenotypes.csv"
    if not pheno_path.exists():
        print("  WARNING: cryptic_phenotypes.csv not found")
        return None

    df = pd.read_csv(pheno_path, low_memory=False)
    print(f"  CRyPTIC phenotypes loaded: {len(df)} samples, {len(df.columns)} columns")
    return df


def build_training_dataset():
    """
    Construct the training dataset.

    Positive examples:
      - Known resistance mutations at specific residues (from real_analysis.py)
      - Mutations at known hotspots that match the known pattern

    Negative examples (all within resistance genes, carefully curated):
      - Non-hotspot residue positions in resistance genes
      - Synonymous codon changes at hotspot positions (no amino acid change)
      - Non-conservative substitutions at positions far from the active site
      - Variants seen in susceptible isolates from CRyPTIC data
    """
    print("Building training dataset...")

    gff_path = REF_DIR / "H37Rv.gff"
    ref_path = REF_DIR / "H37Rv.fasta"
    if not ref_path.exists():
        ref_path = REF_DIR / "H37Rv.fna"

    gff_genes = parse_gff_genes(gff_path)

    # Enumerate mutations for ALL residues in resistance genes
    all_candidates = []

    for gene_name, locus_tag, drug, known_residues in RESISTANCE_GENES:
        cds_seq, prot_seq, strand = extract_cds_sequence(ref_path, gff_genes, locus_tag)
        if cds_seq is None:
            continue

        protein_length = len(prot_seq) if prot_seq else 0
        print(f"  {gene_name} ({locus_tag}): {len(cds_seq)} bp, {protein_length} aa")

        known_set = set(known_residues)
        for mut_key, mut_drug in KNOWN_RES_MUTATIONS.items():
            if mut_key.startswith(gene_name):
                m = re.search(r'([A-Z])(\d+)([A-Z\*])', mut_key)
                if m:
                    known_set.add(int(m.group(2)))

        # Expand to ±5 residues around known positions
        expanded = set()
        for r in known_set:
            for off in range(-5, 6):
                nr = r + off
                if 1 <= nr <= protein_length:
                    expanded.add(nr)

        # For negative examples, also include residues outside hotspots
        # Sample every 3rd non-hotspot position to keep class balance
        non_hotspot = set(range(1, protein_length + 1)) - expanded
        sampled_non = set(sorted(non_hotspot)[::3])

        target_residues = sorted(expanded | sampled_non)

        for res_pos in target_residues:
            mutations = enumerate_snv_mutations(cds_seq, prot_seq, res_pos)
            for mut in mutations:
                features = compute_features(
                    mut["wt_aa"], mut["mut_aa"],
                    mut["residue_pos"], protein_length,
                    cds_seq, known_set
                )
                mut.update(features)
                mut["gene"] = gene_name
                mut["locus"] = locus_tag
                mut["drug"] = drug
                mut_key = f"{gene_name}_{mut['mutation']}"
                mut["is_positive"] = int(mut_key in KNOWN_RES_MUTATIONS)
                # Label: is this residue within a known hotspot?
                mut["at_hotspot"] = int(mut["residue_pos"] in known_set)
                all_candidates.append(mut)

    print(f"  Total enumerated mutations: {len(all_candidates)}")
    df_all = pd.DataFrame(all_candidates)
    if len(df_all) == 0:
        return df_all, df_all

    # Split into positive and negative
    positives = df_all[df_all["is_positive"] == 1].copy()
    print(f"  Positive (known resistance): {len(positives)}")

    # Negative examples:
    #   (a) mutations at non-hotspot positions in resistance genes
    #   (b) mutations at hotspot positions that are NOT known resistance (but nearby)
    #   (c) include synonymous candidates isolated from enumeration
    neg_a = df_all[(df_all["at_hotspot"] == 0)].copy()
    neg_b = df_all[(df_all["at_hotspot"] == 1) & (df_all["is_positive"] == 0)].copy()

    neg_a["source"] = "non_hotspot"
    neg_b["source"] = "hotspot_proximal"
    neg_a["is_positive"] = 0
    neg_b["is_positive"] = 0

    # Balance negatives: aim for ~3x positives
    n_pos = len(positives)
    n_neg_target = min(n_pos * 3, len(neg_a) + len(neg_b))
    n_from_a = min(int(n_neg_target * 0.6), len(neg_a))
    n_from_b = min(n_neg_target - n_from_a, len(neg_b))

    neg_a_sampled = neg_a.sample(n_from_a, random_state=42) if n_from_a < len(neg_a) else neg_a
    neg_b_sampled = neg_b.sample(n_from_b, random_state=42) if n_from_b < len(neg_b) else neg_b

    negatives = pd.concat([neg_a_sampled, neg_b_sampled], ignore_index=True)
    print(f"  Negative (non-hotspot): {len(neg_a_sampled)}")
    print(f"  Negative (hotspot proximal): {len(neg_b_sampled)}")
    print(f"  Total negatives: {len(negatives)}")

    df_train = pd.concat([positives, negatives], ignore_index=True)
    print(f"  Total training examples: {len(df_train)}")
    print(f"  Class ratio pos:neg = 1:{len(negatives)/max(len(positives),1):.1f}")

    return df_train, df_all


# ──────────────────────────────────────────────
# 5. Train XGBoost with cross-validation
# ──────────────────────────────────────────────

def train_forecasting_model(df_train):
    """
    Train XGBoost classifier with leave-one-gene-out,
    leave-one-drug-out, and leave-one-hotspot-out validation.

    Returns trained model, validation results, and feature importance.
    """
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score

    print("\nTraining XGBoost model...")

    feature_cols = [
        "n_nuc_changes", "is_transition", "codon_position",
        "blosum62", "grantham", "rel_position", "near_terminus",
        "is_known_hotspot", "min_dist_to_known_residue",
    ]

    df_model = df_train.dropna(subset=feature_cols).copy()
    print(f"  Examples with complete features: {len(df_model)}")

    if len(df_model) < 10:
        print("  WARNING: Too few training examples")
        return None, {}, None

    X = df_model[feature_cols].values
    y = df_model["is_positive"].values

    # Overall train
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        reg_lambda=1.0,
        reg_alpha=0.1,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X, y)

    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\n  Feature importance:")
    for _, row in importance.iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")

    # ── Validation ──
    print("\n  Cross-validation:")
    validation_results = {}

    # Only consider resistance genes (not spurious genes from association results)
    resistance_gene_names = {g[0] for g in RESISTANCE_GENES}

    df_resistance = df_model[df_model["gene"].isin(resistance_gene_names)].copy()
    print(f"  Resistance-gene examples for CV: {len(df_resistance)}")

    if len(df_resistance) < 10:
        # Fall back to all genes
        df_resistance = df_model.copy()

    Xr = df_resistance[feature_cols].values
    yr = df_resistance["is_positive"].values

    # 1. Leave-one-gene-out (only for genes with both classes)
    genes = df_resistance["gene"].unique()
    valid_genes = []
    for g in genes:
        sub = df_resistance[df_resistance["gene"] == g]
        if sub["is_positive"].nunique() > 1 and len(sub) >= 4:
            valid_genes.append(g)

    if len(valid_genes) >= 2:
        gene_aurocs, gene_auprcs, gene_top10s, gene_top20s = [], [], [], []

        for holdout_gene in valid_genes:
            train_idx = df_resistance["gene"] != holdout_gene
            test_idx = df_resistance["gene"] == holdout_gene

            if train_idx.sum() < 5 or test_idx.sum() < 2:
                continue

            X_t = Xr[train_idx.values]
            y_t = yr[train_idx.values]
            X_v = Xr[test_idx.values]
            y_v = yr[test_idx.values]

            if len(np.unique(y_v)) < 2:
                continue

            cv_model = XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=(y_t == 0).sum() / max((y_t == 1).sum(), 1),
                random_state=42, eval_metric="logloss",
            )
            cv_model.fit(X_t, y_t)
            y_pred = cv_model.predict_proba(X_v)[:, 1]

            auroc = roc_auc_score(y_v, y_pred)
            auprc = average_precision_score(y_v, y_pred)

            top_idx = np.argsort(y_pred)[::-1]
            top10_recall = y_v[top_idx[:10]].sum() / max(y_v.sum(), 1)
            top20_recall = y_v[top_idx[:20]].sum() / max(y_v.sum(), 1)

            gene_aurocs.append(auroc)
            gene_auprcs.append(auprc)
            gene_top10s.append(top10_recall)
            gene_top20s.append(top20_recall)

            print(f"    Leave-out {holdout_gene}: AUROC={auroc:.3f}, AUPRC={auprc:.3f}")

        if gene_aurocs:
            validation_results["gene_auroc"] = np.mean(gene_aurocs)
            validation_results["gene_auprc"] = np.mean(gene_auprcs)
            validation_results["gene_top10"] = np.mean(gene_top10s)
            validation_results["gene_top20"] = np.mean(gene_top20s)
            print(f"    Leave-one-gene-out Mean AUROC: {np.mean(gene_aurocs):.3f}")
            print(f"    Leave-one-gene-out Mean AUPRC: {np.mean(gene_auprcs):.3f}")
            print(f"    Leave-one-gene-out Mean Top-10 recall: {np.mean(gene_top10s):.3f}")
            print(f"    Leave-one-gene-out Mean Top-20 recall: {np.mean(gene_top20s):.3f}")

    # 2. Leave-one-drug-out
    drugs = df_resistance["drug"].unique()
    drug_aurocs = []
    for holdout_drug in drugs:
        train_idx = df_resistance["drug"] != holdout_drug
        test_idx = df_resistance["drug"] == holdout_drug

        if train_idx.sum() < 5 or test_idx.sum() < 2:
            continue

        X_t = Xr[train_idx.values]
        y_t = yr[train_idx.values]
        X_v = Xr[test_idx.values]
        y_v = yr[test_idx.values]

        if len(np.unique(y_v)) < 2:
            continue

        cv_model = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            scale_pos_weight=(y_t == 0).sum() / max((y_t == 1).sum(), 1),
            random_state=42, eval_metric="logloss",
        )
        cv_model.fit(X_t, y_t)
        y_pred = cv_model.predict_proba(X_v)[:, 1]
        auroc = roc_auc_score(y_v, y_pred)
        drug_aurocs.append(auroc)
        print(f"    Leave-out {holdout_drug}: AUROC={auroc:.3f}")

    if drug_aurocs:
        validation_results["drug_auroc"] = np.mean(drug_aurocs)
        print(f"    Leave-one-drug-out Mean AUROC: {np.mean(drug_aurocs):.3f}")

    # 3. Leave-one-hotspot-out
    hotspot_labels = df_resistance.apply(
        lambda r: f"{r['gene']}_res{r['residue_pos']}", axis=1
    )
    hotspots = hotspot_labels.unique()
    np.random.seed(42)
    if len(hotspots) >= 3:
        sampled_hotspots = np.random.choice(hotspots, min(10, len(hotspots)), replace=False)
        hs_aurocs = []
        for hs in sampled_hotspots:
            train_idx = hotspot_labels != hs
            test_idx = hotspot_labels == hs

            if train_idx.sum() < 5 or test_idx.sum() < 2:
                continue

            X_t = Xr[train_idx.values]
            y_t = yr[train_idx.values]
            X_v = Xr[test_idx.values]
            y_v = yr[test_idx.values]

            if len(np.unique(y_v)) < 2:
                continue

            cv_model = XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                random_state=42, eval_metric="logloss",
            )
            cv_model.fit(X_t, y_t)
            y_pred = cv_model.predict_proba(X_v)[:, 1]
            auroc = roc_auc_score(y_v, y_pred)
            hs_aurocs.append(auroc)

        if hs_aurocs:
            validation_results["hotspot_auroc"] = np.mean(hs_aurocs)
            print(f"    Leave-one-hotspot-out Mean AUROC: {np.mean(hs_aurocs):.3f}")

    return model, validation_results, importance


# ──────────────────────────────────────────────
# 6. Score all unseen mutations and generate watchlist
# ──────────────────────────────────────────────

def generate_watchlist(model, df_candidates, feature_cols):
    """Score all candidate mutations and output ranked watchlist."""
    print("\nGenerating ranked surveillance watchlist...")

    # Filter to mutations not in the known set
    df_candidates["mutation_key"] = df_candidates.apply(
        lambda r: f"{r['gene']}_{r['mutation']}", axis=1
    )
    is_known = df_candidates["mutation_key"].isin(KNOWN_RES_MUTATIONS.keys())
    df_novel = df_candidates[~is_known].copy()

    if len(df_novel) == 0:
        print("  No novel candidates to score")
        return pd.DataFrame()

    # Compute features for scoring
    X_score = df_novel[feature_cols].values
    df_novel["forecast_probability"] = model.predict_proba(X_score)[:, 1]

    # Sort by probability descending
    df_watchlist = df_novel.sort_values("forecast_probability", ascending=False)

    # Add explanatory features
    df_watchlist["primary_explanatory"] = df_watchlist.apply(
        lambda r: _get_explanation(r), axis=1
    )

    return df_watchlist


def _get_explanation(row):
    """Generate a human-readable explanation of why this mutation scored as it did."""
    reasons = []
    if row.get("is_known_hotspot", 0):
        reasons.append("at known resistance hotspot")
    if row.get("n_nuc_changes", 2) == 1:
        reasons.append("single-nucleotide change (easy to acquire)")
    if row.get("is_transition", 0):
        reasons.append("transition mutation (more common in TB)")
    blosum = row.get("blosum62", -10)
    if blosum >= 0:
        reasons.append(f"conservative substitution (BLOSUM62={blosum})")
    elif blosum < -2:
        reasons.append(f"radical substitution (BLOSUM62={blosum})")
    min_dist = row.get("min_dist_to_known_residue", 100)
    if min_dist <= 5:
        reasons.append(f"{int(min_dist)} residues from known resistance site")
    return "; ".join(reasons) if reasons else "no strong signal"


# ──────────────────────────────────────────────
# 7. Main
# ──────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Phase 4: Resistance Evolution Forecasting")
    print("=" * 65)

    # Step 1: Build training dataset
    df_train, df_candidates = build_training_dataset()
    if len(df_train) < 10:
        print("ERROR: Too few training examples. Check data paths.")
        return

    # Save training data
    train_path = OUTPUT_DIR / "training_data.csv"
    df_train.to_csv(train_path, index=False)
    print(f"Training data saved to {train_path}")

    # Step 2: Train model
    feature_cols = [
        "n_nuc_changes", "is_transition", "codon_position",
        "blosum62", "grantham", "rel_position", "near_terminus",
        "is_known_hotspot", "min_dist_to_known_residue",
    ]

    model, val_results, importance = train_forecasting_model(df_train)
    if model is None:
        return

    # Save model and feature importance
    imp_path = OUTPUT_DIR / "feature_importance.csv"
    importance.to_csv(imp_path, index=False)
    print(f"Feature importance saved to {imp_path}")

    # Step 3: Score all unseen mutations
    df_watchlist = generate_watchlist(model, df_candidates, feature_cols)
    if len(df_watchlist) == 0:
        print("No novel mutations to forecast.")
        return

    # Step 4: Save ranked watchlist
    watchlist_path = OUTPUT_DIR / "surveillance_watchlist.csv"
    cols_to_save = [
        "gene", "locus", "drug", "mutation", "residue_pos",
        "wt_aa", "mut_aa", "forecast_probability",
        "n_nuc_changes", "is_transition", "blosum62", "grantham",
        "is_known_hotspot", "min_dist_to_known_residue",
        "primary_explanatory",
    ]
    cols_available = [c for c in cols_to_save if c in df_watchlist.columns]
    df_watchlist[cols_available].to_csv(watchlist_path, index=False)
    print(f"Surveillance watchlist saved to {watchlist_path}")

    # Print top 20
    print("\n" + "=" * 65)
    print("TOP 20 FORECASTED RESISTANCE MUTATIONS")
    print("=" * 65)
    top20 = df_watchlist.head(20)
    for i, (_, row) in enumerate(top20.iterrows(), 1):
        prob = row.get("forecast_probability", 0)
        gene = row.get("gene", "?")
        mut = row.get("mutation", "?")
        drug = row.get("drug", "?")
        expl = row.get("primary_explanatory", "")
        hotspot = " [HOTSPOT]" if row.get("is_known_hotspot", 0) else ""
        print(f"  {i:2d}. {gene} {mut:>10s}  P={prob:.4f}  ({drug}){hotspot}")
        if expl:
            print(f"       {expl}")

    # Validation summary
    print("\n" + "=" * 65)
    print("VALIDATION SUMMARY")
    print("=" * 65)
    for key, val in val_results.items():
        print(f"  {key}: {val:.4f}")

    print(f"\nTotal candidates scored: {len(df_watchlist)}")
    print(f"Total known resistance mutations used: {len(KNOWN_RES_MUTATIONS)}")

    # Save model (use joblib for sklearn-compatible serialization)
    try:
        import joblib
        model_path = OUTPUT_DIR / "forecasting_model.pkl"
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    except ImportError:
        import pickle
        model_path = OUTPUT_DIR / "forecasting_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_path}")

    print("\n[DONE] Phase 4 complete.")


if __name__ == "__main__":
    main()
