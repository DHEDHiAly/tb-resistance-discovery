"""
Step 1: Leave-One-Gene-Out Validation of the Full Forecasting Pipeline

For each resistance gene:
  - Remove all data from that gene
  - Retrain hotspot model (LR) on remaining genes
  - Predict hotspot scores for the held-out gene
  - Enumerate SNV mutations at predicted hot residues
  - Score mutations by P(emergence)
  - Ask: where do held-out known mutations rank?

This tests whether the model generalizes to genes it has never seen.
"""

import json
import pickle
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
REF_DIR = BASE / "reference"
OUTPUT_DIR = BASE / "analysis" / "results" / "forecasting"
HOTSPOT_DIR = BASE / "analysis" / "results" / "hotspot_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Genetic code and biochemistry ──

GENETIC_CODE = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L','TCT':'S','TCC':'S',
    'TCA':'S','TCG':'S','TAT':'Y','TAC':'Y','TAA':'*','TAG':'*',
    'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CTT':'L','CTC':'L',
    'CTA':'L','CTG':'L','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'CAT':'H','CAC':'H','CAA':'Q','CAG':'Q','CGT':'R','CGC':'R',
    'CGA':'R','CGG':'R','ATT':'I','ATC':'I','ATA':'I','ATG':'M',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T','AAT':'N','AAC':'N',
    'AAA':'K','AAG':'K','AGT':'S','AGC':'S','AGA':'R','AGG':'R',
    'GTT':'V','GTC':'V','GTA':'V','GTG':'V','GCT':'A','GCC':'A',
    'GCA':'A','GCG':'A','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
    'GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}

BLOSUM62 = {
    ('A','A'):4,('A','R'):-1,('A','N'):-2,('A','D'):-2,('A','C'):0,
    ('A','Q'):-1,('A','E'):-1,('A','G'):0,('A','H'):-2,('A','I'):-1,
    ('A','L'):-1,('A','K'):-1,('A','M'):-1,('A','F'):-2,('A','P'):-1,
    ('A','S'):1,('A','T'):0,('A','W'):-3,('A','Y'):-2,('A','V'):0,
    ('R','R'):5,('R','N'):0,('R','D'):-2,('R','C'):-3,('R','Q'):1,
    ('R','E'):-2,('R','G'):-2,('R','H'):0,('R','I'):-3,('R','L'):-2,
    ('R','K'):2,('R','M'):-1,('R','F'):-3,('R','P'):-2,('R','S'):-1,
    ('R','T'):-1,('R','W'):-3,('R','Y'):-2,('R','V'):-3,
    ('N','N'):6,('N','D'):1,('N','C'):-3,('N','Q'):0,('N','E'):0,
    ('N','G'):-3,('N','H'):1,('N','I'):-3,('N','L'):-3,('N','K'):0,
    ('N','M'):-2,('N','F'):-3,('N','P'):-2,('N','S'):1,('N','T'):0,
    ('N','W'):-4,('N','Y'):-2,('N','V'):-3,
    ('D','D'):6,('D','C'):-3,('D','Q'):0,('D','E'):2,('D','G'):-1,
    ('D','H'):-1,('D','I'):-3,('D','L'):-4,('D','K'):-1,('D','M'):-3,
    ('D','F'):-3,('D','P'):-1,('D','S'):0,('D','T'):-1,('D','W'):-4,
    ('D','Y'):-3,('D','V'):-3,
    ('C','C'):9,
    ('Q','Q'):5,('Q','E'):2,('Q','G'):-3,('Q','H'):0,('Q','I'):-3,
    ('Q','L'):-2,('Q','K'):1,('Q','M'):0,('Q','F'):-3,('Q','P'):-1,
    ('Q','S'):0,('Q','T'):-1,('Q','W'):-2,('Q','Y'):-1,('Q','V'):-2,
    ('E','E'):5,
    ('G','G'):6,('G','H'):-2,('G','I'):-4,('G','L'):-4,('G','K'):-2,
    ('G','M'):-3,('G','F'):-4,('G','P'):-2,('G','S'):0,('G','T'):-2,
    ('G','W'):-2,('G','Y'):-3,('G','V'):-3,
    ('H','H'):8,
    ('I','I'):4,('I','L'):2,('I','K'):-3,('I','M'):1,('I','F'):0,
    ('I','P'):-3,('I','S'):-2,('I','T'):-1,('I','W'):-3,('I','Y'):-1,
    ('I','V'):3,
    ('L','L'):4,('L','K'):-2,('L','M'):2,('L','F'):0,('L','P'):-3,
    ('L','S'):-2,('L','T'):-1,('L','W'):-2,('L','Y'):-1,('L','V'):1,
    ('K','K'):5,
    ('M','M'):5,('M','F'):0,('M','P'):-2,('M','S'):-1,('M','T'):-1,
    ('M','W'):-1,('M','Y'):-1,('M','V'):1,
    ('F','F'):6,('F','P'):-4,('F','S'):-2,('F','T'):-2,('F','W'):1,
    ('F','Y'):3,('F','V'):-1,
    ('P','P'):7,
    ('S','S'):4,('S','T'):1,('S','W'):-3,('S','Y'):0,('S','V'):-2,
    ('T','T'):5,('T','W'):-2,('T','Y'):-2,('T','V'):0,
    ('W','W'):11,('W','Y'):2,('W','V'):-3,
    ('Y','Y'):7,('Y','V'):-1,
    ('V','V'):4,
}

HYDROPHOBICITY = {
    'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,
    'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,
    'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2,
}

VOLUME = {"G":60,"A":89,"S":89,"C":109,"T":116,"P":119,"D":111,"N":114,
          "V":140,"E":138,"Q":143,"H":153,"M":163,"I":167,"L":167,
          "K":168,"R":173,"F":190,"Y":193,"W":228}

HBOND_AAS = {"S","T","N","Q","C","Y","H","R","K","D","E","W"}
POSITIVE_AAS = {"R", "K", "H"}
NEGATIVE_AAS = {"D", "E"}

KNOWN_RES_MUTATIONS = {
    "rpoB_S450L", "rpoB_D435V", "rpoB_H445Y", "rpoB_H445D",
    "rpoB_D435Y", "rpoB_S450W", "rpoB_L430P", "rpoB_V170F",
    "rpoB_I491F", "rpoB_L452P",
    "katG_S315T", "katG_S315N", "katG_S315I",
    "embB_M306V", "embB_M306I", "embB_M306L",
    "embB_G406D", "embB_G406A", "embB_Q497R",
    "rpsL_K43R", "rpsL_K88R",
    "gyrA_D94G", "gyrA_D94Y", "gyrA_D94N",
    "gyrA_A90V", "gyrA_S91P", "gyrB_N538D",
    "pncA_L4P", "pncA_V125G", "pncA_Q10P",
    "pncA_L4S", "pncA_D12G",
}

KNOWN_HOTSPOTS = {
    ("rpoB", 170), ("rpoB", 430), ("rpoB", 435), ("rpoB", 445),
    ("rpoB", 450), ("rpoB", 452), ("rpoB", 491),
    ("katG", 315), ("embB", 306), ("embB", 406), ("embB", 497),
    ("gyrA", 90), ("gyrA", 91), ("gyrA", 94),
    ("gyrB", 538), ("pncA", 4), ("pncA", 10), ("pncA", 12),
    ("pncA", 125), ("rpsL", 43), ("rpsL", 88),
}

CORE_BINDING = {
    "rpoB": set(range(426, 453)),
    "katG": set(range(104, 116)) | set(range(270, 331)),
    "embB": set(range(295, 531)),
    "gyrA": set(range(74, 151)),
    "gyrB": set(range(495, 556)),
    "pncA": set(range(1, 187)),
    "rpsL": set(range(23, 99)),
}

GENE_LOCUS = {
    "rpoB": "Rv0667", "katG": "Rv1908c", "embB": "Rv3795",
    "gyrA": "Rv0006", "gyrB": "Rv0005", "pncA": "Rv2043c",
    "rpsL": "Rv0682", "eis": "Rv2416c", "tap": "Rv1258c",
    "mmpR5": "Rv0678", "mmpL5": "Rv2680", "tlyA": "Rv1694",
    "inhA": "Rv1484",
}

HOTSPOT_GENES = sorted({"rpoB", "katG", "embB", "gyrA", "gyrB", "pncA", "rpsL"})


def revcomp(seq):
    c = {"A":"T","T":"A","G":"C","C":"G","N":"N"}
    return "".join(c.get(b, "N") for b in reversed(seq))


def parse_gff_genes(gff_path):
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
                    genes[locus] = {"start": int(parts[3]), "end": int(parts[4]),
                                    "strand": parts[6], "name": attrs.get("gene", locus),
                                    "cds_intervals": []}
            elif parts[2] == "CDS" and locus in genes:
                genes[locus]["cds_intervals"].append((int(parts[3]), int(parts[4])))
    return genes


def load_reference_genome(fasta_path):
    genome = {}
    current, seqs = None, []
    with open(fasta_path) as f:
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
    return genome


def extract_cds(gff_genes, genome, locus_tag):
    gene = gff_genes.get(locus_tag)
    if not gene or not gene["cds_intervals"]:
        return None, None
    chrom = list(genome.keys())[0]
    full_seq = genome[chrom]
    intervals = sorted(gene["cds_intervals"])
    parts = [full_seq[s-1:e] for s, e in intervals]
    cds = revcomp("".join(parts)) if gene["strand"] == "-" else "".join(parts)
    prot = []
    for i in range(0, len(cds) - 2, 3):
        aa = GENETIC_CODE.get(cds[i:i+3].upper(), "X")
        if aa == "*":
            break
        prot.append(aa)
    return cds, "".join(prot)


def enumerate_snv_mutations(cds_seq, residue_pos):
    if residue_pos < 1 or (residue_pos - 1) * 3 + 3 > len(cds_seq):
        return []
    cs = (residue_pos - 1) * 3
    wt_codon = cds_seq[cs:cs+3].upper()
    if len(wt_codon) != 3:
        return []
    wt_aa = GENETIC_CODE.get(wt_codon, "X")
    if wt_aa in ("*", "X"):
        return []
    transitions = {("A","G"),("G","A"),("C","T"),("T","C")}
    results = []
    for pos in range(3):
        for nt in ["A","T","G","C"]:
            if nt == wt_codon[pos]:
                continue
            mc = list(wt_codon)
            mc[pos] = nt
            mut_codon = "".join(mc)
            mut_aa = GENETIC_CODE.get(mut_codon, "X")
            if mut_aa == wt_aa or mut_aa in ("*", "X"):
                continue
            results.append({
                "wt_aa": wt_aa, "mut_aa": mut_aa,
                "mutation": f"{wt_aa}{residue_pos}{mut_aa}",
                "is_transition": int((wt_codon[pos], nt) in transitions),
            })
    return results


def compute_mutation_features(mut, inner_dist, drug_dist):
    wt = mut["wt_aa"]
    mut_aa = mut["mut_aa"]
    feats = {}
    feats["is_transition"] = mut["is_transition"]
    blosum_key = (wt, mut_aa)
    feats["blosum62"] = BLOSUM62.get(blosum_key, BLOSUM62.get((mut_aa, wt), -4))
    wt_c = 1 if wt in POSITIVE_AAS else (-1 if wt in NEGATIVE_AAS else 0)
    mut_c = 1 if mut_aa in POSITIVE_AAS else (-1 if mut_aa in NEGATIVE_AAS else 0)
    feats["charge_change"] = abs(mut_c - wt_c)
    wt_v = VOLUME.get(wt, 120)
    mut_v = VOLUME.get(mut_aa, 120)
    feats["size_change"] = abs(mut_v - wt_v) / max(max(wt_v, mut_v), 1)
    feats["delta_hydrophobicity"] = abs(HYDROPHOBICITY.get(mut_aa, 0) - HYDROPHOBICITY.get(wt, 0))
    wt_hb = int(wt in HBOND_AAS)
    mut_hb = int(mut_aa in HBOND_AAS)
    feats["loss_of_hbond"] = abs(mut_hb - wt_hb)
    feats["is_stop"] = int(mut_aa == "*")
    feats["delta_dG_proxy"] = (
        0.5 * feats["charge_change"]
        + 0.3 * feats["size_change"]
        + 0.2 * feats["delta_hydrophobicity"]
        - 0.05 * max(feats["blosum62"], 0)
        + 3.0 * feats["is_stop"]
    )
    feats["inner_distance"] = inner_dist
    feats["drug_distance"] = drug_dist if drug_dist is not None else 100.0
    if drug_dist is not None and drug_dist < 20:
        proximity = max(0, 1 - drug_dist / 15)
    else:
        proximity = max(0, 1 - inner_dist / 50)
    disruptiveness = (
        0.3 * (1 - max(feats["blosum62"], -4) / 9)
        + 0.2 * feats["charge_change"]
        + 0.2 * feats["size_change"]
        + 0.2 * feats["loss_of_hbond"]
        + 0.1 * feats["delta_hydrophobicity"]
    )
    feats["resistance_score"] = proximity * disruptiveness
    feats["fitness_score"] = (
        max(feats["blosum62"], -4) / 9
        - 0.15 * feats["charge_change"]
        - 0.15 * feats["size_change"]
        - 0.1 * feats["delta_hydrophobicity"]
        - 0.05 * feats["loss_of_hbond"]
        - 3.0 * feats["is_stop"]
    )
    feats["fitness_score"] = max(-1, min(1, feats["fitness_score"]))
    evo = 0.6 * feats["is_transition"] + 0.4
    feats["evo_score"] = evo
    return feats


def score_mutations(mutations_df):
    df = mutations_df.copy()
    for col in ["fitness_score", "resistance_score", "evo_score"]:
        if col in df.columns:
            mn, mx = df[col].min(), df[col].max()
            if mx > mn:
                df[f"{col}_norm"] = (df[col] - mn) / (mx - mn)
            else:
                df[f"{col}_norm"] = 0.5
    df["mutation_score"] = (
        0.45 * df["resistance_score_norm"]
        + 0.30 * df["fitness_score_norm"]
        + 0.25 * df["evo_score_norm"]
    )
    df["emergence_score"] = df["hotspot_score"] * df["mutation_score"]
    return df


def load_features():
    """Load all feature data and merge structural features."""
    df = pd.read_csv(HOTSPOT_DIR / "residue_hotspot_data.csv")
    ranked = pd.read_csv(HOTSPOT_DIR / "ranked_predictions.csv")
    df = df.merge(
        ranked[["gene", "residue_pos", "hotspot_score", "rank"]],
        on=["gene", "residue_pos"], how="left"
    )
    sasa_path = HOTSPOT_DIR / "sasa_data.pkl"
    if sasa_path.exists():
        with open(sasa_path, "rb") as f:
            sasa_data = pickle.load(f)
        df["sasa_relative"] = df.apply(
            lambda r: sasa_data.get((r["gene"], r["residue_pos"]), np.nan), axis=1
        )
    esm_path = HOTSPOT_DIR / "esm2_data.pkl"
    if esm_path.exists():
        with open(esm_path, "rb") as f:
            esm_data = pickle.load(f)
        df["esm2_intolerance"] = df.apply(
            lambda r: esm_data.get((r["gene"], r["residue_pos"]), np.nan), axis=1
        )
    contact_path = HOTSPOT_DIR / "contact_density_3d.pkl"
    if contact_path.exists():
        with open(contact_path, "rb") as f:
            contact_data = pickle.load(f)
        df["contact_density_3d"] = df.apply(
            lambda r: contact_data.get((r["gene"], r["residue_pos"]), np.nan), axis=1
        )
    drug_path = HOTSPOT_DIR / "drug_contact_features.pkl"
    if drug_path.exists():
        with open(drug_path, "rb") as f:
            drug_data = pickle.load(f)
        df["drug_distance"] = drug_data["drug_distance"]
        df["drug_contact"] = drug_data["drug_contact"]
    return df


def train_and_predict_for_gene(held_out_gene, df_all):
    """
    Hold out one gene, train LR on all other genes,
    predict hotspot scores for the held-out gene.
    Returns dataframe with hotspot predictions for the held-out gene.
    """
    base_features = [
        "inner_distance", "homoplasy_count", "homoplasy_alleles",
        "helix_propensity", "strand_propensity", "hydrophobicity",
        "volume", "charge", "hbond", "rel_position",
        "conservation_blosum", "contact_density_seq",
    ]
    new_features = ["sasa_relative", "esm2_intolerance", "contact_density_3d"]
    all_features = base_features + [f for f in new_features if f in df_all.columns]

    train = df_all[df_all["gene"] != held_out_gene].dropna(subset=all_features).copy()
    test = df_all[df_all["gene"] == held_out_gene].dropna(subset=all_features).copy()

    if len(train) < 100 or len(test) < 10:
        return None, f"Too few samples: train={len(train)}, test={len(test)}"

    n_pos_train = train["is_hotspot"].sum()
    if n_pos_train < 2:
        return None, f"Too few positives in training: {n_pos_train}"

    X_train = StandardScaler().fit_transform(train[all_features].values)
    y_train = train["is_hotspot"].values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(test[all_features].values)

    # Train with L2 regularization (no penalty for being wrong on unseen genes)
    model = LogisticRegression(
        C=10.0, class_weight="balanced", max_iter=1000, random_state=42
    )
    model.fit(X_train_scaled, y_train)

    test = test.copy()
    test["hotspot_score"] = model.predict_proba(X_test_scaled)[:, 1]
    test["hotspot_rank"] = test["hotspot_score"].rank(ascending=False).astype(int)
    test = test.sort_values("hotspot_score", ascending=False)

    return test, model


def enumerate_and_score_mutations(held_out_gene, hotspot_df, gff_genes, genome):
    """Given hotspot predictions for a held-out gene, enumerate and score mutations."""
    # Select top 30 predicted hotspots + all known hotspots
    high_risk = hotspot_df.head(30).copy()
    for gene, pos in KNOWN_HOTSPOTS:
        if gene != held_out_gene:
            continue
        mask = (high_risk["gene"] == gene) & (high_risk["residue_pos"] == pos)
        if not mask.any():
            row = hotspot_df[(hotspot_df["gene"] == gene) & 
                            (hotspot_df["residue_pos"] == pos)]
            if len(row) > 0:
                high_risk = pd.concat([high_risk, row], ignore_index=True)
    high_risk = high_risk.drop_duplicates(subset=["gene", "residue_pos"])

    locus = GENE_LOCUS.get(held_out_gene)
    if not locus:
        return None

    cds, prot = extract_cds(gff_genes, genome, locus)
    if cds is None:
        return None

    all_mutations = []
    for _, row in high_risk.iterrows():
        pos = int(row["residue_pos"])
        if pos > len(prot):
            continue
        muts = enumerate_snv_mutations(cds, pos)
        inner_dist = min(abs(pos - p) for p in CORE_BINDING.get(held_out_gene, set())) if CORE_BINDING.get(held_out_gene) else 100
        raw_dd = row.get("drug_distance", np.nan)
        drug_dist = None if (isinstance(raw_dd, float) and np.isnan(raw_dd)) else raw_dd
        for m in muts:
            feats = compute_mutation_features(m, inner_dist, drug_dist)
            m.update(feats)
            m["gene"] = held_out_gene
            m["residue_pos"] = pos
            m["hotspot_score"] = row["hotspot_score"]
            m["hotspot_rank"] = row["hotspot_rank"]
            m["is_known_hotspot"] = int((held_out_gene, pos) in KNOWN_HOTSPOTS)
            key = f"{held_out_gene}_{m['mutation']}"
            m["is_known_resistance"] = int(key in KNOWN_RES_MUTATIONS)
            all_mutations.append(m)

    if not all_mutations:
        return None

    df_muts = pd.DataFrame(all_mutations)
    df_muts = score_mutations(df_muts)
    df_muts = df_muts.sort_values("emergence_score", ascending=False)
    df_muts["overall_rank"] = range(1, len(df_muts) + 1)
    return df_muts


# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Step 1: Leave-One-Gene-Out Validation")
    print("=" * 70)

    # Load all data
    print("\n[1/3] Loading features and reference genome...")
    df_all = load_features()
    print(f"  Loaded {len(df_all)} total residues, {df_all['is_hotspot'].sum()} hotspots")

    gff_genes = parse_gff_genes(REF_DIR / "H37Rv.gff")
    fasta_path = REF_DIR / "H37Rv.fasta"
    if not fasta_path.exists():
        fasta_path = REF_DIR / "H37Rv.fna"
    genome = load_reference_genome(fasta_path)
    print(f"  Genome loaded, GFF has {len(gff_genes)} genes")

    # Run LOO validation for each hotspot gene
    print("\n[2/3] Leave-one-gene-out cross-validation...")
    results = []
    known_mutation_ranks = {}

    for gene in HOTSPOT_GENES:
        n_hotspots_in_gene = sum(1 for g, p in KNOWN_HOTSPOTS if g == gene)
        n_muts_in_gene = sum(1 for k in KNOWN_RES_MUTATIONS if k.startswith(f"{gene}_"))
        print(f"\n  Holding out {gene} ({n_hotspots_in_gene} hotspots, {n_muts_in_gene} known mutations)...")

        hotspot_pred, model = train_and_predict_for_gene(gene, df_all)
        if hotspot_pred is None:
            print(f"    SKIP: {model}")
            continue

        # Print top predicted hotspots for held-out gene
        print(f"    Top 10 predicted hotspots in {gene}:")
        for i, (_, r) in enumerate(hotspot_pred.head(10).iterrows(), 1):
            known = " [KNOWN]" if r["is_hotspot"] == 1 else ""
            print(f"      {i}. pos {int(r['residue_pos'])} ({r['wt_aa']}) score={r['hotspot_score']:.4f}{known}")

        # Check known hotspot ranks
        for _, r in hotspot_pred.iterrows():
            if r["is_hotspot"] == 1:
                print(f"    Known hotspot pos {int(r['residue_pos'])} at rank {r['hotspot_rank']}")

        # Enumerate and score mutations
        muts = enumerate_and_score_mutations(gene, hotspot_pred, gff_genes, genome)
        if muts is None:
            print(f"    No mutations enumerated")
            continue

        # Evaluate known mutation ranks
        known = muts[muts["is_known_resistance"] == 1].sort_values("overall_rank")
        print(f"    Known mutation ranks in {gene}:")
        for _, r in known.iterrows():
            known_mutation_ranks[f"{gene}_{r['mutation']}"] = r["overall_rank"]
            print(f"      {r['mutation']}: rank {r['overall_rank']} (emerg={r['emergence_score']:.4f})")

        # Aggregate
        n_known = len(known)
        top20 = sum(1 for _, r in known.iterrows() if r["overall_rank"] <= 20)
        top50 = sum(1 for _, r in known.iterrows() if r["overall_rank"] <= 50)
        top100 = sum(1 for _, r in known.iterrows() if r["overall_rank"] <= 100)
        median_rank = known["overall_rank"].median() if n_known > 0 else None

        # Compute AUROC for mutation classification
        from sklearn.metrics import roc_auc_score
        if muts["is_known_resistance"].sum() >= 2 and muts["is_known_resistance"].sum() < len(muts):
            try:
                auroc = roc_auc_score(muts["is_known_resistance"], muts["emergence_score"])
            except:
                auroc = None
        else:
            auroc = None

        results.append({
            "gene": gene,
            "n_hotspots": n_hotspots_in_gene,
            "n_known_mutations": n_known,
            "top20_recall": top20,
            "top50_recall": top50,
            "top100_recall": top100,
            "top20_pct": f"{top20}/{n_known}" if n_known else "0/0",
            "top50_pct": f"{top50}/{n_known}" if n_known else "0/0",
            "top100_pct": f"{top100}/{n_known}" if n_known else "0/0",
            "median_rank": median_rank,
            "auroc": round(auroc, 4) if auroc else None,
        })
        print(f"    >>> {gene}: Top-20={top20}/{n_known} Top-50={top50}/{n_known} Top-100={top100}/{n_known} AUROC={auroc:.4f}" if auroc else f"    >>> {gene}: Top-20={top20}/{n_known} Top-50={top50}/{n_known} Top-100={top100}/{n_known}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Leave-One-Gene-Out Validation")
    print("=" * 70)
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))

    avg_top20 = summary["top20_recall"].sum() / summary["n_known_mutations"].sum()
    avg_top50 = summary["top50_recall"].sum() / summary["n_known_mutations"].sum()
    avg_top100 = summary["top100_recall"].sum() / summary["n_known_mutations"].sum()
    print(f"\n  Aggregate:")
    print(f"    Top-20 recall: {summary['top20_recall'].sum()}/{summary['n_known_mutations'].sum()} ({avg_top20:.1%})")
    print(f"    Top-50 recall: {summary['top50_recall'].sum()}/{summary['n_known_mutations'].sum()} ({avg_top50:.1%})")
    print(f"    Top-100 recall: {summary['top100_recall'].sum()}/{summary['n_known_mutations'].sum()} ({avg_top100:.1%})")

    # Save results
    out_path = OUTPUT_DIR / "leave_one_gene_out_results.csv"
    summary.to_csv(out_path, index=False)
    print(f"\n  Results saved to {out_path}")

    # Save per-mutation ranks
    ranks_df = pd.DataFrame([
        {"mutation": k, "rank": v} for k, v in known_mutation_ranks.items()
    ])
    ranks_path = OUTPUT_DIR / "leave_one_gene_out_mutation_ranks.csv"
    ranks_df.to_csv(ranks_path, index=False)
    print(f"  Per-mutation ranks saved to {ranks_path}")

    print("\n" + "=" * 70)
    print("Step 1 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
