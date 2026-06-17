"""
Phase 4e: Mutation-Level Forecasting via Hotspot Prior

Pipeline:
  1. Load residue-level hotspot predictions (Stage 1 logistic regression)
  2. Filter to high-risk residues (top N by hotspot_score)
  3. Enumerate all SNV-accessible mutations at those residues
  4. Compute per-mutation features (fitness, resistance, evo)
  5. Score P(emergence) = P(hotspot | residue) x P(mutation | features)
  6. Generate ranked surveillance watchlist
"""

import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
REF_DIR = BASE / "reference"
RESULTS_DIR = BASE / "analysis" / "results"
OUTPUT_DIR = RESULTS_DIR / "forecasting"
HOTSPOT_DIR = RESULTS_DIR / "hotspot_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Genetic code and biochemistry

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
    ('H','H'):8,('H','I'):-3,('H','L'):-3,('H','K'):-1,('H','M'):-2,
    ('H','F'):-1,('H','P'):-2,('H','S'):-1,('H','T'):-2,('H','W'):-2,
    ('H','Y'):2,('H','V'):-3,
    ('I','I'):4,('I','L'):2,('I','K'):-3,('I','M'):1,('I','F'):0,
    ('I','P'):-3,('I','S'):-2,('I','T'):-1,('I','W'):-3,('I','Y'):-1,
    ('I','V'):3,
    ('L','L'):4,('L','K'):-2,('L','M'):2,('L','F'):0,('L','P'):-3,
    ('L','S'):-2,('L','T'):-1,('L','W'):-2,('L','Y'):-1,('L','V'):1,
    ('K','K'):5,('K','M'):-1,('K','F'):-3,('K','P'):-2,('K','S'):0,
    ('K','T'):-1,('K','W'):-3,('K','Y'):-2,('K','V'):-2,
    ('M','M'):5,('M','F'):0,('M','P'):-2,('M','S'):-1,('M','T'):-1,
    ('M','W'):-1,('M','Y'):-1,('M','V'):1,
    ('F','F'):6,('F','P'):-4,('F','S'):-2,('F','T'):-2,('F','W'):1,
    ('F','Y'):3,('F','V'):-1,
    ('P','P'):7,('P','S'):-1,('P','T'):-1,('P','W'):-4,('P','Y'):-3,
    ('P','V'):-2,
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

# Known resistance mutations (for validation only)
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

# Hotspot residues: 21 known positions
KNOWN_HOTSPOTS = {
    ("rpoB", 170), ("rpoB", 430), ("rpoB", 435), ("rpoB", 445),
    ("rpoB", 450), ("rpoB", 452), ("rpoB", 491),
    ("katG", 315), ("embB", 306), ("embB", 406), ("embB", 497),
    ("gyrA", 90), ("gyrA", 91), ("gyrA", 94),
    ("gyrB", 538), ("pncA", 4), ("pncA", 10), ("pncA", 12),
    ("pncA", 125), ("rpsL", 43), ("rpsL", 88),
}

# Drug-gene mapping for inner_distance
CORE_BINDING = {
    "rpoB": set(range(426, 453)),
    "katG": set(range(104, 116)) | set(range(270, 331)),
    "embB": set(range(295, 531)),
    "gyrA": set(range(74, 151)),
    "gyrB": set(range(495, 556)),
    "pncA": set(range(1, 187)),
    "rpsL": set(range(23, 99)),
}


# STEP 1: LOAD HOTSPOT PREDICTIONS

def load_hotspot_predictions():
    """Load ranked hotspot predictions and feature data."""
    ranked_path = HOTSPOT_DIR / "ranked_predictions.csv"
    feature_path = HOTSPOT_DIR / "residue_hotspot_data.csv"
    
    if not ranked_path.exists():
        print("ERROR: Run 04c_stage1_features.py first")
        sys.exit(1)
    
    ranked = pd.read_csv(ranked_path)
    features = pd.read_csv(feature_path)
    
    # Merge hotspot score and rank
    df = features.merge(
        ranked[["gene", "residue_pos", "hotspot_score", "rank"]],
        on=["gene", "residue_pos"], how="left"
    )
    
    print(f"Loaded {len(df)} residues, {df['is_hotspot'].sum()} known hotspots")
    return df


# STEP 2: ENUMERATE MUTATIONS AT HIGH-RISK RESIDUES

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
                    genes[locus] = {
                        "start": int(parts[3]), "end": int(parts[4]),
                        "strand": parts[6], "name": attrs.get("gene", locus),
                        "cds_intervals": []
                    }
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
    """Return all SNV-accessible mutations at a given position."""
    if residue_pos < 1 or (residue_pos - 1) * 3 + 3 > len(cds_seq):
        return []
    cs = (residue_pos - 1) * 3
    wt_codon = cds_seq[cs:cs+3].upper()
    if len(wt_codon) != 3:
        return []
    wt_aa = GENETIC_CODE.get(wt_codon, "X")
    if wt_aa == "*" or wt_aa == "X":
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
                "codon_position": pos,
            })
    return results


# STEP 3: COMPUTE MUTATION FEATURES

def compute_mutation_features(mut, inner_dist, drug_dist):
    """Compute fitness, resistance, and accessibility scores for a mutation."""
    wt = mut["wt_aa"]
    mut_aa = mut["mut_aa"]
    feats = {}
    
    # Evolutionary accessibility
    feats["is_transition"] = mut["is_transition"]
    
    # Fitness preservation
    blosum_key = (wt, mut_aa)
    feats["blosum62"] = BLOSUM62.get(blosum_key, BLOSUM62.get((mut_aa, wt), -4))
    
    # Charge change
    wt_c = 1 if wt in POSITIVE_AAS else (-1 if wt in NEGATIVE_AAS else 0)
    mut_c = 1 if mut_aa in POSITIVE_AAS else (-1 if mut_aa in NEGATIVE_AAS else 0)
    feats["charge_change"] = abs(mut_c - wt_c)
    
    # Size change (normalized)
    wt_v = VOLUME.get(wt, 120)
    mut_v = VOLUME.get(mut_aa, 120)
    feats["size_change"] = abs(mut_v - wt_v) / max(max(wt_v, mut_v), 1)
    
    # Hydrophobicity change
    feats["delta_hydrophobicity"] = abs(HYDROPHOBICITY.get(mut_aa, 0) - HYDROPHOBICITY.get(wt, 0))
    
    # H-bond capacity change
    wt_hb = int(wt in HBOND_AAS)
    mut_hb = int(mut_aa in HBOND_AAS)
    feats["loss_of_hbond"] = abs(mut_hb - wt_hb)
    
    # Stop codon
    feats["is_stop"] = int(mut_aa == "*")
    
    # ΔΔG proxy (fitness cost estimate)
    feats["delta_dG_proxy"] = (
        0.5 * feats["charge_change"]
        + 0.3 * feats["size_change"]
        + 0.2 * feats["delta_hydrophobicity"]
        - 0.05 * max(feats["blosum62"], 0)
        + 3.0 * feats["is_stop"]
    )
    
    # Grantham distance
    # Using a simple physicochemical distance based on available properties
    feats["grantham"] = feats["blosum62"] * -1  # rough inverse: lower blosum = larger distance
    
    # Resistance potential
    feats["inner_distance"] = inner_dist
    feats["drug_distance"] = drug_dist if drug_dist is not None else 100.0
    
    # Composite: resistance plausibility
    # Close to drug + disruptive mutation = more likely to confer resistance
    # Use 3D drug_distance preferentially when available (from docking / co-crystal)
    if drug_dist is not None and drug_dist < 20:
        proximity = max(0, 1 - drug_dist / 15)  # 1 at 0A, ~0.73 at 4A, 0 at 15A+
    else:
        proximity = max(0, 1 - inner_dist / 50)  # sequence-based fallback
    
    disruptiveness = (
        0.3 * (1 - max(feats["blosum62"], -4) / 9)  # normalized 0-1, 1 = radical
        + 0.2 * feats["charge_change"]
        + 0.2 * feats["size_change"]
        + 0.2 * feats["loss_of_hbond"]
        + 0.1 * feats["delta_hydrophobicity"]
    )
    feats["resistance_score"] = proximity * disruptiveness
    
    # Combined scores
    # Fitness score (higher = better fitness)
    feats["fitness_score"] = (
        max(feats["blosum62"], -4) / 9  # normalized 0-1
        - 0.15 * feats["charge_change"]
        - 0.15 * feats["size_change"]
        - 0.1 * feats["delta_hydrophobicity"]
        - 0.05 * feats["loss_of_hbond"]
        - 3.0 * feats["is_stop"]
    )
    feats["fitness_score"] = max(-1, min(1, feats["fitness_score"]))
    
    # Evolutionary accessibility score
    evo = 0.6 * feats["is_transition"] + 0.4  # 0.4 for transversion, 1.0 for transition
    feats["evo_score"] = evo
    
    return feats


# STEP 4: SCORE MUTATIONS

def score_mutations(mutations_df):
    """
    Score each mutation for P(emergence).
    
    P(emergence) = P(hotspot | residue) x P(mutation | features)
    
    P(mutation | features) combines:
      - resistance_score: likelihood mutation confers resistance
        (proximity to drug + biochemical disruption)
      - fitness_score: ability to preserve protein function  
        (small, conservative changes near drug-binding site)
      - evo_score: evolutionary accessibility (transition > transversion)
    
    Known resistance mutations span conservative (S450L, K43R) to radical
    (D435Y, V170F). The scoring must balance: radical changes can confer
    strong resistance but may carry fitness costs.
    """
    df = mutations_df.copy()
    
    # Normalize scores to 0-1
    for col in ["fitness_score", "resistance_score", "evo_score"]:
        if col in df.columns:
            mn, mx = df[col].min(), df[col].max()
            if mx > mn:
                df[f"{col}_norm"] = (df[col] - mn) / (mx - mn)
            else:
                df[f"{col}_norm"] = 0.5
    
    # P(mutation | hotspot) as weighted combination
    # resistance_score gets highest weight — the mutation must plausibly
    # disrupt drug binding. Radical mutations at drug-contact positions
    # score highly here even if fitness cost is high.
    df["mutation_score"] = (
        0.45 * df["resistance_score_norm"]
        + 0.30 * df["fitness_score_norm"]
        + 0.25 * df["evo_score_norm"]
    )
    
    # P(emergence) = P(hotspot) x P(mutation | features)
    df["emergence_score"] = df["hotspot_score"] * df["mutation_score"]
    
    return df


# MAIN PIPELINE

def main():
    print("=" * 60)
    print("Phase 4e: Mutation-Level Forecasting")
    print("=" * 60)
    
    # Step 1: Load hotspot predictions
    print("\n[1/4] Loading hotspot predictions...")
    hotspot_df = load_hotspot_predictions()
    
    # Step 2: Load reference genome data
    print("\n[2/4] Loading reference genome and GFF...")
    gff_path = REF_DIR / "H37Rv.gff"
    fasta_path = REF_DIR / "H37Rv.fasta"
    if not fasta_path.exists():
        fasta_path = REF_DIR / "H37Rv.fna"
    
    gff_genes = parse_gff_genes(gff_path)
    genome = load_reference_genome(fasta_path)
    
    # Gene -> locus mapping for CDS extraction
    GENE_LOCUS = {
        "rpoB": "Rv0667", "katG": "Rv1908c", "embB": "Rv3795",
        "gyrA": "Rv0006", "gyrB": "Rv0005", "pncA": "Rv2043c",
        "rpsL": "Rv0682", "eis": "Rv2416c", "tap": "Rv1258c",
        "mmpR5": "Rv0678", "mmpL5": "Rv2680", "tlyA": "Rv1694",
        "inhA": "Rv1484",
    }
    
    # Step 3: Select high-risk residues and enumerate mutations
    print("\n[3/4] Enumerating mutations at high-risk residues...")
    
    # Take top 50 residues by hotspot score + all known hotspots
    high_risk = hotspot_df.nlargest(50, "hotspot_score")
    
    # Ensure all known hotspots are included
    for gene, pos in KNOWN_HOTSPOTS:
        mask = (high_risk["gene"] == gene) & (high_risk["residue_pos"] == pos)
        if not mask.any():
            row = hotspot_df[(hotspot_df["gene"] == gene) & 
                             (hotspot_df["residue_pos"] == pos)]
            if len(row) > 0:
                high_risk = pd.concat([high_risk, row], ignore_index=True)
    
    high_risk = high_risk.drop_duplicates(subset=["gene", "residue_pos"])
    print(f"  High-risk residues: {len(high_risk)}")
    print(f"  Known hotspots included: {high_risk['is_hotspot'].sum()} / 21")
    
    # Enumerate mutations
    all_mutations = []
    for _, row in high_risk.iterrows():
        gene = row["gene"]
        pos = int(row["residue_pos"])
        locus = GENE_LOCUS.get(gene)
        if not locus:
            continue
        
        cds, prot = extract_cds(gff_genes, genome, locus)
        if cds is None:
            continue
        
        # Verify position
        if pos > len(prot):
            continue
        
        muts = enumerate_snv_mutations(cds, pos)
        
        # Compute features for each mutation
        inner_dist = min(abs(pos - p) for p in CORE_BINDING.get(gene, set())) if CORE_BINDING.get(gene) else 100
        raw_dd = row.get("drug_distance", np.nan)
        drug_dist = None if (isinstance(raw_dd, float) and np.isnan(raw_dd)) else raw_dd
        
        for m in muts:
            feats = compute_mutation_features(m, inner_dist, drug_dist)
            m.update(feats)
            m["gene"] = gene
            m["residue_pos"] = pos
            m["locus"] = locus
            m["hotspot_score"] = row["hotspot_score"]
            m["rank"] = row["rank"]
            m["is_known_hotspot"] = int((gene, pos) in KNOWN_HOTSPOTS)
            
            key = f"{gene}_{m['mutation']}"
            m["is_known_resistance"] = int(key in KNOWN_RES_MUTATIONS)
            
            all_mutations.append(m)
    
    df_muts = pd.DataFrame(all_mutations)
    print(f"  Total SNV mutations enumerated: {len(df_muts)}")
    print(f"  Known resistance mutations included: {df_muts['is_known_resistance'].sum()} / {len(KNOWN_RES_MUTATIONS)}")
    
    # Step 4: Score and rank
    print("\n[4/4] Scoring and ranking mutations...")
    df_muts = score_mutations(df_muts)
    df_muts = df_muts.sort_values("emergence_score", ascending=False)
    df_muts["overall_rank"] = range(1, len(df_muts) + 1)
    
    # Validation: known resistance mutations in top N
    known = df_muts[df_muts["is_known_resistance"] == 1].copy()
    print(f"\n  Validation: Known resistance mutations:")
    print(f"  {'Mutation':<20} {'Gene':<8} {'Rank':<8} {'Score':<10}")
    print("  " + "-" * 50)
    for _, r in known.sort_values("overall_rank").iterrows():
        print(f"  {r['mutation']:<20} {r['gene']:<8} {r['overall_rank']:<8} {r['emergence_score']:<10.4f}")
    
    # Top-20 recall for known mutations
    top20 = df_muts.head(20)
    n_known_in_top20 = top20["is_known_resistance"].sum()
    print(f"\n  Known resistance mutations in top 20: {n_known_in_top20} / {len(known)}")
    print(f"  Known resistance mutations in top 50: {df_muts.head(50)['is_known_resistance'].sum()} / {len(known)}")
    print(f"  Known resistance mutations in top 100: {df_muts.head(100)['is_known_resistance'].sum()} / {len(known)}")
    
    # Output: Surveillance watchlist
    # Add status columns for clarity
    df_muts["status_known_who"] = df_muts["is_known_resistance"]
    df_muts["status_novel"] = 1 - df_muts["is_known_resistance"]
    # Tier assignment placeholder (populated by CRyPTIC validation)
    df_muts["tier_cryptic"] = 0

    output_cols = [
        "overall_rank", "gene", "residue_pos", "wt_aa", "mut_aa", "mutation",
        "emergence_score", "hotspot_score", "mutation_score",
        "fitness_score", "resistance_score", "evo_score",
        "blosum62", "charge_change", "size_change", "loss_of_hbond",
        "delta_dG_proxy", "inner_distance", "is_transition",
        "is_known_hotspot", "is_known_resistance",
        "status_known_who", "status_novel",
    ]
    available = [c for c in output_cols if c in df_muts.columns]
    df_out = df_muts[available].copy()
    
    # Round floats
    for c in df_out.select_dtypes(include=[float]).columns:
        df_out[c] = df_out[c].round(4)
    
    out_path = OUTPUT_DIR / "emergence_watchlist.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n  Watchlist saved to {out_path}")
    print(f"  Total candidates: {len(df_out)}")
    
    # Print top 30
    print(f"\n  Top 30 Predicted Emerging Resistance Mutations:")
    print(f"  {'Rank':<6} {'Mutation':<16} {'Gene':<8} {'P(hot)':<10} {'P(mut)':<10} {'P(emerg)':<10} {'Known':<10}")
    print("  " + "-" * 75)
    for _, r in df_out.head(30).iterrows():
        known_str = "[KNOWN]" if r.get("is_known_resistance", 0) else ""
        print(f"  {r['overall_rank']:<6} {r['mutation']:<16} {r['gene']:<8} {r['hotspot_score']:<10.4f} {r['mutation_score']:<10.4f} {r['emergence_score']:<10.4f} {known_str:<10}")
    
    print("\n" + "=" * 60)
    print("Phase 4e complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
