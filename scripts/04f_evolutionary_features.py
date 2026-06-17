"""
Stage 4: Evolutionary Features — Shannon Entropy from MSA

For each resistance gene, queries UniProt for homologous sequences across
Mycobacterium and computes per-residue Shannon entropy. Higher entropy
means more evolutionary flexibility (many amino acids tolerated at that
position), which is a strong predictor of mutational tolerance.
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE / "analysis" / "results" / "hotspot_model"
UNIPROT_IDS = {
    "rpoB": "P9WGY9", "katG": "P9WIE5", "embB": "P9WNL7",
    "gyrA": "P9WG47", "gyrB": "P9WG45", "pncA": "I6XD65",
    "rpsL": "P9WH63", "eis": "P9WFK7", "tap": "P9WJX9",
    "mmpR5": "I6Y8F7", "mmpL5": "P9WJV1", "tlyA": "P9WJ63",
    "inhA": "P9WGR1",
}

MAX_SEQS = 200
MIN_IDENTITY = 0.4


def get_ref_sequence(gene):
    """Get the H37Rv reference sequence for a gene from the residue data."""
    df = pd.read_csv(OUTPUT_DIR / "residue_hotspot_data.csv")
    gdf = df[df["gene"] == gene].sort_values("residue_pos")
    seq = "".join(gdf["wt_aa"].values)
    positions = gdf["residue_pos"].values.tolist()
    return seq, positions


def fetch_homologs_via_blast(query_seq, gene, max_seqs=MAX_SEQS):
    """Fetch homologous sequences using UniProt BLAST API."""
    url = "https://rest.uniprot.org/blast/"
    headers = {"Accept": "application/json"}
    data = {
        "sequence": query_seq,
        "program": "blastp",
        "matrix": "BLOSUM62",
        "alignments": max_seqs,
        "scores": max_seqs,
        "exp": 10,
    }
    try:
        r = requests.post(url, headers=headers, data=data, timeout=60)
        if r.status_code != 200:
            print(f"  WARNING: BLAST for {gene} returned {r.status_code}")
            return []
        job_id = r.json().get("jobId")
        if not job_id:
            return []
        # Poll for results
        poll_url = f"https://rest.uniprot.org/blast/{job_id}"
        for _ in range(30):
            time.sleep(2)
            pr = requests.get(poll_url, headers=headers, timeout=30)
            if pr.status_code == 200:
                data = pr.json()
                if data.get("status") == "FINISHED":
                    break
        else:
            print(f"  WARNING: BLAST for {gene} timed out")
            return []
        # Extract results
        results_url = f"https://rest.uniprot.org/blast/{job_id}/results"
        rr = requests.get(results_url, headers={"Accept": "text/xml"}, timeout=30)
        if rr.status_code != 200:
            print(f"  WARNING: Could not fetch BLAST results for {gene}")
            return []
        # Parse FASTA from results
        seqs = []
        current_id = None
        current_seq = []
        for line in rr.text.split("\n"):
            if line.startswith(">"):
                if current_id and current_seq:
                    seqs.append((current_id, "".join(current_seq)))
                current_id = line[1:].split()[0]
                current_seq = []
            elif line.strip():
                current_seq.append(line.strip())
        if current_id and current_seq:
            seqs.append((current_id, "".join(current_seq)))
        print(f"  Got {len(seqs)} BLAST hits for {gene}")
        return seqs[:max_seqs]
    except Exception as e:
        print(f"  ERROR: BLAST for {gene} failed: {e}")
        return []


def compute_shannon_entropy_from_sequences(ref_seq, homolog_seqs, positions, min_identity=MIN_IDENTITY):
    """Compute per-position Shannon entropy from a set of aligned sequences.
    
    Uses pairwise alignment against the reference to map each homolog
    to the reference coordinate system, then counts amino acid frequencies
    at each reference position.
    """
    from Bio import pairwise2
    from Bio.Seq import Seq
    
    n_ref = len(ref_seq)
    # Count matrix: [position][amino_acid] = count
    counts = {i: {} for i in range(n_ref)}
    total = {i: 0 for i in range(n_ref)}
    
    for seq_id, homolog_seq in homolog_seqs:
        if len(homolog_seq) < n_ref * 0.3 or len(homolog_seq) > n_ref * 3:
            continue
        # Align to reference using global alignment
        aligns = pairwise2.align.globalmx(ref_seq, homolog_seq, 2, -1)
        if not aligns:
            continue
        best = aligns[0]
        ref_aligned = best.seqA
        hom_aligned = best.seqB
        # Count amino acids at aligned reference positions
        ref_idx = 0
        for i in range(len(ref_aligned)):
            if ref_aligned[i] != "-":
                if hom_aligned[i] != "-":
                    aa = hom_aligned[i]
                    counts[ref_idx][aa] = counts[ref_idx].get(aa, 0) + 1
                    total[ref_idx] += 1
                ref_idx += 1
    
    # Compute Shannon entropy per position: H = -sum(p_i * log2(p_i))
    entropy = {}
    coverage = {}
    for i in range(n_ref):
        if total[i] < 3:
            entropy[positions[i]] = np.nan
            coverage[positions[i]] = total[i]
            continue
        freqs = np.array(list(counts[i].values())) / total[i]
        entropy[positions[i]] = float(-np.sum(freqs * np.log2(freqs + 1e-10)))
        coverage[positions[i]] = total[i]
    
    return entropy, coverage


def compute_evolutionary_features():
    """Compute Shannon entropy and gap frequency for each gene."""
    print("=" * 60)
    print("STAGE 4: EVOLUTIONARY FEATURES (MSA Shannon Entropy)")
    print("=" * 60)
    
    all_entropy = {}
    all_coverage = {}
    
    for gene, uniprot_id in UNIPROT_IDS.items():
        print(f"\n[{gene}] Computing MSA features...")
        ref_seq, positions = get_ref_sequence(gene)
        if not ref_seq or len(ref_seq) < 20:
            print(f"  SKIP: reference sequence too short ({len(ref_seq)})")
            continue
        
        # BLAST for homologs
        print(f"  Reference length: {len(ref_seq)}, querying UniProt...")
        homologs = fetch_homologs_via_blast(ref_seq, gene)
        
        if len(homologs) < 5:
            print(f"  WARNING: only {len(homologs)} homologs found, trying broader search")
            # Try with the UniProt ID search as fallback
            try:
                fallback_url = f"https://rest.uniprot.org/uniprotkb/search?query=((gene:{gene})%20AND%20(taxonomy_name:Mycobacterium))&format=fasta&size=50"
                fb = requests.get(fallback_url, timeout=30)
                if fb.status_code == 200:
                    seqs = []
                    for block in fb.text.strip().split("\n>"):
                        lines = block.split("\n")
                        seq_id = lines[0].split("|")[1] if "|" in lines[0] else lines[0]
                        sequence = "".join(lines[1:]).replace("\n", "")
                        if len(sequence) > 20:
                            seqs.append((seq_id, sequence))
                    if len(seqs) > len(homologs):
                        homologs = seqs
                        print(f"  Fallback got {len(homologs)} sequences")
            except Exception as e:
                print(f"  Fallback failed: {e}")
        
        if len(homologs) < 5:
            print(f"  SKIP: insufficient homologs ({len(homologs)})")
            continue
        
        # Compute Shannon entropy
        entropy, coverage = compute_shannon_entropy_from_sequences(ref_seq, homologs, positions)
        n_valid = sum(1 for v in entropy.values() if not np.isnan(v))
        print(f"  Shannon entropy computed for {n_valid}/{len(positions)} positions from {len(homologs)} homologs")
        
        for pos in positions:
            if pos in entropy:
                all_entropy[(gene, pos)] = entropy[pos]
                all_coverage[(gene, pos)] = coverage[pos]
    
    # Save features
    ent_series = pd.Series(all_entropy, name="shannon_entropy")
    cov_series = pd.Series(all_coverage, name="msa_coverage")
    feat_df = pd.DataFrame({"shannon_entropy": ent_series, "msa_coverage": cov_series})
    feat_df.index = pd.MultiIndex.from_tuples(feat_df.index, names=["gene", "residue_pos"])
    feat_df = feat_df.reset_index()
    
    out_path = OUTPUT_DIR / "msa_features.pkl"
    feat_df.to_pickle(out_path)
    
    n_total = len(feat_df.dropna(subset=["shannon_entropy"]))
    n_genes = feat_df["gene"].nunique()
    print(f"\n  Saved MSA features for {n_total} positions across {n_genes} genes")
    print(f"  File: {out_path}")
    return feat_df


def main():
    compute_evolutionary_features()


if __name__ == "__main__":
    main()
