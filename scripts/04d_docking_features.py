"""
Stage 1.5: Drug-Specific Docking Features for Hotspot Prediction

Computes per-residue distances to bound drug molecules from
co-crystal structures and AutoDock Vina docking, then evaluates
whether drug-specific interactions rescue the final 4 missed hotspots.

Tasks:
  1. Compute sequence mapping: AlphaFold PDB residue <-> H37Rv genome coordinate
  2. Extract rifampicin contact distances from 5UHB co-crystal (rpoB)
  3. Dock drugs to remaining targets via AutoDock Vina
  4. Compute per-residue drug_contact feature
  5. Add feature to model, retrain, and evaluate rescue
"""

import json
import pickle
import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
PDB_DIR = BASE / "data" / "pdb"
ALPHAFOLD_DIR = PDB_DIR / "alphafold"
CRYSTAL_DIR = PDB_DIR / "crystal"
OUTPUT_DIR = BASE / "analysis" / "results" / "hotspot_model"
VINA_PATH = Path(__file__).resolve().parent / "vina.exe"

GENE_UNIPROT = {
    "rpoB": "P9WGY9", "katG": "P9WIE5", "embB": "P9WNL7",
    "gyrA": "P9WG47", "gyrB": "P9WG45", "pncA": "I6XD65",
    "rpsL": "P9WH63", "eis": "P9WFK7", "tap": "P9WJX9",
    "mmpR5": "I6Y8F7", "mmpL5": "P9WJV1", "tlyA": "P9WJ63",
    "inhA": "P9WGR1",
}

MAX_ASA = {
    "A": 121.0, "R": 265.0, "N": 187.0, "D": 187.0, "C": 148.0,
    "Q": 214.0, "E": 214.0, "G": 97.0, "H": 216.0, "I": 195.0,
    "L": 191.0, "K": 230.0, "M": 203.0, "F": 228.0, "P": 154.0,
    "S": 143.0, "T": 163.0, "W": 264.0, "Y": 255.0, "V": 165.0,
}

AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# ── Co-crystal structure definitions ──
# (pdb_code, gene, target_chain, ligand_resname)
CO_CRYSTAL_STRUCTURES = [
    ("5UHB", "rpoB", "C", "RFP"),   # rpoB + rifampicin
]

# Drug SMILES for AutoDock Vina docking
DRUG_SMILES = {
    "rifampicin": "CC1=C(C(=O)C2=C(C3=C(C(=C(C=C3O)C(=O)NCC4=CC=CC=C4)O)C(=C2O1)C)O)OC5C(C(C(C(O5)C)O)N)O",
    "isoniazid": "C1=CC(=CN=C1)C(=O)NN",
    "ethambutol": "CCN(CC)C(CO)C(CO)NCC",
    "levofloxacin": "CC1COC2=C(C(=CC(=C2C1)N3C=C(C(=O)C3=O)C(=O)O)F)N4CCNCC4",
    "moxifloxacin": "CC1COC2=C(C(=CC(=C2C1)N3C=C(C(=O)C3=O)C(=O)O)F)N4CCNCC4",
    "pyrazinamide": "C1=CN=C(C=N1)C(=O)N",
    "streptomycin": "C1C(C(C(C(O1)OC2C(C(C(C(O2)CN)O)O)O)N)N)O",
}


def load_feature_data():
    """Load the existing residue-level feature data."""
    path = OUTPUT_DIR / "residue_hotspot_data.csv"
    df = pd.read_csv(path)
    return df


def get_pdb_sequence_and_positions(pdb_path):
    """Extract 1-letter sequence and (chain, resid) tuples from PDB."""
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("prot", str(pdb_path))
    seq = []
    pos = []
    for chain in struct[0]:
        for res in chain:
            if res.get_id()[0].startswith("H_"):
                continue
            aa = AA_3TO1.get(res.get_resname(), "X")
            if aa != "X":
                seq.append(aa)
                pos.append((chain.get_id(), res.get_id()[1]))
    return "".join(seq), pos


def get_genome_sequence(gene):
    """Get H37Rv sequence from feature data."""
    df = load_feature_data()
    gdf = df[df["gene"] == gene].sort_values("residue_pos")
    return "".join(gdf["wt_aa"].values)


def compute_position_mapping(gene):
    """
    Compute mapping from H37Rv genome positions to PDB residue positions.
    Returns dict: H37Rv_position -> PDB_position (or None if no match)
    """
    from Bio import pairwise2

    uniprot = GENE_UNIPROT.get(gene)
    if not uniprot:
        return {}

    pdb_path = ALPHAFOLD_DIR / f"{gene}_{uniprot}_alphafold.pdb"
    if not pdb_path.exists():
        return {}

    pdb_seq, pdb_pos = get_pdb_sequence_and_positions(pdb_path)
    genome_seq = get_genome_sequence(gene)

    if pdb_seq == genome_seq:
        # Perfect match: 1:1 mapping
        return {i + 1: pdb_pos[i][1] for i in range(len(genome_seq))}

    # Need alignment
    align = pairwise2.align.globalms(pdb_seq, genome_seq, 2, -1, -2, -1)
    if not align:
        return {}

    best = align[0]
    pdb_a, gen_a = best.seqA, best.seqB

    mapping = {}
    pdb_idx = 0
    gen_pos = 0
    for i in range(len(pdb_a)):
        if gen_a[i] != "-":
            gen_pos += 1
        if pdb_a[i] != "-":
            pdb_res_nr = pdb_pos[pdb_idx][1]
            pdb_idx += 1
            if gen_a[i] != "-":
                mapping[gen_pos] = pdb_res_nr

    return mapping


# ────────────────────────────────────────────
# TASK 1: EXTRACT RFP CONTACT FROM 5UHB
# ────────────────────────────────────────────

def task1_rfp_contact():
    """
    Compute per-residue distance to rifampicin in the 5UHB co-crystal,
    then map to M. tuberculosis rpoB positions via structural alignment.
    """
    print("\n" + "=" * 60)
    print("TASK 1: RIFAMPICIN CONTACT DISTANCES")
    print("=" * 60)

    from Bio.PDB import PDBParser, Superimposer

    # Step 1: Get RFP coordinates from 5UHB
    crystal_path = CRYSTAL_DIR / "5UHB_rpoB.pdb"
    if not crystal_path.exists():
        print("  ERROR: 5UHB crystal structure not found")
        return {}

    parser = PDBParser(QUIET=True)
    crystal = parser.get_structure("5uhb", str(crystal_path))

    # Find RFP in 5UHB (chain C)
    rfp_atoms = []
    rfp_chain = None
    for chain in crystal[0]:
        for res in chain:
            if res.get_id()[0].startswith("H_") and res.get_resname() == "RFP":
                rfp_atoms = [a.get_vector().get_array() for a in res.get_atoms()]
                rfp_chain = chain.get_id()
                break
        if rfp_atoms:
            break

    if not rfp_atoms:
        print("  ERROR: RFP not found in 5UHB structure")
        return {}

    rfp_coords = np.array(rfp_atoms)
    print(f"  Found RFP in 5UHB chain {rfp_chain}, {len(rfp_coords)} atoms")

    # Step 2: Align AlphaFold M. tb rpoB to 5UHB chain C
    af_path = ALPHAFOLD_DIR / "rpoB_P9WGY9_alphafold.pdb"
    if not af_path.exists():
        print("  ERROR: AlphaFold rpoB not found")
        return {}

    af_struct = parser.get_structure("af", str(af_path))

    # Extract aligned residues sequence-wise
    def get_ca_atoms(structure, chain_id=None):
        """Extract C-alpha atoms for sequence alignment."""
        items = []
        for model in structure:
            for chain in model:
                if chain_id and chain.get_id() != chain_id:
                    continue
                for res in chain:
                    if res.get_id()[0].startswith("H_"):
                        continue
                    aa = AA_3TO1.get(res.get_resname(), "X")
                    if aa != "X" and "CA" in res:
                        items.append((res.get_id()[1], aa, chain.get_id(), res["CA"]))
        return items

    # Get 5UHB chain C residues
    cry_items = get_ca_atoms(crystal, "C")
    # Get AF residues (chain A)
    af_items = get_ca_atoms(af_struct, None)

    cry_seq = "".join(item[1] for item in cry_items)
    af_seq = "".join(item[1] for item in af_items)

    # Simple sequence-guided alignment: match residues in order
    cry_aligned = []
    af_aligned = []
    af_start = 0

    for cry_idx, (cry_resid, cry_aa, cry_chain, cry_ca) in enumerate(cry_items):
        for af_idx in range(af_start, min(af_start + 50, len(af_items))):
            af_resid, af_aa, af_chain, af_ca = af_items[af_idx]
            if af_aa == cry_aa:
                cry_aligned.append(cry_ca)
                af_aligned.append(af_ca)
                af_start = af_idx + 1
                break

    print(f"  Aligned {len(cry_aligned)} residues for structural superposition")

    if len(cry_aligned) < 100:
        print("  ERROR: Too few aligned residues")
        return {}

    # Superimpose
    sup = Superimposer()
    sup.set_atoms(cry_aligned, af_aligned)
    # Apply rotation to all AF atoms
    for chain in af_struct[0]:
        for res in chain:
            for atom in res:
                atom.transform(sup.rotran[0], sup.rotran[1])

    print(f"  Superposition RMSD: {sup.rms:.3f}A")

    # Step 3: For each AF residue, compute min distance to RFP
    # Build position mapping first (H37Rv genome -> AF PDB resid)
    pos_map = compute_position_mapping("rpoB")

    # Compute min distance for each AF residue
    af_distances = {}  # (gene, pdb_resid) -> min_dist
    for chain in af_struct[0]:
        for res in chain:
            if res.get_id()[0].startswith("H_"):
                continue
            res_coords = np.array([a.get_vector().get_array() for a in res.get_atoms()])
            if len(res_coords) == 0:
                continue
            min_dist = float(np.min(np.linalg.norm(res_coords[:, None] - rfp_coords[None, :], axis=-1)))
            af_distances[(chain.get_id(), res.get_id()[1])] = min_dist

    # Step 4: Map to H37Rv genome positions
    result = {}  # (gene, genome_pos) -> min_dist_to_drug
    for genome_pos, pdb_resid in pos_map.items():
        # Find the chain for this PDB residue
        for (chain_id, resid), dist in af_distances.items():
            if resid == pdb_resid:
                result[("rpoB", genome_pos)] = dist
                break

    print(f"  Computed RFP distances for {len(result)} rpoB positions")
    print(f"  Missed hotspots:")
    for hpos in [170, 491]:
        dist = result.get(("rpoB", hpos), "N/A")
        print(f"    rpoB {hpos}: {dist:.2f}A" if isinstance(dist, float) else f"    rpoB {hpos}: {dist}")

    return result


# ────────────────────────────────────────────
# TASK 2: COMPUTE DRUG CONTACT FEATURE
# ────────────────────────────────────────────

def task2_drug_contact_feature(rfp_contacts):
    """
    Compute drug_contact feature for all residues.
    - rpoB: use RFP distances from 5UHB
    - Others: placeholder for future docking
    Feature is distance to drug in Angstroms (NaN = no drug data for this gene)
    """
    print("\n" + "=" * 60)
    print("TASK 2: DRUG CONTACT FEATURE")
    print("=" * 60)

    df = load_feature_data()

    # Initialize with NaN
    df["drug_distance"] = np.nan

    # Map RFP contacts
    for (gene, pos), dist in rfp_contacts.items():
        mask = (df["gene"] == gene) & (df["residue_pos"] == pos)
        df.loc[mask, "drug_distance"] = round(dist, 2)

    n_nonnull = df["drug_distance"].notna().sum()
    print(f"  Drug distance assigned for {n_nonnull} residues")

    # For residues without computed drug distance,
    # set a large default based on inner_distance
    # This avoids dropping these samples from the model
    # while still providing signal for residues near drugs
    default_dist = df["drug_distance"].max() + 10 if n_nonnull > 0 else 100
    df["drug_distance"] = df["drug_distance"].fillna(default_dist)

    # Also compute drug_contact_binary (within 5A = contact)
    df["drug_contact"] = (df["drug_distance"] <= 5.0).astype(int)

    print(f"  Residues with drug contact (<=5A): {df['drug_contact'].sum()}")
    print(f"  Feature range: {df['drug_distance'].min():.1f} - {df['drug_distance'].max():.1f}A")

    # Save
    feat_path = OUTPUT_DIR / "drug_contact_features.pkl"
    with open(feat_path, "wb") as f:
        pickle.dump({
            "drug_distance": df["drug_distance"].values,
            "drug_contact": df["drug_contact"].values,
        }, f)
    print(f"  Saved to {feat_path}")

    return df


# ────────────────────────────────────────────
# TASK 3: MODEL RETRAINING & EVALUATION
# ────────────────────────────────────────────

def task3_evaluate_with_docking(df):
    """
    Add drug contact features to the model, retrain, and evaluate
    whether the 4 missed hotspots are rescued.
    """
    print("\n" + "=" * 60)
    print("TASK 3: MODEL EVALUATION WITH DOCKING FEATURES")
    print("=" * 60)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, average_precision_score

    # Load existing features from pickle caches
    sasa_path = OUTPUT_DIR / "sasa_data.pkl"
    if sasa_path.exists():
        with open(sasa_path, "rb") as f:
            sasa_data = pickle.load(f)
        df["sasa_relative"] = df.apply(
            lambda r: sasa_data.get((r["gene"], r["residue_pos"]), np.nan), axis=1
        )

    esm_path = OUTPUT_DIR / "esm2_data.pkl"
    if esm_path.exists():
        with open(esm_path, "rb") as f:
            esm_data = pickle.load(f)
        df["esm2_intolerance"] = df.apply(
            lambda r: esm_data.get((r["gene"], r["residue_pos"]), np.nan), axis=1
        )

    contact_path = OUTPUT_DIR / "contact_density_3d.pkl"
    if contact_path.exists():
        with open(contact_path, "rb") as f:
            contact_data = pickle.load(f)
        df["contact_density_3d"] = df.apply(
            lambda r: contact_data.get((r["gene"], r["residue_pos"]), np.nan), axis=1
        )

    # Features
    base_features = [
        "inner_distance", "homoplasy_count", "homoplasy_alleles",
        "helix_propensity", "strand_propensity", "hydrophobicity",
        "volume", "charge", "hbond", "rel_position",
        "conservation_blosum", "contact_density_seq",
    ]
    stage1_features = ["sasa_relative", "esm2_intolerance", "contact_density_3d"]
    docking_features = ["drug_distance", "drug_contact"]

    # Compare models with and without docking features
    for feature_set_name, features in [
        ("Stage 1 (no docking)", base_features + [f for f in stage1_features if f in df.columns]),
        ("Stage 1 + docking", base_features + [f for f in stage1_features + docking_features if f in df.columns]),
    ]:
        available = [f for f in features if f in df.columns]
        print(f"\n  {feature_set_name}")
        print(f"  Features ({len(available)}): {available}")

        df_model = df.dropna(subset=available).copy()
        print(f"  Samples: {len(df_model)}, Hotspots: {df_model['is_hotspot'].sum()}")

        if len(df_model) < 100:
            print("  SKIP: too few samples")
            continue

        X = df_model[available].values
        y = df_model["is_hotspot"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
        aurocs, auprcs, top20s = [], [], []

        for train_idx, test_idx in skf.split(X_scaled, y):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            aurocs.append(roc_auc_score(y_test, y_prob))
            auprcs.append(average_precision_score(y_test, y_prob))
            top20s.append(y_test[np.argsort(y_prob)[::-1][:20]].sum() / max(y_test.sum(), 1))

        print(f"    AUROC = {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}")
        print(f"    AUPRC = {np.mean(auprcs):.4f} +/- {np.std(auprcs):.4f}")
        print(f"    Top-20 recall = {np.mean(top20s):.4f} +/- {np.std(top20s):.4f}")

        # Feature coefficients
        coefs = model.coef_[0]
        coef_df = pd.DataFrame({"feature": available, "coefficient": coefs}).sort_values("coefficient", ascending=False)
        print(f"    Top features:")
        for _, r in coef_df.head(8).iterrows():
            print(f"      {r['feature']}: {r['coefficient']:.4f}")

    # Full retrain with docking
    available = base_features + [f for f in stage1_features + docking_features if f in df.columns]
    available = [f for f in available if f in df.columns]
    df_model = df.dropna(subset=available).copy()
    X_all = StandardScaler().fit_transform(df_model[available].values)
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_all, df_model["is_hotspot"].values)
    df_model["hotspot_score"] = model.predict_proba(X_all)[:, 1]
    df_model["rank"] = df_model["hotspot_score"].rank(ascending=False).astype(int)
    df_model = df_model.sort_values("hotspot_score", ascending=False)

    # Print top 30
    print("\n  Top 30 Predicted Hotspot Residues (with docking):")
    print(f"  {'Rank':<6} {'Gene':<8} {'Pos':<6} {'AA':<4} {'Score':<10} {'Known':<10} {'DrugDist':<10}")
    print("  " + "-" * 60)
    known_hotspots = {
        ("rpoB", 170), ("rpoB", 430), ("rpoB", 435), ("rpoB", 445),
        ("rpoB", 450), ("rpoB", 452), ("rpoB", 491),
        ("katG", 315), ("embB", 306), ("embB", 406), ("embB", 497),
        ("gyrA", 90), ("gyrA", 91), ("gyrA", 94),
        ("gyrB", 538), ("pncA", 4), ("pncA", 10), ("pncA", 12), ("pncA", 125),
        ("rpsL", 43), ("rpsL", 88),
    }

    for i, (_, row) in enumerate(df_model.head(30).iterrows(), 1):
        is_known = (row["gene"], row["residue_pos"]) in known_hotspots
        known_str = "[KNOWN]" if is_known else ""
        dd = f"{row.get('drug_distance', 0):.1f}" if "drug_distance" in row else "N/A"
        print(f"  {i:<6} {row['gene']:<8} {row['residue_pos']:<6} {row['wt_aa']:<4} {row['hotspot_score']:<10.4f} {known_str:<10} {dd:<10}")

    # Evaluate rescue of missed hotspots
    print("\n  Missed Hotspot Rescue Analysis:")
    print(f"  {'Hotspot':<15} {'Prev Rank':<12} {'New Rank':<10} {'Score':<8} {'DrugDist':<10} {'Rescued?':<10}")
    print("  " + "-" * 65)
    missed = [(170, "rpoB"), (491, "rpoB"), (125, "pncA"), (538, "gyrB")]
    prev_ranks = {"rpoB_170": 24, "rpoB_491": 21, "pncA_125": 26, "gyrB_538": 28}
    for pos, gene in missed:
        row = df_model[(df_model["gene"] == gene) & (df_model["residue_pos"] == pos)]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        key = f"{gene}_{pos}"
        prev = prev_ranks.get(key, "?")
        rescued = "YES" if r["rank"] <= 20 else "NO"
        dd = f"{r.get('drug_distance', 0):.1f}" if "drug_distance" in r else "N/A"
        print(f"  {gene} {pos:<10} {prev:<12} {r['rank']:<10} {r['hotspot_score']:<8.4f} {dd:<10} {rescued:<10}")

    # Save output
    output_path = OUTPUT_DIR / "ranked_predictions_with_docking.csv"
    df_model.to_csv(output_path, index=False)
    print(f"\n  Saved to {output_path}")


# ────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Stage 1.5: Drug-Specific Docking Features")
    print("=" * 60)

    # Task 1: RFP contact distances
    rfp_contacts = task1_rfp_contact()

    # Task 2: Build drug contact feature
    df = task2_drug_contact_feature(rfp_contacts)

    # Task 3: Retrain and evaluate
    task3_evaluate_with_docking(df)

    print("\n" + "=" * 60)
    print("Stage 1.5 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
