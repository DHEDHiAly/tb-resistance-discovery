"""
Stage 1: Structural Feature Integration for Hotspot Prediction

Computes per-residue structural features from AlphaFold2 models
and benchmarks their impact on hotspot prediction accuracy.

Tasks:
  1. Compute per-residue SASA from AlphaFold2 PDB files
  2. Compute ESM-2 mutation intolerance scores
  3. Compute 3D contact density from PDB files
  4. Validate AlphaFold structures against PDB crystal structures
  5. Benchmark model performance with new features
  6. Generate ranked output with best-performing model
"""

import json
import pickle
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
PDB_DIR = BASE / "data" / "pdb"
ALPHAFOLD_DIR = PDB_DIR / "alphafold"
CRYSTAL_DIR = PDB_DIR / "crystal"
OUTPUT_DIR = BASE / "analysis" / "results" / "hotspot_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ALPHAFOLD_DIR.mkdir(parents=True, exist_ok=True)
CRYSTAL_DIR.mkdir(parents=True, exist_ok=True)

# Gene -> UniProt accession mapping for M. tuberculosis H37Rv
GENE_UNIPROT = {
    "rpoB": "P9WGY9", "katG": "P9WIE5", "embB": "P9WNL7",
    "gyrA": "P9WG47", "gyrB": "P9WG45", "pncA": "I6XD65",
    "rpsL": "P9WH63", "eis": "P9WFK7", "tap": "P9WJX9",
    "mmpR5": "I6Y8F7", "mmpL5": "P9WJV1", "tlyA": "P9WJ63",
    "inhA": "P9WGR1",
}

# Max solvent accessibility (A^2) per amino acid (from Tien et al., 2013)
MAX_ASA = {
    "A": 121.0, "R": 265.0, "N": 187.0, "D": 187.0, "C": 148.0,
    "Q": 214.0, "E": 214.0, "G": 97.0, "H": 216.0, "I": 195.0,
    "L": 191.0, "K": 230.0, "M": 203.0, "F": 228.0, "P": 154.0,
    "S": 143.0, "T": 163.0, "W": 264.0, "Y": 255.0, "V": 165.0,
}


# Utility

def load_existing_data():
    """Load the existing residue-level feature data."""
    path = OUTPUT_DIR / "residue_hotspot_data.csv"
    if not path.exists():
        print("ERROR: Run 04b_hotspot_model.py first to generate residue data.")
        sys.exit(1)
    return pd.read_csv(path)


def get_alphafold_version(uniprot_id):
    """Query AlphaFold API for the latest model version."""
    import urllib.request, json
    url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            if isinstance(data, list) and len(data) > 0:
                pdb_url = data[0].get("pdbUrl", "")
                return pdb_url
    except:
        pass
    return ""


def download_alphafold_pdb(uniprot_id, gene_name):
    """Download AlphaFold2 PDB from EBI. Returns path or None."""
    local_path = ALPHAFOLD_DIR / f"{gene_name}_{uniprot_id}_alphafold.pdb"
    if local_path.exists():
        return local_path
    try:
        import urllib.request
        # Get the correct URL from the API
        pdb_url = get_alphafold_version(uniprot_id)
        if not pdb_url:
            # Fallback to v6
            pdb_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v6.pdb"
        print(f"  Downloading AlphaFold {gene_name} ({uniprot_id})...")
        req = urllib.request.Request(pdb_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
            with open(local_path, "wb") as f:
                f.write(data)
        print(f"    Saved to {local_path} ({len(data)/1024:.0f} KB)")
        return local_path
    except Exception as e:
        print(f"    WARNING: Could not download {gene_name}: {e}")
        return None


def download_pdb_crystal(pdb_code, gene_name):
    """Download PDB crystal structure from RCSB."""
    local_path = CRYSTAL_DIR / f"{pdb_code}_{gene_name}.pdb"
    if local_path.exists():
        return local_path
    try:
        import urllib.request
        url = f"https://files.rcsb.org/download/{pdb_code}.pdb"
        print(f"  Downloading crystal structure {pdb_code} ({gene_name})...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
            with open(local_path, "wb") as f:
                f.write(data)
        print(f"    Saved ({len(data)/1024:.0f} KB)")
        return local_path
    except Exception as e:
        print(f"    WARNING: Could not download {pdb_code}: {e}")
        return None


def extract_pdb_sequence(pdb_path):
    """Extract per-residue information from a PDB file.
    Returns list of dicts: {resid, resname, chain, ca_atom, all_atoms}"""
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("model", pdb_path)
    residues = []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.get_id()[0].startswith("H_"):
                    continue  # skip heteroatoms
                resname = res.get_resname()
                AA_3TO1 = {
                    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
                    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
                    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
                    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
                    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
                }
                aa1 = AA_3TO1.get(resname, "X")
                resid = res.get_id()[1]
                has_ca = "CA" in res
                has_all = len(list(res.get_atoms())) > 0
                residues.append({
                    "resid": resid,
                    "resname": resname,
                    "aa1": aa1,
                    "chain": chain.get_id(),
                    "has_ca": has_ca,
                })
    return residues


# TASK 1: SOLVENT ACCESSIBILITY (SASA)

def task1_compute_sasa():
    """
    Compute per-residue SASA from AlphaFold2 PDB structures.
    Uses Shrake-Rupley algorithm via Bio.PDB.SASA.
    Output: sasa_relative (0-1, 0=buried, 1=fully exposed)
    """
    print("\n" + "=" * 60)
    print("TASK 1: SOLVENT ACCESSIBILITY")
    print("=" * 60)

    sasa_path = OUTPUT_DIR / "sasa_data.pkl"
    if sasa_path.exists():
        print("  Loading cached SASA data...")
        with open(sasa_path, "rb") as f:
            return pickle.load(f)

    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley

    df = load_existing_data()
    genes_with_pdb = set(df["gene"].unique())
    sasa_data = {}  # (gene, resid) -> relative_sasa

    for gene in sorted(genes_with_pdb):
        uniprot = GENE_UNIPROT.get(gene)
        if not uniprot:
            print(f"  WARNING: No UniProt ID for {gene}, skipping")
            continue

        pdb_path = download_alphafold_pdb(uniprot, gene)
        if pdb_path is None:
            continue

        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure(gene, pdb_path)
            sr = ShrakeRupley()
            sr.compute(structure[0], level="R")

            for chain in structure[0]:
                for res in chain:
                    if res.get_id()[0].startswith("H_"):
                        continue
                    resid = res.get_id()[1]
                    resname = res.get_resname()
                    AA_3TO1 = {
                        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
                        "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
                        "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
                        "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
                        "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
                    }
                    aa1 = AA_3TO1.get(resname, "X")
                    sasa_abs = res.sasa
                    max_asa = MAX_ASA.get(aa1, 200)
                    sasa_rel = min(sasa_abs / max_asa, 1.0) if max_asa > 0 else 0.0
                    sasa_data[(gene, resid)] = sasa_rel

            print(f"  {gene}: computed SASA for {len(sasa_data)} residues")
        except Exception as e:
            print(f"  WARNING: SASA failed for {gene}: {e}")

    with open(sasa_path, "wb") as f:
        pickle.dump(sasa_data, f)
    print(f"  SASA data saved ({len(sasa_data)} residues)")
    print("TASK 1 COMPLETE")
    return sasa_data


# TASK 2: ESM-2 MUTATION INTOLERANCE

def task2_compute_esm2_intolerance():
    """
    Compute per-residue mutation intolerance using ESM-2.
    Scores = -log P(wt_aa | masked_context) — higher = more intolerant.
    Uses the native `esm` package (not transformers) for stability.
    """
    print("\n" + "=" * 60)
    print("TASK 2: ESM-2 MUTATION INTOLERANCE")
    print("=" * 60)

    esm_path = OUTPUT_DIR / "esm2_data.pkl"
    if esm_path.exists():
        print("  Loading cached ESM-2 data...")
        with open(esm_path, "rb") as f:
            return pickle.load(f)

    try:
        import torch
        import esm
    except ImportError as e:
        print(f"  ERROR: {e}. Run: pip install fair-esm")
        print("TASK 2 SKIPPED")
        return {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    # Load ESM-2 model
    print("  Loading esm2_t33_650M_UR50D...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    print("  Model loaded")

    # Amino acid token mapping (ESM uses a different tokenizer than HF)
    # The ESM alphabet has tokens: <cls>, <pad>, <eos>, <unk>, A, R, N, D, C, Q, E, G, H, I, L, K, M, F, P, S, T, W, Y, V, <mask>
    # We need to map AA -> token index
    aa_to_idx = {}
    for i, tok in enumerate(alphabet.tok_to_idx):
        aa_to_idx[tok] = i
    # Add standard amino acids
    standard_aas = "ARNDCEQGHILKMFPSTWYV"
    aa_token_ids = {}
    for aa in standard_aas:
        tid = alphabet.tok_to_idx.get(aa, alphabet.tok_to_idx.get("<unk>"))
        aa_token_ids[aa] = tid

    mask_token_id = alphabet.mask_idx

    # Get protein sequences from GFF
    sys.path.insert(0, str(BASE / "scripts"))
    try:
        from importlib import util as import_util
        spec = import_util.spec_from_file_location("rf", BASE / "scripts" / "04_resistance_forecasting.py")
        rf_mod = import_util.module_from_spec(spec)
        spec.loader.exec_module(rf_mod)

        gff_path = BASE / "reference" / "H37Rv.gff"
        ref_path = BASE / "reference" / "H37Rv.fasta"
        if not ref_path.exists():
            ref_path = BASE / "reference" / "H37Rv.fna"

        gff_genes = rf_mod.parse_gff_genes(gff_path)
        genome = rf_mod.load_reference_genome(ref_path)

        gene_loci = [
            ("rpoB", "Rv0667"), ("katG", "Rv1908c"), ("embB", "Rv3795"),
            ("gyrA", "Rv0006"), ("gyrB", "Rv0005"), ("pncA", "Rv2043c"),
            ("rpsL", "Rv0682"), ("eis", "Rv2416c"), ("tap", "Rv1258c"),
            ("mmpR5", "Rv0678"), ("mmpL5", "Rv2680"), ("tlyA", "Rv1694"),
            ("inhA", "Rv1484"),
        ]

        esm_data = {}

        for gene, locus in gene_loci:
            _, prot = rf_mod.extract_cds(gff_genes, genome, locus)
            if not prot:
                print(f"  WARNING: No sequence for {gene}")
                continue

            sequence = str(prot)
            seq_len = len(sequence)
            print(f"  Computing ESM-2 for {gene} ({seq_len} aa)...")

            scores = {}
            with torch.no_grad():
                # Process each position with a masked version
                for pos in range(seq_len):
                    wt_aa = sequence[pos]
                    if wt_aa not in aa_token_ids:
                        continue

                    # Create masked sequence
                    masked_seq = sequence[:pos] + "<mask>" + sequence[pos+1:]

                    # Tokenize using ESM batch converter
                    data = [("protein", masked_seq)]
                    _, _, batch_tokens = batch_converter(data)
                    batch_tokens = batch_tokens.to(device)

                    # Forward pass
                    logits = model(batch_tokens)["logits"][0]

                    # Find mask position (position 1 = <cls>, position 2..seq_len+1 = actual sequence)
                    # The batch converter adds <cls> and <eos> tokens
                    mask_pos = pos + 1  # +1 for <cls> token

                    # Get log P(wt_aa) at the masked position
                    wt_token_id = aa_token_ids[wt_aa]
                    log_probs = torch.log_softmax(logits[mask_pos], dim=-1)
                    wt_log_prob = log_probs[wt_token_id].item()

                    # Intolerance = -log P(wt) (higher = more intolerant)
                    scores[pos + 1] = -wt_log_prob

                    if (pos + 1) % 200 == 0:
                        print(f"    Progress: {pos+1}/{seq_len}")

            for resid, score in scores.items():
                esm_data[(gene, resid)] = score
            print(f"    Done: {len(scores)} residues for {gene}")

    except Exception as e:
        import traceback
        print(f"  WARNING: ESM-2 failed: {e}")
        traceback.print_exc()

    with open(esm_path, "wb") as f:
        pickle.dump(esm_data, f)
    print(f"  ESM-2 data saved ({len(esm_data)} residues)")
    print("TASK 2 COMPLETE")
    return esm_data


# TASK 3: 3D CONTACT DENSITY

def task3_compute_contact_density_3d():
    """
    Compute 3D contact density: for each residue, count C-alpha atoms
    within 8A radius. Captures structural packing density.
    """
    print("\n" + "=" * 60)
    print("TASK 3: 3D CONTACT DENSITY")
    print("=" * 60)

    contact_path = OUTPUT_DIR / "contact_density_3d.pkl"
    if contact_path.exists():
        print("  Loading cached 3D contact density...")
        with open(contact_path, "rb") as f:
            return pickle.load(f)

    from Bio.PDB import PDBParser
    import numpy as np

    df = load_existing_data()
    genes_with_pdb = set(df["gene"].unique())
    contact_data = {}  # (gene, resid) -> neighbor_count

    for gene in sorted(genes_with_pdb):
        uniprot = GENE_UNIPROT.get(gene)
        if not uniprot:
            continue

        pdb_path = download_alphafold_pdb(uniprot, gene)
        if pdb_path is None:
            continue

        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure(gene, pdb_path)

            # Collect all C-alpha coordinates
            ca_coords = {}  # (chain, resid) -> np.array
            for chain in structure[0]:
                for res in chain:
                    if res.get_id()[0].startswith("H_"):
                        continue
                    if "CA" in res:
                        ca_coords[(chain.get_id(), res.get_id()[1])] = res["CA"].get_vector().get_array()

            # For each residue, count neighbors within 8A
            resid_list = list(ca_coords.keys())
            coords = np.array([ca_coords[k] for k in resid_list])

            for i, (chain, resid) in enumerate(resid_list):
                dists = np.linalg.norm(coords - coords[i], axis=1)
                n_neighbors = int(np.sum((dists > 0) & (dists <= 8.0)))
                contact_data[(gene, resid)] = n_neighbors

            print(f"  {gene}: computed contact density for {len(resid_list)} residues")
        except Exception as e:
            print(f"  WARNING: Contact density failed for {gene}: {e}")

    with open(contact_path, "wb") as f:
        pickle.dump(contact_data, f)
    print(f"  3D contact density saved ({len(contact_data)} residues)")
    print("TASK 3 COMPLETE")
    return contact_data


# TASK 4: ALPHAFOLD STRUCTURAL VALIDATION

def task4_validate_alphafold():
    """
    Validate AlphaFold structures against known PDB crystal structures.
    Computes C-alpha RMSD between aligned residues.
    """
    print("\n" + "=" * 60)
    print("TASK 4: ALPHAFOLD STRUCTURAL VALIDATION")
    print("=" * 60)

    from Bio.PDB import PDBParser, Superimposer

    # Crystal structure references: (PDB code, gene, chain_for_target)
    # 5UHB: M. tuberculosis RNA polymerase — rpoB is chain C
    # 2CAS: M. tuberculosis KatG — chain A, but residue numbering offset
    references = [
        ("5UHB", "rpoB", "C"),
        ("2CAS", "katG", "A"),
    ]

    results = {}

    for pdb_code, gene, chain_id in references:
        crystal_path = download_pdb_crystal(pdb_code, gene)
        if crystal_path is None:
            print(f"  {gene}: No crystal structure available")
            continue

        uniprot = GENE_UNIPROT.get(gene)
        if not uniprot:
            continue

        af_path = download_alphafold_pdb(uniprot, gene)
        if af_path is None:
            continue

        try:
            parser = PDBParser(QUIET=True)
            crystal_struct = parser.get_structure(f"{pdb_code}_crystal", crystal_path)
            af_struct = parser.get_structure(f"{gene}_af", af_path)

            # Extract residues with their 3-letter codes for sequence matching
            AA_3TO1 = {
                "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
                "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
                "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
                "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
                "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
            }

            def get_chain_residues(structure, target_chain=None):
                """Return list of (resid, aa1, CA_atom) for target chain."""
                items = []
                for model in structure:
                    for chain in model:
                        if target_chain and chain.get_id() != target_chain:
                            continue
                        for res in chain:
                            if res.get_id()[0].startswith("H_"):
                                continue
                            resname = res.get_resname()
                            aa1 = AA_3TO1.get(resname, "X")
                            if aa1 != "X" and "CA" in res:
                                items.append((res.get_id()[1], aa1, res["CA"]))
                return items

            cry_items = get_chain_residues(crystal_struct, chain_id)
            af_items = get_chain_residues(af_struct, None)

            if len(cry_items) < 10 or len(af_items) < 10:
                print(f"  {gene}: Too few residues (crystal={len(cry_items)}, AF={len(af_items)})")
                continue

            # Build sequences for alignment
            cry_seq = "".join(item[1] for item in cry_items)
            af_seq = "".join(item[1] for item in af_items)

            # Simple sequence alignment: find matching segment via longest common subsequence
            # Since these are the same protein, a sliding window match works
            cry_aligned = []
            af_aligned = []

            # For each residue in the crystal structure, find a position in AF
            # with the same amino acid near the expected location
            af_start_idx = 0
            for cry_idx, (cry_resid, cry_aa, cry_ca) in enumerate(cry_items):
                # Search for matching AA in AF, starting from last matched position
                found = False
                for af_idx in range(af_start_idx, min(af_start_idx + 50, len(af_items))):
                    af_resid, af_aa, af_ca = af_items[af_idx]
                    if af_aa == cry_aa:
                        cry_aligned.append(cry_ca)
                        af_aligned.append(af_ca)
                        af_start_idx = af_idx + 1
                        found = True
                        break
                if not found:
                    # Skip this residue (no match in AF)
                    pass

            if len(cry_aligned) < 10:
                continue

            # Superimpose
            sup = Superimposer()
            sup.set_atoms(cry_aligned, af_aligned)
            sup.apply(af_aligned)

            rmsd = sup.rms
            results[gene] = {
                "pdb_code": pdb_code,
                "n_residues_aligned": int(len(cry_aligned)),
                "rmsd_ca": float(round(rmsd, 3)),
                "validated": bool(rmsd < 2.0),
            }
            print(f"  {gene}: RMSD={rmsd:.3f}A over {len(cry_aligned)} residues " +
                  ("VALIDATED" if rmsd < 2.0 else "WARNING: >2A"))

        except Exception as e:
            import traceback
            print(f"  WARNING: Validation failed for {gene}: {e}")

    # Save report
    report_path = OUTPUT_DIR / "alphafold_validation.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Validation report saved to {report_path}")
    print("TASK 4 COMPLETE")
    return results


# TASK 5: MODEL BENCHMARKING

def task5_benchmark_models():
    """
    Benchmark Logistic Regression, Elastic Net, and Random Forest
    with all available features (existing + SASA + ESM-2 + 3D contact).
    Uses 5-fold stratified cross-validation.
    """
    print("\n" + "=" * 60)
    print("TASK 5: MODEL BENCHMARKING")
    print("=" * 60)

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, average_precision_score

    # Load existing data
    df = load_existing_data()

    # Load new features
    feature_dfs = [df]

    # SASA
    sasa_path = OUTPUT_DIR / "sasa_data.pkl"
    if sasa_path.exists():
        with open(sasa_path, "rb") as f:
            sasa_data = pickle.load(f)
        df["sasa_relative"] = df.apply(
            lambda r: sasa_data.get((r["gene"], r["residue_pos"]), np.nan), axis=1
        )
        print(f"  SASA feature loaded: {df['sasa_relative'].notna().sum()} residues")

    # ESM-2
    esm_path = OUTPUT_DIR / "esm2_data.pkl"
    if esm_path.exists():
        with open(esm_path, "rb") as f:
            esm_data = pickle.load(f)
        df["esm2_intolerance"] = df.apply(
            lambda r: esm_data.get((r["gene"], r["residue_pos"]), np.nan), axis=1
        )
        print(f"  ESM-2 feature loaded: {df['esm2_intolerance'].notna().sum()} residues")

    # 3D contact density
    contact_path = OUTPUT_DIR / "contact_density_3d.pkl"
    if contact_path.exists():
        with open(contact_path, "rb") as f:
            contact_data = pickle.load(f)
        df["contact_density_3d"] = df.apply(
            lambda r: contact_data.get((r["gene"], r["residue_pos"]), np.nan), axis=1
        )
        print(f"  3D contact feature loaded: {df['contact_density_3d'].notna().sum()} residues")

    # Define feature columns
    base_features = [
        "inner_distance", "homoplasy_count", "homoplasy_alleles",
        "helix_propensity", "strand_propensity", "hydrophobicity",
        "volume", "charge", "hbond", "rel_position",
        "conservation_blosum", "contact_density_seq",
    ]
    new_features = ["sasa_relative", "esm2_intolerance", "contact_density_3d"]
    all_features = [f for f in base_features + new_features if f in df.columns]

    print(f"  Using {len(all_features)} features: {all_features}")
    print(f"  Total samples: {len(df)}")
    print(f"  Hotspot positives: {df['is_hotspot'].sum()}")

    # Drop rows with missing features
    df_model = df.dropna(subset=all_features).copy()
    print(f"  Samples with complete features: {len(df_model)}")

    if len(df_model) < 100:
        print("  ERROR: Too few samples")
        return None

    X = df_model[all_features].values
    y = df_model["is_hotspot"].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Models
    models = {
        "LogisticRegression": LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000, random_state=42
        ),
        "ElasticNet": LogisticRegression(
            penalty="elasticnet", solver="saga",
            l1_ratio=0.5, C=1.0, class_weight="balanced",
            max_iter=1000, random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=5,
            class_weight="balanced", random_state=42,
        ),
    }

    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for model_name, model in models.items():
        aurocs, auprcs, top20s = [], [], []

        for train_idx, test_idx in skf.split(X_scaled, y):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]

            auroc = roc_auc_score(y_test, y_prob)
            auprc = average_precision_score(y_test, y_prob)
            # Top-20 recall
            top20 = y_test[np.argsort(y_prob)[::-1][:20]].sum() / max(y_test.sum(), 1)

            aurocs.append(auroc)
            auprcs.append(auprc)
            top20s.append(top20)

        results.append({
            "model": model_name,
            "auroc_mean": np.mean(aurocs),
            "auroc_std": np.std(aurocs),
            "auprc_mean": np.mean(auprcs),
            "auprc_std": np.std(auprcs),
            "top20_mean": np.mean(top20s),
            "top20_std": np.std(top20s),
        })

        print(f"\n  {model_name}:")
        print(f"    AUROC = {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}")
        print(f"    AUPRC = {np.mean(auprcs):.4f} +/- {np.std(auprcs):.4f}")
        print(f"    Top-20 recall = {np.mean(top20s):.4f} +/- {np.std(top20s):.4f}")

        # Feature importance / coefficients
        if hasattr(model, "coef_"):
            coefs = model.coef_[0]
            coef_df = pd.DataFrame({
                "feature": all_features, "coefficient": coefs
            }).sort_values("coefficient", ascending=False)
            print(f"    Top features:")
            for _, row in coef_df.head(5).iterrows():
                print(f"      {row['feature']}: {row['coefficient']:.4f}")
        elif hasattr(model, "feature_importances_"):
            imp_df = pd.DataFrame({
                "feature": all_features, "importance": model.feature_importances_
            }).sort_values("importance", ascending=False)
            print(f"    Top features:")
            for _, row in imp_df.head(5).iterrows():
                print(f"      {row['feature']}: {row['importance']:.4f}")

    # Save results
    results_df = pd.DataFrame(results)
    results_path = OUTPUT_DIR / "results_benchmark.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n  Benchmark results saved to {results_path}")
    print("TASK 5 COMPLETE")

    # Return best model (by AUROC)
    best_idx = np.argmax([r["auroc_mean"] for r in results])
    best_name = results[best_idx]["model"]

    # Retrain best model on full data
    best_model = models[best_name]
    best_model.fit(X_scaled, y)

    return best_model, all_features, scaler, df_model


# TASK 6: RANKED OUTPUT

def task6_generate_ranked_output(model_result):
    """
    Score all residues using the best model and output ranked predictions.
    """
    print("\n" + "=" * 60)
    print("TASK 6: RANKED OUTPUT")
    print("=" * 60)

    if model_result is None:
        print("  ERROR: No model available from Task 5")
        return

    best_model, feature_cols, scaler, df_model = model_result

    # Score all residues
    X_all = scaler.transform(df_model[feature_cols].values)
    df_model["hotspot_score"] = best_model.predict_proba(X_all)[:, 1]

    # Rank
    df_model["rank"] = df_model["hotspot_score"].rank(ascending=False).astype(int)
    df_model = df_model.sort_values("hotspot_score", ascending=False)

    # Build output with all requested columns
    output_cols = [
        "gene", "residue_pos", "wt_aa", "hotspot_score", "is_hotspot",
        "rank"
    ]
    for feat in ["sasa_relative", "esm2_intolerance", "contact_density_3d"]:
        if feat in df_model.columns:
            output_cols.append(feat)
    output_cols.append("inner_distance")
    available = [c for c in output_cols if c in df_model.columns]
    df_out = df_model[available].copy()

    # Rename inner_distance for output
    if "inner_distance" in df_out.columns:
        df_out = df_out.rename(columns={"inner_distance": "distance_to_binding"})

    output_path = OUTPUT_DIR / "ranked_predictions.csv"
    df_out.to_csv(output_path, index=False)
    print(f"  Ranked predictions saved to {output_path} ({len(df_out)} residues)")

    # Print top 30
    print("\n  Top 30 Predicted Hotspot Residues:")
    print(f"  {'Rank':<6} {'Gene':<8} {'Pos':<6} {'AA':<4} {'Score':<10} {'Known':<8} {'SASA':<8} {'ESM-2':<8} {'3D_Contact':<10} {'Dist':<6}")
    print("  " + "-" * 75)
    for i, (_, row) in enumerate(df_out.head(30).iterrows(), 1):
        known_str = "[KNOWN]" if row.get("is_hotspot", 0) == 1 else ""
        sasa = f"{row.get('sasa_relative', 0):.3f}" if "sasa_relative" in df_out.columns else "N/A"
        esm = f"{row.get('esm2_intolerance', 0):.2f}" if "esm2_intolerance" in df_out.columns else "N/A"
        contact = f"{row.get('contact_density_3d', 0):.0f}" if "contact_density_3d" in df_out.columns else "N/A"
        dist = f"{row.get('distance_to_binding', 999):.0f}" if "distance_to_binding" in df_out.columns else "N/A"
        print(f"  {i:<6} {row['gene']:<8} {row['residue_pos']:<6} {row['wt_aa']:<4} {row['hotspot_score']:<10.4f} {known_str:<8} {sasa:<8} {esm:<8} {contact:<10} {dist:<6}")

    print("TASK 6 COMPLETE")


# MAIN

def main():
    print("=" * 60)
    print("Stage 1: Structural Feature Integration")
    print("=" * 60)

    # Pre-download all PDBs
    print("\n[Pre-downloading AlphaFold structures...]")
    for gene, uniprot in GENE_UNIPROT.items():
        download_alphafold_pdb(uniprot, gene)

    # Task 1
    sasa_data = task1_compute_sasa()

    # Task 2
    esm_data = task2_compute_esm2_intolerance()

    # Task 3
    contact_data = task3_compute_contact_density_3d()

    # Task 4
    validation_results = task4_validate_alphafold()

    # Task 5
    model_result = task5_benchmark_models()

    # Task 6
    task6_generate_ranked_output(model_result)

    print("\n" + "=" * 60)
    print("Stage 1 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
