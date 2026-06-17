"""
Stage 3: Per-Gene Drug Contact Features via Co-Crystal + Pocket Distance

For every resistance gene, computes a per-residue distance to the
drug-binding region. Uses co-crystal structures where available (rpoB,
gyrA) and a pocket-centroid proxy for all other genes. Then retrains
the hotspot model and evaluates the per-gene improvement.

Key change from v1: each gene gets its own drug-distance value instead
of a constant 92.0 A fill for non-rpoB genes.
"""

import json
import pickle
import re
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

GENE_UNIPROT = {
    "rpoB": "P9WGY9", "katG": "P9WIE5", "embB": "P9WNL7",
    "gyrA": "P9WG47", "gyrB": "P9WG45", "pncA": "I6XD65",
    "rpsL": "P9WH63", "eis": "P9WFK7", "tap": "P9WJX9",
    "mmpR5": "I6Y8F7", "mmpL5": "P9WJV1", "tlyA": "P9WJ63",
    "inhA": "P9WGR1",
}

AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# Known binding-pocket residue positions per gene (H37Rv numbering)
# These define the pocket centroid for non-co-crystal genes
POCKET_RESIDUES = {
    "rpoB": list(range(426, 453)),   # RRDR
    "katG": [315],                     # INH activation site
    "embB": [306, 406, 497],          # EMB resistance cluster
    "gyrA": [90, 91, 94],             # FQ binding
    "gyrB": [538],                    # FQ resistance
    "pncA": [4, 7, 8, 9, 10, 11, 12, 13, 20, 49, 51, 71, 85, 96, 125, 133, 138],  # PZA active site core
    "rpsL": [43, 88],                 # STR binding
    "eis": [48, 49, 50, 51, 52, 53, 54, 57, 58, 59, 60, 63, 64, 65, 66,
            74, 75, 78, 85, 86, 87, 88, 92, 93, 94, 98, 99, 100, 103, 104,
            105, 116, 117, 118, 119, 120, 121, 122, 126, 128, 130, 134,
            139, 140, 142, 144, 147, 189, 289, 295, 310, 350],
    "tap": [0],  # membrane protein, no reliable pocket definition
    "mmpR5": [36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
              51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
              65, 66, 67, 68, 69, 70, 71, 72, 82, 89, 90, 91, 92, 93,
              94, 95, 96, 97, 98, 99, 100, 101, 104, 107, 108, 110,
              112, 114, 115, 117, 118, 119, 120, 121, 122, 123, 124,
              125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
              136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146,
              147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157],
    "mmpL5": [0],  # membrane protein, no reliable pocket definition
    "tlyA": [0],   # rRNA methyltransferase; binding pocket uncertain
    "inhA": [14, 16, 17, 18, 21, 65, 66, 67, 68, 94, 95, 96, 97, 98,
             99, 100, 101, 102, 103, 104, 105, 106, 110, 113, 122, 147,
             148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
             160, 161, 162, 163, 164, 165, 172, 173, 174, 175, 176, 177,
             178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189,
             190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201,
             202, 203],
}

# Co-crystal definitions: (pdb_code, gene, chain, ligand_resname)
CO_CRYSTALS = [
    ("5UHB", "rpoB", "C", "RFP"),
    ("5BS8", "gyrA", "A", "MFX"),
]


def load_feature_data():
    path = OUTPUT_DIR / "residue_hotspot_data.csv"
    df = pd.read_csv(path)
    return df


def get_genome_sequence(gene):
    df = load_feature_data()
    gdf = df[df["gene"] == gene].sort_values("residue_pos")
    return "".join(gdf["wt_aa"].values)


def get_pdb_sequence_and_positions(pdb_path):
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


def compute_smith_waterman_mapping(gene):
    """Map H37Rv genome positions to PDB residue IDs via Smith-Waterman."""
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
        return {i + 1: pdb_pos[i][1] for i in range(len(genome_seq))}

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


def extract_co_crystal_distances(gene, pdb_code, chain_id, ligand_resn):
    """Extract per-residue distances to a co-crystallized ligand."""
    from Bio.PDB import PDBParser, Superimposer

    crystal_path = CRYSTAL_DIR / f"{pdb_code}_{gene}.pdb"
    if not crystal_path.exists():
        print(f"  WARNING: {crystal_path} not found")
        return {}

    parser = PDBParser(QUIET=True)
    crystal = parser.get_structure("crystal", str(crystal_path))

    # Find ligand atoms
    lig_atoms = []
    for model in crystal:
        for chain in model:
            for res in chain:
                if res.get_id()[0].startswith("H_") and res.get_resname() == ligand_resn:
                    lig_atoms.extend([a.get_vector().get_array() for a in res.get_atoms()])
                elif res.get_resname() == ligand_resn and not res.get_id()[0].startswith("H_"):
                    lig_atoms.extend([a.get_vector().get_array() for a in res.get_atoms()])

    if not lig_atoms:
        print(f"  WARNING: {ligand_resn} not found in {pdb_code}")
        return {}

    lig_coords = np.array(lig_atoms)
    print(f"  Found {ligand_resn} in {pdb_code}: {len(lig_coords)} atoms")

    # Align AlphaFold structure to crystal
    af_path = ALPHAFOLD_DIR / f"{gene}_{GENE_UNIPROT[gene]}_alphafold.pdb"
    if not af_path.exists():
        return {}

    af_struct = parser.get_structure("af", str(af_path))

    # Get C-alpha atoms from both structures for alignment
    def get_ca_items(structure, chain_filter=None):
        items = []
        for model in structure:
            for chain in model:
                if chain_filter and chain.get_id() != chain_filter:
                    continue
                for res in chain:
                    if res.get_id()[0].startswith("H_"):
                        continue
                    aa = AA_3TO1.get(res.get_resname(), "X")
                    if aa != "X" and "CA" in res:
                        items.append((res.get_id()[1], aa, chain.get_id(), res["CA"]))
        return items

    cry_items = get_ca_items(crystal, chain_id)
    af_items = get_ca_items(af_struct, None)

    cry_seq = "".join(item[1] for item in cry_items)
    af_seq = "".join(item[1] for item in af_items)

    # Smith-Waterman alignment for structural superposition
    from Bio import pairwise2
    align = pairwise2.align.globalms(cry_seq, af_seq, 2, -1, -2, -1)
    if not align:
        print("  WARNING: alignment failed for structural superposition")
        return {}

    best = align[0]
    cry_a, af_a = best.seqA, best.seqB

    cry_aligned_ca = []
    af_aligned_ca = []
    cry_idx = 0
    af_idx = 0
    for i in range(len(cry_a)):
        if cry_a[i] != "-" and af_a[i] != "-":
            cry_aligned_ca.append(cry_items[cry_idx][3])
            af_aligned_ca.append(af_items[af_idx][3])
        if cry_a[i] != "-":
            cry_idx += 1
        if af_a[i] != "-":
            af_idx += 1

    if len(cry_aligned_ca) < 50:
        print(f"  WARNING: too few aligned residues ({len(cry_aligned_ca)})")
        return {}

    # Superpose
    sup = Superimposer()
    sup.set_atoms(cry_aligned_ca, af_aligned_ca)
    for model in af_struct:
        for chain in model:
            for res in chain:
                for atom in res:
                    atom.transform(sup.rotran[0], sup.rotran[1])

    print(f"  Superposition RMSD: {sup.rms:.3f}A over {len(cry_aligned_ca)} residues")

    # Get genome mapping
    pos_map = compute_smith_waterman_mapping(gene)

    # Compute distances
    distances = {}
    for model in af_struct:
        for chain in model:
            for res in chain:
                if res.get_id()[0].startswith("H_"):
                    continue
                res_coords = np.array([a.get_vector().get_array() for a in res.get_atoms()])
                if len(res_coords) == 0:
                    continue
                min_dist = float(np.min(np.linalg.norm(
                    res_coords[:, None] - lig_coords[None, :], axis=-1
                )))
                distances[(chain.get_id(), res.get_id()[1])] = min_dist

    # Map to genome positions
    result = {}
    for genome_pos, pdb_resid in pos_map.items():
        for (ch, resid), dist in distances.items():
            if resid == pdb_resid:
                result[("rpoB" if gene == "rpoB" else gene, genome_pos)] = dist
                break

    print(f"  Computed distances for {len(result)} positions")
    return result


def dilate_pocket(pocket_res, gene, distance_threshold=10.0):
    """Expand pocket residues to include all residues within threshold A of any pocket CA."""
    uniprot = GENE_UNIPROT.get(gene)
    if not uniprot or not pocket_res or pocket_res[0] == 0:
        return pocket_res
    from Bio.PDB import PDBParser
    pdb_path = ALPHAFOLD_DIR / f"{gene}_{uniprot}_alphafold.pdb"
    if not pdb_path.exists():
        return pocket_res
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("af", str(pdb_path))
    pos_map = compute_smith_waterman_mapping(gene)
    pdb_to_genome = {v: k for k, v in pos_map.items()}
    pocket_ca_coords = []
    for model in struct:
        for chain in model:
            for res in chain:
                if res.get_id()[0].startswith("H_"):
                    continue
                resid = res.get_id()[1]
                genome_pos = pdb_to_genome.get(resid)
                if genome_pos in pocket_res and "CA" in res:
                    pocket_ca_coords.append(res["CA"].get_vector().get_array())
    if not pocket_ca_coords:
        return pocket_res
    pocket_coords = np.array(pocket_ca_coords)
    dilated = set(pocket_res)
    for model in struct:
        for chain in model:
            for res in chain:
                if res.get_id()[0].startswith("H_"):
                    continue
                resid = res.get_id()[1]
                genome_pos = pdb_to_genome.get(resid)
                if genome_pos is None or genome_pos in dilated:
                    continue
                if "CA" not in res:
                    continue
                ca = res["CA"].get_vector().get_array()
                if np.min(np.linalg.norm(pocket_coords - ca, axis=1)) <= distance_threshold:
                    dilated.add(genome_pos)
    return sorted(dilated)


def compute_pocket_distances(gene):
    """For genes without co-crystals, compute distance to pocket centroid."""
    from Bio.PDB import PDBParser

    uniprot = GENE_UNIPROT.get(gene)
    pocket_res = POCKET_RESIDUES.get(gene, [])
    if not pocket_res or pocket_res[0] == 0:
        return {}

    # Dilate pocket to include structural neighbors of binding site
    dilated = dilate_pocket(pocket_res, gene, distance_threshold=10.0)
    if len(dilated) > len(pocket_res):
        print(f"  Dilated pocket: {len(pocket_res)} -> {len(dilated)} residues")
    pocket_res = dilated

    af_path = ALPHAFOLD_DIR / f"{gene}_{uniprot}_alphafold.pdb"
    if not af_path.exists():
        return {}

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("af", str(af_path))

    # Get mapping
    pos_map = compute_smith_waterman_mapping(gene)
    # Reverse: pdb_resid -> genome_pos
    pdb_to_genome = {v: k for k, v in pos_map.items()}

    # Find pocket residue CA coordinates
    pocket_ca_coords = []
    for model in struct:
        for chain in model:
            for res in chain:
                if res.get_id()[0].startswith("H_"):
                    continue
                resid = res.get_id()[1]
                # Check if this PDB residue maps to a known pocket position
                genome_pos = pdb_to_genome.get(resid)
                if genome_pos in pocket_res and "CA" in res:
                    pocket_ca_coords.append(res["CA"].get_vector().get_array())

    if not pocket_ca_coords:
        return {}

    pocket_centroid = np.mean(pocket_ca_coords, axis=0)
    print(f"  Pocket centroid from {len(pocket_ca_coords)} residues: {pocket_centroid}")

    # Compute distance of each residue to pocket centroid
    result = {}
    for model in struct:
        for chain in model:
            for res in chain:
                if res.get_id()[0].startswith("H_"):
                    continue
                resid = res.get_id()[1]
                genome_pos = pdb_to_genome.get(resid)
                if genome_pos is None:
                    continue
                if "CA" in res:
                    ca = res["CA"].get_vector().get_array()
                    dist = float(np.linalg.norm(ca - pocket_centroid))
                    result[(gene, genome_pos)] = dist

    # Also compute min heavy-atom distance to any pocket residue
    result_fine = {}
    for model in struct:
        for chain in model:
            for res_target in chain:
                if res_target.get_id()[0].startswith("H_"):
                    continue
                resid = res_target.get_id()[1]
                genome_pos = pdb_to_genome.get(resid)
                if genome_pos is None:
                    continue

                target_coords = np.array([a.get_vector().get_array()
                                          for a in res_target.get_atoms()])
                if len(target_coords) == 0:
                    continue

                # Compute min distance to any pocket residue atom
                min_to_pocket = float("inf")
                for model2 in struct:
                    for chain2 in model2:
                        for res_pocket in chain2:
                            if res_pocket.get_id()[0].startswith("H_"):
                                continue
                            p_resid = res_pocket.get_id()[1]
                            p_genome = pdb_to_genome.get(p_resid)
                            if p_genome is None or p_genome not in pocket_res:
                                continue
                            # Exclude the query residue to avoid self-distance=0 circularity
                            if p_genome == genome_pos:
                                continue
                            pocket_atom_coords = np.array([
                                a.get_vector().get_array() for a in res_pocket.get_atoms()
                            ])
                            if len(pocket_atom_coords) == 0:
                                continue
                            d = np.min(np.linalg.norm(
                                target_coords[:, None] - pocket_atom_coords[None, :], axis=-1
                            ))
                            if d < min_to_pocket:
                                min_to_pocket = d

                if min_to_pocket != float("inf"):
                    result_fine[(gene, genome_pos)] = round(min_to_pocket, 2)

    return result_fine


def compute_plddt_features():
    """Extract pLDDT score (AlphaFold confidence) from each AlphaFold PDB.
    Returns dict: (gene, residue_pos) -> (plddt_score, local_env_score)
    where local_env_score = average pLDDT of residues within 8A in 3D space.
    """
    from Bio.PDB import PDBParser, NeighborSearch
    import warnings
    warnings.filterwarnings("ignore")

    result_plddt = {}
    result_env = {}

    for gene, uniprot in GENE_UNIPROT.items():
        pdb_path = ALPHAFOLD_DIR / f"{gene}_{uniprot}_alphafold.pdb"
        if not pdb_path.exists():
            continue

        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("af", str(pdb_path))
        pos_map = compute_smith_waterman_mapping(gene)

        # Extract pLDDT (stored in B-factor column in AlphaFold PDBs)
        ca_atoms = []
        res_plddt = {}
        for model in struct:
            for chain in model:
                for res in chain:
                    if res.get_id()[0].startswith("H_"):
                        continue
                    resid = res.get_id()[1]
                    genome_pos = pos_map.get(resid)
                    if genome_pos is None:
                        continue
                    bfactor = res.get_list()[0].get_bfactor() if res.get_list() else 0.0
                    res_plddt[(gene, genome_pos)] = round(bfactor, 2)
                    if "CA" in res:
                        ca_atoms.append((res["CA"], gene, genome_pos))

        # Compute local environment score: average pLDDT of neighbors within 8A
        if len(ca_atoms) > 5:
            ca_coords = np.array([a[0].get_vector().get_array() for a in ca_atoms])
            for i, (atom, gene_i, pos_i) in enumerate(ca_atoms):
                atom_coord = atom.get_vector().get_array()
                dists = np.linalg.norm(ca_coords - atom_coord, axis=1)
                neighbors = np.where((dists > 0) & (dists <= 8.0))[0]
                if len(neighbors) > 0:
                    neighbor_plddt = np.mean([res_plddt.get((ca_atoms[j][1], ca_atoms[j][2]), 50.0)
                                              for j in neighbors])
                    result_env[(gene_i, pos_i)] = round(neighbor_plddt, 2)

        for key, val in res_plddt.items():
            result_plddt[key] = val

    plddt_series = pd.Series(result_plddt, name="plddt_score")
    env_series = pd.Series(result_env, name="plddt_environment")
    plddt_df = pd.DataFrame({"plddt_score": plddt_series, "plddt_environment": env_series})
    plddt_df.index = pd.MultiIndex.from_tuples(plddt_df.index, names=["gene", "residue_pos"])
    plddt_df = plddt_df.reset_index()

    # Save
    plddt_df.to_pickle(OUTPUT_DIR / "plddt_data.pkl")
    print(f"  pLDDT features saved for {len(result_plddt)} residues across {len(plddt_df['gene'].unique())} genes")
    return result_plddt, result_env


def proximity_transform(distances, k=10.0):
    """Convert raw drug distance to a bounded proximity score [0,1].
    proximity = 1 / (1 + d/k)
    At d=0: proximity=1.0, at d=k: proximity=0.5, at d=3k: proximity=0.25
    """
    return {key: round(1.0 / (1.0 + dist / k), 4) for key, dist in distances.items()}


def compute_all_drug_distances():
    """Compute drug-distance feature for every gene."""
    all_distances = {}

    # Co-crystal genes
    for pdb_code, gene, chain, ligand_resn in CO_CRYSTALS:
        print(f"\n  [{gene}] Co-crystal {pdb_code} + {ligand_resn}")
        dists = extract_co_crystal_distances(gene, pdb_code, chain, ligand_resn)
        all_distances.update(dists)

    # Pocket-distance genes (all 13 genes, co-crystal results will overwrite)
    for gene in GENE_UNIPROT:
        if gene in [c[1] for c in CO_CRYSTALS]:
            continue
        print(f"\n  [{gene}] Pocket-distance proxy")
        dists = compute_pocket_distances(gene)
        all_distances.update(dists)

    return all_distances


def integrate_and_evaluate(all_distances):
    """Add drug_distance feature, retrain model, evaluate per-gene AUROC."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.calibration import CalibratedClassifierCV
    from xgboost import XGBClassifier

    print("\n" + "=" * 60)
    print("INTEGRATING DRUG DISTANCE FEATURE + RETRAINING")
    print("=" * 60)

    # Load Stage 1 data
    df = load_feature_data()

    # Add features we need
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

    plddt_path = OUTPUT_DIR / "plddt_data.pkl"
    if plddt_path.exists():
        plddt_df = pd.read_pickle(plddt_path)
        # Drop duplicate pLDDT columns before merge
        for c in ['plddt_score_x','plddt_environment_x','plddt_score_y','plddt_environment_y']:
            if c in df.columns:
                del df[c]
        df = df.merge(plddt_df, on=["gene", "residue_pos"], how="left")

    # Add mutation_sensitivity removed: only 2 unique values (0.778, 1.0), 99.7% constant.
    # BLOSUM-based approximation too coarse to capture TB-specific mutation spectrum.

    # Expand known hotspots: add inhA known resistance residues
    for pos in [21, 94, 95, 99, 103, 203]:
        mask = (df["gene"] == "inhA") & (df["residue_pos"] == pos)
        df.loc[mask, "is_hotspot"] = 1

    # Expand with CRyPTIC-validated Tier 1-2 residues (from independent clinical data)
    cryptic_new_positives = [("gyrA", 88), ("inhA", 194), ("eis", 59), ("inhA", 16), ("rpoB", 483)]
    df["is_cryptic_hotspot"] = 0
    for gene, pos in cryptic_new_positives:
        mask = (df["gene"] == gene) & (df["residue_pos"] == pos)
        df.loc[mask, "is_hotspot"] = 1
        df.loc[mask, "is_cryptic_hotspot"] = 1
    print(f"  Added {len(cryptic_new_positives)} CRyPTIC-validated positive residues")
    print(f"  Total hotspots: {df['is_hotspot'].sum()} ({df['is_cryptic_hotspot'].sum()} from CRyPTIC)")

    # Assign drug_distance (raw) + drug_proximity (saturating transform)
    drug_dist_col = np.full(len(df), np.nan)
    for (gene, pos), dist in all_distances.items():
        mask = (df["gene"] == gene) & (df["residue_pos"] == pos)
        drug_dist_col[mask] = dist

    # Fill remaining NaN with a large default
    default_val = 100.0
    drug_dist_col = np.where(np.isnan(drug_dist_col), default_val, drug_dist_col)
    df["drug_distance"] = drug_dist_col
    # Proximity score: bounded [0,1], 1.0 at d=0, 0.5 at d=10A, ~0.09 at d=100A
    df["drug_proximity"] = 1.0 / (1.0 + df["drug_distance"] / 10.0)

    # Print coverage per gene
    print("\n  Drug distance coverage per gene:")
    for gene in sorted(GENE_UNIPROT.keys()):
        gdf = df[df["gene"] == gene]
        n_with_data = (gdf["drug_distance"] < default_val).sum()
        n_total = len(gdf)
        print(f"    {gene}: {n_with_data}/{n_total} residues with real drug distance")

    # Define features
    base_features = [
        "inner_distance", "homoplasy_count", "homoplasy_alleles",
        "helix_propensity", "strand_propensity", "hydrophobicity",
        "volume", "charge", "hbond", "rel_position",
        "conservation_blosum", "contact_density_seq",
    ]
    new_features = ["sasa_relative", "esm2_intolerance", "contact_density_3d",
                     "plddt_score", "plddt_environment"]
    stage1_features = [f for f in base_features + new_features if f in df.columns]
    all_features = stage1_features + ["drug_proximity"]

    df_model = df.dropna(subset=all_features).copy()
    y = df_model["is_hotspot"].values
    print(f"\n  Model samples: {len(df_model)}, positives: {y.sum()}")

    # Per-gene evaluation
    print("\n  Per-gene AUROC comparison:")
    print(f"  {'Gene':<8} {'Stage1 AUROC':<15} {'+Docking AUROC':<17} {'Delta':<8} {'n_pos':<6}")
    print(f"  {'-'*56}")

    gene_results = {}
    for gene in sorted(GENE_UNIPROT.keys()):
        gmask = df_model["gene"] == gene
        gy = y[gmask]
        if gy.sum() < 2:
            n_res = gmask.sum()
            n_pos = int(gy.sum())
            print(f"  {gene:<8} {'insufficient':<15} {'insufficient':<17} {'N/A':<8} {n_pos:<6} "
                  f"(only {n_pos} positive(s) among {n_res} residues — need >= 2)")
            gene_results[gene] = {"stage1_auroc": None, "stage3_auroc": None, "delta": None,
                                  "status": f"insufficient positives ({n_pos}/{n_res})"}
            continue

        gx1 = df_model.loc[gmask, stage1_features].values
        gx2 = df_model.loc[gmask, all_features].values

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        aucs1, aucs2 = [], []
        for train_idx, test_idx in skf.split(gx1, gy):
            scaler = StandardScaler()
            gx1_train = scaler.fit_transform(gx1[train_idx])
            gx1_test = scaler.transform(gx1[test_idx])
            gx2_train = scaler.fit_transform(gx2[train_idx])
            gx2_test = scaler.transform(gx2[test_idx])

            m1 = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
            m1.fit(gx1_train, gy[train_idx])
            aucs1.append(roc_auc_score(gy[test_idx], m1.predict_proba(gx1_test)[:, 1]))

            m2 = XGBClassifier(scale_pos_weight=10, max_depth=6, learning_rate=0.05,
                               n_estimators=300, subsample=0.8, colsample_bytree=0.8,
                               eval_metric="logloss", random_state=42)
            m2.fit(gx2[train_idx], gy[train_idx])
            aucs2.append(roc_auc_score(gy[test_idx], m2.predict_proba(gx2[test_idx])[:, 1]))

        a1 = np.mean(aucs1)
        a2 = np.mean(aucs2)
        delta = a2 - a1
        gene_results[gene] = {"stage1_auroc": a1, "stage3_auroc": a2, "delta": delta}
        print(f"  {gene:<8} {a1:<15.4f} {a2:<17.4f} {delta:<+8.4f} {gy.sum():<6}")

    # Global 5-fold CV (StratifiedKFold — residue-level)
    print("\n  Global 5-fold CV (StratifiedKFold — residue-level):")
    X1 = df_model[stage1_features].values
    X2 = df_model[all_features].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    glob_auc1, glob_auc2 = [], []
    glob_ap1, glob_ap2 = [], []
    top20_1, top20_2 = [], []

    for train_idx, test_idx in skf.split(X2, y):
        scaler = StandardScaler()
        X1_train = scaler.fit_transform(X1[train_idx])
        X1_test = scaler.transform(X1[test_idx])
        X2_train = scaler.fit_transform(X2[train_idx])
        X2_test = scaler.transform(X2[test_idx])

        m1 = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
        m1.fit(X1_train, y[train_idx])
        p1 = m1.predict_proba(X1_test)[:, 1]

        m2 = XGBClassifier(scale_pos_weight=10, max_depth=6, learning_rate=0.05,
                           n_estimators=300, subsample=0.8, colsample_bytree=0.8,
                           eval_metric="logloss", random_state=42)
        m2.fit(X2[train_idx], y[train_idx])
        p2 = m2.predict_proba(X2[test_idx])[:, 1]

        glob_auc1.append(roc_auc_score(y[test_idx], p1))
        glob_auc2.append(roc_auc_score(y[test_idx], p2))
        glob_ap1.append(average_precision_score(y[test_idx], p1))
        glob_ap2.append(average_precision_score(y[test_idx], p2))
        top20_1.append(y[test_idx][np.argsort(p1)[::-1][:20]].sum() / max(y[test_idx].sum(), 1))
        top20_2.append(y[test_idx][np.argsort(p2)[::-1][:20]].sum() / max(y[test_idx].sum(), 1))

    res_strat_auc = float(np.mean(glob_auc2))
    res_strat_ap = float(np.mean(glob_ap2))
    print(f"    Stage 1: AUROC={np.mean(glob_auc1):.4f}+-{np.std(glob_auc1):.4f}  "
          f"AUPRC={np.mean(glob_ap1):.4f}  Top20={np.mean(top20_1):.3f}")
    print(f"    Stage 3: AUROC={res_strat_auc:.4f}+-{np.std(glob_auc2):.4f}  "
          f"AUPRC={res_strat_ap:.4f}  Top20={np.mean(top20_2):.3f}")

    # GroupKFold by gene (more conservative — no gene-level leakage)
    print("\n  GroupKFold by gene (5 folds):")
    genes = df_model["gene"].values
    gkf = GroupKFold(n_splits=5)
    gkf_auc2, gkf_ap2 = [], []
    gkf_top20_2 = []
    n_folds_completed = 0

    for train_idx, test_idx in gkf.split(X2, y, groups=genes):
        test_genes = set(genes[test_idx])
        n_pos_test = y[test_idx].sum()
        if n_pos_test < 2:
            print(f"    SKIP fold: held-out genes {test_genes} have {n_pos_test} positives (<2)")
            continue
        scaler = StandardScaler()
        X2_train = scaler.fit_transform(X2[train_idx])
        X2_test = scaler.transform(X2[test_idx])

        m2 = XGBClassifier(scale_pos_weight=10, max_depth=6, learning_rate=0.05,
                           n_estimators=300, subsample=0.8, colsample_bytree=0.8,
                           eval_metric="logloss", random_state=42)
        m2.fit(X2[train_idx], y[train_idx])
        p2 = m2.predict_proba(X2[test_idx])[:, 1]

        gkf_auc2.append(roc_auc_score(y[test_idx], p2))
        gkf_ap2.append(average_precision_score(y[test_idx], p2))
        gkf_top20_2.append(y[test_idx][np.argsort(p2)[::-1][:20]].sum() / max(y[test_idx].sum(), 1))
        n_folds_completed += 1

    if n_folds_completed > 0:
        print(f"    Stage 3: AUROC={np.mean(gkf_auc2):.4f}+-{np.std(gkf_auc2):.4f}  "
              f"AUPRC={np.mean(gkf_ap2):.4f}  Top20={np.mean(gkf_top20_2):.3f}")
        print(f"    ({n_folds_completed}/5 folds completed; some folds skipped if <2 positives)")
    else:
        print("    No GroupKFold folds had >=2 positives — cannot compute")

    # Save updated data
    df.to_csv(OUTPUT_DIR / "residue_hotspot_data_with_docking.csv", index=False)
    print(f"\n  Updated data saved")

    results = {
        "global": {
            "stage1_auroc": float(np.mean(glob_auc1)),
            "stage3_auroc": res_strat_auc,
            "stage3_groupkfold_auroc": float(np.mean(gkf_auc2)) if n_folds_completed > 0 else None,
            "stage3_groupkfold_auprc": float(np.mean(gkf_ap2)) if n_folds_completed > 0 else None,
            "stage1_auprc": float(np.mean(glob_ap1)),
            "stage3_auprc": res_strat_ap,
            "stage1_top20": float(np.mean(top20_1)),
            "stage3_top20": float(np.mean(top20_2)),
        },
        "per_gene": gene_results,
    }
    return results


def main():
    print("=" * 60)
    print("Stage 3: Per-Gene Drug Contact Features")
    print("=" * 60)

    print("\n[Phase 1] Computing drug distances for all genes...")
    all_distances = compute_all_drug_distances()
    print(f"\n  Total residues with drug-distance data: {len(all_distances)}")

    # Save distances
    dist_path = OUTPUT_DIR / "drug_distances.pkl"
    with open(dist_path, "wb") as f:
        pickle.dump(all_distances, f)
    print(f"  Distances saved to {dist_path}")

    print("\n[Phase 1b] Computing pLDDT confidence features...")
    compute_plddt_features()

    print("\n[Phase 2] Integrating feature and retraining model...")
    results = integrate_and_evaluate(all_distances)

    # Save results
    results_path = OUTPUT_DIR / "stage3_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    # Phase 3: Retrain on full data, produce ranked predictions
    # (overwrites 04c outputs so 04e onwards pick up Stage 3 features)
    print("\n" + "=" * 60)
    print("PHASE 3: FULL MODEL RANKED PREDICTIONS")
    print("=" * 60)

    from xgboost import XGBClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV

    # Reload data with drug_distance
    df = load_feature_data()
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

    plddt_path = OUTPUT_DIR / "plddt_data.pkl"
    if plddt_path.exists():
        plddt_df = pd.read_pickle(plddt_path)
        for c in ['plddt_score_x','plddt_environment_x','plddt_score_y','plddt_environment_y',
                   'plddt_score','plddt_environment']:
            if c in df.columns:
                del df[c]
        df = df.merge(plddt_df, on=["gene", "residue_pos"], how="left")
    for c in list(df.columns):
        if c.startswith('plddt_') and c.endswith(('_x','_y')):
            base = c.rsplit('_', 1)[0]
            if base not in df.columns:
                df.rename(columns={c: base}, inplace=True)
            else:
                del df[c]
    # Clean any merge-suffixed pLDDT columns
    for c in list(df.columns):
        if c.startswith('plddt_') and c.endswith(('_x','_y')):
            base = c.rsplit('_', 1)[0]
            if base not in df.columns:
                df.rename(columns={c: base}, inplace=True)
            else:
                del df[c]

    # mutation_sensitivity removed: only 2 unique values, 99.7% constant. BLOSUM too coarse.

    # Apply drug distances + transform to proximity
    drug_dist_col = np.full(len(df), 100.0)
    for (gene, pos), dist in all_distances.items():
        mask = (df["gene"] == gene) & (df["residue_pos"] == pos)
        drug_dist_col[mask] = dist
    df["drug_distance"] = drug_dist_col
    df["drug_proximity"] = 1.0 / (1.0 + df["drug_distance"] / 10.0)

    # Expand known hotspots: add inhA residues to is_hotspot
    for pos in [21, 94, 95, 99, 103, 203]:
        mask = (df["gene"] == "inhA") & (df["residue_pos"] == pos)
        df.loc[mask, "is_hotspot"] = 1

    # Expand with CRyPTIC-validated Tier 1-2 residues
    cryptic_new_positives = [("gyrA", 88), ("inhA", 194), ("eis", 59), ("inhA", 16), ("rpoB", 483)]
    df["is_cryptic_hotspot"] = 0
    for gene, pos in cryptic_new_positives:
        mask = (df["gene"] == gene) & (df["residue_pos"] == pos)
        df.loc[mask, "is_hotspot"] = 1
        df.loc[mask, "is_cryptic_hotspot"] = 1

    n_total = df["is_hotspot"].sum()
    n_cryptic = df["is_cryptic_hotspot"].sum()
    print(f"\n  Expanded hotspots: {n_total} total ({n_cryptic} from CRyPTIC)")

    base_features = [
        "inner_distance", "homoplasy_count", "homoplasy_alleles",
        "helix_propensity", "strand_propensity", "hydrophobicity",
        "volume", "charge", "hbond", "rel_position",
        "conservation_blosum", "contact_density_seq",
    ]
    new_features = ["sasa_relative", "esm2_intolerance", "contact_density_3d",
                     "plddt_score", "plddt_environment"]
    all_feat = [f for f in base_features + new_features if f in df.columns] + ["drug_proximity"]

    df_model = df.dropna(subset=all_feat).copy()
    y = df_model["is_hotspot"].values
    X = df_model[all_feat].values

    base_model = XGBClassifier(scale_pos_weight=10, max_depth=6, learning_rate=0.05,
                               n_estimators=300, subsample=0.8, colsample_bytree=0.8,
                               eval_metric="logloss", random_state=42)
    # Calibrate with Platt scaling (isotonic for larger datasets)
    # Using 5-fold internal CV calibration to avoid overfitting
    model = CalibratedClassifierCV(base_model, method="sigmoid", cv=5)
    model.fit(X, y)
    df_model["hotspot_raw_xgb"] = base_model.fit(X, y).predict_proba(X)[:, 1]
    df_model["hotspot_score"] = model.predict_proba(X)[:, 1]
    print(f"\n  Calibrated model: Platt-scaled XGBoost probabilities")
    print(f"  Raw XGBoost range: [{df_model['hotspot_raw_xgb'].min():.4f}, {df_model['hotspot_raw_xgb'].max():.4f}]")
    print(f"  Calibrated range:  [{df_model['hotspot_score'].min():.4f}, {df_model['hotspot_score'].max():.4f}]")
    sc = df_model["hotspot_score"]
    df_model["rank_numeric"] = sc.rank(ascending=False)

    # Save ranked predictions (column 'rank' for 04e compatibility)
    ranked_cols = ["gene", "residue_pos", "wt_aa", "is_hotspot", "is_cryptic_hotspot",
                    "drug_distance", "drug_proximity", "hotspot_score", "hotspot_raw_xgb",
                    "rank_numeric"]
    ranked_cols = [c for c in ranked_cols if c in df_model.columns]
    ranked = df_model[ranked_cols].copy()
    ranked.columns = [c if c != "rank_numeric" else "rank" for c in ranked.columns]
    ranked["rank"] = ranked["rank"].astype(int)
    ranked = ranked.sort_values("rank")
    ranked_path = OUTPUT_DIR / "ranked_predictions.csv"
    ranked.to_csv(ranked_path, index=False)
    print(f"  Ranked predictions saved to {ranked_path}")

    # Show Top 20
    has_prox = "drug_proximity" in df_model.columns
    print("\n  Top 20 Predicted Hotspots (Stage 3):")
    hdr = f"  {'Rank':<6} {'Gene':<8} {'Pos':<6} {'AA':<4} {'Score':<8} {'Known':<10}"
    if has_prox:
        hdr += f" {'Prox':<7}"
    print(hdr)
    print(f"  {'-'*(51 + (8 if has_prox else 0))}")
    for i, (_, r) in enumerate(ranked.head(26).iterrows()):
        known = "[KNOWN]" if r["is_hotspot"] else ""
        line = f"  {i+1:<6} {r['gene']:<8} {int(r['residue_pos']):<6} {r['wt_aa']:<4} "
        line += f"{r['hotspot_score']:<8.4f} {known:<10}"
        if has_prox and "drug_proximity" in r:
            line += f" {r['drug_proximity']:<7.4f}"
        print(line)

    # Save feature importance (from base XGBoost, not CalibratedClassifierCV wrapper)
    base_importances = base_model.feature_importances_
    coef_df = pd.DataFrame({"feature": all_feat, "importance": base_importances})
    coef_df = coef_df.sort_values("importance", ascending=False)
    coef_path = OUTPUT_DIR / "feature_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)
    print(f"\n  Feature importances saved to {coef_path}")
    print("\n  Feature importance (XGBoost gain):")
    for _, r in coef_df.iterrows():
        print(f"    {r['feature']}: {r['importance']:.4f}")

    # Save updated feature data (drop prediction/rank columns to avoid merge conflicts)
    drop_cols = [c for c in df_model.columns
                 if c in ("hotspot_score", "rank_numeric", "rank",
                          "stage3_score", "stage3_rank",
                          "stage0_rank", "stage1_rank",
                          "stage0_score", "stage1_score",
                          "overall_rank")]
    df_out = df_model.drop(columns=drop_cols)
    feature_data_path = OUTPUT_DIR / "residue_hotspot_data.csv"
    df_out.to_csv(feature_data_path, index=False)
    print(f"\n  Feature data updated at {feature_data_path}")

    print("\n" + "=" * 60)
    print("Stage 3 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
