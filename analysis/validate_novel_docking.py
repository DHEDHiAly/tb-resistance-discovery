"""
AutoDock Vina binding-score validation for novel / Tier-4 forecast mutations.

Parses existing *_docked.pdbqt scores, runs missing WT vs mutant pairs,
and ranks Tier-4 candidates by ML score + Vina structural coherence.

ΔΔG = mut_binding - wt_binding  (positive => weaker binding => supports resistance)
Categories: STRONG (>=0.4), MODERATE (>=0.15), WEAK (>=0.05), NONE (<0.05)
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
PDB = BASE / "data" / "pdb"
OUT = BASE / "analysis" / "results"
VINA = Path(r"C:\Users\Guest Kellis Lab\AppData\Local\Programs\Python\Python312\vina.exe")

NOVEL_FIVE = [
    ("gyrB", "Q538L", "MFX"),
    ("rpsL", "K43E", "STR"),
    ("eis", "V59A", "AMK"),
    ("rpoB", "V170I", "RFP"),
    ("inhA", "I16V", "triclosan"),
]

# Known WT/mut docked file pairs (mut stem -> wt stem)
DOCK_PAIRS = {
    ("gyrB", "Q538L"): ("gyrB_Q538L_docked.pdbqt", "gyrB_WT_docked.pdbqt"),
    ("rpoB", "V170I"): ("rpoB_V170I_docked.pdbqt", "rpoB_WT_docked.pdbqt"),
    ("inhA", "I16V"): ("inhA_I16V_tric_docked.pdbqt", "inhA_WT_new_docked.pdbqt"),
    ("gyrA", "S91L"): (None, "gyrA_WT_docked.pdbqt"),  # mutant not yet docked
    ("gyrA", "A90T"): ("gyrA_A90T_docked.pdbqt", "gyrA_WT_docked.pdbqt"),
    ("inhA", "S94A"): ("inhA_S94A_tric_docked.pdbqt", "inhA_WT_new_docked.pdbqt"),
    ("rpoB", "L430R"): ("rpoB_L430R_docked.pdbqt", "rpoB_WT_docked.pdbqt"),
}

RECEPTOR_GRID = {
    "rpsL": {"receptor_wt": "rpsL_receptor.pdbqt", "receptor_mut": "rpsL_K43E.pdbqt",
             "ligand": "STR_ligand_fixed.pdbqt", "center": (5.932, -12.167, 9.239)},
    "eis": {"receptor_wt": "eis_receptor.pdbqt", "receptor_mut": "eis_V59A.pdbqt",
            "ligand": "AMK_ligand_fixed.pdbqt", "center": (20.006, -7.940, -12.869)},
    "gyrA": {"receptor_wt": "gyrA_receptor.pdbqt", "receptor_mut": None,
             "ligand": "MFX_ligand.pdbqt", "center": (37.5, 3.1, 21.5),
             "mut_residue": 91, "mut_aa": "L"},
}


def parse_vina_best(pdbqt_path: Path) -> float | None:
    if not pdbqt_path.exists():
        return None
    text = pdbqt_path.read_text(errors="ignore")
    m = re.search(r"REMARK VINA RESULT:\s+(-?\d+\.\d+)", text)
    return float(m.group(1)) if m else None


def categorize(ddg: float | None) -> str:
    if ddg is None:
        return "NO_SCORE"
    if ddg >= 0.4:
        return "STRONG"
    if ddg >= 0.15:
        return "MODERATE"
    if ddg >= 0.05:
        return "WEAK"
    return "NONE"


def prepare_ligand(name: str, smiles: str) -> Path:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    out = PDB / f"{name}_ligand_fixed.pdbqt"
    if out.exists():
        return out
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.MMFFOptimizeMolecule(mol)
    setup = MoleculePreparation().prepare(mol)[0]
    pdbqt = PDBQTWriterLegacy.write_string(setup)[0]
    out.write_text(pdbqt)
    return out


def run_vina(receptor: Path, ligand: Path, center: tuple, out: Path,
             exhaustiveness: int = 8) -> float | None:
    cx, cy, cz = center
    cmd = [
        str(VINA),
        "--receptor", str(receptor),
        "--ligand", str(ligand),
        "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
        "--size_x", "22", "--size_y", "22", "--size_z", "22",
        "--exhaustiveness", str(exhaustiveness),
        "--out", str(out),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        err = getattr(e, "stderr", "") or str(e)
        print(f"  Vina failed for {out.name}: {err[:200]}")
        return None
    return parse_vina_best(out)


def score_pair(gene: str, mutation: str) -> dict:
    key = (gene, mutation)
    mut_file, wt_file = DOCK_PAIRS.get(key, (None, None))
    wt_score = parse_vina_best(PDB / wt_file) if wt_file else None
    mut_score = parse_vina_best(PDB / mut_file) if mut_file else None
    ddg = (mut_score - wt_score) if (wt_score is not None and mut_score is not None) else None
    return {
        "gene": gene, "mutation": mutation,
        "wt_binding_kcal_mol": wt_score,
        "mut_binding_kcal_mol": mut_score,
        "delta_delta_G": ddg,
        "vina_category": categorize(ddg),
        "wt_file": wt_file, "mut_file": mut_file,
    }


def dock_rpsl_k43e() -> dict:
    print("Docking rpsL K43E vs WT (streptomycin)...")
    prepare_ligand("STR", "CN[C@H]1[C@H](O)[C@@H](O)[C@H](CO)O[C@H]1O[C@H]2[C@H](O)[C@@H](O)[C@H](N)[C@H](O)[C@H]2N(C)C")
    cfg = RECEPTOR_GRID["rpsL"]
    lig = PDB / cfg["ligand"]
    wt_out = PDB / "rpsL_WT_STR_docked.pdbqt"
    mut_out = PDB / "rpsL_K43E_STR_docked.pdbqt"
    wt = run_vina(PDB / cfg["receptor_wt"], lig, cfg["center"], wt_out)
    mut = run_vina(PDB / cfg["receptor_mut"], lig, cfg["center"], mut_out)
    ddg = (mut - wt) if (wt is not None and mut is not None) else None
    return {"gene": "rpsL", "mutation": "K43E", "wt_binding_kcal_mol": wt,
            "mut_binding_kcal_mol": mut, "delta_delta_G": ddg,
            "vina_category": categorize(ddg)}


def dock_eis_v59a() -> dict:
    print("Docking eis V59A vs WT (amikacin)...")
    # Amikacin SMILES (simplified aminoglycoside core)
    prepare_ligand("AMK", "NC[C@@H]1O[C@H](O[C@@H]2[C@@H](O)[C@H](O)[C@@H](N)[C@H](O)[C@H]2O)[C@H](N)[C@@H](O)[C@H]1O")
    cfg = RECEPTOR_GRID["eis"]
    lig = PDB / cfg["ligand"]
    wt_out = PDB / "eis_WT_AMK_docked.pdbqt"
    mut_out = PDB / "eis_V59A_AMK_docked.pdbqt"
    wt = run_vina(PDB / cfg["receptor_wt"], lig, cfg["center"], wt_out)
    mut = run_vina(PDB / cfg["receptor_mut"], lig, cfg["center"], mut_out)
    ddg = (mut - wt) if (wt is not None and mut is not None) else None
    return {"gene": "eis", "mutation": "V59A", "wt_binding_kcal_mol": wt,
            "mut_binding_kcal_mol": mut, "delta_delta_G": ddg,
            "vina_category": categorize(ddg)}


def extract_tier4_top(n: int = 30) -> pd.DataFrame:
    val = pd.read_csv(OUT / "forecasting" / "cryptic_tiered_validation.csv")
    t4 = val[val["tier"] == 4].sort_values("rank").head(n)
    struct = pd.read_csv(OUT / "structural_validation_candidates.csv")
    merged = t4.merge(
        struct[["mutation", "gene", "drug", "drug_distance", "structure"]],
        on=["mutation", "gene"], how="left"
    )
    return merged


def main():
    results = {"novel_five": [], "tier4_dockable": [], "interpretation": {}}

    # --- Score existing docked pairs ---
    for gene, mut, _drug in NOVEL_FIVE:
        if (gene, mut) in DOCK_PAIRS:
            r = score_pair(gene, mut)
            results["novel_five"].append(r)
            print(f"  {gene} {mut}: WT={r['wt_binding_kcal_mol']} mut={r['mut_binding_kcal_mol']} "
                  f"ddG={r['delta_delta_G']} [{r['vina_category']}]")

    # --- Run missing dockings ---
    if not any(r["mutation"] == "K43E" and r["mut_binding_kcal_mol"] for r in results["novel_five"]):
        results["novel_five"] = [r for r in results["novel_five"] if r["mutation"] != "K43E"]
        results["novel_five"].append(dock_rpsl_k43e())

    if not any(r["mutation"] == "V59A" for r in results["novel_five"]):
        results["novel_five"].append(dock_eis_v59a())

    # --- Tier-4 candidates with co-crystal structures (dockable via Vina) ---
    t4 = extract_tier4_top(50)
    crystal_genes = {"rpoB", "gyrA", "gyrB"}
    for _, row in t4.iterrows():
        if row["gene"] not in crystal_genes:
            continue
        mut = row["mutation"]
        gene = row["gene"]
        if (gene, mut) in DOCK_PAIRS:
            r = score_pair(gene, mut)
            r["emergence_score"] = row["emergence_score"]
            r["rank"] = int(row["rank"])
            r["drug"] = row.get("drug", "")
            results["tier4_dockable"].append(r)

    # --- Verdict summary ---
    vina_supports = [r for r in results["novel_five"]
                     if r.get("delta_delta_G") is not None and r["delta_delta_G"] >= 0.05]
    vina_contradicts = [r for r in results["novel_five"]
                        if r.get("delta_delta_G") is not None and r["delta_delta_G"] < -0.05]
    vina_blind = [r for r in results["novel_five"]
                  if r.get("delta_delta_G") is None or abs(r.get("delta_delta_G", 0)) < 0.05]

    results["interpretation"] = {
        "vina_supports_resistance": [f"{r['gene']} {r['mutation']}" for r in vina_supports],
        "vina_contradicts_resistance": [f"{r['gene']} {r['mutation']}" for r in vina_contradicts],
        "vina_inconclusive": [f"{r['gene']} {r['mutation']}" for r in vina_blind],
        "note": (
            "Positive ΔΔG = mutant binds weaker (supports resistance). "
            "Vina cannot score charge-reversal, allosteric, or remote mechanisms."
        ),
    }

    out_path = OUT / "novel_docking_validation.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")

    # Print summary table
    print("\n=== NOVEL 5 — Vina Binding Scores ===")
    print(f"{'Mutation':<12} {'WT':>8} {'Mut':>8} {'ddG':>8} {'Category':<10} {'Vina verdict'}")
    print("-" * 70)
    for r in results["novel_five"]:
        wt = r.get("wt_binding_kcal_mol")
        mut = r.get("mut_binding_kcal_mol")
        ddg = r.get("delta_delta_G")
        wt_s = f"{wt:.3f}" if wt is not None else "  N/A"
        mut_s = f"{mut:.3f}" if mut is not None else "  N/A"
        ddg_s = f"{ddg:+.3f}" if ddg is not None else "   N/A"
        if ddg is not None and ddg >= 0.05:
            verdict = "Supports resistance"
        elif ddg is not None and ddg < -0.05:
            verdict = "Contradicts (binds tighter)"
        else:
            verdict = "Inconclusive for Vina"
        print(f"{r['gene']} {r['mutation']:<6} {wt_s:>8} {mut_s:>8} {ddg_s:>8} "
              f"{r.get('vina_category','?'):<10} {verdict}")

    # Top tier-4 with positive Vina signal
    positive_t4 = sorted(
        [r for r in results["tier4_dockable"] if r.get("delta_delta_G") and r["delta_delta_G"] >= 0.15],
        key=lambda x: -x["delta_delta_G"]
    )
    if positive_t4:
        print("\n=== Tier-4 with strongest Vina ΔΔG (co-crystal genes) ===")
        for r in positive_t4[:10]:
            print(f"  rank {r['rank']:>3} {r['gene']} {r['mutation']}: ddG={r['delta_delta_G']:+.3f} "
                  f"(ML score {r['emergence_score']:.3f})")


if __name__ == "__main__":
    main()
