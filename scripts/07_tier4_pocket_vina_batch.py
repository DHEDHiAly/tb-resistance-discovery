"""
Build mutant receptors, run AutoDock Vina on Tier-4 pocket-direct candidates,
and report novel (tier 4, 0 carriers) mutations with positive ddG.
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
FORECAST = OUT / "forecasting"
VINA = Path(r"C:\Users\Guest Kellis Lab\AppData\Local\Programs\Python\Python312\vina.exe")

sys.path.insert(0, str(BASE / "scripts"))
from sidechain_builder import mutate_residue_heavy, prepare_pdbqt  # noqa: E402

AA1_TO3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}

GENE_DOCK = {
    "gyrA": {
        "receptor": "gyrA_receptor.pdbqt",
        "ligand": "MFX_ligand.pdbqt",
        "wt_docked": "gyrA_WT_docked.pdbqt",
        "chain": "A",
        "pdb_residue": lambda pos: pos,
    },
    "rpoB": {
        "receptor": "rpoB_receptor.pdbqt",
        "ligand": "RFP_ligand.pdbqt",
        "wt_docked": "rpoB_WT_docked.pdbqt",
        "chain": "C",
        "pdb_residue": lambda pos: pos + 6,
    },
    "gyrB": {
        "receptor": "gyrB_receptor.pdbqt",
        "ligand": "MFX_ligand.pdbqt",
        "wt_docked": "gyrB_WT_docked.pdbqt",
        "chain": "A",
        "pdb_residue": lambda pos: pos,
    },
}

_WT_BASELINE: dict[str, float] = {}

MODERATE_DDG = 0.15
WEAK_DDG = 0.05


def parse_mutation(name: str) -> tuple[str, int, str, str]:
    m = re.match(r"^([A-Z])(\d+)([A-Z])$", name)
    if not m:
        raise ValueError(f"Bad mutation name: {name}")
    return m.group(1), int(m.group(2)), m.group(3), AA1_TO3[m.group(3)]


def parse_vina_best(path: Path) -> float | None:
    if not path.exists():
        return None
    m = re.search(r"REMARK VINA RESULT:\s+(-?\d+\.\d+)", path.read_text(errors="ignore"))
    return float(m.group(1)) if m else None


def grid_center_from_docked(wt_docked: Path) -> tuple[float, float, float]:
    coords = []
    for line in wt_docked.read_text(errors="ignore").splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[17:20].strip() in {"MFX", "RFP"} or "UNL" in line[17:27]:
            coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    if not coords:
        raise RuntimeError(f"No ligand coords in {wt_docked}")
    xs, ys, zs = zip(*coords)
    return sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)


def pdbqt_to_pdb(pdbqt_in: Path, pdb_out: Path) -> None:
    lines = []
    for line in pdbqt_in.read_text(errors="ignore").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            lines.append(line[:66] + "\n")
    lines.append("END\n")
    pdb_out.write_text("".join(lines))


def build_mutant_pdbqt(gene: str, mutation: str) -> Path:
    out_pdbqt = PDB / f"{gene}_{mutation}.pdbqt"
    if out_pdbqt.exists():
        return out_pdbqt

    wt_aa, pos, mut_aa, mut_aa3 = parse_mutation(mutation)
    cfg = GENE_DOCK[gene]
    pdb_res = cfg["pdb_residue"](pos)
    chain = cfg["chain"]

    pdb_in = PDB / f"_build_{gene}_receptor.pdb"
    pdb_mut = PDB / f"_build_{gene}_{mutation}.pdb"
    pdbqt_to_pdb(PDB / cfg["receptor"], pdb_in)

    ok = mutate_residue_heavy(
        str(pdb_in), str(pdb_mut), chain, pdb_res, mut_aa3, wt_aa
    )
    if not ok:
        raise RuntimeError(f"Failed to build {gene} {mutation} at PDB res {pdb_res}")

    n_atoms = prepare_pdbqt(str(pdb_mut), str(out_pdbqt))
    if n_atoms < 100:
        raise RuntimeError(f"PDBQT too small for {gene} {mutation}: {n_atoms} atoms")

    pdb_in.unlink(missing_ok=True)
    pdb_mut.unlink(missing_ok=True)
    return out_pdbqt


def get_wt_baseline(gene: str, exhaustiveness: int = 16) -> float | None:
    """Redock WT receptor once per gene for a fair ddG reference."""
    if gene in _WT_BASELINE:
        return _WT_BASELINE[gene]

    cfg = GENE_DOCK[gene]
    cx, cy, cz = grid_center_from_docked(PDB / cfg["wt_docked"])
    out = PDB / f"{gene}_WT_redock.pdbqt"
    if out.exists():
        score = parse_vina_best(out)
        if score is not None:
            _WT_BASELINE[gene] = score
            return score

    cmd = [
        str(VINA),
        "--receptor", str(PDB / cfg["receptor"]),
        "--ligand", str(PDB / cfg["ligand"]),
        "--center_x", f"{cx:.3f}",
        "--center_y", f"{cy:.3f}",
        "--center_z", f"{cz:.3f}",
        "--size_x", "22", "--size_y", "22", "--size_z", "22",
        "--exhaustiveness", str(exhaustiveness),
        "--energy_range", "4",
        "--out", str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        return None
    score = parse_vina_best(out)
    if score is not None:
        _WT_BASELINE[gene] = score
    return score


def run_vina(gene: str, mutation: str, exhaustiveness: int = 16) -> tuple[float | None, Path]:
    cfg = GENE_DOCK[gene]
    receptor = build_mutant_pdbqt(gene, mutation)
    ligand = PDB / cfg["ligand"]
    out = PDB / f"{gene}_{mutation}_docked.pdbqt"
    log = OUT / "docking" / f"{gene}_{mutation}_vina.log"

    if out.exists():
        score = parse_vina_best(out)
        if score is not None:
            return score, out

    cx, cy, cz = grid_center_from_docked(PDB / cfg["wt_docked"])
    cmd = [
        str(VINA),
        "--receptor", str(receptor),
        "--ligand", str(ligand),
        "--center_x", f"{cx:.3f}",
        "--center_y", f"{cy:.3f}",
        "--center_z", f"{cz:.3f}",
        "--size_x", "22", "--size_y", "22", "--size_z", "22",
        "--exhaustiveness", str(exhaustiveness),
        "--energy_range", "4",
        "--out", str(out),
    ]
    log.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    log.write_text(proc.stdout + "\n" + proc.stderr)
    if proc.returncode != 0:
        print(f"  Vina error {gene} {mutation}: {proc.stderr[:300]}")
        return None, out
    return parse_vina_best(out), out


def categorize(ddg: float | None) -> str:
    if ddg is None:
        return "FAILED"
    if ddg >= 0.4:
        return "STRONG"
    if ddg >= MODERATE_DDG:
        return "MODERATE"
    if ddg >= WEAK_DDG:
        return "WEAK"
    return "NONE"


def main():
    subprocess.run([sys.executable, str(BASE / "scripts" / "06_filter_pocket_candidates.py")], check=True)
    pocket = pd.read_csv(FORECAST / "tier4_pocket_direct_matrix.csv")

    results = []
    validated_novel = []

    print(f"Docking {len(pocket)} Tier-4 pocket-direct candidates...\n")
    for _, row in pocket.iterrows():
        gene = row["gene"]
        mutation = row["mutation"]
        if gene not in GENE_DOCK:
            continue

        print(f"  {gene} {mutation} (rank {row['rank']}, score {row['emergence_score']:.3f})...")
        mut_score, docked = run_vina(gene, mutation)
        wt_score = get_wt_baseline(gene)
        ddg = (mut_score - wt_score) if (mut_score is not None and wt_score is not None) else None
        cat = categorize(ddg)

        entry = {
            "gene": gene,
            "mutation": mutation,
            "rank": int(row["rank"]),
            "emergence_score": float(row["emergence_score"]),
            "n_carriers": int(row["n_carriers"]),
            "tier": int(row["tier"]),
            "drug_distance": float(row["drug_distance"]),
            "wt_binding_kcal_mol": wt_score,
            "mut_binding_kcal_mol": mut_score,
            "delta_delta_G": ddg,
            "vina_category": cat,
            "docked_file": docked.name if docked else None,
            "structurally_validated_novel": (
                row["tier"] == 4
                and row["n_carriers"] == 0
                and ddg is not None
                and ddg >= MODERATE_DDG
            ),
        }
        results.append(entry)

        if entry["structurally_validated_novel"]:
            validated_novel.append(entry)
            print(f"    >>> VALIDATED NOVEL: ddG={ddg:+.3f} ({cat})")

        elif ddg is not None:
            print(f"    ddG={ddg:+.3f} ({cat})")
        else:
            print("    FAILED")

    out_json = OUT / "tier4_pocket_vina_results.json"
    out_json.write_text(json.dumps({"results": results, "validated_novel": validated_novel}, indent=2))

    out_csv = FORECAST / "tier4_pocket_vina_scores.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)

    print(f"\nSaved {out_json}")
    print(f"Saved {out_csv}")

    if validated_novel:
        best = max(validated_novel, key=lambda x: x["delta_delta_G"])
        print("\n=== STRUCTURALLY VALIDATED NOVEL MUTATION(S) ===")
        for v in sorted(validated_novel, key=lambda x: -x["delta_delta_G"]):
            print(
                f"  {v['gene']} {v['mutation']}  rank={v['rank']}  "
                f"ddG={v['delta_delta_G']:+.3f} kcal/mol  ({v['vina_category']})"
            )
        print(f"\nTop hit: {best['gene']} {best['mutation']} ddG={best['delta_delta_G']:+.3f}")
    else:
        weak = [r for r in results if r.get("delta_delta_G") and r["delta_delta_G"] >= WEAK_DDG]
        weak.sort(key=lambda x: -x["delta_delta_G"])
        print("\nNo novel mutation reached MODERATE threshold (ddG >= 0.15).")
        if weak:
            print("Closest WEAK hits (ddG >= 0.05):")
            for r in weak[:5]:
                print(f"  {r['gene']} {r['mutation']} ddG={r['delta_delta_G']:+.3f} ({r['vina_category']})")


if __name__ == "__main__":
    main()
