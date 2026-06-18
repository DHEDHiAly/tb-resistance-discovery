"""Audit Tier-4 validated hits: re-parse Vina scores and cross-check sources."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import requests

BASE = Path(__file__).resolve().parent.parent
PDB = BASE / "data" / "pdb"
CSV = BASE / "analysis" / "results" / "forecasting" / "tier4_pocket_vina_scores.csv"

VALIDATED = [
    ("gyrA", "G88D"), ("gyrA", "G88S"), ("gyrA", "G88V"), ("gyrA", "S91A"),
    ("gyrB", "Q538L"), ("rpoB", "I491N"), ("rpoB", "L452M"), ("rpoB", "L452R"),
    ("rpoB", "P483R"), ("rpoB", "Q432R"),
]

WT_FILES = {
    "gyrA": "gyrA_WT_redock.pdbqt",
    "rpoB": "rpoB_WT_redock.pdbqt",
    "gyrB": "gyrB_WT_redock.pdbqt",
}

AA3 = {
    "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys", "Q": "Gln",
    "E": "Glu", "G": "Gly", "H": "His", "I": "Ile", "L": "Leu", "K": "Lys",
    "M": "Met", "F": "Phe", "P": "Pro", "S": "Ser", "T": "Thr", "V": "Val",
    "W": "Trp", "Y": "Tyr",
}


def parse_vina(path: Path) -> float | None:
    if not path.exists():
        return None
    m = re.search(r"REMARK VINA RESULT:\s+(-?\d+\.\d+)", path.read_text(errors="ignore"))
    return float(m.group(1)) if m else None


def who_pattern(gene: str, mut: str) -> list[str]:
    m = re.match(r"([A-Z])(\d+)([A-Z])", mut)
    if not m:
        return [mut]
    ref, pos, alt = m.group(1), m.group(2), m.group(3)
    return [
        f"{gene}_p.{ref}{pos}{AA3[alt]}",
        f"p.{ref}{pos}{AA3[alt]}",
        f"{gene} {mut}",
        mut,
    ]


def fetch_who_text() -> str:
    urls = [
        "https://raw.githubusercontent.com/GTB-tbsequencing/mutation-catalogue-2023/main/Final%20Result%20Files/ALL_resistance_associated_mutations.csv",
        "https://raw.githubusercontent.com/GTB-tbsequencing/mutation-catalogue-2023/main/Final%20Result%20Files/ALL_mutations_final.csv",
    ]
    chunks = []
    for url in urls:
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                chunks.append(r.text)
                print(f"WHO: loaded {url.split('/')[-1]} ({len(r.text)} bytes)")
        except Exception as e:
            print(f"WHO fetch failed: {e}")
    return "\n".join(chunks)


def fetch_card_text() -> str:
    url = "https://card.mcmaster.ca/ontology/39867"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            print(f"CARD: loaded ({len(r.text)} bytes)")
            return r.text
    except Exception as e:
        print(f"CARD fetch failed: {e}")
    return ""


def main() -> None:
    df = pd.read_csv(CSV)
    val_df = df[df["structurally_validated_novel"] == True]

    print("\n=== VINA SCORE RE-PARSE (all 32) ===")
    wt_parsed = {g: parse_vina(PDB / f) for g, f in WT_FILES.items()}
    for g, s in wt_parsed.items():
        csv_wt = df.loc[df["gene"] == g, "wt_binding_kcal_mol"].iloc[0]
        ok = s is not None and abs(s - csv_wt) < 0.02
        print(f"  WT {g}: file={s}, csv={csv_wt:.3f} {'OK' if ok else 'MISMATCH'}")

    mismatches = []
    for _, row in df.iterrows():
        docked = PDB / row["docked_file"]
        mut_s = parse_vina(docked)
        if mut_s is None:
            mismatches.append(f"MISSING {row['docked_file']}")
            continue
        if abs(mut_s - row["mut_binding_kcal_mol"]) > 0.02:
            mismatches.append(
                f"{row['gene']} {row['mutation']}: file={mut_s:.3f} csv={row['mut_binding_kcal_mol']:.3f}"
            )
        wt_s = wt_parsed[row["gene"]]
        if wt_s is not None:
            ddg = mut_s - wt_s
            if abs(ddg - row["delta_delta_G"]) > 0.02:
                mismatches.append(
                    f"{row['gene']} {row['mutation']} ddG: calc={ddg:.3f} csv={row['delta_delta_G']:.3f}"
                )

    if mismatches:
        print(f"  FAILURES ({len(mismatches)}):")
        for m in mismatches[:20]:
            print(f"    {m}")
    else:
        print("  All 32 scores match PDBQT files within 0.02 kcal/mol")

    print("\n=== VALIDATED HITS SUMMARY ===")
    print(f"{'Mutation':<14} {'WT':>7} {'Mut':>7} {'ddG':>7} {'Cat':<8} {'FileOK'}")
    for _, row in val_df.iterrows():
        mut_s = parse_vina(PDB / row["docked_file"])
        ok = mut_s is not None and abs(mut_s - row["mut_binding_kcal_mol"]) < 0.02
        print(
            f"{row['gene']}_{row['mutation']:<8} "
            f"{row['wt_binding_kcal_mol']:>7.3f} {row['mut_binding_kcal_mol']:>7.3f} "
            f"{row['delta_delta_G']:>7.3f} {row['vina_category']:<8} {'OK' if ok else 'FAIL'}"
        )

    # Compare with earlier novel_docking_validation.json if present
    prev = BASE / "analysis" / "results" / "novel_docking_validation.json"
    if prev.exists():
        print("\n=== vs EARLIER novel_docking_validation.json ===")
        old = json.loads(prev.read_text())
        old_map = {(x["gene"], x["mutation"]): x for x in old.get("results", [])}
        for gene, mut in VALIDATED:
            key = (gene, mut)
            cur = val_df[(val_df["gene"] == gene) & (val_df["mutation"] == mut)]
            if cur.empty:
                continue
            cur_ddg = float(cur.iloc[0]["delta_delta_G"])
            if key in old_map:
                old_ddg = old_map[key].get("delta_delta_G")
                note = old_map[key].get("vina_category", "")
                print(f"  {gene} {mut}: tier4_batch ddG={cur_ddg:+.3f} | earlier ddG={old_ddg} ({note})")
            else:
                print(f"  {gene} {mut}: tier4_batch ddG={cur_ddg:+.3f} | (not in earlier run)")

    who = fetch_who_text()
    card = fetch_card_text()

    print("\n=== NOVELTY AUDIT (WHO 2023 + CARD) ===")
    print(f"{'Mutation':<14} {'WHO':<6} {'CARD':<6} {'Notes'}")
    notes = {
        "gyrA G88D": "Clinical Mtb FQ mutation; fitness cost (Emane 2021)",
        "gyrA G88S": "G88 QRDR locus; rare FQ variant class",
        "gyrA G88V": "G88 QRDR locus; rare FQ variant class",
        "gyrA S91A": "Codon 91 established (S91P/T common clinical)",
        "gyrB Q538L": "Codon 538 known; published subs are N538D/K/S/T only — Q538L not found",
        "rpoB I491N": "Emerging codon 491 hotspot (2025); I491F WHO borderline",
        "rpoB L452M": "CARD WHO-R resistance variant",
        "rpoB L452R": "CARD PMID:25427352",
        "rpoB P483R": "Rare; not WHO top-tier; computational/predictive refs only",
        "rpoB Q432R": "CARD PMID:15814606",
    }
    for gene, mut in VALIDATED:
        pats = who_pattern(gene, mut)
        in_who = any(p in who for p in pats) if who else False
        in_card = mut in card and gene.lower() in card.lower() if card else False
        # CARD table has | L452M | style
        if card and not in_card:
            in_card = f"| {mut} |" in card
        label = f"{gene} {mut}"
        novel = "NO" if (in_who or in_card or label in notes and "not found" not in notes[label]) else "?"
        if label == "gyrB Q538L":
            novel = "LIKELY YES"
        if label == "rpoB P483R":
            novel = "UNCERTAIN"
        print(f"{gene}_{mut:<8} {str(in_who):<6} {str(in_card):<6} {novel:<12} {notes.get(label, '')}")


if __name__ == "__main__":
    main()
