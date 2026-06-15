"""
Stage 0: Residue-level hotspot propensity prediction.

Hypothesis: Residue-level "hotspot propensity" is predictable from
sequence-level features using a simple logistic regression model.

This is a proof-of-concept / hypothesis test, NOT a production model.
"""

import gzip
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
REF_DIR = BASE / "reference"
RESULTS_DIR = BASE / "analysis" / "results"
OUTPUT_DIR = RESULTS_DIR / "hotspot_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "resistance_forecasting",
    str(BASE / "scripts" / "04_resistance_forecasting.py"),
)
rf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rf)

RESISTANCE_GENES = rf.RESISTANCE_GENES
CORE_BINDING_RESIDUES = rf.CORE_BINDING_RESIDUES
KNOWN_RES_MUTATIONS = rf.KNOWN_RES_MUTATIONS
GENE_NAME_TO_LOCUS = rf.GENE_NAME_TO_LOCUS
parse_gff_genes = rf.parse_gff_genes
load_reference_genome = rf.load_reference_genome
extract_cds = rf.extract_cds
HELIX_PROPENSITY = rf.HELIX_PROPENSITY
STRAND_PROPENSITY = rf.STRAND_PROPENSITY
HYDROPHOBICITY = rf.HYDROPHOBICITY
HBOND = rf.HBOND

# Amino acid volume (side-chain volume in A^3)
VOLUME = {
    "G": 60, "A": 89, "S": 89, "C": 109, "T": 116, "P": 119,
    "D": 111, "N": 114, "V": 140, "E": 138, "Q": 143, "H": 153,
    "M": 163, "I": 167, "L": 167, "K": 168, "R": 173, "F": 190,
    "Y": 193, "W": 228,
}

# Complete BLOSUM62 diagonal (self-score, conservation proxy)
BLOSUM62_SELF = {
    "A": 4, "R": 5, "N": 6, "D": 6, "C": 9, "Q": 5, "E": 5,
    "G": 6, "H": 8, "I": 4, "L": 4, "K": 5, "M": 5, "F": 6,
    "P": 7, "S": 4, "T": 5, "W": 11, "Y": 7, "V": 4,
}


def parse_known_hotspots():
    """Extract unique (gene, residue_pos) pairs from KNOWN_RES_MUTATIONS."""
    hotspots = {}
    for key in KNOWN_RES_MUTATIONS:
        m = re.search(r'([A-Z])(\d+)([A-Z\*])', key)
        if m:
            gene = key.split("_")[0]
            res = int(m.group(2))
            hotspots[(gene, res)] = 1
    return hotspots


def get_codon_genomic_positions(gff_genes, locus_tag):
    """Return list of (codon_num, [pos1, pos2, pos3]) for a gene's CDS."""
    gene = gff_genes.get(locus_tag)
    if not gene or not gene["cds_intervals"]:
        return []
    intervals = sorted(gene["cds_intervals"])
    total_bp = sum(e - s + 1 for s, e in intervals)
    n_codons = total_bp // 3
    result = []
    if gene["strand"] == "+":
        s = intervals[0][0]
        for k in range(n_codons):
            p1 = s + 3 * k
            p2 = s + 3 * k + 1
            p3 = s + 3 * k + 2
            result.append((k + 1, [p1, p2, p3]))
    else:
        e = intervals[-1][1]
        for k in range(n_codons):
            p1 = e - 3 * k
            p2 = e - 3 * k - 1
            p3 = e - 3 * k - 2
            result.append((k + 1, [p1, p2, p3]))
    return result


def parse_vcf_homoplasy(vcf_path, gff_genes):
    """
    Parse VCF and return homoplasy data per (gene, codon).

    Returns dict: (gene_name, codon_num) -> {"count": int, "alleles": set}
    """
    locus_to_gene = {g[1]: g[0] for g in RESISTANCE_GENES}

    pos_to_gene_codon = {}
    for gene_name, locus_tag, _, _ in RESISTANCE_GENES:
        codon_positions = get_codon_genomic_positions(gff_genes, locus_tag)
        for codon_num, positions in codon_positions:
            for pos in positions:
                pos_to_gene_codon[pos] = (gene_name, codon_num, positions)

    homoplasy = defaultdict(lambda: {"count": 0, "alleles": set()})

    with gzip.open(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            chrom, pos_str, _, ref, alt_str, _, _, info_str, fmt = parts[:9]
            pos = int(pos_str)

            if pos not in pos_to_gene_codon:
                continue

            gene_name, codon_num, positions = pos_to_gene_codon[pos]
            samples = parts[9:] if len(parts) > 9 else []
            fmt_fields = fmt.split(":")

            try:
                gt_idx = fmt_fields.index("GT")
            except ValueError:
                continue

            alt_alleles = alt_str.split(",")
            n_alt = 0

            for sample in samples:
                sample_fields = sample.split(":")
                gt = sample_fields[gt_idx] if gt_idx < len(sample_fields) else "./."
                if gt == "./." or gt == ".|." or gt == ".":
                    continue
                gt_vals = re.split(r"[/|]", gt)
                for gv in gt_vals:
                    if gv != "0" and gv != ".":
                        n_alt += 1

            if n_alt > 0:
                homoplasy[(gene_name, codon_num)]["count"] += n_alt
                for alt in alt_alleles:
                    homoplasy[(gene_name, codon_num)]["alleles"].add(alt)

    return dict(homoplasy)


def build_residue_dataframe():
    """Build the residue-level DataFrame for all resistance genes."""
    gff_path = REF_DIR / "H37Rv.gff"
    fasta_path = REF_DIR / "H37Rv.fasta"
    if not fasta_path.exists():
        fasta_path = REF_DIR / "H37Rv.fna"
    vcf_path = BASE / "data" / "demo" / "drprg_sparse.vcf.gz"

    print("Loading GFF and reference genome...")
    gff_genes = parse_gff_genes(gff_path)
    genome = load_reference_genome(fasta_path)

    print("Parsing VCF for homoplasy...")
    homoplasy_data = parse_vcf_homoplasy(vcf_path, gff_genes)
    print(f"  Found homoplasy data for {len(homoplasy_data)} gene-codon pairs")

    known_hotspots = parse_known_hotspots()
    print(f"  Known hotspot residues: {len(known_hotspots)}")

    rows = []
    positive_set = {"R", "K", "H"}
    negative_set = {"D", "E"}

    for gene_name, locus_tag, drug, pocket in RESISTANCE_GENES:
        cds, prot = extract_cds(gff_genes, genome, locus_tag)
        if cds is None or prot is None:
            print(f"  SKIP {gene_name} ({locus_tag}): no CDS found")
            continue

        prot_len = len(prot)
        core_residues = CORE_BINDING_RESIDUES.get(gene_name, set())

        print(f"  {gene_name} ({locus_tag}): {prot_len} aa")

        for res_pos in range(1, prot_len + 1):
            wt_aa = prot[res_pos - 1]
            if wt_aa == "X":
                continue

            is_hs = 1 if (gene_name, res_pos) in known_hotspots else 0

            inner_dist = 500
            if core_residues:
                inner_dist = min(abs(res_pos - p) for p in core_residues)

            hp_key = (gene_name, res_pos)
            h_count = homoplasy_data.get(hp_key, {}).get("count", 0)
            h_alleles = len(homoplasy_data.get(hp_key, {}).get("alleles", set()))

            helix = HELIX_PROPENSITY.get(wt_aa, 0.0)
            strand = STRAND_PROPENSITY.get(wt_aa, 0.0)
            hydro = HYDROPHOBICITY.get(wt_aa, 0.0)
            vol = VOLUME.get(wt_aa, 120)
            charge = 0
            if wt_aa in positive_set:
                charge = 1
            elif wt_aa in negative_set:
                charge = -1
            hbond = HBOND.get(wt_aa, 0)
            rel_pos = res_pos / max(prot_len, 1)
            cons_blosum = BLOSUM62_SELF.get(wt_aa, 0)

            contact_density = 0
            if core_residues:
                contact_density = sum(
                    1 for p in core_residues if abs(res_pos - p) <= 50
                )

            rows.append({
                "gene": gene_name,
                "locus": locus_tag,
                "residue_pos": res_pos,
                "wt_aa": wt_aa,
                "is_hotspot": is_hs,
                "inner_distance": inner_dist,
                "homoplasy_count": h_count,
                "homoplasy_alleles": h_alleles,
                "helix_propensity": helix,
                "strand_propensity": strand,
                "hydrophobicity": hydro,
                "volume": vol,
                "charge": charge,
                "hbond": hbond,
                "rel_position": rel_pos,
                "conservation_blosum": cons_blosum,
                "contact_density_seq": contact_density,
            })

    df = pd.DataFrame(rows)
    print(f"\nTotal residues: {len(df)}")
    print(f"Hotspot residues: {df['is_hotspot'].sum()}")
    return df


def get_feature_cols():
    return [
        "inner_distance", "homoplasy_count", "homoplasy_alleles",
        "helix_propensity", "strand_propensity", "hydrophobicity",
        "volume", "charge", "hbond", "rel_position",
        "conservation_blosum", "contact_density_seq",
    ]


def run_logistic_regression(df):
    """Fit weighted logistic regression and return model + predictions."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    feature_cols = get_feature_cols()
    df_model = df.dropna(subset=feature_cols).copy()
    print(f"\nTraining samples: {len(df_model)}")
    print(f"Hotspot positives: {df_model['is_hotspot'].sum()}")

    X = df_model[feature_cols].values
    y = df_model["is_hotspot"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        class_weight="balanced", C=1.0, max_iter=1000, random_state=42
    )
    model.fit(X_scaled, y)

    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": model.coef_[0],
    }).sort_values("coefficient", ascending=False)

    print("\nFeature coefficients (what matters most?):")
    for _, r in coef_df.iterrows():
        print(f"  {r['feature']}: {r['coefficient']:.4f}")

    return model, scaler, coef_df, df_model, feature_cols


def run_lo_hotspot_validation(model, scaler, df_model, feature_cols):
    """Leave-one-hotspot-out validation: rank of held-out hotspot."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    hotspot_df = df_model[df_model["is_hotspot"] == 1].copy()
    hotspot_keys = hotspot_df.apply(
        lambda r: f"{r['gene']}_res{r['residue_pos']}", axis=1
    ).unique()

    print(f"\nLeave-one-hotspot-out validation ({len(hotspot_keys)} hotspots):")
    records = []
    full_X = scaler.transform(df_model[feature_cols].values)

    for hs_key in hotspot_keys:
        gene, res_str = hs_key.split("_res")
        res_pos = int(res_str)

        train_mask = ~(
            (df_model["gene"] == gene) & (df_model["residue_pos"] == res_pos)
        )
        test_mask = (df_model["gene"] == gene) & (df_model["residue_pos"] == res_pos)

        if train_mask.sum() < 10 or test_mask.sum() == 0:
            continue

        X_train = df_model.loc[train_mask, feature_cols].values
        y_train = df_model.loc[train_mask, "is_hotspot"].values

        if len(np.unique(y_train)) < 2:
            continue

        fold_scaler = StandardScaler()
        X_train_scaled = fold_scaler.fit_transform(X_train)

        fold_model = LogisticRegression(
            class_weight="balanced", C=1.0, max_iter=1000, random_state=42
        )
        fold_model.fit(X_train_scaled, y_train)

        X_full_scaled = fold_scaler.transform(df_model[feature_cols].values)
        full_preds = fold_model.predict_proba(X_full_scaled)[:, 1]

        full_rank = np.argsort(full_preds)[::-1]
        held_out_idx = df_model[test_mask].index[0]
        rank = int(np.where(full_rank == held_out_idx)[0][0]) + 1

        total = len(full_preds)
        records.append({
            "hotspot": hs_key,
            "gene": gene,
            "residue_pos": res_pos,
            "rank": rank,
            "total": total,
            "in_top20": int(rank <= 20),
            "in_top50": int(rank <= 50),
            "in_top100": int(rank <= 100),
            "in_top200": int(rank <= 200),
        })

    val_df = pd.DataFrame(records)
    if len(val_df) == 0:
        print("  No valid hotspot folds")
        return val_df

    print(f"  Top-20 recall: {val_df['in_top20'].mean():.3f}")
    print(f"  Top-50 recall: {val_df['in_top50'].mean():.3f}")
    print(f"  Top-100 recall: {val_df['in_top100'].mean():.3f}")
    print(f"  Top-200 recall: {val_df['in_top200'].mean():.3f}")
    print(f"  Mean rank: {val_df['rank'].mean():.1f}")
    print(f"  Median rank: {val_df['rank'].median():.1f}")

    return val_df


def run_lo_gene_validation(model, scaler, df_model, feature_cols):
    """Leave-one-gene-out validation: predict whole held-out gene."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    genes_with_hotspots = sorted(
        df_model[df_model["is_hotspot"] == 1]["gene"].unique()
    )
    print(f"\nLeave-one-gene-out validation ({len(genes_with_hotspots)} genes):")

    records = []
    for holdout_gene in genes_with_hotspots:
        train_mask = df_model["gene"] != holdout_gene
        test_mask = df_model["gene"] == holdout_gene

        if train_mask.sum() < 10 or test_mask.sum() < 2:
            continue

        X_train = df_model.loc[train_mask, feature_cols].values
        y_train = df_model.loc[train_mask, "is_hotspot"].values

        if len(np.unique(y_train)) < 2:
            continue

        fold_scaler = StandardScaler()
        X_train_scaled = fold_scaler.fit_transform(X_train)

        fold_model = LogisticRegression(
            class_weight="balanced", C=1.0, max_iter=1000, random_state=42
        )
        fold_model.fit(X_train_scaled, y_train)

        X_test = df_model.loc[test_mask, feature_cols].values
        X_test_scaled = fold_scaler.transform(X_test)
        test_preds = fold_model.predict_proba(X_test_scaled)[:, 1]

        test_df = df_model.loc[test_mask].copy()
        test_df["pred_prob"] = test_preds
        test_df = test_df.sort_values("pred_prob", ascending=False)

        n_hotspot = test_df["is_hotspot"].sum()
        if n_hotspot == 0:
            continue

        hotspot_ranks = []
        for idx in test_df[test_df["is_hotspot"] == 1].index:
            rank = list(test_df.index).index(idx) + 1
            hotspot_ranks.append(rank)

        min_rank = min(hotspot_ranks)
        top20_recall = sum(1 for r in hotspot_ranks if r <= 20) / max(n_hotspot, 1)

        records.append({
            "held_out_gene": holdout_gene,
            "n_residues": len(test_df),
            "n_hotspots": int(n_hotspot),
            "min_hotspot_rank": min_rank,
            "top20_recall": top20_recall,
            "mean_hotspot_rank": np.mean(hotspot_ranks),
            "median_hotspot_rank": np.median(hotspot_ranks),
        })

        print(f"  {holdout_gene}: min_rank={min_rank}, "
              f"top20_recall={top20_recall:.3f}")

    return pd.DataFrame(records) if records else pd.DataFrame()


def compute_lo_gene_auroc(df_model, feature_cols):
    """Compute AUROC from aggregated leave-one-gene-out predictions."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    genes_with_hotspots = sorted(
        df_model[df_model["is_hotspot"] == 1]["gene"].unique()
    )

    all_y = []
    all_preds = []

    for holdout_gene in genes_with_hotspots:
        train_mask = df_model["gene"] != holdout_gene
        test_mask = df_model["gene"] == holdout_gene

        if train_mask.sum() < 10 or test_mask.sum() < 2:
            continue

        X_train = df_model.loc[train_mask, feature_cols].values
        y_train = df_model.loc[train_mask, "is_hotspot"].values

        if len(np.unique(y_train)) < 2:
            continue

        fold_scaler = StandardScaler()
        X_train_scaled = fold_scaler.fit_transform(X_train)

        fold_model = LogisticRegression(
            class_weight="balanced", C=1.0, max_iter=1000, random_state=42
        )
        fold_model.fit(X_train_scaled, y_train)

        X_test = df_model.loc[test_mask, feature_cols].values
        X_test_scaled = fold_scaler.transform(X_test)
        y_test = df_model.loc[test_mask, "is_hotspot"].values
        preds = fold_model.predict_proba(X_test_scaled)[:, 1]

        all_y.extend(y_test)
        all_preds.extend(preds)

    if len(np.unique(all_y)) < 2:
        return np.nan
    return roc_auc_score(all_y, all_preds)


def main():
    print("=" * 70)
    print("Stage 0: Residue-Level Hotspot Propensity Prediction")
    print("=" * 70)

    # Step 1: Build residue-level DataFrame
    print("\n[1] Building residue-level feature DataFrame...")
    df = build_residue_dataframe()

    # Step 2: Fit weighted logistic regression
    print("\n[2] Fitting weighted logistic regression...")
    model, scaler, coef_df, df_model, feature_cols = run_logistic_regression(df)
    coef_df.to_csv(OUTPUT_DIR / "feature_coefficients.csv", index=False)

    # Step 3: Leave-one-hotspot-out validation
    print("\n[3] Leave-one-hotspot-out validation...")
    lo_hs_df = run_lo_hotspot_validation(model, scaler, df_model, feature_cols)
    lo_hs_df.to_csv(OUTPUT_DIR / "lo_hotspot_validation.csv", index=False)

    # Step 4: AUROC from leave-one-gene-out aggregated predictions
    auroc = compute_lo_gene_auroc(df_model, feature_cols)
    print(f"\n  Leave-one-gene-out AUROC: {auroc:.4f}")

    # Step 5: Leave-one-gene-out validation
    print("\n[4] Leave-one-gene-out validation...")
    lo_gene_df = run_lo_gene_validation(model, scaler, df_model, feature_cols)
    if len(lo_gene_df) > 0:
        lo_gene_df.to_csv(
            OUTPUT_DIR / "lo_gene_validation.csv", index=False
        )

    # Step 6: Full prediction on all residues
    print("\n[5] Scoring all residues...")
    full_X_scaled = scaler.transform(df_model[feature_cols].values)
    df_model = df_model.copy()
    df_model["hotspot_probability"] = model.predict_proba(full_X_scaled)[:, 1]
    df_model = df_model.sort_values("hotspot_probability", ascending=False)

    # Add predictions back to full df, preserving all rows
    df["hotspot_probability"] = df_model["hotspot_probability"]  # aligns on index (dropped rows get NaN)
    df.to_csv(OUTPUT_DIR / "residue_hotspot_data.csv", index=False)
    print(f"  Full residue data saved: {OUTPUT_DIR / 'residue_hotspot_data.csv'}")

    top200 = df_model.head(200)[
        ["gene", "locus", "residue_pos", "wt_aa", "is_hotspot",
         "hotspot_probability"] + feature_cols
    ]
    top200.to_csv(OUTPUT_DIR / "predicted_hotspots_top200.csv", index=False)
    print(f"  Top 200 saved to {OUTPUT_DIR / 'predicted_hotspots_top200.csv'}")

    # Step 7: Report
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\n--- Feature Coefficients ---")
    for _, r in coef_df.iterrows():
        print(f"  {r['feature']}: {r['coefficient']:.4f}")

    print("\n--- Leave-One-Hotspot-Out Validation ---")
    if len(lo_hs_df) > 0:
        print(f"  Top-20 recall:   {lo_hs_df['in_top20'].mean():.3f} "
              f"({lo_hs_df['in_top20'].sum()}/{len(lo_hs_df)})")
        print(f"  Top-50 recall:   {lo_hs_df['in_top50'].mean():.3f}")
        print(f"  Top-100 recall:  {lo_hs_df['in_top100'].mean():.3f}")
        print(f"  Top-200 recall:  {lo_hs_df['in_top200'].mean():.3f}")
        print(f"  Mean rank:   {lo_hs_df['rank'].mean():.1f}")
        print(f"  Median rank: {lo_hs_df['rank'].median():.1f}")
        print(f"  AUROC:       {auroc:.4f}")
        print("\n  Per-hotspot ranks:")
        for _, r in lo_hs_df.iterrows():
            flag = " <<<" if r["in_top20"] else ""
            print(f"    {r['hotspot']:20s} rank={r['rank']:4d}/{r['total']}{flag}")

    print("\n--- Leave-One-Gene-Out Validation ---")
    if len(lo_gene_df) > 0:
        for _, r in lo_gene_df.iterrows():
            print(f"  {r['held_out_gene']:10s}: min_rank={r['min_hotspot_rank']:3d}, "
                  f"top20_recall={r['top20_recall']:.3f}, "
                  f"mean_rank={r['mean_hotspot_rank']:.1f}")

    print("\n--- Top 20 Predicted Hotspot Residues ---")
    top20 = df_model.head(20)
    for i, (_, row) in enumerate(top20.iterrows(), 1):
        known = " [KNOWN]" if row["is_hotspot"] else ""
        print(f"  {i:2d}. {row['gene']:6s} {row['wt_aa']}{row['residue_pos']:4d}  "
              f"p={row['hotspot_probability']:.4f}{known}")

    print("\n--- Where Known Hotspots Fall in Global Ranking ---")
    known_in_data = df_model[df_model["is_hotspot"] == 1].copy()
    for i, (_, row) in enumerate(known_in_data.iterrows(), 1):
        rank = list(df_model.index).index(row.name) + 1
        print(f"  {row['gene']:6s} {row['wt_aa']}{row['residue_pos']:4d}  "
              f"rank={rank:4d}/{len(df_model)}  p={row['hotspot_probability']:.4f}")

    print("\n[DONE] Stage 0 complete.")


if __name__ == "__main__":
    main()
