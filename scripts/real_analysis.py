"""
TB Resistance Discovery — Real Data Analysis.
Takes a merged multi-sample VCF, annotates variants with gene information,
builds a mutation matrix, runs Fisher's exact test, and generates results.
"""

import gzip
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 200

BASE = Path(__file__).resolve().parent.parent
VCF = BASE / "variants" / "merged.vcf.gz"
GFF = BASE / "reference" / "H37Rv.gff"
RESULTS = BASE / "analysis" / "results"
FIGURES = RESULTS / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# Known resistance mutations: (gene, protein_change) -> drug
KNOWN_RES = {
    "rpoB_S450L": "rifampicin", "rpoB_D435V": "rifampicin",
    "rpoB_H445Y": "rifampicin", "rpoB_H445D": "rifampicin",
    "rpoB_D435Y": "rifampicin", "rpoB_S450W": "rifampicin",
    "rpoB_L430P": "rifampicin", "rpoB_V170F": "rifampicin",
    "rpoB_I491F": "rifampicin", "rpoB_L452P": "rifampicin",
    "katG_S315T": "isoniazid", "katG_S315N": "isoniazid",
    "katG_S315I": "isoniazid", "fabG1_t-8a": "isoniazid",
    "fabG1_c-15t": "isoniazid", "inhA_c-15t": "isoniazid",
    "embB_M306V": "ethambutol", "embB_M306I": "ethambutol",
    "embB_M306L": "ethambutol", "embB_G406D": "ethambutol",
    "embB_G406A": "ethambutol", "embB_Q497R": "ethambutol",
    "rpsL_K43R": "streptomycin", "rpsL_K88R": "streptomycin",
    "rrs_a1401g": "kanamycin", "rrs_c1402t": "capreomycin",
    "rrs_g1484t": "kanamycin",
    "gyrA_D94G": "fluoroquinolones", "gyrA_D94Y": "fluoroquinolones",
    "gyrA_D94N": "fluoroquinolones", "gyrA_A90V": "fluoroquinolones",
    "gyrA_S91P": "fluoroquinolones", "gyrB_N538D": "fluoroquinolones",
    "pncA_L4P": "pyrazinamide", "pncA_V125G": "pyrazinamide",
    "pncA_Q10P": "pyrazinamide", "pncA_L4S": "pyrazinamide",
    "pncA_a-11g": "pyrazinamide", "pncA_D12G": "pyrazinamide",
    "ehA_A341V": "ethionamide", "gid_G73A": "streptomycin",
    "gid_P84L": "streptomycin", "ddn_L49P": "delamanid",
    "eis_g-10a": "kanamycin", "eis_c-12t": "kanamycin",
    "eis_c-14t": "kanamycin",
}


def parse_gff(gff_path):
    """Parse GFF to get gene intervals."""
    genes = []
    with open(gff_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            seqid, source, ftype, start, end, _, strand, _, attr = parts
            if ftype == "gene":
                name_match = re.search(r"Name=([^;]+)", attr)
                gene = name_match.group(1) if name_match else "unknown"
                genes.append({
                    "gene": gene,
                    "start": int(start),
                    "end": int(end),
                    "strand": strand,
                })
    return genes


def parse_vcf(vcf_path, genes):
    """Parse merged VCF and annotate variants with gene info."""
    sample_names = []
    records = []

    opener = gzip.open(vcf_path, "rt") if str(vcf_path).endswith(".gz") else open(vcf_path)

    for line in opener:
        if line.startswith("##"):
            continue
        if line.startswith("#CHROM"):
            parts = line.strip().split("\t")
            sample_names = parts[9:]
            continue

        parts = line.strip().split("\t")
        chrom = parts[0]
        pos = int(parts[1])
        ref = parts[3]
        alt = parts[4]
        quals = parts[5]
        fmt = parts[8]
        sample_data = parts[9:]
        sample_names = [re.search(r'(ERR\d+|SRR\d+|DRR\d+)', s).group(1) if re.search(r'(ERR\d+|SRR\d+|DRR\d+)', s) else s for s in sample_names]

        fmt_keys = fmt.split(":")
        gt_idx = fmt_keys.index("GT") if "GT" in fmt_keys else -1

        genotypes = []
        for sd in sample_data:
            fields = sd.split(":")
            if gt_idx >= 0 and gt_idx < len(fields):
                gt = fields[gt_idx]
            else:
                gt = "./."
            genotypes.append(gt)

        # Determine if multiallelic
        alt_alleles = alt.split(",")

        # Find which gene this variant falls in
        gene_name = "intergenic"
        for g in genes:
            if g["start"] <= pos <= g["end"]:
                gene_name = g["gene"]
                break

        for record in records:
            pass

        records.append({
            "chrom": chrom,
            "pos": pos,
            "ref": ref,
            "alt": alt,
            "gene": gene_name,
            "qual": float(quals) if quals != "." else 0,
            "genotypes": genotypes,
        })

    opener.close()
    return records, sample_names


def build_mutation_matrix(records, sample_names):
    """Build binary matrix: variants x samples."""
    # Clean sample names: extract just the accession (ERR/SRR)
    clean_names = []
    for s in sample_names:
        m = re.search(r'(ERR\d+|SRR\d+|DRR\d+|ERR\d+)', s)
        clean_names.append(m.group(1) if m else s)

    variant_ids = []
    rows = []

    for r in records:
        non_ref = sum(1 for g in r["genotypes"] if g not in ("./.", "0/0", "0|0"))
        if non_ref == 0:
            continue

        var_id = f"{r['gene']}_{r['chrom']}:{r['pos']}{r['ref']}>{r['alt']}"
        variant_ids.append(var_id)

        row = []
        for g in r["genotypes"]:
            is_alt = g not in ("./.", "0/0", "0|0")
            row.append(1 if is_alt else 0)
        rows.append(row)

    df = pd.DataFrame(rows, index=variant_ids, columns=clean_names)
    return df


def get_resistance_phenotypes():
    """Define which samples are resistant (MDR) vs susceptible (Pan-sus)."""
    return {
        "ERR036186": "Pan-susceptible",
        "ERR036187": "Pan-susceptible",
        "ERR036190": "Pan-susceptible",
        "ERR038741": "MDR",
        "ERR037486": "MDR",
        "ERR036189": "MDR",
    }


def run_association(mutation_matrix, phenotypes):
    """Fisher's exact test for each mutation."""
    results = []
    resistant = [s for s, p in phenotypes.items() if p != "Pan-susceptible"]
    susceptible = [s for s, p in phenotypes.items() if p == "Pan-susceptible"]
    n_res = len(resistant)
    n_sus = len(susceptible)

    for var_id in mutation_matrix.index:
        in_res = sum(1 for s in resistant if s in mutation_matrix.columns and mutation_matrix.loc[var_id, s] == 1)
        in_sus = sum(1 for s in susceptible if s in mutation_matrix.columns and mutation_matrix.loc[var_id, s] == 1)

        not_res = n_res - in_res
        not_sus = n_sus - in_sus

        if min(in_res, in_sus, not_res, not_sus) < 0:
            continue

        or_val, p = fisher_exact([[in_res, not_res], [in_sus, not_sus]])

        # Extract gene from var_id
        gene = var_id.split("_")[0]
        mutation = var_id.split("_", 1)[1] if "_" in var_id else var_id

        is_known = gene in {k.split("_")[0] for k in KNOWN_RES}
        matched_known = None
        for km, drug in KNOWN_RES.items():
            if km.split("_")[0] == gene:
                matched_known = drug

        results.append({
            "variant_id": var_id,
            "gene": gene,
            "mutation": mutation,
            "in_resistant": in_res,
            "in_susceptible": in_sus,
            "n_resistant": n_res,
            "n_susceptible": n_sus,
            "odds_ratio": or_val,
            "p_value": p,
            "is_known_resistance_gene": gene in {k.split("_")[0] for k in KNOWN_RES},
        })

    df = pd.DataFrame(results)
    if not df.empty:
        _, p_adj, _, _ = multipletests(df["p_value"].fillna(1), method="fdr_bh")
        df["p_corrected"] = p_adj
        df["significant"] = df["p_corrected"] < 0.05
        df["neg_log10_p"] = -np.log10(df["p_value"].clip(lower=1e-10))
        df["neg_log10_p_adj"] = -np.log10(df["p_corrected"].clip(lower=1e-10))

    return df.sort_values("p_corrected")


def generate_figures(assoc_df, mutation_matrix, phenotypes):
    """Generate all figures."""

    assoc_df = assoc_df.copy()
    assoc_df["_rank"] = range(len(assoc_df))

    # 1. Manhattan plot
    fig, ax = plt.subplots(figsize=(12, 5))
    known = assoc_df[assoc_df["is_known_resistance_gene"] == True]
    other = assoc_df[assoc_df["is_known_resistance_gene"] != True]

    ax.scatter(other["_rank"], other["neg_log10_p"], c="#888888", s=30, alpha=0.5, label="Other genes")
    ax.scatter(known["_rank"], known["neg_log10_p"], c="#E74C3C", s=60, alpha=0.85,
               edgecolors="black", linewidth=0.5, label="Known resistance genes", zorder=5)

    for _, r in known.iterrows():
        ax.annotate(f"{r['gene']}\n{r['mutation'][:20]}",
                    (r["_rank"], r["neg_log10_p"]), fontsize=6, alpha=0.8,
                    ha="center", va="bottom", rotation=45)

    sig_line = -np.log10(0.05)
    ax.axhline(sig_line, color="red", linestyle="--", alpha=0.5, label="p=0.05 (nominal)")

    ax.set_xlabel("Variants (sorted by p-value)", fontsize=12)
    ax.set_ylabel("−log₁₀(p-value)", fontsize=12)
    ax.set_title("TB Resistance Mutation Association (real data, n=6)", fontsize=14)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(FIGURES / "manhattan_plot.png", bbox_inches="tight")
    print(f"  Saved manhattan_plot.png")
    plt.close(fig)

    # 2. Variant heatmap
    pheno_map = {"Pan-susceptible": 0, "MDR": 1}
    col_colors = [pheno_map.get(phenotypes.get(s, "?"), 0) for s in mutation_matrix.columns]
    cmap = plt.matplotlib.colors.ListedColormap(["#2ECC71", "#E74C3C"])

    if mutation_matrix.shape[0] > 50:
        # Top 50 most variable
        var_rates = mutation_matrix.mean(axis=1)
        top_var = mutation_matrix.loc[var_rates.sort_values(ascending=False).index[:50]]
    else:
        top_var = mutation_matrix

    fig, ax = plt.subplots(figsize=(10, max(6, top_var.shape[0] * 0.3)))
    im = ax.imshow(top_var.values, aspect="auto", cmap="Blues", interpolation="nearest")

    ax.set_xticks(range(len(top_var.columns)))
    ax.set_xticklabels(top_var.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(top_var.index)))
    ax.set_yticklabels(top_var.index, fontsize=5)
    ax.set_xlabel("Sample", fontsize=11)
    ax.set_ylabel("Variant", fontsize=11)
    ax.set_title("Variant Presence Matrix (top 50 by frequency)", fontsize=13)

    from matplotlib.patches import Patch
    legend = [Patch(facecolor="#2ECC71", label="Pan-susceptible"),
              Patch(facecolor="#E74C3C", label="MDR")]
    ax.legend(handles=legend, loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(FIGURES / "variant_heatmap.png", bbox_inches="tight")
    print(f"  Saved variant_heatmap.png")
    plt.close(fig)

    # 3. PCA/UMAP embedding
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        embed = pca.fit_transform(mutation_matrix.T.values)
        embed_df = pd.DataFrame(embed, index=mutation_matrix.columns, columns=["PC1", "PC2"])
        embed_df["phenotype"] = [phenotypes.get(s, "unknown") for s in embed_df.index]

        fig, ax = plt.subplots(figsize=(8, 6))
        for pheno, color in [("Pan-susceptible", "#2ECC71"), ("MDR", "#E74C3C")]:
            mask = embed_df["phenotype"] == pheno
            ax.scatter(embed_df.loc[mask, "PC1"], embed_df.loc[mask, "PC2"],
                       c=color, label=pheno, s=200, edgecolors="black",
                       linewidth=1, alpha=0.85)

        for idx in embed_df.index:
            ax.annotate(idx, (embed_df.loc[idx, "PC1"], embed_df.loc[idx, "PC2"]),
                       fontsize=9, alpha=0.7, xytext=(5, 5), textcoords="offset points")

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_title("TB Genome Embedding (PCA)", fontsize=14)
        ax.legend(fontsize=10)
        plt.tight_layout()
        fig.savefig(FIGURES / "pca_embedding.png", bbox_inches="tight")
        print(f"  Saved pca_embedding.png")
        plt.close(fig)

    except Exception as e:
        print(f"  PCA failed: {e}")

    # 4. Known resistance mutation presence
    known_in_data = assoc_df[assoc_df["is_known_resistance_gene"]].copy()
    if not known_in_data.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        top_known = known_in_data.head(20)
        colors = ["#E74C3C" if r["in_resistant"] > r["in_susceptible"] else "#2ECC71"
                  for _, r in top_known.iterrows()]
        bars = ax.barh(range(len(top_known)), top_known["in_resistant"].values,
                       color=colors, edgecolor="black", linewidth=0.5)

        ax.set_yticks(range(len(top_known)))
        ax.set_yticklabels([f"{r['gene']} {r['mutation'][:15]}" for _, r in top_known.iterrows()],
                          fontsize=8)
        ax.set_xlabel("Resistant samples with variant", fontsize=11)
        ax.set_title("Known Resistance Gene Variants (CRyPTIC data)", fontsize=13)
        plt.tight_layout()
        fig.savefig(FIGURES / "known_resistance_variants.png", bbox_inches="tight")
        print(f"  Saved known_resistance_variants.png")
        plt.close(fig)


def main():
    print("=" * 60)
    print("TB Resistance Discovery — Real Data Analysis")
    print("=" * 60)

    print("\n[1/5] Parsing GFF annotation...")
    genes = parse_gff(GFF)
    print(f"  {len(genes)} genes loaded")

    print("\n[2/5] Parsing VCF...")
    records, sample_names = parse_vcf(VCF, genes)
    print(f"  {len(records)} variant records, {len(sample_names)} samples")

    print("\n[3/5] Building mutation matrix...")
    mutation_matrix = build_mutation_matrix(records, sample_names)
    print(f"  Matrix: {mutation_matrix.shape[0]} variants x {mutation_matrix.shape[1]} samples")
    print(f"  Variant rate: {mutation_matrix.mean().mean():.2f} variants per sample")

    # Save mutation matrix
    mutation_matrix.to_csv(RESULTS / "mutation_matrix_real.csv")
    print(f"  Saved mutation_matrix_real.csv")

    print("\n[4/5] Running association testing...")
    phenotypes = get_resistance_phenotypes()
    print(f"  Samples: {phenotypes}")

    assoc_df = run_association(mutation_matrix, phenotypes)
    print(f"  {len(assoc_df)} variant-gene pairs tested")
    print(f"  Significant (FDR < 0.05): {assoc_df['significant'].sum()}")

    # Save results
    assoc_df.to_csv(RESULTS / "association_results_real.csv", index=False)
    print(f"  Saved association_results_real.csv")

    # Print top associations
    print("\n  Top associations by p-value:")
    for _, r in assoc_df.head(10).iterrows():
        print(f"    {r['gene']:20s} {r['mutation']:25s} "
              f"R={r['in_resistant']}/{r['n_resistant']} "
              f"S={r['in_susceptible']}/{r['n_susceptible']} "
              f"OR={r['odds_ratio']:.2f} p={r['p_value']:.4f} "
              f"{'*' if r['significant'] else ''} "
              f"{'[KNOWN]' if r['is_known_resistance_gene'] else ''}")

    print("\n[5/5] Generating figures...")
    generate_figures(assoc_df, mutation_matrix, phenotypes)

    print(f"\n{'='*60}")
    print(f"Analysis complete!")
    print(f"Results: {RESULTS}")
    print(f"Figures: {FIGURES}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
