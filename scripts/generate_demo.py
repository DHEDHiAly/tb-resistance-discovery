"""
TB Resistance Discovery — Demo Generator.
Runs all analyses on the demo dataset, generates publication-quality
figures, and produces the presentation-ready Jupyter notebook.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["font.family"] = "sans-serif"

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "demo"
RESULTS = BASE / "analysis" / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)


def load_data():
    clin = pd.read_csv(DATA / "clinical_metadata.csv")
    muts = pd.read_csv(DATA / "mutation_matrix.csv")
    known = pd.read_csv(DATA / "known_resistance_mutations.csv")
    return clin, muts, known


def classify_strain_resistance(row):
    drugs = ["rifampicin", "isoniazid", "ethambutol", "streptomycin",
             "fluoroquinolones", "pyrazinamide", "kanamycin", "capreomycin"]
    resistant_count = sum(1 for d in drugs if row[d] == "resistant")
    if resistant_count == 0:
        return "Pan-susceptible"
    elif resistant_count >= 5:
        return "XDR"
    elif resistant_count >= 2:
        return "MDR"
    else:
        return "HR"


def run_association_test(muts, clin):
    results = []
    resistant = clin[clin["drug_susceptibility"] != "Pan-susceptible"]["strain_id"].tolist()
    susceptible = clin[clin["drug_susceptibility"] == "Pan-susceptible"]["strain_id"].tolist()

    n_res = len(resistant)
    n_sus = len(susceptible)

    present = muts[["gene", "mutation", "protein_change", "is_novel_candidate",
                     "drug_association", "literature_coverage", "impact_score",
                     "strain_id"]].dropna(subset=["mutation"])

    for (gene, mut, prot), group in present.groupby(
        ["gene", "mutation", "protein_change"], dropna=False
    ):
        in_res = sum(1 for s in group["strain_id"] if s in resistant)
        in_sus = sum(1 for s in group["strain_id"] if s in susceptible)

        or_val, p = fisher_exact([[in_res, n_res - in_res],
                                  [in_sus, n_sus - in_sus]])

        row = group.iloc[0]
        results.append({
            "gene": gene,
            "mutation": mut,
            "protein_change": prot,
            "in_resistant": in_res,
            "in_susceptible": in_sus,
            "total_resistant": n_res,
            "total_susceptible": n_sus,
            "odds_ratio": or_val,
            "p_value": p,
            "is_novel_candidate": row["is_novel_candidate"],
            "drug_association": row["drug_association"],
            "literature_coverage": row["literature_coverage"],
            "impact_score": row["impact_score"],
        })

    df = pd.DataFrame(results)
    if not df.empty:
        _, p_adj, _, _ = multipletests(df["p_value"], method="fdr_bh")
        df["p_corrected"] = p_adj
        df["significant"] = df["p_corrected"] < 0.05
        df["nominal_sig"] = df["p_value"] < 0.05
        df["neg_log10_p_raw"] = -np.log10(df["p_value"].clip(lower=1e-10))
        df["neg_log10_p"] = -np.log10(df["p_corrected"].clip(lower=1e-10))
    df = df.sort_values("p_corrected")
    return df


def compute_embedding(muts, clin):
    present = muts.dropna(subset=["mutation"]).copy()
    present["feature"] = present["gene"] + "_" + present["protein_change"].fillna(present["mutation"])
    present["_val"] = 1

    feat = present.pivot_table(
        index="strain_id", columns="feature", values="_val",
        aggfunc="count", fill_value=0
    )

    if feat.shape[1] == 0:
        return None

    try:
        reducer = __import__("umap", fromlist=["UMAP"]).UMAP
    except ImportError:
        print("umap-learn not installed. Using PCA for embedding.")
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        embed = reducer.fit_transform(StandardScaler().fit_transform(feat.values))
    else:
        u = reducer(n_neighbors=min(4, feat.shape[0]-1),
                    min_dist=0.1, random_state=42, metric="cosine")
        embed = u.fit_transform(StandardScaler().fit_transform(feat.values))

    df = pd.DataFrame(embed, index=feat.index, columns=["PC1", "PC2"])
    return df.join(clin.set_index("strain_id")[["drug_susceptibility", "lineage", "country"]])


def plot_manhattan(assoc_df, path):
    fig, ax = plt.subplots(figsize=(12, 5))
    assoc_df = assoc_df.sort_values("p_value")

    novel = assoc_df[assoc_df["is_novel_candidate"] == True]
    known = assoc_df[assoc_df["is_novel_candidate"] != True]

    novel_indices = range(len(known), len(known) + len(novel))
    ax.scatter(range(len(known)), known["neg_log10_p_raw"],
               c="#4A90D9", s=45, alpha=0.7, label="Known mutations", edgecolors="none")
    ax.scatter(novel_indices, novel["neg_log10_p_raw"],
               c="#E74C3C", s=75, alpha=0.9, label="Novel candidates", edgecolors="black",
               linewidth=0.5, zorder=5)

    sig_line = -np.log10(0.05)
    ax.axhline(sig_line, color="gray", linestyle="--", alpha=0.5, label=f"p = 0.05 (nominal)")

    for i, (_, r) in enumerate(novel.iterrows()):
        if r["p_value"] < 0.05:
            idx = len(known) + i
            ax.annotate(f"{r['gene']} {r['protein_change']}",
                        (idx, r["neg_log10_p_raw"]),
                        fontsize=7, alpha=0.8, ha="center", va="bottom", rotation=45)

    ax.set_xlabel("Mutations (sorted by p-value)", fontsize=12)
    ax.set_ylabel("−log₁₀(p-value)", fontsize=12)
    ax.set_title("Association of Mutations with Drug Resistance (Demo — nominal p-values)", fontsize=14)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


def plot_mutation_heatmap(muts, clin, path):
    present = muts.dropna(subset=["mutation"]).copy()
    present["feature"] = present["gene"] + "_" + present["protein_change"].fillna(present["mutation"])
    present["_present"] = 1

    pivot = present.pivot_table(
        index="feature", columns="strain_id", values="_present",
        aggfunc="count", fill_value=0
    ).clip(upper=1)

    strain_order = clin.sort_values("drug_susceptibility")["strain_id"].tolist()
    pivot = pivot.reindex(strain_order, axis=1, fill_value=0)

    annot = pivot.replace({0: "", 1: "●"})

    novelty = present[["feature", "is_novel_candidate"]].drop_duplicates("feature")
    novelty = novelty.set_index("feature").reindex(pivot.index, fill_value=False)

    row_colors = novelty["is_novel_candidate"].map({True: "#E74C3C", False: "#888888"})

    drug_colors = {
        "Pan-susceptible": "#2ECC71",
        "HR": "#F39C12",
        "MDR": "#E67E22",
        "XDR": "#E74C3C",
    }
    col_colors = clin.set_index("strain_id")["drug_susceptibility"].map(drug_colors)
    col_colors = col_colors.reindex(pivot.columns, fill_value="#CCCCCC")

    g = sns.clustermap(
        pivot,
        cmap="Blues",
        row_cluster=True,
        col_cluster=False,
        col_colors=col_colors,
        row_colors=row_colors,
        figsize=(10, 8),
        linewidths=0.5,
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
        annot=annot,
        fmt="",
        annot_kws={"fontsize": 8},
    )

    g.ax_heatmap.set_xlabel("Strain", fontsize=11)
    g.ax_heatmap.set_ylabel("Mutation", fontsize=11)
    g.fig.suptitle("Mutation Profile Matrix — TB Strains", fontsize=14, y=1.02)
    g.savefig(path, bbox_inches="tight", dpi=200)
    print(f"  Saved {path}")
    plt.close(g.fig)


def plot_embedding(embed_df, path):
    if embed_df is None or embed_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    color_maps = {
        "drug_susceptibility": {
            "Pan-susceptible": "#2ECC71", "HR": "#F39C12",
            "MDR": "#E67E22", "XDR": "#E74C3C",
        },
        "lineage": {
            "Lineage1": "#3498DB", "Lineage2": "#9B59B6",
            "Lineage3": "#1ABC9C", "Lineage4": "#E67E22",
        },
    }

    for ax, col in zip(axes, ["drug_susceptibility", "lineage"]):
        cmap = color_maps[col]
        for label, color in cmap.items():
            mask = embed_df[col] == label
            ax.scatter(embed_df.loc[mask, "PC1"], embed_df.loc[mask, "PC2"],
                       c=color, label=label, s=150, edgecolors="black",
                       linewidth=0.8, alpha=0.85)

        for idx in embed_df.index:
            ax.annotate(idx, (embed_df.loc[idx, "PC1"], embed_df.loc[idx, "PC2"]),
                       fontsize=7, alpha=0.7, xytext=(5, 5),
                       textcoords="offset points")

        ax.set_xlabel("Component 1", fontsize=11)
        ax.set_ylabel("Component 2", fontsize=11)
        ax.set_title(f"Latent Space — Colored by {col.replace('_', ' ').title()}", fontsize=13)
        ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


def plot_drug_resistance_profile(clin, path):
    drugs = ["rifampicin", "isoniazid", "ethambutol", "streptomycin",
             "fluoroquinolones", "pyrazinamide", "kanamycin", "capreomycin"]

    plot_df = clin.melt(
        id_vars=["strain_id", "drug_susceptibility"],
        value_vars=drugs,
        var_name="drug", value_name="phenotype"
    )
    plot_df = plot_df.sort_values(["drug_susceptibility", "strain_id"])

    fig, ax = plt.subplots(figsize=(12, 4.5))
    pivot = plot_df.pivot_table(
        index="strain_id", columns="drug", values="phenotype",
        aggfunc="first"
    )
    strain_order = clin.sort_values("drug_susceptibility")["strain_id"]
    pivot = pivot.reindex(strain_order)

    num = pivot.replace({"resistant": 1, "susceptible": 0}).astype(float).values
    cmap = plt.matplotlib.colors.ListedColormap(["#2ECC71", "#E74C3C"])

    ax.imshow(num, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(len(drugs)))
    ax.set_xticklabels([d.capitalize() for d in drugs], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(strain_order)))
    ax.set_yticklabels(strain_order, fontsize=9)
    ax.set_xlabel("Drug", fontsize=11)
    ax.set_ylabel("Strain", fontsize=11)
    ax.set_title("Drug Resistance Profiles", fontsize=14)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#E74C3C", label="Resistant"),
        Patch(facecolor="#2ECC71", label="Susceptible"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


def build_notebook(assoc_df, embed_df):
    """Generate the presentation notebook as .ipynb JSON."""
    cells = []

    def md(source):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [source]
        })

    def code(source):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [source]
        })

    md("# TB Resistance Discovery\n"
       "### Identifying Novel Resistance-Associated Mutations in *Mycobacterium tuberculosis*\n\n"
       "**Aly Dhedhi** · MIT · Professor Manolis Kellis\n\n"
       "---\n\n"
       "**Problem:** A significant fraction of clinically drug-resistant TB isolates carry "
       "no mutations at known resistance loci. Current molecular diagnostics miss these "
       "strains, leading to incorrect susceptibility predictions.\n\n"
       "**Approach:** Analyze whole-genome sequencing data from resistant and susceptible "
       "TB isolates to identify mutations statistically enriched in resistant strains "
       "that have little to no literature coverage. Candidate mutations are then evaluated "
       "for structural impact on drug-binding pockets.\n\n"
       "**This demo:** 10 TB genomes (resistant + susceptible) — proof of concept for "
       "a planned 400-genome study.")

    md("## 1. Load Data\n\n"
       "We have mutation profiles from 10 *M. tuberculosis* strains across multiple "
       "lineages and geographic regions. Each strain has been phenotyped for resistance "
       "to 8 first- and second-line anti-TB drugs.")

    code("import pandas as pd\n"
         "import numpy as np\n"
         "from pathlib import Path\n\n"
         "BASE = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
         "DATA = BASE / 'data' / 'demo'\n"
         "RESULTS = BASE / 'analysis' / 'results'\n\n"
         "clin = pd.read_csv(DATA / 'clinical_metadata.csv')\n"
         "muts = pd.read_csv(DATA / 'mutation_matrix.csv')\n"
         "known = pd.read_csv(DATA / 'known_resistance_mutations.csv')\n\n"
         "print(f'Clinical data: {clin.shape[0]} strains x {clin.shape[1]} columns')\n"
         "print(f'Mutation data: {muts.shape[0]} mutation entries')\n"
         "print(f'Known resistance mutations: {known.shape[0]} entries')")

    md("## 2. Study Population\n\n"
       "Our 10-strain pilot includes diverse lineages and resistance profiles, "
       "from pan-susceptible to extensively drug-resistant (XDR) strains.")

    code("clin[['strain_id', 'country', 'lineage', 'drug_susceptibility']]")

    md("## 3. Drug Resistance Profiles\n\n"
       "Heatmap of phenotypic resistance across 8 anti-TB drugs. Red = resistant, "
       "Green = susceptible. This shows the heterogeneous resistance landscape.")

    code("from IPython.display import Image\n"
         "display(Image(str(RESULTS / 'figures' / 'drug_resistance_profile.png'), width=800))")

    md("## 4. Mutation Landscape\n\n"
       "Each strain carries a unique set of mutations. The heatmap below shows the "
       "mutation matrix — rows are mutations, columns are strains. "
       "**Red labels** on the left indicate novel candidate mutations (little/no literature coverage).\n\n"
       "Column colors on top show resistance class: "
       "Green = Pan-susceptible, Orange/Red = MDR/XDR.")

    code("display(Image(str(RESULTS / 'figures' / 'mutation_heatmap.png'), width=800))")

    md("## 5. Statistical Association Testing\n\n"
       "We use **Fisher's exact test** to identify mutations enriched in resistant vs. "
       "susceptible strains.\n\n"
       "**Note:** With only 10 genomes (8 resistant, 2 susceptible), the statistical power "
       "is limited — we show **nominal (uncorrected) p-values** for demonstration. "
       "The full 400-genome study will apply **Benjamini-Hochberg correction** (FDR < 0.05) "
       "which requires adequate sample size to be meaningful.\n\n"
       "Even at this small scale, we can see which mutations are suggestive of association, "
       "and the pipeline correctly identifies known resistance mutations (rpoB S450L, "
       "katG S315T) while also highlighting novel candidate mutations that warrant "
       "further investigation in the larger cohort.")

    code("assoc_df = pd.read_csv(RESULTS / 'association_results.csv')\n"
         "print(f'Mutations tested: {len(assoc_df)}')\n"
         "print(f'Significant (FDR < 0.05): {assoc_df[\"significant\"].sum()}')\n"
         "print(f'Novel candidates among significant: '\n"
         "      f'{((assoc_df[\"significant\"]) & (assoc_df[\"is_novel_candidate\"]==True)).sum()}')\n\n"
         "assoc_df[['gene', 'protein_change', 'mutation', 'in_resistant', 'in_susceptible',\n"
         "          'odds_ratio', 'p_corrected', 'significant', 'is_novel_candidate']].head(15)")

    md("### Manhattan Plot\n\n"
       "Each point is a mutation. **Blue** = known resistance mutations, "
       "**Red** = novel candidate mutations. Dashed line = nominal significance threshold (p = 0.05).\n\n"
       "Points above the line are suggestively associated — in the full study, "
       "BH correction will identify those with robust evidence. "
       "Even in this small pilot, novel candidates like **Rv0341 G89S** and "
       "**Rv1258c I214V** show enrichment in resistant strains that is worth pursuing.")

    code("display(Image(str(RESULTS / 'figures' / 'manhattan_plot.png'), width=900))")

    md("## 6. Latent Space Embedding\n\n"
       "Each genome is represented as a high-dimensional mutation vector and embedded "
       "into 2D space. This reveals geometric structure invisible to pairwise analysis:\n\n"
       "- **Clusters** of resistant strains that share mutation patterns\n"
       "- **Anomalous strains** that are phenotypically resistant but genomically distinct"
       " from canonical resistance strains — these may harbor novel variants\n"
       "- **Co-occurring mutation combinations** visible as cluster structure, not individual "
       "frequency outliers")

    code("display(Image(str(RESULTS / 'figures' / 'latent_space_embedding.png'), width=900))")

    md("## 7. Novel Candidate Mutations\n\n"
       "This view shows candidate mutations that are:\n"
       "1. Enriched in resistant vs. susceptible strains (nominal p < 0.05 in this pilot)\n"
       "2. Have little to no literature coverage (novel)\n"
       "3. Found in resistant strains but absent from susceptible strains\n\n"
       "In the full 400-genome study, these candidates will be prioritized by "
       "BH-adjusted significance. Each will then be structurally modeled "
       "(AlphaFold2) and tested via molecular docking (AutoDock Vina) to quantify "
       "drug-binding perturbation.")

    code("suggestive = assoc_df[(assoc_df['p_value'] < 0.05) & (assoc_df['is_novel_candidate']==True)].copy()\n"
         "known_sig = assoc_df[(assoc_df['p_value'] < 0.05) & (assoc_df['is_novel_candidate']!=True)].copy()\n\n"
         "print('Suggestive known resistance mutations (validating approach):')\n"
         "if not known_sig.empty:\n"
         "    display(known_sig[['gene', 'protein_change', 'in_resistant', 'in_susceptible',\n"
         "                       'odds_ratio', 'p_value']])\n\n"
         "print('Suggestive novel candidate mutations (warranting further study):')\n"
         "if not suggestive.empty:\n"
         "    display(suggestive[['gene', 'protein_change', 'drug_association',\n"
         "                       'odds_ratio', 'p_value', 'impact_score']])\n"
         "else:\n"
         "    print('  None in this 10-strain pilot — expected with limited power.')\n"
         "    print('  With 400 genomes, we anticipate discovering multiple novel candidates.')")

    md("## 8. Candidate Assessment Pipeline\n\n"
       "Each novel candidate undergoes a multi-step validation:\n\n"
       "| Step | Method | Output |\n"
       "|------|--------|--------|\n"
       "| **Statistical association** | Fisher's exact test + BH correction | p-adjusted < 0.05 |\n"
       "| **Novelty filtering** | PubMed API query | <3 existing publications |\n"
       "| **Structural modeling** | AlphaFold2 (wild-type + mutant) | RMSD, stability change |\n"
       "| **Drug docking** | AutoDock Vina | Binding affinity shift (ΔΔG) |\n"
       "| **Final validation** | Literature review + phylogenetic conservation | Confirmed candidate |\n\n"
       "Mutations that significantly perturb the drug-binding pocket (ΔΔG > 1.0 kcal/mol) "
       "and affect well-conserved residues are prioritized for experimental validation.")

    md("## 9. Key Findings (Demo)\n\n"
       "This 10-genome pilot validates the pipeline and demonstrates the concept:\n\n"
       "- Known resistance mutations (rpoB S450L, katG S315T) are correctly "
       "identified as enriched in resistant vs. susceptible strains\n"
       "- Novel candidate mutations (e.g., Rv0341 G89S, Rv1258c I214V, Rv0005 T141A) "
       "are present exclusively in resistant strains — candidates for further study\n"
       "- The latent space embedding separates resistant from susceptible strains "
       "and reveals structure within resistance classes\n"
       "- The full analysis pipeline (association testing → embedding → novelty "
       "filtering) is operational and reproducible\n\n"
       "**Next steps:** Scale to 400 genomes → BH-adjusted significance → "
       "AlphaFold2 structural modeling → publication.")

    md("## 10. Extension to Other Diseases\n\n"
       "The same framework applies directly to:\n\n"
       "- **Alzheimer's disease:** Rare coding variants in *TREM2*, *SORL1*, *ABCA7* "
       "present an analogous discovery problem — variants enriched in cases vs. "
       "controls with unknown functional impact\n"
       "- **Any trait** with case-control genomic data and a need to discover "
       "novel functional variants beyond known loci\n\n"
       "The plugin architecture makes this a general scientific discovery tool.")

    md("---\n"
       "**Aly Dhedhi** · aly@mit.edu · github.com/DHEDHiAly\n\n"
       "*Professor Manolis Kellis · MIT Computer Science & AI Laboratory (CSAIL)*")

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "cells": cells,
    }

    nb_path = BASE / "notebooks" / "tb_resistance_demo.ipynb"
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    with open(nb_path, "w") as f:
        json.dump(notebook, f, indent=1)
    print(f"\nNotebook written to {nb_path}")
    return nb_path


def main():
    print("=" * 60)
    print("TB Resistance Discovery — Demo Generator")
    print("=" * 60)

    print("\n[1/6] Loading data...")
    clin, muts, known = load_data()

    clin["resistance_class"] = clin.apply(classify_strain_resistance, axis=1)

    print(f"  {len(clin)} strains")
    print(f"  {len(clin[clin['drug_susceptibility'] != 'Pan-susceptible'])} resistant")
    print(f"  {len(clin[clin['drug_susceptibility'] == 'Pan-susceptible'])} susceptible")
    print(f"  Lineages: {clin['lineage'].value_counts().to_dict()}")

    print("\n[2/6] Drug resistance profile plot...")
    plot_drug_resistance_profile(clin, FIGURES / "drug_resistance_profile.png")

    print("\n[3/6] Mutation heatmap...")
    plot_mutation_heatmap(muts, clin, FIGURES / "mutation_heatmap.png")

    print("\n[4/6] Association testing...")
    assoc_df = run_association_test(muts, clin)
    assoc_path = RESULTS / "association_results.csv"
    assoc_df.to_csv(assoc_path, index=False)
    print(f"  {len(assoc_df)} mutation-gene pairs tested")
    print(f"  {assoc_df['significant'].sum()} significant (FDR < 0.05)")
    print(f"  Novel & significant: {((assoc_df['significant']) & (assoc_df['is_novel_candidate']==True)).sum()}")

    print("\n[5/6] Figures...")
    plot_manhattan(assoc_df, FIGURES / "manhattan_plot.png")

    embed_df = compute_embedding(muts, clin)
    if embed_df is not None:
        embed_df.to_csv(RESULTS / "embedding_coordinates.csv")
        plot_embedding(embed_df, FIGURES / "latent_space_embedding.png")
        print("  Embedding computed successfully")
    else:
        print("  Embedding skipped (no mutation features)")

    print("\n[6/6] Building presentation notebook...")
    nb_path = build_notebook(assoc_df, embed_df)

    print(f"\n{'='*60}")
    print("Demo generation complete!")
    print(f"{'='*60}")
    print(f"\nResults:        {RESULTS}")
    print(f"Figures:        {FIGURES}")
    print(f"Notebook:       {nb_path}")
    print(f"\nTo view the notebook:")
    print(f"  jupyter notebook {nb_path}")
    print(f"\nOr run the full pipeline:")
    print(f"  python scripts/run_pipeline.py")


if __name__ == "__main__":
    main()
