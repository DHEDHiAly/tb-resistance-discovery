"""
Latent space embedding of TB genome mutation profiles.
Projects high-dimensional mutation vectors into 2D space
using UMAP for visualizing resistance-associated clusters
and identifying anomalous strains with potential novel variants.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from umap import UMAP


def build_feature_matrix(
    mutation_matrix: str,
    clinical: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    muts = pd.read_csv(mutation_matrix)
    clin = pd.read_csv(clinical)

    present = muts.dropna(subset=["mutation"]).copy()
    present["feature_id"] = present["gene"] + "_" + present["protein_change"].fillna(present["mutation"])
    present["_val"] = 1

    feature_table = present.pivot_table(
        index="strain_id",
        columns="feature_id",
        values="_val",
        aggfunc="count",
        fill_value=0,
    ).clip(upper=1).astype(int)

    feature_cols = feature_table.columns.tolist()
    mat = feature_table.values
    strain_ids = feature_table.index.tolist()

    clin_indexed = clin.set_index("strain_id").loc[strain_ids]
    labels = clin_indexed["drug_susceptibility"].values
    lineages = clin_indexed["lineage"].values
    countries = clin_indexed["country"].values

    result_df = pd.DataFrame(
        mat,
        index=strain_ids,
        columns=feature_cols,
    )

    return result_df, clin_indexed, feature_cols


def compute_embedding(
    mutation_matrix: str,
    clinical: str,
    output_dir: str,
    n_neighbors: int = 5,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    feat_df, clin_df, features = build_feature_matrix(mutation_matrix, clinical)

    scaler = StandardScaler()
    mat_scaled = scaler.fit_transform(feat_df.values)

    n_components = min(feat_df.shape[0] - 1, 2)
    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=max(n_components, 2),
        random_state=random_state,
        metric="cosine",
    )
    embedding = reducer.fit_transform(mat_scaled)

    embed_df = pd.DataFrame(
        embedding,
        index=feat_df.index,
        columns=["UMAP1", "UMAP2"],
    )
    embed_df = embed_df.join(clin_df[["drug_susceptibility", "lineage", "country"]])
    embed_df = embed_df.join(feat_df)

    for label_col, palette_title in [
        ("drug_susceptibility", "Drug Susceptibility"),
        ("lineage", "Lineage"),
        ("country", "Country"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 8))
        unique_labels = embed_df[label_col].unique()
        n_labels = len(unique_labels)
        colors = sns.color_palette("husl", n_labels)
        label_color_map = dict(zip(sorted(unique_labels), colors))

        for label in sorted(unique_labels):
            mask = embed_df[label_col] == label
            ax.scatter(
                embed_df.loc[mask, "UMAP1"],
                embed_df.loc[mask, "UMAP2"],
                c=[label_color_map[label]],
                label=label,
                s=120,
                edgecolors="black",
                linewidth=0.8,
                alpha=0.85,
            )

        for idx in embed_df.index:
            ax.annotate(
                idx,
                (embed_df.loc[idx, "UMAP1"], embed_df.loc[idx, "UMAP2"]),
                fontsize=7,
                alpha=0.7,
                xytext=(5, 5),
                textcoords="offset points",
            )

        ax.set_xlabel("UMAP 1", fontsize=12)
        ax.set_ylabel("UMAP 2", fontsize=12)
        ax.set_title(f"TB Genome Embedding — Colored by {palette_title}", fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
        plt.tight_layout()
        fig_path = out / f"umap_by_{label_col}.png"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        print(f"Saved {fig_path}")
        plt.close(fig)

    embed_df.to_csv(out / "embedding_coordinates.csv")
    print(f"\nEmbedding complete. {len(embed_df)} strains in {embedding.shape[1]}D space.")
    print(f"  Resistance classes: {embed_df['drug_susceptibility'].value_counts().to_dict()}")

    return embed_df


def main():
    parser = argparse.ArgumentParser(
        description="Latent space embedding of TB mutation profiles"
    )
    parser.add_argument("--mutations", default="data/demo/mutation_matrix.csv")
    parser.add_argument("--clinical", default="data/demo/clinical_metadata.csv")
    parser.add_argument("--output", default="analysis/results")
    parser.add_argument("--neighbors", type=int, default=5)
    args = parser.parse_args()

    compute_embedding(
        args.mutations,
        args.clinical,
        args.output,
        n_neighbors=args.neighbors,
    )


if __name__ == "__main__":
    main()
