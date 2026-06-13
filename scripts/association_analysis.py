"""
Statistical association testing for TB resistance mutations.
Performs Fisher's exact test with Benjamini-Hochberg correction
to identify mutations significantly enriched in resistant strains.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


def load_data(mutation_matrix: str, clinical: str) -> pd.DataFrame:
    clin = pd.read_csv(clinical)
    muts = pd.read_csv(mutation_matrix)

    muts["present"] = muts["mutation"].notna() & (muts["mutation"] != "")
    strain_total_muts = muts.groupby("strain_id")["present"].sum().rename("total_mutations")

    strain_data = clin.set_index("strain_id").join(strain_total_muts).fillna(0)
    strain_data["total_mutations"] = strain_data["total_mutations"].astype(int)

    return strain_data, muts


def run_association_test(
    mutation_matrix: str,
    clinical: str,
    output: str,
    drug: str = "rifampicin",
    pval_threshold: float = 0.05,
) -> pd.DataFrame:
    strain_data, muts = load_data(mutation_matrix, clinical)

    if drug in strain_data.columns:
        resistant_strains = strain_data[
            strain_data[drug] == "resistant"
        ].index.tolist()
    else:
        resistant_strains = strain_data[
            strain_data["drug_susceptibility"].str.contains(
                "resistant", case=False, na=False
            )
        ].index.tolist()

    susceptible_strains = strain_data.index.difference(resistant_strains).tolist()

    results = []
    mutation_groups = muts[["gene", "mutation", "protein_change", "drug_association",
                             "literature_coverage", "is_novel_candidate", "impact_score",
                             "strain_id"]].dropna(subset=["mutation"])

    for (gene, mut, prot), group in mutation_groups.groupby(
        ["gene", "mutation", "protein_change"], dropna=False
    ):
        in_resistant = sum(1 for s in group["strain_id"] if s in resistant_strains)
        in_susceptible = sum(1 for s in group["strain_id"] if s in susceptible_strains)
        not_in_resistant = len(resistant_strains) - in_resistant
        not_in_susceptible = len(susceptible_strains) - in_susceptible

        table = [[in_resistant, not_in_resistant],
                 [in_susceptible, not_in_susceptible]]

        if min(in_resistant, in_susceptible, not_in_resistant, not_in_susceptible) < 0:
            continue

        odds_ratio, p_value = fisher_exact(table)

        row = group.iloc[0]
        results.append({
            "gene": gene,
            "mutation": mut,
            "protein_change": prot,
            "drug_association": row["drug_association"],
            "literature_coverage": row["literature_coverage"],
            "is_novel_candidate": row["is_novel_candidate"],
            "impact_score": row["impact_score"],
            "in_resistant": in_resistant,
            "in_susceptible": in_susceptible,
            "total_resistant": len(resistant_strains),
            "total_susceptible": len(susceptible_strains),
            "odds_ratio": odds_ratio,
            "p_value": p_value,
            "enrichment_direction": "resistant" if odds_ratio > 1 else "susceptible",
        })

    result_df = pd.DataFrame(results)

    if not result_df.empty:
        reject, p_corrected, _, _ = multipletests(
            result_df["p_value"], method="fdr_bh"
        )
        result_df["p_corrected"] = p_corrected
        result_df["significant"] = reject & (p_corrected < pval_threshold)

    result_df = result_df.sort_values("p_corrected")
    result_df.to_csv(output, index=False)
    print(f"Results written to {output}")
    print(f"  Total mutation-gene pairs tested: {len(result_df)}")
    print(f"  Significant after BH correction:  {result_df['significant'].sum()}")

    novel_sig = result_df[result_df["is_novel_candidate"] & result_df["significant"]]
    print(f"  Significant novel candidates:     {len(novel_sig)}")
    if not novel_sig.empty:
        print("\nNovel candidate mutations significantly associated with resistance:")
        for _, r in novel_sig.iterrows():
            print(f"  {r['gene']} {r['protein_change']} "
                  f"(OR={r['odds_ratio']:.2f}, p_adj={r['p_corrected']:.4f})")

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Fisher's exact test for TB resistance mutation association"
    )
    parser.add_argument("--mutations", default="data/demo/mutation_matrix.csv")
    parser.add_argument("--clinical", default="data/demo/clinical_metadata.csv")
    parser.add_argument("--output", default="analysis/results/association_results.csv")
    parser.add_argument("--drug", default="rifampicin")
    parser.add_argument("--pval", type=float, default=0.05)
    args = parser.parse_args()

    run_association_test(
        args.mutations, args.clinical, args.output,
        drug=args.drug, pval_threshold=args.pval,
    )


if __name__ == "__main__":
    main()
