"""
Novelty filter for TB resistance mutations.
Cross-references candidate mutations against PubMed literature
to identify mutations with little or no prior characterization.
"""

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


def search_pubmed(query: str, retmax: int = 20) -> list[str]:
    """Search PubMed and return list of PMIDs."""
    params = urllib.parse.urlencode({
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": retmax,
    })
    url = f"{PUBMED_SEARCH_URL}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
            return data.get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        print(f"  PubMed search failed for '{query}': {e}")
        return []


def get_citation_count(pmids: list[str]) -> int:
    if not pmids:
        return 0
    params = urllib.parse.urlencode({
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "json",
    })
    url = f"{PUBMED_SUMMARY_URL}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
            return len(data.get("result", {}).get("uids", []))
    except Exception:
        return 0


def novelty_score(coverage: str) -> float:
    mapping = {"high": 0.0, "low": 0.5, "none": 1.0, "unknown": 0.3}
    return mapping.get(coverage, 0.0)


def filter_candidates(
    association_results: str,
    output: str,
    pubmed_batch: int = 5,
) -> pd.DataFrame:
    df = pd.read_csv(association_results)

    if "literature_coverage" not in df.columns:
        if "is_novel_candidate" in df.columns:
            df["literature_coverage"] = df["is_novel_candidate"].apply(
                lambda x: "none" if x else "high"
            )
        else:
            df["literature_coverage"] = "unknown"

    candidates = df[
        (df["p_corrected"] < 0.05) &
        (df["literature_coverage"].isin(["none", "low"]))
    ].copy()

    print(f"Checking {len(candidates)} novel candidates against PubMed...")

    candidate_novelty = []
    for i, (_, row) in enumerate(candidates.iterrows()):
        gene = row["gene"]
        mutation = row.get("mutation", "")
        protein = row.get("protein_change", "")

        queries = [
            f"Mycobacterium tuberculosis {gene} {protein} resistance",
            f"TB {gene} {mutation} drug resistance",
            f"{gene} mutation tuberculosis resistance",
        ]

        all_pmids = []
        for query in queries:
            pmids = search_pubmed(query)
            all_pmids.extend(pmids)
            if i % pubmed_batch == 0:
                time.sleep(0.5)

        unique_pmids = list(set(all_pmids))
        citation_count = get_citation_count(unique_pmids)

        is_novel = citation_count < 3
        candidate_novelty.append({
            "gene": gene,
            "mutation": mutation,
            "protein_change": protein,
            "pmids_found": citation_count,
            "is_novel": is_novel,
            "literature_coverage": row.get("literature_coverage", "none"),
            "novelty_score": novelty_score(row.get("literature_coverage", "none")),
            "p_corrected": row["p_corrected"],
            "odds_ratio": row["odds_ratio"],
            "drug_association": row.get("drug_association", "unknown"),
        })

        status = "NOVEL" if is_novel else "KNOWN"
        print(f"  [{status}] {gene} {protein}: {citation_count} PMIDs")

    result_df = pd.DataFrame(candidate_novelty)
    result_df.to_csv(output, index=False)
    print(f"\nFiltered candidates written to {output}")
    print(f"  Total novel candidates: {result_df['is_novel'].sum()}")

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Novelty filter for TB resistance mutations"
    )
    parser.add_argument("--input", default="analysis/results/association_results.csv")
    parser.add_argument("--output", default="analysis/results/novel_candidates.csv")
    parser.add_argument("--pubmed-batch", type=int, default=5,
                       help="Query PubMed every N candidates")
    args = parser.parse_args()

    filter_candidates(args.input, args.output, args.pubmed_batch)


if __name__ == "__main__":
    main()
