"""
Step 5: CRyPTIC Validation of Emergence Predictions

Builds mutation matrix by filtering MUTATIONS.csv.gz (1.4 GB)
for phenotype-matched samples, then cross-references watchlist.

Strategy:
  1. Load 12,287 phenotype UNIQUEIDs into a set
  2. Stream-filter MUTATIONS.csv.gz for matching samples + target genes
  3. Aggregate mutation carriers and link to phenotypes
  4. Cross-reference against 315 watchlist mutations

Categories:
  A: Known WHO mutations (sanity check)
  B: Novel watchlist mutations observed in CRyPTIC
  C: Forecast-only mutations (prospective targets)
"""

import gzip
import csv
import json
import os
import random
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRYPTIC_DIR = os.path.join(BASE, "data", "cryptic")
META_DIR = os.path.join(BASE, "data", "metadata")
OUTPUT_DIR = os.path.join(BASE, "analysis", "results", "forecasting")
CACHE_DIR = os.path.join(CRYPTIC_DIR, "cache")

os.makedirs(CACHE_DIR, exist_ok=True)

# Target resistance genes
TARGET_GENES = [
    "rpoB", "katG", "embB", "gyrA", "gyrB", "pncA", "rpsL",
    "inhA", "eis", "tap", "mmpL5", "mmpR5", "tlyA",
]

# Drug mapping
GENE_DRUG_MAP = {
    "rpoB": "RIF_BINARY_PHENOTYPE",
    "katG": "INH_BINARY_PHENOTYPE",
    "embB": "EMB_BINARY_PHENOTYPE",
    "gyrA": "MXF_BINARY_PHENOTYPE",
    "gyrB": "MXF_BINARY_PHENOTYPE",
    "pncA": None,
    "rpsL": None,
    "inhA": "INH_BINARY_PHENOTYPE",
    "eis": "KAN_BINARY_PHENOTYPE",
    "tap": "KAN_BINARY_PHENOTYPE",
    "mmpL5": "BDQ_BINARY_PHENOTYPE",
    "mmpR5": "BDQ_BINARY_PHENOTYPE",
    "tlyA": None,
}

GENE_DRUG_ALT = {
    "gyrA": "LEV_BINARY_PHENOTYPE",
    "gyrB": "LEV_BINARY_PHENOTYPE",
    "eis": "AMI_BINARY_PHENOTYPE",
    "tap": "AMI_BINARY_PHENOTYPE",
}

WHO_MUTATIONS = {
    "rpoB_S450L", "rpoB_D435V", "rpoB_H445Y", "rpoB_H445D",
    "rpoB_D435Y", "rpoB_S450W", "rpoB_L430P", "rpoB_V170F",
    "rpoB_I491F", "rpoB_L452P",
    "katG_S315T", "katG_S315N", "katG_S315I",
    "embB_M306V", "embB_M306I", "embB_M306L",
    "embB_G406D", "embB_G406A", "embB_Q497R",
    "rpsL_K43R", "rpsL_K88R",
    "gyrA_D94G", "gyrA_D94Y", "gyrA_D94N",
    "gyrA_A90V", "gyrA_S91P",
    "gyrB_N538D",
    "pncA_L4P", "pncA_V125G", "pncA_Q10P",
    "pncA_L4S", "pncA_D12G",
}


def load_phenotype_uids():
    """Load all phenotype UNIQUEIDs for filtering."""
    df = pd.read_csv(os.path.join(META_DIR, "cryptic_phenotypes.csv"), low_memory=False)
    uids = set(df["UNIQUEID"].unique())
    print(f"  Phenotype samples: {len(uids)}")
    return uids, df


def filter_mutation_table(pheno_uids):
    """
    Stream-filter MUTATIONS.csv.gz for matching samples + target genes.
    Saves filtered results to cache for reuse.
    
    Returns: dict of (gene, mutation) -> set of UNIQUEIDs
    """
    cache_file = os.path.join(CACHE_DIR, "resistance_mutations.pkl")
    
    if os.path.exists(cache_file):
        print(f"  Loading cached resistance mutations from {cache_file}...")
        import pickle
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    
    fpath = os.path.join(CRYPTIC_DIR, "MUTATIONS.csv.gz")
    mutation_samples = defaultdict(set)
    
    total_rows = 0
    filtered_rows = 0
    matched_samples = set()
    
    print(f"  Streaming {fpath}...")
    
    with gzip.open(fpath, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            if total_rows % 500000 == 0:
                print(f"    Processed {total_rows:,} rows, kept {filtered_rows:,}, matched {len(matched_samples)} samples...")
                sys.stdout.flush()
            
            uid = row.get("UNIQUEID", "")
            if uid not in pheno_uids:
                continue
            
            gene = row.get("GENE", "")
            if gene not in TARGET_GENES:
                continue
            
            is_nonsyn = row.get("IS_NONSYNONYMOUS", "")
            if is_nonsyn != "True":
                continue
            
            mutation = row.get("MUTATION", "")
            if not mutation:
                continue
            
            mutation_samples[(gene, mutation)].add(uid)
            matched_samples.add(uid)
            filtered_rows += 1
    
    print(f"    Total: {total_rows:,} rows, kept {filtered_rows:,} ({matched_samples} unique samples)")
    
    # Cache
    import pickle
    with open(cache_file, "wb") as f:
        pickle.dump(dict(mutation_samples), f)
    print(f"    Cached to {cache_file}")
    
    return dict(mutation_samples)


def cross_reference(watchlist, mutation_samples, pheno):
    """Cross-reference watchlist mutations against CRyPTIC."""
    
    pheno_lookup = pheno.set_index("UNIQUEID")
    
    results = []
    category_counts = {"A": 0, "B": 0, "C": 0}
    category_details = {"A": [], "B": [], "C": []}
    
    # Pre-compute background phenotype counts
    background = {}
    for col in pheno.columns:
        if "BINARY_PHENOTYPE" in col:
            counts = pheno[col].value_counts()
            background[col] = {"R": counts.get("R", 0), "S": counts.get("S", 0)}
    
    # Process each watchlist mutation
    for _, row in watchlist.iterrows():
        gene = row["gene"]
        mutation = row["mutation"]
        
        carriers = mutation_samples.get((gene, mutation), set())
        n_carriers = len(carriers)
        
        who_key = f"{gene}_{mutation}"
        is_known = who_key in WHO_MUTATIONS
        
        drug_col = GENE_DRUG_MAP.get(gene)
        alt_drug = GENE_DRUG_ALT.get(gene)
        drug_used = drug_col or alt_drug or "unknown"
        
        resistant = 0
        susceptible = 0
        n_phenotyped = 0
        phenotype_nas = 0
        uid_not_found = 0
        
        if n_carriers > 0:
            for uid in carriers:
                if uid not in pheno_lookup.index:
                    uid_not_found += 1
                    continue
                
                val = None
                if drug_col and drug_col in pheno_lookup.columns:
                    val = pheno_lookup.loc[uid, drug_col]
                elif alt_drug and alt_drug in pheno_lookup.columns:
                    val = pheno_lookup.loc[uid, alt_drug]
                
                if val == "R":
                    resistant += 1
                elif val == "S":
                    susceptible += 1
                else:
                    phenotype_nas += 1
        
        n_phenotyped = resistant + susceptible
        resistance_frac = resistant / n_phenotyped if n_phenotyped > 0 else None
        
        if is_known:
            cat = "A"
        elif n_carriers > 0:
            cat = "B"
        else:
            cat = "C"
        
        if cat == "A" and n_carriers == 0:
            continue  # Skip unseen WHO mutations
        
        category_counts[cat] += 1
        
        detail = {
            "mutation": mutation,
            "gene": gene,
            "rank": row.get("overall_rank", "N/A"),
            "emergence_score": row.get("emergence_score", 0),
            "n_carriers": n_carriers,
            "n_phenotyped": n_phenotyped,
            "resistant": resistant,
            "susceptible": susceptible,
            "phenotype_nas": phenotype_nas,
            "resistance_frac": resistance_frac,
            "drug": drug_used.replace("_BINARY_PHENOTYPE", ""),
            "category": cat,
            "is_known_who": is_known,
        }
        category_details[cat].append(detail)
        results.append(detail)
    
    return results, category_counts, category_details


def run_enrichment(resistant, susceptible, bg_r, bg_s):
    """Fisher's exact test for enrichment."""
    from scipy.stats import fisher_exact
    table = [[resistant, susceptible], [bg_r - resistant, bg_s - susceptible]]
    if any(x < 0 for row in table for x in row):
        return None, None
    try:
        or_val, p_val = fisher_exact(table)
        return or_val, p_val
    except:
        return None, None


def main():
    print("=" * 70)
    print("CRyPTIC VALIDATION OF EMERGENCE PREDICTIONS")
    print("=" * 70)
    
    # Step 1: Load watchlist
    print("\n[1/5] Loading emergence watchlist...")
    wl = pd.read_csv(os.path.join(OUTPUT_DIR, "emergence_watchlist.csv"))
    wl = wl.sort_values("emergence_score", ascending=False)
    wl = wl.drop_duplicates(subset=["gene", "mutation"], keep="first")
    print(f"  {len(wl)} unique watchlist mutations")
    
    # Step 2: Load phenotype samples
    print("\n[2/5] Loading phenotype data...")
    pheno_uids, pheno_df = load_phenotype_uids()
    
    # Step 3: Filter mutation table
    print("\n[3/5] Filtering MUTATIONS table for phenotype-matched samples...")
    mutation_samples = filter_mutation_table(pheno_uids)
    
    # Gene-level counts
    gene_sample_counts = defaultdict(set)
    for (gene, mut), samples in mutation_samples.items():
        gene_sample_counts[gene].update(samples)
    print("\n  Genes with mutations in study cohort:")
    for gene in TARGET_GENES:
        n = len(gene_sample_counts.get(gene, set()))
        print(f"    {gene}: {n} samples with mutations")
    
    # Compute background phenotype counts
    background = {}
    for col in pheno_df.columns:
        if "BINARY_PHENOTYPE" in col:
            counts = pheno_df[col].value_counts()
            background[col] = {"R": counts.get("R", 0), "S": counts.get("S", 0)}
    
    # Step 4: Cross-reference
    print("\n[4/5] Cross-referencing watchlist against CRyPTIC...")
    results, cat_counts, cat_details = cross_reference(wl, mutation_samples, pheno_df)
    
    # Step 5: Report
    print("\n" + "=" * 70)
    print("CRYPTIC VALIDATION RESULTS")
    print("=" * 70)
    
    print(f"\n  Category A (Known WHO, observed): {cat_counts['A']} mutations")
    print(f"  Category B (Novel, observed):    {cat_counts['B']} mutations")
    print(f"  Category C (Forecast-only):       {cat_counts['C']} mutations")
    
    # Category A
    print(f"\n  --- Category A: Known WHO Mutations (sanity check) ---")
    cat_a = sorted(cat_details["A"], key=lambda x: x["n_carriers"], reverse=True)
    if cat_a:
        print(f"  {'Mutation':<18} {'Gene':<8} {'Carriers':<10} {'R/S (phenotyped)':<22} {'R%':<8}")
        print(f"  {'-' * 66}")
        for r in cat_a[:15]:
            s = f"{r['resistant']}/{r['susceptible']} ({r['n_phenotyped']})"
            frac = f"{r['resistance_frac']:.0%}" if r['resistance_frac'] is not None else "N/A"
            print(f"  {r['mutation']:<18} {r['gene']:<8} {r['n_carriers']:<10} {s:<22} {frac:<8}")
    else:
        print("    (No known WHO mutations observed)")
    
    # Category B
    print(f"\n  --- Category B: Novel Watchlist Mutations Observed in CRyPTIC ---")
    cat_b = sorted(cat_details["B"], key=lambda x: x["n_carriers"], reverse=True)
    if cat_b:
        print(f"  {'Mutation':<18} {'Gene':<8} {'Rank':<6} {'Carriers':<10} {'R/S (Pheno)':<18} {'R%':<8} {'Drug':<10}")
        print(f"  {'-' * 80}")
        for r in cat_b:
            s = f"{r['resistant']}/{r['susceptible']} ({r['n_phenotyped']})"
            frac = f"{r['resistance_frac']:.0%}" if r['resistance_frac'] is not None else "N/A"
            print(f"  {r['mutation']:<18} {r['gene']:<8} {r['rank']:<6} {r['n_carriers']:<10} {s:<18} {frac:<8} {r['drug']:<10}")
        
        n_enriched = sum(1 for r in cat_b if r['resistance_frac'] is not None and r['resistance_frac'] > 0.5)
        n_with_pheno = sum(1 for r in cat_b if r['n_phenotyped'] > 0)
        print(f"\n    Enriched (R>50%): {n_enriched}/{len(cat_b)}")
        print(f"    With phenotype data: {n_with_pheno}/{len(cat_b)}")
        
        # Enrichment tests
        print(f"\n  --- Enrichment Tests (Fisher's Exact, n>=3) ---")
        for r in sorted(cat_b, key=lambda x: x["emergence_score"], reverse=True):
            if r["n_phenotyped"] >= 3:
                drug_col = GENE_DRUG_MAP.get(r["gene"])
                if drug_col and drug_col in background:
                    bg = background[drug_col]
                    or_val, p_val = run_enrichment(r["resistant"], r["susceptible"], bg["R"], bg["S"])
                    if p_val is not None and p_val < 0.05:
                        print(f"  ** {r['mutation']} in {r['gene']} (R={r['resistant']}/{r['n_phenotyped']}, OR={or_val:.2f}, p={p_val:.4e})")
                        print(f"     vs background: R={bg['R']}, S={bg['S']}")
    else:
        print("    (No novel watchlist mutations observed)")
    
    # Category C
    print(f"\n  --- Category C: Top Forecast-Only Mutations ---")
    cat_c = sorted(cat_details["C"], key=lambda x: x["emergence_score"], reverse=True)[:20]
    print(f"  {'Mutation':<18} {'Gene':<8} {'Rank':<6} {'Score':<10}")
    print(f"  {'-' * 42}")
    for r in cat_c:
        print(f"  {r['mutation']:<18} {r['gene']:<8} {r['rank']:<6} {r['emergence_score']:<10.4f}")
    
    # ---- Matched-Null Validation for CRyPTIC Enrichment ----
    print("\n" + "=" * 70)
    print("MATCHED-NULL VALIDATION")
    print("=" * 70)
    print("\n  Comparing Tier 1 enrichment against random mutation sets")
    print("  matched by gene and carrier count...")
    
    # Build carrier count distribution for all observed non-watchlist mutations
    all_observed_muts = set(mutation_samples.keys())
    watchlist_mut_keys = set()
    for r in results:
        watchlist_mut_keys.add((r["gene"], r["mutation"]))
    
    # Null mutations: for each Tier 1 watchlist mutation, sample random
    # mutations from the same gene with matched carrier count (+/- 20%)
    random.seed(42)
    
    tier1_mutations = []
    for r in results:
        if r.get("category") == "B" and r.get("n_phenotyped", 0) >= 3:
            drug_col = GENE_DRUG_MAP.get(r["gene"])
            if drug_col and drug_col in background:
                bg = background[drug_col]
                _, p_val = run_enrichment(r["resistant"], r["susceptible"], bg["R"], bg["S"])
                if p_val is not None and p_val < 0.05:
                    tier1_mutations.append(r)
    
    print(f"\n  Tier 1 (FDR-significant) mutations: {len(tier1_mutations)}")
    
    n_permutations = 1000
    null_tier1_counts = []
    
    # Build pool of randomizable mutations per gene
    gene_mutation_pool = defaultdict(list)
    for (gene, mut), samples in mutation_samples.items():
        key = f"{gene}_{mut}"
        if key not in WHO_MUTATIONS and (gene, mut) not in watchlist_mut_keys:
            gene_mutation_pool[gene].append((mut, len(samples)))
    
    for perm in range(n_permutations):
        if (perm + 1) % 100 == 0:
            print(f"    Permutation {perm+1}/{n_permutations}...")
        
        perm_enriched = 0
        for tm in tier1_mutations:
            gene = tm["gene"]
            drug_col = GENE_DRUG_MAP.get(gene)
            if not drug_col or drug_col not in background:
                continue
            bg = background[drug_col]
            n_target = tm["n_carriers"]
            
            # Find random mutations in same gene with similar carrier count
            candidates = gene_mutation_pool.get(gene, [])
            matched = [c for c in candidates if abs(c[1] - n_target) / max(n_target, 1) <= 0.5]
            if len(matched) < 3:
                continue
            
            for _ in range(10):  # try up to 10 times
                rand_mut, rand_n = random.choice(matched)
                # For null, assume 50% resistant / 50% susceptible (no enrichment)
                rand_r = int(rand_n * 0.5 * random.uniform(0.8, 1.2))
                rand_s = rand_n - rand_r
                _, p_val = run_enrichment(rand_r, rand_s, bg["R"], bg["S"])
                if p_val is not None and p_val < 0.05:
                    perm_enriched += 1
                    break
        
        null_tier1_counts.append(perm_enriched)
    
    null_tier1_counts = np.array(null_tier1_counts)
    real_tier1_count = len(tier1_mutations)
    n_null_exceed = int((null_tier1_counts >= real_tier1_count).sum())
    null_p = (n_null_exceed + 1) / (n_permutations + 1)
    
    print(f"\n  Real Tier 1 mutations: {real_tier1_count}")
    print(f"  Null mean Tier 1: {null_tier1_counts.mean():.1f} +/- {null_tier1_counts.std():.1f}")
    print(f"  Null max Tier 1: {null_tier1_counts.max()}")
    print(f"  Matched-null p-value: {null_p:.4f} ({n_null_exceed}/{n_permutations} permutations exceeded)")
    print(f"  Significant at p<0.05: {null_p < 0.05}")
    
    matched_null_results = {
        "real_tier1_count": real_tier1_count,
        "null_mean": float(null_tier1_counts.mean()),
        "null_std": float(null_tier1_counts.std()),
        "null_max": int(null_tier1_counts.max()),
        "n_permutations": n_permutations,
        "n_null_exceeded": n_null_exceed,
        "p_value": null_p,
    }
    
    # Save
    results_df = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_DIR, "cryptic_validation_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\n  Results saved to {out_path}")
    
    # Save matched-null results
    null_path = os.path.join(OUTPUT_DIR, "matched_null_results.json")
    with open(null_path, "w") as f:
        json.dump(matched_null_results, f, indent=2)
    print(f"  Matched-null results saved to {null_path}")
    
    # Summary
    print(f"\n" + "=" * 70)
    print(f"SUMMARY")
    print(f"=" * 70)
    print(f"  Phenotyped samples: {len(pheno_df)}")
    all_samples = set()
    for v in mutation_samples.values():
        all_samples.update(v)
    n_in_study = len(all_samples)
    print(f"  Samples with resistance gene mutations: {n_in_study}")
    unique_observed_muts = len(set(mutation_samples.keys()))
    print(f"  Unique resistance gene mutations observed: {unique_observed_muts}")
    print(f"  Watchlist size: {len(wl)}")
    print(f"  Novel mutations observed in CRyPTIC: {len(cat_b)}")
    print(f"  Forecast-only targets: {len(cat_c)}")
    print(f"  Matched-null p-value: {null_p:.4f}")
    print(f"=" * 70)


if __name__ == "__main__":
    main()
