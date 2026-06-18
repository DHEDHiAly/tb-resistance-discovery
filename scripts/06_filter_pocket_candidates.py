"""Extract Tier-4 forecast-only mutations in co-crystal direct binding pockets."""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "analysis" / "results" / "forecasting"

df = pd.read_csv(OUT / "cryptic_tiered_validation.csv")
struct = pd.read_csv(BASE / "analysis" / "results" / "structural_validation_candidates.csv")

tier4_pool = df[(df["tier"] == 4) & (df["rank"] > 20)].merge(
    struct[["gene", "mutation", "drug_distance", "drug", "structure", "residue_pos"]],
    on=["gene", "mutation"],
    how="left",
)

pocket_validated_candidates = tier4_pool[
    (tier4_pool["gene"].isin(["rpoB", "gyrA", "gyrB"]))
    & (tier4_pool["drug_distance"] <= 4.5)
].sort_values(by="emergence_score", ascending=False)

output_path = OUT / "tier4_pocket_direct_matrix.csv"
pocket_validated_candidates.to_csv(output_path, index=False)

print(f"Success: Extracted {len(pocket_validated_candidates)} co-crystal variants.")
print(f"Matrix saved to {output_path} ordered by emergence priority score.")
