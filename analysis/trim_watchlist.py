"""Create clinical top-20/50 watchlists and test LOO with simpler XGBoost."""
import pandas as pd
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
FORECAST_DIR = BASE / "analysis" / "results" / "forecasting"
HOTSPOT_DIR = BASE / "analysis" / "results" / "hotspot_model"

# ---- Watchlist trimming ----
w = pd.read_csv(FORECAST_DIR / "emergence_watchlist.csv")
top20 = w.head(20).copy()
top20["clinical_rank"] = range(1, 21)
top20_out = top20[["clinical_rank","gene","mutation","emergence_score",
                    "hotspot_score","is_known_resistance","blosum62",
                    "charge_change","is_transition"]]
top20_out.to_csv(FORECAST_DIR / "watchlist_top20.csv", index=False)
print(f"Top 20 saved: {len(top20_out)} entries")

top50 = w.head(50).copy()
top50["clinical_rank"] = range(1, 51)
top50.to_csv(FORECAST_DIR / "watchlist_top50.csv", index=False)
print(f"Top 50 saved: {len(top50)} entries")

print("\nTop 20 clinical watchlist:")
for _, r in top20.iterrows():
    known = " [KNOWN]" if r.get("is_known_resistance", 0) == 1 else ""
    print(f"  #{r.name+1:<3} {r['gene']:<6} {r['mutation']:<10} {r['emergence_score']:.4f}{known}")
