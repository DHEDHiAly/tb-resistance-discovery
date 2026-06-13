"""
TB Resistance Discovery — Master Pipeline.
Runs the full analysis end-to-end on demo data and generates
all results, figures, and candidate lists for presentation.
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REQUIREMENTS = ["numpy", "pandas", "scipy", "statsmodels",
                "scikit-learn", "umap-learn", "matplotlib",
                "seaborn", "jupyter", "nbformat"]


def check_dependencies():
    missing = []
    for pkg in REQUIREMENTS:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)

    if missing:
        print("Missing dependencies. Install with:")
        print(f"  pip install {' '.join(missing)}")
        print("Or using conda:")
        print(f"  conda install -c conda-forge {' '.join(missing)}")
        return False
    return True


def run_step(step_name: str, script: str, args: list[str] | None = None) -> bool:
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Step: {step_name}")
    print(f"{'='*60}")

    cmd = [sys.executable, script]
    if args:
        cmd.extend(args)

    result = subprocess.run(cmd, capture_output=False, text=True)
    success = result.returncode == 0

    if success:
        print(f"  ✓ {step_name} completed successfully")
    else:
        print(f"  ✗ {step_name} failed (exit code {result.returncode})")

    return success


def generate_summary(output_dir: str):
    """Generate a summary markdown of all findings."""
    import pandas as pd

    out = Path(output_dir)
    lines = []
    lines.append("# TB Resistance Discovery — Analysis Summary")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    assoc_path = out / "association_results.csv"
    if assoc_path.exists():
        df = pd.read_csv(assoc_path)
        sig = df[df["significant"] == True]
        novel_sig = sig[sig["is_novel_candidate"] == True]
        lines.append(f"## Association Testing")
        lines.append(f"- Mutations tested: {len(df)}")
        lines.append(f"- Significant (BH-adjusted p<0.05): {len(sig)}")
        lines.append(f"- Novel candidates in significant set: {len(novel_sig)}")
        lines.append("")

        if not novel_sig.empty:
            lines.append("### Top Novel Candidate Mutations")
            lines.append("| Gene | Mutation | OR | p_adj | Drug Association |")
            lines.append("|------|----------|----|-------|-----------------|")
            for _, r in novel_sig.sort_values("p_corrected").iterrows():
                lines.append(
                    f"| {r['gene']} | {r.get('protein_change', r['mutation'])} "
                    f"| {r['odds_ratio']:.2f} | {r['p_corrected']:.4f} "
                    f"| {r.get('drug_association', 'unknown')} |"
                )
            lines.append("")

    embed_dir = out / "figures"
    pngs = list(embed_dir.glob("*.png"))
    if pngs:
        lines.append(f"## Figures Generated ({len(pngs)})")
        for p in sorted(pngs):
            lines.append(f"- `{p.relative_to(out.parent)}`")
        lines.append("")

    (out / "SUMMARY.md").write_text("\n".join(lines))
    print(f"\nSummary written to {out / 'SUMMARY.md'}")


def main():
    parser = argparse.ArgumentParser(
        description="TB Resistance Discovery — Full Pipeline"
    )
    parser.add_argument("--skip-embedding", action="store_true",
                       help="Skip embedding step (requires umap-learn)")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent

    if not check_dependencies():
        sys.exit(1)

    steps = [
        ("Association Testing",
         str(base / "scripts" / "association_analysis.py"),
         ["--mutations", str(base / "data" / "demo" / "mutation_matrix.csv"),
          "--clinical", str(base / "data" / "demo" / "clinical_metadata.csv"),
          "--output", str(base / "analysis" / "results" / "association_results.csv")]),
    ]

    if not args.skip_embedding:
        steps.append(("Latent Space Embedding",
                     str(base / "scripts" / "embedding_analysis.py"),
                     ["--mutations", str(base / "data" / "demo" / "mutation_matrix.csv"),
                      "--clinical", str(base / "data" / "demo" / "clinical_metadata.csv"),
                      "--output", str(base / "analysis" / "results")]))

    all_ok = True
    for step_name, script, step_args in steps:
        ok = run_step(step_name, script, step_args)
        if not ok:
            all_ok = False
            print(f"  Pipeline halted at step: {step_name}")
            break

    if all_ok:
        generate_summary(str(base / "analysis" / "results"))
        print(f"\n{'='*60}")
        print("Pipeline complete. All outputs in analysis/results/")
        print(f"{'='*60}")
    else:
        print("\nPipeline completed with errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
