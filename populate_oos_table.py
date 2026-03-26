"""
populate_oos_table.py
─────────────────────────────────────────────────────────────────────────────
Auto-populates the OOS table in geoshock_paper_v8.tex from outputs/oos_results.csv.

Usage
─────
    python populate_oos_table.py                          # default paths
    python populate_oos_table.py --oos outputs/oos_results.csv --paper geoshock_paper_v8.tex

Run AFTER:
    python run.py --skip-var      # generates outputs/oos_results.csv
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np


def fmt_stars(p: float) -> str:
    """Return significance stars for DM p-value."""
    if np.isnan(p):
        return ""
    if p < 0.01:
        return "$^{***}$"
    if p < 0.05:
        return "$^{**}$"
    if p < 0.10:
        return "$^{*}$"
    return ""


def build_table_rows(oos_df: pd.DataFrame) -> str:
    """Build the LaTeX tabular rows for the OOS table."""
    horizons   = [3, 6, 12]
    tail_taus  = [0.05, 0.10]
    lines      = []

    for idx_h, h in enumerate(horizons):
        if idx_h > 0:
            lines.append(r"\midrule")
        for tau in tail_taus:
            row = oos_df[(oos_df["horizon"] == h) & (oos_df["quantile"] == tau)]
            if len(row) == 0:
                lines.append(f" & {tau:.2f} & --- & --- & --- & --- & --- & --- \\\\")
                continue
            row = row.iloc[0]
            n_oos    = int(row["n_oos"])
            pb_base  = f"{row['pb_base']:.3f}"
            pb_enh   = f"{row['pb_enh']:.3f}"
            impr     = row["oos_impr%"]
            dm       = row["dm_stat"]
            pval     = row["dm_pval"]
            stars    = fmt_stars(pval)

            impr_str = f"{impr:.1f}\\%" if not np.isnan(impr) else "---"
            dm_str   = f"{dm:.2f}{stars}" if not np.isnan(dm) else "---"
            p_str    = f"{pval:.3f}" if not np.isnan(pval) else "---"

            # Bold improvement > 5% at left tail
            if not np.isnan(impr) and impr > 5.0 and tau <= 0.10:
                impr_str = f"\\textbf{{{impr:.1f}\\%}}"

            if tau == 0.05:
                h_str = f"\\multirow{{2}}{{*}}{{{h}}}"
            else:
                h_str = ""

            lines.append(
                f" {h_str} & {tau:.2f} & {n_oos} & "
                f"{pb_base} & {pb_enh} & {impr_str} & "
                f"{dm_str} & {p_str} \\\\"
            )

    return "\n".join(lines)


def populate_paper(
    oos_path:   str = "outputs/oos_results.csv",
    paper_path: str = "geoshock_paper_v8.tex",
    dry_run:    bool = False,
) -> None:
    """
    Replace the OOS table rows in the paper with populated values.
    The table is identified by the \\label{tab:oos} marker.
    """
    oos_df = pd.read_csv(oos_path)
    print(f"Loaded OOS results: {len(oos_df)} rows from {oos_path}")

    paper_text = Path(paper_path).read_text(encoding="utf-8")

    # Build populated rows
    new_rows = build_table_rows(oos_df)

    # Print as LaTeX
    print("\n=== POPULATED TABLE ROWS (copy into LaTeX if needed) ===")
    print(new_rows)

    # Also try to auto-replace in the .tex file
    # Pattern: look for \midrule\n...\n\bottomrule inside tab:oos table
    # We mark with the \label{tab:oos} and find the tabular block
    tab_pattern = re.compile(
        r'(\\caption\{Pseudo-Out-of-Sample.*?\\label\{tab:oos\}.*?\\midrule\n)'
        r'(.*?)'
        r'(\\bottomrule)',
        re.DOTALL,
    )
    match = tab_pattern.search(paper_text)
    if not match:
        print("\nCould not find \\label{tab:oos} table block in paper — manual paste needed.")
        return

    old_block = match.group(0)
    new_block = match.group(1) + new_rows + "\n" + match.group(3)

    new_text = paper_text.replace(old_block, new_block)

    # Also replace \oosfill markers if any remain
    n_fills = new_text.count(r"\oosfill")
    if n_fills > 0:
        print(f"\nWarning: {n_fills} \\oosfill markers still in paper after replacement.")

    if dry_run:
        print("\n[DRY RUN] Would have written updated paper.")
        return

    Path(paper_path).write_text(new_text, encoding="utf-8")
    print(f"\nPaper updated: {paper_path}")
    print("Recompile with:  pdflatex geoshock_paper_v8.tex")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oos",   default="outputs/oos_results.csv",
                    help="Path to OOS results CSV")
    ap.add_argument("--paper", default="geoshock_paper_v8.tex",
                    help="Path to LaTeX paper file")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print output but don't write to paper")
    args = ap.parse_args()

    if not Path(args.oos).exists():
        print(f"ERROR: OOS results not found at {args.oos}")
        print("Run first:  python run.py --skip-var")
        sys.exit(1)

    populate_paper(
        oos_path=args.oos,
        paper_path=args.paper,
        dry_run=args.dry_run,
    )


# ─────────────────────────────────────────────────────────────────────────────
def populate_iv_table(
    iv_path:    str = "outputs/iv_results.csv",
    paper_path: str = "geoshock_paper_v8.tex",
) -> None:
    """
    Replace IV table placeholders [F_h3], [pF_h3], etc. in the paper
    with actual values from outputs/iv_results.csv.

    Usage
    -----
    python populate_oos_table.py --iv outputs/iv_results.csv --paper geoshock_paper_v8.tex
    """
    if not Path(iv_path).exists():
        print(f"IV results not found at {iv_path} — skipping IV table population")
        return

    iv_df = pd.read_csv(iv_path)
    print(f"Loaded IV results: {len(iv_df)} rows from {iv_path}")

    paper = Path(paper_path).read_text(encoding="utf-8")

    for _, row in iv_df.iterrows():
        h = int(row["h"])
        paper = paper.replace(f"[F_h{h}]",    f"{row['F_stat']:.1f}")
        paper = paper.replace(f"[pF_h{h}]",   f"{row['F_pval']:.4f}")
        paper = paper.replace(f"[J_h{h}]",    f"{row['Sargan_J']:.2f}")
        paper = paper.replace(f"[pJ_h{h}]",   f"{row['Sargan_p']:.3f}")
        paper = paper.replace(f"[DWH_h{h}]",  f"{row['DWH_t']:.2f}")
        paper = paper.replace(f"[pDWH_h{h}]", f"{row['DWH_p']:.3f}")

    Path(paper_path).write_text(paper, encoding="utf-8")
    print(f"IV table populated in {paper_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oos",   default="outputs/oos_results.csv")
    ap.add_argument("--iv",    default="outputs/iv_results.csv")
    ap.add_argument("--paper", default="geoshock_paper_v8.tex")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if Path(args.oos).exists():
        populate_paper(oos_path=args.oos, paper_path=args.paper, dry_run=args.dry_run)
    else:
        print(f"OOS results not found at {args.oos} — skipping OOS table")

    if not args.dry_run:
        populate_iv_table(iv_path=args.iv, paper_path=args.paper)
