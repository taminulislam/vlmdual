"""
Aggregate the results/human_eval_responses.csv produced by the Streamlit app
into paper-ready numbers and figures:

  - Overall distribution of Q1 / Q2 / Q3 responses
  - Breakdown by predicted class (Healthy / Transitional / Acidotic)
  - Breakdown by correctness (correct vs wrong predictions)
  - Fleiss' κ inter-rater agreement (≥2 raters rating the same samples)
  - Correlation between model confidence and perceived quality

Writes:
  results/human_eval_summary.csv
  results/human_eval_summary.tex
  results/human_eval_bars.png
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

Q1_OPTS = ["Yes", "Mostly", "Partially", "No"]
Q2_OPTS = ["Very useful", "Useful", "Somewhat useful", "Not useful"]
Q3_OPTS = ["Definitely", "Probably", "Unsure", "No"]


def _fleiss_kappa(ratings: np.ndarray) -> float:
    """Fleiss' κ for a [n_items, n_categories] count matrix.

    Each row must sum to the same n_raters (but we accept unequal rows by
    using the mean; standard Fleiss assumes equal).
    """
    n = ratings.shape[0]
    r = ratings.sum(axis=1).mean()
    p = ratings.sum(axis=0) / (n * r)
    P = ((ratings**2).sum(axis=1) - r) / (r * (r - 1))
    P_bar = P.mean()
    Pe_bar = (p**2).sum()
    if Pe_bar >= 1:
        return 1.0
    return float((P_bar - Pe_bar) / (1 - Pe_bar))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--responses",
        default="/work/nvme/bgte/tislam6/ACID_Journal/results/human_eval_responses.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="/work/nvme/bgte/tislam6/ACID_Journal/results",
    )
    args = parser.parse_args()

    if not os.path.exists(args.responses):
        print(f"{args.responses} not found — collect responses first")
        return 1

    df = pd.read_csv(args.responses)
    print(f"Responses: {len(df)} from raters: {sorted(df['rater'].unique())}")

    # --- Overall distributions ---
    rows = []
    for qname, col, opts in [
        ("factual_correct", "q1_factual", Q1_OPTS),
        ("clinically_useful", "q2_useful", Q2_OPTS),
        ("would_trust", "q3_trust", Q3_OPTS),
    ]:
        counts = Counter(df[col])
        total = sum(counts.values())
        for opt in opts:
            pct = counts.get(opt, 0) / max(1, total)
            rows.append({"question": qname, "option": opt, "count": counts.get(opt, 0), "pct": pct})
    overall = pd.DataFrame(rows)
    overall.to_csv(os.path.join(args.out_dir, "human_eval_summary.csv"), index=False)

    with open(os.path.join(args.out_dir, "human_eval_summary.tex"), "w") as f:
        f.write(overall.to_latex(index=False, float_format="%.3f"))

    # --- Breakdown by predicted class ---
    by_class_rows = []
    for cls in ["Healthy", "Transitional", "Acidotic"]:
        sub = df[df["pred_name"] == cls]
        if sub.empty:
            continue
        by_class_rows.append(
            {
                "pred": cls,
                "n": len(sub),
                "q1_yes_or_mostly": float(sub["q1_factual"].isin(["Yes", "Mostly"]).mean()),
                "q2_useful_or_very": float(sub["q2_useful"].isin(["Very useful", "Useful"]).mean()),
                "q3_def_or_prob": float(sub["q3_trust"].isin(["Definitely", "Probably"]).mean()),
            }
        )
    pd.DataFrame(by_class_rows).to_csv(
        os.path.join(args.out_dir, "human_eval_by_class.csv"), index=False
    )

    # --- Correct vs wrong ---
    by_correct = df.groupby("correct").agg(
        q1_yes_mostly=("q1_factual", lambda s: s.isin(["Yes", "Mostly"]).mean()),
        q2_useful=("q2_useful", lambda s: s.isin(["Very useful", "Useful"]).mean()),
        q3_trust=("q3_trust", lambda s: s.isin(["Definitely", "Probably"]).mean()),
        n=("sample_id", "count"),
    )
    by_correct.to_csv(os.path.join(args.out_dir, "human_eval_by_correctness.csv"))

    # --- Fleiss κ on duplicate items (when ≥2 raters rated the same sample_id) ---
    fleiss_rows = []
    for qname, col, opts in [
        ("factual_correct", "q1_factual", Q1_OPTS),
        ("clinically_useful", "q2_useful", Q2_OPTS),
        ("would_trust", "q3_trust", Q3_OPTS),
    ]:
        counts = []
        for sid, sub in df.groupby("sample_id"):
            if len(sub) < 2:
                continue
            vec = [int((sub[col] == o).sum()) for o in opts]
            counts.append(vec)
        if len(counts) >= 2:
            kappa = _fleiss_kappa(np.array(counts, dtype=np.int64))
        else:
            kappa = float("nan")
        fleiss_rows.append({"question": qname, "n_items_with_2+_raters": len(counts), "fleiss_kappa": kappa})
    pd.DataFrame(fleiss_rows).to_csv(
        os.path.join(args.out_dir, "human_eval_fleiss_kappa.csv"), index=False
    )

    # --- Confidence vs perceived quality (Q1 as ordinal) ---
    q1_rank = {"Yes": 4, "Mostly": 3, "Partially": 2, "No": 1}
    df["q1_rank"] = df["q1_factual"].map(q1_rank)
    corr = df[["confidence", "q1_rank"]].corr().iloc[0, 1]
    print(f"Confidence vs Q1 rank Spearman-ish corr: {corr:.3f}")

    # --- Bar chart of overall responses ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (qname, col, opts) in zip(
        axes,
        [
            ("Q1: factual correctness", "q1_factual", Q1_OPTS),
            ("Q2: clinical usefulness", "q2_useful", Q2_OPTS),
            ("Q3: would you trust", "q3_trust", Q3_OPTS),
        ],
    ):
        counts = Counter(df[col])
        heights = [counts.get(o, 0) for o in opts]
        ax.bar(opts, heights, color="C0")
        ax.set_title(qname)
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "human_eval_bars.png"), dpi=150)
    plt.close(fig)

    print("Wrote human_eval_summary.csv, by_class, by_correctness, fleiss_kappa, bars.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
