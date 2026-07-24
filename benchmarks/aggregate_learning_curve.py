"""
Aggregate the Pretrained GIN learning curve (reviewer request) into a
plotting-ready CSV and a figure.

Reads all benchmarks/output/learning_curve/results_pretrained_gin_lc_frac*_seed*.json
files (produced by pretrained_gin_learning_curve.py), averages metrics over
seeds within each training-set fraction, and writes:

  metrics/learning_curve_pretrained_gin.csv          (one row per run)
  metrics/learning_curve_pretrained_gin_summary.csv  (mean/std per fraction)
  figures/learning_curve_pretrained_gin.png          (test RMSE/Pearson and
                                                        diastereomer ordering
                                                        accuracy vs train size)
"""

import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
LC_DIR      = os.path.join(SCRIPT_DIR, "output", "learning_curve")
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)
METRICS_DIR = os.path.join(REPO_ROOT, "metrics")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_runs() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(LC_DIR, "results_pretrained_gin_lc_frac*_seed*.json")))
    if not files:
        raise FileNotFoundError(
            f"No learning curve results found in {LC_DIR}.\n"
            f"Run benchmarks/submit_pretrained_gin_learning_curve.sh (or "
            f"pretrained_gin_learning_curve.py directly) first."
        )

    rows = []
    for f in files:
        d = json.load(open(f))
        cfg = d["config"]
        rows.append({
            "frac":                cfg["train_frac"],
            "train_size":          cfg["train_size"],
            "seed":                d["seed"],
            "test_rmse":           d["test_metrics"]["rmse"],
            "test_pearson":        d["test_metrics"]["pearson"],
            "test_spearman":       d["test_metrics"]["spearman"],
            "test_r2":             d["test_metrics"]["r2"],
            "test_mae":            d["test_metrics"]["mae"],
            "stereo_ordering_acc": d["stereo_metrics"]["ordering_acc"],
            "stereo_delta_pearson": d["stereo_metrics"]["delta_pearson"],
            "epochs_run":          d["training"]["epochs_run"],
        })
    df = pd.DataFrame(rows).sort_values(["frac", "seed"]).reset_index(drop=True)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in df.columns if c not in ("frac", "seed", "train_size")]
    agg = df.groupby("frac").agg(
        train_size=("train_size", "mean"),
        n_seeds=("seed", "count"),
        **{f"{c}_mean": (c, "mean") for c in metric_cols},
        **{f"{c}_std":  (c, "std")  for c in metric_cols},
    ).reset_index().sort_values("frac")
    return agg


def plot(summary: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    x = summary["train_size"].values

    ax = axes[0]
    ax.errorbar(x, summary["test_rmse_mean"], yerr=summary["test_rmse_std"].fillna(0),
                marker="o", color="#6baed6", capsize=3, label="Test RMSE")
    ax.set_xscale("log")
    ax.set_xlabel("Training set size (log scale)")
    ax.set_ylabel("Test RMSE (lower is better)")
    ax.set_title("Pretrained GIN — retention time (B) RMSE")
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.errorbar(x, summary["stereo_ordering_acc_mean"], yerr=summary["stereo_ordering_acc_std"].fillna(0),
                 marker="o", color="#e6550d", capsize=3, label="Diastereomer ordering accuracy")
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance (0.5)")
    ax2.set_xscale("log")
    ax2.set_xlabel("Training set size (log scale)")
    ax2.set_ylabel("Ordering accuracy")
    ax2.set_ylim(0.4, 1.0)
    ax2.set_title("Pretrained GIN — diastereomer (D/L) ordering accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle("Learning curve: Pretrained GIN vs. training set size", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Figure saved to {out_path}")


def main() -> None:
    df = load_runs()
    print(f"Loaded {len(df)} runs across {df['frac'].nunique()} fractions "
          f"({sorted(df['frac'].unique())})")

    per_run_path = os.path.join(METRICS_DIR, "learning_curve_pretrained_gin.csv")
    df.to_csv(per_run_path, index=False)
    print(f"Per-run CSV saved to {per_run_path}")

    summary = summarize(df)
    summary_path = os.path.join(METRICS_DIR, "learning_curve_pretrained_gin_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Summary CSV saved to {summary_path}")

    print("\n" + summary[["frac", "train_size", "n_seeds", "test_rmse_mean", "test_rmse_std",
                           "test_pearson_mean", "stereo_ordering_acc_mean", "stereo_ordering_acc_std"]]
          .to_string(index=False))

    fig_path = os.path.join(FIGURES_DIR, "learning_curve_pretrained_gin.png")
    plot(summary, fig_path)


if __name__ == "__main__":
    main()
