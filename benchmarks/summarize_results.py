"""
Summarize benchmark results across seeds and generate comparison figures.
Reads all results_<model>_seed<N>.json files from benchmarks/output/,
averages metrics over seeds, writes a summary CSV/JSON, and saves figures.
"""

import json
import os
import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
FIGURES_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "figures")
METRICS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "metrics")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Pretty display names
# ---------------------------------------------------------------------------
MODEL_NAMES = {
    "results_deeplc":             "DeepLC",
    "results_deeprt_capsnet":     "DeepRT-CapsNet",
    "results_morgan_mlp":         "Morgan FP MLP",
    "results_pretrained_gin":     "Pretrained GIN",
    "results_transformer_scratch": "Transformer (scratch)",
}

# Consistent colour palette (one colour per model)
PALETTE = {
    "DeepLC":                 "#4C72B0",
    "DeepRT-CapsNet":         "#DD8452",
    "Morgan FP MLP":          "#55A868",
    "Pretrained GIN":         "#C44E52",
    "Transformer (scratch)":  "#8172B2",
}

# ---------------------------------------------------------------------------
# 1. Load all result files
# ---------------------------------------------------------------------------
def load_results(output_dir: str) -> dict[str, list[dict]]:
    """Return {benchmark_key: [result_dict, ...]} across all seeds."""
    pattern = os.path.join(output_dir, "results_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No result files found in {output_dir}")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        key = data.get("benchmark", os.path.splitext(os.path.basename(fpath))[0])
        grouped[key].append(data)

    print(f"Loaded {len(files)} files covering {len(grouped)} models:")
    for k, v in grouped.items():
        print(f"  {k}: {len(v)} seeds")
    return dict(grouped)


# ---------------------------------------------------------------------------
# 2. Aggregate metrics over seeds
# ---------------------------------------------------------------------------
METRICS_FLAT = {
    # key in flat record → (section, field)
    "rmse":               ("test_metrics",             "rmse"),
    "mae":                ("test_metrics",             "mae"),
    "pearson":            ("test_metrics",             "pearson"),
    "spearman":           ("test_metrics",             "spearman"),
    "kendall":            ("test_metrics",             "kendall"),
    # stereo pairs (D-Phe vs L-Phe)
    "ordering_acc":       ("stereo_metrics",           "ordering_acc"),
    "delta_pearson":      ("stereo_metrics",           "delta_pearson"),
    "delta_spearman":     ("stereo_metrics",           "delta_spearman"),
    "mean_pred_delta":    ("stereo_metrics",           "mean_pred_delta"),
    "n_correct":          ("stereo_metrics",           "n_correct"),
    "n_pairs":            ("stereo_metrics",           "n_pairs"),
    # tag pairs (with F/f tag vs without)
    "tag_delta_pearson":  ("tag_pair_metrics",         "delta_pearson"),
    "tag_delta_spearman": ("tag_pair_metrics",         "delta_spearman"),
    "tag_delta_rmse":     ("tag_pair_metrics",         "delta_rmse"),
    "tag_delta_mae":      ("tag_pair_metrics",         "delta_mae"),
    "tag_ordering_acc":   ("tag_pair_metrics",         "ordering_acc"),
    # substitution pairs (differ in one position)
    "sub_delta_pearson":  ("substitution_pair_metrics","delta_pearson"),
    "sub_delta_spearman": ("substitution_pair_metrics","delta_spearman"),
    "sub_delta_rmse":     ("substitution_pair_metrics","delta_rmse"),
    "sub_delta_mae":      ("substitution_pair_metrics","delta_mae"),
    "sub_ordering_acc":   ("substitution_pair_metrics","ordering_acc"),
}


def aggregate(grouped: dict[str, list[dict]]) -> pd.DataFrame:
    """Return a DataFrame with mean ± std for every metric, one row per model."""
    rows = []
    for key, results in grouped.items():
        display = MODEL_NAMES.get(key, key)
        seed_vals: dict[str, list[float]] = defaultdict(list)

        for r in results:
            for col, (section, field) in METRICS_FLAT.items():
                val = r.get(section, {}).get(field, None)
                if val is not None:
                    seed_vals[col].append(float(val))

        row = {"model": display, "n_seeds": len(results)}
        for col, vals in seed_vals.items():
            arr = np.array(vals)
            row[f"{col}_mean"] = arr.mean()
            row[f"{col}_std"]  = arr.std(ddof=1) if len(arr) > 1 else 0.0
            row[f"{col}_min"]  = arr.min()
            row[f"{col}_max"]  = arr.max()
        rows.append(row)

    df = pd.DataFrame(rows)
    # Sort by ordering accuracy (highest first); fall back if column absent
    sort_col = "ordering_acc_mean" if "ordering_acc_mean" in df.columns else df.columns[2]
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 3. Save summary files
# ---------------------------------------------------------------------------
def save_summary(df: pd.DataFrame):
    csv_path  = os.path.join(METRICS_DIR, "summary.csv")
    json_path = os.path.join(METRICS_DIR, "summary.json")

    df.to_csv(csv_path, index=False)

    # Also write a human-friendly nested JSON
    summary = {}
    for _, row in df.iterrows():
        model = row["model"]

        def _section(keys):
            out = {}
            for m in keys:
                mean_key = f"{m}_mean"
                if mean_key in row and not np.isnan(row[mean_key]):
                    out[m] = {"mean": round(float(row[f"{m}_mean"]), 4),
                               "std":  round(float(row[f"{m}_std"]),  4)}
            return out

        summary[model] = {
            "n_seeds": int(row["n_seeds"]),
            "test_metrics": _section(["rmse", "mae", "pearson", "spearman", "kendall"]),
            "stereo_metrics": _section(["ordering_acc", "delta_pearson", "delta_spearman",
                                        "mean_pred_delta", "n_correct"]),
            "tag_pair_metrics": _section(["tag_delta_pearson", "tag_delta_spearman",
                                          "tag_delta_rmse", "tag_delta_mae", "tag_ordering_acc"]),
            "substitution_pair_metrics": _section(["sub_delta_pearson", "sub_delta_spearman",
                                                   "sub_delta_rmse", "sub_delta_mae", "sub_ordering_acc"]),
        }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary written to:\n  {csv_path}\n  {json_path}")


# ---------------------------------------------------------------------------
# 4. Plotting helpers
# ---------------------------------------------------------------------------
def _bar_plot(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
    *,
    highlight_best: bool = True,
    higher_is_better: bool = True,
    ylim: tuple | None = None,
):
    mean_col = f"{metric}_mean"
    std_col  = f"{metric}_std"
    if mean_col not in df.columns or df[mean_col].isna().all():
        print(f"  Skipped (no data): {filename}")
        return

    models = df["model"].tolist()
    means  = df[mean_col].tolist()
    stds   = df[std_col].tolist()
    colours = [PALETTE.get(m, "#999999") for m in models]

    if highlight_best:
        best_idx = int(np.argmax(means) if higher_is_better else np.argmin(means))
        edge_colours = ["gold" if i == best_idx else "none" for i in range(len(models))]
        edge_widths  = [2.5    if i == best_idx else 0      for i in range(len(models))]
    else:
        edge_colours = ["none"] * len(models)
        edge_widths  = [0]      * len(models)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(models))
    bars = ax.bar(
        x, means, yerr=stds, capsize=4,
        color=colours, edgecolor=edge_colours, linewidth=edge_widths,
        error_kw=dict(elinewidth=1.2, ecolor="black", capthick=1.2),
        zorder=3,
    )

    # Annotate value on top of each bar
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + (ylim[1] - ylim[0]) * 0.01 if ylim else bar.get_height() + std + 0.005,
            f"{mean:.3f}",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    if ylim:
        ax.set_ylim(*ylim)

    if highlight_best:
        gold_patch = mpatches.Patch(facecolor="none", edgecolor="gold",
                                     linewidth=2.5, label="Best model")
        ax.legend(handles=[gold_patch], fontsize=8, loc="upper right")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _multi_metric_bar(
    df: pd.DataFrame,
    metrics: list[str],
    labels: list[str],
    title: str,
    ylabel: str,
    filename: str,
    *,
    ylim: tuple | None = None,
):
    """Grouped bar chart comparing multiple metrics side-by-side across models."""
    # Drop metrics whose column is entirely absent or NaN
    metrics = [m for m in metrics if f"{m}_mean" in df.columns and not df[f"{m}_mean"].isna().all()]
    labels  = labels[:len(metrics)]
    if not metrics:
        print(f"  Skipped (no data): {filename}")
        return
    models  = df["model"].tolist()
    n_models  = len(models)
    n_metrics = len(metrics)

    x     = np.arange(n_models)
    width = 0.8 / n_metrics
    offsets = np.linspace(-(n_metrics - 1) / 2, (n_metrics - 1) / 2, n_metrics) * width

    fig, ax = plt.subplots(figsize=(9, 5))

    metric_colours = plt.cm.Set2(np.linspace(0, 1, n_metrics))  # type: ignore[attr-defined]

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        means = df[f"{metric}_mean"].tolist()
        stds  = df[f"{metric}_std"].tolist()
        ax.bar(
            x + offsets[i], means, width, yerr=stds, capsize=3,
            label=label, color=metric_colours[i],
            error_kw=dict(elinewidth=1.0, ecolor="black", capthick=1.0),
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    if ylim:
        ax.set_ylim(*ylim)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _strip_plot(grouped: dict[str, list[dict]], metric_section: str, metric_field: str,
                title: str, ylabel: str, filename: str, *, highlight_mean: bool = True):
    """Show individual seed values as strips, with mean/std overlay."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    model_keys = list(MODEL_NAMES.keys())
    display_order = []
    seed_data = []

    for key in model_keys:
        if key not in grouped:
            continue
        vals = [r.get(metric_section, {}).get(metric_field) for r in grouped[key]]
        vals = [v for v in vals if v is not None]
        display_order.append(MODEL_NAMES[key])
        seed_data.append(np.array(vals, dtype=float))

    # Sort by mean (descending for ordering_acc)
    order = np.argsort([-d.mean() for d in seed_data])
    display_order = [display_order[i] for i in order]
    seed_data     = [seed_data[i]     for i in order]

    x = np.arange(len(display_order))
    for i, (name, vals) in enumerate(zip(display_order, seed_data)):
        colour = PALETTE.get(name, "#999999")
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(i + jitter, vals, color=colour, alpha=0.6, s=30, zorder=3)
        if highlight_mean:
            ax.errorbar(i, vals.mean(), yerr=vals.std(ddof=1) if len(vals) > 1 else 0,
                        fmt="D", color="black", capsize=5, capthick=1.5,
                        markersize=6, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(display_order, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    mean_marker = mpatches.Patch(facecolor="black", label="Mean ± std")
    ax.legend(handles=[mean_marker], fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 5. Radar / spider chart for a holistic view
# ---------------------------------------------------------------------------
def _radar_chart(df: pd.DataFrame, filename: str):
    """Normalised radar chart across 8 metrics (higher always = better)."""
    metric_cols  = ["ordering_acc_mean", "pearson_mean", "spearman_mean",
                    "kendall_mean", "delta_pearson_mean", "delta_spearman_mean",
                    "tag_delta_pearson_mean", "sub_delta_pearson_mean"]
    metric_labels = ["Ordering\nAcc", "Pearson", "Spearman", "Kendall",
                     "Stereo\nΔ Pearson", "Stereo\nΔ Spearman",
                     "Tag\nΔ Pearson", "Sub\nΔ Pearson"]
    # Keep only columns that exist (older result files may lack the new ones)
    available = [(c, l) for c, l in zip(metric_cols, metric_labels) if c in df.columns]
    metric_cols, metric_labels = zip(*available) if available else ([], [])

    values = df[list(metric_cols)].values.astype(float)

    # Shift negative values so minimum per column = 0 → max = 1
    col_min = values.min(axis=0)
    col_max = values.max(axis=0)
    denom = np.where(col_max - col_min == 0, 1, col_max - col_min)
    norm = (values - col_min) / denom

    N = len(metric_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for idx, (_, row) in enumerate(df.iterrows()):
        name = row["model"]
        vals = norm[idx].tolist() + [norm[idx][0]]
        colour = PALETTE.get(name, "#999999")
        ax.plot(angles, vals, "o-", linewidth=1.8, color=colour, label=name)
        ax.fill(angles, vals, alpha=0.07, color=colour)

    ax.set_thetagrids(np.degrees(angles[:-1]), metric_labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=7)
    ax.set_title("Normalised performance radar\n(higher = better for all axes)",
                 fontsize=10, fontweight="bold", pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 6. Print a pretty console table
# ---------------------------------------------------------------------------
def _fmt(df: pd.DataFrame, row, col: str, w: int = 8, d: int = 4) -> str:
    """Format mean ± std if columns exist, else '  n/a  '."""
    mc, sc = f"{col}_mean", f"{col}_std"
    if mc not in df.columns or pd.isna(row.get(mc)):
        return f"{'n/a':>{w+7}}"
    return f"{row[mc]:>{w}.{d}f} {row[sc]:>6.{d}f}"


def print_table(df: pd.DataFrame):
    print("\n" + "=" * 90)
    print(f"{'Model':<25} {'Ord.Acc':>9} {'±':>6}  {'Pearson':>9} {'±':>6}  "
          f"{'Spearman':>9} {'±':>6}  {'RMSE':>8} {'±':>6}  {'MAE':>8} {'±':>6}")
    print("=" * 90)
    for _, row in df.iterrows():
        print(
            f"{row['model']:<25} "
            f"{_fmt(df, row, 'ordering_acc')}  "
            f"{_fmt(df, row, 'pearson')}  "
            f"{_fmt(df, row, 'spearman')}  "
            f"{_fmt(df, row, 'rmse', d=3)}  "
            f"{_fmt(df, row, 'mae', d=3)}"
        )
    print("=" * 90)


def print_pair_table(df: pd.DataFrame):
    """Console table for tag-pair and substitution-pair delta metrics."""
    print("\n" + "=" * 110)
    print(f"{'Model':<25} "
          f"{'Tag ΔPearson':>13} {'±':>6}  {'Tag ΔSpear':>11} {'±':>6}  {'Tag OrdAcc':>11} {'±':>6}  "
          f"{'Sub ΔPearson':>13} {'±':>6}  {'Sub ΔSpear':>11} {'±':>6}  {'Sub OrdAcc':>11} {'±':>6}")
    print("=" * 110)
    for _, row in df.iterrows():
        print(
            f"{row['model']:<25} "
            f"{_fmt(df, row, 'tag_delta_pearson', w=13)}  "
            f"{_fmt(df, row, 'tag_delta_spearman', w=11)}  "
            f"{_fmt(df, row, 'tag_ordering_acc', w=11)}  "
            f"{_fmt(df, row, 'sub_delta_pearson', w=13)}  "
            f"{_fmt(df, row, 'sub_delta_spearman', w=11)}  "
            f"{_fmt(df, row, 'sub_ordering_acc', w=11)}"
        )
    print("=" * 110)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Load
    grouped = load_results(OUTPUT_DIR)
    df = aggregate(grouped)

    # Save summary files
    save_summary(df)

    # Console tables
    print_table(df)
    print_pair_table(df)

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------
    print("\nGenerating figures …")

    # --- MOST IMPORTANT: ordering accuracy ---
    _bar_plot(
        df, "ordering_acc",
        title="Stereoisomer Ordering Accuracy (D/L-Phe pairs)",
        ylabel="Ordering Accuracy",
        filename="ordering_accuracy.png",
        higher_is_better=True,
        ylim=(0.45, 0.80),
    )

    _strip_plot(
        grouped, "stereo_metrics", "ordering_acc",
        title="Ordering Accuracy — per-seed distribution",
        ylabel="Ordering Accuracy",
        filename="ordering_accuracy_seeds.png",
    )

    # --- Test metrics (regression) ---
    _bar_plot(
        df, "rmse",
        title="Root Mean Squared Error (lower is better)",
        ylabel="RMSE (minutes)",
        filename="rmse.png",
        higher_is_better=False,
    )

    _bar_plot(
        df, "mae",
        title="Mean Absolute Error (lower is better)",
        ylabel="MAE (minutes)",
        filename="mae.png",
        higher_is_better=False,
    )

    _bar_plot(
        df, "pearson",
        title="Pearson Correlation",
        ylabel="Pearson r",
        filename="pearson.png",
        ylim=(0.4, 1.0),
    )

    _bar_plot(
        df, "spearman",
        title="Spearman Correlation",
        ylabel="Spearman ρ",
        filename="spearman.png",
        ylim=(0.4, 1.0),
    )

    _bar_plot(
        df, "kendall",
        title="Kendall Tau",
        ylabel="Kendall τ",
        filename="kendall.png",
        ylim=(0.2, 0.8),
    )

    # --- Grouped: all correlations together ---
    _multi_metric_bar(
        df,
        metrics=["pearson", "spearman", "kendall"],
        labels=["Pearson r", "Spearman ρ", "Kendall τ"],
        title="Correlation Metrics Comparison",
        ylabel="Correlation",
        filename="correlations_grouped.png",
        ylim=(0.3, 1.0),
    )

    # --- Stereo delta correlations ---
    _multi_metric_bar(
        df,
        metrics=["ordering_acc", "delta_pearson", "delta_spearman"],
        labels=["Ordering Acc", "ΔRT Pearson", "ΔRT Spearman"],
        title="Stereoisomer Discrimination Metrics (D/L-Phe pairs)",
        ylabel="Score",
        filename="stereo_metrics_grouped.png",
    )

    # --- Tag-pair metrics ---
    _bar_plot(
        df, "tag_delta_pearson",
        title="Tag-Pair ΔRT Pearson (tagged vs untagged peptides)",
        ylabel="Pearson r  (predicted Δ vs true Δ)",
        filename="tag_pair_delta_pearson.png",
        higher_is_better=True,
    )

    _multi_metric_bar(
        df,
        metrics=["tag_ordering_acc", "tag_delta_pearson", "tag_delta_spearman"],
        labels=["Ordering Acc", "ΔRT Pearson", "ΔRT Spearman"],
        title="Tag-Pair Discrimination Metrics (tagged vs untagged)",
        ylabel="Score",
        filename="tag_pair_metrics_grouped.png",
    )

    # --- Substitution-pair metrics ---
    _bar_plot(
        df, "sub_delta_pearson",
        title="Substitution-Pair ΔRT Pearson (single-position substitutions)",
        ylabel="Pearson r  (predicted Δ vs true Δ)",
        filename="sub_pair_delta_pearson.png",
        higher_is_better=True,
    )

    _multi_metric_bar(
        df,
        metrics=["sub_ordering_acc", "sub_delta_pearson", "sub_delta_spearman"],
        labels=["Ordering Acc", "ΔRT Pearson", "ΔRT Spearman"],
        title="Substitution-Pair Discrimination Metrics (single-position substitutions)",
        ylabel="Score",
        filename="sub_pair_metrics_grouped.png",
    )

    # --- Cross-pair-type Δ Pearson comparison ---
    _multi_metric_bar(
        df,
        metrics=["delta_pearson", "tag_delta_pearson", "sub_delta_pearson"],
        labels=["Stereo (D/L-Phe)", "Tag (tagged/untagged)", "Substitution (1-pos)"],
        title="ΔRT Pearson Across All Pair Types",
        ylabel="Pearson r  (predicted Δ vs true Δ)",
        filename="delta_pearson_all_pairs.png",
    )

    # --- Cross-pair-type ordering accuracy comparison ---
    _multi_metric_bar(
        df,
        metrics=["ordering_acc", "tag_ordering_acc", "sub_ordering_acc"],
        labels=["Stereo (D/L-Phe)", "Tag (tagged/untagged)", "Substitution (1-pos)"],
        title="Ordering Accuracy Across All Pair Types",
        ylabel="Ordering Accuracy",
        filename="ordering_acc_all_pairs.png",
    )

    # --- Radar chart ---
    _radar_chart(df, "radar_overview.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
