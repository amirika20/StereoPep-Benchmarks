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
# Pretty display names and fixed model order
# ---------------------------------------------------------------------------
MODEL_NAMES = {
    "results_gin_scratch":          "GIN",
    "results_pretrained_gin":       "Pretrained GIN",
    "results_pepland":              "Pretrained PepLand",
    "results_hybrid_base":          "PeptideCLM-2-Hybrid",
    "results_esm3_sm_embedding":    "ESM3-small",
    "results_esmc_300m_embedding":  "ESMC-300M",
    "results_esmc_600m_embedding":  "ESMC-600M",
    "results_transformer_scratch":  "Transformer",
    "results_deeplc":               "DeepLC",
    "results_deeprt_capsnet":       "DeepRT-CapsNet",
    "results_morgan_mlp":           "Morgan FP MLP",
}

MODEL_ORDER = [
    "GIN",
    "Pretrained GIN",
    "Pretrained PepLand",
    "PeptideCLM-2-Hybrid",
    "ESM3-small",
    "ESMC-300M",
    "ESMC-600M",
    "Transformer",
    "DeepLC",
    "DeepRT-CapsNet",
    "Morgan FP MLP",
]

# Consistent colour palette (one colour per model)
PALETTE = {
    "GIN":                "#2171b5",
    "Pretrained GIN":     "#6baed6",
    "Pretrained PepLand": "#17becf",
    "PeptideCLM-2-Hybrid": "#e377c2",
    "ESM3-small":         "#238b45",
    "ESMC-300M":          "#74c476",
    "ESMC-600M":          "#00441b",
    "Transformer":        "#7a51a1",
    "DeepLC":             "#e6550d",
    "DeepRT-CapsNet":     "#c44e52",
    "Morgan FP MLP":      "#8c6d31",
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
    "mse":          ("test_metrics", "mse"),
    "rmse":         ("test_metrics", "rmse"),
    "mae":          ("test_metrics", "mae"),
    "mean_error":   ("test_metrics", "mean_error"),
    "r2":           ("test_metrics", "r2"),
    "pearson":      ("test_metrics", "pearson"),
    "spearman":     ("test_metrics", "spearman"),
    "kendall":      ("test_metrics", "kendall"),
    # stereo pairs test (D-Phe vs L-Phe)
    "ordering_acc":       ("stereo_metrics",           "ordering_acc"),
    "delta_pearson":      ("stereo_metrics",           "delta_pearson"),
    "delta_spearman":     ("stereo_metrics",           "delta_spearman"),
    "delta_kendall":      ("stereo_metrics",           "delta_kendall"),
    "delta_rmse":         ("stereo_metrics",           "delta_rmse"),
    "delta_mae":          ("stereo_metrics",           "delta_mae"),
    "delta_auc":          ("stereo_metrics",           "delta_auc"),
    "mean_pred_delta":    ("stereo_metrics",           "mean_pred_delta"),
    "n_correct":          ("stereo_metrics",           "n_correct"),
    "n_pairs":            ("stereo_metrics",           "n_pairs"),
    # stereo pairs trainval (D-Phe vs L-Phe, on training+val data)
    "tv_ordering_acc":    ("stereo_trainval_metrics",  "ordering_acc"),
    "tv_delta_pearson":   ("stereo_trainval_metrics",  "delta_pearson"),
    "tv_delta_spearman":  ("stereo_trainval_metrics",  "delta_spearman"),
    "tv_delta_kendall":   ("stereo_trainval_metrics",  "delta_kendall"),
    "tv_delta_rmse":      ("stereo_trainval_metrics",  "delta_rmse"),
    "tv_delta_mae":       ("stereo_trainval_metrics",  "delta_mae"),
    "tv_delta_auc":       ("stereo_trainval_metrics",  "delta_auc"),
    "tv_mean_pred_delta": ("stereo_trainval_metrics",  "mean_pred_delta"),
    "tv_n_correct":       ("stereo_trainval_metrics",  "n_correct"),
    "tv_n_pairs":         ("stereo_trainval_metrics",  "n_pairs"),
    # tag pairs (with F/f tag vs without)
    "tag_delta_pearson":  ("tag_pair_metrics",         "delta_pearson"),
    "tag_delta_spearman": ("tag_pair_metrics",         "delta_spearman"),
    "tag_delta_kendall":  ("tag_pair_metrics",         "delta_kendall"),
    "tag_delta_rmse":     ("tag_pair_metrics",         "delta_rmse"),
    "tag_delta_mae":      ("tag_pair_metrics",         "delta_mae"),
    "tag_delta_auc":      ("tag_pair_metrics",         "delta_auc"),
    "tag_ordering_acc":   ("tag_pair_metrics",         "ordering_acc"),
    # substitution pairs (differ in one position)
    "sub_delta_pearson":  ("substitution_pair_metrics","delta_pearson"),
    "sub_delta_spearman": ("substitution_pair_metrics","delta_spearman"),
    "sub_delta_kendall":  ("substitution_pair_metrics","delta_kendall"),
    "sub_delta_rmse":     ("substitution_pair_metrics","delta_rmse"),
    "sub_delta_mae":      ("substitution_pair_metrics","delta_mae"),
    "sub_delta_auc":      ("substitution_pair_metrics","delta_auc"),
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
    # Sort by MODEL_ORDER; unknown models go to the end
    order_map = {name: i for i, name in enumerate(MODEL_ORDER)}
    df["_sort_key"] = df["model"].map(lambda m: order_map.get(m, len(MODEL_ORDER)))
    df = df.sort_values("_sort_key").drop(columns=["_sort_key"]).reset_index(drop=True)
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
            "test_metrics": _section(["mse", "rmse", "mae", "mean_error", "r2",
                                       "pearson", "spearman", "kendall"]),
            "stereo_metrics": _section(["ordering_acc", "delta_pearson", "delta_spearman",
                                        "mean_pred_delta", "n_correct"]),
            "stereo_trainval_metrics": _section(["tv_ordering_acc", "tv_delta_pearson",
                                                 "tv_delta_spearman", "tv_mean_pred_delta",
                                                 "tv_n_correct"]),
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

    # Collect data in MODEL_ORDER
    key_by_display = {v: k for k, v in MODEL_NAMES.items()}
    display_order = []
    seed_data = []

    for display_name in MODEL_ORDER:
        key = key_by_display.get(display_name)
        if key is None or key not in grouped:
            continue
        vals = [r.get(metric_section, {}).get(metric_field) for r in grouped[key]]
        vals = [v for v in vals if v is not None]
        display_order.append(display_name)
        seed_data.append(np.array(vals, dtype=float))

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
                    "kendall_mean", "delta_auc_mean", "delta_spearman_mean",
                    "tag_ordering_acc_mean", "sub_ordering_acc_mean"]
    metric_labels = ["Diast.\nAcc", "Pearson", "Spearman", "Kendall",
                     "Diast.\nAUC", "Diast.\nΔ Spearman",
                     "Add.Mut.\nAcc", "Pt.Mut.\nAcc"]
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
    print("\n" + "=" * 120)
    print(f"{'Model':<25} {'Pearson':>9} {'±':>6}  {'Spearman':>9} {'±':>6}  {'Kendall':>9} {'±':>6}  "
          f"{'R²':>8} {'±':>6}  {'RMSE':>8} {'±':>6}  {'MSE':>8} {'±':>6}  "
          f"{'MAE':>8} {'±':>6}  {'Mean Err':>9} {'±':>6}")
    print("=" * 120)
    for _, row in df.iterrows():
        print(
            f"{row['model']:<25} "
            f"{_fmt(df, row, 'pearson')}  "
            f"{_fmt(df, row, 'spearman')}  "
            f"{_fmt(df, row, 'kendall')}  "
            f"{_fmt(df, row, 'r2', d=3)}  "
            f"{_fmt(df, row, 'rmse', d=3)}  "
            f"{_fmt(df, row, 'mse', d=3)}  "
            f"{_fmt(df, row, 'mae', d=3)}  "
            f"{_fmt(df, row, 'mean_error', d=3)}"
        )
    print("=" * 120)


def print_pair_table(df: pd.DataFrame):
    """Console table for addition-mutation-pair and point-mutation-pair delta metrics."""
    print("\n" + "=" * 110)
    print(f"{'Model':<25} "
          f"{'Add.Mut. ΔPearson':>17} {'±':>6}  {'Add.Mut. ΔSpear':>15} {'±':>6}  {'Add.Mut. PwAcc':>14} {'±':>6}  "
          f"{'Pt.Mut. ΔPearson':>16} {'±':>6}  {'Pt.Mut. ΔSpear':>14} {'±':>6}  {'Pt.Mut. PwAcc':>13} {'±':>6}")
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
# 7. LaTeX table generation
# ---------------------------------------------------------------------------
def _latex_table(
    df: pd.DataFrame,
    columns: list[tuple[str, str]],   # [(metric_key, header), ...]
    caption: str,
    label: str,
) -> str:
    """
    Build a NeurIPS-compliant LaTeX table.

    Follows NeurIPS formatting rules:
    - Caption (lower-case except first word) placed BEFORE the tabular.
    - No vertical rules anywhere.
    - Only \toprule, \midrule (after header), \bottomrule — no rules between rows.
    - Column headers rotated 45° to save horizontal space.
    - Each data cell uses \makecell: mean on line 1, ±std on line 2.
    - Best value per column is bold.
    - Whole table set in \small.

    Required packages:
        \\usepackage{booktabs}
        \\usepackage{makecell}
        \\usepackage{graphicx}
    """
    def _higher(key: str) -> bool:
        low_keys = {"mse", "rmse", "mae", "mean_error", "error"}
        return not any(k in key.lower() for k in low_keys)

    def _closest_to_zero(key: str) -> bool:
        return "mean_error" in key.lower()

    # Determine best row index per metric column
    best: dict[str, int] = {}
    for key, _ in columns:
        mc = f"{key}_mean"
        if mc not in df.columns or df[mc].isna().all():
            continue
        vals = df[mc].values.astype(float)
        with np.errstate(invalid="ignore"):
            if _closest_to_zero(key):
                best[key] = int(np.nanargmin(np.abs(vals)))
            else:
                best[key] = int(np.nanargmax(vals) if _higher(key) else np.nanargmin(vals))

    # No vertical rules — l for model name, c for each metric
    col_spec = "l" + "c" * len(columns)

    # Rotated headers to keep columns narrow
    rotated_headers = " & ".join(
        [r"\textbf{Model}"]
        + [r"\rotatebox{45}{\textbf{" + h + r"}}" for _, h in columns]
    )

    rows_tex = []
    for ridx, (_, row) in enumerate(df.iterrows()):
        cells = [row["model"].replace("_", r"\_")]
        for key, _ in columns:
            mc, sc = f"{key}_mean", f"{key}_std"
            if mc not in df.columns or pd.isna(row.get(mc)):
                cells.append("--")
            else:
                mean_str = f"{row[mc]:.3f}"
                std_str  = f"$\\pm${row[sc]:.3f}"
                cell = r"\makecell{" + mean_str + r" \\ " + std_str + r"}"
                if best.get(key) == ridx:
                    cell = r"\textbf{" + cell + r"}"
                cells.append(cell)
        # NeurIPS: no \midrule between data rows
        rows_tex.append("  " + " & ".join(cells) + r" \\")

    body = "\n".join(rows_tex)

    tex = (
        r"\begin{table}[htbp]" + "\n"
        r"  \caption{" + caption + r"}" + "\n"
        r"  \label{" + label + r"}" + "\n"
        r"  \centering" + "\n"
        r"  \small" + "\n"
        r"  \begin{tabular}{" + col_spec + r"}" + "\n"
        r"    \toprule" + "\n"
        f"    {rotated_headers} \\\\\n"
        r"    \midrule" + "\n"
        + body + "\n"
        r"    \bottomrule" + "\n"
        r"  \end{tabular}" + "\n"
        r"\end{table}" + "\n"
    )
    return tex


def save_latex_tables(df: pd.DataFrame):
    """Write four .tex files — one per evaluation category."""

    tables = [
        # ---- 1. Overall regression performance --------------------------------
        (
            [
                ("pearson",    "Pearson $r$"),
                ("spearman",   "Spearman $\\rho$"),
                ("kendall",    "Kendall $\\tau$"),
                ("r2",         "$R^2$"),
                ("rmse",       "RMSE"),
                ("mae",        "MAE"),
                ("mean_error", "Mean Error"),
            ],
            "Overall regression performance on B\\% prediction (mean $\\pm$ std over seeds; "
            "bold denotes best per column).",
            "tab:overall_performance",
            "latex_overall_performance.tex",
        ),
        # ---- 2. Diastereomer performance (test) --------------------------------
        (
            [
                ("ordering_acc",   "Pairwise Acc."),
                ("delta_pearson",  "$\\Delta$ Pearson"),
                ("delta_spearman", "$\\Delta$ Spearman"),
                ("delta_kendall",  "$\\Delta$ Kendall"),
                ("delta_auc",      "$\\Delta$ AUC"),
                ("delta_rmse",     "$\\Delta$ RMSE"),
                ("delta_mae",      "$\\Delta$ MAE"),
            ],
            "Diastereomer-pair performance (D/L-Phe pairs, test set; mean $\\pm$ std over seeds; "
            "bold denotes best per column).",
            "tab:diastereomer_performance",
            "latex_diastereomer_performance.tex",
        ),
        # ---- 2b. Diastereomer performance (trainval) --------------------------
        (
            [
                ("tv_ordering_acc",   "Pairwise Acc."),
                ("tv_delta_pearson",  "$\\Delta$ Pearson"),
                ("tv_delta_spearman", "$\\Delta$ Spearman"),
                ("tv_delta_kendall",  "$\\Delta$ Kendall"),
                ("tv_delta_auc",      "$\\Delta$ AUC"),
                ("tv_delta_rmse",     "$\\Delta$ RMSE"),
                ("tv_delta_mae",      "$\\Delta$ MAE"),
            ],
            "Diastereomer-pair performance (D/L-Phe pairs, train+val set; mean $\\pm$ std over seeds; "
            "bold denotes best per column).",
            "tab:diastereomer_trainval_performance",
            "latex_diastereomer_trainval_performance.tex",
        ),
        # ---- 2c. Trainval stereo results (ordered by pairwise acc) ------------
        (
            [
                ("tv_ordering_acc",   "Pairwise Acc."),
                ("tv_delta_auc",      "$\\Delta$ AUC"),
                ("tv_delta_pearson",  "$\\Delta$ Pearson"),
                ("tv_delta_spearman", "$\\Delta$ Spearman"),
                ("tv_delta_kendall",  "$\\Delta$ Kendall"),
                ("tv_delta_rmse",     "$\\Delta$ RMSE"),
                ("tv_delta_mae",      "$\\Delta$ MAE"),
            ],
            "Trainval stereo results: diastereomer-pair performance on train+val set "
            "(mean $\\pm$ std over seeds; bold denotes best per column).",
            "tab:trainval_stereo_results",
            "latex_trainval_stereo_results.tex",
        ),
        # ---- 3. Point-mutation (substitution) performance ---------------------
        (
            [
                ("sub_ordering_acc",   "Pairwise Acc."),
                ("sub_delta_pearson",  "$\\Delta$ Pearson"),
                ("sub_delta_spearman", "$\\Delta$ Spearman"),
                ("sub_delta_kendall",  "$\\Delta$ Kendall"),
                ("sub_delta_auc",      "$\\Delta$ AUC"),
                ("sub_delta_rmse",     "$\\Delta$ RMSE"),
                ("sub_delta_mae",      "$\\Delta$ MAE"),
            ],
            "Point-mutation pair performance (single amino-acid substitutions; "
            "mean $\\pm$ std over seeds; bold denotes best per column).",
            "tab:point_mutation_performance",
            "latex_point_mutation_performance.tex",
        ),
        # ---- 4. Tagging (addition-mutation) performance -----------------------
        (
            [
                ("tag_ordering_acc",   "Pairwise Acc."),
                ("tag_delta_pearson",  "$\\Delta$ Pearson"),
                ("tag_delta_spearman", "$\\Delta$ Spearman"),
                ("tag_delta_kendall",  "$\\Delta$ Kendall"),
                ("tag_delta_auc",      "$\\Delta$ AUC"),
                ("tag_delta_rmse",     "$\\Delta$ RMSE"),
                ("tag_delta_mae",      "$\\Delta$ MAE"),
            ],
            "Tagging (addition-mutation) pair performance (peptides with/without "
            "F/f tag; mean $\\pm$ std over seeds; bold denotes best per column).",
            "tab:tagging_performance",
            "latex_tagging_performance.tex",
        ),
    ]

    print("\nGenerating LaTeX tables …")
    for cols, caption, label, fname in tables:
        tex = _latex_table(df, cols, caption, label)
        path = os.path.join(METRICS_DIR, fname)
        with open(path, "w") as f:
            f.write(tex)
        print(f"  Saved: {path}")


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

    # LaTeX tables
    save_latex_tables(df)

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------
    print("\nGenerating figures …")

    # --- MOST IMPORTANT: pairwise accuracy (test) ---
    _bar_plot(
        df, "ordering_acc",
        title="Diastereomer Pair Pairwise Accuracy — Test Set (D/L-Phe pairs)",
        ylabel="Pairwise Accuracy",
        filename="pairwise_accuracy_diastereomer.png",
        higher_is_better=True,
        ylim=(0.45, 0.80),
    )

    _strip_plot(
        grouped, "stereo_metrics", "ordering_acc",
        title="Diastereomer Pair Pairwise Accuracy (test) — per-seed distribution",
        ylabel="Pairwise Accuracy",
        filename="pairwise_accuracy_diastereomer_seeds.png",
    )

    # --- Pairwise accuracy on trainval (seen data — overfitting diagnostic) ---
    _bar_plot(
        df, "tv_ordering_acc",
        title="Diastereomer Pair Pairwise Accuracy — Train+Val Set (D/L-Phe pairs)",
        ylabel="Pairwise Accuracy",
        filename="pairwise_accuracy_diastereomer_trainval.png",
        higher_is_better=True,
        ylim=(0.45, 1.00),
    )

    _strip_plot(
        grouped, "stereo_trainval_metrics", "ordering_acc",
        title="Diastereomer Pair Pairwise Accuracy (train+val) — per-seed distribution",
        ylabel="Pairwise Accuracy",
        filename="pairwise_accuracy_diastereomer_trainval_seeds.png",
    )

    # --- Test vs trainval pairwise accuracy comparison ---
    _multi_metric_bar(
        df,
        metrics=["ordering_acc", "tv_ordering_acc"],
        labels=["Test", "Train+Val"],
        title="Diastereomer Pairwise Accuracy: Test vs Train+Val",
        ylabel="Pairwise Accuracy",
        filename="pairwise_accuracy_test_vs_trainval.png",
    )

    # --- Test metrics (regression) ---
    _bar_plot(
        df, "mse",
        title="Mean Squared Error (lower is better)",
        ylabel="MSE (minutes²)",
        filename="mse.png",
        higher_is_better=False,
    )

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
        df, "mean_error",
        title="Mean Error / Bias (closer to 0 is better)",
        ylabel="Mean Error (minutes)",
        filename="mean_error.png",
        higher_is_better=False,
        highlight_best=False,
    )

    _bar_plot(
        df, "r2",
        title="R² Score (higher is better)",
        ylabel="R²",
        filename="r2.png",
        higher_is_better=True,
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
        title="Kendall τ",
        ylabel="Kendall τ",
        filename="kendall.png",
        ylim=(0.2, 0.8),
    )

    # --- Grouped: all correlations together ---
    _multi_metric_bar(
        df,
        metrics=["pearson", "spearman", "kendall"],
        labels=["Pearson r", "Spearman ρ", "Kendall τ"],
        title="Correlation Metrics",
        ylabel="Correlation",
        filename="correlations_grouped.png",
        ylim=(0.3, 1.0),
    )

    # --- Grouped: all error metrics together ---
    _multi_metric_bar(
        df,
        metrics=["mse", "rmse", "mae"],
        labels=["MSE", "RMSE", "MAE"],
        title="Error Metrics (lower is better)",
        ylabel="Error (minutes / minutes²)",
        filename="error_metrics_grouped.png",
    )

    # --- Diastereomer pair pairwise metrics (test) ---
    _multi_metric_bar(
        df,
        metrics=["ordering_acc", "delta_kendall", "delta_spearman", "delta_auc"],
        labels=["Pairwise Accuracy", "Kendall τ", "Spearman ρ", "AUC"],
        title="Diastereomer Pair Pairwise Metrics — Test Set (D/L-Phe pairs)",
        ylabel="Score",
        filename="diastereomer_metrics_grouped.png",
    )

    _multi_metric_bar(
        df,
        metrics=["delta_mae", "delta_rmse"],
        labels=["MAE on Δ", "RMSE on Δ"],
        title="Diastereomer Pair ΔRT Error — Test Set (D/L-Phe pairs)",
        ylabel="Error (minutes)",
        filename="diastereomer_delta_error.png",
    )

    # --- Diastereomer pair pairwise metrics (trainval) ---
    _multi_metric_bar(
        df,
        metrics=["tv_ordering_acc", "tv_delta_kendall", "tv_delta_spearman", "tv_delta_auc"],
        labels=["Pairwise Accuracy", "Kendall τ", "Spearman ρ", "AUC"],
        title="Diastereomer Pair Pairwise Metrics — Train+Val Set (D/L-Phe pairs)",
        ylabel="Score",
        filename="diastereomer_metrics_trainval_grouped.png",
    )

    _multi_metric_bar(
        df,
        metrics=["tv_delta_mae", "tv_delta_rmse"],
        labels=["MAE on Δ", "RMSE on Δ"],
        title="Diastereomer Pair ΔRT Error — Train+Val Set (D/L-Phe pairs)",
        ylabel="Error (minutes)",
        filename="diastereomer_delta_error_trainval.png",
    )

    # --- Addition mutation pair metrics ---
    _bar_plot(
        df, "tag_delta_pearson",
        title="Addition Mutation Pair ΔRT Pearson",
        ylabel="Pearson r  (predicted Δ vs true Δ)",
        filename="addition_mutation_delta_pearson.png",
        higher_is_better=True,
    )

    _multi_metric_bar(
        df,
        metrics=["tag_ordering_acc", "tag_delta_kendall", "tag_delta_spearman", "tag_delta_auc"],
        labels=["Pairwise Accuracy", "Kendall τ", "Spearman ρ", "AUC"],
        title="Addition Mutation Pair Pairwise Metrics",
        ylabel="Score",
        filename="addition_mutation_metrics_grouped.png",
    )

    _multi_metric_bar(
        df,
        metrics=["tag_delta_mae", "tag_delta_rmse"],
        labels=["MAE on Δ", "RMSE on Δ"],
        title="Addition Mutation Pair ΔRT Error",
        ylabel="Error (minutes)",
        filename="addition_mutation_delta_error.png",
    )

    # --- Point mutation pair metrics ---
    _bar_plot(
        df, "sub_delta_pearson",
        title="Point Mutation Pair ΔRT Pearson",
        ylabel="Pearson r  (predicted Δ vs true Δ)",
        filename="point_mutation_delta_pearson.png",
        higher_is_better=True,
    )

    _multi_metric_bar(
        df,
        metrics=["sub_ordering_acc", "sub_delta_kendall", "sub_delta_spearman", "sub_delta_auc"],
        labels=["Pairwise Accuracy", "Kendall τ", "Spearman ρ", "AUC"],
        title="Point Mutation Pair Pairwise Metrics",
        ylabel="Score",
        filename="point_mutation_metrics_grouped.png",
    )

    _multi_metric_bar(
        df,
        metrics=["sub_delta_mae", "sub_delta_rmse"],
        labels=["MAE on Δ", "RMSE on Δ"],
        title="Point Mutation Pair ΔRT Error",
        ylabel="Error (minutes)",
        filename="point_mutation_delta_error.png",
    )

    # --- Cross-pair-type Δ Pearson comparison ---
    _multi_metric_bar(
        df,
        metrics=["delta_pearson", "tag_delta_pearson", "sub_delta_pearson"],
        labels=["Diastereomer", "Addition Mutation", "Point Mutation"],
        title="ΔRT Pearson Across All Pair Types",
        ylabel="Pearson r  (predicted Δ vs true Δ)",
        filename="delta_pearson_all_pairs.png",
    )

    # --- Cross-pair-type pairwise accuracy comparison ---
    _multi_metric_bar(
        df,
        metrics=["ordering_acc", "tag_ordering_acc", "sub_ordering_acc"],
        labels=["Diastereomer", "Addition Mutation", "Point Mutation"],
        title="Pairwise Accuracy Across All Pair Types",
        ylabel="Pairwise Accuracy",
        filename="pairwise_accuracy_all_pairs.png",
    )

    # --- Cross-pair-type AUC comparison ---
    _multi_metric_bar(
        df,
        metrics=["delta_auc", "tag_delta_auc", "sub_delta_auc"],
        labels=["Diastereomer", "Addition Mutation", "Point Mutation"],
        title="AUC on Pairwise Comparisons Across All Pair Types",
        ylabel="AUC",
        filename="auc_all_pairs.png",
    )

    # --- Radar chart ---
    _radar_chart(df, "radar_overview.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
