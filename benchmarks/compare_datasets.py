"""
Compare test Pearson r across datasets (peptag vs dia).
Produces a LaTeX table where rows = datasets, columns = models.
"""

import json
import os
import glob
from collections import defaultdict

import numpy as np

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PEPTAG_DIR   = os.path.join(SCRIPT_DIR, "output")
DIA_DIR      = os.path.join(os.path.dirname(SCRIPT_DIR), "benchmarks_dia", "output")
METRICS_DIR  = os.path.join(os.path.dirname(SCRIPT_DIR), "metrics")

os.makedirs(METRICS_DIR, exist_ok=True)

MODEL_NAMES = {
    "results_gin_scratch":         "GIN",
    "results_pretrained_gin":      "Pretrained GIN",
    "results_esm3_sm_embedding":   "ESM3-small",
    "results_esmc_300m_embedding": "ESMC-300M",
    "results_esmc_600m_embedding": "ESMC-600M",
    "results_transformer_scratch": "Transformer",
    "results_deeplc":              "DeepLC",
    "results_deeprt_capsnet":      "DeepRT-CapsNet",
    "results_morgan_mlp":          "Morgan FP MLP",
}

# dia files use a "_dia" suffix in the benchmark key
MODEL_NAMES_DIA = {k + "_dia": v for k, v in MODEL_NAMES.items()}

MODEL_ORDER = [
    "GIN",
    "Pretrained GIN",
    "ESM3-small",
    "ESMC-300M",
    "ESMC-600M",
    "Transformer",
    "DeepLC",
    "DeepRT-CapsNet",
    "Morgan FP MLP",
]


def load_pearson(output_dir: str, name_map: dict) -> dict[str, tuple[float, float]]:
    """Return {display_name: (mean_pearson, std_pearson)} from all seed files."""
    pattern = os.path.join(output_dir, "results_*.json")
    files = sorted(glob.glob(pattern))

    grouped: dict[str, list[float]] = defaultdict(list)
    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        key = data.get("benchmark", os.path.splitext(os.path.basename(fpath))[0])
        pearson = data.get("test_metrics", {}).get("pearson")
        if pearson is not None:
            grouped[key].append(float(pearson))

    result = {}
    for key, vals in grouped.items():
        display = name_map.get(key)
        if display is None:
            continue
        arr = np.array(vals)
        result[display] = (arr.mean(), arr.std(ddof=1) if len(arr) > 1 else 0.0)
    return result


def build_latex_table(
    peptag: dict[str, tuple[float, float]],
    dia:    dict[str, tuple[float, float]],
) -> str:
    # Only include models present in both datasets
    all_models = [m for m in MODEL_ORDER if m in peptag and m in dia]

    # Best per column (dataset)
    def best_col(data: dict) -> str:
        vals = {m: data[m][0] for m in all_models if m in data}
        return max(vals, key=vals.__getitem__) if vals else ""

    best_peptag = best_col(peptag)
    best_dia    = best_col(dia)

    def cell(data: dict, model: str, best: str) -> str:
        if model not in data:
            return "--"
        mean, std = data[model]
        s = r"\makecell{" + f"{mean:.3f}" + r" \\ $\pm$" + f"{std:.3f}" + r"}"
        if model == best:
            s = r"\textbf{" + s + r"}"
        return s

    col_headers = " & ".join(
        [r"\textbf{Dataset}"]
        + [r"\rotatebox{45}{\textbf{" + m + r"}}" for m in all_models]
    )

    rows = []
    for dataset_name, data, best in [("PepTag", peptag, best_peptag),
                                      ("DIA",    dia,    best_dia)]:
        cells = [dataset_name] + [cell(data, m, best) for m in all_models]
        rows.append("  " + " & ".join(cells) + r" \\")

    col_spec = "l" + "c" * len(all_models)
    body = "\n".join(rows)

    tex = (
        r"\begin{table}[htbp]" + "\n"
        r"  \caption{Test Pearson $r$ for retention time prediction across datasets "
        r"(mean $\pm$ std over seeds; bold denotes best model per dataset).}" + "\n"
        r"  \label{tab:dataset_comparison_pearson}" + "\n"
        r"  \centering" + "\n"
        r"  \small" + "\n"
        r"  \begin{tabular}{" + col_spec + r"}" + "\n"
        r"    \toprule" + "\n"
        f"    {col_headers} \\\\\n"
        r"    \midrule" + "\n"
        + body + "\n"
        r"    \bottomrule" + "\n"
        r"  \end{tabular}" + "\n"
        r"\end{table}" + "\n"
    )
    return tex


def main():
    peptag = load_pearson(PEPTAG_DIR, MODEL_NAMES)
    dia    = load_pearson(DIA_DIR,    MODEL_NAMES_DIA)

    print("PepTag Pearson r:")
    for m in MODEL_ORDER:
        if m in peptag:
            print(f"  {m:<22}: {peptag[m][0]:.4f} ± {peptag[m][1]:.4f}")

    print("\nDIA Pearson r:")
    for m in MODEL_ORDER:
        if m in dia:
            print(f"  {m:<22}: {dia[m][0]:.4f} ± {dia[m][1]:.4f}")

    tex = build_latex_table(peptag, dia)

    out_path = os.path.join(METRICS_DIR, "latex_dataset_comparison_pearson.tex")
    with open(out_path, "w") as f:
        f.write(tex)
    print(f"\nLaTeX table saved to: {out_path}")
    print("\n--- Table preview ---\n")
    print(tex)


if __name__ == "__main__":
    main()
