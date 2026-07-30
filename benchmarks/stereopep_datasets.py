"""Shared dataset selection for the StereoPep benchmarks.

Every benchmark script exposes the same `--dataset` flag with three options:

  stereopep     the full benchmark (default). Includes the diastereomer,
                terminal-tag and point-mutation pair splits.
  natural       the canonical-amino-acid-only subset, i.e. every peptide that
                contains no D-Phe ('f'). Published as its own HuggingFace
                config. 26,199 / 2,910 / 1,851 rows (train/val/test).
  noncanonical  the exact complement: only peptides that DO contain D-Phe.
                15,256 / 1,697 / 875 rows. Derived here by filtering the full
                config rather than shipped as a separate HF config.

`natural` and `noncanonical` together partition `stereopep` exactly, which is
the point of the pair: the two ablations answer complementary questions.
`natural` asks whether general retention-time prediction survives when every
stereochemically ambiguous example is removed. `noncanonical` asks the
converse — whether it also holds on the non-canonical peptides alone. This
module asserts the partition is exact on every load, so a change to the
upstream dataset that broke the complement relationship would fail loudly
instead of silently producing an ablation that no longer complements anything.

Both subsets skip the pair evaluations, for different reasons. `natural`
contains no D-Phe peptides at all, so diastereomeric pairs do not exist in it.
`noncanonical` contains the D-form of each pair but never the L-form, so
scoring a pair would require predicting a peptide drawn from the excluded half
of the split — a cross-subset transfer test, not the within-subset regression
check these ablations are for. Use the full `stereopep` dataset for pair
metrics.
"""

from __future__ import annotations

from datasets import load_dataset as hf_load_dataset

DATASET_CHOICES = ("stereopep", "natural", "noncanonical")

DATASET_HELP = (
    "'stereopep' (default): full dataset, includes diastereomer/tag/mutation pair evals. "
    "'natural': canonical-amino-acid-only subset (no D-Phe). "
    "'noncanonical': the complement, D-Phe-containing peptides only. "
    "Both subsets skip the pair evals — see benchmarks/stereopep_datasets.py for why."
)

# D-Phe is written as lowercase 'f' in the Peptide sequence column.
D_PHE = "f"

# Expected split sizes, used to verify the natural/noncanonical partition.
_EXPECTED = {
    "natural":      {"train": 26199, "val": 2910, "test": 1851},
    "noncanonical": {"train": 15256, "val": 1697, "test": 875},
}


def has_pair_splits(dataset: str) -> bool:
    """True only for the full dataset; the two subsets have no usable pair splits."""
    return dataset == "stereopep"


def results_stem(base: str, dataset: str) -> str:
    """Output stem for a run. The full dataset keeps the historical bare name."""
    return base if dataset == "stereopep" else f"{base}_{dataset}"


def load_benchmark_dataset(hf_repo: str, dataset: str):
    """Load the DatasetDict for one of DATASET_CHOICES.

    'noncanonical' is derived by filtering the full config, and is checked
    against the published 'natural' config to confirm the two are exact
    complements.
    """
    if dataset not in DATASET_CHOICES:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {DATASET_CHOICES}")

    if dataset == "natural":
        return hf_load_dataset(hf_repo, "natural")

    full = hf_load_dataset(hf_repo, "StereoPep")
    if dataset == "stereopep":
        return full

    subset = full.filter(lambda row: D_PHE in row["Peptide"])
    _verify_complement(hf_repo, full, subset)
    return subset


def _verify_complement(hf_repo: str, full, subset) -> None:
    """Assert noncanonical is exactly stereopep minus natural."""
    natural = hf_load_dataset(hf_repo, "natural")
    for split, expected in _EXPECTED["noncanonical"].items():
        n_sub, n_nat, n_full = len(subset[split]), len(natural[split]), len(full[split])
        if n_sub != expected:
            raise RuntimeError(
                f"noncanonical/{split} has {n_sub} rows, expected {expected}. The "
                f"upstream dataset changed; re-check the D-Phe filter before "
                f"reporting this ablation."
            )
        if n_sub + n_nat != n_full:
            raise RuntimeError(
                f"{split}: natural ({n_nat}) + noncanonical ({n_sub}) = {n_nat + n_sub} "
                f"!= stereopep ({n_full}). The two subsets are no longer an exact "
                f"partition, so they are not complementary ablations."
            )
        if any(D_PHE in p for p in natural[split]["Peptide"]):
            raise RuntimeError(f"natural/{split} unexpectedly contains D-Phe peptides.")
        if not all(D_PHE in p for p in subset[split]["Peptide"]):
            raise RuntimeError(f"noncanonical/{split} contains a peptide without D-Phe.")
    print("[data] verified: natural + noncanonical partition stereopep exactly "
          "(every noncanonical peptide contains D-Phe, no natural peptide does)")
