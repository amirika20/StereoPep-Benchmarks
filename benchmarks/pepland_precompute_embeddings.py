"""
GPU precompute step for the PepLand benchmark.

PepLand (arXiv:2311.04419) depends on `dgl`, which only ships GPU wheels
pinned to specific torch versions (see benchmarks/pepland_src/NOTICE.md).
Those pins are incompatible with the torch version used by every other
benchmark in this repo, so PepLand embedding extraction is isolated into
this standalone script, meant to be run ONCE in a separate conda env:

    conda create -n pepland_gpu python=3.11 -y
    conda activate pepland_gpu
    pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
    pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
    pip install mlflow rdkit omegaconf datasets pandas tqdm

    python benchmarks/pepland_precompute_embeddings.py

The pretrained PepLand backbone is frozen (feature-extraction only, no
fine-tuning — see PepLandFeatureExtractor(freeze=True) in pepland_src), so
this only needs to run once. It embeds every unique SMILES string that
appears anywhere in the StereoPep dataset (main splits + all pair splits)
and caches {SMILES: embedding} to:

    benchmarks/pretrained_weights/pepland_embeddings.pt

benchmarks/pepland.py (run normally in the `deeprt` env, no dgl/mlflow
required) loads that cache and trains/evaluates an MLP head on top, exactly
like every other benchmark in this repo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from tqdm import tqdm
from datasets import load_dataset as hf_load_dataset
from rdkit import Chem

# ── make the vendored `pepland` package importable ────────────────────────────
_PEPLAND_SRC = Path(__file__).parent / "pepland_src"
sys.path.insert(0, str(_PEPLAND_SRC))
from pepland.model.core import PepLandFeatureExtractor  # noqa: E402

# ── config ────────────────────────────────────────────────────────────────────
HF_REPO      = "stereopep-ano/StereoPep"
MODEL_PATH   = Path(__file__).parent / "pretrained_weights" / "pepland_model"
OUTPUT_PATH  = Path(__file__).parent / "pretrained_weights" / "pepland_embeddings.pt"
POOLING      = "avg"     # 'avg' | 'max' | 'gru'
BATCH_SIZE   = 256
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


def collect_all_smiles() -> list[str]:
    """Gather every unique SMILES string used anywhere in the StereoPep dataset."""
    print("[data] Loading StereoPep dataset splits …")
    ds              = hf_load_dataset(HF_REPO, "StereoPep")
    stereo          = hf_load_dataset(HF_REPO, "diastereomer_pairs")["diastereomer_pairs"]
    stereo_trainval = hf_load_dataset(HF_REPO, "diastereomer_pairs")["diastereomer_pairs_trainval"]
    tag_pairs       = hf_load_dataset(HF_REPO, "terminal_tag_pairs")["terminal_tag_pairs"]
    sub_pairs       = hf_load_dataset(HF_REPO, "point_mutant_pairs")["point_mutant_pairs"]

    smiles: set[str] = set()
    for split in ("train", "val", "test"):
        smiles.update(ds[split]["SMILES"])
    for pair_ds, cols in (
        (stereo,          ("SMILES_f", "SMILES_F")),
        (stereo_trainval, ("SMILES_f", "SMILES_F")),
        (tag_pairs,       ("SMILES_untagged", "SMILES_tagged")),
        (sub_pairs,       ("SMILES_1", "SMILES_2")),
    ):
        for col in cols:
            smiles.update(pair_ds[col])

    print(f"[data] {len(smiles)} unique SMILES collected")
    return sorted(smiles)


def filter_valid(smiles: list[str]) -> tuple[list[str], list[str]]:
    """Split into (valid, invalid) using RDKit parsing — PepLand's tokenizer
    raises on unparsable SMILES, and a single bad entry would otherwise abort
    a whole batch."""
    valid, invalid = [], []
    for smi in tqdm(smiles, desc="Validating SMILES"):
        (valid if Chem.MolFromSmiles(smi) is not None else invalid).append(smi)
    if invalid:
        print(f"[WARNING] {len(invalid)} SMILES failed RDKit parsing and will be skipped:")
        for s in invalid[:10]:
            print(f"    {s}")
    return valid, invalid


@torch.no_grad()
def embed_all(model: PepLandFeatureExtractor, smiles: list[str]) -> dict[str, torch.Tensor]:
    embeddings: dict[str, torch.Tensor] = {}
    for i in tqdm(range(0, len(smiles), BATCH_SIZE), desc="Embedding"):
        batch = smiles[i : i + BATCH_SIZE]
        try:
            pep_embeds = model(batch)   # (B, hidden_dim)
        except Exception as e:
            print(f"[WARNING] Batch starting at index {i} failed ({e}); "
                  f"falling back to per-item embedding")
            for smi in batch:
                try:
                    embeddings[smi] = model([smi])[0].cpu()
                except Exception as e2:
                    print(f"[WARNING] Skipping unembeddable SMILES: {smi} ({e2})")
            continue
        for smi, emb in zip(batch, pep_embeds):
            embeddings[smi] = emb.cpu()
    return embeddings


def main() -> None:
    print(f"Device: {DEVICE}")
    if DEVICE == "cpu":
        print("[WARNING] No GPU detected — this will be slow. "
              "Check `conda activate pepland_gpu` and CUDA visibility.")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"PepLand checkpoint not found at {MODEL_PATH}. "
            f"Expected the vendored MLflow model bundle "
            f"(benchmarks/pretrained_weights/pepland_model/)."
        )

    print(f"[model] Loading PepLand from {MODEL_PATH} (pooling={POOLING}) …")
    model = PepLandFeatureExtractor(str(MODEL_PATH), pooling=POOLING, freeze=True)
    model = model.to(DEVICE)
    model.eval()

    all_smiles = collect_all_smiles()
    valid_smiles, _ = filter_valid(all_smiles)

    embeddings = embed_all(model, valid_smiles)
    print(f"[done] Embedded {len(embeddings)}/{len(all_smiles)} SMILES")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"pooling": POOLING, "embeddings": embeddings}, OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
