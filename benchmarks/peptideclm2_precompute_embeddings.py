"""
Precompute step for the PeptideCLM-2 benchmark.

Unlike PepLand, PeptideCLM-2 needs no separate conda env (it's plain
`transformers`, already installed in the `deeprt` env) — but
peptideclm2_embedding.py originally recomputed embeddings live on every
single run, which meant a full ~85k-example forward pass through a
100M+ parameter transformer *every seed*. Since the backbone is frozen,
that's pure waste across a multi-seed sweep.

This script embeds every unique SMILES string used anywhere in the
StereoPep dataset (main splits + all pair splits) ONCE per model variant
and caches {SMILES: embedding} to
benchmarks/pretrained_weights/peptideclm2_{model_key}_embeddings.pt.

peptideclm2_embedding.py then loads that cache instead of the live model,
so a full seed sweep only pays the embedding cost once, not once per seed.

Usage:
  python benchmarks/peptideclm2_precompute_embeddings.py --model hybrid_base
  python benchmarks/peptideclm2_precompute_embeddings.py --model all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset as hf_load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Reuse the model registry / config from the main benchmark script
import sys
sys.path.insert(0, str(Path(__file__).parent))
import peptideclm2_embedding as pc  # noqa: E402

WEIGHTS_DIR = Path(__file__).parent / "pretrained_weights"


def load_model(model_key: str):
    cfg = pc.MODELS[model_key]
    print(f"[PeptideCLM-2] Loading {cfg['display_name']} ({cfg['hf_name']}) on {pc.DEVICE} "
          f"[trust_remote_code=True: executes model code hosted in the HF repo] …")
    tok = AutoTokenizer.from_pretrained(cfg["hf_name"], trust_remote_code=True)
    model = AutoModel.from_pretrained(cfg["hf_name"], trust_remote_code=True, use_safetensors=True)
    model = model.to(pc.DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    embed_dim = model.config.embed_dim
    return model, tok, embed_dim


def collect_all_smiles() -> list[str]:
    print("[data] Loading StereoPep dataset splits …")
    ds              = hf_load_dataset(pc.HF_REPO, "StereoPep")
    stereo          = hf_load_dataset(pc.HF_REPO, "diastereomer_pairs")["diastereomer_pairs"]
    stereo_trainval = hf_load_dataset(pc.HF_REPO, "diastereomer_pairs")["diastereomer_pairs_trainval"]
    tag_pairs       = hf_load_dataset(pc.HF_REPO, "terminal_tag_pairs")["terminal_tag_pairs"]
    sub_pairs       = hf_load_dataset(pc.HF_REPO, "point_mutant_pairs")["point_mutant_pairs"]

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


@torch.no_grad()
def embed_all(model, tok, smiles: list[str]) -> dict[str, torch.Tensor]:
    embeddings: dict[str, torch.Tensor] = {}
    for i in tqdm(range(0, len(smiles), pc.ENCODE_BATCH), desc="Embedding"):
        batch = smiles[i : i + pc.ENCODE_BATCH]
        try:
            inputs = tok(batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(pc.DEVICE) for k, v in inputs.items()}
            outputs = model(**inputs)
            pooled = outputs.mean_pool.cpu()
        except Exception as e:
            print(f"[WARNING] Batch starting at index {i} failed ({e}); "
                  f"falling back to per-item embedding")
            for smi in batch:
                try:
                    inp = tok([smi], return_tensors="pt", padding=True, truncation=True)
                    inp = {k: v.to(pc.DEVICE) for k, v in inp.items()}
                    embeddings[smi] = model(**inp).mean_pool.cpu()[0]
                except Exception as e2:
                    print(f"[WARNING] Skipping unembeddable SMILES: {smi} ({e2})")
            continue
        for smi, emb in zip(batch, pooled):
            embeddings[smi] = emb
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute PeptideCLM-2 embeddings")
    parser.add_argument(
        "--model", default="hybrid_base", metavar="MODEL",
        help="Model(s) to precompute embeddings for. Same keys as peptideclm2_embedding.py "
             "(comma-separated, or 'all'). Default: hybrid_base",
    )
    args = parser.parse_args()

    if args.model.strip().lower() == "all":
        model_keys = pc.ALL_MODELS
    else:
        model_keys = [k.strip() for k in args.model.split(",")]
        unknown = [k for k in model_keys if k not in pc.MODELS]
        if unknown:
            parser.error(f"Unknown model(s): {unknown}\nChoose from: {', '.join(pc.ALL_MODELS)}, all")

    print(f"Device: {pc.DEVICE}")
    all_smiles = collect_all_smiles()

    for model_key in model_keys:
        cfg = pc.MODELS[model_key]
        out_path = WEIGHTS_DIR / f"peptideclm2_{model_key}_embeddings.pt"
        print(f"\n{'='*60}\nModel: {cfg['display_name']}\n{'='*60}")

        model, tok, embed_dim = load_model(model_key)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Backbone parameters: {n_params:,} (frozen)  |  embed_dim={embed_dim}")

        embeddings = embed_all(model, tok, all_smiles)
        print(f"[done] Embedded {len(embeddings)}/{len(all_smiles)} SMILES")

        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"hf_name": cfg["hf_name"], "embed_dim": embed_dim, "embeddings": embeddings}, out_path)
        print(f"Saved to {out_path}")

        del model, tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
