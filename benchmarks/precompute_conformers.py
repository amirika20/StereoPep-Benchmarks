"""
Precompute and cache all 3D conformers for the PepTag dataset.

Run this once before training egnn_3d.py to avoid regenerating expensive
RDKit ETKDGv3 conformers on every run.

Usage:
    python benchmarks/precompute_conformers.py [--conf-seed 42] [--cache-dir benchmarks/cache/conformers]

Output:
    One .pt file per (dataset, column, conf_seed) combination, stored in cache_dir.
    egnn_3d.py will automatically pick these up when the same cache_dir is passed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset as hf_load_dataset
from tqdm import tqdm

# Import conformer generation from egnn_3d (same repo)
import sys
sys.path.insert(0, str(Path(__file__).parent))
from egnn_3d import encode_smiles_3d, HF_REPO

DEFAULT_CACHE_DIR = Path(__file__).parent / "cache" / "conformers"


def cache_path(cache_dir: Path, name: str, conf_seed: int) -> Path:
    return cache_dir / f"{name}_seed{conf_seed}.pt"


def save_graphs(graphs, bad, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"graphs": graphs, "bad": bad}, path)
    print(f"  Saved {len(graphs)} graphs ({len(bad)} failed) → {path}")


def precompute_split(
    smiles_list: list[str],
    name: str,
    conf_seed: int,
    cache_dir: Path,
    force: bool = False,
) -> None:
    path = cache_path(cache_dir, name, conf_seed)
    if path.exists() and not force:
        print(f"  Skipping {name} (already cached at {path})")
        return
    graphs, bad = encode_smiles_3d(smiles_list, desc=name, conf_seed=conf_seed)
    save_graphs(graphs, bad, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute 3D conformers for egnn_3d.py")
    parser.add_argument("--conf-seed",  type=int, default=42, help="RDKit conformer random seed")
    parser.add_argument("--cache-dir",  type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--force",      action="store_true", help="Recompute even if cache exists")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    conf_seed = args.conf_seed

    print(f"Cache directory : {cache_dir}")
    print(f"Conformer seed  : {conf_seed}")
    print(f"Force recompute : {args.force}")
    print()

    print("Loading PepTag dataset …")
    ds        = hf_load_dataset(HF_REPO, "peptag")
    sp        = hf_load_dataset(HF_REPO, "stereo_pairs")["stereo_pairs"]
    tag_pairs = hf_load_dataset(HF_REPO, "tag_pairs")["tag_pairs"]
    sub_pairs = hf_load_dataset(HF_REPO, "substitution_pairs")["substitution_pairs"]
    print()

    # ── main splits ──────────────────────────────────────────────────────────
    print("=== Main splits ===")
    for split in ("train", "val", "test"):
        precompute_split(
            list(ds[split]["SMILES"]),
            name=f"peptag_{split}",
            conf_seed=conf_seed,
            cache_dir=cache_dir,
            force=args.force,
        )
    print()

    # ── stereo pairs ─────────────────────────────────────────────────────────
    print("=== Stereo pairs ===")
    precompute_split(
        list(sp["SMILES_f"]),
        name="stereo_pairs_SMILES_f",
        conf_seed=conf_seed,
        cache_dir=cache_dir,
        force=args.force,
    )
    precompute_split(
        list(sp["SMILES_F"]),
        name="stereo_pairs_SMILES_F",
        conf_seed=conf_seed,
        cache_dir=cache_dir,
        force=args.force,
    )
    print()

    # ── tag pairs ─────────────────────────────────────────────────────────────
    print("=== Tag pairs ===")
    precompute_split(
        list(tag_pairs["SMILES_untagged"]),
        name="tag_pairs_SMILES_untagged",
        conf_seed=conf_seed,
        cache_dir=cache_dir,
        force=args.force,
    )
    precompute_split(
        list(tag_pairs["SMILES_tagged"]),
        name="tag_pairs_SMILES_tagged",
        conf_seed=conf_seed,
        cache_dir=cache_dir,
        force=args.force,
    )
    print()

    # ── substitution pairs ───────────────────────────────────────────────────
    print("=== Substitution pairs ===")
    precompute_split(
        list(sub_pairs["SMILES_1"]),
        name="substitution_pairs_SMILES_1",
        conf_seed=conf_seed,
        cache_dir=cache_dir,
        force=args.force,
    )
    precompute_split(
        list(sub_pairs["SMILES_2"]),
        name="substitution_pairs_SMILES_2",
        conf_seed=conf_seed,
        cache_dir=cache_dir,
        force=args.force,
    )
    print()

    print("Done. All conformers cached.")


if __name__ == "__main__":
    main()
