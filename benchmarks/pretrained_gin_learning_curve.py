"""
Learning curve for the Pretrained GIN benchmark (reviewer request).

Reuses benchmarks/pretrained_gin.py as-is (imported as a module, not
duplicated) — same architecture, same pretrained ZINC15/bio-assay GIN
encoder, same training loop and eval suite. This script only adds one
thing: training on a random subset of the train split (a fixed fraction,
drawn reproducibly from --seed) while keeping val/test/pair-splits at full
size, so performance-vs-training-size can be plotted.

Chosen for the learning curve because pretrained_gin.py has the best
diastereomer ordering accuracy (0.644 vs next-best 0.632 for gin_scratch)
and by far the lowest seed-to-seed variance (std 0.011 vs 0.121) among
models compared, per benchmarks/output — i.e. the most reliable model on
the paper's central stereochemistry metric.

One run = one (fraction, seed) point. Sweep the grid via
submit_pretrained_gin_learning_curve.sh (SLURM array over the flattened
grid) or by calling this script directly in a loop.

Usage:
  python benchmarks/pretrained_gin_learning_curve.py --frac 0.1 --seed 0

Results are written to
benchmarks/output/learning_curve/results_pretrained_gin_lc_frac{F}_seed{N}.json
Aggregate them into a plotting-ready CSV with:
  python benchmarks/aggregate_learning_curve.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset as hf_load_dataset

sys.path.insert(0, str(Path(__file__).parent))
import pretrained_gin as pg  # noqa: E402

MIN_SUBSET_SIZE = 50   # floor so tiny fractions don't degenerate to a handful of examples
LC_RESULTS_DIR  = pg.RESULTS_DIR / "learning_curve"
LC_WEIGHTS_DIR  = pg.WEIGHTS_DIR / "learning_curve"


def subsample(graphs: list, y: np.ndarray, frac: float, seed: int) -> tuple[list, np.ndarray, int]:
    n_total = len(y)
    n_sub = max(MIN_SUBSET_SIZE, round(frac * n_total))
    n_sub = min(n_sub, n_total)
    rng = np.random.RandomState(seed)
    idx = rng.choice(n_total, size=n_sub, replace=False)
    return [graphs[i] for i in idx], y[idx], n_sub


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrained GIN learning curve (one frac/seed point)")
    parser.add_argument("--frac", type=float, required=True,
                         help="Fraction of the train split to use, in (0, 1]")
    parser.add_argument("--seed", type=int, default=0,
                         help="Seed — controls both the train subsample draw and model init (default: 0)")
    parser.add_argument("--epochs", type=int, default=pg.MAX_EPOCHS,
                         help=f"Max training epochs (default: {pg.MAX_EPOCHS})")
    args = parser.parse_args()

    if not (0.0 < args.frac <= 1.0):
        parser.error("--frac must be in (0, 1]")

    pg.MAX_EPOCHS = args.epochs
    pg.PATIENCE   = max(1, int(0.10 * pg.MAX_EPOCHS))

    print(f"Device: {pg.DEVICE}  |  frac={args.frac}  |  seed={args.seed}  "
          f"|  max_epochs={pg.MAX_EPOCHS}  |  patience={pg.PATIENCE}")

    print("Checking pretrained GIN weights …")
    pg.download_weights()

    print("Loading stereopep dataset …")
    ds              = hf_load_dataset(pg.HF_REPO, "StereoPep")
    sp              = hf_load_dataset(pg.HF_REPO, "diastereomer_pairs")["diastereomer_pairs"]
    stereo_trainval = hf_load_dataset(pg.HF_REPO, "diastereomer_pairs")["diastereomer_pairs_trainval"]
    terminal_tag_pairs = hf_load_dataset(pg.HF_REPO, "terminal_tag_pairs")["terminal_tag_pairs"]
    sub_pairs       = hf_load_dataset(pg.HF_REPO, "point_mutant_pairs")["point_mutant_pairs"]

    graphs_train_full, _ = pg.encode_smiles(ds["train"]["SMILES"], desc="Graphs train (full)")
    graphs_val,        _ = pg.encode_smiles(ds["val"]["SMILES"],   desc="Graphs val       ")
    graphs_test,       _ = pg.encode_smiles(ds["test"]["SMILES"],  desc="Graphs test      ")

    y_train_full = np.array(ds["train"]["B"], dtype=np.float32)
    y_val        = np.array(ds["val"]["B"],   dtype=np.float32)
    y_test       = np.array(ds["test"]["B"],  dtype=np.float32)

    graphs_train, y_train, n_sub = subsample(graphs_train_full, y_train_full, args.frac, args.seed)
    print(f"  train subset: {n_sub}/{len(y_train_full)} ({args.frac:.1%})  "
          f"val={len(y_val)}  test={len(y_test)}")

    weights_path = LC_WEIGHTS_DIR / f"results_pretrained_gin_lc_frac{args.frac:.3f}_seed{args.seed}.pt"
    test_metrics, train_metrics, stereo_metrics, stereo_trainval_metrics, tag_metrics, sub_metrics, history = pg.run_one_seed(
        args.seed, graphs_train, graphs_val, graphs_test, y_train, y_val, y_test, sp,
        stereo_trainval, terminal_tag_pairs, sub_pairs, weights_path=weights_path,
    )

    config = {
        "train_frac": args.frac, "train_size": n_sub, "train_size_full": len(y_train_full),
        "gin_layers": pg.GIN_LAYERS, "gin_emb_dim": pg.GIN_EMB_DIM,
        "head_hidden": pg.HEAD_HIDDEN, "head_layers": pg.HEAD_LAYERS,
        "dropout": pg.DROPOUT, "lr_backbone": pg.LR_BACKBONE, "lr_head": pg.LR_HEAD,
        "weight_decay": pg.WEIGHT_DECAY, "batch_size": pg.BATCH_SIZE,
        "max_epochs": pg.MAX_EPOCHS, "patience": pg.PATIENCE, "lr_patience": pg.LR_PATIENCE,
        "device": pg.DEVICE,
    }
    training = {
        "epochs_run": history[-1]["epoch"],
        "best_val_loss": min(h["val_loss"] for h in history),
    }
    stem = f"results_pretrained_gin_lc_frac{args.frac:.3f}"
    pg.save_results(args.seed, test_metrics, train_metrics, stereo_metrics, stereo_trainval_metrics,
                     tag_metrics, sub_metrics, training, config, LC_RESULTS_DIR, stem)


if __name__ == "__main__":
    main()
