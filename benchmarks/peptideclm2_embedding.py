"""
PeptideCLM-2 embedding benchmark for the StereoPep dataset.

PeptideCLM-2 (https://github.com/AaronFeller/PeptideCLM-2, Wilke Lab, UT
Austin) is a suite of 9 SMILES-based transformer encoders — 3 pretraining
objectives (MLM / Hybrid / MTR) x 3 sizes (small/base/large, 32M-337M
params) — trained on 100M+ molecules for peptide chemistry. It supersedes
the original PeptideCLM-23M.

Unlike PepLand, this needs no vendored source or isolated env: it loads
directly off the HuggingFace Hub via standard `transformers` (already
installed in this repo's main env), using `trust_remote_code=True` since
the model class (a rotary-attention transformer, see `ChemPepMTR.py` in
each HF repo) is hosted alongside the weights rather than built into the
`transformers` library. That means loading these models executes Python
code from aaronfeller's HF repos — reviewed and consistent with the
model's own documented usage, but worth knowing.

The backbone is frozen (feature-extraction only, no fine-tuning) — SMILES
input already encodes chirality/non-canonical residues natively, so unlike
esm3_embedding.py there's no missing-token vocabulary patch needed.

Embeddings are precomputed ONCE per model variant (not per seed) by
peptideclm2_precompute_embeddings.py, which caches {SMILES: embedding} to
benchmarks/pretrained_weights/peptideclm2_{model_key}_embeddings.pt. This
script just loads that cache (no transformers model loading, no GPU
forward pass through the backbone at all) and trains/evaluates an MLP
head on top — exactly like morgan_fp_mlp.py / pepland.py. Run the
precompute script first:

    python benchmarks/peptideclm2_precompute_embeddings.py --model hybrid_base

Pass --model <key>       to run a single variant (default: hybrid_base)
Pass --model <k1>,<k2>   to run a subset
Pass --model all          to run every one of the 9 variants sequentially
(each needs its own precomputed cache file first)

Results are written to benchmarks/output/results_{model_key}_seed{N}.json.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset as hf_load_dataset
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

# ── config ────────────────────────────────────────────────────────────────────
HF_REPO = "stereopep-ano/StereoPep"
EMBEDDINGS_DIR = Path(__file__).parent / "pretrained_weights"

# Model registry — 3 pretraining objectives x 3 sizes.
MODELS: dict[str, dict] = {
    f"{obj}_{size}": dict(
        display_name=f"PeptideCLM-2 {obj.upper()} ({size}, {params})",
        hf_name=f"aaronfeller/peptideclm-2-{obj}-{size}",
    )
    for obj, in [("mlm",), ("hybrid",), ("mtr",)]
    for size, params in [("small", "32M"), ("base", "0.1B"), ("large", "0.3B")]
}
ALL_MODELS = list(MODELS.keys())

HIDDEN_DIM   = 512
N_LAYERS     = 3
DROPOUT      = 0.1
LR           = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 256    # MLP head training batch (frozen embeddings — cheap)
ENCODE_BATCH = 32     # backbone forward-pass batch (the expensive part)
MAX_EPOCHS   = 100
PATIENCE     = 10     # overridden at runtime to 0.1 * MAX_EPOCHS
LR_PATIENCE  = 10
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR  = Path(__file__).parent / "output"
WEIGHTS_DIR  = Path(__file__).parent / "weights"


# ── embedding cache ───────────────────────────────────────────────────────────

_CACHES: dict[str, dict] = {}   # model_key -> {"embed_dim": int, "embeddings": {smiles: tensor}}


def load_embedding_cache(model_key: str) -> dict:
    if model_key in _CACHES:
        return _CACHES[model_key]
    path = EMBEDDINGS_DIR / f"peptideclm2_{model_key}_embeddings.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Precomputed embeddings for '{model_key}' not found at {path}.\n"
            f"Run this first:\n\n"
            f"    python benchmarks/peptideclm2_precompute_embeddings.py --model {model_key}\n"
        )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    _CACHES[model_key] = ckpt
    print(f"[cache] Loaded {len(ckpt['embeddings'])} embeddings for {model_key} "
          f"(dim={ckpt['embed_dim']}) from {path}")
    return ckpt


def embed_sequences(model_key: str, smiles_list: list[str]) -> tuple[np.ndarray, list[int]]:
    """Look up precomputed embeddings. Returns (matrix, bad_indices) — bad_indices
    are rows missing from the cache (skipped during precompute, e.g. bad SMILES)."""
    ckpt = load_embedding_cache(model_key)
    cache, embed_dim = ckpt["embeddings"], ckpt["embed_dim"]
    embs, bad = [], []
    for i, smi in enumerate(smiles_list):
        emb = cache.get(smi)
        if emb is None:
            bad.append(i)
            embs.append(np.zeros(embed_dim, dtype=np.float32))
        else:
            embs.append(emb.numpy().astype(np.float32))
    if bad:
        print(f"[WARNING] {len(bad)}/{len(smiles_list)} SMILES missing from the "
              f"PeptideCLM-2 embedding cache for {model_key}.")
    return np.stack(embs), bad


# ── model (MLP head) ─────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_layers: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(dim, hidden), nn.LayerNorm(hidden),
                       nn.GELU(), nn.Dropout(dropout)]
            dim = hidden
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── training helpers ──────────────────────────────────────────────────────────

def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.float32)))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


def train(model: MLP, train_loader: DataLoader, val_loader: DataLoader) -> list[dict]:
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=LR_PATIENCE, factor=0.5, min_lr=1e-5)
    criterion = nn.MSELoss()
    history = []
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(y_b)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                val_loss += criterion(model(X_b), y_b).item() * len(y_b)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0:
            print(f"  epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}")

        if no_improve >= PATIENCE:
            print(f"  early stop at epoch {epoch} (no val improvement for {PATIENCE} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def predict(model: MLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            xb = torch.from_numpy(X[i : i + BATCH_SIZE]).to(DEVICE)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds)


# ── metrics ───────────────────────────────────────────────────────────────────

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse  = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae  = float(mean_absolute_error(y_true, y_pred))
    pr, _ = stats.pearsonr(y_true, y_pred)
    sr, _ = stats.spearmanr(y_true, y_pred)
    kr, _ = stats.kendalltau(y_true, y_pred)
    return dict(mse=mse, rmse=rmse, mae=mae, mean_error=float(np.mean(y_pred - y_true)),
                r2=float(r2_score(y_true, y_pred)), pearson=float(pr), spearman=float(sr), kendall=float(kr))


def pair_delta_metrics(true_delta: np.ndarray, pred_delta: np.ndarray) -> dict:
    rmse   = float(np.sqrt(mean_squared_error(true_delta, pred_delta)))
    mae    = float(mean_absolute_error(true_delta, pred_delta))
    pr, _  = stats.pearsonr(true_delta, pred_delta)
    sr, _  = stats.spearmanr(true_delta, pred_delta)
    kr, _  = stats.kendalltau(true_delta, pred_delta)
    mask   = np.sign(true_delta) != 0
    n_eval = int(mask.sum())
    n_corr = int((np.sign(true_delta[mask]) == np.sign(pred_delta[mask])).sum())
    _nz = true_delta != 0
    if _nz.sum() > 1 and len(np.unique((true_delta[_nz] > 0).astype(int))) > 1:
        delta_auc = float(roc_auc_score((true_delta[_nz] > 0).astype(int), pred_delta[_nz]))
    else:
        delta_auc = float("nan")
    return dict(
        n_pairs=len(true_delta),
        delta_pearson=float(pr), delta_spearman=float(sr), delta_kendall=float(kr),
        delta_rmse=rmse, delta_mae=mae, delta_auc=delta_auc,
        ordering_acc=float(n_corr / n_eval) if n_eval > 0 else float("nan"),
        n_correct=n_corr, n_evaluated=n_eval,
        mean_true_delta=float(true_delta.mean()), mean_pred_delta=float(pred_delta.mean()),
    )


def stereo_ordering_accuracy(mlp: MLP, emb_f: np.ndarray, emb_F: np.ndarray, stereo_ds) -> dict:
    B_f     = np.array(stereo_ds["B_f"], dtype=np.float64)
    B_F     = np.array(stereo_ds["B_F"], dtype=np.float64)
    delta_B = np.array(stereo_ds["delta_B"], dtype=np.float64)

    pred_f = predict(mlp, emb_f)
    pred_F = predict(mlp, emb_F)
    pred_delta = pred_f - pred_F

    true_sign = np.sign(delta_B)
    pred_sign = np.sign(pred_delta)
    correct   = int((true_sign == pred_sign).sum())
    total     = len(delta_B)

    pr, _  = stats.pearsonr(delta_B, pred_delta)
    sr, _  = stats.spearmanr(delta_B, pred_delta)
    kr, _  = stats.kendalltau(delta_B, pred_delta)
    rmse   = float(np.sqrt(mean_squared_error(delta_B, pred_delta)))
    mae    = float(mean_absolute_error(delta_B, pred_delta))
    _nz = delta_B != 0
    if _nz.sum() > 1 and len(np.unique((delta_B[_nz] > 0).astype(int))) > 1:
        delta_auc = float(roc_auc_score((delta_B[_nz] > 0).astype(int), pred_delta[_nz]))
    else:
        delta_auc = float("nan")

    return dict(
        n_pairs=total, n_correct=correct,
        ordering_acc=float(correct / total) if total > 0 else float("nan"),
        delta_pearson=float(pr), delta_spearman=float(sr), delta_kendall=float(kr),
        delta_rmse=rmse, delta_mae=mae, delta_auc=delta_auc,
        mean_true_delta=float(delta_B.mean()), mean_pred_delta=float(pred_delta.mean()),
    )


def eval_pair_metrics(mlp: MLP, emb_a: np.ndarray, emb_b: np.ndarray, ds) -> dict:
    delta_B = np.array(ds["delta_B"], dtype=np.float64)
    pred_a  = predict(mlp, emb_a)
    pred_b  = predict(mlp, emb_b)
    return pair_delta_metrics(delta_B, pred_a - pred_b)


# ── reporting ─────────────────────────────────────────────────────────────────

class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def save_results(
    seed: int, test_metrics: dict, train_metrics: dict, stereo_metrics: dict,
    stereo_trainval_metrics: dict, tag_pair_metrics: dict, substitution_pair_metrics: dict,
    training: dict, config: dict, output_dir: Path, stem: str,
) -> None:
    result = {
        "benchmark": stem, "seed": seed,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "config": config, "training": training,
        "test_metrics": test_metrics, "train_metrics": train_metrics,
        "stereo_metrics": stereo_metrics, "stereo_trainval_metrics": stereo_trainval_metrics,
        "tag_pair_metrics": tag_pair_metrics, "substitution_pair_metrics": substitution_pair_metrics,
    }
    out = output_dir / f"{stem}_seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, cls=_NpEncoder))
    print(f"\nResults saved to {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def run_one_seed(
    seed: int, embed_dim: int,
    X_train, X_val, X_test, y_train, y_val, y_test,
    emb_stereo_f, emb_stereo_F, stereo,
    emb_strv_f, emb_strv_F, stereo_trainval,
    emb_tag_a, emb_tag_b, tag_pairs,
    emb_sub_a, emb_sub_b, sub_pairs,
    weights_path: Path | None = None,
) -> tuple[dict, dict, dict, dict, dict, dict, list[dict]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   shuffle=False)

    mlp = MLP(embed_dim, HIDDEN_DIM, N_LAYERS, DROPOUT).to(DEVICE)

    if weights_path is not None and weights_path.exists():
        print(f"  [weights] Loading from {weights_path} — skipping training")
        ckpt = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        mlp.load_state_dict(ckpt["state_dict"])
        mlp.eval()
        history = ckpt["history"]
    else:
        print("  Training …")
        t0 = time.time()
        history = train(mlp, train_loader, val_loader)
        print(f"  Training done in {time.time() - t0:.1f}s")
        if weights_path is not None:
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": mlp.state_dict(), "history": history}, weights_path)
            print(f"  [weights] Saved to {weights_path}")

    y_pred_test  = predict(mlp, X_test)
    test_metrics = regression_metrics(y_test, y_pred_test)
    print(f"  RMSE={test_metrics['rmse']:.4f}  Pearson={test_metrics['pearson']:+.4f}"
          f"  Spearman={test_metrics['spearman']:+.4f}")

    y_pred_train  = predict(mlp, X_train)
    train_metrics = regression_metrics(y_train, y_pred_train)
    print(f"  [train] RMSE={train_metrics['rmse']:.4f}  Pearson={train_metrics['pearson']:+.4f}")

    if stereo is not None:
        stereo_metrics = stereo_ordering_accuracy(mlp, emb_stereo_f, emb_stereo_F, stereo)
        print(f"  Ordering accuracy: {stereo_metrics['ordering_acc']:.4f}"
              f"  ({stereo_metrics['n_correct']}/{stereo_metrics['n_pairs']})")

        stereo_trainval_metrics = stereo_ordering_accuracy(mlp, emb_strv_f, emb_strv_F, stereo_trainval)
        print(f"  Trainval ordering accuracy: {stereo_trainval_metrics['ordering_acc']:.4f}"
              f"  ({stereo_trainval_metrics['n_correct']}/{stereo_trainval_metrics['n_pairs']})")

        tag_metrics = eval_pair_metrics(mlp, emb_tag_a, emb_tag_b, tag_pairs)
        print(f"  Tag-pair delta Pearson: {tag_metrics['delta_pearson']:+.4f}")
        sub_metrics = eval_pair_metrics(mlp, emb_sub_a, emb_sub_b, sub_pairs)
        print(f"  Substitution-pair delta Pearson: {sub_metrics['delta_pearson']:+.4f}")
    else:
        print("  (pair evals skipped — natural-only dataset has no diastereomer/tag/mutation pairs)")
        stereo_metrics = stereo_trainval_metrics = tag_metrics = sub_metrics = None

    return test_metrics, train_metrics, stereo_metrics, stereo_trainval_metrics, tag_metrics, sub_metrics, history


def main() -> None:
    global MAX_EPOCHS, PATIENCE, LR_PATIENCE

    parser = argparse.ArgumentParser(
        description="PeptideCLM-2 embedding benchmark",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=0, help="Training seed (default: 0)")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help=f"Max training epochs (default: {MAX_EPOCHS})")
    parser.add_argument(
        "--model", default="hybrid_base", metavar="MODEL",
        help=(
            "Model(s) to benchmark. Options:\n"
            + "\n".join(f"  {k:12s} – {v['display_name']}" for k, v in MODELS.items())
            + "\n  all          – run all 9 variants sequentially\n"
            "Comma-separate multiple keys, e.g. --model mlm_base,hybrid_base\n"
            "(default: hybrid_base)"
        ),
    )
    parser.add_argument("--dataset", choices=["stereopep", "natural"], default="stereopep",
                        help="'stereopep' (default): full dataset, includes diastereomer/tag/mutation "
                             "pair evals. 'natural': canonical-amino-acid-only subset; pair evals are "
                             "skipped since they require non-canonical/D-form peptides. All natural "
                             "SMILES are already covered by the existing embedding caches.")
    args = parser.parse_args()

    seed       = args.seed
    MAX_EPOCHS = args.epochs
    PATIENCE   = max(1, int(0.10 * MAX_EPOCHS))
    natural    = args.dataset == "natural"

    if args.model.strip().lower() == "all":
        model_keys = ALL_MODELS
    else:
        model_keys = [k.strip() for k in args.model.split(",")]
        unknown = [k for k in model_keys if k not in MODELS]
        if unknown:
            parser.error(f"Unknown model(s): {unknown}\nChoose from: {', '.join(ALL_MODELS)}, all")

    print(f"Device: {DEVICE}  |  seed={seed}  |  dataset={args.dataset}  "
          f"|  max_epochs={MAX_EPOCHS}  |  patience={PATIENCE}")
    print(f"Models : {model_keys}")

    print(f"\n[data] Loading stereopep dataset (config={'natural' if natural else 'StereoPep'}) …")
    ds = hf_load_dataset(HF_REPO, "natural" if natural else "StereoPep")
    if natural:
        stereo = stereo_trainval = tag_pairs = sub_pairs = None
    else:
        stereo          = hf_load_dataset(HF_REPO, "diastereomer_pairs")["diastereomer_pairs"]
        stereo_trainval = hf_load_dataset(HF_REPO, "diastereomer_pairs")["diastereomer_pairs_trainval"]
        tag_pairs       = hf_load_dataset(HF_REPO, "terminal_tag_pairs")["terminal_tag_pairs"]
        sub_pairs       = hf_load_dataset(HF_REPO, "point_mutant_pairs")["point_mutant_pairs"]

    y_train = np.array(ds["train"]["B"], dtype=np.float32)
    y_val   = np.array(ds["val"]["B"],   dtype=np.float32)
    y_test  = np.array(ds["test"]["B"],  dtype=np.float32)

    for model_key in model_keys:
        cfg = MODELS[model_key]
        print(f"\n{'='*60}")
        print(f"Model : {cfg['display_name']}")
        print(f"{'='*60}")
        t0 = time.time()

        cache = load_embedding_cache(model_key)
        embed_dim = cache["embed_dim"]

        X_train, _ = embed_sequences(model_key, ds["train"]["SMILES"])
        X_val,   _ = embed_sequences(model_key, ds["val"]["SMILES"])
        X_test,  _ = embed_sequences(model_key, ds["test"]["SMILES"])

        if not natural:
            emb_stereo_f, _ = embed_sequences(model_key, stereo["SMILES_f"])
            emb_stereo_F, _ = embed_sequences(model_key, stereo["SMILES_F"])
            emb_strv_f,   _ = embed_sequences(model_key, stereo_trainval["SMILES_f"])
            emb_strv_F,   _ = embed_sequences(model_key, stereo_trainval["SMILES_F"])
            emb_tag_a,    _ = embed_sequences(model_key, tag_pairs["SMILES_untagged"])
            emb_tag_b,    _ = embed_sequences(model_key, tag_pairs["SMILES_tagged"])
            emb_sub_a,    _ = embed_sequences(model_key, sub_pairs["SMILES_1"])
            emb_sub_b,    _ = embed_sequences(model_key, sub_pairs["SMILES_2"])
        else:
            emb_stereo_f = emb_stereo_F = emb_strv_f = emb_strv_F = None
            emb_tag_a = emb_tag_b = emb_sub_a = emb_sub_b = None

        stem = f"results_{model_key}_natural" if natural else f"results_{model_key}"
        weights_path = WEIGHTS_DIR / f"{stem}_seed{seed}.pt"
        print(f"\n── Seed {seed} ──")
        test_metrics, train_metrics, stereo_metrics, stereo_trainval_metrics, tag_metrics, sub_metrics, history = run_one_seed(
            seed, embed_dim, X_train, X_val, X_test, y_train, y_val, y_test,
            emb_stereo_f, emb_stereo_F, stereo,
            emb_strv_f, emb_strv_F, stereo_trainval,
            emb_tag_a, emb_tag_b, tag_pairs,
            emb_sub_a, emb_sub_b, sub_pairs,
            weights_path=weights_path,
        )

        elapsed = time.time() - t0
        print(f"\nTotal time for {model_key}: {elapsed:.1f}s")

        config = {
            "dataset": args.dataset,
            "peptideclm2_model": cfg["hf_name"],
            "embed_dim": embed_dim,
            "hidden_dim": HIDDEN_DIM, "n_layers": N_LAYERS, "dropout": DROPOUT,
            "lr": LR, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "lr_patience": LR_PATIENCE, "device": DEVICE,
        }
        training = {"epochs_run": history[-1]["epoch"], "best_val_loss": min(h["val_loss"] for h in history)}
        save_results(seed, test_metrics, train_metrics, stereo_metrics, stereo_trainval_metrics,
                     tag_metrics, sub_metrics, training, config, RESULTS_DIR, stem)


if __name__ == "__main__":
    main()
