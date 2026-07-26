"""
Morgan Fingerprint MLP benchmark for the StereoPep dataset.

Pipeline:
  1. Compute Morgan fingerprints (radius=2, 2048 bits) from full peptide SMILES.
  2. Train a small MLP to predict B (retention time, normalised 0-100).
  3. Evaluate on test split with regression metrics.
  4. Evaluate stereochemistry ordering accuracy on the diastereomer_pairs split:
     for every (f, F) pair, check whether the model predicts the correct
     elution order (D-Phe vs L-Phe).

Usage:
  python morgan_fp_mlp.py [--seed N]   # run a single seed (default: 0)

Results are written to benchmarks/results_morgan_mlp_seed{N}.json.
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
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import rdFingerprintGenerator
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

# ── config ────────────────────────────────────────────────────────────────────
HF_REPO       = "stereopep-ano/StereoPep"
FP_RADIUS     = 2
FP_NBITS      = 2048
FP_CHIRALITY  = True   # must be True to distinguish D-Phe ('f') from L-Phe ('F')
HIDDEN_DIM    = 512
N_LAYERS      = 3          # number of hidden layers
DROPOUT       = 0.1
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
BATCH_SIZE    = 256
MAX_EPOCHS    = 100
PATIENCE      = 10         # overridden at runtime to 0.1 * MAX_EPOCHS
LR_PATIENCE   = 10
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR   = Path(__file__).parent / "output"
WEIGHTS_DIR   = Path(__file__).parent / "weights"

# ── fingerprints ──────────────────────────────────────────────────────────────

_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=FP_RADIUS, fpSize=FP_NBITS, includeChirality=True
)


def smiles_to_fp(smiles: str) -> np.ndarray | None:
    mol = MolFromSmiles(smiles)
    if mol is None:
        return None
    return _FP_GEN.GetFingerprintAsNumPy(mol).astype(np.float32)


def encode_split(smiles_list: list[str]) -> tuple[np.ndarray, list[int]]:
    """Return (matrix, bad_indices) — bad_indices are rows that failed to parse."""
    fps, bad = [], []
    for i, smi in enumerate(smiles_list):
        fp = smiles_to_fp(smi)
        if fp is None:
            bad.append(i)
            fps.append(np.zeros(FP_NBITS, dtype=np.float32))
        else:
            fps.append(fp)
    return np.stack(fps), bad


# ── model ─────────────────────────────────────────────────────────────────────

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
    ds = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y.astype(np.float32)),
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


def train(
    model: MLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> list[dict]:
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=LR_PATIENCE, factor=0.5, min_lr=1e-5
    )
    criterion = nn.MSELoss()
    history = []
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            opt.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(y_b)
        train_loss /= len(train_loader.dataset)

        # --- val ---
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
    """Delta prediction quality metrics for any matched pair type."""
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
        delta_pearson=float(pr),
        delta_spearman=float(sr),
        delta_kendall=float(kr),
        delta_rmse=rmse,
        delta_mae=mae,
        delta_auc=delta_auc,
        ordering_acc=float(n_corr / n_eval) if n_eval > 0 else float("nan"),
        n_correct=n_corr,
        n_evaluated=n_eval,
        mean_true_delta=float(true_delta.mean()),
        mean_pred_delta=float(pred_delta.mean()),
    )


# ── stereo ordering ───────────────────────────────────────────────────────────

def stereo_ordering_accuracy(
    model: MLP,
    stereo_ds,
) -> dict:
    smiles_f = stereo_ds["SMILES_f"]
    smiles_F = stereo_ds["SMILES_F"]
    B_f      = np.array(stereo_ds["B_f"], dtype=np.float64)
    B_F      = np.array(stereo_ds["B_F"], dtype=np.float64)
    delta_B  = np.array(stereo_ds["delta_B"], dtype=np.float64)   # B_f - B_F

    X_f, bad_f = encode_split(smiles_f)
    X_F, bad_F = encode_split(smiles_F)

    pred_f = predict(model, X_f)
    pred_F = predict(model, X_F)
    pred_delta = pred_f - pred_F

    # Mask out any failed parses
    bad = set(bad_f) | set(bad_F)
    mask = np.ones(len(delta_B), dtype=bool)
    for i in bad:
        mask[i] = False

    true_sign = np.sign(delta_B[mask])
    pred_sign = np.sign(pred_delta[mask])

    correct = (true_sign == pred_sign).sum()
    total   = mask.sum()
    accuracy = correct / total if total > 0 else float("nan")

    # Correlation of predicted delta vs true delta
    pr, _  = stats.pearsonr(delta_B[mask], pred_delta[mask])
    sr, _  = stats.spearmanr(delta_B[mask], pred_delta[mask])
    kr, _  = stats.kendalltau(delta_B[mask], pred_delta[mask])
    rmse   = float(np.sqrt(mean_squared_error(delta_B[mask], pred_delta[mask])))
    mae    = float(mean_absolute_error(delta_B[mask], pred_delta[mask]))
    _nz    = delta_B[mask] != 0
    if _nz.sum() > 1 and len(np.unique((delta_B[mask][_nz] > 0).astype(int))) > 1:
        delta_auc = float(roc_auc_score((delta_B[mask][_nz] > 0).astype(int), pred_delta[mask][_nz]))
    else:
        delta_auc = float("nan")

    return {
        "n_pairs":         total,
        "n_correct":       int(correct),
        "ordering_acc":    accuracy,
        "delta_pearson":   pr,
        "delta_spearman":  sr,
        "delta_kendall":   float(kr),
        "delta_rmse":      rmse,
        "delta_mae":       mae,
        "delta_auc":       delta_auc,
        "mean_true_delta": float(delta_B[mask].mean()),
        "mean_pred_delta": float(pred_delta[mask].mean()),
    }


def eval_pair_metrics(
    model: MLP,
    ds,
    smiles_col_a: str,
    smiles_col_b: str,
) -> dict:
    """Evaluate predicted delta for any pair split (terminal_tag_pairs / point_mutant_pairs)."""
    delta_B    = np.array(ds["delta_B"], dtype=np.float64)
    X_a, bad_a = encode_split(list(ds[smiles_col_a]))
    X_b, bad_b = encode_split(list(ds[smiles_col_b]))

    pred_a     = predict(model, X_a)
    pred_b     = predict(model, X_b)
    pred_delta = pred_a - pred_b

    bad = set(bad_a) | set(bad_b)
    mask = np.ones(len(delta_B), dtype=bool)
    for i in bad:
        mask[i] = False

    return pair_delta_metrics(delta_B[mask], pred_delta[mask])


# ── reporting ─────────────────────────────────────────────────────────────────

class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def save_results(
    seed: int,
    test_metrics: dict,
    train_metrics: dict,
    stereo_metrics: dict,
    stereo_trainval_metrics: dict,
    tag_pair_metrics: dict,
    substitution_pair_metrics: dict,
    training: dict,
    config: dict,
    output_dir: Path,
    stem: str,
) -> None:
    result = {
        "benchmark": stem,
        "seed": seed,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "config": config,
        "training": training,
        "test_metrics": test_metrics,
        "train_metrics": train_metrics,
        "stereo_metrics": stereo_metrics,
        "stereo_trainval_metrics": stereo_trainval_metrics,
        "tag_pair_metrics": tag_pair_metrics,
        "substitution_pair_metrics": substitution_pair_metrics,
    }
    out = output_dir / f"{stem}_seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, cls=_NpEncoder))
    print(f"\nResults saved to {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def run_one_seed(seed: int, X_train, X_val, X_test, y_train, y_val, y_test, sp,
                 stereo_trainval, terminal_tag_pairs, sub_pairs, weights_path: Path | None = None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   shuffle=False)

    model = MLP(FP_NBITS, HIDDEN_DIM, N_LAYERS, DROPOUT).to(DEVICE)

    if weights_path is not None and weights_path.exists():
        print(f"  [weights] Loading from {weights_path} — skipping training")
        ckpt    = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        history = ckpt["history"]
    else:
        print(f"  Training …")
        t0 = time.time()
        history = train(model, train_loader, val_loader)
        print(f"  Training done in {time.time() - t0:.1f}s")
        if weights_path is not None:
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "history": history}, weights_path)
            print(f"  [weights] Saved to {weights_path}")

    y_pred_test = predict(model, X_test)
    test_metrics = regression_metrics(y_test, y_pred_test)
    print(f"  RMSE={test_metrics['rmse']:.4f}  Pearson={test_metrics['pearson']:+.4f}"
          f"  Spearman={test_metrics['spearman']:+.4f}")

    y_pred_train  = predict(model, X_train)
    train_metrics = regression_metrics(y_train, y_pred_train)
    print(f"  [train] RMSE={train_metrics['rmse']:.4f}  Pearson={train_metrics['pearson']:+.4f}")

    if sp is not None:
        stereo_metrics = stereo_ordering_accuracy(model, sp)
        print(f"  Ordering accuracy: {stereo_metrics['ordering_acc']:.4f}"
              f"  ({stereo_metrics['n_correct']}/{stereo_metrics['n_pairs']})")

        stereo_trainval_metrics = stereo_ordering_accuracy(model, stereo_trainval)
        print(f"  Trainval ordering accuracy: {stereo_trainval_metrics['ordering_acc']:.4f}"
              f"  ({stereo_trainval_metrics['n_correct']}/{stereo_trainval_metrics['n_pairs']})")

        tag_metrics = eval_pair_metrics(model, terminal_tag_pairs, "SMILES_untagged", "SMILES_tagged")
        print(f"  Tag-pair delta Pearson: {tag_metrics['delta_pearson']:+.4f}")
        sub_metrics = eval_pair_metrics(model, sub_pairs, "SMILES_1", "SMILES_2")
        print(f"  Substitution-pair delta Pearson: {sub_metrics['delta_pearson']:+.4f}")
    else:
        print("  (pair evals skipped — natural-only dataset has no diastereomer/tag/mutation pairs)")
        stereo_metrics = stereo_trainval_metrics = tag_metrics = sub_metrics = None

    return test_metrics, train_metrics, stereo_metrics, stereo_trainval_metrics, tag_metrics, sub_metrics, history


def main() -> None:
    global MAX_EPOCHS, PATIENCE, LR_PATIENCE

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Training seed (default: 0)"
    )
    parser.add_argument(
        "--epochs", type=int, default=MAX_EPOCHS,
        help=f"Max training epochs (default: {MAX_EPOCHS})"
    )
    parser.add_argument("--dataset", choices=["stereopep", "natural"], default="stereopep",
                        help="'stereopep' (default): full dataset, includes diastereomer/tag/mutation "
                             "pair evals. 'natural': canonical-amino-acid-only subset; pair evals are "
                             "skipped since they require non-canonical/D-form peptides.")
    args = parser.parse_args()
    seed = args.seed
    natural = args.dataset == "natural"

    MAX_EPOCHS  = args.epochs
    PATIENCE    = max(1, int(0.10 * MAX_EPOCHS))

    print(f"Device: {DEVICE}")
    print(f"Running seed: {seed}  |  dataset: {args.dataset}")

    # Load dataset once
    print(f"Loading stereopep dataset (config={'natural' if natural else 'StereoPep'}) …")
    ds = hf_load_dataset(HF_REPO, "natural" if natural else "StereoPep")
    if natural:
        sp = stereo_trainval = terminal_tag_pairs = sub_pairs = None
    else:
        sp              = hf_load_dataset(HF_REPO, "diastereomer_pairs")["diastereomer_pairs"]
        stereo_trainval = hf_load_dataset(HF_REPO, "diastereomer_pairs")["diastereomer_pairs_trainval"]
        terminal_tag_pairs       = hf_load_dataset(HF_REPO, "terminal_tag_pairs")["terminal_tag_pairs"]
        sub_pairs       = hf_load_dataset(HF_REPO, "point_mutant_pairs")["point_mutant_pairs"]

    # Encode fingerprints once (deterministic)
    for split_name in ("train", "val", "test"):
        print(f"Encoding {split_name} fingerprints …")
    X_train, _ = encode_split(ds["train"]["SMILES"])
    X_val,   _ = encode_split(ds["val"]["SMILES"])
    X_test,  _ = encode_split(ds["test"]["SMILES"])

    y_train = np.array(ds["train"]["B"], dtype=np.float32)
    y_val   = np.array(ds["val"]["B"],   dtype=np.float32)
    y_test  = np.array(ds["test"]["B"],  dtype=np.float32)

    print(f"  train={len(y_train)}  val={len(y_val)}  test={len(y_test)}")
    n_params = sum(p.numel() for p in MLP(FP_NBITS, HIDDEN_DIM, N_LAYERS, DROPOUT).parameters())
    print(f"Model parameters: {n_params:,}")

    stem = "results_morgan_mlp_natural" if natural else "results_morgan_mlp"
    weights_path = WEIGHTS_DIR / f"{stem}_seed{seed}.pt"
    print(f"\n── Seed {seed} ──")
    test_metrics, train_metrics, stereo_metrics, stereo_trainval_metrics, tag_metrics, sub_metrics, history = run_one_seed(
        seed, X_train, X_val, X_test, y_train, y_val, y_test, sp, stereo_trainval, terminal_tag_pairs, sub_pairs,
        weights_path=weights_path,
    )

    config = {
        "dataset": args.dataset,
        "fp_radius": FP_RADIUS, "fp_nbits": FP_NBITS, "fp_chirality": FP_CHIRALITY,
        "hidden_dim": HIDDEN_DIM, "n_layers": N_LAYERS, "dropout": DROPOUT,
        "lr": LR, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "lr_patience": LR_PATIENCE, "device": DEVICE,
    }
    training = {
        "epochs_run": history[-1]["epoch"],
        "best_val_loss": min(h["val_loss"] for h in history),
    }
    save_results(seed, test_metrics, train_metrics, stereo_metrics, stereo_trainval_metrics,
                 tag_metrics, sub_metrics, training, config, RESULTS_DIR, stem)


if __name__ == "__main__":
    main()
