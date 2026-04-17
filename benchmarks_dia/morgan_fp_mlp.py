"""
Morgan Fingerprint MLP benchmark for the DIA dataset.

Same architecture as benchmarks/morgan_fp_mlp.py, adapted for local DIA
retention-time data.  SMILES are generated from standard AA sequences using
RDKit's MolFromFASTA.  No diastereomer/tag/substitution pair metrics.

Results written to benchmarks_dia/output/results_morgan_fp_mlp_dia_seed{N}.json.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ── config ────────────────────────────────────────────────────────────────────
DATA_DIR      = Path(__file__).parent.parent / "data"

FP_RADIUS     = 2
FP_NBITS      = 2048
HIDDEN_DIM    = 512
N_LAYERS      = 3
DROPOUT       = 0.1
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
BATCH_SIZE    = 256
MAX_EPOCHS    = 100
PATIENCE      = 10
LR_PATIENCE   = 10
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR   = Path(__file__).parent / "output"
WEIGHTS_DIR   = Path(__file__).parent / "weights"

_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=FP_RADIUS, fpSize=FP_NBITS, includeChirality=False
)


# ── data loading ──────────────────────────────────────────────────────────────

def load_dia_data():
    train = pd.read_csv(DATA_DIR / "dia_train.txt", sep="\t")
    val   = pd.read_csv(DATA_DIR / "dia_val.txt",   sep="\t")
    test  = pd.read_csv(DATA_DIR / "dia_test.txt",  sep="\t")
    return (
        list(train["sequence"]), np.array(train["RT"], dtype=np.float32),
        list(val["sequence"]),   np.array(val["RT"],   dtype=np.float32),
        list(test["sequence"]),  np.array(test["RT"],  dtype=np.float32),
    )


# ── SMILES generation ─────────────────────────────────────────────────────────

def seq_to_smiles(seq: str) -> str | None:
    """Generate canonical SMILES for a linear peptide from single-letter AA codes."""
    mol = Chem.MolFromFASTA(seq)
    return Chem.MolToSmiles(mol) if mol is not None else None


def seq_to_fp(seq: str) -> np.ndarray | None:
    smiles = seq_to_smiles(seq)
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _FP_GEN.GetFingerprintAsNumPy(mol).astype(np.float32)


def encode_seqs(seqs: list[str], desc: str = "Encoding") -> tuple[np.ndarray, list[int]]:
    fps, bad = [], []
    for i, seq in enumerate(tqdm(seqs, desc=desc, leave=False)):
        fp = seq_to_fp(seq)
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

def make_loader(X, y, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.float32)))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


def train(model, train_loader, val_loader) -> list[dict]:
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=LR_PATIENCE, factor=0.5, min_lr=1e-5
    )
    criterion = nn.MSELoss()
    history   = []
    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0

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
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1

        if epoch % 10 == 0:
            print(f"  epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}")
        if no_improve >= PATIENCE:
            print(f"  early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def predict(model, X) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            preds.append(model(torch.from_numpy(X[i:i+BATCH_SIZE]).to(DEVICE)).cpu().numpy())
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
                r2=float(r2_score(y_true, y_pred)), pearson=float(pr),
                spearman=float(sr), kendall=float(kr))


# ── reporting ─────────────────────────────────────────────────────────────────

class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def save_results(seed, test_metrics, train_metrics, training, config, output_dir, stem):
    result = {
        "benchmark": stem,
        "dataset": "dia",
        "seed": seed,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "config": config,
        "training": training,
        "test_metrics": test_metrics,
        "train_metrics": train_metrics,
    }
    out = output_dir / f"{stem}_seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, cls=_NpEncoder))
    print(f"\nResults saved to {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def run_one_seed(seed, X_train, X_val, X_test, y_train, y_val, y_test,
                 weights_path=None):
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
        print("  Training …")
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

    return test_metrics, train_metrics, history


def main() -> None:
    global MAX_EPOCHS, PATIENCE, LR_PATIENCE
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",   type=int, default=0)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args()
    seed = args.seed

    MAX_EPOCHS  = args.epochs
    PATIENCE    = max(1, int(0.10 * MAX_EPOCHS))

    print(f"Device: {DEVICE}  |  seed={seed}")
    t0 = time.time()

    print("[data] Loading DIA dataset …")
    train_seqs, y_train, val_seqs, y_val, test_seqs, y_test = load_dia_data()
    print(f"  train={len(train_seqs)}  val={len(val_seqs)}  test={len(test_seqs)}")

    print("[fingerprints] Generating Morgan FP from sequences …")
    X_train, bad_tr = encode_seqs(train_seqs, "Train")
    X_val,   bad_va = encode_seqs(val_seqs,   "Val  ")
    X_test,  bad_te = encode_seqs(test_seqs,  "Test ")
    if bad_tr:
        print(f"  WARNING: {len(bad_tr)} train sequences failed SMILES generation")

    n_params = sum(p.numel() for p in MLP(FP_NBITS, HIDDEN_DIM, N_LAYERS, DROPOUT).parameters())
    print(f"Model parameters: {n_params:,}")

    weights_path = WEIGHTS_DIR / f"results_morgan_fp_mlp_dia_seed{seed}.pt"
    print(f"\n── Seed {seed} ──")
    test_metrics, train_metrics, history = run_one_seed(
        seed, X_train, X_val, X_test, y_train, y_val, y_test,
        weights_path=weights_path,
    )

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    config = {
        "fp_radius": FP_RADIUS, "fp_nbits": FP_NBITS,
        "hidden_dim": HIDDEN_DIM, "n_layers": N_LAYERS, "dropout": DROPOUT,
        "lr": LR, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "lr_patience": LR_PATIENCE, "device": DEVICE,
    }
    training = {
        "epochs_run": history[-1]["epoch"],
        "best_val_loss": min(h["val_loss"] for h in history),
    }
    save_results(seed, test_metrics, train_metrics, training, config,
                 RESULTS_DIR, "results_morgan_fp_mlp_dia")


if __name__ == "__main__":
    main()
