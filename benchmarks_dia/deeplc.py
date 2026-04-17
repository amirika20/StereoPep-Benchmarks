"""
DeepLC benchmark for the DIA dataset.

Same convolutional architecture as benchmarks/deeplc.py (Bouwmeester et al.,
2021, Nature Methods), adapted for the local DIA retention-time data.

Four parallel encoding paths:
  1. Amino acid composition  — (MAX_LEN × 6) atom counts [C,H,N,O,P,S]
  2. Diamino acid composition — (MAX_LEN//2 × 6) non-overlapping pair counts
  3. One-hot encoding         — (MAX_LEN × 20) canonical AA, tanh, 2 filters
  4. Global features          — 55-dim: length + total atoms + terminal atoms

Three models trained with kernel sizes 2, 4, 8 are averaged (ensemble).
MAX_LEN is set to 80 to cover the longest DIA sequences (~66 AA).
No diastereomer/tag/substitution pair metrics — DIA data has none.

Results written to benchmarks_dia/output/results_deeplc_dia_seed{N}.json.
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
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ── config ────────────────────────────────────────────────────────────────────
DATA_DIR     = Path(__file__).parent.parent / "data"
MAX_LEN      = 80          # pad/truncate to this many residues (DIA max ~66)
KERNEL_SIZES = [2, 4, 8]   # three models → averaged ensemble

CONV_FILTERS = 64
DENSE_HIDDEN = 128
N_DENSE      = 6
L1_ALPHA     = 2.5e-7
NEG_SLOPE    = 0.01
MAX_ACT      = 20.0

LR           = 1e-3
BATCH_SIZE   = 256
MAX_EPOCHS   = 50
PATIENCE     = 8
LR_PATIENCE  = 10
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR  = Path(__file__).parent / "output"
WEIGHTS_DIR  = Path(__file__).parent / "weights"


# ── atom counts ───────────────────────────────────────────────────────────────
_AA_ATOMS: dict[str, list[float]] = {
    "A":  [  3.,  5.,  1.,  1.,  0.,  0.],
    "R":  [  6., 12.,  4.,  1.,  0.,  0.],
    "N":  [  4.,  6.,  2.,  2.,  0.,  0.],
    "D":  [  4.,  5.,  1.,  3.,  0.,  0.],
    "C":  [  3.,  5.,  1.,  1.,  0.,  1.],
    "E":  [  5.,  7.,  1.,  3.,  0.,  0.],
    "Q":  [  5.,  8.,  2.,  2.,  0.,  0.],
    "G":  [  2.,  3.,  1.,  1.,  0.,  0.],
    "H":  [  6.,  7.,  3.,  1.,  0.,  0.],
    "I":  [  6., 11.,  1.,  1.,  0.,  0.],
    "L":  [  6., 11.,  1.,  1.,  0.,  0.],
    "K":  [  6., 12.,  2.,  1.,  0.,  0.],
    "M":  [  5.,  9.,  1.,  1.,  0.,  1.],
    "F":  [  9.,  9.,  1.,  1.,  0.,  0.],
    "P":  [  5.,  7.,  1.,  1.,  0.,  0.],
    "S":  [  3.,  5.,  1.,  2.,  0.,  0.],
    "T":  [  4.,  7.,  1.,  2.,  0.,  0.],
    "W":  [ 11., 10.,  2.,  1.,  0.,  0.],
    "Y":  [  9.,  9.,  1.,  2.,  0.,  0.],
    "V":  [  5.,  9.,  1.,  1.,  0.,  0.],
    "X":  [  0.,  0.,  0.,  0.,  0.,  0.],
}
_ZEROS6 = [0.] * 6
_CANON  = list("ACDEFGHIKLMNPQRSTVWY")
_AA2OH  = {aa: i for i, aa in enumerate(_CANON)}
N_OH_CHANNELS = 20


def _av(aa: str) -> list[float]:
    return _AA_ATOMS.get(aa, _ZEROS6)


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


# ── feature encoding ──────────────────────────────────────────────────────────

def encode_aa(seq: str) -> np.ndarray:
    out = np.zeros((6, MAX_LEN), dtype=np.float32)
    for i, aa in enumerate(seq[:MAX_LEN]):
        out[:, i] = _av(aa)
    return out


def encode_diamino(seq: str) -> np.ndarray:
    out = np.zeros((6, MAX_LEN // 2), dtype=np.float32)
    for i in range(0, len(seq[:MAX_LEN]), 2):
        pair = np.array(_av(seq[i]))
        if i + 1 < len(seq):
            pair = pair + np.array(_av(seq[i + 1]))
        out[:, i // 2] = pair
    return out


def encode_onehot(seq: str) -> np.ndarray:
    out = np.zeros((N_OH_CHANNELS, MAX_LEN), dtype=np.float32)
    for i, aa in enumerate(seq[:MAX_LEN]):
        idx = _AA2OH.get(aa)
        if idx is not None:
            out[idx, i] = 1.0
    return out


def encode_global(seq: str) -> np.ndarray:
    feats: list[float] = [float(len(seq))]
    total = np.zeros(6, dtype=np.float32)
    for aa in seq[:MAX_LEN]:
        total += _av(aa)
    feats.extend(total.tolist())
    for i in range(4):
        feats.extend(_av(seq[i]) if i < len(seq) else _ZEROS6)
    n = len(seq)
    for offset in range(min(4, n), 0, -1):
        feats.extend(_av(seq[-offset]))
    for _ in range(4 - min(4, n)):
        feats.extend(_ZEROS6)
    assert len(feats) == 55
    return np.array(feats, dtype=np.float32)


def encode_split(sequences: list[str], desc: str = "Encoding"):
    N = len(sequences)
    aa_buf  = np.zeros((N, 6,  MAX_LEN),      dtype=np.float32)
    dia_buf = np.zeros((N, 6,  MAX_LEN // 2), dtype=np.float32)
    oh_buf  = np.zeros((N, N_OH_CHANNELS, MAX_LEN), dtype=np.float32)
    gl_buf  = np.zeros((N, 55),               dtype=np.float32)
    for i, seq in enumerate(tqdm(sequences, desc=desc, leave=False)):
        aa_buf[i]  = encode_aa(seq)
        dia_buf[i] = encode_diamino(seq)
        oh_buf[i]  = encode_onehot(seq)
        gl_buf[i]  = encode_global(seq)
    return (
        torch.from_numpy(aa_buf),
        torch.from_numpy(dia_buf),
        torch.from_numpy(oh_buf),
        torch.from_numpy(gl_buf),
    )


# ── model ─────────────────────────────────────────────────────────────────────

class CappedLeakyReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, NEG_SLOPE).clamp(max=MAX_ACT)


class DeepLC(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        K = kernel_size
        self.aa_path = nn.Sequential(
            nn.Conv1d(6,            CONV_FILTERS, K, padding="same"), CappedLeakyReLU(), nn.MaxPool1d(2),
            nn.Conv1d(CONV_FILTERS, CONV_FILTERS, K, padding="same"), CappedLeakyReLU(), nn.MaxPool1d(2),
        )
        self.dia_path = nn.Sequential(
            nn.Conv1d(6,            CONV_FILTERS, K, padding="same"), CappedLeakyReLU(), nn.MaxPool1d(2),
            nn.Conv1d(CONV_FILTERS, CONV_FILTERS, K, padding="same"), CappedLeakyReLU(), nn.MaxPool1d(2),
        )
        self.oh_path = nn.Sequential(
            nn.Conv1d(N_OH_CHANNELS, 2, K, padding="same"), nn.Tanh(), nn.MaxPool1d(2),
        )
        self.gl_path = nn.Sequential(
            nn.Linear(55, DENSE_HIDDEN), CappedLeakyReLU(),
            nn.Linear(DENSE_HIDDEN, DENSE_HIDDEN), CappedLeakyReLU(),
        )
        with torch.no_grad():
            combined_dim = (
                self.aa_path(torch.zeros(1, 6, MAX_LEN)).flatten(1).shape[1]
                + self.dia_path(torch.zeros(1, 6, MAX_LEN // 2)).flatten(1).shape[1]
                + self.oh_path(torch.zeros(1, N_OH_CHANNELS, MAX_LEN)).flatten(1).shape[1]
                + self.gl_path(torch.zeros(1, 55)).shape[1]
            )
        dims = [combined_dim] + [DENSE_HIDDEN] * (N_DENSE - 1) + [1]
        layers: list[nn.Module] = []
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(d_in, d_out))
            if i < N_DENSE - 1:
                layers.append(CappedLeakyReLU())
        self.combined = nn.Sequential(*layers)

    def forward(self, aa, dia, oh, gl):
        x = torch.cat([
            self.aa_path(aa).flatten(1),
            self.dia_path(dia).flatten(1),
            self.oh_path(oh).flatten(1),
            self.gl_path(gl),
        ], dim=1)
        return self.combined(x).squeeze(-1)

    def l1_loss(self) -> torch.Tensor:
        total = torch.tensor(0., device=next(self.parameters()).device)
        excluded = set(self.oh_path.parameters()) | set(self.combined[-1].parameters())
        for p in self.parameters():
            if p not in excluded:
                total = total + p.abs().sum()
        return total


# ── training ──────────────────────────────────────────────────────────────────

def train_one(model, train_loader, val_loader, kernel_size):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=LR_PATIENCE, factor=0.5, min_lr=1e-5
    )
    criterion = nn.MSELoss()
    best_val   = float("inf")
    best_state = None
    no_improve = 0
    history: list[dict] = []

    epoch_bar = tqdm(range(1, MAX_EPOCHS + 1),
                     desc=f"  kernel={kernel_size}", unit="epoch", leave=True)
    for epoch in epoch_bar:
        model.train()
        train_loss = 0.0
        for aa, dia, oh, gl, y in tqdm(train_loader, desc="    train", leave=False):
            aa, dia, oh, gl, y = aa.to(DEVICE), dia.to(DEVICE), oh.to(DEVICE), gl.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(aa, dia, oh, gl)
            loss = criterion(pred, y) + L1_ALPHA * model.l1_loss()
            loss.backward()
            optimizer.step()
            train_loss += criterion(pred.detach(), y).item() * len(y)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for aa, dia, oh, gl, y in tqdm(val_loader, desc="    val  ", leave=False):
                aa, dia, oh, gl, y = aa.to(DEVICE), dia.to(DEVICE), oh.to(DEVICE), gl.to(DEVICE), y.to(DEVICE)
                val_loss += criterion(model(aa, dia, oh, gl), y).item() * len(y)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        epoch_bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}",
                              best=f"{best_val:.4f}", patience=no_improve)
        if no_improve >= PATIENCE:
            tqdm.write(f"    early stop at epoch {epoch}")
            break

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return history


# ── inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(models, aa, dia, oh, gl) -> np.ndarray:
    loader = DataLoader(TensorDataset(aa, dia, oh, gl), batch_size=BATCH_SIZE)
    all_preds = []
    for model in models:
        model.eval()
        preds = []
        for aa_b, dia_b, oh_b, gl_b in loader:
            preds.append(model(aa_b.to(DEVICE), dia_b.to(DEVICE),
                               oh_b.to(DEVICE), gl_b.to(DEVICE)).cpu().numpy())
        all_preds.append(np.concatenate(preds))
    return np.mean(all_preds, axis=0)


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

def run_one_seed(seed, train_loader, val_loader,
                 aa_tr, dia_tr, oh_tr, gl_tr, y_train,
                 aa_te, dia_te, oh_te, gl_te, y_test,
                 weights_path=None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    trained_models: list[DeepLC] = []
    histories: dict[int, list[dict]] = {}

    if weights_path is not None and weights_path.exists():
        print(f"  [weights] Loading from {weights_path} — skipping training")
        ckpt = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        for K in KERNEL_SIZES:
            model = DeepLC(kernel_size=K).to(DEVICE)
            model.load_state_dict(ckpt[f"kernel_{K}"])
            model.eval()
            trained_models.append(model)
        histories = ckpt["histories"]
    else:
        for K in KERNEL_SIZES:
            print(f"  [train] kernel_size={K}")
            model   = DeepLC(kernel_size=K).to(DEVICE)
            history = train_one(model, train_loader, val_loader, K)
            trained_models.append(model)
            histories[K] = history
        if weights_path is not None:
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {f"kernel_{K}": m.state_dict() for K, m in zip(KERNEL_SIZES, trained_models)}
                | {"histories": histories},
                weights_path,
            )
            print(f"  [weights] Saved to {weights_path}")

    y_pred       = predict(trained_models, aa_te, dia_te, oh_te, gl_te)
    test_metrics = regression_metrics(y_test, y_pred)
    print(f"  RMSE={test_metrics['rmse']:.4f}  "
          f"Pearson={test_metrics['pearson']:+.4f}  "
          f"Spearman={test_metrics['spearman']:+.4f}")

    y_pred_train  = predict(trained_models, aa_tr, dia_tr, oh_tr, gl_tr)
    train_metrics = regression_metrics(y_train, y_pred_train)
    print(f"  [train] RMSE={train_metrics['rmse']:.4f}  Pearson={train_metrics['pearson']:+.4f}")

    return test_metrics, train_metrics, histories


def main() -> None:
    global MAX_EPOCHS, PATIENCE
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",   type=int, default=0)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args()
    seed = args.seed

    MAX_EPOCHS = args.epochs
    PATIENCE   = max(1, int(0.1 * MAX_EPOCHS))

    print(f"Device: {DEVICE}  |  seed={seed}  |  max_epochs={MAX_EPOCHS}  |  patience={PATIENCE}")
    t0 = time.time()

    print("[data] Loading DIA dataset …")
    train_seqs, y_train_np, val_seqs, y_val_np, test_seqs, y_test_np = load_dia_data()
    print(f"  train={len(train_seqs)}  val={len(val_seqs)}  test={len(test_seqs)}")

    print("[encode] Building feature tensors …")
    aa_tr, dia_tr, oh_tr, gl_tr = encode_split(train_seqs, "Train")
    aa_va, dia_va, oh_va, gl_va = encode_split(val_seqs,   "Val  ")
    aa_te, dia_te, oh_te, gl_te = encode_split(test_seqs,  "Test ")

    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    y_val   = torch.tensor(y_val_np,   dtype=torch.float32)

    def make_loader(aa, dia, oh, gl, y, shuffle=False):
        return DataLoader(TensorDataset(aa, dia, oh, gl, y),
                          batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = make_loader(aa_tr, dia_tr, oh_tr, gl_tr, y_train, shuffle=True)
    val_loader   = make_loader(aa_va, dia_va, oh_va, gl_va, y_val)

    weights_path = WEIGHTS_DIR / f"results_deeplc_dia_seed{seed}.pt"
    print(f"\n── Seed {seed} ──")
    test_metrics, train_metrics, histories = run_one_seed(
        seed, train_loader, val_loader,
        aa_tr, dia_tr, oh_tr, gl_tr, y_train_np,
        aa_te, dia_te, oh_te, gl_te, y_test_np,
        weights_path=weights_path,
    )

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    config = {
        "kernel_sizes": KERNEL_SIZES, "conv_filters": CONV_FILTERS,
        "dense_hidden": DENSE_HIDDEN, "n_dense": N_DENSE, "max_len": MAX_LEN,
        "l1_alpha": L1_ALPHA, "lr": LR, "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "device": DEVICE,
    }
    training = {
        f"kernel_{k}": {"epochs_run": hs[-1]["epoch"], "best_val_loss": min(h["val_loss"] for h in hs)}
        for k, hs in histories.items()
    }
    save_results(seed, test_metrics, train_metrics, training, config,
                 RESULTS_DIR, "results_deeplc_dia")


if __name__ == "__main__":
    main()
