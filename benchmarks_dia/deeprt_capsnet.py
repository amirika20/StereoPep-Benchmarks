"""
DeepRT CapsNet benchmark for the DIA dataset.

Same architecture as benchmarks/deeprt_capsnet.py (Tang et al., 2020,
Analytical Chemistry), adapted for local DIA retention-time data.

The CapsNet output (RSS of a 16-D squashed capsule vector) is bounded by
√16 ≈ 4, so RT values are min-max normalised to [0.05, 0.95] during
training and de-normalised before computing metrics.
MAX_LEN is set to 80 (DIA max ~66 AA).
No diastereomer/tag/substitution pair metrics.

Results written to benchmarks_dia/output/results_deeprt_capsnet_dia_seed{N}.json.
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
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ── config ────────────────────────────────────────────────────────────────────
DATA_DIR      = Path(__file__).parent.parent / "data"

MAX_LEN       = 80           # pad/truncate (DIA max ~66)
KERNEL_SIZES  = [8, 10, 12]  # ensemble

EMB_SIZE      = 20
CONV_FILTERS  = 256
NUM_ROUTING   = 1

LR            = 1e-3
BATCH_SIZE    = 64
MAX_EPOCHS    = 50
PATIENCE      = 8
LR_PATIENCE   = 10
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR   = Path(__file__).parent / "output"
WEIGHTS_DIR   = Path(__file__).parent / "weights"

# ── vocabulary ────────────────────────────────────────────────────────────────
_CANON  = list("ACDEFGHIKLMNPQRSTVWY")
_VOCAB  = ["PAD"] + _CANON          # 21 tokens (no D-Phe in DIA)
_TOK2ID = {aa: i for i, aa in enumerate(_VOCAB)}
VOCAB_SIZE = len(_VOCAB)
PAD_ID     = 0


def tokenize_batch(seqs: list[str]) -> torch.Tensor:
    tokens = torch.zeros(len(seqs), MAX_LEN, dtype=torch.long)
    for i, seq in enumerate(seqs):
        ids = [_TOK2ID.get(aa, PAD_ID) for aa in seq[:MAX_LEN]]
        tokens[i, : len(ids)] = torch.tensor(ids)
    return tokens


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


# ── capsule helpers ───────────────────────────────────────────────────────────

def _softmax_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    t = x.transpose(dim, x.dim() - 1)
    s = F.softmax(t.contiguous().view(-1, t.size(-1)), dim=1)
    return s.view(*t.size()).transpose(dim, x.dim() - 1)


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels,
                 kernel_size=None, stride=1, num_iterations=NUM_ROUTING):
        super().__init__()
        self.num_route_nodes = num_route_nodes
        self.num_iterations  = num_iterations
        self.num_capsules    = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(
                torch.randn(num_capsules, num_route_nodes, in_channels, out_channels)
            )
        else:
            self.capsules = nn.ModuleList([
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=kernel_size, stride=stride, padding=0)
                for _ in range(num_capsules)
            ])

    def squash(self, tensor, dim=-1):
        sq_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        return (sq_norm / (1 + sq_norm)) * tensor / torch.sqrt(sq_norm + 1e-8)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]
            logits = Variable(torch.zeros(*priors.size(), device=x.device))
            for i in range(self.num_iterations):
                probs   = _softmax_dim(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))
                if i != self.num_iterations - 1:
                    logits = logits + (priors * outputs).sum(dim=-1, keepdim=True)
        else:
            outputs = [cap(x).view(x.size(0), -1, 1) for cap in self.capsules]
            outputs = self.squash(torch.cat(outputs, dim=-1))
        return outputs


# ── model ─────────────────────────────────────────────────────────────────────

def _digit_nodes(max_len: int, k: int) -> int:
    return 32 * 1 * (max_len - k * 2 + 2 - k + 1)


class CapsNet(nn.Module):
    def __init__(self, conv_kernel: int):
        super().__init__()
        k = conv_kernel
        self.emb   = nn.Embedding(VOCAB_SIZE, EMB_SIZE, padding_idx=PAD_ID)
        self.conv1 = nn.Conv2d(1, CONV_FILTERS, kernel_size=(EMB_SIZE, k), stride=1)
        self.bn1   = nn.BatchNorm2d(CONV_FILTERS)
        self.conv2 = nn.Conv2d(CONV_FILTERS, CONV_FILTERS, kernel_size=(1, k), stride=1)
        self.bn2   = nn.BatchNorm2d(CONV_FILTERS)
        self.primary_capsules = CapsuleLayer(
            num_capsules=8, num_route_nodes=-1,
            in_channels=CONV_FILTERS, out_channels=32,
            kernel_size=(1, k), stride=1,
        )
        digit_nodes = _digit_nodes(MAX_LEN, k)
        self.digit_capsules = CapsuleLayer(
            num_capsules=1, num_route_nodes=digit_nodes,
            in_channels=8, out_channels=16,
        )

    def forward(self, x):
        x = self.emb(x)
        x = x.transpose(1, 2).unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return (x ** 2).sum(dim=-1) ** 0.5


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
        for tokens, y in tqdm(train_loader, desc="    train", leave=False):
            tokens, y = tokens.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(tokens)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(y)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for tokens, y in tqdm(val_loader, desc="    val  ", leave=False):
                tokens, y = tokens.to(DEVICE), y.to(DEVICE)
                val_loss += criterion(model(tokens), y).item() * len(y)
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
def predict(models, tokens) -> np.ndarray:
    loader    = DataLoader(TensorDataset(tokens), batch_size=BATCH_SIZE)
    all_preds = []
    for model in models:
        model.eval()
        preds = []
        for (tok_b,) in loader:
            preds.append(model(tok_b.to(DEVICE)).cpu().numpy())
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
                 tok_tr, y_train_raw,
                 tok_te, y_test_raw,
                 y_min, y_max,
                 weights_path=None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    trained_models: list[CapsNet]    = []
    histories: dict[int, list[dict]] = {}

    if weights_path is not None and weights_path.exists():
        print(f"  [weights] Loading from {weights_path} — skipping training")
        ckpt = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        for K in KERNEL_SIZES:
            model = CapsNet(conv_kernel=K).to(DEVICE)
            model.load_state_dict(ckpt[f"kernel_{K}"])
            model.eval()
            trained_models.append(model)
        histories = ckpt["histories"]
    else:
        for K in KERNEL_SIZES:
            print(f"  [train] kernel_size={K}")
            model   = CapsNet(conv_kernel=K).to(DEVICE)
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

    # Predict and denormalise
    pred_norm_te    = predict(trained_models, tok_te)
    y_pred_te       = pred_norm_te * (y_max - y_min) + y_min
    test_metrics    = regression_metrics(y_test_raw, y_pred_te)
    print(f"  RMSE={test_metrics['rmse']:.4f}  "
          f"Pearson={test_metrics['pearson']:+.4f}  "
          f"Spearman={test_metrics['spearman']:+.4f}")

    pred_norm_tr  = predict(trained_models, tok_tr)
    y_pred_tr     = pred_norm_tr * (y_max - y_min) + y_min
    train_metrics = regression_metrics(y_train_raw, y_pred_tr)
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
    train_seqs, y_train_raw, val_seqs, y_val_raw, test_seqs, y_test_raw = load_dia_data()
    print(f"  train={len(train_seqs)}  val={len(val_seqs)}  test={len(test_seqs)}")

    # Min-max normalise targets to [0.05, 0.95] to keep within capsule RSS range
    y_min = float(y_train_raw.min())
    y_max = float(y_train_raw.max())
    print(f"  RT range (train): [{y_min:.1f}, {y_max:.1f}]")

    def norm(y):
        return 0.05 + 0.9 * (y - y_min) / (y_max - y_min)

    y_train_n = norm(y_train_raw)
    y_val_n   = norm(y_val_raw)

    print("[tokenize] Building token tensors …")
    tok_tr = tokenize_batch(train_seqs)
    tok_va = tokenize_batch(val_seqs)
    tok_te = tokenize_batch(test_seqs)

    y_train_t = torch.tensor(y_train_n, dtype=torch.float32)
    y_val_t   = torch.tensor(y_val_n,   dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(tok_tr, y_train_t),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(tok_va, y_val_t),
                              batch_size=BATCH_SIZE)

    weights_path = WEIGHTS_DIR / f"results_deeprt_capsnet_dia_seed{seed}.pt"
    print(f"\n── Seed {seed} ──")
    test_metrics, train_metrics, histories = run_one_seed(
        seed, train_loader, val_loader,
        tok_tr, y_train_raw, tok_te, y_test_raw,
        y_min, y_max, weights_path=weights_path,
    )

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    config = {
        "kernel_sizes": KERNEL_SIZES, "emb_size": EMB_SIZE, "conv_filters": CONV_FILTERS,
        "num_routing": NUM_ROUTING, "max_len": MAX_LEN, "lr": LR, "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "device": DEVICE,
        "y_min": y_min, "y_max": y_max,
    }
    training = {
        f"kernel_{k}": {"epochs_run": hs[-1]["epoch"], "best_val_loss": min(h["val_loss"] for h in hs)}
        for k, hs in histories.items()
    }
    save_results(seed, test_metrics, train_metrics, training, config,
                 RESULTS_DIR, "results_deeprt_capsnet_dia")


if __name__ == "__main__":
    main()
