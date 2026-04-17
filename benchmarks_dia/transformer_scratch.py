"""
Transformer benchmark (trained from scratch) for the DIA dataset.

Same architecture as benchmarks/transformer_scratch.py, adapted for local
DIA retention-time data.  All DIA sequences use standard amino acids only
(no D-Phe 'f'), so the vocabulary is reduced to 25 tokens.
MAX_SEQ_LEN is set to 80 to cover the longest DIA sequences (~66 AA).
No diastereomer/tag/substitution pair metrics.

Results written to benchmarks_dia/output/results_transformer_scratch_dia_seed{N}.json.
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
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ── config ────────────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).parent.parent / "data"

D_MODEL     = 128
N_HEADS     = 4
N_LAYERS    = 4
FFN_MULT    = 4
DROPOUT     = 0.1
MAX_SEQ_LEN = 80      # covers DIA max length ~66

LR           = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 256
MAX_EPOCHS   = 50
PATIENCE     = 8
LR_PATIENCE  = 10
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR  = Path(__file__).parent / "output"
WEIGHTS_DIR  = Path(__file__).parent / "weights"


# ── tokenizer ─────────────────────────────────────────────────────────────────
_VOCAB = [
    "PAD", "UNK", "BOS", "EOS", "MASK",                      # 0-4
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",        # 5-14
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",        # 15-24
]
PAD_ID     = 0
BOS_ID     = 2
EOS_ID     = 3
VOCAB_SIZE = len(_VOCAB)  # 25
_TOK2ID    = {t: i for i, t in enumerate(_VOCAB)}


def tokenize_batch(seqs: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.full((len(seqs), MAX_SEQ_LEN), PAD_ID, dtype=torch.long)
    mask   = torch.zeros(len(seqs), MAX_SEQ_LEN, dtype=torch.bool)
    for i, seq in enumerate(seqs):
        ids = [BOS_ID] + [_TOK2ID.get(ch, 1) for ch in seq] + [EOS_ID]
        ids = ids[:MAX_SEQ_LEN]
        n   = len(ids)
        tokens[i, :n] = torch.tensor(ids)
        mask[i,   :n] = True
    return tokens, mask


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


# ── model ─────────────────────────────────────────────────────────────────────

class AttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores  = self.score(x).squeeze(-1)
        scores  = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (weights * x).sum(dim=1)


class PeptideTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int   = VOCAB_SIZE,
        d_model:    int   = D_MODEL,
        n_heads:    int   = N_HEADS,
        n_layers:   int   = N_LAYERS,
        ffn_mult:   int   = FFN_MULT,
        dropout:    float = DROPOUT,
        max_len:    int   = MAX_SEQ_LEN,
    ):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_embed = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * ffn_mult,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.pool = AttentionPooling(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )

    def forward(self, tokens, mask):
        B, L = tokens.shape
        pos  = torch.arange(L, device=tokens.device).unsqueeze(0)
        x    = self.tok_embed(tokens) + self.pos_embed(pos)
        x    = self.encoder(x, src_key_padding_mask=~mask)
        x    = self.pool(x, mask)
        return self.head(x).squeeze(-1)


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


# ── prediction helper ─────────────────────────────────────────────────────────

def predict_from_seqs(model, seqs: list[str]) -> np.ndarray:
    tokens, mask = tokenize_batch(seqs)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), BATCH_SIZE):
            out.append(model(tokens[i:i+BATCH_SIZE].to(DEVICE),
                             mask[i:i+BATCH_SIZE].to(DEVICE)).cpu())
    return torch.cat(out).numpy()


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
                 train_seqs, y_train, test_seqs, y_test,
                 weights_path=None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = PeptideTransformer().to(DEVICE)

    if weights_path is not None and weights_path.exists():
        print(f"  [weights] Loading from {weights_path} — skipping training")
        ckpt    = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        history = ckpt["history"]
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=LR_PATIENCE, factor=0.5, min_lr=1e-5
        )
        criterion = nn.MSELoss()
        best_val_loss = float("inf")
        best_state    = None
        no_improve    = 0
        history: list[dict] = []

        epoch_bar = tqdm(range(1, MAX_EPOCHS + 1), desc="  Training", unit="epoch")
        for epoch in epoch_bar:
            model.train()
            train_loss = 0.0
            for tokens, mask, y in tqdm(train_loader, desc=f"    epoch {epoch:3d} train", leave=False):
                tokens, mask, y = tokens.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(tokens, mask), y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * len(y)
            train_loss /= len(train_loader.dataset)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for tokens, mask, y in tqdm(val_loader, desc=f"    epoch {epoch:3d} val  ", leave=False):
                    val_loss += criterion(
                        model(tokens.to(DEVICE), mask.to(DEVICE)), y.to(DEVICE)
                    ).item() * len(y)
            val_loss /= len(val_loader.dataset)

            scheduler.step(val_loss)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve    = 0
            else:
                no_improve += 1

            epoch_bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}",
                                  best=f"{best_val_loss:.4f}", patience=no_improve)
            if no_improve >= PATIENCE:
                print(f"\n    Early stop at epoch {epoch}")
                break

        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        if weights_path is not None:
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "history": history}, weights_path)
            print(f"  [weights] Saved to {weights_path}")

    y_pred       = predict_from_seqs(model, test_seqs)
    test_metrics = regression_metrics(y_test, y_pred)
    print(f"  RMSE={test_metrics['rmse']:.4f}  "
          f"Pearson={test_metrics['pearson']:+.4f}  "
          f"Spearman={test_metrics['spearman']:+.4f}")

    y_pred_train  = predict_from_seqs(model, train_seqs)
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

    print(f"Device: {DEVICE}  |  seed={seed}  |  max_epochs={MAX_EPOCHS}  |  patience={PATIENCE}")
    t0 = time.time()

    print("[data] Loading DIA dataset …")
    train_seqs, y_train_np, val_seqs, y_val_np, test_seqs, y_test_np = load_dia_data()
    print(f"  train={len(train_seqs)}  val={len(val_seqs)}  test={len(test_seqs)}")

    print("[tokenize] Building token tensors …")
    train_tok, train_mask = tokenize_batch(train_seqs)
    val_tok,   val_mask   = tokenize_batch(val_seqs)

    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    y_val   = torch.tensor(y_val_np,   dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(train_tok, train_mask, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(val_tok, val_mask, y_val),
                              batch_size=BATCH_SIZE)

    n_params = sum(p.numel() for p in PeptideTransformer().parameters() if p.requires_grad)
    print(f"[model] PeptideTransformer | {n_params:,} params | device={DEVICE}")

    weights_path = WEIGHTS_DIR / f"results_transformer_scratch_dia_seed{seed}.pt"
    print(f"\n── Seed {seed} ──")
    test_metrics, train_metrics, history = run_one_seed(
        seed, train_loader, val_loader,
        train_seqs, y_train_np, test_seqs, y_test_np,
        weights_path=weights_path,
    )

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    config = {
        "d_model": D_MODEL, "n_heads": N_HEADS, "n_layers": N_LAYERS,
        "ffn_mult": FFN_MULT, "dropout": DROPOUT, "max_seq_len": MAX_SEQ_LEN,
        "vocab_size": VOCAB_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE, "lr_patience": LR_PATIENCE, "device": DEVICE,
    }
    training = {
        "epochs_run": history[-1]["epoch"],
        "best_val_loss": min(h["val_loss"] for h in history),
    }
    save_results(seed, test_metrics, train_metrics, training, config,
                 RESULTS_DIR, "results_transformer_scratch_dia")


if __name__ == "__main__":
    main()
