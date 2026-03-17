"""
Transformer benchmark (trained from scratch) for the PepTag dataset.

Builds a small transformer encoder with attention pooling to predict B
(retention time, normalised 0-100) directly from peptide sequences.

Vocabulary (26 tokens) matches data/PEPLM_WORDS.csv used by the existing
DeepRT codebase — 'f' (D-Phe) is a first-class token at index 5, so the
model can learn its representation end-to-end.

Architecture
------------
  Token + position embedding
  → N × TransformerEncoderLayer  (Pre-LN, batch_first)
  → AttentionPooling              (learnable weighted mean over positions)
  → 2-layer MLP head              → scalar B prediction

Self-contained: no imports from other project modules.

Results are written to benchmarks/results_transformer_scratch_seed{N}.json.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ── config ────────────────────────────────────────────────────────────────────
HF_REPO     = "amirka20/peptag"

D_MODEL     = 128
N_HEADS     = 4
N_LAYERS    = 4
FFN_MULT    = 4       # feedforward hidden dim = D_MODEL × FFN_MULT
DROPOUT     = 0.1
MAX_SEQ_LEN = 64      # peptides are short; pad/truncate to this length

LR           = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 256
MAX_EPOCHS   = 50
PATIENCE     = 8          # overridden at runtime to 0.1 * MAX_EPOCHS
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR  = Path(__file__).parent


# ── tokenizer ─────────────────────────────────────────────────────────────────
# Vocab mirrors data/PEPLM_WORDS.csv + special tokens (same order as DeepRT).
_VOCAB = [
    "PAD", "UNK", "BOS", "EOS", "MASK",                      # 0-4
    "f", "A", "C", "D", "E", "F", "G", "H", "I", "K",        # 5-14
    "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",   # 15-25
]
PAD_ID     = 0
BOS_ID     = 2
EOS_ID     = 3
VOCAB_SIZE = len(_VOCAB)   # 26
_TOK2ID    = {t: i for i, t in enumerate(_VOCAB)}


def tokenize_batch(seqs: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Character-level tokenisation with BOS/EOS and padding to MAX_SEQ_LEN.

    Returns
    -------
    tokens : (N, MAX_SEQ_LEN)  LongTensor
    mask   : (N, MAX_SEQ_LEN)  BoolTensor, True = valid position
    """
    tokens = torch.full((len(seqs), MAX_SEQ_LEN), PAD_ID, dtype=torch.long)
    mask   = torch.zeros(len(seqs), MAX_SEQ_LEN, dtype=torch.bool)
    for i, seq in enumerate(seqs):
        ids = [BOS_ID] + [_TOK2ID.get(ch, 1) for ch in seq] + [EOS_ID]
        ids = ids[:MAX_SEQ_LEN]
        n   = len(ids)
        tokens[i, :n] = torch.tensor(ids)
        mask[i,   :n] = True
    return tokens, mask


# ── model ─────────────────────────────────────────────────────────────────────

class AttentionPooling(nn.Module):
    """Learnable weighted mean over sequence positions."""

    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)  mask: (B, L) True=valid
        scores  = self.score(x).squeeze(-1)                      # (B, L)
        scores  = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)    # (B, L, 1)
        return (weights * x).sum(dim=1)                          # (B, D)


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
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ffn_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,    # Pre-LN: more stable for small datasets
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.pool = AttentionPooling(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # tokens: (B, L)  mask: (B, L) True=valid
        B, L = tokens.shape
        pos  = torch.arange(L, device=tokens.device).unsqueeze(0)  # (1, L)
        x    = self.tok_embed(tokens) + self.pos_embed(pos)         # (B, L, D)
        # PyTorch convention: src_key_padding_mask True = IGNORE (inverted)
        x    = self.encoder(x, src_key_padding_mask=~mask)
        x    = self.pool(x, mask)                                   # (B, D)
        return self.head(x).squeeze(-1)                             # (B,)


# ── metrics ───────────────────────────────────────────────────────────────────

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse    = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae     = float(mean_absolute_error(y_true, y_pred))
    pr, _   = stats.pearsonr(y_true, y_pred)
    sr, _   = stats.spearmanr(y_true, y_pred)
    kr, _   = stats.kendalltau(y_true, y_pred)
    return dict(rmse=rmse, mae=mae, pearson=pr, spearman=sr, kendall=kr)


# ── prediction helper ─────────────────────────────────────────────────────────

def predict_from_seqs(model: PeptideTransformer, seqs: list[str]) -> np.ndarray:
    tokens, mask = tokenize_batch(seqs)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), BATCH_SIZE):
            out.append(model(tokens[i:i+BATCH_SIZE].to(DEVICE),
                             mask[i:i+BATCH_SIZE].to(DEVICE)).cpu())
    return torch.cat(out).numpy()


# ── stereo ordering ───────────────────────────────────────────────────────────

def stereo_ordering_accuracy(model: PeptideTransformer, stereo_ds) -> dict:
    seqs_f  = stereo_ds["Sequence_f"]
    seqs_F  = stereo_ds["Sequence_F"]
    B_f     = np.array(stereo_ds["B_f"],    dtype=np.float64)
    B_F     = np.array(stereo_ds["B_F"],    dtype=np.float64)
    delta_B = np.array(stereo_ds["delta_B"], dtype=np.float64)  # B_f − B_F

    pred_f     = predict_from_seqs(model, seqs_f)
    pred_F     = predict_from_seqs(model, seqs_F)
    pred_delta = pred_f - pred_F

    true_sign = np.sign(delta_B)
    pred_sign = np.sign(pred_delta)
    correct   = (true_sign == pred_sign).sum()
    n_pairs   = len(delta_B)
    accuracy  = correct / n_pairs

    pr, _ = stats.pearsonr(delta_B, pred_delta)
    sr, _ = stats.spearmanr(delta_B, pred_delta)

    return dict(
        n_pairs=n_pairs,
        n_correct=int(correct),
        ordering_acc=float(accuracy),
        delta_pearson=float(pr),
        delta_spearman=float(sr),
        mean_true_delta=float(delta_B.mean()),
        mean_pred_delta=float(pred_delta.mean()),
    )


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
    stereo_metrics: dict,
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
        "stereo_metrics": stereo_metrics,
    }
    out = output_dir / f"{stem}_seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, cls=_NpEncoder))
    print(f"\nResults saved to {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def run_one_seed(
    seed: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_seqs: list[str],
    y_test: np.ndarray,
    stereo,
) -> tuple[dict, dict, list[dict]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model     = PeptideTransformer().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, min_lr=1e-5
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

        epoch_bar.set_postfix(
            train=f"{train_loss:.4f}", val=f"{val_loss:.4f}",
            best=f"{best_val_loss:.4f}", patience=no_improve,
        )

        if no_improve >= PATIENCE:
            print(f"\n    Early stop at epoch {epoch}")
            break

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    y_pred       = predict_from_seqs(model, test_seqs)
    test_metrics = regression_metrics(y_test, y_pred)
    print(f"  RMSE={test_metrics['rmse']:.4f}  "
          f"Pearson={test_metrics['pearson']:+.4f}  "
          f"Spearman={test_metrics['spearman']:+.4f}")

    stereo_metrics = stereo_ordering_accuracy(model, stereo)
    print(f"  Ordering accuracy: {stereo_metrics['ordering_acc']:.4f}  "
          f"({stereo_metrics['n_correct']}/{stereo_metrics['n_pairs']})")

    return test_metrics, stereo_metrics, history


def main() -> None:
    global MAX_EPOCHS, PATIENCE
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0,
                        help="Training seed (default: 0)")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS,
                        help=f"Max training epochs (default: {MAX_EPOCHS})")
    args = parser.parse_args()
    seed = args.seed

    MAX_EPOCHS = args.epochs
    PATIENCE   = max(1, int(0.1 * MAX_EPOCHS))

    print(f"Device: {DEVICE}  |  seed={seed}  |  max_epochs={MAX_EPOCHS}  |  patience={PATIENCE}")
    t0 = time.time()

    print("[data] Loading peptag dataset …")
    ds     = hf_load_dataset(HF_REPO, "peptag")
    stereo = hf_load_dataset(HF_REPO, "stereo_pairs")["stereo_pairs"]

    print("[tokenize] Building token tensors …")
    train_tok, train_mask = tokenize_batch(ds["train"]["Peptide"])
    val_tok,   val_mask   = tokenize_batch(ds["val"]["Peptide"])

    y_train = torch.tensor(ds["train"]["B"], dtype=torch.float32)
    y_val   = torch.tensor(ds["val"]["B"],   dtype=torch.float32)
    y_test  = np.array(ds["test"]["B"],      dtype=np.float32)

    train_loader = DataLoader(
        TensorDataset(train_tok, train_mask, y_train),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_tok, val_mask, y_val),
        batch_size=BATCH_SIZE,
    )

    n_params = sum(p.numel() for p in PeptideTransformer().parameters() if p.requires_grad)
    print(f"[model] PeptideTransformer | {n_params:,} params | device={DEVICE}")

    print(f"\n── Seed {seed} ──")
    test_metrics, stereo_metrics, history = run_one_seed(
        seed, train_loader, val_loader, ds["test"]["Peptide"], y_test, stereo
    )

    print(f"\nTotal time: {time.time() - t0:.1f}s")

    config = {
        "d_model": D_MODEL, "n_heads": N_HEADS, "n_layers": N_LAYERS,
        "ffn_mult": FFN_MULT, "dropout": DROPOUT, "max_seq_len": MAX_SEQ_LEN,
        "vocab_size": VOCAB_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "device": DEVICE,
    }
    training = {
        "epochs_run": history[-1]["epoch"],
        "best_val_loss": min(h["val_loss"] for h in history),
    }
    save_results(seed, test_metrics, stereo_metrics, training, config, RESULTS_DIR, "results_transformer_scratch")


if __name__ == "__main__":
    main()
