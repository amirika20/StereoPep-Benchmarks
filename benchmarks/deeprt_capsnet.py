"""
DeepRT CapsNet benchmark for the PepTag dataset.

Implements the DeepRT Capsule Network (CapsNet) architecture for peptide
retention-time prediction (Tang et al., 2020, Analytical Chemistry).

Architecture
------------
  Amino-acid embedding  (20-D learned vectors)
  → Conv2d  (256 filters, kernel (EMB_SIZE, conv1_kernel)) + BN + ReLU
  → Conv2d  (256 filters, kernel (1, conv1_kernel))        + BN + ReLU
  → PrimaryCapsules  (8 × 32 capsules, kernel (1, conv2_kernel))
  → DigitCapsules    (1 capsule, 16-D, dynamic routing, 1 iteration)
  → RSS of 16-D vector  →  scalar RT prediction

Three models are trained with conv kernel sizes 8, 10, 12 and their
predictions averaged (ensemble), following the original DeepRT approach.

'f' (D-Phe) is a first-class vocabulary token with its own learned
embedding vector, distinct from 'F' (L-Phe).

Results are written to benchmarks/results_deeprt_capsnet_seed{N}.json.
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
import torch.nn.functional as F
from datasets import load_dataset as hf_load_dataset
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ── config ────────────────────────────────────────────────────────────────────
HF_REPO       = "amirka20/peptag"

MAX_LEN       = 50           # pad / truncate to this many residues
KERNEL_SIZES  = [8, 10, 12]  # ensemble — one model per kernel size

EMB_SIZE      = 20           # amino-acid embedding dimension
CONV_FILTERS  = 256
NUM_ROUTING   = 1            # dynamic-routing iterations

LR            = 1e-3
BATCH_SIZE    = 64
MAX_EPOCHS    = 50
PATIENCE      = 8            # overridden at runtime to 0.1 * MAX_EPOCHS
LR_PATIENCE   = 10
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR   = Path(__file__).parent / "output"


# ── vocabulary ────────────────────────────────────────────────────────────────
# PAD=0, then 20 canonical AAs + 'f' (D-Phe) as index 21
_CANON  = list("ACDEFGHIKLMNPQRSTVWY")   # indices 1–20
_VOCAB  = ["PAD"] + _CANON + ["f"]       # 22 tokens total
_TOK2ID = {aa: i for i, aa in enumerate(_VOCAB)}
VOCAB_SIZE = len(_VOCAB)                 # 22
PAD_ID     = 0


def tokenize_batch(seqs: list[str]) -> torch.Tensor:
    """(N, MAX_LEN) LongTensor — character-level, zero-padded."""
    tokens = torch.zeros(len(seqs), MAX_LEN, dtype=torch.long)
    for i, seq in enumerate(seqs):
        ids = [_TOK2ID.get(aa, PAD_ID) for aa in seq[:MAX_LEN]]
        tokens[i, : len(ids)] = torch.tensor(ids)
    return tokens


# ── capsule helpers ───────────────────────────────────────────────────────────

def _softmax_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Softmax along an arbitrary dimension."""
    t = x.transpose(dim, x.dim() - 1)
    s = F.softmax(t.contiguous().view(-1, t.size(-1)), dim=1)
    return s.view(*t.size()).transpose(dim, x.dim() - 1)


class CapsuleLayer(nn.Module):
    """Primary capsules (conv-based) or digit capsules (routing-based)."""

    def __init__(
        self,
        num_capsules: int,
        num_route_nodes: int,   # -1 → use conv capsules
        in_channels: int,
        out_channels: int,
        kernel_size=None,
        stride: int = 1,
        num_iterations: int = NUM_ROUTING,
    ):
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

    def squash(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        sq_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        return (sq_norm / (1 + sq_norm)) * tensor / torch.sqrt(sq_norm + 1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_route_nodes != -1:
            # routing-based digit capsules
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]
            logits = Variable(torch.zeros(*priors.size(), device=x.device))
            for i in range(self.num_iterations):
                probs   = _softmax_dim(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))
                if i != self.num_iterations - 1:
                    logits = logits + (priors * outputs).sum(dim=-1, keepdim=True)
        else:
            # conv-based primary capsules
            outputs = [cap(x).view(x.size(0), -1, 1) for cap in self.capsules]
            outputs = self.squash(torch.cat(outputs, dim=-1))
        return outputs


# ── model ─────────────────────────────────────────────────────────────────────

def _digit_nodes(max_len: int, k: int) -> int:
    """Number of route nodes into the digit capsule layer."""
    return 32 * 1 * (max_len - k * 2 + 2 - k + 1)


class CapsNet(nn.Module):
    """
    Single CapsNet for one kernel size k.
    Ensemble is handled externally by averaging three instances.
    """

    def __init__(self, conv_kernel: int):
        super().__init__()
        k = conv_kernel

        self.emb  = nn.Embedding(VOCAB_SIZE, EMB_SIZE, padding_idx=PAD_ID)

        # Two convolutional layers
        self.conv1 = nn.Conv2d(1, CONV_FILTERS,
                               kernel_size=(EMB_SIZE, k), stride=1)
        self.bn1   = nn.BatchNorm2d(CONV_FILTERS)

        self.conv2 = nn.Conv2d(CONV_FILTERS, CONV_FILTERS,
                               kernel_size=(1, k), stride=1)
        self.bn2   = nn.BatchNorm2d(CONV_FILTERS)

        # Primary capsules
        self.primary_capsules = CapsuleLayer(
            num_capsules=8,
            num_route_nodes=-1,
            in_channels=CONV_FILTERS,
            out_channels=32,
            kernel_size=(1, k),
            stride=1,
        )

        # Digit capsule (1 capsule → regression)
        digit_nodes = _digit_nodes(MAX_LEN, k)
        self.digit_capsules = CapsuleLayer(
            num_capsules=1,
            num_route_nodes=digit_nodes,
            in_channels=8,
            out_channels=16,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, MAX_LEN) int64
        x = self.emb(x)                         # (B, L, 20)
        x = x.transpose(1, 2).unsqueeze(1)      # (B, 1, 20, L)

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)

        x = self.primary_capsules(x)            # (B, nodes, 8)
        x = self.digit_capsules(x).squeeze()    # (B, 16)
        if x.dim() == 1:                        # single-sample batch
            x = x.unsqueeze(0)

        rss = (x ** 2).sum(dim=-1) ** 0.5       # (B,)
        return rss


# ── training ──────────────────────────────────────────────────────────────────

def train_one(
    model: CapsNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    kernel_size: int,
) -> list[dict]:
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

        epoch_bar.set_postfix(
            train=f"{train_loss:.4f}", val=f"{val_loss:.4f}",
            best=f"{best_val:.4f}", patience=no_improve,
        )

        if no_improve >= PATIENCE:
            tqdm.write(f"    early stop at epoch {epoch}")
            break

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return history


# ── inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(models: list[CapsNet], tokens: torch.Tensor) -> np.ndarray:
    """Ensemble prediction: average over all models."""
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
    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae   = float(mean_absolute_error(y_true, y_pred))
    pr, _ = stats.pearsonr(y_true, y_pred)
    sr, _ = stats.spearmanr(y_true, y_pred)
    kr, _ = stats.kendalltau(y_true, y_pred)
    return dict(rmse=rmse, mae=mae, pearson=pr, spearman=sr, kendall=kr)


def pair_delta_metrics(true_delta: np.ndarray, pred_delta: np.ndarray) -> dict:
    """Delta prediction quality metrics for any matched pair type."""
    rmse   = float(np.sqrt(mean_squared_error(true_delta, pred_delta)))
    mae    = float(mean_absolute_error(true_delta, pred_delta))
    pr, _  = stats.pearsonr(true_delta, pred_delta)
    sr, _  = stats.spearmanr(true_delta, pred_delta)
    mask   = np.sign(true_delta) != 0
    n_eval = int(mask.sum())
    n_corr = int((np.sign(true_delta[mask]) == np.sign(pred_delta[mask])).sum())
    return dict(
        n_pairs=len(true_delta),
        delta_pearson=float(pr),
        delta_spearman=float(sr),
        delta_rmse=rmse,
        delta_mae=mae,
        ordering_acc=float(n_corr / n_eval) if n_eval > 0 else float("nan"),
        n_correct=n_corr,
        n_evaluated=n_eval,
        mean_true_delta=float(true_delta.mean()),
        mean_pred_delta=float(pred_delta.mean()),
    )


def stereo_ordering_accuracy(
    models: list[CapsNet],
    stereo_ds,
    y_min: float,
    y_max: float,
) -> dict:
    seqs_f  = list(stereo_ds["Sequence_f"])
    seqs_F  = list(stereo_ds["Sequence_F"])
    B_f     = np.array(stereo_ds["B_f"],     dtype=np.float64)
    B_F     = np.array(stereo_ds["B_F"],     dtype=np.float64)
    delta_B = np.array(stereo_ds["delta_B"], dtype=np.float64)

    tok_f = tokenize_batch(seqs_f)
    tok_F = tokenize_batch(seqs_F)

    pred_f     = predict(models, tok_f) * (y_max - y_min) + y_min
    pred_F     = predict(models, tok_F) * (y_max - y_min) + y_min
    pred_delta = pred_f - pred_F

    true_sign = np.sign(delta_B)
    pred_sign = np.sign(pred_delta)
    correct   = (true_sign == pred_sign).sum()
    n_pairs   = len(delta_B)
    pr, _     = stats.pearsonr(delta_B, pred_delta)
    sr, _     = stats.spearmanr(delta_B, pred_delta)

    return dict(
        n_pairs=n_pairs,
        n_correct=int(correct),
        ordering_acc=float(correct / n_pairs),
        delta_pearson=float(pr),
        delta_spearman=float(sr),
        mean_true_delta=float(delta_B.mean()),
        mean_pred_delta=float(pred_delta.mean()),
    )


def eval_pair_metrics(
    models: list[CapsNet],
    ds,
    seq_col_a: str,
    seq_col_b: str,
    y_min: float,
    y_max: float,
) -> dict:
    """Evaluate predicted delta for any pair split (tag_pairs / substitution_pairs)."""
    seqs_a  = list(ds[seq_col_a])
    seqs_b  = list(ds[seq_col_b])
    delta_B = np.array(ds["delta_B"], dtype=np.float64)

    tok_a = tokenize_batch(seqs_a)
    tok_b = tokenize_batch(seqs_b)

    pred_a     = predict(models, tok_a) * (y_max - y_min) + y_min
    pred_b     = predict(models, tok_b) * (y_max - y_min) + y_min
    pred_delta = pred_a - pred_b

    return pair_delta_metrics(delta_B, pred_delta)


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
        "stereo_metrics": stereo_metrics,
        "tag_pair_metrics": tag_pair_metrics,
        "substitution_pair_metrics": substitution_pair_metrics,
    }
    out = output_dir / f"{stem}_seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, cls=_NpEncoder))
    print(f"\nResults saved to {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def run_one_seed(
    seed: int,
    train_loader,
    val_loader,
    tok_te: torch.Tensor,
    y_test: np.ndarray,
    stereo,
    y_min: float,
    y_max: float,
    tag_pairs,
    sub_pairs,
) -> tuple[dict, dict, dict, dict, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    trained_models: list[CapsNet]    = []
    histories: dict[int, list[dict]] = {}

    for K in KERNEL_SIZES:
        print(f"  [train] kernel_size={K}  digit_nodes={_digit_nodes(MAX_LEN, K)}")
        model   = CapsNet(conv_kernel=K).to(DEVICE)
        history = train_one(model, train_loader, val_loader, K)
        trained_models.append(model)
        histories[K] = history

    y_pred       = predict(trained_models, tok_te) * (y_max - y_min) + y_min
    test_metrics = regression_metrics(y_test, y_pred)
    print(f"  RMSE={test_metrics['rmse']:.4f}  "
          f"Pearson={test_metrics['pearson']:+.4f}  "
          f"Spearman={test_metrics['spearman']:+.4f}")

    stereo_metrics = stereo_ordering_accuracy(trained_models, stereo, y_min, y_max)
    print(f"  Ordering accuracy: {stereo_metrics['ordering_acc']:.4f}  "
          f"({stereo_metrics['n_correct']}/{stereo_metrics['n_pairs']})")

    tag_metrics = eval_pair_metrics(trained_models, tag_pairs, "Sequence_tag", "Sequence_notag", y_min, y_max)
    print(f"  Tag-pair delta Pearson: {tag_metrics['delta_pearson']:+.4f}")
    sub_metrics = eval_pair_metrics(trained_models, sub_pairs, "Sequence_1", "Sequence_2", y_min, y_max)
    print(f"  Substitution-pair delta Pearson: {sub_metrics['delta_pearson']:+.4f}")

    return test_metrics, stereo_metrics, tag_metrics, sub_metrics, histories


def main() -> None:
    global MAX_EPOCHS, PATIENCE, LR_PATIENCE
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0,
                        help="Training seed (default: 0)")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS,
                        help=f"Max training epochs (default: {MAX_EPOCHS})")
    args = parser.parse_args()
    seed = args.seed

    MAX_EPOCHS  = args.epochs
    PATIENCE    = max(1, int(0.10 * MAX_EPOCHS))

    print(f"Device: {DEVICE}  |  seed={seed}  |  max_epochs={MAX_EPOCHS}  |  patience={PATIENCE}")
    t0 = time.time()

    print("[data] Loading peptag dataset …")
    ds        = hf_load_dataset(HF_REPO, "peptag")
    stereo    = hf_load_dataset(HF_REPO, "stereo_pairs")["stereo_pairs"]
    tag_pairs = hf_load_dataset(HF_REPO, "tag_pairs")["tag_pairs"]
    sub_pairs = hf_load_dataset(HF_REPO, "substitution_pairs")["substitution_pairs"]

    print("[tokenize] Building token tensors …")
    tok_tr = tokenize_batch(list(ds["train"]["Peptide"]))
    tok_va = tokenize_batch(list(ds["val"]["Peptide"]))
    tok_te = tokenize_batch(list(ds["test"]["Peptide"]))

    y_train_raw = torch.tensor(ds["train"]["B"], dtype=torch.float32)
    y_val_raw   = torch.tensor(ds["val"]["B"],   dtype=torch.float32)
    y_test      = np.array(ds["test"]["B"],      dtype=np.float32)

    y_min = float(y_train_raw.min())
    y_max = float(y_train_raw.max())
    y_train = (y_train_raw - y_min) / (y_max - y_min)
    y_val   = (y_val_raw   - y_min) / (y_max - y_min)
    print(f"  Target range (train): [{y_min:.2f}, {y_max:.2f}]  → normalised to [0, 1]")

    train_loader = DataLoader(
        TensorDataset(tok_tr, y_train), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(tok_va, y_val), batch_size=BATCH_SIZE
    )

    print(f"\n── Seed {seed} ──")
    test_metrics, stereo_metrics, tag_metrics, sub_metrics, histories = run_one_seed(
        seed, train_loader, val_loader, tok_te, y_test, stereo, y_min, y_max,
        tag_pairs, sub_pairs,
    )

    print(f"\nTotal time: {time.time() - t0:.1f}s")

    config = {
        "kernel_sizes": KERNEL_SIZES, "emb_size": EMB_SIZE, "conv_filters": CONV_FILTERS,
        "num_routing": NUM_ROUTING, "max_len": MAX_LEN, "vocab_size": VOCAB_SIZE,
        "lr": LR, "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "lr_patience": LR_PATIENCE, "device": DEVICE,
        "y_min": y_min, "y_max": y_max,
    }
    training = {
        f"kernel_{k}": {"epochs_run": hs[-1]["epoch"], "best_val_loss": min(h["val_loss"] for h in hs)}
        for k, hs in histories.items()
    }
    save_results(seed, test_metrics, stereo_metrics, tag_metrics, sub_metrics,
                 training, config, RESULTS_DIR, "results_deeprt_capsnet")


if __name__ == "__main__":
    main()
