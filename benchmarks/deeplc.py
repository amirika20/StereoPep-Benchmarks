"""
DeepLC benchmark for the PepTag dataset.

Implements the DeepLC convolutional architecture (Bouwmeester et al., 2021,
Nature Methods).  Four parallel encoding paths process the same peptide:

  1. Amino acid composition  — (MAX_LEN × 6) atom counts [C,H,N,O,P,S]
  2. Diamino acid composition — (MAX_LEN//2 × 6) non-overlapping pair counts
  3. One-hot encoding         — (MAX_LEN × 20) canonical AA, tanh, 2 filters
  4. Global features          — 55-dim: length + total atoms + terminal atoms

Paths 1–3 use Conv1d + MaxPool1d blocks.  All four outputs are flattened,
concatenated, and fed to six dense layers.

Three models are trained with kernel sizes 2, 4, 8 and their predictions
averaged (ensemble), matching the original DeepLC approach.

'f' (D-Phe) is handled natively:
  • atom-count paths  — same formula as Phe (C9 H9 N1 O1, stereoisomer)
  • one-hot path      — mapped to 'F' column (same residue, different config)

All layers except the output and the one-hot path use L1 regularisation
with α = 2.5e-7 and a leaky ReLU capped at 20.  The one-hot path uses tanh.

Adapted MAX_LEN=20 (original paper uses 60; peptag sequences are 6–17 AA).

Results are written to benchmarks/results_deeplc.txt.
"""

from __future__ import annotations

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
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ── config ────────────────────────────────────────────────────────────────────
HF_REPO      = "amirka20/peptag"

MAX_LEN      = 20          # pad/truncate to this many residues
KERNEL_SIZES = [2, 4, 8]   # three models → averaged ensemble

CONV_FILTERS = 64          # filters in conv paths
DENSE_HIDDEN = 128         # units in final dense layers
N_DENSE      = 6           # number of layers in the final combined path
L1_ALPHA     = 2.5e-7
NEG_SLOPE    = 0.01
MAX_ACT      = 20.0        # leaky-ReLU cap

LR           = 1e-3
WEIGHT_DECAY = 0.0         # manual L1 used instead
BATCH_SIZE   = 256
MAX_EPOCHS   = 50
PATIENCE     = 8
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_FILE = Path(__file__).parent / "results_deeplc.txt"


# ── atom counts ────────────────────────────────────────────────────────────────
# Residue formulas [C, H, N, O, P, S] (free AA minus H2O for peptide bond)

_AA_ATOMS: dict[str, list[float]] = {
    #          C    H    N    O    P    S
    "A":  [  3.,  5.,  1.,  1.,  0.,  0.],  # Ala
    "R":  [  6., 12.,  4.,  1.,  0.,  0.],  # Arg
    "N":  [  4.,  6.,  2.,  2.,  0.,  0.],  # Asn
    "D":  [  4.,  5.,  1.,  3.,  0.,  0.],  # Asp
    "C":  [  3.,  5.,  1.,  1.,  0.,  1.],  # Cys
    "E":  [  5.,  7.,  1.,  3.,  0.,  0.],  # Glu
    "Q":  [  5.,  8.,  2.,  2.,  0.,  0.],  # Gln
    "G":  [  2.,  3.,  1.,  1.,  0.,  0.],  # Gly
    "H":  [  6.,  7.,  3.,  1.,  0.,  0.],  # His
    "I":  [  6., 11.,  1.,  1.,  0.,  0.],  # Ile
    "L":  [  6., 11.,  1.,  1.,  0.,  0.],  # Leu
    "K":  [  6., 12.,  2.,  1.,  0.,  0.],  # Lys
    "M":  [  5.,  9.,  1.,  1.,  0.,  1.],  # Met
    "F":  [  9.,  9.,  1.,  1.,  0.,  0.],  # Phe (L)
    "f":  [  9.,  9.,  1.,  1.,  0.,  0.],  # Phe (D) — same formula
    "P":  [  5.,  7.,  1.,  1.,  0.,  0.],  # Pro
    "S":  [  3.,  5.,  1.,  2.,  0.,  0.],  # Ser
    "T":  [  4.,  7.,  1.,  2.,  0.,  0.],  # Thr
    "W":  [ 11., 10.,  2.,  1.,  0.,  0.],  # Trp
    "Y":  [  9.,  9.,  1.,  2.,  0.,  0.],  # Tyr
    "V":  [  5.,  9.,  1.,  1.,  0.,  0.],  # Val
    "X":  [  0.,  0.,  0.,  0.,  0.,  0.],  # padding
}

_ZEROS6 = [0.] * 6

# 21 columns: 20 canonical AAs + 'f' (D-Phe) as its own column
_CANON = list("ACDEFGHIKLMNPQRSTVWY")
_AA2OH = {aa: i for i, aa in enumerate(_CANON)}
_AA2OH["f"] = 20    # D-Phe gets a dedicated column — distinct from L-Phe (col 6)
N_OH_CHANNELS = 21


def _av(aa: str) -> list[float]:
    return _AA_ATOMS.get(aa, _ZEROS6)


# ── feature encoding ──────────────────────────────────────────────────────────

def encode_aa(seq: str) -> np.ndarray:
    """(6, MAX_LEN) atom counts per position — channels-first for Conv1d."""
    out = np.zeros((6, MAX_LEN), dtype=np.float32)
    for i, aa in enumerate(seq[:MAX_LEN]):
        out[:, i] = _av(aa)
    return out


def encode_diamino(seq: str) -> np.ndarray:
    """(6, MAX_LEN//2) summed atom counts for non-overlapping pairs."""
    out = np.zeros((6, MAX_LEN // 2), dtype=np.float32)
    for i in range(0, len(seq[:MAX_LEN]), 2):
        pair = np.array(_av(seq[i]))
        if i + 1 < len(seq):
            pair = pair + np.array(_av(seq[i + 1]))
        out[:, i // 2] = pair
    return out


def encode_onehot(seq: str) -> np.ndarray:
    """(21, MAX_LEN) one-hot — 20 canonical AAs + col 20 for D-Phe 'f'."""
    out = np.zeros((N_OH_CHANNELS, MAX_LEN), dtype=np.float32)
    for i, aa in enumerate(seq[:MAX_LEN]):
        idx = _AA2OH.get(aa)
        if idx is not None:
            out[idx, i] = 1.0
    return out


def encode_global(seq: str) -> np.ndarray:
    """
    55-dim global feature vector:
      [0]      length
      [1:7]    total atom counts (C,H,N,O,P,S)
      [7:31]   first 4 positions × 6 atoms   (zero-padded if seq < 4)
      [31:55]  last  4 positions × 6 atoms   (zero-padded if seq < 4)
    """
    feats: list[float] = []
    feats.append(float(len(seq)))

    total = np.zeros(6, dtype=np.float32)
    for aa in seq[:MAX_LEN]:
        total += _av(aa)
    feats.extend(total.tolist())

    # first 4
    for i in range(4):
        feats.extend(_av(seq[i]) if i < len(seq) else _ZEROS6)

    # last 4 (in sequence order, zero-padded for short seqs)
    n = len(seq)
    for offset in range(min(4, n), 0, -1):
        feats.extend(_av(seq[-offset]))
    for _ in range(4 - min(4, n)):
        feats.extend(_ZEROS6)

    assert len(feats) == 55, f"expected 55 global features, got {len(feats)}"
    return np.array(feats, dtype=np.float32)


def encode_split(
    sequences: list[str], desc: str = "Encoding"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode a full split.  Returns (aa, dia, oh, gl) tensors."""
    N = len(sequences)
    aa_buf  = np.zeros((N, 6,  MAX_LEN),       dtype=np.float32)
    dia_buf = np.zeros((N, 6,  MAX_LEN // 2),  dtype=np.float32)
    oh_buf  = np.zeros((N, N_OH_CHANNELS, MAX_LEN), dtype=np.float32)
    gl_buf  = np.zeros((N, 55),                 dtype=np.float32)
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
    """Leaky ReLU with activations capped at max_val (as in DeepLC)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, NEG_SLOPE).clamp(max=MAX_ACT)


class DeepLC(nn.Module):
    """
    Single DeepLC model for one kernel size K.
    Ensemble is handled externally by averaging predictions of three instances.
    """

    def __init__(self, kernel_size: int):
        super().__init__()
        K = kernel_size

        # ── Path 1: amino acid composition (6 channels, MAX_LEN positions)
        self.aa_path = nn.Sequential(
            nn.Conv1d(6,            CONV_FILTERS, K, padding="same"),
            CappedLeakyReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(CONV_FILTERS, CONV_FILTERS, K, padding="same"),
            CappedLeakyReLU(),
            nn.MaxPool1d(2),
        )

        # ── Path 2: diamino acid composition (6 channels, MAX_LEN//2 positions)
        self.dia_path = nn.Sequential(
            nn.Conv1d(6,            CONV_FILTERS, K, padding="same"),
            CappedLeakyReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(CONV_FILTERS, CONV_FILTERS, K, padding="same"),
            CappedLeakyReLU(),
            nn.MaxPool1d(2),
        )

        # ── Path 3: one-hot (21 channels: 20 AAs + 'f', MAX_LEN positions)
        #    Limited to 2 filters and tanh; reduces impact of this path
        self.oh_path = nn.Sequential(
            nn.Conv1d(N_OH_CHANNELS, 2, K, padding="same"),
            nn.Tanh(),
            nn.MaxPool1d(2),
        )

        # ── Path 4: global features (55-dim dense)
        self.gl_path = nn.Sequential(
            nn.Linear(55, DENSE_HIDDEN), CappedLeakyReLU(),
            nn.Linear(DENSE_HIDDEN, DENSE_HIDDEN), CappedLeakyReLU(),
        )

        # Compute concatenated dimension via a dry-run (avoids hard-coding
        # sizes that depend on kernel and pooling interactions)
        with torch.no_grad():
            _aa  = torch.zeros(1, 6,  MAX_LEN)
            _dia = torch.zeros(1, 6,  MAX_LEN // 2)
            _oh  = torch.zeros(1, N_OH_CHANNELS, MAX_LEN)
            _gl  = torch.zeros(1, 55)
            combined_dim = (
                self.aa_path(_aa).flatten(1).shape[1]
                + self.dia_path(_dia).flatten(1).shape[1]
                + self.oh_path(_oh).flatten(1).shape[1]
                + self.gl_path(_gl).shape[1]
            )

        # ── Final path: N_DENSE dense layers
        dims = [combined_dim] + [DENSE_HIDDEN] * (N_DENSE - 1) + [1]
        layers: list[nn.Module] = []
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(d_in, d_out))
            if i < N_DENSE - 1:           # no activation after output layer
                layers.append(CappedLeakyReLU())
        self.combined = nn.Sequential(*layers)

    def forward(
        self,
        aa:  torch.Tensor,   # (B, 6,  MAX_LEN)
        dia: torch.Tensor,   # (B, 6,  MAX_LEN//2)
        oh:  torch.Tensor,   # (B, 21, MAX_LEN)
        gl:  torch.Tensor,   # (B, 55)
    ) -> torch.Tensor:       # (B,)
        x = torch.cat([
            self.aa_path(aa).flatten(1),
            self.dia_path(dia).flatten(1),
            self.oh_path(oh).flatten(1),
            self.gl_path(gl),
        ], dim=1)
        return self.combined(x).squeeze(-1)

    def l1_loss(self) -> torch.Tensor:
        """L1 penalty on all parameters except one-hot path and output layer."""
        total = torch.tensor(0., device=next(self.parameters()).device)
        excluded = set(self.oh_path.parameters()) | set(self.combined[-1].parameters())
        for p in self.parameters():
            if p not in excluded:
                total = total + p.abs().sum()
        return total


# ── training ──────────────────────────────────────────────────────────────────

def train_one(
    model: DeepLC,
    train_loader: DataLoader,
    val_loader: DataLoader,
    kernel_size: int,
) -> list[dict]:
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, min_lr=1e-5
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
        for aa, dia, oh, gl, y in tqdm(train_loader,
                                       desc=f"    train", leave=False):
            aa, dia, oh, gl, y = (
                aa.to(DEVICE), dia.to(DEVICE), oh.to(DEVICE),
                gl.to(DEVICE), y.to(DEVICE),
            )
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
            for aa, dia, oh, gl, y in tqdm(val_loader,
                                           desc=f"    val  ", leave=False):
                aa, dia, oh, gl, y = (
                    aa.to(DEVICE), dia.to(DEVICE), oh.to(DEVICE),
                    gl.to(DEVICE), y.to(DEVICE),
                )
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
def predict(
    models: list[DeepLC],
    aa: torch.Tensor, dia: torch.Tensor, oh: torch.Tensor, gl: torch.Tensor,
) -> np.ndarray:
    """Ensemble prediction: average over all models in the list."""
    loader = DataLoader(
        TensorDataset(aa, dia, oh, gl), batch_size=BATCH_SIZE
    )
    all_preds = []
    for model in models:
        model.eval()
        preds = []
        for aa_b, dia_b, oh_b, gl_b in loader:
            aa_b, dia_b, oh_b, gl_b = (
                aa_b.to(DEVICE), dia_b.to(DEVICE),
                oh_b.to(DEVICE), gl_b.to(DEVICE),
            )
            preds.append(model(aa_b, dia_b, oh_b, gl_b).cpu().numpy())
        all_preds.append(np.concatenate(preds))
    return np.mean(all_preds, axis=0)


# ── metrics ───────────────────────────────────────────────────────────────────

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse    = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae     = float(mean_absolute_error(y_true, y_pred))
    pr, _   = stats.pearsonr(y_true, y_pred)
    sr, _   = stats.spearmanr(y_true, y_pred)
    kr, _   = stats.kendalltau(y_true, y_pred)
    return dict(rmse=rmse, mae=mae, pearson=pr, spearman=sr, kendall=kr)


def stereo_ordering_accuracy(
    models: list[DeepLC],
    stereo_ds,
) -> dict:
    seqs_f  = list(stereo_ds["Sequence_f"])
    seqs_F  = list(stereo_ds["Sequence_F"])
    B_f     = np.array(stereo_ds["B_f"],     dtype=np.float64)
    B_F     = np.array(stereo_ds["B_F"],     dtype=np.float64)
    delta_B = np.array(stereo_ds["delta_B"], dtype=np.float64)

    aa_f, dia_f, oh_f, gl_f = encode_split(seqs_f, desc="Stereo D-Phe")
    aa_F, dia_F, oh_F, gl_F = encode_split(seqs_F, desc="Stereo L-Phe")

    pred_f     = predict(models, aa_f, dia_f, oh_f, gl_f)
    pred_F     = predict(models, aa_F, dia_F, oh_F, gl_F)
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


# ── reporting ─────────────────────────────────────────────────────────────────

def write_results(
    test_metrics:    dict,
    stereo_metrics:  dict,
    histories:       dict[int, list[dict]],
    n_params_each:   int,
) -> None:
    total_epochs = {k: h[-1]["epoch"] for k, h in histories.items()}
    best_vals    = {k: min(h["val_loss"] for h in hs)
                   for k, hs in histories.items()}

    lines = [
        "=" * 70,
        "DeepLC (Bouwmeester et al., 2021) — Retention Time Prediction",
        f"Dataset : {HF_REPO}",
        f"Run at  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "Model configuration",
        "-" * 40,
        f"  Architecture        : 4-path CNN + 6 dense layers (ensemble)",
        f"  Kernel sizes        : {KERNEL_SIZES}  (one model each, averaged)",
        f"  Conv filters        : {CONV_FILTERS}",
        f"  Dense hidden        : {DENSE_HIDDEN}",
        f"  Final dense layers  : {N_DENSE}",
        f"  MAX_LEN             : {MAX_LEN}",
        f"  L1 alpha            : {L1_ALPHA}",
        f"  Leaky ReLU cap      : {MAX_ACT}",
        f"  Params per model    : {n_params_each:,}",
        f"  Optimizer           : Adam  lr={LR}",
        f"  Batch size          : {BATCH_SIZE}",
        f"  Max epochs          : {MAX_EPOCHS}  (early stop patience={PATIENCE})",
        f"  Device              : {DEVICE}",
        "",
        "Training summary (per kernel size)",
        "-" * 40,
    ]
    for k in KERNEL_SIZES:
        lines.append(
            f"  kernel={k}  stopped={total_epochs[k]:3d}  "
            f"best_val={best_vals[k]:.4f}"
        )
    lines += [
        "",
        "Test-split regression metrics  (ensemble)",
        "-" * 40,
        f"  RMSE                : {test_metrics['rmse']:.4f}",
        f"  MAE                 : {test_metrics['mae']:.4f}",
        f"  Pearson  r          : {test_metrics['pearson']:+.4f}",
        f"  Spearman r          : {test_metrics['spearman']:+.4f}",
        f"  Kendall  τ          : {test_metrics['kendall']:+.4f}",
        "",
        "Stereo-pair ordering (D-Phe vs L-Phe)  (ensemble)",
        "-" * 40,
        f"  N pairs evaluated   : {stereo_metrics['n_pairs']}",
        f"  Correct order       : {stereo_metrics['n_correct']}",
        f"  Ordering accuracy   : {stereo_metrics['ordering_acc']:.4f}",
        f"  Δ Pearson  r        : {stereo_metrics['delta_pearson']:+.4f}",
        f"  Δ Spearman r        : {stereo_metrics['delta_spearman']:+.4f}",
        f"  Mean true  Δ B      : {stereo_metrics['mean_true_delta']:+.4f}",
        f"  Mean pred  Δ B      : {stereo_metrics['mean_pred_delta']:+.4f}",
        "",
    ]

    for k, hs in histories.items():
        lines.append(f"Training loss curve  kernel={k}  (every 10 epochs + last)")
        lines.append("-" * 40)
        reported = {h["epoch"] for h in hs if h["epoch"] % 10 == 0}
        reported.add(hs[-1]["epoch"])
        for h in hs:
            if h["epoch"] in reported:
                lines.append(
                    f"  epoch {h['epoch']:3d}  "
                    f"train={h['train_loss']:.4f}  val={h['val_loss']:.4f}"
                )
        lines.append("")

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text("\n".join(lines) + "\n")
    print(f"\nResults written to {RESULTS_FILE}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    print("[data] Loading peptag dataset …")
    ds     = hf_load_dataset(HF_REPO, "peptag")
    stereo = hf_load_dataset(HF_REPO, "stereo_pairs")["stereo_pairs"]

    print("[encode] Building feature tensors …")
    aa_tr, dia_tr, oh_tr, gl_tr = encode_split(list(ds["train"]["Peptide"]), "Train")
    aa_va, dia_va, oh_va, gl_va = encode_split(list(ds["val"]["Peptide"]),   "Val  ")
    aa_te, dia_te, oh_te, gl_te = encode_split(list(ds["test"]["Peptide"]),  "Test ")

    y_train = torch.tensor(ds["train"]["B"], dtype=torch.float32)
    y_val   = torch.tensor(ds["val"]["B"],   dtype=torch.float32)
    y_test  = np.array(ds["test"]["B"],      dtype=np.float32)

    def make_loader(aa, dia, oh, gl, y, shuffle=False):
        return DataLoader(
            TensorDataset(aa, dia, oh, gl, y),
            batch_size=BATCH_SIZE, shuffle=shuffle,
        )

    train_loader = make_loader(aa_tr, dia_tr, oh_tr, gl_tr, y_train, shuffle=True)
    val_loader   = make_loader(aa_va, dia_va, oh_va, gl_va, y_val)

    # Train one model per kernel size
    trained_models: list[DeepLC] = []
    histories: dict[int, list[dict]] = {}

    for K in KERNEL_SIZES:
        print(f"\n[train] kernel_size={K}")
        model = DeepLC(kernel_size=K).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Params: {n_params:,}")
        history = train_one(model, train_loader, val_loader, K)
        trained_models.append(model)
        histories[K] = history

    n_params_each = sum(
        p.numel() for p in trained_models[0].parameters() if p.requires_grad
    )

    print("\n[eval] Ensemble test metrics …")
    y_pred = predict(trained_models, aa_te, dia_te, oh_te, gl_te)
    test_metrics = regression_metrics(y_test, y_pred)
    print(f"  RMSE={test_metrics['rmse']:.4f}  "
          f"Pearson={test_metrics['pearson']:+.4f}  "
          f"Spearman={test_metrics['spearman']:+.4f}")

    print("[eval] Stereo-pair ordering …")
    stereo_metrics = stereo_ordering_accuracy(trained_models, stereo)
    print(f"  Ordering accuracy: {stereo_metrics['ordering_acc']:.4f}  "
          f"({stereo_metrics['n_correct']}/{stereo_metrics['n_pairs']})")

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    write_results(test_metrics, stereo_metrics, histories, n_params_each)


if __name__ == "__main__":
    main()
