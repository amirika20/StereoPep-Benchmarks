"""
ESM3 embedding benchmark for the PepTag dataset.

Uses ESM3-small (esm3_sm_open_v0, ~300 M params) as a mostly-frozen encoder.

D-Phenylalanine ('f') is not in the ESM3 vocabulary.  We add it as a new token
and represent it with a *learnable* embedding vector initialised as a copy of
L-Phe ('F').  Only this single vector and the MLP prediction head are trained;
all other ESM3 parameters stay frozen.

The learnable 'f' embedding is implemented via a differentiable mask-blend in
LearnableTokenEmbed, so gradients flow through it correctly even though the rest
of the backbone is frozen.

Because 'f_embedding' changes each step, ESM3 must be called on-the-fly during
training — pre-computed embeddings would become stale.

Pipeline
--------
1. Load ESM3-small; replace sequence_embed with LearnableTokenEmbed for 'f'.
2. Train loop: tokenise + ESM3 forward + mean-pool per batch (on-the-fly).
   Gradients flow through f_embedding and the MLP head only.
3. Evaluate on the test split with regression metrics (pre-computed after train).
4. Evaluate stereochemistry ordering accuracy on the stereo_pairs split.

Results are written to benchmarks/results_esm3_embedding.txt.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset as hf_load_dataset
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

# ── config ────────────────────────────────────────────────────────────────────
HF_REPO      = "amirka20/peptag"
ESM_MODEL    = "esm3_sm_open_v0"

HIDDEN_DIM   = 512
N_LAYERS     = 3
DROPOUT      = 0.1
LR           = 1e-3
WEIGHT_DECAY = 1e-4
TRAIN_BATCH  = 32    # smaller: each batch runs a full ESM3 forward pass
ENCODE_BATCH = 32    # batch size for no-grad encoding (val/test/stereo)
MAX_EPOCHS   = 20
PATIENCE     = 5
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_FILE = Path(__file__).parent / "results_esm3_embedding.txt"


# ── learnable single-token embedding ─────────────────────────────────────────

class LearnableTokenEmbed(nn.Module):
    """
    Wraps a frozen nn.Embedding and replaces one token id with a learnable vector.

    The blend is differentiable w.r.t. f_embedding:

        output = frozen(x) * (1 - mask) + f_embedding * mask

    where mask == 1 at positions where token id == token_id.
    Gradients pass through the f_embedding term; the frozen term has no grad.
    """

    def __init__(self, frozen_embed: nn.Embedding, token_id: int, init: torch.Tensor):
        super().__init__()
        self.frozen_embed = frozen_embed   # requires_grad already False
        self.token_id     = token_id
        self.f_embedding  = nn.Parameter(init.clone())   # the only learnable part

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base  = self.frozen_embed(x)                            # (B, L, dim) — no grad
        mask  = (x == self.token_id).float().unsqueeze(-1)      # (B, L, 1)
        f_exp = self.f_embedding.expand_as(base)                # (B, L, dim)
        return base * (1.0 - mask) + f_exp * mask               # differentiable


# ── ESM3 setup ────────────────────────────────────────────────────────────────

def load_esm3():
    """Load ESM3-small, freeze backbone, replace sequence_embed with learnable wrapper."""
    from esm.pretrained import ESM3_sm_open_v0

    print(f"[ESM3] Loading {ESM_MODEL} on {DEVICE} …")
    esm = ESM3_sm_open_v0(DEVICE)
    esm.eval()
    for p in esm.parameters():
        p.requires_grad_(False)

    tok   = esm.tokenizers.sequence
    vocab = tok.get_vocab()

    if "f" not in vocab:
        tok.add_tokens(["f"])
        f_id  = tok.convert_tokens_to_ids("f")
        F_id  = vocab["F"]
        f_init = esm.encoder.sequence_embed.weight[F_id].detach()
        esm.encoder.sequence_embed = LearnableTokenEmbed(
            esm.encoder.sequence_embed, f_id, f_init
        ).to(DEVICE)
        print(f"[ESM3] Added learnable 'f' (D-Phe) token_id={f_id}, init from 'F' id={F_id}")
    else:
        print("[ESM3] Token 'f' already in vocabulary")

    return esm, tok


# ── embedding helpers ─────────────────────────────────────────────────────────

def embed_batch(esm, tok, sequences: list[str]) -> torch.Tensor:
    """
    Tokenise + ESM3 forward + mean-pool for a list of sequences.
    Returns (B, 1536).  Supports gradient flow (no torch.no_grad here).
    """
    tokens    = tok(sequences, return_tensors="pt", padding=True)
    input_ids = tokens["input_ids"].to(DEVICE)
    attn_mask = tokens["attention_mask"].to(DEVICE)

    out = esm(sequence_tokens=input_ids)
    emb = out.embeddings                        # (B, L, 1536)

    # Exclude CLS (pos 0) and EOS (last real token) from mean pool
    residue_mask = attn_mask.clone().float()
    residue_mask[:, 0] = 0.0
    seq_lens = attn_mask.sum(dim=1)
    for b, l in enumerate(seq_lens):
        residue_mask[b, l - 1] = 0.0

    residue_mask = residue_mask.unsqueeze(-1)   # (B, L, 1)
    pooled = (emb * residue_mask).sum(dim=1) / residue_mask.sum(dim=1).clamp(min=1)
    return pooled                               # (B, 1536)


@torch.no_grad()
def embed_sequences(esm, tok, sequences: list[str], desc: str = "Embedding") -> np.ndarray:
    """Pre-compute embeddings for a list of sequences (no gradient). Returns (N, 1536)."""
    all_embs = []
    for i in tqdm(range(0, len(sequences), ENCODE_BATCH), desc=desc):
        batch = sequences[i : i + ENCODE_BATCH]
        all_embs.append(embed_batch(esm, tok, batch).cpu().float().numpy())
    return np.concatenate(all_embs, axis=0)


# ── data loading ──────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    def __init__(self, sequences: list[str], y: np.ndarray):
        self.sequences = sequences
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.y[idx]


def seq_collate(batch):
    seqs, ys = zip(*batch)
    return list(seqs), torch.stack(ys)


# ── MLP head ──────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_layers: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout)]
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(model: MLP, embeddings: np.ndarray, y: np.ndarray) -> dict:
    model.eval()
    X = torch.tensor(embeddings, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X), batch_size=256)
    preds = []
    with torch.no_grad():
        for (batch,) in loader:
            preds.append(model(batch.to(DEVICE)).cpu())
    preds   = torch.cat(preds).numpy()
    targets = y
    rmse    = float(np.sqrt(mean_squared_error(targets, preds)))
    mae     = float(mean_absolute_error(targets, preds))
    r,   _  = stats.pearsonr(targets, preds)
    rho, _  = stats.spearmanr(targets, preds)
    tau, _  = stats.kendalltau(targets, preds)
    return dict(rmse=rmse, mae=mae, pearson=r, spearman=rho, kendall=tau)


def stereo_ordering_accuracy(model: MLP, esm, tok, stereo_ds) -> dict:
    seqs_f = stereo_ds["Sequence_f"]
    seqs_F = stereo_ds["Sequence_F"]
    B_f    = np.array(stereo_ds["B_f"])
    B_F    = np.array(stereo_ds["B_F"])

    emb_f = embed_sequences(esm, tok, seqs_f, desc="Stereo D-Phe")
    emb_F = embed_sequences(esm, tok, seqs_F, desc="Stereo L-Phe")

    model.eval()
    with torch.no_grad():
        pred_f = model(torch.tensor(emb_f).to(DEVICE)).cpu().numpy()
        pred_F = model(torch.tensor(emb_F).to(DEVICE)).cpu().numpy()

    true_order = np.sign(B_f - B_F)
    pred_order = np.sign(pred_f - pred_F)

    tied_true = (true_order == 0).sum()
    tied_pred = (pred_order == 0).sum()
    mask      = true_order != 0
    correct   = (true_order[mask] == pred_order[mask]).sum()
    n_eval    = mask.sum()
    acc       = correct / n_eval if n_eval > 0 else float("nan")

    return dict(
        n_pairs=len(seqs_f),
        n_tied_true=int(tied_true),
        n_tied_pred=int(tied_pred),
        n_evaluated=int(n_eval),
        n_correct=int(correct),
        ordering_accuracy=float(acc),
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    print("[data] Loading peptag dataset …")
    ds     = hf_load_dataset(HF_REPO, "peptag")
    stereo = hf_load_dataset(HF_REPO, "stereo_pairs")["stereo_pairs"]

    esm, tok = load_esm3()

    # Collect all learnable parameters: f_embedding + MLP head (added below)
    esm_learnable = [p for p in esm.parameters() if p.requires_grad]
    print(f"[ESM3] Learnable ESM3 params: {sum(p.numel() for p in esm_learnable)}  "
          f"(just the 'f' embedding vector)")

    y_train = np.array(ds["train"]["B"], dtype=np.float32)
    y_val   = np.array(ds["val"]["B"],   dtype=np.float32)

    train_loader = DataLoader(
        SequenceDataset(ds["train"]["Peptide"], y_train),
        batch_size=TRAIN_BATCH, shuffle=True, collate_fn=seq_collate,
    )
    val_loader = DataLoader(
        SequenceDataset(ds["val"]["Peptide"], y_val),
        batch_size=TRAIN_BATCH, shuffle=False, collate_fn=seq_collate,
    )

    emb_dim = 1536
    print(f"\n[train] MLP head + f_embedding on ESM3  (dim={emb_dim}) | device={DEVICE}")
    mlp       = MLP(emb_dim, HIDDEN_DIM, N_LAYERS, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(
        esm_learnable + list(mlp.parameters()), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_mlp_state = None
    best_f_emb     = None
    no_improve     = 0

    epoch_bar = tqdm(range(1, MAX_EPOCHS + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        mlp.train()
        esm.train()   # allow Dropout in ESM3 to be active if any
        train_loss = 0.0

        for seqs, y in tqdm(train_loader, desc=f"  epoch {epoch:3d} train", leave=False):
            y = y.to(DEVICE)
            optimizer.zero_grad()
            pooled = embed_batch(esm, tok, seqs)   # f_embedding gradient flows here
            loss   = criterion(mlp(pooled), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
        train_loss /= len(train_loader.dataset)

        mlp.eval()
        esm.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seqs, y in tqdm(val_loader, desc=f"  epoch {epoch:3d} val  ", leave=False):
                pooled    = embed_batch(esm, tok, seqs)
                val_loss += criterion(mlp(pooled), y.to(DEVICE)).item() * len(y)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_mlp_state = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
            best_f_emb     = esm_learnable[0].detach().cpu().clone()
            no_improve     = 0
        else:
            no_improve += 1

        epoch_bar.set_postfix(
            train=f"{train_loss:.4f}", val=f"{val_loss:.4f}",
            best=f"{best_val_loss:.4f}", patience=no_improve,
        )

        if no_improve >= PATIENCE:
            print(f"\n  Early stop at epoch {epoch}")
            break

    # Restore best weights
    mlp.load_state_dict({k: v.to(DEVICE) for k, v in best_mlp_state.items()})
    with torch.no_grad():
        esm_learnable[0].copy_(best_f_emb.to(DEVICE))
    mlp.eval()
    esm.eval()

    # Final evaluation — pre-compute embeddings once with best weights
    print("\n[eval] Pre-computing test embeddings …")
    emb_test = embed_sequences(esm, tok, ds["test"]["Peptide"], desc="Test  ")
    y_test   = np.array(ds["test"]["B"], dtype=np.float32)

    print("[eval] Computing test metrics …")
    test_metrics = evaluate(mlp, emb_test, y_test)

    print("[eval] Stereo-pair ordering accuracy …")
    stereo_metrics = stereo_ordering_accuracy(mlp, esm, tok, stereo)

    elapsed = time.time() - t0

    lines = [
        "=" * 60,
        "ESM3 Embedding MLP Benchmark  (learnable D-Phe token)",
        f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: {ESM_MODEL}  |  Embedding dim: {emb_dim}",
        f"Learnable params: f_embedding (1×{emb_dim}) + MLP head",
        f"MLP: {N_LAYERS} hidden layers × {HIDDEN_DIM}  |  dropout={DROPOUT}",
        f"Epochs (max {MAX_EPOCHS}, patience {PATIENCE})  |  "
        f"LR={LR}  batch={TRAIN_BATCH}",
        f"Device: {DEVICE}  |  Total time: {elapsed:.1f}s",
        "-" * 60,
        "Test split metrics:",
        f"  RMSE      : {test_metrics['rmse']:.4f}",
        f"  MAE       : {test_metrics['mae']:.4f}",
        f"  Pearson r : {test_metrics['pearson']:.4f}",
        f"  Spearman ρ: {test_metrics['spearman']:.4f}",
        f"  Kendall τ : {test_metrics['kendall']:.4f}",
        "-" * 60,
        "Stereo-pair ordering (D-Phe 'f' vs L-Phe 'F'):",
        f"  Total pairs    : {stereo_metrics['n_pairs']}",
        f"  Tied (true)    : {stereo_metrics['n_tied_true']}",
        f"  Tied (pred)    : {stereo_metrics['n_tied_pred']}",
        f"  Evaluated      : {stereo_metrics['n_evaluated']}",
        f"  Correct        : {stereo_metrics['n_correct']}",
        f"  Ordering acc.  : {stereo_metrics['ordering_accuracy']:.4f}",
        "=" * 60,
    ]

    report = "\n".join(lines)
    print("\n" + report)
    RESULTS_FILE.write_text(report + "\n")
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
