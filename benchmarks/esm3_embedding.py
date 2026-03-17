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

Results are written to benchmarks/results_esm3_embedding_seed{N}.json.
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
PATIENCE     = 5      # overridden at runtime to 0.1 * MAX_EPOCHS
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR  = Path(__file__).parent


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
        ordering_acc=float(acc),
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
    esm,
    tok,
    f_emb_init: torch.Tensor,
    esm_learnable: list,
    train_loader,
    val_loader,
    y_test: np.ndarray,
    stereo,
) -> tuple[dict, dict, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Reset learnable f_embedding to its initial value
    with torch.no_grad():
        esm_learnable[0].copy_(f_emb_init.to(DEVICE))

    emb_dim = esm_learnable[0].shape[0]
    mlp     = MLP(emb_dim, HIDDEN_DIM, N_LAYERS, DROPOUT).to(DEVICE)

    optimizer = torch.optim.Adam(
        esm_learnable + list(mlp.parameters()), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss  = float("inf")
    best_mlp_state = None
    best_f_emb     = None
    no_improve     = 0
    epochs_run     = 0

    epoch_bar = tqdm(range(1, MAX_EPOCHS + 1), desc="  Training", unit="epoch")
    for epoch in epoch_bar:
        mlp.train()
        esm.train()
        train_loss = 0.0
        for seqs, y in tqdm(train_loader, desc=f"    epoch {epoch:3d} train", leave=False):
            y = y.to(DEVICE)
            optimizer.zero_grad()
            pooled = embed_batch(esm, tok, seqs)
            loss   = criterion(mlp(pooled), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
        train_loss /= len(train_loader.dataset)

        mlp.eval()
        esm.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seqs, y in tqdm(val_loader, desc=f"    epoch {epoch:3d} val  ", leave=False):
                pooled    = embed_batch(esm, tok, seqs)
                val_loss += criterion(mlp(pooled), y.to(DEVICE)).item() * len(y)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)
        epochs_run = epoch

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
            print(f"\n    Early stop at epoch {epoch}")
            break

    mlp.load_state_dict({k: v.to(DEVICE) for k, v in best_mlp_state.items()})
    with torch.no_grad():
        esm_learnable[0].copy_(best_f_emb.to(DEVICE))
    mlp.eval()
    esm.eval()

    print("  [eval] Pre-computing test embeddings …")
    emb_test     = embed_sequences(esm, tok, list(ds_test_peptides), desc="Test  ")
    test_metrics = evaluate(mlp, emb_test, y_test)
    print(f"  RMSE={test_metrics['rmse']:.4f}  "
          f"Pearson={test_metrics['pearson']:+.4f}  "
          f"Spearman={test_metrics['spearman']:+.4f}")

    stereo_metrics = stereo_ordering_accuracy(mlp, esm, tok, stereo)
    print(f"  Ordering accuracy: {stereo_metrics['ordering_acc']:.4f}  "
          f"({stereo_metrics['n_correct']}/{stereo_metrics['n_evaluated']})")

    training_summary = {"epochs_run": epochs_run, "best_val_loss": best_val_loss}
    return test_metrics, stereo_metrics, training_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0,
                        help="Training seed (default: 0)")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS,
                        help=f"Max training epochs (default: {MAX_EPOCHS})")
    args = parser.parse_args()
    seed = args.seed

    global MAX_EPOCHS, PATIENCE, ds_test_peptides
    MAX_EPOCHS = args.epochs
    PATIENCE   = max(1, int(0.1 * MAX_EPOCHS))

    print(f"Device: {DEVICE}  |  seed={seed}  |  max_epochs={MAX_EPOCHS}  |  patience={PATIENCE}")
    t0 = time.time()

    print("[data] Loading peptag dataset …")
    ds     = hf_load_dataset(HF_REPO, "peptag")
    stereo = hf_load_dataset(HF_REPO, "stereo_pairs")["stereo_pairs"]
    ds_test_peptides = ds["test"]["Peptide"]

    esm, tok = load_esm3()

    esm_learnable = [p for p in esm.parameters() if p.requires_grad]
    print(f"[ESM3] Learnable ESM3 params: {sum(p.numel() for p in esm_learnable)}  "
          f"(just the 'f' embedding vector)")

    # Snapshot the initial f_embedding so we can reset it between seeds
    f_emb_init = esm_learnable[0].detach().cpu().clone()
    emb_dim    = f_emb_init.shape[0]

    y_train = np.array(ds["train"]["B"], dtype=np.float32)
    y_val   = np.array(ds["val"]["B"],   dtype=np.float32)
    y_test  = np.array(ds["test"]["B"],  dtype=np.float32)

    train_loader = DataLoader(
        SequenceDataset(ds["train"]["Peptide"], y_train),
        batch_size=TRAIN_BATCH, shuffle=True, collate_fn=seq_collate,
    )
    val_loader = DataLoader(
        SequenceDataset(ds["val"]["Peptide"], y_val),
        batch_size=TRAIN_BATCH, shuffle=False, collate_fn=seq_collate,
    )

    print(f"\n── Seed {seed} ──")
    test_metrics, stereo_metrics, training_summary = run_one_seed(
        seed, esm, tok, f_emb_init, esm_learnable,
        train_loader, val_loader, y_test, stereo,
    )

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    config = {
        "esm_model": ESM_MODEL, "emb_dim": emb_dim, "hidden_dim": HIDDEN_DIM,
        "n_layers": N_LAYERS, "dropout": DROPOUT, "lr": LR,
        "weight_decay": WEIGHT_DECAY, "train_batch": TRAIN_BATCH,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "device": DEVICE,
    }
    save_results(seed, test_metrics, stereo_metrics, training_summary, config, RESULTS_DIR, "results_esm3_embedding")


if __name__ == "__main__":
    main()
