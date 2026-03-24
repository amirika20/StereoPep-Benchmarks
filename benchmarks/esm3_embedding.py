"""
ESM embedding benchmark for the PepTag dataset.

Supports the full ESM3 and ESM-C (Cambrian) model families:
  esm3_sm    – ESM3-small  (esm3_sm_open_v0,   ~300 M params)
  esmc_300m  – ESM-C 300 M (esmc_300m_2024_12, ~300 M params)
  esmc_600m  – ESM-C 600 M (esmc_600m_2024_12, ~600 M params)

Pass --model <key>        to run a single model.
Pass --model <k1>,<k2>   to run a subset.
Pass --model all          to run every model sequentially.

D-Phenylalanine ('f') is not in the ESM vocabulary.  We add it as a new token
and represent it with a *learnable* embedding vector initialised as a copy of
L-Phe ('F').  Only this single vector and the MLP prediction head are trained;
all other backbone parameters stay frozen.

The learnable 'f' embedding is implemented via a differentiable mask-blend in
LearnableTokenEmbed, so gradients flow through it correctly even though the rest
of the backbone is frozen.

Because 'f_embedding' changes each step, the model must be called on-the-fly
during training — pre-computed embeddings would become stale.

Pipeline
--------
1. Load backbone; freeze it; replace the sequence embedding with LearnableTokenEmbed for 'f'.
2. Train loop: tokenise → backbone forward → mean-pool per batch (on-the-fly).
   Gradients flow through f_embedding and the MLP head only.
3. Evaluate on the test split with regression metrics (pre-computed after train).
4. Evaluate stereochemistry ordering accuracy on the stereo_pairs split.

Results are written to benchmarks/output/results_{model_key}_embedding_seed{N}.json.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
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

# ── HuggingFace auth ──────────────────────────────────────────────────────────
# Set HF_TOKEN env var (e.g. in your SLURM script) to authenticate for gated models.
_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    from huggingface_hub import login as _hf_login
    _hf_login(token=_hf_token, add_to_git_credential=False)

# ── config ────────────────────────────────────────────────────────────────────
HF_REPO      = "amirka20/peptag"

# Model registry ──────────────────────────────────────────────────────────────
# import_fn  : name of the loader in esm.pretrained
# hf_name    : canonical model name (written to results JSON)
# family     : "esm3" or "esmc" — controls which attribute holds the seq embed / tokenizer
MODELS: dict[str, dict] = {
    "esm3_sm": dict(
        display_name="ESM3-small (esm3_sm_open_v0)",
        hf_name="esm3_sm_open_v0",
        import_fn="ESM3_sm_open_v0",
        family="esm3",
    ),
    "esmc_300m": dict(
        display_name="ESM-C 300M (esmc_300m_2024_12)",
        hf_name="esmc_300m_2024_12",
        import_fn="ESMC_300M_202412",
        family="esmc",
    ),
    "esmc_600m": dict(
        display_name="ESM-C 600M (esmc_600m_2024_12)",
        hf_name="esmc_600m_2024_12",
        import_fn="ESMC_600M_202412",
        family="esmc",
    ),
}
ALL_MODELS = list(MODELS.keys())

HIDDEN_DIM   = 512
N_LAYERS     = 3
DROPOUT      = 0.1
LR           = 1e-3
WEIGHT_DECAY = 1e-4
TRAIN_BATCH  = 32    # smaller: each batch runs a full backbone forward pass
ENCODE_BATCH = 32    # batch size for no-grad encoding (val/test/stereo)
MAX_EPOCHS   = 20
PATIENCE     = 5      # overridden at runtime to 0.1 * MAX_EPOCHS
LR_PATIENCE  = 2      # overridden at runtime to 0.05 * MAX_EPOCHS
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR  = Path(__file__).parent / "output"


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


# ── model helpers ─────────────────────────────────────────────────────────────

def _get_seq_embed(model: nn.Module, family: str) -> nn.Embedding:
    """Return the raw sequence embedding layer for the given model family."""
    if family == "esm3":
        return model.encoder.sequence_embed   # ESM3: EncodeInputs.sequence_embed
    if family == "esmc":
        return model.embed                    # ESM-C: ESMC.embed
    raise ValueError(f"Unknown model family '{family}'")


def _set_seq_embed(model: nn.Module, family: str, new_embed: nn.Module) -> None:
    """Replace the sequence embedding layer in-place."""
    if family == "esm3":
        model.encoder.sequence_embed = new_embed
    elif family == "esmc":
        model.embed = new_embed
    else:
        raise ValueError(f"Unknown model family '{family}'")


def _get_tokenizer(model: nn.Module, family: str):
    """Return the sequence tokenizer for the given model family."""
    if family == "esm3":
        return model.tokenizers.sequence   # TokenizerCollectionProtocol
    if family == "esmc":
        return model.tokenizer             # EsmSequenceTokenizer
    raise ValueError(f"Unknown model family '{family}'")


def load_model(model_key: str):
    """
    Load an ESM backbone, freeze it, and patch in a learnable 'f' (D-Phe) token.
    Returns (model, tokenizer).
    """
    cfg     = MODELS[model_key]
    family  = cfg["family"]
    mod     = importlib.import_module("esm.pretrained")
    loader  = getattr(mod, cfg["import_fn"])

    print(f"[ESM] Loading {cfg['display_name']} on {DEVICE} …")
    model = loader(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tok   = _get_tokenizer(model, family)
    vocab = tok.get_vocab()

    if "f" not in vocab:
        tok.add_tokens(["f"])
        f_id    = tok.convert_tokens_to_ids("f")
        F_id    = vocab["F"]
        seq_emb = _get_seq_embed(model, family)
        f_init  = seq_emb.weight[F_id].detach()
        new_emb = LearnableTokenEmbed(seq_emb, f_id, f_init).to(DEVICE)
        _set_seq_embed(model, family, new_emb)
        print(f"[ESM] Added learnable 'f' (D-Phe) token_id={f_id}, init from 'F' id={F_id}")
    else:
        print("[ESM] Token 'f' already in vocabulary")

    return model, tok


# ── embedding helpers ─────────────────────────────────────────────────────────

def embed_batch(model, tok, sequences: list[str]) -> torch.Tensor:
    """
    Tokenise → backbone forward → mean-pool for a list of sequences.
    Returns (B, D).  Supports gradient flow (no torch.no_grad here).
    Works for both ESM3 and ESM-C families.
    """
    tokens    = tok(sequences, return_tensors="pt", padding=True)
    input_ids = tokens["input_ids"].to(DEVICE)
    attn_mask = tokens["attention_mask"].to(DEVICE)

    out = model(sequence_tokens=input_ids)
    emb = out.embeddings                        # (B, L, D)

    # Exclude CLS (pos 0) and EOS (last real token) from mean pool
    residue_mask = attn_mask.clone().float()
    residue_mask[:, 0] = 0.0
    seq_lens = attn_mask.sum(dim=1)
    for b, l in enumerate(seq_lens):
        residue_mask[b, l - 1] = 0.0

    residue_mask = residue_mask.unsqueeze(-1)   # (B, L, 1)
    pooled = (emb * residue_mask).sum(dim=1) / residue_mask.sum(dim=1).clamp(min=1)
    return pooled                               # (B, D)


@torch.no_grad()
def embed_sequences(model, tok, sequences: list[str], desc: str = "Embedding") -> np.ndarray:
    """Pre-compute embeddings for a list of sequences (no gradient). Returns (N, D)."""
    all_embs = []
    for i in tqdm(range(0, len(sequences), ENCODE_BATCH), desc=desc):
        batch = sequences[i : i + ENCODE_BATCH]
        all_embs.append(embed_batch(model, tok, batch).cpu().float().numpy())
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


def stereo_ordering_accuracy(mlp: MLP, model, tok, stereo_ds) -> dict:
    seqs_f = stereo_ds["Sequence_f"]
    seqs_F = stereo_ds["Sequence_F"]
    B_f    = np.array(stereo_ds["B_f"])
    B_F    = np.array(stereo_ds["B_F"])

    emb_f = embed_sequences(model, tok, seqs_f, desc="Stereo D-Phe")
    emb_F = embed_sequences(model, tok, seqs_F, desc="Stereo L-Phe")

    mlp.eval()
    with torch.no_grad():
        pred_f = mlp(torch.tensor(emb_f).to(DEVICE)).cpu().numpy()
        pred_F = mlp(torch.tensor(emb_F).to(DEVICE)).cpu().numpy()

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
    model,
    tok,
    f_emb_init: torch.Tensor,
    model_learnable: list,
    train_loader,
    val_loader,
    y_test: np.ndarray,
    stereo,
) -> tuple[dict, dict, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Reset learnable f_embedding to its initial value
    with torch.no_grad():
        model_learnable[0].copy_(f_emb_init.to(DEVICE))

    emb_dim = model_learnable[0].shape[0]
    mlp     = MLP(emb_dim, HIDDEN_DIM, N_LAYERS, DROPOUT).to(DEVICE)

    optimizer = torch.optim.Adam(
        model_learnable + list(mlp.parameters()), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=LR_PATIENCE, factor=0.5, min_lr=1e-6)
    criterion = nn.MSELoss()

    best_val_loss  = float("inf")
    best_mlp_state = None
    best_f_emb     = None
    no_improve     = 0
    epochs_run     = 0

    epoch_bar = tqdm(range(1, MAX_EPOCHS + 1), desc="  Training", unit="epoch")
    for epoch in epoch_bar:
        mlp.train()
        model.train()
        train_loss = 0.0
        for seqs, y in tqdm(train_loader, desc=f"    epoch {epoch:3d} train", leave=False):
            y = y.to(DEVICE)
            optimizer.zero_grad()
            pooled = embed_batch(model, tok, seqs)
            loss   = criterion(mlp(pooled), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
        train_loss /= len(train_loader.dataset)

        mlp.eval()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seqs, y in tqdm(val_loader, desc=f"    epoch {epoch:3d} val  ", leave=False):
                pooled    = embed_batch(model, tok, seqs)
                val_loss += criterion(mlp(pooled), y.to(DEVICE)).item() * len(y)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)
        epochs_run = epoch

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_mlp_state = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
            best_f_emb     = model_learnable[0].detach().cpu().clone()
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
        model_learnable[0].copy_(best_f_emb.to(DEVICE))
    mlp.eval()
    model.eval()

    print("  [eval] Pre-computing test embeddings …")
    emb_test     = embed_sequences(model, tok, list(ds_test_peptides), desc="Test  ")
    test_metrics = evaluate(mlp, emb_test, y_test)
    print(f"  RMSE={test_metrics['rmse']:.4f}  "
          f"Pearson={test_metrics['pearson']:+.4f}  "
          f"Spearman={test_metrics['spearman']:+.4f}")

    stereo_metrics = stereo_ordering_accuracy(mlp, model, tok, stereo)
    print(f"  Ordering accuracy: {stereo_metrics['ordering_acc']:.4f}  "
          f"({stereo_metrics['n_correct']}/{stereo_metrics['n_evaluated']})")

    training_summary = {"epochs_run": epochs_run, "best_val_loss": best_val_loss}
    return test_metrics, stereo_metrics, training_summary


def main() -> None:
    global MAX_EPOCHS, PATIENCE, LR_PATIENCE, ds_test_peptides

    parser = argparse.ArgumentParser(
        description="ESM embedding benchmark (ESM3 / ESM-C families)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=0,
                        help="Training seed (default: 0)")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS,
                        help=f"Max training epochs (default: {MAX_EPOCHS})")
    parser.add_argument(
        "--model",
        default="esm3_sm",
        metavar="MODEL",
        help=(
            "Model(s) to benchmark. Options:\n"
            + "\n".join(f"  {k:12s} – {v['display_name']}" for k, v in MODELS.items())
            + "\n  all          – run all models sequentially\n"
            "Comma-separate multiple keys, e.g. --model esm3_sm,esmc_300m\n"
            f"(default: esm3_sm)"
        ),
    )
    args = parser.parse_args()

    seed        = args.seed
    MAX_EPOCHS  = args.epochs
    PATIENCE    = max(1, int(0.10 * MAX_EPOCHS))
    LR_PATIENCE = max(1, int(0.05 * MAX_EPOCHS))

    # Resolve requested model keys
    if args.model.strip().lower() == "all":
        model_keys = ALL_MODELS
    else:
        model_keys = [k.strip() for k in args.model.split(",")]
        unknown = [k for k in model_keys if k not in MODELS]
        if unknown:
            parser.error(
                f"Unknown model(s): {unknown}\n"
                f"Choose from: {', '.join(ALL_MODELS)}, all"
            )

    print(f"Device: {DEVICE}  |  seed={seed}  |  max_epochs={MAX_EPOCHS}  |  patience={PATIENCE}")
    print(f"Models : {model_keys}")

    # Load dataset once; share across all model runs
    print("\n[data] Loading peptag dataset …")
    ds     = hf_load_dataset(HF_REPO, "peptag")
    stereo = hf_load_dataset(HF_REPO, "stereo_pairs")["stereo_pairs"]
    ds_test_peptides = ds["test"]["Peptide"]

    y_train = np.array(ds["train"]["B"], dtype=np.float32)
    y_val   = np.array(ds["val"]["B"],   dtype=np.float32)
    y_test  = np.array(ds["test"]["B"],  dtype=np.float32)

    for model_key in model_keys:
        cfg = MODELS[model_key]
        print(f"\n{'='*60}")
        print(f"Model : {cfg['display_name']}")
        print(f"{'='*60}")
        t0 = time.time()

        model, tok = load_model(model_key)

        model_learnable = [p for p in model.parameters() if p.requires_grad]
        print(f"[ESM] Learnable params: {sum(p.numel() for p in model_learnable)}  "
              f"(just the 'f' embedding vector)")

        f_emb_init = model_learnable[0].detach().cpu().clone()
        emb_dim    = f_emb_init.shape[0]

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
            seed, model, tok, f_emb_init, model_learnable,
            train_loader, val_loader, y_test, stereo,
        )

        elapsed = time.time() - t0
        print(f"\nTotal time for {model_key}: {elapsed:.1f}s")

        config = {
            "esm_model":    cfg["hf_name"],
            "model_family": cfg["family"],
            "emb_dim":      emb_dim,
            "hidden_dim":   HIDDEN_DIM,
            "n_layers":     N_LAYERS,
            "dropout":      DROPOUT,
            "lr":           LR,
            "weight_decay": WEIGHT_DECAY,
            "train_batch":  TRAIN_BATCH,
            "max_epochs":   MAX_EPOCHS,
            "patience":     PATIENCE,
            "lr_patience":  LR_PATIENCE,
            "device":       DEVICE,
        }
        stem = f"results_{model_key}_embedding"
        save_results(seed, test_metrics, stereo_metrics, training_summary, config,
                     RESULTS_DIR, stem)

        # Free GPU memory before loading the next model
        del model, tok, model_learnable, f_emb_init, train_loader, val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
