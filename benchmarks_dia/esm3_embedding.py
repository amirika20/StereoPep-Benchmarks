"""
ESM embedding benchmark for the DIA dataset.

Same pipeline as benchmarks/esm3_embedding.py, adapted for local DIA
retention-time data.  DIA sequences contain only standard amino acids
(no D-Phe 'f'), so no learnable token embedding is needed.

Supports ESM3 and ESM-C model families:
  esm3_sm    – ESM3-small  (esm3_sm_open_v0)
  esmc_300m  – ESM-C 300 M (esmc_300m_2024_12)
  esmc_600m  – ESM-C 600 M (esmc_600m_2024_12)

Backbone is frozen; only the MLP regression head is trained.
No diastereomer/tag/substitution pair metrics.

Results written to benchmarks_dia/output/results_{model_key}_embedding_dia_seed{N}.json.
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
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    from huggingface_hub import login as _hf_login
    _hf_login(token=_hf_token, add_to_git_credential=False)

# ── config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"

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
TRAIN_BATCH  = 32
ENCODE_BATCH = 32
MAX_EPOCHS   = 20
PATIENCE     = 5
LR_PATIENCE  = 10
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR  = Path(__file__).parent / "output"
WEIGHTS_DIR  = Path(__file__).parent / "weights"


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


# ── model helpers ─────────────────────────────────────────────────────────────

def _get_tokenizer(model, family: str):
    if family == "esm3":
        return model.tokenizers.sequence
    if family == "esmc":
        return model.tokenizer
    raise ValueError(f"Unknown ESM family '{family}'")


def load_esm_model(model_key: str):
    cfg    = MODELS[model_key]
    family = cfg["family"]
    mod    = importlib.import_module("esm.pretrained")
    loader = getattr(mod, cfg["import_fn"])

    print(f"[ESM] Loading {cfg['display_name']} on {DEVICE} …")
    model = loader(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tok = _get_tokenizer(model, family)
    return model, tok


# ── embedding ─────────────────────────────────────────────────────────────────

def embed_batch(model, tok, sequences: list[str]) -> torch.Tensor:
    tokens    = tok(sequences, return_tensors="pt", padding=True)
    input_ids = tokens["input_ids"].to(DEVICE)
    attn_mask = tokens["attention_mask"].to(DEVICE)

    out = model(sequence_tokens=input_ids)
    emb = out.embeddings                        # (B, L, D)

    residue_mask = attn_mask.clone().float()
    residue_mask[:, 0] = 0.0
    seq_lens = attn_mask.sum(dim=1)
    for b, l in enumerate(seq_lens):
        residue_mask[b, l - 1] = 0.0

    residue_mask = residue_mask.unsqueeze(-1)
    pooled = (emb * residue_mask).sum(dim=1) / residue_mask.sum(dim=1).clamp(min=1)
    return pooled


@torch.no_grad()
def embed_sequences(model, tok, sequences: list[str], desc: str = "Embedding") -> np.ndarray:
    all_embs = []
    for i in tqdm(range(0, len(sequences), ENCODE_BATCH), desc=desc):
        batch = sequences[i:i+ENCODE_BATCH]
        all_embs.append(embed_batch(model, tok, batch).cpu().float().numpy())
    return np.concatenate(all_embs, axis=0)


# ── dataset ───────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    def __init__(self, sequences: list[str], y: np.ndarray):
        self.sequences = sequences
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):  return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx], self.y[idx]


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


def evaluate(mlp: MLP, embeddings: np.ndarray, y: np.ndarray) -> dict:
    mlp.eval()
    X = torch.tensor(embeddings, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X), batch_size=256)
    preds = []
    with torch.no_grad():
        for (batch,) in loader:
            preds.append(mlp(batch.to(DEVICE)).cpu())
    return regression_metrics(y, torch.cat(preds).numpy())


# ── reporting ─────────────────────────────────────────────────────────────────

class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def save_results(seed, test_metrics, train_metrics, training, config, output_dir, stem):
    result = {
        "benchmark": stem, "dataset": "dia", "seed": seed,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "config": config, "training": training,
        "test_metrics": test_metrics, "train_metrics": train_metrics,
    }
    out = output_dir / f"{stem}_seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, cls=_NpEncoder))
    print(f"\nResults saved to {out}")


# ── run one seed ──────────────────────────────────────────────────────────────

def run_one_seed(
    seed: int,
    model,
    tok,
    train_loader,
    val_loader,
    train_seqs: list[str],
    y_train: np.ndarray,
    test_seqs: list[str],
    y_test: np.ndarray,
    emb_dim: int,
    weights_path=None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    mlp = MLP(emb_dim, HIDDEN_DIM, N_LAYERS, DROPOUT).to(DEVICE)

    if weights_path is not None and weights_path.exists():
        print(f"  [weights] Loading from {weights_path} — skipping training")
        ckpt = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        mlp.load_state_dict(ckpt["mlp"])
        mlp.eval()
        epochs_run    = ckpt["epochs_run"]
        best_val_loss = ckpt["best_val_loss"]
    else:
        optimizer = torch.optim.Adam(mlp.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=LR_PATIENCE, factor=0.5, min_lr=1e-6)
        criterion = nn.MSELoss()

        best_val_loss  = float("inf")
        best_mlp_state = None
        no_improve     = 0
        epochs_run     = 0

        epoch_bar = tqdm(range(1, MAX_EPOCHS + 1), desc="  Training", unit="epoch")
        for epoch in epoch_bar:
            mlp.train()
            model.eval()   # backbone stays frozen
            train_loss = 0.0
            for seqs, y in tqdm(train_loader, desc=f"    epoch {epoch:3d} train", leave=False):
                y = y.to(DEVICE)
                optimizer.zero_grad()
                with torch.no_grad():
                    pooled = embed_batch(model, tok, seqs)
                loss = criterion(mlp(pooled), y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(y)
            train_loss /= len(train_loader.dataset)

            mlp.eval()
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
                no_improve     = 0
            else:
                no_improve += 1

            epoch_bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}",
                                  best=f"{best_val_loss:.4f}", patience=no_improve)
            if no_improve >= PATIENCE:
                print(f"\n    Early stop at epoch {epoch}")
                break

        mlp.load_state_dict({k: v.to(DEVICE) for k, v in best_mlp_state.items()})
        mlp.eval()

        if weights_path is not None:
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"mlp": mlp.state_dict(), "epochs_run": epochs_run,
                        "best_val_loss": best_val_loss}, weights_path)
            print(f"  [weights] Saved to {weights_path}")

    print("  [eval] Pre-computing test embeddings …")
    emb_test     = embed_sequences(model, tok, test_seqs, desc="Test  ")
    test_metrics = evaluate(mlp, emb_test, y_test)
    print(f"  RMSE={test_metrics['rmse']:.4f}  "
          f"Pearson={test_metrics['pearson']:+.4f}  "
          f"Spearman={test_metrics['spearman']:+.4f}")

    print("  [eval] Pre-computing train embeddings …")
    emb_train     = embed_sequences(model, tok, train_seqs, desc="Train ")
    train_metrics = evaluate(mlp, emb_train, y_train)
    print(f"  [train] RMSE={train_metrics['rmse']:.4f}  Pearson={train_metrics['pearson']:+.4f}")

    return test_metrics, train_metrics, {"epochs_run": epochs_run, "best_val_loss": best_val_loss}


# ── main ──────────────────────────────────────────────────────────────────────

def run_model(model_key: str, seed: int) -> None:
    cfg    = MODELS[model_key]
    family = cfg["family"]

    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"Model: {cfg['display_name']}")
    print(f"{'='*60}")

    model, tok = load_esm_model(model_key)

    print("[data] Loading DIA dataset …")
    train_seqs, y_train, val_seqs, y_val, test_seqs, y_test = load_dia_data()
    print(f"  train={len(train_seqs)}  val={len(val_seqs)}  test={len(test_seqs)}")

    # Determine embedding dimension via a single forward pass
    with torch.no_grad():
        tokens    = tok([train_seqs[0]], return_tensors="pt", padding=True)
        input_ids = tokens["input_ids"].to(DEVICE)
        out       = model(sequence_tokens=input_ids)
        emb_dim   = out.embeddings.shape[-1]
    print(f"  Embedding dim: {emb_dim}")

    train_loader = DataLoader(
        SequenceDataset(train_seqs, y_train),
        batch_size=TRAIN_BATCH, shuffle=True, collate_fn=seq_collate,
    )
    val_loader = DataLoader(
        SequenceDataset(val_seqs, y_val),
        batch_size=TRAIN_BATCH, collate_fn=seq_collate,
    )

    stem         = f"results_{model_key}_embedding_dia"
    weights_path = WEIGHTS_DIR / f"{stem}_seed{seed}.pt"
    print(f"\n── Seed {seed} ──")
    test_metrics, train_metrics, training = run_one_seed(
        seed, model, tok,
        train_loader, val_loader,
        train_seqs, y_train, test_seqs, y_test,
        emb_dim, weights_path=weights_path,
    )

    print(f"\nTotal time for {model_key}: {time.time() - t0:.1f}s")
    config = {
        "model_key": model_key, "display_name": cfg["display_name"],
        "hf_name": cfg["hf_name"], "family": family, "emb_dim": emb_dim,
        "hidden_dim": HIDDEN_DIM, "n_layers": N_LAYERS, "dropout": DROPOUT,
        "lr": LR, "weight_decay": WEIGHT_DECAY,
        "train_batch": TRAIN_BATCH, "encode_batch": ENCODE_BATCH,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "lr_patience": LR_PATIENCE, "device": DEVICE,
    }
    save_results(seed, test_metrics, train_metrics, training, config,
                 RESULTS_DIR, stem)


def main() -> None:
    global MAX_EPOCHS, PATIENCE, LR_PATIENCE
    parser = argparse.ArgumentParser(
        description="ESM embedding benchmark for DIA dataset",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--seed",   type=int, default=0)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument(
        "--model", default="esm3_sm",
        help=(
            "Model(s) to benchmark:\n"
            + "\n".join(f"  {k:12s} – {v['display_name']}" for k, v in MODELS.items())
            + "\n  all          – run all models sequentially\n"
            "Comma-separate for multiple, e.g. --model esm3_sm,esmc_300m"
        ),
    )
    args = parser.parse_args()
    seed = args.seed

    MAX_EPOCHS  = args.epochs
    PATIENCE    = max(1, int(0.10 * MAX_EPOCHS))

    if args.model == "all":
        model_keys = ALL_MODELS
    else:
        model_keys = [k.strip() for k in args.model.split(",")]

    unknown = [k for k in model_keys if k not in MODELS]
    if unknown:
        parser.error(f"Unknown model(s): {', '.join(unknown)}. "
                     f"Valid: {', '.join(ALL_MODELS)}")

    print(f"Device: {DEVICE}  |  seed={seed}  |  max_epochs={MAX_EPOCHS}  |  patience={PATIENCE}")
    for model_key in model_keys:
        run_model(model_key, seed)


if __name__ == "__main__":
    main()
