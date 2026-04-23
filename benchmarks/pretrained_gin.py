"""
Pretrained GIN benchmark for the StereoPep dataset.

Uses the Graph Isomorphism Network (GIN) pretrained on 2M molecules from
ZINC15 + ~450k bio-assay labels (Hu et al., ICLR 2020 — "Strategies for
Pre-training Graph Neural Networks").

Pretrained checkpoint: gin_supervised_contextpred
  → combines context-prediction self-supervision with supervised bio-activity
    pre-training; consistently among the best-performing variants.

Pipeline:
  1. Download pretrained GIN weights (snap-stanford, chem domain).
  2. Convert full peptide SMILES → molecular graphs (atom/bond features
     matching the exact featurisation used during pre-training).
  3. Fine-tune the GNN end-to-end together with a small MLP head to predict
     B (retention time, normalised 0-100) on the stereopep train split.
  4. Evaluate on test split with regression metrics.
  5. Evaluate stereochemistry ordering accuracy on stereo_pairs (D-Phe vs
     L-Phe).

Results are written to benchmarks/results_pretrained_gin_seed{N}.json.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset as hf_load_dataset
from rdkit import Chem
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops
from torch.utils.data import DataLoader as PlainDataLoader

# ── config ────────────────────────────────────────────────────────────────────
HF_REPO       = "amirka20/StereoPep"

# GIN architecture — must match the pretrained weights exactly
NUM_ATOM_TYPES     = 120
NUM_CHIRALITY_TAGS = 3
NUM_BOND_TYPES     = 6
NUM_BOND_DIRS      = 3
GIN_LAYERS         = 5
GIN_EMB_DIM        = 300

# Pretrained checkpoint (chem/supervised_contextpred from snap-stanford)
PRETRAINED_URL = (
    "https://raw.githubusercontent.com/snap-stanford/pretrain-gnns"
    "/master/chem/model_gin/supervised_contextpred.pth"
)
PRETRAINED_DIR  = Path(__file__).parent / "pretrained_weights"
WEIGHTS_FILE    = PRETRAINED_DIR / "gin_supervised_contextpred.pth"

# Fine-tuning
HEAD_HIDDEN   = 256
HEAD_LAYERS   = 2
DROPOUT       = 0.1
LR_BACKBONE   = 1e-4   # lower LR for pretrained GNN backbone
LR_HEAD       = 1e-3   # higher LR for the new prediction head
WEIGHT_DECAY  = 1e-4
BATCH_SIZE    = 64     # smaller batches — each sample is a molecular graph
MAX_EPOCHS    = 20
PATIENCE      = 5      # overridden at runtime to 0.1 * MAX_EPOCHS
LR_PATIENCE   = 10
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR   = Path(__file__).parent / "output"
WEIGHTS_DIR   = Path(__file__).parent / "weights"


# ── pretrained weight download ────────────────────────────────────────────────

def download_weights() -> None:
    if WEIGHTS_FILE.exists():
        print(f"  Weights already cached at {WEIGHTS_FILE}")
        return
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading pretrained GIN from snap-stanford …")
    try:
        urllib.request.urlretrieve(PRETRAINED_URL, WEIGHTS_FILE)
        print(f"  Saved to {WEIGHTS_FILE}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download pretrained weights from {PRETRAINED_URL}.\n"
            f"Error: {e}\n"
            f"Download manually and place at {WEIGHTS_FILE}"
        )


# ── molecular graph featurisation ─────────────────────────────────────────────
# Atom/bond feature encoding must match exactly what was used during pre-training
# (see snap-stanford/pretrain-gnns/chem/loader.py).

_BOND_TYPE_MAP = {
    Chem.rdchem.BondType.SINGLE:   0,
    Chem.rdchem.BondType.DOUBLE:   1,
    Chem.rdchem.BondType.TRIPLE:   2,
    Chem.rdchem.BondType.AROMATIC: 3,
}
_BOND_DIR_MAP = {
    Chem.rdchem.BondDir.NONE:        0,
    Chem.rdchem.BondDir.ENDUPRIGHT:  1,
    Chem.rdchem.BondDir.ENDDOWNRIGHT: 2,
}
_CHIRALITY_MAP = {
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED:    0,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
    Chem.rdchem.ChiralType.CHI_OTHER:          2,  # rare; map to 2 (stay in-bounds)
}


def smiles_to_graph(smiles: str) -> Data | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    # Node features: [atomic_num_idx, chirality_idx]
    atom_feats = []
    for atom in mol.GetAtoms():
        atomic_num_idx = min(atom.GetAtomicNum() - 1, 117)  # 0-indexed, capped
        chirality_idx  = _CHIRALITY_MAP.get(atom.GetChiralTag(), 0)
        atom_feats.append([atomic_num_idx, chirality_idx])

    # Edge features: bidirectional, [bond_type_idx, bond_dir_idx]
    src, dst, edge_feats = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt  = _BOND_TYPE_MAP.get(bond.GetBondType(), 3)
        bd  = _BOND_DIR_MAP.get(bond.GetBondDir(), 0)
        src += [i, j];  dst += [j, i]
        edge_feats += [[bt, bd], [bt, bd]]

    x          = torch.tensor(atom_feats,  dtype=torch.long)
    edge_index = torch.tensor([src, dst],  dtype=torch.long)
    edge_attr  = torch.tensor(edge_feats,  dtype=torch.long)

    # Handle molecules with no bonds (shouldn't happen for peptides)
    if edge_index.numel() == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 2), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ── GIN model (architecture must match snap-stanford pretrained weights) ──────

class GINConv(MessagePassing):
    """GIN layer with edge-feature integration (Hu et al. 2020)."""

    def __init__(self, emb_dim: int):
        super().__init__(aggr="add")
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim),
        )
        self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPES, emb_dim)
        self.edge_embedding2 = nn.Embedding(NUM_BOND_DIRS,  emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight)
        nn.init.xavier_uniform_(self.edge_embedding2.weight)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        # Add self-loops and corresponding "self-loop" edge features
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros(x.size(0), 2, dtype=torch.long, device=x.device)
        self_loop_attr[:, 0] = 4  # bond type index reserved for self-loops
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)

        edge_emb = self.edge_embedding1(edge_attr[:, 0]) + \
                   self.edge_embedding2(edge_attr[:, 1])
        return self.propagate(edge_index, x=x, edge_attr=edge_emb)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return x_j + edge_attr

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        return self.mlp(aggr_out)


class GINEncoder(nn.Module):
    """5-layer GIN encoder; produces node-level embeddings."""

    def __init__(
        self,
        num_layers: int = GIN_LAYERS,
        emb_dim: int = GIN_EMB_DIM,
        drop_ratio: float = 0.0,
    ):
        super().__init__()
        self.x_embedding1  = nn.Embedding(NUM_ATOM_TYPES,     emb_dim)
        self.x_embedding2  = nn.Embedding(NUM_CHIRALITY_TAGS, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight)
        nn.init.xavier_uniform_(self.x_embedding2.weight)

        self.gnns        = nn.ModuleList([GINConv(emb_dim) for _ in range(num_layers)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layers)])
        self.num_layers  = num_layers
        self.drop_ratio  = drop_ratio

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        for i, (gnn, bn) in enumerate(zip(self.gnns, self.batch_norms)):
            h = gnn(h, edge_index, edge_attr)
            h = bn(h)
            if i < self.num_layers - 1:
                h = F.relu(h)
            h = F.dropout(h, p=self.drop_ratio, training=self.training)
        return h


class GINPredictor(nn.Module):
    """GIN encoder + global mean pooling + MLP head."""

    def __init__(
        self,
        head_hidden: int = HEAD_HIDDEN,
        head_layers: int = HEAD_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.encoder = GINEncoder(drop_ratio=dropout)

        layers: list[nn.Module] = []
        in_dim = GIN_EMB_DIM
        for _ in range(head_layers):
            layers += [nn.Linear(in_dim, head_hidden), nn.LayerNorm(head_hidden),
                       nn.GELU(), nn.Dropout(dropout)]
            in_dim = head_hidden
        layers.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, data: Batch) -> torch.Tensor:
        node_emb  = self.encoder(data.x, data.edge_index, data.edge_attr)
        graph_emb = global_mean_pool(node_emb, data.batch)
        return self.head(graph_emb).squeeze(-1)

    def load_pretrained_encoder(self, path: Path) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        missing, unexpected = self.encoder.load_state_dict(state, strict=True)
        if missing:
            print(f"  WARNING — missing keys: {missing}")
        if unexpected:
            print(f"  WARNING — unexpected keys: {unexpected}")


# ── dataset helpers ───────────────────────────────────────────────────────────

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, graphs: list[Data], targets: np.ndarray):
        self.graphs  = graphs
        self.targets = targets

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx].clone()
        g.y = torch.tensor(self.targets[idx], dtype=torch.float)
        return g


def collate_fn(batch: list[Data]) -> Batch:
    return Batch.from_data_list(batch)


def encode_smiles(smiles_list: list[str], desc: str = "Encoding") -> tuple[list[Data], list[int]]:
    """Convert SMILES → graphs; return (graphs, bad_indices)."""
    graphs, bad = [], []
    for i, smi in enumerate(tqdm(smiles_list, desc=desc, leave=False)):
        g = smiles_to_graph(smi)
        if g is None:
            bad.append(i)
            # placeholder — will be masked out
            graphs.append(Data(
                x=torch.zeros((1, 2), dtype=torch.long),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 2),  dtype=torch.long),
            ))
        else:
            graphs.append(g)
    return graphs, bad


# ── training ──────────────────────────────────────────────────────────────────

def train(
    model: GINPredictor,
    train_loader: PlainDataLoader,
    val_loader: PlainDataLoader,
) -> list[dict]:
    # Separate optimiser groups for backbone (lower LR) and head
    backbone_params = list(model.encoder.parameters())
    head_params     = list(model.head.parameters())
    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": head_params,     "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=LR_PATIENCE, factor=0.5, min_lr=1e-6
    )
    criterion = nn.MSELoss()
    history = []
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    epoch_bar = tqdm(range(1, MAX_EPOCHS + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"  epoch {epoch:3d} train", leave=False):
            batch = batch.to(DEVICE)
            opt.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch.y)
            loss.backward()
            opt.step()
            train_loss += loss.item() * batch.num_graphs
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"  epoch {epoch:3d} val  ", leave=False):
                batch = batch.to(DEVICE)
                val_loss += criterion(model(batch), batch.y).item() * batch.num_graphs
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        epoch_bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}",
                              best=f"{best_val_loss:.4f}", patience=no_improve)

        if no_improve >= PATIENCE:
            print(f"\n  Early stop at epoch {epoch} (no val improvement for {PATIENCE} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def predict(model: GINPredictor, graphs: list[Data]) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(graphs), BATCH_SIZE):
            batch = Batch.from_data_list(graphs[i : i + BATCH_SIZE]).to(DEVICE)
            preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds)


# ── metrics ───────────────────────────────────────────────────────────────────

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mse":        float(mean_squared_error(y_true, y_pred)),
        "rmse":       float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":        float(mean_absolute_error(y_true, y_pred)),
        "mean_error": float(np.mean(y_pred - y_true)),
        "r2":         float(r2_score(y_true, y_pred)),
        "pearson":    float(stats.pearsonr(y_true, y_pred)[0]),
        "spearman":   float(stats.spearmanr(y_true, y_pred)[0]),
        "kendall":    float(stats.kendalltau(y_true, y_pred)[0]),
    }


def pair_delta_metrics(true_delta: np.ndarray, pred_delta: np.ndarray) -> dict:
    """Delta prediction quality metrics for any matched pair type."""
    rmse   = float(np.sqrt(mean_squared_error(true_delta, pred_delta)))
    mae    = float(mean_absolute_error(true_delta, pred_delta))
    pr, _  = stats.pearsonr(true_delta, pred_delta)
    sr, _  = stats.spearmanr(true_delta, pred_delta)
    kr, _  = stats.kendalltau(true_delta, pred_delta)
    mask   = np.sign(true_delta) != 0
    n_eval = int(mask.sum())
    n_corr = int((np.sign(true_delta[mask]) == np.sign(pred_delta[mask])).sum())
    _nz = true_delta != 0
    if _nz.sum() > 1 and len(np.unique((true_delta[_nz] > 0).astype(int))) > 1:
        delta_auc = float(roc_auc_score((true_delta[_nz] > 0).astype(int), pred_delta[_nz]))
    else:
        delta_auc = float("nan")
    return dict(
        n_pairs=len(true_delta),
        delta_pearson=float(pr),
        delta_spearman=float(sr),
        delta_kendall=float(kr),
        delta_rmse=rmse,
        delta_mae=mae,
        delta_auc=delta_auc,
        ordering_acc=float(n_corr / n_eval) if n_eval > 0 else float("nan"),
        n_correct=n_corr,
        n_evaluated=n_eval,
        mean_true_delta=float(true_delta.mean()),
        mean_pred_delta=float(pred_delta.mean()),
    )


def stereo_ordering_accuracy(model: GINPredictor, stereo_ds) -> dict:
    graphs_f, bad_f = encode_smiles(stereo_ds["SMILES_f"], desc="Stereo D-Phe")
    graphs_F, bad_F = encode_smiles(stereo_ds["SMILES_F"], desc="Stereo L-Phe")
    B_f     = np.array(stereo_ds["B_f"],    dtype=np.float64)
    B_F     = np.array(stereo_ds["B_F"],    dtype=np.float64)
    delta_B = np.array(stereo_ds["delta_B"], dtype=np.float64)

    pred_f = predict(model, graphs_f)
    pred_F = predict(model, graphs_F)
    pred_delta = pred_f - pred_F

    bad = set(bad_f) | set(bad_F)
    mask = np.ones(len(delta_B), dtype=bool)
    for i in bad:
        mask[i] = False

    true_sign = np.sign(delta_B[mask])
    pred_sign = np.sign(pred_delta[mask])
    correct   = int((true_sign == pred_sign).sum())
    total     = int(mask.sum())

    pr   = stats.pearsonr(delta_B[mask], pred_delta[mask])[0]
    sr   = stats.spearmanr(delta_B[mask], pred_delta[mask])[0]
    kr   = stats.kendalltau(delta_B[mask], pred_delta[mask])[0]
    rmse = float(np.sqrt(mean_squared_error(delta_B[mask], pred_delta[mask])))
    mae  = float(mean_absolute_error(delta_B[mask], pred_delta[mask]))
    _nz  = delta_B[mask] != 0
    if _nz.sum() > 1 and len(np.unique((delta_B[mask][_nz] > 0).astype(int))) > 1:
        delta_auc = float(roc_auc_score((delta_B[mask][_nz] > 0).astype(int), pred_delta[mask][_nz]))
    else:
        delta_auc = float("nan")

    return {
        "n_pairs":         total,
        "n_correct":       correct,
        "ordering_acc":    correct / total if total > 0 else float("nan"),
        "delta_pearson":   float(pr),
        "delta_spearman":  float(sr),
        "delta_kendall":   float(kr),
        "delta_rmse":      rmse,
        "delta_mae":       mae,
        "delta_auc":       delta_auc,
        "mean_true_delta": float(delta_B[mask].mean()),
        "mean_pred_delta": float(pred_delta[mask].mean()),
    }


def eval_pair_metrics(
    model: GINPredictor,
    ds,
    smiles_col_a: str,
    smiles_col_b: str,
) -> dict:
    """Evaluate predicted delta for any pair split (tag_pairs / substitution_pairs)."""
    delta_B    = np.array(ds["delta_B"], dtype=np.float64)
    graphs_a, bad_a = encode_smiles(list(ds[smiles_col_a]), desc=f"Pairs {smiles_col_a}")
    graphs_b, bad_b = encode_smiles(list(ds[smiles_col_b]), desc=f"Pairs {smiles_col_b}")

    pred_a     = predict(model, graphs_a)
    pred_b     = predict(model, graphs_b)
    pred_delta = pred_a - pred_b

    bad = set(bad_a) | set(bad_b)
    mask = np.ones(len(delta_B), dtype=bool)
    for i in bad:
        mask[i] = False

    return pair_delta_metrics(delta_B[mask], pred_delta[mask])


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
    train_metrics: dict,
    stereo_metrics: dict,
    stereo_trainval_metrics: dict,
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
        "train_metrics": train_metrics,
        "stereo_metrics": stereo_metrics,
        "stereo_trainval_metrics": stereo_trainval_metrics,
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
    graphs_train: list,
    graphs_val: list,
    graphs_test: list,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    sp,
    stereo_trainval,
    tag_pairs,
    sub_pairs,
    weights_path: Path | None = None,
) -> tuple[dict, dict, dict, dict, dict, list[dict]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = PlainDataLoader(
        GraphDataset(graphs_train, y_train), batch_size=BATCH_SIZE,
        shuffle=True,  collate_fn=collate_fn, num_workers=0,
    )
    val_loader = PlainDataLoader(
        GraphDataset(graphs_val, y_val), batch_size=BATCH_SIZE,
        shuffle=False, collate_fn=collate_fn, num_workers=0,
    )

    model = GINPredictor().to(DEVICE)

    if weights_path is not None and weights_path.exists():
        print(f"  [weights] Loading from {weights_path} — skipping training")
        ckpt    = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        history = ckpt["history"]
    else:
        model.load_pretrained_encoder(WEIGHTS_FILE)
        history = train(model, train_loader, val_loader)
        if weights_path is not None:
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "history": history}, weights_path)
            print(f"  [weights] Saved to {weights_path}")

    y_pred_test  = predict(model, graphs_test)
    test_metrics = regression_metrics(y_test, y_pred_test)
    print(f"  RMSE={test_metrics['rmse']:.4f}  Pearson={test_metrics['pearson']:+.4f}"
          f"  Spearman={test_metrics['spearman']:+.4f}")

    y_pred_train  = predict(model, graphs_train)
    train_metrics = regression_metrics(y_train, y_pred_train)
    print(f"  [train] RMSE={train_metrics['rmse']:.4f}  Pearson={train_metrics['pearson']:+.4f}")

    stereo_metrics = stereo_ordering_accuracy(model, sp)
    print(f"  Ordering accuracy: {stereo_metrics['ordering_acc']:.4f}"
          f"  ({stereo_metrics['n_correct']}/{stereo_metrics['n_pairs']})")

    stereo_trainval_metrics = stereo_ordering_accuracy(model, stereo_trainval)
    print(f"  Trainval ordering accuracy: {stereo_trainval_metrics['ordering_acc']:.4f}"
          f"  ({stereo_trainval_metrics['n_correct']}/{stereo_trainval_metrics['n_pairs']})")

    tag_metrics = eval_pair_metrics(model, tag_pairs, "SMILES_untagged", "SMILES_tagged")
    print(f"  Tag-pair delta Pearson: {tag_metrics['delta_pearson']:+.4f}")
    sub_metrics = eval_pair_metrics(model, sub_pairs, "SMILES_1", "SMILES_2")
    print(f"  Substitution-pair delta Pearson: {sub_metrics['delta_pearson']:+.4f}")

    return test_metrics, train_metrics, stereo_metrics, stereo_trainval_metrics, tag_metrics, sub_metrics, history


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

    print("Checking pretrained GIN weights …")
    download_weights()

    print("Loading stereopep dataset …")
    ds        = hf_load_dataset(HF_REPO, "StereoPep")
    sp              = hf_load_dataset(HF_REPO, "stereo_pairs")["stereo_pairs"]
    stereo_trainval = hf_load_dataset(HF_REPO, "stereo_pairs")["stereo_pairs_trainval"]
    tag_pairs       = hf_load_dataset(HF_REPO, "tag_pairs")["tag_pairs"]
    sub_pairs       = hf_load_dataset(HF_REPO, "substitution_pairs")["substitution_pairs"]

    graphs_train, _ = encode_smiles(ds["train"]["SMILES"], desc="Graphs train")
    graphs_val,   _ = encode_smiles(ds["val"]["SMILES"],   desc="Graphs val  ")
    graphs_test,  _ = encode_smiles(ds["test"]["SMILES"],  desc="Graphs test ")

    y_train = np.array(ds["train"]["B"], dtype=np.float32)
    y_val   = np.array(ds["val"]["B"],   dtype=np.float32)
    y_test  = np.array(ds["test"]["B"],  dtype=np.float32)
    print(f"  train={len(y_train)}  val={len(y_val)}  test={len(y_test)}")

    weights_path = WEIGHTS_DIR / f"results_pretrained_gin_seed{seed}.pt"
    print(f"\n── Seed {seed} ──")
    test_metrics, train_metrics, stereo_metrics, stereo_trainval_metrics, tag_metrics, sub_metrics, history = run_one_seed(
        seed, graphs_train, graphs_val, graphs_test, y_train, y_val, y_test, sp,
        stereo_trainval, tag_pairs, sub_pairs, weights_path=weights_path,
    )

    config = {
        "gin_layers": GIN_LAYERS, "gin_emb_dim": GIN_EMB_DIM,
        "head_hidden": HEAD_HIDDEN, "head_layers": HEAD_LAYERS,
        "dropout": DROPOUT, "lr_backbone": LR_BACKBONE, "lr_head": LR_HEAD,
        "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "lr_patience": LR_PATIENCE, "device": DEVICE,
    }
    training = {
        "epochs_run": history[-1]["epoch"],
        "best_val_loss": min(h["val_loss"] for h in history),
    }
    save_results(seed, test_metrics, train_metrics, stereo_metrics, stereo_trainval_metrics,
                 tag_metrics, sub_metrics, training, config, RESULTS_DIR, "results_pretrained_gin")


if __name__ == "__main__":
    main()
