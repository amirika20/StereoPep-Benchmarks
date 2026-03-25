"""
GIN trained from scratch on the PepTag dataset.

Uses the same Graph Isomorphism Network (GIN) architecture as pretrained_gin.py
(Hu et al., ICLR 2020) but initialises all weights randomly and trains end-to-end
directly on the PepTag retention-time data — no external checkpoint required.

Pipeline:
  1. Convert full peptide SMILES → molecular graphs (same atom/bond featurisation
     as pretrained_gin.py so results are directly comparable).
  2. Train the GNN + MLP head from scratch to predict B (retention time, 0-100).
  3. Evaluate on test split with regression metrics.
  4. Evaluate stereochemistry ordering accuracy on stereo_pairs (D-Phe vs L-Phe).

Results are written to benchmarks/output/results_gin_scratch_seed{N}.json.
"""

from __future__ import annotations

import argparse
import json
import time
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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops
from torch.utils.data import DataLoader as PlainDataLoader

# ── config ────────────────────────────────────────────────────────────────────
HF_REPO = "amirka20/peptag"

# GIN architecture (kept identical to pretrained_gin.py for fair comparison)
NUM_ATOM_TYPES     = 120
NUM_CHIRALITY_TAGS = 3
NUM_BOND_TYPES     = 6
NUM_BOND_DIRS      = 3
GIN_LAYERS         = 5
GIN_EMB_DIM        = 300

# Training from scratch — single LR (no backbone/head split needed)
HEAD_HIDDEN   = 256
HEAD_LAYERS   = 2
DROPOUT       = 0.1
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
BATCH_SIZE    = 64
MAX_EPOCHS    = 50    # more epochs since starting from random init
PATIENCE      = 5     # overridden at runtime to 0.1 * MAX_EPOCHS
LR_PATIENCE   = 10
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR = Path(__file__).parent / "output"


# ── molecular graph featurisation ─────────────────────────────────────────────
# Atom/bond feature encoding matches snap-stanford/pretrain-gnns/chem/loader.py
# so results are directly comparable with pretrained_gin.py.

_BOND_TYPE_MAP = {
    Chem.rdchem.BondType.SINGLE:   0,
    Chem.rdchem.BondType.DOUBLE:   1,
    Chem.rdchem.BondType.TRIPLE:   2,
    Chem.rdchem.BondType.AROMATIC: 3,
}
_BOND_DIR_MAP = {
    Chem.rdchem.BondDir.NONE:         0,
    Chem.rdchem.BondDir.ENDUPRIGHT:   1,
    Chem.rdchem.BondDir.ENDDOWNRIGHT: 2,
}
_CHIRALITY_MAP = {
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED:     0,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:  1,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
    Chem.rdchem.ChiralType.CHI_OTHER:           2,
}


def smiles_to_graph(smiles: str) -> Data | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    atom_feats = []
    for atom in mol.GetAtoms():
        atomic_num_idx = min(atom.GetAtomicNum() - 1, 117)
        chirality_idx  = _CHIRALITY_MAP.get(atom.GetChiralTag(), 0)
        atom_feats.append([atomic_num_idx, chirality_idx])

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

    if edge_index.numel() == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 2), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ── GIN model ─────────────────────────────────────────────────────────────────

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
    graphs, bad = [], []
    for i, smi in enumerate(tqdm(smiles_list, desc=desc, leave=False)):
        g = smiles_to_graph(smi)
        if g is None:
            bad.append(i)
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
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
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
        "rmse":     float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":      float(mean_absolute_error(y_true, y_pred)),
        "pearson":  stats.pearsonr(y_true, y_pred)[0],
        "spearman": stats.spearmanr(y_true, y_pred)[0],
        "kendall":  stats.kendalltau(y_true, y_pred)[0],
    }


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


def stereo_ordering_accuracy(model: GINPredictor, stereo_ds) -> dict:
    graphs_f, bad_f = encode_smiles(stereo_ds["SMILES_f"], desc="Stereo D-Phe")
    graphs_F, bad_F = encode_smiles(stereo_ds["SMILES_F"], desc="Stereo L-Phe")
    B_f     = np.array(stereo_ds["B_f"],     dtype=np.float64)
    B_F     = np.array(stereo_ds["B_F"],     dtype=np.float64)
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

    pr = stats.pearsonr(delta_B[mask], pred_delta[mask])[0]
    sr = stats.spearmanr(delta_B[mask], pred_delta[mask])[0]

    return {
        "n_pairs":         total,
        "n_correct":       correct,
        "ordering_acc":    correct / total if total > 0 else float("nan"),
        "delta_pearson":   float(pr),
        "delta_spearman":  float(sr),
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
    graphs_train: list,
    graphs_val: list,
    graphs_test: list,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    sp,
    tag_pairs,
    sub_pairs,
) -> tuple[dict, dict, dict, dict, list[dict]]:
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

    model = GINPredictor().to(DEVICE)  # randomly initialised — no pretrained weights

    history      = train(model, train_loader, val_loader)
    y_pred_test  = predict(model, graphs_test)
    test_metrics = regression_metrics(y_test, y_pred_test)
    print(f"  RMSE={test_metrics['rmse']:.4f}  Pearson={test_metrics['pearson']:+.4f}"
          f"  Spearman={test_metrics['spearman']:+.4f}")

    stereo_metrics = stereo_ordering_accuracy(model, sp)
    print(f"  Ordering accuracy: {stereo_metrics['ordering_acc']:.4f}"
          f"  ({stereo_metrics['n_correct']}/{stereo_metrics['n_pairs']})")

    tag_metrics = eval_pair_metrics(model, tag_pairs, "SMILES_tag", "SMILES_notag")
    print(f"  Tag-pair delta Pearson: {tag_metrics['delta_pearson']:+.4f}")
    sub_metrics = eval_pair_metrics(model, sub_pairs, "SMILES_1", "SMILES_2")
    print(f"  Substitution-pair delta Pearson: {sub_metrics['delta_pearson']:+.4f}")

    return test_metrics, stereo_metrics, tag_metrics, sub_metrics, history


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

    print("Loading peptag dataset …")
    ds        = hf_load_dataset(HF_REPO, "peptag")
    sp        = hf_load_dataset(HF_REPO, "stereo_pairs")["stereo_pairs"]
    tag_pairs = hf_load_dataset(HF_REPO, "tag_pairs")["tag_pairs"]
    sub_pairs = hf_load_dataset(HF_REPO, "substitution_pairs")["substitution_pairs"]

    graphs_train, _ = encode_smiles(ds["train"]["SMILES"], desc="Graphs train")
    graphs_val,   _ = encode_smiles(ds["val"]["SMILES"],   desc="Graphs val  ")
    graphs_test,  _ = encode_smiles(ds["test"]["SMILES"],  desc="Graphs test ")

    y_train = np.array(ds["train"]["B"], dtype=np.float32)
    y_val   = np.array(ds["val"]["B"],   dtype=np.float32)
    y_test  = np.array(ds["test"]["B"],  dtype=np.float32)
    print(f"  train={len(y_train)}  val={len(y_val)}  test={len(y_test)}")

    print(f"\n── Seed {seed} ──")
    test_metrics, stereo_metrics, tag_metrics, sub_metrics, history = run_one_seed(
        seed, graphs_train, graphs_val, graphs_test, y_train, y_val, y_test, sp,
        tag_pairs, sub_pairs,
    )

    config = {
        "gin_layers": GIN_LAYERS, "gin_emb_dim": GIN_EMB_DIM,
        "head_hidden": HEAD_HIDDEN, "head_layers": HEAD_LAYERS,
        "dropout": DROPOUT, "lr": LR,
        "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "lr_patience": LR_PATIENCE, "device": DEVICE,
        "pretrained": False,
    }
    training = {
        "epochs_run": history[-1]["epoch"],
        "best_val_loss": min(h["val_loss"] for h in history),
    }
    save_results(seed, test_metrics, stereo_metrics, tag_metrics, sub_metrics,
                 training, config, RESULTS_DIR, "results_gin_scratch")


if __name__ == "__main__":
    main()
