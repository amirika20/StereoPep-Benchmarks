"""
EGNN-3D: Equivariant Graph Neural Network on RDKit 3D conformers.

Pipeline:
  1. Convert peptide SMILES → 3D conformers via RDKit ETKDGv3 (cheapest
     conformer algorithm; no force-field minimisation so it scales to long
     peptides).
  2. Build a radius-graph (cutoff 5 Å) over heavy-atom 3D coordinates.
  3. Train an E(n)-Equivariant GNN (Satorras et al., NeurIPS 2021) that
     jointly updates node features and 3D coordinates in each layer.
     Implementation uses PyTorch Geometric's MessagePassing — the update
     equations follow the original paper exactly:
       m_ij = φ_e(h_i, h_j, ||x_i-x_j||²)
       x_i  ← x_i + Σ_j (x_i-x_j)·φ_x(m_ij)   [equivariant coord update]
       h_i  ← φ_h(h_i, Σ_j m_ij)
  4. Predict B (retention time %, 0–100) with a global-mean-pool + MLP head.
  5. Evaluate on test split and on stereo/tag/substitution pair benchmarks.

Results → benchmarks/output/results_egnn_3d_seed{N}.json
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset as hf_load_dataset
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader as PlainDataLoader
from torch_geometric.data import Batch, Data
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.nn import radius_graph
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# ── config ─────────────────────────────────────────────────────────────────────
HF_REPO = "amirka20/peptag"

# 3D graph
RADIUS_CUTOFF     = 5.0    # Ångström — typical for small-molecule GNNs
MAX_NUM_NEIGHBORS = 32

# EGNN architecture
EGNN_LAYERS   = 4
EGNN_HIDDEN   = 128        # node feature dimension inside EGNN
ATOM_EMB_DIM  = 64         # embedding dim for atomic-number lookup
NUM_ATOM_TYPES = 120        # covers elements up to Z=118 + padding

# MLP head
HEAD_HIDDEN = 128
HEAD_LAYERS = 2
DROPOUT     = 0.1

# Training
LR           = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 32           # smaller than GIN because 3D graphs are denser
MAX_EPOCHS   = 50
PATIENCE     = 5            # overridden at runtime to 10 % of MAX_EPOCHS
LR_PATIENCE  = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR = Path(__file__).parent / "output"

_CHIRALITY_MAP = {
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED:     0,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:  1,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
    Chem.rdchem.ChiralType.CHI_OTHER:           2,
}


# ── 3D conformer generation ────────────────────────────────────────────────────

def _etkdg_params(seed: int = 42) -> AllChem.EmbedParameters:
    """ETKDGv3 with a fixed random seed — cheapest RDKit 3D algorithm."""
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    p.useSmallRingTorsions = True   # improves small-ring geometry at no cost
    p.useMacrocycleTorsions = True  # helps macrocyclic peptides
    p.enforceChirality = True
    return p


def smiles_to_3d_graph(smiles: str, conf_seed: int = 42) -> Data | None:
    """
    SMILES → PyG Data with 3D heavy-atom positions.

    Returns None if conformer embedding fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    # Add Hs so ETKDGv3 can produce physically reasonable geometry,
    # then strip them again so the graph stays compact.
    mol_h = Chem.AddHs(mol)
    params = _etkdg_params(seed=conf_seed)
    result = AllChem.EmbedMolecule(mol_h, params)

    if result == -1:
        # Second attempt with a different seed (ETKDGv3 can fail on unusual SMILES)
        params.randomSeed = conf_seed + 1000
        result = AllChem.EmbedMolecule(mol_h, params)

    if result == -1:
        return None

    mol_h = Chem.RemoveHs(mol_h)
    conf = mol_h.GetConformer()

    pos = torch.tensor(conf.GetPositions(), dtype=torch.float)   # (N, 3)

    # Node features: atomic number + chirality (same convention as GIN benchmarks)
    atom_feats = []
    for atom in mol_h.GetAtoms():
        atomic_idx  = min(atom.GetAtomicNum() - 1, NUM_ATOM_TYPES - 1)
        chiral_idx  = _CHIRALITY_MAP.get(atom.GetChiralTag(), 0)
        atom_feats.append([atomic_idx, chiral_idx])

    x = torch.tensor(atom_feats, dtype=torch.long)   # (N, 2)

    return Data(x=x, pos=pos)


def encode_smiles_3d(
    smiles_list: list[str],
    desc: str = "3D encoding",
    conf_seed: int = 42,
) -> tuple[list[Data], list[int]]:
    graphs, bad = [], []
    for i, smi in enumerate(tqdm(smiles_list, desc=desc, leave=False)):
        g = smiles_to_3d_graph(smi, conf_seed=conf_seed)
        if g is None:
            bad.append(i)
            # Placeholder so indices stay aligned
            graphs.append(Data(
                x=torch.zeros((1, 2), dtype=torch.long),
                pos=torch.zeros((1, 3), dtype=torch.float),
            ))
        else:
            graphs.append(g)
    return graphs, bad


# ── EGNN ───────────────────────────────────────────────────────────────────────
# Reference: "E(n) Equivariant Graph Neural Networks"
#             Satorras, Hoogeboom, Welling — NeurIPS 2021
#             https://arxiv.org/abs/2102.09844
#
# Each layer:
#   m_ij = φ_e(h_i, h_j, ||x_i - x_j||²)         edge message (invariant)
#   x_i  ← x_i + Σ_j (x_i - x_j) · φ_x(m_ij)    coord update (equivariant)
#   agg_i = Σ_j m_ij
#   h_i  ← φ_h(h_i, agg_i)                         node update (invariant)

class EGNNConv(MessagePassing):
    """
    Single E(n)-equivariant message-passing layer.

    Args:
        in_dim:     input node feature dimension
        hidden_dim: hidden dimension for the φ_e / φ_x / φ_h MLPs
        out_dim:    output node feature dimension
        act:        activation class (default: SiLU)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        act: type[nn.Module] = nn.SiLU,
    ):
        super().__init__(aggr="add")  # Σ aggregation
        self.in_dim  = in_dim
        self.out_dim = out_dim

        # φ_e: (h_i || h_j || dist²) → m_ij
        self.phi_e = nn.Sequential(
            nn.Linear(in_dim * 2 + 1, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
        )

        # φ_x: m_ij → scalar weight for coord update (output must stay bounded)
        self.phi_x = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),   # bounds the coordinate update
        )

        # φ_h: (h_i || agg_i) → h_i'
        self.phi_h = nn.Sequential(
            nn.Linear(in_dim + hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, out_dim),
        )

        # Residual connection when dims match
        self.residual = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

    def forward(
        self,
        h: torch.Tensor,      # (N, in_dim)   node features
        pos: torch.Tensor,    # (N, 3)        3D coordinates
        edge_index: torch.Tensor,  # (2, E)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # propagate collects messages and calls aggregate + update
        agg_msg, coord_update = self.propagate(
            edge_index, h=h, pos=pos, size=(h.size(0), h.size(0))
        )
        # node feature update
        h_new = self.phi_h(torch.cat([h, agg_msg], dim=-1)) + self.residual(h)
        # coordinate update (equivariant — linear combination of relative vectors)
        pos_new = pos + coord_update
        return h_new, pos_new

    def message(
        self,
        h_i: torch.Tensor,    # (E, in_dim)
        h_j: torch.Tensor,    # (E, in_dim)
        pos_i: torch.Tensor,  # (E, 3)
        pos_j: torch.Tensor,  # (E, 3)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        diff       = pos_i - pos_j                         # (E, 3)
        dist_sq    = (diff ** 2).sum(dim=-1, keepdim=True) # (E, 1)
        edge_input = torch.cat([h_i, h_j, dist_sq], dim=-1)
        m_ij       = self.phi_e(edge_input)                # (E, hidden)
        # equivariant weight for coord update
        coord_w    = self.phi_x(m_ij)                      # (E, 1)
        # return messages as a pair; PyG unpacks via aggregate
        return m_ij, diff * coord_w                        # (E, hidden), (E, 3)

    def aggregate(
        self,
        inputs: tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        dim_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        msg, coord_delta = inputs
        agg_msg   = super().aggregate(msg,         index, dim_size=dim_size)
        agg_coord = super().aggregate(coord_delta, index, dim_size=dim_size)
        return agg_msg, agg_coord

    def update(
        self,
        agg: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return agg  # pass through; node/coord update done in forward()


class EGNNEncoder(nn.Module):
    """
    Stacked EGNN layers.

    Input node features (atomic-num + chirality) are first embedded to
    `hidden_dim`; subsequent layers keep that dimension.
    """

    def __init__(
        self,
        num_layers: int = EGNN_LAYERS,
        atom_emb_dim: int = ATOM_EMB_DIM,
        hidden_dim: int = EGNN_HIDDEN,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.atom_emb1 = nn.Embedding(NUM_ATOM_TYPES, atom_emb_dim)
        self.atom_emb2 = nn.Embedding(3, atom_emb_dim)  # chirality (3 types)
        nn.init.xavier_uniform_(self.atom_emb1.weight)
        nn.init.xavier_uniform_(self.atom_emb2.weight)

        in_dim = atom_emb_dim  # atom_emb1 + atom_emb2 summed, so dim stays atom_emb_dim
        layers = []
        for i in range(num_layers):
            out_dim = hidden_dim
            layers.append(EGNNConv(in_dim, hidden_dim, out_dim))
            in_dim = out_dim
        self.layers  = nn.ModuleList(layers)
        self.dropout  = dropout
        self.out_dim  = hidden_dim

    def forward(
        self,
        x: torch.Tensor,       # (N, 2) — atomic_idx, chirality_idx
        pos: torch.Tensor,     # (N, 3)
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        h = self.atom_emb1(x[:, 0]) + self.atom_emb2(x[:, 1])
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h  # (N, hidden_dim) — equivariant coord updates discarded at end


class EGNNPredictor(nn.Module):
    """EGNN encoder + radius-graph construction + global-mean-pool + MLP head."""

    def __init__(
        self,
        num_layers: int  = EGNN_LAYERS,
        hidden_dim: int  = EGNN_HIDDEN,
        head_hidden: int = HEAD_HIDDEN,
        head_layers: int = HEAD_LAYERS,
        dropout: float   = DROPOUT,
        cutoff: float    = RADIUS_CUTOFF,
        max_neighbors: int = MAX_NUM_NEIGHBORS,
    ):
        super().__init__()
        self.encoder      = EGNNEncoder(num_layers, ATOM_EMB_DIM, hidden_dim, dropout)
        self.cutoff       = cutoff
        self.max_neighbors = max_neighbors

        head: list[nn.Module] = []
        in_dim = hidden_dim
        for _ in range(head_layers):
            head += [nn.Linear(in_dim, head_hidden),
                     nn.LayerNorm(head_hidden),
                     nn.GELU(),
                     nn.Dropout(dropout)]
            in_dim = head_hidden
        head.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*head)

    def forward(self, data: Batch) -> torch.Tensor:
        pos        = data.pos
        batch_idx  = data.batch

        # Build radius graph on-the-fly (equivariant: no precomputed bonds needed)
        edge_index = radius_graph(
            pos,
            r=self.cutoff,
            batch=batch_idx,
            max_num_neighbors=self.max_neighbors,
            loop=False,
        )

        node_emb  = self.encoder(data.x, pos, edge_index)
        graph_emb = global_mean_pool(node_emb, batch_idx)
        return self.head(graph_emb).squeeze(-1)


# ── dataset helpers ────────────────────────────────────────────────────────────

class GraphDataset3D(torch.utils.data.Dataset):
    def __init__(self, graphs: list[Data], targets: np.ndarray):
        self.graphs  = graphs
        self.targets = targets

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        g = self.graphs[idx].clone()
        g.y = torch.tensor(self.targets[idx], dtype=torch.float)
        return g


def collate_fn(batch: list[Data]) -> Batch:
    return Batch.from_data_list(batch)


# ── training ──────────────────────────────────────────────────────────────────

def train(
    model: EGNNPredictor,
    train_loader: PlainDataLoader,
    val_loader: PlainDataLoader,
) -> list[dict]:
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=LR_PATIENCE, factor=0.5, min_lr=1e-6
    )
    criterion = nn.MSELoss()
    history   = []
    best_val  = float("inf")
    best_state = None
    no_improve = 0

    epoch_bar = tqdm(range(1, MAX_EPOCHS + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"  e{epoch:3d} train", leave=False):
            batch = batch.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(batch), batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * batch.num_graphs
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"  e{epoch:3d} val  ", leave=False):
                batch = batch.to(DEVICE)
                val_loss += criterion(model(batch), batch.y).item() * batch.num_graphs
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
            print(f"\n  Early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def predict(model: EGNNPredictor, graphs: list[Data]) -> np.ndarray:
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
        "pearson":  float(stats.pearsonr(y_true, y_pred)[0]),
        "spearman": float(stats.spearmanr(y_true, y_pred)[0]),
        "kendall":  float(stats.kendalltau(y_true, y_pred)[0]),
    }


def pair_delta_metrics(true_delta: np.ndarray, pred_delta: np.ndarray) -> dict:
    rmse  = float(np.sqrt(mean_squared_error(true_delta, pred_delta)))
    mae   = float(mean_absolute_error(true_delta, pred_delta))
    pr, _ = stats.pearsonr(true_delta, pred_delta)
    sr, _ = stats.spearmanr(true_delta, pred_delta)
    mask  = np.sign(true_delta) != 0
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


def stereo_ordering_accuracy(model: EGNNPredictor, stereo_ds) -> dict:
    graphs_f, bad_f = encode_smiles_3d(stereo_ds["SMILES_f"], desc="Stereo D-Phe")
    graphs_F, bad_F = encode_smiles_3d(stereo_ds["SMILES_F"], desc="Stereo L-Phe")
    delta_B = np.array(stereo_ds["delta_B"], dtype=np.float64)

    pred_f     = predict(model, graphs_f)
    pred_F     = predict(model, graphs_F)
    pred_delta = pred_f - pred_F

    bad  = set(bad_f) | set(bad_F)
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
    model: EGNNPredictor,
    ds,
    smiles_col_a: str,
    smiles_col_b: str,
) -> dict:
    delta_B  = np.array(ds["delta_B"], dtype=np.float64)
    graphs_a, bad_a = encode_smiles_3d(list(ds[smiles_col_a]), desc=f"Pairs {smiles_col_a}")
    graphs_b, bad_b = encode_smiles_3d(list(ds[smiles_col_b]), desc=f"Pairs {smiles_col_b}")

    pred_a     = predict(model, graphs_a)
    pred_b     = predict(model, graphs_b)
    pred_delta = pred_a - pred_b

    bad  = set(bad_a) | set(bad_b)
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
    sub_pair_metrics: dict,
    training: dict,
    config: dict,
    output_dir: Path,
    stem: str,
) -> None:
    result = {
        "benchmark":               stem,
        "seed":                    seed,
        "timestamp":               datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config":                  config,
        "training":                training,
        "test_metrics":            test_metrics,
        "stereo_metrics":          stereo_metrics,
        "tag_pair_metrics":        tag_pair_metrics,
        "substitution_pair_metrics": sub_pair_metrics,
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
        GraphDataset3D(graphs_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = PlainDataLoader(
        GraphDataset3D(graphs_val, y_val),
        batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    model = EGNNPredictor().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    history      = train(model, train_loader, val_loader)
    y_pred_test  = predict(model, graphs_test)
    test_metrics = regression_metrics(y_test, y_pred_test)
    print(f"  RMSE={test_metrics['rmse']:.4f}  Pearson={test_metrics['pearson']:+.4f}"
          f"  Spearman={test_metrics['spearman']:+.4f}")

    stereo_metrics = stereo_ordering_accuracy(model, sp)
    print(f"  Stereo ordering acc: {stereo_metrics['ordering_acc']:.4f}"
          f"  ({stereo_metrics['n_correct']}/{stereo_metrics['n_pairs']})")

    tag_metrics = eval_pair_metrics(model, tag_pairs, "SMILES_untagged", "SMILES_tagged")
    print(f"  Tag-pair delta Pearson: {tag_metrics['delta_pearson']:+.4f}")

    sub_metrics = eval_pair_metrics(model, sub_pairs, "SMILES_1", "SMILES_2")
    print(f"  Substitution-pair delta Pearson: {sub_metrics['delta_pearson']:+.4f}")

    return test_metrics, stereo_metrics, tag_metrics, sub_metrics, history


def main() -> None:
    global MAX_EPOCHS, PATIENCE, LR_PATIENCE

    parser = argparse.ArgumentParser(
        description="EGNN benchmark with RDKit 3D conformers on PepTag"
    )
    parser.add_argument("--seed",   type=int, default=0)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args()
    seed = args.seed

    MAX_EPOCHS  = args.epochs
    PATIENCE    = max(1, int(0.10 * MAX_EPOCHS))

    print(f"Device: {DEVICE}  |  seed={seed}  |  max_epochs={MAX_EPOCHS}"
          f"  |  patience={PATIENCE}  |  cutoff={RADIUS_CUTOFF}Å")

    print("Loading PepTag dataset …")
    ds        = hf_load_dataset(HF_REPO, "peptag")
    sp        = hf_load_dataset(HF_REPO, "stereo_pairs")["stereo_pairs"]
    tag_pairs = hf_load_dataset(HF_REPO, "tag_pairs")["tag_pairs"]
    sub_pairs = hf_load_dataset(HF_REPO, "substitution_pairs")["substitution_pairs"]

    print("Generating 3D conformers (ETKDGv3) …")
    graphs_train, bad_train = encode_smiles_3d(ds["train"]["SMILES"], desc="Train 3D")
    graphs_val,   bad_val   = encode_smiles_3d(ds["val"]["SMILES"],   desc="Val 3D  ")
    graphs_test,  bad_test  = encode_smiles_3d(ds["test"]["SMILES"],  desc="Test 3D ")
    print(f"  Failed embeddings — train:{len(bad_train)}  val:{len(bad_val)}"
          f"  test:{len(bad_test)}")

    y_train = np.array(ds["train"]["B"], dtype=np.float32)
    y_val   = np.array(ds["val"]["B"],   dtype=np.float32)
    y_test  = np.array(ds["test"]["B"],  dtype=np.float32)
    print(f"  train={len(y_train)}  val={len(y_val)}  test={len(y_test)}")

    print(f"\n── Seed {seed} ──")
    test_m, stereo_m, tag_m, sub_m, history = run_one_seed(
        seed,
        graphs_train, graphs_val, graphs_test,
        y_train, y_val, y_test,
        sp, tag_pairs, sub_pairs,
    )

    config = {
        "egnn_layers":     EGNN_LAYERS,
        "egnn_hidden":     EGNN_HIDDEN,
        "atom_emb_dim":    ATOM_EMB_DIM,
        "radius_cutoff_A": RADIUS_CUTOFF,
        "max_neighbors":   MAX_NUM_NEIGHBORS,
        "head_hidden":     HEAD_HIDDEN,
        "head_layers":     HEAD_LAYERS,
        "dropout":         DROPOUT,
        "lr":              LR,
        "weight_decay":    WEIGHT_DECAY,
        "batch_size":      BATCH_SIZE,
        "max_epochs":      MAX_EPOCHS,
        "patience":        PATIENCE,
        "lr_patience":     LR_PATIENCE,
        "device":          DEVICE,
        "conformer_algo":  "ETKDGv3",
    }
    training = {
        "epochs_run":   history[-1]["epoch"],
        "best_val_loss": min(h["val_loss"] for h in history),
    }
    save_results(
        seed, test_m, stereo_m, tag_m, sub_m,
        training, config, RESULTS_DIR, "results_egnn_3d",
    )


if __name__ == "__main__":
    main()
