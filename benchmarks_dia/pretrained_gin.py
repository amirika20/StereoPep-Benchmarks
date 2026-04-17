"""
Pretrained GIN benchmark for the DIA dataset.

Same architecture as benchmarks/pretrained_gin.py (Hu et al., ICLR 2020),
adapted for local DIA retention-time data.  SMILES are generated from
standard AA sequences using RDKit's MolFromFASTA.
No diastereomer/tag/substitution pair metrics.

Results written to benchmarks_dia/output/results_pretrained_gin_dia_seed{N}.json.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops
from torch.utils.data import DataLoader as PlainDataLoader
from tqdm import tqdm

# ── config ────────────────────────────────────────────────────────────────────
DATA_DIR       = Path(__file__).parent.parent / "data"

NUM_ATOM_TYPES     = 120
NUM_CHIRALITY_TAGS = 3
NUM_BOND_TYPES     = 6
NUM_BOND_DIRS      = 3
GIN_LAYERS         = 5
GIN_EMB_DIM        = 300

PRETRAINED_URL = (
    "https://raw.githubusercontent.com/snap-stanford/pretrain-gnns"
    "/master/chem/model_gin/supervised_contextpred.pth"
)
PRETRAINED_DIR  = Path(__file__).parent / "pretrained_weights"
WEIGHTS_FILE    = PRETRAINED_DIR / "gin_supervised_contextpred.pth"

HEAD_HIDDEN   = 256
HEAD_LAYERS   = 2
DROPOUT       = 0.1
LR_BACKBONE   = 1e-4
LR_HEAD       = 1e-3
WEIGHT_DECAY  = 1e-4
BATCH_SIZE    = 64
MAX_EPOCHS    = 20
PATIENCE      = 5
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
    print("  Downloading pretrained GIN from snap-stanford …")
    try:
        urllib.request.urlretrieve(PRETRAINED_URL, WEIGHTS_FILE)
        print(f"  Saved to {WEIGHTS_FILE}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download pretrained weights.\nError: {e}\n"
            f"Download manually and place at {WEIGHTS_FILE}"
        )


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


# ── SMILES generation ─────────────────────────────────────────────────────────

def seq_to_smiles(seq: str) -> str | None:
    mol = Chem.MolFromFASTA(seq)
    return Chem.MolToSmiles(mol) if mol is not None else None


# ── molecular graph featurisation ─────────────────────────────────────────────

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
        atom_feats.append([
            min(atom.GetAtomicNum() - 1, 117),
            _CHIRALITY_MAP.get(atom.GetChiralTag(), 0),
        ])
    src, dst, edge_feats = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = _BOND_TYPE_MAP.get(bond.GetBondType(), 3)
        bd = _BOND_DIR_MAP.get(bond.GetBondDir(), 0)
        src += [i, j]; dst += [j, i]
        edge_feats += [[bt, bd], [bt, bd]]
    x          = torch.tensor(atom_feats, dtype=torch.long)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr  = torch.tensor(edge_feats, dtype=torch.long)
    if edge_index.numel() == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 2), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def encode_seqs(seqs: list[str], desc: str = "Encoding") -> tuple[list[Data], list[int]]:
    graphs, bad = [], []
    for i, seq in enumerate(tqdm(seqs, desc=desc, leave=False)):
        smiles = seq_to_smiles(seq)
        g = smiles_to_graph(smiles) if smiles else None
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


# ── GIN model ─────────────────────────────────────────────────────────────────

class GINConv(MessagePassing):
    def __init__(self, emb_dim: int):
        super().__init__(aggr="add")
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim), nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim),
        )
        self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPES, emb_dim)
        self.edge_embedding2 = nn.Embedding(NUM_BOND_DIRS,  emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight)
        nn.init.xavier_uniform_(self.edge_embedding2.weight)

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros(x.size(0), 2, dtype=torch.long, device=x.device)
        self_loop_attr[:, 0] = 4
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        edge_emb = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        return self.propagate(edge_index, x=x, edge_attr=edge_emb)

    def message(self, x_j, edge_attr): return x_j + edge_attr
    def update(self, aggr_out):        return self.mlp(aggr_out)


class GINEncoder(nn.Module):
    def __init__(self, num_layers=GIN_LAYERS, emb_dim=GIN_EMB_DIM, drop_ratio=0.0):
        super().__init__()
        self.x_embedding1  = nn.Embedding(NUM_ATOM_TYPES,     emb_dim)
        self.x_embedding2  = nn.Embedding(NUM_CHIRALITY_TAGS, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight)
        nn.init.xavier_uniform_(self.x_embedding2.weight)
        self.gnns        = nn.ModuleList([GINConv(emb_dim) for _ in range(num_layers)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layers)])
        self.num_layers  = num_layers
        self.drop_ratio  = drop_ratio

    def forward(self, x, edge_index, edge_attr):
        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        for i, (gnn, bn) in enumerate(zip(self.gnns, self.batch_norms)):
            h = gnn(h, edge_index, edge_attr)
            h = bn(h)
            if i < self.num_layers - 1:
                h = F.relu(h)
            h = F.dropout(h, p=self.drop_ratio, training=self.training)
        return h


class GINPredictor(nn.Module):
    def __init__(self, head_hidden=HEAD_HIDDEN, head_layers=HEAD_LAYERS, dropout=DROPOUT):
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

    def forward(self, data: Batch):
        node_emb  = self.encoder(data.x, data.edge_index, data.edge_attr)
        graph_emb = global_mean_pool(node_emb, data.batch)
        return self.head(graph_emb).squeeze(-1)

    def load_pretrained_encoder(self, path: Path) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.encoder.load_state_dict(state, strict=True)


# ── dataset helpers ───────────────────────────────────────────────────────────

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, targets):
        self.graphs  = graphs
        self.targets = targets

    def __len__(self):  return len(self.graphs)

    def __getitem__(self, idx):
        g   = self.graphs[idx].clone()
        g.y = torch.tensor(self.targets[idx], dtype=torch.float)
        return g


def collate_fn(batch): return Batch.from_data_list(batch)


# ── training ──────────────────────────────────────────────────────────────────

def train(model, train_loader, val_loader) -> list[dict]:
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
    history   = []
    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0

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
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1

        epoch_bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}",
                              best=f"{best_val_loss:.4f}", patience=no_improve)
        if no_improve >= PATIENCE:
            print(f"\n  Early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def predict(model, graphs) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(graphs), BATCH_SIZE):
            batch = Batch.from_data_list(graphs[i:i+BATCH_SIZE]).to(DEVICE)
            preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds)


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


# ── reporting ─────────────────────────────────────────────────────────────────

class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def save_results(seed, test_metrics, train_metrics, training, config, output_dir, stem):
    result = {
        "benchmark": stem,
        "dataset": "dia",
        "seed": seed,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "config": config,
        "training": training,
        "test_metrics": test_metrics,
        "train_metrics": train_metrics,
    }
    out = output_dir / f"{stem}_seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, cls=_NpEncoder))
    print(f"\nResults saved to {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def run_one_seed(seed, graphs_train, graphs_val, graphs_test,
                 y_train, y_val, y_test, weights_path=None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = PlainDataLoader(
        GraphDataset(graphs_train, y_train), batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn, num_workers=0,
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

    return test_metrics, train_metrics, history


def main() -> None:
    global MAX_EPOCHS, PATIENCE, LR_PATIENCE
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",   type=int, default=0)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args()
    seed = args.seed

    MAX_EPOCHS  = args.epochs
    PATIENCE    = max(1, int(0.10 * MAX_EPOCHS))

    print(f"Device: {DEVICE}  |  seed={seed}  |  max_epochs={MAX_EPOCHS}  |  patience={PATIENCE}")
    t0 = time.time()

    print("Checking pretrained GIN weights …")
    download_weights()

    print("[data] Loading DIA dataset …")
    train_seqs, y_train, val_seqs, y_val, test_seqs, y_test = load_dia_data()
    print(f"  train={len(train_seqs)}  val={len(val_seqs)}  test={len(test_seqs)}")

    print("[graphs] Building molecular graphs from sequences …")
    graphs_train, _ = encode_seqs(train_seqs, "Graphs train")
    graphs_val,   _ = encode_seqs(val_seqs,   "Graphs val  ")
    graphs_test,  _ = encode_seqs(test_seqs,  "Graphs test ")

    weights_path = WEIGHTS_DIR / f"results_pretrained_gin_dia_seed{seed}.pt"
    print(f"\n── Seed {seed} ──")
    test_metrics, train_metrics, history = run_one_seed(
        seed, graphs_train, graphs_val, graphs_test,
        y_train, y_val, y_test, weights_path=weights_path,
    )

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    config = {
        "gin_layers": GIN_LAYERS, "gin_emb_dim": GIN_EMB_DIM,
        "head_hidden": HEAD_HIDDEN, "head_layers": HEAD_LAYERS,
        "dropout": DROPOUT, "lr_backbone": LR_BACKBONE, "lr_head": LR_HEAD,
        "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "lr_patience": LR_PATIENCE, "device": DEVICE,
    }
    training = {
        "epochs_run": history[-1]["epoch"],
        "best_val_loss": min(h["val_loss"] for h in history),
    }
    save_results(seed, test_metrics, train_metrics, training, config,
                 RESULTS_DIR, "results_pretrained_gin_dia")


if __name__ == "__main__":
    main()
