"""
EGNN-3D benchmark for the DIA dataset.

Same E(n)-equivariant GNN as benchmarks/egnn_3d.py (Satorras et al., NeurIPS
2021), adapted for local DIA retention-time data.
SMILES are generated from sequences using RDKit's MolFromFASTA; 3D conformers
are produced with RDKit ETKDGv3 and cached for reuse.

NOTE: Generating 3D conformers for ~146k DIA peptides takes considerable time
(10–60 min depending on hardware).  Cached conformers are stored in
benchmarks_dia/cache/conformers/ and reused across seeds.

No diastereomer/tag/substitution pair metrics.

Results written to benchmarks_dia/output/results_egnn_3d_dia_seed{N}.json.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader as PlainDataLoader
from torch_geometric.data import Batch, Data
from torch_geometric.nn import MessagePassing, global_mean_pool, radius_graph
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# ── config ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"

RADIUS_CUTOFF     = 5.0
MAX_NUM_NEIGHBORS = 32
EGNN_LAYERS   = 4
EGNN_HIDDEN   = 128
ATOM_EMB_DIM  = 64
NUM_ATOM_TYPES = 120
HEAD_HIDDEN = 128
HEAD_LAYERS = 2
DROPOUT     = 0.1
LR           = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 32
MAX_EPOCHS   = 50
PATIENCE     = 5
LR_PATIENCE  = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR  = Path(__file__).parent / "output"
WEIGHTS_DIR  = Path(__file__).parent / "weights"
CACHE_DIR    = Path(__file__).parent / "cache" / "conformers"

_CHIRALITY_MAP = {
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED:     0,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:  1,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
    Chem.rdchem.ChiralType.CHI_OTHER:           2,
}


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


# ── 3D conformer generation ────────────────────────────────────────────────────

def _etkdg_params(seed: int = 42) -> AllChem.EmbedParameters:
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    p.useSmallRingTorsions    = True
    p.useMacrocycleTorsions   = True
    p.enforceChirality        = True
    return p


def smiles_to_3d_graph(smiles: str, conf_seed: int = 42) -> Data | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    mol_h  = Chem.AddHs(mol)
    params = _etkdg_params(seed=conf_seed)
    result = AllChem.EmbedMolecule(mol_h, params)
    if result == -1:
        params.randomSeed = conf_seed + 1000
        result = AllChem.EmbedMolecule(mol_h, params)
    if result == -1:
        return None
    mol_h = Chem.RemoveHs(mol_h)
    conf  = mol_h.GetConformer()
    pos   = torch.tensor(conf.GetPositions(), dtype=torch.float)
    atom_feats = [
        [min(a.GetAtomicNum() - 1, NUM_ATOM_TYPES - 1),
         _CHIRALITY_MAP.get(a.GetChiralTag(), 0)]
        for a in mol_h.GetAtoms()
    ]
    return Data(x=torch.tensor(atom_feats, dtype=torch.long), pos=pos)


def encode_seqs_3d(seqs: list[str], desc: str = "3D encoding",
                   conf_seed: int = 42) -> tuple[list[Data], list[int]]:
    graphs, bad = [], []
    for i, seq in enumerate(tqdm(seqs, desc=desc, leave=False)):
        smiles = seq_to_smiles(seq)
        g = smiles_to_3d_graph(smiles, conf_seed) if smiles else None
        if g is None:
            bad.append(i)
            graphs.append(Data(x=torch.zeros((1, 2), dtype=torch.long),
                               pos=torch.zeros((1, 3), dtype=torch.float)))
        else:
            graphs.append(g)
    return graphs, bad


def load_or_encode_seqs_3d(seqs, name, conf_seed=42, cache_dir=None):
    if cache_dir is not None:
        cache_file = cache_dir / f"{name}_seed{conf_seed}.pt"
        if cache_file.exists():
            payload = torch.load(cache_file, weights_only=False)
            graphs, bad = payload["graphs"], payload["bad"]
            print(f"  Loaded {len(graphs)} conformers from cache: {cache_file.name}")
            return graphs, bad

    graphs, bad = encode_seqs_3d(seqs, desc=name, conf_seed=conf_seed)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{name}_seed{conf_seed}.pt"
        torch.save({"graphs": graphs, "bad": bad}, cache_file)
        print(f"  Conformer cache saved: {cache_file.name}")

    return graphs, bad


# ── EGNN ───────────────────────────────────────────────────────────────────────

class EGNNConv(MessagePassing):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128):
        super().__init__(aggr="add")
        self.phi_e = nn.Sequential(
            nn.Linear(2 * in_dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.phi_x = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1),
        )
        self.phi_h = nn.Sequential(
            nn.Linear(in_dim + hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, h: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor):
        return self.propagate(edge_index, h=h, pos=pos, size=None)

    def message(self, h_i, h_j, pos_i, pos_j):
        diff = pos_i - pos_j
        dist_sq = (diff ** 2).sum(dim=-1, keepdim=True)
        m_ij = self.phi_e(torch.cat([h_i, h_j, dist_sq], dim=-1))
        delta_x = diff * self.phi_x(m_ij)
        return m_ij, delta_x

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        m_ij, delta_x = inputs
        agg_m     = super().aggregate(m_ij,    index, ptr=ptr, dim_size=dim_size)
        agg_delta = super().aggregate(delta_x, index, ptr=ptr, dim_size=dim_size)
        return agg_m, agg_delta

    def update(self, agg_out, h, pos):
        agg_m, agg_delta = agg_out
        new_pos = pos + agg_delta
        new_h   = self.phi_h(torch.cat([h, agg_m], dim=-1))
        return new_h, new_pos


class EGNN(nn.Module):
    def __init__(self, n_layers=EGNN_LAYERS, hidden=EGNN_HIDDEN,
                 atom_emb_dim=ATOM_EMB_DIM, num_atom_types=NUM_ATOM_TYPES,
                 head_hidden=HEAD_HIDDEN, head_layers=HEAD_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.atom_embed = nn.Embedding(num_atom_types, atom_emb_dim)
        self.chiral_embed = nn.Embedding(3, atom_emb_dim)
        in_dim = atom_emb_dim * 2

        self.layers = nn.ModuleList()
        dim = in_dim
        for _ in range(n_layers):
            self.layers.append(EGNNConv(dim, hidden, hidden=hidden))
            dim = hidden

        mlp_layers: list[nn.Module] = []
        for _ in range(head_layers):
            mlp_layers += [nn.Linear(dim, head_hidden), nn.GELU(), nn.Dropout(dropout)]
            dim = head_hidden
        mlp_layers.append(nn.Linear(dim, 1))
        self.head = nn.Sequential(*mlp_layers)

    def forward(self, data: Batch) -> torch.Tensor:
        x    = data.x
        pos  = data.pos
        h    = self.atom_embed(x[:, 0]) + self.chiral_embed(x[:, 1])

        for layer in self.layers:
            edge_index = radius_graph(pos, r=RADIUS_CUTOFF, batch=data.batch,
                                      max_num_neighbors=MAX_NUM_NEIGHBORS)
            out = layer(h, pos, edge_index)
            if isinstance(out, tuple):
                h, pos = out
            else:
                h = out

        g = global_mean_pool(h, data.batch)
        return self.head(g).squeeze(-1)


# ── dataset helpers ───────────────────────────────────────────────────────────

class Graph3DDataset(torch.utils.data.Dataset):
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

def train_model(model, train_loader, val_loader) -> list[dict]:
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=LR_PATIENCE, factor=0.5, min_lr=1e-6)
    criterion = nn.MSELoss()
    history   = []
    best_val  = float("inf")
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
            loss.backward(); opt.step()
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

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        epoch_bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}",
                              best=f"{best_val:.4f}", patience=no_improve)
        if no_improve >= PATIENCE:
            print(f"\n  Early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def predict_graphs(model, graphs) -> np.ndarray:
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
        "benchmark": stem, "dataset": "dia", "seed": seed,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "config": config, "training": training,
        "test_metrics": test_metrics, "train_metrics": train_metrics,
    }
    out = output_dir / f"{stem}_seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, cls=_NpEncoder))
    print(f"\nResults saved to {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def run_one_seed(seed, graphs_train, graphs_val, graphs_test,
                 y_train, y_val, y_test, bad_train, bad_test,
                 weights_path=None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Mask out failed conformers
    good_train = np.ones(len(y_train), dtype=bool)
    good_test  = np.ones(len(y_test),  dtype=bool)
    for i in bad_train: good_train[i] = False
    for i in bad_test:  good_test[i]  = False

    g_tr = [g for g, ok in zip(graphs_train, good_train) if ok]
    y_tr = y_train[good_train]
    g_te = [g for g, ok in zip(graphs_test, good_test) if ok]
    y_te = y_test[good_test]

    train_loader = PlainDataLoader(
        Graph3DDataset(g_tr, y_tr), batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn, num_workers=0,
    )
    val_loader = PlainDataLoader(
        Graph3DDataset(graphs_val, y_val), batch_size=BATCH_SIZE,
        shuffle=False, collate_fn=collate_fn, num_workers=0,
    )

    model = EGNN().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  EGNN | {n_params:,} params")

    if weights_path is not None and weights_path.exists():
        print(f"  [weights] Loading from {weights_path} — skipping training")
        ckpt    = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        history = ckpt["history"]
    else:
        history = train_model(model, train_loader, val_loader)
        if weights_path is not None:
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "history": history}, weights_path)
            print(f"  [weights] Saved to {weights_path}")

    y_pred_test  = predict_graphs(model, g_te)
    test_metrics = regression_metrics(y_te, y_pred_test)
    print(f"  RMSE={test_metrics['rmse']:.4f}  Pearson={test_metrics['pearson']:+.4f}"
          f"  Spearman={test_metrics['spearman']:+.4f}")

    y_pred_train  = predict_graphs(model, g_tr)
    train_metrics = regression_metrics(y_tr, y_pred_train)
    print(f"  [train] RMSE={train_metrics['rmse']:.4f}  Pearson={train_metrics['pearson']:+.4f}")

    return test_metrics, train_metrics, history


def main() -> None:
    global MAX_EPOCHS, PATIENCE, LR_PATIENCE
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",      type=int, default=0)
    parser.add_argument("--epochs",    type=int, default=MAX_EPOCHS)
    parser.add_argument("--conf-seed", type=int, default=42,
                        help="Random seed for RDKit conformer generation (default: 42)")
    parser.add_argument("--no-cache",  action="store_true",
                        help="Disable conformer caching")
    args = parser.parse_args()
    seed      = args.seed
    conf_seed = args.conf_seed

    MAX_EPOCHS  = args.epochs
    PATIENCE    = max(1, int(0.10 * MAX_EPOCHS))
    cache_dir   = None if args.no_cache else CACHE_DIR

    print(f"Device: {DEVICE}  |  seed={seed}  |  max_epochs={MAX_EPOCHS}  |  patience={PATIENCE}")
    t0 = time.time()

    print("[data] Loading DIA dataset …")
    train_seqs, y_train, val_seqs, y_val, test_seqs, y_test = load_dia_data()
    print(f"  train={len(train_seqs)}  val={len(val_seqs)}  test={len(test_seqs)}")

    print("[3D] Generating/loading 3D conformers (this may take a while for large datasets) …")
    graphs_train, bad_tr = load_or_encode_seqs_3d(train_seqs, "dia_train", conf_seed, cache_dir)
    graphs_val,   _      = load_or_encode_seqs_3d(val_seqs,   "dia_val",   conf_seed, cache_dir)
    graphs_test,  bad_te = load_or_encode_seqs_3d(test_seqs,  "dia_test",  conf_seed, cache_dir)

    if bad_tr:
        print(f"  WARNING: {len(bad_tr)} train conformers failed")
    if bad_te:
        print(f"  WARNING: {len(bad_te)} test conformers failed")

    weights_path = WEIGHTS_DIR / f"results_egnn_3d_dia_seed{seed}.pt"
    print(f"\n── Seed {seed} ──")
    test_metrics, train_metrics, history = run_one_seed(
        seed, graphs_train, graphs_val, graphs_test,
        y_train, y_val, y_test, bad_tr, bad_te,
        weights_path=weights_path,
    )

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    config = {
        "radius_cutoff": RADIUS_CUTOFF, "max_num_neighbors": MAX_NUM_NEIGHBORS,
        "egnn_layers": EGNN_LAYERS, "egnn_hidden": EGNN_HIDDEN,
        "atom_emb_dim": ATOM_EMB_DIM, "head_hidden": HEAD_HIDDEN,
        "head_layers": HEAD_LAYERS, "dropout": DROPOUT,
        "lr": LR, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "lr_patience": LR_PATIENCE, "conf_seed": conf_seed, "device": DEVICE,
    }
    training = {"epochs_run": history[-1]["epoch"],
                "best_val_loss": min(h["val_loss"] for h in history)}
    save_results(seed, test_metrics, train_metrics, training, config,
                 RESULTS_DIR, "results_egnn_3d_dia")


if __name__ == "__main__":
    main()
