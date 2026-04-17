"""
PepMNet benchmark for the DIA dataset.

Same hierarchical GNN as benchmarks/pepmnet.py (Garzon-Otero et al., 2024),
adapted for local DIA retention-time data.
No diastereomer/tag/substitution pair metrics.

Extra dependencies: pip install biopython

Results written to benchmarks_dia/output/results_pepmnet_dia_seed{N}.json.
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
from rdkit.Chem import Crippen, Descriptors
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader
from torch_geometric.nn import ARMAConv, NNConv, global_add_pool
from torch_geometric.utils import scatter
from tqdm import tqdm

try:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
except ImportError:
    raise ImportError("BioPython is required: pip install biopython")

warnings.filterwarnings("ignore", category=UserWarning)

# ── config ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"

HIDDEN_NN_1  = 500
HIDDEN_NN_2  = 250
HIDDEN_NN_3  = 100
HIDDEN_GAT_1 = 15
HIDDEN_FCN_1 = 100
HIDDEN_FCN_2 = 50
HIDDEN_FCN_3 = 10
DROPOUT      = 0.0

LR           = 1e-3
WEIGHT_DECAY = 0.0
BATCH_SIZE   = 16
MAX_EPOCHS   = 100
PATIENCE     = 10
LR_PATIENCE  = 10
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR = Path(__file__).parent / "output"
WEIGHTS_DIR = Path(__file__).parent / "weights"

_AA_UPPER_MAP: dict[str, str] = {}   # No non-standard residues in DIA data

_SAMPLE_SEQS = [
    "WNSLKIDNLDA", "RIPVMMNWYW", "PQPPVEEEDEHFDDTVVCLDTYNCDLHFK",
    "SCRYSQRPSFYRWELYFNGRMWCP",
    "MNQKHSSDFVVIKAVEDGVNVIGLTRGTDTKFHHSEKLDKGEVIIAQFTEHTSAIKVRGEALIQTAYGEMKSEKK",
    "VLSIVACSSGCGSGKTAASCVATCGNKCFTNVGSLC",
    "MKHFLTYLSTAPVLAAIWMTITAGILIEFNRFYPDLLFHPL",
    "GILGKLWEGFKSIV", "FNQWTTWCYHHMVPYCDYCHFKR", "GLLALLGELAEHLGSKI",
    "GCKKYRRFRWKFKGKLWLWG", "EIEKFDKSKLK",
    "GGTIFDCGETCFLGTCYTPGCSCGNYGFCYGTN",
    "MKVLVLITLAVLGAMFVWTSAAELEERGSDQRDSPAWVKSMERIFQSEERACREWLGGCSKDADCCAHLECRKKWPYHCVWDWTVRK",
    "LKMLGMLFHNIRNILKTV", "WRPGRWWRPGRWWRPGRWWRPGRW",
    "GLWQIFSSKEEGKDNSQQKSKGDQAKEL", "RWMAWPTHKERNWYMTW", "HRILMRARQMMT",
    "RRSRFGRFFKKVRKQLGRVLRHSRITVGGRMRF",
    "ACDEFGHIKLMNPQRSTVWY", "HRKIFLWAMPCNVGSQYDET",
]
_STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")


def _sequence_to_helm(sequence: str, polymer_id: str = "PEPTIDE1") -> str:
    seq = sequence.replace("(ac)", "[ac].").rstrip(".")
    return f"{polymer_id}{{{'.'.join(seq)}}}$$$$"


def _safe_sequence(seq: str) -> str:
    return "".join(_AA_UPPER_MAP.get(c, c) for c in seq)


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


# ── feature extraction ─────────────────────────────────────────────────────────

def build_feature_dicts():
    atomic_num_raw, aromatic_raw, degree_raw, h_raw, hybrid_raw, ival_raw = \
        [], [], [], [], [], []
    btype_raw, ring_raw, conj_raw, barom_raw, vali_raw, valf_raw = \
        [], [], [], [], [], []

    for seq in _SAMPLE_SEQS:
        mol = Chem.MolFromHELM(_sequence_to_helm(seq))
        if mol is None:
            continue
        atomic_num_raw.extend(atom.GetAtomicNum()        for atom in mol.GetAtoms())
        aromatic_raw.extend(int(atom.GetIsAromatic())    for atom in mol.GetAtoms())
        degree_raw.extend(atom.GetDegree()               for atom in mol.GetAtoms())
        h_raw.extend(atom.GetTotalNumHs()                for atom in mol.GetAtoms())
        hybrid_raw.extend(atom.GetHybridization().real   for atom in mol.GetAtoms())
        ival_raw.extend(atom.GetImplicitValence()        for atom in mol.GetAtoms())
        for bond in mol.GetBonds():
            btype_raw.append(bond.GetBondTypeAsDouble())
            ring_raw.append(int(bond.IsInRing()))
            conj_raw.append(int(bond.GetIsConjugated()))
            barom_raw.append(int(bond.GetIsAromatic()))
            vali_raw.append(int(bond.GetValenceContrib(bond.GetBeginAtom())))
            valf_raw.append(int(bond.GetValenceContrib(bond.GetEndAtom())))

    def _fit_ohe(values):
        enc = OneHotEncoder(sparse_output=False)
        enc.fit(np.array(sorted(set(values))).reshape(-1, 1))
        return enc

    enc_atomic  = _fit_ohe(atomic_num_raw)
    enc_arom    = _fit_ohe(aromatic_raw)
    enc_degree  = _fit_ohe(degree_raw)
    enc_h       = _fit_ohe(h_raw)
    enc_hybrid  = _fit_ohe(hybrid_raw)
    enc_ival    = _fit_ohe(ival_raw)
    enc_btype   = _fit_ohe(btype_raw)
    enc_ring    = _fit_ohe(ring_raw)
    enc_conj    = _fit_ohe(conj_raw)
    enc_barom   = _fit_ohe(barom_raw)
    enc_vali    = _fit_ohe(vali_raw)
    enc_valf    = _fit_ohe(valf_raw)

    node_ft_dict: dict[str, np.ndarray] = {}
    for a, ar, d, h, hy, iv in zip(
        atomic_num_raw, aromatic_raw, degree_raw, h_raw, hybrid_raw, ival_raw
    ):
        key = f"{a}_{ar}_{d}_{h}_{hy}_{iv}"
        if key not in node_ft_dict:
            node_ft_dict[key] = np.concatenate([
                enc_atomic.transform([[a]])[0], enc_arom.transform([[ar]])[0],
                enc_degree.transform([[d]])[0], enc_h.transform([[h]])[0],
                enc_hybrid.transform([[hy]])[0], enc_ival.transform([[iv]])[0],
            ])

    edge_ft_dict: dict[str, np.ndarray] = {}
    for bt, rg, cj, ba, vi, vf in zip(
        btype_raw, ring_raw, conj_raw, barom_raw, vali_raw, valf_raw
    ):
        key = f"{bt:.1f}_{rg:.1f}_{cj:.1f}_{ba:.1f}_{vi:.1f}_{vf:.1f}"
        if key not in edge_ft_dict:
            edge_ft_dict[key] = np.concatenate([
                enc_btype.transform([[bt]])[0], enc_ring.transform([[rg]])[0],
                enc_conj.transform([[cj]])[0], enc_barom.transform([[ba]])[0],
                enc_vali.transform([[vi]])[0], enc_valf.transform([[vf]])[0],
            ])

    wts, aroms, hydros, charges, pisos, logps, atom_cts = [], [], [], [], [], [], []
    for aa in _STANDARD_AAS:
        mol_aa = Chem.MolFromHELM(_sequence_to_helm(aa))
        bp = ProteinAnalysis(aa)
        wts.append(round(Descriptors.MolWt(mol_aa), 4))
        aroms.append(round(bp.aromaticity(), 4))
        hydros.append(round(bp.gravy(), 4))
        charges.append(round(bp.charge_at_pH(7), 4))
        pisos.append(round(bp.isoelectric_point(), 4))
        logps.append(round(Crippen.MolLogP(mol_aa), 4))
        atom_cts.append(round(float(mol_aa.GetNumAtoms()), 4))

    enc_aa_arom = _fit_ohe(aroms)
    aa_ft_dict: dict[str, np.ndarray] = {}
    for wt, ar, hy, ch, pi, lp, ac in zip(wts, aroms, hydros, charges, pisos, logps, atom_cts):
        key = f"{wt}_{ar}_{hy}_{ch}_{pi}_{lp}_{ac}"
        if key not in aa_ft_dict:
            aa_ft_dict[key] = np.concatenate([
                [wt], enc_aa_arom.transform([[ar]])[0], [hy], [ch], [pi], [lp], [ac],
            ])

    node_dim    = next(iter(node_ft_dict.values())).shape[0]
    edge_dim    = next(iter(edge_ft_dict.values())).shape[0]
    aa_feat_dim = next(iter(aa_ft_dict.values())).shape[0] + 1  # +1 stereo flag

    return node_ft_dict, edge_ft_dict, aa_ft_dict, node_dim, edge_dim, aa_feat_dim


def _get_monomer_labels(mol):
    peptide_bonds, non_peptide = [], []
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        n1, n2 = a1.GetAtomicNum(), a2.GetAtomicNum()
        nb1 = [nb.GetAtomicNum() for nb in a1.GetNeighbors()]
        nb2 = [nb.GetAtomicNum() for nb in a2.GetNeighbors()]
        h1, h2 = a1.GetHybridization(), a2.GetHybridization()
        hs1, hs2 = a1.GetTotalNumHs(), a2.GetTotalNumHs()
        bt  = str(bond.GetBondType())
        cj  = str(bond.GetIsConjugated())
        pair = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        is_peptide = (
            n1 == 6 and n2 == 7 and 8 in nb1
            and str(h1) == "SP2" and str(h2) == "SP2"
            and hs1 == 0 and (hs2 == 0 or hs2 == 1)
            and cj == "True" and bt == "SINGLE"
        ) or (
            n1 == 7 and n2 == 6 and 8 in nb2
            and str(h1) == "SP2" and str(h2) == "SP2"
            and (hs1 == 0 or hs1 == 1) and hs2 == 0
            and cj == "True" and bt == "SINGLE"
        )
        peptide_bonds.append(pair)
        if not is_peptide:
            non_peptide.append(pair)

    unique_bonds = list(set(peptide_bonds).symmetric_difference(set(non_peptide)))
    break_idx = [
        mol.GetBondBetweenAtoms(a, b).GetIdx() for a, b in unique_bonds
        if mol.GetBondBetweenAtoms(a, b) is not None
    ]
    if not break_idx:
        return torch.zeros(mol.GetNumAtoms(), dtype=torch.long)

    mol_f = Chem.FragmentOnBonds(mol, break_idx, addDummies=False)
    frags  = Chem.GetMolFrags(mol_f)
    labels = np.empty(mol.GetNumAtoms(), dtype=np.int64)
    for frag_idx, atom_indices in enumerate(frags):
        for atom_i in atom_indices:
            if atom_i < mol.GetNumAtoms():
                labels[atom_i] = frag_idx
    return torch.tensor(labels, dtype=torch.long)


def _get_aa_features_for_seq(original_seq, safe_seq, aa_ft_dict, fallback_dim=8):
    rows = []
    for orig_aa, safe_aa in zip(original_seq, safe_seq):
        mol_aa = Chem.MolFromHELM(_sequence_to_helm(safe_aa))
        bp = ProteinAnalysis(safe_aa)
        wt = round(Descriptors.MolWt(mol_aa), 4)
        ar = round(bp.aromaticity(), 4)
        hy = round(bp.gravy(), 4)
        ch = round(bp.charge_at_pH(7), 4)
        pi = round(bp.isoelectric_point(), 4)
        lp = round(Crippen.MolLogP(mol_aa), 4)
        ac = round(float(mol_aa.GetNumAtoms()), 4)
        key  = f"{wt}_{ar}_{hy}_{ch}_{pi}_{lp}_{ac}"
        base = aa_ft_dict[key] if key in aa_ft_dict else np.zeros(fallback_dim)
        stereo_flag = np.array([1.0 if orig_aa.islower() else 0.0])
        rows.append(np.concatenate([base, stereo_flag]))
    return torch.tensor(np.array(rows), dtype=torch.float32)


def _aa_edge_index(n):
    if n <= 1:
        return torch.zeros((2, 0), dtype=torch.long)
    src = list(range(n - 1)) + list(range(1, n))
    dst = list(range(1, n)) + list(range(n - 1))
    return torch.tensor([src, dst], dtype=torch.long)


class PepMNetSample:
    __slots__ = ("x", "edge_index", "edge_attr", "monomer_labels",
                 "aa_features", "amino_index", "y")

    def __init__(self, x, edge_index, edge_attr, monomer_labels, aa_features, amino_index, y):
        self.x              = x
        self.edge_index     = edge_index
        self.edge_attr      = edge_attr
        self.monomer_labels = monomer_labels
        self.aa_features    = aa_features
        self.amino_index    = amino_index
        self.y              = y


def sequence_to_sample(seq, y, node_ft_dict, edge_ft_dict, aa_ft_dict, node_dim, edge_dim):
    safe = _safe_sequence(seq)
    mol = Chem.MolFromHELM(_sequence_to_helm(safe))
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    node_rows = []
    for atom in mol.GetAtoms():
        k = (f"{atom.GetAtomicNum()}_{int(atom.GetIsAromatic())}"
             f"_{atom.GetDegree()}_{atom.GetTotalNumHs()}"
             f"_{atom.GetHybridization().real}_{atom.GetImplicitValence()}")
        node_rows.append(node_ft_dict[k] if k in node_ft_dict else np.zeros(node_dim))

    src_idx, dst_idx, edge_rows = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        k = (f"{bond.GetBondTypeAsDouble():.1f}"
             f"_{int(bond.IsInRing()):.1f}"
             f"_{int(bond.GetIsConjugated()):.1f}"
             f"_{int(bond.GetIsAromatic()):.1f}"
             f"_{int(bond.GetValenceContrib(bond.GetBeginAtom())):.1f}"
             f"_{int(bond.GetValenceContrib(bond.GetEndAtom())):.1f}")
        feat = edge_ft_dict[k] if k in edge_ft_dict else np.zeros(edge_dim)
        src_idx += [i, j]; dst_idx += [j, i]
        edge_rows += [feat, feat]

    x          = torch.tensor(np.array(node_rows), dtype=torch.float32)
    edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
    edge_attr  = torch.tensor(np.array(edge_rows) if edge_rows else
                              np.zeros((0, edge_dim)), dtype=torch.float32)

    monomer_labels = _get_monomer_labels(mol)
    n_aa           = int(monomer_labels.max().item()) + 1
    aa_features    = _get_aa_features_for_seq(seq, safe, aa_ft_dict)
    amino_index    = _aa_edge_index(n_aa)
    y_tensor       = torch.tensor([y], dtype=torch.float32) if y is not None else None

    return PepMNetSample(x, edge_index, edge_attr, monomer_labels, aa_features, amino_index, y_tensor)


# ── model ──────────────────────────────────────────────────────────────────────

class PepMNetRT(nn.Module):
    def __init__(self, node_dim, edge_dim, aa_feat_dim=9,
                 hidden_nn_1=HIDDEN_NN_1, hidden_nn_2=HIDDEN_NN_2,
                 hidden_nn_3=HIDDEN_NN_3, hidden_gat_1=HIDDEN_GAT_1,
                 hidden_fcn_1=HIDDEN_FCN_1, hidden_fcn_2=HIDDEN_FCN_2,
                 hidden_fcn_3=HIDDEN_FCN_3, dropout=DROPOUT):
        super().__init__()
        self.nn_conv_1 = NNConv(node_dim, hidden_nn_1,
                                nn=nn.Linear(edge_dim, node_dim * hidden_nn_1), aggr="add")
        self.nn_conv_2 = NNConv(hidden_nn_1, hidden_nn_2,
                                nn=nn.Linear(edge_dim, hidden_nn_1 * hidden_nn_2), aggr="add")
        self.nn_conv_3 = NNConv(hidden_nn_2, hidden_nn_3,
                                nn=nn.Linear(edge_dim, hidden_nn_2 * hidden_nn_3), aggr="add")
        aa_in = hidden_nn_3 + aa_feat_dim
        self.nn_gat_1 = ARMAConv(aa_in, hidden_gat_1, num_stacks=3, num_layers=7,
                                 shared_weights=False, dropout=0.0)
        self.linear1 = nn.Linear(hidden_gat_1, hidden_fcn_1)
        self.linear2 = nn.Linear(hidden_fcn_1, hidden_fcn_2)
        self.linear3 = nn.Linear(hidden_fcn_2, hidden_fcn_3)
        self.linear4 = nn.Linear(hidden_fcn_3, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, idx_batch,
                aa_features_list, amino_index_list, monomer_labels):
        x = F.relu(self.nn_conv_1(x, edge_index, edge_attr))
        x = F.relu(self.nn_conv_2(x, edge_index, edge_attr))
        x = F.relu(self.nn_conv_3(x, edge_index, edge_attr))

        peptide_reps = []
        for i in range(len(aa_features_list)):
            mask = idx_batch == i
            xi   = x[mask]
            xi   = scatter(xi, monomer_labels[mask], dim=0, reduce="sum")
            xi   = torch.cat([xi, aa_features_list[i]], dim=1)
            xi   = F.relu(self.nn_gat_1(xi, amino_index_list[i]))
            batch_all = torch.zeros(xi.size(0), dtype=torch.long, device=xi.device)
            peptide_reps.append(global_add_pool(xi, batch_all))

        p = torch.cat(peptide_reps, dim=0)
        p = self.dropout(p);  p = F.relu(self.linear1(p))
        p = self.dropout(p);  p = F.relu(self.linear2(p))
        p = self.dropout(p);  p = F.relu(self.linear3(p))
        return self.linear4(p).view(-1)


# ── dataset & collation ────────────────────────────────────────────────────────

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, samples): self.samples = samples
    def __len__(self):            return len(self.samples)
    def __getitem__(self, idx):   return self.samples[idx]


class PepMNetBatch:
    __slots__ = ("x", "edge_index", "edge_attr", "idx_batch",
                 "aa_features_list", "amino_index_list", "monomer_labels", "y", "num_graphs")

    def __init__(self, samples):
        self.num_graphs = len(samples)
        edge_index_list, offset = [], 0
        for s in samples:
            edge_index_list.append(s.edge_index + offset)
            offset += s.x.size(0)
        self.x          = torch.cat([s.x for s in samples], dim=0)
        self.edge_index = torch.cat(edge_index_list, dim=1) if edge_index_list else \
                          torch.zeros((2, 0), dtype=torch.long)
        self.edge_attr  = torch.cat([s.edge_attr for s in samples], dim=0)
        self.idx_batch  = torch.cat([torch.full((s.x.size(0),), i, dtype=torch.long)
                                     for i, s in enumerate(samples)])
        self.monomer_labels   = torch.cat([s.monomer_labels for s in samples])
        self.aa_features_list = [s.aa_features for s in samples]
        self.amino_index_list = [s.amino_index  for s in samples]
        self.y = torch.cat([s.y for s in samples]) if samples[0].y is not None else None

    def to(self, device):
        self.x              = self.x.to(device)
        self.edge_index     = self.edge_index.to(device)
        self.edge_attr      = self.edge_attr.to(device)
        self.idx_batch      = self.idx_batch.to(device)
        self.monomer_labels = self.monomer_labels.to(device)
        self.aa_features_list = [t.to(device) for t in self.aa_features_list]
        self.amino_index_list = [t.to(device) for t in self.amino_index_list]
        if self.y is not None:
            self.y = self.y.to(device)
        return self


def collate_pepmnet(samples): return PepMNetBatch(samples)


# ── training & inference ───────────────────────────────────────────────────────

def _forward(model, batch):
    return model(batch.x, batch.edge_index, batch.edge_attr, batch.idx_batch,
                 batch.aa_features_list, batch.amino_index_list, batch.monomer_labels)


def train(model, train_loader, val_loader) -> list[dict]:
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=LR_PATIENCE, factor=0.5, min_lr=1e-6)
    criterion = nn.MSELoss()
    history, best_val, best_state, no_improve = [], float("inf"), None, 0

    bar = tqdm(range(1, MAX_EPOCHS + 1), desc="Training", unit="epoch")
    for epoch in bar:
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            loss = criterion(_forward(model, batch), batch.y)
            loss.backward(); opt.step()
            train_loss += loss.item() * batch.num_graphs
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                val_loss += criterion(_forward(model, batch), batch.y).item() * batch.num_graphs
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}",
                        best=f"{best_val:.4f}", patience=no_improve)
        if no_improve >= PATIENCE:
            print(f"\n  Early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def predict(model, samples) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(samples), BATCH_SIZE):
            batch = PepMNetBatch(samples[i:i+BATCH_SIZE]).to(DEVICE)
            preds.append(_forward(model, batch).cpu().numpy())
    return np.concatenate(preds)


# ── metrics ────────────────────────────────────────────────────────────────────

def regression_metrics(y_true, y_pred) -> dict:
    mse = float(mean_squared_error(y_true, y_pred))
    return dict(mse=mse, rmse=float(np.sqrt(mse)),
                mae=float(mean_absolute_error(y_true, y_pred)),
                mean_error=float(np.mean(y_pred - y_true)),
                r2=float(r2_score(y_true, y_pred)),
                pearson=float(stats.pearsonr(y_true, y_pred)[0]),
                spearman=float(stats.spearmanr(y_true, y_pred)[0]),
                kendall=float(stats.kendalltau(y_true, y_pred)[0]))


# ── reporting ──────────────────────────────────────────────────────────────────

class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def save_results(seed, test_metrics, train_metrics, training, config, output_dir, stem):
    result = {
        "benchmark": stem, "dataset": "dia", "seed": seed,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": config, "training": training,
        "test_metrics": test_metrics, "train_metrics": train_metrics,
    }
    out = output_dir / f"{stem}_seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, cls=_NpEncoder))
    print(f"\nResults saved to {out}")


# ── main ───────────────────────────────────────────────────────────────────────

def encode_sequences(seqs, ys, node_ft_dict, edge_ft_dict, aa_ft_dict,
                     node_dim, edge_dim, aa_feat_dim, desc="Encoding"):
    samples, bad = [], []
    ys = ys if ys is not None else [None] * len(seqs)
    for seq, y in tqdm(zip(seqs, ys), total=len(seqs), desc=desc, leave=False):
        s = sequence_to_sample(seq, y, node_ft_dict, edge_ft_dict, aa_ft_dict, node_dim, edge_dim)
        if s is not None:
            samples.append(s)
        else:
            bad.append(seq)
    if bad:
        print(f"  [{desc}] skipped {len(bad)} invalid sequences")
    return samples


def run_one_seed(seed, train_samples, val_samples, test_samples,
                 y_train, y_test, node_dim, edge_dim, aa_feat_dim,
                 weights_path=None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(SequenceDataset(train_samples), batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_pepmnet, num_workers=0)
    val_loader   = DataLoader(SequenceDataset(val_samples), batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=collate_pepmnet, num_workers=0)
    model = PepMNetRT(node_dim=node_dim, edge_dim=edge_dim, aa_feat_dim=aa_feat_dim).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  PepMNetRT | {n_params:,} params | node_dim={node_dim} edge_dim={edge_dim} aa_feat_dim={aa_feat_dim}")

    if weights_path is not None and weights_path.exists():
        print(f"  [weights] Loading from {weights_path} — skipping training")
        ckpt    = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        history = ckpt["history"]
    else:
        history = train(model, train_loader, val_loader)
        if weights_path is not None:
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "history": history}, weights_path)
            print(f"  [weights] Saved to {weights_path}")

    y_pred_test  = predict(model, test_samples)
    test_metrics = regression_metrics(y_test, y_pred_test)
    print(f"  RMSE={test_metrics['rmse']:.4f}  Pearson={test_metrics['pearson']:+.4f}"
          f"  Spearman={test_metrics['spearman']:+.4f}")

    y_pred_train  = predict(model, train_samples)
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

    print("[features] Building feature dictionaries …")
    node_ft_dict, edge_ft_dict, aa_ft_dict, node_dim, edge_dim, aa_feat_dim = build_feature_dicts()
    print(f"  node_dim={node_dim}  edge_dim={edge_dim}  aa_feat_dim={aa_feat_dim}")

    print("[data] Loading DIA dataset …")
    train_seqs, y_train_raw, val_seqs, y_val_raw, test_seqs, y_test_raw = load_dia_data()
    print(f"  train={len(train_seqs)}  val={len(val_seqs)}  test={len(test_seqs)}")

    kwargs = dict(node_ft_dict=node_ft_dict, edge_ft_dict=edge_ft_dict,
                  aa_ft_dict=aa_ft_dict, node_dim=node_dim, edge_dim=edge_dim,
                  aa_feat_dim=aa_feat_dim)

    train_samples = encode_sequences(train_seqs, y_train_raw.tolist(), desc="Graphs train", **kwargs)
    val_samples   = encode_sequences(val_seqs,   y_val_raw.tolist(),   desc="Graphs val  ", **kwargs)
    test_samples  = encode_sequences(test_seqs,  y_test_raw.tolist(),  desc="Graphs test ", **kwargs)

    y_train = np.array([s.y.item() for s in train_samples], dtype=np.float32)
    y_test  = np.array([s.y.item() for s in test_samples],  dtype=np.float32)
    print(f"  after filtering: train={len(train_samples)}  val={len(val_samples)}  test={len(test_samples)}")

    weights_path = WEIGHTS_DIR / f"results_pepmnet_dia_seed{seed}.pt"
    print(f"\n── Seed {seed} ──")
    test_metrics, train_metrics, history = run_one_seed(
        seed, train_samples, val_samples, test_samples,
        y_train, y_test, node_dim, edge_dim, aa_feat_dim,
        weights_path=weights_path,
    )

    config = {
        "hidden_nn_1": HIDDEN_NN_1, "hidden_nn_2": HIDDEN_NN_2,
        "hidden_nn_3": HIDDEN_NN_3, "hidden_gat_1": HIDDEN_GAT_1,
        "hidden_fcn_1": HIDDEN_FCN_1, "hidden_fcn_2": HIDDEN_FCN_2,
        "hidden_fcn_3": HIDDEN_FCN_3, "dropout": DROPOUT,
        "node_dim": node_dim, "edge_dim": edge_dim, "aa_feat_dim": aa_feat_dim,
        "lr": LR, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "lr_patience": LR_PATIENCE, "device": DEVICE,
    }
    training = {"epochs_run": history[-1]["epoch"],
                "best_val_loss": min(h["val_loss"] for h in history)}
    save_results(seed, test_metrics, train_metrics, training, config,
                 RESULTS_DIR, "results_pepmnet_dia")
    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
