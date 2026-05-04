"""
PepMNet benchmark for the StereoPep dataset.

Implements the hierarchical GNN from:
  "PepMNet: a hybrid deep learning model for predicting peptide properties
   using hierarchical graph representations"
  (Garzon-Otero et al., 2024 — https://github.com/danielgarzonotero/PepMNet)

Architecture:
  Atom-level: 3 × NNConv (edge-conditioned graph convolution)
  → scatter-sum readout per amino acid
  → concatenate 8-dim biochemical AA features
  Amino acid-level: ARMAConv
  → sum pooling over amino acids
  → 4-layer MLP head → scalar retention-time prediction

Peptides with non-standard residues (e.g. 'f' for D-Phe) are handled by
mapping lowercase codes to their uppercase equivalents before HELM/RDKit
parsing — the 2D molecular graph cannot distinguish D/L stereoisomers, so
stereo-ordering accuracy is expected to be ~0.5.

Results are written to benchmarks/output/results_pepmnet_seed{N}.json.

Extra dependencies (beyond the standard benchmark requirements):
  pip install biopython
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset as hf_load_dataset
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
HF_REPO = "amirka20/StereoPep"

# PepMNet RT hyperparameters (from paper / rt_main.py defaults)
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
BATCH_SIZE   = 16   # small: forward pass loops per-sample
MAX_EPOCHS   = 100
PATIENCE     = 10
LR_PATIENCE  = 10
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR = Path(__file__).parent / "output"
WEIGHTS_DIR = Path(__file__).parent / "weights"

# Uppercase fallback map for non-standard single-letter codes
_AA_UPPER_MAP: dict[str, str] = {
    "f": "F",  # D-Phe → L-Phe (same 2D structure; chirality invisible in graph)
}


# ── feature extraction ─────────────────────────────────────────────────────────
# Mirrors PepMNet's src/utils.py get_features() and src/aminoacids_features.py

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
    """Convert a standard-AA sequence to HELM notation for RDKit."""
    seq = sequence.replace("(ac)", "[ac].").rstrip(".")
    helm_body = ".".join(seq)
    return f"{polymer_id}{{{helm_body}}}$$$$"


def _safe_sequence(seq: str) -> str:
    """Map non-standard residue codes to their standard-AA equivalents."""
    return "".join(_AA_UPPER_MAP.get(c, c) for c in seq)


def build_feature_dicts() -> tuple[dict, dict, dict, int, int]:
    """
    Build node, edge, and amino acid feature look-up dicts from sample peptides.

    Returns
    -------
    node_ft_dict : str → np.ndarray (node feature vector)
    edge_ft_dict : str → np.ndarray (edge feature vector)
    aa_ft_dict   : str → np.ndarray (amino acid feature vector, dim=8)
    node_dim     : int  (length of node feature vector)
    edge_dim     : int  (length of edge feature vector)
    """
    # ── collect raw values from sample peptides ────────────────────────────
    atomic_num_raw, aromatic_raw, degree_raw, h_raw, hybrid_raw, ival_raw = \
        [], [], [], [], [], []
    btype_raw, ring_raw, conj_raw, barom_raw, vali_raw, valf_raw = \
        [], [], [], [], [], []

    for seq in _SAMPLE_SEQS:
        safe = _safe_sequence(seq)
        mol = Chem.MolFromHELM(_sequence_to_helm(safe))
        if mol is None:
            continue
        atomic_num_raw.extend(atom.GetAtomicNum()        for atom in mol.GetAtoms())
        aromatic_raw.extend(int(atom.GetIsAromatic())    for atom in mol.GetAtoms())
        degree_raw.extend(atom.GetDegree()               for atom in mol.GetAtoms())
        h_raw.extend(atom.GetTotalNumHs()                for atom in mol.GetAtoms())
        hybrid_raw.extend(atom.GetHybridization().real   for atom in mol.GetAtoms())
        ival_raw.extend(atom.GetImplicitValence()        for atom in mol.GetAtoms())

        for bond in mol.GetBonds():
            bt = bond.GetBondTypeAsDouble()
            btype_raw.append(bt)
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

    # ── build node feature dict ────────────────────────────────────────────
    node_ft_dict: dict[str, np.ndarray] = {}
    for a, ar, d, h, hy, iv in zip(
        atomic_num_raw, aromatic_raw, degree_raw, h_raw, hybrid_raw, ival_raw
    ):
        key = f"{a}_{ar}_{d}_{h}_{hy}_{iv}"
        if key not in node_ft_dict:
            node_ft_dict[key] = np.concatenate([
                enc_atomic.transform([[a]])[0],
                enc_arom.transform([[ar]])[0],
                enc_degree.transform([[d]])[0],
                enc_h.transform([[h]])[0],
                enc_hybrid.transform([[hy]])[0],
                enc_ival.transform([[iv]])[0],
            ])

    # ── build edge feature dict ────────────────────────────────────────────
    edge_ft_dict: dict[str, np.ndarray] = {}
    for bt, rg, cj, ba, vi, vf in zip(
        btype_raw, ring_raw, conj_raw, barom_raw, vali_raw, valf_raw
    ):
        key = f"{bt:.1f}_{rg:.1f}_{cj:.1f}_{ba:.1f}_{vi:.1f}_{vf:.1f}"
        if key not in edge_ft_dict:
            edge_ft_dict[key] = np.concatenate([
                enc_btype.transform([[bt]])[0],
                enc_ring.transform([[rg]])[0],
                enc_conj.transform([[cj]])[0],
                enc_barom.transform([[ba]])[0],
                enc_vali.transform([[vi]])[0],
                enc_valf.transform([[vf]])[0],
            ])

    # ── build amino acid feature dict ──────────────────────────────────────
    wts, aroms, hydros, charges, pisos, logps, atom_cts = \
        [], [], [], [], [], [], []
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
    for wt, ar, hy, ch, pi, lp, ac in zip(
        wts, aroms, hydros, charges, pisos, logps, atom_cts
    ):
        key = f"{wt}_{ar}_{hy}_{ch}_{pi}_{lp}_{ac}"
        if key not in aa_ft_dict:
            aa_ft_dict[key] = np.concatenate([
                [wt],
                enc_aa_arom.transform([[ar]])[0],
                [hy], [ch], [pi], [lp], [ac],
            ])

    # Derive dimensions from one representative entry
    node_dim   = next(iter(node_ft_dict.values())).shape[0]
    edge_dim   = next(iter(edge_ft_dict.values())).shape[0]
    # +1 for the D/L stereo flag appended per residue at runtime
    aa_feat_dim = next(iter(aa_ft_dict.values())).shape[0] + 1

    return node_ft_dict, edge_ft_dict, aa_ft_dict, node_dim, edge_dim, aa_feat_dim


# ── per-atom label extraction (which amino acid owns each atom) ────────────────

def _get_monomer_labels(mol: Chem.Mol) -> torch.Tensor:
    """
    Fragment molecule at peptide bonds and label each atom with its
    residue index (0-based).  Mirrors PepMNet's get_label_aminoacid_atoms().
    """
    peptide_bonds, non_peptide = [], []
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        n1, n2 = a1.GetAtomicNum(), a2.GetAtomicNum()
        nb1 = [nb.GetAtomicNum() for nb in a1.GetNeighbors()]
        nb2 = [nb.GetAtomicNum() for nb in a2.GetNeighbors()]
        h1 = a1.GetHybridization()
        h2 = a2.GetHybridization()
        hs1 = a1.GetTotalNumHs()
        hs2 = a2.GetTotalNumHs()
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
    frags = Chem.GetMolFrags(mol_f)
    labels = np.empty(mol.GetNumAtoms(), dtype=np.int64)
    for frag_idx, atom_indices in enumerate(frags):
        for atom_i in atom_indices:
            if atom_i < mol.GetNumAtoms():
                labels[atom_i] = frag_idx
    return torch.tensor(labels, dtype=torch.long)


def _get_aa_features_for_seq(
    original_seq: str,
    safe_seq: str,
    aa_ft_dict: dict,
    fallback_dim: int = 8,
) -> torch.Tensor:
    """
    Compute per-residue feature vectors for each residue.

    Each vector = 8-dim biochemical features (from PepMNet)
                  + 1-dim D/L stereo flag (1.0 = D-amino acid, 0.0 = L).

    original_seq : sequence with original casing (lowercase = D-residue).
    safe_seq     : same sequence with lowercase mapped to uppercase for RDKit.
    """
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
        key = f"{wt}_{ar}_{hy}_{ch}_{pi}_{lp}_{ac}"
        base = aa_ft_dict[key] if key in aa_ft_dict else np.zeros(fallback_dim)
        # Append D/L stereo flag: 1.0 for D-residues (lowercase), 0.0 for L
        stereo_flag = np.array([1.0 if orig_aa.islower() else 0.0])
        rows.append(np.concatenate([base, stereo_flag]))
    return torch.tensor(np.array(rows), dtype=torch.float32)


def _aa_edge_index(n: int) -> torch.Tensor:
    """Bidirectional linear chain for n amino acids → shape (2, 2*(n-1))."""
    if n <= 1:
        return torch.zeros((2, 0), dtype=torch.long)
    src = list(range(n - 1)) + list(range(1, n))
    dst = list(range(1, n)) + list(range(n - 1))
    return torch.tensor([src, dst], dtype=torch.long)


# ── data class and featuriser ──────────────────────────────────────────────────

class PepMNetSample:
    """Holds all tensors for a single peptide (CPU)."""
    __slots__ = (
        "x", "edge_index", "edge_attr", "monomer_labels",
        "aa_features", "amino_index", "y",
    )

    def __init__(
        self,
        x, edge_index, edge_attr, monomer_labels,
        aa_features, amino_index, y,
    ):
        self.x             = x
        self.edge_index    = edge_index
        self.edge_attr     = edge_attr
        self.monomer_labels = monomer_labels
        self.aa_features   = aa_features   # (n_aa, 8)
        self.amino_index   = amino_index   # (2, n_aa_edges)
        self.y             = y             # scalar float tensor or None


def sequence_to_sample(
    seq: str,
    y: float | None,
    node_ft_dict: dict,
    edge_ft_dict: dict,
    aa_ft_dict: dict,
    node_dim: int,
    edge_dim: int,
) -> PepMNetSample | None:
    """Convert a peptide sequence string to a PepMNetSample."""
    safe = _safe_sequence(seq)
    mol = Chem.MolFromHELM(_sequence_to_helm(safe))
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    # Node features
    node_rows = []
    for atom in mol.GetAtoms():
        k = (f"{atom.GetAtomicNum()}_{int(atom.GetIsAromatic())}"
             f"_{atom.GetDegree()}_{atom.GetTotalNumHs()}"
             f"_{atom.GetHybridization().real}_{atom.GetImplicitValence()}")
        if k in node_ft_dict:
            node_rows.append(node_ft_dict[k])
        else:
            node_rows.append(np.zeros(node_dim))

    # Edge index + edge features
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
        src_idx += [i, j]
        dst_idx += [j, i]
        edge_rows += [feat, feat]

    x          = torch.tensor(np.array(node_rows), dtype=torch.float32)
    edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
    edge_attr  = torch.tensor(np.array(edge_rows) if edge_rows else
                              np.zeros((0, edge_dim)), dtype=torch.float32)

    monomer_labels = _get_monomer_labels(mol)
    n_aa           = int(monomer_labels.max().item()) + 1

    aa_features  = _get_aa_features_for_seq(seq, safe, aa_ft_dict)
    amino_index  = _aa_edge_index(n_aa)

    y_tensor = torch.tensor([y], dtype=torch.float32) if y is not None else None

    return PepMNetSample(x, edge_index, edge_attr, monomer_labels,
                         aa_features, amino_index, y_tensor)


# ── model ──────────────────────────────────────────────────────────────────────

class PepMNetRT(nn.Module):
    """
    Hierarchical GNN for retention-time regression.
    Faithful re-implementation of rt_pepmnet from PepMNet.
    """

    def __init__(
        self,
        node_dim:     int,
        edge_dim:     int,
        aa_feat_dim:  int = 9,   # 8 biochemical + 1 D/L stereo flag
        hidden_nn_1:  int = HIDDEN_NN_1,
        hidden_nn_2:  int = HIDDEN_NN_2,
        hidden_nn_3:  int = HIDDEN_NN_3,
        hidden_gat_1: int = HIDDEN_GAT_1,
        hidden_fcn_1: int = HIDDEN_FCN_1,
        hidden_fcn_2: int = HIDDEN_FCN_2,
        hidden_fcn_3: int = HIDDEN_FCN_3,
        dropout:      float = DROPOUT,
    ):
        super().__init__()

        # Three NNConv layers at atom level
        self.nn_conv_1 = NNConv(
            node_dim, hidden_nn_1,
            nn=nn.Linear(edge_dim, node_dim * hidden_nn_1),
            aggr="add",
        )
        self.nn_conv_2 = NNConv(
            hidden_nn_1, hidden_nn_2,
            nn=nn.Linear(edge_dim, hidden_nn_1 * hidden_nn_2),
            aggr="add",
        )
        self.nn_conv_3 = NNConv(
            hidden_nn_2, hidden_nn_3,
            nn=nn.Linear(edge_dim, hidden_nn_2 * hidden_nn_3),
            aggr="add",
        )

        # ARMAConv at amino acid level
        # Input = atom readout (hidden_nn_3) + external AA features (aa_feat_dim)
        aa_in = hidden_nn_3 + aa_feat_dim
        self.nn_gat_1 = ARMAConv(
            aa_in, hidden_gat_1,
            num_stacks=3, num_layers=7,
            shared_weights=False, dropout=0.0,
        )

        # MLP head
        self.linear1 = nn.Linear(hidden_gat_1, hidden_fcn_1)
        self.linear2 = nn.Linear(hidden_fcn_1, hidden_fcn_2)
        self.linear3 = nn.Linear(hidden_fcn_2, hidden_fcn_3)
        self.linear4 = nn.Linear(hidden_fcn_3, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x:                torch.Tensor,         # (total_atoms, node_dim)
        edge_index:       torch.Tensor,         # (2, total_bonds)
        edge_attr:        torch.Tensor,         # (total_bonds, edge_dim)
        idx_batch:        torch.Tensor,         # (total_atoms,) sample index
        aa_features_list: list[torch.Tensor],   # [n_samples] each (n_aa_i, 8)
        amino_index_list: list[torch.Tensor],   # [n_samples] each (2, n_aa_edges_i)
        monomer_labels:   torch.Tensor,         # (total_atoms,) AA index per atom
    ) -> torch.Tensor:

        # ── atom-level processing ──────────────────────────────────────────
        x = F.relu(self.nn_conv_1(x, edge_index, edge_attr))
        x = F.relu(self.nn_conv_2(x, edge_index, edge_attr))
        x = F.relu(self.nn_conv_3(x, edge_index, edge_attr))

        # ── per-sample amino acid-level processing ─────────────────────────
        peptide_reps = []
        n_samples = len(aa_features_list)
        for i in range(n_samples):
            mask  = idx_batch == i
            xi    = x[mask]                         # (n_atoms_i, hidden_nn_3)
            ml_i  = monomer_labels[mask]             # (n_atoms_i,)

            # Scatter-sum atoms → amino acid representations
            xi = scatter(xi, ml_i, dim=0, reduce="sum")  # (n_aa_i, hidden_nn_3)

            # Concatenate biochemical AA features
            xi = torch.cat([xi, aa_features_list[i]], dim=1)  # (n_aa_i, hidden_nn_3+8)

            # ARMAConv over linear peptide chain
            xi = F.relu(self.nn_gat_1(xi, amino_index_list[i]))  # (n_aa_i, hidden_gat_1)

            # Sum pooling → peptide vector
            batch_all = torch.zeros(xi.size(0), dtype=torch.long, device=xi.device)
            xi = global_add_pool(xi, batch_all)  # (1, hidden_gat_1)
            peptide_reps.append(xi)

        p = torch.cat(peptide_reps, dim=0)  # (n_samples, hidden_gat_1)

        p = self.dropout(p)
        p = F.relu(self.linear1(p))
        p = self.dropout(p)
        p = F.relu(self.linear2(p))
        p = self.dropout(p)
        p = F.relu(self.linear3(p))
        p = self.linear4(p)

        return p.view(-1)


# ── dataset & batch collation ──────────────────────────────────────────────────

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[PepMNetSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class PepMNetBatch:
    """Collated batch for PepMNetRT.forward()."""
    __slots__ = (
        "x", "edge_index", "edge_attr", "idx_batch",
        "aa_features_list", "amino_index_list", "monomer_labels", "y",
        "num_graphs",
    )

    def __init__(self, samples: list[PepMNetSample]):
        self.num_graphs = len(samples)

        # Concatenate atom-level tensors
        atom_offsets, edge_index_list = [], []
        offset = 0
        for s in samples:
            edge_index_list.append(s.edge_index + offset)
            atom_offsets.append(offset)
            offset += s.x.size(0)

        self.x          = torch.cat([s.x for s in samples], dim=0)
        self.edge_index = torch.cat(edge_index_list, dim=1) if edge_index_list else \
                          torch.zeros((2, 0), dtype=torch.long)
        self.edge_attr  = torch.cat([s.edge_attr for s in samples], dim=0)

        # Batch index: which sample each atom belongs to
        self.idx_batch = torch.cat([
            torch.full((s.x.size(0),), i, dtype=torch.long)
            for i, s in enumerate(samples)
        ])

        # Monomer labels: local (0-indexed per sample), concatenated
        self.monomer_labels = torch.cat([s.monomer_labels for s in samples])

        # Per-sample lists (not concatenated; forward iterates over them)
        self.aa_features_list = [s.aa_features for s in samples]
        self.amino_index_list = [s.amino_index  for s in samples]

        # Targets
        if samples[0].y is not None:
            self.y = torch.cat([s.y for s in samples])
        else:
            self.y = None

    def to(self, device: str | torch.device) -> "PepMNetBatch":
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


def collate_pepmnet(samples: list[PepMNetSample]) -> PepMNetBatch:
    return PepMNetBatch(samples)


# ── training & inference ───────────────────────────────────────────────────────

def _forward(model: PepMNetRT, batch: PepMNetBatch) -> torch.Tensor:
    return model(
        batch.x, batch.edge_index, batch.edge_attr,
        batch.idx_batch, batch.aa_features_list,
        batch.amino_index_list, batch.monomer_labels,
    )


def train(
    model: PepMNetRT,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> list[dict]:
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=LR_PATIENCE, factor=0.5, min_lr=1e-6
    )
    criterion = nn.MSELoss()
    history: list[dict] = []
    best_val, best_state, no_improve = float("inf"), None, 0

    bar = tqdm(range(1, MAX_EPOCHS + 1), desc="Training", unit="epoch")
    for epoch in bar:
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            pred = _forward(model, batch)
            loss = criterion(pred, batch.y)
            loss.backward()
            opt.step()
            train_loss += loss.item() * batch.num_graphs
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                val_loss += criterion(_forward(model, batch), batch.y).item() \
                            * batch.num_graphs
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


def predict(model: PepMNetRT, samples: list[PepMNetSample]) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(samples), BATCH_SIZE):
            batch = PepMNetBatch(samples[i : i + BATCH_SIZE]).to(DEVICE)
            preds.append(_forward(model, batch).cpu().numpy())
    return np.concatenate(preds)


# ── metrics ────────────────────────────────────────────────────────────────────

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mse":        mse,
        "rmse":       float(np.sqrt(mse)),
        "mae":        float(mean_absolute_error(y_true, y_pred)),
        "mean_error": float(np.mean(y_pred - y_true)),
        "r2":         float(r2_score(y_true, y_pred)),
        "pearson":    float(stats.pearsonr(y_true, y_pred)[0]),
        "spearman":   float(stats.spearmanr(y_true, y_pred)[0]),
        "kendall":    float(stats.kendalltau(y_true, y_pred)[0]),
    }


def stereo_ordering_accuracy(
    model: PepMNetRT,
    stereo_ds,
    node_ft_dict: dict,
    edge_ft_dict: dict,
    aa_ft_dict: dict,
    node_dim: int,
    edge_dim: int,
    aa_feat_dim: int,
) -> dict:
    def encode(seqs):
        out, bad = [], []
        for i, seq in enumerate(tqdm(seqs, desc="  Stereo graphs", leave=False)):
            s = sequence_to_sample(seq, None, node_ft_dict, edge_ft_dict,
                                   aa_ft_dict, node_dim, edge_dim)
            if s is None:
                bad.append(i)
                s = sequence_to_sample("A", None, node_ft_dict, edge_ft_dict,
                                       aa_ft_dict, node_dim, edge_dim)  # dummy placeholder
            out.append(s)
        return out, bad

    samples_f, bad_f = encode(stereo_ds["Sequence_f"])
    samples_F, bad_F = encode(stereo_ds["Sequence_F"])

    pred_f = predict(model, samples_f)
    pred_F = predict(model, samples_F)
    pred_delta = pred_f - pred_F

    delta_B = np.array(stereo_ds["delta_B"], dtype=np.float64)
    bad = set(bad_f) | set(bad_F)
    mask = np.ones(len(delta_B), dtype=bool)
    for i in bad:
        mask[i] = False

    true_sign = np.sign(delta_B[mask])
    pred_sign = np.sign(pred_delta[mask])
    correct   = int((true_sign == pred_sign).sum())
    total     = int(mask.sum())
    pr        = float(stats.pearsonr(delta_B[mask],  pred_delta[mask])[0])
    sr        = float(stats.spearmanr(delta_B[mask], pred_delta[mask])[0])

    return {
        "n_pairs":         total,
        "n_correct":       correct,
        "ordering_acc":    correct / total if total > 0 else float("nan"),
        "delta_pearson":   pr,
        "delta_spearman":  sr,
        "mean_true_delta": float(delta_B[mask].mean()),
        "mean_pred_delta": float(pred_delta[mask].mean()),
    }


# ── reporting ──────────────────────────────────────────────────────────────────

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
    training: dict,
    config: dict,
    output_dir: Path,
    stem: str,
) -> None:
    result = {
        "benchmark": stem,
        "seed":      seed,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config":    config,
        "training":  training,
        "test_metrics":          test_metrics,
        "train_metrics":         train_metrics,
        "stereo_metrics":        stereo_metrics,
        "stereo_trainval_metrics": stereo_trainval_metrics,
    }
    out = output_dir / f"{stem}_seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, cls=_NpEncoder))
    print(f"\nResults saved to {out}")


# ── main ───────────────────────────────────────────────────────────────────────

def encode_sequences(
    seqs: list[str],
    ys: list[float] | None,
    node_ft_dict: dict,
    edge_ft_dict: dict,
    aa_ft_dict: dict,
    node_dim: int,
    edge_dim: int,
    aa_feat_dim: int,   # accepted for uniform **kwargs; not used directly here
    desc: str = "Encoding",
) -> list[PepMNetSample]:
    samples, bad = [], []
    ys = ys if ys is not None else [None] * len(seqs)
    for seq, y in tqdm(zip(seqs, ys), total=len(seqs), desc=desc, leave=False):
        s = sequence_to_sample(seq, y, node_ft_dict, edge_ft_dict,
                               aa_ft_dict, node_dim, edge_dim)
        if s is not None:
            samples.append(s)
        else:
            bad.append(seq)
    if bad:
        print(f"  [{desc}] skipped {len(bad)} invalid sequences")
    return samples


def run_one_seed(
    seed: int,
    train_samples: list[PepMNetSample],
    val_samples:   list[PepMNetSample],
    test_samples:  list[PepMNetSample],
    y_train: np.ndarray,
    y_test: np.ndarray,
    stereo_ds,
    stereo_trainval_ds,
    node_ft_dict: dict,
    edge_ft_dict: dict,
    aa_ft_dict: dict,
    node_dim: int,
    edge_dim: int,
    aa_feat_dim: int,
    weights_path: Path | None = None,
) -> tuple[dict, dict, dict, dict, list[dict]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(
        SequenceDataset(train_samples), batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=collate_pepmnet, num_workers=0,
    )
    val_loader = DataLoader(
        SequenceDataset(val_samples), batch_size=BATCH_SIZE,
        shuffle=False, collate_fn=collate_pepmnet, num_workers=0,
    )

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

    stereo_metrics = stereo_ordering_accuracy(
        model, stereo_ds, node_ft_dict, edge_ft_dict, aa_ft_dict, node_dim, edge_dim, aa_feat_dim
    )
    print(f"  Ordering accuracy: {stereo_metrics['ordering_acc']:.4f}"
          f"  ({stereo_metrics['n_correct']}/{stereo_metrics['n_pairs']})")

    stereo_trainval_metrics = stereo_ordering_accuracy(
        model, stereo_trainval_ds, node_ft_dict, edge_ft_dict, aa_ft_dict, node_dim, edge_dim, aa_feat_dim
    )
    print(f"  Trainval ordering accuracy: {stereo_trainval_metrics['ordering_acc']:.4f}"
          f"  ({stereo_trainval_metrics['n_correct']}/{stereo_trainval_metrics['n_pairs']})")

    return test_metrics, train_metrics, stereo_metrics, stereo_trainval_metrics, history


def main() -> None:
    global MAX_EPOCHS, PATIENCE, LR_PATIENCE
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args()
    seed = args.seed

    MAX_EPOCHS  = args.epochs
    PATIENCE    = max(1, int(0.10 * MAX_EPOCHS))

    print(f"Device: {DEVICE}  |  seed={seed}  |  max_epochs={MAX_EPOCHS}  |  patience={PATIENCE}")
    t0 = time.time()

    print("[features] Building feature dictionaries from sample peptides …")
    node_ft_dict, edge_ft_dict, aa_ft_dict, node_dim, edge_dim, aa_feat_dim = build_feature_dicts()
    print(f"  node_dim={node_dim}  edge_dim={edge_dim}  aa_feat_dim={aa_feat_dim}")

    print("[data] Loading stereopep dataset …")
    ds     = hf_load_dataset(HF_REPO, "StereoPep")
    stereo          = hf_load_dataset(HF_REPO, "diastereomer_pairs")["diastereomer_pairs"]
    stereo_trainval = hf_load_dataset(HF_REPO, "diastereomer_pairs")["diastereomer_pairs_trainval"]

    kwargs = dict(
        node_ft_dict=node_ft_dict, edge_ft_dict=edge_ft_dict,
        aa_ft_dict=aa_ft_dict, node_dim=node_dim, edge_dim=edge_dim,
        aa_feat_dim=aa_feat_dim,
    )
    train_samples = encode_sequences(
        ds["train"]["Peptide"], ds["train"]["B"], desc="Graphs train", **kwargs)
    val_samples   = encode_sequences(
        ds["val"]["Peptide"],   ds["val"]["B"],   desc="Graphs val  ", **kwargs)
    test_samples  = encode_sequences(
        ds["test"]["Peptide"],  ds["test"]["B"],  desc="Graphs test ", **kwargs)

    y_train = np.array(ds["train"]["B"], dtype=np.float32)[: len(train_samples)]
    y_test  = np.array(ds["test"]["B"],  dtype=np.float32)[: len(test_samples)]

    print(f"  train={len(train_samples)}  val={len(val_samples)}  test={len(test_samples)}")

    weights_path = WEIGHTS_DIR / f"results_pepmnet_seed{seed}.pt"
    print(f"\n── Seed {seed} ──")
    test_metrics, train_metrics, stereo_metrics, stereo_trainval_metrics, history = run_one_seed(
        seed, train_samples, val_samples, test_samples, y_train, y_test, stereo, stereo_trainval,
        weights_path=weights_path, **kwargs
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
    training = {
        "epochs_run":    history[-1]["epoch"],
        "best_val_loss": min(h["val_loss"] for h in history),
    }
    save_results(seed, test_metrics, train_metrics, stereo_metrics, stereo_trainval_metrics,
                 training, config, RESULTS_DIR, "results_pepmnet")
    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
