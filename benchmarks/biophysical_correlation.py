"""
Biophysical feature correlation benchmark for the PepTag dataset.

Computes a suite of per-peptide biophysical descriptors from sequences and
SMILES representations, then measures their Pearson / Spearman / Kendall
correlations against the retention-time target (B, normalised 0-100).

Results are written to benchmarks/results.txt.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
from datasets import load_dataset as hf_load_dataset
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from scipy import stats

# ── paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_FILE = Path(__file__).parent / "output" / "results.txt"
HF_REPO = "amirka20/peptag"

# ── per-residue lookup tables ─────────────────────────────────────────────────
# Kyte-Doolittle hydrophobicity scale
KD_HYDROPHOBICITY: dict[str, float] = {
    "A":  1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C":  2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I":  4.5,
    "L":  3.8, "K": -3.9, "M":  1.9, "F":  2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V":  4.2,
    "f":  2.8,  # D-Phe same scale as F
}

# Approximate pKa values for ionisable side chains
PKA_SIDE_CHAIN: dict[str, float] = {
    "D": 3.65, "E": 4.25, "H": 6.00,
    "C": 8.18, "Y": 10.07, "K": 10.53,
    "R": 12.48,
}
PKA_NTERM = 8.0
PKA_CTERM = 3.1

# Residue molecular weights (monoisotopic, Da) — side-chain residue mass
RESIDUE_MW: dict[str, float] = {
    "A": 71.037, "R": 156.101, "N": 114.043, "D": 115.027, "C": 103.009,
    "Q": 128.059, "E": 129.043, "G": 57.021,  "H": 137.059, "I": 113.084,
    "L": 113.084, "K": 128.095, "M": 131.040, "F": 147.068, "P": 97.053,
    "S": 87.032,  "T": 101.048, "W": 186.079, "Y": 163.063, "V": 99.068,
    "f": 147.068,  # D-Phe
}
WATER_MW = 18.011

# ── SMILES per-residue → RDKit descriptors ────────────────────────────────────
_SMILES_MAP: dict[str, str] = {}

def _load_smiles() -> None:
    smiles_file = DATA_DIR / "SMILES.csv"
    with open(smiles_file) as fh:
        next(fh)  # header
        for line in fh:
            aa, smi = line.strip().split(",")
            _SMILES_MAP[aa] = smi
    # D-Phe ('f') uses the same SMILES as F if not already present
    if "f" not in _SMILES_MAP and "F" in _SMILES_MAP:
        _SMILES_MAP["f"] = _SMILES_MAP["F"]


def _rdkit_residue_props() -> dict[str, dict[str, float]]:
    """Compute RDKit descriptors for each amino acid residue."""
    props: dict[str, dict[str, float]] = {}
    for aa, smi in _SMILES_MAP.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        props[aa] = {
            "logp":    Descriptors.MolLogP(mol),
            "mw":      Descriptors.MolWt(mol),
            "tpsa":    rdMolDescriptors.CalcTPSA(mol),
            "hbd":     rdMolDescriptors.CalcNumHBD(mol),
            "hba":     rdMolDescriptors.CalcNumHBA(mol),
            "rot_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "rings":   rdMolDescriptors.CalcNumRings(mol),
        }
    return props


# ── feature computation ───────────────────────────────────────────────────────

def gravy(seq: str) -> float:
    """Grand Average of Hydropathicity (Kyte-Doolittle)."""
    scores = [KD_HYDROPHOBICITY.get(aa, 0.0) for aa in seq]
    return sum(scores) / len(scores) if scores else 0.0


def peptide_mw(seq: str) -> float:
    """Approximate monoisotopic molecular weight (Da)."""
    return sum(RESIDUE_MW.get(aa, 111.1) for aa in seq) + WATER_MW


def charge_at_ph(seq: str, ph: float = 2.0) -> float:
    """Net charge of the peptide at a given pH."""
    charge = 0.0
    # N-terminus
    charge += 1.0 / (1.0 + 10 ** (ph - PKA_NTERM))
    # C-terminus
    charge -= 1.0 / (1.0 + 10 ** (PKA_CTERM - ph))
    for aa in seq:
        pka = PKA_SIDE_CHAIN.get(aa)
        if pka is None:
            continue
        if aa in ("D", "E", "C", "Y"):       # acidic
            charge -= 1.0 / (1.0 + 10 ** (pka - ph))
        elif aa in ("H", "K", "R"):           # basic
            charge += 1.0 / (1.0 + 10 ** (ph - pka))
    return charge


def pI(seq: str) -> float:
    """Isoelectric point (binary search)."""
    lo, hi = 0.0, 14.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if charge_at_ph(seq, mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def hydrophobic_fraction(seq: str) -> float:
    hydrophobic = set("AVILMFYWP")
    return sum(aa in hydrophobic for aa in seq) / len(seq)


def aromatic_fraction(seq: str) -> float:
    aromatic = set("FWYf")
    return sum(aa in aromatic for aa in seq) / len(seq)


def proline_fraction(seq: str) -> float:
    return seq.count("P") / len(seq)


def charged_fraction(seq: str, ph: float = 2.0) -> float:
    """Fraction of charged residues at a given pH (simplified)."""
    charged = set("DEKRH")
    return sum(aa in charged for aa in seq) / len(seq)


def _mean_residue_prop(seq: str, prop: str, residue_props: dict) -> float:
    vals = [residue_props[aa][prop] for aa in seq if aa in residue_props]
    return sum(vals) / len(vals) if vals else 0.0


def compute_features(
    sequences: list[str],
    residue_props: dict,
) -> dict[str, np.ndarray]:
    n = len(sequences)
    feats: dict[str, list[float]] = {
        "length":             [],
        "mw":                 [],
        "gravy":              [],
        "pI":                 [],
        "charge_pH2":         [],
        "charge_pH7":         [],
        "hydrophobic_frac":   [],
        "aromatic_frac":      [],
        "proline_frac":       [],
        "charged_frac":       [],
        "mean_logp":          [],
        "mean_tpsa":          [],
        "mean_hbd":           [],
        "mean_hba":           [],
        "mean_rot_bonds":     [],
        "mean_rings":         [],
    }

    for seq in sequences:
        feats["length"].append(float(len(seq)))
        feats["mw"].append(peptide_mw(seq))
        feats["gravy"].append(gravy(seq))
        feats["pI"].append(pI(seq))
        feats["charge_pH2"].append(charge_at_ph(seq, 2.0))
        feats["charge_pH7"].append(charge_at_ph(seq, 7.0))
        feats["hydrophobic_frac"].append(hydrophobic_fraction(seq))
        feats["aromatic_frac"].append(aromatic_fraction(seq))
        feats["proline_frac"].append(proline_fraction(seq))
        feats["charged_frac"].append(charged_fraction(seq))
        feats["mean_logp"].append(_mean_residue_prop(seq, "logp", residue_props))
        feats["mean_tpsa"].append(_mean_residue_prop(seq, "tpsa", residue_props))
        feats["mean_hbd"].append(_mean_residue_prop(seq, "hbd", residue_props))
        feats["mean_hba"].append(_mean_residue_prop(seq, "hba", residue_props))
        feats["mean_rot_bonds"].append(_mean_residue_prop(seq, "rot_bonds", residue_props))
        feats["mean_rings"].append(_mean_residue_prop(seq, "rings", residue_props))

    return {k: np.array(v) for k, v in feats.items()}


# ── correlation helpers ───────────────────────────────────────────────────────

def correlations(x: np.ndarray, y: np.ndarray) -> dict[str, tuple[float, float]]:
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    kr, kp = stats.kendalltau(x, y)
    return {
        "pearson":  (pr, pp),
        "spearman": (sr, sp),
        "kendall":  (kr, kp),
    }


# ── reporting ─────────────────────────────────────────────────────────────────

def _fmt_corr(label: str, corrs: dict) -> str:
    lines = [f"  {label}"]
    for method, (r, p) in corrs.items():
        lines.append(f"    {method:10s}  r={r:+.4f}  p={p:.3e}")
    return "\n".join(lines)


def write_results(
    feature_corrs: dict[str, dict],
    split_name: str,
    n_samples: int,
    output_file: Path,
) -> None:
    lines = [
        "=" * 70,
        f"Biophysical Feature × Retention Time Correlations",
        f"Dataset : {HF_REPO}  |  Split : {split_name}  |  N = {n_samples}",
        f"Run at  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        f"{'Feature':<22} {'Pearson r':>10} {'Spearman r':>12} {'Kendall τ':>11}",
        "-" * 60,
    ]
    for feat, corrs in sorted(
        feature_corrs.items(), key=lambda x: -abs(x[1]["spearman"][0])
    ):
        pr = corrs["pearson"][0]
        sr = corrs["spearman"][0]
        kr = corrs["kendall"][0]
        lines.append(f"{feat:<22} {pr:>+10.4f} {sr:>+12.4f} {kr:>+11.4f}")

    lines += [
        "",
        "Detailed p-values",
        "-" * 60,
    ]
    for feat, corrs in sorted(
        feature_corrs.items(), key=lambda x: -abs(x[1]["spearman"][0])
    ):
        lines.append(_fmt_corr(feat, corrs))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines) + "\n")
    print(f"Results written to {output_file}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading SMILES map …")
    _load_smiles()
    residue_props = _rdkit_residue_props()

    print(f"Loading dataset from {HF_REPO} …")
    ds = hf_load_dataset(HF_REPO, "peptag")

    # Use the full train split for correlation analysis (largest sample)
    split = ds["train"]
    sequences: list[str] = split["Peptide"]
    rt: np.ndarray = np.array(split["B"], dtype=float)

    print(f"Computing features for {len(sequences)} peptides …")
    features = compute_features(sequences, residue_props)

    print("Computing correlations …")
    feature_corrs: dict[str, dict] = {}
    for feat_name, feat_vals in features.items():
        feature_corrs[feat_name] = correlations(feat_vals, rt)

    write_results(feature_corrs, "train", len(sequences), RESULTS_FILE)

    # Print a quick summary to stdout as well
    print()
    print(f"{'Feature':<22} {'Pearson r':>10} {'Spearman r':>12} {'Kendall τ':>11}")
    print("-" * 60)
    for feat, corrs in sorted(
        feature_corrs.items(), key=lambda x: -abs(x[1]["spearman"][0])
    ):
        pr = corrs["pearson"][0]
        sr = corrs["spearman"][0]
        kr = corrs["kendall"][0]
        print(f"{feat:<22} {pr:>+10.4f} {sr:>+12.4f} {kr:>+11.4f}")


if __name__ == "__main__":
    main()
