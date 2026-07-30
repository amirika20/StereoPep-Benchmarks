"""
Precompute step for the ChemBERTa-2 benchmark.

Embeds every unique SMILES string used anywhere in the StereoPep dataset
(main splits + all pair splits) ONCE per model variant and caches
{SMILES: embedding} to
benchmarks/pretrained_weights/chemberta2_{model_key}_embeddings.pt.

chemberta2_embedding.py then loads that cache instead of the live model, so
a full multi-seed sweep pays the backbone forward pass once rather than once
per seed. Same arrangement as peptideclm2_precompute_embeddings.py.

Usage:
  python benchmarks/chemberta2_precompute_embeddings.py --model mtr_77m
  python benchmarks/chemberta2_precompute_embeddings.py --model all

── Why this file hand-rolls the tokenizer ──

The DeepChem ChemBERTa-2 repos ship an atom-level SMILES `vocab.json`
(single-character atoms/bonds, plus multi-character bracket tokens such as
`[C@@H]`, `[nH]`, `[NH3+]`, and ring-closure tokens `%10`-`%99`) together
with a `merges.txt` that contains only a version header — zero merge rules.
Their `tokenizer_config.json` nevertheless declares a byte-level BPE, so
`AutoTokenizer.from_pretrained` yields a tokenizer that can only emit the
*single-character* vocabulary entries and silently discards anything else.
Because `[`, `]`, `@` and `H` are not single-character entries, stereo
descriptors vanish:

    N[C@@H](Cc1ccccc1)C(=O)O   ->   N C ( C c 1 c c c c c 1 ) C ( = O ) O

Both members of every diastereomeric pair then produce identical input ids,
which would make ChemBERTa-2 score exactly chance on stereo discrimination
for reasons of our own tokenization rather than anything about the model.

We therefore tokenize with the atom-wise SMILES regex these vocabularies
were built for (Schwaller et al. 2018, "Found in Translation"; the same
pattern used by `deepchem.feat.SmilesTokenizer`) and map tokens through the
checkpoint's own `vocab.json`. Implemented here rather than pulled from
`deepchem` to avoid adding a heavyweight dependency for one regex.

`verify_tokenization` asserts the three properties we need before spending
GPU time, and the run aborts if any fails:
  1. lossless      — ''.join(tokens) reconstructs the input exactly
  2. in-vocabulary — no token maps to [UNK]
  3. stereo-aware  — D-/L-Phe pairs get different id sequences
On the StereoPep SMILES all three hold (100.0% coverage, 0 OOV, 0 pair
collisions; longest sequence 244 tokens against a 512 position limit).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from datasets import load_dataset as hf_load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoModel

# Reuse the model registry / config from the main benchmark script
import sys
sys.path.insert(0, str(Path(__file__).parent))
import chemberta2_embedding as cb  # noqa: E402

WEIGHTS_DIR = Path(__file__).parent / "pretrained_weights"

# Atom-wise SMILES tokenization regex (Schwaller et al. 2018). Order matters:
# the bracket alternative comes first so `[C@@H]` is consumed whole, and the
# two-character element symbols (`Br`, `Cl`) precede their one-character
# prefixes so they are not split into `B`+`r` / `C`+`l`.
SMILES_TOKEN_PATTERN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p"
    r"|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"
)


class SmilesAtomTokenizer:
    """Atom-wise SMILES tokenizer over a ChemBERTa-2 checkpoint's own vocab.json.

    Emits `[CLS] … [SEP]` and pads with `[PAD]`, matching the DeepChem
    SmilesTokenizer layout these checkpoints were trained with (the vocab
    reserves PAD=0, UNK=11, CLS=12, SEP=13, MASK=14).
    """

    def __init__(self, hf_name: str):
        with open(hf_hub_download(hf_name, "vocab.json")) as f:
            self.vocab: dict[str, int] = json.load(f)
        missing = [t for t in ("[PAD]", "[UNK]", "[CLS]", "[SEP]") if t not in self.vocab]
        if missing:
            raise ValueError(f"{hf_name} vocab.json lacks required special tokens: {missing}")
        self.pad_id = self.vocab["[PAD]"]
        self.unk_id = self.vocab["[UNK]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]

    def tokenize(self, smiles: str) -> list[str]:
        return SMILES_TOKEN_PATTERN.findall(smiles)

    def encode(self, smiles: str, max_length: int) -> list[int]:
        ids = [self.vocab.get(t, self.unk_id) for t in self.tokenize(smiles)]
        # Reserve two slots for [CLS] / [SEP].
        ids = ids[: max_length - 2]
        return [self.cls_id, *ids, self.sep_id]

    def batch_encode(self, batch: list[str], max_length: int) -> dict[str, torch.Tensor]:
        seqs = [self.encode(s, max_length) for s in batch]
        width = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), width), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            attention_mask[i, : len(s)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def verify_tokenization(tok: SmilesAtomTokenizer, smiles: list[str], pairs: list[tuple[str, str]]) -> None:
    """Fail fast if tokenization is lossy, out-of-vocabulary, or stereo-blind."""
    n_tok = n_oov = 0
    oov_examples: set[str] = set()
    longest = 0
    for s in smiles:
        toks = tok.tokenize(s)
        if "".join(toks) != s:
            raise RuntimeError(
                f"Tokenization is lossy — ''.join(tokens) != input for:\n  {s}\n"
                f"  got: {''.join(toks)}"
            )
        longest = max(longest, len(toks) + 2)
        n_tok += len(toks)
        for t in toks:
            if t not in tok.vocab:
                n_oov += 1
                oov_examples.add(t)
    if n_oov:
        raise RuntimeError(
            f"{n_oov}/{n_tok} tokens are out of vocabulary "
            f"(examples: {sorted(oov_examples)[:10]}). The vocab is supposed to "
            f"cover the StereoPep alphabet exactly; investigate before embedding."
        )

    collisions = sum(1 for a, b in pairs if tok.encode(a, cb.MAX_LENGTH) == tok.encode(b, cb.MAX_LENGTH))
    if collisions:
        raise RuntimeError(
            f"{collisions}/{len(pairs)} diastereomeric pairs tokenize identically — "
            f"stereochemistry is being lost, so the benchmark would measure chance "
            f"performance for tokenization reasons rather than model reasons. "
            f"See this module's docstring."
        )

    print(f"[verify] {len(smiles)} SMILES: lossless, {n_tok} tokens, 0 OOV, "
          f"longest sequence {longest} tokens (limit {cb.MAX_LENGTH})")
    print(f"[verify] {len(pairs)} diastereomeric pairs tokenize distinctly (0 collisions)")
    if longest > cb.MAX_LENGTH:
        raise RuntimeError(
            f"Longest sequence ({longest}) exceeds the model's position limit "
            f"({cb.MAX_LENGTH}); embeddings would be silently truncated."
        )


def load_model(model_key: str):
    cfg = cb.MODELS[model_key]
    print(f"[ChemBERTa-2] Loading {cfg['display_name']} ({cfg['hf_name']}) on {cb.DEVICE} …")
    model = AutoModel.from_pretrained(cfg["hf_name"]).to(cb.DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = SmilesAtomTokenizer(cfg["hf_name"])
    return model, tok, model.config.hidden_size


def collect_all_smiles() -> tuple[list[str], list[tuple[str, str]]]:
    """Return (all unique SMILES, diastereomeric pairs) across every split."""
    print("[data] Loading StereoPep dataset splits …")
    ds              = hf_load_dataset(cb.HF_REPO, "StereoPep")
    stereo          = hf_load_dataset(cb.HF_REPO, "diastereomer_pairs")["diastereomer_pairs"]
    stereo_trainval = hf_load_dataset(cb.HF_REPO, "diastereomer_pairs")["diastereomer_pairs_trainval"]
    tag_pairs       = hf_load_dataset(cb.HF_REPO, "terminal_tag_pairs")["terminal_tag_pairs"]
    sub_pairs       = hf_load_dataset(cb.HF_REPO, "point_mutant_pairs")["point_mutant_pairs"]

    smiles: set[str] = set()
    for split in ("train", "val", "test"):
        smiles.update(ds[split]["SMILES"])
    for pair_ds, cols in (
        (stereo,          ("SMILES_f", "SMILES_F")),
        (stereo_trainval, ("SMILES_f", "SMILES_F")),
        (tag_pairs,       ("SMILES_untagged", "SMILES_tagged")),
        (sub_pairs,       ("SMILES_1", "SMILES_2")),
    ):
        for col in cols:
            smiles.update(pair_ds[col])

    pairs = [
        *zip(stereo["SMILES_f"], stereo["SMILES_F"]),
        *zip(stereo_trainval["SMILES_f"], stereo_trainval["SMILES_F"]),
    ]
    print(f"[data] {len(smiles)} unique SMILES collected, {len(pairs)} diastereomeric pairs")
    return sorted(smiles), pairs


@torch.no_grad()
def embed_all(model, tok: SmilesAtomTokenizer, smiles: list[str]) -> dict[str, torch.Tensor]:
    """Mean-pool the final hidden states over non-padding tokens."""
    embeddings: dict[str, torch.Tensor] = {}
    for i in tqdm(range(0, len(smiles), cb.ENCODE_BATCH), desc="Embedding"):
        batch = smiles[i : i + cb.ENCODE_BATCH]
        try:
            inputs = tok.batch_encode(batch, cb.MAX_LENGTH)
            inputs = {k: v.to(cb.DEVICE) for k, v in inputs.items()}
            hidden = model(**inputs).last_hidden_state          # (B, T, H)
            mask   = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = ((hidden * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu()
        except Exception as e:
            print(f"[WARNING] Batch starting at index {i} failed ({e}); "
                  f"falling back to per-item embedding")
            for smi in batch:
                try:
                    inp = tok.batch_encode([smi], cb.MAX_LENGTH)
                    inp = {k: v.to(cb.DEVICE) for k, v in inp.items()}
                    h = model(**inp).last_hidden_state
                    m = inp["attention_mask"].unsqueeze(-1).to(h.dtype)
                    embeddings[smi] = ((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu()[0]
                except Exception as e2:
                    print(f"[WARNING] Skipping unembeddable SMILES: {smi} ({e2})")
            continue
        for smi, emb in zip(batch, pooled):
            embeddings[smi] = emb
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute ChemBERTa-2 embeddings")
    parser.add_argument(
        "--model", default="mtr_77m", metavar="MODEL",
        help="Model(s) to precompute embeddings for. Same keys as chemberta2_embedding.py "
             "(comma-separated, or 'all'). Default: mtr_77m",
    )
    args = parser.parse_args()

    if args.model.strip().lower() == "all":
        model_keys = cb.ALL_MODELS
    else:
        model_keys = [k.strip() for k in args.model.split(",")]
        unknown = [k for k in model_keys if k not in cb.MODELS]
        if unknown:
            parser.error(f"Unknown model(s): {unknown}\nChoose from: {', '.join(cb.ALL_MODELS)}, all")

    print(f"Device: {cb.DEVICE}")
    all_smiles, pairs = collect_all_smiles()

    for model_key in model_keys:
        cfg = cb.MODELS[model_key]
        out_path = WEIGHTS_DIR / f"chemberta2_{model_key}_embeddings.pt"
        print(f"\n{'='*60}\nModel: {cfg['display_name']}\n{'='*60}")

        model, tok, embed_dim = load_model(model_key)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Backbone parameters: {n_params:,} (frozen)  |  embed_dim={embed_dim}")

        verify_tokenization(tok, all_smiles, pairs)

        embeddings = embed_all(model, tok, all_smiles)
        print(f"[done] Embedded {len(embeddings)}/{len(all_smiles)} SMILES")

        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"hf_name": cfg["hf_name"], "embed_dim": embed_dim,
                    "tokenizer": "smiles_atom_regex", "embeddings": embeddings}, out_path)
        print(f"Saved to {out_path}")

        del model, tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
