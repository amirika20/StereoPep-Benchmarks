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

── Tokenization: we use the checkpoint's own tokenizer, deliberately ──

`AutoTokenizer.from_pretrained` is the correct tokenizer for these
checkpoints, and it *discards stereochemistry*. That is a property of the
released model, not a bug on our side, and reproducing it faithfully is the
whole point of this benchmark.

The mechanics: `vocab.json` is an atom-level SMILES vocabulary containing
multi-character entries such as `[C@@H]`, `[nH]` and `[NH3+]`, but
`merges.txt` holds only a version header (zero merge rules) while
`tokenizer_config.json` declares a byte-level BPE. A BPE with no merges can
only emit single-character vocabulary entries, and characters absent from
the vocabulary are dropped silently. `[`, `]`, `@`, `H`, `+` and `-` are all
absent, so:

    N[C@@H](Cc1ccccc1)C(=O)O   ->   N C ( C c 1 c c c c c 1 ) C ( = O ) O

An earlier version of this script assumed the multi-character entries were
the intended tokenization and reimplemented the atom-wise SMILES regex
(Schwaller et al. 2018) to reach them. That was wrong. The published weights
were never trained against those token ids, as three independent checks
show:

  1. Masked-LM probe. Under atom-level tokenization the MLM head's top-1
     accuracy on multi-character positions is 0.000 (0/180 trials) and true
     tokens rank in the hundreds with p ~ 0 (`[nH]` in an indole context:
     rank 369). Under the native tokenizer the same model scores 0.986
     (355/360). A model trained on atom-level tokens cannot behave this way.
  2. Embedding geometry. Multi-character rows have a suspiciously tight norm
     distribution (2.391 +- 0.053) versus single-character rows
     (3.880 +- 0.423), consistent with never receiving gradient.
  3. Context degradation. Atom-level tokenization also drops single-character
     top-1 accuracy (0.550 vs 0.986), as expected when the surrounding
     sequence is out of distribution.

`vocab.json`'s multi-character entries are therefore vestigial. Feeding them
to the model means feeding untrained embedding rows, so the resulting numbers
would characterize nothing.

Consequence for this benchmark: both members of every diastereomeric pair
become byte-identical input ids, so their embeddings are identical, predicted
deltas are exactly zero, and the discrimination metrics degenerate --
ΔAUC exactly 0.5, ordering accuracy 0.0, Δ-correlations undefined (nan).
This is *incapability by construction*, not a noisy measurement, and it is
the honest result to report. `characterize_tokenization` quantifies the
collapse and records it in the cache so downstream scripts can report it as
a measured property rather than rediscovering it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset as hf_load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Reuse the model registry / config from the main benchmark script
import sys
sys.path.insert(0, str(Path(__file__).parent))
import chemberta2_embedding as cb  # noqa: E402

WEIGHTS_DIR = Path(__file__).parent / "pretrained_weights"


def load_model(model_key: str):
    cfg = cb.MODELS[model_key]
    print(f"[ChemBERTa-2] Loading {cfg['display_name']} ({cfg['hf_name']}) on {cb.DEVICE} …")
    model = AutoModel.from_pretrained(cfg["hf_name"]).to(cb.DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(cfg["hf_name"])
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


def characterize_tokenization(tok, smiles: list[str], pairs: list[tuple[str, str]]) -> dict:
    """Quantify what the checkpoint's tokenizer does to our inputs.

    Deliberately does NOT abort on stereochemical collapse: that collapse is
    the finding. Aborts only on truncation, which would silently corrupt
    inputs in a way unrelated to the phenomenon under study.
    """
    lengths = [len(tok(s)["input_ids"]) for s in smiles]
    longest = max(lengths)

    # Does the tokenization preserve the input string?
    lossy = sum(1 for s in smiles[:2000]
                if tok.decode(tok(s)["input_ids"], skip_special_tokens=True) != s)

    # Do the two members of each diastereomeric pair survive as distinct inputs?
    collapsed = sum(1 for a, b in pairs if tok(a)["input_ids"] == tok(b)["input_ids"])

    print(f"[tokenizer] {len(smiles)} SMILES | longest {longest} tokens "
          f"(limit {cb.MAX_LENGTH}) | median {sorted(lengths)[len(lengths)//2]}")
    print(f"[tokenizer] round-trip mismatches: {lossy}/2000 sampled "
          f"({'lossy — characters are dropped' if lossy else 'lossless'})")
    print(f"[tokenizer] diastereomeric pairs with IDENTICAL input ids: "
          f"{collapsed}/{len(pairs)} ({100 * collapsed / max(1, len(pairs)):.1f}%)")
    if collapsed:
        print("[tokenizer] --> stereochemistry is discarded by this checkpoint's tokenizer, "
              "so D-/L-forms are indistinguishable to the model by construction. "
              "Discrimination metrics will degenerate (ΔAUC 0.5, accuracy 0.0, "
              "Δ-correlations nan). This is the expected, honest result — see the "
              "module docstring.")

    if longest > cb.MAX_LENGTH:
        raise RuntimeError(
            f"Longest sequence ({longest}) exceeds the model's position limit "
            f"({cb.MAX_LENGTH}); embeddings would be silently truncated."
        )
    return {
        "tokenizer": "native_autotokenizer",
        "longest_tokens": longest,
        "lossy_roundtrips_per_2000": lossy,
        "stereo_pairs_checked": len(pairs),
        "stereo_pairs_collapsed": collapsed,
    }


@torch.no_grad()
def embed_all(model, tok, smiles: list[str]) -> dict[str, torch.Tensor]:
    """Mean-pool the final hidden states over non-padding tokens."""
    embeddings: dict[str, torch.Tensor] = {}
    for i in tqdm(range(0, len(smiles), cb.ENCODE_BATCH), desc="Embedding"):
        batch = smiles[i : i + cb.ENCODE_BATCH]
        try:
            inputs = tok(batch, return_tensors="pt", padding=True,
                         truncation=True, max_length=cb.MAX_LENGTH)
            inputs = {k: v.to(cb.DEVICE) for k, v in inputs.items()}
            hidden = model(**inputs).last_hidden_state          # (B, T, H)
            mask   = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = ((hidden * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu()
        except Exception as e:
            print(f"[WARNING] Batch starting at index {i} failed ({e}); "
                  f"falling back to per-item embedding")
            for smi in batch:
                try:
                    inp = tok([smi], return_tensors="pt", padding=True,
                              truncation=True, max_length=cb.MAX_LENGTH)
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

        diag = characterize_tokenization(tok, all_smiles, pairs)

        embeddings = embed_all(model, tok, all_smiles)
        print(f"[done] Embedded {len(embeddings)}/{len(all_smiles)} SMILES")

        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"hf_name": cfg["hf_name"], "embed_dim": embed_dim,
                    "embeddings": embeddings, **diag}, out_path)
        print(f"Saved to {out_path}")

        del model, tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
