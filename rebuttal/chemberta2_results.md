# Additional baseline: ChemBERTa-2 (as promised during discussion)

We committed to adding **ChemBERTa-2** as a baseline before the end of the discussion period. That run is now complete, and we report the results here.

ChemBERTa-2 (Ahmad et al., 2022) is a RoBERTa encoder pretrained on ~77M PubChem SMILES under two objectives: masked language modelling (**MLM**) and multi-task regression over 200 computed molecular properties (**MTR**). We benchmark both headline 77M checkpoints released by DeepChem. It fills a genuine gap in our suite: every other SMILES-level pretrained baseline we run (PepLand, PeptideCLM-2) is peptide-specific, whereas ChemBERTa-2 is a *general-purpose small-molecule* SMILES language model trained on a corpus containing a very large number of chiral centres. It is therefore the most direct test available of the reviewer's canonical-to-non-canonical transfer question: does broad small-molecule pretraining transfer to stereochemical discrimination in peptides?

Protocol is identical to our other frozen-representation baselines (PepLand, PeptideCLM-2): the backbone is frozen, an MLP head is trained on mean-pooled token embeddings, 10 seeds, mean ± std reported. Rows in *italics* are reproduced from the tables already shown, for context only; the new results are in **bold**.

**Table A. Overall regression performance on %B prediction (test set, mean ± std over 10 seeds).**

| Model | Pearson *r* | Spearman *ρ* | Kendall *τ* | *R²* | RMSE | MAE |
|---|---|---|---|---|---|---|
| **ChemBERTa-2-MLM** | 0.79 ± 0.04 | 0.83 ± 0.04 | 0.66 ± 0.04 | 0.53 ± 0.08 | 7.04 ± 0.60 | 4.78 ± 0.40 |
| **ChemBERTa-2-MTR** | 0.72 ± 0.04 | 0.75 ± 0.04 | 0.58 ± 0.03 | 0.46 ± 0.06 | 7.60 ± 0.41 | 5.29 ± 0.28 |
| *Pretrained PepLand (ref.)* | *0.79 ± 0.01* | *0.84 ± 0.01* | *0.66 ± 0.01* | *0.49 ± 0.04* | *7.38 ± 0.29* | *5.00 ± 0.28* |
| *PeptideCLM-2-Hybrid (ref.)* | *0.75 ± 0.02* | *0.79 ± 0.02* | *0.61 ± 0.02* | *0.49 ± 0.04* | *7.36 ± 0.26* | *5.01 ± 0.18* |
| *DeepLC, best overall (ref.)* | *0.80 ± 0.02* | *0.85 ± 0.02* | *0.67 ± 0.02* | *0.58 ± 0.03* | *6.68 ± 0.24* | *4.68 ± 0.28* |

ChemBERTa-2-MLM is a competitive regressor, matching PepLand on Pearson *r* (0.786 for both) and ranking third of the thirteen models on *R²* and RMSE, behind only DeepLC and GIN. It does not exceed DeepLC on any regression metric. The MTR variant is consistently weaker despite its property-prediction pretraining objective, and its seed-to-seed variance is noticeably higher than the peptide-specific baselines' (±0.04 on Pearson *r* vs. ±0.01 for PepLand).

**Table B. Diastereomer-pair (D-Phe/L-Phe) discrimination, test set (mean ± std over 10 seeds).** 1.0 = perfect; ΔAUC 0.5 / Δ*r* 0 = chance.

| Model | Pairwise Acc. | Δ Pearson | Δ AUC |
|---|---|---|---|
| **ChemBERTa-2-MLM** | 0.63 ± 0.02 | 0.17 ± 0.02 | 0.57 ± 0.02 |
| **ChemBERTa-2-MTR** | 0.63 ± 0.03 | 0.18 ± 0.03 | 0.61 ± 0.03 |
| *Pretrained PepLand (ref.)* | *0.64 ± 0.00* | *0.19 ± 0.01* | *0.61 ± 0.01* |
| *PeptideCLM-2-Hybrid (ref.)* | *0.64 ± 0.01* | *−0.12 ± 0.05* | *0.42 ± 0.04* |
| *DeepLC (ref.)* | *0.60 ± 0.11* | *−0.05 ± 0.12* | *0.43 ± 0.11* |

The result is consistent with our central claim, and we think it strengthens it. ChemBERTa-2's pretraining corpus is not larger than PeptideCLM-2's (~77M PubChem molecules vs. 100M+), but it is far more chemically diverse: it spans general drug-like chemical space rather than peptides, and therefore covers a much broader range of stereochemical contexts. Its input SMILES also fully and explicitly specify the inverted stereocentre, and we verified that the stereo descriptor survives tokenization (see the caveat below). Even so, it reaches only ΔAUC 0.61 (MTR) and 0.57 (MLM) against a 0.5 chance baseline, with pairwise accuracy at 0.63 — squarely in the same band as every other model tested. So the gap is explained neither by a representation that discards the stereo descriptor, nor by pretraining that lacks stereochemical variety: broad general-purpose chemical pretraining does not transfer to this task either. Notably, the *best* discriminators in our suite now come from two very different pretraining regimes — peptide-specific (PepLand) and general small-molecule (ChemBERTa-2-MTR) — and both stall at the same ΔAUC ≈ 0.61, which we read as evidence that the limitation is architectural/representational rather than a matter of pretraining domain or scale.

We also note one correction to our earlier response to reviewer hzjD. We wrote that PepLand "achieves the highest ΔAUC of any model." With ChemBERTa-2-MTR at 0.61 ± 0.03 against PepLand's 0.61 ± 0.01, the two are statistically indistinguishable, and we will revise that sentence to say that PepLand and ChemBERTa-2-MTR are jointly the strongest at ΔAUC ≈ 0.61 rather than claiming a single winner.

**Natural-only subset.** For consistency with the canonical-only ablation reported to reviewer 8ABX, we also ran both variants on the natural (canonical-amino-acid-only) configuration, 3 seeds:

| Model | Pearson *r*, full | Pearson *r*, natural-only | *R²*, full | *R²*, natural-only |
|---|---|---|---|---|
| ChemBERTa-2-MLM | 0.79 | 0.79 | 0.53 | 0.51 |
| ChemBERTa-2-MTR | 0.72 | 0.77 | 0.46 | 0.43 |

As with the other models, general regression performance holds on the canonical-only subset, so it is not an artifact of the epimeric-pair structure of the full dataset.

## A reproducibility caveat worth recording

While implementing this baseline we found a tokenizer issue that we want to flag explicitly, because it would silently manufacture a *false confirmation* of our own thesis, and because anyone reproducing a ChemBERTa-2 result on a stereochemistry task is likely to hit it.

The DeepChem ChemBERTa-2 repositories ship an atom-level SMILES `vocab.json` — it contains multi-character bracket tokens such as `[C@@H]`, `[nH]` and `[NH3+]` — but the accompanying `merges.txt` contains only a version header, i.e. zero merge rules, while `tokenizer_config.json` declares a byte-level BPE. Loaded the obvious way, via `AutoTokenizer.from_pretrained`, the resulting tokenizer can only ever emit the *single-character* vocabulary entries and silently discards any character that is not one of them. Because `[`, `]`, `@` and `H` are not single-character entries, stereo descriptors are dropped outright:

```
N[C@@H](Cc1ccccc1)C(=O)O   ->   N C ( C c 1 c c c c c 1 ) C ( = O ) O
```

No `[UNK]` token is emitted to signal the loss. Under this tokenization, both members of **every** diastereomeric pair in our benchmark map to byte-identical input ids, so the model is mathematically incapable of discriminating them and scores exactly chance — a result that would look like a clean confirmation of our central claim while in fact measuring nothing but a tokenizer defect.

We therefore tokenize with the atom-wise SMILES regex these vocabularies were built for (Schwaller et al., 2018; the same pattern used by `deepchem.feat.SmilesTokenizer`), mapping tokens through each checkpoint's own `vocab.json`. On the StereoPep SMILES we verified that this tokenization is lossless (concatenating the tokens reconstructs the input exactly), that vocabulary coverage is 100.0% with 0 out-of-vocabulary tokens across 8,189,468 tokens, that all 8,881 diastereomeric pairs receive distinct id sequences, and that the longest sequence is 289 tokens against the model's 512-position limit, so nothing is truncated. Our benchmark asserts each of these properties and aborts rather than reporting a chance-level number if any fails.

The numbers in Tables A and B are from the corrected tokenization. We also ran the same check on our other two SMILES-level pretrained baselines to rule out an analogous silent failure there: across all 542 held-out diastereomeric pairs, PepLand and PeptideCLM-2-Hybrid both produce distinct embeddings for the D- and L-forms (0 identical pairs in each case). The near- or below-chance discrimination we report for those models is therefore a property of the models, not an artifact of our preprocessing. We will add a short version of this note to the appendix so the distinction is on the record.
