We thank the reviewer for the positive assessment of the dataset and for a set of specific, well-motivated questions. We address each comment and question below.

## Weaknesses

### Baseline selection

We wish to thank reviewer 8ABX for this comment. As another reviewer requested a broader set of baselines, we have added two additional benchmarked models, **PepLand** and **PeptideCLM-2**, both of which are pre-trained representation models that operate directly on SMILES strings rather than on canonical-residue tokenizations. Neither outperforms our existing best-performing model (DeepLC) on the primary regression task, and on stereo-discrimination specifically they are among the strongest candidates we have tested so far, though the improvement over chance remains modest. We believe this broader model selection addresses the concern raised here as well, and we will report the corresponding tables in the main text.

## Major aspects

### Chiral descriptors in "Stereochemistry in molecular machine learning"

We wish to thank reviewer 8ABX for this pointer that will no doubt reinforce our literature review on chirality. We will add a discussion of chiral-descriptor-based approaches to this section, citing https://doi.org/10.1002/cem.3037 alongside the representation-learning baselines already covered.

### Page 5, Line 147: side-chain stereochemistry (Ile, Thr)

We thank reviewer 8ABX for their concern on amino acids harboring more than one stereogenic center. Side-chain stereocenters in the case of Thr and Ile were explicitly encoded, not just the alpha-carbon. Our SMILES-generation code sets a fixed, chemically correct beta-carbon configuration for every side-chain-chiral residue, for example:

```
'I': '[C@@H](CC)C',   # L-Ile: beta-C is (S)
'T': '[C@@H](O)C',    # L-Thr: beta-C is (R)
```

To ensure that readers will not stumble upon the same question, we will clarify the text at Line 147 to state explicitly that both alpha-carbon and side-chain stereocenters (Ile, Thr) are set to their correct L-configuration throughout, so the isomeric SMILES fully specify stereochemistry rather than only the substituted alpha-carbon.

### Page 5, Line 152: split description

- **Why "synthesized independently" matters:** the test set is not a held-out subsample of the same synthesis batches used for train/val. It comes from a separate one-bead-one-compound (OBOC) library (`rawdata/test/`) synthesized on its own. This makes it a true held-out benchmark rather than a random split of the same underlying material. We agree with reviewer 8ABX that this was not described accurately enough and we will make this distinction explicit in the main text.

- **Test set size (~5.9%, 2,726 of 46,062 + 2,726 rows):** this was not a deliberately chosen split ratio. It reflects however much material that independent library batch yielded. We will state this plainly rather than implying it was a designed proportion, as discussed in the previous point.

- **Composition/length comparison, test vs. train+val:**

  | | Train+Val (n=46,062) | Test (n=2,726) |
  |---|---|---|
  | Terminal: R / K | 50.6% / 49.4% | 37.7% / 62.3% |
  | Phe-tag: L / D / label-free | 38.9% / 36.8% / 24.3% | 38.2% / 32.1% / 29.7% |
  | Length range present | 6–17 | 7, 9–11, 13–15 |
  | Dominant lengths | spread across 6–15 | 11-mer (29%), 13-mer (31%) |

  Phe-tagging proportions are similar across splits, but terminal scaffold and length distributions differ meaningfully, since the test library was synthesized independently rather than stratified to match train/val. We see this as a strength for measuring genuine generalization (the model cannot rely on matching the train/val length or scaffold distribution), but agree with reviewer 8ABX that it should be reported explicitly and will add a version of this table to the main text or appendix.

### Page 6, Line 202: ESM tokenizer nuance

We agree with this nuance brought forth by reviewer 8ABX and will revise the text. To be precise about what we do: the new 'f' (D-Phe) token's embedding is **initialized as a copy of the L-Phe ('F') embedding**, and only that single vector (plus the MLP head) is subsequently fine-tuned; the rest of the backbone stays constant. This confirms the reviewer's point exactly, initializing from L-Phe is a heuristic choice we made, not something the model itself provides, and it is not obvious this is the best choice (e.g., initializing from L-Gly, or a different residue, is equally defensible a priori). We will revise Line 202 to state this precisely and to note that this heuristic is specific to token-level protein language models such as ESM; representations that are not built on a fixed canonical-residue vocabulary would not require such a patch in the first place, so the open transferability question we pose is specifically about token-level protein LMs, not representation learning in general.

### Ablation: canonical-only vs. non-canonical-only training

We appreciate this suggestion from reviewer 8ABX as a way to isolate whether the strong overall regression performance depends on the presence of D-Phe/epimeric structure in training, rather than reflecting genuine generalization. We have already run a canonical-only ("natural") ablation: removing every D-Phe-tagged (non-canonical) peptide from train/val/test and retraining our benchmark models on natural, all-L-amino-acid sequences alone (mean over 3 seeds). Despite training on notably less data (train: 41,455 → 26,199; test: 2,726 → 1,851, roughly a 37% reduction in each), performance is comparable or better:

| Model | Pearson *r*, full (n=2,726) | Pearson *r*, natural-only (n=1,851) | *R²*, full (n=2,726) | *R²*, natural-only (n=1,851) |
|---|---|---|---|---|
| GIN | 0.79 | 0.77 | 0.56 | 0.53 |
| DeepLC | 0.80 | 0.83 | 0.58 | 0.59 |
| Transformer | 0.75 | 0.74 | 0.51 | 0.48 |
| DeepRT-CapsNet | 0.76 | 0.77 | 0.51 | 0.53 |
| Morgan FP MLP | 0.51 | 0.56 | 0.16 | 0.28 |

Overall regression performance on the natural-only subset is comparable to, and in some cases slightly better than, performance on the full dataset, despite roughly a third less training and test data. This indicates the models' general retention-time prediction ability is not an artifact of the D-Phe/epimeric-pair structure present in the full dataset; it holds, and if anything improves, when every non-canonical, stereochemically ambiguous example (and the corresponding data) is removed.

To more directly test the complementary case, we will additionally run this ablation the other way, restricting training and evaluation to non-canonical (D-Phe-containing) peptides only, and report whether overall regression performance also holds on that subset. We will report this soon.

### Learning curve

Similarly to reviewer 8ABX, we were also interested in this question and we have already run this exact experiment to answer it. We trained Pretrained GIN (our most reliable model on the diastereomer-discrimination metric, both in accuracy and seed-to-seed stability) on log-spaced fractions of the training set, from 1% (415 examples) to 100% (41,455 examples), 3 seeds per fraction, and evaluated both overall regression performance and accuracy at each point.

**Table: Pretrained GIN learning curve (mean ± std over 3 seeds).**

| Train fraction | Train size | RMSE | Pearson *r* | Accuracy |
|---|---|---|---|---|
| 1% | 415 | 7.63 ± 0.02 | 0.71 ± 0.01 | 0.634 ± 0.005 |
| 2% | 829 | 7.91 ± 0.36 | 0.70 ± 0.03 | 0.613 ± 0.029 |
| 5% | 2,073 | 8.03 ± 0.21 | 0.69 ± 0.03 | 0.624 ± 0.017 |
| 10% | 4,146 | 8.06 ± 0.36 | 0.67 ± 0.04 | 0.629 ± 0.002 |
| 20% | 8,291 | 7.72 ± 0.50 | 0.70 ± 0.03 | 0.638 ± 0.005 |
| 40% | 16,582 | 7.84 ± 0.34 | 0.70 ± 0.03 | 0.645 ± 0.013 |
| 70% | 29,018 | 7.40 ± 0.17 | 0.74 ± 0.01 | 0.634 ± 0.010 |
| 100% | 41,455 | 7.65 ± 0.12 | 0.73 ± 0.02 | 0.651 ± 0.015 |

Accuracy is essentially flat (0.61–0.65) across nearly two orders of magnitude of training data, while overall regression performance shows only a mild, noisy improvement with more data. If the failure to discriminate D-/L-stereoisomers were primarily a data-scarcity problem, we would expect a clear upward trend in accuracy with training set size, as is visible to a modest degree in the regression metrics. Instead, the model plateaus almost immediately. This supports our interpretation that the limitation is architectural/representational rather than a matter of insufficient training data, and we will add this figure and discussion to the paper.

## Minor aspects

### Epimers vs. diastereomeric pairs

We agree with reviewer 8ABX that the "epimer" terminology is more accurate than "diastereomers" and will replace "diastereomeric pairs" with "epimers" (or "epimeric pairs") throughout the manuscript, since our pairs differ at exactly one stereocenter, which is the more precise term in the organic chemistry sense.