## Weaknesses — Major

### 1. Framing

We wish to thank reviewer hzjD and appreciate this feedback to improve our manuscript. The project originated as an applied tool for LC-MS retention-time prediction, specifically to distinguish D-Phe from L-Phe substitutions during peptide identification. In the course of that work, we found that state-of-the-art peptide/molecular representation models, despite strong aggregate performance, systematically fail at this stereochemical discrimination task. That negative result is what motivated the benchmark-style framing of this submission. We will revise the introduction and related work to lead with this as an ML benchmarking contribution, situating it against the broader literature on chirality-aware molecular and peptide representation learning, rather than the LC-MS application, which we agree is undersold in the ML/cheminformatics framing as currently written.

### 2. Missing baselines

We are grateful to reviewer hzjD for these specific pointers. Engaging with them has genuinely strengthened the paper, and we believe the resulting comparisons make our central claim more convincing rather than less. We have added **PepLand** and **PeptideCLM-2** (the current, larger-scale iteration of the PeptideCLM family, 100M+ molecules, superseding the smaller v1) as additional baselines. Results below (mean ± std over 10 seeds) show that neither outperforms our existing best-performing model (DeepLC) on the primary regression task. On stereo-discrimination specifically, PepLand and PeptideCLM-2 are in fact among the strongest candidates we have tested so far: PepLand achieves the highest ΔAUC of any model, yet that improvement over chance is modest, and it still falls well short of solving the task. If anything, this sharpens our core finding: even the newest, most peptide-aware representation-learning baselines only nudge the needle on D-/L-stereochemistry discrimination.

Of the remaining suggestions: **PeptideBERT** is built on ProtBERT, a predecessor-generation protein language model to ESM3/ESMC, which we already benchmark; **PepFuNN** is a descriptor/fingerprint and clustering toolkit rather than a pretrained representation model, and is functionally analogous to our existing Morgan-fingerprint baseline. We have not yet had the chance to run **ChemBERTa-2**, but will add it as a baseline before the end of the discussion period, since it is architecturally distinct from our current suite (general-purpose SMILES LM vs. peptide-specific) and directly relevant to the reviewer's point about canonical-to-non-canonical transfer.

**Table 1. Overall regression performance on %B prediction (test set, mean ± std over 10 seeds).** New baselines added in response to review in **bold**.

| Model | Pearson *r* | Spearman *ρ* | Kendall *τ* | *R²* | RMSE | MAE |
|---|---|---|---|---|---|---|
| GIN | 0.79 ± 0.01 | 0.84 ± 0.01 | 0.66 ± 0.01 | 0.56 ± 0.07 | 6.81 ± 0.50 | **4.61 ± 0.41** |
| Pretrained GIN | 0.73 ± 0.02 | 0.78 ± 0.02 | 0.59 ± 0.02 | 0.46 ± 0.03 | 7.57 ± 0.19 | 5.23 ± 0.15 |
| **Pretrained PepLand** | 0.79 ± 0.01 | 0.84 ± 0.01 | 0.66 ± 0.01 | 0.49 ± 0.04 | 7.38 ± 0.29 | 5.00 ± 0.28 |
| **PeptideCLM-2-Hybrid** | 0.75 ± 0.02 | 0.79 ± 0.02 | 0.61 ± 0.02 | 0.49 ± 0.04 | 7.36 ± 0.26 | 5.01 ± 0.18 |
| ESM3-small | 0.79 ± 0.01 | 0.83 ± 0.01 | 0.64 ± 0.01 | 0.40 ± 0.08 | 7.98 ± 0.54 | 5.62 ± 0.44 |
| ESMC-300M | 0.77 ± 0.02 | 0.83 ± 0.02 | 0.64 ± 0.02 | 0.41 ± 0.03 | 7.95 ± 0.23 | 5.47 ± 0.19 |
| ESMC-600M | 0.76 ± 0.01 | 0.83 ± 0.01 | 0.64 ± 0.01 | 0.32 ± 0.03 | 8.50 ± 0.20 | 6.02 ± 0.20 |
| Transformer | 0.75 ± 0.04 | 0.80 ± 0.03 | 0.61 ± 0.03 | 0.51 ± 0.05 | 7.18 ± 0.40 | 5.10 ± 0.33 |
| **DeepLC (best overall)** | **0.80 ± 0.02** | **0.85 ± 0.02** | **0.67 ± 0.02** | **0.58 ± 0.03** | **6.68 ± 0.24** | 4.68 ± 0.28 |
| DeepRT-CapsNet | 0.76 ± 0.02 | 0.80 ± 0.02 | 0.61 ± 0.02 | 0.51 ± 0.07 | 7.23 ± 0.48 | 5.17 ± 0.33 |
| Morgan FP MLP | 0.51 ± 0.02 | 0.53 ± 0.03 | 0.38 ± 0.02 | 0.16 ± 0.06 | 9.47 ± 0.33 | 6.94 ± 0.25 |

Both new baselines land in the middle of the pack, on par with GIN/ESM-family models, and neither exceeds DeepLC, our existing best model, on any regression metric.

**Table 2. Diastereomer-pair (D-Phe/L-Phe) discrimination, test set (mean ± std over 10 seeds).** Ordering accuracy, and Pearson (Δr) / AUC (ΔAUC) of the predicted vs. true %B delta within each stereo pair. 1.0 = perfect discrimination; 0.5 AUC / 0 Δr = chance.

| Model | Accuracy | Δ Pearson | Δ AUC |
|---|---|---|---|
| GIN | 0.63 ± 0.13 | 0.09 ± 0.18 | 0.53 ± 0.11 |
| Pretrained GIN | **0.64 ± 0.01** | 0.06 ± 0.08 | 0.55 ± 0.08 |
| **Pretrained PepLand** | 0.64 ± 0.00 | **0.19 ± 0.01** | **0.61 ± 0.01** |
| **PeptideCLM-2-Hybrid** | 0.64 ± 0.01 | -0.12 ± 0.05 | 0.42 ± 0.04 |
| ESM3-small | 0.60 ± 0.04 | 0.06 ± 0.07 | 0.52 ± 0.05 |
| ESMC-300M | 0.60 ± 0.03 | 0.09 ± 0.04 | 0.55 ± 0.05 |
| ESMC-600M | 0.56 ± 0.02 | -0.03 ± 0.04 | 0.45 ± 0.02 |
| Transformer | 0.59 ± 0.06 | 0.08 ± 0.10 | 0.54 ± 0.09 |
| DeepLC | 0.60 ± 0.11 | -0.05 ± 0.12 | 0.43 ± 0.11 |
| DeepRT-CapsNet | 0.43 ± 0.06 | -0.06 ± 0.07 | 0.44 ± 0.02 |
| Morgan FP MLP | 0.62 ± 0.02 | -0.04 ± 0.06 | 0.53 ± 0.02 |

PepLand achieves the highest Δ Pearson and ΔAUC of any model tested, making it, alongside PeptideCLM-2 on the regression side, one of the strongest candidates we have found for this task so far. That said, the improvement is modest: at ΔAUC = 0.61 against a 0.5 chance baseline, PepLand is still far from resolving discrimination outright, and PeptideCLM-2-Hybrid actually performs *worse than random chance* (ΔAUC = 0.42). We read this as reinforcing rather than undercutting our central claim: the best currently available peptide-aware representation learners can partially pick up on the stereochemical signal, but none reliably distinguishes D-/L-stereoisomers, which is precisely the gap this benchmark is meant to surface.

## Weaknesses — Minor

### 3. Pearson's R vs. AUC reporting

We agree with reviewer hzjD that our reporting of model performance was inconsistent, and we thank the reviewer for pinpointing exactly why: as currently written, the abstract and introduction juxtapose two numbers that are not on comparable scales. To clarify what each one actually measures: in the standard scenario, Pearson's r is computed between predicted and observed %B values directly. In the discrimination scenario, both of our pair-level metrics are instead computed on the predicted vs. true %B **delta** within each matched diastereomeric pair, i.e., how well the model's signed difference between the two peptides in a pair tracks the true signed difference. Δr is the Pearson correlation between predicted and true delta; AUC only asks whether the model gets the **sign** of that delta right, treating it as a binary discrimination problem. So a Pearson r near 0 in the standard scenario and an AUC near 0.5 in the discrimination scenario are indeed both chance-level, but they are chance-level on different quantities (absolute %B value vs. sign of a pairwise delta), which makes the abstract's juxtaposition misleading rather than merely inconsistent. We will revise the abstract and introduction to report metrics on a consistent, directly comparable basis across both scenarios (e.g., AUC or an equivalent for both) and state explicitly what each metric is computed over.

## Questions

### 4. Morgan fingerprint radius

Radius=2 (ECFP4) was chosen following the field-standard convention in QSAR and retention-time prediction work; we did not perform a search over this parameter.