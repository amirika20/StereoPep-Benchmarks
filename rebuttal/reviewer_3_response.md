# Response to Reviewer 3

We wish to thank reviewer aVAU for their careful reading of our work. All comments made were thoroughly considered and we agree that they will bring added value to our manuscript. Accordingly, we wish to address each point below, in order.

## Weaknesses

### 1. Proxy readout and title scope

We thank the reviewer for this thoughtful comment. We agree that the title and presentation can better reflect the scope of our benchmark. Our dataset is designed to evaluate the experimentally measured consequences of **single-stereocenter inversion** in matched diastereomeric peptide pairs, rather than stereochemistry in its broadest sense. Within each matched pair, the peptides differ only in the configuration of a single stereocenter, making stereocenter inversion the controlled experimental perturbation. Consequently, the benchmark evaluates whether molecular models can predict the experimentally observed property changes arising from this specific stereochemical perturbation, rather than all forms of stereochemistry (e.g., cis/trans isomerism, multiple stereocenters, or other stereochemical phenomena).

To make this scope explicit and avoid overgeneralization, we will revise the title, abstract, and corresponding text in the camera-ready version. For example, a title such as **"StereoPep: A Benchmark of Synthetic Diastereomeric Peptides for Evaluating Molecular Models under Single-Stereocenter Inversion"** more accurately reflects the intended scope of the benchmark while preserving its central contribution.

### 2. SMILES protonation and tool use

We thank reviewer aVAU for this comment and agree that a more precise description of the protonation step is warranted. No third-party protonation or tautomer-standardization tool was used (e.g., Dimorphite-DL, OpenBabel, RDKit's `Reionizer`). Protonation is assigned deterministically at SMILES-construction time using the most probable protonation state (`data_processing.py`, `peptide_to_smiles`/`_build_backbone_smiles`; Appendix D), using literature pKa values for each ionizable group at pH 3 in HPLC aqueous mobile phase (H2O + 0.1% formic acid, see below for precise pKa values):

- N-terminal α-amine → protonated `[NH3+]` (pKa ≈ 8)
- Lys ε-amine → protonated `[NH3+]` (pKa ≈ 10.5)
- Arg guanidinium → protonated `[NH2+]=` form (pKa ≈ 12.5)
- His imidazole → protonated `[NH+]` imidazolium tautomer (pKa ≈ 6)
- Asp/Glu and the C-terminus → neutral `[COOH]` carboxylic acid (pKa ≈ 3.1–4.1)

Every peptide receives this single, explicit protonation state rather than being passed through a generic pH-enumeration tool. We will add this explanation and the per-residue pKa table to the Methods. We will also correct the wording "canonical SMILES" (Section 3, Appendix D) to "deterministic isomeric SMILES," as no canonicalization algorithm was applied; the string is constructed directly, not RDKit-canonicalized. We wish to convey to the reader that no implicit step in determining the protonation species was added.

### 3. Confirming matched pairs differ only in the intended chiral tag, and that labels are correct

We thank reviewer aVAU for this concern. We have manually verified all diastereomeric pairs rather than assumed. Across all 8,881 D-Phe/L-Phe pairs (held-out test set + trainval pool), we regenerated each pair's SMILES, stripped all `@`/`@@` stereo descriptors, and confirmed the resulting strings are character-for-character identical in 8,881/8,881 cases (100%): one molecular graph, differing only in the stereo descriptor at the intended Phe α-carbon. This follows from the construction (every residue is serialized in a fixed substituent order: backbone-N → Cα → side chain → C=O, so D vs. L changes exactly one `@`/`@@` token and nothing else), and we confirmed it on the full pair population, not a sample. We will reword Section 3/Appendix D's phrase "map to distinct molecular graphs," which reads as implying a topology difference, to "distinct stereodescriptors on an identical constitutional graph."

On label correctness: the D/L assignment is not read off the mass spectrometer (D- and L-Phe are isobaric and indistinguishable by mass spectrometers) but is instead determined by which synthesis run a PSM came from, encoded via file name and enforced by `clean_psm()`'s sequence-validation step, which rejects any peptide not matching the expected terminal-residue pattern for that run before the D/L relabeling is applied.

### 4. Tautomer/protonation standardization beyond pH 3

We did not enumerate multiple tautomers or run a general-purpose standardization pass, because every ionizable/tautomerizable group in the alphabet is fixed to a single, literature-supported dominant microstate at pH 3 rather than left ambiguous: His is fixed to the Nε-protonated imidazolium tautomer, Lys/Arg/the N-terminus are fixed fully protonated, and Asp/Glu/the C-terminus are fixed neutral. This assignment is applied identically to every sequence, so it cannot introduce sequence-dependent noise, but it does mean we report a single fixed-pH representation rather than a population ensemble over protonation/tautomer states. We will state this explicitly as a scope limitation rather than leaving it implicit.

### 5. Duplicate handling

We thank reviewer aVAU for this comment and agree that the handling of duplicates was not clearly stated. We have handled de-duplication in two stages. **Within a library**: for each unique peptide, the highest-intensity PSM is retained (`clean_psm()`); any sequence attributed to two different protein IDs is separately flagged as a possible cross-contamination case rather than silently merged. **Across libraries**: for peptides observed in more than one synthesis batch (1,205 of them), the two readings are averaged (mean %B, mode terminal tag). To characterize how much duplicate readings can actually differ across libraries, we computed the spread across all 1,205 pairs: the signed difference has mean ≈ −0.30 %B, std ≈ 6.26 %B, and the absolute (max−min) spread per pair averages 3.73 %B (std 5.04 %B), with 27.5% of pairs spreading by more than 5 %B and 10.4% by more than 10 %B (max 35.46 %B). We will report this distribution in Appendix D alongside the averaging rule, so the reader can judge the noise this introduces rather than only being told duplicates were merged.

### 6. Peptide count discrepancy between abstract and manuscript

We thank reviewer aVAU for their careful read of the paper and spotting of this typographical error. We will trace and reconcile all mentions of library sizes: the introduction's **48,789** is a typo; every other occurrence (contributions bullet, Section 3, Appendix D) correctly states **48,788**, which matches our independent recount (46,062 unique peptides in train+val plus 2,726 in the held-out test split, with zero sequence overlap between them). We will fix the single digit in the abstract so the number is consistent throughout.

## Limitations

### 7. Ruling out trivial failure modes

We agree with reviewer aVAU that careful revision of possible model failures is warranted, especially in the case of negative results. Here are some steps that ensure our manuscript is free of trivial mistakes:

- **Encoding bugs / chiral-tag ordering (data side):** every residue is serialized with the same fixed substituent order regardless of sequence context, so there is no ordering artifact beyond the actual stereo descriptor (confirmed in point 3 above, 8,881/8,881 pairs).
- **Encoding bugs / chiral-tag ordering (model side):** every benchmarked model has an explicit, separately-addressable D/L signal, so near-random discrimination cannot be explained by the input withholding that signal. Morgan FP MLP uses chirality-aware fingerprint hashing (Appendix E). GIN and Pretrained GIN encode an explicit chirality-tag feature per atom (`ci ∈ {0,1,2}`: unspecified/CW/CCW), never collapsed. The from-scratch Transformer and the fine-tuned PLMs (ESM3-small, ESM-C 300M/600M) assign D-Phe a dedicated token/embedding (Eq. 1). DeepLC extends its one-hot pathway with a dedicated index for D-Phe specifically because its atomic-composition pathways are otherwise blind to it (Appendix E). DeepRT-CapsNet assigns D-Phe its own vocabulary index (22-token vocabulary) and thus its own learned embedding. Table 4's near-random AUC is therefore a failure to use an available signal, not a representation that hid it, as stated directly in Section 4 ("even when they are explicitly designed to be chirality-aware").
- **Tautomer/protonation standardization:** addressed in point 4 above: a single fixed, literature-supported microstate is applied uniformly, so it cannot be a source of sequence-dependent label or feature noise.
- **Noisy labels:** PSMs are filtered on intensity > 0 and |Δmass| < 0.1 Da before RT is used, with the two-stage duplicate handling and quantified duplicate-spread stats in point 5; rows falling outside the calibrated gradient window are dropped, not imputed.
- **Split leakage:** verified computationally that train, val, and test share zero peptide sequences; the test set is an independently synthesized library (Section 3, "Splits and availability"), never used to build train/val. The diastereomer, point-mutant, and terminal-tag pair benchmarks are likewise built separately within the test split and within the trainval pool, never mixed, so the headline stereo-discrimination result (Table 4) is measured on peptides never seen in any form during training. Table 5 (train+val pool, AUC → 0.76) further shows the near-random test-set result is not an artifact of test-set size.
