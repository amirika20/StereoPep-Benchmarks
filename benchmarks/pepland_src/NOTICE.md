Vendored from https://github.com/zhangruochi/pepland (MIT License), commit fetched 2026-07-23.

Only the files needed for `PepLandFeatureExtractor` inference are included
(`model/core.py`, `utils/commons.py`, `utils/process.py`,
`tokenizer/pep2fragments.py`, `tokenizer/vocabs/Vocab_SIZE258.txt`) — the
training-only modules (`model/data.py`, `model/hgt.py`, `model/model.py`,
`model/util.py`) are not needed because `PepLandFeatureExtractor` loads the
pretrained model via `mlflow.pytorch.load_model`, which reconstructs the
model class from the code snapshot bundled inside the checkpoint itself
(see `benchmarks/pretrained_weights/pepland_model/code/`).

See the paper: PepLand: a large-scale pre-trained peptide representation
model for a comprehensive landscape of both canonical and non-canonical
amino acids (arXiv:2311.04419).
