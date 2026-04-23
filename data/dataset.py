from pathlib import Path

import torch
from torch.utils.data import Dataset
from datasets import load_dataset as hf_load_dataset

from data.Tokenizer import PeptideTokenizer, ContinuousValueTokenizer


VOCAB_FILE = Path(__file__).parent / "PEPLM_WORDS.csv"
HF_REPO = "amirka20/StereoPep"


class PeptideDataset(Dataset):
    def __init__(self, hf_split):
        self.sequences = hf_split["Peptide"]
        self.B = torch.tensor(hf_split["B"], dtype=torch.float)
        self.Z = torch.tensor(hf_split["Z"], dtype=torch.long)
        self.M = torch.tensor([m / 1000 for m in hf_split["M"]], dtype=torch.float)

        self.sequence_tokenizer = PeptideTokenizer(str(VOCAB_FILE))
        self.B_tokenizer = ContinuousValueTokenizer("B", 100, 10)

        self.sequence_tokens, self.attention_mask = self.sequence_tokenizer.tokenize_batch(self.sequences)
        self.B_tokens = self.B_tokenizer.tokenize_batch(self.B)
        self.B_soft = self.B_tokenizer.batch_soft(self.B)

    def __getitem__(self, idx):
        return {
            "sequences": self.sequences[idx],
            "sequence_tokens": self.sequence_tokens[idx],
            "attention_mask": self.attention_mask[idx],
            "B": self.B[idx],
            "B_tokens": self.B_tokens[idx],
            "B_soft": self.B_soft[idx],
            "Z": self.Z[idx],
            "M": self.M[idx],
        }

    def __len__(self):
        return len(self.sequences)


def load_stereopep():
    ds = hf_load_dataset(HF_REPO, "StereoPep")
    return (
        PeptideDataset(ds["train"]),
        PeptideDataset(ds["val"]),
        PeptideDataset(ds["test"]),
    )
