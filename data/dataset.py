import os
import sys
sys.path.append("/home/amirabbas-kazeminia/Projects/DeepRT_test")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from data.Tokenizer import PeptideTokenizer, ContinuousValueTokenizer
import torch.nn.functional as F
import math
import ast


from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from collections import namedtuple
from torch.nn.utils.rnn import pad_sequence


import torch
import pickle



def make_gaussian_soft_labels(indices, vocab_size, std=0.5):
    """
    indices: (B,) long tensor with class indices
    Returns: (B, vocab_size) soft label distribution
    """
    B = indices.shape[0]
    soft_labels = torch.zeros(B, vocab_size, device=indices.device)
    
    for i in range(B):
        idx = indices[i].item()
        x = torch.arange(vocab_size, device=indices.device)
        dist = torch.exp(-(x - idx) ** 2 / (2 * std ** 2))
        dist[0] = 0  # token 0 is for MASK
        soft_labels[i] = dist / dist.sum()
    
    return soft_labels


class PeptideDataset(Dataset):
    def __init__(self, data_dict):
        self.sequences = data_dict['sequences']
        self.B = data_dict['B']

        self.sequence_tokenizer = PeptideTokenizer('/home/amirabbas-kazeminia/Projects/DeepRT_test/data/PEPLM_WORDS.csv')
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
        }

    def __len__(self):
        return len(self.sequences)


def stratified_split(data_dict, stratify_by, train_size=0.7, val_size=0.15, test_size=0.15, n_bins=5, random_state=42):
    total = len(data_dict['sequences'])
    strat_values = np.array(data_dict[stratify_by])
    bin_labels = pd.qcut(strat_values, q=n_bins, duplicates='drop').codes

    indices = np.arange(total)
    idx_temp, idx_test, bin_temp, bin_test = train_test_split(
        indices, bin_labels, test_size=test_size, stratify=bin_labels, random_state=random_state)

    strat_temp_values = strat_values[idx_temp]
    bin_temp_2 = pd.qcut(strat_temp_values, q=n_bins, duplicates='drop').codes
    val_fraction = val_size / (train_size + val_size)

    idx_train, idx_val, _, _ = train_test_split(
        idx_temp, bin_temp_2, test_size=val_fraction, stratify=bin_temp_2, random_state=random_state)

    def subset(indices):
        return {k: [v[i] for i in indices] for k, v in data_dict.items()}
    
    pd.DataFrame(subset(idx_train)).to_csv(f"data/split/train_{random_state}.csv")
    pd.DataFrame(subset(idx_val)).to_csv(f"data/split/val_{random_state}.csv")
    pd.DataFrame(subset(idx_test)).to_csv(f"data/split/test_{random_state}.csv")

    return PeptideDataset(subset(idx_train)), PeptideDataset(subset(idx_val)), PeptideDataset(subset(idx_test))



if __name__ == "__main__":
    df = pd.read_csv("/home/amirabbas-kazeminia/Projects/PepLM/Data/ML_DATA.csv")
    data = {
        'sequences': df['sequence'].tolist(),
        'LCMS_RT': df['LCMS_RT'].tolist(),
        'CX': df['CX_RT'].tolist(),
        'RP': df['RP_RT'].tolist(),
        'ES5': df['ES5'].tolist(),
        'ES25': df['ES25'].tolist(),
        'E25': df['E25'].tolist(),
        'T': df['T'].tolist(),
        'Z': df['Z'].tolist(),
        'Mass': [math.log(x) for x in df['Mass']],
        'target_dist': [ast.literal_eval(x) for x in df['target_dist']]
    }
    print(max(data["LCMS_RT"]))
    # train_set, val_set, test_set = stratified_split(data, "LCMS_RT")
    # all_sequence_tokens = torch.cat([train_set.sequence_tokens, val_set.sequence_tokens, test_set.sequence_tokens])
    # print(compute_token_distribution(all_sequence_tokens, train_set.sequence_tokenizer))
    # print(train_set[:10]['RP'])
    # print(train_set[:10]['RP_tokens'])