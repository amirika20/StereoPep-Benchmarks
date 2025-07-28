import torch
import pandas as pd
import math

class PeptideTokenizer:
    def __init__(self, vocab_file):
        self.special_tokens = ['PAD', 'UNK', 'BOS', 'EOS', 'MASK']
        amino_acids = list(pd.read_csv(vocab_file).to_dict()['words'].values())

        self.vocab = self.special_tokens + amino_acids
        self.token_to_idx = {tok: i for i, tok in enumerate(self.vocab)}
        self.idx_to_token = {i: tok for tok, i in self.token_to_idx.items()}
        self.pad_token_id = self.token_to_idx['PAD']
        self.mask_token_id = self.token_to_idx['MASK']
        self.bos_token_id = self.token_to_idx['BOS']
        self.eos_token_id = self.token_to_idx['EOS']
        self.unk_token_id = self.token_to_idx['UNK']
        self.token_to_idx['_'] = self.mask_token_id
        self.vocab_size = len(self.vocab)

    def tokenize(self, sequence, add_special_tokens=True):
        # tokens = sequence.strip().split()
        tokens = list(sequence)
        if add_special_tokens:
            tokens = ['BOS'] + tokens + ['EOS']
        indices = [self.token_to_idx.get(tok, self.token_to_idx['UNK']) for tok in tokens]
        return torch.tensor(indices, dtype=torch.long)

    def pad_sequence(self, token_tensor, max_length):
        padded = torch.full((max_length,), self.pad_token_id, dtype=torch.long)
        length = min(len(token_tensor), max_length)
        padded[:length] = token_tensor[:length]

        attention_mask = torch.zeros(max_length, dtype=torch.long)
        attention_mask[:length] = 1
        return padded, attention_mask

    def detokenize(self, indices, remove_special_tokens=True):
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        tokens = [self.idx_to_token.get(i, 'UNK') for i in indices]
        if remove_special_tokens:
            tokens = [t for t in tokens if t not in self.special_tokens]
        return ' '.join(tokens)
    
    def tokenize_batch(self, sequences, max_length=None, add_special_tokens=True):
        """
        Tokenizes a list of peptide sequences with padding and attention masks.
        
        Args:
            sequences (List[str]): list of space-separated peptide sequences
            max_length (int): pad all sequences to this length (optional).
                            If None, use length of the longest sequence.
            add_special_tokens (bool): whether to add BOS/EOS tokens
        
        Returns:
            token_ids: (B, L) LongTensor
            attention_masks: (B, L) LongTensor
        """
        tokenized = [self.tokenize(seq, add_special_tokens) for seq in sequences]

        if max_length is None:
            max_length = max(len(seq) for seq in tokenized)

        batch_tokens = []
        batch_masks = []
        for tok in tokenized:
            padded, mask = self.pad_sequence(tok, max_length)
            batch_tokens.append(padded)
            batch_masks.append(mask)

        return torch.stack(batch_tokens), torch.stack(batch_masks)

class ContinuousValueTokenizer:
    def __init__(self, name, max_value, num_bins):
        """
        Args:
            name (str): feature name (e.g. "EFF")
            max_value (float): maximum possible value in dataset
            num_bins (int): number of bins to discretize values into
        """
        self.name = name
        self.max_value = max_value
        self.num_bins = num_bins
        self.delta = self.max_value/num_bins

        self.bin_edges = torch.linspace(0, max_value, num_bins + 1)  # Includes upper edge
        self.lower_bound = self.bin_edges[:-1]
        self.upper_bound = self.bin_edges[1:]
        self.special_token = ['MASK']
        self.tokens = self.special_token + [
            f"{float(self.bin_edges[i]):.2f}-{float(self.bin_edges[i+1]):.2f}"
            for i in range(num_bins)
        ]

        self.token_to_idx = {tok: i for i, tok in enumerate(self.tokens)}
        self.idx_to_token = {i: tok for tok, i in self.token_to_idx.items()}
        self.vocab_size = len(self.tokens)
        self.mask_token_id = self.token_to_idx["MASK"]

    def _value_to_bin(self, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        if math.isnan(value):
            return self.mask_token_id
        bin_idx = int((value / self.max_value) * self.num_bins)
        return max(0, min(self.num_bins - 1, bin_idx)) + 1

    def tokenize(self, value):
        token_id = self._value_to_bin(value)
        return torch.tensor(token_id, dtype=torch.long)

    def tokenize_batch(self, values):
        if isinstance(values, torch.Tensor):
            values = values.cpu().tolist()
        indices = []
        for value in values:
            token_id = self._value_to_bin(value)
            indices.append(token_id)
        return torch.tensor(indices, dtype=torch.long)

    def gaussian_soft(self, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        if math.isnan(value):
            dist = torch.zeros((self.num_bins+1))
            dist[0] = 1
            return dist
        normal = torch.distributions.Normal(loc=value, scale=self.delta/2)
        _cdf = normal.cdf(self.lower_bound)
        cdf_ = normal.cdf(self.upper_bound)
        dist = cdf_ - _cdf
        dist = dist / dist.sum()
        return torch.cat([torch.tensor([0.0]),dist])

    def batch_soft(self, values):
        if isinstance(values, torch.Tensor):
            values = values.cpu().tolist()
        dists = []
        for value in values:
            dists.append(self.gaussian_soft(value))
        return torch.stack(dists)

    def decode(self, token_idx):
        token = self.idx_to_token[int(token_idx)]
        if token == "MASK":
            return None
        return token


if __name__ == "__main__":
    LCMS_RT_tokenizer = ContinuousValueTokenizer("LCMS_RT", 60, 6)
    RT = [1,7,9,34,45,24]
    print(LCMS_RT_tokenizer._value_to_bin(10))
    print(LCMS_RT_tokenizer.batch_soft([10,20,25]))