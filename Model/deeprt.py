import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F


from layers.regression_head import RegressionHead
from layers.transformer_stack import TransformerStack
from layers.attention_pooling import AttentionPooling
from data.Tokenizer import PeptideTokenizer, ContinuousValueTokenizer

from sklearn.metrics import f1_score, cohen_kappa_score, matthews_corrcoef
from scipy.stats import kendalltau, spearmanr
from tqdm import tqdm

def pairwise_logistic_loss(pred_scores, true_scores, min_delta=1.0):
    B = pred_scores.size(0)
    device = pred_scores.device
    # Create all i < j indices (upper triangle)
    i, j = torch.triu_indices(B, B, offset=1, device=pred_scores.device)

    delta = true_scores[i] - true_scores[j]
    keep = delta.abs() >= min_delta
    i, j = i[keep], j[keep]

    if i.numel() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    label = torch.sign(true_scores[i] - true_scores[j])
    label[label == 0] = 1.0  # avoid zero label

    return F.margin_ranking_loss(pred_scores[i], pred_scores[j], label,margin=1)


class DeepRT(pl.LightningModule):
    """
    Deep Learning model for predicting discrete B token and continuous B score for peptide sequences.
    Combines a transformer backbone with attention pooling and two regression heads.

    Outputs:
        - B_logits: logits for tokenized B value prediction
        - B_score: scalar for ranking (regression)
    """

    def __init__(self, d_model, n_heads, n_layers, lr=1e-4, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # === Tokenizers ===
        self.sequence_tokenizer = PeptideTokenizer(
            '/home/amirabbas-kazeminia/Projects/DeepRT_test/data/PEPLM_WORDS.csv'
        )
        self.B_tokenizer = ContinuousValueTokenizer("B", max_value=100, num_bins=10)

        # === Embedding & Transformer Backbone ===
        self.sequence_embedding = nn.Embedding(
            self.sequence_tokenizer.vocab_size, d_model,
            padding_idx=self.sequence_tokenizer.pad_token_id
        )
        self.transformers = TransformerStack(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers, bias=True
        )

        # === Heads ===
        self.pooling = AttentionPooling(d_model)
        self.B_head = RegressionHead(d_model, self.B_tokenizer.vocab_size)  # Classification over B bins
        self.B_score_head = RegressionHead(d_model, 1)  # Scalar for ranking B scores
        self.Z_head = RegressionHead(d_model, 10)  # Scalar for ranking B scores
        self.M_head = RegressionHead(d_model, 1)  # Scalar for ranking B scores

        # === Misc state ===
        self.test_B_pred = []
        self.test_B_target = []
        self.test_B_probs = []
        self.metric_logs = {"train": [], "val": [], "test": []}

        # Optional: compile model for speed (requires PyTorch 2.0+)
        self = torch.compile(self)

    def forward(self, sequence_tokens, sequence_attention_mask):
        """
        Forward pass: sequence -> transformer -> pooled -> heads
        Returns B logits, pooled embeddings, and B score.
        """
        seq_embed = self.sequence_embedding(sequence_tokens)
        x, _, _ = self.transformers(seq_embed, sequence_attention_mask.bool())

        pooled = self.pooling(x)
        B_logits = self.B_head(pooled)         # Discrete B prediction
        B_score = self.B_score_head(pooled)    # Continuous B score for ranking
        Z_logits = self.Z_head(pooled)
        M_pred = self.M_head(pooled)

        return {
            "B_logits": B_logits,      # (B, V)
            "embedding": x,           # (B, L, d_model)
            "B_score": B_score.squeeze(),  # (B,)
            "Z_logits": Z_logits,
            "M_pred": M_pred.squeeze()
        }

    def _step(self, batch, batch_idx, mode):
        """
        Shared training/validation/test step logic.
        Computes predictions, losses, logs metrics.
        """
        # === Forward pass ===
        model_output = self(batch["sequence_tokens"], batch["attention_mask"])

        # === Loss: KL-divergence for classification + pairwise logistic for ranking ===
        B_logits = model_output["B_logits"]
        B_loss = F.kl_div(
            F.log_softmax(B_logits, dim=-1), batch["B_soft"],
            reduction='batchmean'
        )

        # === Loss: KL-divergence for classification + pairwise logistic for ranking ===
        B_score = model_output["B_score"]
        pairwise_loss = pairwise_logistic_loss(B_score, batch["B"])

        Z_loss = F.cross_entropy(model_output["Z_logits"],batch["Z"])

        M_loss = F.mse_loss(model_output["M_pred"], batch["M"])

        total_loss = B_loss + 0.25*pairwise_loss + 0.25*Z_loss + 0.25*M_loss

        # === Log learning rate ===
        lr = self.trainer.optimizers[0].param_groups[0]['lr'] * 1e6
        self.log("lr", lr, prog_bar=True, on_step=True, logger=True)

        # === Metrics Logging ===
        with torch.no_grad():
            metrics = self.compute_metrics(batch, model_output)
            metrics["loss"] = total_loss.item()
            self.metric_logs[mode].append(metrics)

        # === Save test predictions ===
        if mode == 'test':
            self.test_B_pred.append(B_logits.argmax(dim=-1).cpu())
            self.test_B_target.append(batch["B"].cpu())
            self.test_B_probs.append(
                F.softmax(B_logits, dim=-1).max(dim=-1)[0].cpu()
            )

        return total_loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="test")

    def predict_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="inference")


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-2, weight_decay=1e-2)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",         # Minimize validation loss
            factor=0.6,         # Reduce LR by half
            patience=3,         # Wait 3 epochs of no improvement
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",   # This must match your logged val loss key
                "interval": "epoch",
                "frequency": 1
            }
        }


    def compute_metrics(self, batch, model_output, k=3):
        B_logits = model_output["B_logits"]
        B_tokens = batch["B_tokens"]
        B_true_vals = batch["B"].cpu()
        B_scores = model_output["B_score"].detach().cpu()
        B = B_logits.size(0)

        # ----- Predictions -----
        probs = F.softmax(B_logits[:,1:], dim=1)
        B_preds = B_logits.argmax(dim=-1)
        topk_preds = probs.topk(k=k, dim=1).indices
        Z_preds = model_output['Z_logits'].argmax(dim=-1)

        # ----- Classification Metrics -----
        y_true = B_tokens.cpu().numpy()
        y_pred = B_preds.cpu().numpy()

        try:
            f1_macro = f1_score(y_true, y_pred, average='macro')
        except Exception:
            f1_macro = float('nan')

        try:
            cohen_kappa = cohen_kappa_score(y_true, y_pred)
        except Exception:
            cohen_kappa = float('nan')

        try:
            mcc = matthews_corrcoef(y_true, y_pred)
        except Exception:
            mcc = float('nan')

        topk_acc = (topk_preds == B_tokens.unsqueeze(1)).any(dim=1).float().mean()

        # ----- B Approximation -----
        centers = torch.arange(5, 100, 10).unsqueeze(0)  # assumes bins [5,10,...95]
        B_pred_vals = (probs.cpu() * centers).sum(dim=1)  # skip index 0 (e.g. mask)
        cls_kendall = kendalltau(B_true_vals, B_pred_vals).correlation
        cls_spearman = spearmanr(B_true_vals, B_pred_vals).correlation


        # ----- Ranking Metrics -----

        rank_kendall = kendalltau(B_true_vals, B_scores).correlation
        rank_spearman = spearmanr(B_true_vals, B_scores).correlation
        

        try:
            # Pairwise ranking accuracy: fraction of pairs correctly ranked
            diff_true = B_true_vals.unsqueeze(1) - B_true_vals.unsqueeze(0)
            diff_pred = B_scores.unsqueeze(1) - B_scores.unsqueeze(0)
            mask = diff_true != 0
            correct_pairs = ((diff_true * diff_pred) > 0)[mask]
            pairwise_acc = correct_pairs.float().mean().item()
        except Exception:
            pairwise_acc = float('nan')

        # ----- Return All Metrics -----
        return {
            "B_acc": (B_preds == B_tokens).float().mean().item(),
            "Z_acc": (Z_preds == batch['Z']).float().mean().item(),
            "f1_macro": f1_macro,
            "cohen_kappa": cohen_kappa,
            "mcc": mcc,
            f"top{k}_acc": topk_acc.item(),
            "B_MSE": F.mse_loss(B_pred_vals, B_true_vals).item(),
            "M_MSE": F.mse_loss(model_output['M_pred'],batch["M"]).item(),
            "rank_kendall_tau": rank_kendall,
            "rank_spearman": rank_spearman,
            "cls_kendall_tau": cls_kendall,
            "cls_spearman": cls_spearman,
            "pairwise_ranking_acc": pairwise_acc,
        }

    def on_train_epoch_end(self):
        self.log_metrics("train")
    
    def on_validation_epoch_end(self):
        self.log_metrics("val")

    def on_test_epoch_end(self):
        self.log_metrics("test")


    def log_metrics(self, mode):
        if not self.metric_logs[mode]:
            return

        logs = {}
        for key in self.metric_logs[mode][0].keys():
            logs[key] = torch.tensor([m[key] for m in self.metric_logs[mode]], device=self.device).mean()

        for k, v in logs.items():
            self.log(f"{mode}_{k}", v, on_step=False, on_epoch=True, prog_bar=True)

        self.metric_logs[mode] = []  # Reset after logging


    def predict(self, sequences, topk=3):
        """
        Predict B scores for a batch of sequences.
        
        Args:
            sequences (List[str]): list of input sequences
            topk (int): number of top predictions to return

        Returns:
            List[Dict]: one dictionary per sequence with keys:
                - B_probs: softmax vector
                - B_pred: predicted bin label (string)
                - B_approx: continuous B value
                - B_entropy: entropy of B distribution
                - topk: list of (label, prob) tuples
        """
        # Tokenize input
        sequence_tokens, attention_mask = self.sequence_tokenizer.tokenize_batch(sequences)
        self.eval()

        with torch.no_grad():
            model_output = self(sequence_tokens, attention_mask)
            B_logits = model_output["B_logits"]  # shape: (B, V)
            B_probs = F.softmax(B_logits[:,1:], dim=-1)  # shape: (B, V)

        # Create bin centers to estimate B_approx
        centers = torch.arange(5, 100, 10).unsqueeze(0).to(B_probs.device)  # shape: (1, 19)
        B_probs_trimmed = B_probs  # Remove mask index (0)
        B_approx = (B_probs_trimmed * centers).sum(dim=1)  # shape: (B,)

        return {
            "sequences":sequences,
            "B_probs":B_probs,
            "B_scores":model_output["B_score"].cpu().numpy()
        }
    
    def get_embedding(self, sequences):
        batch_size = 64
        AA_embed = []
        post_transformer = []
        post_pooling = []

        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_seqs = sequences[i:i + batch_size]

            # Tokenize and embed
            sequence_tokens, attention_mask = self.sequence_tokenizer.tokenize_batch(batch_seqs)
            with torch.no_grad():
                sequence_embed = self.sequence_embedding(sequence_tokens)
                x, _,_ = self.transformers(sequence_embed, attention_mask.bool())
                pooled = self.pooling(x, attention_mask)

                AA_embed.append(sequence_embed.mean(dim=1))
                post_transformer.append(x.mean(dim=1))
                post_pooling.append(pooled)

        return torch.cat(AA_embed, dim=0), torch.cat(post_transformer,dim=0), torch.cat(post_pooling, dim=0)




if __name__ == "__main__":
    seq = "GLY MDAB MAMF _ MGLN MTHR MLYS MLYS MTRP MLYS"
    model = PepLM.load_from_checkpoint("/home/amirabbas-kazeminia/Projects/PepLM/weights/best_model.ckpt")
    model.generate(seq, 11, 1)
    # print(model.predict(seq))