import matplotlib
matplotlib.use("Agg")  # << Add this first
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from umap import UMAP

def plot_predictions(y_true, y_pred, path, prob=None,
                     title="Prediction vs Ground Truth",
                     xlabel="True Values", ylabel="Predicted Values"):
    """
    Plots predicted vs actual values with a reference line y=x,
    and optionally color-codes by probability.

    Args:
        y_true (array-like): Ground truth values
        y_pred (array-like): Predicted values
        path (str): Where to save the plot
        prob (array-like or None): Prediction confidence/probability per point
        title (str): Plot title
        xlabel (str): Label for x-axis
        ylabel (str): Label for y-axis
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    plt.figure(figsize=(7, 6))

    if prob is not None:
        prob = np.array(prob)
        scatter = plt.scatter(y_true, y_pred, c=prob, cmap='viridis', s=30, alpha=0.8, edgecolor='k')
        cbar = plt.colorbar(scatter)
        cbar.set_label("Prediction Confidence")
    else:
        plt.scatter(y_true, y_pred, alpha=0.6, edgecolor='k')


    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.title(title, fontsize=15)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    
import random

def plot_umap_embeddings(model, datamodule):

    model.eval()
    device = next(model.parameters()).device

    all_embeddings = []
    all_lcms_rt = []
    all_sequences = []

    dataloader = datamodule.train_dataloader()
    with torch.no_grad():
        for batch in dataloader:
            embedding, sequences, LCMS_RT = model.get_embedding(batch)
            all_embeddings.append(embedding.cpu())
            all_sequences.extend(sequences)
            all_lcms_rt.append(LCMS_RT.cpu())
            # all_e25.append(E25.cpu())

    embeddings = torch.cat(all_embeddings).numpy()
    lcms_rt_labels = torch.cat(all_lcms_rt).numpy()
    # e25_labels = torch.cat(all_e25).numpy()

    embeddings = StandardScaler().fit_transform(embeddings)
    reducer = UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    embedding_2d = reducer.fit_transform(embeddings)

    font = {'size': 16}
    plt.rc('font', **font)

    def annotate_sequence_below(fig, seq, y_offset=0.02):
        fig.text(0.5, y_offset, f"Highlighted sequence: {seq}", ha='center', fontsize=14, wrap=True)

    def plot_umap(ax, labels, cmap, title, filename):
        idx = random.randint(0, len(embedding_2d) - 1)
        x, y = embedding_2d[idx]
        seq = all_sequences[idx]

        scatter = ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap=cmap, s=8, alpha=0.8)
        ax.plot(x, y, 'o', markersize=8, color='red', markeredgecolor='black')
        ax.set_title(title, fontsize=18)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(title.split()[-1], fontsize=16)
        return seq  # return the sequence so we can annotate it

    # ---- LCMS_RT plot ----
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    seq1 = plot_umap(ax1, lcms_rt_labels, "viridis", "UMAP of Peptide Embeddings (LCMS_RT)", "umap_peptides_lcms_rt.png")
    annotate_sequence_below(fig1, seq1)
    fig1.tight_layout(rect=[0, 0.05, 1, 1])  # leave room at bottom
    fig1.savefig("umap_peptides_B.png", dpi=300)
    plt.close(fig1)
