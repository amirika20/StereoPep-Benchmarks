import rich.console

# Backup the original method
original_clear_live = rich.console.Console.clear_live

# Define a safe wrapper
def safe_clear_live(self):
    if getattr(self, "_live_stack", []):
        original_clear_live(self)

# Apply the monkey patch
rich.console.Console.clear_live = safe_clear_live

from Model.deeprt import DeepRT
from data.dataset import *

from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import RichProgressBar, EarlyStopping


class PepDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, random_state=42):
        super().__init__()
        self.batch_size = batch_size
        self.random_state = random_state
        

    def setup(self, stage=None):
        df = pd.read_csv("/home/amirabbas-kazeminia/Projects/DeepRT/data/ML_DATA.csv")
        data = {
            'sequences': df['Peptide'].tolist(),
            'B': df['B'].tolist(),
            'Z': df['Z'].tolist(),
            'M': [m/1000 for m in df['M'].tolist()]
        }
        self.train_set, self.val_set, self.test_set = stratified_split(data, "B", random_state=self.random_state)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)


def train(seed, d_model, n_heads, n_layers):
    model = DeepRT(d_model=d_model, n_heads=n_heads, n_layers=n_layers)
    datamodule = PepDataModule(128, random_state=seed)
    logger = TensorBoardLogger("tb_logs", name="testing")

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=20,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=100,
        gradient_clip_val=1,
        callbacks=[early_stop_callback, RichProgressBar()],
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=logger,  # e.g., TensorBoardLogger or WandbLogger
        num_sanity_val_steps=0,
        )
    trainer.fit(model, datamodule=datamodule)
    test_metrics = trainer.test(model, datamodule=datamodule)[0]
    trainer.save_checkpoint(f"weights/{n_layers}layers_{n_heads}heads_{d_model}dim_{seed}seed.ckpt", weights_only=True)
    return test_metrics

def k_training(d_model, num_heads, n_layers):
    all_metrics = []
    seeds = [10]
    # d_model = 12
    # num_heads = 2
    # n_layers = 2
    for index in range(len(seeds)):
        torch.manual_seed(seeds[index])
        torch.cuda.manual_seed(seeds[index])
        torch.cuda.manual_seed_all(seeds[index])
        print(f"\n🔁 Run {index + 1}/{len(seeds)} (random_state = {seeds[index]})")
        test_metrics = train(seeds[index], d_model, num_heads, n_layers)
        test_metrics['run'] = index +1
        all_metrics.append(test_metrics)

    # Build table
    output_csv = f"metrics/{n_layers}layers_{num_heads}heads_{d_model}dim_metrics_summary.csv"
    df = pd.DataFrame(all_metrics)
    df.loc['mean'] = df.drop(columns='run').mean(numeric_only=True)
    df.to_csv(output_csv, index=False)
    print(f"\n📁 Saved metrics to {output_csv}")


if __name__ == "__main__":
    k_training(24,2,1)