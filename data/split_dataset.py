import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = "dia.txt"
TRAIN_FILE = "dia_train.txt"
VAL_FILE   = "dia_val.txt"
TEST_FILE  = "dia_test.txt"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Number of quantile bins used for stratification of the continuous RT target
N_BINS = 20

df = pd.read_csv(INPUT_FILE, sep="\t")

# Bin RT into quantiles so stratified split preserves the RT distribution
df["_rt_bin"] = pd.qcut(df["RT"], q=N_BINS, labels=False, duplicates="drop")

# First split: train vs (val + test)
train_df, temp_df = train_test_split(
    df,
    test_size=(VAL_RATIO + TEST_RATIO),
    stratify=df["_rt_bin"],
    random_state=42,
)

# Second split: val vs test (relative sizes within the temp split)
relative_test_size = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
val_df, test_df = train_test_split(
    temp_df,
    test_size=relative_test_size,
    stratify=temp_df["_rt_bin"],
    random_state=42,
)

# Drop the helper column before saving
for split_df in (train_df, val_df, test_df):
    split_df.drop(columns=["_rt_bin"], inplace=True)

train_df.to_csv(TRAIN_FILE, sep="\t", index=False)
val_df.to_csv(VAL_FILE,   sep="\t", index=False)
test_df.to_csv(TEST_FILE,  sep="\t", index=False)

print(f"Total samples : {len(df):>7}")
print(f"Train samples : {len(train_df):>7}  ({len(train_df)/len(df)*100:.1f}%)")
print(f"Val   samples : {len(val_df):>7}  ({len(val_df)/len(df)*100:.1f}%)")
print(f"Test  samples : {len(test_df):>7}  ({len(test_df)/len(df)*100:.1f}%)")
print(f"\nSaved: {TRAIN_FILE}, {VAL_FILE}, {TEST_FILE}")
