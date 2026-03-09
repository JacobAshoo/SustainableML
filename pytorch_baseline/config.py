import torch

SEED = 42
BATCH = 128
EPOCHS = 2
LR = 1e-3
NW = 2
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT_CSV = "baseline_clean_metrics.csv"
