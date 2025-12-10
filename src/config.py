import torch
from pathlib import Path

# Paths
BASE_DATA_PATH = Path("./data")
OUTPUT_PATH = Path("./output")
OUTPUT_PATH.mkdir(exist_ok=True)

# Hyperparameters
SEED = 1526
IMG_SIZE = 300
BATCH_SIZE = 32
NUM_CLASSES = 7
FOLDS = 5
EPOCHS = 15
LR = 3e-4
WEIGHT_DECAY = 1e-4

# Compute
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 2
