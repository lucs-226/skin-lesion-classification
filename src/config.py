import torch
from pathlib import Path

# --- Google Drive IDs ---
# ZIP file containing 5 weights: effnetb3_fold0.pth ... effnetb3_fold4.pth
WEIGHTS_ZIP_ID = "YOUR_WEIGHTS_ZIP_ID_HERE"

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Specific Paths
WEIGHTS_DIR = OUTPUT_DIR
TRAIN_CSV = DATA_DIR / "metadata.csv"

# --- Hyperparameters (Matches Notebook) ---
SEED = 1526
IMG_SIZE = 300
BATCH_SIZE = 32
NUM_CLASSES = 7
FOLDS = 5
NUM_WORKERS = 2

# Compute
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class Mapping
LABELS = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASSES_MAP = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in CLASSES_MAP.items()}
