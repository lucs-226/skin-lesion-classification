import torch
from pathlib import Path

# --- Google Drive IDs (YOU MUST FILL THESE) ---
# Zip containing: effnetb3_fold0.pth, ..., effnetb3_fold4.pth
WEIGHTS_ZIP_ID = "INSERT_YOUR_WEIGHTS_ZIP_ID_HERE"
# Zip containing the DermX dataset
EXTERNAL_TEST_ID = "INSERT_YOUR_DERMX_ZIP_ID_HERE"

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Specific Paths
WEIGHTS_DIR = OUTPUT_DIR  # Where weights will be extracted
TEST_ZIP_PATH = DATA_DIR / "test_external.zip"
TEST_EXTRACT_DIR = DATA_DIR / "external_dermx"

# --- Hyperparameters ---
SEED = 1526
IMG_SIZE = 300
BATCH_SIZE = 32
NUM_CLASSES = 7
FOLDS = 5
NUM_WORKERS = 2

# Compute
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classes
LABELS = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASSES_MAP = {label: i for i, label in enumerate(LABELS)}
