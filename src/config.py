import torch
from pathlib import Path

# --- Google Drive IDs ---
# Main Dataset (HAM10000 + Metadata.csv)
MAIN_DATA_ID = "1_-9TkAOGK4DpSZPK45QbOfPKprmu3K1f"
# External Test Dataset (Unified/DermX)
EXTERNAL_DATA_ID = "1nmWhcZTJgfhhEq1AfHjpQxqS5Pn-FMrt"

# --- Local Directories ---
# Dynamically created to keep repo clean
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# --- File Paths ---
# Main training CSV (used for training labels and test set blacklisting)
TRAIN_CSV_PATH = DATA_DIR / "HAM10000_metadata.csv"
TRAIN_IMG_DIRS = [
    DATA_DIR / "HAM10000_images_part_1",
    DATA_DIR / "HAM10000_images_part_2"
]

# External Test Path (Folder structure extracted from zip)
# Assuming the external zip extracts to a folder named 'Unified_dataset' or 'val'
TEST_DATA_DIR = DATA_DIR / "Unified_dataset" / "val" 

# --- Hyperparameters ---
SEED = 1526
IMG_SIZE = 300
BATCH_SIZE = 32
NUM_CLASSES = 7
FOLDS = 5
EPOCHS = 15
LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Mappings ---
LABELS = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASSES_MAP = {label: i for i, label in enumerate(LABELS)}
