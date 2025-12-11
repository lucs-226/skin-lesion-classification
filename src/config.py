import torch
from pathlib import Path

# --- IDs (FILL THESE) ---
MAIN_DATA_ID = "https://drive.google.com/file/d/1_-9TkAOGK4DpSZPK45QbOfPKprmu3K1f/view?usp=drive_link"
EXTERNAL_DATA_ID = "https://drive.google.com/file/d/1nmWhcZTJgfhhEq1AfHjpQxqS5Pn-FMrt/view?usp=drive_link"

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
PLOTS_DIR = OUTPUT_DIR / "eda_plots"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# Files
TRAIN_CSV = DATA_DIR / "metadata.csv"
TEST_CSV = DATA_DIR / "external_dermx" / "metadata.csv" # Adjust relative path

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

# --- Compute ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Mappings ---
LABELS = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASSES_MAP = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in CLASSES_MAP.items()}
