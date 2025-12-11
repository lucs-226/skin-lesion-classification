import os
import random
import numpy as np
import torch
import gdown
import zipfile

def seed_everything(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def download_weights(file_id, output_dir):
    """Downloads and extracts the weights zip if not present."""
    # Check if the first fold exists as a proxy for all weights
    if not (output_dir / "effnetb3_fold0.pth").exists():
        print("Downloading weights...")
        zip_path = output_dir / "weights.zip"
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, str(zip_path), quiet=False)
        
        print("Extracting weights...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
