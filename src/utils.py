import os
import random
import numpy as np
import torch
import gdown
import zipfile

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def download_and_extract(file_id, dest_folder):
    if not dest_folder.exists():
        print(f"Downloading data to {dest_folder}...")
        dest_folder.mkdir(parents=True, exist_ok=True)
        zip_path = dest_folder / "temp.zip"
        gdown.download(f'https://drive.google.com/uc?id={file_id}', str(zip_path), quiet=False)
        
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(dest_folder)
        os.remove(zip_path)

def run_tta(model, images):
    """Test Time Augmentation: Average of Original + Horizontal Flip."""
    # Forward 1
    p1 = model(images).softmax(1)
    # Forward 2 (Flip)
    p2 = model(torch.flip(images, dims=[3])).softmax(1)
    return (p1 + p2) / 2.0
