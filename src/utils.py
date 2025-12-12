import os
import random
import numpy as np
import torch
import gdown
import zipfile

def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def download_and_extract(file_id, dest_folder):
    """Downloads zip from Drive and extracts to dest_folder if not present."""
    if not dest_folder.exists() or not any(dest_folder.iterdir()):
        print(f"Downloading data to {dest_folder}...")
        dest_folder.mkdir(parents=True, exist_ok=True)
        zip_path = dest_folder / "temp.zip"
        
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, str(zip_path), quiet=False)
        
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(dest_folder)
        os.remove(zip_path)

def run_tta(model, loader, device):
    """
    Test Time Augmentation: Average of Original + Horizontal Flip.
    Matches notebook logic.
    """
    model.eval()
    probs = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            # Forward 1: Original
            p1 = model(images).softmax(1)
            # Forward 2: Horizontal Flip
            p2 = model(torch.flip(images, dims=[3])).softmax(1)
            # Average
            avg_p = (p1 + p2) / 2.0
            probs.append(avg_p.cpu().numpy())
    return np.concatenate(probs)
