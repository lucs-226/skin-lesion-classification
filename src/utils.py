import os
import random
import numpy as np
import torch
import gdown
import zipfile

def seed_everything(seed):
    """Exact reproduction of the notebook seeding."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def download_and_extract(file_id, zip_path, extract_to):
    """Handles Drive download and extraction."""
    if not os.path.exists(extract_to):
        print(f"Downloading {zip_path}...")
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, str(zip_path), quiet=False)
        
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

def run_tta(model, loader, device):
    """
    Test Time Augmentation: Average of Original + Horizontal Flip.
    Exact implementation from the notebook.
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
