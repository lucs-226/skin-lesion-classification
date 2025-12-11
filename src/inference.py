import torch
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path

from src.config import (
    DEVICE, OUTPUT_DIR, FOLDS, 
    WEIGHTS_ZIP_ID, WEIGHTS_DIR
)
from src.utils import download_and_extract
from src.model import build_model

def ensure_weights():
    """Checks if weights exist, otherwise downloads/extracts them."""
    # Check for the first fold as a proxy
    if not (WEIGHTS_DIR / "effnetb3_fold0.pth").exists():
        zip_target = OUTPUT_DIR / "weights.zip"
        download_and_extract(WEIGHTS_ZIP_ID, str(zip_target), str(OUTPUT_DIR))

def ensemble_inference(model, weights_paths, loader, device):
    """
    Efficient ensemble: Reuses the same model instance, swapping weights.
    Loops through all 5 folds.
    """
    ensure_weights()
    
    ensemble_logits = None
    valid_weights = [Path(w) for w in weights_paths if Path(w).exists()]
    
    if not valid_weights:
        raise FileNotFoundError(f"No weight files found in: {weights_paths}")

    model.to(device)

    for w_path in tqdm(valid_weights, desc="Ensemble Inference"):
        # Load weights into the existing model instance
        state_dict = torch.load(w_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        
        fold_probs = []
        with torch.no_grad():
            for img, _ in loader:
                img = img.to(device)
                
                # TTA: Average (Original + Horizontal Flip)
                p1 = model(img).softmax(1)
                p2 = model(torch.flip(img, dims=[3])).softmax(1)
                avg_p = (p1 + p2) / 2.0
                
                fold_probs.append(avg_p.cpu().numpy())
        
        fold_res = np.concatenate(fold_probs)
        
        if ensemble_logits is None:
            ensemble_logits = fold_res
        else:
            ensemble_logits += fold_res

    return ensemble_logits / len(valid_weights)
