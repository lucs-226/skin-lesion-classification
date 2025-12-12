import torch
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.config import *
from src.utils import set_seed, download_and_extract
from src.data import ExternalTestDataset, create_filtered_test_set, val_transform
from src.model import build_model
from src.visualization import plot_confusion_matrix

def ensemble_inference(model, weights_paths, loader, device):
    """
    Ensemble over folds using Weight Swapping + TTA.
    """
    ensemble_logits = None
    valid_weights = [w for w in weights_paths if w.exists()]
    
    if not valid_weights:
        raise FileNotFoundError("No weight files found in output/.")
        
    model.to(device)
    
    for w_path in tqdm(valid_weights, desc="Ensemble Inference"):
        model.load_state_dict(torch.load(w_path, map_location=device))
        model.eval()
        
        fold_probs = []
        with torch.no_grad():
            for img, _ in loader:
                img = img.to(device)
                
                # TTA: Original + Flip
                p1 = model(img).softmax(1)
                p2 = model(torch.flip(img, dims=[3])).softmax(1)
                avg = (p1 + p2) / 2.0
                
                fold_probs.append(avg.cpu().numpy())
        
        fold_res = np.concatenate(fold_probs)
        if ensemble_logits is None:
            ensemble_logits = fold_res
        else:
            ensemble_logits += fold_res
            
    return ensemble_logits / len(valid_weights)

if __name__ == "__main__":
    set_seed(SEED)
    
    # 1. Download Data
    # Main needed for blacklist csv
    download_and_extract(MAIN_DATA_ID, DATA_DIR) 
    # External dataset
    download_and_extract(EXTERNAL_DATA_ID, DATA_DIR / "Unified_dataset")
    
    # 2. Prepare Data
    print("Generating Filtered Test Set from folders...")
    # NOTE: ensure TEST_DATA_DIR points to the extracted folders (e.g., 'val')
    df_test = create_filtered_test_set(TEST_DATA_DIR, TRAIN_CSV_PATH, LABELS)
    print(f"External Test Samples: {len(df_test)}")
    
    test_loader = DataLoader(
        ExternalTestDataset(df_test, CLASSES_MAP, val_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    
    # 3. Ensemble
    model = build_model(NUM_CLASSES, pretrained=False)
    weights = [OUTPUT_DIR / f"effnetb3_fold{i}.pth" for i in range(FOLDS)]
    
    print("Running Inference...")
    probs = ensemble_inference(model, weights, test_loader, DEVICE)
    preds = np.argmax(probs, axis=1)
    
    # 4. Report
    y_true = df_test['label'].map(CLASSES_MAP).values
    print("\nEXTERNAL TEST REPORT")
    print(classification_report(y_true, preds, target_names=LABELS))
    
    # Plot
    cm = confusion_matrix(y_true, preds)
    plot_confusion_matrix(cm, LABELS)
