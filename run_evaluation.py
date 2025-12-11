import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.config import *
from src.utils import seed_everything, download_and_extract, run_tta
from src.data import SkinDataset, get_transforms
from src.model import build_model
from src.visualization import plot_confusion_matrix

def ensemble_predict(loader):
    models = []
    # Load all 5 folds
    for i in range(FOLDS):
        m = build_model(pretrained=False)
        m.load_state_dict(torch.load(OUTPUT_DIR / f"effnetb3_fold{i}.pth", map_location=DEVICE))
        m.eval()
        models.append(m)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            fold_probs = []
            
            for m in models:
                # TTA logic via utility
                p = run_tta(m, images)
                fold_probs.append(p.cpu().numpy())
            
            # Average over folds
            avg_probs = np.mean(fold_probs, axis=0)
            all_preds.extend(np.argmax(avg_probs, axis=1))
            all_labels.extend(labels.numpy())
            
    return all_labels, all_preds

if __name__ == "__main__":
    seed_everything(SEED)
    download_and_extract(EXTERNAL_DATA_ID, DATA_DIR / "external")
    
    # Load External Test Data
    df_test = pd.read_csv(TEST_CSV)
    test_loader = DataLoader(
        SkinDataset(df_test, get_transforms('valid')), 
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    
    print("Running Ensemble Inference on External Test...")
    y_true, y_pred = ensemble_predict(test_loader)
    
    print(classification_report(y_true, y_pred, target_names=LABELS))
    
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, "external_test_confusion_matrix.png")
