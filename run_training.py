import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from tqdm import tqdm

from src.config import *
from src.utils import seed_everything, download_and_extract, run_tta
from src.data import SkinDataset, get_transforms, get_sampler
from src.model import build_model, FocalLoss

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for img, label in tqdm(loader, desc="Train", leave=False):
        img, label = img.to(DEVICE), label.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(img), label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    for img, label in loader:
        img, label = img.to(DEVICE), label.to(DEVICE)
        loss = criterion(model(img), label)
        total_loss += loss.item()
    return total_loss / len(loader)

if __name__ == "__main__":
    seed_everything(SEED)
    
    # 1. Setup Data
    if not TRAIN_CSV.exists():
        download_and_extract(MAIN_DATA_ID, DATA_DIR)
    
    df = pd.read_csv(TRAIN_CSV)
    targets = df['label_idx'].values
    
    # Arrays to store Out-Of-Fold predictions (Internal Validation)
    oof_preds = np.zeros((len(df), NUM_CLASSES))
    oof_targets = np.zeros(len(df))
    
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    
    print(f"Starting Training: {FOLDS} Folds on {DEVICE}")

    # 2. K-Fold Loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        print(f"\n>>> FOLD {fold+1}/{FOLDS}")
        
        # Data Split
        train_ds = Subset(SkinDataset(df, get_transforms('train')), train_idx)
        val_ds = Subset(SkinDataset(df, get_transforms('valid')), val_idx)
        
        # Weighted Sampler for class imbalance
        sampler = get_sampler(targets[train_idx])
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        
        # Model Setup
        model = build_model(pretrained=True)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = FocalLoss()
        
        # Training Loop
        best_loss = float('inf')
        save_path = OUTPUT_DIR / f"effnetb3_fold{fold}.pth"
        
        for epoch in range(EPOCHS):
            t_loss = train_one_epoch(model, train_loader, criterion, optimizer)
            v_loss = validate(model, val_loader, criterion)
            scheduler.step()
            
            if v_loss < best_loss:
                best_loss = v_loss
                torch.save(model.state_dict(), save_path)
            
            print(f"Ep {epoch+1:02d} | T: {t_loss:.4f} | V: {v_loss:.4f} | Best: {best_loss:.4f}")
        
        # --- OOF Inference Logic ---
        print("Running TTA on Validation Fold...")
        # Reload best model for this fold
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
        model.eval()
        
        fold_probs = []
        with torch.no_grad():
            for img, label in val_loader:
                img = img.to(DEVICE)
                # Apply TTA (Original + Flip) defined in src.utils
                avg_p = run_tta(model, img)
                fold_probs.append(avg_p.cpu().numpy())
        
        # Store predictions in the global OOF array
        fold_probs = np.concatenate(fold_probs)
        oof_preds[val_idx] = fold_probs
        oof_targets[val_idx] = targets[val_idx]

    # 3. Final Internal Report (OOF)
    print("\n" + "="*40)
    print("INTERNAL DATASET EVALUATION (OOF)")
    print("="*40)
    
    final_preds = np.argmax(oof_preds, axis=1)
    print(classification_report(oof_targets, final_preds, target_names=LABELS))
    
    # Optional: Save OOF results
    np.save(OUTPUT_DIR / "oof_preds.npy", oof_preds)
    print("Training Complete. OOF Predictions saved.")
