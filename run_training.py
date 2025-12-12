import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

from src.config import *
from src.utils import set_seed, download_and_extract, run_tta
from src.data import SkinDataset, train_transform, val_transform, get_sampler
from src.model import build_model, FocalLoss

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    return running_loss / len(loader)

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(loader)

if __name__ == "__main__":
    set_seed(SEED)
    
    # 1. Prepare Data
    download_and_extract(MAIN_DATA_ID, DATA_DIR)
    
    df = pd.read_csv(TRAIN_CSV_PATH)
    
    # Map images
    image_map = {}
    for d in TRAIN_IMG_DIRS:
        if d.exists():
            for x in glob.glob(os.path.join(d, "*.jpg")):
                image_map[os.path.splitext(os.path.basename(x))[0]] = x
            
    df['path'] = df['image_id'].map(image_map)
    df = df.dropna(subset=['path'])
    
    # Deduplicate
    df_unique = df.drop_duplicates(subset=['lesion_id'], keep='first').reset_index(drop=True)
    df_unique['label_idx'] = df_unique['dx'].map(CLASSES_MAP)
    
    targets = df_unique['label_idx'].values
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    
    # Arrays for OOF metrics
    oof_preds = np.zeros((len(df_unique), NUM_CLASSES))
    oof_targets = np.zeros(len(df_unique))
    
    print(f"Starting Training: {FOLDS} Folds, {EPOCHS} Epochs")

    # 2. K-Fold Loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        print(f"\n>>> FOLD {fold+1}/{FOLDS}")
        
        # Subsets & Sampler
        train_sub = Subset(SkinDataset(df_unique, train_transform), train_idx)
        val_sub = Subset(SkinDataset(df_unique, val_transform), val_idx)
        
        sampler = get_sampler(targets[train_idx])
        train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        
        # Setup
        model = build_model(NUM_CLASSES, pretrained=True)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = FocalLoss(gamma=2.0)
        
        best_loss = float('inf')
        save_path = OUTPUT_DIR / f"effnetb3_fold{fold}.pth"
        
        # Epoch Loop
        for epoch in range(EPOCHS):
            t_loss = train_epoch(model, train_loader, optimizer, criterion)
            v_loss = validate(model, val_loader, criterion)
            scheduler.step()
            
            if v_loss < best_loss:
                best_loss = v_loss
                torch.save(model.state_dict(), save_path)
            
            print(f"Ep {epoch+1:02d} | Train: {t_loss:.4f} | Val: {v_loss:.4f} | Best: {best_loss:.4f}")
            
        # 3. OOF Inference (Load Best & Run TTA)
        print("Running Validation TTA...")
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
        
        fold_probs = run_tta(model, val_loader, DEVICE)
        
        # Store in global OOF arrays
        for i, idx in enumerate(val_idx):
            oof_preds[idx] = fold_probs[i]
            oof_targets[idx] = targets[idx]
            
    # 4. Final OOF Report
    print("\n" + "="*30)
    print("INTERNAL OOF REPORT")
    print("="*30)
    final_preds = np.argmax(oof_preds, axis=1)
    print(classification_report(oof_targets, final_preds, target_names=LABELS))
