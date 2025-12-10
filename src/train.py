import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from torch.cuda.amp import GradScaler, autocast

from src import config, data, model, utils

def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    avg_loss = 0
    
    for x, y in loader:
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        
        optimizer.zero_grad()
        with autocast():
            out = model(x)
            loss = criterion(out, y)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        avg_loss += loss.item()
        
    return avg_loss / len(loader)

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    avg_loss = 0
    for x, y in loader:
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        out = model(x)
        loss = criterion(out, y)
        avg_loss += loss.item()
    return avg_loss / len(loader)

def main():
    utils.set_seed(config.SEED)
    print(f"Device: {config.DEVICE} | Img Size: {config.IMG_SIZE}")

    # Data Prep
    df, _ = data.get_df(str(config.BASE_DATA_PATH))
    train_tf, val_tf = data.get_transforms(config.IMG_SIZE)
    targets = df['label_idx'].values
    
    skf = StratifiedKFold(n_splits=config.FOLDS, shuffle=True, random_state=config.SEED)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        print(f"\n>>> Starting Fold {fold+1}/{config.FOLDS}")
        
        # Datasets & Loaders
        train_ds = Subset(data.SkinDataset(df, train_tf), train_idx)
        val_ds = Subset(data.SkinDataset(df, val_tf), val_idx)
        
        sampler = data.get_weighted_sampler(targets[train_idx])
        
        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, sampler=sampler, num_workers=config.NUM_WORKERS)
        val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
        
        # Model & Optimization
        net = model.build_model(config.NUM_CLASSES).to(config.DEVICE)
        optimizer = optim.AdamW(net.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
        criterion = utils.FocalLoss(gamma=2.0)
        scaler = GradScaler()
        
        # Training Loop
        best_loss = float('inf')
        
        for epoch in range(config.EPOCHS):
            t_loss = train_one_epoch(net, train_loader, optimizer, criterion, scaler)
            v_loss = validate(net, val_loader, criterion)
            scheduler.step()
            
            if v_loss < best_loss:
                best_loss = v_loss
                save_path = config.OUTPUT_PATH / f"effnetb3_fold{fold}.pth"
                torch.save(net.state_dict(), save_path)
            
            # Minimalist Log
            print(f"Ep {epoch+1:02d} | T_loss: {t_loss:.4f} | V_loss: {v_loss:.4f} | Best: {best_loss:.4f}")

if __name__ == "__main__":
    main()
