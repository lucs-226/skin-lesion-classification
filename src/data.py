import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from src.config import IMG_SIZE, SEED

# --- Transforms ---
stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=45, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(*stats),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.15)) 
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

# --- Datasets ---
class SkinDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, row['label_idx']

class ExternalTestDataset(Dataset):
    def __init__(self, df, classes_map, transform=None):
        self.df = df.reset_index(drop=True)
        self.classes_map = classes_map
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['filepath']).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label_idx = self.classes_map[row['label']]
        return img, label_idx

# --- Helpers ---
def get_sampler(labels):
    class_weights = 1. / np.bincount(labels)
    sample_weights = [class_weights[t] for t in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights))

def create_filtered_test_set(data_dir, train_csv_path, target_classes, test_size=2000):
    """
    Scans folders for external test data and removes images 
    present in the training blacklist to avoid leakage.
    """
    # 1. Load Blacklist (Training IDs)
    if os.path.exists(train_csv_path):
        df_train = pd.read_csv(train_csv_path)
        # Assuming image_id is the filename without extension
        blacklist_ids = set(df_train['image_id'].astype(str).apply(lambda x: Path(x).stem))
    else:
        print("Warning: Training CSV not found. Skipping leakage check.")
        blacklist_ids = set()
    
    valid_data = []
    
    # 2. Scan folders
    for class_name in target_classes:
        class_path = Path(data_dir) / class_name
        if not class_path.exists():
            continue
            
        for f in class_path.glob("*"):
            if f.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                img_id = f.stem
                if img_id not in blacklist_ids:
                    valid_data.append({
                        'filepath': str(f),
                        'label': class_name,
                        'image_id': img_id
                    })
    
    df_clean = pd.DataFrame(valid_data)
    
    if len(df_clean) == 0:
        raise ValueError(f"No valid images found in {data_dir}")

    # 3. Stratified Sampling
    # If we have enough data, we sample. Otherwise use all.
    if len(df_clean) > test_size:
        _, df_test = train_test_split(
            df_clean,
            test_size=test_size,
            stratify=df_clean['label'],
            random_state=SEED
        )
    else:
        df_test = df_clean.copy()
        
    return df_test
