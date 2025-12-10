import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from . import config

def get_df(base_path: str):
    """Loads, cleans, and prepares the dataframe, removing duplicates."""
    img_dirs = [
        os.path.join(base_path, "HAM10000_images_part_1"),
        os.path.join(base_path, "HAM10000_images_part_2")
    ]
    
    df = pd.read_csv(os.path.join(base_path, "HAM10000_metadata.csv"))
    
    # Image path mapping
    image_map = {
        os.path.splitext(os.path.basename(x))[0]: x 
        for d in img_dirs 
        for x in glob.glob(os.path.join(d, "*.jpg"))
    }
    df['path'] = df['image_id'].map(image_map)
    
    # Remove duplicates (Data Leakage prevention)
    df_unique = df.drop_duplicates(subset=['lesion_id'], keep='first').reset_index(drop=True)
    
    # Target encoding
    classes = sorted(df_unique['dx'].unique())
    class_map = {label: idx for idx, label in enumerate(classes)}
    df_unique['label_idx'] = df_unique['dx'].map(class_map)
    
    return df_unique, class_map

class SkinDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        label = torch.tensor(row['label_idx'], dtype=torch.long)
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

def get_transforms(img_size):
    """Defines augmentation pipelines."""
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_tf, val_tf

def get_weighted_sampler(targets):
    """Creates a sampler to balance classes during training."""
    class_counts = np.bincount(targets)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[t] for t in targets]
    return WeightedRandomSampler(sample_weights, len(sample_weights))
