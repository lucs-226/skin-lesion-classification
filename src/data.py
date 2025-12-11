import torch
import numpy as np
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from src.config import IMG_SIZE

def get_transforms(mode='valid'):
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_sampler(labels):
    class_counts = np.bincount(labels)
    weights = 1. / class_counts
    samples_weights = [weights[t] for t in labels]
    return WeightedRandomSampler(samples_weights, len(samples_weights))

class SkinDataset(Dataset):
    def __init__(self, df, transform=None, img_col='image_path', label_col='label_idx'):
        self.df = df
        self.transform = transform
        self.img_col = img_col
        self.label_col = label_col
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        path = self.df.iloc[idx][self.img_col]
        image = Image.open(path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        if self.label_col in self.df.columns:
            return image, torch.tensor(self.df.iloc[idx][self.label_col], dtype=torch.long)
        return image
