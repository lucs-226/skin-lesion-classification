import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from src.config import IMG_SIZE

def get_transforms(mode='valid'):
    """
    Returns the exact transformations used in the notebook.
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation/Inference transforms
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class SkinDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Adjust 'image_path' based on your actual dataframe structure
        img_path = self.df.iloc[idx]['image_path']
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Assumes 'label_idx' column exists for training
        label = torch.tensor(self.df.iloc[idx]['label_idx'], dtype=torch.long)
        return image, label
