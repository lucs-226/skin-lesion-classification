import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from src.config import IMG_SIZE

# Exact Preprocessing from notebook
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ExternalTestDataset(Dataset):
    """
    Dataset class for the external test set (DermX).
    """
    def __init__(self, df, class_map, transform=None):
        self.df = df
        self.class_map = class_map
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Adjust path joining based on your actual csv structure
        img_path = self.df.iloc[idx]['image_path']
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label_str = self.df.iloc[idx]['label']
        label = self.class_map[label_str]
        
        return image, torch.tensor(label, dtype=torch.long)
