import torch
import torch.nn as nn
from torchvision import models
from src.config import DEVICE, NUM_CLASSES

def build_model(num_classes=NUM_CLASSES):
    """
    Exact architecture reconstruction: EfficientNet-B3.
    """
    # weights=None because we load custom weights later.
    model = models.efficientnet_b3(weights=None)
    
    in_features = model.classifier[1].in_features
    
    # Exact classifier head from notebook
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )
    
    return model.to(DEVICE)
