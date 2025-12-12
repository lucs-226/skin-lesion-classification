import torch
import torch.nn as nn
from torchvision import models
from src.config import DEVICE, NUM_CLASSES

def build_model(num_classes=NUM_CLASSES, pretrained=True):
    weights = 'IMAGENET1K_V1' if pretrained else None
    model = models.efficientnet_b3(weights=weights)
    
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )
    return model.to(DEVICE)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        return loss.mean()
