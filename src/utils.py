import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import matplotlib.pyplot as plt

def set_seed(seed: int = 42):
    """Ensures experiment reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class FocalLoss(nn.Module):
    """Handles class imbalance by down-weighting easy examples."""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean': return focal_loss.mean()
        return focal_loss.sum()

def plot_smoothed_loss(history, smooth=0.85):
    """
    Plots training and validation loss with exponential smoothing.
    Input: history = {'train_loss': [...], 'val_loss': [...]}
    """
    def _smooth(scalars, weight):
        last = scalars[0]
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(10, 6))
    # Raw data (transparent)
    plt.plot(epochs, history['train_loss'], alpha=0.3, color='blue', label='Train Raw')
    plt.plot(epochs, history['val_loss'], alpha=0.3, color='orange', label='Val Raw')
    
    # Smoothed data (solid)
    plt.plot(epochs, _smooth(history['train_loss'], smooth), color='blue', lw=2, label='Train Smooth')
    plt.plot(epochs, _smooth(history['val_loss'], smooth), color='orange', lw=2, label='Val Smooth')
    
    plt.title("Training Dynamics")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
