import torch.nn as nn
from torchvision import models

def build_model(num_classes: int, pretrained: bool = True):
    """Initializes EfficientNet-B3 by modifying the classification head."""
    model = models.efficientnet_b3(weights='DEFAULT' if pretrained else None)
    
    # Replace final classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model
