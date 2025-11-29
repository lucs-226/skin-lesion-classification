"""
Data augmentation functions for training
"""

import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import random
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ImageNet stats for denormalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def augment_image(
    img_array: np.ndarray,
    augmentation_type: str = 'random',
    is_normalized: bool = False
) -> List[np.ndarray]:
    """
    Apply data augmentation to an image
    
    Args:
        img_array: Image array (H, W, C)
        augmentation_type: Type of augmentation ('flip', 'rotate', 'random', 'all')
        is_normalized: Whether image is ImageNet normalized
    
    Returns:
        List of augmented images
    """
    # Denormalize if needed
    if is_normalized:
        img_array = (img_array * IMAGENET_STD + IMAGENET_MEAN).clip(0, 1)
    
    # Ensure [0, 1] range
    if img_array.max() <= 1.0:
        img_array = img_array
    else:
        img_array = img_array / 255.0
    
    # Convert to PIL
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    augmented = []
    
    if augmentation_type in ['flip', 'all', 'random']:
        # Horizontal flip
        img_flipped = ImageOps.mirror(img)
        augmented.append(np.array(img_flipped) / 255.0)
    
    if augmentation_type in ['rotate', 'all']:
        # Fixed rotations
        for angle in [90, 180, 270]:
            img_rotated = img.rotate(angle, expand=False)
            augmented.append(np.array(img_rotated) / 255.0)
    
    if augmentation_type == 'random':
        # Random rotation (-20 to +20 degrees)
        angle = random.uniform(-20, 20)
        img_rotated = img.rotate(angle, expand=False, fillcolor=(0, 0, 0))
        augmented.append(np.array(img_rotated) / 255.0)
        
        # Random brightness
        enhancer = ImageEnhance.Brightness(img)
        factor = random.uniform(0.8, 1.2)
        img_bright = enhancer.enhance(factor)
        augmented.append(np.array(img_bright) / 255.0)
    
    # Re-normalize if needed
    if is_normalized:
        augmented = [(aug - IMAGENET_MEAN) / IMAGENET_STD for aug in augmented]
    
    return augmented


if __name__ == "__main__":
    print("✓ Augmentation module loaded successfully")
