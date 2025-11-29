"""
Image preprocessing functions for skin lesion images

This module provides various preprocessing techniques including resizing,
normalization, contrast enhancement, and CLAHE
"""

import numpy as np
from PIL import Image, ImageEnhance
import cv2
from typing import Tuple, Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ImageNet normalization constants (for transfer learning)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def preprocess_image(
    img_path: str,
    target_size: Tuple[int, int] = (224, 224),
    normalize: Optional[str] = 'imagenet',
    enhance_contrast: bool = True,
    contrast_factor: float = 1.2,
    use_clahe: bool = False,
    clahe_clip_limit: float = 2.0,
    clahe_tile_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Preprocess a single image with multiple options
    
    Args:
        img_path: Path to image file
        target_size: Target dimensions (width, height)
        normalize: 'imagenet' for ImageNet stats, 'simple' for [0,1], None for no normalization
        enhance_contrast: Apply contrast enhancement
        contrast_factor: Contrast enhancement factor (1.0 = no change)
        use_clahe: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe_clip_limit: CLAHE clip limit
        clahe_tile_size: CLAHE tile grid size
    
    Returns:
        Preprocessed image as numpy array
    """
    # Load image and ensure RGB
    img = Image.open(img_path).convert('RGB')
    
    # Resize with high-quality resampling
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # Optional contrast enhancement
    if enhance_contrast and not use_clahe:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Optional CLAHE (advanced contrast enhancement)
    if use_clahe:
        img_array = apply_clahe(
            img_array,
            clip_limit=clahe_clip_limit,
            tile_size=clahe_tile_size
        )
    
    # Normalize to [0, 1]
    img_array = img_array.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization if requested
    if normalize == 'imagenet':
        img_array = normalize_imagenet(img_array)
    elif normalize == 'simple':
        pass  # Already in [0, 1]
    elif normalize is None:
        img_array = img_array * 255.0  # Back to [0, 255]
    
    return img_array


def apply_clahe(
    img_array: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    
    CLAHE is applied to the L channel of LAB color space to enhance
    contrast while preserving color information
    
    Args:
        img_array: Input image array (RGB, uint8 or float)
        clip_limit: Threshold for contrast limiting
        tile_size: Size of grid for histogram equalization
    
    Returns:
        Image with enhanced contrast
    """
    # Ensure uint8 format
    if img_array.dtype == np.float32 or img_array.dtype == np.float64:
        img_array = (img_array * 255).astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to RGB
    img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return img_array


def normalize_imagenet(img_array: np.ndarray) -> np.ndarray:
    """
    Normalize image using ImageNet statistics
    
    Args:
        img_array: Image array in range [0, 1]
    
    Returns:
        Normalized image array
    """
    return (img_array - IMAGENET_MEAN) / IMAGENET_STD


def denormalize_imagenet(img_array: np.ndarray) -> np.ndarray:
    """
    Denormalize image from ImageNet statistics back to [0, 1]
    
    Args:
        img_array: Normalized image array
    
    Returns:
        Denormalized image array in [0, 1]
    """
    img_array = (img_array * IMAGENET_STD + IMAGENET_MEAN)
    return np.clip(img_array, 0, 1)


def resize_with_padding(
    img: Union[Image.Image, np.ndarray],
    target_size: Tuple[int, int],
    fill_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio using padding
    
    Args:
        img: Input image (PIL Image or numpy array)
        target_size: Target dimensions (width, height)
        fill_color: Color for padding (RGB)
    
    Returns:
        Resized and padded image as numpy array
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    # Calculate aspect ratios
    img_ratio = img.width / img.height
    target_ratio = target_size[0] / target_size[1]
    
    if img_ratio > target_ratio:
        # Image is wider - fit to width
        new_width = target_size[0]
        new_height = int(new_width / img_ratio)
    else:
        # Image is taller - fit to height
        new_height = target_size[1]
        new_width = int(new_height * img_ratio)
    
    # Resize image
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create padded image
    padded_img = Image.new('RGB', target_size, fill_color)
    
    # Calculate paste position (center)
    paste_x = (target_size[0] - new_width) // 2
    paste_y = (target_size[1] - new_height) // 2
    
    # Paste resized image onto padded canvas
    padded_img.paste(img, (paste_x, paste_y))
    
    return np.array(padded_img)


def preprocess_batch(
    img_paths: list,
    target_size: Tuple[int, int] = (224, 224),
    normalize: Optional[str] = 'imagenet',
    **kwargs
) -> np.ndarray:
    """
    Preprocess a batch of images
    
    Args:
        img_paths: List of image file paths
        target_size: Target dimensions
        normalize: Normalization method
        **kwargs: Additional arguments for preprocess_image
    
    Returns:
        Batch of preprocessed images as numpy array (N, H, W, C)
    """
    batch = []
    
    for img_path in img_paths:
        try:
            img = preprocess_image(
                img_path,
                target_size=target_size,
                normalize=normalize,
                **kwargs
            )
            batch.append(img)
        except Exception as e:
            logger.error(f"Error preprocessing {img_path}: {e}")
            # Add a blank image to maintain batch size
            if normalize == 'imagenet':
                blank = -IMAGENET_MEAN / IMAGENET_STD
            else:
                blank = np.zeros((*target_size, 3))
            batch.append(blank)
    
    return np.array(batch)


def get_image_statistics(
    img_paths: list,
    sample_size: Optional[int] = None
) -> dict:
    """
    Calculate statistics from a set of images
    
    Args:
        img_paths: List of image paths
        sample_size: Number of images to sample (None = use all)
    
    Returns:
        Dictionary with image statistics
    """
    import random
    
    if sample_size and sample_size < len(img_paths):
        img_paths = random.sample(img_paths, sample_size)
    
    widths = []
    heights = []
    aspect_ratios = []
    mean_colors = []
    
    logger.info(f"Calculating statistics for {len(img_paths)} images...")
    
    for img_path in img_paths:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h)
                
                # Calculate mean color
                img_array = np.array(img.convert('RGB'))
                mean_colors.append(img_array.mean(axis=(0, 1)))
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
    
    stats = {
        'width': {
            'min': min(widths),
            'max': max(widths),
            'mean': np.mean(widths),
            'std': np.std(widths)
        },
        'height': {
            'min': min(heights),
            'max': max(heights),
            'mean': np.mean(heights),
            'std': np.std(heights)
        },
        'aspect_ratio': {
            'min': min(aspect_ratios),
            'max': max(aspect_ratios),
            'mean': np.mean(aspect_ratios),
            'std': np.std(aspect_ratios)
        },
        'mean_color': {
            'r': np.mean([c[0] for c in mean_colors]),
            'g': np.mean([c[1] for c in mean_colors]),
            'b': np.mean([c[2] for c in mean_colors])
        }
    }
    
    logger.info(f"Image statistics:")
    logger.info(f"  Width: {stats['width']['mean']:.1f} ± {stats['width']['std']:.1f} (range: {stats['width']['min']}-{stats['width']['max']})")
    logger.info(f"  Height: {stats['height']['mean']:.1f} ± {stats['height']['std']:.1f} (range: {stats['height']['min']}-{stats['height']['max']})")
    logger.info(f"  Aspect ratio: {stats['aspect_ratio']['mean']:.2f} ± {stats['aspect_ratio']['std']:.2f}")
    
    return stats


if __name__ == "__main__":
    # Example usage
    import os
    
    # Test with a sample image
    sample_image = "data/HAM10000_images_part_1/ISIC_0024306.jpg"
    
    if os.path.exists(sample_image):
        # Test different preprocessing options
        img_basic = preprocess_image(sample_image, normalize='simple', enhance_contrast=False)
        img_contrast = preprocess_image(sample_image, normalize='simple', enhance_contrast=True)
        img_clahe = preprocess_image(sample_image, normalize='simple', use_clahe=True)
        img_imagenet = preprocess_image(sample_image, normalize='imagenet')
        
        print("✓ Preprocessing test successful!")
        print(f"  Basic: shape={img_basic.shape}, range=[{img_basic.min():.2f}, {img_basic.max():.2f}]")
        print(f"  Contrast: shape={img_contrast.shape}, range=[{img_contrast.min():.2f}, {img_contrast.max():.2f}]")
        print(f"  CLAHE: shape={img_clahe.shape}, range=[{img_clahe.min():.2f}, {img_clahe.max():.2f}]")
        print(f"  ImageNet: shape={img_imagenet.shape}, range=[{img_imagenet.min():.2f}, {img_imagenet.max():.2f}]")
    else:
        print(f"Sample image not found: {sample_image}")
