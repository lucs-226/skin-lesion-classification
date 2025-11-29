"""
Data loading utilities for HAM10000 dataset

This module provides functions to load, validate, and prepare the HAM10000 dataset
"""

import pandas as pd
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_image_map(image_dirs: List[str]) -> Dict[str, str]:
    """
    Create a mapping from image filenames to their full paths
    
    Args:
        image_dirs: List of directories containing images
    
    Returns:
        Dictionary mapping filename to full path
    """
    image_map = {}
    
    for directory in image_dirs:
        if not os.path.exists(directory):
            logger.warning(f"Directory not found: {directory}")
            continue
            
        for filename in os.listdir(directory):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_map[filename] = os.path.join(directory, filename)
    
    logger.info(f"Found {len(image_map)} images across {len(image_dirs)} directories")
    return image_map


def load_ham10000_data(
    metadata_path: str,
    image_dirs: List[str],
    validate: bool = True
) -> pd.DataFrame:
    """
    Load HAM10000 dataset with metadata and image paths
    
    Args:
        metadata_path: Path to CSV file containing metadata
        image_dirs: List of directories containing images
        validate: Whether to validate data integrity
    
    Returns:
        DataFrame with image metadata and file paths
    """
    logger.info("Loading HAM10000 metadata...")
    
    # Load metadata CSV
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    df = pd.read_csv(metadata_path)
    logger.info(f"Loaded {len(df)} metadata entries")
    
    # Create image mapping
    image_map = create_image_map(image_dirs)
    
    # Add file paths to dataframe
    df['path'] = df['image_id'].apply(lambda x: image_map.get(f"{x}.jpg", None))
    
    if validate:
        df = validate_data(df)
    
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate dataset integrity and handle issues
    
    Args:
        df: DataFrame with image metadata
    
    Returns:
        Validated DataFrame
    """
    logger.info("Validating dataset...")
    
    # Check for missing image files
    missing_images = df[df['path'].isna()]
    if len(missing_images) > 0:
        logger.warning(f"Found {len(missing_images)} images without file paths")
        logger.info("Removing entries with missing images...")
        df = df.dropna(subset=['path'])
    
    # Check for duplicate image_ids
    duplicates = df[df.duplicated(subset=['image_id'], keep=False)]
    if len(duplicates) > 0:
        logger.warning(f"Found {len(duplicates)} duplicate image_ids")
        logger.info("Keeping first occurrence of duplicates...")
        df = df.drop_duplicates(subset=['image_id'], keep='first')
    
    # Check for duplicate lesion_ids (expected - same lesion, different photos)
    duplicate_lesions = df[df.duplicated(subset=['lesion_id'], keep=False)]
    if len(duplicate_lesions) > 0:
        logger.info(f"Note: {len(duplicate_lesions)} images share lesion_ids (same lesion, multiple photos)")
    
    # Verify all paths exist
    missing_files = []
    for idx, row in df.iterrows():
        if not os.path.exists(row['path']):
            missing_files.append(row['image_id'])
    
    if missing_files:
        logger.warning(f"Found {len(missing_files)} files that don't exist on disk")
        df = df[df['path'].apply(os.path.exists)]
    
    # Check for required columns
    required_columns = ['image_id', 'lesion_id', 'dx', 'path']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    logger.info(f"Validation complete. Final dataset size: {len(df)} images")
    
    return df.reset_index(drop=True)


def get_class_distribution(df: pd.DataFrame, class_column: str = 'dx') -> pd.Series:
    """
    Get the distribution of classes in the dataset
    
    Args:
        df: DataFrame with class labels
        class_column: Name of column containing class labels
    
    Returns:
        Series with class counts
    """
    distribution = df[class_column].value_counts()
    logger.info(f"\nClass distribution:")
    for cls, count in distribution.items():
        percentage = (count / len(df)) * 100
        logger.info(f"  {cls:6s}: {count:5d} ({percentage:5.2f}%)")
    
    return distribution


def get_metadata_stats(df: pd.DataFrame) -> Dict:
    """
    Calculate statistics from metadata
    
    Args:
        df: DataFrame with metadata
    
    Returns:
        Dictionary with metadata statistics
    """
    stats = {
        'total_images': len(df),
        'unique_lesions': df['lesion_id'].nunique(),
        'age': {
            'mean': df['age'].mean(),
            'median': df['age'].median(),
            'min': df['age'].min(),
            'max': df['age'].max(),
            'missing': df['age'].isna().sum()
        },
        'sex': df['sex'].value_counts().to_dict(),
        'localization': df['localization'].value_counts().to_dict(),
        'dx_type': df['dx_type'].value_counts().to_dict() if 'dx_type' in df.columns else {}
    }
    
    logger.info(f"\nDataset statistics:")
    logger.info(f"  Total images: {stats['total_images']}")
    logger.info(f"  Unique lesions: {stats['unique_lesions']}")
    logger.info(f"  Age: mean={stats['age']['mean']:.1f}, range=[{stats['age']['min']:.0f}, {stats['age']['max']:.0f}]")
    
    return stats


def check_image_integrity(df: pd.DataFrame, sample_size: int = 100) -> Tuple[List, List]:
    """
    Check if images can be loaded properly
    
    Args:
        df: DataFrame with image paths
        sample_size: Number of images to check
    
    Returns:
        Tuple of (corrupted_images, valid_images)
    """
    from PIL import Image
    
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    corrupted = []
    valid = []
    
    logger.info(f"Checking integrity of {len(sample_df)} sample images...")
    
    for idx, row in sample_df.iterrows():
        try:
            with Image.open(row['path']) as img:
                img.verify()  # Verify image integrity
                valid.append(row['image_id'])
        except Exception as e:
            logger.error(f"Corrupted image: {row['image_id']} - {e}")
            corrupted.append(row['image_id'])
    
    if corrupted:
        logger.warning(f"Found {len(corrupted)} corrupted images out of {len(sample_df)} checked")
    else:
        logger.info(f"All {len(valid)} checked images are valid")
    
    return corrupted, valid


if __name__ == "__main__":
    # Example usage
    base_path = "data/"
    metadata_path = os.path.join(base_path, "HAM10000_metadata.csv")
    image_dirs = [
        os.path.join(base_path, "HAM10000_images_part_1"),
        os.path.join(base_path, "HAM10000_images_part_2")
    ]
    
    # Load data
    df = load_ham10000_data(metadata_path, image_dirs)
    
    # Get statistics
    distribution = get_class_distribution(df)
    stats = get_metadata_stats(df)
    
    # Check image integrity
    corrupted, valid = check_image_integrity(df, sample_size=100)
    
    print(f"\n✓ Data loading complete!")
    print(f"  Total valid images: {len(df)}")
    print(f"  Classes: {len(distribution)}")
    print(f"  Unique lesions: {stats['unique_lesions']}")
