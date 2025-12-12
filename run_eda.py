import os
import glob
import pandas as pd
from pathlib import Path
from src.config import MAIN_DATA_ID, DATA_DIR, TRAIN_CSV_PATH, TRAIN_IMG_DIRS, CLASSES_MAP
from src.utils import download_and_extract
from src.visualization import analyze_class_imbalance, show_class_samples

if __name__ == "__main__":
    print("--- Starting EDA ---")
    
    # 1. Download Data
    download_and_extract(MAIN_DATA_ID, DATA_DIR)
    
    # 2. Prepare Dataframe
    if not TRAIN_CSV_PATH.exists():
        raise FileNotFoundError(f"Metadata not found at {TRAIN_CSV_PATH}")
        
    df = pd.read_csv(TRAIN_CSV_PATH)
    
    # Create image map (filename -> full path)
    image_map = {}
    for d in TRAIN_IMG_DIRS:
        if d.exists():
            for x in glob.glob(os.path.join(d, "*.jpg")):
                image_map[os.path.splitext(os.path.basename(x))[0]] = x
            
    df['path'] = df['image_id'].map(image_map)
    
    # Filter missing images
    df = df.dropna(subset=['path'])
    
    # Deduplicate based on lesion_id (as per notebook)
    df_unique = df.drop_duplicates(subset=['lesion_id'], keep='first').reset_index(drop=True)
    df_unique['label_idx'] = df_unique['dx'].map(CLASSES_MAP)
    
    print(f"Original samples: {len(df)}")
    print(f"Cleaned samples (deduplicated): {len(df_unique)}")
    
    # 3. Visualize
    analyze_class_imbalance(df_unique)
    show_class_samples(df_unique)
