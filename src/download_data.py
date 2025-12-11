import os
import shutil
from pathlib import Path
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    KaggleApi = None

from src import config

def setup_data():
    """
    Handles data setup for both Kaggle (internal copy) and Local (API download) environments.
    """
    dest_path = config.BASE_DATA_PATH
    kaggle_source = Path("/kaggle/input/skin-cancer-mnist-ham10000")
    
    # Check if data exists
    if dest_path.exists() and any(dest_path.iterdir()):
        print(f"Dataset found in {dest_path}. Skipping.")
        return

    # Kaggle Environment: Copy internal data
    if kaggle_source.exists():
        print("Kaggle detected. Copying internal data...")
        try:
            shutil.copytree(kaggle_source, dest_path, dirs_exist_ok=True)
            print(f"Data copied to {dest_path}")
        except Exception as e:
            print(f"Copy error: {e}")

    # Local Environment: Download via API
    else:
        print("Local detected. Downloading via API...")
        
        if KaggleApi is None:
            raise ImportError("Kaggle lib missing. Run 'pip install kaggle'.")
            
        try:
            api = KaggleApi()
            api.authenticate()
            
            dataset_name = "kmader/skin-cancer-mnist-ham10000"
            
            if not dest_path.exists():
                dest_path.mkdir(parents=True)
                
            api.dataset_download_files(dataset_name, path=dest_path, unzip=True)
            print(f"Download complete at {dest_path}")
            
        except Exception as e:
            print(f"API Error: {e}")
            print("Ensure 'kaggle.json' is in ~/.kaggle/")

if __name__ == "__main__":
    setup_data()
