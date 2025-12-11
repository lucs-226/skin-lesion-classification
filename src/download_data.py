import os
import shutil
from pathlib import Path
from src import config

def setup_data():
    dest_path = config.BASE_DATA_PATH
    kaggle_source = Path("/kaggle/input/skin-cancer-mnist-ham10000")
    
    if dest_path.exists() and any(dest_path.iterdir()):
        print(f"Data found in {dest_path}")
        return

    # Kaggle Environment
    if kaggle_source.exists():
        print("Kaggle detected. Copying internal data...")
        try:
            shutil.copytree(kaggle_source, dest_path, dirs_exist_ok=True)
            print(f"Data copied to {dest_path}")
            return
        except Exception as e:
            print(f"Copy error: {e}")
            return

    # Local Environment
    print("Local detected. Downloading via API...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        dataset_name = "kmader/skin-cancer-mnist-ham10000"
        
        if not dest_path.exists():
            dest_path.mkdir(parents=True)
            
        print(f"Downloading {dataset_name}...")
        api.dataset_download_files(dataset_name, path=dest_path, unzip=True)
        print(f"Download complete at {dest_path}")
        
    except Exception as e:
        print(f"API Error: {e}")
        print("Ensure 'kaggle.json' is in ~/.kaggle/")

if __name__ == "__main__":
    setup_data()
