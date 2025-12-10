import os
from kaggle.api.kaggle_api_extended import KaggleApi
from src import config

def download_and_extract():
    """
    Downloads the HAM10000 dataset from Kaggle and extracts it into data/.
    Requires the kaggle.json file in ~/.kaggle/
    """
    dataset_name = "kmader/skin-cancer-mnist-ham10000"
    download_path = config.BASE_DATA_PATH
    
    print(f"Downloading dataset {dataset_name}...")
    
    api = KaggleApi()
    api.authenticate()
    
    if not download_path.exists():
        download_path.mkdir(parents=True)

    # Automatically downloads and unzips
    api.dataset_download_files(dataset_name, path=download_path, unzip=True)
    
    print("Download complete. Data is in", download_path)

if __name__ == "__main__":
    download_and_extract()
