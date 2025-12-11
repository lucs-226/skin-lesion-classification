import pandas as pd
from src.config import MAIN_DATA_ID, DATA_DIR, TRAIN_CSV
from src.utils import download_and_extract
from src.visualization import plot_class_distribution

if __name__ == "__main__":
    # 1. Get Data
    download_and_extract(MAIN_DATA_ID, DATA_DIR)
    
    # 2. Load Metadata
    df = pd.read_csv(TRAIN_CSV)
    print("Dataset Loaded. Shape:", df.shape)
    
    # 3. Generate Plots
    print("Generating EDA plots...")
    plot_class_distribution(df, target_col='dx') # Assuming 'dx' is the string label col
    print("EDA Complete. Check output/eda_plots/")
