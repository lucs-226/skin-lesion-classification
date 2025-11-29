#!/bin/bash

# Download HAM10000 dataset from Kaggle
# Requires Kaggle API credentials

echo "Downloading HAM10000 dataset..."

# Check if kaggle is installed
if ! command -v kaggle &> /dev/null
then
    echo "Kaggle CLI not found. Installing..."
    pip install kaggle
fi

# Download dataset
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data/

# Unzip
cd data/
unzip -q skin-cancer-mnist-ham10000.zip
rm skin-cancer-mnist-ham10000.zip

echo "✓ Dataset downloaded successfully!"
